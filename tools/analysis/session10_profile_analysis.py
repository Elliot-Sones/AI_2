#!/usr/bin/env python3
"""Compare diagnostic PPO traces using CPU nesting and CUDA launch correlation.

This reads exported traces only. Warmup is selected by outer rollout/update
ordinal, independently for each kind; GPU timestamps never determine a cycle.
No throughput or speedup claims are derived from instrumented traces.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

OUTER_REGIONS = ("collect_rollout", "ppo_update")
DETAIL_REGIONS = ("optimizer_step", "rollout_buffer_add", "critic_only_forward",
                  "gae", "action_forward", "native_environment_step")
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
CUDA_API_CATEGORIES = {"cuda_runtime", "cuda_driver"}
NOTES = [
    "Diagnostic counts only; profiler durations are not headline throughput or speedup evidence.",
    "Optimizer and other child regions are included in outer region totals; do not sum them together.",
    "CPU attribution requires the same process/thread and complete interval containment; ambiguous outer overlap remains unscoped.",
    "CUDA attribution uses runtime launch correlation, falling back to a CPU operator External id; GPU timestamp overlap alone is never attribution.",
    "GPU duration sums may overlap across streams; GPU active union is reported separately and includes only attributed kernel/memcpy/memset spans.",
    "GPU user annotations are excluded from CPU cycle detection and GPU kernel/active-time counts.",
    "Host region inclusive time and GPU attributed work have different timing semantics; GPU work can finish after its CPU region.",
    "CPU aten::_local_scalar_dense can remain after Adam counters move to CPU. CUDA runtime calls and memcpy events nested under scalar extraction distinguish actual readbacks.",
    "boolean_validation means an aten::is_nonzero ancestor; it can include distribution or other boolean checks. Other nonoptimizer scalars are not proven to be metrics alone.",
    "Unscoped work includes startup/closeout and unmatched GPU events; missing correlation metadata is not zero GPU work.",
]


class _Intervals:
    """Find containing anchors without scanning every earlier trace event."""
    def __init__(self, anchors):
        grouped = defaultdict(list)
        for identifier, event in anchors:
            grouped[(event.get("pid"), event.get("tid"))].append(
                (float(event["ts"]), float(event["ts"] + event["dur"]), identifier)
            )
        self.groups = {}
        for thread, spans in grouped.items():
            spans.sort()
            maximum = -math.inf
            prefix = []
            for _start, end, _identifier in spans:
                maximum = max(maximum, end)
                prefix.append(maximum)
            self.groups[thread] = (spans, [span[0] for span in spans], prefix)

    def containing(self, event):
        group = self.groups.get((event.get("pid"), event.get("tid")))
        if group is None:
            return []
        spans, starts, prefix = group
        start, end = float(event["ts"]), float(event["ts"] + event["dur"])
        position = bisect_right(starts, start) - 1
        result = []
        while position >= 0 and prefix[position] >= end:
            if spans[position][1] >= end:
                result.append(spans[position][2])
            position -= 1
        return result


def _detail_name(event):
    name = str(event.get("name", ""))
    if name.startswith("Optimizer.step#"):
        return "optimizer_step"
    return name if name in DETAIL_REGIONS else None


def _empty_stats():
    return {"calls": 0, "host_region_inclusive_us": 0.0,
            "cpu_op_counts": {}, "cpu_annotation_counts": {},
            "cuda_runtime_counts": {}, "cuda_runtime_duration_us_sum": {},
            "cuda_driver_counts": {}, "cuda_driver_duration_us_sum": {},
            "scalar_calls_by_context": {}, "scalar_cuda_runtime_counts": {},
            "scalar_gpu_event_counts": {}, "gpu_event_counts": {},
            "gpu_kernel_names": {}, "gpu_duration_us_sum": {}, "_gpu_intervals": []}


def _increment(mapping, key, amount=1):
    mapping[key] = mapping.get(key, 0) + amount


def _union_duration(intervals):
    duration, end = 0.0, -math.inf
    for start, finish in sorted(intervals):
        duration += max(0.0, finish - max(start, end))
        end = max(end, finish)
    return duration


def _external_id(event):
    args = event.get("args") or {}
    return args.get("External id", args.get("external_id"))


def analyze_trace(trace: dict[str, Any], *, warmup_cycles: int = 1) -> dict[str, Any]:
    if type(warmup_cycles) is not int or warmup_cycles < 0:
        raise ValueError("warmup_cycles must be a nonnegative integer")
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise ValueError("trace must contain a traceEvents list")
    if trace.get("metadata", {}).get("capture_available") is False:
        raise ValueError("trace is a failed capture receipt")
    events = []
    invalid_complete_events = 0
    for event in trace["traceEvents"]:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        if any(type(event.get(key)) not in (float, int) or not math.isfinite(event[key])
               for key in ("ts", "dur")) or event["dur"] < 0:
            invalid_complete_events += 1
            continue
        events.append(event)
    cpu = [(index, event) for index, event in enumerate(events)
           if event.get("cat") not in GPU_CATEGORIES | {"gpu_user_annotation"}]
    outer = [(index, event) for index, event in cpu if event.get("name") in OUTER_REGIONS]
    details = [(index, event) for index, event in cpu if _detail_name(event)]
    scalar = [(index, event) for index, event in cpu if event.get("name") == "aten::_local_scalar_dense"]
    boolean = [(index, event) for index, event in cpu if event.get("name") == "aten::is_nonzero"]
    outer_index, detail_index = _Intervals(outer), _Intervals(details)
    scalar_index, boolean_index = _Intervals(scalar), _Intervals(boolean)
    outer_phase = {}
    cycles = {}
    for name in OUTER_REGIONS:
        selected = sorted(((index, event) for index, event in outer if event["name"] == name),
                          key=lambda pair: pair[1]["ts"])
        cycles[name] = {"total": len(selected), "warmup": min(len(selected), warmup_cycles),
                        "measured": max(0, len(selected) - warmup_cycles)}
        for ordinal, (index, _event) in enumerate(selected):
            outer_phase[index] = (name, "warmup" if ordinal < warmup_cycles else "measured")
    regions = {name: {phase: _empty_stats() for phase in ("warmup", "measured")}
               for name in (*OUTER_REGIONS, *DETAIL_REGIONS)}
    unscoped = _empty_stats()
    attribution = {"ambiguous_cpu_outer": 0, "gpu_by_correlation": 0,
                   "gpu_by_external_id": 0, "gpu_ambiguous": 0, "gpu_unmatched": 0,
                   "gpu_unscoped": 0, "invalid_complete_events": invalid_complete_events,
                   "gpu_annotation_events_excluded": sum(event.get("cat") == "gpu_user_annotation" for event in events)}
    correlation_contexts = defaultdict(set)
    external_contexts = defaultdict(set)

    def context_for(event):
        parents = outer_index.containing(event)
        if len(parents) > 1:
            attribution["ambiguous_cpu_outer"] += 1
        outer_context = outer_phase[parents[0]] if len(parents) == 1 else (None, None)
        detail_names = tuple(sorted({_detail_name(events[index]) for index in detail_index.containing(event)}))
        kind = None
        if scalar_index.containing(event):
            kind = ("optimizer" if "optimizer_step" in detail_names else
                    "boolean_validation" if boolean_index.containing(event) else "other")
        return (*outer_context, detail_names, kind)

    def buckets(context):
        name, phase, detail_names, _kind = context
        if name is None:
            return [(None, unscoped)]
        return [(name, regions[name][phase]),
                *((detail, regions[detail][phase]) for detail in detail_names)]

    for _index, event in cpu:
        context = context_for(event)
        kind = context[-1]
        cat, name = event.get("cat"), str(event.get("name", ""))
        for bucket_name, stats in buckets(context):
            if name == bucket_name or (bucket_name == "optimizer_step" and _detail_name(event) == bucket_name):
                stats["calls"] += 1
                stats["host_region_inclusive_us"] += float(event["dur"])
            if cat == "cpu_op":
                _increment(stats["cpu_op_counts"], name)
            elif cat == "user_annotation":
                _increment(stats["cpu_annotation_counts"], name)
            elif cat in CUDA_API_CATEGORIES:
                _increment(stats[f"{cat}_counts"], name)
                _increment(stats[f"{cat}_duration_us_sum"], name, float(event["dur"]))
                if kind is not None:
                    _increment(stats["scalar_cuda_runtime_counts"].setdefault(kind, {}), name)
            if name == "aten::_local_scalar_dense":
                _increment(stats["scalar_calls_by_context"], kind or "other")
        if cat in CUDA_API_CATEGORIES:
            correlation = (event.get("args") or {}).get("correlation")
            if correlation is not None:
                correlation_contexts[correlation].add(context)
        if cat == "cpu_op" and _external_id(event) is not None:
            external_contexts[_external_id(event)].add(context)

    for event in events:
        category = event.get("cat")
        if category not in GPU_CATEGORIES:
            continue
        correlation = (event.get("args") or {}).get("correlation")
        candidates = correlation_contexts.get(correlation, set()) if correlation is not None else set()
        method = "gpu_by_correlation"
        if not candidates:
            external = _external_id(event)
            candidates = external_contexts.get(external, set()) if external is not None else set()
            method = "gpu_by_external_id"
        if len(candidates) == 1:
            context = next(iter(candidates))
            attribution[method] += 1
        else:
            context = (None, None, (), None)
            attribution["gpu_ambiguous" if candidates else "gpu_unmatched"] += 1
        if context[0] is None:
            attribution["gpu_unscoped"] += 1
        for _bucket_name, stats in buckets(context):
            _increment(stats["gpu_event_counts"], category)
            _increment(stats["gpu_duration_us_sum"], category, float(event["dur"]))
            if category == "kernel":
                _increment(stats["gpu_kernel_names"], str(event.get("name", "")))
            if context[-1] is not None:
                _increment(stats["scalar_gpu_event_counts"].setdefault(context[-1], {}), category)
            stats["_gpu_intervals"].append((float(event["ts"]), float(event["ts"] + event["dur"])))
    for stats in [unscoped, *(stats for phases in regions.values() for stats in phases.values())]:
        stats["gpu_active_union_us"] = _union_duration(stats.pop("_gpu_intervals"))
    warnings = []
    if attribution["gpu_unmatched"] or attribution["gpu_ambiguous"]:
        warnings.append("Some GPU work could not be uniquely correlated to CPU launch context.")
    if attribution["ambiguous_cpu_outer"]:
        warnings.append("Overlapping outer CPU ranges made some event attribution ambiguous.")
    if invalid_complete_events:
        warnings.append("Malformed complete events were excluded; their count is reported.")
    if not cycles["ppo_update"]["measured"] or not cycles["collect_rollout"]["measured"]:
        warnings.append("Trace has no measured update or rollout after the requested warmup exclusion.")
    return {"warmup_cycles": warmup_cycles, "cycles": cycles, "regions": regions,
            "unscoped": unscoped, "attribution": attribution, "warnings": warnings}


def _numeric_delta(baseline, candidate):
    if isinstance(baseline, dict) or isinstance(candidate, dict):
        baseline, candidate = baseline or {}, candidate or {}
        return {key: _numeric_delta(baseline.get(key), candidate.get(key))
                for key in sorted(set(baseline) | set(candidate))}
    return (candidate or 0) - (baseline or 0)


def compare_analyses(baseline, candidate):
    warnings = []
    if baseline["cycles"] != candidate["cycles"]:
        warnings.append("Outer cycle counts differ; raw count differences are not workload-normalized.")
    if baseline["warmup_cycles"] != candidate["warmup_cycles"]:
        raise ValueError("baseline and candidate warmup exclusions must match")
    return {"schema": "ai2.native_ppo_profile_analysis.v1", "diagnostic_only": True,
            "headline_sps_eligible": False, "warmup_cycles": baseline["warmup_cycles"],
            "baseline": baseline, "candidate": candidate,
            "delta_candidate_minus_baseline": _numeric_delta(baseline["regions"], candidate["regions"]),
            "warnings": warnings, "interpretation": list(NOTES)}


def _load_trace(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open(encoding="utf-8") as handle:
        trace = json.load(handle)
    return trace, {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-cycles", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    analyses, inputs = {}, {}
    for name in ("baseline", "candidate"):
        trace, inputs[name] = _load_trace(getattr(args, name))
        analyses[name] = analyze_trace(trace, warmup_cycles=args.warmup_cycles)
        del trace
    report = compare_analyses(analyses["baseline"], analyses["candidate"])
    report["inputs"] = inputs
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"diagnostic_trace_comparison={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
