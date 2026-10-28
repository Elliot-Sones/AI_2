#!/usr/bin/env python3
"""Opt-in CUDA training diagnostic with named Torch profiler regions.

Uses benchmark_native_lane_training's existing campaign path, with one warmup
and three measured cycles by default. The trace covers the whole driver; the
aggregate attributes only named methods. Profiled SPS is diagnostic and must
not replace the unprofiled matched benchmark's headline throughput.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import source_path

from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from functools import wraps
import hashlib
from pathlib import Path
import threading
import time
from typing import Any, Callable

ROOT = PROJECT_ROOT
REGIONS = (
    "action_forward", "critic_only_forward", "native_environment_step",
    "rollout_buffer_add", "gae", "ppo_update", "collect_rollout",
)
INTERPRETATION = [
    "Diagnostic profiler overhead is included. Profiled campaign SPS is not headline throughput; use an unprofiled matched run.",
    "Named regions are nested: collect_rollout contains forwards, environment stepping, buffer adds and GAE. Do not sum inclusive region totals.",
    "Host wall inclusive seconds include nested named calls; exclusive seconds subtract only named child calls, not all Python or Torch children.",
    "Host times include Python work, kernel submission and any existing waits. No per-region CUDA synchronization is inserted.",
    "Primary CPU/self-CPU and CUDA/self-CUDA metrics come only from one matching CPU user-range aggregate; CUDA fields are its attributed kernel durations when supported, not elapsed GPU wall time.",
    "Same-name CUDA annotations are retained separately by device. Their durations are annotation spans, not kernel time; do not sum them with CPU-attributed CUDA durations.",
    "Missing or ambiguous CPU user-range aggregates produce null primary profiler metrics, not zero; host-clock measurements remain independently available.",
    "On legacy Torch without annotation metadata, a unique CPU aggregate for a known instrumented region is used with an explicit legacy warning; explicit non-annotations remain excluded.",
    "The whole-driver trace and diagnostic driver wall include startup and checkpointing. Work outside named methods is excluded from region attribution.",
    "collect_rollout is a composite including existing callbacks/logging; callback checkpointing inside collect is not isolated as environment or policy compute.",
    "critic_only_forward includes both timeout bootstrapping and the final rollout value prediction; action_forward already includes its policy value head.",
    "native_environment_step measures NativeLaneEnv.step_tensor. With CPU lanes, adapter CPU/GPU transfers outside that method remain in collect_rollout exclusive time.",
    "Named aggregates include warmup cycles as well as measured cycles. They are attribution diagnostics, not warmed-up benchmark measurements.",
]
_MISSING = object()


@contextmanager
def instrument_regions(*, policy_class: type, env_class: type, buffer_class: type, ppo_class: type,
                       record_function: Callable[..., Any], clock: Callable[[], float] = time.perf_counter):
    """Temporarily wrap existing methods, preserving arguments, results and errors."""
    stats = {name: {"calls": 0, "failed_calls": 0, "host_wall_inclusive_seconds": 0.0,
                    "host_wall_exclusive_seconds": 0.0} for name in REGIONS}
    frames = threading.local()
    targets = (
        (policy_class, "forward", "action_forward"),
        (policy_class, "predict_values", "critic_only_forward"),
        (env_class, "step_tensor", "native_environment_step"),
        (buffer_class, "add", "rollout_buffer_add"),
        (buffer_class, "compute_returns_and_advantage", "gae"),
        (ppo_class, "train", "ppo_update"),
        (ppo_class, "collect_rollouts", "collect_rollout"),
    )

    def restore(owner, method, own_value):
        if own_value is _MISSING:
            delattr(owner, method)
        else:
            setattr(owner, method, own_value)

    def wrapped_method(original, region):
        @wraps(original)
        def measured(*args, **kwargs):
            if not hasattr(frames, "stack"):
                frames.stack = []
            frame = {"child_seconds": 0.0}
            frames.stack.append(frame)
            started = clock()
            stat = stats[region]
            stat["calls"] += 1
            try:
                with record_function(region):
                    return original(*args, **kwargs)
            except BaseException:
                stat["failed_calls"] += 1
                raise
            finally:
                elapsed = clock() - started
                frames.stack.pop()
                stat["host_wall_inclusive_seconds"] += elapsed
                stat["host_wall_exclusive_seconds"] += max(0.0, elapsed - frame["child_seconds"])
                if frames.stack:
                    frames.stack[-1]["child_seconds"] += elapsed

        return measured

    with ExitStack() as restore_stack:
        for owner, method, region in targets:
            own_value = owner.__dict__.get(method, _MISSING)
            original = getattr(owner, method)
            if not callable(original):
                raise TypeError(f"{owner.__name__}.{method} is not callable")
            setattr(owner, method, wrapped_method(original, region))
            restore_stack.callback(restore, owner, method, own_value)
        yield stats


def _metric(event: Any, *names: str) -> float | None:
    if event is None:
        return None
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return None


def _device_type(event: Any) -> str:
    device = getattr(event, "device_type", None)
    return str(device).rsplit(".", 1)[-1] if device is not None else "UNKNOWN"


def _cpu_metrics(event: Any) -> dict[str, Any]:
    return {
        "profiler_event_count": int(event.count) if event is not None else None,
        "cpu_time_total_us": _metric(event, "cpu_time_total"),
        "self_cpu_time_total_us": _metric(event, "self_cpu_time_total"),
        "cuda_time_total_us": _metric(event, "device_time_total", "cuda_time_total"),
        "self_cuda_time_total_us": _metric(event, "self_device_time_total", "self_cuda_time_total"),
    }


def aggregate_regions(events: Any, stats: dict[str, Any]) -> dict[str, Any]:
    """Preserve device-aware averages; never conflate CUDA spans with kernels."""
    by_name = {name: [] for name in REGIONS}
    for event in events:
        if event.key in by_name:
            by_name[event.key].append(event)
    regions = {}
    warnings = []
    legacy_warnings = []
    for name in REGIONS:
        grouped = {}
        cpu_ranges = []
        for event in by_name[name]:
            device = _device_type(event)
            annotation = getattr(event, "is_user_annotation", None)
            record = {"profiler_event_count": int(event.count), "is_user_annotation": annotation}
            if device == "CPU":
                record.update(_cpu_metrics(event))
                record["timing_semantics"] = "cpu_time_and_attributed_kernel_time"
                if annotation is True or annotation is None:
                    cpu_ranges.append(event)
            else:
                record.update(
                    span_time_total_us=_metric(event, "device_time_total", "cuda_time_total"),
                    self_span_time_total_us=_metric(event, "self_device_time_total", "self_cuda_time_total"),
                    timing_semantics="annotation_span_not_kernel_time" if device == "CUDA" and annotation is True
                    else "legacy_named_span_not_kernel_time" if device == "CUDA" and annotation is None
                    else "unclassified_device_span",
                )
            grouped.setdefault(device, []).append(record)
        status = "matched" if len(cpu_ranges) == 1 else "missing" if not cpu_ranges else "ambiguous"
        primary = cpu_ranges[0] if status == "matched" else None
        metadata = None if primary is None else "explicit" if getattr(primary, "is_user_annotation", None) is True else "unavailable_legacy"
        regions[name] = dict(stats[name], **_cpu_metrics(primary), cpu_aggregate_status=status,
                             cpu_annotation_metadata=metadata, profiler_aggregates_by_device=grouped)
        if metadata == "unavailable_legacy":
            legacy_warnings.append(f"{name}: legacy Torch annotation metadata unavailable; using the unique CPU aggregate for this instrumented region.")
        if stats[name]["calls"] and status != "matched":
            warnings.append(f"{name}: {status} CPU user-range aggregate; primary profiler metrics are null.")
    return {"regions": regions, "cpu_attribution_complete": not warnings,
            "attribution_warnings": warnings, "legacy_annotation_warnings": legacy_warnings,
            "interpretation": list(INTERPRETATION)}


def source_hashes() -> dict[str, str]:
    import ai2.training.benchmark_native_lane_training as driver
    import ai2.simulation.native_lane_vec_env as native

    extra = ("benchmark_campaign.py", "native_navigation_episode.py", "native_navigation_contact.py")
    paths = [Path(__file__).resolve(), *(source_path(ROOT, name) for name in (*driver.CANDIDATE_FILES, *extra)), *native.LANE_SOURCE_FILES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required profiling source missing: {path}")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def build_parser():
    import ai2.training.benchmark_native_lane_training as driver

    parser = driver.campaign.build_parser()
    parser.description = __doc__
    parser.set_defaults(warmup=1, cycles=3)
    parser.add_argument("--native-lanes", action="store_true", default=True)
    parser.add_argument("--lane-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--lane-build-dir", default=str(driver.DEFAULT_BUILD_DIR))
    parser.add_argument("--lane-record-capacity", type=int, default=None)
    return parser


def run(args: Any) -> int:
    if str(args.device).split(":", 1)[0] != "cuda":
        raise ValueError("native lane profiling requires a CUDA policy device")
    if not args.native_lanes or args.mode != "baseline" or args.resident_buffer or args.rollout_policy_graph:
        raise ValueError("profiling requires native lanes, baseline timing, and no resident-buffer or policy-graph substitution")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA is required for native lane profiling; no CPU policy fallback")
    activity = torch.profiler.ProfilerActivity
    if activity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("Torch profiler CUDA activity support is required")

    import ai2.training.benchmark_native_lane_training as driver
    from ai2.simulation.native_lane_vec_env import NativeLaneEnv
    from stable_baselines3.common.policies import ActorCriticPolicy
    from ai2.training.tensor_ppo import TensorPPO
    from ai2.training.tensor_rollout_buffer import TensorRolloutBuffer

    output = driver.campaign.validate_output_path(Path(args.output))
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "native_lane_profile.trace.json"
    summary_path = output / "native_lane_profile.json"
    report = {
        "schema": "ai2.native_lanes.profile.v1", "status": "running", "diagnostic_only": True,
        "headline_sps_eligible": False, "started_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": str(args.device), "gpu_name": torch.cuda.get_device_name(torch.device(args.device)),
        "lane_device": args.lane_device, "source_hashes": source_hashes(),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "activities": ["CPU", "CUDA"], "trace_path": str(trace_path),
        "record_shapes": False, "with_stack": False, "profile_memory": False,
        "interpretation": list(INTERPRETATION),
    }
    driver.campaign.write_json(summary_path, report)
    stats = None
    failure = None
    code = 1
    started = time.perf_counter()
    profiler = torch.profiler.profile(activities=[activity.CPU, activity.CUDA], record_shapes=False,
                                      with_stack=False, profile_memory=False)
    try:
        with profiler:
            with instrument_regions(policy_class=ActorCriticPolicy, env_class=NativeLaneEnv,
                                    buffer_class=TensorRolloutBuffer, ppo_class=TensorPPO,
                                    record_function=torch.profiler.record_function) as stats:
                code = driver.run(args)
    except BaseException as exc:
        failure = (exc, exc.__traceback__)
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["diagnostic_driver_wall_seconds"] = time.perf_counter() - started
    report["exit_code"] = int(code)
    report["status"] = "completed" if code == 0 and failure is None else "failed"
    try:
        if stats is not None:
            report.update(aggregate_regions(profiler.key_averages(), stats))
        profiler.export_chrome_trace(str(trace_path))
        events = profiler.events()
        report["cuda_trace_event_count"] = sum(_device_type(event) == "CUDA" for event in events)
        report["source_hashes_unchanged"] = source_hashes() == report["source_hashes"]
        capture_errors = []
        if not report["source_hashes_unchanged"]:
            capture_errors.append("Source files changed during profiling; this report cannot identify one fixed source revision.")
        if report["cuda_trace_event_count"] == 0:
            capture_errors.append("No CUDA device events were captured; CUDA timing attribution is unavailable in this trace.")
        report["capture_validation_errors"] = capture_errors
        if capture_errors:
            report["interpretation"].extend(capture_errors)
            raise RuntimeError(" ".join(capture_errors))
    except Exception as exc:
        report["artifact_error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "failed"
        report["exit_code"] = 1
        if failure is None:
            failure = (exc, exc.__traceback__)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    driver.campaign.write_json(summary_path, report)
    print(f"native_lane_profile={summary_path}")
    print("Diagnostic only: profiler overhead is included; do not use this run as headline SPS.")
    if failure is not None:
        raise failure[0].with_traceback(failure[1])
    return int(code)


def main(argv: list[str] | None = None) -> int:
    import ai2.training.benchmark_native_lane_training as driver

    parser = build_parser()
    args = driver.campaign.validate_args(parser, parser.parse_args(argv))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
