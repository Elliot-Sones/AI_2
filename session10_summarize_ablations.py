#!/usr/bin/env python3
"""Summarize downloaded native PPO suites without loading models or changing inputs.

Usage: --suite PATH [--suite PATH ...] --output FRESH.json
Exit 0: report written without findings; 1: report written with negative findings;
2: invalid invocation or output failure. Small samples remain descriptive even
with exit 0. Recorded source hashes are checked for consistency, not against
unavailable remote source files. Artifact hashes make the report reproducible.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


FLAGS = ("optimizer_cpu_steps", "deferred_metrics", "deferred_timeouts",
         "boundary_validation", "compile_loss", "compile_gae")
GEOMETRY = ("n_envs", "n_steps", "batch_size", "epochs", "warmup", "cycles", "group_size")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value: Any, name: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def number(value: Any, name: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"{name} must be finite numeric")
    require(value > 0 if positive else value >= 0, f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def close(actual: Any, expected: float, name: str) -> None:
    require(math.isclose(number(actual, name), expected, rel_tol=1e-8, abs_tol=1e-9),
            f"{name} does not reconcile: {actual!r} != {expected!r}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


class Artifacts:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def read(self, path: Path, *, lines: bool = False) -> Any:
        try:
            data = path.read_bytes()
            self.records[str(path)] = {"path": str(path), "size": len(data),
                                       "sha256": hashlib.sha256(data).hexdigest()}
            text = data.decode("utf-8")
            values = text.splitlines() if lines else [text]
            parsed = []
            for index, value in enumerate(values, 1):
                require(bool(value.strip()), f"blank JSON record at line {index}")
                item = json.loads(value, object_pairs_hook=unique_object, parse_constant=reject_constant)
                require(isinstance(item, dict), f"expected JSON object at record {index}")
                parsed.append(item)
            return parsed if lines else parsed[0]
        except (OSError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc


def source_hashes(files: Any) -> dict[str, str]:
    require(isinstance(files, dict), "source files must be a mapping")
    require({"checkpoint", "config", "train_job"} <= files.keys(), "source manifest missing checkpoint/config/train_job")
    hashes = {}
    for name, record in files.items():
        require(isinstance(record, dict) and record.get("exists") is True, f"source input unavailable: {name}")
        digest = record.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                f"invalid source hash: {name}")
        hashes[name] = digest
    return hashes


def validate_flags(flags: Any) -> dict[str, bool]:
    require(isinstance(flags, dict) and set(flags) == set(FLAGS)
            and all(type(value) is bool for value in flags.values()), "invalid optimization flags")
    require(not flags["compile_loss"] or flags["deferred_metrics"], "compile_loss requires deferred_metrics")
    return flags


def validate_timings(job: dict, summary: dict, rows: list[dict], record: dict) -> dict:
    warmup, cycles = job["warmup"], job["cycles"]
    total_cycles, transitions = warmup + cycles, job["n_envs"] * job["n_steps"]
    require(summary.get("status") == "completed", f"campaign summary is {summary.get('status')!r}, not completed")
    require(summary.get("mode") == "baseline", "campaign must use unprofiled baseline timing mode")
    expected_counts = {"completed_cycles_total": total_cycles, "completed_measured_cycles": cycles,
                       "requested_measured_cycles": cycles, "warmup_cycles": warmup,
                       "measured_transitions": cycles * transitions}
    for key, expected in expected_counts.items():
        require(integer(summary.get(key), key) == expected, f"summary {key} differs from planned count")
    plan = summary.get("plan")
    require(isinstance(plan, dict), "missing summary geometry")
    expected_plan = {key: job[key] for key in GEOMETRY if key != "group_size"}
    expected_plan.update(transitions_per_cycle=transitions, steps=total_cycles * transitions,
                         minibatches_per_epoch=math.ceil(transitions / job["batch_size"]),
                         minibatches_per_cycle=math.ceil(transitions / job["batch_size"]) * job["epochs"])
    for key, expected in expected_plan.items():
        require(integer(plan.get(key), key) == expected, f"summary geometry {key} does not match job manifest")
    require(len(rows) == total_cycles, "campaign_metrics.jsonl count does not match warmup plus measured cycles")
    previous_steps = None
    for index, row in enumerate(rows, 1):
        require(integer(row.get("cycle_index"), "cycle_index", 1) == index, "noncontiguous cycle indices")
        require(row.get("phase") == ("warmup" if index <= warmup else "measured"), "incorrect warmup/measured phase")
        require(integer(row.get("measured_cycle"), "measured_cycle") == max(0, index - warmup), "incorrect measured cycle index")
        require(row.get("rollout_completed") is True, "incomplete rollout record")
        require(integer(row.get("transitions"), "transitions", 1) == transitions, "incorrect transitions per cycle")
        steps = integer(row.get("num_timesteps"), "num_timesteps", transitions)
        if previous_steps is not None:
            require(steps - previous_steps == transitions, "num_timesteps does not advance by one rollout")
        previous_steps = steps
        elapsed = number(row.get("cycle_seconds"), "cycle_seconds", positive=True)
        collect = number(row.get("rollout_seconds"), "rollout_seconds", positive=True)
        update = number(row.get("update_seconds"), "update_seconds", positive=True)
        residual = number(row.get("boundary_seconds"), "boundary_seconds")
        close(collect + update + residual, elapsed, "cycle component residual")
        close(row.get("transitions_per_second"), transitions / elapsed, "cycle SPS")
    measured = rows[warmup:]
    sums = {"measured_cycle_seconds": sum(row["cycle_seconds"] for row in measured),
            "measured_collect_seconds": sum(row["rollout_seconds"] for row in measured),
            "measured_update_seconds": sum(row["update_seconds"] for row in measured),
            "measured_boundary_seconds": sum(row["boundary_seconds"] for row in measured)}
    for key, expected in sums.items():
        close(summary.get(key), expected, key)
    sps = cycles * transitions / sums["measured_cycle_seconds"]
    close(summary.get("measured_transitions_per_second"), sps, "summary SPS")
    warmup_seconds = sum(row["cycle_seconds"] for row in rows[:warmup])
    close(summary.get("learning_wall_seconds"), warmup_seconds + sums["measured_cycle_seconds"], "learning wall")
    full_wall = number(record.get("ended_unix"), "ended_unix") - number(record.get("started_unix"), "started_unix")
    require(full_wall + 1e-6 >= warmup_seconds + sums["measured_cycle_seconds"], "full job wall is shorter than learning window")
    return {**sums, "sps": sps, "measured_transitions": cycles * transitions,
            "cycle_seconds_mean": sums["measured_cycle_seconds"] / cycles,
            "collect_seconds_mean": sums["measured_collect_seconds"] / cycles,
            "update_seconds_mean": sums["measured_update_seconds"] / cycles,
            "boundary_seconds_mean": sums["measured_boundary_seconds"] / cycles,
            "warmup_seconds": warmup_seconds, "full_job_wall_seconds": full_wall,
            "startup_seconds": number(summary.get("startup_seconds"), "startup_seconds"),
            "starting_num_timesteps": rows[0]["num_timesteps"] - transitions,
            "ending_num_timesteps": rows[-1]["num_timesteps"]}


def inspect_job(suite: Path, mode: str, entry: Any, records: list[dict], hashes: dict,
                artifacts: Artifacts, initial_errors: list[str]) -> dict:
    result = {"suite": str(suite), "mode": mode, "eligible": False, "errors": list(initial_errors),
              "compile_required": False, "compile_headline_eligible": None}
    try:
        require(isinstance(entry, dict) and isinstance(entry.get("job"), dict), "invalid planned job")
        job = entry["job"]
        result.update({key: job.get(key) for key in ("name", "arm", "seed", *GEOMETRY)})
        name = job.get("name")
        require(isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name,
                "job name must be a single directory name")
        require(isinstance(job.get("arm"), str) and bool(job["arm"]), "invalid arm")
        integer(job.get("seed"), "seed")
        for key in GEOMETRY:
            integer(job.get(key), key, 0 if key == "warmup" else 1)
        require(job.get("resident_buffer") is False and job["group_size"] == 1, "unexpected resident/grouped workload")
        flags = validate_flags(entry.get("optimizations"))
        require(job["arm"] != "baseline" or not any(flags.values()), "baseline must have all flags disabled")
        result.update(flags=flags, source_hashes=hashes, compile_required=flags["compile_loss"] or flags["compile_gae"])
        matching = [record for record in records if record.get("name") == name]
        require(len(matching) == 1, "missing or duplicate job execution record")
        record = matching[0]
        result["execution_status"] = record.get("status")
        require(record.get("status") == "completed" and type(record.get("return_code")) is int
                and record["return_code"] == 0 and not record.get("error"),
                f"execution not completed: status={record.get('status')!r}, error={record.get('error')!r}")
        for key in ("arm", "seed", "n_envs"):
            require(record.get(key) == job[key], f"execution record {key} differs from plan")
        # Downloaded manifests retain remote output paths. Only local suite/name is read.
        directory = suite / name
        result["output"] = str(directory)
        manifest = artifacts.read(directory / "job_manifest.json")
        require(manifest.get("job") == job, "job manifest identity/geometry differs from suite plan")
        require(validate_flags(manifest.get("optimizations")) == flags, "manifest flags differ from suite plan")
        require(source_hashes(manifest.get("files")) == hashes, "job source hashes differ from suite source manifest")
        require(str(manifest.get("policy_device")).split(":")[0] == "cuda", "manifest policy is not CUDA")
        provenance = artifacts.read(directory / "native_lane_provenance.json")
        require(provenance.get("schema") == "ai2.native_ppo_ablations.v1", "invalid experimental provenance schema")
        require(provenance.get("lane_device") == "cuda" and str(provenance.get("policy_device")).split(":")[0] == "cuda",
                "provenance must use CUDA policy and native CUDA lanes")
        require(validate_flags(provenance.get("optimizations")) == flags, "provenance optimization flags differ from manifest")
        summary = artifacts.read(directory / "campaign_summary.json")
        rows = artifacts.read(directory / "campaign_metrics.jsonl", lines=True)
        result["timings"] = validate_timings(job, summary, rows, record)
        if result["compile_required"]:
            result["compile_headline_eligible"] = False
            runtime = artifacts.read(directory / "optimization_runtime.json")
            result["compile_runtime_claimed_eligible"] = runtime.get("compile_headline_eligible") is True
            result["measured_recompile_cycles"] = runtime.get("measured_recompile_cycles")
            require(result["compile_runtime_claimed_eligible"], "compiled job is headline-ineligible")
            require(runtime.get("measured_recompile_cycles") == [], "measured recompilation recorded")
            histories = runtime.get("compiler_history_per_update")
            require(isinstance(histories, list) and bool(histories), "missing compiler histories")
            require(job["warmup"] >= 1, "compiled job requires warmup")
            for history in histories:
                require(isinstance(history, list) and len(history) == job["warmup"] + job["cycles"]
                        and all(isinstance(snapshot, dict) for snapshot in history), "incomplete compiler history")
                require(all(history[i] == history[i - 1] for i in range(job["warmup"], len(history))),
                        "compiler counters changed during measured cycles")
            result["compile_headline_eligible"] = True
        result["eligible"] = not result["errors"]
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        result["errors"].append(str(exc))
    return result


def workload(job: dict) -> str:
    return json.dumps({"geometry": {key: job[key] for key in GEOMETRY}, "source_hashes": job["source_hashes"],
                       "starting_num_timesteps": job["timings"]["starting_num_timesteps"]}, sort_keys=True)


def compare_jobs(jobs: list[dict], findings: list[dict]) -> list[dict]:
    groups = {}
    for job in jobs:
        if (not isinstance(job.get("arm"), str) or not job["arm"] or job["arm"] == "baseline"
                or type(job.get("n_envs")) is not int or type(job.get("seed")) is not int):
            continue
        key = (job["suite"], job["mode"], job.get("n_envs"), job["arm"])
        group = groups.setdefault(key, {"suite": key[0], "mode": key[1], "n_envs": key[2], "arm": key[3],
                                        "pairs": [], "unpaired_jobs": [], "workloads": set(), "flag_sets": set()})
        matches = [baseline for baseline in jobs if baseline.get("arm") == "baseline"
                   and all(baseline.get(field) == job.get(field) for field in ("suite", "mode", "seed", "n_envs"))]
        reason = None
        if not job["eligible"]:
            reason = "candidate ineligible"
        elif len(matches) != 1 or not matches[0]["eligible"]:
            reason = "missing, duplicate, or ineligible same-suite baseline"
        elif workload(job) != workload(matches[0]):
            reason = "unequal workload or source hashes versus baseline"
        if reason:
            group["unpaired_jobs"].append({"name": job.get("name"), "seed": job.get("seed"), "reason": reason})
            findings.append({"suite": job["suite"], "job": job.get("name"), "error": reason})
            continue
        baseline = matches[0]
        ratio = job["timings"]["sps"] / baseline["timings"]["sps"]
        group["pairs"].append({"seed": job["seed"], "baseline": baseline["name"], "candidate": job["name"],
                               "ratio": ratio, "throughput_gain_percent": 100 * (ratio - 1),
                               "time_saving_percent": 100 * (1 - 1 / ratio)})
        group["workloads"].add(workload(job))
        group["flag_sets"].add(json.dumps(job["flags"], sort_keys=True))
    for group in groups.values():
        compatible = len(group.pop("workloads")) <= 1 and len(group.pop("flag_sets")) <= 1
        if not compatible:
            findings.append({"suite": group["suite"], "arm": group["arm"], "error": "unequal workloads or flags across seeds; not aggregated"})
        pairs = sorted(group["pairs"], key=lambda pair: pair["seed"])
        group["pairs"] = pairs
        seeds = sorted({pair["seed"] for pair in pairs})
        unique = len(seeds) == len(pairs)
        if not unique:
            findings.append({"suite": group["suite"], "arm": group["arm"], "error": "duplicate seed pairs; not aggregated"})
        ratios = [pair["ratio"] for pair in pairs] if compatible and unique else []
        group.update(seeds=seeds, count_seeds=len(seeds), claim_eligible=group["mode"] == "paired" and len(seeds) >= 3 and bool(ratios),
                     interpretation="screen_descriptive" if group["mode"] == "screen" else "paired_descriptive" if len(seeds) < 3 else "paired_estimate",
                     ratio_geomean=math.exp(statistics.mean(math.log(ratio) for ratio in ratios)) if ratios else None,
                     ratio_median=statistics.median(ratios) if ratios else None,
                     ratio_min=min(ratios) if ratios else None, ratio_max=max(ratios) if ratios else None)
    return list(groups.values())


def summarize_suites(paths: list[Path]) -> dict:
    artifacts, suites, jobs, findings, seen = Artifacts(), [], [], [], set()
    for path in paths:
        suite = Path(path).resolve()
        if suite in seen:
            findings.append({"suite": str(suite), "error": "duplicate suite input; counted once"})
            continue
        seen.add(suite)
        suite_result = {"suite": str(suite), "errors": []}
        suites.append(suite_result)
        try:
            plan = artifacts.read(suite / "native_ppo_ablations_plan.json")
            require(plan.get("schema") == "ai2.native_ppo_ablations.plan.v1", "invalid suite plan schema")
            mode = plan.get("mode")
            require(mode in ("screen", "paired"), "unknown suite mode")
            suite_result["mode"] = mode
            require(plan.get("dry_run") is False, "dry-run suite has no execution evidence")
            require(plan.get("policy_device") == "cuda" and plan.get("lane_device") == "cuda", "suite must use CUDA")
            entries = plan.get("jobs")
            require(isinstance(entries, list) and bool(entries), "suite plan has no jobs")
            hashes, records, job_errors = {}, [], []
            try:
                hashes = source_hashes(artifacts.read(suite / "source_manifest.json"))
            except ValueError as exc:
                job_errors.append(str(exc))
            try:
                report = artifacts.read(suite / "suite_report.json")
                suite_result["execution_status"] = report.get("status")
                require(report.get("mode") == mode, "suite report/plan mode mismatch")
                records = report.get("jobs")
                require(isinstance(records, list) and all(isinstance(record, dict) for record in records), "invalid suite execution records")
                if report.get("status") != "completed":
                    suite_result["errors"].append(f"suite execution status: {report.get('status')!r}")
                require(integer(report.get("planned_jobs"), "planned_jobs") == len(entries), "planned job count mismatch")
                require(integer(report.get("completed_jobs"), "completed_jobs") == sum(record.get("status") == "completed" for record in records),
                        "completed job count mismatch")
            except (ValueError, TypeError) as exc:
                job_errors.append(str(exc))
                if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
                    records = []
            identities = [entry.get("job", {}) if isinstance(entry, dict) else {} for entry in entries]
            names = Counter(str(job.get("name")) for job in identities if isinstance(job, dict))
            keys = Counter(str((job.get("arm"), job.get("seed"), job.get("n_envs"))) for job in identities if isinstance(job, dict))
            for record in records:
                if str(record.get("name")) not in names:
                    suite_result["errors"].append(f"unplanned execution record: {record.get('name')!r}")
            for entry, identity in zip(entries, identities):
                errors = list(job_errors)
                if isinstance(identity, dict) and (names[str(identity.get("name"))] > 1 or keys[str((identity.get("arm"), identity.get("seed"), identity.get("n_envs")))] > 1):
                    errors.append("duplicate planned name or arm/seed/environment identity")
                job = inspect_job(suite, mode, entry, records, hashes, artifacts, errors)
                jobs.append(job)
                findings.extend({"suite": str(suite), "job": job.get("name"), "error": error} for error in job["errors"])
        except (ValueError, TypeError, KeyError) as exc:
            suite_result["errors"].append(str(exc))
        findings.extend({"suite": str(suite), "error": error} for error in suite_result["errors"])
    comparisons = compare_jobs(jobs, findings)
    return {"schema": "ai2.session10.ablation_summary.v1", "status": "findings" if findings else "complete",
            "suites": suites, "jobs": jobs, "comparisons": comparisons, "findings": findings,
            "input_artifacts": [artifacts.records[key] for key in sorted(artifacts.records)],
            "notes": ["SPS is measured transitions / summed measured cycle seconds, never mean per-cycle SPS.",
                      "Warmup is excluded; compiled jobs require unchanged measured compiler counters, not compile-time subtraction.",
                      "Full job wall includes subprocess setup and closeout; startup_seconds covers only harness startup.",
                      "Pairs never cross suite, mode, seed, environment count, workload, or recorded source hashes.",
                      "Screening and fewer than three paired seeds are descriptive; claim_eligible is a sample-count gate, not significance or numerical-equivalence proof.",
                      "Source validation checks recorded hash consistency only; original remote files are not reread.",
                      "No trajectory, checkpoint, quality, GPU telemetry, or statistical-significance claims are inferred."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), f"output already exists: {args.output}")
        report = summarize_suites(args.suite)
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        return 0 if report["status"] == "complete" else 1
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
