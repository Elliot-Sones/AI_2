#!/usr/bin/env python3
"""Plan or sequentially run source-bound native CUDA PPO ablations.

Dry-run is the default. Execution requires --run and a current native CUDA
parity receipt. Compiled and differently batched arms require measured numeric
validation; a completed timing job is not evidence of gameplay quality.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import benchmark_campaign
import campaign_native_lanes as native_campaign
import campaign_suite


ROOT = Path(__file__).resolve().parent
FLAGS = ("optimizer_cpu_steps", "deferred_metrics", "deferred_timeouts",
         "boundary_validation", "compile_loss", "compile_gae")
ARM_FLAGS = {
    "baseline": (),
    "counters": ("optimizer_cpu_steps",),
    "metrics": ("deferred_metrics",),
    "timeouts": ("deferred_timeouts",),
    "validation": ("boundary_validation",),
    "compiled_loss": ("deferred_metrics", "compile_loss"),
    "compiled_gae": ("compile_gae",),
    "combined_exact": ("optimizer_cpu_steps", "deferred_metrics", "boundary_validation"),
    "combined_collector": ("optimizer_cpu_steps", "deferred_metrics", "boundary_validation", "deferred_timeouts"),
    "combined_all": FLAGS,
}
SCREEN_ARMS = tuple(ARM_FLAGS)[:7]
REQUIRED_EXPERIMENT_SOURCES = (
    "campaign_native_ppo_ablations.py", "experimental_native_lane_training.py",
    "ppo_optimizer_counters.py", "ppo_update_optimizations.py", "tensor_ppo.py",
    "tensor_rollout_buffer.py", "benchmark_native_lane_training.py",
)
# Share the lock with the existing native-lane launcher, not just this script.
GPU_LOCK_PATH = native_campaign.GPU_LOCK_PATH
gpu_lock = native_campaign.gpu_lock
require_idle_gpu = native_campaign.require_idle_gpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screen", "paired"), required=True)
    parser.add_argument("--arms", nargs="+", help="arm names; defaults to seven screening arms or baseline combined_exact")
    parser.add_argument("--combined", nargs="+", metavar="FLAG",
                        help="define a combined arm using optimization flag names or individual arm aliases")
    parser.add_argument("--source", type=Path, default=benchmark_campaign.DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=benchmark_campaign.DEFAULT_TRAIN_JOB)
    parser.add_argument("--config", type=Path, default=benchmark_campaign.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=benchmark_campaign.DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--per-job-timeout", type=float, default=campaign_suite.MAX_PER_JOB_TIMEOUT)
    parser.add_argument("--parity-receipt", type=Path)
    parser.add_argument("--run", action="store_true", help="execute sequential CUDA jobs; default writes only a plan")
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("source", "train_job", "config", "checkpoint", "output", "parity_receipt"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).resolve())
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= 0:
        raise ValueError("--deadline-unix must be positive and finite")
    if not math.isfinite(args.per_job_timeout) or not 0 < args.per_job_timeout <= campaign_suite.MAX_PER_JOB_TIMEOUT:
        raise ValueError("--per-job-timeout must be positive, finite, and <=900")
    combined = set()
    for token in args.combined or ():
        name = token.replace("-", "_")
        if name in FLAGS:
            combined.add(name)
        elif name in ARM_FLAGS and name != "baseline":
            combined.update(ARM_FLAGS[name])
        else:
            raise ValueError(f"unknown combined optimization: {token}")
    if "compile_loss" in combined and "deferred_metrics" not in combined:
        raise ValueError("compile_loss requires deferred_metrics in --combined")
    args.combined = tuple(name for name in FLAGS if name in combined)
    if args.arms is None:
        args.arms = SCREEN_ARMS if args.mode == "screen" else ("baseline", "combined" if combined else "combined_exact")
    else:
        args.arms = tuple(args.arms)
    if not args.arms or len(set(args.arms)) != len(args.arms):
        raise ValueError("--arms must contain distinct arm names")
    for arm in args.arms:
        if arm not in ARM_FLAGS and not (arm == "combined" and combined):
            raise ValueError(f"unknown or undefined arm: {arm}")
    if args.mode == "paired" and ("baseline" not in args.arms or len(args.arms) < 2):
        raise ValueError("paired mode requires baseline and at least one selected comparison arm")
    return args


def optimizations_for_arm(args: argparse.Namespace, arm: str) -> dict[str, bool]:
    selected = args.combined if arm == "combined" else ARM_FLAGS[arm]
    return {flag: flag in selected for flag in FLAGS}


def plan_jobs(args: argparse.Namespace) -> list[campaign_suite.SuiteJob]:
    jobs = []
    seeds = (45, 46, 47) if args.mode == "paired" else (45,)
    for seed_index, seed in enumerate(seeds):
        shift = seed_index % len(args.arms)
        arms = args.arms[shift:] + args.arms[:shift] if args.mode == "paired" else args.arms
        for n_envs in (256, 4096):
            for arm in arms:
                name = f"native-ppo-{args.mode}-env{n_envs}-seed{seed}-{arm}"
                cycles = (100 if n_envs == 256 else 20) if args.mode == "paired" else (10 if n_envs == 256 else 5)
                jobs.append(campaign_suite.SuiteJob(
                    name=name, arm=arm, seed=seed, output=args.output / name,
                    n_envs=n_envs, n_steps=128, batch_size=8192, epochs=8,
                    group_size=1, warmup=2, cycles=cycles,
                ))
    return jobs


def command_for_job(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> list[str]:
    command = [sys.executable, str(ROOT / "experimental_native_lane_training.py"),
               "--source", str(args.source), "--train-job", str(args.train_job),
               "--config", str(args.config), "--checkpoint", str(args.checkpoint),
               "--output", str(job.output), "--device", "cuda", "--lane-device", "cuda",
               "--native-lanes", "--phase", "0b", "--mode", "baseline",
               "--maxseconds", str(args.per_job_timeout), *job.command_args]
    command.extend("--" + name.replace("_", "-") for name, enabled in optimizations_for_arm(args, job.arm).items() if enabled)
    return command


def env_for_job(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> dict[str, str]:
    env = campaign_suite.whitelisted_env(job=job, args=args)
    options = optimizations_for_arm(args, job.arm)
    enabled = ",".join(name for name, value in options.items() if value) or "none"
    env.update({
        "WANDB_MODE": "offline",
        "WANDB_RUN_GROUP": f"native-ppo-ablations-{args.mode}",
        "WANDB_NOTES": (f"Goal: matched native CUDA PPO throughput ablations; arm_flags={enabled}; "
                        f"{campaign_suite.wandb_notes(args, job)}; numeric equivalence must be measured"),
        "AI2_NATIVE_INPUT": "0", "AI2_NATIVE_OBSERVATION": "0",
        "PYTHONPATH": os.pathsep.join(dict.fromkeys((str(ROOT), str(args.source)))),
    })
    return env


def source_manifest(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    inputs = native_campaign.source_inputs(args)
    inputs.update({name: ROOT / name for name in REQUIRED_EXPERIMENT_SOURCES})
    records = {name: campaign_suite.file_record(path) for name, path in inputs.items()}
    missing = [record["path"] for record in records.values() if not record["exists"]]
    if missing:
        raise FileNotFoundError(f"required input missing: {missing}")
    return records


def validate_job_summary(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> str | None:
    error = native_campaign.validate_job_summary(job)
    if error is not None:
        return error
    try:
        provenance = json.loads((job.output / "native_lane_provenance.json").read_text(encoding="utf-8"))
        if not isinstance(provenance, dict) or provenance.get("schema") != "ai2.native_ppo_ablations.v1":
            return "missing experimental native PPO provenance schema"
        if provenance.get("lane_device") != "cuda" or str(provenance.get("policy_device")).split(":", 1)[0] != "cuda":
            return "job provenance requires CUDA policy and native CUDA lanes"
        options = provenance.get("optimizations")
        if (not isinstance(options, dict) or any(type(value) is not bool for value in options.values())
                or options != optimizations_for_arm(args, job.arm)):
            return "job optimization provenance does not match the requested arm"
        if options["compile_loss"] or options["compile_gae"]:
            runtime = json.loads((job.output / "optimization_runtime.json").read_text(encoding="utf-8"))
            if not isinstance(runtime, dict) or runtime.get("compile_headline_eligible") is not True:
                return "compiled job requires compile_headline_eligible=true with no measured recompilation"
    except (OSError, ValueError, TypeError) as exc:
        return f"invalid or missing optimization provenance: {type(exc).__name__}: {exc}"
    return None


def run_suite(args: argparse.Namespace, *,
              runner: Callable = campaign_suite.run_subprocess_job, clock: Any = time) -> int:
    args = validate_args(args)
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = plan_jobs(args)
    started = clock.time()
    records: list[dict[str, Any]] = []

    def report(status: str, error: str | None = None) -> None:
        campaign_suite.write_suite_report(args.output, status=status, args=args, started=started,
                                          ended=clock.time(), planned_jobs=len(jobs), records=records)
        if error is not None:
            path = args.output / "suite_report.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            campaign_suite.write_json(path, {**payload, "error": error})
        campaign_suite.write_active_job(args.output, {"status": status, "pid": None, "ended_unix": clock.time()})

    try:
        receipt = native_campaign.validate_parity_receipt(args) if args.run else None
        files = source_manifest(args)
        if receipt is not None:
            for name, expected_hash in receipt["source_hashes"].items():
                record = campaign_suite.file_record(native_campaign.ROOT / name)
                if record.get("sha256") != expected_hash:
                    raise ValueError(f"qualification source hash changed during validation: {name}")
                files[name] = record
        campaign_suite.write_json(args.output / "source_manifest.json", files)
        campaign_suite.write_json(args.output / "native_ppo_ablations_plan.json", {
            "schema": "ai2.native_ppo_ablations.plan.v1", "mode": args.mode, "dry_run": not args.run,
            "policy_device": "cuda", "lane_device": "cuda", "source_manifest": "source_manifest.json",
            "comparison_note": "Compare arms within each environment count; 256 and 4096 change rollout geometry.",
            "numerical_equivalence": "Must be measured independently; combined_exact is an arm name, not a validation claim.",
            "jobs": [{"job": dataclasses.asdict(job), "optimizations": optimizations_for_arm(args, job.arm),
                      "command": command_for_job(args, job), "env": env_for_job(args, job)} for job in jobs],
        })
    except Exception as exc:
        report("validation_failed", f"{type(exc).__name__}: {exc}")
        return 2
    if not args.run:
        report("dry_run")
        return 0

    try:
        with gpu_lock(GPU_LOCK_PATH):
            for index, job in enumerate(jobs):
                try:
                    native_campaign.ensure_sources_unchanged(files)
                except RuntimeError as exc:
                    report("source_changed", str(exc))
                    return 4
                remaining = args.deadline_unix - clock.time()
                required = args.per_job_timeout + campaign_suite.RESERVE_SECONDS
                if remaining < required:
                    records.append({"name": job.name, "status": "skipped_deadline",
                                    "remaining_seconds": remaining, "required_seconds": required})
                    report("deadline_insufficient")
                    return 3
                require_idle_gpu()
                job.output.mkdir(exist_ok=False)
                command, env = command_for_job(args, job), env_for_job(args, job)
                campaign_suite.write_json(job.output / "job_manifest.json", {
                    "job": dataclasses.asdict(job), "command": command, "env": env, "files": files,
                    "optimizations": optimizations_for_arm(args, job.arm),
                    "command_sha256": campaign_suite.sha256_bytes(json.dumps(command, sort_keys=True).encode()),
                    "env_sha256": campaign_suite.sha256_bytes(json.dumps(env, sort_keys=True).encode()),
                    "policy_device": "cuda", "created_unix": clock.time(), "log_path": str(job.output / "job.log"),
                })
                job_started = clock.time()
                record = {"name": job.name, "index": index, "arm": job.arm, "seed": job.seed,
                          "n_envs": job.n_envs, "status": "running", "output": str(job.output),
                          "started_unix": job_started, "return_code": None, "error": None}
                records.append(record)
                report("running")
                try:
                    rc = runner(job, command, env, args.per_job_timeout)
                    elapsed = clock.time() - job_started
                    error = validate_job_summary(args, job) if rc == 0 else f"process exit code {rc}"
                    if elapsed >= args.per_job_timeout:
                        rc, error = 124, f"job exceeded timeout ({elapsed:.3f}s)"
                except Exception as exc:
                    rc, error = 1, f"{type(exc).__name__}: {exc}"
                record.update(return_code=rc, ended_unix=clock.time(), error=error,
                              status="completed" if error is None else "failed")
                if clock.time() >= args.deadline_unix:
                    record.update(status="failed", error="job finished after deadline")
                    report("deadline_exceeded")
                    return 3
                if error is not None:
                    report("aborted")
                    return 2 if rc == 0 else 1
                try:
                    native_campaign.ensure_sources_unchanged(files)
                except RuntimeError as exc:
                    record.update(status="failed", error=str(exc))
                    report("source_changed", str(exc))
                    return 4
                report("running")
    except Exception as exc:
        report("validation_failed" if not records else "aborted", f"{type(exc).__name__}: {exc}")
        return 2
    report("completed")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run_suite(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
