#!/usr/bin/env python3
"""Bounded native-lane qualification benchmarks; dry-run unless --run is given.

The paired mode compares Python/Pymunk, native CPU, and native CUDA simulators
with CUDA policies. Scaling changes rollout geometry and is reported separately.
Execution requires a source-bound, non-skipped native CUDA parity receipt.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import SOURCE_LAYOUT_VERSION, source_inventory, source_path

import argparse
from contextlib import contextmanager
import dataclasses
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import ai2.runtime.benchmark_campaign as benchmark_campaign
import ai2.runtime.campaign_suite as campaign_suite


ROOT = PROJECT_ROOT
ARMS = ("python_cpu", "native_cpu", "native_cuda")
NATIVE_ARMS = ARMS[1:]
RESERVE_SECONDS = campaign_suite.RESERVE_SECONDS
GPU_LOCK_PATH = Path(tempfile.gettempdir()) / f"ai2-native-lanes-gpu-{os.getuid()}.lock"
RECEIPT_SCHEMA = "ai2.native_lanes.parity.v1"
REQUIRED_QUALIFICATION_SOURCES = (
    "qualify_native_lanes.py", "native_lane_vec_env.py", "native_lane_cpu_adapter.py",
    "benchmark_native_lane_training.py", "tensor_ppo.py", "tensor_rollout_buffer.py",
    "native_navigation_episode.py", "native_navigation_contact.py", "native/navigation_spaces.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "paired", "scaling"), required=True)
    parser.add_argument("--source", type=Path, default=benchmark_campaign.DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=benchmark_campaign.DEFAULT_TRAIN_JOB)
    parser.add_argument("--config", type=Path, default=benchmark_campaign.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=benchmark_campaign.DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--per-job-timeout", type=float, default=campaign_suite.MAX_PER_JOB_TIMEOUT)
    parser.add_argument("--parity-receipt", type=Path)
    parser.add_argument("--run", action="store_true", help="execute sequential CUDA policy jobs; default only writes a plan")
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
    return args


def plan_jobs(args: argparse.Namespace) -> list[campaign_suite.SuiteJob]:
    jobs = []
    seeds = (45, 46, 47) if args.mode == "paired" else (45,)
    sizes = (1024, 4096) if args.mode == "scaling" else (256,)
    for index, seed in enumerate(seeds):
        arms = ARMS[index:] + ARMS[:index] if args.mode == "paired" else NATIVE_ARMS
        for n_envs in sizes:
            for arm in arms:
                name = f"native-lanes-{args.mode}-env{n_envs}-seed{seed}-{arm}"
                jobs.append(campaign_suite.SuiteJob(
                    name=name, arm=arm, seed=seed, output=args.output / name,
                    n_envs=n_envs, n_steps=128, batch_size=8192, epochs=8,
                    group_size=4 if arm == "python_cpu" else 1,
                    warmup=0 if args.mode == "smoke" else 2,
                    cycles={"smoke": 2, "paired": 100, "scaling": 20}[args.mode],
                ))
    if args.mode == "scaling":
        jobs.sort(key=lambda job: job.arm != "native_cuda")
    return jobs


def command_for_job(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> list[str]:
    native = job.arm != "python_cpu"
    harness = source_path(ROOT, "benchmark_native_lane_training.py" if native else "benchmark_campaign.py")
    command = [sys.executable, str(harness), "--source", str(args.source),
               "--train-job", str(args.train_job), "--config", str(args.config),
               "--checkpoint", str(args.checkpoint), "--output", str(job.output),
               "--device", "cuda", "--phase", "0b", "--maxseconds", str(args.per_job_timeout),
               *job.command_args]
    if native:
        command.extend(("--native-lanes", "--lane-device", job.arm.removeprefix("native_")))
    else:
        command.extend(("--grouped-env-path", str(source_path(ROOT, "native_isolated_vec_env.py"))))
    return command


def env_for_job(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> dict[str, str]:
    env = campaign_suite.whitelisted_env(job=job, args=args)
    native_patch = "1" if job.arm == "python_cpu" else "0"
    env.update({"AI2_NATIVE_INPUT": native_patch, "AI2_NATIVE_OBSERVATION": native_patch})
    env["WANDB_RUN_GROUP"] = f"native-lanes-{args.mode}"
    return env


def native_source_hashes() -> dict[str, str]:
    from ai2.simulation.native_lane_vec_env import LANE_SOURCE_FILES

    # Hash every declared file; a missing input must never disappear from a receipt gate.
    return {path.name: campaign_suite.sha256_file(path) for path in LANE_SOURCE_FILES}


def source_inputs(args: argparse.Namespace) -> dict[str, Path]:
    inputs = {"checkpoint": args.checkpoint, "config": args.config, "train_job": args.train_job}
    inputs.update(source_inventory(ROOT))
    inputs.update({f"native/{path.name}": path for path in sorted((ROOT / "native").iterdir())
                   if path.is_file() and path.suffix in (".h", ".c", ".cu", ".json")})
    inputs.update({f"source/{path.relative_to(args.source).as_posix()}": path
                   for path in sorted(args.source.rglob("*"))
                   if path.is_file() and path.suffix in (".py", ".yaml", ".yml", ".json")
                   and "__pycache__" not in path.parts})
    inputs["source/train.py"] = source_path(args.source, "train.py")
    inputs["source/UTMIST-AI2-main/environment/environment.py"] = (
        args.source / "UTMIST-AI2-main/environment/environment.py")
    if args.parity_receipt is not None:
        inputs["parity_receipt"] = args.parity_receipt
    return inputs


def source_manifest(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    records = {name: campaign_suite.file_record(path) for name, path in source_inputs(args).items()}
    missing = [record["path"] for record in records.values() if not record["exists"]]
    if missing:
        raise FileNotFoundError(f"required input missing: {missing}")
    return records


def validate_parity_receipt(args: argparse.Namespace) -> dict[str, Any]:
    if args.parity_receipt is None:
        raise ValueError("--run requires --parity-receipt from qualify_native_lanes.py")
    receipt = json.loads(args.parity_receipt.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError(f"native lane parity receipt requires schema {RECEIPT_SCHEMA}")
    if receipt.get("source_layout_version") != SOURCE_LAYOUT_VERSION:
        raise ValueError(f"native lane parity receipt requires source layout {SOURCE_LAYOUT_VERSION}")
    for key, expected in (("status", "passed"), ("exit_code", 0), ("skipped", 0)):
        if receipt.get(key) != expected:
            raise ValueError(f"native lane parity receipt {key} must be {expected!r}")
    native_sources = native_source_hashes()
    if receipt.get("native_sources") != native_sources:
        raise ValueError("native lane parity receipt native_sources do not match current sources")
    source_hashes = receipt.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("native lane parity receipt source_hashes missing")
    inventory = source_inventory(ROOT)
    required = (set(REQUIRED_QUALIFICATION_SOURCES) | set(inventory)
                | {f"native/{name}" for name in native_sources})
    missing = required - source_hashes.keys()
    if missing:
        raise ValueError(f"native lane parity receipt source_hashes missing: {sorted(missing)}")
    root = ROOT.resolve()
    for name, expected_hash in source_hashes.items():
        relative = Path(name)
        if (not name or "\\" in name or "\x00" in name or relative.is_absolute()
                or ".." in relative.parts or relative.as_posix() != name):
            raise ValueError(f"unsafe qualification source path: {name!r}")
        path = inventory.get(name, source_path(root, relative)).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"unsafe qualification source path outside repository: {name!r}")
        if not path.is_file() or campaign_suite.sha256_file(path) != expected_hash:
            raise ValueError(f"native lane parity receipt source hash does not match: {name}")
    return receipt


def ensure_sources_unchanged(records: dict[str, dict[str, Any]]) -> None:
    for name, record in records.items():
        path = Path(record["path"])
        if not path.is_file() or campaign_suite.sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"source changed during suite: {name} ({path})")


@contextmanager
def gpu_lock(path: Path):
    """Exclude another instance of this launcher, even with a different output path."""
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another native lane campaign holds the GPU lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def require_idle_gpu() -> None:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=15, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"cannot verify idle CUDA GPU: {result.stderr.strip()}")
    if result.stdout.strip():
        raise RuntimeError(f"GPU has active compute jobs; refusing concurrent benchmark: {result.stdout.strip()}")


def validate_job_summary(job: campaign_suite.SuiteJob) -> str | None:
    try:
        summary = json.loads((job.output / "campaign_summary.json").read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or summary.get("status") != "completed":
            return "summary status must be completed"
        expected = {"completed_cycles_total": job.warmup + job.cycles,
                    "completed_measured_cycles": job.cycles, "requested_measured_cycles": job.cycles,
                    "warmup_cycles": job.warmup}
        for name, value in expected.items():
            if type(summary.get(name)) is not int or summary[name] != value:
                return f"summary {name} expected={value} actual={summary.get(name)!r}"
        plan = summary.get("plan")
        if not isinstance(plan, dict):
            return "summary plan missing"
        for name, value in {"n_envs": job.n_envs, "n_steps": job.n_steps, "batch_size": job.batch_size,
                            "epochs": job.epochs, "transitions_per_cycle": job.n_envs * job.n_steps}.items():
            if type(plan.get(name)) is not int or plan[name] != value:
                return f"summary plan {name} expected={value} actual={plan.get(name)!r}"
        return campaign_suite.validate_job_summary(job)
    except (OSError, ValueError, TypeError) as exc:
        return f"invalid or missing summary: {type(exc).__name__}: {exc}"


def run_suite(args: argparse.Namespace, *,
              runner: Callable = campaign_suite.run_subprocess_job, clock: Any = time) -> int:
    args = validate_args(args)
    # A fresh output preserves both failed and completed evidence from previous runs.
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
        campaign_suite.write_active_job(args.output, {"status": status, "pid": None,
                                                      "ended_unix": clock.time()})

    try:
        receipt = validate_parity_receipt(args) if args.run else None
        files = source_manifest(args)
        if receipt is not None:
            # Keep every qualified source bound throughout the suite, including nested extras.
            for name, expected_hash in receipt["source_hashes"].items():
                record = files.get(name) or campaign_suite.file_record(source_path(ROOT, name))
                if record.get("sha256") != expected_hash:
                    raise ValueError(f"qualification source hash changed during validation: {name}")
                files[name] = record
        campaign_suite.write_json(args.output / "source_manifest.json", files)
        campaign_suite.write_json(args.output / "native_lanes_plan.json", {
            "schema": "ai2.native_lanes.plan.v1", "mode": args.mode, "dry_run": not args.run,
            "source_layout_version": SOURCE_LAYOUT_VERSION,
            "rollout_shape_changed": args.mode == "scaling", "policy_device": "cuda",
            "comparison_note": ("Scaling changes transitions per rollout from 32768 to 131072 and 524288; "
                                "report separately from the matched paired comparison." if args.mode == "scaling"
                                else "Same rollout geometry across arms; completed-update throughput only."),
            "jobs": [{"job": dataclasses.asdict(job), "command": command_for_job(args, job),
                      "env": env_for_job(args, job)} for job in jobs],
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
                    ensure_sources_unchanged(files)
                except RuntimeError as exc:
                    report("source_changed", str(exc))
                    return 4
                remaining = args.deadline_unix - clock.time()
                required = args.per_job_timeout + RESERVE_SECONDS
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
                    "source_layout_version": SOURCE_LAYOUT_VERSION,
                    "command_sha256": campaign_suite.sha256_bytes(json.dumps(command, sort_keys=True).encode()),
                    "env_sha256": campaign_suite.sha256_bytes(json.dumps(env, sort_keys=True).encode()),
                    "policy_device": "cuda", "created_unix": clock.time(),
                    "log_path": str(job.output / "job.log"),
                })
                job_started = clock.time()
                record = {"name": job.name, "index": index, "arm": job.arm, "seed": job.seed,
                          "status": "running", "output": str(job.output), "started_unix": job_started,
                          "return_code": None, "error": None}
                records.append(record)
                report("running")
                try:
                    rc = runner(job, command, env, args.per_job_timeout)
                    elapsed = clock.time() - job_started
                    error = validate_job_summary(job) if rc == 0 else f"process exit code {rc}"
                    # A child handling SIGTERM with exit 0 is still a timed-out job.
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
                    ensure_sources_unchanged(files)
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
