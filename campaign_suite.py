#!/usr/bin/env python3
"""Bounded sequential launcher for the throughput-scaling benchmark campaign."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import benchmark_campaign


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HARNESS = REPO_ROOT / "benchmark_campaign.py"
DEFAULT_ROLLOUT_CACHE = REPO_ROOT / "rollout_cache.py"
DEFAULT_GROUPED_ENV = REPO_ROOT / "grouped_vec_env.py"
RESERVE_SECONDS = 60
MAX_PER_JOB_TIMEOUT = 900
ENV_WHITELIST = (
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYGAME_HIDE_SUPPORT_PROMPT",
    "SDL_AUDIODRIVER",
    "SDL_VIDEODRIVER",
    "WANDB_MODE",
    "WANDB_NAME",
    "WANDB_NOTES",
    "WANDB_RUN_GROUP",
)


@dataclasses.dataclass(frozen=True)
class SuiteJob:
    name: str
    arm: str
    seed: int
    output: Path
    n_envs: int = 64
    n_steps: int = 512
    batch_size: int = 8192
    epochs: int = 8
    warmup: int = 2
    cycles: int = 100
    group_size: int = 1
    resident_buffer: bool = False

    @property
    def command_args(self) -> list[str]:
        args = [
            "--n-envs",
            str(self.n_envs),
            "--n-steps",
            str(self.n_steps),
            "--batch-size",
            str(self.batch_size),
            "--epochs",
            str(self.epochs),
            "--seed",
            str(self.seed),
            "--warmup",
            str(self.warmup),
            "--cycles",
            str(self.cycles),
            "--group-size",
            str(self.group_size),
        ]
        if self.resident_buffer:
            args.append("--resident-buffer")
        return args


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    return {
        "path": str(resolved),
        "exists": True,
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["paired-cache", "smoke-cache", "grouped-screen"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--per-job-timeout", type=float, default=MAX_PER_JOB_TIMEOUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", default="0b")
    parser.add_argument("--config", type=Path, default=benchmark_campaign.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=benchmark_campaign.DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=benchmark_campaign.DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=benchmark_campaign.DEFAULT_TRAIN_JOB)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--rollout-cache-path", type=Path, default=DEFAULT_ROLLOUT_CACHE)
    parser.add_argument("--grouped-env-path", type=Path, default=DEFAULT_GROUPED_ENV)
    parser.add_argument("--suite-seed", type=int, default=20260906)
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.output.is_absolute():
        args.output = args.output.resolve()
    if args.per_job_timeout <= 0:
        raise ValueError("--per-job-timeout must be positive")
    if args.per_job_timeout > MAX_PER_JOB_TIMEOUT:
        raise ValueError("--per-job-timeout must be <= 900")
    for attr in ("config", "checkpoint", "source", "train_job", "harness", "rollout_cache_path", "grouped_env_path"):
        setattr(args, attr, Path(getattr(args, attr)).resolve())
    args.output = Path(args.output).resolve()
    return args


def _paired_jobs(args: argparse.Namespace, *, seeds: list[int], warmup: int, cycles: int, prefix: str) -> list[SuiteJob]:
    jobs: list[SuiteJob] = []
    start_with_cache = bool(random.Random(args.suite_seed).getrandbits(1))
    for index, seed in enumerate(seeds):
        arms = ["cache", "control"] if (start_with_cache ^ bool(index % 2)) else ["control", "cache"]
        for arm in arms:
            jobs.append(
                SuiteJob(
                    name=f"{prefix}-seed{seed}-{arm}",
                    arm=arm,
                    seed=seed,
                    output=args.output / f"{prefix}-seed{seed}-{arm}",
                    warmup=warmup,
                    cycles=cycles,
                    resident_buffer=arm == "cache",
                )
            )
    return jobs


def plan_jobs(args: argparse.Namespace) -> list[SuiteJob]:
    if args.mode == "paired-cache":
        return _paired_jobs(args, seeds=[45, 46, 47], warmup=2, cycles=100, prefix="paired-cache")
    if args.mode == "smoke-cache":
        return [
            SuiteJob(
                name=f"smoke-cache-seed45-{arm}",
                arm=arm,
                seed=45,
                output=args.output / f"smoke-cache-seed45-{arm}",
                warmup=1,
                cycles=2,
                resident_buffer=arm == "cache",
            )
            for arm in ("control", "cache")
        ]
    validate_grouped_parity_marker(args)
    return [
        SuiteJob(name="grouped-screen-env64-group2", arm="grouped", seed=45, output=args.output / "grouped-screen-env64-group2", cycles=20, group_size=2),
        SuiteJob(name="grouped-screen-env64-group4", arm="grouped", seed=45, output=args.output / "grouped-screen-env64-group4", cycles=20, group_size=4),
        SuiteJob(name="grouped-screen-env64-group8", arm="grouped", seed=45, output=args.output / "grouped-screen-env64-group8", cycles=20, group_size=8),
        SuiteJob(name="grouped-screen-env128-group8", arm="grouped", seed=45, output=args.output / "grouped-screen-env128-group8", n_envs=128, n_steps=256, cycles=20, group_size=8),
        SuiteJob(name="grouped-screen-env256-group8", arm="grouped", seed=45, output=args.output / "grouped-screen-env256-group8", n_envs=256, n_steps=128, cycles=20, group_size=8),
    ]


def grouped_parity_marker_path(args: argparse.Namespace) -> Path:
    return args.output / "grouped_game_parity_passed.json"


def frozen_environment_path(args: argparse.Namespace) -> Path:
    return args.source / "UTMIST-AI2-main" / "environment" / "environment.py"


def marker_int(marker: dict[str, Any], key: str) -> int:
    try:
        return int(marker[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"grouped-screen parity marker {key} must be an integer: {marker.get(key)!r}") from exc


def validate_grouped_parity_marker(args: argparse.Namespace) -> dict[str, Any]:
    marker_path = grouped_parity_marker_path(args)
    if not marker_path.is_file():
        raise RuntimeError(f"grouped-screen requires parity marker before speed runs: {marker_path}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"grouped-screen parity marker malformed: {marker_path}") from exc
    if not isinstance(marker, dict):
        raise RuntimeError(f"grouped-screen parity marker must be an object: {marker_path}")
    if marker.get("status") != "passed":
        raise RuntimeError(f"grouped-screen parity marker status must be 'passed': {marker.get('status')!r}")
    if marker_int(marker, "exit_code") != 0:
        raise RuntimeError(f"grouped-screen parity marker exit_code must be 0: {marker.get('exit_code')!r}")
    if marker_int(marker, "skipped") != 0:
        raise RuntimeError(f"grouped-screen parity marker skipped must be 0: {marker.get('skipped')!r}")
    if marker_int(marker, "steps_per_case") < 1000:
        raise RuntimeError(f"grouped-screen parity marker steps_per_case must be >= 1000: {marker.get('steps_per_case')!r}")
    group_sizes = marker.get("group_sizes")
    try:
        actual_group_sizes = {int(size) for size in group_sizes} if isinstance(group_sizes, list) else set()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"grouped-screen parity marker group_sizes must include 2, 4, and 8: {group_sizes!r}") from exc
    if not {2, 4, 8}.issubset(actual_group_sizes):
        raise RuntimeError(f"grouped-screen parity marker group_sizes must include 2, 4, and 8: {group_sizes!r}")
    runner_sha256 = marker.get("runner_sha256")
    try:
        actual_runner_sha256 = sha256_file(args.grouped_env_path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"grouped-screen grouped env path missing for runner_sha256: {args.grouped_env_path}") from exc
    if runner_sha256 != actual_runner_sha256:
        raise RuntimeError(
            "grouped-screen parity marker runner_sha256 does not match current grouped env: "
            f"marker={runner_sha256!r} actual={actual_runner_sha256!r}"
        )
    frozen_sha256 = marker.get("frozen_environment_sha256")
    try:
        actual_frozen_sha256 = sha256_file(frozen_environment_path(args))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"grouped-screen frozen environment path missing for frozen_environment_sha256: {frozen_environment_path(args)}"
        ) from exc
    if frozen_sha256 != actual_frozen_sha256:
        raise RuntimeError(
            "grouped-screen parity marker frozen_environment_sha256 does not match current frozen environment: "
            f"marker={frozen_sha256!r} actual={actual_frozen_sha256!r}"
        )
    return marker


def command_for_job(args: argparse.Namespace, job: SuiteJob) -> list[str]:
    return [
        sys.executable,
        str(args.harness),
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--source",
        str(args.source),
        "--train-job",
        str(args.train_job),
        "--output",
        str(job.output),
        "--device",
        str(args.device),
        "--phase",
        str(args.phase),
        "--rollout-cache-path",
        str(args.rollout_cache_path),
        "--grouped-env-path",
        str(args.grouped_env_path),
        *job.command_args,
    ]


def expected_completed_cycles(job: SuiteJob) -> int:
    return job.warmup + job.cycles


def wandb_notes(args: argparse.Namespace, job: SuiteJob) -> str:
    grouping = f"group_size={job.group_size}" if job.group_size > 1 else "group_size=1"
    return (
        "Goal: throughput-scaling benchmark only; not gameplay quality proof. "
        f"job={job.name}; arm={job.arm}; seed={job.seed}; {grouping}; "
        f"n_envs={job.n_envs}; n_steps={job.n_steps}; warmup={job.warmup}; "
        f"measured_cycles={job.cycles}; expected_completed_cycles={expected_completed_cycles(job)}; "
        f"checkpoint={Path(args.checkpoint).resolve()}"
    )


def whitelisted_env(*, job: SuiteJob, args: argparse.Namespace) -> dict[str, str]:
    env = {
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
        "PYGAME_HIDE_SUPPORT_PROMPT": os.environ.get("PYGAME_HIDE_SUPPORT_PROMPT", "1"),
        "SDL_AUDIODRIVER": os.environ.get("SDL_AUDIODRIVER", "dummy"),
        "SDL_VIDEODRIVER": os.environ.get("SDL_VIDEODRIVER", "dummy"),
        "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
        "WANDB_NAME": job.name,
        "WANDB_NOTES": wandb_notes(args, job),
        "WANDB_RUN_GROUP": "throughput-scaling-20260906",
    }
    return {key: env[key] for key in ENV_WHITELIST}


def copy_snapshot(path: Path, snapshot_dir: Path, name: str) -> None:
    if not Path(path).is_file():
        return
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, snapshot_dir / name)


def write_job_manifest(args: argparse.Namespace, job: SuiteJob, command: list[str], env: dict[str, str]) -> None:
    files = {
        "harness": file_record(args.harness),
        "rollout_cache": file_record(args.rollout_cache_path),
        "grouped_env": file_record(args.grouped_env_path),
        "train_job": file_record(args.train_job),
        "config": file_record(args.config),
        "checkpoint": file_record(args.checkpoint),
    }
    if args.mode == "grouped-screen":
        files["grouped_parity_marker"] = file_record(grouped_parity_marker_path(args))
    snapshot = job.output / "source_snapshot"
    copy_snapshot(args.harness, snapshot, "benchmark_campaign.py")
    copy_snapshot(args.rollout_cache_path, snapshot, "rollout_cache.py")
    copy_snapshot(args.grouped_env_path, snapshot, "grouped_vec_env.py")
    copy_snapshot(args.train_job, snapshot, "train_job.py")
    write_json(
        job.output / "job_manifest.json",
        {
            "job": dataclasses.asdict(job),
            "command": command,
            "command_sha256": sha256_bytes(json.dumps(command, sort_keys=True).encode("utf-8")),
            "env": env,
            "env_sha256": sha256_bytes(json.dumps(env, sort_keys=True).encode("utf-8")),
            "files": files,
            "created_unix": time.time(),
            "log_path": str(job.output / "job.log"),
        },
    )


def write_active_job(output: Path, payload: dict[str, Any]) -> None:
    write_json(output / "active_job.json", payload)


def write_suite_report(
    output: Path,
    *,
    status: str,
    args: argparse.Namespace,
    started: float,
    ended: float,
    planned_jobs: int,
    records: list[dict[str, Any]],
) -> None:
    write_json(
        output / "suite_report.json",
        {
            "status": status,
            "mode": args.mode,
            "started_unix": started,
            "ended_unix": ended,
            "deadline_unix": args.deadline_unix,
            "per_job_timeout": args.per_job_timeout,
            "reserve_seconds": RESERVE_SECONDS,
            "planned_jobs": planned_jobs,
            "completed_jobs": sum(1 for record in records if record.get("status") == "completed"),
            "jobs": records,
        },
    )


def run_subprocess_job(job: SuiteJob, command: list[str], env: dict[str, str], timeout: float) -> int:
    merged_env = os.environ.copy()
    merged_env.update(env)
    log_path = job.output / "job.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=merged_env,
            start_new_session=True,
        )
        write_active_job(
            job.output.parent,
            {
                "name": job.name,
                "pid": int(process.pid),
                "status": "running",
                "started_unix": time.time(),
                "output": str(job.output),
                "log_path": str(log_path),
            },
        )
        try:
            return int(process.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return int(process.wait(timeout=15))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return 124


def validate_job_summary(job: SuiteJob) -> str | None:
    summary_path = job.output / "campaign_summary.json"
    if not summary_path.is_file():
        return f"missing summary: {summary_path}"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("status") != "completed":
        return f"summary status was {summary.get('status')!r}"
    completed = summary.get("completed_measured_cycles")
    requested = summary.get("requested_measured_cycles", job.cycles)
    if int(completed) != job.cycles or int(requested) != job.cycles:
        return f"measured cycles expected={job.cycles} completed={completed} requested={requested}"
    warmup = summary.get("warmup_cycles", job.warmup)
    if int(warmup) != job.warmup:
        return f"warmup cycles expected={job.warmup} actual={warmup}"
    plan = summary.get("plan") or {}
    transitions = plan.get("transitions_per_cycle", job.n_envs * job.n_steps)
    if int(transitions) != job.n_envs * job.n_steps:
        return f"transitions per cycle expected={job.n_envs * job.n_steps} actual={transitions}"
    return None


def ensure_no_completed_outputs(jobs: list[SuiteJob]) -> None:
    for job in jobs:
        summary_path = job.output / "campaign_summary.json"
        if not summary_path.is_file():
            continue
        try:
            status = json.loads(summary_path.read_text(encoding="utf-8")).get("status")
        except json.JSONDecodeError:
            status = None
        if status == "completed":
            raise FileExistsError(f"refusing to overwrite completed output: {job.output}")


def run_suite(
    args: argparse.Namespace,
    *,
    runner: Callable[[SuiteJob, list[str], dict[str, str], float], int] = run_subprocess_job,
    clock: Any = time,
) -> int:
    jobs = plan_jobs(args)
    ensure_no_completed_outputs(jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    started = clock.time()
    records: list[dict[str, Any]] = []
    status = "completed"
    exit_code = 0

    for index, job in enumerate(jobs):
        remaining = float(args.deadline_unix) - float(clock.time())
        required = float(args.per_job_timeout) + RESERVE_SECONDS
        if remaining < required:
            status = "deadline_insufficient"
            exit_code = 3
            records.append(
                {
                    "name": job.name,
                    "status": "skipped_deadline",
                    "remaining_seconds": remaining,
                    "required_seconds": required,
                }
            )
            write_suite_report(
                args.output,
                status=status,
                args=args,
                started=started,
                ended=clock.time(),
                planned_jobs=len(jobs),
                records=records,
            )
            break
        job.output.mkdir(parents=True, exist_ok=False)
        command = command_for_job(args, job)
        env = whitelisted_env(job=job, args=args)
        write_job_manifest(args, job, command, env)
        job_started = clock.time()
        running_record = {
            "name": job.name,
            "index": index,
            "arm": job.arm,
            "seed": job.seed,
            "output": str(job.output),
            "return_code": None,
            "started_unix": job_started,
            "ended_unix": None,
            "status": "running",
            "error": None,
        }
        records.append(running_record)
        write_active_job(
            args.output,
            {
                "name": job.name,
                "pid": None,
                "status": "running",
                "started_unix": job_started,
                "output": str(job.output),
                "log_path": str(job.output / "job.log"),
            },
        )
        write_suite_report(
            args.output,
            status="running",
            args=args,
            started=started,
            ended=clock.time(),
            planned_jobs=len(jobs),
            records=records,
        )
        return_code = runner(job, command, env, float(args.per_job_timeout))
        error = validate_job_summary(job) if return_code == 0 else f"process exit code {return_code}"
        running_record.update(
            {
                "return_code": return_code,
                "ended_unix": clock.time(),
                "status": "completed" if error is None else "failed",
                "error": error,
            }
        )
        write_suite_report(
            args.output,
            status=status if error is None else "aborted",
            args=args,
            started=started,
            ended=clock.time(),
            planned_jobs=len(jobs),
            records=records,
        )
        if error is not None:
            status = "aborted"
            exit_code = 2 if return_code == 0 else 1
            break

    write_active_job(
        args.output,
        {
            "name": records[-1]["name"] if records else None,
            "pid": None,
            "status": status,
            "ended_unix": clock.time(),
        },
    )
    write_suite_report(
        args.output,
        status=status,
        args=args,
        started=started,
        ended=clock.time(),
        planned_jobs=len(jobs),
        records=records,
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = validate_args(parser.parse_args(argv))
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
