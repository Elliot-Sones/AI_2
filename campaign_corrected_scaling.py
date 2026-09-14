#!/usr/bin/env python3
"""Dry-run-first launcher for the corrected group4 high-env scaling screen."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import benchmark_campaign
import campaign_suite


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HARNESS = REPO_ROOT / "benchmark_campaign.py"
DEFAULT_ROLLOUT_CACHE = REPO_ROOT / "rollout_cache.py"
DEFAULT_GROUPED_ENV = REPO_ROOT / "grouped_vec_env.py"
SEED_ENV_ORDER = (
    (45, (32, 64, 128, 256)),
    (46, (64, 128, 256, 32)),
    (47, (128, 256, 32, 64)),
)
ROLLOUT_STEPS = {32: 1024, 64: 512, 128: 256, 256: 128}
GROUP_SIZE = 4
TRANSITIONS_PER_CYCLE = 32768
WARMUP_CYCLES = 2
MEASURED_CYCLES = 100
BATCH_SIZE = 8192
EPOCHS = 8
MAX_PER_JOB_TIMEOUT = 600
RESERVE_SECONDS = 300
MIN_FREE_DISK_BYTES = 3 * 1024**3
INVOCATION_PROVENANCE = "invocation_provenance.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--execute", action="store_true", help="run CUDA jobs; default is dry-run only")
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.output = Path(args.output).resolve()
    for attr in ("config", "checkpoint", "source", "train_job", "harness", "rollout_cache_path", "grouped_env_path"):
        setattr(args, attr, Path(getattr(args, attr)).resolve())
    args.per_job_timeout = float(args.per_job_timeout)
    args.deadline_unix = float(args.deadline_unix)
    if not math.isfinite(args.per_job_timeout) or args.per_job_timeout <= 0:
        raise ValueError("--per-job-timeout must be positive")
    if args.per_job_timeout > MAX_PER_JOB_TIMEOUT:
        raise ValueError(f"--per-job-timeout must be <= {MAX_PER_JOB_TIMEOUT}")
    if not math.isfinite(args.deadline_unix):
        raise ValueError("--deadline-unix must be finite")
    if bool(args.execute) and args.deadline_unix <= time.time():
        raise ValueError("--deadline-unix must be in the future for --execute")
    args.mode = "grouped-screen"
    return args


def plan_jobs(args: argparse.Namespace) -> list[campaign_suite.SuiteJob]:
    jobs: list[campaign_suite.SuiteJob] = []
    for seed, env_order in SEED_ENV_ORDER:
        for n_envs in env_order:
            n_steps = ROLLOUT_STEPS[n_envs]
            jobs.append(
                campaign_suite.SuiteJob(
                    name=f"corrected-scaling-seed{seed}-env{n_envs}-group4",
                    arm="grouped",
                    seed=seed,
                    output=args.output / f"corrected-scaling-seed{seed}-env{n_envs}-group4",
                    n_envs=n_envs,
                    n_steps=n_steps,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    warmup=WARMUP_CYCLES,
                    cycles=MEASURED_CYCLES,
                    group_size=GROUP_SIZE,
                    resident_buffer=False,
                )
            )
    return jobs


def preflight(
    args: argparse.Namespace,
    *,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    if str(args.device).lower() != "cuda":
        raise ValueError("--execute requires CUDA device")
    if not math.isfinite(float(args.per_job_timeout)) or float(args.per_job_timeout) <= 0:
        raise ValueError("--per-job-timeout must be positive")
    if float(args.per_job_timeout) > MAX_PER_JOB_TIMEOUT:
        raise ValueError(f"--per-job-timeout must be <= {MAX_PER_JOB_TIMEOUT}")
    free_bytes = ensure_disk_free(args.output, disk_usage=disk_usage)
    marker = campaign_suite.validate_grouped_parity_marker(args)
    require_existing_file_records(provenance(args))
    return {"free_disk_bytes": free_bytes, "grouped_parity_marker": marker}


def ensure_disk_free(output: Path, *, disk_usage: Callable[[Path], Any] = shutil.disk_usage) -> int:
    free_bytes = int(disk_usage(Path(output)).free)
    if free_bytes <= MIN_FREE_DISK_BYTES:
        raise RuntimeError(f"--execute requires more than at least 3 GiB free disk at {output}; got {free_bytes} bytes")
    return free_bytes


def command_maxseconds(per_job_timeout: float) -> float:
    timeout = float(per_job_timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--per-job-timeout must be positive")
    return max(0.001, timeout * 0.95)


def format_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def command_for_job(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> list[str]:
    command = campaign_suite.command_for_job(args, job)
    return [*command, "--maxseconds", format_seconds(command_maxseconds(float(args.per_job_timeout)))]


def dry_run(args: argparse.Namespace, jobs: list[campaign_suite.SuiteJob]) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    commands = [command_for_job(args, job) for job in jobs]
    append_invocation_provenance(args, jobs=jobs, status="dry_run", created=time.time())
    payload = {
        "status": "dry_run",
        "execute": False,
        "planned_jobs": len(jobs),
        "transitions_per_cycle": TRANSITIONS_PER_CYCLE,
        "reserve_seconds": RESERVE_SECONDS,
        "per_job_timeout": args.per_job_timeout,
        "deadline_unix": args.deadline_unix,
        "provenance": provenance(args),
        "jobs": [
            {
                "name": job.name,
                "seed": job.seed,
                "n_envs": job.n_envs,
                "n_steps": job.n_steps,
                "batch_size": job.batch_size,
                "epochs": job.epochs,
                "warmup": job.warmup,
                "cycles": job.cycles,
                "group_size": job.group_size,
                "transitions_per_cycle": job.n_envs * job.n_steps,
                "command": command,
            }
            for job, command in zip(jobs, commands, strict=True)
        ],
    }
    campaign_suite.write_json(args.output / "dry_run_plan.json", payload)
    write_suite_report(
        args.output,
        status="dry_run",
        args=args,
        started=time.time(),
        ended=time.time(),
        planned_jobs=len(jobs),
        records=[],
    )
    return 0


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "harness": campaign_suite.file_record(args.harness),
        "rollout_cache": campaign_suite.file_record(args.rollout_cache_path),
        "grouped_env": campaign_suite.file_record(args.grouped_env_path),
        "train_job": campaign_suite.file_record(args.train_job),
        "config": campaign_suite.file_record(args.config),
        "checkpoint": campaign_suite.file_record(args.checkpoint),
        "grouped_parity_marker": campaign_suite.file_record(campaign_suite.grouped_parity_marker_path(args)),
        "frozen_train": campaign_suite.file_record(frozen_train_path(args)),
        "frozen_environment": campaign_suite.file_record(campaign_suite.frozen_environment_path(args)),
    }


def frozen_train_path(args: argparse.Namespace) -> Path:
    return args.source / "train.py"


def require_existing_file_records(records: dict[str, Any]) -> None:
    for name, record in records.items():
        if not isinstance(record, dict) or record.get("exists") is not True:
            path = record.get("path") if isinstance(record, dict) else None
            raise RuntimeError(f"required provenance file is missing: {name} path={path}")


def write_job_manifest(
    args: argparse.Namespace,
    job: campaign_suite.SuiteJob,
    command: list[str],
    env: dict[str, str],
) -> None:
    campaign_suite.write_job_manifest(args, job, command, env)
    manifest_path = job.output / "job_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.setdefault("files", {})
    files["frozen_train"] = campaign_suite.file_record(frozen_train_path(args))
    files["frozen_environment"] = campaign_suite.file_record(campaign_suite.frozen_environment_path(args))
    manifest["corrected_scaling"] = {
        "transitions_per_cycle": TRANSITIONS_PER_CYCLE,
        "group_size": GROUP_SIZE,
        "runner_timeout_seconds": float(args.per_job_timeout),
        "local_maxseconds": command_maxseconds(float(args.per_job_timeout)),
    }
    campaign_suite.write_json(manifest_path, manifest)


def expected_file_records(args: argparse.Namespace) -> dict[str, Any]:
    return provenance(args)


def existing_completed(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> bool:
    if not job.output.exists():
        return False
    error = campaign_suite.validate_job_summary(job)
    if error is not None:
        raise FileExistsError(f"refusing incomplete existing output: {job.output}: {error}")
    manifest_error = validate_existing_manifest(args, job)
    if manifest_error is not None:
        raise FileExistsError(f"refusing existing completed output with mismatched controls: {job.output}: {manifest_error}")
    return True


def validate_existing_manifest(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> str | None:
    manifest_path = job.output / "job_manifest.json"
    if not manifest_path.is_file():
        return f"missing manifest: {manifest_path}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"malformed manifest: {exc}"
    if manifest.get("job") != campaign_suite.json_safe(dataclasses.asdict(job)):
        return "job controls do not match current plan"
    expected_command = command_for_job(args, job)
    if manifest.get("command") != expected_command:
        return "command does not match current plan"
    expected_env = campaign_suite.whitelisted_env(job=job, args=args)
    if manifest.get("env") != expected_env:
        return "env does not match current plan"
    files = manifest.get("files")
    if not isinstance(files, dict):
        return "files manifest is missing"
    expected_files = expected_file_records(args)
    for name, expected in expected_files.items():
        if expected.get("exists") is not True:
            return f"{name} current file record is missing"
        if files.get(name) != expected:
            return f"{name} file record does not match current inputs"
    return None


def append_invocation_provenance(
    args: argparse.Namespace,
    *,
    jobs: list[campaign_suite.SuiteJob],
    status: str,
    created: float,
    preflight_record: dict[str, Any] | None = None,
) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    record = {
        "status": status,
        "created_unix": created,
        "execute": bool(args.execute),
        "planned_jobs": len(jobs),
        "transitions_per_cycle": TRANSITIONS_PER_CYCLE,
        "reserve_seconds": RESERVE_SECONDS,
        "per_job_timeout": float(args.per_job_timeout),
        "local_maxseconds": command_maxseconds(float(args.per_job_timeout)),
        "deadline_unix": float(args.deadline_unix),
        "provenance": provenance(args),
    }
    if preflight_record is not None:
        record["preflight"] = preflight_record
    with (args.output / INVOCATION_PROVENANCE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(campaign_suite.json_safe(record), sort_keys=True) + "\n")


def run_suite(
    args: argparse.Namespace,
    *,
    runner: Callable[[campaign_suite.SuiteJob, list[str], dict[str, str], float], int] = campaign_suite.run_subprocess_job,
    clock: Any = time,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> int:
    jobs = plan_jobs(args)
    validate_plan(jobs)
    if not args.execute:
        return dry_run(args, jobs)

    args.output.mkdir(parents=True, exist_ok=True)
    ensure_no_running_active_job(args.output)
    started = clock.time()
    if float(args.deadline_unix) <= float(started):
        raise ValueError("--deadline-unix must be in the future for --execute")
    preflight_record = preflight(args, disk_usage=disk_usage)
    append_invocation_provenance(args, jobs=jobs, status="execute", created=started, preflight_record=preflight_record)
    records: list[dict[str, Any]] = []
    status = "completed"
    exit_code = 0

    for index, job in enumerate(jobs):
        if existing_completed(args, job):
            records.append(completed_existing_record(index, job, clock.time()))
            write_suite_report(
                args.output,
                status="running",
                args=args,
                started=started,
                ended=clock.time(),
                planned_jobs=len(jobs),
                records=records,
            )
            continue
        if stop_requested(args.output):
            status = "stop_requested"
            exit_code = 4
            break
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
            break
        ensure_disk_free(args.output, disk_usage=disk_usage)
        job.output.mkdir(parents=True, exist_ok=False)
        command = command_for_job(args, job)
        env = campaign_suite.whitelisted_env(job=job, args=args)
        write_job_manifest(args, job, command, env)
        record = running_record(index, job, clock.time())
        records.append(record)
        campaign_suite.write_active_job(
            args.output,
            {
                "name": job.name,
                "pid": None,
                "status": "running",
                "started_unix": record["started_unix"],
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
        error = campaign_suite.validate_job_summary(job) if return_code == 0 else f"process exit code {return_code}"
        record.update(
            {
                "return_code": return_code,
                "ended_unix": clock.time(),
                "status": "completed" if error is None else "failed",
                "error": error,
            }
        )
        if error is not None:
            status = "aborted"
            exit_code = 2 if return_code == 0 else 1
            break

    campaign_suite.write_active_job(
        args.output,
        {"name": records[-1]["name"] if records else None, "pid": None, "status": status, "ended_unix": clock.time()},
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
    campaign_suite.write_suite_report(
        output,
        status=status,
        args=args,
        started=started,
        ended=ended,
        planned_jobs=planned_jobs,
        records=records,
    )
    report_path = output / "suite_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["completed_jobs"] = sum(
        1 for record in records if record.get("status") in {"completed", "completed_existing"}
    )
    campaign_suite.write_json(report_path, report)


def validate_plan(jobs: list[campaign_suite.SuiteJob]) -> None:
    if len(jobs) != 12:
        raise RuntimeError("corrected scaling plan must contain exactly 12 jobs")
    for job in jobs:
        if job.group_size != GROUP_SIZE or job.arm != "grouped" or job.resident_buffer:
            raise RuntimeError(f"invalid corrected scaling job: {job}")
        if job.n_envs * job.n_steps != TRANSITIONS_PER_CYCLE:
            raise RuntimeError(f"job {job.name} does not have {TRANSITIONS_PER_CYCLE} transitions/cycle")


def stop_requested(output: Path) -> bool:
    return (output / "stop.request").exists()


def ensure_no_running_active_job(output: Path) -> None:
    active_path = output / "active_job.json"
    if not active_path.is_file():
        return
    try:
        active = json.loads(active_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"active job file is malformed: {active_path}") from exc
    if isinstance(active, dict) and active.get("status") == "running":
        raise RuntimeError(f"refusing to start while active job is running: {active.get('name')!r}")


def completed_existing_record(index: int, job: campaign_suite.SuiteJob, now: float) -> dict[str, Any]:
    return {
        "name": job.name,
        "index": index,
        "arm": job.arm,
        "seed": job.seed,
        "n_envs": job.n_envs,
        "n_steps": job.n_steps,
        "group_size": job.group_size,
        "output": str(job.output),
        "return_code": 0,
        "started_unix": None,
        "ended_unix": now,
        "status": "completed_existing",
        "error": None,
    }


def running_record(index: int, job: campaign_suite.SuiteJob, now: float) -> dict[str, Any]:
    return {
        "name": job.name,
        "index": index,
        "arm": job.arm,
        "seed": job.seed,
        "n_envs": job.n_envs,
        "n_steps": job.n_steps,
        "group_size": job.group_size,
        "output": str(job.output),
        "return_code": None,
        "started_unix": now,
        "ended_unix": None,
        "status": "running",
        "error": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = validate_args(parser.parse_args(argv))
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
