#!/usr/bin/env python3
"""Matched sustained launcher for grouped VecEnv throughput pairs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import benchmark_campaign
import campaign_suite


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HARNESS = REPO_ROOT / "benchmark_campaign.py"
DEFAULT_ROLLOUT_CACHE = REPO_ROOT / "rollout_cache.py"
DEFAULT_GROUPED_ENV = REPO_ROOT / "grouped_vec_env.py"
SEEDS = (45, 46, 47)
GROUPED_PAIR_GROUPS = (4, 1)
WARMUP_CYCLES = 2
MEASURED_CYCLES = 100
N_ENVS = 64
N_STEPS = 512
BATCH_SIZE = 8192
EPOCHS = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--per-job-timeout", type=float, default=campaign_suite.MAX_PER_JOB_TIMEOUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", default="0b")
    parser.add_argument("--config", type=Path, default=benchmark_campaign.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=benchmark_campaign.DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=benchmark_campaign.DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=benchmark_campaign.DEFAULT_TRAIN_JOB)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--rollout-cache-path", type=Path, default=DEFAULT_ROLLOUT_CACHE)
    parser.add_argument("--grouped-env-path", type=Path, default=DEFAULT_GROUPED_ENV)
    parser.add_argument("--pair-offset", type=int, default=0)
    parser.add_argument("--pair-limit", type=int)
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.per_job_timeout <= 0:
        raise ValueError("--per-job-timeout must be positive")
    if args.per_job_timeout > campaign_suite.MAX_PER_JOB_TIMEOUT:
        raise ValueError(f"--per-job-timeout must be <= {campaign_suite.MAX_PER_JOB_TIMEOUT}")
    if args.pair_offset < 0:
        raise ValueError("--pair-offset must be nonnegative")
    if args.pair_limit is not None and args.pair_limit <= 0:
        raise ValueError("--pair-limit must be positive")
    if args.pair_offset >= len(SEEDS):
        raise ValueError("--pair-offset/--pair-limit must select at least one seed pair")
    if args.pair_limit is not None and args.pair_offset + args.pair_limit > len(SEEDS):
        raise ValueError("--pair-offset + --pair-limit exceeds available seed pairs")
    args.output = Path(args.output).resolve()
    for attr in ("config", "checkpoint", "source", "train_job", "harness", "rollout_cache_path", "grouped_env_path"):
        setattr(args, attr, Path(getattr(args, attr)).resolve())
    args.mode = "grouped-pairs"
    return args


def selected_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    end = len(SEEDS) if args.pair_limit is None else args.pair_offset + args.pair_limit
    return SEEDS[args.pair_offset:end]


def plan_jobs(args: argparse.Namespace) -> list[campaign_suite.SuiteJob]:
    jobs: list[campaign_suite.SuiteJob] = []
    for pair_index, seed in enumerate(selected_seeds(args), start=args.pair_offset):
        groups = GROUPED_PAIR_GROUPS if pair_index % 2 == 0 else tuple(reversed(GROUPED_PAIR_GROUPS))
        for group_size in groups:
            label = f"group{group_size}"
            jobs.append(
                campaign_suite.SuiteJob(
                    name=f"grouped-pairs-seed{seed}-{label}",
                    arm="grouped" if group_size > 1 else "control",
                    seed=seed,
                    output=args.output / f"grouped-pairs-seed{seed}-{label}",
                    n_envs=N_ENVS,
                    n_steps=N_STEPS,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    warmup=WARMUP_CYCLES,
                    cycles=MEASURED_CYCLES,
                    group_size=group_size,
                    resident_buffer=False,
                )
            )
    return jobs


def grouped_pair_parity_marker_path(args: argparse.Namespace) -> Path:
    return args.output / "grouped_game_parity_passed.json"


def frozen_environment_path(args: argparse.Namespace) -> Path:
    return args.source / "UTMIST-AI2-main" / "environment" / "environment.py"


def _marker_int(marker: dict[str, Any], key: str) -> int:
    try:
        return int(marker[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"grouped-pairs parity marker {key} must be an integer: {marker.get(key)!r}") from exc


def validate_grouped_pair_parity_marker(args: argparse.Namespace) -> dict[str, Any]:
    marker_path = grouped_pair_parity_marker_path(args)
    if not marker_path.is_file():
        raise RuntimeError(f"grouped-pairs requires parity marker before speed runs: {marker_path}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"grouped-pairs parity marker malformed: {marker_path}") from exc
    if not isinstance(marker, dict):
        raise RuntimeError(f"grouped-pairs parity marker must be an object: {marker_path}")
    if marker.get("status") != "passed":
        raise RuntimeError(f"grouped-pairs parity marker status must be 'passed': {marker.get('status')!r}")
    if _marker_int(marker, "exit_code") != 0:
        raise RuntimeError(f"grouped-pairs parity marker exit_code must be 0: {marker.get('exit_code')!r}")
    if _marker_int(marker, "skipped") != 0:
        raise RuntimeError(f"grouped-pairs parity marker skipped must be 0: {marker.get('skipped')!r}")
    if _marker_int(marker, "steps_per_case") < 1000:
        raise RuntimeError(f"grouped-pairs parity marker steps_per_case must be >= 1000: {marker.get('steps_per_case')!r}")
    group_sizes = marker.get("group_sizes")
    try:
        actual_group_sizes = {int(size) for size in group_sizes} if isinstance(group_sizes, list) else set()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"grouped-pairs parity marker group_sizes must include 4: {group_sizes!r}") from exc
    if 4 not in actual_group_sizes:
        raise RuntimeError(f"grouped-pairs parity marker group_sizes must include 4: {group_sizes!r}")
    runner_sha256 = marker.get("runner_sha256")
    try:
        actual_runner_sha256 = campaign_suite.sha256_file(args.grouped_env_path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"grouped-pairs grouped env path missing for runner_sha256: {args.grouped_env_path}") from exc
    if runner_sha256 != actual_runner_sha256:
        raise RuntimeError(
            "grouped-pairs parity marker runner_sha256 does not match current grouped env: "
            f"marker={runner_sha256!r} actual={actual_runner_sha256!r}"
        )
    frozen_sha256 = marker.get("frozen_environment_sha256")
    try:
        actual_frozen_sha256 = campaign_suite.sha256_file(frozen_environment_path(args))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"grouped-pairs frozen environment path missing for frozen_environment_sha256: {frozen_environment_path(args)}"
        ) from exc
    if frozen_sha256 != actual_frozen_sha256:
        raise RuntimeError(
            "grouped-pairs parity marker frozen_environment_sha256 does not match current frozen environment: "
            f"marker={frozen_sha256!r} actual={actual_frozen_sha256!r}"
        )
    return marker


def _manifest_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values["mode"] = "grouped-screen"
    return argparse.Namespace(**values)


def write_job_manifest(args: argparse.Namespace, job: campaign_suite.SuiteJob, command: list[str], env: dict[str, str]) -> None:
    campaign_suite.write_job_manifest(_manifest_args(args), job, command, env)


def run_suite(
    args: argparse.Namespace,
    *,
    runner: Callable[[campaign_suite.SuiteJob, list[str], dict[str, str], float], int] = campaign_suite.run_subprocess_job,
    clock: Any = time,
) -> int:
    validate_grouped_pair_parity_marker(args)
    jobs = plan_jobs(args)
    campaign_suite.ensure_no_completed_outputs(jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    started = clock.time()
    records: list[dict[str, Any]] = []
    status = "completed"
    exit_code = 0

    for index, job in enumerate(jobs):
        remaining = float(args.deadline_unix) - float(clock.time())
        required = float(args.per_job_timeout) + campaign_suite.RESERVE_SECONDS
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
            campaign_suite.write_suite_report(
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
        command = campaign_suite.command_for_job(args, job)
        env = campaign_suite.whitelisted_env(job=job, args=args)
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
        campaign_suite.write_active_job(
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
        campaign_suite.write_suite_report(
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
        running_record.update(
            {
                "return_code": return_code,
                "ended_unix": clock.time(),
                "status": "completed" if error is None else "failed",
                "error": error,
            }
        )
        campaign_suite.write_suite_report(
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

    campaign_suite.write_active_job(
        args.output,
        {
            "name": records[-1]["name"] if records else None,
            "pid": None,
            "status": status,
            "ended_unix": clock.time(),
        },
    )
    campaign_suite.write_suite_report(
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
