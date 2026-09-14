#!/usr/bin/env python3
"""Dry-run-by-default candidate grouped scaling screen.

This is a short throughput screen, not an acceptance or speedup claim.  It
reuses the frozen ``benchmark_campaign.py`` harness and only varies grouped
worker process topology for a fixed current workload: 256 envs, 128-step
rollouts, batch 8192, 8 epochs, phase 0b, seed 45, 2 warmup cycles and 30
measured cycles.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import benchmark_campaign
import validate_grouped_candidate


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HARNESS = REPO_ROOT / "benchmark_campaign.py"
DEFAULT_GROUPED_ENV = REPO_ROOT / "native_shared_batched_vec_env.py"
DEFAULT_PARITY_RECEIPT_NAME = "candidate_game_parity_passed.json"
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
SUPPORTED_GROUP_SIZES = (4, 8)
N_ENVS = 256
N_STEPS = 128
BATCH_SIZE = 8192
EPOCHS = 8
SEED = 45
WARMUP = 2
CYCLES = 30
PER_JOB_TIMEOUT_SECONDS = 300.0
RESERVE_SECONDS = 300.0


@dataclasses.dataclass(frozen=True)
class CandidateJob:
    name: str
    group_size: int
    output: Path
    n_envs: int = N_ENVS
    n_steps: int = N_STEPS
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    seed: int = SEED
    warmup: int = WARMUP
    cycles: int = CYCLES


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_if_new_or_same(path: Path, payload: Any) -> None:
    text = json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"existing file differs; refusing overwrite: {path}")
    path.write_text(text, encoding="utf-8")


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


def parse_group_sizes(raw: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--group-sizes must be comma-separated integers") from exc
    if not values:
        raise ValueError("--group-sizes must include at least one supported group size")
    unsupported = [value for value in values if value not in SUPPORTED_GROUP_SIZES]
    if unsupported:
        raise ValueError(f"supported group sizes are {list(SUPPORTED_GROUP_SIZES)}; got unsupported {unsupported}")
    if len(set(values)) != len(values):
        raise ValueError("--group-sizes must not contain duplicates")
    return values


def frozen_environment_path(source: Path) -> Path:
    return Path(source) / "UTMIST-AI2-main" / "environment" / "environment.py"


def safe_snapshot_name(label: str, path: Path) -> str:
    safe_label = label.replace("::", "__").replace("/", "__")
    suffix = Path(path).suffix
    if suffix and not safe_label.endswith(suffix):
        return f"{safe_label}{suffix}"
    return safe_label


def required_parity_sources(source: Path, train_job: Path, candidate: Path) -> list[Path]:
    return validate_grouped_candidate.required_sources(Path(source), Path(train_job), Path(candidate))


def parity_source_label(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return f"parity_source::{resolved.relative_to(REPO_ROOT).as_posix()}"
    except ValueError:
        return f"parity_source::{resolved.name}"


def source_inputs(args: argparse.Namespace) -> dict[str, Path]:
    inputs = {
        "driver": Path(__file__).resolve(),
        "harness": Path(args.harness).resolve(),
        "grouped_env": Path(args.grouped_env_path).resolve(),
        "train_job": Path(args.train_job).resolve(),
        "config": Path(args.config).resolve(),
        "checkpoint": Path(args.checkpoint).resolve(),
        "source_train": Path(args.source).resolve() / "train.py",
        "frozen_environment": frozen_environment_path(Path(args.source).resolve()),
        "parity_receipt": Path(args.parity_receipt).resolve(),
    }
    seen = {Path(path).resolve() for path in inputs.values()}
    for path in required_parity_sources(Path(args.source).resolve(), Path(args.train_job).resolve(), Path(args.grouped_env_path).resolve()):
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        inputs[parity_source_label(resolved)] = resolved
        seen.add(resolved)
    return inputs


def validate_source_inputs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    records = {name: file_record(path) for name, path in source_inputs(args).items()}
    missing = [name for name, record in records.items() if not record.get("exists")]
    if missing:
        raise RuntimeError(f"missing required source inputs: {', '.join(missing)}")
    return records


def _receipt_int(receipt: dict[str, Any], key: str) -> int:
    try:
        return int(receipt[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"parity receipt {key} must be an integer: {receipt.get(key)!r}") from exc


def _receipt_tests_run(receipt: dict[str, Any]) -> int:
    for key in ("testsRun", "tests_run", "tests", "ran"):
        if key in receipt:
            return _receipt_int(receipt, key)
    raise RuntimeError("parity receipt must include testsRun")


def validate_parity_receipt(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = Path(args.parity_receipt).resolve()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"candidate scaling requires parity receipt before launch: {receipt_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"parity receipt malformed: {receipt_path}") from exc
    if not isinstance(receipt, dict):
        raise RuntimeError("parity receipt must be a JSON object")
    if receipt.get("status") != "passed":
        raise RuntimeError(f"parity receipt status must be 'passed': {receipt.get('status')!r}")
    if _receipt_int(receipt, "exit_code") != 0:
        raise RuntimeError(f"parity receipt exit_code must be 0: {receipt.get('exit_code')!r}")
    for key in ("failures", "errors", "skipped"):
        if _receipt_int(receipt, key) != 0:
            raise RuntimeError(f"parity receipt {key} must be 0: {receipt.get(key)!r}")
    tests_run = _receipt_tests_run(receipt)
    if tests_run != 6:
        raise RuntimeError(f"parity receipt testsRun must be 6: {tests_run}")
    if _receipt_int(receipt, "steps_per_case") < 1000:
        raise RuntimeError(f"parity receipt steps_per_case must be >=1000: {receipt.get('steps_per_case')!r}")
    try:
        group_sizes = {int(value) for value in receipt.get("group_sizes", [])}
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"parity receipt group_sizes must include 2, 4, and 8: {receipt.get('group_sizes')!r}") from exc
    if not {2, 4, 8}.issubset(group_sizes):
        raise RuntimeError(f"parity receipt group_sizes must include 2, 4, and 8: {sorted(group_sizes)}")
    expected_candidate_sha = sha256_file(Path(args.grouped_env_path).resolve())
    actual_candidate_sha = receipt.get("candidate_sha256", receipt.get("runner_sha256"))
    if actual_candidate_sha != expected_candidate_sha:
        raise RuntimeError(
            "parity receipt candidate sha does not match current grouped candidate: "
            f"receipt={actual_candidate_sha!r} actual={expected_candidate_sha!r}"
        )
    candidate_path = receipt.get("candidate_grouped_path")
    if candidate_path and Path(candidate_path).resolve() != Path(args.grouped_env_path).resolve():
        raise RuntimeError(f"parity receipt candidate_grouped_path mismatch: {candidate_path}")
    source_hashes_before = receipt.get("sourcehashes_before")
    source_hashes_after = receipt.get("sourcehashes_after")
    if not isinstance(source_hashes_before, dict) or not source_hashes_before:
        raise RuntimeError("parity receipt must include sourcehashes_before")
    if not isinstance(source_hashes_after, dict) or not source_hashes_after:
        raise RuntimeError("parity receipt must include sourcehashes_after")
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("parity receipt sourcehashes_before/sourcehashes_after differ")
    for path in required_parity_sources(
        Path(args.source).resolve(),
        Path(args.train_job).resolve(),
        Path(args.grouped_env_path).resolve(),
    ):
        resolved = str(Path(path).resolve())
        expected_sha = source_hashes_before.get(resolved)
        if expected_sha is None:
            raise RuntimeError(f"parity receipt missing required source hash: {resolved}")
        actual_sha = sha256_file(Path(path).resolve())
        if expected_sha != actual_sha:
            raise RuntimeError(
                "parity receipt required source hash does not match current file: "
                f"path={resolved} receipt={expected_sha!r} actual={actual_sha!r}"
            )
    source = receipt.get("source") if isinstance(receipt.get("source"), dict) else {}
    receipt_env_sha = source.get("frozen_environment_sha256") or receipt.get("frozen_environment_sha256")
    if receipt_env_sha is not None:
        actual_env_sha = sha256_file(frozen_environment_path(Path(args.source).resolve()))
        if receipt_env_sha != actual_env_sha:
            raise RuntimeError(
                "parity receipt frozen environment sha does not match current source: "
                f"receipt={receipt_env_sha!r} actual={actual_env_sha!r}"
            )
    minimum_set = receipt.get("minimum_set", receipt.get("minimumset"))
    if minimum_set is not None and (not isinstance(minimum_set, list) or not minimum_set):
        raise RuntimeError("parity receipt must include non-empty minimum_set")
    return receipt


def plan_jobs(args: argparse.Namespace) -> list[CandidateJob]:
    return [
        CandidateJob(
            name=f"candidate-scaling-env{N_ENVS}-step{N_STEPS}-group{group_size}",
            group_size=group_size,
            output=Path(args.output) / f"candidate-scaling-env{N_ENVS}-step{N_STEPS}-group{group_size}",
        )
        for group_size in args.group_sizes
    ]


def command_for_job(args: argparse.Namespace, job: CandidateJob) -> list[str]:
    return [
        sys.executable,
        str(Path(args.harness).resolve()),
        "--config",
        str(Path(args.config).resolve()),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--source",
        str(Path(args.source).resolve()),
        "--train-job",
        str(Path(args.train_job).resolve()),
        "--output",
        str(Path(job.output).resolve()),
        "--device",
        str(args.device),
        "--phase",
        str(args.phase),
        "--grouped-env-path",
        str(Path(args.grouped_env_path).resolve()),
        "--n-envs",
        str(job.n_envs),
        "--n-steps",
        str(job.n_steps),
        "--batch-size",
        str(job.batch_size),
        "--epochs",
        str(job.epochs),
        "--seed",
        str(job.seed),
        "--warmup",
        str(job.warmup),
        "--cycles",
        str(job.cycles),
        "--group-size",
        str(job.group_size),
        "--maxseconds",
        str(float(args.per_job_timeout)),
    ]


def wandb_notes(args: argparse.Namespace, job: CandidateJob) -> str:
    return (
        "Goal: candidate grouped scaling short screen only; not acceptance or speedup claim. "
        f"job={job.name}; seed={job.seed}; group_size={job.group_size}; "
        f"n_envs={job.n_envs}; n_steps={job.n_steps}; batch_size={job.batch_size}; epochs={job.epochs}; "
        f"warmup={job.warmup}; measured_cycles={job.cycles}; checkpoint={Path(args.checkpoint).resolve()}"
    )


def whitelisted_env(args: argparse.Namespace, job: CandidateJob) -> dict[str, str]:
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
        "WANDB_RUN_GROUP": "candidate-scaling-20260906",
    }
    return {key: env[key] for key in ENV_WHITELIST}


def copy_snapshot(path: Path, snapshot_dir: Path, name: str) -> None:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved, snapshot_dir / name)


def copy_source_input_snapshots(args: argparse.Namespace, snapshot_dir: Path) -> None:
    """Copy every hashed launch input into a per-job immutable snapshot."""

    snapshot_names = {
        "driver": "campaign_candidate_scaling.py",
        "harness": "benchmark_campaign.py",
        "grouped_env": "candidate_grouped_env.py",
        "train_job": "train_job.py",
        "config": Path(args.config).name,
        "checkpoint": Path(args.checkpoint).name,
        "source_train": "train.py",
        "frozen_environment": "environment.py",
        "parity_receipt": DEFAULT_PARITY_RECEIPT_NAME,
    }
    for key, path in source_inputs(args).items():
        copy_snapshot(path, snapshot_dir, snapshot_names.get(key, safe_snapshot_name(key, path)))


def write_source_manifest(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    records = validate_source_inputs(args)
    write_json_if_new_or_same(Path(args.output) / "source_manifest.json", records)
    return records


def write_job_manifest(
    args: argparse.Namespace,
    job: CandidateJob,
    *,
    command: list[str],
    env: dict[str, str],
    source_records: dict[str, dict[str, Any]],
) -> None:
    snapshot = Path(job.output) / "source_snapshot"
    copy_source_input_snapshots(args, snapshot)
    write_json(
        Path(job.output) / "job_manifest.json",
        {
            "schema": "ai2.candidate_scaling.job.v1",
            "job": dataclasses.asdict(job),
            "command": command,
            "command_sha256": sha256_bytes(json.dumps(command, sort_keys=True).encode("utf-8")),
            "env": env,
            "env_sha256": sha256_bytes(json.dumps(env, sort_keys=True).encode("utf-8")),
            "files": source_records,
            "created_unix": time.time(),
            "log_path": str(Path(job.output) / "job.log"),
        },
    )


def validate_job_summary(job: CandidateJob) -> str | None:
    summary_path = Path(job.output) / "campaign_summary.json"
    if not summary_path.is_file():
        return f"missing summary: {summary_path}"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"malformed summary: {summary_path}"
    if summary.get("status") != "completed":
        return f"summary status was {summary.get('status')!r}"
    if int(summary.get("completed_measured_cycles", -1)) != job.cycles:
        return f"measured cycles expected={job.cycles} completed={summary.get('completed_measured_cycles')}"
    if int(summary.get("requested_measured_cycles", -1)) != job.cycles:
        return f"requested measured cycles expected={job.cycles} actual={summary.get('requested_measured_cycles')}"
    if int(summary.get("warmup_cycles", -1)) != job.warmup:
        return f"warmup cycles expected={job.warmup} actual={summary.get('warmup_cycles')}"
    transitions = (summary.get("plan") or {}).get("transitions_per_cycle")
    if int(transitions) != job.n_envs * job.n_steps:
        return f"transitions per cycle expected={job.n_envs * job.n_steps} actual={transitions}"
    return None


def ensure_no_completed_outputs(jobs: list[CandidateJob]) -> None:
    for job in jobs:
        summary_path = Path(job.output) / "campaign_summary.json"
        if not summary_path.is_file():
            continue
        try:
            status = json.loads(summary_path.read_text(encoding="utf-8")).get("status")
        except json.JSONDecodeError:
            status = None
        if status == "completed":
            raise FileExistsError(f"refusing to overwrite completed output: {job.output}")


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
    error: str | None = None,
) -> None:
    write_json(
        output / "suite_report.json",
        {
            "schema": "ai2.candidate_scaling.suite.v1",
            "status": status,
            "error": error,
            "started_unix": started,
            "ended_unix": ended,
            "deadline_unix": args.deadline_unix,
            "per_job_timeout": args.per_job_timeout,
            "reserve_seconds": RESERVE_SECONDS,
            "planned_jobs": planned_jobs,
            "completed_jobs": sum(1 for record in records if record.get("status") == "completed"),
            "jobs": records,
            "short_screen_note": "screening run only; not an acceptance or speedup claim",
        },
    )


def run_subprocess_job(job: CandidateJob, command: list[str], env: dict[str, str], timeout: float) -> int:
    merged_env = os.environ.copy()
    merged_env.update(env)
    log_path = Path(job.output) / "job.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=merged_env,
            start_new_session=True,
        )
        write_active_job(
            Path(job.output).parent,
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


def run_suite(
    args: argparse.Namespace,
    *,
    runner: Callable[[CandidateJob, list[str], dict[str, str], float], int] = run_subprocess_job,
    clock: Any = time,
) -> int:
    jobs = plan_jobs(args)
    started = float(clock.time())
    records: list[dict[str, Any]] = []
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        source_records = write_source_manifest(args)
        receipt = validate_parity_receipt(args)
        ensure_no_completed_outputs(jobs)
    except Exception as exc:
        write_suite_report(
            args.output,
            status="validation_failed",
            args=args,
            started=started,
            ended=float(clock.time()),
            planned_jobs=len(jobs),
            records=records,
            error=f"{type(exc).__name__}: {exc}",
        )
        return 2

    write_json_if_new_or_same(
        args.output / "scaling_plan.json",
        {
            "schema": "ai2.candidate_scaling.plan.v1",
            "candidate": file_record(args.grouped_env_path),
            "parity_receipt": file_record(args.parity_receipt),
            "parity_receipt_status": receipt.get("status"),
            "jobs": [dataclasses.asdict(job) for job in jobs],
            "dryrun_by_default": True,
            "requires_run_flag": True,
            "short_screen_note": "screening run only; not an acceptance or speedup claim",
        },
    )

    if not args.run:
        write_suite_report(
            args.output,
            status="dryrun",
            args=args,
            started=started,
            ended=float(clock.time()),
            planned_jobs=len(jobs),
            records=[dataclasses.asdict(job) for job in jobs],
        )
        return 0

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
                    "index": index,
                    "status": "skipped_deadline",
                    "remaining_seconds": remaining,
                    "required_seconds": required,
                }
            )
            break
        Path(job.output).mkdir(parents=True, exist_ok=False)
        command = command_for_job(args, job)
        env = whitelisted_env(args, job)
        write_job_manifest(args, job, command=command, env=env, source_records=source_records)
        running_record = {
            "name": job.name,
            "index": index,
            "group_size": job.group_size,
            "output": str(job.output),
            "return_code": None,
            "started_unix": float(clock.time()),
            "ended_unix": None,
            "status": "running",
            "error": None,
        }
        records.append(running_record)
        write_suite_report(
            args.output,
            status="running",
            args=args,
            started=started,
            ended=float(clock.time()),
            planned_jobs=len(jobs),
            records=records,
        )
        return_code = runner(job, command, env, float(args.per_job_timeout))
        error = validate_job_summary(job) if return_code == 0 else f"process exit code {return_code}"
        running_record.update(
            {
                "return_code": return_code,
                "ended_unix": float(clock.time()),
                "status": "completed" if error is None else "failed",
                "error": error,
            }
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
            "ended_unix": float(clock.time()),
        },
    )
    write_suite_report(
        args.output,
        status=status,
        args=args,
        started=started,
        ended=float(clock.time()),
        planned_jobs=len(jobs),
        records=records,
    )
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--run", action="store_true", help="actually launch benchmark jobs; default is dry-run")
    parser.add_argument("--per-job-timeout", type=float, default=PER_JOB_TIMEOUT_SECONDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", default="0b")
    parser.add_argument("--config", type=Path, default=benchmark_campaign.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=benchmark_campaign.DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=benchmark_campaign.DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=benchmark_campaign.DEFAULT_TRAIN_JOB)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--grouped-env-path", type=Path, default=DEFAULT_GROUPED_ENV)
    parser.add_argument("--parity-receipt", type=Path)
    parser.add_argument("--group-sizes", default="4,8")
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.output = Path(args.output).resolve()
    if not args.output.is_absolute():
        raise ValueError("--output must be absolute")
    if args.per_job_timeout <= 0 or args.per_job_timeout > PER_JOB_TIMEOUT_SECONDS:
        raise ValueError("--per-job-timeout must be in (0, 300]")
    if args.device != "cuda":
        raise ValueError("candidate scaling screen is CUDA-only")
    if str(args.phase) != "0b":
        raise ValueError("candidate scaling screen is fixed to phase0b")
    args.group_sizes = parse_group_sizes(args.group_sizes)
    for attr in ("config", "checkpoint", "source", "train_job", "harness", "grouped_env_path"):
        setattr(args, attr, Path(getattr(args, attr)).resolve())
    if args.parity_receipt is None:
        args.parity_receipt = args.output / DEFAULT_PARITY_RECEIPT_NAME
    args.parity_receipt = Path(args.parity_receipt).resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = validate_args(build_parser().parse_args(argv))
        return run_suite(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
