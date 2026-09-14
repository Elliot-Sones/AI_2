#!/usr/bin/env python3
"""Matched-pairs launcher for original grouped env vs native candidate grouped env.

Dry-run is the default. Use ``--run`` only after a parity summary has been
validated and the output directory is fresh.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import benchmark_campaign
import campaign_suite
import validate_grouped_candidate


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HARNESS = REPO_ROOT / "benchmark_campaign.py"
DEFAULT_BASELINE_GROUPED = REPO_ROOT / "grouped_vec_env.py"
MAX_PER_JOB_TIMEOUT = 600.0
RESERVE_SECONDS = 300.0
DEFAULT_PHASE = "0b"
DEFAULT_MAXSECONDS = 600.0
SEEDS = (45, 46, 47)
NATIVE_FLAG_ENV = ("AI2_NATIVE_INPUT", "AI2_NATIVE_OBSERVATION")
PAIR_SHAPE = {
    "n_envs": 256,
    "n_steps": 128,
    "batch_size": 8192,
    "epochs": 8,
    "warmup": 2,
    "cycles": 100,
    "group_size": 4,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(campaign_suite.json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def frozen_environment_path(args: argparse.Namespace) -> Path:
    return Path(args.source) / "UTMIST-AI2-main" / "environment" / "environment.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--candidate-grouped-path", type=Path, required=True)
    parser.add_argument("--parity-summary", type=Path, required=True)
    parser.add_argument("--candidate-extra-arg", action="append", default=[])
    parser.add_argument("--candidate-env", action="append", default=[])
    parser.add_argument("--native-input", action="store_true")
    parser.add_argument("--native-observation", action="store_true")
    parser.add_argument("--run", action="store_true", help="execute jobs; default is dry-run only")
    parser.add_argument("--per-job-timeout", type=float, default=MAX_PER_JOB_TIMEOUT)
    parser.add_argument("--maxseconds", type=float, default=DEFAULT_MAXSECONDS)
    parser.add_argument("--phase", default=DEFAULT_PHASE)
    parser.add_argument("--source", type=Path, default=benchmark_campaign.DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=benchmark_campaign.DEFAULT_TRAIN_JOB)
    parser.add_argument("--config", type=Path, default=benchmark_campaign.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=benchmark_campaign.DEFAULT_CHECKPOINT)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--baseline-grouped-path", type=Path, default=DEFAULT_BASELINE_GROUPED)
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if not Path(args.output).is_absolute():
        args.output = Path(args.output).resolve()
    for attr in (
        "candidate_grouped_path",
        "parity_summary",
        "source",
        "train_job",
        "config",
        "checkpoint",
        "harness",
        "baseline_grouped_path",
    ):
        setattr(args, attr, Path(getattr(args, attr)).resolve())
    if args.per_job_timeout <= 0 or args.per_job_timeout > MAX_PER_JOB_TIMEOUT:
        raise ValueError("--per-job-timeout must be >0 and <=600")
    if not hasattr(args, "maxseconds"):
        args.maxseconds = DEFAULT_MAXSECONDS
    if not hasattr(args, "phase"):
        args.phase = DEFAULT_PHASE
    if args.maxseconds <= 0 or args.maxseconds > MAX_PER_JOB_TIMEOUT:
        raise ValueError("--maxseconds must be >0 and <=600")
    _validate_candidate_extra_args(args.candidate_extra_arg or [])
    _validate_candidate_env_args(args.candidate_env or [])
    return args


def _validate_candidate_extra_args(extra_args: list[str]) -> None:
    if extra_args:
        raise ValueError(
            "candidate-extra-arg is disabled for this bounded matched-pairs runner; "
            "use a new reviewed runner for any workload or behavior change"
        )


def _validate_candidate_env_args(candidate_env: list[str]) -> None:
    if candidate_env:
        raise ValueError(
            "candidate-env is disabled for this bounded matched-pairs runner; "
            "use --native-input/--native-observation for the only allowed candidate runtime flags"
        )


def expected_native_runtime_env(args: argparse.Namespace) -> dict[str, str]:
    return {
        "AI2_NATIVE_INPUT": "1" if getattr(args, "native_input", False) else "0",
        "AI2_NATIVE_OBSERVATION": "1" if getattr(args, "native_observation", False) else "0",
    }


def parse_candidate_env(args: argparse.Namespace) -> dict[str, str]:
    _validate_candidate_env_args(args.candidate_env or [])
    return expected_native_runtime_env(args)


def plan_jobs(args: argparse.Namespace) -> list[campaign_suite.SuiteJob]:
    jobs: list[campaign_suite.SuiteJob] = []
    for index, seed in enumerate(SEEDS):
        arms = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        for arm in arms:
            jobs.append(
                campaign_suite.SuiteJob(
                    name=f"candidate-pairs-seed{seed}-{arm}",
                    arm=arm,
                    seed=seed,
                    output=Path(args.output) / f"candidate-pairs-seed{seed}-{arm}",
                    **PAIR_SHAPE,
                )
            )
    return jobs


def command_for_job(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> list[str]:
    grouped_path = args.candidate_grouped_path if job.arm == "candidate" else args.baseline_grouped_path
    command = [
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
        "cuda",
        "--phase",
        str(getattr(args, "phase", DEFAULT_PHASE)),
        "--maxseconds",
        str(getattr(args, "maxseconds", DEFAULT_MAXSECONDS)),
        "--grouped-env-path",
        str(grouped_path),
        *job.command_args,
    ]
    return command


def env_for_job(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> dict[str, str]:
    env = campaign_suite.whitelisted_env(job=job, args=_suite_env_args(args))
    if job.arm == "candidate":
        env.update(parse_candidate_env(args))
    else:
        env.update({key: "0" for key in NATIVE_FLAG_ENV})
    return env


def _suite_env_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=args.checkpoint,
        mode="candidate-pairs",
    )


def _manifest_args(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> argparse.Namespace:
    return argparse.Namespace(
        mode="candidate-pairs",
        harness=args.harness,
        rollout_cache_path=Path(args.source) / "rollout_cache.py",
        grouped_env_path=args.candidate_grouped_path if job.arm == "candidate" else args.baseline_grouped_path,
        train_job=args.train_job,
        config=args.config,
        checkpoint=args.checkpoint,
    )


def _source_record(label: str, path: Path, *, copy_to: Path | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    record = {
        "label": label,
        "path": str(resolved),
        "exists": resolved.is_file(),
    }
    if resolved.is_file():
        record.update({"size": resolved.stat().st_size, "sha256": sha256_file(resolved)})
        if copy_to is not None:
            copy_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, copy_to)
    return record


def required_parity_sources(args: argparse.Namespace) -> list[Path]:
    return validate_grouped_candidate.required_sources(
        Path(args.source),
        Path(args.train_job),
        Path(args.candidate_grouped_path),
    )


def minimum_parity_source_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {str(Path(path).resolve()): sha256_file(Path(path)) for path in required_parity_sources(args)}


def _label_for_source(args: argparse.Namespace, path: Path) -> str:
    resolved = Path(path).resolve()
    source = Path(args.source).resolve()
    label_overrides = {
        Path(args.candidate_grouped_path).resolve(): "candidate_grouped_path",
        Path(args.train_job).resolve(): "train_job",
        Path(args.config).resolve(): "config",
        Path(args.checkpoint).resolve(): "checkpoint",
        Path(args.harness).resolve(): "harness",
        Path(args.baseline_grouped_path).resolve(): "baseline_grouped_path",
        Path(args.parity_summary).resolve(): "parity_summary",
        Path(__file__).resolve(): "candidate_pairs_runner",
        (REPO_ROOT / "campaign_suite.py").resolve(): "campaign_suite.py",
        (REPO_ROOT / "validate_grouped_candidate.py").resolve(): "validate_grouped_candidate.py",
        (REPO_ROOT / "batched_rng_vec_env.py").resolve(): "batched_rng_vec_env.py",
        frozen_environment_path(args).resolve(): "frozen_environment",
        (source / "train.py").resolve(): "source/train.py",
    }
    if resolved in label_overrides:
        return label_overrides[resolved]
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def source_files(args: argparse.Namespace) -> list[tuple[str, Path]]:
    files = [
        ("candidate_pairs_runner", Path(__file__)),
        ("train_job", Path(args.train_job)),
        ("config", Path(args.config)),
        ("checkpoint", Path(args.checkpoint)),
        ("harness", Path(args.harness)),
        ("baseline_grouped_path", Path(args.baseline_grouped_path)),
        ("parity_summary", Path(args.parity_summary)),
    ]
    files.extend((_label_for_source(args, path), path) for path in required_parity_sources(args))
    files.append(("campaign_suite.py", REPO_ROOT / "campaign_suite.py"))

    deduped: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, path in files:
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append((label, resolved))
    return deduped


def snapshot_sources(args: argparse.Namespace, *, archive: bool) -> dict[str, str]:
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    archive_dir = Path(args.output) / "source_archive"
    for label, path in source_files(args):
        copy_to = None
        if archive and label != "checkpoint":
            safe_name = label.replace("/", "__")
            suffix = Path(path).suffix
            copy_to = archive_dir / f"{safe_name}{suffix}"
        record = _source_record(label, path, copy_to=copy_to)
        if not record["exists"]:
            raise FileNotFoundError(f"required source missing for {label}: {Path(path).resolve()}")
        records.append(record)
        if record["exists"]:
            hashes[record["path"]] = record["sha256"]
    if archive:
        write_json(archive_dir / "source_manifest.json", {"created_unix": time.time(), "files": records})
    return hashes


def ensure_sources_unchanged(expected: dict[str, str]) -> None:
    for raw_path, expected_hash in expected.items():
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"source changed before job: missing {path}")
        actual = sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(f"source changed before job: {path} expected={expected_hash} actual={actual}")


def validate_parity_summary(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.parity_summary)
    if not path.is_file():
        raise RuntimeError(f"missing parity summary: {path}")
    marker = json.loads(path.read_text(encoding="utf-8"))
    if int(marker.get("exit_code", -1)) != 0:
        raise RuntimeError(f"parity summary exit_code must be 0: {marker.get('exit_code')!r}")
    if int(marker.get("testsRun", 0)) != 6:
        raise RuntimeError(f"parity summary testsRun must be 6: {marker.get('testsRun')!r}")
    for key in ("failures", "errors", "skipped"):
        if int(marker.get(key, -1)) != 0:
            raise RuntimeError(f"parity summary {key} must be 0: {marker.get(key)!r}")
    if int(marker.get("steps_per_case", 0)) < 1000:
        raise RuntimeError(f"parity summary steps_per_case must be >=1000: {marker.get('steps_per_case')!r}")
    group_sizes = marker.get("group_sizes")
    try:
        actual_group_sizes = {int(size) for size in group_sizes}
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"parity summary group_sizes must include 2, 4, and 8: {group_sizes!r}") from exc
    if not {2, 4, 8}.issubset(actual_group_sizes):
        raise RuntimeError(f"parity summary group_sizes must include 2, 4, and 8: {group_sizes!r}")
    candidate_path = Path(marker.get("candidate_grouped_path", "")).resolve()
    if candidate_path != Path(args.candidate_grouped_path).resolve():
        raise RuntimeError(f"parity summary candidate_grouped_path mismatch: {candidate_path}")
    actual_runner_sha = sha256_file(args.candidate_grouped_path)
    if marker.get("runner_sha256") != actual_runner_sha:
        raise RuntimeError(
            f"parity summary runner_sha256 mismatch: marker={marker.get('runner_sha256')!r} actual={actual_runner_sha!r}"
        )
    actual_frozen_sha = sha256_file(frozen_environment_path(args))
    if marker.get("frozen_environment_sha256") != actual_frozen_sha:
        raise RuntimeError(
            "parity summary frozen_environment_sha256 mismatch: "
            f"marker={marker.get('frozen_environment_sha256')!r} actual={actual_frozen_sha!r}"
        )
    before = marker.get("sourcehashes_before") or {}
    after = marker.get("sourcehashes_after") or {}
    if not isinstance(before, dict) or not isinstance(after, dict) or not before:
        raise RuntimeError("parity summary must include non-empty sourcehashes_before/sourcehashes_after")
    if before != after:
        raise RuntimeError("parity summary sourcehashes_before/sourcehashes_after differ")
    required_hashes = minimum_parity_source_hashes(args)
    missing_required = sorted(set(required_hashes) - set(after))
    if missing_required:
        raise RuntimeError(f"missing required parity source hashes: {missing_required}")
    for raw_path, required_hash in required_hashes.items():
        if before.get(raw_path) != required_hash:
            raise RuntimeError(
                f"parity summary sourcehashes_before stale for {raw_path}: "
                f"marker={before.get(raw_path)!r} actual={required_hash!r}"
            )
        if after.get(raw_path) != required_hash:
            raise RuntimeError(
                f"parity summary sourcehashes_after stale for {raw_path}: "
                f"marker={after.get(raw_path)!r} actual={required_hash!r}"
            )
    for raw_path, expected_hash in after.items():
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"parity summary sourcehashes_after current file missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"parity summary sourcehashes_after stale for {path}: marker={expected_hash!r} actual={actual_hash!r}"
            )
    runtime_env = marker.get("runtime_env") or {}
    if not isinstance(runtime_env, dict):
        raise RuntimeError("parity summary runtime_env must be an object")
    expected_runtime = expected_native_runtime_env(args)
    for key, expected_value in expected_runtime.items():
        if str(runtime_env.get(key)) != expected_value:
            raise RuntimeError(
                f"parity summary runtime_env {key} mismatch: marker={runtime_env.get(key)!r} expected={expected_value!r}"
            )
    return marker


def ensure_output_fresh(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")


def write_dryrun_plan(args: argparse.Namespace, jobs: list[campaign_suite.SuiteJob], source_hashes: dict[str, str]) -> None:
    write_json(
        Path(args.output) / "dryrun_plan.json",
        {
            "status": "dry_run",
            "cuda_only": True,
            "per_job_timeout": args.per_job_timeout,
            "reserve_seconds": RESERVE_SECONDS,
            "candidate_env": parse_candidate_env(args),
            "candidate_extra_arg": args.candidate_extra_arg,
            "source_hashes": source_hashes,
            "jobs": [dataclasses.asdict(job) for job in jobs],
        },
    )


def run_suite(
    args: argparse.Namespace,
    *,
    runner: Callable[[campaign_suite.SuiteJob, list[str], dict[str, str], float], int] = campaign_suite.run_subprocess_job,
    clock: Any = time,
) -> int:
    args = validate_args(args)
    jobs = plan_jobs(args)
    ensure_output_fresh(Path(args.output))
    validate_parity_summary(args)
    Path(args.output).mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot_sources(args, archive=True)
    if not args.run:
        write_dryrun_plan(args, jobs, source_hashes)
        campaign_suite.write_suite_report(
            Path(args.output),
            status="dry_run",
            args=argparse.Namespace(mode="candidate-pairs", deadline_unix=args.deadline_unix, per_job_timeout=args.per_job_timeout),
            started=clock.time(),
            ended=clock.time(),
            planned_jobs=len(jobs),
            records=[],
        )
        return 0

    records: list[dict[str, Any]] = []
    started = clock.time()
    for index, job in enumerate(jobs):
        try:
            ensure_sources_unchanged(source_hashes)
        except RuntimeError as exc:
            records.append({"name": job.name, "status": "skipped_source_changed", "error": str(exc), "index": index})
            campaign_suite.write_suite_report(
                Path(args.output),
                status="source_changed",
                args=argparse.Namespace(mode="candidate-pairs", deadline_unix=args.deadline_unix, per_job_timeout=args.per_job_timeout),
                started=started,
                ended=clock.time(),
                planned_jobs=len(jobs),
                records=records,
            )
            return 4
        remaining = float(args.deadline_unix) - float(clock.time())
        required = float(args.per_job_timeout) + RESERVE_SECONDS
        if remaining < required:
            records.append({"name": job.name, "status": "skipped_deadline", "remaining_seconds": remaining, "required_seconds": required})
            campaign_suite.write_suite_report(
                Path(args.output),
                status="deadline_insufficient",
                args=argparse.Namespace(mode="candidate-pairs", deadline_unix=args.deadline_unix, per_job_timeout=args.per_job_timeout),
                started=started,
                ended=clock.time(),
                planned_jobs=len(jobs),
                records=records,
            )
            return 3
        job.output.mkdir(parents=True, exist_ok=False)
        command = command_for_job(args, job)
        env = env_for_job(args, job)
        campaign_suite.write_job_manifest(_manifest_args(args, job), job, command, env)
        rc = runner(job, command, env, float(args.per_job_timeout))
        error = campaign_suite.validate_job_summary(job) if rc == 0 else f"process exit code {rc}"
        records.append(
            {
                "name": job.name,
                "index": index,
                "arm": job.arm,
                "seed": job.seed,
                "output": str(job.output),
                "return_code": rc,
                "status": "completed" if error is None else "failed",
                "error": error,
            }
        )
        if error is not None:
            campaign_suite.write_suite_report(
                Path(args.output),
                status="aborted",
                args=argparse.Namespace(mode="candidate-pairs", deadline_unix=args.deadline_unix, per_job_timeout=args.per_job_timeout),
                started=started,
                ended=clock.time(),
                planned_jobs=len(jobs),
                records=records,
            )
            return 2 if rc == 0 else 1
    campaign_suite.write_suite_report(
        Path(args.output),
        status="completed",
        args=argparse.Namespace(mode="candidate-pairs", deadline_unix=args.deadline_unix, per_job_timeout=args.per_job_timeout),
        started=started,
        ended=clock.time(),
        planned_jobs=len(jobs),
        records=records,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = validate_args(parser.parse_args(argv))
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
