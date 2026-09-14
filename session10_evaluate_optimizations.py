#!/usr/bin/env python3
"""Bounded CUDA held-out evaluation of explicitly named session 10 checkpoints.

Adapted from session9_evaluate_native_lanes.py. Dry-run is the default. The
frozen evaluator runs original Python games; policy evaluation must attest CUDA.
This helper performs no training and only reads checkpoint ZIP JSON metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Callable
import zipfile

FROZEN_EVALUATOR_SHA256 = "df82280ca48cdc6bd804ae003aedfcc581350c7e425aa4291ac9849b3d14a005"
MAX_SECONDS, RESERVE_SECONDS, EPISODES = 780, 10, 20
PANELS = {"0b": 93001, "0a": 94001}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/AI_2/native-ppo-session10"))
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--per-job-timeout", type=float, default=90)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    args.root, args.output = args.root.resolve(), args.output.resolve()
    args.source, args.evaluator = args.root / "variants/corrected", args.root / "scripts/evaluate_modes.py"
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= 0:
        parser.error("--deadline-unix must be positive and finite")
    if not math.isfinite(args.per_job_timeout) or not 0 < args.per_job_timeout <= 120:
        parser.error("--per-job-timeout must be positive, finite, and <=120")
    args.checkpoints = {}
    for item in args.checkpoint:
        label, separator, path = item.partition("=")
        if not separator or not path or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None:
            parser.error("checkpoints must use LABEL=PATH with a simple alphanumeric/dash/underscore label")
        if label in args.checkpoints:
            parser.error(f"duplicate checkpoint label: {label}")
        args.checkpoints[label] = Path(path).resolve()
    if len([label for label in args.checkpoints if label != "initial"]) < 2:
        parser.error("provide at least two named final checkpoints; optional initial is exempt from equal-step comparison")
    return args


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def plan_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = []
    for label, checkpoint in args.checkpoints.items():
        with zipfile.ZipFile(checkpoint) as archive:
            data = json.loads(archive.read("data"))
        steps, updates = data.get("num_timesteps"), data.get("_n_updates")
        if type(steps) is not int or steps < 0 or type(updates) is not int or updates < 0:
            raise ValueError(f"checkpoint requires nonnegative integer step/update metadata: {label}")
        jobs.append({"label": label, "checkpoint": checkpoint, "expected_steps": steps,
                     "expected_updates": updates, "output": args.output / label, "log": args.output / f"{label}.log"})
    finals = {(job["expected_steps"], job["expected_updates"]) for job in jobs if job["label"] != "initial"}
    if len(finals) != 1:
        raise ValueError("compared final checkpoints must have equal num_timesteps and _n_updates")
    final_steps, final_updates = next(iter(finals))
    for job in jobs:
        if job["label"] == "initial" and (job["expected_steps"] > final_steps or job["expected_updates"] > final_updates):
            raise ValueError("initial checkpoint cannot be later than compared finals")
    return jobs


def command(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    return [sys.executable, str(args.evaluator), "--repo", str(args.source),
            "--checkpoint", str(job["checkpoint"]), "--output", str(job["output"]),
            "--device", "cuda", "--episodes", str(EPISODES),
            "--wide-seed", str(PANELS["0b"]), "--near-seed", str(PANELS["0a"])]


def input_hashes(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> dict[str, str]:
    if sha256(args.evaluator) != FROZEN_EVALUATOR_SHA256:
        raise ValueError("frozen evaluate_modes.py SHA256 mismatch")
    paths = [Path(__file__).resolve(), args.evaluator, args.source / "config.yaml", args.source / "control.yaml",
             args.source / "train.py", args.source / "learning_eval.py",
             args.source / "UTMIST-AI2-main/environment/environment.py", *(job["checkpoint"] for job in jobs)]
    return {str(path): sha256(path) for path in paths}


def verify_hashes(hashes: dict[str, str]) -> None:
    for path, expected in hashes.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"evaluation input changed: {path}")


def require_idle_gpu() -> None:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=False)
    if result.returncode != 0 or result.stdout.strip():
        raise RuntimeError(f"GPU must be idle and verified: {result.stdout.strip()} {result.stderr.strip()}")


def run_subprocess(job: dict[str, Any], argv: list[str], timeout: float) -> int:
    env = os.environ.copy()
    env.update({"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "WANDB_MODE": "disabled"})
    with job["log"].open("xb") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            return 124


def validate_evaluation(args: argparse.Namespace, job: dict[str, Any], checkpoint_hash: str, config_hash: str) -> dict[str, Any]:
    metadata = read_json(job["output"] / "metadata.json")
    expected = {"checkpoint": str(job["checkpoint"]), "checkpoint_sha256": checkpoint_hash,
                "source": str(args.source), "config_sha256": config_hash,
                "cumulative_steps": job["expected_steps"], "requested_device": "cuda"}
    if metadata.get("training_performed") is not False or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("evaluation checkpoint/config/source/steps/training attestation mismatch")
    devices = [metadata.get(key) for key in ("actual_model_device", "actual_policy_device", "device")]
    if any(not isinstance(device, str) or re.fullmatch(r"cuda(?::[0-9]+)?", device) is None for device in devices):
        raise ValueError("evaluation requires actual CUDA model and policy device attestations")
    # SB3 retains the unindexed model device ('cuda'), while policy parameters
    # report its resolved ordinal ('cuda:0'). Only explicit ordinals can conflict.
    explicit_devices = {device for device in devices if device != "cuda"}
    if len(explicit_devices) > 1:
        raise ValueError("evaluation model and policy device attestations disagree")
    summaries = []
    for phase, seed_start in PANELS.items():
        for mode in ("deterministic", "sampled"):
            shard = read_json(job["output"] / f"{phase}-{mode}.json")
            if shard.get("phase_key") != phase or shard.get("deterministic") is not (mode == "deterministic"):
                raise ValueError(f"evaluation phase/mode mismatch: {phase}-{mode}")
            if shard.get("observation_shape") != [256]:
                raise ValueError("evaluation observation shape must be [256]")
            episodes = shard.get("episodes", [])
            if [episode.get("seed") for episode in episodes] != list(range(seed_start, seed_start + EPISODES)):
                raise ValueError(f"evaluation episode count/seeds mismatch: {phase}-{mode}")
            summary = shard.get("summary", {})
            if type(summary.get("episodes")) is not int or summary["episodes"] != EPISODES:
                raise ValueError(f"evaluation summary count mismatch: {phase}-{mode}")
            for key in ("success_rate", "fall_rate", "nav_falls", "navigation_targets_per_episode", "timeout_rate"):
                value = summary.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"evaluation summary invalid {key}: {phase}-{mode}")
                if key.endswith("_rate") and value > 1:
                    raise ValueError(f"evaluation summary rate exceeds one: {key}")
            summaries.append({"phase": phase, "mode": mode, "summary": summary})
    if read_json(job["output"] / "summary.json") != summaries:
        raise ValueError("evaluation aggregate summary does not match four completed shards")
    return {"metadata": str(job["output"] / "metadata.json"), "summaries": summaries,
            "verified_episodes": EPISODES * len(PANELS) * 2}


def run(args: argparse.Namespace, *, runner: Callable = run_subprocess,
        clock: Any = time, idle_check: Callable = require_idle_gpu) -> int:
    args.output.mkdir(parents=True, exist_ok=False)
    started = clock.time()
    deadline = min(args.deadline_unix, started + MAX_SECONDS)
    records, jobs = [], []

    def report(status: str, error: str | None = None) -> None:
        write_json(args.output / "evaluation_report.json", {
            "schema": "ai2.session10.optimizations.evaluation.v1", "status": status, "error": error,
            "started_unix": started, "ended_unix": clock.time(), "duration_seconds": clock.time() - started,
            "deadline_unix": args.deadline_unix, "effective_deadline_unix": deadline, "maximum_seconds": MAX_SECONDS,
            "planned_jobs": len(args.checkpoints), "completed_jobs": sum(row["status"] == "completed" for row in records),
            "expected_episodes_per_checkpoint": 80, "seeds": PANELS, "jobs": records,
            "scope": "held-out frozen Python navigation with CUDA policy; equal final training counts; no training",
            "evaluation_inputs": str(args.output / "evaluation_inputs.json"),
        })

    try:
        jobs = plan_jobs(args)
        hashes = input_hashes(args, jobs)
        write_json(args.output / "evaluation_inputs.json", {"sha256": hashes, "frozen_evaluator_sha256": FROZEN_EVALUATOR_SHA256})
        write_json(args.output / "evaluation_plan.json", {
            "jobs": [{**job, "command": command(args, job)} for job in jobs],
            "dry_run": not args.run, "effective_deadline_unix": deadline})
    except Exception as exc:
        report("validation_failed", f"{type(exc).__name__}: {exc}")
        return 2
    if not args.run:
        report("dry_run")
        return 0

    def skip_remaining(index: int, status: str) -> None:
        records.extend({"label": job["label"], "status": status, "command": command(args, job)} for job in jobs[index:])

    for index, job in enumerate(jobs):
        if deadline - clock.time() < args.per_job_timeout + RESERVE_SECONDS:
            skip_remaining(index, "skipped_deadline")
            report("deadline_insufficient")
            return 3
        try:
            verify_hashes(hashes)
            idle_check()
        except Exception as exc:
            skip_remaining(index, "not_run_validation_failed")
            report("validation_failed", f"{type(exc).__name__}: {exc}")
            return 2
        # Hash reads and the idle probe consume deadline budget too.
        if deadline - clock.time() < args.per_job_timeout + RESERVE_SECONDS:
            skip_remaining(index, "skipped_deadline")
            report("deadline_insufficient")
            return 3
        record = {**job, "status": "running", "command": command(args, job), "started_unix": clock.time(),
                  "checkpoint_sha256": hashes[str(job["checkpoint"])], "return_code": None, "error": None}
        records.append(record)
        report("running")
        rc = 1
        try:
            rc = runner(job, record["command"], args.per_job_timeout)
            if rc != 0:
                raise RuntimeError(f"evaluation process exit code {rc}")
            if clock.time() - record["started_unix"] >= args.per_job_timeout:
                rc = 124
                raise RuntimeError("evaluation process exceeded timeout")
            record["evidence"] = validate_evaluation(args, job, record["checkpoint_sha256"], hashes[str(args.source / "config.yaml")])
            verify_hashes(hashes)
            record["status"] = "completed"
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record.update(return_code=rc, ended_unix=clock.time(), duration_seconds=clock.time() - record["started_unix"])
        if clock.time() >= deadline:
            record.update(status="failed", error="evaluation finished after deadline")
            skip_remaining(index + 1, "skipped_deadline")
            report("deadline_exceeded")
            return 3
        if record["status"] != "completed":
            skip_remaining(index + 1, "not_run_after_failure")
            report("failed", record["error"])
            return 1
        report("running")
    report("completed")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        code = run(args)
        print(f"evaluation_report={args.output / 'evaluation_report.json'}", flush=True)
        return code
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
