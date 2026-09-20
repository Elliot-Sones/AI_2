#!/usr/bin/env python3
"""Run one remote AI_2 GPU campaign job with a lock, logs and receipts."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


RESERVE_SECONDS = 120.0
SCHEMA_VERSION = "ai2.campaign_job.v1"
SECRET_ENV_KEYS = (
    "VASTAI_API_KEY",
    "WANDB_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "KAGGLE_API_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)


def now() -> float:
    return time.time()


def atomic_json(path: Path | str, value: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def validate_name(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name or Path(name).name != name:
        raise ValueError("job name must be a single path-safe filename component")
    return name


def job_env(root: Path) -> dict[str, str]:
    root = root.resolve()
    source = root / "source"
    inputs = root / "inputs"
    env = dict(os.environ)
    for key in SECRET_ENV_KEYS:
        env.pop(key, None)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
        WANDB_MODE="offline",
        AI2_FROZEN_SOURCE_ROOT=str((inputs / "frozen").resolve()),
        AI2_CONTACT_REFERENCE=str((inputs / "reference" / "combat_curriculum.py").resolve()),
        AI2_CONTACT_CHECKPOINT=str((inputs / "calibration" / "final.zip").resolve()),
    )
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(source) if not existing else str(source) + os.pathsep + existing
    return env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("name")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command required after --")
    try:
        validate_name(args.name)
    except ValueError as exc:
        parser.error(str(exc))
    args.root = args.root.resolve()
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    root = args.root
    source = root / "source"
    jobs = root / "jobs"
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")
    jobs.mkdir(parents=True, exist_ok=True)
    log_path = jobs / f"{args.name}.log"
    receipt_path = jobs / f"{args.name}.json"
    started_path = jobs / f"{args.name}.started.json"
    lock_path = jobs / "gpu-job.lock"
    if log_path.exists() or receipt_path.exists() or started_path.exists():
        raise FileExistsError(f"job output already exists for {args.name}")
    timeout = float(args.deadline_unix) - now() - RESERVE_SECONDS
    if timeout <= 0:
        raise TimeoutError("deadline does not leave the required reserve before launch")

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("GPU job lock is already held") from exc
        start = now()
        proc: subprocess.Popen[str] | None = None
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "name": args.name,
            "argv": list(args.command),
            "root": str(root),
            "cwd": str(source),
            "log": str(log_path.resolve()),
            "start_unix": start,
            "deadline_unix": float(args.deadline_unix),
            "reserve_seconds": RESERVE_SECONDS,
        }
        try:
            with log_path.open("x", encoding="utf8") as log:
                proc = subprocess.Popen(
                    args.command,
                    cwd=source,
                    env=job_env(root),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    text=True,
                )
                receipt["pid"] = int(proc.pid)
                atomic_json(started_path, receipt)
                try:
                    returncode = proc.wait(timeout=timeout)
                    timed_out = False
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        returncode = proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        returncode = proc.wait(timeout=15)
                    receipt.update(end_unix=now(), returncode="timeout", timed_out=True)
                    atomic_json(receipt_path, receipt)
                    raise TimeoutError("job exceeded deadline reserve") from exc
            receipt.update(end_unix=now(), returncode=int(returncode), timed_out=timed_out)
            atomic_json(receipt_path, receipt)
            return receipt
        except BaseException:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=15)
            raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    result = main(sys.argv[1:])
    print(json.dumps({"name": result["name"], "returncode": result["returncode"]}, sort_keys=True))
    raise SystemExit(0 if result["returncode"] == 0 else int(result["returncode"]))
