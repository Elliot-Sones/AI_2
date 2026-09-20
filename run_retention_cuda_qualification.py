#!/usr/bin/env python3
"""CUDA qualification gate for the native retention curriculum.

This command is intentionally a gate, not a trainer. It proves the navigation
and contact CUDA contracts, runs one alternating navigation/contact smoke
update through the planned retention block helper, records GPU samples when
available, and seals an immutable local JSON receipt. It never falls back to CPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import math
import re
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
SCHEMA = "ai2.native_retention_cuda_qualification.v1"
NAVIGATION_PHASE = "0b"
PRODUCTION_GATE_GEOMETRY = {"lanes": 4096, "n_steps": 512, "batch_size": 16384, "n_epochs": 4}
SMOKE_SCHEDULE = ("navigation", "combat")
FROZEN_SOURCE_REQUIRED = (
    "train.py",
    "control.yaml",
    "positions.json",
    "UTMIST-AI2-main/environment/environment.py",
)
LOCAL_SOURCE_CANDIDATES = (
    "run_retention_cuda_qualification.py",
    "native_curriculum_ppo.py",
    "native_navigation_curriculum_env.py",
    "train_retention_campaign_block.py",
    "qualify_navigation_curriculum.py",
    "tests/test_contact_action_update_parity.py",
    "tests/test_native_combat_update_parity.py",
    "tests/test_retention_cuda_qualification.py",
)


class QualificationError(RuntimeError):
    """Raised when a required CUDA qualification gate cannot pass."""


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path | str, value: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def source_manifest(source: Path) -> dict[str, str]:
    source = Path(source)
    missing = [name for name in FROZEN_SOURCE_REQUIRED if not (source / name).is_file()]
    if missing:
        raise QualificationError(f"frozen source is missing required files: {missing}")
    manifest: dict[str, str] = {}
    for name in FROZEN_SOURCE_REQUIRED:
        manifest[f"frozen/{name}"] = sha256(source / name)
    for name in LOCAL_SOURCE_CANDIDATES:
        path = ROOT / name
        if path.is_file():
            manifest[f"local/{name}"] = sha256(path)
    return manifest


def artifact_hashes(output_dir: Path, *, exclude: set[str] | None = None) -> dict[str, dict[str, Any]]:
    exclude = set() if exclude is None else set(exclude)
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        rel = path.relative_to(output_dir).as_posix()
        if rel in exclude:
            continue
        rows[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return rows


def write_inventory(output_dir: Path, *, block_id: str) -> dict[str, Any]:
    files = []
    for rel, row in artifact_hashes(output_dir, exclude={"inventory.json"}).items():
        files.append({"path": rel, **row})
    inventory = {
        "schema": "ai2.native_retention_cuda_qualification.inventory.v1",
        "block_id": block_id,
        "files": files,
    }
    atomic_json(output_dir / "inventory.json", inventory)
    return inventory


def require_planned_interfaces(import_module: Callable[[str], Any] = importlib.import_module) -> None:
    ppo = import_module("native_curriculum_ppo")
    model = getattr(ppo, "NativeCurriculumPPO", None)
    if model is None or not callable(getattr(model, "set_task_environment", None)):
        raise QualificationError("NativeCurriculumPPO.set_task_environment is required for alternating retention")
    retention = import_module("train_retention_campaign_block")
    if not callable(getattr(retention, "run_qualification_smoke", None)):
        raise QualificationError("train_retention_campaign_block.run_qualification_smoke is required")


def run_navigation_phase0b(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    import_module: Callable[[str], Any],
) -> dict[str, Any]:
    module = import_module("qualify_navigation_curriculum")
    receipt_path = output_dir / "navigation-phase0b-parity.json"
    code = int(module.main([
        "--source", str(args.source),
        "--config", str(args.config),
        "--phase", NAVIGATION_PHASE,
        "--build-dir", str(args.build_dir / "navigation-phase0b"),
        "--output", str(receipt_path),
        "--device", "cuda",
    ]))
    if not receipt_path.is_file():
        raise QualificationError("navigation phase-0b parity did not write a receipt")
    receipt = json.loads(receipt_path.read_text())
    if code != 0 or receipt.get("status") != "passed" or receipt.get("cuda_qualified") is not True:
        raise QualificationError(f"navigation phase-0b CUDA parity failed: {receipt.get('error', receipt)}")
    validate_navigation_phase0b_receipt(receipt)
    return {
        "status": "passed",
        "phase": NAVIGATION_PHASE,
        "device": "cuda",
        "receipt": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "python_cases": len(receipt.get("python_cases", [])),
        "cuda_cases": len(receipt.get("cuda_cases", [])),
    }


def validate_navigation_phase0b_receipt(receipt: dict[str, Any]) -> None:
    python_cases = receipt.get("python_cases")
    cuda_cases = receipt.get("cuda_cases")
    if not isinstance(python_cases, list) or len(python_cases) != 2:
        raise QualificationError("navigation phase-0b parity must record exactly 2 Python cases")
    if not isinstance(cuda_cases, list) or len(cuda_cases) != 4:
        raise QualificationError("navigation phase-0b parity must record exactly 4 CUDA cases")
    for family, cases in (("Python", python_cases), ("CUDA", cuda_cases)):
        for index, case in enumerate(cases):
            if case.get("status") != "passed":
                raise QualificationError(f"navigation {family} case {index} was {case.get('status')}")
            ticks = case.get("ticks")
            if not isinstance(ticks, int) or ticks <= 0:
                raise QualificationError(f"navigation {family} case {index} did not record positive ticks")


def run_unittest_gate(
    module: str,
    *,
    cwd: Path,
    env: dict[str, str],
    command_runner: Callable[..., Any],
    timeout: float,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", module]
    completed = command_runner(command, cwd=str(cwd), env=env, timeout=timeout, capture_output=True, text=True)
    result = {
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
    }
    if completed.returncode != 0:
        raise QualificationError(f"{module} failed with exit code {completed.returncode}")
    combined = f"{completed.stdout or ''}\n{completed.stderr or ''}"
    ran = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    if ran is None:
        raise QualificationError(f"{module} did not report a unittest test count")
    count = int(ran.group(1))
    if count <= 0:
        raise QualificationError(f"{module} ran zero tests")
    if re.search(r"\bskipped=\d*[1-9]\d*\b|^\s*s", combined, re.IGNORECASE | re.MULTILINE):
        raise QualificationError(f"{module} reported skipped tests")
    result["tests_run"] = count
    return result


def _require_finite_number(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise QualificationError(f"{label} must be a finite non-negative number")


def validate_smoke_result(result: dict[str, Any], *, artifact_path: Path | None = None) -> None:
    if result.get("status") != "passed":
        raise QualificationError(f"alternating retention smoke failed: {result.get('error', result)}")
    if tuple(result.get("schedule", ())) != SMOKE_SCHEDULE:
        raise QualificationError("alternating retention smoke must run navigation then combat")
    changed = result.get("actor_changed", {})
    if changed.get("navigation") is not True or changed.get("combat") is not True:
        raise QualificationError("shared actor must change on both task updates")
    matrix = result.get("critic_change_matrix", {})
    if (matrix.get("navigation", {}).get("navigation") is not True
            or matrix.get("navigation", {}).get("combat") is not False
            or matrix.get("combat", {}).get("combat") is not True
            or matrix.get("combat", {}).get("navigation") is not False):
        raise QualificationError("only the active critic may change in each task update")
    if result.get("optimizer_steps") != result.get("nominal_optimizer_steps"):
        raise QualificationError("optimizer step count must match the nominal count")
    if result.get("kl_early_stop") is not False:
        raise QualificationError("qualification smoke must not KL early-stop")
    if result.get("save_load_exact") is not True:
        raise QualificationError("save/load exactness must cover model, critics, optimizer, counters and masks")
    continuation = result.get("next_task_continuation", {})
    if continuation.get("navigation") is not True or continuation.get("combat") is not True:
        raise QualificationError("loaded checkpoint must continue to either next task")
    timings = result.get("timings")
    if not isinstance(timings, dict) or not timings:
        raise QualificationError("qualification smoke must report detailed timings")
    for key, value in timings.items():
        _require_finite_number(value, f"qualification smoke timing {key}")
    rollouts = result.get("rollouts")
    if not isinstance(rollouts, list) or not rollouts:
        raise QualificationError("qualification smoke must report rollout details")
    for index, rollout in enumerate(rollouts):
        if rollout.get("task") not in SMOKE_SCHEDULE:
            raise QualificationError(f"qualification smoke rollout {index} has an invalid task")
        for key, value in rollout.items():
            if key.endswith("_seconds"):
                _require_finite_number(value, f"qualification smoke rollout {index} {key}")
    if artifact_path is not None and not artifact_path.is_file():
        raise QualificationError("alternating retention smoke artifact is missing")


def run_alternating_smoke(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    import_module: Callable[[str], Any],
) -> dict[str, Any]:
    retention = import_module("train_retention_campaign_block")
    cfg = {
        "lanes": args.lanes,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "seed": args.seed,
        "device": "cuda",
        "learning_rate": args.learning_rate,
    }
    result = retention.run_qualification_smoke(
        schedule=SMOKE_SCHEDULE,
        cfg=cfg,
        source=args.source,
        config=args.config,
        build_dir=args.build_dir / "retention-smoke",
        output_dir=output_dir / "alternating-smoke",
        checkpoint=args.checkpoint,
    )
    if not isinstance(result, dict):
        raise QualificationError("alternating retention smoke must return a JSON-compatible dict")
    artifact_path = output_dir / "alternating-smoke" / "alternating-smoke.json"
    validate_smoke_result(result, artifact_path=artifact_path)
    return {"status": "passed", **result}


def _parse_dmon_samples(stdout: str) -> dict[str, Any]:
    samples = []
    max_sm_util = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        samples.append(stripped)
        parts = stripped.split()
        if len(parts) >= 2:
            try:
                value = int(parts[1])
            except ValueError:
                continue
            max_sm_util = value if max_sm_util is None else max(max_sm_util, value)
    return {"samples": samples, "max_sm_util": max_sm_util}


def start_nvidia_dmon(
    *,
    cwd: Path,
    env: dict[str, str],
    dmon_popen: Callable[..., Any],
) -> dict[str, Any]:
    command = ["nvidia-smi", "dmon", "-s", "u", "-d", "1"]
    try:
        process = dmon_popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "samples": [], "max_sm_util": None}
    return {"available": True, "command": command, "process": process}


def finish_nvidia_dmon(monitor: dict[str, Any]) -> dict[str, Any]:
    process = monitor.pop("process", None)
    if process is None:
        return monitor
    stdout = ""
    stderr = ""
    returncode = None
    try:
        if getattr(process, "returncode", None) is None:
            process.terminate()
        if callable(getattr(process, "communicate", None)):
            stdout, stderr = process.communicate(timeout=5)
        else:
            process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        if callable(getattr(process, "communicate", None)):
            stdout, stderr = process.communicate(timeout=5)
    finally:
        returncode = getattr(process, "returncode", None)
    parsed = _parse_dmon_samples(stdout or "")
    return {
        **monitor,
        "returncode": returncode,
        "samples": parsed["samples"],
        "max_sm_util": parsed["max_sm_util"],
        "stderr_tail": (stderr or "")[-2000:],
    }


def validate_utilization_samples(samples: dict[str, Any], *, production_gate: bool) -> None:
    if not production_gate:
        return
    if samples.get("available") is not True or not samples.get("samples"):
        raise QualificationError("production gate requires usable nvidia-smi dmon utilization evidence")
    if not isinstance(samples.get("max_sm_util"), int) or samples["max_sm_util"] <= 0:
        raise QualificationError("production gate requires non-zero SM utilization evidence")


def run_qualification(
    args: argparse.Namespace,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
    command_runner: Callable[..., Any] = subprocess.run,
    dmon_popen: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    output_dir = args.output_dir
    receipt_path = output_dir / "qualification.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = "offline"
    run_env = dict(os.environ)
    run_env["WANDB_MODE"] = "offline"
    run_env["PYTHONPATH"] = str(ROOT) + os.pathsep + run_env.get("PYTHONPATH", "")
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "exit_code": 1,
        "cuda_qualified": False,
        "offline_wandb": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": "cuda",
        "source": str(args.source),
        "config": str(args.config),
        "build_dir": str(args.build_dir),
        "geometry": {
            "lanes": args.lanes,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
        },
        "production_gate_args": dict(PRODUCTION_GATE_GEOMETRY),
        "seed": args.seed,
        "gates": {},
    }
    try:
        require_planned_interfaces(import_module)
        receipt["source_hashes"] = source_manifest(args.source)
        receipt["checkpoint"] = str(args.checkpoint)
        receipt["checkpoint_expected_sha256"] = args.checkpoint_sha256
        actual_checkpoint_sha = sha256(args.checkpoint)
        receipt["checkpoint_sha256"] = actual_checkpoint_sha
        if actual_checkpoint_sha != args.checkpoint_sha256:
            raise QualificationError("checkpoint sha256 mismatch")
        receipt["checkpoint_verified"] = True
        run_env["AI2_CONTACT_CHECKPOINT"] = str(args.checkpoint)
        dmon_monitor = start_nvidia_dmon(cwd=ROOT, env=run_env, dmon_popen=dmon_popen)
        receipt["gates"]["navigation_phase0b_parity"] = run_navigation_phase0b(
            args, output_dir=output_dir, import_module=import_module)
        receipt["gates"]["contact_action_update_parity"] = run_unittest_gate(
            "tests.test_contact_action_update_parity", cwd=ROOT, env=run_env,
            command_runner=command_runner, timeout=args.test_timeout)
        receipt["gates"]["native_combat_update_parity"] = run_unittest_gate(
            "tests.test_native_combat_update_parity", cwd=ROOT, env=run_env,
            command_runner=command_runner, timeout=args.test_timeout)
        receipt["gates"]["alternating_retention_smoke"] = run_alternating_smoke(
            args, output_dir=output_dir, import_module=import_module)
        receipt["utilization_samples"] = finish_nvidia_dmon(dmon_monitor)
        validate_utilization_samples(receipt["utilization_samples"], production_gate=args.production_gate)
        receipt.update(status="passed", exit_code=0, cuda_qualified=True)
    except Exception as exc:
        if "dmon_monitor" in locals() and "utilization_samples" not in receipt:
            receipt["utilization_samples"] = finish_nvidia_dmon(dmon_monitor)
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    hashes = artifact_hashes(output_dir, exclude={"qualification.json"})
    receipt["artifact_hashes"] = hashes
    atomic_json(receipt_path, receipt)
    write_inventory(output_dir, block_id=output_dir.name)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=145)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--test-timeout", type=float, default=1200.0)
    parser.add_argument("--production-gate", action="store_true")
    args = parser.parse_args(argv)
    for name in ("lanes", "n_steps", "batch_size", "n_epochs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (args.lanes * args.n_steps) % args.batch_size:
        parser.error("--batch-size must divide lanes*n-steps")
    if args.production_gate:
        for name, expected in PRODUCTION_GATE_GEOMETRY.items():
            if getattr(args, name) != expected:
                parser.error(f"--production-gate requires --{name.replace('_', '-')} {expected}")
    args.source = args.source.resolve()
    args.config = args.config.resolve()
    args.build_dir = args.build_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.checkpoint_sha256):
        parser.error("--checkpoint-sha256 must be a 64-character hex digest")
    args.checkpoint_sha256 = args.checkpoint_sha256.lower()
    return args


def main(
    argv: list[str] | None = None,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
    command_runner: Callable[..., Any] = subprocess.run,
    dmon_popen: Callable[..., Any] = subprocess.Popen,
) -> int:
    args = parse_args(argv)
    receipt = run_qualification(
        args,
        import_module=import_module,
        command_runner=command_runner,
        dmon_popen=dmon_popen,
    )
    print(json.dumps({
        "output": str(args.output_dir / "qualification.json"),
        "status": receipt["status"],
        "cuda_qualified": receipt["cuda_qualified"],
        "error": receipt.get("error"),
    }, sort_keys=True))
    return int(receipt["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
