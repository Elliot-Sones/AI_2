#!/usr/bin/env python3
"""Run fail-closed B2 re-engagement qualification and seal a receipt."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "ai2.native_reengagement_qualification.v1"
QUALIFICATION_VERSION = "b2-reengagement-qualification-v1"
CURRICULUM_VERSION = "bridge_b2_reengagement-v1"
CONTROL_MASK = (True, True, True, True, True, False, True, True, False, False)
HELD_OUT_RANGES = (range(134000, 134020), range(144000, 144100))
HELD_OUT_SEEDS = frozenset(seed for group in HELD_OUT_RANGES for seed in group)
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "NETRC")
SECRET_PREFIXES = ("AWS_", "GOOGLE_", "GCP_", "OPENAI_", "ANTHROPIC_", "WANDB_API_KEY")
LOCAL_STAGE_MODULES = (
    "tests/test_reengagement_curriculum.py",
    "tests/test_native_reengagement_episode.py",
)
CUDA_STAGE_MODULES = (
    "tests/test_native_reengagement_lanes.py",
)
SOURCE_FILES = (
    "qualify_native_reengagement.py",
    "reengagement_curriculum.py",
    "native_reengagement_episode.py",
    "native_reengagement_lane_vec_env.py",
    "native_curriculum_ppo.py",
    "native/reengagement_episode.h",
    "native/reengagement_episode.c",
    "native/reengagement_lanes.h",
    "native/reengagement_lanes.c",
    "native/reengagement_lanes.cu",
)


class QualificationError(RuntimeError):
    """Raised when B2 qualification cannot safely pass."""


@dataclass(frozen=True)
class SystemProbe:
    cuda_available: bool
    cuda_device_count: int
    gpu_host: bool


@dataclass(frozen=True)
class CollectorSmokeResult:
    status: str
    reason: str
    finite_losses: bool
    optimizer_steps: int
    update_counter_before: int
    update_counter_after: int
    b2_episode_rows: list[dict[str, Any]]
    b1_regression: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8")
    os.replace(temporary, path)


def scrubbed_env(args: argparse.Namespace) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(upper.startswith(prefix) for prefix in SECRET_PREFIXES):
            continue
        if any(marker in upper for marker in SECRET_MARKERS):
            continue
        env[key] = value
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
        WANDB_MODE="offline",
        AI2_FROZEN_SOURCE_ROOT=str(args.frozen_root),
        AI2_REENGAGEMENT_REFERENCE_ROOT=str(args.reference_root),
        AI2_REENGAGEMENT_CHECKPOINT=str(args.checkpoint),
        AI2_REENGAGEMENT_QUALIFICATION_DIR=str(args.output_dir),
    )
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def probe_system() -> SystemProbe:
    cuda_available = False
    cuda_device_count = 0
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception:
        cuda_available = False
        cuda_device_count = 0
    return SystemProbe(
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        gpu_host=cuda_available or shutil.which("nvidia-smi") is not None,
    )


def inventory(probe: SystemProbe) -> dict[str, Any]:
    compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    nvcc = shutil.which("nvcc")
    pytorch: dict[str, Any] = {"available": False}
    try:
        import torch

        pytorch = {
            "available": True,
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        pytorch = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": platform.platform(),
        "compiler": {
            "cc": compiler,
            "nvcc": nvcc,
        },
        "cuda": asdict(probe),
        "pytorch": pytorch,
    }


def source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = ROOT / name
        if path.is_file():
            hashes[name] = sha256(path)
        else:
            hashes[name] = "missing"
    return hashes


def scenario_hashes(seeds: list[int] | tuple[int, ...] | None = None) -> dict[str, Any]:
    if seeds is None:
        seeds = [120000 + index for index in range(64)]
    overlap = sorted(HELD_OUT_SEEDS.intersection(int(seed) for seed in seeds))
    if overlap:
        raise QualificationError(f"training scenarios overlap held-out B2 seeds: {overlap[:8]}")
    digest = hashlib.sha256(
        json.dumps(list(map(int, seeds)), separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {
        "generator": "b2-reengagement-qualification-scenarios-v1",
        "count": len(seeds),
        "sha256": digest,
        "overlap_with_held_out": overlap,
        "held_out_ranges": [[134000, 134019], [144000, 144099]],
    }


def valid_b2_episode_row() -> dict[str, Any]:
    return {
        "curriculum_task": "combat",
        "allowed_control_mask": CONTROL_MASK,
        "curriculum_version": CURRICULUM_VERSION,
        "reengagement": {
            "diagnostics": {
                "version": CURRICULUM_VERSION,
                "trace_status": "reconstructable",
                "achieved_first_hit": True,
                "achieved_pursuit": True,
                "achieved_second_hit": True,
                "achieved_safe_resolution": True,
                "first_hit_attack_id": 11,
                "second_hit_attack_id": 12,
            }
        },
    }


def validate_b2_episode_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise QualificationError("collector smoke produced no B2 episode rows")
    for row in rows:
        if (
            row.get("curriculum_task") != "combat"
            or tuple(row.get("allowed_control_mask", ())) != CONTROL_MASK
            or row.get("curriculum_version") != CURRICULUM_VERSION
        ):
            raise QualificationError("Native episode records have a missing or wrong curriculum task/mask/version")
        diagnostics = ((row.get("reengagement") or {}).get("diagnostics") or {})
        if diagnostics.get("version") not in (None, CURRICULUM_VERSION):
            raise QualificationError("Native episode diagnostics have a missing or wrong B2 version")
        if diagnostics.get("trace_status") == "ambiguous" or diagnostics.get("ambiguous_trace"):
            raise QualificationError("collector smoke produced ambiguous trace evidence")
        if diagnostics.get("success") or diagnostics.get("sequence_completed"):
            first = int(diagnostics.get("first_hit_attack_id", 0) or 0)
            second = int(diagnostics.get("second_hit_attack_id", 0) or 0)
            if first == 0 or second == 0 or first == second:
                raise QualificationError("successful B2 row is not reconstructable from two distinct attacks")


def validate_collector_smoke(result: CollectorSmokeResult) -> dict[str, Any]:
    if result.status != "passed":
        raise QualificationError(result.reason or "collector smoke did not pass")
    if not result.finite_losses:
        raise QualificationError("collector smoke losses must be finite")
    if result.optimizer_steps < 1:
        raise QualificationError("collector smoke must execute at least one optimizer step")
    if result.update_counter_after <= result.update_counter_before:
        raise QualificationError("collector smoke must increment the learner update counter")
    validate_b2_episode_rows(result.b2_episode_rows)
    if result.b1_regression.get("status") != "passed":
        raise QualificationError("B1 regression evidence did not pass")
    return {
        "status": result.status,
        "reason": result.reason,
        "finite_losses": result.finite_losses,
        "optimizer_steps": result.optimizer_steps,
        "update_counter_before": result.update_counter_before,
        "update_counter_after": result.update_counter_after,
        "update_counter_delta": result.update_counter_after - result.update_counter_before,
        "episode_rows": len(result.b2_episode_rows),
        "b1_regression": result.b1_regression,
    }


def validate_junit(path: Path, stage: str, *, required_cuda: bool) -> dict[str, Any]:
    cases = ET.parse(path).getroot().findall(".//testcase")
    if not cases:
        raise QualificationError(f"{stage}: no test cases ran")
    skips: list[dict[str, str | None]] = []
    for case in cases:
        skipped = case.find("skipped")
        if skipped is None:
            continue
        row = {"name": case.get("name"), "message": skipped.get("message")}
        skips.append(row)
        if required_cuda:
            raise QualificationError(f"{stage}: skipped required CUDA test: {row['message']}")
    return {"cases": len(cases), "skips": skips}


def run_pytest_stage(
    stage: str,
    modules: tuple[str, ...],
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    command_runner: Callable[..., Any],
    required_cuda: bool,
) -> dict[str, Any]:
    xml_path = args.output_dir / f"{stage}.xml"
    log_path = args.output_dir / f"{stage}.log"
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", f"--junitxml={xml_path}", *modules]
    with log_path.open("x", encoding="utf8") as log:
        completed = command_runner(
            command,
            cwd=str(ROOT),
            env=env,
            timeout=args.test_timeout,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if not xml_path.is_file():
        raise QualificationError(f"{stage}: pytest did not write JUnit XML")
    junit = validate_junit(xml_path, stage, required_cuda=required_cuda)
    if completed.returncode != 0:
        raise QualificationError(f"{stage} failed with exit code {completed.returncode}")
    return {
        "status": "passed",
        "reason": "pytest passed",
        "command": command,
        "cases": junit["cases"],
        "skips": junit["skips"],
        "log_sha256": sha256(log_path),
        "xml_sha256": sha256(xml_path),
    }


def run_real_collector_smoke(args: argparse.Namespace) -> CollectorSmokeResult:
    probe = probe_system()
    if not probe.cuda_available:
        return CollectorSmokeResult(
            status="pending",
            reason="CUDA unavailable; collector/update smoke requires a GPU host",
            finite_losses=False,
            optimizer_steps=0,
            update_counter_before=0,
            update_counter_after=0,
            b2_episode_rows=[],
            b1_regression={"status": "pending"},
        )
    env = None
    try:
        import native_curriculum_ppo as ppo
        import native_reengagement_lane_vec_env as reengagement_env

        env = reengagement_env.build_reengagement_env(
            16,
            device="cuda",
            seed=120000,
            episodes_per_lane=8,
            record_capacity=512,
        )
        model = ppo.NativeCurriculumPPO(
            env=env,
            batch_size=8192,
            update_mode="deferred_metrics",
            action_head="bernoulli",
            expected_task_versions={"combat": CURRICULUM_VERSION},
            device=env.device,
            seed=120000,
            verbose=0,
        )
        before = int(model._n_updates)
        model.learn(total_timesteps=model.n_steps * model.n_envs)
        after = int(model._n_updates)
        diagnostics = dict(getattr(model, "last_update_diagnostics", {}) or {})
        finite = all(
            math_is_finite(value)
            for value in diagnostics.values()
            if isinstance(value, (int, float))
        )
        rows = list(getattr(model.rollout_buffer, "drained_episode_infos", []) or [])
        b1_versions = ppo.normalized_task_versions()
        ppo.validate_episode_infos(
            [
                {
                    "curriculum_task": "combat",
                    "curriculum_version": ppo.CONTACT_VERSION,
                    "allowed_control_mask": ppo.COMBAT_MASK,
                }
            ],
            "combat",
            ppo.COMBAT_MASK,
            b1_versions,
        )
        return CollectorSmokeResult(
            status="passed",
            reason="real CUDA collector plus one learner train call completed",
            finite_losses=finite,
            optimizer_steps=int(diagnostics.get("optimizer_steps", 0)),
            update_counter_before=before,
            update_counter_after=after,
            b2_episode_rows=rows,
            b1_regression={"status": "passed", "checks": ["default_b1_episode_version"]},
        )
    except Exception as exc:
        return CollectorSmokeResult(
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            finite_losses=False,
            optimizer_steps=0,
            update_counter_before=0,
            update_counter_after=0,
            b2_episode_rows=[],
            b1_regression={"status": "failed"},
        )
    finally:
        if env is not None:
            env.close()


def math_is_finite(value: int | float) -> bool:
    return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))


def stage_row(status: str, *, required: bool, reason: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "required": required,
        "reason": reason,
        "evidence": evidence or {},
    }


def run_qualification(
    args: argparse.Namespace,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    collector_smoke: Callable[[argparse.Namespace], CollectorSmokeResult] = run_real_collector_smoke,
    system_probe: Callable[[], SystemProbe] = probe_system,
) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    probe = system_probe()
    env = scrubbed_env(args)
    started_at = utc_now()
    stage_status: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    error: str | None = None
    try:
        artifacts["source_hashes"] = source_hashes()
        artifacts["scenario_hashes"] = scenario_hashes()
        artifacts["inventory"] = inventory(probe)
        stage_status["source_build_inventory"] = stage_row(
            "passed",
            required=True,
            reason="source/build inventory recorded",
            evidence={
                "source_files": len(artifacts["source_hashes"]),
                "missing_sources": [
                    name for name, digest in artifacts["source_hashes"].items() if digest == "missing"
                ],
            },
        )
        local = run_pytest_stage(
            "python_oracle_native_cpu",
            LOCAL_STAGE_MODULES,
            args=args,
            env=env,
            command_runner=command_runner,
            required_cuda=False,
        )
        stage_status["python_oracle_native_cpu"] = stage_row(
            "passed", required=True, reason="oracle/native CPU parity passed", evidence=local
        )
        if probe.cuda_available:
            cuda = run_pytest_stage(
                "native_cpu_cuda",
                CUDA_STAGE_MODULES,
                args=args,
                env=env,
                command_runner=command_runner,
                required_cuda=True,
            )
            stage_status["native_cpu_cuda"] = stage_row(
                "passed", required=True, reason="native CPU/CUDA parity passed", evidence=cuda
            )
            smoke_payload = validate_collector_smoke(collector_smoke(args))
            artifacts["collector_learner_smoke"] = smoke_payload
            stage_status["collector_learner_smoke"] = stage_row(
                "passed",
                required=True,
                reason="collector smoke and one learner update passed",
                evidence=smoke_payload,
            )
        else:
            stage_status["native_cpu_cuda"] = stage_row(
                "pending",
                required=probe.gpu_host,
                reason="CUDA unavailable on this host",
            )
            stage_status["collector_learner_smoke"] = stage_row(
                "pending",
                required=probe.gpu_host,
                reason="CUDA unavailable on this host",
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if "native_cpu_cuda" not in stage_status:
            stage_status["native_cpu_cuda"] = stage_row(
                "failed" if probe.gpu_host else "pending",
                required=probe.gpu_host,
                reason=str(exc),
            )
        elif stage_status["native_cpu_cuda"]["status"] == "passed" and "collector_learner_smoke" not in stage_status:
            stage_status["collector_learner_smoke"] = stage_row(
                "failed",
                required=probe.gpu_host,
                reason=str(exc),
            )
        if "collector_learner_smoke" not in stage_status:
            stage_status["collector_learner_smoke"] = stage_row(
                "failed" if probe.gpu_host else "pending",
                required=probe.gpu_host,
                reason=str(exc),
            )
    required_failed = [
        name
        for name, row in stage_status.items()
        if row["required"] and row["status"] != "passed"
    ]
    required_pending = [
        name
        for name, row in stage_status.items()
        if row["required"] is False and row["status"] == "pending"
    ]
    qualified = not required_failed and not required_pending and probe.cuda_available
    overall_status = "passed" if qualified else "failed" if required_failed else "pending"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "qualification_version": QUALIFICATION_VERSION,
        "qualified": qualified,
        "overall_status": overall_status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "command_receipt": {
            "argv": list(args.command_receipt),
            "cwd": str(ROOT),
            "executable": sys.executable,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "reference_root": str(args.reference_root),
        "frozen_root": str(args.frozen_root),
        "version_contract": {
            "curriculum_task": "combat",
            "curriculum_version": CURRICULUM_VERSION,
            "allowed_control_mask": CONTROL_MASK,
        },
        "stage_status": stage_status,
        "source_hashes": artifacts.get("source_hashes", {}),
        "scenario_hashes": artifacts.get("scenario_hashes", {}),
        "inventory": artifacts.get("inventory", inventory(probe)),
        "collector_learner_smoke": artifacts.get("collector_learner_smoke"),
    }
    if error:
        receipt["error"] = error
    candidate = args.output_dir / "qualification.candidate.json"
    atomic_json(candidate, receipt)
    os.replace(candidate, args.output_dir / "qualification.json")
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv if argv is None else [sys.argv[0], *argv])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if args.test_timeout <= 0:
        parser.error("--test-timeout must be positive")
    args.output_dir = args.output_dir.resolve()
    args.reference_root = args.reference_root.resolve()
    args.frozen_root = args.frozen_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.command_receipt = raw
    return args


def main(
    argv: list[str] | None = None,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    collector_smoke: Callable[[argparse.Namespace], CollectorSmokeResult] = run_real_collector_smoke,
    system_probe: Callable[[], SystemProbe] = probe_system,
) -> int:
    args = parse_args(argv)
    receipt = run_qualification(
        args,
        command_runner=command_runner,
        collector_smoke=collector_smoke,
        system_probe=system_probe,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "qualification.json"),
                "qualified": receipt.get("qualified") is True,
                "overall_status": receipt.get("overall_status"),
                "error": receipt.get("error"),
            },
            sort_keys=True,
        )
    )
    return 0 if receipt.get("qualified") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
