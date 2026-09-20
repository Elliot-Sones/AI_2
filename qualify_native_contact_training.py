#!/usr/bin/env python3
"""Run current-source native contact CUDA qualification and seal a receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable
import xml.etree.ElementTree as ET

import benchmark_native_contact_training as base


ROOT = Path(__file__).resolve().parent

CHECK_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("original_cpu_exact", ("tests/test_native_contact_episode.py",)),
    ("cpu_cuda_exact", ("tests/test_native_contact_lanes.py",)),
    ("learner_update_exact", ("tests/test_native_combat_update_parity.py", "tests/test_contact_action_update_parity.py")),
)
ALLOWED_SKIP_MESSAGE = "two CUDA devices required"
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "NETRC")
SECRET_PREFIXES = ("AWS_", "GOOGLE_", "GCP_", "OPENAI_", "ANTHROPIC_", "WANDB_API_KEY")


class QualificationError(RuntimeError):
    """Raised when current-source native contact qualification cannot pass."""


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
        AI2_CONTACT_REFERENCE=str(args.reference_root / "combat_curriculum.py"),
        AI2_CONTACT_CHECKPOINT=str(args.checkpoint),
        AI2_COMBAT_PARITY_ARTIFACT_DIR=str(args.output_dir / "learner-artifacts"),
        AI2_CONTACT_UPDATE_PARITY_ARTIFACTS=str(args.output_dir / "contact-learner-artifacts"),
    )
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def validate_junit(path: Path, check: str) -> dict[str, Any]:
    cases = ET.parse(path).getroot().findall(".//testcase")
    if not cases:
        raise QualificationError(f"{check}: No test cases ran")
    skips: list[dict[str, str | None]] = []
    for case in cases:
        skipped = case.find("skipped")
        if skipped is None:
            continue
        message = skipped.get("message")
        skips.append({"name": case.get("name"), "message": message})
        if message != ALLOWED_SKIP_MESSAGE:
            raise QualificationError(f"{check}: unexpected skipped test {case.get('name')}: {message}")
    return {"cases": len(cases), "skips": skips}


def run_check(
    check: str,
    modules: tuple[str, ...],
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    command_runner: Callable[..., Any],
) -> dict[str, Any]:
    xml_path = args.output_dir / f"{check}.xml"
    log_path = args.output_dir / f"{check}.log"
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
        raise QualificationError(f"{check}: pytest did not write JUnit XML")
    junit = validate_junit(xml_path, check)
    if completed.returncode != 0:
        raise QualificationError(f"{check} failed with exit code {completed.returncode}")
    summary = {
        "check": check,
        "passed": True,
        "command": command,
        "cases": junit["cases"],
        "skips": junit["skips"],
        "log_sha256": base.sha256(log_path),
        "xml_sha256": base.sha256(xml_path),
    }
    path = args.output_dir / f"{check}.json"
    base.atomic_json(path, summary)
    return summary


def seal_inventory(output_dir: Path) -> None:
    if not (output_dir / "inventory.json").exists():
        base.write_inventory(output_dir, block_id=output_dir.name)


def run_qualification(
    args: argparse.Namespace,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    env = scrubbed_env(args)
    receipt: dict[str, Any] = {
        "schema_version": base.QUALIFICATION_SCHEMA,
        "qualified": False,
        "stage_version": base.CONTACT_VERSION,
        "checkpoint_sha256": base.CHECKPOINT_SHA256,
        "source_sha256": {},
        "checks": {check: False for check in base.CHECKS},
        "evidence": [],
        "started_at": started_at,
        "offline_wandb": True,
        "device": "cuda",
        "check_groups": {check: list(modules) for check, modules in CHECK_GROUPS},
    }
    try:
        receipt["checkpoint"] = base.validate_checkpoint(args.checkpoint)
        receipt["source_sha256"] = base.qualification_sources(args.reference_root, args.frozen_root)
        checks: dict[str, bool] = {}
        evidence: list[dict[str, str]] = []
        for check, modules in CHECK_GROUPS:
            summary = run_check(check, modules, args=args, env=env, command_runner=command_runner)
            checks[check] = True
            evidence_path = args.output_dir / f"{check}.json"
            evidence.append({"check": check, "path": evidence_path.name, "sha256": base.sha256(evidence_path)})
        receipt.update(qualified=True, checks=checks, evidence=evidence)
        candidate_path = args.output_dir / "qualification.candidate.json"
        base.atomic_json(candidate_path, receipt | {
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        base.require_qualification(candidate_path, args.reference_root, args.frozen_root)
        os.link(candidate_path, args.output_dir / "qualification.json")
    except Exception as exc:
        receipt["qualified"] = False
        receipt["checks"] = {check: False for check in base.CHECKS}
        receipt["evidence"] = []
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        if not (args.output_dir / "qualification.json").exists():
            base.atomic_json(args.output_dir / "qualification.json", receipt)
    finally:
        seal_inventory(args.output_dir)
    return json.loads((args.output_dir / "qualification.json").read_text())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    return args


def main(argv: list[str] | None = None, *, command_runner: Callable[..., Any] = subprocess.run) -> int:
    args = parse_args(argv)
    receipt = run_qualification(args, command_runner=command_runner)
    print(json.dumps({
        "output": str(args.output_dir / "qualification.json"),
        "qualified": receipt.get("qualified") is True,
        "error": receipt.get("error"),
    }, sort_keys=True))
    return 0 if receipt.get("qualified") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
