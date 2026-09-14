#!/usr/bin/env python3
"""Post-hoc CUDA-only E6 initial-checkpoint baseline evaluator.

This script plans read-only dev20 evaluations for the nine E6 chunk01
``initial.zip`` checkpoints. It performs no training. By default it only writes
the immutable invocation plan/provenance; pass ``--execute`` on the GPU host to
run commands through the existing campaign_learning safety gates.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import campaign_learning as learning


DEFAULT_CAMPAIGN_ROOT = Path("artifacts/e6-pilots-v1")
DEFAULT_EVAL_KIND = "initial_dev20"
DEFAULT_DEV_EPISODES = 20
DEFAULT_NEAR_SEED = 84001
DEFAULT_WIDE_SEED = 83001
DEFAULT_EVAL_TIMEOUT_SECONDS = learning.DEFAULT_DEV_EVAL_TIMEOUT_SECONDS
DEFAULT_RESERVE_SECONDS = learning.DEFAULT_RESERVE_SECONDS


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_for_spec(campaign_root: Path, spec: learning.RunSpec) -> Path:
    return Path(campaign_root).resolve() / "artifacts" / "sources" / spec.label


def config_for_spec(campaign_root: Path, spec: learning.RunSpec) -> Path:
    return source_for_spec(campaign_root, spec) / "config.yaml"


def chunk01_run_for_spec(campaign_root: Path, spec: learning.RunSpec) -> Path:
    return Path(campaign_root).resolve() / "artifacts" / "runs" / spec.label / "chunk01"


def initial_checkpoint_for_spec(campaign_root: Path, spec: learning.RunSpec) -> Path:
    return chunk01_run_for_spec(campaign_root, spec) / "initial.zip"


def preverify_initial_checkpoint(campaign_root: Path, spec: learning.RunSpec) -> dict[str, Any]:
    run_dir = chunk01_run_for_spec(campaign_root, spec)
    checkpoint = run_dir / "initial.zip"
    if not checkpoint.is_file():
        raise RuntimeError(f"missing required initial checkpoint for {spec.label}: {checkpoint}")

    result = _read_json_if_present(run_dir / "result.json")
    metadata = _read_json_if_present(run_dir / "metadata.json")
    if result is not None:
        reported_initial = Path(result.get("initial_checkpoint") or "")
        if reported_initial.resolve() != checkpoint.resolve():
            raise RuntimeError(
                f"initial checkpoint result path mismatch for {spec.label}: "
                f"expected {checkpoint}, got {reported_initial}"
            )
        if int(result.get("starting_steps", 0)) != 0:
            raise RuntimeError(f"initial checkpoint was not recorded before training for {spec.label}")
    if metadata is not None:
        if metadata.get("checkpoint") is not None:
            raise RuntimeError(
                f"initial training metadata recorded checkpoint for {spec.label}: {metadata.get('checkpoint')}"
            )
        controls = metadata.get("ppo_controls", {})
        if controls.get("reset_num_timesteps") is not True:
            raise RuntimeError(f"initial training metadata did not confirm fresh reset for {spec.label}")
        if metadata.get("seed") != spec.seed:
            raise RuntimeError(f"initial training metadata seed mismatch for {spec.label}")

    return {
        "checkpoint": learning.file_record(checkpoint),
        "training_result": str(run_dir / "result.json") if result is not None else None,
        "training_metadata": str(run_dir / "metadata.json") if metadata is not None else None,
        "result_starting_steps": result.get("starting_steps") if result is not None else None,
        "metadata_checkpoint": metadata.get("checkpoint") if metadata is not None else None,
    }


def build_initial_eval_command(
    *,
    python: str,
    evaluate_modes: Path,
    campaign_root: Path,
    output_root: Path,
    spec: learning.RunSpec,
    episodes: int,
    near_seed: int,
    wide_seed: int,
    timeout_seconds: float,
) -> learning.CommandSpec:
    source = source_for_spec(campaign_root, spec)
    config = config_for_spec(campaign_root, spec)
    if not source.is_dir():
        raise RuntimeError(f"missing per-arm source for {spec.label}: {source}")
    if not config.is_file():
        raise RuntimeError(f"missing per-arm config for {spec.label}: {config}")
    checkpoint = initial_checkpoint_for_spec(campaign_root, spec)
    initial_record = preverify_initial_checkpoint(campaign_root, spec)
    command = learning.build_eval_command(
        python=python,
        evaluate_modes=evaluate_modes,
        source=source,
        spec=spec,
        chunk_index=1,
        eval_kind=DEFAULT_EVAL_KIND,
        checkpoint=checkpoint,
        output=Path(output_root).resolve() / "artifacts" / "runs" / spec.label / "chunk01_initial_eval_dev20",
        episodes=episodes,
        near_seed=near_seed,
        wide_seed=wide_seed,
        timeout_seconds=timeout_seconds,
    )
    command.metadata.update(
        {
            "output_root": str(Path(output_root).resolve()),
            "checkpoint_role": "chunk01_initial",
            "expected_cumulative_steps": 0,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": learning.sha256_file(checkpoint),
            "config": str(config.resolve()),
            "config_sha256": learning.sha256_file(config),
            "near_seed": int(near_seed),
            "wide_seed": int(wide_seed),
            "training_initial_record": initial_record,
            "training_performed": False,
        }
    )
    return command


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    learning.require_cuda_device(args.device)
    commands = [
        build_initial_eval_command(
            python=args.python,
            evaluate_modes=args.evaluate_modes,
            campaign_root=args.campaign_root,
            output_root=args.output,
            spec=spec,
            episodes=args.episodes,
            near_seed=args.near_seed,
            wide_seed=args.wide_seed,
            timeout_seconds=args.eval_timeout,
        )
        for spec in learning.default_run_specs()
    ]
    return {
        "label": "e6_chunk01_initial_dev20_baselines",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(Path(args.campaign_root).resolve()),
        "evaluate_modes": str(Path(args.evaluate_modes).resolve()),
        "output": str(Path(args.output).resolve()),
        "device": args.device,
        "episodes": int(args.episodes),
        "near_seed": int(args.near_seed),
        "wide_seed": int(args.wide_seed),
        "checkpoint_role": "chunk01_initial",
        "training_performed": False,
        "purpose": "Post-hoc random-initial-policy baseline before claiming trained-policy gain.",
        "commands": [dataclasses.asdict(command) for command in commands],
    }


def materialize_plan_inputs(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    output = Path(plan["output"])
    plan_sha = learning.payload_sha256(plan)
    learning.write_json(output / "artifacts" / "campaign_initial_eval_plan.json", plan)
    provenance = {
        "campaign_root": learning.file_record(Path(args.campaign_root) / "artifacts" / "campaign_learning_plan.json"),
        "campaign_provenance": learning.file_record(Path(args.campaign_root) / "artifacts" / "campaign_learning_provenance.json"),
        "evaluate_modes": learning.file_record(args.evaluate_modes),
        "harness": learning.file_record(Path(__file__)),
        "source_configs": {
            spec.label: learning.file_record(config_for_spec(args.campaign_root, spec))
            for spec in learning.default_run_specs()
        },
        "initial_checkpoints": {
            spec.label: learning.file_record(initial_checkpoint_for_spec(args.campaign_root, spec))
            for spec in learning.default_run_specs()
        },
    }
    learning.write_json(output / "artifacts" / "campaign_initial_eval_provenance.json", provenance)
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan_sha,
        "plan_path": str(output / "artifacts" / "campaign_initial_eval_plan.json"),
        "provenance_path": str(output / "artifacts" / "campaign_initial_eval_provenance.json"),
        "args": {key: learning.json_safe(value) for key, value in vars(args).items()},
        "provenance": provenance,
    }
    receipt_dir = output / "artifacts" / "invocation_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    nonce = time.time_ns()
    receipt_path = receipt_dir / f"{nonce}-{plan_sha[:12]}.json"
    while receipt_path.exists():
        nonce += 1
        receipt_path = receipt_dir / f"{nonce}-{plan_sha[:12]}.json"
    learning.write_json(receipt_path, receipt)
    learning.write_json(
        output / "artifacts" / "campaign_initial_eval_latest_receipt.json",
        {"receipt": str(receipt_path), "plan_sha256": plan_sha},
    )


def validate_initial_eval_artifacts(command: learning.CommandSpec) -> dict[str, Any]:
    evidence = learning.validate_eval_artifacts(command)
    metadata_path = Path(command.output) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_steps = int(command.metadata.get("expected_cumulative_steps", 0))
    actual_steps = int(metadata.get("cumulative_steps", -1))
    if actual_steps != expected_steps:
        raise RuntimeError(
            f"model cumulative steps mismatch for {command.label}: expected {expected_steps}, got {actual_steps}"
        )
    expected_checkpoint_sha = command.metadata.get("checkpoint_sha256")
    if expected_checkpoint_sha and metadata.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise RuntimeError(f"initial eval checkpoint sha metadata mismatch for {command.label}")
    evidence["cumulative_steps"] = actual_steps
    return evidence


def execute_plan(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    commands = [
        learning.CommandSpec(**{**item, "cwd": Path(item["cwd"]), "output": Path(item["output"])})
        for item in plan["commands"]
    ]
    results: list[dict[str, Any]] = []
    manifest_path = Path(plan["output"]) / "artifacts" / "campaign_initial_eval_results.json"
    for command in commands:
        try:
            result = learning.run_command(command, deadline=args.deadline, reserve=args.reserve_seconds)
            if result.get("status") in {"completed", "skipped_existing"}:
                result["evidence"] = validate_initial_eval_artifacts(command)
        except Exception as exc:
            result = {
                "label": command.label,
                "kind": command.kind,
                "status": "failed",
                "output": str(command.output),
                "error": str(exc),
            }
        results.append(result)
        if result["status"] not in {"completed", "skipped_existing"}:
            learning.write_json(manifest_path, {"status": "failed", "results": results})
            return {"status": "failed", "results": results}
        learning.write_json(manifest_path, {"status": "running", "results": results})
    payload = {"status": "completed", "results": results}
    learning.write_json(manifest_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--evaluate-modes", type=Path, default=learning.DEFAULT_EVALUATE_MODES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=DEFAULT_DEV_EPISODES)
    parser.add_argument("--near-seed", type=int, default=DEFAULT_NEAR_SEED)
    parser.add_argument("--wide-seed", type=int, default=DEFAULT_WIDE_SEED)
    parser.add_argument("--eval-timeout", type=float, default=DEFAULT_EVAL_TIMEOUT_SECONDS)
    parser.add_argument("--reserve-seconds", type=float, default=DEFAULT_RESERVE_SECONDS)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--execute", action="store_true", help="actually run the generated eval commands sequentially")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    learning.require_cuda_device(args.device)
    if args.episodes != DEFAULT_DEV_EPISODES:
        raise SystemExit(f"--episodes must remain dev20 ({DEFAULT_DEV_EPISODES}) for this baseline")
    if args.near_seed != DEFAULT_NEAR_SEED or args.wide_seed != DEFAULT_WIDE_SEED:
        raise SystemExit("--near-seed/--wide-seed must remain 84001/83001 for dev20 comparability")
    if args.eval_timeout <= 0 or not math.isfinite(args.eval_timeout):
        raise SystemExit("--eval-timeout must be positive and finite")
    for attr in ("campaign_root", "evaluate_modes"):
        path = Path(getattr(args, attr))
        if not path.exists():
            raise SystemExit(f"{attr} does not exist: {path}")
        setattr(args, attr, path.resolve())
    args.output = Path(args.output).resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    materialize_plan_inputs(args, plan)
    result = {
        "plan": str(args.output / "artifacts" / "campaign_initial_eval_plan.json"),
        "commands": len(plan["commands"]),
        "executed": False,
    }
    if args.execute:
        result["run"] = execute_plan(args, plan)
        result["executed"] = True
    print(json.dumps(learning.json_safe(result), sort_keys=True))
    if args.execute and result.get("run", {}).get("status") != "completed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
