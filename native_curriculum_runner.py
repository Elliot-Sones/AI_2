#!/usr/bin/env python3
"""Thin native curriculum conductor.

This module intentionally coordinates one already-existing native block and the
immutable full-game evaluator. It does not implement a new PPO lesson.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "ai2.native_curriculum_runner.v1"
SCHEDULE_SCHEMA_VERSION = "ai2.native_curriculum_schedule.v1"
SUPPORTED_BLOCKS = {"native_contact"}
REQUIRED_MODES = ("deterministic", "sampled")


def run_schedule(
    *,
    schedule_path: str | Path,
    input_checkpoint: str | Path,
    output_root: str | Path,
    train_block: Callable[..., Mapping[str, Any]] | None = None,
    policy_loader: Callable[..., Any] | None = None,
    evaluator: Callable[..., Mapping[str, Any]] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    schedule_path = Path(schedule_path)
    input_checkpoint = Path(input_checkpoint)
    output_root = Path(output_root)
    if not input_checkpoint.is_file():
        raise FileNotFoundError(f"input checkpoint does not exist: {input_checkpoint}")

    schedule = _load_schedule(schedule_path)
    train_block = train_block or _default_train_block
    policy_loader = policy_loader or _default_policy_loader
    evaluator = evaluator or _default_evaluator

    if output_root.exists() and not resume and any(output_root.iterdir()):
        raise FileExistsError(f"output root is nonempty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    current_checkpoint = input_checkpoint
    phase_receipts: list[dict[str, Any]] = []
    final_decision = "promote"
    final_checkpoint = current_checkpoint

    for index, phase in enumerate(schedule["phases"]):
        phase_dir = output_root / f"phase-{index:03d}-{phase['id']}"
        receipt_path = phase_dir / "phase_receipt.json"
        if receipt_path.exists():
            if not resume:
                raise FileExistsError(f"phase receipt already exists: {receipt_path}")
            receipt = _read_json(receipt_path)
            if receipt.get("decision") != "promote":
                raise ValueError("cannot resume from non-promoted phase")
            current_checkpoint = Path(receipt["next_checkpoint"]["path"])
            phase_receipts.append(receipt)
            final_checkpoint = current_checkpoint
            continue

        phase_dir.mkdir(parents=True, exist_ok=False)
        block_dir = phase_dir / "block"
        eval_dir = phase_dir / "eval"
        phase_input = current_checkpoint

        try:
            train_result = train_block(phase=phase, checkpoint_path=phase_input, output_dir=block_dir)
            trained_checkpoint = _resolve_trained_checkpoint(train_result, block_dir)
            block_inventory = _require_file(block_dir / "inventory.json", "block inventory")
        except BaseException as exc:
            receipt = _failure_receipt(
                schedule_path=schedule_path,
                input_checkpoint=phase_input,
                phase=phase,
                phase_dir=phase_dir,
                stage="train",
                error=exc,
            )
            _atomic_json(receipt_path, receipt)
            raise

        try:
            policy = policy_loader(trained_checkpoint, phase=phase)
            eval_result = evaluator(policy=policy, phase=phase, checkpoint_path=trained_checkpoint, output_dir=eval_dir)
            eval_manifest = _require_file(eval_dir / "manifest.json", "evaluation manifest")
            metrics = _metrics_by_mode(eval_result, eval_dir)
            gate = _evaluate_gate(metrics, phase["gates"])
        except BaseException as exc:
            receipt = _failure_receipt(
                schedule_path=schedule_path,
                input_checkpoint=phase_input,
                phase=phase,
                phase_dir=phase_dir,
                stage="eval",
                error=exc,
                block_inventory=block_inventory if "block_inventory" in locals() else None,
            )
            _atomic_json(receipt_path, receipt)
            raise

        decision = "promote" if gate["passed"] else "retain"
        next_checkpoint = trained_checkpoint if decision == "promote" else phase_input
        receipt = _phase_receipt(
            schedule_path=schedule_path,
            input_checkpoint=phase_input,
            phase=phase,
            phase_dir=phase_dir,
            decision=decision,
            next_checkpoint=next_checkpoint,
            metrics_by_mode=metrics,
            gate=gate,
            block_inventory=block_inventory,
            eval_manifest=eval_manifest,
        )
        _atomic_json(receipt_path, receipt)
        phase_receipts.append(receipt)
        current_checkpoint = next_checkpoint
        final_checkpoint = next_checkpoint
        if decision != "promote":
            final_decision = decision
            break

    result = {
        "schema_version": SCHEMA_VERSION,
        "decision": final_decision,
        "schedule": _artifact_record(schedule_path),
        "input_checkpoint": _artifact_record(input_checkpoint),
        "next_checkpoint": _artifact_record(final_checkpoint),
        "phases": [
            {
                "phase": receipt["phase"],
                "decision": receipt["decision"],
                "next_checkpoint": receipt["next_checkpoint"],
                "artifacts": receipt.get("artifacts", {}),
            }
            for receipt in phase_receipts
        ],
        "phase_receipts": [
            _artifact_record(output_root / f"phase-{index:03d}-{receipt['phase']['id']}" / "phase_receipt.json")
            for index, receipt in enumerate(phase_receipts)
        ],
        "source_sha256": _source_hashes(),
    }
    root_receipt = output_root / "curriculum_receipt.json"
    if root_receipt.exists():
        if not resume:
            raise FileExistsError(root_receipt)
        return _read_json(root_receipt)
    _atomic_json(root_receipt, result)
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_schedule(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError("schedule must be a JSON object")
    _reject_unknown(data, {"schema_version", "phases"}, "schedule")
    if data.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        raise ValueError(f"schedule schema_version must be {SCHEDULE_SCHEMA_VERSION!r}")
    phases = data.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("schedule requires at least one phase")
    seen: set[str] = set()
    validated = []
    for raw in phases:
        phase = _validate_phase(raw)
        if phase["id"] in seen:
            raise ValueError(f"duplicate phase id: {phase['id']}")
        seen.add(phase["id"])
        validated.append(phase)
    return {"schema_version": data["schema_version"], "phases": validated}


def _validate_phase(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("phase must be a JSON object")
    _reject_unknown(raw, {"id", "block", "evaluation", "gates"}, "phase")
    phase_id = raw.get("id")
    if not isinstance(phase_id, str) or not phase_id or "/" in phase_id or ".." in phase_id:
        raise ValueError("phase id must be a simple nonempty string")
    block = raw.get("block")
    if not isinstance(block, dict):
        raise ValueError("phase block must be an object")
    _reject_unknown(block, {"type", "args"}, "phase block")
    if block.get("type") not in SUPPORTED_BLOCKS:
        raise ValueError(f"unsupported block type: {block.get('type')!r}")
    if not isinstance(block.get("args", {}), dict):
        raise ValueError("block args must be an object")
    evaluation = raw.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("phase evaluation must be an object")
    _reject_unknown(evaluation, {"modes", "episode_max", "cases", "policy_kind", "action_head", "device"}, "phase evaluation")
    modes = tuple(evaluation.get("modes", REQUIRED_MODES))
    if set(REQUIRED_MODES) - set(modes):
        raise ValueError("phase evaluation requires deterministic and sampled modes")
    gates = raw.get("gates")
    if not isinstance(gates, dict) or set(REQUIRED_MODES) - set(gates):
        raise ValueError("phase gates require deterministic and sampled entries")
    for mode, gate in gates.items():
        if mode not in REQUIRED_MODES:
            raise ValueError(f"unsupported gate mode: {mode}")
        if not isinstance(gate, dict) or not gate:
            raise ValueError(f"{mode} gate must be a nonempty object")
        _reject_unknown(gate, {"win_rate_min", "fall_rate_max", "hit_rate_min", "damage_done_min"}, f"{mode} gate")
    return {
        "id": phase_id,
        "block": {"type": block["type"], "args": dict(block.get("args", {}))},
        "evaluation": {**evaluation, "modes": list(modes)},
        "gates": {mode: dict(gates[mode]) for mode in REQUIRED_MODES},
    }


def _default_train_block(*, phase: Mapping[str, Any], checkpoint_path: Path, output_dir: Path) -> Mapping[str, Any]:
    import benchmark_native_contact_training as contact

    args = dict(phase["block"].get("args", {}))
    args.setdefault("checkpoint", checkpoint_path)
    args.setdefault("output", output_dir)
    if "deadline_unix" not in args:
        raise ValueError("native_contact block args require deadline_unix for the real runner")
    namespace = Namespace(**{key.replace("-", "_"): value for key, value in args.items()})
    result = contact.run_block(namespace)
    return {"checkpoint": output_dir / "final.zip", "result": result}


def _default_policy_loader(checkpoint_path: Path, *, phase: Mapping[str, Any]) -> Any:
    import evaluate_fullgame_bank as eval_bank

    config = phase.get("evaluation", {})
    kind = config.get("policy_kind", "actor")
    device = config.get("device", "cpu")
    action_head = config.get("action_head", "auto")
    if kind == "sb3":
        return eval_bank.SB3PolicyAdapter.load(checkpoint_path, device=device, action_head=action_head)
    if kind == "actor":
        return eval_bank.ActorPolicyAdapter.load(checkpoint_path, device=device, action_head=action_head)
    raise ValueError(f"unsupported policy_kind: {kind!r}")


def _default_evaluator(*, policy: Any, phase: Mapping[str, Any], checkpoint_path: Path, output_dir: Path) -> Mapping[str, Any]:
    import evaluate_fullgame_bank as eval_bank

    config = phase.get("evaluation", {})
    bank = eval_bank.build_default_bank()
    wanted = config.get("cases", "all")
    if wanted != "all":
        wanted_ids = set(wanted)
        bank = [case for case in bank if case.case_id in wanted_ids]
        missing = wanted_ids - {case.case_id for case in bank}
        if missing:
            raise ValueError(f"unknown evaluation case IDs: {sorted(missing)}")
    return eval_bank.run_evaluation(
        policy=policy,
        cases=bank,
        modes=tuple(config.get("modes", REQUIRED_MODES)),
        output_dir=output_dir,
        episode_max=config.get("episode_max"),
        checkpoint_path=checkpoint_path,
    )


def _resolve_trained_checkpoint(train_result: Mapping[str, Any], block_dir: Path) -> Path:
    candidate = train_result.get("checkpoint") if isinstance(train_result, Mapping) else None
    checkpoint = Path(candidate) if candidate is not None else block_dir / "final.zip"
    return _require_file(checkpoint, "trained checkpoint")


def _metrics_by_mode(eval_result: Mapping[str, Any], eval_dir: Path) -> dict[str, dict[str, float]]:
    episodes = list(eval_result.get("episodes", [])) if isinstance(eval_result, Mapping) else []
    if not episodes:
        episodes_path = eval_dir / "episodes.jsonl"
        if episodes_path.is_file():
            episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf8").splitlines() if line.strip()]
    by_mode: dict[str, list[Mapping[str, Any]]] = {mode: [] for mode in REQUIRED_MODES}
    for episode in episodes:
        mode = episode.get("mode")
        if mode in by_mode:
            by_mode[mode].append(episode)
    missing = [mode for mode, rows in by_mode.items() if not rows]
    if missing:
        raise ValueError(f"missing required evaluation mode: {missing}")
    return {mode: _summarize_mode(rows) for mode, rows in by_mode.items()}


def _summarize_mode(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    episodes = len(rows)
    wins = sum(1 for row in rows if row.get("result") == "win")
    falls = sum(1 for row in rows if int(row.get("self_stock_losses", 0)) > 0)
    hits = sum(int(row.get("hits", 0)) for row in rows)
    damage = sum(float(row.get("damage_done", 0.0)) for row in rows)
    return {
        "episodes": float(episodes),
        "win_rate": wins / episodes,
        "fall_rate": falls / episodes,
        "hit_rate": hits / episodes,
        "damage_done": damage / episodes,
    }


def _evaluate_gate(metrics_by_mode: Mapping[str, Mapping[str, float]], gates: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    failed: list[str] = []
    checks: dict[str, Any] = {}
    for mode in REQUIRED_MODES:
        metrics = metrics_by_mode[mode]
        gate = gates[mode]
        checks[mode] = {}
        for key, threshold in gate.items():
            if key.endswith("_min"):
                metric_name = key[:-4]
                passed = metrics[metric_name] >= float(threshold)
            elif key.endswith("_max"):
                metric_name = key[:-4]
                passed = metrics[metric_name] <= float(threshold)
            else:
                raise ValueError(f"unsupported gate key: {key}")
            checks[mode][key] = {"actual": metrics[metric_name], "threshold": float(threshold), "passed": passed}
            if not passed:
                failed.append(f"{mode}.{metric_name}")
    return {"passed": not failed, "failed": failed, "checks": checks}


def _phase_receipt(
    *,
    schedule_path: Path,
    input_checkpoint: Path,
    phase: Mapping[str, Any],
    phase_dir: Path,
    decision: str,
    next_checkpoint: Path,
    metrics_by_mode: Mapping[str, Any],
    gate: Mapping[str, Any],
    block_inventory: Path,
    eval_manifest: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": dict(phase),
        "decision": decision,
        "schedule": _artifact_record(schedule_path),
        "input_checkpoint": _artifact_record(input_checkpoint),
        "next_checkpoint": _artifact_record(next_checkpoint),
        "metrics_by_mode": metrics_by_mode,
        "gate": gate,
        "artifacts": {
            "phase_dir": {"path": str(phase_dir)},
            "block_inventory": _artifact_record(block_inventory),
            "eval_manifest": _artifact_record(eval_manifest),
        },
        "source_sha256": _source_hashes(),
    }


def _failure_receipt(
    *,
    schedule_path: Path,
    input_checkpoint: Path,
    phase: Mapping[str, Any],
    phase_dir: Path,
    stage: str,
    error: BaseException,
    block_inventory: Path | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"phase_dir": {"path": str(phase_dir)}}
    if block_inventory is not None and block_inventory.is_file():
        artifacts["block_inventory"] = _artifact_record(block_inventory)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": dict(phase),
        "decision": "fail",
        "schedule": _artifact_record(schedule_path),
        "input_checkpoint": _artifact_record(input_checkpoint),
        "next_checkpoint": _artifact_record(input_checkpoint),
        "artifacts": artifacts,
        "failure": {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "source_sha256": _source_hashes(),
    }


def _source_hashes() -> dict[str, str | None]:
    root = Path(__file__).resolve().parent
    names = ("native_curriculum_runner.py", "benchmark_native_contact_training.py", "evaluate_fullgame_bank.py")
    return {name: sha256_file(root / name) if (root / name).is_file() else None for name in names}


def _artifact_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown {label} keys: {sorted(unknown)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_schedule(
        schedule_path=args.schedule,
        input_checkpoint=args.input_checkpoint,
        output_root=args.output_root,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
