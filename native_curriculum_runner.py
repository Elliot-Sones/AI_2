#!/usr/bin/env python3
"""Thin native curriculum conductor.

This module intentionally coordinates one already-existing native block and the
immutable full-game evaluator. It does not implement a new PPO lesson.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "ai2.native_curriculum_runner.v1"
SCHEDULE_SCHEMA_VERSION = "ai2.native_curriculum_schedule.v1"
TRAIN_BLOCKS = {
    "native_contact": "benchmark_native_contact_training",
    "native_reengagement": "train_reengagement_campaign_block",
}
SUPPORTED_BLOCKS = frozenset(TRAIN_BLOCKS)
REQUIRED_MODES = ("deterministic", "sampled")
RATE_GATES = {"win_rate_min", "fall_rate_max", "hit_rate_min"}
NONNEGATIVE_GATES = {"damage_done_min"}
EVAL_SCHEMA_VERSION = "fullgame-eval-bank-v1"
ACTION_HEADS = {"auto", "gaussian", "bernoulli_10"}
POLICY_KINDS = {"sb3", "actor"}
EPISODE_RESULTS = {"win", "loss", "draw"}
TERMINAL_CAUSES = {"self_fall", "opponent_fall", "win", "loss", "draw", "timeout", "episode_max"}
EPISODE_COUNT_FIELDS = {
    "ticks",
    "max_ticks",
    "attack_requested",
    "attack_starts",
    "hits",
    "self_stock_losses",
    "opponent_stock_losses",
}
EPISODE_VALUE_FIELDS = {"damage_done", "damage_taken"}


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
    schedule_record = _artifact_record(schedule_path)
    input_record = _artifact_record(input_checkpoint)
    source_hashes = _source_hashes()
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
            current_checkpoint, resumed_decision = _validate_resume_phase(
                receipt=receipt,
                phase=phase,
                phase_dir=phase_dir,
                schedule_record=schedule_record,
                input_checkpoint=current_checkpoint,
                source_hashes=source_hashes,
            )
            phase_receipts.append(receipt)
            final_checkpoint = current_checkpoint
            if resumed_decision == "retain":
                final_decision = "retain"
                break
            continue

        phase_dir.mkdir(parents=True, exist_ok=False)
        block_dir = phase_dir / "block"
        eval_dir = phase_dir / "eval"
        phase_input = current_checkpoint

        try:
            train_result = train_block(phase=phase, checkpoint_path=phase_input, output_dir=block_dir)
            trained_checkpoint = _resolve_trained_checkpoint(train_result, block_dir)
            block_inventory = _verify_block_inventory(block_dir, trained_checkpoint)
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
            eval_manifest, episodes = _verify_evaluation(
                eval_dir=eval_dir,
                checkpoint=trained_checkpoint,
                phase=phase,
                eval_result=eval_result,
            )
            metrics = _metrics_by_mode(episodes)
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
            trained_checkpoint=trained_checkpoint,
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
        "schedule": schedule_record,
        "input_checkpoint": input_record,
        "next_checkpoint": _artifact_record(final_checkpoint),
        "phases": [
            {
                "phase": receipt["phase"],
                "decision": receipt["decision"],
                "trained_checkpoint": receipt["trained_checkpoint"],
                "next_checkpoint": receipt["next_checkpoint"],
                "artifacts": receipt.get("artifacts", {}),
            }
            for receipt in phase_receipts
        ],
        "phase_receipts": [
            _artifact_record(output_root / f"phase-{index:03d}-{receipt['phase']['id']}" / "phase_receipt.json")
            for index, receipt in enumerate(phase_receipts)
        ],
        "source_sha256": source_hashes,
    }
    root_receipt = output_root / "curriculum_receipt.json"
    if root_receipt.exists():
        if not resume:
            raise FileExistsError(root_receipt)
        existing = _read_json(root_receipt)
        if existing != result:
            raise ValueError("resume root receipt does not match current verified state")
        return existing
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
    if not isinstance(phase_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", phase_id) is None:
        raise ValueError("phase id must be a simple nonempty string")
    block = raw.get("block")
    if not isinstance(block, dict):
        raise ValueError("phase block must be an object")
    _reject_unknown(block, {"type", "args"}, "phase block")
    if block.get("type") not in SUPPORTED_BLOCKS:
        raise ValueError(f"unsupported block type: {block.get('type')!r}")
    if not isinstance(block.get("args", {}), dict):
        raise ValueError("block args must be an object")
    if {"checkpoint", "output"} & set(block.get("args", {})):
        raise ValueError("block args cannot override conductor checkpoint or output")
    evaluation = raw.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("phase evaluation must be an object")
    _reject_unknown(evaluation, {"modes", "episode_max", "cases", "policy_kind", "action_head", "device"}, "phase evaluation")
    required_evaluation = {"modes", "policy_kind", "action_head", "device"}
    missing_evaluation = required_evaluation - set(evaluation)
    if missing_evaluation:
        raise ValueError(f"phase evaluation requires explicit {sorted(missing_evaluation)}")
    modes = tuple(evaluation["modes"])
    if modes != REQUIRED_MODES and set(modes) != set(REQUIRED_MODES):
        raise ValueError("phase evaluation requires exactly deterministic and sampled modes")
    if len(modes) != len(set(modes)):
        raise ValueError("phase evaluation modes cannot contain duplicates")
    if evaluation["policy_kind"] not in POLICY_KINDS:
        raise ValueError(f"unsupported policy_kind: {evaluation['policy_kind']!r}")
    if evaluation["action_head"] not in ACTION_HEADS:
        raise ValueError(f"unsupported action_head: {evaluation['action_head']!r}")
    if not isinstance(evaluation["device"], str) or not evaluation["device"].startswith("cuda"):
        raise ValueError("phase evaluation device must be an explicit CUDA device")
    episode_max = evaluation.get("episode_max")
    if episode_max is not None and (type(episode_max) is not int or episode_max <= 0):
        raise ValueError("phase evaluation episode_max must be a positive integer")
    cases = evaluation.get("cases", "all")
    if cases != "all":
        if not isinstance(cases, list) or not cases or any(not isinstance(case, str) or not case for case in cases):
            raise ValueError("phase evaluation cases must be 'all' or a nonempty string list")
        if len(cases) != len(set(cases)):
            raise ValueError("phase evaluation cases cannot contain duplicates")
    gates = raw.get("gates")
    if not isinstance(gates, dict) or set(REQUIRED_MODES) - set(gates):
        raise ValueError("phase gates require deterministic and sampled entries")
    for mode, gate in gates.items():
        if mode not in REQUIRED_MODES:
            raise ValueError(f"unsupported gate mode: {mode}")
        if not isinstance(gate, dict) or not gate:
            raise ValueError(f"{mode} gate must be a nonempty object")
        _reject_unknown(gate, RATE_GATES | NONNEGATIVE_GATES, f"{mode} gate")
        for key, threshold in gate.items():
            if type(threshold) not in (int, float) or not math.isfinite(float(threshold)):
                raise ValueError(f"{mode}.{key} gate must be a finite number")
            value = float(threshold)
            if key in RATE_GATES and not 0.0 <= value <= 1.0:
                raise ValueError(f"{mode}.{key} gate must be between zero and one")
            if key in NONNEGATIVE_GATES and value < 0.0:
                raise ValueError(f"{mode}.{key} gate must be nonnegative")
    return {
        "id": phase_id,
        "block": {"type": block["type"], "args": dict(block.get("args", {}))},
        "evaluation": {**evaluation, "modes": list(modes)},
        "gates": {mode: dict(gates[mode]) for mode in REQUIRED_MODES},
    }


def _default_train_block(*, phase: Mapping[str, Any], checkpoint_path: Path, output_dir: Path) -> Mapping[str, Any]:
    block_type = phase["block"]["type"]
    module_name = TRAIN_BLOCKS.get(block_type)
    if module_name is None:
        raise ValueError(f"unsupported block type: {block_type!r}")
    train_module = importlib.import_module(module_name)

    configured = dict(phase["block"].get("args", {}))
    configured.pop("checkpoint", None)
    configured.pop("output", None)
    if "deadline_unix" not in configured:
        raise ValueError(f"{block_type} block args require deadline_unix for the real runner")
    argv = ["--checkpoint", str(checkpoint_path), "--output", str(output_dir)]
    for key, value in configured.items():
        if isinstance(value, (dict, list, tuple, bool)):
            raise ValueError(f"{block_type} block arg {key!r} must be a scalar")
        if value is None:
            continue
        argv.extend(("--" + key.replace("_", "-"), str(value)))
    try:
        namespace = train_module.parse_args(argv)
    except SystemExit as exc:
        raise ValueError(f"{block_type} block args do not satisfy the real runner contract") from exc
    if Path(namespace.checkpoint) != checkpoint_path or Path(namespace.output) != output_dir:
        raise ValueError(f"{block_type} parser changed conductor-owned paths")
    result = train_module.run_block(namespace)
    return {"checkpoint": output_dir / "final.zip", "result": result}


def _default_policy_loader(checkpoint_path: Path, *, phase: Mapping[str, Any]) -> Any:
    import evaluate_fullgame_bank as eval_bank

    config = phase["evaluation"]
    kind = config["policy_kind"]
    device = config["device"]
    action_head = config["action_head"]
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


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("inventory contains an unsafe path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("inventory contains an unsafe path")
    return pure.as_posix()


def _regular_file_under(root: Path, relative: str, label: str) -> Path:
    root = root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root must be a real directory")
    path = root / _safe_relative_path(relative)
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"{label} cannot contain symlinks")
        cursor = cursor.parent
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} file is missing or not regular: {relative}")
    return path


def _verify_block_inventory(block_dir: Path, checkpoint: Path) -> Path:
    block_dir = block_dir.absolute()
    inventory_path = _require_file(block_dir / "inventory.json", "block inventory")
    inventory = _read_json(inventory_path)
    if not isinstance(inventory, dict) or inventory.get("schema_version") != "curriculum-inventory-v1":
        raise ValueError("block inventory schema is invalid")
    rows = inventory.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("block inventory files are missing")
    expected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("block inventory row is invalid")
        relative = _safe_relative_path(row.get("path"))
        if relative in expected:
            raise ValueError("block inventory contains duplicate paths")
        size, digest = row.get("bytes"), row.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("block inventory row metadata is invalid")
        path = _regular_file_under(block_dir, relative, "block inventory")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError("block inventory checksum mismatch")
        expected[relative] = row
    actual = {
        path.relative_to(block_dir).as_posix()
        for path in block_dir.rglob("*")
        if path.is_file() and path.name != "inventory.json"
    }
    if actual != set(expected):
        raise ValueError("block inventory does not cover the complete block")
    try:
        checkpoint_relative = checkpoint.absolute().relative_to(block_dir).as_posix()
    except ValueError:
        raise ValueError("trained checkpoint is outside the block inventory") from None
    if checkpoint_relative not in expected:
        raise ValueError("trained checkpoint is absent from the block inventory")
    checkpoint_path = _regular_file_under(block_dir, checkpoint_relative, "trained checkpoint")
    return inventory_path


def _verify_evaluation(
    *,
    eval_dir: Path,
    checkpoint: Path,
    phase: Mapping[str, Any],
    eval_result: Mapping[str, Any],
) -> tuple[Path, list[Mapping[str, Any]]]:
    manifest_path = _require_file(eval_dir / "manifest.json", "evaluation manifest")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != EVAL_SCHEMA_VERSION:
        raise ValueError("evaluation manifest schema is invalid")
    checkpoint_record = manifest.get("checkpoint")
    if not isinstance(checkpoint_record, dict) or checkpoint_record.get("sha256") != sha256_file(checkpoint):
        raise ValueError("evaluation manifest checkpoint hash does not match")
    manifest_checkpoint = checkpoint_record.get("path")
    if not isinstance(manifest_checkpoint, str) or Path(manifest_checkpoint).absolute() != checkpoint.absolute():
        raise ValueError("evaluation manifest checkpoint path does not match")
    modes = manifest.get("modes")
    if not isinstance(modes, list) or tuple(modes) != tuple(phase["evaluation"]["modes"]):
        raise ValueError("evaluation manifest modes do not match the phase")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation manifest has no cases")
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError("evaluation manifest case IDs are invalid")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation manifest case IDs are duplicated")
    configured_cases = phase["evaluation"].get("cases", "all")
    if configured_cases == "all":
        import evaluate_fullgame_bank as eval_bank

        expected_case_ids = [case.case_id for case in eval_bank.build_default_bank()]
        if set(case_ids) != set(expected_case_ids) or len(case_ids) != len(expected_case_ids):
            raise ValueError("evaluation manifest does not contain the complete default evaluation case bank")
    elif set(case_ids) != set(configured_cases) or len(case_ids) != len(configured_cases):
        raise ValueError("evaluation manifest cases do not match the phase")

    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or "episodes.jsonl" not in hashes:
        raise ValueError("evaluation manifest does not hash episodes.jsonl")
    actual_files = {
        path.relative_to(eval_dir).as_posix()
        for path in eval_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != set(hashes):
        raise ValueError("evaluation manifest does not cover the complete evaluation")
    for relative, digest in hashes.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("evaluation manifest artifact hash is invalid")
        path = _regular_file_under(eval_dir, relative, "evaluation manifest")
        if sha256_file(path) != digest:
            raise ValueError("evaluation manifest artifact checksum mismatch")
    if isinstance(eval_result, Mapping) and isinstance(eval_result.get("manifest"), Mapping):
        if dict(eval_result["manifest"]) != manifest:
            raise ValueError("evaluation returned manifest differs from committed manifest")

    episodes: list[Mapping[str, Any]] = []
    for line_number, line in enumerate((eval_dir / "episodes.jsonl").read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"evaluation manifest episodes line {line_number} is invalid JSON") from None
        if not isinstance(row, dict):
            raise ValueError("evaluation manifest episode row is not an object")
        _validate_episode_row(row, manifest_cases={case["case_id"]: case for case in cases}, phase=phase)
        episodes.append(row)
    expected_pairs = {(case_id, mode) for case_id in case_ids for mode in modes}
    actual_pairs = [(row.get("case_id"), row.get("mode")) for row in episodes]
    if len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != expected_pairs:
        raise ValueError("evaluation manifest episode coverage is incomplete or duplicated")
    return manifest_path, episodes


def _validate_episode_row(
    row: Mapping[str, Any],
    *,
    manifest_cases: Mapping[str, Mapping[str, Any]],
    phase: Mapping[str, Any],
) -> None:
    if row.get("schema_version") != EVAL_SCHEMA_VERSION:
        raise ValueError("evaluation episode schema_version is invalid")
    if row.get("result") not in EPISODE_RESULTS:
        raise ValueError("evaluation episode result is invalid")
    if row.get("terminal_cause") not in TERMINAL_CAUSES:
        raise ValueError("evaluation episode terminal_cause is invalid")
    for field in EPISODE_COUNT_FIELDS:
        value = row.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"evaluation episode {field} must be a nonnegative integer")
    for field in EPISODE_VALUE_FIELDS:
        value = row.get(field)
        if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"evaluation episode {field} must be finite and nonnegative")
    if row["ticks"] > row["max_ticks"]:
        raise ValueError("evaluation episode ticks exceed max_ticks")
    case = manifest_cases.get(row.get("case_id"))
    if case is None:
        raise ValueError("evaluation episode case_id is absent from manifest")
    configured_max = phase["evaluation"].get("episode_max")
    expected_max = configured_max if configured_max is not None else case.get("max_ticks")
    if type(expected_max) is not int or expected_max <= 0 or row["max_ticks"] != expected_max:
        raise ValueError("evaluation episode max_ticks does not match the phase or case")


def _metrics_by_mode(episodes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
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
    falls = 0
    hits = 0
    hit_episodes = 0
    damage = 0.0
    for row in rows:
        stock_losses = row.get("self_stock_losses", 0)
        row_hits = row.get("hits", 0)
        row_damage = row.get("damage_done", 0.0)
        if type(stock_losses) not in (int, float) or not math.isfinite(float(stock_losses)) or stock_losses < 0:
            raise ValueError("evaluation self_stock_losses must be finite and nonnegative")
        if type(row_hits) not in (int, float) or not math.isfinite(float(row_hits)) or row_hits < 0:
            raise ValueError("evaluation hits must be finite and nonnegative")
        if type(row_damage) not in (int, float) or not math.isfinite(float(row_damage)) or row_damage < 0:
            raise ValueError("evaluation damage_done must be finite and nonnegative")
        falls += int(stock_losses > 0)
        hits += int(row_hits)
        hit_episodes += int(row_hits > 0)
        damage += float(row_damage)
    return {
        "episodes": float(episodes),
        "win_rate": wins / episodes,
        "fall_rate": falls / episodes,
        "hit_rate": hit_episodes / episodes,
        "hits_per_episode": hits / episodes,
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


def _verify_artifact_record(record: Any, *, expected_path: Path, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"resume {label} record is missing")
    recorded_path = record.get("path")
    size = record.get("bytes")
    digest = record.get("sha256")
    if (
        not isinstance(recorded_path, str)
        or type(size) is not int
        or size < 0
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError(f"resume {label} record is invalid")
    path = _require_file(Path(recorded_path), f"resume {label}")
    if path.absolute() != expected_path.absolute():
        raise ValueError(f"resume {label} path does not match")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise ValueError(f"resume {label} bytes do not match")
    return path


def _validate_resume_phase(
    *,
    receipt: Mapping[str, Any],
    phase: Mapping[str, Any],
    phase_dir: Path,
    schedule_record: Mapping[str, Any],
    input_checkpoint: Path,
    source_hashes: Mapping[str, str | None],
) -> tuple[Path, str]:
    decision = receipt.get("decision")
    if decision not in {"promote", "retain"}:
        raise ValueError("cannot resume from failed or invalid phase")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("resume phase receipt schema does not match")
    if receipt.get("phase") != dict(phase):
        raise ValueError("resume phase receipt does not match the current phase")
    if receipt.get("source_sha256") != dict(source_hashes):
        raise ValueError("resume phase receipt source hashes do not match")
    schedule = _verify_artifact_record(
        receipt.get("schedule"), expected_path=Path(schedule_record["path"]), label="schedule")
    if _artifact_record(schedule) != dict(schedule_record):
        raise ValueError("resume schedule receipt does not match")
    _verify_artifact_record(
        receipt.get("input_checkpoint"), expected_path=input_checkpoint, label="input checkpoint")
    trained_record = receipt.get("trained_checkpoint")
    if not isinstance(trained_record, dict) or not isinstance(trained_record.get("path"), str):
        raise ValueError("resume trained checkpoint record is invalid")
    trained_checkpoint = _verify_artifact_record(
        trained_record, expected_path=Path(trained_record["path"]), label="trained checkpoint")
    next_record = receipt.get("next_checkpoint")
    if not isinstance(next_record, dict) or not isinstance(next_record.get("path"), str):
        raise ValueError("resume next checkpoint record is invalid")
    next_checkpoint = _verify_artifact_record(
        next_record, expected_path=Path(next_record["path"]), label="next checkpoint")
    if decision == "promote" and next_checkpoint.absolute() != trained_checkpoint.absolute():
        raise ValueError("resume promoted phase does not select its trained checkpoint")
    if decision == "retain" and next_checkpoint.absolute() != input_checkpoint.absolute():
        raise ValueError("resume retained phase does not preserve its input checkpoint")
    block_inventory = _verify_block_inventory(phase_dir / "block", trained_checkpoint)
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("resume phase artifacts are missing")
    _verify_artifact_record(
        artifacts.get("block_inventory"), expected_path=block_inventory, label="block inventory")
    eval_manifest, episodes = _verify_evaluation(
        eval_dir=phase_dir / "eval", checkpoint=trained_checkpoint, phase=phase, eval_result={})
    _verify_artifact_record(
        artifacts.get("eval_manifest"), expected_path=eval_manifest, label="evaluation manifest")
    metrics = _metrics_by_mode(episodes)
    gate = _evaluate_gate(metrics, phase["gates"])
    expected_passed = decision == "promote"
    if metrics != receipt.get("metrics_by_mode") or gate != receipt.get("gate") or gate["passed"] != expected_passed:
        raise ValueError("resume phase metrics or promotion gate do not match verified evidence")
    return next_checkpoint, decision


def _phase_receipt(
    *,
    schedule_path: Path,
    input_checkpoint: Path,
    phase: Mapping[str, Any],
    phase_dir: Path,
    decision: str,
    trained_checkpoint: Path,
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
        "trained_checkpoint": _artifact_record(trained_checkpoint),
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
            "message": f"{stage} failed; raw exception details withheld",
        },
        "source_sha256": _source_hashes(),
    }


def _source_hashes() -> dict[str, str | None]:
    root = Path(__file__).resolve().parent
    names = (
        "native_curriculum_runner.py",
        "benchmark_native_contact_training.py",
        "train_reengagement_campaign_block.py",
        "evaluate_fullgame_bank.py",
    )
    return {name: sha256_file(root / name) if (root / name).is_file() else None for name in names}


def _artifact_record(path: str | Path) -> dict[str, Any]:
    path = _require_file(Path(path), "artifact")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    path = _require_file(path, "JSON artifact")
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
