#!/usr/bin/env python3
"""Pinned trace-first evaluator for the B2 re-engagement curriculum."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

import reengagement_curriculum as curriculum


SCHEMA_VERSION = "ai2.reengagement_evaluation.v1"
EVALUATOR_VERSION = "b2-reengagement-evaluator-v1"
GENERATOR_VERSION = curriculum.BANK_VERSION
CURRICULUM_VERSION = curriculum.CURRICULUM_VERSION
TRACE_VERSION = curriculum.TRACE_VERSION
MODES = ("deterministic", "sampled")
FRAME_STACK = 4
OBS_WIDTH = 64
STACKED_OBS_WIDTH = OBS_WIDTH * FRAME_STACK
ACTION_WIDTH = 10


@dataclasses.dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    seed: int
    mode: str
    player_slot: int
    direction: int
    scenario: Mapping[str, Any]

    @property
    def scenario_identity(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "player_slot": self.player_slot,
            "direction": self.direction,
            "gap": self.scenario.get("gap"),
            "midpoint": self.scenario.get("midpoint"),
            "scenario_id": self.scenario.get("id"),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "mode": self.mode,
            "player_slot": self.player_slot,
            "direction": self.direction,
            "scenario": dict(self.scenario),
            "scenario_identity": self.scenario_identity,
        }


@dataclasses.dataclass(frozen=True)
class CaseBank:
    panel: str
    generator_version: str
    bank_sha256: str
    cases: Sequence[EvaluationCase]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "panel": self.panel,
            "metric_version": CURRICULUM_VERSION,
            "generator_version": self.generator_version,
            "bank_sha256": self.bank_sha256,
            "modes": list(MODES),
            "cases": [case.to_json() for case in self.cases],
            "reserved_training_seeds": [],
        }


def build_case_bank(panel: str) -> CaseBank:
    if panel == "development":
        seeds = range(134000, 134020)
    elif panel == "confirmation":
        seeds = range(144000, 144100)
    else:
        raise ValueError("panel must be development or confirmation")
    cases: list[EvaluationCase] = []
    seeded_scenarios = []
    for index, seed in enumerate(seeds):
        slot = index % 2
        direction = -1 if (index // 2) % 2 == 0 else 1
        seeded_scenarios.append((index, seed, slot, direction, curriculum.scenario_from_seed(seed, player_slot=slot, direction=direction)))
    for mode in MODES:
        for _index, seed, slot, direction, scenario in seeded_scenarios:
            case_id = f"reengagement-{mode}-p{slot}-s{seed}"
            cases.append(
                EvaluationCase(
                    case_id=case_id,
                    seed=seed,
                    mode=mode,
                    player_slot=slot,
                    direction=direction,
                    scenario=scenario,
                )
            )
    sha = _bank_sha(cases, panel=panel, generator_version=GENERATOR_VERSION)
    bank = CaseBank(panel, GENERATOR_VERSION, sha, cases)
    validate_bank_manifest(bank.to_manifest())
    return bank


def validate_bank_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    panel = str(manifest.get("panel") or "")
    if manifest.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("bank generator version mismatch")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("bank manifest cases must be a nonempty list")
    ids = [case.get("case_id") for case in cases if isinstance(case, Mapping)]
    if len(ids) != len(cases) or len(ids) != len(set(ids)):
        raise ValueError("duplicate or malformed bank case IDs")
    mode_seed = [(case.get("mode"), case.get("seed")) for case in cases if isinstance(case, Mapping)]
    if len(mode_seed) != len(set(mode_seed)):
        raise ValueError("duplicate bank mode/seed pairs")
    _validate_exact_bank_cases(panel, cases)
    reserved = set(manifest.get("reserved_training_seeds") or [])
    eval_seeds = {case.get("seed") for case in cases if isinstance(case, Mapping)}
    if reserved & eval_seeds:
        raise ValueError("training/evaluation seed overlap")
    rebuilt = [
        EvaluationCase(
            case_id=str(case["case_id"]),
            seed=int(case["seed"]),
            mode=str(case["mode"]),
            player_slot=int(case["player_slot"]),
            direction=int(case["direction"]),
            scenario=dict(case["scenario"]),
        )
        for case in cases
    ]
    expected_sha = _bank_sha(rebuilt, panel=panel, generator_version=str(manifest["generator_version"]))
    if manifest.get("bank_sha256") != expected_sha:
        raise ValueError("bank SHA mismatch")
    return {"bank_sha256": expected_sha, "cases": len(rebuilt)}


def reconstruct_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = _reconstruct_trace_or_raise(trace)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "evidence_valid": False,
            "failure_category": _failure_category(trace),
            "invalid_reason": str(exc),
            "first_hit": _is_positive_tick(trace.get("first_hit_tick")),
            "pursuit": False,
            "second_hit": False,
            "safe_resolution": False,
            "damage": _damage(trace),
            "timing": _timing(trace),
        }
    return result


def summarize_episodes(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "episodes": 0,
        "modes": {mode: _empty_mode_summary() for mode in MODES},
        "failure_taxonomy": {},
    }
    for row in rows:
        trace = row["trace"]
        mode = str(trace.get("mode") or getattr(row.get("case"), "mode", ""))
        if mode not in MODES:
            raise ValueError("episode trace has invalid mode")
        slot = str(_int(trace.get("player_slot"), "player_slot"))
        if slot not in ("0", "1"):
            raise ValueError("episode trace has invalid player_slot")
        reconstruction = reconstruct_trace(trace)
        mode_summary = summary["modes"][mode]
        slot_summary = mode_summary["slots"][slot]
        for target in (mode_summary, slot_summary):
            _accumulate_counts(target, trace, reconstruction)
        taxonomy = reconstruction["failure_category"]
        summary["failure_taxonomy"][taxonomy] = summary["failure_taxonomy"].get(taxonomy, 0) + 1
        summary["episodes"] += 1
    return summary


def require_cuda(device: str = "cuda") -> str:
    if device != "cuda" and not device.startswith("cuda:"):
        raise RuntimeError("B2 evaluation policy inference requires CUDA")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on host installation
        raise RuntimeError("CUDA unavailable: torch import failed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    resolved = "cuda:0" if device == "cuda" else device
    index = int(resolved.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device unavailable: {resolved}")
    return resolved


def load_cuda_policy(checkpoint: str | Path, *, device: str = "cuda", ppo_class: Any | None = None) -> Any:
    resolved = require_cuda(device)
    if ppo_class is None:
        from stable_baselines3 import PPO

        ppo_class = PPO
    policy = ppo_class.load(str(checkpoint), device=resolved)
    params = list(policy.policy.parameters()) if hasattr(policy, "policy") else []
    if not params or any(getattr(getattr(param, "device", None), "type", None) != "cuda" for param in params):
        raise RuntimeError("CUDA policy load did not place all policy parameters on CUDA")
    return policy


def run_evaluation(
    *,
    policy: Any,
    bank: CaseBank,
    output_dir: str | Path,
    env_factory: Callable[[EvaluationCase, Path], Any] | None = None,
    command: Sequence[str] | None = None,
    checkpoint_path: str | Path | None = None,
    source_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    _preflight_policy(policy)
    exact_bank_required = bank.bank_sha256 != "0" * 64
    if bank.bank_sha256 == "0" * 64:
        bank = CaseBank(
            bank.panel,
            bank.generator_version,
            _bank_sha(list(bank.cases), panel=bank.panel, generator_version=bank.generator_version),
            list(bank.cases),
        )
    if exact_bank_required:
        validate_bank_manifest(bank.to_manifest())
    else:
        _validate_injected_bank(bank)
    if checkpoint_path is not None:
        _checkpoint_record(checkpoint_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        raise FileExistsError(tmp)
    started = time.time()
    tmp.mkdir(parents=True, exist_ok=False)
    (tmp / "episodes").mkdir()
    (tmp / "traces").mkdir()
    env_factory = env_factory or _real_env_factory
    episodes: list[dict[str, Any]] = []
    try:
        _atomic_json(tmp / "bank-manifest.json", bank.to_manifest())
        for case in bank.cases:
            episode = _run_one_case(policy, case, env_factory, tmp)
            episodes.append(episode)
            _atomic_json(tmp / "episodes" / f"{case.case_id}.json", episode)
            _write_jsonl_once(tmp / "traces" / f"{case.case_id}.jsonl", episode["trace_rows"])
        summary = summarize_episodes(episodes)
        _atomic_json(tmp / "summary.json", summary)
        _atomic_json(tmp / "environment-inventory.json", _environment_inventory())
        _atomic_json(tmp / "command-receipt.json", _command_receipt(command, started))
        checkpoint = _checkpoint_record(checkpoint_path)
        _atomic_json(tmp / "checkpoint-hash.json", checkpoint)
        _atomic_json(tmp / "source-hashes.json", _source_hashes(source_paths))
        result = {
            "schema_version": SCHEMA_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
            "curriculum_version": CURRICULUM_VERSION,
            "bank_sha256": bank.bank_sha256,
            "checkpoint": checkpoint,
            "summary": summary,
            "episodes": len(episodes),
        }
        _atomic_json(tmp / "result.json", result)
        _atomic_json(tmp / "artifact-inventory.json", _artifact_inventory(tmp))
        os.rename(tmp, output)
        return result
    except Exception:
        if tmp.exists():
            for path in sorted(tmp.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            tmp.rmdir()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frozen-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--panel", choices=("development", "confirmation"), default="development")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    bank = build_case_bank(args.panel)
    policy = load_cuda_policy(args.checkpoint, device=args.device)
    run_evaluation(
        policy=policy,
        bank=bank,
        output_dir=args.output,
        env_factory=lambda case, trace_dir: curriculum.make_reengagement_env(
            args.frozen_root,
            seed=case.seed,
            player_slot=case.player_slot,
            trace_dir=trace_dir,
        ),
        command=sys.argv,
        checkpoint_path=args.checkpoint,
        source_paths=[Path(__file__), Path(curriculum.__file__)],
    )
    return 0


def _reconstruct_trace_or_raise(trace: Mapping[str, Any]) -> dict[str, Any]:
    if trace.get("schema_version") != TRACE_VERSION:
        raise ValueError("wrong trace schema version")
    if trace.get("curriculum_version") != CURRICULUM_VERSION:
        raise ValueError("wrong curriculum version")
    player_slot = _int(trace.get("player_slot"), "player_slot")
    if player_slot not in (0, 1):
        raise ValueError("player_slot must be 0 or 1")
    hit_ids = list(trace.get("distinct_hit_ids") or [])
    if len(hit_ids) != 2 or len(set(hit_ids)) != 2 or any(not _nonzero_attack_id(value) for value in hit_ids):
        raise ValueError("trace must prove two distinct nonzero attack IDs")
    damage_events = trace.get("attributed_damage_events")
    if not isinstance(damage_events, list) or len(damage_events) < 2:
        raise ValueError("trace must include attributed damage events")
    first_tick = _positive_tick(trace.get("first_hit_tick"), "first_hit_tick")
    _require_hit_event(trace, ordinal=1, tick=first_tick, attack_id=hit_ids[0])
    _require_damage_event(damage_events[0], player_slot, 1 - player_slot, hit_ids[0], first_tick)
    relocation_tick = _positive_tick(trace.get("relocation_tick"), "relocation_tick")
    if relocation_tick < first_tick:
        raise ValueError("relocation cannot precede first hit")
    relocation_gap = _float_at_least(trace.get("relocation_gap"), 2.0, "relocation_gap")
    _require_clean_relocation(trace, relocation_tick, relocation_gap)
    pursuit_tick = _positive_tick(trace.get("pursuit_tick"), "pursuit_tick")
    if not 1 <= pursuit_tick - relocation_tick <= 60:
        raise ValueError("pursuit must occur within inclusive 60 ticks after relocation")
    gap_closed = _float_at_least(trace.get("gap_closed"), 1.0, "gap_closed")
    learner_progress = _float_at_least(trace.get("learner_progress"), 1.0, "learner_progress")
    second_tick = _positive_tick(trace.get("second_hit_tick"), "second_hit_tick")
    if not 1 <= second_tick - relocation_tick <= 90:
        raise ValueError("second hit must occur within inclusive 90 ticks after relocation")
    if second_tick <= first_tick:
        raise ValueError("second hit must follow first hit")
    _require_hit_event(trace, ordinal=2, tick=second_tick, attack_id=hit_ids[1])
    _require_damage_event(damage_events[1], player_slot, 1 - player_slot, hit_ids[1], second_tick)
    safe = _safe_resolution(trace, player_slot)
    if not safe:
        raise ValueError("trace must prove safe resolution")
    success = trace.get("outcome") == "success"
    if not success:
        raise ValueError("trace outcome is not success")
    return {
        "success": True,
        "evidence_valid": True,
        "failure_category": "success",
        "first_hit": True,
        "pursuit": True,
        "second_hit": True,
        "safe_resolution": True,
        "hit_attack_ids": hit_ids,
        "damage": _damage(trace),
        "timing": {
            **_timing(trace),
            "relocation_gap": relocation_gap,
            "gap_closed": gap_closed,
            "learner_progress": learner_progress,
        },
    }


def _run_one_case(policy: Any, case: EvaluationCase, env_factory: Callable[[EvaluationCase, Path], Any], root: Path) -> dict[str, Any]:
    env = env_factory(case, None)
    trace_rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset(seed=case.seed, options={"scenario": case.scenario})
        stack = FrameStack()
        stacked_obs = stack.reset(obs)
        if isinstance(info, Mapping) and isinstance(info.get("trace"), Mapping):
            trace_rows.append(_case_trace_row(info["trace"], case))
        done = False
        ticks = 0
        while not done and ticks < 300:
            action = _policy_action(policy, stacked_obs, deterministic=case.mode == "deterministic")
            obs, _reward, terminated, truncated, info = env.step(action)
            stacked_obs = stack.step(obs)
            if isinstance(info, Mapping) and isinstance(info.get("trace"), Mapping):
                trace_rows.append(_case_trace_row(info["trace"], case))
            done = bool(terminated or truncated)
            ticks += 1
        if not trace_rows:
            raise ValueError("episode produced no trace")
        trace = _episode_trace(trace_rows, case)
        reconstruction = reconstruct_trace(trace)
        return {
            "schema_version": SCHEMA_VERSION,
            "case": case.to_json(),
            "trace": trace,
            "trace_rows": trace_rows,
            "reconstruction": reconstruction,
            "success": bool(reconstruction["success"]),
            "evidence_valid": bool(reconstruction["evidence_valid"]),
            "failure_category": reconstruction["failure_category"],
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _policy_action(policy: Any, obs: Any, *, deterministic: bool) -> np.ndarray:
    predicted = policy.predict(obs, deterministic=deterministic)
    action = predicted[0] if isinstance(predicted, tuple) else predicted
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (ACTION_WIDTH,) or not np.isfinite(values).all():
        raise ValueError("policy must return one finite 10-wide action")
    return values


class FrameStack:
    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []

    def reset(self, obs: Any) -> np.ndarray:
        frame = _validate_observation(obs)
        zero = np.zeros_like(frame, dtype=np.float32)
        self._frames = [zero.copy() for _ in range(FRAME_STACK - 1)] + [frame.copy()]
        return self.get()

    def step(self, obs: Any) -> np.ndarray:
        frame = _validate_observation(obs)
        self._frames.pop(0)
        self._frames.append(frame.copy())
        return self.get()

    def get(self) -> np.ndarray:
        stacked = np.concatenate(self._frames, axis=0).astype(np.float32, copy=False)
        if stacked.shape != (STACKED_OBS_WIDTH,):
            raise ValueError("stacked observation must be 256 wide")
        return stacked


def _empty_mode_summary() -> dict[str, Any]:
    return {
        "episodes": 0,
        "full_sequence_successes": 0,
        "first_hit": 0,
        "pursuit": 0,
        "second_hit": 0,
        "safe_resolution": 0,
        "falls": 0,
        "deadlines": 0,
        "invalid_or_ambiguous": 0,
        "damage": {"total": 0.0, "average": 0.0},
        "timing": {"ticks_total": 0, "average_ticks": 0.0, "first_hit_ticks": [], "pursuit_ticks": [], "second_hit_ticks": []},
        "slots": {"0": _empty_slot_summary(), "1": _empty_slot_summary()},
    }


def _empty_slot_summary() -> dict[str, Any]:
    return {
        "episodes": 0,
        "full_sequence_successes": 0,
        "first_hit": 0,
        "pursuit": 0,
        "second_hit": 0,
        "safe_resolution": 0,
        "falls": 0,
        "deadlines": 0,
        "invalid_or_ambiguous": 0,
        "damage": {"total": 0.0, "average": 0.0},
        "timing": {"ticks_total": 0, "average_ticks": 0.0, "first_hit_ticks": [], "pursuit_ticks": [], "second_hit_ticks": []},
    }


def _accumulate_counts(target: dict[str, Any], trace: Mapping[str, Any], reconstruction: Mapping[str, Any]) -> None:
    target["episodes"] += 1
    target["full_sequence_successes"] += int(bool(reconstruction["success"]))
    target["first_hit"] += int(bool(reconstruction.get("first_hit") or trace.get("first_hit_tick") is not None))
    target["pursuit"] += int(bool(reconstruction.get("pursuit") or trace.get("pursuit_tick") is not None))
    target["second_hit"] += int(bool(reconstruction.get("second_hit") or trace.get("second_hit_tick") is not None))
    target["safe_resolution"] += int(bool(reconstruction.get("safe_resolution") or _safe_resolution(trace, int(trace.get("player_slot", 0)))))
    falls = _learner_falls(trace)
    target["falls"] += falls
    if str(trace.get("outcome")) in {"response_deadline", "second_hit_deadline", "total_deadline"}:
        target["deadlines"] += 1
    target["invalid_or_ambiguous"] += int(not bool(reconstruction["evidence_valid"]))
    target["damage"]["total"] += float(reconstruction["damage"]["dealt"])
    target["damage"]["average"] = target["damage"]["total"] / target["episodes"]
    timing = target["timing"]
    ticks = int(trace.get("tick") or 0)
    timing["ticks_total"] += ticks
    timing["average_ticks"] = timing["ticks_total"] / target["episodes"]
    for key, output in (("first_hit_tick", "first_hit_ticks"), ("pursuit_tick", "pursuit_ticks"), ("second_hit_tick", "second_hit_ticks")):
        if trace.get(key) is not None:
            timing[output].append(int(trace[key]))


def _failure_category(trace: Mapping[str, Any]) -> str:
    outcome = str(trace.get("outcome") or "")
    if outcome == "success":
        return "invalid_success_evidence"
    if outcome in {"own_stock_loss", "opponent_stock_loss"}:
        return outcome
    if outcome == "unsafe_support":
        return "unsafe_support"
    if outcome == "response_deadline":
        return "failed_pursuit"
    if outcome == "second_hit_deadline" and trace.get("pursuit_tick") is not None:
        return "pursued_but_missed_second"
    if trace.get("first_hit_tick") is not None and trace.get("relocation_tick") is None:
        return "first_hit_only"
    if trace.get("first_hit_tick") is not None and trace.get("relocation_tick") is not None and trace.get("pursuit_tick") is None:
        if "renewed_movement_request_ticks" in trace and int(trace.get("renewed_movement_request_ticks") or 0) == 0:
            return "first_hit_only"
        return "failed_pursuit"
    if outcome == "total_deadline" and trace.get("first_hit_tick") is None:
        if int(trace.get("attack_start_count") or 0) > 0:
            return "attack_started_missed"
        if int(trace.get("light_request_ticks") or 0) > 0:
            return "requested_but_no_start"
        if int(trace.get("movement_request_ticks") or 0) > 0 and float(trace.get("max_initial_approach_progress") or 0.0) < 1.0:
            return "no_approach"
        return "no_attack_request"
    if outcome in {"total_deadline", "second_hit_deadline"}:
        return "deadline"
    return outcome or "invalid_or_ambiguous"


def _safe_resolution(trace: Mapping[str, Any], player_slot: int) -> bool:
    if trace.get("outcome") != "success":
        return False
    if _any_falls_or_stock_changes(trace):
        return False
    if trace.get("min_clearance") is not None and float(trace["min_clearance"]) < 0.2:
        return False
    players = trace.get("post_players")
    if not isinstance(players, list) or len(players) < 2:
        return False
    seen = {int(player.get("player_slot", -1)): player for player in players if isinstance(player, Mapping)}
    if set(seen) != {0, 1}:
        return False
    return all(_player_alive_safe(player) for player in seen.values())


def _learner_falls(trace: Mapping[str, Any]) -> int:
    raw_slot = trace.get("player_slot")
    if type(raw_slot) is not int:
        return 0
    slot = str(raw_slot)
    falls = trace.get("falls") or {}
    stock_changes = trace.get("stock_changes") or {}
    return max(int(falls.get(slot, falls.get(raw_slot, 0)) or 0), int(stock_changes.get(slot, stock_changes.get(raw_slot, 0)) or 0))


def _any_falls_or_stock_changes(trace: Mapping[str, Any]) -> bool:
    falls = trace.get("falls") or {}
    stock_changes = trace.get("stock_changes") or {}
    for slot in (0, 1, "0", "1"):
        if int(falls.get(slot, 0) or 0) > 0 or int(stock_changes.get(slot, 0) or 0) > 0:
            return True
    return False


def _player_alive_safe(player: Mapping[str, Any]) -> bool:
    alive = player.get("alive")
    if alive is None:
        stocks = player.get("stocks")
        alive = isinstance(stocks, int) and stocks > 0
    return bool(
        alive is True
        and player.get("grounded") is True
        and player.get("fully_supported") is True
        and float(player.get("horizontal_floor_clearance", -1.0)) >= 0.2
    )


def _damage(trace: Mapping[str, Any]) -> dict[str, float]:
    events = trace.get("attributed_damage_events") or []
    dealt = sum(float(event.get("amount", 0.0)) for event in events if isinstance(event, Mapping))
    return {"dealt": dealt, "events": len(events)}


def _timing(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ticks": trace.get("tick"),
        "first_hit_tick": trace.get("first_hit_tick"),
        "relocation_tick": trace.get("relocation_tick"),
        "pursuit_tick": trace.get("pursuit_tick"),
        "second_hit_tick": trace.get("second_hit_tick"),
    }


def _require_damage_event(event: Mapping[str, Any], source: int, target: int, attack_id: Any, tick: int) -> None:
    if event.get("tick") != tick:
        raise ValueError("damage event tick mismatch")
    if event.get("source_slot") != source or event.get("target_slot") != target:
        raise ValueError("malformed damage attribution")
    if event.get("attack_id") != attack_id or float(event.get("amount", 0.0)) <= 0.0:
        raise ValueError("malformed attack damage evidence")


def _require_hit_event(trace: Mapping[str, Any], *, ordinal: int, tick: int, attack_id: Any) -> None:
    events = trace.get("events") or []
    matches = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("type") == "hit_credit"
        and event.get("ordinal") == ordinal
        and event.get("tick") == tick
        and event.get("attack_id") == attack_id
    ]
    if len(matches) != 1:
        raise ValueError("hit_credit event proof mismatch")


def _require_clean_relocation(trace: Mapping[str, Any], relocation_tick: int, relocation_gap: float) -> None:
    if not trace.get("relocation_pre_target") or not trace.get("relocation_post_target"):
        raise ValueError("trace must include clean relocation snapshots")
    events = [
        event
        for event in (trace.get("events") or [])
        if isinstance(event, Mapping) and event.get("type") == "relocation" and event.get("tick") == relocation_tick
    ]
    if len(events) != 1:
        raise ValueError("trace must include exact relocation event")
    event = events[0]
    if event.get("reason") != "encounter_boundary" or float(event.get("relocation_gap", -1.0)) != relocation_gap:
        raise ValueError("relocation event linkage mismatch")
    boundary = trace.get("relocation_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("trace must prove clean relocation boundary")
    if boundary.get("attack_resolved") is not True or boundary.get("stun_resolved") is not True:
        raise ValueError("relocation before clean attack/stun boundary")
    if boundary.get("learner_unchanged") is not True:
        raise ValueError("relocation changed learner state")
    preserved = boundary.get("target_preserved")
    required = ("stocks", "damage", "damage_done", "damage_taken_total")
    if not isinstance(preserved, Mapping) or any(preserved.get(name) is not True for name in required):
        raise ValueError("relocation did not preserve target counters")
    players = boundary.get("players")
    if not isinstance(players, list) or len(players) < 2 or not all(_player_alive_safe(player) for player in players):
        raise ValueError("relocation boundary players are not safely supported")


def _has_event(trace: Mapping[str, Any], kind: str, tick: int) -> bool:
    events = trace.get("events") or []
    return any(isinstance(event, Mapping) and event.get("type") == kind and int(event.get("tick", -1)) == tick for event in events)


def _case_trace_row(trace: Mapping[str, Any], case: EvaluationCase) -> dict[str, Any]:
    return {**dict(trace), "case_id": case.case_id, "mode": case.mode}


def _episode_trace(trace_rows: Sequence[Mapping[str, Any]], case: EvaluationCase) -> dict[str, Any]:
    final = dict(trace_rows[-1])
    events = []
    damage = []
    relocation_pre_target = final.get("relocation_pre_target")
    relocation_post_target = final.get("relocation_post_target")
    relocation_boundary = final.get("relocation_boundary")
    for row in trace_rows:
        for event in row.get("events") or []:
            if isinstance(event, Mapping):
                events.append(dict(event))
                if event.get("type") == "relocation":
                    relocation_pre_target = event.get("pre_target", relocation_pre_target)
                    relocation_post_target = event.get("post_target", relocation_post_target)
        damage.extend(dict(event) for event in row.get("attributed_damage_events") or [] if isinstance(event, Mapping))
        if row.get("relocation_boundary") is not None:
            relocation_boundary = row["relocation_boundary"]
    final.update(
        {
            "case_id": case.case_id,
            "mode": case.mode,
            "events": events,
            "attributed_damage_events": _unique_damage_events(damage or final.get("attributed_damage_events") or []),
            "relocation_pre_target": relocation_pre_target,
            "relocation_post_target": relocation_post_target,
            "relocation_boundary": relocation_boundary,
            "trace_row_count": len(trace_rows),
        }
    )
    return final


def _unique_damage_events(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for event in events:
        key = (event.get("tick"), event.get("attack_id"), event.get("source_slot"), event.get("target_slot"), event.get("amount"))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(event))
    return result


def _validate_observation(obs: Any) -> np.ndarray:
    values = np.asarray(obs, dtype=np.float32)
    if values.shape != (OBS_WIDTH,) or not np.isfinite(values).all():
        raise ValueError("B2 evaluator observations must be 64-wide finite vectors before 4-frame stacking")
    return values


def _preflight_policy(policy: Any) -> None:
    width = getattr(policy, "expected_observation_width", None)
    if width is None and hasattr(policy, "observation_space"):
        shape = getattr(policy.observation_space, "shape", None)
        width = shape[0] if shape else None
    if width is not None and int(width) != STACKED_OBS_WIDTH:
        raise ValueError("B2 evaluator policy input must be 256 wide")


def _validate_exact_bank_cases(panel: str, cases: Sequence[Mapping[str, Any]]) -> None:
    if panel == "development":
        seeds = list(range(134000, 134020))
    elif panel == "confirmation":
        seeds = list(range(144000, 144100))
    else:
        raise ValueError("bank panel must be development or confirmation")
    expected_count = len(seeds) * len(MODES)
    if len(cases) != expected_count:
        raise ValueError(f"bank must contain exact {expected_count} cases for {panel}")
    by_key = {(case.get("mode"), case.get("seed")): case for case in cases if isinstance(case, Mapping)}
    for mode in MODES:
        mode_cases = [case for case in cases if isinstance(case, Mapping) and case.get("mode") == mode]
        if [case.get("seed") for case in mode_cases] != seeds:
            raise ValueError(f"{panel} {mode} bank must use the exact pinned seed range")
    for index, seed in enumerate(seeds):
        slot = index % 2
        direction = -1 if (index // 2) % 2 == 0 else 1
        expected_scenario = curriculum.scenario_from_seed(seed, player_slot=slot, direction=direction)
        scenarios = []
        for mode in MODES:
            case = by_key.get((mode, seed))
            if case is None:
                raise ValueError("bank must contain exact deterministic/sampled mode seed pairs")
            expected_id = f"reengagement-{mode}-p{slot}-s{seed}"
            if case.get("case_id") != expected_id:
                raise ValueError("bank case ID does not match pinned contract")
            if case.get("player_slot") != slot:
                raise ValueError("bank player slot does not match pinned contract")
            if case.get("direction") != direction:
                raise ValueError("bank direction does not match pinned contract")
            scenario = dict(case.get("scenario") or {})
            comparable = {key: scenario.get(key) for key in ("seed", "player_slot", "direction", "gap", "midpoint", "id")}
            if comparable != {key: expected_scenario.get(key) for key in comparable}:
                raise ValueError("bank scenario regeneration mismatch")
            scenarios.append(comparable)
        if scenarios[0] != scenarios[1]:
            raise ValueError("deterministic and sampled cases must share identical scenario identity")


def _positive_tick(value: Any, name: str = "tick") -> int:
    tick = _int(value, name)
    if tick <= 0:
        raise ValueError(f"{name} must be positive")
    return tick


def _is_positive_tick(value: Any) -> bool:
    return type(value) is int and value > 0


def _int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _float_at_least(value: Any, minimum: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not np.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _nonzero_attack_id(value: Any) -> bool:
    return value not in (None, "", 0, "0")


def _bank_sha(cases: Sequence[EvaluationCase], *, panel: str, generator_version: str) -> str:
    payload = {
        "panel": panel,
        "generator_version": generator_version,
        "metric_version": CURRICULUM_VERSION,
        "cases": [case.to_json() for case in cases],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _checkpoint_record(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"checkpoint_path": None, "checkpoint_sha256": None}
    value = Path(path)
    return {"checkpoint_path": str(value), "checkpoint_sha256": _sha256(value)}


def _source_hashes(paths: Sequence[Path]) -> dict[str, Any]:
    selected = [Path(__file__), Path(curriculum.__file__)]
    selected.extend(Path(path) for path in paths)
    rows = []
    seen = set()
    for path in selected:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        rows.append({"path": str(path), "sha256": _sha256(resolved)})
    return {"schema_version": SCHEMA_VERSION, "sources": rows}


def _environment_inventory() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "cwd": str(Path.cwd()),
    }


def _artifact_inventory(directory: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "artifact-inventory.json" or path.name.endswith(".tmp"):
            continue
        rows.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "files": rows}


def _command_receipt(command: Sequence[str] | None, started: float) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": list(command or sys.argv),
        "started_unix": started,
        "finished_unix": time.time(),
        "git_head": _git_head(),
    }


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _real_env_factory(case: EvaluationCase, trace_dir: Path) -> Any:
    return curriculum.make_reengagement_env(".", seed=case.seed, player_slot=case.player_slot, trace_dir=trace_dir)


def _write_jsonl_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    raise SystemExit(main())
