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
    reserved = set(manifest.get("reserved_training_seeds") or [])
    eval_seeds = {case.get("seed") for case in cases if isinstance(case, Mapping)}
    if reserved & eval_seeds:
        raise ValueError("training/evaluation seed overlap")
    panel = str(manifest.get("panel") or "")
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
    if bank.bank_sha256 == "0" * 64:
        bank = CaseBank(
            bank.panel,
            bank.generator_version,
            _bank_sha(list(bank.cases), panel=bank.panel, generator_version=bank.generator_version),
            list(bank.cases),
        )
    validate_bank_manifest(bank.to_manifest())
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
            _write_jsonl_once(tmp / "traces" / f"{case.case_id}.jsonl", [episode["trace"]])
        summary = summarize_episodes(episodes)
        _atomic_json(tmp / "summary.json", summary)
        _atomic_json(tmp / "environment-inventory.json", _environment_inventory())
        _atomic_json(tmp / "command-receipt.json", _command_receipt(command, started))
        _atomic_json(tmp / "checkpoint-hash.json", _checkpoint_record(checkpoint_path))
        _atomic_json(tmp / "source-hashes.json", _source_hashes(source_paths))
        result = {
            "schema_version": SCHEMA_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
            "curriculum_version": CURRICULUM_VERSION,
            "bank_sha256": bank.bank_sha256,
            "summary": summary,
            "episodes": len(episodes),
        }
        _atomic_json(tmp / "result.json", result)
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
    _require_damage_event(damage_events[0], player_slot, 1 - player_slot, hit_ids[0])
    _require_damage_event(damage_events[1], player_slot, 1 - player_slot, hit_ids[1])
    first_tick = _positive_tick(trace.get("first_hit_tick"), "first_hit_tick")
    relocation_tick = _positive_tick(trace.get("relocation_tick"), "relocation_tick")
    if relocation_tick < first_tick:
        raise ValueError("relocation cannot precede first hit")
    relocation_gap = _float_at_least(trace.get("relocation_gap"), 2.0, "relocation_gap")
    if not trace.get("relocation_pre_target") or not trace.get("relocation_post_target"):
        raise ValueError("trace must include clean relocation snapshots")
    if not _has_event(trace, "relocation", relocation_tick):
        raise ValueError("trace must include relocation event")
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
    env = env_factory(case, root / "traces")
    trace: Mapping[str, Any] | None = None
    try:
        obs, info = env.reset(seed=case.seed, options={"scenario": case.scenario})
        trace = info.get("trace") if isinstance(info, Mapping) else None
        done = False
        ticks = 0
        while not done and ticks < 300:
            action = _policy_action(policy, obs, deterministic=case.mode == "deterministic")
            obs, _reward, terminated, truncated, info = env.step(action)
            if isinstance(info, Mapping) and isinstance(info.get("trace"), Mapping):
                trace = info["trace"]
            done = bool(terminated or truncated)
            ticks += 1
        if trace is None:
            raise ValueError("episode produced no trace")
        trace = {**dict(trace), "case_id": case.case_id, "mode": case.mode}
        reconstruction = reconstruct_trace(trace)
        return {
            "schema_version": SCHEMA_VERSION,
            "case": case.to_json(),
            "trace": trace,
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
    if outcome in {"own_stock_loss", "opponent_stock_loss", "unsafe_support"}:
        return outcome
    if outcome == "response_deadline":
        return "failed_renewed_pursuit"
    if outcome in {"second_hit_deadline", "total_deadline"} and trace.get("pursuit_tick") is not None:
        return "pursued_missed_second"
    if trace.get("first_hit_tick") is not None and trace.get("relocation_tick") is None:
        return "first_hit_only"
    return outcome or "invalid_or_ambiguous"


def _safe_resolution(trace: Mapping[str, Any], player_slot: int) -> bool:
    if trace.get("outcome") != "success":
        return False
    if _learner_falls(trace) > 0:
        return False
    if trace.get("min_clearance") is not None and float(trace["min_clearance"]) < 0.2:
        return False
    players = trace.get("post_players")
    if not isinstance(players, list) or len(players) < 2:
        return False
    learner = next((player for player in players if int(player.get("player_slot", -1)) == player_slot), None)
    if learner is None:
        return False
    return bool(learner.get("grounded") and learner.get("fully_supported") and float(learner.get("horizontal_floor_clearance", 0.0)) >= 0.2)


def _learner_falls(trace: Mapping[str, Any]) -> int:
    raw_slot = trace.get("player_slot")
    if type(raw_slot) is not int:
        return 0
    slot = str(raw_slot)
    falls = trace.get("falls") or {}
    stock_changes = trace.get("stock_changes") or {}
    return max(int(falls.get(slot, falls.get(raw_slot, 0)) or 0), int(stock_changes.get(slot, stock_changes.get(raw_slot, 0)) or 0))


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


def _require_damage_event(event: Mapping[str, Any], source: int, target: int, attack_id: Any) -> None:
    if event.get("source_slot") != source or event.get("target_slot") != target:
        raise ValueError("malformed damage attribution")
    if event.get("attack_id") != attack_id or float(event.get("amount", 0.0)) <= 0.0:
        raise ValueError("malformed attack damage evidence")


def _has_event(trace: Mapping[str, Any], kind: str, tick: int) -> bool:
    events = trace.get("events") or []
    return any(isinstance(event, Mapping) and event.get("type") == kind and int(event.get("tick", -1)) == tick for event in events)


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
