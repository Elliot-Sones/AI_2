"""Fixed-action parity receipt for the native full-game MVP.

This harness intentionally separates two claims:

* original Python capture versus the serial native ``GameParityAdapter``, using
  recorded reset and per-step RNG draws;
* resident native CPU/CUDA lane parity, using the same recorded reset and
  per-step RNG draws, plus forced terminal/autoreset probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from native_fullgame_action_table import canonical_action_table


SCHEMA_VERSION = "ai2.fixed_action_trace_parity.v1"
HIGH_RISK_ACTIONS = {5, 6, 7, 8, 9}
TRACE_FIELDS_CHECKED = (
    "outputs.obs",
    "outputs.rewards",
    "outputs.terminated",
    "outputs.truncated",
    "state.raw.steps",
    "state.raw.terminated",
    "state.raw.max_timesteps",
    "state.players.x",
    "state.players.y",
    "state.players.vx",
    "state.players.vy",
    "state.players.state_id",
    "state.players.state",
    "state.players.bank",
    "state.players.weapon",
    "state.players.input",
    "events",
    "draws_consumed",
)
FORCED_AUTORESET_KINDS = ("termination", "truncation")
MANIFEST_SCHEMA_VERSION = "ai2.resident_stream_gate_manifest.v1"
SOURCE_HASH_FILES = (
    "fixed_action_trace_parity.py",
    "native_game_lane_vec_env.py",
    "native/game_lanes.c",
    "native/game_lanes.cu",
    "native/game_episode.c",
    "native/game_episode.h",
    "native/navigation_episode.c",
    "native/navigation_episode.h",
)
ACTION_SPECIFIC_FIELDS = {
    0: ("state.players.y", "state.players.vy", "state.players.input[0]"),
    1: ("state.players.x", "state.players.vx", "state.players.input[1]"),
    2: ("state.players.y", "state.players.vy", "state.players.input[2]"),
    3: ("state.players.x", "state.players.vx", "state.players.input[3]"),
    4: ("state.players.bank", "state.players.input[4]"),
    5: ("state.players.weapon", "events.weapon", "state.players.input[5]"),
    6: ("state.players.state_id", "state.players.state", "state.players.input[6]"),
    7: ("state.players.bank.8", "events.damage", "state.players.input[7]"),
    8: ("state.players.bank.8", "events.damage", "state.players.input[8]"),
    9: ("state.players.bank.12", "state.players.state", "state.players.input[9]"),
}


def _parse_int_csv(value: str) -> list[int]:
    if not value:
        raise argparse.ArgumentTypeError("empty integer list")
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value!r}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("empty integer list")
    return parsed


def _fixed_action(action_index: int) -> list[list[float]]:
    if action_index not in canonical_action_table():
        raise ValueError(f"unknown action index: {action_index}")
    action = [[0.0] * 10, [0.0] * 10]
    action[0][action_index] = 1.0
    return action


def _post_terminal_reset_draws(side: int = 0, phase: int = 1) -> list[dict[str, float]]:
    return [
        {"kind": 1, "a": 1, "b": 0, "value": side},
        *[
            {"kind": kind, "a": a, "b": b, "value": value}
            for _ in range(2)
            for kind, a, b, value in (
                (2, 2.6, 6.5, 4.5),
                (3, 500, 700, 600),
                (3, 0, 350, phase),
            )
        ],
    ]


def build_fixed_action_suite(
    *, seeds: list[int], horizon: int, actions: list[int] | None = None
) -> dict[str, Any]:
    """Build canonical fixed-action cases for every requested seed/action."""

    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("seeds must be a nonempty integer list")
    table = canonical_action_table()
    requested_actions = list(table) if actions is None else list(actions)
    unknown = sorted(set(requested_actions) - set(table))
    if unknown:
        raise ValueError(f"unknown action indices: {unknown}")
    cases = []
    for seed in seeds:
        for action_index in requested_actions:
            row = table[action_index]
            cases.append(
                {
                    "seed": seed,
                    "action_index": action_index,
                    "key": row["key"],
                    "meaning": row["meaning"],
                    "trace_required": bool(row["trace_required"]),
                    "high_risk": action_index in HIGH_RISK_ACTIONS,
                    "horizon": horizon,
                    "actions": [_fixed_action(action_index) for _ in range(horizon)],
                    "action_specific_fields": list(ACTION_SPECIFIC_FIELDS[action_index]),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "actions": requested_actions,
        "seeds": list(seeds),
        "horizon": horizon,
        "cases": cases,
    }


def _same_number(left: float, right: float, *, atol: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= atol


def _diff_records(expected: Any, actual: Any, path: str, *, atol: float) -> list[dict[str, Any]]:
    if type(expected) is not type(actual):
        return [
            {
                "field": path,
                "kind": "type",
                "expected": type(expected).__name__,
                "actual": type(actual).__name__,
            }
        ]
    if isinstance(expected, dict):
        diffs = [
            {"field": f"{path}.{key}", "kind": "missing"}
            for key in sorted(expected.keys() - actual.keys())
        ]
        diffs += [
            {"field": f"{path}.{key}", "kind": "unexpected"}
            for key in sorted(actual.keys() - expected.keys())
        ]
        for key in sorted(expected.keys() & actual.keys()):
            diffs.extend(_diff_records(expected[key], actual[key], f"{path}.{key}", atol=atol))
        return diffs
    if isinstance(expected, list):
        diffs = []
        if len(expected) != len(actual):
            diffs.append(
                {
                    "field": path,
                    "kind": "length",
                    "expected": len(expected),
                    "actual": len(actual),
                }
            )
        for index, (left, right) in enumerate(zip(expected, actual)):
            diffs.extend(_diff_records(left, right, f"{path}[{index}]", atol=atol))
        return diffs
    if isinstance(expected, float):
        return [] if _same_number(expected, actual, atol=atol) else [
            {"field": path, "kind": "value", "expected": expected, "actual": actual}
        ]
    return [] if expected == actual else [
        {"field": path, "kind": "value", "expected": expected, "actual": actual}
    ]


def compare_trace_pair(left: dict[str, Any], right: dict[str, Any], *, atol: float = 0.0) -> dict[str, Any]:
    """Compare trace frames exactly enough to make any mismatch a blocker."""

    action_index = left.get("action_index", right.get("action_index"))
    diffs = _diff_records(left.get("frames", []), right.get("frames", []), "frames", atol=atol)
    passed = not diffs
    message = (
        f"action_index={action_index} exact"
        if passed
        else f"action_index={action_index} mismatch at {diffs[0]['field']}"
    )
    return {
        "passed": passed,
        "severity": None if passed else "blocker",
        "message": message,
        "first_difference": None if passed else diffs[0],
        "differences": diffs[:30],
        "fields_checked": list(TRACE_FIELDS_CHECKED),
    }


def resident_replay_step_supported() -> bool:
    """Task 2B adds this API; fail closed until it is actually exposed."""

    try:
        import native_game_lane_vec_env as lanes
    except Exception:
        return False
    backend = getattr(lanes, "GameLaneBackend", None)
    if backend is None:
        return False
    return all(hasattr(backend, name) for name in ("replay_step", "rng_status"))


def _rng_exact(case: dict[str, Any]) -> bool:
    rng = case.get("rng")
    return (
        isinstance(rng, dict)
        and _is_exact_int(rng.get("cursor"))
        and _is_exact_int(rng.get("count"))
        and _is_exact_int(rng.get("error"))
        and rng["error"] == 0
        and rng["cursor"] == rng["count"]
    )


def _autoreset_exact(case: dict[str, Any], kind: str) -> bool:
    evidence = case.get(f"resident_{kind}_autoreset")
    return (
        isinstance(evidence, dict)
        and evidence.get("status") == "exact"
        and evidence.get("same_step_autoreset") is True
        and evidence.get("same_step_reset_obs_exact") is True
        and evidence.get("healthy_subsequent_step") is True
    )


def _cuda_stream_exact(case: dict[str, Any]) -> bool:
    stream = case.get("resident_cuda_stream")
    return (
        isinstance(stream, dict)
        and stream.get("mode") == "non_default"
        and stream.get("used_non_default_stream") is True
        and stream.get("synchronized_before_compare") is True
    )


def _cuda_observation_checks_exact(case: dict[str, Any]) -> bool:
    checks = case.get("resident_cuda_observation_checks")
    if not isinstance(checks, dict):
        return False
    required = (
        "reset_observation_exact",
        "step_observation_exact",
        "reward_exact",
        "terminated_exact",
        "truncated_exact",
        "terminal_observation_exact",
    )
    if not all(checks.get(key) is True for key in required):
        return False
    return (
        isinstance(checks.get("reset_observation_sha256"), str)
        and bool(checks["reset_observation_sha256"])
        and isinstance(checks.get("terminal_observation_sha256"), str)
        and bool(checks["terminal_observation_sha256"])
    )


def _is_exact_int(value: Any) -> bool:
    return type(value) is int


def _case_evidence_blockers(case: dict[str, Any], *, horizon: int) -> list[dict[str, Any]]:
    blockers = []
    action_index = case.get("action_index")
    seed = case.get("seed")
    if case.get("reset_compared") is not True:
        blockers.append(
            {
                "action_index": action_index,
                "seed": seed,
                "reason": "reset_not_compared",
                "first_divergence": "reset comparison evidence is not exact",
            }
        )
    if not _is_exact_int(case.get("compared_ticks")) or case["compared_ticks"] != horizon:
        blockers.append(
            {
                "action_index": action_index,
                "seed": seed,
                "reason": "compared_ticks_incomplete",
                "first_divergence": f"expected {horizon} compared ticks, got {case.get('compared_ticks')}",
            }
        )
    if not _rng_exact(case):
        blockers.append(
            {
                "action_index": action_index,
                "seed": seed,
                "reason": "rng_consumption_inexact",
                "first_divergence": "RNG cursor/count/error evidence is not exact",
            }
        )
    if case.get("terminal_autoreset_evidence") not in {"exact", "not_exercised"}:
        blockers.append(
            {
                "action_index": action_index,
                "seed": seed,
                "reason": "terminal_autoreset_evidence_missing",
                "first_divergence": "terminal/autoreset evidence must be exact or explicitly not_exercised",
            }
        )
    if case.get("resident_cpu_recorded_replay_exact") is not True:
        blockers.append(
            {
                "action_index": action_index,
                "seed": seed,
                "reason": case.get("resident_cpu_blocker", "resident_cpu_recorded_replay_missing"),
                "first_divergence": case.get("resident_cpu_first_divergence")
                or case.get("resident_replay_first_divergence"),
            }
        )
    return blockers


def _expected_matrix(
    *, expected_actions: list[int], expected_seeds: list[int], horizon: int
) -> set[tuple[int, int]]:
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not expected_actions or any(type(action) is not int for action in expected_actions):
        raise ValueError("expected_actions must be a nonempty integer list")
    if not expected_seeds or any(type(seed) is not int for seed in expected_seeds):
        raise ValueError("expected_seeds must be a nonempty integer list")
    return {(action, seed) for action in expected_actions for seed in expected_seeds}


def _matrix_blockers(
    cases: list[dict[str, Any]],
    *,
    expected_actions: list[int],
    expected_seeds: list[int],
    horizon: int,
) -> list[dict[str, Any]]:
    expected = _expected_matrix(
        expected_actions=expected_actions, expected_seeds=expected_seeds, horizon=horizon
    )
    observed: dict[tuple[int, int], int] = {}
    blockers = []
    for case in cases:
        action_index = case.get("action_index")
        seed = case.get("seed")
        if not _is_exact_int(action_index) or not _is_exact_int(seed):
            blockers.append(
                {
                    "action_index": action_index,
                    "seed": seed,
                    "reason": "invalid_action_seed_case",
                    "first_divergence": "action_index and seed must be exact ints, not bools",
                }
            )
        else:
            key = (action_index, seed)
            observed[key] = observed.get(key, 0) + 1
            if key not in expected:
                blockers.append(
                    {
                        "action_index": action_index,
                        "seed": seed,
                        "reason": "unexpected_action_seed_case",
                        "first_divergence": f"unexpected case {key}",
                    }
                )
        if not _is_exact_int(case.get("horizon")) or case["horizon"] != horizon:
            blockers.append(
                {
                    "action_index": case.get("action_index"),
                    "seed": case.get("seed"),
                    "reason": "horizon_mismatch",
                    "first_divergence": f"expected {horizon}, got {case.get('horizon')}",
                }
            )
    for action_index, seed in sorted(expected - set(observed)):
        blockers.append(
            {
                "action_index": action_index,
                "seed": seed,
                "reason": "missing_action_seed_case",
                "first_divergence": "requested action/seed case was not run",
            }
        )
    for (action_index, seed), count in sorted(observed.items()):
        if count > 1:
            blockers.append(
                {
                    "action_index": action_index,
                    "seed": seed,
                    "reason": "duplicate_action_seed_case",
                    "first_divergence": f"{count} duplicate cases",
                }
            )
    return blockers


def three_way_exact_allowed(
    cases: list[dict[str, Any]],
    *,
    expected_actions: list[int],
    expected_seeds: list[int],
    horizon: int,
) -> bool:
    if not cases:
        return False
    if _matrix_blockers(
        cases,
        expected_actions=expected_actions,
        expected_seeds=expected_seeds,
        horizon=horizon,
    ):
        return False
    for case in cases:
        if case.get("python_reference_exact") is not True:
            return False
        if case.get("native_cpu_cuda_exact") is not True:
            return False
        if case.get("resident_cpu_generated_rng_recorded") is not True:
            return False
        if case.get("resident_cpu_recorded_replay_exact") is not True:
            return False
        if case.get("resident_lanes_replay_step_supported") is not True:
            return False
        if case.get("reset_compared") is not True:
            return False
        if not _is_exact_int(case.get("compared_ticks")) or case["compared_ticks"] != horizon:
            return False
        if case.get("terminal_autoreset_evidence") not in {"exact", "not_exercised"}:
            return False
        if not _rng_exact(case):
            return False
    if not any(_autoreset_exact(case, "termination") for case in cases):
        return False
    if not any(_autoreset_exact(case, "truncation") for case in cases):
        return False
    return True


def build_receipt(
    *,
    cases: list[dict[str, Any]],
    backends: list[str],
    expected_actions: list[int],
    expected_seeds: list[int],
    horizon: int,
) -> dict[str, Any]:
    blockers = _matrix_blockers(
        cases,
        expected_actions=expected_actions,
        expected_seeds=expected_seeds,
        horizon=horizon,
    )
    for case in cases:
        blockers.extend(_case_evidence_blockers(case, horizon=horizon))
        if "python" not in backends:
            blockers.append(
                {
                    "action_index": case.get("action_index"),
                    "seed": case.get("seed"),
                    "reason": "python_reference_backend_missing",
                    "first_divergence": "Task 2 requires original Python capture vs serial native replay",
                }
            )
        elif case.get("python_reference_exact") is not True:
            blockers.append(
                {
                    "action_index": case.get("action_index"),
                    "seed": case.get("seed"),
                    "reason": case.get("blocker", "python_reference_mismatch"),
                    "first_divergence": case.get("first_divergence"),
                }
            )
        if "native_cpu" in backends and case.get("resident_cpu_generated_rng_recorded") is not True:
            blockers.append(
                {
                    "action_index": case.get("action_index"),
                    "seed": case.get("seed"),
                    "reason": case.get("resident_cpu_blocker", "resident_cpu_generated_rng_missing"),
                    "first_divergence": case.get("resident_cpu_first_divergence"),
                }
            )
        if "native_cuda" in backends and case.get("native_cpu_cuda_exact") is not True:
            blockers.append(
                {
                    "action_index": case.get("action_index"),
                    "seed": case.get("seed"),
                    "reason": "native_cpu_cuda_exact_missing",
                    "first_divergence": case.get("native_cpu_cuda_exact_note"),
                }
            )
        if "native_cuda" in backends and case.get("native_cpu_cuda_exact") is True:
            if not _cuda_stream_exact(case):
                blockers.append(
                    {
                        "action_index": case.get("action_index"),
                        "seed": case.get("seed"),
                        "reason": "native_cuda_nondefault_stream_unverified",
                        "first_divergence": "resident CUDA replay must execute on a synchronized non-default stream",
                    }
                )
            if not _cuda_observation_checks_exact(case):
                blockers.append(
                    {
                        "action_index": case.get("action_index"),
                        "seed": case.get("seed"),
                        "reason": "native_cuda_observation_checks_incomplete",
                        "first_divergence": (
                            "resident CUDA replay must record exact reset/step/terminal observations, "
                            "rewards, terminated, and truncated checks"
                        ),
                    }
                )
    has_termination_autoreset = any(_autoreset_exact(case, "termination") for case in cases)
    has_truncation_autoreset = any(_autoreset_exact(case, "truncation") for case in cases)
    if not has_termination_autoreset:
        blockers.append(
            {
                "action_index": None,
                "seed": None,
                "reason": "termination_autoreset_evidence_missing",
                "first_divergence": "need at least one exact termination same-step autoreset plus healthy next step",
            }
        )
    if not has_truncation_autoreset:
        blockers.append(
            {
                "action_index": None,
                "seed": None,
                "reason": "truncation_autoreset_evidence_missing",
                "first_divergence": "need at least one exact truncation same-step autoreset plus healthy next step",
            }
        )
    python_reference_exact = bool(cases) and all(
        case.get("python_reference_exact") is True for case in cases
    )
    resident_cpu_recorded = (
        bool(cases)
        and "native_cpu" in backends
        and all(case.get("resident_cpu_generated_rng_recorded") is True for case in cases)
    )
    native_cpu_cuda_values = [
        case.get("native_cpu_cuda_exact")
        for case in cases
        if case.get("native_cpu_cuda_exact") is not None
    ]
    native_cpu_cuda_exact = (
        None if not native_cpu_cuda_values else all(value is True for value in native_cpu_cuda_values)
    )
    resident_supported = bool(cases) and all(
        case.get("resident_lanes_replay_step_supported") is True for case in cases
    )
    three_way_exact = three_way_exact_allowed(
        cases,
        expected_actions=expected_actions,
        expected_seeds=expected_seeds,
        horizon=horizon,
    )
    promotion_blockers = []
    if native_cpu_cuda_exact is not True:
        promotion_blockers.append(
            {
                "reason": "native_cuda_replay_unverified",
                "detail": "promotion requires exact resident CUDA replay evidence, not CPU-only execution",
            }
        )
    if "native_cuda" in backends and native_cpu_cuda_exact is True:
        if not all(_cuda_stream_exact(case) for case in cases):
            promotion_blockers.append(
                {
                    "reason": "native_cuda_nondefault_stream_unverified",
                    "detail": "promotion requires resident CUDA replay on a synchronized non-default PyTorch stream",
                }
            )
        if not all(_cuda_observation_checks_exact(case) for case in cases):
            promotion_blockers.append(
                {
                    "reason": "native_cuda_observation_checks_incomplete",
                    "detail": (
                        "promotion requires exact reset observations, step observations, rewards, done flags, "
                        "and terminal/reset observation evidence"
                    ),
                }
            )
    if not has_termination_autoreset:
        promotion_blockers.append(
            {
                "reason": "termination_autoreset_evidence_missing",
                "detail": "promotion requires exact termination same-step reset obs and healthy follow-up",
            }
        )
    if not has_truncation_autoreset:
        promotion_blockers.append(
            {
                "reason": "truncation_autoreset_evidence_missing",
                "detail": "promotion requires exact truncation same-step reset obs and healthy follow-up",
            }
        )
    if any(case.get("resident_cpu_recorded_replay_exact") is not True for case in cases):
        promotion_blockers.append(
            {
                "reason": "resident_cpu_recorded_replay_inexact",
                "detail": "promotion requires exact recorded-draw resident CPU replay for every matrix case",
            }
        )
    cuda_gate_exact = (
        "native_cuda" not in backends
        or (
            native_cpu_cuda_exact is True
            and all(_cuda_stream_exact(case) for case in cases)
            and all(_cuda_observation_checks_exact(case) for case in cases)
        )
    )
    promotion_passed = three_way_exact and cuda_gate_exact
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blockers else "completed",
        "backends": list(backends),
        "expected_actions": list(expected_actions),
        "expected_seeds": list(expected_seeds),
        "horizon": horizon,
        "python_reference_exact": python_reference_exact,
        "resident_cpu_generated_rng_recorded": resident_cpu_recorded,
        "native_cpu_cuda_exact": native_cpu_cuda_exact,
        "native_cpu_cuda_exact_note": (
            "unverified locally; resident CUDA lane not requested or unavailable"
            if native_cpu_cuda_exact is None
            else None
        ),
        "resident_lanes_replay_step_supported": resident_supported,
        "termination_autoreset_exact": has_termination_autoreset,
        "truncation_autoreset_exact": has_truncation_autoreset,
        "resident_lanes_replay_step_supported_note": (
            "false until Task 2B exposes per-step replay for resident lanes"
            if not resident_supported
            else None
        ),
        "native_cuda_nondefault_stream_exact": (
            None if "native_cuda" not in backends else all(_cuda_stream_exact(case) for case in cases)
        ),
        "native_cuda_observation_checks_exact": (
            None if "native_cuda" not in backends else all(_cuda_observation_checks_exact(case) for case in cases)
        ),
        "three_way_exact": three_way_exact,
        "promotion_status": "passed" if promotion_passed else "blocked",
        "promotion_blockers": [] if promotion_passed else promotion_blockers,
        "three_way_exact_note": (
            "requires exact original replay, CPU/CUDA resident parity, replay_step support, "
            "and exact RNG cursor/count/error for every action and seed"
            if not three_way_exact
            else None
        ),
        "cases": cases,
        "blockers": blockers,
        "fields_checked": list(TRACE_FIELDS_CHECKED),
        "high_risk_actions": sorted(HIGH_RISK_ACTIONS),
    }


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _float32_pair(outputs: dict[str, Any], key: str) -> Any:
    import numpy as np

    return np.asarray([outputs[key]["0"], outputs[key]["1"]], dtype=np.float32)


def _bool_pair(outputs: dict[str, Any], key: str) -> bool:
    value = outputs[key]
    if type(value) is not bool:
        raise ValueError(f"expected bool output field {key}")
    return value


def _compare_tensor_to_expected(tensor: Any, expected: Any, field: str) -> list[dict[str, Any]]:
    import numpy as np

    actual = tensor.detach().cpu().contiguous().numpy()
    if actual.shape != expected.shape:
        return [{"field": field, "kind": "shape", "expected": expected.shape, "actual": actual.shape}]
    if actual.dtype != expected.dtype:
        return [{"field": field, "kind": "dtype", "expected": str(expected.dtype), "actual": str(actual.dtype)}]
    if np.array_equal(actual, expected):
        return []
    where = np.argwhere(actual != expected)
    index = tuple(int(value) for value in where[0]) if where.size else ()
    return [
        {
            "field": f"{field}{index}",
            "kind": "value",
            "expected": float(expected[index]),
            "actual": float(actual[index]),
        }
    ]


def _compare_resident_outputs(
    *,
    obs: Any,
    terminal: Any,
    rewards: Any,
    terminated: Any,
    truncated: Any,
    expected_outputs: dict[str, Any],
    tick: int,
) -> list[dict[str, Any]]:
    import numpy as np

    done = _bool_pair(expected_outputs, "terminated") or _bool_pair(expected_outputs, "truncated")
    observed_obs = terminal if done else obs
    diffs = _compare_tensor_to_expected(
        observed_obs,
        _float32_pair(expected_outputs, "obs"),
        f"tick[{tick}].outputs.obs",
    )
    expected_rewards = np.asarray(
        [expected_outputs["rewards"]["0"], expected_outputs["rewards"]["1"]],
        dtype=np.float32,
    )
    diffs.extend(_compare_tensor_to_expected(rewards, expected_rewards, f"tick[{tick}].outputs.rewards"))
    actual_terminated = [bool(value) for value in terminated.detach().cpu().tolist()]
    actual_truncated = [bool(value) for value in truncated.detach().cpu().tolist()]
    expected_terminated = [_bool_pair(expected_outputs, "terminated")] * 2
    expected_truncated = [_bool_pair(expected_outputs, "truncated")] * 2
    if actual_terminated != expected_terminated:
        diffs.append(
            {
                "field": f"tick[{tick}].outputs.terminated",
                "kind": "value",
                "expected": expected_terminated,
                "actual": actual_terminated,
            }
        )
    if actual_truncated != expected_truncated:
        diffs.append(
            {
                "field": f"tick[{tick}].outputs.truncated",
                "kind": "value",
                "expected": expected_truncated,
                "actual": actual_truncated,
            }
        )
    return diffs


def _resident_trace_comparable(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(trace, dict):
        return None
    return {
        key: trace.get(key)
        for key in (
            "ticks",
            "valid_steps",
            "final_observation_sha256",
            "final_rewards",
            "terminated_count",
            "truncated_count",
            "terminal_observation_checked",
            "rng_statuses",
            "reset_draw_count",
            "tick_draw_count",
        )
    }


def _install_gdown_fuzzy_compat(gdown_module: Any | None = None) -> bool:
    """Allow frozen source that still passes gdown.download(fuzzy=True)."""

    if gdown_module is None:
        try:
            import gdown as gdown_module
        except ModuleNotFoundError:
            return False
    original = gdown_module.download
    if getattr(original, "_ai2_fuzzy_compat", False):
        return True

    def download_without_fuzzy(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("fuzzy", None)
        return original(*args, **kwargs)

    download_without_fuzzy._ai2_fuzzy_compat = True
    gdown_module.download = download_without_fuzzy
    return True


@contextmanager
def _prepared_capture_root(source_root: Path):
    source_root = Path(source_root).resolve()
    if (source_root / "assets").is_dir():
        yield source_root
        return
    environment_assets = source_root / "UTMIST-AI2-main" / "environment" / "assets"
    if not environment_assets.is_dir():
        yield source_root
        return
    with tempfile.TemporaryDirectory(prefix="ai2-capture-root-") as directory:
        prepared = Path(directory)
        try:
            copy_function = os.link
            shutil.copytree(
                source_root / "UTMIST-AI2-main",
                prepared / "UTMIST-AI2-main",
                copy_function=copy_function,
                symlinks=True,
            )
        except OSError:
            shutil.copytree(
                source_root / "UTMIST-AI2-main",
                prepared / "UTMIST-AI2-main",
                symlinks=True,
            )
        (prepared / "assets").symlink_to(environment_assets, target_is_directory=True)
        yield prepared


def _run_original_vs_serial_native(
    case: dict[str, Any], frozen_root: Path, *, scenario: dict[str, Any] | None = None
) -> dict[str, Any]:
    _install_gdown_fuzzy_compat()
    from combat_episode_capture import capture_raw
    from combat_episode_parity import GAMEPLAY_SCOPE, compare_capture
    from native_game_snapshot import GameParityAdapter

    document = capture_raw(frozen_root, seed=case["seed"], actions=case["actions"], scenario=scenario)
    report = compare_capture(document, GameParityAdapter(), scope=GAMEPLAY_SCOPE)
    rng_count = len(document["reset"]["draws"]) + sum(len(row["draws"]) for row in document["ticks"])
    terminal_seen = any(
        row["outputs"]["terminated"] or row["outputs"]["truncated"] for row in document["ticks"]
    )
    return {
        "python_reference_exact": bool(report["exact"]),
        "document": document,
        "compared_ticks": int(report["compared_ticks"]),
        "reset_compared": bool(report["reset_compared"]),
        "first_divergence": report["first_divergence"],
        "rng": {"cursor": rng_count, "count": rng_count, "error": 0 if report["exact"] else 1},
        "terminal_autoreset_evidence": "exact" if terminal_seen and report["exact"] else "not_exercised",
        "coverage": report["coverage"],
    }


def _run_resident_replay_trace(
    case: dict[str, Any],
    document: dict[str, Any],
    backend_name: str,
    *,
    scenario: dict[str, Any] | None = None,
    post_terminal_reset_draws: list[dict[str, float]] | None = None,
    compare_reset: bool = True,
    cuda_stream_mode: str = "current",
) -> dict[str, Any]:
    from native_game_lane_vec_env import GameLaneBackend

    backend = GameLaneBackend(1, backend_name)
    torch = backend.torch
    device = backend.device
    if backend.kind != "cuda" and cuda_stream_mode != "current":
        raise ValueError("non-current stream mode is only valid for native_cuda")
    if cuda_stream_mode not in {"current", "non_default"}:
        raise ValueError("cuda_stream_mode must be current or non_default")
    obs = torch.empty((2, 64), dtype=torch.float32, device=device)
    terminal = torch.empty_like(obs)
    rewards = torch.empty(2, dtype=torch.float32, device=device)
    terminated = torch.empty(2, dtype=torch.bool, device=device)
    truncated = torch.empty(2, dtype=torch.bool, device=device)
    statuses = []
    diffs: list[dict[str, Any]] = []
    reset_diffs: list[dict[str, Any]] = []
    selected_stream = None
    default_stream_id = None
    selected_stream_id = None
    synchronized_before_compare = backend.kind != "cuda"

    def _sync_before_compare() -> None:
        nonlocal synchronized_before_compare
        if selected_stream is not None:
            selected_stream.synchronize()
            synchronized_before_compare = True

    def _stream_scope():
        return torch.cuda.stream(selected_stream) if selected_stream is not None else nullcontext()

    try:
        if backend.kind == "cuda" and cuda_stream_mode == "non_default":
            default_stream = torch.cuda.default_stream(device)
            current_stream = torch.cuda.current_stream(device)
            selected_stream = torch.cuda.Stream(device=device)
            selected_stream.wait_stream(current_stream)
            default_stream_id = int(default_stream.cuda_stream)
            selected_stream_id = int(selected_stream.cuda_stream)
        elif backend.kind == "cuda":
            current_stream = torch.cuda.current_stream(device)
            default_stream_id = int(torch.cuda.default_stream(device).cuda_stream)
            selected_stream_id = int(current_stream.cuda_stream)
        with _stream_scope():
            backend.replay_reset(0, document["reset"]["draws"], obs)
            if scenario:
                players = scenario.get("players", [{}, {}])
                stocks = [
                    int(players[index]["stocks"]) if index < len(players) and "stocks" in players[index] else -1
                    for index in range(2)
                ]
                backend.configure_episode(
                    0,
                    stocks=None if stocks == [-1, -1] else stocks,
                    max_timesteps=scenario.get("max_timesteps"),
                )
        if compare_reset:
            _sync_before_compare()
            reset_diffs = _compare_tensor_to_expected(
                obs,
                _float32_pair(document["reset"]["outputs"], "obs"),
                "reset.outputs.obs",
            )
            diffs.extend(reset_diffs)
        terminal_observation_checked = False
        terminated_count = 0
        truncated_count = 0
        final_reward = [0.0, 0.0]
        for tick, row in enumerate(document["ticks"], 1):
            actions = torch.tensor(row["actions"], dtype=torch.float32, device=device)
            draws = list(row.get("resident_draws", row["draws"]))
            if post_terminal_reset_draws and (
                row["outputs"]["terminated"] or row["outputs"]["truncated"]
            ):
                draws = [*draws, *post_terminal_reset_draws]
            with _stream_scope():
                backend.replay_step(0, actions, draws, obs, terminal, rewards, terminated, truncated)
                status = backend.rng_status(0)
            _sync_before_compare()
            statuses.append({"tick": tick, **status, "expected_count": len(draws)})
            if status != {"mode": 0, "cursor": len(draws), "count": len(draws), "error": 0}:
                diffs.append(
                    {
                        "field": f"tick[{tick}].rng",
                        "kind": "value",
                        "expected": {"mode": 0, "cursor": len(draws), "count": len(draws), "error": 0},
                        "actual": status,
                    }
                )
            diffs.extend(
                _compare_resident_outputs(
                    obs=obs,
                    terminal=terminal,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                    expected_outputs=row["outputs"],
                    tick=tick,
                )
            )
            final_reward = [float(value) for value in rewards.tolist()]
            if bool(terminated.any().item()) or bool(truncated.any().item()):
                terminal_observation_checked = tuple(terminal.shape) == (2, 64)
            terminated_count += int(terminated.to(torch.int32).sum().item())
            truncated_count += int(truncated.to(torch.int32).sum().item())
        backend.drain()
        _sync_before_compare()
        total_draws = len(document["reset"]["draws"]) + sum(status["count"] for status in statuses)
        exact = not diffs
        terminal_seen = any(row["outputs"]["terminated"] or row["outputs"]["truncated"] for row in document["ticks"])
        step_diffs = [diff for diff in diffs if str(diff.get("field", "")).startswith("tick[")]
        obs_diffs = [diff for diff in step_diffs if ".outputs.obs" in str(diff.get("field", ""))]
        reward_diffs = [diff for diff in step_diffs if ".outputs.rewards" in str(diff.get("field", ""))]
        terminated_diffs = [diff for diff in step_diffs if ".outputs.terminated" in str(diff.get("field", ""))]
        truncated_diffs = [diff for diff in step_diffs if ".outputs.truncated" in str(diff.get("field", ""))]
        trace = {
            "mode": "recorded_draw_replay",
            "backend": backend_name,
            "ticks": len(document["ticks"]),
            "valid_steps": [int(value) for value in backend.valid_steps.tolist()],
            "final_observation_sha256": _tensor_sha256(obs),
            "final_rewards": final_reward,
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "terminal_observation_checked": terminal_observation_checked,
            "rng_statuses": statuses,
            "reset_draw_count": len(document["reset"]["draws"]),
            "tick_draw_count": total_draws - len(document["reset"]["draws"]),
            "note": "resident native recorded-draw per-step replay against Python capture",
        }
        result = {
            "resident_cpu_recorded_replay_exact": exact,
            "resident_cpu_generated_rng_recorded": exact,
            "resident_cpu_recorded_replay_trace": trace,
            "resident_cpu_generated_rng_trace": trace,
            "resident_cpu_generated_rng_alias_note": (
                "deprecated compatibility alias; use resident_cpu_recorded_replay_exact/trace"
            ),
            "resident_lanes_replay_step_supported": True,
            "resident_replay_first_divergence": None if exact else diffs[0],
            "rng": {
                "cursor": total_draws if exact else 0,
                "count": total_draws,
                "error": 0 if exact else 1,
            },
            "terminal_autoreset_evidence": (
                "exact" if exact and terminal_observation_checked else "not_exercised" if exact and not terminal_seen else "missing"
            ),
        }
        if backend.kind == "cuda":
            result["resident_cuda_stream"] = {
                "mode": cuda_stream_mode,
                "default_stream": default_stream_id,
                "selected_stream": selected_stream_id,
                "used_non_default_stream": (
                    cuda_stream_mode == "non_default"
                    and selected_stream_id is not None
                    and selected_stream_id != default_stream_id
                ),
                "synchronized_before_compare": synchronized_before_compare,
            }
            result["resident_cuda_observation_checks"] = {
                "reset_observation_exact": not reset_diffs,
                "step_observation_exact": not obs_diffs,
                "reward_exact": not reward_diffs,
                "terminated_exact": not terminated_diffs,
                "truncated_exact": not truncated_diffs,
                "terminal_observation_exact": (not terminal_seen)
                or (terminal_observation_checked and not obs_diffs),
                "reset_observation_sha256": _tensor_sha256(obs if not compare_reset else torch.as_tensor(
                    _float32_pair(document["reset"]["outputs"], "obs"), dtype=torch.float32, device=device
                )),
                "terminal_observation_sha256": _tensor_sha256(terminal if terminal_seen else obs),
            }
        return result
    finally:
        backend.close()


def _run_resident_cpu_generated_rng_trace(case: dict[str, Any]) -> dict[str, Any]:
    from native_game_lane_vec_env import GameLaneBackend

    backend = GameLaneBackend(1, "cpu")
    torch = backend.torch
    obs = torch.empty((2, 64), dtype=torch.float32)
    terminal = torch.empty_like(obs)
    rewards = torch.empty(2, dtype=torch.float32)
    terminated = torch.empty(2, dtype=torch.bool)
    truncated = torch.empty(2, dtype=torch.bool)
    try:
        backend.reset([case["seed"], case["seed"]], obs)
        for action in case["actions"]:
            actions = torch.tensor(action, dtype=torch.float32)
            backend.step(actions, obs, terminal, rewards, terminated, truncated)
        backend.drain()
        return {"resident_cpu_generated_rng_recorded": True}
    finally:
        backend.close()


def _forced_autoreset_case(case: dict[str, Any], kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if kind == "termination":
        scenario = {"players": [{"stocks": 0}, {}]}
    elif kind == "truncation":
        scenario = {"max_timesteps": 1}
    else:
        raise ValueError(f"unknown autoreset kind: {kind}")
    forced = {
        **case,
        "horizon": 1,
        "actions": [case["actions"][0]],
    }
    return forced, scenario


def _run_forced_autoreset_evidence(
    case: dict[str, Any],
    *,
    kind: str,
    capture_root: Path,
    backends: list[str],
) -> dict[str, Any]:
    from native_game_lane_vec_env import GameLaneBackend

    forced, scenario = _forced_autoreset_case(case, kind)
    post_reset = _post_terminal_reset_draws(phase=11 if kind == "termination" else 22)
    serial = _run_original_vs_serial_native(forced, capture_root, scenario=scenario)
    document = serial.pop("document")
    tick = document["ticks"][0]
    if kind == "termination" and tick["outputs"]["terminated"] is not True:
        return {"status": "missing", "reason": "forced termination did not terminate"}
    if kind == "truncation" and tick["outputs"]["truncated"] is not True:
        return {"status": "missing", "reason": "forced truncation did not truncate"}
    if serial.get("python_reference_exact") is not True:
        return {"status": "missing", "reason": serial.get("first_divergence")}
    cpu = _run_resident_replay_trace(
        forced,
        document,
        "cpu",
        scenario=scenario,
        post_terminal_reset_draws=post_reset,
        compare_reset=False,
    )
    if cpu.get("resident_cpu_recorded_replay_exact") is not True:
        return {
            "status": "missing",
            "reason": cpu.get("resident_replay_first_divergence"),
            "same_step_autoreset": False,
            "same_step_reset_obs_exact": False,
            "healthy_subsequent_step": False,
        }
    backend = GameLaneBackend(1, "cpu")
    torch = backend.torch
    obs = torch.empty((2, 64), dtype=torch.float32)
    terminal = torch.empty_like(obs)
    rewards = torch.empty(2, dtype=torch.float32)
    terminated = torch.empty(2, dtype=torch.bool)
    truncated = torch.empty(2, dtype=torch.bool)
    try:
        backend.replay_reset(0, document["reset"]["draws"], obs)
        players = scenario.get("players", [{}, {}])
        stocks = [
            int(players[index]["stocks"]) if index < len(players) and "stocks" in players[index] else -1
            for index in range(2)
        ]
        backend.configure_episode(
            0,
            stocks=None if stocks == [-1, -1] else stocks,
            max_timesteps=scenario.get("max_timesteps"),
        )
        actions = torch.tensor(tick["actions"], dtype=torch.float32)
        backend.replay_step(0, actions, [*tick["draws"], *post_reset], obs, terminal, rewards, terminated, truncated)
        same_step = bool(terminated.any().item() or truncated.any().item())
        expected_reset = torch.empty_like(obs)
        expected_backend = GameLaneBackend(1, "cpu")
        try:
            expected_backend.replay_reset(0, post_reset, expected_reset)
            reset_obs_diffs = _compare_tensor_to_expected(
                obs,
                expected_reset.detach().cpu().contiguous().numpy(),
                f"{kind}.same_step_reset_obs",
            )
        finally:
            expected_backend.close()
        same_step_reset_obs_exact = not reset_obs_diffs
        followup_actions = torch.zeros((2, 10), dtype=torch.float32)
        backend.replay_step(0, followup_actions, [], obs, terminal, rewards, terminated, truncated)
        healthy = (
            backend.rng_status(0) == {"mode": 0, "cursor": 0, "count": 0, "error": 0}
            and not bool(terminated.any().item())
            and not bool(truncated.any().item())
        )
    finally:
        backend.close()
    cuda_exact = None
    cuda_note = "native_cuda backend not requested"
    if "native_cuda" in backends:
        try:
            cuda = _run_resident_replay_trace(
                forced,
                document,
                "cuda",
                scenario=scenario,
                post_terminal_reset_draws=post_reset,
                compare_reset=False,
                cuda_stream_mode="non_default",
            )
        except Exception as exc:
            cuda_exact = False
            cuda_note = f"{type(exc).__name__}: {exc}"
        else:
            cuda_exact = cuda.get("resident_cpu_recorded_replay_exact") is True
            cuda_note = None if cuda_exact else cuda.get("resident_replay_first_divergence")
    return {
        "status": "exact" if same_step and same_step_reset_obs_exact and healthy else "missing",
        "same_step_autoreset": same_step,
        "same_step_reset_obs_exact": same_step_reset_obs_exact,
        "same_step_reset_obs_first_difference": None if same_step_reset_obs_exact else reset_obs_diffs[0],
        "healthy_subsequent_step": healthy,
        "python_reference_exact": True,
        "resident_cpu_recorded_replay_exact": True,
        "resident_cuda_recorded_replay_exact": cuda_exact,
        "resident_cuda_note": cuda_note,
        "post_terminal_reset_draw_count": len(post_reset),
        "terminal_tick_draw_count": len(tick["draws"]),
        "terminal_replay_draw_count": len(tick["draws"]) + len(post_reset),
    }


def run_suite(
    *,
    frozen_root: Path,
    backends: list[str],
    actions: list[int],
    seeds: list[int],
    horizon: int,
) -> dict[str, Any]:
    suite = build_fixed_action_suite(seeds=seeds, horizon=horizon, actions=actions)
    replay_supported = resident_replay_step_supported()
    case_results = []
    with _prepared_capture_root(frozen_root) as capture_root:
        for case_index, case in enumerate(suite["cases"]):
            result = {
                key: case[key]
                for key in (
                    "seed",
                    "action_index",
                    "key",
                    "meaning",
                    "trace_required",
                    "high_risk",
                    "horizon",
                    "action_specific_fields",
                )
            }
            if "python" in backends or "native_cpu" in backends:
                try:
                    python_result = _run_original_vs_serial_native(case, capture_root)
                    document = python_result.pop("document")
                    result.update(python_result)
                except ModuleNotFoundError as exc:
                    document = None
                    result.update(
                        {
                            "python_reference_exact": False,
                            "compared_ticks": 0,
                            "reset_compared": False,
                            "first_divergence": str(exc),
                            "blocker": "missing_capture_dependency",
                            "terminal_autoreset_evidence": "missing",
                            "rng": {"cursor": 0, "count": 0, "error": 1},
                        }
                    )
                except Exception as exc:
                    document = None
                    result.update(
                        {
                            "python_reference_exact": False,
                            "compared_ticks": 0,
                            "reset_compared": False,
                            "first_divergence": f"{type(exc).__name__}: {exc}",
                            "blocker": "python_reference_replay_failed",
                            "terminal_autoreset_evidence": "missing",
                            "rng": {"cursor": 0, "count": 0, "error": 1},
                        }
                    )
            else:
                document = None
                result.update(
                    {
                        "python_reference_exact": None,
                        "compared_ticks": 0,
                        "reset_compared": False,
                        "first_divergence": "python/native_cpu backend not requested",
                        "terminal_autoreset_evidence": "missing",
                        "rng": {"cursor": 0, "count": 0, "error": 1},
                    }
                )
            if "native_cpu" in backends:
                try:
                    if document is None:
                        raise RuntimeError("resident replay requires original Python capture document")
                    cpu_result = _run_resident_replay_trace(case, document, "cpu")
                    result.update(cpu_result)
                    result["resident_cpu_recorded_replay_exact"] = cpu_result.get(
                        "resident_cpu_recorded_replay_exact", False
                    )
                    result["native_cpu_cuda_exact"] = None
                    result["native_cpu_cuda_exact_note"] = "native_cuda backend not requested"
                    if "native_cuda" in backends:
                        try:
                            cuda_result = _run_resident_replay_trace(
                                case,
                                document,
                                "cuda",
                                cuda_stream_mode="non_default",
                            )
                        except Exception as cuda_exc:
                            result["native_cpu_cuda_exact"] = False
                            result["native_cpu_cuda_exact_note"] = f"{type(cuda_exc).__name__}: {cuda_exc}"
                            result["resident_cuda_recorded_replay_trace"] = None
                            result["resident_cuda_replay_trace"] = None
                        else:
                            result["resident_cuda_recorded_replay_trace"] = cuda_result.get(
                                "resident_cpu_recorded_replay_trace",
                                cuda_result.get("resident_cpu_generated_rng_trace"),
                            )
                            result["resident_cuda_replay_trace"] = result["resident_cuda_recorded_replay_trace"]
                            result["resident_cuda_recorded_replay_exact"] = cuda_result.get(
                                "resident_cpu_recorded_replay_exact"
                            )
                            result["resident_cuda_stream"] = cuda_result.get("resident_cuda_stream")
                            result["resident_cuda_observation_checks"] = cuda_result.get(
                                "resident_cuda_observation_checks"
                            )
                            cpu_trace = _resident_trace_comparable(
                                result.get(
                                    "resident_cpu_recorded_replay_trace",
                                    result.get("resident_cpu_generated_rng_trace"),
                                )
                            )
                            cuda_trace = _resident_trace_comparable(
                                cuda_result.get(
                                    "resident_cpu_recorded_replay_trace",
                                    cuda_result.get("resident_cpu_generated_rng_trace"),
                                )
                            )
                            result["native_cpu_cuda_exact"] = (
                                result["resident_cpu_generated_rng_recorded"] is True
                                and cuda_result.get("resident_cpu_generated_rng_recorded") is True
                                and cpu_trace == cuda_trace
                            )
                            result["native_cpu_cuda_exact_note"] = (
                                None
                                if result["native_cpu_cuda_exact"]
                                else {
                                    "cuda_first_divergence": cuda_result.get("resident_replay_first_divergence"),
                                    "cpu_trace": cpu_trace,
                                    "cuda_trace": cuda_trace,
                                }
                            )
                except Exception as exc:
                    result.update(
                        {
                            "resident_cpu_recorded_replay_exact": False,
                            "resident_cpu_generated_rng_recorded": False,
                            "resident_lanes_replay_step_supported": replay_supported,
                            "resident_cpu_blocker": "resident_cpu_recorded_replay_failed",
                            "resident_cpu_first_divergence": f"{type(exc).__name__}: {exc}",
                            "native_cpu_cuda_exact": None,
                            "native_cpu_cuda_exact_note": "resident CPU recorded replay failed before CUDA comparison",
                        }
                    )
            else:
                result["resident_cpu_recorded_replay_exact"] = False
                result["resident_cpu_generated_rng_recorded"] = False
                result["resident_lanes_replay_step_supported"] = replay_supported
                result["native_cpu_cuda_exact"] = None
                result["native_cpu_cuda_exact_note"] = "native_cpu backend not requested"
            if case_index == 0 and document is not None and result.get("resident_cpu_recorded_replay_exact") is True:
                for kind in FORCED_AUTORESET_KINDS:
                    try:
                        result[f"resident_{kind}_autoreset"] = _run_forced_autoreset_evidence(
                            case,
                            kind=kind,
                            capture_root=capture_root,
                            backends=backends,
                        )
                    except Exception as exc:
                        result[f"resident_{kind}_autoreset"] = {
                            "status": "missing",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "same_step_autoreset": False,
                            "same_step_reset_obs_exact": False,
                            "healthy_subsequent_step": False,
                        }
            result["resident_lane_claim"] = (
                "recorded-draw resident replay compared against Python capture"
                if result.get("resident_cpu_recorded_replay_exact") is True
                else "resident recorded-draw replay unavailable or failed"
            )
            case_results.append(result)
    return build_receipt(
        cases=case_results,
        backends=backends,
        expected_actions=actions,
        expected_seeds=seeds,
        horizon=horizon,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode()
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.write(b"\n")
        handle.flush()
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(root: Path | None = None) -> dict[str, dict[str, Any]]:
    root = (root or Path(__file__).resolve().parent).resolve()
    hashes: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_HASH_FILES:
        path = root / relative
        if path.is_file():
            hashes[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        else:
            hashes[relative] = {"missing": True}
    return hashes


def _cuda_skip_status(receipt: dict[str, Any]) -> dict[str, Any]:
    backends = receipt.get("backends", [])
    if "native_cuda" not in backends:
        return {"status": "not_requested"}
    if receipt.get("native_cpu_cuda_exact") is True:
        return {"status": "not_skipped"}
    notes = []
    for case in receipt.get("cases", []):
        note = case.get("native_cpu_cuda_exact_note")
        if note is not None:
            notes.append(note)
        trace = case.get("resident_cuda_recorded_replay_trace")
        if trace is None and case.get("native_cpu_cuda_exact") is False:
            notes.append("resident_cuda_recorded_replay_trace missing")
    blockers = receipt.get("blockers", []) + receipt.get("promotion_blockers", [])
    for blocker in blockers:
        text = json.dumps(blocker, sort_keys=True, default=str)
        if "CUDA unavailable" in text or "cuda unavailable" in text or "torch.cuda.is_available" in text:
            return {"status": "blocked_no_cuda", "evidence": blocker}
    joined = " ".join(str(note) for note in notes)
    if "CUDA unavailable" in joined or "cuda unavailable" in joined or "nvcc unavailable" in joined:
        return {"status": "blocked_no_cuda", "evidence": joined}
    return {"status": "blocked_cuda_failed", "evidence": notes[:3]}


def build_manifest(*, receipt: dict[str, Any], output: Path, argv: list[str]) -> dict[str, Any]:
    report = output / "trace-report.json"
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": receipt.get("status"),
        "promotion_status": receipt.get("promotion_status"),
        "created_unix": time.time(),
        "command": ["python", "-B", "fixed_action_trace_parity.py", *argv],
        "artifacts": {
            "trace_report": str(report),
            "manifest": str(output / "resident-stream-gate-manifest.json"),
        },
        "backends": receipt.get("backends"),
        "expected_actions": receipt.get("expected_actions"),
        "expected_seeds": receipt.get("expected_seeds"),
        "horizon": receipt.get("horizon"),
        "cuda_skip": _cuda_skip_status(receipt),
        "source_hashes": _source_hashes(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", required=True, type=Path)
    parser.add_argument("--backends", required=True, type=lambda value: value.split(","))
    parser.add_argument("--actions", required=True, type=_parse_int_csv)
    parser.add_argument("--seeds", required=True, type=_parse_int_csv)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    receipt = run_suite(
        frozen_root=args.frozen_root,
        backends=args.backends,
        actions=args.actions,
        seeds=args.seeds,
        horizon=args.horizon,
    )
    _write_json_atomic(args.output / "trace-report.json", receipt)
    _write_json_atomic(
        args.output / "resident-stream-gate-manifest.json",
        build_manifest(receipt=receipt, output=args.output, argv=raw_argv),
    )
    print(json.dumps({"status": receipt["status"], "report": str(args.output / "trace-report.json")}, sort_keys=True))
    return 0 if not receipt["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
