"""Fail-closed benchmark receipt contract for AI2 training comparisons."""

from __future__ import annotations

import json
import math
from typing import Any


SCHEMA_VERSION = "ai2.benchmark_contract.v1"
HEADLINE_UNIT = "agent_transition"
END_TO_END_STAGES = (
    "policy_inference",
    "action_sampling",
    "environment_step",
    "rollout_storage",
    "returns_and_advantages",
    "ppo_update",
    "device_synchronization",
)


class ContractError(ValueError):
    """Raised when a benchmark receipt cannot support a safe comparison."""


_REQUIRED_PATHS = (
    "schema_version",
    "system.hardware.accelerator_model",
    "system.hardware.accelerator_count",
    "workload.unit",
    "workload.timing_scope",
    "workload.included_stages",
    "workload.warmup_policy",
    "workload.synchronization",
    "workload.overlap.mode",
    "workload.overlap.evidence",
    "workload.environment.env_id",
    "workload.environment.observation_contract",
    "workload.environment.action_contract",
    "workload.environment.tick_semantics",
    "workload.environment.terminal_bootstrap",
    "workload.model.architecture",
    "workload.model.parameter_count",
    "workload.model.precision",
    "workload.rollout.num_envs",
    "workload.rollout.horizon",
    "workload.rollout.transitions_per_batch",
    "algorithm.optimizer",
    "algorithm.learning_rate",
    "algorithm.gamma",
    "algorithm.gae_lambda",
    "algorithm.clip_range",
    "algorithm.entropy_coef",
    "algorithm.value_coef",
    "algorithm.max_grad_norm",
    "algorithm.sample_reuse_epochs",
    "algorithm.minibatches",
    "algorithm.ppo_epochs",
    "algorithm.replay_semantics",
    "algorithm.actor_staleness",
    "algorithm.advantage_normalization",
    "algorithm.terminal_bootstrap",
    "timing.elapsed_source",
    "timing.elapsed_seconds",
    "measurement.agent_transitions",
)

_AMBIGUOUS_COUNTERS = (
    "steps",
    "sps",
    "game_ticks",
    "episodes",
    "optimizer_calls",
)

_THROUGHPUT_PATHS = (
    "workload.timing_scope",
    "workload.included_stages",
    "workload.warmup_policy",
    "workload.synchronization",
    "workload.overlap.mode",
    "workload.overlap.evidence",
    "system.hardware.accelerator_model",
    "system.hardware.accelerator_count",
    "system.hardware.cpu_model",
    "workload.environment.env_id",
    "workload.environment.observation_contract",
    "workload.environment.action_contract",
    "workload.environment.tick_semantics",
    "workload.environment.terminal_bootstrap",
    "workload.model.architecture",
    "workload.model.parameter_count",
    "workload.model.precision",
    "workload.rollout.num_envs",
    "workload.rollout.horizon",
    "workload.rollout.transitions_per_batch",
    "algorithm.minibatches",
    "algorithm.ppo_epochs",
    "algorithm.sample_reuse_epochs",
)

_LEARNING_PATHS = (
    "workload.environment.env_id",
    "workload.environment.observation_contract",
    "workload.environment.action_contract",
    "workload.environment.tick_semantics",
    "workload.environment.terminal_bootstrap",
    "workload.model.architecture",
    "workload.model.parameter_count",
    "workload.model.precision",
    "workload.rollout.horizon",
    "workload.rollout.transitions_per_batch",
    "algorithm.optimizer",
    "algorithm.learning_rate",
    "algorithm.gamma",
    "algorithm.gae_lambda",
    "algorithm.clip_range",
    "algorithm.entropy_coef",
    "algorithm.value_coef",
    "algorithm.max_grad_norm",
    "algorithm.sample_reuse_epochs",
    "algorithm.minibatches",
    "algorithm.ppo_epochs",
    "algorithm.replay_semantics",
    "algorithm.actor_staleness",
    "algorithm.advantage_normalization",
    "algorithm.terminal_bootstrap",
)

_PROMOTION_EVIDENCE = (
    "correctness.env_trace_parity",
    "correctness.rollout_parity",
    "correctness.fixed_transition_quality",
    "correctness.fixed_wall_clock_quality",
)

_ALLOWED_TIMING_SCOPES = {
    "collection",
    "end_to_end",
    "env_only",
    "time_to_quality",
    "update",
}
_ALLOWED_OVERLAP_MODES = {"none", "async"}

_POSITIVE_INTEGER_PATHS = (
    "system.hardware.accelerator_count",
    "workload.model.parameter_count",
    "workload.rollout.num_envs",
    "workload.rollout.horizon",
    "workload.rollout.transitions_per_batch",
    "algorithm.sample_reuse_epochs",
    "algorithm.minibatches",
    "algorithm.ppo_epochs",
    "measurement.agent_transitions",
)


def validate_receipt(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate one receipt and return a normalized JSON-safe deep copy."""
    receipt = _json_safe_copy(raw)
    for path in _REQUIRED_PATHS:
        _require_path(receipt, path)

    if receipt["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {SCHEMA_VERSION}")

    for path in _POSITIVE_INTEGER_PATHS:
        _positive_int(_get(receipt, path), path)

    unit = _get(receipt, "workload.unit")
    if unit != HEADLINE_UNIT:
        raise ContractError(
            f"workload.unit must be {HEADLINE_UNIT}; never use steps, SPS, ticks, "
            "episodes, or optimizer calls as headline transitions"
        )
    measurement = receipt["measurement"]
    measurement_keys = {_counter_key(str(key)) for key in measurement}
    for counter in _AMBIGUOUS_COUNTERS:
        if _counter_key(counter) in measurement_keys:
            raise ContractError(
                f"measurement.{counter} is ambiguous; use measured {HEADLINE_UNIT}"
            )

    timing_scope = _get(receipt, "workload.timing_scope")
    if timing_scope not in _ALLOWED_TIMING_SCOPES:
        raise ContractError(
            f"workload.timing_scope must be one of {sorted(_ALLOWED_TIMING_SCOPES)}"
        )

    stages = _get(receipt, "workload.included_stages")
    if not isinstance(stages, list) or not stages:
        raise ContractError("workload.included_stages must be a non-empty list")
    if not all(isinstance(stage, str) for stage in stages):
        raise ContractError("workload.included_stages must contain only strings")
    if len(set(stages)) != len(stages):
        raise ContractError("workload.included_stages must be unique")
    if timing_scope == "end_to_end":
        if set(stages) != set(END_TO_END_STAGES):
            raise ContractError(
                "workload.included_stages must contain the canonical end-to-end stages"
            )

    if _get(receipt, "timing.elapsed_source") != "wall_clock":
        raise ContractError("timing.elapsed_source must be wall_clock")
    _validate_stage_seconds(receipt["timing"])

    elapsed = _positive_finite(_get(receipt, "timing.elapsed_seconds"), "elapsed_seconds")
    transitions = _get(receipt, "measurement.agent_transitions")
    derived_rate = transitions / elapsed
    claimed_rate = receipt["measurement"].get("claimed_rate")
    if claimed_rate is not None:
        claimed_rate = _positive_finite(claimed_rate, "claimed_rate")
        if not math.isclose(claimed_rate, derived_rate, rel_tol=1e-6, abs_tol=1e-9):
            raise ContractError(
                f"measurement.claimed_rate disagrees with derived rate {derived_rate}"
            )
    receipt["measurement"]["derived_rate"] = derived_rate

    overlap_mode = _get(receipt, "workload.overlap.mode")
    if overlap_mode not in _ALLOWED_OVERLAP_MODES:
        raise ContractError(
            f"workload.overlap.mode must be one of {sorted(_ALLOWED_OVERLAP_MODES)}"
        )
    overlap_evidence = _get(receipt, "workload.overlap.evidence")
    if not isinstance(overlap_evidence, list) or not overlap_evidence:
        raise ContractError("workload.overlap.evidence must be a non-empty list")
    if overlap_mode == "async":
        if not {"cuda_trace", "event_timeline"}.intersection(overlap_evidence):
            raise ContractError(
                "workload.overlap.evidence requires cuda_trace or event_timeline"
            )

    return receipt


def compare_receipts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare two validated-compatible receipts without guessing missing semantics."""
    normalized_left = validate_receipt(left)
    normalized_right = validate_receipt(right)

    throughput_blockers = _mismatched_paths(
        normalized_left, normalized_right, _THROUGHPUT_PATHS
    )
    learning_blockers = _mismatched_paths(
        normalized_left, normalized_right, _LEARNING_PATHS
    )

    throughput_comparable = not throughput_blockers
    learning_behavior_comparable = not learning_blockers
    promotion_blockers = []
    if not throughput_comparable:
        promotion_blockers.append("throughput_comparable")
    if not learning_behavior_comparable:
        promotion_blockers.append("learning_behavior_comparable")
    for path in _PROMOTION_EVIDENCE:
        if _get(normalized_right, path, default=False) is not True:
            promotion_blockers.append(path)

    promotion_eligible = not promotion_blockers
    left_rate = _get(normalized_left, "measurement.derived_rate")
    right_rate = _get(normalized_right, "measurement.derived_rate")

    return {
        "throughput_comparable": throughput_comparable,
        "learning_behavior_comparable": learning_behavior_comparable,
        "promotion_eligible": promotion_eligible,
        "speedup_ratio": right_rate / left_rate if throughput_comparable else None,
        "left_agent_transitions_per_second": left_rate,
        "right_agent_transitions_per_second": right_rate,
        "throughput_blockers": throughput_blockers,
        "learning_blockers": learning_blockers,
        "promotion_blockers": promotion_blockers,
    }


def _json_safe_copy(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(raw, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ContractError("receipt must be JSON-safe with finite numeric values") from exc


def _positive_finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{name} must be numeric")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ContractError(f"{name} must be finite")
    if as_float <= 0:
        raise ContractError(f"{name} must be positive")
    return as_float


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{name} must be a positive integer")
    if value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _counter_key(value: str) -> str:
    return value.casefold().replace("_", "").replace("-", "").replace(" ", "")


def _validate_stage_seconds(timing: dict[str, Any]) -> None:
    stage_seconds = timing.get("stage_seconds")
    if stage_seconds is None:
        return
    if not isinstance(stage_seconds, dict):
        raise ContractError("timing.stage_seconds must be an object")
    for stage, seconds in stage_seconds.items():
        if not isinstance(stage, str):
            raise ContractError("timing.stage_seconds keys must be strings")
        _positive_finite(seconds, f"timing.stage_seconds.{stage}")


def _require_path(receipt: dict[str, Any], path: str) -> None:
    marker = object()
    if _get(receipt, path, default=marker) is marker:
        raise ContractError(f"missing required field {path}")


def _get(receipt: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = receipt
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _mismatched_paths(
    left: dict[str, Any], right: dict[str, Any], paths: tuple[str, ...]
) -> list[str]:
    blockers = []
    for path in paths:
        marker = object()
        left_value = _get(left, path, default=marker)
        right_value = _get(right, path, default=marker)
        if left_value != right_value:
            blockers.append(path)
    return blockers
