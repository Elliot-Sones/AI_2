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
AI2_BOX10_BUTTONS = (
    "w/up",
    "a/left",
    "s/down",
    "d/right",
    "space/jump",
    "h/pickup_drop",
    "l/dodge_backdash",
    "j/light",
    "k/heavy",
    "g/taunt",
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
    "workload.quality_target",
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
    "workload.quality_target",
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
_ALLOWED_OVERLAP_EVIDENCE = {"cuda_trace", "event_timeline", "not_applicable"}
_ASYNC_OVERLAP_EVIDENCE = {"cuda_trace", "event_timeline"}
_QUALITY_TARGET_DIRECTIONS = {"at_least", "at_most"}
_QUALITY_TARGET_AGGREGATIONS = {"mean", "min", "max"}

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
    _validate_ppo_hyperparameters(receipt)

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
    if timing_scope == "time_to_quality":
        _validate_quality_target(receipt["workload"].get("quality_target"))

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

    _require_dict(_get(receipt, "workload.warmup_policy"), "workload.warmup_policy")
    _require_dict(_get(receipt, "workload.synchronization"), "workload.synchronization")
    _require_dict(_get(receipt, "workload.overlap"), "workload.overlap")
    overlap_mode = _get(receipt, "workload.overlap.mode")
    if overlap_mode not in _ALLOWED_OVERLAP_MODES:
        raise ContractError(
            f"workload.overlap.mode must be one of {sorted(_ALLOWED_OVERLAP_MODES)}"
        )
    overlap_evidence = _get(receipt, "workload.overlap.evidence")
    if not isinstance(overlap_evidence, list) or not overlap_evidence:
        raise ContractError("workload.overlap.evidence must be a non-empty list")
    if not all(isinstance(item, str) for item in overlap_evidence):
        raise ContractError("workload.overlap.evidence must contain only strings")
    unknown_evidence = set(overlap_evidence) - _ALLOWED_OVERLAP_EVIDENCE
    if unknown_evidence:
        raise ContractError(
            "workload.overlap.evidence contains unknown evidence tokens: "
            f"{sorted(unknown_evidence)}"
        )
    if overlap_mode == "async":
        if not _ASYNC_OVERLAP_EVIDENCE.intersection(overlap_evidence):
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


def ai2_native_receipt(result: dict[str, Any], backend_name: str) -> dict[str, Any]:
    """Convert one passed raw AI2 native benchmark backend into a canonical receipt."""
    if result.get("status") != "passed":
        raise ContractError("top-level benchmark status must be passed")
    if not _non_empty_string(backend_name):
        raise ContractError("backend_name must be a non-empty string")

    config = _require_dict(result.get("config"), "config")
    provenance = _require_dict(result.get("provenance"), "provenance")
    _validate_native_provenance(config, provenance)
    backend = _select_backend(result, backend_name)
    _validate_native_backend(backend_name, backend, config)

    summary = backend["summary"]
    parallel_agents = config["games"] * 2
    transitions_per_batch = parallel_agents * config["n_steps"]
    minibatches = transitions_per_batch // config["batch_size"]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "system": {
            "hardware": {
                "accelerator_model": provenance["gpu"],
                "accelerator_count": 1,
                "cpu_model": provenance.get("system", "unknown"),
                "cpu_count": provenance.get("cpu_count"),
                "cuda_version": provenance.get("cuda_version"),
            },
            "software": {
                "python": provenance.get("python"),
                "packages": provenance.get("packages", {}),
                "policy_device": provenance.get("policy_device"),
                "matmul_tf32": provenance.get("matmul_tf32"),
            },
        },
        "workload": {
            "unit": HEADLINE_UNIT,
            "timing_scope": "end_to_end",
            "included_stages": list(END_TO_END_STAGES),
            "quality_target": None,
            "warmup_policy": {
                "mode": "discard_first_n_rollouts",
                "rollouts": config["warmup_rollouts"],
            },
            "synchronization": {
                "device": config["device"],
                "method": "torch.cuda.synchronize around collection and PPO update",
            },
            "overlap": {"mode": "none", "evidence": ["not_applicable"]},
            "environment": {
                "env_id": "WarehouseBrawl/raw_native_fullgame",
                "semantics_hash": provenance["environment_sha256"],
                "attack_assets_sha256": provenance["attack_assets_sha256"],
                "observation_contract": {
                    "type": "box",
                    "shape": [256],
                    "dtype": "float32",
                    "frames": 4,
                    "per_player_features": 64,
                    "stack_order": "oldest_to_newest",
                },
                "action_contract": {
                    "type": "box",
                    "shape": [10],
                    "dtype": "float32",
                    "range": [-1.0, 1.0],
                    "button_threshold": 0.5,
                    "buttons": list(AI2_BOX10_BUTTONS),
                },
                "tick_semantics": "one_environment_tick_yields_two_agent_transitions",
                "terminal_bootstrap": "terminal_observation_with_gae_timeout_bootstrap",
                "player_order": provenance.get("player_order"),
                "game_seed_rule": provenance.get("game_seed_rule"),
            },
            "model": {
                "architecture": {
                    "type": "stable_baselines3_actor_critic_mlp_policy",
                    "observation_shape": [256],
                    "action_shape": [10],
                    "policy_layers": list(config["net_arch"]),
                    "value_layers": list(config["net_arch"]),
                    "activation": "torch.nn.Tanh",
                    "distribution": "diagonal_gaussian_box_actions",
                    "shared_policy": "both_players",
                },
                "parameter_count": backend["initial_policy"]["parameter_count"],
                "precision": "fp32",
            },
            "rollout": {
                "num_envs": parallel_agents,
                "horizon": config["n_steps"],
                "transitions_per_batch": transitions_per_batch,
                "measured_rollouts": config["measured_rollouts"],
                "warmup_rollouts": config["warmup_rollouts"],
            },
            "source_provenance": {
                "sources": provenance["sources"],
                "ppo_update": provenance["ppo_update"],
                "benchmark_schema": result.get("schema"),
                "backend": backend_name,
                "exclusions": provenance.get("exclusions"),
                "limitation": provenance.get("limitation"),
            },
        },
        "algorithm": {
            "optimizer": "adam",
            "learning_rate": config["learning_rate"],
            "gamma": config["gamma"],
            "gae_lambda": config["gae_lambda"],
            "clip_range": config["clip_range"],
            "entropy_coef": config["ent_coef"],
            "value_coef": config["vf_coef"],
            "max_grad_norm": config["max_grad_norm"],
            "sample_reuse_epochs": config["n_epochs"],
            "minibatches": minibatches,
            "ppo_epochs": config["n_epochs"],
            "replay_semantics": "on_policy_no_replay",
            "actor_staleness": "synchronous",
            "advantage_normalization": "per_minibatch",
            "terminal_bootstrap": "terminal_observation_with_gae_timeout_bootstrap",
            "update_mode": config.get("update_mode", "sb3"),
        },
        "timing": {
            "elapsed_source": "wall_clock",
            "elapsed_seconds": summary["total_seconds"],
            "stage_seconds": {
                "collection": summary["collection_seconds"],
                "ppo_update": summary["update_seconds"],
            },
        },
        "measurement": {
            "agent_transitions": summary["player_transitions"],
            "claimed_rate": summary["total_player_transitions_per_second"],
        },
        "correctness": {
            "env_trace_parity": False,
            "rollout_parity": False,
            "fixed_transition_quality": False,
            "fixed_wall_clock_quality": False,
        },
        "correctness_evidence": {
            "env_trace_parity": {
                "status": "not_run",
                "reason": "raw benchmark sidecar does not execute parity gates",
            },
            "rollout_parity": {
                "status": "not_run",
                "reason": "raw benchmark sidecar does not execute rollout parity",
            },
            "fixed_transition_quality": {
                "status": "not_run",
                "reason": "raw benchmark sidecar is speed evidence, not mastery evidence",
            },
            "fixed_wall_clock_quality": {
                "status": "not_run",
                "reason": "raw benchmark sidecar is speed evidence, not mastery evidence",
            },
        },
    }
    return validate_receipt(receipt)


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


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return value


def _validate_ppo_hyperparameters(receipt: dict[str, Any]) -> None:
    _bounded_number(_get(receipt, "algorithm.learning_rate"), "algorithm.learning_rate", lower=0.0)
    _bounded_number(_get(receipt, "algorithm.gamma"), "algorithm.gamma", lower=0.0, upper=1.0, include_lower=True)
    _bounded_number(
        _get(receipt, "algorithm.gae_lambda"),
        "algorithm.gae_lambda",
        lower=0.0,
        upper=1.0,
        include_lower=True,
    )
    _bounded_number(_get(receipt, "algorithm.clip_range"), "algorithm.clip_range", lower=0.0, upper=1.0)
    _bounded_number(
        _get(receipt, "algorithm.entropy_coef"),
        "algorithm.entropy_coef",
        lower=0.0,
        include_lower=True,
    )
    _bounded_number(
        _get(receipt, "algorithm.value_coef"),
        "algorithm.value_coef",
        lower=0.0,
        include_lower=True,
    )
    _bounded_number(_get(receipt, "algorithm.max_grad_norm"), "algorithm.max_grad_norm", lower=0.0)


def _bounded_number(
    value: Any,
    name: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    include_lower: bool = False,
    include_upper: bool = True,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{name} must be numeric")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ContractError(f"{name} must be finite")
    if lower is not None:
        if include_lower:
            invalid = as_float < lower
        else:
            invalid = as_float <= lower
        if invalid:
            comparator = ">=" if include_lower else ">"
            raise ContractError(f"{name} must be {comparator} {lower}")
    if upper is not None:
        if include_upper:
            invalid = as_float > upper
        else:
            invalid = as_float >= upper
        if invalid:
            comparator = "<=" if include_upper else "<"
            raise ContractError(f"{name} must be {comparator} {upper}")
    return as_float


def _validate_quality_target(value: Any) -> None:
    if not isinstance(value, dict):
        raise ContractError("workload.quality_target must be an object")
    required = (
        "metric",
        "threshold",
        "direction",
        "evaluation_protocol",
        "aggregation",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ContractError(f"workload.quality_target missing {missing}")
    if not _non_empty_string(value["metric"]):
        raise ContractError("workload.quality_target.metric must be a non-empty string")
    threshold = value["threshold"]
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ContractError("workload.quality_target.threshold must be numeric")
    if not math.isfinite(float(threshold)):
        raise ContractError("workload.quality_target.threshold must be finite")
    if value["direction"] not in _QUALITY_TARGET_DIRECTIONS:
        raise ContractError(
            "workload.quality_target.direction must be at_least or at_most"
        )
    if not _non_empty_string(value["evaluation_protocol"]):
        raise ContractError(
            "workload.quality_target.evaluation_protocol must be a non-empty string"
        )
    if value["aggregation"] not in _QUALITY_TARGET_AGGREGATIONS:
        raise ContractError(
            "workload.quality_target.aggregation must be one of "
            f"{sorted(_QUALITY_TARGET_AGGREGATIONS)}"
        )


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


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


def _select_backend(result: dict[str, Any], backend_name: str) -> dict[str, Any]:
    backends = result.get("backends")
    if not isinstance(backends, list):
        raise ContractError("backends must be a list")
    matches = [backend for backend in backends if isinstance(backend, dict) and backend.get("backend") == backend_name]
    if len(matches) != 1:
        raise ContractError(f"backend {backend_name} must appear exactly once")
    return matches[0]


def _validate_native_provenance(config: dict[str, Any], provenance: dict[str, Any]) -> None:
    if provenance.get("config") != config:
        raise ContractError("config must match provenance.config")
    for path in (
        "environment_sha256",
        "attack_assets_sha256",
        "gpu",
        "workload",
        "exclusions",
        "limitation",
    ):
        if not _non_empty_string(provenance.get(path)):
            raise ContractError(f"provenance.{path} must be a non-empty string")
    sources = _require_dict(provenance.get("sources"), "provenance.sources")
    required_sources = (
        "benchmark_native_combat_training.py",
        "native_game_lane_vec_env.py",
        "tensor_ppo.py",
        "tensor_rollout_buffer.py",
        "native/navigation_spaces.json",
    )
    missing = [source for source in required_sources if not _non_empty_string(sources.get(source))]
    if missing:
        raise ContractError(f"provenance.sources missing {missing}")
    ppo_update = _require_dict(provenance.get("ppo_update"), "provenance.ppo_update")
    if ppo_update.get("mode") != config.get("update_mode"):
        raise ContractError("provenance.ppo_update.mode must match config.update_mode")
    _require_dict(ppo_update.get("source_guard"), "provenance.ppo_update.source_guard")


def _validate_native_backend(
    backend_name: str, backend: dict[str, Any], config: dict[str, Any]
) -> None:
    if backend.get("status") != "passed":
        raise ContractError(f"backend {backend_name} status must be passed")
    if backend.get("parameters_changed") is not True:
        raise ContractError("parameters changed must be true")
    initial = _require_dict(backend.get("initial_policy"), "initial_policy")
    final = _require_dict(backend.get("final_policy"), "final_policy")
    _positive_int(initial.get("parameter_count"), "initial_policy.parameter_count")
    if final.get("parameter_count") != initial.get("parameter_count"):
        raise ContractError("final_policy.parameter_count must match initial_policy.parameter_count")
    if not _non_empty_string(initial.get("sha256")) or not _non_empty_string(final.get("sha256")):
        raise ContractError("policy sha256 fields must be non-empty")
    if final.get("sha256") == initial.get("sha256"):
        raise ContractError("policy parameters did not change")

    for key in (
        "games",
        "n_steps",
        "batch_size",
        "n_epochs",
        "warmup_rollouts",
        "measured_rollouts",
    ):
        _positive_int(config.get(key), f"config.{key}")
    for key in (
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
    ):
        _positive_finite(config.get(key), f"config.{key}")
    if config["batch_size"] <= 1:
        raise ContractError("config.batch_size must exceed one")
    transitions_per_rollout = config["games"] * config["n_steps"] * 2
    if transitions_per_rollout % config["batch_size"]:
        raise ContractError("config.batch_size must divide transitions per rollout")
    net_arch = config.get("net_arch")
    if not isinstance(net_arch, (list, tuple)) or not net_arch:
        raise ContractError("config.net_arch must be a non-empty list or tuple")
    for index, value in enumerate(net_arch):
        _positive_int(value, f"config.net_arch[{index}]")

    rows = backend.get("rollouts")
    if not isinstance(rows, list):
        raise ContractError("rollouts must be a list")
    expected_rows = config["warmup_rollouts"] + config["measured_rollouts"]
    if len(rows) != expected_rows:
        raise ContractError("rollouts length must match warmup plus measured rollouts")
    measured = [row for row in rows if row.get("measured") is True]
    warmups = [row for row in rows if row.get("measured") is False]
    if len(measured) != config["measured_rollouts"] or len(warmups) != config["warmup_rollouts"]:
        raise ContractError("measured rollout count must match config")
    for row in rows:
        _validate_native_rollout_row(row, config, transitions_per_rollout)

    summary = _require_dict(backend.get("summary"), "summary")
    expected_summary = {
        "game_ticks": sum(row["game_ticks"] for row in measured),
        "player_transitions": sum(row["player_transitions"] for row in measured),
        "collection_seconds": sum(row["collection_seconds"] for row in measured),
        "update_seconds": sum(row["update_seconds"] for row in measured),
        "total_seconds": sum(row["total_seconds"] for row in measured),
    }
    for key, expected in expected_summary.items():
        actual = summary.get(key)
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
            raise ContractError(f"summary totals mismatch for {key}")
    _assert_rate(
        summary.get("total_player_transitions_per_second"),
        summary["player_transitions"],
        summary["total_seconds"],
        "summary.total_player_transitions_per_second",
    )
    _assert_rate(
        summary.get("collection_player_transitions_per_second"),
        summary["player_transitions"],
        summary["collection_seconds"],
        "summary.collection_player_transitions_per_second",
    )


def _validate_native_rollout_row(
    row: dict[str, Any], config: dict[str, Any], transitions_per_rollout: int
) -> None:
    if not isinstance(row, dict):
        raise ContractError("rollout rows must be objects")
    expected_ticks = config["games"] * config["n_steps"]
    if row.get("game_ticks") != expected_ticks:
        raise ContractError("exact game ticks mismatch")
    if row.get("player_transitions") != transitions_per_rollout:
        raise ContractError("exact player transitions mismatch")
    errors = row.get("errors")
    if not isinstance(errors, list) or len(errors) != config["games"] or any(value != 0 for value in errors):
        raise ContractError("rollout errors must be all zero")
    if row.get("losses_finite") is not True:
        raise ContractError("finite losses are required")
    metrics = _require_dict(row.get("metrics"), "rollout.metrics")
    for key in ("train/loss", "train/policy_gradient_loss", "train/value_loss", "train/entropy_loss"):
        _positive_or_negative_finite(metrics.get(key), f"rollout.metrics.{key}")
    for key in ("collection_seconds", "update_seconds", "total_seconds"):
        _positive_finite(row.get(key), f"rollout.{key}")
    if not math.isclose(
        float(row["collection_seconds"]) + float(row["update_seconds"]),
        float(row["total_seconds"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ContractError("rollout total_seconds must equal collection plus update")


def _assert_rate(rate: Any, numerator: Any, seconds: Any, name: str) -> None:
    actual = _positive_finite(rate, name)
    expected = _positive_finite(numerator, f"{name}.numerator") / _positive_finite(seconds, f"{name}.seconds")
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
        raise ContractError(f"{name} disagrees with measured totals")


def _positive_or_negative_finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{name} must be numeric")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ContractError(f"{name} must be finite")
    return as_float
