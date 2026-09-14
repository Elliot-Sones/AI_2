#!/usr/bin/env python3
"""Strict capture records for the FrozenOpponentWrapper navigation reducer.

This module instruments an existing frozen wrapper instance at runtime. It does
not mutate game mechanics, infer base transition state from old replay records,
or sample native targets. The capture seam is deliberately narrow:

* snapshot wrapper/player state before ``FrozenOpponentWrapper.step``;
* intercept the wrapper's underlying ``env.step`` call to record the actual
  masked action, base reward/done flags, and post-physics player/support inputs
  before wrapper side effects;
* intercept ``_spawn_target`` to record the actual Python-chosen handoff target;
* snapshot final wrapper state and returned reward/info.

The resulting JSONL schema is intended as the Python-oracle side of a
C-compatible navigation reward/episode reducer differential test.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator


CAPTURE_SCHEMA = "navigation_contract_capture_v1"
CONTRACT_SCOPE = "frozen_opponent_navigation_reducer_v1"
CONTRACT_VERSION = "warehouse-brawl-navigation-reducer-e0-task2"
EXCLUDED_SCOPES = (
    "pymunk_internal_solver_state",
    "physics_integration",
    "native_target_rng",
    "observation_frame_stack",
    "combat_attack_substate",
)
ACTION_MASK_INDICES = (5, 7, 8, 9)

WRAPPER_FIELD_NAMES = (
    "min_dist",
    "max_dist",
    "fall_penalty",
    "completion_mode",
    "control_mode",
    "settle_speed",
    "max_targets",
    "time_limit",
    "settle_steps",
    "post_goal_required_steps",
    "target_position",
    "last_distance",
    "targets_reached",
    "total_targets",
    "episode_steps",
    "targets_this_episode",
    "hold_steps",
    "post_goal_steps",
    "goal_reached_step",
    "target_touched",
    "fell_this_episode",
    "last_stocks",
    "falls",
    "interval_targets_reached",
    "interval_falls",
    "steps_since_log",
    "interval_episodes",
)

REQUIRED_METADATA_PATHS = (
    ("type",),
    ("schema",),
    ("contract_version",),
    ("scope",),
    ("source", "train_sha256"),
    ("source", "train_path"),
    ("config", "sha256"),
    ("config", "path"),
    ("case_source",),
)

REQUIRED_RECORD_PATHS = (
    ("type",),
    ("schema",),
    ("scope",),
    ("tick",),
    ("case_id",),
    ("input", "original_action"),
    ("input", "executed_action"),
    ("input", "action_immutable"),
    ("input", "pre_wrapper"),
    ("input", "post_physics", "base_reward"),
    ("input", "post_physics", "underlying_terminated"),
    ("input", "post_physics", "underlying_truncated"),
    ("input", "post_physics", "player_present"),
    ("input", "target_handoff"),
    ("expected", "final_reward"),
    ("expected", "terminated"),
    ("expected", "truncated"),
    ("expected", "nav_flags"),
    ("expected", "after_wrapper"),
)


class NavigationContractError(ValueError):
    """Fail-closed validation/comparison error for capture records."""

    def __init__(self, reason: str, message: str, field: str | None = None) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.field = field


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NavigationContractError("nonfinite_value", "float is not finite")
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "x") and hasattr(value, "y"):
        return [_json_safe(value.x), _json_safe(value.y)]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    return str(value)


def _vec2(value: Any, *, default: list[float] | None = None) -> list[Any] | None:
    if value is None:
        return default
    if hasattr(value, "x") and hasattr(value, "y"):
        return [_json_safe(value.x), _json_safe(value.y)]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [_json_safe(value[0]), _json_safe(value[1])]
    try:
        return [_json_safe(value[0]), _json_safe(value[1])]
    except Exception:
        return default


def _get_path(item: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = item
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise NavigationContractError("missing_field", f"Missing required field {'.'.join(path)}", ".".join(path))
        current = current[key]
    return current


def _check_finite(value: Any, path: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NavigationContractError("nonfinite_value", f"{path} is not finite", path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{path}.{key}" if path else str(key))


def _validate_action(action: Any, field: str) -> None:
    if not isinstance(action, list) or len(action) != 10:
        raise NavigationContractError("invalid_action", f"{field} must be exactly 10 finite values", field)
    for index, item in enumerate(action):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise NavigationContractError("invalid_action", f"{field}[{index}] must be numeric", f"{field}[{index}]")
        if not math.isfinite(float(item)):
            raise NavigationContractError("nonfinite_value", f"{field}[{index}] is not finite", f"{field}[{index}]")


def _validate_vec2(value: Any, field: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise NavigationContractError("missing_field", f"{field} must be a two-element vector", field)
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise NavigationContractError("missing_field", f"{field}[{index}] must be numeric", f"{field}[{index}]")
        if not math.isfinite(float(item)):
            raise NavigationContractError("nonfinite_value", f"{field}[{index}] is not finite", f"{field}[{index}]")


def _validate_wrapper_snapshot(snapshot: Any, field: str) -> None:
    if not isinstance(snapshot, dict):
        raise NavigationContractError("missing_field", f"{field} must be an object", field)
    for name in WRAPPER_FIELD_NAMES:
        if name not in snapshot:
            raise NavigationContractError("missing_field", f"{field}.{name} is required", f"{field}.{name}")


def _is_truthy_event(value: Any) -> bool:
    return value not in (False, None, 0, "", [], {})


def _requires_handoff_target(record: dict[str, Any], handoff: Any, flags: Any) -> bool:
    if isinstance(handoff, dict) and handoff.get("occurred") is True:
        return True
    if not isinstance(flags, dict) or not _is_truthy_event(flags.get("nav_reached")):
        return False
    pre_wrapper = record["input"]["pre_wrapper"]
    targets_this_episode = _as_int(pre_wrapper["targets_this_episode"], "input.pre_wrapper.targets_this_episode")
    max_targets = _as_int(pre_wrapper["max_targets"], "input.pre_wrapper.max_targets")
    return targets_this_episode < max_targets - 1


def validate_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        raise NavigationContractError("schema_mismatch", "metadata must be an object")
    for path in REQUIRED_METADATA_PATHS:
        _get_path(metadata, path)
    if metadata["type"] != "metadata" or metadata["schema"] != CAPTURE_SCHEMA:
        raise NavigationContractError("schema_mismatch", "unsupported metadata schema")
    if metadata["contract_version"] != CONTRACT_VERSION or metadata["scope"] != CONTRACT_SCOPE:
        raise NavigationContractError("schema_mismatch", "unsupported contract version/scope")
    train_sha = metadata["source"]["train_sha256"]
    train_path = metadata["source"]["train_path"]
    config_sha = metadata["config"]["sha256"]
    config_path = metadata["config"]["path"]
    if not _is_sha256_hex(train_sha):
        raise NavigationContractError("source_mismatch", "source.train_sha256 must be a sha256 hex string", "source.train_sha256")
    if not isinstance(train_path, str) or train_path in ("", "."):
        raise NavigationContractError("source_mismatch", "source.train_path must be non-empty", "source.train_path")
    if not _is_sha256_hex(config_sha):
        raise NavigationContractError("config_mismatch", "config.sha256 must be a sha256 hex string", "config.sha256")
    if not isinstance(config_path, str) or config_path in ("", "."):
        raise NavigationContractError("config_mismatch", "config.path must be non-empty", "config.path")
    _check_finite(metadata, "")


def validate_record(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise NavigationContractError("schema_mismatch", "record must be an object")
    for path in REQUIRED_RECORD_PATHS:
        _get_path(record, path)
    if record["type"] != "record" or record["schema"] != CAPTURE_SCHEMA or record["scope"] != CONTRACT_SCOPE:
        raise NavigationContractError("schema_mismatch", "unsupported record schema/scope")
    if not isinstance(record["tick"], int) or isinstance(record["tick"], bool) or record["tick"] < 0:
        raise NavigationContractError("invalid_tick", "tick must be a non-negative integer", "tick")
    _validate_action(record["input"]["original_action"], "input.original_action")
    _validate_action(record["input"]["executed_action"], "input.executed_action")
    if record["input"]["action_immutable"] is not True:
        raise NavigationContractError("action_immutable", "input action mutated during wrapper step", "input.action_immutable")
    _validate_wrapper_snapshot(record["input"]["pre_wrapper"], "input.pre_wrapper")
    _validate_wrapper_snapshot(record["expected"]["after_wrapper"], "expected.after_wrapper")
    post_physics = record["input"]["post_physics"]
    if post_physics["player_present"] is True:
        for path in (
            ("input", "post_physics", "player_position"),
            ("input", "post_physics", "player_velocity"),
            ("input", "post_physics", "player_stocks"),
            ("input", "post_physics", "player_grounded"),
            ("input", "post_physics", "support_source"),
            ("input", "post_physics", "support_velocity"),
        ):
            _get_path(record, path)
        _validate_vec2(post_physics["player_position"], "input.post_physics.player_position")
        _validate_vec2(post_physics["player_velocity"], "input.post_physics.player_velocity")
        _validate_vec2(post_physics["support_velocity"], "input.post_physics.support_velocity")
    elif post_physics["player_present"] is not False:
        raise NavigationContractError("missing_field", "input.post_physics.player_present must be boolean", "input.post_physics.player_present")
    handoff = record["input"]["target_handoff"]
    flags = record["expected"]["nav_flags"]
    if _requires_handoff_target(record, handoff, flags):
        _validate_vec2(_get_path(record, ("input", "target_handoff", "target_after")), "input.target_handoff.target_after")
    _check_finite(record, "")


def validate_records(records: Iterable[dict[str, Any]]) -> None:
    items = list(records)
    if not items:
        raise NavigationContractError("zero_records", "at least one capture record is required")
    for record in items:
        validate_record(record)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def default_metadata(
    *,
    source_train_path: Path,
    case_source: str,
    source_train_sha256: str | None = None,
    config_path: Path | None = None,
    config_sha256: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_path = Path(source_train_path)
    sha = source_train_sha256
    if sha is None:
        if not train_path.exists():
            raise NavigationContractError("missing_source", f"frozen train.py does not exist: {train_path}", "source.train_path")
        sha = file_sha256(train_path)
    config_path_text = str(config_path) if config_path is not None else None
    if config_sha256 is None and config_path is not None and config_path.exists() and config_path.is_file():
        config_sha256 = file_sha256(config_path)
    metadata = {
        "type": "metadata",
        "schema": CAPTURE_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "scope": CONTRACT_SCOPE,
        "case_source": str(case_source),
        "source": {
            "train_path": str(train_path),
            "train_sha256": sha,
        },
        "config": {
            "path": config_path_text,
            "sha256": config_sha256,
        },
        "capture": {
            "excluded_scopes": list(EXCLUDED_SCOPES),
            "action_mask_indices": list(ACTION_MASK_INDICES),
            "support_contract": "settle_support_uses_player_on_platform_velocity_else_zero",
        },
    }
    if extra:
        metadata["extra"] = _json_safe(extra)
    validate_metadata(metadata)
    return metadata


def mask_navigation_action(action: Any, *, control_mode: str) -> list[Any]:
    masked = list(_json_safe(action))
    if control_mode == "movement":
        for index in ACTION_MASK_INDICES:
            if index < len(masked):
                masked[index] = 0.0
    return masked


def snapshot_wrapper_state(wrapper: Any) -> dict[str, Any]:
    if getattr(wrapper, "debug_logs", False) is True:
        raise NavigationContractError(
            "unsupported_scope",
            "debug_logs=True can reset interval counters via _log_stats and is outside the native reducer contract",
            "wrapper.debug_logs",
        )
    state: dict[str, Any] = {}
    for name in WRAPPER_FIELD_NAMES:
        if not hasattr(wrapper, name):
            raise NavigationContractError("missing_field", f"wrapper field {name} is required", f"wrapper.{name}")
        state[name] = _json_safe(getattr(wrapper, name))
    return state


def _player_grounded(player: Any) -> bool:
    checker = getattr(player, "is_on_floor", None)
    if callable(checker):
        return bool(checker())
    return False


def snapshot_post_physics(wrapper: Any, base_transition: tuple[Any, Any, Any, Any, Any]) -> dict[str, Any]:
    _obs, base_reward, underlying_terminated, underlying_truncated, info = base_transition
    player = wrapper._get_player()
    base = {
        "base_reward": _json_safe(base_reward),
        "underlying_terminated": bool(underlying_terminated),
        "underlying_truncated": bool(underlying_truncated),
        "underlying_info": _json_safe(info),
    }
    if player is None:
        return {
            **base,
            "player_present": False,
        }
    body = getattr(player, "body", None)
    if body is None:
        raise NavigationContractError("missing_field", "player body is required when player is present", "input.post_physics.player_body")
    position = _vec2(getattr(body, "position", None))
    velocity = _vec2(getattr(body, "velocity", None))
    if position is None:
        raise NavigationContractError("missing_field", "player position is required", "input.post_physics.player_position")
    if velocity is None:
        raise NavigationContractError("missing_field", "player velocity is required", "input.post_physics.player_velocity")
    support = getattr(player, "on_platform", None)
    if support is not None:
        try:
            raw_support_velocity = getattr(support, "velocity")
        except Exception as exc:
            raise NavigationContractError(
                "missing_field",
                "player.on_platform.velocity is required when on_platform is present",
                "input.post_physics.support_velocity",
            ) from exc
        support_velocity = _vec2(raw_support_velocity)
        if support_velocity is None:
            raise NavigationContractError(
                "missing_field",
                "player.on_platform.velocity is required when on_platform is present",
                "input.post_physics.support_velocity",
            )
    else:
        support_velocity = [0.0, 0.0]
    return {
        **base,
        "player_present": True,
        "player_position": position,
        "player_velocity": velocity,
        "player_stocks": _json_safe(getattr(player, "stocks", None)),
        "player_grounded": _player_grounded(player),
        "support_source": "player.on_platform" if support is not None else "none",
        "support_velocity": support_velocity,
    }


def _transition_expected(result: tuple[Any, Any, Any, Any, Any], after_wrapper: dict[str, Any]) -> dict[str, Any]:
    _obs, final_reward, terminated, truncated, info = result
    info = info if isinstance(info, dict) else {}
    nav_flags = {
        key: _json_safe(info[key])
        for key in sorted(info)
        if key.startswith("nav_")
    }
    return {
        "final_reward": _json_safe(final_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "nav_flags": nav_flags,
        "after_wrapper": after_wrapper,
    }


def _target_handoff_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"occurred": False, "events": []}
    last = events[-1]
    return {
        "occurred": True,
        "target_after": last.get("target_after"),
        "events": events,
    }


def _observed_nav_events(info: dict[str, Any]) -> list[str]:
    event_keys = []
    for key in sorted(info):
        if not key.startswith("nav_"):
            continue
        if key in ("nav_contract", "nav_control_mode", "nav_hold_steps", "nav_post_goal_steps"):
            continue
        value = info[key]
        if value not in (False, None, 0, "", [], {}):
            event_keys.append(key)
    return event_keys


def capture_step(wrapper: Any, action: Any, *, tick: int, case_id: str = "real-game") -> dict[str, Any]:
    """Capture one real ``FrozenOpponentWrapper.step`` call with restored hooks."""

    if tick < 0:
        raise NavigationContractError("invalid_tick", "tick must be non-negative", "tick")
    original_action = _json_safe(copy.deepcopy(action))
    pre_wrapper = snapshot_wrapper_state(wrapper)
    had_env_step_attr = "step" in getattr(wrapper.env, "__dict__", {})
    had_spawn_attr = "_spawn_target" in getattr(wrapper, "__dict__", {})
    original_env_step = wrapper.env.step
    original_spawn_target = getattr(wrapper, "_spawn_target")
    context: dict[str, Any] = {
        "executed_action": None,
        "post_physics": None,
        "target_events": [],
    }

    def step_proxy(step_action: Any) -> Any:
        context["executed_action"] = _json_safe(copy.deepcopy(step_action))
        result = original_env_step(step_action)
        context["post_physics"] = snapshot_post_physics(wrapper, result)
        return result

    def spawn_proxy(player_pos: Any) -> Any:
        before = _json_safe(copy.deepcopy(getattr(wrapper, "target_position", None)))
        total_before = _json_safe(getattr(wrapper, "total_targets", None))
        result = original_spawn_target(player_pos)
        context["target_events"].append(
            {
                "player_position": _vec2(player_pos),
                "target_before": before,
                "target_after": _json_safe(copy.deepcopy(getattr(wrapper, "target_position", None))),
                "total_targets_before": total_before,
                "total_targets_after": _json_safe(getattr(wrapper, "total_targets", None)),
            }
        )
        return result

    try:
        wrapper.env.step = step_proxy
        wrapper._spawn_target = spawn_proxy
        result = wrapper.step(action)
    finally:
        if had_env_step_attr:
            wrapper.env.step = original_env_step
        else:
            try:
                delattr(wrapper.env, "step")
            except AttributeError:
                pass
        if had_spawn_attr:
            wrapper._spawn_target = original_spawn_target
        else:
            try:
                delattr(wrapper, "_spawn_target")
            except AttributeError:
                pass

    if context["executed_action"] is None or context["post_physics"] is None:
        raise NavigationContractError("missing_field", "instrumented wrapper did not call underlying env.step", "input.post_physics")

    after_wrapper = snapshot_wrapper_state(wrapper)
    expected = _transition_expected(result, after_wrapper)
    record = {
        "type": "record",
        "schema": CAPTURE_SCHEMA,
        "scope": CONTRACT_SCOPE,
        "tick": int(tick),
        "case_id": str(case_id),
        "input": {
            "original_action": original_action,
            "executed_action": context["executed_action"],
            "action_immutable": original_action == _json_safe(action),
            "pre_wrapper": pre_wrapper,
            "post_physics": context["post_physics"],
            "target_handoff": _target_handoff_from_events(context["target_events"]),
        },
        "expected": expected,
        "observed_events": _observed_nav_events(expected["nav_flags"]),
    }
    validate_record(record)
    return record


def write_jsonl(path: Path, metadata: dict[str, Any], records: Iterable[dict[str, Any]]) -> int:
    validate_metadata(metadata)
    items = list(records)
    validate_records(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        for record in items:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return len(items)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NavigationContractError("schema_mismatch", f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise NavigationContractError("schema_mismatch", f"{path}:{line_number}: item must be an object")
            yield item


def read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    items = list(_iter_jsonl(path))
    if not items:
        raise NavigationContractError("zero_records", "capture file is empty")
    metadata = items[0]
    records = items[1:]
    validate_metadata(metadata)
    validate_records(records)
    return metadata, records


def _difference(left: Any, right: Any, path: str, *, abs_tol: float = 1e-9) -> dict[str, Any] | None:
    if isinstance(left, float) or isinstance(right, float):
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return {"field": path, "expected": left, "actual": right}
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            return {"field": path, "expected": left, "actual": right}
        if abs(float(left) - float(right)) > abs_tol:
            return {"field": path, "expected": left, "actual": right}
        return None
    if type(left) is not type(right):
        return {"field": path, "expected": left, "actual": right}
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            key = sorted(left_keys ^ right_keys)[0]
            return {"field": f"{path}.{key}" if path else key, "expected": left.get(key), "actual": right.get(key)}
        for key in sorted(left):
            diff = _difference(left[key], right[key], f"{path}.{key}" if path else key, abs_tol=abs_tol)
            if diff is not None:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"field": path, "expected": left, "actual": right}
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            diff = _difference(left_item, right_item, f"{path}[{index}]", abs_tol=abs_tol)
            if diff is not None:
                return diff
        return None
    if left != right:
        return {"field": path, "expected": left, "actual": right}
    return None


def compare_native_results(records: Iterable[dict[str, Any]], native_results: Iterable[dict[str, Any]], *, abs_tol: float = 1e-9) -> dict[str, Any]:
    record_list = list(records)
    native_list = list(native_results)
    validate_records(record_list)
    if not native_list:
        return {"passed": False, "reason": "zero_native_results", "compared_records": 0, "first_difference": None}
    if len(record_list) != len(native_list):
        return {
            "passed": False,
            "reason": "length_mismatch",
            "compared_records": min(len(record_list), len(native_list)),
            "first_difference": {"field": "records.length", "expected": len(record_list), "actual": len(native_list)},
        }
    for index, (record, native) in enumerate(zip(record_list, native_list)):
        if not isinstance(native, dict):
            return {
                "passed": False,
                "reason": "schema_mismatch",
                "compared_records": index,
                "first_difference": {"tick": record["tick"], "field": "native", "expected": "object", "actual": native},
            }
        _check_finite(native, "native")
        if "expected" in native or "input" in native:
            expected = expected_native_result(record)
        else:
            expected = record["expected"]
        path = "" if ("expected" in native or "input" in native) else "expected"
        diff = _difference(expected, native, path, abs_tol=abs_tol)
        if diff is not None:
            diff["tick"] = record["tick"]
            return {"passed": False, "reason": "value_mismatch", "compared_records": index + 1, "first_difference": diff}
    return {"passed": True, "reason": "match", "compared_records": len(record_list), "first_difference": None}


def _record_projection(record: dict[str, Any]) -> dict[str, Any]:
    validate_record(record)
    return {
        "tick": record["tick"],
        "case_id": record["case_id"],
        "input": copy.deepcopy(record["input"]),
        "expected": copy.deepcopy(record["expected"]),
    }


def expected_native_result(record: dict[str, Any]) -> dict[str, Any]:
    """Return the explicit native-result projection for a capture record.

    This intentionally maps the reducer-visible action fields and the complete
    expected wrapper output object. Missing wrapper output fields in a native
    receipt therefore fail comparison instead of being silently dropped.
    """

    validate_record(record)
    return {
        "input": {
            "executed_action": copy.deepcopy(record["input"]["executed_action"]),
            "action_immutable": bool(record["input"]["action_immutable"]),
        },
        "expected": copy.deepcopy(record["expected"]),
    }


def _native_module() -> Any:
    import native_navigation_contract as native

    return native


def _as_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise NavigationContractError("missing_field", f"{field} must be numeric", field)
    return float(value)


def _as_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise NavigationContractError("missing_field", f"{field} must be int", field)
    return int(value)


def _as_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise NavigationContractError("missing_field", f"{field} must be bool", field)
    return value


def _as_vec2_or_none(value: Any, field: str) -> tuple[float, float] | None:
    if value is None:
        return None
    _validate_vec2(value, field)
    return (float(value[0]), float(value[1]))


def _config_from_wrapper(wrapper_state: dict[str, Any]) -> Any:
    native = _native_module()
    return native.NavigationConfig(
        fall_penalty=_as_float(wrapper_state["fall_penalty"], "wrapper.fall_penalty"),
        completion_mode=str(wrapper_state["completion_mode"]),
        control_mode=str(wrapper_state["control_mode"]),
        settle_speed=_as_float(wrapper_state["settle_speed"], "wrapper.settle_speed"),
        max_targets=_as_int(wrapper_state["max_targets"], "wrapper.max_targets"),
        time_limit_steps=_as_int(wrapper_state["time_limit"], "wrapper.time_limit"),
        settle_steps=_as_int(wrapper_state["settle_steps"], "wrapper.settle_steps"),
        post_goal_required_steps=_as_int(wrapper_state["post_goal_required_steps"], "wrapper.post_goal_required_steps"),
    )


def _state_from_wrapper(wrapper_state: dict[str, Any]) -> Any:
    native = _native_module()
    target = _as_vec2_or_none(wrapper_state["target_position"], "wrapper.target_position")
    last_distance_valid = wrapper_state["last_distance"] is not None
    goal_reached_step_valid = wrapper_state["goal_reached_step"] is not None
    return native.NavigationState(
        target_x=0.0 if target is None else target[0],
        target_y=0.0 if target is None else target[1],
        target_valid=target is not None,
        last_distance=0.0 if not last_distance_valid else _as_float(wrapper_state["last_distance"], "wrapper.last_distance"),
        last_distance_valid=last_distance_valid,
        targets_reached=_as_int(wrapper_state["targets_reached"], "wrapper.targets_reached"),
        total_targets=_as_int(wrapper_state["total_targets"], "wrapper.total_targets"),
        episode_steps=_as_int(wrapper_state["episode_steps"], "wrapper.episode_steps"),
        targets_this_episode=_as_int(wrapper_state["targets_this_episode"], "wrapper.targets_this_episode"),
        hold_steps=_as_int(wrapper_state["hold_steps"], "wrapper.hold_steps"),
        post_goal_steps=_as_int(wrapper_state["post_goal_steps"], "wrapper.post_goal_steps"),
        goal_reached_step=0 if not goal_reached_step_valid else _as_int(wrapper_state["goal_reached_step"], "wrapper.goal_reached_step"),
        goal_reached_step_valid=goal_reached_step_valid,
        target_touched=_as_bool(wrapper_state["target_touched"], "wrapper.target_touched"),
        fell_this_episode=_as_bool(wrapper_state["fell_this_episode"], "wrapper.fell_this_episode"),
        last_stocks=_as_int(wrapper_state["last_stocks"], "wrapper.last_stocks"),
        falls=_as_int(wrapper_state["falls"], "wrapper.falls"),
        interval_targets_reached=_as_int(wrapper_state["interval_targets_reached"], "wrapper.interval_targets_reached"),
        interval_falls=_as_int(wrapper_state["interval_falls"], "wrapper.interval_falls"),
        steps_since_log=_as_int(wrapper_state["steps_since_log"], "wrapper.steps_since_log"),
        interval_episodes=_as_int(wrapper_state["interval_episodes"], "wrapper.interval_episodes"),
    )


def _post_physics_from_record(post_physics: dict[str, Any], handoff: dict[str, Any]) -> Any:
    native = _native_module()
    player_present = _as_bool(post_physics["player_present"], "input.post_physics.player_present")
    target_after = handoff.get("target_after") if isinstance(handoff, dict) and handoff.get("occurred") is True else None
    next_target = _as_vec2_or_none(target_after, "input.target_handoff.target_after")
    if player_present:
        player_position = _as_vec2_or_none(post_physics["player_position"], "input.post_physics.player_position")
        player_velocity = _as_vec2_or_none(post_physics["player_velocity"], "input.post_physics.player_velocity")
        player_stocks = _as_int(post_physics["player_stocks"], "input.post_physics.player_stocks")
        player_grounded = _as_bool(post_physics["player_grounded"], "input.post_physics.player_grounded")
        support_velocity = _as_vec2_or_none(post_physics["support_velocity"], "input.post_physics.support_velocity")
    else:
        player_position = None
        player_velocity = None
        player_stocks = None
        player_grounded = None
        support_velocity = (0.0, 0.0)
    assert support_velocity is not None
    return native.NavigationPostPhysics(
        raw_reward=_as_float(post_physics["base_reward"], "input.post_physics.base_reward"),
        raw_terminated=_as_bool(post_physics["underlying_terminated"], "input.post_physics.underlying_terminated"),
        raw_truncated=_as_bool(post_physics["underlying_truncated"], "input.post_physics.underlying_truncated"),
        player_present=player_present,
        player_x=0.0 if player_position is None else player_position[0],
        player_y=0.0 if player_position is None else player_position[1],
        velocity_x=0.0 if player_velocity is None else player_velocity[0],
        velocity_y=0.0 if player_velocity is None else player_velocity[1],
        support_velocity_x=support_velocity[0],
        support_velocity_y=support_velocity[1],
        support_velocity_valid=bool(player_present and post_physics.get("support_source") == "player.on_platform"),
        grounded=bool(player_grounded) if player_grounded is not None else False,
        stocks=0 if player_stocks is None else player_stocks,
        next_target_x=0.0 if next_target is None else next_target[0],
        next_target_y=0.0 if next_target is None else next_target[1],
        next_target_valid=next_target is not None,
    )


def _target_position_from_native_state(state: Any) -> list[float] | None:
    if bool(getattr(state, "target_valid")):
        return [_json_safe(float(getattr(state, "target_x"))), _json_safe(float(getattr(state, "target_y")))]
    return None


def _goal_reached_step_from_native_state(state: Any) -> int | None:
    if bool(getattr(state, "goal_reached_step_valid")):
        return int(getattr(state, "goal_reached_step"))
    return None


def _namespace_from_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return argparse.Namespace(**{key: _namespace_from_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace_from_dict(item) for item in value]
    return value


def _nav_flags_from_raw_info(record: dict[str, Any]) -> dict[str, Any]:
    post_physics = record["input"]["post_physics"]
    info = post_physics.get("underlying_info", {})
    if not isinstance(info, dict):
        return {}
    return {
        key: _json_safe(info[key])
        for key in sorted(info)
        if key.startswith("nav_")
    }


def _expected_from_native_step_result(record: dict[str, Any], result: Any) -> dict[str, Any]:
    pre = record["input"]["pre_wrapper"]
    state = result.state
    nav_flags = _nav_flags_from_raw_info(record)
    if record["input"]["post_physics"]["player_present"] is True:
        nav_flags.update(
            {
                "nav_contract": "active_sequence_v3" if pre["completion_mode"] == "sequence" else "sustained_v2",
                "nav_control_mode": pre["control_mode"],
                "nav_sequence_complete": bool(result.success and pre["completion_mode"] == "sequence"),
                "nav_hold_steps": int(state.hold_steps),
                "nav_post_goal_steps": int(state.post_goal_steps),
                "nav_terminal_horizon": bool(result.horizon),
                "nav_success": bool(result.success),
            }
        )
        if bool(result.fall):
            nav_flags["nav_fall"] = 1
        if bool(result.touched):
            nav_flags["nav_touched"] = 1
        if bool(result.reached):
            nav_flags["nav_reached"] = 1
    after_wrapper = {
        "min_dist": copy.deepcopy(pre["min_dist"]),
        "max_dist": copy.deepcopy(pre["max_dist"]),
        "fall_penalty": copy.deepcopy(pre["fall_penalty"]),
        "completion_mode": copy.deepcopy(pre["completion_mode"]),
        "control_mode": copy.deepcopy(pre["control_mode"]),
        "settle_speed": copy.deepcopy(pre["settle_speed"]),
        "max_targets": copy.deepcopy(pre["max_targets"]),
        "time_limit": copy.deepcopy(pre["time_limit"]),
        "settle_steps": copy.deepcopy(pre["settle_steps"]),
        "post_goal_required_steps": copy.deepcopy(pre["post_goal_required_steps"]),
        "target_position": _target_position_from_native_state(state),
        "last_distance": float(state.last_distance) if bool(state.last_distance_valid) else None,
        "targets_reached": int(state.targets_reached),
        "total_targets": int(state.total_targets),
        "episode_steps": int(state.episode_steps),
        "targets_this_episode": int(state.targets_this_episode),
        "hold_steps": int(state.hold_steps),
        "post_goal_steps": int(state.post_goal_steps),
        "goal_reached_step": _goal_reached_step_from_native_state(state),
        "target_touched": bool(state.target_touched),
        "fell_this_episode": bool(state.fell_this_episode),
        "last_stocks": int(state.last_stocks),
        "falls": int(state.falls),
        "interval_targets_reached": int(state.interval_targets_reached),
        "interval_falls": int(state.interval_falls),
        "steps_since_log": int(state.steps_since_log),
        "interval_episodes": int(state.interval_episodes),
    }
    return {
        "final_reward": float(result.reward),
        "terminated": bool(result.terminated),
        "truncated": bool(result.truncated),
        "nav_flags": _json_safe(nav_flags),
        "after_wrapper": _json_safe(after_wrapper),
    }


def native_case_from_record(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Convert a capture record into explicit scalar native reducer inputs."""

    validate_record(record)
    pre_wrapper = record["input"]["pre_wrapper"]
    return (
        _config_from_wrapper(pre_wrapper),
        _state_from_wrapper(pre_wrapper),
        _post_physics_from_record(record["input"]["post_physics"], record["input"]["target_handoff"]),
    )


def native_result_for_record(record: dict[str, Any], result: Any) -> dict[str, Any]:
    """Normalize a native result object/dict to the expected-output dict shape."""

    validate_record(record)
    if isinstance(result, dict):
        if "state" not in result:
            raise NavigationContractError(
                "schema_mismatch",
                "native result dict must contain serialized native state",
                "native_result.state",
            )
        result = _namespace_from_dict(result)
    state = getattr(result, "state", None)
    if state is not None:
        return _expected_from_native_step_result(record, result)
    raise NavigationContractError("schema_mismatch", "native result must be a dataclass, namedtuple, or dict", "native_result")


def compare_capture_files(left_path: Path, right_path: Path, *, abs_tol: float = 1e-9) -> dict[str, Any]:
    left_meta, left_records = read_jsonl(left_path)
    right_meta, right_records = read_jsonl(right_path)
    if left_meta["source"]["train_sha256"] != right_meta["source"]["train_sha256"]:
        return {"passed": False, "reason": "source_mismatch", "compared_records": 0, "first_difference": {"field": "source.train_sha256"}}
    for field in (("source", "train_path"), ("config", "path"), ("config", "sha256"), ("case_source",)):
        if _get_path(left_meta, field) != _get_path(right_meta, field):
            return {
                "passed": False,
                "reason": "metadata_mismatch",
                "compared_records": 0,
                "first_difference": {"field": ".".join(field), "expected": _get_path(left_meta, field), "actual": _get_path(right_meta, field)},
            }
    if left_meta["contract_version"] != right_meta["contract_version"] or left_meta["scope"] != right_meta["scope"]:
        return {"passed": False, "reason": "schema_mismatch", "compared_records": 0, "first_difference": {"field": "contract"}}
    if len(left_records) != len(right_records):
        return {
            "passed": False,
            "reason": "length_mismatch",
            "compared_records": min(len(left_records), len(right_records)),
            "first_difference": {"field": "records.length", "expected": len(left_records), "actual": len(right_records)},
        }
    for index, (left, right) in enumerate(zip(left_records, right_records)):
        left_projection = _record_projection(left)
        right_projection = _record_projection(right)
        diff = _difference(left_projection, right_projection, "", abs_tol=abs_tol)
        if diff is not None:
            diff["tick"] = left.get("tick")
            return {"passed": False, "reason": "value_mismatch", "compared_records": index + 1, "first_difference": diff}
    return {"passed": True, "reason": "match", "compared_records": len(left_records), "first_difference": None}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate a capture JSONL file")
    validate.add_argument("path", type=Path)
    compare = subparsers.add_parser("compare", help="Compare two capture JSONL files")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "validate":
        metadata, records = read_jsonl(args.path)
        print(json.dumps({"passed": True, "schema": metadata["schema"], "records": len(records)}, sort_keys=True))
        return 0
    if args.command == "compare":
        receipt = compare_capture_files(args.left, args.right)
        print(json.dumps(receipt, sort_keys=True))
        return 0 if receipt["passed"] else 1
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
