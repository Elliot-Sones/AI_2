#!/usr/bin/env python3
"""Replay contract capture and comparison helpers for simulator parity work.

The comparator is intentionally import-light: comparing recorded files does not
import pygame, torch, gymnasium, or the game package. Capture paths import the
current repo lazily and start from reproducible resets; this does not provide a
full Pymunk solver snapshot/restore contract.
"""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import source_path

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator


REPLAY_SCHEMA = "ai2.replay.e0.v1"
GAME_CONTRACT_VERSION = "warehouse-brawl-python-e0"
CONTRACT_SCOPE = "navigation_state_replay_v1"
EXCLUDED_STATE_SCOPES = (
    "combat_attack_substate",
    "pymunk_internal_solver_snapshot",
)
EXCLUDED_RUNTIME_FIELDS = (
    "transition.info.episode.t",
)
SOLVER_LIMIT_NOTE = (
    "Replay capture starts from reproducible resets and records observed game/wrapper "
    "state; full Pymunk internal solver snapshot/restore is not available."
)
REQUIRED_METADATA_KEYS = (
    "type",
    "schema",
    "source",
    "config",
    "rng",
    "actions",
    "controls",
    "game_contract_version",
    "capture",
)
REQUIRED_RECORD_KEYS = ("type", "schema", "tick", "action", "transition", "state", "discrete_events")
REQUIRED_STATE_PATHS = (
    ("state", "env", "steps"),
    ("state", "env", "seed"),
    ("state", "env", "max_timesteps"),
    ("state", "env", "terminated"),
    ("state", "env", "rewards"),
    ("state", "players"),
    ("state", "moving_platforms"),
    ("state", "contacts"),
    ("state", "spawners"),
    ("state", "wrapper"),
    ("state", "coverage"),
    ("state", "solver_note"),
)
REQUIRED_METADATA_PATHS = (
    ("capture", "contract_scope"),
    ("capture", "excluded_state"),
    ("capture", "excluded_runtime_fields"),
)
REQUIRED_PLAYER_KEYS = (
    "agent_id",
    "state",
    "state_timers",
    "facing",
    "weapon",
    "stocks",
    "damage",
    "damage_taken_this_stock",
    "damage_taken_total",
    "damage_done",
    "position",
    "velocity",
    "smooth_x_vel",
    "target_vel",
    "damage_velocity",
    "pickup_lock_until",
    "support",
    "grounded",
    "cur_action",
)
REQUIRED_PLATFORM_KEYS = ("id", "name", "body_id", "position", "velocity", "moving_to_w2", "waypoint1", "waypoint2")
REQUIRED_FIXTURE_LABELS = (
    "startup",
    "walk",
    "turn",
    "jump",
    "edge",
    "idle",
    "timeout",
    "reset",
    "targethandoff",
    "platformlanding",
    "platformriding",
    "platformreversal",
)
REQUIRED_EVENT_GATES = (
    "timeout",
    "reset",
    "targethandoff",
    "platformlanding",
    "platformriding",
    "platformreversal",
)
RAW_EVENT_GATES = tuple(event for event in REQUIRED_EVENT_GATES if event != "targethandoff")
NAVIGATION_EVENT_GATES = REQUIRED_EVENT_GATES
CONTROLLED_PLAYER_ID = 0
NAVIGATION_FIXTURE_SCHEDULE = (
    ("startup", 5),
    ("walk", 35),
    ("turn", 12),
    ("jump", 18),
    ("edge", 35),
    ("idle", 10),
    ("targethandoff", 60),
    ("platformlanding", 80),
    ("platformriding", 80),
    ("platformreversal", 80),
    ("timeout", 120),
    ("reset", 1),
)
COMPARE_METADATA_KEYS = (
    ("config", "sha256"),
    ("config", "phase"),
    ("rng", "action_sha256"),
    ("actions", "source"),
    ("actions", "script"),
    ("controls", "kind"),
    ("controls", "player_indices"),
    ("capture", "mode"),
    ("capture", "steps"),
    ("capture", "frame_stack"),
    ("capture", "contract_scope"),
    ("capture", "excluded_runtime_fields"),
)


@dataclasses.dataclass(frozen=True)
class ReplayComparison:
    ok: bool
    reason: str
    message: str
    compared_records: int = 0
    first_difference: dict[str, Any] | None = None


class ReplayValidationError(ValueError):
    def __init__(self, reason: str, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.field = field


def prepare_source_imports(source_root: Path) -> None:
    source_root = source_root.resolve()
    loaded_game = getattr(sys.modules.get("environment.environment"), "__file__", None)
    expected_game = source_root / "UTMIST-AI2-main/environment/environment.py"
    if not expected_game.is_file():
        raise FileNotFoundError(f"Selected game source is missing: {expected_game}")
    if loaded_game is not None and Path(loaded_game).resolve() != expected_game.resolve():
        raise RuntimeError("A different game source is already loaded; use a fresh process for the selected source")
    os.chdir(source_root)
    for module_name in ("benchmark_training", "train"):
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is not None and not _path_is_relative_to(Path(module_file).resolve(), source_root):
            del sys.modules[module_name]
    for path in (source_root, source_root / "UTMIST-AI2-main"):
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "x") and hasattr(value, "y"):
        return [json_safe(value.x), json_safe(value.y)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _finite_number(value: Any) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def _validate_finite(value: Any, field: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{field}.{key}" if field else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{field}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not _finite_number(value):
            raise ReplayValidationError("nonfinite_values", f"Non-finite value at {field}", field)


def _missing_keys(item: dict[str, Any], required: Iterable[str]) -> list[str]:
    return [key for key in required if key not in item]


def _require_path(item: dict[str, Any], path: tuple[Any, ...]) -> None:
    current: Any = item
    field = []
    for key in path:
        if isinstance(key, int):
            field[-1] = f"{field[-1]}[{key}]"
            if not isinstance(current, list) or len(current) <= key:
                raise ReplayValidationError("missing_keys", f"Missing required field {'.'.join(field)}", ".".join(field))
            current = current[key]
            continue
        field.append(str(key))
        if not isinstance(current, dict) or key not in current:
            raise ReplayValidationError("missing_keys", f"Missing required field {'.'.join(field)}", ".".join(field))
        current = current[key]


def validate_metadata(metadata: dict[str, Any]) -> None:
    missing = _missing_keys(metadata, REQUIRED_METADATA_KEYS)
    if missing:
        raise ReplayValidationError("missing_keys", f"Metadata missing keys: {missing}", missing[0])
    for path in REQUIRED_METADATA_PATHS:
        _require_path(metadata, path)
    if metadata["type"] != "metadata":
        raise ReplayValidationError("schema_mismatch", "First JSONL item must be metadata")
    if metadata["schema"] != REPLAY_SCHEMA:
        raise ReplayValidationError("schema_mismatch", f"Unsupported replay schema {metadata['schema']!r}")
    if metadata["game_contract_version"] != GAME_CONTRACT_VERSION:
        raise ReplayValidationError(
            "unsupported_game_contract_version",
            f"Unsupported game contract version {metadata['game_contract_version']!r}",
            "game_contract_version",
        )
    _validate_finite(metadata, "")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_non_null(value: Any, field: str) -> None:
    if value is None:
        raise ReplayValidationError("invalid_required_anchor", f"Required anchor {field} is null", field)


def _require_vector(value: Any, field: str, *, length: int = 2) -> None:
    _require_non_null(value, field)
    if not isinstance(value, list) or len(value) < length:
        raise ReplayValidationError("invalid_required_anchor", f"Required anchor {field} must be a vector", field)
    for index, item in enumerate(value[:length]):
        if not _is_number(item):
            raise ReplayValidationError("invalid_required_anchor", f"Required anchor {field}[{index}] must be finite", f"{field}[{index}]")
        if not math.isfinite(float(item)):
            raise ReplayValidationError("nonfinite_values", f"Non-finite value at {field}[{index}]", f"{field}[{index}]")


def _validate_action_anchor(action: Any) -> None:
    _require_non_null(action, "action")
    if isinstance(action, dict):
        if not action:
            raise ReplayValidationError("invalid_required_anchor", "Required anchor action is empty", "action")
        for key in sorted(action):
            value = action[key]
            field = f"action.{key}"
            _require_non_null(value, field)
            if isinstance(value, list) and not value:
                raise ReplayValidationError("invalid_required_anchor", f"Required anchor {field} is empty", field)
    elif isinstance(action, list):
        if not action:
            raise ReplayValidationError("invalid_required_anchor", "Required anchor action is empty", "action")
        for index, value in enumerate(action):
            _require_non_null(value, f"action[{index}]")
    else:
        raise ReplayValidationError("invalid_required_anchor", "Required anchor action must be dict or list", "action")


def validate_record(record: dict[str, Any]) -> None:
    missing = _missing_keys(record, REQUIRED_RECORD_KEYS)
    if missing:
        raise ReplayValidationError("missing_keys", f"Record missing keys: {missing}", missing[0])
    if record["type"] != "record" or record["schema"] != REPLAY_SCHEMA:
        raise ReplayValidationError("schema_mismatch", "Record type/schema mismatch")
    _validate_action_anchor(record["action"])
    for path in REQUIRED_STATE_PATHS:
        _require_path(record, path)
    players = record["state"]["players"]
    if not isinstance(players, list) or len(players) != 2:
        raise ReplayValidationError("missing_keys", "Replay record must contain exactly two player states", "state.players")
    for index, player in enumerate(players):
        if not isinstance(player, dict):
            raise ReplayValidationError("missing_keys", f"Player {index} is not an object", f"state.players[{index}]")
        missing = _missing_keys(player, REQUIRED_PLAYER_KEYS)
        if missing:
            raise ReplayValidationError(
                "missing_keys",
                f"Player {index} missing keys: {missing}",
                f"state.players[{index}].{missing[0]}",
            )
        prefix = f"state.players[{index}]"
        _require_non_null(player["agent_id"], f"{prefix}.agent_id")
        _require_non_null(player["state"], f"{prefix}.state")
        _require_non_null(player["state_timers"], f"{prefix}.state_timers")
        _require_non_null(player["facing"], f"{prefix}.facing")
        _require_non_null(player["weapon"], f"{prefix}.weapon")
        _require_vector(player["position"], f"{prefix}.position")
        _require_vector(player["velocity"], f"{prefix}.velocity")
        _require_vector(player["target_vel"], f"{prefix}.target_vel")
        _require_vector(player["damage_velocity"], f"{prefix}.damage_velocity")
        _require_non_null(player["grounded"], f"{prefix}.grounded")
        _require_non_null(player["cur_action"], f"{prefix}.cur_action")
    platforms = record["state"]["moving_platforms"]
    if not isinstance(platforms, list) or not platforms:
        raise ReplayValidationError("missing_keys", "Replay record must contain moving platform state", "state.moving_platforms")
    for index, platform in enumerate(platforms):
        if not isinstance(platform, dict):
            raise ReplayValidationError("missing_keys", f"Platform {index} is not an object", f"state.moving_platforms[{index}]")
        missing = _missing_keys(platform, REQUIRED_PLATFORM_KEYS)
        if missing:
            raise ReplayValidationError(
                "missing_keys",
                f"Platform {index} missing keys: {missing}",
                f"state.moving_platforms[{index}].{missing[0]}",
            )
        prefix = f"state.moving_platforms[{index}]"
        _require_non_null(platform["id"], f"{prefix}.id")
        _require_non_null(platform["name"], f"{prefix}.name")
        _require_non_null(platform["body_id"], f"{prefix}.body_id")
        _require_vector(platform["position"], f"{prefix}.position")
        _require_vector(platform["velocity"], f"{prefix}.velocity")
        _require_non_null(platform["moving_to_w2"], f"{prefix}.moving_to_w2")
        _require_vector(platform["waypoint1"], f"{prefix}.waypoint1")
        _require_vector(platform["waypoint2"], f"{prefix}.waypoint2")
    _validate_finite(record, "")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ReplayValidationError("schema_mismatch", f"{path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ReplayValidationError("schema_mismatch", f"{path}:{line_number}: item is not an object")
            yield item


def _get_nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _metadata_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for path in COMPARE_METADATA_KEYS:
        if _get_nested(left, path) != _get_nested(right, path):
            return False
    return True


def _difference(field: str, left: Any, right: Any, tick: Any) -> dict[str, Any]:
    return {"tick": tick, "field": field, "left": left, "right": right}


def _compare_values(left: Any, right: Any, *, field: str, tick: Any, abs_tol: float, rel_tol: float) -> dict[str, Any] | None:
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return _difference(field, sorted(left_keys), sorted(right_keys), tick)
        for key in sorted(left):
            child = _compare_values(
                left[key],
                right[key],
                field=f"{field}.{key}" if field else str(key),
                tick=tick,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
            if child is not None:
                return child
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return _difference(field, len(left), len(right), tick)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            child = _compare_values(
                left_item,
                right_item,
                field=f"{field}[{index}]",
                tick=tick,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
            if child is not None:
                return child
        return None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        if not math.isclose(float(left), float(right), abs_tol=abs_tol, rel_tol=rel_tol):
            return _difference(field, left, right, tick)
        return None
    if left != right:
        return _difference(field, left, right, tick)
    return None


def _without_monitor_episode_wallclock(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    info = value.get("info")
    if not isinstance(info, dict):
        return value
    episode = info.get("episode")
    if not isinstance(episode, dict) or "t" not in episode:
        return value
    normalized = dict(value)
    normalized_info = dict(info)
    normalized_episode = dict(episode)
    del normalized_episode["t"]
    if normalized_episode:
        normalized_info["episode"] = normalized_episode
    else:
        del normalized_info["episode"]
    normalized["info"] = normalized_info
    return normalized


def _comparison_record(record: dict[str, Any]) -> dict[str, Any]:
    transition = record.get("transition")
    normalized_transition = _without_monitor_episode_wallclock(transition)
    if normalized_transition is transition:
        return record
    normalized = dict(record)
    normalized["transition"] = normalized_transition
    return normalized


def compare_replay_files(left_path: Path, right_path: Path, *, abs_tol: float = 0.0, rel_tol: float = 0.0) -> ReplayComparison:
    if abs_tol < 0 or rel_tol < 0 or not math.isfinite(abs_tol) or not math.isfinite(rel_tol):
        return ReplayComparison(False, "nonfinite_values", "Tolerances must be finite non-negative numbers")
    try:
        left_iter = _iter_jsonl(Path(left_path))
        right_iter = _iter_jsonl(Path(right_path))
        left_metadata = next(left_iter, None)
        right_metadata = next(right_iter, None)
        if left_metadata is None or right_metadata is None:
            return ReplayComparison(False, "zero_compared_records", "Both replay files must contain metadata and at least one record")
        validate_metadata(left_metadata)
        validate_metadata(right_metadata)
        if not _metadata_compatible(left_metadata, right_metadata):
            return ReplayComparison(False, "incompatible_comparison_metadata", "Replay metadata does not describe the same comparable action/config capture")

        compared = 0
        while True:
            left_record = next(left_iter, None)
            right_record = next(right_iter, None)
            if left_record is None or right_record is None:
                if left_record is not None or right_record is not None:
                    return ReplayComparison(False, "length_mismatch", "Replay record counts differ", compared)
                break
            validate_record(left_record)
            validate_record(right_record)
            compared += 1
            tick = left_record.get("tick")
            if tick != right_record.get("tick"):
                return ReplayComparison(
                    False,
                    "value_mismatch",
                    "First divergent tick value differs",
                    compared,
                    _difference("tick", tick, right_record.get("tick"), tick),
                )
            first = _compare_values(
                _comparison_record(left_record),
                _comparison_record(right_record),
                field="",
                tick=tick,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
            if first is not None:
                return ReplayComparison(False, "value_mismatch", "Replay records diverged", compared, first)
        if compared == 0:
            return ReplayComparison(False, "zero_compared_records", "No replay records were compared", 0)
        return ReplayComparison(True, "match", "Replay records match", compared)
    except ReplayValidationError as exc:
        first = {"field": exc.field} if exc.field else None
        return ReplayComparison(False, exc.reason, str(exc), 0, first)


def _vec2(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if hasattr(value, "x") and hasattr(value, "y"):
        return [json_safe(value.x), json_safe(value.y)]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [json_safe(value[0]), json_safe(value[1])]
    return json_safe(value)


def _state_timer_snapshot(state: Any) -> dict[str, Any]:
    names = (
        "jumps_left",
        "recoveries_left",
        "jump_timer",
        "dodge_timer",
        "dodge_cooldown",
        "stun_frames",
        "move_type",
        "frames_into_recovery",
        "recovery_frames",
    )
    return {name: json_safe(getattr(state, name)) for name in names if hasattr(state, name)}


def stable_platform_body_id(platform_id: Any) -> str | None:
    if platform_id is None:
        return None
    return f"platform:{platform_id}"


def _shape_intersects(left: Any, right: Any) -> bool:
    try:
        return bool(left.cache_bb().intersects(right.cache_bb()))
    except Exception:
        return False


def _shape_world_bb(shape: Any) -> dict[str, Any] | None:
    try:
        bb = shape.cache_bb()
    except Exception:
        return None
    if bb is None:
        return None
    bounds = {}
    for name in ("left", "right", "bottom", "top"):
        if not hasattr(bb, name):
            return None
        bounds[name] = json_safe(getattr(bb, name))
    return bounds


def _arbiter_shape_roles(shapes: Iterable[Any], platform_ids: dict[int, Any], player: Any) -> list[dict[str, Any]]:
    roles: list[dict[str, Any]] = []
    player_shape = getattr(player, "shape", None)
    for order, shape in enumerate(shapes):
        entry: dict[str, Any] = {"order": order, "role": "unknown"}
        if shape is player_shape:
            entry = {
                "order": order,
                "role": "player",
                "agent_id": json_safe(getattr(player, "agent_id", None)),
            }
        else:
            platform_id = getattr(shape, "platform_id", None)
            shape_body = getattr(shape, "body", None)
            if platform_id is None and id(shape_body) in platform_ids:
                platform_id = platform_ids[id(shape_body)]
            if platform_id is not None:
                entry = {
                    "order": order,
                    "role": "platform",
                    "platform_id": json_safe(platform_id),
                    "platform_body_id": stable_platform_body_id(platform_id),
                }
        roles.append(entry)
    return roles


def _player_is_on_floor(player: Any) -> bool:
    try:
        checker = getattr(player, "is_on_floor")
        return bool(checker())
    except Exception:
        return False


def geometric_platform_support(player: Any, platforms: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Return observable support using the frozen game's is_on_floor + BB seam."""

    if not _player_is_on_floor(player):
        return None
    player_shape = getattr(player, "shape", None)
    player_body = getattr(player, "body", None)
    player_position = getattr(player_body, "position", None)
    for platform in platforms:
        obj = platform["object"]
        shape = getattr(obj, "shape", None)
        body = getattr(obj, "body", None)
        if player_shape is None or shape is None or not _shape_intersects(player_shape, shape):
            continue
        platform_position = getattr(body, "position", None)
        if player_position is not None and platform_position is not None:
            try:
                if player_position.y > platform_position.y + 0.05:
                    continue
            except Exception:
                pass
        platform_id = platform["id"]
        return {
            "kind": "platform",
            "id": json_safe(platform_id),
            "body_id": stable_platform_body_id(platform_id),
            "source": "geometry",
            "method": "is_on_floor_bb",
        }
    return None


def _arbiter_contact_points(arbiter: Any) -> list[dict[str, Any]]:
    point_set = getattr(arbiter, "contact_point_set", None)
    points = []
    for point in getattr(point_set, "points", []) or []:
        points.append(
            {
                "point_a": _vec2(getattr(point, "point_a", None)),
                "point_b": _vec2(getattr(point, "point_b", None)),
                "distance": json_safe(getattr(point, "distance", None)),
            }
        )
    return points


def _pymunk_contacts_for_player(player: Any, platform_ids: dict[int, Any]) -> list[dict[str, Any]]:
    body = getattr(player, "body", None)
    each_arbiter = getattr(body, "each_arbiter", None)
    if not callable(each_arbiter):
        return []
    contacts: list[dict[str, Any]] = []

    def copy_arbiter(arbiter: Any) -> None:
        shapes = list(getattr(arbiter, "shapes", ()) or ())
        platform_id = None
        for shape in shapes:
            shape_platform_id = getattr(shape, "platform_id", None)
            if shape_platform_id is not None:
                platform_id = shape_platform_id
                break
            shape_body = getattr(shape, "body", None)
            if id(shape_body) in platform_ids:
                platform_id = platform_ids[id(shape_body)]
                break
        if platform_id is None:
            return
        contact_id = f"arbiter:player:{json_safe(getattr(player, 'agent_id', None))}:platform:{json_safe(platform_id)}:{len(contacts)}"
        contacts.append(
            {
                "kind": "pymunk_arbiter",
                "source": "body.each_arbiter",
                "player_agent_id": json_safe(getattr(player, "agent_id", None)),
                "platform_id": json_safe(platform_id),
                "platform_body_id": stable_platform_body_id(platform_id),
                "contact_id": contact_id,
                "shape_roles": _arbiter_shape_roles(shapes, platform_ids, player),
                "normal_role": "shape0_to_shape1_raw",
                "normal": _vec2(getattr(getattr(arbiter, "contact_point_set", None), "normal", None)),
                "points": _arbiter_contact_points(arbiter),
                "total_impulse": _vec2(getattr(arbiter, "total_impulse", None)),
                "total_ke": json_safe(getattr(arbiter, "total_ke", None)),
            }
        )

    try:
        each_arbiter(copy_arbiter)
    except Exception:
        return contacts
    return contacts


def capture_player_state(
    player: Any,
    platform_ids: dict[int, Any] | None = None,
    geometric_support: dict[str, Any] | None = None,
) -> dict[str, Any]:
    support = getattr(player, "on_platform", None)
    platform_id = platform_ids.get(id(support)) if platform_ids is not None else None
    support_state = (
        {
            "kind": "platform",
            "id": json_safe(platform_id),
            "body_id": stable_platform_body_id(platform_id),
            "source": "pymunk_on_platform",
            "method": "player.on_platform",
        }
        if platform_id is not None
        else geometric_support
    )
    state = getattr(player, "state", None)
    return {
        "agent_id": json_safe(getattr(player, "agent_id", None)),
        "state": type(state).__name__ if state is not None else None,
        "state_timers": _state_timer_snapshot(state),
        "facing": str(getattr(player, "facing", None)),
        "weapon": json_safe(getattr(player, "weapon", None)),
        "stocks": json_safe(getattr(player, "stocks", None)),
        "damage": json_safe(getattr(player, "damage", None)),
        "damage_taken_this_stock": json_safe(getattr(player, "damage_taken_this_stock", None)),
        "damage_taken_total": json_safe(getattr(player, "damage_taken_total", None)),
        "damage_done": json_safe(getattr(player, "damage_done", None)),
        "position": _vec2(getattr(getattr(player, "body", None), "position", None)),
        "velocity": _vec2(getattr(getattr(player, "body", None), "velocity", None)),
        "smooth_x_vel": json_safe(getattr(player, "smoothXVel", None)),
        "target_vel": _vec2(getattr(player, "target_vel", None)),
        "damage_velocity": _vec2(getattr(player, "damage_velocity", None)),
        "pickup_lock_until": json_safe(getattr(player, "pickup_lock_until", None)),
        "support": support_state,
        "grounded": _player_is_on_floor(player),
        "cur_action": json_safe(getattr(player, "cur_action", None)),
    }


def capture_spawner_state(spawner: Any) -> dict[str, Any]:
    active_weapon = getattr(spawner, "active_weapon", None)
    vfx = getattr(spawner, "vfx", None)
    return {
        "id": json_safe(getattr(spawner, "id", None)),
        "kind": type(spawner).__name__,
        "world_pos": json_safe(getattr(spawner, "world_pos", None)),
        "flag": json_safe(getattr(spawner, "flag", None)),
        "last_spawn_frame": json_safe(getattr(spawner, "last_spawn_frame", None)),
        "cooldown_frames": json_safe(getattr(spawner, "cooldown_frames", None)),
        "despawn_frames": json_safe(getattr(spawner, "despawn_frames", None)),
        "weapon_name": json_safe(getattr(spawner, "weapon_name", None)),
        "active_weapon": json_safe(getattr(active_weapon, "name", None)),
        "vfx": {
            "state": json_safe(getattr(vfx, "state", None)),
            "state_timer": json_safe(getattr(vfx, "state_timer", None)),
        } if vfx is not None else None,
    }


def coverage_ledger(
    dispatched_fixture_labels: Iterable[str],
    verified_events: Iterable[str] = (),
    required_event_gates: Iterable[str] = REQUIRED_EVENT_GATES,
) -> dict[str, Any]:
    dispatched_list = [str(item) for item in dispatched_fixture_labels]
    verified_list = sorted({str(item) for item in verified_events})
    verified_set = set(verified_list)
    required_list = [str(item) for item in required_event_gates]
    return {
        "dispatched_fixture_labels": dispatched_list,
        "verified_events": verified_list,
        "required_event_gates": required_list,
        "remaining_event_gates": [event for event in required_list if event not in verified_set],
    }


def event_gates_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "raw":
        return RAW_EVENT_GATES
    if mode == "navigation":
        return NAVIGATION_EVENT_GATES
    raise ValueError(f"Unknown replay mode {mode!r}")


def capture_env_state(
    env: Any,
    *,
    coverage_labels: Iterable[str] = (),
    verified_events: Iterable[str] = (),
    required_event_gates: Iterable[str] = REQUIRED_EVENT_GATES,
) -> dict[str, Any]:
    raw = unwrap_raw_env(env)
    platforms = []
    platform_objects = []
    platform_ids = {}
    for name, obj in sorted(getattr(raw, "objects", {}).items()):
        shape = getattr(obj, "shape", None)
        if hasattr(shape, "platform_id"):
            body = getattr(obj, "body", None)
            if body is not None:
                platform_ids[id(body)] = shape.platform_id
            platform_objects.append({"id": shape.platform_id, "name": name, "object": obj})
            platforms.append(
                {
                    "id": json_safe(shape.platform_id),
                    "name": name,
                    "body_id": stable_platform_body_id(shape.platform_id),
                    "position": _vec2(getattr(body, "position", None)),
                    "velocity": _vec2(getattr(body, "velocity", None)),
                    "moving_to_w2": json_safe(getattr(obj, "moving_to_w2", None)),
                    "waypoint1": json_safe(getattr(obj, "waypoint1", None)),
                    "waypoint2": json_safe(getattr(obj, "waypoint2", None)),
                    "geometry": {
                        "world_bb": _shape_world_bb(shape),
                        "endpoint_tolerance": 0.05,
                    },
                }
            )
    contacts = []
    raw_players = list(getattr(raw, "players", []))
    geometric_supports = {
        id(player): geometric_platform_support(player, platform_objects)
        for player in raw_players
    }
    players = [
        capture_player_state(player, platform_ids, geometric_supports.get(id(player)))
        for player in raw_players
    ]
    for player in raw_players:
        contacts.extend(_pymunk_contacts_for_player(player, platform_ids))
    for player in players:
        support = player.get("support")
        if support is not None:
            source = support.get("source")
            kind = "geometric_platform_support" if source == "geometry" else "player_platform_support"
            contact_source = "is_on_floor_bb" if source == "geometry" else "player.on_platform"
            contacts.append(
                {
                    "kind": kind,
                    "source": contact_source,
                    "player_agent_id": player.get("agent_id"),
                    "platform_id": support.get("id"),
                    "platform_body_id": support.get("body_id"),
                    "contact_id": f"{source}:player:{player.get('agent_id')}:platform:{support.get('id')}",
                }
            )
    return {
        "env": {
            "steps": json_safe(getattr(raw, "steps", None)),
            "seed": json_safe(getattr(raw, "seed", None)),
            "max_timesteps": json_safe(getattr(raw, "max_timesteps", None)),
            "terminated": json_safe(getattr(raw, "terminated", None)),
            "rewards": json_safe(getattr(raw, "rewards", None)),
        },
        "players": players,
        "moving_platforms": platforms,
        "contacts": contacts,
        "spawners": [
            capture_spawner_state(spawner)
            for spawner in getattr(getattr(raw, "weapon_controller", None), "spawners", [])
        ],
        "wrapper": capture_wrapper_state(env),
        "coverage": coverage_ledger(
            coverage_labels,
            verified_events=verified_events,
            required_event_gates=required_event_gates,
        ),
        "solver_note": SOLVER_LIMIT_NOTE,
    }


def capture_wrapper_state(env: Any) -> dict[str, Any]:
    wrappers = []
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        entry = {"kind": type(current).__name__}
        if hasattr(current, "n_stack"):
            entry["n_stack"] = json_safe(getattr(current, "n_stack"))
        if hasattr(current, "stackedobs"):
            stacked = getattr(current, "stackedobs")
            entry["stackedobs_shape"] = json_safe(getattr(stacked, "shape", None))
        for name in (
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
            "history_length",
            "action_dim",
            "zero_history",
        ):
            if hasattr(current, name):
                entry[name] = json_safe(getattr(current, name))
        if hasattr(current, "history"):
            entry["history"] = json_safe(list(getattr(current, "history")))
        wrappers.append(entry)
        if hasattr(current, "venv"):
            current = getattr(current, "venv")
        elif hasattr(current, "envs") and getattr(current, "envs"):
            current = getattr(current, "envs")[0]
        else:
            current = getattr(current, "env", None)
    return {"chain": wrappers}


def unwrap_raw_env(env: Any) -> Any:
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        if hasattr(current, "players") and hasattr(current, "step"):
            return current
        seen.add(id(current))
        if hasattr(current, "raw_env"):
            current = getattr(current, "raw_env")
        elif hasattr(current, "venv"):
            current = getattr(current, "venv")
        elif hasattr(current, "envs") and getattr(current, "envs"):
            current = getattr(current, "envs")[0]
        else:
            current = getattr(current, "env", None)
    return env


def _single_lane_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _single_lane_info(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    return value


def _strict_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _transition_extra_fields(terminated: Any, truncated: Any, info: Any) -> dict[str, Any]:
    done = terminated is True or truncated is True
    terminal_observation = None
    has_terminal_observation = False
    if isinstance(info, dict) and "terminal_observation" in info:
        terminal_observation = info.get("terminal_observation")
        has_terminal_observation = True
    extra = {
        "auto_reset": bool(done and has_terminal_observation),
        "post_reset_state": bool(done and has_terminal_observation),
    }
    if has_terminal_observation:
        extra["terminal_observation"] = terminal_observation
    return extra


def transition_summary(transition: Any) -> dict[str, Any]:
    safe = json_safe(transition)
    if isinstance(safe, list) and len(safe) >= 5:
        terminated = _single_lane_value(safe[2])
        truncated = _single_lane_value(safe[3])
        info = _single_lane_info(safe[4])
        return {
            "observation": safe[0],
            "reward": _single_lane_value(safe[1]),
            "terminated": terminated,
            "truncated": truncated,
            "info": _without_monitor_episode_wallclock({"info": info})["info"],
            **_transition_extra_fields(terminated, truncated, info),
        }
    if isinstance(safe, list) and len(safe) == 4:
        done = _single_lane_value(safe[2])
        info = _single_lane_info(safe[3])
        done_bool = _strict_bool(done)
        time_limit_truncated = _truthy_transition_flag(info, ("TimeLimit.truncated",))
        if done_bool is None:
            terminated = done
            truncated = False
        else:
            terminated = bool(done_bool and not time_limit_truncated)
            truncated = bool(done_bool and time_limit_truncated)
        return {
            "observation": safe[0],
            "reward": _single_lane_value(safe[1]),
            "terminated": terminated,
            "truncated": truncated,
            "info": _without_monitor_episode_wallclock({"info": info})["info"],
            **_transition_extra_fields(terminated, truncated, info),
        }
    return {"raw": safe}


def discrete_events_from_record(transition: Any, env: Any) -> list[str]:
    events = []
    summary = transition_summary(transition)
    if summary.get("terminated") is True:
        events.append("terminated")
    if summary.get("truncated") is True:
        events.append("truncated")
    if summary.get("auto_reset") is True:
        events.append("auto_reset")
    if summary.get("post_reset_state") is True:
        events.append("post_reset_state")
    info = summary.get("info")
    if isinstance(info, dict):
        for key in sorted(info):
            value = info[key]
            if isinstance(value, bool) and value:
                events.append(key)
            elif key.startswith("nav_") and value not in (False, None, 0):
                events.append(f"{key}={value}")
    raw = unwrap_raw_env(env)
    logger = getattr(raw, "logger", None)
    if isinstance(logger, list):
        for index, item in enumerate(logger):
            if isinstance(item, dict):
                transition_name = item.get("transition")
                move_type = item.get("move_type")
                if transition_name:
                    events.append(f"player{index}.transition={transition_name}")
                if move_type not in ("", None):
                    events.append(f"player{index}.move_type={move_type}")
    return events


def _truthy_transition_flag(info: Any, names: Iterable[str]) -> bool:
    if not isinstance(info, dict):
        return False
    for name in names:
        value = info.get(name)
        if value not in (False, None, 0, "", [], {}):
            return True
    return False


def _wrapper_counters(wrapper_state: dict[str, Any], names: Iterable[str]) -> dict[str, int]:
    counters = {}
    for entry in wrapper_state.get("chain", []):
        if not isinstance(entry, dict):
            continue
        for name in names:
            value = entry.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                counters[name] = max(counters.get(name, int(value)), int(value))
    return counters


def _platform_by_id(state: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    platforms = {}
    for platform in state.get("moving_platforms", []):
        if isinstance(platform, dict):
            platforms[platform.get("id")] = platform
    return platforms


def _platform_is_moving(platform: dict[str, Any]) -> bool:
    if isinstance(platform.get("moving_to_w2"), bool):
        return True
    velocity = platform.get("velocity")
    if isinstance(velocity, list):
        return any(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0
            for value in velocity[:2]
        )
    return False


def _controlled_grounded_support_ids(state: dict[str, Any], moving_platform_ids: set[Any]) -> set[Any]:
    supported: set[Any] = set()
    for player in state.get("players", []):
        if not isinstance(player, dict):
            continue
        if player.get("agent_id") != CONTROLLED_PLAYER_ID or player.get("grounded") is not True:
            continue
        support = player.get("support")
        if not isinstance(support, dict) or support.get("kind") != "platform":
            continue
        platform_id = support.get("id")
        if platform_id in moving_platform_ids:
            supported.add(platform_id)
    return supported


def _velocity_sign(platform: dict[str, Any]) -> tuple[int, int]:
    velocity = platform.get("velocity")
    if not isinstance(velocity, list) or len(velocity) < 2:
        return (0, 0)
    signs = []
    for value in velocity[:2]:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value == 0:
            signs.append(0)
        else:
            signs.append(1 if value > 0 else -1)
    return (signs[0], signs[1])


def verified_events_from_state(
    transition: Any,
    state: dict[str, Any],
    *,
    previous_state: dict[str, Any] | None = None,
    reset_event: bool = False,
) -> set[str]:
    """Infer reached coverage gates from observed state only.

    Fixture labels describe inputs we dispatched. This intentionally keeps event
    gates open unless transition/state fields show the event happened.
    """

    events: set[str] = set()
    summary = transition_summary(transition)
    info = summary.get("info")
    auto_reset = summary.get("auto_reset") is True
    reset_boundary = reset_event or auto_reset
    if summary.get("truncated") is True or _truthy_transition_flag(info, ("TimeLimit.truncated", "timeout", "nav_timeout")):
        events.add("timeout")
    if reset_boundary:
        events.add("reset")
    if _truthy_transition_flag(info, ("nav_success", "nav_reached", "target_reached", "target_handoff", "sequence_complete")):
        events.add("targethandoff")

    wrapper = state.get("wrapper", {})
    effective_previous_state = None if reset_boundary else previous_state
    previous_wrapper = effective_previous_state.get("wrapper", {}) if effective_previous_state else {}
    counter_names = ("targets_reached", "targets_this_episode")
    current_targets = _wrapper_counters(wrapper, counter_names)
    previous_targets = _wrapper_counters(previous_wrapper, counter_names)
    for name, value in current_targets.items():
        previous_value = previous_targets.get(name)
        if previous_value is not None and value > previous_value:
            events.add("targethandoff")
            break

    platforms = _platform_by_id(state)
    moving_platform_ids = {platform_id for platform_id, platform in platforms.items() if _platform_is_moving(platform)}
    current_support_ids = _controlled_grounded_support_ids(state, moving_platform_ids)
    previous_platforms = _platform_by_id(effective_previous_state) if effective_previous_state else {}
    previous_moving_platform_ids = {
        platform_id for platform_id, platform in previous_platforms.items() if _platform_is_moving(platform)
    }
    previous_support_ids = (
        _controlled_grounded_support_ids(effective_previous_state, previous_moving_platform_ids)
        if effective_previous_state
        else set()
    )
    if current_support_ids and not reset_boundary:
        if current_support_ids - previous_support_ids:
            events.add("platformlanding")
        if current_support_ids & previous_support_ids:
            events.add("platformriding")

    same_supported_platforms = set() if reset_boundary else current_support_ids & previous_support_ids
    if same_supported_platforms and previous_platforms:
        for platform_id in same_supported_platforms:
            platform = platforms.get(platform_id)
            previous = previous_platforms.get(platform_id)
            if platform is None or previous is None:
                continue
            if previous is None:
                continue
            if platform.get("moving_to_w2") is not None and previous.get("moving_to_w2") is not None:
                if platform.get("moving_to_w2") != previous.get("moving_to_w2"):
                    events.add("platformreversal")
                    break
            current_sign = _velocity_sign(platform)
            previous_sign = _velocity_sign(previous)
            if any(c != 0 and p != 0 and c != p for c, p in zip(current_sign, previous_sign)):
                events.add("platformreversal")
                break
    return events


def make_replay_record(
    tick: int,
    action: Any,
    transition: Any,
    env: Any,
    *,
    coverage_labels: Iterable[str] = (),
    previous_state: dict[str, Any] | None = None,
    reset_event: bool = False,
    verified_events: Iterable[str] = (),
    required_event_gates: Iterable[str] = REQUIRED_EVENT_GATES,
) -> dict[str, Any]:
    state = capture_env_state(
        env,
        coverage_labels=coverage_labels,
        required_event_gates=required_event_gates,
    )
    all_verified = set(verified_events)
    all_verified.update(
        verified_events_from_state(
            transition,
            state,
            previous_state=previous_state,
            reset_event=reset_event,
        )
    )
    state["coverage"] = coverage_ledger(
        coverage_labels,
        verified_events=all_verified,
        required_event_gates=required_event_gates,
    )
    record = {
        "type": "record",
        "schema": REPLAY_SCHEMA,
        "tick": int(tick),
        "action": json_safe(action),
        "transition": transition_summary(transition),
        "state": state,
        "discrete_events": discrete_events_from_record(transition, env),
    }
    validate_record(record)
    return record


def _action_vector(space: Any, pressed: Iterable[int]) -> Any:
    try:
        import numpy as np

        action = np.zeros(tuple(space.shape), dtype=space.dtype)
    except Exception:
        action = [0] * 10
    for index in pressed:
        action[index] = 1
    return action


def scripted_raw_actions(space: Any, names: Iterable[str]) -> list[dict[int, Any]]:
    mapping = {
        "startup": (),
        "idle": (),
        "walk": (3,),
        "turn": (1,),
        "jump": (4,),
        "edge": (3, 2),
        "edgehold": (3, 2),
        "attack": (7,),
        "timeout": (),
        "reset": (),
        "targethandoff": (3,),
        "platformlanding": (4,),
        "platformriding": (3,),
        "platformreversal": (1,),
    }
    actions = []
    for name in names:
        if name not in mapping:
            raise ValueError(f"Unknown scripted action {name!r}")
        player = _action_vector(space, mapping[name])
        opponent = _action_vector(space, ())
        actions.append({0: player, 1: opponent})
    return actions


def navigation_fixture_labels(steps: int, schedule: Iterable[tuple[str, int]] = NAVIGATION_FIXTURE_SCHEDULE) -> list[str]:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    labels: list[str] = []
    for label, duration in schedule:
        if duration <= 0:
            raise ValueError(f"Fixture duration must be positive for {label!r}")
        labels.extend([str(label)] * int(duration))
        if len(labels) >= steps:
            return labels[:steps]
    if not labels:
        return []
    while len(labels) < steps:
        labels.append("idle")
    return labels[:steps]


def default_metadata(
    source_root: Path,
    config_path: Path | None,
    seed: int,
    action_seed: int,
    action_sha256: str,
    mode: str,
    steps: int,
    *,
    phase: str | None = None,
    frame_stack: int | None = None,
    action_script: Iterable[str] | None = None,
    action_source: str | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    config_hash = file_digest(config_path) if config_path and config_path.exists() else None
    env_path = source_root / "UTMIST-AI2-main/environment/environment.py"
    navigation = mode == "navigation"
    return {
        "type": "metadata",
        "schema": REPLAY_SCHEMA,
        "game_contract_version": GAME_CONTRACT_VERSION,
        "source": {
            "root": str(source_root),
            "environment_sha256": file_digest(env_path) if env_path.exists() else None,
        },
        "config": {
            "path": str(config_path) if config_path else None,
            "sha256": config_hash,
            "phase": phase,
        },
        "rng": {
            "seed": seed,
            "action_seed": action_seed,
            "action_sha256": action_sha256,
            "reset_seed_policy": "seed+tick+1",
        },
        "actions": {
            "source": action_source or ("scripted-fixture" if navigation else "benchmark_training.action_plan"),
            "script": list(action_script) if action_script is not None else (list(REQUIRED_FIXTURE_LABELS) if navigation else None),
        },
        "controls": {
            "kind": "navigation_movement_multibinary_10" if navigation else "raw_multibinary_10",
            "player_indices": [0] if navigation else [0, 1],
        },
        "capture": {
            "mode": mode,
            "steps": steps,
            "frame_stack": frame_stack,
            "contract_scope": CONTRACT_SCOPE,
            "excluded_state": list(EXCLUDED_STATE_SCOPES),
            "excluded_runtime_fields": list(EXCLUDED_RUNTIME_FIELDS),
            "note": SOLVER_LIMIT_NOTE,
        },
    }


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_actions(actions: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for item in actions:
        payload = json.dumps(json_safe(item), sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl(path: Path, metadata: dict[str, Any], records: Iterable[dict[str, Any]]) -> int:
    validate_metadata(metadata)
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        for record in records:
            validate_record(record)
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError("Refusing to write replay with zero records")
    return count


def require_event_gates_satisfied(records: Iterable[dict[str, Any]]) -> None:
    last_record = None
    for record in records:
        last_record = record
    if last_record is None:
        raise ValueError("Cannot require event gates on zero replay records")
    coverage = last_record.get("state", {}).get("coverage", {})
    remaining = coverage.get("remaining_event_gates")
    if remaining:
        raise ValueError(f"Replay still has remaining event gates: {remaining}")


def load_frozen_train(source_root: Path, wrapper_source: Path):
    prepare_source_imports(source_root)
    spec = importlib.util.spec_from_file_location("ai2_replay_frozen_train", wrapper_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen train module from {wrapper_source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_replay_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("Navigation replay capture requires PyYAML") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_benchmark_helper(source_root: Path):
    """Keep a frozen helper when supplied; payload-only roots use current tooling."""
    source_root = Path(source_root).resolve()
    candidate = source_path(source_root, "benchmark_training.py")
    if (source_root / "ai2/paths.py").is_file() and not candidate.is_file():
        raise FileNotFoundError(f"Selected benchmark helper is missing: {candidate}")
    if source_root == PROJECT_ROOT.resolve() or not candidate.is_file():
        from ai2.evaluation import benchmark_training
        return benchmark_training
    spec = importlib.util.spec_from_file_location("benchmark_training", candidate)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load selected benchmark helper: {candidate}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_navigation_env(source_root: Path, config_path: Path, wrapper_source: Path, seed: int, frame_stack: int = 4):
    prepare_source_imports(source_root)
    bench = load_benchmark_helper(source_root)

    bench.set_headless_environment()
    bench.seed_everything(seed)
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    frozen_train = load_frozen_train(source_root, wrapper_source)
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    params = load_replay_config(config_path)
    phase_key = str(params.get("curriculum", {}).get("phase", "0b"))
    phase_config = frozen_train.PhaseConfig(params, phase_key)
    if not phase_config.is_navigation:
        raise ValueError(f"Navigation replay requires a navigation phase, got {phase_key!r}")
    resolution = frozen_train.get_resolution(params.get("environment_settings", {}).get("resolution", "LOW"))

    def make_one():
        bench.seed_everything(seed)
        return frozen_train.make_nav_env(phase_config, resolution, params)

    env = DummyVecEnv([make_one])
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    return env, params, phase_config


def capture_raw_replay(
    source_root: Path,
    output: Path,
    *,
    seed: int,
    steps: int,
    config_path: Path | None = None,
    action_seed: int | None = None,
    require_event_gates: bool = False,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    prepare_source_imports(source_root)
    bench = load_benchmark_helper(source_root)

    bench.set_headless_environment()
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    frozen_flat_helper = source_root / "benchmark_training.py"
    if Path(bench.__file__).resolve() == frozen_flat_helper:
        env = bench.build_raw_env(seed)
    else:
        env = bench.build_raw_env(seed, source_root=source_root)
    expected_environment = source_root / "UTMIST-AI2-main/environment/environment.py"
    loaded_environment = getattr(sys.modules.get("environment.environment"), "__file__", None)
    if loaded_environment is None or Path(loaded_environment).resolve() != expected_environment.resolve():
        env.close()
        raise RuntimeError(f"Replay game source does not match selected source: {expected_environment}")
    action_seed = seed if action_seed is None else action_seed
    actions = bench.action_plan(env.action_space, steps, action_seed, n_agents=2)
    metadata = default_metadata(source_root, config_path, seed, action_seed, bench.hash_actions(actions), "raw", steps)
    records = []
    try:
        env.reset(seed=seed)
        previous_state = None
        verified_events: set[str] = set()
        reset_event = False
        for tick, action in enumerate(actions):
            transition = env.step(action)
            record = make_replay_record(
                tick,
                action,
                transition,
                env,
                previous_state=previous_state,
                reset_event=reset_event,
                verified_events=verified_events,
                required_event_gates=event_gates_for_mode("raw"),
            )
            verified_events.update(record["state"]["coverage"]["verified_events"])
            record["state"]["coverage"] = coverage_ledger(
                (),
                verified_events=verified_events,
                required_event_gates=event_gates_for_mode("raw"),
            )
            records.append(record)
            summary = record["transition"]
            if summary.get("terminated") is True or summary.get("truncated") is True:
                env.reset(seed=seed + tick + 1)
                reset_event = True
            else:
                reset_event = False
            previous_state = record["state"]
    finally:
        env.close()
    if require_event_gates:
        require_event_gates_satisfied(records)
    written = write_jsonl(output, metadata, records)
    return {"output": str(output), "records": written, "metadata": metadata}


def capture_navigation_replay(
    source_root: Path,
    output: Path,
    *,
    config_path: Path,
    wrapper_source: Path,
    seed: int,
    steps: int,
    labels: Iterable[str] | None = None,
    frame_stack: int = 4,
    require_event_gates: bool = False,
) -> dict[str, Any]:
    labels = list(labels if labels is not None else navigation_fixture_labels(steps))
    env, _params, phase_config = build_navigation_env(source_root, config_path, wrapper_source, seed, frame_stack)
    actions = scripted_navigation_actions(env.action_space, labels)
    if len(actions) < steps:
        repeats = (steps + len(actions) - 1) // len(actions)
        actions = (actions * repeats)[:steps]
        labels = (labels * repeats)[:steps]
    else:
        actions = actions[:steps]
        labels = labels[:steps]
    metadata = default_metadata(
        source_root,
        config_path,
        seed,
        seed,
        hash_actions(actions),
        "navigation",
        steps,
        phase=str(phase_config.phase_key),
        frame_stack=frame_stack,
        action_script=labels,
    )
    records = []
    try:
        env.reset()
        reached = []
        previous_state = None
        verified_events: set[str] = set()
        for tick, (label, action) in enumerate(zip(labels, actions)):
            reset_event = False
            if label == "reset":
                env.reset()
                reset_event = True
                previous_state = None
            transition = env.step([action])
            reached.append(label)
            record = make_replay_record(
                tick,
                action,
                transition,
                env,
                coverage_labels=reached,
                previous_state=previous_state,
                reset_event=reset_event,
                verified_events=verified_events,
                required_event_gates=event_gates_for_mode("navigation"),
            )
            verified_events.update(record["state"]["coverage"]["verified_events"])
            record["state"]["coverage"] = coverage_ledger(
                reached,
                verified_events=verified_events,
                required_event_gates=event_gates_for_mode("navigation"),
            )
            records.append(record)
            previous_state = record["state"]
    finally:
        env.close()
    if require_event_gates:
        require_event_gates_satisfied(records)
    written = write_jsonl(output, metadata, records)
    return {"output": str(output), "records": written, "metadata": metadata}


def scripted_navigation_actions(space: Any, names: Iterable[str]) -> list[Any]:
    raw = scripted_raw_actions(space, names)
    return [item[0] for item in raw]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Capture raw replay JSONL")
    capture.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    capture.add_argument("--config", type=Path)
    capture.add_argument("--wrapper-source", type=Path)
    capture.add_argument("--mode", choices=("raw", "navigation"), default="raw")
    capture.add_argument("--seed", type=int, default=42)
    capture.add_argument("--action-seed", type=int)
    capture.add_argument("--steps", type=int, default=10_000)
    capture.add_argument("--frame-stack", type=int, default=4)
    capture.add_argument(
        "--require-event-gates",
        action="store_true",
        help="Fail capture if the final coverage ledger still has remaining required event gates.",
    )
    capture.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare", help="Compare two replay JSONL files")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--abs-tol", type=float, default=0.0)
    compare.add_argument("--rel-tol", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "capture":
        if args.mode == "navigation":
            if args.config is None or args.wrapper_source is None:
                raise SystemExit("--mode navigation requires --config and --wrapper-source")
            result = capture_navigation_replay(
                args.source_root,
                args.output,
                config_path=args.config,
                wrapper_source=args.wrapper_source,
                seed=args.seed,
                steps=args.steps,
                frame_stack=args.frame_stack,
                require_event_gates=args.require_event_gates,
            )
        else:
            result = capture_raw_replay(
                args.source_root,
                args.output,
                seed=args.seed,
                steps=args.steps,
                config_path=args.config,
                action_seed=args.action_seed,
                require_event_gates=args.require_event_gates,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "compare":
        result = compare_replay_files(args.left, args.right, abs_tol=args.abs_tol, rel_tol=args.rel_tol)
        print(json.dumps(dataclasses.asdict(result), sort_keys=True))
        return 0 if result.ok else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
