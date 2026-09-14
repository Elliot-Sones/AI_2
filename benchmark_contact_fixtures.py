#!/usr/bin/env python3
"""Bounded CPU-only search for replay contact/navigation fixture traces.

This script intentionally uses reproducible resets plus action streams only. It
does not teleport bodies, restore arbitrary Pymunk solver state, or run policy
training. Successful traces are JSONL replay-contract files; unsuccessful scans
still write a summary explaining which event gates remained unobserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import benchmark_replay as replay


DEFAULT_SEARCH_GATES = (
    "platformlanding",
    "platformriding",
    "platformreversal",
    "targethandoff",
)

CONTACT_EVIDENCE_GATES = (
    "platformlanding",
    "platformriding",
    "platformreversal",
    "contact_underside",
    "contact_side",
    "contact_edge",
)

CONTROLLED_PLAYER_ID = 0

CONTACT_REQUIRED_FIELDS = {
    "geometric_platform_support": ("kind", "source", "player_agent_id", "platform_id", "platform_body_id", "contact_id"),
    "pymunk_arbiter": (
        "kind",
        "source",
        "player_agent_id",
        "platform_id",
        "platform_body_id",
        "contact_id",
        "shape_roles",
        "normal_role",
        "normal",
        "points",
        "total_impulse",
        "total_ke",
    ),
}


def search_gates_for_mode(mode: str) -> tuple[str, ...]:
    return tuple(gate for gate in DEFAULT_SEARCH_GATES if gate in replay.event_gates_for_mode(mode))


def action_source_for_controller(controller: str) -> str:
    if controller == "scripted":
        return "scripted-contact-fixture"
    if controller == "adaptive":
        return "adaptive-contact-fixture"
    if controller == "edge":
        return "adaptive-edge-contact-fixture"
    raise ValueError(f"Unknown fixture controller {controller!r}")


def _zero_action(space: Any) -> Any:
    size = 10
    try:
        shape = tuple(getattr(space, "shape"))
        if len(shape) == 1:
            size = int(shape[0])
    except Exception:
        pass
    try:
        import numpy as np

        dtype = getattr(space, "dtype", float)
        return np.zeros((size,), dtype=dtype)
    except Exception:
        return [0] * size


def _legal_action(space: Any, pressed: Iterable[int]) -> Any:
    action = _zero_action(space)
    size = len(action)
    for index in pressed:
        if 0 <= int(index) < size:
            action[int(index)] = 1
    return action


def _player0(state: dict[str, Any]) -> dict[str, Any] | None:
    for player in state.get("players", []):
        if isinstance(player, dict) and player.get("agent_id") == 0:
            return player
    return None


def _first_moving_platform(state: dict[str, Any]) -> dict[str, Any] | None:
    for platform in state.get("moving_platforms", []):
        if not isinstance(platform, dict):
            continue
        if isinstance(platform.get("moving_to_w2"), bool):
            return platform
        velocity = platform.get("velocity")
        if isinstance(velocity, list) and any(isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0 for v in velocity[:2]):
            return platform
    return None


def _wrapper_target(state: dict[str, Any]) -> list[float] | None:
    for entry in state.get("wrapper", {}).get("chain", []):
        if not isinstance(entry, dict):
            continue
        target = entry.get("target_position")
        if isinstance(target, list) and len(target) >= 2:
            try:
                return [float(target[0]), float(target[1])]
            except Exception:
                return None
    return None


def _platform_endpoint_target(state: dict[str, Any]) -> list[float] | None:
    player = _player0(state)
    platform = _first_moving_platform(state)
    if player is None or platform is None:
        return None
    geometry = platform.get("geometry")
    world_bb = geometry.get("world_bb") if isinstance(geometry, dict) else None
    if not isinstance(world_bb, dict):
        return None
    try:
        left = float(world_bb["left"])
        right = float(world_bb["right"])
        top = float(world_bb["top"])
    except Exception:
        return None
    velocity = platform.get("velocity")
    vx = float(velocity[0]) if isinstance(velocity, list) and velocity and isinstance(velocity[0], (int, float)) and not isinstance(velocity[0], bool) else 0.0
    if vx > 0:
        return [right, top]
    if vx < 0:
        return [left, top]
    if platform.get("moving_to_w2") is True:
        return [right, top]
    if platform.get("moving_to_w2") is False:
        return [left, top]
    player_pos = player.get("position") if isinstance(player.get("position"), list) else [0.0, 0.0]
    player_x = float(player_pos[0])
    return [left if abs(player_x - left) < abs(player_x - right) else right, top]


class AdaptiveContactController:
    """Deterministic state-feedback fixture controller.

    The controller only chooses legal action vectors from observed state. It
    never mutates the game state, teleports bodies, changes rewards, or credits
    gates directly; replay coverage remains derived from recorded state.
    """

    def __init__(self, mode: str, action_space: Any, *, target: str = "center"):
        if mode not in ("raw", "navigation"):
            raise ValueError(f"Unknown replay mode {mode!r}")
        if target not in ("center", "edge"):
            raise ValueError(f"Unknown fixture target {target!r}")
        self.mode = mode
        self.target = target
        self.action_space = action_space
        self.support_ticks = 0
        self.last_support_id = None

    def _action(self, pressed: Iterable[int]) -> Any:
        return _legal_action(self.action_space, pressed)

    def _raw_action(self, pressed: Iterable[int]) -> dict[int, Any]:
        return {0: self._action(pressed), 1: self._action(())}

    def next_action(self, state: dict[str, Any], tick: int = 0) -> tuple[str, Any]:
        if self.target == "edge":
            label, pressed = self._edge_pressed(state)
            if self.mode == "navigation":
                return label, self._action(pressed)
            return label, self._raw_action(pressed)
        if self.mode == "navigation":
            label, pressed = self._navigation_pressed(state)
            return label, self._action(pressed)
        label, pressed = self._raw_pressed(state, tick)
        return label, self._raw_action(pressed)

    def observe_record(self, record: dict[str, Any]) -> None:
        transition = record.get("transition", {})
        if transition.get("auto_reset") is True or transition.get("post_reset_state") is True:
            self.support_ticks = 0
            self.last_support_id = None
            return
        player = _player0(record.get("state", {}))
        support = player.get("support") if player else None
        support_id = support.get("id") if isinstance(support, dict) and player.get("grounded") is True else None
        if support_id is None:
            self.support_ticks = 0
            self.last_support_id = None
        elif support_id == self.last_support_id:
            self.support_ticks += 1
        else:
            self.last_support_id = support_id
            self.support_ticks = 1

    def _raw_pressed(self, state: dict[str, Any], tick: int) -> tuple[str, tuple[int, ...]]:
        player = _player0(state)
        platform = _first_moving_platform(state)
        if player is None or platform is None:
            return "adaptive_idle", ()
        support = player.get("support")
        if isinstance(support, dict) and support.get("id") == platform.get("id") and player.get("grounded") is True:
            return "adaptive_platform_ride", ()
        player_pos = player.get("position") if isinstance(player.get("position"), list) else [0.0, 0.0]
        platform_pos = platform.get("position") if isinstance(platform.get("position"), list) else [0.0, 0.0]
        dx = float(platform_pos[0]) - float(player_pos[0])
        if abs(dx) <= 0.75:
            return "adaptive_platform_jump", (4,)
        if dx > 0:
            return "adaptive_platform_right", (3,)
        return "adaptive_platform_left", (1,)

    def _navigation_pressed(self, state: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
        player = _player0(state)
        target = _wrapper_target(state)
        if player is None or target is None:
            return "adaptive_target_idle", ()
        player_pos = player.get("position") if isinstance(player.get("position"), list) else [0.0, 0.0]
        dx = float(target[0]) - float(player_pos[0])
        dy = float(target[1]) - float(player_pos[1])
        if abs(dx) <= 0.35:
            return "adaptive_target_settle", ()
        pressed: list[int] = [3 if dx > 0 else 1]
        if dy < -0.35 and player.get("grounded") is True:
            pressed.append(4)
        return ("adaptive_target_right" if dx > 0 else "adaptive_target_left"), tuple(pressed)

    def _edge_pressed(self, state: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
        player = _player0(state)
        target = _platform_endpoint_target(state)
        if player is None or target is None:
            return "adaptive_edge_idle", ()
        player_pos = player.get("position") if isinstance(player.get("position"), list) else [0.0, 0.0]
        dx = float(target[0]) - float(player_pos[0])
        pressed: list[int] = []
        if abs(dx) > 0.18:
            pressed.append(3 if dx > 0 else 1)
        if player.get("grounded") is True and abs(dx) <= 0.45:
            pressed.append(4)
        if 4 in pressed:
            return "adaptive_edge_jump", tuple(pressed)
        if 3 in pressed:
            return "adaptive_edge_right", tuple(pressed)
        if 1 in pressed:
            return "adaptive_edge_left", tuple(pressed)
        return "adaptive_edge_settle", tuple(pressed)


def fixture_labels_for_search(steps: int) -> list[str]:
    """Deterministic long-horizon labels for reset-driven fixture search."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    schedule = (
        ("startup", 5),
        ("walk", 90),
        ("turn", 12),
        ("walk", 45),
        ("jump", 24),
        ("edge", 90),
        ("idle", 180),
        ("platformlanding", 120),
        ("platformriding", 180),
        ("platformreversal", 240),
        ("targethandoff", 180),
        ("timeout", 620),
    )
    labels: list[str] = []
    for label, duration in schedule:
        labels.extend([label] * duration)
        if len(labels) >= steps:
            return labels[:steps]
    while len(labels) < steps:
        labels.append("idle")
    return labels[:steps]


def _platform_by_id(state: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    platforms = {}
    for platform in state.get("moving_platforms", []):
        if isinstance(platform, dict):
            platforms[platform.get("id")] = platform
    return platforms


def _is_nonzero_velocity(platform: dict[str, Any]) -> bool:
    velocity = platform.get("velocity")
    if not isinstance(velocity, list):
        return False
    return any(isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0 for value in velocity[:2])


def _is_moving_platform(platform: dict[str, Any] | None) -> bool:
    if not isinstance(platform, dict):
        return False
    return isinstance(platform.get("moving_to_w2"), bool) or _is_nonzero_velocity(platform)


def _contact_source(contact: dict[str, Any]) -> str | None:
    if contact.get("kind") == "pymunk_arbiter" and contact.get("source") == "body.each_arbiter":
        return "arbiter"
    if contact.get("kind") == "geometric_platform_support" and contact.get("source") == "is_on_floor_bb":
        return "geometric"
    return None


def _controlled_support_id(state: dict[str, Any]) -> Any:
    for player in state.get("players", []):
        if not isinstance(player, dict):
            continue
        if player.get("agent_id") != CONTROLLED_PLAYER_ID or player.get("grounded") is not True:
            continue
        support = player.get("support")
        if isinstance(support, dict) and support.get("kind") == "platform":
            return support.get("id")
    return None


def _platform_velocity_sign(platform: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(platform, dict):
        return (0, 0)
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


def _platform_reversed(previous: dict[str, Any] | None, current: dict[str, Any] | None) -> bool:
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    if previous.get("moving_to_w2") is not None and current.get("moving_to_w2") is not None:
        if previous.get("moving_to_w2") != current.get("moving_to_w2"):
            return True
    previous_sign = _platform_velocity_sign(previous)
    current_sign = _platform_velocity_sign(current)
    return any(c != 0 and p != 0 and c != p for c, p in zip(current_sign, previous_sign))


def _validate_contact_evidence_contract(contact: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    if not isinstance(contact, dict):
        return False, {"reason": "not_object", "field": None, "contact_id": None}
    kind = contact.get("kind")
    required = CONTACT_REQUIRED_FIELDS.get(kind)
    if required is None:
        return False, {"reason": "unsupported_kind", "field": "kind", "contact_id": contact.get("contact_id"), "kind": kind}
    for field in required:
        if field not in contact:
            return False, {
                "reason": "missing_field",
                "field": field,
                "contact_id": contact.get("contact_id"),
                "kind": kind,
            }
    if kind == "geometric_platform_support" and contact.get("source") != "is_on_floor_bb":
        return False, {"reason": "unsupported_source", "field": "source", "contact_id": contact.get("contact_id"), "source": contact.get("source")}
    if kind == "pymunk_arbiter":
        if contact.get("source") != "body.each_arbiter":
            return False, {"reason": "unsupported_source", "field": "source", "contact_id": contact.get("contact_id"), "source": contact.get("source")}
        if contact.get("normal_role") != "shape0_to_shape1_raw":
            return False, {"reason": "unsupported_normal_role", "field": "normal_role", "contact_id": contact.get("contact_id"), "normal_role": contact.get("normal_role")}
        roles = contact.get("shape_roles")
        if not isinstance(roles, list) or len(roles) < 2:
            return False, {"reason": "invalid_shape_roles", "field": "shape_roles", "contact_id": contact.get("contact_id")}
        has_player = any(isinstance(role, dict) and role.get("role") == "player" and role.get("agent_id") == CONTROLLED_PLAYER_ID for role in roles)
        has_platform = any(
            isinstance(role, dict)
            and role.get("role") == "platform"
            and role.get("platform_id") == contact.get("platform_id")
            and role.get("platform_body_id") == contact.get("platform_body_id")
            for role in roles
        )
        if not has_player or not has_platform:
            return False, {"reason": "shape_roles_mismatch", "field": "shape_roles", "contact_id": contact.get("contact_id")}
        if not isinstance(contact.get("points"), list):
            return False, {"reason": "invalid_points", "field": "points", "contact_id": contact.get("contact_id")}
        normal = contact.get("normal")
        if not isinstance(normal, list) or len(normal) < 2:
            return False, {"reason": "invalid_normal", "field": "normal", "contact_id": contact.get("contact_id")}
    return True, None


def _controlled_moving_platform_contacts(state: dict[str, Any]) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str]], list[dict[str, Any]]]:
    platforms = _platform_by_id(state)
    evidence = []
    invalid = []
    for contact in state.get("contacts", []):
        if not isinstance(contact, dict):
            invalid.append({"reason": "not_object", "field": None, "contact_id": None})
            continue
        if contact.get("player_agent_id") is not None and contact.get("player_agent_id") != CONTROLLED_PLAYER_ID:
            continue
        valid, error = _validate_contact_evidence_contract(contact)
        if not valid:
            if error is not None:
                invalid.append(error)
            continue
        if contact.get("player_agent_id") != CONTROLLED_PLAYER_ID:
            continue
        source = _contact_source(contact)
        if source is None:
            continue
        platform = platforms.get(contact.get("platform_id"))
        if not _is_moving_platform(platform):
            continue
        evidence.append((contact, platform, source))
    return evidence, invalid


def _contact_evidence_record(
    gate: str,
    record: dict[str, Any],
    contact: dict[str, Any],
    platform: dict[str, Any],
    source: str,
    previous_platform: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        "gate": gate,
        "tick": record.get("tick"),
        "contact_id": contact.get("contact_id"),
        "source": source,
        "contact_kind": contact.get("kind"),
        "platform_id": contact.get("platform_id"),
        "moving_to_w2": platform.get("moving_to_w2"),
        "velocity": platform.get("velocity"),
    }
    if "normal" in contact:
        item["normal"] = contact.get("normal")
    if previous_platform is not None:
        item["previous_moving_to_w2"] = previous_platform.get("moving_to_w2")
        item["previous_velocity"] = previous_platform.get("velocity")
    return item


def _point_pair(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except Exception:
        return None


def _contact_points(contact: dict[str, Any]) -> list[list[float]]:
    points = []
    for point in contact.get("points", []):
        if not isinstance(point, dict):
            continue
        for name in ("point_a", "point_b"):
            parsed = _point_pair(point.get(name))
            if parsed is not None:
                points.append(parsed)
    return points


def _platform_world_bb(platform: dict[str, Any]) -> dict[str, float] | None:
    geometry = platform.get("geometry")
    world_bb = geometry.get("world_bb") if isinstance(geometry, dict) else None
    if not isinstance(world_bb, dict):
        return None
    try:
        return {name: float(world_bb[name]) for name in ("left", "right", "bottom", "top")}
    except Exception:
        return None


def _endpoint_tolerance(platform: dict[str, Any]) -> float:
    geometry = platform.get("geometry")
    if isinstance(geometry, dict):
        try:
            return float(geometry.get("endpoint_tolerance", 0.05))
        except Exception:
            return 0.05
    return 0.05


def _has_point_near(points: list[list[float]], *, x: float | None = None, y: float | None = None, tolerance: float) -> bool:
    for point in points:
        x_ok = True if x is None else abs(point[0] - x) <= tolerance
        y_ok = True if y is None else abs(point[1] - y) <= tolerance
        if x_ok and y_ok:
            return True
    return False


def _contact_face_gates(contact: dict[str, Any], record: dict[str, Any], platform: dict[str, Any], source: str) -> list[tuple[str, dict[str, Any]]]:
    if source != "arbiter":
        return []
    normal = contact.get("normal")
    if not isinstance(normal, list) or len(normal) < 2:
        return []
    try:
        nx = float(normal[0])
        ny = float(normal[1])
    except Exception:
        return []
    points = _contact_points(contact)
    world_bb = _platform_world_bb(platform)
    if not points or world_bb is None:
        return []
    tolerance = _endpoint_tolerance(platform)
    gates: list[tuple[str, dict[str, Any]]] = []
    base = _contact_evidence_record("", record, contact, platform, source)
    if ny <= -0.5 and _has_point_near(points, y=world_bb["bottom"], tolerance=tolerance):
        gates.append(("contact_underside", {**base, "gate": "contact_underside"}))
    if abs(nx) >= 0.5 and abs(ny) < 0.5 and (
        _has_point_near(points, x=world_bb["left"], tolerance=tolerance)
        or _has_point_near(points, x=world_bb["right"], tolerance=tolerance)
    ):
        gates.append(("contact_side", {**base, "gate": "contact_side"}))
    near_endpoint = (
        _has_point_near(points, x=world_bb["left"], y=world_bb["bottom"], tolerance=tolerance)
        or _has_point_near(points, x=world_bb["left"], y=world_bb["top"], tolerance=tolerance)
        or _has_point_near(points, x=world_bb["right"], y=world_bb["bottom"], tolerance=tolerance)
        or _has_point_near(points, x=world_bb["right"], y=world_bb["top"], tolerance=tolerance)
    )
    if abs(nx) >= 0.5 and abs(ny) >= 0.5 and near_endpoint:
        gates.append(("contact_edge", {**base, "gate": "contact_edge"}))
    return gates


def _platform_event_ids(state: dict[str, Any], previous_state: dict[str, Any] | None, *, reset_boundary: bool) -> dict[str, set[Any]]:
    current_support_id = _controlled_support_id(state)
    platforms = _platform_by_id(state)
    current_platform = platforms.get(current_support_id)
    if current_support_id is None or not _is_moving_platform(current_platform) or reset_boundary:
        return {"platformlanding": set(), "platformriding": set(), "platformreversal": set()}
    previous_support_id = _controlled_support_id(previous_state) if previous_state else None
    previous_platforms = _platform_by_id(previous_state) if previous_state else {}
    previous_platform = previous_platforms.get(current_support_id)
    landed = {current_support_id} if current_support_id != previous_support_id else set()
    riding = {current_support_id} if current_support_id == previous_support_id and _is_moving_platform(previous_platform) else set()
    reversal = {current_support_id} if current_support_id in riding and _platform_reversed(previous_platform, current_platform) else set()
    return {
        "platformlanding": landed,
        "platformriding": riding,
        "platformreversal": reversal,
    }


def summarize_contact_evidence(
    records: Iterable[dict[str, Any]],
    *,
    required_gates: Iterable[str] = CONTACT_EVIDENCE_GATES,
) -> dict[str, Any]:
    record_list = list(records)
    required = list(required_gates)
    evidence: dict[str, list[dict[str, Any]]] = {gate: [] for gate in required}
    invalid_contacts: list[dict[str, Any]] = []
    previous_state = None
    for record in record_list:
        state = record.get("state", {})
        contacts, invalid = _controlled_moving_platform_contacts(state)
        invalid_contacts.extend(
            {**item, "tick": record.get("tick")}
            for item in invalid
        )
        transition = record.get("transition", {})
        reset_boundary = isinstance(transition, dict) and transition.get("post_reset_state") is True
        platform_events = _platform_event_ids(state, previous_state, reset_boundary=reset_boundary)
        previous_platforms = _platform_by_id(previous_state) if previous_state else {}
        for contact, platform, source in contacts:
            for gate in ("platformlanding", "platformriding", "platformreversal"):
                if contact.get("platform_id") in platform_events.get(gate, set()) and gate in evidence:
                    previous_platform = previous_platforms.get(contact.get("platform_id"))
                    evidence[gate].append(_contact_evidence_record(gate, record, contact, platform, source, previous_platform))
            for gate, item in _contact_face_gates(contact, record, platform, source):
                if gate in evidence:
                    evidence[gate].append(item)
        previous_state = state
    found = [gate for gate in required if evidence.get(gate)]
    missing = [gate for gate in required if gate not in found]
    return {
        "required_contact_evidence_gates": required,
        "found_contact_evidence_gates": found,
        "missing_contact_evidence_gates": missing,
        "contact_evidence": evidence,
        "invalid_contact_evidence": invalid_contacts,
    }


def summarize_records(records: Iterable[dict[str, Any]], *, required_gates: Iterable[str] = DEFAULT_SEARCH_GATES) -> dict[str, Any]:
    records = list(records)
    required = list(required_gates)
    contact_summary = summarize_contact_evidence(records)
    if not records:
        return {
            "records": 0,
            "found_event_gates": [],
            "missing_event_gates": required,
            "dispatched_fixture_labels": [],
            "final_coverage": None,
            **contact_summary,
        }
    final_coverage = records[-1]["state"]["coverage"]
    found = sorted(set(final_coverage.get("verified_events", [])) & set(required))
    missing = [gate for gate in required if gate not in found]
    dispatched = []
    for label in final_coverage.get("dispatched_fixture_labels", []):
        if label not in dispatched:
            dispatched.append(label)
    return {
        "records": len(records),
        "found_event_gates": found,
        "missing_event_gates": missing,
        "dispatched_fixture_labels": dispatched,
        "final_coverage": final_coverage,
        **contact_summary,
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metadata_for_fixture_records(
    source_root: Path,
    config_path: Path | None,
    *,
    seed: int,
    mode: str,
    records: Iterable[dict[str, Any]],
    reached_labels: Iterable[str],
    controller: str,
    frame_stack: int | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    record_list = list(records)
    reached = list(reached_labels)
    return replay.default_metadata(
        source_root,
        config_path,
        seed,
        seed,
        replay.hash_actions([record["action"] for record in record_list]),
        mode,
        len(record_list),
        phase=phase,
        frame_stack=frame_stack,
        action_script=reached,
        action_source=action_source_for_controller(controller),
    )


def scan_raw_seed(source_root: Path, output: Path, *, seed: int, steps: int, controller: str = "scripted") -> dict[str, Any]:
    replay.prepare_source_imports(source_root)
    import benchmark_training as bench

    bench.set_headless_environment()
    env = bench.build_raw_env(seed)
    labels = fixture_labels_for_search(steps)
    records = []
    try:
        env.reset(seed=seed)
        actions = replay.scripted_raw_actions(env.action_space, labels) if controller == "scripted" else None
        adaptive = AdaptiveContactController("raw", env.action_space, target="edge" if controller == "edge" else "center") if controller in ("adaptive", "edge") else None
        current_state = replay.capture_env_state(
            env,
            required_event_gates=replay.event_gates_for_mode("raw"),
        ) if adaptive is not None else None
        reached = []
        previous_state = None
        verified_events: set[str] = set()
        for tick in range(steps):
            if adaptive is None:
                label = labels[tick]
                action = actions[tick]
            else:
                label, action = adaptive.next_action(current_state or {}, tick)
            transition = env.step(action)
            reached.append(label)
            record = replay.make_replay_record(
                tick,
                action,
                transition,
                env,
                coverage_labels=reached,
                previous_state=previous_state,
                verified_events=verified_events,
                required_event_gates=replay.event_gates_for_mode("raw"),
            )
            verified_events.update(record["state"]["coverage"]["verified_events"])
            record["state"]["coverage"] = replay.coverage_ledger(
                reached,
                verified_events=verified_events,
                required_event_gates=replay.event_gates_for_mode("raw"),
            )
            records.append(record)
            if adaptive is not None:
                adaptive.observe_record(record)
                current_state = record["state"]
            previous_state = record["state"]
            summary = record["transition"]
            if summary.get("terminated") is True or summary.get("truncated") is True:
                break
    finally:
        env.close()
    metadata = metadata_for_fixture_records(
        source_root,
        None,
        seed=seed,
        mode="raw",
        records=records,
        reached_labels=reached,
        controller=controller,
    )
    replay.write_jsonl(output, metadata, records)
    summary = summarize_records(records, required_gates=search_gates_for_mode("raw"))
    summary.update({"seed": seed, "trace": str(output), "mode": "raw", "controller": controller})
    return summary


def scan_navigation_seed(
    source_root: Path,
    output: Path,
    *,
    config_path: Path,
    wrapper_source: Path,
    seed: int,
    steps: int,
    frame_stack: int,
    controller: str = "scripted",
) -> dict[str, Any]:
    if controller == "scripted":
        result = replay.capture_navigation_replay(
            source_root,
            output,
            config_path=config_path,
            wrapper_source=wrapper_source,
            seed=seed,
            steps=steps,
            labels=fixture_labels_for_search(steps),
            frame_stack=frame_stack,
        )
    else:
        env, _params, phase_config = replay.build_navigation_env(source_root, config_path, wrapper_source, seed, frame_stack)
        adaptive = AdaptiveContactController("navigation", env.action_space, target="edge" if controller == "edge" else "center")
        records = []
        reached = []
        verified_events: set[str] = set()
        try:
            env.reset()
            current_state = replay.capture_env_state(
                env,
                required_event_gates=replay.event_gates_for_mode("navigation"),
            )
            previous_state = None
            for tick in range(steps):
                label, action = adaptive.next_action(current_state, tick)
                transition = env.step([action])
                reached.append(label)
                record = replay.make_replay_record(
                    tick,
                    action,
                    transition,
                    env,
                    coverage_labels=reached,
                    previous_state=previous_state,
                    verified_events=verified_events,
                    required_event_gates=replay.event_gates_for_mode("navigation"),
                )
                verified_events.update(record["state"]["coverage"]["verified_events"])
                record["state"]["coverage"] = replay.coverage_ledger(
                    reached,
                    verified_events=verified_events,
                    required_event_gates=replay.event_gates_for_mode("navigation"),
                )
                records.append(record)
                adaptive.observe_record(record)
                current_state = record["state"]
                previous_state = record["state"]
        finally:
            env.close()
        metadata = metadata_for_fixture_records(
            source_root,
            config_path,
            seed=seed,
            mode="navigation",
            records=records,
            reached_labels=reached,
            controller=controller,
            phase=str(phase_config.phase_key),
            frame_stack=frame_stack,
        )
        replay.write_jsonl(output, metadata, records)
        result = {"output": str(output), "records": len(records), "metadata": metadata}
    records = []
    with output.open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    summary = summarize_records(records, required_gates=search_gates_for_mode("navigation"))
    summary.update({"seed": seed, "trace": str(output), "mode": "navigation", "controller": controller, "capture": result})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "navigation"), default="raw")
    parser.add_argument("--controller", choices=("scripted", "adaptive", "edge"), default="scripted")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--wrapper-source", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--stop-when-complete", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.seed_count <= 0:
        raise SystemExit("--seed-count must be positive")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.mode == "navigation" and (args.config is None or args.wrapper_source is None):
        raise SystemExit("--mode navigation requires --config and --wrapper-source")

    summaries = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        trace = args.output_dir / f"{args.mode}_seed{seed}_contact_fixture.jsonl"
        if args.mode == "navigation":
            summary = scan_navigation_seed(
                args.source_root,
                trace,
                config_path=args.config,
                wrapper_source=args.wrapper_source,
                seed=seed,
                steps=args.steps,
                frame_stack=args.frame_stack,
                controller=args.controller,
            )
        else:
            summary = scan_raw_seed(args.source_root, trace, seed=seed, steps=args.steps, controller=args.controller)
        summaries.append(summary)
        if args.stop_when_complete and not summary["missing_event_gates"]:
            break

    required_gates = search_gates_for_mode(args.mode)
    found = sorted({gate for summary in summaries for gate in summary["found_event_gates"]})
    missing = [gate for gate in required_gates if gate not in found]
    required_contact_gates = list(CONTACT_EVIDENCE_GATES)
    found_contact = sorted({gate for summary in summaries for gate in summary["found_contact_evidence_gates"]})
    missing_contact = [gate for gate in required_contact_gates if gate not in found_contact]
    combined = {
        "contract": "reset_action_stream_only_no_midtrajectory_teleport",
        "mode": args.mode,
        "controller": args.controller,
        "required_event_gates": list(required_gates),
        "required_contact_evidence_gates": required_contact_gates,
        "source_root": str(args.source_root.resolve()),
        "steps": args.steps,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "found_event_gates": found,
        "missing_event_gates": missing,
        "found_contact_evidence_gates": found_contact,
        "missing_contact_evidence_gates": missing_contact,
        "runs": summaries,
    }
    _write_summary(args.output_dir / "contact_fixture_summary.json", combined)
    print(json.dumps(combined, sort_keys=True))
    return 0 if not missing and not missing_contact else 2


if __name__ == "__main__":
    raise SystemExit(main())
