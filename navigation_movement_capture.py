#!/usr/bin/env python3
"""Read-only Player.physics_process seam captures for native movement checks.

This captures the movement reducer-visible seam only: decoded input, player
state/timer bank, body position/velocity fields, and oracle facts such as
grounded/platform overlap.  It does not step Pymunk itself, restore solver
state, train a policy, or teleport mid-trajectory states.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable


KEY_NAMES = ("W", "A", "S", "D", "space", "h", "l", "j", "k", "g")
INPUT_WIDTH = 35
NAVIGATION_MASK_ACTION_INDICES = (5, 7, 8, 9)
SUPPORTED_STATE_IDS = {0, 1, 2, 3, 4, 6, 7, 9, 10, 11}
STATE_ID_BY_NAME = {
    "WalkingState": 0,
    "StandingState": 1,
    "TurnaroundState": 2,
    "AirTurnaroundState": 3,
    "SprintingState": 4,
    "StunState": 5,
    "InAirState": 6,
    "DodgeState": 7,
    "AttackState": 8,
    "DashState": 9,
    "BackDashState": 10,
    "KOState": 11,
    "TauntState": 12,
}
STATE_BANK_ORDER = (
    ("walking", 0),
    ("standing", 1),
    ("turnaround", 2),
    ("air_turnaround", 3),
    ("sprinting", 4),
    ("stun", 5),
    ("in_air", 6),
    ("dodge", 7),
    ("attack", 8),
    ("dash", 9),
    ("backdash", 10),
    ("KO", 11),
    ("taunt", 12),
)
SLOT_FIELDS = (
    "valid",
    "invincible_timer",
    "dodge_cooldown",
    "stun_time_stored",
    "jump_timer",
    "jumps_left",
    "recoveries_left",
    "is_base",
    "turnaround_timer",
    "dash_timer",
    "backdash_timer",
    "dodge_timer",
    "dodge_is_grounded",
    "ko_timer",
)


class UnsupportedMovementRecord(ValueError):
    def __init__(self, reason: str, field: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.field = field


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _int(value: Any, field: str) -> int:
    result = int(value)
    if result != value and not isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return result


def _vec2(value: Any, field: str) -> list[float]:
    if hasattr(value, "x") and hasattr(value, "y"):
        return [_finite(value.x, f"{field}.x"), _finite(value.y, f"{field}.y")]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [_finite(value[0], f"{field}[0]"), _finite(value[1], f"{field}[1]")]
    raise ValueError(f"{field} must be a 2-vector")


def _state_name(state: Any) -> str:
    return type(state).__name__


def _state_id(state: Any) -> int:
    name = _state_name(state)
    if name not in STATE_ID_BY_NAME:
        raise UnsupportedMovementRecord("unsupported_movement_state", "before.native_state.state_id", f"unsupported movement state {name}")
    return STATE_ID_BY_NAME[name]


def _facing_int(value: Any) -> int:
    try:
        raw = int(value)
    except Exception:
        raw = 1 if "RIGHT" in str(value) else -1
    return 1 if raw >= 0 else -1


def _velocity_func_id(body: Any) -> int:
    velocity_func = getattr(body, "_velocity_func_base", None)
    if velocity_func is None:
        velocity_func = getattr(body, "velocity_func", None)
    update_velocity = getattr(type(body), "update_velocity", None)
    return 0 if velocity_func is None or velocity_func == update_velocity else 1


def _slot_value(state: Any, name: str) -> Any:
    value = getattr(state, name, None)
    return None if callable(value) else value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _mask_single_action_for_navigation(action: Any) -> Any:
    if isinstance(action, tuple):
        masked = list(action)
    elif isinstance(action, list):
        masked = list(action)
    elif hasattr(action, "copy"):
        masked = action.copy()
    else:
        masked = list(action)
    for index in NAVIGATION_MASK_ACTION_INDICES:
        if index < len(masked):
            masked[index] = 0
    return masked


def mask_action_for_mode(action: Any, mode: str) -> Any:
    if mode == "raw":
        return copy.deepcopy(action)
    if mode != "navigation":
        raise ValueError(f"unsupported movement mask mode {mode!r}")
    if isinstance(action, dict):
        return {key: _mask_single_action_for_navigation(value) for key, value in action.items()}
    return _mask_single_action_for_navigation(action)


def decode_input(player: Any) -> list[int]:
    handler = player.input
    decoded = [0] * INPUT_WIDTH
    statuses = handler.key_status
    for index, key in enumerate(KEY_NAMES):
        status = statuses[key]
        decoded[index] = int(bool(getattr(status, "held", False)))
        decoded[10 + index] = int(bool(getattr(status, "just_pressed", False)))
        decoded[20 + index] = int(bool(getattr(status, "just_released", False)))
    decoded[30] = _int(getattr(handler, "raw_vertical"), "input.raw_vertical")
    decoded[31] = _int(getattr(handler, "raw_horizontal"), "input.raw_horizontal")
    decoded[32] = int(bool(getattr(handler, "no_horizontal")))
    hstate = getattr(handler, "horizontal_state")
    decoded[33] = int(getattr(hstate, "value", 0))
    decoded[34] = _int(getattr(handler, "last_direction", getattr(hstate, "get_last_int", lambda: 0)()), "input.last_direction")
    return decoded


def _slot_for(state: Any) -> dict[str, Any]:
    result = {
        "valid": bool(_slot_value(state, "valid") if _slot_value(state, "valid") is not None else True),
        "invincible_timer": _int(_slot_value(state, "invincible_timer") if _slot_value(state, "invincible_timer") is not None else 0, "slot.invincible_timer"),
        "dodge_cooldown": _int(_slot_value(state, "dodge_cooldown") if _slot_value(state, "dodge_cooldown") is not None else 0, "slot.dodge_cooldown"),
        "stun_time_stored": _int(_slot_value(state, "stun_time_stored") if _slot_value(state, "stun_time_stored") is not None else 0, "slot.stun_time_stored"),
        "jump_timer": _slot_value(state, "jump_timer"),
        "jumps_left": _slot_value(state, "jumps_left"),
        "recoveries_left": _slot_value(state, "recoveries_left"),
        "is_base": _slot_value(state, "is_base"),
        "turnaround_timer": _slot_value(state, "turnaround_timer"),
        "dash_timer": _slot_value(state, "dash_timer"),
        "backdash_timer": _slot_value(state, "backdash_timer"),
        "dodge_timer": _slot_value(state, "dodge_timer"),
        "dodge_is_grounded": _slot_value(state, "is_grounded"),
        "ko_timer": _slot_value(state, "timer"),
    }
    for field in SLOT_FIELDS:
        value = result[field]
        if isinstance(value, bool) or value is None:
            continue
        result[field] = _int(value, f"slot.{field}")
    return result


def movement_bank(player: Any) -> list[dict[str, Any]]:
    states = getattr(player, "states")
    bank = []
    for name, state_id in STATE_BANK_ORDER:
        slot = _slot_for(states[name])
        slot["state_id"] = state_id
        slot["state_name"] = _state_name(states[name])
        bank.append(slot)
    return bank


def native_state_dict(player: Any) -> dict[str, Any]:
    body = player.body
    position = _vec2(body.position, "body.position")
    velocity = _vec2(body.velocity, "body.velocity")
    start = _vec2(player.start_position, "player.start_position")
    damage_velocity = _vec2(getattr(player, "damage_velocity", (0.0, 0.0)), "player.damage_velocity")
    target_vel = _vec2(getattr(player, "target_vel", (0.0, 0.0)), "player.target_vel")
    return {
        "x": position[0],
        "y": position[1],
        "vx": velocity[0],
        "vy": velocity[1],
        "start_x": start[0],
        "start_y": start[1],
        "damage_vx": damage_velocity[0],
        "damage_vy": damage_velocity[1],
        "target_vx": target_vel[0],
        "target_vy": target_vel[1],
        "prev_x": _finite(getattr(player, "prev_x"), "player.prev_x"),
        "prev_y": _finite(getattr(player, "prev_y"), "player.prev_y"),
        "facing": _facing_int(getattr(player, "facing")),
        "state_id": _state_id(player.state),
        "state_name": _state_name(player.state),
        "stocks": _int(getattr(player, "stocks"), "player.stocks"),
        "velocity_func": _velocity_func_id(body),
        "body_type": _int(getattr(body, "body_type"), "body.body_type"),
        "bank": movement_bank(player),
    }


def _is_grounded(player: Any) -> bool:
    checker = getattr(player, "is_on_floor", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return bool(getattr(player, "grounded", False))


def _platform_overlap_and_velocity(player: Any) -> tuple[bool, list[float]]:
    platform = getattr(player, "on_platform", None)
    platform_obj = None
    objects = getattr(getattr(player, "env", None), "objects", {})
    if platform is not None:
        for obj in objects.values():
            if getattr(obj, "body", None) is platform:
                platform_obj = obj
                break
    if platform_obj is None:
        platform_obj = objects.get("platform1") if isinstance(objects, dict) else None
    overlap = False
    if platform_obj is not None:
        player_shape = getattr(player, "shape", None)
        platform_shape = getattr(platform_obj, "shape", None)
        try:
            overlap = bool(player_shape.cache_bb().intersects(platform_shape.cache_bb()))
        except Exception:
            overlap = False
        velocity = _vec2(getattr(getattr(platform_obj, "body", None), "velocity", (0.0, 0.0)), "platform.velocity")
    else:
        velocity = [0.0, 0.0]
    return overlap, velocity


def movement_facts(player: Any) -> dict[str, Any]:
    overlap, velocity = _platform_overlap_and_velocity(player)
    position = _vec2(player.body.position, "body.position")
    start = _vec2(player.start_position, "player.start_position")
    return {
        "grounded": _is_grounded(player),
        "platform_overlap": overlap,
        "platform_velocity": velocity,
        "on_platform_present": getattr(player, "on_platform", None) is not None,
        "at_start_position": all(math.isclose(position[i], start[i], rel_tol=0.0, abs_tol=1e-9) for i in range(2)),
    }


def capture_player_movement(player: Any, *, tick: int, player_label: str) -> dict[str, Any]:
    decoded = decode_input(player)
    return {
        "type": "movement_record",
        "schema": "ai2-navigation-movement-capture-v1",
        "tick": tick,
        "player_label": player_label,
        "agent_id": getattr(player, "agent_id", None),
        "input_width": INPUT_WIDTH,
        "input": {"decoded": decoded},
        "facts": movement_facts(player),
        "before": {"native_state": native_state_dict(player)},
    }


def finish_player_movement(record: dict[str, Any], player: Any) -> dict[str, Any]:
    output = copy.deepcopy(record)
    output["after"] = {"native_state": native_state_dict(player)}
    output["post_facts"] = movement_facts(player)
    return output


class PlayerMovementRecorder:
    """Temporarily wrap Player.physics_process on concrete player instances."""

    def __init__(self, players: Iterable[Any]):
        self.players = list(players)
        self.records: list[dict[str, Any]] = []
        self._originals: dict[int, tuple[Any, Any, Any]] = {}
        self._action_provenance: dict[str, Any] | None = None

    def set_action_provenance(self, *, action_index: int, raw_action: Any, executed_action: Any, mask_mode: str) -> None:
        self._action_provenance = {
            "action_index": action_index,
            "mask_mode": mask_mode,
            "raw_action": _json_safe(raw_action),
            "executed_action": _json_safe(executed_action),
        }

    def attach(self, players: Iterable[Any] | None = None) -> None:
        if players is not None:
            self.players = list(players)
        for index, player in enumerate(self.players):
            if id(player) in self._originals:
                continue
            original = player.physics_process
            label = f"players/{getattr(player, 'agent_id', index)}"

            def wrapped(delta, *, _player=player, _original=original, _label=label):
                dt = _finite(delta, "dt")
                before = capture_player_movement(_player, tick=len(self.records), player_label=_label)
                result = _original(dt)
                record = {**finish_player_movement(before, _player), "dt": dt}
                if self._action_provenance is not None:
                    record["action_provenance"] = copy.deepcopy(self._action_provenance)
                self.records.append(record)
                return result

            self._originals[id(player)] = (player, original, wrapped)
            player.physics_process = wrapped

    def restore(self) -> None:
        for player, original, wrapped in list(self._originals.values()):
            if getattr(player, "physics_process", None) is wrapped:
                player.physics_process = original
        self._originals.clear()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *_):
        self.restore()


def _slot_to_native(movement: Any, slot: dict[str, Any]) -> Any:
    values = {field: slot.get(field) for field in SLOT_FIELDS}
    return movement.MovementSlot(**values)


def _state_to_native(movement: Any, state: dict[str, Any]) -> Any:
    if state["state_id"] not in SUPPORTED_STATE_IDS:
        raise UnsupportedMovementRecord("unsupported_movement_state", "before.native_state.state_id", f"unsupported movement state {state['state_name']}")
    return movement.MovementState(
        x=state["x"],
        y=state["y"],
        vx=state["vx"],
        vy=state["vy"],
        start_x=state["start_x"],
        start_y=state["start_y"],
        damage_vx=state["damage_vx"],
        damage_vy=state["damage_vy"],
        target_vx=state["target_vx"],
        target_vy=state["target_vy"],
        prev_x=state["prev_x"],
        prev_y=state["prev_y"],
        facing=state["facing"],
        state_id=state["state_id"],
        stocks=state["stocks"],
        velocity_func=state["velocity_func"],
        body_type=state["body_type"],
        bank=tuple(_slot_to_native(movement, slot) for slot in state["bank"]),
    )


def _first_difference(actual: Any, expected: Any, field: str) -> str | None:
    if hasattr(actual, "__dataclass_fields__"):
        actual = {name: getattr(actual, name) for name in actual.__dataclass_fields__}
    if isinstance(expected, dict) and "state_name" in expected:
        expected = {key: value for key, value in expected.items() if key != "state_name"}
    if isinstance(expected, dict) and ".bank[" in field:
        expected = {key: value for key, value in expected.items() if key not in ("state_id", "state_name")}
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in sorted(set(actual) | set(expected)):
            if key not in actual or key not in expected:
                return f"{field}.{key}"
            child = _first_difference(actual[key], expected[key], f"{field}.{key}")
            if child:
                return child
        return None
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return field
        for index, (a_item, e_item) in enumerate(zip(actual, expected)):
            child = _first_difference(a_item, e_item, f"{field}[{index}]")
            if child:
                return child
        return None
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and not isinstance(actual, bool) and not isinstance(expected, bool):
        return None if math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9) else field
    return None if actual == expected else field


def _native_case(movement: Any, record: dict[str, Any]) -> tuple[Any, list[int], bool, bool, tuple[float, float]]:
    state = _state_to_native(movement, record["before"]["native_state"])
    facts = record["facts"]
    return (
        state,
        record["input"]["decoded"],
        bool(facts["grounded"]),
        bool(facts["platform_overlap"]),
        tuple(facts["platform_velocity"]),
    )


def _compare_native_result(record: dict[str, Any], result: Any) -> dict[str, Any]:
    field = _first_difference(result.state, record["after"]["native_state"], "after.native_state")
    if field:
        return {"ok": False, "reason": "value_mismatch", "field": field, "events": result.events}
    return {"ok": True, "reason": "matched", "events": result.events}


def compare_record_with_native(record: dict[str, Any], *, backend: str = "python", c_reference: Any = None) -> dict[str, Any]:
    try:
        import native_navigation_movement as movement

        state, decoded, grounded, platform_overlap, platform_velocity = _native_case(movement, record)
        config = movement.default_config()
        if backend == "python":
            result = movement.step_movement(
                config,
                state,
                decoded,
                grounded=grounded,
                platform_overlap=platform_overlap,
                platform_velocity=platform_velocity,
            )
        elif backend == "c":
            reference = c_reference or movement.load_c_reference()
            result = reference.step(
                config,
                state,
                decoded,
                grounded=grounded,
                platform_overlap=platform_overlap,
                platform_velocity=platform_velocity,
            )
        else:
            raise ValueError(f"unsupported native movement backend {backend!r}")
    except UnsupportedMovementRecord as exc:
        return {"ok": False, "reason": exc.reason, "field": exc.field, "message": str(exc)}
    except ValueError as exc:
        return {"ok": False, "reason": "native_reducer_rejected", "field": "before.native_state", "message": str(exc)}
    return _compare_native_result(record, result)


def compare_records_with_native(records: Iterable[dict[str, Any]], *, backend: str = "python") -> dict[str, Any]:
    records = list(records)
    failures = []
    if backend == "cuda":
        try:
            import native_navigation_movement as movement

            config = movement.default_config()
            reference = movement.load_cuda_reference()
            cases = [_native_case(movement, record) for record in records]
            results = reference.step_batch(config, cases)
            for index, (record, result) in enumerate(zip(records, results)):
                item = _compare_native_result(record, result)
                if not item["ok"]:
                    failures.append({"index": index, "tick": record.get("tick"), **item})
        except UnsupportedMovementRecord as exc:
            failures.append({"index": len(failures), "tick": None, "ok": False, "reason": exc.reason, "field": exc.field, "message": str(exc)})
        except Exception as exc:
            failures.append({"index": len(failures), "tick": None, "ok": False, "reason": "native_reducer_rejected", "field": "before.native_state", "message": str(exc)})
    else:
        c_reference = None
        if backend == "c":
            import native_navigation_movement as movement

            c_reference = movement.load_c_reference()
        for index, record in enumerate(records):
            item = compare_record_with_native(record, backend=backend, c_reference=c_reference)
            if not item["ok"]:
                failures.append({"index": index, "tick": record.get("tick"), **item})
    return {
        "backend": backend,
        "records": len(records),
        "ok": not failures,
        "failures": failures,
        "first_failure": failures[0] if failures else None,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_file(source_root: Path) -> Path:
    return source_root / "UTMIST-AI2-main" / "environment" / "environment.py"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verification_print_summary(verification: dict[str, Any], *, sample_size: int = 5) -> dict[str, Any]:
    summary = {key: value for key, value in verification.items() if key != "failures"}
    failures = verification.get("failures")
    if isinstance(failures, list):
        summary["failure_count"] = len(failures)
        summary["failure_sample"] = failures[:sample_size]
    return summary


def capture_game_movement(
    source_root: Path,
    output: Path,
    *,
    seed: int = 0,
    steps: int = 120,
    action_seed: int | None = None,
    source_sha256: str | None = None,
    prepare_source: Callable[[Path], None] | None = None,
    build_env: Callable[[int], Any] | None = None,
    make_actions: Callable[[Any, int, int, int], list[Any]] | None = None,
    step_env: Callable[[Any, Any], Any] | None = None,
    verify_backend: str = "python",
    mask_mode: str = "navigation",
) -> dict[str, Any]:
    if steps < 0 or steps > 1200:
        raise ValueError("steps must be between 0 and 1200")
    source_root = source_root.resolve()
    output = output.resolve()
    source_path = _source_file(source_root)
    actual_sha = sha256_file(source_path)
    if source_sha256 is not None and actual_sha != source_sha256:
        raise ValueError("source sha mismatch")
    action_seed = seed if action_seed is None else action_seed
    original_cwd = Path.cwd()
    env = None
    try:
        if prepare_source is None:
            import benchmark_replay

            prepare_source = benchmark_replay.prepare_source_imports
        prepare_source(source_root)
        if build_env is None or make_actions is None:
            import benchmark_training as bench

            bench.set_headless_environment()
            build_env = build_env or bench.build_raw_env
            make_actions = make_actions or bench.action_plan
            try:
                import torch

                torch.set_num_threads(1)
            except Exception:
                pass
        env = build_env(seed)
        if hasattr(env, "reset"):
            env.reset(seed=seed)
        actions = make_actions(getattr(env, "action_space", None), steps, action_seed, 2)
        recorder = PlayerMovementRecorder(getattr(env, "players", []))
        try:
            with recorder:
                for action_index, action in enumerate(actions):
                    raw_action = copy.deepcopy(action)
                    executed_action = mask_action_for_mode(action, mask_mode)
                    recorder.set_action_provenance(
                        action_index=action_index,
                        raw_action=raw_action,
                        executed_action=executed_action,
                        mask_mode=mask_mode,
                    )
                    recorder.attach(getattr(env, "players", []))
                    (step_env or (lambda game, act: game.step(act)))(env, executed_action)
        finally:
            recorder.restore()
        payload = {
            "contract": "actual_player_physics_process_pre_post_observation_only",
            "source": {"root": str(source_root), "environment_sha256": actual_sha},
            "seed": seed,
            "action_seed": action_seed,
            "steps_requested": steps,
            "mask_mode": mask_mode,
            "records_captured": len(recorder.records),
            "records": recorder.records,
            "native_verification": compare_records_with_native(recorder.records, backend=verify_backend),
        }
        _write_json(output, payload)
        return payload
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        os.chdir(original_cwd)


def main(argv: list[str] | None = None, *, capture_fn: Callable[..., dict[str, Any]] = capture_game_movement) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-seed", type=int)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--verify-backend", choices=("python", "c", "cuda"), default="python")
    parser.add_argument("--mask-mode", choices=("navigation", "raw"), default="navigation")
    args = parser.parse_args(argv)
    payload = capture_fn(
        source_root=args.source_root,
        output=args.output,
        seed=args.seed,
        action_seed=args.action_seed,
        steps=args.steps,
        source_sha256=args.source_sha256,
        verify_backend=args.verify_backend,
        mask_mode=args.mask_mode,
    )
    verification = payload.get("native_verification", {})
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records_captured": payload["records_captured"],
                "native_verification": _verification_print_summary(verification),
            },
            sort_keys=True,
        )
    )
    return 0 if verification.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
