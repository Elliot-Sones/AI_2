#!/usr/bin/env python3
"""Validate the opt-in native game patch against the frozen Python game.

This is a closed-loop scripted parity harness, not a training benchmark.  Each
arm runs in a separate process from the same reset seed: one original Python
game, one with ``native_game_patch`` explicitly installed.  The captured replay
JSONL files are then compared with the existing ``benchmark_replay`` contract.
"""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import source_path

import argparse
import ast
import dataclasses
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence


MAX_STEPS = 1200
DEFAULT_SEEDS = "45,46,47"
RAW_COMBAT_SCHEDULE = (
    ("startup", 5),
    ("approach", 85),
    ("idle", 4),
    ("attack", 45),
    ("idle", 12),
    ("attack", 45),
    ("walk_right", 8),
    ("walk_left", 8),
    ("dodge", 6),
    ("pickup", 6),
    ("jump", 8),
    ("attack", 45),
    ("edge", 10),
    ("idle", 4),
)
NAVIGATION_MASKED_SCHEDULE = (
    ("startup", 5),
    ("walk", 5),
    ("turn", 3),
    ("jump", 4),
    ("edge", 5),
    ("platformlanding", 10),
    ("platformriding", 10),
    ("platformreversal", 10),
    ("targethandoff", 10),
    ("idle", 5),
)


class ValidationError(RuntimeError):
    pass


def _scheduled_reset_ticks(steps: int) -> set[int]:
    return {tick for tick in (300, 600, 900) if tick < steps}


def _expand_schedule(schedule: Iterable[tuple[str, int]], steps: int) -> list[str]:
    if steps < 1 or steps > MAX_STEPS:
        raise ValidationError(f"steps must be between 1 and {MAX_STEPS}")
    labels: list[str] = []
    reset_ticks = _scheduled_reset_ticks(steps)
    schedule_items = list(schedule)
    schedule_index = 0
    while len(labels) < steps:
        tick = len(labels)
        if tick in reset_ticks:
            labels.append("reset")
            continue
        label, count = schedule_items[schedule_index % len(schedule_items)]
        if count <= 0:
            raise ValidationError(f"invalid non-positive duration for {label!r}")
        for _ in range(count):
            if len(labels) >= steps:
                break
            if len(labels) in reset_ticks:
                labels.append("reset")
            else:
                labels.append(label)
        schedule_index += 1
    return labels[:steps]


def raw_combat_labels(steps: int) -> list[str]:
    return _expand_schedule(RAW_COMBAT_SCHEDULE, steps)


def masked_navigation_labels(steps: int) -> list[str]:
    labels = _expand_schedule(NAVIGATION_MASKED_SCHEDULE, steps)
    if any(label == "attack" for label in labels):
        raise ValidationError("masked navigation labels must not include combat attack")
    return labels


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not seeds:
        raise ValidationError("at least one seed is required")
    return seeds


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _action_vector(space: Any, pressed: Iterable[int]) -> Any:
    try:
        import numpy as np

        action = np.zeros(tuple(space.shape), dtype=space.dtype)
    except Exception:
        action = [0] * 10
    for index in pressed:
        action[index] = 1
    return action


def scripted_raw_combat_actions(space: Any, labels: Iterable[str]) -> list[dict[str, Any]]:
    mapping = {
        "startup": (),
        "idle": (),
        "approach": (3,),
        "walk": (3,),
        "walk_right": (3,),
        "walk_left": (1,),
        "turn": (1,),
        "jump": (4,),
        "edge": (3, 2),
        "attack": (7,),
        "dodge": (6,),
        "pickup": (5,),
        "reset": (),
    }
    actions = []
    for label in labels:
        if label not in mapping:
            raise ValidationError(f"unknown raw combat label {label!r}")
        player = _action_vector(space, mapping[label])
        opponent = _action_vector(space, ())
        actions.append(
            {
                "label": label,
                "env_action": {0: player, 1: opponent},
                "record_action": {"label": label, "0": _json_safe(player), "1": _json_safe(opponent)},
            }
        )
    return actions


def _players_from_env(env: Any) -> list[Any]:
    players = getattr(_raw_game_env(env), "players", []) or []
    if isinstance(players, dict):
        return list(players.values())
    return list(players)


def _player_x(player: Any) -> float | None:
    try:
        return float(player.body.position.x)
    except Exception:
        try:
            return float(player.body.position[0])
        except Exception:
            return None


def _object_xy(obj: Any) -> tuple[float, float] | None:
    body = getattr(obj, "body", None)
    position = getattr(body, "position", None)
    if position is None:
        shape = getattr(obj, "shape", None)
        body = getattr(shape, "body", None)
        position = getattr(body, "position", None)
    try:
        return float(position.x), float(position.y)
    except Exception:
        try:
            return float(position[0]), float(position[1])
        except Exception:
            return None


def _set_player_pose(player: Any, x: float, y: float) -> None:
    body = getattr(player, "body", None)
    if body is None:
        raise ValidationError("combat fixture requires player body")
    body.position = (x, y)
    body.velocity = (0.0, 0.0)
    if hasattr(player, "prev_x"):
        player.prev_x = x
    if hasattr(player, "prev_y"):
        player.prev_y = y
    if hasattr(player, "target_vel"):
        player.target_vel = (0.0, 0.0)
    if hasattr(player, "damage_velocity"):
        player.damage_velocity = (0.0, 0.0)


def apply_initial_combat_fixture(env: Any, *, reason: str) -> dict[str, Any]:
    """Place both players in a deterministic, real-game combat start scene.

    This intentionally runs only immediately after a normal environment reset.
    It does not restore a solver snapshot or fabricate hit state; it only sets
    the two player bodies on the existing right-side ground so legal attack
    inputs can exercise damage/stun deterministically in both parity arms.
    """

    raw = _raw_game_env(env)
    players = _players_from_env(raw)
    if len(players) < 2:
        raise ValidationError("combat fixture requires two players")
    objects = getattr(raw, "objects", {}) or {}
    ground = objects.get("ground1") if isinstance(objects, dict) else None
    ground_xy = _object_xy(ground)
    if ground_xy is None:
        surface_x, surface_y = 4.5, 1.0
    else:
        surface_x, surface_y = ground_xy
    y = surface_y - 1.0
    positions = [(surface_x - 0.35, y), (surface_x + 0.45, y)]
    _set_player_pose(players[0], *positions[0])
    _set_player_pose(players[1], *positions[1])
    return {
        "reason": reason,
        "kind": "post_reset_body_pose_only",
        "surface": "ground1",
        "surface_position": [round(surface_x, 6), round(surface_y, 6)],
        "player_positions": [[round(x, 6), round(y, 6)] for x, y in positions],
        "velocity": [0.0, 0.0],
        "solver_state_restored": False,
        "source_evidence": {
            "surface": "environment.py _setup creates ground1 at x=4.5 y=1 width=10",
            "reset_boundary": "fixture is applied only after env.reset or terminal auto-reset",
        },
    }


def _toward_key(from_x: float, to_x: float) -> int:
    return 3 if to_x >= from_x else 1


def _combat_vectors(space: Any, p0_pressed: Iterable[int], p1_pressed: Iterable[int]) -> tuple[Any, Any]:
    return _action_vector(space, p0_pressed), _action_vector(space, p1_pressed)


def combat_coverage_action(env: Any, space: Any, label: str) -> dict[str, Any]:
    """Deterministic actual-game combat controller for raw closed-loop parity.

    The controller only chooses legal action vectors from observed positions; it
    does not mutate positions or teleport game state.  When players are too far
    apart during an attack window, it uses that window to close distance first.
    """

    static_mapping = {
        "startup": ((), (), "idle"),
        "idle": ((), (), "idle"),
        "walk": ((3,), (1,), "approach"),
        "walk_right": ((3,), (), "walk_right"),
        "walk_left": ((1,), (), "walk_left"),
        "turn": ((1,), (), "turn"),
        "jump": ((4,), (), "jump"),
        "edge": ((3, 2), (), "edge"),
        "dodge": ((6,), (), "dodge"),
        "pickup": ((5,), (), "pickup"),
        "reset": ((), (), "reset"),
    }
    if label != "approach" and label != "attack":
        if label not in static_mapping:
            raise ValidationError(f"unknown raw combat label {label!r}")
        p0_pressed, p1_pressed, intent = static_mapping[label]
        p0, p1 = _combat_vectors(space, p0_pressed, p1_pressed)
    else:
        players = _players_from_env(env)
        x0 = _player_x(players[0]) if len(players) >= 1 else None
        x1 = _player_x(players[1]) if len(players) >= 2 else None
        if x0 is None or x1 is None:
            p0_pressed, p1_pressed = ((3,), (1,)) if label == "approach" else ((7,), ())
            intent = "fallback"
        else:
            p0_toward = _toward_key(x0, x1)
            p1_toward = _toward_key(x1, x0)
            if label == "approach" or abs(x1 - x0) > 1.35:
                p0_pressed, p1_pressed = (p0_toward,), (p1_toward,)
                intent = "close_distance"
            else:
                p0_pressed, p1_pressed = (p0_toward, 7), (p1_toward, 7)
                intent = "attack"
        p0, p1 = _combat_vectors(space, p0_pressed, p1_pressed)
    return {
        "label": label,
        "env_action": {0: p0, 1: p1},
        "record_action": {
            "label": label,
            "controller": "deterministic_position_combat_v1",
            "controller_intent": intent,
            "0": _json_safe(p0),
            "1": _json_safe(p1),
        },
    }


def _reset_transition(reset_result: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    if isinstance(reset_result, tuple) and len(reset_result) == 2 and isinstance(reset_result[1], dict):
        observation, info = reset_result
    else:
        observation, info = reset_result, {}
    merged_info = dict(info)
    merged_info["reset_observation"] = True
    return observation, 0.0, False, False, merged_info


GAMEPLAY_EVENT_GATES = ("hit", "KO", "stun")


def gameplay_coverage(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    found: set[str] = set()
    for record in records:
        for event in record.get("discrete_events", []) or []:
            event_text = str(event).lower()
            if "hit" in event_text:
                found.add("hit")
            if "stun" in event_text:
                found.add("stun")
            if "ko" in event_text:
                found.add("KO")
        for player in record.get("state", {}).get("players", []) or []:
            state_name = str(player.get("state", "")).lower() if isinstance(player, dict) else ""
            if "stun" in state_name:
                found.add("stun")
            if "ko" in state_name:
                found.add("KO")
            if not isinstance(player, dict):
                continue
            timers = player.get("state_timers") or {}
            if isinstance(timers, dict) and any("stun" in str(key).lower() and float(value or 0) > 0 for key, value in timers.items()):
                found.add("stun")
            if any(float(player.get(key) or 0) > 0 for key in ("damage", "damage_done", "damage_taken_total", "damage_taken_this_stock")):
                found.add("hit")
            stocks = player.get("stocks")
            if stocks is not None and float(stocks) < 3:
                found.add("KO")
    return {
        "required": list(GAMEPLAY_EVENT_GATES),
        "found": [event for event in GAMEPLAY_EVENT_GATES if event in found],
        "missing": [event for event in GAMEPLAY_EVENT_GATES if event not in found],
    }


def _replay_records(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("type") == "record":
                rows.append(item)
    return rows


def _replay_metadata(path: Path) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("type") == "metadata":
                return item
            return None
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _module_import(module_name: str) -> Any:
    return importlib.import_module(module_name)


def _raw_game_env(env: Any) -> Any:
    seen: set[int] = set()
    queue: list[Any] = [env]
    fallback = env
    for _ in range(32):
        if not queue:
            break
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        players = getattr(current, "players", None)
        if players:
            return current
        for attr in ("raw_env", "env", "unwrapped", "venv", "game", "base_env"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                queue.append(child)
        for child in getattr(current, "envs", []) or []:
            if child is not None and child is not current:
                queue.append(child)
    return fallback


def _dropped_weapon_spawner_qualified_name(env: Any, source_root: Path) -> str | None:
    raw = _raw_game_env(env)
    module_names = [getattr(type(raw), "__module__", ""), "environment.environment"]
    for module_name in module_names:
        if not module_name:
            continue
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = _module_import(module_name)
            except ModuleNotFoundError:
                continue
        cls = getattr(module, "DroppedWeaponSpawner", None)
        if cls is not None:
            return f"{getattr(cls, '__module__', module.__name__)}.{getattr(cls, '__qualname__', 'DroppedWeaponSpawner')}"
    source_path = source_root / "UTMIST-AI2-main" / "environment" / "environment.py"
    if source_path.exists():
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except SyntaxError:
            return None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "DroppedWeaponSpawner":
                return "environment.environment.DroppedWeaponSpawner"
    return None


def _install_native_patch(
    enabled: bool,
    *,
    env: Any,
    source_root: Path,
    native_input: bool,
    native_observation: bool,
    patch_module: Any | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    base_info = {
        "enabled": bool(enabled),
        "enable_input": bool(native_input) if enabled else False,
        "enable_observation": bool(native_observation) if enabled else False,
        "dropped_weapon_spawner_qualified_name": _dropped_weapon_spawner_qualified_name(env, source_root),
    }
    if not enabled:
        return None, base_info
    patch_module = patch_module or _module_import("ai2.simulation.native_game_patch")
    source_path = source_root / "UTMIST-AI2-main" / "environment" / "environment.py"
    if not source_path.exists():
        raise ValidationError(f"missing frozen environment source: {source_path}")
    expected_sha256 = getattr(patch_module, "FROZEN_ENV_SHA256", None)
    install_kwargs = {
        "source_path": source_path,
        "enabled": True,
        "strict": True,
        "enable_input": bool(native_input),
        "enable_observation": bool(native_observation),
    }
    if expected_sha256 is not None:
        install_kwargs["expected_sha256"] = expected_sha256
    patch_target = _raw_game_env(env)
    active = patch_module.install_native_game_patch(patch_target, **install_kwargs)
    description = patch_module.describe_native_game_patch() if hasattr(patch_module, "describe_native_game_patch") else {}
    if active is not None and hasattr(active, "receipt"):
        description = {**description, "active_receipt": active.receipt()}
    return active, {**base_info, "description": description}


def _transition_done(record: dict[str, Any]) -> bool:
    transition = record.get("transition", {})
    return transition.get("terminated") is True or transition.get("truncated") is True


def _capture_raw(
    *,
    source_root: Path,
    config: Path | None,
    output: Path,
    seed: int,
    steps: int,
    native_patch: bool,
    native_input: bool,
    native_observation: bool,
    replay: Any,
    bench: Any,
    patch_module: Any | None,
) -> dict[str, Any]:
    replay.prepare_source_imports(source_root)
    bench.set_headless_environment()
    if hasattr(bench, "seed_everything"):
        bench.seed_everything(seed)
    patch_active = None
    patch_info = {"enabled": False}
    env = None
    try:
        helper_file = getattr(bench, "__file__", None)
        if helper_file is not None and Path(helper_file).resolve() == source_root / "benchmark_training.py":
            env = bench.build_raw_env(seed)
        else:
            env = bench.build_raw_env(seed, source_root=source_root)
        patch_active, patch_info = _install_native_patch(
            native_patch,
            env=env,
            source_root=source_root,
            native_input=native_input,
            native_observation=native_observation,
            patch_module=patch_module,
        )
        env.reset(seed=seed)
        pending_fixture = apply_initial_combat_fixture(env, reason="initial-reset")
        labels = raw_combat_labels(steps)
        records = []
        record_actions = []
        previous_state = None
        verified_events: set[str] = set()
        reset_event = False
        for tick, label in enumerate(labels[:steps]):
            action = combat_coverage_action(env, env.action_space, label)
            record_actions.append(action["record_action"])
            label = labels[tick]
            if label == "reset":
                transition = _reset_transition(env.reset(seed=seed + tick))
                pending_fixture = apply_initial_combat_fixture(env, reason="scripted-reset")
                reset_event = True
                previous_state = None
            else:
                transition = env.step(action["env_action"])
            if pending_fixture is not None:
                action["record_action"]["combat_fixture"] = pending_fixture
                pending_fixture = None
            record = replay.make_replay_record(
                tick,
                action["record_action"],
                transition,
                env,
                coverage_labels=labels[: tick + 1],
                previous_state=previous_state,
                reset_event=reset_event,
                verified_events=verified_events,
                required_event_gates=replay.event_gates_for_mode("raw") if hasattr(replay, "event_gates_for_mode") else (),
            )
            coverage = record.get("state", {}).get("coverage", {})
            verified_events.update(coverage.get("verified_events", []))
            records.append(record)
            if _transition_done(record):
                env.reset(seed=seed + tick + 1)
                pending_fixture = apply_initial_combat_fixture(env, reason="auto-reset")
                reset_event = True
                previous_state = None
            else:
                reset_event = False
                previous_state = record.get("state")
        metadata = replay.default_metadata(
            source_root,
            config,
            seed,
            seed,
            replay.hash_actions(record_actions),
            "raw",
            steps,
            action_script=labels,
            action_source="deterministic-fixtured-combat-native-game-patch-parity-v2",
        )
        metadata["native_patch"] = patch_info
        metadata["combat_fixture"] = {
            "kind": "post_reset_body_pose_only",
            "surface": "ground1",
            "solver_state_restored": False,
            "applied_after": ["initial reset", "scripted reset", "auto reset when terminal"],
            "purpose": "actual damage/stun/respawn parity coverage without changing training workload",
            "source_evidence": {
                "surface": "environment.py _setup creates ground1 at x=4.5 y=1 width=10",
                "reset_boundary": "fixture is applied only after env.reset or terminal auto-reset",
            },
        }
        written = replay.write_jsonl(output, metadata, records)
        return {
            "mode": "raw",
            "seed": seed,
            "output": str(output),
            "records": written,
            "native_patch": patch_info,
            "gameplay_coverage": gameplay_coverage(records),
        }
    finally:
        if patch_active is not None and hasattr(patch_active, "restore"):
            patch_active.restore()
        if env is not None and hasattr(env, "close"):
            env.close()


def _capture_navigation(
    *,
    source_root: Path,
    config: Path,
    wrapper_source: Path | None,
    output: Path,
    seed: int,
    steps: int,
    native_patch: bool,
    native_input: bool,
    native_observation: bool,
    frame_stack: int,
    replay: Any,
    patch_module: Any | None,
) -> dict[str, Any]:
    env = None
    patch_active = None
    patch_info = {"enabled": False}
    try:
        env, _params, phase_config = replay.build_navigation_env(source_root, config, wrapper_source or source_path(source_root, "train.py"), seed, frame_stack)
        patch_active, patch_info = _install_native_patch(
            native_patch,
            env=env,
            source_root=source_root,
            native_input=native_input,
            native_observation=native_observation,
            patch_module=patch_module,
        )
        env.reset()
        labels = masked_navigation_labels(steps)
        actions = replay.scripted_navigation_actions(env.action_space, labels)
        metadata = replay.default_metadata(
            source_root,
            config,
            seed,
            seed,
            replay.hash_actions(actions),
            "navigation",
            steps,
            phase=str(getattr(phase_config, "phase_key", "")),
            frame_stack=frame_stack,
            action_script=labels,
            action_source="scripted-masked-navigation-native-game-patch-parity",
        )
        metadata["native_patch"] = patch_info
        records = []
        previous_state = None
        verified_events: set[str] = set()
        for tick, (label, action) in enumerate(zip(labels, actions)):
            reset_event = False
            if label == "reset":
                transition = _reset_transition(env.reset())
                reset_event = True
                previous_state = None
            else:
                transition = env.step([action])
            record = replay.make_replay_record(
                tick,
                action,
                transition,
                env,
                coverage_labels=labels[: tick + 1],
                previous_state=previous_state,
                reset_event=reset_event,
                verified_events=verified_events,
                required_event_gates=replay.event_gates_for_mode("navigation") if hasattr(replay, "event_gates_for_mode") else (),
            )
            coverage = record.get("state", {}).get("coverage", {})
            verified_events.update(coverage.get("verified_events", []))
            records.append(record)
            previous_state = record.get("state")
        written = replay.write_jsonl(output, metadata, records)
        return {
            "mode": "navigation",
            "seed": seed,
            "output": str(output),
            "records": written,
            "native_patch": patch_info,
            "gameplay_coverage": gameplay_coverage(records),
        }
    finally:
        if patch_active is not None and hasattr(patch_active, "restore"):
            patch_active.restore()
        if env is not None and hasattr(env, "close"):
            env.close()


def capture_run(
    *,
    source_root: Path,
    config: Path | None,
    wrapper_source: Path | None,
    output: Path,
    mode: str,
    seed: int,
    steps: int,
    native_patch: bool,
    frame_stack: int,
    native_input: bool = False,
    native_observation: bool = False,
    replay: Any | None = None,
    bench: Any | None = None,
    patch_module: Any | None = None,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    output = Path(output).resolve()
    if steps < 1 or steps > MAX_STEPS:
        raise ValidationError(f"steps must be between 1 and {MAX_STEPS}")
    replay = replay or _module_import("ai2.evaluation.benchmark_replay")
    original_cwd = Path.cwd()
    try:
        if mode == "raw":
            bench = bench or replay.load_benchmark_helper(source_root)
            return _capture_raw(
                source_root=source_root,
                config=config,
                output=output,
                seed=seed,
                steps=steps,
                native_patch=native_patch,
                native_input=native_input,
                native_observation=native_observation,
                replay=replay,
                bench=bench,
                patch_module=patch_module,
            )
        if mode == "navigation":
            if config is None:
                raise ValidationError("navigation mode requires --config")
            return _capture_navigation(
                source_root=source_root,
                config=Path(config).resolve(),
                wrapper_source=Path(wrapper_source).resolve() if wrapper_source is not None else None,
                output=output,
                seed=seed,
                steps=steps,
                native_patch=native_patch,
                native_input=native_input,
                native_observation=native_observation,
                frame_stack=frame_stack,
                replay=replay,
                patch_module=patch_module,
            )
        raise ValidationError(f"unsupported mode {mode!r}")
    finally:
        os.chdir(original_cwd)


def build_pair_commands(
    *,
    script: Path,
    source_root: Path,
    config: Path | None,
    output_dir: Path,
    mode: str,
    seed: int,
    steps: int,
    wrapper_source: Path | None,
    frame_stack: int,
    native_input: bool = False,
    native_observation: bool = False,
    python: str | None = None,
) -> dict[str, list[str]]:
    python = python or sys.executable
    commands: dict[str, list[str]] = {}
    for arm, enabled in (("original", "0"), ("native", "1")):
        command = [
            python,
            str(Path(script).resolve()),
            "--run-one",
            "--source-root",
            str(Path(source_root).resolve()),
            "--output",
            str((Path(output_dir) / f"{mode}_seed{seed}_{arm}.jsonl").resolve()),
            "--mode",
            mode,
            "--seed",
            str(seed),
            "--steps",
            str(steps),
            f"--native-patch={enabled}",
            "--frame-stack",
            str(frame_stack),
        ]
        if config is not None:
            command.extend(["--config", str(Path(config).resolve())])
        if wrapper_source is not None:
            command.extend(["--wrapper-source", str(Path(wrapper_source).resolve())])
        if enabled == "1":
            if native_input:
                command.append("--native-input")
            if native_observation:
                command.append("--native-observation")
        commands[arm] = command
    return commands


def _comparison_to_dict(result: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(result):
        return dataclasses.asdict(result)
    return {
        "ok": bool(getattr(result, "ok")),
        "reason": getattr(result, "reason", None),
        "message": getattr(result, "message", None),
        "compared_records": getattr(result, "compared_records", None),
        "first_difference": getattr(result, "first_difference", None),
    }


def compare_pair(original: Path, native: Path, receipt: Path, *, replay: Any | None = None, abs_tol: float = 0.0, rel_tol: float = 0.0) -> dict[str, Any]:
    replay = replay or _module_import("ai2.evaluation.benchmark_replay")
    comparison = _comparison_to_dict(replay.compare_replay_files(original, native, abs_tol=abs_tol, rel_tol=rel_tol))
    original_metadata = _replay_metadata(original)
    native_metadata = _replay_metadata(native)
    original_source_sha = (original_metadata or {}).get("source", {}).get("environment_sha256")
    native_source_sha = (native_metadata or {}).get("source", {}).get("environment_sha256")
    if original_source_sha != native_source_sha:
        comparison = {
            "ok": False,
            "reason": "source_mismatch",
            "message": "Original/native replay source hashes differ",
            "compared_records": comparison.get("compared_records", 0),
            "first_difference": {
                "field": "source.environment_sha256",
                "left": original_source_sha,
                "right": native_source_sha,
            },
        }
    coverage = {
        "original": gameplay_coverage(_replay_records(original)),
        "native": gameplay_coverage(_replay_records(native)),
    }
    payload = {
        "status": "passed" if comparison["ok"] else "failed",
        "ok": comparison["ok"],
        "original": str(Path(original).resolve()),
        "native": str(Path(native).resolve()),
        "gameplay_coverage": coverage,
        **comparison,
    }
    _write_json(receipt, payload)
    return payload


def _run_command(command: Sequence[str], timeout_seconds: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": list(command),
            "returncode": 124,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or f"timed out after {timeout_seconds}s")[-4000:] if isinstance(exc.stderr, str) else f"timed out after {timeout_seconds}s",
            "timed_out": True,
        }
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def run_suite(
    *,
    source_root: Path,
    config: Path | None,
    output_dir: Path,
    modes: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    wrapper_source: Path | None,
    frame_stack: int,
    python: str | None = None,
    script: Path | None = None,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
    child_timeout: int = 900,
    native_input: bool = False,
    native_observation: bool = False,
    command_runner: Any | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script = script or Path(__file__).resolve()
    if steps < 1 or steps > MAX_STEPS:
        raise ValidationError(f"steps must be between 1 and {MAX_STEPS}")
    if not modes:
        raise ValidationError("at least one mode is required")
    if not seeds:
        raise ValidationError("at least one seed is required")
    if child_timeout <= 0:
        raise ValidationError("child_timeout must be positive")
    command_runner = command_runner or _run_command
    pairs = []
    for mode in modes:
        if mode == "navigation" and config is None:
            raise ValidationError("navigation mode requires --config")
        for seed in seeds:
            commands = build_pair_commands(
                script=script,
                source_root=source_root,
                config=config,
                output_dir=output_dir,
                mode=mode,
                seed=seed,
                steps=steps,
                wrapper_source=wrapper_source,
                frame_stack=frame_stack,
                native_input=native_input,
                native_observation=native_observation,
                python=python,
            )
            original_run = command_runner(commands["original"], child_timeout)
            native_run = command_runner(commands["native"], child_timeout)
            receipt = output_dir / f"{mode}_seed{seed}_comparison.json"
            if original_run["returncode"] != 0 or native_run["returncode"] != 0:
                comparison = {
                    "status": "failed",
                    "ok": False,
                    "reason": "child_capture_failed",
                    "original_run": original_run,
                    "native_run": native_run,
                }
                _write_json(receipt, comparison)
            else:
                comparison = compare_pair(
                    output_dir / f"{mode}_seed{seed}_original.jsonl",
                    output_dir / f"{mode}_seed{seed}_native.jsonl",
                    receipt,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                )
                comparison["original_run"] = original_run
                comparison["native_run"] = native_run
                _write_json(receipt, comparison)
            pairs.append({"mode": mode, "seed": seed, "receipt": str(receipt), **comparison})
    ok = all(pair.get("ok") is True for pair in pairs)
    summary = {
        "status": "passed" if ok else "failed",
        "ok": ok,
        "contract": "closed_loop_scripted_original_vs_opt_in_native_game_patch",
        "source_root": str(Path(source_root).resolve()),
        "config": str(Path(config).resolve()) if config else None,
        "steps": steps,
        "seeds": list(seeds),
        "modes": list(modes),
        "native_flags": {
            "native_input": bool(native_input),
            "native_observation": bool(native_observation),
        },
        "pairs": pairs,
    }
    _write_json(output_dir / "native_game_patch_validation_summary.json", summary)
    return summary


def _parse_modes(raw: str) -> list[str]:
    if raw == "both":
        return ["raw", "navigation"]
    if raw in ("raw", "navigation"):
        return [raw]
    raise ValidationError(f"unknown mode {raw!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--wrapper-source", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output directory for suite, JSONL file for --run-one")
    parser.add_argument("--mode", choices=("raw", "navigation", "both"), default="both")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--native-patch", choices=("0", "1"), default="0", help=argparse.SUPPRESS)
    parser.add_argument("--native-input", action="store_true", help="Enable native input patch only for the native arm")
    parser.add_argument("--native-observation", action="store_true", help="Enable native observation patch only for the native arm")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--abs-tol", type=float, default=0.0)
    parser.add_argument("--rel-tol", type=float, default=0.0)
    parser.add_argument("--child-timeout", type=int, default=900, help="Per child capture timeout in seconds")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.run_one:
            if args.seed is None:
                raise ValidationError("--run-one requires --seed")
            if args.mode == "both":
                raise ValidationError("--run-one requires a concrete --mode")
            result = capture_run(
                source_root=args.source_root,
                config=args.config,
                wrapper_source=args.wrapper_source,
                output=args.output,
                mode=args.mode,
                seed=args.seed,
                steps=args.steps,
                native_patch=args.native_patch == "1",
                native_input=args.native_input,
                native_observation=args.native_observation,
                frame_stack=args.frame_stack,
            )
            print(json.dumps(_json_safe(result), sort_keys=True))
            return 0
        summary = run_suite(
            source_root=args.source_root,
            config=args.config,
            output_dir=args.output,
            modes=_parse_modes(args.mode),
            seeds=parse_seed_list(args.seeds),
            steps=args.steps,
            wrapper_source=args.wrapper_source,
            frame_stack=args.frame_stack,
            python=args.python,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
            child_timeout=args.child_timeout,
            native_input=args.native_input,
            native_observation=args.native_observation,
        )
        print(json.dumps(_json_safe(summary), sort_keys=True))
        return 0 if summary["ok"] else 2
    except ValidationError as exc:
        payload = {"status": "failed", "ok": False, "reason": "validation_error", "message": str(exc)}
        _write_json(args.output if args.run_one else Path(args.output) / "native_game_patch_validation_summary.json", payload)
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
