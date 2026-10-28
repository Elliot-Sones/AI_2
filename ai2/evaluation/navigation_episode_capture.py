#!/usr/bin/env python3
"""Closed-loop capture of the frozen navigation environment for native replay.

Runs the real frozen training stack (SelfPlayWarehouseBrawl, Float32Wrapper,
FrozenOpponentWrapper in the configured navigation phase) with a controller and
records, per episode, the reset draws and reset observation, then per tick the
action handed to the wrapper, every random draw the game made during the step
(random.getrandbits/uniform/randint and np.random.randint), the wrapper outputs
(observation bits, reward bits, terminated, truncated, navigation info) and a
diagnostic snapshot of the hidden state: both players' movement state and
input decoder, the platform, the weapon spawners, the wrapper counters and the
Chipmunk bodies/broadphase/arbiter cache read from the live space.

The native closed loop (native/navigation_episode.h) is driven only by the
recorded actions and draws; the snapshots exist to name the first divergent
field when the outputs disagree.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import source_path

import argparse
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = PROJECT_ROOT
if sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))

from ai2.evaluation.navigation_contact_capture import SHAPE, SpaceRecorder, _addr, _u64, find_raw_env, random_actions, sha256_file, validate_layout, velocity_mode_of  # noqa: E402

SCHEMA = "ai2.navigation_episode_capture.v1"
KEY_NAMES = ("W", "A", "S", "D", "space", "h", "l", "j", "k", "g")
STATE_IDS = {
    "WalkingState": 0, "StandingState": 1, "TurnaroundState": 2, "AirTurnaroundState": 3, "SprintingState": 4, "StunState": 5,
    "InAirState": 6, "DodgeState": 7, "AttackState": 8, "DashState": 9, "BackDashState": 10, "KOState": 11, "TauntState": 12,
}
STATE_KEYS = (("walking", 0), ("standing", 1), ("turnaround", 2), ("air_turnaround", 3), ("sprinting", 4), ("stun", 5), ("in_air", 6),
              ("dodge", 7), ("attack", 8), ("dash", 9), ("backdash", 10), ("KO", 11), ("taunt", 12))
SLOT_ATTRS = ("invincible_timer", "dodge_cooldown", "stun_time_stored", "jump_timer", "jumps_left", "recoveries_left", "is_base",
              "turnaround_timer", "dash_timer", "backdash_timer", "dodge_timer", "is_grounded", "timer")
NAV_INFO_KEYS = ("nav_touched", "nav_reached", "nav_fall", "nav_success", "nav_terminal_horizon", "nav_hold_steps", "nav_post_goal_steps",
                 "nav_sequence_complete")
DRAW_KINDS = {"getrandbits1": 1, "uniform": 2, "randint": 3, "np_randint": 4}


class DrawRecorder:
    """Records the game's random draws by patching the module-level functions it calls."""

    def __init__(self) -> None:
        self.draws: list[dict[str, Any]] = []
        self._originals: tuple[Any, ...] | None = None

    def add(self, kind: int, a: float, b: float, value: float, **extra: Any) -> None:
        entry = {"kind": kind, "a": float(a), "b": float(b), "value": float(value)}
        entry.update(extra)
        self.draws.append(entry)

    def take(self) -> list[dict[str, Any]]:
        out = list(self.draws)
        self.draws.clear()
        return out

    def install(self) -> None:
        import random as random_module

        import numpy as np

        if self._originals is not None:
            return
        orig_getrandbits = random_module.getrandbits
        orig_uniform = random_module.uniform
        orig_randint = random_module.randint
        orig_np_randint = np.random.randint
        recorder = self

        def getrandbits(k: int) -> int:
            value = orig_getrandbits(k)
            recorder.add(DRAW_KINDS["getrandbits1"] if k == 1 else -1, k, 0.0, value, name="getrandbits")
            return value

        def uniform(a: float, b: float) -> float:
            value = orig_uniform(a, b)
            recorder.add(DRAW_KINDS["uniform"], a, b, value, name="uniform")
            return value

        def randint(a: int, b: int) -> int:
            value = orig_randint(a, b)
            recorder.add(DRAW_KINDS["randint"], a, b, value, name="randint")
            return value

        def np_randint(*args: Any, **kwargs: Any) -> Any:
            value = orig_np_randint(*args, **kwargs)
            if len(args) == 1 and not kwargs:
                recorder.add(DRAW_KINDS["np_randint"], int(args[0]), 0.0, int(value), name="np_randint")
            else:
                recorder.add(-1, 0.0, 0.0, 0.0, name="np_randint_unsupported", args=[str(a) for a in args])
            return value

        self._originals = (orig_getrandbits, orig_uniform, orig_randint, orig_np_randint)
        random_module.getrandbits = getrandbits  # type: ignore[assignment]
        random_module.uniform = uniform  # type: ignore[assignment]
        random_module.randint = randint  # type: ignore[assignment]
        np.random.randint = np_randint  # type: ignore[assignment]

    def restore(self) -> None:
        import random as random_module

        import numpy as np

        if self._originals is None:
            return
        random_module.getrandbits, random_module.uniform, random_module.randint, np.random.randint = self._originals  # type: ignore[assignment]
        self._originals = None


def _int_value(value: Any, path: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != int(value):
            raise ValueError(f"{path}: non-integer timer {value!r}")
        return int(value)
    raise ValueError(f"{path}: unsupported value {value!r}")


def decode_input(player: Any) -> list[int]:
    handler = player.input
    decoded = [0] * 35
    for index, key in enumerate(KEY_NAMES):
        status = handler.key_status[key]
        decoded[index] = int(bool(status.held))
        decoded[10 + index] = int(bool(status.just_pressed))
        decoded[20 + index] = int(bool(status.just_released))
    decoded[30] = _int_value(handler.raw_vertical, "input.raw_vertical")
    decoded[31] = _int_value(handler.raw_horizontal, "input.raw_horizontal")
    decoded[32] = int(bool(handler.no_horizontal))
    decoded[33] = int(handler.horizontal_state.value)
    decoded[34] = _int_value(handler.last_direction, "input.last_direction")
    return decoded


def player_snapshot(player: Any) -> dict[str, Any]:
    body = player.body
    bank: dict[str, dict[str, int]] = {}
    for name, sid in STATE_KEYS:
        obj = player.states[name]
        slot = {}
        for attr in SLOT_ATTRS:
            if attr in obj.__dict__:
                slot[attr] = _int_value(obj.__dict__[attr], f"bank[{sid}].{attr}")
        bank[str(sid)] = slot
    return {
        "x": float(body.position.x), "y": float(body.position.y), "vx": float(body.velocity.x), "vy": float(body.velocity.y),
        "facing": int(player.facing), "state": type(player.state).__name__, "state_id": STATE_IDS[type(player.state).__name__],
        "stocks": int(player.stocks), "velocity_mode": velocity_mode_of(body), "damage": float(player.damage), "weapon": str(player.weapon),
        "start": [float(player.start_position[0]), float(player.start_position[1])],
        "input": decode_input(player), "bank": bank,
    }


def wrapper_snapshot(wrapper: Any) -> dict[str, Any]:
    target = wrapper.target_position
    return {
        "target": None if target is None else [float(target[0]), float(target[1])],
        "last_distance": None if wrapper.last_distance is None else float(wrapper.last_distance),
        "episode_steps": int(wrapper.episode_steps), "targets_this_episode": int(wrapper.targets_this_episode),
        "hold_steps": int(wrapper.hold_steps), "post_goal_steps": int(wrapper.post_goal_steps),
        "goal_reached_step": None if wrapper.goal_reached_step is None else int(wrapper.goal_reached_step),
        "target_touched": bool(wrapper.target_touched), "fell_this_episode": bool(wrapper.fell_this_episode), "last_stocks": int(wrapper.last_stocks),
    }


def wrapper_config(wrapper: Any) -> dict[str, Any]:
    return {
        "min_dist": float(wrapper.min_dist), "max_dist": float(wrapper.max_dist), "fall_penalty": float(wrapper.fall_penalty),
        "completion_mode": str(wrapper.completion_mode), "control_mode": str(wrapper.control_mode), "settle_speed": float(wrapper.settle_speed),
        "max_targets": int(wrapper.max_targets), "time_limit": int(wrapper.time_limit), "settle_steps": int(wrapper.settle_steps),
        "post_goal_required_steps": int(wrapper.post_goal_required_steps), "grounded_positions": len(wrapper.grounded_positions),
    }


def spawner_snapshot(spawner: Any, rec: SpaceRecorder) -> dict[str, Any]:
    weapon = spawner.active_weapon
    out = {
        "world_pos": [float(spawner.world_pos[0]), float(spawner.world_pos[1])],
        "og_cooldown_frames": int(spawner.og_cooldown_frames), "cooldown_frames": int(spawner.cooldown_frames),
        "last_spawn_frame": int(spawner.last_spawn_frame), "active": weapon is not None,
    }
    if weapon is not None:
        out["weapon_name"] = str(weapon.name)
        out["spawn_frame"] = int(weapon.spawn_frame)
        if weapon.shape is not None:
            out["hashid"] = _u64(_addr(weapon.shape._shape), SHAPE["hashid"])
    return out


def _hash_map(rec: SpaceRecorder) -> dict[int, int]:
    return {c["index"]: int(c["hashid"]) for c in rec.shape_config()}


def world_snapshot(rec: SpaceRecorder) -> dict[str, Any]:
    rec.sync_extras()
    hashes = _hash_map(rec)
    bodies = {}
    for cfg in rec.shape_config():
        bodies[str(cfg["hashid"])] = {
            "label": cfg["label"], "body": rec.body_state(cfg["body"]),
            "shape": {k: cfg[k] for k in ("verts", "radius", "elasticity", "friction", "collision_type", "filter", "sensor", "surface_velocity")},
        }
    tree = rec.tree_state()
    leaves = {}
    for key, leaf in tree["leaves"].items():
        leaves[str(hashes[int(key)])] = {"bb": leaf["bb"], "stamp": leaf["stamp"],
                                         "pairs": [[hashes.get(a, -2), hashes.get(b, -2), pid] for a, b, pid in leaf["pairs"]]}
    tree_out = {"stamp": tree["stamp"], "leaves": leaves, "order": [hashes[k] for k in tree["order"]]}

    def arbiter(entry: dict[str, Any]) -> dict[str, Any]:
        a, b = hashes.get(entry["a"], -1), hashes.get(entry["b"], -1)
        return {"a": a, "b": b, "e": entry["e"], "u": entry["u"], "surface_vr": entry["surface_vr"], "n": entry["n"], "count": entry["count"],
                "swapped": entry["swapped"], "stamp": entry["stamp"], "state": entry["state"], "contacts": entry["contacts"]}

    cache = {}
    for entry in rec.cache_list():
        arb = arbiter(entry)
        cache[f"{min(arb['a'], arb['b'])}-{max(arb['a'], arb['b'])}"] = arb
    solve = [arbiter(entry) for entry in rec.solve_list()]
    return {"bodies": bodies, "tree": tree_out, "cache": cache, "solve": solve, "stamp": rec.snapshot()["stamp"],
            "shape_id_counter": rec.space_config()["shape_id_counter"]}


def snapshot(raw: Any, wrapper: Any, rec: SpaceRecorder) -> dict[str, Any]:
    platform = raw.objects["platform1"]
    return {
        "players": [player_snapshot(p) for p in raw.players],
        "platform": {"x": float(platform.body.position.x), "y": float(platform.body.position.y), "vx": float(platform.body.velocity.x),
                     "vy": float(platform.body.velocity.y), "moving_to_w2": bool(platform.moving_to_w2),
                     "waypoint1": [float(v) for v in platform.waypoint1], "waypoint2": [float(v) for v in platform.waypoint2]},
        "spawners": [spawner_snapshot(s, rec) for s in raw.weapon_controller.spawners],
        "wrapper": wrapper_snapshot(wrapper),
        "raw": {"steps": int(raw.steps), "terminated": bool(getattr(raw, "terminated", False))},
        "world": world_snapshot(rec),
    }


def _obs_list(obs: Any) -> list[float]:
    import numpy as np

    arr = np.asarray(obs)
    if arr.dtype != np.float32 or arr.shape != (64,):
        raise ValueError(f"unexpected wrapper observation {arr.dtype} {arr.shape}")
    return [float(x) for x in arr.tolist()]


def _info_flags(info: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in NAV_INFO_KEYS:
        value = info.get(key, 0)
        out[key] = int(value) if isinstance(value, (bool, int)) else value
    return out


def capture(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pymunk

    replay = importlib.import_module('ai2.evaluation.benchmark_replay')
    validator = importlib.import_module('tools.validation.validate_native_navigation')
    source_root = Path(args.source_root).resolve()
    config = Path(args.config).resolve()
    wrapper_source = Path(args.wrapper_source).resolve() if args.wrapper_source else source_path(source_root, "train.py")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    original_cwd = Path.cwd()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows: list[dict[str, Any]] = []
    episodes = 0
    ticks = 0
    recorder = DrawRecorder()
    wrapper_cfg: dict[str, Any] | None = None
    try:
        for seed in seeds:
            env = None
            try:
                env, _params, _phase = replay.build_navigation_env(source_root, config, wrapper_source, seed, frame_stack=args.frame_stack)
                wrapper = validator.find_frozen_opponent_wrapper(env)
                wrapper_cfg = wrapper_config(wrapper)
                recorder.install()
                episode = 0

                def do_reset() -> tuple[Any, SpaceRecorder]:
                    recorder.take()
                    obs, _info = wrapper.reset()
                    draws = recorder.take()
                    raw_env = find_raw_env(wrapper)
                    rec = SpaceRecorder(raw_env)
                    layout = validate_layout(rec)
                    rows.append({"header": {"seed": seed, "episode": episode, "draws": draws, "obs": _obs_list(obs), "layout": layout,
                                            "space": rec.space_config(), "state": snapshot(raw_env, wrapper, rec)}})
                    return raw_env, rec

                raw_env, rec = do_reset()
                episodes += 1
                if args.controller == "adaptive":
                    actions = validator._adaptive_actions(wrapper, args.steps)
                elif args.controller == "scripted":
                    actions = validator._scripted_actions(wrapper, args.steps)
                else:
                    actions = random_actions(wrapper.action_space, args.steps, seed)
                for tick, (label, action) in enumerate(actions):
                    recorder.take()
                    obs, reward, terminated, truncated, info = wrapper.step(action)
                    draws = recorder.take()
                    action_list = [float(x) for x in np.asarray(action, dtype=np.float32).tolist()]
                    rows.append({"tick": {"seed": seed, "episode": episode, "tick": tick, "label": label, "action": action_list, "draws": draws,
                                          "obs": _obs_list(obs), "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
                                          "info": _info_flags(info), "state": snapshot(raw_env, wrapper, rec)}})
                    ticks += 1
                    if terminated or truncated:
                        episode += 1
                        raw_env, rec = do_reset()
                        episodes += 1
            finally:
                recorder.restore()
                if env is not None:
                    env.close()
    finally:
        os.chdir(original_cwd)
    if ticks == 0:
        raise RuntimeError("capture produced zero ticks")
    metadata = {
        "schema": SCHEMA, "created_unix": time.time(), "controller": args.controller, "seeds": seeds, "steps_per_seed": args.steps,
        "ticks": ticks, "episodes": episodes, "pymunk": importlib.metadata.version("pymunk"), "chipmunk": getattr(pymunk, "chipmunk_version", None),
        "source": {"root": str(source_root), "environment_sha256": sha256_file(source_root / "UTMIST-AI2-main/environment/environment.py"),
                   "train_sha256": sha256_file(wrapper_source), "config_sha256": sha256_file(config),
                   "positions_sha256": sha256_file(source_root / "positions.json")},
        "capture_sha256": sha256_file(Path(__file__)),
        "wrapper": wrapper_cfg,
        "note": "Closed-loop record: actions, random draws, wrapper outputs and hidden-state snapshots per tick; bodies keyed by Chipmunk shape hashid.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if output.suffix == ".gz" else open
    with opener(output, "wt", encoding="utf-8") as handle:  # type: ignore[call-overload]
        handle.write(json.dumps({"metadata": metadata}) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return {"output": str(output), "ticks": ticks, "episodes": episodes}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--wrapper-source", type=Path)
    parser.add_argument("--controller", choices=("adaptive", "scripted", "random"), default="adaptive")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", default="45")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = capture(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
