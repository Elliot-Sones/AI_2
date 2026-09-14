"""Immutable, policy-free, raw two-player captures of the frozen original game.

Snapshots are comparison evidence, never per-tick simulator inputs. Rendering
is excluded; asset-derived gameplay constants and RNG draws are not excluded.
Class observers call the original methods exactly once and restore in finally.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from enum import Enum
import functools
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

from navigation_contact_capture import SpaceRecorder, validate_layout, velocity_mode_of
from navigation_episode_capture import DrawRecorder, STATE_KEYS, decode_input, world_snapshot

SCHEMA = "ai2.combat_episode_capture.v1"
FROZEN_SHA = "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"
PLAYER_FIELDS = ("agent_id", "weapon", "facing", "stocks", "damage", "damage_taken_this_stock", "damage_taken_total",
                 "damage_done", "damage_taken_this_frame", "damage_velocity", "target_vel", "smoothXVel", "prev_x", "prev_y",
                 "just_got_hit", "pickup_lock_until", "start_position", "cur_action", "hitboxes_to_draw", "points_to_draw")
PLAYER_CONSTANTS = ("delta", "mass", "moment", "move_speed", "jump_speed", "max_fall_speed", "fast_fall_ease", "in_air_ease",
                    "bounce_coef", "run_speed", "dash_speed", "backdash_speed", "turnaround_time", "taunt_time", "backdash_time",
                    "dodge_time", "grounded_dodge_cooldown", "smoothTimeX", "air_dodge_cooldown", "invincible_time", "jump_cooldown",
                    "dash_time", "dash_cooldown")
PLAYER_REQUIRED = {"agent_id", "weapon", "facing", "stocks", "damage", "damage_velocity", "target_vel", "hurtbox", "bank",
                   "state_id", "state", "input", "input_previous", "x", "y", "vx", "vy", "velocity_mode", "on_platform", "constants"}
STATE_REQUIRED = {"players", "platform", "spawners", "weapon_pool", "objects", "raw", "world"}
POWER_REQUIRED = set("power_index power_id cast_idx total_frame_count frames_into_recovery recovery_frames hit_anyone "
                     "dealt_position_target_exists current_dealt_position_target agents_hit_this_power is_switching_casts "
                     "past_point_positions in_recovery casts".split())
BODY_REQUIRED = set("index type velocity_mode m m_inv i i_inv cog p v f a w t transform v_bias w_bias".split())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value: Any) -> Any:
    """Strict serialization: no repr fallbacks or nonfinite JSON numbers."""
    if isinstance(value, Enum):
        return plain(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite simulation evidence")
        return value
    if isinstance(value, dict):
        if any(not isinstance(k, (str, int)) for k in value):
            raise ValueError("unsupported mapping key")
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if type(value).__module__.split(".")[0] == "numpy":
        return plain(value.tolist())
    if type(value).__name__ in ("Vec2d", "Vector2", "Rect"):
        return [plain(v) for v in value]
    raise ValueError(f"unsupported evidence type: {type(value).__name__}")


def capsule_snapshot(collider: Any) -> dict[str, Any]:
    return {k: plain(v) for k, v in vars(collider).items()}


def attack_snapshot(manager: Any, game: Any) -> dict[str, Any]:
    asset = None
    for weapon, bank in (("Punch", game.attacks), ("Spear", game.spear_attacks), ("Hammer", game.hammer_attacks)):
        for move, data in bank.items():
            if data is manager.move_data:
                asset = {"weapon": weapon, "move": int(move)}
    if asset is None:
        raise ValueError("unresolved attack asset identity")
    power = manager.current_power
    power_index = next(i for i, item in enumerate(manager.move_data["powers"]) if item is power.power_data)
    result = {k: plain(v) for k, v in vars(manager).items()
              if k not in {"p", "move_data", "current_power", "all_hit_agents", "hit_agent"}}
    result.update(asset=asset, all_hit_agents=[int(p.agent_id) for p in manager.all_hit_agents],
                  hit_agent=None if manager.hit_agent is None else int(manager.hit_agent.agent_id))
    pstate = {k: plain(v) for k, v in vars(power).items() if k not in {"p", "power_data", "casts", "agents_hit_this_power"}}
    pstate.update(power_index=power_index, agents_hit_this_power=[int(p.agent_id) for p in power.agents_hit_this_power],
                  casts=[{k: plain(v) for k, v in vars(cast).items() if k != "cast_data"} for cast in power.casts])
    result["current_power"] = pstate
    return result


def player_snapshot(player: Any) -> dict[str, Any]:
    result = {k: plain(getattr(player, k)) for k in PLAYER_FIELDS if hasattr(player, k)}
    result.update(x=float(player.body.position.x), y=float(player.body.position.y), vx=float(player.body.velocity.x),
                  vy=float(player.body.velocity.y), state=type(player.state).__name__,
                  state_id=int(player.state_mapping[type(player.state).__name__]),
                  input=decode_input(player), input_previous=plain(player.input.prev_state),
                  velocity_mode=velocity_mode_of(player.body), hurtbox=capsule_snapshot(player.hurtbox_collider),
                  on_platform=None if player.on_platform is None else next(k for k, v in player.env.objects.items()
                                                                          if getattr(v, "body", None) is player.on_platform),
                  constants={k: plain(getattr(player, k)) for k in PLAYER_CONSTANTS})
    if hasattr(player.body, "gravity_scale"):
        result["gravity_scale"] = plain(player.body.gravity_scale)
    result["bank"] = {}
    for name, index in STATE_KEYS:
        state = player.states[name]
        result["bank"][str(index)] = {k: attack_snapshot(v, player.env) if k == "move_manager" else plain(v)
                                      for k, v in vars(state).items() if k != "p"}
    return result


class Snapshotter:
    def __init__(self, game: Any):
        self.game = game
        self.space = SpaceRecorder(game)
        self.layout = validate_layout(self.space)
        self.weapons: list[Any] = []

    def weapon_id(self, weapon: Any) -> int | None:
        if weapon is None:
            return None
        if not any(w is weapon for w in self.weapons):
            self.weapons.append(weapon)
        return next(i for i, w in enumerate(self.weapons) if w is weapon)

    def weapon(self, weapon: Any) -> dict[str, Any]:
        from navigation_contact_capture import SHAPE, _addr, _u64
        return {"id": self.weapon_id(weapon), "name": weapon.name, "active": bool(weapon.active), "world_pos": plain(weapon.world_pos),
                "spawn_frame": getattr(weapon, "spawn_frame", None), "physics_on": bool(weapon.physics_on), "fall_speed": weapon.fall_speed,
                "image_size": list(weapon.image.get_size()),
                "shape_hashid": None if weapon.shape is None else _u64(_addr(weapon.shape._shape), SHAPE["hashid"])}

    def snapshot(self) -> dict[str, Any]:
        game = self.game
        platform = game.objects["platform1"]
        spawners = []
        for spawner in game.weapon_controller.spawners:
            fields = ("id", "world_pos", "og_cooldown_frames", "cooldown_frames", "last_spawn_frame", "despawn_frames", "flag",
                      "weapon_name", "_spawned_once", "done", "flipped", "_weapon_obj_key")
            row = {k: plain(getattr(spawner, k)) for k in fields if hasattr(spawner, k)}
            row.update(type=type(spawner).__name__, active_weapon=None if spawner.active_weapon is None else self.weapon(spawner.active_weapon))
            if hasattr(spawner, "vfx"):
                row["spawn_duration"] = int(spawner.vfx._steps("spawn"))
            spawners.append(row)
        world = world_snapshot(self.space)
        # Static/kinematic Chipmunk masses are mathematically infinite. Encode
        # only these explicit constants; NaN and all other infinities fail.
        for row in world["bodies"].values():
            for key in ("m", "i"):
                if row["body"][key] == math.inf:
                    row["body"][key] = {"constant": "+infinity"}
        return plain({"players": [player_snapshot(p) for p in game.players],
                      "platform": {"x": platform.body.position.x, "y": platform.body.position.y, "vx": platform.body.velocity.x,
                                   "vy": platform.body.velocity.y, "moving_to_w2": platform.moving_to_w2,
                                   "waypoint1": platform.waypoint1, "waypoint2": platform.waypoint2},
                      "spawners": spawners, "weapon_pool": [self.weapon(w) for w in game.weapon_pool.pool],
                      "objects": list(game.objects),
                      "raw": {"steps": game.steps, "terminated": bool(getattr(game, "terminated", False)), "dt": game.dt,
                              "max_timesteps": game.max_timesteps, "next_spawner_id": getattr(game, "_next_spawner_id", None),
                              "space_config": self.space.space_config()}, "world": world})


class EventRecorder:
    """Observe ordered original calls; do not recompute collision or damage."""
    def __init__(self, game: Any, module: Any):
        self.game, self.module = game, module
        self.events: list[dict[str, Any]] = []
        self.originals: list[tuple[Any, str, Any, bool]] = []
        self.phase: int | None = None
        self.tick = 0
        self.attack_ids = [0, 0]
        self.power: Any = None
        self.hitbox_index = 0
        self.spawner: Any = None
        self.gameplay_sequence = 0
        self.power_offsets = {}
        offset = 0
        for weapon, bank in (("Punch", game.attacks), ("Spear", game.spear_attacks), ("Hammer", game.hammer_attacks)):
            for move, data in sorted(bank.items(), key=lambda item: int(item[0])):
                self.power_offsets[(weapon, int(move))] = offset
                offset += len(data["powers"])

    def emit(self, kind: str, **values: Any) -> None:
        self.events.append(plain({"tick": self.tick, "sequence": len(self.events), "phase_player": self.phase,
                                  "kind": kind, **values}))

    def patch(self, owner: Any, name: str, callback: Any) -> None:
        original = getattr(owner, name)
        self.originals.append((owner, name, original, name in vars(owner)))
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return callback(original, *args, **kwargs)
        setattr(owner, name, wrapped)

    def gameplay(self, kind: int, source: int, *, target=-1, hitbox=-1, amount=0.0, next_power=None, weapon=None, spawner_id=None):
        player = self.game.players[source]
        manager = getattr(player.states["attack"], "move_manager", None)
        move_index, power_index, cast_index = -1, -1, 0
        if manager is not None:
            snap = attack_snapshot(manager, self.game)
            move_index = ("Punch", "Spear", "Hammer").index(snap["asset"]["weapon"]) * 11 + snap["asset"]["move"] - 2
            power_index = self.power_offsets[(snap["asset"]["weapon"], snap["asset"]["move"])] + snap["current_power"]["power_index"]
            cast_index = manager.current_power.cast_idx
            if next_power is not None:
                local = next(i for i, data in enumerate(manager.move_data["powers"]) if data is next_power.power_data)
                power_index = self.power_offsets[(snap["asset"]["weapon"], snap["asset"]["move"])] + local
                cast_index = 0
        event = {"kind": kind, "source": source, "target": target, "move": move_index, "power": power_index,
                 "cast": cast_index, "hitbox": hitbox, "attack_id": self.attack_ids[source], "amount": float(amount),
                 "phase_player": -1 if self.phase is None else self.phase, "event_sequence": self.gameplay_sequence}
        if kind >= 100:
            event.update(move=-1, power=-1, cast=-1, hitbox=-1, attack_id=0, weapon=weapon, spawner_id=spawner_id)
        self.gameplay_sequence += 1
        self.emit("gameplay", event=event)

    def install(self) -> None:
        module = self.module
        def physics(original, player, dt):
            self.phase = int(player.agent_id)
            self.emit("player_physics_begin", player_slot=player.agent_id, players=[player_snapshot(p) for p in self.game.players])
            result = original(player, dt)
            self.emit("player_physics_end", player_slot=player.agent_id, players=[player_snapshot(p) for p in self.game.players])
            self.phase = None
            return result
        def enter(original, state):
            result = original(state)
            slot = int(state.p.agent_id)
            self.emit("attack_started", player_slot=slot, attack_id=self.attack_ids[slot], move=int(state.move_type),
                      weapon=state.p.weapon, facing=int(state.p.facing))
            return result
        def give_move(original, state, move):
            result = original(state, move)
            self.attack_ids[state.p.agent_id] += 1
            self.gameplay(1, state.p.agent_id)
            return result
        def end(original, state):
            self.emit("attack_ended", player_slot=state.p.agent_id, attack_id=self.attack_ids[state.p.agent_id], move=int(state.move_type))
            result = original(state)
            self.gameplay(5, state.p.agent_id)
            return result
        def power(original, obj, *args, **kwargs):
            previous = self.power
            self.power = obj
            self.hitbox_index = 0
            before = (obj.power_id, obj.cast_idx, obj.casts[obj.cast_idx].frame_idx)
            try:
                result = original(obj, *args, **kwargs)
                if result[1] is not None:
                    self.gameplay(9, obj.p.agent_id, next_power=result[1])
                self.emit("power_step", player_slot=obj.p.agent_id, attack_id=self.attack_ids[obj.p.agent_id], before=list(before),
                          after=[obj.power_id, obj.cast_idx, obj.casts[obj.cast_idx].frame_idx],
                          next_power_id=None if result[1] is None else result[1].power_id, done=bool(result[0]))
                return result
            finally:
                self.power = previous
        def collision(original, collider, other):
            result = original(collider, other)
            if self.power is not None:
                source = self.power.p
                vulnerable = bool(source.opponent.state.vulnerable())
                if result and vulnerable:
                    self.gameplay(3, source.agent_id, target=1-source.agent_id, hitbox=self.hitbox_index)
                self.hitbox_index += 1
                self.emit("attack_collision", source_slot=source.agent_id, target_slot=1 - source.agent_id,
                          attack_id=self.attack_ids[source.agent_id], power_id=self.power.power_id, cast_index=self.power.cast_idx,
                          cast_frame=self.power.casts[self.power.cast_idx].frame_idx, intersects=bool(result),
                          vulnerable=vulnerable, hurtbox=capsule_snapshot(collider), hitbox=capsule_snapshot(other))
            return result
        def damage(original, player, *args, **kwargs):
            before = player_snapshot(player)
            result = original(player, *args, **kwargs)
            source = 1 - int(player.agent_id)
            self.gameplay(4, source, target=player.agent_id, amount=args[0] if args else kwargs["damage_default"])
            self.emit("damage", source_slot=source, target_slot=player.agent_id, attack_id=self.attack_ids[source],
                      source="Player.apply_damage", arguments=plain(args), keyword_arguments=plain(kwargs), before=before, after=player_snapshot(player))
            return result
        def respawn(original, player):
            before = player_snapshot(player)
            result = original(player)
            self.gameplay(8, player.agent_id, target=player.agent_id)
            self.emit("respawn", player_slot=player.agent_id, before=before, after=player_snapshot(player))
            return result
        def active(original, player, *args, **kwargs):
            hitboxes = args[0] if args else kwargs.get("hitboxes")
            result = original(player, *args, **kwargs)
            if self.power is not None and hitboxes is not None:
                self.gameplay(2, player.agent_id)
            return result
        def stun_enter(original, state):
            result = original(state)
            self.gameplay(6, state.p.agent_id, target=state.p.agent_id, amount=state.stun_frames)
            return result
        def ko_enter(original, state):
            result = original(state)
            self.gameplay(7, state.p.agent_id, target=state.p.agent_id)
            return result
        def pickup(original, spawner, *args, **kwargs):
            previous = self.spawner
            self.spawner = spawner
            try:
                return original(spawner, *args, **kwargs)
            finally:
                self.spawner = previous
        def signal(original, signal_obj, *args, **kwargs):
            result = original(signal_obj, *args, **kwargs)
            for name in ("knockout_signal", "win_signal", "hit_during_stun", "weapon_drop_signal", "weapon_equip_signal"):
                if signal_obj is getattr(self.game, name):
                    self.emit(name, arguments=args, keyword_arguments=kwargs)
                    if name in {"weapon_drop_signal", "weapon_equip_signal"}:
                        slot = 0 if kwargs["agent"] == "player" else 1
                        spawner = self.game.weapon_controller.spawners[-1] if name == "weapon_drop_signal" else self.spawner
                        if spawner is None:
                            raise ValueError("unresolved weapon event spawner")
                        weapon_name = spawner.weapon_name if name == "weapon_drop_signal" else self.game.players[slot].weapon
                        self.gameplay(101 if name == "weapon_drop_signal" else 102, slot,
                                      weapon=("Spear", "Hammer").index(weapon_name), spawner_id=spawner.id)
            return result
        for owner, name, hook in ((module.Player, "physics_process", physics), (module.AttackState, "enter", enter),
                                  (module.AttackState, "give_move", give_move), (module.Player, "set_hitboxes_to_draw", active),
                                  (module.StunState, "enter", stun_enter), (module.KOState, "enter", ko_enter),
                                  (module.WeaponSpawner, "try_pick_up", pickup), (module.DroppedWeaponSpawner, "try_pick_up", pickup),
                                  (module.AttackState, "exit", end), (module.Power, "do_power", power),
                                  (module.CapsuleCollider, "intersects", collision), (module.Player, "apply_damage", damage),
                                  (module.Player, "respawn", respawn), (module.Signal, "emit", signal)):
            self.patch(owner, name, hook)

    def restore(self) -> None:
        for owner, name, original, owned in reversed(self.originals):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.originals.clear()


@contextmanager
def original_game(source_root: Path, seed: int):
    """Load only the verified original game; restore cwd/path/global RNG."""
    import numpy as np
    source_root = Path(source_root).resolve()
    source = source_root / "UTMIST-AI2-main/environment/environment.py"
    if sha256(source) != FROZEN_SHA:
        raise ValueError("frozen environment source hash mismatch")
    cwd, paths, bytecode = Path.cwd(), list(sys.path), sys.dont_write_bytecode
    py_state, np_state = random.getstate(), np.random.get_state()
    game = None
    try:
        sys.dont_write_bytecode = True
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.chdir(source_root)
        sys.path.insert(0, str(source_root / "UTMIST-AI2-main"))
        existing = sys.modules.get("environment.environment")
        if existing is not None and Path(existing.__file__).resolve() != source:
            raise ValueError("a different game module is already loaded; use an isolated process")
        module = importlib.import_module("environment.environment")
        if Path(module.__file__).resolve() != source:
            raise ValueError("resolved game module is not frozen source")
        random.seed(seed)
        np.random.seed(seed)
        with redirect_stdout(io.StringIO()):
            game = module.WarehouseBrawl(train_mode=True)
        yield game, module
    finally:
        if game is not None:
            game.close()
        random.setstate(py_state)
        np.random.set_state(np_state)
        sys.dont_write_bytecode = bytecode
        sys.path[:] = paths
        os.chdir(cwd)


def apply_scenario(game: Any, module: Any, scenario: dict[str, Any]) -> None:
    """Declared one-time initial fixtures, never per-tick corrections."""
    if set(scenario) - {"players", "max_timesteps"}:
        raise ValueError("unsupported scenario field")
    if "max_timesteps" in scenario:
        value = scenario["max_timesteps"]
        if type(value) is not int or value <= 0:
            raise ValueError("invalid scenario timeout")
        game.max_timesteps = value
    players = scenario.get("players", [{}, {}])
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError("scenario requires two player fixtures")
    for player, row in zip(game.players, players):
        if set(row) - {"position", "velocity", "facing", "stocks", "damage", "weapon", "state"}:
            raise ValueError("unsupported player fixture")
        for field in ("position", "velocity"):
            if field in row:
                if not isinstance(row[field], list) or len(row[field]) != 2:
                    raise ValueError("invalid vector fixture")
                setattr(player.body, field, row[field])
        for field in ("stocks", "damage", "weapon"):
            if field in row:
                setattr(player, field, row[field])
        if player.weapon not in {"Punch", "Spear", "Hammer"}:
            raise ValueError("invalid fixture weapon")
        if "facing" in row:
            if row["facing"] not in (-1, 1):
                raise ValueError("invalid facing fixture")
            player.facing = module.Facing(row["facing"])
        if "state" in row:
            if row["state"] not in {"standing", "in_air"}:
                raise ValueError("fixture can initialize only standing or in_air")
            player.state = player.states[row["state"]]
            if row["state"] == "in_air":
                player.state.refresh()
        if "position" in row:
            player.hurtbox_collider.center = player.body.position
            game.space.reindex_shapes_for_body(player.body)


def outputs(observation, rewards, terminated=False, truncated=False, info=None):
    import numpy as np
    arrays = {str(slot): np.asarray(observation[slot]) for slot in (0, 1)}
    if any(a.shape != (64,) or a.dtype != np.float64 for a in arrays.values()):
        raise ValueError("raw observations must be float64[64] for both players")
    return plain({"obs": arrays, "obs_dtype": "float64", "rewards": rewards, "terminated": bool(terminated),
                  "truncated": bool(truncated), "info": {} if info is None else info})


def capture_raw(source_root: Path, *, seed: int, actions: list, scenario=None) -> dict[str, Any]:
    import numpy as np
    import pymunk
    actions = plain(actions)
    scenario = plain({} if scenario is None else scenario)
    for row in actions:
        validate_actions(row)
    if not actions or type(seed) is not int:
        raise ValueError("capture requires an integer seed and nonempty action sequence")
    source_root = Path(source_root).resolve()
    with original_game(source_root, seed) as (game, module):
        draws = DrawRecorder()
        observer = EventRecorder(game, module)
        try:
            random.seed(seed)
            np.random.seed(seed)
            draws.install()
            observation, info = game.reset(seed=seed)
            apply_scenario(game, module, scenario)
            observation = game._observe_all()
            snapshotter = Snapshotter(game)
            reset = {"draws": draws.take(), "outputs": outputs(observation, game.rewards, info=info),
                     "state": snapshotter.snapshot(), "events": []}
            observer.install()
            ticks = []
            for tick, action in enumerate(actions, 1):
                observer.events = []
                observer.tick = tick
                observer.gameplay_sequence = 0
                transition = game.step({i: np.asarray(action[i], dtype=np.float64) for i in (0, 1)})
                ticks.append({"tick": tick, "actions": action, "draws": draws.take(), "outputs": outputs(*transition),
                              "state": snapshotter.snapshot(), "events": observer.events})
                if transition[2] or transition[3]:
                    break
            env_dir = source_root / "UTMIST-AI2-main/environment"
            attacks = {str(p.relative_to(env_dir)): sha256(p) for name in ("unarmed_attacks", "spear_attacks", "hammer_attacks")
                       for p in sorted((env_dir / name).glob("*.json"))}
            sources = {name: sha256(Path(__file__).with_name(name)) for name in
                       ("combat_episode_capture.py", "navigation_episode_capture.py", "navigation_contact_capture.py")}
            document = {"schema": SCHEMA, "metadata": {"environment_sha256": FROZEN_SHA, "attack_sha256": attacks,
                        "capture_sources": sources, "seed": seed, "scenario": scenario, "requested_ticks": len(actions), "ticks": len(ticks),
                        "pymunk": importlib.metadata.version("pymunk"), "pygame": importlib.metadata.version("pygame"),
                        "chipmunk": str(pymunk.chipmunk_version), "layout": snapshotter.layout,
                        "contract": "raw_two_player_ordered_physics_v1", "rendering": False}, "reset": reset, "ticks": ticks}
            validate_capture(document)
            return document
        finally:
            observer.restore()
            draws.restore()


def validate_actions(actions: Any) -> None:
    if not isinstance(actions, list) or len(actions) != 2 or any(not isinstance(row, list) or len(row) != 10 for row in actions):
        raise ValueError("actions must be two unmasked 10-element arrays")
    if any(type(v) not in (int, float) or not math.isfinite(v) for row in actions for v in row):
        raise ValueError("invalid action value")


def validate_capture(document: dict[str, Any]) -> None:
    try:
        _validate_capture(document)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("malformed combat evidence schema") from exc


def validate_state(state: dict[str, Any], tick: int) -> None:
    if set(state) != STATE_REQUIRED or len(state["players"]) != 2 or state["raw"]["steps"] != tick:
        raise ValueError("incomplete world state")
    if not {"steps", "terminated", "dt", "max_timesteps", "next_spawner_id", "space_config"} <= set(state["raw"]):
        raise ValueError("incomplete raw counters")
    world = state["world"]
    if set(world) != {"bodies", "tree", "cache", "solve", "stamp", "shape_id_counter"} or len(world["bodies"]) < 5:
        raise ValueError("incomplete physics world")
    if set(world["tree"]) != {"stamp", "leaves", "order"}:
        raise ValueError("incomplete broadphase state")
    for body in world["bodies"].values():
        if set(body) != {"label", "body", "shape"} or set(body["body"]) != BODY_REQUIRED:
            raise ValueError("incomplete physics body")
        if set(body["shape"]) != {"verts", "radius", "elasticity", "friction", "collision_type", "filter", "sensor", "surface_velocity"}:
            raise ValueError("incomplete physics shape")
    for slot, player in enumerate(state["players"]):
        if not PLAYER_REQUIRED <= set(player) or player["agent_id"] != slot or set(player["bank"]) != {str(i) for i in range(13)}:
            raise ValueError("incomplete player state")
        if set(player["constants"]) != set(PLAYER_CONSTANTS) or len(player["input"]) != 35:
            raise ValueError("incomplete player constants/input")
        if not {"center", "width", "height", "radius", "is_circle"} <= set(player["hurtbox"]):
            raise ValueError("incomplete hurtbox")
        for bank in player["bank"].values():
            if not {"invincible_timer", "dodge_cooldown", "stun_time_stored"} <= set(bank):
                raise ValueError("incomplete state bank")
        attack = player["bank"]["8"]
        if "move_type" in attack or player["state_id"] == 8:
            manager = attack["move_manager"]
            if not {"frame", "cooldown", "move_facing_direction", "asset", "all_hit_agents", "hit_agent", "keys", "current_power"} <= set(manager):
                raise ValueError("incomplete attack manager")
            power = manager["current_power"]
            if not POWER_REQUIRED <= set(power) or not power["casts"]:
                raise ValueError("incomplete attack power")
            for cast in power["casts"]:
                if set(cast) != {"frame_idx", "startup_frames", "attack_frames", "base_damage", "variable_force", "fixed_force",
                                 "hit_angle_deg", "must_be_held", "collision_check_points", "hitboxes"}:
                    raise ValueError("incomplete attack cast")
    for spawner in state["spawners"]:
        if not {"id", "type", "world_pos", "cooldown_frames", "last_spawn_frame", "despawn_frames", "active_weapon"} <= set(spawner):
            raise ValueError("incomplete weapon spawner")


def _validate_capture(document: dict[str, Any]) -> None:
    plain(document)
    if set(document) != {"schema", "metadata", "reset", "ticks"} or document["schema"] != SCHEMA:
        raise ValueError("unsupported combat capture schema")
    metadata = document["metadata"]
    if metadata.get("environment_sha256") != FROZEN_SHA or len(metadata.get("attack_sha256", {})) != 33:
        raise ValueError("missing or incorrect frozen provenance")
    if not document["ticks"] or metadata.get("ticks") != len(document["ticks"]):
        raise ValueError("invalid capture tick inventory")
    terminal = False
    for index, row in enumerate([document["reset"], *document["ticks"]]):
        required = {"draws", "outputs", "state", "events"} | ({"tick", "actions"} if index else set())
        if set(row) != required or (index and (row["tick"] != index or terminal)):
            raise ValueError("invalid or discontinuous capture row")
        if index:
            validate_actions(row["actions"])
        out, state = row["outputs"], row["state"]
        if set(out) != {"obs", "obs_dtype", "rewards", "terminated", "truncated", "info"}:
            raise ValueError("incomplete raw outputs")
        if out["obs_dtype"] != "float64" or set(out["obs"]) != {"0", "1"} or any(len(v) != 64 for v in out["obs"].values()):
            raise ValueError("invalid raw observation schema")
        if set(out["rewards"]) != {"0", "1"} or type(out["terminated"]) is not bool or type(out["truncated"]) is not bool:
            raise ValueError("invalid reward or done schema")
        validate_state(state, index)
        for sequence, event in enumerate(row["events"]):
            if event.get("sequence") != sequence or event.get("tick") != index or not isinstance(event.get("kind"), str):
                raise ValueError("invalid sequential event evidence")
        for draw in row["draws"]:
            if draw.get("kind") not in (1, 2, 3, 4) or not {"a", "b", "value"} <= set(draw):
                raise ValueError("unsupported RNG request")
        terminal = out["terminated"] or out["truncated"]


def write_capture(document: dict[str, Any], output: Path) -> dict[str, Any]:
    validate_capture(document)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if output.suffix == ".gz":
        payload = gzip.compress(payload, mtime=0)
    descriptor, name = tempfile.mkstemp(prefix=".combat-capture-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(name, output)  # Atomic and refuses an existing target, unlike replace.
    finally:
        os.unlink(name)
    return {"path": str(output), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "ticks": len(document["ticks"])}


def load_capture(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    path = Path(path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("capture checksum mismatch")
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    document = json.loads(payload)
    validate_capture(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--actions-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    document = capture_raw(args.source_root, seed=args.seed, actions=json.loads(args.actions_json.read_text()))
    print(json.dumps(write_capture(document, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
