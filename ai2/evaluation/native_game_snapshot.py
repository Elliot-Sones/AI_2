"""Explicit gameplay_projection_v1 adapter; no reference tick-state injection.

The only Python-side history is input actions and observer-assigned identities.
Simulation state and semantic events come from the native engine. Immutable
attack constructors are reconstructed from hash-verified frozen JSON assets.
This does not claim parity of intermediate diagnostic phase snapshots.
"""
from __future__ import annotations

from ai2.paths import source_path

import copy
import ctypes as C
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import ai2.simulation.native_navigation_episode as nav
from ai2.simulation.native_attack_tables import DEFAULT_SOURCE, FLAGS, MOVE_PATHS, POWER_TIMES, build_tables
from ai2.simulation.native_game_episode import D, GameReference, Result

WEAPONS = ("Punch", "Spear", "Hammer")
STATE_NAMES = ("WalkingState", "StandingState", "TurnaroundState", "AirTurnaroundState", "SprintingState", "StunState",
               "InAirState", "DodgeState", "AttackState", "DashState", "BackDashState", "KOState", "TauntState")
KEYS = ("W", "A", "S", "D", "space", "h", "l", "j", "k", "g")
BASE_OBJECTS = ("ground1", "ground2", "platform1", "player", "opponent")
BODY_LABELS = ("ground1", "ground2", "platform1", "player0", "player1")
SPAWN_ASSETS = {
    "regular": ("spawnervfx/spawn.gif", "b76d02d162556865d56366d37a9648591a461deab40f989e20671c3263f77de2"),
    "Spear": ("spearvfx/spawn.gif", "c829f4d0aeb1053203410d2c820dc454164e6b9a86558ce5c427ad4d661b3378"),
    "Hammer": ("hammervfx/spawn.gif", "788c088e31d712289448006322eb6089c61386bc49a5e388849fccac551bf028"),
}
POWER_BOOLS = dict(zip(FLAGS, ("is_charge", "disable_caster_gravity", "disable_hit_gravity", "target_all_hit_agents",
                              "transition_on_instant_hit", "on_hit_velocity_set_active", "allow_left_right_mobility",
                              "hit_all_hit_agents", "enable_floor_drag", "damage_over_life_of_hitbox")))


class AttackMetadata:
    """Verified constructor data only; never a Python game or policy instance."""
    def __init__(self, source=DEFAULT_SOURCE):
        from PIL import Image, ImageSequence
        source = Path(source)
        self.tables = build_tables(source)
        self.assets = [json.loads((source / path).read_text()) for path in MOVE_PATHS]
        self.power_offsets = []
        offset = 0
        for asset in self.assets:
            self.power_offsets.append(offset)
            offset += len(asset["powers"])
        self.spawn_durations = {}
        for name, (relative, digest) in SPAWN_ASSETS.items():
            path = source_path(source, relative)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("spawn animation provenance mismatch")
            with Image.open(path) as image:
                self.spawn_durations[name] = sum(max(1, round(frame.info.get("duration", 100) / 1000 * 30))
                                                 for frame in ImageSequence.Iterator(image))

    def snapshot(self, attack):
        move = attack.move_idx
        if not 0 <= move < len(self.assets):
            raise RuntimeError("invalid native attack move")
        local_power = attack.power_idx - self.power_offsets[move]
        powers = self.assets[move]["powers"]
        if not 0 <= local_power < len(powers):
            raise RuntimeError("invalid native attack power")
        data = powers[local_power]
        power = {name: data.get(key, 0) // 2 for name, key in POWER_TIMES.items()}
        power.update({name: data.get(key, False) for key, name in POWER_BOOLS.items()})
        power.update(power_id=data.get("powerID", -1), hit_angle_deg=data.get("hitAngleDeg", 0.0),
                     on_hit_velocity_set_magnitude=data.get("onHitVelocitySetMagnitude", 0.0),
                     on_hit_velocity_set_direction_deg=data.get("onHitVelocitySetDirectionDeg", 0.0))
        for prefix in ("hit", "miss", "ground"):
            power[f"on_{prefix}_next_power_index"] = data.get(f"on{prefix.title()}NextPowerIndex", -1)
        power["last_power"] = power["on_hit_next_power_index"] == power["on_miss_next_power_index"] == -1
        casts = []
        for index, cast in enumerate(data["casts"]):
            casts.append({"frame_idx": int(attack.cast_frame[index]), "startup_frames": cast.get("startupFrames", 0) // 2,
                          "attack_frames": cast.get("attackFrames", 0) // 2, "base_damage": cast.get("baseDamage", 0),
                          "variable_force": cast.get("variableForce", 0.0), "fixed_force": cast.get("fixedForce", 0.0),
                          "hit_angle_deg": cast.get("hitAngleDeg", 0.0), "must_be_held": cast.get("mustBeHeld", False),
                          "collision_check_points": copy.deepcopy(cast.get("collisionCheckPoints", [])),
                          "hitboxes": copy.deepcopy(cast.get("hitboxes", []))})
        if attack.dealt_target_components not in (2, 3) or attack.past_point_count not in (0, 1):
            raise RuntimeError("unsupported native attack vector capacity")
        target = [attack.dealt_target_x, attack.dealt_target_y]
        if attack.dealt_target_components == 3:
            target.append(attack.dealt_target_mult)
        power.update(cast_idx=int(attack.cast_idx), total_frame_count=int(attack.total_frame_count),
                     frames_into_recovery=int(attack.frames_into_recovery),
                     recovery_frames=power["fixed_recovery"] + math.floor(power["recovery"] / 1.426),
                     hit_anyone=bool(attack.hit_anyone), dealt_position_target_exists=bool(attack.dealt_target_exists),
                     current_dealt_position_target=target, is_switching_casts=bool(attack.is_switching_casts),
                     past_point_positions=[] if not attack.past_point_count else [[attack.past_point_x, attack.past_point_y]],
                     in_recovery=bool(attack.in_recovery), power_index=local_power,
                     agents_hit_this_power=[i for i in range(2) if attack.power_hit_mask & (1 << i)], casts=casts)
        return {"cooldown": int(attack.cooldown), "frame": int(attack.move_frame), "move_facing_direction": int(attack.move_facing),
                "keys": {"LIGHT": "j", "HEAVY": "k", "THROW": "l"},
                "asset": {"weapon": WEAPONS[move // 11], "move": move % 11 + 2},
                "all_hit_agents": [i for i in range(2) if attack.all_hit_mask & (1 << i)],
                "hit_agent": None if attack.first_hit_agent == -1 else int(attack.first_hit_agent), "current_power": power}


class GameParityAdapter(GameReference):
    """Replay-only diagnostic adapter. reset fixtures are the sole state writes."""
    def __init__(self, library=None, *, source=DEFAULT_SOURCE):
        self.metadata = AttackMetadata(source)
        super().__init__(library)
        self._ready = False
        self._actions = [[0.] * 10 for _ in range(2)]
        self._weapon_ids = {}
        self._objects = {}

    def _replay(self, draws):
        if not isinstance(draws, list):
            raise ValueError("RNG draws must be a list")
        for draw in draws:
            if not isinstance(draw, dict) or type(draw.get("kind")) is not int or draw["kind"] not in (1, 2, 3, 4):
                raise ValueError("invalid RNG draw kind")
            if any(type(draw.get(k)) not in (int, float) or not math.isfinite(draw[k]) for k in ("a", "b", "value")):
                raise ValueError("invalid RNG draw values")
        return self.nav.replay_rng(draws)

    @staticmethod
    def _consumed(rng):
        if rng.error or rng.cursor != rng.count:
            raise RuntimeError(f"RNG draw mismatch: consumed {rng.cursor}/{rng.count}, error {rng.error}")
        return int(rng.cursor)

    def reset(self, reset_draws, scenario):
        if not isinstance(scenario, dict) or set(scenario) - {"players", "max_timesteps"}:
            raise ValueError("unsupported reset scenario")
        fixtures = scenario.get("players", [{}, {}])
        if not isinstance(fixtures, list) or len(fixtures) != 2:
            raise ValueError("scenario needs two player fixtures")
        for fixture in fixtures:
            if not isinstance(fixture, dict) or set(fixture) - {"position", "velocity", "facing", "stocks", "damage", "weapon", "state"}:
                raise ValueError("unsupported player fixture")
        limit = scenario.get("max_timesteps", 9000)
        if type(limit) is not int or limit <= 0:
            raise ValueError("invalid max_timesteps")
        if limit != 9000 and not hasattr(self.state, "max_timesteps"):
            raise ValueError("native configurable max_timesteps is unavailable")
        self._ready = False
        rng, storage = self._replay(reset_draws)
        result = Result()
        self._check(self.lib.ai2g_reset_export(C.byref(self.state), C.byref(self.cfg), C.byref(rng), 48., C.byref(result)))
        self._consumed(rng)
        if hasattr(self.state, "max_timesteps"):
            self.state.max_timesteps = limit
        for i, fixture in enumerate(fixtures):
            player = self.scene.players[i]
            body = self.scene.world.bodies[player.body]
            position, velocity = fixture.get("position", [body.p.x, body.p.y]), fixture.get("velocity", [body.v.x, body.v.y])
            if len(position) != 2 or len(velocity) != 2:
                raise ValueError("fixture vectors must have two coordinates")
            if "position" in fixture:
                self.fixture_player_position(i, *position)
            if "velocity" in fixture:
                body.v.x, body.v.y = velocity
            for key in ("facing", "stocks"):
                if key in fixture:
                    if type(fixture[key]) is not int or (key == "facing" and fixture[key] not in (-1, 1)):
                        raise ValueError(f"invalid {key} fixture")
                    setattr(player.ms, key, fixture[key])
            if "damage" in fixture:
                player.damage = float(fixture["damage"])
            if "weapon" in fixture:
                if fixture["weapon"] not in WEAPONS:
                    raise ValueError("invalid weapon fixture")
                player.weapon = WEAPONS.index(fixture["weapon"])
            if "state" in fixture:
                if fixture["state"] not in ("standing", "in_air"):
                    raise ValueError("unsupported state fixture")
                player.ms.state_id = 1 if fixture["state"] == "standing" else 6
                if fixture["state"] == "in_air":
                    for field, value in ((4, 0), (5, 2), (6, 1)):
                        player.ms.bank[6].present[field] = 1
                        player.ms.bank[6].value[field] = value
        if any(fixtures):
            # Reset fixtures change state after reset_export observed it. This
            # is the only refresh; ordinary step outputs must remain untouched.
            self._check(self.lib.ai2g_observe_export(C.byref(self.state), C.cast(result.obs, C.POINTER(D))))
        self._actions = [[0.] * 10 for _ in range(2)]
        self._weapon_ids, self._objects = {}, {}
        self._ready = True
        return self._record(result, self._consumed(rng), reset=True)

    def step(self, actions, draws):
        if not self._ready:
            raise RuntimeError("reset is required before stepping")
        values = np.asarray(actions, dtype=np.float64)
        if values.shape != (2, 10) or not np.isfinite(values).all():
            raise ValueError("actions must have shape (2,10) and finite values")
        values = np.ascontiguousarray(values)
        rng, storage = self._replay(draws)
        result = Result()
        self._check(self.lib.ai2g_step_export(C.byref(self.state), values.ctypes.data_as(C.POINTER(D)), C.byref(rng),
                                            C.byref(self.table), C.byref(result)))
        self._actions = values.tolist()
        return self._record(result, self._consumed(rng))

    def player_snapshot(self, i):
        player, combat = self.scene.players[i], self.combat[i]
        ms, body = player.ms, self.scene.world.bodies[player.body]
        bank = {}
        for sid in range(13):
            slot = ms.bank[sid]
            bank[str(sid)] = {nav.SLOT_NAMES[k]: bool(slot.value[k]) if nav.SLOT_NAMES[k] in ("is_base", "is_grounded") else int(slot.value[k])
                              for k in range(1, 14) if slot.present[k]}
        for mask, sid, key, value in ((1, 5, "stun_frames", combat.stun_frames), (2, 12, "taunt_timer", combat.taunt_timer),
                                       (4, 12, "seed", combat.taunt_seed), (8, 8, "seed", combat.attack_seed)):
            if combat.attribute_presence & mask:
                bank[str(sid)][key] = int(value)
        if combat.attack.valid:
            bank["8"].update(move_type=int(combat.attack.move_idx % 11 + 2), move_manager=self.metadata.snapshot(combat.attack))
        hurtbox = {"center": [combat.hurtbox_x, combat.hurtbox_y], "width": combat.hurtbox_width,
                   "height": combat.hurtbox_height, "radius": min(combat.hurtbox_width, combat.hurtbox_height) / 2,
                   "is_circle": combat.hurtbox_width == combat.hurtbox_height}
        if combat.hurtbox_attributes_present:
            hurtbox.update(offset=[combat.hurtbox_offset_x, combat.hurtbox_offset_y], size=[combat.hurtbox_size_x, combat.hurtbox_size_y])
        row = {"agent_id": i, "weapon": WEAPONS[player.weapon], "facing": int(ms.facing), "stocks": int(ms.stocks),
               "damage": float(player.damage), "damage_taken_this_stock": combat.damage_taken_this_stock,
               "damage_taken_total": combat.damage_taken_total, "damage_done": combat.damage_done,
               "damage_velocity": [ms.damage_vx, ms.damage_vy], "target_vel": [ms.target_vx, ms.target_vy],
               "smoothXVel": 0., "prev_x": ms.prev_x, "prev_y": ms.prev_y, "just_got_hit": bool(combat.just_got_hit),
               "start_position": [ms.start_x, ms.start_y], "cur_action": copy.deepcopy(self._actions[i]),
               "x": body.p.x, "y": body.p.y, "vx": body.v.x, "vy": body.v.y, "state": STATE_NAMES[ms.state_id],
               "state_id": int(ms.state_id), "input": list(player.input_state),
               "input_previous": {key: bool(player.input_state[k]) for k, key in enumerate(KEYS)},
               # Frozen pre_solve_oneway's setter is inside a disabled string;
               # the only live on_platform writes are constructor/separate None.
               "velocity_mode": int(body.velocity_mode), "hurtbox": hurtbox, "on_platform": None, "bank": bank}
        if self.scene.steps:
            row["damage_taken_this_frame"] = combat.damage_taken_this_frame
        if self.state.weapon_aux[i].pickup_lock_until >= 0:
            row["pickup_lock_until"] = int(self.state.weapon_aux[i].pickup_lock_until)
        if combat.attribute_presence & 16:
            row["gravity_scale"] = int(combat.gravity_scale)
        return row

    def _weapon(self, slot):
        weapon = self.state.weapons.weapons[slot]
        if slot not in self._weapon_ids:
            self._weapon_ids[slot] = len(self._weapon_ids)
        return {"id": self._weapon_ids[slot], "name": WEAPONS[weapon.weapon], "active": bool(weapon.active),
                "world_pos": list(weapon.world_pos), "spawn_frame": int(weapon.spawn_frame), "physics_on": bool(weapon.physics_on),
                "fall_speed": .1, "image_size": [40, 16],
                "shape_hashid": None if weapon.shape < 0 else int(self.scene.world.shapes[weapon.shape].hashid)}

    def snapshot(self):
        base = nav.native_snapshot(self.scene)
        world = base["world"]
        for index in range(self.scene.world.shape_count):
            if not self.scene.world.shape_active[index]:
                continue
            shape = self.scene.world.shapes[index]
            body = self.scene.world.bodies[shape.body]
            key = str(shape.hashid)
            world["bodies"][key]["label"] = BODY_LABELS[shape.hashid] if shape.hashid < 5 else f"WeaponGO:{shape.hashid}"
            world["bodies"][key]["body"].update(index=int(shape.hashid), type=int(body.type), velocity_mode=int(body.velocity_mode),
                                                   m=body.m, m_inv=body.m_inv, i=body.i, i_inv=body.i_inv, cog=[body.cog.x, body.cog.y])
            for name in ("m", "i"):
                if world["bodies"][key]["body"][name] == math.inf:
                    world["bodies"][key]["body"][name] = {"constant": "+infinity"}
        weapons = self.state.weapons
        spawners, current_objects = [], {}
        for index in range(weapons.spawner_count):
            spawner = weapons.spawners[index]
            active = None if spawner.weapon_slot < 0 else self._weapon(spawner.weapon_slot)
            row = {"id": int(spawner.id), "world_pos": list(spawner.world_pos), "og_cooldown_frames": int(spawner.og_cooldown_frames),
                   "cooldown_frames": int(spawner.cooldown_frames), "last_spawn_frame": int(spawner.last_spawn_frame),
                   "despawn_frames": int(spawner.despawn_frames), "type": "DroppedWeaponSpawner" if spawner.kind else "WeaponSpawner",
                   "active_weapon": active}
            if not spawner.kind:
                row.update(flag=bool(spawner.flag), spawn_duration=self.metadata.spawn_durations["regular"])
            else:
                row.update(weapon_name=WEAPONS[spawner.weapon], _spawned_once=bool(spawner.spawned_once),
                           done=bool(spawner.done), flipped=bool(spawner.flipped))
                if spawner.spawned_once:
                    row["_weapon_obj_key"] = f"Dropped{WEAPONS[spawner.weapon]}{spawner.id}"
                    row["spawn_duration"] = self.metadata.spawn_durations[WEAPONS[spawner.weapon]]
            if active is not None:
                key = f"{'Dropped' if spawner.kind else ''}{active['name']}{spawner.id}"
                current_objects[key] = (spawner.weapon_slot, active["shape_hashid"])
            spawners.append(row)
        for key in list(self._objects):
            if current_objects.get(key) != self._objects[key]:
                del self._objects[key]
        self._objects.update(current_objects)
        w = self.scene.world
        config = {"iterations": int(w.iterations), "gravity": [w.gravity.x, w.gravity.y], "damping": w.damping,
                  "collision_slop": w.slop, "collision_bias": w.bias, "collision_persistence": int(w.persistence),
                  "shape_id_counter": int(self.scene.shape_id_counter), "dt": self.scene.cfg.dt}
        return {"players": [self.player_snapshot(i) for i in range(2)], "platform": base["platform"], "spawners": spawners,
                "weapon_pool": [self._weapon(weapons.pool[i]) for i in range(weapons.pool_count)],
                "objects": [*BASE_OBJECTS, *self._objects],
                "raw": {"steps": int(self.scene.steps), "terminated": bool(self.scene.raw_terminated), "dt": self.scene.cfg.dt,
                        "max_timesteps": int(self.state.max_timesteps), "next_spawner_id": None if not weapons.next_spawner_id else int(weapons.next_spawner_id),
                        "space_config": config}, "world": world}

    def events(self):
        result = []
        for combat in self.combat:
            if combat.event_overflow:
                raise RuntimeError("native semantic event overflow")
            for event in combat.events[:combat.event_count]:
                result.append({name: float(getattr(event, name)) if name == "amount" else int(getattr(event, name))
                               for name, _ in event._fields_})
        combat_count = int(self.combat[0].total_tick_events)
        if combat_count != len(result):
            raise RuntimeError("native semantic event count mismatch")
        weapons = self.state.weapons
        if not 0 <= weapons.event_count <= len(weapons.events):
            raise RuntimeError("native weapon event overflow")
        # The frozen tick executes all weapon phases after both player physics
        # phases; the append-only weapon array preserves their actual order.
        for index, event in enumerate(weapons.events[:weapons.event_count]):
            if event.kind not in (1, 2) or event.weapon not in (1, 2):
                raise RuntimeError("unknown native weapon event")
            result.append({"kind": 100 + int(event.kind), "source": int(event.agent), "target": -1,
                           "move": -1, "power": -1, "cast": -1, "hitbox": -1, "attack_id": 0, "amount": 0.,
                           "phase_player": -1, "event_sequence": combat_count + index,
                           "weapon": int(event.weapon) - 1, "spawner_id": int(event.spawner_id)})
        result.sort(key=lambda event: event["event_sequence"])
        if [event["event_sequence"] for event in result] != list(range(len(result))):
            raise RuntimeError("native semantic event sequence gap or duplicate")
        return result

    def _record(self, result, consumed, *, reset=False):
        outputs = {"obs": {str(i): list(result.obs[i]) for i in range(2)}, "obs_dtype": "float64",
                   "rewards": {str(i): 0. if reset else float(result.reward[i]) for i in range(2)},
                   "terminated": bool(result.terminated), "truncated": bool(result.truncated), "info": {}}
        return {"outputs": outputs, "state": self.snapshot(), "events": [] if reset else self.events(), "draws_consumed": consumed}
