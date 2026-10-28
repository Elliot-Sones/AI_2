"""Native movement reducer parity and fail-closed tests.

The reducer covers the frozen navigation-reachable player movement state
machine only.  Pymunk integration/collision solving remains outside this
contract; grounded/platform facts are explicit inputs captured from the oracle.
"""

from __future__ import annotations

import ctypes
import ast
import hashlib
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ENV_SHA256 = "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"
WIDTH = 35

STATE_WALKING = 0
STATE_STANDING = 1
STATE_TURNAROUND = 2
STATE_AIR_TURNAROUND = 3
STATE_SPRINTING = 4
STATE_IN_AIR = 6
STATE_DODGE = 7
STATE_DASH = 9
STATE_BACKDASH = 10
STATE_KO = 11

KEY_W, KEY_A, KEY_S, KEY_D, KEY_SPACE, KEY_H, KEY_L, KEY_J, KEY_K, KEY_G = range(10)
HELD, PRESSED, RELEASED = 0, 10, 20
RAW_VERTICAL, RAW_HORIZONTAL, NO_HORIZONTAL, HORIZONTAL_STATE, LAST_DIRECTION = 30, 31, 32, 33, 34
HSTATE_NONE, HSTATE_LEFT, HSTATE_RIGHT, HSTATE_LEFT_RIGHT, HSTATE_RIGHT_LEFT = range(5)
SUPPORTED_STATE_IDS = (
    STATE_WALKING, STATE_STANDING, STATE_TURNAROUND, STATE_AIR_TURNAROUND,
    STATE_SPRINTING, STATE_IN_AIR, STATE_DODGE, STATE_DASH, STATE_BACKDASH, STATE_KO,
)
_FROZEN_MOVEMENT_NS = None


def resolve_frozen_env():
    candidates = []
    explicit = os.environ.get("AI2_MOVEMENT_ORACLE")
    if explicit:
        candidates.append(Path(explicit))
    replay_source = os.environ.get("AI2_REPLAY_SOURCE")
    if replay_source:
        candidates.append(Path(replay_source) / "UTMIST-AI2-main/environment/environment.py")
    legacy_stage = os.environ.get("AI2_FROZEN_STAGE_SOURCE")
    if legacy_stage:
        candidates.append(Path(legacy_stage))
    candidates.append(
        ROOT.parent / "AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/"
        "payload/variants/corrected/UTMIST-AI2-main/environment/environment.py"
    )
    checked = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.is_file():
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != FROZEN_ENV_SHA256:
                raise AssertionError(f"frozen movement oracle hash mismatch for {candidate}: {digest}")
            return candidate
    raise AssertionError(f"frozen movement oracle not found; checked {checked}")


class Vec2:
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, index):
        return (self.x, self.y)[index]

    def __eq__(self, other):
        try:
            return self.x == other.x and self.y == other.y
        except AttributeError:
            return False


class Body:
    DYNAMIC = 0

    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0):
        self.position = Vec2(x, y)
        self._velocity = Vec2(vx, vy)
        self.velocity_func = Body.update_velocity
        self.body_type = Body.DYNAMIC

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = Vec2(value[0], value[1])

    @staticmethod
    def update_velocity(*_args):
        return None


class BB:
    def __init__(self, overlaps=True):
        self.overlaps = overlaps

    def intersects(self, _other):
        return self.overlaps


class Shape:
    def __init__(self, overlaps=True):
        self.overlaps = overlaps

    def cache_bb(self):
        return BB(self.overlaps)


class Signal:
    def __init__(self):
        self.calls = []

    def emit(self, **kwargs):
        self.calls.append(kwargs)


class DummyState:
    def __init__(self, player):
        self.p = player

    def set_stun(self, value):
        self.stun_frames = value

    def refresh(self):
        pass

    def give_move(self, value):
        self.move_type = value


class MoveType(Enum):
    NONE = 0
    RECOVERY = 1


def frozen_movement_namespace():
    global _FROZEN_MOVEMENT_NS
    if _FROZEN_MOVEMENT_NS is not None:
        return _FROZEN_MOVEMENT_NS
    frozen_env = resolve_frozen_env()
    source = frozen_env.read_text()
    tree = ast.parse(source, filename=str(frozen_env))
    names = {
        "Facing", "KeyStatus", "HorizontalState", "PlayerInputHandler",
        "PlayerObjectState", "GroundState", "InAirState", "WalkingState",
        "SprintingState", "StandingState", "TurnaroundState", "AirTurnaroundState",
        "KOState", "DashState", "BackDashState", "DodgeState",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    player_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Player")
    functions = [node for node in player_node.body if isinstance(node, ast.FunctionDef) and node.name in ("physics_process", "move_toward")]
    module = ast.Module(body=nodes + functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "ABC": __import__("abc").ABC,
        "dataclass": dataclass,
        "Enum": Enum,
        "np": np,
        "pymunk": types.SimpleNamespace(Vec2d=Vec2, Body=Body),
        "MoveType": MoveType,
        "random": __import__("random"),
        "__name__": __name__,
    }
    exec(compile(module, filename=str(frozen_env), mode="exec"), namespace)
    _FROZEN_MOVEMENT_NS = namespace
    return namespace


def oracle_input(ns, decoded_values):
    handler = ns["PlayerInputHandler"]()
    for i, key in enumerate(handler.key_names):
        handler.key_status[key].held = bool(decoded_values[HELD + i])
        handler.key_status[key].just_pressed = bool(decoded_values[PRESSED + i])
        handler.key_status[key].just_released = bool(decoded_values[RELEASED + i])
    handler.raw_vertical = float(decoded_values[RAW_VERTICAL])
    handler.raw_horizontal = float(decoded_values[RAW_HORIZONTAL])
    handler.no_horizontal = bool(decoded_values[NO_HORIZONTAL])
    hstate_values = list(ns["HorizontalState"])
    handler.horizontal_state = hstate_values[decoded_values[HORIZONTAL_STATE]]
    handler.last_direction = int(decoded_values[LAST_DIRECTION])
    return handler


def native_to_oracle_state(movement, native_state, decoded_values, *, grounded, platform_overlap, platform_velocity):
    ns = frozen_movement_namespace()
    player = types.SimpleNamespace()
    player.agent_id = 0
    player.env = types.SimpleNamespace(
        stage_width_tiles=29.8,
        stage_height_tiles=16.8,
        fps=30,
        logger={0: {}, 1: {}},
        knockout_signal=Signal(),
        hit_during_stun=Signal(),
        objects={
            "platform1": types.SimpleNamespace(shape=Shape(platform_overlap), body=Body(vx=platform_velocity[0], vy=platform_velocity[1])),
        },
    )
    player.body = Body(native_state.x, native_state.y, native_state.vx, native_state.vy)
    player.body.velocity_func = Body.update_velocity if native_state.velocity_func == 0 else ns["DodgeState"].no_gravity_velocity_func
    player.body.body_type = native_state.body_type
    player.shape = Shape(platform_overlap)
    player.start_position = (native_state.start_x, native_state.start_y)
    player.facing = ns["Facing"].RIGHT if native_state.facing == 1 else ns["Facing"].LEFT
    player.input = oracle_input(ns, decoded_values)
    player.stocks = native_state.stocks
    player.move_speed = 6.75
    player.jump_speed = 8.9
    player.max_fall_speed = 12
    player.fast_fall_ease = 14.25 / 30
    player.in_air_ease = 10.75 / 30
    player.bounce_coef = 0.7
    player.run_speed = 8
    player.dash_speed = 10
    player.backdash_speed = 4
    player.turnaround_time = 1
    player.taunt_time = 30
    player.backdash_time = 7
    player.dodge_time = 10
    player.grounded_dodge_cooldown = 30
    player.smoothTimeX = 0.33 * 30
    player.air_dodge_cooldown = 82
    player.invincible_time = 90
    player.jump_cooldown = 6.0
    player.dash_time = 9.0
    player.dash_cooldown = 8
    player.damage_velocity = (native_state.damage_vx, native_state.damage_vy)
    player.target_vel = (native_state.target_vx, native_state.target_vy)
    player.prev_x = native_state.prev_x
    player.prev_y = native_state.prev_y
    player.damage = 0
    player.damage_taken_this_stock = 0
    player.smoothXVel = 0
    player.weapon = "Punch"
    player.hurtbox_collider = types.SimpleNamespace(center=None)
    player.state_str = type("Dummy", (), {}).__name__
    player.is_on_floor = lambda: grounded
    player.get_move = lambda: MoveType.NONE
    player.move_toward = ns["move_toward"]

    def respawn(self):
        self.body.position = Vec2(*self.start_position)
        self.body.velocity = Vec2(0, 0)
        self.damage = 0
        self.damage_taken_this_stock = 0
        self.smoothXVel = 0
        self.target_vel = (0, 0)
        self.weapon = "Punch"

    player.respawn = types.MethodType(respawn, player)

    state_classes = {
        "walking": ns["WalkingState"],
        "standing": ns["StandingState"],
        "turnaround": ns["TurnaroundState"],
        "air_turnaround": ns["AirTurnaroundState"],
        "sprinting": ns["SprintingState"],
        "in_air": ns["InAirState"],
        "dodge": ns["DodgeState"],
        "dash": ns["DashState"],
        "backdash": ns["BackDashState"],
        "KO": ns["KOState"],
    }
    player.states = {name: cls(player) for name, cls in state_classes.items()}
    player.states["stun"] = DummyState(player)
    player.states["attack"] = DummyState(player)
    player.states["taunt"] = DummyState(player)
    id_to_name = {
        0: "walking", 1: "standing", 2: "turnaround", 3: "air_turnaround", 4: "sprinting",
        6: "in_air", 7: "dodge", 9: "dash", 10: "backdash", 11: "KO",
    }
    for sid, slot in enumerate(native_state.bank):
        name = id_to_name.get(sid)
        if name is None:
            continue
        obj = player.states[name]
        for attr in (
            "invincible_timer", "dodge_cooldown", "stun_time_stored", "jump_timer", "jumps_left",
            "recoveries_left", "is_base", "turnaround_timer", "dash_timer", "backdash_timer",
            "dodge_timer", "ko_timer",
        ):
            value = getattr(slot, attr, None)
            if value is not None:
                setattr(obj, "timer" if attr == "ko_timer" else attr, value)
        if slot.dodge_is_grounded is not None and sid == STATE_DODGE:
            setattr(obj, "is_grounded", bool(slot.dodge_is_grounded))
    player.state = player.states[id_to_name[native_state.state_id]]
    player.physics_process = types.MethodType(ns["physics_process"], player)
    return ns, player


def oracle_to_native(movement, player, before_bank):
    id_by_name = {
        "WalkingState": 0, "StandingState": 1, "TurnaroundState": 2, "AirTurnaroundState": 3,
        "SprintingState": 4, "InAirState": 6, "DodgeState": 7, "DashState": 9,
        "BackDashState": 10, "KOState": 11,
    }
    state_id = id_by_name[type(player.state).__name__]
    bank = list(before_bank)
    for name, sid in (("walking", 0), ("standing", 1), ("turnaround", 2), ("air_turnaround", 3),
                      ("sprinting", 4), ("in_air", 6), ("dodge", 7), ("dash", 9),
                      ("backdash", 10), ("KO", 11)):
        obj = player.states[name]
        updates = {
            "valid": True,
            "invincible_timer": int(getattr(obj, "invincible_timer", 0)),
            "dodge_cooldown": int(getattr(obj, "dodge_cooldown", 0)),
            "stun_time_stored": int(getattr(obj, "stun_time_stored", 0)),
        }
        for attr in ("jump_timer", "jumps_left", "recoveries_left", "is_base", "turnaround_timer",
                     "dash_timer", "backdash_timer", "dodge_timer"):
            if attr in obj.__dict__:
                updates[attr] = getattr(obj, attr)
        if sid == STATE_DODGE and "is_grounded" in obj.__dict__:
            updates["dodge_is_grounded"] = bool(getattr(obj, "is_grounded"))
        if sid == STATE_KO and "timer" in obj.__dict__:
            updates["ko_timer"] = int(obj.timer)
        bank[sid] = movement.MovementSlot(**{**bank[sid].__dict__, **updates})
    return movement.MovementState(
        x=player.body.position.x,
        y=player.body.position.y,
        vx=player.body.velocity.x,
        vy=player.body.velocity.y,
        start_x=player.start_position[0],
        start_y=player.start_position[1],
        damage_vx=0.0,
        damage_vy=0.0,
        target_vx=0.0,
        target_vy=0.0,
        prev_x=player.prev_x,
        prev_y=player.prev_y,
        facing=1 if int(player.facing) == 1 else -1,
        state_id=state_id,
        stocks=player.stocks,
        velocity_func=movement.VELOCITY_DEFAULT if player.body.velocity_func == Body.update_velocity else movement.VELOCITY_NO_GRAVITY,
        body_type=player.body.body_type,
        bank=tuple(bank),
    )


def decoded(*, held=(), pressed=(), released=(), raw_horizontal=0, raw_vertical=0,
            no_horizontal=True, horizontal_state=HSTATE_NONE, last_direction=1):
    values = [0] * WIDTH
    for key in held:
        values[HELD + key] = 1
    for key in pressed:
        values[PRESSED + key] = 1
    for key in released:
        values[RELEASED + key] = 1
    values[RAW_VERTICAL] = int(raw_vertical)
    values[RAW_HORIZONTAL] = int(raw_horizontal)
    values[NO_HORIZONTAL] = int(no_horizontal)
    values[HORIZONTAL_STATE] = int(horizontal_state)
    values[LAST_DIRECTION] = int(last_direction)
    return tuple(values)


class MovementTest(unittest.TestCase):
    def setUp(self):
        import ai2.simulation.native_navigation_movement as movement

        self.m = movement
        self.cfg = movement.default_config()

    def assertCloseState(self, actual, expected, label='state'):
        self.assertEqual(actual.state_id, expected.state_id, label)
        self.assertEqual(actual.facing, expected.facing, label)
        self.assertEqual(actual.stocks, expected.stocks, label)
        self.assertEqual(actual.velocity_func, expected.velocity_func, label)
        self.assertAlmostEqual(actual.x, expected.x, places=9, msg=label)
        self.assertAlmostEqual(actual.y, expected.y, places=9, msg=label)
        self.assertAlmostEqual(actual.vx, expected.vx, places=9, msg=label)
        self.assertAlmostEqual(actual.vy, expected.vy, places=9, msg=label)
        self.assertEqual(actual.bank, expected.bank, label)

    def test_public_api_has_fixed_state_bank(self):
        state = self.m.default_state(state_id=STATE_IN_AIR)
        self.assertEqual(len(state.bank), 13)
        self.assertEqual(state.bank[STATE_IN_AIR].valid, True)
        self.assertEqual(state.bank[STATE_IN_AIR].jump_timer, 0)
        self.assertEqual(state.bank[STATE_IN_AIR].jumps_left, 0)
        self.assertEqual(state.bank[STATE_IN_AIR].recoveries_left, 0)
        self.assertEqual(state.bank[STATE_IN_AIR].is_base, True)

    def test_frozen_oracle_resolves_remote_replay_source_and_verifies_hash(self):
        global _FROZEN_MOVEMENT_NS
        previous = {name: os.environ.get(name) for name in ("AI2_MOVEMENT_ORACLE", "AI2_REPLAY_SOURCE", "AI2_FROZEN_STAGE_SOURCE")}
        _FROZEN_MOVEMENT_NS = None
        with tempfile.TemporaryDirectory(prefix="ai2-frozen-oracle-") as tmp:
            root = Path(tmp) / "variants/corrected"
            env_path = root / "UTMIST-AI2-main/environment/environment.py"
            env_path.parent.mkdir(parents=True)
            source = resolve_frozen_env().read_bytes()
            env_path.write_bytes(source)
            os.environ.pop("AI2_MOVEMENT_ORACLE", None)
            os.environ["AI2_REPLAY_SOURCE"] = str(root)
            os.environ.pop("AI2_FROZEN_STAGE_SOURCE", None)
            self.assertEqual(resolve_frozen_env(), env_path)
            env_path.write_text("wrong source")
            with self.assertRaisesRegex(AssertionError, "hash mismatch"):
                resolve_frozen_env()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _FROZEN_MOVEMENT_NS = None

    def test_frozen_source_contract_anchors_are_still_the_rules_under_test(self):
        source = resolve_frozen_env().read_text()
        self.assertIn("'WalkingState': 0", source)
        self.assertIn("'DodgeState': 7", source)
        self.assertIn("'DashState': 9", source)
        self.assertIn("'BackDashState': 10", source)
        self.assertIn("'KOState': 11", source)
        order = [
            "new_state.reset(self.state)",
            "self.state.exit()",
            "self.state = new_state",
            "self.state.enter()",
        ]
        positions = [source.index(token) for token in order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("self.p.body.velocity_func = DodgeState.no_gravity_velocity_func", source)
        self.assertIn("self.p.body.velocity_func = pymunk.Body.update_velocity", source)
        self.assertIn("return Facing.RIGHT if direction > 0 else Facing.LEFT", source)

    def test_rejects_unsupported_states_and_combat_inputs_but_keeps_dodge_button(self):
        for state_id in (5, 8, 12):
            with self.assertRaisesRegex(ValueError, "unsupported movement state"):
                self.m.step_movement(self.cfg, self.m.default_state(state_id=state_id), decoded())
        for combat_key in (KEY_H, KEY_J, KEY_K, KEY_G):
            with self.assertRaisesRegex(ValueError, "combat input"):
                self.m.step_movement(
                    self.cfg,
                    self.m.default_state(state_id=STATE_STANDING),
                    decoded(held=(combat_key,), pressed=(combat_key,)),
                    grounded=True,
                )
        result = self.m.step_movement(
            self.cfg,
            self.m.default_state(state_id=STATE_STANDING),
            decoded(held=(KEY_L,), pressed=(KEY_L,)),
            grounded=True,
        )
        self.assertEqual(result.state.state_id, STATE_DODGE)

    def test_actual_frozen_ast_player_physics_process_matches_native_reducer(self):
        cases = []
        standing = self.m.default_state(state_id=STATE_STANDING, facing=1, vx=2.0, vy=0.25)
        cases.append((standing, decoded(held=(KEY_D,), pressed=(KEY_D,), raw_horizontal=1,
                                        no_horizontal=False, horizontal_state=HSTATE_RIGHT, last_direction=1),
                      dict(grounded=True), "standing-to-walking"))
        walking = self.m.default_state(state_id=STATE_WALKING, facing=1, vx=0.0, vy=0.5)
        cases.append((walking, decoded(held=(KEY_D, KEY_L), pressed=(KEY_L,), raw_horizontal=1,
                                       no_horizontal=False, horizontal_state=HSTATE_RIGHT, last_direction=1),
                      dict(grounded=True), "walking-to-dash"))
        sprinting = self.m.default_state(state_id=STATE_SPRINTING, facing=-1, vx=0.0, vy=0.5)
        cases.append((sprinting, decoded(held=(KEY_A,), raw_horizontal=-1,
                                         no_horizontal=False, horizontal_state=HSTATE_LEFT, last_direction=-1),
                      dict(grounded=True), "sprinting-velocity"))
        turning = self.m.replace_slot(self.m.default_state(state_id=STATE_TURNAROUND, facing=1),
                                      STATE_TURNAROUND, turnaround_timer=0)
        cases.append((turning, decoded(held=(KEY_D,), raw_horizontal=1, no_horizontal=False,
                                       horizontal_state=HSTATE_RIGHT, last_direction=1),
                      dict(grounded=True), "turnaround-self-transition"))
        air = self.m.replace_slot(self.m.default_state(state_id=STATE_IN_AIR, facing=1, vx=3.0, vy=2.0),
                                  STATE_IN_AIR, jump_timer=0, jumps_left=2, recoveries_left=1, is_base=True)
        cases.append((air, decoded(held=(KEY_A,), pressed=(KEY_A,), raw_horizontal=-1,
                                   no_horizontal=False, horizontal_state=HSTATE_LEFT, last_direction=-1),
                      dict(grounded=False), "inair-to-airturnaround"))
        air_turn = self.m.replace_slot(self.m.default_state(state_id=STATE_AIR_TURNAROUND, facing=1, vx=1.0, vy=2.0),
                                       STATE_AIR_TURNAROUND, jump_timer=3, jumps_left=1,
                                       recoveries_left=1, turnaround_timer=0, is_base=False)
        cases.append((air_turn, decoded(held=(KEY_A,), raw_horizontal=-1, no_horizontal=False,
                                        horizontal_state=HSTATE_LEFT, last_direction=-1),
                      dict(grounded=False), "airturnaround-complete"))
        falling = self.m.replace_slot(self.m.default_state(state_id=STATE_IN_AIR, facing=1, vx=3.0, vy=2.0),
                                      STATE_IN_AIR, jump_timer=0, jumps_left=2, recoveries_left=1, is_base=False)
        cases.append((falling, decoded(held=(KEY_S,), raw_vertical=-1, raw_horizontal=1,
                                       no_horizontal=False, horizontal_state=HSTATE_RIGHT, last_direction=1),
                      dict(grounded=False), "fastfall-air-ease"))
        neutral_jump = self.m.replace_slot(self.m.default_state(state_id=STATE_IN_AIR, facing=1, vx=0.0, vy=2.0),
                                           STATE_IN_AIR, jump_timer=0, jumps_left=2, recoveries_left=1, is_base=False)
        cases.append((neutral_jump, decoded(held=(KEY_SPACE,), pressed=(KEY_SPACE,), raw_horizontal=0,
                                            no_horizontal=True, horizontal_state=HSTATE_NONE, last_direction=-1),
                      dict(grounded=False), "inair-neutral-jump-faces-left"))
        dash = self.m.replace_slot(self.m.default_state(state_id=STATE_DASH, facing=1),
                                   STATE_DASH, dash_timer=1)
        cases.append((dash, decoded(held=(KEY_D,), raw_horizontal=1, no_horizontal=False,
                                    horizontal_state=HSTATE_RIGHT, last_direction=1),
                      dict(grounded=True), "dash-complete"))
        backdash = self.m.replace_slot(self.m.default_state(state_id=STATE_BACKDASH, facing=1),
                                       STATE_BACKDASH, backdash_timer=1)
        cases.append((backdash, decoded(raw_horizontal=0, no_horizontal=True),
                      dict(grounded=True), "backdash-complete"))
        dodge = self.m.replace_slot(self.m.default_state(state_id=STATE_DODGE, velocity_func=self.m.VELOCITY_NO_GRAVITY),
                                    STATE_DODGE, dodge_timer=1, dodge_is_grounded=True)
        cases.append((dodge, decoded(), dict(grounded=True), "grounded-dodge-exit"))
        ko = self.m.replace_slot(self.m.default_state(state_id=STATE_KO, start_x=-5.0, start_y=0.0,
                                                      x=0.0, y=0.0, velocity_func=self.m.VELOCITY_NO_GRAVITY),
                                 STATE_KO, ko_timer=1)
        cases.append((ko, decoded(), dict(grounded=False), "ko-respawn"))

        for state, inp, kwargs, label in cases:
            kwargs.setdefault("platform_overlap", False)
            kwargs.setdefault("platform_velocity", (0.0, 0.0))
            expected_ns, player = native_to_oracle_state(self.m, state, inp, **kwargs)
            player.physics_process(1 / 30)
            expected = oracle_to_native(self.m, player, state.bank)
            actual = self.m.step_movement(self.cfg, state, inp, **kwargs).state
            self.assertCloseState(actual, expected, label)

        pending_stun = self.m.replace_slot(self.m.default_state(state_id=STATE_STANDING),
                                           STATE_STANDING, stun_time_stored=3)
        with self.assertRaisesRegex(ValueError, "unsupported movement state"):
            self.m.step_movement(self.cfg, pending_stun, decoded(), grounded=True)

    def random_state(self, rng, state_id):
        state = self.m.default_state(
            state_id=state_id,
            x=rng.uniform(-13.5, 13.5),
            y=rng.uniform(-7.5, 7.5),
            vx=rng.uniform(-4.0, 4.0),
            vy=rng.uniform(-3.0, 5.0),
            start_x=rng.choice([-5.0, 5.0]),
            start_y=0.0,
            damage_vx=rng.uniform(-0.2, 0.2),
            damage_vy=rng.uniform(-0.2, 0.2),
            target_vx=rng.uniform(-0.2, 0.2),
            target_vy=rng.uniform(-0.2, 0.2),
            facing=rng.choice([-1, 1]),
            stocks=rng.randrange(1, 4),
            velocity_func=self.m.VELOCITY_NO_GRAVITY if state_id in (STATE_DODGE, STATE_KO) else self.m.VELOCITY_DEFAULT,
        )
        for sid in SUPPORTED_STATE_IDS:
            state = self.m.replace_slot(
                state,
                sid,
                invincible_timer=rng.randrange(0, 4),
                dodge_cooldown=rng.randrange(0, 4),
                stun_time_stored=0,
            )
        state = self.m.replace_slot(state, STATE_TURNAROUND, turnaround_timer=rng.randrange(0, 3))
        state = self.m.replace_slot(state, STATE_AIR_TURNAROUND, turnaround_timer=rng.randrange(0, 3),
                                    jump_timer=rng.randrange(0, 4), jumps_left=rng.randrange(0, 3),
                                    recoveries_left=rng.randrange(0, 2), is_base=False)
        state = self.m.replace_slot(state, STATE_IN_AIR, jump_timer=rng.randrange(0, 4),
                                    jumps_left=rng.randrange(0, 3), recoveries_left=rng.randrange(0, 2),
                                    is_base=rng.choice([False, True]))
        state = self.m.replace_slot(state, STATE_DASH, dash_timer=rng.randrange(1, 4))
        state = self.m.replace_slot(state, STATE_BACKDASH, backdash_timer=rng.randrange(1, 4))
        state = self.m.replace_slot(state, STATE_DODGE, dodge_timer=rng.randrange(1, 4),
                                    dodge_is_grounded=rng.choice([False, True]),
                                    jump_timer=rng.randrange(0, 4), jumps_left=rng.randrange(0, 3),
                                    recoveries_left=rng.randrange(0, 2))
        state = self.m.replace_slot(state, STATE_KO, ko_timer=rng.randrange(1, 4))
        return state

    def random_decoded_input(self, rng):
        held = set()
        if rng.random() < 0.35:
            held.add(KEY_A)
        if rng.random() < 0.35:
            held.add(KEY_D)
        if rng.random() < 0.18:
            held.add(KEY_W)
        if rng.random() < 0.18:
            held.add(KEY_S)
        if rng.random() < 0.12:
            held.add(KEY_SPACE)
        if rng.random() < 0.12:
            held.add(KEY_L)
        pressed = {key for key in held if rng.random() < 0.35}
        released = {key for key in (KEY_A, KEY_D, KEY_SPACE, KEY_L) if key not in held and rng.random() < 0.08}
        raw_horizontal = int(KEY_D in held) - int(KEY_A in held)
        raw_vertical = int(KEY_W in held) - int(KEY_S in held)
        if raw_horizontal > 0:
            hstate = rng.choice([HSTATE_RIGHT, HSTATE_LEFT_RIGHT])
            last = 1
        elif raw_horizontal < 0:
            hstate = rng.choice([HSTATE_LEFT, HSTATE_RIGHT_LEFT])
            last = -1
        else:
            hstate = HSTATE_NONE
            last = 0
        return decoded(
            held=held,
            pressed=pressed,
            released=released,
            raw_horizontal=raw_horizontal,
            raw_vertical=raw_vertical,
            no_horizontal=(raw_horizontal == 0),
            horizontal_state=hstate,
            last_direction=last,
        )

    def test_randomized_actual_frozen_ast_oracle_matches_python_and_c(self):
        rng = random.Random(20260906)
        c = self.m.load_c_reference()
        seen = set()
        for idx in range(240):
            state_id = SUPPORTED_STATE_IDS[idx % len(SUPPORTED_STATE_IDS)]
            state = self.random_state(rng, state_id)
            # Force some boundary/transition-heavy lanes.
            if idx % 37 == 0:
                state = self.m.replace_slot(state, state_id, dodge_cooldown=0)
            if idx % 41 == 0:
                state = self.m.replace_slot(state, state_id, invincible_timer=3)
            if idx % 43 == 0:
                state = self.m.default_state(state_id=state_id, x=15.0, y=0.0, stocks=3,
                                             velocity_func=self.m.VELOCITY_NO_GRAVITY if state_id in (STATE_DODGE, STATE_KO) else self.m.VELOCITY_DEFAULT)
                state = self.m.replace_slot(state, state_id, stun_time_stored=0)
                if state_id == STATE_KO:
                    state = self.m.replace_slot(state, STATE_KO, ko_timer=3)
            inp = self.random_decoded_input(rng)
            kwargs = {
                "grounded": rng.choice([False, True]),
                "platform_overlap": rng.choice([False, True]),
                "platform_velocity": (rng.uniform(-1.0, 1.0), rng.uniform(-0.4, 0.4)),
            }
            _, player = native_to_oracle_state(self.m, state, inp, **kwargs)
            player.physics_process(1 / 30)
            expected = oracle_to_native(self.m, player, state.bank)
            py = self.m.step_movement(self.cfg, state, inp, **kwargs)
            cc = c.step(self.cfg, state, inp, **kwargs)
            self.assertCloseState(py.state, expected, f"python-random-{idx}-state{state_id}")
            self.assertCloseState(cc.state, expected, f"c-random-{idx}-state{state_id}")
            seen.add(state_id)
        self.assertEqual(seen, set(SUPPORTED_STATE_IDS))

    def test_counter_overflow_and_invalid_bank_values_fail_closed(self):
        bad_timer = self.m.replace_slot(self.m.default_state(state_id=STATE_IN_AIR),
                                        STATE_IN_AIR, jump_timer=2**31)
        with self.assertRaisesRegex(ValueError, "int32"):
            self.m.step_movement(self.cfg, bad_timer, decoded(), grounded=False)
        bad_bool = self.m.replace_slot(self.m.default_state(state_id=STATE_DODGE),
                                       STATE_DODGE, dodge_is_grounded=2)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.m.step_movement(self.cfg, bad_bool, decoded(), grounded=False)

    def test_standing_walk_jump_and_platform_velocity_match_frozen_rules(self):
        start = self.m.default_state(state_id=STATE_STANDING, facing=1, vx=2.0, vy=0.25)
        walk = self.m.step_movement(
            self.cfg,
            start,
            decoded(held=(KEY_D,), pressed=(KEY_D,), raw_horizontal=1, no_horizontal=False,
                    horizontal_state=HSTATE_RIGHT, last_direction=1),
            grounded=True,
        )
        self.assertEqual(walk.state.state_id, STATE_WALKING)
        self.assertEqual(walk.state.facing, 1)
        self.assertEqual(walk.events, ("transition",))

        riding = self.m.step_movement(
            self.cfg,
            self.m.default_state(state_id=STATE_STANDING, vx=3.0, vy=1.25),
            decoded(),
            grounded=True,
            platform_overlap=True,
            platform_velocity=(0.4, -0.2),
        )
        self.assertEqual(riding.state.state_id, STATE_STANDING)
        self.assertAlmostEqual(riding.state.vx, 0.4)
        self.assertAlmostEqual(riding.state.vy, -0.2)

        jump = self.m.step_movement(
            self.cfg,
            self.m.default_state(state_id=STATE_STANDING, facing=-1),
            decoded(held=(KEY_SPACE,), pressed=(KEY_SPACE,), raw_horizontal=1,
                    no_horizontal=False, horizontal_state=HSTATE_RIGHT, last_direction=1),
            grounded=True,
        )
        self.assertEqual(jump.state.state_id, STATE_IN_AIR)
        self.assertEqual(jump.state.facing, 1)
        self.assertAlmostEqual(jump.state.vy, -self.cfg.jump_speed)
        self.assertEqual(jump.state.bank[STATE_IN_AIR].jumps_left, 2)
        self.assertEqual(jump.state.bank[STATE_IN_AIR].recoveries_left, 1)

    def test_air_turnaround_fastfall_double_jump_and_landing(self):
        air = self.m.default_state(state_id=STATE_IN_AIR, facing=1, vx=3.0, vy=2.0)
        air = self.m.replace_slot(air, STATE_IN_AIR, jump_timer=0, jumps_left=2, recoveries_left=1, is_base=True)
        turning = self.m.step_movement(
            self.cfg,
            air,
            decoded(held=(KEY_A,), pressed=(KEY_A,), raw_horizontal=-1, no_horizontal=False,
                    horizontal_state=HSTATE_LEFT, last_direction=-1),
            grounded=False,
        )
        self.assertEqual(turning.state.state_id, STATE_AIR_TURNAROUND)
        self.assertAlmostEqual(turning.state.vx, 1.0)

        fastfall = self.m.step_movement(
            self.cfg,
            self.m.replace_slot(air, STATE_IN_AIR, is_base=False),
            decoded(held=(KEY_S,), raw_vertical=-1, raw_horizontal=1, no_horizontal=False,
                    horizontal_state=HSTATE_RIGHT, last_direction=1),
            grounded=False,
        )
        self.assertAlmostEqual(fastfall.state.vy, min(2.0 + self.cfg.fast_fall_ease, self.cfg.max_fall_speed))
        self.assertAlmostEqual(fastfall.state.vx, 3.0 + self.cfg.in_air_ease)

        double_jump = self.m.step_movement(
            self.cfg,
            self.m.replace_slot(air, STATE_IN_AIR, is_base=False),
            decoded(held=(KEY_SPACE,), pressed=(KEY_SPACE,), raw_horizontal=-1,
                    no_horizontal=False, horizontal_state=HSTATE_LEFT, last_direction=-1),
            grounded=False,
        )
        self.assertEqual(double_jump.state.state_id, STATE_IN_AIR)
        self.assertEqual(double_jump.state.facing, -1)
        self.assertEqual(double_jump.state.bank[STATE_IN_AIR].jumps_left, 1)
        self.assertEqual(double_jump.state.bank[STATE_IN_AIR].jump_timer, 5)
        self.assertAlmostEqual(double_jump.state.vy, -self.cfg.jump_speed)

        landed = self.m.step_movement(
            self.cfg,
            self.m.replace_slot(air, STATE_IN_AIR, is_base=False),
            decoded(raw_horizontal=0, no_horizontal=True),
            grounded=True,
        )
        self.assertEqual(landed.state.state_id, STATE_STANDING)

    def test_dash_backdash_dodge_timers_and_exit_side_effects(self):
        dash = self.m.step_movement(
            self.cfg,
            self.m.default_state(state_id=STATE_WALKING, facing=1),
            decoded(held=(KEY_D, KEY_L), pressed=(KEY_L,), raw_horizontal=1,
                    no_horizontal=False, horizontal_state=HSTATE_RIGHT, last_direction=1),
            grounded=True,
        )
        self.assertEqual(dash.state.state_id, STATE_DASH)
        self.assertEqual(dash.state.bank[STATE_DASH].dash_timer, self.cfg.dash_time)
        tick = dash.state
        for _ in range(self.cfg.dash_time):
            out = self.m.step_movement(self.cfg, tick, decoded(held=(KEY_D,), raw_horizontal=1,
                                                                no_horizontal=False), grounded=True)
            tick = out.state
        self.assertEqual(tick.state_id, STATE_SPRINTING)

        backdash = self.m.step_movement(
            self.cfg,
            self.m.default_state(state_id=STATE_WALKING, facing=1),
            decoded(held=(KEY_A, KEY_L), pressed=(KEY_A, KEY_L), raw_horizontal=-1,
                    no_horizontal=False, horizontal_state=HSTATE_LEFT, last_direction=-1),
            grounded=True,
        )
        self.assertEqual(backdash.state.state_id, STATE_BACKDASH)
        self.assertEqual(backdash.state.bank[STATE_BACKDASH].backdash_timer, self.cfg.backdash_time)
        self.assertAlmostEqual(
            self.m.step_movement(self.cfg, backdash.state, decoded(raw_horizontal=-1, no_horizontal=False), grounded=True).state.vx,
            -self.cfg.backdash_speed,
        )

        dodge = self.m.step_movement(
            self.cfg,
            self.m.default_state(state_id=STATE_STANDING, vx=5.0, vy=5.0),
            decoded(held=(KEY_L,), pressed=(KEY_L,)),
            grounded=True,
        )
        self.assertEqual(dodge.state.velocity_func, self.m.VELOCITY_NO_GRAVITY)
        self.assertEqual(dodge.state.bank[STATE_DODGE].dodge_timer, self.cfg.dodge_time)
        tick = dodge.state
        for _ in range(self.cfg.dodge_time):
            out = self.m.step_movement(self.cfg, tick, decoded(), grounded=True)
            tick = out.state
        self.assertEqual(tick.state_id, STATE_STANDING)
        self.assertEqual(tick.velocity_func, self.m.VELOCITY_DEFAULT)
        self.assertAlmostEqual(tick.vx, 0.0)
        self.assertAlmostEqual(tick.vy, 0.0)

    def test_turnaround_self_transition_and_bank_persistence(self):
        state = self.m.default_state(state_id=STATE_TURNAROUND, facing=1)
        state = self.m.replace_slot(state, STATE_TURNAROUND, turnaround_timer=0)
        out = self.m.step_movement(
            self.cfg,
            state,
            decoded(held=(KEY_D,), raw_horizontal=1, no_horizontal=False,
                    horizontal_state=HSTATE_RIGHT, last_direction=1),
            grounded=True,
        )
        self.assertEqual(out.state.state_id, STATE_TURNAROUND)
        self.assertEqual(out.state.facing, -1)
        self.assertIn("transition", out.events)
        self.assertIn("enter_turnaround", out.events)

        banked = self.m.default_state(state_id=STATE_STANDING)
        banked = self.m.replace_slot(banked, STATE_BACKDASH, valid=True, backdash_timer=4)
        after = self.m.step_movement(self.cfg, banked, decoded(), grounded=True).state
        self.assertEqual(after.bank[STATE_BACKDASH].backdash_timer, 4)

    def test_killbox_ko_enter_timer_and_respawn_boundary(self):
        killed = self.m.step_movement(
            self.cfg,
            self.m.default_state(state_id=STATE_STANDING, x=100.0, stocks=3),
            decoded(),
            grounded=False,
        )
        self.assertEqual(killed.state.state_id, STATE_KO)
        self.assertEqual(killed.state.stocks, 2)
        self.assertEqual(killed.state.velocity_func, self.m.VELOCITY_NO_GRAVITY)
        self.assertEqual(killed.state.bank[STATE_KO].ko_timer, 90)

        almost = self.m.default_state(state_id=STATE_KO, x=9.0, y=0.0, start_x=-5.0, start_y=0.0, stocks=2,
                                      velocity_func=self.m.VELOCITY_NO_GRAVITY)
        almost = self.m.replace_slot(almost, STATE_KO, ko_timer=1)
        respawned = self.m.step_movement(self.cfg, almost, decoded(), grounded=False).state
        self.assertEqual(respawned.state_id, STATE_IN_AIR)
        self.assertEqual(respawned.x, -5.0)
        self.assertEqual(respawned.y, 0.0)
        self.assertEqual(respawned.velocity_func, self.m.VELOCITY_DEFAULT)
        self.assertEqual(respawned.bank[STATE_IN_AIR].jumps_left, 0)
        self.assertEqual(respawned.bank[STATE_IN_AIR].recoveries_left, 0)

    def test_c_reference_matches_python_and_rejects_bad_inputs_without_mutation(self):
        c = self.m.load_c_reference()
        cases = [
            (
                self.m.default_state(state_id=STATE_STANDING),
                decoded(held=(KEY_D,), pressed=(KEY_D,), raw_horizontal=1, no_horizontal=False,
                        horizontal_state=HSTATE_RIGHT, last_direction=1),
                dict(grounded=True),
            ),
            (
                self.m.default_state(state_id=STATE_IN_AIR, vx=1.0, vy=3.0),
                decoded(held=(KEY_S,), raw_vertical=-1, raw_horizontal=-1, no_horizontal=False),
                dict(grounded=False),
            ),
            (
                self.m.replace_slot(
                    self.m.default_state(state_id=STATE_IN_AIR, facing=1, vx=0.0, vy=2.0),
                    STATE_IN_AIR,
                    jump_timer=0,
                    jumps_left=2,
                    recoveries_left=1,
                    is_base=False,
                ),
                decoded(held=(KEY_SPACE,), pressed=(KEY_SPACE,), raw_horizontal=0,
                        no_horizontal=True, horizontal_state=HSTATE_NONE, last_direction=-1),
                dict(grounded=False),
            ),
            (
                self.m.default_state(state_id=STATE_STANDING, x=100.0),
                decoded(),
                dict(grounded=False),
            ),
        ]
        for state, inp, kwargs in cases:
            py = self.m.step_movement(self.cfg, state, inp, **kwargs)
            cc = c.step(self.cfg, state, inp, **kwargs)
            self.assertCloseState(cc.state, py.state)
            self.assertEqual(cc.events, py.events)

        sentinel = self.m.default_state(state_id=STATE_STANDING, vx=77.0)
        with self.assertRaises(ValueError):
            c.step(self.cfg, sentinel, (0,) * 34 + (9,), grounded=True)
        self.assertEqual(sentinel.vx, 77.0)

    def test_existing_native_input_decoder_feeds_movement_decoder_contract(self):
        with tempfile.TemporaryDirectory(prefix="ai2-input-for-movement-") as build_dir:
            out = Path(build_dir) / "navigation_input.so"
            subprocess.run(
                [os.environ.get("CC", "cc"), "-std=c99", "-O2", "-ffp-contract=off",
                 "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
                 str(ROOT / "native/navigation_input.c"), "-o", str(out)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            lib = ctypes.CDLL(str(out))
            lib.ai2_input_init.argtypes = [ctypes.POINTER(ctypes.c_int32)]
            lib.ai2_input_step.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
            ]
            before = (ctypes.c_int32 * WIDTH)()
            after = (ctypes.c_int32 * WIDTH)()
            lib.ai2_input_init(before)
            action = (ctypes.c_double * 10)(0, 0, 0, 1, 0, 0, 1, 0, 0, 0)
            self.assertEqual(lib.ai2_input_step(action, before, after), 0)
            result = self.m.step_movement(
                self.cfg,
                self.m.default_state(state_id=STATE_STANDING, facing=1),
                tuple(after),
                grounded=True,
            )
            self.assertEqual(result.state.state_id, STATE_WALKING)

    def test_cuda_batch_lane_isolation_matches_c_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable; CUDA validation must run remotely")
        if not torch.cuda.is_available() or not shutil.which("nvcc"):
            self.skipTest("CUDA runtime/compiler unavailable")
        c = self.m.load_c_reference()
        cuda = self.m.load_cuda_reference()
        cases = [
            (self.m.default_state(state_id=STATE_STANDING, x=float(i), vx=float(i % 3)),
             decoded(held=(KEY_D,), raw_horizontal=1, no_horizontal=False), True, False, (0.0, 0.0))
            for i in range(8)
        ]
        expected = [c.step(self.cfg, s, a, grounded=g, platform_overlap=o, platform_velocity=p).state for s, a, g, o, p in cases]
        c_states = [self.m._to_c_state(s) for s, _, _, _, _ in cases]
        c_inputs = [self.m._to_c_input(a, g, o, p) for _, a, g, o, p in cases]
        state_bytes = cuda._bytes_tensor(torch, c_states, self.m._CState, "cuda")
        input_bytes = cuda._bytes_tensor(torch, c_inputs, self.m._CInput, "cuda")
        out_bytes = torch.full((len(cases), ctypes.sizeof(self.m._CResult)), 77, dtype=torch.uint8, device="cuda")
        errors = torch.empty((len(cases),), dtype=torch.int32, device="cuda")
        with self.assertRaisesRegex(ValueError, "CUDA tensor"):
            cuda.step_batch_tensors(self.cfg, state_bytes.cpu(), input_bytes, out_bytes, errors)
        with self.assertRaisesRegex(ValueError, "dtype"):
            cuda.step_batch_tensors(self.cfg, state_bytes.to(torch.int32), input_bytes, out_bytes, errors)
        noncontig = torch.empty((ctypes.sizeof(self.m._CState), len(cases)), dtype=torch.uint8, device="cuda").transpose(0, 1)
        self.assertFalse(noncontig.is_contiguous())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            cuda.step_batch_tensors(self.cfg, noncontig, input_bytes, out_bytes, errors)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            actual = cuda.step_batch(self.cfg, cases, stream=stream)
        for got, exp in zip(actual, expected):
            self.assertCloseState(got.state, exp)
        bad_cases = list(cases)
        bad_cases[3] = (self.m.default_state(state_id=STATE_STANDING), (0,) * 34 + (9,), True, False, (0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "lane errors"):
            cuda.step_batch(self.cfg, bad_cases, stream=torch.cuda.Stream())


if __name__ == "__main__":
    unittest.main()
