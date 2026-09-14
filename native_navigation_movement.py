#!/usr/bin/env python3
"""Shared native movement reducer for frozen navigation player states.

Scope: reproduces the state/timer/velocity-function branch logic around
``Player.state.physics_process`` and ``Player.physics_process`` for
navigation-reachable states.  It consumes decoded input[35] from
``native/navigation_input`` plus oracle-supplied grounded/platform facts.
It does not step Pymunk, resolve contacts, execute attacks, or infer floor
state from geometry.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence

ROOT = Path(__file__).resolve().parent
NATIVE_DIR = ROOT / "native"

STATE_WALKING = 0
STATE_STANDING = 1
STATE_TURNAROUND = 2
STATE_AIR_TURNAROUND = 3
STATE_SPRINTING = 4
STATE_STUN = 5
STATE_IN_AIR = 6
STATE_DODGE = 7
STATE_ATTACK = 8
STATE_DASH = 9
STATE_BACKDASH = 10
STATE_KO = 11
STATE_TAUNT = 12

SUPPORTED_STATES = {
    STATE_WALKING,
    STATE_STANDING,
    STATE_TURNAROUND,
    STATE_AIR_TURNAROUND,
    STATE_SPRINTING,
    STATE_IN_AIR,
    STATE_DODGE,
    STATE_DASH,
    STATE_BACKDASH,
    STATE_KO,
}

VELOCITY_DEFAULT = 0
VELOCITY_NO_GRAVITY = 1

KEY_W, KEY_A, KEY_S, KEY_D, KEY_SPACE, KEY_H, KEY_L, KEY_J, KEY_K, KEY_G = range(10)
OFF_HELD, OFF_PRESSED, OFF_RELEASED = 0, 10, 20
OFF_RAW_VERTICAL, OFF_RAW_HORIZONTAL, OFF_NO_HORIZONTAL, OFF_HORIZONTAL_STATE, OFF_LAST_DIRECTION = 30, 31, 32, 33, 34
INPUT_WIDTH = 35
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

EVENTS = (
    ("transition", 1 << 0),
    ("enter_turnaround", 1 << 1),
    ("enter_air_turnaround", 1 << 2),
    ("enter_dash", 1 << 3),
    ("enter_backdash", 1 << 4),
    ("enter_dodge", 1 << 5),
    ("exit_dodge", 1 << 6),
    ("enter_ko", 1 << 7),
    ("exit_ko", 1 << 8),
    ("respawn", 1 << 9),
)


@dataclass(frozen=True)
class MovementConfig:
    stage_width_tiles: float = 29.8
    stage_height_tiles: float = 16.8
    move_speed: float = 6.75
    jump_speed: float = 8.9
    max_fall_speed: float = 12.0
    fast_fall_ease: float = 14.25 / 30.0
    in_air_ease: float = 10.75 / 30.0
    run_speed: float = 8.0
    dash_speed: float = 10.0
    backdash_speed: float = 4.0
    turnaround_time: int = 1
    backdash_time: int = 7
    dodge_time: int = 10
    grounded_dodge_cooldown: int = 30
    air_dodge_cooldown: int = 82
    invincible_time: int = 90
    jump_cooldown: int = 6
    dash_time: int = 9


@dataclass(frozen=True)
class MovementSlot:
    valid: bool = True
    invincible_timer: int = 0
    dodge_cooldown: int = 0
    stun_time_stored: int = 0
    jump_timer: int | None = None
    jumps_left: int | None = None
    recoveries_left: int | None = None
    is_base: bool | None = None
    turnaround_timer: int | None = None
    dash_timer: int | None = None
    backdash_timer: int | None = None
    dodge_timer: int | None = None
    dodge_is_grounded: bool | None = None
    ko_timer: int | None = None


@dataclass(frozen=True)
class MovementState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    start_x: float = 0.0
    start_y: float = 0.0
    damage_vx: float = 0.0
    damage_vy: float = 0.0
    target_vx: float = 0.0
    target_vy: float = 0.0
    prev_x: float = 0.0
    prev_y: float = 0.0
    facing: int = 1
    state_id: int = STATE_IN_AIR
    stocks: int = 3
    velocity_func: int = VELOCITY_DEFAULT
    body_type: int = 0
    bank: tuple[MovementSlot, ...] = ()


@dataclass(frozen=True)
class MovementStepResult:
    state: MovementState
    transitioned: bool
    from_state: int
    to_state: int
    events: tuple[str, ...]


def default_config() -> MovementConfig:
    return MovementConfig()


def _default_bank() -> tuple[MovementSlot, ...]:
    bank = [MovementSlot() for _ in range(13)]
    bank[STATE_IN_AIR] = replace(bank[STATE_IN_AIR], jump_timer=0, jumps_left=0, recoveries_left=0, is_base=True)
    return tuple(bank)


def default_state(*, state_id: int = STATE_IN_AIR, **updates) -> MovementState:
    if state_id not in SUPPORTED_STATES:
        raise ValueError(f"unsupported movement state {state_id}")
    return MovementState(state_id=state_id, bank=_default_bank(), **updates)


def replace_slot(state: MovementState, state_id: int, **updates) -> MovementState:
    bank = list(state.bank or _default_bank())
    bank[state_id] = replace(bank[state_id], **updates)
    return replace(state, bank=tuple(bank))


def _validate(config: MovementConfig, state: MovementState, decoded_input: Sequence[int]) -> tuple[int, ...]:
    if state.state_id not in SUPPORTED_STATES:
        raise ValueError(f"unsupported movement state {state.state_id}")
    values = [config.stage_width_tiles, config.stage_height_tiles, config.move_speed, config.jump_speed,
              config.max_fall_speed, config.fast_fall_ease, config.in_air_ease, config.run_speed,
              config.dash_speed, config.backdash_speed, state.x, state.y, state.vx, state.vy,
              state.start_x, state.start_y, state.damage_vx, state.damage_vy, state.target_vx,
              state.target_vy, state.prev_x, state.prev_y]
    if any(not math.isfinite(float(v)) for v in values):
        raise ValueError("movement fields must be finite")
    if len(decoded_input) != INPUT_WIDTH:
        raise ValueError("decoded input must have length 35")
    inp = tuple(int(v) for v in decoded_input)
    if any(inp[i] != decoded_input[i] for i in range(INPUT_WIDTH)):
        raise ValueError("decoded input values must be integers")
    if not (0 <= inp[OFF_HORIZONTAL_STATE] <= 4):
        raise ValueError("invalid horizontal state")
    if any(inp[OFF_HELD + k] for k in (KEY_H, KEY_J, KEY_K, KEY_G)):
        raise ValueError("combat input keys are outside navigation movement contract")
    if len(state.bank) != 13:
        raise ValueError("movement state bank must contain 13 slots")
    int_config_fields = (
        config.turnaround_time, config.backdash_time, config.dodge_time,
        config.grounded_dodge_cooldown, config.air_dodge_cooldown,
        config.invincible_time, config.jump_cooldown, config.dash_time,
        state.facing, state.state_id, state.stocks, state.velocity_func, state.body_type,
    )
    if any(type(v) is bool or int(v) != v or int(v) < INT32_MIN or int(v) > INT32_MAX for v in int_config_fields):
        raise ValueError("movement integer fields must fit int32")
    bool_fields = {"valid", "is_base", "dodge_is_grounded"}
    for slot in state.bank:
        for name, value in slot.__dict__.items():
            if value is None:
                continue
            if name in bool_fields:
                if type(value) is not bool:
                    raise ValueError("movement state bank boolean fields must be bool")
            elif type(value) is bool or int(value) != value or int(value) < INT32_MIN or int(value) > INT32_MAX:
                raise ValueError("movement state bank integer fields must fit int32")
    if state.bank[state.state_id].stun_time_stored > 0:
        raise ValueError("unsupported movement state stun transition")
    if state.facing not in (-1, 1):
        raise ValueError("facing must be -1 or 1")
    return inp


def _events(mask: int) -> tuple[str, ...]:
    return tuple(name for name, bit in EVENTS if mask & bit)


def _move_toward(current: float, target: float, delta: float) -> float:
    if current == target:
        return target
    direction = (target - current) / abs(target - current)
    if delta >= 0:
        candidate = current + delta * direction
        return min(candidate, target) if direction > 0 else max(candidate, target)
    return current - abs(delta) * direction


def _hstate_last_int(value: int) -> int:
    if value in (1, 4):
        return -1
    if value in (2, 3):
        return 1
    return 0


def _turn_check(facing: int, direction: int) -> bool:
    return (facing == 1 and direction < 0) or (facing == -1 and direction > 0)


def _facing_from_direction(direction: int) -> int:
    return 1 if direction > 0 else -1


def _ground_state(inp: Sequence[int]) -> int:
    return STATE_WALKING if abs(inp[OFF_RAW_HORIZONTAL]) > 0 else STATE_STANDING


def _slot_set(state: MovementState, sid: int, **updates) -> MovementState:
    return replace_slot(state, sid, valid=True, **updates)


def _common_reset(state: MovementState, target: int, old: int) -> MovementState:
    old_slot = state.bank[old]
    updates = dict(stun_time_stored=0, invincible_timer=old_slot.invincible_timer, dodge_cooldown=old_slot.dodge_cooldown)
    if target in (STATE_WALKING, STATE_STANDING, STATE_TURNAROUND, STATE_SPRINTING, STATE_DASH, STATE_BACKDASH, STATE_KO):
        updates["dash_timer"] = old_slot.dash_timer or 0
    return _slot_set(state, target, **updates)


def _enter(state: MovementState, config: MovementConfig, target: int) -> tuple[MovementState, int]:
    mask = 0
    if target == STATE_IN_AIR:
        state = _slot_set(state, STATE_IN_AIR, is_base=True)
    elif target == STATE_TURNAROUND:
        state = _slot_set(state, target, turnaround_timer=config.turnaround_time)
        mask |= 1 << 1
    elif target == STATE_AIR_TURNAROUND:
        state = _slot_set(state, target, turnaround_timer=config.turnaround_time, is_base=False)
        state = replace(state, vx=state.vx / 3.0)
        mask |= 1 << 2
    elif target == STATE_DASH:
        state = _slot_set(state, target, dash_timer=config.dash_time)
        mask |= 1 << 3
    elif target == STATE_BACKDASH:
        state = _slot_set(state, target, backdash_timer=config.backdash_time)
        mask |= 1 << 4
    elif target == STATE_DODGE:
        state = _slot_set(state, target, dodge_timer=config.dodge_time)
        state = replace(state, velocity_func=VELOCITY_NO_GRAVITY, vx=0.0, vy=0.0)
        mask |= 1 << 5
    elif target == STATE_KO:
        state = _slot_set(state, target, ko_timer=90)
        state = replace(state, stocks=max(0, state.stocks - 1), velocity_func=VELOCITY_NO_GRAVITY, vx=0.0, vy=0.0)
        mask |= 1 << 7
    return state, mask


def _exit(state: MovementState, old: int) -> tuple[MovementState, int]:
    if old == STATE_DODGE:
        return replace(state, body_type=0, velocity_func=VELOCITY_DEFAULT, vx=0.0, vy=0.0), 1 << 6
    if old == STATE_KO:
        state = _slot_set(state, STATE_KO, invincible_timer=90)
        return replace(state, body_type=0, velocity_func=VELOCITY_DEFAULT, vx=0.0, vy=0.0), 1 << 8
    return state, 0


def _transition(state: MovementState, config: MovementConfig, old: int, target: int) -> tuple[MovementState, int]:
    state = _common_reset(state, target, old)
    state, exit_mask = _exit(state, old)
    state = replace(state, state_id=target)
    state, enter_mask = _enter(state, config, target)
    return state, (1 << 0) | exit_mask | enter_mask


def _common(config: MovementConfig, state: MovementState, current: int) -> tuple[MovementState, int | None]:
    if abs(state.x) > math.floor(config.stage_width_tiles / 2.0) or abs(state.y) > math.floor(config.stage_height_tiles / 2.0):
        return state, STATE_KO
    slot = state.bank[current]
    if slot.stun_time_stored > 0:
        raise ValueError("unsupported movement state stun transition")
    state = _slot_set(state, current,
                      invincible_timer=max(0, slot.invincible_timer - 1),
                      dodge_cooldown=max(0, slot.dodge_cooldown - 1))
    return state, None


def _refresh_in_air(state: MovementState) -> MovementState:
    return _slot_set(state, STATE_IN_AIR, jump_timer=0, jumps_left=2, recoveries_left=1)


def _ground_common(config: MovementConfig, state: MovementState, inp: Sequence[int], current: int, grounded: bool) -> tuple[MovementState, int | None]:
    state, target = _common(config, state, current)
    if target is not None:
        return state, target
    direction = inp[OFF_RAW_HORIZONTAL]
    near_still = abs(direction) == 0
    if inp[OFF_PRESSED + KEY_SPACE] and grounded:
        state = replace(state, vy=-config.jump_speed, facing=_facing_from_direction(direction) if not near_still else state.facing)
        return _refresh_in_air(state), STATE_IN_AIR
    if not grounded:
        return _refresh_in_air(state), STATE_IN_AIR
    if near_still and inp[OFF_PRESSED + KEY_L] and state.bank[current].dodge_cooldown <= 0:
        state = _slot_set(state, current, dodge_cooldown=config.grounded_dodge_cooldown)
        state = _slot_set(state, STATE_DODGE, dodge_is_grounded=True)
        return state, STATE_DODGE
    return state, None


def _state_physics(config: MovementConfig, state: MovementState, inp: Sequence[int], grounded: bool,
                   platform_overlap: bool, platform_velocity: tuple[float, float]) -> tuple[MovementState, int | None, int]:
    current = state.state_id
    if current == STATE_STANDING:
        state, target = _ground_common(config, state, inp, current, grounded)
        if target is not None:
            return state, target, 0
        direction = inp[OFF_RAW_HORIZONTAL]
        if _turn_check(state.facing, direction):
            return state, STATE_BACKDASH if inp[OFF_PRESSED + KEY_L] else STATE_TURNAROUND, 0
        if abs(direction) > 0:
            return replace(state, facing=_facing_from_direction(direction)), STATE_WALKING, 0
        if platform_overlap and not inp[OFF_HELD + KEY_S] and grounded:
            return replace(state, vx=platform_velocity[0], vy=platform_velocity[1]), None, 0
        return replace(state, vx=_move_toward(state.vx, 0.0, config.move_speed)), None, 0
    if current in (STATE_WALKING, STATE_SPRINTING):
        state, target = _ground_common(config, state, inp, current, grounded)
        if target is not None:
            return state, target, 0
        original = KEY_D if state.facing == 1 else KEY_A
        opposite = KEY_A if state.facing == 1 else KEY_D
        if inp[OFF_PRESSED + opposite] or (not inp[OFF_HELD + original] and inp[OFF_HELD + opposite]):
            return state, STATE_BACKDASH if inp[OFF_PRESSED + KEY_L] else STATE_TURNAROUND, 0
        if inp[OFF_NO_HORIZONTAL]:
            return state, STATE_STANDING, 0
        if inp[OFF_PRESSED + KEY_L]:
            return state, STATE_DASH, 0
        speed = config.move_speed if current == STATE_WALKING else config.run_speed
        if current == STATE_WALKING and platform_overlap and not inp[OFF_HELD + KEY_S] and grounded:
            return replace(state, vx=state.facing * speed + platform_velocity[0], vy=state.vy + platform_velocity[1]), None, 0
        return replace(state, vx=state.facing * speed), None, 0
    if current == STATE_TURNAROUND:
        state, target = _ground_common(config, state, inp, current, grounded)
        if target is not None:
            return state, target, 0
        timer = state.bank[current].turnaround_timer or 0
        if timer <= 0:
            old = state.facing
            state = replace(state, facing=-state.facing)
            last = _hstate_last_int(inp[OFF_HORIZONTAL_STATE])
            if -old == last:
                return state, STATE_WALKING, 0
            if old == last:
                return state, STATE_TURNAROUND, 0
            return state, STATE_STANDING, 0
        if inp[OFF_PRESSED + KEY_SPACE] and grounded:
            return replace(state, facing=-state.facing, vy=-config.jump_speed), STATE_IN_AIR, 0
        if inp[OFF_PRESSED + KEY_L]:
            return state, STATE_BACKDASH, 0
        return _slot_set(state, current, turnaround_timer=max(0, timer - 1)), None, 0
    if current == STATE_IN_AIR:
        state, target = _common(config, state, current)
        if target is not None:
            return state, target, 0
        slot = state.bank[current]
        direction = inp[OFF_RAW_HORIZONTAL]
        if (slot.is_base if slot.is_base is not None else True) and _turn_check(state.facing, direction):
            state = _slot_set(state, STATE_AIR_TURNAROUND,
                              jump_timer=slot.jump_timer or 0,
                              jumps_left=slot.jumps_left or 0,
                              recoveries_left=slot.recoveries_left or 0)
            return state, STATE_AIR_TURNAROUND, 0
        vy = _move_toward(state.vy, config.max_fall_speed, config.fast_fall_ease) if inp[OFF_RAW_VERTICAL] < 0 and state.vy > 0 else min(state.vy, config.max_fall_speed)
        state = replace(state, vy=vy, vx=_move_toward(state.vx, direction * config.move_speed, config.in_air_ease))
        if grounded:
            return state, _ground_state(inp), 0
        jump_timer = slot.jump_timer or 0
        jumps_left = slot.jumps_left or 0
        if inp[OFF_PRESSED + KEY_SPACE] and jump_timer <= 0 and jumps_left > 0:
            state = replace(state, vy=-config.jump_speed, facing=_facing_from_direction(direction))
            jump_timer = config.jump_cooldown
            jumps_left -= 1
        jump_timer = max(0, jump_timer - 1)
        state = _slot_set(state, current, jump_timer=jump_timer, jumps_left=jumps_left)
        if inp[OFF_PRESSED + KEY_L] and state.bank[current].dodge_cooldown <= 0:
            state = _slot_set(state, current, dodge_cooldown=config.air_dodge_cooldown)
            state = _slot_set(state, STATE_DODGE, jump_timer=jump_timer, jumps_left=jumps_left,
                              recoveries_left=slot.recoveries_left or 0, dodge_is_grounded=False)
            return state, STATE_DODGE, 0
        return state, None, 0
    if current == STATE_AIR_TURNAROUND:
        state, target = _common(config, state, current)
        if target is not None:
            return state, target, 0
        slot = state.bank[current]
        direction = inp[OFF_RAW_HORIZONTAL]
        vy = _move_toward(state.vy, config.max_fall_speed, config.fast_fall_ease) if inp[OFF_RAW_VERTICAL] < 0 and state.vy > 0 else min(state.vy, config.max_fall_speed)
        state = replace(state, vy=vy, vx=_move_toward(state.vx, direction * config.move_speed, config.in_air_ease))
        if grounded:
            return state, _ground_state(inp), 0
        jump_timer = slot.jump_timer or 0
        jumps_left = slot.jumps_left or 0
        if inp[OFF_PRESSED + KEY_SPACE] and jump_timer <= 0 and jumps_left > 0:
            state = replace(state, vy=-config.jump_speed, facing=_facing_from_direction(direction))
            jump_timer = config.jump_cooldown
            jumps_left -= 1
        jump_timer = max(0, jump_timer - 1)
        state = _slot_set(state, current, jump_timer=jump_timer, jumps_left=jumps_left)
        if inp[OFF_PRESSED + KEY_L] and state.bank[current].dodge_cooldown <= 0:
            state = _slot_set(state, current, dodge_cooldown=config.air_dodge_cooldown)
            state = _slot_set(state, STATE_DODGE, jump_timer=jump_timer, jumps_left=jumps_left,
                              recoveries_left=slot.recoveries_left or 0, dodge_is_grounded=False)
            return state, STATE_DODGE, 0
        timer = slot.turnaround_timer or 0
        if timer <= 0:
            old = state.facing
            state = replace(state, facing=-state.facing)
            last = _hstate_last_int(inp[OFF_HORIZONTAL_STATE])
            dest = STATE_IN_AIR if -old == last else (STATE_AIR_TURNAROUND if old == last else STATE_IN_AIR)
            state = _slot_set(state, dest, jump_timer=jump_timer,
                              jumps_left=jumps_left, recoveries_left=slot.recoveries_left or 0)
            return state, dest, 0
        return _slot_set(state, current, turnaround_timer=max(0, timer - 1)), None, 0
    if current == STATE_DASH:
        state, target = _ground_common(config, state, inp, current, grounded)
        if target is not None:
            return state, target, 0
        timer = max(0, (state.bank[current].dash_timer or 0) - 1)
        state = _slot_set(replace(state, vx=state.facing * config.dash_speed), current, dash_timer=timer)
        return state, STATE_SPRINTING if timer <= 0 else None, 0
    if current == STATE_BACKDASH:
        state, target = _common(config, state, current)
        if target is not None:
            return state, target, 0
        timer = max(0, (state.bank[current].backdash_timer or 0) - 1)
        state = _slot_set(replace(state, vx=-state.facing * config.backdash_speed), current, backdash_timer=timer)
        return state, _ground_state(inp) if timer <= 0 else None, 0
    if current == STATE_DODGE:
        state, target = _common(config, state, current)
        if target is not None:
            return state, target, 0
        slot = state.bank[current]
        timer = max(0, (slot.dodge_timer or 0) - 1)
        state = _slot_set(state, current, dodge_timer=timer)
        if timer <= 0:
            if slot.dodge_is_grounded:
                return state, _ground_state(inp), 0
            state = _slot_set(state, STATE_IN_AIR, jumps_left=slot.jumps_left or 0, jump_timer=0,
                              recoveries_left=slot.recoveries_left or 0)
            return state, STATE_IN_AIR, 0
        return state, None, 0
    if current == STATE_KO:
        state, _ = _common(config, state, current)
        timer = (state.bank[current].ko_timer or 0) - 1
        state = _slot_set(state, current, ko_timer=timer)
        if timer <= 0:
            state = replace(state, x=state.start_x, y=state.start_y, vx=0.0, vy=0.0,
                            damage_vx=0.0, damage_vy=0.0, target_vx=0.0, target_vy=0.0)
            state = _slot_set(state, STATE_IN_AIR, jumps_left=0, recoveries_left=0)
            return state, STATE_IN_AIR, 1 << 9
        return state, None, 0
    raise ValueError(f"unsupported movement state {current}")


def step_movement(config: MovementConfig, state: MovementState, decoded_input: Sequence[int], *,
                  grounded: bool = False, platform_overlap: bool = False,
                  platform_velocity: tuple[float, float] = (0.0, 0.0)) -> MovementStepResult:
    inp = _validate(config, state, decoded_input)
    if not all(math.isfinite(float(v)) for v in platform_velocity):
        raise ValueError("platform velocity must be finite")
    working, target, mask = _state_physics(config, state, inp, bool(grounded), bool(platform_overlap), platform_velocity)
    old = state.state_id
    working = replace(working,
                      vx=working.vx + working.damage_vx + working.target_vx,
                      vy=working.vy + working.damage_vy + working.target_vy)
    if target is not None:
        working, transition_mask = _transition(working, config, old, target)
        mask |= transition_mask
    working = replace(working,
                      prev_x=working.x, prev_y=working.y,
                      damage_vx=0.0, damage_vy=0.0, target_vx=0.0, target_vy=0.0)
    return MovementStepResult(working, target is not None, old, working.state_id, _events(mask))


class _CConfig(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in (
        "stage_width_tiles", "stage_height_tiles", "move_speed", "jump_speed", "max_fall_speed",
        "fast_fall_ease", "in_air_ease", "run_speed", "dash_speed", "backdash_speed")]
    _fields_ += [(name, ctypes.c_int32) for name in (
        "turnaround_time", "backdash_time", "dodge_time", "grounded_dodge_cooldown", "air_dodge_cooldown",
        "invincible_time", "jump_cooldown", "dash_time")]


class _CSlot(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int32 * 14), ("present", ctypes.c_uint8 * 14)]


class _CState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in (
        "x", "y", "vx", "vy", "start_x", "start_y", "damage_vx", "damage_vy",
        "target_vx", "target_vy", "prev_x", "prev_y")]
    _fields_ += [(name, ctypes.c_int32) for name in ("facing", "state_id", "stocks", "velocity_func", "body_type")]
    _fields_ += [("bank", _CSlot * 13)]


class _CInput(ctypes.Structure):
    _fields_ = [("decoded", ctypes.c_int32 * 35), ("grounded", ctypes.c_uint8), ("platform_overlap", ctypes.c_uint8),
                ("platform_vx", ctypes.c_double), ("platform_vy", ctypes.c_double)]


class _CResult(ctypes.Structure):
    _fields_ = [("state", _CState), ("transitioned", ctypes.c_uint8),
                ("from_state", ctypes.c_int32), ("to_state", ctypes.c_int32), ("event_mask", ctypes.c_uint32)]


def _to_c_config(config: MovementConfig) -> _CConfig:
    return _CConfig(*(getattr(config, field) for field, _ in _CConfig._fields_))


def _to_c_state(state: MovementState) -> _CState:
    c = _CState()
    for field, _ in _CState._fields_[:12]:
        setattr(c, field, float(getattr(state, field)))
    for field, _ in _CState._fields_[12:17]:
        setattr(c, field, int(getattr(state, field)))
    bank = state.bank or _default_bank()
    names = list(MovementSlot.__dataclass_fields__)
    for sid, slot in enumerate(bank):
        for fid, name in enumerate(names):
            value = getattr(slot, name)
            if value is None:
                c.bank[sid].present[fid] = 0
                c.bank[sid].value[fid] = 0
            else:
                c.bank[sid].present[fid] = 1
                c.bank[sid].value[fid] = int(value)
    return c


def _from_c_state(c: _CState) -> MovementState:
    names = list(MovementSlot.__dataclass_fields__)
    slots = []
    for sid in range(13):
        values = {}
        for fid, name in enumerate(names):
            if c.bank[sid].present[fid]:
                if name in ("valid", "is_base", "dodge_is_grounded"):
                    values[name] = bool(c.bank[sid].value[fid])
                else:
                    values[name] = int(c.bank[sid].value[fid])
            else:
                values[name] = None if name not in ("valid", "invincible_timer", "dodge_cooldown", "stun_time_stored") else (False if name == "valid" else 0)
        slots.append(MovementSlot(**values))
    return MovementState(
        x=c.x, y=c.y, vx=c.vx, vy=c.vy, start_x=c.start_x, start_y=c.start_y,
        damage_vx=c.damage_vx, damage_vy=c.damage_vy, target_vx=c.target_vx, target_vy=c.target_vy,
        prev_x=c.prev_x, prev_y=c.prev_y, facing=c.facing, state_id=c.state_id, stocks=c.stocks,
        velocity_func=c.velocity_func, body_type=c.body_type, bank=tuple(slots),
    )


def _to_c_input(decoded_input: Sequence[int], grounded: bool, platform_overlap: bool, platform_velocity: tuple[float, float]) -> _CInput:
    return _CInput((ctypes.c_int32 * 35)(*decoded_input), int(bool(grounded)), int(bool(platform_overlap)),
                   float(platform_velocity[0]), float(platform_velocity[1]))


class CMovementReference:
    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib
        self.lib.ai2_movement_step.argtypes = [ctypes.POINTER(_CConfig), ctypes.POINTER(_CState), ctypes.POINTER(_CInput), ctypes.POINTER(_CResult)]
        self.lib.ai2_movement_step.restype = ctypes.c_int

    def step(self, config: MovementConfig, state: MovementState, decoded_input: Sequence[int], *,
             grounded: bool = False, platform_overlap: bool = False,
             platform_velocity: tuple[float, float] = (0.0, 0.0)) -> MovementStepResult:
        inp = _validate(config, state, decoded_input)
        ccfg = _to_c_config(config)
        cstate = _to_c_state(state)
        cin = _to_c_input(inp, grounded, platform_overlap, platform_velocity)
        out = _CResult()
        status = self.lib.ai2_movement_step(ctypes.byref(ccfg), ctypes.byref(cstate), ctypes.byref(cin), ctypes.byref(out))
        if status == 2:
            raise ValueError("unsupported movement state")
        if status == 3:
            raise ValueError("combat input")
        if status != 0:
            raise ValueError(f"movement C reducer failed status={status}")
        return MovementStepResult(_from_c_state(out.state), bool(out.transitioned), int(out.from_state), int(out.to_state), _events(int(out.event_mask)))


def load_c_reference() -> CMovementReference:
    build = Path(tempfile.mkdtemp(prefix="ai2-movement-"))
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    out = build / f"navigation_movement{suffix}"
    cmd = [os.environ.get("CC", "cc"), "-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
           "-fPIC", "-shared", str(NATIVE_DIR / "navigation_movement.c"), "-o", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    return CMovementReference(ctypes.CDLL(str(out)))


class CudaMovementReference:
    def __init__(self, lib: ctypes.CDLL, torch_module):
        self.lib = lib
        self.torch = torch_module
        self.lib.ai2_movement_step_cuda.argtypes = [
            ctypes.POINTER(_CConfig),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.ai2_movement_step_cuda.restype = ctypes.c_int

    @staticmethod
    def _bytes_tensor(torch_module, structs: Sequence[ctypes.Structure], struct_type, device: str):
        import numpy as np

        arr = np.empty((len(structs), ctypes.sizeof(struct_type)), dtype=np.uint8)
        for i, value in enumerate(structs):
            arr[i, :] = np.frombuffer(bytes(value), dtype=np.uint8)
        return torch_module.as_tensor(arr, device=device)

    def _validate_tensor(self, name, tensor, dtype, shape):
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} shape must be {tuple(shape)}")

    def step_batch_tensors(self, config: MovementConfig, state_bytes, input_bytes, out_bytes, errors, *, stream=None):
        torch = self.torch
        if state_bytes.ndim != 2:
            raise ValueError("state_bytes must be rank-2")
        lanes = int(state_bytes.shape[0])
        if lanes <= 0:
            raise ValueError("CUDA movement batch requires at least one lane")
        self._validate_tensor("state_bytes", state_bytes, torch.uint8, (lanes, ctypes.sizeof(_CState)))
        self._validate_tensor("input_bytes", input_bytes, torch.uint8, (lanes, ctypes.sizeof(_CInput)))
        self._validate_tensor("out_bytes", out_bytes, torch.uint8, (lanes, ctypes.sizeof(_CResult)))
        self._validate_tensor("errors", errors, torch.int32, (lanes,))
        use_stream = stream or torch.cuda.current_stream()
        status = self.lib.ai2_movement_step_cuda(
            ctypes.byref(_to_c_config(config)),
            state_bytes.data_ptr(),
            input_bytes.data_ptr(),
            out_bytes.data_ptr(),
            errors.data_ptr(),
            lanes,
            use_stream.cuda_stream,
        )
        if status != 0:
            raise RuntimeError(f"movement CUDA launch failed status={status}")
        use_stream.synchronize()
        return errors

    def step_batch(self, config: MovementConfig, cases, *, stream=None):
        if not cases:
            raise ValueError("CUDA movement batch requires at least one lane")
        torch = self.torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = None
        c_states = []
        c_inputs = []
        for case in cases:
            state, decoded_input, grounded, platform_overlap, platform_velocity = case
            inp = _validate(config, state, decoded_input)
            c_states.append(_to_c_state(state))
            c_inputs.append(_to_c_input(inp, grounded, platform_overlap, platform_velocity))
        device = "cuda"
        state_bytes = self._bytes_tensor(torch, c_states, _CState, device)
        input_bytes = self._bytes_tensor(torch, c_inputs, _CInput, device)
        out_bytes = torch.full((len(cases), ctypes.sizeof(_CResult)), 77, dtype=torch.uint8, device=device)
        errors = torch.empty((len(cases),), dtype=torch.int32, device=device)
        self.step_batch_tensors(config, state_bytes, input_bytes, out_bytes, errors, stream=stream)
        error_values = errors.cpu().numpy().tolist()
        if any(error_values):
            raise ValueError(f"movement CUDA lane errors: {error_values}")
        host = out_bytes.cpu().numpy()
        results = []
        for lane in range(len(cases)):
            raw = host[lane].tobytes()
            c_result = _CResult.from_buffer_copy(raw)
            results.append(MovementStepResult(
                _from_c_state(c_result.state),
                bool(c_result.transitioned),
                int(c_result.from_state),
                int(c_result.to_state),
                _events(int(c_result.event_mask)),
            ))
        return results


def load_cuda_reference():
    if not shutil.which("nvcc"):
        raise RuntimeError("nvcc unavailable")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch unavailable") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    build = Path(tempfile.mkdtemp(prefix="ai2-movement-cuda-"))
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    out = build / f"navigation_movement_cuda{suffix}"
    cmd = ["nvcc", "-O2", "--fmad=false", "--shared", "-Xcompiler", "-fPIC",
           str(NATIVE_DIR / "navigation_movement.cu"), "-o", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return CudaMovementReference(ctypes.CDLL(str(out)), torch)
