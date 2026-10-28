#!/usr/bin/env python3
"""Native navigation-rule reducer for frozen Phase 0 wrapper semantics.

This module owns only the post-physics wrapper bookkeeping/reward transition
from ``FrozenOpponentWrapper.step``.  It deliberately does not own physics,
target RNG, observation packing, reset spawning, or PPO/training behavior.
"""

from __future__ import annotations

from ai2.paths import REPO_ROOT as PROJECT_ROOT

import ctypes
from dataclasses import dataclass, replace
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

import numpy as np


ROOT = PROJECT_ROOT
NATIVE_DIR = ROOT / "native"
FROZEN_WRAPPER_SOURCE = Path(
    "/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/"
    "baselines/navigation-repair-20260905/payload/variants/corrected/train.py"
)
FROZEN_WRAPPER_LINE_RANGE = "171-525"
TOLERANCE = 1e-9
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class NavigationConfig:
    completion_mode: str
    control_mode: str
    max_targets: int
    time_limit_steps: int
    settle_steps: int
    post_goal_required_steps: int
    settle_speed: float
    fall_penalty: float


@dataclass(frozen=True)
class NavigationState:
    target_x: float
    target_y: float
    target_valid: bool
    last_distance: float
    last_distance_valid: bool
    episode_steps: int
    targets_this_episode: int
    hold_steps: int
    post_goal_steps: int
    goal_reached_step: int
    goal_reached_step_valid: bool
    last_stocks: int
    fell_this_episode: bool
    target_touched: bool
    targets_reached: int
    total_targets: int
    falls: int
    steps_since_log: int
    interval_targets_reached: int
    interval_falls: int
    interval_episodes: int


@dataclass(frozen=True)
class NavigationPostPhysics:
    player_present: bool
    player_x: float
    player_y: float
    velocity_x: float
    velocity_y: float
    support_velocity_x: float
    support_velocity_y: float
    support_velocity_valid: bool
    grounded: bool
    stocks: int
    raw_reward: float
    raw_terminated: bool
    raw_truncated: bool
    next_target_x: float
    next_target_y: float
    next_target_valid: bool


@dataclass(frozen=True)
class NavigationStepResult:
    state: NavigationState
    reward: float
    nav_reward: float
    terminated: bool
    truncated: bool
    underlying_done: bool
    touched: bool
    reached: bool
    fall: bool
    handoff: bool
    success: bool
    horizon: bool


def replace_state(state: NavigationState, **updates: Any) -> NavigationState:
    return replace(state, **updates)


def replace_post(post: NavigationPostPhysics, **updates: Any) -> NavigationPostPhysics:
    return replace(post, **updates)


def mask_movement_action(action: Sequence[float]) -> np.ndarray:
    masked = np.array(action, copy=True)
    if masked.shape != (10,):
        raise ValueError("action must have shape (10,)")
    if not np.all(np.isfinite(masked)):
        raise ValueError("action must be finite")
    masked[[5, 7, 8, 9]] = 0
    return masked


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _bool01(value: Any) -> bool:
    return value is True or value is False or value == 0 or value == 1


def _int32(value: Any) -> bool:
    try:
        as_int = int(value)
    except Exception:
        return False
    return as_int == value and INT32_MIN <= as_int <= INT32_MAX


def _validate_config(config: NavigationConfig) -> None:
    if config.completion_mode not in ("horizon", "sequence"):
        raise ValueError("completion_mode must be horizon or sequence")
    if config.control_mode not in ("full", "movement"):
        raise ValueError("control_mode must be full or movement")
    int_fields = [
        config.max_targets,
        config.time_limit_steps,
        config.settle_steps,
        config.post_goal_required_steps,
    ]
    if any(not _int32(value) for value in int_fields):
        raise ValueError("config integer fields must fit int32")
    if int(config.max_targets) < 1:
        raise ValueError("max_targets must be positive")
    if config.completion_mode == "sequence" and config.max_targets < 2:
        raise ValueError("sequence completion requires at least two targets")
    if int(config.time_limit_steps) <= 0 or int(config.settle_steps) <= 0 or int(config.post_goal_required_steps) <= 0:
        raise ValueError("time/settle/post-goal windows must be positive")
    if int(config.settle_steps) > INT32_MAX - int(config.post_goal_required_steps):
        raise ValueError("settle/post-goal window sum must fit int32")
    if not _finite(config.settle_speed, config.fall_penalty) or config.settle_speed <= 0.0:
        raise ValueError("settle_speed and fall_penalty must be finite")


def _validate_state(state: NavigationState) -> None:
    floats = [state.target_x, state.target_y, state.last_distance]
    if not _finite(*floats):
        raise ValueError("state floating fields must be finite")
    ints = [
        state.episode_steps,
        state.targets_this_episode,
        state.hold_steps,
        state.post_goal_steps,
        state.goal_reached_step,
        state.last_stocks,
        state.targets_reached,
        state.total_targets,
        state.falls,
        state.steps_since_log,
        state.interval_targets_reached,
        state.interval_falls,
        state.interval_episodes,
    ]
    if any(not _int32(value) for value in ints):
        raise ValueError("state integer counters must fit int32")
    if any(int(value) < 0 for value in ints):
        raise ValueError("state counters must be nonnegative")
    bools = [
        state.target_valid,
        state.last_distance_valid,
        state.goal_reached_step_valid,
        state.fell_this_episode,
        state.target_touched,
    ]
    if any(not _bool01(value) for value in bools):
        raise ValueError("state boolean flags must be 0 or 1")


def _validate_post(post: NavigationPostPhysics) -> None:
    if not _finite(
        post.player_x,
        post.player_y,
        post.velocity_x,
        post.velocity_y,
        post.support_velocity_x,
        post.support_velocity_y,
        post.raw_reward,
        post.next_target_x,
        post.next_target_y,
    ):
        raise ValueError("post-physics floating fields must be finite")
    if not _int32(post.stocks):
        raise ValueError("stocks must fit int32")
    if int(post.stocks) < 0:
        raise ValueError("stocks must be nonnegative")
    bools = [
        post.player_present,
        post.support_velocity_valid,
        post.grounded,
        post.raw_terminated,
        post.raw_truncated,
        post.next_target_valid,
    ]
    if any(not _bool01(value) for value in bools):
        raise ValueError("post boolean flags must be 0 or 1")


def _validate_pre_arithmetic_bounds(state: NavigationState) -> None:
    incremented_counters = [
        state.episode_steps,
        state.steps_since_log,
        state.hold_steps,
        state.falls,
        state.interval_falls,
        state.targets_reached,
        state.interval_targets_reached,
        state.targets_this_episode,
        state.total_targets,
    ]
    if any(int(value) >= INT32_MAX for value in incremented_counters):
        raise ValueError("pre-arithmetic state counters must be less than int32 max")


def _distance(player_x: float, player_y: float, target_x: float, target_y: float, target_valid: bool) -> float:
    if not target_valid:
        return 0.0
    return math.sqrt((player_x - target_x) ** 2 + (player_y - target_y) ** 2)


def _is_settled(config: NavigationConfig, post: NavigationPostPhysics, distance: float) -> bool:
    if distance >= 1.0 or not post.grounded:
        return False
    support_x = post.support_velocity_x if post.support_velocity_valid else 0.0
    support_y = post.support_velocity_y if post.support_velocity_valid else 0.0
    relative_speed = math.hypot(post.velocity_x - support_x, post.velocity_y - support_y)
    return bool(distance < 1.0 and relative_speed <= config.settle_speed)


def step_navigation_scalar(
    config: NavigationConfig,
    state: NavigationState,
    post: NavigationPostPhysics,
) -> NavigationStepResult:
    _validate_config(config)
    _validate_state(state)
    _validate_post(post)
    _validate_pre_arithmetic_bounds(state)

    early_state = replace_state(
        state,
        episode_steps=state.episode_steps + 1,
        steps_since_log=state.steps_since_log + 1,
    )

    if not post.player_present:
        return NavigationStepResult(
            state=early_state,
            reward=post.raw_reward,
            nav_reward=0.0,
            terminated=post.raw_terminated,
            truncated=post.raw_truncated,
            underlying_done=bool(post.raw_terminated or post.raw_truncated),
            touched=False,
            reached=False,
            fall=False,
            handoff=False,
            success=False,
            horizon=False,
        )

    target_x = state.target_x
    target_y = state.target_y
    target_valid = state.target_valid
    last_distance = state.last_distance
    last_distance_valid = state.last_distance_valid
    episode_steps = early_state.episode_steps
    steps_since_log = early_state.steps_since_log
    hold_steps = state.hold_steps
    post_goal_steps = state.post_goal_steps
    goal_reached_step = state.goal_reached_step
    goal_reached_step_valid = state.goal_reached_step_valid
    targets_this_episode = state.targets_this_episode
    target_touched = state.target_touched
    targets_reached = state.targets_reached
    total_targets = state.total_targets
    falls = state.falls
    interval_targets_reached = state.interval_targets_reached
    interval_falls = state.interval_falls
    fell_this_episode = state.fell_this_episode

    nav_reward = 0.0
    terminated = bool(post.raw_terminated)
    truncated = bool(post.raw_truncated)
    underlying_done = bool(terminated or truncated)
    touched = False
    reached = False
    fall = False
    handoff = False

    current_dist = _distance(post.player_x, post.player_y, target_x, target_y, target_valid)
    if last_distance_valid:
        nav_reward += (last_distance - current_dist) * 2.0

    current_stocks = int(post.stocks)
    if current_stocks < state.last_stocks:
        nav_reward += config.fall_penalty
        falls += 1
        interval_falls += 1
        fell_this_episode = True
        terminated = True
        fall = True

    active = not (underlying_done or fell_this_episode)

    if active and current_dist < 1.0 and not target_touched:
        target_touched = True
        touched = True

    if active and _is_settled(config, post, current_dist):
        hold_limit = config.settle_steps
        if config.completion_mode == "sequence" and targets_this_episode >= config.max_targets:
            hold_limit += config.post_goal_required_steps
        hold_steps = min(hold_steps + 1, hold_limit)
    else:
        hold_steps = 0

    if active and hold_steps >= config.settle_steps and targets_this_episode < config.max_targets:
        nav_reward += 5.0
        targets_reached += 1
        interval_targets_reached += 1
        reached = True
        targets_this_episode += 1
        if targets_this_episode < config.max_targets:
            if not post.next_target_valid:
                raise ValueError("next target is required for navigation handoff")
            target_x = post.next_target_x
            target_y = post.next_target_y
            target_valid = True
            total_targets += 1
            hold_steps = 0
            target_touched = False
            handoff = True
        else:
            goal_reached_step = episode_steps
            goal_reached_step_valid = True

    if goal_reached_step_valid:
        if config.completion_mode == "sequence":
            post_goal_steps = max(0, hold_steps - config.settle_steps)
        else:
            post_goal_steps = episode_steps - goal_reached_step

    last_distance = _distance(post.player_x, post.player_y, target_x, target_y, target_valid)
    last_distance_valid = True

    x_pos = abs(post.player_x)
    if x_pos > 8.0:
        edge_factor = (x_pos - 8.0) / 4.0
        nav_reward -= 0.1 * min(edge_factor, 1.0)
    if post.player_y > 3.0:
        y_edge_factor = (post.player_y - 3.0) / 3.0
        nav_reward -= 0.1 * min(y_edge_factor, 1.0)

    horizon = bool(episode_steps >= config.time_limit_steps and not underlying_done)
    completed = bool(
        active
        and targets_this_episode >= config.max_targets
        and hold_steps >= config.settle_steps
        and post_goal_steps >= config.post_goal_required_steps
    )
    success = bool(completed and ((not horizon) if config.completion_mode == "sequence" else horizon))
    if success and config.completion_mode == "sequence":
        terminated = True
    if horizon:
        truncated = True

    next_state = NavigationState(
        target_x=target_x,
        target_y=target_y,
        target_valid=target_valid,
        last_distance=last_distance,
        last_distance_valid=last_distance_valid,
        episode_steps=episode_steps,
        targets_this_episode=targets_this_episode,
        hold_steps=hold_steps,
        post_goal_steps=post_goal_steps,
        goal_reached_step=goal_reached_step,
        goal_reached_step_valid=goal_reached_step_valid,
        last_stocks=current_stocks,
        fell_this_episode=fell_this_episode,
        target_touched=target_touched,
        targets_reached=targets_reached,
        total_targets=total_targets,
        falls=falls,
        steps_since_log=steps_since_log,
        interval_targets_reached=interval_targets_reached,
        interval_falls=interval_falls,
        interval_episodes=state.interval_episodes,
    )
    return NavigationStepResult(
        state=next_state,
        reward=post.raw_reward + nav_reward,
        nav_reward=nav_reward,
        terminated=terminated,
        truncated=truncated,
        underlying_done=underlying_done,
        touched=touched,
        reached=reached,
        fall=fall,
        handoff=handoff,
        success=success,
        horizon=horizon,
    )


def step_navigation_batch_python(
    config: NavigationConfig,
    states: Sequence[NavigationState],
    posts: Sequence[NavigationPostPhysics],
) -> list[NavigationStepResult]:
    if len(states) != len(posts):
        raise ValueError("states and posts must have the same length")
    return [step_navigation_scalar(config, state, post) for state, post in zip(states, posts)]


def _shared_library_suffix() -> str:
    return ".dylib" if platform.system() == "Darwin" else ".so"


def can_compile_cpu_reference() -> bool:
    return shutil.which(os.environ.get("CC") or "cc") is not None


class CConfig(ctypes.Structure):
    _fields_ = [
        ("completion_mode", ctypes.c_int32),
        ("control_mode", ctypes.c_int32),
        ("max_targets", ctypes.c_int32),
        ("time_limit_steps", ctypes.c_int32),
        ("settle_steps", ctypes.c_int32),
        ("post_goal_required_steps", ctypes.c_int32),
        ("settle_speed", ctypes.c_double),
        ("fall_penalty", ctypes.c_double),
    ]


class CState(ctypes.Structure):
    _fields_ = [
        ("target_x", ctypes.c_double),
        ("target_y", ctypes.c_double),
        ("target_valid", ctypes.c_uint8),
        ("last_distance", ctypes.c_double),
        ("last_distance_valid", ctypes.c_uint8),
        ("episode_steps", ctypes.c_int32),
        ("targets_this_episode", ctypes.c_int32),
        ("hold_steps", ctypes.c_int32),
        ("post_goal_steps", ctypes.c_int32),
        ("goal_reached_step", ctypes.c_int32),
        ("goal_reached_step_valid", ctypes.c_uint8),
        ("last_stocks", ctypes.c_int32),
        ("fell_this_episode", ctypes.c_uint8),
        ("target_touched", ctypes.c_uint8),
        ("targets_reached", ctypes.c_int32),
        ("total_targets", ctypes.c_int32),
        ("falls", ctypes.c_int32),
        ("steps_since_log", ctypes.c_int32),
        ("interval_targets_reached", ctypes.c_int32),
        ("interval_falls", ctypes.c_int32),
        ("interval_episodes", ctypes.c_int32),
    ]


class CPost(ctypes.Structure):
    _fields_ = [
        ("player_present", ctypes.c_uint8),
        ("player_x", ctypes.c_double),
        ("player_y", ctypes.c_double),
        ("velocity_x", ctypes.c_double),
        ("velocity_y", ctypes.c_double),
        ("support_velocity_x", ctypes.c_double),
        ("support_velocity_y", ctypes.c_double),
        ("support_velocity_valid", ctypes.c_uint8),
        ("grounded", ctypes.c_uint8),
        ("stocks", ctypes.c_int32),
        ("raw_reward", ctypes.c_double),
        ("raw_terminated", ctypes.c_uint8),
        ("raw_truncated", ctypes.c_uint8),
        ("next_target_x", ctypes.c_double),
        ("next_target_y", ctypes.c_double),
        ("next_target_valid", ctypes.c_uint8),
    ]


class CResult(ctypes.Structure):
    _fields_ = [
        ("state", CState),
        ("reward", ctypes.c_double),
        ("nav_reward", ctypes.c_double),
        ("terminated", ctypes.c_uint8),
        ("truncated", ctypes.c_uint8),
        ("underlying_done", ctypes.c_uint8),
        ("touched", ctypes.c_uint8),
        ("reached", ctypes.c_uint8),
        ("fall", ctypes.c_uint8),
        ("handoff", ctypes.c_uint8),
        ("success", ctypes.c_uint8),
        ("horizon", ctypes.c_uint8),
    ]


def _config_to_c(config: NavigationConfig) -> CConfig:
    return CConfig(
        1 if config.completion_mode == "sequence" else 0,
        1 if config.control_mode == "movement" else 0,
        int(config.max_targets),
        int(config.time_limit_steps),
        int(config.settle_steps),
        int(config.post_goal_required_steps),
        float(config.settle_speed),
        float(config.fall_penalty),
    )


def _state_to_c(state: NavigationState) -> CState:
    return CState(
        state.target_x,
        state.target_y,
        int(state.target_valid),
        state.last_distance,
        int(state.last_distance_valid),
        state.episode_steps,
        state.targets_this_episode,
        state.hold_steps,
        state.post_goal_steps,
        state.goal_reached_step,
        int(state.goal_reached_step_valid),
        state.last_stocks,
        int(state.fell_this_episode),
        int(state.target_touched),
        state.targets_reached,
        state.total_targets,
        state.falls,
        state.steps_since_log,
        state.interval_targets_reached,
        state.interval_falls,
        state.interval_episodes,
    )


def _post_to_c(post: NavigationPostPhysics) -> CPost:
    return CPost(
        int(post.player_present),
        post.player_x,
        post.player_y,
        post.velocity_x,
        post.velocity_y,
        post.support_velocity_x,
        post.support_velocity_y,
        int(post.support_velocity_valid),
        int(post.grounded),
        int(post.stocks),
        post.raw_reward,
        int(post.raw_terminated),
        int(post.raw_truncated),
        post.next_target_x,
        post.next_target_y,
        int(post.next_target_valid),
    )


def _state_from_c(state: CState) -> NavigationState:
    return NavigationState(
        target_x=float(state.target_x),
        target_y=float(state.target_y),
        target_valid=bool(state.target_valid),
        last_distance=float(state.last_distance),
        last_distance_valid=bool(state.last_distance_valid),
        episode_steps=int(state.episode_steps),
        targets_this_episode=int(state.targets_this_episode),
        hold_steps=int(state.hold_steps),
        post_goal_steps=int(state.post_goal_steps),
        goal_reached_step=int(state.goal_reached_step),
        goal_reached_step_valid=bool(state.goal_reached_step_valid),
        last_stocks=int(state.last_stocks),
        fell_this_episode=bool(state.fell_this_episode),
        target_touched=bool(state.target_touched),
        targets_reached=int(state.targets_reached),
        total_targets=int(state.total_targets),
        falls=int(state.falls),
        steps_since_log=int(state.steps_since_log),
        interval_targets_reached=int(state.interval_targets_reached),
        interval_falls=int(state.interval_falls),
        interval_episodes=int(state.interval_episodes),
    )


def _result_from_c(result: CResult) -> NavigationStepResult:
    return NavigationStepResult(
        state=_state_from_c(result.state),
        reward=float(result.reward),
        nav_reward=float(result.nav_reward),
        terminated=bool(result.terminated),
        truncated=bool(result.truncated),
        underlying_done=bool(result.underlying_done),
        touched=bool(result.touched),
        reached=bool(result.reached),
        fall=bool(result.fall),
        handoff=bool(result.handoff),
        success=bool(result.success),
        horizon=bool(result.horizon),
    )


def compile_cpu_reference(build_dir: Path) -> dict[str, Any]:
    compiler = os.environ.get("CC") or "cc"
    if shutil.which(compiler) is None:
        raise RuntimeError("no C compiler available")
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / f"libnavigation_contract_cpu{_shared_library_suffix()}"
    command = [
        compiler,
        "-O3",
        "-std=c99",
        "-ffp-contract=off",
        "-fPIC",
        str(NATIVE_DIR / "navigation_contract.c"),
        "-o",
        str(output),
    ]
    if platform.system() == "Darwin":
        command.insert(3, "-dynamiclib")
    else:
        command.insert(3, "-shared")
        command.append("-lm")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"CPU navigation contract compilation failed:\n{completed.stdout}")
    return {"path": str(output), "command": command, "compiler_output": completed.stdout}


class CpuNavigationContract:
    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.lib = ctypes.CDLL(str(library_path))
        self._step = self.lib.navigation_contract_step_one
        self._step.argtypes = [
            ctypes.POINTER(CConfig),
            ctypes.POINTER(CState),
            ctypes.POINTER(CPost),
            ctypes.POINTER(CResult),
        ]
        self._step.restype = ctypes.c_int

    def step(self, config: NavigationConfig, state: NavigationState, post: NavigationPostPhysics) -> NavigationStepResult:
        out = CResult()
        status = self._step(
            ctypes.byref(_config_to_c(config)),
            ctypes.byref(_state_to_c(state)),
            ctypes.byref(_post_to_c(post)),
            ctypes.byref(out),
        )
        if status == 2:
            raise ValueError("next target is required for navigation handoff")
        if status == 3:
            raise ValueError("invalid navigation input")
        if status != 0:
            raise RuntimeError(f"navigation_contract_step_one failed with status {status}")
        return _result_from_c(out)


def load_cpu_reference(build_dir: Path) -> CpuNavigationContract:
    result = compile_cpu_reference(build_dir)
    return CpuNavigationContract(Path(result["path"]))


def _nvcc() -> str | None:
    return shutil.which("nvcc")


def can_run_cuda_contract() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available() and _nvcc() is not None)


def tensors_from_cases(
    config: NavigationConfig,
    states: Sequence[NavigationState],
    posts: Sequence[NavigationPostPhysics],
    *,
    torch: Any,
    device: str,
) -> dict[str, Any]:
    _validate_config(config)
    if len(states) != len(posts):
        raise ValueError("states and posts must have the same length")
    for state in states:
        _validate_state(state)
    for post in posts:
        _validate_post(post)
    lanes = len(states)
    state_xy = torch.tensor([[s.target_x, s.target_y] for s in states], dtype=torch.float64, device=device)
    last_distance = torch.tensor([s.last_distance for s in states], dtype=torch.float64, device=device)
    player_xy = torch.tensor([[p.player_x, p.player_y] for p in posts], dtype=torch.float64, device=device)
    velocity_xy = torch.tensor([[p.velocity_x, p.velocity_y] for p in posts], dtype=torch.float64, device=device)
    support_velocity_xy = torch.tensor(
        [[p.support_velocity_x, p.support_velocity_y] for p in posts], dtype=torch.float64, device=device
    )
    raw_reward = torch.tensor([p.raw_reward for p in posts], dtype=torch.float64, device=device)
    next_target_xy = torch.tensor([[p.next_target_x, p.next_target_y] for p in posts], dtype=torch.float64, device=device)
    state_i32 = torch.tensor(
        [
            [
                s.episode_steps,
                s.targets_this_episode,
                s.hold_steps,
                s.post_goal_steps,
                s.goal_reached_step,
                s.last_stocks,
                s.targets_reached,
                s.total_targets,
                s.falls,
                s.steps_since_log,
                s.interval_targets_reached,
                s.interval_falls,
                s.interval_episodes,
            ]
            for s in states
        ],
        dtype=torch.int32,
        device=device,
    )
    flags = torch.tensor(
        [
            [
                int(s.target_valid),
                int(s.last_distance_valid),
                int(s.goal_reached_step_valid),
                int(s.fell_this_episode),
                int(s.target_touched),
                int(p.player_present),
                int(p.support_velocity_valid),
                int(p.grounded),
                int(p.raw_terminated),
                int(p.raw_truncated),
                int(p.next_target_valid),
            ]
            for s, p in zip(states, posts)
        ],
        dtype=torch.uint8,
        device=device,
    )
    post_i32 = torch.tensor([[p.stocks] for p in posts], dtype=torch.int32, device=device)
    return {
        "lanes": lanes,
        "config": config,
        "state_xy": state_xy,
        "last_distance": last_distance,
        "player_xy": player_xy,
        "velocity_xy": velocity_xy,
        "support_velocity_xy": support_velocity_xy,
        "raw_reward": raw_reward,
        "next_target_xy": next_target_xy,
        "state_i32": state_i32,
        "flags": flags,
        "post_i32": post_i32,
    }


class CudaNavigationContract:
    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.lib = ctypes.CDLL(str(library_path))
        self._step = self.lib.navigation_contract_step_cuda
        self._step.restype = ctypes.c_int

    def step(self, tensors: dict[str, Any]) -> dict[str, Any]:
        import torch

        required_float64_2 = ("state_xy", "player_xy", "velocity_xy", "support_velocity_xy", "next_target_xy")
        required_float64_1 = ("last_distance", "raw_reward")
        required_i32 = ("state_i32", "post_i32")
        required_u8 = ("flags",)
        all_names = required_float64_2 + required_float64_1 + required_i32 + required_u8
        first_device = None
        for name in all_names:
            tensor = tensors[name]
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be CUDA; no CPU fallback")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            first_device = str(tensor.device) if first_device is None else first_device
            if str(tensor.device) != first_device:
                raise ValueError("all tensors must be on one CUDA device")
        for name in required_float64_2:
            if tensors[name].dtype != torch.float64 or tensors[name].ndim != 2 or tensors[name].shape[1] != 2:
                raise ValueError(f"{name} must be CUDA float64 [lanes,2]")
        for name in required_float64_1:
            if tensors[name].dtype != torch.float64 or tensors[name].ndim != 1:
                raise ValueError(f"{name} must be CUDA float64 [lanes]")
        for name in required_i32:
            if tensors[name].dtype != torch.int32 or tensors[name].ndim != 2:
                raise ValueError(f"{name} must be CUDA int32 matrix")
        for name in required_u8:
            if tensors[name].dtype != torch.uint8 or tensors[name].ndim != 2:
                raise ValueError(f"{name} must be CUDA uint8 matrix")
        if int(tensors["state_i32"].shape[1]) != 13:
            raise ValueError("state_i32 must have width 13")
        if int(tensors["post_i32"].shape[1]) != 1:
            raise ValueError("post_i32 must have width 1")
        if int(tensors["flags"].shape[1]) != 11:
            raise ValueError("flags must have width 11")
        lanes = int(tensors["lanes"])
        if lanes <= 0:
            raise ValueError("lanes must be positive")
        for name in all_names:
            if int(tensors[name].shape[0]) != lanes:
                raise ValueError(f"{name} lane count mismatch")
        config = _config_to_c(tensors["config"])
        out_state_xy = torch.empty_like(tensors["state_xy"])
        out_last_distance = torch.empty_like(tensors["last_distance"])
        out_state_i32 = torch.empty_like(tensors["state_i32"])
        out_flags = torch.empty((lanes, 14), dtype=torch.uint8, device=tensors["flags"].device)
        out_reward = torch.empty_like(tensors["raw_reward"])
        out_nav_reward = torch.empty_like(tensors["raw_reward"])
        out_errors = torch.empty((lanes,), dtype=torch.int32, device=tensors["flags"].device)
        with torch.cuda.device(tensors["state_xy"].device):
            stream = ctypes.c_void_p(int(torch.cuda.current_stream(tensors["state_xy"].device).cuda_stream))
            self._step.argtypes = [
                ctypes.POINTER(CConfig),
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            status = self._step(
                ctypes.byref(config),
                ctypes.c_void_p(tensors["state_xy"].data_ptr()),
                ctypes.c_void_p(tensors["last_distance"].data_ptr()),
                ctypes.c_void_p(tensors["state_i32"].data_ptr()),
                ctypes.c_void_p(tensors["flags"].data_ptr()),
                ctypes.c_void_p(tensors["player_xy"].data_ptr()),
                ctypes.c_void_p(tensors["velocity_xy"].data_ptr()),
                ctypes.c_void_p(tensors["support_velocity_xy"].data_ptr()),
                ctypes.c_void_p(tensors["raw_reward"].data_ptr()),
                ctypes.c_void_p(tensors["post_i32"].data_ptr()),
                ctypes.c_void_p(tensors["next_target_xy"].data_ptr()),
                ctypes.c_void_p(out_state_xy.data_ptr()),
                ctypes.c_void_p(out_last_distance.data_ptr()),
                ctypes.c_void_p(out_state_i32.data_ptr()),
                ctypes.c_void_p(out_flags.data_ptr()),
                ctypes.c_void_p(out_reward.data_ptr()),
                ctypes.c_void_p(out_nav_reward.data_ptr()),
                ctypes.c_void_p(out_errors.data_ptr()),
                ctypes.c_int(lanes),
                stream,
            )
        if status != 0:
            raise RuntimeError(f"navigation_contract_step_cuda failed with status {status}")
        max_error = int(out_errors.max().item())
        if max_error != 0:
            error_lanes = torch.nonzero(out_errors, as_tuple=False).flatten().detach().cpu().tolist()
            raise RuntimeError(
                f"navigation_contract_step_cuda per-lane errors: code={max_error} lanes={error_lanes[:10]} "
                "(validation sync path, not resident performance path)"
            )
        return {
            "state_xy": out_state_xy,
            "last_distance": out_last_distance,
            "state_i32": out_state_i32,
            "flags": out_flags,
            "reward": out_reward,
            "nav_reward": out_nav_reward,
            "errors": out_errors,
            "episode_steps": out_state_i32[:, 0],
        }


def unpack_cuda_results(result: dict[str, Any]) -> list[NavigationStepResult]:
    state_xy = result["state_xy"].detach().cpu().numpy()
    last_distance = result["last_distance"].detach().cpu().numpy()
    state_i32 = result["state_i32"].detach().cpu().numpy()
    flags = result["flags"].detach().cpu().numpy()
    rewards = result["reward"].detach().cpu().numpy()
    nav_rewards = result["nav_reward"].detach().cpu().numpy()
    unpacked: list[NavigationStepResult] = []
    for i in range(state_xy.shape[0]):
        state = NavigationState(
            target_x=float(state_xy[i, 0]),
            target_y=float(state_xy[i, 1]),
            target_valid=bool(flags[i, 0]),
            last_distance=float(last_distance[i]),
            last_distance_valid=bool(flags[i, 1]),
            episode_steps=int(state_i32[i, 0]),
            targets_this_episode=int(state_i32[i, 1]),
            hold_steps=int(state_i32[i, 2]),
            post_goal_steps=int(state_i32[i, 3]),
            goal_reached_step=int(state_i32[i, 4]),
            goal_reached_step_valid=bool(flags[i, 2]),
            last_stocks=int(state_i32[i, 5]),
            fell_this_episode=bool(flags[i, 3]),
            target_touched=bool(flags[i, 4]),
            targets_reached=int(state_i32[i, 6]),
            total_targets=int(state_i32[i, 7]),
            falls=int(state_i32[i, 8]),
            steps_since_log=int(state_i32[i, 9]),
            interval_targets_reached=int(state_i32[i, 10]),
            interval_falls=int(state_i32[i, 11]),
            interval_episodes=int(state_i32[i, 12]),
        )
        unpacked.append(
            NavigationStepResult(
                state=state,
                reward=float(rewards[i]),
                nav_reward=float(nav_rewards[i]),
                terminated=bool(flags[i, 5]),
                truncated=bool(flags[i, 6]),
                underlying_done=bool(flags[i, 7]),
                touched=bool(flags[i, 8]),
                reached=bool(flags[i, 9]),
                fall=bool(flags[i, 10]),
                handoff=bool(flags[i, 11]),
                success=bool(flags[i, 12]),
                horizon=bool(flags[i, 13]),
            )
        )
    return unpacked


def compile_cuda_contract(build_dir: Path) -> dict[str, Any]:
    nvcc = _nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc is not available")
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / "libnavigation_contract_cuda.so"
    command = [
        nvcc,
        "-O3",
        "--fmad=false",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        str(NATIVE_DIR / "navigation_contract.cu"),
        "-o",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"CUDA navigation contract compilation failed:\n{completed.stdout}")
    return {"path": str(output), "command": command, "compiler_output": completed.stdout}


def load_cuda_contract(build_dir: Path) -> CudaNavigationContract:
    result = compile_cuda_contract(build_dir)
    return CudaNavigationContract(Path(result["path"]))
