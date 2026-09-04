#!/usr/bin/env python3
"""Cheap vector-environment transport control for AI_2 benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
ACTION_DIM = 10
RESET_EVERY_STEPS = 1_800


class CheapTransportEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 664, seed: int = 0) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        self._obs = np.zeros((self.obs_dim,), dtype=np.float32)
        self._empty_info: dict[str, Any] = {}
        self._rng = np.random.default_rng(seed)
        self.steps_since_reset = 0
        self.step_count = 0
        self.reset_count = 0
        self.terminal_reset_count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.steps_since_reset = 0
        self.reset_count += 1
        return self._obs, self._empty_info

    def step(self, action):
        self.step_count += 1
        self.steps_since_reset += 1
        truncated = self.steps_since_reset >= RESET_EVERY_STEPS
        if truncated:
            self.terminal_reset_count += 1
        return self._obs, 0.0, False, truncated, self._empty_info


def make_env(obs_dim: int, seed: int):
    def _factory():
        return CheapTransportEnv(obs_dim=obs_dim, seed=seed)

    return _factory


def make_vec_env(n_envs: int, obs_dim: int, frame_stack: int, seed: int):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

    env_fns = [make_env(obs_dim, seed + index) for index in range(n_envs)]
    if n_envs == 1:
        vec_env, vec_env_type = DummyVecEnv(env_fns), "DummyVecEnv"
    else:
        vec_env, vec_env_type = SubprocVecEnv(env_fns, start_method="spawn"), "SubprocVecEnv"
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return vec_env, vec_env_type


def validate_aggregate_counts(n_envs: int, aggregate_steps: int, aggregate_warmup_steps: int) -> None:
    if aggregate_steps <= 0:
        raise ValueError("--steps must be positive")
    if aggregate_warmup_steps < 0:
        raise ValueError("--warmup-steps must be zero or positive")
    if n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if aggregate_steps % n_envs != 0:
        raise ValueError("--steps must be divisible by --n-envs")
    if aggregate_warmup_steps % n_envs != 0:
        raise ValueError("--warmup-steps must be divisible by --n-envs")


def _base_vec_env(env):
    if hasattr(env, "venv"):
        env = env.venv
    return env


def _sum_attr(env, attr_name: str) -> int | None:
    base_env = _base_vec_env(env)
    if hasattr(base_env, "get_attr"):
        try:
            return int(sum(base_env.get_attr(attr_name)))
        except Exception:
            pass
    if not hasattr(base_env, "envs"):
        return None
    return int(sum(getattr(item, attr_name, 0) for item in base_env.envs))


def set_headless_environment() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="ai2-mpl-"))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ai2-cache-"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_transport_repeat(
    *,
    n_envs: int,
    aggregate_steps: int,
    aggregate_warmup_steps: int,
    obs_dim: int,
    frame_stack: int,
    repeat: int,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    validate_aggregate_counts(n_envs, aggregate_steps, aggregate_warmup_steps)
    vector_steps = aggregate_steps // n_envs
    vector_warmup_steps = aggregate_warmup_steps // n_envs
    output_dir.mkdir(parents=True, exist_ok=True)
    env, vec_env_type = make_vec_env(n_envs, obs_dim, frame_stack, seed + repeat)
    action = np.zeros((n_envs, ACTION_DIM), dtype=np.float32)
    stacked_obs_shape = list(env.observation_space.shape)

    try:
        env.reset()
        for _ in range(vector_warmup_steps):
            env.step(action)
        env.reset()
        measured_reset_start = _sum_attr(env, "reset_count") or 0
        measured_terminal_start = _sum_attr(env, "terminal_reset_count") or 0

        start = time.perf_counter()
        for _ in range(vector_steps):
            env.step(action)
        seconds = time.perf_counter() - start
        measured_reset_end = _sum_attr(env, "reset_count") or measured_reset_start
        measured_terminal_end = (
            _sum_attr(env, "terminal_reset_count") or measured_terminal_start
        )
    finally:
        env.close()

    result = {
        "label": "cheap_environment_transport_control",
        "repeat": repeat,
        "seed": seed + repeat,
        "n_envs": n_envs,
        "vec_env_type": vec_env_type,
        "obs_dim": obs_dim,
        "frame_stack": frame_stack,
        "base_obs_shape": [obs_dim],
        "stacked_obs_shape": stacked_obs_shape,
        "action_dim": ACTION_DIM,
        "aggregate_transitions": aggregate_steps,
        "aggregate_warmup_transitions": aggregate_warmup_steps,
        "vector_step_calls": vector_steps,
        "vector_warmup_step_calls": vector_warmup_steps,
        "vector_reset_calls": 2,
        "cheap_env_total_reset_count": measured_reset_end,
        "cheap_env_total_terminal_resets": measured_terminal_end,
        "cheap_env_measured_reset_count": measured_reset_end - measured_reset_start,
        "cheap_env_measured_terminal_resets": measured_terminal_end - measured_terminal_start,
        "seconds": seconds,
        "steps_per_second": aggregate_steps / seconds if seconds else 0.0,
        "note": "Constant preallocated observations and empty infos; no graphics, physics, model, or real game logic, so this is not an exact real-env overhead fraction.",
    }
    (output_dir / f"transport_repeat{repeat}.json").write_text(
        json.dumps({"result": result}, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def aggregate_results(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [item["steps_per_second"] for item in repeats]
    return {
        "label": "cheap_environment_transport_control",
        "repeats": repeats,
        "steps_per_second_median": statistics.median(rates),
        "steps_per_second_min": min(rates),
        "steps_per_second_max": max(rates),
        "note": "Transport-only control with constant observations and empty infos; no actual physics/game/model work.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cheap SB3 vector-environment transport without game/model work."
    )
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, required=True, help="Aggregate measured transitions")
    parser.add_argument(
        "--warmup-steps", type=int, default=0, help="Aggregate warmup transitions excluded from timing"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--obs-dim", type=int, default=664)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.obs_dim <= 0:
        parser.error("--obs-dim must be positive")
    if args.frame_stack <= 0:
        parser.error("--frame-stack must be positive")
    try:
        validate_aggregate_counts(args.n_envs, args.steps, args.warmup_steps)
    except ValueError as exc:
        parser.error(str(exc))
    args.output = args.output.resolve()
    return args


def main() -> int:
    set_headless_environment()
    args = parse_args()
    seed_everything(args.seed)
    results = [
        run_transport_repeat(
            n_envs=args.n_envs,
            aggregate_steps=args.steps,
            aggregate_warmup_steps=args.warmup_steps,
            obs_dim=args.obs_dim,
            frame_stack=args.frame_stack,
            repeat=repeat,
            output_dir=args.output,
            seed=args.seed,
        )
        for repeat in range(args.repeats)
    ]
    summary = aggregate_results(results)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved cheap transport control: {args.output / 'summary.json'}")
    print(
        f"cheap_environment_transport_control: "
        f"{summary['steps_per_second_median']:.1f} aggregate transitions/s, "
        f"{args.n_envs} envs, {args.repeats} repeats"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
