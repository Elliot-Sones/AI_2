"""Diagnostic-only CPU method timing inside each environment process.

Nested spans have inclusive and exclusive times. Worker totals are CPU-side
elapsed work, NOT additive to the parent process's environment-wait time.
"""

from collections import defaultdict
from functools import partial, wraps
from pathlib import Path
from time import perf_counter

import gymnasium as gym


class Spans:
    def __init__(self):
        self.enabled = False
        self.stack = []
        self.rows = defaultdict(lambda: [0, 0.0, 0.0])

    def wrap(self, function, label):
        @wraps(function)
        def timed(*args, **kwargs):
            if not self.enabled:
                return function(*args, **kwargs)
            frame = [perf_counter(), 0.0]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = perf_counter() - frame[0]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += elapsed
                row = self.rows[label]
                row[0] += 1
                row[1] += elapsed
                row[2] += elapsed - frame[1]
        return timed

    def patch(self, target, method, label):
        setattr(target, method, self.wrap(getattr(target, method), label))


class WorkerTimingWrapper(gym.Wrapper):
    def __init__(self, env, spans, worker_index):
        super().__init__(env)
        self.spans = spans
        self.worker_index = worker_index
        self.step_events = []
        self.step = spans.wrap(self.step, "worker.env_step")
        self.reset = spans.wrap(self.reset, "worker.env_reset")

    def start_worker_measurement(self):
        self.spans.rows.clear()
        self.step_events.clear()
        self.spans.enabled = True
        return self.worker_index

    def finish_worker_measurement(self):
        self.spans.enabled = False
        return {
            "worker_index": self.worker_index,
            "step_events": self.step_events,
            "timing_note": "Inclusive rows nest. Exclusive rows omit nested spans. Worker elapsed work overlaps parent env wait and other workers.",
            "components": {
                label: {"count": row[0], "inclusive_seconds": row[1], "exclusive_seconds": row[2]}
                for label, row in sorted(self.spans.rows.items())
            },
        }

    def step(self, action):
        if not self.spans.enabled:
            return self.env.step(action)
        start = perf_counter()
        result = self.env.step(action)
        end = perf_counter()
        self.step_events.append([start, end])
        return result


def _make_worker(config_path, seed, torch_threads, worker_index):
    import benchmark_training as bt
    bt.set_headless_environment()
    bt.add_repo_imports()
    bt.seed_everything(seed + worker_index)
    bt.configure_torch("cpu", torch_threads)
    import train
    import pymunk
    import environment.environment as game
    import environment.agent as agents
    from stable_baselines3.common.monitor import Monitor

    params = bt.load_config(Path(config_path))
    phase = train.PhaseConfig(params, params["curriculum"]["phase"])
    env = train.make_nav_env(phase, train.get_resolution(params.get("environment_settings", {}).get("resolution", "LOW")), params)
    spans = Spans()
    for target, method, label in [
        (game.WarehouseBrawl, "step", "game.step"),
        (game.WarehouseBrawl, "reset", "game.reset"),
        (game.WarehouseBrawl, "observe", "game.observation"),
        (game.Player, "get_obs", "game.player_observation"),
        (game.Player, "physics_process", "game.player_physics_state"),
        (game.Player, "process", "game.player_input"),
        (game.Player, "is_on_floor", "game.floor_check"),
        (pymunk.Space, "step", "game.native_physics_with_callbacks"),
        (agents.ConstantAgent, "predict", "wrapper.constant_opponent"),
        (agents.RewardManager, "process", "wrapper.base_reward"),
        (agents.SelfPlayWarehouseBrawl, "step", "wrapper.selfplay_step"),
        (agents.SelfPlayWarehouseBrawl, "reset", "wrapper.selfplay_reset"),
        (train.FrozenOpponentWrapper, "step", "wrapper.navigation_reward_and_end_checks"),
        (train.FrozenOpponentWrapper, "reset", "wrapper.navigation_reset"),
        (train.Float32Wrapper, "observation", "wrapper.float32_observation"),
        (train.OpponentHistoryWrapper, "step", "wrapper.history_step"),
        (train.OpponentHistoryWrapper, "_augment_obs", "wrapper.history_augmentation"),
        (Monitor, "step", "wrapper.monitor_step"),
        (Monitor, "reset", "wrapper.monitor_reset"),
    ]:
        spans.patch(target, method, label)
    controller = type(env.unwrapped.raw_env.weapon_controller)
    spans.patch(controller, "try_pick_up_all", "game.weapon_pickups")
    spans.patch(controller, "update", "game.weapon_update")
    return WorkerTimingWrapper(env, spans, worker_index)


def build_env(config_path, seed, n_envs, torch_threads):
    import benchmark_training as bt
    bt.set_headless_environment()
    bt.add_repo_imports()
    from train import PhaseConfig
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
    params = bt.load_config(Path(config_path))
    phase = PhaseConfig(params, params["curriculum"]["phase"])
    if not phase.is_navigation:
        raise ValueError("This diagnostic is scoped to navigation")
    factories = [partial(_make_worker, str(Path(config_path).resolve()), seed, torch_threads, index) for index in range(n_envs)]
    env = SubprocVecEnv(factories, start_method="spawn") if n_envs > 1 else DummyVecEnv(factories)
    if int(params.get("frame_stack", 1)) > 1:
        env = VecFrameStack(env, n_stack=int(params["frame_stack"]))
    return env, params, phase
