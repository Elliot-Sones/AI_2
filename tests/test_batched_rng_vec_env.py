"""Behavior checks for batched outer-RNG preservation in grouped workers."""

from __future__ import annotations

import random
import sys
import unittest

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from grouped_vec_env import GroupedSubprocVecEnv as OriginalGroupedSubprocVecEnv
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on light local Python only.
    gym = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

if _IMPORT_ERROR is None:
    from batched_rng_vec_env import (
        BatchedRngGroupedSubprocVecEnv,
        _BatchController,
        _BatchedGlobalRngStates,
    )


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing grouped runtime dependency: {getattr(_IMPORT_ERROR, 'name', _IMPORT_ERROR)}")
class BatchedRngVecEnvTests(unittest.TestCase):
    def make_global_rng_env(self, env_id: int, *, terminate_every: int = 0):
        class GlobalRngEnv(gym.Env):
            metadata = {"render_modes": []}
            render_mode = None

            def __init__(self):
                self.env_id = int(env_id)
                self.terminate_every = int(terminate_every)
                self.step_count = 0
                self.reset_count = 0
                self.observation_space = spaces.Box(low=-10_000, high=10_000, shape=(4,), dtype=np.float32)
                self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)

            def _obs(self, marker: int) -> np.ndarray:
                return np.asarray(
                    [
                        self.env_id,
                        marker,
                        int(random.random() * 1_000_000),
                        int(np.random.random() * 1_000_000),
                    ],
                    dtype=np.float32,
                )

            def reset(self, *, seed=None, options=None):
                self.step_count = 0
                self.reset_count += 1
                return self._obs(100 + self.reset_count), {
                    "env_id": self.env_id,
                    "reset_count": self.reset_count,
                    "py": random.random(),
                    "np": float(np.random.random()),
                }

            def step(self, action):
                self.step_count += 1
                terminated = bool(self.terminate_every and self.step_count >= self.terminate_every)
                reward = float(self.env_id + np.asarray(action, dtype=np.float32).sum())
                return self._obs(200 + self.step_count), reward, terminated, False, {
                    "env_id": self.env_id,
                    "step_count": self.step_count,
                    "py": random.random(),
                    "np": float(np.random.random()),
                }

            def ping(self, prefix: str = "") -> str:
                return f"{prefix}{self.env_id}:{self.step_count}:{self.reset_count}"

        def factory():
            random.seed(20_000 + env_id)
            np.random.seed(30_000 + env_id)
            return GlobalRngEnv()

        return factory

    def compare_infos(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for exp, got in zip(expected, actual):
            self.assertEqual(exp.keys(), got.keys())
            for key in exp:
                if key == "terminal_observation":
                    np.testing.assert_array_equal(got[key], exp[key])
                else:
                    self.assertEqual(got[key], exp[key])

    def compare_steps(self, expected, actual):
        exp_obs, exp_rewards, exp_dones, exp_infos = expected
        obs, rewards, dones, infos = actual
        np.testing.assert_array_equal(obs, exp_obs)
        np.testing.assert_array_equal(rewards, exp_rewards)
        np.testing.assert_array_equal(dones, exp_dones)
        self.compare_infos(exp_infos, infos)

    def test_batched_group_matches_original_for_global_rng_and_staggered_autoresets(self):
        env_fns = [
            self.make_global_rng_env(0, terminate_every=2),
            self.make_global_rng_env(1, terminate_every=0),
            self.make_global_rng_env(2, terminate_every=3),
            self.make_global_rng_env(3, terminate_every=1),
            self.make_global_rng_env(4, terminate_every=4),
        ]
        random.seed(777)
        np.random.seed(888)
        original = OriginalGroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        random.seed(777)
        np.random.seed(888)
        batched = BatchedRngGroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        try:
            np.testing.assert_array_equal(batched.reset(), original.reset())
            self.assertEqual(batched.reset_infos, original.reset_infos)
            for step in range(5):
                actions = np.full((5, 2), step + 0.25, dtype=np.float32)
                self.compare_steps(original.step(actions), batched.step(actions))
            self.assertEqual(
                batched.env_method("ping", "x-", indices=[0, 2, 4]),
                original.env_method("ping", "x-", indices=[0, 2, 4]),
            )
        finally:
            original.close()
            batched.close()

    def test_batched_rng_restores_outer_state_once_per_command(self):
        controller = _BatchController()
        states = _BatchedGlobalRngStates(2, controller)

        random.seed(101)
        np.random.seed(202)
        expected_py = random.random()
        expected_np = float(np.random.random())

        random.seed(101)
        np.random.seed(202)
        controller.begin("step")
        states.run(0, lambda: (random.random(), np.random.random()))
        states.run(1, lambda: (random.random(), np.random.random()))
        controller.end()

        self.assertEqual(random.random(), expected_py)
        self.assertEqual(float(np.random.random()), expected_np)
        self.assertEqual(controller.outer_save_count, 1)
        self.assertEqual(controller.outer_restore_count, 1)

    def test_batched_rng_restores_outer_state_after_callback_exception(self):
        controller = _BatchController()
        states = _BatchedGlobalRngStates(1, controller)

        random.seed(303)
        np.random.seed(404)
        expected_py = random.random()
        expected_np = float(np.random.random())

        random.seed(303)
        np.random.seed(404)
        controller.begin("step")

        def boom():
            random.random()
            np.random.random()
            raise RuntimeError("callback failed")

        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            states.run(0, boom)

        self.assertFalse(controller.active)
        self.assertEqual(random.random(), expected_py)
        self.assertEqual(float(np.random.random()), expected_np)
        self.assertEqual(controller.outer_save_count, 1)
        self.assertEqual(controller.outer_restore_count, 1)

    def test_parent_worker_target_restores_after_constructor_exception(self):
        import batched_rng_vec_env
        import grouped_vec_env

        original_worker = grouped_vec_env._worker
        original_init = grouped_vec_env.GroupedSubprocVecEnv.__init__

        def raising_init(self, *args, **kwargs):
            self.closed = True
            raise RuntimeError("spawn failed")

        grouped_vec_env.GroupedSubprocVecEnv.__init__ = raising_init
        try:
            with self.assertRaisesRegex(RuntimeError, "spawn failed"):
                batched_rng_vec_env.BatchedRngGroupedSubprocVecEnv([], group_size=2)
        finally:
            grouped_vec_env.GroupedSubprocVecEnv.__init__ = original_init

        self.assertIs(grouped_vec_env._worker, original_worker)


if __name__ == "__main__":
    unittest.main()
