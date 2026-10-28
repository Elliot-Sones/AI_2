"""Behavior checks for shared-action grouped VecEnv transport."""

from __future__ import annotations

import pickle
import random
import unittest

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from ai2.simulation.grouped_vec_env import GroupedSubprocVecEnv as OriginalGroupedSubprocVecEnv
except ModuleNotFoundError as exc:  # pragma: no cover - light local Python skips runtime tests.
    gym = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

if _IMPORT_ERROR is None:
    from ai2.simulation.shared_action_vec_env import (
        ACTION_DIMENSION,
        GroupedSubprocVecEnv,
        SharedBatchedRngGroupedSubprocVecEnv,
        SharedActionGroupedSubprocVecEnv,
        _SharedActionRemote,
        _STEP_SHARED_BYTES,
    )


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing grouped runtime dependency: {getattr(_IMPORT_ERROR, 'name', _IMPORT_ERROR)}")
class SharedActionVecEnvTests(unittest.TestCase):
    def make_global_rng_env(self, env_id: int, *, terminate_every: int = 0):
        class GlobalRngTenActionEnv(gym.Env):
            metadata = {"render_modes": []}
            render_mode = None

            def __init__(self):
                self.env_id = int(env_id)
                self.terminate_every = int(terminate_every)
                self.step_count = 0
                self.reset_count = 0
                self.observation_space = spaces.Box(low=-10_000, high=10_000, shape=(4,), dtype=np.float32)
                self.action_space = spaces.Box(low=-10, high=10, shape=(ACTION_DIMENSION,), dtype=np.float32)

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
                    "action_sum": float(np.asarray(action, dtype=np.float32).sum()),
                    "py": random.random(),
                    "np": float(np.random.random()),
                }

            def ping(self, prefix: str = "") -> str:
                return f"{prefix}{self.env_id}:{self.step_count}:{self.reset_count}"

        def factory():
            random.seed(40_000 + env_id)
            np.random.seed(50_000 + env_id)
            return GlobalRngTenActionEnv()

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

    def test_default_grouped_alias_remains_shared_action_only(self):
        self.assertIs(GroupedSubprocVecEnv, SharedActionGroupedSubprocVecEnv)
        self.assertIsNot(GroupedSubprocVecEnv, SharedBatchedRngGroupedSubprocVecEnv)

    def test_shared_action_group_matches_original_for_global_rng_and_staggered_autoresets(self):
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
        shared = SharedActionGroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        try:
            np.testing.assert_array_equal(shared.reset(), original.reset())
            self.assertEqual(shared.reset_infos, original.reset_infos)
            for step in range(5):
                actions = np.full((5, ACTION_DIMENSION), step + 0.25, dtype=np.float32)
                self.compare_steps(original.step(actions), shared.step(actions))
            self.assertEqual(
                shared.env_method("ping", "x-", indices=[0, 2, 4]),
                original.env_method("ping", "x-", indices=[0, 2, 4]),
            )
        finally:
            original.close()
            shared.close()

    def test_step_async_sends_cached_shared_token_and_copies_actions(self):
        env_fns = [self.make_global_rng_env(0), self.make_global_rng_env(1)]
        shared = SharedActionGroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        try:
            shared.reset()
            sent_payloads = []
            original_send_bytes = shared.remotes[0].send_bytes

            def record_send_bytes(payload):
                sent_payloads.append(payload)
                return original_send_bytes(payload)

            shared.remotes[0].send_bytes = record_send_bytes
            actions = np.ones((2, ACTION_DIMENSION), dtype=np.float32)
            shared.step_async(actions)
            actions[:] = 9.0
            _obs, rewards, _dones, infos = shared.step_wait()
        finally:
            shared.close()

        self.assertEqual(sent_payloads, [_STEP_SHARED_BYTES])
        self.assertEqual(pickle.loads(sent_payloads[0]), ("step_shared", None))
        np.testing.assert_array_equal(rewards, np.asarray([ACTION_DIMENSION, ACTION_DIMENSION + 1], dtype=np.float64))
        self.assertEqual(infos[0]["action_sum"], float(ACTION_DIMENSION))

    def test_shared_batched_rng_group_matches_original_with_staggered_autoresets(self):
        env_fns = [
            self.make_global_rng_env(0, terminate_every=2),
            self.make_global_rng_env(1, terminate_every=0),
            self.make_global_rng_env(2, terminate_every=3),
            self.make_global_rng_env(3, terminate_every=1),
            self.make_global_rng_env(4, terminate_every=4),
        ]
        random.seed(1777)
        np.random.seed(1888)
        original = OriginalGroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        random.seed(1777)
        np.random.seed(1888)
        shared_batched = SharedBatchedRngGroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        try:
            np.testing.assert_array_equal(shared_batched.reset(), original.reset())
            self.assertEqual(shared_batched.reset_infos, original.reset_infos)
            for step in range(5):
                actions = np.full((5, ACTION_DIMENSION), step + 0.5, dtype=np.float32)
                self.compare_steps(original.step(actions), shared_batched.step(actions))
            self.assertEqual(
                shared_batched.env_method("ping", "x-", indices=[0, 2, 4]),
                original.env_method("ping", "x-", indices=[0, 2, 4]),
            )
        finally:
            original.close()
            shared_batched.close()

    def test_shared_batched_rng_uses_cached_shared_token_transport(self):
        env_fns = [self.make_global_rng_env(0), self.make_global_rng_env(1)]
        shared_batched = SharedBatchedRngGroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        try:
            shared_batched.reset()
            sent_payloads = []
            original_send_bytes = shared_batched.remotes[0].send_bytes

            def record_send_bytes(payload):
                sent_payloads.append(payload)
                return original_send_bytes(payload)

            shared_batched.remotes[0].send_bytes = record_send_bytes
            actions = np.ones((2, ACTION_DIMENSION), dtype=np.float32)
            shared_batched.step_async(actions)
            actions[:] = 9.0
            _obs, rewards, _dones, infos = shared_batched.step_wait()
        finally:
            shared_batched.close()

        self.assertEqual(sent_payloads, [_STEP_SHARED_BYTES])
        np.testing.assert_array_equal(rewards, np.asarray([ACTION_DIMENSION, ACTION_DIMENSION + 1], dtype=np.float64))
        self.assertEqual(infos[0]["action_sum"], float(ACTION_DIMENSION))

    def test_rejects_non_ten_dimensional_action_space(self):
        class TwoActionEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                return np.zeros((1,), dtype=np.float32), {}

            def step(self, action):
                return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

        with self.assertRaisesRegex(ValueError, "10D"):
            SharedActionGroupedSubprocVecEnv([TwoActionEnv], group_size=1, start_method="spawn")

    def test_rejects_overwrite_while_pending(self):
        env_fns = [self.make_global_rng_env(0), self.make_global_rng_env(1)]
        shared = SharedActionGroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        try:
            shared.reset()
            actions = np.zeros((2, ACTION_DIMENSION), dtype=np.float32)
            shared.step_async(actions)
            with self.assertRaisesRegex(RuntimeError, "Already waiting"):
                shared.step_async(actions)
            shared.step_wait()
        finally:
            shared.close()

    def test_remote_adapter_translates_shared_step_to_normal_step_with_copy(self):
        backing = np.arange(20, dtype=np.float32).reshape(2, ACTION_DIMENSION)

        class FakeRemote:
            def recv(self):
                return ("step_shared", None)

            def send(self, value):
                return value

            def send_bytes(self, value):
                return value

            def close(self):
                return None

        adapter = _SharedActionRemote(FakeRemote(), backing)
        cmd, action_copy = adapter.recv()
        backing[:] = -1

        self.assertEqual(cmd, "step")
        np.testing.assert_array_equal(action_copy, np.arange(20, dtype=np.float32).reshape(2, ACTION_DIMENSION))


if __name__ == "__main__":
    unittest.main()
