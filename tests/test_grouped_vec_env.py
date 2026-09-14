"""Behavior checks for grouped subprocess vector environments."""

from __future__ import annotations

import unittest
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from grouped_vec_env import GroupedSubprocVecEnv


class ToyGroupedEnv(gym.Env):
    metadata = {"render_modes": []}
    render_mode = None

    def __init__(
        self,
        env_id: int,
        *,
        terminate_every: int = 0,
        truncate_every: int = 0,
        obs_dtype=np.float32,
        action_dtype=np.float32,
        obs_shape=(3,),
        action_shape=(2,),
    ) -> None:
        super().__init__()
        self.env_id = int(env_id)
        self.terminate_every = int(terminate_every)
        self.truncate_every = int(truncate_every)
        self.observation_space = spaces.Box(low=-1_000, high=1_000, shape=obs_shape, dtype=obs_dtype)
        self.action_space = spaces.Box(low=-10, high=10, shape=action_shape, dtype=action_dtype)
        self.step_count = 0
        self.reset_count = 0
        self.close_count = 0
        self.last_seed = None
        self.last_options = None
        self.marker = f"env-{env_id}"

    def _obs(self, phase: int) -> np.ndarray:
        return np.full(self.observation_space.shape, self.env_id * 100 + phase, dtype=self.observation_space.dtype)

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.reset_count += 1
        self.last_seed = seed
        self.last_options = options
        option_offset = 0 if not options else int(options.get("offset", 0))
        return self._obs(10 + self.reset_count + option_offset), {
            "env_id": self.env_id,
            "seed": seed,
            "options": options,
            "reset_count": self.reset_count,
        }

    def step(self, action):
        self.step_count += 1
        terminated = bool(self.terminate_every and self.step_count >= self.terminate_every)
        truncated = bool(self.truncate_every and self.step_count >= self.truncate_every)
        reward = float(self.env_id + np.asarray(action, dtype=np.float32).sum())
        return self._obs(50 + self.step_count), reward, terminated, truncated, {
            "env_id": self.env_id,
            "step_count": self.step_count,
            "action_sum": float(np.asarray(action).sum()),
        }

    def close(self):
        self.close_count += 1

    def ping(self, prefix: str = "") -> str:
        return f"{prefix}{self.env_id}:{self.step_count}:{self.reset_count}"

    def set_truncate_every(self, truncate_every: int) -> None:
        self.truncate_every = int(truncate_every)


class ToyWrapper(gym.Wrapper):
    pass


class GlobalRngEnv(ToyGroupedEnv):
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        obs, info = super().reset(seed=seed, options=options)
        info["reset_py_random"] = random.random()
        info["reset_np_random"] = float(np.random.random())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["step_py_random"] = random.random()
        info["step_np_random"] = float(np.random.random())
        return obs, reward, terminated, truncated, info


class HighPrecisionRewardEnv(ToyGroupedEnv):
    def step(self, action):
        obs, _reward, terminated, truncated, info = super().step(action)
        reward = float(self.env_id + 1.0 + 2**-40)
        info["reward"] = reward
        return obs, reward, terminated, truncated, info


class FactorySeededGlobalResetEnv(ToyGroupedEnv):
    def __init__(self, env_id: int) -> None:
        super().__init__(env_id)
        self.factory_py_random = random.random()
        self.factory_np_random = float(np.random.random())

    def _obs(self, phase: int) -> np.ndarray:
        py_component = int(self.last_py_random * 1000) if hasattr(self, "last_py_random") else 0
        np_component = int(self.last_np_random * 1000) if hasattr(self, "last_np_random") else 0
        return np.asarray(
            [self.env_id, phase + py_component, phase + np_component],
            dtype=self.observation_space.dtype,
        )

    def reset(self, *, seed=None, options=None):
        self.last_py_random = random.random()
        self.last_np_random = float(np.random.random())
        obs, info = super().reset(seed=seed, options=options)
        info.update(
            {
                "factory_py_random": self.factory_py_random,
                "factory_np_random": self.factory_np_random,
                "reset_py_random": self.last_py_random,
                "reset_np_random": self.last_np_random,
            }
        )
        return obs, info


def make_env(env_id: int, **kwargs):
    def _factory():
        return ToyGroupedEnv(env_id, **kwargs)

    return _factory


def make_wrapped_env(env_id: int):
    def _factory():
        return ToyWrapper(ToyGroupedEnv(env_id))

    return _factory


def make_global_rng_env(env_id: int):
    def _factory():
        return GlobalRngEnv(env_id)

    return _factory


def make_high_precision_reward_env(env_id: int):
    def _factory():
        return HighPrecisionRewardEnv(env_id)

    return _factory


def make_factory_seeded_global_reset_env(env_id: int):
    def _factory():
        random.seed(10_000 + env_id)
        np.random.seed(10_000 + env_id)
        return FactorySeededGlobalResetEnv(env_id)

    return _factory


def _assert_step_equal(testcase: unittest.TestCase, expected, actual) -> None:
    exp_obs, exp_rewards, exp_dones, exp_infos = expected
    obs, rewards, dones, infos = actual
    np.testing.assert_array_equal(obs, exp_obs)
    np.testing.assert_array_equal(rewards, exp_rewards)
    np.testing.assert_array_equal(dones, exp_dones)
    testcase.assertEqual(len(infos), len(exp_infos))
    for expected_info, actual_info in zip(exp_infos, infos):
        testcase.assertEqual(expected_info.keys(), actual_info.keys())
        for key in expected_info:
            if key == "terminal_observation":
                np.testing.assert_array_equal(actual_info[key], expected_info[key])
            else:
                testcase.assertEqual(actual_info[key], expected_info[key])


class GroupedSubprocVecEnvTest(unittest.TestCase):
    def test_reset_seed_options_and_step_order_match_dummy_vec_env(self):
        env_fns = [make_env(index) for index in range(5)]
        grouped = GroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        dummy = DummyVecEnv(env_fns)
        try:
            grouped.seed(100)
            dummy.seed(100)
            options = [{"offset": index} for index in range(5)]
            grouped.set_options(options)
            dummy.set_options(options)

            np.testing.assert_array_equal(grouped.reset(), dummy.reset())
            self.assertEqual(grouped.reset_infos, dummy.reset_infos)

            actions = np.arange(10, dtype=np.float32).reshape(5, 2)
            _assert_step_equal(self, dummy.step(actions), grouped.step(actions))
        finally:
            grouped.close()
            dummy.close()

    def test_basic_reset_and_step_order_match_subproc_vec_env(self):
        env_fns = [make_env(index) for index in range(4)]
        grouped = GroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        subproc = SubprocVecEnv(env_fns, start_method="spawn")
        try:
            np.testing.assert_array_equal(grouped.reset(), subproc.reset())
            actions = np.arange(8, dtype=np.float32).reshape(4, 2)
            _assert_step_equal(self, subproc.step(actions), grouped.step(actions))
        finally:
            grouped.close()
            subproc.close()

    def test_python_float_rewards_preserve_stock_float64_precision(self):
        env_fns = [make_high_precision_reward_env(index) for index in range(3)]
        grouped = GroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        subproc = SubprocVecEnv(env_fns, start_method="spawn")
        try:
            grouped.reset()
            subproc.reset()
            actions = np.zeros((3, 2), dtype=np.float32)

            expected_rewards = subproc.step(actions)[1]
            actual_rewards = grouped.step(actions)[1]

            self.assertEqual(actual_rewards.dtype, expected_rewards.dtype)
            self.assertEqual(actual_rewards.dtype, np.float64)
            np.testing.assert_array_equal(actual_rewards, expected_rewards)
        finally:
            grouped.close()
            subproc.close()

    def test_terminal_and_timeout_infos_match_dummy_vec_env(self):
        env_fns = [
            make_env(0, terminate_every=1),
            make_env(1, truncate_every=1),
            make_env(2, terminate_every=2),
        ]
        grouped = GroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        dummy = DummyVecEnv(env_fns)
        try:
            grouped.reset()
            dummy.reset()
            actions = np.zeros((3, 2), dtype=np.float32)

            _assert_step_equal(self, dummy.step(actions), grouped.step(actions))
            self.assertTrue(grouped.step(actions)[2][2])
        finally:
            grouped.close()
            dummy.close()

    def test_global_random_streams_are_isolated_between_grouped_envs(self):
        env_fns = [make_global_rng_env(index) for index in range(4)]
        actions = np.zeros((4, 2), dtype=np.float32)
        grouped_pairs = GroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        grouped_singles = GroupedSubprocVecEnv(env_fns, group_size=1, start_method="spawn")
        try:
            grouped_pairs.seed(700)
            grouped_singles.seed(700)
            grouped_pairs.reset()
            grouped_singles.reset()

            for _ in range(3):
                _assert_step_equal(self, grouped_singles.step(actions), grouped_pairs.step(actions))
        finally:
            grouped_pairs.close()
            grouped_singles.close()

    def test_explicit_seed_reset_preserves_factory_global_rng_state_like_subproc_vec_env(self):
        env_fns = [make_factory_seeded_global_reset_env(index) for index in range(4)]
        grouped = GroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        subproc = SubprocVecEnv(env_fns, start_method="spawn")
        try:
            grouped.seed(123)
            subproc.seed(123)

            np.testing.assert_array_equal(grouped.reset(), subproc.reset())
            self.assertEqual(list(grouped.reset_infos), list(subproc.reset_infos))
        finally:
            grouped.close()
            subproc.close()

    def test_global_random_autoreset_streams_are_isolated(self):
        env_fns = [
            make_global_rng_env(0),
            make_global_rng_env(1),
            make_global_rng_env(2),
        ]
        actions = np.zeros((3, 2), dtype=np.float32)
        grouped_pairs = GroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn")
        grouped_singles = GroupedSubprocVecEnv(env_fns, group_size=1, start_method="spawn")
        try:
            grouped_pairs.seed(900)
            grouped_singles.seed(900)
            grouped_pairs.reset()
            grouped_singles.reset()
            grouped_pairs.env_method("set_truncate_every", 2)
            grouped_singles.env_method("set_truncate_every", 2)

            _assert_step_equal(self, grouped_singles.step(actions), grouped_pairs.step(actions))
            _assert_step_equal(self, grouped_singles.step(actions), grouped_pairs.step(actions))
        finally:
            grouped_pairs.close()
            grouped_singles.close()

    def test_partial_group_shapes_and_shared_buffers(self):
        grouped = GroupedSubprocVecEnv([make_env(index) for index in range(5)], group_size=2, start_method="spawn")
        try:
            obs = grouped.reset()
            self.assertEqual(obs.shape, (5, 3))
            self.assertEqual(grouped.num_groups, 3)
            self.assertEqual(grouped.group_sizes, [2, 2, 1])

            actions = np.zeros((5, 2), dtype=np.float32)
            obs, rewards, dones, infos = grouped.step(actions)

            self.assertEqual(obs.shape, (5, 3))
            self.assertEqual(rewards.shape, (5,))
            self.assertEqual(dones.shape, (5,))
            self.assertEqual([info["env_id"] for info in infos], [0, 1, 2, 3, 4])
            self.assertTrue(all(buffer.shape[0] <= 2 for buffer in grouped.shared_observations))
        finally:
            grouped.close()

    def test_env_method_get_attr_set_attr_and_wrapped_indices(self):
        grouped = GroupedSubprocVecEnv([make_wrapped_env(0), make_wrapped_env(1), make_wrapped_env(2)], group_size=2, start_method="spawn")
        try:
            grouped.reset()
            self.assertEqual(grouped.get_attr("marker", indices=[2, 0]), ["env-2", "env-0"])
            grouped.set_attr("marker", "changed", indices=[1])
            self.assertEqual(grouped.get_attr("marker", indices=[0, 1, 2]), ["env-0", "changed", "env-2"])
            self.assertEqual(grouped.env_method("ping", "p:", indices=[1, 2]), ["p:1:0:1", "p:2:0:1"])
            self.assertEqual(grouped.env_is_wrapped(ToyWrapper, indices=[0, 2]), [True, True])
        finally:
            grouped.close()

    def test_step_async_waiting_and_close(self):
        grouped = GroupedSubprocVecEnv([make_env(0), make_env(1), make_env(2)], group_size=2, start_method="spawn")
        actions = np.zeros((3, 2), dtype=np.float32)
        try:
            grouped.reset()
            grouped.step_async(actions)
            self.assertTrue(grouped.waiting)
            obs, rewards, dones, infos = grouped.step_wait()
            self.assertFalse(grouped.waiting)
            self.assertEqual(obs.shape, (3, 3))
            self.assertEqual(len(infos), 3)
        finally:
            grouped.close()
        self.assertTrue(grouped.closed)
        self.assertTrue(all(not process.is_alive() for process in grouped.processes))

    def test_step_async_rejects_wrong_action_batch_shape(self):
        grouped = GroupedSubprocVecEnv([make_env(0), make_env(1), make_env(2)], group_size=2, start_method="spawn")
        try:
            with self.assertRaisesRegex(ValueError, "action batch"):
                grouped.step_async(np.zeros((2, 2), dtype=np.float32))
            self.assertFalse(grouped.waiting)
        finally:
            grouped.close()

    def test_unsupported_spaces_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Box float32 observation"):
            GroupedSubprocVecEnv([make_env(0, obs_dtype=np.float64)], group_size=1, start_method="spawn")

        with self.assertRaisesRegex(ValueError, "Box float32 action"):
            GroupedSubprocVecEnv([make_env(0, action_dtype=np.float64)], group_size=1, start_method="spawn")

        with self.assertRaisesRegex(ValueError, "same observation/action spaces"):
            GroupedSubprocVecEnv([make_env(0), make_env(1, obs_shape=(4,))], group_size=2, start_method="spawn")

    def test_group_size_validation(self):
        with self.assertRaisesRegex(ValueError, "group_size"):
            GroupedSubprocVecEnv([make_env(0)], group_size=0, start_method="spawn")

        with self.assertRaisesRegex(ValueError, "at least one"):
            GroupedSubprocVecEnv([], group_size=2, start_method="spawn")


if __name__ == "__main__":
    unittest.main()
