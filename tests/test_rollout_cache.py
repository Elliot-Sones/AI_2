"""Parity tests for optional GPU-resident PPO rollout caching."""

from __future__ import annotations

import unittest
from unittest import mock

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from rollout_cache import CachedRolloutBuffer


def _torch_device() -> th.device:
    return th.device("cuda", th.cuda.current_device()) if th.cuda.is_available() else th.device("cpu")


def _new_buffer(cls, *, buffer_size: int = 5, n_envs: int = 3):
    return cls(
        buffer_size,
        spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
        spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
        device=_torch_device(),
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=n_envs,
    )


def _fill_buffer(buffer: RolloutBuffer) -> RolloutBuffer:
    shape = (buffer.buffer_size, buffer.n_envs)
    rows = np.arange(buffer.buffer_size * buffer.n_envs, dtype=np.float32).reshape(shape)
    buffer.observations[...] = np.stack((rows, rows + 100, rows + 200), axis=-1)
    buffer.actions[...] = np.stack((rows + 300, rows + 400), axis=-1)
    buffer.values[...] = rows + 500
    buffer.log_probs[...] = rows + 600
    buffer.advantages[...] = rows + 700
    buffer.returns[...] = rows + 800
    buffer.full = True
    buffer.pos = buffer.buffer_size
    buffer.generator_ready = False
    return buffer


def _sample_batches(buffer: RolloutBuffer, *, batch_size: int, epochs: int):
    batches = []
    for _ in range(epochs):
        batches.append(
            [
                tuple(field.detach().cpu().clone() for field in sample)
                for sample in buffer.get(batch_size=batch_size)
            ]
        )
    return batches


def _assert_batches_equal(testcase: unittest.TestCase, expected, actual) -> None:
    testcase.assertEqual(len(expected), len(actual))
    for expected_epoch, actual_epoch in zip(expected, actual):
        testcase.assertEqual(len(expected_epoch), len(actual_epoch))
        for expected_batch, actual_batch in zip(expected_epoch, actual_epoch):
            testcase.assertEqual(len(expected_batch), len(actual_batch))
            for expected_field, actual_field in zip(expected_batch, actual_batch):
                testcase.assertEqual(expected_field.dtype, actual_field.dtype)
                testcase.assertEqual(expected_field.shape, actual_field.shape)
                th.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)


class TinyDeterministicEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.array([0.0, 1.0, 2.0], dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        observation = np.array(
            [self.step_count, self.step_count + 1, self.step_count + 2],
            dtype=np.float32,
        )
        reward = float(self.step_count) / 10.0
        terminated = self.step_count >= 4
        truncated = False
        return observation, reward, terminated, truncated, {}


def _make_ppo(rollout_buffer_class):
    return PPO(
        "MlpPolicy",
        DummyVecEnv([TinyDeterministicEnv]),
        rollout_buffer_class=rollout_buffer_class,
        device="cuda",
        seed=321,
        n_steps=4,
        batch_size=4,
        n_epochs=2,
        learning_rate=1e-3,
        gamma=0.9,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [8]},
        verbose=0,
    )


def _train_one_cuda_update(rollout_buffer_class):
    np.random.seed(999)
    th.manual_seed(999)
    model = _make_ppo(rollout_buffer_class)
    model.learn(total_timesteps=4, progress_bar=False)
    return model


def _assert_policy_and_optimizer_close(testcase: unittest.TestCase, expected: PPO, actual: PPO) -> None:
    for name, expected_tensor in expected.policy.state_dict().items():
        actual_tensor = actual.policy.state_dict()[name]
        th.testing.assert_close(
            actual_tensor.detach().cpu(),
            expected_tensor.detach().cpu(),
            rtol=1e-6,
            atol=1e-7,
            msg=lambda msg, key=name: f"{key}: {msg}",
        )

    expected_state = expected.policy.optimizer.state_dict()["state"]
    actual_state = actual.policy.optimizer.state_dict()["state"]
    testcase.assertEqual(expected_state.keys(), actual_state.keys())
    for param_key in expected_state:
        testcase.assertEqual(expected_state[param_key].keys(), actual_state[param_key].keys())
        for state_key, expected_value in expected_state[param_key].items():
            actual_value = actual_state[param_key][state_key]
            if th.is_tensor(expected_value):
                th.testing.assert_close(
                    actual_value.detach().cpu(),
                    expected_value.detach().cpu(),
                    rtol=1e-6,
                    atol=1e-7,
                    msg=lambda msg, key=(param_key, state_key): f"{key}: {msg}",
                )
            else:
                testcase.assertEqual(expected_value, actual_value)


class CachedRolloutBufferTest(unittest.TestCase):
    def test_matches_sb3_values_dtypes_shapes_and_order_across_eight_epochs(self):
        baseline = _fill_buffer(_new_buffer(RolloutBuffer))
        cached = _fill_buffer(_new_buffer(CachedRolloutBuffer))

        np.random.seed(123)
        expected = _sample_batches(baseline, batch_size=4, epochs=8)
        np.random.seed(123)
        actual = _sample_batches(cached, batch_size=4, epochs=8)

        _assert_batches_equal(self, expected, actual)
        self.assertTrue(cached.generator_ready)
        self.assertEqual(cached.cache_upload_counts, {
            "observations": 1,
            "actions": 1,
            "values": 1,
            "log_probs": 1,
            "advantages": 1,
            "returns": 1,
        })
        self.assertTrue(all(tensor.device == _torch_device() for tensor in cached.cached_tensors.values()))

    def test_get_keeps_partial_minibatch(self):
        cached = _fill_buffer(_new_buffer(CachedRolloutBuffer, buffer_size=5, n_envs=3))

        batches = list(cached.get(batch_size=4))

        self.assertEqual([sample.observations.shape[0] for sample in batches], [4, 4, 4, 3])

    def test_get_requires_complete_rollout(self):
        cached = _new_buffer(CachedRolloutBuffer)
        cached.full = False

        with self.assertRaises(AssertionError):
            list(cached.get(batch_size=4))
        self.assertEqual(cached.cached_tensors, {})

    def test_reset_invalidates_cache_before_next_rollout(self):
        cached = _fill_buffer(_new_buffer(CachedRolloutBuffer))
        list(cached.get(batch_size=4))
        first_cached_obs = cached.cached_tensors["observations"]

        cached.reset()

        self.assertFalse(cached.generator_ready)
        self.assertEqual(cached.cached_tensors, {})
        self.assertFalse(cached.cache_valid)

        _fill_buffer(cached)
        cached.observations[..., 0] += 1000
        first_batch = next(cached.get(batch_size=15))

        self.assertIsNot(cached.cached_tensors["observations"], first_cached_obs)
        self.assertGreaterEqual(first_batch.observations[:, 0].min().item(), 1000)

    def test_clear_cache_releases_device_tensors_but_preserves_flattened_host_rollout(self):
        cached = _fill_buffer(_new_buffer(CachedRolloutBuffer))
        list(cached.get(batch_size=4))

        cached.clear_cache()

        self.assertTrue(cached.generator_ready)
        self.assertFalse(cached.cache_valid)
        self.assertEqual(cached.cached_tensors, {})
        second_pass = list(cached.get(batch_size=6))
        self.assertEqual([sample.observations.shape[0] for sample in second_pass], [6, 6, 3])
        self.assertEqual(cached.cache_upload_counts["observations"], 2)

    def test_cache_is_uploaded_once_per_rollout_not_once_per_epoch_or_minibatch(self):
        cached = _fill_buffer(_new_buffer(CachedRolloutBuffer, buffer_size=6, n_envs=2))

        for _ in range(8):
            list(cached.get(batch_size=5))

        self.assertEqual(sum(cached.cache_upload_counts.values()), 6)

    def test_cached_samples_are_indexed_from_cached_device_storage(self):
        cached = _fill_buffer(_new_buffer(CachedRolloutBuffer))
        list(cached.get(batch_size=4))

        with mock.patch.object(cached, "to_torch", side_effect=AssertionError("unexpected host upload")):
            batches = list(cached.get(batch_size=4))

        self.assertEqual(sum(batch.observations.shape[0] for batch in batches), 15)

    @unittest.skipUnless(th.cuda.is_available(), "one-update parity is CUDA-only")
    def test_one_update_cuda_policy_and_optimizer_parity(self):
        baseline = _train_one_cuda_update(RolloutBuffer)
        cached = _train_one_cuda_update(CachedRolloutBuffer)

        _assert_policy_and_optimizer_close(self, baseline, cached)


if __name__ == "__main__":
    unittest.main()
