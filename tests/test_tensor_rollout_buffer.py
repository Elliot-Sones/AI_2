"""Tests for the optional CUDA-resident PPO rollout buffer."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import unittest

import numpy as np


def _deps(*, require_cuda=True):
    try:
        import torch
        from gymnasium import spaces
        from stable_baselines3.common.buffers import RolloutBuffer
    except Exception as exc:  # pragma: no cover - exercised on dependency-light hosts
        raise unittest.SkipTest(f"torch/gymnasium/SB3 unavailable: {exc}") from exc
    if require_cuda and not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA unavailable")
    return torch, spaces, RolloutBuffer


class TensorRolloutBufferStaticTests(unittest.TestCase):
    def test_module_exports_tensor_rollout_buffer(self):
        module = importlib.import_module("tensor_rollout_buffer")

        self.assertTrue(hasattr(module, "TensorRolloutBuffer"))

    def test_source_does_not_subclass_sb3_numpy_rollout_buffer_or_allocate_numpy_storage(self):
        source = (Path(__file__).resolve().parents[1] / "tensor_rollout_buffer.py").read_text()

        self.assertNotIn("(RolloutBuffer)", source)
        self.assertNotIn("np.empty", source)
        self.assertNotIn("np.zeros", source)
        self.assertNotIn("np.asarray", source)
        self.assertNotIn(".cpu().numpy()", source)


class TensorRolloutBufferCudaTests(unittest.TestCase):
    def setUp(self):
        torch, spaces, RolloutBuffer = _deps()
        self.torch = torch
        self.spaces = spaces
        self.RolloutBuffer = RolloutBuffer
        self.module = importlib.import_module("tensor_rollout_buffer")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.obs_space = spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)

    def new_tensor_buffer(self, *, buffer_size=5, n_envs=3):
        return self.module.TensorRolloutBuffer(
            buffer_size,
            self.obs_space,
            self.action_space,
            device=self.device,
            gae_lambda=0.95,
            gamma=0.99,
            n_envs=n_envs,
        )

    def new_sb3_buffer(self, *, buffer_size=5, n_envs=3):
        return self.RolloutBuffer(
            buffer_size,
            self.obs_space,
            self.action_space,
            device=self.device,
            gae_lambda=0.95,
            gamma=0.99,
            n_envs=n_envs,
        )

    def fill_tensor_buffer(self, buffer):
        torch = self.torch
        for step in range(buffer.buffer_size):
            rows = torch.arange(buffer.n_envs, dtype=torch.float32, device=self.device) + step * 10
            obs = torch.stack((rows, rows + 100, rows + 200), dim=1)
            actions = torch.stack((rows + 300, rows + 400), dim=1)
            rewards = rows + 0.25
            episode_starts = torch.zeros(buffer.n_envs, dtype=torch.float32, device=self.device)
            values = rows + 500
            log_probs = rows + 600
            buffer.add(obs, actions, rewards, episode_starts, values, log_probs)
        buffer.advantages.copy_(torch.arange(buffer.buffer_size * buffer.n_envs, dtype=torch.float32, device=self.device).view(buffer.buffer_size, buffer.n_envs) + 700)
        buffer.returns.copy_(torch.arange(buffer.buffer_size * buffer.n_envs, dtype=torch.float32, device=self.device).view(buffer.buffer_size, buffer.n_envs) + 800)
        buffer._computed_returns_and_advantage = True
        return buffer

    def fill_sb3_buffer(self, buffer):
        shape = (buffer.buffer_size, buffer.n_envs)
        rows = np.arange(buffer.buffer_size * buffer.n_envs, dtype=np.float32).reshape(shape)
        # Use env-major-compatible data: step * 10 + env.
        rows = np.stack([np.arange(buffer.n_envs, dtype=np.float32) + step * 10 for step in range(buffer.buffer_size)])
        buffer.observations[...] = np.stack((rows, rows + 100, rows + 200), axis=-1)
        buffer.actions[...] = np.stack((rows + 300, rows + 400), axis=-1)
        buffer.rewards[...] = rows + 0.25
        buffer.episode_starts[...] = 0.0
        buffer.values[...] = rows + 500
        buffer.log_probs[...] = rows + 600
        buffer.advantages[...] = np.arange(buffer.buffer_size * buffer.n_envs, dtype=np.float32).reshape(shape) + 700
        buffer.returns[...] = np.arange(buffer.buffer_size * buffer.n_envs, dtype=np.float32).reshape(shape) + 800
        buffer.full = True
        buffer.pos = buffer.buffer_size
        buffer.generator_ready = False
        return buffer

    def collect_batches(self, buffer, *, batch_size, epochs):
        batches = []
        for _ in range(epochs):
            epoch = []
            for sample in buffer.get(batch_size=batch_size):
                epoch.append(tuple(field.detach().cpu().clone() for field in sample))
            batches.append(epoch)
        return batches

    def assert_batches_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for expected_epoch, actual_epoch in zip(expected, actual):
            self.assertEqual(len(expected_epoch), len(actual_epoch))
            for expected_batch, actual_batch in zip(expected_epoch, actual_epoch):
                self.assertEqual(len(expected_batch), len(actual_batch))
                for expected_field, actual_field in zip(expected_batch, actual_batch):
                    self.assertEqual(expected_field.shape, actual_field.shape)
                    self.assertEqual(expected_field.dtype, actual_field.dtype)
                    self.torch.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)

    def test_matches_sb3_get_order_and_values_across_eight_epochs_with_partial_batch(self):
        baseline = self.fill_sb3_buffer(self.new_sb3_buffer(buffer_size=5, n_envs=3))
        tensor_buffer = self.fill_tensor_buffer(self.new_tensor_buffer(buffer_size=5, n_envs=3))

        np.random.seed(123)
        expected = self.collect_batches(baseline, batch_size=4, epochs=8)
        np.random.seed(123)
        actual = self.collect_batches(tensor_buffer, batch_size=4, epochs=8)

        self.assert_batches_equal(expected, actual)
        self.assertTrue(tensor_buffer.generator_ready)
        self.assertEqual([batch[0].shape[0] for batch in actual[0]], [4, 4, 4, 3])

    def test_add_snapshots_inputs_and_rejects_cpu_or_malformed_inputs(self):
        torch = self.torch
        buffer = self.new_tensor_buffer(buffer_size=2, n_envs=2)
        obs = torch.ones((2, 3), dtype=torch.float32, device=self.device)
        actions = torch.ones((2, 2), dtype=torch.float32, device=self.device)
        rewards = torch.ones((2,), dtype=torch.float32, device=self.device)
        starts = torch.zeros((2,), dtype=torch.float32, device=self.device)
        values = torch.ones((2, 1), dtype=torch.float32, device=self.device)
        log_probs = torch.ones((2,), dtype=torch.float32, device=self.device)

        buffer.add(obs, actions, rewards, starts, values, log_probs)
        obs.add_(99)
        values.add_(99)

        self.torch.testing.assert_close(buffer.observations[0], torch.ones((2, 3), dtype=torch.float32, device=self.device))
        self.torch.testing.assert_close(buffer.values[0], torch.ones((2,), dtype=torch.float32, device=self.device))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            buffer.add(obs.cpu(), actions, rewards, starts, values, log_probs)
        with self.assertRaisesRegex(ValueError, "shape"):
            buffer.add(obs, actions[:, :1], rewards, starts, values, log_probs)
        with self.assertRaisesRegex(ValueError, "float32"):
            buffer.add(obs.double(), actions, rewards, starts, values, log_probs)

    def test_compute_returns_and_advantage_matches_sb3_terminal_boundaries(self):
        torch = self.torch
        baseline = self.new_sb3_buffer(buffer_size=4, n_envs=2)
        tensor_buffer = self.new_tensor_buffer(buffer_size=4, n_envs=2)
        rewards = np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.0], [1.5, 4.0]], dtype=np.float32)
        starts = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        values = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]], dtype=np.float32)
        baseline.rewards[...] = rewards
        baseline.episode_starts[...] = starts
        baseline.values[...] = values
        baseline.full = True
        baseline.pos = baseline.buffer_size
        tensor_buffer.rewards.copy_(torch.as_tensor(rewards, dtype=torch.float32, device=self.device))
        tensor_buffer.episode_starts.copy_(torch.as_tensor(starts, dtype=torch.float32, device=self.device))
        tensor_buffer.values.copy_(torch.as_tensor(values, dtype=torch.float32, device=self.device))
        tensor_buffer.full = True
        tensor_buffer.pos = tensor_buffer.buffer_size
        last_values = torch.tensor([1.1, -0.2], dtype=torch.float32, device=self.device)
        dones = np.array([False, True])

        baseline.compute_returns_and_advantage(last_values=last_values, dones=dones)
        tensor_buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=torch.as_tensor(dones, dtype=torch.float32, device=self.device),
        )

        self.torch.testing.assert_close(tensor_buffer.advantages.cpu(), torch.as_tensor(baseline.advantages), rtol=0, atol=1e-6)
        self.torch.testing.assert_close(tensor_buffer.returns.cpu(), torch.as_tensor(baseline.returns), rtol=0, atol=1e-6)

    def test_reset_reuses_storage_and_get_requires_full_rollout(self):
        buffer = self.fill_tensor_buffer(self.new_tensor_buffer(buffer_size=2, n_envs=2))
        original_storage = buffer.observations.data_ptr()
        list(buffer.get(batch_size=3))

        buffer.reset()

        self.assertFalse(buffer.full)
        self.assertFalse(buffer.generator_ready)
        self.assertEqual(buffer.pos, 0)
        self.assertEqual(buffer.observations.data_ptr(), original_storage)
        with self.assertRaisesRegex(RuntimeError, "full"):
            list(buffer.get(batch_size=2))

    def test_get_requires_returns_and_advantages_to_be_computed(self):
        buffer = self.fill_tensor_buffer(self.new_tensor_buffer(buffer_size=2, n_envs=2))
        buffer._computed_returns_and_advantage = False

        with self.assertRaisesRegex(RuntimeError, "returns"):
            list(buffer.get(batch_size=2))

    def test_get_rejects_non_strict_positive_integer_batch_size(self):
        buffer = self.fill_tensor_buffer(self.new_tensor_buffer(buffer_size=2, n_envs=2))

        for batch_size in (0, -1, True, 1.5):
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    list(buffer.get(batch_size=batch_size))

    def test_nonfinite_or_nonbinary_data_fails_at_rollout_boundary(self):
        torch = self.torch
        buffer = self.new_tensor_buffer(buffer_size=1, n_envs=2)
        obs = torch.zeros((2, 3), dtype=torch.float32, device=self.device)
        actions = torch.zeros((2, 2), dtype=torch.float32, device=self.device)
        rewards = torch.tensor([1.0, float("nan")], dtype=torch.float32, device=self.device)
        starts = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self.device)
        values = torch.zeros((2,), dtype=torch.float32, device=self.device)
        log_probs = torch.zeros((2,), dtype=torch.float32, device=self.device)

        buffer.add(obs, actions, rewards, starts, values, log_probs)
        self.assertTrue(buffer.full)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            buffer.compute_returns_and_advantage(values, torch.zeros((2,), dtype=torch.float32, device=self.device))

        buffer = self.new_tensor_buffer(buffer_size=1, n_envs=2)
        bad_starts = torch.tensor([0.0, 0.5], dtype=torch.float32, device=self.device)
        buffer.add(obs, actions, torch.zeros((2,), dtype=torch.float32, device=self.device), bad_starts, values, log_probs)
        with self.assertRaisesRegex(ValueError, "non-binary"):
            buffer.compute_returns_and_advantage(values, torch.zeros((2,), dtype=torch.float32, device=self.device))

        buffer = self.new_tensor_buffer(buffer_size=1, n_envs=2)
        buffer.add(obs, actions, torch.zeros((2,), dtype=torch.float32, device=self.device), starts, values, log_probs)
        with self.assertRaisesRegex(ValueError, "non-binary"):
            buffer.compute_returns_and_advantage(values, torch.tensor([0.0, 0.25], dtype=torch.float32, device=self.device))

    def test_compute_returns_requires_full_rollout_without_bare_assert(self):
        torch = self.torch
        buffer = self.new_tensor_buffer(buffer_size=2, n_envs=2)

        with self.assertRaisesRegex(RuntimeError, "full"):
            buffer.compute_returns_and_advantage(
                last_values=torch.zeros((2,), dtype=torch.float32, device=self.device),
                dones=torch.zeros((2,), dtype=torch.float32, device=self.device),
            )

    def test_rejects_unsupported_spaces_and_cpu_device(self):
        torch, spaces, _ = _deps()
        with self.assertRaisesRegex(ValueError, "Box"):
            self.module.TensorRolloutBuffer(2, spaces.Discrete(3), self.action_space, device=self.device, gae_lambda=0.95, gamma=0.99, n_envs=1)
        with self.assertRaisesRegex(ValueError, "float32"):
            bad_obs = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float64)
            self.module.TensorRolloutBuffer(2, bad_obs, self.action_space, device=self.device, gae_lambda=0.95, gamma=0.99, n_envs=1)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.module.TensorRolloutBuffer(2, self.obs_space, self.action_space, device=torch.device("cpu"), gae_lambda=0.95, gamma=0.99, n_envs=1)

    def test_device_cuda_without_index_accepts_current_cuda_tensors(self):
        torch = self.torch
        buffer = self.module.TensorRolloutBuffer(
            1,
            self.obs_space,
            self.action_space,
            device="cuda",
            gae_lambda=0.95,
            gamma=0.99,
            n_envs=1,
        )
        obs = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        actions = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
        scalar = torch.zeros((1,), dtype=torch.float32, device=self.device)

        buffer.add(obs, actions, scalar, scalar, scalar, scalar)

        self.assertEqual(buffer.observations.device, self.device)

    def test_add_rejects_requires_grad_inputs(self):
        torch = self.torch
        buffer = self.new_tensor_buffer(buffer_size=1, n_envs=1)
        obs = torch.zeros((1, 3), dtype=torch.float32, device=self.device, requires_grad=True)
        actions = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
        scalar = torch.zeros((1,), dtype=torch.float32, device=self.device)

        with self.assertRaisesRegex(ValueError, "requires_grad"):
            buffer.add(obs, actions, scalar, scalar, scalar, scalar)


class TensorRolloutBufferDependencyTests(unittest.TestCase):
    def test_cpu_device_is_rejected_before_any_cuda_allocation(self):
        torch, spaces, _ = _deps(require_cuda=False)
        module = importlib.import_module("tensor_rollout_buffer")
        obs_space = spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "CUDA"):
            module.TensorRolloutBuffer(
                2,
                obs_space,
                action_space,
                device=torch.device("cpu"),
                gae_lambda=0.95,
                gamma=0.99,
                n_envs=1,
            )

    def test_constructor_rejects_bad_scalars_before_cuda_allocation(self):
        torch, spaces, _ = _deps(require_cuda=False)
        module = importlib.import_module("tensor_rollout_buffer")
        obs_space = spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)

        bad_cases = [
            {"buffer_size": True, "n_envs": 1, "gae_lambda": 0.95, "gamma": 0.99, "message": "buffer_size"},
            {"buffer_size": 2.0, "n_envs": 1, "gae_lambda": 0.95, "gamma": 0.99, "message": "buffer_size"},
            {"buffer_size": 2, "n_envs": False, "gae_lambda": 0.95, "gamma": 0.99, "message": "n_envs"},
            {"buffer_size": 2, "n_envs": 1.0, "gae_lambda": 0.95, "gamma": 0.99, "message": "n_envs"},
            {"buffer_size": 2, "n_envs": 1, "gae_lambda": float("nan"), "gamma": 0.99, "message": "gae_lambda"},
            {"buffer_size": 2, "n_envs": 1, "gae_lambda": 0.95, "gamma": float("inf"), "message": "gamma"},
            {"buffer_size": 2, "n_envs": 1, "gae_lambda": -0.1, "gamma": 0.99, "message": "gae_lambda"},
            {"buffer_size": 2, "n_envs": 1, "gae_lambda": 0.95, "gamma": 1.1, "message": "gamma"},
        ]
        for case in bad_cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, case["message"]):
                    module.TensorRolloutBuffer(
                        case["buffer_size"],
                        obs_space,
                        action_space,
                        device=torch.device("cuda:0"),
                        n_envs=case["n_envs"],
                        gae_lambda=case["gae_lambda"],
                        gamma=case["gamma"],
                    )


if __name__ == "__main__":
    unittest.main()
