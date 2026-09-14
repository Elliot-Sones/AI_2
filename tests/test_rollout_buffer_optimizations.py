"""Opt-in boundary validation and GAE compilation, with real tensor fixtures."""

from __future__ import annotations

import importlib
import inspect
import unittest
from unittest.mock import patch

import numpy as np


FIELDS = ("observations", "actions", "rewards", "episode_starts", "values", "log_probs")


class _BufferTestCase(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from gymnasium import spaces
        except ImportError as exc:  # pragma: no cover - dependency-light Python
            raise unittest.SkipTest(str(exc)) from exc
        self.torch = torch
        self.spaces = spaces
        self.module = importlib.import_module("tensor_rollout_buffer")

    def new_buffer(self, *, mode="rollout", compile_gae=False, device="cpu", size=4):
        cls = self.module.TensorRolloutBuffer
        kwargs = dict(
            buffer_size=size,
            observation_space=self.spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
            action_space=self.spaces.Box(-10, 10, shape=(2,), dtype=np.float32),
            device=device, gae_lambda=0.95, gamma=0.99, n_envs=2,
            validation_mode=mode, compile_gae=compile_gae,
        )
        if device == "cpu":
            # The production CUDA admission gate stays covered by existing and
            # CUDA tests. Only allocation admission is bypassed for CPU math.
            with patch.object(cls, "_validate_constructor", lambda _self: None):
                return cls(**kwargs)
        return cls(**kwargs)

    def fill_buffer(self, buffer, *, bad_field=None, bad_value=float("nan")):
        torch = self.torch

        def add_steps():
            for step in range(buffer.buffer_size):
                obs = torch.full((2, 3), step * 0.25, device=buffer.device)
                action = torch.full((2, 2), step * -0.125, device=buffer.device)
                reward = torch.tensor([1.0 + step, -0.2 * step], device=buffer.device)
                starts = torch.tensor([float(step == 2), float(step == 1)], device=buffer.device)
                values = torch.tensor([step * 0.1, step * 0.3], device=buffer.device)
                log_probs = torch.full((2,), -0.5 - step * 0.125, device=buffer.device)
                inputs = [obs, action, reward, starts, values, log_probs]
                if step == 0 and bad_field is not None:
                    inputs[FIELDS.index(bad_field)].view(-1)[0] = bad_value
                buffer.add(*inputs)

        if buffer.device.type == "cpu":
            # CPU tests exercise real storage/check/GAE operations. CUDA-only
            # input metadata is tested through the full public API on CUDA.
            with patch.object(buffer, "_require_cuda_float32", lambda _n, t, _s: t), patch.object(
                buffer, "_require_vector_or_column", lambda _n, t: t
            ):
                add_steps()
        else:
            add_steps()
        return buffer

    def boundary(self, buffer):
        self.assertTrue(
            hasattr(buffer, "_validate_rollout_data"),
            "rollout-wide validation is missing",
        )
        buffer._validate_rollout_data()
        buffer._raise_if_invalid_data()

    def run_scan(self, buffer):
        self.assertTrue(hasattr(self.module, "_gae_scan"), "fixed-shape GAE scan is missing")
        torch = self.torch
        self.module._gae_scan(
            buffer.rewards, buffer.episode_starts, buffer.values,
            torch.tensor([0.4, -0.8], device=buffer.device),
            torch.tensor([0.0, 1.0], device=buffer.device),
            buffer.advantages, buffer.returns, buffer.gamma, buffer.gae_lambda,
        )
        buffer._computed_returns_and_advantage = True


class RolloutBufferOptimizationCpuTests(_BufferTestCase):
    def test_options_default_to_original_execution(self):
        parameters = inspect.signature(self.module.TensorRolloutBuffer).parameters
        self.assertIn("validation_mode", parameters)
        self.assertIn("compile_gae", parameters)
        self.assertEqual(parameters["validation_mode"].default, "step")
        self.assertIs(parameters["compile_gae"].default, False)

    def test_postload_options_validate_assignments(self):
        buffer = self.new_buffer()
        for invalid in ("boundary", "", None, True):
            with self.subTest(mode=invalid):
                with self.assertRaisesRegex(ValueError, "validation_mode"):
                    buffer.validation_mode = invalid
        for invalid in (1, "yes", None):
            with self.subTest(compile_gae=invalid):
                with self.assertRaisesRegex(ValueError, "compile_gae"):
                    buffer.compile_gae = invalid
        buffer.validation_mode = "step"
        buffer.compile_gae = True
        self.assertEqual(buffer.validation_mode, "step")
        self.assertIs(buffer.compile_gae, True)

    def test_compile_preserves_distribution_validation_when_compiler_initializes(self):
        torch = self.torch
        distribution = torch.distributions.Distribution
        initial = distribution._validate_args
        try:
            for enabled in (False, True):
                with self.subTest(enabled=enabled):
                    distribution.set_default_validate_args(enabled)

                    def compiler(*args, **kwargs):
                        distribution.set_default_validate_args(not enabled)
                        return args[0]

                    self.assertTrue(hasattr(self.module, "_compile_gae_scan"), "compile validation guard is missing")
                    with patch.object(torch, "compile", side_effect=compiler):
                        compiled = self.module._compile_gae_scan()
                    self.assertIs(compiled, self.module._gae_scan)
                    self.assertIs(distribution._validate_args, enabled)
        finally:
            distribution.set_default_validate_args(initial)

    def test_compile_preserves_distribution_validation_on_compiler_failure(self):
        torch = self.torch
        distribution = torch.distributions.Distribution
        initial = distribution._validate_args
        try:
            for enabled in (False, True):
                with self.subTest(enabled=enabled):
                    distribution.set_default_validate_args(enabled)

                    def compiler(*args, **kwargs):
                        distribution.set_default_validate_args(not enabled)
                        raise RuntimeError("compiler unavailable")

                    self.assertTrue(hasattr(self.module, "_compile_gae_scan"), "compile validation guard is missing")
                    with patch.object(torch, "compile", side_effect=compiler):
                        with self.assertRaisesRegex(RuntimeError, "compiler unavailable"):
                            self.module._compile_gae_scan()
                    self.assertIs(distribution._validate_args, enabled)
        finally:
            distribution.set_default_validate_args(initial)

    def test_rollout_add_defers_numerical_checks_but_step_mode_remains_sticky(self):
        for mode in ("step", "rollout"):
            with self.subTest(mode=mode):
                buffer = self.fill_buffer(self.new_buffer(mode=mode), bad_field="rewards")
                self.assertEqual(bool(buffer._invalid_data.item()), mode == "step")
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.boundary(buffer)

    def test_boundary_checks_all_six_fields_for_nan_and_infinities(self):
        for field in FIELDS:
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, invalid=invalid):
                    buffer = self.fill_buffer(self.new_buffer())
                    getattr(buffer, field).view(-1)[-1] = invalid
                    with self.assertRaisesRegex(ValueError, "nonfinite"):
                        self.boundary(buffer)

    def test_boundary_rejects_nonbinary_starts(self):
        for invalid in (-1.0, 0.25, 2.0):
            with self.subTest(invalid=invalid):
                buffer = self.fill_buffer(self.new_buffer())
                buffer.episode_starts[-1, -1] = invalid
                with self.assertRaisesRegex(ValueError, "non-binary"):
                    self.boundary(buffer)

    def test_repeated_boundary_validation_detects_later_mutation(self):
        buffer = self.fill_buffer(self.new_buffer())
        self.boundary(buffer)
        self.boundary(buffer)
        buffer.actions[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.boundary(buffer)
        buffer.actions[0, 0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.boundary(buffer)

    def test_repair_before_first_boundary_is_visible_only_to_rollout_mode(self):
        for mode in ("step", "rollout"):
            with self.subTest(mode=mode):
                buffer = self.fill_buffer(self.new_buffer(mode=mode), bad_field="rewards")
                buffer.rewards[0, 0] = 1.0
                if mode == "step":
                    with self.assertRaisesRegex(ValueError, "nonfinite"):
                        self.boundary(buffer)
                else:
                    self.boundary(buffer)

    def test_reset_clears_invalid_latch_and_reuses_every_storage_field(self):
        buffer = self.fill_buffer(self.new_buffer(), bad_field="values")
        pointers = {field: getattr(buffer, field).data_ptr() for field in FIELDS + ("advantages", "returns")}
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.boundary(buffer)
        buffer.reset()
        self.assertFalse(buffer._invalid_data.item())
        self.assertFalse(buffer._computed_returns_and_advantage)
        self.assertFalse(buffer.generator_ready)
        self.assertEqual(buffer.pos, 0)
        self.assertFalse(buffer.full)
        self.fill_buffer(buffer)
        self.boundary(buffer)
        for field, pointer in pointers.items():
            self.assertEqual(getattr(buffer, field).data_ptr(), pointer)

    def test_explicit_timeout_reward_finite_check_stays_active(self):
        buffer = self.fill_buffer(self.new_buffer())
        buffer._accumulate_finite_check(self.torch.tensor([float("nan")]))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.boundary(buffer)

    def test_training_boundary_rechecks_all_stored_fields_each_epoch(self):
        for field in FIELDS + ("advantages", "returns"):
            with self.subTest(field=field):
                buffer = self.fill_buffer(self.new_buffer())
                self.run_scan(buffer)
                list(buffer.get(batch_size=3))
                getattr(buffer, field).view(-1)[0] = float("nan")
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    list(buffer.get(batch_size=3))

    def test_finite_inputs_that_overflow_gae_cannot_reach_training(self):
        for mode, compiled in (("rollout", False), ("step", True)):
            with self.subTest(mode=mode, compiled=compiled):
                buffer = self.fill_buffer(self.new_buffer(mode=mode, compile_gae=compiled))
                buffer.rewards.fill_(3e38)
                buffer.episode_starts.zero_()
                self.run_scan(buffer)
                self.assertFalse(self.torch.isfinite(buffer.returns).all())
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    list(buffer.get(batch_size=3))

    def test_eager_scan_matches_original_operation_order_exactly(self):
        torch = self.torch
        for size in (1, 4, 17):
            with self.subTest(size=size):
                buffer = self.fill_buffer(self.new_buffer(size=size))
                expected = torch.empty_like(buffer.advantages)
                last_values = torch.tensor([0.4, -0.8])
                dones = torch.tensor([0.0, 1.0])
                last_gae_lam = torch.zeros((buffer.n_envs,), dtype=torch.float32)
                for step in reversed(range(buffer.buffer_size)):
                    if step == buffer.buffer_size - 1:
                        next_non_terminal = 1.0 - dones
                        next_values = last_values
                    else:
                        next_non_terminal = 1.0 - buffer.episode_starts[step + 1]
                        next_values = buffer.values[step + 1]
                    delta = buffer.rewards[step] + buffer.gamma * next_values * next_non_terminal - buffer.values[step]
                    last_gae_lam = delta + buffer.gamma * buffer.gae_lambda * next_non_terminal * last_gae_lam
                    expected[step].copy_(last_gae_lam)
                self.run_scan(buffer)
                torch.testing.assert_close(buffer.advantages, expected, rtol=0, atol=0)
                torch.testing.assert_close(buffer.returns, expected + buffer.values, rtol=0, atol=0)

    def test_validation_modes_preserve_all_fields_and_numpy_minibatch_order(self):
        torch = self.torch
        buffers = [self.fill_buffer(self.new_buffer(mode=mode)) for mode in ("step", "rollout")]
        for buffer in buffers:
            self.run_scan(buffer)
        for field in FIELDS + ("advantages", "returns"):
            torch.testing.assert_close(getattr(buffers[0], field), getattr(buffers[1], field), rtol=0, atol=0)
        epochs = []
        for buffer in buffers:
            np.random.seed(739)
            epochs.append([
                [tuple(field.clone() for field in batch) for batch in buffer.get(batch_size=3)]
                for _ in range(3)
            ])
        for left_epoch, right_epoch in zip(*epochs):
            for left_batch, right_batch in zip(left_epoch, right_epoch):
                for left, right in zip(left_batch, right_batch):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)


class RolloutBufferOptimizationCudaTests(_BufferTestCase):
    def setUp(self):
        super().setUp()
        if not self.torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")

    def compute(self, buffer, last_values=None, dones=None):
        torch = self.torch
        if last_values is None:
            last_values = torch.tensor([0.4, -0.8], device="cuda")
        if dones is None:
            dones = torch.tensor([False, True], device="cuda")
        buffer.compute_returns_and_advantage(last_values, dones)

    def test_rollout_mode_validates_every_field_before_gae(self):
        for field in FIELDS:
            with self.subTest(field=field):
                buffer = self.fill_buffer(self.new_buffer(device="cuda"))
                getattr(buffer, field).view(-1)[0] = float("nan")
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.compute(buffer)

    def test_last_values_and_dones_validation_remains_active_in_both_modes(self):
        torch = self.torch
        for mode in ("step", "rollout"):
            for field, invalid in (("last_values", float("nan")), ("last_values", float("inf")),
                                   ("dones", float("nan")), ("dones", float("inf")), ("dones", 0.25)):
                with self.subTest(mode=mode, field=field, invalid=invalid):
                    buffer = self.fill_buffer(self.new_buffer(device="cuda", mode=mode))
                    data = torch.tensor([invalid, 0.0], device="cuda")
                    with self.assertRaisesRegex(ValueError, "nonfinite|non-binary"):
                        self.compute(buffer, **{field: data})

    def test_rollout_mode_preserves_input_metadata_rejection_and_full_boundary(self):
        torch = self.torch
        buffer = self.new_buffer(device="cuda")
        with self.assertRaisesRegex(RuntimeError, "full"):
            self.compute(buffer)
        obs = torch.zeros((2, 3), device="cuda")
        actions = torch.zeros((2, 2), device="cuda")
        scalar = torch.zeros(2, device="cuda")
        for invalid, message in ((obs.cpu(), "CUDA"), (obs.double(), "float32"),
                                 (obs[:, :2], "shape"), (obs.clone().requires_grad_(), "requires_grad")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    buffer.add(invalid, actions, scalar, scalar, scalar, scalar)

    def test_boundary_mode_full_gae_and_training_samples_match_step_mode_exactly(self):
        torch = self.torch
        buffers = [self.fill_buffer(self.new_buffer(mode=mode, device="cuda")) for mode in ("step", "rollout")]
        for buffer in buffers:
            self.compute(buffer)
        for field in FIELDS + ("advantages", "returns"):
            torch.testing.assert_close(getattr(buffers[0], field), getattr(buffers[1], field), rtol=0, atol=0)
        batches = []
        for buffer in buffers:
            np.random.seed(739)
            batches.append([tuple(field.clone() for field in batch) for batch in buffer.get(3)])
        for left, right in zip(*batches):
            for left_field, right_field in zip(left, right):
                torch.testing.assert_close(left_field, right_field, rtol=0, atol=0)

    def test_compiled_scan_quantifies_numerical_difference_and_reuses_storage(self):
        torch = self.torch
        eager = self.fill_buffer(self.new_buffer(device="cuda", size=17))
        compiled = self.fill_buffer(self.new_buffer(device="cuda", size=17, compile_gae=True))
        pointers = (compiled.advantages.data_ptr(), compiled.returns.data_ptr())
        self.compute(eager)
        self.compute(compiled)
        self.assertIsNotNone(compiled._compiled_gae)
        compiled_fn = compiled._compiled_gae
        for name in ("advantages", "returns"):
            left, right = getattr(eager, name), getattr(compiled, name)
            difference = (left - right).abs().max().item()
            print(f"compiled GAE {name} max_abs_diff={difference:.9g}")
            torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-6)
        self.compute(compiled)
        self.assertIs(compiled._compiled_gae, compiled_fn)
        self.assertEqual((compiled.advantages.data_ptr(), compiled.returns.data_ptr()), pointers)

    def test_compiler_error_propagates_without_eager_fallback(self):
        buffer = self.fill_buffer(self.new_buffer(device="cuda", compile_gae=True))
        with patch.object(self.torch, "compile", side_effect=RuntimeError("compiler unavailable")):
            with self.assertRaisesRegex(RuntimeError, "compiler unavailable"):
                self.compute(buffer)
        self.assertFalse(buffer._computed_returns_and_advantage)


if __name__ == "__main__":
    unittest.main()
