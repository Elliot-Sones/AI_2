"""Placement-only Adam state normalization, including real CUDA update parity."""

from __future__ import annotations

import importlib
import io
import unittest


class _AdamStepTestCase(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency-light Python
            raise unittest.SkipTest(f"torch unavailable: {exc}") from exc
        self.torch = torch

    def normalize(self, optimizer):
        spec = importlib.util.find_spec("ai2.training.ppo_optimizer_counters")
        self.assertIsNotNone(spec, "Adam step-device normalization is missing")
        module = importlib.import_module('ai2.training.ppo_optimizer_counters')
        return module.normalize_adam_step_devices(optimizer)

    def initialized_optimizer(self, *, device="cpu", optimizer_type=None, **kwargs):
        torch = self.torch
        params = [
            torch.nn.Parameter(torch.linspace(-0.5, 0.5, 12, device=device).reshape(3, 4)),
            torch.nn.Parameter(torch.linspace(0.2, 0.6, 3, device=device)),
        ]
        optimizer = (optimizer_type or torch.optim.Adam)(
            params, lr=0.002, betas=(0.85, 0.99), eps=1e-5, **kwargs
        )
        for index, param in enumerate(params):
            param.grad = torch.full_like(param, 0.125 * (index + 1))
        optimizer.step()
        return params, optimizer

    def assert_same_state(self, left, right):
        torch = self.torch
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            if isinstance(left[key], torch.Tensor):
                torch.testing.assert_close(
                    left[key].cpu(), right[key].cpu(), rtol=0, atol=0
                )
            else:
                self.assertEqual(left[key], right[key])


class AdamStepDeviceTests(_AdamStepTestCase):
    def test_fresh_optimizer_does_not_create_state(self):
        torch = self.torch
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.Adam([parameter])

        self.assertEqual(self.normalize(optimizer), 0)
        self.assertEqual(len(optimizer.state), 0)

    def test_empty_parameter_group_and_empty_state_are_unchanged(self):
        torch = self.torch
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.Adam([{"params": []}, {"params": [parameter]}])
        empty_state = optimizer.state[parameter]

        self.assertEqual(self.normalize(optimizer), 0)
        self.assertIs(optimizer.state[parameter], empty_state)
        self.assertEqual(empty_state, {})

    def test_cpu_counters_are_idempotent_and_preserve_all_state(self):
        torch = self.torch
        for optimizer_type in (torch.optim.Adam, torch.optim.AdamW):
            with self.subTest(optimizer=optimizer_type.__name__):
                params, optimizer = self.initialized_optimizer(
                    optimizer_type=optimizer_type, amsgrad=True, foreach=True
                )
                original_states = {
                    param: dict(optimizer.state[param]) for param in params
                }
                original_groups = [dict(group) for group in optimizer.param_groups]
                original_values = [param.detach().clone() for param in params]
                rng_before = torch.get_rng_state().clone()

                self.assertEqual(self.normalize(optimizer), 0)
                self.assertEqual(self.normalize(optimizer), 0)

                self.assertEqual(optimizer.param_groups, original_groups)
                torch.testing.assert_close(torch.get_rng_state(), rng_before)
                for param, original_value in zip(params, original_values):
                    torch.testing.assert_close(param, original_value, rtol=0, atol=0)
                    for name, value in original_states[param].items():
                        self.assertIs(optimizer.state[param][name], value)

    def test_capturable_and_fused_groups_are_excluded_per_group(self):
        torch = self.torch
        params = [torch.nn.Parameter(torch.ones(2)) for _ in range(3)]
        optimizer = torch.optim.Adam(
            [
                {"params": [params[0]], "capturable": True},
                {"params": [params[1]], "fused": True},
                {"params": [params[2]]},
            ]
        )
        # Meta counters cannot be copied to CPU: excluded groups must not inspect
        # or materialize their state, even without CUDA on the test host.
        counters = [torch.empty((), device="meta") for _ in params[:2]]
        for param, counter in zip(params, counters):
            optimizer.state[param]["step"] = counter
        optimizer.state[params[2]]["step"] = torch.tensor(7.0)

        self.assertEqual(self.normalize(optimizer), 0)
        for param, counter in zip(params, counters):
            self.assertIs(optimizer.state[param]["step"], counter)

    def test_rejects_optimizers_without_adam_state_semantics(self):
        torch = self.torch
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        optimizer.state[parameter]["step"] = torch.tensor(2.0)

        with self.assertRaisesRegex(TypeError, "Adam"):
            self.normalize(optimizer)

    def test_rejects_missing_step_in_nonempty_state(self):
        params, optimizer = self.initialized_optimizer()
        del optimizer.state[params[1]]["step"]

        with self.assertRaisesRegex(ValueError, "step"):
            self.normalize(optimizer)

    def test_rejects_nontensor_steps(self):
        params, optimizer = self.initialized_optimizer()
        optimizer.state[params[1]]["step"] = 12

        with self.assertRaisesRegex(TypeError, "step.*tensor"):
            self.normalize(optimizer)

    def test_rejects_nonscalar_steps_without_mutation(self):
        torch = self.torch
        for shape in ((1,), (2,), (1, 1)):
            with self.subTest(shape=shape):
                params, optimizer = self.initialized_optimizer()
                counter = optimizer.state[params[0]]["step"]
                malformed = torch.ones(shape)
                optimizer.state[params[1]]["step"] = malformed

                with self.assertRaisesRegex(ValueError, "step.*scalar"):
                    self.normalize(optimizer)

                self.assertIs(optimizer.state[params[0]]["step"], counter)
                self.assertIs(optimizer.state[params[1]]["step"], malformed)

    def test_rejects_unmaterialized_step_without_mutation(self):
        torch = self.torch
        params, optimizer = self.initialized_optimizer()
        counter = optimizer.state[params[0]]["step"]
        optimizer.state[params[1]]["step"] = torch.empty((), device="meta")

        with self.assertRaisesRegex(ValueError, "step.*meta"):
            self.normalize(optimizer)

        self.assertIs(optimizer.state[params[0]]["step"], counter)

    def test_cpu_state_dict_save_reload_is_unchanged(self):
        torch = self.torch
        params, optimizer = self.initialized_optimizer(amsgrad=True)
        self.assertEqual(self.normalize(optimizer), 0)
        checkpoint = io.BytesIO()
        torch.save(optimizer.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored_params, restored = self.initialized_optimizer(amsgrad=True)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))

        self.assertEqual(self.normalize(restored), 0)
        for original, loaded in zip(params, restored_params):
            self.assert_same_state(optimizer.state[original], restored.state[loaded])


class AdamStepDeviceCudaTests(_AdamStepTestCase):
    def setUp(self):
        super().setUp()
        if not self.torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")

    def test_moves_only_scalar_counters_preserving_dtype_values_and_identity(self):
        torch = self.torch
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                params, optimizer = self.initialized_optimizer(
                    device="cuda", amsgrad=True, foreach=True
                )
                for param in params:
                    optimizer.state[param]["step"] = torch.tensor(
                        123.0, device="cuda", dtype=dtype
                    )
                original_states = {
                    param: dict(optimizer.state[param]) for param in params
                }
                original_groups = [dict(group) for group in optimizer.param_groups]
                original_values = [param.detach().clone() for param in params]
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state().clone()

                self.assertEqual(self.normalize(optimizer), len(params))
                self.assertEqual(self.normalize(optimizer), 0)

                self.assertEqual(optimizer.param_groups, original_groups)
                torch.testing.assert_close(torch.get_rng_state(), cpu_rng)
                torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng)
                for param, value in zip(params, original_values):
                    torch.testing.assert_close(param, value, rtol=0, atol=0)
                    state = optimizer.state[param]
                    self.assertEqual(state["step"].device.type, "cpu")
                    self.assertEqual(state["step"].dtype, dtype)
                    self.assertEqual(state["step"].item(), 123.0)
                    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        self.assertIs(state[name], original_states[param][name])
                    self.assert_same_state(state, original_states[param])

    def test_all_eligible_state_is_validated_before_moving_any_counter(self):
        torch = self.torch
        params, optimizer = self.initialized_optimizer(device="cuda")
        first_counter = optimizer.state[params[0]]["step"].cuda()
        optimizer.state[params[0]]["step"] = first_counter
        optimizer.state[params[1]]["step"] = torch.ones(2, device="cuda")

        with self.assertRaisesRegex(ValueError, "step.*scalar"):
            self.normalize(optimizer)

        self.assertIs(optimizer.state[params[0]]["step"], first_counter)
        self.assertEqual(first_counter.device.type, "cuda")

    def test_mixed_cuda_groups_move_only_eligible_counters(self):
        torch = self.torch
        params = [torch.nn.Parameter(torch.ones(2, device="cuda")) for _ in range(3)]
        optimizer = torch.optim.Adam(
            [
                {"params": [params[0]], "capturable": True},
                {"params": [params[1]], "fused": True},
                {"params": [params[2]]},
            ]
        )
        counters = [torch.tensor(5.0, device="cuda") for _ in params]
        for param, counter in zip(params, counters):
            optimizer.state[param]["step"] = counter

        self.assertEqual(self.normalize(optimizer), 1)
        for param, counter in zip(params[:2], counters[:2]):
            self.assertIs(optimizer.state[param]["step"], counter)
        self.assertEqual(optimizer.state[params[2]]["step"].device.type, "cpu")

    def test_cuda_remapped_checkpoint_counters_can_be_normalized_again(self):
        torch = self.torch
        params, optimizer = self.initialized_optimizer(device="cuda", amsgrad=True)
        checkpoint = io.BytesIO()
        torch.save(optimizer.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored_params, restored = self.initialized_optimizer(device="cuda", amsgrad=True)
        restored.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=True)
        )
        for param in restored_params:
            self.assertEqual(restored.state[param]["step"].device.type, "cuda")

        self.assertEqual(self.normalize(restored), len(restored_params))
        self.assertEqual(self.normalize(restored), 0)
        for original, loaded in zip(params, restored_params):
            self.assert_same_state(optimizer.state[original], restored.state[loaded])

    def test_identical_gradient_updates_match_cuda_counter_baseline_exactly(self):
        torch = self.torch
        for foreach in (None, False, True):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(foreach=foreach, counter_dtype=dtype):
                    baseline_params, baseline = self.initialized_optimizer(
                        device="cuda", amsgrad=True, foreach=foreach
                    )
                    candidate_params, candidate = self.initialized_optimizer(
                        device="cuda", amsgrad=True, foreach=foreach
                    )
                    for params, optimizer in (
                        (baseline_params, baseline), (candidate_params, candidate)
                    ):
                        for param in params:
                            optimizer.state[param]["step"] = torch.tensor(
                                21.0, device="cuda", dtype=dtype
                            )

                    self.assertEqual(self.normalize(candidate), len(candidate_params))
                    for update in range(5):
                        for index, (left, right) in enumerate(
                            zip(baseline_params, candidate_params)
                        ):
                            gradient = torch.sin(
                                torch.arange(left.numel(), device="cuda").reshape_as(left)
                                + update + index * 0.25
                            )
                            left.grad = gradient.clone()
                            right.grad = gradient.clone()
                        baseline.step()
                        candidate.step()
                        for left, right in zip(baseline_params, candidate_params):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                            self.assert_same_state(baseline.state[left], candidate.state[right])


if __name__ == "__main__":
    unittest.main()
