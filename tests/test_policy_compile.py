"""Unit tests for E4 policy compile diagnostic planning/accounting."""

from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path


class FakeNoGrad:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        self.owner.no_grad_depth += 1

    def __exit__(self, exc_type, exc, tb):
        self.owner.no_grad_depth -= 1


class FakeCuda:
    def __init__(self):
        self.set_devices = []
        self.synced = 0

    def is_available(self):
        return True

    def set_device(self, device):
        self.set_devices.append(device)

    def synchronize(self):
        self.synced += 1

    def reset_peak_memory_stats(self):
        pass

    def max_memory_allocated(self):
        return 0


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()
        self.no_grad_depth = 0
        self.compile_called_under_no_grad = False
        self.helper_called_under_no_grad = []

    def device(self, value):
        return value

    def empty(self, *args, **kwargs):
        return object()

    def compile(self, helper, fullgraph, dynamic):
        self.compile_called_under_no_grad = self.no_grad_depth > 0

        def compiled(obs, eps):
            self.helper_called_under_no_grad.append(self.no_grad_depth > 0)
            return helper(obs, eps)

        return compiled

    def no_grad(self):
        return FakeNoGrad(self)


class NumberTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.value


class DiffTorch:
    def abs(self, value):
        return abs(value)

    def max(self, value):
        return NumberTensor(value)


class ScalarTensor:
    def __init__(self, value, ptr):
        self.value = value
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr

    def detach(self):
        return self

    def clone(self):
        return ScalarTensor(self.value, self._ptr + 1000)

    def __sub__(self, other):
        return self.value - other.value


class MathTensor:
    def __init__(self, value):
        self.value = float(value)

    def log(self):
        import math

        return MathTensor(math.log(self.value))

    def sum(self, dim=1):
        return self.value

    def __pow__(self, power):
        return MathTensor(self.value**power)

    def __sub__(self, other):
        other_value = other.value if isinstance(other, MathTensor) else other
        return MathTensor(self.value - other_value)

    def __rsub__(self, other):
        other_value = other.value if isinstance(other, MathTensor) else other
        return MathTensor(other_value - self.value)

    def __truediv__(self, other):
        other_value = other.value if isinstance(other, MathTensor) else other
        return MathTensor(self.value / other_value)

    def __rtruediv__(self, other):
        other_value = other.value if isinstance(other, MathTensor) else other
        return MathTensor(other_value / self.value)

    def __mul__(self, other):
        other_value = other.value if isinstance(other, MathTensor) else other
        return MathTensor(self.value * other_value)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return MathTensor(-self.value)


class PolicyCompileTests(unittest.TestCase):
    def test_actual_tensor_parity_gate_handles_ulp_near_zero_and_nonfinite(self):
        import ai2.training.benchmark_policy_compile as e4
        try:
            import torch
        except ImportError:
            self.skipTest("torch runtime unavailable")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for left_value, right_value, accepted in (
            (300.0, 300.0 + 2.0 ** -15, True),
            (300.0, 300.003, False),
            (0.0, 2e-6, False),
        ):
            left = torch.tensor([left_value], dtype=torch.float32, device=device)
            right = torch.tensor([right_value], dtype=torch.float32, device=device)
            report = e4.tensor_parity_report(torch, left, right, "actions", base_atol=1e-5)
            if accepted:
                e4.require_parity_reports({"actions": report})
            else:
                with self.assertRaisesRegex(RuntimeError, "parity"):
                    e4.require_parity_reports({"actions": report})
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                e4.tensor_parity_report(
                    torch, torch.tensor([value], device=device), torch.zeros(1, device=device),
                    "actions", base_atol=1e-5,
                )

    def test_parity_gate_is_scale_aware_but_rejects_real_errors(self):
        import ai2.training.benchmark_policy_compile as e4

        eps = 2.0 ** -23
        atol, rtol = e4.parity_tolerances("actions", eps, 1e-5)
        self.assertEqual(atol, 1e-6)
        self.assertEqual(rtol, 4 * eps)
        # One float32 ULP around 300 is acceptable, but not a 0.003 drift.
        self.assertLess((2.0 ** -15) / (atol + rtol * 300.0), 1.0)
        self.assertGreater(0.003 / (atol + rtol * 300.0), 1.0)
        self.assertGreater(2e-6 / (atol + rtol * 0.0), 1.0)
        with self.assertRaisesRegex(RuntimeError, "parity"):
            e4.require_parity_reports({"actions": {"max_normalized_error": 1.01}})
        for invalid in (float("nan"), float("inf")):
            with self.assertRaisesRegex(RuntimeError, "parity"):
                e4.require_parity_reports({"actions": {"max_normalized_error": invalid}})
        e4.require_parity_reports({"actions": {"max_normalized_error": 1.0}})

    def test_static_policy_param_count_matches_known_current_architecture(self):
        import ai2.training.benchmark_policy_compile as e4

        self.assertEqual(
            e4.count_actor_critic_policy_params(obs_dim=256, action_dim=10, net_arch=[512, 512, 256]),
            1_053_973,
        )

    def test_build_plan_refuses_cpu_and_requires_work(self):
        import ai2.training.benchmark_policy_compile as e4

        with self.assertRaisesRegex(ValueError, "GPU-only"):
            e4.build_plan(device="cpu")
        with self.assertRaisesRegex(ValueError, "iterations"):
            e4.build_plan(device="cuda", iterations=0)
        with self.assertRaisesRegex(ValueError, "batch"):
            e4.build_plan(device="cuda", batches=[64, 0])

    def test_rate_record_labels_rng_inclusion_and_roundtrip_mode(self):
        import ai2.training.benchmark_policy_compile as e4

        record = e4.rate_record(
            label="eager_device_resident",
            arm="eager",
            batch_size=64,
            iterations=200,
            seconds=2.0,
            rng_included=True,
            cpu_roundtrip=False,
        )

        self.assertEqual(record["samples_per_second"], 6400)
        self.assertEqual(record["rng_generation"], "inside_timed_loop")
        self.assertEqual(record["transfer_mode"], "device_resident")

    def test_storage_alias_guard_rejects_same_data_pointer(self):
        import ai2.training.benchmark_policy_compile as e4

        class TensorLike:
            def __init__(self, ptr):
                self._ptr = ptr

            def data_ptr(self):
                return self._ptr

        with self.assertRaisesRegex(RuntimeError, "alias"):
            e4.assert_no_storage_alias({"actions": TensorLike(7), "values": TensorLike(7)})

    def test_compile_failure_is_explicit_unsupported_not_successful_fallback(self):
        import ai2.training.benchmark_policy_compile as e4

        payload = e4.compile_unsupported_record(RuntimeError("graph break"))

        self.assertEqual(payload["arm"], "compiled")
        self.assertEqual(payload["status"], "unsupported")
        self.assertFalse(payload["fallback_success"])
        self.assertIn("graph break", payload["error"])

    def test_summary_writes_provenance_and_separates_compile_cost(self):
        import ai2.training.benchmark_policy_compile as e4

        with tempfile.TemporaryDirectory(prefix="e4-policy-compile-") as tmp:
            output = Path(tmp)
            payload = e4.write_summary(
                output,
                plan=e4.build_plan(device="cuda", batches=[64], iterations=200),
                phases=[
                    {"label": "eager_device_resident", "arm": "eager", "seconds": 1.0},
                    {"label": "compiled_device_resident", "arm": "compiled", "seconds": 0.5},
                ],
                compile_costs=[{"batch_size": 64, "seconds": 3.0}],
                parity=[{"batch_size": 64, "max_abs_diff": 0.0}],
                provenance={"checkpoint": {"path": "/x/model.zip", "sha256": "abc"}},
            )
            written = json.loads((output / "benchmark_policy_compile.json").read_text())

        self.assertEqual(payload["label"], "e4_policy_compile_static_gaussian")
        self.assertEqual(written["compile_costs"][0]["seconds"], 3.0)
        self.assertEqual(written["provenance"]["checkpoint"]["sha256"], "abc")

    def test_cuda_device_normalization_accepts_bare_cuda(self):
        import ai2.training.benchmark_policy_compile as e4

        self.assertEqual(e4.normalize_cuda_device("cuda"), "cuda:0")
        self.assertEqual(e4.normalize_cuda_device("cuda:1"), "cuda:1")
        with self.assertRaisesRegex(RuntimeError, "GPU-only"):
            e4.normalize_cuda_device("cpu")

    def test_compile_helper_runs_probe_under_no_grad(self):
        import ai2.training.benchmark_policy_compile as e4

        torch = FakeTorch()

        def helper(obs, eps):
            torch.helper_called_under_no_grad.append(torch.no_grad_depth > 0)
            return ("actions", "values", "log_prob")

        compiled, record = e4.compile_helper(torch, helper, object(), object(), "cuda:0")

        self.assertIsNotNone(compiled)
        self.assertEqual(record["status"], "compiled")
        self.assertTrue(torch.compile_called_under_no_grad)
        self.assertTrue(all(torch.helper_called_under_no_grad))

    def test_max_abs_diff_rejects_nonfinite_values(self):
        import ai2.training.benchmark_policy_compile as e4

        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            e4.max_abs_diff(DiffTorch(), float("nan"), 0.0)

    def test_exact_gaussian_logprob_uses_sampled_actions_not_ideal_eps_only(self):
        import ai2.training.benchmark_policy_compile as e4

        class TorchOps:
            def exp(self, value):
                return MathTensor(2.0)

        log_prob = e4.gaussian_log_prob_from_actions(
            TorchOps(),
            actions=MathTensor(5.1),
            mean_actions=MathTensor(1.0),
            log_std=0.6931471805599453,
            half_log_two_pi=0.9189385332046727,
        )

        ideal_eps_only = -0.5 * 2.0**2 - 0.6931471805599453 - 0.9189385332046727
        self.assertNotEqual(log_prob, ideal_eps_only)
        self.assertAlmostEqual(log_prob, -0.5 * ((5.1 - 1.0) / 2.0) ** 2 - 0.6931471805599453 - 0.9189385332046727)

    def test_gaussian_logprob_tensor_helper_has_no_dynamic_type_branches(self):
        import ai2.training.benchmark_policy_compile as e4

        source = inspect.getsource(e4.gaussian_log_prob_from_actions)

        self.assertNotIn("hasattr", source)
        self.assertNotIn("isinstance", source)
        self.assertNotIn("torch.log", source)

    def test_cross_call_storage_reuse_is_rejected(self):
        import ai2.training.benchmark_policy_compile as e4

        with self.assertRaisesRegex(RuntimeError, "reused output storage"):
            e4.assert_no_cross_call_storage_alias(
                (ScalarTensor(1.0, 42),),
                (ScalarTensor(2.0, 42),),
                first_label="first",
                second_label="fresh",
            )

    def test_snapshot_guard_detects_output_overwrite(self):
        import ai2.training.benchmark_policy_compile as e4

        original = ScalarTensor(1.0, 10)
        snapshot = e4.clone_output_tuple((original,))
        original.value = 2.0

        with self.assertRaisesRegex(RuntimeError, "overwritten"):
            e4.assert_output_snapshot_unchanged(DiffTorch(), snapshot, (original,), label="compiled")

    def test_fresh_epsilon_must_change_actions(self):
        import ai2.training.benchmark_policy_compile as e4

        with self.assertRaisesRegex(RuntimeError, "fresh epsilon"):
            e4.assert_fresh_action_response(
                DiffTorch(),
                ScalarTensor(1.0, 10),
                ScalarTensor(1.0, 11),
                label="compiled",
            )

        e4.assert_fresh_action_response(
            DiffTorch(),
            ScalarTensor(1.0, 10),
            ScalarTensor(2.0, 11),
            label="compiled",
        )


if __name__ == "__main__":
    unittest.main()
