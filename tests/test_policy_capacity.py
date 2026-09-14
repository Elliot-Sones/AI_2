"""Import-light checks for the GPU-only policy capacity diagnostic."""

from __future__ import annotations

import json
import inspect
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PolicyCapacityTest(unittest.TestCase):
    def test_generic_cuda_uses_current_index_and_explicit_index_is_preserved(self):
        import benchmark_policy_capacity as cap

        for device, expected in [('cuda', 1), ('cuda:0', 0)]:
            with self.subTest(device=device):
                cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 1,
                                       set_device=Mock())
                torch = SimpleNamespace(cuda=cuda, empty=Mock(),
                    device=lambda value: SimpleNamespace(index=int(value.split(':')[1]) if ':' in value else None))
                modules = {'torch': torch, 'numpy': SimpleNamespace(),
                           'gymnasium': SimpleNamespace(),
                           'stable_baselines3.common.policies': SimpleNamespace(ActorCriticPolicy=object)}
                with patch.dict(sys.modules, modules):
                    cap.require_gpu_runtime(device)
                cuda.set_device.assert_called_once_with(expected)

    def test_parameter_formula_matches_expected_sb3_box_action_counts(self):
        import benchmark_policy_capacity as cap

        expected = {
            (512, 512, 256): 1_053_973,
            (1024, 1024, 1024): 4_736_021,
            (1536, 1536, 1536): 10_249_749,
        }

        for arch, params in expected.items():
            with self.subTest(arch=arch):
                self.assertEqual(
                    cap.count_actor_critic_policy_params(
                        obs_dim=256,
                        action_dim=10,
                        net_arch=list(arch),
                    ),
                    params,
                )

    def test_default_plan_labels_gpu_only_and_synthetic_update_scope(self):
        import benchmark_policy_capacity as cap

        plan = cap.build_plan()

        self.assertEqual(plan["label"], "policy_capacity_gpu_only")
        self.assertTrue(plan["no_cpu_fallback"])
        self.assertEqual(plan["obs_space"], {"type": "Box", "shape": [256]})
        self.assertEqual(plan["action_space"], {"type": "Box", "shape": [10]})
        self.assertEqual(plan["inference_batches"], [64, 128, 256, 512])
        self.assertEqual(plan["update_batch_size"], 8192)
        self.assertEqual(plan["warmup_calls"], 20)
        self.assertEqual(plan["measured_calls"], 200)
        self.assertEqual(plan["inference_policy_mode"], "stochastic")
        self.assertIn("synthetic update", plan["notes"][0].lower())
        self.assertNotIn("full training", " ".join(plan["notes"]).lower())
        self.assertEqual(
            [item["param_count"] for item in plan["architectures"]],
            [1_053_973, 4_736_021, 10_249_749],
        )

    def test_rate_record_reports_explicit_units(self):
        import benchmark_policy_capacity as cap

        record = cap.rate_record(
            label="inference_device_resident",
            calls=20,
            batch_size=64,
            seconds=0.5,
            peak_memory_bytes=1234,
            extra={"not_sb3_training_ceiling": True},
        )

        self.assertEqual(record["samples"], 1280)
        self.assertEqual(record["calls_per_second"], 40)
        self.assertEqual(record["samples_per_second"], 2560)
        self.assertEqual(record["mean_ms_per_call"], 25)
        self.assertEqual(record["peak_memory_bytes"], 1234)
        self.assertTrue(record["not_sb3_training_ceiling"])

    def test_timed_inference_sections_do_not_force_validation_synchronization(self):
        import benchmark_policy_capacity as cap

        source = inspect.getsource(cap.measure_inference)
        for timed_section in source.split("start = time.perf_counter()")[1:]:
            measured_body = timed_section.split("seconds = time.perf_counter() - start", 1)[0]
            self.assertNotIn("assert_finite_tensor", measured_body)
            self.assertNotIn(".item(", measured_body)

    def test_synthetic_update_warmup_and_timed_body_exclude_rng_and_validation_sync(self):
        import benchmark_policy_capacity as cap

        source = inspect.getsource(cap.measure_synthetic_update)
        loss_source = inspect.getsource(cap.synthetic_ppo_loss)
        self.assertIn("for _ in range(warmup):", source)
        self.assertNotIn("range(warmup + iterations)", source)
        self.assertIn("advantages = torch.randn_like(old_log_prob)", source)
        self.assertIn("returns = torch.randn_like(old_log_prob)", source)
        self.assertNotIn("randn_like", loss_source)
        timed_body = source.split("start = time.perf_counter()", 1)[1].split(
            "seconds = time.perf_counter() - start",
            1,
        )[0]
        self.assertIn("validate=False", timed_body)
        self.assertNotIn("assert_finite_tensor", timed_body)
        self.assertNotIn(".item(", timed_body)
        self.assertNotIn("randn_like", timed_body)
        self.assertNotIn("randn(", timed_body)
        self.assertIn("policy.optimizer.zero_grad(set_to_none=True)", timed_body)
        self.assertIn("loss.backward()", timed_body)
        self.assertIn("clip_grad_norm_", timed_body)
        self.assertIn("policy.optimizer.step()", timed_body)

    def test_dry_run_cli_writes_json_without_gpu_or_sb3_imports(self):
        with tempfile.TemporaryDirectory(prefix="ai2-policy-capacity-test-") as tmp:
            output = Path(tmp) / "policy_capacity.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmark_policy_capacity.py"),
                    "--dry-run",
                    "--output",
                    str(output),
                    "--architectures",
                    "512,512,256",
                    "--batches",
                    "64,128",
                    "--measured-calls",
                    "7",
                    "--warmup-calls",
                    "3",
                    "--update-iterations",
                    "2",
                    "--update-warmup",
                    "1",
                    "--seed",
                    "123",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue(output.is_file(), result.stdout)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["mode"], "dry_run")
            self.assertEqual(payload["plan"]["inference_batches"], [64, 128])
            self.assertEqual(payload["plan"]["measured_calls"], 7)
            self.assertEqual(payload["plan"]["warmup_calls"], 3)
            self.assertEqual(payload["plan"]["update_iterations"], 2)
            self.assertEqual(payload["plan"]["update_warmup"], 1)
            self.assertEqual(payload["plan"]["seed"], 123)
            self.assertEqual(len(payload["plan"]["architectures"]), 1)
            self.assertEqual(payload["plan"]["architectures"][0]["param_count"], 1_053_973)
            self.assertIn("policy_capacity_gpu_only", result.stdout)


if __name__ == "__main__":
    unittest.main()
