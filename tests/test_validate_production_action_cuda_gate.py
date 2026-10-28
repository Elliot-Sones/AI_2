from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ai2.training.benchmark_native_contact_training as benchmark


class ProductionActionCudaGateTests(unittest.TestCase):
    def subject(self):
        return importlib.import_module('tools.validation.validate_production_action_cuda_gate')

    def test_default_without_cuda_fails_closed_and_writes_no_receipt(self):
        gate = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with patch.object(gate.torch.cuda, "is_available", return_value=False):
                code = gate.main(["--output", str(output)])
            self.assertNotEqual(code, 0)
            self.assertFalse(output.exists())

    def test_allow_cuda_skip_is_explicitly_non_gating_and_hash_bound(self):
        gate = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with patch.object(gate.torch.cuda, "is_available", return_value=False):
                code = gate.main(["--output", str(output), "--allow-cuda-skip"])
            self.assertEqual(code, 0)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "skipped")
            self.assertFalse(receipt["gating"])
            self.assertIn("--allow-cuda-skip", receipt["command"]["argv"])
            self.assertEqual(receipt["cuda"]["available"], False)
            self.assertIn("validate_production_action_cuda_gate.py", receipt["source_hashes"])
            self.assertIn("curriculum_ppo.py", receipt["source_hashes"])
            self.assertIn("native_curriculum_ppo.py", receipt["source_hashes"])
            self.assertIn("benchmark_native_contact_training.py", receipt["source_hashes"])

    def test_output_must_be_fresh_regular_path_even_for_skip(self):
        gate = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            output.write_text("old\n", encoding="utf-8")
            with patch.object(gate.torch.cuda, "is_available", return_value=False):
                code = gate.main(["--output", str(output), "--allow-cuda-skip"])
            self.assertNotEqual(code, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "old\n")

            target = Path(directory) / "target.json"
            symlink = Path(directory) / "symlink.json"
            symlink.symlink_to(target)
            with patch.object(gate.torch.cuda, "is_available", return_value=False):
                code = gate.main(["--output", str(symlink), "--allow-cuda-skip"])
            self.assertNotEqual(code, 0)
            self.assertFalse(target.exists())

    def test_gate_declares_all_production_action_arms(self):
        gate = self.subject()
        self.assertEqual(
            gate.ACTION_ARMS,
            ("G_keep", "G_neutral", "B_neutral", "B_projected"),
        )

    def test_success_path_spies_on_checkpoint_load_for_every_arm(self):
        gate = self.subject()
        calls = []

        class Learner:
            @classmethod
            def load_contact_checkpoint(cls, checkpoint, env, **kwargs):
                calls.append((checkpoint, env, kwargs))
                return SimpleNamespace(action_ab_arm=kwargs["action_ab_arm"])

        env = object()
        checkpoint = Path("canonical.zip")
        for arm in gate.ACTION_ARMS:
            model = gate.load_production_checkpoint(
                Learner,
                checkpoint=checkpoint,
                expected_sha256="a" * 64,
                env=env,
                arm=arm,
                update_mode="sb3",
            )
            self.assertEqual(model.action_ab_arm, arm)

        self.assertEqual([call[2]["action_ab_arm"] for call in calls], list(gate.ACTION_ARMS))
        self.assertTrue(all(call[0] == checkpoint for call in calls))
        self.assertTrue(all(call[2]["expected_sha256"] == "a" * 64 for call in calls))
        self.assertTrue(all(call[2]["joint"] is True for call in calls))

    def test_default_update_mode_tracks_benchmark_default(self):
        gate = self.subject()
        args = gate.parse_args(["--output", "receipt.json", "--checkpoint", "checkpoint.zip"])
        self.assertEqual(args.update_mode, benchmark.ContactBenchmarkConfig().update_mode)

    def test_fixture_uses_production_lane_geometry(self):
        gate = self.subject()
        fixture = gate._build_fixture_env(gate.torch.device("cpu"))
        lanes = benchmark.ContactBenchmarkConfig().lanes
        self.assertEqual(fixture.num_envs, lanes)
        self.assertEqual(tuple(fixture.reset_tensor().shape), (lanes, 256))

    def test_success_path_spies_on_benchmark_validate_runtime_per_arm(self):
        gate = self.subject()
        calls = []

        def validate_runtime(model, cfg):
            calls.append((model, cfg))

        for arm in gate.ACTION_ARMS:
            model = SimpleNamespace(action_ab_arm=arm)
            expected_head = benchmark.ACTION_EXPERIMENT_ARMS[arm]["action_head"]
            result = gate.validate_loaded_runtime(
                model,
                benchmark,
                arm=arm,
                update_mode=benchmark.ContactBenchmarkConfig().update_mode,
                validate_runtime_fn=validate_runtime,
            )
            self.assertEqual(result["update_mode"], benchmark.ContactBenchmarkConfig().update_mode)
            self.assertEqual(result["action_ab_arm"], arm)
            self.assertEqual(result["action_head"], expected_head)

        self.assertEqual([cfg.action_ab_arm for _model, cfg in calls], list(gate.ACTION_ARMS))
        self.assertTrue(all(cfg.update_mode == benchmark.ContactBenchmarkConfig().update_mode for _model, cfg in calls))
        self.assertTrue(all(model.action_ab_arm == cfg.action_ab_arm for model, cfg in calls))

    def test_success_path_spies_on_export_and_load_actor_mismatch_rejection(self):
        gate = self.subject()
        calls = []

        class Model:
            def export_actor(self, path):
                calls.append(("export_actor", path.name))
                path.write_bytes(b"actor")
                return {"action_head": "gaussian", "action_ab_arm": "G_keep"}

        def load_actor(path, *, device, expected_action_head=None, expected_action_ab_arm=None):
            calls.append(("load_actor", path.name, str(device), expected_action_head, expected_action_ab_arm))
            if expected_action_head == "bernoulli_10":
                raise ValueError("Exported actor action head differs from expected action head")
            if expected_action_ab_arm == "B_projected":
                raise ValueError("Exported actor action arm differs from expected action arm")
            return SimpleNamespace(action_head="gaussian"), {"action_head": "gaussian", "action_ab_arm": "G_keep"}

        with tempfile.TemporaryDirectory() as directory:
            result = gate.export_and_verify_actor(
                Model(),
                load_actor,
                arm="G_keep",
                expected_head="gaussian",
                device="cuda:0",
                work_dir=Path(directory),
            )

        self.assertEqual(result["metadata_action_head"], "gaussian")
        self.assertEqual(result["metadata_action_ab_arm"], "G_keep")
        self.assertEqual(len(result["mismatch_rejections"]), 2)
        self.assertEqual(calls[0][0], "export_actor")
        self.assertEqual([call[0] for call in calls].count("load_actor"), 3)

    def test_cuda_success_requires_checkpoint_argument(self):
        gate = self.subject()
        args = SimpleNamespace(
            checkpoint=None,
            expected_sha256=gate.CANONICAL_CHECKPOINT_SHA256,
            update_mode=benchmark.ContactBenchmarkConfig().update_mode,
            _argv=["validate_production_action_cuda_gate.py", "--output", "receipt.json"],
        )
        fake_torch = SimpleNamespace(
            device=lambda *parts: "cuda:0",
            cuda=SimpleNamespace(
                current_device=lambda: 0,
                set_device=lambda device: None,
                get_device_name=lambda device: "Fake CUDA",
            ),
            version=SimpleNamespace(cuda="fake"),
        )
        with patch.object(gate, "_require_real_torch", return_value=fake_torch):
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                gate.run_cuda_gate(args, Path.cwd())


if __name__ == "__main__":
    unittest.main()
