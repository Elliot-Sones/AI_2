from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


class KLDiagnosticsRunnerTests(unittest.TestCase):
    def test_dry_run_writes_bounded_replay_plan_without_requiring_gpu(self):
        module = importlib.import_module("kl_diagnostics_runner")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plan.json"

            exit_code = module.main([
                "--checkpoint", "/tmp/canonical-contact-checkpoint.zip",
                "--expected-sha256", "a" * 64,
                "--output", str(output),
                "--dry-run",
            ])

            self.assertEqual(exit_code, 0)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["schema_version"], "ai2.kl_diagnostics_replay_plan.v1")
            self.assertEqual(plan["checkpoint"]["path"], "/tmp/canonical-contact-checkpoint.zip")
            self.assertEqual(plan["checkpoint"]["expected_sha256"], "a" * 64)
            self.assertEqual(plan["geometry"]["lanes"], 64)
            self.assertEqual(plan["geometry"]["n_steps"], 512)
            self.assertEqual(plan["geometry"]["n_epochs"], 4)
            self.assertEqual(plan["geometry"]["batch_size"], 8192)
            self.assertEqual(plan["geometry"]["nominal_optimizer_steps"], 16)
            self.assertEqual(plan["ppo"]["target_kl"], 0.01)
            self.assertTrue(plan["diagnostics"]["enabled"])
            self.assertTrue(plan["dry_run"])
            self.assertEqual(plan["status"], "dry_run_unverified")
            self.assertFalse(plan["verification"]["checkpoint_loaded"])
            self.assertFalse(plan["verification"]["rollouts_collected"])
            self.assertFalse(plan["verification"]["update_executed"])

    def test_sb3_update_mode_is_rejected_for_diagnostics(self):
        module = importlib.import_module("kl_diagnostics_runner")
        config = module.KLDiagnosticConfig(
            checkpoint="/tmp/canonical-contact-checkpoint.zip",
            expected_sha256="a" * 64,
            output="/tmp/plan.json",
            update_mode="sb3",
        )

        with self.assertRaisesRegex(ValueError, "deferred_metrics"):
            module.replay_plan(config)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plan.json"
            with self.assertRaisesRegex(ValueError, "deferred_metrics"):
                module.main([
                    "--checkpoint", "/tmp/canonical-contact-checkpoint.zip",
                    "--expected-sha256", "a" * 64,
                    "--output", str(output),
                    "--update-mode", "sb3",
                    "--dry-run",
                ])
            self.assertFalse(output.exists())

    def test_jsonl_sink_publishes_once_without_mixing_existing_artifacts(self):
        module = importlib.import_module("kl_diagnostics_runner")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sink = module.JSONLDiagnosticSink(path)

            sink({"epoch": 0, "minibatch_ordinal": 0})
            sink({"epoch": 0, "minibatch_ordinal": 1})
            self.assertFalse(path.exists())
            sink.close()

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows, [
                {"epoch": 0, "minibatch_ordinal": 0},
                {"epoch": 0, "minibatch_ordinal": 1},
            ])

            with self.assertRaises(FileExistsError):
                module.JSONLDiagnosticSink(path)

    def test_write_json_publishes_once_without_overwriting_existing_artifact(self):
        module = importlib.import_module("kl_diagnostics_runner")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "receipt.json"

            module.write_json(output, {"status": "first"})
            with self.assertRaises(FileExistsError):
                module.write_json(output, {"status": "second"})

            self.assertEqual(json.loads(output.read_text()), {"status": "first"})

    def test_run_replay_cleans_trace_temp_when_setup_learn_fails(self):
        module = importlib.import_module("kl_diagnostics_runner")

        class FakeModel:
            env = object()
            rollout_buffer = object()
            last_update_diagnostics = {}

            def _setup_learn(self, **_kwargs):
                raise RuntimeError("setup failed after sink creation")

        fake_model = FakeModel()
        replacements = {
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
            "stable_baselines3": types.ModuleType("stable_baselines3"),
            "stable_baselines3.common": types.ModuleType("stable_baselines3.common"),
            "stable_baselines3.common.callbacks": types.SimpleNamespace(BaseCallback=type("BaseCallback", (), {})),
            "native_contact_lane_vec_env": types.SimpleNamespace(
                build_contact_env=lambda lanes, device, seed: {"lanes": lanes, "device": device, "seed": seed}
            ),
            "native_curriculum_ppo": types.SimpleNamespace(
                NativeCurriculumPPO=types.SimpleNamespace(
                    load_contact_checkpoint=lambda *_args, **_kwargs: fake_model
                )
            ),
            "tensor_ppo": types.SimpleNamespace(
                TensorVecEnv=lambda raw_env, device: {"raw_env": raw_env, "device": device}
            ),
        }
        previous = {name: sys.modules.get(name) for name in replacements}
        try:
            sys.modules.update(replacements)
            with tempfile.TemporaryDirectory() as tmp:
                trace = Path(tmp) / "trace.jsonl"
                config = module.KLDiagnosticConfig(
                    checkpoint="/tmp/canonical-contact-checkpoint.zip",
                    expected_sha256="a" * 64,
                    output=str(Path(tmp) / "receipt.json"),
                    trace=str(trace),
                )

                with self.assertRaisesRegex(RuntimeError, "setup failed"):
                    module.run_replay(config)

                self.assertFalse(trace.exists())
                self.assertEqual(list(Path(tmp).glob(f".{trace.name}.*.tmp")), [])
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
