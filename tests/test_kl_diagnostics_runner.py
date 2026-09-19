from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
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

    def test_main_preflights_output_and_trace_before_real_replay(self):
        module = importlib.import_module("kl_diagnostics_runner")
        original_run_replay = module.run_replay
        calls = []

        def fake_run_replay(_config):
            calls.append("called")
            return {"status": "should not be written"}

        module.run_replay = fake_run_replay
        try:
            for existing_name in ("receipt.json", "trace.jsonl"):
                with self.subTest(existing_name=existing_name), tempfile.TemporaryDirectory() as tmp:
                    output = Path(tmp) / "receipt.json"
                    trace = Path(tmp) / "trace.jsonl"
                    (Path(tmp) / existing_name).write_text("already here\n")

                    with self.assertRaisesRegex(FileExistsError, existing_name):
                        module.main([
                            "--checkpoint", "/tmp/canonical-contact-checkpoint.zip",
                            "--expected-sha256", "a" * 64,
                            "--output", str(output),
                            "--trace", str(trace),
                        ])

                    self.assertEqual(calls, [])
        finally:
            module.run_replay = original_run_replay

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
            "torch": types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: True),
                manual_seed=lambda _seed: None,
            ),
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

    def test_run_replay_seeds_python_numpy_and_torch_after_checkpoint_load_before_collection(self):
        module = importlib.import_module("kl_diagnostics_runner")
        events = []

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def manual_seed(seed):
                events.append(("torch.manual_seed", seed))

        class FakeRandom:
            @staticmethod
            def seed(seed):
                events.append(("random.seed", seed))

        class FakeNumpy:
            class random:
                @staticmethod
                def seed(seed):
                    events.append(("numpy.random.seed", seed))

        class FakeModel:
            env = object()
            rollout_buffer = object()
            n_steps = 512
            last_update_diagnostics = {"optimizer_steps": 1, "attempted_epochs": 1, "kl_early_stop": True}

            def _setup_learn(self, **kwargs):
                events.append(("_setup_learn", kwargs["total_timesteps"]))
                return None, object()

            def collect_rollouts(self, *_args):
                events.append(("collect_rollouts",))
                return True

            def train(self):
                events.append(("train",))
                self.update_diagnostic_sink({"epoch": 0, "minibatch_ordinal": 0})

        fake_model = FakeModel()
        replacements = {
            "torch": FakeTorch,
            "stable_baselines3": types.ModuleType("stable_baselines3"),
            "stable_baselines3.common": types.ModuleType("stable_baselines3.common"),
            "stable_baselines3.common.callbacks": types.SimpleNamespace(BaseCallback=type("BaseCallback", (), {})),
            "native_contact_lane_vec_env": types.SimpleNamespace(
                build_contact_env=lambda lanes, device, seed: events.append(("build_env", seed)) or object()
            ),
            "native_curriculum_ppo": types.SimpleNamespace(
                NativeCurriculumPPO=types.SimpleNamespace(
                    load_contact_checkpoint=lambda *_args, **_kwargs: events.append(("load_checkpoint",)) or fake_model
                )
            ),
            "tensor_ppo": types.SimpleNamespace(
                TensorVecEnv=lambda raw_env, device: events.append(("wrap_env", device)) or object()
            ),
        }
        previous = {name: sys.modules.get(name) for name in replacements}
        try:
            sys.modules.update(replacements)
            with mock.patch.object(module, "random", FakeRandom, create=True), \
                    mock.patch.object(module, "np", FakeNumpy, create=True), \
                    tempfile.TemporaryDirectory() as tmp:
                config = module.KLDiagnosticConfig(
                    checkpoint="/tmp/canonical-contact-checkpoint.zip",
                    expected_sha256="a" * 64,
                    output=str(Path(tmp) / "receipt.json"),
                    trace=str(Path(tmp) / "trace.jsonl"),
                    seed=987,
                )

                result = module.run_replay(config)

            self.assertEqual(result["runner"]["seed"], 987)
            self.assertEqual(result["verification"]["rng_seed_applied"], True)
            self.assertEqual(result["verification"]["rng_seed_order"], "after_checkpoint_load_before_setup_collect")
            self.assertEqual(
                events,
                [
                    ("build_env", 987),
                    ("wrap_env", "cuda"),
                    ("load_checkpoint",),
                    ("random.seed", 987),
                    ("numpy.random.seed", 987),
                    ("torch.manual_seed", 987),
                    ("_setup_learn", 32768),
                    ("collect_rollouts",),
                    ("train",),
                ],
            )
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
