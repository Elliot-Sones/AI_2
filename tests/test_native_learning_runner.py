"""Guardrails for actual native learning, without requiring CUDA to collect tests."""

from ai2.paths import source_path
import argparse
import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import zipfile


def config():
    return {"frame_stack": 4, "opponent_history": {"enabled": False},
            "policy_kwargs": {"net_arch": [512, 512, 256]},
            "ppo_settings": {"learning_rate": 3e-4, "clip_range": .2, "nav_ent_coef": 0},
            "phases": {"0b": {"type": "navigation", "navigation": {
                "control_mode": "movement", "completion_mode": "sequence",
                "multi_target": 3, "time_limit_seconds": 20}}}}


def checkpoint(path, steps, updates):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": steps, "_n_updates": updates}))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.runner = importlib.import_module('ai2.training.native_learning_runner')

    def test_validates_all_three_architecture_counts(self):
        for arch, count in [([512, 512, 256], 1053973), ([1024] * 3, 4736021), ([1536] * 3, 10249749)]:
            params = config()
            params["policy_kwargs"]["net_arch"] = arch
            receipt = self.runner.validate_config(params, "0b", count)
            self.assertEqual(receipt["parameter_count"], count)

    def test_rejects_unsupported_navigation_before_runtime(self):
        for key, value in [("control_mode", "full"), ("completion_mode", "horizon"),
                           ("multi_target", 1), ("multi_target", 2.5),
                           ("time_limit_seconds", 0), ("time_limit_seconds", float("inf")),
                           ("time_limit_seconds", 1024 / 30 + .00001)]:
            params = config()
            params["phases"]["0b"]["navigation"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.runner.validate_config(params, "0b", 1053973)
        for key, value in [("frame_stack", 3), ("opponent_history", {"enabled": True})]:
            params = config()
            params[key] = value
            with self.assertRaises(ValueError):
                self.runner.validate_config(params, "0b", 1053973)
        with self.assertRaises(ValueError):
            self.runner.validate_config(config(), "0c", 1053973)
        with self.assertRaisesRegex(ValueError, "parameter"):
            self.runner.validate_config(config(), "0b", 4736021)

    def test_horizon_uses_ceil_at_native_capacity(self):
        params = config()
        params["phases"]["0b"]["navigation"]["time_limit_seconds"] = 1024 / 30
        self.assertEqual(self.runner.validate_config(params, "0b", 1053973)["horizon_steps"], 1024)
        params["phases"]["0b"]["navigation"]["time_limit_seconds"] = .034
        self.assertEqual(self.runner.validate_config(params, "0b", 1053973)["horizon_steps"], 2)

    def test_additional_steps_round_up_from_resumed_step(self):
        receipt = self.runner.step_expectations(5000000 - 2097152, 64, 512, 2097152)
        self.assertEqual(receipt["cycles"], 89)
        self.assertEqual(receipt["expected_final_steps"], 5013504)
        self.assertEqual(receipt["expected_additional_steps"], 2916352)
        self.assertGreater(receipt["checkpoint_interval"], 5013504)
        with self.assertRaises(ValueError):
            self.runner.step_expectations(0, 64, 512, 0)

    def test_requires_explicit_cuda_and_actual_cuda_tensor(self):
        for argv in [[], ["--device", "cpu"], ["--device", "cuda-fake"], ["--device"]]:
            with self.assertRaises(SystemExit):
                self.runner.require_cuda_arg(argv)
        self.assertEqual(self.runner.require_cuda_arg(["--device=cuda:1"]), "cuda:1")
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True),
                                      empty=lambda *a, **kw: types.SimpleNamespace(device="cpu", is_cuda=False))
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.runner.require_actual_cuda(types.SimpleNamespace(torch=torch), "cuda")

    def test_output_reuse_is_rejected_without_touching_existing_artifacts(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp).resolve() / "out"
            self.assertEqual(self.runner.validate_empty_output(output), output)
            output.mkdir()
            self.assertEqual(self.runner.validate_empty_output(output), output)
            saved = output / "final.zip"
            saved.write_bytes(b"existing model")
            with self.assertRaisesRegex(ValueError, "empty"):
                self.runner.validate_empty_output(output)
            self.assertEqual(saved.read_bytes(), b"existing model")

    def test_schedules_cannot_silently_change_across_resumed_chunks(self):
        for key, value in [("learning_rate", [3e-4, 1e-6]), ("clip_range", [.2, .05]), ("nav_ent_coef", .05)]:
            params = config()
            params["ppo_settings"][key] = value
            with self.assertRaisesRegex(ValueError, "constant"):
                self.runner.validate_config(params, "0b", 1053973)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = importlib.import_module('ai2.training.native_learning_runner')

    def exercise(self, *, resume=False, failure=False, mismatch=False, incomplete=False,
                 preupdate_final=False, validate_artifacts=False, wrong_arch=False):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "source"
            source.mkdir()
            (source_path(source, "train.py")).write_text("# frozen source\n")
            job_path = root / "train_job.py"
            job_path.write_text("# frozen job\n")
            cfg = root / "config.yaml"
            cfg.write_text(json.dumps(config()))
            output = root / "out"
            start, initial_updates = (2097152, 512) if resume else (0, 0)
            input_path = root / "resume.zip"
            if resume:
                checkpoint(input_path, start, initial_updates)

            class TensorPPO:
                def __init__(self):
                    self.num_timesteps = start
                    self._n_updates = initial_updates
                    self.policy = types.SimpleNamespace(net_arch=[256, 512, 512] if wrong_arch else [512, 512, 256], parameters=lambda: [
                        types.SimpleNamespace(device="cuda:0", numel=lambda: 1053973)])
                def collect_rollouts(self):
                    self.num_timesteps += 32768
                    return True
                def train(self):
                    self._n_updates += 8

            class Callback:
                def __init__(self, output_dir, max_seconds, checkpoint_interval=250000, verbose=0):
                    self.checkpoint_interval = checkpoint_interval

            def parser():
                result = argparse.ArgumentParser()
                for name in ["config", "phase", "device", "output", "checkpoint", "opponents"]:
                    result.add_argument("--" + name)
                for name in ["seed", "steps", "n-envs", "n-steps", "batch-size", "epochs"]:
                    result.add_argument("--" + name, type=int)
                result.add_argument("--max-seconds", type=float)
                result.add_argument("--history", action="store_true")
                return result

            job = types.SimpleNamespace(PPO=object, build_env=object, JobCallback=Callback,
                CHECKPOINT_INTERVAL=250000, build_parser=parser, yaml=types.SimpleNamespace(safe_load=json.load),
                _parameter_count=lambda agent: 1 if mismatch else 1053973,
                _write_json=lambda path, payload: Path(path).write_text(json.dumps(payload)),
                torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, synchronize=lambda *a: None),
                    empty=lambda *a, **kw: types.SimpleNamespace(device="cuda:0", is_cuda=True)))
            original_count, original_write = job._parameter_count, job._write_json
            seen = {}

            def run_job(args):
                self.assertEqual(Path.cwd(), source)
                seen["checkpoint"] = args.checkpoint
                callback = job.JobCallback(output, None)
                self.assertGreater(callback.checkpoint_interval, start + 65536)
                model = job.PPO()
                parameter_count = job._parameter_count(model)
                job._write_json(output / "metadata.json", {
                    "seed": args.seed, "checkpoint": args.checkpoint, "parameter_count": parameter_count,
                    "n_envs": args.n_envs, "ppo_controls": {"net_arch": [512, 512, 256],
                        "n_steps": args.n_steps, "batch_size": args.batch_size, "n_epochs": args.epochs,
                        "ent_coef": 0, "reset_num_timesteps": args.checkpoint is None}})
                checkpoint(output / "initial.zip", start, initial_updates)
                if failure:
                    raise RuntimeError("training failed")
                for _ in range(1 if incomplete else 2):
                    model.collect_rollouts()
                    model.train()
                for name in ["final.zip", "latest.zip"]:
                    checkpoint(output / name, model.num_timesteps, model._n_updates - (8 if preupdate_final else 0))
                job._write_json(output / "result.json", {"status": "completed", "starting_steps": start,
                    "ending_steps": model.num_timesteps, "actual_steps": model.num_timesteps - start, "wall_seconds": .5,
                    "initial_checkpoint": str(output / "initial.zip"), "latest_checkpoint": str(output / "latest.zip"),
                    "final_checkpoint": str(output / "final.zip")})
            job.run_job = run_job
            argv = ["--source", str(source), "--train-job", str(job_path), "--lane-build-dir", str(root / "build"),
                    "--expected-parameters", "1053973", "--config", str(cfg), "--phase", "0b", "--seed", "145",
                    "--steps", "40000", "--n-envs", "64", "--n-steps", "512", "--batch-size", "8192",
                    "--epochs", "8", "--device", "cuda", "--output", str(output)]
            if resume:
                argv += ["--checkpoint", str(input_path)]
            previous_cwd, previous_path = Path.cwd(), list(sys.path)
            with patch.object(self.runner.campaign, "import_train_job_module", return_value=job), patch.dict(
                    sys.modules, {"ai2.training.tensor_ppo": types.SimpleNamespace(TensorPPO=TensorPPO)}):
                code = self.runner.run(argv)
            self.assertEqual(Path.cwd(), previous_cwd)
            self.assertEqual(sys.path, previous_path)
            self.assertIs(job.PPO, object)
            self.assertIs(job.JobCallback, Callback)
            self.assertIs(job._parameter_count, original_count)
            self.assertIs(job._write_json, original_write)
            self.assertEqual(job.CHECKPOINT_INTERVAL, 250000)
            receipt = json.loads((output / "native_learning_receipt.json").read_text())
            if validate_artifacts:
                import ai2.runtime.campaign_learning as campaign_learning
                command = types.SimpleNamespace(output=output, label="native-test", metadata={
                    "rounded_steps": 65536, "expected_starting_steps": start, "expected_ending_steps": start + 65536,
                    "checkpoint": str(input_path) if resume else None, "seed": 145,
                    "config_sha256": self.runner.campaign.sha256_file(cfg), "expected_parameter_count": 1053973,
                    "net_arch": [512, 512, 256], "nav_ent_coef": 0,
                    "controls": {"n_envs": 64, "n_steps": 512, "batch_size": 8192, "epochs": 8}})
                evidence = campaign_learning.validate_train_artifacts(command)
                self.assertEqual(evidence["actual_steps"], 65536)
            return code, receipt, seen

    def test_fresh_run_is_fresh_and_measures_every_real_update(self):
        code, receipt, seen = self.exercise()
        self.assertEqual(code, 0)
        self.assertIsNone(seen["checkpoint"])
        self.assertEqual(receipt["expected_final_steps"], 65536)
        self.assertEqual(receipt["completed_cycles"], 2)
        self.assertTrue(receipt["final_completed_update_verified"])
        self.assertIsNone(receipt["files"]["checkpoint"])

    def test_resume_preserves_initial_steps_and_checkpoint_hash(self):
        code, receipt, seen = self.exercise(resume=True)
        self.assertEqual(code, 0)
        self.assertEqual(receipt["starting_steps"], 2097152)
        self.assertTrue(receipt["files"]["checkpoint"]["sha256"])

    def test_restores_patches_on_exception_or_parameter_mismatch(self):
        for kind in [{"failure": True}, {"mismatch": True}, {"wrong_arch": True}]:
            code, receipt, _ = self.exercise(**kind)
            self.assertEqual(code, 1)
            self.assertFalse(receipt["final_completed_update_verified"])

    def test_missing_update_does_not_count_as_completed(self):
        code, receipt, _ = self.exercise(incomplete=True)
        self.assertEqual(code, 2)
        self.assertFalse(receipt["final_completed_update_verified"])

    def test_preupdate_final_checkpoint_is_rejected_even_when_steps_match(self):
        code, receipt, _ = self.exercise(preupdate_final=True)
        self.assertEqual(code, 2)
        self.assertFalse(receipt["final_completed_update_verified"])

    def test_runner_artifacts_satisfy_existing_campaign_validator(self):
        for resume in (False, True):
            code, _, _ = self.exercise(resume=resume, validate_artifacts=True)
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
