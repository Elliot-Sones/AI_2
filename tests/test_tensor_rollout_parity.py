"""Tests for the real-game TensorPPO one-update parity gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmark_tensor_rollout_parity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_tensor_rollout_parity", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorRolloutParityTests(unittest.TestCase):
    def complete_receipt(self, *, label="control"):
        arrays = {
            "observations": np.zeros((128, 256, 256), dtype=np.float32),
            "actions": np.zeros((128, 256, 10), dtype=np.float32),
            "episode_starts": np.zeros((128, 256), dtype=np.float32),
            "rewards": np.zeros((128, 256), dtype=np.float32),
            "values": np.zeros((128, 256), dtype=np.float32),
            "log_probs": np.zeros((128, 256), dtype=np.float32),
            "advantages": np.zeros((128, 256), dtype=np.float32),
            "returns": np.zeros((128, 256), dtype=np.float32),
        }
        return {
            "label": label,
            "status": "completed",
            "expected_steps": 32768,
            "actual_steps": 32768,
            "expected_epochs": 8,
            "expected_optimizer_steps": 32,
            "collect_calls": 1,
            "train_calls": 1,
            "n_updates_delta": 8,
            "optimizer_step_calls": 32,
            "num_timesteps": 16033824,
            "arrays": arrays,
            "counters": {
                "num_timesteps": 16033824,
                "n_updates_before": 0,
                "n_updates_after": 8,
            },
            "rng": {
                "pre_collect": {"python": "a", "numpy": "b", "torch_cpu": "c", "torch_cuda": "d"},
                "post_collect": {"python": "e", "numpy": "f", "torch_cpu": "g", "torch_cuda": "h"},
                "post_train": {"python": "i", "numpy": "j", "torch_cpu": "k", "torch_cuda": "l"},
            },
            "policy": {"w": np.zeros((4,), dtype=np.float32)},
            "optimizer": {"state.0.exp_avg": np.zeros((4,), dtype=np.float32), "param_groups": [{"lr": 0.001}]},
            "losses": {"train/policy_gradient_loss": 0.0},
            "train_result": {"status": "completed", "error": None, "stop_reason": None, "actual_steps": 32768},
        }

    def test_parser_defaults_match_real_game_gate_controls(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "out"
            args = module.validate_args(module.build_parser().parse_args(["--output", str(output.resolve())]))

        self.assertEqual(args.n_envs, 256)
        self.assertEqual(args.n_steps, 128)
        self.assertEqual(args.batch_size, 8192)
        self.assertEqual(args.epochs, 8)
        self.assertEqual(args.group_size, 4)
        self.assertEqual(args.device, "cuda")

    def test_validate_args_hydrates_real_campaign_helper_defaults(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            output = root / "out"
            args = module.validate_args(
                module.build_parser().parse_args(["--output", str(output.resolve()), "--source", str(source)])
            )
            old_notes = os.environ.pop("WANDB_NOTES", None)
            try:
                module.benchmark_campaign.configure_environment(args)
            finally:
                notes = os.environ.pop("WANDB_NOTES", "")
                if old_notes is not None:
                    os.environ["WANDB_NOTES"] = old_notes

        self.assertEqual(args.mode, "baseline")
        self.assertFalse(args.resident_buffer)
        self.assertFalse(args.rollout_policy_graph)
        self.assertEqual(args.profile_timing, "natural")
        self.assertTrue(hasattr(args, "pipeline_hooks"))
        self.assertTrue(hasattr(args, "update_hooks"))
        self.assertTrue(hasattr(args, "rollout_cache_path"))
        self.assertEqual(args.rollout_cache_path, source.resolve() / "rollout_cache.py")
        self.assertTrue(hasattr(args, "rollout_policy_graph_path"))
        self.assertIn("mode=baseline", notes)
        self.assertIn("resident_buffer=False", notes)

    def test_compare_parity_reports_first_raw_observation_divergence(self):
        module = load_module()
        control = self.complete_receipt(label="control")
        tensor = self.complete_receipt(label="tensor")
        tensor["arrays"]["observations"][0, 0, 192] = np.float32(5.0)

        report = module.compare_parity(control, tensor)

        self.assertFalse(report["parity_passed"])
        self.assertEqual(report["first_divergence"]["field"], "observations")
        self.assertEqual(report["first_divergence"]["index"], [0, 0, 192])
        self.assertEqual(report["first_divergence"]["control"], 0.0)
        self.assertEqual(report["first_divergence"]["tensor"], 5.0)

    def test_compare_parity_uses_explicit_float_tolerances(self):
        module = load_module()
        control = self.complete_receipt(label="control")
        tensor = self.complete_receipt(label="tensor")
        tensor["arrays"]["rewards"][0, 0] = np.float32(5e-7)

        ok = module.compare_parity(control, tensor)

        self.assertTrue(ok["parity_passed"])
        tensor["arrays"]["advantages"][0, 0] = np.float32(2e-5)
        bad = module.compare_parity(control, tensor)
        self.assertFalse(bad["parity_passed"])
        self.assertEqual(bad["first_divergence"]["field"], "advantages")
        self.assertEqual(bad["tolerances"]["advantages"], 1e-5)

    def test_compare_parity_rejects_nonfinite_even_when_both_sides_match(self):
        module = load_module()
        control = self.complete_receipt(label="control")
        tensor = self.complete_receipt(label="tensor")
        control["arrays"]["rewards"][0, 0] = np.float32(np.nan)
        tensor["arrays"]["rewards"][0, 0] = np.float32(np.nan)

        report = module.compare_parity(control, tensor)

        self.assertFalse(report["parity_passed"])
        self.assertEqual(report["first_divergence"]["field"], "rewards")
        self.assertEqual(report["first_divergence"]["kind"], "nonfinite")

    def test_compare_parity_rejects_empty_policy_or_optimizer_maps(self):
        module = load_module()
        control = self.complete_receipt(label="control")
        tensor = self.complete_receipt(label="tensor")
        control["policy"] = {}
        tensor["policy"] = {}

        report = module.compare_parity(control, tensor)

        self.assertFalse(report["parity_passed"])
        self.assertEqual(report["first_divergence"]["field"], "policy")

    def test_required_source_snapshot_fails_closed_on_missing_file(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = argparse.Namespace(
                source=root / "source",
                train_job=root / "scripts" / "train_job.py",
                config=root / "source" / "control.yaml",
                checkpoint=root / "checkpoint.zip",
                output=root / "out",
                grouped_env_path=root / "grouped_vec_env.py",
            )
            (root / "source").mkdir()
            (root / "scripts").mkdir()
            (root / "source" / "train.py").write_text("train", encoding="utf-8")
            (root / "source" / "UTMIST-AI2-main" / "environment").mkdir(parents=True)
            (root / "source" / "UTMIST-AI2-main" / "environment" / "environment.py").write_text("env", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "missing required input"):
                module.input_file_snapshot(args)

    def test_run_configures_campaign_environment_before_training(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            scripts = root / "scripts"
            source.mkdir()
            scripts.mkdir()
            (source / "UTMIST-AI2-main" / "environment").mkdir(parents=True)
            paths = {
                "source_train": source / "train.py",
                "source_environment": source / "UTMIST-AI2-main" / "environment" / "environment.py",
                "train_job": scripts / "train_job.py",
                "config": source / "control.yaml",
                "checkpoint": root / "selected.zip",
                "grouped_env": root / "grouped_vec_env.py",
            }
            for path in paths.values():
                path.write_text(path.name, encoding="utf-8")

            calls = []

            class FakeTrainJob:
                PPO = type("PPO", (), {})
                SubprocVecEnv = object()

                @staticmethod
                def run_job(train_args):
                    calls.append(("run_job", train_args.output))
                    out = Path(train_args.output)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "result.json").write_text(
                        json.dumps({"status": "completed", "actual_steps": 32768, "error": None, "stop_reason": None}),
                        encoding="utf-8",
                    )
                    (out / "metadata.json").write_text("{}", encoding="utf-8")

            original_configure = module.benchmark_campaign.configure_environment
            original_import = module.benchmark_campaign.import_train_job_module
            original_patch = module.benchmark_campaign.patch_grouped_vec_env
            original_make = module.make_parity_ppo_class
            try:
                module.benchmark_campaign.configure_environment = lambda args: calls.append(("configure", args.output))
                module.benchmark_campaign.import_train_job_module = lambda path: FakeTrainJob
                module.benchmark_campaign.patch_grouped_vec_env = lambda job, args: None
                module.make_parity_ppo_class = lambda base, **kwargs: base
                args = module.validate_args(
                    argparse.Namespace(
                        source=source,
                        train_job=paths["train_job"],
                        config=paths["config"],
                        checkpoint=paths["checkpoint"],
                        output=root / "out",
                        n_envs=256,
                        n_steps=128,
                        batch_size=8192,
                        epochs=8,
                        seed=45,
                        phase="0b",
                        device="cuda",
                        maxseconds=600,
                        history=False,
                        opponents=None,
                        group_size=4,
                        grouped_env_path=paths["grouped_env"],
                    )
                )
                # This fake run fails later because no real PPO captures arrays,
                # but configure_environment must happen before train_job.run_job.
                with self.assertRaises(RuntimeError):
                    module.run(args)
            finally:
                module.benchmark_campaign.configure_environment = original_configure
                module.benchmark_campaign.import_train_job_module = original_import
                module.benchmark_campaign.patch_grouped_vec_env = original_patch
                module.make_parity_ppo_class = original_make

            self.assertGreaterEqual(len(calls), 2)
            self.assertEqual(calls[0][0], "configure")
            self.assertEqual(calls[1][0], "run_job")


if __name__ == "__main__":
    unittest.main()
