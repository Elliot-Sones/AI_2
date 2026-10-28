"""Pure qualification receipt/comparison checks; these do not train on CPU."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class NativePPOQualificationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("tools.validation.qualify_native_ppo_optimizations"))
        self.q = importlib.import_module('tools.validation.qualify_native_ppo_optimizations')

    def fixture(self):
        return {
            "rollout": {name: np.arange(6, dtype=np.float32).reshape(2, 3)
                        for name in self.q.ROLLOUT_FIELDS},
            "policy": {"weight": np.ones((2, 3), dtype=np.float32)},
            "optimizer": {"state": {0: {"step": np.array(31, dtype=np.float32),
                                        "exp_avg": np.zeros((2, 3), dtype=np.float32)}},
                          "param_groups": [{"params": [0], "capturable": False, "lr": 0.0003}]},
            "rng": {"cuda": [np.arange(8, dtype=np.uint8)]},
            "n_updates": 7,
        }

    def test_exact_tree_comparison_reports_checked_arrays_and_elements(self):
        baseline = self.fixture()
        result = self.q.compare_snapshots(baseline, copy.deepcopy(baseline))
        self.assertEqual(result["max_abs_difference"], 0.0)
        self.assertGreater(result["arrays_checked"], len(self.q.ROLLOUT_FIELDS))
        self.assertGreater(result["elements_checked"], 50)

    def test_each_rollout_field_is_compared_exactly(self):
        for name in self.q.ROLLOUT_FIELDS:
            baseline, actual = self.fixture(), self.fixture()
            actual["rollout"][name].flat[0] = np.float32(1e-12)
            with self.subTest(name=name), self.assertRaisesRegex(self.q.QualificationError, name):
                self.q.compare_snapshots(baseline, actual)

    def test_shape_dtype_nonfinite_and_missing_fields_fail_closed(self):
        for replacement in (np.zeros((3, 2), dtype=np.float32),
                            np.ones((2, 3), dtype=np.float64),
                            np.full((2, 3), np.nan, dtype=np.float32),
                            np.full((2, 3), np.inf, dtype=np.float32)):
            baseline, actual = self.fixture(), self.fixture()
            actual["policy"]["weight"] = replacement
            with self.assertRaises(self.q.QualificationError):
                self.q.compare_snapshots(baseline, actual)
        actual = self.fixture()
        del actual["optimizer"]
        with self.assertRaises(self.q.QualificationError):
            self.q.compare_snapshots(self.fixture(), actual)

    def test_optimizer_step_moments_groups_and_rng_are_compared(self):
        baseline = self.fixture()
        for change in (
            lambda x: x["optimizer"]["state"][0]["step"].fill(32),
            lambda x: x["optimizer"]["state"][0]["exp_avg"].fill(0.01),
            lambda x: x["optimizer"]["param_groups"][0].update(lr=0.01),
            lambda x: x["rng"]["cuda"][0].fill(0),
        ):
            actual = copy.deepcopy(baseline)
            change(actual)
            with self.assertRaises(self.q.QualificationError):
                self.q.compare_snapshots(baseline, actual)

    def test_source_manifest_requires_all_declared_inputs(self):
        hashes = self.q.source_manifest()
        self.assertTrue(set(self.q.SOURCE_FILES) <= set(hashes))
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                self.q.source_manifest(root=Path(folder), names=("absent.py",))

    def test_parser_defaults_and_scope_are_fixed(self):
        args = self.q.build_parser().parse_args(["--checkpoint", "selected.zip", "--output", "receipt.json"])
        self.assertEqual((args.n_envs, args.n_steps, args.updates, args.batch_size, args.epochs),
                         (256, 128, 3, 8192, 8))
        arms = self.q.optimization_arms()
        self.assertEqual(list(arms), ["baseline", "optimizer_cpu_steps", "deferred_metrics",
                                     "boundary_validation", "combined"])
        for options in arms.values():
            self.assertFalse(options.deferred_timeouts)
            self.assertFalse(options.compile_loss)
            self.assertFalse(options.compile_gae)

    def test_cuda_census_rejects_cpu_parameters_or_moments(self):
        census = {"all_parameters_cuda": True, "all_moments_cuda": True,
                  "step_count": 17, "step_devices": {"cpu": 17}}
        self.q.validate_census(census, normalized=True)
        for key in ("all_parameters_cuda", "all_moments_cuda"):
            broken = dict(census, **{key: False})
            with self.assertRaises(self.q.QualificationError):
                self.q.validate_census(broken, normalized=True)
        with self.assertRaises(self.q.QualificationError):
            self.q.validate_census(dict(census, step_devices={"cuda:0": 17}), normalized=True)

    def test_receipt_pass_requires_complete_arm_updates_and_reload(self):
        arms = {}
        for name in self.q.optimization_arms():
            arms[name] = {"updates": [{"exact": True}] * 3,
                          "initial_exact": True, "save_reload": {"exact": True}}
        self.q.validate_completed_arms(arms, updates=3)
        for broken in (dict(arms, combined={"updates": [], "initial_exact": True,
                                           "save_reload": {"exact": True}}),
                       {key: value for key, value in arms.items() if key != "combined"}):
            with self.assertRaises(self.q.QualificationError):
                self.q.validate_completed_arms(broken, updates=3)

    def test_cpu_host_writes_failure_receipt_without_training(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if torch.cuda.is_available():
            self.skipTest("CPU rejection check requires a CPU host")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "receipt.json"
            code = self.q.main(["--checkpoint", str(Path(folder) / "unused.zip"), "--output", str(output)])
            receipt = json.loads(output.read_text())
        self.assertEqual(code, 1)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["skipped"], 0)
        self.assertIn("CUDA", receipt["error"])
        self.assertFalse(receipt["arms"])


if __name__ == "__main__":
    unittest.main()
