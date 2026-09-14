"""The curriculum gate must qualify the selected frozen phase and fail closed."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def corrected_config():
    common = {"fall_penalty": -3.0, "time_limit_seconds": 20, "multi_target": 3,
              "completion_mode": "sequence", "control_mode": "movement",
              "settle_seconds": 0.3, "post_goal_seconds": 0.5}
    return {"curriculum": {"phase": "0b"}, "phases": {
        "0a": {"type": "navigation", "navigation": dict(common, min_dist=1.5, max_dist=3)},
        "0b": {"type": "navigation", "navigation": dict(common, min_dist=2.0, max_dist=6.0)},
    }}


class CurriculumQualificationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("qualify_navigation_curriculum"),
                             "curriculum qualification command must exist")
        self.q = importlib.import_module("qualify_navigation_curriculum")

    def test_phase_selection_preserves_both_navigation_configs(self):
        original = corrected_config()
        selected, wrapper = self.q.select_phase(original, "0a")
        self.assertEqual(original["curriculum"]["phase"], "0b")
        self.assertEqual(selected["curriculum"]["phase"], "0a")
        self.assertEqual(selected["phases"], original["phases"])
        self.assertEqual(wrapper, {"min_dist": 1.5, "max_dist": 3.0, "fall_penalty": -3.0,
                         "completion_mode": "sequence", "control_mode": "movement", "settle_speed": 1.0,
                         "max_targets": 3, "time_limit": 600, "settle_steps": 9, "post_goal_required_steps": 15})
        self.assertEqual(self.q.select_phase(original, "0b")[1]["max_dist"], 6.0)

    def test_wrong_phase_contract_fails_instead_of_silently_shortening_it(self):
        for key, value in (("time_limit_seconds", 90), ("multi_target", 1),
                           ("completion_mode", "horizon"), ("control_mode", "full"),
                           ("settle_seconds", 1), ("post_goal_seconds", 3),
                           ("fall_penalty", -5), ("min_dist", 2), ("max_dist", 6)):
            config = corrected_config()
            config["phases"]["0a"]["navigation"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.q.select_phase(config, "0a")

    def test_live_wrapper_mismatch_is_rejected(self):
        expected = self.q.select_phase(corrected_config(), "0a")[1]
        self.q.require_wrapper_config(expected, dict(expected, grounded_positions=967))
        with self.assertRaisesRegex(self.q.QualificationError, "max_dist"):
            self.q.require_wrapper_config(expected, dict(expected, max_dist=6.0))

    def test_cpu_receipt_records_python_native_cases_and_hashes(self):
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tmp / "config.json"
            original = corrected_config()
            original["phases"][1] = {"type": "combat"}
            config.write_text(yaml.safe_dump(original, sort_keys=False))
            receipt = tmp / "receipt.json"
            cases = [{"status": "passed", "controller": c, "ticks": 1200, "episodes": 2,
                      "wrapper_config": self.q.select_phase(corrected_config(), "0a")[1]}
                     for c in ("adaptive", "random")]
            with patch.object(self.q, "frozen_source_manifest", return_value={"train.py": "abc"}), \
                    patch.object(self.q, "native_source_manifest", return_value={"source_hashes": {"native.c": "def"}}), \
                    patch.object(self.q, "run_python_cases", return_value=cases) as python_cases, \
                    patch.object(self.q, "run_case") as cuda_case:
                code = self.q.main(["--source", str(tmp / "frozen"), "--config", str(config),
                                    "--phase", "0a", "--build-dir", str(tmp / "build"),
                                    "--output", str(receipt), "--device", "cpu"])
            result = json.loads(receipt.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["qualification_scope"], "python_to_native_cpu")
            self.assertFalse(result["cuda_qualified"])
            self.assertEqual(result["skipped"], 0)
            self.assertEqual(result["phase"], "0a")
            self.assertEqual(result["counts"], {"python_frames": 2400, "python_episodes": 4})
            self.assertEqual(len(result["config_sha256"]), 64)
            self.assertEqual(result["frozen_source_hashes"], {"train.py": "abc"})
            self.assertEqual(yaml.safe_load(config.read_text()), original)
            self.assertEqual(python_cases.call_args.kwargs["steps"], 1200)
            cuda_case.assert_not_called()

    def test_cuda_uses_four_required_cases_after_python_parity(self):
        wrapper = self.q.select_phase(corrected_config(), "0b")[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tmp / "config.json"
            config.write_text(json.dumps(corrected_config()))
            receipt = tmp / "receipt.json"
            python_results = [{"status": "passed", "controller": c, "ticks": 1200,
                               "episodes": 2, "wrapper_config": wrapper} for c in ("adaptive", "random")]
            with patch.object(self.q, "frozen_source_manifest", return_value={}), \
                    patch.object(self.q, "native_source_manifest", return_value={}), \
                    patch.object(self.q, "_require_cuda"), \
                    patch.object(self.q, "run_python_cases", return_value=python_results), \
                    patch.object(self.q, "run_case", return_value={"status": "passed", "wrapper_config": wrapper,
                                      "counts": {"episode_records": 2}}) as cuda_case:
                code = self.q.main(["--source", str(tmp / "frozen"), "--config", str(config), "--phase", "0b",
                                    "--build-dir", str(tmp / "build"), "--output", str(receipt), "--device", "cuda"])
            self.assertEqual(code, 0)
            self.assertEqual([(c.kwargs["lanes"], c.kwargs["stream"]) for c in cuda_case.call_args_list],
                             [(64, "default"), (64, "nondefault"), (256, "default"), (256, "nondefault")])
            for call in cuda_case.call_args_list:
                self.assertEqual(call.kwargs["steps"], 1200)
                self.assertEqual(call.kwargs["wrapper_cfg"], wrapper)
            self.assertTrue(json.loads(receipt.read_text())["cuda_qualified"])

    def test_missing_cuda_or_skipped_or_incomplete_python_case_cannot_pass(self):
        for failure in ("cuda", "skip", "short", "no_episodes", "skipped_result"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                config = tmp / "config.json"
                config.write_text(json.dumps(corrected_config()))
                receipt = tmp / "receipt.json"
                wrapper = self.q.select_phase(corrected_config(), "0b")[1]
                cases = [{"status": "passed", "controller": c, "ticks": 1200, "episodes": 2,
                          "wrapper_config": wrapper} for c in ("adaptive", "random")]
                if failure == "short":
                    cases[0]["ticks"] = 1199
                if failure == "no_episodes":
                    cases[0]["episodes"] = 0
                if failure == "skipped_result":
                    cases[0]["skipped"] = 1
                error = self.q.QualificationError("actual CUDA required") if failure == "cuda" else None
                python_error = unittest.SkipTest("missing frozen runtime") if failure == "skip" else None
                with patch.object(self.q, "frozen_source_manifest", return_value={}), \
                        patch.object(self.q, "native_source_manifest", return_value={}), \
                        patch.object(self.q, "_require_cuda", side_effect=error), \
                        patch.object(self.q, "run_python_cases", return_value=cases, side_effect=python_error), \
                        patch.object(self.q, "run_case", return_value={"status": "failed"}) as cuda_case:
                    code = self.q.main(["--source", str(tmp / "frozen"), "--config", str(config), "--phase", "0b",
                                        "--build-dir", str(tmp / "build"), "--output", str(receipt), "--device", "cuda"])
                self.assertEqual(code, 1)
                result = json.loads(receipt.read_text())
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["cuda_qualified"])
                self.assertEqual(result["skipped"], 0)
                cuda_case.assert_not_called()

    def test_source_changes_invalidate_cuda_receipt(self):
        wrapper = self.q.select_phase(corrected_config(), "0b")[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tmp / "config.json"
            config.write_text(json.dumps(corrected_config()))
            receipt = tmp / "receipt.json"
            cases = [{"status": "passed", "controller": c, "ticks": 1200, "episodes": 2,
                      "wrapper_config": wrapper} for c in ("adaptive", "random")]
            with patch.object(self.q, "frozen_source_manifest", side_effect=[{"train.py": "before"}, {"train.py": "after"}]), \
                    patch.object(self.q, "native_source_manifest", return_value={}), \
                    patch.object(self.q, "_require_cuda"), \
                    patch.object(self.q, "run_python_cases", return_value=cases), \
                    patch.object(self.q, "run_case", return_value={"status": "passed", "wrapper_config": wrapper,
                                      "counts": {"episode_records": 2}}):
                code = self.q.main(["--source", str(tmp / "frozen"), "--config", str(config), "--phase", "0b",
                                    "--build-dir", str(tmp / "build"), "--output", str(receipt), "--device", "cuda"])
            result = json.loads(receipt.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["cuda_qualified"])
            self.assertIn("source changed", result["error"])


if __name__ == "__main__":
    unittest.main()
