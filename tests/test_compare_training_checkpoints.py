"""Tests for exact behavior-state checkpoint comparison."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))


def torch_or_skip():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("torch is required for checkpoint fixture tests") from exc
    return torch


def write_member_checkpoint(path: Path, *, policy: dict, optimizer: dict, metadata: dict) -> Path:
    torch = torch_or_skip()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        policy_path = tmp / "policy.pth"
        optimizer_path = tmp / "policy.optimizer.pth"
        torch.save(policy, policy_path)
        torch.save(optimizer, optimizer_path)
        with zipfile.ZipFile(path, "w") as archive:
            archive.write(policy_path, "policy.pth")
            archive.write(optimizer_path, "policy.optimizer.pth")
            archive.writestr("data", json.dumps(metadata))
    return path


class CompareTrainingCheckpointsTests(unittest.TestCase):
    def base_payloads(self):
        torch = torch_or_skip()
        policy = {
            "mlp_extractor.policy_net.0.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "mlp_extractor.policy_net.0.bias": torch.tensor([0.25], dtype=torch.float32),
            "action_net.weight": torch.tensor([[3.0]], dtype=torch.float64),
        }
        optimizer = {
            "state": {
                0: {
                    "step": torch.tensor(4.0),
                    "exp_avg": torch.tensor([0.5], dtype=torch.float32),
                }
            },
            "param_groups": [{"lr": 0.0003, "eps": 1e-5, "params": [0]}],
        }
        metadata = {
            "num_timesteps": 3276800,
            "_n_updates": 800,
            "n_steps": 128,
            "n_envs": 256,
            "batch_size": 8192,
            "n_epochs": 8,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "irrelevant_timestamp": "ignored",
        }
        return policy, optimizer, metadata

    def test_identical_behavior_state_passes_even_when_zip_bytes_differ(self):
        import ai2.evaluation.compare_training_checkpoints as compare

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            policy, optimizer, metadata = self.base_payloads()
            baseline = write_member_checkpoint(tmp / "baseline.zip", policy=policy, optimizer=optimizer, metadata=metadata)
            candidate = write_member_checkpoint(
                tmp / "candidate.zip",
                policy=policy,
                optimizer=optimizer,
                metadata={**metadata, "irrelevant_timestamp": "different"},
            )

            result = compare.compare_checkpoints(baseline, candidate)

        self.assertTrue(result["exact_match"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["policy"]["param_count"], 4)
        self.assertEqual(result["metadata"]["mismatches"], [])
        self.assertEqual(result["policy"]["first_mismatch"], None)
        self.assertNotEqual(result["inputs"]["baseline"]["sha256"], result["inputs"]["candidate"]["sha256"])

    def test_reports_first_tensor_value_difference_and_nonzero_exit(self):
        import ai2.evaluation.compare_training_checkpoints as compare

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            torch = torch_or_skip()
            policy, optimizer, metadata = self.base_payloads()
            candidate_policy = dict(policy)
            candidate_policy["mlp_extractor.policy_net.0.weight"] = torch.tensor([[1.0, 2.5]], dtype=torch.float32)
            baseline = write_member_checkpoint(tmp / "baseline.zip", policy=policy, optimizer=optimizer, metadata=metadata)
            candidate = write_member_checkpoint(tmp / "candidate.zip", policy=candidate_policy, optimizer=optimizer, metadata=metadata)

            result = compare.compare_checkpoints(baseline, candidate)

        self.assertFalse(result["exact_match"])
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["policy"]["first_mismatch"]["name"], "mlp_extractor.policy_net.0.weight")
        self.assertEqual(result["policy"]["first_mismatch"]["kind"], "value")
        self.assertEqual(result["policy"]["max_abs_diff"], 0.5)

    def test_rejects_dtype_metadata_and_nonfinite_mismatches(self):
        import ai2.evaluation.compare_training_checkpoints as compare

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            torch = torch_or_skip()
            policy, optimizer, metadata = self.base_payloads()
            candidate_policy = dict(policy)
            candidate_policy["action_net.weight"] = torch.tensor([[3.0]], dtype=torch.float32)
            candidate_optimizer = {
                "state": {0: {"step": torch.tensor(float("nan")), "exp_avg": torch.tensor([0.5])}},
                "param_groups": [{"lr": 0.0003, "eps": 1e-5, "params": [0]}],
            }
            candidate_metadata = {**metadata, "n_envs": 128}
            baseline = write_member_checkpoint(tmp / "baseline.zip", policy=policy, optimizer=optimizer, metadata=metadata)
            candidate = write_member_checkpoint(
                tmp / "candidate.zip",
                policy=candidate_policy,
                optimizer=candidate_optimizer,
                metadata=candidate_metadata,
            )

            result = compare.compare_checkpoints(baseline, candidate)

        self.assertFalse(result["exact_match"])
        self.assertFalse(result["finite"])
        self.assertEqual(result["policy"]["first_mismatch"]["kind"], "dtype")
        self.assertEqual(result["metadata"]["mismatches"][0]["key"], "n_envs")
        self.assertEqual(result["optimizer"]["first_mismatch"]["kind"], "nonfinite")

    def test_cli_writes_fresh_json_and_exits_nonzero_on_mismatch(self):
        import ai2.evaluation.compare_training_checkpoints as compare

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            torch = torch_or_skip()
            policy, optimizer, metadata = self.base_payloads()
            candidate_policy = {**policy, "new.weight": torch.tensor([1.0])}
            baseline = write_member_checkpoint(tmp / "baseline.zip", policy=policy, optimizer=optimizer, metadata=metadata)
            candidate = write_member_checkpoint(tmp / "candidate.zip", policy=candidate_policy, optimizer=optimizer, metadata=metadata)
            output = tmp / "compare.json"

            code = compare.main(
                [
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(code, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["exact_match"])
            self.assertIn("new.weight", report["policy"]["missing_baseline"])

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rerun = compare.main(
                    [
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--output",
                        str(output),
                    ]
                )
            self.assertNotEqual(rerun, 0)
            self.assertIn("refusing to overwrite", stderr.getvalue())

    def test_missing_checkpoint_member_is_a_clean_failure(self):
        import ai2.evaluation.compare_training_checkpoints as compare

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            torch = torch_or_skip()
            policy, optimizer, metadata = self.base_payloads()
            good = write_member_checkpoint(tmp / "good.zip", policy=policy, optimizer=optimizer, metadata=metadata)
            bad = tmp / "bad.zip"
            temp = tmp / "policy.pth"
            torch.save(policy, temp)
            with zipfile.ZipFile(bad, "w") as archive:
                archive.write(temp, "policy.pth")

            with self.assertRaisesRegex(compare.CheckpointCompareError, "policy.optimizer.pth"):
                compare.compare_checkpoints(good, bad)

    # --- hardening: cases that must not pass vacuously or on unusual values ---

    def test_metadata_key_missing_on_both_sides_fails(self):
        import ai2.evaluation.compare_training_checkpoints as compare

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            policy, optimizer, metadata = self.base_payloads()
            del metadata["num_timesteps"]
            baseline = write_member_checkpoint(tmp / "baseline.zip", policy=policy, optimizer=optimizer, metadata=metadata)
            candidate = write_member_checkpoint(
                tmp / "candidate.zip",
                policy=policy,
                optimizer=optimizer,
                metadata={**metadata, "irrelevant_timestamp": "different"},
            )

            result = compare.compare_checkpoints(baseline, candidate)

        self.assertFalse(result["exact_match"])
        self.assertEqual(result["exit_code"], 1)
        self.assertFalse(result["metadata"]["exact"])
        self.assertEqual(result["metadata"]["first_mismatch"]["key"], "num_timesteps")
        self.assertEqual(result["metadata"]["first_mismatch"]["kind"], "missing")
        self.assertIn("num_timesteps", result["metadata"]["missing_baseline"])
        self.assertIn("num_timesteps", result["metadata"]["missing_candidate"])

    def test_empty_state_sections_do_not_pass_vacuously(self):
        import ai2.evaluation.compare_training_checkpoints as compare

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            _policy, _optimizer, metadata = self.base_payloads()
            baseline = write_member_checkpoint(tmp / "baseline.zip", policy={}, optimizer={}, metadata=metadata)
            candidate = write_member_checkpoint(
                tmp / "candidate.zip",
                policy={},
                optimizer={},
                metadata={**metadata, "irrelevant_timestamp": "different"},
            )

            result = compare.compare_checkpoints(baseline, candidate)

        self.assertFalse(result["exact_match"])
        self.assertEqual(result["exit_code"], 1)
        for section in ("policy", "optimizer"):
            self.assertFalse(result[section]["exact"])
            self.assertEqual(result[section]["tensor_count"], 0)
            self.assertEqual(result[section]["first_mismatch"]["kind"], "empty")
        self.assertEqual(result["policy"]["param_count"], 0)

    def test_signed_zero_difference_is_not_exact(self):
        import ai2.evaluation.compare_training_checkpoints as compare

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            torch = torch_or_skip()
            policy, optimizer, metadata = self.base_payloads()
            policy["mlp_extractor.policy_net.0.bias"] = torch.tensor([0.0], dtype=torch.float32)
            candidate_policy = dict(policy)
            candidate_policy["mlp_extractor.policy_net.0.bias"] = torch.tensor([-0.0], dtype=torch.float32)
            baseline = write_member_checkpoint(tmp / "baseline.zip", policy=policy, optimizer=optimizer, metadata=metadata)
            candidate = write_member_checkpoint(tmp / "candidate.zip", policy=candidate_policy, optimizer=optimizer, metadata=metadata)

            result = compare.compare_checkpoints(baseline, candidate)

        self.assertFalse(result["exact_match"])
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["policy"]["max_abs_diff"], 0.0)
        self.assertEqual(result["policy"]["first_mismatch"]["name"], "mlp_extractor.policy_net.0.bias")
        self.assertEqual(result["policy"]["first_mismatch"]["kind"], "bits")
        self.assertNotEqual(result["policy"]["digest_baseline"], result["policy"]["digest_candidate"])

    def test_identical_nonfinite_values_on_both_sides_fail(self):
        import ai2.evaluation.compare_training_checkpoints as compare

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            torch = torch_or_skip()
            policy, optimizer, metadata = self.base_payloads()
            optimizer["state"][0]["step"] = torch.tensor(float("nan"))
            baseline = write_member_checkpoint(tmp / "baseline.zip", policy=policy, optimizer=optimizer, metadata=metadata)
            candidate = write_member_checkpoint(
                tmp / "candidate.zip",
                policy=policy,
                optimizer=optimizer,
                metadata={**metadata, "irrelevant_timestamp": "different"},
            )

            result = compare.compare_checkpoints(baseline, candidate)

        self.assertFalse(result["exact_match"])
        self.assertFalse(result["finite"])
        self.assertFalse(result["optimizer"]["finite"])
        mismatch = result["optimizer"]["first_mismatch"]
        self.assertEqual(mismatch["kind"], "nonfinite")
        self.assertFalse(mismatch["baseline_finite"])
        self.assertFalse(mismatch["candidate_finite"])

    def test_unsupported_array_dtype_is_a_mismatch(self):
        import ai2.evaluation.compare_training_checkpoints as compare

        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest("numpy is required") from exc

        left = {"names": np.array(["a", "b"]), "weight": np.zeros((2,), dtype=np.float32)}
        right = {"names": np.array(["a", "b"]), "weight": np.zeros((2,), dtype=np.float32)}

        section = compare.compare_state_section("policy", left, right)

        self.assertFalse(section["exact"])
        self.assertFalse(section["finite"])
        self.assertEqual(section["first_mismatch"]["name"], "names")
        self.assertEqual(section["first_mismatch"]["kind"], "unsupported_dtype")

    def test_comparing_a_checkpoint_with_itself_or_a_byte_copy_is_rejected(self):
        import ai2.evaluation.compare_training_checkpoints as compare
        import shutil

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            policy, optimizer, metadata = self.base_payloads()
            baseline = write_member_checkpoint(tmp / "baseline.zip", policy=policy, optimizer=optimizer, metadata=metadata)
            copy = tmp / "copy.zip"
            shutil.copyfile(baseline, copy)

            with self.assertRaisesRegex(compare.CheckpointCompareError, "same file"):
                compare.compare_checkpoints(baseline, baseline)
            with self.assertRaisesRegex(compare.CheckpointCompareError, "byte-identical"):
                compare.compare_checkpoints(baseline, copy)

            output = tmp / "compare.json"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = compare.main(["--baseline", str(baseline), "--candidate", str(copy), "--output", str(output)])
            self.assertEqual(code, 2)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "error")
            self.assertIn("byte-identical", report["error"])

    def test_expected_metadata_values_are_enforced(self):
        import ai2.evaluation.compare_training_checkpoints as compare

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            policy, optimizer, metadata = self.base_payloads()
            baseline = write_member_checkpoint(tmp / "baseline.zip", policy=policy, optimizer=optimizer, metadata=metadata)
            candidate = write_member_checkpoint(
                tmp / "candidate.zip",
                policy=policy,
                optimizer=optimizer,
                metadata={**metadata, "irrelevant_timestamp": "different"},
            )

            wrong = compare.compare_checkpoints(baseline, candidate, expect={"num_timesteps": 1})
            self.assertFalse(wrong["exact_match"])
            self.assertEqual(wrong["metadata"]["first_mismatch"]["key"], "num_timesteps")
            self.assertEqual(wrong["metadata"]["first_mismatch"]["kind"], "expected")
            self.assertEqual(wrong["metadata"]["first_mismatch"]["expected"], 1)

            output = tmp / "compare.json"
            code = compare.main(
                [
                    "--baseline", str(baseline), "--candidate", str(candidate), "--output", str(output),
                    "--expect", "num_timesteps=3276800", "--expect", "n_envs=256",
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["exact_match"])
            self.assertEqual(report["comparison_config"]["expected_metadata"], {"num_timesteps": 3276800, "n_envs": 256})

            with self.assertRaises(SystemExit):
                compare.main(["--baseline", str(baseline), "--candidate", str(candidate), "--output", str(tmp / "x.json"), "--expect", "bad"])


if __name__ == "__main__":
    unittest.main()
