from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ProductionActionCudaGateTests(unittest.TestCase):
    def subject(self):
        return importlib.import_module("validate_production_action_cuda_gate")

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

    def test_gate_source_uses_production_policy_ppo_and_export_load_path(self):
        gate = self.subject()
        source = inspect.getsource(gate)
        self.assertIn("CurriculumPolicy", source)
        self.assertIn("NativeCurriculumPPO", source)
        self.assertIn("benchmark_native_contact_training", source)
        self.assertIn("export_actor", source)
        self.assertIn("load_actor", source)
        self.assertIn("load_contact_checkpoint", source)
        self.assertNotIn("GaussianBox10", source)
        self.assertNotIn("Bernoulli10", source)
        self.assertNotIn("class Toy", source)


if __name__ == "__main__":
    unittest.main()
