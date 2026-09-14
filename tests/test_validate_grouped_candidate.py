import argparse
import importlib
from pathlib import Path
import tempfile
import unittest


class CandidateReceiptTests(unittest.TestCase):
    def module(self):
        return importlib.import_module("validate_grouped_candidate")

    def test_failures_errors_skips_and_empty_run_reject(self):
        module = self.module()
        for changes in ({"testsRun": 0}, {"failures": [1]}, {"errors": [1]}, {"skipped": [1]}):
            result = argparse.Namespace(testsRun=6, failures=[], errors=[], skipped=[])
            for name, value in changes.items():
                setattr(result, name, value)
            self.assertFalse(module.result_passed(result))
        self.assertTrue(module.result_passed(argparse.Namespace(testsRun=6, failures=[], errors=[], skipped=[])))

    def test_source_snapshot_fails_closed_and_detects_changes(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.py"
            with self.assertRaises(FileNotFoundError):
                module.snapshot([path])
            path.write_text("first")
            before = module.snapshot([path])
            path.write_text("second")
            self.assertNotEqual(before, module.snapshot([path]))

    def test_native_flags_are_explicitly_cleared_for_control_arm(self):
        # The candidate factory, not the base game, owns these opt-in flags.
        module = self.module()
        env = module.runtime_environment(Path("/source"), Path("/job.py"), Path("/candidate.py"), input_enabled=False, observation_enabled=False)
        self.assertEqual(env["AI2_NATIVE_INPUT"], "0")
        self.assertEqual(env["AI2_NATIVE_OBSERVATION"], "0")
        self.assertEqual(env["AI2_GROUPED_GAME_STEPS"], "1000")

    def test_required_sources_include_transitive_runtime_dependencies(self):
        module = self.module()
        paths = module.required_sources(Path("/source"), Path("/job.py"), Path("/candidate.py"))
        names = {path.name for path in paths}
        # environment.py imports environment.constants; train.py imports the opponent agents from environment.agent.
        for dependency in ("__init__.py", "constants.py", "agent.py"):
            self.assertIn((Path("/source") / "UTMIST-AI2-main/environment" / dependency).resolve(), paths)
        self.assertTrue({"validate_grouped_candidate.py", "benchmark_campaign.py", "batched_rng_vec_env.py", "isolated_rng_vec_env.py", "shared_action_vec_env.py", "grouped_vec_env.py", "train.py", "control.yaml", "environment.py", "test_grouped_game.py", "test_grouped_candidate_game.py"}.issubset(names))


if __name__ == "__main__":
    unittest.main()
