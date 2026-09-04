"""Integration checks for benchmark accounting, using the unchanged simulator."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkAccountingTest(unittest.TestCase):
    def test_complete_ppo_updates_and_step_counts(self):
        # In SB3, stopping from on_step at the rollout boundary skips the final
        # optimizer update. A speed baseline must include that update.
        with tempfile.TemporaryDirectory(prefix="ai2-benchmark-test-") as output:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmark_training.py"),
                    "--mode", "all",
                    "--steps", "64",
                    "--warmup-steps", "8",
                    "--repeats", "1",
                    "--n-envs", "1",
                    "--n-steps", "8",
                    "--batch-size", "8",
                    "--ppo-rollouts", "2",
                    "--output", output,
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            summary_path = next(Path(output).glob("baseline_*/summary.json"))
            summary = json.loads(summary_path.read_text())
            repeats = {item["mode"]: item for item in summary["repeats"]}
            self.assertEqual(repeats["raw"]["measured_steps"], 64)
            self.assertEqual(repeats["wrapped"]["measured_steps"], 64)
            ppo = repeats["ppo"]
            self.assertEqual(ppo["measured_steps"], 16)
            self.assertEqual(len(ppo["extra"]["rollout_seconds"]), 2)
            self.assertEqual(len(ppo["extra"]["update_seconds"]), 2)
            for item in repeats.values():
                self.assertGreater(item["total_seconds"], 0)
                self.assertAlmostEqual(
                    item["steps_per_second"],
                    item["measured_steps"] / item["total_seconds"],
                )
            self.assertLessEqual(
                ppo["extra"]["rollout_seconds_total"]
                + ppo["extra"]["update_seconds_total"],
                ppo["total_seconds"],
            )


if __name__ == "__main__":
    unittest.main()
