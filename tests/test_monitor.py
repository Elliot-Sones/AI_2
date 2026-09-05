"""Tests for the instance-level GPU/CPU monitor run (offline, no network)."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MonitorOnceTest(unittest.TestCase):
    def test_once_logs_one_sample_and_writes_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env.update({"WANDB_MODE": "offline", "WANDB_DIR": tmp, "WANDB_SILENT": "true", "WANDB_CONSOLE": "off"})
            record_path = os.path.join(tmp, "monitor_run.json")
            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "monitor.py"),
                    "--once", "--name", "test-monitor", "--project", "ai2-test",
                    "--group", "run-42", "--record", record_path,
                ],
                cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            with open(record_path) as f:
                record = json.load(f)
            self.assertEqual(record["name"], "test-monitor")
            self.assertEqual(record["group"], "run-42")
            self.assertIn("id", record)
            self.assertIn("cpu/util", result.stdout)


if __name__ == "__main__":
    unittest.main()
