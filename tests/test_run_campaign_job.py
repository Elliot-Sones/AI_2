from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


class RunCampaignJobTests(unittest.TestCase):
    def subject(self):
        import run_campaign_job

        return run_campaign_job

    def make_root(self, parent: Path) -> Path:
        root = parent / "remote-root"
        (root / "source").mkdir(parents=True)
        return root

    def test_parser_requires_root_deadline_name_and_command_after_separator(self):
        m = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            args = m.parse_args([
                "--root", str(root),
                "--deadline-unix", "2000000000",
                "qualify",
                "--",
                sys.executable,
                "-c",
                "print('ok')",
            ])
            self.assertEqual(args.root, root.resolve())
            self.assertEqual(args.deadline_unix, 2000000000.0)
            self.assertEqual(args.name, "qualify")
            self.assertEqual(args.command[:2], [sys.executable, "-c"])

        with self.assertRaises(SystemExit):
            m.parse_args(["name", "--", sys.executable, "-c", "print('missing root')"])
        with self.assertRaises(SystemExit):
            m.parse_args(["--root", "/tmp/x", "--deadline-unix", "2000000000", "name"])
        with self.assertRaises(SystemExit):
            m.parse_args(["--root", "/tmp/x", "--deadline-unix", "2000000000", "../bad", "--", "true"])

    def test_job_runs_from_source_forces_offline_env_scrubs_secrets_and_writes_immutable_receipts(self):
        m = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            command = [
                sys.executable,
                "-c",
                (
                    "import json, os, pathlib; "
                    "keys=['WANDB_MODE','PYTHONPATH','AI2_FROZEN_SOURCE_ROOT','VASTAI_API_KEY',"
                    "'WANDB_API_KEY','AWS_SECRET_ACCESS_KEY','OPENAI_API_KEY','HF_TOKEN']; "
                    "pathlib.Path('env.json').write_text(json.dumps({k: os.environ.get(k) for k in keys}, sort_keys=True)); "
                    "print('child-ran')"
                ),
            ]
            with patch.dict(os.environ, {
                "VASTAI_API_KEY": "vast-secret",
                "WANDB_API_KEY": "wandb-secret",
                "AWS_SECRET_ACCESS_KEY": "aws-secret",
                "OPENAI_API_KEY": "openai-secret",
                "HF_TOKEN": "hf-secret",
                "PYTHONPATH": "/already",
            }, clear=False):
                result = m.main([
                    "--root", str(root),
                    "--deadline-unix", str(time.time() + 300),
                    "qualification",
                    "--",
                    *command,
                ])

            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["schema_version"], "ai2.campaign_job.v1")
            jobs = root / "jobs"
            log_path = jobs / "qualification.log"
            started_path = jobs / "qualification.started.json"
            receipt_path = jobs / "qualification.json"
            self.assertEqual(log_path.read_text(), "child-ran\n")
            started = json.loads(started_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(started["cwd"], str((root / "source").resolve()))
            self.assertEqual(receipt["cwd"], str((root / "source").resolve()))
            self.assertEqual(receipt["log"], str(log_path.resolve()))
            env = json.loads((root / "source" / "env.json").read_text())
            self.assertEqual(env["WANDB_MODE"], "offline")
            self.assertEqual(env["AI2_FROZEN_SOURCE_ROOT"], str((root / "inputs" / "frozen").resolve()))
            self.assertEqual(env["PYTHONPATH"], str((root / "source").resolve()) + os.pathsep + "/already")
            for key in ("VASTAI_API_KEY", "WANDB_API_KEY", "AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
                self.assertIsNone(env[key])

            with self.assertRaises(FileExistsError):
                m.main([
                    "--root", str(root),
                    "--deadline-unix", str(time.time() + 300),
                    "qualification",
                    "--",
                    sys.executable,
                    "-c",
                    "print('again')",
                ])

    def test_deadline_must_leave_timeout_reserve_before_launch(self):
        m = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            with self.assertRaisesRegex(TimeoutError, "reserve"):
                m.main([
                    "--root", str(root),
                    "--deadline-unix", str(time.time() + m.RESERVE_SECONDS - 1),
                    "too-late",
                    "--",
                    sys.executable,
                    "-c",
                    "print('should not run')",
                ])
            self.assertFalse((root / "jobs" / "too-late.started.json").exists())

    def test_nonzero_child_still_writes_started_final_receipts_and_log(self):
        m = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            result = m.main([
                "--root", str(root),
                "--deadline-unix", str(time.time() + 300),
                "failing-job",
                "--",
                sys.executable,
                "-c",
                "print('child failed before exit'); raise SystemExit(7)",
            ])

            jobs = root / "jobs"
            log_path = jobs / "failing-job.log"
            started_path = jobs / "failing-job.started.json"
            receipt_path = jobs / "failing-job.json"
            started = json.loads(started_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(result["returncode"], 7)
            self.assertEqual(receipt["returncode"], 7)
            self.assertNotIn("returncode", started)
            self.assertEqual(log_path.read_text(), "child failed before exit\n")

            with self.assertRaises(FileExistsError):
                m.main([
                    "--root", str(root),
                    "--deadline-unix", str(time.time() + 300),
                    "failing-job",
                    "--",
                    sys.executable,
                    "-c",
                    "print('should not overwrite failed receipts')",
                ])


if __name__ == "__main__":
    unittest.main()
