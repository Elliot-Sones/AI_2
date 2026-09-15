from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path


class KLDiagnosticsRunnerTests(unittest.TestCase):
    def test_dry_run_writes_bounded_replay_plan_without_requiring_gpu(self):
        module = importlib.import_module("kl_diagnostics_runner")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plan.json"

            exit_code = module.main([
                "--checkpoint", "/tmp/canonical-contact-checkpoint.zip",
                "--expected-sha256", "a" * 64,
                "--output", str(output),
                "--dry-run",
            ])

            self.assertEqual(exit_code, 0)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["schema_version"], "ai2.kl_diagnostics_replay_plan.v1")
            self.assertEqual(plan["checkpoint"]["path"], "/tmp/canonical-contact-checkpoint.zip")
            self.assertEqual(plan["checkpoint"]["expected_sha256"], "a" * 64)
            self.assertEqual(plan["geometry"]["lanes"], 64)
            self.assertEqual(plan["geometry"]["n_steps"], 512)
            self.assertEqual(plan["geometry"]["n_epochs"], 4)
            self.assertEqual(plan["geometry"]["batch_size"], 8192)
            self.assertEqual(plan["geometry"]["nominal_optimizer_steps"], 16)
            self.assertEqual(plan["ppo"]["target_kl"], 0.01)
            self.assertTrue(plan["diagnostics"]["enabled"])
            self.assertTrue(plan["dry_run"])

    def test_jsonl_sink_appends_one_diagnostic_per_line(self):
        module = importlib.import_module("kl_diagnostics_runner")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sink = module.JSONLDiagnosticSink(path)

            sink({"epoch": 0, "minibatch_ordinal": 0})
            sink({"epoch": 0, "minibatch_ordinal": 1})

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows, [
                {"epoch": 0, "minibatch_ordinal": 0},
                {"epoch": 0, "minibatch_ordinal": 1},
            ])
