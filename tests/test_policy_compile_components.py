"""Unit tests for E4 policy compile component-isolation diagnostics."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path


def require_torch():
    try:
        import torch
    except ImportError:
        raise unittest.SkipTest("torch runtime unavailable")
    return torch


class PolicyCompileComponentTests(unittest.TestCase):
    def test_worst_location_report_adds_coordinates_and_operand_scales(self):
        torch = require_torch()

        import tools.benchmarks.benchmark_policy_compile_components as components

        left = torch.tensor([[10.0, 0.0], [100.0, 7.0]], dtype=torch.float32)
        right = torch.tensor([[10.0, 3e-6], [100.0, 7.0]], dtype=torch.float32)
        report = components.located_parity_report(
            torch,
            left,
            right,
            "actions",
            base_atol=1e-5,
            operands={"mean": torch.full_like(left, 200.0), "eps_times_std": torch.full_like(left, -199.0)},
        )

        self.assertEqual(report["worst"]["coordinates"], [0, 1])
        self.assertGreater(report["max_normalized_error"], 1.0)
        self.assertEqual(report["worst"]["left"], 0.0)
        self.assertAlmostEqual(report["worst"]["operands"]["mean"], 200.0)
        self.assertAlmostEqual(report["worst"]["operands"]["eps_times_std"], -199.0)

    def test_component_result_captures_failures_without_throwing(self):
        torch = require_torch()

        import tools.benchmarks.benchmark_policy_compile_components as components

        def eager():
            return (torch.tensor([0.0], dtype=torch.float32),)

        def compiled():
            return (torch.tensor([2e-6], dtype=torch.float32),)

        result = components.compare_component(
            torch,
            name="tiny_action",
            output_names=("actions",),
            eager_call=eager,
            compiled_call=compiled,
            tolerance=1e-5,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("tensor parity failed", result["error"])
        self.assertEqual(result["reports"]["actions"]["worst"]["coordinates"], [0])

    def test_component_result_captures_exceptions_without_silent_loss(self):
        torch = require_torch()

        import tools.benchmarks.benchmark_policy_compile_components as components

        def eager():
            return (torch.tensor([0.0], dtype=torch.float32),)

        def compiled():
            raise RuntimeError("boom")

        result = components.compare_component(
            torch,
            name="exploding_component",
            output_names=("actions",),
            eager_call=eager,
            compiled_call=compiled,
            tolerance=1e-5,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("boom", result["error"])
        self.assertEqual(result["reports"], {})

    def test_mixed_arm_summary_requires_all_gates(self):
        import tools.benchmarks.benchmark_policy_compile_components as components

        results = [
            {"name": "features_latents", "status": "passed"},
            {"name": "means", "status": "passed"},
            {"name": "critic_values", "status": "passed"},
            {"name": "sampling_fixed_eager_operands", "status": "failed"},
            {"name": "logprob_fixed_actions", "status": "passed"},
            {"name": "mixed_compiled_trunk_stock_gaussian", "status": "passed"},
        ]

        summary = components.summarize_results(results)

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["failed_components"], ["sampling_fixed_eager_operands"])
        self.assertEqual(summary["passed_components"], 5)

    def test_write_summary_records_incomplete_failures(self):
        import tools.benchmarks.benchmark_policy_compile_components as components

        with tempfile.TemporaryDirectory(prefix="policy-components-") as tmp:
            payload = components.write_summary(
                Path(tmp),
                plan={"label": "component-test"},
                components=[{"name": "actions", "status": "failed", "error": "parity"}],
                provenance={"files": {"checkpoint": {"sha256": "abc"}}},
            )
            written = json.loads((Path(tmp) / "benchmark_policy_compile_components.json").read_text())

        self.assertEqual(payload["status"], "failed")
        self.assertEqual(written["components"][0]["error"], "parity")
        self.assertEqual(written["provenance"]["files"]["checkpoint"]["sha256"], "abc")

    def test_parse_args_rejects_cpu(self):
        import tools.benchmarks.benchmark_policy_compile_components as components

        with self.assertRaises(SystemExit):
            components.parse_args(["--output", "/tmp/x", "--device", "cpu"])


if __name__ == "__main__":
    unittest.main()
