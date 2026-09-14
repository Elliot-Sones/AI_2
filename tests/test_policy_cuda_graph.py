"""Dependency-light tests for the standalone E4 CUDA graph policy probe."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import benchmark_policy_cuda_graph as graph


class PolicyCudaGraphTests(unittest.TestCase):
    def test_plan_refuses_cpu_and_records_static_shapes(self):
        with self.assertRaisesRegex(ValueError, "GPU-only"):
            graph.build_plan(device="cpu")
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                graph.parse_args(["--output", "/tmp/out", "--device", "cpu"])

        plan = graph.build_plan(device="cuda", batches=[64], iterations=200, warmup=5)

        self.assertEqual(plan["label"], "e4_policy_cuda_graph_static_gaussian")
        self.assertEqual(plan["batches"], [64])
        self.assertEqual(plan["obs_dim"], 256)
        self.assertEqual(plan["action_dim"], 10)
        self.assertEqual(plan["capture"]["input_shapes"], {"obs": [64, 256], "eps": [64, 10]})
        self.assertEqual(plan["integration"], "standalone_inference_only")
        self.assertFalse(plan["claims_game_sps"])

    def test_phase_record_separates_capture_cost_and_pipeline_labels(self):
        capture = graph.phase_record(
            label="capture_setup",
            batch_size=64,
            iterations=1,
            seconds=0.25,
            transfer_mode="capture_setup",
            includes_graph_replay=True,
            includes_output_clone=False,
        )
        replay = graph.phase_record(
            label="graph_resident_replay",
            batch_size=64,
            iterations=200,
            seconds=0.10,
            transfer_mode="device_resident_static_inputs",
            includes_graph_replay=True,
            includes_output_clone=True,
            cuda_event_seconds=0.08,
        )

        self.assertEqual(capture["mean_ms_per_iteration"], 250.0)
        self.assertEqual(replay["samples"], 64 * 200)
        self.assertEqual(replay["transfer_mode"], "device_resident_static_inputs")
        self.assertTrue(replay["includes_output_clone"])
        self.assertEqual(replay["timing"]["synchronized_host_seconds"], 0.10)
        self.assertEqual(replay["timing"]["cuda_event_seconds"], 0.08)

    def test_parameter_signature_records_storage_identity(self):
        class FakeParam:
            shape = (2, 3)
            stride_value = (3, 1)
            dtype = "float32"
            device = "cuda:0"

            def data_ptr(self):
                return 123

            def stride(self):
                return self.stride_value

        class FakePolicy:
            def named_parameters(self):
                return [("layer.weight", FakeParam())]

        self.assertEqual(
            graph.parameter_storage_signature(FakePolicy()),
            [{"name": "layer.weight", "data_ptr": 123, "shape": [2, 3], "stride": [3, 1], "dtype": "float32", "device": "cuda:0"}],
        )

    def test_summary_requires_all_gate_sections(self):
        phases = [{"label": "graph_resident_replay", "status": "completed"}]
        parity = {"replay_vs_eager": {"actions": {"max_normalized_error": 0.0}}}

        with self.assertRaisesRegex(RuntimeError, "missing parity gate"):
            graph.summarize_status(phases=phases, parity=parity)

        parity.update(
            {
                "fresh_inputs": {"action_diff": 1.0},
                "permutation": {"actions": {"max_normalized_error": 0.0}},
                "weight_update": {"changed_after_update": True, "restored_after_state_restore": True},
                "finite": {"passed": True},
                "output_lifetime": {"passed": True},
                "helper_direct_sb3": {"passed": True},
                "graph_direct_sb3": {"passed": True},
            }
        )
        self.assertEqual(graph.summarize_status(phases=phases, parity=parity)["status"], "completed")

        parity["weight_update"]["changed_after_update"] = False
        with self.assertRaisesRegex(RuntimeError, "weight update"):
            graph.summarize_status(phases=phases, parity=parity)

    def test_summary_accepts_batch_aggregated_gate_sections(self):
        parity = {
            "replay_vs_eager": {"batches": {"64": {}}},
            "fresh_inputs": {"batches": {"64": {"action_diff": 1.0}}},
            "permutation": {"batches": {"64": {}}},
            "weight_update": {"batches": {"64": {"changed_after_update": True, "restored_after_state_restore": True}}},
            "finite": {"batches": {"64": {"passed": True}}},
            "output_lifetime": {"batches": {"64": {"passed": True}}},
            "helper_direct_sb3": {"batches": {"64": {"passed": True}}},
            "graph_direct_sb3": {"batches": {"64": {"passed": True}}},
        }

        self.assertEqual(graph.summarize_status(phases=[], parity=parity)["status"], "completed")

    def test_summary_requires_direct_sb3_gates(self):
        parity = {
            "replay_vs_eager": {},
            "fresh_inputs": {},
            "permutation": {},
            "weight_update": {"changed_after_update": True, "restored_after_state_restore": True},
            "finite": {"passed": True},
            "output_lifetime": {"passed": True},
            "helper_direct_sb3": {"passed": True},
            "graph_direct_sb3": {"passed": False},
        }

        with self.assertRaisesRegex(RuntimeError, "graph direct SB3"):
            graph.summarize_status(phases=[], parity=parity)

    def test_file_snapshot_requires_existing_files_and_detects_drift(self):
        with tempfile.TemporaryDirectory(prefix="cuda-graph-files-") as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.zip"
            config = root / "control.yaml"
            source = root / "source"
            source.mkdir()
            train = source / "train.py"
            helper = root / "helper.py"
            harness = root / "harness.py"
            for path in (checkpoint, config, train, helper, harness):
                path.write_text(path.name, encoding="utf-8")

            snapshot = graph.required_file_snapshot(
                checkpoint=checkpoint,
                config=config,
                source=source,
                harness=harness,
                helper=helper,
            )
            graph.assert_file_snapshot_unchanged(snapshot)
            helper.write_text("changed", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "changed during run"):
                graph.assert_file_snapshot_unchanged(snapshot)

            with self.assertRaisesRegex(RuntimeError, "missing required file"):
                graph.required_file_snapshot(
                    checkpoint=root / "missing.zip",
                    config=config,
                    source=source,
                    harness=harness,
                    helper=helper,
                )

    def test_write_summary_records_provenance_without_training_claims(self):
        with tempfile.TemporaryDirectory(prefix="cuda-graph-summary-") as tmp:
            output = Path(tmp)
            payload = graph.write_summary(
                output,
                plan=graph.build_plan(device="cuda", batches=[64]),
                phases=[],
                parity={
                    "replay_vs_eager": {},
                    "fresh_inputs": {},
                    "permutation": {},
                    "weight_update": {"changed_after_update": True, "restored_after_state_restore": True},
                    "finite": {"passed": True},
                    "output_lifetime": {"passed": True},
                    "helper_direct_sb3": {"passed": True},
                    "graph_direct_sb3": {"passed": True},
                },
                provenance={"files": {"checkpoint": {"exists": True}}},
            )

            written = json.loads((output / "benchmark_policy_cuda_graph.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["label"], "e4_policy_cuda_graph_static_gaussian")
        self.assertEqual(written["plan"]["integration"], "standalone_inference_only")
        self.assertFalse(written["plan"]["claims_game_sps"])


if __name__ == "__main__":
    unittest.main()
