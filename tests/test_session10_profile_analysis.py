"""Synthetic CPU nesting and asynchronous CUDA trace attribution contracts."""
from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout


def event(name, start, duration, *, cat="cpu_op", pid=1, tid=1, **args):
    return {"name": name, "cat": cat, "ph": "X", "pid": pid, "tid": tid,
            "ts": start, "dur": duration, "args": args}


def trace_fixture():
    return {"traceEvents": [
        event("ppo_update", 0, 100, cat="user_annotation"),
        event("aten::add", 10, 5, **{"External id": 1}),
        event("cudaLaunchKernel", 11, 1, cat="cuda_runtime", correlation=11),
        event("warmup_kernel", 205, 10, cat="kernel", pid=0, tid=7, correlation=11),
        event("ppo_update", 200, 100, cat="user_annotation"),
        event("Optimizer.step#Adam.step", 210, 60, cat="user_annotation"),
        event("aten::_local_scalar_dense", 220, 20, **{"External id": 2}),
        event("cudaStreamSynchronize", 225, 2, cat="cuda_runtime"),
        event("cudaMemcpyAsync", 228, 2, cat="cuda_runtime", correlation=12),
        event("Memcpy DtoH", 405, 10, cat="gpu_memcpy", pid=0, tid=7, correlation=12),
        event("aten::add", 245, 20, **{"External id": 3}),
        event("cudaLaunchKernel", 250, 1, cat="cuda_runtime", correlation=13),
        event("update_kernel", 400, 10, cat="kernel", pid=0, tid=7, correlation=13),
        event("aten::is_nonzero", 275, 10),
        event("aten::_local_scalar_dense", 276, 8),
        event("cudaStreamSynchronize", 278, 1, cat="cuda_runtime"),
        event("aten::_local_scalar_dense", 288, 8),
        event("cudaStreamSynchronize", 290, 1, cat="cuda_runtime"),
    ]}


class Session10TraceAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("session10_profile_analysis"),
                             "session10 trace analyzer is missing")
        self.module = importlib.import_module("session10_profile_analysis")

    def analyze(self, trace=None, warmup=1):
        return self.module.analyze_trace(trace or trace_fixture(), warmup_cycles=warmup)

    def test_warmup_and_optimizer_counts_follow_cpu_launch_not_gpu_timestamp(self):
        result = self.analyze()
        warmup = result["regions"]["ppo_update"]["warmup"]
        measured = result["regions"]["ppo_update"]["measured"]
        optimizer = result["regions"]["optimizer_step"]["measured"]
        self.assertEqual(result["cycles"]["ppo_update"], {"total": 2, "warmup": 1, "measured": 1})
        self.assertEqual(warmup["gpu_event_counts"]["kernel"], 1)
        self.assertEqual(measured["gpu_event_counts"], {"kernel": 1, "gpu_memcpy": 1})
        self.assertEqual(optimizer["calls"], 1)
        self.assertEqual(optimizer["scalar_calls_by_context"], {"optimizer": 1})
        self.assertEqual(optimizer["scalar_cuda_runtime_counts"]["optimizer"],
                         {"cudaStreamSynchronize": 1, "cudaMemcpyAsync": 1})
        self.assertEqual(measured["scalar_calls_by_context"],
                         {"optimizer": 1, "boolean_validation": 1, "other": 1})
        self.assertEqual(result["attribution"]["gpu_by_correlation"], 3)

    def test_overlapping_gpu_work_has_separate_sum_and_union(self):
        measured = self.analyze()["regions"]["ppo_update"]["measured"]
        self.assertEqual(measured["gpu_duration_us_sum"], {"kernel": 10.0, "gpu_memcpy": 10.0})
        self.assertEqual(measured["gpu_active_union_us"], 15.0)
        self.assertEqual(measured["host_region_inclusive_us"], 100.0)

    def test_cpu_events_must_share_thread_and_fit_whole_region(self):
        trace = {"traceEvents": [
            event("collect_rollout", 10, 20, cat="user_annotation"),
            event("at_end", 29, 1),
            event("crosses_end", 29, 2),
            event("wrong_thread", 15, 1, tid=2),
            event("wrong_process", 15, 1, pid=2),
        ]}
        result = self.analyze(trace, warmup=0)
        counts = result["regions"]["collect_rollout"]["measured"]["cpu_op_counts"]
        self.assertEqual(counts, {"at_end": 1})
        self.assertEqual(result["unscoped"]["cpu_op_counts"],
                         {"crosses_end": 1, "wrong_thread": 1, "wrong_process": 1})

    def test_ambiguous_overlapping_outer_regions_are_not_double_attributed(self):
        trace = {"traceEvents": [
            event("collect_rollout", 0, 100, cat="user_annotation"),
            event("ppo_update", 50, 100, cat="user_annotation"),
            event("ambiguous", 60, 5),
        ]}
        result = self.analyze(trace, warmup=0)
        self.assertEqual(result["attribution"]["ambiguous_cpu_outer"], 1)
        self.assertEqual(result["unscoped"]["cpu_op_counts"]["ambiguous"], 1)
        for region in ("collect_rollout", "ppo_update"):
            self.assertNotIn("ambiguous", result["regions"][region]["measured"]["cpu_op_counts"])

    def test_external_id_fallback_attributes_gpu_work_after_cpu_range_ends(self):
        trace = {"traceEvents": [
            event("ppo_update", 0, 20, cat="user_annotation"),
            event("aten::mul", 5, 10, **{"External id": 17}),
            event("late_kernel", 500, 10, cat="kernel", pid=0, tid=8, **{"External id": 17}),
            event("unrelated_kernel", 6, 2, cat="kernel", pid=0, tid=8),
        ]}
        result = self.analyze(trace, warmup=0)
        self.assertEqual(result["attribution"]["gpu_by_external_id"], 1)
        self.assertEqual(result["attribution"]["gpu_unmatched"], 1)
        self.assertEqual(result["regions"]["ppo_update"]["measured"]["gpu_event_counts"], {"kernel": 1})
        self.assertEqual(result["unscoped"]["gpu_event_counts"], {"kernel": 1})

    def test_cuda_driver_launches_are_correlated_for_compiled_kernels(self):
        trace = {"traceEvents": [
            event("ppo_update", 0, 20, cat="user_annotation"),
            event("cuLaunchKernel", 5, 1, cat="cuda_driver", correlation=23),
            event("triton_kernel", 50, 10, cat="kernel", pid=0, correlation=23),
        ]}
        result = self.analyze(trace, warmup=0)
        measured = result["regions"]["ppo_update"]["measured"]
        self.assertEqual(result["attribution"]["gpu_by_correlation"], 1)
        self.assertEqual(measured["cuda_driver_counts"], {"cuLaunchKernel": 1})
        self.assertEqual(measured["gpu_kernel_names"], {"triton_kernel": 1})

    def test_gpu_annotations_do_not_become_cpu_cycles_or_kernel_time(self):
        trace = {"traceEvents": [
            event("ppo_update", 0, 20, cat="user_annotation"),
            event("ppo_update", 2, 100, cat="gpu_user_annotation", pid=0, tid=7),
        ]}
        result = self.analyze(trace, warmup=0)
        self.assertEqual(result["cycles"]["ppo_update"], {"total": 1, "warmup": 0, "measured": 1})
        self.assertEqual(result["regions"]["ppo_update"]["measured"]["gpu_active_union_us"], 0)
        self.assertEqual(result["attribution"]["gpu_annotation_events_excluded"], 1)

    def test_conflicting_duplicate_correlations_remain_unattributed(self):
        trace = {"traceEvents": [
            event("ppo_update", 0, 20, cat="user_annotation"),
            event("cudaLaunchKernel", 5, 1, cat="cuda_runtime", correlation=7),
            event("ppo_update", 30, 20, cat="user_annotation"),
            event("cudaLaunchKernel", 35, 1, cat="cuda_runtime", correlation=7),
            event("kernel", 60, 10, cat="kernel", pid=0, correlation=7),
        ]}
        result = self.analyze(trace)
        self.assertEqual(result["attribution"]["gpu_ambiguous"], 1)
        self.assertEqual(result["unscoped"]["gpu_event_counts"]["kernel"], 1)

    def test_rollout_and_update_warmup_cycles_are_counted_independently(self):
        trace = {"traceEvents": [
            event("collect_rollout", 0, 10, cat="user_annotation"),
            event("ppo_update", 10, 10, cat="user_annotation"),
            event("collect_rollout", 20, 10, cat="user_annotation"),
            event("ppo_update", 30, 10, cat="user_annotation"),
        ]}
        result = self.analyze(trace)
        for name in ("collect_rollout", "ppo_update"):
            self.assertEqual(result["regions"][name]["warmup"]["calls"], 1)
            self.assertEqual(result["regions"][name]["measured"]["calls"], 1)

    def test_comparison_emits_candidate_minus_baseline_without_throughput_claim(self):
        baseline = self.analyze()
        trace = trace_fixture()
        trace["traceEvents"] = [item for item in trace["traceEvents"] if item["name"] != "cudaStreamSynchronize"]
        candidate = self.analyze(trace)
        result = self.module.compare_analyses(baseline, candidate)
        measured = result["delta_candidate_minus_baseline"]["ppo_update"]["measured"]
        self.assertEqual(measured["cuda_runtime_counts"]["cudaStreamSynchronize"], -3)
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["headline_sps_eligible"])
        self.assertNotIn("speedup", result)

    def test_rejects_invalid_warmup_and_non_trace_payloads(self):
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "warmup"):
                self.analyze(warmup=invalid)
        for invalid in ({}, {"traceEvents": {}}, {"traceEvents": [], "metadata": {"capture_available": False}}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "trace|capture"):
                self.module.analyze_trace(invalid, warmup_cycles=0)

    def test_cli_writes_provenance_and_does_not_change_trace_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate, output = root / "baseline.json", root / "candidate.json", root / "counts.json"
            payload = json.dumps(trace_fixture())
            baseline.write_text(payload)
            candidate.write_text(payload)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(self.module.main(["--baseline", str(baseline), "--candidate", str(candidate),
                                                  "--output", str(output)]), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["warmup_cycles"], 1)
            self.assertEqual(len(report["inputs"]["baseline"]["sha256"]), 64)
            self.assertEqual(baseline.read_text(), payload)
            self.assertEqual(candidate.read_text(), payload)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                self.module.main(["--baseline", str(baseline), "--candidate", str(candidate), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
