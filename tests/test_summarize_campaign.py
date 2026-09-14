"""Tests for paired campaign summarization."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
import zipfile

import summarize_campaign as summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class SummarizeCampaignTest(unittest.TestCase):
    def make_run(self, root: Path, seed: int, arm: str, sps: float, start: float = 1000.0) -> Path:
        run = root / f"paired-cache-seed{seed}-{arm}"
        transitions = 100_000
        cycle_seconds = 500.0 / sps
        measured_seconds = transitions / sps
        row_transitions = transitions // 2
        write_json(
            run / "campaign_summary.json",
            {
                "status": "completed",
                "startup_seconds": 2.0,
                "learning_wall_seconds": 20.0,
                "measured_cycle_seconds": measured_seconds,
                "measured_collect_seconds": 6.0,
                "measured_update_seconds": 3.0,
                "measured_boundary_seconds": 0.25,
                "measured_transitions": transitions,
                "measured_transitions_per_second": sps,
                "completed_measured_cycles": 2,
                "plan": {"warmup": 1, "cycles": 2, "n_envs": 64, "n_steps": 512},
            },
        )
        write_jsonl(
            run / "campaign_metrics.jsonl",
            [
                {
                    "phase": "measured",
                    "cycle_index": 2,
                    "timestamp_unix": start + cycle_seconds,
                    "cycle_seconds": cycle_seconds,
                    "transitions": row_transitions,
                    "transitions_per_second": row_transitions / cycle_seconds,
                    "rollout_seconds": 3.0,
                    "update_seconds": 1.5,
                    "boundary_seconds": 0.1,
                },
                {
                    "phase": "measured",
                    "cycle_index": 3,
                    "timestamp_unix": start + measured_seconds,
                    "cycle_seconds": measured_seconds - cycle_seconds,
                    "transitions": transitions - row_transitions,
                    "transitions_per_second": (transitions - row_transitions) / (measured_seconds - cycle_seconds),
                    "rollout_seconds": 3.0,
                    "update_seconds": 1.5,
                    "boundary_seconds": 0.15,
                },
            ],
        )
        return run

    def test_fail_closed_when_expected_pairs_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root, 45, "cache", 120.0)

            with self.assertRaisesRegex(summary.SummaryError, "missing run"):
                summary.summarize_campaign(root, expected_seeds=[45], allow_incomplete=False)

    def test_strict_rejects_completed_summary_with_missing_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root, 45, "control", 100.0)
            self.make_run(root, 45, "cache", 125.0)
            (run / "campaign_metrics.jsonl").unlink()

            with self.assertRaisesRegex(summary.SummaryError, "measured row count"):
                summary.summarize_campaign(root, expected_seeds=[45], allow_incomplete=False)

    def test_strict_rejects_metric_count_index_transition_and_sps_mismatches(self):
        corruptions = [
            ("missing measured row", lambda rows: rows[:1], "measured row count"),
            ("duplicate cycle index", lambda rows: [dict(rows[0]), dict(rows[0])], "contiguous unique"),
            ("bad transitions", lambda rows: [dict(rows[0], transitions=1), rows[1]], "transitions"),
            ("bad seconds", lambda rows: [dict(rows[0], cycle_seconds=999.0), rows[1]], "timing"),
            ("bad sps", lambda rows: [dict(rows[0], transitions_per_second=1.0), rows[1]], "SPS"),
        ]
        for _label, mutate, pattern in corruptions:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = self.make_run(root, 45, "control", 100.0)
                self.make_run(root, 45, "cache", 125.0)
                rows = [json.loads(line) for line in (run / "campaign_metrics.jsonl").read_text().splitlines()]
                write_jsonl(run / "campaign_metrics.jsonl", mutate(rows))

                with self.assertRaisesRegex(summary.SummaryError, pattern):
                    summary.summarize_campaign(root, expected_seeds=[45], allow_incomplete=False)

    def test_partial_mode_labels_invalid_completed_runs_and_does_not_pair_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root, 45, "control", 100.0)
            run = self.make_run(root, 45, "cache", 125.0)
            (run / "campaign_metrics.jsonl").unlink()

            result = summary.summarize_campaign(root, expected_seeds=[45], allow_incomplete=True)

        invalid = [run for run in result["runs"] if run["arm"] == "cache"][0]
        self.assertEqual(invalid["status"], "invalid")
        self.assertIn("validation_error", invalid)
        self.assertEqual(result["pairs"], [])

    def test_pairs_ratios_and_geomean_across_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, control_sps, cache_sps in [(45, 100.0, 110.0), (46, 200.0, 250.0), (47, 400.0, 500.0)]:
                self.make_run(root, seed, "control", control_sps, start=1000.0 + seed)
                self.make_run(root, seed, "cache", cache_sps, start=2000.0 + seed)

            result = summary.summarize_campaign(root, expected_seeds=[45, 46, 47])

        ratios = [pair["cache_over_control_sps"] for pair in result["pairs"]]
        self.assertEqual([pair["seed"] for pair in result["pairs"]], [45, 46, 47])
        self.assertEqual(ratios, [1.1, 1.25, 1.25])
        self.assertEqual(result["aggregate"]["n_pairs"], 3)
        self.assertAlmostEqual(result["aggregate"]["median_cache_over_control_sps"], 1.25)
        self.assertAlmostEqual(result["aggregate"]["geomean_cache_over_control_sps"], (1.1 * 1.25 * 1.25) ** (1 / 3))
        self.assertNotIn("confidence_interval", result["aggregate"])

    def test_monitor_samples_are_correlated_to_measured_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root, 45, "control", 10_000.0, start=1000.0)
            self.make_run(root, 45, "cache", 12_500.0, start=1010.0)
            write_json(root.parent / "monitor_run.json", {"started_at": "1970-01-01T00:16:30"})
            write_jsonl(
                root.parent / "monitor.log",
                [
                    {"monitor/uptime_s": 5.0, "gpu/util": 1.0, "gpu/mem_used_mb": 100.0},
                    {"monitor/uptime_s": 10.0, "gpu/util": 10.0, "gpu/mem_used_mb": 200.0},
                    {"monitor/uptime_s": 15.0, "gpu/util": 20.0, "gpu/mem_used_mb": 300.0},
                    {"monitor/uptime_s": 40.0, "gpu/util": 99.0, "gpu/mem_used_mb": 900.0},
                ],
            )

            result = summary.summarize_campaign(root, expected_seeds=[45])

        gpu = result["monitor"]["measured_window_gpu"]
        self.assertEqual(gpu["sample_count"], 2)
        self.assertEqual(gpu["util_avg"], 15.0)
        self.assertEqual(gpu["util_p50"], 15.0)
        self.assertEqual(gpu["util_p95"], 19.5)
        self.assertEqual(gpu["util_max"], 20.0)
        self.assertEqual(gpu["vram_peak_mb"], 300.0)

    def test_profile_breakdown_has_named_stages_and_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root, 45, "control", 100.0)
            self.make_run(root, 45, "cache", 125.0)

            result = summary.summarize_campaign(root, expected_seeds=[45])

        breakdown = [run for run in result["runs"] if run["arm"] == "control"][0]["profile_breakdown"]
        self.assertEqual([stage["stage"] for stage in breakdown], ["collect", "update", "boundary", "residual"])
        self.assertAlmostEqual(breakdown[-1]["seconds"], 990.75)

    def test_summarize_path_does_not_import_torch(self):
        sys.modules.pop("torch", None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root, 45, "control", 100.0)
            self.make_run(root, 45, "cache", 125.0)

            summary.summarize_campaign(root, expected_seeds=[45])

        self.assertNotIn("torch", sys.modules)


class CompareCheckpointsTest(unittest.TestCase):
    def torch_or_skip(self):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest("torch not installed locally") from exc
        return torch

    def write_checkpoint(self, path: Path, policy: dict, optimizer: dict) -> None:
        torch = self.torch_or_skip()
        for member, payload in [("policy.pth", policy), ("policy.optimizer.pth", optimizer)]:
            temp = path.parent / member
            torch.save(payload, temp)
            with zipfile.ZipFile(path, "a") as archive:
                archive.write(temp, member)
            temp.unlink()

    def test_checkpoint_compare_rejects_missing_keys_dtype_and_non_tensor_differences(self):
        torch = self.torch_or_skip()
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.zip"
            right = Path(tmp) / "right.zip"
            self.write_checkpoint(
                left,
                {"layer": torch.tensor([1.0], dtype=torch.float32), "only_left": torch.tensor([1.0])},
                {"state": {"lr": 0.001}, "param_groups": [{"eps": 1e-5}]},
            )
            self.write_checkpoint(
                right,
                {"layer": torch.tensor([1.0], dtype=torch.float64), "only_right": torch.tensor([1.0])},
                {"state": {"lr": 0.002}, "param_groups": [{"eps": 1e-5}]},
            )

            result = summary.compare_checkpoints(left, right)

        self.assertFalse(result["within_tolerance"])
        self.assertFalse(result["exact"])
        self.assertGreater(result["issue_count"], 0)
        members = {member["member"]: member for member in result["members"]}
        self.assertIn("only_left", members["policy.pth"]["missing_right"])
        self.assertIn("only_right", members["policy.pth"]["missing_left"])
        differences = members["policy.optimizer.pth"]["non_tensor_differences"]
        lr_difference = next(difference for difference in differences if difference["name"] == "state.lr")
        self.assertEqual(lr_difference["left"], "0.001")
        self.assertEqual(lr_difference["right"], "0.002")
        self.assertEqual(lr_difference["type_left"], "float")
        self.assertEqual(lr_difference["type_right"], "float")

    def test_checkpoint_compare_rejects_invalid_tolerance_and_nonfinite_diff(self):
        torch = self.torch_or_skip()
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.zip"
            right = Path(tmp) / "right.zip"
            self.write_checkpoint(left, {"layer": torch.tensor([float("nan")])}, {"state": {}})
            self.write_checkpoint(right, {"layer": torch.tensor([0.0])}, {"state": {}})

            with self.assertRaisesRegex(summary.SummaryError, "tolerance"):
                summary.compare_checkpoints(left, right, tolerance=-1.0)

            result = summary.compare_checkpoints(left, right, tolerance=0.0)

        self.assertFalse(result["within_tolerance"])
        self.assertFalse(result["exact"])
        self.assertFalse(result["finite"])


if __name__ == "__main__":
    unittest.main()
