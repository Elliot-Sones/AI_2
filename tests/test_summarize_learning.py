"""Tests for dependency-light E6 learning result summarization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tools.analysis.summarize_learning as sl


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def train_result(label: str, *, chunk: int, start: int, end: int, final_sha: str, input_path=None, input_sha=None):
    run_label = label.rsplit("_chunk", 1)[0]
    output = f"/remote/runs/{run_label}/chunk{chunk:02d}"
    return {
        "label": label,
        "kind": "train",
        "status": "completed",
        "started_unix": 100.0 + chunk,
        "ended_unix": 110.0 + chunk,
        "output": output,
        "evidence": {
            "starting_steps": start,
            "ending_steps": end,
            "actual_steps": end - start,
            "wall_seconds": 2.5,
            "parameter_count": 1053973,
            "input_checkpoint": input_path,
            "input_checkpoint_sha256": input_sha,
            "initial_checkpoint": f"{output}/initial.zip",
            "latest_checkpoint": f"{output}/latest.zip",
            "final_checkpoint": f"{output}/final.zip",
            "checkpoint_evidence": {
                "initial_checkpoint": {"path": f"{output}/initial.zip", "sha256": f"initial-{chunk}", "size": 10, "exists": True},
                "latest_checkpoint": {"path": f"{output}/latest.zip", "sha256": final_sha, "size": 20, "exists": True},
                "final_checkpoint": {"path": f"{output}/final.zip", "sha256": final_sha, "size": 20, "exists": True},
            },
        },
    }


def eval_result(label: str, *, episodes: int = 20, success: float = 0.9, timeout: float = 0.1):
    summaries = {}
    for phase in ("0a", "0b"):
        for mode in ("deterministic", "sampled"):
            summaries[f"{phase}-{mode}"] = {
                "episodes": episodes,
                "success_rate": success,
                "fall_rate": 0.05,
                "timeout_rate": timeout,
                "nav_falls": 1,
                "timeouts": int(round(timeout * episodes)),
                "avg_steps": 123.0,
            }
    return {
        "label": label,
        "kind": "eval",
        "status": "completed",
        "started_unix": 120.0,
        "ended_unix": 127.0,
        "evidence": {"summaries": summaries, "metadata": "/remote/eval/metadata.json"},
    }


def skipped_from_completed(result: dict, status: str):
    return {
        "label": result["label"],
        "kind": result["kind"],
        "status": status,
        "output": result.get("output"),
        "evidence": result.get("evidence"),
    }


class SummarizeLearningTests(unittest.TestCase):
    def test_summarizes_completed_smoke_like_results(self):
        first = train_result("arch-512-512-256_seed145_chunk01_train", chunk=1, start=0, end=32, final_sha="sha1")
        second = train_result(
            "arch-512-512-256_seed145_chunk02_train",
            chunk=2,
            start=32,
            end=64,
            final_sha="sha2",
            input_path=first["evidence"]["final_checkpoint"],
            input_sha="sha1",
        )
        ev = eval_result("arch-512-512-256_seed145_chunk02_eval_dev20", episodes=2, success=0.0, timeout=1.0)

        summary = sl.summarize_results([first, second, ev], allow_partial=False)

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["totals"]["actual_steps"], 64)
        self.assertEqual(summary["totals"]["train_process_wall_seconds"], 20.0)
        self.assertEqual(summary["totals"]["train_learning_wall_seconds"], 5.0)
        self.assertEqual(summary["totals"]["eval_process_wall_seconds"], 7.0)
        arm = summary["arms"]["arch-512-512-256_seed145"]
        self.assertEqual(arm["parameter_count"], 1053973)
        self.assertEqual(arm["latest_checkpoint"]["sha256"], "sha2")
        self.assertEqual(arm["lineage"][1]["input_checkpoint_sha256"], "sha1")
        self.assertEqual(arm["evals"]["chunk02_eval_dev20"]["0a-deterministic"]["episodes"], 2)
        self.assertEqual(arm["evals"]["chunk02_eval_dev20"]["0a-deterministic"]["timeout_rate"], 1.0)

    def test_rejects_bad_resume_lineage_sha(self):
        first = train_result("arch-512-512-256_seed145_chunk01_train", chunk=1, start=0, end=32, final_sha="sha1")
        second = train_result(
            "arch-512-512-256_seed145_chunk02_train",
            chunk=2,
            start=32,
            end=64,
            final_sha="sha2",
            input_path=first["evidence"]["final_checkpoint"],
            input_sha="wrong",
        )

        with self.assertRaisesRegex(sl.SummaryError, "lineage sha mismatch"):
            sl.summarize_results([first, second], allow_partial=False)

    def test_rejects_invalid_eval_metrics(self):
        first = train_result("arch-512-512-256_seed145_chunk01_train", chunk=1, start=0, end=32, final_sha="sha1")
        ev = eval_result("arch-512-512-256_seed145_chunk01_eval_dev20")
        ev["evidence"]["summaries"]["0a-sampled"]["success_rate"] = float("nan")

        with self.assertRaisesRegex(sl.SummaryError, "non-finite"):
            sl.summarize_results([first, ev], allow_partial=False)

    def test_requires_partial_for_pending_plan_commands(self):
        with tempfile.TemporaryDirectory(prefix="learning-summary-") as tmp:
            root = Path(tmp)
            results = [train_result("arch-512-512-256_seed145_chunk01_train", chunk=1, start=0, end=32, final_sha="sha1")]
            write_json(root / "smoke_results.json", results)
            write_json(
                root / "artifacts" / "campaign_learning_plan.json",
                {"commands": [{"label": results[0]["label"]}, {"label": "arch-512-512-256_seed145_chunk01_eval_dev20"}]},
            )

            with self.assertRaisesRegex(sl.SummaryError, "pending commands"):
                sl.summarize_campaign(root, allow_partial=False)

            partial = sl.summarize_campaign(root, allow_partial=True)

        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["pending"]["count"], 1)
        self.assertEqual(partial["pending"]["arms"], ["arch-512-512-256_seed145"])
        self.assertTrue(partial["source"]["plan"].endswith("artifacts/campaign_learning_plan.json"))

    def test_hydrates_skipped_completed_receipts_from_execution_history(self):
        first = train_result("arch-512-512-256_seed145_chunk01_train", chunk=1, start=0, end=32, final_sha="sha1")
        second = train_result(
            "arch-512-512-256_seed145_chunk02_train",
            chunk=2,
            start=32,
            end=64,
            final_sha="sha2",
            input_path=first["evidence"]["final_checkpoint"],
            input_sha="sha1",
        )
        ev = eval_result("arch-512-512-256_seed145_chunk02_eval_dev20", episodes=2, success=0.0, timeout=1.0)
        with tempfile.TemporaryDirectory(prefix="learning-summary-history-") as tmp:
            root = Path(tmp)
            current = [
                skipped_from_completed(first, "skipped_completed"),
                skipped_from_completed(second, "skipped_completed"),
                skipped_from_completed(ev, "skipped_existing"),
            ]
            write_json(root / "artifacts" / "campaign_learning_results.json", {"status": "completed", "results": current})
            write_json(root / "artifacts" / "execution_history" / "prior.json", {"status": "completed", "results": [first, second, ev]})

            summary = sl.summarize_campaign(root)

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["totals"]["actual_steps"], 64)
        self.assertEqual(summary["totals"]["train_process_wall_seconds"], 20.0)
        self.assertEqual(summary["totals"]["eval_process_wall_seconds"], 7.0)
        self.assertEqual(summary["history"]["hydrated_labels"], [first["label"], second["label"], ev["label"]])
        arm = summary["arms"]["arch-512-512-256_seed145"]
        self.assertEqual(arm["lineage"][1]["input_checkpoint_sha256"], "sha1")

    def test_rejects_skipped_completed_without_matching_history(self):
        first = train_result("arch-512-512-256_seed145_chunk01_train", chunk=1, start=0, end=32, final_sha="sha1")
        with tempfile.TemporaryDirectory(prefix="learning-summary-history-missing-") as tmp:
            root = Path(tmp)
            write_json(root / "artifacts" / "campaign_learning_results.json", {"status": "completed", "results": [skipped_from_completed(first, "skipped_completed")]})

            with self.assertRaisesRegex(sl.SummaryError, "missing execution history"):
                sl.summarize_campaign(root)

    def test_accepts_single_completed_receipt_history_file(self):
        first = train_result("arch-512-512-256_seed145_chunk01_train", chunk=1, start=0, end=32, final_sha="sha1")
        with tempfile.TemporaryDirectory(prefix="learning-summary-single-history-") as tmp:
            root = Path(tmp)
            write_json(root / "artifacts" / "campaign_learning_results.json", {"status": "completed", "results": [skipped_from_completed(first, "skipped_completed")]})
            write_json(root / "artifacts" / "execution_history" / "single.json", first)

            summary = sl.summarize_campaign(root)

        self.assertEqual(summary["totals"]["actual_steps"], 32)
        self.assertEqual(summary["history"]["hydrated_labels"], [first["label"]])

    def test_rejects_skipped_evidence_conflicting_with_history(self):
        first = train_result("arch-512-512-256_seed145_chunk01_train", chunk=1, start=0, end=32, final_sha="sha1")
        skipped = skipped_from_completed(first, "skipped_completed")
        skipped["evidence"] = json.loads(json.dumps(skipped["evidence"]))
        skipped["evidence"]["checkpoint_evidence"]["final_checkpoint"]["sha256"] = "different"
        with tempfile.TemporaryDirectory(prefix="learning-summary-history-conflict-") as tmp:
            root = Path(tmp)
            write_json(root / "artifacts" / "campaign_learning_results.json", {"status": "completed", "results": [skipped]})
            write_json(root / "artifacts" / "execution_history" / "prior.json", {"status": "completed", "results": [first]})

            with self.assertRaisesRegex(sl.SummaryError, "conflicts with execution history"):
                sl.summarize_campaign(root)

    def test_rejects_conflicting_duplicate_history_labels(self):
        first = train_result("arch-512-512-256_seed145_chunk01_train", chunk=1, start=0, end=32, final_sha="sha1")
        duplicate = train_result("arch-512-512-256_seed145_chunk01_train", chunk=1, start=0, end=32, final_sha="other")
        with tempfile.TemporaryDirectory(prefix="learning-summary-history-dupe-") as tmp:
            root = Path(tmp)
            write_json(root / "artifacts" / "campaign_learning_results.json", {"status": "completed", "results": [skipped_from_completed(first, "skipped_completed")]})
            write_json(root / "artifacts" / "execution_history" / "a.json", {"status": "completed", "results": [first]})
            write_json(root / "artifacts" / "execution_history" / "b.json", {"status": "completed", "results": [duplicate]})

            with self.assertRaisesRegex(sl.SummaryError, "conflicting execution history"):
                sl.summarize_campaign(root)


if __name__ == "__main__":
    unittest.main()
