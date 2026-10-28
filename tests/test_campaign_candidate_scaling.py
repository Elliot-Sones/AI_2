"""Tests for the opt-in candidate grouped scaling screen driver."""

from __future__ import annotations

from ai2.paths import source_path
import importlib

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

scaling = importlib.import_module('baselines.throughput-scaling-20260906.scripts.campaign_candidate_scaling')


def write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class FakeClock:
    def __init__(self, value: float = 10_000.0) -> None:
        self.value = float(value)

    def time(self) -> float:
        return self.value


class CandidateScalingTests(unittest.TestCase):
    def make_paths(self, root: Path) -> SimpleNamespace:
        source = root / "source"
        grouped = write(source_path(root, "native_shared_batched_vec_env.py"), "candidate\n")
        paths = SimpleNamespace(
            output=root / "out",
            harness=write(source_path(root, "benchmark_campaign.py"), "harness\n"),
            config=write(source / "control.yaml", "phase: 0b\n"),
            checkpoint=write(root / "selected_16001056.zip", "checkpoint\n"),
            source=source,
            train_job=write(root / "train_job.py", "train job\n"),
            grouped_env_path=grouped,
            frozen_environment=write(source / "UTMIST-AI2-main/environment/environment.py", "env\n"),
        )
        write(source_path(source, "train.py"), "train\n")
        return paths

    def write_receipt(self, paths: SimpleNamespace, *, overrides: dict | None = None) -> Path:
        payload = {
            "status": "passed",
            "exit_code": 0,
            "skipped": 0,
            "failures": 0,
            "errors": 0,
            "testsRun": 6,
            "steps_per_case": 1000,
            "group_sizes": [2, 4, 8],
            "runner_sha256": scaling.sha256_file(paths.grouped_env_path),
            "candidate_grouped_path": str(paths.grouped_env_path.resolve()),
            "frozen_environment_sha256": scaling.sha256_file(paths.frozen_environment),
            "sourcehashes_before": {
                str(path): scaling.sha256_file(path)
                for path in scaling.required_parity_sources(paths.source, paths.train_job, paths.grouped_env_path)
            },
            "sourcehashes_after": {
                str(path): scaling.sha256_file(path)
                for path in scaling.required_parity_sources(paths.source, paths.train_job, paths.grouped_env_path)
            },
            "minimum_set": ["fixed_action", "reset", "timeouts"],
        }
        if overrides:
            payload.update(overrides)
        receipt = write(paths.output / "candidate_game_parity_passed.json", json.dumps(payload))
        return receipt

    def make_args(self, paths: SimpleNamespace, **overrides) -> argparse.Namespace:
        receipt = overrides.pop("parity_receipt", paths.output / "candidate_game_parity_passed.json")
        kwargs = dict(
            output=paths.output,
            deadline_unix=20_000.0,
            run=False,
            per_job_timeout=300.0,
            device="cuda",
            phase="0b",
            config=paths.config,
            checkpoint=paths.checkpoint,
            source=paths.source,
            train_job=paths.train_job,
            harness=paths.harness,
            grouped_env_path=paths.grouped_env_path,
            parity_receipt=receipt,
            group_sizes="4,8",
        )
        kwargs.update(overrides)
        return scaling.validate_args(argparse.Namespace(**kwargs))

    def test_default_dry_run_writes_fixed_256x128_group4_and_group8_plan_without_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            self.write_receipt(paths)
            args = self.make_args(paths)
            calls = []

            code = scaling.run_suite(args, runner=lambda *_args: calls.append(_args) or 0, clock=FakeClock())

            self.assertEqual(code, 0)
            self.assertEqual(calls, [])
            report = json.loads((paths.output / "suite_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "dryrun")
            self.assertEqual(report["planned_jobs"], 2)
            planned = report["jobs"]
            self.assertEqual([job["group_size"] for job in planned], [4, 8])
            self.assertTrue(all(job["n_envs"] == 256 and job["n_steps"] == 128 for job in planned))
            self.assertTrue(all(job["warmup"] == 2 and job["cycles"] == 30 for job in planned))
            self.assertEqual(report["short_screen_note"], "screening run only; not an acceptance or speedup claim")

    def test_run_executes_after_parity_receipt_and_source_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            receipt = self.write_receipt(paths)
            args = self.make_args(paths, run=True, parity_receipt=receipt)
            seen = []

            def fake_runner(job, command, env, timeout):
                seen.append((job, command, env, timeout))
                write(
                    job.output / "campaign_summary.json",
                    json.dumps(
                        {
                            "status": "completed",
                            "completed_measured_cycles": job.cycles,
                            "requested_measured_cycles": job.cycles,
                            "warmup_cycles": job.warmup,
                            "plan": {"transitions_per_cycle": job.n_envs * job.n_steps},
                        }
                    ),
                )
                return 0

            code = scaling.run_suite(args, runner=fake_runner, clock=FakeClock())

            self.assertEqual(code, 0)
            self.assertEqual(len(seen), 2)
            first_job, first_command, first_env, first_timeout = seen[0]
            self.assertEqual(first_job.name, "candidate-scaling-env256-step128-group4")
            self.assertEqual(first_timeout, 300.0)
            self.assertIn("--n-envs", first_command)
            self.assertIn("256", first_command)
            self.assertIn("--n-steps", first_command)
            self.assertIn("128", first_command)
            self.assertIn("--group-size", first_command)
            self.assertIn("4", first_command)
            self.assertIn("--maxseconds", first_command)
            self.assertIn("300.0", first_command)
            self.assertEqual(first_env["WANDB_MODE"], "offline")
            manifest = json.loads((first_job.output / "job_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["files"]["parity_receipt"]["sha256"], scaling.sha256_file(receipt))
            self.assertEqual(manifest["files"]["grouped_env"]["sha256"], scaling.sha256_file(paths.grouped_env_path))
            self.assertIn("parity_source::isolated_rng_vec_env.py", manifest["files"])
            snapshot = first_job.output / "source_snapshot"
            self.assertTrue((source_path(snapshot, "campaign_candidate_scaling.py")).is_file())
            self.assertTrue((source_path(snapshot, "benchmark_campaign.py")).is_file())
            self.assertTrue((snapshot / "candidate_grouped_env.py").is_file())
            self.assertTrue((snapshot / "train_job.py").is_file())
            self.assertTrue((snapshot / "control.yaml").is_file())
            self.assertTrue((snapshot / "selected_16001056.zip").is_file())
            self.assertTrue((source_path(snapshot, "train.py")).is_file())
            self.assertTrue((snapshot / "environment.py").is_file())
            self.assertTrue((snapshot / "candidate_game_parity_passed.json").is_file())
            self.assertTrue((snapshot / "parity_source__isolated_rng_vec_env.py").is_file())
            self.assertTrue((paths.output / "source_manifest.json").is_file())

    def test_parity_receipt_must_match_real_validator_exact_six_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            receipt = self.write_receipt(paths, overrides={"testsRun": 7})
            args = self.make_args(paths, parity_receipt=receipt)

            with self.assertRaisesRegex(RuntimeError, "testsRun must be 6"):
                scaling.validate_parity_receipt(args)

    def test_real_validator_receipt_without_literal_minimum_set_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            receipt = self.write_receipt(paths)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload.pop("minimum_set")
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            args = self.make_args(paths, parity_receipt=receipt)

            self.assertEqual(scaling.validate_parity_receipt(args)["testsRun"], 6)

    def test_parity_receipt_requires_complete_current_required_source_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            receipt = self.write_receipt(paths)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["sourcehashes_before"].pop(next(path for path in payload["sourcehashes_before"] if path.endswith("isolated_rng_vec_env.py")))
            payload["sourcehashes_after"] = dict(payload["sourcehashes_before"])
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            args = self.make_args(paths, parity_receipt=receipt)

            with self.assertRaisesRegex(RuntimeError, "missing required source hash"):
                scaling.validate_parity_receipt(args)

    def test_rejects_group16_until_separately_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            self.write_receipt(paths)
            with self.assertRaisesRegex(ValueError, "supported group sizes"):
                self.make_args(paths, group_sizes="4,16")

    def test_run_rejects_bad_or_incomplete_parity_receipt_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            bad = self.write_receipt(paths, overrides={"candidate_sha256": "bad", "minimum_set": []})
            args = self.make_args(paths, run=True, parity_receipt=bad)
            calls = []

            code = scaling.run_suite(args, runner=lambda *_args: calls.append(_args) or 0, clock=FakeClock())

            self.assertEqual(code, 2)
            self.assertEqual(calls, [])
            report = json.loads((paths.output / "suite_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "validation_failed")
            self.assertIn("candidate sha", report["error"])

    def test_deadline_skips_before_launch_when_timeout_plus_reserve_does_not_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            self.write_receipt(paths)
            args = self.make_args(paths, run=True, deadline_unix=10_500.0)
            calls = []

            code = scaling.run_suite(args, runner=lambda *_args: calls.append(_args) or 0, clock=FakeClock())

            self.assertEqual(code, 3)
            self.assertEqual(calls, [])
            report = json.loads((paths.output / "suite_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "deadline_insufficient")
            self.assertEqual(report["jobs"][0]["status"], "skipped_deadline")
            self.assertEqual(report["jobs"][0]["required_seconds"], 600.0)


if __name__ == "__main__":
    unittest.main()
