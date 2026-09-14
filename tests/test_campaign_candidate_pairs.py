"""Tests for the bounded native-candidate matched-pairs campaign runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))


def write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def marker_path_strings(args: argparse.Namespace) -> list[str]:
    marker = json.loads(args.parity_summary.read_text(encoding="utf-8"))
    return list((marker.get("sourcehashes_after") or {}).keys())


class CampaignCandidatePairsTests(unittest.TestCase):
    def make_args(self, tmp: Path, *, output: Path | None = None, parity: Path | None = None):
        import campaign_candidate_pairs as pairs

        source = tmp / "source"
        frozen = write(source / "UTMIST-AI2-main/environment/environment.py", "frozen\n")
        # The validator hashes the whole frozen environment package, not just environment.py.
        write(source / "UTMIST-AI2-main/environment/__init__.py", "")
        write(source / "UTMIST-AI2-main/environment/constants.py", "constants\n")
        write(source / "UTMIST-AI2-main/environment/agent.py", "agents\n")
        config = write(source / "control.yaml", "config\n")
        source_train = write(source / "train.py", "source train\n")
        checkpoint = write(tmp / "checkpoint.zip", "checkpoint\n")
        train = write(tmp / "train_job.py", "train\n")
        harness = write(tmp / "benchmark_campaign.py", "harness\n")
        baseline_grouped = write(tmp / "grouped_vec_env.py", "baseline grouped\n")
        candidate_grouped = write(tmp / "native_grouped_vec_env.py", "candidate grouped\n")
        native_helper = write(tmp / "native_game_patch.py", "native patch\n")
        native_c = write(tmp / "native/live_input.c", "native c\n")
        parity_path = parity or tmp / "parity.json"
        write(
            parity_path,
            json.dumps(
                {
                    "exit_code": 0,
                    "testsRun": 6,
                    "failures": 0,
                    "errors": 0,
                    "skipped": 0,
                    "steps_per_case": 1000,
                    "group_sizes": [2, 4, 8],
                    "candidate_grouped_path": str(candidate_grouped.resolve()),
                    "runner_sha256": pairs.sha256_file(candidate_grouped),
                    "frozen_environment_sha256": pairs.sha256_file(frozen),
                    "runtime_env": {
                        "AI2_NATIVE_INPUT": "1",
                        "AI2_NATIVE_OBSERVATION": "1",
                    },
                    "sourcehashes_before": {},
                    "sourcehashes_after": {},
                }
            )
            + "\n",
        )
        args = argparse.Namespace(
            output=(output or tmp / "out").resolve(),
            deadline_unix=10_000.0,
            per_job_timeout=600.0,
            candidate_grouped_path=candidate_grouped.resolve(),
            parity_summary=parity_path.resolve(),
            source=source.resolve(),
            train_job=train.resolve(),
            config=config.resolve(),
            checkpoint=checkpoint.resolve(),
            harness=harness.resolve(),
            baseline_grouped_path=baseline_grouped.resolve(),
            candidate_extra_arg=[],
            candidate_env=[],
            native_input=True,
            native_observation=True,
            run=False,
            phase="0b",
            maxseconds=600.0,
        )
        marker = json.loads(parity_path.read_text(encoding="utf-8"))
        required = pairs.minimum_parity_source_hashes(args)
        marker["sourcehashes_before"] = dict(required)
        marker["sourcehashes_after"] = dict(required)
        parity_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
        return args

    def test_plan_uses_three_alternating_matched_pairs_with_fixed_campaign_shape(self):
        import campaign_candidate_pairs as pairs

        with tempfile.TemporaryDirectory() as raw_tmp:
            args = self.make_args(Path(raw_tmp))
            jobs = pairs.plan_jobs(args)

        self.assertEqual(
            [(job.seed, job.arm) for job in jobs],
            [
                (45, "baseline"),
                (45, "candidate"),
                (46, "candidate"),
                (46, "baseline"),
                (47, "baseline"),
                (47, "candidate"),
            ],
        )
        self.assertTrue(all(job.n_envs == 256 for job in jobs))
        self.assertTrue(all(job.n_steps == 128 for job in jobs))
        self.assertTrue(all(job.group_size == 4 for job in jobs))
        self.assertTrue(all(job.batch_size == 8192 for job in jobs))
        self.assertTrue(all(job.epochs == 8 for job in jobs))
        self.assertTrue(all(job.warmup == 2 for job in jobs))
        self.assertTrue(all(job.cycles == 100 for job in jobs))

    def test_commands_keep_candidate_path_and_flags_off_baseline(self):
        import campaign_candidate_pairs as pairs

        with tempfile.TemporaryDirectory() as raw_tmp:
            args = self.make_args(Path(raw_tmp))
            baseline, candidate = pairs.plan_jobs(args)[:2]
            baseline_command = pairs.command_for_job(args, baseline)
            candidate_command = pairs.command_for_job(args, candidate)
            baseline_env = pairs.env_for_job(args, baseline)
            candidate_env = pairs.env_for_job(args, candidate)

        self.assertIn(str(args.baseline_grouped_path), baseline_command)
        self.assertNotIn(str(args.candidate_grouped_path), baseline_command)
        self.assertNotIn("--candidate-only", baseline_command)
        self.assertIn("--phase", baseline_command)
        self.assertEqual(baseline_command[baseline_command.index("--phase") + 1], "0b")
        self.assertIn("--maxseconds", baseline_command)
        self.assertEqual(baseline_command[baseline_command.index("--maxseconds") + 1], "600.0")
        self.assertEqual(baseline_env["AI2_NATIVE_INPUT"], "0")
        self.assertEqual(baseline_env["AI2_NATIVE_OBSERVATION"], "0")
        self.assertIn(str(args.candidate_grouped_path), candidate_command)
        self.assertNotIn("--candidate-only", candidate_command)
        self.assertNotIn("AI2_CANDIDATE_MODE", candidate_env)
        self.assertEqual(candidate_env["AI2_NATIVE_INPUT"], "1")
        self.assertEqual(candidate_env["AI2_NATIVE_OBSERVATION"], "1")

    def test_parity_summary_must_bind_current_candidate_and_frozen_source_hashes(self):
        import campaign_candidate_pairs as pairs

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            args = self.make_args(tmp)
            pairs.validate_parity_summary(args)
            args.candidate_grouped_path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "runner_sha256"):
                pairs.validate_parity_summary(args)

    def test_parity_summary_rejects_stale_helper_hashes_and_weak_runtime_evidence(self):
        import campaign_candidate_pairs as pairs

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            args = self.make_args(tmp)
            marker = json.loads(args.parity_summary.read_text())
            marker["testsRun"] = 5
            args.parity_summary.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "testsRun"):
                pairs.validate_parity_summary(args)

            args = self.make_args(tmp / "again")
            marker = json.loads(args.parity_summary.read_text())
            marker["steps_per_case"] = 999
            args.parity_summary.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "steps_per_case"):
                pairs.validate_parity_summary(args)

            args = self.make_args(tmp / "stale")
            stale_path = next(path for path in marker_path_strings(args) if path.endswith("native_game_patch.py"))
            marker = json.loads(args.parity_summary.read_text())
            marker["sourcehashes_before"][stale_path] = "0" * 64
            marker["sourcehashes_after"][stale_path] = "0" * 64
            args.parity_summary.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "sourcehashes_before|sourcehashes_after"):
                pairs.validate_parity_summary(args)

    def test_parity_summary_requires_complete_validator_dependency_set(self):
        import campaign_candidate_pairs as pairs
        import validate_grouped_candidate

        with tempfile.TemporaryDirectory() as raw_tmp:
            args = self.make_args(Path(raw_tmp))
            expected = {str(path) for path in validate_grouped_candidate.required_sources(
                args.source,
                args.train_job,
                args.candidate_grouped_path,
            )}
            marker = json.loads(args.parity_summary.read_text())
            self.assertTrue(expected.issubset(set(marker["sourcehashes_after"])))
            removed = sorted(expected)[0]
            marker["sourcehashes_before"].pop(removed)
            marker["sourcehashes_after"].pop(removed)
            args.parity_summary.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing required parity source"):
                pairs.validate_parity_summary(args)

    def test_candidate_extra_args_and_candidate_env_are_rejected_for_this_scope(self):
        import campaign_candidate_pairs as pairs

        with tempfile.TemporaryDirectory() as raw_tmp:
            args = self.make_args(Path(raw_tmp))
            args.candidate_extra_arg = ["--history"]
            with self.assertRaisesRegex(ValueError, "candidate-extra-arg"):
                pairs.validate_args(args)
            args = self.make_args(Path(raw_tmp) / "env")
            args.candidate_env = ["AI2_CANDIDATE_MODE=native"]
            with self.assertRaisesRegex(ValueError, "candidate-env"):
                pairs.validate_args(args)

    def test_required_source_files_must_exist_for_archive(self):
        import campaign_candidate_pairs as pairs

        with tempfile.TemporaryDirectory() as raw_tmp:
            args = self.make_args(Path(raw_tmp))
            args.train_job.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "train_job"):
                pairs.run_suite(args, runner=lambda *items: 0, clock=types.SimpleNamespace(time=lambda: 100.0))

    def test_dry_run_refuses_existing_output_and_writes_source_archive_without_running_jobs(self):
        import campaign_candidate_pairs as pairs

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            args = self.make_args(tmp)
            calls = []
            code = pairs.run_suite(
                args,
                runner=lambda *items: calls.append(items) or 0,
                clock=types.SimpleNamespace(time=lambda: 100.0),
            )
            self.assertEqual(code, 0)
            self.assertEqual(calls, [])
            self.assertTrue((args.output / "dryrun_plan.json").is_file())
            archive_manifest = json.loads((args.output / "source_archive/source_manifest.json").read_text())
            paths = {record["label"] for record in archive_manifest["files"]}
            self.assertIn("candidate_grouped_path", paths)
            self.assertIn("frozen_environment", paths)
            self.assertIn("campaign_suite.py", paths)
            self.assertIn("validate_grouped_candidate.py", paths)
            self.assertIn("batched_rng_vec_env.py", paths)
            self.assertIn("source/train.py", paths)
            self.assertIn("train_job", paths)
            self.assertIn("config", paths)
            self.assertIn("checkpoint", paths)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                pairs.run_suite(args, runner=lambda *items: 0, clock=types.SimpleNamespace(time=lambda: 100.0))

    def test_main_dry_run_cli_executes_without_mocked_runner(self):
        import campaign_candidate_pairs as pairs

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            args = self.make_args(tmp)
            code = pairs.main(
                [
                    "--output",
                    str(args.output),
                    "--deadline-unix",
                    str(args.deadline_unix),
                    "--candidate-grouped-path",
                    str(args.candidate_grouped_path),
                    "--parity-summary",
                    str(args.parity_summary),
                    "--source",
                    str(args.source),
                    "--train-job",
                    str(args.train_job),
                    "--config",
                    str(args.config),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--harness",
                    str(args.harness),
                    "--baseline-grouped-path",
                    str(args.baseline_grouped_path),
                    "--native-input",
                    "--native-observation",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((args.output / "suite_report.json").is_file())

    def test_runner_refuses_if_candidate_source_hash_changes_before_job(self):
        import campaign_candidate_pairs as pairs

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            args = self.make_args(tmp)
            args.run = True
            calls = []

            def runner(job, command, env, timeout):
                calls.append(job.name)
                write(job.output / "campaign_summary.json", json.dumps({
                    "status": "completed",
                    "completed_measured_cycles": job.cycles,
                    "requested_measured_cycles": job.cycles,
                    "warmup_cycles": job.warmup,
                    "plan": {"transitions_per_cycle": job.n_envs * job.n_steps},
                }) + "\n")
                if len(calls) == 1:
                    args.candidate_grouped_path.write_text("changed\n", encoding="utf-8")
                return 0

            code = pairs.run_suite(args, runner=runner, clock=types.SimpleNamespace(time=lambda: 100.0))
            report = json.loads((args.output / "suite_report.json").read_text())
            self.assertEqual(code, 4)
            self.assertEqual(len(calls), 1)
            self.assertEqual(report["status"], "source_changed")


if __name__ == "__main__":
    unittest.main()
