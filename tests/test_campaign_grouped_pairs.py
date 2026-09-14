import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "campaign_grouped_pairs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("campaign_grouped_pairs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["campaign_grouped_pairs"] = module
    spec.loader.exec_module(module)
    return module


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_parity_marker(output, grouped_env_path, frozen_environment_path, **overrides):
    marker = {
        "status": "passed",
        "exit_code": 0,
        "skipped": 0,
        "steps_per_case": 1000,
        "group_sizes": [4],
        "runner_sha256": sha256_file(grouped_env_path),
        "frozen_environment_sha256": sha256_file(frozen_environment_path),
    }
    marker.update(overrides)
    write_text(Path(output) / "grouped_game_parity_passed.json", json.dumps(marker))
    return marker


class FakeClock:
    def __init__(self, now=1000.0):
        self.value = float(now)

    def time(self):
        return self.value


class FakeRunner:
    def __init__(self):
        self.commands = []

    def __call__(self, job, command, env, timeout):
        self.commands.append((job, command, env, timeout))
        summary = {
            "status": "completed",
            "completed_measured_cycles": job.cycles,
            "requested_measured_cycles": job.cycles,
            "warmup_cycles": job.warmup,
            "plan": {
                "transitions_per_cycle": job.n_envs * job.n_steps,
                "steps": (job.cycles + job.warmup) * job.n_envs * job.n_steps,
            },
        }
        job.output.mkdir(parents=True, exist_ok=True)
        (job.output / "campaign_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (job.output / "job.log").write_text("ok\n", encoding="utf-8")
        return 0


class CampaignGroupedPairsTests(unittest.TestCase):
    def test_plan_has_matched_seed_pairs_and_alternates_starting_grouped(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            args = module.build_parser().parse_args(["--output", str(output), "--deadline-unix", "9999"])
            parsed = module.validate_args(args)

            jobs = module.plan_jobs(parsed)

        self.assertEqual([job.name for job in jobs], [
            "grouped-pairs-seed45-group4",
            "grouped-pairs-seed45-group1",
            "grouped-pairs-seed46-group1",
            "grouped-pairs-seed46-group4",
            "grouped-pairs-seed47-group4",
            "grouped-pairs-seed47-group1",
        ])
        self.assertEqual([(job.seed, job.group_size) for job in jobs], [
            (45, 4),
            (45, 1),
            (46, 1),
            (46, 4),
            (47, 4),
            (47, 1),
        ])
        self.assertTrue(all(job.n_envs == 64 and job.n_steps == 512 for job in jobs))
        self.assertTrue(all(job.warmup == 2 and job.cycles == 100 for job in jobs))
        self.assertTrue(all(job.batch_size == 8192 and job.epochs == 8 for job in jobs))
        self.assertTrue(all(job.resident_buffer is False for job in jobs))

    def test_pair_offset_and_limit_select_whole_pairs_without_resuming_or_skipping(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            args = module.build_parser().parse_args(
                ["--output", str(output), "--deadline-unix", "9999", "--pair-offset", "1", "--pair-limit", "1"]
            )

            jobs = module.plan_jobs(module.validate_args(args))

        self.assertEqual([(job.seed, job.group_size) for job in jobs], [(46, 1), (46, 4)])

    def test_pair_selection_must_not_silently_select_zero_jobs(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            args = module.build_parser().parse_args(
                ["--output", str(output), "--deadline-unix", "9999", "--pair-offset", "3"]
            )

            with self.assertRaisesRegex(ValueError, "select at least one"):
                module.validate_args(args)

    def test_parity_marker_hash_gate_is_required_before_any_arm_runs(self):
        module = load_module()
        runner = FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in ("benchmark_campaign.py", "rollout_cache.py", "grouped_vec_env.py", "train_job.py"):
                write_text(tmp_path / name, name)
            config = write_text(tmp_path / "control.yaml", "config")
            checkpoint = write_text(tmp_path / "selected.zip", "checkpoint")
            frozen_env = write_text(
                tmp_path / "source" / "UTMIST-AI2-main" / "environment" / "environment.py",
                "environment",
            )
            output = tmp_path / "suite"
            args = module.build_parser().parse_args(
                [
                    "--output",
                    str(output),
                    "--deadline-unix",
                    "9999",
                    "--source",
                    str(tmp_path / "source"),
                    "--train-job",
                    str(tmp_path / "train_job.py"),
                    "--config",
                    str(config),
                    "--checkpoint",
                    str(checkpoint),
                    "--harness",
                    str(tmp_path / "benchmark_campaign.py"),
                    "--rollout-cache-path",
                    str(tmp_path / "rollout_cache.py"),
                    "--grouped-env-path",
                    str(tmp_path / "grouped_vec_env.py"),
                    "--pair-limit",
                    "1",
                ]
            )
            parsed = module.validate_args(args)

            with self.assertRaisesRegex(RuntimeError, "parity marker"):
                module.run_suite(parsed, runner=runner, clock=FakeClock())
            self.assertEqual(runner.commands, [])

            write_parity_marker(output, tmp_path / "grouped_vec_env.py", frozen_env)
            code = module.run_suite(parsed, runner=runner, clock=FakeClock())
            manifest_grouped = json.loads((output / "grouped-pairs-seed45-group4" / "job_manifest.json").read_text())
            manifest_control = json.loads((output / "grouped-pairs-seed45-group1" / "job_manifest.json").read_text())

        self.assertEqual(code, 0)
        self.assertEqual(len(runner.commands), 2)
        self.assertTrue(manifest_grouped["files"]["grouped_parity_marker"]["exists"])
        self.assertTrue(manifest_control["files"]["grouped_parity_marker"]["exists"])
        self.assertIn("--group-size", runner.commands[0][1])
        self.assertIn("4", runner.commands[0][1])
        self.assertIn("--group-size", runner.commands[1][1])
        self.assertIn("1", runner.commands[1][1])

    def test_parity_marker_rejects_stale_or_insufficient_group4_gate(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "suite"
            grouped_env = write_text(tmp_path / "grouped_vec_env.py", "runner-v1")
            frozen_env = write_text(
                tmp_path / "source" / "UTMIST-AI2-main" / "environment" / "environment.py",
                "environment-v1",
            )
            args = module.validate_args(
                module.build_parser().parse_args(
                    [
                        "--output",
                        str(output),
                        "--deadline-unix",
                        "9999",
                        "--source",
                        str(tmp_path / "source"),
                        "--grouped-env-path",
                        str(grouped_env),
                    ]
                )
            )

            write_parity_marker(output, grouped_env, frozen_env, group_sizes=[2, 8])
            with self.assertRaisesRegex(RuntimeError, "group_sizes"):
                module.validate_grouped_pair_parity_marker(args)

            write_parity_marker(output, grouped_env, frozen_env)
            grouped_env.write_text("runner-v2", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "runner_sha256"):
                module.validate_grouped_pair_parity_marker(args)

            grouped_env.write_text("runner-v1", encoding="utf-8")
            write_parity_marker(output, grouped_env, frozen_env)
            frozen_env.write_text("environment-v2", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "frozen_environment_sha256"):
                module.validate_grouped_pair_parity_marker(args)

    def test_deadline_and_completed_output_guards_reuse_campaign_suite_behavior(self):
        module = load_module()
        runner = FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in ("benchmark_campaign.py", "rollout_cache.py", "grouped_vec_env.py", "train_job.py"):
                write_text(tmp_path / name, name)
            config = write_text(tmp_path / "control.yaml", "config")
            checkpoint = write_text(tmp_path / "selected.zip", "checkpoint")
            frozen_env = write_text(
                tmp_path / "source" / "UTMIST-AI2-main" / "environment" / "environment.py",
                "environment",
            )
            output = tmp_path / "suite"
            write_parity_marker(output, tmp_path / "grouped_vec_env.py", frozen_env)
            completed = output / "grouped-pairs-seed45-group4" / "campaign_summary.json"
            completed.parent.mkdir(parents=True)
            completed.write_text('{"status":"completed"}', encoding="utf-8")
            base_args = [
                "--output",
                str(output),
                "--deadline-unix",
                "9999",
                "--source",
                str(tmp_path / "source"),
                "--train-job",
                str(tmp_path / "train_job.py"),
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--harness",
                str(tmp_path / "benchmark_campaign.py"),
                "--rollout-cache-path",
                str(tmp_path / "rollout_cache.py"),
                "--grouped-env-path",
                str(tmp_path / "grouped_vec_env.py"),
                "--pair-limit",
                "1",
            ]

            parsed = module.validate_args(module.build_parser().parse_args(base_args))
            with self.assertRaisesRegex(FileExistsError, "completed output"):
                module.run_suite(parsed, runner=runner, clock=FakeClock())

            output2 = tmp_path / "suite2"
            write_parity_marker(output2, tmp_path / "grouped_vec_env.py", frozen_env)
            args2 = module.validate_args(
                module.build_parser().parse_args(
                    [
                        "--output",
                        str(output2),
                        "--deadline-unix",
                        str(FakeClock().time() + 1000),
                        "--per-job-timeout",
                        "900",
                        "--source",
                        str(tmp_path / "source"),
                        "--train-job",
                        str(tmp_path / "train_job.py"),
                        "--config",
                        str(config),
                        "--checkpoint",
                        str(checkpoint),
                        "--harness",
                        str(tmp_path / "benchmark_campaign.py"),
                        "--rollout-cache-path",
                        str(tmp_path / "rollout_cache.py"),
                        "--grouped-env-path",
                        str(tmp_path / "grouped_vec_env.py"),
                        "--pair-limit",
                        "1",
                    ]
                )
            )
            clock = FakeClock()

            def slow_runner(job, command, env, timeout):
                result = runner(job, command, env, timeout)
                clock.value += timeout
                return result

            code = module.run_suite(args2, runner=slow_runner, clock=clock)
            report = json.loads((output2 / "suite_report.json").read_text())

        self.assertEqual(code, 3)
        self.assertEqual(len(runner.commands), 1)
        self.assertEqual(report["status"], "deadline_insufficient")
        self.assertEqual(report["completed_jobs"], 1)


if __name__ == "__main__":
    unittest.main()
