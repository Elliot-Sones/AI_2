import importlib.util
import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "campaign_suite.py"


def load_suite():
    spec = importlib.util.spec_from_file_location("campaign_suite", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["campaign_suite"] = module
    spec.loader.exec_module(module)
    return module


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_grouped_parity_marker(output, grouped_env_path, frozen_environment_path, **overrides):
    marker = {
        "status": "passed",
        "exit_code": 0,
        "skipped": 0,
        "steps_per_case": 1000,
        "group_sizes": [2, 4, 8],
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


class CampaignSuiteTests(unittest.TestCase):
    def test_paired_cache_plan_has_seed_pairs_default_shape_and_deterministic_order(self):
        suite = load_suite()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            args = suite.build_parser().parse_args(
                ["--mode", "paired-cache", "--output", str(output), "--deadline-unix", "9999"]
            )

            jobs = suite.plan_jobs(suite.validate_args(args))

        self.assertEqual(len(jobs), 6)
        self.assertEqual(sorted((job.seed, job.arm) for job in jobs), [
            (45, "cache"),
            (45, "control"),
            (46, "cache"),
            (46, "control"),
            (47, "cache"),
            (47, "control"),
        ])
        self.assertEqual([job.seed for job in jobs], [45, 45, 46, 46, 47, 47])
        self.assertNotEqual([job.arm for job in jobs[:2]], [job.arm for job in jobs[2:4]])
        self.assertTrue(all(job.n_envs == 64 and job.n_steps == 512 for job in jobs))
        self.assertTrue(all(job.cycles == 100 and job.warmup == 2 for job in jobs))
        self.assertTrue(any("--resident-buffer" in job.command_args for job in jobs if job.arm == "cache"))
        self.assertTrue(all("--resident-buffer" not in job.command_args for job in jobs if job.arm == "control"))

    def test_smoke_cache_plan_runs_one_control_cache_pair_with_short_cycle_count(self):
        suite = load_suite()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            args = suite.build_parser().parse_args(
                ["--mode", "smoke-cache", "--output", str(output), "--deadline-unix", "9999"]
            )

            jobs = suite.plan_jobs(suite.validate_args(args))

        self.assertEqual([(job.seed, job.arm, job.warmup, job.cycles) for job in jobs], [
            (45, "control", 1, 2),
            (45, "cache", 1, 2),
        ])

    def test_grouped_screen_plan_requires_parity_and_keeps_rollout_size_constant(self):
        suite = load_suite()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "suite"
            grouped_env = write_text(tmp_path / "grouped_vec_env.py", "runner")
            frozen_env = write_text(tmp_path / "source" / "UTMIST-AI2-main" / "environment" / "environment.py", "environment")
            args = suite.build_parser().parse_args(
                [
                    "--mode",
                    "grouped-screen",
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
            parsed = suite.validate_args(args)
            with self.assertRaisesRegex(RuntimeError, "parity"):
                suite.plan_jobs(parsed)

            write_grouped_parity_marker(output, grouped_env, frozen_env)
            jobs = suite.plan_jobs(parsed)

        self.assertEqual([(job.n_envs, job.n_steps, job.group_size) for job in jobs], [
            (64, 512, 2),
            (64, 512, 4),
            (64, 512, 8),
            (128, 256, 8),
            (256, 128, 8),
        ])
        self.assertTrue(all(job.n_envs * job.n_steps == 32768 for job in jobs))
        self.assertTrue(all(job.cycles == 20 and job.warmup == 2 for job in jobs))

    def test_grouped_screen_rejects_empty_malformed_failed_incomplete_or_stale_parity_marker(self):
        suite = load_suite()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "suite"
            grouped_env = write_text(tmp_path / "grouped_vec_env.py", "runner-v1")
            frozen_env = write_text(tmp_path / "source" / "UTMIST-AI2-main" / "environment" / "environment.py", "environment-v1")
            args = suite.validate_args(
                suite.build_parser().parse_args(
                    [
                        "--mode",
                        "grouped-screen",
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
            marker_path = output / "grouped_game_parity_passed.json"

            for content, message in [
                ("", "malformed"),
                ("[]", "object"),
                (json.dumps({"status": "failed"}), "status"),
            ]:
                write_text(marker_path, content)
                with self.assertRaisesRegex(RuntimeError, message):
                    suite.plan_jobs(args)

            write_grouped_parity_marker(output, grouped_env, frozen_env, skipped=1)
            with self.assertRaisesRegex(RuntimeError, "skipped"):
                suite.plan_jobs(args)

            write_grouped_parity_marker(output, grouped_env, frozen_env, steps_per_case=999)
            with self.assertRaisesRegex(RuntimeError, "steps_per_case"):
                suite.plan_jobs(args)

            write_grouped_parity_marker(output, grouped_env, frozen_env, steps_per_case="many")
            with self.assertRaisesRegex(RuntimeError, "steps_per_case"):
                suite.plan_jobs(args)

            write_grouped_parity_marker(output, grouped_env, frozen_env, group_sizes=[2, 8])
            with self.assertRaisesRegex(RuntimeError, "group_sizes"):
                suite.plan_jobs(args)

            write_grouped_parity_marker(output, grouped_env, frozen_env, group_sizes=["x"])
            with self.assertRaisesRegex(RuntimeError, "group_sizes"):
                suite.plan_jobs(args)

            write_grouped_parity_marker(output, grouped_env, frozen_env)
            grouped_env.write_text("runner-v2", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "runner_sha256"):
                suite.plan_jobs(args)

            grouped_env.write_text("runner-v1", encoding="utf-8")
            write_grouped_parity_marker(output, grouped_env, frozen_env)
            frozen_env.write_text("environment-v2", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "frozen_environment_sha256"):
                suite.plan_jobs(args)

    def test_grouped_screen_records_parity_marker_provenance_in_job_manifest(self):
        suite = load_suite()
        runner = FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in ("benchmark_campaign.py", "rollout_cache.py", "grouped_vec_env.py", "train_job.py"):
                write_text(tmp_path / name, name)
            config = write_text(tmp_path / "control.yaml", "config")
            checkpoint = write_text(tmp_path / "selected.zip", "checkpoint")
            frozen_env = write_text(tmp_path / "source" / "UTMIST-AI2-main" / "environment" / "environment.py", "environment")
            output = tmp_path / "suite"
            write_grouped_parity_marker(output, tmp_path / "grouped_vec_env.py", frozen_env)
            args = suite.build_parser().parse_args(
                [
                    "--mode",
                    "grouped-screen",
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
                ]
            )

            code = suite.run_suite(suite.validate_args(args), runner=runner, clock=FakeClock())
            manifest = json.loads((output / "grouped-screen-env64-group2" / "job_manifest.json").read_text())
            expected_marker_sha256 = sha256_file(output / "grouped_game_parity_passed.json")

        self.assertEqual(code, 0)
        self.assertTrue(manifest["files"]["grouped_parity_marker"]["exists"])
        self.assertEqual(
            manifest["files"]["grouped_parity_marker"]["sha256"],
            expected_marker_sha256,
        )

    def test_run_suite_refuses_completed_output_and_skips_jobs_that_cannot_finish_before_deadline(self):
        suite = load_suite()
        clock = FakeClock()
        runner = FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            completed = output / "paired-cache-seed45-cache" / "campaign_summary.json"
            completed.parent.mkdir(parents=True)
            completed.write_text('{"status":"completed"}', encoding="utf-8")
            args = suite.build_parser().parse_args(
                ["--mode", "paired-cache", "--output", str(output), "--deadline-unix", "9999"]
            )
            parsed = suite.validate_args(args)
            with self.assertRaisesRegex(FileExistsError, "completed output"):
                suite.run_suite(parsed, runner=runner, clock=clock)

            output2 = Path(tmp) / "suite2"
            args2 = suite.build_parser().parse_args(
                [
                    "--mode",
                    "smoke-cache",
                    "--output",
                    str(output2),
                    "--deadline-unix",
                    str(clock.time() + 1000),
                    "--per-job-timeout",
                    "900",
                ]
            )
            def slow_runner(job, command, env, timeout):
                result = runner(job, command, env, timeout)
                clock.value += timeout
                return result

            code = suite.run_suite(suite.validate_args(args2), runner=slow_runner, clock=clock)

        self.assertEqual(code, 3)
        self.assertEqual(len(runner.commands), 1)

    def test_run_suite_writes_report_and_aborts_when_summary_counts_do_not_match(self):
        suite = load_suite()

        def bad_runner(job, command, env, timeout):
            job.output.mkdir(parents=True, exist_ok=True)
            (job.output / "campaign_summary.json").write_text(
                json.dumps({"status": "completed", "completed_measured_cycles": job.cycles - 1}),
                encoding="utf-8",
            )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            args = suite.build_parser().parse_args(
                ["--mode", "smoke-cache", "--output", str(output), "--deadline-unix", "9999"]
            )
            code = suite.run_suite(suite.validate_args(args), runner=bad_runner, clock=FakeClock())
            report = json.loads((output / "suite_report.json").read_text())

        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(report["completed_jobs"], 0)
        self.assertIn("measured cycles", report["jobs"][0]["error"])

    def test_run_suite_records_manifest_with_commands_hashes_and_whitelisted_env(self):
        suite = load_suite()
        runner = FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in ("benchmark_campaign.py", "rollout_cache.py", "grouped_vec_env.py", "train_job.py"):
                (tmp_path / name).write_text(name, encoding="utf-8")
            config = tmp_path / "control.yaml"
            checkpoint = tmp_path / "selected.zip"
            config.write_text("config", encoding="utf-8")
            checkpoint.write_text("checkpoint", encoding="utf-8")
            output = tmp_path / "suite"
            args = suite.build_parser().parse_args(
                [
                    "--mode",
                    "smoke-cache",
                    "--output",
                    str(output),
                    "--deadline-unix",
                    "9999",
                    "--source",
                    str(tmp_path),
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
                ]
            )

            code = suite.run_suite(suite.validate_args(args), runner=runner, clock=FakeClock())
            manifest = json.loads((output / "smoke-cache-seed45-cache" / "job_manifest.json").read_text())
            report = json.loads((output / "suite_report.json").read_text())

        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["completed_jobs"], 2)
        self.assertIn("command_sha256", manifest)
        self.assertIn("env_sha256", manifest)
        self.assertEqual(sorted(manifest["env"]), [
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "PYGAME_HIDE_SUPPORT_PROMPT",
            "SDL_AUDIODRIVER",
            "SDL_VIDEODRIVER",
            "WANDB_MODE",
            "WANDB_NAME",
            "WANDB_NOTES",
            "WANDB_RUN_GROUP",
        ])
        self.assertEqual(manifest["env"]["WANDB_NAME"], "smoke-cache-seed45-cache")
        self.assertEqual(manifest["env"]["WANDB_RUN_GROUP"], "throughput-scaling-20260906")
        self.assertIn("arm=cache", manifest["env"]["WANDB_NOTES"])
        self.assertIn("expected_completed_cycles=3", manifest["env"]["WANDB_NOTES"])
        self.assertIn(str(checkpoint), manifest["env"]["WANDB_NOTES"])
        self.assertIn("not gameplay quality", manifest["env"]["WANDB_NOTES"])
        self.assertTrue(manifest["files"]["harness"]["exists"])
        self.assertTrue(manifest["files"]["rollout_cache"]["exists"])
        self.assertTrue(manifest["files"]["grouped_env"]["exists"])
        self.assertTrue(manifest["files"]["train_job"]["exists"])
        self.assertTrue(manifest["files"]["config"]["exists"])
        self.assertTrue(manifest["files"]["checkpoint"]["exists"])

    def test_run_suite_writes_incremental_report_and_active_job_before_wait(self):
        suite = load_suite()
        observed = []

        def inspecting_runner(job, command, env, timeout):
            report = json.loads((job.output.parent / "suite_report.json").read_text())
            active = json.loads((job.output.parent / "active_job.json").read_text())
            observed.append((report, active))
            self.assertEqual(active["name"], job.name)
            self.assertEqual(active["pid"], None)
            self.assertEqual(active["status"], "running")
            job.output.mkdir(parents=True, exist_ok=True)
            (job.output / "campaign_summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "completed_measured_cycles": job.cycles,
                        "requested_measured_cycles": job.cycles,
                        "warmup_cycles": job.warmup,
                        "plan": {"transitions_per_cycle": job.n_envs * job.n_steps},
                    }
                ),
                encoding="utf-8",
            )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            args = suite.build_parser().parse_args(
                ["--mode", "smoke-cache", "--output", str(output), "--deadline-unix", "9999"]
            )
            code = suite.run_suite(suite.validate_args(args), runner=inspecting_runner, clock=FakeClock())
            final_report = json.loads((output / "suite_report.json").read_text())
            active = json.loads((output / "active_job.json").read_text())

        self.assertEqual(code, 0)
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0][0]["status"], "running")
        self.assertEqual(observed[0][0]["jobs"][0]["status"], "running")
        self.assertEqual(observed[1][0]["completed_jobs"], 1)
        self.assertEqual(final_report["completed_jobs"], 2)
        self.assertEqual(active["status"], "completed")

    def test_real_subprocess_runner_records_child_pid_before_wait(self):
        suite = load_suite()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            job = suite.SuiteJob(name="pid-check", arm="control", seed=45, output=output / "pid-check", cycles=1)
            job.output.mkdir(parents=True)
            command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import json, sys; "
                    "Path(sys.argv[1]).write_text(json.dumps({'status':'completed',"
                    "'completed_measured_cycles':1,'requested_measured_cycles':1,"
                    "'warmup_cycles':2,'plan':{'transitions_per_cycle':32768}}))"
                ),
                str(job.output / "campaign_summary.json"),
            ]
            code = suite.run_subprocess_job(
                job,
                command,
                suite.whitelisted_env(job=job, args=type("Args", (), {"checkpoint": Path("/tmp/ckpt.zip")})()),
                10,
            )
            active = json.loads((job.output.parent / "active_job.json").read_text())

        self.assertEqual(code, 0)
        self.assertIsInstance(active["pid"], int)
        self.assertGreater(active["pid"], 0)


if __name__ == "__main__":
    unittest.main()
