"""CPU-only planning, provenance, and sequential supervision checks."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class Clock:
    now = 1000.0

    def time(self):
        return self.now


class CampaignNativePPOAblationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "campaign_native_ppo_ablations.py").is_file(), "ablation launcher must exist")
        self.suite = importlib.import_module("campaign_native_ppo_ablations")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clock = Clock()

    def args(self, mode="screen", *, run=False, extra=()):
        source = self.root / "source"
        for path in (source / "train.py", source / "UTMIST-AI2-main/environment/environment.py",
                     self.root / "train_job.py", self.root / "control.yaml", self.root / "checkpoint.zip"):
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("fixture\n", encoding="utf-8")
        argv = ["--mode", mode, "--source", str(source), "--train-job", str(self.root / "train_job.py"),
                "--config", str(self.root / "control.yaml"), "--checkpoint", str(self.root / "checkpoint.zip"),
                "--output", str(self.root / "output"), "--deadline-unix", "99999", *extra]
        if run:
            argv += ["--run", "--parity-receipt", str(self.root / "receipt.json")]
        return self.suite.validate_args(self.suite.build_parser().parse_args(argv))

    def receipt(self, args):
        import qualify_native_lanes

        value = {"schema": "ai2.native_lanes.parity.v1", "status": "passed", "exit_code": 0, "skipped": 0,
                 **qualify_native_lanes.source_manifest()}
        self.suite.campaign_suite.write_json(args.parity_receipt, value)
        return value

    def completed_output(self, args, job):
        write_json = self.suite.campaign_suite.write_json
        write_json(job.output / "campaign_summary.json", {
            "status": "completed", "completed_cycles_total": job.warmup + job.cycles,
            "completed_measured_cycles": job.cycles, "requested_measured_cycles": job.cycles,
            "warmup_cycles": job.warmup,
            "plan": {"n_envs": job.n_envs, "n_steps": job.n_steps, "batch_size": job.batch_size,
                     "epochs": job.epochs, "transitions_per_cycle": job.n_envs * job.n_steps},
        })
        write_json(job.output / "native_lane_provenance.json", {
            "schema": "ai2.native_ppo_ablations.v1", "lane_device": "cuda", "policy_device": "cuda",
            "optimizations": self.suite.optimizations_for_arm(args, job.arm),
        })
        write_json(job.output / "optimization_runtime.json", {"compile_headline_eligible": True})
        (job.output / "job.log").write_text("raw output\n", encoding="utf-8")

    def run_suite(self, args, runner):
        with mock.patch.object(self.suite, "require_idle_gpu"), mock.patch.object(self.suite, "GPU_LOCK_PATH", self.root / "gpu.lock"):
            return self.suite.run_suite(args, runner=runner, clock=self.clock)

    def report(self, args):
        return json.loads((args.output / "suite_report.json").read_text())

    def test_screen_defaults_to_seven_independent_arms_at_two_sizes(self):
        jobs = self.suite.plan_jobs(self.args())
        expected = ("baseline", "counters", "metrics", "timeouts", "validation", "compiled_loss", "compiled_gae")
        self.assertEqual(len(jobs), 14)
        for size in (256, 4096):
            selected = [job for job in jobs if job.n_envs == size]
            self.assertEqual(tuple(job.arm for job in selected), expected)
            self.assertTrue(all((job.seed, job.n_steps, job.batch_size, job.epochs, job.warmup, job.cycles) ==
                                (45, 128, 8192, 8, 2, 10 if size == 256 else 5) for job in selected))

    def test_paired_rotates_selected_arms_with_matched_geometry_and_seed(self):
        jobs = self.suite.plan_jobs(self.args("paired", extra=("--arms", "baseline", "combined_collector")))
        self.assertEqual(len(jobs), 12)
        for seed in (45, 46, 47):
            for size in (256, 4096):
                selected = [job for job in jobs if job.seed == seed and job.n_envs == size]
                self.assertEqual([job.arm for job in selected], ["combined_collector", "baseline"] if seed == 46 else ["baseline", "combined_collector"])
                self.assertTrue(all(job.cycles == (100 if size == 256 else 20) and job.warmup == 2 for job in selected))

    def test_commands_roundtrip_through_driver_and_flags_are_independent(self):
        import experimental_native_lane_training as driver

        args = self.args()
        expected = {
            "baseline": set(), "counters": {"optimizer_cpu_steps"}, "metrics": {"deferred_metrics"},
            "timeouts": {"deferred_timeouts"}, "validation": {"boundary_validation"},
            "compiled_loss": {"deferred_metrics", "compile_loss"}, "compiled_gae": {"compile_gae"},
        }
        for job in self.suite.plan_jobs(args):
            command = self.suite.command_for_job(args, job)
            self.assertEqual(Path(command[1]).name, "experimental_native_lane_training.py")
            parsed = driver.build_parser().parse_args(command[2:])
            options = driver.validate_experiment(parsed)
            self.assertEqual({name for name, value in vars(options).items() if value}, expected[job.arm])
            self.assertEqual((parsed.device, parsed.lane_device, parsed.mode), ("cuda", "cuda", "baseline"))
            self.assertEqual(Path(parsed.source), args.source)
            self.assertEqual(Path(parsed.checkpoint), args.checkpoint)

    def test_custom_and_predeclared_combinations_are_explicit(self):
        args = self.args("paired", extra=("--combined", "counters", "deferred_metrics", "boundary-validation"))
        self.assertEqual({job.arm for job in self.suite.plan_jobs(args)}, {"baseline", "combined"})
        self.assertEqual(self.suite.optimizations_for_arm(args, "combined"), self.suite.optimizations_for_arm(args, "combined_exact"))
        self.assertTrue(all(self.suite.optimizations_for_arm(args, "combined_all").values()))

    def test_invalid_selection_deadline_and_timeout_are_rejected(self):
        for extra in (("--arms", "unknown"), ("--arms", "baseline", "baseline"),
                      ("--arms", "combined"), ("--combined", "compile_loss"),
                      ("--combined", "typo"), ("--per-job-timeout", "901"),
                      ("--per-job-timeout", "nan"), ("--deadline-unix", "inf")):
            with self.subTest(extra=extra), self.assertRaises((ValueError, SystemExit)):
                self.args(extra=extra)
        with self.assertRaises(ValueError):
            self.args("paired", extra=("--arms", "counters"))

    def test_environment_forces_offline_cuda_native_execution_with_purpose(self):
        args = self.args()
        job = self.suite.plan_jobs(args)[1]
        with mock.patch.dict("os.environ", {"WANDB_MODE": "online", "AI2_NATIVE_INPUT": "1", "PYTHONPATH": "unrelated"}):
            env = self.suite.env_for_job(args, job)
        self.assertEqual(env["WANDB_MODE"], "offline")
        self.assertEqual((env["AI2_NATIVE_INPUT"], env["AI2_NATIVE_OBSERVATION"]), ("0", "0"))
        self.assertIn(str(args.source), env["PYTHONPATH"])
        self.assertIn("optimizer_cpu_steps", env["WANDB_NOTES"])
        self.assertIn("throughput", env["WANDB_NOTES"])

    def test_dry_run_preserves_plan_hashes_without_gpu_or_subprocess(self):
        args = self.args()
        runner = mock.Mock(side_effect=AssertionError("dry run launched"))
        with mock.patch.object(self.suite, "require_idle_gpu", side_effect=AssertionError("dry run probed GPU")):
            self.assertEqual(self.suite.run_suite(args, runner=runner, clock=self.clock), 0)
        runner.assert_not_called()
        self.assertEqual(self.report(args)["status"], "dry_run")
        manifest = json.loads((args.output / "source_manifest.json").read_text())
        self.assertEqual(manifest["checkpoint"]["sha256"], self.suite.campaign_suite.sha256_file(args.checkpoint))
        self.assertIn("ppo_update_optimizations.py", manifest)
        plan = json.loads((args.output / "native_ppo_ablations_plan.json").read_text())
        self.assertEqual(len(plan["jobs"]), 14)
        self.assertEqual(plan["policy_device"], "cuda")
        self.assertTrue(all(record["env"]["WANDB_MODE"] == "offline" for record in plan["jobs"]))

    def test_run_rejects_missing_receipt_and_missing_required_source(self):
        args = self.args(run=True)
        runner = mock.Mock()
        self.assertEqual(self.run_suite(args, runner), 2)
        runner.assert_not_called()
        self.assertEqual(self.report(args)["status"], "validation_failed")
        args.output = self.root / "missing-source"
        args.run = False
        args.parity_receipt = None
        (args.source / "train.py").unlink()
        self.assertEqual(self.run_suite(args, runner), 2)
        self.assertIn("required input missing", self.report(args)["error"])

    def test_missing_experiment_implementation_cannot_disappear_from_manifest(self):
        args = self.args()
        with mock.patch.object(self.suite, "ROOT", self.root / "incomplete-candidate"):
            self.assertEqual(self.run_suite(args, mock.Mock()), 2)
        error = self.report(args)["error"]
        self.assertIn("ppo_update_optimizations.py", error)
        self.assertIn("experimental_native_lane_training.py", error)

    def test_current_collector_hash_is_required_by_receipt_gate(self):
        args = self.args(run=True)
        receipt = self.receipt(args)
        receipt["source_hashes"]["tensor_ppo.py"] = "0" * 64
        self.suite.campaign_suite.write_json(args.parity_receipt, receipt)
        runner = mock.Mock()
        self.assertEqual(self.run_suite(args, runner), 2)
        runner.assert_not_called()
        self.assertIn("tensor_ppo.py", self.report(args)["error"])

    def test_run_is_sequential_and_validates_each_job_provenance(self):
        args = self.args(run=True, extra=("--arms", "baseline", "counters"))
        self.receipt(args)
        calls = []

        def runner(job, command, env, timeout):
            self.assertEqual(self.report(args)["completed_jobs"], len(calls))
            self.assertTrue((job.output / "job_manifest.json").is_file())
            self.completed_output(args, job)
            calls.append(job)
            return 0

        self.assertEqual(self.run_suite(args, runner), 0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(self.report(args)["status"], "completed")
        self.assertTrue(all((job.output / "job.log").read_text() == "raw output\n" for job in calls))

    def test_failed_provenance_or_changed_source_stops_before_next_job(self):
        for failure in ("flags", "checkpoint"):
            with self.subTest(failure=failure):
                args = self.args(run=True, extra=("--arms", "baseline", "counters"))
                args.output = self.root / failure
                self.receipt(args)
                calls = []

                def runner(job, command, env, timeout):
                    self.completed_output(args, job)
                    calls.append(job)
                    if failure == "flags":
                        path = job.output / "native_lane_provenance.json"
                        value = json.loads(path.read_text())
                        value["optimizations"]["deferred_metrics"] = True
                        self.suite.campaign_suite.write_json(path, value)
                    else:
                        args.checkpoint.write_text("changed\n")
                    return 0

                self.assertNotEqual(self.run_suite(args, runner), 0)
                self.assertEqual(len(calls), 1)
                self.assertEqual(self.report(args)["status"], "aborted" if failure == "flags" else "source_changed")
                self.assertTrue((calls[0].output / "job.log").is_file())

    def test_compiled_jobs_require_explicit_clean_measured_compilation_receipt(self):
        args = self.args(extra=("--arms", "compiled_loss", "compiled_gae"))
        for job in self.suite.plan_jobs(args)[:2]:
            self.completed_output(args, job)
            self.assertIsNone(self.suite.validate_job_summary(args, job))
            path = job.output / "optimization_runtime.json"
            for eligibility in (False, None, 1):
                self.suite.campaign_suite.write_json(path, {"compile_headline_eligible": eligibility})
                with self.subTest(arm=job.arm, eligibility=eligibility):
                    self.assertIsNotNone(self.suite.validate_job_summary(args, job))
            path.unlink()
            self.assertIsNotNone(self.suite.validate_job_summary(args, job))

    def test_deadline_and_busy_gpu_prevent_subprocess_start(self):
        for failure in ("deadline", "busy"):
            with self.subTest(failure=failure):
                args = self.args(run=True)
                args.output = self.root / failure
                self.receipt(args)
                runner = mock.Mock()
                if failure == "deadline":
                    args.deadline_unix = self.clock.time() + 100
                    self.assertEqual(self.run_suite(args, runner), 3)
                else:
                    with mock.patch.object(self.suite, "GPU_LOCK_PATH", self.root / "gpu.lock"), mock.patch.object(self.suite, "require_idle_gpu", side_effect=RuntimeError("GPU busy")):
                        self.assertEqual(self.suite.run_suite(args, runner=runner, clock=self.clock), 2)
                runner.assert_not_called()

    def test_failed_process_and_timeout_keep_raw_logs_and_stop_remaining_jobs(self):
        for failure in ("nonzero", "exception", "timeout"):
            with self.subTest(failure=failure):
                args = self.args(run=True)
                args.output = self.root / failure
                self.receipt(args)
                calls = []

                def runner(job, command, env, timeout):
                    self.completed_output(args, job)
                    calls.append(job)
                    if failure == "nonzero":
                        return 23
                    if failure == "exception":
                        raise RuntimeError("subprocess failed")
                    self.clock.now += timeout
                    return 0

                self.assertEqual(self.run_suite(args, runner), 1)
                self.assertEqual(len(calls), 1)
                report = self.report(args)
                self.assertEqual(report["status"], "aborted")
                self.assertEqual(report["jobs"][0]["status"], "failed")
                self.assertEqual(report["jobs"][0]["return_code"], {"nonzero": 23, "exception": 1, "timeout": 124}[failure])
                self.assertEqual((calls[0].output / "job.log").read_text(), "raw output\n")

    def test_existing_result_root_is_never_overwritten(self):
        args = self.args()
        args.output.mkdir()
        sentinel = args.output / "failed.log"
        sentinel.write_text("keep me\n")
        with self.assertRaises(FileExistsError):
            self.run_suite(args, mock.Mock())
        self.assertEqual(sentinel.read_text(), "keep me\n")


if __name__ == "__main__":
    unittest.main()
