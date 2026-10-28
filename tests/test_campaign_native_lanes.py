"""CPU-only command, qualification-gate, and sequential supervisor checks."""

from ai2.paths import source_inventory, source_path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = source_path(ROOT, "campaign_native_lanes.py")


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def time(self):
        return self.now


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def completed_summary(job):
    return {
        "status": "completed",
        "completed_cycles_total": job.warmup + job.cycles,
        "completed_measured_cycles": job.cycles,
        "requested_measured_cycles": job.cycles,
        "warmup_cycles": job.warmup,
        "plan": {
            "n_envs": job.n_envs, "n_steps": job.n_steps,
            "batch_size": job.batch_size, "epochs": job.epochs,
            "transitions_per_cycle": job.n_envs * job.n_steps,
        },
    }


class CampaignNativeLanesTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE_PATH.is_file(), "native lane campaign launcher must exist")
        spec = importlib.util.spec_from_file_location("campaign_native_lanes", MODULE_PATH)
        self.suite = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.suite
        spec.loader.exec_module(self.suite)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clock = Clock()

    def args(self, mode="smoke", run=False, **overrides):
        source = self.root / "source"
        for path in (source_path(source, "train.py"), source / "UTMIST-AI2-main/environment/environment.py",
                     self.root / "train_job.py", self.root / "control.yaml", self.root / "checkpoint.zip"):
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("fixture\n", encoding="utf-8")
        argv = ["--mode", mode, "--source", str(source), "--train-job", str(self.root / "train_job.py"),
                "--config", str(self.root / "control.yaml"), "--checkpoint", str(self.root / "checkpoint.zip"),
                "--output", str(self.root / "output"), "--deadline-unix", "99999"]
        if run:
            argv += ["--run", "--parity-receipt", str(self.root / "receipt.json")]
        args = self.suite.validate_args(self.suite.build_parser().parse_args(argv))
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def receipt(self, args, **overrides):
        import tools.validation.qualify_native_lanes as qualify_native_lanes

        value = {
            "schema": "ai2.native_lanes.parity.v1", "status": "passed", "exit_code": 0, "skipped": 0,
            **qualify_native_lanes.source_manifest(),
        }
        value.update(overrides)
        write_json(args.parity_receipt, value)
        return value

    def run_suite(self, args, runner):
        with mock.patch.object(self.suite, "require_idle_gpu"), mock.patch.object(self.suite, "GPU_LOCK_PATH", self.root / "gpu.lock"):
            return self.suite.run_suite(args, runner=runner, clock=self.clock)

    def report(self, args):
        return json.loads((args.output / "suite_report.json").read_text())

    def test_paired_plan_rotates_all_three_arms_and_matches_rollout_geometry(self):
        jobs = self.suite.plan_jobs(self.args("paired"))
        self.assertEqual([(job.seed, job.arm) for job in jobs], [
            (45, "python_cpu"), (45, "native_cpu"), (45, "native_cuda"),
            (46, "native_cpu"), (46, "native_cuda"), (46, "python_cpu"),
            (47, "native_cuda"), (47, "python_cpu"), (47, "native_cpu"),
        ])
        self.assertTrue(all((job.n_envs, job.n_steps, job.batch_size, job.epochs, job.warmup, job.cycles) ==
                            (256, 128, 8192, 8, 2, 100) for job in jobs))

    def test_smoke_has_exactly_two_updates_each_native_arm(self):
        jobs = self.suite.plan_jobs(self.args())
        self.assertEqual([(job.arm, job.warmup, job.cycles) for job in jobs],
                         [("native_cpu", 0, 2), ("native_cuda", 0, 2)])

    def test_scaling_is_native_only_with_explicit_rollout_change(self):
        args = self.args("scaling")
        jobs = self.suite.plan_jobs(args)
        self.assertEqual({(job.n_envs, job.arm) for job in jobs},
                         {(n, arm) for n in (1024, 4096) for arm in ("native_cpu", "native_cuda")})
        self.assertEqual([(job.n_envs, job.arm) for job in jobs[:2]],
                         [(1024, "native_cuda"), (4096, "native_cuda")])
        self.assertTrue(all((job.seed, job.n_steps, job.batch_size, job.epochs, job.warmup, job.cycles) ==
                            (45, 128, 8192, 8, 2, 20) for job in jobs))
        self.assertEqual(self.run_suite(args, mock.Mock(side_effect=AssertionError("dry-run launched"))), 0)
        plan = json.loads((args.output / "native_lanes_plan.json").read_text())
        self.assertEqual(plan["source_layout_version"], 2)
        self.assertTrue(plan["rollout_shape_changed"])
        self.assertIn("131072", plan["comparison_note"])
        self.assertIn("524288", plan["comparison_note"])

    def test_commands_use_cuda_policy_and_correct_environment_backend(self):
        args = self.args("paired")
        for job in self.suite.plan_jobs(args)[:3]:
            with self.subTest(arm=job.arm):
                command = self.suite.command_for_job(args, job)
                env = self.suite.env_for_job(args, job)
                self.assertEqual(env["AI2_NATIVE_INPUT"], "1" if job.arm == "python_cpu" else "0")
                self.assertEqual(env["AI2_NATIVE_OBSERVATION"], "1" if job.arm == "python_cpu" else "0")
                self.assertEqual(command[command.index("--device") + 1], "cuda")
                self.assertEqual(command[command.index("--source") + 1], str(args.source))
                self.assertEqual(command[command.index("--checkpoint") + 1], str(args.checkpoint))
                self.assertEqual(command[command.index("--group-size") + 1], "4" if job.arm == "python_cpu" else "1")
                if job.arm == "python_cpu":
                    self.assertEqual(Path(command[1]).name, "benchmark_campaign.py")
                    self.assertEqual(Path(command[command.index("--grouped-env-path") + 1]).name,
                                     "native_isolated_vec_env.py")
                    self.assertNotIn("--native-lanes", command)
                else:
                    self.assertEqual(Path(command[1]).name, "benchmark_native_lane_training.py")
                    self.assertIn("--native-lanes", command)
                    self.assertEqual(command[command.index("--lane-device") + 1], job.arm.removeprefix("native_"))

    def test_native_arms_bypass_real_grouped_environment_patch_without_frozen_helper(self):
        campaign = self.suite.benchmark_campaign
        for mode in ("smoke", "paired", "scaling"):
            args = self.args(mode)
            for job in self.suite.plan_jobs(args):
                if job.arm == "python_cpu":
                    continue
                with self.subTest(mode=mode, arm=job.arm, n_envs=job.n_envs):
                    command = self.suite.command_for_job(args, job)
                    self.assertNotIn("--grouped-env-path", command)
                    patch_args = SimpleNamespace(
                        group_size=int(command[command.index("--group-size") + 1]),
                        grouped_env_path=source_path(args.source, "grouped_vec_env.py"),
                    )
                    self.assertFalse(patch_args.grouped_env_path.exists())
                    original = object()
                    train_job = SimpleNamespace(SubprocVecEnv=original)
                    with mock.patch.object(campaign, "load_grouped_env_module", side_effect=AssertionError("unexpected grouped helper import")) as loader:
                        self.assertIsNone(campaign.patch_grouped_vec_env(train_job, patch_args))
                    loader.assert_not_called()
                    self.assertIs(train_job.SubprocVecEnv, original)

    def test_dry_run_needs_no_receipt_and_retains_commands_and_hashes(self):
        args = self.args("paired")
        runner = mock.Mock()
        self.assertEqual(self.run_suite(args, runner), 0)
        runner.assert_not_called()
        self.assertEqual(self.report(args)["status"], "dry_run")
        manifest = json.loads((args.output / "source_manifest.json").read_text())
        self.assertEqual(manifest["checkpoint"]["sha256"], self.suite.campaign_suite.sha256_file(args.checkpoint))
        self.assertIn("source/train.py", manifest)
        self.assertIn("native/navigation_lanes.cu", manifest)
        plan = json.loads((args.output / "native_lanes_plan.json").read_text())
        self.assertEqual(len(plan["jobs"]), 9)
        self.assertTrue(all(record["command"] for record in plan["jobs"]))

    def test_manifest_covers_packaged_implementation_and_layout_support(self):
        inputs = self.suite.source_inputs(self.args())
        for name in ("ai2/paths.py", "ai2/__init__.py", "ai2/runtime/__init__.py",
                     "ai2/runtime/campaign_native_lanes.py", "ai2/training/curriculum_ppo.py",
                     "curriculum_ppo.py"):
            self.assertEqual(inputs[name], ROOT / name)

    def test_receipt_rejects_flat_layout_receipt_even_with_current_candidate_hashes(self):
        args = self.args(run=True)
        receipt = self.receipt(args)
        for version in (None, 1):
            receipt["source_layout_version"] = version
            write_json(args.parity_receipt, receipt)
            with self.assertRaisesRegex(ValueError, "source layout"):
                self.suite.validate_parity_receipt(args)

    def test_receipt_requires_layout_support_hashes(self):
        args = self.args(run=True)
        receipt = self.receipt(args)
        for name in ("ai2/paths.py", "ai2/__init__.py", "ai2/training/__init__.py"):
            altered = dict(receipt["source_hashes"])
            altered.pop(name, None)
            write_json(args.parity_receipt, {**receipt, "source_hashes": altered})
            with self.assertRaisesRegex(ValueError, "missing"):
                self.suite.validate_parity_receipt(args)

    def test_run_requires_receipt_and_rejects_legacy_failed_skipped_or_changed_sources(self):
        cases = [None, {"schema": "legacy"}, {"status": "failed"}, {"skipped": 1}, {"exit_code": 1},
                 {"native_sources": {}}, {"source_hashes": {"native_lane_vec_env.py": "stale"}}]
        for index, overrides in enumerate(cases):
            with self.subTest(overrides=overrides):
                args = self.args(run=True, output=self.root / f"output{index}")
                if overrides is not None:
                    self.receipt(args, **overrides)
                runner = mock.Mock()
                self.assertEqual(self.run_suite(args, runner), 2)
                runner.assert_not_called()
                self.assertEqual(self.report(args)["status"], "validation_failed")

    def test_receipt_rejects_changed_tensor_adapter_and_every_other_recorded_source(self):
        args = self.args(run=True)
        receipt = self.receipt(args)
        recorded = receipt["source_hashes"]
        self.assertIn("tensor_ppo.py", recorded)
        self.assertIn("native_lane_cpu_adapter.py", recorded)
        for name in (*recorded, "campaign_suite.py"):
            with self.subTest(name=name):
                altered = {**recorded, name: "0" * 64}
                write_json(args.parity_receipt, {**receipt, "source_hashes": altered})
                with self.assertRaisesRegex(ValueError, "hash"):
                    self.suite.validate_parity_receipt(args)

    def test_receipt_requires_complete_navigation_collector_adapter_sources(self):
        args = self.args(run=True)
        receipt = self.receipt(args)
        required = ("qualify_native_lanes.py", "native_lane_vec_env.py", "native_lane_cpu_adapter.py",
                    "benchmark_native_lane_training.py", "tensor_ppo.py", "tensor_rollout_buffer.py",
                    "native_navigation_episode.py", "native_navigation_contact.py", "native/navigation_spaces.json",
                    *(f"native/{name}" for name in receipt["native_sources"]))
        for name in required:
            with self.subTest(name=name):
                altered = dict(receipt["source_hashes"])
                del altered[name]
                write_json(args.parity_receipt, {**receipt, "source_hashes": altered})
                with self.assertRaisesRegex(ValueError, "missing"):
                    self.suite.validate_parity_receipt(args)

    def test_receipt_rejects_absolute_parent_and_noncanonical_source_paths(self):
        args = self.args(run=True)
        receipt = self.receipt(args)
        for name in (str(args.checkpoint), "../checkpoint.zip", "native/../../checkpoint.zip", "",
                     "./native_lane_vec_env.py", "native//navigation_lanes.cu", "native\\navigation_lanes.cu"):
            with self.subTest(name=name):
                altered = {**receipt["source_hashes"], name: self.suite.campaign_suite.sha256_file(args.checkpoint)}
                write_json(args.parity_receipt, {**receipt, "source_hashes": altered})
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    self.suite.validate_parity_receipt(args)

    def test_receipt_rejects_symlink_escaping_repository_even_with_matching_hash(self):
        args = self.args(run=True)
        receipt = self.receipt(args)
        candidate_root = self.root / "candidate"
        inventory = source_inventory(ROOT)
        for name in receipt["source_hashes"]:
            path = candidate_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(inventory.get(name, source_path(ROOT, name)).read_bytes())
        (candidate_root / "escaped.py").symlink_to(args.checkpoint)
        receipt["source_hashes"]["escaped.py"] = self.suite.campaign_suite.sha256_file(args.checkpoint)
        write_json(args.parity_receipt, receipt)
        with mock.patch.object(self.suite, "ROOT", candidate_root):
            with self.assertRaisesRegex(ValueError, "unsafe"):
                self.suite.validate_parity_receipt(args)

    def test_run_finishes_sequentially_and_retains_raw_results_and_manifests(self):
        args = self.args(run=True)
        self.receipt(args)
        calls = []
        def runner(job, command, env, timeout):
            if calls:
                self.assertTrue((calls[-1].output / "campaign_summary.json").is_file())
                self.assertEqual(self.report(args)["completed_jobs"], len(calls))
            calls.append(job)
            write_json(job.output / "campaign_summary.json", completed_summary(job))
            (job.output / "job.log").write_text("raw output\n", encoding="utf-8")
            return 0
        self.assertEqual(self.run_suite(args, runner), 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.report(args)["status"], "completed")
        for job in calls:
            self.assertTrue((job.output / "campaign_summary.json").is_file())
            manifest = json.loads((job.output / "job_manifest.json").read_text())
            self.assertEqual(manifest["files"]["checkpoint"]["sha256"], self.suite.campaign_suite.sha256_file(args.checkpoint))
            self.assertEqual(manifest["command"], self.suite.command_for_job(args, job))
            self.assertEqual(manifest["policy_device"], "cuda")
            self.assertEqual(manifest["source_layout_version"], 2)

    def test_process_failure_or_exception_stops_next_arm(self):
        for index, error in enumerate((7, OSError("cannot launch"))):
            with self.subTest(error=error):
                args = self.args(run=True, output=self.root / f"failed{index}")
                self.receipt(args)
                runner = mock.Mock(side_effect=error) if isinstance(error, Exception) else mock.Mock(return_value=error)
                self.assertNotEqual(self.run_suite(args, runner), 0)
                self.assertEqual(runner.call_count, 1)
                self.assertEqual(self.report(args)["status"], "aborted")
                self.assertEqual(self.report(args)["completed_jobs"], 0)

    def test_missing_malformed_or_incomplete_result_never_passes(self):
        for index, mutation in enumerate(("missing", "malformed", "nonobject", "short", "missing_total", "bad_batch")):
            with self.subTest(mutation=mutation):
                args = self.args(run=True, output=self.root / f"badsummary{index}")
                self.receipt(args)
                calls = []
                def runner(job, command, env, timeout):
                    calls.append(job)
                    result = completed_summary(job)
                    if mutation == "missing":
                        return 0
                    if mutation == "malformed":
                        (job.output / "campaign_summary.json").write_text("{", encoding="utf-8")
                        return 0
                    if mutation == "nonobject":
                        result = []
                    elif mutation == "short":
                        result["completed_measured_cycles"] -= 1
                    elif mutation == "missing_total":
                        del result["completed_cycles_total"]
                    elif mutation == "bad_batch":
                        result["plan"]["batch_size"] = 1
                    write_json(job.output / "campaign_summary.json", result)
                    return 0
                self.assertEqual(self.run_suite(args, runner), 2)
                self.assertEqual(len(calls), 1)
                self.assertEqual(self.report(args)["completed_jobs"], 0)

    def test_insufficient_deadline_launches_nothing(self):
        args = self.args(run=True, deadline_unix=self.clock.now + 1)
        self.receipt(args)
        runner = mock.Mock()
        self.assertEqual(self.run_suite(args, runner), 3)
        runner.assert_not_called()
        self.assertEqual(self.report(args)["status"], "deadline_insufficient")

    def test_deadline_rechecked_between_jobs_and_after_last_job(self):
        for index, after_first in enumerate((True, False)):
            with self.subTest(after_first=after_first):
                args = self.args(run=True, output=self.root / f"deadline{index}")
                self.receipt(args)
                calls = []
                def runner(job, command, env, timeout):
                    calls.append(job)
                    write_json(job.output / "campaign_summary.json", completed_summary(job))
                    if after_first or len(calls) == 2:
                        self.clock.now = args.deadline_unix + 1
                    return 0
                self.assertEqual(self.run_suite(args, runner), 3)
                self.assertEqual(len(calls), 1 if after_first else 2)
                self.assertNotEqual(self.report(args)["status"], "completed")
                self.clock.now = 1000

    def test_source_change_between_jobs_aborts(self):
        args = self.args(run=True)
        self.receipt(args)
        calls = []
        def runner(job, command, env, timeout):
            calls.append(job)
            write_json(job.output / "campaign_summary.json", completed_summary(job))
            args.config.write_text("modified\n", encoding="utf-8")
            return 0
        self.assertEqual(self.run_suite(args, runner), 4)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.report(args)["status"], "source_changed")

    def test_successful_child_exit_after_timeout_still_fails(self):
        args = self.args(run=True)
        self.receipt(args)
        calls = []
        def runner(job, command, env, timeout):
            calls.append(job)
            write_json(job.output / "campaign_summary.json", completed_summary(job))
            self.clock.now += timeout
            return 0
        self.assertEqual(self.run_suite(args, runner), 1)
        self.assertEqual(len(calls), 1)
        report = self.report(args)
        self.assertEqual(report["status"], "aborted")
        self.assertEqual(report["completed_jobs"], 0)
        self.assertEqual(report["jobs"][0]["return_code"], 124)
        self.assertIn("timeout", report["jobs"][0]["error"])

    def test_existing_output_is_never_overwritten(self):
        args = self.args()
        args.output.mkdir()
        protected = args.output / "user-result.json"
        protected.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_suite(args, mock.Mock())
        self.assertEqual(protected.read_text(), "preserve")

    def test_gpu_lock_blocks_second_launcher(self):
        path = self.root / "exclusive.lock"
        with self.suite.gpu_lock(path):
            with self.assertRaisesRegex(RuntimeError, "another native lane campaign"):
                with self.suite.gpu_lock(path):
                    self.fail("second launcher acquired GPU lock")
        with self.suite.gpu_lock(path):
            pass

    def test_gpu_probe_rejects_other_compute_jobs_or_unavailable_driver(self):
        for process in (subprocess.CompletedProcess([], 0, "12345\n", ""),
                        subprocess.CompletedProcess([], 1, "", "driver unavailable")):
            with mock.patch.object(self.suite.subprocess, "run", return_value=process):
                with self.assertRaises(RuntimeError):
                    self.suite.require_idle_gpu()
        with mock.patch.object(self.suite.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            self.suite.require_idle_gpu()

    def test_gpu_probe_failure_prevents_launch(self):
        args = self.args(run=True)
        self.receipt(args)
        runner = mock.Mock()
        with mock.patch.object(self.suite, "require_idle_gpu", side_effect=RuntimeError("GPU busy")), \
             mock.patch.object(self.suite, "GPU_LOCK_PATH", self.root / "gpu.lock"):
            self.assertEqual(self.suite.run_suite(args, runner=runner, clock=self.clock), 2)
        runner.assert_not_called()
        self.assertIn("GPU busy", self.report(args)["error"])

    def test_invalid_timeout_or_deadline_fails_validation(self):
        for option, value in (("--per-job-timeout", "0"), ("--per-job-timeout", "901"),
                              ("--per-job-timeout", "nan"), ("--deadline-unix", "nan"),
                              ("--deadline-unix", "inf")):
            with self.subTest(option=option, value=value):
                argv = ["--mode", "smoke", "--output", str(self.root / "unused"), "--deadline-unix", "9999", option, value]
                with self.assertRaises(ValueError):
                    self.suite.validate_args(self.suite.build_parser().parse_args(argv))


if __name__ == "__main__":
    unittest.main()
