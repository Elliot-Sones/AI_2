
from ai2.paths import source_path
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest


class CampaignCorrectedScalingTests(unittest.TestCase):
    def test_plan_is_exact_rotated_group4_matrix(self):
        module = load_module()
        args = module.validate_args(base_args())

        jobs = module.plan_jobs(args)

        self.assertEqual(len(jobs), 12)
        self.assertEqual(
            [(job.seed, job.n_envs, job.n_steps, job.group_size) for job in jobs],
            [
                (45, 32, 1024, 4),
                (45, 64, 512, 4),
                (45, 128, 256, 4),
                (45, 256, 128, 4),
                (46, 64, 512, 4),
                (46, 128, 256, 4),
                (46, 256, 128, 4),
                (46, 32, 1024, 4),
                (47, 128, 256, 4),
                (47, 256, 128, 4),
                (47, 32, 1024, 4),
                (47, 64, 512, 4),
            ],
        )
        for job in jobs:
            self.assertEqual(job.arm, "grouped")
            self.assertEqual(job.batch_size, 8192)
            self.assertEqual(job.epochs, 8)
            self.assertEqual(job.warmup, 2)
            self.assertEqual(job.cycles, 100)
            self.assertFalse(job.resident_buffer)
            self.assertEqual(job.n_envs * job.n_steps, 32768)

    def test_dry_run_default_writes_plan_without_executing(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = module.validate_args(base_args(output=Path(tmp) / "suite"))
            runner = RecordingRunner()

            exit_code = module.run_suite(args, runner=runner, clock=FakeClock())

            self.assertEqual(exit_code, 0)
            self.assertEqual(runner.calls, [])
            plan = json.loads((args.output / "dry_run_plan.json").read_text())
            self.assertEqual(plan["status"], "dry_run")
            self.assertFalse(plan["execute"])
            self.assertEqual(plan["planned_jobs"], 12)
            self.assertEqual(plan["transitions_per_cycle"], 32768)
            self.assertEqual(plan["jobs"][0]["name"], "corrected-scaling-seed45-env32-group4")
            self.assertIn("--maxseconds", plan["jobs"][0]["command"])
            self.assertIn("provenance", plan)
            provenance_lines = (args.output / "invocation_provenance.jsonl").read_text().splitlines()
            self.assertEqual(len(provenance_lines), 1)
            self.assertEqual(json.loads(provenance_lines[0])["status"], "dry_run")
            self.assertTrue((args.output / "suite_report.json").exists())
            self.assertFalse((args.output / "corrected-scaling-seed45-env32-group4").exists())

    def test_execute_requires_cuda_timeout_disk_and_parity_marker(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, device="cpu"))
            with self.assertRaisesRegex(ValueError, "CUDA"):
                module.preflight(args, disk_usage=FakeDiskUsage(free=4 * 1024**3))

            with self.assertRaisesRegex(ValueError, "per-job-timeout"):
                module.validate_args(base_args(output=root / "out", **paths, execute=True, per_job_timeout=601))

            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, per_job_timeout=600))
            disk_usage = FakeDiskUsage(free=3 * 1024**3)
            with self.assertRaisesRegex(RuntimeError, "at least 3 GiB free disk"):
                module.preflight(args, disk_usage=disk_usage)
            self.assertEqual(disk_usage.paths, [args.output])

            write_marker(args, runner_sha="wrong", frozen_sha=sha256_file(paths["source"] / "UTMIST-AI2-main/environment/environment.py"))
            with self.assertRaisesRegex(RuntimeError, "runner_sha256"):
                module.preflight(args, disk_usage=FakeDiskUsage(free=4 * 1024**3))

    def test_execute_requires_all_required_provenance_files_to_exist(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True))
            write_valid_marker(args)

            paths["harness"].unlink()
            with self.assertRaisesRegex(RuntimeError, "harness"):
                module.preflight(args, disk_usage=FakeDiskUsage(free=4 * 1024**3))

    def test_frozen_train_provenance_uses_wrapper_source_train_contract(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True))
            write_valid_marker(args)

            self.assertEqual(module.frozen_train_path(args), source_path(args.source, "train.py"))
            record = module.provenance(args)["frozen_train"]
            self.assertTrue(record["exists"])
            self.assertEqual(record["path"], str((source_path(args.source, "train.py")).resolve()))

    def test_validate_args_rejects_nonfinite_timeout_and_past_execute_deadline(self):
        module = load_module()

        for timeout in (float("nan"), float("inf"), 0.0, -1.0, 601.0):
            with self.assertRaisesRegex(ValueError, "per-job-timeout"):
                module.validate_args(base_args(per_job_timeout=timeout))
        for deadline in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "deadline-unix"):
                module.validate_args(base_args(deadline_unix=deadline))
        with self.assertRaisesRegex(ValueError, "deadline-unix"):
            module.validate_args(base_args(execute=True, deadline_unix=time.time() - 1.0))

    def test_command_wrapper_adds_graceful_maxseconds_below_subprocess_timeout(self):
        module = load_module()
        args = module.validate_args(base_args(per_job_timeout=240.0))
        command = module.command_for_job(args, module.plan_jobs(args)[0])

        self.assertIn("--maxseconds", command)
        value = float(command[command.index("--maxseconds") + 1])
        self.assertGreater(value, 0.0)
        self.assertLess(value, args.per_job_timeout)

    def test_execute_runs_sequentially_saves_manifest_and_resumes_only_completed_jobs(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, deadline_unix=time.time() + 10_000.0))
            write_valid_marker(args)
            first_job = module.plan_jobs(args)[0]
            write_matching_completed_job(module, args, first_job)
            runner = RecordingRunner(write_success=True)

            exit_code = module.run_suite(args, runner=runner, clock=FakeClock(start=100.0), disk_usage=FakeDiskUsage(free=4 * 1024**3))

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(runner.calls), 11)
            self.assertEqual(runner.calls[0][0].name, "corrected-scaling-seed45-env64-group4")
            report = json.loads((args.output / "suite_report.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["planned_jobs"], 12)
            self.assertEqual(report["completed_jobs"], 12)
            self.assertEqual(report["jobs"][0]["status"], "completed_existing")
            invocation_records = [
                json.loads(line)
                for line in (args.output / "invocation_provenance.jsonl").read_text().splitlines()
            ]
            self.assertEqual(invocation_records[-1]["status"], "execute")
            self.assertEqual(invocation_records[-1]["provenance"]["config"]["path"], str(args.config))
            self.assertEqual(invocation_records[-1]["provenance"]["grouped_parity_marker"]["exists"], True)
            manifest = json.loads((runner.calls[0][0].output / "job_manifest.json").read_text())
            self.assertEqual(manifest["job"]["group_size"], 4)
            self.assertIn("command_sha256", manifest)
            self.assertIn("frozen_train", manifest["files"])
            self.assertIn("frozen_environment", manifest["files"])
            self.assertIn("source_snapshot", "\n".join(str(path) for path in runner.calls[0][0].output.rglob("*")))

    def test_resume_requires_matching_manifest_command_env_and_file_hashes(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, deadline_unix=time.time() + 10_000.0))
            write_valid_marker(args)
            job = module.plan_jobs(args)[0]

            write_completed_summary(job)
            with self.assertRaisesRegex(FileExistsError, "missing manifest"):
                module.existing_completed(args, job)

            write_matching_completed_job(module, args, job)
            self.assertTrue(module.existing_completed(args, job))

            manifest_path = job.output / "job_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["command"].append("--unexpected")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(FileExistsError, "command"):
                module.existing_completed(args, job)

            write_matching_completed_job(module, args, job)
            manifest = json.loads(manifest_path.read_text())
            manifest["env"]["WANDB_NAME"] = "wrong"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(FileExistsError, "env"):
                module.existing_completed(args, job)

            write_matching_completed_job(module, args, job)
            paths["config"].write_text("changed\n")
            with self.assertRaisesRegex(FileExistsError, "config"):
                module.existing_completed(args, job)

            write_matching_completed_job(module, args, job)
            (source_path(paths["source"], "train.py")).write_text("changed source train\n")
            with self.assertRaisesRegex(FileExistsError, "frozen_train"):
                module.existing_completed(args, job)

            write_matching_completed_job(module, args, job)
            manifest_path = job.output / "job_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["rollout_cache"]["exists"] = False
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(FileExistsError, "rollout_cache"):
                module.existing_completed(args, job)

    def test_invocation_provenance_is_append_only_and_does_not_overwrite_old_file(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = module.validate_args(base_args(output=Path(tmp) / "suite"))
            args.output.mkdir(parents=True)
            legacy = args.output / "suite_provenance.json"
            legacy.write_text('{"legacy": true}\n')

            module.run_suite(args, runner=RecordingRunner(), clock=FakeClock())
            module.run_suite(args, runner=RecordingRunner(), clock=FakeClock())

            self.assertEqual(legacy.read_text(), '{"legacy": true}\n')
            records = [json.loads(line) for line in (args.output / "invocation_provenance.jsonl").read_text().splitlines()]
            self.assertEqual([record["status"] for record in records], ["dry_run", "dry_run"])
            self.assertLess(records[0]["created_unix"], records[1]["created_unix"])

    def test_execute_refuses_incomplete_existing_output_and_obeys_stop_request_between_jobs(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, deadline_unix=time.time() + 10_000.0))
            write_valid_marker(args)
            incomplete = module.plan_jobs(args)[0].output
            incomplete.mkdir(parents=True)
            (incomplete / "campaign_summary.json").write_text(json.dumps({"status": "failed"}))
            with self.assertRaisesRegex(FileExistsError, "incomplete existing output"):
                module.run_suite(args, runner=RecordingRunner(), clock=FakeClock(start=100.0), disk_usage=FakeDiskUsage(free=4 * 1024**3))

            args = module.validate_args(base_args(output=root / "out2", **paths, execute=True, deadline_unix=time.time() + 10_000.0))
            write_valid_marker(args)
            runner = RecordingRunner(write_success=True, stop_after_first=True)
            exit_code = module.run_suite(args, runner=runner, clock=FakeClock(start=100.0), disk_usage=FakeDiskUsage(free=4 * 1024**3))
            self.assertEqual(exit_code, 4)
            self.assertEqual(len(runner.calls), 1)
            report = json.loads((args.output / "suite_report.json").read_text())
            self.assertEqual(report["status"], "stop_requested")

    def test_execute_refuses_preexisting_running_active_job(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, deadline_unix=time.time() + 10_000.0))
            write_valid_marker(args)
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "active_job.json").write_text(json.dumps({"status": "running", "name": "old-job"}))

            with self.assertRaisesRegex(RuntimeError, "active job"):
                module.run_suite(args, runner=RecordingRunner(), clock=FakeClock(start=100.0), disk_usage=FakeDiskUsage(free=4 * 1024**3))

    def test_deadline_requires_timeout_plus_reserve_and_does_not_start_job(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            deadline = time.time() + 10_000.0
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, deadline_unix=deadline))
            write_valid_marker(args)
            runner = RecordingRunner()

            exit_code = module.run_suite(args, runner=runner, clock=FakeClock(start=deadline - 899.0), disk_usage=FakeDiskUsage(free=4 * 1024**3))

            self.assertEqual(exit_code, 3)
            self.assertEqual(runner.calls, [])
            report = json.loads((args.output / "suite_report.json").read_text())
            self.assertEqual(report["status"], "deadline_insufficient")
            self.assertEqual(report["jobs"][0]["status"], "skipped_deadline")


def load_module():
    path = source_path(Path(__file__).resolve().parents[1], "campaign_corrected_scaling.py")
    spec = importlib.util.spec_from_file_location("campaign_corrected_scaling", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["campaign_corrected_scaling"] = module
    spec.loader.exec_module(module)
    return module


def base_args(**overrides):
    values = {
        "output": Path("out"),
        "deadline_unix": time.time() + 10_000.0,
        "per_job_timeout": 600.0,
        "device": "cuda",
        "phase": "0b",
        "config": Path("control.yaml"),
        "checkpoint": Path("selected.zip"),
        "source": Path("source"),
        "train_job": Path("train_job.py"),
        "harness": Path("benchmark_campaign.py"),
        "rollout_cache_path": Path("rollout_cache.py"),
        "grouped_env_path": Path("grouped_vec_env.py"),
        "execute": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def create_required_files(root):
    source = root / "source"
    frozen_train = source_path(source, "train.py")
    frozen_env = source / "UTMIST-AI2-main/environment/environment.py"
    frozen_env.parent.mkdir(parents=True)
    source.mkdir(parents=True, exist_ok=True)
    frozen_train.write_text("frozen-train\n")
    frozen_env.write_text("frozen-env\n")
    files = {
        "config": root / "control.yaml",
        "checkpoint": root / "selected.zip",
        "source": source,
        "train_job": root / "train_job.py",
        "harness": source_path(root, "benchmark_campaign.py"),
        "rollout_cache_path": source_path(root, "rollout_cache.py"),
        "grouped_env_path": source_path(root, "grouped_vec_env.py"),
    }
    for key, path in files.items():
        if key == "source":
            continue
        path.write_text(f"{key}\n")
    return files


def write_marker(args, *, runner_sha, frozen_sha):
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "grouped_game_parity_passed.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "exit_code": 0,
                "skipped": 0,
                "steps_per_case": 1000,
                "group_sizes": [2, 4, 8],
                "runner_sha256": runner_sha,
                "frozen_environment_sha256": frozen_sha,
            }
        )
    )


def write_valid_marker(args):
    write_marker(
        args,
        runner_sha=sha256_file(args.grouped_env_path),
        frozen_sha=sha256_file(args.source / "UTMIST-AI2-main/environment/environment.py"),
    )


def write_completed_summary(job):
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
        )
    )


def write_matching_completed_job(module, args, job):
    command = module.command_for_job(args, job)
    env = module.campaign_suite.whitelisted_env(job=job, args=args)
    job.output.mkdir(parents=True, exist_ok=True)
    module.write_job_manifest(args, job, command, env)
    write_completed_summary(job)


def sha256_file(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeDiskUsage:
    def __init__(self, *, free):
        self.free = free
        self.paths = []

    def __call__(self, path):
        self.paths.append(Path(path))
        return type("DiskUsage", (), {"total": 8 * 1024**3, "used": 8 * 1024**3 - self.free, "free": self.free})()


class RecordingRunner:
    def __init__(self, *, write_success=False, stop_after_first=False):
        self.calls = []
        self.write_success = write_success
        self.stop_after_first = stop_after_first

    def __call__(self, job, command, env, timeout):
        self.calls.append((job, command, env, timeout))
        if self.stop_after_first:
            (job.output.parent / "stop.request").write_text("stop\n")
        if self.write_success:
            write_completed_summary(job)
        return 0


class FakeClock:
    def __init__(self, start=0.0):
        self.current = start

    def time(self):
        self.current += 1.0
        return self.current


if __name__ == "__main__":
    unittest.main()
