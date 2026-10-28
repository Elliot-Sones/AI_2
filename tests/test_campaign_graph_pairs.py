
from ai2.paths import source_path
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = source_path(ROOT, "campaign_graph_pairs.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CampaignGraphPairsTests(unittest.TestCase):
    def test_plan_is_three_alternating_control_graph_pairs(self):
        module = load_module()
        args = module.validate_args(base_args())

        jobs = module.plan_jobs(args)

        self.assertEqual(
            [(job.seed, job.arm, job.n_envs, job.n_steps, job.group_size, job.resident_buffer) for job in jobs],
            [
                (45, "control", 64, 512, 1, False),
                (45, "graph", 64, 512, 1, False),
                (46, "graph", 64, 512, 1, False),
                (46, "control", 64, 512, 1, False),
                (47, "control", 64, 512, 1, False),
                (47, "graph", 64, 512, 1, False),
            ],
        )
        for job in jobs:
            self.assertEqual(job.batch_size, 8192)
            self.assertEqual(job.epochs, 8)
            self.assertEqual(job.warmup, 2)
            self.assertEqual(job.cycles, 100)
            self.assertEqual(job.n_envs * job.n_steps, 32768)

    def test_plan_accepts_only_reviewed_256x128_group4_geometry(self):
        module = load_module()
        args = module.validate_args(base_args(n_envs=256, n_steps=128, group_size=4))

        jobs = module.plan_jobs(args)

        self.assertEqual({(job.n_envs, job.n_steps, job.group_size) for job in jobs}, {(256, 128, 4)})
        self.assertEqual(jobs[0].n_envs * jobs[0].n_steps, 32768)

        with self.assertRaisesRegex(ValueError, "unsupported graph-pairs geometry"):
            module.validate_args(base_args(n_envs=128, n_steps=256, group_size=4))

    def test_dry_run_exercises_validation_for_256_group4_geometry(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = module.validate_args(base_args(output=Path(tmp) / "suite", n_envs=256, n_steps=128, group_size=4))

            exit_code = module.run_suite(args, runner=RecordingRunner(), clock=FakeClock())

            self.assertEqual(exit_code, 0)
            plan = json.loads((args.output / "dry_run_plan.json").read_text())
            self.assertEqual({(job["n_envs"], job["n_steps"], job["group_size"]) for job in plan["jobs"]}, {(256, 128, 4)})
            self.assertIn("--group-size", plan["jobs"][0]["command"])
            self.assertEqual(plan["jobs"][0]["command"][plan["jobs"][0]["command"].index("--group-size") + 1], "4")

    def test_command_delta_is_only_rollout_policy_graph_flag_and_path(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            paths = create_required_files(Path(tmp))
            args = module.validate_args(base_args(output=Path(tmp) / "out", **paths, per_job_timeout=600.0))
            jobs = module.plan_jobs(args)

            control = module.command_for_job(args, jobs[0])
            graph = module.command_for_job(args, jobs[1])

            self.assertNotIn("--rollout-policy-graph", control)
            self.assertIn("--rollout-policy-graph", graph)
            self.assertEqual(graph[graph.index("--rollout-policy-graph-path") + 1], str(args.rollout_policy_graph_path))
            self.assertEqual(control[control.index("--group-size") + 1], "1")
            self.assertEqual(graph[graph.index("--group-size") + 1], "1")
            self.assertEqual(control[control.index("--rollout-cache-path") + 1], str(args.rollout_cache_path))
            self.assertEqual(graph[graph.index("--grouped-env-path") + 1], str(args.grouped_env_path))
            self.assertEqual(control[control.index("--maxseconds") + 1], "570")
            self.assertEqual(graph[graph.index("--maxseconds") + 1], "570")

            graph_without_flags = remove_graph_flags(graph)
            self.assertEqual(normalize_output_path(control), normalize_output_path(graph_without_flags))

    def test_dry_run_writes_plan_and_invocation_without_executing(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = module.validate_args(base_args(output=Path(tmp) / "suite"))
            runner = RecordingRunner()

            exit_code = module.run_suite(args, runner=runner, clock=FakeClock())

            self.assertEqual(exit_code, 0)
            self.assertEqual(runner.calls, [])
            plan = json.loads((args.output / "dry_run_plan.json").read_text())
            self.assertEqual(plan["status"], "dry_run")
            self.assertEqual(plan["planned_jobs"], 6)
            self.assertEqual(plan["transitions_per_cycle"], 32768)
            self.assertEqual(plan["jobs"][1]["arm"], "graph")
            self.assertIn("--rollout-policy-graph", plan["jobs"][1]["command"])
            records = [json.loads(line) for line in (args.output / "invocation_provenance.jsonl").read_text().splitlines()]
            self.assertEqual([record["status"] for record in records], ["dry_run"])
            self.assertEqual(records[0]["provenance"]["parity_summary"]["exists"], False)
            self.assertTrue((args.output / "suite_report.json").exists())

    def test_execute_preflight_requires_cuda_inputs_runtime_graph_and_parity_gate(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, device="cpu"))
            with self.assertRaisesRegex(ValueError, "CUDA"):
                module.preflight(args, disk_usage=FakeDiskUsage(free=4 * 1024**3))

            with self.assertRaisesRegex(ValueError, "per-job-timeout"):
                module.validate_args(base_args(output=root / "out", **paths, execute=True, per_job_timeout=601))

            args = module.validate_args(base_args(output=root / "out", **paths, execute=True))
            with self.assertRaisesRegex(RuntimeError, "at least 3 GiB free disk"):
                module.preflight(args, disk_usage=FakeDiskUsage(free=3 * 1024**3))

            with self.assertRaisesRegex(RuntimeError, "parity"):
                module.preflight(args, disk_usage=FakeDiskUsage(free=4 * 1024**3))

            write_parity_summary(args, byte_parity=False)
            with self.assertRaisesRegex(RuntimeError, "byte_parity"):
                module.preflight(args, disk_usage=FakeDiskUsage(free=4 * 1024**3))

            write_parity_summary(args, helper_sha="changed")
            with self.assertRaisesRegex(RuntimeError, "rollout_policy_graph"):
                module.preflight(args, disk_usage=FakeDiskUsage(free=4 * 1024**3))

            write_parity_summary(args)
            paths["rollout_policy_graph_path"].unlink()
            with self.assertRaisesRegex(RuntimeError, "rollout_policy_graph"):
                module.preflight(args, disk_usage=FakeDiskUsage(free=4 * 1024**3))

    def test_execute_runs_sequentially_writes_manifest_and_validates_graph_summary(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, deadline_unix=time.time() + 10_000.0))
            write_parity_summary(args)
            runner = RecordingRunner(write_success=True)

            exit_code = module.run_suite(args, runner=runner, clock=FakeClock(start=100.0), disk_usage=FakeDiskUsage(free=4 * 1024**3))

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(runner.calls), 6)
            self.assertEqual(runner.calls[0][0].name, "graph-pairs-seed45-control")
            self.assertEqual(runner.calls[1][0].name, "graph-pairs-seed45-graph")
            manifest = json.loads((runner.calls[1][0].output / "job_manifest.json").read_text())
            self.assertIn("rollout_policy_graph", manifest["files"])
            self.assertIn("parity_summary", manifest["files"])
            self.assertEqual(manifest["graph_pairs"]["parity"]["byte_parity"], True)
            self.assertEqual(manifest["job"]["group_size"], 1)
            report = json.loads((args.output / "suite_report.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["completed_jobs"], 6)

    def test_execute_requires_completed_100_cycles_and_runtime_graph_records(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, deadline_unix=time.time() + 10_000.0))
            write_parity_summary(args)
            runner = RecordingRunner(write_success=True, bad_graph_runtime=True)

            exit_code = module.run_suite(args, runner=runner, clock=FakeClock(start=100.0), disk_usage=FakeDiskUsage(free=4 * 1024**3))

            self.assertEqual(exit_code, 2)
            report = json.loads((args.output / "suite_report.json").read_text())
            self.assertEqual(report["status"], "aborted")
            self.assertIn("graph", report["jobs"][1]["error"])

            args2 = module.validate_args(base_args(output=root / "out2", **paths, execute=True, deadline_unix=time.time() + 10_000.0))
            write_parity_summary(args2)
            runner2 = RecordingRunner(write_success=True, completed_cycles=99)
            exit_code = module.run_suite(args2, runner=runner2, clock=FakeClock(start=100.0), disk_usage=FakeDiskUsage(free=4 * 1024**3))
            self.assertEqual(exit_code, 2)
            report = json.loads((args2.output / "suite_report.json").read_text())
            self.assertIn("measured cycles", report["jobs"][0]["error"])

    def test_resume_requires_matching_manifest_command_env_and_file_hashes(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, deadline_unix=time.time() + 10_000.0))
            write_parity_summary(args)
            job = module.plan_jobs(args)[0]

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

            paths["config"].write_text("config\n")
            write_parity_summary(args)
            write_matching_completed_job(module, args, job)
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["parity_summary"]["sha256"] = "wrong"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(FileExistsError, "parity_summary"):
                module.existing_completed(args, job)

    def test_parity_summary_validation_requires_current_grouped_env_hash(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, n_envs=256, n_steps=128, group_size=4))
            write_parity_summary(args)

            module.validate_parity_summary(args)

            args.grouped_env_path.write_text("changed-grouped-env\n")
            with self.assertRaisesRegex(RuntimeError, "grouped_env"):
                module.validate_parity_summary(args)

    def test_parity_summary_validation_requires_matching_geometry_controls(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            args = module.validate_args(base_args(output=root / "out", **paths, n_envs=256, n_steps=128, group_size=4))
            write_parity_summary(args, control_overrides={"group_size": 1})

            with self.assertRaisesRegex(RuntimeError, "group_size"):
                module.validate_parity_summary(args)

    def test_deadline_stop_request_and_active_job_are_fail_closed(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = create_required_files(root)
            deadline = time.time() + 10_000.0
            args = module.validate_args(base_args(output=root / "out", **paths, execute=True, deadline_unix=deadline))
            write_parity_summary(args)
            runner = RecordingRunner()

            exit_code = module.run_suite(args, runner=runner, clock=FakeClock(start=deadline - 899.0), disk_usage=FakeDiskUsage(free=4 * 1024**3))
            self.assertEqual(exit_code, 3)
            self.assertEqual(runner.calls, [])

            args2 = module.validate_args(base_args(output=root / "out2", **paths, execute=True, deadline_unix=time.time() + 10_000.0))
            write_parity_summary(args2)
            args2.output.mkdir(parents=True, exist_ok=True)
            (args2.output / "active_job.json").write_text(json.dumps({"status": "running", "name": "old"}))
            with self.assertRaisesRegex(RuntimeError, "active job"):
                module.run_suite(args2, runner=RecordingRunner(), clock=FakeClock(start=100.0), disk_usage=FakeDiskUsage(free=4 * 1024**3))

            args3 = module.validate_args(base_args(output=root / "out3", **paths, execute=True, deadline_unix=time.time() + 10_000.0))
            write_parity_summary(args3)
            runner3 = RecordingRunner(write_success=True, stop_after_first=True)
            exit_code = module.run_suite(args3, runner=runner3, clock=FakeClock(start=100.0), disk_usage=FakeDiskUsage(free=4 * 1024**3))
            self.assertEqual(exit_code, 4)
            self.assertEqual(len(runner3.calls), 1)


def load_module():
    spec = importlib.util.spec_from_file_location("campaign_graph_pairs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["campaign_graph_pairs"] = module
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
        "rollout_policy_graph_path": Path("rollout_policy_graph.py"),
        "parity_summary": None,
        "verifier_path": Path("benchmark_graph_rollout_parity.py"),
        "policy_compile_path": Path("benchmark_policy_compile.py"),
        "policy_cuda_graph_path": Path("benchmark_policy_cuda_graph.py"),
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
        "rollout_policy_graph_path": source_path(root, "rollout_policy_graph.py"),
        "verifier_path": source_path(root, "benchmark_graph_rollout_parity.py"),
        "policy_compile_path": source_path(root, "benchmark_policy_compile.py"),
        "policy_cuda_graph_path": source_path(root, "benchmark_policy_cuda_graph.py"),
    }
    for key, path in files.items():
        if key == "source":
            continue
        path.write_text(f"{key}\n")
    return {
        "config": files["config"],
        "checkpoint": files["checkpoint"],
        "source": files["source"],
        "train_job": files["train_job"],
        "harness": files["harness"],
        "rollout_cache_path": files["rollout_cache_path"],
        "grouped_env_path": files["grouped_env_path"],
        "rollout_policy_graph_path": files["rollout_policy_graph_path"],
        "verifier_path": files["verifier_path"],
        "policy_compile_path": files["policy_compile_path"],
        "policy_cuda_graph_path": files["policy_cuda_graph_path"],
    }


def write_parity_summary(args, *, byte_parity=True, helper_sha=None, control_overrides=None):
    path = args.parity_summary or (args.output / "graph_rollout_parity_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = lambda p: sha256_file(Path(p))
    helper_sha = helper_sha or sha(args.rollout_policy_graph_path)
    file_records = {
        "source_train": file_record(source_path(args.source, "train.py")),
        "config": file_record(args.config),
        "train_job": file_record(args.train_job),
        "checkpoint": file_record(args.checkpoint),
        "verifier": file_record(args.verifier_path),
        "campaign": file_record(args.harness),
        "grouped_env": file_record(args.grouped_env_path),
        "rollout_policy_graph": {**file_record(args.rollout_policy_graph_path), "sha256": helper_sha},
        "policy_compile": file_record(args.policy_compile_path),
        "policy_cuda_graph": file_record(args.policy_cuda_graph_path),
    }
    controls = {
        "n_envs": int(args.n_envs),
        "n_steps": int(args.n_steps),
        "batch_size": 8192,
        "epochs": 8,
        "seed": 45,
        "phase": "0b",
        "device": "cuda",
        "checkpoint": str(args.checkpoint),
        "group_size": int(args.group_size),
        "grouped_env_path": str(args.grouped_env_path),
    }
    control_controls = {**controls, **(control_overrides or {})}
    control_before = {**file_records, "controls": control_controls}
    graph_before = {**file_records, "controls": controls}
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "diagnostic": "graph_rollout_one_update_parity",
                "parity": {
                    "byte_parity": byte_parity,
                    "expected_steps": 32768,
                    "expected_epochs": 8,
                    "expected_optimizer_steps": 32,
                },
                "control": {"provenance_before": control_before, "provenance_after": control_before},
                "graph": {
                    "provenance_before": graph_before,
                    "provenance_after": graph_before,
                    "rollout_policy_graph": graph_runtime_summary(),
                },
            }
        ),
        encoding="utf-8",
    )


def graph_runtime_summary(*, good=True):
    if not good:
        return {"captures": 0, "replays": 0, "preflight": {"status": "failed"}}
    metric = {"max_normalized_error": 0.0}
    report = {
        "deterministic_zero_eps_vs_policy": {"actions": metric, "values": metric, "log_prob": metric},
        "stochastic_action_evaluate_actions": {"values": metric, "log_prob": metric},
        "permutation_equivariance": {"actions": metric, "values": metric, "log_prob": metric},
        "fresh_epsilon_action_diff": 1.0,
        "max_abs_diff": 0.0,
        "base_atol": 1e-5,
        "relative_tolerance_dtype_eps": 4,
    }
    return {
        "captures": 1,
        "replays": 512,
        "preflight_replays": 3,
        "preflight": {
            "status": "passed",
            "stochastic_repeats": 3,
            "setup_rng_restored": True,
            "stochastic_exact": [{"passed": True}, {"passed": True}, {"passed": True}],
            "deterministic_no_rng": {"passed": True},
        },
        "input_spec": {"device": "cuda:0"},
        "eps_spec": {"device": "cuda:0"},
        "helper_direct_sb3": report,
        "graph_direct_sb3": report,
    }


def write_campaign_summary(job, *, completed_cycles=None, graph_runtime=None):
    completed_cycles = job.cycles if completed_cycles is None else completed_cycles
    summary = {
        "status": "completed",
        "completed_measured_cycles": completed_cycles,
        "requested_measured_cycles": job.cycles,
        "warmup_cycles": job.warmup,
        "plan": {"transitions_per_cycle": job.n_envs * job.n_steps},
        "rollout_policy_graph": {
            "enabled": job.arm == "graph",
            "records": [graph_runtime if graph_runtime is not None else graph_runtime_summary()] if job.arm == "graph" else [],
        },
    }
    job.output.mkdir(parents=True, exist_ok=True)
    (job.output / "campaign_summary.json").write_text(json.dumps(summary))


def write_matching_completed_job(module, args, job):
    command = module.command_for_job(args, job)
    env = module.campaign_suite.whitelisted_env(job=job, args=args)
    job.output.mkdir(parents=True, exist_ok=True)
    module.write_job_manifest(args, job, command, env)
    write_campaign_summary(job)


def remove_graph_flags(command):
    result = []
    skip_next = False
    for value in command:
        if skip_next:
            skip_next = False
            continue
        if value == "--rollout-policy-graph":
            continue
        if value == "--rollout-policy-graph-path":
            skip_next = True
            continue
        result.append(value)
    return result


def normalize_output_path(command):
    result = list(command)
    result[result.index("--output") + 1] = "<output>"
    return result


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "size": path.stat().st_size if path.is_file() else None}


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
    def __init__(self, *, write_success=False, stop_after_first=False, completed_cycles=None, bad_graph_runtime=False):
        self.calls = []
        self.write_success = write_success
        self.stop_after_first = stop_after_first
        self.completed_cycles = completed_cycles
        self.bad_graph_runtime = bad_graph_runtime

    def __call__(self, job, command, env, timeout):
        self.calls.append((job, command, env, timeout))
        if self.stop_after_first:
            (job.output.parent / "stop.request").write_text("stop\n")
        if self.write_success:
            graph_runtime = graph_runtime_summary(good=not self.bad_graph_runtime) if job.arm == "graph" else None
            write_campaign_summary(job, completed_cycles=self.completed_cycles, graph_runtime=graph_runtime)
        return 0


class FakeClock:
    def __init__(self, start=0.0):
        self.current = start

    def time(self):
        self.current += 1.0
        return self.current


if __name__ == "__main__":
    unittest.main()
