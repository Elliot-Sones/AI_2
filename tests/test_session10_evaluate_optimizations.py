"""Local metadata/receipt tests; evaluator execution is replaced at its boundary."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class Clock:
    now = 1000.0

    def time(self):
        return self.now


class Session10EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("session10_evaluate_optimizations"))
        self.m = importlib.import_module("session10_evaluate_optimizations")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for name in ("scripts/evaluate_modes.py", "variants/corrected/config.yaml", "variants/corrected/control.yaml",
                     "variants/corrected/train.py", "variants/corrected/learning_eval.py",
                     "variants/corrected/UTMIST-AI2-main/environment/environment.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frozen fixture source")
        digest = hashlib.sha256((self.root / "scripts/evaluate_modes.py").read_bytes()).hexdigest()
        guard = patch.object(self.m, "FROZEN_EVALUATOR_SHA256", digest)
        guard.start()
        self.addCleanup(guard.stop)
        self.checkpoints = {name: self.root / f"{name}.zip" for name in ("initial", "baseline", "selected")}
        for name, path in self.checkpoints.items():
            self.checkpoint(path, steps=16001056 if name == "initial" else 19343392,
                            updates=3904 if name == "initial" else 4720)
        self.clock, self.calls = Clock(), []

    def checkpoint(self, path, *, steps, updates):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data", json.dumps({"num_timesteps": steps, "_n_updates": updates}))

    def args(self, *, run=True, deadline=1780):
        argv = ["--root", str(self.root), "--output", str(self.root / "evaluation"),
                "--deadline-unix", str(deadline)]
        for label, path in self.checkpoints.items():
            argv += ["--checkpoint", f"{label}={path}"]
        return self.m.parse_args([*argv, *(["--run"] if run else [])])

    def evaluation_output(self, job):
        write(job["output"] / "metadata.json", {
            "training_performed": False, "checkpoint": str(job["checkpoint"]),
            "checkpoint_sha256": self.m.sha256(job["checkpoint"]),
            "source": str(self.root / "variants/corrected"),
            "config_sha256": self.m.sha256(self.root / "variants/corrected/config.yaml"),
            "cumulative_steps": job["expected_steps"], "requested_device": "cuda",
            "actual_model_device": "cuda:0", "actual_policy_device": "cuda:0", "device": "cuda:0"})
        summaries = []
        for phase, start in self.m.PANELS.items():
            for mode in ("deterministic", "sampled"):
                summary = {"episodes": 20, "success_rate": .5, "fall_rate": .1,
                           "nav_falls": 2, "navigation_targets_per_episode": 1, "timeout_rate": .4}
                write(job["output"] / f"{phase}-{mode}.json", {
                    "phase_key": phase, "deterministic": mode == "deterministic", "observation_shape": [256],
                    "episodes": [{"seed": seed} for seed in range(start, start + 20)], "summary": summary})
                summaries.append({"phase": phase, "mode": mode, "summary": summary})
        write(job["output"] / "summary.json", summaries)

    def runner(self, job, argv, timeout):
        self.calls.append((job, argv, timeout))
        self.clock.now += 1
        self.evaluation_output(job)
        return 0

    def run_case(self, *, args=None, runner=None, idle_check=None):
        return self.m.run(args or self.args(), runner=runner or self.runner,
                          clock=self.clock, idle_check=idle_check or (lambda: None))

    def report(self):
        return json.loads((self.root / "evaluation/evaluation_report.json").read_text())

    def test_dry_run_hashes_and_plans_without_gpu_or_process(self):
        def forbidden(*_):
            self.fail("dry-run touched GPU or process")
        self.assertEqual(self.run_case(args=self.args(run=False), runner=forbidden, idle_check=forbidden), 0)
        self.assertEqual(self.report()["status"], "dry_run")
        plan = json.loads((self.root / "evaluation/evaluation_plan.json").read_text())
        self.assertEqual([job["expected_steps"] for job in plan["jobs"]], [16001056, 19343392, 19343392])

    def test_three_explicit_cuda_jobs_attest_all_eighty_episodes(self):
        self.assertEqual(self.run_case(), 0)
        self.assertEqual(self.report()["completed_jobs"], 3)
        self.assertEqual(self.report()["expected_episodes_per_checkpoint"], 80)
        for _, argv, timeout in self.calls:
            self.assertEqual(argv[argv.index("--device") + 1], "cuda")
            self.assertEqual(argv[argv.index("--episodes") + 1], "20")
            self.assertEqual(timeout, 90)

    def test_arbitrary_number_of_named_finals_uses_checkpoint_metadata(self):
        for name in ("baseline-seed46", "selected-seed46", "selected-timeout"):
            self.checkpoints[name] = self.root / f"{name}.zip"
            self.checkpoint(self.checkpoints[name], steps=19343392, updates=4720)
        self.assertEqual(self.run_case(), 0)
        self.assertEqual(self.report()["completed_jobs"], 6)

    def test_mismatched_final_steps_or_updates_rejected_before_execution(self):
        self.checkpoint(self.checkpoints["selected"], steps=19343393, updates=4720)
        self.assertEqual(self.run_case(), 2)
        self.assertFalse(self.calls)

    def test_initial_steps_are_exempt_but_cannot_exceed_finals(self):
        self.checkpoint(self.checkpoints["initial"], steps=20000000, updates=4800)
        self.assertEqual(self.run_case(), 2)
        self.assertFalse(self.calls)

    def test_changed_frozen_evaluator_rejected(self):
        (self.root / "scripts/evaluate_modes.py").write_text("changed evaluator")
        self.assertEqual(self.run_case(), 2)
        self.assertFalse(self.calls)

    def test_source_drift_after_process_fails_closed(self):
        def drift(job, argv, timeout):
            self.runner(job, argv, timeout)
            (self.root / "variants/corrected/train.py").write_text("changed")
            return 0
        self.assertNotEqual(self.run_case(runner=drift), 0)
        self.assertEqual(self.report()["completed_jobs"], 0)
        self.assertEqual(len(self.calls), 1)

    def test_cpu_device_hash_training_and_seed_attestations_rejected(self):
        args = self.args()
        job = self.m.plan_jobs(args)[0]
        mutations = [("metadata.json", "actual_policy_device", "cpu"),
                     ("metadata.json", "checkpoint_sha256", "0" * 64),
                     ("metadata.json", "training_performed", True),
                     ("metadata.json", "cumulative_steps", 0)]
        for filename, key, value in mutations:
            self.evaluation_output(job)
            path = job["output"] / filename
            data = json.loads(path.read_text())
            data[key] = value
            write(path, data)
            with self.assertRaises(ValueError):
                self.m.validate_evaluation(args, job, self.m.sha256(job["checkpoint"]),
                                           self.m.sha256(args.source / "config.yaml"))
        self.evaluation_output(job)
        path = job["output"] / "0b-sampled.json"
        shard = json.loads(path.read_text())
        shard["episodes"][0]["seed"] = 1
        write(path, shard)
        with self.assertRaisesRegex(ValueError, "seeds"):
            self.m.validate_evaluation(args, job, self.m.sha256(job["checkpoint"]), self.m.sha256(args.source / "config.yaml"))

    def test_frozen_evaluator_cuda_alias_matches_parameter_device(self):
        args = self.args()
        job = self.m.plan_jobs(args)[0]
        fields = ("actual_model_device", "actual_policy_device", "device")
        # Real failed receipt: SB3 model.device is 'cuda', parameters are 'cuda:0'.
        for devices in (("cuda", "cuda:0", "cuda:0"), ("cuda", "cuda:2", "cuda:2"),
                        ("cuda:2", "cuda", "cuda:2"), ("cuda:0", "cuda:0", "cuda")):
            with self.subTest(devices=devices):
                self.evaluation_output(job)
                path = job["output"] / "metadata.json"
                metadata = json.loads(path.read_text())
                metadata.update(zip(fields, devices))
                write(path, metadata)
                result = self.m.validate_evaluation(
                    args, job, self.m.sha256(job["checkpoint"]), self.m.sha256(args.source / "config.yaml"))
                self.assertEqual(result["verified_episodes"], 80)

    def test_cuda_alias_does_not_hide_mismatched_explicit_ordinals(self):
        args = self.args()
        job = self.m.plan_jobs(args)[0]
        fields = ("actual_model_device", "actual_policy_device", "device")
        for devices in (("cuda:0", "cuda:1", "cuda:1"), ("cuda", "cuda:0", "cuda:1"),
                        ("cuda:0", "cuda", "cuda:1"), ("cuda:0", "cuda:1", "cuda")):
            with self.subTest(devices=devices):
                self.evaluation_output(job)
                path = job["output"] / "metadata.json"
                metadata = json.loads(path.read_text())
                metadata.update(zip(fields, devices))
                write(path, metadata)
                with self.assertRaisesRegex(ValueError, "disagree"):
                    self.m.validate_evaluation(
                        args, job, self.m.sha256(job["checkpoint"]), self.m.sha256(args.source / "config.yaml"))

    def test_cuda_alias_never_allows_cpu_or_invalid_device_attestations(self):
        args = self.args()
        job = self.m.plan_jobs(args)[0]
        for field in ("actual_model_device", "actual_policy_device", "device"):
            for invalid in ("cpu", "mps", "cuda:-1", "cuda:", "cuda:any", None):
                with self.subTest(field=field, invalid=invalid):
                    self.evaluation_output(job)
                    path = job["output"] / "metadata.json"
                    metadata = json.loads(path.read_text())
                    metadata["actual_model_device"] = "cuda"
                    metadata[field] = invalid
                    write(path, metadata)
                    with self.assertRaisesRegex(ValueError, "actual CUDA"):
                        self.m.validate_evaluation(
                            args, job, self.m.sha256(job["checkpoint"]), self.m.sha256(args.source / "config.yaml"))

    def test_busy_gpu_and_insufficient_deadline_never_launch(self):
        def busy():
            raise RuntimeError("GPU busy")
        self.assertEqual(self.run_case(idle_check=busy), 2)
        self.assertFalse(self.calls)

    def test_deadline_and_timeout_fail_closed(self):
        self.assertEqual(self.run_case(args=self.args(deadline=1050)), 3)
        self.assertFalse(self.calls)

    def test_process_timeout_is_not_a_completed_evaluation(self):
        self.assertEqual(self.run_case(runner=lambda *_: 124), 1)
        self.assertEqual(self.report()["completed_jobs"], 0)

    def test_stubborn_process_has_no_unbounded_wait_after_kill(self):
        class StubbornProcess:
            pid = 12345

            def wait(self, timeout=None):
                if timeout is None:
                    raise AssertionError("process termination wait must be bounded")
                raise subprocess.TimeoutExpired("fixture", timeout)

        with patch.object(self.m.subprocess, "Popen", return_value=StubbornProcess()), \
                patch.object(self.m.os, "killpg"):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.m.run_subprocess({"log": self.root / "stubborn.log"}, ["fixture"], 0.01)

    def test_idle_probe_time_is_charged_before_job_launch(self):
        def slow_idle_probe():
            self.clock.now += 15
        self.assertEqual(self.run_case(args=self.args(deadline=1105), idle_check=slow_idle_probe), 3)
        self.assertFalse(self.calls)

    def test_existing_output_never_overwritten(self):
        self.assertEqual(self.run_case(args=self.args(run=False)), 0)
        original = (self.root / "evaluation/evaluation_report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_case()
        self.assertEqual((self.root / "evaluation/evaluation_report.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
