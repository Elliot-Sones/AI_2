from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest


class RetentionCudaQualificationTests(unittest.TestCase):
    def subject(self):
        import run_retention_cuda_qualification as qualification

        return qualification

    def checkpoint_args(self, root: Path, q):
        checkpoint = root / "b40.zip"
        checkpoint.write_bytes(b"checkpoint")
        calibration = root / "contact-calibration.zip"
        calibration.write_bytes(b"contact calibration checkpoint")
        q.CONTACT_CALIBRATION_CHECKPOINT_SHA256 = q.sha256(calibration)
        return (
            checkpoint,
            calibration,
            [
                "--checkpoint", str(checkpoint),
                "--checkpoint-sha256", q.sha256(checkpoint),
                "--contact-calibration-checkpoint", str(calibration),
            ],
        )

    def test_rejects_cpu_and_freezes_production_gate_geometry(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source_tree(root)
            config = root / "config.yaml"
            config.write_text("curriculum: {}\n")
            _, _, checkpoint_args = self.checkpoint_args(root, q)
            with self.assertRaisesRegex(SystemExit, "2"):
                q.main([
                    "--source", str(source),
                    "--config", str(config),
                    "--build-dir", str(root / "build"),
                    "--output-dir", str(root / "out"),
                    "--device", "cpu",
                    *checkpoint_args,
                ])
            with self.assertRaisesRegex(SystemExit, "2"):
                q.main([
                    "--source", str(source),
                    "--config", str(config),
                    "--build-dir", str(root / "build"),
                    "--output-dir", str(root / "out2"),
                    "--device", "cuda",
                    *checkpoint_args,
                    "--production-gate",
                    "--lanes", "64",
                ])
            args = q.parse_args([
                "--source", str(source),
                "--config", str(config),
                "--build-dir", str(root / "build"),
                "--output-dir", str(root / "out3"),
                "--device", "cuda",
                *checkpoint_args,
                "--production-gate",
                "--lanes", "4096",
                "--n-steps", "512",
                "--batch-size", "16384",
                "--n-epochs", "4",
            ])
            self.assertEqual(args.lanes, 4096)
            self.assertEqual(args.n_steps, 512)
            self.assertEqual(args.batch_size, 16384)
            self.assertEqual(args.n_epochs, 4)

    def test_success_path_invokes_all_cuda_gates_and_writes_immutable_receipt(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source_tree(root)
            config = root / "config.yaml"
            config.write_text("curriculum: {}\n")
            checkpoint, calibration, checkpoint_args = self.checkpoint_args(root, q)
            checkpoint_sha = q.sha256(checkpoint)
            calibration_sha = q.sha256(calibration)
            output = root / "qualification"
            nav_calls: list[list[str]] = []
            commands: list[list[str]] = []
            smoke_calls = []
            events = []
            nav_module = self.nav_module(nav_calls, events=events)
            retention_module = self.retention_module(smoke_calls, events=events)
            modules = {
                "qualify_navigation_curriculum": nav_module,
                "train_retention_campaign_block": retention_module,
                "native_curriculum_ppo": ModuleType("native_curriculum_ppo"),
            }
            modules["native_curriculum_ppo"].NativeCurriculumPPO = type(
                "NativeCurriculumPPO", (), {"set_task_environment": lambda self, task, env, calibration=False: None}
            )

            code = q.main([
                "--source", str(source),
                "--config", str(config),
                "--build-dir", str(root / "build"),
                "--output-dir", str(output),
                "--device", "cuda",
                *checkpoint_args,
                "--lanes", "8",
                "--n-steps", "4",
                "--batch-size", "16",
                "--n-epochs", "1",
                "--seed", "77",
            ], import_module=lambda name: modules[name], command_runner=self.command_runner(commands),
                dmon_popen=self.dmon_popen(events))

            receipt = json.loads((output / "qualification.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(receipt["status"], "passed")
            self.assertTrue(receipt["offline_wandb"])
            self.assertEqual(os.environ.get("WANDB_MODE"), "offline")
            self.assertEqual(receipt["geometry"], {"lanes": 8, "n_steps": 4, "batch_size": 16, "n_epochs": 1})
            self.assertEqual(receipt["production_gate_args"], q.PRODUCTION_GATE_GEOMETRY)
            self.assertEqual(
                receipt["external_preconditions"]["vast_inventory_before_create"],
                "required_by_orchestrator_before_this_runner",
            )
            self.assertEqual(
                receipt["external_preconditions"]["artifact_download_and_hash_verify"],
                "required_by_orchestrator_after_this_runner",
            )
            self.assertEqual(receipt["checkpoint_expected_sha256"], checkpoint_sha)
            self.assertEqual(receipt["checkpoint_sha256"], q.sha256(checkpoint))
            self.assertTrue(receipt["checkpoint_verified"])
            self.assertEqual(receipt["contact_calibration_checkpoint"], str(calibration.resolve()))
            self.assertEqual(receipt["contact_calibration_checkpoint_expected_sha256"], calibration_sha)
            self.assertEqual(receipt["contact_calibration_checkpoint_sha256"], calibration_sha)
            self.assertTrue(receipt["contact_calibration_checkpoint_verified"])
            self.assertEqual(receipt["gates"]["navigation_phase0b_parity"]["status"], "passed")
            self.assertEqual(receipt["gates"]["navigation_phase0b_parity"]["python_cases"], 2)
            self.assertEqual(receipt["gates"]["navigation_phase0b_parity"]["cuda_cases"], 4)
            self.assertEqual(receipt["gates"]["contact_action_update_parity"]["status"], "passed")
            self.assertEqual(receipt["gates"]["native_combat_update_parity"]["status"], "passed")
            self.assertEqual(receipt["gates"]["alternating_retention_smoke"]["status"], "passed")
            self.assertEqual(receipt["gates"]["alternating_retention_smoke"]["optimizer_steps"], 2)
            self.assertFalse(receipt["gates"]["alternating_retention_smoke"]["kl_early_stop"])
            self.assertEqual(receipt["gates"]["alternating_retention_smoke"]["critic_change_matrix"]["navigation"]["combat"], False)
            self.assertEqual(receipt["gates"]["alternating_retention_smoke"]["critic_change_matrix"]["combat"]["navigation"], False)
            self.assertIn("utilization_samples", receipt)
            self.assertTrue(receipt["utilization_samples"]["available"])
            self.assertGreater(receipt["utilization_samples"]["max_sm_util"], 0)
            raw_utilization = output / "gpu-utilization-dmon.txt"
            self.assertEqual(raw_utilization.read_text(), "# gpu sm mem enc dec\n0 7 10 0 0\n0 11 12 0 0\n")
            self.assertEqual(receipt["utilization_samples"]["raw_artifact"]["path"], "gpu-utilization-dmon.txt")
            self.assertEqual(receipt["utilization_samples"]["raw_artifact"]["bytes"], raw_utilization.stat().st_size)
            self.assertEqual(receipt["utilization_samples"]["raw_artifact"]["sha256"], q.sha256(raw_utilization))
            self.assertEqual(receipt["utilization_samples"]["sample_count"], 2)
            self.assertEqual(receipt["utilization_samples"]["max_sm_util"], 11)
            self.assertEqual(receipt["utilization_samples"]["mean_sm_util"], 9.0)
            self.assertIn("source_hashes", receipt)
            self.assertEqual(receipt["artifact_hashes"]["gpu-utilization-dmon.txt"]["sha256"],
                             q.sha256(raw_utilization))
            self.assertEqual(receipt["artifact_hashes"]["navigation-phase0b-parity.json"]["sha256"],
                             q.sha256(output / "navigation-phase0b-parity.json"))
            self.assertEqual(receipt["artifact_hashes"]["alternating-smoke/alternating-smoke.json"]["sha256"],
                             q.sha256(output / "alternating-smoke" / "alternating-smoke.json"))
            self.assertNotIn("qualification.json", receipt["artifact_hashes"])
            inventory = json.loads((output / "inventory.json").read_text())
            self.assertEqual(inventory["schema_version"], "curriculum-inventory-v1")
            self.assertEqual(inventory["block_id"], output.name)
            self.assertNotIn("schema", inventory)
            inventory_hashes = {row["path"]: row["sha256"] for row in inventory["files"]}
            self.assertEqual(inventory_hashes["qualification.json"], q.sha256(output / "qualification.json"))
            self.assertEqual(inventory_hashes["gpu-utilization-dmon.txt"], q.sha256(raw_utilization))
            self.assertEqual(inventory_hashes["navigation-phase0b-parity.json"],
                             q.sha256(output / "navigation-phase0b-parity.json"))
            self.assertNotIn("inventory.json", inventory_hashes)
            self.assertNotIn("preservation_ack.json", inventory_hashes)
            self.assertEqual(nav_calls[0][nav_calls[0].index("--phase") + 1], "0b")
            self.assertEqual(nav_calls[0][nav_calls[0].index("--device") + 1], "cuda")
            invoked_modules = [
                command[-1]
                for command, _env in commands
                if command[:3] == [sys.executable, "-m", "unittest"]
            ]
            self.assertEqual(invoked_modules, [
                "tests.test_contact_action_update_parity",
                "tests.test_native_combat_update_parity",
            ])
            self.assertEqual(smoke_calls[0]["schedule"], ("navigation", "combat"))
            self.assertEqual(smoke_calls[0]["cfg"]["lanes"], 8)
            self.assertEqual(smoke_calls[0]["cfg"]["n_steps"], 4)
            self.assertEqual(smoke_calls[0]["checkpoint"], checkpoint.resolve())
            command_env = commands[0][1]
            self.assertEqual(command_env["WANDB_MODE"], "offline")
            self.assertEqual(command_env["AI2_CONTACT_CHECKPOINT"], str(calibration.resolve()))
            self.assertLess(events.index("dmon-start"), events.index("nav"))
            self.assertLess(events.index("smoke"), events.index("dmon-terminate"))
            with self.assertRaises(FileExistsError):
                q.run_qualification(q.parse_args([
                    "--source", str(source),
                    "--config", str(config),
                    "--build-dir", str(root / "build2"),
                    "--output-dir", str(output),
                    "--device", "cuda",
                    *checkpoint_args,
                ]), import_module=lambda name: modules[name], command_runner=self.command_runner(commands))

    def test_write_inventory_uses_campaign_schema_excludes_self_ack_and_rejects_unsafe_artifacts(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "qualification-block"
            output.mkdir()
            (output / "qualification.json").write_text("{}\n")
            (output / "manifest.json").write_text("{}\n")
            (output / "preservation_ack.json").write_text("{}\n")
            inventory = q.write_inventory(output, block_id=output.name)
            self.assertEqual(inventory["schema_version"], "curriculum-inventory-v1")
            self.assertEqual(inventory["block_id"], output.name)
            self.assertNotIn("schema", inventory)
            paths = [row["path"] for row in inventory["files"]]
            self.assertEqual(paths, ["manifest.json", "qualification.json"])
            hashes = {row["path"]: row for row in inventory["files"]}
            self.assertEqual(hashes["qualification.json"]["bytes"], (output / "qualification.json").stat().st_size)
            self.assertEqual(hashes["qualification.json"]["sha256"], q.sha256(output / "qualification.json"))
            self.assertNotIn("inventory.json", paths)
            self.assertNotIn("preservation_ack.json", paths)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "qualification-block"
            output.mkdir()
            (output / "qualification.json").write_text("{}\n")
            (output / "unfinished.tmp").write_text("partial")
            with self.assertRaisesRegex(ValueError, "unfinished artifact"):
                q.write_inventory(output, block_id=output.name)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "qualification-block"
            output.mkdir()
            (output / "qualification.json").write_text("{}\n")
            os.symlink(output / "qualification.json", output / "qualification-link.json")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                q.write_inventory(output, block_id=output.name)

    def test_checkpoint_is_required_and_expected_sha_must_match(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source_tree(root)
            config = root / "config.yaml"
            config.write_text("curriculum: {}\n")
            with self.assertRaises(SystemExit):
                q.parse_args([
                    "--source", str(source),
                    "--config", str(config),
                    "--build-dir", str(root / "build"),
                    "--output-dir", str(root / "out"),
                    "--device", "cuda",
                ])
            checkpoint, _, checkpoint_args = self.checkpoint_args(root, q)
            output = root / "out2"
            modules = self.modules_for_success()
            code = q.main([
                "--source", str(source),
                "--config", str(config),
                "--build-dir", str(root / "build"),
                "--output-dir", str(output),
                "--device", "cuda",
                "--checkpoint", str(checkpoint),
                "--checkpoint-sha256", "0" * 64,
                *checkpoint_args[4:],
            ], import_module=lambda name: modules[name], command_runner=self.command_runner([]),
                dmon_popen=self.dmon_popen([]))
            receipt = json.loads((output / "qualification.json").read_text())
            self.assertEqual(code, 1)
            self.assertIn("checkpoint sha256 mismatch", receipt["error"])
            self.assertFalse(receipt.get("checkpoint_verified", False))

    def test_contact_calibration_checkpoint_is_required_and_exact_sha_must_match_before_child_gates(self):
        q = self.subject()
        self.assertEqual(
            q.CANONICAL_CONTACT_CALIBRATION_CHECKPOINT_SHA256,
            "301ce3c72924f3e5c62f265f88292b8952e86e6d71d9a0552213cb8f3c6784b6",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source_tree(root)
            config = root / "config.yaml"
            config.write_text("curriculum: {}\n")
            checkpoint = root / "b40.zip"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaises(SystemExit):
                q.parse_args([
                    "--source", str(source),
                    "--config", str(config),
                    "--build-dir", str(root / "build"),
                    "--output-dir", str(root / "out"),
                    "--device", "cuda",
                    "--checkpoint", str(checkpoint),
                    "--checkpoint-sha256", q.sha256(checkpoint),
                ])

            calibration = root / "contact-calibration.zip"
            calibration.write_bytes(b"not the canonical calibration checkpoint")
            output = root / "out2"
            modules = self.modules_for_success()
            commands: list[list[str]] = []
            code = q.main([
                "--source", str(source),
                "--config", str(config),
                "--build-dir", str(root / "build"),
                "--output-dir", str(output),
                "--device", "cuda",
                "--checkpoint", str(checkpoint),
                "--checkpoint-sha256", q.sha256(checkpoint),
                "--contact-calibration-checkpoint", str(calibration),
            ], import_module=lambda name: modules[name], command_runner=self.command_runner(commands),
                dmon_popen=self.dmon_popen([]))
            receipt = json.loads((output / "qualification.json").read_text())
            self.assertEqual(code, 1)
            self.assertIn("contact calibration checkpoint sha256 mismatch", receipt["error"])
            self.assertFalse(receipt.get("contact_calibration_checkpoint_verified", False))
            self.assertEqual(commands, [])

    def test_unittest_gate_rejects_skips_and_zero_test_runs(self):
        q = self.subject()
        env = {"WANDB_MODE": "offline", "AI2_CONTACT_CHECKPOINT": "/tmp/model.zip"}

        def skipped(command, *, cwd, env, timeout, capture_output, text):
            return type("Completed", (), {
                "returncode": 0,
                "stdout": "",
                "stderr": "Ran 3 tests in 0.1s\n\nOK (skipped=1)\n",
            })()

        with self.assertRaisesRegex(q.QualificationError, "skipped"):
            q.run_unittest_gate("tests.test_contact_action_update_parity", cwd=Path("."), env=env,
                                command_runner=skipped, timeout=1)

        def zero(command, *, cwd, env, timeout, capture_output, text):
            return type("Completed", (), {
                "returncode": 0,
                "stdout": "",
                "stderr": "Ran 0 tests in 0.0s\n\nOK\n",
            })()

        with self.assertRaisesRegex(q.QualificationError, "zero tests"):
            q.run_unittest_gate("tests.test_contact_action_update_parity", cwd=Path("."), env=env,
                                command_runner=zero, timeout=1)

    def test_production_gate_requires_usable_gpu_utilization_evidence(self):
        q = self.subject()
        q.validate_utilization_samples({"available": False, "samples": [], "max_sm_util": None}, production_gate=False)
        with self.assertRaisesRegex(q.QualificationError, "utilization evidence"):
            q.validate_utilization_samples({"available": False, "samples": [], "max_sm_util": None}, production_gate=True)
        with self.assertRaisesRegex(q.QualificationError, "raw dmon artifact"):
            q.validate_utilization_samples({
                "available": True,
                "samples": ["0 7 10"],
                "sample_count": 1,
                "max_sm_util": 7,
                "mean_sm_util": 7.0,
                "raw_artifact": {"path": "gpu-utilization-dmon.txt", "bytes": 0, "sha256": "0" * 64},
            }, production_gate=True)
        with self.assertRaisesRegex(q.QualificationError, "valid dmon samples"):
            q.validate_utilization_samples({
                "available": True,
                "samples": ["bad row"],
                "sample_count": 0,
                "max_sm_util": None,
                "mean_sm_util": None,
                "raw_artifact": {"path": "gpu-utilization-dmon.txt", "bytes": 9, "sha256": "0" * 64},
            }, production_gate=True)
        with self.assertRaisesRegex(q.QualificationError, "non-zero SM"):
            q.validate_utilization_samples({
                "available": True,
                "samples": ["0 0 10"],
                "sample_count": 1,
                "max_sm_util": 0,
                "mean_sm_util": 0.0,
                "raw_artifact": {"path": "gpu-utilization-dmon.txt", "bytes": 8, "sha256": "0" * 64},
            }, production_gate=True)
        q.validate_utilization_samples({
            "available": True,
            "samples": ["0 7 10"],
            "sample_count": 1,
            "max_sm_util": 7,
            "mean_sm_util": 7.0,
            "raw_artifact": {"path": "gpu-utilization-dmon.txt", "bytes": 8, "sha256": "0" * 64},
        }, production_gate=True)

    def test_navigation_receipt_must_cover_exact_cases_ticks_and_no_skips(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source_tree(root)
            config = root / "config.yaml"
            config.write_text("curriculum: {}\n")
            _, _, checkpoint_args = self.checkpoint_args(root, q)
            args = q.parse_args([
                "--source", str(source),
                "--config", str(config),
                "--build-dir", str(root / "build"),
                "--output-dir", str(root / "out"),
                "--device", "cuda",
                *checkpoint_args,
            ])
            bad_module = self.nav_module([], python_cases=1, cuda_cases=4)
            with self.assertRaisesRegex(q.QualificationError, "2 Python"):
                q.run_navigation_phase0b(args, output_dir=root / "out", import_module=lambda name: bad_module)
            skipped_module = self.nav_module([], cuda_status="skipped")
            with self.assertRaisesRegex(q.QualificationError, "skipped"):
                q.run_navigation_phase0b(args, output_dir=root / "out2", import_module=lambda name: skipped_module)

    def test_smoke_result_requires_artifact_and_finite_detailed_timings(self):
        q = self.subject()
        result = self.retention_result()
        result["timings"] = {"navigation": 0.25, "combat": float("nan")}
        with self.assertRaisesRegex(q.QualificationError, "finite"):
            q.validate_smoke_result(result, artifact_path=Path("/definitely/missing.json"))
        result = self.retention_result()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "alternating-smoke.json"
            with self.assertRaisesRegex(q.QualificationError, "artifact"):
                q.validate_smoke_result(result, artifact_path=artifact)

    def test_real_modules_expose_planned_interfaces(self):
        q = self.subject()
        try:
            q.require_planned_interfaces()
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest("torch is not installed in this local interpreter")
            raise

    def test_any_failed_gate_exits_nonzero_and_records_failure(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source_tree(root)
            config = root / "config.yaml"
            config.write_text("curriculum: {}\n")
            output = root / "qualification"
            _, _, checkpoint_args = self.checkpoint_args(root, q)
            nav_module = self.nav_module([])
            retention_module = self.retention_module([], smoke_result={"status": "failed", "error": "bad smoke"})
            modules = {
                "qualify_navigation_curriculum": nav_module,
                "train_retention_campaign_block": retention_module,
                "native_curriculum_ppo": ModuleType("native_curriculum_ppo"),
            }
            modules["native_curriculum_ppo"].NativeCurriculumPPO = type(
                "NativeCurriculumPPO", (), {"set_task_environment": lambda self, task, env, calibration=False: None}
            )
            code = q.main([
                "--source", str(source),
                "--config", str(config),
                "--build-dir", str(root / "build"),
                "--output-dir", str(output),
                "--device", "cuda",
                *checkpoint_args,
            ], import_module=lambda name: modules[name], command_runner=self.command_runner([]))
            receipt = json.loads((output / "qualification.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(receipt["status"], "failed")
            self.assertIn("bad smoke", receipt["error"])
            self.assertFalse(receipt["cuda_qualified"])

    def test_missing_planned_interfaces_fail_closed(self):
        q = self.subject()
        ppo_module = ModuleType("native_curriculum_ppo")
        retention_module = ModuleType("train_retention_campaign_block")
        with self.assertRaisesRegex(q.QualificationError, "set_task_environment"):
            q.require_planned_interfaces(lambda name: {
                "native_curriculum_ppo": ppo_module,
                "train_retention_campaign_block": retention_module,
            }[name])

    def source_tree(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        for name in q_required_source_files():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{name}\n")
        return source

    def modules_for_success(self):
        modules = {
            "qualify_navigation_curriculum": self.nav_module([]),
            "train_retention_campaign_block": self.retention_module([]),
            "native_curriculum_ppo": ModuleType("native_curriculum_ppo"),
        }
        modules["native_curriculum_ppo"].NativeCurriculumPPO = type(
            "NativeCurriculumPPO", (), {"set_task_environment": lambda self, task, env, calibration=False: None}
        )
        return modules

    def nav_module(self, calls, *, events=None, python_cases=2, cuda_cases=4, cuda_status="passed"):
        module = ModuleType("qualify_navigation_curriculum")

        def main(argv):
            calls.append(list(argv))
            if events is not None:
                events.append("nav")
            output = Path(argv[argv.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "schema": "ai2.navigation_curriculum.parity.v1",
                "status": "passed",
                "phase": "0b",
                "device": "cuda",
                "cuda_qualified": True,
                "python_cases": [
                    {"name": f"python-{index}", "status": "passed", "ticks": 240}
                    for index in range(python_cases)
                ],
                "cuda_cases": [
                    {"name": f"cuda-{index}", "status": cuda_status, "ticks": 240}
                    for index in range(cuda_cases)
                ],
            }))
            return 0

        module.main = main
        return module

    def retention_result(self):
        return {
            "status": "passed",
            "schedule": ["navigation", "combat"],
            "actor_changed": {"navigation": True, "combat": True},
            "critic_change_matrix": {
                "navigation": {"navigation": True, "combat": False},
                "combat": {"navigation": False, "combat": True},
            },
            "optimizer_steps": 2,
            "nominal_optimizer_steps": 2,
            "kl_early_stop": False,
            "save_load_exact": True,
            "next_task_continuation": {"navigation": True, "combat": True},
            "timings": {"navigation": 0.1, "combat": 0.2, "save_load": 0.05},
            "rollouts": [
                {"task": "navigation", "collect_seconds": 0.1, "train_seconds": 0.2},
                {"task": "combat", "collect_seconds": 0.1, "train_seconds": 0.2},
            ],
        }

    def retention_module(self, calls, smoke_result=None, events=None):
        module = ModuleType("train_retention_campaign_block")

        def run_qualification_smoke(**kwargs):
            calls.append(kwargs)
            if events is not None:
                events.append("smoke")
            result = smoke_result or self.retention_result()
            result["optimizer_steps"] = kwargs["cfg"]["n_epochs"] * 2
            result["nominal_optimizer_steps"] = kwargs["cfg"]["n_epochs"] * 2
            path = Path(kwargs["output_dir"]) / "alternating-smoke.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
            return result

        module.run_qualification_smoke = run_qualification_smoke
        return module

    def command_runner(self, commands):
        def run(command, *, cwd, env, timeout, capture_output, text):
            commands.append((list(command), dict(env)))
            if command[:2] == ["nvidia-smi", "dmon"]:
                return type("Completed", (), {"returncode": 0, "stdout": "# gpu sm mem enc dec\n0 5 10 0 0\n", "stderr": ""})()
            return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": "Ran 3 tests in 0.1s\n\nOK\n"})()

        return run

    def dmon_popen(self, events):
        class FakeProcess:
            def __init__(self):
                self.returncode = None
                self.stdout_payload = "# gpu sm mem enc dec\n0 7 10 0 0\n0 11 12 0 0\n"
                self.stderr_payload = ""

            def terminate(self):
                events.append("dmon-terminate")
                self.returncode = 0

            def wait(self, timeout=None):
                return self.returncode or 0

            def communicate(self, timeout=None):
                if self.returncode is None:
                    self.returncode = 0
                return self.stdout_payload, self.stderr_payload

            def kill(self):
                events.append("dmon-kill")
                self.returncode = -9

        def start(command, *, cwd, env, stdout, stderr, text):
            events.append("dmon-start")
            return FakeProcess()

        return start


def q_required_source_files():
    return (
        "train.py",
        "control.yaml",
        "positions.json",
        "UTMIST-AI2-main/environment/environment.py",
    )


if __name__ == "__main__":
    unittest.main()
