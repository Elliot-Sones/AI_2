from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class QualifyNativeReengagementTests(unittest.TestCase):
    def subject(self):
        import qualify_native_reengagement as qualification

        return qualification

    def write_junit(self, path: Path, *, cases: int = 1, skip_message: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        suite = ET.Element("testsuite")
        for index in range(cases):
            case = ET.SubElement(suite, "testcase", name=f"case-{index}")
            if skip_message is not None and index == 0:
                ET.SubElement(case, "skipped", message=skip_message)
        ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)

    def test_mac_receipt_proves_local_cpu_and_leaves_cuda_pending_fail_closed(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qualification"
            checkpoint = root / "B_projected-050m.zip"
            checkpoint.write_bytes(b"parent")
            commands = []

            def runner(command, *, cwd, env, timeout, stdout, stderr, text):
                commands.append((list(command), dict(env)))
                xml_arg = next(part for part in command if part.startswith("--junitxml="))
                self.write_junit(Path(xml_arg.split("=", 1)[1]), cases=2)
                stdout.write("local parity passed\n")
                return SimpleNamespace(returncode=0)

            code = q.main(
                [
                    "--output-dir",
                    str(output),
                    "--reference-root",
                    str(root / "reference"),
                    "--frozen-root",
                    str(root / "frozen"),
                    "--checkpoint",
                    str(checkpoint),
                    "--test-timeout",
                    "7",
                ],
                command_runner=runner,
                system_probe=lambda: q.SystemProbe(cuda_available=False, cuda_device_count=0, gpu_host=False),
            )

            receipt = json.loads((output / "qualification.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(receipt["schema_version"], q.SCHEMA_VERSION)
            self.assertEqual(receipt["qualification_version"], q.QUALIFICATION_VERSION)
            self.assertFalse(receipt["qualified"])
            self.assertEqual(receipt["overall_status"], "pending")
            self.assertEqual(receipt["checkpoint"]["sha256"], q.sha256(checkpoint))
            self.assertIn("command_receipt", receipt)
            self.assertEqual(receipt["stage_status"]["python_oracle_native_cpu"]["status"], "passed")
            self.assertEqual(receipt["stage_status"]["native_cpu_cuda"]["status"], "pending")
            self.assertEqual(receipt["stage_status"]["collector_learner_smoke"]["status"], "pending")
            self.assertFalse(receipt["stage_status"]["native_cpu_cuda"]["required"])
            self.assertFalse(receipt["stage_status"]["collector_learner_smoke"]["required"])
            self.assertEqual(receipt["stage_status"]["source_build_inventory"]["status"], "passed")
            self.assertEqual(receipt["version_contract"]["curriculum_version"], q.CURRICULUM_VERSION)
            self.assertEqual(tuple(receipt["version_contract"]["allowed_control_mask"]), q.CONTROL_MASK)
            self.assertEqual(receipt["scenario_hashes"]["overlap_with_held_out"], [])
            self.assertTrue(receipt["source_hashes"]["qualify_native_reengagement.py"])
            self.assertIn("python", receipt["inventory"])
            self.assertIn("pytorch", receipt["inventory"])
            self.assertEqual(len(commands), 1)
            self.assertIn("tests/test_native_reengagement_episode.py", commands[0][0])
            self.assertIn("tests/test_reengagement_curriculum.py", commands[0][0])
            self.assertNotIn("WANDB_API_KEY", commands[0][1])

    def test_gpu_host_requires_cuda_and_collector_stages_to_pass(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qualification"
            checkpoint = root / "parent.zip"
            checkpoint.write_bytes(b"parent")

            def runner(command, *, cwd, env, timeout, stdout, stderr, text):
                xml_arg = next(part for part in command if part.startswith("--junitxml="))
                stage = Path(xml_arg.split("=", 1)[1]).stem
                if stage == "native_cpu_cuda":
                    self.write_junit(Path(xml_arg.split("=", 1)[1]), cases=1, skip_message="CUDA unavailable")
                else:
                    self.write_junit(Path(xml_arg.split("=", 1)[1]), cases=1)
                stdout.write(stage + "\n")
                return SimpleNamespace(returncode=0)

            code = q.main(
                [
                    "--output-dir",
                    str(output),
                    "--reference-root",
                    str(root / "reference"),
                    "--frozen-root",
                    str(root / "frozen"),
                    "--checkpoint",
                    str(checkpoint),
                ],
                command_runner=runner,
                system_probe=lambda: q.SystemProbe(cuda_available=True, cuda_device_count=1, gpu_host=True),
            )

            receipt = json.loads((output / "qualification.json").read_text())
            self.assertEqual(code, 1)
            self.assertFalse(receipt["qualified"])
            self.assertEqual(receipt["stage_status"]["native_cpu_cuda"]["status"], "failed")
            self.assertIn("skipped required CUDA", receipt["stage_status"]["native_cpu_cuda"]["reason"])

    def test_success_requires_all_stages_version_purity_and_one_finite_update(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qualification"
            checkpoint = root / "parent.zip"
            checkpoint.write_bytes(b"parent")
            stage_names = []

            def runner(command, *, cwd, env, timeout, stdout, stderr, text):
                xml_arg = next(part for part in command if part.startswith("--junitxml="))
                stage = Path(xml_arg.split("=", 1)[1]).stem
                stage_names.append(stage)
                self.write_junit(Path(xml_arg.split("=", 1)[1]), cases=3)
                stdout.write(stage + " passed\n")
                return SimpleNamespace(returncode=0)

            code = q.main(
                [
                    "--output-dir",
                    str(output),
                    "--reference-root",
                    str(root / "reference"),
                    "--frozen-root",
                    str(root / "frozen"),
                    "--checkpoint",
                    str(checkpoint),
                ],
                command_runner=runner,
                collector_smoke=lambda _: q.CollectorSmokeResult(
                    status="passed",
                    reason="one real update",
                    finite_losses=True,
                    optimizer_steps=1,
                    update_counter_before=4,
                    update_counter_after=5,
                    b2_episode_rows=[q.valid_b2_episode_row()],
                    b1_regression={"status": "passed", "checks": ["default_version"]},
                ),
                system_probe=lambda: q.SystemProbe(cuda_available=True, cuda_device_count=1, gpu_host=True),
            )

            receipt = json.loads((output / "qualification.json").read_text())
            self.assertEqual(code, 0)
            self.assertTrue(receipt["qualified"])
            self.assertEqual(receipt["overall_status"], "passed")
            self.assertEqual(stage_names, ["python_oracle_native_cpu", "native_cpu_cuda"])
            self.assertEqual(receipt["stage_status"]["collector_learner_smoke"]["status"], "passed")
            self.assertEqual(receipt["collector_learner_smoke"]["optimizer_steps"], 1)
            self.assertEqual(receipt["collector_learner_smoke"]["update_counter_delta"], 1)

    def test_fail_closed_validators_cover_seed_version_mask_trace_loss_and_optimizer(self):
        q = self.subject()
        with self.assertRaisesRegex(q.QualificationError, "held-out"):
            q.scenario_hashes([134000])
        with self.assertRaisesRegex(q.QualificationError, "curriculum task/mask/version"):
            q.validate_b2_episode_rows([dict(q.valid_b2_episode_row(), curriculum_version="bridge_b1_contact-v1")])
        with self.assertRaisesRegex(q.QualificationError, "curriculum task/mask/version"):
            q.validate_b2_episode_rows([dict(q.valid_b2_episode_row(), allowed_control_mask=(True,) * 10)])
        ambiguous = q.valid_b2_episode_row()
        ambiguous["reengagement"]["diagnostics"]["trace_status"] = "ambiguous"
        with self.assertRaisesRegex(q.QualificationError, "ambiguous"):
            q.validate_b2_episode_rows([ambiguous])
        with self.assertRaisesRegex(q.QualificationError, "finite"):
            q.validate_collector_smoke(
                q.CollectorSmokeResult(
                    status="passed",
                    reason="bad",
                    finite_losses=False,
                    optimizer_steps=1,
                    update_counter_before=0,
                    update_counter_after=1,
                    b2_episode_rows=[q.valid_b2_episode_row()],
                    b1_regression={"status": "passed"},
                )
            )
        with self.assertRaisesRegex(q.QualificationError, "optimizer"):
            q.validate_collector_smoke(
                q.CollectorSmokeResult(
                    status="passed",
                    reason="bad",
                    finite_losses=True,
                    optimizer_steps=0,
                    update_counter_before=0,
                    update_counter_after=0,
                    b2_episode_rows=[q.valid_b2_episode_row()],
                    b1_regression={"status": "passed"},
                )
            )

    def test_refuses_overwrite_before_running_any_stage(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qualification"
            output.mkdir()
            checkpoint = root / "parent.zip"
            checkpoint.write_bytes(b"parent")
            runner_called = False

            def runner(*args, **kwargs):
                nonlocal runner_called
                runner_called = True
                return SimpleNamespace(returncode=0)

            with self.assertRaises(FileExistsError):
                q.run_qualification(
                    q.parse_args(
                        [
                            "--output-dir",
                            str(output),
                            "--reference-root",
                            str(root / "reference"),
                            "--frozen-root",
                            str(root / "frozen"),
                            "--checkpoint",
                            str(checkpoint),
                        ]
                    ),
                    command_runner=runner,
                    system_probe=lambda: q.SystemProbe(cuda_available=False, cuda_device_count=0, gpu_host=False),
                )
            self.assertFalse(runner_called)

    def test_atomic_publish_never_leaves_candidate_only_receipt(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qualification"
            checkpoint = root / "parent.zip"
            checkpoint.write_bytes(b"parent")

            def runner(command, *, cwd, env, timeout, stdout, stderr, text):
                xml_arg = next(part for part in command if part.startswith("--junitxml="))
                self.write_junit(Path(xml_arg.split("=", 1)[1]), cases=1)
                stdout.write("ok\n")
                return SimpleNamespace(returncode=0)

            with patch.object(q.os, "replace", side_effect=OSError("injected atomic failure")):
                with self.assertRaises(OSError):
                    q.run_qualification(
                        q.parse_args(
                            [
                                "--output-dir",
                                str(output),
                                "--reference-root",
                                str(root / "reference"),
                                "--frozen-root",
                                str(root / "frozen"),
                                "--checkpoint",
                                str(checkpoint),
                            ]
                        ),
                        command_runner=runner,
                        system_probe=lambda: q.SystemProbe(cuda_available=False, cuda_device_count=0, gpu_host=False),
                    )
            self.assertFalse((output / "qualification.json").exists())


if __name__ == "__main__":
    unittest.main()
