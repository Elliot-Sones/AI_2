from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch


class QualifyNativeContactTrainingTests(unittest.TestCase):
    def subject(self):
        import tools.validation.qualify_native_contact_training as qualification

        return qualification

    def test_success_runs_all_three_cuda_groups_and_self_verifies_receipt(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "final.zip"
            checkpoint.write_bytes(b"canonical")
            output = root / "qualification"
            commands: list[tuple[list[str], dict[str, str]]] = []
            validated = []
            verified = []

            def runner(command, *, cwd, env, timeout, stdout, stderr, text):
                commands.append((list(command), dict(env)))
                xml_arg = next(part for part in command if part.startswith("--junitxml="))
                xml_path = Path(xml_arg.split("=", 1)[1])
                self.write_junit(xml_path, cases=3)
                stdout.write("pytest passed\n")
                return type("Completed", (), {"returncode": 0})()

            with patch.object(q.base, "validate_checkpoint", return_value={"sha256": "c" * 64, "bytes": 9}) as validate, \
                    patch.object(q.base, "qualification_sources", return_value={"source.py": "a" * 64}), \
                    patch.object(q.base, "require_qualification", side_effect=lambda path, ref, frozen: verified.append((path, ref, frozen)) or {"qualified": True}):
                code = q.main([
                    "--output-dir", str(output),
                    "--reference-root", str(root / "reference"),
                    "--frozen-root", str(root / "frozen"),
                    "--checkpoint", str(checkpoint),
                    "--test-timeout", "9",
                ], command_runner=runner)
                validated.append(validate.call_args.args[0])

            receipt = json.loads((output / "qualification.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(receipt["schema_version"], q.base.QUALIFICATION_SCHEMA)
            self.assertTrue(receipt["qualified"])
            self.assertEqual(receipt["checkpoint_sha256"], q.base.CHECKPOINT_SHA256)
            self.assertEqual(receipt["source_sha256"], {"source.py": "a" * 64})
            self.assertEqual(receipt["checks"], {
                "original_cpu_exact": True,
                "cpu_cuda_exact": True,
                "learner_update_exact": True,
            })
            self.assertEqual({row["check"] for row in receipt["evidence"]}, set(q.base.CHECKS))
            self.assertEqual(validated, [checkpoint.resolve()])
            self.assertEqual(verified, [((output / "qualification.candidate.json").resolve(), (root / "reference").resolve(), (root / "frozen").resolve())])
            self.assertEqual([[part for part in command[0] if part.endswith(".py")] for command in commands], [
                ["tests/test_native_contact_episode.py"],
                ["tests/test_native_contact_lanes.py"],
                ["tests/test_native_combat_update_parity.py", "tests/test_contact_action_update_parity.py"],
            ])
            for command, env in commands:
                self.assertEqual(command[:4], [q.sys.executable, "-m", "pytest", "-q"])
                self.assertIn("--tb=short", command)
                self.assertTrue(any(part.startswith("--junitxml=" + str(output.resolve())) for part in command))
                self.assertEqual(env["WANDB_MODE"], "offline")
                self.assertEqual(env["SDL_VIDEODRIVER"], "dummy")
                self.assertEqual(env["SDL_AUDIODRIVER"], "dummy")
                self.assertEqual(env["AI2_CONTACT_CHECKPOINT"], str(checkpoint.resolve()))
                self.assertEqual(env["AI2_FROZEN_SOURCE_ROOT"], str((root / "frozen").resolve()))
                self.assertNotIn("WANDB_API_KEY", env)
                self.assertNotIn("OPENAI_API_KEY", env)
            inventory = json.loads((output / "inventory.json").read_text())
            self.assertEqual(inventory["schema_version"], "curriculum-inventory-v1")
            inventory_paths = {row["path"] for row in inventory["files"]}
            self.assertIn("qualification.json", inventory_paths)
            self.assertIn("original_cpu_exact.json", inventory_paths)

    def test_rejects_output_reuse_before_checkpoint_validation(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qualification"
            output.mkdir()
            checkpoint = root / "final.zip"
            checkpoint.write_bytes(b"canonical")
            with patch.object(q.base, "validate_checkpoint") as validate:
                with self.assertRaises(FileExistsError):
                    q.run_qualification(q.parse_args([
                        "--output-dir", str(output),
                        "--reference-root", str(root / "reference"),
                        "--frozen-root", str(root / "frozen"),
                        "--checkpoint", str(checkpoint),
                    ]), command_runner=lambda **_: None)
                validate.assert_not_called()

    def test_rejects_zero_cases_and_unapproved_skips(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / "cases.xml"
            log = root / "cases.log"
            self.write_junit(xml, cases=0)
            with self.assertRaisesRegex(q.QualificationError, "No test cases"):
                q.validate_junit(xml, "zero")
            self.write_junit(xml, cases=2, skip_message="CUDA is unavailable")
            with self.assertRaisesRegex(q.QualificationError, "skipped"):
                q.validate_junit(xml, "bad-skip")
            self.write_junit(xml, cases=2, skip_message="two CUDA devices required")
            summary = q.validate_junit(xml, "allowed")
            log.write_text("ok", encoding="utf8")
            self.assertEqual(summary["cases"], 2)
            self.assertEqual(summary["skips"], [{"name": "case-0", "message": "two CUDA devices required"}])

    def test_failed_group_writes_failure_receipt_and_inventory(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "final.zip"
            checkpoint.write_bytes(b"canonical")
            output = root / "qualification"

            def runner(command, *, cwd, env, timeout, stdout, stderr, text):
                xml_arg = next(part for part in command if part.startswith("--junitxml="))
                self.write_junit(Path(xml_arg.split("=", 1)[1]), cases=1)
                stdout.write("failed\n")
                return type("Completed", (), {"returncode": 2})()

            with patch.object(q.base, "validate_checkpoint", return_value={"sha256": "c" * 64}), \
                    patch.object(q.base, "qualification_sources", return_value={"source.py": "a" * 64}), \
                    patch.object(q.base, "require_qualification") as require:
                code = q.main([
                    "--output-dir", str(output),
                    "--reference-root", str(root / "reference"),
                    "--frozen-root", str(root / "frozen"),
                    "--checkpoint", str(checkpoint),
                ], command_runner=runner)

            receipt = json.loads((output / "qualification.json").read_text())
            self.assertEqual(code, 1)
            self.assertFalse(receipt["qualified"])
            self.assertIn("failed with exit code", receipt["error"])
            self.assertTrue((output / "inventory.json").is_file())
            require.assert_not_called()

    def test_self_verification_failure_does_not_publish_qualified_receipt(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "final.zip"
            checkpoint.write_bytes(b"canonical")
            output = root / "qualification"

            def runner(command, *, cwd, env, timeout, stdout, stderr, text):
                xml_arg = next(part for part in command if part.startswith("--junitxml="))
                self.write_junit(Path(xml_arg.split("=", 1)[1]), cases=2)
                stdout.write("passed\n")
                return type("Completed", (), {"returncode": 0})()

            with patch.object(q.base, "validate_checkpoint", return_value={"sha256": "c" * 64}), \
                    patch.object(q.base, "qualification_sources", return_value={"source.py": "a" * 64}), \
                    patch.object(q.base, "require_qualification", side_effect=ValueError("stale source")):
                code = q.main([
                    "--output-dir", str(output),
                    "--reference-root", str(root / "reference"),
                    "--frozen-root", str(root / "frozen"),
                    "--checkpoint", str(checkpoint),
                ], command_runner=runner)

            receipt = json.loads((output / "qualification.json").read_text())
            self.assertEqual(code, 1)
            self.assertFalse(receipt["qualified"])
            self.assertIn("stale source", receipt["error"])
            candidate = json.loads((output / "qualification.candidate.json").read_text())
            self.assertTrue(candidate["qualified"])

    def write_junit(self, path: Path, *, cases: int, skip_message: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        suite = ET.Element("testsuite")
        for index in range(cases):
            case = ET.SubElement(suite, "testcase", name=f"case-{index}")
            if skip_message is not None and index == 0:
                ET.SubElement(case, "skipped", message=skip_message)
        ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    unittest.main()
