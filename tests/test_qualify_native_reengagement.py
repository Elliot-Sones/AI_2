from __future__ import annotations

from ai2.paths import source_path
import ai2.training
import ai2.simulation

import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class QualifyNativeReengagementTests(unittest.TestCase):
    def subject(self):
        import tools.validation.qualify_native_reengagement as qualification

        return qualification

    def write_junit(
        self,
        path: Path,
        *,
        cases: int = 1,
        skip_message: str | None = None,
        case_names: list[str] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        suite = ET.Element("testsuite")
        for index in range(cases):
            case = ET.SubElement(
                suite,
                "testcase",
                name=case_names[index] if case_names is not None else f"case-{index}",
            )
            if skip_message is not None and index == 0:
                ET.SubElement(case, "skipped", message=skip_message)
        ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)

    def write_root_marker(self, root: Path, name: str = "oracle.txt", body: str = "oracle") -> Path:
        path = source_path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf8")
        return path

    def test_mac_receipt_proves_local_cpu_and_leaves_cuda_pending_fail_closed(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qualification"
            reference_marker = self.write_root_marker(root / "reference", body="reference oracle")
            frozen_marker = self.write_root_marker(root / "frozen", body="frozen oracle")
            checkpoint = root / "B_projected-050m.zip"
            checkpoint.write_bytes(b"parent")
            expected_checkpoint = q.sha256(checkpoint)
            commands = []

            def runner(command, *, cwd, env, timeout, stdout, stderr, text):
                commands.append((list(command), dict(env)))
                xml_arg = next(part for part in command if part.startswith("--junitxml="))
                self.write_junit(Path(xml_arg.split("=", 1)[1]), cases=2)
                stdout.write("local parity passed\n")
                return SimpleNamespace(returncode=0)

            with patch.object(q, "EXPECTED_PARENT_SHA256", expected_checkpoint):
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
            self.assertEqual(receipt["stage_status"]["b1_behavior_regression"]["status"], "passed")
            self.assertEqual(receipt["version_contract"]["curriculum_version"], q.CURRICULUM_VERSION)
            self.assertEqual(tuple(receipt["version_contract"]["allowed_control_mask"]), q.CONTROL_MASK)
            self.assertEqual(receipt["scenario_hashes"]["overlap_with_held_out"], [])
            self.assertTrue(receipt["source_hashes"]["qualify_native_reengagement.py"])
            self.assertEqual(receipt["source_sha256"], receipt["source_hashes"])
            self.assertEqual(
                receipt["source_hashes"][f"reference_root::{reference_marker.relative_to(root / 'reference')}"],
                q.sha256(reference_marker),
            )
            self.assertEqual(
                receipt["source_hashes"][f"frozen_root::{frozen_marker.relative_to(root / 'frozen')}"],
                q.sha256(frozen_marker),
            )
            self.assertEqual(receipt["provenance"]["reference_root_sha256"], receipt["root_hashes"]["reference_root"])
            self.assertEqual(receipt["provenance"]["frozen_root_sha256"], receipt["root_hashes"]["frozen_root"])
            self.assertEqual(receipt["checkpoint"]["expected_sha256"], expected_checkpoint)
            self.assertEqual(receipt["provenance"]["reference_root"], str((root / "reference").resolve()))
            self.assertEqual(receipt["provenance"]["frozen_root"], str((root / "frozen").resolve()))
            self.assertIn("python", receipt["inventory"])
            self.assertIn("pytorch", receipt["inventory"])
            self.assertEqual(len(commands), 2)
            self.assertIn("tests/test_native_contact_episode.py", commands[0][0])
            self.assertIn("tests/test_native_reengagement_episode.py", commands[1][0])
            self.assertIn("tests/test_reengagement_curriculum.py", commands[1][0])
            for _command, env in commands:
                self.assertNotIn("WANDB_API_KEY", env)
                self.assertEqual(
                    env["AI2_CONTACT_REFERENCE"],
                    str((root / "reference" / "combat_curriculum.py").resolve()),
                )

    def test_gpu_host_requires_cuda_and_collector_stages_to_pass(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qualification"
            self.write_root_marker(root / "reference")
            self.write_root_marker(root / "frozen")
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

            with patch.object(q, "EXPECTED_PARENT_SHA256", q.sha256(checkpoint)):
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
            self.write_root_marker(root / "reference")
            self.write_root_marker(root / "frozen")
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

            with patch.object(q, "EXPECTED_PARENT_SHA256", q.sha256(checkpoint)):
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
                    collector_smoke=lambda _, **_kwargs: q.CollectorSmokeResult(
                    status="passed",
                    reason="one real update",
                    training_metrics={"train/value_loss": 0.25, "train/entropy_loss": -0.5},
                    attempted_epochs=4,
                    optimizer_steps=1,
                    train_invocations=1,
                    update_counter_before=4,
                    update_counter_after=8,
                    b2_episode_rows=[q.valid_b2_episode_row()],
                    b1_regression={
                        "status": "passed",
                        "log_sha256": "a" * 64,
                        "xml_sha256": "b" * 64,
                        "command": ["pytest", "tests/test_native_contact_episode.py"],
                    },
                    checkpoint_sha256=q.sha256(checkpoint),
                    parent_metadata={
                        "action_ab_arm": "B_projected",
                        "action_head": "bernoulli_10",
                        "policy_action_head": "bernoulli_10",
                        "task_versions": {"combat": q.CURRICULUM_VERSION},
                    },
                    ),
                    system_probe=lambda: q.SystemProbe(cuda_available=True, cuda_device_count=1, gpu_host=True),
                )

            receipt = json.loads((output / "qualification.json").read_text())
            self.assertEqual(code, 0)
            self.assertTrue(receipt["qualified"])
            self.assertEqual(receipt["overall_status"], "passed")
            self.assertEqual(stage_names, ["b1_behavior_regression", "python_oracle_native_cpu", "native_cpu_cuda"])
            self.assertEqual(receipt["stage_status"]["collector_learner_smoke"]["status"], "passed")
            self.assertEqual(receipt["collector_learner_smoke"]["optimizer_steps"], 1)
            self.assertEqual(receipt["collector_learner_smoke"]["train_invocations"], 1)
            self.assertEqual(receipt["collector_learner_smoke"]["attempted_epochs"], 4)
            self.assertEqual(receipt["collector_learner_smoke"]["update_counter_delta"], 4)
            self.assertEqual(receipt["collector_learner_smoke"]["training_metrics"]["train/value_loss"], 0.25)
            self.assertEqual(receipt["checkpoint"]["sha256"], q.sha256(checkpoint))

    def test_fail_closed_validators_cover_seed_version_mask_trace_loss_and_optimizer(self):
        q = self.subject()
        with self.assertRaisesRegex(q.QualificationError, "held-out"):
            q.scenario_hashes([134000])
        with self.assertRaisesRegex(q.QualificationError, "curriculum task/mask/version"):
            q.validate_b2_episode_rows([dict(q.valid_b2_episode_row(), curriculum_version="bridge_b1_contact-v1")])
        with self.assertRaisesRegex(q.QualificationError, "curriculum task/mask/version"):
            q.validate_b2_episode_rows([dict(q.valid_b2_episode_row(), allowed_control_mask=(True,) * 10)])
        with self.assertRaisesRegex(q.QualificationError, "curriculum task/mask/version"):
            q.validate_b2_episode_rows(
                [
                    dict(
                        q.valid_b2_episode_row(),
                        allowed_control_mask=[int(value) for value in q.CONTROL_MASK],
                    )
                ]
            )
        ambiguous = q.valid_b2_episode_row()
        ambiguous["reengagement"]["diagnostics"]["trace_status"] = "ambiguous"
        with self.assertRaisesRegex(q.QualificationError, "ambiguous"):
            q.validate_b2_episode_rows([ambiguous])
        missing_diagnostics = q.valid_b2_episode_row()
        missing_diagnostics["reengagement"] = {}
        with self.assertRaisesRegex(q.QualificationError, "diagnostics"):
            q.validate_b2_episode_rows([missing_diagnostics])
        missing_status = q.valid_b2_episode_row()
        missing_status["reengagement"]["diagnostics"].pop("trace_status")
        with self.assertRaisesRegex(q.QualificationError, "trace_status"):
            q.validate_b2_episode_rows([missing_status])
        missing_field = q.valid_b2_episode_row()
        missing_field["reengagement"]["diagnostics"].pop("achieved_second_hit")
        with self.assertRaisesRegex(q.QualificationError, "diagnostic field"):
            q.validate_b2_episode_rows([missing_field])
        missing_attack = q.valid_b2_episode_row()
        missing_attack["reengagement"]["diagnostics"].pop("second_hit_attack_id")
        with self.assertRaisesRegex(q.QualificationError, "diagnostic field"):
            q.validate_b2_episode_rows([missing_attack])
        failed_episode = q.valid_b2_episode_row()
        failed_diagnostics = failed_episode["reengagement"]["diagnostics"]
        failed_diagnostics.update(
            success=False,
            sequence_completed=False,
            achieved_first_hit=False,
            achieved_relocation=False,
            achieved_pursuit=False,
            achieved_second_hit=False,
            achieved_safe_resolution=False,
            first_hit_attack_id=None,
            second_hit_attack_id=None,
        )
        q.validate_b2_episode_rows([failed_episode])
        with self.assertRaisesRegex(q.QualificationError, "loss"):
            q.validate_collector_smoke(
                q.CollectorSmokeResult(
                    status="passed",
                    reason="bad",
                    training_metrics={},
                    attempted_epochs=1,
                    optimizer_steps=1,
                    train_invocations=1,
                    update_counter_before=0,
                    update_counter_after=1,
                    b2_episode_rows=[q.valid_b2_episode_row()],
                    b1_regression={"status": "passed", "log_sha256": "a" * 64, "xml_sha256": "b" * 64},
                    checkpoint_sha256=q.EXPECTED_PARENT_SHA256,
                    parent_metadata={"action_ab_arm": "B_projected", "action_head": "bernoulli_10",
                                     "policy_action_head": "bernoulli_10",
                                     "task_versions": {"combat": q.CURRICULUM_VERSION}},
                )
            )
        with self.assertRaisesRegex(q.QualificationError, "optimizer"):
            q.validate_collector_smoke(
                q.CollectorSmokeResult(
                    status="passed",
                    reason="bad",
                    training_metrics={"train/value_loss": 0.25, "train/entropy_loss": -0.5},
                    attempted_epochs=1,
                    optimizer_steps=0,
                    train_invocations=1,
                    update_counter_before=0,
                    update_counter_after=0,
                    b2_episode_rows=[q.valid_b2_episode_row()],
                    b1_regression={"status": "passed", "log_sha256": "a" * 64, "xml_sha256": "b" * 64},
                    checkpoint_sha256=q.EXPECTED_PARENT_SHA256,
                    parent_metadata={"action_ab_arm": "B_projected", "action_head": "bernoulli_10",
                                     "policy_action_head": "bernoulli_10",
                                     "task_versions": {"combat": q.CURRICULUM_VERSION}},
                )
            )
        with self.assertRaisesRegex(q.QualificationError, "exactly one train"):
            q.validate_collector_smoke(
                q.CollectorSmokeResult(
                    status="passed",
                    reason="bad",
                    training_metrics={"train/value_loss": 0.25, "train/entropy_loss": -0.5},
                    attempted_epochs=1,
                    optimizer_steps=1,
                    train_invocations=2,
                    update_counter_before=0,
                    update_counter_after=1,
                    b2_episode_rows=[q.valid_b2_episode_row()],
                    b1_regression={"status": "passed", "log_sha256": "a" * 64, "xml_sha256": "b" * 64},
                    checkpoint_sha256=q.EXPECTED_PARENT_SHA256,
                    parent_metadata={"action_ab_arm": "B_projected", "action_head": "bernoulli_10",
                                     "policy_action_head": "bernoulli_10",
                                     "task_versions": {"combat": q.CURRICULUM_VERSION}},
                )
            )
        with self.assertRaisesRegex(q.QualificationError, "update counter"):
            q.validate_collector_smoke(
                q.CollectorSmokeResult(
                    status="passed",
                    reason="bad",
                    training_metrics={"train/value_loss": 0.25, "train/entropy_loss": -0.5},
                    attempted_epochs=4,
                    optimizer_steps=1,
                    train_invocations=1,
                    update_counter_before=0,
                    update_counter_after=1,
                    b2_episode_rows=[q.valid_b2_episode_row()],
                    b1_regression={"status": "passed", "log_sha256": "a" * 64, "xml_sha256": "b" * 64},
                    checkpoint_sha256=q.EXPECTED_PARENT_SHA256,
                    parent_metadata={"action_ab_arm": "B_projected", "action_head": "bernoulli_10",
                                     "policy_action_head": "bernoulli_10",
                                     "task_versions": {"combat": q.CURRICULUM_VERSION}},
                )
            )
        with self.assertRaisesRegex(q.QualificationError, "B1 regression"):
            q.validate_collector_smoke(
                q.CollectorSmokeResult(
                    status="passed",
                    reason="bad",
                    training_metrics={"train/value_loss": 0.25, "train/entropy_loss": -0.5},
                    attempted_epochs=1,
                    optimizer_steps=1,
                    train_invocations=1,
                    update_counter_before=0,
                    update_counter_after=1,
                    b2_episode_rows=[q.valid_b2_episode_row()],
                    b1_regression={"status": "passed"},
                    checkpoint_sha256=q.EXPECTED_PARENT_SHA256,
                    parent_metadata={"action_ab_arm": "B_projected", "action_head": "bernoulli_10",
                                     "policy_action_head": "bernoulli_10",
                                     "task_versions": {"combat": q.CURRICULUM_VERSION}},
                )
            )
        with self.assertRaisesRegex(q.QualificationError, "policy action head"):
            q.validate_collector_smoke(
                q.CollectorSmokeResult(
                    status="passed",
                    reason="bad",
                    training_metrics={"train/value_loss": 0.25, "train/entropy_loss": -0.5},
                    attempted_epochs=1,
                    optimizer_steps=1,
                    train_invocations=1,
                    update_counter_before=0,
                    update_counter_after=1,
                    b2_episode_rows=[q.valid_b2_episode_row()],
                    b1_regression={"status": "passed", "log_sha256": "a" * 64, "xml_sha256": "b" * 64},
                    checkpoint_sha256=q.EXPECTED_PARENT_SHA256,
                    parent_metadata={"action_ab_arm": "B_projected", "action_head": "bernoulli_10",
                                     "policy_action_head": "bernoulli",
                                     "task_versions": {"combat": q.CURRICULUM_VERSION}},
                )
            )

    def test_cuda_parity_allows_only_explicit_multi_device_skip(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / "cases.xml"
            self.write_junit(
                xml,
                cases=2,
                skip_message="two CUDA devices required",
                case_names=["test_native_cuda_two_device_scale", "test_native_cpu_cuda_single_device_parity"],
            )
            summary = q.validate_junit(xml, "native_cpu_cuda", required_cuda=True)
            self.assertEqual(
                summary["optional_skips"],
                [{"name": "test_native_cuda_two_device_scale", "message": "two CUDA devices required"}],
            )
            self.assertEqual(summary["executed"], 1)
            self.write_junit(
                xml,
                cases=1,
                skip_message="two CUDA devices required",
                case_names=["test_native_cuda_two_device_scale"],
            )
            with self.assertRaisesRegex(q.QualificationError, "no required CUDA parity test executed"):
                q.validate_junit(xml, "native_cpu_cuda", required_cuda=True)
            self.write_junit(xml, cases=2, skip_message="CUDA unavailable")
            with self.assertRaisesRegex(q.QualificationError, "skipped required CUDA"):
                q.validate_junit(xml, "native_cpu_cuda", required_cuda=True)
            self.write_junit(xml, cases=1, skip_message="CPU fixture missing")
            with self.assertRaisesRegex(q.QualificationError, "skipped required CPU"):
                q.validate_junit(xml, "python_oracle_native_cpu", required_cuda=False)

    def test_missing_required_source_and_wrong_checkpoint_hash_fail(self):
        q = self.subject()
        with self.assertRaisesRegex(q.QualificationError, "required qualification source missing"):
            q.source_hashes(("definitely_missing_required_source.py",))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_root_marker(root / "reference", body="reference")
            self.write_root_marker(root / "frozen", body="frozen")
            hashes = q.qualification_source_hashes(root / "reference", root / "frozen")
            self.assertEqual(hashes["source_sha256"]["train_reengagement_campaign_block.py"], q.sha256(source_path(q.ROOT, "train_reengagement_campaign_block.py")))
            self.assertEqual(hashes["source_sha256"]["native/navigation_episode.cu"], q.sha256(q.ROOT / "native/navigation_episode.cu"))
            self.assertEqual(hashes["source_sha256"]["reference_root::oracle.txt"], q.sha256(root / "reference" / "oracle.txt"))
            self.assertEqual(hashes["source_sha256"]["frozen_root::oracle.txt"], q.sha256(root / "frozen" / "oracle.txt"))
            with self.assertRaisesRegex(q.QualificationError, "required reference_root missing"):
                q.qualification_source_hashes(root / "missing-reference", root / "frozen")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "parent.zip"
            checkpoint.write_bytes(b"wrong-parent")
            with self.assertRaisesRegex(q.QualificationError, "checkpoint hash"):
                q.checkpoint_receipt(checkpoint)

    def test_real_collector_smoke_loads_parent_checkpoint_and_extracts_logger_metrics(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "parent.zip"
            checkpoint.write_bytes(b"parent")
            output = root / "qualification"
            args = q.parse_args([
                "--output-dir", str(output),
                "--reference-root", str(root / "reference"),
                "--frozen-root", str(root / "frozen"),
                "--checkpoint", str(checkpoint),
            ])
            calls = {}

            class FakeEnv:
                device = "cuda:0"
                num_envs = 16

                def close(self):
                    calls["closed"] = True

            class FakeModel:
                def __init__(self):
                    self._n_updates = 7
                    self.num_timesteps = 60_391_424
                    self.task_versions = {"combat": q.CURRICULUM_VERSION}
                    self.action_ab_arm = "B_projected"
                    self.action_head = "bernoulli_10"
                    self.policy = SimpleNamespace(action_head="bernoulli_10", optimizer=object())
                    self.rollout_buffer = SimpleNamespace(drained_episode_infos=[q.valid_b2_episode_row()])
                    self.last_update_diagnostics = {}
                    self.logger = SimpleNamespace(name_to_value={})
                    self.n_steps = 512
                    self.n_envs = 16
                    self.starting_checkpoint_sha256 = q.EXPECTED_PARENT_SHA256

                def learn(self, total_timesteps):
                    calls["learn_timesteps"] = total_timesteps
                    self.train()
                    self.num_timesteps += total_timesteps

                def train(self):
                    self._n_updates += 1
                    self.last_update_diagnostics = {
                        "attempted_epochs": 1,
                        "optimizer_steps": 3,
                        "kl_early_stop": False,
                    }
                    self.logger.name_to_value = {
                        "train/value_loss": 0.25,
                        "train/entropy_loss": -0.5,
                        "train/policy_gradient_loss": 0.125,
                    }

            class FakePPO:
                CONTACT_VERSION = "bridge_b1_contact-v1"
                COMBAT_MASK = q.CONTROL_MASK

                @staticmethod
                def normalized_task_versions(expected=None):
                    return {"combat": "bridge_b1_contact-v1", **(expected or {})}

                @staticmethod
                def validate_episode_infos(infos, task, task_mask, expected_versions):
                    calls["b1_validate"] = (infos, task, task_mask, expected_versions)

                class NativeCurriculumPPO:
                    @staticmethod
                    def load_contact_checkpoint(path, env, **kwargs):
                        calls["load"] = (path, env, kwargs)
                        return FakeModel()

            fake_env_module = SimpleNamespace(
                build_reengagement_env=lambda *args, **kwargs: calls.setdefault(
                    "env_args", (args, kwargs)
                ) and FakeEnv()
            )
            with patch.object(q, "probe_system", return_value=q.SystemProbe(True, 1, True)), \
                    patch.object(ai2.training, "native_curriculum_ppo", FakePPO, create=True), \
                    patch.object(ai2.simulation, "native_reengagement_lane_vec_env", fake_env_module, create=True), \
                    patch.dict(sys.modules, {
                        "ai2.training.native_curriculum_ppo": FakePPO,
                        "ai2.simulation.native_reengagement_lane_vec_env": fake_env_module,
                    }):
                result = q.run_real_collector_smoke(args, b1_regression={
                    "status": "passed",
                    "log_sha256": "a" * 64,
                    "xml_sha256": "b" * 64,
                    "command": ["pytest", "tests/test_native_contact_episode.py"],
                })

            self.assertEqual(result.status, "passed", result.reason)
            self.assertEqual(calls["load"][0], checkpoint.resolve())
            self.assertEqual(calls["load"][2]["expected_sha256"], q.EXPECTED_PARENT_SHA256)
            self.assertEqual(calls["load"][2]["action_ab_arm"], "B_projected")
            self.assertEqual(calls["load"][2]["action_head"], "bernoulli_10")
            self.assertEqual(calls["load"][2]["expected_task_versions"], {"combat": q.CURRICULUM_VERSION})
            self.assertEqual(calls["env_args"][0], (16,))
            self.assertGreaterEqual(calls["env_args"][1]["episodes_per_lane"], 513)
            self.assertGreaterEqual(calls["env_args"][1]["record_capacity"], 16 * 512)
            self.assertEqual(calls["learn_timesteps"], 16 * 512)
            self.assertTrue(calls["closed"])
            self.assertEqual(result.training_metrics["train/value_loss"], 0.25)
            self.assertEqual(result.train_invocations, 1)
            self.assertEqual(result.attempted_epochs, 1)
            self.assertEqual(result.update_counter_after - result.update_counter_before, 1)

    def test_refuses_overwrite_before_running_any_stage(self):
        q = self.subject()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qualification"
            self.write_root_marker(root / "reference")
            self.write_root_marker(root / "frozen")
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
            self.write_root_marker(root / "reference")
            self.write_root_marker(root / "frozen")
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
