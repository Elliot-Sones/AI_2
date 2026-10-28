"""Tests for E6 fresh-architecture campaign planning and safety."""

from __future__ import annotations

from ai2.paths import source_path

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class CampaignLearningTests(unittest.TestCase):
    def write_train_artifacts(self, output: Path, *, command_metadata: dict, result_overrides=None, metadata_overrides=None):
        output.mkdir(parents=True, exist_ok=True)
        for name, content in (("initial.zip", "initial\n"), ("latest.zip", "latest\n"), ("final.zip", "final\n")):
            (output / name).write_text(content, encoding="utf-8")
        result = {
            "status": "completed",
            "stop_reason": None,
            "starting_steps": command_metadata.get("expected_starting_steps", 0),
            "ending_steps": command_metadata.get("expected_ending_steps", command_metadata.get("rounded_steps", 0)),
            "actual_steps": command_metadata.get("rounded_steps", 0),
            "wall_seconds": 12.5,
            "initial_checkpoint": str(output / "initial.zip"),
            "latest_checkpoint": str(output / "latest.zip"),
            "final_checkpoint": str(output / "final.zip"),
        }
        if result_overrides:
            result.update(result_overrides)
        metadata = {
            "seed": command_metadata.get("seed", 145),
            "config_sha256": command_metadata.get("config_sha256", "cfg"),
            "parameter_count": command_metadata.get("expected_parameter_count", 1_053_973),
            "checkpoint": command_metadata.get("checkpoint"),
            "input_checkpoint_sha256": None,
            "ppo_controls": {
                "net_arch": command_metadata.get("net_arch", [512, 512, 256]),
                "n_steps": command_metadata.get("controls", {}).get("n_steps", 512),
                "batch_size": command_metadata.get("controls", {}).get("batch_size", 8192),
                "n_epochs": command_metadata.get("controls", {}).get("epochs", 8),
                "ent_coef": command_metadata.get("nav_ent_coef", 0.0),
                "reset_num_timesteps": command_metadata.get("checkpoint") is None,
            },
            "n_envs": command_metadata.get("controls", {}).get("n_envs", 64),
        }
        if metadata_overrides:
            metadata.update(metadata_overrides)
        (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (output / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (output / "runtime_device.json").write_text(json.dumps({"policy_device": "cuda:0", "all_policy_parameters_cuda": True}), encoding="utf-8")

    def base_train_command(self, root: Path, *, output_name="train", checkpoint: Path | None = None, chunk_index=1):
        import ai2.runtime.campaign_learning as e6

        metadata = {
            "seed": 145,
            "config_sha256": "cfg",
            "requested_steps": 500_000,
            "rounded_steps": 524_288,
            "expected_starting_steps": (chunk_index - 1) * 524_288,
            "expected_ending_steps": chunk_index * 524_288,
            "expected_parameter_count": 1_053_973,
            "net_arch": [512, 512, 256],
            "checkpoint": str(checkpoint.resolve()) if checkpoint is not None else None,
            "controls": {"n_envs": 64, "n_steps": 512, "batch_size": 8192, "epochs": 8},
            "nav_ent_coef": 0.0,
        }
        return e6.CommandSpec("train", "train", root, [], root / output_name, 1, metadata)

    def test_campaign_matrix_uses_three_architectures_and_three_seeds(self):
        import ai2.runtime.campaign_learning as e6

        specs = e6.default_run_specs()

        self.assertEqual(len(specs), 9)
        self.assertEqual({tuple(spec.net_arch) for spec in specs}, set(e6.DEFAULT_ARCHITECTURES))
        self.assertEqual({spec.seed for spec in specs}, {145, 146, 147})

    def test_train_commands_start_from_scratch_then_resume_additional_steps(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-plan-") as tmp:
            root = Path(tmp)
            source = root / "source"
            scripts = root / "scripts"
            config = root / "artifacts" / "configs" / "arch-512-512-256_seed145.yaml"
            source.mkdir()
            scripts.mkdir()
            train_job = scripts / "train_job.py"
            train_job.write_text("# train\n", encoding="utf-8")
            spec = e6.RunSpec(net_arch=(512, 512, 256), seed=145)

            first = e6.build_train_command(
                python="/venv/bin/python",
                runner=source_path(scripts, "learning_runner.py"),
                train_job=train_job,
                grouped_env_path=source_path(scripts, "grouped_vec_env.py"),
                source=source,
                config=config,
                spec=spec,
                chunk_index=1,
                previous_checkpoint=None,
                output=root / "artifacts" / "runs" / "one",
                steps=500_000,
                max_seconds=900,
                controls=e6.TrainControls(),
            )
            second = e6.build_train_command(
                python="/venv/bin/python",
                runner=source_path(scripts, "learning_runner.py"),
                train_job=train_job,
                grouped_env_path=source_path(scripts, "grouped_vec_env.py"),
                source=source,
                config=config,
                spec=spec,
                chunk_index=2,
                previous_checkpoint=root / "artifacts" / "runs" / "one" / "final.zip",
                output=root / "artifacts" / "runs" / "two",
                steps=500_000,
                max_seconds=900,
                controls=e6.TrainControls(),
            )

        self.assertNotIn("--checkpoint", first.argv)
        self.assertIn("--checkpoint", second.argv)
        self.assertEqual(second.argv[second.argv.index("--steps") + 1], "500000")
        self.assertEqual(second.metadata["steps_semantics"], "additional_timesteps")
        self.assertEqual(first.cwd, source.resolve())

    def test_config_writer_only_changes_policy_architecture(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-config-") as tmp:
            base = Path(tmp) / "control.yaml"
            out = Path(tmp) / "out.yaml"
            base.write_text(
                "\n".join(
                    [
                        "policy_kwargs:",
                        "  net_arch:",
                        "  - 512",
                        "  - 512",
                        "  - 256",
                        "ppo_settings:",
                        "  learning_rate:",
                        "  - 0.0003",
                        "  - 0.000001",
                        "  clip_range:",
                        "  - 0.2",
                        "  - 0.05",
                        "  nav_ent_coef: 0.05",
                        "  batch_size: 8192",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            e6.write_arch_config(base, out, (1024, 1024, 1024))
            text = out.read_text(encoding="utf-8")

        self.assertIn("  - 1024", text)
        self.assertIn("  batch_size: 8192", text)
        self.assertIn("  learning_rate: 0.0003", text)
        self.assertIn("  clip_range: 0.2", text)
        self.assertIn("  nav_ent_coef: 0.0", text)
        self.assertNotIn("  - 256", text)
        self.assertNotIn("0.000001", text)
        self.assertNotIn("0.05", text)

    def test_config_dict_mutation_records_only_intended_effective_controls(self):
        import ai2.runtime.campaign_learning as e6

        original = {
            "policy_kwargs": {"net_arch": [512, 512, 256]},
            "ppo_settings": {
                "learning_rate": [0.0003, 0.000001],
                "clip_range": [0.2, 0.05],
                "nav_ent_coef": 0.05,
                "batch_size": 8192,
            },
        }

        mutated = e6.mutate_config_dict(original, (1536, 1536, 1536))

        self.assertEqual(mutated["policy_kwargs"]["net_arch"], [1536, 1536, 1536])
        self.assertEqual(mutated["ppo_settings"]["learning_rate"], 0.0003)
        self.assertEqual(mutated["ppo_settings"]["clip_range"], 0.2)
        self.assertEqual(mutated["ppo_settings"]["nav_ent_coef"], 0.0)
        self.assertEqual(mutated["ppo_settings"]["batch_size"], 8192)
        self.assertEqual(original["policy_kwargs"]["net_arch"], [512, 512, 256])

    def test_deadline_admission_refuses_when_timeout_plus_reserve_does_not_fit(self):
        import ai2.runtime.campaign_learning as e6

        with self.assertRaisesRegex(RuntimeError, "insufficient deadline"):
            e6.assert_deadline_allows(now=1_000, deadline=1_500, timeout=300, reserve=300)

    def test_completed_output_is_not_overwritten(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-completed-") as tmp:
            output = Path(tmp) / "job"
            output.mkdir()
            (output / "result.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

            self.assertTrue(e6.is_completed_output(output))
            with self.assertRaisesRegex(RuntimeError, "completed output"):
                e6.assert_not_completed_output(output)

    def test_dev_mastery_requires_near_phase0a_deterministic_and_sampled(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-eval-") as tmp:
            output = Path(tmp)
            for mode, rate in [("deterministic", 0.95), ("sampled", 0.9)]:
                (output / f"0a-{mode}.json").write_text(
                    json.dumps({"summary": {"episodes": 20, "success_rate": rate, "fall_rate": 0.0, "nav_falls": 0, "avg_damage_dealt": 0.0, "avg_damage_taken": 0.0}}),
                    encoding="utf-8",
                )
            (output / "0b-deterministic.json").write_text(
                json.dumps({"summary": {"episodes": 20, "success_rate": 0.0}}),
                encoding="utf-8",
            )

            verdict = e6.evaluate_mastery(output, required_episodes=20, threshold=0.9)

        self.assertTrue(verdict["candidate_mastery"])
        self.assertEqual(verdict["phase"], "0a")

    def test_existing_variant_source_allows_same_generated_config_bytes(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-source-") as tmp:
            root = Path(tmp)
            frozen = root / "frozen"
            variant = root / "variant"
            config = root / "generated.yaml"
            frozen.mkdir()
            variant.mkdir()
            (source_path(frozen, "train.py")).write_text("# frozen\n", encoding="utf-8")
            config.write_text("new: true\n", encoding="utf-8")
            (variant / "config.yaml").write_text("new: true\n", encoding="utf-8")

            e6.prepare_variant_source(frozen, variant, config)

            self.assertEqual((variant / "config.yaml").read_text(encoding="utf-8"), "new: true\n")

    def test_rounded_ppo_steps_are_recorded_and_validated(self):
        import ai2.runtime.campaign_learning as e6

        self.assertEqual(e6.rounded_rollout_steps(500_000, n_envs=128, n_steps=256), 524_288)
        with tempfile.TemporaryDirectory(prefix="e6-artifacts-") as tmp:
            command = self.base_train_command(Path(tmp))
            self.write_train_artifacts(command.output, command_metadata=command.metadata)

            evidence = e6.validate_train_artifacts(command)

        self.assertEqual(evidence["actual_steps"], 524_288)
        self.assertTrue(evidence["final_checkpoint"].endswith("final.zip"))
        self.assertTrue(evidence["latest_checkpoint"].endswith("latest.zip"))
        self.assertTrue(evidence["initial_checkpoint"].endswith("initial.zip"))

    def test_train_artifact_validation_rejects_timeout_or_partial_steps(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-badtrain-") as tmp:
            command = self.base_train_command(Path(tmp))
            self.write_train_artifacts(command.output, command_metadata=command.metadata, result_overrides={"stop_reason": "max_seconds"})

            with self.assertRaisesRegex(RuntimeError, "stop_reason"):
                e6.validate_train_artifacts(command)

    def test_train_validation_rejects_completed_result_with_error_text(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-result-error-") as tmp:
            command = self.base_train_command(Path(tmp))
            self.write_train_artifacts(command.output, command_metadata=command.metadata, result_overrides={"error": "late failure"})

            with self.assertRaisesRegex(RuntimeError, "train error was not clean"):
                e6.validate_train_artifacts(command)

    def test_train_validation_rejects_wrong_resume_checkpoint_path_and_changed_bytes(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-lineage-") as tmp:
            root = Path(tmp)
            expected = root / "chunk01" / "final.zip"
            expected.parent.mkdir()
            expected.write_text("checkpoint-v1\n", encoding="utf-8")
            command = self.base_train_command(root, checkpoint=expected, chunk_index=2)
            self.write_train_artifacts(
                command.output,
                command_metadata=command.metadata,
                metadata_overrides={
                    "checkpoint": str(root / "other.zip"),
                    "input_checkpoint_sha256": e6.sha256_file(expected),
                },
            )

            with self.assertRaisesRegex(RuntimeError, "checkpoint metadata mismatch"):
                e6.validate_train_artifacts(command)

            self.write_train_artifacts(
                command.output,
                command_metadata=command.metadata,
                metadata_overrides={
                    "checkpoint": str(expected.resolve()),
                    "input_checkpoint_sha256": "stale-sha",
                },
            )

            with self.assertRaisesRegex(RuntimeError, "input checkpoint sha mismatch"):
                e6.validate_train_artifacts(command)

    def test_train_validation_rejects_bad_checkpoint_evidence_paths_and_controls(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-controlproof-") as tmp:
            root = Path(tmp)
            command = self.base_train_command(root)
            self.write_train_artifacts(
                command.output,
                command_metadata=command.metadata,
                result_overrides={"latest_checkpoint": str(root / "wrong-latest.zip")},
            )

            with self.assertRaisesRegex(RuntimeError, "latest_checkpoint path mismatch"):
                e6.validate_train_artifacts(command)

            self.write_train_artifacts(
                command.output,
                command_metadata=command.metadata,
                metadata_overrides={
                    "ppo_controls": {
                        "net_arch": [512, 512, 256],
                        "n_steps": 256,
                        "batch_size": 8192,
                        "n_epochs": 8,
                        "ent_coef": 0.0,
                        "reset_num_timesteps": True,
                    },
                    "n_envs": 64,
                },
            )

            with self.assertRaisesRegex(RuntimeError, "n_steps metadata mismatch"):
                e6.validate_train_artifacts(command)

    def test_mastery_requires_exact_counts_finite_success_and_low_falls(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-mastery-") as tmp:
            output = Path(tmp)
            for mode in ("deterministic", "sampled"):
                (output / f"0a-{mode}.json").write_text(
                    json.dumps(
                        {
                            "summary": {
                                "episodes": 20,
                                "success_rate": 0.95,
                                "fall_rate": 0.05,
                                "nav_falls": 1,
                                "avg_damage_dealt": 0.0,
                                "avg_damage_taken": 0.0,
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            good = e6.evaluate_mastery(output, required_episodes=20, threshold=0.9, max_fall_rate=0.05)
            (output / "0a-sampled.json").write_text(json.dumps({"summary": {"episodes": 20, "success_rate": 0.95, "fall_rate": 0.10, "nav_falls": 2, "avg_damage_dealt": 0.0, "avg_damage_taken": 0.0}}), encoding="utf-8")
            bad = e6.evaluate_mastery(output, required_episodes=20, threshold=0.9, max_fall_rate=0.05)

        self.assertTrue(good["candidate_mastery"])
        self.assertFalse(bad["candidate_mastery"])

    def test_eval_artifact_validation_checks_metadata_and_four_outputs(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-evalmeta-") as tmp:
            root = Path(tmp)
            output = root / "eval"
            output.mkdir()
            checkpoint = root / "final.zip"
            checkpoint.write_text("checkpoint-v1\n", encoding="utf-8")
            for phase in ("0a", "0b"):
                for mode in ("deterministic", "sampled"):
                    (output / f"{phase}-{mode}.json").write_text(
                        json.dumps({"summary": {"episodes": 20, "success_rate": 0.9, "fall_rate": 0.0, "nav_falls": 0, "avg_damage_dealt": 0.0, "avg_damage_taken": 0.0}}),
                        encoding="utf-8",
                    )
            (output / "metadata.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": e6.sha256_file(checkpoint),
                        "config_sha256": "cfg",
                        "requested_device": "cuda",
                        "actual_policy_device": "cuda:0",
                        "training_performed": False,
                    }
                ),
                encoding="utf-8",
            )
            command = e6.CommandSpec("eval", "eval", root, [], output, 1, {"episodes": 20, "checkpoint": str(checkpoint), "config_sha256": "cfg"})

            evidence = e6.validate_eval_artifacts(command)

        self.assertEqual(set(evidence["summaries"]), {"0a-deterministic", "0a-sampled", "0b-deterministic", "0b-sampled"})

    def test_eval_validation_rejects_checkpoint_sha_mismatch(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-evalsha-") as tmp:
            root = Path(tmp)
            checkpoint = root / "final.zip"
            checkpoint.write_text("checkpoint-v1\n", encoding="utf-8")
            output = root / "eval"
            output.mkdir()
            for phase in ("0a", "0b"):
                for mode in ("deterministic", "sampled"):
                    (output / f"{phase}-{mode}.json").write_text(
                        json.dumps({"summary": {"episodes": 20, "success_rate": 0.9, "fall_rate": 0.0, "nav_falls": 0, "avg_damage_dealt": 0.0, "avg_damage_taken": 0.0}}),
                        encoding="utf-8",
                    )
            (output / "metadata.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": "stale",
                        "config_sha256": "cfg",
                        "requested_device": "cuda",
                        "actual_policy_device": "cuda:0",
                        "training_performed": False,
                    }
                ),
                encoding="utf-8",
            )
            command = e6.CommandSpec("eval", "eval", root, [], output, 1, {"episodes": 20, "checkpoint": str(checkpoint), "config_sha256": "cfg"})

            with self.assertRaisesRegex(RuntimeError, "eval checkpoint sha mismatch"):
                e6.validate_eval_artifacts(command)

    def test_plan_orders_all_arms_for_chunk_before_next_chunk(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-order-") as tmp:
            root = Path(tmp)
            for name in ("train_job.py", "evaluate_modes.py", "learning_runner.py", "grouped_vec_env.py", "control.yaml"):
                (source_path(root, name)).write_text("# file\n", encoding="utf-8")
            source = root / "source"
            source.mkdir()
            args = type(
                "Args",
                (),
                {
                    "device": "cuda",
                    "output": root / "out",
                    "payload": root,
                    "source": source,
                    "config": root / "control.yaml",
                    "train_job": root / "train_job.py",
                    "evaluate_modes": root / "evaluate_modes.py",
                    "learning_runner": source_path(root, "learning_runner.py"),
                    "grouped_env_path": source_path(root, "grouped_vec_env.py"),
                    "python": "/venv/bin/python",
                    "n_envs": 128,
                    "n_steps": 256,
                    "batch_size": 8192,
                    "epochs": 8,
                    "group_size": 4,
                    "train_max_seconds": 1200,
                    "chunk_steps": 500_000,
                    "chunks": 2,
                    "dev_episodes": 20,
                    "full_episodes": 100,
                    "mastery_threshold": 0.9,
                    "dev_eval_timeout": 420,
                    "full_eval_timeout": 1800,
                    "obs_dim": 256,
                    "action_dim": 10,
                },
            )()

            plan = e6.build_plan(args)
            train_labels = [item["metadata"]["run_label"] for item in plan["commands"] if item["kind"] == "train"]

        self.assertEqual(len(train_labels), 18)
        self.assertEqual(train_labels[:9], train_labels[9:])

    def test_materialize_rejects_existing_config_with_different_bytes(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-materialize-") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "config.yaml").write_text("policy_kwargs:\n  net_arch:\n  - 512\nppo_settings:\n  nav_ent_coef: 0.05\n", encoding="utf-8")
            for name in ("train_job.py", "evaluate_modes.py", "learning_runner.py", "grouped_vec_env.py"):
                (source_path(root, name)).write_text("# file\n", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "device": "cuda",
                    "output": root / "out",
                    "payload": root,
                    "source": source,
                    "config": source / "config.yaml",
                    "train_job": root / "train_job.py",
                    "evaluate_modes": root / "evaluate_modes.py",
                    "learning_runner": source_path(root, "learning_runner.py"),
                    "grouped_env_path": source_path(root, "grouped_vec_env.py"),
                    "python": "/venv/bin/python",
                    "n_envs": 64,
                    "n_steps": 512,
                    "batch_size": 8192,
                    "epochs": 8,
                    "group_size": 1,
                    "train_max_seconds": 1200,
                    "chunk_steps": 500_000,
                    "chunks": 1,
                    "dev_episodes": 20,
                    "full_episodes": 100,
                    "mastery_threshold": 0.9,
                    "dev_eval_timeout": 420,
                    "full_eval_timeout": 1800,
                    "obs_dim": 256,
                    "action_dim": 10,
                },
            )()
            plan = e6.build_plan(args)
            stale = root / "out" / "artifacts" / "configs" / "arch-512-512-256_seed145.yaml"
            stale.parent.mkdir(parents=True)
            stale.write_text("different\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "existing file differs"):
                e6.materialize_plan_inputs(args, plan)

    def test_materialize_preserves_unique_invocation_receipts_on_resume(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-receipts-") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "config.yaml").write_text("policy_kwargs:\n  net_arch:\n  - 512\nppo_settings:\n  nav_ent_coef: 0.05\n", encoding="utf-8")
            for name in ("train_job.py", "evaluate_modes.py", "learning_runner.py", "grouped_vec_env.py"):
                (source_path(root, name)).write_text("# file\n", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "device": "cuda",
                    "output": root / "out",
                    "payload": root,
                    "source": source,
                    "config": source / "config.yaml",
                    "train_job": root / "train_job.py",
                    "evaluate_modes": root / "evaluate_modes.py",
                    "learning_runner": source_path(root, "learning_runner.py"),
                    "grouped_env_path": source_path(root, "grouped_vec_env.py"),
                    "python": "/venv/bin/python",
                    "n_envs": 64,
                    "n_steps": 512,
                    "batch_size": 8192,
                    "epochs": 8,
                    "group_size": 1,
                    "train_max_seconds": 1200,
                    "chunk_steps": 500_000,
                    "chunks": 1,
                    "dev_episodes": 20,
                    "full_episodes": 100,
                    "mastery_threshold": 0.9,
                    "dev_eval_timeout": 420,
                    "full_eval_timeout": 1800,
                    "obs_dim": 256,
                    "action_dim": 10,
                },
            )()
            first = e6.build_plan(args)
            e6.materialize_plan_inputs(args, first)
            second = e6.build_plan(args)
            e6.materialize_plan_inputs(args, second)

            receipts = sorted((root / "out" / "artifacts" / "invocation_receipts").glob("*.json"))
            receipt_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in receipts]

        self.assertEqual(len(receipts), 2)
        self.assertTrue(all(payload.get("plan_sha256") for payload in receipt_payloads))

    def test_main_returns_nonzero_when_execute_result_fails(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-main-") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for name in ("control.yaml", "train_job.py", "evaluate_modes.py", "learning_runner.py", "grouped_vec_env.py"):
                (source_path(root, name)).write_text("# file\n", encoding="utf-8")
            with patch.object(e6, "execute_plan", return_value={"status": "failed", "results": []}), patch("builtins.print"):
                code = e6.main(
                    [
                        "--payload",
                        str(root),
                        "--source",
                        str(source),
                        "--config",
                        str(root / "control.yaml"),
                        "--train-job",
                        str(root / "train_job.py"),
                        "--evaluate-modes",
                        str(root / "evaluate_modes.py"),
                        "--learning-runner",
                        str(source_path(root, "learning_runner.py")),
                        "--grouped-env-path",
                        str(source_path(root, "grouped_vec_env.py")),
                        "--output",
                        str(root / "out"),
                        "--execute",
                    ]
                )

        self.assertEqual(code, 1)

    def test_execute_plan_treats_timeout_as_failed(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-timeout-") as tmp:
            root = Path(tmp)
            command = e6.CommandSpec(
                label="arch-512-512-256_seed145-chunk01-train",
                kind="train",
                cwd=root,
                argv=["/venv/bin/python", "train.py"],
                output=root / "artifacts" / "runs" / "arch-512-512-256_seed145" / "chunk01",
                timeout_seconds=120,
                metadata={"run_label": "arch-512-512-256_seed145"},
            )
            plan = {"output": str(root), "commands": [e6.json_safe(command)]}
            args = type("Args", (), {"deadline": None, "reserve_seconds": 120})()

            with patch.object(
                e6,
                "run_command",
                return_value={"label": command.label, "kind": "train", "status": "timeout"},
            ):
                result = e6.execute_plan(args, plan)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["results"][0]["status"], "timeout")

    def test_execute_plan_records_validation_exception_as_failed(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-invalid-") as tmp:
            root = Path(tmp)
            command = e6.CommandSpec(
                label="arch-512-512-256_seed145-chunk01-train",
                kind="train",
                cwd=root,
                argv=["/venv/bin/python", "train.py"],
                output=root / "artifacts" / "runs" / "arch-512-512-256_seed145" / "chunk01",
                timeout_seconds=120,
                metadata={"run_label": "arch-512-512-256_seed145"},
            )
            plan = {"output": str(root), "commands": [e6.json_safe(command)]}
            args = type("Args", (), {"deadline": None, "reserve_seconds": 120})()

            with patch.object(e6, "run_command", side_effect=RuntimeError("invalid artifact")):
                result = e6.execute_plan(args, plan)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["results"][0]["status"], "failed")
        self.assertIn("invalid artifact", result["results"][0]["error"])

    def test_run_command_refuses_new_launch_when_free_disk_below_floor(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-lowdisk-") as tmp:
            root = Path(tmp)
            command = e6.CommandSpec(
                label="train",
                kind="train",
                cwd=root,
                argv=["/bin/true"],
                output=root / "artifacts" / "runs" / "train",
                timeout_seconds=1,
                metadata={"output_root": str(root)},
            )
            fake_usage = type("Usage", (), {"free": 3 * 1024**3 - 1})()

            with patch.object(e6.shutil, "disk_usage", return_value=fake_usage), patch.object(e6.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "insufficient free disk"):
                    e6.run_command(command, deadline=None, reserve=120)

        popen.assert_not_called()

    def test_run_command_honors_stop_request_before_new_launch(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-stop-") as tmp:
            root = Path(tmp)
            (root / "stop.request").write_text("pause\n", encoding="utf-8")
            command = e6.CommandSpec(
                label="train",
                kind="train",
                cwd=root,
                argv=["/bin/true"],
                output=root / "artifacts" / "runs" / "train",
                timeout_seconds=1,
                metadata={"output_root": str(root)},
            )

            with patch.object(e6.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "stop.request"):
                    e6.run_command(command, deadline=None, reserve=120)

        popen.assert_not_called()

    def test_completed_train_skip_bypasses_disk_and_stop_launch_gates(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-skip-gates-") as tmp:
            root = Path(tmp)
            (root / "stop.request").write_text("pause\n", encoding="utf-8")
            command = self.base_train_command(root)
            command.metadata["output_root"] = str(root)
            self.write_train_artifacts(command.output, command_metadata=command.metadata)

            with patch.object(e6.shutil, "disk_usage", side_effect=AssertionError("disk gate should not run")):
                result = e6.run_command(command, deadline=None, reserve=120)

        self.assertEqual(result["status"], "skipped_completed")

    def test_execute_plan_archives_previous_completed_manifest_before_resume_overwrite(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-history-") as tmp:
            root = Path(tmp)
            manifest = root / "artifacts" / "campaign_learning_results.json"
            manifest.parent.mkdir(parents=True)
            previous_bytes = (
                b'{\n'
                b'  "status": "completed",\n'
                b'  "results": [{"label": "original", "evidence": {"wall_seconds": 12.5}}]\n'
                b'}\n'
            )
            manifest.write_bytes(previous_bytes)
            command = e6.CommandSpec(
                label="arch-512-512-256_seed145_chunk01_train",
                kind="train",
                cwd=root,
                argv=["/venv/bin/python", "train.py"],
                output=root / "artifacts" / "runs" / "arch-512-512-256_seed145" / "chunk01",
                timeout_seconds=120,
                metadata={"run_label": "arch-512-512-256_seed145"},
            )
            plan = {"output": str(root), "commands": [e6.json_safe(command)]}
            args = type("Args", (), {"deadline": None, "reserve_seconds": 120, "dev_episodes": 20, "mastery_threshold": 0.9})()

            with patch.object(
                e6,
                "run_command",
                return_value={"label": command.label, "kind": "train", "status": "timeout"},
            ):
                result = e6.execute_plan(args, plan)

            expected_archive = (
                root
                / "artifacts"
                / "execution_history"
                / f"{hashlib.sha256(previous_bytes).hexdigest()}.json"
            )
            archived_bytes = expected_archive.read_bytes()

        self.assertEqual(result["status"], "failed")
        self.assertEqual(archived_bytes, previous_bytes)

    def test_archiving_existing_execution_manifest_is_idempotent_for_same_bytes(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-history-idempotent-") as tmp:
            root = Path(tmp)
            manifest = root / "artifacts" / "campaign_learning_results.json"
            manifest.parent.mkdir(parents=True)
            previous_bytes = b'{"status":"completed","results":[{"label":"original"}]}\n'
            manifest.write_bytes(previous_bytes)

            first = e6.archive_existing_execution_manifest(manifest)
            second = e6.archive_existing_execution_manifest(manifest)
            first_bytes = first.read_bytes()

        self.assertEqual(first, second)
        self.assertEqual(first_bytes, previous_bytes)

    def test_execute_plan_rejects_malformed_previous_manifest_before_overwrite(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-history-bad-") as tmp:
            root = Path(tmp)
            manifest = root / "artifacts" / "campaign_learning_results.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_bytes(b'{"status": "completed", "results": [}\n')
            command = e6.CommandSpec(
                label="arch-512-512-256_seed145_chunk01_train",
                kind="train",
                cwd=root,
                argv=["/venv/bin/python", "train.py"],
                output=root / "artifacts" / "runs" / "arch-512-512-256_seed145" / "chunk01",
                timeout_seconds=120,
                metadata={"run_label": "arch-512-512-256_seed145"},
            )
            plan = {"output": str(root), "commands": [e6.json_safe(command)]}
            args = type("Args", (), {"deadline": None, "reserve_seconds": 120, "dev_episodes": 20, "mastery_threshold": 0.9})()

            with patch.object(e6, "run_command", side_effect=AssertionError("must fail before launch")):
                with self.assertRaisesRegex(RuntimeError, "malformed existing execution manifest"):
                    e6.execute_plan(args, plan)

        self.assertFalse((root / "artifacts" / "execution_history").exists())

    def test_execute_plan_rejects_invalid_previous_manifest_schema_before_overwrite(self):
        import ai2.runtime.campaign_learning as e6

        with tempfile.TemporaryDirectory(prefix="e6-history-invalid-") as tmp:
            root = Path(tmp)
            manifest = root / "artifacts" / "campaign_learning_results.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("[]\n", encoding="utf-8")
            command = e6.CommandSpec(
                label="arch-512-512-256_seed145_chunk01_train",
                kind="train",
                cwd=root,
                argv=["/venv/bin/python", "train.py"],
                output=root / "artifacts" / "runs" / "arch-512-512-256_seed145" / "chunk01",
                timeout_seconds=120,
                metadata={"run_label": "arch-512-512-256_seed145"},
            )
            plan = {"output": str(root), "commands": [e6.json_safe(command)]}
            args = type("Args", (), {"deadline": None, "reserve_seconds": 120, "dev_episodes": 20, "mastery_threshold": 0.9})()

            with patch.object(e6, "run_command", side_effect=AssertionError("must fail before launch")):
                with self.assertRaisesRegex(RuntimeError, "invalid existing execution manifest"):
                    e6.execute_plan(args, plan)

        self.assertFalse((root / "artifacts" / "execution_history").exists())


if __name__ == "__main__":
    unittest.main()
