"""Tests for post-hoc E6 initial-checkpoint evaluation planning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class CampaignInitialEvalTests(unittest.TestCase):
    def make_campaign_root(self, root: Path):
        import campaign_learning as learning

        campaign_root = root / "campaign"
        for spec in learning.default_run_specs():
            label = spec.label
            source = campaign_root / "artifacts" / "sources" / label
            config = campaign_root / "artifacts" / "configs" / f"{label}.yaml"
            run = campaign_root / "artifacts" / "runs" / label / "chunk01"
            source.mkdir(parents=True)
            run.mkdir(parents=True)
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(f"label: {label}\n", encoding="utf-8")
            (source / "config.yaml").write_text(f"label: {label}\n", encoding="utf-8")
            (run / "initial.zip").write_text(f"initial {label}\n", encoding="utf-8")
            (run / "result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "starting_steps": 0,
                        "initial_checkpoint": str((run / "initial.zip").resolve()),
                    }
                ),
                encoding="utf-8",
            )
            (run / "metadata.json").write_text(
                json.dumps(
                    {
                        "checkpoint": None,
                        "seed": spec.seed,
                        "ppo_controls": {"reset_num_timesteps": True},
                    }
                ),
                encoding="utf-8",
            )
        return campaign_root

    def write_eval_artifacts(self, output: Path, *, checkpoint: Path, config_sha: str, cumulative_steps: int = 0):
        import campaign_learning as learning

        output.mkdir(parents=True, exist_ok=True)
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": learning.sha256_file(checkpoint),
                    "config_sha256": config_sha,
                    "requested_device": "cuda",
                    "actual_policy_device": "cuda:0",
                    "training_performed": False,
                    "cumulative_steps": cumulative_steps,
                }
            ),
            encoding="utf-8",
        )
        for phase in ("0a", "0b"):
            for mode in ("deterministic", "sampled"):
                (output / f"{phase}-{mode}.json").write_text(
                    json.dumps(
                        {
                            "summary": {
                                "episodes": 20,
                                "success_rate": 0.0,
                                "fall_rate": 0.0,
                                "nav_falls": 0,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
        (output / "summary.json").write_text("[]", encoding="utf-8")

    def test_build_plan_creates_nine_initial_dev20_cuda_eval_commands(self):
        import campaign_initial_eval as initial
        import campaign_learning as learning

        with tempfile.TemporaryDirectory(prefix="e6-initial-plan-") as tmp:
            root = Path(tmp)
            campaign_root = self.make_campaign_root(root)
            (root / "evaluate_modes.py").write_text("# eval\n", encoding="utf-8")

            args = initial.parse_args(
                [
                    "--campaign-root",
                    str(campaign_root),
                    "--output",
                    str(root / "out"),
                    "--evaluate-modes",
                    str(root / "evaluate_modes.py"),
                    "--python",
                    "/venv/bin/python",
                ]
            )
            plan = initial.build_plan(args)

        self.assertEqual(len(plan["commands"]), 9)
        labels = {command["metadata"]["run_label"] for command in plan["commands"]}
        self.assertEqual(labels, {spec.label for spec in learning.default_run_specs()})
        first = plan["commands"][0]
        self.assertEqual(first["kind"], "eval")
        self.assertEqual(first["metadata"]["eval_kind"], "initial_dev20")
        self.assertEqual(first["metadata"]["checkpoint_role"], "chunk01_initial")
        self.assertEqual(first["metadata"]["expected_cumulative_steps"], 0)
        self.assertEqual(first["metadata"]["near_seed"], 84001)
        self.assertEqual(first["metadata"]["wide_seed"], 83001)
        self.assertIn("--device", first["argv"])
        self.assertEqual(first["argv"][first["argv"].index("--device") + 1], "cuda")
        self.assertEqual(first["argv"][first["argv"].index("--episodes") + 1], "20")
        self.assertEqual(first["metadata"]["checkpoint"], first["argv"][first["argv"].index("--checkpoint") + 1])
        self.assertTrue(first["metadata"]["checkpoint_sha256"])
        self.assertTrue(first["metadata"]["config_sha256"])

    def test_build_plan_fails_when_required_initial_checkpoint_missing(self):
        import campaign_initial_eval as initial
        import campaign_learning as learning

        with tempfile.TemporaryDirectory(prefix="e6-missing-initial-") as tmp:
            root = Path(tmp)
            campaign_root = self.make_campaign_root(root)
            missing = (
                campaign_root
                / "artifacts"
                / "runs"
                / learning.default_run_specs()[0].label
                / "chunk01"
                / "initial.zip"
            )
            missing.unlink()
            evaluate_modes = root / "evaluate_modes.py"
            evaluate_modes.write_text("# eval\n", encoding="utf-8")
            args = initial.parse_args(
                ["--campaign-root", str(campaign_root), "--output", str(root / "out"), "--evaluate-modes", str(evaluate_modes)]
            )

            with self.assertRaisesRegex(RuntimeError, "missing required initial checkpoint"):
                initial.build_plan(args)

    def test_preverify_rejects_initial_record_that_started_from_checkpoint(self):
        import campaign_initial_eval as initial
        import campaign_learning as learning

        with tempfile.TemporaryDirectory(prefix="e6-bad-initial-meta-") as tmp:
            root = Path(tmp)
            campaign_root = self.make_campaign_root(root)
            run = campaign_root / "artifacts" / "runs" / learning.default_run_specs()[0].label / "chunk01"
            (run / "metadata.json").write_text(json.dumps({"checkpoint": str(root / "resume.zip")}), encoding="utf-8")
            evaluate_modes = root / "evaluate_modes.py"
            evaluate_modes.write_text("# eval\n", encoding="utf-8")
            args = initial.parse_args(
                ["--campaign-root", str(campaign_root), "--output", str(root / "out"), "--evaluate-modes", str(evaluate_modes)]
            )

            with self.assertRaisesRegex(RuntimeError, "initial training metadata recorded checkpoint"):
                initial.build_plan(args)

    def test_validate_initial_eval_rejects_nonzero_model_cumulative_steps(self):
        import campaign_initial_eval as initial
        import campaign_learning as learning

        with tempfile.TemporaryDirectory(prefix="e6-cumulative-") as tmp:
            root = Path(tmp)
            checkpoint = root / "initial.zip"
            checkpoint.write_text("initial\n", encoding="utf-8")
            output = root / "eval"
            config = root / "config.yaml"
            config.write_text("config\n", encoding="utf-8")
            self.write_eval_artifacts(output, checkpoint=checkpoint, config_sha=learning.sha256_file(config), cumulative_steps=12)
            command = learning.CommandSpec(
                "initial",
                "eval",
                root,
                [],
                output,
                1,
                {
                    "episodes": 20,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": learning.sha256_file(checkpoint),
                    "config_sha256": learning.sha256_file(config),
                    "expected_cumulative_steps": 0,
                },
            )

            with self.assertRaisesRegex(RuntimeError, "model cumulative steps mismatch"):
                initial.validate_initial_eval_artifacts(command)

    def test_execute_plan_reuses_campaign_learning_run_command_and_validates_initial_metadata(self):
        import campaign_initial_eval as initial
        import campaign_learning as learning

        with tempfile.TemporaryDirectory(prefix="e6-execute-initial-") as tmp:
            root = Path(tmp)
            checkpoint = root / "initial.zip"
            checkpoint.write_text("initial\n", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text("config\n", encoding="utf-8")
            output = root / "eval"
            command = learning.CommandSpec(
                "initial",
                "eval",
                root,
                [],
                output,
                1,
                {
                    "episodes": 20,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": learning.sha256_file(checkpoint),
                    "config_sha256": learning.sha256_file(config),
                    "expected_cumulative_steps": 0,
                },
            )
            self.write_eval_artifacts(output, checkpoint=checkpoint, config_sha=learning.sha256_file(config), cumulative_steps=0)
            plan = {"output": str(root), "commands": [learning.json_safe(command)]}
            args = type("Args", (), {"deadline": None, "reserve_seconds": 300.0})()

            with patch.object(initial.learning, "run_command", return_value={"label": "initial", "status": "skipped_existing"}):
                result = initial.execute_plan(args, plan)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["results"][0]["evidence"]["cumulative_steps"], 0)

    def test_main_defaults_to_dry_run_and_writes_plan_provenance_without_execute(self):
        import campaign_initial_eval as initial

        with tempfile.TemporaryDirectory(prefix="e6-initial-main-") as tmp:
            root = Path(tmp)
            campaign_root = self.make_campaign_root(root)
            evaluate_modes = root / "evaluate_modes.py"
            output = root / "out"
            evaluate_modes.write_text("# eval\n", encoding="utf-8")

            with patch.object(initial, "execute_plan", side_effect=AssertionError("dry run should not execute")):
                code = initial.main(
                    [
                        "--campaign-root",
                        str(campaign_root),
                        "--output",
                        str(output),
                        "--evaluate-modes",
                        str(evaluate_modes),
                    ]
                )

            plan = json.loads((output / "artifacts" / "campaign_initial_eval_plan.json").read_text(encoding="utf-8"))
            provenance = json.loads((output / "artifacts" / "campaign_initial_eval_provenance.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(len(plan["commands"]), 9)
        self.assertFalse(plan["training_performed"])
        self.assertEqual(plan["near_seed"], 84001)
        self.assertEqual(plan["wide_seed"], 83001)
        self.assertEqual(len(provenance["initial_checkpoints"]), 9)


if __name__ == "__main__":
    unittest.main()
