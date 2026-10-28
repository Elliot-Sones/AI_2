import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest import mock


class NativeCurriculumRunnerTests(unittest.TestCase):
    @staticmethod
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write_schedule(self, root, *, gates=None, phase_id="contact_a", block_type="native_contact"):
        schedule = {
            "schema_version": "ai2.native_curriculum_schedule.v1",
            "phases": [
                {
                    "id": phase_id,
                    "block": {"type": block_type, "args": {"rollouts": 1}},
                    "evaluation": {
                        "modes": ["deterministic", "sampled"],
                        "episode_max": 4,
                        "cases": ["fake_case"],
                        "policy_kind": "sb3",
                        "action_head": "auto",
                        "device": "cuda",
                    },
                    "gates": gates
                    if gates is not None
                    else {
                        "deterministic": {"win_rate_min": 0.5, "fall_rate_max": 0.25},
                        "sampled": {"win_rate_min": 0.5, "fall_rate_max": 0.25},
                    },
                }
            ],
        }
        path = Path(root) / "schedule.json"
        path.write_text(json.dumps(schedule), encoding="utf8")
        return path

    def fake_train(self, calls, *, fail=False):
        def train(*, phase, checkpoint_path, output_dir):
            calls.append(("train", phase["id"], str(checkpoint_path), str(output_dir)))
            if fail:
                raise RuntimeError("training exploded")
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=False)
            (output / "final.zip").write_bytes(b"trained checkpoint")
            (output / "result.json").write_text(json.dumps({"status": "trained_not_evaluated"}), encoding="utf8")
            import ai2.training.benchmark_native_contact_training as contact

            contact.write_inventory(output, block_id=output.name)
            return {"checkpoint": output / "final.zip", "result": {"status": "trained_not_evaluated"}}

        return train

    def fake_policy_loader(self, calls):
        def load(checkpoint_path, *, phase):
            calls.append(("load_policy", phase["id"], str(checkpoint_path)))
            return {"checkpoint_path": str(checkpoint_path)}

        return load

    def fake_evaluator(self, calls, *, episodes=None, fail=False, omit_manifest=False):
        def evaluate(*, policy, phase, checkpoint_path, output_dir):
            calls.append(("eval", phase["id"], str(checkpoint_path), str(output_dir), policy["checkpoint_path"]))
            if fail:
                raise RuntimeError("evaluation exploded")
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=False)
            rows = episodes or [
                {"mode": "deterministic", "result": "win", "self_stock_losses": 0, "terminal_cause": "opponent_fall", "hits": 2, "damage_done": 5.0},
                {"mode": "sampled", "result": "win", "self_stock_losses": 0, "terminal_cause": "opponent_fall", "hits": 1, "damage_done": 3.0},
            ]
            rows = [
                {
                    "schema_version": "fullgame-eval-bank-v1",
                    "case_id": "fake_case",
                    "max_ticks": 4,
                    "ticks": 4,
                    "attack_requested": 1,
                    "attack_starts": 1,
                    "opponent_stock_losses": 0,
                    "damage_taken": 0.0,
                    **row,
                }
                for row in rows
            ]
            (output / "episodes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf8",
            )
            summary = {"episodes": len(rows), "wins": sum(1 for row in rows if row["result"] == "win")}
            if not omit_manifest:
                manifest = {
                    "schema_version": "fullgame-eval-bank-v1",
                    "checkpoint": {"path": str(checkpoint_path), "sha256": self.sha256(checkpoint_path)},
                    "modes": list(phase["evaluation"]["modes"]),
                    "cases": [{"case_id": "fake_case", "max_ticks": 4}],
                    "artifact_sha256": {"episodes.jsonl": self.sha256(output / "episodes.jsonl")},
                }
                (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
            return {"episodes": rows, "summary": summary}

        return evaluate

    def test_accepts_native_contact_phase_and_promotes_with_both_modes_passing(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input checkpoint")
            schedule = self.write_schedule(root)
            calls = []

            result = runner.run_schedule(
                schedule_path=schedule,
                input_checkpoint=checkpoint,
                output_root=root / "run",
                train_block=self.fake_train(calls),
                policy_loader=self.fake_policy_loader(calls),
                evaluator=self.fake_evaluator(calls),
            )

            self.assertEqual(result["decision"], "promote")
            self.assertEqual([call[0] for call in calls], ["train", "load_policy", "eval"])
            phase_receipt = json.loads((root / "run" / "phase-000-contact_a" / "phase_receipt.json").read_text())
            self.assertEqual(phase_receipt["decision"], "promote")
            self.assertEqual(phase_receipt["next_checkpoint"]["path"], str(root / "run" / "phase-000-contact_a" / "block" / "final.zip"))
            self.assertIn("deterministic", phase_receipt["metrics_by_mode"])
            self.assertIn("sampled", phase_receipt["metrics_by_mode"])
            self.assertEqual(
                phase_receipt["artifacts"]["block_inventory"]["path"],
                str(root / "run" / "phase-000-contact_a" / "block" / "inventory.json"),
            )
            self.assertEqual(
                phase_receipt["artifacts"]["eval_manifest"]["path"],
                str(root / "run" / "phase-000-contact_a" / "eval" / "manifest.json"),
            )
            self.assertEqual(
                result["phases"][0]["artifacts"]["block_inventory"]["path"],
                phase_receipt["artifacts"]["block_inventory"]["path"],
            )
            self.assertEqual(
                result["phases"][0]["artifacts"]["eval_manifest"]["path"],
                phase_receipt["artifacts"]["eval_manifest"]["path"],
            )
            self.assertEqual(result["input_checkpoint"]["sha256"], runner.sha256_file(checkpoint))
            self.assertEqual(result["schedule"]["sha256"], runner.sha256_file(schedule))

    def test_gate_miss_retains_input_checkpoint_without_promoting_failed_output(self):
        import ai2.training.native_curriculum_runner as runner

        weak_sampled = [
            {"mode": "deterministic", "result": "win", "self_stock_losses": 0, "terminal_cause": "opponent_fall", "hits": 2, "damage_done": 5.0},
            {"mode": "sampled", "result": "loss", "self_stock_losses": 1, "terminal_cause": "self_fall", "hits": 0, "damage_done": 0.0},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input checkpoint")
            result = runner.run_schedule(
                schedule_path=self.write_schedule(root),
                input_checkpoint=checkpoint,
                output_root=root / "run",
                train_block=self.fake_train([]),
                policy_loader=self.fake_policy_loader([]),
                evaluator=self.fake_evaluator([], episodes=weak_sampled),
            )

            self.assertEqual(result["decision"], "retain")
            phase_receipt = json.loads((root / "run" / "phase-000-contact_a" / "phase_receipt.json").read_text())
            self.assertEqual(phase_receipt["decision"], "retain")
            self.assertEqual(phase_receipt["next_checkpoint"]["path"], str(checkpoint))
            self.assertEqual(
                phase_receipt["trained_checkpoint"]["path"],
                str(root / "run" / "phase-000-contact_a" / "block" / "final.zip"),
            )
            self.assertFalse(phase_receipt["gate"]["passed"])
            self.assertIn("sampled.win_rate", phase_receipt["gate"]["failed"])

    def test_completed_retain_run_resumes_idempotently_without_using_rejected_weights(self):
        import ai2.training.native_curriculum_runner as runner

        weak_sampled = [
            {"mode": "deterministic", "result": "win", "self_stock_losses": 0,
             "terminal_cause": "opponent_fall", "hits": 2, "damage_done": 5.0},
            {"mode": "sampled", "result": "loss", "self_stock_losses": 1,
             "terminal_cause": "self_fall", "hits": 0, "damage_done": 0.0},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input checkpoint")
            schedule = self.write_schedule(root)
            output = root / "run"
            calls = []
            first = runner.run_schedule(
                schedule_path=schedule,
                input_checkpoint=checkpoint,
                output_root=output,
                train_block=self.fake_train(calls),
                policy_loader=self.fake_policy_loader(calls),
                evaluator=self.fake_evaluator(calls, episodes=weak_sampled),
            )
            calls.clear()

            resumed = runner.run_schedule(
                schedule_path=schedule,
                input_checkpoint=checkpoint,
                output_root=output,
                train_block=self.fake_train(calls),
                policy_loader=self.fake_policy_loader(calls),
                evaluator=self.fake_evaluator(calls),
                resume=True,
            )

            self.assertEqual(first["decision"], "retain")
            self.assertEqual(resumed, first)
            self.assertEqual(resumed["next_checkpoint"]["path"], str(checkpoint))
            self.assertEqual(calls, [])

    def test_hit_rate_is_episode_success_frequency_not_hits_per_episode(self):
        import ai2.training.native_curriculum_runner as runner

        metrics = runner._summarize_mode([
            {"result": "win", "self_stock_losses": 0, "hits": 3, "damage_done": 9.0},
            {"result": "loss", "self_stock_losses": 1, "hits": 0, "damage_done": 0.0},
        ])

        self.assertEqual(metrics["hit_rate"], 0.5)
        self.assertEqual(metrics["hits_per_episode"], 1.5)

    def test_evaluation_rows_reject_invalid_results_and_fractional_event_counts(self):
        import ai2.training.native_curriculum_runner as runner

        invalid_rows = {
            "result": [
                {"mode": "deterministic", "result": "victory", "self_stock_losses": 0,
                 "terminal_cause": "opponent_fall", "hits": 1, "damage_done": 2.0},
                {"mode": "sampled", "result": "win", "self_stock_losses": 0,
                 "terminal_cause": "opponent_fall", "hits": 1, "damage_done": 2.0},
            ],
            "hits": [
                {"mode": "deterministic", "result": "win", "self_stock_losses": 0,
                 "terminal_cause": "opponent_fall", "hits": 0.5, "damage_done": 2.0},
                {"mode": "sampled", "result": "win", "self_stock_losses": 0,
                 "terminal_cause": "opponent_fall", "hits": 1, "damage_done": 2.0},
            ],
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input checkpoint")
            for label, rows in invalid_rows.items():
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, label):
                        runner.run_schedule(
                            schedule_path=self.write_schedule(root),
                            input_checkpoint=checkpoint,
                            output_root=root / f"invalid-{label}",
                            train_block=self.fake_train([]),
                            policy_loader=self.fake_policy_loader([]),
                            evaluator=self.fake_evaluator([], episodes=rows),
                        )

    def test_all_cases_means_the_complete_default_evaluation_bank(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input checkpoint")
            schedule_path = self.write_schedule(root)
            schedule = json.loads(schedule_path.read_text())
            schedule["phases"][0]["evaluation"]["cases"] = "all"
            schedule_path.write_text(json.dumps(schedule), encoding="utf8")

            with self.assertRaisesRegex(ValueError, "default evaluation case bank"):
                runner.run_schedule(
                    schedule_path=schedule_path,
                    input_checkpoint=checkpoint,
                    output_root=root / "incomplete-all",
                    train_block=self.fake_train([]),
                    policy_loader=self.fake_policy_loader([]),
                    evaluator=self.fake_evaluator([]),
                )

    def test_schedule_fails_closed_without_deterministic_and_sampled_gates(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input checkpoint")
            schedule = self.write_schedule(root, gates={"deterministic": {"win_rate_min": 0.1}})
            with self.assertRaisesRegex(ValueError, "deterministic and sampled"):
                runner.run_schedule(
                    schedule_path=schedule,
                    input_checkpoint=checkpoint,
                    output_root=root / "run",
                    train_block=self.fake_train([]),
                    policy_loader=self.fake_policy_loader([]),
                    evaluator=self.fake_evaluator([]),
                )

    def test_evaluation_missing_sampled_mode_fails_and_cannot_promote(self):
        import ai2.training.native_curriculum_runner as runner

        deterministic_only = [
            {"mode": "deterministic", "result": "win", "self_stock_losses": 0, "terminal_cause": "opponent_fall", "hits": 2, "damage_done": 5.0},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input checkpoint")
            with self.assertRaisesRegex(ValueError, "episode coverage"):
                runner.run_schedule(
                    schedule_path=self.write_schedule(root),
                    input_checkpoint=checkpoint,
                    output_root=root / "run",
                    train_block=self.fake_train([]),
                    policy_loader=self.fake_policy_loader([]),
                    evaluator=self.fake_evaluator([], episodes=deterministic_only),
                )
            phase_receipt = json.loads((root / "run" / "phase-000-contact_a" / "phase_receipt.json").read_text())
            self.assertEqual(phase_receipt["decision"], "fail")
            self.assertEqual(phase_receipt["next_checkpoint"]["path"], str(checkpoint))

    def test_training_or_evaluation_failure_preserves_failure_receipt(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input checkpoint")
            with self.assertRaisesRegex(RuntimeError, "training exploded"):
                runner.run_schedule(
                    schedule_path=self.write_schedule(root),
                    input_checkpoint=checkpoint,
                    output_root=root / "train_fail",
                    train_block=self.fake_train([], fail=True),
                    policy_loader=self.fake_policy_loader([]),
                    evaluator=self.fake_evaluator([]),
                )
            receipt = json.loads((root / "train_fail" / "phase-000-contact_a" / "phase_receipt.json").read_text())
            self.assertEqual(receipt["decision"], "fail")
            self.assertEqual(receipt["failure"]["stage"], "train")

            with self.assertRaisesRegex(RuntimeError, "evaluation exploded"):
                runner.run_schedule(
                    schedule_path=self.write_schedule(root),
                    input_checkpoint=checkpoint,
                    output_root=root / "eval_fail",
                    train_block=self.fake_train([]),
                    policy_loader=self.fake_policy_loader([]),
                    evaluator=self.fake_evaluator([], fail=True),
                )
            receipt = json.loads((root / "eval_fail" / "phase-000-contact_a" / "phase_receipt.json").read_text())
            self.assertEqual(receipt["decision"], "fail")
            self.assertEqual(receipt["failure"]["stage"], "eval")

    def test_resume_skips_accepted_phase_and_never_uses_rejected_output(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input checkpoint")
            calls = []
            first = runner.run_schedule(
                schedule_path=self.write_schedule(root),
                input_checkpoint=checkpoint,
                output_root=root / "run",
                train_block=self.fake_train(calls),
                policy_loader=self.fake_policy_loader(calls),
                evaluator=self.fake_evaluator(calls),
            )
            calls.clear()
            resumed = runner.run_schedule(
                schedule_path=self.write_schedule(root),
                input_checkpoint=checkpoint,
                output_root=root / "run",
                train_block=self.fake_train(calls),
                policy_loader=self.fake_policy_loader(calls),
                evaluator=self.fake_evaluator(calls),
                resume=True,
            )
            self.assertEqual(resumed["decision"], "promote")
            self.assertEqual(resumed["next_checkpoint"], first["next_checkpoint"])
            self.assertEqual(calls, [])

            rejected = root / "rejected" / "phase-000-contact_a"
            rejected.mkdir(parents=True)
            (rejected / "phase_receipt.json").write_text(
                json.dumps({"decision": "retain", "next_checkpoint": {"path": str(rejected / "block" / "final.zip")}}),
                encoding="utf8",
            )
            with self.assertRaisesRegex(ValueError, "resume phase receipt schema"):
                runner.run_schedule(
                    schedule_path=self.write_schedule(root),
                    input_checkpoint=checkpoint,
                    output_root=root / "rejected",
                    train_block=self.fake_train([]),
                    policy_loader=self.fake_policy_loader([]),
                    evaluator=self.fake_evaluator([]),
                    resume=True,
                )

    def test_default_contact_adapter_uses_parser_defaults_and_forces_owned_paths(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected_checkpoint = root / "input.zip"
            expected_checkpoint.write_bytes(b"input")
            output = root / "block"
            foreign = root / "foreign.zip"
            foreign.write_bytes(b"foreign")
            observed = {}

            def parse_args(argv):
                parser = argparse.ArgumentParser()
                parser.add_argument("--checkpoint", type=Path, required=True)
                parser.add_argument("--output", type=Path, required=True)
                parser.add_argument("--deadline-unix", type=float, required=True)
                parser.add_argument("--batch-size", type=int, default=8192)
                parser.add_argument("--rollouts", type=int, default=4)
                parser.add_argument("--seed", type=int, default=145)
                parser.add_argument("--device", default="cuda")
                parser.add_argument("--update-mode", default="deferred_metrics")
                parser.add_argument("--action-ab-arm", default="G_keep")
                return parser.parse_args(argv)

            def run_block(args):
                observed.update(vars(args))
                output.mkdir()
                (output / "final.zip").write_bytes(b"trained")
                return {"status": "trained_not_evaluated"}

            fake_contact = types.SimpleNamespace(parse_args=parse_args, run_block=run_block)
            phase = {"block": {"type": "native_contact", "args": {
                "deadline_unix": 9999999999.0,
                "rollouts": 2,
                "checkpoint": str(foreign),
                "output": str(root / "foreign-output"),
            }}}
            with mock.patch.dict(sys.modules, {"ai2.training.benchmark_native_contact_training": fake_contact}):
                result = runner._default_train_block(
                    phase=phase, checkpoint_path=expected_checkpoint, output_dir=output)

            self.assertEqual(observed["checkpoint"], expected_checkpoint)
            self.assertEqual(observed["output"], output)
            self.assertEqual(observed["batch_size"], 8192)
            self.assertEqual(observed["seed"], 145)
            self.assertEqual(observed["device"], "cuda")
            self.assertEqual(observed["update_mode"], "deferred_metrics")
            self.assertEqual(result["checkpoint"], output / "final.zip")

    def test_default_train_adapter_dispatches_contact_and_reengagement_modules(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input")
            observed = []

            def module(label):
                def parse_args(argv):
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--checkpoint", type=Path, required=True)
                    parser.add_argument("--output", type=Path, required=True)
                    parser.add_argument("--deadline-unix", type=float, required=True)
                    parser.add_argument("--checkpoint-sha256")
                    return parser.parse_args(argv)

                def run_block(args):
                    observed.append((label, args.checkpoint, args.output))
                    args.output.mkdir()
                    (args.output / "final.zip").write_bytes(label.encode("utf8"))
                    return {"status": "trained_not_evaluated"}

                return types.SimpleNamespace(
                    parse_args=parse_args,
                    run_block=run_block,
                    PARENT_CHECKPOINT_SHA256="a" * 64,
                )

            fake_modules = {
                "ai2.training.benchmark_native_contact_training": module("contact"),
                "ai2.training.train_reengagement_campaign_block": module("reengagement"),
            }
            with mock.patch.dict(sys.modules, fake_modules):
                for block_type, label in (
                    ("native_contact", "contact"),
                    ("native_reengagement", "reengagement"),
                ):
                    output = root / label
                    result = runner._default_train_block(
                        phase={"block": {"type": block_type, "args": {"deadline_unix": 9999999999.0}}},
                        checkpoint_path=checkpoint,
                        output_dir=output,
                    )
                    self.assertEqual(result["checkpoint"], output / "final.zip")

            self.assertEqual(
                observed,
                [
                    ("contact", checkpoint, root / "contact"),
                    ("reengagement", checkpoint, root / "reengagement"),
                ],
            )

    def test_default_reengagement_adapter_synthesizes_fresh_and_resume_cycle_arguments(self):
        import ai2.training.native_curriculum_runner as runner
        import ai2.training.train_reengagement_campaign_block as reengagement

        with TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.zip"
            parent.write_bytes(b"canonical parent stand-in")
            checkpoint_ack = root / "ack.json"
            checkpoint_ack.write_text("{}", encoding="utf8")
            common_args = {
                "checkpoint_ack": checkpoint_ack,
                "frozen_root": root / "frozen",
                "reference_root": root / "reference",
                "qualification": root / "qualification.json",
                "deadline_unix": 9999999999.0,
                "schedule": "R4E1N",
                "require_wandb": True,
                "checkpoint_sha256": "0" * 64,
                "resume_parent_result": root / "stale-result.json",
            }
            observed = []

            def fake_run_block(args):
                observed.append(vars(args).copy())
                args.output.mkdir()
                (args.output / "final.zip").write_bytes(f"cycle-{args.cycle_index}".encode("utf8"))
                (args.output / "result.json").write_text(json.dumps({
                    "schema_version": reengagement.SCHEMA_VERSION,
                    "cycle_index": args.cycle_index,
                    "checkpoint_sha256": runner.sha256_file(args.output / "final.zip"),
                    "schedule_id": "R4E1N",
                    "schedule": list(reengagement.SCHEDULES["R4E1N"]),
                    "expected_task_versions": reengagement.EXPECTED_TASK_VERSIONS,
                    "root_parent_checkpoint_sha256": reengagement.PARENT_CHECKPOINT_SHA256,
                    "final_model_counters": {"num_timesteps": 60_391_424 + args.cycle_index * 10_485_760},
                }), encoding="utf8")
                return {"status": "trained_not_evaluated"}

            phase_1 = {"block": {"type": "native_reengagement", "args": dict(common_args)}}
            cycle_1_dir = root / "cycle-001"
            with mock.patch.object(reengagement, "run_block", side_effect=fake_run_block):
                first = runner._default_train_block(
                    phase=phase_1,
                    checkpoint_path=parent,
                    output_dir=cycle_1_dir,
                )

                phase_2 = {"block": {"type": "native_reengagement", "args": {
                    **common_args,
                    "cycle_index": 2,
                }}}
                cycle_2_dir = root / "cycle-002"
                second = runner._default_train_block(
                    phase=phase_2,
                    checkpoint_path=first["checkpoint"],
                    output_dir=cycle_2_dir,
                )

            self.assertEqual(first["checkpoint"], cycle_1_dir / "final.zip")
            self.assertEqual(second["checkpoint"], cycle_2_dir / "final.zip")
            self.assertEqual(observed[0]["checkpoint"], parent)
            self.assertEqual(observed[0]["output"], cycle_1_dir)
            self.assertFalse(observed[0]["resume"])
            self.assertTrue(observed[0]["require_wandb"])
            self.assertEqual(observed[0]["cycle_index"], 1)
            self.assertEqual(observed[0]["checkpoint_sha256"], reengagement.PARENT_CHECKPOINT_SHA256)
            self.assertIsNone(observed[0]["resume_parent_result"])
            self.assertEqual(observed[1]["checkpoint"], cycle_1_dir / "final.zip")
            self.assertEqual(observed[1]["output"], cycle_2_dir)
            self.assertTrue(observed[1]["resume"])
            self.assertTrue(observed[1]["require_wandb"])
            self.assertEqual(observed[1]["cycle_index"], 2)
            self.assertEqual(observed[1]["checkpoint_sha256"], runner.sha256_file(cycle_1_dir / "final.zip"))
            self.assertEqual(observed[1]["resume_parent_result"], cycle_1_dir / "result.json")
            self.assertEqual(observed[1]["checkpoint_ack"], cycle_1_dir / "preservation_ack.json")

            phase_bad_wandb = {"block": {"type": "native_reengagement", "args": {
                **common_args,
                "require_wandb": False,
            }}}
            with self.assertRaisesRegex(ValueError, "require_wandb"):
                runner._default_train_block(
                    phase=phase_bad_wandb,
                    checkpoint_path=parent,
                    output_dir=root / "bad-wandb",
                )

    def test_both_block_types_reject_phase_owned_checkpoint_and_output(self):
        import ai2.training.native_curriculum_runner as runner

        for block_type in ("native_contact", "native_reengagement"):
            for owned_name in ("checkpoint", "output"):
                with self.subTest(block_type=block_type, owned_name=owned_name):
                    raw = {
                        "id": "phase",
                        "block": {"type": block_type, "args": {owned_name: "foreign"}},
                        "evaluation": {
                            "modes": ["deterministic", "sampled"],
                            "policy_kind": "sb3",
                            "action_head": "auto",
                            "device": "cuda",
                        },
                        "gates": {
                            "deterministic": {"win_rate_min": 0.5},
                            "sampled": {"win_rate_min": 0.5},
                        },
                    }
                    with self.assertRaisesRegex(ValueError, "cannot override conductor checkpoint or output"):
                        runner._validate_phase(raw)

    def test_default_contact_adapter_matches_the_real_parser_contract(self):
        import ai2.training.benchmark_native_contact_training as contact
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input")
            output = root / "block"
            observed = {}

            def fake_run_block(args):
                observed.update(vars(args))
                output.mkdir()
                (output / "final.zip").write_bytes(b"trained")
                return {"status": "trained_not_evaluated"}

            phase = {"block": {"type": "native_contact", "args": {
                "deadline_unix": 9999999999.0,
                "qualification": root / "qualification.json",
                "reference_root": root / "reference",
                "frozen_root": root / "frozen",
                "rollouts": 2,
            }}}
            with mock.patch.object(contact, "run_block", side_effect=fake_run_block):
                result = runner._default_train_block(
                    phase=phase, checkpoint_path=checkpoint, output_dir=output)

            self.assertEqual(observed["checkpoint"], checkpoint)
            self.assertEqual(observed["output"], output)
            self.assertEqual(observed["batch_size"], 8192)
            self.assertEqual(observed["seed"], 145)
            self.assertEqual(observed["device"], "cuda")
            self.assertEqual(observed["action_ab_arm"], "G_keep")
            self.assertEqual(result["checkpoint"], output / "final.zip")

    def test_schedule_requires_explicit_evaluation_policy_contract_and_finite_gates(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input")
            schedule = json.loads(self.write_schedule(root).read_text())
            del schedule["phases"][0]["evaluation"]["policy_kind"]
            (root / "schedule.json").write_text(json.dumps(schedule), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "policy_kind"):
                runner.run_schedule(schedule_path=root / "schedule.json", input_checkpoint=checkpoint,
                    output_root=root / "missing-contract", train_block=self.fake_train([]),
                    policy_loader=self.fake_policy_loader([]), evaluator=self.fake_evaluator([]))

            schedule = json.loads(self.write_schedule(root).read_text())
            schedule["phases"][0]["gates"]["deterministic"]["win_rate_min"] = float("nan")
            (root / "schedule.json").write_text(json.dumps(schedule), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "finite"):
                runner.run_schedule(schedule_path=root / "schedule.json", input_checkpoint=checkpoint,
                    output_root=root / "nonfinite-gate", train_block=self.fake_train([]),
                    policy_loader=self.fake_policy_loader([]), evaluator=self.fake_evaluator([]))

    def test_tampered_checkpoint_inventory_or_eval_manifest_can_never_promote(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input")

            def tampered_train(**kwargs):
                result = self.fake_train([])(**kwargs)
                Path(result["checkpoint"]).write_bytes(b"changed after inventory")
                return result

            with self.assertRaisesRegex(ValueError, "inventory"):
                runner.run_schedule(schedule_path=self.write_schedule(root), input_checkpoint=checkpoint,
                    output_root=root / "bad-checkpoint", train_block=tampered_train,
                    policy_loader=self.fake_policy_loader([]), evaluator=self.fake_evaluator([]))

            def tampered_eval(**kwargs):
                result = self.fake_evaluator([])(**kwargs)
                (Path(kwargs["output_dir"]) / "episodes.jsonl").write_text("{}\n", encoding="utf8")
                return result

            with self.assertRaisesRegex(ValueError, "manifest"):
                runner.run_schedule(schedule_path=self.write_schedule(root), input_checkpoint=checkpoint,
                    output_root=root / "bad-eval", train_block=self.fake_train([]),
                    policy_loader=self.fake_policy_loader([]), evaluator=tampered_eval)

    def test_checkpoint_outside_block_and_in_memory_eval_claims_cannot_promote(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input")

            def outside_train(**kwargs):
                result = self.fake_train([])(**kwargs)
                outside = root / "outside.zip"
                outside.write_bytes(Path(result["checkpoint"]).read_bytes())
                return {**result, "checkpoint": outside}

            with self.assertRaisesRegex(ValueError, "outside the block inventory"):
                runner.run_schedule(schedule_path=self.write_schedule(root), input_checkpoint=checkpoint,
                    output_root=root / "outside", train_block=outside_train,
                    policy_loader=self.fake_policy_loader([]), evaluator=self.fake_evaluator([]))

            losing_rows = [
                {"mode": "deterministic", "result": "loss", "self_stock_losses": 1,
                 "terminal_cause": "self_fall", "hits": 0, "damage_done": 0.0},
                {"mode": "sampled", "result": "loss", "self_stock_losses": 1,
                 "terminal_cause": "self_fall", "hits": 0, "damage_done": 0.0},
            ]

            def lying_evaluator(**kwargs):
                result = self.fake_evaluator([], episodes=losing_rows)(**kwargs)
                result["episodes"] = [
                    {"case_id": "fake_case", "mode": mode, "result": "win", "self_stock_losses": 0, "hits": 99, "damage_done": 99.0}
                    for mode in ("deterministic", "sampled")
                ]
                return result

            result = runner.run_schedule(schedule_path=self.write_schedule(root), input_checkpoint=checkpoint,
                output_root=root / "lying-eval", train_block=self.fake_train([]),
                policy_loader=self.fake_policy_loader([]), evaluator=lying_evaluator)
            self.assertEqual(result["decision"], "retain")

    def test_resume_rejects_checkpoint_or_receipt_tampering(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input")
            output = root / "run"
            runner.run_schedule(schedule_path=self.write_schedule(root), input_checkpoint=checkpoint,
                output_root=output, train_block=self.fake_train([]),
                policy_loader=self.fake_policy_loader([]), evaluator=self.fake_evaluator([]))
            trained = output / "phase-000-contact_a" / "block" / "final.zip"
            trained.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "resume"):
                runner.run_schedule(schedule_path=root / "schedule.json", input_checkpoint=checkpoint,
                    output_root=output, train_block=self.fake_train([]),
                    policy_loader=self.fake_policy_loader([]), evaluator=self.fake_evaluator([]), resume=True)

    def test_resume_rejects_a_changed_but_valid_schedule(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input")
            output = root / "run"
            schedule_path = self.write_schedule(root)
            runner.run_schedule(schedule_path=schedule_path, input_checkpoint=checkpoint,
                output_root=output, train_block=self.fake_train([]),
                policy_loader=self.fake_policy_loader([]), evaluator=self.fake_evaluator([]))
            schedule = json.loads(schedule_path.read_text())
            schedule["phases"][0]["gates"]["deterministic"]["win_rate_min"] = 0.75
            schedule_path.write_text(json.dumps(schedule), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "resume"):
                runner.run_schedule(schedule_path=schedule_path, input_checkpoint=checkpoint,
                    output_root=output, train_block=self.fake_train([]),
                    policy_loader=self.fake_policy_loader([]), evaluator=self.fake_evaluator([]), resume=True)

    def test_failure_receipts_do_not_persist_provider_credentials(self):
        import ai2.training.native_curriculum_runner as runner

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input")
            secret = "Authorization-Bearer-should-not-be-saved"

            def fail(**_kwargs):
                raise RuntimeError(secret)

            with self.assertRaises(RuntimeError):
                runner.run_schedule(schedule_path=self.write_schedule(root), input_checkpoint=checkpoint,
                    output_root=root / "failed", train_block=fail,
                    policy_loader=self.fake_policy_loader([]), evaluator=self.fake_evaluator([]))
            receipt = (root / "failed" / "phase-000-contact_a" / "phase_receipt.json").read_text()
            self.assertNotIn(secret, receipt)
            self.assertNotIn("traceback", receipt)


if __name__ == "__main__":
    unittest.main()
