import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class NativeCurriculumRunnerTests(unittest.TestCase):
    def write_schedule(self, root, *, gates=None, phase_id="contact_a"):
        schedule = {
            "schema_version": "ai2.native_curriculum_schedule.v1",
            "phases": [
                {
                    "id": phase_id,
                    "block": {"type": "native_contact", "args": {"rollouts": 1}},
                    "evaluation": {"modes": ["deterministic", "sampled"], "episode_max": 4},
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
            import benchmark_native_contact_training as contact

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
            (output / "episodes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf8",
            )
            summary = {"episodes": len(rows), "wins": sum(1 for row in rows if row["result"] == "win")}
            if not omit_manifest:
                (output / "manifest.json").write_text(json.dumps({"summary": summary}), encoding="utf8")
            return {"episodes": rows, "summary": summary, "manifest": {"summary": summary}}

        return evaluate

    def test_accepts_native_contact_phase_and_promotes_with_both_modes_passing(self):
        import native_curriculum_runner as runner

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
        import native_curriculum_runner as runner

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
            self.assertFalse(phase_receipt["gate"]["passed"])
            self.assertIn("sampled.win_rate", phase_receipt["gate"]["failed"])

    def test_schedule_fails_closed_without_deterministic_and_sampled_gates(self):
        import native_curriculum_runner as runner

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
        import native_curriculum_runner as runner

        deterministic_only = [
            {"mode": "deterministic", "result": "win", "self_stock_losses": 0, "terminal_cause": "opponent_fall", "hits": 2, "damage_done": 5.0},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.zip"
            checkpoint.write_bytes(b"input checkpoint")
            with self.assertRaisesRegex(ValueError, "missing required evaluation mode"):
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
        import native_curriculum_runner as runner

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
        import native_curriculum_runner as runner

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
            with self.assertRaisesRegex(ValueError, "cannot resume from non-promoted phase"):
                runner.run_schedule(
                    schedule_path=self.write_schedule(root),
                    input_checkpoint=checkpoint,
                    output_root=root / "rejected",
                    train_block=self.fake_train([]),
                    policy_loader=self.fake_policy_loader([]),
                    evaluator=self.fake_evaluator([]),
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
