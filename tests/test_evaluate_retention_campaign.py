import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class RetentionEvaluationTests(unittest.TestCase):
    def subject(self):
        self.assertIsNotNone(importlib.util.find_spec("evaluate_retention_campaign"))
        import evaluate_retention_campaign

        return evaluate_retention_campaign

    def test_builds_exact_three_task_original_game_commands_with_six_mode_schema(self):
        m = self.subject()
        args = SimpleNamespace(
            reference_root=Path("/reference"),
            frozen_root=Path("/oracle"),
            output=Path("/output"),
            checkpoint=Path("/trained.zip"),
            episodes=20,
            deadline_unix=10000,
        )
        jobs = m.evaluation_jobs(args)
        self.assertEqual(
            [(job["task"], job["domain"]) for job in jobs],
            [("combat", "contact"), ("near", "nearby"), ("wide", "wide")],
        )
        commands = [m.evaluation_command(args, job) for job in jobs]
        self.assertTrue(all("/reference/evaluate_combat_curriculum.py" in command for command in commands))
        self.assertTrue(all(command[command.index("--episodes") + 1] == "20" for command in commands))
        self.assertTrue(all("--mode" not in command for command in commands))

    def test_normalizes_prior_like_summaries_to_exact_gate_schema(self):
        m = self.subject()
        summaries = {
            "combat": {
                "modes": {
                    "deterministic": {"episodes": 20, "successes": 18, "self_losses": 0, "opponent_self_losses": 0, "timeouts": 0, "ambiguous": 0},
                    "sampled": {"episodes": 20, "successes": 14, "self_losses": 0, "opponent_self_losses": 0, "timeouts": 0, "ambiguous": 0},
                }
            },
            "near": {
                "modes": {
                    "deterministic": {"episodes": 20, "successes": 19, "falls": 1},
                    "sampled": {"episodes": 20, "successes": 19, "falls": 1},
                }
            },
            "wide": {
                "modes": {
                    "deterministic": {"episodes": 20, "successes": 13, "falls": 1},
                    "sampled": {"episodes": 20, "successes": 13, "falls": 2},
                }
            },
        }
        result = m.build_result(
            summaries,
            candidate={"checkpoint_sha256": "a" * 64},
            qualification={"qualified": True},
            qualification_sha256="b" * 64,
            wall_seconds=1.5,
        )
        self.assertEqual(result["schema_version"], "ai2.retention_evaluation.v1")
        self.assertEqual(result["panels"]["contact"]["deterministic"]["successes"], 18)
        self.assertEqual(result["panels"]["wide"]["sampled"]["falls"], 2)
        self.assertEqual(result["raw_panels"]["wide"], summaries["wide"])

    def test_missing_contact_attribution_is_marked_ambiguous_not_clean(self):
        m = self.subject()
        summaries = {
            "combat": {
                "modes": {
                    "deterministic": {"episodes": 20, "successes": 19},
                    "sampled": {"episodes": 20, "successes": 14},
                }
            },
            "near": {
                "modes": {
                    "deterministic": {"episodes": 20, "successes": 20, "falls": 0},
                    "sampled": {"episodes": 20, "successes": 20, "falls": 0},
                }
            },
            "wide": {
                "modes": {
                    "deterministic": {"episodes": 20, "successes": 13, "falls": 7},
                    "sampled": {"episodes": 20, "successes": 13, "falls": 7},
                }
            },
        }
        result = m.build_result(
            summaries,
            candidate={"checkpoint_sha256": "a" * 64},
            qualification={"qualified": True},
            qualification_sha256="b" * 64,
            wall_seconds=1.5,
        )
        self.assertEqual(result["panels"]["contact"]["deterministic"]["ambiguous"], 20)
        import continue_retention_campaign
        gate = continue_retention_campaign.diagnostic_gate(result, continue_retention_campaign.B40_BASELINE)
        self.assertFalse(gate.eligible)
        self.assertIn("contact_deterministic_ambiguous", gate.reason)

    def test_contact_attribution_derives_actual_b40_like_episode_rows_without_marking_safe_successes_ambiguous(self):
        m = self.subject()
        deterministic = [
            {"mode": "deterministic", "outcome": "success", "success": True, "evidence_valid": True, "training_qualified": True, "stock_events": []}
            for _ in range(19)
        ] + [
            {"mode": "deterministic", "outcome": "timeout", "success": False, "evidence_valid": True, "training_qualified": True, "stock_events": []}
        ]
        sampled = [
            {"mode": "sampled", "outcome": "success", "success": True, "evidence_valid": True, "training_qualified": True, "stock_events": []}
            for _ in range(14)
        ] + [
            {"mode": "sampled", "outcome": "opponent_stock_loss", "success": False, "stock_events": [{"attribution": "unknown"}]}
            for _ in range(2)
        ] + [
            {"mode": "sampled", "outcome": "unsafe_floor_support", "success": False, "evidence_valid": False, "training_qualified": False, "stock_events": []}
            for _ in range(2)
        ] + [
            {"mode": "sampled", "outcome": "no_attack", "success": False, "evidence_valid": True, "training_qualified": False, "stock_events": []}
            for _ in range(2)
        ]
        summaries = {
            "combat": {
                "modes": {
                    "deterministic": {"episodes": 20, "successes": 19},
                    "sampled": {"episodes": 20, "successes": 14},
                },
                "episodes": deterministic + sampled,
            },
            "near": {
                "modes": {
                    "deterministic": {"episodes": 20, "successes": 20, "falls": 0},
                    "sampled": {"episodes": 20, "successes": 20, "falls": 0},
                }
            },
            "wide": {
                "modes": {
                    "deterministic": {"episodes": 20, "successes": 13, "falls": 7},
                    "sampled": {"episodes": 20, "successes": 13, "falls": 7},
                }
            },
        }
        result = m.build_result(
            summaries,
            candidate={"checkpoint_sha256": "a" * 64},
            qualification={"qualified": True},
            qualification_sha256="b" * 64,
            wall_seconds=1.5,
        )
        self.assertEqual(result["panels"]["contact"]["deterministic"]["timeouts"], 1)
        self.assertEqual(result["panels"]["contact"]["deterministic"]["ambiguous"], 0)
        self.assertEqual(result["panels"]["contact"]["sampled"]["ambiguous"], 2)

    def test_contact_attribution_fails_closed_for_unverified_success_or_unknown_outcome(self):
        m = self.subject()
        rows = [
            {
                "mode": "deterministic",
                "outcome": "success",
                "success": True,
                "evidence_valid": False,
                "training_qualified": True,
                "stock_events": [],
            },
            {
                "mode": "deterministic",
                "outcome": "mystery_terminal",
                "success": False,
                "evidence_valid": True,
                "training_qualified": False,
                "stock_events": [],
            },
        ]
        counts = m._derive_contact_attribution(
            {"episodes": rows}, "deterministic", len(rows)
        )
        self.assertEqual(counts["ambiguous"], 2)
        self.assertEqual(counts["self_losses"], 0)
        self.assertEqual(counts["opponent_self_losses"], 0)
        self.assertEqual(counts["timeouts"], 0)

    def test_evaluate_requires_preserved_checkpoint_ack_and_qualification_before_subprocess(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "final.zip"
            checkpoint.write_bytes(b"checkpoint")
            (root / "qualification.json").write_text("{}", encoding="utf8")
            args = SimpleNamespace(
                checkpoint=checkpoint,
                checkpoint_sha256=m.sha256(checkpoint),
                verified_ack=root / "preservation_ack.json",
                qualification=root / "qualification.json",
                reference_root=root,
                frozen_root=root,
                output=root / "eval",
                deadline_unix=10**12,
                episodes=20,
            )
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                out = Path(command[command.index("--output") + 1])
                out.mkdir(parents=True)
                domain = command[command.index("--task") + 1]
                payload = {
                    "modes": {
                        "deterministic": {"episodes": 20, "successes": 20 if domain != "wide" else 13, "falls": 0},
                        "sampled": {"episodes": 20, "successes": 20 if domain != "wide" else 13, "falls": 0},
                    }
                }
                if domain == "combat":
                    for mode in payload["modes"].values():
                        mode.update({"self_losses": 0, "opponent_self_losses": 0, "timeouts": 0, "ambiguous": 0})
                m.atomic_json(out / "summary.json", payload)
                return SimpleNamespace(returncode=0)

            with patch.object(m, "require_qualification", return_value={"qualified": True}) as qualification, \
                 patch.object(m, "require_preserved_checkpoint", return_value={"checkpoint_sha256": args.checkpoint_sha256}) as preserved, \
                 patch.object(m, "require_cuda"), \
                 patch.object(m.subprocess, "run", side_effect=run):
                result = m.evaluate(args)
            preserved.assert_called_once()
            qualification.assert_called_once()
            self.assertEqual(len(calls), 3)
            self.assertTrue((args.output / "inventory.json").is_file())
            self.assertEqual(result["status"], "evaluated_not_promoted")

    def test_evaluate_rejects_reference_policy_shadow_mismatch_before_subprocess(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            local_root = root / "local"
            reference_root = root / "reference"
            local_root.mkdir()
            reference_root.mkdir()
            (local_root / "curriculum_ppo.py").write_text("POLICY_VERSION = 'qualified'\n", encoding="utf8")
            (reference_root / "curriculum_ppo.py").write_text("POLICY_VERSION = 'shadow'\n", encoding="utf8")
            (root / "qualification.json").write_text("{}", encoding="utf8")
            checkpoint = root / "final.zip"
            checkpoint.write_bytes(b"checkpoint")
            args = SimpleNamespace(
                checkpoint=checkpoint,
                checkpoint_sha256=m.sha256(checkpoint),
                verified_ack=root / "preservation_ack.json",
                qualification=root / "qualification.json",
                reference_root=reference_root,
                frozen_root=root / "frozen",
                output=root / "eval",
                deadline_unix=10**12,
                episodes=20,
            )

            with patch.object(m, "ROOT", local_root), \
                 patch.object(m, "require_qualification", return_value={"qualified": True}), \
                 patch.object(m, "require_preserved_checkpoint", return_value={"checkpoint_sha256": args.checkpoint_sha256}), \
                 patch.object(m, "require_cuda"), \
                 patch.object(m.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "reference evaluation policy source differs"):
                    m.evaluate(args)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
