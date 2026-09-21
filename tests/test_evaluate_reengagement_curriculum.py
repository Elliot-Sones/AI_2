import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch


class EvaluateReengagementCurriculumTests(unittest.TestCase):
    def subject(self):
        import evaluate_reengagement_curriculum as module

        return module

    def success_trace(self, *, mode="deterministic", slot=0, seed=134000):
        case_id = f"reengagement-{mode}-p{slot}-s{seed}"
        return {
            "schema_version": "b2-reengagement-trace-v1",
            "case_id": case_id,
            "mode": mode,
            "seed": seed,
            "player_slot": slot,
            "curriculum_version": "bridge_b2_reengagement-v1",
            "outcome": "success",
            "tick": 120,
            "first_hit_tick": 20,
            "second_hit_tick": 80,
            "relocation_tick": 40,
            "relocation_gap": 3.5,
            "pursuit_tick": 70,
            "gap_closed": 1.2,
            "learner_progress": 1.1,
            "distinct_hit_ids": ["attack-1", "attack-2"],
            "relocation_pre_target": {"x": 6.0, "state": "StandingState"},
            "relocation_post_target": {"x": 1.0, "state": "StandingState", "grounded": True, "fully_supported": True},
            "relocation_boundary": {
                "learner_unchanged": True,
                "attack_resolved": True,
                "stun_resolved": True,
                "players": [
                    {"player_slot": 0, "alive": True, "grounded": True, "fully_supported": True, "horizontal_floor_clearance": 0.4},
                    {"player_slot": 1, "alive": True, "grounded": True, "fully_supported": True, "horizontal_floor_clearance": 0.4},
                ],
                "target_preserved": {"stocks": True, "damage": True, "damage_done": True, "damage_taken_total": True},
            },
            "attributed_damage_events": [
                {"tick": 20, "source_slot": slot, "target_slot": 1 - slot, "attack_id": "attack-1", "amount": 4.0},
                {"tick": 80, "source_slot": slot, "target_slot": 1 - slot, "attack_id": "attack-2", "amount": 5.0},
            ],
            "post_players": [
                {"player_slot": 0, "alive": True, "grounded": True, "fully_supported": True, "horizontal_floor_clearance": 0.4},
                {"player_slot": 1, "alive": True, "grounded": True, "fully_supported": True, "horizontal_floor_clearance": 0.4},
            ],
            "events": [
                {"type": "hit_credit", "tick": 20, "ordinal": 1, "attack_id": "attack-1"},
                {"type": "relocation", "tick": 40, "reason": "encounter_boundary", "relocation_gap": 3.5},
                {"type": "pursuit", "tick": 70, "gap_closed": 1.2, "learner_progress": 1.1},
                {"type": "hit_credit", "tick": 80, "ordinal": 2, "attack_id": "attack-2"},
                {"type": "episode_ended", "tick": 120, "outcome": "success"},
            ],
            "falls": {str(slot): 0, str(1 - slot): 0},
            "stock_changes": {str(slot): 0, str(1 - slot): 0},
            "reward_totals": {"first_hit": 0.5, "pursuit": 0.5, "second_hit": 1.0, "completion": 2.0},
        }

    def test_default_banks_are_pinned_balanced_disjoint_and_hashed(self):
        module = self.subject()

        dev = module.build_case_bank("development")
        confirmation = module.build_case_bank("confirmation")

        self.assertEqual(dev.generator_version, "b2-reengagement-bank-v1")
        self.assertEqual([case.seed for case in dev.cases[:20]], list(range(134000, 134020)))
        self.assertEqual([case.seed for case in confirmation.cases[:100]], list(range(144000, 144100)))
        self.assertEqual(len(dev.cases), 40)
        self.assertEqual(len(confirmation.cases), 200)
        self.assertEqual(len({case.case_id for case in dev.cases}), len(dev.cases))
        self.assertFalse({case.seed for case in dev.cases} & {case.seed for case in confirmation.cases})
        for index, seed in enumerate(range(134000, 134020)):
            pair = [case for case in dev.cases if case.seed == seed]
            self.assertEqual({case.mode for case in pair}, {"deterministic", "sampled"})
            self.assertEqual(pair[0].scenario_identity, pair[1].scenario_identity)
            self.assertEqual({case.player_slot for case in pair}, {index % 2})
            expected_direction = -1 if (index // 2) % 2 == 0 else 1
            self.assertEqual({case.direction for case in pair}, {expected_direction})
            self.assertEqual({case.case_id for case in pair}, {f"reengagement-deterministic-p{index % 2}-s{seed}", f"reengagement-sampled-p{index % 2}-s{seed}"})
        self.assertRegex(dev.bank_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(module.validate_bank_manifest(dev.to_manifest())["bank_sha256"], dev.bank_sha256)
        for bad_panel, expected in (("development", 40), ("confirmation", 200)):
            manifest = module.build_case_bank(bad_panel).to_manifest()
            manifest["cases"] = manifest["cases"][:-1]
            with self.assertRaisesRegex(ValueError, "exact"):
                module.validate_bank_manifest(manifest)
            manifest = module.build_case_bank(bad_panel).to_manifest()
            manifest["cases"][0]["direction"] *= -1
            manifest["bank_sha256"] = module._bank_sha(
                [
                    module.EvaluationCase(
                        case_id=case["case_id"],
                        seed=case["seed"],
                        mode=case["mode"],
                        player_slot=case["player_slot"],
                        direction=case["direction"],
                        scenario=case["scenario"],
                    )
                    for case in manifest["cases"]
                ],
                panel=bad_panel,
                generator_version=manifest["generator_version"],
            )
            with self.assertRaisesRegex(ValueError, "direction"):
                module.validate_bank_manifest(manifest)
            self.assertEqual(expected, len(module.build_case_bank(bad_panel).cases))

    def test_bank_manifest_rejects_overlap_duplicates_or_wrong_version(self):
        module = self.subject()
        manifest = module.build_case_bank("development").to_manifest()
        duplicated = json.loads(json.dumps(manifest))
        duplicated["cases"][1] = dict(duplicated["cases"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.validate_bank_manifest(duplicated)
        wrong_version = json.loads(json.dumps(manifest))
        wrong_version["generator_version"] = "old"
        with self.assertRaisesRegex(ValueError, "generator"):
            module.validate_bank_manifest(wrong_version)
        overlap = json.loads(json.dumps(manifest))
        overlap["reserved_training_seeds"] = [134000]
        with self.assertRaisesRegex(ValueError, "overlap"):
            module.validate_bank_manifest(overlap)

    def test_trace_reconstruction_ignores_success_label_without_complete_evidence(self):
        module = self.subject()
        baseline = module.reconstruct_trace(self.success_trace())
        self.assertTrue(baseline["success"])
        self.assertTrue(baseline["evidence_valid"])

        mutations = {
            "same attack": lambda trace: trace.update(distinct_hit_ids=["attack-1", "attack-1"]),
            "missing relocation": lambda trace: trace.pop("relocation_post_target"),
            "weak pursuit": lambda trace: trace.update(gap_closed=0.99),
            "late second hit": lambda trace: trace.update(second_hit_tick=131),
            "unsafe": lambda trace: trace.update(min_clearance=0.1),
            "zero attack": lambda trace: trace.update(distinct_hit_ids=["attack-1", "0"]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                trace = self.success_trace()
                mutate(trace)
                result = module.reconstruct_trace(trace)
                self.assertFalse(result["success"])
                self.assertFalse(result["evidence_valid"])
                self.assertNotEqual(result["failure_category"], "success")

    def test_trace_reconstruction_requires_hit_events_relocation_boundary_and_both_safe_players(self):
        module = self.subject()
        cases = {
            "missing hit ordinal": lambda trace: trace["events"].pop(0),
            "wrong damage tick": lambda trace: trace["attributed_damage_events"][1].update(tick=81),
            "unresolved relocation": lambda trace: trace["relocation_boundary"].update(attack_resolved=False),
            "learner changed": lambda trace: trace["relocation_boundary"].update(learner_unchanged=False),
            "target counters not preserved": lambda trace: trace["relocation_boundary"]["target_preserved"].update(damage=False),
            "target unsafe at end": lambda trace: trace["post_players"][1].update(fully_supported=False),
            "target stock changed": lambda trace: trace.update(stock_changes={str(trace["player_slot"]): 0, str(1 - trace["player_slot"]): 1}),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                trace = self.success_trace()
                mutate(trace)
                result = module.reconstruct_trace(trace)
                self.assertFalse(result["success"])
                self.assertFalse(result["evidence_valid"])

    def test_failure_taxonomy_uses_exact_review_labels(self):
        module = self.subject()
        scenarios = {
            "no_approach": {"outcome": "total_deadline", "movement_request_ticks": 3, "max_initial_approach_progress": 0.2},
            "no_attack_request": {"outcome": "total_deadline", "movement_request_ticks": 0, "light_request_ticks": 0},
            "requested_but_no_start": {"outcome": "total_deadline", "light_request_ticks": 2, "attack_start_count": 0},
            "attack_started_missed": {"outcome": "total_deadline", "light_request_ticks": 2, "attack_start_count": 1},
            "first_hit_only": {"outcome": "total_deadline", "first_hit_tick": 20, "relocation_tick": 40, "renewed_movement_request_ticks": 0},
            "failed_pursuit": {"outcome": "response_deadline", "first_hit_tick": 20, "relocation_tick": 40, "renewed_movement_request_ticks": 2},
            "pursued_but_missed_second": {"outcome": "second_hit_deadline", "first_hit_tick": 20, "relocation_tick": 40, "pursuit_tick": 70},
            "unsafe_support": {"outcome": "unsafe_support"},
            "stock_loss": {"outcome": "own_stock_loss"},
            "deadline": {"outcome": "total_deadline", "first_hit_tick": 20, "relocation_tick": 40, "pursuit_tick": 70},
        }
        for expected, fields in scenarios.items():
            with self.subTest(expected=expected):
                trace = {"player_slot": 0, **fields}
                self.assertEqual(module.reconstruct_trace(trace)["failure_category"], expected)

    def test_stock_loss_taxonomy_collapses_own_and_opponent_stock_losses(self):
        module = self.subject()
        for outcome in ("own_stock_loss", "opponent_stock_loss"):
            with self.subTest(outcome=outcome):
                self.assertEqual(
                    module.reconstruct_trace({"player_slot": 0, "outcome": outcome})["failure_category"],
                    "stock_loss",
                )

    def test_summary_splits_modes_slots_and_reports_b2_taxonomy(self):
        module = self.subject()
        rows = [
            {"case": module.EvaluationCase("reengagement-deterministic-p0-s134000", 134000, "deterministic", 0, -1, {}) , "trace": self.success_trace(mode="deterministic", slot=0, seed=134000)},
            {"case": module.EvaluationCase("reengagement-deterministic-p1-s134001", 134001, "deterministic", 1, -1, {}) , "trace": {**self.success_trace(mode="deterministic", slot=1, seed=134001), "outcome": "own_stock_loss", "falls": {"1": 1}, "stock_changes": {"1": 1}}},
            {"case": module.EvaluationCase("reengagement-sampled-p0-s134000", 134000, "sampled", 0, -1, {}) , "trace": {**self.success_trace(mode="sampled", slot=0, seed=134000), "outcome": "second_hit_deadline", "second_hit_tick": None}},
        ]

        summary = module.summarize_episodes(rows)

        self.assertEqual(summary["modes"]["deterministic"]["episodes"], 2)
        self.assertEqual(summary["modes"]["deterministic"]["slots"]["0"]["full_sequence_successes"], 1)
        self.assertEqual(summary["modes"]["deterministic"]["slots"]["1"]["falls"], 1)
        self.assertEqual(summary["modes"]["sampled"]["deadlines"], 1)
        self.assertIn("first_hit", summary["modes"]["sampled"])
        self.assertIn("pursuit", summary["modes"]["sampled"])
        self.assertIn("second_hit", summary["modes"]["sampled"])
        self.assertIn("safe_resolution", summary["modes"]["sampled"])
        self.assertIn("damage", summary["modes"]["deterministic"])
        self.assertIn("timing", summary["modes"]["deterministic"])
        self.assertEqual(summary["failure_taxonomy"]["success"], 1)
        self.assertEqual(summary["failure_taxonomy"]["stock_loss"], 1)
        self.assertEqual(summary["failure_taxonomy"]["pursued_but_missed_second"], 1)

    def test_cuda_policy_loader_refuses_silent_cpu_fallback(self):
        module = self.subject()
        ppo = Mock()
        with patch.object(module, "require_cuda", side_effect=RuntimeError("CUDA unavailable")):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                module.load_cuda_policy(Path("checkpoint.zip"), ppo_class=ppo)
        ppo.load.assert_not_called()

        policy = Mock()
        policy.policy.parameters.return_value = [type("Param", (), {"device": type("Device", (), {"type": "cpu"})()})()]
        ppo.load.return_value = policy
        with patch.object(module, "require_cuda", return_value="cuda:0"):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                module.load_cuda_policy(Path("checkpoint.zip"), ppo_class=ppo)
        ppo.load.assert_called_with("checkpoint.zip", device="cuda:0")

    def test_run_evaluation_writes_immutable_atomic_artifacts(self):
        module = self.subject()

        class Policy:
            source_path = None
            expected_observation_width = 256

            def __init__(self):
                self.seen_shapes = []

            def predict(self, _obs, *, deterministic):
                self.seen_shapes.append(tuple(_obs.shape))
                return [0.0] * 10

        class Env:
            def __init__(self, case):
                self.case = case
                self.trace = self.success_trace(case.mode, case.player_slot, case.seed)
                self.done = False

            def reset(self, *, seed=None, options=None):
                return [0.0] * 64, {"trace": self.trace}

            def step(self, action):
                self.done = True
                return [0.0] * 64, 1.0, True, False, {"trace": self.trace}

            def close(self):
                pass

            @staticmethod
            def success_trace(mode, slot, seed):
                return EvaluateReengagementCurriculumTests().success_trace(mode=mode, slot=slot, seed=seed)

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "eval"
            policy = Policy()
            result = module.run_evaluation(
                policy=policy,
                bank=module.build_case_bank("development"),
                output_dir=output,
                env_factory=lambda case, trace_dir: Env(case),
                command=["evaluate", "--fake"],
                checkpoint_path=None,
                source_paths=[Path("evaluate_reengagement_curriculum.py")],
            )
            self.assertEqual(policy.seen_shapes, [(256,)] * 40)
            self.assertTrue((output / "episodes" / "reengagement-deterministic-p0-s134000.json").is_file())
            self.assertTrue((output / "traces" / "reengagement-deterministic-p0-s134000.jsonl").is_file())
            for name in ("bank-manifest.json", "summary.json", "result.json", "environment-inventory.json", "command-receipt.json", "source-hashes.json", "artifact-inventory.json"):
                self.assertTrue((output / name).is_file())
            inventory = json.loads((output / "artifact-inventory.json").read_text())
            inventoried = {row["path"] for row in inventory["files"]}
            self.assertIn("traces/reengagement-deterministic-p0-s134000.jsonl", inventoried)
            self.assertTrue(all(":" not in path.name for path in (output / "traces").iterdir()))
            self.assertEqual(result["summary"]["episodes"], 40)
            with self.assertRaises(FileExistsError):
                module.run_evaluation(
                    policy=Policy(),
                    bank=module.build_case_bank("development"),
                    output_dir=output,
                    env_factory=lambda case, trace_dir: Env(case),
                    command=["again"],
                )

    def test_preflight_rejects_bad_observation_space_before_episode_construction(self):
        module = self.subject()

        class BadPolicy:
            expected_observation_width = 64

            def predict(self, _obs, *, deterministic):
                return [0.0] * 10

        with TemporaryDirectory() as tmp:
            env_factory = Mock()
            with self.assertRaisesRegex(ValueError, "256"):
                module.run_evaluation(
                    policy=BadPolicy(),
                    bank=module.build_case_bank("development"),
                    output_dir=Path(tmp) / "eval",
                    env_factory=env_factory,
                )
            env_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
