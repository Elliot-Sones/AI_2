from __future__ import annotations

import json
import unittest
from unittest.mock import patch


class NativeCombatRewardGateTests(unittest.TestCase):
    def test_local_gate_runs_cpu_oracle_and_skips_only_cuda_when_allowed(self):
        import ai2.simulation.native_combat_reward_gate as gate

        with gate.temporary_output_dir("gate-test-") as output_dir:
            result = gate.run_gate(output_dir=output_dir, allow_cuda_skip=True)

            self.assertIn(result["status"], {"passed", "skipped_cuda"})
            self.assertEqual(result["cpu"]["status"], "passed")
            self.assertEqual(result["cpu"]["scenario_count"], len(gate.SCENARIOS))
            self.assertEqual(result["scenario_names"], [case.name for case in gate.SCENARIOS])
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "events.jsonl").is_file())
            self.assertTrue(result["source_hashes"]["native/combat_reward.h"]["sha256"])
            if result["status"] == "skipped_cuda":
                self.assertEqual(result["cuda"]["status"], "skipped")
                self.assertIn("CUDA", result["cuda"]["reason"])

            rows = [
                json.loads(line)
                for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["scenario"] for row in rows], [case.name for case in gate.SCENARIOS])
            components = {component for row in rows for component in row["components"]}
            self.assertTrue(
                {
                    "attack_start",
                    "landed_hit_damage",
                    "ko_win_signs",
                    "clean_win",
                    "approach_proximity",
                    "fall_walkoff",
                    "recovery",
                    "terminal_vs_truncation",
                    "episode_reset",
                    "invalid_config_byte_preservation",
                }.issubset(components)
            )
            invalid = next(row for row in rows if row["scenario"] == "invalid_config_preserves_state")
            self.assertNotEqual(invalid["cpu"]["step_rc"], 0)
            self.assertTrue(invalid["cpu"]["state_bytes_preserved"])
            if result["status"] == "skipped_cuda":
                self.assertTrue(all(row["comparison_status"] == "not_run" for row in rows))
                self.assertTrue(all(row["exact"] is False for row in rows))
                self.assertTrue(all(row["cuda"] is None for row in rows))
            else:
                self.assertEqual(result["cuda"]["status"], "passed")
                self.assertTrue(all(row["comparison_status"] == "matched" for row in rows))
                self.assertTrue(all(row["exact"] is True for row in rows))
                self.assertTrue(all(row["cuda"] is not None for row in rows))

    def test_semantic_expectations_are_asserted_for_every_scenario(self):
        import ai2.simulation.native_combat_reward_gate as gate

        with gate.temporary_output_dir("gate-test-") as output_dir:
            gate.run_gate(output_dir=output_dir, allow_cuda_skip=True)
            rows = [
                json.loads(line)
                for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        by_name = {row["scenario"]: row["cpu"] for row in rows}
        self.assertEqual(by_name["approach_proximity"]["approach_delta"], 1.0)
        self.assertEqual(by_name["approach_proximity"]["distance"], 7.0)
        self.assertEqual(by_name["approach_proximity"]["terms"][0][0], 1.0)
        self.assertEqual(by_name["approach_proximity"]["terms"][1][0], 1.0)

        attack = by_name["attack_contact_damage"]
        self.assertEqual(attack["attack_starts"], [1, 0])
        self.assertEqual(attack["contacts"], [1, 0])
        self.assertEqual(attack["damage_events"], [1, 0])
        self.assertEqual(attack["damage_done"], [7.5, 0.0])
        self.assertEqual(attack["damage_taken"], [0.0, 7.5])
        self.assertEqual(attack["terms"][0][3], 7.5)
        self.assertEqual(attack["terms"][1][4], 7.5)

        terminal = by_name["terminal_ko_win_clean"]
        self.assertTrue(terminal["terminated"])
        self.assertFalse(terminal["truncated"])
        self.assertEqual(terminal["win_mask"], 1)
        self.assertEqual(terminal["ko_events"], [0, 1])
        self.assertEqual(terminal["terms"][0][6], 1.0)
        self.assertEqual(terminal["terms"][1][6], -1.0)
        self.assertEqual(terminal["terms"][0][7], 1.0)
        self.assertEqual(terminal["terms"][1][7], -1.0)
        self.assertEqual(terminal["clean_win"], [1, 0])

        walkoff = by_name["fall_walkoff"]
        self.assertEqual(walkoff["offstage"], [1, 0])
        self.assertEqual(walkoff["walkoff"], [1, 0])
        self.assertEqual(walkoff["terms"][0][10], 1.0)
        self.assertEqual(walkoff["terms"][0][11], 1.0)

        recovery = by_name["recovery"]
        self.assertEqual(recovery["recovery"], [1, 0])
        self.assertEqual(recovery["terms"][0][12], 1.0)

        timeout = by_name["timeout_truncation"]
        self.assertFalse(timeout["terminated"])
        self.assertTrue(timeout["truncated"])
        self.assertTrue(timeout["record"]["timeout"])
        self.assertEqual(timeout["terms"][0][9], 1.0)
        self.assertEqual(timeout["terms"][1][9], 1.0)

        reset = by_name["episode_reset"]
        self.assertEqual(reset["state_tick"], 0)
        self.assertEqual(reset["returns"], [0.0, 0.0])
        self.assertEqual(reset["term_totals"][0], [0.0] * gate.AI2CR_TERMS)
        self.assertEqual(reset["term_totals"][1], [0.0] * gate.AI2CR_TERMS)

        invalid = by_name["invalid_config_preserves_state"]
        self.assertNotEqual(invalid["step_rc"], 0)
        self.assertTrue(invalid["state_bytes_preserved"])
        self.assertEqual(invalid["before_hash"], invalid["after_hash"])

    def test_runtime_semantic_validation_rejects_wrong_labels(self):
        import ai2.simulation.native_combat_reward_gate as gate

        with gate.temporary_output_dir("gate-test-") as output_dir:
            gate.run_gate(output_dir=output_dir, allow_cuda_skip=True)
            rows = [
                json.loads(line)
                for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
        attack = next(row["cpu"] for row in rows if row["scenario"] == "attack_contact_damage")
        corrupted = json.loads(json.dumps(attack))
        corrupted["damage_done"][0] = 0.0

        with self.assertRaisesRegex(gate.GateError, "attack_contact_damage"):
            gate.validate_semantics("attack_contact_damage", corrupted)

    def test_event_rows_report_matched_mismatched_and_not_run_explicitly(self):
        import ai2.simulation.native_combat_reward_gate as gate

        cpu = {"scenario_id": 0, "total": [1.0, 1.0]}
        matched = gate._event_rows([cpu], [dict(cpu)])[0]
        self.assertTrue(matched["exact"])
        self.assertEqual(matched["comparison_status"], "matched")
        self.assertIsNone(matched["first_mismatch"])

        cuda = {"scenario_id": 0, "total": [1.0, 2.0]}
        mismatched = gate._event_rows([cpu], [cuda])[0]
        self.assertFalse(mismatched["exact"])
        self.assertEqual(mismatched["comparison_status"], "mismatched")
        self.assertEqual(mismatched["first_mismatch"]["path"], "total[1]")

        not_run = gate._event_rows([cpu], None)[0]
        self.assertFalse(not_run["exact"])
        self.assertEqual(not_run["comparison_status"], "not_run")
        self.assertIsNone(not_run["cuda"])

    def test_output_preflight_rejects_existing_files_before_compile(self):
        import ai2.simulation.native_combat_reward_gate as gate

        with gate.temporary_output_dir("gate-test-") as output_dir:
            output_dir.mkdir(exist_ok=True)
            (output_dir / "summary.json").write_text("old\n", encoding="utf-8")
            with patch.object(gate, "run_backend") as run_backend:
                with self.assertRaisesRegex(gate.GateError, "summary.json"):
                    gate.run_gate(output_dir=output_dir, allow_cuda_skip=True)
                run_backend.assert_not_called()

    def test_output_preflight_rejects_symlinked_output_dir_before_compile(self):
        import ai2.simulation.native_combat_reward_gate as gate

        with gate.temporary_output_dir("gate-test-") as root:
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with patch.object(gate, "run_backend") as run_backend:
                with self.assertRaisesRegex(gate.GateError, "symlink"):
                    gate.run_gate(output_dir=link, allow_cuda_skip=True)
                run_backend.assert_not_called()

    def test_abi_validation_rejects_size_or_offset_mismatch(self):
        import ai2.simulation.native_combat_reward_gate as gate

        abi = gate.expected_abi()
        gate.validate_abi(dict(abi))
        broken = dict(abi)
        broken["sizeof"] += 8

        with self.assertRaisesRegex(gate.GateError, "ABI"):
            gate.validate_abi(broken)

    def test_mismatch_report_names_exact_field_and_values(self):
        import ai2.simulation.native_combat_reward_gate as gate

        left = {"total": [1.0, 0.0], "nested": {"value": 3}}
        right = {"total": [1.0, 2.0], "nested": {"value": 3}}

        mismatch = gate.first_mismatch(left, right)

        self.assertEqual(mismatch["path"], "total[1]")
        self.assertEqual(mismatch["cpu"], 0.0)
        self.assertEqual(mismatch["cuda"], 2.0)

    def test_require_cuda_fails_closed_without_cuda(self):
        import ai2.simulation.native_combat_reward_gate as gate

        if gate.cuda_available():
            self.skipTest("CUDA available on this host")
        with gate.temporary_output_dir("gate-test-") as output_dir:
            with self.assertRaisesRegex(gate.GateError, "CUDA"):
                gate.run_gate(output_dir=output_dir, require_cuda=True, allow_cuda_skip=False)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["cpu"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
