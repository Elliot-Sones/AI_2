from __future__ import annotations

import json
import unittest


class NativeCombatRewardGateTests(unittest.TestCase):
    def test_local_gate_runs_cpu_oracle_and_skips_only_cuda_when_allowed(self):
        import native_combat_reward_gate as gate

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

    def test_mismatch_report_names_exact_field_and_values(self):
        import native_combat_reward_gate as gate

        left = {"total": [1.0, 0.0], "nested": {"value": 3}}
        right = {"total": [1.0, 2.0], "nested": {"value": 3}}

        mismatch = gate.first_mismatch(left, right)

        self.assertEqual(mismatch["path"], "total[1]")
        self.assertEqual(mismatch["cpu"], 0.0)
        self.assertEqual(mismatch["cuda"], 2.0)

    def test_require_cuda_fails_closed_without_cuda(self):
        import native_combat_reward_gate as gate

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
