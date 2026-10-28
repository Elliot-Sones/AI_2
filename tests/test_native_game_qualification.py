"""Small actual-reference checks; comprehensive scenarios are generated, not run."""
from __future__ import annotations

from ai2.paths import source_path

import hashlib
import importlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ai2.simulation.native_attack_tables import DEFAULT_SOURCE

ROOT = Path(__file__).resolve().parents[1]


class QualificationTests(unittest.TestCase):
    def module(self):
        self.assertTrue((source_path(ROOT, "native_game_qualification.py")).is_file(), "qualification implementation missing")
        return importlib.import_module('ai2.evaluation.native_game_qualification')

    def test_presets_reproduce_inputs_and_do_not_claim_observed_coverage(self):
        m = self.module()
        self.assertEqual(m.build_cases("integration", seed=17), m.build_cases("integration", seed=17))
        smoke = m.build_cases("smoke", seed=17)
        self.assertEqual([len(case["actions"]) for case in smoke], [3, 3, 3])
        self.assertNotEqual(m.build_cases("smoke", seed=18)[1]["actions"], smoke[1]["actions"])
        cases = m.build_cases("integration", seed=17)
        self.assertEqual(hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                         "1cf6dbe73d70dbaaaeb96e4e2620ea556b9c33c7fe929eb23f7e41d270385022")
        self.assertTrue({"idle", "random", "simultaneous", "drop_pickup", "ko_respawn", "match_end", "timeout", "moving_ledge"}
                        <= {case["family"] for case in cases})
        self.assertTrue(all(set(case) == {"case_id", "seed", "family", "scenario", "actions", "intent"} for case in cases))
        full = m.build_cases("comprehensive", seed=17)
        self.assertEqual(sum(case["family"] == "move_candidate" for case in full), 264)
        binaries = [case for case in full if case["family"] == "binary_input"]
        self.assertEqual(len(binaries), 1024)
        self.assertEqual({tuple(c["actions"][0][0]) for c in binaries}, {tuple(c["actions"][0][1]) for c in binaries})

    def test_targeted_preset_is_separate_and_reports_actual_evidence(self):
        m = self.module()
        self.assertTrue(hasattr(m, "targeted_evidence"), "targeted behavioral evidence missing")
        cases = m.build_cases("targeted", seed=17)
        self.assertEqual({case["family"] for case in cases}, {"dodge_overlap", "ordered_simultaneous", "max_charge"})
        self.assertLessEqual(sum(len(case["actions"]) for case in cases), 700)
        self.assertEqual(len(m.build_cases("integration", seed=17)), 12)
        from ai2.evaluation.combat_episode_capture import capture_raw
        from ai2.evaluation.combat_episode_parity import compare_capture
        from ai2.evaluation.native_game_snapshot import GameParityAdapter
        observed = []
        for case in cases:
            with self.subTest(case=case["case_id"]):
                document = capture_raw(DEFAULT_SOURCE.parent.parent, seed=case["seed"],
                                       actions=case["actions"], scenario=case["scenario"])
                report = compare_capture(document, GameParityAdapter(), scope="gameplay_projection_v1")
                self.assertTrue(report["exact"], report["first_divergence"])
                observed.append((case, m.targeted_evidence(document)))
        for case, evidence in observed:
            self.assertTrue(all("raw_event_sequence" in event for event in evidence["damage"]), case["case_id"])
            if case["family"] == "dodge_overlap":
                self.assertTrue(evidence["dodge_invulnerable_overlaps"], case["case_id"])
            if case["family"] == "ordered_simultaneous":
                self.assertTrue(evidence["simultaneous_attack_start_ticks"], case["case_id"])
                self.assertTrue(evidence["phase0_damage_before_phase1_stun"], case["case_id"])
            if case["family"] == "max_charge":
                self.assertTrue(evidence["max_charge_then_hits"], case["case_id"])

    def test_completion_preset_uses_reset_only_geometry_and_ground_settling(self):
        cases = self.module().build_cases("coverage_completion", seed=17)
        targets = {tuple(c["intent"][k] for k in ("weapon", "move", "slot", "facing", "candidate_outcome")) for c in cases}
        self.assertEqual(len(targets), 52)
        self.assertEqual(len(cases), 156)
        for case in cases:
            self.assertEqual(set(case["scenario"]), {"players"})
            self.assertEqual(len(case["actions"]), 100)
            if case["intent"]["move"] < 8:
                self.assertFalse(any(any(player) for row in case["actions"][:15] for player in row))
        self.assertEqual(len(self.module().build_cases("integration")), 12)

    def test_final_completion_targets_only_eight_residual_spear_hits(self):
        cases = self.module().build_cases("coverage_completion_final")
        targets = {tuple(c["intent"][k] for k in ("weapon", "move", "slot", "facing", "candidate_outcome")) for c in cases}
        self.assertEqual(len(targets), 8)
        self.assertEqual(len(cases), 24)
        self.assertTrue(all(c["intent"]["weapon"] == "Spear" for c in cases))

    def test_original_terminal_and_timeout_are_both_preserved(self):
        m = self.module()
        boundary = [case for case in m.build_cases("comprehensive", seed=17) if case["family"] == "terminal_timeout"]
        self.assertEqual(len(boundary), 1, "simultaneous terminal/timeout fixture missing")
        self.assertEqual(len(m.build_cases("terminal_timeout", seed=17)), 1)
        from ai2.evaluation.combat_episode_capture import capture_raw
        from ai2.evaluation.combat_episode_parity import compare_capture
        from ai2.evaluation.native_game_snapshot import GameParityAdapter
        case = boundary[0]
        document = capture_raw(DEFAULT_SOURCE.parent.parent, seed=case["seed"],
                               actions=case["actions"], scenario=case["scenario"])
        self.assertEqual(len(document["ticks"]), 2)
        self.assertTrue(document["ticks"][-1]["outputs"]["terminated"])
        self.assertTrue(document["ticks"][-1]["outputs"]["truncated"])
        report = compare_capture(document, GameParityAdapter(), scope="gameplay_projection_v1")
        self.assertTrue(report["exact"], report["first_divergence"])

    def test_actual_smoke_preserves_hashes_and_refuses_overwrite(self):
        m = self.module()
        with tempfile.TemporaryDirectory(prefix="ai2-qualification-test-") as folder:
            output = Path(folder) / "smoke"
            result = m.run_qualification(DEFAULT_SOURCE.parent.parent, output, preset="smoke", seed=17)
            self.assertTrue(result["exact"], {"status": result["status"], "error": result.get("error"),
                                            "divergences": [c["report"]["first_divergence"] for c in result["cases"]]})
            self.assertEqual(result["compared_ticks"], 9)
            self.assertFalse(result["full_combat_coverage"])
            self.assertEqual(result["status"], "exact_for_recorded_cases")
            receipt = json.loads((output / "receipt.json").read_text())
            for row in receipt["files"]:
                path = output / row["path"]
                self.assertEqual(path.stat().st_size, row["bytes"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
            self.assertEqual(json.loads((output / "summary.json").read_text()), result)
            self.assertTrue(result["provenance"]["native_library"]["sha256"])
            before = (output / "summary.json").read_bytes()
            with self.assertRaises(FileExistsError):
                m.run_qualification(DEFAULT_SOURCE.parent.parent, output, preset="smoke", seed=17)
            self.assertEqual((output / "summary.json").read_bytes(), before)

    def test_actual_native_negative_control_stops_and_preserves_first_case(self):
        m = self.module()
        from ai2.evaluation.native_game_snapshot import GameParityAdapter

        class CorruptReward(GameParityAdapter):
            def step(self, actions, draws):
                row = super().step(actions, draws)
                row["outputs"]["rewards"]["0"] += .125
                return row

        with tempfile.TemporaryDirectory(prefix="ai2-qualification-negative-") as folder:
            output = Path(folder) / "bad"
            result = m.run_qualification(DEFAULT_SOURCE.parent.parent, output, preset="smoke", seed=17,
                                         adapter_factory=CorruptReward)
            self.assertFalse(result["exact"])
            self.assertEqual(result["status"], "mismatch")
            self.assertEqual(len(result["cases"]), 1)
            self.assertEqual(result["cases"][0]["report"]["first_divergence"]["tick"], 1)
            self.assertEqual(len(list(output.glob("*/capture.json.gz"))), 1)
            inputs = json.loads((output / "idle-000" / "inputs.json").read_text())
            self.assertEqual(inputs["seed"], 17)
            self.assertEqual(len(inputs["actions"]), 3)
            self.assertTrue((output / "receipt.json").is_file())

    def test_cli_lists_without_loading_adapter_and_reports_success(self):
        m = self.module()
        command = [sys.executable, str(source_path(ROOT, "native_game_qualification.py")), "--preset", "smoke", "--seed", "17"]
        listed = subprocess.run([*command, "--list"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(json.loads(listed.stdout)["requested_ticks"], 9)
        with tempfile.TemporaryDirectory(prefix="ai2-qualification-cli-") as folder:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = m.main(["--preset", "smoke", "--seed", "17", "--source-root", str(DEFAULT_SOURCE.parent.parent),
                                 "--output", str(Path(folder) / "smoke")])
            self.assertEqual(status, 0, stdout.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["status"], "exact_for_recorded_cases")


if __name__ == "__main__":
    unittest.main()
