"""Strict comparison must catch corruption, not silently project it away."""
from __future__ import annotations

from ai2.paths import source_path

import copy
import importlib
import math
from pathlib import Path
import unittest

from tests import test_combat_episode_capture as capture_tests

ROOT = Path(__file__).resolve().parents[1]


class RecordedAdapter:
    """Protocol test double, not evidence that a native simulator is correct."""
    def __init__(self, document, mutate=None):
        self.doc = document
        self.index = 0
        self.mutate = mutate

    def result(self, row):
        out = {k: copy.deepcopy(row[k]) for k in ("outputs", "state", "events")}
        out["draws_consumed"] = len(row["draws"])
        if self.mutate and self.index == 1:
            self.mutate(out)
        return out

    def reset(self, draws, scenario):
        assert draws == self.doc["reset"]["draws"]
        assert scenario == self.doc["metadata"]["scenario"]
        return self.result(self.doc["reset"])

    def step(self, actions, draws):
        row = self.doc["ticks"][self.index]
        assert actions == row["actions"] and draws == row["draws"]
        self.index += 1
        return self.result(row)


class CombatParityTests(unittest.TestCase):
    document = None
    module = capture_tests.CombatCaptureTests.module
    document_copy = capture_tests.CombatCaptureTests.document_copy
    def parity(self):
        self.assertTrue((source_path(ROOT, "combat_episode_parity.py")).is_file(), "combat parity implementation missing")
        return importlib.import_module('ai2.evaluation.combat_episode_parity')

    def test_protocol_compares_every_field_without_state_injection(self):
        m = self.parity()
        doc = self.document_copy()
        result = m.compare_capture(doc, RecordedAdapter(doc))
        self.assertTrue(result["exact"])
        self.assertEqual(result["compared_ticks"], 45)
        self.assertFalse(result["full_combat_coverage"])

    def test_weapon_attack_event_rng_and_signed_zero_mutations_fail(self):
        m = self.parity()
        doc = self.document_copy()
        mutations = [lambda r: r["state"]["players"][0].__setitem__("weapon", "Spear"),
                     lambda r: r["state"]["players"][0]["bank"]["8"].__setitem__("stun_time_stored", 99),
                     lambda r: r["events"].append({"kind": "invented"}),
                     lambda r: r.__setitem__("draws_consumed", -1),
                     lambda r: r["state"]["players"][0].pop("hurtbox")]
        for mutate in mutations:
            verdict = m.compare_capture(doc, RecordedAdapter(doc, mutate))
            self.assertFalse(verdict["exact"])
            self.assertEqual(verdict["first_divergence"]["tick"], 1)
        self.assertTrue(m.diff_records({"x": 0.0}, {"x": -0.0}))

    def test_coverage_uses_actual_enum_ids_two_through_twelve(self):
        missing = self.parity().coverage([self.document_copy()])["missing_move_cases"]
        self.assertTrue(any(case[1] == 12 for case in missing))
        self.assertFalse(any(case[1] == 1 for case in missing))

    def test_explicit_projection_retains_gameplay_and_declares_exclusions(self):
        m = self.parity()
        self.assertTrue(hasattr(m, "gameplay_projection"), "explicit gameplay projection missing")
        doc = self.document_copy()
        projected = m.gameplay_projection(doc["ticks"][25])
        self.assertNotIn("constants", projected["state"]["players"][0])
        self.assertIn("weapon", projected["state"]["players"][0])
        self.assertIn("move_manager", projected["state"]["players"][0]["bank"]["8"])
        reset = m.gameplay_projection(doc["reset"])
        self.assertTrue(all(type(v) is float for v in reset["state"]["players"][0]["start_position"]))
        self.assertTrue(all(type(v) is float for v in reset["state"]["platform"]["waypoint1"]))
        self.assertEqual(projected["state"]["world"], doc["ticks"][25]["state"]["world"])
        events = [e for row in doc["ticks"] for e in m.gameplay_projection(row)["events"]]
        self.assertTrue(any(e["kind"] == 1 for e in events))
        self.assertTrue(any(e["kind"] == 2 for e in events))
        for row in doc["ticks"]:
            semantic = m.gameplay_projection(row)["events"]
            self.assertEqual([e["event_sequence"] for e in semantic], list(range(len(semantic))))

    def test_power_vectors_normalize_types_without_rounding(self):
        m = self.parity()
        row = self.document_copy()["ticks"][25]
        power = row["state"]["players"][0]["bank"]["8"]["move_manager"]["current_power"]
        power["current_dealt_position_target"] = [0, 1, 2.0]
        power["past_point_positions"] = [[0, 1], [2, 3.0]]
        floating = copy.deepcopy(row)
        float_power = floating["state"]["players"][0]["bank"]["8"]["move_manager"]["current_power"]
        float_power["current_dealt_position_target"] = [0.0, 1.0, 2.0]
        float_power["past_point_positions"] = [[0.0, 1.0], [2.0, 3.0]]
        self.assertFalse(m.diff_records(m.gameplay_projection(row), m.gameplay_projection(floating)))
        float_power["past_point_positions"][0][1] = math.nextafter(1.0, 2.0)
        self.assertTrue(m.diff_records(m.gameplay_projection(row), m.gameplay_projection(floating)))


if __name__ == "__main__":
    unittest.main()
