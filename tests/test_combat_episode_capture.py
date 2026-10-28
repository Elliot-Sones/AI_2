"""Actual raw-game capture: two players, deterministic draws, no policy."""
from __future__ import annotations

from ai2.paths import source_path

import copy
import importlib
import os
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(os.environ.get("AI2_FROZEN_SOURCE_ROOT", "/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected"))


class CombatCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = None

    def module(self):
        self.assertTrue((source_path(ROOT, "combat_episode_capture.py")).is_file(), "combat capture implementation missing")
        return importlib.import_module('ai2.evaluation.combat_episode_capture')

    def document_copy(self):
        m = self.module()
        if type(self).document is None:
            actions = [[[0.0] * 10, [0.0] * 10] for _ in range(45)]
            for tick in range(20, 45):
                actions[tick][0][7] = 1.0
                actions[tick][1][8] = 1.0
            type(self).document = m.capture_raw(FROZEN, seed=17, actions=actions)
        return copy.deepcopy(self.document)

    def test_captures_unmasked_both_players_and_full_attack_state(self):
        m = self.module()
        doc = self.document_copy()
        m.validate_capture(doc)
        self.assertEqual(len(doc["ticks"]), 45)
        self.assertEqual(doc["ticks"][20]["actions"][1][8], 1.0)
        self.assertEqual(set(doc["reset"]["outputs"]["obs"]), {"0", "1"})
        self.assertEqual(len(doc["reset"]["outputs"]["obs"]["1"]), 64)
        players = [p for row in doc["ticks"] for p in row["state"]["players"]]
        self.assertTrue(any("move_manager" in p["bank"]["8"] for p in players))
        self.assertTrue(all("weapon" in p and "hurtbox" in p and "damage_velocity" in p for p in players))
        self.assertTrue(any(row["draws"] for row in doc["ticks"]))

    def test_actual_capture_repeats_exactly_and_observer_restores(self):
        m = self.module()
        first = self.document_copy()
        second = m.capture_raw(FROZEN, seed=17, actions=[row["actions"] for row in first["ticks"]])
        self.assertEqual(first, second)
        import environment.environment as game
        self.assertEqual(game.Player.apply_damage.__name__, "apply_damage")

    def test_default_capture_reset_and_steps_match_uninstrumented_original(self):
        import numpy as np
        m = self.module()
        captured = self.document_copy()
        with m.original_game(FROZEN, 17) as (game, _module):
            random.seed(17)
            np.random.seed(17)
            observation, info = game.reset(seed=17)
            snapshots = m.Snapshotter(game)
            self.assertEqual(snapshots.snapshot(), captured["reset"]["state"])
            self.assertEqual(m.outputs(observation, game.rewards, info=info), captured["reset"]["outputs"])
            for row in captured["ticks"]:
                transition = game.step({slot: np.asarray(row["actions"][slot]) for slot in (0, 1)})
                self.assertEqual(m.outputs(*transition), row["outputs"])
                self.assertEqual(snapshots.snapshot(), row["state"])

    def test_schema_rejects_missing_fields_nonfinite_and_tick_gap(self):
        m = self.module()
        for mutate in (lambda d: d["ticks"][0]["state"]["players"][0].pop("weapon"),
                       lambda d: d["ticks"][0]["actions"][0].__setitem__(7, float("nan")),
                       lambda d: d["ticks"][0].__setitem__("tick", 2)):
            doc = self.document_copy()
            mutate(doc)
            with self.assertRaises(ValueError):
                m.validate_capture(doc)

    def test_schema_requires_full_world_and_active_attack_internals(self):
        m = self.module()
        doc = self.document_copy()
        doc["reset"]["state"]["world"].pop("bodies")
        with self.assertRaises(ValueError):
            m.validate_capture(doc)
        doc = self.document_copy()
        for row in doc["ticks"]:
            attack = row["state"]["players"][0]["bank"]["8"]
            if "move_manager" in attack:
                attack["move_manager"]["current_power"].pop("total_frame_count")
                break
        with self.assertRaises(ValueError):
            m.validate_capture(doc)

    def test_immutable_write_hash_roundtrip_and_tamper_rejected(self):
        m = self.module()
        doc = self.document_copy()
        with tempfile.TemporaryDirectory(prefix="ai2-combat-capture-") as folder:
            path = Path(folder) / "capture.json.gz"
            receipt = m.write_capture(doc, path)
            self.assertEqual(m.load_capture(path, expected_sha256=receipt["sha256"]), doc)
            with self.assertRaises(FileExistsError):
                m.write_capture(doc, path)
            with self.assertRaises(ValueError):
                m.load_capture(path, expected_sha256="0" * 64)

    def test_actual_contact_and_ordered_damage_events(self):
        m = self.module()
        actions = [[[0.0] * 10, [0.0] * 10] for _ in range(45)]
        for row in actions[5:]:
            row[0][7] = 1.0
        doc = m.capture_raw(FROZEN, seed=17, actions=actions, scenario={"players": [
            {"position": [3.5, 0.49], "facing": 1}, {"position": [4.3, 0.49], "facing": -1}]})
        events = [e for row in doc["ticks"] for e in row["events"]]
        self.assertTrue(any(e["kind"] == "attack_collision" and e["intersects"] for e in events))
        damage = [e for e in events if e["kind"] == "damage"]
        self.assertTrue(damage)
        self.assertTrue(all(e["after"]["damage_taken_total"] > e["before"]["damage_taken_total"] for e in damage))
        self.assertTrue(all(e["source_slot"] == 0 and e["target_slot"] == 1 for e in damage))

    def test_post_physics_weapon_drop_uses_same_tick_identity(self):
        m = self.module()
        actions = [[[0.0] * 10, [0.0] * 10] for _ in range(3)]
        actions[0][0][5] = 1.0
        doc = m.capture_raw(FROZEN, seed=17, actions=actions, scenario={"players": [{"weapon": "Spear"}, {}]})
        dropped = [e for e in doc["ticks"][0]["events"] if e["kind"] == "weapon_drop_signal"]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["tick"], 1)
        self.assertEqual(doc["ticks"][0]["state"]["players"][0]["weapon"], "Punch")
        self.assertEqual(len(doc["ticks"][0]["state"]["spawners"]), 3)

    def test_platform_body_reference_is_stably_named(self):
        m = self.module()
        with m.original_game(FROZEN, 17) as (game, _module):
            # Frozen pre_solve_oneway stores the platform BODY, not GameObject.
            game.players[0].on_platform = game.objects["platform1"].body
            self.assertEqual(m.player_snapshot(game.players[0])["on_platform"], "platform1")


if __name__ == "__main__":
    unittest.main()
