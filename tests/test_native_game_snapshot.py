"""Original-game comparisons, not native snapshot self-identity tests."""
import copy
import unittest

from combat_episode_capture import capture_raw
from combat_episode_parity import compare_capture, diff_records, gameplay_projection
from native_attack_tables import DEFAULT_SOURCE


class NativeGameSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from native_game_snapshot import GameParityAdapter
        cls.adapter_type = GameParityAdapter
        cls.idle = capture_raw(DEFAULT_SOURCE.parent.parent, seed=17,
                               actions=[[[0.] * 10, [0.] * 10] for _ in range(4)])

    def test_idle_original_capture_exact_reset_and_ticks(self):
        report = compare_capture(self.idle, self.adapter_type(), scope="gameplay_projection_v1")
        self.assertTrue(report["exact"], report["first_divergence"])
        self.assertEqual(report["compared_ticks"], 4)

    def test_corrupt_native_step_observation_fails_original_comparison(self):
        for slot, index in ((0, 0), (1, 63)):
            with self.subTest(slot=slot, index=index):
                class CorruptObservation(self.adapter_type):
                    def _record(self, result, consumed, *, reset=False):
                        if not reset:
                            result.obs[slot][index] = 12345.
                        return super()._record(result, consumed, reset=reset)

                report = compare_capture(self.idle, CorruptObservation(), scope="gameplay_projection_v1")
                self.assertFalse(report["exact"], "native step observation corruption was hidden")
                self.assertEqual(report["first_divergence"]["tick"], 1)
                self.assertEqual(report["first_divergence"]["differences"], [{
                    "field": f"record.outputs.obs.{slot}[{index}]", "kind": "value",
                    "expected": self.idle["ticks"][0]["outputs"]["obs"][str(slot)][index], "actual": 12345.,
                }])

    def test_original_attack_constructor_and_persistent_bank(self):
        action = [[0.] * 10, [0.] * 10]
        action[0][7] = 1.
        document = capture_raw(DEFAULT_SOURCE.parent.parent, seed=17, actions=[action] * 45)
        report = compare_capture(document, self.adapter_type(), scope="gameplay_projection_v1")
        self.assertTrue(report["exact"], report["first_divergence"])

    def test_wrong_and_unused_rng_fail_closed(self):
        for change in ("argument", "unused"):
            draws = copy.deepcopy(self.idle["reset"]["draws"])
            if change == "argument":
                draws[0]["a"] += 1
            else:
                draws.append(copy.deepcopy(draws[-1]))
            with self.subTest(change=change), self.assertRaisesRegex((ValueError, RuntimeError), "(RNG|draw|native)"):
                self.adapter_type().reset(draws, {})

    def test_drop_fixture_uses_original_order_and_weapon_metadata(self):
        action = [[0.] * 10, [0.] * 10]
        action[0][5] = 1.
        document = capture_raw(DEFAULT_SOURCE.parent.parent, seed=17, actions=[action] * 4,
                               scenario={"players": [{"weapon": "Spear"}, {}]})
        report = compare_capture(document, self.adapter_type(), scope="gameplay_projection_v1")
        self.assertTrue(report["exact"], report["first_divergence"])

    def test_spawn_animation_metadata_is_hash_verified(self):
        adapter = self.adapter_type()
        self.assertEqual(adapter.metadata.spawn_durations, {"regular": 36, "Spear": 6, "Hammer": 6})

    def test_taunt_attribute_presence_matches_original(self):
        action = [[0.] * 10, [0.] * 10]
        action[0][9] = 1.
        document = capture_raw(DEFAULT_SOURCE.parent.parent, seed=17, actions=[[[0.] * 10, [0.] * 10]] * 20 + [action] * 40)
        self.assertTrue(any("taunt_timer" in row["state"]["players"][0]["bank"]["12"] for row in document["ticks"]))
        report = compare_capture(document, self.adapter_type(), scope="gameplay_projection_v1")
        self.assertTrue(report["exact"], report["first_divergence"])

    def test_position_fixture_contact_and_damage_are_not_injected_per_tick(self):
        action = [[0.] * 10, [0.] * 10]
        action[0][7] = 1.
        scenario = {"players": [{"position": [4., .2], "state": "standing", "facing": 1},
                                {"position": [4.9, .2], "state": "standing", "facing": -1}]}
        document = capture_raw(DEFAULT_SOURCE.parent.parent, seed=17, actions=[action] * 35, scenario=scenario)
        self.assertTrue(any(event["kind"] == "damage" for row in document["ticks"] for event in row["events"]))
        report = compare_capture(document, self.adapter_type(), scope="gameplay_projection_v1")
        self.assertTrue(report["exact"], report["first_divergence"])

    def test_explicit_timeout_uses_native_truncation(self):
        document = capture_raw(DEFAULT_SOURCE.parent.parent, seed=17, actions=[[[0.] * 10, [0.] * 10]] * 4,
                               scenario={"max_timesteps": 3})
        self.assertEqual(len(document["ticks"]), 3)
        report = compare_capture(document, self.adapter_type(), scope="gameplay_projection_v1")
        self.assertTrue(report["exact"], report["first_divergence"])

    def test_ko_respawn_and_attribute_presence_match_original(self):
        document = capture_raw(DEFAULT_SOURCE.parent.parent, seed=17, actions=[[[0.] * 10, [0.] * 10]] * 100,
                               scenario={"players": [{"position": [15., 0.]}, {}]})
        self.assertTrue(any(event["kind"] == "respawn" for row in document["ticks"] for event in row["events"]))
        report = compare_capture(document, self.adapter_type(), scope="gameplay_projection_v1")
        self.assertTrue(report["exact"], report["first_divergence"])

    def test_weapon_attacks_from_both_player_slots(self):
        for weapon in ("Spear", "Hammer"):
            for slot in (0, 1):
                with self.subTest(weapon=weapon, slot=slot):
                    action = [[0.] * 10, [0.] * 10]
                    action[slot][7] = 1.
                    fixtures = [{}, {}]
                    fixtures[slot] = {"weapon": weapon}
                    document = capture_raw(DEFAULT_SOURCE.parent.parent, seed=17, actions=[action] * 40,
                                           scenario={"players": fixtures})
                    report = compare_capture(document, self.adapter_type(), scope="gameplay_projection_v1")
                    self.assertTrue(report["exact"], report["first_divergence"])

    def test_native_bank_and_world_mutation_fail_against_original(self):
        for field in ("bank", "world", "timeout"):
            adapter = self.adapter_type()
            adapter.reset(self.idle["reset"]["draws"], {})
            expected = gameplay_projection(self.idle["reset"])["state"]
            if field == "bank":
                adapter.scene.players[1].ms.bank[6].value[5] += 1
            elif field == "world":
                adapter.scene.world.bodies[adapter.scene.players[0].body].v.x = .125
            else:
                adapter.state.max_timesteps = 17
            differences = diff_records(expected, adapter.snapshot())
            self.assertTrue(differences, field)

    def test_snapshot_negative_controls_remain_visible(self):
        expected = gameplay_projection(self.idle["ticks"][0])
        for mutate in (
            lambda row: row["state"]["players"][0]["bank"]["6"].update(jumps_left=9),
            lambda row: row["state"]["players"][0]["hurtbox"].update(width=.99),
            lambda row: row["outputs"]["rewards"].update({"0": 1.}),
            lambda row: row["events"].append({"kind": 3}),
        ):
            actual = copy.deepcopy(expected)
            mutate(actual)
            self.assertTrue(diff_records(expected, actual))


if __name__ == "__main__":
    unittest.main()
