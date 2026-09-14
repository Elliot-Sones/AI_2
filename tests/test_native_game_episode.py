"""Raw full-game integration tests: no navigation wrapper or Python game loop."""

import unittest
import ctypes as C
import numpy as np
from native_game_episode import GameReference, Result


class GameEpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = GameReference()

    def setUp(self):
        self.ref.cfg.time_limit_steps = 9000
        self.ref.reset_generated(17)

    def test_reset_both_observers_and_no_target_wrapper(self):
        out = self.ref.observe()
        self.assertEqual(out.shape, (2, 64))
        self.assertEqual(out.dtype, np.float64)
        np.testing.assert_array_equal(out[0, :32], out[1, 32:])
        np.testing.assert_array_equal(out[0, 32:], out[1, :32])
        self.assertEqual(self.ref.scene.wrap.target_valid, 0)
        self.assertEqual(self.ref.scene.world.bodies[4].p.y, 0)

    def test_second_player_actions_are_not_discarded(self):
        actions = np.zeros((2, 10))
        actions[1, 3] = 1
        for _ in range(15):
            self.ref.step_generated(actions)
        self.assertGreater(self.ref.scene.world.bodies[4].v.x, 0)
        self.assertEqual(self.ref.scene.players[1].input_state[3], 1)

    def test_combat_buttons_are_not_masked(self):
        actions = np.zeros((2, 10))
        actions[:, 7] = 1
        for _ in range(5):
            self.ref.step_generated(actions)
        self.assertGreater(sum(e.attack_starts for e in self.ref.combat), 0)
        self.assertEqual(self.ref.scene.players[0].input_state[7], 1)
        self.assertEqual(self.ref.scene.players[1].input_state[7], 1)

    def test_last_stock_ko_terminates_on_following_tick(self):
        self.ref.place_player(0, 15, 0)
        self.ref.scene.players[0].ms.stocks = 1
        first = self.ref.step_generated(np.zeros((2, 10)))
        self.assertFalse(first.terminated)
        self.assertEqual(self.ref.scene.players[0].ms.stocks, 0)
        second = self.ref.step_generated(np.zeros((2, 10)))
        self.assertTrue(second.terminated)
        self.assertEqual(second.win_mask, 2)

    def test_invalid_actions_fail_before_mutating(self):
        before = bytes(self.ref.state)
        for actions in (np.zeros((10,)), np.full((2, 10), np.nan)):
            with self.assertRaises(ValueError):
                self.ref.step_generated(actions)
            self.assertEqual(bytes(self.ref.state), before)

    def test_invalid_raw_dt_and_horizon_rejected_before_state_or_rng_mutation(self):
        for field, value in (('dt', float('inf')), ('dt', float('nan')), ('dt', 0.0),
                             ('dt', -1.0), ('time_limit_steps', 0), ('time_limit_steps', 9001)):
            with self.subTest(field=field, value=value):
                previous = getattr(self.ref.cfg, field)
                setattr(self.ref.cfg, field, value)
                state, rng = bytes(self.ref.state), bytes(self.ref.rng)
                out = Result()
                code = self.ref.lib.ai2g_reset_export(C.byref(self.ref.state), C.byref(self.ref.cfg),
                                                     C.byref(self.ref.rng), 48, C.byref(out))
                setattr(self.ref.cfg, field, previous)
                self.assertEqual(code, 2)
                self.assertEqual(bytes(self.ref.state), state)
                self.assertEqual(bytes(self.ref.rng), rng)

    def test_platform_table_covers_full_match(self):
        self.assertGreaterEqual(self.ref.table.count, 9000)

    def test_zero_movement_reward_preserves_accumulator_positive_zero(self):
        result = self.ref.step_generated(np.zeros((2, 10)))
        for value in result.reward:
            self.assertEqual(value, 0.0)
            self.assertFalse(np.signbit(value))

    def test_explicit_position_fixture_reindexes_and_syncs_hurtbox(self):
        world = self.ref.scene.world
        before_stamp = world.dyn_tree.stamp
        self.ref.fixture_player_position(0, 0.0, -1.0)
        player = self.ref.scene.players[0]
        self.assertEqual(self.ref.combat[0].hurtbox_x, 0.0)
        self.assertEqual(self.ref.combat[0].hurtbox_y, -1.0)
        self.assertEqual(world.dyn_tree.stamp, before_stamp + 1)
        # ReindexObject advances the tree stamp even when the fat BB contains it.
        self.ref.fixture_player_position(0, 0.0, -1.0)
        self.assertEqual(world.dyn_tree.stamp, before_stamp + 2)
        self.assertEqual(player.ms.prev_x, 5.0)

    def test_declared_short_timeout_fixture(self):
        self.ref.cfg.time_limit_steps = 3
        self.ref.reset_generated(17)
        actions = np.zeros((2, 10))
        self.assertFalse(self.ref.step_generated(actions).truncated)
        self.assertFalse(self.ref.step_generated(actions).truncated)
        self.assertTrue(self.ref.step_generated(actions).truncated)

    def test_full_length_match_and_raw_post_terminal_step(self):
        actions = np.zeros((2, 10))
        for tick in range(1, 9001):
            result = self.ref.step_generated(actions)
            self.assertEqual(bool(result.truncated), tick == 9000)
        self.assertEqual(self.ref.scene.steps, 9000)
        self.assertTrue(self.ref.step_generated(actions).truncated)
        self.assertEqual(self.ref.scene.steps, 9001)


if __name__ == "__main__":
    unittest.main()
