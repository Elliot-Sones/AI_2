"""Regression tests for reusing player observations at transition boundaries."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="ai2-mpl-"))

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "UTMIST-AI2-main"))

import pygame


pygame.display.init()
pygame.display.set_mode((1, 1))

env_mod = importlib.import_module("environment.environment")


def _zero_actions() -> dict[int, np.ndarray]:
    return {0: np.zeros(10), 1: np.zeros(10)}


def _new_game() -> env_mod.WarehouseBrawl:
    return env_mod.WarehouseBrawl()


def _counting_get_obs(calls: list[int]):
    real_get_obs = env_mod.Player.get_obs

    def counted_get_obs(player):
        calls.append(player.agent_id)
        return real_get_obs(player)

    return counted_get_obs


class ObservationReuseTest(unittest.TestCase):
    def test_reset_returns_observations_matching_fresh_observe_for_each_agent(self):
        game = _new_game()

        observations, info = game.reset()

        self.assertEqual(info, {})
        self.assertEqual(set(observations), {0, 1})
        for agent in game.agents:
            expected = game.observe(agent)
            with self.subTest(agent=agent):
                self.assertIsInstance(observations[agent], np.ndarray)
                self.assertEqual(observations[agent].shape, (64,))
                self.assertEqual(observations[agent].dtype, expected.dtype)
                self.assertTrue(
                    np.issubdtype(observations[agent].dtype, np.number),
                    observations[agent].dtype,
                )
                np.testing.assert_array_equal(observations[agent], expected)

    def test_reset_returns_reversed_player_perspectives_with_independent_arrays(self):
        game = _new_game()

        observations, _ = game.reset()

        self.assertIsNot(observations[0], observations[1])
        np.testing.assert_array_equal(observations[0][:32], observations[1][32:])
        np.testing.assert_array_equal(observations[0][32:], observations[1][:32])

        self.assertFalse(np.shares_memory(observations[0], observations[1]))
        original_agent_one = observations[1].copy()
        observations[0][0] += 123.0
        np.testing.assert_array_equal(observations[1], original_agent_one)

    def test_observe_reflects_direct_player_state_mutations_after_reset(self):
        game = _new_game()
        game.reset()
        before = game.observe(0)

        game.players[0].damage += 70
        after = game.observe(0)

        self.assertNotEqual(after[12], before[12])
        self.assertAlmostEqual(after[12], game.players[0].damage / 700.0)

    def test_reset_replaces_players_and_returns_observations_for_new_players(self):
        game = _new_game()
        first_players = tuple(game.players)

        observations, _ = game.reset()

        self.assertIsNot(game.players[0], first_players[0])
        self.assertIsNot(game.players[1], first_players[1])
        for agent in game.agents:
            np.testing.assert_array_equal(observations[agent], game.observe(agent))

    def test_step_returns_terminal_observations_when_player_loses_last_stock(self):
        game = _new_game()
        game.players[0].stocks = 0

        observations, rewards, terminated, truncated, info = game.step(_zero_actions())

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {})
        self.assertEqual(set(rewards), {0, 1})
        for agent in game.agents:
            self.assertEqual(observations[agent].shape, (64,))
            np.testing.assert_array_equal(observations[agent], game.observe(agent))

    def test_step_returns_truncated_observations_at_time_limit(self):
        game = _new_game()
        game.max_timesteps = 1

        observations, _rewards, terminated, truncated, info = game.step(_zero_actions())

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info, {})
        for agent in game.agents:
            self.assertEqual(observations[agent].shape, (64,))
            np.testing.assert_array_equal(observations[agent], game.observe(agent))

    def test_reset_reuses_each_real_player_observation_once(self):
        game = _new_game()
        calls: list[int] = []

        with mock.patch.object(env_mod.Player, "get_obs", _counting_get_obs(calls)):
            observations, _ = game.reset()

        self.assertEqual(len(calls), 2)
        self.assertEqual(sorted(calls), [0, 1])
        for agent in game.agents:
            self.assertEqual(observations[agent].shape, (64,))

    def test_step_reuses_each_real_player_observation_once(self):
        game = _new_game()
        calls: list[int] = []

        with mock.patch.object(env_mod.Player, "get_obs", _counting_get_obs(calls)):
            observations, _rewards, _terminated, _truncated, _info = game.step(_zero_actions())

        self.assertEqual(len(calls), 2)
        self.assertEqual(sorted(calls), [0, 1])
        for agent in game.agents:
            self.assertEqual(observations[agent].shape, (64,))


if __name__ == "__main__":
    unittest.main()
