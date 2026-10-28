import importlib
import importlib.util
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

import numpy as np


FROZEN = Path(os.environ.get('AI2_FROZEN_ROOT', '/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected'))


class NavigationTests(unittest.TestCase):
    def setUp(self):
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.assertIsNotNone(importlib.util.find_spec('curriculum_navigation'), 'historical navigation adapter is missing')
        self.module = importlib.import_module('curriculum_navigation')

    def make_env(self, **kwargs):
        env = self.module.make_navigation_env(FROZEN, seed=84001, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_historical_scenarios_use_recorded_evaluation_config(self):
        for variant, distances, first_seed in [('near', (1.5, 3.0), 84001), ('wide', (2.0, 6.0), 83001)]:
            with self.subTest(variant=variant):
                env = self.make_env(variant=variant)
                nav = env.navigation_wrapper
                self.assertEqual((nav.min_dist, nav.max_dist), distances)
                self.assertEqual((nav.time_limit, nav.settle_steps, nav.post_goal_required_steps, nav.max_targets), (600, 9, 15, 3))
                self.assertEqual(nav.fall_penalty, -3.0)
                scenarios = self.module.historical_scenarios(variant, count=20)
                self.assertEqual([s['seed'] for s in scenarios], list(range(first_seed, first_seed + 20)))
                self.assertTrue(all(s['player_slot'] == 0 for s in scenarios))

    def test_reset_observation_mask_and_step_preserve_caller_action(self):
        env = self.make_env()
        obs, info = env.reset(seed=84001)
        self.assertEqual(obs.shape, (64,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(info['curriculum_task'], 'navigation')
        self.assertEqual(np.flatnonzero(info['allowed_control_mask']).tolist(), [0, 1, 2, 3, 4, 6])
        self.assertTrue(all(type(x) is bool for x in info['allowed_control_mask']))
        action = np.full(10, 2.0)
        _, _, _, _, info = env.step(action)
        np.testing.assert_array_equal(action, np.full(10, 2.0))
        self.assertEqual(info['navigation_trace']['executed_action'], [1., 1., 1., 1., 1., 0., 1., 0., 0., 0.])
        self.assertEqual(info['navigation_trace']['raw_action'], [2.] * 10)
        self.assertEqual(info['navigation_trace']['opponent_action'], [0.] * 10)
        json.dumps(info, allow_nan=False)

    def test_slot_zero_matches_original_factory_trajectory(self):
        env = self.make_env()
        train = self.module.load_frozen_train(FROZEN)
        params = self.module.navigation_params('near')
        with self.module._working_directory(FROZEN / 'UTMIST-AI2-main/environment'):
            original = train.make_nav_env(train.PhaseConfig(params, '0a'), train.get_resolution('LOW'), params)
        self.addCleanup(original.close)
        random.seed(84001)
        np.random.seed(84001)
        with self.module._working_directory(FROZEN / 'UTMIST-AI2-main/environment'):
            expected, _ = original.reset(seed=84001)
        actual, _ = env.reset(seed=84001)
        np.testing.assert_array_equal(actual, expected)
        for tick in range(90):
            action = np.zeros(10, dtype=np.float32)
            action[1 if tick < 20 else 3] = float(tick % 3 == 0)
            exp = original.step(action)
            got = env.step(action)
            np.testing.assert_array_equal(got[0], exp[0])
            self.assertEqual(got[1:4], exp[1:4])
            self.assertEqual(got[4]['nav_success'], exp[4]['nav_success'])
            if got[2] or got[3]:
                break

    def test_slot_one_observation_actions_and_fall_refer_to_actual_player_one(self):
        env = self.make_env(player_slot=1)
        obs, _ = env.reset(seed=84001)
        np.testing.assert_array_equal(obs, env.raw_env.observe(1).astype(np.float32))
        self.assertIs(env.navigation_wrapper._get_player(), env.raw_env.players[1])
        action = np.zeros(10)
        action[3] = 1
        _, _, _, _, info = env.step(action)
        np.testing.assert_array_equal(env.raw_env.cur_action[1], action)
        np.testing.assert_array_equal(env.raw_env.cur_action[0], np.zeros(10))
        env.raw_env.players[1].stocks -= 1
        _, _, terminated, _, info = env.step(np.zeros(10))
        self.assertTrue(terminated)
        self.assertEqual(info['episode_diagnostics']['outcome'], 'fall')
        self.assertTrue(info['episode_diagnostics']['fall'])
        self.assertFalse(info['episode_diagnostics']['success'])

    def test_seeded_resets_and_scenario_are_reproducible(self):
        env = self.make_env()
        first, _ = env.reset(seed=84001)
        for _ in range(3):
            env.step(np.zeros(10))
        replay, info = env.reset(options={'scenario': {'id': 'retention-near-84001', 'seed': 84001, 'player_slot': 0}})
        np.testing.assert_array_equal(first, replay)
        self.assertEqual(info['scenario_id'], 'retention-near-84001')
        self.assertEqual(info['seed'], 84001)

    def test_body_overlap_is_versioned_and_does_not_change_historical_settling(self):
        import pymunk
        from types import SimpleNamespace
        def player_at(x, y):
            body = pymunk.Body(1, float('inf'))
            body.position = x, y
            shape = pymunk.Poly.create_box(body, (0.87, 1.0))
            return SimpleNamespace(body=body, shape=shape, is_on_floor=lambda: True, on_platform=None)
        first, second = player_at(0, 0), player_at(.8, .7)
        self.assertTrue(self.module.bodies_overlap(first, second))
        original = self.make_env()
        self.assertFalse(original.navigation_wrapper._is_settled_at_target(first, np.hypot(.8, .7)))
        second.body.position = .9, .7
        self.assertFalse(self.module.bodies_overlap(first, second))
        overlap = self.make_env(metric_version=self.module.BODY_OVERLAP_METRIC)
        _, info = overlap.reset(seed=84001)
        self.assertEqual(info['metric_version'], 'body_overlap_sequence_v1')
        self.assertEqual(original.metric_version, 'active_sequence_v3')

    def test_idle_times_out_and_retains_terminal_diagnostics(self):
        env = self.make_env()
        env.reset(seed=84001)
        for _ in range(600):
            _, _, terminated, truncated, info = env.step(np.zeros(10))
            if terminated or truncated:
                break
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        diag = info['episode_diagnostics']
        self.assertEqual(diag['steps'], 600)
        self.assertEqual(diag['outcome'], 'timeout')
        self.assertFalse(diag['success'])
        self.assertFalse(diag['fall'])
        self.assertEqual(diag['metric_version'], 'active_sequence_v3')

    def test_invalid_inputs_fail_closed(self):
        for kwargs in ({'variant': 'other'}, {'player_slot': 2}, {'player_slot': 1.0}, {'metric_version': 'unknown'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_env(**kwargs)
        env = self.make_env()
        env.reset()
        for action in (np.zeros(9), np.full(10, np.nan)):
            with self.assertRaises(ValueError):
                env.step(action)

    def test_controller_completes_three_target_sequence_and_matches_frozen_factory(self):
        env = self.make_env()
        train = self.module.load_frozen_train(FROZEN)
        params = self.module.navigation_params('near')
        with self.module._working_directory(FROZEN / 'UTMIST-AI2-main/environment'):
            original = train.make_nav_env(train.PhaseConfig(params, '0a'), train.get_resolution('LOW'), params)
            self.addCleanup(original.close)
            random.seed(84001)
            np.random.seed(84001)
            original.reset(seed=84001)
        env.reset(seed=84001)
        for _ in range(600):
            action = np.zeros(10, dtype=np.float32)
            player = env.navigation_wrapper._get_player()
            dx = env.navigation_wrapper.target_position[0] - player.body.position.x
            dy = env.navigation_wrapper.target_position[1] - player.body.position.y
            if abs(dx) > .45:
                action[3 if dx > 0 else 1] = 1
            if dy < -1 and abs(dx) < 3:
                action[4] = 1
            expected = original.step(action)
            actual = env.step(action)
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1:4], expected[1:4])
            if actual[2] or actual[3]:
                break
        diag = actual[4]['episode_diagnostics']
        self.assertTrue(diag['success'])
        self.assertEqual(diag['targets_reached'], 3)
        self.assertEqual(diag['post_goal_steps'], 15)

    def test_game_resets_and_steps_do_not_consume_callers_rng(self):
        env = self.make_env()
        random.seed(42)
        np.random.seed(42)
        expected = (random.random(), np.random.random())
        random.seed(42)
        np.random.seed(42)
        env.reset(seed=84001)
        env.step(np.zeros(10))
        self.assertEqual((random.random(), np.random.random()), expected)

    def test_subprocess_factory_preserves_zero_padded_four_frame_contract(self):
        code = '''
from functools import partial
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from curriculum_navigation import make_navigation_env
if __name__ == '__main__':
    env = VecFrameStack(SubprocVecEnv([
        partial(make_navigation_env, FROZEN_ROOT, 84001, player_slot=slot)
        for slot in (0, 1)
    ], start_method='spawn'), n_stack=4)
    try:
        obs = env.reset()
        assert obs.shape == (2, 256)
        assert obs.dtype == np.float32
        assert np.all(obs[:, :192] == 0)
        next_obs, _, _, infos = env.step(np.zeros((2, 10)))
        np.testing.assert_array_equal(next_obs[:, 128:192], obs[:, 192:])
        assert [info['player_slot'] for info in infos] == [0, 1]
    finally:
        env.close()
'''.replace('FROZEN_ROOT', repr(str(FROZEN)))
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
