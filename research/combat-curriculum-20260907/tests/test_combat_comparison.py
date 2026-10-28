import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from combat_comparison import (
    build_cases, classify_outcome, relative_stats, update_trace,
    write_json, run_episode,
)


class CombatComparisonTests(unittest.TestCase):
    def test_paired_schedule(self):
        cases = build_cases(['medium145', 'large145'], ['phase1', 'phase2', 'phase3', 'phase4'], range(92001, 92011))
        self.assertEqual(len(cases), 320)
        self.assertEqual(len({case['id'] for case in cases}), 320)
        for candidate in ['medium145', 'large145']:
            subset = [case for case in cases if case['candidate'] == candidate]
            self.assertEqual(len(subset), 160)
            self.assertEqual({case['candidate_slot'] for case in subset}, {0, 1})
            self.assertEqual({case['deterministic'] for case in subset}, {False, True})

    def test_invalid_schedule(self):
        for candidates, opponents, seeds in [([], ['b'], [1]), (['a', 'a'], ['b'], [1]), (['../a'], ['b'], [1]), (['a'], ['b'], [1, 1]), (['a'], ['b'], [])]:
            with self.subTest(candidates=candidates, seeds=seeds), self.assertRaises(ValueError):
                build_cases(candidates, opponents, seeds)

    def test_timeout_is_not_win_even_with_stock_lead(self):
        self.assertEqual(classify_outcome([3, 1], False, True, 0), 'timeout')

    def test_elimination_relative_to_slot(self):
        self.assertEqual(classify_outcome([3, 0], True, False, 0), 'win')
        self.assertEqual(classify_outcome([3, 0], True, False, 1), 'loss')
        self.assertEqual(classify_outcome([0, 0], True, True, 0), 'draw')

    def test_post_physics_final_stock_loss_precedes_late_game_flag(self):
        self.assertEqual(classify_outcome([3, 0], False, True, 0), 'win')
        self.assertEqual(classify_outcome([0, 3], False, False, 0), 'loss')
        case = build_cases(['a'], ['b'], [92001])[0]
        class FinalStockEnv(FakeEnv):
            def get_stats(self, slot):
                from types import SimpleNamespace
                return SimpleNamespace(lives_left=0 if self.steps else 1, damage_done=0, damage_taken=0)
        result = run_episode(FinalStockEnv(), [FakePolicy(), FakePolicy()], case, max_steps=3, deadline_unix=10**12, seed_fn=lambda seed: None)
        self.assertEqual(result['steps'], 1)
        self.assertEqual(result['outcome'], 'draw')
        self.assertFalse(result['terminated'])
        self.assertEqual(result['end_reason'], 'post_step_stock_elimination')
        self.assertEqual(result['stock_loss_events'], [{'step': 1, 'slot': 0, 'before': 1, 'after': 0}, {'step': 1, 'slot': 1, 'before': 1, 'after': 0}])

    def test_invalid_terminal_rejected(self):
        for stocks, term, trunc, slot in [([3, 2], True, False, 0), ([3, 2], False, False, 0), ([3, -1], True, False, 0), ([3, 0], True, False, 2)]:
            with self.subTest(stocks=stocks, slot=slot), self.assertRaises(ValueError):
                classify_outcome(stocks, term, trunc, slot)

    def test_relative_stats(self):
        players = [{'stocks': 2, 'damage_done': 20, 'damage_taken': 60}, {'stocks': 1, 'damage_done': 60, 'damage_taken': 20}]
        self.assertEqual(relative_stats(players, 1), {'candidate': players[1], 'opponent': players[0]})

    def test_trace_is_order_independent_but_sensitive_to_state(self):
        first, second, third = (hashlib.sha256() for _ in range(3))
        update_trace(first, {'step': 1, 'actions': [0, 1]})
        update_trace(second, {'actions': [0, 1], 'step': 1})
        update_trace(third, {'step': 1, 'actions': [1, 0]})
        self.assertEqual(first.hexdigest(), second.hexdigest())
        self.assertNotEqual(first.hexdigest(), third.hexdigest())

    def test_json_rejects_nonfinite_and_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'case.json'
            with self.assertRaises(ValueError):
                write_json(target, {'bad': float('nan')})
            self.assertFalse(target.exists())
            write_json(target, {'ok': True})
            self.assertEqual(json.loads(target.read_text()), {'ok': True})
            with self.assertRaises(FileExistsError):
                write_json(target, {'ok': False})

    def test_episode_resets_both_policies_and_replays_exactly(self):
        case = build_cases(['a'], ['b'], [92001])[0]
        policies = [FakePolicy(), FakePolicy()]
        first = run_episode(FakeEnv(), policies, case, max_steps=3, deadline_unix=10**12, seed_fn=lambda seed: None)
        second = run_episode(FakeEnv(), policies, case, max_steps=3, deadline_unix=10**12, seed_fn=lambda seed: None)
        self.assertEqual(first['trace_sha256'], second['trace_sha256'])
        self.assertEqual(first['outcome'], 'timeout')
        self.assertEqual(first['steps'], 3)
        self.assertEqual([p.resets for p in policies], [2, 2])
        self.assertEqual(first['stock_losses'], [0, 0])
        self.assertEqual(first['action_active_counts'][0], [0]*10)

    def test_expired_deadline_rejected_before_any_step(self):
        case = build_cases(['a'], ['b'], [1])[0]
        env = FakeEnv()
        with self.assertRaises(TimeoutError):
            run_episode(env, [FakePolicy(), FakePolicy()], case, max_steps=3, deadline_unix=1, seed_fn=lambda seed: None)
        self.assertEqual(env.steps, 0)


class FakePolicy:
    def __init__(self):
        self.resets = 0

    def get_env_info(self, env):
        self.resets += 1

    def predict(self, obs):
        return np.zeros(10, dtype=np.float32)


class FakeEnv:
    fps = 30

    def __init__(self):
        self.steps = 0

    def reset(self, seed=None):
        self.steps = 0
        return {0: np.zeros(64), 1: np.ones(64)}, {}

    def get_stats(self, slot):
        from types import SimpleNamespace
        return SimpleNamespace(lives_left=3, damage_done=0, damage_taken=0)

    def step(self, actions):
        self.steps += 1
        return {0: np.zeros(64), 1: np.ones(64)}, {0: 0, 1: 0}, False, self.steps >= self.max_timesteps, {}


if __name__ == '__main__':
    unittest.main()
