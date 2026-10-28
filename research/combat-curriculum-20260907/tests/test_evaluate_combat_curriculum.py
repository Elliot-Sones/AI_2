import unittest
import numpy as np
from evaluate_combat_curriculum import FrameHistory, evaluation_cases, run_case


class EvaluationTests(unittest.TestCase):
    def test_frame_history_zero_pads_and_resets(self):
        frames = FrameHistory()
        out = frames.push(np.ones(64))
        np.testing.assert_array_equal(out[:192], 0)
        np.testing.assert_array_equal(out[192:], 1)
        frames.push(np.full(64, 2))
        frames.reset()
        np.testing.assert_array_equal(frames.push(np.ones(64))[:192], 0)
        with self.assertRaises(ValueError):
            frames.push(np.zeros(256))

    def test_cases_balance_modes_and_sides_with_fresh_seeds(self):
        cases = evaluation_cases('combat', count_per_mode=20)
        self.assertEqual(len(cases), 40)
        for mode in ('deterministic', 'sampled'):
            selected = [r for r in cases if r['mode'] == mode]
            self.assertEqual(sum(r['player_slot'] == 0 for r in selected), 10)
            self.assertEqual(sum(r['player_slot'] == 1 for r in selected), 10)
        self.assertFalse(set(r['seed'] for r in cases) & set(range(83001, 84101)))
        self.assertEqual({(r['player_slot'], r['direction']) for r in cases},
                         {(0, -1), (0, 1), (1, -1), (1, 1)})
        self.assertEqual({r['metric_version'] for r in cases}, {'bridge_b1_contact-v1'})

    def test_historical_reproduction_explicitly_slot_zero(self):
        cases = evaluation_cases('near', count_per_mode=20, historical=True)
        self.assertEqual({r['player_slot'] for r in cases}, {0})
        self.assertEqual(cases[0]['seed'], 84001)
        self.assertEqual(cases[0]['metric_version'], 'active_sequence_v3')

    def test_run_case_keeps_action_and_terminal_reason(self):
        class FakeEnv:
            def reset(self, **kwargs):
                self.ticks = 0
                return np.zeros(64), {}
            def step(self, action):
                self.ticks += 1
                return np.zeros(64), -0.001, False, True, {
                    'episode_diagnostics': {'success': False, 'outcome': 'timeout', 'own_stock_loss': False}}
        case = evaluation_cases('near', count_per_mode=2)[0]
        result = run_case(FakeEnv(), lambda obs, deterministic: np.zeros(10), case,
                          deadline_unix=10**12)
        self.assertEqual(result['ticks'], 1)
        self.assertEqual(result['outcome'], 'timeout')
        self.assertFalse(result['success'])
        self.assertEqual(result['mode'], 'deterministic')

    def test_trace_labels_scripted_mean_and_sampled_actions_truthfully(self):
        class FakeEnv:
            def reset(self, **kwargs):
                return np.zeros(64), {}
            def step(self, action):
                return np.zeros(64), 0., True, False, {
                    'navigation_trace': {'tick': 1},
                    'episode_diagnostics': {'success': False, 'outcome': 'timeout'}}
        for mode, explicit_source, expected in (
                ('deterministic', None, 'policy_mean'),
                ('sampled', None, 'raw_gaussian_sample'),
                ('deterministic', 'scripted_controller', 'scripted_controller')):
            case = evaluation_cases('near', count_per_mode=2)[0]
            case['mode'] = mode
            if explicit_source:
                case['action_source'] = explicit_source
            traces = []
            run_case(FakeEnv(), lambda obs, deterministic: np.zeros(10), case,
                     deadline_unix=10**12, trace_sink=traces.append)
            self.assertEqual(traces[0]['action_source'], expected)


if __name__ == '__main__':
    unittest.main()
