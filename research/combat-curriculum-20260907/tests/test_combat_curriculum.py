import importlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np


FROZEN = Path(os.environ.get('AI2_FROZEN_ROOT', '/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected'))


class B1GameTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('combat_curriculum'), 'B1 original-game environment is missing')
        self.module = importlib.import_module('combat_curriculum')
        self.env = self.module.make_b1_env(FROZEN, seed=17)
        self.addCleanup(self.env.close)

    def test_reset_is_raw64_same_floor_and_exact_mask(self):
        obs, info = self.env.reset(seed=17)
        self.assertEqual(obs.shape, (64,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(tuple(self.module.B1_ACTIVE_INDICES), (0, 1, 2, 3, 4, 6, 7))
        self.assertEqual(info['curriculum_task'], 'combat')
        trace = info['trace']
        self.assertTrue(all(p['grounded'] for p in trace['post_players']))
        self.assertGreater(abs(obs[0]-obs[32]), self.env.measured_light_reach)
        obs, reward, term, trunc, info = self.env.step(np.full(10, 2.0))
        self.assertEqual(info['trace']['executed_action'], [1., 1., 1., 1., 1., 0., 1., 1., 0., 0.])
        self.assertEqual(info['trace']['opponent_action'], [0.]*10)
        json.dumps(info, allow_nan=False)

    def test_half_threshold_does_not_start_attack(self):
        self.env.reset(seed=17)
        _, _, _, _, info = self.env.step(np.full(10, .5))
        self.assertFalse(any(e['type'] in ('attack_press', 'attack_started') for e in info['trace']['events']))

    def test_observation_box_matches_checkpoint_and_traces_preserve_observations(self):
        raw = self.env.raw_env.observation_space
        np.testing.assert_array_equal(self.env.observation_space.low, raw.low.astype(np.float32))
        np.testing.assert_array_equal(self.env.observation_space.high, raw.high.astype(np.float32))
        obs, _ = self.env.reset(seed=17)
        next_obs, _, _, _, info = self.env.step(np.zeros(10))
        np.testing.assert_array_equal(info['trace']['observation_before'], obs)
        np.testing.assert_array_equal(info['trace']['observation_after'], next_obs)
        self.assertEqual(info['trace']['reengagement_metric'], 'horizontal_gap_and_learner_progress_v1')
        self.assertTrue(all('stun_time_stored' in p for p in info['trace']['post_players']))

    def test_finite_task_deadline_is_terminal_without_ppo_time_limit_bootstrap(self):
        self.env.reset(seed=17)
        self.env.tick = 299
        _, reward, terminated, truncated, info = self.env.step(np.zeros(10))
        self.assertTrue(terminated, 'B1 task failure must not bootstrap a terminal value')
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, -1.001)
        self.assertEqual(info['episode_diagnostics']['outcome'], 'timeout')

    def test_neutral_multihit_matches_unwrapped_game_and_gets_one_credit(self):
        reference = self.module.make_b1_env(FROZEN, seed=17)
        self.addCleanup(reference.close)
        for env in (self.env, reference):
            env.reset(seed=17)
            env._place_fresh_player(0, 4.1, env.floor_y, 1)
            env._place_fresh_player(1, 4.9, env.floor_y, -1)
        events, target_x = [], []
        for tick in range(40):
            action = np.zeros(10)
            action[7] = int(tick == 0)
            _, _, _, _, info = self.env.step(action)
            reference.raw_env.step({0: action, 1: np.zeros(10)})
            for p, q in zip(self.env.raw_env.players, reference.raw_env.players):
                np.testing.assert_array_equal(p.body.position, q.body.position)
                np.testing.assert_array_equal(p.body.velocity, q.body.velocity)
                self.assertEqual(p.damage, q.damage)
                self.assertEqual(type(p.state), type(q.state))
            events.extend(info['trace']['events'])
            target_x.append(self.env.raw_env.players[1].body.position.x)
        damage = [e for e in events if e['type'] == 'damage']
        self.assertGreaterEqual(len(damage), 4)
        self.assertEqual(len({e['attack_id'] for e in damage}), 1)
        self.assertEqual(len([e for e in events if e['type'] == 'hit_credit']), 1)
        self.assertEqual(len([e for e in events if e['type'] == 'relocation']), 0)
        self.assertGreater(max(target_x)-min(target_x), .25)
        self.assertTrue(all(e['source'] == 'original_game.Power.do_power' for e in damage))

    def test_original_light_geometry_is_measured_and_brackets_spawn_gap(self):
        self.assertTrue(hasattr(self.module, 'qualify_light_geometry'), 'Missing original-game geometry qualification')
        report = self.module.qualify_light_geometry(FROZEN)
        self.assertEqual(report['source_sha256'], self.module.ORIGINAL_GAME_SHA256)
        self.assertLessEqual(report['max_observed_hit_gap'], self.env.measured_light_reach)
        self.assertGreater(report['max_observed_hit_gap'], 1.5)
        self.assertEqual(set(report['directions']), {-1, 1})
        self.assertEqual(set(report['player_slots']), {0, 1})
        self.assertGreaterEqual(report['fixtures'], 100)
        self.assertGreaterEqual(report['max_attack_ticks']['NLIGHT'], 31)
        self.assertTrue('airborne' in report, 'Enabled aerial light controls need a measured reach panel')
        self.assertGreater(report['airborne']['max_observed_hit_gap'], self.env.measured_light_reach)
        self.assertGreaterEqual(report['airborne']['fixtures'], 100)
        self.assertFalse(report['qualified'])

    def test_stock_loss_suppresses_same_tick_real_hit_and_timeout(self):
        self.env.reset(seed=17)
        self.env._place_fresh_player(0, 4.1, self.env.floor_y, 1)
        target = self.env._place_fresh_player(1, 4.9, self.env.floor_y, -1)
        original_damage = target.apply_damage
        def lose_stock_on_real_hit(*args, **kwargs):
            original_damage(*args, **kwargs)
            self.env.raw_env.players[0].stocks -= 1
        target.apply_damage = lose_stock_on_real_hit
        self.env.tick = 296
        for tick in range(4):
            action = np.zeros(10)
            action[7] = int(tick == 0)
            _, reward, term, trunc, info = self.env.step(action)
        self.assertEqual(info['trace']['tick'], 300)
        self.assertEqual(info['episode_diagnostics']['outcome'], 'own_stock_loss')
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertAlmostEqual(reward, -5.001)
        self.assertEqual(info['trace']['reward_terms']['contact'], 0.)
        self.assertEqual(info['trace']['reward_terms']['completion'], 0.)
        self.assertEqual(info['trace']['reward_terms']['timeout'], 0.)

    def test_relocation_uses_new_clean_body_only_after_resolved_interaction(self):
        self.env.reset(seed=17)
        self.env._place_fresh_player(0, 2.45, self.env.floor_y, 1)
        target = self.env._place_fresh_player(1, 3.3, self.env.floor_y, -1)
        old_shape, old_body = target.shape, target.body
        for tick in range(150):
            action = np.zeros(10)
            action[7] = int(tick == 0)
            _, _, term, trunc, info = self.env.step(action)
            events = info['trace']['events']
            if any(e['type'] == 'relocation' for e in events):
                replacement = self.env.raw_env.players[1]
                self.assertIsNot(replacement.body, old_body)
                self.assertNotIn(old_shape, self.env.raw_env.space.shapes)
                self.assertNotIn(old_body, self.env.raw_env.space.bodies)
                self.assertIsNone(replacement.on_platform)
                self.assertEqual(tuple(replacement.body.velocity), (0., 0.))
                self.assertEqual(replacement.damage, 0.)
                self.assertEqual(replacement.damage_velocity, (0, 0))
                self.assertEqual(replacement.target_vel, (0, 0))
                self.assertEqual(type(replacement.state).__name__, 'StandingState')
                self.assertIs(replacement.opponent, self.env.raw_env.players[0])
                relocation = next(e for e in events if e['type'] == 'relocation')
                self.assertGreaterEqual(relocation['gap'], self.env.measured_light_reach+1.)
                self.assertTrue(self.env._resolved())
                break
            self.assertIs(self.env.raw_env.players[1].body, old_body)
            self.assertFalse(term or trunc)
        else:
            self.fail('Competent controller never produced relocation')

    def run_controller(self, controller, **scenario):
        self.env.reset(seed=17, options={'scenario': scenario})
        traces = []
        for _ in range(300):
            _, _, term, trunc, info = self.env.step(controller(self.env))
            traces.append(info['trace'])
            if term or trunc:
                return info, traces
        self.fail('B1 exceeded 300 ticks')

    def test_scripted_sequence_is_rejected_by_training_qualification(self):
        self.assertTrue(hasattr(self.module, 'B1_TRAINING_QUALIFIED'), 'Missing safety qualification gate')
        self.assertFalse(self.module.B1_TRAINING_QUALIFIED)
        for slot in (0, 1):
            for direction in (-1, 1):
                with self.subTest(slot=slot, direction=direction):
                    info, traces = self.run_controller(self.module.scripted_b1_action, player_slot=slot, direction=direction)
                    self.assertEqual(info['episode_diagnostics']['outcome'], 'unsafe_floor_support')
                    self.assertFalse(info['episode_diagnostics']['success'])
                    events = [e for t in traces for e in t['events']]
                    hits = [e for e in events if e['type'] == 'hit_credit']
                    self.assertEqual(len(hits), 1)
                    self.assertEqual(len({e['attack_id'] for e in hits}), 1)
                    self.assertEqual(len([e for e in events if e['type'] == 'relocation']), 0)
                    self.assertLessEqual(info['episode_diagnostics']['positive_reward'], 4.)
                    learner_clearance = [min(t['post_players'][slot]['x']-.435-2.,
                                             7.-t['post_players'][slot]['x']-.435) for t in traces]
                    self.assertLess(min(learner_clearance), 0., 'Update the versioned prototype safety receipt if geometry changes')
                    self.assertFalse(info['episode_diagnostics']['training_qualified'])
                    self.assertEqual(info['episode_diagnostics']['qualification_blocker'], self.module.B1_QUALIFICATION_BLOCKER)

    def test_seeded_game_stream_is_private_and_reproducible_when_interleaved(self):
        random.seed(731)
        np.random.seed(913)
        python_before, numpy_before = random.getstate(), np.random.get_state()
        other = self.module.make_b1_env(FROZEN, seed=17)
        self.addCleanup(other.close)
        self.assertEqual(random.getstate(), python_before, 'Game construction consumed caller Python RNG')
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        self.env.reset(seed=17)
        other.reset(seed=17)
        for _ in range(100):
            action = self.module.scripted_b1_action(self.env)
            python_before, numpy_before = random.getstate(), np.random.get_state()
            obs1, reward1, done1, trunc1, info1 = self.env.step(action)
            self.assertEqual(random.getstate(), python_before, 'Game step consumed caller Python RNG')
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
            random.random()
            np.random.rand()
            obs2, reward2, done2, trunc2, info2 = other.step(action)
            np.testing.assert_array_equal(obs1, obs2)
            self.assertEqual((reward1, done1, trunc1), (reward2, done2, trunc2))
            self.assertEqual(self.env.raw_env.players[self.env.player_slot].states['attack'].__dict__.get('seed'),
                             other.raw_env.players[other.player_slot].states['attack'].__dict__.get('seed'))
            if done1 or trunc1:
                break

    def test_idle_held_light_and_walkoff_cannot_pass(self):
        for buttons in ((), (7,), (3,), (9,), (2, 7), (0, 7), (1, 3, 7),
                        (4, 7), (4, 2, 7), (1, 7), (3, 7), tuple(range(10))):
            def controller(env):
                action = np.zeros(10)
                action[list(buttons)] = 1
                return action
            with self.subTest(buttons=buttons):
                info, _ = self.run_controller(controller)
                self.assertNotEqual(info['episode_diagnostics']['outcome'], 'success')

    def test_held_light_can_earn_first_contact_but_fails_reengagement(self):
        self.env.reset(seed=17)
        self.env._place_fresh_player(0, 2.5, self.env.floor_y, 1)
        self.env._place_fresh_player(1, 3.3, self.env.floor_y, -1)
        action = np.zeros(10)
        action[7] = 1
        events = []
        for _ in range(300):
            _, _, term, trunc, info = self.env.step(action)
            events.extend(info['trace']['events'])
            if term or trunc:
                break
        self.assertEqual(info['episode_diagnostics']['positive_reward'], 1.)
        # Repeated live attacks may keep the first interaction unresolved. In
        # that case no relocation is legal and the finite total clock fails it.
        self.assertIn(info['episode_diagnostics']['outcome'], ('response_deadline', 'timeout'))
        self.assertLessEqual(len([e for e in events if e['type'] == 'relocation']), 1)

    def test_response_and_second_hit_deadlines_fail_at_exact_tick(self):
        for pursue in (False, True):
            with self.subTest(pursue=pursue):
                self.env.reset(seed=17)
                self.env._place_fresh_player(0, 2.45, self.env.floor_y, 1)
                self.env._place_fresh_player(1, 3.3, self.env.floor_y, -1)
                tick = 0
                while self.env.relocation_tick is None:
                    action = np.zeros(10)
                    action[7] = int(tick == 0)
                    tick += 1
                    _, _, term, trunc, info = self.env.step(action)
                    self.assertFalse(term or trunc)
                boundary = self.env.relocation_tick
                while self.env.outcome is None:
                    action = np.zeros(10)
                    if pursue and self.env.pursuit_tick is None:
                        p, q = self.env.raw_env.players
                        action[3 if q.body.position.x > p.body.position.x else 1] = 1
                    _, _, term, trunc, info = self.env.step(action)
                self.assertEqual(self.env.tick-boundary, 90 if pursue else 60)
                self.assertEqual(self.env.outcome, 'second_hit_deadline' if pursue else 'response_deadline')
                self.assertTrue(term)
                self.assertFalse(trunc)

    def test_vec_env_does_not_mark_task_failure_for_time_limit_bootstrap(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        vector = DummyVecEnv([lambda: self.env])
        self.addCleanup(vector.close)
        vector.reset()
        self.env.tick = 299
        _, reward, done, infos = vector.step(np.zeros((1, 10), dtype=np.float32))
        self.assertTrue(done[0])
        self.assertFalse(infos[0]['TimeLimit.truncated'])
        self.assertAlmostEqual(float(reward[0]), -1.001, places=5)
        self.assertEqual(infos[0]['terminal_observation'].shape, (64,))


class B1ContactTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('combat_curriculum')
        self.assertTrue(hasattr(self.module, 'make_b1_contact_env'), 'Approved contact lesson is missing')
        self.env = self.module.make_b1_contact_env(FROZEN, seed=17)
        self.addCleanup(self.env.close)

    def run_controller(self, controller, *, seed=17, slot=0, direction=1):
        observation, reset_info = self.env.reset(seed=seed, options={'scenario': {
            'player_slot': slot, 'direction': direction, 'id': f'contact-{seed}-{slot}-{direction}'}})
        traces = []
        for _ in range(300):
            next_observation, reward, terminated, truncated, info = self.env.step(controller(self.env))
            trace = info['trace']
            np.testing.assert_array_equal(trace['observation_before'], observation)
            np.testing.assert_array_equal(trace['observation_after'], next_observation)
            self.assertAlmostEqual(reward, sum(trace['reward_terms'].values()))
            if traces:
                self.assertEqual(trace['pre_players'], traces[-1]['post_players'])
            traces.append(trace)
            observation = next_observation
            if terminated or truncated:
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                return reset_info, info, traces
        self.fail('Contact lesson exceeded finite 300-tick horizon')

    def test_seeded_centered_spawns_have_full_body_margin_and_raw_contract(self):
        starts = set()
        for seed in range(20):
            for slot in (0, 1):
                for direction in (-1, 1):
                    obs, info = self.env.reset(seed=seed, options={'scenario': {
                        'player_slot': slot, 'direction': direction}})
                    trace = info['trace']
                    p, q = trace['post_players'][slot], trace['post_players'][1-slot]
                    self.assertEqual(trace['stage'], 'bridge_b1_contact')
                    self.assertEqual(trace['stage_version'], 'bridge_b1_contact-v1')
                    self.assertEqual(trace['initial_direction'], direction)
                    self.assertEqual(trace['initial_learner_x'], p['x'])
                    self.assertEqual(trace['initial_target_x'], q['x'])
                    self.assertEqual(trace['approach_progress'], 0.)
                    self.assertIsNone(trace['approach_progress_at_first_hit'])
                    self.assertGreaterEqual(direction*(q['x']-p['x']), 3.1)
                    self.assertLessEqual(direction*(q['x']-p['x']), 3.4)
                    self.assertLessEqual(abs((p['x']+q['x'])/2-4.5), .15)
                    self.assertTrue(all(x['fully_supported'] for x in (p, q)))
                    self.assertTrue(all(x['horizontal_floor_clearance'] >= .2 for x in (p, q)))
                    self.assertEqual(obs.shape, (64,))
                    self.assertEqual(obs.dtype, np.float32)
                    starts.add((p['x'], q['x']))
                    json.dumps(info, allow_nan=False)
        self.assertGreater(len(starts), 20)
        np.testing.assert_array_equal(self.env.observation_space.low, self.env.raw_env.observation_space.low.astype(np.float32))
        _, _, _, _, info = self.env.step(np.full(10, 2.))
        self.assertEqual(info['trace']['executed_action'], [1., 1., 1., 1., 1., 0., 1., 1., 0., 0.])
        self.assertEqual(info['trace']['opponent_action'], [0.]*10)
        self.assertFalse(self.module.B1_TRAINING_QUALIFIED)

    def test_scripted_contact_is_attributed_safe_and_never_relocates(self):
        for slot in (0, 1):
            for direction in (-1, 1):
                with self.subTest(slot=slot, direction=direction):
                    reset, info, traces = self.run_controller(self.module.scripted_contact_action, slot=slot, direction=direction)
                    summary = info['episode_diagnostics']
                    self.assertEqual(summary['outcome'], 'success')
                    self.assertEqual(summary['stage_version'], 'bridge_b1_contact-v1')
                    self.assertEqual(summary['reward_terms']['contact'], 1.)
                    self.assertEqual(summary['reward_terms']['completion'], 2.)
                    self.assertEqual(summary['positive_reward'], 3.)
                    self.assertTrue(self.env._resolved())
                    final = traces[-1]['post_players']
                    self.assertTrue(all(p['fully_supported'] and p['stocks'] > 0 for p in final))
                    self.assertGreaterEqual(final[slot]['horizontal_floor_clearance'], .2)
                    events = [event for trace in traces for event in trace['events']]
                    self.assertFalse(any(event['type'] == 'relocation' for event in events))
                    credits = [event for event in events if event['type'] == 'hit_credit']
                    self.assertEqual(len(credits), 1)
                    damage = [event for event in events if event['type'] == 'damage']
                    self.assertTrue(damage)
                    self.assertTrue(all(event['attack_id'] == credits[0]['attack_id'] for event in damage))
                    first = traces[summary['first_hit_tick']-1]
                    progress = direction*(first['post_players'][slot]['x']-reset['trace']['initial_learner_x'])
                    self.assertGreaterEqual(progress, .25)
                    self.assertEqual(first['approach_progress_at_first_hit'], progress)
                    self.assertTrue(all(trace['encounter'] == 1 for trace in traces))
                    for trace in traces[summary['first_hit_tick']:]:
                        self.assertEqual(trace['executed_action'], [0.]*10, 'Script must settle without relocation retreat')

    def test_geometry_receipt_covers_balanced_jitter_and_negative_controls(self):
        report = self.module.qualify_contact_geometry(FROZEN)
        self.assertTrue(report['qualified'], report)
        self.assertEqual(report['stage_version'], 'bridge_b1_contact-v1')
        self.assertEqual(report['source_sha256'], self.module.ORIGINAL_GAME_SHA256)
        self.assertEqual(report['adapter_source_sha256'], hashlib.sha256(Path(self.module.__file__).read_bytes()).hexdigest())
        self.assertEqual(report['seed_panel']['seeds'], list(range(20)))
        self.assertEqual(len(report['positive_cases']), 80)
        self.assertEqual(len(report['negative_cases']), 240)
        self.assertTrue(all(case['outcome'] == 'success' for case in report['positive_cases']))
        self.assertTrue(all(case['outcome'] != 'success' for case in report['negative_cases']))
        self.assertTrue(all(case['spawn_min_clearance'] >= .2 for case in report['positive_cases']))
        self.assertTrue(all(case['final_learner_clearance'] >= .2 for case in report['positive_cases']))
        self.assertTrue(all(case['relocations'] == 0 for case in report['positive_cases']+report['negative_cases']))
        self.assertEqual({case['controller'] for case in report['negative_cases']}, {'idle', 'stationary_held_light', 'walkoff'})
        json.dumps(report, allow_nan=False)

    def test_contact_qualification_does_not_promote_the_rejected_two_hit_lesson(self):
        self.assertTrue(self.module.CONTACT_TRAINING_QUALIFIED)
        self.assertFalse(self.module.B1_TRAINING_QUALIFIED)
        self.assertEqual(self.module.CONTACT_VERSION, 'bridge_b1_contact-v1')
        _, info = self.env.reset(seed=17)
        self.assertTrue(info['trace']['training_qualified'])

    def test_contact_own_stock_loss_suppresses_simultaneous_damage_and_deadline(self):
        self.env.reset(seed=17, options={'scenario': {'direction': 1}})
        self.env._place_fresh_player(0, 4.1, self.env.floor_y, 1)
        target = self.env._place_fresh_player(1, 4.9, self.env.floor_y, -1)
        original_damage = target.apply_damage
        def lose_stock_on_real_hit(*args, **kwargs):
            original_damage(*args, **kwargs)
            self.env.raw_env.players[0].stocks -= 1
        target.apply_damage = lose_stock_on_real_hit
        self.env.tick = 296
        for tick in range(4):
            action = np.zeros(10)
            action[7] = int(tick == 0)
            _, reward, terminated, truncated, info = self.env.step(action)
        self.assertEqual(info['trace']['tick'], 300)
        self.assertEqual(info['episode_diagnostics']['outcome'], 'own_stock_loss')
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, -5.001)
        self.assertEqual(info['trace']['reward_terms'], {
            'contact': 0., 'completion': 0., 'stock_loss': -5., 'timeout': 0., 'tick': -.001})
        self.assertIsNone(self.env.first_hit_tick)

    def test_contact_tick_trajectory_matches_unwrapped_original_game(self):
        reference = self.module.make_b1_contact_env(FROZEN, seed=17)
        self.addCleanup(reference.close)
        for env in (self.env, reference):
            env.reset(seed=17, options={'scenario': {'direction': 1}})
        original_bodies = [p.body for p in self.env.raw_env.players]
        for _ in range(300):
            action = self.module.scripted_contact_action(self.env)
            obs, _, done, _, _ = self.env.step(action)
            with reference._rng_scope():
                reference.raw_env.step({0: action, 1: np.zeros(10)})
            np.testing.assert_array_equal(obs, reference._observation())
            for slot, (p, q) in enumerate(zip(self.env.raw_env.players, reference.raw_env.players)):
                self.assertIs(p.body, original_bodies[slot])
                np.testing.assert_array_equal(p.body.position, q.body.position)
                np.testing.assert_array_equal(p.body.velocity, q.body.velocity)
                self.assertEqual(p.damage, q.damage)
                self.assertEqual(type(p.state), type(q.state))
            if done:
                break
        self.assertEqual(self.env.outcome, 'success')

    def test_first_hit_progress_is_latched_and_cannot_be_repaired_after_contact(self):
        self.env.reset(seed=17, options={'scenario': {'direction': 1}})
        self.env._place_fresh_player(0, 4.1, self.env.floor_y, 1)
        self.env._place_fresh_player(1, 4.9, self.env.floor_y, -1)
        self.env.initial_learner_x = 4.1
        self.env.initial_target_x = 4.9
        for tick in range(300):
            action = np.zeros(10)
            action[7] = int(tick == 0)
            _, _, done, _, info = self.env.step(action)
            if done:
                break
        self.assertIsNotNone(self.env.first_hit_tick)
        self.assertLess(info['trace']['approach_progress_at_first_hit'], .25)
        self.assertNotEqual(self.env.outcome, 'success')
        self.assertEqual(info['episode_diagnostics']['positive_reward'], 1.)

    def test_contact_private_rng_trace_file_and_terminal_semantics(self):
        other = self.module.make_b1_contact_env(FROZEN, seed=17)
        self.addCleanup(other.close)
        for env in (self.env, other):
            env.reset(seed=17)
        python_before, numpy_before = random.getstate(), np.random.get_state()
        for _ in range(300):
            action = self.module.scripted_contact_action(self.env)
            first = self.env.step(action)
            second = other.step(action)
            np.testing.assert_array_equal(first[0], second[0])
            self.assertEqual(first[1:4], second[1:4])
            if first[2]:
                break
        self.assertEqual(random.getstate(), python_before)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        with tempfile.TemporaryDirectory() as directory:
            logged = self.module.make_b1_contact_env(FROZEN, seed=17, trace_dir=directory)
            self.addCleanup(logged.close)
            logged.tick = 299
            _, reward, terminated, truncated, info = logged.step(np.zeros(10))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(reward, -1.001)
            traces = [json.loads(line) for path in Path(directory).glob('*.jsonl') for line in path.read_text().splitlines()]
            self.assertEqual(len(traces), 2)
            self.assertTrue(all(trace['stage_version'] == 'bridge_b1_contact-v1' for trace in traces))
            self.assertEqual(traces[-1], info['trace'])


if __name__ == '__main__':
    unittest.main()
