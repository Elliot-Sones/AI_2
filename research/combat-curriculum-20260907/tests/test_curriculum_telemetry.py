import copy
import gzip
import hashlib
import importlib
import os
from pathlib import Path
import tempfile
import unittest


def player(slot, x, *, state='WalkingState', damage=0, attack_id=None):
    return dict(player_slot=slot, x=x, y=0., vx=0., vy=0., facing=1 if slot == 0 else -1,
                stocks=3, damage=damage, state=state, grounded=True, fully_supported=True,
                horizontal_floor_clearance=.5, attack_id=attack_id, stun_time_stored=0.)


def frame(tick=1, *, pre=None, post=None, events=None, raw=None, outcome='ongoing', encounter=1):
    pre = pre or [player(0, 0.), player(1, 3.)]
    starts = {e['attack_id'] for e in events or [] if e['type'] == 'attack_started'}
    contact = any(e['type'] == 'damage' and e.get('source_slot') == 0 and e.get('attack_id') in starts for e in events or [])
    terms = dict(contact=float(contact), completion=2. if outcome == 'success' else 0.,
                 stock_loss=-5. if outcome == 'own_stock_loss' else 0.,
                 timeout=-1. if outcome in ('timeout', 'response_deadline', 'second_hit_deadline', 'unsafe_relocation') else 0., tick=-.001)
    return dict(schema_version='b1-trace-v1', stage='bridge_b1', tick=tick,
                episode_id=1, scenario_id='synthetic-1', seed=17, player_slot=0,
                pre_players=pre, post_players=post or copy.deepcopy(pre),
                raw_action=raw or [0.] * 10, executed_action=[min(1., max(0., value)) if i in (0,1,2,3,4,6,7) else 0. for i, value in enumerate(raw or [0.] * 10)],
                allowed_mask=[i in (0,1,2,3,4,6,7) for i in range(10)],
                opponent_action=[0.] * 10, events=events or [], reward_terms=terms,
                encounter=encounter, outcome=outcome, measured_light_reach=1.)


def contact_traces(*, slot=0, direction=1, progress=.25, mode='deterministic'):
    initial = [player(0, 0.), player(1, 0.)]
    initial[1-slot]['x'] = 2.*direction
    approached = copy.deepcopy(initial)
    approached[slot]['x'] = progress*direction
    hit_state = copy.deepcopy(approached)
    hit_state[slot].update(state='AttackState', attack_id='contact')
    hit_state[1-slot]['damage'] = 2.
    final = copy.deepcopy(hit_state)
    final[slot].update(state='WalkingState', attack_id=None)
    start = dict(type='attack_started', player_slot=slot, attack_id='contact', attack_type='NLIGHT', encounter=1)
    hit = dict(type='damage', source_slot=slot, target_slot=1-slot, attack_id='contact', amount=2., source='game', encounter=1)
    end = dict(type='attack_ended', player_slot=slot, attack_id='contact', encounter=1)
    traces = [frame(1, pre=initial, post=approached), frame(2, pre=approached, post=hit_state, events=[start, hit]),
              frame(3, pre=hit_state, post=final, events=[end], outcome='success')]
    traces[1]['reward_terms']['contact'] = 1.
    for trace in traces:
        trace.update(stage='bridge_b1_contact', stage_version='bridge_b1_contact-v1', player_slot=slot,
                     mode=mode, training_qualified=True)
    return traces


class TelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.telemetry = importlib.import_module('curriculum_telemetry') if importlib.util.find_spec('curriculum_telemetry') else None

    def setUp(self):
        self.assertIsNotNone(self.telemetry, 'curriculum_telemetry implementation is missing')

    def test_attack_requests_starts_damage_frames_and_unique_hits_are_separate(self):
        raw = [0.] * 10
        raw[7] = 2.
        start = {'type': 'attack_started', 'player_slot': 0, 'attack_id': 'a1', 'attack_type': 'light', 'encounter': 1}
        damage = {'type': 'damage', 'source_slot': 0, 'target_slot': 1, 'attack_id': 'a1', 'amount': 2., 'source': 'original_game.Power.do_power', 'encounter': 1}
        p0 = [player(0, 0.), player(1, 3.)]
        p1 = [player(0, .5, state='AttackState', attack_id='a1'), player(1, 3., damage=2)]
        p2 = [player(0, .5, state='AttackState', attack_id='a1'), player(1, 3., damage=4)]
        p3 = [player(0, .5), player(1, 3., damage=4)]
        traces = [frame(1, pre=p0, post=p1, raw=raw, events=[start, damage]),
                  frame(2, pre=p1, post=p2, raw=raw, events=[damage]),
                  frame(3, pre=p2, post=p3, events=[dict(type='attack_ended', player_slot=0, attack_id='a1', attack_type='light')], outcome='timeout')]
        result = self.telemetry.aggregate_episode(traces)
        self.assertEqual(result['attack']['raw_requested_ticks'], 2)
        self.assertEqual(result['attack']['raw_press_run_lengths'], [2])
        self.assertEqual(result['attack']['started_instances'], 1)
        self.assertEqual(result['attack']['damage_frames'], 2)
        self.assertEqual(result['attack']['unique_hit_instances'], 1)
        self.assertEqual(result['attack']['damage_dealt'], 4.)
        self.assertEqual(result['attack']['first_hit_seconds'], 1/30)
        self.assertEqual(result['attack']['attack_state_ticks'], 2)
        self.assertEqual(result['attack']['hit_rate_denominator'], 1)
        self.assertEqual(result['failures'][0]['tick'], 3)
        self.assertEqual(result['failures'][0]['distance'], 2.5)

    def test_contact_only_success_both_slots_modes_and_fixed_initial_direction(self):
        for slot in (0, 1):
            for direction in (-1, 1):
                for mode in ('deterministic', 'sampled'):
                    with self.subTest(slot=slot, direction=direction, mode=mode):
                        result = self.telemetry.aggregate_episode(contact_traces(slot=slot, direction=direction, mode=mode), learner_slot=slot, mode=mode)
                        self.assertTrue(result['success'])
                        self.assertEqual(result['metric_version'], 'bridge_b1_contact-v1')
                        self.assertEqual(result['stage'], 'bridge_b1_contact')
                        self.assertIsNone(result['movement']['pursuit'])
                        approach = result['movement']['contact_approach']
                        self.assertEqual(approach['metric_version'], 'initial_target_horizontal_progress_v1')
                        self.assertEqual(approach['initial_direction'], direction)
                        self.assertEqual(approach['first_credited_hit_tick'], 2)
                        self.assertEqual(approach['learner_progress_at_first_hit'], .25)
                        self.assertEqual(result['reward_terms'], dict(contact=1., completion=2., stock_loss=0., timeout=0., tick=-.003))

    def test_contact_only_rejects_missing_approach_at_first_hit_and_unsafe_finish(self):
        for progress in (0., .249, -.25):
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                self.telemetry.aggregate_episode(contact_traces(progress=progress))
        # Moving only after the first hit cannot repair the approach condition.
        late = contact_traces(progress=0.)
        late[-1]['post_players'][0]['x'] = 1.
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode(late)
        for slot, key, value in [(0, 'fully_supported', False), (1, 'fully_supported', False), (0, 'grounded', False),
                                 (0, 'horizontal_floor_clearance', .199), (0, 'stocks', 0),
                                 (1, 'stun_time_stored', 1.)]:
            unsafe = contact_traces()
            unsafe[-1]['post_players'][slot][key] = value
            with self.subTest(slot=slot, key=key), self.assertRaises(ValueError):
                self.telemetry.aggregate_episode(unsafe)

    def test_contact_only_rejects_relocation_metric_drift_and_late_episode(self):
        relocated = contact_traces()
        relocated[0]['events'].append(dict(type='relocation', player_slot=1, reason='encounter_boundary',
                                          **{'from': {'x':2., 'y':0.}, 'to': {'x':2., 'y':0.}}))
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode(relocated)
        drift = contact_traces()
        drift[-1].update(stage='bridge_b1', stage_version='bridge_b1-v1')
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode(drift)
        late = contact_traces()[0]
        late['tick'] = 301
        with self.assertRaises(ValueError):
            self.telemetry.validate_trace(late)
        premature = contact_traces()[0]
        premature.update(outcome='timeout', tick=299)
        premature['reward_terms']['timeout'] = -1.
        with self.assertRaises(ValueError):
            self.telemetry.validate_trace(premature)
        premature['tick'] = 300
        self.telemetry.validate_trace(premature)
        pursuit_failure = contact_traces()[0]
        pursuit_failure['outcome'] = 'response_deadline'
        with self.assertRaises(ValueError):
            self.telemetry.validate_trace(pursuit_failure)

    def test_contact_only_failure_does_not_invent_reengagement(self):
        trace = contact_traces()[0]
        trace.update(outcome='unsafe_floor_support')
        trace['post_players'][0]['fully_supported'] = False
        result = self.telemetry.aggregate_episode([trace])
        self.assertFalse(result['success'])
        self.assertNotIn('failed_reengagement', result['diagnoses'])
        self.assertIsNone(result['movement']['pursuit'])
        insufficient = contact_traces(progress=0.)
        insufficient[-1]['outcome'] = 'insufficient_approach'
        insufficient[-1]['reward_terms']['completion'] = 0.
        result = self.telemetry.aggregate_episode(insufficient)
        self.assertFalse(result['success'])
        self.assertIn('insufficient_approach_before_contact', result['diagnoses'])
        self.assertNotIn('failed_reengagement', result['diagnoses'])
        self.assertEqual(result['reward_terms']['contact'], 1.)
        self.assertEqual(result['reward_terms']['timeout'], 0.)

    def test_contact_only_multiframe_damage_keeps_one_reward_and_clearance_boundary(self):
        traces = contact_traces()
        hit = copy.deepcopy(traces[1]['events'][1])
        traces[2]['events'].insert(0, hit)
        traces[2]['post_players'][1]['damage'] = 4.
        traces[2]['post_players'][0]['horizontal_floor_clearance'] = .2
        result = self.telemetry.aggregate_episode(traces)
        self.assertTrue(result['success'])
        self.assertEqual(result['attack']['damage_frames'], 2)
        self.assertEqual(result['attack']['unique_hit_instances'], 1)
        self.assertEqual(result['reward_terms']['contact'], 1.)
        self.assertEqual(result['movement']['contact_approach']['first_credited_hit_tick'], 2)
        traces[2]['reward_terms']['contact'] = 1.
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode(traces)

    def test_original_game_contact_script_reconstructs_both_slots_and_directions(self):
        frozen = Path(os.environ.get('AI2_FROZEN_ROOT', '/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected'))
        if not frozen.is_dir():
            self.skipTest('Set AI2_FROZEN_ROOT to run original-game contact integration')
        from combat_curriculum import make_b1_contact_env, scripted_contact_action
        for slot in (0, 1):
            for direction in (-1, 1):
                with self.subTest(slot=slot, direction=direction):
                    env = make_b1_contact_env(frozen, seed=17, player_slot=slot)
                    try:
                        env.reset(seed=17, options={'scenario': dict(player_slot=slot, direction=direction)})
                        traces = []
                        for _ in range(300):
                            _, _, terminated, truncated, info = env.step(scripted_contact_action(env))
                            traces.append(info['trace'])
                            if terminated or truncated:
                                break
                        result = self.telemetry.aggregate_episode(traces, learner_slot=slot)
                        diagnostics = info['episode_diagnostics']
                        self.assertTrue(result['success'])
                        self.assertIs(result['training_qualified'], diagnostics['training_qualified'])
                        self.assertEqual(result['metric_version'], 'bridge_b1_contact-v1')
                        self.assertFalse(result['fell'])
                        self.assertEqual(result['movement']['relocations'], [])
                        self.assertIsNone(result['movement']['pursuit'])
                        self.assertEqual(result['reward_terms'], diagnostics['reward_terms'])
                        self.assertEqual(result['attack']['first_hit_tick'], diagnostics['first_hit_tick'])
                        self.assertAlmostEqual(result['movement']['contact_approach']['learner_progress_at_first_hit'],
                                               traces[-1]['approach_progress_at_first_hit'])
                    finally:
                        env.close()

    def test_relocation_is_excluded_from_actual_approach(self):
        p0 = [player(0, 0.), player(1, 3.)]
        p1 = [player(0, .5), player(1, 10.)]
        relocation = dict(type='relocation', player_slot=1, **{'from': {'x': 3., 'y': 0.}, 'to': {'x': 10., 'y': 0.}}, reason='encounter_boundary')
        result = self.telemetry.aggregate_episode([frame(pre=p0, post=p1, events=[relocation], outcome='timeout', encounter=2)])
        self.assertAlmostEqual(result['movement']['net_toward_target_displacement'], .5)
        self.assertAlmostEqual(result['movement']['distance_reduction_excluding_relocation'], .5)
        self.assertEqual(result['movement']['relocations'][0]['tick'], 1)

    def test_failures_distinguish_no_attempt_blocked_and_whiff_with_exact_evidence(self):
        raw = [0.] * 10
        raw[7] = 1.
        idle = self.telemetry.aggregate_episode([frame(outcome='timeout')])
        blocked = self.telemetry.aggregate_episode([frame(raw=raw, outcome='timeout')])
        events = [dict(type='attack_started', player_slot=0, attack_id=1, attack_type='light'),
                  dict(type='attack_ended', player_slot=0, attack_id=1, attack_type='light')]
        whiff = self.telemetry.aggregate_episode([frame(raw=raw, events=events, outcome='timeout')])
        self.assertIn('no_attack_attempt', idle['diagnoses'])
        self.assertIn('requested_but_no_attack_started', blocked['diagnoses'])
        self.assertIn('started_but_no_hit', whiff['diagnoses'])
        self.assertEqual(whiff['attack']['ended_without_damage'], 1)
        self.assertEqual(whiff['failures'][0]['raw_action'][7], 1.)
        self.assertEqual(whiff['attack']['instances'][0]['status'], 'ended_without_damage')

    def test_invalid_or_unreconstructable_evidence_fails_closed(self):
        bad = frame()
        bad['post_players'][0]['x'] = float('nan')
        with self.assertRaises(ValueError):
            self.telemetry.validate_trace(bad)
        for mutate in [lambda x: x['executed_action'].__setitem__(8, 1.),
                       lambda x: x['pre_players'].pop(),
                       lambda x: x['events'].append(dict(type='damage', amount=2., source_slot=0, target_slot=1, attack_id='missing', source='game'))]:
            trace = frame()
            mutate(trace)
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                self.telemetry.aggregate_episode([trace])
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode([frame(1), frame(3)])
        changed = frame(2)
        changed['pre_players'][0]['x'] = 9.
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode([frame(1), changed])

    def test_atomic_compressed_inventory_roundtrip_rejects_tampering_and_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = self.telemetry.AtomicTraceWriter(folder, chunk_size=2)
            traces = [frame(1), frame(2), frame(3, outcome='timeout')]
            for trace in traces:
                writer.append(trace)
            inventory = writer.close()
            self.assertEqual(inventory['frames'], 3)
            self.assertEqual(len(inventory['chunks']), 2)
            self.assertEqual(self.telemetry.load_trace_inventory(Path(folder) / 'inventory.json'), traces)
            for chunk in inventory['chunks']:
                content = (Path(folder) / chunk['path']).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), chunk['sha256'])
                self.assertEqual(len(gzip.decompress(content).splitlines()), chunk['frames'])
            with self.assertRaises(FileExistsError):
                self.telemetry.AtomicTraceWriter(folder)
            (Path(folder) / inventory['chunks'][0]['path']).write_bytes(b'corruption')
            with self.assertRaises(ValueError):
                self.telemetry.load_trace_inventory(Path(folder) / 'inventory.json')

    def test_float_variation_is_not_button_variation(self):
        a, b = [0.01] * 10, [0.49] * 10
        report = self.telemetry.aggregate_episode([frame(1, raw=a), frame(2, raw=b, outcome='timeout')])
        self.assertFalse(report['actions']['pressed_buttons_varied'])
        self.assertEqual(report['actions']['unique_pressed_patterns'], 1)
        self.assertEqual(report['movement']['time_in_measured_range_seconds'], 0.)

    def test_two_resolved_light_hits_and_reengagement_reconstruct_success(self):
        p0 = [player(0, 0.), player(1, 3.)]
        p1 = [player(0, 2.), player(1, 3., damage=2.)]
        start1 = dict(type='attack_started', player_slot=0, attack_id='first', attack_type='light', encounter=1)
        hit1 = dict(type='damage', source_slot=0, target_slot=1, attack_id='first', amount=2., source='game', encounter=1)
        end1 = dict(type='attack_ended', player_slot=0, attack_id='first', attack_type='light', encounter=1)
        p2 = [player(0, 2.), player(1, 6.)]
        relocate = dict(type='relocation', player_slot=1, **{'from': {'x':3., 'y':0.}, 'to': {'x':6., 'y':0.}}, reason='encounter_boundary')
        p3 = [player(0, 3.), player(1, 6.)]
        p4 = [player(0, 5.), player(1, 6., damage=2.)]
        start2 = dict(start1, attack_id='second', encounter=2)
        hit2 = dict(hit1, attack_id='second', encounter=2)
        end2 = dict(end1, attack_id='second', encounter=2)
        traces = [frame(1, pre=p0, post=p1, events=[start1, hit1, end1]),
                  frame(2, pre=p1, post=p2, events=[relocate], encounter=2),
                  frame(3, pre=p2, post=p3, encounter=2),
                  frame(4, pre=p3, post=p4, events=[start2, hit2, end2], outcome='success', encounter=2)]
        report = self.telemetry.aggregate_episode(traces)
        self.assertTrue(report['success'])
        self.assertEqual(report['movement']['pursuit']['response_tick'], 3)
        self.assertEqual(report['attack']['unique_hit_instances'], 2)
        delayed_end = copy.deepcopy(traces)
        delayed_end[-1]['events'].insert(0, delayed_end[0]['events'].pop())
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode(delayed_end)
        unsupported = copy.deepcopy(traces)
        unsupported[-1]['post_players'][0]['fully_supported'] = False
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode(unsupported)
        unsupported_boundary = copy.deepcopy(traces)
        unsupported_boundary[1]['post_players'][0]['fully_supported'] = False
        unsupported_boundary[2]['pre_players'][0]['fully_supported'] = False
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode(unsupported_boundary)
        unresolved_boundary = copy.deepcopy(traces)
        unresolved_boundary[1]['post_players'][1]['stun_time_stored'] = 1.
        unresolved_boundary[2]['pre_players'][1]['stun_time_stored'] = 1.
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode(unresolved_boundary)
        unresolved = copy.deepcopy(traces)
        unresolved[-1]['events'].pop()
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode(unresolved)
        for field, value in [('state', 'StunState'), ('stun_time_stored', 2.)]:
            unresolved_stun = copy.deepcopy(traces)
            unresolved_stun[-1]['post_players'][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.telemetry.aggregate_episode(unresolved_stun)
        fabricated = [frame(outcome='success')]
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode(fabricated)

    def test_malformed_snapshot_and_non_game_event_types_raise_value_error(self):
        bad = frame()
        bad['pre_players'][0] = []
        with self.assertRaises(ValueError):
            self.telemetry.validate_trace(bad)

    def test_writer_interleaves_envs_and_inventory_rejects_symlink_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = self.telemetry.AtomicTraceWriter(Path(folder)/'traces', chunk_size=2)
            traces = [dict(frame(1), env_index=0, episode_id=1), dict(frame(1), env_index=1, episode_id=1),
                      dict(frame(2, outcome='timeout'), env_index=0, episode_id=1)]
            for trace in traces:
                writer.append(trace)
            inventory = writer.close()
            self.assertEqual(self.telemetry.load_trace_inventory(Path(folder)/'traces'/'inventory.json'), traces)
            chunk = Path(folder)/'traces'/inventory['chunks'][0]['path']
            outside = Path(folder)/'relocated.gz'
            chunk.rename(outside)
            chunk.symlink_to(outside)
            with self.assertRaises(ValueError):
                self.telemetry.load_trace_inventory(Path(folder)/'traces'/'inventory.json')

    def test_attack_contact_evidence_records_vulnerability_without_inventing_hit(self):
        events = [dict(type='attack_started', player_slot=0, attack_id='a', attack_type='light'),
                  dict(type='collision', source_slot=0, target_slot=1, attack_id='a', intersects=True, vulnerable=False),
                  dict(type='attack_ended', player_slot=0, attack_id='a', attack_type='light')]
        report = self.telemetry.aggregate_episode([frame(events=events, outcome='timeout')])
        self.assertEqual(report['attack']['unique_hit_instances'], 0)
        contact = report['attack']['instances'][0]['contacts'][0]
        self.assertFalse(contact['vulnerable'])
        self.assertEqual(contact['evidence']['tick'], 1)
        self.assertEqual(contact['evidence']['target']['x'], 3.)
        self.assertEqual(report['actions']['raw_means'], [0.] * 10)

    def test_movement_lock_requires_sustained_attack_movement_request_and_no_displacement(self):
        pre = [player(0, 0., state='AttackState'), player(1, 3.)]
        raw = [0.] * 10
        raw[1] = 1.
        traces = [frame(tick, pre=copy.deepcopy(pre), raw=raw, outcome='timeout' if tick == 60 else None) for tick in range(1,61)]
        locked = self.telemetry.aggregate_episode(traces)
        self.assertTrue(locked['systemic_movement_lock'])
        self.assertEqual(locked['movement']['lock_windows'][0]['start_tick'], 1)
        self.assertEqual(locked['movement']['lock_windows'][0]['end_tick'], 60)
        idle = copy.deepcopy(traces)
        for trace in idle:
            trace['raw_action'][1] = trace['executed_action'][1] = 0.
        self.assertFalse(self.telemetry.aggregate_episode(idle)['systemic_movement_lock'])
        moving = copy.deepcopy(traces)
        for index, trace in enumerate(moving):
            trace['pre_players'][0]['x'] = index*.02
            trace['post_players'][0]['x'] = (index+1)*.02
        self.assertFalse(self.telemetry.aggregate_episode(moving)['systemic_movement_lock'])

    def test_reengagement_uses_horizontal_progress_and_real_light_move_names(self):
        start = dict(type='attack_started', player_slot=0, attack_id='a', attack_type='NLIGHT', encounter=1)
        hit = dict(type='damage', source_slot=0, target_slot=1, attack_id='a', amount=2., source='game', encounter=1)
        end = dict(type='attack_ended', player_slot=0, attack_id='a', attack_type='NLIGHT', encounter=1)
        p0 = [player(0, 0.), player(1, 2.)]
        p1 = [player(0, 1.), player(1, 2., damage=2.)]
        p2 = [player(0, 1.), player(1, 5.)]
        p3 = [player(0, 2.), player(1, 5.)]
        p3[0]['y'] = -3.
        p4 = [player(0, 4.), player(1, 5., damage=2.)]
        relocation = dict(type='relocation', player_slot=1, **{'from': {'x':2.,'y':0.}, 'to': {'x':5.,'y':0.}}, reason='encounter_boundary')
        traces = [frame(1, pre=p0, post=p1, events=[start,hit,end]), frame(2, pre=p1, post=p2, events=[relocation], encounter=2),
                  frame(3, pre=p2, post=p3, encounter=2), frame(4, pre=p3, post=p4, events=[dict(e,attack_id='b',encounter=2) for e in [start,hit,end]], encounter=2, outcome='success')]
        result = self.telemetry.aggregate_episode(traces)
        self.assertTrue(result['success'])
        self.assertEqual(result['movement']['pursuit']['response_tick'], 3)
        self.assertEqual(result['movement']['pursuit']['metric_version'], 'horizontal_gap_and_learner_progress_v1')

    def test_unknown_damage_remains_unattributed_and_never_becomes_a_hit(self):
        event = dict(type='damage', source_slot=None, target_slot=1, attack_id=None, amount=2., source='unknown')
        try:
            result = self.telemetry.aggregate_episode([frame(events=[event], outcome='timeout')])
        except ValueError as error:
            self.fail(f'Explicit unknown damage is valid evidence, not malformed: {error}')
        self.assertEqual(result['attack']['unique_hit_instances'], 0)
        self.assertEqual(result['attack']['unattributed_damage'], 2.)
        self.assertIsNone(result['damage_events'][0]['source_slot'])
        contradictory = frame(events=[dict(event, source_slot=0, attack_id='a')], outcome='timeout')
        with self.assertRaises(ValueError):
            self.telemetry.validate_trace(contradictory)

    def test_missing_or_changed_episode_identity_is_rejected(self):
        for key in ('episode_id', 'scenario_id', 'seed', 'player_slot'):
            missing = frame(outcome='timeout')
            missing.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.telemetry.aggregate_episode([missing])
        for key, value in [('scenario_id', 'other'), ('seed', 18)]:
            first, second = frame(1), frame(2, outcome='timeout')
            second[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.telemetry.aggregate_episode([first, second])

    def test_reward_terms_and_stock_outcome_must_match_events(self):
        for field, value in [('contact', 200.), ('contact', 1.), ('completion', 2.), ('stock_loss', -5.), ('timeout', 0.), ('tick', 0.)]:
            bad = frame(outcome='timeout')
            bad['reward_terms'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.telemetry.aggregate_episode([bad])
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode([frame(outcome='own_stock_loss')])
        lost = frame(outcome='own_stock_loss')
        lost['post_players'][0]['stocks'] -= 1
        self.assertTrue(self.telemetry.aggregate_episode([lost])['fell'])
        lost['outcome'] = 'timeout'
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode([lost])

    def test_b1_mask_and_unopposed_movement_are_required(self):
        bad = frame(outcome='timeout')
        bad['allowed_mask'][8] = True
        with self.assertRaises(ValueError):
            self.telemetry.validate_trace(bad)
        p = [player(0, 0., state='AttackState'), player(1, 3.)]
        raw = [0.] * 10
        raw[1] = raw[3] = 1.
        traces = [frame(tick, pre=copy.deepcopy(p), raw=raw, outcome='timeout' if tick == 60 else None) for tick in range(1, 61)]
        self.assertFalse(self.telemetry.aggregate_episode(traces)['systemic_movement_lock'])

    def test_movement_lock_uses_left_one_and_right_three_not_up_zero(self):
        state = [player(0, 0., state='AttackState'), player(1, 3.)]
        for controls, expected in [((3, 7), True), ((1, 7), True), ((0, 7), False), ((1, 3, 7), False)]:
            raw = [0.] * 10
            for control in controls:
                raw[control] = 1.
            traces = [frame(tick, pre=copy.deepcopy(state), raw=raw, outcome='timeout' if tick == 60 else None)
                      for tick in range(1, 61)]
            with self.subTest(controls=controls):
                report = self.telemetry.aggregate_episode(traces)
                self.assertIs(report['systemic_movement_lock'], expected)
                if expected:
                    self.assertEqual(report['movement']['lock_windows'][0]['movement_requested_ticks'], 60)

    def test_mode_and_attack_encounter_identity_cannot_drift(self):
        bad = frame(outcome='timeout')
        bad['mode'] = 'sampled'
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode([bad], mode='deterministic')
        bad = frame(outcome='timeout', events=[dict(type='attack_started', player_slot=0, attack_id='a', attack_type='light', encounter=3)])
        with self.assertRaises(ValueError):
            self.telemetry.validate_trace(bad)

    def test_repeated_contact_reward_or_damage_after_attack_end_is_rejected(self):
        start = dict(type='attack_started', player_slot=0, attack_id='a', attack_type='light', encounter=1)
        hit = dict(type='damage', source_slot=0, target_slot=1, attack_id='a', amount=2., source='game')
        first, second = frame(1, events=[start, hit]), frame(2, events=[hit], outcome='timeout')
        second['reward_terms']['contact'] = 1.
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode([first, second])
        second['reward_terms']['contact'] = 0.
        first['events'].append(dict(type='attack_ended', player_slot=0, attack_id='a'))
        with self.assertRaises(ValueError):
            self.telemetry.aggregate_episode([first, second])

    def test_unsafe_floor_support_is_not_success_or_a_stock_fall(self):
        unsafe = frame(outcome='unsafe_floor_support')
        unsafe.update(training_qualified=False, qualification_blocker='Measured floor-support failure')
        unsafe['post_players'][0]['fully_supported'] = False
        result = self.telemetry.aggregate_episode([unsafe])
        self.assertFalse(result['success'])
        self.assertFalse(result['fell'])
        self.assertEqual(result['reward_terms']['timeout'], 0.)
        self.assertEqual(result['reward_terms']['completion'], 0.)
        self.assertIn('unsafe_floor_support', result['diagnoses'])
        self.assertIs(result.get('training_qualified'), False)
        self.assertEqual(result.get('qualification_blocker'), unsafe['qualification_blocker'])

    def test_pursuit_progress_uses_the_fixed_direction_at_relocation(self):
        p0 = [player(0, 2.), player(1, 3.)]
        p1 = [player(0, 2.), player(1, 6.)]
        p2 = [player(0, 10.), player(1, 6.)]
        p3 = [player(0, 2.), player(1, 6.)]
        relocation = dict(type='relocation', player_slot=1, **{'from': {'x':3.,'y':0.}, 'to': {'x':6.,'y':0.}}, reason='encounter_boundary')
        traces = [frame(1, pre=p0, post=p1, events=[relocation], encounter=2),
                  frame(2, pre=p1, post=p2, encounter=2), frame(3, pre=p2, post=p3, encounter=2, outcome='timeout')]
        result = self.telemetry.aggregate_episode(traces)
        self.assertEqual(result['movement']['pursuit']['toward_target_displacement'], 0.)

    def test_original_game_prototype_is_rejected_for_unsafe_floor_support(self):
        frozen = Path(os.environ.get('AI2_FROZEN_ROOT', '/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected'))
        if not frozen.is_dir():
            self.skipTest('Set AI2_FROZEN_ROOT to run original-game evidence integration')
        from combat_curriculum import make_b1_env, scripted_b1_action, B1_TRAINING_QUALIFIED
        self.assertFalse(B1_TRAINING_QUALIFIED)
        for slot in (0, 1):
            for direction in (-1, 1):
                with self.subTest(slot=slot, direction=direction):
                    env = make_b1_env(frozen, seed=17)
                    try:
                        env.reset(seed=17, options={'scenario': dict(player_slot=slot, direction=direction)})
                        traces = []
                        for _ in range(300):
                            _, _, terminated, truncated, info = env.step(scripted_b1_action(env))
                            traces.append(info['trace'])
                            if terminated or truncated:
                                break
                        result = self.telemetry.aggregate_episode(traces, learner_slot=slot)
                        diagnostics = info['episode_diagnostics']
                        self.assertFalse(result['success'])
                        self.assertFalse(result['fell'])
                        self.assertEqual(result['outcome'], 'unsafe_floor_support')
                        self.assertEqual(result['attack']['unique_hit_instances'], len(diagnostics['distinct_hit_ids']))
                        self.assertEqual(result['attack']['first_hit_tick'], diagnostics['first_hit_tick'])
                        self.assertEqual(result['movement']['pursuit']['response_tick'] if result['movement']['pursuit'] else None, diagnostics['pursuit_tick'])
                        self.assertEqual(result['reward_terms'], diagnostics['reward_terms'])
                    finally:
                        env.close()


if __name__ == '__main__':
    unittest.main()
