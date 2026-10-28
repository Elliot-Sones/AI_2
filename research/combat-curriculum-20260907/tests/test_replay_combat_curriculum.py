import importlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np


FROZEN = Path(os.environ.get('AI2_FROZEN_ROOT', '/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected'))


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('replay_combat_curriculum'), 'policy-free verified replay is missing')
        self.module = importlib.import_module('replay_combat_curriculum')
        self.temporary = tempfile.TemporaryDirectory(prefix='ai2-replay-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def record(self, *, player_slot=1, direction=-1, metric_version='bridge_b1-v1'):
        import combat_curriculum
        from curriculum_telemetry import AtomicTraceWriter, load_trace_inventory
        from evaluate_combat_curriculum import run_case
        if metric_version == 'bridge_b1_contact-v1':
            factory, controller = combat_curriculum.make_b1_contact_env, combat_curriculum.scripted_contact_action
        else:
            factory, controller = combat_curriculum.make_b1_env, combat_curriculum.scripted_b1_action
        env = factory(FROZEN, seed=17, player_slot=player_slot)
        case = {'id': 'scripted-replay', 'seed': 17, 'player_slot': player_slot,
                'direction': direction, 'task': 'combat', 'mode': 'deterministic',
                'action_source': 'scripted_controller',
                'metric_version': metric_version, 'bank': 'scripted-qualification'}
        writer = AtomicTraceWriter(self.directory / 'recorded', chunk_size=31)
        try:
            random.seed(case['seed'])
            np.random.seed(case['seed'])
            result = run_case(env, lambda obs, deterministic: controller(env), case,
                              deadline_unix=time.time() + 120, trace_sink=writer.append)
        finally:
            writer.close()
            env.close()
        self.assertTrue(result['terminated'] or result['truncated'])
        self.assertTrue(result['outcome'])
        case_path = self.directory / 'case.json'
        case_path.write_text(json.dumps(result))
        inventory = self.directory / 'recorded/inventory.json'
        return result, case_path, inventory, load_trace_inventory(inventory)

    def altered_inventory(self, traces):
        from curriculum_telemetry import AtomicTraceWriter
        writer = AtomicTraceWriter(self.directory / 'altered')
        for trace in traces:
            writer.append(trace)
        writer.close()
        return self.directory / 'altered/inventory.json'

    def test_positive_replay_has_exact_observations_geometry_events_and_frame_count(self):
        case, path, inventory, _ = self.record()
        frames = []
        receipt = self.module.replay_case(FROZEN, path, inventory, self.directory / 'replay',
                                         frame_sink=lambda frame: frames.append((frame.shape, frame.dtype)))
        self.assertTrue(receipt['verified'])
        self.assertIsNone(receipt['first_failure_tick'])
        self.assertEqual(receipt['max_state_error'], 0.)
        self.assertTrue(receipt['observation_bitwise_match'])
        self.assertTrue(receipt['events_match'])
        self.assertEqual(receipt['frames'], case['ticks'])
        self.assertEqual(receipt['fps'], 30)
        self.assertEqual(receipt['learner_slot'], 1)
        self.assertEqual(receipt.get('learner_marker_color'), 'BLUE')
        self.assertEqual(receipt['direction'], -1)
        self.assertEqual(len(frames), case['ticks'])
        self.assertTrue(all(shape == (360, 640, 3) and dtype == np.uint8 for shape, dtype in frames))
        saved = json.loads((self.directory / 'replay/receipt.json').read_text())
        self.assertEqual(saved, receipt)
        self.assertTrue((self.directory / 'replay/receipt.sha256.json').exists(), 'Durable receipt checksum is missing')
        hashes = json.loads((self.directory / 'replay/receipt.sha256.json').read_text())
        self.assertEqual(hashes['receipt.json'], self.module.sha256(self.directory / 'replay/receipt.json'))
        self.assertEqual(len(receipt['input_sha256']['trace_inventory']), 64)
        self.assertEqual(len(receipt['source_sha256']['original_game']), 64)
        self.assertEqual(receipt['movie'], None)
        self.assertEqual(receipt['policy_inference_performed'], False)

    def test_tampered_observation_fails_at_exact_tick_even_with_state_tolerance(self):
        case, _, _, traces = self.record()
        traces[4]['observation_after'][0] += .25
        inventory = self.altered_inventory(traces)
        receipt = self.module.replay_case(FROZEN, case, inventory, self.directory / 'mismatch',
                                         max_error=1., frame_sink=lambda frame: None)
        self.assertFalse(receipt['verified'])
        self.assertEqual(receipt['first_failure_tick'], 5)
        self.assertFalse(receipt['observation_bitwise_match'])
        self.assertGreater(receipt['max_observation_error'], 0)
        self.assertFalse((self.directory / 'mismatch/replay.mp4').exists())
        self.assertTrue((self.directory / 'mismatch/receipt.sha256.json').exists())

    def test_attack_sequence_identity_is_not_normalized_away(self):
        case, _, _, traces = self.record()
        target = next(trace for trace in traces if any(event['type'] == 'attack_started' for event in trace['events']))
        event = next(event for event in target['events'] if event['type'] == 'attack_started')
        event['attack_id'] += ':tampered'
        receipt = self.module.replay_case(FROZEN, case, self.altered_inventory(traces),
                                         self.directory / 'mismatch', frame_sink=lambda frame: None)
        self.assertFalse(receipt['verified'])
        self.assertEqual(receipt['first_failure_tick'], target['tick'])
        self.assertFalse(receipt['events_match'])

    def test_tampered_numeric_player_state_is_not_hidden_by_identity_normalization(self):
        case, _, _, traces = self.record()
        traces[2]['post_players'][1]['x'] += .125
        receipt = self.module.replay_case(FROZEN, case, self.altered_inventory(traces),
                                         self.directory / 'mismatch', frame_sink=lambda frame: None)
        self.assertFalse(receipt['verified'])
        self.assertEqual(receipt['first_failure_tick'], 3)
        self.assertAlmostEqual(receipt['max_state_error'], .125)
        self.assertIn('post_players', receipt['failure']['field'])

    def test_new_numeric_snapshot_fields_are_measured_and_use_explicit_tolerance(self):
        case, _, _, traces = self.record()
        traces[2]['post_players'][1]['horizontal_floor_clearance'] += .125
        inventory = self.altered_inventory(traces)
        receipt = self.module.replay_case(FROZEN, case, inventory,
                                         self.directory / 'exact', frame_sink=lambda frame: None)
        self.assertFalse(receipt['verified'])
        self.assertEqual(receipt['first_failure_tick'], 3)
        self.assertAlmostEqual(receipt['max_state_error'], .125)
        receipt = self.module.replay_case(FROZEN, case, inventory,
                                         self.directory / 'tolerant', max_error=.125, frame_sink=lambda frame: None)
        self.assertTrue(receipt['verified'], receipt)
        self.assertTrue(receipt['observation_bitwise_match'])
        self.assertAlmostEqual(receipt['max_state_error'], .125)

    def test_tampered_hit_event_cannot_pass(self):
        case, _, _, traces = self.record()
        target = next(trace for trace in traces if any(event['type'] == 'damage' for event in trace['events']))
        target['events'] = [event for event in target['events'] if event['type'] != 'damage']
        receipt = self.module.replay_case(FROZEN, case, self.altered_inventory(traces),
                                         self.directory / 'mismatch', frame_sink=lambda frame: None)
        self.assertFalse(receipt['verified'])
        self.assertEqual(receipt['first_failure_tick'], target['tick'])
        self.assertFalse(receipt['events_match'])

    def test_wrong_direction_and_terminal_case_claim_fail_closed(self):
        case, _, inventory, _ = self.record()
        changed = dict(case, direction=1)
        receipt = self.module.replay_case(FROZEN, changed, inventory,
                                         self.directory / 'wrong-direction', frame_sink=lambda frame: None)
        self.assertFalse(receipt['verified'])
        self.assertEqual(receipt['first_failure_tick'], 1)
        changed = dict(case, outcome='timeout', success=False)
        receipt = self.module.replay_case(FROZEN, changed, inventory,
                                         self.directory / 'wrong-outcome', frame_sink=lambda frame: None)
        self.assertFalse(receipt['verified'])
        self.assertEqual(receipt['first_failure_tick'], case['ticks'])

    def test_output_is_immutable_and_missing_direction_is_rejected(self):
        case, _, inventory, _ = self.record()
        output = self.directory / 'occupied'
        output.mkdir()
        with self.assertRaises(FileExistsError):
            self.module.replay_case(FROZEN, case, inventory, output, frame_sink=lambda frame: None)
        case.pop('direction')
        with self.assertRaisesRegex(ValueError, 'direction'):
            self.module.replay_case(FROZEN, case, inventory, self.directory / 'invalid', frame_sink=lambda frame: None)

    def test_deterministic_recording_cannot_be_relabelled_as_sampled(self):
        case, _, inventory, _ = self.record()
        with self.assertRaisesRegex(ValueError, 'mode'):
            self.module.replay_case(FROZEN, dict(case, mode='sampled'), inventory,
                                    self.directory / 'relabelled', frame_sink=lambda frame: None)

    def test_mixed_trace_metric_versions_are_rejected_before_output(self):
        case, _, _, traces = self.record()
        traces[4]['stage_version'] = 'bridge_b1_contact-v1'
        output = self.directory / 'mixed-version'
        with self.assertRaises(ValueError):
            self.module.replay_case(FROZEN, case, self.altered_inventory(traces), output,
                                    frame_sink=lambda frame: None)
        self.assertFalse(output.exists())

    def test_unknown_case_metric_version_has_no_default_environment(self):
        case, _, inventory, _ = self.record()
        output = self.directory / 'unknown-version'
        with self.assertRaises(ValueError):
            self.module.replay_case(FROZEN, dict(case, metric_version='bridge_b1_contact-v99'),
                                    inventory, output, frame_sink=lambda frame: None)
        self.assertFalse(output.exists())

    def test_case_stage_cannot_contradict_its_metric_version(self):
        case, _, inventory, _ = self.record()
        output = self.directory / 'wrong-case-stage'
        with self.assertRaisesRegex(ValueError, 'stage'):
            self.module.replay_case(FROZEN, dict(case, stage='bridge_b1_contact'), inventory,
                                    output, frame_sink=lambda frame: None)
        self.assertFalse(output.exists())

    def test_old_trace_cannot_be_relabelled_as_contact_lesson(self):
        case, _, inventory, _ = self.record()
        output = self.directory / 'relabelled-metric'
        with self.assertRaisesRegex(ValueError, 'stage_version'):
            self.module.replay_case(FROZEN, dict(case, metric_version='bridge_b1_contact-v1', stage='bridge_b1_contact'),
                                    inventory, output, frame_sink=lambda frame: None)
        self.assertFalse(output.exists())

    def test_contact_lesson_replays_one_actual_hit_and_safe_completion_exactly(self):
        case, path, inventory, traces = self.record(metric_version='bridge_b1_contact-v1')
        self.assertTrue(case['success'], case)
        self.assertTrue(all(trace['stage'] == 'bridge_b1_contact' for trace in traces))
        self.assertFalse(any(event['type'] == 'relocation' for trace in traces for event in trace['events']))
        self.assertTrue(all(player['fully_supported'] for player in traces[-1]['post_players']))
        self.assertGreaterEqual(traces[-1]['post_players'][case['player_slot']]['horizontal_floor_clearance'], .2)
        frames = []
        receipt = self.module.replay_case(FROZEN, path, inventory, self.directory / 'contact-replay',
                                         frame_sink=lambda frame: frames.append(frame.shape))
        self.assertTrue(receipt['verified'], receipt)
        self.assertEqual(receipt['metric_version'], 'bridge_b1_contact-v1')
        self.assertEqual(receipt['outcome'], 'success')
        self.assertEqual(receipt['attacks_connected'], 1)
        self.assertEqual(receipt['max_state_error'], 0.)
        self.assertEqual(receipt['max_observation_error'], 0.)
        self.assertTrue(receipt['observation_bitwise_match'])
        self.assertTrue(receipt['events_match'])
        self.assertEqual(receipt['frames'], case['ticks'])
        self.assertEqual(len(frames), case['ticks'])

    def test_contact_trace_cannot_be_relabelled_as_old_lesson(self):
        case, _, inventory, _ = self.record(metric_version='bridge_b1_contact-v1')
        output = self.directory / 'relabelled-old'
        with self.assertRaisesRegex(ValueError, 'stage_version'):
            self.module.replay_case(FROZEN, dict(case, metric_version='bridge_b1-v1', stage='bridge_b1'),
                                    inventory, output, frame_sink=lambda frame: None)
        self.assertFalse(output.exists())

    def test_contact_approach_evidence_is_compared_exactly(self):
        case, _, _, traces = self.record(metric_version='bridge_b1_contact-v1')
        traces[4]['approach_progress'] += .125
        receipt = self.module.replay_case(FROZEN, case, self.altered_inventory(traces),
                                         self.directory / 'tampered-approach', frame_sink=lambda frame: None)
        self.assertFalse(receipt['verified'])
        self.assertEqual(receipt['first_failure_tick'], 5)
        self.assertEqual(receipt['failure']['field'], 'approach_progress')

    def test_existing_ffmpeg_encodes_one_frame_per_tick_at_normal_speed(self):
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            self.skipTest('Existing ffmpeg and ffprobe executables are required')
        case, _, inventory, _ = self.record(player_slot=0, direction=1)
        output = self.directory / 'movie'
        receipt = self.module.replay_case(FROZEN, case, inventory, output)
        self.assertTrue(receipt['verified'], receipt)
        movie = output / receipt['movie']['path']
        self.assertGreater(movie.stat().st_size, 1000)
        self.assertEqual(receipt['movie']['sha256'], self.module.sha256(movie))
        metadata = json.loads(subprocess.run([
            shutil.which('ffprobe'), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate',
            '-of', 'json', str(movie),
        ], capture_output=True, text=True, check=True, timeout=30).stdout)['streams'][0]
        self.assertEqual(int(metadata['nb_read_frames']), case['ticks'])
        self.assertEqual(metadata['r_frame_rate'], '30/1')
        self.assertEqual((metadata['height'], metadata['width']), (360, 640))


if __name__ == '__main__':
    unittest.main()
