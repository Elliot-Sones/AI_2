import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest


class FakeRun:
    id = 'test-run'
    url = 'https://wandb.ai/example/utmist-ai2/runs/test-run'
    path = 'example/utmist-ai2/test-run'

    def __init__(self, fail_log=False):
        self.logs, self.summary, self.finished = [], {}, []
        self.fail_log = fail_log

    def log(self, scalars):
        if self.fail_log:
            raise RuntimeError('secret-credential-must-not-escape')
        self.logs.append(scalars)

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


class FakeSDK:
    def __init__(self, fail_log=False):
        self.run = FakeRun(fail_log)
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(kwargs)
        self.run.id = kwargs['id']
        return self.run


class FakeClock:
    def __init__(self):
        self.now = 100.
        self.on_wait = None

    def __call__(self):
        return self.now

    def wait(self, seconds):
        if self.on_wait:
            self.on_wait()
        self.now += seconds
        return False


class MonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module('curriculum_monitor') if importlib.util.find_spec('curriculum_monitor') else None

    def setUp(self):
        self.assertIsNotNone(self.module, 'local scalar monitor is missing')

    def test_rollout_allowlist_excludes_credentials_source_and_trajectories(self):
        row = dict(transitions=32768, cumulative_timesteps=10059776, collection_seconds=4., update_seconds=1.,
                   steps_per_second_including_setup_logging=6553.6, learned_action_std=[.3]*10,
                   ppo_metrics={'train/approx_kl':.004, 'train/combat/value_loss':.2, 'unknown':10., 'train/loss':float('nan')},
                   api_key='secret', source='private-code', raw_action=[.1]*10, credential_metric=123.)
        result = self.module.extract_scalars('artifacts/E1-round1/rollout-00.json', row)
        self.assertEqual(result['E1/train/transitions'], 32768)
        self.assertEqual(result['E1/train/ppo/approx_kl'], .004)
        self.assertEqual(result['E1/train/action_std/7'], .3)
        self.assertFalse(any('secret' in key or 'source' in key or 'raw_action' in key or 'unknown' in key for key in result))
        self.assertTrue(all(type(value) in (int, float) for value in result.values()))
        json.dumps(result, allow_nan=False)

    def test_eval_summary_modes_and_attack_diagnostics_are_scalar_only(self):
        row = dict(task='combat', modes=dict(deterministic=dict(episodes=20, successes=18, falls=1, mean_ticks=65.),
                                             sampled=dict(episodes=20, successes=17, falls=2)),
                   episodes=[dict(mode='deterministic', attack=dict(started_instances=3, unique_hit_instances=2, first_hit_seconds=.8, raw_action='secret'))])
        result = self.module.extract_scalars('eval/E2-round1/summary.json', row)
        self.assertEqual(result['E2/eval/combat/deterministic/success_rate'], .9)
        self.assertEqual(result['E2/eval/combat/sampled/falls'], 2)
        self.assertEqual(result['E2/eval/combat/deterministic/mean_attack_starts'], 3.)
        self.assertEqual(result['E2/eval/combat/deterministic/mean_time_to_first_hit_seconds'], .8)
        row['deterministic'] = row.pop('modes')['deterministic']
        self.assertEqual(self.module.extract_scalars('eval/E2-round1/summary.json', row)['E2/eval/combat/deterministic/episodes'], 20)

    def test_deadline_dedup_verification_transition_and_persistent_run_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            block = root/'artifacts'/'E1-round1'
            block.mkdir(parents=True)
            record = block/'rollout-00.json'
            record.write_text(json.dumps(dict(transitions=32768, cumulative_timesteps=10059776)))
            sdk, clock = FakeSDK(), FakeClock()
            def acknowledge():
                inventory = dict(schema_version='curriculum-inventory-v1', block_id='E1-round1', files=[dict(path=record.name, sha256=hashlib.sha256(record.read_bytes()).hexdigest(), bytes=record.stat().st_size)])
                (block/'inventory.json').write_text(json.dumps(inventory))
                (block/'preservation_ack.json').write_text(json.dumps(dict(verified=True, block_id='E1-round1', inventory_sha256=hashlib.sha256((block/'inventory.json').read_bytes()).hexdigest())))
            clock.on_wait = acknowledge
            self.module.run_monitor(root, 145., sdk, clock=clock, wait=clock.wait)
            self.assertEqual(len(sdk.calls), 1)
            self.assertEqual(sdk.calls[0]['project'], 'utmist-ai2')
            self.assertEqual(sdk.calls[0]['group'], 'session13')
            self.assertFalse(sdk.calls[0]['save_code'])
            self.assertEqual([r['evidence/local_verified'] for r in sdk.run.logs], [0, 1])
            self.assertEqual(sdk.run.finished, [0])
            status = json.loads((root/'monitor'/'status.json').read_text())
            self.assertEqual(status['outcome'], 'deadline')
            self.assertEqual(status['pid'], os.getpid())
            self.assertEqual(len((root/'monitor'/'scalar-history.jsonl').read_text().splitlines()), 2)
            saved_id = json.loads((root/'monitor'/'run.json').read_text())['id']
            second = FakeSDK()
            self.module.run_monitor(root, 160., second, clock=clock, wait=clock.wait)
            self.assertEqual(second.calls[0]['id'], saved_id)
            self.assertEqual(second.run.logs, [])

    def test_partial_json_retries_without_leaking_contents_and_sigterm_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            block = root/'artifacts'/'E2-round1'
            block.mkdir(parents=True)
            record = block/'rollout-01.json'
            record.write_text('{"api_key":"secret-credential"')
            sdk, clock = FakeSDK(), FakeClock()
            def repair_then_stop():
                if clock.now == 100:
                    record.write_text(json.dumps(dict(transitions=32768)))
                else:
                    signal.raise_signal(signal.SIGTERM)
            clock.on_wait = repair_then_stop
            self.module.run_monitor(root, 200., sdk, clock=clock, wait=clock.wait)
            self.assertEqual(len(sdk.run.logs), 1)
            status = (root/'monitor'/'status.json').read_text()
            self.assertNotIn('secret-credential', status)
            self.assertEqual(json.loads(status)['outcome'], 'sigterm')
            self.assertEqual(sdk.run.finished, [0])

    def test_nested_guard_downloads_are_discovered_without_following_external_links(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            block = root/'guard'/'retrieved'/'artifacts'/'E1-round1'
            block.mkdir(parents=True)
            (block/'rollout-00.json').write_text('{"transitions":32768}')
            external = Path(outside)/'rollout-01.json'
            external.write_text('{"transitions":999}')
            (block/'rollout-01.json').symlink_to(external)
            (root/'outside').symlink_to(outside, target_is_directory=True)
            sdk, clock = FakeSDK(), FakeClock()
            self.module.run_monitor(root, 120., sdk, clock=clock, wait=clock.wait)
            self.assertEqual(sdk.run.logs, [{'E1/train/transitions':32768, 'evidence/local_verified':0}])
            record = json.loads((root/'monitor'/'scalar-history.jsonl').read_text())
            self.assertEqual(record['relative_path'], 'guard/retrieved/artifacts/E1-round1/rollout-00.json')

    def test_run_notes_describe_contact_only_lesson(self):
        notes = ' '.join(self.module.NOTES.values())
        for expected in ('bridge_b1_contact-v1', '0.25', '0.2', '10 seconds', 'one real light hit', 'both players'):
            self.assertIn(expected, notes)
        for obsolete in ('relocated', 'Two distinct real light hits', 'second hit', '60 ticks'):
            self.assertNotIn(obsolete, notes)

    def test_sdk_failure_closes_run_and_status_never_contains_error_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            block = root/'artifacts'/'calibration'
            block.mkdir(parents=True)
            (block/'rollout-00.json').write_text('{"transitions":32768}')
            sdk, clock = FakeSDK(fail_log=True), FakeClock()
            self.assertEqual(self.module.run_monitor(root, 140., sdk, clock=clock, wait=clock.wait), 1)
            self.assertEqual(sdk.run.finished, [1])
            status = (root/'monitor'/'status.json').read_text()
            self.assertNotIn('secret-credential', status)
            self.assertEqual(json.loads(status)['outcome'], 'error')

    def test_invalid_deadline_never_initializes_sdk(self):
        with tempfile.TemporaryDirectory() as folder:
            sdk, clock = FakeSDK(), FakeClock()
            for deadline in (float('nan'), float('inf'), 99.):
                with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                    self.module.run_monitor(Path(folder), deadline, sdk, clock=clock, wait=clock.wait)
            self.assertEqual(sdk.calls, [])

    def test_second_process_cannot_overwrite_active_monitor_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root, sdk, clock = Path(folder), FakeSDK(), FakeClock()
            second = FakeSDK()
            def attempt_second():
                before = (root/'monitor'/'status.json').read_bytes()
                with self.assertRaises(ValueError):
                    self.module.run_monitor(root, 180., second, clock=clock, wait=clock.wait)
                self.assertEqual((root/'monitor'/'status.json').read_bytes(), before)
            clock.on_wait = attempt_second
            self.module.run_monitor(root, 120., sdk, clock=clock, wait=clock.wait)
            self.assertEqual(second.calls, [])
            self.assertEqual(sdk.run.finished, [0])

    def test_finish_runs_even_if_outcome_summary_write_fails(self):
        class BrokenSummary(dict):
            def __setitem__(self, key, value):
                raise RuntimeError('secret-summary-error')
        with tempfile.TemporaryDirectory() as folder:
            sdk, clock = FakeSDK(), FakeClock()
            sdk.run.summary = BrokenSummary()
            self.module.run_monitor(Path(folder), 120., sdk, clock=clock, wait=clock.wait)
            self.assertEqual(sdk.run.finished, [1])

    def test_rollout_completed_episode_outcomes_and_hit_counts_are_available_before_eval(self):
        row = dict(transitions=32768, episodes=[dict(success=True, own_stock_loss=False, distinct_hit_ids=['a','b'], first_hit_tick=15),
                                               dict(success=False, own_stock_loss=True, distinct_hit_ids=[], first_hit_tick=None)])
        result = self.module.extract_scalars('artifacts/E1-round1/rollout-00.json', row)
        self.assertEqual(result.get('E1/train/completed_episodes'), 2)
        self.assertEqual(result.get('E1/train/success_rate'), .5)
        self.assertEqual(result.get('E1/train/falls'), 1)
        self.assertEqual(result.get('E1/train/mean_hits'), 1.)
        self.assertEqual(result.get('E1/train/mean_time_to_first_hit_seconds'), .5)

    def test_navigation_rollout_fall_key_is_counted(self):
        row = dict(transitions=32768, episodes=[dict(success=False, fall=True),
                                               dict(success=True, fall=False)])
        result = self.module.extract_scalars('guard/retrieved/artifacts/E2-round2/rollout-04.json', row)
        self.assertEqual(result['E2/train/falls'], 1)
        self.assertEqual(result['E2/train/fall_rate'], .5)


if __name__ == '__main__':
    unittest.main()
