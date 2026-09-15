"""Contract-only local tests. No local policy construction or CPU training."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class ContactRunnerTests(unittest.TestCase):
    def subject(self):
        self.assertIsNotNone(importlib.util.find_spec('benchmark_native_contact_training'),
                             'native contact benchmark runner has not been implemented')
        import benchmark_native_contact_training
        return benchmark_native_contact_training

    def test_geometry_keeps_one_learner_per_game_and_only_batch_changes(self):
        m = self.subject()
        small = m.ContactBenchmarkConfig()
        large = replace(small, batch_size=16384)
        self.assertEqual(small.transitions_per_rollout, 32768)
        self.assertEqual(small.game_ticks_per_rollout, small.transitions_per_rollout)
        self.assertEqual(small.nominal_optimizer_steps, 16)
        self.assertEqual(large.nominal_optimizer_steps, 8)
        for changes in ({'device': 'cpu'}, {'lanes': 32}, {'n_steps': 128},
                        {'batch_size': 4096}, {'n_epochs': 8}, {'rollouts': 0},
                        {'seed': -1}, {'action_head': 'factored'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(small, **changes)

    def test_deadline_finite_expired_and_reserve(self):
        m = self.subject()
        for deadline in (float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError):
                m.validate_deadline(deadline, now=100)
        with self.assertRaises(TimeoutError):
            m.validate_deadline(200, now=100, reserve_seconds=120)
        m.validate_deadline(300, now=100, reserve_seconds=120)

    def test_checkpoint_rejects_wrong_bytes_without_loading_weights(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.zip'
            path.touch()
            with self.assertRaisesRegex(ValueError, 'canonical calibrated checkpoint'):
                m.validate_checkpoint(path)

    def test_checkpoint_preservation_requires_matching_ack_and_listed_bytes(self):
        m = self.subject()
        self.assertTrue(hasattr(m, 'require_checkpoint_preservation'))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'final.zip'
            checkpoint.write_bytes(b'already preserved calibration')
            m.write_inventory(root, block_id='calibration')
            ack = root / 'preservation_ack.json'
            ack.write_text(json.dumps({'verified': True, 'block_id': 'calibration',
                'inventory_sha256': m.sha256(root / 'inventory.json')}))
            result = m.require_checkpoint_preservation(checkpoint)
            self.assertEqual(result['checkpoint_sha256'], m.sha256(checkpoint))
            ack.write_text(json.dumps({'verified': False}))
            with self.assertRaises(ValueError):
                m.require_checkpoint_preservation(checkpoint)

    def test_qualification_binds_every_source_and_all_three_gates(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            receipt = Path(directory) / 'proof.json'
            sources = {'native/contact_episode.h': 'a' * 64, 'reference/combat_curriculum.py': 'b' * 64}
            artifact = Path(directory) / 'test-evidence.json'
            artifact.write_text('{"passed":true}')
            proof = {'schema_version': m.QUALIFICATION_SCHEMA, 'qualified': True,
                     'stage_version': m.CONTACT_VERSION, 'source_sha256': sources,
                     'checkpoint_sha256': m.CHECKPOINT_SHA256,
                     'evidence': [{'check': check, 'path': artifact.name, 'sha256': m.sha256(artifact)}
                                  for check in m.CHECKS],
                     'checks': {'original_cpu_exact': True, 'cpu_cuda_exact': True,
                                'learner_update_exact': True}}
            with patch.object(m, 'qualification_sources', return_value=sources):
                for change in ({'qualified': False}, {'checks': {'original_cpu_exact': True}},
                               {'source_sha256': {**sources, 'native/contact_episode.h': 'c' * 64}},
                               {'stage_version': 'bridge_b1-v1'}):
                    receipt.write_text(json.dumps({**proof, **change}))
                    with self.assertRaises(ValueError):
                        m.require_qualification(receipt, Path(directory), Path(directory))
                receipt.write_text(json.dumps(proof))
                self.assertEqual(m.require_qualification(receipt, Path(directory), Path(directory)), proof)
                artifact.write_text('{"passed":false}')
                with self.assertRaisesRegex(ValueError, 'evidence'):
                    m.require_qualification(receipt, Path(directory), Path(directory))

    def test_qualification_hashes_actual_observation_action_space_json(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'UTMIST-AI2-main/environment').mkdir(parents=True)
            (root / 'UTMIST-AI2-main/environment/environment.py').touch()
            (root / 'train.py').touch()
            (root / 'tracking.py').touch()
            for name in m.REFERENCE_FILES:
                (root / name).touch()
            sources = m.qualification_sources(root, root)
            self.assertIn('native/navigation_spaces.json', sources)
            self.assertEqual(sources['native/navigation_spaces.json'],
                             m.sha256(m.ROOT / 'native/navigation_spaces.json'))

    def test_timing_uses_ratio_of_totals_not_average_rates(self):
        m = self.subject()
        rows = [{'transitions': 32768, 'game_ticks': 32768, 'collection_seconds': 1.,
                 'update_seconds': 1., 'total_seconds': 2.},
                {'transitions': 32768, 'game_ticks': 32768, 'collection_seconds': 1.,
                 'update_seconds': 3., 'total_seconds': 4.}]
        result = m.summarize_rollouts(rows)
        self.assertEqual(result['transitions'], 65536)
        self.assertEqual(result['training_steps_per_second'], 65536 / 6)
        self.assertEqual(result['collection_steps_per_second'], 65536 / 2)
        self.assertEqual(result['learner_transitions_per_game_tick'], 1)
        self.assertFalse(result['mastery_established'])

    def test_inventory_is_immutable_and_rejects_unfinished_artifacts(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            m.atomic_json(root / 'result.json', {'a': 1})
            with self.assertRaises(FileExistsError):
                m.atomic_json(root / 'result.json', {'a': 2})
            pending = root / 'checkpoint.tmp'
            pending.touch()
            with self.assertRaises(ValueError):
                m.write_inventory(root, block_id='test')
            pending.unlink()
            inventory = m.write_inventory(root, block_id='test')
            self.assertEqual(inventory['schema_version'], 'curriculum-inventory-v1')
            self.assertEqual(inventory['files'][0]['path'], 'result.json')

    def test_output_reuse_rejected_before_hardware_or_checkpoint_load(self):
        m = self.subject()
        with TemporaryDirectory() as directory, patch.object(m, 'require_cuda') as cuda:
            args = SimpleNamespace(output=Path(directory))
            with self.assertRaises(FileExistsError):
                m.run_block(args)
            cuda.assert_not_called()

    def test_actual_update_work_must_be_reported(self):
        m = self.subject()
        cfg = m.ContactBenchmarkConfig()
        good = dict(attempted_epochs=2, optimizer_steps=5, kl_early_stop=True)
        self.assertEqual(m.validate_update_diagnostics(good, cfg), good)
        for bad in ({}, dict(good, optimizer_steps=17), dict(good, kl_early_stop=1),
                    dict(attempted_epochs=4, optimizer_steps=15, kl_early_stop=False)):
            with self.assertRaises(ValueError):
                m.validate_update_diagnostics(bad, cfg)

    def test_action_press_metrics_report_sample_executed_and_deterministic_rates(self):
        m = self.subject()
        import torch
        mask = torch.tensor([True, True, True, True, True, False, True, True, False, False])

        class Policy:
            action_head = 'bernoulli_10'
            allowed_control_mask = mask
            def __call__(self, obs, deterministic=False):
                actions = torch.zeros((obs.shape[0], 10), dtype=torch.float32)
                actions[:, 0] = 1
                actions[:, 7] = 1 if deterministic else 0
                return actions, None, None
            def decode_actions(self, actions):
                return actions * mask.to(dtype=actions.dtype)

        buffer = SimpleNamespace(
            actions=torch.tensor([[[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]]], dtype=torch.float32),
            executed_actions=torch.tensor([[[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]]], dtype=torch.float32),
            observations=torch.zeros((1, 1, 256), dtype=torch.float32),
        )
        metrics = m.action_press_metrics(SimpleNamespace(policy=Policy(), rollout_buffer=buffer))
        self.assertTrue(metrics['action_metrics_available'])
        self.assertEqual(metrics['action_head'], 'bernoulli_10')
        self.assertAlmostEqual(metrics['policy_sample_press_rate_all'], .2)
        self.assertAlmostEqual(metrics['executed_press_rate_all'], .2)
        self.assertAlmostEqual(metrics['deterministic_press_rate_active'], 2 / 7)

    def test_collector_episode_records_are_not_silently_dropped(self):
        m = self.subject()
        self.assertTrue(hasattr(m, 'completed_episode_infos'), 'callback needs explicit native infos contract')
        info = {'episode': {'r': 1., 'l': 200}, 'contact': {'diagnostics': {'outcome': 'success'}}}
        self.assertEqual(m.completed_episode_infos({'infos': [info]}), [info])
        with self.assertRaises(RuntimeError):
            m.completed_episode_infos({'drained_infos': [info]})
        with self.assertRaises(RuntimeError):
            m.completed_episode_infos({'infos': [{'episode': {'r': 1., 'l': 1}}]})

    def test_parser_requires_qualification_and_finite_deadline(self):
        m = self.subject()
        args = m.parse_args(['--checkpoint', 'a.zip', '--frozen-root', 'oracle',
            '--reference-root', 'reference', '--qualification', 'proof.json', '--output', 'out',
            '--deadline-unix', '2000', '--batch-size', '16384', '--rollouts', '8',
            '--action-head', 'bernoulli_10'])
        self.assertEqual(args.batch_size, 16384)
        self.assertEqual(args.rollouts, 8)
        self.assertEqual(args.action_head, 'bernoulli_10')
        self.assertEqual(args.seed, 145)

    def test_complete_mocked_block_preserves_each_rollout_and_actual_work(self):
        self.exercise_mocked_block()

    def test_failed_second_checkpoint_preserves_completed_and_incomplete_artifacts(self):
        self.exercise_mocked_block(fail_second_save=True)

    def exercise_mocked_block(self, *, fail_second_save=False):
        """Exercise runner orchestration only; no actual policy exists in this test."""
        m = self.subject()
        import torch
        import native_curriculum_ppo
        import tensor_ppo
        from types import ModuleType
        lane_module = ModuleType('native_contact_lane_vec_env')
        cfg = m.ContactBenchmarkConfig(rollouts=2)
        backend = SimpleNamespace(valid_steps=np.zeros(64, dtype=np.uint64), errors=np.zeros(64),
                                  scenario_metadata={'generator': 'test-only'})
        closed = []
        raw = SimpleNamespace(backend=backend, close=lambda: closed.append(True))
        lane_module.build_contact_env = lambda *args, **kwargs: raw
        losses = {key: .1 for key in ('train/loss', 'train/policy_gradient_loss',
                                      'train/value_loss', 'train/entropy_loss')}
        policy = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type='cuda'))])
        model = SimpleNamespace(**{name: getattr(cfg, name) for name in ('n_steps', 'batch_size',
            'n_epochs', 'gamma', 'gae_lambda', 'target_kl', 'ent_coef')}, n_envs=64,
            num_timesteps=m.CHECKPOINT_STEPS, task='combat', calibration=False, arm='E1',
            action_head='gaussian', policy=policy,
            lr_schedule=lambda _: .0001, clip_range=lambda _: .1, env=raw, rollout_buffer=object(),
            logger=SimpleNamespace(name_to_value=losses, dump=lambda **_: None),
            last_update_diagnostics={'attempted_epochs': 4, 'optimizer_steps': 16, 'kl_early_stop': False})
        model.set_logger = lambda logger: None
        callback = SimpleNamespace(episodes=[{'contact': {'diagnostics': {'outcome': 'timeout'}}}],
                                   on_training_start=lambda *args: None, on_training_end=lambda: None)
        def setup(total, **kwargs):
            model._total_timesteps = model.num_timesteps + total
            return model._total_timesteps, callback
        def collect(*args):
            backend.valid_steps += cfg.n_steps
            model.num_timesteps += cfg.transitions_per_rollout
            return True
        model._setup_learn = setup
        model.collect_rollouts = collect
        model._update_current_progress_remaining = lambda *args: None
        model.train = lambda: None
        def save(path):
            Path(path).write_bytes(b'mocked checkpoint, not policy weights')
            if fail_second_save and path.name == 'rollout-001.tmp.zip':
                raise TimeoutError('injected interrupted second checkpoint')
        model.save = save
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proof = root / 'proof.json'
            proof.write_text('{}')
            args = SimpleNamespace(output=root / 'run', batch_size=8192, rollouts=2, seed=145,
                device='cuda', update_mode='deferred_metrics', action_head='gaussian', deadline_unix=10**12,
                checkpoint=root / 'calibrated.zip', qualification=proof, reference_root=root, frozen_root=root)
            with patch.object(m, 'validate_checkpoint', return_value={'sha256': m.CHECKPOINT_SHA256}), \
                 patch.object(m, 'require_checkpoint_preservation', return_value={'verified': True}), \
                 patch.object(m, 'require_qualification', return_value={}), \
                 patch.object(m, 'require_cuda', return_value=torch.device('cuda:0')), \
                 patch.dict('sys.modules', native_contact_lane_vec_env=lane_module), \
                 patch.object(native_curriculum_ppo.NativeCurriculumPPO, 'load_contact_checkpoint', return_value=model), \
                 patch.object(tensor_ppo, 'TensorVecEnv', side_effect=lambda env, **kwargs: env), \
                 patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'get_device_name', return_value='mock'), \
                 patch.object(torch, 'manual_seed'), patch.object(torch, 'set_num_threads'), \
                 patch('builtins.print'), \
                 patch.dict('os.environ', {}, clear=False):
                if fail_second_save:
                    with self.assertRaisesRegex(TimeoutError, 'interrupted second'):
                        m.run_block(args)
                else:
                    result = m.run_block(args)
            if not fail_second_save:
                self.assertEqual(result['transitions'], 65536)
                self.assertEqual(result['status'], 'trained_not_evaluated')
                self.assertEqual(result['rollouts'][0]['episode_count'], 1)
            self.assertEqual(len(closed), 1)
            self.assertTrue((args.output / 'inventory.json').is_file(),
                            'failed attempts also need retrievable complete inventories')
            inventory = json.loads((args.output / 'inventory.json').read_text())
            paths = {row['path'] for row in inventory['files']}
            if fail_second_save:
                self.assertTrue({'rollout-000.zip', 'failure.json', 'incomplete-artifacts.json'} <= paths)
                self.assertNotIn('final.zip', paths)
                self.assertNotIn('rollout-001.zip', paths)
                failure = json.loads((args.output / 'failure.json').read_text())
                self.assertEqual(failure['last_completed_checkpoint'], 'rollout-000.zip')
                mapping = json.loads((args.output / 'incomplete-artifacts.json').read_text())['artifacts']
                self.assertEqual(mapping[0]['original_path'], 'rollout-001.tmp.zip')
                self.assertFalse(mapping[0]['valid_checkpoint'])
                self.assertNotIn('.tmp', mapping[0]['preserved_path'])
            else:
                self.assertTrue({'rollout-000.zip', 'rollout-001.zip', 'final.zip', 'result.json'} <= paths)
            actual = {p.relative_to(args.output).as_posix() for p in args.output.rglob('*') if p.is_file()}
            self.assertEqual(paths | {'inventory.json'}, actual)
            for item in inventory['files']:
                path = args.output / item['path']
                self.assertEqual(item['bytes'], path.stat().st_size)
                self.assertEqual(item['sha256'], m.sha256(path))
                self.assertNotIn('.tmp', item['path'])


if __name__ == '__main__':
    unittest.main()
