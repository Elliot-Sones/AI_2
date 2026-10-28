import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from train_combat_curriculum import (
    PilotConfig, block_schedule, require_preserved, validate_parent,
    write_inventory, validate_deadline, atomic_json,
    expected_parent_block, validate_runtime,
    NavigationTraceWriter,
)


class RunnerContractTests(unittest.TestCase):
    def test_unqualified_task_refuses_before_cuda_or_model_work(self):
        from types import SimpleNamespace
        import time
        from train_combat_curriculum import run_block
        with patch('train_combat_curriculum.require_cuda') as cuda:
            with self.assertRaisesRegex((RuntimeError, ValueError), 'qualification'):
                run_block(SimpleNamespace(deadline_unix=time.time() + 1000,
                                          qualification=None, frozen_root=Path('/missing')))
            cuda.assert_not_called()

    def test_contact_qualification_binds_metric_game_and_adapter(self):
        import train_combat_curriculum as runner
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            game = root / 'UTMIST-AI2-main/environment/environment.py'
            game.parent.mkdir(parents=True)
            game.write_text('frozen game')
            proof = {'qualified': True, 'stage_version': 'bridge_b1_contact-v1',
                     'source_sha256': runner.sha256(game),
                     'adapter_source_sha256': runner.sha256(Path(runner.__file__).with_name('combat_curriculum.py'))}
            receipt = root / 'qualification.json'
            receipt.write_text(json.dumps(proof))
            self.assertEqual(runner.require_contact_qualification(receipt, root), proof)
            for change in ({'qualified': False}, {'stage_version': 'bridge_b1-v1'},
                           {'source_sha256': '0' * 64}, {'adapter_source_sha256': '0' * 64}):
                receipt.write_text(json.dumps({**proof, **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    runner.require_contact_qualification(receipt, root)

    def test_rollout_reseeds_loaded_vector_environment_and_rngs(self):
        import random
        import numpy as np
        import torch
        from train_combat_curriculum import seed_rollout
        env = Mock()
        seed_rollout(env, 20145)
        first = (random.random(), np.random.random(), torch.rand(1).item())
        seed_rollout(env, 20145)
        second = (random.random(), np.random.random(), torch.rand(1).item())
        self.assertEqual(first, second)
        self.assertEqual(env.seed.call_count, 2)
        env.seed.assert_called_with(20145)

    def test_finite_paired_schedule_counts_and_order(self):
        blocks = block_schedule()
        self.assertEqual([b['id'] for b in blocks], [
            'calibration', 'E1-round1', 'E2-round1', 'E2-round2',
            'E1-round2', 'E1-round3', 'E2-round3'])
        self.assertEqual(sum(b['transitions'] for b in blocks), 1015808)
        self.assertEqual(blocks[0]['tasks'], ['combat'])
        self.assertEqual(blocks[3]['tasks'], ['combat'] * 4 + ['wide'])
        self.assertEqual(blocks[-1]['tasks'], ['combat'] * 4 + ['near'])

    def test_geometry_is_fixed_and_finite(self):
        cfg = PilotConfig()
        self.assertEqual(cfg.n_envs * cfg.n_steps, 32768)
        self.assertEqual(cfg.batch_size, 8192)
        self.assertEqual(cfg.parent_steps, 10027008)
        self.assertEqual(cfg.learning_rate, 0.0001)
        with self.assertRaises(ValueError):
            PilotConfig(n_envs=0)

    def test_arm_lineage_does_not_follow_other_arm(self):
        self.assertEqual(expected_parent_block('E2-round1'), ('calibration', 10059776))
        self.assertEqual(expected_parent_block('E1-round2'), ('E1-round1', 10223616))
        self.assertEqual(expected_parent_block('E2-round3'), ('E2-round2', 10387456))

    def test_runtime_rejects_saved_hyperparameter_drift(self):
        from types import SimpleNamespace
        cfg = PilotConfig()
        model = SimpleNamespace(n_envs=64, n_steps=512, batch_size=8192, n_epochs=4,
                    gamma=.99, gae_lambda=.95, vf_coef=.5, max_grad_norm=.5,
                    target_kl=.01, lr_schedule=lambda _: .0001, clip_range=lambda _: .1,
                    policy=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{'lr': .0001}])))
        validate_runtime(model, cfg)
        model.n_steps = 256
        with self.assertRaises(ValueError):
            validate_runtime(model, cfg)

    def test_parent_hash_failure_never_silently_starts_fresh(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder) / 'parent.zip'
            parent.write_bytes(b'not the parent')
            with self.assertRaises(ValueError):
                validate_parent(parent)

    def test_deadline_fails_closed(self):
        for deadline in (0, float('nan'), float('inf'), 150):
            with self.subTest(deadline=deadline), self.assertRaises((ValueError, TimeoutError)):
                validate_deadline(deadline, now=100, reserve_seconds=60)
        validate_deadline(200, now=100, reserve_seconds=60)

    def test_inventory_and_ack_require_exact_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'final.zip').write_bytes(b'checkpoint')
            (root / 'traces').mkdir()
            (root / 'traces' / 'inventory.json').write_text('{}')
            inventory = write_inventory(root, block_id='E1-round1')
            self.assertEqual(len(inventory['files']), 2)
            self.assertEqual(inventory['files'][0]['path'], 'final.zip')
            self.assertEqual(inventory['files'][0]['sha256'], hashlib.sha256(b'checkpoint').hexdigest())
            ack = root / 'local-verified.json'
            with self.assertRaises(FileNotFoundError):
                require_preserved(root, ack)
            atomic_json(ack, {'block_id': 'E1-round1', 'verified': True, 'inventory_sha256': 'bad'})
            with self.assertRaises(ValueError):
                require_preserved(root, ack)
            ack.unlink()
            atomic_json(ack, {'block_id': 'E1-round1', 'verified': True,
                              'inventory_sha256': hashlib.sha256((root / 'inventory.json').read_bytes()).hexdigest()})
            require_preserved(root, ack)

    def test_atomic_json_rejects_overwrite_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'record.json'
            with self.assertRaises(ValueError):
                atomic_json(target, {'value': float('nan')})
            self.assertFalse(target.exists())
            atomic_json(target, {'ok': True})
            self.assertEqual(json.loads(target.read_text()), {'ok': True})
            with self.assertRaises(FileExistsError):
                atomic_json(target, {'ok': False})

    def test_navigation_trace_is_atomic_and_keeps_coordinates(self):
        import gzip
        with tempfile.TemporaryDirectory() as folder:
            writer = NavigationTraceWriter(Path(folder))
            frame = {'tick': 1, 'raw_action': [0.] * 10,
                     'post_players': [{'position': [1, 2]}, {'position': [3, 4]}]}
            writer.append(frame)
            self.assertFalse((Path(folder) / 'navigation.jsonl.gz').exists())
            writer.close()
            with gzip.open(Path(folder) / 'navigation.jsonl.gz', 'rt') as handle:
                row = json.loads(handle.readline())
            self.assertEqual(row['post_players'][1]['position'], [3, 4])
            self.assertEqual(row['schema_version'], 'navigation-trace-v1')


if __name__ == '__main__':
    unittest.main()
