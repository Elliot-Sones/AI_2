"""Offline validator tests: fake policy/CUDA boundaries, never CPU inference."""
from contextlib import chdir, nullcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


class ArrayTensor:
    """Minimal transport double; no neural network or torch computation."""
    def __init__(self, value, device='cpu'):
        self.value = np.asarray(value, dtype=np.float32)
        self.device = device
        self.is_cuda = str(device).startswith('cuda')
        self.dtype = 'float32'
        self.shape = self.value.shape

    def to(self, device):
        return ArrayTensor(self.value.copy(), device)

    def detach(self):
        return self

    def cpu(self):
        return self.to('cpu')

    def numpy(self):
        return self.value

    def contiguous(self):
        return self


def observation(x=0.0):
    value = np.zeros((2, 256), dtype=np.float32)
    value[:, 192:194] = [[x, 1.], [-x, 2.]]
    value[:, 205] = 3
    return value


def benchmark(checkpoint):
    return dict(schema='ai2.raw_combat_training_benchmark.v1', status='passed',
                config=dict(seed=0, games=8),
                backends=[dict(backend='native_cuda', status='passed',
                               final_policy=dict(sha256='a'*64, parameter_count=100),
                               checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())])


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.validation.validate_native_combat_checkpoint'),
                             'checkpoint validator has not been implemented')
        self.mod = importlib.import_module('tools.validation.validate_native_combat_checkpoint')

    def test_config_is_cuda_only_bounded_and_distinct(self):
        config = self.mod.ValidationConfig()
        self.assertEqual(config.seeds, (3, 7, 101))
        self.assertEqual(config.ticks, 256)
        for kwargs in ({'device': 'cpu'}, {'device': 'auto'}, {'ticks': 0},
                       {'ticks': True}, {'ticks': 257}, {'seeds': ()}, {'seeds': (3, 3)},
                       {'seeds': (3, 7, 101, 103)},
                       {'seeds': (True,)}, {'seeds': (2**32-1,)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.mod.ValidationConfig(**kwargs)

    def test_backend_binding_rejects_ambiguous_failed_or_nonheldout(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'policy.zip'
            checkpoint.write_bytes(b'synthetic zip stand-in; never loaded')
            value = benchmark(checkpoint)
            row = self.mod.expected_backend(value, 'native_cuda', (3, 7, 101))
            self.assertEqual(row['final_policy']['sha256'], 'a'*64)
            with self.assertRaisesRegex(ValueError, 'held.out'):
                self.mod.expected_backend(value, 'native_cuda', (2,))
            value['backends'].append(value['backends'][0].copy())
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                self.mod.expected_backend(value, 'native_cuda', (3,))
            value['backends'].pop()
            value['backends'][0]['status'] = 'failed'
            with self.assertRaisesRegex(ValueError, 'passed'):
                self.mod.expected_backend(value, 'native_cuda', (3,))

    def test_trace_uses_latest_own_player_fields_and_terminal_flags(self):
        before, after = observation(1.), observation(4.)
        after[:, :2] = 999  # Oldest frame must never be interpreted as current.
        after[:, 204] = [.25, .5]
        after[1, 205] = 2
        raw = np.full((2, 10), 2., dtype=np.float32)
        row = self.mod.trace_row(3, 1, before, after, raw, np.clip(raw, -1, 1),
                                 [0., -1.], [True, True], [False, False], [{'episode': {'l': 1}}])
        self.assertEqual(row['after'][0]['position_xy'], [4., 1.])
        self.assertEqual(row['after'][1]['position_xy'], [-4., 2.])
        self.assertEqual(row['damage_delta_from_obs'], [175., 350.])
        self.assertEqual(row['stock_delta'], [0, -1])
        self.assertEqual(row['stock_lost'], [False, True])
        self.assertEqual(row['done'], [True, True])
        self.assertEqual(row['raw_actions'][0], [2.]*10)
        self.assertEqual(row['actions'][0], [1.]*10)
        self.assertEqual(row['episode_infos'], [{'episode': {'l': 1}}])
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.mod.player_fields(np.full((2, 256), np.nan))
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.mod.player_fields(np.zeros((1, 256)))

    def test_action_forward_stays_cuda_then_clips_for_cpu_simulation(self):
        raw = np.tile(np.linspace(-2, 2, 10, dtype=np.float32), (2, 1))
        policy = Mock(return_value=(ArrayTensor(raw, 'cuda:0'), None, None))
        torch = SimpleNamespace(float32='float32', inference_mode=nullcontext,
                                isfinite=lambda value: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: bool(np.isfinite(value.value).all()))))
        space = SimpleNamespace(shape=(10,), low=np.full(10, -1.), high=np.full(10, 1.))
        clipped, sampled = self.mod.cuda_actions(policy, ArrayTensor(observation()), space,
                                                 'cuda:0', torch_module=torch)
        self.assertEqual(policy.call_args.args[0].device, 'cuda:0')
        self.assertTrue(policy.call_args.kwargs['deterministic'])
        np.testing.assert_array_equal(sampled, raw)
        np.testing.assert_array_equal(clipped.numpy(), np.clip(raw, -1, 1))
        self.assertEqual(clipped.device, 'cpu')
        policy.return_value = (ArrayTensor(raw), None, None)
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            self.mod.cuda_actions(policy, ArrayTensor(observation()), space, 'cuda:0', torch_module=torch)
        policy.return_value = (ArrayTensor(np.full((2, 10), np.nan), 'cuda:0'), None, None)
        with self.assertRaisesRegex(RuntimeError, 'finite'):
            self.mod.cuda_actions(policy, ArrayTensor(observation()), space, 'cuda:0', torch_module=torch)
        policy.reset_mock()
        policy.return_value = (ArrayTensor(raw, 'cuda:0'), None, None)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.mod.cuda_actions(policy, ArrayTensor(np.full((2, 256), np.nan)), space, 'cuda:0', torch_module=torch)
        policy.assert_not_called()

    def test_direct_loader_checks_cuda_before_deserialization_and_loads_standard_ppo(self):
        from ai2.simulation.native_lane_vec_env import frozen_spaces
        obs_space, act_space = frozen_spaces()
        policy = SimpleNamespace(squash_output=False)
        ppo = SimpleNamespace(load=Mock(return_value=SimpleNamespace(policy=policy,
                              observation_space=obs_space, action_space=act_space)))
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            self.mod.load_cuda_policy(Path('not-opened.zip'), 'cpu', ppo_class=ppo)
        ppo.load.assert_not_called()
        with patch.object(self.mod, 'require_cuda', return_value='cuda:0'):
            self.assertIs(self.mod.load_cuda_policy(Path('fake.zip'), 'cuda', ppo_class=ppo), policy)
            ppo.load.assert_called_once_with('fake.zip', device='cuda:0')
            policy.squash_output = True
            with self.assertRaisesRegex(ValueError, 'unsquashed'):
                self.mod.load_cuda_policy(Path('fake.zip'), 'cuda', ppo_class=ppo)

    def test_missing_cuda_fails_before_any_checkpoint_load_or_env_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            with patch.object(self.mod, 'require_cuda', side_effect=RuntimeError('CUDA unavailable')), \
                 patch.object(self.mod, 'load_cuda_policy') as load, \
                 patch.object(self.mod, 'PythonGameCPUEnv') as environment:
                with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                    self.mod.run_validation(self.mod.ValidationConfig(), base/'missing.zip',
                                            base/'result.json', base/'frozen', base/'out')
                load.assert_not_called()
                environment.assert_not_called()
            summary = json.loads((base/'out/summary.json').read_text())
            self.assertEqual(summary['status'], 'failed')
            self.assertIn('CUDA unavailable', summary['error'])

    def test_mocked_end_to_end_preserves_checkpoint_and_checks_final_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            checkpoint = base/'policy.zip'
            checkpoint.write_bytes(b'never interpreted as a real policy')
            result = base/'benchmark.json'
            result.write_text(json.dumps(benchmark(checkpoint)))
            (base/'frozen').mkdir()
            terminal = observation(4.)
            terminal[1, 205] = 2
            step = SimpleNamespace(observations=ArrayTensor(observation(99.)),
                                   terminal_observations=ArrayTensor(terminal),
                                   rewards=ArrayTensor([0, -1]), terminated=ArrayTensor([1, 1]),
                                   truncated=ArrayTensor([0, 0]))
            env = SimpleNamespace(reset_tensor=Mock(return_value=ArrayTensor(observation(1.))),
                                  step_tensor=Mock(return_value=step), close=Mock(side_effect=lambda: os.chdir(base)),
                                  drain_episode_infos=Mock(return_value=[{'episode': {'l': 1}}]),
                                  num_envs=2, observation_space=SimpleNamespace(shape=(256,)),
                                  action_space=SimpleNamespace(shape=(10,)))
            policy = SimpleNamespace(set_training_mode=Mock())
            receipt = dict(sha256='a'*64, parameter_count=100)

            def make_env(*args, **kwargs):
                # Real original_game keeps cwd at frozen_root until close().
                os.chdir(base/'frozen')
                return env

            with patch.object(self.mod, 'require_cuda', return_value='cuda:0'), \
                 patch.object(self.mod, 'source_receipt', return_value={'environment_sha256': 'b'*64}), \
                 patch.object(self.mod, 'load_cuda_policy', return_value=policy), \
                 patch.object(self.mod, 'policy_receipt', return_value=receipt), \
                 patch.object(self.mod, 'PythonGameCPUEnv', side_effect=make_env), \
                 patch.object(self.mod, 'cuda_actions', return_value=(ArrayTensor(np.zeros((2, 10))), np.zeros((2, 10)))), \
                 chdir(base):
                summary = self.mod.run_validation(self.mod.ValidationConfig(seeds=(3,), ticks=2),
                                                 Path('policy.zip'), Path('benchmark.json'), Path('frozen'), Path('out'))
            self.assertEqual(summary['status'], 'passed')
            self.assertEqual(summary['game_ticks'], 1)
            self.assertEqual(summary['player_transitions'], 2)
            self.assertEqual(summary['claim'], 'infrastructure validation only; not evidence of strong combat play')
            rows = [json.loads(line) for line in (base/'out/seed-3.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['after'][0]['position_xy'], [4., 1.])
            self.assertEqual(env.step_tensor.call_count, 1)
            env.close.assert_called_once()
            env.reset_tensor.assert_called_once_with([3, 4])
            self.assertEqual(checkpoint.read_bytes(), b'never interpreted as a real policy')
            inventory = json.loads((base/'out/artifacts.json').read_text())
            for name, digest in inventory['sha256'].items():
                self.assertEqual(hashlib.sha256((base/'out'/name).read_bytes()).hexdigest(), digest)
            with self.assertRaisesRegex(ValueError, 'new'):
                self.mod.run_validation(self.mod.ValidationConfig(), checkpoint, result, base/'frozen', base/'out')

    def test_mismatched_loaded_parameters_fail_before_simulation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            checkpoint = base/'policy.zip'
            checkpoint.write_bytes(b'not loaded')
            result = base/'benchmark.json'
            result.write_text(json.dumps(benchmark(checkpoint)))
            with patch.object(self.mod, 'require_cuda', return_value='cuda:0'), \
                 patch.object(self.mod, 'source_receipt', return_value={}), \
                 patch.object(self.mod, 'load_cuda_policy', return_value=object()), \
                 patch.object(self.mod, 'policy_receipt', return_value=dict(sha256='c'*64, parameter_count=100)), \
                 patch.object(self.mod, 'PythonGameCPUEnv') as environment:
                with self.assertRaisesRegex(RuntimeError, 'parameter hash'):
                    self.mod.run_validation(self.mod.ValidationConfig(), checkpoint, result, base/'frozen', base/'out')
                environment.assert_not_called()
            self.assertEqual(json.loads((base/'out/summary.json').read_text())['status'], 'failed')

    def test_changed_checkpoint_bytes_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            checkpoint = base/'policy.zip'
            checkpoint.write_bytes(b'original')
            result = base/'benchmark.json'
            result.write_text(json.dumps(benchmark(checkpoint)))
            checkpoint.write_bytes(b'corrupted')
            with patch.object(self.mod, 'require_cuda', return_value='cuda:0'), \
                 patch.object(self.mod, 'load_cuda_policy') as load:
                with self.assertRaisesRegex(RuntimeError, 'file hash'):
                    self.mod.run_validation(self.mod.ValidationConfig(), checkpoint, result, base/'frozen', base/'out')
                load.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b'corrupted')


if __name__ == '__main__':
    unittest.main()
