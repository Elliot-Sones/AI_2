"""Driver tests perform no policy inference or training on the local CPU."""
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, call, patch

import numpy as np


FROZEN = Path(os.environ.get('AI2_FROZEN_SOURCE_ROOT', '/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected'))


class ConfigAndMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module('benchmark_native_combat_training')

    def test_config_counts_games_and_player_transitions_separately(self):
        config = self.mod.BenchmarkConfig(games=3, n_steps=8, batch_size=16,
                                          warmup_rollouts=1, measured_rollouts=2)
        self.assertEqual(config.game_ticks_per_rollout, 24)
        self.assertEqual(config.player_transitions_per_rollout, 48)
        for kwargs in ({'games': True}, {'n_steps': 0}, {'batch_size': 3},
                       {'net_arch': ()}, {'net_arch': (0,)}, {'device': 'cpu'},
                       {'python_workers': 0}, {'measured_rollouts': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.mod.BenchmarkConfig(**kwargs)

    def test_update_mode_is_default_off_and_rejects_unknown_values(self):
        defaults = asdict(self.mod.BenchmarkConfig())
        self.assertIn('update_mode', defaults)
        self.assertEqual(defaults['update_mode'], 'sb3')
        selected = asdict(self.mod.BenchmarkConfig(update_mode='deferred_metrics'))
        self.assertEqual(selected.pop('update_mode'), 'deferred_metrics')
        defaults.pop('update_mode')
        self.assertEqual(selected, defaults, 'update selection must not change the PPO workload')
        for value in ('compiled', '', None, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'update_mode'):
                self.mod.BenchmarkConfig(update_mode=value)

    def test_cli_propagates_only_the_selected_update_mode(self):
        with patch.object(self.mod, 'run_benchmark') as run:
            self.assertEqual(self.mod.main(['--frozen-root', str(FROZEN), '--output', '/tmp/combat-cli',
                                           '--update-mode', 'deferred_metrics']), 0)
        config, output, frozen = run.call_args.args
        self.assertEqual(config.update_mode, 'deferred_metrics')
        self.assertEqual(config, self.mod.BenchmarkConfig(update_mode='deferred_metrics'))
        self.assertEqual((output, frozen), (Path('/tmp/combat-cli'), FROZEN))

    def test_update_selection_preserves_original_class_and_routes_actual_buffer(self):
        from tensor_ppo import TensorPPO
        from ppo_update_optimizations import SB3_VERSION, SB3_TRAIN_SHA256
        self.assertTrue(hasattr(self.mod, 'make_benchmark_ppo_class'))
        self.assertIs(self.mod.make_benchmark_ppo_class(self.mod.BenchmarkConfig()), TensorPPO)
        supported = dict(version=SB3_VERSION, train_sha256=SB3_TRAIN_SHA256, supported=True)
        # Isolate the routing boundary: no constructor, forward pass, or update executes locally.
        with patch('ppo_update_optimizations.sb3_train_source_metadata', return_value=supported):
            selected = self.mod.make_benchmark_ppo_class(self.mod.BenchmarkConfig(update_mode='deferred_metrics'))
        self.assertTrue(issubclass(selected, TensorPPO))
        model = object.__new__(selected)
        buffer = model.rollout_buffer = object()
        with patch('ppo_update_optimizations.deferred_metrics_train') as update, \
             patch.object(TensorPPO, 'train', side_effect=AssertionError('stats proxy must not be installed')):
            selected.train(model)
        update.assert_called_once_with(model, loss_function=None)
        self.assertIs(model.rollout_buffer, buffer)

    def test_deferred_source_guard_rejects_before_environment_creation(self):
        self.assertTrue(hasattr(self.mod, 'make_benchmark_ppo_class'))
        config = self.mod.BenchmarkConfig(update_mode='deferred_metrics')
        unsupported = dict(version='unsupported-test-version', train_sha256='bad-source', supported=False)
        with patch('ppo_update_optimizations.sb3_train_source_metadata', return_value=unsupported), \
             patch.object(self.mod, 'build_backend') as build:
            with self.assertRaisesRegex(RuntimeError, 'unsupported-test-version'):
                self.mod.run_backend('native_cuda', config, Path('/tmp/no-run'), FROZEN, 'cuda:0')
        build.assert_not_called()

    def test_provenance_records_update_implementation_and_source_guard(self):
        if not FROZEN.is_dir():
            self.skipTest('frozen original game not installed')
        from ppo_update_optimizations import SB3_VERSION, SB3_TRAIN_SHA256, sb3_train_source_metadata
        actual_source = sb3_train_source_metadata()
        for mode in ('sb3', 'deferred_metrics'):
            with self.subTest(mode=mode), patch('torch.cuda.get_device_name', return_value='metadata-only test'):
                config = self.mod.BenchmarkConfig(update_mode=mode)
                result = self.mod.provenance(config, FROZEN, 'cuda:0')
                self.assertEqual(result['config']['update_mode'], mode)
                update = result['ppo_update']
                self.assertEqual(update['mode'], mode)
                self.assertEqual(update['source_guard'], dict(required=mode == 'deferred_metrics',
                    expected_version=SB3_VERSION, expected_train_sha256=SB3_TRAIN_SHA256, **actual_source))
                for name in ('ppo_update_optimizations.py', 'experimental_native_lane_training.py',
                             'tensor_ppo.py', 'tensor_rollout_buffer.py'):
                    digest = hashlib.sha256((self.mod.ROOT/name).read_bytes()).hexdigest()
                    self.assertEqual(result['sources'][name], digest)

    def test_cuda_gate_prevents_environment_or_model_creation(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            self.mod.require_cuda('cuda', torch_module=fake_torch)
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(self.mod, 'require_cuda', side_effect=RuntimeError('CUDA unavailable')):
                with patch.object(self.mod, 'build_backend') as build:
                    with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                        self.mod.run_benchmark(self.mod.BenchmarkConfig(), Path(temporary)/'run', FROZEN)
                    build.assert_not_called()
            receipt = json.loads((Path(temporary)/'run/result.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertIn('CUDA unavailable', receipt['error'])

    def test_measured_boundaries_and_actual_valid_tick_accounting(self):
        config = self.mod.BenchmarkConfig(games=2, n_steps=4, batch_size=8)
        events = []
        stats = SimpleNamespace(valid_steps=np.array([7, 7]), errors=np.zeros(2, dtype=int))
        callback = SimpleNamespace(rollout_infos=[{'episode': {'r': 0, 'l': 4},
                                                   'combat': {'game': 0, 'player': 0, 'episode_index': 0}}])
        def collect(*args, **kwargs):
            events.append('collect')
            stats.valid_steps += 4
            return True
        model = SimpleNamespace(env=None, rollout_buffer=object(), collect_rollouts=collect,
                                train=lambda: events.append('update'), num_timesteps=16,
                                _total_timesteps=16, _update_current_progress_remaining=lambda *a: None,
                                logger=SimpleNamespace(name_to_value={'train/loss': 1.0,
                                    'train/policy_gradient_loss': 0.1, 'train/value_loss': 0.2,
                                    'train/entropy_loss': -6.0, 'train/explained_variance': float('nan')}))
        with patch.object(self.mod.time, 'perf_counter', side_effect=[10., 12., 15.]) as perf, \
             patch.object(self.mod.time, 'time', side_effect=[1000., 1005.]) as wall:
            boundaries = Mock()
            boundaries.attach_mock(perf, 'perf')
            boundaries.attach_mock(wall, 'wall')
            synchronize = Mock(side_effect=lambda: events.append('sync'))
            boundaries.attach_mock(synchronize, 'sync')
            row = self.mod.measure_rollout(model, callback, stats, config,
                                          synchronize=synchronize)
        self.assertEqual(perf.call_count, 3)
        self.assertEqual(wall.call_count, 2)
        self.assertEqual(boundaries.mock_calls, [call.sync(), call.wall(), call.perf(),
            call.sync(), call.perf(), call.sync(), call.perf(), call.wall()])
        self.assertEqual(row['started_unix'], 1000.)
        self.assertEqual(row['finished_unix'], 1005.)
        self.assertEqual(events, ['sync', 'collect', 'sync', 'update', 'sync'])
        self.assertEqual(row['game_ticks'], 8)
        self.assertEqual(row['player_transitions'], 16)
        self.assertEqual(row['collection_seconds'], 2.)
        self.assertEqual(row['update_seconds'], 3.)
        self.assertEqual(row['total_seconds'], 5.)
        self.assertEqual(row['collection_game_ticks_per_second'], 4.)
        self.assertEqual(row['total_player_transitions_per_second'], 3.2)
        self.assertTrue(row['losses_finite'])
        self.assertIsNone(row['metrics']['train/explained_variance'])
        self.assertEqual(row['errors'], [0, 0])

    def test_stalled_ticks_or_nonfinite_losses_fail_closed(self):
        config = self.mod.BenchmarkConfig(games=1, n_steps=2, batch_size=4)
        stats = SimpleNamespace(valid_steps=np.array([0]), errors=np.array([0]))
        model = SimpleNamespace(env=None, rollout_buffer=None, num_timesteps=4,
                                _total_timesteps=4, _update_current_progress_remaining=lambda *a: None,
                                collect_rollouts=lambda *a, **kw: True, train=lambda: None,
                                logger=SimpleNamespace(name_to_value={'train/loss': float('nan')}))
        with self.assertRaisesRegex(RuntimeError, 'valid game ticks'):
            self.mod.measure_rollout(model, SimpleNamespace(rollout_infos=[]), stats, config,
                                     synchronize=lambda: None)
        with self.assertRaisesRegex(RuntimeError, 'finite loss'):
            self.mod.loss_metrics(model.logger.name_to_value)

    def test_summary_excludes_warmup_and_uses_ratio_of_totals(self):
        rows = [dict(measured=False, game_ticks=10, player_transitions=20,
                     collection_seconds=100, update_seconds=100, total_seconds=200),
                dict(measured=True, game_ticks=10, player_transitions=20,
                     collection_seconds=1, update_seconds=1, total_seconds=2),
                dict(measured=True, game_ticks=10, player_transitions=20,
                     collection_seconds=3, update_seconds=1, total_seconds=4)]
        summary = self.mod.summarize_rollouts(rows)
        self.assertEqual(summary['game_ticks'], 20)
        self.assertEqual(summary['total_seconds'], 6)
        self.assertEqual(summary['collection_game_ticks_per_second'], 5)
        self.assertAlmostEqual(summary['total_player_transitions_per_second'], 40/6)

    def test_worker_errors_and_existing_artifacts_are_never_silently_accepted(self):
        pipe = SimpleNamespace(poll=lambda timeout: True, recv=lambda: ('error', 'worker traceback'))
        with self.assertRaisesRegex(RuntimeError, 'worker traceback'):
            self.mod.PythonGameCPUEnv._receive(pipe)
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary)/'result.json'
            receipt.write_text('preserve this')
            with self.assertRaisesRegex(ValueError, 'overwritten'):
                self.mod.run_benchmark(self.mod.BenchmarkConfig(), Path(temporary), FROZEN)
            self.assertEqual(receipt.read_text(), 'preserve this')

    def test_cuda_device_contract_is_checked_before_mocked_training(self):
        self.assert_mocked_driver('sb3')

    def test_deferred_driver_preserves_timing_accounting_and_ppo_settings(self):
        self.assert_mocked_driver('deferred_metrics')

    def assert_mocked_driver(self, update_mode):
        import torch
        from stable_baselines3.common.logger import configure
        from ppo_update_optimizations import SB3_VERSION, SB3_TRAIN_SHA256
        config = self.mod.BenchmarkConfig(games=1, n_steps=2, batch_size=4,
                                          warmup_rollouts=1, measured_rollouts=2, update_mode=update_mode)
        creations = []
        models = []
        current_backend = None
        class FakeModel:
            """Control-flow stand-in only: no torch module, policy call, or optimizer."""
            def __init__(self, policy, env, **kwargs):
                creations.append(kwargs)
                models.append(weakref.ref(self))
                self.policy, self.env, self.rollout_buffer = 'no actual policy', env, SimpleNamespace()
                self.num_timesteps = 0
            def set_logger(self, logger):
                self.logger = logger
            def _setup_learn(self, total, callback):
                self._total_timesteps = total
                return total, callback
            def collect_rollouts(self, env, callback, buffer, steps):
                current_backend.valid_steps += steps
                self.num_timesteps += steps*2
                callback.rollout_infos = []
                return True
            def _update_current_progress_remaining(self, *args):
                pass
            def train(self):
                for key in self_outer.mod.LOSS_KEYS:
                    self.logger.record(key, 1.)
            def save(self, path):
                Path(path).write_bytes(b'mocked checkpoint, not a trained model')
        self_outer = self
        def build(*args):
            nonlocal current_backend
            current_backend = SimpleNamespace(valid_steps=np.array([0], dtype=np.uint64), errors=np.array([0]))
            env = SimpleNamespace(num_envs=2, obs_shape=(256,), action_space=type('Box', (), {})(), close=lambda: None)
            return env, current_backend
        callback = SimpleNamespace(rollout_infos=[], on_training_start=lambda *a: None, on_training_end=lambda: None)
        def training_start(local_values, global_values):
            callback.locals, callback.globals = local_values, global_values
            callback.model = local_values['model']
        callback.on_training_start = training_start
        receipts = [{'sha256': name, 'parameter_count': 42} for _ in config.backends for name in ('initial', 'changed')]
        loggers = []
        deferred_calls = []
        def deferred_control_flow(model, *, loss_function):
            deferred_calls.append(loss_function)
            self.assertFalse(model.defer_timeout_values)
            self.assertFalse(model.rollout_buffer.compile_gae)
            self.assertEqual(model.rollout_buffer.validation_mode, 'step')
            FakeModel.train(model)
        def tracked_logger(*args):
            logger = configure(*args)
            loggers.append(logger)
            return logger
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(self.mod, 'require_cuda', return_value='cuda:0'), \
                 patch.object(self.mod, 'provenance', return_value={'test_double': True}), \
                 patch.object(self.mod, 'build_backend', side_effect=build), \
                 patch.object(self.mod, '_make_callback', return_value=callback), \
                 patch.object(self.mod, 'policy_receipt', side_effect=receipts), \
                 patch('tensor_ppo.TensorPPO', FakeModel), patch.object(torch.cuda, 'synchronize'), \
                 patch('ppo_update_optimizations.sb3_train_source_metadata', return_value=dict(
                     version=SB3_VERSION, train_sha256=SB3_TRAIN_SHA256, supported=True)), \
                 patch('ppo_update_optimizations.deferred_metrics_train', new=deferred_control_flow), \
                 patch('stable_baselines3.common.logger.configure', side_effect=tracked_logger), \
                 redirect_stdout(io.StringIO()):
                result = self.mod.run_benchmark(config, Path(temporary)/'run', FROZEN)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(len(creations), 3)
            self.assertTrue(all(kwargs == creations[0] for kwargs in creations))
            self.assertEqual(creations[0]['device'], 'cuda:0')
            self.assertEqual(creations[0]['policy_kwargs']['net_arch'], {'pi': [256, 256], 'vf': [256, 256]})
            self.assertEqual(creations[0]['n_epochs'], config.n_epochs)
            self.assertEqual(creations[0]['ent_coef'], config.ent_coef)
            self.assertIsNone(creations[0]['target_kl'])
            self.assertEqual(deferred_calls, [None]*9 if update_mode == 'deferred_metrics' else [])
            for backend in result['backends']:
                self.assertEqual(backend['summary']['game_ticks'], 4)
                self.assertEqual(backend['summary']['player_transitions'], 8)
                self.assertEqual([row['measured'] for row in backend['rollouts']], [False, True, True])
                self.assertTrue(Path(backend['checkpoint']).is_file())
                self.assertTrue(backend['parameters_changed'])
                self.assertEqual(len(backend['checkpoint_sha256']), 64)
                self.assertTrue((Path(backend['checkpoint']).parent/'benchmark.log').is_file())
            self.assertTrue(all(writer.file.closed for logger in loggers for writer in logger.output_formats))
            self.assertTrue(all(model() is None for model in models), 'previous models retained by callback cycles')


class OriginalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module('benchmark_native_combat_training')

    def test_original_reset_step_and_rng_isolation(self):
        if not FROZEN.is_dir():
            self.skipTest('frozen original game not installed')
        caller_py, caller_np = random.getstate(), np.random.get_state()
        with self.mod.OriginalGameBatch(FROZEN, [0, 1]) as batch:
            reset = batch.reset([17, 19])
            self.assertEqual(reset.shape, (4, 256))
            self.assertEqual(np.count_nonzero(reset[:, :192]), 0)
            previous = reset.copy()
            actions = np.zeros((4, 10), dtype=np.float32)
            actions[0, 3] = actions[3, 1] = 1
            step = batch.step(actions)
            np.testing.assert_array_equal(step['observations'][:, 128:192], previous[:, 192:])
            self.assertEqual(step['valid_steps'].tolist(), [1, 1])
            for game_index, game in enumerate(batch.games):
                for player in range(2):
                    row = game_index*2+player
                    np.testing.assert_array_equal(step['observations'][row, -64:],
                                                  game.observe(player).astype(np.float32))
                    self.assertEqual(step['rewards'][row], game.rewards[player])
            replay_start = batch.reset([17, 19])
            replay_step = batch.step(actions)
            np.testing.assert_array_equal(replay_start, previous)
            np.testing.assert_array_equal(replay_step['observations'], step['observations'])
        self.assertEqual(random.getstate(), caller_py)
        current_np = np.random.get_state()
        self.assertEqual(current_np[0], caller_np[0])
        np.testing.assert_array_equal(current_np[1], caller_np[1])
        self.assertEqual(current_np[2:], caller_np[2:])

    def test_both_terminal_frames_preserved_before_pair_autoreset(self):
        if not FROZEN.is_dir():
            self.skipTest('frozen original game not installed')
        with self.mod.OriginalGameBatch(FROZEN, [4]) as batch:
            initial = batch.reset([8]).copy()
            batch.games[0].max_timesteps = 1  # Test fixture only, not benchmark configuration.
            step = batch.step(np.zeros((2, 10), dtype=np.float32))
            self.assertEqual(step['truncated'].tolist(), [True, True])
            np.testing.assert_array_equal(step['terminal_observations'][:, 128:192], initial[:, -64:])
            self.assertEqual(np.count_nonzero(step['observations'][:, :192]), 0)
            infos = step['infos']
            self.assertEqual([v['combat']['player'] for v in infos], [0, 1])
            self.assertTrue(all(v['combat']['game'] == 4 and v['episode']['l'] == 1 for v in infos))

    def test_spawn_parallel_matches_serial_saved_actions(self):
        if not FROZEN.is_dir():
            self.skipTest('frozen original game not installed')
        serial = self.mod.PythonGameCPUEnv(2, FROZEN, seed=3, workers=1)
        parallel = self.mod.PythonGameCPUEnv(2, FROZEN, seed=3, workers=2)
        try:
            import torch
            torch.testing.assert_close(serial.reset_tensor(), parallel.reset_tensor(), rtol=0, atol=0)
            actions = torch.zeros((4, 10), dtype=torch.float32)
            actions[0, 7] = actions[3, 4] = 1
            for _ in range(4):
                a, b = serial.step_tensor(actions), parallel.step_tensor(actions)
                for key in ('observations', 'rewards', 'terminated', 'truncated', 'terminal_observations'):
                    torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=0)
            self.assertEqual(parallel.backend.valid_steps.tolist(), [4, 4])
        finally:
            parallel.close()
            serial.close()


if __name__ == '__main__':
    unittest.main()
