"""Local orchestration tests only: no model construction, inference, or CUDA."""
from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_profile_native_lane_training import fake_classes


class ProfileCombatTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('profile_native_combat_training'),
                             'raw combat profiler entrypoint must exist')
        self.module = importlib.import_module('profile_native_combat_training')
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name) / 'profile'
        self.args = self.module.build_parser().parse_args(
            ['--frozen-root', self.folder.name, '--output', str(self.output)])

    def harness(self, *, cuda=True, device_events=True, failure=None):
        classes = fake_classes()
        class Averages(list):
            def table(self, **kwargs):
                return str(kwargs)
        averages = Averages(SimpleNamespace(key=name, count=1, device_type='CPU', is_user_annotation=True,
                                            cpu_time_total=10., self_cpu_time_total=8.,
                                            device_time_total=5., self_device_time_total=3.)
                            for name in self.module.regions.REGIONS)
        profiler = SimpleNamespace(key_averages=lambda: averages,
                                   events=lambda: [SimpleNamespace(device_type='CUDA')] if device_events else [],
                                   export_chrome_trace=lambda path: Path(path).write_text('{"traceEvents":[]}'))
        from contextlib import nullcontext
        torch = SimpleNamespace(__version__='mock', version=SimpleNamespace(cuda='mock'),
            device=lambda name, index=None: SimpleNamespace(index=index),
            cuda=SimpleNamespace(is_available=lambda: cuda, current_device=lambda: 0,
                                 device_count=lambda: 1, get_device_name=lambda device: 'mock GPU'),
            profiler=SimpleNamespace(ProfilerActivity=SimpleNamespace(CPU='CPU', CUDA='CUDA'),
                                     supported_activities=lambda: {'CPU', 'CUDA'},
                                     profile=lambda **kwargs: nullcontext(profiler),
                                     record_function=lambda name: nullcontext()))
        calls = []
        def benchmark(config, output, frozen):
            calls.append(config)
            classes['ppo_class']().collect_rollouts(2)
            classes['ppo_class']().train(1)
            if failure:
                raise failure
            output.mkdir()
            (output / 'result.json').write_text('{"status":"passed"}')
            return {'status': 'passed'}
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {
            'torch': torch,
            'stable_baselines3.common.policies': SimpleNamespace(ActorCriticPolicy=classes['policy_class']),
            'tensor_ppo': SimpleNamespace(TensorPPO=classes['ppo_class'], TensorVecEnv=classes['env_class']),
            'tensor_rollout_buffer': SimpleNamespace(TensorRolloutBuffer=classes['buffer_class']),
        }))
        stack.enter_context(patch.object(self.module.driver, 'run_benchmark', benchmark))
        self.addCleanup(stack.close)
        return calls, classes

    def test_fixed_matched_workload_and_diagnostic_artifacts(self):
        calls, classes = self.harness()
        original = classes['policy_class'].forward
        self.assertEqual(self.module.run(self.args), 0)
        config, = calls
        self.assertEqual((config.games, config.n_steps, config.net_arch, config.batch_size, config.n_epochs,
                          config.warmup_rollouts, config.measured_rollouts, config.backends),
                         (128, 128, (512, 512, 256), 4096, 8, 1, 1, ('native_cuda',)))
        report = json.loads((self.output / 'profile.json').read_text())
        self.assertEqual(report['status'], 'passed')
        self.assertTrue(report['diagnostic_only'])
        self.assertFalse(report['headline_throughput_eligible'])
        self.assertIn('overhead', ' '.join(report['interpretation']).lower())
        self.assertIn('TensorVecEnv.step_tensor', ' '.join(report['interpretation']))
        self.assertEqual(report['regions']['native_environment_step']['calls'], 1)
        self.assertEqual(report['regions']['critic_only_forward']['calls'], 2)
        self.assertEqual(report['regions']['ppo_update']['calls'], 1)
        self.assertTrue(report['source_hashes_unchanged'])
        for name in ('trace.json', 'cpu-table.txt', 'device-table.txt', 'benchmark/result.json'):
            self.assertTrue((self.output / name).is_file(), name)
        self.assertIs(classes['policy_class'].forward, original)
        self.assertNotIn('train', classes['ppo_class'].__dict__)

    def test_cuda_unavailable_prevents_benchmark(self):
        calls, _ = self.harness(cuda=False)
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            self.module.run(self.args)
        self.assertFalse(calls)
        self.assertFalse(self.output.exists())

    def test_cpu_device_is_rejected(self):
        calls, _ = self.harness()
        self.args.device = 'cpu'
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            self.module.run(self.args)
        self.assertFalse(calls)

    def test_existing_artifacts_are_never_overwritten(self):
        calls, _ = self.harness()
        self.output.mkdir()
        marker = self.output / 'prior.json'
        marker.write_text('immutable')
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.module.run(self.args)
        self.assertEqual(marker.read_text(), 'immutable')
        self.assertFalse(calls)

    def test_benchmark_error_preserved_with_failed_trace_and_methods_restored(self):
        failure = RuntimeError('benchmark failed')
        _, classes = self.harness(failure=failure)
        original = classes['policy_class'].forward
        with self.assertRaises(RuntimeError) as caught:
            self.module.run(self.args)
        self.assertIs(caught.exception, failure)
        report = json.loads((self.output / 'profile.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertTrue((self.output / 'trace.json').exists())
        self.assertIs(classes['policy_class'].forward, original)

    def test_no_device_events_is_failed_capture_not_cuda_evidence(self):
        self.harness(device_events=False)
        with self.assertRaisesRegex(RuntimeError, 'CUDA.*events'):
            self.module.run(self.args)
        report = json.loads((self.output / 'profile.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['cuda_trace_event_count'], 0)

    def test_missing_instrumented_work_is_not_a_successful_profile(self):
        _, classes = self.harness()
        classes['ppo_class'].collect_rollouts = lambda self, value: None
        with self.assertRaisesRegex(RuntimeError, 'regions did not execute'):
            self.module.run(self.args)
        report = json.loads((self.output / 'profile.json').read_text())
        self.assertEqual(report['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
