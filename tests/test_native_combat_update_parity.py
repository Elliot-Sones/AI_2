"""Exact update parity on actual native combat rollouts; never train on CPU.

Run on the pinned SB3 2.5.0 CUDA host. Local discovery skips before policy
construction. Existing test_ppo_update_optimizations covers objective/KL cases;
this test qualifies the real combat TensorRolloutBuffer and checkpoint path.
"""
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_ppo_update_optimizations as update_tests
from tests import test_native_game_collector as collector_tests


class NativeCombatUpdateParityTests(unittest.TestCase):
    # Reuse the existing exact nested tensor/optimizer/metric comparator only,
    # without inheriting or running its CPU policy-training test cases.
    assert_tree_equal = update_tests.PPOUpdateParityTests.assert_tree_equal

    def setUp(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f'Torch unavailable: {exc}')
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable; no local policy construction or training')
        self.torch = torch
        import benchmark_native_combat_training as benchmark
        from ppo_update_optimizations import sb3_train_source_metadata
        metadata = sb3_train_source_metadata()
        self.assertTrue(metadata['supported'], f'CUDA parity requires pinned SB3 train source: {metadata}')
        self.benchmark = benchmark
        self.config = benchmark.BenchmarkConfig(backends=('native_cuda',))
        self.device = benchmark.require_cuda(self.config.device)
        self.addCleanup(self.restore_rng, self.rng_state())
        artifact_root = os.environ.get('AI2_COMBAT_PARITY_ARTIFACT_DIR')
        if artifact_root:
            Path(artifact_root).mkdir(parents=True, exist_ok=True)
            self.artifacts = Path(tempfile.mkdtemp(prefix=self._testMethodName+'-', dir=artifact_root))
        else:
            temporary = tempfile.TemporaryDirectory(prefix='combat-update-parity-')
            self.addCleanup(temporary.cleanup)
            self.artifacts = Path(temporary.name)
        self.rollout_artifacts = 0

    def rng_state(self):
        return (random.getstate(), np.random.get_state(), self.torch.get_rng_state().clone(),
                [state.clone() for state in self.torch.cuda.get_rng_state_all()])

    def restore_rng(self, state):
        random.setstate(state[0])
        np.random.set_state(state[1])
        self.torch.set_rng_state(state[2])
        self.torch.cuda.set_rng_state_all(state[3])

    def make_model(self, mode, *, checkpoint=None, horizon=None):
        import native_game_lane_vec_env as native
        from stable_baselines3.common.logger import configure
        config = replace(self.config, update_mode=mode)
        ppo_class = self.benchmark.make_benchmark_ppo_class(config)
        factory = native.GameReference
        fixture = (nullcontext() if horizon is None else patch.object(native, 'GameReference',
            lambda: collector_tests._short_horizon_reference(factory, horizon)))
        with fixture:
            env, backend = self.benchmark.build_backend('native_cuda', config, Path('.'), self.device)
        self.addCleanup(env.close)
        if checkpoint is None:
            model = ppo_class('MlpPolicy', env, device=self.device, seed=config.seed,
                n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
                learning_rate=config.learning_rate, gamma=config.gamma, gae_lambda=config.gae_lambda,
                clip_range=config.clip_range, ent_coef=config.ent_coef, vf_coef=config.vf_coef,
                max_grad_norm=config.max_grad_norm, normalize_advantage=True, target_kl=None,
                policy_kwargs={'net_arch': {'pi': list(config.net_arch), 'vf': list(config.net_arch)},
                               'activation_fn': self.torch.nn.Tanh}, verbose=0)
        else:
            model = ppo_class.load(checkpoint, env=env, device=self.device)
        logger = configure(folder=None, format_strings=[])
        self.addCleanup(logger.close)
        model.set_logger(logger)
        callback = self.benchmark._make_callback()
        if horizon is not None:
            class Capture(type(callback)):
                def _on_rollout_start(self):
                    super()._on_rollout_start()
                    self.rows = []

                def _on_step(self):
                    step = self.locals['step']
                    self.rows.append(dict(pre=self.locals['pre_action_obs'].detach().clone(),
                        actions=self.locals['clipped_actions'].detach().clone(),
                        obs=step.observations.detach().clone(), raw_rewards=step.rewards.detach().clone(),
                        terminated=step.terminated.detach().clone(), truncated=step.truncated.detach().clone(),
                        terminal=step.terminal_observations.detach().clone()))
                    return True
            callback = Capture()
        _, callback = model._setup_learn(config.player_transitions_per_rollout*4,
            callback=callback, reset_num_timesteps=checkpoint is None)
        self.addCleanup(self.release_callback, callback)
        self.assertFalse(model.defer_timeout_values)
        self.assertEqual(model.rollout_buffer.validation_mode, 'step')
        self.assertFalse(model.rollout_buffer.compile_gae)
        self.assertFalse(hasattr(model, '_experimental_loss_function'))
        return model, backend, callback

    @staticmethod
    def release_callback(callback):
        callback.locals, callback.globals, callback.model = {}, {}, None

    def replay_saved_rollout(self, source, target, evidence):
        from tensor_rollout_buffer import TensorRolloutBuffer
        self.assertIs(type(source), TensorRolloutBuffer)
        self.assertIs(type(target), TensorRolloutBuffer)
        self.assertTrue(source.full and source._computed_returns_and_advantage)
        names = ('observations', 'actions', 'rewards', 'episode_starts', 'values',
                 'log_probs', 'advantages', 'returns', '_invalid_data')
        payload = dict(config=asdict(self.config), evidence=evidence,
            tensors={name: getattr(source, name).detach().cpu() for name in names},
            pos=source.pos, full=source.full,
            computed_returns_and_advantage=source._computed_returns_and_advantage)
        path = self.artifacts/f'rollout-{self.rollout_artifacts:02d}.pt'
        self.rollout_artifacts += 1
        self.torch.save(payload, path)
        loaded = self.torch.load(path, map_location=self.device, weights_only=True)
        self.assertEqual(loaded['config'], asdict(self.config))
        self.assertEqual(loaded['evidence'], evidence)
        target.reset()
        for name in names:
            original = getattr(source, name)
            self.assertTrue(original.is_cuda)
            getattr(target, name).copy_(loaded['tensors'][name])
            self.assert_tree_equal(original, getattr(target, name))
        target.pos, target.full = loaded['pos'], loaded['full']
        target._computed_returns_and_advantage = loaded['computed_returns_and_advantage']
        return dict(rollout_file=path.name, rollout_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    collected_on='native_cuda', replayed_from_disk=True, tensors_exact=True, evidence=evidence)

    def sample_trace(self, model):
        samples = []
        original = model.rollout_buffer._get_samples_from_device_indices

        def sample(indices, samples_cls):
            samples.append(indices.detach().cpu().clone())
            return original(indices, samples_cls)

        model.rollout_buffer._get_samples_from_device_indices = sample
        return samples

    def compare_real_updates(self, reference_pack, candidate_pack, *, repeats, verify_rollout=None):
        reference, backend, callback = reference_pack
        candidate = candidate_pack[0]
        traces = self.sample_trace(reference), self.sample_trace(candidate)
        self.assert_tree_equal(reference.policy.state_dict(), candidate.policy.state_dict())
        self.assert_tree_equal(reference.policy.optimizer.state_dict(), candidate.policy.optimizer.state_dict())
        initial = {name: value.detach().clone() for name, value in reference.policy.state_dict().items()}
        initial_updates = reference._n_updates
        for index in range(repeats):
            with self.subTest(update=index):
                before = np.array(backend.valid_steps, copy=True)
                self.assertTrue(reference.collect_rollouts(reference.env, callback,
                    reference.rollout_buffer, self.config.n_steps))
                np.testing.assert_array_equal(np.asarray(backend.valid_steps) - before,
                                              np.full(self.config.games, self.config.n_steps))
                self.assertFalse(np.asarray(backend.errors).any())
                self.assertEqual(reference.rollout_buffer.observations.shape,
                                 (self.config.n_steps, 2*self.config.games, 256))
                evidence = {'fixture': 'default benchmark horizon'} if verify_rollout is None else verify_rollout()
                receipt = self.replay_saved_rollout(reference.rollout_buffer, candidate.rollout_buffer, evidence)
                candidate.num_timesteps = reference.num_timesteps
                for model in (reference, candidate):
                    model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
                initial_rng = self.rng_state()
                reference.train()
                reference_rng = self.rng_state()
                self.restore_rng(initial_rng)
                candidate.train()
                self.assert_tree_equal(reference_rng, self.rng_state())
                self.assert_tree_equal(traces[0], traces[1])
                self.assert_tree_equal(reference.policy.state_dict(), candidate.policy.state_dict())
                self.assert_tree_equal(reference.policy.optimizer.state_dict(), candidate.policy.optimizer.state_dict())
                self.assert_tree_equal(dict(reference.logger.name_to_value), dict(candidate.logger.name_to_value))
                self.assertEqual(reference._n_updates, candidate._n_updates)
                self.assertEqual(reference._n_updates, initial_updates+(index+1)*self.config.n_epochs)
                self.benchmark.loss_metrics(reference.logger.name_to_value)
                receipt.update(status='passed', parameters_exact=True, optimizer_exact=True, rng_exact=True,
                    metrics_exact=True, sample_order_exact=True, n_updates=reference._n_updates,
                    metrics=self.benchmark.loss_metrics(reference.logger.name_to_value))
                self.artifacts.joinpath(receipt['rollout_file']+'.json').write_text(
                    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
        expected_samples = repeats*self.config.n_epochs*self.config.player_transitions_per_rollout//self.config.batch_size
        self.assertEqual(len(traces[0]), expected_samples)
        self.assertTrue(any(not self.torch.equal(initial[name], value)
                            for name, value in reference.policy.state_dict().items()))

    def test_actual_timeout_autoreset_rollouts_replay_and_update_exactly(self):
        import native_game_lane_vec_env as native
        torch, horizon = self.torch, 3
        # Declared test fixture only: native context creation receives a short
        # horizon. No gameplay state, collected tick, reward, or PPO setting is edited.
        reference = self.make_model('sb3', horizon=horizon)
        candidate = self.make_model('deferred_metrics', horizon=horizon)
        factory = native.GameReference
        with patch.object(native, 'GameReference',
                          lambda: collector_tests._short_horizon_reference(factory, horizon)):
            oracle = native.build_game_env(self.config.games, device='cpu', seed=self.config.seed,
                                           record_capacity=self.config.game_ticks_per_rollout)
        self.addCleanup(oracle.close)
        model, _, callback = reference
        previous = oracle.reset_tensor(seeds=model.tensor_vec_env.env._seeds.tolist()).to(self.device)
        previous_done = torch.ones(2*self.config.games, device=self.device)
        collected_ticks = 0
        self.assert_tree_equal(previous, model._last_obs)

        def verify_timeout_rollout():
            nonlocal previous, previous_done, collected_ticks
            self.assertEqual(len(callback.rows), self.config.n_steps)
            timeout_games = 0
            with torch.no_grad():
                for tick, row in enumerate(callback.rows):
                    # Policy-free serial C oracle consumes the actual CUDA actions.
                    expected = oracle.step_tensor(row['actions'].cpu())
                    terminal = expected.terminal_observations.to(self.device)
                    reset_or_next = expected.observations.to(self.device)
                    raw_rewards = expected.rewards.to(self.device)
                    terminated, truncated = expected.terminated.to(self.device), expected.truncated.to(self.device)
                    self.assert_tree_equal(row['pre'], previous)
                    self.assert_tree_equal(row['terminal'], terminal)
                    self.assert_tree_equal(row['terminal'][:, :192], row['pre'][:, 64:])
                    self.assert_tree_equal(row['obs'], reset_or_next)
                    self.assert_tree_equal(row['raw_rewards'], raw_rewards)
                    self.assert_tree_equal(row['terminated'], terminated)
                    self.assert_tree_equal(row['truncated'], truncated)
                    self.assertFalse(bool(terminated.any()))
                    collected_ticks += 1
                    self.assertEqual(bool(truncated.all()), collected_ticks % horizon == 0)
                    self.assert_tree_equal(model.rollout_buffer.episode_starts[tick], previous_done)
                    terminal_values = model.policy.predict_values(terminal).flatten()
                    corrected_rewards = torch.where(truncated & ~terminated,
                        raw_rewards+model.gamma*terminal_values, raw_rewards)
                    self.assert_tree_equal(model.rollout_buffer.rewards[tick], corrected_rewards)
                    if bool(truncated.all()):
                        timeout_games += self.config.games
                        self.assertGreater(int(torch.count_nonzero(terminal[:, :192])), 0)
                        self.assertEqual(int(torch.count_nonzero(row['obs'][:, :192])), 0)
                        self.assertFalse(torch.equal(row['terminal'], row['obs']))
                        self.assertFalse(torch.equal(terminal_values, model.policy.predict_values(row['obs']).flatten()))
                        self.assertFalse(torch.equal(corrected_rewards, raw_rewards))
                    previous, previous_done = reset_or_next, (terminated | truncated).float()
            self.assertGreater(timeout_games, 0, 'timeout coverage must be observed, not inferred')
            self.assertEqual(len(callback.rollout_infos), 2*timeout_games)
            self.assertTrue(all(info['episode']['l'] == horizon for info in callback.rollout_infos))
            self.assertEqual(len(oracle.drain_episode_infos()), 2*timeout_games)
            return dict(fixture='declared native three-tick timeout', time_limit_steps=horizon,
                        timed_out_games=timeout_games, timed_out_players=2*timeout_games,
                        terminal_frames_exact=True, reset_stacks_exact=True, bootstrap_rewards_exact=True)

        self.compare_real_updates(reference, candidate, repeats=2, verify_rollout=verify_timeout_rollout)

    def test_real_combat_repeated_updates_and_checkpoint_reload_are_exact(self):
        reference = self.make_model('sb3')
        candidate = self.make_model('deferred_metrics')
        self.compare_real_updates(reference, candidate, repeats=3)
        with tempfile.TemporaryDirectory(prefix='combat-update-parity-') as temporary:
            checkpoints = [Path(temporary)/f'{mode}.zip' for mode in ('sb3', 'deferred_metrics')]
            for pack, path in zip((reference, candidate), checkpoints):
                pack[0].save(path)
                self.assertTrue(path.is_file())
            loaded_reference = self.make_model('sb3', checkpoint=checkpoints[0])
            loaded_candidate = self.make_model('deferred_metrics', checkpoint=checkpoints[1])
            for original, loaded in ((reference[0], loaded_reference[0]), (candidate[0], loaded_candidate[0])):
                self.assert_tree_equal(original.policy.state_dict(), loaded.policy.state_dict())
                # SB3 load may relocate optimizer scalar counters. The two
                # loaded arms must match each other exactly, including device.
                original_optimizer = original.policy.optimizer.state_dict()
                loaded_optimizer = loaded.policy.optimizer.state_dict()
                self.assert_tree_equal(original_optimizer['param_groups'], loaded_optimizer['param_groups'])
                self.assertEqual(original_optimizer['state'].keys(), loaded_optimizer['state'].keys())
                for parameter, values in original_optimizer['state'].items():
                    restored = loaded_optimizer['state'][parameter]
                    self.assertEqual(values.keys(), restored.keys())
                    for name, value in values.items():
                        if self.torch.is_tensor(value):
                            self.assert_tree_equal(value.cpu(), restored[name].cpu())
                        else:
                            self.assert_tree_equal(value, restored[name])
                self.assertEqual(original._n_updates, loaded._n_updates)
                self.assertEqual(original.num_timesteps, loaded.num_timesteps)
            self.compare_real_updates(loaded_reference, loaded_candidate, repeats=1)


if __name__ == '__main__':
    unittest.main()
