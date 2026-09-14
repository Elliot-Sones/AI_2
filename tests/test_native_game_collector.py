"""Actual full-game CUDA collector timeouts; no local CPU policy fallback.

The only setup substitution is a declared three-tick native configuration.
Every game advances normally after construction, including native autoresets.
The serial C game oracle consumes saved actions, never a CPU policy.
"""
from contextlib import ExitStack
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def _short_horizon_reference(factory, horizon):
    reference = factory()
    reference.cfg.time_limit_steps = horizon
    return reference


class HorizonFixtureTests(unittest.TestCase):
    def test_fixture_changes_only_declared_horizon_before_context_creation(self):
        original = SimpleNamespace(cfg=SimpleNamespace(time_limit_steps=9000, dt=1/30), table=object())
        factory = Mock(return_value=original)
        function = globals().get('_short_horizon_reference')
        self.assertIsNotNone(function, 'short-horizon configuration fixture must exist')
        result = function(factory, 3)
        self.assertIs(result, original)
        factory.assert_called_once_with()
        self.assertEqual(result.cfg.time_limit_steps, 3)
        self.assertEqual(result.cfg.dt, 1/30)


@unittest.skipUnless(shutil.which('nvcc'), 'actual CUDA native collector requires nvcc; no CPU policy fallback')
class NativeGameCollectorCudaTests(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('actual NVIDIA CUDA device unavailable; no CPU policy fallback')
        self.torch = torch
        self.device = torch.device('cuda', torch.cuda.current_device())

    def _verify_collector(self, *, substitute_reset_terminal=False):
        import native_game_lane_vec_env as native
        from stable_baselines3.common.callbacks import BaseCallback
        from tensor_ppo import TensorPPO, TensorStep, TensorVecEnv

        torch, device = self.torch, self.device
        games, horizon, steps, seed = 2, 3, 7, 17
        # Only configuration construction is substituted. No state/tick edits.
        reference_factory = native.GameReference
        with patch.object(native, 'GameReference',
                          lambda: _short_horizon_reference(reference_factory, horizon)):
            raw = native.build_game_env(games, device=device, seed=seed, record_capacity=16)
            self.addCleanup(raw.close)
            oracle = native.build_game_env(games, device='cpu', seed=seed, record_capacity=16)
            self.addCleanup(oracle.close)
        adapter = TensorVecEnv(raw, device=device)
        model = TensorPPO('MlpPolicy', adapter, device=device, seed=seed, n_steps=steps,
                         batch_size=4, n_epochs=1, gamma=0.9, gae_lambda=0.95,
                         policy_kwargs={'net_arch': [8]}, verbose=0)
        self.assertTrue(all(parameter.is_cuda and parameter.device == device
                            for parameter in model.policy.parameters()))
        self.assertFalse(model.defer_timeout_values, 'exercise the benchmark default collector')

        class Capture(BaseCallback):
            def __init__(self):
                super().__init__()
                self.rows, self.infos = [], []

            def _on_step(self):
                step = self.locals['step']
                self.rows.append(dict(pre=self.locals['pre_action_obs'].detach().clone(),
                    actions=self.locals['clipped_actions'].detach().clone(),
                    obs=step.observations.detach().clone(), raw_rewards=step.rewards.detach().clone(),
                    terminated=step.terminated.detach().clone(), truncated=step.truncated.detach().clone(),
                    terminal=step.terminal_observations.detach().clone()))
                return True

            def _on_rollout_end(self):
                self.infos = list(self.locals['drained_infos'])

        _, callback = model._setup_learn(steps * games * 2, callback=Capture())
        def detach_callback():
            callback.locals, callback.globals, callback.model = {}, {}, None
        self.addCleanup(detach_callback)
        callback.on_training_start({}, {})
        initial = oracle.reset_tensor(seeds=raw._seeds.tolist()).clone()
        torch.testing.assert_close(model._last_obs, initial.to(device), rtol=0, atol=0)
        with ExitStack() as substitutions:
            if substitute_reset_terminal:
                original_step = raw.step_tensor

                def wrong_terminal(actions):
                    result = original_step(actions)
                    return TensorStep(result.observations, result.rewards, result.terminated,
                                      result.truncated, result.observations)

                substitutions.enter_context(patch.object(raw, 'step_tensor', wrong_terminal))
            self.assertTrue(model.collect_rollouts(model.env, callback, model.rollout_buffer, steps))
        callback.on_training_end()
        self.assertEqual(len(callback.rows), steps)
        self.assertEqual(model.num_timesteps, steps * games * 2)
        self.assertTrue(model.rollout_buffer.full)

        corrected, values, dones, starts = [], [], [], []
        previous_done = torch.ones(games * 2, dtype=torch.float32, device=device)
        timeout_count = 0
        with torch.no_grad():
            for tick, row in enumerate(callback.rows):
                torch.testing.assert_close(row['pre'], initial.to(device), rtol=0, atol=0)
                # Policy-free serial C oracle consumes the exact executed CUDA actions.
                expected = oracle.step_tensor(row['actions'].cpu())
                terminal = expected.terminal_observations.to(device)
                rewards = expected.rewards.to(device)
                term, trunc = expected.terminated.to(device), expected.truncated.to(device)
                torch.testing.assert_close(row['terminal'], terminal, rtol=0, atol=0,
                                           msg=f'tick {tick + 1}: true terminal observation differs')
                torch.testing.assert_close(row['obs'], expected.observations.to(device), rtol=0, atol=0)
                torch.testing.assert_close(row['raw_rewards'], rewards, rtol=0, atol=0)
                torch.testing.assert_close(row['terminated'], term, rtol=0, atol=0)
                torch.testing.assert_close(row['truncated'], trunc, rtol=0, atol=0)
                self.assertFalse(bool(term.any()), 'short horizon must truncate, not terminate')
                self.assertEqual(bool(trunc.all()), (tick + 1) % horizon == 0)
                # Independent value calls use oracle terminal frames, not the collector's
                # captured bootstrap values. Every policy operation stays on CUDA.
                terminal_values = model.policy.predict_values(terminal).flatten()
                expected_rewards = torch.where(trunc & ~term, rewards + model.gamma * terminal_values, rewards)
                corrected.append(expected_rewards)
                values.append(model.policy.predict_values(initial.to(device)).flatten())
                dones.append(term | trunc)
                starts.append(previous_done)
                previous_done = (term | trunc).to(torch.float32)
                if bool(trunc.all()):
                    timeout_count += games
                    self.assertGreater(int(torch.count_nonzero(terminal[:, :192])), 0)
                    self.assertEqual(int(torch.count_nonzero(row['obs'][:, :192])), 0)
                    reset_values = model.policy.predict_values(row['obs']).flatten()
                    self.assertFalse(torch.equal(terminal_values, reset_values),
                                     'fixture must distinguish terminal and reset critic values')
                    self.assertFalse(torch.equal(expected_rewards, rewards), 'nonzero bootstrap required')
                initial = expected.observations.clone()
            corrected, values = torch.stack(corrected), torch.stack(values)
            torch.testing.assert_close(model.rollout_buffer.rewards, corrected, rtol=0, atol=0,
                                       msg='timeout-corrected rewards disagree with independent terminal values')
            torch.testing.assert_close(model.rollout_buffer.values, values, rtol=0, atol=0)
            torch.testing.assert_close(model.rollout_buffer.episode_starts, torch.stack(starts), rtol=0, atol=0)
            # Independent eager GAE recurrence, never calls the production buffer helper.
            advantages = torch.empty_like(corrected)
            future_advantage = torch.zeros(games * 2, device=device)
            next_value = model.policy.predict_values(initial.to(device)).flatten()
            for tick in reversed(range(steps)):
                continuing = 1.0 - dones[tick].to(torch.float32)
                delta = corrected[tick] + model.gamma * next_value * continuing - values[tick]
                future_advantage = delta + model.gamma * model.gae_lambda * continuing * future_advantage
                advantages[tick] = future_advantage
                next_value = values[tick]
            torch.testing.assert_close(model.rollout_buffer.advantages, advantages, rtol=0, atol=0)
            torch.testing.assert_close(model.rollout_buffer.returns, advantages + values, rtol=0, atol=0)
        self.assertEqual(timeout_count, 4)
        self.assertEqual(raw.backend.valid_steps.tolist(), [steps] * games)
        self.assertEqual(raw.backend.errors.tolist(), [0] * games)

        def records(infos):
            return sorted((info['combat']['game'], info['combat']['player'], info['combat']['episode_index'],
                           info['episode']['l'], info['episode']['r'], info['combat']['stocks'],
                           info['combat']['damage_done'], info['combat']['damage_taken'], info['combat']['win_mask'])
                          for info in infos)

        expected_infos = oracle.drain_episode_infos()
        self.assertEqual(len(callback.infos), 8)  # Four actual games, both player records.
        self.assertEqual(records(callback.infos), records(expected_infos))
        self.assertTrue(all(info['episode']['l'] == horizon for info in callback.infos))

    def test_actual_native_timeouts_bootstrap_true_terminal_frames_and_gae(self):
        self._verify_collector()

    def test_reset_observation_substitution_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'true terminal observation differs'):
            self._verify_collector(substitute_reset_terminal=True)


if __name__ == '__main__':
    unittest.main()
