"""Tests for the CUDA-native TensorPPO collection seam."""

from __future__ import annotations

import importlib
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np


def _deps(*, require_cuda=True):
    try:
        import torch
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:  # pragma: no cover - dependency-light local Python
        raise unittest.SkipTest(f"torch/gymnasium/SB3 unavailable: {exc}") from exc
    if require_cuda and not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA unavailable")
    return torch, spaces, PPO, BaseCallback, DummyVecEnv


class TensorPPOStaticTests(unittest.TestCase):
    def test_module_exports_api_without_runtime_dependencies(self):
        module = importlib.import_module("tensor_ppo")

        self.assertTrue(hasattr(module, "TensorPPO"))
        self.assertTrue(hasattr(module, "TensorStep"))
        self.assertTrue(hasattr(module, "TensorVecEnv"))

    def test_source_preserves_sb3_train_and_rejects_cpu_step_fallbacks(self):
        source = (Path(__file__).resolve().parents[1] / "tensor_ppo.py").read_text()

        self.assertIn("super().train()", source)
        self.assertIn("_TensorRolloutStatsProxy", source)
        self.assertNotIn("def step_wait(self):\n        return", source)
        self.assertNotIn("def step_async(self, actions):\n        return", source)
        self.assertNotIn(".cpu().numpy()", source)


class TensorPPODependencyTests(unittest.TestCase):
    def test_tensor_vec_env_rejects_unsupported_spaces_and_cpu_methods(self):
        torch, spaces, _PPO, _BaseCallback, _DummyVecEnv = _deps(require_cuda=False)
        module = importlib.import_module("tensor_ppo")

        class Env:
            num_envs = 2
            observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
            action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

            def reset_tensor(self, seeds=None):
                raise AssertionError("not called")

            def step_tensor(self, clipped_actions):
                raise AssertionError("not called")

        with self.assertRaisesRegex(ValueError, "CUDA-only"):
            module.TensorVecEnv(Env(), device=torch.device("cpu"))

        adapter = module.TensorVecEnv(Env(), device="cuda")
        self.assertEqual(adapter.device, torch.device("cuda:0"))
        with self.assertRaisesRegex(RuntimeError, "CPU"):
            adapter.reset()
        with self.assertRaisesRegex(RuntimeError, "CPU"):
            adapter.step_async(np.zeros((2, 10), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "1D"):
            bad = Env()
            bad.observation_space = spaces.Box(low=-1, high=1, shape=(2, 2), dtype=np.float32)
            module.TensorVecEnv(bad, device=torch.device("cuda:0"))
        with self.assertRaisesRegex(ValueError, "10D"):
            bad = Env()
            bad.action_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
            module.TensorVecEnv(bad, device=torch.device("cuda:0"))
        with self.assertRaisesRegex(ValueError, "num_envs"):
            bad = Env()
            bad.num_envs = 2.0
            module.TensorVecEnv(bad, device=torch.device("cuda:0"))

    def test_tensor_vec_env_shim_forwards_sb3_seed_to_underlying_env(self):
        torch, spaces, _PPO, _BaseCallback, _DummyVecEnv = _deps(require_cuda=False)
        module = importlib.import_module("tensor_ppo")

        class Env:
            num_envs = 3
            observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
            action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

            def __init__(self):
                self.seed_calls = []

            def seed(self, seeds=None):
                self.seed_calls.append(seeds)
                return seeds

            def reset_tensor(self, seeds=None):
                raise AssertionError("not called")

            def step_tensor(self, clipped_actions):
                raise AssertionError("not called")

        env = Env()
        tensor_env = module.TensorVecEnv(env, device=torch.device("cuda:0"))

        returned = tensor_env.as_sb3_vec_env().seed(123)

        self.assertEqual(returned, [123, 124, 125])
        self.assertEqual(env.seed_calls, [[123, 124, 125]])

    def test_tensor_vec_env_validates_optional_host_timeout_hints(self):
        torch, spaces, _PPO, _BaseCallback, _DummyVecEnv = _deps(require_cuda=False)
        module = importlib.import_module("tensor_ppo")

        class Env:
            num_envs = 3
            observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
            action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

            def __init__(self):
                self.indices = np.asarray([1], dtype=np.int64)
                self.has_timeouts = True

            def reset_tensor(self, seeds=None):
                raise AssertionError("not called")

            def step_tensor(self, clipped_actions):
                raise AssertionError("not called")

            def host_timeout_indices(self):
                return self.indices

            def host_has_timeouts(self):
                return self.has_timeouts

        env = Env()
        tensor_env = module.TensorVecEnv(env, device=torch.device("cuda:0"))

        np.testing.assert_array_equal(tensor_env.host_timeout_indices(), np.asarray([1], dtype=np.int64))
        self.assertTrue(tensor_env.host_has_timeouts())
        env.indices = np.asarray([3], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "host_timeout_indices"):
            tensor_env.host_timeout_indices()
        env.indices = np.asarray([1.0], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "integer"):
            tensor_env.host_timeout_indices()
        env.has_timeouts = "yes"
        with self.assertRaisesRegex(ValueError, "bool"):
            tensor_env.host_has_timeouts()


class TensorPPODeferredTimeoutTests(unittest.TestCase):
    """Exercise the real collector on CPU fixtures; CUDA integration is below."""

    def setUp(self):
        self.torch, self.spaces, _PPO, _Callback, _VecEnv = _deps(require_cuda=False)
        self.module = importlib.import_module("tensor_ppo")
        self.device = self.torch.device("cpu")

    def make_fixture(self, *, deferred=True, timeouts=True, stop_after=None, batch_size=3):
        torch, module = self.torch, self.module
        from stable_baselines3.common.buffers import RolloutBuffer
        n_envs, n_steps, obs_dim = 4, 3, 3
        obs_space = self.spaces.Box(-1000, 1000, shape=(obs_dim,), dtype=np.float32)
        action_space = self.spaces.Box(-1, 1, shape=(10,), dtype=np.float32)

        class Env:
            num_envs = n_envs
            obs_shape = (obs_dim,)

            def __init__(self):
                self.step_count = 0
                self.obs = torch.zeros((n_envs, obs_dim))
                self.terminal = torch.zeros_like(self.obs)
                self.terminated = torch.zeros(n_envs, dtype=torch.bool)
                self.truncated = torch.zeros(n_envs, dtype=torch.bool)
                self.reward = torch.zeros(n_envs)

            def step_tensor(self, actions):
                del actions
                step = self.step_count
                self.step_count += 1
                self.obs.fill_(self.step_count)
                # Every returned tensor aliases storage that the next step overwrites.
                self.terminal.copy_(16 * self.step_count + torch.arange(n_envs).unsqueeze(1))
                self.reward.fill_(self.step_count)
                self.terminated.zero_()
                self.truncated.zero_()
                if timeouts:
                    if step == 0:
                        self.truncated[[0, 1]] = True
                        self.terminated[[1, 2]] = True
                    elif step == 1:
                        self.truncated[2] = True
                    else:
                        self.truncated[[0, 3]] = True
                return module.TensorStep(self.obs, self.reward, self.terminated, self.truncated, self.terminal)

            def host_step_infos(self):
                return []

            def host_step_dones(self):
                return None

            def host_timeout_indices(self):
                return None

            def host_has_timeouts(self):
                return None

            def drain_episode_infos(self):
                return [{"episode": {"r": 6.0, "l": 3}}]

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value_inputs = []
                self.value_modes = []

            def set_training_mode(self, training):
                self.train(training)

            def forward(self, obs):
                return torch.zeros((len(obs), 10)), obs[:, :1] * 0.25, torch.zeros(len(obs))

            def predict_values(self, obs):
                self.value_inputs.append(obs.clone())
                self.value_modes.append((self.training, torch.is_grad_enabled()))
                return obs[:, :1] * 0.25

        class Buffer:
            """Tensor storage seam with stock SB3's independently implemented GAE."""

            def __init__(self):
                self.stock = RolloutBuffer(n_steps, obs_space, action_space, device="cpu", gamma=0.9, gae_lambda=0.95, n_envs=n_envs)
                self.reset()

            def reset(self):
                self.stock.reset()
                self.rewards = torch.zeros((n_steps, n_envs))
                self.invalid = torch.zeros((), dtype=torch.bool)
                self.computed = False

            @property
            def pos(self):
                return self.stock.pos

            @property
            def full(self):
                return self.stock.full

            def _accumulate_finite_check(self, data):
                self.invalid.logical_or_(~torch.isfinite(data).all())

            def add(self, obs, action, reward, episode_start, value, log_prob):
                self.rewards[self.pos].copy_(reward)
                self._accumulate_finite_check(reward)
                self.stock.add(obs.numpy(), action.numpy(), reward.numpy(), episode_start.numpy(), value, log_prob)

            def compute_returns_and_advantage(self, last_values, dones):
                if self.invalid:
                    raise ValueError("rollout buffer contains nonfinite data")
                self.stock.rewards[:] = self.rewards.numpy()
                self.stock.compute_returns_and_advantage(last_values, dones.numpy())
                self.advantages = torch.from_numpy(self.stock.advantages.copy())
                self.returns = torch.from_numpy(self.stock.returns.copy())
                self.computed = True

        class Callback:
            def __init__(self):
                self.locals = {}
                self.raw_rewards = []
                self.ends = 0

            def on_rollout_start(self):
                pass

            def update_locals(self, local_vars):
                self.locals.update(local_vars)

            def on_step(self):
                self.raw_rewards.append(self.locals["rewards"].clone())
                return stop_after is None or len(self.raw_rewards) < stop_after

            def on_rollout_end(self):
                self.ends += 1
                self.end_rewards = self.locals["rewards"].clone()

        model = object.__new__(module.TensorPPO)
        model.tensor_vec_env = Env()
        model.policy = Policy()
        model.device = self.device
        model.gamma = 0.9
        model.num_timesteps = 0
        model._last_obs = model.tensor_vec_env.obs.clone()
        model._last_episode_starts = torch.ones(n_envs)
        model._action_low_tensor = torch.full((10,), -1.0)
        model._action_high_tensor = torch.full((10,), 1.0)
        model.ep_info_buffer = model.ep_success_buffer = None
        model._stats_window_size = 10
        if deferred is not None:
            model.defer_timeout_values = deferred
        model.terminal_value_batch_size = batch_size
        model.rollout_buffer = Buffer()
        return model, Callback()

    def collect(self, model, callback):
        return model.collect_rollouts(None, callback, model.rollout_buffer, n_rollout_steps=3)

    def test_deferred_timeout_values_default_is_off(self):
        self.assertFalse(self.module.TensorPPO.defer_timeout_values)

    def test_deferred_timeout_values_default_keeps_immediate_full_batch_bootstrap(self):
        model, callback = self.make_fixture(deferred=None, batch_size=0)
        self.assertTrue(self.collect(model, callback))
        self.assertEqual([len(obs) for obs in model.policy.value_inputs], [4, 4, 4, 4])

    def test_deferred_timeout_values_expose_uncorrected_prior_rewards_until_boundary(self):
        prior_rewards = []
        for deferred in (False, True):
            model, callback = self.make_fixture(deferred=deferred)
            original = callback.on_step
            observed = []

            def record_prior_rewards():
                if model.rollout_buffer.pos:
                    observed.append(model.rollout_buffer.rewards[0, 0].clone())
                return original()

            callback.on_step = record_prior_rewards
            self.assertTrue(self.collect(model, callback))
            prior_rewards.append(observed)
            self.assertAlmostEqual(float(model.rollout_buffer.rewards[0, 0]), 4.6, places=5)
        self.torch.testing.assert_close(self.torch.stack(prior_rewards[0]), self.torch.full((2,), 4.6), rtol=0, atol=0)
        self.torch.testing.assert_close(self.torch.stack(prior_rewards[1]), self.torch.ones(2), rtol=0, atol=0)

    def test_deferred_timeout_values_snapshot_aliases_and_preserve_rewards_and_gae(self):
        baseline, baseline_callback = self.make_fixture(deferred=False)
        model, callback = self.make_fixture()
        self.assertTrue(self.collect(baseline, baseline_callback))
        self.assertTrue(self.collect(model, callback))

        torch = self.torch
        for name in ("rewards", "advantages", "returns"):
            torch.testing.assert_close(getattr(model.rollout_buffer, name), getattr(baseline.rollout_buffer, name), rtol=0, atol=0)
        # Only four true timeouts across the entire rollout, in bounded batches.
        timeout_inputs = [obs for obs in model.policy.value_inputs if bool((obs >= 16).all())]
        self.assertEqual([len(obs) for obs in timeout_inputs], [3, 1])
        torch.testing.assert_close(torch.cat(timeout_inputs)[:, 0], torch.tensor([16., 34., 48., 51.]), rtol=0, atol=0)
        self.assertEqual(model.policy.value_modes, [(False, False)] * 3)
        torch.testing.assert_close(model.rollout_buffer.rewards[0, 1:3], torch.ones(2), rtol=0, atol=0)
        for step, reward in enumerate(callback.raw_rewards, 1):
            torch.testing.assert_close(reward, torch.full((4,), float(step)), rtol=0, atol=0)
        torch.testing.assert_close(callback.end_rewards, baseline_callback.end_rewards, rtol=0, atol=0)
        self.assertEqual(list(model.ep_info_buffer), [{"r": 6.0, "l": 3}])

    def test_deferred_timeout_values_skip_all_terminal_calls_without_timeouts(self):
        model, callback = self.make_fixture(timeouts=False)
        self.assertTrue(self.collect(model, callback))
        self.assertEqual(len(model.policy.value_inputs), 1)
        self.torch.testing.assert_close(model.rollout_buffer.rewards, self.torch.arange(1., 4.).unsqueeze(1).expand(3, 4), rtol=0, atol=0)

    def test_deferred_timeout_values_cancelled_rollout_corrects_only_accepted_steps(self):
        for stop_after in (1, 2, 3):
            with self.subTest(stop_after=stop_after):
                baseline, baseline_callback = self.make_fixture(deferred=False, stop_after=stop_after)
                model, callback = self.make_fixture(stop_after=stop_after)
                self.assertFalse(self.collect(baseline, baseline_callback))
                self.assertFalse(self.collect(model, callback))
                self.assertEqual(model.rollout_buffer.pos, stop_after - 1)
                self.assertFalse(model.rollout_buffer.computed)
                self.assertEqual(callback.ends, 0)
                self.torch.testing.assert_close(model.rollout_buffer.rewards, baseline.rollout_buffer.rewards, rtol=0, atol=0)
                self.assertEqual(sum(len(obs) for obs in model.policy.value_inputs), max(0, stop_after - 1))

    def test_deferred_timeout_values_reject_invalid_batch_size_before_stepping(self):
        for batch_size in (0, -1, True, 1.5):
            with self.subTest(batch_size=batch_size):
                model, callback = self.make_fixture(batch_size=batch_size)
                with self.assertRaisesRegex(ValueError, "terminal_value_batch_size"):
                    self.collect(model, callback)
                self.assertEqual(model.tensor_vec_env.step_count, 0)

    def test_deferred_timeout_values_nonfinite_bootstrap_is_rejected(self):
        model, callback = self.make_fixture()
        original = model.policy.predict_values

        def nonfinite_terminal_value(obs):
            return original(obs) * self.torch.where(obs[:, :1] >= 16, float("nan"), 1.0)

        model.policy.predict_values = nonfinite_terminal_value
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.collect(model, callback)

    def test_deferred_timeout_values_release_snapshots_on_completion_cancellation_and_error(self):
        for outcome in ("complete", "cancel", "error"):
            with self.subTest(outcome=outcome):
                model, callback = self.make_fixture(stop_after=2 if outcome == "cancel" else None)
                if outcome == "error":
                    original = callback.on_step

                    def raising_callback():
                        result = original()
                        if len(callback.raw_rewards) == 2:
                            raise RuntimeError("callback failed")
                        return result

                    callback.on_step = raising_callback
                    with self.assertRaisesRegex(RuntimeError, "callback failed"):
                        self.collect(model, callback)
                else:
                    self.collect(model, callback)
                pending = callback.locals["deferred_timeouts"]
                self.assertIsNone(pending.observations)
                self.assertIsNone(pending.timeout_masks)

    def test_deferred_timeout_values_reject_malformed_critic_outputs(self):
        for bad in ("shape", "dtype", "device"):
            with self.subTest(bad=bad):
                model, callback = self.make_fixture()
                original = model.policy.predict_values

                def malformed(obs):
                    values = original(obs)
                    if bad == "shape":
                        return values[:1]
                    if bad == "dtype":
                        return values.double()
                    return values.to("meta")

                model.policy.predict_values = malformed
                with self.assertRaisesRegex(ValueError, "terminal values"):
                    self.collect(model, callback)

    def test_deferred_timeout_values_repeated_rollout_does_not_reapply_old_timeouts(self):
        model, callback = self.make_fixture()
        self.assertTrue(self.collect(model, callback))
        first_call_count = len(model.policy.value_inputs)
        model.tensor_vec_env.host_has_timeouts = lambda: False
        self.assertTrue(self.collect(model, callback))
        self.assertEqual(len(model.policy.value_inputs) - first_call_count, 1)
        self.torch.testing.assert_close(model.rollout_buffer.rewards, self.torch.arange(4., 7.).unsqueeze(1).expand(3, 4), rtol=0, atol=0)


class TensorPPOCudaTests(unittest.TestCase):
    def setUp(self):
        torch, spaces, PPO, BaseCallback, DummyVecEnv = _deps()
        self.torch = torch
        self.spaces = spaces
        self.PPO = PPO
        self.BaseCallback = BaseCallback
        self.DummyVecEnv = DummyVecEnv
        self.module = importlib.import_module("tensor_ppo")
        self.buffer_module = importlib.import_module("tensor_rollout_buffer")
        self.device = torch.device("cuda", torch.cuda.current_device())

    def make_env(self, *, n_envs=4, obs_dim=3):
        torch = self.torch
        spaces = self.spaces
        device = self.device
        import gymnasium as gym

        class Env:
            def __init__(self):
                self.num_envs = n_envs
                self.observation_space = spaces.Box(low=-100, high=100, shape=(obs_dim,), dtype=np.float32)
                self.action_space = spaces.Box(low=-1e-4, high=1e-4, shape=(10,), dtype=np.float32)
                self.obs = torch.zeros((n_envs, obs_dim), dtype=torch.float32, device=device)
                self.step_count = 0
                self.reset_calls = 0
                self.seen_actions = []
                self.terminal_batches = []
                self.timeout_indices = torch.empty((0,), dtype=torch.int64, device=device)

            def reset_tensor(self, seeds=None):
                self.reset_calls += 1
                self.step_count = 0
                base = torch.arange(n_envs, dtype=torch.float32, device=device).unsqueeze(1)
                self.obs = base + torch.arange(obs_dim, dtype=torch.float32, device=device)
                return self.obs

            def step_tensor(self, clipped_actions):
                self.seen_actions.append(clipped_actions.detach().clone())
                self.step_count += 1
                terminal_obs = self.obs + 1000.0
                terminated = torch.zeros((n_envs,), dtype=torch.bool, device=device)
                truncated = torch.zeros((n_envs,), dtype=torch.bool, device=device)
                if self.step_count == 1:
                    terminated[1] = True
                    truncated[2] = True
                self.timeout_indices = torch.nonzero(truncated & ~terminated, as_tuple=False).flatten()
                rewards = torch.arange(n_envs, dtype=torch.float32, device=device) + self.step_count
                self.obs = self.obs + 10.0
                reset_obs = torch.full_like(self.obs, -7.0)
                self.obs = torch.where((terminated | truncated).unsqueeze(1), reset_obs, self.obs)
                self.terminal_batches.append(terminal_obs)
                return self_module.TensorStep(self.obs, rewards, terminated, truncated, terminal_obs)

            def drain_episode_infos(self):
                return [{"episode": {"r": 1.0, "l": 1}}]

        self_module = self.module
        return Env()

    def make_no_timeout_env_with_host_hint(self, *, n_envs=4, obs_dim=3):
        torch = self.torch
        spaces = self.spaces
        device = self.device

        class Env:
            def __init__(self):
                self.num_envs = n_envs
                self.observation_space = spaces.Box(low=-100, high=100, shape=(obs_dim,), dtype=np.float32)
                self.action_space = spaces.Box(low=-1e-4, high=1e-4, shape=(10,), dtype=np.float32)
                self.obs = torch.zeros((n_envs, obs_dim), dtype=torch.float32, device=device)

            def reset_tensor(self, seeds=None):
                del seeds
                self.obs.zero_()
                return self.obs

            def step_tensor(self, clipped_actions):
                del clipped_actions
                self.obs = self.obs + 1.0
                return self_module.TensorStep(
                    self.obs,
                    torch.ones((n_envs,), dtype=torch.float32, device=device),
                    torch.zeros((n_envs,), dtype=torch.bool, device=device),
                    torch.zeros((n_envs,), dtype=torch.bool, device=device),
                )

            def host_timeout_indices(self):
                return np.empty((0,), dtype=np.int64)

        self_module = self.module
        return Env()

    def make_model(self, env, *, n_steps=2, batch_size=4, n_epochs=2):
        adapter = self.module.TensorVecEnv(env, device=self.device)
        model = self.module.TensorPPO(
            "MlpPolicy",
            adapter,
            device=self.device,
            seed=123,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.9,
            gae_lambda=0.95,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [8]},
            verbose=0,
        )
        model._last_obs = adapter.reset_tensor()
        model._last_episode_starts = self.torch.ones((env.num_envs,), dtype=self.torch.float32, device=self.device)
        return model

    def make_callback(self, model, *, stop_after=None, init=True):
        base = self.BaseCallback

        class Callback(base):
            def __init__(self):
                super().__init__()
                self.rollout_starts = 0
                self.rollout_ends = 0
                self.steps = 0

            def _on_rollout_start(self):
                self.rollout_starts += 1

            def _on_step(self):
                self.steps += 1
                return stop_after is None or self.steps < stop_after

            def _on_rollout_end(self):
                self.rollout_ends += 1

        callback = Callback()
        if init:
            callback.init_callback(model)
        return callback

    def make_paired_fixture_envs(self, *, n_envs=4, obs_dim=12):
        torch = self.torch
        spaces = self.spaces
        device = self.device
        import gymnasium as gym
        action_space = spaces.Box(low=-1e-4, high=1e-4, shape=(10,), dtype=np.float32)
        observation_space = spaces.Box(low=-1000, high=1000, shape=(obs_dim,), dtype=np.float32)

        def make_obs_np(lane, step):
            frames = []
            for frame in range(4):
                base = np.float32(lane * 10 + step * 0.5 + frame)
                frames.extend([base, np.float32(base + 0.125), np.float32(base - 0.25)])
            return np.asarray(frames, dtype=np.float32)

        def flags(lane, step):
            terminated = step == 0 and lane in (1, 3)
            truncated = step == 0 and lane in (2, 3)
            return terminated, truncated

        def reward_np(lane, step, clipped_action):
            return np.float32(0.25 * lane + 0.5 * (step + 1) + 0.01 * np.asarray(clipped_action, dtype=np.float32).sum())

        class StockLaneEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self, lane):
                self.lane = lane
                self.observation_space = observation_space
                self.action_space = action_space
                self.step_count = 0
                self.seen_actions = []
                self.reset_calls = 0
                self.obs = make_obs_np(lane, 0)

            def reset(self, seed=None, options=None):
                del seed, options
                self.reset_calls += 1
                self.step_count = 0
                self.obs = make_obs_np(self.lane, self.step_count)
                return self.obs.copy(), {}

            def step(self, action):
                clipped = np.asarray(action, dtype=np.float32).copy()
                self.seen_actions.append(clipped.copy())
                terminal_obs = self.obs.copy()
                terminated, truncated = flags(self.lane, self.step_count)
                reward = reward_np(self.lane, self.step_count, clipped)
                self.step_count += 1
                next_obs = make_obs_np(self.lane, self.step_count)
                self.obs = next_obs
                obs_for_step = terminal_obs if (terminated or truncated) else next_obs
                info = {"lane": self.lane}
                return obs_for_step.copy(), reward, terminated, truncated, info

        class TensorFixtureEnv:
            def __init__(self):
                self.num_envs = n_envs
                self.observation_space = observation_space
                self.action_space = action_space
                self.step_counts = torch.zeros((n_envs,), dtype=torch.int64, device=device)
                self.obs = torch.empty((n_envs, obs_dim), dtype=torch.float32, device=device)
                self.seen_actions = []
                self.reset_calls = 0
                self.reset_tensor()

            def _obs_batch(self):
                lanes = torch.arange(n_envs, dtype=torch.float32, device=device).unsqueeze(1)
                steps = self.step_counts.to(dtype=torch.float32).unsqueeze(1)
                parts = []
                for frame in range(4):
                    base = lanes * 10.0 + steps * 0.5 + float(frame)
                    parts.extend([base, base + 0.125, base - 0.25])
                return torch.cat(parts, dim=1).contiguous()

            def reset_tensor(self, seeds=None):
                del seeds
                self.reset_calls += 1
                self.step_counts.zero_()
                self.obs.copy_(self._obs_batch())
                return self.obs

            def step_tensor(self, clipped_actions):
                self.seen_actions.append(clipped_actions.detach().clone())
                terminal_obs = self.obs.clone()
                lanes = torch.arange(n_envs, device=device)
                terminated = (self.step_counts == 0) & ((lanes == 1) | (lanes == 3))
                truncated = (self.step_counts == 0) & ((lanes == 2) | (lanes == 3))
                rewards = lanes.to(dtype=torch.float32) * 0.25 + (self.step_counts.to(dtype=torch.float32) + 1.0) * 0.5
                rewards = rewards + clipped_actions.sum(dim=1) * 0.01
                self.step_counts.add_(1)
                done = terminated | truncated
                self.step_counts.masked_fill_(done, 0)
                self.obs.copy_(self._obs_batch())
                return self_module.TensorStep(self.obs, rewards.contiguous(), terminated.contiguous(), truncated.contiguous(), terminal_obs)

            def drain_episode_infos(self):
                return [{"episode": {"r": 1.0, "l": 1}} for _ in range(n_envs)]

        self_module = self.module
        stock_lanes = [StockLaneEnv(i) for i in range(n_envs)]
        stock_env = self.DummyVecEnv([lambda lane=lane: lane for lane in stock_lanes])
        tensor_env = TensorFixtureEnv()
        return stock_env, stock_lanes, tensor_env

    def make_paired_models(self, *, n_steps=3, batch_size=4, n_epochs=8, target_kl=None):
        torch = self.torch
        stock_env, stock_lanes, tensor_env = self.make_paired_fixture_envs()
        tensor_adapter = self.module.TensorVecEnv(tensor_env, device=self.device)
        common_kwargs = dict(
            device=self.device,
            seed=20260906,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.9,
            gae_lambda=0.8,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=target_kl,
            policy_kwargs={"net_arch": [16]},
            learning_rate=0.001,
            verbose=0,
        )
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        np.random.seed(777)
        stock_model = self.PPO("MlpPolicy", stock_env, **common_kwargs)
        tensor_model = self.module.TensorPPO("MlpPolicy", tensor_adapter, **common_kwargs)
        tensor_model.policy.load_state_dict(stock_model.policy.state_dict())
        tensor_model.policy.optimizer.load_state_dict(stock_model.policy.optimizer.state_dict())
        return stock_model, stock_lanes, tensor_model, tensor_env

    def reset_rng_to(self, np_state, cpu_state, cuda_states):
        self.torch.set_rng_state(cpu_state)
        self.torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(np_state)

    def collect_with_same_rng(self, stock_model, tensor_model):
        torch = self.torch
        total_timesteps = stock_model.n_steps * stock_model.env.num_envs
        stock_raw_callback = self.make_callback(stock_model, init=False)
        tensor_raw_callback = self.make_callback(tensor_model, init=False)
        _stock_total, stock_callback = stock_model._setup_learn(
            total_timesteps=total_timesteps,
            callback=stock_raw_callback,
            reset_num_timesteps=True,
        )
        _tensor_total, tensor_callback = tensor_model._setup_learn(
            total_timesteps=total_timesteps,
            callback=tensor_raw_callback,
            reset_num_timesteps=True,
        )
        np_state = np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()

        self.reset_rng_to(np_state, cpu_state, cuda_states)
        stock_ok = stock_model.collect_rollouts(stock_model.env, stock_callback, stock_model.rollout_buffer, stock_model.n_steps)
        self.reset_rng_to(np_state, cpu_state, cuda_states)
        tensor_ok = tensor_model.collect_rollouts(tensor_model.env, tensor_callback, tensor_model.rollout_buffer, tensor_model.n_steps)
        stock_model._update_current_progress_remaining(stock_model.num_timesteps, stock_model._total_timesteps)
        tensor_model._update_current_progress_remaining(tensor_model.num_timesteps, tensor_model._total_timesteps)

        self.assertTrue(stock_ok)
        self.assertTrue(tensor_ok)
        return stock_callback, tensor_callback

    def assert_rollout_buffers_close(self, stock_buffer, tensor_buffer):
        torch = self.torch
        comparisons = [
            ("observations", stock_buffer.observations, tensor_buffer.observations),
            ("actions", stock_buffer.actions, tensor_buffer.actions),
            ("rewards", stock_buffer.rewards, tensor_buffer.rewards),
            ("episode_starts", stock_buffer.episode_starts, tensor_buffer.episode_starts),
            ("values", stock_buffer.values, tensor_buffer.values),
            ("log_probs", stock_buffer.log_probs, tensor_buffer.log_probs),
            ("advantages", stock_buffer.advantages, tensor_buffer.advantages),
            ("returns", stock_buffer.returns, tensor_buffer.returns),
        ]
        for name, expected, actual in comparisons:
            with self.subTest(field=name):
                torch.testing.assert_close(actual.detach().cpu(), torch.as_tensor(expected), rtol=0, atol=2e-5)

    def assert_models_close(self, stock_model, tensor_model):
        torch = self.torch
        for (name, expected), (_, actual) in zip(stock_model.policy.state_dict().items(), tensor_model.policy.state_dict().items()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=2e-6)
        stock_opt = stock_model.policy.optimizer.state_dict()
        tensor_opt = tensor_model.policy.optimizer.state_dict()
        self.assertEqual(stock_opt["param_groups"], tensor_opt["param_groups"])
        self.assertEqual(stock_opt["state"].keys(), tensor_opt["state"].keys())
        for key in stock_opt["state"]:
            for field in stock_opt["state"][key]:
                expected = stock_opt["state"][key][field]
                actual = tensor_opt["state"][key][field]
                if torch.is_tensor(expected):
                    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=2e-6)
                else:
                    self.assertEqual(actual, expected)

    def test_collect_rollouts_stores_raw_actions_bootstraps_timeouts_and_snapshots_obs(self):
        torch = self.torch
        env = self.make_env(n_envs=4)
        model = self.make_model(env, n_steps=2)
        buffer = model.rollout_buffer
        callback = self.make_callback(model)

        ok = model.collect_rollouts(model.env, callback, buffer, n_rollout_steps=2)

        self.assertTrue(ok)
        self.assertTrue(buffer.full)
        self.assertEqual(model.num_timesteps, 8)
        self.assertFalse(torch.equal(buffer.observations[0], env.obs))
        self.assertTrue(torch.all(env.seen_actions[0] <= 1e-4))
        self.assertTrue(torch.all(env.seen_actions[0] >= -1e-4))
        self.assertTrue(torch.any(buffer.actions[0] > 1e-4) | torch.any(buffer.actions[0] < -1e-4))
        self.assertTrue(torch.equal(buffer.episode_starts[1], torch.tensor([0, 1, 1, 0], dtype=torch.float32, device=self.device)))
        self.assertEqual(callback.rollout_starts, 1)
        self.assertEqual(callback.steps, 2)
        self.assertEqual(callback.rollout_ends, 1)

    def test_collect_rollouts_skips_terminal_value_batch_when_host_reports_no_timeouts(self):
        env = self.make_no_timeout_env_with_host_hint(n_envs=4)
        model = self.make_model(env, n_steps=3)
        callback = self.make_callback(model)
        original_predict_values = model.policy.predict_values
        call_shapes = []

        def counted_predict_values(obs):
            call_shapes.append(tuple(obs.shape))
            return original_predict_values(obs)

        model.policy.predict_values = counted_predict_values
        try:
            ok = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=3)
        finally:
            model.policy.predict_values = original_predict_values

        self.assertTrue(ok)
        self.assertEqual(call_shapes, [(4, 3)])

    def test_collect_rollouts_early_stop_matches_sb3_before_storing_step(self):
        env = self.make_env(n_envs=4)
        model = self.make_model(env, n_steps=2)
        callback = self.make_callback(model, stop_after=1)

        ok = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=2)

        self.assertFalse(ok)
        self.assertEqual(model.rollout_buffer.pos, 0)
        self.assertFalse(model.rollout_buffer.full)
        self.assertEqual(callback.rollout_starts, 1)
        self.assertEqual(callback.steps, 1)
        self.assertEqual(callback.rollout_ends, 0)

    def test_deferred_cuda_critic_matches_rollout_and_updated_policy_with_tolerance(self):
        torch = self.torch
        _stock_env, _lanes, baseline_env = self.make_paired_fixture_envs()
        _stock_env, _lanes, deferred_env = self.make_paired_fixture_envs()
        # Mirror the native environment's reused terminal observation and flag buffers.
        terminal_storage = torch.empty((4, 12), device=self.device)
        terminated_storage = torch.empty(4, dtype=torch.bool, device=self.device)
        truncated_storage = torch.empty_like(terminated_storage)
        original_step = deferred_env.step_tensor

        def aliased_step(actions):
            step = original_step(actions)
            terminal_storage.copy_(step.terminal_observations)
            terminated_storage.copy_(step.terminated)
            truncated_storage.copy_(step.truncated)
            return self.module.TensorStep(step.observations, step.rewards, terminated_storage, truncated_storage, terminal_storage)

        deferred_env.step_tensor = aliased_step
        baseline = self.make_model(baseline_env, n_steps=3, n_epochs=2)
        deferred = self.make_model(deferred_env, n_steps=3, n_epochs=2)
        deferred.defer_timeout_values = True
        deferred.terminal_value_batch_size = 2
        callbacks = []
        for model in (baseline, deferred):
            _, callback = model._setup_learn(12, self.make_callback(model, init=False))
            callbacks.append(callback)
        np_state = np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()
        initial_weights = {key: value.clone() for key, value in deferred.policy.state_dict().items()}
        original_predict = deferred.policy.predict_values
        call_shapes = []

        def counted_predict(obs):
            call_shapes.append(tuple(obs.shape))
            self.assertFalse(deferred.policy.training)
            self.assertFalse(torch.is_grad_enabled())
            return original_predict(obs)

        deferred.policy.predict_values = counted_predict
        try:
            for model, callback in zip((baseline, deferred), callbacks):
                self.reset_rng_to(np_state, cpu_state, cuda_states)
                self.assertTrue(model.collect_rollouts(model.env, callback, model.rollout_buffer, 3))
                model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
        finally:
            deferred.policy.predict_values = original_predict
        self.assertEqual(call_shapes, [(2, 12), (1, 12), (4, 12)])
        for key, expected in initial_weights.items():
            torch.testing.assert_close(deferred.policy.state_dict()[key], expected, rtol=0, atol=0)
        for name in ("observations", "actions", "episode_starts", "values", "log_probs"):
            torch.testing.assert_close(getattr(deferred.rollout_buffer, name), getattr(baseline.rollout_buffer, name), rtol=0, atol=0)
        for name in ("rewards", "advantages", "returns"):
            torch.testing.assert_close(getattr(deferred.rollout_buffer, name), getattr(baseline.rollout_buffer, name), rtol=0, atol=2e-5)
        for model in (baseline, deferred):
            np.random.seed(4242)
            model.train()
        self.assert_models_close(baseline, deferred)

    def test_deferred_cuda_critic_handles_no_timeouts_with_and_without_host_hint(self):
        for host_hint in (True, False):
            with self.subTest(host_hint=host_hint):
                env = self.make_no_timeout_env_with_host_hint(n_envs=4)
                if not host_hint:
                    original_step = env.step_tensor

                    def no_hint_step(actions):
                        step = original_step(actions)
                        return self.module.TensorStep(step.observations, step.rewards, step.terminated, step.truncated, step.observations)

                    env.step_tensor = no_hint_step
                    env.host_timeout_indices = lambda: None
                model = self.make_model(env, n_steps=3)
                model.defer_timeout_values = True
                original_predict = model.policy.predict_values
                call_shapes = []

                def counted_predict(obs):
                    call_shapes.append(tuple(obs.shape))
                    return original_predict(obs)

                model.policy.predict_values = counted_predict
                self.assertTrue(model.collect_rollouts(model.env, self.make_callback(model), model.rollout_buffer, 3))
                self.assertEqual(call_shapes, [(4, 3)])
                self.torch.testing.assert_close(model.rollout_buffer.rewards, self.torch.ones_like(model.rollout_buffer.rewards), rtol=0, atol=0)

    def test_deferred_cuda_critic_rejects_nonfinite_bootstrap_before_gae(self):
        model = self.make_model(self.make_env(), n_steps=2)
        model.defer_timeout_values = True
        original_predict = model.policy.predict_values

        def nonfinite_predict(obs):
            return original_predict(obs) * float("nan")

        model.policy.predict_values = nonfinite_predict
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            model.collect_rollouts(model.env, self.make_callback(model), model.rollout_buffer, 2)

    def test_tensor_step_validation_rejects_bad_device_dtype_or_shape(self):
        torch = self.torch
        env = self.make_env(n_envs=2)
        adapter = self.module.TensorVecEnv(env, device=self.device)
        good = self.module.TensorStep(
            torch.zeros((2, 3), dtype=torch.float32, device=self.device),
            torch.zeros((2,), dtype=torch.float32, device=self.device),
            torch.zeros((2,), dtype=torch.bool, device=self.device),
            torch.zeros((2,), dtype=torch.bool, device=self.device),
            torch.zeros((2, 3), dtype=torch.float32, device=self.device),
        )

        adapter.validate_step(good)
        with self.assertRaisesRegex(ValueError, "observations"):
            adapter.validate_step(self.module.TensorStep(good.observations[:, :2], good.rewards, good.terminated, good.truncated, good.terminal_observations))
        with self.assertRaisesRegex(ValueError, "rewards"):
            adapter.validate_step(self.module.TensorStep(good.observations, good.rewards.double(), good.terminated, good.truncated, good.terminal_observations))
        with self.assertRaisesRegex(ValueError, "terminal_observations"):
            adapter.validate_step(self.module.TensorStep(good.observations, good.rewards, good.terminated, good.truncated, None))

    def test_train_uses_sb3_train_with_temporary_numpy_stats_proxy_and_restores_buffer(self):
        env = self.make_env(n_envs=4)
        model = self.make_model(env, n_steps=2, batch_size=4, n_epochs=1)
        _total_timesteps, callback = model._setup_learn(total_timesteps=8, callback=None, reset_num_timesteps=False)
        self.assertTrue(model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=2))
        original_buffer = model.rollout_buffer

        model.train()

        self.assertIs(model.rollout_buffer, original_buffer)
        self.assertGreaterEqual(model._n_updates, 1)

    def test_rollout_collection_matches_stock_sb3_with_numpy_fixture_env(self):
        torch = self.torch
        stock_model, stock_lanes, tensor_model, tensor_env = self.make_paired_models(n_steps=3, batch_size=4, n_epochs=1)

        stock_callback, tensor_callback = self.collect_with_same_rng(stock_model, tensor_model)

        self.assert_rollout_buffers_close(stock_model.rollout_buffer, tensor_model.rollout_buffer)
        clipped_stock_actions = torch.as_tensor(np.stack([lane.seen_actions[0] for lane in stock_lanes]), dtype=torch.float32)
        torch.testing.assert_close(tensor_env.seen_actions[0].detach().cpu(), clipped_stock_actions, rtol=0, atol=0)
        self.assertTrue(torch.any(tensor_model.rollout_buffer.actions[0] > 1e-4) | torch.any(tensor_model.rollout_buffer.actions[0] < -1e-4))
        # Lane 3 is both terminated and truncated.  It must not receive the timeout bootstrap bonus.
        lane3_raw_reward = 0.25 * 3 + 0.5 + 0.01 * float(tensor_env.seen_actions[0][3].sum())
        torch.testing.assert_close(tensor_model.rollout_buffer.rewards[0, 3].detach().cpu(), torch.tensor(lane3_raw_reward), rtol=0, atol=2e-5)
        torch.testing.assert_close(tensor_model.rollout_buffer.rewards.detach().cpu(), torch.as_tensor(stock_model.rollout_buffer.rewards), rtol=0, atol=2e-5)
        self.assertFalse(torch.equal(tensor_model.rollout_buffer.observations[0], tensor_env.obs))
        self.assertEqual(stock_callback.steps, tensor_callback.steps)
        self.assertEqual(stock_callback.rollout_ends, tensor_callback.rollout_ends)

    def test_train_matches_stock_sb3_parameters_gradients_optimizer_and_epochs(self):
        torch = self.torch
        stock_model, _stock_lanes, tensor_model, _tensor_env = self.make_paired_models(n_steps=3, batch_size=4, n_epochs=8)
        self.collect_with_same_rng(stock_model, tensor_model)

        np.random.seed(4242)
        stock_model.train()
        np.random.seed(4242)
        original_buffer = tensor_model.rollout_buffer
        tensor_model.train()

        self.assertIs(tensor_model.rollout_buffer, original_buffer)
        self.assertEqual(stock_model._n_updates, 8)
        self.assertEqual(tensor_model._n_updates, stock_model._n_updates)
        self.assert_models_close(stock_model, tensor_model)
        for (name, stock_param), (_, tensor_param) in zip(stock_model.policy.named_parameters(), tensor_model.policy.named_parameters()):
            if stock_param.grad is not None:
                with self.subTest(gradient=name):
                    torch.testing.assert_close(tensor_param.grad.detach().cpu(), stock_param.grad.detach().cpu(), rtol=0, atol=2e-6)

    def test_target_kl_early_stop_matches_stock_sb3(self):
        stock_model, _stock_lanes, tensor_model, _tensor_env = self.make_paired_models(n_steps=3, batch_size=4, n_epochs=8, target_kl=1e-12)
        self.collect_with_same_rng(stock_model, tensor_model)

        np.random.seed(31337)
        stock_model.train()
        np.random.seed(31337)
        tensor_model.train()

        self.assertEqual(tensor_model._n_updates, stock_model._n_updates)
        self.assertLess(tensor_model._n_updates, 8)
        self.assert_models_close(stock_model, tensor_model)

    def test_from_sb3_preserves_global_rng_state_around_tensor_constructor(self):
        torch = self.torch
        stock_model, _stock_lanes, _tensor_model, tensor_env = self.make_paired_models(n_steps=3, batch_size=4, n_epochs=1)
        tensor_adapter = self.module.TensorVecEnv(tensor_env, device=self.device)
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        expected_py = random.getstate()
        expected_np = np.random.get_state()
        expected_cpu = torch.get_rng_state()
        expected_cuda = torch.cuda.get_rng_state_all()

        _converted = self.module.TensorPPO.from_sb3(stock_model, tensor_adapter)

        py_state = random.getstate()
        np_state = np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all()
        self.assertEqual(py_state, expected_py)
        for exp, act in zip(expected_np, np_state):
            if isinstance(exp, np.ndarray):
                np.testing.assert_array_equal(act, exp)
            else:
                self.assertEqual(act, exp)
        torch.testing.assert_close(cpu_state, expected_cpu, rtol=0, atol=0)
        for exp, act in zip(expected_cuda, cuda_state):
            torch.testing.assert_close(act, exp, rtol=0, atol=0)

    def test_tensor_ppo_save_excludes_live_tensor_env_and_loads_as_standard_ppo(self):
        import zipfile

        spaces = self.spaces
        import gymnasium as gym

        class StockEnv(gym.Env):
            metadata = {"render_modes": []}
            observation_space = spaces.Box(-1, 1, shape=(256,), dtype=np.float32)
            action_space = spaces.Box(-1, 1, shape=(10,), dtype=np.float32)

            def reset(self, seed=None, options=None):
                del seed, options
                return np.zeros((256,), dtype=np.float32), {}

            def step(self, action):
                del action
                return np.zeros((256,), dtype=np.float32), 0.0, False, False, {}

        vec_env = self.DummyVecEnv([lambda: StockEnv() for _ in range(4)])
        cpu_adapter = importlib.import_module("tensor_cpu_vec_env").CPUGameTensorAdapter(vec_env, device=self.device)
        tensor_env = self.module.TensorVecEnv(cpu_adapter, device=self.device)
        model = self.module.TensorPPO(
            "MlpPolicy",
            tensor_env,
            device=self.device,
            seed=55,
            n_steps=2,
            batch_size=4,
            n_epochs=1,
            policy_kwargs={"net_arch": [8]},
            verbose=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tensor_model.zip"
            model.save(path)
            with zipfile.ZipFile(path) as archive:
                serialized_data = archive.read("data").decode("utf-8")
            loaded_as_ppo = self.PPO.load(path, env=vec_env, device=self.device)
            loaded_cpu = self.PPO.load(path, env=vec_env, device="cpu")
            with self.assertRaisesRegex(ValueError, "CUDA-only"):
                self.module.TensorPPO.load(path, env=tensor_env, device="cpu")

        self.assertNotIn("tensor_vec_env", serialized_data)
        self.assertNotIn("_action_low_tensor", serialized_data)
        self.assertNotIn("_action_high_tensor", serialized_data)
        self.assertNotIn("_last_obs", serialized_data)
        self.assertNotIn("_last_episode_starts", serialized_data)
        self.assertNotIn("rollout_buffer_class", serialized_data)
        self.assertNotIn("TensorRolloutBuffer", serialized_data)
        self.assertIsInstance(loaded_as_ppo, self.PPO)
        self.assertIsInstance(loaded_cpu, self.PPO)
        self.assertIsNone(loaded_cpu._last_obs)

    def test_public_learn_uses_tensor_reset_for_fresh_and_repeated_runs(self):
        env = self.make_env(n_envs=4)
        model = self.make_model(env, n_steps=2, batch_size=4, n_epochs=1)
        model._last_obs = None
        model._last_episode_starts = None

        model.learn(total_timesteps=8, reset_num_timesteps=True)
        first_reset_calls = env.reset_calls if hasattr(env, "reset_calls") else 1
        first_timesteps = model.num_timesteps
        model.learn(total_timesteps=8, reset_num_timesteps=False)
        self.assertEqual(model.num_timesteps, first_timesteps + 8)
        if hasattr(env, "reset_calls"):
            self.assertEqual(env.reset_calls, first_reset_calls)

        model.learn(total_timesteps=8, reset_num_timesteps=True)
        self.assertEqual(model.num_timesteps, 8)
        if hasattr(env, "reset_calls"):
            self.assertEqual(env.reset_calls, first_reset_calls + 1)


if __name__ == "__main__":
    unittest.main()
