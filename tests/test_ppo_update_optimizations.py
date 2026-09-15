"""Source-bound PPO update parity; CPU checks algebra, CUDA checks execution."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import importlib.util
import inspect
import io
import unittest
import warnings

import numpy as np


def _dependencies():
    try:
        import gymnasium as gym
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.logger import configure
    except ImportError as exc:
        raise unittest.SkipTest(f"PPO dependencies unavailable: {exc}") from exc
    return gym, torch, PPO, configure


class PPOUpdateInterfaceTests(unittest.TestCase):
    def test_opt_in_update_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("ppo_update_optimizations"))

    def test_diagnostic_summary_and_digest_helpers_are_dependency_light(self):
        module = importlib.import_module("ppo_update_optimizations")
        actions = np.asarray([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32)
        old_log_prob = np.asarray([-0.7, -0.2], dtype=np.float32)

        summary = module.diagnostic_array_summary(actions)
        digest = module.diagnostic_minibatch_digest(actions=actions, old_log_prob=old_log_prob)

        self.assertEqual(summary["shape"], [2, 2])
        self.assertEqual(summary["dtype"], "float32")
        self.assertAlmostEqual(summary["mean"], 0.1875)
        self.assertAlmostEqual(summary["min"], -1.0)
        self.assertAlmostEqual(summary["max"], 1.0)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, module.diagnostic_minibatch_digest(actions=actions, old_log_prob=old_log_prob))


class PPOUpdateParityTests(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        self.gym, self.torch, self.PPO, self.configure = _dependencies()
        self.assertIsNotNone(importlib.util.find_spec("ppo_update_optimizations"))
        self.module = importlib.import_module("ppo_update_optimizations")
        self.torch.set_num_threads(1)

    def make_model(self, *, discrete=False, batch_size=3, n_steps=6,
                   clip_range_vf=None, normalize_advantage=True,
                   target_kl=None, stochastic=False):
        gym, torch = self.gym, self.torch

        class FixtureEnv(gym.Env):
            observation_space = gym.spaces.Box(-10, 10, (3,), dtype=np.float32)
            action_space = (gym.spaces.Discrete(3) if discrete else
                            gym.spaces.Box(-1, 1, (10,), dtype=np.float32))

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(3, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(3, dtype=np.float32), 0.0, False, False, {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = self.PPO(
                "MlpPolicy", FixtureEnv(), n_steps=n_steps, batch_size=batch_size,
                n_epochs=3, learning_rate=0.003, ent_coef=0.013, vf_coef=0.6,
                clip_range=0.12, clip_range_vf=clip_range_vf,
                normalize_advantage=normalize_advantage, target_kl=target_kl,
                seed=124, device=self.device,
                policy_kwargs={"net_arch": [8],
                               "activation_fn": torch.nn.Dropout if stochastic else torch.nn.Tanh},
            )
        model.set_logger(self.configure(folder=None, format_strings=[]))
        observations = torch.arange(n_steps * 3, device=self.device, dtype=torch.float32).reshape(n_steps, 3) / 20
        with torch.no_grad():
            actions, values, log_probs = model.policy(observations)
        buffer = model.rollout_buffer
        for index in range(n_steps):
            buffer.add(
                observations[index:index + 1].cpu().numpy(),
                actions[index:index + 1].cpu().numpy(),
                np.asarray([(-1.0) ** index * (index + 1) / 3], dtype=np.float32),
                np.asarray([index == 0], dtype=np.float32),
                values[index:index + 1], log_probs[index:index + 1],
            )
        buffer.compute_returns_and_advantage(
            last_values=torch.tensor([0.2], device=self.device), dones=np.asarray([False]))
        return model

    def instrument(self, model, *, no_entropy=False, stop_after=None):
        evaluations, gradients = [], []
        evaluate = model.policy.evaluate_actions

        def evaluate_actions(observations, actions):
            evaluations.append(observations.detach().cpu().clone())
            values, log_prob, entropy = evaluate(observations, actions)
            if stop_after is not None and len(evaluations) > stop_after:
                log_prob = log_prob + 2.0
            return values, log_prob, None if no_entropy else entropy

        model.policy.evaluate_actions = evaluate_actions

        def record_gradients(optimizer, args, kwargs):
            gradients.append([None if p.grad is None else p.grad.detach().cpu().clone()
                              for p in model.policy.parameters()])

        model.policy.optimizer.register_step_pre_hook(record_gradients)
        return evaluations, gradients

    def rng_state(self):
        return (np.random.get_state(), self.torch.get_rng_state().clone(),
                self.torch.cuda.get_rng_state_all() if self.torch.cuda.is_available() else [])

    def restore_rng(self, state):
        np.random.set_state(state[0])
        self.torch.set_rng_state(state[1])
        if state[2]:
            self.torch.cuda.set_rng_state_all(state[2])

    def assert_tree_equal(self, left, right, *, exact=True):
        torch = self.torch
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0 if exact else 1e-5,
                                       atol=0 if exact else 1e-6, equal_nan=True)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key], exact=exact)
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b, exact=exact)
        elif isinstance(left, (float, np.floating)):
            np.testing.assert_allclose(left, right, rtol=0 if exact else 1e-5,
                                       atol=0 if exact else 1e-6, equal_nan=True)
        else:
            self.assertEqual(left, right)

    def compare_updates(self, reference, candidate, *, repeats=2, no_entropy=False,
                        stop_after=None, loss_function=None, exact=True):
        reference_trace = self.instrument(reference, no_entropy=no_entropy, stop_after=stop_after)
        candidate_trace = self.instrument(candidate, no_entropy=no_entropy, stop_after=stop_after)
        for _ in range(repeats):
            initial_rng = self.rng_state()
            self.PPO.train(reference)
            reference_rng = self.rng_state()
            self.restore_rng(initial_rng)
            self.module.deferred_metrics_train(candidate, loss_function=loss_function)
            self.assert_tree_equal(reference_rng, self.rng_state())
            self.assert_tree_equal(reference.policy.state_dict(), candidate.policy.state_dict(), exact=exact)
            self.assert_tree_equal(reference.policy.optimizer.state_dict(),
                                   candidate.policy.optimizer.state_dict(), exact=exact)
            self.assert_tree_equal(reference_trace, candidate_trace, exact=exact)
            self.assert_tree_equal(dict(reference.logger.name_to_value),
                                   dict(candidate.logger.name_to_value), exact=exact)
            self.assertEqual(reference._n_updates, candidate._n_updates)
        return reference_trace

    def test_reference_train_source_is_exact_sb3_250(self):
        digest = hashlib.sha256(inspect.getsource(self.PPO.train).encode()).hexdigest()
        self.assertEqual(digest, "7c8d09940123242f0ac401c5ec2cc858b8bf3339f571c222382f048ffd8306ae")

    def test_source_metadata_reports_supported_version_and_hash(self):
        self.assertTrue(hasattr(self.module, "sb3_train_source_metadata"))
        metadata = self.module.sb3_train_source_metadata()
        self.assertTrue(metadata["supported"])
        self.assertEqual(metadata["version"], "2.5.0")
        self.assertEqual(metadata["train_sha256"],
                         "7c8d09940123242f0ac401c5ec2cc858b8bf3339f571c222382f048ffd8306ae")

    def test_unsupported_sb3_version_fails_before_training(self):
        import stable_baselines3

        reference = self.make_model()
        version = stable_baselines3.__version__
        try:
            stable_baselines3.__version__ = "unsupported-test-version"
            with self.assertRaisesRegex(RuntimeError, "unsupported-test-version"):
                self.module.deferred_metrics_train(reference)
            self.assertFalse(reference.policy.optimizer.state)
        finally:
            stable_baselines3.__version__ = version

    def test_exact_update_gradients_optimizer_logs_rng_and_repeated_train(self):
        for options in ({}, {"clip_range_vf": 0.15}, {"normalize_advantage": False},
                        {"discrete": True}, {"n_steps": 7}, {"stochastic": True}):
            with self.subTest(**options):
                reference = self.make_model(**options)
                self.compare_updates(reference, copy.deepcopy(reference))

    def test_entropy_fallback_preserves_objective_and_statistics(self):
        reference = self.make_model()
        self.compare_updates(reference, copy.deepcopy(reference), no_entropy=True)

    def test_target_kl_stop_skips_current_step_and_counts_attempted_epoch(self):
        for stop_after in (0, 2):
            with self.subTest(stop_after=stop_after):
                reference = self.make_model(target_kl=0.02)
                candidate = copy.deepcopy(reference)
                reference.verbose = candidate.verbose = 1
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    evaluations, gradients = self.compare_updates(
                        reference, candidate, repeats=1, stop_after=stop_after)
                self.assertEqual(len(evaluations), stop_after + 1)
                self.assertEqual(len(gradients), stop_after)
                self.assertEqual(candidate._n_updates, stop_after // 2 + 1)
                messages = output.getvalue().splitlines()
                self.assertEqual(messages[0], messages[1])
                self.assertIn("Early stopping", messages[0])

    def test_disabled_update_diagnostics_preserve_exact_control_behavior(self):
        reference = self.make_model(target_kl=0.02)
        candidate = copy.deepcopy(reference)
        candidate.update_diagnostic_sink = None

        self.compare_updates(reference, candidate, repeats=1, stop_after=1)

    def test_update_diagnostics_records_attempted_minibatch_and_kl_stop(self):
        candidate = self.make_model(target_kl=0.02)
        rows = []
        candidate.update_diagnostic_sink = rows.append
        self.instrument(candidate, stop_after=0)

        with contextlib.redirect_stdout(io.StringIO()):
            self.module.deferred_metrics_train(candidate)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["epoch"], 0)
        self.assertEqual(row["minibatch_ordinal"], 0)
        self.assertEqual(row["minibatch_in_epoch"], 0)
        self.assertEqual(row["target_kl"], 0.02)
        self.assertEqual(row["target_kl_threshold"], 0.03)
        self.assertGreater(row["pre_step_approx_kl"], row["target_kl_threshold"])
        self.assertFalse(row["optimizer_step_executed"])
        self.assertTrue(row["early_stop"])
        self.assertIn("policy_loss", row)
        self.assertIn("value_loss", row)
        self.assertIn("entropy_loss", row)
        self.assertIn("entropy", row)
        self.assertIn("clip_fraction", row)
        self.assertIsNone(row["gradient_norm"])
        self.assertEqual(row["learning_rate"], 0.003)
        self.assertEqual(row["sample_count"], 3)
        self.assertRegex(row["sample_digest"], r"^[0-9a-f]{64}$")
        self.assertEqual(row["actions"]["shape"], [3, 10])
        self.assertEqual(row["evaluate_actions_actions"]["shape"], [3, 10])
        self.assertIn("mean", row["old_values"])
        self.assertIn("mean", row["log_prob"])
        self.assertIn("mean", row["old_log_prob"])

    def test_update_diagnostic_digest_includes_old_values_and_evaluated_actions(self):
        module = self.module
        torch = self.torch
        candidate = self.make_model(target_kl=0.02)
        self.instrument(candidate, stop_after=0)
        rollout_data = next(candidate.rollout_buffer.get(candidate.batch_size))
        actions = rollout_data.actions
        values, log_prob, entropy = candidate.policy.evaluate_actions(rollout_data.observations, actions)
        loss, policy_loss, value_loss, entropy_loss, clip_fraction, approx_kl_div = module.ppo_loss(
            values, log_prob, entropy, rollout_data.advantages, rollout_data.old_log_prob,
            rollout_data.old_values, rollout_data.returns, 0.12, None, candidate.ent_coef,
            candidate.vf_coef, candidate.normalize_advantage,
        )
        rows = []

        module._emit_update_diagnostic(
            rows.append,
            epoch=0,
            minibatch_in_epoch=0,
            minibatch_ordinal=0,
            rollout_data=rollout_data,
            evaluate_actions_actions=actions,
            log_prob=log_prob,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            clip_fraction=clip_fraction,
            approx_kl_div=approx_kl_div,
            target_kl=0.02,
            optimizer_step_executed=False,
            early_stop=True,
            gradient_norm=None,
            learning_rate=0.003,
            approx_kl_host=approx_kl_div.detach().cpu().numpy(),
        )

        row = rows[0]
        self.assertEqual(
            row["sample_digest"],
            module.diagnostic_minibatch_digest(
                observations=rollout_data.observations,
                actions=rollout_data.actions,
                evaluate_actions_actions=actions,
                old_log_prob=rollout_data.old_log_prob,
                old_values=rollout_data.old_values,
                advantages=rollout_data.advantages,
                returns=rollout_data.returns,
            ),
        )
        self.assertNotEqual(
            row["sample_digest"],
            module.diagnostic_minibatch_digest(
                observations=rollout_data.observations,
                actions=rollout_data.actions,
                evaluate_actions_actions=actions + torch.full_like(actions, 0.01),
                old_log_prob=rollout_data.old_log_prob,
                old_values=rollout_data.old_values,
                advantages=rollout_data.advantages,
                returns=rollout_data.returns,
            ),
        )
        self.assertNotEqual(
            row["sample_digest"],
            module.diagnostic_minibatch_digest(
                observations=rollout_data.observations,
                actions=rollout_data.actions,
                evaluate_actions_actions=actions,
                old_log_prob=rollout_data.old_log_prob,
                old_values=rollout_data.old_values + torch.full_like(rollout_data.old_values, 0.01),
                advantages=rollout_data.advantages,
                returns=rollout_data.returns,
            ),
        )

    def test_update_diagnostics_records_executed_step_gradient_norm(self):
        candidate = self.make_model(target_kl=0.02)
        rows = []
        candidate.update_diagnostic_sink = rows.append
        self.instrument(candidate, stop_after=1)

        with contextlib.redirect_stdout(io.StringIO()):
            self.module.deferred_metrics_train(candidate)

        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["optimizer_step_executed"])
        self.assertFalse(rows[0]["early_stop"])
        self.assertGreaterEqual(rows[0]["gradient_norm"], 0.0)
        self.assertFalse(rows[1]["optimizer_step_executed"])
        self.assertTrue(rows[1]["early_stop"])

    def test_invalid_distribution_still_raises_before_optimizer_update(self):
        reference = self.make_model()
        reference.rollout_buffer.observations[:] = np.nan
        candidate = copy.deepcopy(reference)
        original_policy = copy.deepcopy(candidate.policy.state_dict())
        for model, train in ((reference, self.PPO.train),
                             (candidate, self.module.deferred_metrics_train)):
            with self.assertRaises(ValueError):
                train(model)
            self.assertEqual(model._n_updates, 0)
            self.assertFalse(model.policy.optimizer.state)
            self.assert_tree_equal(original_policy, model.policy.state_dict())

    def test_readbacks_drop_by_four_metrics_per_minibatch(self):
        torch = self.torch
        reference = self.make_model()
        candidate = copy.deepcopy(reference)
        counts = []
        for model, train in ((reference, self.PPO.train),
                             (candidate, self.module.deferred_metrics_train)):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                train(model)
            counts.append(sum(event.count for event in profile.key_averages()
                              if event.key == "aten::item"))
        self.assertGreaterEqual(counts[0] - counts[1], 4 * 2 * reference.n_epochs)

    def test_compiled_fixed_shape_math_matches_eager_updates_and_rng(self):
        reference = self.make_model(stochastic=True)
        compiled_loss = self.module.compile_ppo_loss(backend="aot_eager")
        self.compare_updates(reference, copy.deepcopy(reference), loss_function=compiled_loss)

    def test_compile_preserves_distribution_validation_default(self):
        distribution = self.torch.distributions.Distribution
        original = distribution._validate_args
        try:
            for requested in (True, False):
                distribution.set_default_validate_args(requested)
                compiled = self.module.compile_ppo_loss(backend="aot_eager")
                self.assertIs(distribution._validate_args, requested)
                reference = self.make_model()
                self.module.deferred_metrics_train(reference, loss_function=compiled)
                self.assertIs(distribution._validate_args, requested)
        finally:
            distribution.set_default_validate_args(original)

    def test_compile_failure_propagates(self):
        def failing_backend(graph, example_inputs):
            raise RuntimeError("intentional compile backend failure")

        reference = self.make_model()
        compiled_loss = self.module.compile_ppo_loss(backend=failing_backend)
        with self.assertRaisesRegex(Exception, "intentional compile backend failure"):
            self.module.deferred_metrics_train(reference, loss_function=compiled_loss)


class PPOUpdateCudaParityTests(PPOUpdateParityTests):
    device = "cuda"

    def setUp(self):
        _, torch, _, _ = _dependencies()
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable; CPU parity does not establish CUDA performance")
        super().setUp()

    def test_native_tensor_buffer_explained_variance_and_updates_match_proxy(self):
        from tensor_ppo import _TensorRolloutStatsProxy
        from tensor_rollout_buffer import TensorRolloutBuffer

        reference = self.make_model()
        candidate = copy.deepcopy(reference)
        for model in (reference, candidate):
            old_buffer = model.rollout_buffer
            buffer = TensorRolloutBuffer(
                model.n_steps, model.observation_space, model.action_space,
                self.device, model.gae_lambda, model.gamma)
            for name in ("observations", "actions", "rewards", "episode_starts", "values",
                         "log_probs", "advantages", "returns"):
                getattr(buffer, name).copy_(self.torch.as_tensor(getattr(old_buffer, name), device=self.device))
            buffer.full = True
            buffer.pos = model.n_steps
            buffer._computed_returns_and_advantage = True
            model.rollout_buffer = buffer
        reference.rollout_buffer = _TensorRolloutStatsProxy(reference.rollout_buffer)
        self.compare_updates(reference, candidate)

    def test_inductor_update_parity_with_explicit_fusion_tolerance(self):
        reference = self.make_model(stochastic=True)
        self.compare_updates(reference, copy.deepcopy(reference), repeats=2,
                             loss_function=self.module.compile_ppo_loss(), exact=False)


if __name__ == "__main__":
    unittest.main()
