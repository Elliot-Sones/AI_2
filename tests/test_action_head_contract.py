from __future__ import annotations

import math
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_POLICY_DEPS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("gymnasium") is not None

if HAS_POLICY_DEPS:
    import torch as th
    import gymnasium as gym
    from gymnasium import spaces


@unittest.skipUnless(HAS_POLICY_DEPS, "torch/gymnasium are not installed in this local contract environment")
class ActionHeadContractTests(unittest.TestCase):
    def setUp(self):
        import curriculum_ppo as module

        self.module = module
        self.obs_space = spaces.Box(-10, 10, shape=(256,), dtype=np.float32)
        self.action_space = spaces.Box(0, 1, shape=(10,), dtype=np.float32)

    def policy(self, action_head="gaussian"):
        return self.module.CurriculumPolicy(
            self.obs_space,
            self.action_space,
            lr_schedule=lambda _: 1e-4,
            net_arch={"pi": [8], "vf": [8]},
            action_head=action_head,
        )

    def test_gaussian_action_head_preserves_raw_policy_actions_and_decode(self):
        policy = self.policy("gaussian")
        raw = th.tensor(
            [[-0.25, 0.0, 0.49, 0.5, 0.75, 1.0, 1.25, 0.8, 0.9, 1.0]],
            dtype=th.float32,
        )
        decoded = policy.decode_actions(raw)
        expected = raw.clamp(0, 1) * th.tensor(self.module.TASK_MASKS["combat"], dtype=th.float32)
        th.testing.assert_close(decoded, expected, rtol=0, atol=0)
        self.assertEqual(policy.action_contract()["policy_action_domain"], "gaussian_raw_float32_box_10")
        self.assertEqual(policy.action_contract()["executed_action_domain"], "float32_box_10_clipped_masked")

    def test_bernoulli_action_head_decodes_exact_buttons_and_masks_probability_math(self):
        policy = self.policy("bernoulli_10")
        policy.set_task("combat", calibration=False)
        with th.no_grad():
            policy.action_net.weight.zero_()
            policy.action_net.bias.copy_(th.tensor(
                [-20.0, 20.0, 0.0, 0.25, -0.25, 20.0, 2.0, 0.0, 20.0, 20.0],
                dtype=th.float32,
            ))

        obs = th.zeros((1, 256), dtype=th.float32)
        actions, _values, log_prob = policy(obs, deterministic=True)
        expected = th.tensor([[0, 1, 0, 1, 0, 0, 1, 0, 0, 0]], dtype=th.float32)
        th.testing.assert_close(actions, expected, rtol=0, atol=0)
        th.testing.assert_close(policy.decode_actions(actions), expected, rtol=0, atol=0)

        _eval_values, eval_log_prob, entropy = policy.evaluate_actions(obs, actions)
        th.testing.assert_close(log_prob, eval_log_prob, rtol=0, atol=0)
        mask = th.tensor(self.module.TASK_MASKS["combat"], dtype=th.bool)
        logits = policy.action_net(policy.mlp_extractor.policy_net(obs))
        expected_log_prob = th.distributions.Bernoulli(logits=logits).log_prob(actions)[..., mask].sum(-1)
        expected_entropy = th.distributions.Bernoulli(logits=logits).entropy()[..., mask].sum(-1)
        th.testing.assert_close(log_prob, expected_log_prob, rtol=0, atol=0)
        th.testing.assert_close(entropy, expected_entropy, rtol=0, atol=0)

    def test_bernoulli_distribution_rejects_nonfinite_logits_at_source(self):
        policy = self.policy("bernoulli_10")
        for value in (float("nan"), float("inf"), float("-inf")):
            logits = th.zeros((2, 10), dtype=th.float32)
            logits[0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                policy.action_dist.proba_distribution(logits)

    def test_bernoulli_inactive_keys_have_zero_gradient_and_are_rejected_if_nonbinary(self):
        policy = self.policy("bernoulli_10")
        policy.set_task("combat", calibration=False)
        obs = th.zeros((2, 256), dtype=th.float32)
        actions, _values, log_prob = policy(obs, deterministic=False)
        self.assertTrue(th.equal(actions[..., [5, 8, 9]], th.zeros_like(actions[..., [5, 8, 9]])))
        (-log_prob.mean()).backward()
        self.assertTrue(th.equal(policy.action_net.weight.grad[[5, 8, 9]], th.zeros_like(policy.action_net.weight.grad[[5, 8, 9]])))
        bad = actions.clone()
        bad[0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "Bernoulli"):
            policy.evaluate_actions(obs, bad)
        inactive_bad = actions.clone()
        inactive_bad[0, 5] = 1.0
        with self.assertRaisesRegex(ValueError, "inactive"):
            policy.evaluate_actions(obs, inactive_bad)

    def test_bernoulli_export_schema_round_trips_and_rejects_head_mismatch(self):
        model = self.module.CurriculumPPO(
            env=self.fixture_env(),
            action_head="bernoulli_10",
            _init_setup_model=True,
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "actor.pt"
            metadata = model.export_actor(path)
            self.assertEqual(metadata["action_head"], "bernoulli_10")
            actor, loaded = self.module.load_actor(path)
            self.assertEqual(loaded["action_head"], "bernoulli_10")
            self.assertEqual(actor.action_head, "bernoulli_10")

            payload = th.load(path, map_location="cpu", weights_only=True)
            payload["metadata"]["action_head"] = "gaussian"
            bad = Path(temporary) / "bad.pt"
            th.save(payload, bad)
            with self.assertRaisesRegex(ValueError, "action head"):
                self.module.load_actor(bad, expected_action_head="bernoulli_10")

    def test_export_actor_rejects_nonfinite_parameters_for_gaussian_and_bernoulli(self):
        for action_head in ("gaussian", "bernoulli_10"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(action_head=action_head, value=value):
                    model = self.module.CurriculumPPO(
                        env=self.fixture_env(),
                        action_head=action_head,
                        _init_setup_model=True,
                        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
                    )
                    with th.no_grad():
                        model.policy.action_net.bias[0] = value
                    with tempfile.TemporaryDirectory() as temporary:
                        with self.assertRaisesRegex(ValueError, "finite"):
                            model.export_actor(Path(temporary) / "actor.pt")

    def test_actor_predict_rejects_nonfinite_inputs_and_outputs_for_both_heads(self):
        for action_head in ("gaussian", "bernoulli_10"):
            actor = self.module.CurriculumActor({"pi": [8]}, action_head=action_head)
            for value in (float("nan"), float("inf"), float("-inf")):
                bad_observation = np.zeros((1, 256), dtype=np.float32)
                bad_observation[0, 0] = value
                with self.subTest(action_head=action_head, field="input", value=value):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        actor.predict(bad_observation, deterministic=True)

                with self.subTest(action_head=action_head, field="head", value=value):
                    with th.no_grad():
                        actor.action_net.bias.fill_(0)
                        actor.log_std.zero_()
                        actor.action_net.bias[0] = value
                    with self.assertRaisesRegex(ValueError, "finite"):
                        actor.predict(np.zeros((1, 256), dtype=np.float32), deterministic=True)

        gaussian = self.module.CurriculumActor({"pi": [8]}, action_head="gaussian")
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(action_head="gaussian", field="std", value=value):
                with th.no_grad():
                    gaussian.action_net.bias.zero_()
                    gaussian.log_std.zero_()
                    gaussian.log_std[0] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    gaussian.predict(np.zeros((1, 256), dtype=np.float32), deterministic=False)

    def test_full_checkpoint_preserves_bernoulli_action_head_identity(self):
        model = self.module.CurriculumPPO(
            env=self.fixture_env(),
            action_head="bernoulli_10",
            _init_setup_model=True,
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.zip"
            model.save(path)
            restored = self.module.CurriculumPPO.load(path, env=self.fixture_env())
            self.assertEqual(restored.action_head, "bernoulli_10")
            self.assertEqual(restored.policy.action_head, "bernoulli_10")

    def test_bernoulli_from_parent_copies_trunk_but_initializes_fresh_neutral_head(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import ActorCriticPolicy

        parent = PPO(
            ActorCriticPolicy,
            self.fixture_env(),
            n_steps=8,
            batch_size=8,
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
            seed=3,
        )
        with th.no_grad():
            parent.policy.action_net.bias.fill_(3.0)
        child = self.module.CurriculumPPO.from_parent(
            parent,
            self.fixture_env(),
            action_head="bernoulli_10",
            calibration=False,
            n_steps=8,
            batch_size=8,
        )
        self.assertEqual(child.policy.action_head, "bernoulli_10")
        self.assertEqual(
            child.optimizer_provenance,
            "fresh neutral bernoulli_10 action head; parent trunk/value copied; fresh Adam optimizer not transferred",
        )
        th.testing.assert_close(
            child.policy.mlp_extractor.policy_net[0].weight,
            parent.policy.mlp_extractor.policy_net[0].weight,
            rtol=0,
            atol=0,
        )
        th.testing.assert_close(child.policy.action_net.weight, th.zeros_like(child.policy.action_net.weight), rtol=0, atol=0)
        th.testing.assert_close(child.policy.action_net.bias, th.zeros_like(child.policy.action_net.bias), rtol=0, atol=0)

    def test_action_ab_arms_are_explicit_and_do_not_silently_neutralize(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import ActorCriticPolicy

        parent = PPO(
            ActorCriticPolicy,
            self.fixture_env(),
            n_steps=8,
            batch_size=8,
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
            seed=31,
        )
        with th.no_grad():
            parent.policy.action_net.weight.fill_(0.25)
            parent.policy.action_net.bias.fill_(0.75)
            parent.policy.log_std.fill_(math.log(0.3))
        kept = self.module.CurriculumPPO.from_parent(
            parent, self.fixture_env(), action_ab_arm="G_keep", n_steps=8, batch_size=8,
        )
        neutral = self.module.CurriculumPPO.from_parent(
            parent, self.fixture_env(), action_ab_arm="G_neutral", n_steps=8, batch_size=8,
        )
        projected = self.module.CurriculumPPO.from_parent(
            parent, self.fixture_env(), action_ab_arm="B_projected", n_steps=8, batch_size=8,
        )

        self.assertEqual(kept.action_ab_arm, "G_keep")
        self.assertEqual(neutral.action_ab_arm, "G_neutral")
        self.assertEqual(projected.action_ab_arm, "B_projected")
        th.testing.assert_close(kept.policy.action_net.bias, parent.policy.action_net.bias, rtol=0, atol=0)
        th.testing.assert_close(neutral.policy.action_net.bias, th.zeros_like(neutral.policy.action_net.bias), rtol=0, atol=0)
        self.assertFalse(th.equal(projected.policy.action_net.bias, th.zeros_like(projected.policy.action_net.bias)))
        self.assertIn("projected", projected.optimizer_provenance)

    def test_projected_bernoulli_matches_gaussian_deterministic_buttons_and_sample_probabilities(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import ActorCriticPolicy

        parent = PPO(
            ActorCriticPolicy,
            self.fixture_env(),
            n_steps=8,
            batch_size=8,
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
            seed=37,
        )
        with th.no_grad():
            parent.policy.action_net.weight.zero_()
            parent.policy.action_net.bias.copy_(th.tensor(
                [0.1, 0.49, 0.5, 0.5001, 0.8, 0.7, 0.2, 1.0, 0.6, 0.4],
                dtype=th.float32,
            ))
            parent.policy.log_std.fill_(math.log(0.35))
        child = self.module.CurriculumPPO.from_parent(
            parent, self.fixture_env(), action_ab_arm="B_projected", n_steps=8, batch_size=8,
        )
        observations = th.zeros((5, 256), dtype=th.float32)
        with th.no_grad():
            parent_means = parent.policy.action_net(parent.policy.mlp_extractor.policy_net(observations))
            child_logits = child.policy.action_net(child.policy.mlp_extractor.policy_net(observations))
        parent_buttons = parent_means > self.module.GAUSSIAN_BUTTON_THRESHOLD
        child_buttons = child_logits > 0
        th.testing.assert_close(child_buttons.to(th.float32), parent_buttons.to(th.float32), rtol=0, atol=0)
        self.assertFalse(bool(child_buttons[0, 2]), "exact threshold boundary must remain unpressed")

        gaussian_probs = th.distributions.Normal(parent_means, parent.policy.log_std.exp()).cdf(
            parent_means.new_full(parent_means.shape, self.module.GAUSSIAN_BUTTON_THRESHOLD)
        )
        gaussian_press_probs = 1 - gaussian_probs
        bernoulli_press_probs = th.sigmoid(child_logits)
        max_error = (gaussian_press_probs - bernoulli_press_probs).abs().max().item()
        self.assertLessEqual(max_error, self.module.BERNOULLI_PROJECTION_MAX_PROB_ERROR)

        child.policy.set_task("combat", calibration=False)
        actions, _values, log_prob = child.policy(observations, deterministic=False)
        loss = -log_prob.mean()
        loss.backward()
        self.assertTrue(th.isfinite(log_prob).all())
        _eval_values, eval_log_prob, entropy = child.policy.evaluate_actions(observations, actions)
        self.assertTrue(th.isfinite(eval_log_prob).all())
        self.assertTrue(th.isfinite(entropy).all())
        self.assertTrue(all(parameter.grad is None or th.isfinite(parameter.grad).all()
                            for parameter in child.policy.parameters()))

    def test_action_arm_metadata_round_trips_and_rejects_mismatch(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import ActorCriticPolicy

        parent = PPO(
            ActorCriticPolicy,
            self.fixture_env(),
            n_steps=8,
            batch_size=8,
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
            seed=41,
        )
        child = self.module.CurriculumPPO.from_parent(
            parent, self.fixture_env(), action_ab_arm="B_projected", n_steps=8, batch_size=8,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "actor.pt"
            metadata = child.export_actor(path)
            self.assertEqual(metadata["action_ab_arm"], "B_projected")
            _actor, loaded = self.module.load_actor(path, expected_action_head="bernoulli_10",
                                                    expected_action_ab_arm="B_projected")
            self.assertEqual(loaded["action_init_metadata"]["strategy"], "projected")
            with self.assertRaisesRegex(ValueError, "action arm"):
                self.module.load_actor(path, expected_action_ab_arm="B_neutral")

    def test_from_parent_preserves_caller_rng_state(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import ActorCriticPolicy

        parent = PPO(
            ActorCriticPolicy,
            self.fixture_env(),
            n_steps=8,
            batch_size=8,
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
            seed=43,
        )
        random_state = np.random.get_state()
        torch_state = th.get_rng_state()
        _child = self.module.CurriculumPPO.from_parent(
            parent, self.fixture_env(), action_ab_arm="B_projected", n_steps=8, batch_size=8,
        )
        after_random = np.random.get_state()
        self.assertEqual(random_state[0], after_random[0])
        np.testing.assert_array_equal(random_state[1], after_random[1])
        self.assertEqual(random_state[2:], after_random[2:])
        th.testing.assert_close(th.get_rng_state(), torch_state, rtol=0, atol=0)

    def test_bernoulli_normalized_entropy_has_unit_scale_and_gaussian_is_unbounded(self):
        bernoulli = self.policy("bernoulli_10")
        entropy = th.full((3,), len(bernoulli.active_indices) * math.log(2), dtype=th.float32)
        th.testing.assert_close(bernoulli.normalized_entropy(entropy), th.ones(3), rtol=0, atol=1e-6)
        gaussian = self.policy("gaussian")
        self.assertIsNone(gaussian.normalized_entropy(th.ones(1)))

    def test_invalid_action_head_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "action_head"):
            self.policy("multidiscrete")

    def fixture_env(self):
        class FixtureEnv(gym.Env):
            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(256, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(256, dtype=np.float32), 0.0, False, False, {}

        env = FixtureEnv()
        env.observation_space = self.obs_space
        env.action_space = self.action_space
        return env


if __name__ == "__main__":
    unittest.main()
