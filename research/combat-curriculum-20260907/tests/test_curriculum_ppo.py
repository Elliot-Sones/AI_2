"""Synthetic CPU contract tests; never load or evaluate the experiment parent."""

import copy
import importlib
import importlib.util
import math
import tempfile
import unittest
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


NAVIGATION_MASK = np.array([1, 1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
COMBAT_MASK = NAVIGATION_MASK.copy()
COMBAT_MASK[7] = True


class SyntheticTask(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, shape=(256,), dtype=np.float32)
    action_space = gym.spaces.Box(0, 1, shape=(10,), dtype=np.float32)

    def __init__(self, task="combat", wrong_labels=False):
        self.task = task
        self.wrong_labels = wrong_labels
        self.tick = 0
        self.received = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tick = 0
        return np.linspace(-1, 1, 256, dtype=np.float32), {}

    def step(self, action):
        self.received.append(action.copy())
        self.tick += 1
        obs = np.linspace(-1, 1, 256, dtype=np.float32) + self.tick / 100
        return obs, float(0.2 + action[7] + action[0]), False, False, {
            "curriculum_task": "wrong" if self.wrong_labels else self.task,
            "allowed_control_mask": (
                COMBAT_MASK if self.task == "combat" else NAVIGATION_MASK
            ).copy(),
        }


class CurriculumPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        th.set_num_threads(1)

    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("curriculum_ppo"),
            "The curriculum PPO extension has not been implemented",
        )
        self.module = importlib.import_module("curriculum_ppo")

    def parent(self):
        return PPO(
            "MlpPolicy", SyntheticTask(), device="cpu", seed=123,
            n_steps=8, batch_size=8, n_epochs=1,
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
        )

    def model(self, *, task="combat", calibration=True, arm="E1", parent=None):
        return self.module.CurriculumPPO.from_parent(
            parent or self.parent(), SyntheticTask(task), task=task,
            calibration=calibration, arm=arm, n_steps=8, batch_size=8,
            n_epochs=2, device="cpu", seed=321,
        )

    def assert_state_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(th.equal(left[key], right[key]), key)

    def test_transfer_changes_only_light_actor_row_and_preserves_navigation(self):
        parent = self.parent()
        original = copy.deepcopy(parent.policy.state_dict())
        parent.num_timesteps = 100
        model = self.model(parent=parent)
        policy = model.policy
        self.assert_state_equal(parent.policy.state_dict(), original)
        self.assert_state_equal(
            policy.mlp_extractor.policy_net.state_dict(),
            parent.policy.mlp_extractor.policy_net.state_dict(),
        )
        self.assert_state_equal(
            policy.mlp_extractor.value_net.state_dict(),
            parent.policy.mlp_extractor.value_net.state_dict(),
        )
        self.assert_state_equal(policy.value_net.state_dict(), parent.policy.value_net.state_dict())
        rows = [0, 1, 2, 3, 4, 5, 6, 8, 9]
        self.assertTrue(th.equal(policy.action_net.weight[rows], parent.policy.action_net.weight[rows]))
        self.assertTrue(th.equal(policy.action_net.bias[rows], parent.policy.action_net.bias[rows]))
        self.assertTrue(th.equal(policy.log_std[rows], parent.policy.log_std[rows]))
        self.assertTrue(th.equal(policy.action_net.weight[7], th.zeros(8)))
        self.assertEqual(policy.action_net.bias[7].item(), 0)
        self.assertAlmostEqual(policy.log_std[7].item(), math.log(0.3), places=6)
        self.assertEqual(model.num_timesteps, 100)
        self.assertEqual(model.parent_num_timesteps, 100)
        self.assertEqual(len(policy.optimizer.state), 0)
        self.assertEqual(policy.active_indices, (0, 1, 2, 3, 4, 6, 7))
        self.assertTrue(policy.calibration)
        obs = th.randn(3, 256)
        policy.set_task("navigation")
        self.assertTrue(th.equal(policy.predict_values(obs), parent.policy.predict_values(obs)))
        policy.set_task("combat", calibration=True)
        self.assertFalse(th.equal(policy.predict_values(obs), parent.policy.predict_values(obs)))
        self.assertFalse(set(map(id, policy.combat_value_net.parameters())) & set(map(id, policy.value_net.parameters())))

    def test_active_gaussian_matches_reference_and_ignores_disabled_dimensions(self):
        mean = th.randn(4, 10, requires_grad=True)
        log_std = th.linspace(-1, 1, 10, requires_grad=True)
        raw_actions = th.linspace(-3, 3, 40).reshape(4, 10)
        dist = self.module.MaskedDiagGaussianDistribution(10, COMBAT_MASK)
        dist.proba_distribution(mean, log_std)
        normal = th.distributions.Normal(mean, log_std.exp())
        expected_log = normal.log_prob(raw_actions)[:, COMBAT_MASK].sum(-1)
        expected_entropy = normal.entropy()[:, COMBAT_MASK].sum(-1)
        th.testing.assert_close(dist.log_prob(raw_actions), expected_log)
        th.testing.assert_close(dist.entropy(), expected_entropy)
        old_log = dist.log_prob(raw_actions).detach()
        (dist.log_prob(raw_actions).mean() + dist.entropy().mean()).backward()
        self.assertTrue(th.equal(mean.grad[:, ~COMBAT_MASK], th.zeros(4, 3)))
        self.assertTrue(th.equal(log_std.grad[~COMBAT_MASK], th.zeros(3)))
        changed_mean, changed_std = mean.detach().clone(), log_std.detach().clone()
        changed_mean[:, ~COMBAT_MASK] += 1000
        changed_std[~COMBAT_MASK] += 4
        dist.proba_distribution(changed_mean, changed_std)
        delta = dist.log_prob(raw_actions) - old_log
        self.assertTrue(th.equal(delta.exp(), th.ones(4)))
        self.assertEqual(((delta.exp() - 1) - delta).mean().item(), 0)
        th.testing.assert_close(dist.entropy(), expected_entropy)
        self.assertFalse(th.equal(dist.log_prob(raw_actions.clamp(0, 1)), old_log))

    def test_calibration_freezes_movement_and_navigation_critic(self):
        model = self.model()
        before = copy.deepcopy(model.policy.state_dict())
        model.learn(8, reset_num_timesteps=False)
        after = model.policy.state_dict()
        for key in before:
            if key.startswith(("mlp_extractor.", "value_net.")):
                self.assertTrue(th.equal(before[key], after[key]), key)
        rows = [0, 1, 2, 3, 4, 5, 6, 8, 9]
        for key in ("action_net.weight", "action_net.bias", "log_std"):
            self.assertTrue(th.equal(before[key][rows], after[key][rows]), key)
        self.assertFalse(th.equal(before["log_std"][7], after["log_std"][7]))
        self.assertTrue(any(not th.equal(before[k], after[k]) for k in before if k.startswith("combat_")))
        self.assertEqual(model.num_timesteps, model.parent_num_timesteps + 8)

    def test_navigation_keeps_combat_critic_and_light_rows_fixed_despite_momentum(self):
        model = self.model(calibration=False)
        model.learn(8, reset_num_timesteps=False)
        model.set_env(SyntheticTask("navigation"))
        model.set_task("navigation")
        before = copy.deepcopy(model.policy.state_dict())
        model.learn(8, reset_num_timesteps=False)
        after = model.policy.state_dict()
        for key in before:
            if key.startswith("combat_"):
                self.assertTrue(th.equal(before[key], after[key]), key)
        inactive = [5, 7, 8, 9]
        for key in ("action_net.weight", "action_net.bias", "log_std"):
            self.assertTrue(th.equal(before[key][inactive], after[key][inactive]), key)
        self.assertTrue(any(not th.equal(before[k], after[k]) for k in before if k.startswith("mlp_extractor.policy_net")))

    def test_collection_retains_raw_samples_execution_masks_and_task_labels(self):
        model = self.model()
        _, callback = model._setup_learn(8, reset_num_timesteps=False)
        self.assertTrue(model.collect_rollouts(model.env, callback, model.rollout_buffer, 8))
        buffer = model.rollout_buffer
        self.assertTrue(np.any((buffer.actions < 0) | (buffer.actions > 1)))
        expected = np.clip(buffer.actions, 0, 1) * COMBAT_MASK
        np.testing.assert_array_equal(buffer.executed_actions, expected)
        np.testing.assert_array_equal(buffer.allowed_control_masks, np.broadcast_to(COMBAT_MASK, (8, 1, 10)))
        self.assertTrue(np.all(buffer.tasks == "combat"))
        for index, action in enumerate(model.env.envs[0].unwrapped.received):
            np.testing.assert_array_equal(action, expected[index, 0])
        with th.no_grad():
            _, log_probs, _ = model.policy.evaluate_actions(
                th.as_tensor(buffer.observations[:, 0]), th.as_tensor(buffer.actions[:, 0])
            )
        np.testing.assert_allclose(buffer.log_probs[:, 0], log_probs.numpy(), atol=2e-6)
        with self.assertRaisesRegex(ValueError, "rollout"):
            model.set_task("navigation")
        buffer.tasks[0, 0] = "navigation"
        with self.assertRaisesRegex(ValueError, "task|rollout"):
            model.train()

    def test_wrong_environment_label_and_modified_mask_fail_closed(self):
        model = self.model()
        model.set_env(SyntheticTask(wrong_labels=True))
        with self.assertRaisesRegex(ValueError, "task"):
            model.learn(8, reset_num_timesteps=False)
        model = self.model()
        _, callback = model._setup_learn(8, reset_num_timesteps=False)
        model.collect_rollouts(model.env, callback, model.rollout_buffer, 8)
        model.rollout_buffer.allowed_control_masks[0, 0, 5] = True
        with self.assertRaisesRegex(ValueError, "mask"):
            model.train()

    def test_predict_applies_mask_and_save_reload_preserves_both_critics_and_optimizer(self):
        model = self.model(calibration=False, arm="E2")
        model.learn(8, reset_num_timesteps=False)
        model.set_env(SyntheticTask("navigation"))
        model.set_task("navigation")
        model.learn(8, reset_num_timesteps=False)
        obs = np.linspace(-1, 1, 256, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.zip"
            model.save(path)
            with warnings.catch_warnings(record=True) as emitted:
                restored = self.module.CurriculumPPO.load(path, device="cpu", env=SyntheticTask("navigation"))
            self.assertEqual(emitted, [], "Load should validate restored geometry, not temporary defaults")
            self.assertEqual(restored.task, "navigation")
            self.assertFalse(restored.calibration)
            self.assertEqual(restored.arm, "E2")
            self.assertEqual(restored.num_timesteps, model.num_timesteps)
            self.assertEqual(restored.parent_num_timesteps, model.parent_num_timesteps)
            self.assert_state_equal(model.policy.state_dict(), restored.policy.state_dict())
            left, right = model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict()
            self.assertEqual(left["param_groups"], right["param_groups"])
            self.assertEqual(left["state"].keys(), right["state"].keys())
            for key in left["state"]:
                self.assert_state_equal(left["state"][key], right["state"][key])
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])
            self.assertTrue(np.all(restored.predict(obs, deterministic=False)[0][~NAVIGATION_MASK] == 0))
            before = restored.num_timesteps
            model.set_env(SyntheticTask("navigation"), force_reset=True)
            th.manual_seed(999)
            np.random.seed(999)
            model.learn(8, reset_num_timesteps=False)
            th.manual_seed(999)
            np.random.seed(999)
            restored.learn(8, reset_num_timesteps=False)
            self.assertEqual(restored.num_timesteps, before + 8)
            self.assert_state_equal(model.policy.state_dict(), restored.policy.state_dict())

    def test_rollout_cannot_change_task_in_callback_or_be_trained_twice(self):
        class ChangeTask(BaseCallback):
            def _on_step(self):
                self.model.set_task("navigation")
                return True

        model = self.model()
        with self.assertRaisesRegex(ValueError, "rollout"):
            model.learn(8, callback=ChangeTask(), reset_num_timesteps=False)
        model = self.model()
        model.learn(8, reset_num_timesteps=False)
        with self.assertRaisesRegex(ValueError, "unconsumed"):
            model.train()

    def test_task_buffer_rejects_mixing_and_noncanonical_mask_even_if_policy_is_tampered(self):
        model = self.model()
        buffer = model.rollout_buffer
        args = (
            np.zeros((1, 256), dtype=np.float32), np.zeros((1, 10), dtype=np.float32),
            np.zeros(1), np.ones(1), th.zeros(1, 1), th.zeros(1),
        )
        buffer.add(*args, task="combat", mask=COMBAT_MASK, executed_actions=np.zeros((1, 10)))
        with self.assertRaisesRegex(ValueError, "mix"):
            buffer.add(*args, task="navigation", mask=NAVIGATION_MASK, executed_actions=np.zeros((1, 10)))
        buffer.reset()
        _, callback = model._setup_learn(8, reset_num_timesteps=False)
        model.collect_rollouts(model.env, callback, buffer, 8)
        model.policy.allowed_control_mask[5] = True
        buffer.allowed_control_masks[:, :, 5] = True
        with self.assertRaisesRegex(ValueError, "mask"):
            model.train()

    def test_rollout_rejects_nonfinite_actions_before_environment_step(self):
        model = self.model()
        with th.no_grad():
            model.policy.action_net.bias[0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            model.learn(8, reset_num_timesteps=False)
        self.assertEqual(model.env.envs[0].unwrapped.received, [])

    def test_nonfinite_rewards_are_rejected_before_buffer_add(self):
        class InvalidReward(SyntheticTask):
            def step(self, action):
                obs, _, terminated, truncated, info = super().step(action)
                return obs, float("nan"), terminated, truncated, info

        model = self.model()
        model.set_env(InvalidReward())
        with self.assertRaisesRegex(ValueError, "finite"):
            model.learn(8, reset_num_timesteps=False)
        self.assertEqual(model.rollout_buffer.pos, 0)

    def test_nonfinite_returns_are_rejected_before_training(self):
        model = self.model()
        _, callback = model._setup_learn(8, reset_num_timesteps=False)
        model.collect_rollouts(model.env, callback, model.rollout_buffer, 8)
        model.rollout_buffer.returns[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            model.train()
        self.assertEqual(len(model.policy.optimizer.state), 0)

    def test_nonfinite_gradients_are_rejected_before_optimizer_step(self):
        model = self.model()
        original = model.policy.log_std.detach().clone()
        (model.policy.log_std.sum() * float("nan")).backward()
        with self.assertRaisesRegex(ValueError, "finite"):
            model.policy.optimizer.step()
        self.assertTrue(th.equal(model.policy.log_std, original))
        self.assertEqual(len(model.policy.optimizer.state), 0)

    def test_actor_export_is_ten_outputs_and_exact_for_either_task(self):
        model = self.model(calibration=False)
        model.learn(8, reset_num_timesteps=False)
        obs = th.randn(5, 256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            model.export_actor(path, stage="bridge_b1")
            actor, metadata = self.module.load_actor(path, device="cpu")
            with th.no_grad():
                expected = model.policy.get_distribution(obs).distribution.mean
                self.assertTrue(th.equal(actor(obs), expected))
                self.assertEqual(actor(obs).shape, (5, 10))
                self.assertTrue(th.equal(actor.log_std, model.policy.log_std))
            self.assertEqual(metadata["stage"], "bridge_b1")
            self.assertEqual(metadata["allowed_control_mask"], COMBAT_MASK.tolist())
            np.testing.assert_array_equal(actor.predict(obs[0].numpy(), deterministic=True)[0], model.predict(obs[0].numpy(), deterministic=True)[0])

    def test_settings_are_constant_and_only_combat_entropy_differs_between_arms(self):
        for arm, combat_entropy in (("E1", 0.001), ("E2", 0.003)):
            model = self.model(arm=arm)
            self.assertEqual(model.ent_coef, 0.001)  # calibration always E1
            model.set_task("combat", calibration=False)
            self.assertEqual(model.ent_coef, combat_entropy)
            self.assertEqual(model.lr_schedule(0), 0.0001)
            self.assertEqual(model.lr_schedule(1), 0.0001)
            self.assertEqual(model.clip_range(0), 0.1)
            self.assertEqual(model.target_kl, 0.01)
            self.assertEqual((model.gamma, model.gae_lambda, model.vf_coef, model.max_grad_norm), (0.99, 0.95, 0.5, 0.5))
            model.set_task("navigation")
            self.assertEqual(model.ent_coef, 0.001)
            with self.assertRaisesRegex(ValueError, "calibration"):
                model.set_task("navigation", calibration=True)


if __name__ == "__main__":
    unittest.main()
