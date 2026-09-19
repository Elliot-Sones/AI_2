"""CUDA-only update parity for the four contact action A/B arms.

This gate compares ``sb3`` and ``deferred_metrics`` update semantics within the
same action head/initialization arm. It deliberately does not compare Gaussian
and Bernoulli arms to each other.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from tests import test_ppo_update_optimizations as update_tests


class ContactActionUpdateParityTests(unittest.TestCase):
    assert_tree_equal = update_tests.PPOUpdateParityTests.assert_tree_equal

    arms = ("G_keep", "G_neutral", "B_neutral", "B_projected")

    def setUp(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"Torch unavailable: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable; contact update parity must not fall back to CPU")
        checkpoint = os.environ.get("AI2_CONTACT_CHECKPOINT")
        if not checkpoint:
            self.skipTest("Set AI2_CONTACT_CHECKPOINT to the canonical calibrated contact checkpoint")
        self.torch = torch
        self.device = torch.device("cuda", torch.cuda.current_device())
        import native_curriculum_ppo as native_ppo
        from ppo_update_optimizations import sb3_train_source_metadata
        native_ppo.verify_checkpoint(checkpoint, native_ppo.CALIBRATED_CHECKPOINT_SHA256)
        self.assertTrue(native_ppo.curriculum_source_metadata()["supported"])
        self.assertTrue(sb3_train_source_metadata()["supported"])
        self.assertEqual(native_ppo.contact_settings(update_mode="sb3")["target_kl"], 0.01)
        self.native_ppo = native_ppo
        self.checkpoint = Path(checkpoint)
        self.addCleanup(self.restore_rng, self.rng_state())
        artifact_root = os.environ.get("AI2_CONTACT_UPDATE_PARITY_ARTIFACTS")
        if artifact_root:
            Path(artifact_root).mkdir(parents=True, exist_ok=True)
            self.artifacts = Path(tempfile.mkdtemp(prefix=self._testMethodName + "-", dir=artifact_root))
        else:
            temporary = tempfile.TemporaryDirectory(prefix="contact-action-update-parity-")
            self.addCleanup(temporary.cleanup)
            self.artifacts = Path(temporary.name)

    def rng_state(self):
        torch = self.torch
        return (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()],
        )

    def restore_rng(self, state):
        random.setstate(state[0])
        np.random.set_state(state[1])
        self.torch.set_rng_state(state[2])
        self.torch.cuda.set_rng_state_all(state[3])

    def make_env(self, *, seed):
        from native_contact_lane_vec_env import build_contact_env
        from tensor_ppo import TensorVecEnv

        raw = build_contact_env(
            64,
            device=self.device,
            seed=seed,
            episodes_per_lane=513,
            record_capacity=64 * 512,
        )
        self.addCleanup(raw.close)
        return TensorVecEnv(raw, device=self.device)

    def make_model(self, *, arm, update_mode, seed):
        from stable_baselines3.common.logger import configure

        env = self.make_env(seed=seed)
        model = self.native_ppo.NativeCurriculumPPO.load_contact_checkpoint(
            self.checkpoint,
            env,
            expected_sha256=self.native_ppo.CALIBRATED_CHECKPOINT_SHA256,
            batch_size=8192,
            update_mode=update_mode,
            joint=True,
            action_ab_arm=arm,
        )
        logger = configure(folder=None, format_strings=[])
        self.addCleanup(logger.close)
        model.set_logger(logger)
        self.assertEqual(model.update_mode, update_mode)
        self.assertEqual(model.action_ab_arm, arm)
        self.assertEqual(model.target_kl, 0.01)
        self.assertEqual(model.n_envs, 64)
        self.assertEqual(model.n_steps, 512)
        self.assertEqual(model.batch_size, 8192)
        self.assertEqual(model.n_epochs, 4)
        self.assertTrue(all(parameter.device.type == "cuda" for parameter in model.policy.parameters()))
        return model

    def setup_learn(self, model):
        from stable_baselines3.common.callbacks import BaseCallback

        class Callback(BaseCallback):
            def _on_step(self):
                return True

        _, callback = model._setup_learn(64 * 512, callback=Callback(), reset_num_timesteps=False)
        callback.on_training_start(locals(), globals())
        return callback

    def save_and_replay_rollout(self, source_buffer, target_buffer, *, arm):
        names = (
            "observations",
            "actions",
            "rewards",
            "episode_starts",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "executed_actions",
            "task_ids",
            "allowed_control_masks",
            "_invalid_data",
        )
        self.assertTrue(source_buffer.full)
        self.assertTrue(source_buffer._computed_returns_and_advantage)
        payload = {name: getattr(source_buffer, name).detach().cpu() for name in names}
        path = self.artifacts / f"{self._testMethodName}-{arm}-rollout.pt"
        self.torch.save(payload, path)
        loaded = self.torch.load(path, map_location=self.device, weights_only=True)
        target_buffer.reset()
        for name in names:
            original = getattr(source_buffer, name)
            self.assertEqual(original.device.type, "cuda", name)
            getattr(target_buffer, name).copy_(loaded[name])
            self.assert_tree_equal(original, getattr(target_buffer, name))
        target_buffer.pos = source_buffer.pos
        target_buffer.full = source_buffer.full
        target_buffer.task = source_buffer.task
        target_buffer.consumed = False
        target_buffer._computed_returns_and_advantage = source_buffer._computed_returns_and_advantage
        return {
            "rollout_file": path.name,
            "rollout_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tensors_exact_after_reload": True,
        }

    def sample_trace(self, model):
        samples = []
        original = model.rollout_buffer._get_samples_from_device_indices

        def sample(indices, samples_cls):
            samples.append(indices.detach().cpu().clone())
            return original(indices, samples_cls)

        model.rollout_buffer._get_samples_from_device_indices = sample
        return samples

    def assert_any_parameter_changed(self, before, after):
        self.assertTrue(
            any(not self.torch.equal(before[name], value) for name, value in after.items()),
            "at least one optimizer step must change policy parameters",
        )

    def test_same_arm_sb3_and_deferred_metrics_update_exactly_from_one_saved_rollout(self):
        from benchmark_native_contact_training import ACTION_EXPERIMENT_ARMS, loss_metrics

        receipts = []
        for index, arm in enumerate(self.arms):
            with self.subTest(arm=arm):
                expected_head = ACTION_EXPERIMENT_ARMS[arm]["action_head"]
                reference = self.make_model(arm=arm, update_mode="sb3", seed=145 + index * 97)
                candidate = self.make_model(arm=arm, update_mode="deferred_metrics", seed=945 + index * 97)
                self.assertEqual(reference.action_head, expected_head)
                self.assertEqual(candidate.action_head, expected_head)
                self.assert_tree_equal(reference.policy.state_dict(), candidate.policy.state_dict())
                self.assert_tree_equal(
                    reference.policy.optimizer.state_dict(),
                    candidate.policy.optimizer.state_dict(),
                )
                callback = self.setup_learn(reference)
                before_steps = np.asarray(reference.tensor_vec_env.env.backend.valid_steps).copy()
                self.assertTrue(
                    reference.collect_rollouts(
                        reference.env,
                        callback,
                        reference.rollout_buffer,
                        reference.n_steps,
                    )
                )
                np.testing.assert_array_equal(
                    np.asarray(reference.tensor_vec_env.env.backend.valid_steps) - before_steps,
                    np.full(64, 512),
                )
                self.assertFalse(np.asarray(reference.tensor_vec_env.env.backend.errors).any())
                receipt = self.save_and_replay_rollout(reference.rollout_buffer, candidate.rollout_buffer, arm=arm)
                candidate.num_timesteps = reference.num_timesteps
                candidate._total_timesteps = reference._total_timesteps
                for model in (reference, candidate):
                    model.rollout_buffer.validate("combat", self.native_ppo.COMBAT_MASK)
                    model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
                reference_trace = self.sample_trace(reference)
                candidate_trace = self.sample_trace(candidate)
                initial = {
                    name: value.detach().clone()
                    for name, value in reference.policy.state_dict().items()
                }
                before_rng = self.rng_state()
                reference.train()
                after_rng = self.rng_state()
                self.restore_rng(before_rng)
                candidate.train()
                self.assert_tree_equal(after_rng, self.rng_state())
                self.assert_tree_equal(reference_trace, candidate_trace)
                self.assert_tree_equal(reference.policy.state_dict(), candidate.policy.state_dict())
                self.assert_tree_equal(
                    reference.policy.optimizer.state_dict(),
                    candidate.policy.optimizer.state_dict(),
                )
                self.assert_tree_equal(dict(reference.logger.name_to_value), dict(candidate.logger.name_to_value))
                self.assertEqual(reference._n_updates, candidate._n_updates)
                self.assertEqual(reference.last_update_diagnostics, candidate.last_update_diagnostics)
                self.assertEqual(reference.last_update_diagnostics["kl_early_stop"],
                                 reference.last_update_diagnostics["optimizer_steps"] < 16)
                self.assertGreater(reference.last_update_diagnostics["optimizer_steps"], 0)
                self.assert_any_parameter_changed(initial, reference.policy.state_dict())
                receipt.update({
                    "arm": arm,
                    "action_head": expected_head,
                    "status": "passed",
                    "update": reference.last_update_diagnostics,
                    "sampled_minibatches": len(reference_trace),
                    "parameters_optimizer_rng_samples_metrics": "exact",
                    "metrics": loss_metrics(reference.logger.name_to_value),
                })
                (self.artifacts / f"{self._testMethodName}-{arm}-receipt.json").write_text(
                    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf8",
                )
                receipts.append(receipt)
        print(json.dumps({"contact_action_update_parity": receipts}, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
