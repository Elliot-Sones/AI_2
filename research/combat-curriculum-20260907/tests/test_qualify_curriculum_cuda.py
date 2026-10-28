"""CPU tensor comparisons only; production qualification rejects CPU models."""

import copy
import importlib
import importlib.util
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch as th


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("qualify_curriculum_cuda"), "CUDA qualification helper is missing")
        self.module = importlib.import_module("qualify_curriculum_cuda")

    def states(self):
        parent = {
            "mlp_extractor.policy_net.0.weight": th.arange(32, dtype=th.float32).reshape(4, 8),
            "mlp_extractor.policy_net.0.bias": th.arange(4, dtype=th.float32),
            "mlp_extractor.value_net.0.weight": th.ones(4, 8),
            "mlp_extractor.value_net.0.bias": th.zeros(4),
            "value_net.weight": th.ones(1, 4), "value_net.bias": th.zeros(1),
            "action_net.weight": th.arange(40, dtype=th.float32).reshape(10, 4),
            "action_net.bias": th.arange(10, dtype=th.float32),
            "log_std": th.linspace(-1, 1, 10),
        }
        derived = copy.deepcopy(parent)
        derived["action_net.weight"][7] = 0
        derived["action_net.bias"][7] = 0
        derived["log_std"][7] = math.log(0.3)
        derived.update({
            "combat_value_features.0.weight": th.full((4, 8), 2.),
            "combat_value_features.0.bias": th.zeros(4),
            "combat_value_net.weight": th.full((1, 4), 2.),
            "combat_value_net.bias": th.zeros(1),
            "allowed_control_mask": th.tensor([1, 1, 1, 1, 1, 0, 1, 1, 0, 0], dtype=th.bool),
        })
        return parent, derived

    def test_transfer_comparison_accepts_only_light_change_and_fresh_critic(self):
        parent, derived = self.states()
        result = self.module.compare_transfer_states(parent, derived)
        self.assertTrue(result["movement_bit_exact"])
        self.assertTrue(result["navigation_critic_bit_exact"])
        self.assertTrue(result["combat_critic_fresh"])
        self.assertEqual(result["changed_actor_rows"], [7])

    def test_transfer_rejects_changed_features_navigation_inactive_rows_and_bad_std(self):
        parent, good = self.states()
        mutations = (
            ("mlp_extractor.policy_net.0.weight", (0, 0)),
            ("mlp_extractor.value_net.0.weight", (0, 0)),
            ("action_net.weight", (0, 0)),
            ("action_net.bias", (5,)),
            ("log_std", (6,)),
            ("log_std", (7,)),
        )
        for name, position in mutations:
            derived = copy.deepcopy(good)
            derived[name][position] += 0.1
            with self.subTest(name=name, position=position), self.assertRaises(ValueError):
                self.module.compare_transfer_states(parent, derived)

    def test_transfer_rejects_copy_of_parent_combat_critic_and_missing_parameters(self):
        parent, derived = self.states()
        derived["combat_value_features.0.weight"].fill_(1)
        derived["combat_value_net.weight"].fill_(1)
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.module.compare_transfer_states(parent, derived)
        del derived["value_net.bias"]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.module.compare_transfer_states(parent, derived)

    def test_protected_fingerprints_ignore_light_and_combat_but_detect_movement(self):
        _, state = self.states()
        snapshot = self.module.protected_fingerprints(state)
        changed = copy.deepcopy(state)
        changed["action_net.weight"][7] += 1
        changed["log_std"][7] += 1
        changed["combat_value_net.weight"] += 1
        self.assertEqual(snapshot, self.module.protected_fingerprints(changed))
        self.module.verify_protected_fingerprints(changed, snapshot)
        changed["action_net.weight"][2, 0] += 1
        with self.assertRaisesRegex(ValueError, "action_net.weight"):
            self.module.verify_protected_fingerprints(changed, snapshot)

    def test_binomial_qualification_uses_predeclared_five_sigma_bound(self):
        result = self.module.press_rate_receipt(4780, sample_count=100000)
        self.assertAlmostEqual(result["expected_probability"], 0.04779035227, places=10)
        self.assertEqual(result["z_limit"], 5)
        self.assertTrue(result["passed"])
        with self.assertRaisesRegex(ValueError, "press"):
            self.module.press_rate_receipt(8000, sample_count=100000)
        for count, total in ((-1, 100000), (100001, 100000), (1, 0)):
            with self.subTest(count=count, total=total), self.assertRaises(ValueError):
                self.module.press_rate_receipt(count, sample_count=total)

    def test_real_qualification_entrypoints_reject_cpu_before_inference(self):
        model = SimpleNamespace(policy=th.nn.Linear(256, 10))
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.module.preflight_transfer(model, model)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.module.validate_calibration(model, {})

    def test_state_comparison_and_fingerprinting_reject_nonfinite_values(self):
        parent, derived = self.states()
        derived["action_net.bias"][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.module.compare_transfer_states(parent, derived)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.module.protected_fingerprints(derived)

    def calibration_fixture(self):
        _, state = self.states()
        fingerprints = {name: self.module._fingerprint(value) for name, value in state.items() if name.startswith("combat_")}
        for name in ("action_net.weight", "action_net.bias", "log_std"):
            fingerprints[name + "[7]"] = self.module._fingerprint(state[name][7])
        snapshot = {"schema": "curriculum-calibration-snapshot-v1", "parent_num_timesteps": 10027008,
                    "start_num_timesteps": 10027008, "expected_calibration_transitions": 32768,
                    "protected_parameters": self.module.protected_fingerprints(state),
                    "calibrated_parameters_before": fingerprints}
        state["log_std"][7] += 0.01
        state["combat_value_net.weight"] += 0.1
        return state, snapshot

    def test_calibration_comparison_requires_exact_count_and_both_updates(self):
        self.assertTrue(callable(getattr(self.module, "compare_calibration_state", None)), "Calibration comparison is missing")
        state, snapshot = self.calibration_fixture()
        receipt = self.module.compare_calibration_state(state, snapshot, num_timesteps=10059776, parent_num_timesteps=10027008)
        self.assertTrue(receipt["light_updated"])
        self.assertTrue(receipt["combat_critic_updated"])
        for counter in (10027008, 10059775, 10059777):
            with self.subTest(counter=counter), self.assertRaisesRegex(ValueError, "32768"):
                self.module.compare_calibration_state(state, snapshot, num_timesteps=counter, parent_num_timesteps=10027008)
        _, unchanged = self.states()
        with self.assertRaisesRegex(ValueError, "update"):
            self.module.compare_calibration_state(unchanged, snapshot, num_timesteps=10059776, parent_num_timesteps=10027008)

    def test_calibration_comparison_rejects_mask_or_navigation_drift(self):
        self.assertTrue(callable(getattr(self.module, "compare_calibration_state", None)), "Calibration comparison is missing")
        state, snapshot = self.calibration_fixture()
        for name, position in (("allowed_control_mask", (5,)), ("value_net.weight", (0, 0))):
            altered = copy.deepcopy(state)
            altered[name][position] = 1 if name == "allowed_control_mask" else 2
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.module.compare_calibration_state(altered, snapshot, num_timesteps=10059776, parent_num_timesteps=10027008)

    def assert_calibration_export_stage(self, expected_stage, **kwargs):
        """Exercise stage routing with tensor fixtures, never policy inference."""
        state, snapshot = self.calibration_fixture()

        class ExportReached(Exception):
            pass

        def capture_export(path, *, stage):
            self.assertEqual(stage, expected_stage)
            raise ExportReached

        optimizer = SimpleNamespace(state={"synthetic": {}}, state_dict=lambda: {"synthetic": {}})
        policy = SimpleNamespace(calibration=True, state_dict=lambda: state, optimizer=optimizer)
        model = SimpleNamespace(policy=policy, task="combat", calibration=True, arm="E1", device="cpu",
                                num_timesteps=10059776, parent_num_timesteps=10027008,
                                export_actor=capture_export)
        with (patch.object(self.module, "_require_cuda"),
              patch.object(self.module, "_outputs_by_task", return_value={}),
              patch("curriculum_ppo.CurriculumPPO.load", return_value=model),
              self.assertRaises(ExportReached)):
            self.module.validate_calibration(model, snapshot, reload_path="synthetic-only.zip", **kwargs)

    def test_calibration_export_defaults_to_contact_lesson(self):
        self.assert_calibration_export_stage("bridge_b1_contact")

    def test_calibration_export_retains_explicit_legacy_stage(self):
        self.assert_calibration_export_stage("bridge_b1", stage="bridge_b1")


if __name__ == "__main__":
    unittest.main()
