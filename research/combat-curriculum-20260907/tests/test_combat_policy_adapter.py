"""Combat adapter contracts; no real checkpoint inference runs on the CPU."""

import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


def box(size, low=0.0, high=1.0):
    return SimpleNamespace(
        shape=(size,), low=np.full(size, low), high=np.full(size, high)
    )


class FakePolicy:
    def __init__(self, device="cpu", parameter_devices=None, buffer_devices=()):
        self.device = SimpleNamespace(type=device)
        self.parameter_devices = parameter_devices or [device]
        self.buffer_devices = buffer_devices

    def parameters(self):
        return iter(
            SimpleNamespace(device=SimpleNamespace(type=device))
            for device in self.parameter_devices
        )

    def buffers(self):
        return iter(
            SimpleNamespace(device=SimpleNamespace(type=device))
            for device in self.buffer_devices
        )


class FakeModel:
    def __init__(self, device="cpu"):
        self.observation_space = box(256)
        self.action_space = box(10)
        self.device = SimpleNamespace(type=device)
        self.policy = FakePolicy(device)
        self.calls = []
        self.result = (np.linspace(0.0, 1.0, 10), None)

    def predict(self, observation, deterministic=True):
        self.calls.append((observation, deterministic))
        return self.result


class AdapterTest(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("combat_policy_adapter") is None:
            self.fail("combat_policy_adapter is not implemented yet")
        self.module = importlib.import_module("combat_policy_adapter")
        self.model = FakeModel()
        self.adapter = self.module.SB3CombatPolicy(self.model, require_cuda=False)

    def test_first_frame_is_zero_padded_and_oldest_frame_rolls_out(self):
        frames = [np.full(64, value, dtype=np.float32) for value in range(1, 7)]
        for index, frame in enumerate(frames):
            self.adapter.predict(frame)
            history = [np.zeros(64, dtype=np.float32)] * 3 + frames[: index + 1]
            expected = np.concatenate(history[-4:])
            np.testing.assert_array_equal(self.model.calls[-1][0], expected)
            self.assertEqual(self.model.calls[-1][0].dtype, np.float32)

    def test_reset_mid_sequence_clears_all_previous_frames(self):
        self.adapter.predict(np.full(64, 17))
        self.adapter.predict(np.full(64, 18))
        self.adapter.reset()
        self.adapter.predict(np.full(64, 23))
        np.testing.assert_array_equal(
            self.model.calls[-1][0], np.r_[np.zeros(192), np.full(64, 23)]
        )

    def test_two_adapters_share_weights_but_not_history(self):
        other = self.module.SB3CombatPolicy(self.model, require_cuda=False)
        self.adapter.predict(np.full(64, 11))
        other.predict(np.full(64, 21))
        self.adapter.predict(np.full(64, 12))
        other.reset()
        other.predict(np.full(64, 22))
        np.testing.assert_array_equal(
            self.model.calls[2][0], np.r_[np.zeros(128), np.full(64, 11), np.full(64, 12)]
        )
        np.testing.assert_array_equal(
            self.model.calls[3][0], np.r_[np.zeros(192), np.full(64, 22)]
        )
        self.assertIs(self.adapter.model, other.model)

    def test_matches_installed_sb3_stack_across_episode_reset(self):
        from gymnasium import spaces
        from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

        reference = StackedObservations(
            1, 4, spaces.Box(-np.inf, np.inf, shape=(64,), dtype=np.float32)
        )
        rng = np.random.default_rng(7)
        for episode in range(2):
            self.adapter.reset()
            for step in range(7):
                observation = rng.normal(size=64).astype(np.float32)
                if step == 0:
                    expected = reference.reset(observation[None, :])
                else:
                    expected, _ = reference.update(
                        observation[None, :], np.array([False]), [{}]
                    )
                self.adapter.predict(observation)
                np.testing.assert_array_equal(self.model.calls[-1][0], expected[0])

    def test_input_and_model_output_buffers_are_not_shared(self):
        observation = np.arange(64, dtype=np.float32)
        original = observation.copy()
        output = self.adapter.predict(observation)
        np.testing.assert_array_equal(observation, original)
        saved_model_input = self.model.calls[0][0]
        observation[:] = -30
        output[:] = 0
        self.adapter.predict(np.full(64, 4))
        np.testing.assert_array_equal(saved_model_input[-64:], original)
        np.testing.assert_array_equal(self.model.calls[1][0][128:192], original)
        np.testing.assert_array_equal(self.model.result[0], np.linspace(0.0, 1.0, 10))
        self.assertFalse(np.shares_memory(output, self.model.result[0]))
        self.assertFalse(np.shares_memory(saved_model_input, self.model.calls[1][0]))

    def test_model_cannot_mutate_adapter_history_through_prediction_input(self):
        def mutating_predict(observation, deterministic):
            self.model.calls.append(observation.copy())
            observation[:] = -99
            return np.zeros(10), None

        self.model.predict = mutating_predict
        self.adapter.predict(np.full(64, 3))
        self.adapter.predict(np.full(64, 4))
        np.testing.assert_array_equal(self.model.calls[1][128:192], np.full(64, 3))

    def test_action_tuple_is_unpacked_clipped_and_float32(self):
        self.model.result = (np.linspace(-1.0, 2.0, 10), "ignored recurrent state")
        action = self.adapter.predict(np.zeros(64))
        self.assertEqual(action.shape, (10,))
        self.assertEqual(action.dtype, np.float32)
        np.testing.assert_allclose(action, np.clip(self.model.result[0], 0, 1))
        self.assertTrue(self.model.calls[-1][1])
        self.assertTrue(self.adapter.deterministic)

    def test_sampled_actions_forward_deterministic_false(self):
        adapter = self.module.SB3CombatPolicy(
            self.model, deterministic=False, require_cuda=False
        )
        adapter.predict(np.zeros(64))
        self.assertFalse(self.model.calls[-1][1])
        self.assertFalse(adapter.deterministic)

    def test_invalid_observations_fail_before_inference(self):
        for observation in (
            np.zeros(63), np.zeros(65), np.zeros((1, 64)),
            np.full(64, np.nan), np.full(64, np.inf), ["invalid"] * 64,
        ):
            with self.subTest(shape=np.shape(observation)):
                with self.assertRaises(ValueError):
                    self.adapter.predict(observation)
        self.assertEqual(self.model.calls, [])

    def test_invalid_actions_and_prediction_tuples_fail_closed(self):
        for prediction in (
            np.zeros(10), (np.zeros(10),), (np.zeros(10), None, None),
            (np.zeros(9), None), (np.zeros((1, 10)), None),
            (np.full(10, np.nan), None), (np.full(10, np.inf), None),
            (["invalid"] * 10, None),
        ):
            with self.subTest(prediction=repr(prediction)):
                self.model.result = prediction
                with self.assertRaises(ValueError):
                    self.adapter.predict(np.zeros(64))

    def test_incompatible_model_spaces_fail_at_construction(self):
        for name, space in (
            ("observation_space", box(64)),
            ("observation_space", SimpleNamespace(shape=(1, 256))),
            ("action_space", box(9)),
            ("action_space", box(10, low=-1)),
            ("action_space", box(10, high=2)),
            ("action_space", box(10, high=np.nan)),
            ("action_space", SimpleNamespace(shape=(10,))),
        ):
            with self.subTest(name=name, space=space):
                model = FakeModel()
                setattr(model, name, space)
                with self.assertRaises(ValueError):
                    self.module.SB3CombatPolicy(model, require_cuda=False)

    def test_get_env_info_validates_box_attributes_and_resets(self):
        self.adapter.predict(np.ones(64))
        self.adapter.get_env_info(
            SimpleNamespace(observation_space=box(64), action_space=box(10))
        )
        self.adapter.predict(np.full(64, 2))
        np.testing.assert_array_equal(self.model.calls[-1][0][:192], np.zeros(192))

    def test_get_env_info_accepts_space_methods(self):
        self.adapter.get_env_info(SimpleNamespace(
            observation_space=lambda agent_id: box(64),
            action_space=lambda agent_id: box(10),
        ))

    def test_get_env_info_rejects_incompatible_raw_spaces(self):
        for observation_space, action_space in (
            (box(256), box(10)), (box(64), box(9)), (box(64), box(10, low=-1)),
        ):
            with self.subTest(observation_space=observation_space, action_space=action_space):
                with self.assertRaises(ValueError):
                    self.adapter.get_env_info(SimpleNamespace(
                        observation_space=observation_space, action_space=action_space
                    ))

    def test_cuda_is_required_by_default_for_model_and_policy(self):
        for model_device, policy_device in (("cpu", "cpu"), ("cpu", "cuda"), ("cuda", "cpu")):
            with self.subTest(model=model_device, policy=policy_device):
                model = FakeModel(model_device)
                model.policy = FakePolicy(policy_device)
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    self.module.SB3CombatPolicy(model)

    def test_cuda_validation_checks_every_parameter_and_buffer(self):
        for parameter_devices, buffer_devices in (
            (["cuda", "cpu"], []), (["cuda"], ["cpu"]),
        ):
            with self.subTest(parameters=parameter_devices, buffers=buffer_devices):
                model = FakeModel("cuda")
                model.policy = FakePolicy("cuda", parameter_devices, buffer_devices)
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    self.module.SB3CombatPolicy(model)

    def test_cuda_labels_and_tensor_devices_are_accepted(self):
        model = FakeModel("cuda")
        adapter = self.module.SB3CombatPolicy(model)
        self.assertIs(adapter.model, model)

    def test_real_sb3_model_cannot_opt_out_of_cuda(self):
        from stable_baselines3 import PPO

        # Allocate no tensors or network: only attach the metadata checked by the adapter.
        model = PPO.__new__(PPO)
        model.observation_space = box(256)
        model.action_space = box(10)
        model.device = SimpleNamespace(type="cpu")
        model.policy = FakePolicy()
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.module.SB3CombatPolicy(model, require_cuda=False)

    def test_import_does_not_load_torch_or_sb3(self):
        result = subprocess.run(
            [sys.executable, "-c", "import combat_policy_adapter, sys; "
             "assert 'torch' not in sys.modules; assert 'stable_baselines3' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_checkpoint_fails_before_cuda_or_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("torch.cuda.is_available") as available, patch("stable_baselines3.PPO.load") as load:
                with self.assertRaises(FileNotFoundError):
                    self.module.load_cuda_checkpoint(Path(directory) / "absent.zip")
                available.assert_not_called()
                load.assert_not_called()

    def test_load_fails_if_cuda_unavailable_without_loading(self):
        with tempfile.NamedTemporaryFile(suffix=".zip") as checkpoint:
            with patch("torch.cuda.is_available", return_value=False), patch("stable_baselines3.PPO.load") as load:
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    self.module.load_cuda_checkpoint(checkpoint.name)
                load.assert_not_called()

    def test_load_uses_positional_path_and_cuda_then_returns_model(self):
        model = FakeModel("cuda")
        with tempfile.NamedTemporaryFile(suffix=".zip") as checkpoint:
            with patch("torch.cuda.is_available", return_value=True), patch("stable_baselines3.PPO.load", return_value=model) as load:
                actual = self.module.load_cuda_checkpoint(Path(checkpoint.name))
                self.assertIs(actual, model)
                load.assert_called_once_with(checkpoint.name, device="cuda")

    def test_load_rejects_cpu_fallback_and_incompatible_model(self):
        incompatible = FakeModel("cuda")
        incompatible.observation_space = box(64)
        for model in (FakeModel(), incompatible):
            with self.subTest(model=model):
                with tempfile.NamedTemporaryFile(suffix=".zip") as checkpoint:
                    with patch("torch.cuda.is_available", return_value=True), patch("stable_baselines3.PPO.load", return_value=model):
                        with self.assertRaises((RuntimeError, ValueError)):
                            self.module.load_cuda_checkpoint(checkpoint.name)


if __name__ == "__main__":
    unittest.main()
