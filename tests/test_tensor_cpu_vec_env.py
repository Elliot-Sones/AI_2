"""Hybrid CPU-game adapter tests for TensorPPO collection."""

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _deps(*, require_cuda=True):
    try:
        import torch
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:  # pragma: no cover - dependency-light hosts
        raise unittest.SkipTest(f"torch/gymnasium/SB3 unavailable: {exc}") from exc
    if require_cuda and not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA unavailable")
    return torch, spaces, PPO, DummyVecEnv


class TensorCPUGameAdapterStaticTests(unittest.TestCase):
    def test_module_import_is_dependency_light_and_labels_hybrid_bridge(self):
        module = importlib.import_module('ai2.simulation.tensor_cpu_vec_env')
        source = Path(importlib.import_module("ai2.simulation.tensor_cpu_vec_env").__file__).read_text()

        self.assertTrue(hasattr(module, "CPUGameTensorAdapter"))
        self.assertIn("hybrid CPU-game", source)


class TensorCPUGameAdapterDependencyTests(unittest.TestCase):
    def test_rejects_cpu_device_without_cuda_allocation(self):
        torch, spaces, _PPO, _DummyVecEnv = _deps(require_cuda=False)
        module = importlib.import_module('ai2.simulation.tensor_cpu_vec_env')

        class Env:
            num_envs = 2
            observation_space = spaces.Box(-1, 1, shape=(256,), dtype=np.float32)
            action_space = spaces.Box(-1, 1, shape=(10,), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "CUDA"):
            module.CPUGameTensorAdapter(Env(), device=torch.device("cpu"))

    def test_rejects_wrong_existing_vecenv_contract(self):
        torch, spaces, _PPO, _DummyVecEnv = _deps(require_cuda=False)
        module = importlib.import_module('ai2.simulation.tensor_cpu_vec_env')

        class Env:
            num_envs = 2
            observation_space = spaces.Box(-1, 1, shape=(64,), dtype=np.float32)
            action_space = spaces.Box(-1, 1, shape=(10,), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "256"):
            module.CPUGameTensorAdapter(Env(), device=torch.device("cuda:0"))

    def test_reset_seed_sequence_uses_sb3_base_seed_contract(self):
        torch, spaces, _PPO, _DummyVecEnv = _deps(require_cuda=False)
        module = importlib.import_module('ai2.simulation.tensor_cpu_vec_env')

        class Env:
            num_envs = 3
            observation_space = spaces.Box(-1, 1, shape=(256,), dtype=np.float32)
            action_space = spaces.Box(-1, 1, shape=(10,), dtype=np.float32)

            def __init__(self):
                self.seed_calls = []

            def seed(self, seed=None):
                self.seed_calls.append(seed)
                return [seed + idx for idx in range(self.num_envs)]

            def reset(self):
                return np.zeros((self.num_envs, 256), dtype=np.float32)

        env = Env()
        adapter = module.CPUGameTensorAdapter(env, device=torch.device("cuda:0"))
        adapter._torch = type(
            "FakeTorch",
            (),
            {"float32": "float32", "as_tensor": lambda self, array, dtype, device: (array, dtype, device)},
        )()

        with self.assertRaisesRegex(ValueError, "integers"):
            adapter.reset_tensor(seeds=[1, True, 3])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            adapter.reset_tensor(seeds=[10, 12, 13])

        adapter.reset_tensor(seeds=[10, 11, 12])
        self.assertEqual(env.seed_calls, [10])

    def test_host_timeout_indices_contract_is_numpy_int64_copy(self):
        module = importlib.import_module('ai2.simulation.tensor_cpu_vec_env')
        adapter = module.CPUGameTensorAdapter.__new__(module.CPUGameTensorAdapter)
        adapter._last_timeout_indices = np.asarray([2, 5], dtype=np.int64)

        first = adapter.host_timeout_indices()
        first[0] = 99

        np.testing.assert_array_equal(adapter.host_timeout_indices(), np.asarray([2, 5], dtype=np.int64))

    def test_no_timeout_step_preserves_host_terminal_info_without_cuda_terminal_upload(self):
        module = importlib.import_module('ai2.simulation.tensor_cpu_vec_env')

        class FakeAction:
            is_cuda = True
            device = "cuda:0"
            dtype = "float32"
            shape = (2, 10)
            requires_grad = False

            def is_contiguous(self):
                return True

            def detach(self):
                return self

            def to(self, device):
                self.copied_to = device
                return self

            def numpy(self):
                return np.zeros((2, 10), dtype=np.float32)

        class FakeTorch:
            float32 = "float32"
            bool = "bool"

            def __init__(self):
                self.uploads = []

            def is_tensor(self, value):
                return isinstance(value, FakeAction)

            def as_tensor(self, array, dtype, device):
                self.uploads.append((np.asarray(array).copy(), dtype, device))
                return {"shape": tuple(np.asarray(array).shape), "dtype": dtype, "device": device}

        class FakeVecEnv:
            def step(self, actions):
                np.testing.assert_array_equal(actions, np.zeros((2, 10), dtype=np.float32))
                obs = np.ones((2, 256), dtype=np.float32)
                rewards = np.asarray([0.0, 1.0], dtype=np.float32)
                dones = np.asarray([False, True])
                infos = [{}, {"terminal_observation": np.full((256,), 7.0, dtype=np.float32)}]
                return obs, rewards, dones, infos

        fake_torch = FakeTorch()
        adapter = module.CPUGameTensorAdapter.__new__(module.CPUGameTensorAdapter)
        adapter.vec_env = FakeVecEnv()
        adapter.device = "cuda:0"
        adapter.num_envs = 2
        adapter._torch = fake_torch
        adapter._last_infos = []
        adapter._last_dones = np.zeros((2,), dtype=bool)
        adapter._last_timeout_indices = np.empty((0,), dtype=np.int64)
        adapter._episode_infos = []

        step = adapter.step_tensor(FakeAction())

        self.assertIsNone(step.terminal_observations)
        self.assertIn("terminal_observation", adapter.host_step_infos()[1])
        np.testing.assert_array_equal(adapter.host_timeout_indices(), np.empty((0,), dtype=np.int64))
        self.assertEqual([upload[0].shape for upload in fake_torch.uploads], [(2, 256), (2,), (2,), (2,)])


class TensorCPUGameAdapterCudaTests(unittest.TestCase):
    def setUp(self):
        torch, spaces, PPO, DummyVecEnv = _deps()
        self.torch = torch
        self.spaces = spaces
        self.PPO = PPO
        self.DummyVecEnv = DummyVecEnv
        self.adapter_module = importlib.import_module('ai2.simulation.tensor_cpu_vec_env')
        self.tensor_ppo_module = importlib.import_module('ai2.training.tensor_ppo')
        self.device = torch.device("cuda", torch.cuda.current_device())

    def make_vec_env(self, *, n_envs=4):
        spaces = self.spaces
        import gymnasium as gym

        class LaneEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self, lane):
                self.lane = lane
                self.observation_space = spaces.Box(-1000, 1000, shape=(256,), dtype=np.float32)
                self.action_space = spaces.Box(-0.25, 0.25, shape=(10,), dtype=np.float32)
                self.step_count = 0
                self.actions = []

            def _obs(self):
                return (np.arange(256, dtype=np.float32) + self.lane * 1000 + self.step_count).astype(np.float32)

            def reset(self, seed=None, options=None):
                del seed, options
                self.step_count = 0
                return self._obs(), {"reset_lane": self.lane}

            def step(self, action):
                clipped = np.asarray(action, dtype=np.float32).copy()
                self.actions.append(clipped)
                terminal_obs = self._obs()
                reward = np.float64(self.lane + 0.01 * clipped.sum())
                terminated = self.step_count == 0 and self.lane in (1, 3)
                truncated = self.step_count == 0 and self.lane in (2, 3)
                done = terminated or truncated
                self.step_count += 1
                # Gymnasium envs return the terminal step observation when an
                # episode ends; SB3 VecEnv then saves that as
                # info["terminal_observation"] before autoreset.
                obs = terminal_obs if done else self._obs()
                info = {
                    "lane": self.lane,
                    "episode": {"r": float(self.lane), "l": self.step_count},
                    "is_success": self.lane == 1,
                }
                if done:
                    info["terminal_observation"] = terminal_obs.copy()
                return obs, reward, terminated, truncated, info

        lanes = [LaneEnv(i) for i in range(n_envs)]
        return self.DummyVecEnv([lambda lane=lane: lane for lane in lanes]), lanes

    def test_adapter_bridges_one_action_batch_and_preserves_terminal_metadata(self):
        torch = self.torch
        vec_env, lanes = self.make_vec_env(n_envs=4)
        adapter = self.adapter_module.CPUGameTensorAdapter(vec_env, device=self.device)
        reset_obs = adapter.reset_tensor()
        raw_actions = torch.linspace(-1.0, 1.0, steps=40, dtype=torch.float32, device=self.device).reshape(4, 10)
        actions = torch.clamp(raw_actions, -0.25, 0.25)

        step = adapter.step_tensor(actions)

        self.assertEqual(reset_obs.shape, (4, 256))
        self.assertTrue(step.observations.is_cuda)
        self.assertTrue(step.rewards.is_cuda)
        self.assertTrue(step.terminated.is_cuda)
        self.assertTrue(step.truncated.is_cuda)
        np.testing.assert_allclose(lanes[0].actions[0], actions.detach().cpu().numpy()[0], rtol=0, atol=0)
        infos = adapter.host_step_infos()
        self.assertEqual(infos[1]["lane"], 1)
        self.assertEqual(infos[2]["TimeLimit.truncated"], True)
        self.assertEqual(adapter.host_step_dones().tolist(), [False, True, True, True])
        np.testing.assert_array_equal(adapter.host_timeout_indices(), np.asarray([2], dtype=np.int64))
        self.assertEqual(adapter.drain_episode_infos(), [])
        torch.testing.assert_close(step.terminal_observations[1].detach().cpu(), torch.as_tensor(np.arange(256, dtype=np.float32) + 1000), rtol=0, atol=0)
        torch.testing.assert_close(step.terminal_observations[0].detach().cpu(), step.observations[0].detach().cpu(), rtol=0, atol=0)

    def test_episode_infos_can_drain_when_no_per_step_callback_consumes_them(self):
        torch = self.torch
        vec_env, _lanes = self.make_vec_env(n_envs=4)
        adapter = self.adapter_module.CPUGameTensorAdapter(vec_env, device=self.device)
        adapter.reset_tensor()
        actions = torch.zeros((4, 10), dtype=torch.float32, device=self.device)

        adapter.step_tensor(actions)

        self.assertEqual(len(adapter.drain_episode_infos()), 4)
        self.assertEqual(adapter.drain_episode_infos(), [])

    def test_tensor_ppo_callbacks_receive_real_host_infos_from_cpu_adapter(self):
        vec_env, _lanes = self.make_vec_env(n_envs=4)
        cpu_adapter = self.adapter_module.CPUGameTensorAdapter(vec_env, device=self.device)
        tensor_env = self.tensor_ppo_module.TensorVecEnv(cpu_adapter, device=self.device)
        model = self.tensor_ppo_module.TensorPPO(
            "MlpPolicy",
            tensor_env,
            device=self.device,
            seed=123,
            n_steps=1,
            batch_size=4,
            n_epochs=1,
            policy_kwargs={"net_arch": [8]},
            verbose=0,
        )
        seen_infos = []
        base = __import__("stable_baselines3.common.callbacks", fromlist=["BaseCallback"]).BaseCallback

        class Callback(base):
            def _on_step(self):
                seen_infos.append(self.locals["infos"])
                return True

        model.learn(total_timesteps=4, callback=Callback())

        self.assertEqual(len(seen_infos), 1)
        self.assertEqual(seen_infos[0][1]["lane"], 1)
        self.assertIn("r", model.ep_info_buffer[0])
        self.assertIn("l", model.ep_info_buffer[0])

    def test_from_sb3_preserves_loaded_policy_optimizer_and_hyperparameters(self):
        torch = self.torch
        vec_env, _lanes = self.make_vec_env(n_envs=4)
        stock_model = self.PPO(
            "MlpPolicy",
            vec_env,
            device=self.device,
            seed=321,
            n_steps=2,
            batch_size=4,
            n_epochs=3,
            gamma=0.91,
            gae_lambda=0.87,
            learning_rate=0.0007,
            ent_coef=0.02,
            vf_coef=0.6,
            max_grad_norm=0.4,
            policy_kwargs={"net_arch": [8]},
            verbose=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model.zip"
            stock_model.save(path)
            loaded = self.PPO.load(path, env=vec_env, device=self.device)
            cpu_adapter = self.adapter_module.CPUGameTensorAdapter(vec_env, device=self.device)
            tensor_env = self.tensor_ppo_module.TensorVecEnv(cpu_adapter, device=self.device)
            tensor_model = self.tensor_ppo_module.TensorPPO.from_sb3(loaded, tensor_env)
            tensor_loaded = self.tensor_ppo_module.TensorPPO.load(path, env=tensor_env, device=self.device)

        self.assertEqual(tensor_model.n_steps, loaded.n_steps)
        self.assertEqual(tensor_model.batch_size, loaded.batch_size)
        self.assertEqual(tensor_model.n_epochs, loaded.n_epochs)
        self.assertEqual(tensor_model.gamma, loaded.gamma)
        self.assertEqual(tensor_model.gae_lambda, loaded.gae_lambda)
        self.assertEqual(tensor_model.ent_coef, loaded.ent_coef)
        self.assertEqual(tensor_model.vf_coef, loaded.vf_coef)
        self.assertEqual(tensor_model.max_grad_norm, loaded.max_grad_norm)
        for (name, expected), (_, actual) in zip(loaded.policy.state_dict().items(), tensor_model.policy.state_dict().items()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0)
        self.assertEqual(tensor_model.policy.optimizer.state_dict()["param_groups"], loaded.policy.optimizer.state_dict()["param_groups"])
        for (name, expected), (_, actual) in zip(loaded.policy.state_dict().items(), tensor_loaded.policy.state_dict().items()):
            with self.subTest(load_parameter=name):
                torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0)
        self.assertEqual(tensor_loaded.policy.optimizer.state_dict()["param_groups"], loaded.policy.optimizer.state_dict()["param_groups"])


if __name__ == "__main__":
    unittest.main()
