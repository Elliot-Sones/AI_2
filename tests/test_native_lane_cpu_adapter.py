"""CPU-native bridge semantics; local tests run C fixtures, never CPU policies.

The local fixture substitutes only action-device validation and CUDA copies.
The CUDA test exercises the real TensorVecEnv device contract when available.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

class NativeLaneCPUAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            importlib.import_module("gymnasium")
        except ImportError as exc:
            raise unittest.SkipTest(f"torch/gymnasium unavailable: {exc}") from exc
        if shutil.which("cc") is None:
            raise unittest.SkipTest("C compiler unavailable")
        cls.torch = torch
        cls.lanes = importlib.import_module("native_lane_vec_env")
        cls.tmp = tempfile.TemporaryDirectory(prefix="ai2-native-cpu-bridge-")
        cls.addClassCleanup(cls.tmp.cleanup)

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("native_lane_cpu_adapter"), "CPU-native CUDA bridge is missing")
        self.module = importlib.import_module("native_lane_cpu_adapter")

    def make_env(self, n=3, *, time_limit=3):
        cfg = {**self.lanes.PHASE_0B, "time_limit": time_limit}
        env = self.lanes.build_cpu_env(n, build_dir=Path(self.tmp.name), wrapper_cfg=cfg, seed=45, contract="off")
        self.addCleanup(env.close)
        return env

    @contextmanager
    def cpu_fixture_transfers(self, adapter):
        """Keep the real C simulation, with CUDA copies represented by independent CPU tensors."""
        torch = self.torch
        original_to = torch.Tensor.to
        transfers = []

        def transfer(tensor, *args, **kwargs):
            device = kwargs.get("device", args[0] if args else None)
            if isinstance(device, (str, torch.device)):
                transfers.append((str(device), tuple(tensor.shape), kwargs.get("non_blocking", False)))
                if torch.device(device).type == "cuda":
                    return tensor.detach().clone()
            return original_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", transfer), patch.object(adapter, "_require_cuda_actions", side_effect=lambda value: value):
            yield transfers

    def test_rejects_cpu_policy_and_non_cpu_simulation(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.module.NativeLaneCPUAdapter(object(), device="cpu")
        cuda_env = types.SimpleNamespace(device=self.torch.device("cuda:0"))
        with self.assertRaisesRegex(ValueError, "CPU"):
            self.module.NativeLaneCPUAdapter(cuda_env, device="cuda:0")

    def test_rejects_host_actions_before_native_step(self):
        env = self.make_env()
        adapter = self.module.NativeLaneCPUAdapter(env, device="cuda:0")
        with patch.object(env, "step_tensor") as step:
            with self.assertRaisesRegex(ValueError, "CUDA"):
                adapter.step_tensor(self.torch.zeros((3, 10), dtype=self.torch.float32))
        step.assert_not_called()

    def test_reset_passes_seed_sequence_and_copies_full_stack(self):
        env, reference = self.make_env(), self.make_env()
        adapter = self.module.NativeLaneCPUAdapter(env, device="cuda:0")
        seeds = [45, 700, 2]
        self.assertEqual(adapter.seed(seeds), seeds)
        with self.cpu_fixture_transfers(adapter) as transfers:
            result = adapter.reset_tensor(seeds=seeds)
        self.assertTrue(self.torch.equal(result, reference.reset_tensor(seeds=seeds)))
        self.assertNotEqual(result.data_ptr(), env.frame_stack.data_ptr())
        self.assertTrue(bool((result[:, :192] == 0).all()))
        self.assertEqual(transfers, [("cuda:0", (3, 256), False)])
        self.assertIs(adapter.observation_space, env.observation_space)
        self.assertIs(adapter.action_space, env.action_space)
        self.assertEqual(adapter.num_envs, 3)

    def test_real_native_steps_preserve_timeout_terminal_stack_autoreset_and_records(self):
        torch = self.torch
        env, reference = self.make_env(), self.make_env()
        adapter = self.module.NativeLaneCPUAdapter(env, device="cuda:0")
        actions = torch.zeros((3, 10), dtype=torch.float32)
        reference.reset_tensor()
        with self.cpu_fixture_transfers(adapter) as transfers:
            previous = adapter.reset_tensor()
            for tick in range(6):
                transfers.clear()
                got = adapter.step_tensor(actions)
                expected = reference.step_tensor(actions)
                for field in ("observations", "rewards", "terminated", "truncated", "terminal_observations"):
                    self.assertTrue(torch.equal(getattr(got, field), getattr(expected, field)), (tick, field))
                self.assertTrue(torch.equal(got.terminal_observations[:, :192], previous[:, 64:]))
                self.assertEqual(bool(got.truncated.all()), tick % 3 == 2)
                self.assertFalse(bool(got.terminated.any()))
                if tick % 3 == 2:
                    self.assertTrue(bool((got.observations[:, :192] == 0).all()))
                    self.assertFalse(torch.equal(got.observations[:, 192:], got.terminal_observations[:, 192:]))
                self.assertEqual([device for device, _shape, _async in transfers], ["cpu"] + ["cuda:0"] * 5)
                self.assertTrue(all(not asynchronous for _, _, asynchronous in transfers))
                previous = got.observations.clone()
        got_infos, expected_infos = adapter.drain_episode_infos(), reference.drain_episode_infos()
        self.assertEqual(len(got_infos), 6)
        for got, expected in zip(got_infos, expected_infos):
            self.assertEqual(got["navigation"], expected["navigation"])
            self.assertEqual(got["episode"]["r"], expected["episode"]["r"])
            self.assertEqual(got["episode"]["l"], 3)
        self.assertEqual(adapter.drain_episode_infos(), [])
        adapter.close()
        self.assertIsNone(env.backend.handle)

    def test_real_native_termination_does_not_become_timeout(self):
        torch = self.torch
        env = self.make_env(time_limit=1)
        adapter = self.module.NativeLaneCPUAdapter(env, device="cuda:0")
        with self.cpu_fixture_transfers(adapter):
            initial = adapter.reset_tensor()
            state = env.backend.lane_state(0)
            state.ep.players[0].ms.stocks = 0
            env.backend.set_lane_state(0, state)
            got = adapter.step_tensor(torch.zeros((3, 10), dtype=torch.float32))
        self.assertTrue(bool(got.terminated[0]))
        self.assertFalse(bool(got.truncated[0]))
        self.assertFalse(bool(got.terminated[1:].any()))
        self.assertTrue(bool(got.truncated[1:].all()))
        self.assertTrue(bool((got.observations[:, :192] == 0).all()))
        self.assertTrue(torch.equal(got.terminal_observations[:, :192], initial[:, 64:]))
        records = adapter.drain_episode_infos()
        self.assertEqual(len(records), 3)
        self.assertTrue(records[0]["navigation"]["terminated"])
        self.assertFalse(records[0]["navigation"]["truncated"])

    def test_cuda_bridge_matches_cpu_reference_without_running_a_policy(self):
        torch = self.torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable; local fixtures do not validate real GPU transfers")
        from tensor_ppo import TensorVecEnv

        env, reference = self.make_env(), self.make_env()
        adapter = self.module.NativeLaneCPUAdapter(env, device="cuda")
        tensor_env = TensorVecEnv(adapter, device="cuda")
        self.assertTrue(torch.equal(tensor_env.reset_tensor().cpu(), reference.reset_tensor()))
        actions = torch.zeros((3, 10), dtype=torch.float32, device="cuda")
        for _ in range(6):
            got = tensor_env.step_tensor(actions)
            expected = reference.step_tensor(actions.cpu())
            for field in ("observations", "rewards", "terminated", "truncated", "terminal_observations"):
                self.assertEqual(getattr(got, field).device, tensor_env.device)
                self.assertTrue(torch.equal(getattr(got, field).cpu(), getattr(expected, field)))
        self.assertEqual(len(tensor_env.drain_episode_infos()), 6)


if __name__ == "__main__":
    unittest.main()
