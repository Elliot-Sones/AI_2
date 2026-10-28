"""Real SB3 StackedObservations differential tests, without policy training."""
import importlib
from pathlib import Path
import unittest
import numpy as np


def _deps():
    try:
        import torch
        from gymnasium.spaces import Box
        from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
    except ImportError as exc:
        raise unittest.SkipTest(f"runtime unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA unavailable")
    return torch, Box, StackedObservations


class TensorFrameStackTests(unittest.TestCase):
    def test_import_is_dependency_light(self):
        self.assertTrue(hasattr(importlib.import_module('ai2.training.tensor_frame_stack'), "TensorFrameStack"))

    def test_source_limits_cpu_sync_to_explicit_check_valid_boundary(self):
        source = Path(importlib.import_module("ai2.training.tensor_frame_stack").__file__).read_text()
        reset_body = source.split("    def reset(")[1].split("    def update(")[0]
        update_body = source.split("    def update(")[1].split("    def check_valid(")[0]

        self.assertIn("def check_valid", source)
        self.assertNotIn(".item()", reset_body)
        self.assertNotIn(".item()", update_body)

    def assert_matches_sb3(self, n_envs, *, ticks):
        torch, Box, StackedObservations = _deps()
        native = importlib.import_module('ai2.training.tensor_frame_stack')
        rng = np.random.default_rng(981 + n_envs)
        space = Box(-np.inf, np.inf, shape=(64,), dtype=np.float32)
        reference = StackedObservations(n_envs, 4, space)
        candidate = native.TensorFrameStack(n_envs, device="cuda")
        obs = rng.standard_normal((n_envs, 64)).astype(np.float32)
        expected = reference.reset(obs)
        actual = candidate.reset(torch.as_tensor(obs, device="cuda"))
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for tick in range(ticks):
            obs = rng.standard_normal((n_envs, 64)).astype(np.float32)
            terminal = rng.standard_normal((n_envs, 64)).astype(np.float32)
            dones = (np.arange(n_envs) + tick) % 7 == 0
            if tick == 20:
                dones[:] = True
            infos = [{"terminal_observation": terminal[i].copy()} if dones[i] else {} for i in range(n_envs)]
            expected, expected_infos = reference.update(obs, dones, infos)
            with torch.cuda.stream(stream):
                actual, final = candidate.update(torch.as_tensor(obs, device="cuda"), torch.as_tensor(terminal, device="cuda"), torch.as_tensor(dones, device="cuda"))
            stream.synchronize()
            np.testing.assert_array_equal(actual.cpu().numpy(), expected)
            for i in np.flatnonzero(dones):
                np.testing.assert_array_equal(final[i].cpu().numpy(), expected_infos[i]["terminal_observation"])
        candidate.check_valid()

    def test_staggered_terminal_and_reset_history_matches_sb3_for_supported_lane_counts(self):
        for n_envs, ticks in ((1, 16), (8, 72), (256, 16), (4096, 4)):
            with self.subTest(n_envs=n_envs):
                self.assert_matches_sb3(n_envs, ticks=ticks)

    def test_alias_and_terminal_snapshot_contract(self):
        torch, _Box, _StackedObservations = _deps()
        native = importlib.import_module('ai2.training.tensor_frame_stack')
        candidate = native.TensorFrameStack(2, device="cuda")
        obs0 = torch.arange(128, dtype=torch.float32, device="cuda").reshape(2, 64)
        reset_out = candidate.reset(obs0)
        self.assertEqual(reset_out.data_ptr(), candidate._stack.data_ptr())

        obs1 = torch.full((2, 64), 7.0, dtype=torch.float32, device="cuda")
        terminal = torch.full((2, 64), 11.0, dtype=torch.float32, device="cuda")
        dones = torch.tensor([True, False], dtype=torch.bool, device="cuda")
        update_out, terminal_stack = candidate.update(obs1, terminal, dones)
        self.assertEqual(update_out.data_ptr(), candidate._stack.data_ptr())
        self.assertEqual(reset_out.data_ptr(), update_out.data_ptr())
        self.assertNotEqual(terminal_stack.data_ptr(), terminal.data_ptr())
        terminal_stack_snapshot = terminal_stack.clone()
        terminal.add_(99.0)
        candidate.update(obs1 + 1.0, terminal + 1.0, dones)
        torch.testing.assert_close(terminal_stack, terminal_stack_snapshot, rtol=0, atol=0)

    def test_producer_stream_inputs_are_valid_when_update_stream_waits(self):
        torch, _Box, _StackedObservations = _deps()
        native = importlib.import_module('ai2.training.tensor_frame_stack')
        candidate = native.TensorFrameStack(4, device="cuda")
        candidate.reset(torch.zeros((4, 64), dtype=torch.float32, device="cuda"))
        producer = torch.cuda.Stream()
        update_stream = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            obs = torch.full((4, 64), 3.0, dtype=torch.float32, device="cuda")
            terminal = torch.full((4, 64), 5.0, dtype=torch.float32, device="cuda")
            dones = torch.tensor([False, True, False, True], dtype=torch.bool, device="cuda")
        update_stream.wait_stream(producer)
        with torch.cuda.stream(update_stream):
            actual, final = candidate.update(obs, terminal, dones)
        update_stream.synchronize()

        self.assertTrue(torch.all(actual[:, -64:] == 3.0).item())
        self.assertTrue(torch.all(final[:, -64:] == 5.0).item())
        candidate.check_valid()

    def test_malformed_inputs_and_constructor_bounds_fail_closed(self):
        torch, _Box, _StackedObservations = _deps()
        native = importlib.import_module('ai2.training.tensor_frame_stack')
        for n_envs in (0, 4097, True, 1.5):
            with self.subTest(n_envs=n_envs):
                with self.assertRaises(ValueError):
                    native.TensorFrameStack(n_envs, device="cuda")

        candidate = native.TensorFrameStack(8, device="cuda")
        good_obs = torch.zeros((8, 64), dtype=torch.float32, device="cuda")
        good_terminal = torch.zeros((8, 64), dtype=torch.float32, device="cuda")
        good_dones = torch.zeros((8,), dtype=torch.bool, device="cuda")
        with self.assertRaises(RuntimeError):
            candidate.update(good_obs, good_terminal, good_dones)
        with self.assertRaises(ValueError):
            candidate.reset(torch.zeros((8, 64), dtype=torch.float32))
        with self.assertRaises(ValueError):
            candidate.reset(torch.zeros((8, 64), dtype=torch.float64, device="cuda"))
        with self.assertRaises(ValueError):
            candidate.reset(torch.zeros((8, 128), dtype=torch.float32, device="cuda")[:, ::2])
        with self.assertRaises(ValueError):
            candidate.reset(torch.zeros((8, 64), dtype=torch.float32, device="cuda", requires_grad=True))

        candidate.reset(good_obs)
        with self.assertRaises(ValueError):
            candidate.update(good_obs, good_terminal, torch.zeros(8, device="cuda"))
        with self.assertRaises(ValueError):
            candidate.update(good_obs[:, :63], good_terminal, good_dones)

    def test_nonfinite_data_accumulates_until_explicit_check_valid_and_reset_clears(self):
        torch, _Box, _StackedObservations = _deps()
        native = importlib.import_module('ai2.training.tensor_frame_stack')
        candidate = native.TensorFrameStack(2, device="cuda")
        obs = torch.zeros((2, 64), dtype=torch.float32, device="cuda")
        obs[0, 0] = float("nan")
        candidate.reset(obs)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            candidate.check_valid()

        clean = torch.zeros((2, 64), dtype=torch.float32, device="cuda")
        candidate.reset(clean)
        candidate.check_valid()
        terminal = clean.clone()
        terminal[1, 0] = float("inf")
        candidate.update(clean, terminal, torch.zeros((2,), dtype=torch.bool, device="cuda"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            candidate.check_valid()


if __name__ == "__main__":
    unittest.main()
