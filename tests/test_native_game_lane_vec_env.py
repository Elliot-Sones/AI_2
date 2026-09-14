"""Policy-free tests for full-game tensor frame stacking and lane bookkeeping."""

import unittest
from contextlib import nullcontext
from unittest.mock import patch
from pathlib import Path
import subprocess
import tempfile
import torch
from native_game_lane_vec_env import GameLaneBackend, build_game_env, compile_lanes


class GameEnvTests(unittest.TestCase):
    def test_two_player_slots_per_game_and_four_frame_stack(self):
        env = build_game_env(2, device="cpu", seed=7)
        try:
            obs = env.reset_tensor().clone()
            self.assertEqual(obs.shape, (4, 256))
            self.assertEqual(torch.count_nonzero(obs[:, :192]).item(), 0)
            act = torch.zeros((4, 10))
            act[:, 7] = 1
            result = env.step_tensor(act)
            torch.testing.assert_close(
                result.observations[:, 128:192], obs[:, 192:256], rtol=0, atol=0
            )
            self.assertEqual(result.rewards.shape, (4,))
            self.assertEqual(result.terminated.tolist(), [False] * 4)
            env.drain_episode_infos()
            self.assertEqual(env.backend.valid_steps.tolist(), [1, 1])
        finally:
            env.close()

    def test_action_validation_and_closed_context(self):
        env = build_game_env(1, device="cpu")
        try:
            env.reset_tensor()
            with self.assertRaises(ValueError):
                env.step_tensor(torch.zeros((1, 10)))
            with self.assertRaises(ValueError):
                env.step_tensor(torch.zeros((2, 10), dtype=torch.float64))
        finally:
            env.close()
        env.close()
        with self.assertRaises(RuntimeError):
            env.reset_tensor()

    def test_invalid_game_counts_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                build_game_env(value, device="cpu")

    def test_destroy_error_invalidates_already_freed_native_handle(self):
        backend = GameLaneBackend(1, 'cpu')
        destroy = backend.destroy
        calls = []

        def destroy_then_report_cuda_error(handle):
            calls.append(handle)
            self.assertEqual(destroy(handle), 0)
            return 303  # CUDA destroy frees before reporting synchronization errors.

        backend.destroy = destroy_then_report_cuda_error
        with self.assertRaisesRegex(RuntimeError, 'destroy: error 303'):
            backend.close()
        self.assertIsNone(backend.handle)
        backend.close()
        self.assertEqual(len(calls), 1)

    def test_successful_reset_clears_rollout_error_cache(self):
        env = build_game_env(1, device='cpu')
        try:
            env.reset_tensor()
            with self.assertRaises(RuntimeError):
                env.step_tensor(torch.full((2, 10), float('nan')))
            with self.assertRaisesRegex(RuntimeError, 'invalid rollout'):
                env.drain_episode_infos()
            self.assertEqual(env.backend.errors.tolist(), [202])
            env.reset_tensor()
            self.assertEqual(env.backend.errors.tolist(), [0])
            env.step_tensor(torch.zeros((2, 10)))
        finally:
            env.close()

    def test_compile_cache_hash_covers_linked_navigation_source(self):
        original_read = Path.read_bytes

        def changed_navigation_bytes(path):
            content = original_read(path)
            return content + b'\n/* hash-only regression fixture */' if path.name == 'navigation_episode.c' else content

        with tempfile.TemporaryDirectory(prefix='game-hash-source-') as directory:
            before = compile_lanes('cpu', Path(directory))
            with patch.object(Path, 'read_bytes', changed_navigation_bytes):
                after = compile_lanes('cpu', Path(directory))
            self.assertNotEqual(before.name, after.name)

    def test_compile_cache_hash_covers_compiler_identity(self):
        original_run = subprocess.run

        def changed_compiler_identity(command, *args, **kwargs):
            if '--version' in command:
                return subprocess.CompletedProcess(command, 0, stdout='different test compiler identity', stderr='')
            return original_run(command, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix='game-hash-compiler-') as directory:
            before = compile_lanes('cpu', Path(directory))
            with patch('native_game_lane_vec_env.subprocess.run', changed_compiler_identity):
                after = compile_lanes('cpu', Path(directory))
            self.assertNotEqual(before.name, after.name)

    def test_cuda_launch_records_tensor_lifetimes_on_collector_stream(self):
        # Mock only the CUDA launch/stream boundaries; use real Tensor shape,
        # dtype, contiguity checks and the real backend step/copy logic.
        backend = GameLaneBackend.__new__(GameLaneBackend)
        backend.torch, backend.kind, backend.device = torch, 'cuda', torch.device('cpu')
        backend.games, backend.lanes = 1, 2
        backend._terminated = torch.zeros(1, dtype=torch.uint8)
        backend._truncated = torch.zeros(1, dtype=torch.uint8)
        backend._guard = nullcontext
        backend._args = lambda *args: args
        backend.step_fn = lambda *args: 0
        arrays = [torch.zeros((2, 10)), torch.zeros((2, 64)), torch.zeros((2, 64)), torch.zeros(2),
                  torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)]
        stream, recorded = object(), []

        def record(tensor, selected):
            self.assertIs(selected, stream)
            recorded.append(id(tensor))

        with patch.object(torch.cuda, 'current_stream', return_value=stream), \
                patch.object(torch.Tensor, 'record_stream', record):
            backend.step(*arrays)
        self.assertEqual(set(recorded), {id(t) for t in arrays + [backend._terminated, backend._truncated]})


if __name__ == "__main__":
    unittest.main()
