"""Tests for the native navigation lanes environment (native_lane_vec_env.py).

CPU-backend tests always run (they compile the host lane library with cc and
need torch for the tensors).  The frozen-game tests need the frozen payload
and a Pymunk 6.2.1 runtime; the CUDA parity test needs nvcc and a CUDA
device.  Those are skipped, not passed, otherwise.
"""
from __future__ import annotations

from ai2.paths import source_path

import ctypes
import importlib
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))

FROZEN = Path(os.environ.get("AI2_FROZEN_SOURCE_ROOT") or "/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected")
OBS, STACK, HISTORY, ACT = 64, 256, 192, 10
TIME_LIMIT = 600
SANE_MAGNITUDE = 64.0


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency-light interpreters
        raise unittest.SkipTest(f"torch unavailable: {exc}") from exc
    return torch


def _compiler_or_skip() -> None:
    if shutil.which(os.environ.get("CC") or "cc") is None:
        raise unittest.SkipTest("no C compiler")


def _frozen_or_skip() -> None:
    for name in ("train.py", "control.yaml", "positions.json", "UTMIST-AI2-main/environment/environment.py"):
        if not (FROZEN / name).is_file():
            raise unittest.SkipTest(f"frozen payload missing: {FROZEN / name}")
    try:
        import pymunk
    except Exception as exc:
        raise unittest.SkipTest(f"pymunk unavailable: {exc}") from exc
    if not str(getattr(pymunk, "version", "")).startswith("6.2.1"):
        raise unittest.SkipTest(f"pymunk {getattr(pymunk, 'version', '?')} is not the frozen 6.2.1 runtime")


class LaneCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _compiler_or_skip()
        cls.torch = _torch()
        cls.tmp = Path(tempfile.mkdtemp(prefix="ai2-lanes-"))
        cls.m = importlib.import_module('ai2.simulation.native_lane_vec_env')
        cls.episode = importlib.import_module('ai2.simulation.native_navigation_episode')
        cls.library = cls.m.compile_lanes_cpu(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def make_env(self, lanes: int, seed: int = 0, **kwargs):
        env = self.m.build_cpu_env(lanes, build_dir=self.tmp, seed=seed, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_error_labels_match_native_codes(self):
        header = (ROOT / "native" / "navigation_episode.h").read_text()
        codes = {int(code): name.lower() for name, code in re.findall(r"^#define AI2N_ERR_(\w+) (\d+)$", header, re.MULTILINE)}
        self.assertEqual(self.m.ERROR_NAMES, codes)

    def test_reset_reinitializes_poisoned_lane_semantic_state(self):
        env, reference = self.make_env(2, seed=45), self.make_env(2, seed=45)
        env.reset_tensor()
        for lane in range(2):
            state = self.m.Lane()
            ctypes.memset(ctypes.byref(state), 0xA5, ctypes.sizeof(state))
            env.backend.set_lane_state(lane, state)
        got, expected = env.reset_tensor(), reference.reset_tensor()
        self.assertTrue(self.torch.equal(got, expected))
        for _ in range(4):
            actions = self.torch.zeros((2, ACT), dtype=self.torch.float32)
            a, b = env.step_tensor(actions), reference.step_tensor(actions)
            for field in ("observations", "rewards", "terminated", "truncated", "terminal_observations"):
                self.assertTrue(self.torch.equal(getattr(a, field), getattr(b, field)))

    def test_struct_layouts_match_library(self):
        lib = ctypes.CDLL(str(self.library))
        lib.ai2n_lanes_cpu_lane_size.restype = ctypes.c_int
        lib.ai2n_lanes_cpu_record_size.restype = ctypes.c_int
        self.assertEqual(lib.ai2n_lanes_cpu_lane_size(), ctypes.sizeof(self.m.Lane))
        self.assertEqual(lib.ai2n_lanes_cpu_record_size(), ctypes.sizeof(self.m.EpisodeRecord))
        self.assertEqual(ctypes.sizeof(self.m.EpisodeRecord), 48)
        self.assertEqual(ctypes.sizeof(self.m.Lane), ctypes.sizeof(self.episode.Episode) + ctypes.sizeof(self.episode.Rng) + 8 + 6 * 4)
        # The lane library also carries the episode reference exports (config finalize, platform table, offsets).
        ref = self.m.make_reference(self.library)
        self.assertEqual(ref.positions_count, 967)

    def test_frozen_spaces_have_the_training_shapes(self):
        obs_space, act_space = self.m.frozen_spaces()
        self.assertEqual(tuple(obs_space.shape), (STACK,))
        self.assertEqual(tuple(act_space.shape), (ACT,))
        self.assertEqual(np.dtype(obs_space.dtype), np.float32)
        self.assertTrue((act_space.low == 0).all() and (act_space.high == 1).all())

    def test_reset_layout_and_seed_determinism(self):
        env = self.make_env(6, seed=45)
        stack = env.reset_tensor()
        self.assertEqual(tuple(stack.shape), (6, STACK))
        self.assertEqual(stack.dtype, self.torch.float32)
        self.assertTrue(bool((stack[:, :HISTORY] == 0).all()))
        frame = stack[:, HISTORY:].numpy()
        # The game's declared Box is advisory (positions reach about 6.5 where it says 1), so check sanity, not bounds.
        self.assertTrue(np.isfinite(frame).all() and (np.abs(frame) <= SANE_MAGNITUDE).all())
        # Lanes are seeded independently, so their reset targets differ; the same seeds reproduce the same frames.
        self.assertGreater(len({frame[i].tobytes() for i in range(6)}), 1)
        again = self.make_env(6, seed=45).reset_tensor().numpy()
        self.assertTrue(np.array_equal(stack.numpy(), again))
        other = self.make_env(6, seed=46).reset_tensor().numpy()
        self.assertFalse(np.array_equal(stack.numpy(), other))
        # Explicit seeds through the TensorVecEnv seed contract.
        seeded = self.make_env(6, seed=0)
        seeded.seed([45, 46, 47, 48, 49, 50])
        self.assertTrue(np.array_equal(seeded.reset_tensor().numpy(), again))

    def test_step_semantics_and_episode_records(self):
        torch = self.torch
        lanes, steps = 8, 1400
        env = self.make_env(lanes, seed=7)
        rng = np.random.default_rng(3)
        prev = env.reset_tensor().clone()
        lane_len = np.zeros(lanes, dtype=np.int64)
        lane_ret = np.zeros(lanes, dtype=np.float64)
        events = []
        for _ in range(steps):
            actions = torch.from_numpy((rng.random((lanes, ACT)) < 0.35).astype(np.float32))
            step = env.step_tensor(actions)
            obs, terminal = step.observations, step.terminal_observations
            self.assertEqual(tuple(obs.shape), (lanes, STACK))
            self.assertEqual(tuple(terminal.shape), (lanes, STACK))
            for tensor in (obs, terminal, step.rewards):
                self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(step.terminated.dtype, torch.bool)
            self.assertEqual(step.truncated.dtype, torch.bool)
            self.assertFalse(bool((step.terminated & step.truncated).any()))
            # VecFrameStack: terminal = previous three frames + final frame; live stack = roll, zero done lanes, append.
            self.assertTrue(torch.equal(terminal[:, :HISTORY], prev[:, OBS:]))
            dones = (step.terminated | step.truncated).numpy()
            for i in range(lanes):
                if dones[i]:
                    self.assertTrue(bool((obs[i, :HISTORY] == 0).all()))
                    self.assertFalse(torch.equal(obs[i, HISTORY:], terminal[i, HISTORY:]))
                else:
                    self.assertTrue(torch.equal(obs[i, :HISTORY], prev[i, OBS:]))
                    self.assertTrue(torch.equal(obs[i, HISTORY:], terminal[i, HISTORY:]))
            frame = obs.numpy()
            self.assertTrue(np.isfinite(frame).all() and (np.abs(frame) <= SANE_MAGNITUDE).all())
            rewards = step.rewards.numpy()
            self.assertTrue(np.isfinite(rewards).all())
            lane_len += 1
            lane_ret += rewards.astype(np.float64)
            for i in np.flatnonzero(dones):
                events.append((i, int(lane_len[i]), float(lane_ret[i]), bool(step.terminated[i]), bool(step.truncated[i])))
                lane_len[i] = 0
                lane_ret[i] = 0.0
            prev = obs.clone()
        infos = env.drain_episode_infos()
        self.assertGreater(len(events), 0)
        self.assertEqual(len(infos), len(events))
        for info, (lane, length, ret, terminated, truncated) in zip(infos, events):
            nav = info["navigation"]
            self.assertEqual(nav["lane"], lane)
            self.assertEqual(info["episode"]["l"], length)
            self.assertLessEqual(abs(info["episode"]["r"] - ret), 1e-3)
            self.assertEqual(nav["terminated"], terminated)
            if truncated:
                self.assertTrue(nav["horizon"] and nav["truncated"] and length == TIME_LIMIT)
            else:
                self.assertTrue(nav["fell"] or nav["success"])
            self.assertLessEqual(length, TIME_LIMIT)
        self.assertEqual(env.drain_episode_infos(), [])

    def test_lane_matches_reference_episode_stepping(self):
        """The lane adds nothing to the tick: same seed and stream give the reference's outputs, with autoreset."""
        torch = self.torch
        env = self.make_env(1, seed=11)
        ref = self.m.make_reference(self.library)
        cfg = self.m.make_config(ref)
        ep, rng = self.episode.Episode(), self.episode.Rng()
        ref.lib.ai2n_rng_seed_export(ctypes.byref(rng), 11, 0)
        self.assertEqual(ref.reset(ep, cfg, rng), 0)
        stack = env.reset_tensor()
        self.assertTrue(np.array_equal(stack[0, HISTORY:].numpy(), np.asarray(ep.obs[:OBS], dtype=np.float32)))
        rng_np = np.random.default_rng(5)
        resets = 0
        for _ in range(900):
            a = (rng_np.random(ACT) < 0.3).astype(np.float32)
            step = env.step_tensor(torch.from_numpy(a[None].copy()))
            rc, out = ref.step(ep, [float(x) for x in a], rng)
            self.assertEqual(rc, 0)
            self.assertTrue(np.array_equal(step.terminal_observations[0, HISTORY:].numpy(), np.asarray(out.obs[:OBS], dtype=np.float32)))
            self.assertEqual(float(step.rewards[0]), float(np.float32(out.reward)))
            self.assertEqual(bool(step.terminated[0]), bool(out.terminated))
            self.assertEqual(bool(step.truncated[0]), bool(out.truncated and not out.terminated))
            if out.terminated or out.truncated:
                self.assertEqual(ref.reset(ep, cfg, rng), 0)
                resets += 1
            self.assertTrue(np.array_equal(step.observations[0, HISTORY:].numpy(), np.asarray(ep.obs[:OBS], dtype=np.float32)))
        self.assertGreater(resets, 0)

    def test_invalid_action_freezes_the_lane_and_raises(self):
        torch = self.torch
        env = self.make_env(3, seed=1)
        env.reset_tensor()
        actions = torch.zeros((3, ACT), dtype=torch.float32)
        actions[1, 0] = float("nan")
        with self.assertRaises(self.m.LaneError) as caught:
            env.step_tensor(actions)
        self.assertEqual(caught.exception.code, 9)
        self.assertEqual(caught.exception.lane, 1)
        with self.assertRaises(self.m.LaneError):
            env.step_tensor(torch.zeros((3, ACT), dtype=torch.float32))
        with self.assertRaises(ValueError):
            env.step_tensor(torch.zeros((3, ACT), dtype=torch.float64))
        with self.assertRaises(ValueError):
            env.step_tensor(torch.zeros((2, ACT), dtype=torch.float32))


class FrozenGameLaneTests(unittest.TestCase):
    """The lane pipeline against the real SB3 stack (Monitor, DummyVecEnv autoreset, VecFrameStack) on the frozen game."""

    def setUp(self):
        # Frozen imports change process state before construction can fail.
        # Independent cleanups still run if construction or env.close raises.
        self.addCleanup(os.chdir, Path.cwd())
        self.addCleanup(sys.path.__setitem__, slice(None), list(sys.path))

    @classmethod
    def setUpClass(cls):
        _compiler_or_skip()
        _frozen_or_skip()
        cls.torch = _torch()
        cls.tmp = Path(tempfile.mkdtemp(prefix="ai2-lanes-frozen-"))
        cls.m = importlib.import_module('ai2.simulation.native_lane_vec_env')
        cls.library = cls.m.compile_lanes_cpu(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _build_game(self, seed: int):
        replay = importlib.import_module('ai2.evaluation.benchmark_replay')
        validator = importlib.import_module('tools.validation.validate_native_navigation')
        env, _params, _phase = replay.build_navigation_env(FROZEN, FROZEN / "control.yaml", source_path(FROZEN, "train.py"), seed, frame_stack=4)
        wrapper = validator.find_frozen_opponent_wrapper(env)
        return env, wrapper

    def test_frozen_spaces_match_the_frozen_game(self):
        original_cwd = Path.cwd()
        env, _wrapper = self._build_game(45)
        try:
            obs_space, act_space = self.m.frozen_spaces()
            self.assertEqual(obs_space, env.observation_space)
            self.assertEqual(act_space, env.action_space)
        finally:
            env.close()
            os.chdir(original_cwd)

    def _run_pipeline(self, seed: int, controller: str, steps: int, min_episodes: int) -> dict:
        torch = self.torch
        capture_mod = importlib.import_module('ai2.evaluation.navigation_episode_capture')
        validator = importlib.import_module('tools.validation.validate_native_navigation')
        original_cwd = Path.cwd()
        recorder = capture_mod.DrawRecorder()
        env = lane_env = None
        try:
            env, wrapper = self._build_game(seed)
            wrapper_cfg = capture_mod.wrapper_config(wrapper)
            raw_env = capture_mod.find_raw_env(wrapper)
            bias = float(capture_mod.SpaceRecorder(raw_env).space_config()["collision_bias"])
            lane_env = self.m.build_cpu_env(1, build_dir=self.tmp, wrapper_cfg=wrapper_cfg, observation_space=env.observation_space,
                                            action_space=env.action_space, collision_bias=bias)
            # Training builds the config without a capture: the runtime default must equal the live space's bias.
            default_cfg = self.m.make_config(self.m.make_reference(self.library), wrapper_cfg)
            self.assertEqual(default_cfg.collision_bias, bias)
            recorder.install()
            recorder.take()
            obs0 = env.reset()
            reset_draws = recorder.take()
            stack = lane_env.replay_reset_tensor(reset_draws)
            self.assertTrue(np.array_equal(stack[0].numpy(), np.asarray(obs0[0], dtype=np.float32)))
            if controller == "adaptive":
                actions = validator._adaptive_actions(wrapper, steps)
            elif controller == "scripted":
                actions = validator._scripted_actions(wrapper, steps)
            else:
                actions = capture_mod.random_actions(wrapper.action_space, steps, seed)
            expected_episodes = []
            ticks = 0
            for _label, action in actions:
                a = np.ascontiguousarray(np.asarray(action, dtype=np.float32).reshape(1, ACT))
                recorder.take()
                obs, rews, dones, infos = env.step(a)
                draws = recorder.take()
                step = lane_env.replay_step_tensor(torch.from_numpy(a.copy()), draws)
                ticks += 1
                self.assertTrue(np.array_equal(step.observations[0].numpy(), np.asarray(obs[0], dtype=np.float32)), f"observation differs at tick {ticks}")
                self.assertEqual(float(step.rewards[0]), float(np.float32(rews[0])), f"reward differs at tick {ticks}")
                self.assertEqual(bool(step.terminated[0] | step.truncated[0]), bool(dones[0]), f"done differs at tick {ticks}")
                self.assertEqual(bool(step.truncated[0]), bool(infos[0].get("TimeLimit.truncated", False)), f"truncation differs at tick {ticks}")
                if dones[0]:
                    terminal = np.asarray(infos[0]["terminal_observation"], dtype=np.float32)
                    self.assertTrue(np.array_equal(step.terminal_observations[0].numpy(), terminal), f"terminal observation differs at tick {ticks}")
                    expected_episodes.append(dict(infos[0]["episode"]))
            drained = lane_env.drain_episode_infos()
            self.assertGreaterEqual(len(expected_episodes), min_episodes)
            self.assertEqual(len(drained), len(expected_episodes))
            for got, exp in zip(drained, expected_episodes):
                self.assertEqual(got["episode"]["l"], int(exp["l"]))
                self.assertLessEqual(abs(got["episode"]["r"] - float(exp["r"])), 1e-6)
            return {"ticks": ticks, "episodes": len(expected_episodes)}
        finally:
            recorder.restore()
            if lane_env is not None:
                lane_env.close()
            if env is not None:
                env.close()
            os.chdir(original_cwd)

    def test_sb3_pipeline_exact_adaptive(self):
        result = self._run_pipeline(45, "adaptive", 420, min_episodes=1)
        self.assertEqual(result["ticks"], 420)

    def test_sb3_pipeline_exact_random(self):
        result = self._run_pipeline(46, "random", 700, min_episodes=1)
        self.assertEqual(result["ticks"], 700)


class FrozenGameTestIsolationTests(unittest.TestCase):
    def test_build_failure_restores_working_directory_and_import_path(self):
        original_cwd, original_path = Path.cwd(), list(sys.path)
        with tempfile.TemporaryDirectory(prefix="ai2-lanes-isolation-") as tmp:
            def fail_build(_seed):
                os.chdir(tmp)
                sys.path.insert(0, tmp)
                raise RuntimeError("injected build failure")

            case = FrozenGameLaneTests("test_frozen_spaces_match_the_frozen_game")
            result = unittest.TestResult()
            try:
                # TestCase.run exercises per-test setup/cleanup without the optional
                # game dependencies from FrozenGameLaneTests.setUpClass.
                with patch.object(case, "_build_game", side_effect=fail_build):
                    case.run(result)
                self.assertEqual(len(result.errors), 1)
                self.assertIn("injected build failure", result.errors[0][1])
                with self.subTest(state="cwd"):
                    self.assertEqual(Path.cwd(), original_cwd)
                with self.subTest(state="sys.path"):
                    self.assertEqual(sys.path, original_path)
            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path


class CudaLaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _compiler_or_skip()
        cls.torch = _torch()
        if not cls.torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        if shutil.which("nvcc") is None:
            raise unittest.SkipTest("nvcc unavailable")
        cls.tmp = Path(tempfile.mkdtemp(prefix="ai2-lanes-cuda-"))
        cls.m = importlib.import_module('ai2.simulation.native_lane_vec_env')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cuda_lanes_match_cpu_lanes(self):
        torch = self.torch
        lanes, steps = 64, 800
        cpu = self.m.build_cpu_env(lanes, build_dir=self.tmp, seed=45, contract="off")
        gpu = self.m.build_cuda_env(lanes, build_dir=self.tmp, seed=45, device="cuda")
        self.addCleanup(cpu.close)
        self.addCleanup(gpu.close)
        rng = np.random.default_rng(9)
        cpu_stack = cpu.reset_tensor().numpy()
        gpu_stack = gpu.reset_tensor().cpu().numpy()
        self.assertTrue(np.array_equal(cpu_stack, gpu_stack))
        reward_bits_differ = 0
        for t in range(steps):
            a = (rng.random((lanes, ACT)) < 0.35).astype(np.float32)
            cs = cpu.step_tensor(torch.from_numpy(a.copy()))
            gs = gpu.step_tensor(torch.from_numpy(a.copy()).to("cuda"))
            self.assertTrue(np.array_equal(cs.observations.numpy(), gs.observations.cpu().numpy()), f"observations differ at step {t}")
            self.assertTrue(np.array_equal(cs.terminal_observations.numpy(), gs.terminal_observations.cpu().numpy()), f"terminal observations differ at step {t}")
            self.assertTrue(np.array_equal(cs.terminated.numpy(), gs.terminated.cpu().numpy()), f"terminated differs at step {t}")
            self.assertTrue(np.array_equal(cs.truncated.numpy(), gs.truncated.cpu().numpy()), f"truncated differs at step {t}")
            cr, gr = cs.rewards.numpy(), gs.rewards.cpu().numpy()
            if not np.array_equal(cr, gr):
                reward_bits_differ += 1
                self.assertTrue(np.allclose(cr, gr, rtol=0, atol=1e-6), f"rewards differ beyond float32 rounding at step {t}")
        # Device squares are correctly rounded while the host keeps libm pow: last-bit reward differences are possible but rare.
        self.assertLessEqual(reward_bits_differ, steps // 100)
        cpu_infos = cpu.drain_episode_infos()
        gpu_infos = gpu.drain_episode_infos()
        key = lambda info: (info["navigation"]["lane"], info["navigation"]["episode_index"])
        cpu_sorted, gpu_sorted = sorted(cpu_infos, key=key), sorted(gpu_infos, key=key)
        self.assertGreater(len(cpu_sorted), 0)
        self.assertEqual([(key(i), i["episode"]["l"], i["navigation"]) for i in cpu_sorted], [(key(i), i["episode"]["l"], i["navigation"]) for i in gpu_sorted])
        for c, g in zip(cpu_sorted, gpu_sorted):
            self.assertLessEqual(abs(c["episode"]["r"] - g["episode"]["r"]), 1e-6)


if __name__ == "__main__":
    unittest.main()
