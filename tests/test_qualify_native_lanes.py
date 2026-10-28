"""Fail-closed receipt reductions plus an actual-CUDA asynchronous smoke test."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import importlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fixture_block():
    """Two steps: a terminal autoreset in lane 0 and a live lane 1."""
    reset = np.zeros((2, 256), dtype=np.float32)
    reset[:, -64:] = 1
    obs = np.zeros((2, 2, 256), dtype=np.float32)
    terminal = np.zeros_like(obs)
    terminal[0, :, :192] = reset[:, 64:]
    terminal[0, :, -64:] = 2
    obs[0] = terminal[0]
    obs[0, 0, :192] = 0
    obs[0, 0, -64:] = 3
    terminal[1, :, :192] = obs[0, :, 64:]
    terminal[1, :, -64:] = 4
    obs[1] = terminal[1]
    return reset, {
        "observations": obs,
        "terminal_observations": terminal,
        "rewards": np.array([[0.5, 0.25], [0.125, -0.5]], dtype=np.float32),
        "terminated": np.array([[True, False], [False, False]]),
        "truncated": np.zeros((2, 2), dtype=bool),
    }


def fixture_info(lane=0, index=0, reward=0.5):
    return {"episode": {"r": reward, "l": 1, "t": 123.0}, "navigation": {
        "lane": lane, "episode_index": index, "success": False, "fell": True,
        "targets": 0, "horizon": False, "terminated": True, "truncated": False}}


class QualificationHelperTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("tools.validation.qualify_native_lanes"), "qualification runner must exist")
        self.q = importlib.import_module('tools.validation.qualify_native_lanes')

    def test_explicit_phase_reaches_both_backends_and_receipt(self):
        import torch
        import ai2.simulation.native_lane_vec_env as native

        self.assertIn("wrapper_cfg", inspect.signature(self.q.run_case).parameters)
        config = dict(native.PHASE_0B, min_dist=1.5, max_dist=3.0, time_limit=4)
        fake_cuda = SimpleNamespace(cuda=SimpleNamespace(default_stream=lambda: None,
                                    stream=lambda _: nullcontext()), from_numpy=torch.from_numpy)
        with tempfile.TemporaryDirectory() as tmp:
            cpu = native.build_cpu_env(2, build_dir=Path(tmp), wrapper_cfg=config, contract="off")
            gpu = native.build_cpu_env(2, build_dir=Path(tmp), wrapper_cfg=config, contract="off")
            with patch.object(self.q, "_require_cuda", return_value=fake_cuda), \
                    patch.object(native, "build_cpu_env", return_value=cpu) as cpu_builder, \
                    patch.object(native, "build_cuda_env", return_value=gpu) as gpu_builder:
                result = self.q.run_case(lanes=2, steps=8, seed=45, build_dir=Path(tmp), wrapper_cfg=config)
            self.assertIs(cpu_builder.call_args.kwargs["wrapper_cfg"], config)
            self.assertIs(gpu_builder.call_args.kwargs["wrapper_cfg"], config)
            self.assertEqual(result["wrapper_config"], config)
            self.assertEqual(result["effective_config"]["min_dist"], 1.5)
            self.assertEqual(result["effective_config"]["max_dist"], 3.0)
            self.assertEqual(result["effective_config"]["time_limit_steps"], 4)
            self.assertEqual(result["counts"]["episode_records"], 4)

    def test_default_configuration_is_recorded_and_backend_mismatch_fails(self):
        import ai2.simulation.native_lane_vec_env as native

        self.assertIn("wrapper_cfg", inspect.signature(self.q.run_case).parameters)
        fake_cuda = SimpleNamespace(cuda=SimpleNamespace(default_stream=lambda: None,
                                    stream=lambda _: nullcontext()))
        with tempfile.TemporaryDirectory() as tmp:
            cpu = native.build_cpu_env(2, build_dir=Path(tmp), contract="off")
            gpu = native.build_cpu_env(2, build_dir=Path(tmp), contract="off",
                                       wrapper_cfg=dict(native.PHASE_0B, min_dist=1.5))
            with patch.object(self.q, "_require_cuda", return_value=fake_cuda), \
                    patch.object(native, "build_cpu_env", return_value=cpu), \
                    patch.object(native, "build_cuda_env", return_value=gpu), \
                    self.assertRaisesRegex(self.q.QualificationError, "config"):
                self.q.run_case(lanes=2, steps=8, seed=45, build_dir=Path(tmp))

    def test_block_reports_counts_and_exact_maxima(self):
        reset, cpu = fixture_block()
        actual = copy.deepcopy(cpu)
        actual["rewards"][0, 0] += np.float32(5e-7)
        result = self.q.compare_block(cpu, actual, reset, reset)
        self.assertEqual(result["counts"]["transitions"], 4)
        self.assertEqual(result["counts"]["terminated"], 1)
        self.assertEqual(result["counts"]["truncated"], 0)
        self.assertEqual(result["counts"]["reward_values_different"], 1)
        self.assertEqual(result["max_differences"]["observations"], 0)
        self.assertGreater(result["max_differences"]["rewards"], 0)

    def test_exact_observations_and_flags_fail_closed(self):
        for field in ("observations", "terminal_observations", "terminated", "truncated"):
            with self.subTest(field=field):
                reset, cpu = fixture_block()
                gpu = copy.deepcopy(cpu)
                gpu[field].flat[0] = 1 if gpu[field].flat[0] == 0 else 0
                with self.assertRaisesRegex(self.q.QualificationError, field):
                    self.q.compare_block(cpu, gpu, reset, reset)

    def test_reward_tolerance_nonfinite_shape_and_dtype_fail_closed(self):
        for value in (float("nan"), float("inf"), 0.500002):
            reset, cpu = fixture_block()
            gpu = copy.deepcopy(cpu)
            gpu["rewards"][0, 0] = value
            with self.subTest(value=value), self.assertRaises(self.q.QualificationError):
                self.q.compare_block(cpu, gpu, reset, reset)
        for change in (lambda arr: arr[:, :1], lambda arr: arr.astype(np.float64)):
            reset, cpu = fixture_block()
            gpu = copy.deepcopy(cpu)
            gpu["rewards"] = change(gpu["rewards"])
            with self.assertRaises(self.q.QualificationError):
                self.q.compare_block(cpu, gpu, reset, reset)

    def test_both_backends_with_same_broken_stack_are_rejected(self):
        reset, cpu = fixture_block()
        cpu["observations"][0, 0, 0] = 7
        with self.assertRaisesRegex(self.q.QualificationError, "history"):
            self.q.compare_block(cpu, copy.deepcopy(cpu), reset, reset)

    def test_reset_requires_zero_history_finite_and_exact_values(self):
        reset, _ = fixture_block()
        self.assertEqual(self.q.compare_reset(reset, reset.copy())["max_differences"]["reset"], 0)
        broken = reset.copy()
        broken[0, 0] = 1
        with self.assertRaisesRegex(self.q.QualificationError, "history"):
            self.q.compare_reset(broken, broken.copy())
        broken = reset.copy()
        broken[0, -1] = np.nan
        with self.assertRaises(self.q.QualificationError):
            self.q.compare_reset(reset, broken)

    def test_records_ignore_arrival_order_and_wall_time_only(self):
        cpu = [fixture_info(0), fixture_info(1)]
        gpu = copy.deepcopy(list(reversed(cpu)))
        gpu[0]["episode"]["t"] = 999
        gpu[0]["episode"]["r"] += 5e-7
        result = self.q.compare_episode_infos(cpu, gpu)
        self.assertEqual(result["counts"]["episode_records"], 2)
        self.assertGreater(result["max_differences"]["episode_returns"], 0)
        self.assertEqual(result["counts"]["episode_returns_different"], 1)

    def test_missing_duplicate_nonfinite_or_mismatched_records_fail_closed(self):
        cpu = [fixture_info()]
        variants = [[], [fixture_info(), fixture_info()], [fixture_info(reward=float("nan"))],
                    [fixture_info(reward=0.500002)], [fixture_info(index=1)]]
        broken = fixture_info()
        broken["navigation"]["targets"] = 1
        variants.append([broken])
        for gpu in variants:
            with self.subTest(gpu=gpu), self.assertRaises(self.q.QualificationError):
                self.q.compare_episode_infos(cpu, gpu)

    def test_record_count_must_match_terminal_events(self):
        reset, cpu = fixture_block()
        result = self.q.compare_block(cpu, copy.deepcopy(cpu), reset, reset)
        with self.assertRaisesRegex(self.q.QualificationError, "record count"):
            self.q.require_episode_count(result, [])
        self.q.require_episode_count(result, [fixture_info()])

    def test_source_manifest_requires_every_declared_source(self):
        manifest = self.q.source_manifest()
        self.assertEqual(manifest["source_layout_version"], 2)
        self.assertEqual(manifest["native_sources"], importlib.import_module('ai2.simulation.native_lane_vec_env').source_hashes())
        self.assertIn("native_lane_vec_env.py", manifest["source_hashes"])
        self.assertIn("native_lane_cpu_adapter.py", manifest["source_hashes"])
        self.assertIn("native/navigation_spaces.json", manifest["source_hashes"])
        for name in ("ai2/paths.py", "ai2/__init__.py", "ai2/simulation/__init__.py",
                     "ai2/training/__init__.py", "tools/validation/__init__.py",
                     "ai2/training/curriculum_ppo.py", "curriculum_ppo.py"):
            self.assertIn(name, manifest["source_hashes"])
        self.assertNotEqual(manifest["source_hashes"]["curriculum_ppo.py"],
                            manifest["source_hashes"]["ai2/training/curriculum_ppo.py"])
        with patch.object(self.q, "CANDIDATE_FILES", ("missing-source-for-test.py",)):
            with self.assertRaises(FileNotFoundError):
                self.q.source_manifest()
        lanes = importlib.import_module('ai2.simulation.native_lane_vec_env')
        with patch.object(lanes, "LANE_SOURCE_FILES", (*lanes.LANE_SOURCE_FILES, ROOT / "missing-native-source.c")):
            with self.assertRaises(FileNotFoundError):
                self.q.source_manifest()

    def test_missing_cuda_emits_failed_receipt_never_a_skip_or_pass(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "parity.json"
            with patch.object(torch.cuda, "is_available", return_value=False):
                code = self.q.main(["--output", str(receipt), "--lanes", "2", "--steps", "650"])
            result = json.loads(receipt.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["exit_code"], 1)
            self.assertEqual(result["skipped"], 0)
            self.assertEqual(result["schema"], "ai2.native_lanes.parity.v1")
            self.assertIn("CUDA", result["error"])

    def test_queue_captures_mutable_outputs_without_host_reads(self):
        import torch
        from types import SimpleNamespace

        class ReusedOutputEnv:
            def __init__(self):
                self.output = SimpleNamespace(**{name: torch.zeros((2, 1)) for name in self_fields})

            def step_tensor(self, action):
                for name in self_fields:
                    getattr(self.output, name).copy_(action)
                return self.output

        self_fields = self.q.FIELDS
        actions = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1)
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("per-step host copy")), \
                patch.object(torch.Tensor, "numpy", side_effect=AssertionError("per-step host read")), \
                patch.object(torch.Tensor, "item", side_effect=AssertionError("per-step scalar read")), \
                patch.object(torch.cuda, "synchronize", side_effect=AssertionError("per-step synchronize")):
            block = self.q._queue_block(ReusedOutputEnv(), actions)
        for value in block.values():
            self.assertTrue(torch.equal(value, actions))

    def test_real_cpu_blocks_validate_autoreset_and_zero_based_episode_indices(self):
        import torch
        import ai2.simulation.native_lane_vec_env as native

        if shutil.which("cc") is None:
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(native.PHASE_0B, time_limit=32)
            env = native.build_cpu_env(2, build_dir=Path(tmp), seed=45, contract="off", wrapper_cfg=config, record_capacity=128)
            self.addCleanup(env.close)
            previous = env.reset_tensor().clone().numpy()
            result = self.q._host_block(self.q._queue_block(env, torch.zeros((100, 2, 10))))
            metrics = self.q.compare_block(result, copy.deepcopy(result), previous, previous.copy())
            infos = env.drain_episode_infos()
            self.q.require_episode_count(metrics, infos)
            records = self.q.compare_episode_infos(infos, copy.deepcopy(infos))
            self.assertGreater(records["counts"]["episode_records"], 0)
            self.assertIn(0, [info["navigation"]["episode_index"] for info in infos])


class CudaQualificationTests(unittest.TestCase):
    def test_real_cuda_default_and_nondefault_streams(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("actual CUDA unavailable")
        if shutil.which("nvcc") is None:
            self.skipTest("nvcc unavailable")
        q = importlib.import_module('tools.validation.qualify_native_lanes')
        with tempfile.TemporaryDirectory() as tmp:
            for stream in ("default", "nondefault"):
                result = q.run_case(lanes=4, steps=650, seed=45, build_dir=Path(tmp), block_size=128, stream=stream)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["counts"]["transitions"], 2600)
                self.assertGreater(result["counts"]["episode_records"], 0)
                self.assertEqual(result["counts"]["rollout_blocks"], 6)
                self.assertEqual(result["counts"]["per_step_host_transfers"], 0)
            self.assertEqual(q.check_async_error(build_dir=Path(tmp))["observed_error_code"], 9)


if __name__ == "__main__":
    unittest.main()
