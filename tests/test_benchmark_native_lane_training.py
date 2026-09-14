"""Tests for the native-lane training driver (benchmark_native_lane_training.py).

These never touch CUDA or the game: the lane builder and the tensor adapter
are replaced by fakes so the wiring, guards and provenance can be checked on
any machine.
"""
from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))


class FakeJobCallback:
    """The frozen callback's host-info writer and observable lifecycle behavior."""

    def __init__(self, output_dir):
        self.episodes_path = Path(output_dir) / "episodes.jsonl"
        self.model = types.SimpleNamespace(num_timesteps=1000, n_envs=2)
        self.num_timesteps = self.model.num_timesteps
        self.locals = {}
        self.step_calls = self.start_calls = self.rollout_calls = 0
        self.keep_running = True

    def _on_training_start(self):
        self.start_calls += 1

    def _write_episodes(self):
        rows = [{"step": self.num_timesteps, "reward": info["episode"]["r"], "length": info["episode"]["l"]}
                for info in self.locals.get("infos", []) if info.get("episode")]
        if rows:
            with self.episodes_path.open("a") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

    def _on_step(self):
        self.step_calls += 1
        self._write_episodes()
        return self.keep_running

    def _on_rollout_end(self):
        self.rollout_calls += 1


class NativeLaneTrainingDriverTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("benchmark_native_lane_training")

    def test_control_does_not_patch_the_importer(self):
        args = types.SimpleNamespace(native_lanes=False)
        original = self.module.campaign.import_train_job_module
        with patch.object(self.module.campaign, "run_campaign", return_value=0) as run:
            self.assertEqual(self.module.run(args), 0)
        run.assert_called_once_with(args)
        self.assertIs(self.module.campaign.import_train_job_module, original)

    def test_cpu_device_rejected_before_running(self):
        args = types.SimpleNamespace(native_lanes=True, device="cpu")
        with patch.object(self.module.campaign, "run_campaign") as run:
            with self.assertRaisesRegex(ValueError, "CUDA"):
                self.module.run(args)
        run.assert_not_called()

    def test_lane_device_cli_defaults_to_cuda_and_accepts_cpu_simulation(self):
        for extra, expected in (([], "cuda"), (["--lane-device", "cpu"], "cpu")):
            with self.subTest(lane_device=expected), patch.object(self.module, "run", return_value=0) as run:
                self.assertEqual(self.module.main(["--native-lanes", "--device", "cuda", "--output", "/tmp/native-lane-cli-test", *extra]), 0)
                self.assertEqual(run.call_args.args[0].lane_device, expected)
                self.assertEqual(run.call_args.args[0].device, "cuda")

    def test_cpu_lane_still_rejects_cpu_policy(self):
        with patch.object(self.module, "run") as run, patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as caught:
                self.module.main(["--native-lanes", "--lane-device", "cpu", "--device", "cpu", "--output", "/tmp/native-lane-cli-test"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("--native-lanes requires --device cuda", stderr.getvalue())
        run.assert_not_called()

    def test_invalid_lane_device_rejected_before_running(self):
        args = types.SimpleNamespace(native_lanes=True, device="cuda", lane_device="mps")
        with patch.object(self.module.campaign, "run_campaign") as run:
            with self.assertRaisesRegex(ValueError, "lane.device"):
                self.module.run(args)
        run.assert_not_called()

    def test_graph_combination_rejected(self):
        args = types.SimpleNamespace(native_lanes=True, device="cuda", rollout_policy_graph=True)
        with self.assertRaisesRegex(ValueError, "graph"):
            self.module.run(args)

    def test_validate_params_guards(self):
        nav = types.SimpleNamespace(is_navigation=True)
        self.module.validate_params({"frame_stack": 4, "opponent_history": {"enabled": False}}, nav)
        with self.assertRaisesRegex(ValueError, "navigation"):
            self.module.validate_params({}, types.SimpleNamespace(is_navigation=False))
        with self.assertRaisesRegex(ValueError, "4-frame"):
            self.module.validate_params({"frame_stack": 1}, nav)
        with self.assertRaisesRegex(ValueError, "opponent history"):
            self.module.validate_params({"opponent_history": {"enabled": True}}, nav)

    def test_context_builds_lanes_from_probe_and_restores_on_failure(self):
        lanes_module = importlib.import_module("native_lane_vec_env")
        tensor_module = importlib.import_module("tensor_ppo")
        built = []
        wrapped = []
        probe_info = {"wrapper_cfg": {"min_dist": 2.0}, "collision_bias": 0.5, "observation_space": "obs", "action_space": "act", "frame_stack": 4}

        def probe(train_module, phase_config, params):
            return dict(probe_info)

        def fake_build(lanes, **kwargs):
            built.append((lanes, kwargs))
            return "lane-env"

        class FakeTensorVecEnv:
            def __init__(self, env, *, device):
                wrapped.append((env, device))

        original_build = object()
        original_ppo = type("OriginalPPO", (), {})
        candidate_ppo = type("CandidatePPO", (), {})
        job = types.SimpleNamespace(build_env=original_build, PPO=original_ppo)
        params = {"frame_stack": 4, "opponent_history": {"enabled": False}}
        phase = types.SimpleNamespace(is_navigation=True)
        with patch.object(lanes_module, "build_cuda_env", fake_build), patch.object(tensor_module, "TensorVecEnv", FakeTensorVecEnv):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with self.module.native_lane_job_context(job, build_dir=Path("/tmp/lanes-build"), ppo_class=candidate_ppo, record_capacity=99, probe=probe):
                    self.assertIs(job.PPO, candidate_ppo)
                    result = job.build_env("train", phase, params, 4096, 45, "cuda")
                    self.assertIsInstance(result, FakeTensorVecEnv)
                    with self.assertRaisesRegex(ValueError, "CUDA"):
                        job.build_env("train", phase, params, 8, 45, "cpu")
                    raise RuntimeError("injected")
        self.assertIs(job.PPO, original_ppo)
        self.assertIs(job.build_env, original_build)
        self.assertEqual(len(built), 1)
        lanes, kwargs = built[0]
        self.assertEqual(lanes, 4096)
        self.assertEqual(kwargs["seed"], 45)
        self.assertEqual(kwargs["device"], "cuda")
        self.assertEqual(kwargs["wrapper_cfg"], probe_info["wrapper_cfg"])
        self.assertEqual(kwargs["collision_bias"], 0.5)
        self.assertEqual(kwargs["observation_space"], "obs")
        self.assertEqual(kwargs["action_space"], "act")
        self.assertEqual(kwargs["record_capacity"], 99)
        self.assertEqual(kwargs["build_dir"], Path("/tmp/lanes-build"))
        self.assertEqual(wrapped, [("lane-env", "cuda")])

    def test_candidate_writes_provenance_and_restores_importer_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(native_lanes=True, device="cuda", output=Path(tmp), lane_build_dir=str(Path(tmp) / "build"), lane_record_capacity=None)
            original = self.module.campaign.import_train_job_module
            with patch.object(self.module.campaign, "run_campaign", side_effect=RuntimeError("injected")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    self.module.run(args)
            self.assertIs(self.module.campaign.import_train_job_module, original)
            record = json.loads((Path(tmp) / "native_lane_provenance.json").read_text())
            self.assertEqual(set(record["candidate"]), set(self.module.CANDIDATE_FILES))
            self.assertTrue(all(value["sha256"] for value in record["candidate"].values()))
            self.assertIn("navigation_lanes.h", record["native_sources"])
            self.assertEqual(record["training_algorithm"], "installed stable_baselines3.PPO.train")
            self.assertEqual(record["lane_device"], "cuda")
            self.assertEqual(record["policy_device"], "cuda")

    def test_cpu_lane_context_keeps_tensor_ppo_and_wraps_host_simulation(self):
        lanes_module = importlib.import_module("native_lane_vec_env")
        tensor_module = importlib.import_module("tensor_ppo")
        self.assertIsNotNone(importlib.util.find_spec("native_lane_cpu_adapter"), "CPU-native CUDA bridge is missing")
        adapter_module = importlib.import_module("native_lane_cpu_adapter")
        probe_info = {"wrapper_cfg": {"min_dist": 2.0}, "collision_bias": 0.5, "observation_space": "obs", "action_space": "act"}
        job = types.SimpleNamespace(build_env=object(), PPO=object())
        originals = job.build_env, job.PPO
        with patch.object(lanes_module, "build_cpu_env", return_value="host-env") as cpu_build, \
             patch.object(lanes_module, "build_cuda_env") as cuda_build, \
             patch.object(adapter_module, "NativeLaneCPUAdapter", return_value="bridge") as bridge, \
             patch.object(tensor_module, "TensorVecEnv", return_value="tensor-env") as tensor_env:
            with self.module.native_lane_job_context(job, build_dir=Path("/tmp/native-lanes"), lane_device="cpu", record_capacity=99,
                                                     probe=lambda *args: probe_info):
                self.assertIs(job.PPO, tensor_module.TensorPPO)
                result = job.build_env("train", types.SimpleNamespace(is_navigation=True), {}, 8, 45, "cuda:1")
                self.assertEqual(result, "tensor-env")
        self.assertEqual((job.build_env, job.PPO), originals)
        cpu_build.assert_called_once_with(8, build_dir=Path("/tmp/native-lanes"), seed=45, wrapper_cfg=probe_info["wrapper_cfg"],
                                          observation_space="obs", action_space="act", collision_bias=0.5, record_capacity=99, contract="off")
        cuda_build.assert_not_called()
        bridge.assert_called_once_with("host-env", device="cuda:1")
        tensor_env.assert_called_once_with("bridge", device="cuda:1")

    def test_cpu_lane_run_records_devices_and_threads_selector_to_job_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(native_lanes=True, device="cuda:1", lane_device="cpu", output=Path(tmp), lane_build_dir=tmp,
                                         lane_record_capacity=99)
            original = self.module.campaign.import_train_job_module
            job = types.SimpleNamespace(build_env=object(), PPO=object())
            with patch.object(self.module.campaign, "import_train_job_module", return_value=job), \
                 patch.object(self.module, "native_lane_job_context") as context, \
                 patch.object(self.module.campaign, "run_campaign", side_effect=lambda args: self.module.campaign.import_train_job_module("job.py")):
                context.return_value.__enter__.return_value = "context-job"
                self.assertEqual(self.module.run(args), "context-job")
            self.assertIs(self.module.campaign.import_train_job_module, original)
            self.assertEqual(context.call_args.kwargs["lane_device"], "cpu")
            record = json.loads((Path(tmp) / "native_lane_provenance.json").read_text())
            self.assertEqual(record["lane_device"], "cpu")
            self.assertEqual(record["policy_device"], "cuda:1")
            self.assertIn("CPU", record["scope"])
            self.assertIn("native_lane_cpu_adapter.py", record["candidate"])
            self.assertEqual(record["collector"], "TensorPPO")
            self.assertIn("D2H", record["step_transfers"])
            self.assertIn("H2D", record["step_transfers"])

    def test_missing_candidate_source_prevents_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(native_lanes=True, device="cuda", output=Path(tmp), lane_build_dir=tmp, lane_record_capacity=None)
            with patch.object(self.module, "CANDIDATE_FILES", ("not-present-input.py",)):
                with patch.object(self.module.campaign, "run_campaign") as run:
                    with self.assertRaises(FileNotFoundError):
                        self.module.run(args)
                    run.assert_not_called()

    @staticmethod
    def native_info(lane, episode_index, length, *, truncated=False):
        return {"episode": {"r": float(episode_index + lane), "l": length, "t": 17.25},
                "navigation": {"lane": lane, "episode_index": episode_index, "success": not truncated,
                               "targets": 3, "fell": False, "horizon": truncated,
                               "terminated": not truncated, "truncated": truncated}}

    def test_native_callback_records_each_drained_episode_once_with_exact_event_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = types.SimpleNamespace(build_env=object(), PPO=object(), JobCallback=FakeJobCallback)
            info00 = self.native_info(0, 0, 3)
            info01 = self.native_info(0, 1, 4, truncated=True)
            info10 = self.native_info(1, 0, 5)
            info02 = self.native_info(0, 2, 8)
            info11 = self.native_info(1, 1, 10, truncated=True)
            with self.module.native_lane_job_context(job, build_dir=Path(tmp), ppo_class=object()):
                callback = job.JobCallback(tmp)
                callback._on_training_start()
                callback.num_timesteps = 1020
                callback.locals = {"infos": [], "drained_infos": [info01, info10, info00]}
                self.assertTrue(callback._on_step())
                callback._on_rollout_end()
                callback.num_timesteps = 1040
                callback.locals = {"infos": [], "drained_infos": [info11, info01, info02]}
                self.assertTrue(callback._on_step())
                callback._on_rollout_end()
                callback._on_rollout_end()  # A repeated drain must not duplicate records.
                self.assertTrue(callback.episodes_path.is_file(), "native episode records were never persisted")
                rows = [json.loads(line) for line in callback.episodes_path.read_text().splitlines()]
            self.assertEqual(callback.start_calls, 1)
            self.assertEqual(callback.step_calls, 2, "rollout logging must not invoke on_step again")
            self.assertEqual(callback.rollout_calls, 3, "base rollout hooks must still execute")
            self.assertEqual([(row["step"], row["lane"], row["episode_index"]) for row in rows],
                             [(1006, 0, 0), (1010, 1, 0), (1014, 0, 1), (1030, 0, 2), (1030, 1, 1)])
            for row, info in zip(rows, (info00, info10, info01, info02, info11)):
                self.assertEqual(row["navigation"], info["navigation"])
                self.assertEqual(row["episode"], info["episode"])
                self.assertEqual(row["reward"], info["episode"]["r"])
                self.assertEqual(row["length"], info["episode"]["l"])
                self.assertEqual(row["step_basis"], "lane_episode_lengths")
                self.assertEqual(row["recorded_at"], "rollout_end")
                self.assertEqual(row["episode_time_basis"], "rollout_drain_elapsed")
                self.assertNotIn("episode_wall_seconds", row)
            self.assertEqual([row["rollout_end_step"] for row in rows], [1020, 1020, 1020, 1040, 1040])
            self.assertIs(job.JobCallback, FakeJobCallback)

    def test_native_callback_missing_record_uses_labeled_rollout_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = types.SimpleNamespace(build_env=object(), PPO=object(), JobCallback=FakeJobCallback)
            with self.module.native_lane_job_context(job, build_dir=Path(tmp), ppo_class=object()):
                callback = job.JobCallback(tmp)
                callback._on_training_start()
                callback.num_timesteps = 1040
                callback.locals = {"infos": [], "drained_infos": [self.native_info(0, 2, 3)]}
                callback._on_rollout_end()
                callback.num_timesteps = 1060
                callback.locals = {"infos": [], "drained_infos": [self.native_info(0, 3, 4)]}
                callback._on_rollout_end()
                self.assertTrue(callback.episodes_path.is_file(), "native episode records were never persisted")
                rows = [json.loads(line) for line in callback.episodes_path.read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [1040, 1060])
            self.assertTrue(all(row["step_basis"] == "rollout_end" for row in rows))

    def test_native_callback_preserves_host_logging_and_step_stop_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = types.SimpleNamespace(build_env=object(), PPO=object(), JobCallback=FakeJobCallback)
            with self.module.native_lane_job_context(job, build_dir=Path(tmp), ppo_class=object()):
                callback = job.JobCallback(tmp)
                callback._on_training_start()
                callback.locals = {"infos": [{"episode": {"r": 2.0, "l": 3}}], "drained_infos": []}
                callback.keep_running = False
                self.assertFalse(callback._on_step())
                callback._on_rollout_end()
                self.assertEqual(callback.step_calls, 1)
                self.assertEqual(callback.rollout_calls, 1)
                rows = [json.loads(line) for line in callback.episodes_path.read_text().splitlines()]
                self.assertEqual(rows, [{"step": 1000, "reward": 2.0, "length": 3}])

    def test_native_callback_class_is_restored_after_context_failure(self):
        job = types.SimpleNamespace(build_env=object(), PPO=object(), JobCallback=FakeJobCallback)
        originals = job.build_env, job.PPO, job.JobCallback
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with self.module.native_lane_job_context(job, build_dir=Path("/tmp/lanes-build"), ppo_class=object()):
                self.assertIsNot(job.JobCallback, FakeJobCallback, "native context did not replace the callback")
                self.assertTrue(issubclass(job.JobCallback, FakeJobCallback))
                raise RuntimeError("injected")
        self.assertEqual((job.build_env, job.PPO, job.JobCallback), originals)


if __name__ == "__main__":
    unittest.main()
