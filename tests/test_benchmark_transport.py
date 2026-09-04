"""Checks for the cheap vector-transport benchmark control."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTransportTest(unittest.TestCase):
    def test_counts_and_reset_boundary_for_direct_runner(self):
        import benchmark_transport as transport

        with tempfile.TemporaryDirectory(prefix="ai2-transport-test-") as output:
            result = transport.run_transport_repeat(
                n_envs=1,
                aggregate_steps=1_805,
                aggregate_warmup_steps=0,
                obs_dim=16,
                frame_stack=4,
                repeat=0,
                output_dir=Path(output),
                seed=7,
            )
            payload = json.loads((Path(output) / "transport_repeat0.json").read_text())

        self.assertEqual(result["label"], "cheap_environment_transport_control")
        self.assertEqual(result["aggregate_transitions"], 1_805)
        self.assertEqual(result["vector_step_calls"], 1_805)
        self.assertEqual(result["vector_reset_calls"], 2)
        self.assertEqual(result["cheap_env_total_reset_count"], 3)
        self.assertEqual(result["cheap_env_total_terminal_resets"], 1)
        self.assertEqual(result["cheap_env_measured_reset_count"], 1)
        self.assertEqual(result["cheap_env_measured_terminal_resets"], 1)
        self.assertEqual(result["obs_dim"], 16)
        self.assertEqual(result["frame_stack"], 4)
        self.assertEqual(result["base_obs_shape"], [16])
        self.assertEqual(result["stacked_obs_shape"], [64])
        self.assertEqual(payload["result"]["aggregate_transitions"], 1_805)
        self.assertGreater(result["seconds"], 0)
        self.assertGreater(result["steps_per_second"], 0)

    def test_cli_smoke_dummyvec_one_env(self):
        with tempfile.TemporaryDirectory(prefix="ai2-transport-cli1-") as output:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmark_transport.py"),
                    "--n-envs",
                    "1",
                    "--steps",
                    "8",
                    "--warmup-steps",
                    "2",
                    "--repeats",
                    "1",
                    "--obs-dim",
                    "32",
                    "--frame-stack",
                    "4",
                    "--output",
                    output,
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            summary = json.loads((Path(output) / "summary.json").read_text())

        repeat = summary["repeats"][0]
        self.assertEqual(repeat["n_envs"], 1)
        self.assertEqual(repeat["aggregate_transitions"], 8)
        self.assertEqual(repeat["vector_step_calls"], 8)
        self.assertEqual(repeat["vec_env_type"], "DummyVecEnv")
        self.assertEqual(repeat["base_obs_shape"], [32])
        self.assertEqual(repeat["stacked_obs_shape"], [128])

    def test_cli_smoke_subproc_two_envs(self):
        with tempfile.TemporaryDirectory(prefix="ai2-transport-cli2-") as output:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmark_transport.py"),
                    "--n-envs",
                    "2",
                    "--steps",
                    "8",
                    "--warmup-steps",
                    "2",
                    "--repeats",
                    "1",
                    "--obs-dim",
                    "32",
                    "--frame-stack",
                    "4",
                    "--output",
                    output,
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            summary = json.loads((Path(output) / "summary.json").read_text())

        repeat = summary["repeats"][0]
        self.assertEqual(repeat["n_envs"], 2)
        self.assertEqual(repeat["aggregate_transitions"], 8)
        self.assertEqual(repeat["vector_step_calls"], 4)
        self.assertEqual(repeat["vec_env_type"], "SubprocVecEnv")
        self.assertEqual(repeat["base_obs_shape"], [32])
        self.assertEqual(repeat["stacked_obs_shape"], [128])

    def test_warmup_reset_boundary_excluded_from_measured_counts(self):
        import benchmark_transport as transport

        with tempfile.TemporaryDirectory(prefix="ai2-transport-warmup-") as output:
            result = transport.run_transport_repeat(
                n_envs=1,
                aggregate_steps=8,
                aggregate_warmup_steps=1_805,
                obs_dim=16,
                frame_stack=4,
                repeat=0,
                output_dir=Path(output),
                seed=7,
            )

        self.assertEqual(result["cheap_env_total_reset_count"], 3)
        self.assertEqual(result["cheap_env_total_terminal_resets"], 1)
        self.assertEqual(result["cheap_env_measured_reset_count"], 0)
        self.assertEqual(result["cheap_env_measured_terminal_resets"], 0)

    def test_parallel_reset_boundary_uses_per_env_vector_steps(self):
        import benchmark_transport as transport

        with tempfile.TemporaryDirectory(prefix="ai2-transport-parallel-reset-") as output:
            result = transport.run_transport_repeat(
                n_envs=2,
                aggregate_steps=1_808,
                aggregate_warmup_steps=0,
                obs_dim=16,
                frame_stack=4,
                repeat=0,
                output_dir=Path(output),
                seed=7,
            )

        self.assertEqual(result["vector_step_calls"], 904)
        self.assertEqual(result["cheap_env_total_reset_count"], 4)
        self.assertEqual(result["cheap_env_total_terminal_resets"], 0)
        self.assertEqual(result["cheap_env_measured_reset_count"], 0)
        self.assertEqual(result["cheap_env_measured_terminal_resets"], 0)


if __name__ == "__main__":
    unittest.main()
