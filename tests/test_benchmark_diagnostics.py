"""Unit checks for standalone PPO diagnostic accounting."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path


class FakeLeafEnv:
    def __init__(self) -> None:
        self.step_calls = 0
        self.reset_calls = 0

    def step(self, action=None):
        self.step_calls += 1
        return None

    def reset(self):
        self.reset_calls += 1
        return None


class FakeVecEnv:
    def __init__(self, leaf: FakeLeafEnv) -> None:
        self.leaf = leaf
        self.closed = False

    def step_wait(self):
        self.leaf.step(None)
        return None, None, [False], [{}]

    def reset(self):
        self.leaf.reset()
        return None

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self) -> None:
        self.forward_calls = 0

    def forward(self, obs=None):
        self.forward_calls += 1
        return None


class FakeModel:
    def __init__(self, *args, **kwargs) -> None:
        self.env = args[1]
        self.n_steps = kwargs["n_steps"]
        self.policy = FakePolicy()
        self.num_timesteps = 0
        self.update_calls = 0
        self.progress_calls = []

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        self.progress_calls.append((num_timesteps, total_timesteps))

    def learn(self, total_timesteps, callback, progress_bar=False, reset_num_timesteps=True):
        callback._on_training_start()
        self._update_current_progress_remaining(total_timesteps, total_timesteps)
        remaining = total_timesteps
        while remaining > 0:
            callback._on_rollout_start()
            self.env.reset()
            for _ in range(min(self.n_steps, remaining)):
                self.policy.forward(None)
                self.env.step_wait()
                self.num_timesteps += 1
                callback.locals = {"dones": [False], "infos": [{}]}
                callback._on_step()
            callback._on_rollout_end()
            self.train()
            remaining -= self.n_steps
        return self

    def train(self) -> None:
        self.update_calls += 1


class BenchmarkDiagnosticsTest(unittest.TestCase):
    def test_timed_patch_counts_only_enabled_region_and_restores(self):
        import benchmark_diagnostics as diag

        leaf = FakeLeafEnv()
        stats = diag.ComponentStats()
        with diag.TimedPatch(leaf, "step", "game_step", stats):
            leaf.step(None)
            self.assertEqual(stats.by_label["game_step"].count, 0)
            with stats.measure_phase():
                leaf.step(None)
            leaf.step(None)

        self.assertEqual(stats.by_label["game_step"].count, 1)
        self.assertIs(leaf.step.__func__, FakeLeafEnv.step)
        leaf.step(None)
        self.assertEqual(stats.by_label["game_step"].count, 1)

    def test_ppo_diagnostic_reports_measured_phase_only_with_full_update_counts(self):
        import benchmark_diagnostics as diag

        leaf = FakeLeafEnv()
        vec_env = FakeVecEnv(leaf)
        args = argparse.Namespace(
            config=Path("config.yaml"),
            device="cpu",
            n_steps=4,
            batch_size=16,
            ppo_rollouts=2,
            output=Path("unused"),
            profile=False,
            seed=11,
            torch_threads=1,
        )

        def build_env(config, seed, n_envs, torch_threads):
            self.assertEqual(n_envs, 1)
            return (
                vec_env,
                {"ppo_settings": {"n_epochs": 1, "time_steps": 5_000_000}, "policy_kwargs": {}},
                object(),
            )

        with tempfile.TemporaryDirectory(prefix="ai2-diag-test-") as output:
            result = diag.run_diagnostic(
                args,
                Path(output),
                env_builder=build_env,
                model_cls=FakeModel,
                raw_env_getter=lambda env: env.leaf,
            )

            payload = json.loads((Path(output) / "benchmark_diagnostics.json").read_text())

        self.assertTrue(vec_env.closed)
        self.assertEqual(result["configured_training_budget"], 5_000_000)
        self.assertEqual(result["schedule_total_timesteps"], 5_000_000)
        self.assertEqual(result["progress_total_timesteps"], [5_000_000, 5_000_000])
        self.assertEqual(result["measured_timesteps"], 8)
        self.assertEqual(result["batch_size"], 4)
        self.assertEqual(result["rollout_count"], 2)
        self.assertEqual(result["update_count"], 2)
        self.assertEqual(result["components"]["game_step"]["count"], 8)
        self.assertEqual(result["components"]["game_reset"]["count"], 2)
        self.assertEqual(result["components"]["policy_forward"]["count"], 8)
        self.assertEqual(payload["result"]["label"], "instrumented_single_env_ppo")
        self.assertEqual(payload["result"]["components"]["vec_reset"]["count"], 2)
        self.assertEqual(payload["result"]["components"]["vec_step_wait"]["count"], 8)


if __name__ == "__main__":
    unittest.main()
