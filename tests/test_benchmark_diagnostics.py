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


class FakeOptimizer:
    def state_dict(self):
        return {"state": {"m": [0.0]}, "param_groups": [{"lr": 0.1}]}


class FakeStatefulPolicy:
    def __init__(self) -> None:
        self.optimizer = FakeOptimizer()
        self.weight = 1.0

    def state_dict(self):
        return {"weight": [self.weight]}


class FakeRolloutBuffer:
    def __init__(self) -> None:
        self.cache_cleared = False
        self.observations = [[0.0]]
        self.actions = [[0.0]]
        self.rewards = [[0.0]]
        self.episode_starts = [[1.0]]
        self.values = [[0.0]]
        self.log_probs = [[0.0]]
        self.advantages = [[0.0]]
        self.returns = [[0.0]]
        self.pos = 0
        self.full = False
        self.buffer_size = 2
        self.n_envs = 64

    def clear_cache(self):
        self.cache_cleared = True


class FakeUpdateAgent:
    def __init__(self) -> None:
        self.env = object()
        self.rollout_buffer = FakeRolloutBuffer()
        self.policy = FakeStatefulPolicy()
        self.n_steps = 2
        self.num_timesteps = 0
        self.collect_calls = 0
        self.train_calls = 0
        self.learn_calls = 0
        self.progress_calls = []
        self.logger_dir = None

    def _setup_learn(self, total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar):
        return total_timesteps, callback

    def _update_current_progress_remaining(self, num_timesteps, total_timesteps):
        self.progress_calls.append((num_timesteps, total_timesteps))

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self.collect_calls += 1
        rollout_buffer.observations = [[1.0], [2.0]]
        rollout_buffer.actions = [[0.25], [-0.25]]
        rollout_buffer.rewards = [[1.0], [0.5]]
        rollout_buffer.episode_starts = [[1.0], [0.0]]
        rollout_buffer.values = [[0.1], [0.2]]
        rollout_buffer.log_probs = [[-0.1], [-0.2]]
        rollout_buffer.advantages = [[0.9], [0.3]]
        rollout_buffer.returns = [[1.0], [0.8]]
        rollout_buffer.pos = n_rollout_steps
        rollout_buffer.full = True
        self.num_timesteps += 64 * n_rollout_steps
        return True

    def train(self):
        self.train_calls += 1
        self.policy.weight += 1.0

    def learn(self, *args, **kwargs):
        self.learn_calls += 1
        raise AssertionError("update ceiling must not call learn()")


class FakeTrainJobForLogger:
    def __init__(self) -> None:
        self.logger_dirs = []

    def _set_model_logger(self, agent, output_dir):
        self.logger_dirs.append(Path(output_dir))
        agent.logger_dir = Path(output_dir)


class BenchmarkDiagnosticsTest(unittest.TestCase):
    def test_union_intervals_merges_overlaps_without_double_counting(self):
        import tools.benchmarks.benchmark_diagnostics as diag

        self.assertEqual(diag.union_intervals([(0, 5), (3, 9), (12, 15), (15, 20)]), 17)
        self.assertEqual(diag.union_intervals([]), 0.0)

    def test_trace_gpu_summary_uses_union_not_sum_of_overlapping_kernels(self):
        import tools.benchmarks.benchmark_diagnostics as diag

        trace = {
            "traceEvents": [
                {"ph": "X", "cat": "kernel", "name": "a", "ts": 0, "dur": 10},
                {"ph": "X", "cat": "kernel", "name": "b", "ts": 5, "dur": 10},
                {"ph": "X", "cat": "gpu_memcpy", "name": "copy", "ts": 20, "dur": 5},
                {"ph": "X", "cat": "cpu_op", "name": "cpu", "ts": 0, "dur": 10},
                {"ph": "X", "cat": "cpu_op", "name": "idle_gap_after_cpu", "ts": 35, "dur": 5},
            ]
        }

        summary = diag.summarize_torch_trace_events(trace)

        self.assertEqual(summary["kernel_us_sum"], 20)
        self.assertEqual(summary["copy_us_sum"], 5)
        self.assertEqual(summary["gpu_active_union_us"], 20)
        self.assertEqual(summary["trace_span_us"], 40)
        self.assertEqual(summary["gpu_active_fraction_of_trace_span"], 0.5)

    def test_phase_summary_reports_finite_workload_and_provenance(self):
        import tools.benchmarks.benchmark_diagnostics as diag

        phases = [
            {"label": "fixed_action_env", "steps": 1000, "seconds": 2.0},
            {"label": "inference_64", "batches": 10, "seconds": 1.0},
        ]
        provenance = {"config": {"path": "/x/config.yaml", "sha256": "abc"}}

        payload = diag.build_summary(phases=phases, provenance=provenance)

        self.assertEqual(payload["phase_count"], 2)
        self.assertEqual(payload["phases"][0]["steps_per_second"], 500)
        self.assertEqual(payload["phases"][1]["batches_per_second"], 10)
        self.assertEqual(payload["provenance"], provenance)

    def test_fixed_update_ceiling_collects_once_then_trains_from_copied_snapshot(self):
        import tools.benchmarks.benchmark_diagnostics as diag

        collector = FakeUpdateAgent()
        collected = diag.collect_rollout_snapshot_from_agent(
            collector,
            callback=None,
            total_timesteps=128,
            n_rollout_steps=2,
            sync=lambda: None,
            label="collect_once",
        )
        snapshot = collected["snapshot"]
        before_norm = snapshot["sample_norm"]

        control = FakeUpdateAgent()
        cache = FakeUpdateAgent()
        control_result = diag.time_buffered_update_from_snapshot(
            control,
            snapshot,
            sync=lambda: None,
            label="control_update",
            reference_fingerprint=snapshot["agent_state_fingerprint"],
        )
        cache_result = diag.time_buffered_update_from_snapshot(
            cache,
            snapshot,
            sync=lambda: None,
            label="cache_update",
            reference_fingerprint=snapshot["agent_state_fingerprint"],
        )

        self.assertEqual(collector.collect_calls, 1)
        self.assertEqual(control.collect_calls, 0)
        self.assertEqual(cache.collect_calls, 0)
        self.assertEqual(control.learn_calls, 0)
        self.assertEqual(cache.learn_calls, 0)
        self.assertEqual(control.train_calls, 1)
        self.assertEqual(cache.train_calls, 1)
        self.assertEqual(control.rollout_buffer.observations, [[1.0], [2.0]])
        self.assertEqual(control.rollout_buffer.full, True)
        self.assertEqual(control.progress_calls, [(128, 128)])
        self.assertEqual(control_result["sample_norm"], before_norm)
        self.assertEqual(control_result["restored_sample_norm"], before_norm)
        self.assertEqual(cache_result["initial_state_fingerprint"], control_result["initial_state_fingerprint"])
        self.assertEqual(cache_result["final_state_fingerprint"], control_result["final_state_fingerprint"])
        self.assertTrue(control_result["state_changed_after_train"])

    def test_direct_update_prepares_logger_before_training(self):
        import tools.benchmarks.benchmark_diagnostics as diag

        agent = FakeUpdateAgent()
        train_job = FakeTrainJobForLogger()
        with tempfile.TemporaryDirectory(prefix="ai2-diag-log-") as output:
            diag.prepare_update_agent_for_train(train_job, agent, Path(output) / "arm")

        self.assertEqual(agent.logger_dir.name, "arm")
        self.assertEqual(train_job.logger_dirs[0].name, "arm")

    def test_direct_update_clears_cache_when_train_raises(self):
        import tools.benchmarks.benchmark_diagnostics as diag

        class RaisingAgent(FakeUpdateAgent):
            def train(self):
                super().train()
                raise RuntimeError("boom")

        collector = FakeUpdateAgent()
        snapshot = diag.collect_rollout_snapshot_from_agent(
            collector,
            callback=None,
            total_timesteps=128,
            n_rollout_steps=2,
            sync=lambda: None,
            label="collect_once",
        )["snapshot"]
        agent = RaisingAgent()

        with self.assertRaises(RuntimeError):
            diag.time_buffered_update_from_snapshot(
                agent,
                snapshot,
                sync=lambda: None,
                label="raising_update",
                reference_fingerprint=snapshot["agent_state_fingerprint"],
            )

        self.assertTrue(agent.rollout_buffer.cache_cleared)

    def test_pair_parity_marks_matching_control_and_cache_finals(self):
        import tools.benchmarks.benchmark_diagnostics as diag

        control = {"label": "control", "initial_state_fingerprint": "initial", "final_state_fingerprint": "abc", "state_changed_after_train": True}
        cache = {"label": "cache", "initial_state_fingerprint": "initial", "final_state_fingerprint": "abc", "state_changed_after_train": True}

        diag.assert_and_record_pair_parity(control, cache, pair_index=3)

        self.assertTrue(control["pair_final_fingerprint_parity"])
        self.assertTrue(cache["pair_final_fingerprint_parity"])
        self.assertEqual(control["paired_with"], "cache")
        self.assertEqual(cache["pair_index"], 3)

    def test_pair_parity_rejects_matching_noop_or_missing_update_evidence(self):
        import tools.benchmarks.benchmark_diagnostics as diag

        for evidence in (
            {"initial_state_fingerprint": "same", "final_state_fingerprint": "same", "state_changed_after_train": False},
            {"initial_state_fingerprint": "same", "final_state_fingerprint": "same", "state_changed_after_train": True},
            {"final_state_fingerprint": "same"},
            {},
        ):
            with self.subTest(evidence=evidence), self.assertRaises(RuntimeError):
                diag.assert_and_record_pair_parity(dict(evidence), dict(evidence), pair_index=0)

    def test_pair_parity_fails_closed_on_mismatched_finals(self):
        import tools.benchmarks.benchmark_diagnostics as diag

        with self.assertRaises(RuntimeError):
            diag.assert_and_record_pair_parity(
                {"label": "control", "initial_state_fingerprint": "initial", "final_state_fingerprint": "abc", "state_changed_after_train": True},
                {"label": "cache", "initial_state_fingerprint": "initial", "final_state_fingerprint": "def", "state_changed_after_train": True},
                pair_index=0,
            )

    def test_rollout_snapshot_fails_closed_when_collection_stops_early(self):
        import tools.benchmarks.benchmark_diagnostics as diag

        class StoppedAgent(FakeUpdateAgent):
            def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
                super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
                return False

        with self.assertRaises(RuntimeError):
            diag.collect_rollout_snapshot_from_agent(
                StoppedAgent(),
                callback=None,
                total_timesteps=128,
                n_rollout_steps=2,
                sync=lambda: None,
                label="stopped",
            )

    def test_timed_patch_counts_only_enabled_region_and_restores(self):
        import tools.benchmarks.benchmark_diagnostics as diag

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
        import tools.benchmarks.benchmark_diagnostics as diag

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
