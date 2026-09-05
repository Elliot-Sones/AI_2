"""Unit checks for the isolated PPO rollout pipeline profiler."""

from __future__ import annotations

import importlib.util
import sys
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_probe():
    spec = importlib.util.spec_from_file_location("probe_pipeline", HERE / "probe_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PipelineProbeTest(unittest.TestCase):
    def test_rollout_shape_keeps_total_rollout_batch_across_env_counts(self):
        probe = load_probe()

        one_env = probe.rollout_shape(n_envs=1, rollout_transitions=8192, requested_batch_size=8192)
        sixteen_env = probe.rollout_shape(
            n_envs=16, rollout_transitions=8192, requested_batch_size=8192
        )

        self.assertEqual(one_env.n_steps, 8192)
        self.assertEqual(one_env.batch_size, 8192)
        self.assertEqual(one_env.transitions_per_rollout, 8192)
        self.assertEqual(sixteen_env.n_steps, 512)
        self.assertEqual(sixteen_env.batch_size, 8192)
        self.assertEqual(sixteen_env.transitions_per_rollout, 8192)
        with self.assertRaisesRegex(ValueError, "divisible"):
            probe.rollout_shape(n_envs=16, rollout_transitions=8193, requested_batch_size=8192)

    def test_ast_instrumentation_wraps_expected_collect_rollout_statements(self):
        probe = load_probe()
        source = '''
def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
    callback.on_rollout_start()
    n_steps = 0
    while n_steps < n_rollout_steps:
        obs_tensor = obs_as_tensor(self._last_obs, self.device)
        with th.no_grad():
            actions, values, log_probs = self.policy(obs_tensor)
        actions = actions.cpu().numpy()
        clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
        new_obs, rewards, dones, infos = env.step(clipped_actions)
        self.num_timesteps += env.num_envs
        callback.update_locals(locals())
        if not callback.on_step():
            return False
        for idx, done in enumerate(dones):
            if done and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = self.policy.predict_values(terminal_obs)[0]
                rewards[idx] += self.gamma * terminal_value
        rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
        self._last_obs = new_obs
        self._last_episode_starts = dones
        n_steps += 1
    with th.no_grad():
        values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
    callback.on_rollout_end()
    return True
'''

        transformed = probe.instrument_collect_rollouts_source(source)
        labels = {
            node.items[0].context_expr.args[0].value
            for node in ast_walk_with_items(transformed)
            if node.items
            and getattr(getattr(node.items[0].context_expr, "func", None), "attr", None) == "stage"
            and node.items[0].context_expr.args
        }

        self.assertTrue(
            {
                "callback_rollout_start",
                "observations_to_tensor",
                "policy_forward",
                "actions_gpu_to_cpu",
                "clip_actions",
                "env_step",
                "callback",
                "info",
                "timeout_bootstrap",
                "buffer_add",
                "final_value",
                "compute_returns_advantage",
                "callback_rollout_end",
            }.issubset(labels)
        )

    def test_timer_reports_exclusive_stage_seconds_and_transition_rates(self):
        probe = load_probe()
        timer = probe.PipelineTimer(enabled=True)
        timer.begin_rollout()
        timer.record_vector_cycle()
        timer.record_vector_cycle()
        with timer.stage("outer"):
            time.sleep(0.001)
            with timer.stage("inner"):
                time.sleep(0.001)
        with timer.stage("env_step"):
            time.sleep(0.001)
        timer.finish_rollout(transitions=32)

        report = timer.report(
            envelope_seconds=timer.rollouts[0].seconds + 0.01,
            transitions=32,
            n_envs=16,
            rollouts=1,
            update_seconds=0.002,
        )

        self.assertEqual(report["vector_cycle_count"], 2)
        self.assertEqual(report["transitions"], 32)
        self.assertGreater(report["stages"]["outer"]["exclusive_seconds"], 0)
        self.assertGreater(report["stages"]["inner"]["exclusive_seconds"], 0)
        self.assertEqual(len(report["env_step_events"]), 1)
        self.assertLess(report["env_step_events"][0][0], report["env_step_events"][0][1])
        self.assertGreaterEqual(report["residual_seconds"], 0)
        self.assertAlmostEqual(report["transitions_per_second"], 32 / report["envelope_seconds"])


def ast_walk_with_items(tree):
    import ast

    return (node for node in ast.walk(tree) if isinstance(node, ast.With))


if __name__ == "__main__":
    unittest.main()
