import math
import unittest

from training_benchmark_contract import (
    ContractError,
    END_TO_END_STAGES,
    SCHEMA_VERSION,
    compare_receipts,
    validate_receipt,
)


def receipt(**overrides):
    base = {
        "schema_version": SCHEMA_VERSION,
        "system": {
            "hardware": {
                "accelerator_model": "RTX 4090",
                "accelerator_count": 1,
                "cpu_model": "test-cpu",
            }
        },
        "workload": {
            "unit": "agent_transition",
            "timing_scope": "end_to_end",
            "included_stages": list(END_TO_END_STAGES),
            "warmup_policy": {"mode": "discard_first", "iterations": 2},
            "synchronization": {"device": "cuda", "method": "torch.cuda.synchronize"},
            "overlap": {"mode": "none", "evidence": ["not_applicable"]},
            "environment": {
                "env_id": "WarehouseBrawl",
                "observation_contract": "native_full_v1",
                "action_contract": "discrete_21_v1",
                "tick_semantics": "one_transition_per_agent_action",
                "terminal_bootstrap": "gae_bootstrap_v1",
            },
            "model": {
                "architecture": "mlp",
                "parameter_count": 123456,
                "precision": "fp32",
            },
            "rollout": {
                "num_envs": 1024,
                "horizon": 128,
                "transitions_per_batch": 131072,
            },
        },
        "algorithm": {
            "optimizer": "adam",
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "sample_reuse_epochs": 4,
            "minibatches": 8,
            "ppo_epochs": 4,
            "replay_semantics": "none",
            "actor_staleness": "synchronous",
            "advantage_normalization": "per_batch",
            "terminal_bootstrap": "gae_bootstrap_v1",
        },
        "timing": {
            "elapsed_seconds": 10.0,
        },
        "measurement": {
            "agent_transitions": 1000,
            "claimed_rate": 100.0,
        },
        "correctness": {
            "env_trace_parity": True,
            "rollout_parity": True,
            "fixed_transition_quality": True,
            "fixed_wall_clock_quality": True,
        },
    }
    return deep_update(base, overrides)


def deep_update(base, overrides):
    result = {}
    for key, value in base.items():
        if isinstance(value, dict):
            result[key] = deep_update(value, overrides.get(key, {}))
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def comparison_with(right_overrides):
    return compare_receipts(receipt(), receipt(**right_overrides))


class ValidationTests(unittest.TestCase):
    def test_rejects_ambiguous_steps_counter(self):
        raw = receipt(
            workload={"unit": "steps"},
            measurement={"agent_transitions": None, "steps": 1000},
        )

        with self.assertRaisesRegex(ContractError, "agent_transition"):
            validate_receipt(raw)

    def test_rejects_missing_required_fields(self):
        raw = receipt()
        del raw["workload"]["warmup_policy"]

        with self.assertRaisesRegex(ContractError, "workload.warmup_policy"):
            validate_receipt(raw)

    def test_derives_rate_and_returns_json_safe_deep_copy(self):
        raw = receipt()
        normalized = validate_receipt(raw)

        self.assertEqual("ai2.benchmark_contract.v1", normalized["schema_version"])
        self.assertEqual(100.0, normalized["measurement"]["derived_rate"])
        self.assertIsNot(normalized, raw)
        raw["workload"]["model"]["parameter_count"] = 1
        self.assertEqual(123456, normalized["workload"]["model"]["parameter_count"])

    def test_rejects_claimed_rate_disagreement(self):
        raw = receipt(measurement={"claimed_rate": 125.0})

        with self.assertRaisesRegex(ContractError, "claimed_rate"):
            validate_receipt(raw)

    def test_rejects_zero_elapsed_and_nonfinite_values(self):
        with self.assertRaisesRegex(ContractError, "elapsed_seconds"):
            validate_receipt(receipt(timing={"elapsed_seconds": 0}))

        with self.assertRaisesRegex(ContractError, "finite"):
            validate_receipt(receipt(measurement={"claimed_rate": math.inf}))

    def test_rejects_end_to_end_receipt_missing_canonical_stages(self):
        raw = receipt(workload={"included_stages": ["environment_step"]})

        with self.assertRaisesRegex(ContractError, "included_stages"):
            validate_receipt(raw)

    def test_requires_cuda_trace_or_event_timeline_for_async_overlap(self):
        raw = receipt(workload={"overlap": {"mode": "async", "evidence": ["config_only"]}})

        with self.assertRaisesRegex(ContractError, "overlap.evidence"):
            validate_receipt(raw)

        ok = receipt(workload={"overlap": {"mode": "async", "evidence": ["cuda_trace"]}})
        self.assertEqual("async", validate_receipt(ok)["workload"]["overlap"]["mode"])


class ComparisonTests(unittest.TestCase):
    def test_matched_receipts_are_comparable_and_promotion_eligible(self):
        result = comparison_with(
            {"timing": {"elapsed_seconds": 5.0}, "measurement": {"claimed_rate": 200.0}}
        )

        self.assertTrue(result["throughput_comparable"])
        self.assertTrue(result["learning_behavior_comparable"])
        self.assertTrue(result["promotion_eligible"])
        self.assertEqual(100.0, result["left_rate"])
        self.assertEqual(200.0, result["right_rate"])
        self.assertEqual(2.0, result["speedup_ratio"])
        self.assertEqual([], result["throughput_blockers"])
        self.assertEqual([], result["learning_behavior_blockers"])
        self.assertEqual([], result["promotion_blockers"])

    def test_gpu_mismatch_blocks_throughput(self):
        result = comparison_with(
            {"system": {"hardware": {"accelerator_model": "RTX 3090"}}}
        )

        self.assertFalse(result["throughput_comparable"])
        self.assertIn("system.hardware.accelerator_model", result["throughput_blockers"])
        self.assertIsNone(result["speedup_ratio"])

    def test_stage_or_scope_mismatch_blocks_end_to_end_speed_claim(self):
        env_only = comparison_with(
            {
                "workload": {
                    "timing_scope": "env_only",
                    "included_stages": ["environment_step"],
                }
            }
        )

        self.assertFalse(env_only["throughput_comparable"])
        self.assertIn("workload.timing_scope", env_only["throughput_blockers"])
        self.assertIn("workload.included_stages", env_only["throughput_blockers"])

    def test_environment_semantics_mismatch_blocks_throughput_and_learning(self):
        result = comparison_with(
            {
                "workload": {
                    "environment": {"observation_contract": "puffer_observation_v2"}
                }
            }
        )

        self.assertFalse(result["throughput_comparable"])
        self.assertFalse(result["learning_behavior_comparable"])
        self.assertIn(
            "workload.environment.observation_contract", result["throughput_blockers"]
        )
        self.assertIn(
            "workload.environment.observation_contract",
            result["learning_behavior_blockers"],
        )

    def test_model_parameters_or_precision_mismatch_blocks_throughput(self):
        result = comparison_with(
            {"workload": {"model": {"parameter_count": 999, "precision": "tf32"}}}
        )

        self.assertFalse(result["throughput_comparable"])
        self.assertIn("workload.model.parameter_count", result["throughput_blockers"])
        self.assertIn("workload.model.precision", result["throughput_blockers"])

    def test_rollout_and_ppo_geometry_mismatch_blocks_throughput(self):
        result = comparison_with(
            {
                "workload": {"rollout": {"num_envs": 2048}},
                "algorithm": {"minibatches": 16, "ppo_epochs": 2},
            }
        )

        self.assertFalse(result["throughput_comparable"])
        self.assertIn("workload.rollout.num_envs", result["throughput_blockers"])
        self.assertIn("algorithm.minibatches", result["throughput_blockers"])
        self.assertIn("algorithm.ppo_epochs", result["throughput_blockers"])

    def test_optimizer_hyperparameters_block_learning_comparison_only(self):
        result = comparison_with({"algorithm": {"learning_rate": 0.001}})

        self.assertTrue(result["throughput_comparable"])
        self.assertFalse(result["learning_behavior_comparable"])
        self.assertIn("algorithm.learning_rate", result["learning_behavior_blockers"])
        self.assertIsNone(result["speedup_ratio"])

    def test_actor_staleness_blocks_learning_comparison(self):
        result = comparison_with({"algorithm": {"actor_staleness": "one_batch_stale"}})

        self.assertTrue(result["throughput_comparable"])
        self.assertFalse(result["learning_behavior_comparable"])
        self.assertIn("algorithm.actor_staleness", result["learning_behavior_blockers"])

    def test_async_evidence_mismatch_blocks_throughput(self):
        result = comparison_with(
            {"workload": {"overlap": {"mode": "async", "evidence": ["event_timeline"]}}}
        )

        self.assertFalse(result["throughput_comparable"])
        self.assertIn("workload.overlap.mode", result["throughput_blockers"])
        self.assertIn("workload.overlap.evidence", result["throughput_blockers"])

    def test_missing_correctness_or_quality_evidence_blocks_promotion(self):
        result = comparison_with({"correctness": {"fixed_wall_clock_quality": False}})

        self.assertTrue(result["throughput_comparable"])
        self.assertTrue(result["learning_behavior_comparable"])
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(["correctness.fixed_wall_clock_quality"], result["promotion_blockers"])
        self.assertIsNone(result["speedup_ratio"])


if __name__ == "__main__":
    unittest.main()
