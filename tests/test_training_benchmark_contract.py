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
            "elapsed_source": "wall_clock",
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

    def test_rejects_ambiguous_measurement_counter_spellings_case_insensitively(self):
        for counter in ("StepS", "SPS", "Game Ticks", "Game_Ticks", "Episodes", "Optimizer Calls", "Optimizer_Calls"):
            with self.subTest(counter=counter):
                raw = receipt(measurement={counter: 1000})

                with self.assertRaisesRegex(ContractError, "ambiguous"):
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

    def test_rejects_non_integer_or_nonpositive_count_fields(self):
        invalid_cases = (
            ("measurement.agent_transitions", {"measurement": {"agent_transitions": 1.5}}),
            ("workload.model.parameter_count", {"workload": {"model": {"parameter_count": 0}}}),
            ("workload.rollout.num_envs", {"workload": {"rollout": {"num_envs": -1}}}),
            ("workload.rollout.horizon", {"workload": {"rollout": {"horizon": 0}}}),
            (
                "workload.rollout.transitions_per_batch",
                {"workload": {"rollout": {"transitions_per_batch": 1.2}}},
            ),
            ("algorithm.sample_reuse_epochs", {"algorithm": {"sample_reuse_epochs": 0}}),
            ("algorithm.minibatches", {"algorithm": {"minibatches": 0}}),
            ("algorithm.ppo_epochs", {"algorithm": {"ppo_epochs": -1}}),
            ("system.hardware.accelerator_count", {"system": {"hardware": {"accelerator_count": 0}}}),
        )

        for path, overrides in invalid_cases:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ContractError, path):
                    validate_receipt(receipt(**overrides))

    def test_rejects_unknown_timing_scope_overlap_mode_and_bad_stages(self):
        invalid_cases = (
            (
                "workload.timing_scope",
                {"workload": {"timing_scope": "collection"}},
            ),
            (
                "workload.overlap.mode",
                {"workload": {"overlap": {"mode": "pipeline", "evidence": ["cuda_trace"]}}},
            ),
            (
                "workload.included_stages",
                {"workload": {"included_stages": list(END_TO_END_STAGES) + ["ppo_update"]}},
            ),
            (
                "workload.included_stages",
                {"workload": {"included_stages": list(END_TO_END_STAGES[:-1]) + [7]}},
            ),
        )

        for path, overrides in invalid_cases:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ContractError, path):
                    validate_receipt(receipt(**overrides))

    def test_requires_wall_clock_elapsed_source(self):
        with self.assertRaisesRegex(ContractError, "timing.elapsed_source"):
            validate_receipt(receipt(timing={"elapsed_source": "summed_stages"}))

    def test_rejects_end_to_end_receipt_missing_canonical_stages(self):
        raw = receipt(workload={"included_stages": ["environment_step"]})

        with self.assertRaisesRegex(ContractError, "included_stages"):
            validate_receipt(raw)

    def test_async_stage_seconds_can_exceed_elapsed_when_elapsed_is_wall_clock(self):
        raw = receipt(
            timing={
                "elapsed_source": "wall_clock",
                "elapsed_seconds": 10.0,
                "stage_seconds": {"rollout": 8.0, "train": 7.0},
            },
            workload={"overlap": {"mode": "async", "evidence": ["event_timeline"]}},
        )

        normalized = validate_receipt(raw)

        self.assertEqual(100.0, normalized["measurement"]["derived_rate"])
        self.assertEqual(15.0, sum(normalized["timing"]["stage_seconds"].values()))

    def test_rejects_async_elapsed_source_summed_stages(self):
        raw = receipt(
            timing={
                "elapsed_source": "summed_stages",
                "elapsed_seconds": 15.0,
                "stage_seconds": {"rollout": 8.0, "train": 7.0},
            },
            workload={"overlap": {"mode": "async", "evidence": ["cuda_trace"]}},
        )

        with self.assertRaisesRegex(ContractError, "timing.elapsed_source"):
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

    def test_throughput_mismatch_classes_block_speedup(self):
        cases = (
            ("workload.warmup_policy", {"workload": {"warmup_policy": {"mode": "none"}}}),
            (
                "workload.synchronization",
                {"workload": {"synchronization": {"device": "cuda", "method": "event"}}},
            ),
            ("system.hardware.accelerator_count", {"system": {"hardware": {"accelerator_count": 2}}}),
            (
                "workload.environment.action_contract",
                {"workload": {"environment": {"action_contract": "discrete_22_v1"}}},
            ),
            (
                "workload.environment.tick_semantics",
                {"workload": {"environment": {"tick_semantics": "one_tick_per_env_step"}}},
            ),
            (
                "workload.environment.terminal_bootstrap",
                {"workload": {"environment": {"terminal_bootstrap": "zero_terminal"}}},
            ),
            ("workload.rollout.horizon", {"workload": {"rollout": {"horizon": 256}}}),
            (
                "workload.rollout.transitions_per_batch",
                {"workload": {"rollout": {"transitions_per_batch": 262144}}},
            ),
            ("algorithm.sample_reuse_epochs", {"algorithm": {"sample_reuse_epochs": 3}}),
        )

        for blocker, overrides in cases:
            with self.subTest(blocker=blocker):
                result = comparison_with(overrides)

                self.assertFalse(result["throughput_comparable"])
                self.assertIn(blocker, result["throughput_blockers"])
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
        result = comparison_with(
            {
                "algorithm": {"learning_rate": 0.001},
                "timing": {"elapsed_seconds": 5.0},
                "measurement": {"claimed_rate": 200.0},
            }
        )

        self.assertTrue(result["throughput_comparable"])
        self.assertFalse(result["learning_behavior_comparable"])
        self.assertIn("algorithm.learning_rate", result["learning_behavior_blockers"])
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(2.0, result["speedup_ratio"])

    def test_actor_staleness_blocks_learning_comparison(self):
        result = comparison_with({"algorithm": {"actor_staleness": "one_batch_stale"}})

        self.assertTrue(result["throughput_comparable"])
        self.assertFalse(result["learning_behavior_comparable"])
        self.assertIn("algorithm.actor_staleness", result["learning_behavior_blockers"])

    def test_learning_mismatch_classes_do_not_block_system_speedup(self):
        cases = (
            ("algorithm.optimizer", {"algorithm": {"optimizer": "sgd"}}),
            ("algorithm.gamma", {"algorithm": {"gamma": 0.98}}),
            ("algorithm.gae_lambda", {"algorithm": {"gae_lambda": 0.9}}),
            ("algorithm.clip_range", {"algorithm": {"clip_range": 0.1}}),
            ("algorithm.entropy_coef", {"algorithm": {"entropy_coef": 0.0}}),
            ("algorithm.value_coef", {"algorithm": {"value_coef": 0.25}}),
            ("algorithm.max_grad_norm", {"algorithm": {"max_grad_norm": 1.0}}),
            ("algorithm.replay_semantics", {"algorithm": {"replay_semantics": "replay_buffer"}}),
            (
                "algorithm.advantage_normalization",
                {"algorithm": {"advantage_normalization": "none"}},
            ),
            ("algorithm.terminal_bootstrap", {"algorithm": {"terminal_bootstrap": "zero_terminal"}}),
        )

        for blocker, overrides in cases:
            with self.subTest(blocker=blocker):
                result = compare_receipts(
                    receipt(),
                    receipt(
                        **deep_update(
                            overrides,
                            {
                                "timing": {"elapsed_seconds": 5.0},
                                "measurement": {"claimed_rate": 200.0},
                            },
                        )
                    ),
                )

                self.assertTrue(result["throughput_comparable"])
                self.assertFalse(result["learning_behavior_comparable"])
                self.assertIn(blocker, result["learning_behavior_blockers"])
                self.assertFalse(result["promotion_eligible"])
                self.assertEqual(2.0, result["speedup_ratio"])

    def test_async_evidence_mismatch_blocks_throughput(self):
        result = comparison_with(
            {"workload": {"overlap": {"mode": "async", "evidence": ["event_timeline"]}}}
        )

        self.assertFalse(result["throughput_comparable"])
        self.assertIn("workload.overlap.mode", result["throughput_blockers"])
        self.assertIn("workload.overlap.evidence", result["throughput_blockers"])

    def test_missing_correctness_or_quality_evidence_blocks_promotion(self):
        result = comparison_with(
            {
                "correctness": {"fixed_wall_clock_quality": False},
                "timing": {"elapsed_seconds": 5.0},
                "measurement": {"claimed_rate": 200.0},
            }
        )

        self.assertTrue(result["throughput_comparable"])
        self.assertTrue(result["learning_behavior_comparable"])
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(["correctness.fixed_wall_clock_quality"], result["promotion_blockers"])
        self.assertEqual(2.0, result["speedup_ratio"])

    def test_each_promotion_evidence_field_is_required_individually(self):
        for field in (
            "env_trace_parity",
            "rollout_parity",
            "fixed_transition_quality",
            "fixed_wall_clock_quality",
        ):
            with self.subTest(field=field):
                result = comparison_with({"correctness": {field: False}})

                self.assertTrue(result["throughput_comparable"])
                self.assertTrue(result["learning_behavior_comparable"])
                self.assertFalse(result["promotion_eligible"])
                self.assertEqual([f"correctness.{field}"], result["promotion_blockers"])
                self.assertEqual(1.0, result["speedup_ratio"])


if __name__ == "__main__":
    unittest.main()
