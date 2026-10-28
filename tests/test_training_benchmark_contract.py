import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import tools.validation.training_benchmark_contract as contract
from tools.validation.training_benchmark_contract import (
    ContractError,
    END_TO_END_STAGES,
    SCHEMA_VERSION,
    ai2_native_receipt,
    compare_receipts,
    validate_receipt,
)


VALID_SEMANTICS_HASH = "e659a10eed28048afd16823b32ded9eabe872d7c3438f0cdabd895259336925c"
OTHER_SEMANTICS_HASH = "9f1b9da8b1aea1e377f00ed8e7ce9e8d4bce6f24a8a12d99d148633f37b9010e"


THROUGHPUT_MISMATCH_CASES = {
    "workload.timing_scope": {
        "workload": {"timing_scope": "env_only", "included_stages": ["environment_step"]}
    },
    "workload.included_stages": {
        "workload": {"timing_scope": "env_only", "included_stages": ["environment_step"]}
    },
    "workload.warmup_policy": {"workload": {"warmup_policy": {"mode": "none"}}},
    "workload.synchronization": {
        "workload": {"synchronization": {"device": "cuda", "method": "event"}}
    },
    "workload.overlap.mode": {
        "workload": {"overlap": {"mode": "async", "evidence": ["event_timeline"]}}
    },
    "workload.overlap.evidence": {
        "workload": {"overlap": {"mode": "async", "evidence": ["event_timeline"]}}
    },
    "system.hardware.accelerator_model": {
        "system": {"hardware": {"accelerator_model": "RTX 3090"}}
    },
    "system.hardware.accelerator_count": {
        "system": {"hardware": {"accelerator_count": 2}}
    },
    "system.hardware.cpu_model": {"system": {"hardware": {"cpu_model": "other-cpu"}}},
    "workload.environment.semantics_hash": {
        "workload": {"environment": {"semantics_hash": OTHER_SEMANTICS_HASH}}
    },
    "workload.environment.env_id": {
        "workload": {"environment": {"env_id": "OtherBrawl"}}
    },
    "workload.environment.observation_contract": {
        "workload": {"environment": {"observation_contract": "puffer_observation_v2"}}
    },
    "workload.environment.action_contract": {
        "workload": {"environment": {"action_contract": "discrete_22_v1"}}
    },
    "workload.environment.tick_semantics": {
        "workload": {"environment": {"tick_semantics": "one_tick_per_env_step"}}
    },
    "workload.environment.terminal_bootstrap": {
        "workload": {"environment": {"terminal_bootstrap": "zero_terminal"}}
    },
    "workload.model.architecture": {"workload": {"model": {"architecture": "cnn"}}},
    "workload.model.parameter_count": {"workload": {"model": {"parameter_count": 999}}},
    "workload.model.precision": {"workload": {"model": {"precision": "tf32"}}},
    "workload.rollout.num_envs": {"workload": {"rollout": {"num_envs": 2048}}},
    "workload.rollout.horizon": {"workload": {"rollout": {"horizon": 256}}},
    "workload.rollout.transitions_per_batch": {
        "workload": {"rollout": {"transitions_per_batch": 262144}}
    },
    "workload.quality_target": {
        "workload": {
            "timing_scope": "time_to_quality",
            "quality_target": {
                "metric": "elo",
                "threshold": 1200.0,
                "direction": "at_least",
                "evaluation_protocol": "fixed_seed_eval_v2",
                "aggregation": "mean",
            },
        }
    },
    "algorithm.minibatches": {"algorithm": {"minibatches": 16}},
    "algorithm.ppo_epochs": {"algorithm": {"ppo_epochs": 2}},
    "algorithm.sample_reuse_epochs": {"algorithm": {"sample_reuse_epochs": 3}},
}


def receipt(**overrides):
    base = {
        "schema_version": SCHEMA_VERSION,
        "system": {
            "hardware": {
                "accelerator_model": "RTX 4090",
                "accelerator_count": 1,
                "cpu_model": "test-cpu",
            },
            "software": {
                "command": "python benchmark.py --measured",
                "runtime": {
                    "python": "3.10.19",
                    "torch": "2.4.1",
                    "cuda": "12.4",
                },
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
                "semantics_hash": VALID_SEMANTICS_HASH,
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
            "source_provenance": {
                "runner": "benchmark.py",
                "sources": {"benchmark.py": "source-sha256-test"},
                "command": "python benchmark.py --measured",
                "runtime": {
                    "python": "3.10.19",
                    "torch": "2.4.1",
                    "cuda": "12.4",
                },
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
    if not isinstance(overrides, dict):
        if isinstance(overrides, list):
            return list(overrides)
        return overrides
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
        for counter in (
            "StepS",
            "SPS",
            "Game Ticks",
            "Game_Ticks",
            "Episodes",
            "Optimizer Calls",
            "Optimizer_Calls",
        ):
            with self.subTest(counter=counter):
                raw = receipt(measurement={counter: 1000})

                with self.assertRaisesRegex(ContractError, "ambiguous"):
                    validate_receipt(raw)

    def test_rejects_missing_required_fields(self):
        raw = receipt()
        del raw["workload"]["warmup_policy"]

        with self.assertRaisesRegex(ContractError, "workload.warmup_policy"):
            validate_receipt(raw)

    def test_requires_cpu_model_for_hardware_comparison(self):
        raw = receipt()
        del raw["system"]["hardware"]["cpu_model"]

        with self.assertRaisesRegex(ContractError, "system.hardware.cpu_model"):
            validate_receipt(raw)

    def test_rejects_unknown_sentinel_cpu_models(self):
        for value in ("unknown", "unknown-cpu-model", "not available", "n/a", "generic"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ContractError, "system.hardware.cpu_model"):
                    validate_receipt(receipt(system={"hardware": {"cpu_model": value}}))

    def test_requires_real_sha256_environment_semantics_hash(self):
        invalid_values = (
            None,
            "",
            "   ",
            "TODO",
            "placeholder",
            "unknown",
            "n/a",
            "0" * 64,
            "g" * 64,
            "abc123",
            True,
            123,
        )
        for value in invalid_values:
            with self.subTest(value=value):
                raw = receipt()
                raw["workload"]["environment"]["semantics_hash"] = value
                with self.assertRaisesRegex(ContractError, "workload.environment.semantics_hash"):
                    validate_receipt(raw)

        raw = receipt()
        raw["workload"]["environment"]["semantics_hash"] = VALID_SEMANTICS_HASH.upper()
        normalized = validate_receipt(raw)
        self.assertEqual(VALID_SEMANTICS_HASH, normalized["workload"]["environment"]["semantics_hash"])

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

    def test_rejects_invalid_ppo_numeric_hyperparameters(self):
        invalid_cases = (
            ("algorithm.learning_rate", {"algorithm": {"learning_rate": 0}}),
            ("algorithm.learning_rate", {"algorithm": {"learning_rate": True}}),
            ("algorithm.gamma", {"algorithm": {"gamma": -0.01}}),
            ("algorithm.gamma", {"algorithm": {"gamma": 1.01}}),
            ("algorithm.gae_lambda", {"algorithm": {"gae_lambda": -0.01}}),
            ("algorithm.gae_lambda", {"algorithm": {"gae_lambda": 1.01}}),
            ("algorithm.clip_range", {"algorithm": {"clip_range": 0}}),
            ("algorithm.clip_range", {"algorithm": {"clip_range": 1.01}}),
            ("algorithm.entropy_coef", {"algorithm": {"entropy_coef": -0.01}}),
            ("algorithm.value_coef", {"algorithm": {"value_coef": -0.01}}),
            ("algorithm.max_grad_norm", {"algorithm": {"max_grad_norm": 0}}),
            ("finite", {"algorithm": {"max_grad_norm": math.inf}}),
        )

        for path, overrides in invalid_cases:
            with self.subTest(path=path, overrides=overrides):
                with self.assertRaisesRegex(ContractError, path):
                    validate_receipt(receipt(**overrides))

    def test_requires_dicts_for_policy_synchronization_and_overlap_fields(self):
        invalid_cases = (
            ("workload.warmup_policy", {"workload": {"warmup_policy": "none"}}),
            ("workload.synchronization", {"workload": {"synchronization": "sync"}}),
            ("workload.overlap", {"workload": {"overlap": "none"}}),
            (
                "workload.overlap.evidence",
                {"workload": {"overlap": {"mode": "none", "evidence": ["ok", 7]}}},
            ),
            (
                "workload.overlap.evidence",
                {"workload": {"overlap": {"mode": "async", "evidence": ["cuda_trace", "config_only"]}}},
            ),
        )

        for path, overrides in invalid_cases:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ContractError, path):
                    validate_receipt(receipt(**overrides))

    def test_rejects_semantically_empty_provenance_and_protocol_objects(self):
        invalid_cases = (
            ("workload.warmup_policy", {"workload": {"warmup_policy": {"mode": ""}}}),
            ("workload.warmup_policy", {"workload": {"warmup_policy": {"mode": False, "iterations": 1}}}),
            (
                "workload.synchronization",
                {"workload": {"synchronization": {"device": "cuda", "method": None}}},
            ),
            (
                "workload.synchronization",
                {"workload": {"synchronization": {"device": False, "method": "event"}}},
            ),
            (
                "workload.source_provenance.sources",
                {"workload": {"source_provenance": {"sources": {"runner.py": False}}}},
            ),
            ("system.software.command", {"system": {"software": {"command": ""}}}),
            (
                "workload.source_provenance.command",
                {"workload": {"source_provenance": {"command": None}}},
            ),
        )

        for expected, overrides in invalid_cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ContractError, expected):
                    validate_receipt(receipt(**overrides))

        empty_cases = (
            ("workload.warmup_policy", lambda raw: raw["workload"].update(warmup_policy={})),
            (
                "workload.warmup_policy",
                lambda raw: raw["workload"].update(warmup_policy={"mode": "discard_first"}),
            ),
            ("workload.synchronization", lambda raw: raw["workload"].update(synchronization={})),
            ("system.software.runtime", lambda raw: raw["system"]["software"].update(runtime={})),
            (
                "system.software.runtime",
                lambda raw: raw["system"]["software"].update(runtime={"placeholder": "filled"}),
            ),
            (
                "system.software.runtime",
                lambda raw: raw["system"]["software"].update(runtime={"matmul_tf32": False}),
            ),
            (
                "workload.source_provenance.sources",
                lambda raw: raw["workload"]["source_provenance"].update(sources={}),
            ),
        )
        for expected, corrupt in empty_cases:
            with self.subTest(expected=expected):
                raw = receipt()
                corrupt(raw)
                with self.assertRaisesRegex(ContractError, expected):
                    validate_receipt(raw)

    def test_rejects_unknown_timing_scope_overlap_mode_and_bad_stages(self):
        invalid_cases = (
            (
                "workload.timing_scope",
                {"workload": {"timing_scope": "profile_only"}},
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

    def test_accepts_all_allowed_timing_scopes_with_explicit_stages(self):
        cases = {
            "env_only": ["environment_step"],
            "collection": [
                "policy_inference",
                "action_sampling",
                "environment_step",
                "rollout_storage",
            ],
            "update": ["returns_and_advantages", "ppo_update", "device_synchronization"],
            "end_to_end": list(END_TO_END_STAGES),
        }

        for scope, stages in cases.items():
            with self.subTest(scope=scope):
                normalized = validate_receipt(
                    receipt(workload={"timing_scope": scope, "included_stages": stages})
                )

                self.assertEqual(scope, normalized["workload"]["timing_scope"])
                self.assertEqual(stages, normalized["workload"]["included_stages"])

    def test_time_to_quality_requires_quality_target(self):
        with self.assertRaisesRegex(ContractError, "workload.quality_target"):
            validate_receipt(
                receipt(
                    workload={
                        "timing_scope": "time_to_quality",
                        "included_stages": list(END_TO_END_STAGES),
                    }
                )
            )

    def test_time_to_quality_accepts_minimal_quality_target_schema(self):
        target = {
            "metric": "elo",
            "threshold": 1000.0,
            "direction": "at_least",
            "evaluation_protocol": "fixed_seed_eval_v1",
            "aggregation": "mean",
        }

        normalized = validate_receipt(
            receipt(
                workload={
                    "timing_scope": "time_to_quality",
                    "included_stages": list(END_TO_END_STAGES),
                    "quality_target": target,
                }
            )
        )

        self.assertEqual(target, normalized["workload"]["quality_target"])

    def test_time_to_quality_rejects_invalid_quality_target_schema(self):
        invalid_cases = (
            (
                "workload.quality_target",
                {"metric": "", "threshold": 1000.0, "direction": "at_least", "evaluation_protocol": "fixed", "aggregation": "mean"},
            ),
            (
                "workload.quality_target",
                {"metric": "elo", "threshold": True, "direction": "at_least", "evaluation_protocol": "fixed", "aggregation": "mean"},
            ),
            (
                "finite",
                {"metric": "elo", "threshold": math.inf, "direction": "at_least", "evaluation_protocol": "fixed", "aggregation": "mean"},
            ),
            (
                "workload.quality_target",
                {"metric": "elo", "threshold": 1000.0, "direction": "higher", "evaluation_protocol": "fixed", "aggregation": "mean"},
            ),
            (
                "workload.quality_target",
                {"metric": "elo", "threshold": 1000.0, "direction": "at_least", "evaluation_protocol": "", "aggregation": "mean"},
            ),
            (
                "workload.quality_target",
                {"metric": "elo", "threshold": 1000.0, "direction": "at_least", "evaluation_protocol": "fixed", "aggregation": "median"},
            ),
            ("workload.quality_target", "elo>=1000"),
        )

        for expected, target in invalid_cases:
            with self.subTest(target=target):
                with self.assertRaisesRegex(ContractError, expected):
                    validate_receipt(
                        receipt(
                            workload={
                                "timing_scope": "time_to_quality",
                                "included_stages": list(END_TO_END_STAGES),
                                "quality_target": target,
                            }
                        )
                    )

    def test_other_timing_scopes_do_not_require_quality_target(self):
        normalized = validate_receipt(
            receipt(
                workload={
                    "timing_scope": "collection",
                    "included_stages": ["policy_inference", "environment_step"],
                }
            )
        )

        self.assertNotIn("quality_target", normalized["workload"])

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

    def test_deep_update_replaces_dict_subtree_with_non_dict_override(self):
        raw = receipt(workload={"quality_target": None})

        self.assertIsNone(raw["workload"]["quality_target"])

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
        self.assertEqual(100.0, result["left_agent_transitions_per_second"])
        self.assertEqual(200.0, result["right_agent_transitions_per_second"])
        self.assertEqual(2.0, result["speedup_ratio"])
        self.assertEqual([], result["throughput_blockers"])
        self.assertEqual([], result["learning_blockers"])
        self.assertEqual([], result["promotion_blockers"])

    def test_every_throughput_blocker_path_has_direct_mismatch_coverage(self):
        self.assertEqual(
            set(contract._THROUGHPUT_PATHS),
            set(THROUGHPUT_MISMATCH_CASES),
        )

    def test_throughput_mismatch_classes_block_speedup(self):
        for blocker, overrides in THROUGHPUT_MISMATCH_CASES.items():
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
                    "environment": {"semantics_hash": OTHER_SEMANTICS_HASH}
                }
            }
        )

        self.assertFalse(result["throughput_comparable"])
        self.assertFalse(result["learning_behavior_comparable"])
        self.assertIn(
            "workload.environment.semantics_hash", result["throughput_blockers"]
        )
        self.assertIn(
            "workload.environment.semantics_hash",
            result["learning_blockers"],
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
        self.assertIn("algorithm.learning_rate", result["learning_blockers"])
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(2.0, result["speedup_ratio"])

    def test_actor_staleness_blocks_learning_comparison(self):
        result = comparison_with({"algorithm": {"actor_staleness": "one_batch_stale"}})

        self.assertTrue(result["throughput_comparable"])
        self.assertFalse(result["learning_behavior_comparable"])
        self.assertIn("algorithm.actor_staleness", result["learning_blockers"])

    def test_learning_mismatch_classes_block_learning_comparison(self):
        cases = (
            ("workload.environment.env_id", {"workload": {"environment": {"env_id": "OtherBrawl"}}}),
            (
                "workload.environment.semantics_hash",
                {"workload": {"environment": {"semantics_hash": OTHER_SEMANTICS_HASH}}},
            ),
            (
                "workload.environment.observation_contract",
                {"workload": {"environment": {"observation_contract": "puffer_observation_v2"}}},
            ),
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
            ("workload.model.architecture", {"workload": {"model": {"architecture": "cnn"}}}),
            ("workload.model.parameter_count", {"workload": {"model": {"parameter_count": 999}}}),
            ("workload.model.precision", {"workload": {"model": {"precision": "tf32"}}}),
            (
                "workload.quality_target",
                {
                    "workload": {
                        "timing_scope": "time_to_quality",
                        "quality_target": {
                            "metric": "elo",
                            "threshold": 1200.0,
                            "direction": "at_least",
                            "evaluation_protocol": "fixed_seed_eval_v2",
                            "aggregation": "mean",
                        },
                    }
                },
            ),
            ("workload.rollout.num_envs", {"workload": {"rollout": {"num_envs": 2048}}}),
            ("workload.rollout.horizon", {"workload": {"rollout": {"horizon": 256}}}),
            (
                "workload.rollout.transitions_per_batch",
                {"workload": {"rollout": {"transitions_per_batch": 262144}}},
            ),
            ("algorithm.optimizer", {"algorithm": {"optimizer": "sgd"}}),
            ("algorithm.gamma", {"algorithm": {"gamma": 0.98}}),
            ("algorithm.gae_lambda", {"algorithm": {"gae_lambda": 0.9}}),
            ("algorithm.clip_range", {"algorithm": {"clip_range": 0.1}}),
            ("algorithm.entropy_coef", {"algorithm": {"entropy_coef": 0.0}}),
            ("algorithm.value_coef", {"algorithm": {"value_coef": 0.25}}),
            ("algorithm.max_grad_norm", {"algorithm": {"max_grad_norm": 1.0}}),
            ("algorithm.sample_reuse_epochs", {"algorithm": {"sample_reuse_epochs": 3}}),
            ("algorithm.minibatches", {"algorithm": {"minibatches": 16}}),
            ("algorithm.ppo_epochs", {"algorithm": {"ppo_epochs": 2}}),
            ("algorithm.replay_semantics", {"algorithm": {"replay_semantics": "replay_buffer"}}),
            (
                "algorithm.advantage_normalization",
                {"algorithm": {"advantage_normalization": "none"}},
            ),
            ("algorithm.terminal_bootstrap", {"algorithm": {"terminal_bootstrap": "zero_terminal"}}),
        )

        for blocker, overrides in cases:
            with self.subTest(blocker=blocker):
                left = receipt()
                if blocker == "workload.quality_target":
                    left = receipt(
                        workload={
                            "timing_scope": "time_to_quality",
                            "quality_target": {
                                "metric": "elo",
                                "threshold": 1000.0,
                                "direction": "at_least",
                                "evaluation_protocol": "fixed_seed_eval_v1",
                                "aggregation": "mean",
                            },
                        }
                    )
                result = compare_receipts(
                    left,
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

                self.assertFalse(result["learning_behavior_comparable"])
                self.assertIn(blocker, result["learning_blockers"])
                self.assertFalse(result["promotion_eligible"])

    def test_placeholder_semantics_hashes_cannot_compare_as_equivalent(self):
        left = receipt()
        right = receipt()
        left["workload"]["environment"]["semantics_hash"] = "TODO"
        right["workload"]["environment"]["semantics_hash"] = "TODO"

        with self.assertRaisesRegex(ContractError, "workload.environment.semantics_hash"):
            compare_receipts(left, right)

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


def native_result(**overrides):
    config = {
        "games": 4,
        "n_steps": 8,
        "batch_size": 16,
        "n_epochs": 3,
        "net_arch": [128, 64],
        "warmup_rollouts": 1,
        "measured_rollouts": 2,
        "seed": 11,
        "device": "cuda",
        "python_workers": 1,
        "learning_rate": 0.00025,
        "gamma": 0.98,
        "gae_lambda": 0.94,
        "clip_range": 0.18,
        "ent_coef": 0.02,
        "vf_coef": 0.6,
        "max_grad_norm": 0.7,
        "backends": ["native_cuda"],
        "update_mode": "sb3",
    }
    row = {
        "backend": "native_cuda",
        "status": "passed",
        "summary": {
            "game_ticks": 64,
            "player_transitions": 128,
            "collection_seconds": 1.25,
            "update_seconds": 2.75,
            "total_seconds": 4.0,
            "collection_game_ticks_per_second": 51.2,
            "collection_player_transitions_per_second": 102.4,
            "total_game_ticks_per_second": 16.0,
            "total_player_transitions_per_second": 32.0,
        },
        "rollouts": [
            {
                "rollout": 0,
                "measured": False,
                "game_ticks": 32,
                "player_transitions": 64,
                "collection_seconds": 0.5,
                "update_seconds": 1.0,
                "total_seconds": 1.5,
                "losses_finite": True,
                "metrics": {
                    "train/loss": 1.0,
                    "train/policy_gradient_loss": 0.1,
                    "train/value_loss": 0.2,
                    "train/entropy_loss": -6.0,
                },
                "errors": [0, 0, 0, 0],
            },
            {
                "rollout": 1,
                "measured": True,
                "game_ticks": 32,
                "player_transitions": 64,
                "collection_seconds": 0.75,
                "update_seconds": 1.25,
                "total_seconds": 2.0,
                "losses_finite": True,
                "metrics": {
                    "train/loss": 0.9,
                    "train/policy_gradient_loss": 0.08,
                    "train/value_loss": 0.18,
                    "train/entropy_loss": -5.5,
                },
                "errors": [0, 0, 0, 0],
            },
            {
                "rollout": 2,
                "measured": True,
                "game_ticks": 32,
                "player_transitions": 64,
                "collection_seconds": 0.5,
                "update_seconds": 1.5,
                "total_seconds": 2.0,
                "losses_finite": True,
                "metrics": {
                    "train/loss": 0.8,
                    "train/policy_gradient_loss": 0.07,
                    "train/value_loss": 0.16,
                    "train/entropy_loss": -5.0,
                },
                "errors": [0, 0, 0, 0],
            },
        ],
        "initial_policy": {"sha256": "0" * 64, "parameter_count": 4242},
        "final_policy": {"sha256": "1" * 64, "parameter_count": 4242},
        "parameters_changed": True,
        "policy_architecture": "ignored because config is canonical",
        "checkpoint": "/tmp/policy_final.zip",
        "checkpoint_sha256": "2" * 64,
        "python_workers": 0,
        "baseline_scope": "native_cuda",
        "measured_work": "collection includes game, policy, transfers, GAE, episode drain; update includes PPO and sync",
    }
    base = {
        "schema": "ai2.raw_combat_training_benchmark.v1",
        "status": "passed",
        "config": config,
        "provenance": {
            "config": dict(config),
            "environment_sha256": VALID_SEMANTICS_HASH,
            "attack_assets_sha256": "a" * 64,
            "sources": {
                "benchmark_native_combat_training.py": "b" * 64,
                "native_game_lane_vec_env.py": "c" * 64,
                "tensor_ppo.py": "d" * 64,
                "tensor_rollout_buffer.py": "f" * 64,
                "native/navigation_spaces.json": "9" * 64,
            },
            "ppo_update": {
                "mode": "sb3",
                "training_algorithm": "installed stable_baselines3.PPO.train",
                "source_guard": {
                    "required": False,
                    "expected_version": "2.5.0",
                    "expected_train_sha256": "8" * 64,
                    "version": "2.5.0",
                    "train_sha256": "8" * 64,
                    "supported": True,
                },
            },
            "system": "Linux-6.8-test",
            "cpu_model": "AMD EPYC 9354P 32-Core Processor",
            "python": "3.10.19",
            "cpu_count": 32,
            "packages": {"torch": "2.4.1", "stable_baselines3": "2.5.0", "numpy": "2.1.1"},
            "cuda_version": "12.4",
            "gpu": "NVIDIA GeForce RTX 4090",
            "policy_device": "cuda:0",
            "torch_cpu_threads": 1,
            "torch_interop_threads": 1,
            "matmul_tf32": False,
            "workload": "raw game rewards, both players jointly train shared Gaussian MLP; Box10, four-frame float32[256]",
            "player_order": "game0/player0, game0/player1, ...",
            "game_seed_rule": "seed+2*game_index",
            "exclusions": "startup/build/initial reset/model initialization, logging, parameter hashing, checkpoint serialization excluded from timing; episode autoresets are timed",
            "limitation": "infrastructure only; not curriculum training or evidence of combat mastery",
        },
        "backends": [row],
    }
    return deep_update(base, overrides)


class NativeAdapterTests(unittest.TestCase):
    def test_ai2_native_receipt_maps_raw_benchmark_to_canonical_contract(self):
        raw = native_result()

        receipt = ai2_native_receipt(raw, "native_cuda")
        normalized = validate_receipt(receipt)

        self.assertEqual("agent_transition", normalized["workload"]["unit"])
        self.assertEqual(128, normalized["measurement"]["agent_transitions"])
        self.assertEqual(4.0, normalized["timing"]["elapsed_seconds"])
        self.assertEqual(32.0, normalized["measurement"]["derived_rate"])
        self.assertEqual("NVIDIA GeForce RTX 4090", normalized["system"]["hardware"]["accelerator_model"])
        self.assertEqual(1, normalized["system"]["hardware"]["accelerator_count"])
        self.assertEqual(8, normalized["workload"]["rollout"]["num_envs"])
        self.assertEqual(8, normalized["workload"]["rollout"]["horizon"])
        self.assertEqual(64, normalized["workload"]["rollout"]["transitions_per_batch"])
        self.assertEqual(4242, normalized["workload"]["model"]["parameter_count"])
        self.assertEqual([128, 64], normalized["workload"]["model"]["architecture"]["policy_layers"])
        self.assertEqual([128, 64], normalized["workload"]["model"]["architecture"]["value_layers"])
        self.assertEqual(4, normalized["algorithm"]["minibatches"])
        self.assertEqual(3, normalized["algorithm"]["ppo_epochs"])
        self.assertEqual(3, normalized["algorithm"]["sample_reuse_epochs"])
        self.assertEqual(
            {
                "type": "box",
                "shape": [10],
                "dtype": "float32",
                "range": [0.0, 1.0],
                "button_threshold": 0.5,
                "buttons": [
                    "w/up",
                    "a/left",
                    "s/down",
                    "d/right",
                    "space/jump",
                    "h/pickup-drop",
                    "l/dash-dodge",
                    "j/light",
                    "k/heavy",
                    "g/taunt",
                ],
            },
            normalized["workload"]["environment"]["action_contract"],
        )
        self.assertEqual([0.0, 1.0], normalized["workload"]["environment"]["action_contract"]["range"])
        self.assertEqual(
            {
                "type": "diagonal_gaussian",
                "shape": [10],
                "postprocess": "clamp_to_box_then_threshold_buttons",
            },
            normalized["workload"]["model"]["raw_action_distribution"],
        )
        self.assertEqual("AMD EPYC 9354P 32-Core Processor", normalized["system"]["hardware"]["cpu_model"])
        self.assertEqual(VALID_SEMANTICS_HASH, normalized["workload"]["environment"]["semantics_hash"])
        self.assertEqual(
            "benchmark_native_combat_training.py",
            normalized["workload"]["source_provenance"]["runner"],
        )
        self.assertEqual(
            "config-backed ai2 native benchmark invocation",
            normalized["system"]["software"]["command"],
        )
        self.assertEqual(
            "3.10.19",
            normalized["system"]["software"]["runtime"]["python"],
        )
        self.assertEqual(
            {
                "env_trace_parity": False,
                "rollout_parity": False,
                "fixed_transition_quality": False,
                "fixed_wall_clock_quality": False,
            },
            normalized["correctness"],
        )
        self.assertEqual("not_run", normalized["correctness_evidence"]["env_trace_parity"]["status"])

    def test_ai2_native_receipt_fails_closed_for_invalid_raw_results(self):
        cases = (
            ("top-level benchmark", lambda raw: raw.update(status="failed")),
            ("backend native_cuda", lambda raw: raw["backends"][0].update(status="failed")),
            ("parameters changed", lambda raw: raw["backends"][0].update(parameters_changed=False)),
            ("policy parameters did not change", lambda raw: raw["backends"][0]["final_policy"].update(sha256="0" * 64)),
            ("finite losses", lambda raw: raw["backends"][0]["rollouts"][1].update(losses_finite=False)),
            ("rollout errors", lambda raw: raw["backends"][0]["rollouts"][1].update(errors=[0, 1, 0, 0])),
            ("exact player transitions", lambda raw: raw["backends"][0]["rollouts"][1].update(player_transitions=63)),
            ("summary totals", lambda raw: raw["backends"][0]["summary"].update(player_transitions=127)),
            ("provenance.sources", lambda raw: raw["provenance"]["sources"].clear()),
            ("provenance.gpu", lambda raw: raw["provenance"].update(gpu="")),
            ("provenance.cpu_model", lambda raw: raw["provenance"].update(cpu_model="")),
            ("provenance.environment_sha256", lambda raw: raw["provenance"].update(environment_sha256="")),
            ("config", lambda raw: raw["provenance"].update(config={"games": 99})),
            ("rollout rows", lambda raw: raw["backends"][0]["rollouts"].__setitem__(1, "not-a-row")),
        )

        for expected, corrupt in cases:
            with self.subTest(expected=expected):
                raw = native_result()
                corrupt(raw)
                with self.assertRaisesRegex(ContractError, expected):
                    ai2_native_receipt(raw, "native_cuda")

    def test_ai2_native_receipt_accepts_zero_warmup_and_zero_optional_coefficients(self):
        raw = native_result()
        raw["config"].update(warmup_rollouts=0, ent_coef=0.0, vf_coef=0.0)
        raw["provenance"]["config"] = dict(raw["config"])
        backend = raw["backends"][0]
        backend["rollouts"] = [dict(row, rollout=index, measured=True) for index, row in enumerate(backend["rollouts"][1:])]

        receipt = ai2_native_receipt(raw, "native_cuda")

        self.assertEqual(0, receipt["workload"]["warmup_policy"]["rollouts"])
        self.assertEqual(0.0, receipt["algorithm"]["entropy_coef"])
        self.assertEqual(0.0, receipt["algorithm"]["value_coef"])


class CLITests(unittest.TestCase):
    def test_validate_compare_and_convert_ai2_commands_write_atomic_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left.json"
            right = root / "right.json"
            comparison = root / "comparison.json"
            raw = root / "result.json"
            converted = root / "receipt.json"
            left.write_text(json.dumps(receipt()), encoding="utf-8")
            right.write_text(json.dumps(receipt(measurement={"agent_transitions": 2000, "claimed_rate": 200.0})), encoding="utf-8")
            raw.write_text(json.dumps(native_result()), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(0, contract.main(["validate", str(left)]))
            self.assertEqual(100.0, json.loads(stdout.getvalue())["measurement"]["derived_rate"])

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(0, contract.main(["compare", str(left), str(right)]))
            printed = json.loads(stdout.getvalue())
            self.assertTrue(printed["throughput_comparable"])
            self.assertEqual(2.0, printed["speedup_ratio"])

            self.assertEqual(
                0,
                contract.main([
                    "compare",
                    str(left),
                    str(right),
                    "--output",
                    str(comparison),
                ]),
            )
            written = json.loads(comparison.read_text(encoding="utf-8"))
            self.assertEqual(2.0, written["speedup_ratio"])

            self.assertEqual(
                0,
                contract.main([
                    "convert-ai2",
                    str(raw),
                    "--backend",
                    "native_cuda",
                    "--output",
                    str(converted),
                ]),
            )
            converted_receipt = validate_receipt(json.loads(converted.read_text(encoding="utf-8")))
            self.assertEqual("native_cuda", converted_receipt["workload"]["source_provenance"]["backend"])
            self.assertEqual(128, converted_receipt["measurement"]["agent_transitions"])

    def test_cli_contract_errors_are_structured_for_bad_input_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.json"
            invalid = root / "invalid.json"
            invalid.write_text("{not json", encoding="utf-8")

            for path in (missing, invalid):
                with self.subTest(path=path.name):
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        self.assertEqual(contract.CLI_CONTRACT_ERROR, contract.main(["validate", str(path)]))
                    payload = json.loads(stdout.getvalue())
                    self.assertIn("error", payload)
                    self.assertNotIn("Traceback", stdout.getvalue())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support required")
    def test_cli_output_tempfile_is_unique_exclusive_and_does_not_clobber_old_tmp_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "receipt.json"
            output = root / "validated.json"
            victim = root / "victim.txt"
            legacy_tmp = root / "validated.json.tmp"
            source.write_text(json.dumps(receipt()), encoding="utf-8")
            victim.write_text("preserve-victim", encoding="utf-8")
            os.symlink(victim, legacy_tmp)

            self.assertEqual(0, contract.main(["validate", str(source), "--output", str(output)]))

            self.assertEqual("preserve-victim", victim.read_text(encoding="utf-8"))
            self.assertTrue(legacy_tmp.is_symlink())
            self.assertTrue(output.is_file())
            validate_receipt(json.loads(output.read_text(encoding="utf-8")))

    def test_cli_output_publish_refuses_destination_created_after_precheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "receipt.json"
            output = root / "validated.json"
            source.write_text(json.dumps(receipt()), encoding="utf-8")

            def racing_link(src, dst):
                Path(dst).write_text("racer-created", encoding="utf-8")
                raise FileExistsError(dst)

            stdout = io.StringIO()
            with patch.object(contract.os, "link", side_effect=racing_link), redirect_stdout(stdout):
                self.assertEqual(
                    contract.CLI_OUTPUT_EXISTS,
                    contract.main(["validate", str(source), "--output", str(output)]),
                )

            self.assertIn("refusing to overwrite", json.loads(stdout.getvalue())["error"])
            self.assertEqual("racer-created", output.read_text(encoding="utf-8"))
            self.assertEqual([], list(root.glob(".validated.json.*.tmp")))

    def test_cli_refuses_overwrite_and_marks_non_comparable_pair_distinctly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left.json"
            right = root / "right.json"
            existing = root / "existing.json"
            left.write_text(json.dumps(receipt()), encoding="utf-8")
            right.write_text(
                json.dumps(receipt(system={"hardware": {"accelerator_model": "RTX 3090"}})),
                encoding="utf-8",
            )
            existing.write_text("do-not-replace", encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    contract.CLI_OUTPUT_EXISTS,
                    contract.main([
                        "validate",
                        str(left),
                        "--output",
                        str(existing),
                    ]),
                )
            self.assertIn("refusing to overwrite", json.loads(stdout.getvalue())["error"])
            self.assertEqual("do-not-replace", existing.read_text(encoding="utf-8"))

            result_code = contract.main([
                "compare",
                str(left),
                str(right),
                "--output",
                str(root / "non_comparable.json"),
            ])
            self.assertEqual(contract.CLI_NON_COMPARABLE, result_code)
            comparison = json.loads((root / "non_comparable.json").read_text(encoding="utf-8"))
            self.assertFalse(comparison["throughput_comparable"])
            self.assertIn("system.hardware.accelerator_model", comparison["throughput_blockers"])


if __name__ == "__main__":
    unittest.main()
