import copy
import json
import tempfile
import types
import unittest
from pathlib import Path

import tools.validation.fixed_action_trace_parity as parity
from tools.validation.fixed_action_trace_parity import (
    build_fixed_action_suite,
    build_receipt,
    compare_trace_pair,
    run_suite,
    three_way_exact_allowed,
)


def exact_autoreset():
    return {
        "status": "exact",
        "same_step_autoreset": True,
        "same_step_reset_obs_exact": True,
        "healthy_subsequent_step": True,
    }


def exact_cuda_stream():
    return {
        "mode": "non_default",
        "used_non_default_stream": True,
        "synchronized_before_compare": True,
    }


def exact_cuda_observation_checks():
    return {
        "reset_observation_exact": True,
        "step_observation_exact": True,
        "reward_exact": True,
        "terminated_exact": True,
        "truncated_exact": True,
        "terminal_observation_exact": True,
        "reset_observation_sha256": "reset",
        "terminal_observation_sha256": "terminal",
    }


def exact_cuda_autoreset():
    return {
        **exact_autoreset(),
        "resident_cuda_recorded_replay_exact": True,
        "resident_cuda_stream": exact_cuda_stream(),
        "resident_cuda_observation_checks": exact_cuda_observation_checks(),
    }


class FixedActionTraceParityTests(unittest.TestCase):
    def test_resident_cpu_replay_executes_default_stream_scope(self):
        from unittest.mock import patch
        try:
            import torch
            import ai2.simulation.native_game_lane_vec_env as native_game_lane_vec_env
        except ImportError as error:
            self.skipTest(str(error))

        backend = types.SimpleNamespace(
            torch=torch, device=torch.device("cpu"), kind="cpu",
            valid_steps=torch.zeros(1, dtype=torch.int64),
        )
        calls = []

        def reset(lane, draws, observation):
            calls.append((lane, draws))
            observation.zero_()

        backend.replay_reset = reset
        backend.drain = lambda: None
        backend.close = lambda: calls.append("closed")
        document = {"reset": {"draws": [], "outputs": {
            "obs": {"0": [0.0] * 64, "1": [0.0] * 64}}}, "ticks": []}
        with patch.object(native_game_lane_vec_env, "GameLaneBackend", return_value=backend):
            result = parity._run_resident_replay_trace({}, document, "cpu")
        self.assertTrue(result["resident_cpu_recorded_replay_exact"])
        self.assertEqual(calls, [(0, []), "closed"])

    def test_suite_covers_all_ten_actions(self):
        suite = build_fixed_action_suite(seeds=[101], horizon=12)
        self.assertEqual({case["action_index"] for case in suite["cases"]}, set(range(10)))
        self.assertTrue(all(case["horizon"] == 12 for case in suite["cases"]))

    def test_mismatch_is_blocker_not_warning(self):
        left = {"action_index": 6, "frames": [{"state_id": "dodge", "x": 0.0}]}
        right = {"action_index": 6, "frames": [{"state_id": "idle", "x": 0.0}]}
        result = compare_trace_pair(left, right, atol=0.0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["severity"], "blocker")
        self.assertIn("action_index=6", result["message"])

    def test_report_requires_high_risk_indices(self):
        suite = build_fixed_action_suite(seeds=[101], horizon=12)
        high_risk = {case["action_index"] for case in suite["cases"] if case["trace_required"]}
        self.assertTrue({5, 6, 7, 8, 9} <= high_risk)

    def test_receipt_separates_original_replay_from_resident_claims(self):
        cases = [
            {
                "action_index": action_index,
                "seed": 101,
                "horizon": 12,
                "python_reference_exact": True,
                "native_cpu_cuda_exact": None,
                "resident_cpu_recorded_replay_exact": True,
                "resident_cpu_generated_rng_recorded": True,
                "resident_lanes_replay_step_supported": False,
                "reset_compared": True,
                "compared_ticks": 12,
                "terminal_autoreset_evidence": "not_exercised",
                "rng": {"cursor": 7, "count": 7, "error": 0},
            }
            for action_index in range(10)
        ]
        receipt = build_receipt(
            cases=cases,
            backends=["python", "native_cpu"],
            expected_actions=list(range(10)),
            expected_seeds=[101],
            horizon=12,
        )
        self.assertTrue(receipt["python_reference_exact"])
        self.assertEqual(receipt["status"], "blocked")
        self.assertIsNone(receipt["native_cpu_cuda_exact"])
        self.assertTrue(receipt["resident_cpu_generated_rng_recorded"])
        self.assertFalse(receipt["resident_lanes_replay_step_supported"])
        self.assertFalse(receipt["three_way_exact"])
        self.assertEqual(receipt["promotion_status"], "blocked")
        self.assertIn("termination_autoreset_evidence_missing", {b["reason"] for b in receipt["blockers"]})

    def test_three_way_exact_requires_every_layer_and_exact_rng_consumption(self):
        complete = [
            {
                "action_index": action_index,
                "seed": seed,
                "horizon": 12,
                "python_reference_exact": True,
                "native_cpu_cuda_exact": True,
                "resident_cpu_recorded_replay_exact": True,
                "resident_cpu_generated_rng_recorded": True,
                "resident_lanes_replay_step_supported": True,
                "reset_compared": True,
                "compared_ticks": 12,
                "terminal_autoreset_evidence": "exact",
                "rng": {"cursor": 5, "count": 5, "error": 0},
            }
            for action_index in range(10)
            for seed in (101, 102)
        ]
        complete[0]["resident_termination_autoreset"] = exact_autoreset()
        complete[1]["resident_truncation_autoreset"] = exact_autoreset()
        self.assertTrue(
            three_way_exact_allowed(
                complete,
                expected_actions=list(range(10)),
                expected_seeds=[101, 102],
                horizon=12,
            )
        )
        for bad_update in (
            {"python_reference_exact": False},
            {"native_cpu_cuda_exact": None},
            {"resident_cpu_recorded_replay_exact": False},
            {"resident_cpu_generated_rng_recorded": False},
            {"resident_lanes_replay_step_supported": False},
            {"horizon": 11},
            {"reset_compared": False},
            {"compared_ticks": 0},
            {"compared_ticks": 11},
            {"terminal_autoreset_evidence": "missing"},
            {"rng": {"cursor": 4, "count": 5, "error": 0}},
            {"rng": {"cursor": 5, "count": 5, "error": 1}},
        ):
            bad = copy.deepcopy(complete)
            bad[0].update(bad_update)
            self.assertFalse(
                three_way_exact_allowed(
                    bad,
                    expected_actions=list(range(10)),
                    expected_seeds=[101, 102],
                    horizon=12,
                ),
                bad_update,
            )
        missing_seed = [case for case in complete if case["seed"] != 102]
        self.assertFalse(
            three_way_exact_allowed(
                missing_seed,
                expected_actions=list(range(10)),
                expected_seeds=[101, 102],
                horizon=12,
            )
        )
        missing_action = [case for case in complete if case["action_index"] != 9]
        self.assertFalse(
            three_way_exact_allowed(
                missing_action,
                expected_actions=list(range(10)),
                expected_seeds=[101, 102],
                horizon=12,
            )
        )
        missing_termination = copy.deepcopy(complete)
        missing_termination[0].pop("resident_termination_autoreset")
        self.assertFalse(
            three_way_exact_allowed(
                missing_termination,
                expected_actions=list(range(10)),
                expected_seeds=[101, 102],
                horizon=12,
            )
        )
        missing_truncation = copy.deepcopy(complete)
        missing_truncation[1].pop("resident_truncation_autoreset")
        self.assertFalse(
            three_way_exact_allowed(
                missing_truncation,
                expected_actions=list(range(10)),
                expected_seeds=[101, 102],
                horizon=12,
            )
        )
        stale_same_step_reset = copy.deepcopy(complete)
        stale_same_step_reset[0]["resident_termination_autoreset"] = {
            "status": "exact",
            "same_step_autoreset": True,
            "same_step_reset_obs_exact": False,
            "healthy_subsequent_step": True,
        }
        self.assertFalse(
            three_way_exact_allowed(
                stale_same_step_reset,
                expected_actions=list(range(10)),
                expected_seeds=[101, 102],
                horizon=12,
            )
        )

    def test_receipt_blocks_missing_python_or_resident_cpu_requested_evidence(self):
        base = {
            "action_index": 0,
            "seed": 101,
            "horizon": 12,
            "python_reference_exact": True,
            "native_cpu_cuda_exact": None,
            "resident_cpu_recorded_replay_exact": True,
            "resident_cpu_generated_rng_recorded": True,
            "resident_lanes_replay_step_supported": False,
            "reset_compared": True,
            "compared_ticks": 12,
            "terminal_autoreset_evidence": "not_exercised",
            "rng": {"cursor": 5, "count": 5, "error": 0},
        }
        no_python = build_receipt(
            cases=[{**base, "python_reference_exact": None}],
            backends=["native_cpu"],
            expected_actions=[0],
            expected_seeds=[101],
            horizon=12,
        )
        self.assertEqual(no_python["status"], "blocked")
        self.assertIn("python_reference_backend_missing", {b["reason"] for b in no_python["blockers"]})

        no_resident = build_receipt(
            cases=[{**base, "resident_cpu_generated_rng_recorded": False}],
            backends=["python", "native_cpu"],
            expected_actions=[0],
            expected_seeds=[101],
            horizon=12,
        )
        self.assertEqual(no_resident["status"], "blocked")
        self.assertIn("resident_cpu_generated_rng_missing", {b["reason"] for b in no_resident["blockers"]})

    def test_receipt_blocks_incomplete_per_case_evidence(self):
        base = {
            "action_index": 0,
            "seed": 101,
            "horizon": 12,
            "python_reference_exact": True,
            "native_cpu_cuda_exact": None,
            "resident_cpu_recorded_replay_exact": True,
            "resident_cpu_generated_rng_recorded": True,
            "resident_lanes_replay_step_supported": False,
            "reset_compared": True,
            "compared_ticks": 12,
            "terminal_autoreset_evidence": "not_exercised",
            "rng": {"cursor": 5, "count": 5, "error": 0},
        }
        bad_cases = (
            ({"reset_compared": False}, "reset_not_compared"),
            ({"compared_ticks": 0}, "compared_ticks_incomplete"),
            ({"compared_ticks": 11}, "compared_ticks_incomplete"),
            ({"terminal_autoreset_evidence": "missing"}, "terminal_autoreset_evidence_missing"),
            ({"rng": {"cursor": 4, "count": 5, "error": 0}}, "rng_consumption_inexact"),
            ({"rng": {"cursor": 5, "count": 5, "error": 1}}, "rng_consumption_inexact"),
        )
        for bad_update, reason in bad_cases:
            with self.subTest(reason=reason, bad_update=bad_update):
                receipt = build_receipt(
                    cases=[{**base, **bad_update}],
                    backends=["python", "native_cpu"],
                    expected_actions=[0],
                    expected_seeds=[101],
                    horizon=12,
                )
                self.assertEqual(receipt["status"], "blocked")
                self.assertIn(reason, {blocker["reason"] for blocker in receipt["blockers"]})

    def test_bool_numeric_evidence_is_rejected_for_status_and_three_way(self):
        base = {
            "action_index": 0,
            "seed": 101,
            "horizon": 1,
            "python_reference_exact": True,
            "native_cpu_cuda_exact": True,
            "resident_cpu_recorded_replay_exact": True,
            "resident_cpu_generated_rng_recorded": True,
            "resident_lanes_replay_step_supported": True,
            "reset_compared": True,
            "compared_ticks": 1,
            "terminal_autoreset_evidence": "exact",
            "rng": {"cursor": 1, "count": 1, "error": 0},
        }
        forged_cases = (
            ({"action_index": True}, "invalid_action_seed_case"),
            ({"seed": True}, "invalid_action_seed_case"),
            ({"horizon": True}, "horizon_mismatch"),
            ({"compared_ticks": True}, "compared_ticks_incomplete"),
            ({"rng": {"cursor": 1, "count": True, "error": 0}}, "rng_consumption_inexact"),
            ({"rng": {"cursor": 1, "count": 1, "error": False}}, "rng_consumption_inexact"),
        )
        for bad_update, reason in forged_cases:
            with self.subTest(reason=reason, bad_update=bad_update):
                forged = {**base, **bad_update}
                receipt = build_receipt(
                    cases=[forged],
                    backends=["python", "native_cpu"],
                    expected_actions=[1 if bad_update.get("action_index") is True else 0],
                    expected_seeds=[1 if bad_update.get("seed") is True else 101],
                    horizon=1,
                )
                self.assertEqual(receipt["status"], "blocked")
                self.assertIn(reason, {blocker["reason"] for blocker in receipt["blockers"]})
                self.assertFalse(
                    three_way_exact_allowed(
                        [forged],
                        expected_actions=[1 if bad_update.get("action_index") is True else 0],
                        expected_seeds=[1 if bad_update.get("seed") is True else 101],
                        horizon=1,
                    )
                )

    def test_missing_capture_dependency_is_explicit_blocker(self):
        original = parity._run_original_vs_serial_native

        def missing_dependency(_case, _frozen_root):
            raise ModuleNotFoundError("No module named 'pymunk'")

        try:
            parity._run_original_vs_serial_native = missing_dependency
            receipt = run_suite(
                frozen_root=Path("/missing"),
                backends=["python", "native_cpu"],
                actions=[0],
                seeds=[101],
                horizon=1,
            )
        finally:
            parity._run_original_vs_serial_native = original
        self.assertEqual(receipt["status"], "blocked")
        self.assertFalse(receipt["python_reference_exact"])
        blockers = {blocker["reason"]: blocker for blocker in receipt["blockers"]}
        self.assertIn("missing_capture_dependency", blockers)
        self.assertIn("pymunk", blockers["missing_capture_dependency"]["first_divergence"])

    def test_resident_cpu_replay_step_uses_recorded_draws_and_actions(self):
        original = parity._run_original_vs_serial_native
        resident = parity._run_resident_replay_trace
        forced = parity._run_forced_autoreset_evidence

        def exact_python(_case, _frozen_root):
            return {
                "python_reference_exact": True,
                "document": {
                    "reset": {
                        "draws": [{"kind": 1, "a": 1, "b": 0, "value": 1}],
                        "outputs": {
                            "obs": {"0": [0.0] * 64, "1": [0.0] * 64},
                            "rewards": {"0": 0.0, "1": 0.0},
                            "terminated": False,
                            "truncated": False,
                        },
                    },
                    "ticks": [
                        {
                            "actions": [[[1.0] + [0.0] * 9, [0.0] * 10][0], [0.0] * 10],
                            "draws": [{"kind": 3, "a": 1, "b": 12, "value": 7}],
                            "outputs": {
                                "obs": {"0": [1.0] * 64, "1": [2.0] * 64},
                                "rewards": {"0": 3.0, "1": 4.0},
                                "terminated": False,
                                "truncated": False,
                            },
                        },
                        {
                            "actions": [[0.0] * 10, [0.0] * 10],
                            "draws": [],
                            "outputs": {
                                "obs": {"0": [5.0] * 64, "1": [6.0] * 64},
                                "rewards": {"0": 7.0, "1": 8.0},
                                "terminated": True,
                                "truncated": False,
                            },
                        },
                    ],
                },
                "compared_ticks": 2,
                "reset_compared": True,
                "first_divergence": None,
                "rng": {"cursor": 2, "count": 2, "error": 0},
                "terminal_autoreset_evidence": "not_exercised",
            }

        calls = []

        def resident_trace(case, document, backend_name):
            calls.append(
                (
                    backend_name,
                    document["reset"]["draws"],
                    [row["actions"] for row in document["ticks"]],
                    [row["draws"] for row in document["ticks"]],
                )
            )
            return {
                "resident_cpu_recorded_replay_exact": True,
                "resident_cpu_generated_rng_recorded": True,
                "resident_cpu_recorded_replay_trace": {
                    "ticks": case["horizon"],
                    "mode": "recorded_draw_replay",
                    "terminal_observation_checked": True,
                },
                "resident_cpu_generated_rng_trace": {
                    "ticks": case["horizon"],
                    "mode": "recorded_draw_replay",
                    "terminal_observation_checked": True,
                },
                "resident_cpu_generated_rng_alias_note": (
                    "deprecated compatibility alias; use resident_cpu_recorded_replay_exact/trace"
                ),
                "resident_lanes_replay_step_supported": True,
                "native_cpu_cuda_exact": None,
                "native_cpu_cuda_exact_note": "native_cuda backend not requested",
                "rng": {"cursor": 0, "count": 0, "error": 0},
                "terminal_autoreset_evidence": "exact",
            }

        try:
            parity._run_original_vs_serial_native = exact_python
            parity._run_resident_replay_trace = resident_trace
            parity._run_forced_autoreset_evidence = lambda *args, **kwargs: exact_cuda_autoreset()
            receipt = run_suite(
                frozen_root=Path("/ok"),
                backends=["python", "native_cpu"],
                actions=[0],
                seeds=[101],
                horizon=2,
            )
        finally:
            parity._run_original_vs_serial_native = original
            parity._run_resident_replay_trace = resident
            parity._run_forced_autoreset_evidence = forced
        self.assertEqual(receipt["status"], "completed")
        self.assertTrue(receipt["resident_cpu_generated_rng_recorded"])
        self.assertEqual(receipt["promotion_status"], "blocked")
        self.assertIn("native_cuda_replay_unverified", {b["reason"] for b in receipt["promotion_blockers"]})
        self.assertEqual(receipt["cases"][0]["resident_cpu_recorded_replay_trace"]["mode"], "recorded_draw_replay")
        self.assertIn("deprecated compatibility alias", receipt["cases"][0]["resident_cpu_generated_rng_alias_note"])
        self.assertEqual(calls[0][0], "cpu")
        self.assertEqual(calls[0][1], [{"kind": 1, "a": 1, "b": 0, "value": 1}])
        self.assertEqual(calls[0][3][0], [{"kind": 3, "a": 1, "b": 12, "value": 7}])

    def test_native_cuda_exact_requires_actual_resident_cuda_replay(self):
        original = parity._run_original_vs_serial_native
        resident = parity._run_resident_replay_trace
        forced = parity._run_forced_autoreset_evidence

        def exact_python(_case, _frozen_root):
            return {
                "python_reference_exact": True,
                "document": {
                    "reset": {
                        "draws": [],
                        "outputs": {
                            "obs": {"0": [0.0] * 64, "1": [0.0] * 64},
                            "rewards": {"0": 0.0, "1": 0.0},
                            "terminated": False,
                            "truncated": False,
                        },
                    },
                    "ticks": [
                        {
                            "actions": [[0.0] * 10, [0.0] * 10],
                            "draws": [],
                            "outputs": {
                                "obs": {"0": [0.0] * 64, "1": [0.0] * 64},
                                "rewards": {"0": 0.0, "1": 0.0},
                                "terminated": False,
                                "truncated": False,
                            },
                        }
                    ],
                },
                "compared_ticks": 1,
                "reset_compared": True,
                "first_divergence": None,
                "rng": {"cursor": 0, "count": 0, "error": 0},
                "terminal_autoreset_evidence": "exact",
            }

        calls = []

        def resident_trace(case, document, backend_name, **kwargs):
            calls.append(backend_name)
            trace = {
                "mode": "recorded_draw_replay",
                "backend": backend_name,
                "ticks": 1,
                "valid_steps": [1],
                "final_observation_sha256": "same",
                "final_rewards": [0.0, 0.0],
                "terminated_count": 0,
                "truncated_count": 0,
                "terminal_observation_checked": False,
                "rng_statuses": [{"tick": 1, "mode": 0, "cursor": 0, "count": 0, "error": 0, "expected_count": 0}],
                "reset_draw_count": 0,
                "tick_draw_count": 0,
            }
            result = {
                "resident_cpu_recorded_replay_exact": True,
                "resident_cpu_generated_rng_recorded": True,
                "resident_cpu_recorded_replay_trace": trace,
                "resident_cpu_generated_rng_trace": trace,
                "resident_cpu_generated_rng_alias_note": (
                    "deprecated compatibility alias; use resident_cpu_recorded_replay_exact/trace"
                ),
                "resident_lanes_replay_step_supported": True,
                "rng": {"cursor": 0, "count": 0, "error": 0},
                "terminal_autoreset_evidence": "exact",
            }
            if backend_name == "cuda":
                result["resident_cuda_stream"] = {
                    "mode": kwargs.get("cuda_stream_mode"),
                    "used_non_default_stream": kwargs.get("cuda_stream_mode") == "non_default",
                    "synchronized_before_compare": True,
                }
                result["resident_cuda_observation_checks"] = {
                    "reset_observation_exact": True,
                    "step_observation_exact": True,
                    "reward_exact": True,
                    "terminated_exact": True,
                    "truncated_exact": True,
                    "terminal_observation_exact": True,
                    "reset_observation_sha256": "reset",
                    "terminal_observation_sha256": "terminal",
                }
            return result

        try:
            parity._run_original_vs_serial_native = exact_python
            parity._run_resident_replay_trace = resident_trace
            parity._run_forced_autoreset_evidence = lambda *args, **kwargs: exact_cuda_autoreset()
            receipt = run_suite(
                frozen_root=Path("/ok"),
                backends=["python", "native_cpu", "native_cuda"],
                actions=[0],
                seeds=[101],
                horizon=1,
            )
        finally:
            parity._run_original_vs_serial_native = original
            parity._run_resident_replay_trace = resident
            parity._run_forced_autoreset_evidence = forced
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(calls, ["cpu", "cuda"])
        self.assertTrue(receipt["native_cpu_cuda_exact"])
        self.assertEqual(receipt["promotion_status"], "passed")
        self.assertEqual(receipt["promotion_blockers"], [])

    def test_native_cuda_promotion_requires_nondefault_stream_evidence_and_observation_checks(self):
        complete = [
            {
                "action_index": action_index,
                "seed": 101,
                "horizon": 1,
                "python_reference_exact": True,
                "native_cpu_cuda_exact": True,
                "resident_cpu_recorded_replay_exact": True,
                "resident_cpu_generated_rng_recorded": True,
                "resident_lanes_replay_step_supported": True,
                "reset_compared": True,
                "compared_ticks": 1,
                "terminal_autoreset_evidence": "exact",
                "rng": {"cursor": 1, "count": 1, "error": 0},
            }
            for action_index in range(10)
        ]
        complete[0]["resident_termination_autoreset"] = exact_autoreset()
        complete[1]["resident_truncation_autoreset"] = exact_autoreset()
        receipt = build_receipt(
            cases=complete,
            backends=["python", "native_cpu", "native_cuda"],
            expected_actions=list(range(10)),
            expected_seeds=[101],
            horizon=1,
        )
        self.assertEqual(receipt["promotion_status"], "blocked")
        reasons = {blocker["reason"] for blocker in receipt["promotion_blockers"]}
        self.assertIn("native_cuda_nondefault_stream_unverified", reasons)
        self.assertIn("native_cuda_observation_checks_incomplete", reasons)

        for case in complete:
            case["resident_cuda_stream"] = {
                "mode": "non_default",
                "used_non_default_stream": True,
                "synchronized_before_compare": True,
            }
            case["resident_cuda_observation_checks"] = {
                "reset_observation_exact": True,
                "step_observation_exact": True,
                "reward_exact": True,
                "terminated_exact": True,
                "truncated_exact": True,
                "terminal_observation_exact": True,
                "reset_observation_sha256": "reset",
                "terminal_observation_sha256": "terminal",
            }
        complete[0]["resident_termination_autoreset"] = exact_cuda_autoreset()
        complete[1]["resident_truncation_autoreset"] = exact_cuda_autoreset()
        receipt = build_receipt(
            cases=complete,
            backends=["python", "native_cpu", "native_cuda"],
            expected_actions=list(range(10)),
            expected_seeds=[101],
            horizon=1,
        )
        self.assertEqual(receipt["promotion_status"], "passed")

    def test_native_cuda_forced_autoresets_require_cuda_replay_exactness(self):
        complete = [
            {
                "action_index": action_index,
                "seed": 101,
                "horizon": 1,
                "python_reference_exact": True,
                "native_cpu_cuda_exact": True,
                "resident_cpu_recorded_replay_exact": True,
                "resident_cpu_generated_rng_recorded": True,
                "resident_lanes_replay_step_supported": True,
                "reset_compared": True,
                "compared_ticks": 1,
                "terminal_autoreset_evidence": "exact",
                "rng": {"cursor": 1, "count": 1, "error": 0},
                "resident_cuda_stream": exact_cuda_stream(),
                "resident_cuda_observation_checks": exact_cuda_observation_checks(),
            }
            for action_index in range(10)
        ]
        complete[0]["resident_termination_autoreset"] = {
            **exact_cuda_autoreset(),
            "resident_cuda_recorded_replay_exact": False,
        }
        complete[1]["resident_truncation_autoreset"] = {
            **exact_cuda_autoreset(),
            "resident_cuda_stream": {"mode": "current", "used_non_default_stream": False, "synchronized_before_compare": True},
        }
        receipt = build_receipt(
            cases=complete,
            backends=["python", "native_cpu", "native_cuda"],
            expected_actions=list(range(10)),
            expected_seeds=[101],
            horizon=1,
        )
        self.assertEqual(receipt["status"], "blocked")
        self.assertEqual(receipt["promotion_status"], "blocked")
        reasons = {blocker["reason"] for blocker in receipt["blockers"] + receipt["promotion_blockers"]}
        self.assertIn("termination_cuda_autoreset_evidence_missing", reasons)
        self.assertIn("truncation_cuda_autoreset_evidence_missing", reasons)

    def test_run_suite_uses_nondefault_cuda_stream_when_native_cuda_requested(self):
        original_python = parity._run_original_vs_serial_native
        original_resident = parity._run_resident_replay_trace
        original_forced = parity._run_forced_autoreset_evidence
        calls = []

        def exact_python(_case, _frozen_root):
            return {
                "python_reference_exact": True,
                "document": {
                    "reset": {
                        "draws": [],
                        "outputs": {
                            "obs": {"0": [0.0] * 64, "1": [0.0] * 64},
                            "rewards": {"0": 0.0, "1": 0.0},
                            "terminated": False,
                            "truncated": False,
                        },
                    },
                    "ticks": [
                        {
                            "actions": [[0.0] * 10, [0.0] * 10],
                            "draws": [],
                            "outputs": {
                                "obs": {"0": [0.0] * 64, "1": [0.0] * 64},
                                "rewards": {"0": 0.0, "1": 0.0},
                                "terminated": False,
                                "truncated": False,
                            },
                        }
                    ],
                },
                "compared_ticks": 1,
                "reset_compared": True,
                "first_divergence": None,
                "rng": {"cursor": 0, "count": 0, "error": 0},
                "terminal_autoreset_evidence": "exact",
            }

        def resident_trace(_case, _document, backend_name, **kwargs):
            calls.append((backend_name, kwargs.get("cuda_stream_mode")))
            result = {
                "resident_cpu_recorded_replay_exact": True,
                "resident_cpu_generated_rng_recorded": True,
                "resident_cpu_recorded_replay_trace": {
                    "ticks": 1,
                    "valid_steps": [1],
                    "final_observation_sha256": "same",
                    "final_rewards": [0.0, 0.0],
                    "terminated_count": 0,
                    "truncated_count": 0,
                    "terminal_observation_checked": True,
                    "rng_statuses": [],
                    "reset_draw_count": 0,
                    "tick_draw_count": 0,
                },
                "resident_cpu_generated_rng_recorded": True,
                "resident_lanes_replay_step_supported": True,
                "rng": {"cursor": 0, "count": 0, "error": 0},
                "terminal_autoreset_evidence": "exact",
            }
            if backend_name == "cuda":
                result["resident_cuda_stream"] = exact_cuda_stream()
                result["resident_cuda_observation_checks"] = exact_cuda_observation_checks()
            return result

        try:
            parity._run_original_vs_serial_native = exact_python
            parity._run_resident_replay_trace = resident_trace
            parity._run_forced_autoreset_evidence = lambda *args, **kwargs: exact_cuda_autoreset()
            receipt = run_suite(
                frozen_root=Path("/ok"),
                backends=["python", "native_cpu", "native_cuda"],
                actions=[0],
                seeds=[101],
                horizon=1,
            )
        finally:
            parity._run_original_vs_serial_native = original_python
            parity._run_resident_replay_trace = original_resident
            parity._run_forced_autoreset_evidence = original_forced
        self.assertIn(("cuda", "non_default"), calls)
        self.assertEqual(receipt["promotion_status"], "passed")

    def test_main_writes_manifest_with_source_hashes_and_no_cuda_blocker(self):
        original = parity.run_suite

        def blocked_suite(**_kwargs):
            return {
                "schema_version": parity.SCHEMA_VERSION,
                "status": "blocked",
                "backends": ["python", "native_cpu", "native_cuda"],
                "cases": [
                    {
                        "action_index": 0,
                        "seed": 101,
                        "native_cpu_cuda_exact": False,
                        "native_cpu_cuda_exact_note": "RuntimeError: CUDA unavailable",
                    }
                ],
                "blockers": [
                    {
                        "action_index": 0,
                        "seed": 101,
                        "reason": "native_cpu_cuda_exact_missing",
                        "first_divergence": "RuntimeError: CUDA unavailable",
                    }
                ],
                "promotion_status": "blocked",
                "promotion_blockers": [{"reason": "native_cuda_replay_unverified"}],
            }

        try:
            parity.run_suite = blocked_suite
            with tempfile.TemporaryDirectory(prefix="resident-stream-gate-") as directory:
                output = Path(directory)
                code = parity.main(
                    [
                        "--frozen-root",
                        ".",
                        "--backends",
                        "python,native_cpu,native_cuda",
                        "--actions",
                        "0",
                        "--seeds",
                        "101",
                        "--horizon",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
                self.assertEqual(code, 1)
                manifest = json.loads((output / "resident-stream-gate-manifest.json").read_text())
                self.assertEqual(manifest["schema_version"], "ai2.resident_stream_gate_manifest.v1")
                self.assertEqual(manifest["status"], "blocked")
                self.assertEqual(manifest["cuda_skip"]["status"], "blocked_no_cuda")
                self.assertIn("fixed_action_trace_parity.py", manifest["source_hashes"])
                self.assertIn("native_game_lane_vec_env.py", manifest["source_hashes"])
                self.assertTrue((output / "trace-report.json").is_file())
        finally:
            parity.run_suite = original

    def test_main_refuses_to_overwrite_existing_trace_report(self):
        original = parity.run_suite
        calls = []

        def unexpected_suite(**_kwargs):
            calls.append("called")
            return {"status": "completed", "blockers": []}

        try:
            parity.run_suite = unexpected_suite
            with tempfile.TemporaryDirectory(prefix="resident-stream-gate-") as directory:
                output = Path(directory)
                report = output / "trace-report.json"
                report.write_text("old report\n", encoding="utf-8")
                code = parity.main(
                    [
                        "--frozen-root",
                        ".",
                        "--backends",
                        "python,native_cpu,native_cuda",
                        "--actions",
                        "0",
                        "--seeds",
                        "101",
                        "--horizon",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
                self.assertEqual(code, 1)
                self.assertEqual(report.read_text(encoding="utf-8"), "old report\n")
                self.assertFalse((output / "resident-stream-gate-manifest.json").exists())
                self.assertEqual(calls, [])
        finally:
            parity.run_suite = original

    def test_main_refuses_to_overwrite_existing_manifest(self):
        original = parity.run_suite
        calls = []

        def unexpected_suite(**_kwargs):
            calls.append("called")
            return {"status": "completed", "blockers": []}

        try:
            parity.run_suite = unexpected_suite
            with tempfile.TemporaryDirectory(prefix="resident-stream-gate-") as directory:
                output = Path(directory)
                manifest = output / "resident-stream-gate-manifest.json"
                manifest.write_text("old manifest\n", encoding="utf-8")
                code = parity.main(
                    [
                        "--frozen-root",
                        ".",
                        "--backends",
                        "python,native_cpu,native_cuda",
                        "--actions",
                        "0",
                        "--seeds",
                        "101",
                        "--horizon",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
                self.assertEqual(code, 1)
                self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest\n")
                self.assertFalse((output / "trace-report.json").exists())
                self.assertEqual(calls, [])
        finally:
            parity.run_suite = original

    def test_gdown_fuzzy_compatibility_shim_preserves_older_download_signature(self):
        calls = []
        fake_gdown = types.SimpleNamespace()

        def download(*args, **kwargs):
            calls.append((args, kwargs))
            return "ok"

        fake_gdown.download = download
        parity._install_gdown_fuzzy_compat(fake_gdown)
        self.assertEqual(fake_gdown.download("url", output="asset.zip", fuzzy=True), "ok")
        self.assertEqual(calls, [(("url",), {"output": "asset.zip"})])

    def test_capture_root_adapter_exposes_environment_assets_without_mutating_source(self):
        with tempfile.TemporaryDirectory(prefix="capture-root-source-") as source_dir:
            source = Path(source_dir)
            env = source / "UTMIST-AI2-main" / "environment"
            assets = env / "assets"
            assets.mkdir(parents=True)
            (env / "environment.py").write_text("# fixture\n")
            with parity._prepared_capture_root(source) as prepared:
                self.assertNotEqual(prepared, source)
                self.assertTrue((prepared / "UTMIST-AI2-main").exists())
                self.assertTrue((prepared / "assets").exists())
            self.assertFalse((source / "assets").exists())


if __name__ == "__main__":
    unittest.main()
