"""B2 re-engagement native CPU parity and ABI tests."""

from __future__ import annotations

import ctypes as C
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(
    os.environ.get(
        "AI2_FROZEN_SOURCE_ROOT",
        "/Users/elliot18/Projects/ml-research/AI_2/baselines/fullgame-20260908/oracle",
    )
)


class NativeReengagementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from native_reengagement_episode import compile_reference

        cls.native_library = compile_reference()
        spec = importlib.util.spec_from_file_location(
            "reengagement_python_reference", ROOT / "reengagement_curriculum.py"
        )
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)
        cls.env = cls.reference.make_reengagement_env(FROZEN)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def native(self):
        from native_reengagement_episode import ReengagementReference

        return ReengagementReference(self.native_library)

    def reset_pair(self, native, seed, slot=None, direction=None):
        from navigation_episode_capture import DrawRecorder

        scenario = native.scenario(seed, seed % 2 if slot is None else slot, direction)
        options = {
            "scenario": {
                "player_slot": scenario.player_slot,
                "direction": scenario.direction,
            }
        }
        recorder = DrawRecorder()
        recorder.install()
        try:
            observation, info = self.env.reset(seed=seed, options=options)
            draws = recorder.take()
        finally:
            recorder.restore()
        self.assertEqual(scenario.gap, self.env.scenario["gap"])
        self.assertEqual(scenario.midpoint, self.env.scenario["midpoint"])
        result = native.reset_replay(scenario, draws)
        self.compare(native, result, observation, 0.0, False, False, info)
        return recorder

    def compare(self, native, result, observation, reward, terminated, truncated, info):
        np.testing.assert_array_max_ulp(
            np.asarray(result.obs, dtype=np.float32),
            observation,
            maxulp=8,
        )
        self.assertEqual(result.reward, reward)
        self.assertEqual(bool(result.terminated), terminated)
        self.assertEqual(bool(result.truncated), truncated)
        trace = info["trace"]
        projection = native.diagnostics(result)
        self.assertEqual(projection["version"], "bridge_b2_reengagement-v1")
        self.assertEqual(projection["outcome"], trace["outcome"])
        self.assertEqual(projection["tick"], trace["tick"])
        self.assertEqual(projection["first_hit_tick"], trace["first_hit_tick"])
        self.assertEqual(projection["second_hit_tick"], trace["second_hit_tick"])
        self.assertEqual(projection["relocation_tick"], trace["relocation_tick"])
        self.assertEqual(projection["pursuit_tick"], trace["pursuit_tick"])
        self.assertEqual(projection["reward_terms"], trace["reward_totals"])
        np.testing.assert_array_equal(result.raw_action, trace["raw_action"])
        np.testing.assert_array_equal(result.executed_action, trace["executed_action"])
        for got, expected in zip(projection["players"], trace["post_players"]):
            for name in (
                "x",
                "y",
                "vx",
                "vy",
                "facing",
                "stocks",
                "damage",
                "state",
                "grounded",
                "fully_supported",
                "horizontal_floor_clearance",
                "stun_time_stored",
            ):
                message = f"tick={trace['tick']} field={name} slot={expected['player_slot']}"
                if isinstance(expected[name], float):
                    self.assertAlmostEqual(got[name], expected[name], places=12, msg=message)
                else:
                    self.assertEqual(got[name], expected[name], message)

    def episode(self, controller, *, seed=134000, slot=0, direction=-1):
        native = self.native()
        recorder = self.reset_pair(native, seed, slot, direction)
        recorder.install()
        try:
            for _ in range(300):
                action = np.zeros(10, dtype=np.float32)
                if controller == "scripted":
                    action = self.reference.scripted_reengagement_action(self.env)
                elif controller == "held_light":
                    action[7] = 1
                elif controller == "walkoff":
                    action[3 if direction == 1 else 1] = 1
                elif controller == "first_then_idle":
                    if self.env.first_hit_tick is None:
                        action = self.reference.scripted_reengagement_action(self.env)
                observation, reward, terminated, truncated, info = self.env.step(action)
                result = native.step_replay(action, recorder.take())
                self.compare(native, result, observation, reward, terminated, truncated, info)
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            return native, result
        finally:
            recorder.restore()

    def test_constants_abi_sizes_reward_order_and_scenario_generation(self):
        from native_reengagement_episode import (
            REWARD_NAMES,
            Scenario,
            State,
            StepResult,
            diagnostics_projection,
            scenario_from_seed,
        )

        self.assertEqual(
            REWARD_NAMES,
            (
                "first_hit",
                "pursuit",
                "second_hit",
                "completion",
                "stock_loss",
                "unsafe_support",
                "deadline",
                "tick",
            ),
        )
        native = self.native()
        for name, cls in (
            ("scenario", Scenario),
            ("state", State),
            ("result", StepResult),
        ):
            self.assertEqual(getattr(native.lib, f"ai2b2_{name}_size")(), C.sizeof(cls))
        first = scenario_from_seed(134000)
        second = scenario_from_seed(134000)
        self.assertEqual(bytes(first), bytes(second))
        self.assertEqual(first.player_slot, 0)
        self.assertEqual(first.direction, -1)
        self.assertGreaterEqual(first.gap, 3.1)
        self.assertLessEqual(first.gap, 3.4)
        self.assertGreaterEqual(first.midpoint, 4.35)
        self.assertLessEqual(first.midpoint, 4.65)
        projection = diagnostics_projection(native.reset_generated(first).diagnostics)
        self.assertEqual(projection["outcome"], None)

    def test_fresh_reset_matches_both_slots_and_directions(self):
        native = self.native()
        for seed in range(134000, 134004):
            for slot in (0, 1):
                for direction in (-1, 1):
                    with self.subTest(seed=seed, slot=slot, direction=direction):
                        self.reset_pair(native, seed, slot, direction)

    def test_scripted_reengagement_matches_reference(self):
        for seed in range(134000, 134004):
            for slot in (0, 1):
                for direction in (-1, 1):
                    with self.subTest(seed=seed, slot=slot, direction=direction):
                        native, result = self.episode(
                            "scripted", seed=seed, slot=slot, direction=direction
                        )
                        projection = native.diagnostics(result)
                        self.assertEqual(projection["outcome"], "success")
                        self.assertIsNotNone(projection["first_hit_attack_id"])
                        self.assertIsNotNone(projection["second_hit_attack_id"])
                        self.assertNotEqual(
                            projection["first_hit_attack_id"],
                            projection["second_hit_attack_id"],
                        )
                        self.assertGreaterEqual(projection["gap_closed"], 1.0)
                        self.assertGreaterEqual(projection["learner_progress"], 1.0)

    def test_shortcut_and_adversarial_controllers_match_reference(self):
        for control in ("idle", "held_light", "walkoff", "first_then_idle"):
            for seed in range(134000, 134002):
                for slot in (0, 1):
                    for direction in (-1, 1):
                        with self.subTest(control=control, seed=seed, slot=slot, direction=direction):
                            native, result = self.episode(
                                control, seed=seed, slot=slot, direction=direction
                            )
                            self.assertNotEqual(
                                native.diagnostics(result)["outcome"], "success"
                            )

    def test_seeded_random_actions_match_reference(self):
        for seed in range(134010, 134012):
            native = self.native()
            recorder = self.reset_pair(native, seed, seed % 2, -1 if seed % 2 == 0 else 1)
            actions = np.random.default_rng(seed).uniform(-0.25, 1.25, size=(80, 10)).astype(np.float32)
            recorder.install()
            try:
                for action in actions:
                    observation, reward, terminated, truncated, info = self.env.step(action)
                    result = native.step_replay(action, recorder.take())
                    self.compare(native, result, observation, reward, terminated, truncated, info)
                    if terminated or truncated:
                        break
            finally:
                recorder.restore()

    def test_invalid_actions_scenarios_post_terminal_and_corrupt_state(self):
        from native_reengagement_episode import Scenario, StepResult

        native = self.native()
        native.reset_generated(native.scenario(134000))
        for bad in (float("nan"), float("inf"), 1e300):
            before = bytes(native.state)
            action = (C.c_double * 10)(*([bad] + [0.0] * 9))
            result = StepResult()
            code = native.lib.ai2b2_step_export(
                C.byref(native.state),
                action,
                C.byref(native.rng),
                C.byref(native.table),
                C.byref(result),
            )
            self.assertEqual(code, 402)
            self.assertEqual(bytes(native.state), before)
        for scenario in (
            Scenario(134000, 2, -1, 3.2, 4.5),
            Scenario(134000, 0, 0, 3.2, 4.5),
            Scenario(134000, 0, 1, 2.9, 4.5),
            Scenario(134000, 0, 1, 3.2, float("nan")),
        ):
            before = bytes(native.state)
            with self.assertRaisesRegex(RuntimeError, "401"):
                native.reset_generated(scenario)
            self.assertEqual(bytes(native.state), before)
        native, result = self.episode("idle", seed=134100, slot=0, direction=-1)
        self.assertEqual(native.diagnostics(result)["outcome"], "total_deadline")
        before = bytes(native.state)
        with self.assertRaisesRegex(RuntimeError, "403"):
            native.step_generated(np.zeros(10))
        self.assertEqual(bytes(native.state), before)
        native.reset_generated(native.scenario(134000))
        native.state.diagnostics.tick = -1
        self.assert_invalid_state(native)
        native.reset_generated(native.scenario(134000))
        native.state.second_stage = 2
        self.assert_invalid_state(native)

    def assert_invalid_state(self, native):
        from native_reengagement_episode import StepResult

        before, rng_before = bytes(native.state), bytes(native.rng)
        result = StepResult()
        code = native.lib.ai2b2_step_export(
            C.byref(native.state),
            (C.c_double * 10)(),
            C.byref(native.rng),
            C.byref(native.table),
            C.byref(result),
        )
        self.assertEqual(code, 405)
        self.assertEqual(result.error, 405)
        self.assertEqual(bytes(native.state), before)
        self.assertEqual(bytes(native.rng), rng_before)

    def test_elapsed_deadline_and_relocation_anchor_boundaries(self):
        native, result = self.episode("first_then_idle", seed=134000, slot=0, direction=-1)
        projection = native.diagnostics(result)
        self.assertEqual(projection["outcome"], "response_deadline")
        self.assertIsNotNone(projection["relocation_tick"])
        self.assertEqual(projection["tick"] - projection["relocation_tick"], 60)
        self.assertEqual(result.reward_terms[6], -1.0)

        native, result = self.episode("idle", seed=134000, slot=0, direction=-1)
        projection = native.diagnostics(result)
        self.assertEqual(projection["outcome"], "total_deadline")
        self.assertEqual(projection["tick"], 300)
        self.assertEqual(result.reward_terms[6], -1.0)

        from reengagement_curriculum import choose_target_anchor

        self.assertEqual(choose_target_anchor(0, 10, 0.435, 5.0, 1), 9.365)
        self.assertEqual(choose_target_anchor(0, 10, 0.435, 5.0, -1), 0.635)
        self.assertEqual(choose_target_anchor(0, 3, 0.435, 1.5, 1), None)

    def test_private_rng_determinism_and_projection_serializable(self):
        first, second, unrelated = self.native(), self.native(), self.native()
        scenario = first.scenario(134000, 1, 1)
        first.reset_generated(scenario, stream=7)
        second.reset_generated(scenario, stream=7)
        unrelated.reset_generated(unrelated.scenario(134020))
        rng = np.random.default_rng(20)
        for _ in range(20):
            action = rng.uniform(-0.1, 1.1, 10)
            a = first.step_generated(action)
            unrelated.step_generated(np.zeros(10))
            b = second.step_generated(action)
            self.assertEqual(bytes(a), bytes(b))
            self.assertEqual(bytes(first.state), bytes(second.state))
        projection = first.diagnostics(a)
        self.assertIn("players", projection)
        self.assertEqual(set(projection["reward_terms"]), set(self.reference.REWARD_NAMES))

    def test_corrupt_stored_scenario_rejected_in_subprocess(self):
        script = textwrap.dedent(
            """
            import ctypes as C
            import sys
            from native_reengagement_episode import ReengagementReference, StepResult
            native = ReengagementReference(sys.argv[1])
            for field, value in (
                ('player_slot', 2), ('direction', 0), ('gap', 3.0),
                ('midpoint', float('nan')),
            ):
                native.reset_generated(native.scenario(134000))
                setattr(native.state.scenario, field, value)
                state_before, rng_before = bytes(native.state), bytes(native.rng)
                result = StepResult()
                code = native.lib.ai2b2_step_export(C.byref(native.state),
                    (C.c_double * 10)(), C.byref(native.rng), C.byref(native.table), C.byref(result))
                assert code == result.error == 401, (field, value, code)
                assert bytes(native.state) == state_before, field
                assert bytes(native.rng) == rng_before, field
            print('stored scenario rejection ok cases=4')
            """
        )
        process = subprocess.run(
            [sys.executable, "-c", script, str(self.native_library)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("stored scenario rejection ok cases=4", process.stdout)

    def test_scenario_helper_rejects_noninteger_identity_fields(self):
        from native_reengagement_episode import scenario_from_seed

        for kwargs in (
            {"seed": -1},
            {"seed": True},
            {"seed": 0, "player_slot": 1.0},
            {"seed": 0, "direction": 1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                scenario_from_seed(**kwargs)


if __name__ == "__main__":
    unittest.main()
