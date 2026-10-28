"""Contact parity: only actions, setup geometry, and recorded RNG drive C."""

from __future__ import annotations

from ai2.paths import source_path

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(
    os.environ.get(
        "AI2_CONTACT_REFERENCE",
        "/private/tmp/ai2-curriculum-20260907.Lyif6J/combat_curriculum.py",
    )
)
FROZEN = Path(
    os.environ.get(
        "AI2_FROZEN_SOURCE_ROOT",
        "/Users/elliot18/Projects/ml-research/AI_2/baselines/fullgame-20260908/oracle",
    )
)


class NativeContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ai2.simulation.native_contact_episode import compile_reference

        # Pin one source build before the game imports Torch/OpenMP. Compiling
        # after that import can abort a forked compiler on macOS, and concurrent
        # unrelated header edits must not change the library between cases.
        cls.native_library = compile_reference()
        spec = importlib.util.spec_from_file_location(
            "contact_python_reference", REFERENCE
        )
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)
        cls.env = cls.reference.make_b1_contact_env(FROZEN)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def native(self):
        self.assertTrue(
            (source_path(ROOT, "native_contact_episode.py")).is_file(),
            "native contact task must exist",
        )
        from ai2.simulation.native_contact_episode import ContactReference

        return ContactReference(self.native_library)

    def reset_pair(self, native, seed, slot, direction):
        from ai2.evaluation.navigation_episode_capture import DrawRecorder

        recorder = DrawRecorder()
        recorder.install()
        try:
            observation, info = self.env.reset(
                seed=seed,
                options={"scenario": {"player_slot": slot, "direction": direction}},
            )
            draws = recorder.take()
        finally:
            recorder.restore()
        scenario = native.scenario(seed, slot, direction)
        self.assertEqual(scenario.gap, self.env.spawn_gap)
        self.assertEqual(scenario.midpoint, self.env.spawn_midpoint)
        result = native.reset_replay(scenario, draws)
        self.compare(native, result, observation, 0.0, False, False, info)
        return recorder

    def compare(self, native, result, observation, reward, terminated, truncated, info):
        np.testing.assert_array_equal(
            np.asarray(result.obs, dtype=np.float32).view(np.uint32),
            observation.view(np.uint32),
        )
        self.assertEqual(result.reward, reward)
        self.assertEqual(bool(result.terminated), terminated)
        self.assertEqual(bool(result.truncated), truncated)
        trace = info["trace"]
        projection = native.diagnostics(result)
        self.assertEqual(projection["outcome"], trace["outcome"])
        self.assertEqual(projection["tick"], trace["tick"])
        self.assertEqual(projection["first_hit_tick"], self.env.first_hit_tick)
        self.assertEqual(projection["reward_terms"], self.env.reward_totals)
        self.assertEqual(
            projection["approach_progress_at_first_hit"],
            self.env.approach_progress_at_first_hit,
        )
        np.testing.assert_array_equal(result.raw_action, trace["raw_action"])
        np.testing.assert_array_equal(result.executed_action, trace["executed_action"])
        self.assertEqual(native.state.game.scene.steps, trace["tick"])
        for index, player in enumerate(self.env.raw_env.players):
            internal = native.state.game.scene.players[index]
            self.assertEqual(internal.ms.start_x, player.start_position[0])
            self.assertEqual(internal.ms.start_y, player.start_position[1])
            self.assertEqual(internal.ms.prev_x, player.prev_x)
            self.assertEqual(internal.ms.prev_y, player.prev_y)
            self.assertEqual(
                native.state.game.combat[index].damage_done, player.damage_done
            )
            self.assertEqual(
                native.state.game.combat[index].damage_taken_total,
                player.damage_taken_total,
            )
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
                self.assertEqual(
                    got[name],
                    expected[name],
                    f"tick={trace['tick']} field={name} slot={expected['player_slot']}",
                )

    def episode(self, controller, *, seed=0, slot=0, direction=1):
        native = self.native()
        recorder = self.reset_pair(native, seed, slot, direction)
        recorder.install()
        try:
            for tick in range(1, 301):
                action = np.zeros(10, dtype=np.float32)
                if controller == "scripted":
                    action = self.reference.scripted_contact_action(self.env)
                elif controller == "held_light":
                    action[7] = 1
                elif controller == "walkoff":
                    action[3 if direction == 1 else 1] = 1
                elif controller == "unsafe_hit":
                    action[3 if direction == 1 else 1] = action[7] = 1
                observation, reward, terminated, truncated, info = self.env.step(action)
                result = native.step_replay(action, recorder.take())
                self.compare(
                    native, result, observation, reward, terminated, truncated, info
                )
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            return native, result
        finally:
            recorder.restore()

    def test_fresh_reset_matches_both_slots_and_directions(self):
        native = self.native()
        for seed in range(20):
            for slot in (0, 1):
                for direction in (-1, 1):
                    with self.subTest(seed=seed, slot=slot, direction=direction):
                        self.reset_pair(native, seed, slot, direction)
                        self.assertEqual(native.state.game.scene.shape_id_counter, 7)
                        for index, player_slot in enumerate((slot, 1 - slot)):
                            player = native.state.game.scene.players[player_slot]
                            self.assertEqual(
                                native.state.game.scene.world.shapes[
                                    player.shape
                                ].hashid,
                                5 + index,
                            )

    def test_scripted_safe_contact_matches_reference(self):
        for seed in range(20):
            for slot in (0, 1):
                for direction in (-1, 1):
                    with self.subTest(seed=seed, slot=slot, direction=direction):
                        native, result = self.episode(
                            "scripted", seed=seed, slot=slot, direction=direction
                        )
                        self.assertEqual(
                            native.diagnostics(result)["outcome"], "success"
                        )

    def test_negative_controls_match_reference(self):
        for control in ("idle", "held_light", "walkoff", "unsafe_hit"):
            for seed in range(20):
                for slot in (0, 1):
                    for direction in (-1, 1):
                        with self.subTest(
                            control=control, seed=seed, slot=slot, direction=direction
                        ):
                            native, result = self.episode(
                                control, seed=seed, slot=slot, direction=direction
                            )
                            self.assertNotEqual(
                                native.diagnostics(result)["outcome"], "success"
                            )

    def test_clipping_mask_and_causal_attack_counters(self):
        native = self.native()
        recorder = self.reset_pair(native, 17, 0, 1)
        recorder.install()
        starts = damage = requests = presses = 0
        try:
            for _ in range(100):
                action = self.reference.scripted_contact_action(self.env)
                action[action == 1] = 3.0
                action[[5, 8, 9]] = 4.0
                action[0] = -2.0
                observation, reward, term, trunc, info = self.env.step(action)
                result = native.step_replay(action, recorder.take())
                self.compare(native, result, observation, reward, term, trunc, info)
                events = info["trace"]["events"]
                starts += sum(e["type"] == "attack_started" for e in events)
                damage += sum(e["type"] == "damage" for e in events)
                presses += sum(e["type"] == "attack_press" for e in events)
                requests += action[7] > 0.5
                self.assertEqual(result.diagnostics.attack_starts, starts)
                self.assertEqual(result.diagnostics.damage_events, damage)
                self.assertEqual(result.diagnostics.light_press_count, presses)
                self.assertEqual(result.diagnostics.light_request_ticks, requests)
                if term:
                    self.assertEqual(result.diagnostics.outcome, 1)
                    break
        finally:
            recorder.restore()

    def test_invalid_actions_and_scenarios_rejected_before_mutation(self):
        import ctypes as C
        from ai2.simulation.native_contact_episode import Scenario, StepResult

        native = self.native()
        native.reset_generated(native.scenario(17))
        for bad in (float("nan"), float("inf"), 1e300):
            before = bytes(native.state)
            action = (C.c_double * 10)(*([bad] + [0.0] * 9))
            result = StepResult()
            code = native.lib.ai2b1_step_export(
                C.byref(native.state),
                action,
                C.byref(native.rng),
                C.byref(native.table),
                C.byref(result),
            )
            self.assertEqual(code, 402)
            self.assertEqual(bytes(native.state), before)
        for scenario in (
            Scenario(17, 2, 1, 3.2, 4.5),
            Scenario(17, 0, 0, 3.2, 4.5),
            Scenario(17, 0, 1, 3.0, 4.5),
            Scenario(17, 0, 1, 3.2, float("nan")),
        ):
            before = bytes(native.state)
            with self.assertRaisesRegex(RuntimeError, "401"):
                native.reset_generated(scenario)
            self.assertEqual(bytes(native.state), before)

    def test_post_terminal_step_rejected_and_fresh_reset_restarts_task(self):
        native = self.native()
        native.reset_generated(native.scenario(17))
        for _ in range(300):
            result = native.step_generated(np.zeros(10))
        self.assertEqual(result.diagnostics.outcome, 6)
        self.assertEqual(result.reward_terms[3], -1.0)
        self.assertFalse(result.truncated)
        before = bytes(native.state)
        with self.assertRaisesRegex(RuntimeError, "403"):
            native.step_generated(np.zeros(10))
        self.assertEqual(bytes(native.state), before)
        result = native.reset_generated(native.scenario(18, 1, -1))
        self.assertEqual(result.diagnostics.tick, 0)
        self.assertEqual(result.diagnostics.outcome, 0)
        self.assertEqual(result.diagnostics.hit_attack_id, 0)

    def test_corrupt_stored_scenario_rejected_before_indexed_access(self):
        # Keep a pre-fix native out-of-bounds access from aborting the test runner.
        script = textwrap.dedent("""
            import ctypes as C
            import sys
            from native_contact_episode import ContactReference, StepResult
            native = ContactReference(sys.argv[1])
            for field, value in (
                ('player_slot', 2), ('player_slot', -1),
                ('direction', 0), ('direction', 2),
                ('gap', float('nan')), ('gap', float('inf')), ('gap', 3.0),
                ('midpoint', float('nan')), ('midpoint', float('inf')), ('midpoint', 4.0),
            ):
                native.reset_generated(native.scenario(17))
                setattr(native.state.scenario, field, value)
                state_before, rng_before = bytes(native.state), bytes(native.rng)
                result = StepResult()
                code = native.lib.ai2b1_step_export(C.byref(native.state),
                    (C.c_double * 10)(), C.byref(native.rng), C.byref(native.table), C.byref(result))
                assert code == result.error == 401, (field, value, code)
                assert bytes(native.state) == state_before, field
                assert bytes(native.rng) == rng_before, field
            print('stored scenario rejection ok cases=10')
        """)
        process = subprocess.run(
            [sys.executable, "-c", script, str(self.native_library)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("stored scenario rejection ok cases=10", process.stdout)

    def test_corrupt_contact_bookkeeping_rejected_before_mutation(self):
        import ctypes as C
        from ai2.simulation.native_contact_episode import StepResult

        native = self.native()
        mutations = [
            ("tick", -1),
            ("tick", 300),
            ("tick", 2**31 - 1),
            ("player_slot", 1),
            ("direction", -1),
            ("seed", 18),
            ("outcome", -1),
            ("outcome", 7),
            ("first_hit_tick", 1),
        ]
        for field in (
            "light_request_ticks",
            "light_press_count",
            "max_light_hold",
            "attack_ticks",
        ):
            mutations.extend(((field, -1), (field, 2**31 - 1)))
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                native.reset_generated(native.scenario(17))
                setattr(native.state.diagnostics, field, value)
                self.assert_invalid_contact_state(native, StepResult, C)
        for field, value in (
            ("previous_light", 2),
            ("current_light_hold", -1),
            ("current_light_hold", 2**31 - 1),
        ):
            with self.subTest(field=field, value=value):
                native.reset_generated(native.scenario(17))
                setattr(native.state, field, value)
                self.assert_invalid_contact_state(native, StepResult, C)
        for slot in (0, 1):
            for value in (-1, 2**31 - 1):
                with self.subTest(unsupported_slot=slot, value=value):
                    native.reset_generated(native.scenario(17))
                    native.state.diagnostics.unsupported_ticks[slot] = value
                    self.assert_invalid_contact_state(native, StepResult, C)
        native.reset_generated(native.scenario(17))
        native.state.game.scene.steps = 1
        self.assert_invalid_contact_state(native, StepResult, C)

    def assert_invalid_contact_state(self, native, result_type, ctypes_module):
        before, rng_before = bytes(native.state), bytes(native.rng)
        result = result_type()
        code = native.lib.ai2b1_step_export(
            ctypes_module.byref(native.state),
            (ctypes_module.c_double * 10)(),
            ctypes_module.byref(native.rng),
            ctypes_module.byref(native.table),
            ctypes_module.byref(result),
        )
        self.assertEqual(code, 405)
        self.assertEqual(result.error, 405)
        self.assertEqual(bytes(native.state), before)
        self.assertEqual(bytes(native.rng), rng_before)

    def test_generated_streams_repeat_and_interleave_independently(self):
        first, second, unrelated = self.native(), self.native(), self.native()
        scenario = first.scenario(17, 1, -1)
        first.reset_generated(scenario, stream=3)
        second.reset_generated(scenario, stream=3)
        unrelated.reset_generated(unrelated.scenario(600))
        rng = np.random.default_rng(20)
        for _ in range(20):
            action = rng.uniform(-0.1, 1.1, 10)
            a = first.step_generated(action)
            unrelated.step_generated(np.zeros(10))
            b = second.step_generated(action)
            self.assertEqual(bytes(a), bytes(b))
            self.assertEqual(bytes(first.state), bytes(second.state))

    def test_scenario_helper_rejects_noninteger_identity_fields(self):
        from ai2.simulation.native_contact_episode import scenario_from_seed

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
