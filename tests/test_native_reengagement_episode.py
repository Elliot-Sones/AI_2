"""B2 re-engagement native CPU parity and ABI tests."""

from __future__ import annotations

import ctypes as C
import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

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
        import pymunk

        from native_reengagement_episode import compile_reference

        pymunk_version = getattr(pymunk, "version", None) or getattr(pymunk, "__version__", None)
        if pymunk_version != "6.2.1":
            raise AssertionError(
                f"B2 oracle parity requires pymunk==6.2.1, got {pymunk_version!r}; "
                "run the suite with --with pymunk==6.2.1"
            )

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

    @classmethod
    def relocation_probe(cls):
        if hasattr(cls, "_relocation_probe"):
            return cls._relocation_probe
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "relocation_probe.c"
            library = Path(directory) / "relocation_probe.so"
            source.write_text(
                textwrap.dedent(
                    """
                    #include "native/reengagement_episode.h"
                    int ai2b2_test_relocate_export(AI2B2State *s) {
                        return ai2b2_relocate(s);
                    }
                    """
                )
            )
            subprocess.run(
                [
                    os.environ.get("CC", "cc"),
                    "-std=c99",
                    "-O2",
                    "-fno-fast-math",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-shared",
                    "-fPIC",
                    "-I",
                    str(ROOT),
                    str(source),
                    "-lm",
                    "-o",
                    str(library),
                ],
                cwd=ROOT,
                check=True,
            )
            cls._relocation_probe_path = ROOT / ".native-build" / f"relocation-probe-{os.getpid()}.so"
            cls._relocation_probe_path.write_bytes(library.read_bytes())
        probe = C.CDLL(str(cls._relocation_probe_path))
        from native_reengagement_episode import State

        probe.ai2b2_test_relocate_export.argtypes = [C.POINTER(State)]
        probe.ai2b2_test_relocate_export.restype = C.c_int
        cls._relocation_probe = probe
        return probe

    def injected_step(self, native, stream, event, action=None):
        from native_reengagement_episode import StepResult

        if action is None:
            action = np.zeros(10, dtype=np.float64)
        action = np.ascontiguousarray(np.asarray(action, dtype=np.float64))
        result = StepResult()
        code = native.lib.ai2b2_step_injected_event_export(
            C.byref(native.state),
            action.ctypes.data_as(C.POINTER(C.c_double)),
            C.byref(native.rng),
            C.byref(native.table),
            stream,
            C.byref(event),
            C.byref(result),
        )
        return code, result

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
        got_obs = np.asarray(result.obs, dtype=np.float32).view(np.uint32)
        expected_obs = np.asarray(observation, dtype=np.float32).view(np.uint32)
        np.testing.assert_array_equal(got_obs, expected_obs)
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
        self.assertEqual(projection["scenario_id"], trace["scenario_id"])
        self.assertEqual(
            projection["sequence_completed"],
            trace["outcome"] == "success"
            and trace["first_hit_tick"] is not None
            and trace["second_hit_tick"] is not None,
        )
        self.assertEqual(projection["success"], trace["outcome"] == "success")
        self.assertEqual(projection["terminal_reason"], trace["outcome"])
        self.assertEqual(projection["attributed_damage_event_count"], len(trace["attributed_damage_events"]))
        self.assertEqual(
            [event["attack_id"] for event in projection["attributed_damage_events"]],
            [projection["first_hit_attack_id"], projection["second_hit_attack_id"]][
                : len(trace["attributed_damage_events"])
            ],
        )
        self.assertEqual(projection["achieved_first_hit"], trace["first_hit_tick"] is not None)
        self.assertEqual(projection["achieved_relocation"], trace["relocation_tick"] is not None)
        self.assertEqual(projection["achieved_pursuit"], trace["pursuit_tick"] is not None)
        self.assertEqual(projection["achieved_second_hit"], trace["second_hit_tick"] is not None)
        self.assertEqual(projection["achieved_safe_resolution"], trace["outcome"] == "success")
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
                elif controller == "pursue_then_idle":
                    if self.env.first_hit_tick is None:
                        action = self.reference.scripted_reengagement_action(self.env)
                    elif self.env.pursuit_tick is None:
                        learner = self.env.raw_env.players[self.env.player_slot]
                        target = self.env.raw_env.players[1 - self.env.player_slot]
                        gap = target.body.position.x - learner.body.position.x
                        action[3 if gap >= 0 else 1] = 1
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
            ABI_LAYOUTS,
            ABI_OFFSETS,
            REWARD_NAMES,
            Diagnostics,
            PlayerEvent,
            PlayerSnapshot,
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
            ("diagnostics", Diagnostics),
            ("player", PlayerSnapshot),
        ):
            self.assertEqual(getattr(native.lib, f"ai2b2_{name}_size")(), C.sizeof(cls))
        prefixes = {cls: prefix for prefix, cls in ABI_LAYOUTS}
        for cls, fields in ABI_OFFSETS.items():
            count_fn = getattr(native.lib, f"ai2b2_{prefixes[cls]}_field_count")
            self.assertEqual(count_fn(), len(cls._fields_))
            for field_name, (export_name, index) in fields.items():
                with self.subTest(cls=cls.__name__, field=field_name):
                    fn = getattr(native.lib, export_name)
                    self.assertEqual(fn(index), getattr(cls, field_name).offset)
        self.assertEqual(C.sizeof(PlayerEvent), native.lib.ai2b2_player_event_size())
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
        self.assertEqual(projection["scenario_id"], "reengagement-s134000-p0-d-1")
        self.assertFalse(projection["achieved_first_hit"])
        self.assertEqual(projection["attributed_damage_events"], [])

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
        def fresh_reset_state():
            candidate = self.native()
            candidate.reset_generated(candidate.scenario(134000))
            return candidate

        def relocated_state():
            candidate = self.native()
            candidate.reset_generated(candidate.scenario(134000))
            code = candidate.lib.ai2b2_debug_prepare_boundary_export(
                C.byref(candidate.state), C.byref(candidate.table), 1, 30, 80
            )
            self.assertEqual(code, 0)
            return candidate

        def pursuit_state():
            candidate = self.native()
            candidate.reset_generated(candidate.scenario(134000))
            code = candidate.lib.ai2b2_debug_prepare_boundary_export(
                C.byref(candidate.state), C.byref(candidate.table), 2, 30, 80
            )
            self.assertEqual(code, 0)
            return candidate

        def completed_state():
            candidate, _result = self.episode("scripted", seed=134000, slot=0, direction=-1)
            return candidate

        cases = (
            ("reset", fresh_reset_state, lambda s: setattr(s.diagnostics, "tick", -1)),
            ("reset", fresh_reset_state, lambda s: setattr(s, "second_stage", 2)),
            ("reset", fresh_reset_state, lambda s: setattr(s, "relocated", 2)),
            ("reset", fresh_reset_state, lambda s: setattr(s, "first_attack_id", 99)),
            ("reset", fresh_reset_state, lambda s: setattr(s.diagnostics, "first_hit_attack_id", 99)),
            ("reset", fresh_reset_state, lambda s: setattr(s.diagnostics, "first_hit_tick", 1)),
            ("reset", fresh_reset_state, lambda s: setattr(s, "second_attack_id", 99)),
            ("reset", fresh_reset_state, lambda s: setattr(s.diagnostics, "second_hit_attack_id", 99)),
            ("reset", fresh_reset_state, lambda s: setattr(s.diagnostics, "second_hit_tick", 1)),
            ("reset", fresh_reset_state, lambda s: setattr(s, "last_second_duplicate_id", 99)),
            ("reset", fresh_reset_state, lambda s: setattr(s.diagnostics, "relocation_tick", 1)),
            ("reset", fresh_reset_state, lambda s: setattr(s.diagnostics, "pursuit_tick", 1)),
            ("reset", fresh_reset_state, lambda s: setattr(s.diagnostics, "pursuit_direction", 1)),
            ("reset", fresh_reset_state, lambda s: setattr(s.diagnostics, "outcome", 1)),
            ("reset", fresh_reset_state, lambda s: setattr(s.diagnostics, "achieved_first_hit", 1)),
            ("relocated", relocated_state, lambda s: setattr(s, "first_attack_id", s.first_attack_id + 1)),
            ("relocated", relocated_state, lambda s: setattr(s.diagnostics, "first_hit_attack_id", s.diagnostics.first_hit_attack_id + 1)),
            ("relocated", relocated_state, lambda s: setattr(s.diagnostics, "first_hit_tick", -1)),
            ("relocated", relocated_state, lambda s: setattr(s.diagnostics, "relocation_tick", -1)),
            ("relocated", relocated_state, lambda s: setattr(s, "second_stage", 0)),
            ("relocated", relocated_state, lambda s: setattr(s, "relocated", 0)),
            ("relocated", relocated_state, lambda s: setattr(s.diagnostics, "achieved_relocation", 0)),
            ("relocated", relocated_state, lambda s: setattr(s.diagnostics, "pursuit_direction", 0)),
            ("relocated", relocated_state, lambda s: setattr(s.diagnostics, "success", 1)),
            ("relocated", relocated_state, lambda s: setattr(s.diagnostics, "sequence_completed", 1)),
            ("relocated", relocated_state, lambda s: setattr(s.diagnostics, "achieved_safe_resolution", 1)),
            ("pursuit", pursuit_state, lambda s: setattr(s.diagnostics, "pursuit_tick", -1)),
            ("pursuit", pursuit_state, lambda s: setattr(s.diagnostics, "achieved_pursuit", 0)),
            ("complete", completed_state, lambda s: setattr(s, "second_attack_id", s.second_attack_id + 1)),
            ("complete", completed_state, lambda s: setattr(s, "last_second_duplicate_id", s.second_attack_id + 1)),
            ("complete", completed_state, lambda s: setattr(s.diagnostics, "second_hit_tick", -1)),
            ("complete", completed_state, lambda s: setattr(s.diagnostics, "achieved_safe_resolution", 0)),
            ("complete", completed_state, lambda s: setattr(s.diagnostics, "success", 0)),
            ("complete", completed_state, lambda s: setattr(s.diagnostics, "sequence_completed", 0)),
        )
        for index, (stage, factory, mutate) in enumerate(cases):
            with self.subTest(index=index, stage=stage):
                candidate = factory()
                mutate(candidate.state)
                self.assert_invalid_state(candidate)

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
        with self.assertRaisesRegex(RuntimeError, "403"):
            native.step_generated(np.zeros(10))

        from native_reengagement_episode import PlayerEvent

        native = self.native()
        native.reset_generated(native.scenario(134000, 0, -1))
        self.assertEqual(
            native.lib.ai2b2_debug_prepare_boundary_export(
                C.byref(native.state), C.byref(native.table), 1, 60, 120
            ),
            0,
        )
        result = native.step_generated(np.zeros(10))
        projection = native.diagnostics(result)
        self.assertEqual(projection["outcome"], None)
        self.assertEqual(projection["pursuit_tick"] - projection["relocation_tick"], 60)
        self.assertEqual(result.reward_terms[1], 0.5)
        self.assertEqual(result.reward_terms[6], 0.0)

        native = self.native()
        native.reset_generated(native.scenario(134000, 0, -1))
        self.assertEqual(
            native.lib.ai2b2_debug_prepare_boundary_export(
                C.byref(native.state), C.byref(native.table), 2, 90, 180
            ),
            0,
        )
        second = PlayerEvent(kind=4, source=0, target=1, move=0, power=0, cast=0,
                             hitbox=0, phase_player=0, event_sequence=0,
                             attack_id=222, amount=1.0)
        code, result = self.injected_step(native, 0, second)
        self.assertEqual(code, 0)
        projection = native.diagnostics(result)
        self.assertEqual(projection["outcome"], "success")
        self.assertEqual(projection["second_hit_tick"] - projection["relocation_tick"], 90)
        self.assertEqual(result.reward_terms[2], 1.0)
        self.assertEqual(result.reward_terms[3], 2.0)
        self.assertEqual(result.reward_terms[6], 0.0)
        with self.assertRaisesRegex(RuntimeError, "403"):
            native.step_generated(np.zeros(10))

        native = self.native()
        native.reset_generated(native.scenario(134000, 0, -1))
        self.assertEqual(
            native.lib.ai2b2_debug_prepare_boundary_export(
                C.byref(native.state), C.byref(native.table), 2, 89, 300
            ),
            0,
        )
        code, result = self.injected_step(native, 0, second)
        self.assertEqual(code, 0)
        projection = native.diagnostics(result)
        self.assertEqual(projection["outcome"], "success")
        self.assertEqual(projection["tick"], 300)
        self.assertEqual(result.reward_terms[2], 1.0)
        self.assertEqual(result.reward_terms[3], 2.0)
        self.assertEqual(result.reward_terms[6], 0.0)
        with self.assertRaisesRegex(RuntimeError, "403"):
            native.step_generated(np.zeros(10))

        native, result = self.episode("pursue_then_idle", seed=134000, slot=0, direction=-1)
        projection = native.diagnostics(result)
        self.assertEqual(projection["outcome"], "second_hit_deadline")
        self.assertIsNotNone(projection["relocation_tick"])
        self.assertIsNotNone(projection["pursuit_tick"])
        self.assertEqual(projection["tick"] - projection["relocation_tick"], 90)
        with self.assertRaisesRegex(RuntimeError, "403"):
            native.step_generated(np.zeros(10))

        native, result = self.episode("idle", seed=134000, slot=0, direction=-1)
        projection = native.diagnostics(result)
        self.assertEqual(projection["outcome"], "total_deadline")
        self.assertEqual(projection["tick"], 300)
        self.assertEqual(result.reward_terms[6], -1.0)
        with self.assertRaisesRegex(RuntimeError, "403"):
            native.step_generated(np.zeros(10))

        from reengagement_curriculum import choose_target_anchor

        self.assertEqual(choose_target_anchor(0, 10, 0.435, 5.0, 1), 9.365)
        self.assertEqual(choose_target_anchor(0, 10, 0.435, 5.0, -1), 0.635)
        self.assertEqual(choose_target_anchor(0, 3, 0.435, 1.5, 1), None)

    def test_relocation_distinguishes_pending_anchor_from_recreation_failure(self):
        from native_reengagement_episode import PlayerEvent

        native = self.native()
        native.reset_generated(native.scenario(134000, 0, -1))
        floor = native.state.game.scene.world.shapes[0]
        for index in range(floor.count):
            floor.v0[index].x = 0.0
        pending_code = self.relocation_probe().ai2b2_test_relocate_export(
            C.byref(native.state)
        )
        self.assertEqual(pending_code, 0)
        self.assertEqual(native.state.diagnostics.relocation_tick, -1)
        self.assertEqual(native.state.second_stage, 0)
        self.assertEqual(native.state.error, 0)

        native = self.native()
        native.reset_generated(native.scenario(134000, 0, -1))
        learner = native.state.scenario.player_slot
        target = 1 - learner
        target_shape = native.state.game.scene.players[target].shape
        native.state.game.scene.world.shape_active[target_shape] = 0
        before, rng_before = bytes(native.state), bytes(native.rng)
        event = PlayerEvent(
            kind=4,
            source=learner,
            target=target,
            move=0,
            power=0,
            cast=0,
            hitbox=0,
            phase_player=0,
            event_sequence=0,
            attack_id=333,
            amount=1.0,
        )
        code, result = self.injected_step(native, learner, event)
        self.assertEqual(code, 5)
        self.assertEqual(result.error, 5)
        self.assertEqual(bytes(native.state), before)
        self.assertEqual(bytes(native.rng), rng_before)
        self.assertEqual(result.reward, 0.0)
        self.assertTrue(np.all(np.asarray(result.reward_terms) == 0.0))
        self.assertEqual(result.diagnostics.relocation_tick, 0)
        self.assertEqual(result.diagnostics.first_hit_attack_id, 0)

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
        self.assertIn(projection["failure_category"], {
            None,
            "success",
            "own_stock_loss",
            "opponent_stock_loss",
            "unsafe_support",
            "response_deadline",
            "second_hit_deadline",
            "total_deadline",
            "first_hit_only",
            "failed_renewed_pursuit",
            "pursued_missed_second",
            "started_missed",
            "requested_no_start",
            "no_approach",
            "no_attack_request",
            "deadline",
        })

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
            check=False,
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

    def test_invalid_positive_damage_evidence_is_rejected(self):
        from native_reengagement_episode import PlayerEvent

        native = self.native()
        native.reset_generated(native.scenario(134000, 0, -1))

        def validate(event):
            result = native.lib.ai2b2_validate_event_export(C.byref(native.state), C.byref(event))
            return result

        valid = PlayerEvent(kind=4, source=0, target=1, move=0, power=0, cast=0,
                            hitbox=0, phase_player=0, event_sequence=0,
                            attack_id=11, amount=1.0)
        self.assertEqual(validate(valid), 0)
        no_event = PlayerEvent(kind=4, source=1, target=0, move=3, power=0, cast=0,
                               hitbox=0, phase_player=0, event_sequence=0,
                               attack_id=0, amount=0.0)
        self.assertEqual(validate(no_event), 0)
        for event in (
            PlayerEvent(kind=4, source=1, target=0, move=0, power=0, cast=0,
                        hitbox=0, phase_player=0, event_sequence=0, attack_id=11, amount=1.0),
            PlayerEvent(kind=4, source=0, target=1, move=0, power=0, cast=0,
                        hitbox=0, phase_player=0, event_sequence=0, attack_id=0, amount=1.0),
            PlayerEvent(kind=4, source=0, target=1, move=3, power=0, cast=0,
                        hitbox=0, phase_player=0, event_sequence=0, attack_id=11, amount=1.0),
        ):
            with self.subTest(event=bytes(event)):
                self.assertEqual(validate(event), 406)

        bad_target_stream_event = PlayerEvent(
            kind=4, source=1, target=0, move=0, power=0, cast=0,
            hitbox=0, phase_player=0, event_sequence=0, attack_id=12, amount=1.0
        )
        native.reset_generated(native.scenario(134000, 0, -1))
        before = bytes(native.state)
        code, result = self.injected_step(native, 1, bad_target_stream_event)
        self.assertEqual(code, 406)
        self.assertEqual(result.error, 406)
        self.assertEqual(bytes(native.state), before)


if __name__ == "__main__":
    unittest.main()
