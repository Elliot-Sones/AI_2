"""Movement seam capture tests for native navigation qualification."""

from __future__ import annotations

import json
import os
import ast
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Vec:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class Body:
    DYNAMIC = 0
    STATIC = 2

    def __init__(self):
        self.position = Vec(1.0, 2.0)
        self.velocity = Vec(0.0, 0.0)
        self.body_type = Body.DYNAMIC


class BB:
    def __init__(self, intersects=True):
        self._intersects = intersects

    def intersects(self, _other):
        return self._intersects


class Shape:
    def __init__(self, intersects=True):
        self._bb = BB(intersects)

    def cache_bb(self):
        return self._bb


class KeyStatus:
    def __init__(self, *, held=False, just_pressed=False, just_released=False):
        self.held = held
        self.just_pressed = just_pressed
        self.just_released = just_released


class HorizontalState:
    def __init__(self, value):
        self.value = value

    def get_last_int(self):
        return -1 if self.value in (1, 4) else 1 if self.value in (2, 3) else 0


class State:
    def __init__(self, name="StandingState"):
        self.name = name
        self.valid = True
        self.invincible_timer = 0
        self.dodge_cooldown = 0
        self.stun_time_stored = 0
        self.jump_timer = 0
        self.jumps_left = 2
        self.recoveries_left = 1
        self.is_base = True

    def __class__(self):  # pragma: no cover - kept explicit for intent, type() ignores instance attrs.
        return type(self.name, (), {})


def named_state(name: str, **attrs):
    cls = type(name, (), {})
    obj = cls()
    obj.valid = True
    obj.invincible_timer = 0
    obj.dodge_cooldown = 0
    obj.stun_time_stored = 0
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def fake_player(*, state_name="StandingState", platform_overlap=True):
    input_handler = NS(
        key_names=["W", "A", "S", "D", "space", "h", "l", "j", "k", "g"],
        key_status={
            "W": KeyStatus(),
            "A": KeyStatus(),
            "S": KeyStatus(),
            "D": KeyStatus(held=True, just_pressed=True),
            "space": KeyStatus(),
            "h": KeyStatus(),
            "l": KeyStatus(),
            "j": KeyStatus(),
            "k": KeyStatus(),
            "g": KeyStatus(),
        },
        raw_vertical=0.0,
        raw_horizontal=1.0,
        no_horizontal=False,
        horizontal_state=HorizontalState(2),
        last_direction=1,
    )
    body = Body()
    states = {
        "walking": named_state("WalkingState", jump_timer=0, jumps_left=2, recoveries_left=1, is_base=True),
        "standing": named_state("StandingState", jump_timer=0, jumps_left=2, recoveries_left=1, is_base=True),
        "turnaround": named_state("TurnaroundState", turnaround_timer=0),
        "air_turnaround": named_state("AirTurnaroundState", turnaround_timer=0),
        "sprinting": named_state("SprintingState", jump_timer=0, jumps_left=2, recoveries_left=1, is_base=True),
        "stun": named_state("StunState"),
        "in_air": named_state("InAirState", jump_timer=0, jumps_left=0, recoveries_left=0, is_base=True),
        "dodge": named_state("DodgeState", dodge_timer=0, is_grounded=True),
        "attack": named_state("AttackState"),
        "dash": named_state("DashState", dash_timer=0),
        "backdash": named_state("BackDashState", backdash_timer=0),
        "KO": named_state("KOState", timer=0),
        "taunt": named_state("TauntState"),
    }
    player = NS(
        agent_id=0,
        input=input_handler,
        body=body,
        shape=Shape(platform_overlap),
        start_position=(1.0, 2.0),
        prev_x=0.5,
        prev_y=1.5,
        facing=1,
        stocks=3,
        damage_velocity=(0.0, 0.0),
        target_vel=(0.0, 0.0),
        states=states,
        state=states[{v.__class__.__name__: k for k, v in states.items()}[state_name]],
        on_platform=None,
        grounded=True,
    )
    platform = NS(body=NS(velocity=Vec(0.25, 0.0)), shape=Shape(platform_overlap))
    player.env = NS(objects={"platform1": platform})
    player.is_on_floor = lambda: player.grounded

    def physics_process(delta):
        if delta < 0:
            raise RuntimeError("bad delta")
        player.body.velocity = Vec(6.75, 0.0)
        player.prev_x = player.body.position.x
        player.prev_y = player.body.position.y

    player.physics_process = physics_process
    return player


class MovementCaptureTests(unittest.TestCase):
    def setUp(self):
        import navigation_movement_capture as capture

        self.capture = capture

    def test_capture_player_records_decoded_input_bank_and_overlap_facts(self):
        record = self.capture.capture_player_movement(fake_player(), tick=3, player_label="players/0")

        self.assertEqual(record["input"]["decoded"][3], 1)
        self.assertEqual(record["input"]["decoded"][13], 1)
        self.assertEqual(record["input"]["decoded"][31:35], [1, 0, 2, 1])
        self.assertEqual(len(record["before"]["native_state"]["bank"]), 13)
        self.assertTrue(record["facts"]["grounded"])
        self.assertTrue(record["facts"]["platform_overlap"])
        self.assertEqual(record["facts"]["platform_velocity"], [0.25, 0.0])

    def test_recorder_patches_instance_method_restores_on_error_and_records_no_failed_step(self):
        player = fake_player()
        original = player.physics_process
        recorder = self.capture.PlayerMovementRecorder([player])
        with self.assertRaisesRegex(RuntimeError, "bad delta"):
            with recorder:
                player.physics_process(-1.0)

        self.assertIs(player.physics_process, original)
        self.assertEqual(recorder.records, [])

    def test_recorder_records_before_and_after_at_player_physics_seam(self):
        player = fake_player()
        recorder = self.capture.PlayerMovementRecorder([player])
        with recorder:
            player.physics_process(1 / 30)

        self.assertEqual(len(recorder.records), 1)
        row = recorder.records[0]
        self.assertEqual(row["before"]["native_state"]["vx"], 0.0)
        self.assertEqual(row["after"]["native_state"]["vx"], 6.75)
        self.assertEqual(row["dt"], 1 / 30)

    def test_recorder_restore_does_not_overwrite_newer_external_hook(self):
        player = fake_player()
        original = player.physics_process
        recorder = self.capture.PlayerMovementRecorder([player])

        def external_hook(delta):
            return original(delta)

        with recorder:
            player.physics_process = external_hook

        self.assertIs(player.physics_process, external_hook)

    def test_navigation_mask_removes_combat_buttons_and_raw_preserves_action(self):
        action = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1]

        self.assertEqual(
            self.capture.mask_action_for_mode(action, "navigation"),
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        )
        self.assertEqual(self.capture.mask_action_for_mode(action, "raw"), action)

    def test_slot_capture_treats_actual_frozen_callable_state_fields_as_absent(self):
        source = ROOT / "UTMIST-AI2-main" / "environment" / "environment.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        collisions = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or not node.name.endswith("State"):
                continue
            method_names = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
            for field in self.capture.SLOT_FIELDS:
                if field in method_names:
                    collisions.append((node.name, field))
        self.assertIn(("AirTurnaroundState", "is_base"), collisions)

        for class_name, field in collisions:
            state = type(class_name, (), {field: lambda self: True})()
            state.invincible_timer = 0
            state.dodge_cooldown = 0
            state.stun_time_stored = 0

            slot = self.capture._slot_for(state)

            self.assertIsNone(slot[field])

    def test_velocity_func_id_uses_pymunk_private_base_when_property_has_no_getter(self):
        class GetterlessBody:
            @staticmethod
            def update_velocity(*_args):
                return None

            @property
            def velocity_func(self):
                raise AttributeError("property 'velocity_func' of 'Body' object has no getter")

        body = GetterlessBody()
        body._velocity_func_base = None
        self.assertEqual(self.capture._velocity_func_id(body), 0)

        body._velocity_func_base = GetterlessBody.update_velocity
        self.assertEqual(self.capture._velocity_func_id(body), 0)

        body._velocity_func_base = lambda *_args: None
        self.assertEqual(self.capture._velocity_func_id(body), 1)

    def test_compare_with_native_rejects_unsupported_state_clearly(self):
        record = self.capture.capture_player_movement(fake_player(state_name="AttackState"), tick=0, player_label="players/0")

        result = self.capture.compare_record_with_native(record)

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "unsupported_movement_state")
        self.assertEqual(result["field"], "before.native_state.state_id")

    def test_compare_with_native_reports_first_field_difference(self):
        player = fake_player()
        before = self.capture.capture_player_movement(player, tick=0, player_label="players/0")
        import native_navigation_movement as movement

        expected = movement.step_movement(
            movement.default_config(),
            self.capture._state_to_native(movement, before["before"]["native_state"]),
            before["input"]["decoded"],
            grounded=before["facts"]["grounded"],
            platform_overlap=before["facts"]["platform_overlap"],
            platform_velocity=tuple(before["facts"]["platform_velocity"]),
        )
        record = {**before, "after": {"native_state": asdict(expected.state)}}
        self.assertTrue(self.capture.compare_record_with_native(record)["ok"])
        changed = json.loads(json.dumps(record))
        changed["after"]["native_state"]["vx"] += 0.25

        result = self.capture.compare_record_with_native(changed)

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "value_mismatch")
        self.assertEqual(result["field"], "after.native_state.vx")

    def test_compare_with_native_ignores_bank_slot_metadata_not_in_native_abi(self):
        player = fake_player()
        before = self.capture.capture_player_movement(player, tick=0, player_label="players/0")
        import native_navigation_movement as movement

        expected = movement.step_movement(
            movement.default_config(),
            self.capture._state_to_native(movement, before["before"]["native_state"]),
            before["input"]["decoded"],
            grounded=before["facts"]["grounded"],
            platform_overlap=before["facts"]["platform_overlap"],
            platform_velocity=tuple(before["facts"]["platform_velocity"]),
        )
        record = {**before, "after": {"native_state": asdict(expected.state)}}
        for state_id, slot in enumerate(record["after"]["native_state"]["bank"]):
            slot["state_id"] = state_id
            slot["state_name"] = f"synthetic-slot-{state_id}"

        result = self.capture.compare_record_with_native(record)

        self.assertTrue(result["ok"], result)

    def test_compare_with_native_still_rejects_actual_bank_field_difference(self):
        player = fake_player()
        before = self.capture.capture_player_movement(player, tick=0, player_label="players/0")
        import native_navigation_movement as movement

        expected = movement.step_movement(
            movement.default_config(),
            self.capture._state_to_native(movement, before["before"]["native_state"]),
            before["input"]["decoded"],
            grounded=before["facts"]["grounded"],
            platform_overlap=before["facts"]["platform_overlap"],
            platform_velocity=tuple(before["facts"]["platform_velocity"]),
        )
        record = {**before, "after": {"native_state": asdict(expected.state)}}
        record["after"]["native_state"]["bank"][0]["dodge_cooldown"] += 1

        result = self.capture.compare_record_with_native(record)

        self.assertFalse(result["ok"])
        self.assertEqual(result["field"], "after.native_state.bank[0].dodge_cooldown")

    def test_compare_records_with_native_reports_counts_and_failures(self):
        player = fake_player()
        before = self.capture.capture_player_movement(player, tick=0, player_label="players/0")
        import native_navigation_movement as movement

        expected = movement.step_movement(
            movement.default_config(),
            self.capture._state_to_native(movement, before["before"]["native_state"]),
            before["input"]["decoded"],
            grounded=before["facts"]["grounded"],
            platform_overlap=before["facts"]["platform_overlap"],
            platform_velocity=tuple(before["facts"]["platform_velocity"]),
        )
        good = {**before, "after": {"native_state": asdict(expected.state)}}
        bad = json.loads(json.dumps(good))
        bad["after"]["native_state"]["vy"] += 1.0

        summary = self.capture.compare_records_with_native([good, bad], backend="python")

        self.assertFalse(summary["ok"])
        self.assertEqual(summary["backend"], "python")
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["failures"][0]["index"], 1)
        self.assertEqual(summary["failures"][0]["field"], "after.native_state.vy")
        self.assertEqual(summary["first_failure"]["field"], "after.native_state.vy")

    def test_capture_game_is_bounded_hash_gated_writes_output_and_restores_cwd(self):
        def build(seed):
            self.assertEqual(seed, 4)
            return NS(players=[fake_player()], action_space=object(), reset=lambda seed: None, close=lambda: None)

        def actions(_space, steps, seed, n_agents):
            self.assertEqual((steps, seed, n_agents), (1, 9, 2))
            return [[0, 0, 0, 1, 0, 1, 1, 1, 1, 1]]

        def step(env, action):
            self.assertEqual(action, [0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
            env.players[0].physics_process(1 / 30)

        with tempfile.TemporaryDirectory(prefix="ai2-move-capture-") as tmp:
            root = Path(tmp) / "src"
            root.mkdir()
            source_file = root / "UTMIST-AI2-main" / "environment" / "environment.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("frozen-source\n")
            expected_sha = self.capture.sha256_file(source_file)
            out = Path(tmp) / "out" / "movement.json"
            before_cwd = Path.cwd()
            payload = self.capture.capture_game_movement(
                root,
                out,
                seed=4,
                action_seed=9,
                steps=1,
                source_sha256=expected_sha,
                prepare_source=lambda path: os.chdir(path),
                build_env=build,
                make_actions=actions,
                step_env=step,
            )

            self.assertEqual(Path.cwd(), before_cwd)
            written = json.loads(out.read_text())

            self.assertEqual(payload["records_captured"], 1)
            self.assertEqual(written["mask_mode"], "navigation")
            self.assertEqual(written["records"][0]["player_label"], "players/0")
            self.assertEqual(written["records"][0]["action_provenance"]["raw_action"], [0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
            self.assertEqual(written["records"][0]["action_provenance"]["executed_action"], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
            self.assertEqual(written["native_verification"]["backend"], "python")

    def test_cli_returns_nonzero_and_prints_first_failure_when_native_verification_fails(self):
        def fake_capture(**_kwargs):
            return {
                "records_captured": 2,
                "native_verification": {
                    "ok": False,
                    "first_failure": {"index": 1, "field": "after.native_state.vx", "reason": "value_mismatch"},
                },
            }

        stdout = StringIO()
        with redirect_stdout(stdout):
            code = self.capture.main(
                [
                    "--source-root",
                    "/tmp/source",
                    "--output",
                    "/tmp/out.json",
                    "--steps",
                    "1",
                    "--source-sha256",
                    "sha",
                ],
                capture_fn=fake_capture,
            )

        self.assertEqual(code, 2)
        self.assertIn("after.native_state.vx", stdout.getvalue())
        self.assertIn("\"ok\": false", stdout.getvalue())

    def test_cli_prints_capped_summary_but_preserves_failure_count(self):
        failures = [
            {"index": index, "field": f"after.native_state.bank[{index}].value", "reason": "value_mismatch"}
            for index in range(12)
        ]

        def fake_capture(**_kwargs):
            return {
                "records_captured": 12,
                "native_verification": {
                    "backend": "cuda",
                    "records": 12,
                    "ok": False,
                    "failures": failures,
                    "first_failure": failures[0],
                },
            }

        stdout = StringIO()
        with redirect_stdout(stdout):
            code = self.capture.main(
                [
                    "--source-root",
                    "/tmp/source",
                    "--output",
                    "/tmp/out.json",
                    "--steps",
                    "1",
                    "--source-sha256",
                    "sha",
                ],
                capture_fn=fake_capture,
            )

        printed = json.loads(stdout.getvalue())

        self.assertEqual(code, 2)
        self.assertEqual(printed["native_verification"]["failure_count"], 12)
        self.assertEqual(len(printed["native_verification"]["failure_sample"]), 5)
        self.assertNotIn("failures", printed["native_verification"])

    def test_capture_game_rejects_large_or_wrong_source_sha(self):
        with tempfile.TemporaryDirectory(prefix="ai2-move-capture-") as tmp:
            root = Path(tmp) / "src"
            source_file = root / "UTMIST-AI2-main" / "environment" / "environment.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("frozen-source\n")
            with self.assertRaisesRegex(ValueError, "steps"):
                self.capture.capture_game_movement(root, Path(tmp) / "out.json", steps=1201)
            with self.assertRaisesRegex(ValueError, "source sha"):
                self.capture.capture_game_movement(root, Path(tmp) / "out.json", steps=0, source_sha256="bad")

    def test_cli_requires_source_sha256_for_strict_source_gate(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                self.capture.main(["--source-root", "/tmp/source", "--output", "/tmp/out.json", "--steps", "0"])


class RealRuntimeMovementCaptureTests(unittest.TestCase):
    def setUp(self):
        try:
            import pymunk  # noqa: F401
        except ImportError:
            self.skipTest("Pymunk runtime unavailable")
        import navigation_movement_capture as capture

        self.capture = capture

    def test_player_fixture_capture_has_supported_contract_shape(self):
        record = self.capture.capture_player_movement(fake_player(), tick=0, player_label="players/0")

        self.assertEqual(record["input_width"], 35)
        self.assertEqual(record["before"]["native_state"]["state_id"], 1)
        self.assertEqual(len(record["before"]["native_state"]["bank"]), 13)


if __name__ == "__main__":
    unittest.main()
