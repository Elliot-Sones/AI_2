"""Tests for closed-loop native game patch validation harness."""

from __future__ import annotations

from ai2.paths import source_path

import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class ValidateNativeGamePatchTests(unittest.TestCase):
    def setUp(self):
        import tools.validation.validate_native_game_patch as validator

        self.validator = validator

    def test_raw_combat_script_contains_attack_and_navigation_script_does_not(self):
        raw = self.validator.raw_combat_labels(1200)
        navigation = self.validator.masked_navigation_labels(1200)

        self.assertGreaterEqual(raw.count("attack"), 10)
        self.assertGreaterEqual(raw.count("approach"), 10)
        self.assertIn("dodge", raw)
        self.assertIn("pickup", raw)
        self.assertIn("walk_left", raw)
        self.assertIn("walk_right", raw)
        self.assertEqual(raw[300], "reset")
        self.assertEqual(raw[600], "reset")
        self.assertEqual(raw[900], "reset")
        self.assertNotIn("attack", navigation)
        self.assertEqual(navigation[300], "reset")
        self.assertEqual(navigation[600], "reset")
        self.assertEqual(navigation[900], "reset")
        self.assertEqual(len(raw), 1200)
        self.assertEqual(len(navigation), 1200)

    def test_combat_coverage_action_walks_players_together_then_attacks(self):
        far_env = NS(players=[NS(body=NS(position=NS(x=-5.0))), NS(body=NS(position=NS(x=5.0)))])
        close_env = NS(players=[NS(body=NS(position=NS(x=-0.2))), NS(body=NS(position=NS(x=0.2)))])
        space = NS(shape=(10,), dtype=int)

        far = self.validator.combat_coverage_action(far_env, space, "attack")
        close = self.validator.combat_coverage_action(close_env, space, "attack")

        self.assertEqual(far["record_action"]["controller_intent"], "close_distance")
        self.assertEqual(far["record_action"]["0"][3], 1)
        self.assertEqual(far["record_action"]["1"][1], 1)
        self.assertEqual(close["record_action"]["controller_intent"], "attack")
        self.assertEqual(close["record_action"]["0"][3], 1)
        self.assertEqual(close["record_action"]["0"][7], 1)
        self.assertEqual(close["record_action"]["1"][1], 1)
        self.assertEqual(close["record_action"]["1"][7], 1)

    def test_initial_combat_fixture_places_players_on_ground1_with_provenance(self):
        env = FakeEnv([])

        fixture = self.validator.apply_initial_combat_fixture(env, reason="initial-reset")

        self.assertEqual(fixture["reason"], "initial-reset")
        self.assertEqual(fixture["surface"], "ground1")
        self.assertEqual(fixture["player_positions"], [[4.0, 0.0], [5.0, 0.0]])
        self.assertEqual(env.players[0].body.position, (4.0, 0.0))
        self.assertEqual(env.players[1].body.position, (5.0, 0.0))
        self.assertEqual(env.players[0].body.velocity, (0.0, 0.0))
        self.assertEqual(env.players[1].body.velocity, (0.0, 0.0))

    def test_raw_capture_applies_combat_fixture_after_initial_and_scripted_reset(self):
        calls: list[tuple] = []
        with tempfile.TemporaryDirectory(prefix="ai2-combat-fixture-") as tmp:
            source = make_source_root(Path(tmp))
            output = Path(tmp) / "raw.jsonl"
            replay = FakeReplayModule(calls)
            result = self.validator._capture_raw(
                source_root=source,
                config=None,
                output=output,
                seed=45,
                steps=302,
                native_patch=False,
                native_input=False,
                native_observation=False,
                replay=replay,
                bench=FakeBenchModule(calls),
                patch_module=None,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        metadata = rows[0]
        fixture_records = [row for row in rows[1:] if row["action"].get("combat_fixture")]
        self.assertEqual(result["records"], 302)
        self.assertEqual(metadata["actions"]["source"], "deterministic-fixtured-combat-native-game-patch-parity-v2")
        self.assertEqual(metadata["combat_fixture"]["surface"], "ground1")
        self.assertEqual(len(replay.hashed_actions), 302)
        self.assertEqual(replay.hashed_actions[0]["combat_fixture"]["reason"], "initial-reset")
        self.assertEqual(replay.hashed_actions[300]["combat_fixture"]["reason"], "scripted-reset")
        self.assertEqual(fixture_records[0]["tick"], 0)
        self.assertEqual(fixture_records[0]["action"]["combat_fixture"]["reason"], "initial-reset")
        self.assertEqual(fixture_records[-1]["tick"], 300)
        self.assertEqual(fixture_records[-1]["action"]["combat_fixture"]["reason"], "scripted-reset")

    def test_gameplay_coverage_detects_hit_from_damage_and_stun_from_state_timers(self):
        records = [
            {
                "discrete_events": [],
                "state": {
                    "players": [
                        {"state": "WalkingState", "state_timers": {}, "damage": 0, "damage_done": 12},
                        {"state": "StunState", "state_timers": {"stun_frames": 8}, "damage": 12, "stocks": 2},
                    ]
                },
            }
        ]

        coverage = self.validator.gameplay_coverage(records)

        self.assertEqual(coverage["missing"], [])
        self.assertEqual(coverage["found"], ["hit", "KO", "stun", "stun_recovery"])

    def test_gameplay_coverage_detects_reset_after_combat_from_trace(self):
        records = [
            {"transition": {"info": {}}, "state": {"players": [{"damage_done": 2}, {"damage": 2}]}},
            {"action": {"label": "reset"}, "transition": {"info": {"reset_observation": True}}, "state": {"players": []}},
        ]

        coverage = self.validator.gameplay_coverage(records, required=("hit", "reset_after_combat"))

        self.assertEqual(coverage["missing"], [])
        self.assertEqual(coverage["found"], ["hit", "reset_after_combat"])

    def test_zero_steps_and_empty_suite_inputs_fail_closed(self):
        with self.assertRaisesRegex(self.validator.ValidationError, "steps must be between 1"):
            self.validator.raw_combat_labels(0)
        with self.assertRaisesRegex(self.validator.ValidationError, "at least one mode"):
            self.validator.run_suite(
                source_root=Path("/src"),
                config=None,
                output_dir=Path("/tmp/out"),
                modes=[],
                seeds=[45],
                steps=1,
                wrapper_source=None,
                frame_stack=1,
            )
        with self.assertRaisesRegex(self.validator.ValidationError, "at least one seed"):
            self.validator.run_suite(
                source_root=Path("/src"),
                config=None,
                output_dir=Path("/tmp/out"),
                modes=["raw"],
                seeds=[],
                steps=1,
                wrapper_source=None,
                frame_stack=1,
            )

    def test_build_pair_commands_use_separate_processes_equal_seed_and_patch_flag(self):
        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-cmd-") as tmp:
            root = Path(tmp) / "src"
            config = Path(tmp) / "control.yaml"
            output = Path(tmp) / "out"
            commands = self.validator.build_pair_commands(
                script=Path("/tool/validate_native_game_patch.py"),
                source_root=root,
                config=config,
                output_dir=output,
                mode="raw",
                seed=45,
                steps=16,
                wrapper_source=None,
                frame_stack=1,
                native_input=True,
                native_observation=True,
            )

        original = commands["original"]
        native = commands["native"]
        self.assertIn("--native-patch=0", original)
        self.assertIn("--native-patch=1", native)
        self.assertNotIn("--native-input", original)
        self.assertNotIn("--native-observation", original)
        self.assertIn("--native-input", native)
        self.assertIn("--native-observation", native)
        self.assertEqual(original[original.index("--seed") + 1], "45")
        self.assertEqual(native[native.index("--seed") + 1], "45")
        self.assertNotEqual(original[original.index("--output") + 1], native[native.index("--output") + 1])

    def test_native_patch_install_receives_exact_frozen_source_before_mutation(self):
        calls = []
        fake_patch = FakePatchModule(calls)
        fake_replay = FakeReplayModule(calls)
        fake_bench = FakeBenchModule(calls)

        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-source-") as tmp:
            source = make_source_root(Path(tmp))
            self.validator.capture_run(
                source_root=source,
                config=None,
                wrapper_source=None,
                output=Path(tmp) / "native.jsonl",
                mode="raw",
                seed=7,
                steps=1,
                native_patch=True,
                native_input=True,
                native_observation=True,
                frame_stack=1,
                replay=fake_replay,
                bench=fake_bench,
                patch_module=fake_patch,
            )

        patch_call = next(item for item in calls if item[0] == "patch")
        self.assertEqual(patch_call[2], str((source / "UTMIST-AI2-main" / "environment" / "environment.py").resolve()))
        self.assertEqual(patch_call[3], "fake-frozen-sha")
        self.assertEqual(patch_call[4], True)
        self.assertEqual(patch_call[5], True)

    def test_native_patch_install_targets_raw_env_inside_vec_wrapper_chain(self):
        calls = []
        raw_env = NS(players=[NS(agent_id=0), NS(agent_id=1)])
        wrapped = NS(venv=NS(envs=[NS(raw_env=raw_env)]))

        class StrictPatchModule(FakePatchModule):
            def install_native_game_patch(self, env, **kwargs):
                calls.append(("patch_target", env is raw_env, hasattr(env, "players")))
                return super().install_native_game_patch(env, **kwargs)

        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-vec-") as tmp:
            source = make_source_root(Path(tmp))
            self.validator._install_native_patch(
                True,
                env=wrapped,
                source_root=source,
                native_input=True,
                native_observation=True,
                patch_module=StrictPatchModule(calls),
            )

        self.assertIn(("patch_target", True, True), calls)

    def test_native_patch_missing_source_fails_before_mutation(self):
        calls = []
        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-source-missing-") as tmp:
            source = make_source_root(Path(tmp))
            (source / "UTMIST-AI2-main" / "environment" / "environment.py").unlink()
            with self.assertRaisesRegex(self.validator.ValidationError, "missing frozen environment source"):
                self.validator.capture_run(
                    source_root=source,
                    config=None,
                    wrapper_source=None,
                    output=Path(tmp) / "native.jsonl",
                    mode="raw",
                    seed=7,
                    steps=1,
                native_patch=True,
                native_input=True,
                native_observation=True,
                frame_stack=1,
                replay=FakeReplayModule(calls),
                bench=FakeBenchModule(calls),
                    patch_module=FakePatchModule(calls),
                )

        self.assertFalse(any(call[0] == "patch" for call in calls))

    def test_capture_run_installs_patch_only_for_native_raw_and_records_combat_actions(self):
        calls = []
        fake_patch = FakePatchModule(calls)
        fake_replay = FakeReplayModule(calls)
        fake_bench = FakeBenchModule(calls)

        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-run-") as tmp:
            source = make_source_root(Path(tmp))
            out = Path(tmp) / "native.jsonl"
            result = self.validator.capture_run(
                source_root=source,
                config=None,
                wrapper_source=None,
                output=out,
                mode="raw",
                seed=7,
                steps=150,
                native_patch=True,
                native_input=True,
                native_observation=True,
                frame_stack=1,
                replay=fake_replay,
                bench=fake_bench,
                patch_module=fake_patch,
            )
            rows = [json.loads(line) for line in out.read_text().splitlines()]

        self.assertTrue(result["native_patch"]["enabled"])
        self.assertEqual(result["native_patch"]["enable_input"], True)
        self.assertEqual(result["native_patch"]["enable_observation"], True)
        self.assertEqual(result["native_patch"]["dropped_weapon_spawner_qualified_name"], "environment.environment.DroppedWeaponSpawner")
        self.assertTrue(any(call[:2] == ("patch", True) for call in calls))
        self.assertIn("attack", rows[0]["actions"]["script"])
        self.assertEqual(rows[0]["native_patch"]["enable_input"], True)
        self.assertEqual(rows[0]["native_patch"]["enable_observation"], True)
        self.assertEqual(rows[0]["native_patch"]["dropped_weapon_spawner_qualified_name"], "environment.environment.DroppedWeaponSpawner")
        self.assertEqual(result["gameplay_coverage"]["missing"], ["hit", "KO", "stun"])
        self.assertEqual(rows[0]["rng"]["seed"], 7)
        attack_rows = [record for record in rows[1:] if record["action"].get("label") == "attack"]
        self.assertTrue(attack_rows)
        self.assertIn(attack_rows[0]["action"].get("controller_intent"), {"close_distance", "attack"})
        self.assertEqual(("reset", 7), calls[calls.index(("reset", 7))])
        self.assertIn(("patch_restore",), calls)

    def test_capture_run_records_first_reset_observation_instead_of_step_after_reset(self):
        calls = []
        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-reset-") as tmp:
            source = make_source_root(Path(tmp))
            out = Path(tmp) / "original.jsonl"
            result = self.validator.capture_run(
                source_root=source,
                config=None,
                wrapper_source=None,
                output=out,
                mode="raw",
                seed=7,
                steps=305,
                native_patch=False,
                native_input=False,
                native_observation=False,
                frame_stack=1,
                replay=FakeReplayModule(calls),
                bench=FakeBenchModule(calls),
                patch_module=FakePatchModule(calls),
            )
            rows = [json.loads(line) for line in out.read_text().splitlines()]

        reset_row = rows[301]
        self.assertEqual(result["records"], 305)
        self.assertEqual(reset_row["tick"], 300)
        self.assertEqual(reset_row["action"]["label"], "reset")
        self.assertEqual(reset_row["transition"]["observation"], ["reset_obs", 307])
        self.assertTrue(reset_row["transition"]["info"]["reset_observation"])
        self.assertNotIn(("step", reset_row["action"]), calls)

    def test_capture_run_original_does_not_install_native_patch(self):
        calls = []
        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-run-") as tmp:
            source = make_source_root(Path(tmp))
            self.validator.capture_run(
                source_root=source,
                config=None,
                wrapper_source=None,
                output=Path(tmp) / "original.jsonl",
                mode="raw",
                seed=2,
                steps=2,
                native_patch=False,
                native_input=True,
                native_observation=True,
                frame_stack=1,
                replay=FakeReplayModule(calls),
                bench=FakeBenchModule(calls),
                patch_module=FakePatchModule(calls),
            )

        self.assertNotIn(("patch", True), calls)

    def test_compare_pair_writes_failure_receipt_with_first_difference(self):
        fake_replay = NS(
            compare_replay_files=lambda *_args, **_kwargs: NS(
                ok=False,
                reason="value_mismatch",
                message="Replay records diverged",
                compared_records=3,
                first_difference={"tick": 2, "field": "state.players[0].grounded", "left": False, "right": True},
            )
        )
        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-compare-") as tmp:
            receipt = Path(tmp) / "receipt.json"
            result = self.validator.compare_pair(Path(tmp) / "original.jsonl", Path(tmp) / "native.jsonl", receipt, replay=fake_replay)
            written = json.loads(receipt.read_text())

        self.assertFalse(result["ok"])
        self.assertEqual(written["status"], "failed")
        self.assertEqual(written["first_difference"]["field"], "state.players[0].grounded")

    def test_compare_pair_uses_real_replay_files_and_reports_missing_gameplay_coverage(self):
        import ai2.evaluation.benchmark_replay as replay

        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-real-replay-") as tmp:
            root = make_source_root(Path(tmp))
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            receipt = Path(tmp) / "receipt.json"
            metadata = replay.default_metadata(root, None, 3, 3, "same", "raw", 1, action_script=["idle"])
            record = valid_replay_record(replay)
            replay.write_jsonl(left, metadata, [record])
            replay.write_jsonl(right, metadata, [record])

            result = self.validator.compare_pair(left, right, receipt, replay=replay)
            written = json.loads(receipt.read_text())

        self.assertTrue(result["ok"])
        self.assertEqual(written["gameplay_coverage"]["original"]["missing"], ["hit", "KO", "stun"])
        self.assertEqual(written["compared_records"], 1)

    def test_compare_pair_fails_when_required_gameplay_coverage_is_missing(self):
        import ai2.evaluation.benchmark_replay as replay

        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-required-coverage-") as tmp:
            root = make_source_root(Path(tmp))
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            receipt = Path(tmp) / "receipt.json"
            metadata = replay.default_metadata(root, None, 3, 3, "same", "raw", 1, action_script=["idle"])
            record = valid_replay_record(replay)
            replay.write_jsonl(left, metadata, [record])
            replay.write_jsonl(right, metadata, [record])

            result = self.validator.compare_pair(
                left,
                right,
                receipt,
                replay=replay,
                required_gameplay_coverage=("hit", "stun", "reset_after_combat"),
            )
            written = json.loads(receipt.read_text())

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "missing_gameplay_coverage")
        self.assertEqual(written["status"], "failed")
        self.assertEqual(written["missing_gameplay_coverage"]["original"], ["hit", "stun", "reset_after_combat"])

    def test_required_gameplay_coverage_cli_argument_is_forwarded_to_suite(self):
        seen = []

        def fake_runner(command, timeout_seconds):
            seen.append((command, timeout_seconds))
            return {"command": list(command), "returncode": 1, "stdout_tail": "", "stderr_tail": "nope"}

        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-required-cli-") as tmp:
            result = self.validator.run_suite(
                source_root=Path(tmp) / "src",
                config=None,
                output_dir=Path(tmp) / "out",
                modes=["raw"],
                seeds=[45],
                steps=1,
                wrapper_source=None,
                frame_stack=1,
                child_timeout=123,
                required_gameplay_coverage=("hit",),
                command_runner=fake_runner,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["required_gameplay_coverage"], ["hit"])

    def test_compare_pair_rejects_mismatched_original_and_native_source_hashes(self):
        import ai2.evaluation.benchmark_replay as replay

        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-source-compare-") as tmp:
            root = make_source_root(Path(tmp))
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            receipt = Path(tmp) / "receipt.json"
            left_metadata = replay.default_metadata(root, None, 3, 3, "same", "raw", 1, action_script=["idle"])
            right_metadata = json.loads(json.dumps(left_metadata))
            right_metadata["source"]["environment_sha256"] = "different"
            record = valid_replay_record(replay)
            replay.write_jsonl(left, left_metadata, [record])
            replay.write_jsonl(right, right_metadata, [record])

            result = self.validator.compare_pair(left, right, receipt, replay=replay)

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "source_mismatch")

    def test_run_suite_passes_child_timeout_to_each_subprocess_runner(self):
        seen = []

        def fake_runner(command, timeout_seconds):
            seen.append((command, timeout_seconds))
            return {"command": list(command), "returncode": 1, "stdout_tail": "", "stderr_tail": "nope"}

        with tempfile.TemporaryDirectory(prefix="ai2-game-patch-timeout-") as tmp:
            self.validator.run_suite(
                source_root=Path(tmp) / "src",
                config=None,
                output_dir=Path(tmp) / "out",
                modes=["raw"],
                seeds=[45],
                steps=1,
                wrapper_source=None,
                frame_stack=1,
                child_timeout=123,
                native_input=True,
                native_observation=True,
                command_runner=fake_runner,
            )

        self.assertEqual([timeout for _command, timeout in seen], [123, 123])


def make_source_root(root: Path) -> Path:
    source = root / "source"
    env = source / "UTMIST-AI2-main" / "environment"
    env.mkdir(parents=True)
    (env / "__init__.py").write_text("", encoding="utf-8")
    (env / "environment.py").write_text(
        "class Player: pass\nclass DroppedWeaponSpawner: pass\n",
        encoding="utf-8",
    )
    (source_path(source, "train.py")).write_text("# wrapper\n", encoding="utf-8")
    return source


class FakeEnv:
    def __init__(self, calls):
        self.calls = calls
        self.action_space = NS(shape=(10,), dtype=int)
        self.objects = {"ground1": NS(body=NS(position=NS(x=4.5, y=1.0)))}
        self.players = [
            NS(agent_id=0, body=NS(position=NS(x=-5.0, y=0.0), velocity=(1.0, 2.0)), prev_x=-5.0, prev_y=0.0),
            NS(agent_id=1, body=NS(position=NS(x=5.0, y=0.0), velocity=(-1.0, 2.0)), prev_x=5.0, prev_y=0.0),
        ]
        self.steps = 0

    def reset(self, seed=None):
        self.calls.append(("reset", seed))
        self.steps = 0
        self.players[0].body.position = NS(x=-5.0, y=0.0)
        self.players[1].body.position = NS(x=5.0, y=0.0)
        return ["reset_obs", seed], {"reset_seed": seed}

    def step(self, action):
        self.steps += 1
        self.calls.append(("step", action))
        return ["obs", 0.0, False, False, {"step": self.steps}]

    def close(self):
        self.calls.append(("close",))


class FakeBenchModule:
    def __init__(self, calls):
        self.calls = calls

    def set_headless_environment(self):
        self.calls.append(("headless",))

    def seed_everything(self, seed):
        self.calls.append(("seed", seed))

    def build_raw_env(self, seed, *, source_root=None):
        self.calls.append(("build_raw", seed))
        return FakeEnv(self.calls)


class FakeReplayModule:
    REPLAY_SCHEMA = "ai2.replay.e0.v1"
    GAME_CONTRACT_VERSION = "warehouse-brawl-python-e0"

    def __init__(self, calls):
        self.calls = calls

    def prepare_source_imports(self, source_root):
        self.calls.append(("prepare", str(source_root)))
        package_root = str(Path(source_root) / "UTMIST-AI2-main")
        if package_root not in sys.path:
            sys.path.insert(0, package_root)

    def hash_actions(self, actions):
        self.hashed_actions = list(actions)
        return str(len(self.hashed_actions))

    def default_metadata(self, source_root, config, seed, action_seed, action_sha256, mode, steps, **kwargs):
        return {
            "type": "metadata",
            "schema": self.REPLAY_SCHEMA,
            "game_contract_version": self.GAME_CONTRACT_VERSION,
            "source": {"root": str(source_root), "environment_sha256": "fake"},
            "config": {"path": str(config) if config else None, "sha256": None, "phase": kwargs.get("phase")},
            "rng": {"seed": seed, "action_seed": action_seed, "action_sha256": action_sha256},
            "actions": {"source": kwargs.get("action_source"), "script": list(kwargs.get("action_script") or [])},
            "controls": {"kind": "raw_multibinary_10", "player_indices": [0, 1]},
            "capture": {"mode": mode, "steps": steps, "frame_stack": kwargs.get("frame_stack"), "contract_scope": "test", "excluded_state": [], "excluded_runtime_fields": []},
        }

    def make_replay_record(self, tick, action, transition, env, **_kwargs):
        return {
            "type": "record",
            "schema": self.REPLAY_SCHEMA,
            "tick": tick,
            "action": action,
            "transition": {"observation": transition[0], "reward": transition[1], "terminated": False, "truncated": False, "info": transition[4]},
            "state": {"env": {"steps": env.steps}, "players": [], "moving_platforms": [], "contacts": [], "spawners": [], "wrapper": {}, "coverage": {}},
            "discrete_events": [],
        }

    def write_jsonl(self, path, metadata, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, sort_keys=True) + "\n")
            count = 0
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1
        return count


class FakePatchModule:
    FROZEN_ENV_SHA256 = "fake-frozen-sha"

    def __init__(self, calls):
        self.calls = calls

    def install_from_environment_module(self, _module, enabled=False):
        self.calls.append(("patch", enabled))

        class Active:
            def restore(inner_self):
                self.calls.append(("patch_restore",))

        return Active()

    def install_native_game_patch(self, env, **kwargs):
        self.calls.append(
            (
                "patch",
                kwargs.get("enabled"),
                str(kwargs.get("source_path")),
                kwargs.get("expected_sha256"),
                kwargs.get("enable_input"),
                kwargs.get("enable_observation"),
            )
        )
        env._fake_patch_installed = True

        class Active:
            def restore(inner_self):
                self.calls.append(("patch_restore",))

            def receipt(inner_self):
                return {"attached": True}

        return Active()

    def describe_native_game_patch(self):
        return {"patched_methods": ["Player.is_on_floor"]}


def valid_replay_record(replay) -> dict:
    player = {
        "agent_id": 0,
        "state": "IdleState",
        "state_timers": {},
        "facing": 1,
        "weapon": "none",
        "stocks": 3,
        "damage": 0,
        "damage_taken_this_stock": 0,
        "damage_taken_total": 0,
        "damage_done": 0,
        "position": [0.0, 0.0],
        "velocity": [0.0, 0.0],
        "smooth_x_vel": 0.0,
        "target_vel": [0.0, 0.0],
        "damage_velocity": [0.0, 0.0],
        "pickup_lock_until": 0,
        "support": None,
        "grounded": True,
        "cur_action": [0] * 10,
    }
    return {
        "type": "record",
        "schema": replay.REPLAY_SCHEMA,
        "tick": 0,
        "action": {0: [0] * 10, 1: [0] * 10},
        "transition": {"observation": [0.0], "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
        "state": {
            "env": {"steps": 1, "seed": 3, "max_timesteps": 1200, "terminated": False, "rewards": [0.0, 0.0]},
            "players": [dict(player), {**player, "agent_id": 1}],
            "moving_platforms": [
                {
                    "id": "platform1",
                    "name": "platform1",
                    "body_id": 10,
                    "position": [0.0, 0.0],
                    "velocity": [1.0, 0.0],
                    "moving_to_w2": True,
                    "waypoint1": [0.0, 0.0],
                    "waypoint2": [10.0, 0.0],
                }
            ],
            "contacts": [],
            "spawners": [],
            "wrapper": {},
            "coverage": replay.coverage_ledger(["idle"], verified_events=[], required_event_gates=()),
            "solver_note": replay.SOLVER_LIMIT_NOTE,
        },
        "discrete_events": [],
    }


if __name__ == "__main__":
    unittest.main()
