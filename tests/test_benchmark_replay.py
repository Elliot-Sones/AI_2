"""Tests for the replay trace contract and fail-closed comparator."""

from __future__ import annotations

from ai2.paths import source_path

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ai2.evaluation.benchmark_replay as replay
import tools.benchmarks.benchmark_contact_fixtures as fixtures


class _Vec:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _BB:
    def __init__(self, intersects=True):
        self._intersects = intersects
        self.left = -0.5
        self.right = 0.5
        self.bottom = -0.5
        self.top = 0.5

    def intersects(self, _other):
        return self._intersects


class _Shape:
    def __init__(self, owner=None, platform_id=None, intersects=True, collision_type=0):
        self.owner = owner
        self.platform_id = platform_id
        self.collision_type = collision_type
        self._bb = _BB(intersects)

    def cache_bb(self):
        return self._bb


class _Body:
    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0):
        self.position = _Vec(x, y)
        self.velocity = _Vec(vx, vy)
        self._arbiters = []

    def each_arbiter(self, callback):
        for arbiter in self._arbiters:
            callback(arbiter)


class _FakeArbiter:
    def __init__(self, shapes, normal=(0.0, 1.0), points=None):
        self.shapes = shapes
        contact_points = [
            type("Point", (), {"point_a": _Vec(point[0], point[1]), "point_b": _Vec(point[0], point[1]), "distance": 0.0})()
            for point in (points or [])
        ]
        self.contact_point_set = type("ContactSet", (), {"normal": _Vec(*normal), "points": contact_points})()
        self.total_impulse = _Vec(0.0, 0.0)
        self.total_ke = 0.0


def _fake_platform_env(
    *,
    geometric=True,
    arbiter=False,
    player_agent_id=0,
    platform_id=1,
    platform_velocity=(0.2, -0.1),
    moving_to_w2=True,
    player_y=0.9,
):
    raw = type("Raw", (), {})()
    raw.steps = 0
    raw.seed = 7
    raw.max_timesteps = 9000
    raw.terminated = False
    raw.rewards = {0: 0.0, 1: 0.0}
    raw.weapon_controller = type("WeaponController", (), {"spawners": []})()
    platform = type("Platform", (), {})()
    platform.body = _Body(0.0, 1.0, platform_velocity[0], platform_velocity[1])
    platform.shape = _Shape(platform, platform_id=platform_id, intersects=geometric, collision_type=3)
    platform.shape.body = platform.body
    platform.waypoint1 = (1.0, 0.0)
    platform.waypoint2 = (-1.0, 2.0)
    platform.moving_to_w2 = moving_to_w2
    player = type("Player", (), {})()
    player.agent_id = player_agent_id
    player.body = _Body(0.0, player_y, 0.0, 0.0)
    player.shape = _Shape(player, intersects=geometric, collision_type=1)
    player.shape.body = player.body
    player.on_platform = None
    player.state = type("State", (), {"jumps_left": 2, "recoveries_left": 1})()
    player.facing = "Facing.RIGHT"
    player.weapon = "Punch"
    player.stocks = 3
    player.damage = 0
    player.damage_taken_this_stock = 0
    player.damage_taken_total = 0
    player.damage_done = 0
    player.smoothXVel = 0
    player.target_vel = (0.0, 0.0)
    player.damage_velocity = (0.0, 0.0)
    player.pickup_lock_until = None
    player.cur_action = [0] * 10
    player.is_on_floor = lambda: geometric
    opponent = copy.copy(player)
    opponent.agent_id = 1
    opponent.body = _Body(5.0, 0.0, 0.0, 0.0)
    opponent.shape = _Shape(opponent, intersects=False, collision_type=2)
    opponent.shape.body = opponent.body
    if arbiter:
        player.body._arbiters.append(_FakeArbiter((player.shape, platform.shape), points=[(0.5, 0.5)]))
    raw.players = [player, opponent]
    raw.objects = {"platform1": platform}
    return raw


def replay_source_root() -> Path:
    return Path(os.environ.get("AI2_REPLAY_SOURCE", ROOT)).resolve()


def wrapper_source_path() -> Path:
    configured = os.environ.get("AI2_REPLAY_WRAPPER_SOURCE")
    if configured:
        path = Path(configured).resolve()
        if path.is_dir():
            return source_path(path, "train.py")
        return path
    remote_layout = ROOT / "variants/corrected/train.py"
    if remote_layout.exists():
        return remote_layout
    return (
        ROOT.parent
        / "AI_2-learning-validation-20260905"
        / "baselines/navigation-repair-20260905/payload/variants/corrected/train.py"
    ).resolve()


def wrapper_config_path() -> Path:
    configured = os.environ.get("AI2_REPLAY_CONFIG")
    if configured:
        return Path(configured).resolve()
    return wrapper_source_path().with_name("control.yaml")


def _metadata(**overrides):
    metadata = {
        "type": "metadata",
        "schema": replay.REPLAY_SCHEMA,
        "source": {"root": "/tmp/source", "environment_sha256": "env-a"},
        "config": {"path": "config.yaml", "sha256": "cfg-a", "phase": None},
        "rng": {"seed": 42, "action_seed": 42, "action_sha256": "actions-a", "reset_seed_policy": "seed+tick+1"},
        "actions": {"source": "benchmark_training.action_plan", "script": None},
        "controls": {"kind": "raw_multibinary_10", "player_indices": [0, 1]},
        "game_contract_version": replay.GAME_CONTRACT_VERSION,
        "capture": {
            "mode": "raw",
            "steps": 2,
            "frame_stack": None,
            "contract_scope": replay.CONTRACT_SCOPE,
            "excluded_state": list(replay.EXCLUDED_STATE_SCOPES),
            "excluded_runtime_fields": list(replay.EXCLUDED_RUNTIME_FIELDS),
            "note": replay.SOLVER_LIMIT_NOTE,
        },
    }
    metadata.update(overrides)
    return metadata


def _record(tick, *, x=1.0, event="idle", reward=0.0):
    return {
        "type": "record",
        "schema": replay.REPLAY_SCHEMA,
        "tick": tick,
        "action": {"0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        "transition": {
            "reward": {"0": reward, "1": 0.0},
            "terminated": False,
            "truncated": False,
        },
        "state": {
            "env": {"steps": tick, "seed": 42, "max_timesteps": 9000, "terminated": False, "rewards": {"0": reward, "1": 0.0}},
            "players": [
                {
                    "agent_id": 0,
                    "position": [x, 2.0],
                    "velocity": [0.0, 0.0],
                    "state": "StandingState",
                    "state_timers": {"jumps_left": 2, "recoveries_left": 1},
                    "facing": "Facing.RIGHT",
                    "weapon": "Punch",
                    "stocks": 3,
                    "damage": 0,
                    "damage_taken_this_stock": 0,
                    "damage_taken_total": 0,
                    "damage_done": 0,
                    "smooth_x_vel": 0,
                    "target_vel": [0.0, 0.0],
                    "damage_velocity": [0.0, 0.0],
                    "pickup_lock_until": None,
                    "support": None,
                    "grounded": True,
                    "cur_action": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                {
                    "agent_id": 1,
                    "position": [-1.0, 2.0],
                    "velocity": [0.0, 0.0],
                    "state": "StandingState",
                    "state_timers": {"jumps_left": 2, "recoveries_left": 1},
                    "facing": "Facing.LEFT",
                    "weapon": "Punch",
                    "stocks": 3,
                    "damage": 0,
                    "damage_taken_this_stock": 0,
                    "damage_taken_total": 0,
                    "damage_done": 0,
                    "smooth_x_vel": 0,
                    "target_vel": [0.0, 0.0],
                    "damage_velocity": [0.0, 0.0],
                    "pickup_lock_until": None,
                    "support": None,
                    "grounded": True,
                    "cur_action": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
            ],
            "moving_platforms": [{"id": 1, "name": "platform1", "body_id": "platform:1", "position": [0.0, 1.0], "velocity": [0.0, 0.0], "moving_to_w2": True, "waypoint1": [1.0, 0.0], "waypoint2": [-1.0, 2.0]}],
            "contacts": [],
            "spawners": [],
            "wrapper": {"chain": []},
            "coverage": replay.coverage_ledger(["startup"], verified_events=[]),
            "solver_note": replay.SOLVER_LIMIT_NOTE,
        },
        "discrete_events": [event],
    }


def _write_trace(path: Path, metadata=None, records=None):
    lines = [metadata or _metadata()]
    lines.extend(records if records is not None else [_record(0), _record(1)])
    path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in lines) + "\n")


class ReplayComparatorTest(unittest.TestCase):
    def test_identical_records_compare_equal_and_report_record_count(self):
        with tempfile.TemporaryDirectory(prefix="ai2-replay-test-") as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            _write_trace(left)
            _write_trace(right)

            result = replay.compare_replay_files(left, right)

        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.compared_records, 2)
        self.assertIsNone(result.first_difference)

    def test_first_float_difference_reports_tick_and_field(self):
        with tempfile.TemporaryDirectory(prefix="ai2-replay-test-") as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            records = [_record(0), _record(1)]
            changed = copy.deepcopy(records)
            changed[1]["state"]["players"][0]["position"][0] = 1.25
            _write_trace(left, records=records)
            _write_trace(right, records=changed)

            result = replay.compare_replay_files(left, right)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "value_mismatch")
        self.assertEqual(result.first_difference["tick"], 1)
        self.assertEqual(result.first_difference["field"], "state.players[0].position[0]")

    def test_float_tolerance_is_explicit_and_default_is_exact(self):
        with tempfile.TemporaryDirectory(prefix="ai2-replay-test-") as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            records = [_record(0)]
            changed = copy.deepcopy(records)
            changed[0]["state"]["players"][0]["position"][0] = 1.00001
            _write_trace(left, records=records)
            _write_trace(right, records=changed)

            exact = replay.compare_replay_files(left, right)
            tolerant = replay.compare_replay_files(left, right, abs_tol=0.001, rel_tol=0.0)

        self.assertFalse(exact.ok)
        self.assertTrue(tolerant.ok, tolerant.message)

    def test_monitor_episode_wallclock_t_is_excluded_but_episode_reward_length_compare(self):
        with tempfile.TemporaryDirectory(prefix="ai2-replay-test-") as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            records = [_record(0)]
            changed = copy.deepcopy(records)
            records[0]["transition"]["info"] = {"episode": {"r": 3.0, "l": 81, "t": 0.051536}}
            changed[0]["transition"]["info"] = {"episode": {"r": 3.0, "l": 81, "t": 0.051712}}
            _write_trace(left, records=records)
            _write_trace(right, records=changed)

            result = replay.compare_replay_files(left, right)

        self.assertTrue(result.ok, result.message)

        with tempfile.TemporaryDirectory(prefix="ai2-replay-test-") as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            records = [_record(0)]
            changed = copy.deepcopy(records)
            records[0]["transition"]["info"] = {"episode": {"r": 3.0, "l": 81, "t": 0.051536}}
            changed[0]["transition"]["info"] = {"episode": {"r": 4.0, "l": 81, "t": 0.051712}}
            _write_trace(left, records=records)
            _write_trace(right, records=changed)

            result = replay.compare_replay_files(left, right)

        self.assertFalse(result.ok)
        self.assertEqual(result.first_difference["field"], "transition.info.episode.r")

    def test_transition_info_numeric_differences_outside_monitor_wallclock_still_fail(self):
        with tempfile.TemporaryDirectory(prefix="ai2-replay-test-") as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            records = [_record(0)]
            changed = copy.deepcopy(records)
            records[0]["transition"]["info"] = {"nav_distance": 1.0, "episode": {"r": 0.0, "l": 1, "t": 0.1}}
            changed[0]["transition"]["info"] = {"nav_distance": 1.25, "episode": {"r": 0.0, "l": 1, "t": 0.2}}
            _write_trace(left, records=records)
            _write_trace(right, records=changed)

            result = replay.compare_replay_files(left, right)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "value_mismatch")
        self.assertEqual(result.first_difference["field"], "transition.info.nav_distance")

    def test_rejects_zero_compared_records(self):
        with tempfile.TemporaryDirectory(prefix="ai2-replay-test-") as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            _write_trace(left, records=[])
            _write_trace(right, records=[])

            result = replay.compare_replay_files(left, right)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "zero_compared_records")

    def test_rejects_schema_mismatch_missing_keys_length_mismatch_nonfinite_and_metadata(self):
        with tempfile.TemporaryDirectory(prefix="ai2-replay-test-") as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"

            _write_trace(left)
            _write_trace(right, metadata=_metadata(schema="ai2.replay.v999"))
            self.assertEqual(replay.compare_replay_files(left, right).reason, "schema_mismatch")

            bad_records = [_record(0)]
            del bad_records[0]["state"]
            _write_trace(right, records=bad_records)
            self.assertEqual(replay.compare_replay_files(left, right).reason, "missing_keys")

            _write_trace(right, records=[_record(0)])
            self.assertEqual(replay.compare_replay_files(left, right).reason, "length_mismatch")

            bad_records = [_record(0), _record(1)]
            bad_records[0]["state"]["players"][0]["position"][0] = float("nan")
            _write_trace(right, records=bad_records)
            self.assertEqual(replay.compare_replay_files(left, right).reason, "nonfinite_values")

            _write_trace(right, metadata=_metadata(rng={"seed": 42, "action_seed": 42, "action_sha256": "actions-b"}))
            self.assertEqual(replay.compare_replay_files(left, right).reason, "incompatible_comparison_metadata")

            _write_trace(right, metadata=_metadata(game_contract_version="native-port-draft"))
            self.assertEqual(replay.compare_replay_files(left, right).reason, "unsupported_game_contract_version")

            _write_trace(right, metadata=_metadata(source={"root": "/tmp/port", "environment_sha256": "env-b"}))
            self.assertTrue(replay.compare_replay_files(left, right).ok)

    def test_missing_required_nested_state_fields_fail_even_when_both_sides_omit_them(self):
        with tempfile.TemporaryDirectory(prefix="ai2-replay-test-") as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            bad_records = [_record(0)]
            del bad_records[0]["state"]["players"][0]["state_timers"]
            _write_trace(left, records=bad_records)
            _write_trace(right, records=copy.deepcopy(bad_records))

            result = replay.compare_replay_files(left, right)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "missing_keys")
        self.assertEqual(result.first_difference["field"], "state.players[0].state_timers")

    def test_null_required_state_anchors_fail_validation_but_declared_nullable_fields_remain_allowed(self):
        valid = _record(0)
        replay.validate_record(valid)
        self.assertIsNone(valid["state"]["players"][0]["support"])
        self.assertIsNone(valid["state"]["players"][0]["pickup_lock_until"])

        cases = [
            ("state.players[0].position", lambda r: r["state"]["players"][0].__setitem__("position", None)),
            ("state.players[0].velocity", lambda r: r["state"]["players"][0].__setitem__("velocity", None)),
            ("state.players[0].state", lambda r: r["state"]["players"][0].__setitem__("state", None)),
            ("state.players[0].cur_action", lambda r: r["state"]["players"][0].__setitem__("cur_action", None)),
            ("state.moving_platforms[0].body_id", lambda r: r["state"]["moving_platforms"][0].__setitem__("body_id", None)),
            ("state.moving_platforms[0].waypoint1", lambda r: r["state"]["moving_platforms"][0].__setitem__("waypoint1", None)),
            ("action.0", lambda r: r["action"].__setitem__("0", None)),
        ]
        for field, mutate in cases:
            bad = _record(0)
            mutate(bad)
            with self.subTest(field=field):
                with self.assertRaises(replay.ReplayValidationError) as ctx:
                    replay.validate_record(bad)
                self.assertEqual(ctx.exception.reason, "invalid_required_anchor")
                self.assertEqual(ctx.exception.field, field)

    def test_discrete_events_compare_exactly(self):
        with tempfile.TemporaryDirectory(prefix="ai2-replay-test-") as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            _write_trace(left, records=[_record(0, event="jump")])
            _write_trace(right, records=[_record(0, event="Jump")])

            result = replay.compare_replay_files(left, right)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "value_mismatch")
        self.assertEqual(result.first_difference["field"], "discrete_events[0]")

    def test_fixture_action_script_and_coverage_ledger_separates_dispatched_labels_from_verified_events(self):
        space = type("Space", (), {"shape": (10,), "dtype": float})()
        actions = replay.scripted_raw_actions(space, replay.REQUIRED_FIXTURE_LABELS)

        self.assertEqual(len(actions), len(replay.REQUIRED_FIXTURE_LABELS))
        ledger = replay.coverage_ledger(replay.REQUIRED_FIXTURE_LABELS, verified_events=[])
        self.assertEqual(ledger["dispatched_fixture_labels"], list(replay.REQUIRED_FIXTURE_LABELS))
        self.assertEqual(ledger["verified_events"], [])
        self.assertIn("timeout", ledger["remaining_event_gates"])
        self.assertIn("targethandoff", ledger["remaining_event_gates"])
        self.assertIn("platformlanding", ledger["remaining_event_gates"])

    def test_raw_fixture_gates_exclude_navigation_target_handoff(self):
        raw_ledger = replay.coverage_ledger(
            ["startup"],
            verified_events=[],
            required_event_gates=replay.event_gates_for_mode("raw"),
        )
        nav_ledger = replay.coverage_ledger(
            ["startup"],
            verified_events=[],
            required_event_gates=replay.event_gates_for_mode("navigation"),
        )

        self.assertNotIn("targethandoff", raw_ledger["required_event_gates"])
        self.assertIn("targethandoff", nav_ledger["required_event_gates"])
        self.assertNotIn("targethandoff", fixtures.search_gates_for_mode("raw"))
        self.assertIn("targethandoff", fixtures.search_gates_for_mode("navigation"))

        event_ledger = replay.coverage_ledger(["startup"], verified_events=["timeout", "platformriding"])
        self.assertIn("timeout", event_ledger["verified_events"])
        self.assertIn("platformriding", event_ledger["verified_events"])
        self.assertNotIn("timeout", event_ledger["remaining_event_gates"])
        self.assertIn("platformlanding", event_ledger["remaining_event_gates"])

    def test_navigation_fixture_schedule_preserves_steps_without_label_only_event_coverage(self):
        labels = replay.navigation_fixture_labels(steps=80)

        self.assertEqual(len(labels), 80)
        self.assertGreater(labels.count("walk"), 10)
        self.assertLessEqual(labels.count("reset"), 1)
        ledger = replay.coverage_ledger(labels, verified_events=[])
        self.assertIn("platformreversal", ledger["remaining_event_gates"])
        self.assertIn("targethandoff", ledger["remaining_event_gates"])

    def test_verified_events_are_inferred_from_state_not_fixture_names(self):
        previous = {
            "contacts": [{"kind": "geometric_platform_support", "player_agent_id": 0, "platform_id": 1, "contact_id": "old"}],
            "players": [{"agent_id": 0, "grounded": True, "support": {"kind": "platform", "id": 1}}],
            "moving_platforms": [{"id": 1, "velocity": [1.0, 0.0], "moving_to_w2": True}],
            "wrapper": {"chain": [{"targets_reached": 0, "total_targets": 3}]},
        }
        current = {
            "contacts": [{"kind": "geometric_platform_support", "player_agent_id": 0, "platform_id": 1, "contact_id": "new"}],
            "players": [{"agent_id": 0, "grounded": True, "support": {"kind": "platform", "id": 1}}],
            "moving_platforms": [{"id": 1, "velocity": [-1.0, 0.0], "moving_to_w2": False}],
            "wrapper": {"chain": [{"targets_reached": 1, "total_targets": 3}]},
        }

        events = replay.verified_events_from_state(
            (None, 0.0, False, True, {"TimeLimit.truncated": True, "nav_reached": True}),
            current,
            previous_state=previous,
            reset_event=True,
        )

        self.assertIn("timeout", events)
        self.assertIn("reset", events)
        self.assertIn("targethandoff", events)
        self.assertNotIn("platformlanding", events)
        self.assertNotIn("platformriding", events)
        self.assertNotIn("platformreversal", events)

    def test_single_lane_vector_transition_flags_are_normalized_to_scalars(self):
        transition = ("obs", [1.5], [False], [{"nav_reached": True, "TimeLimit.truncated": False}])

        summary = replay.transition_summary(transition)
        events = replay.discrete_events_from_record(transition, _fake_platform_env())
        verified = replay.verified_events_from_state(
            transition,
            {"wrapper": {"chain": []}, "contacts": [], "moving_platforms": [], "players": []},
        )

        self.assertEqual(summary["reward"], 1.5)
        self.assertIs(summary["terminated"], False)
        self.assertIs(summary["truncated"], False)
        self.assertEqual(summary["info"], {"nav_reached": True, "TimeLimit.truncated": False})
        self.assertNotIn("terminated", events)
        self.assertIn("nav_reached", events)
        self.assertIn("targethandoff", verified)

    def test_sb3_done_splits_time_limit_truncation_and_marks_post_reset_state(self):
        truncated_transition = (
            "post-reset-obs",
            [0.0],
            [True],
            [{"TimeLimit.truncated": True, "terminal_observation": "terminal-obs"}],
        )
        terminal_transition = (
            "post-reset-obs",
            [0.0],
            [True],
            [{"TimeLimit.truncated": False, "terminal_observation": "terminal-obs"}],
        )

        truncated = replay.transition_summary(truncated_transition)
        terminal = replay.transition_summary(terminal_transition)

        self.assertIs(truncated["terminated"], False)
        self.assertIs(truncated["truncated"], True)
        self.assertIs(truncated["auto_reset"], True)
        self.assertIs(truncated["post_reset_state"], True)
        self.assertEqual(truncated["terminal_observation"], "terminal-obs")
        self.assertIs(terminal["terminated"], True)
        self.assertIs(terminal["truncated"], False)
        self.assertIs(terminal["auto_reset"], True)

    def test_transition_summary_preserves_list_infos_when_not_single_lane(self):
        transition = ("obs", [0.0, 1.0], [False, True], [{"nav_reached": True}, {"timeout": True}])

        summary = replay.transition_summary(transition)

        self.assertEqual(summary["terminated"], [False, True])
        self.assertEqual(summary["info"], [{"nav_reached": True}, {"timeout": True}])

    def test_transition_summary_excludes_monitor_episode_wallclock_but_keeps_reward_length(self):
        transition = (
            "obs",
            [0.0],
            [True],
            [{"episode": {"r": 2.5, "l": 81, "t": 0.051536}, "TimeLimit.truncated": False}],
        )

        summary = replay.transition_summary(transition)

        self.assertEqual(summary["info"]["episode"], {"r": 2.5, "l": 81})
        self.assertNotIn("t", summary["info"]["episode"])

    def test_platform_event_credit_requires_controlled_player_moving_support_and_persistence(self):
        moving_previous = {
            "contacts": [{"kind": "geometric_platform_support", "player_agent_id": 0, "platform_id": 1, "contact_id": "p0"}],
            "players": [{"agent_id": 0, "grounded": True, "support": {"kind": "platform", "id": 1}}],
            "moving_platforms": [{"id": 1, "velocity": [0.2, 0.0], "moving_to_w2": True}],
            "wrapper": {"chain": []},
        }
        moving_current = copy.deepcopy(moving_previous)
        moving_current["contacts"] = [{"kind": "geometric_platform_support", "player_agent_id": 0, "platform_id": 1, "contact_id": "p1"}]

        opponent_only = copy.deepcopy(moving_current)
        opponent_only["contacts"] = [{"kind": "geometric_platform_support", "player_agent_id": 1, "platform_id": 1, "contact_id": "opp"}]
        opponent_only["players"] = [{"agent_id": 1, "grounded": True, "support": {"kind": "platform", "id": 1}}]
        static_ground = copy.deepcopy(moving_current)
        static_ground["moving_platforms"] = [{"id": 1, "velocity": [0.0, 0.0], "moving_to_w2": None}]
        one_shot = copy.deepcopy(moving_current)

        self.assertEqual(
            replay.verified_events_from_state((None, 0.0, False, False, {}), opponent_only, previous_state=moving_previous)
            & {"platformlanding", "platformriding", "platformreversal"},
            set(),
        )
        self.assertEqual(
            replay.verified_events_from_state((None, 0.0, False, False, {}), static_ground, previous_state=moving_previous)
            & {"platformlanding", "platformriding", "platformreversal"},
            set(),
        )
        one_shot_events = replay.verified_events_from_state((None, 0.0, False, False, {}), one_shot, previous_state=None)
        self.assertIn("platformlanding", one_shot_events)
        self.assertNotIn("platformriding", one_shot_events)
        persistent_events = replay.verified_events_from_state(
            (None, 0.0, False, False, {}),
            moving_current,
            previous_state=moving_previous,
        )
        self.assertIn("platformriding", persistent_events)

    def test_platform_reversal_credit_stays_on_same_support_and_never_crosses_reset(self):
        previous = {
            "contacts": [{"kind": "geometric_platform_support", "player_agent_id": 0, "platform_id": 1, "contact_id": "p0"}],
            "players": [{"agent_id": 0, "grounded": True, "support": {"kind": "platform", "id": 1}}],
            "moving_platforms": [
                {"id": 1, "velocity": [0.2, 0.0], "moving_to_w2": True},
                {"id": 2, "velocity": [0.4, 0.0], "moving_to_w2": True},
            ],
            "wrapper": {"chain": []},
        }
        same_support_reversal = {
            "contacts": [{"kind": "geometric_platform_support", "player_agent_id": 0, "platform_id": 1, "contact_id": "p1"}],
            "players": [{"agent_id": 0, "grounded": True, "support": {"kind": "platform", "id": 1}}],
            "moving_platforms": [
                {"id": 1, "velocity": [-0.2, 0.0], "moving_to_w2": False},
                {"id": 2, "velocity": [0.4, 0.0], "moving_to_w2": True},
            ],
            "wrapper": {"chain": []},
        }
        other_platform_reversal = copy.deepcopy(same_support_reversal)
        other_platform_reversal["moving_platforms"] = [
            {"id": 1, "velocity": [0.2, 0.0], "moving_to_w2": True},
            {"id": 2, "velocity": [-0.4, 0.0], "moving_to_w2": False},
        ]

        events = replay.verified_events_from_state((None, 0.0, False, False, {}), same_support_reversal, previous_state=previous)
        other_events = replay.verified_events_from_state((None, 0.0, False, False, {}), other_platform_reversal, previous_state=previous)
        reset_events = replay.verified_events_from_state(
            (None, 0.0, False, False, {}),
            same_support_reversal,
            previous_state=previous,
            reset_event=True,
        )

        self.assertIn("platformreversal", events)
        self.assertNotIn("platformreversal", other_events)
        self.assertNotIn("platformreversal", reset_events)

    def test_cumulative_coverage_retains_events_across_reset_boundary(self):
        env = _fake_platform_env()
        verified_events = set()
        first = replay.make_replay_record(
            0,
            {0: [0] * 10, 1: [0] * 10},
            (None, 0.0, False, True, {"TimeLimit.truncated": True}),
            env,
            verified_events=verified_events,
            required_event_gates=replay.event_gates_for_mode("raw"),
        )
        verified_events.update(first["state"]["coverage"]["verified_events"])
        second = replay.make_replay_record(
            1,
            {0: [0] * 10, 1: [0] * 10},
            (None, 0.0, False, False, {}),
            env,
            previous_state=first["state"],
            reset_event=True,
            verified_events=verified_events,
            required_event_gates=replay.event_gates_for_mode("raw"),
        )

        self.assertIn("timeout", second["state"]["coverage"]["verified_events"])
        self.assertIn("reset", second["state"]["coverage"]["verified_events"])
        self.assertNotIn("targethandoff", second["state"]["coverage"]["required_event_gates"])

    def test_navigation_metadata_records_phase0b_controls_and_frame_stack(self):
        config = wrapper_config_path()
        metadata = replay.default_metadata(
            replay_source_root(),
            config,
            seed=11,
            action_seed=12,
            action_sha256="action-hash",
            mode="navigation",
            steps=3,
            phase="0b",
            frame_stack=4,
        )

        replay.validate_metadata(metadata)
        self.assertEqual(metadata["capture"]["mode"], "navigation")
        self.assertEqual(metadata["capture"]["frame_stack"], 4)
        self.assertEqual(metadata["config"]["phase"], "0b")
        self.assertEqual(metadata["controls"]["kind"], "navigation_movement_multibinary_10")
        self.assertEqual(metadata["actions"]["source"], "scripted-fixture")
        self.assertEqual(metadata["capture"]["contract_scope"], replay.CONTRACT_SCOPE)
        self.assertIn("combat_attack_substate", metadata["capture"]["excluded_state"])
        self.assertEqual(metadata["capture"]["excluded_runtime_fields"], ["transition.info.episode.t"])

    def test_wrapper_source_env_accepts_variant_directory(self):
        old_source = os.environ.get("AI2_REPLAY_WRAPPER_SOURCE")
        old_config = os.environ.get("AI2_REPLAY_CONFIG")
        try:
            with tempfile.TemporaryDirectory(prefix="ai2-wrapper-dir-") as tmp:
                root = Path(tmp)
                (source_path(root, "train.py")).write_text("# wrapper\n", encoding="utf-8")
                (root / "control.yaml").write_text("curriculum: {}\n", encoding="utf-8")
                os.environ["AI2_REPLAY_WRAPPER_SOURCE"] = str(root)
                os.environ.pop("AI2_REPLAY_CONFIG", None)

                self.assertEqual(wrapper_source_path(), source_path(root.resolve(), "train.py"))
                self.assertEqual(wrapper_config_path(), root.resolve() / "control.yaml")
        finally:
            if old_source is None:
                os.environ.pop("AI2_REPLAY_WRAPPER_SOURCE", None)
            else:
                os.environ["AI2_REPLAY_WRAPPER_SOURCE"] = old_source
            if old_config is None:
                os.environ.pop("AI2_REPLAY_CONFIG", None)
            else:
                os.environ["AI2_REPLAY_CONFIG"] = old_config

    def test_metadata_can_record_scripted_raw_fixture_action_source(self):
        metadata = replay.default_metadata(
            replay_source_root(),
            None,
            seed=11,
            action_seed=11,
            action_sha256="action-hash",
            mode="raw",
            steps=2,
            action_script=["startup", "walk"],
            action_source="scripted-contact-fixture",
        )

        self.assertEqual(metadata["actions"]["source"], "scripted-contact-fixture")
        self.assertEqual(metadata["actions"]["script"], ["startup", "walk"])

    def test_required_event_gate_enforcement_fails_diagnostics_with_remaining_gates(self):
        diagnostic = [_record(0)]

        with self.assertRaisesRegex(ValueError, "remaining event gates"):
            replay.require_event_gates_satisfied(diagnostic)

        complete = copy.deepcopy(diagnostic)
        complete[0]["state"]["coverage"] = replay.coverage_ledger(
            ["startup"],
            verified_events=replay.REQUIRED_EVENT_GATES,
        )
        replay.require_event_gates_satisfied(complete)

    def test_capture_state_records_geometric_support_separate_from_pymunk_contacts(self):
        state = replay.capture_env_state(_fake_platform_env(geometric=True, arbiter=True))
        player = state["players"][0]
        contacts = state["contacts"]

        self.assertTrue(player["grounded"])
        self.assertEqual(player["support"]["source"], "geometry")
        self.assertEqual(player["support"]["id"], 1)
        self.assertIn("geometric_platform_support", {contact["kind"] for contact in contacts})
        self.assertIn("pymunk_arbiter", {contact["kind"] for contact in contacts})
        geometric = [contact for contact in contacts if contact["kind"] == "geometric_platform_support"][0]
        arbiter = [contact for contact in contacts if contact["kind"] == "pymunk_arbiter"][0]
        self.assertEqual(geometric["source"], "is_on_floor_bb")
        self.assertEqual(arbiter["source"], "body.each_arbiter")
        self.assertNotEqual(geometric["contact_id"], arbiter["contact_id"])

    def test_capture_state_records_contact_shape_roles_and_platform_geometry(self):
        state = replay.capture_env_state(_fake_platform_env(geometric=True, arbiter=True))
        platform = state["moving_platforms"][0]
        arbiter = [contact for contact in state["contacts"] if contact["kind"] == "pymunk_arbiter"][0]

        self.assertEqual(
            arbiter["shape_roles"],
            [
                {"order": 0, "role": "player", "agent_id": 0},
                {"order": 1, "role": "platform", "platform_id": 1, "platform_body_id": "platform:1"},
            ],
        )
        self.assertEqual(arbiter["normal_role"], "shape0_to_shape1_raw")
        self.assertEqual(arbiter["points"][0]["point_a"], [0.5, 0.5])
        self.assertEqual(
            platform["geometry"]["world_bb"],
            {"left": -0.5, "right": 0.5, "bottom": -0.5, "top": 0.5},
        )

    def test_verified_platform_events_accept_geometric_support_without_label_coverage(self):
        previous = {
            "contacts": [{"kind": "geometric_platform_support", "player_agent_id": 0, "platform_id": 1, "contact_id": "previous"}],
            "players": [{"agent_id": 0, "grounded": True, "support": {"kind": "platform", "id": 1}}],
            "moving_platforms": [{"id": 1, "velocity": [0.2, 0.0], "moving_to_w2": True}],
            "wrapper": {"chain": []},
        }
        state = replay.capture_env_state(_fake_platform_env(geometric=True, arbiter=False))

        events = replay.verified_events_from_state(
            (None, 0.0, False, False, {}),
            state,
            previous_state=previous,
        )

        self.assertNotIn("platformlanding", events)
        self.assertIn("platformriding", events)

    def test_contact_fixture_script_schedule_and_summary_are_deterministic_and_honest(self):
        labels = fixtures.fixture_labels_for_search(steps=40)
        self.assertEqual(len(labels), 40)
        self.assertEqual(labels[:5], ["startup"] * 5)
        self.assertGreater(labels.count("walk"), 10)

        record = _record(0)
        record["state"]["coverage"] = replay.coverage_ledger(
            ["startup", "walk"],
            verified_events=["platformriding"],
        )
        summary = fixtures.summarize_records([record])
        self.assertEqual(summary["records"], 1)
        self.assertIn("platformriding", summary["found_event_gates"])
        self.assertIn("platformlanding", summary["missing_event_gates"])
        self.assertEqual(summary["dispatched_fixture_labels"], ["startup", "walk"])

    def test_contact_fixture_controller_choice_defaults_to_scripted_and_accepts_adaptive(self):
        with tempfile.TemporaryDirectory(prefix="ai2-fixture-cli-") as tmp:
            scripted = fixtures.parse_args(["--source-root", str(ROOT), "--output-dir", tmp])
            adaptive = fixtures.parse_args([
                "--source-root",
                str(ROOT),
                "--output-dir",
                tmp,
                "--controller",
                "adaptive",
            ])

        self.assertEqual(scripted.controller, "scripted")
        self.assertEqual(adaptive.controller, "adaptive")
        self.assertEqual(fixtures.action_source_for_controller("scripted"), "scripted-contact-fixture")
        self.assertEqual(fixtures.action_source_for_controller("adaptive"), "adaptive-contact-fixture")

    def test_adaptive_raw_controller_steers_toward_platform_then_releases_to_ride(self):
        space = type("Space", (), {"shape": (10,), "dtype": float})()
        controller = fixtures.AdaptiveContactController("raw", space)
        state = replay.capture_env_state(_fake_platform_env(geometric=False))
        state["players"][0]["position"] = [-5.0, 0.0]
        state["moving_platforms"][0]["position"] = [0.0, 1.0]

        label, action = controller.next_action(state)
        self.assertEqual(label, "adaptive_platform_right")
        self.assertEqual(list(action[0]), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(action[1]), [0] * 10)

        state["players"][0]["position"] = [5.0, 0.0]
        label, action = controller.next_action(state)
        self.assertEqual(label, "adaptive_platform_left")
        self.assertEqual(list(action[0]), [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

        state["players"][0]["position"] = [0.0, 0.4]
        state["players"][0]["support"] = {"kind": "platform", "id": 1}
        state["players"][0]["grounded"] = True
        controller.support_ticks = 2
        label, action = controller.next_action(state)
        self.assertEqual(label, "adaptive_platform_ride")
        self.assertEqual(list(action[0]), [0] * 10)

    def test_adaptive_navigation_controller_steers_actual_target_and_settles(self):
        space = type("Space", (), {"shape": (10,), "dtype": float})()
        controller = fixtures.AdaptiveContactController("navigation", space)
        state = replay.capture_env_state(_fake_platform_env(geometric=False))
        state["players"][0]["position"] = [-2.0, 0.0]
        state["wrapper"] = {"chain": [{"target_position": [1.0, 0.0]}]}

        label, action = controller.next_action(state)
        self.assertEqual(label, "adaptive_target_right")
        self.assertEqual(list(action), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        state["players"][0]["position"] = [1.1, 0.0]
        label, action = controller.next_action(state)
        self.assertEqual(label, "adaptive_target_settle")
        self.assertEqual(list(action), [0] * 10)

    def test_adaptive_controller_tracks_support_ticks_and_resets_on_reset_phase(self):
        space = type("Space", (), {"shape": (10,), "dtype": float})()
        controller = fixtures.AdaptiveContactController("raw", space)
        record = _record(0)
        record["state"]["players"][0]["support"] = {"kind": "platform", "id": 1}
        record["state"]["players"][0]["grounded"] = True

        controller.observe_record(record)
        controller.observe_record(record)
        self.assertEqual(controller.support_ticks, 2)
        reset_record = copy.deepcopy(record)
        reset_record["transition"] = {"auto_reset": True, "post_reset_state": True}
        controller.observe_record(reset_record)
        self.assertEqual(controller.support_ticks, 0)

    def test_adaptive_scan_metadata_records_observed_adaptive_labels(self):
        metadata = fixtures.metadata_for_fixture_records(
            replay_source_root(),
            None,
            seed=3,
            mode="raw",
            records=[{"action": {0: [0] * 10, 1: [0] * 10}}],
            reached_labels=["adaptive_platform_right"],
            controller="adaptive",
            frame_stack=None,
        )

        self.assertEqual(metadata["actions"]["source"], "adaptive-contact-fixture")
        self.assertEqual(metadata["actions"]["script"], ["adaptive_platform_right"])


class ReplaySourceSelectionTests(unittest.TestCase):
    def test_source_selection_rejects_a_cached_foreign_game_before_changing_cwd(self):
        game = ModuleType("environment.environment")
        game.__file__ = "/foreign/environment/environment.py"
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"environment.environment": game}):
            environment = Path(directory) / "UTMIST-AI2-main/environment/environment.py"
            environment.parent.mkdir(parents=True)
            environment.touch()
            with patch.object(replay.os, "chdir") as chdir:
                with self.assertRaisesRegex(RuntimeError, "different game source"):
                    replay.prepare_source_imports(Path(directory))
                chdir.assert_not_called()

    def test_missing_frozen_game_is_rejected_before_changing_cwd(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(replay.os, "chdir") as chdir:
            with self.assertRaises(FileNotFoundError):
                replay.prepare_source_imports(Path(directory))
            chdir.assert_not_called()

    def test_frozen_helper_is_loaded_from_selected_root(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules):
            source = Path(directory)
            (source / "benchmark_training.py").write_text("SELECTED = 'frozen'\n")
            helper = replay.load_benchmark_helper(source)
            self.assertEqual(helper.SELECTED, "frozen")
            self.assertEqual(Path(helper.__file__), (source / "benchmark_training.py").resolve())

    def test_incomplete_packaged_source_does_not_use_live_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "ai2").mkdir()
            (source / "ai2/paths.py").touch()
            with self.assertRaises(FileNotFoundError):
                replay.load_benchmark_helper(source)

    def test_raw_builder_requires_the_selected_game_source(self):
        from ai2.evaluation import benchmark_training as bench
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                bench.build_raw_env(7, source_root=Path(directory))

    def test_raw_builder_rejects_an_already_loaded_foreign_game(self):
        from ai2.evaluation import benchmark_training as bench
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "path", list(sys.path)):
            source = Path(directory)
            environment = source / "UTMIST-AI2-main/environment/environment.py"
            environment.parent.mkdir(parents=True)
            environment.touch()
            package = ModuleType("environment")
            package.__path__ = []
            agent = ModuleType("environment.agent")
            agent.CameraResolution = SimpleNamespace(LOW="low")
            game = ModuleType("environment.environment")
            game.__file__ = "/foreign/environment/environment.py"
            game.WarehouseBrawl = lambda **kwargs: self.fail("foreign game must not be constructed")
            with patch.dict(sys.modules, {"environment": package, "environment.agent": agent,
                                          "environment.environment": game}), patch.object(bench, "seed_everything"):
                with self.assertRaisesRegex(RuntimeError, "does not match selected source"):
                    bench.build_raw_env(7, source_root=source)


class ReplayEnvironmentSmokeTest(unittest.TestCase):
    def test_scripted_actions_capture_walk_turn_jump_edge_idle_records(self):
        if importlib.util.find_spec("pygame") is None or importlib.util.find_spec("gymnasium") is None:
            self.skipTest("game dependencies are not installed in this interpreter")

        source_root = replay_source_root()
        replay.prepare_source_imports(source_root)
        import ai2.evaluation.benchmark_training as bench

        bench.set_headless_environment()
        env = bench.build_raw_env(7, source_root=source_root)
        try:
            env.reset(seed=7)
            labels = ["startup", "walk", "turn", "jump", "edge", "idle"]
            actions = replay.scripted_raw_actions(env.action_space, labels)
            records = []
            for tick, action in enumerate(actions):
                transition = env.step(action)
                records.append(replay.make_replay_record(tick, action, transition, env, coverage_labels=labels[:tick + 1]))
        finally:
            env.close()

        self.assertEqual([record["tick"] for record in records], list(range(len(labels))))
        for record in records:
            replay.validate_record(record)
            self.assertEqual(len(record["state"]["players"]), 2)
            self.assertIn("moving_platforms", record["state"])
            self.assertIn("contacts", record["state"])
            self.assertIn("coverage", record["state"])
            self.assertIn("terminated", record["transition"])

    def test_navigation_vec_frame_stack_capture_writes_phase0b_records(self):
        if importlib.util.find_spec("pygame") is None or importlib.util.find_spec("gymnasium") is None:
            self.skipTest("game dependencies are not installed in this interpreter")
        if importlib.util.find_spec("stable_baselines3") is None or importlib.util.find_spec("yaml") is None:
            self.skipTest("navigation wrapper dependencies are not installed in this interpreter")

        source_root = replay_source_root()
        wrapper_source = wrapper_source_path()
        config = wrapper_config_path()
        if not wrapper_source.exists() or not config.exists():
            self.skipTest("frozen navigation wrapper/config are not available")
        with tempfile.TemporaryDirectory(prefix="ai2-nav-replay-") as tmp:
            output = Path(tmp) / "nav.jsonl"
            result = replay.capture_navigation_replay(
                source_root,
                output,
                config_path=config,
                wrapper_source=wrapper_source,
                seed=5,
                steps=3,
                labels=["startup", "walk", "idle"],
            )
            items = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(result["records"], 3)
        self.assertEqual(items[0]["capture"]["mode"], "navigation")
        self.assertEqual(items[0]["capture"]["frame_stack"], 4)
        self.assertEqual(items[0]["config"]["phase"], "0b")
        self.assertEqual(len(items[1]["state"]["players"]), 2)
        wrapper_kinds = [entry["kind"] for entry in items[1]["state"]["wrapper"]["chain"]]
        self.assertIn("FrozenOpponentWrapper", wrapper_kinds)


if __name__ == "__main__":
    unittest.main()
