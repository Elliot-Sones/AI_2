"""Tests for strict FrozenOpponentWrapper contract capture records."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import ai2.evaluation.navigation_contract_capture as capture


class _Vec:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y)[index]


class _Body:
    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0):
        self.position = _Vec(x, y)
        self.velocity = _Vec(vx, vy)


class _Player:
    def __init__(self):
        self.agent_id = 0
        self.body = _Body(1.0, 2.0, 0.25, -0.5)
        self.stocks = 3
        self.on_platform = None
        self._grounded = True

    def is_on_floor(self):
        return self._grounded


class _Underlying:
    def __init__(self, player: _Player):
        self.player = player
        self.step_calls = []
        self.raise_on_step = False

    def step(self, action):
        self.step_calls.append(copy.deepcopy(action))
        if self.raise_on_step:
            raise RuntimeError("boom")
        if self.player is None:
            return "base-obs", 1.25, False, False, {"base": True}
        self.player.body.position = _Vec(0.2, 0.0)
        self.player.body.velocity = _Vec(0.1, 0.1)
        return "base-obs", 1.25, False, False, {"base": True}


class _Wrapper:
    def __init__(self):
        self.player = _Player()
        self.env = _Underlying(self.player)
        self.min_dist = 2.0
        self.max_dist = 6.0
        self.fall_penalty = -3.0
        self.completion_mode = "sequence"
        self.control_mode = "movement"
        self.settle_speed = 1.0
        self.max_targets = 3
        self.time_limit = 600
        self.settle_steps = 9
        self.post_goal_required_steps = 15
        self.target_position = [0.0, 0.0]
        self.last_distance = 1.5
        self.targets_reached = 0
        self.total_targets = 1
        self.episode_steps = 0
        self.targets_this_episode = 0
        self.hold_steps = 8
        self.post_goal_steps = 0
        self.goal_reached_step = None
        self.target_touched = False
        self.fell_this_episode = False
        self.last_stocks = 3
        self.falls = 0
        self.interval_targets_reached = 0
        self.interval_falls = 0
        self.steps_since_log = 0
        self.interval_episodes = 0

    def _get_player(self):
        return self.player

    def _freeze_opponent(self):
        return None

    def _current_observation(self, obs):
        return obs

    def _get_distance_to_target(self, player_pos):
        return math.hypot(player_pos[0] - self.target_position[0], player_pos[1] - self.target_position[1])

    def _is_settled_at_target(self, player, distance):
        support = player.on_platform
        support_velocity = support.velocity if support is not None else (0.0, 0.0)
        relative_speed = math.hypot(
            player.body.velocity[0] - support_velocity[0],
            player.body.velocity[1] - support_velocity[1],
        )
        return distance < 1.0 and player.is_on_floor() and relative_speed <= self.settle_speed

    def _navigation_info(self, *, success=False, horizon=False):
        return {
            "nav_contract": "active_sequence_v3",
            "nav_control_mode": self.control_mode,
            "nav_sequence_complete": bool(success and self.completion_mode == "sequence"),
            "nav_hold_steps": self.hold_steps,
            "nav_post_goal_steps": self.post_goal_steps,
            "nav_terminal_horizon": bool(horizon),
            "nav_success": bool(success),
        }

    def _spawn_target(self, player_pos):
        self.target_position = [4.0, 0.0]
        self.total_targets += 1

    def step(self, action):
        if self.control_mode == "movement":
            action = list(action)
            for index in (5, 7, 8, 9):
                action[index] = 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        underlying_done = terminated or truncated
        self.episode_steps += 1
        player = self._get_player()
        player_pos = (player.body.position.x, player.body.position.y)
        self._freeze_opponent()
        obs = self._current_observation(obs)
        nav_reward = 0.0
        current_dist = self._get_distance_to_target(player_pos)
        if self.last_distance is not None:
            nav_reward += (self.last_distance - current_dist) * 2.0
        current_stocks = player.stocks
        if current_stocks < self.last_stocks:
            nav_reward += self.fall_penalty
            self.fell_this_episode = True
            terminated = True
            info["nav_fall"] = 1
        self.last_stocks = current_stocks
        active = not (underlying_done or self.fell_this_episode)
        if active and current_dist < 1.0 and not self.target_touched:
            self.target_touched = True
            info["nav_touched"] = 1
        if active and self._is_settled_at_target(player, current_dist):
            self.hold_steps = min(self.hold_steps + 1, self.settle_steps)
        else:
            self.hold_steps = 0
        if active and self.hold_steps >= self.settle_steps and self.targets_this_episode < self.max_targets:
            nav_reward += 5.0
            self.targets_reached += 1
            self.interval_targets_reached += 1
            info["nav_reached"] = 1
            self.targets_this_episode += 1
            self._spawn_target(player_pos)
            self._freeze_opponent()
            obs = self._current_observation(obs)
            self.hold_steps = 0
            self.target_touched = False
        self.last_distance = self._get_distance_to_target(player_pos)
        info.update(self._navigation_info())
        return obs, reward + nav_reward, terminated, truncated, info


class _FallWrapper(_Wrapper):
    def __init__(self):
        super().__init__()
        self.hold_steps = 8
        self.episode_steps = 599
        self.player.stocks = 2


class _PlayerAbsentWrapper(_Wrapper):
    def __init__(self):
        super().__init__()
        self.player = None
        self.env.player = None

    def _get_player(self):
        return None

    def step(self, action):
        if self.control_mode == "movement":
            action = list(action)
            for index in (5, 7, 8, 9):
                action[index] = 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        if self._get_player() is None:
            return obs, reward, terminated, truncated, info
        raise AssertionError("unreachable")


class _UnreadablePlatform:
    @property
    def velocity(self):
        raise RuntimeError("velocity not readable")


def _native_config():
    import ai2.simulation.native_navigation_contract as native

    return native.NavigationConfig(
        completion_mode="sequence",
        control_mode="movement",
        max_targets=3,
        time_limit_steps=12,
        settle_steps=2,
        post_goal_required_steps=3,
        settle_speed=1.0,
        fall_penalty=-5.0,
    )


def _native_state(**overrides):
    import ai2.simulation.native_navigation_contract as native

    state = native.NavigationState(
        target_x=0.0,
        target_y=0.0,
        target_valid=True,
        last_distance=2.0,
        last_distance_valid=True,
        episode_steps=0,
        targets_this_episode=0,
        hold_steps=0,
        post_goal_steps=0,
        goal_reached_step=0,
        goal_reached_step_valid=False,
        last_stocks=3,
        fell_this_episode=False,
        target_touched=False,
        targets_reached=0,
        total_targets=1,
        falls=0,
        steps_since_log=0,
        interval_targets_reached=0,
        interval_falls=0,
        interval_episodes=0,
    )
    return native.replace_state(state, **overrides)


def _native_post(**overrides):
    import ai2.simulation.native_navigation_contract as native

    post = native.NavigationPostPhysics(
        player_present=True,
        player_x=0.4,
        player_y=0.3,
        velocity_x=0.0,
        velocity_y=0.0,
        support_velocity_x=0.0,
        support_velocity_y=0.0,
        support_velocity_valid=True,
        grounded=True,
        stocks=3,
        raw_reward=1.0,
        raw_terminated=False,
        raw_truncated=False,
        next_target_x=5.0,
        next_target_y=0.0,
        next_target_valid=True,
    )
    return native.replace_post(post, **overrides)


def _capture_from_frozen_oracle(state, post, *, next_target=None, tick=0, case_id="oracle"):
    from tests.test_native_navigation_contract import make_oracle

    wrapper, _env = make_oracle(state, _native_config(), post, next_target=next_target)
    return capture.capture_step(wrapper, [1.0] * 10, tick=tick, case_id=case_id)


class NavigationContractCaptureTests(unittest.TestCase):
    def test_mask_navigation_action_is_immutable_and_masks_only_combat_bits(self):
        original = [1.0] * 10

        masked = capture.mask_navigation_action(original, control_mode="movement")

        self.assertEqual(original, [1.0] * 10)
        self.assertEqual(masked, [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def test_capture_step_records_pre_base_mask_handoff_and_restores_methods(self):
        wrapper = _Wrapper()
        action = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        record = capture.capture_step(wrapper, action, tick=7, case_id="synthetic-handoff")

        self.assertNotIn("step", wrapper.env.__dict__)
        self.assertNotIn("_spawn_target", wrapper.__dict__)
        self.assertEqual(action, [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(record["input"]["original_action"], action)
        self.assertEqual(record["input"]["executed_action"], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(record["input"]["pre_wrapper"]["hold_steps"], 8)
        self.assertEqual(record["input"]["post_physics"]["base_reward"], 1.25)
        self.assertEqual(record["input"]["post_physics"]["support_velocity"], [0.0, 0.0])
        self.assertEqual(record["input"]["target_handoff"]["target_after"], [4.0, 0.0])
        self.assertEqual(record["expected"]["nav_flags"]["nav_reached"], 1)
        self.assertEqual(record["expected"]["after_wrapper"]["target_position"], [4.0, 0.0])
        self.assertEqual(record["observed_events"], ["nav_reached", "nav_touched"])
        self.assertNotIn("nav_success", record["observed_events"])
        self.assertNotIn("nav_contract", record["observed_events"])
        capture.validate_record(record)

    def test_capture_uses_player_on_platform_velocity_not_geometric_support(self):
        wrapper = _Wrapper()
        platform_body = _Body(0.0, 0.0, 3.5, -1.25)
        wrapper.player.on_platform = platform_body

        record = capture.capture_step(wrapper, [0.0] * 10, tick=1, case_id="synthetic-support")

        self.assertEqual(record["input"]["post_physics"]["support_source"], "player.on_platform")
        self.assertEqual(record["input"]["post_physics"]["support_velocity"], [3.5, -1.25])

    def test_capture_restores_intercepts_when_step_raises(self):
        wrapper = _Wrapper()
        wrapper.env.raise_on_step = True

        with self.assertRaises(RuntimeError):
            capture.capture_step(wrapper, [0.0] * 10, tick=3, case_id="synthetic-error")

        self.assertNotIn("step", wrapper.env.__dict__)
        self.assertNotIn("_spawn_target", wrapper.__dict__)

    def test_capture_does_not_fabricate_missing_player_physics_fields(self):
        wrapper = _Wrapper()
        wrapper.env.step = lambda action: ("base-obs", 0.0, False, False, {})
        wrapper.player.body = None

        with self.assertRaisesRegex(capture.NavigationContractError, "missing_field"):
            capture.capture_step(wrapper, [0.0] * 10, tick=0, case_id="missing-body")

    def test_capture_allows_explicit_player_absent_early_return(self):
        wrapper = _Wrapper()
        wrapper.player = None

        def no_player():
            return None

        def early_return(action):
            obs, reward, terminated, truncated, info = wrapper.env.step(action)
            if wrapper._get_player() is None:
                return obs, reward, terminated, truncated, info
            raise AssertionError("unreachable")

        wrapper._get_player = no_player
        wrapper.step = early_return

        record = capture.capture_step(wrapper, [0.0] * 10, tick=0, case_id="no-player")

        self.assertFalse(record["input"]["post_physics"]["player_present"])
        self.assertNotIn("player_position", record["input"]["post_physics"])

    def test_validation_fails_closed_for_empty_missing_and_nonfinite(self):
        with self.assertRaisesRegex(capture.NavigationContractError, "zero_records"):
            capture.validate_records([])

        record = capture.capture_step(_Wrapper(), [0.0] * 10, tick=0, case_id="synthetic")
        missing = copy.deepcopy(record)
        del missing["expected"]["final_reward"]
        with self.assertRaisesRegex(capture.NavigationContractError, "missing_field"):
            capture.validate_record(missing)

        nonfinite = copy.deepcopy(record)
        nonfinite["input"]["post_physics"]["base_reward"] = float("nan")
        with self.assertRaisesRegex(capture.NavigationContractError, "nonfinite_value"):
            capture.validate_record(nonfinite)

        missing_handoff_target = copy.deepcopy(record)
        missing_handoff_target["expected"]["nav_flags"]["nav_reached"] = 1
        missing_handoff_target["input"]["target_handoff"] = {"occurred": True, "events": []}
        with self.assertRaisesRegex(capture.NavigationContractError, "missing_field"):
            capture.validate_record(missing_handoff_target)

        short_action = copy.deepcopy(record)
        short_action["input"]["executed_action"] = [0.0] * 9
        with self.assertRaisesRegex(capture.NavigationContractError, "invalid_action"):
            capture.validate_record(short_action)

        mutable_action = copy.deepcopy(record)
        mutable_action["input"]["action_immutable"] = False
        with self.assertRaisesRegex(capture.NavigationContractError, "action_immutable"):
            capture.validate_record(mutable_action)

        non_numeric_action = copy.deepcopy(record)
        non_numeric_action["input"]["executed_action"][0] = "0.0"
        with self.assertRaisesRegex(capture.NavigationContractError, "invalid_action"):
            capture.validate_record(non_numeric_action)

        non_numeric_vector = copy.deepcopy(record)
        non_numeric_vector["input"]["post_physics"]["player_velocity"][0] = True
        with self.assertRaisesRegex(capture.NavigationContractError, "missing_field"):
            capture.validate_record(non_numeric_vector)

    def test_capture_requires_wrapper_contract_fields(self):
        wrapper = _Wrapper()
        del wrapper.last_distance

        with self.assertRaisesRegex(capture.NavigationContractError, "missing_field"):
            capture.capture_step(wrapper, [0.0] * 10, tick=0, case_id="missing-wrapper-state")

    def test_capture_rejects_debug_logs_scope(self):
        wrapper = _Wrapper()
        wrapper.debug_logs = True

        with self.assertRaisesRegex(capture.NavigationContractError, "unsupported_scope"):
            capture.capture_step(wrapper, [0.0] * 10, tick=0, case_id="debug-logs")

    def test_compare_reports_first_divergent_tick_and_field(self):
        record0 = capture.capture_step(_Wrapper(), [0.0] * 10, tick=0, case_id="case0")
        record1 = capture.capture_step(_Wrapper(), [0.0] * 10, tick=1, case_id="case1")
        native = [
            copy.deepcopy(record0["expected"]),
            copy.deepcopy(record1["expected"]),
        ]
        native[1]["after_wrapper"]["hold_steps"] += 1

        receipt = capture.compare_native_results([record0, record1], native)

        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["first_difference"]["tick"], 1)
        self.assertEqual(receipt["first_difference"]["field"], "expected.after_wrapper.hold_steps")

    def test_compare_native_results_checks_executed_action_and_immutability_when_present(self):
        record = capture.capture_step(_Wrapper(), [1.0] * 10, tick=4, case_id="case-action")
        native = [{
            "input": {
                "executed_action": list(record["input"]["executed_action"]),
                "action_immutable": record["input"]["action_immutable"],
            },
            "expected": copy.deepcopy(record["expected"]),
        }]
        native[0]["input"]["executed_action"][5] = 1.0

        receipt = capture.compare_native_results([record], native)

        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["first_difference"]["tick"], 4)
        self.assertEqual(receipt["first_difference"]["field"], "input.executed_action[5]")

    def test_native_case_export_maps_required_scalars_explicitly(self):
        record = capture.capture_step(_Wrapper(), [1.0] * 10, tick=4, case_id="case-export")

        config, state, post_physics = capture.native_case_from_record(record)

        import ai2.simulation.native_navigation_contract as native

        expected = capture.native_result_for_record(record, native.step_navigation_scalar(config, state, post_physics))
        self.assertIsInstance(config, native.NavigationConfig)
        self.assertIsInstance(state, native.NavigationState)
        self.assertIsInstance(post_physics, native.NavigationPostPhysics)
        self.assertEqual(config.completion_mode, "sequence")
        self.assertEqual(config.time_limit_steps, 600)
        self.assertEqual(state.target_x, 0.0)
        self.assertTrue(state.target_valid)
        self.assertTrue(post_physics.next_target_valid)
        self.assertEqual((post_physics.next_target_x, post_physics.next_target_y), (4.0, 0.0))
        self.assertEqual(state.steps_since_log, 0)
        self.assertEqual(state.interval_episodes, 0)
        self.assertEqual((post_physics.support_velocity_x, post_physics.support_velocity_y), (0.0, 0.0))
        self.assertIn("after_wrapper", expected)

    def test_native_case_export_fails_if_required_wrapper_scalar_missing(self):
        record = _capture_from_frozen_oracle(
            _native_state(hold_steps=1),
            _native_post(next_target_x=5.0, next_target_y=0.0),
            next_target=(5.0, 0.0),
            case_id="case-export",
        )
        del record["input"]["pre_wrapper"]["steps_since_log"]

        with self.assertRaisesRegex(capture.NavigationContractError, "missing_field"):
            capture.native_case_from_record(record)

    def test_compare_files_rejects_source_mismatch_and_readwrite_roundtrip(self):
        record = capture.capture_step(_Wrapper(), [0.0] * 10, tick=0, case_id="case")
        metadata = capture.default_metadata(
            source_train_path=Path("/tmp/frozen/train.py"),
            source_train_sha256="a" * 64,
            config_path=Path("/tmp/frozen/control.yaml"),
            config_sha256="c" * 64,
            case_source="synthetic-method-oracle",
        )
        other_metadata = copy.deepcopy(metadata)
        other_metadata["source"]["train_sha256"] = "b" * 64

        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            capture.write_jsonl(left, metadata, [record])
            capture.write_jsonl(right, other_metadata, [record])
            loaded_meta, loaded_records = capture.read_jsonl(left)

            self.assertEqual(loaded_meta["schema"], capture.CAPTURE_SCHEMA)
            self.assertEqual(len(loaded_records), 1)
            receipt = capture.compare_capture_files(left, right)

        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["reason"], "source_mismatch")

    def test_compare_files_rejects_config_metadata_mismatch(self):
        record = capture.capture_step(_Wrapper(), [0.0] * 10, tick=0, case_id="case")
        metadata = capture.default_metadata(
            source_train_path=Path("/tmp/frozen/train.py"),
            source_train_sha256="a" * 64,
            config_path=Path("/tmp/frozen/control.yaml"),
            config_sha256="c" * 64,
            case_source="synthetic-method-oracle",
        )
        other_metadata = copy.deepcopy(metadata)
        other_metadata["config"]["sha256"] = "d" * 64

        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            capture.write_jsonl(left, metadata, [record])
            capture.write_jsonl(right, other_metadata, [record])
            receipt = capture.compare_capture_files(left, right)

        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["reason"], "metadata_mismatch")
        self.assertEqual(receipt["first_difference"]["field"], "config.sha256")

    def test_metadata_requires_real_nonempty_hex_source_and_config_hashes(self):
        with self.assertRaisesRegex(capture.NavigationContractError, "source_mismatch"):
            capture.default_metadata(
                source_train_path=Path(""),
                source_train_sha256="not-a-sha",
                config_path=Path("/tmp/frozen/control.yaml"),
                config_sha256="c" * 64,
                case_source="synthetic-method-oracle",
            )
        with self.assertRaisesRegex(capture.NavigationContractError, "config_mismatch"):
            capture.default_metadata(
                source_train_path=Path("/tmp/frozen/train.py"),
                source_train_sha256="a" * 64,
                config_path=Path(""),
                config_sha256=None,
                case_source="synthetic-method-oracle",
            )

    def test_on_platform_with_unreadable_velocity_fails_closed(self):
        wrapper = _Wrapper()
        wrapper.player.on_platform = object()

        with self.assertRaisesRegex(capture.NavigationContractError, "missing_field"):
            capture.capture_step(wrapper, [0.0] * 10, tick=0, case_id="bad-support")

        wrapper = _Wrapper()
        wrapper.player.on_platform = _UnreadablePlatform()
        with self.assertRaisesRegex(capture.NavigationContractError, "missing_field"):
            capture.capture_step(wrapper, [0.0] * 10, tick=0, case_id="unreadable-support")

    def test_native_result_for_record_rejects_expected_dict_passthrough(self):
        record = capture.capture_step(_Wrapper(), [0.0] * 10, tick=0, case_id="case-export")

        with self.assertRaisesRegex(capture.NavigationContractError, "schema_mismatch"):
            capture.native_result_for_record(record, capture.expected_native_result(record)["expected"])

    def test_native_result_for_record_maps_serialized_native_result(self):
        import ai2.simulation.native_navigation_contract as native

        record = _capture_from_frozen_oracle(
            _native_state(hold_steps=1),
            _native_post(next_target_x=5.0, next_target_y=0.0),
            next_target=(5.0, 0.0),
            case_id="case-export",
        )
        config, state, post = capture.native_case_from_record(record)
        result = native.step_navigation_scalar(config, state, post)
        serialized = dict(result.__dict__)
        serialized["state"] = dict(result.state.__dict__)

        expected = capture.native_result_for_record(record, serialized)
        receipt = capture.compare_native_results([record], [expected])

        self.assertTrue(receipt["passed"], receipt)

    def test_compiled_c_handoff_result_maps_to_capture_expected(self):
        import ai2.simulation.native_navigation_contract as native

        record = _capture_from_frozen_oracle(
            _native_state(hold_steps=1),
            _native_post(next_target_x=5.0, next_target_y=0.0),
            next_target=(5.0, 0.0),
            case_id="handoff",
        )
        config, state, post = capture.native_case_from_record(record)
        with tempfile.TemporaryDirectory() as tmp:
            result = native.load_cpu_reference(Path(tmp)).step(config, state, post)

        expected = capture.native_result_for_record(record, result)
        receipt = capture.compare_native_results([record], [expected])

        self.assertTrue(receipt["passed"], receipt)
        self.assertEqual(expected["after_wrapper"]["target_position"], [5.0, 0.0])
        self.assertEqual(expected["nav_flags"]["nav_reached"], 1)

    def test_compiled_c_final_target_acquisition_allows_no_handoff_target(self):
        import ai2.simulation.native_navigation_contract as native

        record = _capture_from_frozen_oracle(
            _native_state(targets_this_episode=2, hold_steps=1),
            _native_post(next_target_valid=False),
            case_id="final-target",
        )

        self.assertEqual(record["expected"]["nav_flags"]["nav_reached"], 1)
        self.assertFalse(record["input"]["target_handoff"]["occurred"])
        config, state, post = capture.native_case_from_record(record)
        self.assertFalse(post.next_target_valid)
        with tempfile.TemporaryDirectory() as tmp:
            result = native.load_cpu_reference(Path(tmp)).step(config, state, post)

        expected = capture.native_result_for_record(record, result)
        receipt = capture.compare_native_results([record], [expected])

        self.assertTrue(receipt["passed"], receipt)
        self.assertEqual(expected["after_wrapper"]["targets_this_episode"], 3)
        self.assertEqual(expected["after_wrapper"]["goal_reached_step"], 1)

    def test_validation_rejects_nonfinal_acquisition_without_handoff_target(self):
        record = _capture_from_frozen_oracle(
            _native_state(hold_steps=1),
            _native_post(next_target_x=5.0, next_target_y=0.0),
            next_target=(5.0, 0.0),
            case_id="nonfinal-handoff",
        )
        record["input"]["target_handoff"] = {"occurred": False, "events": []}

        with self.assertRaisesRegex(capture.NavigationContractError, "missing_field"):
            capture.validate_record(record)

    def test_compiled_c_post_goal_hold_success_after_final_target(self):
        import ai2.simulation.native_navigation_contract as native

        record = _capture_from_frozen_oracle(
            _native_state(
                targets_this_episode=3,
                hold_steps=4,
                post_goal_steps=2,
                goal_reached_step=1,
                goal_reached_step_valid=True,
            ),
            _native_post(next_target_valid=False),
            case_id="post-goal-success",
        )
        config, state, post = capture.native_case_from_record(record)
        self.assertFalse(post.next_target_valid)
        with tempfile.TemporaryDirectory() as tmp:
            result = native.load_cpu_reference(Path(tmp)).step(config, state, post)

        expected = capture.native_result_for_record(record, result)
        receipt = capture.compare_native_results([record], [expected])

        self.assertTrue(receipt["passed"], receipt)
        self.assertTrue(expected["terminated"])
        self.assertFalse(expected["truncated"])
        self.assertTrue(expected["nav_flags"]["nav_success"])

    def test_compiled_c_fall_horizon_result_maps_to_capture_expected(self):
        import ai2.simulation.native_navigation_contract as native

        record = _capture_from_frozen_oracle(
            _native_state(episode_steps=11, hold_steps=1),
            _native_post(stocks=2),
            case_id="fall-horizon",
        )
        config, state, post = capture.native_case_from_record(record)
        with tempfile.TemporaryDirectory() as tmp:
            result = native.load_cpu_reference(Path(tmp)).step(config, state, post)

        expected = capture.native_result_for_record(record, result)
        receipt = capture.compare_native_results([record], [expected])

        self.assertTrue(receipt["passed"], receipt)
        self.assertEqual(expected["nav_flags"]["nav_fall"], 1)
        self.assertTrue(expected["terminated"])
        self.assertTrue(expected["truncated"])

    def test_compiled_c_player_absent_result_maps_to_capture_expected(self):
        import ai2.simulation.native_navigation_contract as native

        record = _capture_from_frozen_oracle(
            _native_state(episode_steps=4, steps_since_log=9),
            _native_post(player_present=False, raw_reward=2.5, raw_truncated=True),
            case_id="no-player",
        )
        config, state, post = capture.native_case_from_record(record)
        with tempfile.TemporaryDirectory() as tmp:
            result = native.load_cpu_reference(Path(tmp)).step(config, state, post)

        expected = capture.native_result_for_record(record, result)
        receipt = capture.compare_native_results([record], [expected])

        self.assertTrue(receipt["passed"], receipt)
        self.assertEqual(expected["final_reward"], 2.5)
        self.assertEqual(expected["nav_flags"], {})

    def test_compare_capture_files_rejects_same_outputs_with_different_inputs(self):
        left_record = capture.capture_step(_Wrapper(), [0.0] * 10, tick=0, case_id="case")
        right_record = copy.deepcopy(left_record)
        right_record["input"]["executed_action"][3] = 1.0
        metadata = capture.default_metadata(
            source_train_path=Path("/tmp/frozen/train.py"),
            source_train_sha256="a" * 64,
            config_path=Path("/tmp/frozen/control.yaml"),
            config_sha256="c" * 64,
            case_source="synthetic-method-oracle",
        )

        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            capture.write_jsonl(left, metadata, [left_record])
            capture.write_jsonl(right, metadata, [right_record])
            receipt = capture.compare_capture_files(left, right)

        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["reason"], "value_mismatch")
        self.assertEqual(receipt["first_difference"]["field"], "input.executed_action[3]")


if __name__ == "__main__":
    unittest.main()
