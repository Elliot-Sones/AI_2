import ast
import copy
import math
import os
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np


FROZEN_TRAIN = Path(
    os.environ.get(
        "AI2_FROZEN_TRAIN_SOURCE",
        "/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/"
        "baselines/navigation-repair-20260905/payload/variants/corrected/train.py",
    )
)
TOL = 1e-9


def load_frozen_wrapper_class():
    source = FROZEN_TRAIN.read_text()
    tree = ast.parse(source, filename=str(FROZEN_TRAIN))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FrozenOpponentWrapper")
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)

    class Wrapper:
        def __init__(self, env):
            self.env = env

    namespace = {
        "gym": types.SimpleNamespace(Wrapper=Wrapper),
        "np": np,
        "os": __import__("os"),
        "json": __import__("json"),
    }
    exec(compile(module, filename=str(FROZEN_TRAIN), mode="exec"), namespace)
    return namespace["FrozenOpponentWrapper"]


class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, index):
        return (self.x, self.y)[index]


class Body:
    def __init__(self, x, y, vx=0.0, vy=0.0):
        self.position = Vec2(x, y)
        self.velocity = (vx, vy)


class Platform:
    def __init__(self, vx, vy):
        self.velocity = (vx, vy)


class Player:
    def __init__(self, *, x, y, vx=0.0, vy=0.0, stocks=3, grounded=True, support_velocity=None):
        self.body = Body(x, y, vx, vy)
        self.stocks = stocks
        self._grounded = grounded
        self.on_platform = None if support_velocity is None else Platform(*support_velocity)

    def is_on_floor(self):
        return self._grounded


class DummyRaw:
    def __init__(self, player):
        self.players = [player, Player(x=99.0, y=99.0)]

    def observe(self, index):
        return np.array([index], dtype=np.float32)


class DummyEnv:
    def __init__(self, player, *, reward=1.0, terminated=False, truncated=False):
        self.raw_env = DummyRaw(player)
        self.unwrapped = self
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.last_action = None

    def step(self, action):
        self.last_action = np.array(action, copy=True)
        return np.array([0.0], dtype=np.float32), self.reward, self.terminated, self.truncated, {}


def make_oracle(state, config, post, *, next_target=None):
    wrapper_cls = load_frozen_wrapper_class()
    player = None
    if post.player_present:
        player = Player(
            x=post.player_x,
            y=post.player_y,
            vx=post.velocity_x,
            vy=post.velocity_y,
            stocks=post.stocks,
            grounded=post.grounded,
            support_velocity=(post.support_velocity_x, post.support_velocity_y)
            if post.support_velocity_valid
            else None,
        )
    env = DummyEnv(
        player,
        reward=post.raw_reward,
        terminated=post.raw_terminated,
        truncated=post.raw_truncated,
    )
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper.env = env
    wrapper.debug_logs = False
    wrapper.min_dist = 1.0
    wrapper.max_dist = 12.0
    wrapper.fall_penalty = config.fall_penalty
    wrapper.completion_mode = config.completion_mode
    wrapper.control_mode = config.control_mode
    wrapper.time_limit = config.time_limit_steps
    wrapper.settle_steps = config.settle_steps
    wrapper.post_goal_required_steps = config.post_goal_required_steps
    wrapper.settle_speed = config.settle_speed
    wrapper.max_targets = config.max_targets
    wrapper.log_interval = 10000
    wrapper.target_position = [state.target_x, state.target_y] if state.target_valid else None
    wrapper.last_distance = state.last_distance if state.last_distance_valid else None
    wrapper.targets_reached = state.targets_reached
    wrapper.total_targets = state.total_targets
    wrapper.falls = state.falls
    wrapper.episode_steps = state.episode_steps
    wrapper.targets_this_episode = state.targets_this_episode
    wrapper.hold_steps = state.hold_steps
    wrapper.post_goal_steps = state.post_goal_steps
    wrapper.goal_reached_step = state.goal_reached_step if state.goal_reached_step_valid else None
    wrapper.target_touched = state.target_touched
    wrapper.steps_since_log = state.steps_since_log
    wrapper.interval_targets_reached = state.interval_targets_reached
    wrapper.interval_falls = state.interval_falls
    wrapper.interval_episodes = state.interval_episodes
    wrapper.last_stocks = state.last_stocks
    wrapper.fell_this_episode = state.fell_this_episode

    def deterministic_spawn(player_pos):
        if next_target is None:
            raise AssertionError("oracle asked for a next target")
        wrapper.target_position = [next_target[0], next_target[1]]
        wrapper.total_targets += 1

    wrapper._spawn_target = deterministic_spawn
    return wrapper, env


def compare_result_to_oracle(result, wrapper, reward, terminated, truncated, info):
    scalar_fields = [
        "target_x",
        "target_y",
        "last_distance",
        "goal_reached_step",
    ]
    int_fields = [
        "episode_steps",
        "targets_this_episode",
        "hold_steps",
        "post_goal_steps",
        "targets_reached",
        "total_targets",
        "falls",
        "steps_since_log",
        "interval_targets_reached",
        "interval_falls",
        "interval_episodes",
        "last_stocks",
    ]
    bool_fields = [
        "target_valid",
        "last_distance_valid",
        "goal_reached_step_valid",
        "target_touched",
        "fell_this_episode",
    ]
    state = result.state
    expected_target = wrapper.target_position
    self_target_valid = expected_target is not None
    expected = {
        "target_valid": self_target_valid,
        "target_x": float(expected_target[0]) if self_target_valid else 0.0,
        "target_y": float(expected_target[1]) if self_target_valid else 0.0,
        "last_distance_valid": wrapper.last_distance is not None,
        "last_distance": float(wrapper.last_distance or 0.0),
        "goal_reached_step_valid": wrapper.goal_reached_step is not None,
        "goal_reached_step": int(wrapper.goal_reached_step or 0),
    }
    for name in int_fields:
        expected[name] = getattr(wrapper, name)
    expected["target_touched"] = wrapper.target_touched
    expected["fell_this_episode"] = wrapper.fell_this_episode

    for name in scalar_fields:
        self = getattr(state, name)
        self_expected = expected[name]
        assert math.isclose(self, self_expected, abs_tol=TOL, rel_tol=0.0), (name, self, self_expected)
    for name in int_fields + bool_fields:
        assert getattr(state, name) == expected[name], (name, getattr(state, name), expected[name])
    assert math.isclose(result.reward, reward, abs_tol=TOL, rel_tol=0.0), (result.reward, reward)
    assert math.isclose(result.nav_reward, reward - 1.0, abs_tol=TOL, rel_tol=0.0), result.nav_reward
    assert result.terminated == terminated
    assert result.truncated == truncated
    assert result.fall == bool(info.get("nav_fall", 0))
    assert result.touched == bool(info.get("nav_touched", 0))
    assert result.reached == bool(info.get("nav_reached", 0))
    assert result.success == bool(info["nav_success"])
    assert result.horizon == bool(info["nav_terminal_horizon"])


class NativeNavigationContractTests(unittest.TestCase):
    def setUp(self):
        import ai2.simulation.native_navigation_contract as nav

        self.nav = nav
        self.config = nav.NavigationConfig(
            completion_mode="sequence",
            control_mode="movement",
            max_targets=3,
            time_limit_steps=12,
            settle_steps=2,
            post_goal_required_steps=3,
            settle_speed=1.0,
            fall_penalty=-5.0,
        )

    def state(self, **overrides):
        state = self.nav.NavigationState(
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
        return self.nav.replace_state(state, **overrides)

    def post(self, **overrides):
        post = self.nav.NavigationPostPhysics(
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
        return self.nav.replace_post(post, **overrides)

    def assert_matches_oracle(self, state, post, *, next_target=None):
        original_state = copy.deepcopy(state)
        wrapper, _env = make_oracle(state, self.config, post, next_target=next_target)
        _obs, reward, terminated, truncated, info = wrapper.step(np.ones(10, dtype=np.float64))
        result = self.nav.step_navigation_scalar(self.config, state, post)
        compare_result_to_oracle(result, wrapper, reward, terminated, truncated, info)
        self.assertEqual(state, original_state, "input state must not be mutated")
        return result

    def test_action_mask_zeros_movement_forbidden_indices_without_mutating_input(self):
        action = np.arange(10, dtype=np.float64)
        masked = self.nav.mask_movement_action(action)
        np.testing.assert_array_equal(action, np.arange(10, dtype=np.float64))
        self.assertEqual(masked[5], 0.0)
        self.assertEqual(masked[7], 0.0)
        self.assertEqual(masked[8], 0.0)
        self.assertEqual(masked[9], 0.0)
        self.assertEqual(masked[6], 6.0)

    def test_action_mask_rejects_non_exact_or_nonfinite_actions(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.nav.mask_movement_action(np.zeros(11, dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.nav.mask_movement_action(np.zeros((1, 10), dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.nav.mask_movement_action([0.0, 1.0, 2.0, 3.0, 4.0, float("nan"), 6.0, 7.0, 8.0, 9.0])

    def test_action_mask_matches_frozen_movement_wrapper_step_input(self):
        action = np.arange(10, dtype=np.float64)
        wrapper, env = make_oracle(self.state(), self.config, self.post(player_present=False))

        wrapper.step(action)

        np.testing.assert_array_equal(action, np.arange(10, dtype=np.float64))
        np.testing.assert_array_equal(env.last_action, self.nav.mask_movement_action(action))

    def test_settle_handoff_matches_frozen_wrapper_and_uses_new_target_distance(self):
        state = self.state(hold_steps=1)
        post = self.post(next_target_x=5.0, next_target_y=0.0)

        result = self.assert_matches_oracle(state, post, next_target=(5.0, 0.0))

        self.assertTrue(result.reached)
        self.assertTrue(result.handoff)
        self.assertFalse(result.state.target_touched)
        self.assertEqual(result.state.hold_steps, 0)
        self.assertAlmostEqual(result.state.last_distance, math.hypot(0.4 - 5.0, 0.3 - 0.0), delta=TOL)

    def test_death_frame_prevents_target_bonus_even_at_horizon(self):
        state = self.state(episode_steps=11, hold_steps=1)
        post = self.post(stocks=2)

        result = self.assert_matches_oracle(state, post)

        self.assertTrue(result.fall)
        self.assertFalse(result.reached)
        self.assertFalse(result.success)
        self.assertTrue(result.terminated)
        self.assertTrue(result.truncated)

    def test_sequence_final_goal_needs_continuous_post_goal_hold(self):
        state = self.state(
            targets_this_episode=3,
            hold_steps=4,
            post_goal_steps=2,
            goal_reached_step=1,
            goal_reached_step_valid=True,
        )
        post = self.post()

        result = self.assert_matches_oracle(state, post)

        self.assertTrue(result.success)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.state.post_goal_steps, 3)

    def test_failed_settle_resets_hold_and_post_goal_without_success(self):
        state = self.state(
            targets_this_episode=3,
            hold_steps=4,
            post_goal_steps=2,
            goal_reached_step=1,
            goal_reached_step_valid=True,
        )
        post = self.post(grounded=False)

        result = self.assert_matches_oracle(state, post)

        self.assertEqual(result.state.hold_steps, 0)
        self.assertEqual(result.state.post_goal_steps, 0)
        self.assertFalse(result.success)

    def test_missing_required_next_target_fails_without_mutating_input_state(self):
        state = self.state(hold_steps=1)
        post = self.post(next_target_valid=False)

        with self.assertRaisesRegex(ValueError, "next target"):
            self.nav.step_navigation_scalar(self.config, state, post)
        self.assertEqual(state.hold_steps, 1)
        self.assertEqual(state.targets_this_episode, 0)

    def test_underlying_done_blocks_active_progress_and_horizon(self):
        state = self.state(episode_steps=11, hold_steps=1)
        post = self.post(raw_truncated=True)

        result = self.assert_matches_oracle(state, post)

        self.assertFalse(result.reached)
        self.assertFalse(result.horizon)
        self.assertFalse(result.success)
        self.assertTrue(result.truncated)

    def test_horizon_completion_matches_frozen_wrapper(self):
        config = self.nav.NavigationConfig(
            completion_mode="horizon",
            control_mode="full",
            max_targets=1,
            time_limit_steps=12,
            settle_steps=2,
            post_goal_required_steps=3,
            settle_speed=1.0,
            fall_penalty=-5.0,
        )
        old_config = self.config
        self.config = config
        try:
            state = self.state(
                targets_this_episode=1,
                hold_steps=2,
                goal_reached_step=8,
                goal_reached_step_valid=True,
                episode_steps=11,
            )
            post = self.post()

            result = self.assert_matches_oracle(state, post)

            self.assertTrue(result.horizon)
            self.assertTrue(result.success)
            self.assertFalse(result.terminated)
            self.assertTrue(result.truncated)
        finally:
            self.config = old_config

    def test_finite_validation_rejects_nan_before_mutating(self):
        state = self.state()
        post = self.post(player_x=float("nan"))

        with self.assertRaisesRegex(ValueError, "finite"):
            self.nav.step_navigation_scalar(self.config, state, post)
        self.assertEqual(state.episode_steps, 0)

    def test_validation_rejects_invalid_boolean_flag_and_int32_overflow(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.nav.step_navigation_scalar(self.config, self.state(target_valid=2), self.post())
        with self.assertRaisesRegex(ValueError, "int32"):
            self.nav.tensors_from_cases(
                self.config,
                [self.state(episode_steps=2**40)],
                [self.post()],
                torch=object(),
                device="cuda",
            )

    def test_python_rejects_pre_arithmetic_int32_boundary(self):
        max_i32 = self.nav.INT32_MAX
        boundary_fields = [
            "episode_steps",
            "steps_since_log",
            "hold_steps",
            "falls",
            "interval_falls",
            "targets_reached",
            "interval_targets_reached",
            "targets_this_episode",
            "total_targets",
        ]
        for field in boundary_fields:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "pre-arithmetic"):
                    self.nav.step_navigation_scalar(self.config, self.state(**{field: max_i32}), self.post())

        with self.assertRaisesRegex(ValueError, "settle/post-goal"):
            self.nav.NavigationConfig(
                completion_mode="sequence",
                control_mode="movement",
                max_targets=3,
                time_limit_steps=12,
                settle_steps=max_i32,
                post_goal_required_steps=1,
                settle_speed=1.0,
                fall_penalty=-5.0,
            )
            self.nav.step_navigation_scalar(
                self.nav.NavigationConfig(
                    completion_mode="sequence",
                    control_mode="movement",
                    max_targets=3,
                    time_limit_steps=12,
                    settle_steps=max_i32,
                    post_goal_required_steps=1,
                    settle_speed=1.0,
                    fall_penalty=-5.0,
                ),
                self.state(),
                self.post(),
            )

    def test_missing_player_matches_frozen_counter_increment_and_raw_return(self):
        state = self.state(episode_steps=4, steps_since_log=9)
        post = self.post(player_present=False, raw_reward=2.5, raw_truncated=True)
        wrapper, _env = make_oracle(state, self.config, post)

        _obs, reward, terminated, truncated, info = wrapper.step(np.ones(10, dtype=np.float64))
        result = self.nav.step_navigation_scalar(self.config, state, post)

        self.assertEqual(result.state.episode_steps, wrapper.episode_steps)
        self.assertEqual(result.state.steps_since_log, wrapper.steps_since_log)
        self.assertEqual(result.reward, reward)
        self.assertEqual(result.nav_reward, 0.0)
        self.assertEqual(result.terminated, terminated)
        self.assertEqual(result.truncated, truncated)
        self.assertEqual(info, {})

    def test_python_batch_matches_scalar_for_required_lane_counts_and_permutation(self):
        cases = []
        for index in range(4096):
            state = self.state(
                target_x=float(index % 7),
                target_y=float((index % 5) - 2),
                last_distance=3.0 + (index % 3) * 0.25,
                episode_steps=index % 10,
                hold_steps=index % 2,
            )
            post = self.post(
                player_x=float(index % 7) + 0.2,
                player_y=float((index % 5) - 2) + 0.1,
                velocity_x=0.01 * (index % 4),
                raw_reward=1.0 + index * 1e-6,
            )
            cases.append((state, post))

        for lanes in (1, 256, 1024, 4096):
            batch = self.nav.step_navigation_batch_python(self.config, [s for s, _ in cases[:lanes]], [p for _, p in cases[:lanes]])
            scalar = [self.nav.step_navigation_scalar(self.config, s, p) for s, p in cases[:lanes]]
            self.assertEqual([r.state.episode_steps for r in batch], [r.state.episode_steps for r in scalar])
            np.testing.assert_allclose([r.reward for r in batch], [r.reward for r in scalar], atol=TOL, rtol=0)

            perm = list(reversed(range(lanes)))
            permuted = self.nav.step_navigation_batch_python(
                self.config,
                [cases[i][0] for i in perm],
                [cases[i][1] for i in perm],
            )
            self.assertEqual(
                [r.state.episode_steps for r in permuted],
                [scalar[i].state.episode_steps for i in perm],
            )

    def test_batch_pack_rejects_state_post_length_mismatch_before_tensor_creation(self):
        class TorchSentinel:
            def __getattr__(self, name):
                raise AssertionError(f"torch must not be used after mismatched inputs: {name}")

        with self.assertRaisesRegex(ValueError, "same length"):
            self.nav.tensors_from_cases(
                self.config,
                [self.state(), self.state()],
                [self.post()],
                torch=TorchSentinel(),
                device="cuda",
            )

    def test_compiled_cpu_reference_matches_python_scalar(self):
        if not self.nav.can_compile_cpu_reference():
            self.skipTest("no C compiler available")
        with tempfile.TemporaryDirectory() as tmp:
            cpu = self.nav.load_cpu_reference(Path(tmp))
            state = self.state(hold_steps=1)
            post = self.post(next_target_x=5.0, next_target_y=0.0)

            py_result = self.nav.step_navigation_scalar(self.config, state, post)
            c_result = cpu.step(self.config, state, post)

            self.assertEqual(c_result.state.targets_this_episode, py_result.state.targets_this_episode)
            self.assertEqual(c_result.state.hold_steps, py_result.state.hold_steps)
            self.assertAlmostEqual(c_result.state.last_distance, py_result.state.last_distance, delta=TOL)
            self.assertAlmostEqual(c_result.reward, py_result.reward, delta=TOL)
            self.assertEqual(c_result.reached, py_result.reached)
            self.assertEqual(c_result.handoff, py_result.handoff)

    def test_compiled_cpu_reference_relative_build_dir_uses_callers_cwd(self):
        if not self.nav.can_compile_cpu_reference():
            self.skipTest("no C compiler available")
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                cpu = self.nav.load_cpu_reference(Path("relative-build"))
            finally:
                os.chdir(old_cwd)
            expected_parent = (Path(tmp) / "relative-build").resolve()
            self.assertEqual(Path(cpu.library_path).resolve().parent, expected_parent)
            self.assertTrue(Path(cpu.library_path).exists())
            self.assertEqual(cpu.step(self.config, self.state(), self.post()).state.episode_steps, 1)

    def test_compiled_cpu_reference_reports_missing_next_target(self):
        if not self.nav.can_compile_cpu_reference():
            self.skipTest("no C compiler available")
        with tempfile.TemporaryDirectory() as tmp:
            cpu = self.nav.load_cpu_reference(Path(tmp))
            with self.assertRaisesRegex(ValueError, "next target"):
                cpu.step(self.config, self.state(hold_steps=1), self.post(next_target_valid=False))

    def test_compiled_cpu_reference_rejects_invalid_native_inputs(self):
        if not self.nav.can_compile_cpu_reference():
            self.skipTest("no C compiler available")
        with tempfile.TemporaryDirectory() as tmp:
            cpu = self.nav.load_cpu_reference(Path(tmp))
            with self.assertRaisesRegex(ValueError, "invalid navigation input"):
                cpu.step(self.config, self.state(target_valid=2), self.post())
            with self.assertRaisesRegex(ValueError, "invalid navigation input"):
                cpu.step(self.config, self.state(), self.post(player_x=float("nan")))
            with self.assertRaisesRegex(ValueError, "invalid navigation input"):
                cpu.step(self.config, self.state(episode_steps=self.nav.INT32_MAX), self.post())
            overflow_config = self.nav.NavigationConfig(
                completion_mode="sequence",
                control_mode="movement",
                max_targets=3,
                time_limit_steps=12,
                settle_steps=self.nav.INT32_MAX,
                post_goal_required_steps=1,
                settle_speed=1.0,
                fall_penalty=-5.0,
            )
            with self.assertRaisesRegex(ValueError, "invalid navigation input"):
                cpu.step(overflow_config, self.state(), self.post())

    def test_native_c_and_cuda_wrappers_share_header_rule_core(self):
        root = Path(__file__).resolve().parents[1]
        header = (root / "native" / "navigation_contract.h").read_text()
        cpu_source = (root / "native" / "navigation_contract.c").read_text()
        cuda_source = (root / "native" / "navigation_contract.cu").read_text()

        self.assertIn("navigation_contract_step_core", header)
        self.assertIn("navigation_contract_step_core(config, state, post, out)", cpu_source)
        self.assertIn("navigation_contract_step_core(&config, &state, &post, &result)", cuda_source)
        self.assertNotIn("nav_distance_cuda", cuda_source)
        self.assertNotIn("nav_is_settled_cuda", cuda_source)

    def test_cuda_binding_contract_is_strict_or_explicitly_unavailable(self):
        if not self.nav.can_run_cuda_contract():
            self.skipTest("CUDA torch and nvcc are unavailable")
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            cuda = self.nav.load_cuda_contract(Path(tmp))
            for lanes in (1, 256, 1024, 4096):
                states = [self.state(hold_steps=i % 2) for i in range(lanes)]
                posts = [self.post(player_x=0.2 + i * 0.001) for i in range(lanes)]
                tensors = self.nav.tensors_from_cases(self.config, states, posts, torch=torch, device="cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    result = cuda.step(tensors)
                stream.synchronize()
                self.assertEqual(result["episode_steps"].shape[0], lanes)
                unpacked = self.nav.unpack_cuda_results(result)
                expected = [self.nav.step_navigation_scalar(self.config, s, p) for s, p in zip(states, posts)]
                for got, want in zip(unpacked, expected):
                    self.assertEqual(got.state, want.state)
                    self.assertAlmostEqual(got.reward, want.reward, delta=TOL)
                    self.assertAlmostEqual(got.nav_reward, want.nav_reward, delta=TOL)
                    self.assertEqual(got.terminated, want.terminated)
                    self.assertEqual(got.truncated, want.truncated)
                    self.assertEqual(got.underlying_done, want.underlying_done)
                    self.assertEqual(got.touched, want.touched)
                    self.assertEqual(got.reached, want.reached)
                    self.assertEqual(got.fall, want.fall)
                    self.assertEqual(got.handoff, want.handoff)
                    self.assertEqual(got.success, want.success)
                    self.assertEqual(got.horizon, want.horizon)
                with self.assertRaisesRegex(ValueError, "contiguous"):
                    cuda.step({**tensors, "player_xy": torch.empty((lanes, 4), dtype=torch.float64, device="cuda")[:, ::2]})
                with self.assertRaisesRegex(ValueError, "width 13"):
                    cuda.step({**tensors, "state_i32": torch.empty((lanes, 12), dtype=torch.int32, device="cuda")})
                with self.assertRaisesRegex(ValueError, "width 1"):
                    cuda.step({**tensors, "post_i32": torch.empty((lanes, 2), dtype=torch.int32, device="cuda")})
                with self.assertRaisesRegex(ValueError, "width 11"):
                    cuda.step({**tensors, "flags": torch.empty((lanes, 10), dtype=torch.uint8, device="cuda")})

            lane_error_tensors = self.nav.tensors_from_cases(
                self.config,
                [self.state(hold_steps=1), self.state()],
                [self.post(next_target_valid=False), self.post()],
                torch=torch,
                device="cuda",
            )
            with self.assertRaisesRegex(RuntimeError, "per-lane errors"):
                cuda.step(lane_error_tensors)

            overflow_tensors = self.nav.tensors_from_cases(
                self.config,
                [self.state(), self.state()],
                [self.post(), self.post()],
                torch=torch,
                device="cuda",
            )
            overflow_tensors["state_i32"][0, 0] = self.nav.INT32_MAX
            with self.assertRaisesRegex(RuntimeError, "per-lane errors"):
                cuda.step(overflow_tensors)


if __name__ == "__main__":
    unittest.main()
