import importlib.util
import json
import os
import random
import sys
import tempfile
import time
import types
import unittest
import unittest.mock
from pathlib import Path

try:
    import gymnasium as _spawn_gym
    import numpy as _spawn_np
except ModuleNotFoundError:
    _spawn_gym = None
    _spawn_np = None


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmark_worker_profile.py"


def load_module():
    return load_module_as("benchmark_worker_profile")


def load_module_as(name):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeRawGame:
    def reset(self):
        return "reset"

    def observe(self):
        return "obs"

    def step(self, action):
        self.observe()
        return f"step:{action}"


class FakeEnv:
    def __init__(self):
        self.raw_env = FakeRawGame()
        self.unwrapped = types.SimpleNamespace(raw_env=self.raw_env)
        self.closed = False

    def reset(self):
        return self.raw_env.reset()

    def step(self, action):
        return self.raw_env.step(action)

    def close(self):
        self.closed = True


class FakeVecEnv:
    def __init__(self, fns):
        self.envs = [fn() for fn in fns]
        self.closed = False

    def env_method(self, method_name):
        return [getattr(env, method_name)() for env in self.envs]

    def step_async(self, actions):
        self.actions = list(actions)

    def step_wait(self):
        return [env.step(action) for env, action in zip(self.envs, self.actions)]

    def close(self):
        self.closed = True


if _spawn_gym is not None and _spawn_np is not None:

    class ProfileSpawnRawGame:
        def reset(self):
            return _spawn_np.zeros((1,), dtype=_spawn_np.float32)

        def observe(self):
            return _spawn_np.zeros((1,), dtype=_spawn_np.float32)

        def step(self, action):
            self.observe()
            return _spawn_np.zeros((1,), dtype=_spawn_np.float32), 0.0, False, False, {}


    class ProfileSpawnTinyEnv(_spawn_gym.Env):
        metadata = {}

        def __init__(self):
            self.raw_env = ProfileSpawnRawGame()
            self.observation_space = _spawn_gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=_spawn_np.float32)
            self.action_space = _spawn_gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=_spawn_np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            obs = self.raw_env.reset()
            return obs, {}

        def step(self, action):
            return self.raw_env.step(action)


def make_profile_spawn_tiny_env():
    if _spawn_gym is None or _spawn_np is None:
        raise RuntimeError("missing gymnasium/numpy")
    return ProfileSpawnTinyEnv()


class WorkerProfileTests(unittest.TestCase):
    def test_span_patch_is_idempotent_for_same_span_and_label(self):
        profile = load_module()

        class Target:
            def step(self):
                return "ok"

        spans = profile.Spans()
        spans.patch(Target, "step", "component.step")
        spans.patch(Target, "step", "component.step")

        spans.enabled = True
        self.assertEqual(Target().step(), "ok")
        spans.enabled = False

        self.assertEqual(spans.report()["component.step"]["count"], 1)

    def test_worker_component_patch_does_not_double_count_raw_game_class(self):
        profile = load_module()

        class RawGame:
            def reset(self):
                return "reset"

            def observe(self):
                return "obs"

            def step(self, action):
                self.observe()
                return action

        class EnvWithRaw:
            def __init__(self):
                self.raw_env = RawGame()
                self.unwrapped = types.SimpleNamespace(raw_env=self.raw_env)

            def reset(self):
                return self.raw_env.reset()

            def step(self, action):
                return self.raw_env.step(action)

            def close(self):
                return None

        game_module = types.ModuleType("environment.environment")
        game_module.WarehouseBrawl = RawGame
        package_module = types.ModuleType("environment")

        with unittest.mock.patch.dict(
            sys.modules,
            {"environment": package_module, "environment.environment": game_module},
        ):
            env = profile.profile_worker_env(EnvWithRaw(), worker_index=0)

        env.start_worker_measurement()
        env.step("left")
        components = env.finish_worker_measurement()["components"]

        self.assertEqual(components["game.step"]["count"], 1)
        self.assertEqual(components["game.observation"]["count"], 1)

    def test_shared_class_component_hooks_route_to_current_worker_spans(self):
        profile = load_module()

        class SharedRawGame:
            def reset(self):
                return "reset"

            def observe(self):
                time.sleep(0.001)
                return "obs"

            def step(self, action):
                self.observe()
                return action

        class EnvWithSharedRaw:
            def __init__(self):
                self.raw_env = SharedRawGame()
                self.unwrapped = types.SimpleNamespace(raw_env=self.raw_env)

            def reset(self):
                return self.raw_env.reset()

            def step(self, action):
                return self.raw_env.step(action)

            def close(self):
                return None

        game_module = types.ModuleType("environment.environment")
        game_module.WarehouseBrawl = SharedRawGame
        package_module = types.ModuleType("environment")

        with unittest.mock.patch.dict(
            sys.modules,
            {"environment": package_module, "environment.environment": game_module},
        ):
            env0 = profile.profile_worker_env(EnvWithSharedRaw(), worker_index=0)
            env1 = profile.profile_worker_env(EnvWithSharedRaw(), worker_index=1)

        env0.start_worker_measurement()
        env1.start_worker_measurement()
        env0.step("left")
        env1.step("right")
        report0 = env0.finish_worker_measurement()["components"]
        report1 = env1.finish_worker_measurement()["components"]

        for report in (report0, report1):
            self.assertEqual(report["game.step"]["count"], 1)
            self.assertEqual(report["game.observation"]["count"], 1)
            self.assertLess(
                report["worker.env_step"]["exclusive_wall_seconds"],
                report["worker.env_step"]["inclusive_wall_seconds"],
            )
        self.assertEqual(report0["game.step"]["count"] + report1["game.step"]["count"], 2)
        self.assertEqual(report0["game.observation"]["count"] + report1["game.observation"]["count"], 2)

    def test_worker_step_exception_restores_active_span_context(self):
        profile = load_module()

        class ExplodingRawGame:
            def observe(self):
                return "obs-before-boom"

            def step(self, action):
                self.observe()
                raise RuntimeError(f"boom:{action}")

        class EnvWithExplodingRaw:
            def __init__(self):
                self.raw_env = ExplodingRawGame()
                self.unwrapped = types.SimpleNamespace(raw_env=self.raw_env)

            def reset(self):
                return "reset"

            def step(self, action):
                return self.raw_env.step(action)

            def close(self):
                return None

        game_module = types.ModuleType("environment.environment")
        game_module.WarehouseBrawl = ExplodingRawGame
        package_module = types.ModuleType("environment")

        with unittest.mock.patch.dict(
            sys.modules,
            {"environment": package_module, "environment.environment": game_module},
        ):
            env = profile.profile_worker_env(EnvWithExplodingRaw(), worker_index=0)

        env.start_worker_measurement()
        with self.assertRaisesRegex(RuntimeError, "boom:left"):
            env.step("left")
        components = env.finish_worker_measurement()["components"]

        self.assertIsNone(profile._ACTIVE_WORKER_SPANS.get())
        self.assertEqual(components["worker.env_step"]["count"], 1)
        self.assertEqual(components["game.step"]["count"], 1)
        self.assertEqual(components["game.observation"]["count"], 1)
        self.assertEqual(len(env.step_events), 1)

    def test_profile_worker_env_delegates_to_import_stable_module_when_called_from_main_copy(self):
        stable_profile = load_module()
        main_copy = load_module_as("profile_main_copy_for_delegate")
        main_copy.__name__ = "__main__"

        class Env:
            def close(self):
                return None

        with unittest.mock.patch("importlib.import_module", return_value=stable_profile) as import_module:
            wrapped = main_copy.profile_worker_env(Env(), worker_index=2)

        self.assertIsInstance(wrapped, stable_profile.WorkerTimingWrapper)
        self.assertEqual(len(stable_profile._GROUPED_RNG_SPAN_TARGETS), 1)
        self.assertEqual(len(main_copy._GROUPED_RNG_SPAN_TARGETS), 0)
        import_module.assert_any_call("benchmark_worker_profile")

    def test_worker_component_patch_records_grounded_frozen_hotpath_labels(self):
        profile = load_module()

        class WarehouseBrawl:
            def _observe_all(self):
                return {}

        class Stage:
            def physics_process(self, delta):
                return delta

        class Player:
            def pre_process(self):
                return "ready"

        class Camera:
            def process(self):
                return "frame"

        game_module = types.ModuleType("environment.environment")
        game_module.WarehouseBrawl = WarehouseBrawl
        game_module.Stage = Stage
        game_module.Player = Player
        game_module.Camera = Camera
        package_module = types.ModuleType("environment")

        spans = profile.Spans()
        with unittest.mock.patch.dict(
            sys.modules,
            {"environment": package_module, "environment.environment": game_module},
        ):
            profile.patch_worker_components(types.SimpleNamespace(), spans)

        spans.enabled = True
        WarehouseBrawl()._observe_all()
        Stage().physics_process(1 / 60)
        Player().pre_process()
        Camera().process()
        spans.enabled = False

        components = spans.report()
        self.assertEqual(components["game.observe_all"]["count"], 1)
        self.assertEqual(components["game.stage_platform_physics"]["count"], 1)
        self.assertEqual(components["game.player_pre_process"]["count"], 1)
        self.assertEqual(components["game.camera_process"]["count"], 1)

    def test_worker_wrapper_records_inclusive_and_exclusive_component_spans(self):
        profile = load_module()
        env = profile.profile_worker_env(FakeEnv(), worker_index=3)

        self.assertEqual(env.start_worker_measurement(), 3)
        env.reset()
        env.step("a")
        report = env.finish_worker_measurement()

        self.assertEqual(report["worker_index"], 3)
        components = report["components"]
        self.assertIn("worker.env_step", components)
        self.assertIn("worker.env_reset", components)
        self.assertIn("game.step", components)
        self.assertIn("game.observation", components)
        self.assertEqual(components["worker.env_step"]["count"], 1)
        self.assertEqual(components["game.observation"]["count"], 1)
        self.assertIn("inclusive_wall_seconds", components["worker.env_step"])
        self.assertIn("inclusive_thread_cpu_seconds", components["worker.env_step"])
        self.assertGreaterEqual(
            components["worker.env_step"]["inclusive_wall_seconds"],
            components["worker.env_step"]["exclusive_wall_seconds"],
        )
        self.assertIn("not additive", report["timing_note"])
        self.assertIn("thread CPU", report["timing_note"])

    def test_gymnasium_wrapper_survives_sb3_patch_env_and_get_wrapper_attr(self):
        profile = load_module()
        try:
            import gymnasium as gym
            import numpy as np
            from stable_baselines3.common.vec_env.patch_gym import _patch_env
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing runtime dependency for SB3 wrapper check: {exc.name}")

        class TinyEnv(gym.Env):
            metadata = {}

            def __init__(self):
                self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros((1,), dtype=np.float32), {}

            def step(self, action):
                return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

        wrapped = profile.profile_worker_env(TinyEnv(), worker_index=7)
        patched = _patch_env(wrapped)

        self.assertIsInstance(patched, gym.Env)
        self.assertEqual(patched.get_wrapper_attr("start_worker_measurement")(), 7)
        patched.step(np.zeros((1,), dtype=np.float32))
        self.assertEqual(patched.get_wrapper_attr("finish_worker_measurement")()["worker_index"], 7)

    def test_build_env_patch_wraps_exact_frozen_factories_and_preserves_vec_constructor(self):
        profile = load_module()
        calls = []

        class FakeTrainJob:
            DummyVecEnv = FakeVecEnv
            SubprocVecEnv = FakeVecEnv

            def build_env(self, train_module, phase_config, params, n_envs, seed, device):
                def make_one(index):
                    def thunk():
                        calls.append(("factory", index))
                        return FakeEnv()

                    return thunk

                return self.DummyVecEnv([make_one(index) for index in range(n_envs)])

        train_job = FakeTrainJob()
        profile.install_build_env_profile(train_job)

        env = train_job.build_env(object(), object(), {}, 2, 45, "cuda")
        started = env.env_method("start_worker_measurement")
        env.step_async(["left", "right"])
        env.step_wait()
        rows = env.env_method("finish_worker_measurement")

        self.assertEqual(calls, [("factory", 0), ("factory", 1)])
        self.assertEqual(started, [0, 1])
        self.assertEqual([row["worker_index"] for row in rows], [0, 1])
        self.assertIs(train_job.DummyVecEnv, FakeVecEnv)

    def test_build_env_profile_preserves_later_grouped_subproc_patch(self):
        profile = load_module()
        calls = []

        class OriginalVec(FakeVecEnv):
            pass

        class GroupedVec(FakeVecEnv):
            pass

        class FakeTrainJob:
            DummyVecEnv = OriginalVec
            SubprocVecEnv = OriginalVec

            def build_env(self, train_module, phase_config, params, n_envs, seed, device):
                def make_one(index):
                    def thunk():
                        calls.append(("factory", index))
                        return FakeEnv()

                    return thunk

                return self.SubprocVecEnv([make_one(index) for index in range(n_envs)])

        train_job = FakeTrainJob()
        profile.install_build_env_profile(train_job)
        train_job.SubprocVecEnv = GroupedVec

        env = train_job.build_env(object(), object(), {}, 2, 45, "cuda")

        self.assertIsInstance(env, GroupedVec)
        self.assertEqual(calls, [("factory", 0), ("factory", 1)])
        self.assertIs(train_job.SubprocVecEnv, GroupedVec)

    def test_grouped_rng_state_run_records_exclusive_swap_spans(self):
        profile = load_module()

        class GlobalRngStates:
            def __init__(self):
                import numpy as np

                self.states = [(random.getstate(), np.random.get_state())]

            def run(self, index, func, *, seed=None):
                return func()

        grouped_module = types.ModuleType("grouped_vec_env")
        grouped_module._GlobalRngStates = GlobalRngStates

        with unittest.mock.patch.dict(sys.modules, {"grouped_vec_env": grouped_module}):
            profile.install_grouped_rng_state_profile()
            env = profile.profile_worker_env(FakeEnv(), worker_index=0)

            env.start_worker_measurement()
            GlobalRngStates().run(0, lambda: "ok")
            components = env.finish_worker_measurement()["components"]

        self.assertIn("grouped_rng.save_outer_state", components)
        self.assertIn("grouped_rng.restore_lane_state", components)
        self.assertIn("grouped_rng.capture_lane_state", components)
        self.assertIn("grouped_rng.restore_outer_state", components)
        self.assertEqual(components["grouped_rng.save_outer_state"]["count"], 1)
        self.assertEqual(components["grouped_rng.restore_outer_state"]["count"], 1)
        self.assertLess(
            components["grouped_rng.save_outer_state"]["exclusive_wall_seconds"],
            components["worker.env_step"]["inclusive_wall_seconds"]
            if "worker.env_step" in components
            else float("inf"),
        )

    def test_build_env_patch_fails_closed_when_worker_measurement_cannot_start(self):
        profile = load_module()

        class NoMethodVecEnv(FakeVecEnv):
            def env_method(self, method_name):
                raise AttributeError(method_name)

        class FakeTrainJob:
            DummyVecEnv = NoMethodVecEnv
            SubprocVecEnv = NoMethodVecEnv

            def build_env(self, train_module, phase_config, params, n_envs, seed, device):
                return self.DummyVecEnv([FakeEnv])

        train_job = FakeTrainJob()
        profile.install_build_env_profile(train_job)

        with self.assertRaisesRegex(RuntimeError, "start_worker_measurement"):
            train_job.build_env(object(), object(), {}, 1, 45, "cuda")

    def test_parent_vec_timer_records_dispatch_and_wait_without_worker_wall_claim(self):
        profile = load_module()
        timer = profile.ParentVecTimer()
        env = FakeVecEnv([FakeEnv, FakeEnv])
        profile.patch_parent_vec_timing(env, timer)

        env.step_async(["a", "b"])
        env.step_wait()
        report = timer.report()

        self.assertEqual(report["parent.step_async_send"]["count"], 1)
        self.assertEqual(report["parent.step_wait_recv_stack"]["count"], 1)
        self.assertIn("does not decompose worker CPU", report["note"])

    def test_close_writes_worker_and_parent_profile_before_vec_env_close(self):
        profile = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            env = FakeVecEnv([lambda: profile.profile_worker_env(FakeEnv(), worker_index=0)])
            profile.attach_close_report(env, output, profile.ParentVecTimer())
            env.env_method("start_worker_measurement")
            env.step_async(["a"])
            env.step_wait()

            env.close()

            payload = json.loads((output / "worker_profile.json").read_text())

        self.assertTrue(env.closed)
        self.assertEqual(payload["label"], "worker_env_parent_profile")
        self.assertEqual(payload["worker_reports"][0]["worker_index"], 0)
        self.assertIn("parent", payload)

    def test_close_still_closes_vec_env_when_worker_profile_report_fails(self):
        profile = load_module()

        class BrokenEnv(FakeVecEnv):
            def env_method(self, method_name):
                if method_name == "finish_worker_measurement":
                    raise RuntimeError("worker gone")
                return super().env_method(method_name)

        with tempfile.TemporaryDirectory() as tmp:
            env = BrokenEnv([lambda: profile.profile_worker_env(FakeEnv(), worker_index=0)])
            profile.attach_close_report(env, Path(tmp), profile.ParentVecTimer())

            with self.assertRaisesRegex(RuntimeError, "finish_worker_measurement failed"):
                env.close()

        self.assertTrue(env.closed)

    def test_build_campaign_args_uses_current_64_by_512_checkpoint_defaults(self):
        profile = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            args = profile.build_parser().parse_args(
                [
                    "--output",
                    str(output),
                    "--config",
                    str(Path(tmp) / "control.yaml"),
                    "--checkpoint",
                    str(Path(tmp) / "selected.zip"),
                    "--source",
                    str(Path(tmp) / "source"),
                    "--train-job",
                    str(Path(tmp) / "train_job.py"),
                ]
            )
            campaign_args = profile.build_campaign_args(profile.validate_args(args))

        self.assertEqual(campaign_args.n_envs, 64)
        self.assertEqual(campaign_args.n_steps, 512)
        self.assertEqual(campaign_args.batch_size, 8192)
        self.assertEqual(campaign_args.epochs, 8)
        self.assertEqual(campaign_args.cycles, 10)
        self.assertEqual(campaign_args.warmup, 0)
        self.assertEqual(campaign_args.device, "cuda")
        self.assertEqual(campaign_args.output, output.resolve() / "campaign")
        self.assertEqual(campaign_args.group_size, 1)

    def test_build_campaign_args_includes_every_current_campaign_parser_field(self):
        profile = load_module()
        import benchmark_campaign

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            args = profile.build_parser().parse_args(
                [
                    "--output",
                    str(output),
                    "--config",
                    str(Path(tmp) / "control.yaml"),
                    "--checkpoint",
                    str(Path(tmp) / "selected.zip"),
                    "--source",
                    str(Path(tmp) / "source"),
                    "--train-job",
                    str(Path(tmp) / "train_job.py"),
                ]
            )
            campaign_args = profile.build_campaign_args(profile.validate_args(args))

        parser_dests = {
            action.dest
            for action in benchmark_campaign.build_parser()._actions
            if action.dest != "help"
        }
        missing = sorted(dest for dest in parser_dests if not hasattr(campaign_args, dest))

        self.assertEqual(missing, [])
        self.assertEqual(campaign_args.rollout_policy_graph_capture_warmup, 3)
        self.assertFalse(campaign_args.rollout_policy_graph)

    def test_build_campaign_args_forwards_grouped_geometry_and_grouped_env_path(self):
        profile = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            grouped = Path(tmp) / "grouped_vec_env.py"
            args = profile.build_parser().parse_args(
                [
                    "--output",
                    str(output),
                    "--config",
                    str(Path(tmp) / "control.yaml"),
                    "--checkpoint",
                    str(Path(tmp) / "selected.zip"),
                    "--source",
                    str(Path(tmp) / "source"),
                    "--train-job",
                    str(Path(tmp) / "train_job.py"),
                    "--n-envs",
                    "256",
                    "--n-steps",
                    "128",
                    "--group-size",
                    "4",
                    "--grouped-env-path",
                    str(grouped),
                ]
            )
            campaign_args = profile.build_campaign_args(profile.validate_args(args))

        self.assertEqual(campaign_args.n_envs, 256)
        self.assertEqual(campaign_args.n_steps, 128)
        self.assertEqual(campaign_args.group_size, 4)
        self.assertEqual(campaign_args.grouped_env_path, grouped.resolve())

    def test_actual_grouped_toy_env_records_rng_swap_spans_when_dependencies_exist(self):
        profile = load_module()
        try:
            import gymnasium as gym
            import numpy as np
            import grouped_vec_env
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing grouped runtime dependency: {exc.name}")

        if os.environ.get("AI2_RUN_SPAWN_PROFILE_TEST") != "1":
            self.skipTest("set AI2_RUN_SPAWN_PROFILE_TEST=1 for native grouped spawn smoke")

        class TinyEnv(gym.Env):
            metadata = {}

            def __init__(self):
                self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros((1,), dtype=np.float32), {}

            def step(self, action):
                return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

        profile.install_grouped_rng_state_profile()
        env = grouped_vec_env.GroupedSubprocVecEnv(
            [lambda i=i: profile.profile_worker_env(TinyEnv(), worker_index=i) for i in range(4)],
            group_size=2,
            start_method="spawn",
        )
        try:
            env.reset()
            env.env_method("start_worker_measurement")
            env.step_async(np.zeros((4, 1), dtype=np.float32))
            env.step_wait()
            reports = env.env_method("finish_worker_measurement")
        finally:
            env.close()

        labels = reports[0]["components"]
        self.assertIn("grouped_rng.save_outer_state", labels)
        self.assertIn("grouped_rng.restore_outer_state", labels)

    def test_actual_spawn_main_copy_profile_routes_all_lanes_to_stable_registry(self):
        load_module()
        main_copy = load_module_as("profile_main_copy")
        main_copy.__dict__["__name__"] = "__main__"
        main_copy._ProfiledEnvFactory.__module__ = "benchmark_worker_profile"
        try:
            import gymnasium as gym
            import numpy as np
            import grouped_vec_env
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing grouped runtime dependency: {exc.name}")

        if os.environ.get("AI2_RUN_SPAWN_PROFILE_TEST") != "1":
            self.skipTest("set AI2_RUN_SPAWN_PROFILE_TEST=1 for native grouped spawn smoke")

        wrapped_fns = main_copy._wrap_factories([make_profile_spawn_tiny_env for _ in range(4)])
        env = grouped_vec_env.GroupedSubprocVecEnv(wrapped_fns, group_size=4, start_method="spawn")
        try:
            env.reset()
            env.env_method("start_worker_measurement")
            env.step_async(np.zeros((4, 1), dtype=np.float32))
            env.step_wait()
            reports = env.env_method("finish_worker_measurement")
        finally:
            env.close()

        for report in reports:
            components = report["components"]
            self.assertIn("grouped_rng.save_outer_state", components)
            self.assertIn("grouped_rng.restore_lane_state", components)
            self.assertIn("grouped_rng.capture_lane_state", components)
            self.assertIn("grouped_rng.restore_outer_state", components)
            self.assertEqual(components["game.step"]["count"], 1)
            self.assertEqual(components["game.observation"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
