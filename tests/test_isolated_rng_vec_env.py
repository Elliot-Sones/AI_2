"""Tests for opt-in per-lane RNG object isolation."""

from __future__ import annotations

import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import isolated_rng_vec_env as iso

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - exercised on dependency-light hosts.
    gym = None
    spaces = None


def _exercise_random_sources(random_module: object, np_module: object) -> tuple[object, ...]:
    return (
        random_module.getrandbits(7),
        round(random_module.random(), 15),
        round(random_module.uniform(-3.0, 4.0), 15),
        random_module.randint(2, 9),
        random_module.choice(("a", "b", "c", "d")),
        tuple(random_module.choices(("x", "y", "z"), weights=(1, 3, 2), k=5)),
        int(np_module.random.randint(0, 100)),
        tuple(int(v) for v in np_module.random.randint(0, 8, size=4)),
        round(float(np_module.random.uniform(-1.0, 1.0)), 15),
        tuple(round(float(v), 15) for v in np_module.random.uniform(-2.0, 2.0, size=3)),
    )


class IsolatedRngProxyTest(unittest.TestCase):
    def test_lane_scope_uses_contextvar_not_process_global_stack(self):
        self.assertEqual(iso.current_rng_bundle.__class__.__name__, "_CurrentRngBundle")
        self.assertTrue(hasattr(iso.current_rng_bundle, "_current"))
        import contextvars

        self.assertIsInstance(iso.current_rng_bundle._current, contextvars.ContextVar)

    def test_bundle_matches_global_python_and_numpy_rng_sequences_and_final_states(self):
        random.seed(12345)
        np.random.seed(54321)
        py_state = random.getstate()
        np_state = np.random.get_state()

        random.setstate(py_state)
        np.random.set_state(np_state)
        expected_values = _exercise_random_sources(random, np)
        expected_py_state = random.getstate()
        expected_np_state = np.random.get_state()

        bundle = iso.EnvRngBundle.from_states(py_state, np_state)
        random_proxy = iso.RandomModuleProxy(iso.current_rng_bundle)
        np_proxy = iso.NumpyModuleProxy(np, iso.NumpyRandomProxy(iso.current_rng_bundle))
        with iso.use_rng_bundle(bundle):
            actual_values = _exercise_random_sources(random_proxy, np_proxy)

        self.assertEqual(actual_values, expected_values)
        self.assertEqual(bundle.python_random.getstate(), expected_py_state)
        np.testing.assert_array_equal(bundle.numpy_random.get_state()[1], expected_np_state[1])
        self.assertEqual(bundle.numpy_random.get_state()[0], expected_np_state[0])
        self.assertEqual(bundle.numpy_random.get_state()[2:], expected_np_state[2:])

    def test_proxy_fails_closed_outside_active_lane_context(self):
        random_proxy = iso.RandomModuleProxy(iso.current_rng_bundle)
        np_proxy = iso.NumpyRandomProxy(iso.current_rng_bundle)

        with self.assertRaisesRegex(RuntimeError, "outside an active isolated RNG lane"):
            random_proxy.random()
        with self.assertRaisesRegex(RuntimeError, "outside an active isolated RNG lane"):
            np_proxy.uniform(0.0, 1.0)
        with self.assertRaisesRegex(AttributeError, "not allowed"):
            random_proxy.gauss(0.0, 1.0)

    def test_module_patch_restores_originals_after_exception(self):
        fake_random_module = types.SimpleNamespace(random=random, label="game")
        fake_train_module = types.SimpleNamespace(np=np, label="train")
        original_np_random = np.random
        patch = iso.ScopedRngPatch(
            random_modules=[fake_random_module],
            numpy_modules=[fake_train_module],
        )

        with self.assertRaisesRegex(ValueError, "boom"):
            with patch.installed():
                self.assertIsInstance(fake_random_module.random, iso.RandomModuleProxy)
                self.assertIsInstance(fake_train_module.np, iso.NumpyModuleProxy)
                self.assertIs(np.random, original_np_random)
                raise ValueError("boom")

        self.assertIs(fake_random_module.random, random)
        self.assertIs(fake_train_module.np, np)
        self.assertIs(np.random, original_np_random)

    def test_discover_frozen_rng_modules_uses_loaded_game_and_train_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.py"
            env_path = root / "UTMIST-AI2-main" / "environment" / "environment.py"
            env_path.parent.mkdir(parents=True)
            train_path.write_text("train", encoding="utf-8")
            env_path.write_text("env", encoding="utf-8")
            fake_game_module = types.SimpleNamespace(__file__=str(env_path), random=random)
            fake_agent_module = types.SimpleNamespace(random=random)
            fake_train_module = types.SimpleNamespace(__file__=str(train_path), np=np)
            previous = {
                name: sys.modules.get(name)
                for name in ("environment.environment", "environment.agent", "train")
            }
            try:
                sys.modules["environment.environment"] = fake_game_module
                sys.modules["environment.agent"] = fake_agent_module
                sys.modules["train"] = fake_train_module
                with mock.patch.multiple(
                    iso,
                    FROZEN_TRAIN_SHA256=iso.file_sha256(train_path),
                    FROZEN_ENVIRONMENT_SHA256=iso.file_sha256(env_path),
                ):
                    patch = iso.discover_frozen_rng_modules(root)

                with patch.installed():
                    self.assertIsInstance(fake_game_module.random, iso.RandomModuleProxy)
                    self.assertIsInstance(fake_agent_module.random, iso.RandomModuleProxy)
                    self.assertIsInstance(fake_train_module.np, iso.NumpyModuleProxy)
                    self.assertIs(np.random, np.__dict__["random"])
            finally:
                for name, module in previous.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module

    def test_discover_frozen_rng_modules_fails_when_required_aliases_missing(self):
        previous = {
            name: sys.modules.get(name)
            for name in ("environment.environment", "environment.agent", "train")
        }
        try:
            for name in previous:
                sys.modules.pop(name, None)
            with self.assertRaisesRegex(RuntimeError, "Loaded train.py module not found"):
                iso.discover_frozen_rng_modules()
        finally:
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_verify_frozen_sources_pins_loaded_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.py"
            env_path = root / "UTMIST-AI2-main" / "environment" / "environment.py"
            env_path.parent.mkdir(parents=True)
            train_path.write_text("train", encoding="utf-8")
            env_path.write_text("env", encoding="utf-8")

            fake_train_module = types.SimpleNamespace(__file__=str(train_path), np=np)
            fake_env_module = types.SimpleNamespace(__file__=str(env_path), random=random)
            patch = mock.patch.multiple(
                iso,
                FROZEN_TRAIN_SHA256=iso.file_sha256(train_path),
                FROZEN_ENVIRONMENT_SHA256=iso.file_sha256(env_path),
            )
            previous = {
                name: sys.modules.get(name)
                for name in ("train", "environment.environment")
            }
            try:
                sys.modules["train"] = fake_train_module
                sys.modules["environment.environment"] = fake_env_module
                with patch:
                    record = iso.verify_frozen_sources(root)
                self.assertEqual(record["train_path"], str(train_path.resolve()))
                self.assertEqual(record["environment_path"], str(env_path.resolve()))
            finally:
                for name, module in previous.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module

    def test_verify_frozen_sources_rejects_loaded_module_from_other_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.py"
            env_path = root / "UTMIST-AI2-main" / "environment" / "environment.py"
            other_train = root / "other_train.py"
            env_path.parent.mkdir(parents=True)
            train_path.write_text("train", encoding="utf-8")
            other_train.write_text("train", encoding="utf-8")
            env_path.write_text("env", encoding="utf-8")

            previous_train = sys.modules.get("train")
            previous_env = sys.modules.get("environment.environment")
            sys.modules["train"] = types.SimpleNamespace(__file__=str(other_train), np=np)
            sys.modules["environment.environment"] = types.SimpleNamespace(__file__=str(env_path), random=random)
            try:
                with mock.patch.object(iso, "FROZEN_TRAIN_SHA256", iso.file_sha256(train_path)):
                    with self.assertRaisesRegex(RuntimeError, "train.py source path mismatch"):
                        iso.verify_frozen_sources(root)
            finally:
                if previous_train is None:
                    sys.modules.pop("train", None)
                else:
                    sys.modules["train"] = previous_train
                if previous_env is None:
                    sys.modules.pop("environment.environment", None)
                else:
                    sys.modules["environment.environment"] = previous_env


class IsolatedRngStateManagerTest(unittest.TestCase):
    def test_construction_uses_original_global_state_swap_then_steady_state_uses_no_getset(self):
        manager = iso.IsolatedRngStates(2)
        fake_game_module = types.SimpleNamespace(random=random)
        fake_train_module = types.SimpleNamespace(np=np)

        construction_seen: list[float] = []

        def construction_call() -> None:
            random.seed(111)
            np.random.seed(222)
            construction_seen.append(random.random())
            construction_seen.append(float(np.random.random()))

        manager.run(0, construction_call)
        manager.enable_isolated_dispatch(
            iso.ScopedRngPatch(random_modules=[fake_game_module], numpy_modules=[fake_train_module])
        )

        with mock.patch.object(random, "getstate", side_effect=AssertionError("python getstate steady call")):
            with mock.patch.object(random, "setstate", side_effect=AssertionError("python setstate steady call")):
                with mock.patch.object(np.random, "get_state", side_effect=AssertionError("numpy get_state steady call")):
                    with mock.patch.object(np.random, "set_state", side_effect=AssertionError("numpy set_state steady call")):
                        value = manager.run(0, lambda: (fake_game_module.random.random(), float(fake_train_module.np.random.uniform())))

        control_py = random.Random()
        control_py.seed(111)
        self.assertEqual(construction_seen[0], control_py.random())
        self.assertEqual(value[0], control_py.random())

        control_np = np.random.RandomState()
        control_np.seed(222)
        self.assertEqual(construction_seen[1], float(control_np.random()))
        self.assertEqual(value[1], float(control_np.random()))

    def test_autoreset_sequence_uses_same_lane_bundle(self):
        manager = iso.IsolatedRngStates(1)
        fake_game_module = types.SimpleNamespace(random=random)
        manager.run(0, lambda: (random.seed(700), np.random.seed(800)))
        manager.enable_isolated_dispatch(iso.ScopedRngPatch(random_modules=[fake_game_module], numpy_modules=[]))

        observed = []
        observed.append(manager.run(0, lambda: fake_game_module.random.randint(0, 1_000_000)))
        observed.append(manager.run(0, lambda: fake_game_module.random.randint(0, 1_000_000)))

        control = random.Random()
        control.seed(700)
        self.assertEqual(observed, [control.randint(0, 1_000_000), control.randint(0, 1_000_000)])

    def test_lane_exception_marks_state_fatal_and_refuses_reuse(self):
        manager = iso.IsolatedRngStates(1)
        fake_game_module = types.SimpleNamespace(random=random)
        manager.enable_isolated_dispatch(iso.ScopedRngPatch(random_modules=[fake_game_module], numpy_modules=[]))

        with self.assertRaisesRegex(ValueError, "boom"):
            manager.run(0, lambda: (_ for _ in ()).throw(ValueError("boom")))
        with self.assertRaisesRegex(RuntimeError, "fatal RNG exception"):
            manager.run(0, lambda: fake_game_module.random.random())


class NavigationScopeTest(unittest.TestCase):
    def test_navigation_scope_requires_frozen_opponent_wrapper(self):
        class PlainEnv:
            env = None

        with self.assertRaisesRegex(RuntimeError, "FrozenOpponentWrapper"):
            iso.assert_navigation_constant_opponent_scope([PlainEnv()])

    def test_navigation_scope_accepts_frozen_opponent_wrapper_chain(self):
        class Raw:
            pass

        FrozenOpponentWrapper = type("FrozenOpponentWrapper", (), {})
        wrapper = FrozenOpponentWrapper()
        wrapper.env = Raw()
        monitor = types.SimpleNamespace(env=wrapper)

        iso.assert_navigation_constant_opponent_scope([monitor])


if gym is not None:

    class ToyProxyRngEnv(gym.Env):
        metadata = {"render_modes": []}
        render_mode = None

        def __init__(self, env_id: int) -> None:
            super().__init__()
            self.env_id = int(env_id)
            self.observation_space = spaces.Box(low=-1_000, high=1_000, shape=(3,), dtype=np.float32)
            self.action_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
            self.constructor_py = random.random()
            self.constructor_np = float(np.random.uniform())
            self.reset_count = 0
            self.step_count = 0
            self.truncate_after = 0

        def reset(self, *, seed=None, options=None):
            del options
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            self.step_count = 0
            self.reset_count += 1
            return self._obs(10), {
                "seed": seed,
                "constructor_py": self.constructor_py,
                "constructor_np": self.constructor_np,
                "reset_py": random.random(),
                "reset_np": float(np.random.uniform()),
                "reset_count": self.reset_count,
            }

        def step(self, action):
            del action
            self.step_count += 1
            truncated = bool(self.truncate_after and self.step_count >= self.truncate_after)
            return self._obs(50), 0.0, False, truncated, {
                "step_py": random.random(),
                "step_np": float(np.random.uniform()),
                "step_count": self.step_count,
            }

        def _obs(self, phase: int) -> np.ndarray:
            return np.asarray([self.env_id, phase + self.reset_count, self.step_count], dtype=np.float32)

        def set_truncate_after(self, value: int) -> None:
            self.truncate_after = int(value)


    def make_toy_proxy_rng_env(env_id: int):
        def _factory():
            random.seed(20_000 + env_id)
            np.random.seed(30_000 + env_id)
            return ToyProxyRngEnv(env_id)

        return _factory


class IsolatedRngGroupedWorkerTest(unittest.TestCase):
    @unittest.skipIf(gym is None, "gymnasium unavailable")
    def test_grouped_worker_uses_lane_rng_objects_for_reset_step_and_autoreset(self):
        from isolated_rng_vec_env import GroupedSubprocVecEnv

        env_fns = [make_toy_proxy_rng_env(index) for index in range(2)]
        grouped = GroupedSubprocVecEnv(env_fns, group_size=2, start_method="spawn", rng_scope="test")
        actions = np.zeros((2, 1), dtype=np.float32)
        try:
            grouped.seed(123)
            grouped.reset()
            control_py = [random.Random(123), random.Random(124)]
            control_np = [np.random.RandomState(123), np.random.RandomState(124)]
            reset_py = [rng.random() for rng in control_py]
            reset_np = [float(rng.uniform()) for rng in control_np]
            self.assertEqual([info["reset_py"] for info in grouped.reset_infos], reset_py)
            self.assertEqual([info["reset_np"] for info in grouped.reset_infos], reset_np)

            _obs, _rewards, dones, infos = grouped.step(actions)
            self.assertEqual(list(dones), [False, False])
            step1_py = [rng.random() for rng in control_py]
            step1_np = [float(rng.uniform()) for rng in control_np]
            self.assertEqual([info["step_py"] for info in infos], step1_py)
            self.assertEqual([info["step_np"] for info in infos], step1_np)

            grouped.env_method("set_truncate_after", 2)
            _obs, _rewards, dones, infos = grouped.step(actions)
            self.assertEqual(list(dones), [True, True])
            self.assertTrue(all("terminal_observation" in info for info in infos))
            step2_py = [rng.random() for rng in control_py]
            step2_np = [float(rng.uniform()) for rng in control_np]
            autoreset_py = [rng.random() for rng in control_py]
            autoreset_np = [float(rng.uniform()) for rng in control_np]
            self.assertEqual([info["step_py"] for info in infos], step2_py)
            self.assertEqual([info["step_np"] for info in infos], step2_np)
            self.assertEqual([info["reset_count"] for info in grouped.reset_infos], [2, 2])
            self.assertEqual([info["reset_py"] for info in grouped.reset_infos], autoreset_py)
            self.assertEqual([info["reset_np"] for info in grouped.reset_infos], autoreset_np)
        finally:
            grouped.close()


if __name__ == "__main__":
    unittest.main()
