"""Real-dependency smoke tests for classic combat reward wiring."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import MethodType


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="ai2-mpl-"))

ROOT = Path(__file__).resolve().parents[1]
_PATH_ENTRIES_ADDED = []


def add_test_path(path: Path) -> None:
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
        _PATH_ENTRIES_ADDED.append(entry)


def restore_test_paths() -> None:
    for entry in reversed(_PATH_ENTRIES_ADDED):
        try:
            sys.path.remove(entry)
        except ValueError:
            pass
    _PATH_ENTRIES_ADDED.clear()


add_test_path(ROOT)
add_test_path(ROOT / "UTMIST-AI2-main")

_MISSING = object()


REAL_IMPORT_PREFIXES = (
    "train",
    "environment",
    "src",
    "gymnasium",
    "stable_baselines3",
    "sb3_contrib",
    "torch",
    "gdown",
    "pygame",
    "pymunk",
    "cv2",
    "skvideo",
    "skimage",
    "IPython",
)


def purge_stubbed_imports():
    """Remove lightweight unit-test stubs before importing the real simulator stack."""
    for name in list(sys.modules):
        module = sys.modules[name]
        module_file = getattr(module, "__file__", None)
        is_named_real_import = name in REAL_IMPORT_PREFIXES or any(
            name.startswith(f"{prefix}.") for prefix in REAL_IMPORT_PREFIXES
        )
        is_unit_test_dummy = module.__class__.__name__ == "DummyModule" or (
            module_file is not None and not isinstance(module_file, (str, bytes, os.PathLike))
        )
        root_name = name.split(".", 1)[0]
        root_module = sys.modules.get(root_name)
        root_file = getattr(root_module, "__file__", None)
        is_real_torch_child = (
            root_name == "torch"
            and name != "torch"
            and isinstance(root_file, (str, bytes, os.PathLike))
        )
        is_synthetic_module = module_file is None and is_named_real_import and not is_real_torch_child
        is_repo_module_loaded_under_stubs = name == "train" or name.startswith("environment")
        if is_unit_test_dummy or is_synthetic_module or is_repo_module_loaded_under_stubs:
            sys.modules.pop(name, None)


def capture_repo_game_modules():
    return {
        name: sys.modules.get(name, _MISSING)
        for name in list(sys.modules)
        if name == "train" or name == "environment" or name.startswith("environment.")
    }


def repo_game_modules_loaded_since(snapshot):
    loaded = {}
    for name in list(sys.modules):
        if not (name == "train" or name == "environment" or name.startswith("environment.")):
            continue
        module = sys.modules[name]
        if snapshot.get(name, _MISSING) is module:
            continue
        loaded[name] = module
    return loaded


def restore_repo_game_modules(snapshot, loaded):
    for name, module in sorted(loaded.items(), key=lambda item: item[0].count("."), reverse=True):
        if sys.modules.get(name) is not module:
            continue
        previous = snapshot.get(name, _MISSING)
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def install_test_assets_guard():
    assets_dir = ROOT / "assets"
    if assets_dir.exists():
        return None
    assets_dir.mkdir()
    return assets_dir


def import_real_stack():
    purge_stubbed_imports()
    try:
        import numpy as np
        import torch
        from stable_baselines3.common.vec_env import SubprocVecEnv
        import train
        from environment.agent import SelfPlayWarehouseBrawl
        from environment.environment import WarehouseBrawl
    except Exception as exc:  # pragma: no cover - local lightweight env path
        raise unittest.SkipTest(f"full simulator dependency stack unavailable: {exc}") from exc

    if not getattr(torch, "__file__", None):
        raise unittest.SkipTest("real torch is unavailable; refusing stubbed integration")

    return np, SubprocVecEnv, train, SelfPlayWarehouseBrawl, WarehouseBrawl


class RewardDiagnosticsWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            return close()
        return None

    def _self_play_env(self):
        env = self.env
        while env is not None:
            if hasattr(env, "raw_env") and hasattr(env, "reward_manager"):
                return env
            env = getattr(env, "env", None)
        raise AssertionError("SelfPlayWarehouseBrawl not found in wrapper chain")

    def _phase_wrapper(self):
        env = self.env
        while env is not None:
            if env.__class__.__name__ == "RewardEpisodeResetWrapper":
                return env
            env = getattr(env, "env", None)
        raise AssertionError("RewardEpisodeResetWrapper not found in wrapper chain")

    def _wrapper_names(self):
        names = []
        env = self.env
        while env is not None:
            names.append(env.__class__.__name__)
            env = getattr(env, "env", None)
        return names

    def snapshot(self):
        self_play_env = self._self_play_env()
        phase_wrapper = self._phase_wrapper()
        raw_env = self_play_env.raw_env
        manager = phase_wrapper.reward_manager
        return {
            "pid": os.getpid(),
            "wrapper_names": self._wrapper_names(),
            "raw_env_type": raw_env.__class__.__name__,
            "win_handlers": len(raw_env.win_signal._handlers),
            "knockout_handlers": len(raw_env.knockout_signal._handlers),
            "manager_id": id(manager),
            "win_weight": manager.weights.get("win", 0.0),
            "episode_damage_dealt": manager.episode_damage_dealt,
            "lifetime_damage_dealt": manager.lifetime_damage_dealt,
        }

    def emit_win_and_process(self, agent="player"):
        self_play_env = self._self_play_env()
        raw_env = self_play_env.raw_env
        raw_env.win_signal.emit(agent=agent)
        return self_play_env.reward_manager.process(raw_env, 1 / 30.0)

    def emit_knockout_and_process(self, agent="player"):
        self_play_env = self._self_play_env()
        raw_env = self_play_env.raw_env
        raw_env.knockout_signal.emit(agent=agent)
        return self_play_env.reward_manager.process(raw_env, 1 / 30.0)


def attach_reward_diagnostics(env):
    diagnostics = RewardDiagnosticsWrapper(env)
    env._self_play_env = MethodType(lambda self: diagnostics._self_play_env(), env)
    env._phase_wrapper = MethodType(lambda self: diagnostics._phase_wrapper(), env)
    env._wrapper_names = MethodType(lambda self: diagnostics._wrapper_names(), env)
    env.snapshot = MethodType(lambda self: diagnostics.snapshot(), env)
    env.emit_win_and_process = MethodType(
        lambda self, agent="player": diagnostics.emit_win_and_process(agent),
        env,
    )
    env.emit_knockout_and_process = MethodType(
        lambda self, agent="player": diagnostics.emit_knockout_and_process(agent),
        env,
    )
    return env


def phase_config(train, rewards: dict[str, float]):
    return train.PhaseConfig(
        {
            "phases": {
                1: {
                    "name": "integration combat",
                    "type": "combat",
                    "rewards": rewards,
                    "opponents": {"random_agent": 1.0},
                }
            }
        },
        1,
    )


def combat_params():
    return {
        "self_play_settings": {"save_freq": 1, "max_saved_models": 1},
        "folders": {"parent_dir": "/tmp", "model_name": "ai2-test"},
        "environment_settings": {"resolution": "LOW"},
        "opponent_history": {"enabled": False},
    }


class ClassicRewardImportIsolationTest(unittest.TestCase):
    def test_purge_preserves_real_torch_c_extension_children(self):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - local lightweight env path
            raise unittest.SkipTest(f"real torch unavailable: {exc}") from exc

        if not getattr(torch, "__file__", None):
            raise unittest.SkipTest("real torch unavailable")
        self.assertIs(sys.modules.get("torch"), torch)
        self.assertIn("torch._C._profiler", sys.modules)

        purge_stubbed_imports()

        self.assertIs(sys.modules.get("torch"), torch)
        self.assertIn("torch._C._profiler", sys.modules)
        self.assertIsNotNone(getattr(sys.modules["torch._C._profiler"], "__name__", None))


class ClassicRewardIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_snapshot = capture_repo_game_modules()
        cls.loaded_repo_modules = {}
        cls.created_assets_dir = install_test_assets_guard()
        try:
            (
                cls.np,
                cls.SubprocVecEnv,
                cls.train,
                cls.SelfPlayWarehouseBrawl,
                cls.WarehouseBrawl,
            ) = import_real_stack()
            cls.loaded_repo_modules = repo_game_modules_loaded_since(cls.module_snapshot)
        except Exception:
            cls.loaded_repo_modules = repo_game_modules_loaded_since(cls.module_snapshot)
            restore_repo_game_modules(cls.module_snapshot, cls.loaded_repo_modules)
            restore_test_paths()
            created_assets_dir = getattr(cls, "created_assets_dir", None)
            if created_assets_dir is not None:
                try:
                    created_assets_dir.rmdir()
                except OSError:
                    pass
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            restore_repo_game_modules(
                getattr(cls, "module_snapshot", {}),
                getattr(cls, "loaded_repo_modules", {}),
            )
        finally:
            restore_test_paths()
            created_assets_dir = getattr(cls, "created_assets_dir", None)
            if created_assets_dir is not None:
                try:
                    created_assets_dir.rmdir()
                except OSError:
                    pass

    def make_env(self, rewards):
        env = self.train.make_combat_env(phase_config(self.train, rewards), combat_params(), device="cpu")
        return attach_reward_diagnostics(env)

    def test_actual_warehouse_wrapper_ordering_and_single_signal_handlers(self):
        env = self.make_env({"win": 4.0, "knockout": 2.0})
        try:
            self.assertIsInstance(env._self_play_env(), self.SelfPlayWarehouseBrawl)
            self.assertIsInstance(env._self_play_env().raw_env, self.WarehouseBrawl)
            before = env.snapshot()
            self.assertEqual(before["wrapper_names"][:3], ["RewardEpisodeResetWrapper", "DamageTrackingWrapper", "Float32Wrapper"])
            self.assertEqual(before["raw_env_type"], "WarehouseBrawl")
            self.assertEqual(before["win_handlers"], 1)
            self.assertEqual(before["knockout_handlers"], 1)

            env.reset()
            env.reset()
            after = env.snapshot()
            self.assertEqual(after["win_handlers"], 1)
            self.assertEqual(after["knockout_handlers"], 1)
            self.assertEqual(env.emit_win_and_process("player"), 4.0)
            self.assertEqual(env.emit_win_and_process("opponent"), -4.0)
            self.assertEqual(env.emit_knockout_and_process("player"), -2.0)
            self.assertEqual(env.emit_knockout_and_process("opponent"), 2.0)
        finally:
            env.close()

    def test_subproc_two_env_spawn_keeps_signal_and_reward_state_isolated(self):
        def make_first():
            return self.make_env({"win": 2.0, "knockout": 1.0})

        def make_second():
            return self.make_env({"win": 7.0, "knockout": 1.0})

        vec_env = self.SubprocVecEnv([make_first, make_second], start_method="fork")
        try:
            vec_env.reset()
            before = vec_env.env_method("snapshot")
            self.assertEqual([item["win_handlers"] for item in before], [1, 1])
            self.assertEqual([item["knockout_handlers"] for item in before], [1, 1])
            self.assertEqual([item["win_weight"] for item in before], [2.0, 7.0])
            self.assertEqual(len({item["pid"] for item in before}), 2)

            vec_env.env_method("reset")
            vec_env.env_method("reset")
            after_reset = vec_env.env_method("snapshot")
            self.assertEqual([item["win_handlers"] for item in after_reset], [1, 1])
            self.assertEqual([item["knockout_handlers"] for item in after_reset], [1, 1])

            rewards = vec_env.env_method("emit_win_and_process", "player")
            self.assertEqual(rewards, [2.0, 7.0])
            knockout_rewards = vec_env.env_method("emit_knockout_and_process", "player")
            self.assertEqual(knockout_rewards, [-1.0, -1.0])
            after_reward = vec_env.env_method("snapshot")
            self.assertEqual([item["win_weight"] for item in after_reward], [2.0, 7.0])
            self.assertNotEqual(after_reward[0]["manager_id"], after_reward[1]["manager_id"])
        finally:
            vec_env.close()


class ClassicRewardPostTeardownImportIsolationTest(unittest.TestCase):
    def test_different_source_environment_imports_after_integration_teardown(self):
        snapshot = capture_repo_game_modules()
        with tempfile.TemporaryDirectory(prefix="ai2-env-sentinel-") as directory:
            package = Path(directory) / "environment"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "environment.py").write_text(
                "SENTINEL_SOURCE = 'post-classic-integration-teardown'\n"
                "class WarehouseBrawl:\n"
                "    pass\n"
            )
            sys.path.insert(0, directory)
            try:
                for name in list(sys.modules):
                    if name == "environment" or name.startswith("environment."):
                        sys.modules.pop(name, None)
                import importlib

                importlib.invalidate_caches()
                sentinel = importlib.import_module("environment.environment")
                self.assertEqual(sentinel.SENTINEL_SOURCE, "post-classic-integration-teardown")
                self.assertTrue(Path(sentinel.__file__).resolve().is_relative_to(package.resolve()))
            finally:
                try:
                    sys.path.remove(directory)
                except ValueError:
                    pass
                restore_repo_game_modules(snapshot, capture_repo_game_modules())


if __name__ == "__main__":
    unittest.main()
