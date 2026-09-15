"""Classic combat reproducibility and time-limit contracts."""

from __future__ import annotations

import copy
import importlib
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="ai2-mpl-"))

ROOT = Path(__file__).resolve().parents[1]
_PATH_ENTRIES_ADDED = []
_MISSING = object()


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


def clear_repo_game_modules_for_import():
    for name in sorted(
        [
            name for name in list(sys.modules)
            if name == "train" or name == "environment" or name.startswith("environment.")
        ],
        key=lambda value: value.count("."),
        reverse=True,
    ):
        sys.modules.pop(name, None)


def import_real_stack():
    try:
        import numpy as np
        import train
        from environment.agent import OpponentsCfg, SelfPlayWarehouseBrawl
        from environment.environment import WarehouseBrawl
    except Exception as exc:  # pragma: no cover - local lightweight env path
        raise unittest.SkipTest(f"full simulator dependency stack unavailable: {exc}") from exc

    return np, train, OpponentsCfg, SelfPlayWarehouseBrawl, WarehouseBrawl


def install_test_assets_guard():
    assets_dir = ROOT / "assets"
    if assets_dir.exists():
        return None
    assets_dir.mkdir()
    return assets_dir


def make_params(tmp_dir: str) -> dict:
    return {
        "folders": {"parent_dir": tmp_dir, "model_name": "classic_repro_test"},
        "self_play_settings": {"save_freq": 10_000, "max_saved_models": 2},
        "environment_settings": {
            "max_timesteps": 2700,
            "resolution": "LOW",
            "n_envs": 1,
        },
        "phases": {
            1: {
                "name": "Classic repro phase",
                "type": "combat",
                "rewards": {
                    "distance": 0.0,
                    "aggression": 0.0,
                    "damage_dealt": 0.0,
                    "damage_taken": 0.0,
                    "net_damage": 0.0,
                    "win": 0.0,
                    "knockout": 0.0,
                    "edge_penalty": 0.0,
                },
                "opponents": {"constant_agent": 1.0},
            }
        },
    }


def unwrap_self_play_env(env):
    cursor = env
    while cursor is not None:
        if hasattr(cursor, "raw_env") and hasattr(cursor, "opponent_cfg"):
            return cursor
        cursor = getattr(cursor, "env", None)
    raise AssertionError("SelfPlayWarehouseBrawl not found in wrapper chain")


def raw_signature(raw_env) -> tuple:
    return (
        raw_env.players[0].body.position.x,
        raw_env.players[1].body.position.x,
        tuple(round(spawner.world_pos[0], 6) for spawner in raw_env.weapon_spawners),
        tuple(spawner.cooldown_frames for spawner in raw_env.weapon_spawners),
    )


def null_reward_manager(train_module):
    def null_reward(_env, **_kwargs):
        return 0.0

    return train_module.RewardManager(
        reward_functions={"null": train_module.RewTerm(func=null_reward, weight=0.0)},
        signal_subscriptions={},
    )


class ClassicReproTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.module_snapshot = capture_repo_game_modules()
        self.loaded_repo_modules = {}
        add_test_path(ROOT)
        add_test_path(ROOT / "UTMIST-AI2-main")
        self.assets_guard = install_test_assets_guard()
        try:
            clear_repo_game_modules_for_import()
            self.np, self.train, self.OpponentsCfg, self.SelfPlayWarehouseBrawl, self.WarehouseBrawl = import_real_stack()
            self.loaded_repo_modules = repo_game_modules_loaded_since(self.module_snapshot)
            self.tmp = tempfile.TemporaryDirectory(prefix="ai2-classic-repro-")
        except Exception:
            self.loaded_repo_modules = repo_game_modules_loaded_since(self.module_snapshot)
            self._restore_imports_and_assets()
            raise

    def tearDown(self):
        tmp = getattr(self, "tmp", None)
        if tmp is not None:
            tmp.cleanup()
        self._restore_imports_and_assets()

    def _restore_imports_and_assets(self):
        restore_repo_game_modules(
            getattr(self, "module_snapshot", {}),
            getattr(self, "loaded_repo_modules", {}),
        )
        restore_test_paths()
        if self.assets_guard is not None:
            try:
                self.assets_guard.rmdir()
            except OSError:
                pass

    def test_combat_factory_applies_configured_90_second_time_limit_and_boundary(self):
        params = make_params(self.tmp.name)
        phase = self.train.PhaseConfig(params, 1)
        env = self.train.make_combat_env(phase, params, device="cpu")
        try:
            self_play_env = unwrap_self_play_env(env)
            raw = self_play_env.raw_env
            self.assertEqual(raw.fps, 30)
            self.assertEqual(raw.max_timesteps, 2700)
            env.reset()
            action = self.np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

            raw.steps = raw.max_timesteps - 2
            _, _, _, truncated, _ = env.step(action)
            self.assertFalse(truncated)
            self.assertEqual(raw.steps, 2699)

            _, _, _, truncated, _ = env.step(action)
            self.assertTrue(truncated)
            self.assertEqual(raw.steps, 2700)
        finally:
            env.close()

    def test_configured_max_timesteps_fails_closed_for_invalid_horizons(self):
        params = make_params(self.tmp.name)

        bad_settings = [
            {"time_limit_seconds": 0},
            {"time_limit_seconds": -1},
            {"time_limit_seconds": float("nan")},
            {"time_limit_seconds": float("inf")},
            {"max_timesteps": 0},
            {"max_timesteps": -10},
            {"time_limit_seconds": 90, "max_timesteps": 2699},
        ]
        for settings in bad_settings:
            with self.subTest(settings=settings):
                candidate = copy.deepcopy(params)
                candidate["environment_settings"] = settings
                with self.assertRaises(ValueError):
                    self.train._configured_max_timesteps(candidate, fps=30)

        seconds_only = copy.deepcopy(params)
        seconds_only["environment_settings"] = {"time_limit_seconds": 1.5}
        self.assertEqual(self.train._configured_max_timesteps(seconds_only, fps=30), 45)

    def test_raw_reset_seed_replays_the_same_initial_randomized_setup(self):
        env = self.WarehouseBrawl(train_mode=True)
        try:
            env.reset(seed=123)
            first = raw_signature(env)
            random.random()
            self.np.random.random()

            env.reset(seed=123)
            second = raw_signature(env)

            env.reset(seed=124)
            third = raw_signature(env)

            self.assertEqual(second, first)
            self.assertNotEqual(third, first)
        finally:
            env.close()

    def test_wrapper_reset_forwards_seed_without_forcing_seed_none_to_restart(self):
        env = self.SelfPlayWarehouseBrawl(
            reward_manager=null_reward_manager(self.train),
            opponent_cfg=self.OpponentsCfg(opponents={"constant_agent": (1.0, self.train.partial(self.train.ConstantAgent))}),
        )
        try:
            env.reset(seed=55)
            first = raw_signature(env.raw_env)
            env.reset(seed=55)
            second = raw_signature(env.raw_env)
            env.reset(seed=56)
            third = raw_signature(env.raw_env)
            env.reset()
            none_first = raw_signature(env.raw_env)
            env.reset()
            none_second = raw_signature(env.raw_env)

            self.assertEqual(second, first)
            self.assertNotEqual(third, first)
            self.assertNotEqual(none_second, none_first)
        finally:
            env.close()

    def test_selfplay_wrapper_close_delegates_to_raw_env(self):
        env = self.SelfPlayWarehouseBrawl(
            reward_manager=null_reward_manager(self.train),
            opponent_cfg=self.OpponentsCfg(opponents={"constant_agent": (1.0, self.train.partial(self.train.ConstantAgent))}),
        )
        with mock.patch.object(env.raw_env, "close", autospec=True) as close:
            env.close()

        close.assert_called_once_with()

    def test_default_opponent_config_is_not_shared_between_instances(self):
        first = self.SelfPlayWarehouseBrawl(reward_manager=null_reward_manager(self.train))
        second = self.SelfPlayWarehouseBrawl(reward_manager=null_reward_manager(self.train))
        try:
            self.assertIsNot(first.opponent_cfg, second.opponent_cfg)
            first.opponent_cfg.opponents["extra"] = (0.0, self.train.partial(self.train.ConstantAgent))
            self.assertNotIn("extra", second.opponent_cfg.opponents)
        finally:
            first.close()
            second.close()

    def test_checked_in_config_declares_90_seconds_as_2700_ticks_at_30fps(self):
        import yaml

        with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)

        self.assertEqual(params["environment_settings"]["max_timesteps"], 2700)
        self.assertEqual(params["environment_settings"]["time_limit_seconds"], 90)


class ClassicReproImportIsolationTest(unittest.TestCase):
    def test_different_source_environment_imports_after_repro_tests_restore(self):
        snapshot = capture_repo_game_modules()
        with tempfile.TemporaryDirectory(prefix="ai2-repro-env-sentinel-") as directory:
            package = Path(directory) / "environment"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "environment.py").write_text(
                "SENTINEL_SOURCE = 'post-classic-repro-teardown'\n"
                "class WarehouseBrawl:\n"
                "    pass\n"
            )
            sys.path.insert(0, directory)
            try:
                for name in list(sys.modules):
                    if name == "environment" or name.startswith("environment."):
                        sys.modules.pop(name, None)
                importlib.invalidate_caches()
                sentinel = importlib.import_module("environment.environment")
                self.assertEqual(sentinel.SENTINEL_SOURCE, "post-classic-repro-teardown")
                self.assertTrue(Path(sentinel.__file__).resolve().is_relative_to(package.resolve()))
            finally:
                try:
                    sys.path.remove(directory)
                except ValueError:
                    pass
                restore_repo_game_modules(snapshot, capture_repo_game_modules())


if __name__ == "__main__":
    unittest.main()
