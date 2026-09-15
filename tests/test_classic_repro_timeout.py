"""Classic combat reproducibility and time-limit contracts."""

from __future__ import annotations

import copy
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "UTMIST-AI2-main"))


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


def purge_repo_and_stubbed_imports():
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


def import_real_stack():
    purge_repo_and_stubbed_imports()
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
        self.assets_guard = install_test_assets_guard()
        self.np, self.train, self.OpponentsCfg, self.SelfPlayWarehouseBrawl, self.WarehouseBrawl = import_real_stack()
        self.tmp = tempfile.TemporaryDirectory(prefix="ai2-classic-repro-")

    def tearDown(self):
        self.tmp.cleanup()
        if self.assets_guard is not None:
            self.assets_guard.rmdir()

    def test_combat_factory_applies_configured_90_second_time_limit_to_real_raw_env(self):
        params = make_params(self.tmp.name)
        phase = self.train.PhaseConfig(params, 1)
        env = self.train.make_combat_env(phase, params, device="cpu")
        try:
            self_play_env = unwrap_self_play_env(env)
            raw = self_play_env.raw_env
            self.assertEqual(raw.fps, 30)
            self.assertEqual(raw.max_timesteps, 2700)
            obs, _ = env.reset()
            raw.steps = raw.max_timesteps - 1
            action = self.np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

            _, _, _, truncated, _ = env.step(action)

            self.assertTrue(truncated)
            self.assertEqual(raw.steps, 2700)
        finally:
            env.close()

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


if __name__ == "__main__":
    unittest.main()
