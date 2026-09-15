import json
import os
import random
import tempfile
import unittest
import importlib.util
from functools import partial
from types import SimpleNamespace
from unittest import mock

import numpy as np

HAS_CLASSIC_DEPS = (
    importlib.util.find_spec("gymnasium") is not None
    and importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("stable_baselines3") is not None
)

if HAS_CLASSIC_DEPS:
    from gymnasium import spaces
    import train
    import environment.agent as agent_module


class _FakeEnv:
    def __init__(self):
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.obs_helper = object()
        self.act_helper = object()


class _FakeMonitor:
    pass


class _FakeSaveHandler:
    def __init__(self, history):
        self.history = list(history)

    def get_random_model_path(self):
        if not self.history:
            return None
        return random.choice(self.history)


def _loader_that_should_not_run(file_path, **kwargs):
    raise AssertionError(f"unexpected model load from {file_path}")


def _combat_phase_config(opponents):
    return SimpleNamespace(
        opponents=opponents,
        rewards={
            "distance": 0.0,
            "aggression": 0.0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "net_damage": 0.0,
            "win": 0.0,
            "knockout": 0.0,
            "edge_penalty": 0.0,
        },
    )


class _FakeSignal:
    def __init__(self):
        self.handlers = []

    def connect(self, handler):
        self.handlers.append(handler)


class _FakeRawWarehouseBrawl:
    def __init__(self, *args, **kwargs):
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.obs_helper = object()
        self.act_helper = object()
        self.win_signal = _FakeSignal()
        self.knockout_signal = _FakeSignal()

    def reset(self, seed=None, options=None):
        return {
            0: np.zeros(4, dtype=np.float32),
            1: np.ones(4, dtype=np.float32),
        }, {}

    def step(self, action):
        raise AssertionError("opponent audit should only reset, not step")


@unittest.skipUnless(HAS_CLASSIC_DEPS, "classic RL dependencies are not installed in this local environment")
class OpponentSelectionTelemetryTest(unittest.TestCase):
    def setUp(self):
        self.monitor_patch = mock.patch.object(agent_module, "Monitor", _FakeMonitor)
        self.monitor_patch.start()
        self.addCleanup(self.monitor_patch.stop)

    def test_unpatched_raw_env_constructor_metadata_is_marked_unpatched(self):
        info = train._raw_env_constructor_info()

        self.assertEqual(info["identity"], "environment.environment.WarehouseBrawl")
        self.assertFalse(info["patched"])

    def test_build_combat_opponents_without_telemetry_uses_plain_cfg(self):
        phase_config = SimpleNamespace(opponents={"random_agent": 0.9, "self_play": 0.1})

        cfg = train.build_combat_opponents(phase_config, device="cpu")

        self.assertIs(type(cfg), train.OpponentsCfg)
        self.assertNotIsInstance(cfg, train.TelemetryOpponentsCfg)
        self.assertIn("self_play", cfg.opponents)
        self.assertIsInstance(cfg.opponents["self_play"][1], train.SelfPlayRandom)

    def test_vectorized_save_handler_refreshes_archive_created_after_worker_construction(self):
        with tempfile.TemporaryDirectory(prefix="ai2-vectorized-save-handler-") as tmp:
            handler = train.VectorizedSaveHandler(
                agent=None,
                save_freq=1,
                max_saved=2,
                run_name="self_play_run",
                save_path=tmp,
                mode=train.SaveHandlerMode.FORCE,
            )

            run_dir = os.path.join(tmp, "self_play_run")
            os.makedirs(run_dir, exist_ok=True)
            first = os.path.join(run_dir, "model_000001_steps.zip")
            second = os.path.join(run_dir, "model_000002_steps.zip")
            with open(first, "wb") as f:
                f.write(b"first")
            with open(second, "wb") as f:
                f.write(b"second")

            self.assertFalse(handler.process())

            self.assertEqual(handler.history, [os.path.abspath(first), os.path.abspath(second)])
            self.assertIn(handler.get_random_model_path(), handler.history)

    def test_self_play_callback_saves_initial_interval_and_final_by_timesteps(self):
        class FakeSaveHandler:
            save_freq = 100

            def __init__(self):
                self.saved = []
                self.agent = None
                self.num_timesteps = 0

            def save_agent(self):
                self.saved.append(self.num_timesteps)

            def process(self):
                raise AssertionError("callback should use absolute timestep saves")

        handler = FakeSaveHandler()
        callback = train.SelfPlayCallback(handler)
        callback.model = object()

        callback.num_timesteps = 0
        callback._on_training_start()
        for step in (48, 96, 144, 199, 200):
            callback.num_timesteps = step
            self.assertTrue(callback._on_step())
        callback.num_timesteps = 240
        callback._on_training_end()

        self.assertEqual(handler.saved, [0, 144, 240])

    def test_records_self_play_empty_archive_constant_fallback(self):
        recorder = train.OpponentSelectionTelemetry(
            configured_probabilities={"self_play": 1.0},
            probability_source="unit.phase.opponents",
        )
        save_handler = _FakeSaveHandler(history=[])
        self_play = train.SelfPlayRandom(partial(_loader_that_should_not_run))
        cfg = train.TelemetryOpponentsCfg(
            opponents={"self_play": (1.0, self_play)},
            telemetry=recorder,
        )
        fake_env = _FakeEnv()
        cfg.env = fake_env
        self_play.env = fake_env
        self_play.save_handler = save_handler

        opponent = cfg.on_env_reset()

        self.assertIsInstance(opponent, train.ConstantAgent)
        self.assertEqual(len(recorder.records), 1)
        record = recorder.records[0]
        self.assertEqual(record["configured_key"], "self_play")
        self.assertTrue(record["self_play_requested"])
        self.assertEqual(record["archive_size"], 0)
        self.assertIsNone(record["checkpoint_path_sha256"])
        self.assertIsNone(record["checkpoint_content_sha256"])
        self.assertEqual(record["realized_class"], "ConstantAgent")
        self.assertEqual(record["fallback_reason"], "empty_archive")
        self.assertEqual(record["fallback_class"], "ConstantAgent")

    def test_audit_requires_minimum_selections_and_writes_exclusive_receipt(self):
        phase_config = _combat_phase_config({"self_play": 1.0})
        params = {
            "folders": {
                "parent_dir": tempfile.gettempdir(),
                "model_name": "ai2-opponent-telemetry-test",
            },
            "self_play_settings": {"save_freq": 1000, "max_saved_models": 2},
        }
        with tempfile.TemporaryDirectory(prefix="ai2-opponent-audit-") as tmp:
            receipt_path = os.path.join(tmp, "opponent-audit.json")

            with mock.patch.object(agent_module, "WarehouseBrawl", _FakeRawWarehouseBrawl):
                receipt = train.run_opponent_selection_audit(
                    phase_config=phase_config,
                    params=params,
                    output_path=receipt_path,
                    selections=1000,
                    seed=7,
                    device="cpu",
                )

            self.assertEqual(receipt["summary"]["total_selections"], 1000)
            self.assertEqual(receipt["summary"]["configured_counts"]["self_play"], 1000)
            self.assertEqual(receipt["summary"]["realized_counts"]["ConstantAgent"], 1000)
            self.assertEqual(receipt["summary"]["fallback_counts"]["empty_archive"], 1000)
            self.assertTrue(receipt["checks"]["minimum_selections"]["passed"])
            self.assertTrue(receipt["checks"]["configured_vs_realized_tolerance"]["passed"])
            self.assertEqual(receipt["checks"]["configured_vs_realized_tolerance"]["tolerance"], 0.03)
            self.assertEqual(receipt["audit"]["execution_path"], "make_combat_env.reset")
            self.assertTrue(receipt["audit"]["save_root_isolated"])
            self.assertNotEqual(receipt["audit"]["configured_model_root"], receipt["audit"]["isolated_model_root"])
            self.assertTrue(receipt["audit"]["raw_env_constructor"]["patched"])
            self.assertIn("asset-loading", receipt["audit"]["limitations"])
            with open(receipt_path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["summary"], receipt["summary"])
            self.assertEqual(on_disk["audit"], receipt["audit"])

            with self.assertRaises(FileExistsError):
                with mock.patch.object(agent_module, "WarehouseBrawl", _FakeRawWarehouseBrawl):
                    train.run_opponent_selection_audit(
                        phase_config=phase_config,
                        params=params,
                        output_path=receipt_path,
                        selections=1000,
                        seed=7,
                        device="cpu",
                    )

    def test_audit_isolates_save_root_and_preserves_configured_self_play_dir(self):
        phase_config = _combat_phase_config({"self_play": 1.0})
        with tempfile.TemporaryDirectory(prefix="ai2-real-opponent-audit-") as tmp:
            model_dir = os.path.join(tmp, "model")
            stale_self_play_dir = os.path.join(model_dir, "self_play_run")
            os.makedirs(stale_self_play_dir)
            sentinel_path = os.path.join(stale_self_play_dir, "rl_model_123_steps.zip")
            with open(sentinel_path, "wb") as f:
                f.write(b"not a real checkpoint")
            params = {
                "folders": {"parent_dir": tmp, "model_name": ""},
                "self_play_settings": {"save_freq": 1000, "max_saved_models": 2},
                "environment_settings": {"resolution": "LOW"},
                "opponent_history": {"enabled": False},
            }

            with mock.patch.object(agent_module, "WarehouseBrawl", _FakeRawWarehouseBrawl), \
                    mock.patch("builtins.input", side_effect=AssertionError("audit must not prompt")):
                receipt = train.run_opponent_selection_audit(
                    phase_config=phase_config,
                    params=params,
                    output_path=os.path.join(tmp, "receipt.json"),
                    selections=1000,
                    seed=13,
                    device="cpu",
                )

            self.assertEqual(receipt["summary"]["configured_counts"]["self_play"], 1000)
            self.assertEqual(receipt["summary"]["realized_counts"]["ConstantAgent"], 1000)
            self.assertEqual(receipt["summary"]["fallback_counts"]["empty_archive"], 1000)
            self.assertTrue(all(record["archive_size"] == 0 for record in receipt["records"]))
            self.assertTrue(all(record["checkpoint_content_sha256"] is None for record in receipt["records"]))
            self.assertTrue(os.path.exists(sentinel_path))
            with open(sentinel_path, "rb") as f:
                self.assertEqual(f.read(), b"not a real checkpoint")
            self.assertTrue(receipt["audit"]["save_root_isolated"])
            self.assertEqual(receipt["audit"]["configured_model_root"], os.path.abspath(model_dir))
            self.assertNotEqual(receipt["audit"]["isolated_model_root"], os.path.abspath(model_dir))
            self.assertTrue(receipt["audit"]["raw_env_constructor"]["patched"])

    def test_audit_restores_global_reward_manager_when_starting_none(self):
        phase_config = _combat_phase_config({"self_play": 1.0})
        params = {
            "folders": {"parent_dir": tempfile.gettempdir(), "model_name": "ai2-global-none-test"},
            "self_play_settings": {"save_freq": 1000, "max_saved_models": 2},
            "environment_settings": {"resolution": "LOW"},
            "opponent_history": {"enabled": False},
        }
        train.GLOBAL_REWARD_MANAGER = None
        with tempfile.TemporaryDirectory(prefix="ai2-global-none-audit-") as tmp:
            with mock.patch.object(agent_module, "WarehouseBrawl", _FakeRawWarehouseBrawl):
                train.run_opponent_selection_audit(
                    phase_config=phase_config,
                    params=params,
                    output_path=os.path.join(tmp, "receipt.json"),
                    selections=1000,
                    seed=23,
                    device="cpu",
                )

        self.assertIsNone(train.GLOBAL_REWARD_MANAGER)

    def test_audit_restores_preexisting_global_and_rng_after_success(self):
        phase_config = _combat_phase_config({"self_play": 1.0})
        params = {
            "folders": {"parent_dir": tempfile.gettempdir(), "model_name": "ai2-global-sentinel-test"},
            "self_play_settings": {"save_freq": 1000, "max_saved_models": 2},
            "environment_settings": {"resolution": "LOW"},
            "opponent_history": {"enabled": False},
        }
        sentinel_manager = object()
        train.GLOBAL_REWARD_MANAGER = sentinel_manager
        random.seed(987654)
        before_state = random.getstate()

        with tempfile.TemporaryDirectory(prefix="ai2-global-sentinel-audit-") as tmp:
            with mock.patch.object(agent_module, "WarehouseBrawl", _FakeRawWarehouseBrawl):
                train.run_opponent_selection_audit(
                    phase_config=phase_config,
                    params=params,
                    output_path=os.path.join(tmp, "receipt.json"),
                    selections=1000,
                    seed=29,
                    device="cpu",
                )

        self.assertIs(train.GLOBAL_REWARD_MANAGER, sentinel_manager)
        self.assertEqual(random.getstate(), before_state)

    def test_audit_uses_fresh_manager_then_restores_preexisting_global(self):
        phase_config = _combat_phase_config({"self_play": 1.0})
        params = {
            "folders": {"parent_dir": tempfile.gettempdir(), "model_name": "ai2-fresh-manager-test"},
            "self_play_settings": {"save_freq": 1000, "max_saved_models": 2},
            "environment_settings": {"resolution": "LOW"},
            "opponent_history": {"enabled": False},
        }
        sentinel_manager = object()
        train.GLOBAL_REWARD_MANAGER = sentinel_manager
        original_manager_cls = train.PhaseAwareRewardManager

        with tempfile.TemporaryDirectory(prefix="ai2-fresh-manager-audit-") as tmp:
            with mock.patch.object(agent_module, "WarehouseBrawl", _FakeRawWarehouseBrawl), \
                    mock.patch.object(train, "PhaseAwareRewardManager", wraps=original_manager_cls) as manager_cls:
                train.run_opponent_selection_audit(
                    phase_config=phase_config,
                    params=params,
                    output_path=os.path.join(tmp, "receipt.json"),
                    selections=1000,
                    seed=37,
                    device="cpu",
                )

        self.assertEqual(manager_cls.call_count, 1)
        self.assertIs(train.GLOBAL_REWARD_MANAGER, sentinel_manager)

    def test_audit_restores_global_and_rng_after_construction_failure(self):
        phase_config = _combat_phase_config({"self_play": 1.0})
        params = {
            "folders": {"parent_dir": tempfile.gettempdir(), "model_name": "ai2-construction-failure-test"},
            "self_play_settings": {"save_freq": 1000, "max_saved_models": 2},
            "environment_settings": {"resolution": "LOW"},
            "opponent_history": {"enabled": False},
        }
        sentinel_manager = object()
        train.GLOBAL_REWARD_MANAGER = sentinel_manager
        random.seed(24680)
        before_state = random.getstate()

        with tempfile.TemporaryDirectory(prefix="ai2-construction-failure-audit-") as tmp:
            with mock.patch.object(train, "make_combat_env", side_effect=RuntimeError("forced construction failure")):
                with self.assertRaisesRegex(RuntimeError, "forced construction failure"):
                    train.run_opponent_selection_audit(
                        phase_config=phase_config,
                        params=params,
                        output_path=os.path.join(tmp, "receipt.json"),
                        selections=1000,
                        seed=31,
                        device="cpu",
                    )

        self.assertIs(train.GLOBAL_REWARD_MANAGER, sentinel_manager)
        self.assertEqual(random.getstate(), before_state)

    def test_audit_fails_closed_below_1000_selections(self):
        phase_config = SimpleNamespace(opponents={"random_agent": 1.0})
        params = {
            "folders": {"parent_dir": tempfile.gettempdir(), "model_name": "ai2-opponent-telemetry-test"},
            "self_play_settings": {"save_freq": 1000, "max_saved_models": 2},
        }
        with tempfile.TemporaryDirectory(prefix="ai2-opponent-audit-small-") as tmp:
            with self.assertRaises(ValueError):
                train.run_opponent_selection_audit(
                    phase_config=phase_config,
                    params=params,
                    output_path=os.path.join(tmp, "opponent-audit.json"),
                    selections=999,
                    seed=11,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
