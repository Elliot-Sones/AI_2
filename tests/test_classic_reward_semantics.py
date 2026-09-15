"""Regression tests for classic combat reward plumbing."""

from __future__ import annotations

import importlib
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest import mock


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="ai2-mpl-"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "UTMIST-AI2-main"))


class FakeTensor:
    def __init__(self, value):
        if isinstance(value, FakeTensor):
            value = value.value
        if isinstance(value, list):
            self.value = [float(item) for item in value]
        else:
            self.value = float(value)

    def __mul__(self, other):
        if isinstance(self.value, list):
            return FakeTensor([item * other for item in self.value])
        return FakeTensor(self.value * other)

    __rmul__ = __mul__

    def __add__(self, other):
        other_value = other.value if isinstance(other, FakeTensor) else other
        if isinstance(self.value, list):
            if isinstance(other_value, list):
                return FakeTensor([left + right for left, right in zip(self.value, other_value)])
            return FakeTensor([left + other_value for left in self.value])
        if isinstance(other_value, list):
            return FakeTensor([self.value + right for right in other_value])
        return FakeTensor(self.value + other_value)

    __radd__ = __add__

    def __iadd__(self, other):
        self.value = (self + other).value
        return self

    def __getitem__(self, index):
        return FakeTensor(self.value[index])

    def __setitem__(self, index, value):
        self.value[index] = float(value)

    def __neg__(self):
        if isinstance(self.value, list):
            return FakeTensor([-item for item in self.value])
        return FakeTensor(-self.value)

    def item(self):
        if isinstance(self.value, list):
            return self.value[0]
        return self.value

    def numel(self):
        return len(self.value) if isinstance(self.value, list) else 1

    def __eq__(self, other):
        other_value = other.value if isinstance(other, FakeTensor) else other
        return self.value == other_value

    def __repr__(self):
        return f"FakeTensor({self.value!r})"


class Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Dummy()

    def __getattr__(self, name):
        return Dummy()

    def __iter__(self):
        return iter(())


class DummyModule(ModuleType):
    def __getattr__(self, name):
        value = Dummy
        setattr(self, name, value)
        return value


def install_import_stubs() -> None:
    if "yaml" not in sys.modules:
        yaml_mod = ModuleType("yaml")
        yaml_mod.safe_load = lambda stream: {}
        sys.modules["yaml"] = yaml_mod

    if "torch" not in sys.modules:
        torch_mod = ModuleType("torch")
        torch_mod.Tensor = FakeTensor
        torch_mod.tensor = lambda value: FakeTensor(value)
        torch_mod.zeros = lambda size: FakeTensor([0.0] * size)
        torch_mod.equal = lambda left, right: left == right
        torch_mod.cuda = SimpleNamespace(is_available=lambda: False)
        torch_mod.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
        torch_mod.nn = SimpleNamespace(Module=object)
        sys.modules["torch"] = torch_mod
        sys.modules["torch.nn"] = torch_mod.nn

    if "gymnasium" not in sys.modules:
        gym_mod = ModuleType("gymnasium")
        gym_mod.Env = type("Env", (), {})
        gym_mod.Wrapper = type("Wrapper", (), {"__init__": lambda self, env: setattr(self, "env", env)})
        gym_mod.ObservationWrapper = gym_mod.Wrapper
        gym_mod.spaces = SimpleNamespace(
            Space=type("Space", (), {}),
            Box=lambda *args, **kwargs: SimpleNamespace(args=args, **kwargs),
        )
        sys.modules["gymnasium"] = gym_mod
        sys.modules["gymnasium.spaces"] = gym_mod.spaces

    if "stable_baselines3" not in sys.modules:
        sb3_mod = ModuleType("stable_baselines3")
        sb3_mod.PPO = SimpleNamespace(load=lambda *args, **kwargs: None)
        sb3_mod.A2C = SimpleNamespace(load=lambda *args, **kwargs: None)
        sys.modules["stable_baselines3"] = sb3_mod
        common_mod = ModuleType("stable_baselines3.common")
        sys.modules["stable_baselines3.common"] = common_mod

        base_class_mod = ModuleType("stable_baselines3.common.base_class")
        base_class_mod.BaseAlgorithm = object
        sys.modules["stable_baselines3.common.base_class"] = base_class_mod

        callbacks_mod = ModuleType("stable_baselines3.common.callbacks")
        callbacks_mod.CheckpointCallback = object
        callbacks_mod.BaseCallback = object
        sys.modules["stable_baselines3.common.callbacks"] = callbacks_mod

        env_util_mod = ModuleType("stable_baselines3.common.env_util")
        env_util_mod.make_vec_env = lambda *args, **kwargs: None
        sys.modules["stable_baselines3.common.env_util"] = env_util_mod

        vec_env_mod = ModuleType("stable_baselines3.common.vec_env")
        vec_env_mod.SubprocVecEnv = object
        vec_env_mod.VecMonitor = object
        vec_env_mod.VecFrameStack = object
        vec_env_mod.DummyVecEnv = object
        sys.modules["stable_baselines3.common.vec_env"] = vec_env_mod

        monitor_mod = ModuleType("stable_baselines3.common.monitor")
        monitor_mod.Monitor = lambda env: env
        sys.modules["stable_baselines3.common.monitor"] = monitor_mod

        results_plotter_mod = ModuleType("stable_baselines3.common.results_plotter")
        results_plotter_mod.load_results = lambda *args, **kwargs: None
        results_plotter_mod.ts2xy = lambda *args, **kwargs: ([], [])
        sys.modules["stable_baselines3.common.results_plotter"] = results_plotter_mod

    if "sb3_contrib" not in sys.modules:
        sb3_contrib_mod = ModuleType("sb3_contrib")
        sb3_contrib_mod.RecurrentPPO = SimpleNamespace(load=lambda *args, **kwargs: None)
        sys.modules["sb3_contrib"] = sb3_contrib_mod

    if "imageio" not in sys.modules:
        sys.modules["imageio"] = ModuleType("imageio")

    if "gdown" not in sys.modules:
        gdown_mod = ModuleType("gdown")
        gdown_mod.download = lambda *args, **kwargs: None
        sys.modules["gdown"] = gdown_mod

    for name in [
        "pygame",
        "pygame.gfxdraw",
        "pymunk",
        "pymunk.pygame_util",
        "cv2",
        "skvideo",
        "skvideo.io",
    ]:
        sys.modules.setdefault(name, DummyModule(name))
    sys.modules["pygame"].image = SimpleNamespace(load=lambda *args, **kwargs: Dummy())
    sys.modules["pygame"].Surface = Dummy
    sys.modules["pygame"].__path__ = []
    pygame_locals_mod = ModuleType("pygame.locals")
    pygame_locals_mod.QUIT = 0
    sys.modules.setdefault("pygame.locals", pygame_locals_mod)

    space_debug_mod = ModuleType("pymunk.space_debug_draw_options")
    space_debug_mod.SpaceDebugColor = Dummy
    sys.modules.setdefault("pymunk.space_debug_draw_options", space_debug_mod)

    vec2d_mod = ModuleType("pymunk.vec2d")
    vec2d_mod.Vec2d = Dummy
    sys.modules.setdefault("pymunk.vec2d", vec2d_mod)

    skimage_mod = DummyModule("skimage")
    transform_mod = DummyModule("skimage.transform")
    skimage_mod.transform = transform_mod
    sys.modules.setdefault("skimage", skimage_mod)
    sys.modules.setdefault("skimage.transform", transform_mod)

    ipython_mod = DummyModule("IPython")
    display_mod = ModuleType("IPython.display")
    display_mod.Video = Dummy
    ipython_mod.display = display_mod
    sys.modules.setdefault("IPython", ipython_mod)
    sys.modules.setdefault("IPython.display", display_mod)


install_import_stubs()

import torch  # noqa: E402

train = importlib.import_module("train")
agent_mod = importlib.import_module("environment.agent")
env_mod = importlib.import_module("environment.environment")


class FakeSignal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def emit(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback(**kwargs)


class FakeRawEnv:
    def __init__(self) -> None:
        self.win_signal = FakeSignal()
        self.knockout_signal = FakeSignal()
        self.players = []
        self.logger = [{}]


class FakeCombatEnv:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.reward_manager = kwargs["reward_manager"]
        self.raw_env = FakeRawEnv()
        FakeCombatEnv.instances.append(self)

    def reset(self, *args, **kwargs):
        return "reset"


class FakePosition:
    def __init__(self, x: float, y: float = 0.0) -> None:
        self.x = x
        self.y = y

    def get_distance(self, other: "FakePosition") -> float:
        return abs(self.x - other.x)


class FakePlayer:
    def __init__(
        self,
        *,
        damage_done: float = 0.0,
        damage_taken: float = 0.0,
        x: float = 0.0,
        state=None,
    ) -> None:
        self.damage_done = damage_done
        self.damage_taken_total = damage_taken
        self.body = SimpleNamespace(position=FakePosition(x))
        self.state = state


def phase_config(rewards: dict[str, float]) -> train.PhaseConfig:
    params = {
        "phases": {
            1: {
                "name": "test combat",
                "type": "combat",
                "rewards": rewards,
                "opponents": {"random_agent": 1.0},
            }
        }
    }
    return train.PhaseConfig(params, 1)


def combat_params() -> dict:
    return {
        "self_play_settings": {"save_freq": 1, "max_saved_models": 1},
        "folders": {"parent_dir": "/tmp", "model_name": "ai2-test"},
        "environment_settings": {"resolution": "LOW"},
        "opponent_history": {"enabled": False},
    }


def build_factory_env(config: train.PhaseConfig):
    FakeCombatEnv.instances = []
    with (
        mock.patch.object(train, "SelfPlayWarehouseBrawl", FakeCombatEnv),
        mock.patch.object(train, "Float32Wrapper", side_effect=lambda env: env),
        mock.patch.object(train, "DamageTrackingWrapper", side_effect=lambda env: env),
        mock.patch.object(train, "VectorizedSaveHandler", side_effect=lambda **kwargs: None),
    ):
        returned_env = train.make_combat_env(config, combat_params(), device="cpu")
    return returned_env, FakeCombatEnv.instances[-1]


class ClassicRewardSemanticsTest(unittest.TestCase):
    def tearDown(self) -> None:
        train.GLOBAL_REWARD_MANAGER = None

    def test_factory_subscribes_exact_signal_names_and_shapes(self):
        config = phase_config({"win": 2.0, "knockout": 3.0})

        _, env = build_factory_env(config)

        self.assertEqual(
            set(env.reward_manager.signal_subscriptions),
            {"win", "knockout"},
        )
        for key, expected_signal in {
            "win": "win_signal",
            "knockout": "knockout_signal",
        }.items():
            subscription = env.reward_manager.signal_subscriptions[key]
            self.assertIsInstance(subscription, tuple)
            self.assertEqual(subscription[0], expected_signal)
            self.assertIsInstance(subscription[1], agent_mod.RewTerm)
        self.assertEqual(len(env.raw_env.win_signal.callbacks), 1)
        self.assertEqual(len(env.raw_env.knockout_signal.callbacks), 1)

    def test_signal_rewards_use_player_as_learner_and_remain_zero_sum(self):
        config = phase_config({"win": 2.0, "knockout": 3.0})
        _, env = build_factory_env(config)
        env.raw_env.players = [FakePlayer(), FakePlayer()]

        env.raw_env.win_signal.emit(env=env.raw_env, agent="player")
        reward = env.reward_manager.process(env.raw_env, dt=0.0)
        self.assertEqual(reward, 2.0)

        env.raw_env.win_signal.emit(env=env.raw_env, agent="opponent")
        collected = env.reward_manager.collected_signal_rewards
        self.assertTrue(torch.equal(collected, torch.tensor([-2.0, 2.0])))
        reward = env.reward_manager.process(env.raw_env, dt=0.0)
        self.assertEqual(reward, -2.0)

    def test_phase1_knockout_weight_is_positive_because_learner_ko_is_negated(self):
        config_text = (ROOT / "config.yaml").read_text()
        phase1_block = config_text.split("  2:", 1)[0]
        match = re.search(r"knockout:\s*([-0-9.]+)", phase1_block)
        self.assertIsNotNone(match)
        config = phase_config({"knockout": float(match.group(1))})
        manager = train.PhaseAwareRewardManager(config)

        self.assertEqual(config.rewards["knockout"], 3.0)
        self.assertEqual(manager.compute_knockout_reward(agent_got_ko=True), -3.0)
        self.assertEqual(manager.compute_knockout_reward(agent_got_ko=False), 3.0)

    def test_aggression_rewards_real_attack_state_only(self):
        config = phase_config({"aggression": 0.5})
        manager = train.PhaseAwareRewardManager(config)
        attack_player = FakePlayer(state=env_mod.AttackState.__new__(env_mod.AttackState))
        idle_player = FakePlayer(state=object())

        attack_reward = manager.compute_reward(
            SimpleNamespace(players=[attack_player, FakePlayer()])
        )
        idle_reward = manager.compute_reward(
            SimpleNamespace(players=[idle_player, FakePlayer()])
        )

        self.assertEqual(attack_reward, 0.05)
        self.assertEqual(idle_reward, 0.0)

    def test_zero_weights_noop_for_continuous_rewards(self):
        config = phase_config(
            {
                "distance": 0.0,
                "aggression": 0.0,
                "damage_dealt": 0.0,
                "damage_taken": 0.0,
                "net_damage": 0.0,
                "edge_penalty": 0.0,
            }
        )
        manager = train.PhaseAwareRewardManager(config)

        reward = manager.compute_reward(
            SimpleNamespace(
                players=[
                    FakePlayer(damage_done=10.0, damage_taken=5.0, x=11.0, state=env_mod.AttackState.__new__(env_mod.AttackState)),
                    FakePlayer(x=0.0),
                ]
            )
        )

        self.assertEqual(reward, 0.0)

    def test_reward_manager_reset_clears_pending_signal_rewards(self):
        manager = agent_mod.RewardManager()
        manager.collected_signal_rewards = torch.tensor([4.0, -4.0])

        manager.reset()

        self.assertEqual(manager.collected_signal_rewards, 0.0)

    def test_episode_reset_clears_stale_damage_distance_and_clean_bonus_state(self):
        config = phase_config(
            {
                "distance": 2.0,
                "damage_dealt": 1.0,
                "damage_taken": -1.0,
                "win": 5.0,
                "clean_win_bonus": 2.0,
            }
        )
        manager = train.PhaseAwareRewardManager(config)
        first_env = SimpleNamespace(
            players=[FakePlayer(damage_done=10.0, damage_taken=1.0, x=10.0), FakePlayer(x=0.0)]
        )
        second_env = SimpleNamespace(
            players=[FakePlayer(damage_done=0.0, damage_taken=0.0, x=1.0), FakePlayer(x=0.0)]
        )

        manager.compute_reward(first_env)
        self.assertEqual(manager.compute_win_reward(winner_is_agent=True), 7.0)

        manager.reset_episode()
        reward = manager.compute_reward(second_env)

        self.assertEqual(reward, 0.0)
        self.assertEqual(manager.compute_win_reward(winner_is_agent=True), 5.0)
        self.assertEqual(manager.total_damage_dealt, 10.0)
        self.assertEqual(manager.total_damage_received, 1.0)

    def test_episode_reset_preserves_explicit_lifetime_damage_counters(self):
        config = phase_config({"damage_dealt": 1.0, "damage_taken": -1.0})
        manager = train.PhaseAwareRewardManager(config)
        env = SimpleNamespace(
            players=[FakePlayer(damage_done=8.0, damage_taken=3.0), FakePlayer()]
        )

        manager.compute_reward(env)
        manager.reset_episode()

        self.assertEqual(manager.episode_damage_dealt, 0.0)
        self.assertEqual(manager.episode_damage_received, 0.0)
        self.assertTrue(hasattr(manager, "lifetime_damage_dealt"))
        self.assertTrue(hasattr(manager, "lifetime_damage_received"))
        self.assertEqual(manager.lifetime_damage_dealt, 8.0)
        self.assertEqual(manager.lifetime_damage_received, 3.0)
        self.assertEqual(manager.total_damage_dealt, manager.lifetime_damage_dealt)
        self.assertEqual(manager.total_damage_received, manager.lifetime_damage_received)

    def test_factory_reset_invokes_episode_reset_for_each_env(self):
        config = phase_config({"damage_dealt": 1.0})
        returned_env, env = build_factory_env(config)
        manager = env.reward_manager.reward_functions["phase_reward"].func.reward_manager
        with mock.patch.object(manager, "reset_episode", wraps=manager.reset_episode) as reset_spy:
            returned_env.reset()

        reset_spy.assert_called_once_with()

    def test_two_envs_with_different_phase_configs_do_not_share_weights(self):
        first_config = phase_config({"win": 1.0})
        second_config = phase_config({"win": 9.0})

        _, first_env = build_factory_env(first_config)
        _, second_env = build_factory_env(second_config)

        first_manager = first_env.reward_manager.reward_functions["phase_reward"].func.reward_manager
        second_manager = second_env.reward_manager.reward_functions["phase_reward"].func.reward_manager
        self.assertIsNot(first_manager, second_manager)
        self.assertEqual(first_manager.weights["win"], 1.0)
        self.assertEqual(second_manager.weights["win"], 9.0)


if __name__ == "__main__":
    unittest.main()
