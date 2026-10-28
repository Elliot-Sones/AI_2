"""Opt-in frozen-game patch tests for native navigation helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
import importlib
import os
from pathlib import Path
import random
import sys
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))

FROZEN_ENV_SHA256 = "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"


def resolve_frozen_env() -> Path:
    candidates = []
    if os.environ.get("AI2_NATIVE_GAME_ORACLE"):
        candidates.append(Path(os.environ["AI2_NATIVE_GAME_ORACLE"]))
    if os.environ.get("AI2_REPLAY_SOURCE"):
        candidates.append(Path(os.environ["AI2_REPLAY_SOURCE"]) / "UTMIST-AI2-main/environment/environment.py")
    candidates.append(
        ROOT.parent
        / "AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/"
        "payload/variants/corrected/UTMIST-AI2-main/environment/environment.py"
    )
    for path in candidates:
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != FROZEN_ENV_SHA256:
                raise AssertionError(f"frozen game oracle hash mismatch for {path}: {digest}")
            return path
    raise AssertionError(f"frozen game oracle not found; checked {[str(path) for path in candidates]}")


def load_frozen_is_on_floor():
    source_path = resolve_frozen_env()
    tree = ast.parse(source_path.read_text())
    player = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Player")
    method = next(node for node in player.body if isinstance(node, ast.FunctionDef) and node.name == "is_on_floor")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["is_on_floor"]


def require_pymunk():
    try:
        import pymunk
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("Pymunk oracle unavailable") from exc
    if getattr(pymunk, "version", None) != "6.2.1":
        raise unittest.SkipTest(f"Pymunk oracle must be 6.2.1, got {getattr(pymunk, 'version', None)!r}")
    return pymunk


def make_player_class():
    class Player:
        def is_on_floor(self):
            return self.original_floor_result

    return Player


class RaisingHelper:
    def is_on_floor(self, *args):
        raise RuntimeError("native helper unavailable")


class RaisingInputHelper:
    def update(self, *args):
        raise RuntimeError("native input unavailable")


class FakeObservationHelper:
    def prepare_get_obs(self, player):
        return lambda: ["native", player.original_floor_result]


class AutoLoadedObservationHelper:
    def prepare_get_obs(self, player):
        return lambda: [float(player.agent_id), 1.0]


class ObservationModuleSwap:
    """Install a fake observation module in both places an import can find it.

    `import ai2.simulation.native_live_observation as x` reads the package
    attribute before sys.modules, so swapping only sys.modules leaves the real
    module in use once anything has imported it.
    """

    NAME = "ai2.simulation.native_live_observation"

    def __init__(self, fake):
        import ai2.simulation as package

        self.package = package
        self.fake = fake
        self.old_module = sys.modules.get(self.NAME)
        self.old_attr = package.__dict__.get("native_live_observation")
        sys.modules[self.NAME] = fake
        package.native_live_observation = fake

    def restore(self):
        if self.old_module is None:
            sys.modules.pop(self.NAME, None)
        else:
            sys.modules[self.NAME] = self.old_module
        if self.old_attr is None:
            self.package.__dict__.pop("native_live_observation", None)
        else:
            self.package.native_live_observation = self.old_attr


class RecordingObservationModule(types.ModuleType):
    def __init__(self):
        super().__init__("native_live_observation")
        self.load_calls = []
        self.attach_calls = []
        self.restore_calls = []

    def load_c_extension(self, *, facing_right, dropped_spawner_type):
        self.load_calls.append((facing_right, dropped_spawner_type))
        return AutoLoadedObservationHelper()

    def patch_players_get_obs(self, players, *, helper, source_path=None, expected_sha256=None):
        owner = self

        class Patch:
            def __init__(self):
                self.players = list(players)
                self._originals = []
                self._attached = False

            def attach(self):
                owner.attach_calls.append([getattr(player, "agent_id", None) for player in self.players])
                for player in self.players:
                    original = getattr(player, "get_obs")
                    prepared = helper.prepare_get_obs(player)
                    self._originals.append((player, original, prepared))
                    setattr(player, "get_obs", prepared)
                self._attached = True
                return self

            def restore(self):
                if not self._attached:
                    return
                owner.restore_calls.append([getattr(player, "agent_id", None) for player in self.players])
                for player, original, prepared in self._originals:
                    if getattr(player, "get_obs", None) is prepared:
                        setattr(player, "get_obs", original)
                self._attached = False

        return Patch()


class CountingGeometryBinding:
    def __init__(self, result: bool):
        self.result = result
        self.calls = 0

    def is_on_floor(self):
        self.calls += 1
        return self.result


class CountingGeometryHelper:
    def __init__(self):
        self.bindings = []

    def prepare_is_on_floor(self, *refs):
        binding = CountingGeometryBinding(True)
        self.bindings.append((refs, binding))
        return binding


def load_input_handler_from_frozen_ast():
    source_path = resolve_frozen_env()
    text = source_path.read_text()
    tree = ast.parse(text)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in ("Facing", "KeyStatus", "HorizontalState", "PlayerInputHandler")
    ]
    namespace = {"dataclass": dataclass, "Enum": Enum, "np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["PlayerInputHandler"]


def load_reset_and_observe_all_from_frozen_ast():
    source_path = resolve_frozen_env()
    tree = ast.parse(source_path.read_text())
    env_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WarehouseBrawl")
    selected = [
        node
        for node in env_class.body
        if isinstance(node, ast.FunctionDef) and node.name in ("reset", "_observe_all")
    ]
    namespace = {"pymunk": require_pymunk(), "np": np, "Tuple": tuple, "Any": object}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["reset"], namespace["_observe_all"]


def make_environment_module_with_player(module_name: str, source_path: Path):
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)

    class Facing:
        RIGHT = object()

    class DroppedWeaponSpawner:
        pass

    exec(
        """
class Player:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id
        self.original_floor_result = False

    def is_on_floor(self):
        return self.original_floor_result

    def get_obs(self):
        raise AssertionError("reset-time observation reached unpatched get_obs")
""",
        module.__dict__,
    )
    module.Facing = Facing
    module.DroppedWeaponSpawner = DroppedWeaponSpawner
    sys.modules[module_name] = module
    return module


@unittest.skipIf(resolve_frozen_env() is None, "frozen source unavailable")
class NativeGamePatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pymunk = require_pymunk()
        import ai2.simulation.native_navigation_geometry as geom

        cls.geometry_helper = geom.load_c_helper()
        cls.oracle_method = load_frozen_is_on_floor()
        cls.frozen_env = resolve_frozen_env()

    def make_world(
        self,
        *,
        player_pos=(0.0, 0.0),
        ground1_pos=(4.5, 1.0),
        ground2_pos=(-4.5, 3.0),
        platform_pos=(1.0, 0.0),
        player_angle=0.0,
        platform_angle=0.0,
        player_cls=None,
    ):
        pymunk = self.pymunk
        space = pymunk.Space()
        player_body = pymunk.Body(1.0, 1e9)
        player_shape = pymunk.Poly.create_box(player_body, size=(0.87, 1.0))
        player_body.position = player_pos
        player_body.angle = player_angle
        ground1_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        ground1_shape = pymunk.Poly.create_box(ground1_body, size=(10.0, 1.0))
        ground1_body.position = ground1_pos
        ground2_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        ground2_shape = pymunk.Poly.create_box(ground2_body, size=(10.0, 1.0))
        ground2_body.position = ground2_pos
        platform_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        platform_shape = pymunk.Poly.create_box(platform_body, size=(2.0, 1.0))
        platform_body.position = platform_pos
        platform_body.angle = platform_angle
        space.add(
            player_body,
            player_shape,
            ground1_body,
            ground1_shape,
            ground2_body,
            ground2_shape,
            platform_body,
            platform_shape,
        )
        objects = {
            "ground1": types.SimpleNamespace(body=ground1_body, shape=ground1_shape),
            "ground2": types.SimpleNamespace(body=ground2_body, shape=ground2_shape),
            "platform1": types.SimpleNamespace(body=platform_body, shape=platform_shape),
        }
        player_cls = player_cls or make_player_class()
        player = player_cls()
        player.body = player_body
        player.shape = player_shape
        player.env = types.SimpleNamespace(objects=objects)
        player.original_floor_result = False
        return player, objects, space

    def oracle(self, player) -> bool:
        return bool(types.MethodType(type(self).oracle_method, player)())

    def test_patch_module_is_importable_and_reports_non_space_step_scope(self):
        import ai2.simulation.native_game_patch as patch

        info = patch.describe_native_game_patch()
        self.assertEqual(info["space_step"], "original Pymunk Space.step is not patched")
        self.assertEqual(
            info["patched_methods"],
            [
                "Player.is_on_floor",
                "PlayerInputHandler.update optional via enable_input=True",
                "Player.get_obs optional via enable_observation=True",
            ],
        )
        self.assertEqual(info["enable_input_env_var"], "AI2_NATIVE_INPUT")
        self.assertEqual(info["enable_observation_env_var"], "AI2_NATIVE_OBSERVATION")
        self.assertIn("CPU exact-solver bridge", info["chipmunk_bridge"]["bridge_scope"])

    def test_is_on_floor_uses_prepared_binding_cache_and_refreshes_on_identity_change(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        helper = CountingGeometryHelper()
        player, objects, _ = self.make_world(player_cls=Player, player_pos=(4.5, 1.0))
        with patch.patch_frozen_player_class(Player, helper=helper, allow_synthetic_class=True):
            self.assertTrue(player.is_on_floor())
            self.assertTrue(player.is_on_floor())
            self.assertEqual(len(helper.bindings), 1)
            self.assertEqual(helper.bindings[0][1].calls, 2)
            self.assertIs(getattr(player, "_ai2_floor_binding")[1], helper.bindings[0][1])

            replacement, _, _ = self.make_world(player_cls=Player, player_pos=(4.5, 1.0))
            objects["platform1"].shape = replacement.env.objects["platform1"].shape
            self.assertTrue(player.is_on_floor())
            self.assertEqual(len(helper.bindings), 2)
            self.assertEqual(helper.bindings[1][1].calls, 1)

    def test_patch_matches_actual_frozen_is_on_floor_ast_on_real_pymunk_shapes(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        cases = [
            {"player_pos": (4.5, 1.0)},
            {"player_pos": (-4.5, 3.0)},
            {"player_pos": (1.0, -0.02), "platform_pos": (1.0, 0.0)},
            {"player_pos": (2.251, -0.02), "platform_pos": (1.0, 0.0)},
            {"player_pos": (1.0, -0.02), "platform_pos": (1.0, 0.0), "platform_angle": 0.05},
        ]
        rng = random.Random(20260906)
        for _ in range(100):
            cases.append(
                {
                    "player_pos": (rng.uniform(-7.0, 7.0), rng.uniform(-1.0, 4.0)),
                    "platform_pos": (rng.uniform(-1.5, 1.5), rng.uniform(-0.5, 2.5)),
                    "player_angle": rng.uniform(-0.2, 0.2),
                    "platform_angle": rng.uniform(-0.15, 0.15),
                }
            )

        with patch.patch_frozen_player_class(
            Player,
            helper=self.geometry_helper,
            source_path=self.frozen_env,
            allow_synthetic_class=True,
        ):
            for case in cases:
                player, _, _ = self.make_world(player_cls=Player, **case)
                self.assertEqual(player.is_on_floor(), self.oracle(player), case)

    def test_patch_restores_class_hook_exactly_and_restores_after_exception(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        original = Player.is_on_floor
        with patch.patch_frozen_player_class(Player, helper=self.geometry_helper, allow_synthetic_class=True) as active:
            self.assertIs(active.original_is_on_floor, original)
            self.assertIsNot(Player.is_on_floor, original)
            active.attach()
            self.assertIsNot(Player.is_on_floor, original)
        self.assertIs(Player.is_on_floor, original)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with patch.patch_frozen_player_class(Player, helper=self.geometry_helper, allow_synthetic_class=True):
                raise RuntimeError("boom")
        self.assertIs(Player.is_on_floor, original)

    def test_patch_does_not_replace_pymunk_space_step(self):
        import ai2.simulation.native_game_patch as patch

        original_space_step = self.pymunk.Space.step
        Player = make_player_class()
        with patch.patch_frozen_player_class(Player, helper=self.geometry_helper, allow_synthetic_class=True):
            self.assertIs(self.pymunk.Space.step, original_space_step)
        self.assertIs(self.pymunk.Space.step, original_space_step)

    def test_strict_native_helper_failure_raises_with_context_by_default(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        player, _, _ = self.make_world(player_cls=Player, player_pos=(4.5, 1.0))
        with patch.patch_frozen_player_class(Player, helper=RaisingHelper(), allow_synthetic_class=True) as active:
            with self.assertRaisesRegex(RuntimeError, "native is_on_floor failed"):
                player.is_on_floor()
            receipt = active.receipt()
        self.assertEqual(receipt["native_failures"], 1)
        self.assertIn("native helper unavailable", receipt["last_native_error"])

    def test_explicit_fallback_counts_and_returns_original_method_without_gameplay_change(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        player, _, _ = self.make_world(player_cls=Player, player_pos=(4.5, 1.0))
        player.original_floor_result = True
        with patch.patch_frozen_player_class(
            Player,
            helper=RaisingHelper(),
            strict=False,
            allow_synthetic_class=True,
        ) as active:
            self.assertTrue(player.is_on_floor())
            self.assertEqual(active.receipt()["native_failures"], 1)

    def test_source_hash_mismatch_fails_before_mutating_class(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        original = Player.is_on_floor
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            patch.patch_frozen_player_class(
                Player,
                helper=self.geometry_helper,
                source_path=self.frozen_env,
                expected_sha256="0" * 64,
                allow_synthetic_class=True,
            ).attach()
        self.assertIs(Player.is_on_floor, original)

    def test_optional_env_installer_is_noop_unless_enabled(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        module = types.SimpleNamespace(Player=Player, __file__=str(self.frozen_env))
        original = Player.is_on_floor
        self.assertIsNone(patch.install_from_environment_module(module, enabled=False, helper=self.geometry_helper))
        self.assertIs(Player.is_on_floor, original)
        active = patch.install_from_environment_module(
            module,
            enabled=True,
            helper=self.geometry_helper,
            allow_synthetic_class=True,
        )
        self.assertIsNotNone(active)
        self.assertIsNot(Player.is_on_floor, original)
        active.restore()
        self.assertIs(Player.is_on_floor, original)

    def test_environment_module_installer_hashes_module_file_when_enabled(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        module = types.SimpleNamespace(Player=Player)
        original = Player.is_on_floor

        with self.assertRaisesRegex(ValueError, "environment module source"):
            patch.install_from_environment_module(
                module,
                enabled=True,
                helper=self.geometry_helper,
            )
        self.assertIs(Player.is_on_floor, original)

        module.__file__ = "/definitely/not/frozen.py"
        with self.assertRaisesRegex(ValueError, "environment module source"):
            patch.install_from_environment_module(
                module,
                enabled=True,
                helper=self.geometry_helper,
            )
        self.assertIs(Player.is_on_floor, original)

        active = patch.install_from_environment_module(
            types.SimpleNamespace(Player=Player),
            enabled=True,
            helper=self.geometry_helper,
            allow_synthetic_class=True,
        )
        active.restore()
        self.assertIs(Player.is_on_floor, original)

    def test_install_native_game_patch_requires_source_path_and_stores_strong_reference_on_env(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        player, _, engine = self.make_world(player_cls=Player, player_pos=(4.5, 1.0))
        engine.players = {0: player}
        wrapper = types.SimpleNamespace(raw_env=engine)
        original = Player.is_on_floor

        with self.assertRaisesRegex(ValueError, "source_path is required"):
            patch.install_native_game_patch(wrapper, helper=self.geometry_helper)
        self.assertIs(Player.is_on_floor, original)

        active = patch.install_native_game_patch(
            wrapper,
            source_path=self.frozen_env,
            helper=self.geometry_helper,
            allow_synthetic_class=True,
        )
        self.assertIs(wrapper._ai2_native_game_patch, active)
        self.assertIs(engine._ai2_native_game_patch, active)
        self.assertIsNot(Player.is_on_floor, original)
        self.assertEqual(player.is_on_floor(), self.oracle(player))

        patch.restore_native_game_patch(wrapper)
        self.assertIs(Player.is_on_floor, original)
        self.assertFalse(hasattr(wrapper, "_ai2_native_game_patch"))
        self.assertFalse(hasattr(engine, "_ai2_native_game_patch"))

    def test_install_native_game_patch_is_idempotent_for_reset_hooks(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        player, _, engine = self.make_world(player_cls=Player, player_pos=(4.5, 1.0))
        engine.players = {0: player}
        original = Player.is_on_floor

        first = patch.install_native_game_patch(
            engine,
            source_path=self.frozen_env,
            helper=self.geometry_helper,
            allow_synthetic_class=True,
        )
        patched = Player.is_on_floor
        second = patch.install_native_game_patch(
            engine,
            source_path=self.frozen_env,
            helper=self.geometry_helper,
            allow_synthetic_class=True,
        )

        self.assertIs(first, second)
        self.assertIs(Player.is_on_floor, patched)
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            patch.install_native_game_patch(
                engine,
                source_path=self.frozen_env,
                helper=self.geometry_helper,
                expected_sha256="0" * 64,
                allow_synthetic_class=True,
            )
        self.assertIs(Player.is_on_floor, patched)
        patch.restore_native_game_patch(engine)
        self.assertIs(Player.is_on_floor, original)

    def test_multiple_envs_sharing_same_player_class_do_not_stack_and_refcount_restore(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        player1, _, engine1 = self.make_world(player_cls=Player, player_pos=(4.5, 1.0))
        player2, _, engine2 = self.make_world(player_cls=Player, player_pos=(-4.5, 3.0))
        engine1.players = {0: player1}
        engine2.players = {0: player2}
        original = Player.is_on_floor

        first = patch.install_native_game_patch(
            engine1,
            source_path=self.frozen_env,
            helper=self.geometry_helper,
            allow_synthetic_class=True,
        )
        hook = Player.is_on_floor
        second = patch.install_native_game_patch(
            engine2,
            source_path=self.frozen_env,
            helper=self.geometry_helper,
            allow_synthetic_class=True,
        )

        self.assertIs(Player.is_on_floor, hook)
        self.assertIsNot(first, second)
        patch.restore_native_game_patch(engine1)
        self.assertIs(Player.is_on_floor, hook)
        patch.restore_native_game_patch(engine2)
        self.assertIs(Player.is_on_floor, original)

    def test_restore_does_not_overwrite_foreign_hook_and_source_mismatch_rejected(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        original = Player.is_on_floor
        active = patch.patch_frozen_player_class(
            Player,
            helper=self.geometry_helper,
            allow_synthetic_class=True,
        ).attach()

        def foreign_hook(self):
            return True

        Player.is_on_floor = foreign_hook
        active.restore()
        self.assertIs(Player.is_on_floor, foreign_hook)
        Player.is_on_floor = original

        with self.assertRaisesRegex(ValueError, "Player class source mismatch"):
            patch.patch_frozen_player_class(
                Player,
                helper=self.geometry_helper,
                source_path=self.frozen_env,
            ).attach()
        self.assertIs(Player.is_on_floor, original)

    def test_native_input_default_off_and_opt_in_matches_actual_frozen_input_handler(self):
        import ai2.simulation.native_game_patch as patch
        import ai2.simulation.native_live_input as native_live_input

        Player = make_player_class()
        InputHandler = load_input_handler_from_frozen_ast()
        player, _, engine = self.make_world(player_cls=Player, player_pos=(4.5, 1.0))
        player.input = InputHandler()
        engine.players = {0: player}
        original_update = InputHandler.update

        floor_only = patch.install_native_game_patch(
            engine,
            source_path=self.frozen_env,
            helper=self.geometry_helper,
            allow_synthetic_class=True,
        )
        self.assertIs(InputHandler.update, original_update)
        patch.restore_native_game_patch(engine)
        self.assertIsNotNone(floor_only)

        active = patch.install_native_game_patch(
            engine,
            source_path=self.frozen_env,
            helper=self.geometry_helper,
            enable_input=True,
            input_helper=native_live_input.load_live_input(),
            allow_synthetic_class=True,
        )
        self.assertIsNot(InputHandler.update, original_update)
        reference = InputHandler()
        candidate = InputHandler()
        for mask in (0, 1, 2, 3, 17, 513, 1023):
            action = np.array([(mask >> bit) & 1 for bit in range(10)], dtype=np.float32)
            original_update(reference, action)
            candidate.update(action)
            self.assertEqual(repr(reference), repr(candidate))
        self.assertEqual(active.receipt()["input_native_failures"], 0)
        patch.restore_native_game_patch(engine)
        self.assertIs(InputHandler.update, original_update)

    def test_native_input_strict_failure_and_explicit_fallback_are_counted(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        InputHandler = load_input_handler_from_frozen_ast()
        player, _, engine = self.make_world(player_cls=Player, player_pos=(4.5, 1.0))
        player.input = InputHandler()
        engine.players = {0: player}
        original_update = InputHandler.update

        active = patch.install_native_game_patch(
            engine,
            source_path=self.frozen_env,
            helper=self.geometry_helper,
            enable_input=True,
            input_helper=RaisingInputHelper(),
            allow_synthetic_class=True,
        )
        with self.assertRaisesRegex(RuntimeError, "native input update failed"):
            player.input.update([0] * 10)
        self.assertEqual(active.receipt()["input_native_failures"], 1)
        patch.restore_native_game_patch(engine)
        self.assertIs(InputHandler.update, original_update)

    def test_observation_default_off_and_explicit_helper_patch_restore(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        player, objects, engine = self.make_world(player_cls=Player, player_pos=(4.5, 1.0))
        player.get_obs = lambda: ["original"]
        player.PLATFORM = objects["platform1"]
        engine.players = {0: player}
        original_get_obs = player.get_obs

        floor_only = patch.install_native_game_patch(
            engine,
            source_path=self.frozen_env,
            helper=self.geometry_helper,
            allow_synthetic_class=True,
        )
        self.assertIs(player.get_obs, original_get_obs)
        patch.restore_native_game_patch(engine)
        self.assertIsNotNone(floor_only)

        active = patch.install_native_game_patch(
            engine,
            source_path=self.frozen_env,
            helper=self.geometry_helper,
            enable_observation=True,
            observation_helper=FakeObservationHelper(),
            allow_synthetic_class=True,
        )
        self.assertEqual(player.get_obs(), ["native", False])
        self.assertTrue(active.receipt()["observation_enabled"])
        patch.restore_native_game_patch(engine)
        self.assertIs(player.get_obs, original_get_obs)

    def test_enable_observation_auto_loads_helper_and_hooks_reset_before_first_observe_all(self):
        import ai2.simulation.native_game_patch as patch

        observation_swap = ObservationModuleSwap(RecordingObservationModule())
        fake_observation_module = observation_swap.fake
        module_name = "ai2_fake_frozen_env_for_native_patch"
        env_module = make_environment_module_with_player(module_name, self.frozen_env)
        reset_method, observe_all_method = load_reset_and_observe_all_from_frozen_ast()

        class Camera:
            def reset(self, env):
                env.camera_reset_seen_players = list(getattr(env, "players", []))

            def scale_background(self, env):
                env.scale_background_seen_players = list(getattr(env, "players", []))

        class ActualResetShapeEnv:
            def __init__(self):
                self.agents = [0, 1]
                self.camera = Camera()
                self.players = [env_module.Player(100), env_module.Player(101)]
                self.setup_calls = 0

            def _setup(self):
                self.setup_calls += 1
                self.players = [env_module.Player(0), env_module.Player(1)]

        ActualResetShapeEnv.reset = reset_method
        ActualResetShapeEnv._observe_all = observe_all_method
        engine = ActualResetShapeEnv()
        original_reset = engine.reset

        try:
            active = patch.install_native_game_patch(
                engine,
                source_path=self.frozen_env,
                helper=self.geometry_helper,
                enable_observation=True,
            )
            self.assertEqual(
                fake_observation_module.load_calls,
                [(env_module.Facing.RIGHT, env_module.DroppedWeaponSpawner)],
            )
            obs, info = engine.reset()
            self.assertEqual(info, {})
            self.assertEqual(engine.setup_calls, 1)
            self.assertEqual(fake_observation_module.attach_calls, [[100, 101], [0, 1]])
            self.assertEqual(fake_observation_module.restore_calls, [[100, 101]])
            self.assertEqual(obs[0].tolist(), [0.0, 1.0, 1.0, 1.0])
            self.assertEqual(obs[1].tolist(), [1.0, 1.0, 0.0, 1.0])
            receipt = active.receipt()
            self.assertTrue(receipt["observation_enabled"])
            self.assertEqual(receipt["observation_players"], 2)
            patch.restore_native_game_patch(engine)
            self.assertIs(getattr(engine.reset, "__self__", None), getattr(original_reset, "__self__", None))
            self.assertIs(getattr(engine.reset, "__func__", None), getattr(original_reset, "__func__", None))
            self.assertEqual(fake_observation_module.restore_calls, [[100, 101], [0, 1]])
        finally:
            observation_swap.restore()
            sys.modules.pop(module_name, None)

    def test_reset_hook_restores_only_own_wrapper_and_does_not_stack(self):
        import ai2.simulation.native_game_patch as patch

        observation_swap = ObservationModuleSwap(RecordingObservationModule())
        fake_observation_module = observation_swap.fake
        module_name = "ai2_fake_frozen_env_for_native_patch_refcount"
        env_module = make_environment_module_with_player(module_name, self.frozen_env)

        class Engine:
            def __init__(self, agent_id):
                self.players = [env_module.Player(agent_id)]
                self.reset_calls = 0

            def reset(self):
                self.reset_calls += 1
                self.players = [env_module.Player(self.reset_calls)]
                return self._observe_all(), {}

            def _observe_all(self):
                return {0: self.players[0].get_obs()}

        try:
            engine1 = Engine(10)
            engine2 = Engine(20)
            original1 = engine1.reset
            original2 = engine2.reset
            first = patch.install_native_game_patch(
                engine1,
                source_path=self.frozen_env,
                helper=self.geometry_helper,
                enable_observation=True,
            )
            second = patch.install_native_game_patch(
                engine2,
                source_path=self.frozen_env,
                helper=self.geometry_helper,
                enable_observation=True,
            )
            self.assertIsNot(first, second)
            self.assertNotEqual(engine1.reset, original1)
            self.assertNotEqual(engine2.reset, original2)
            self.assertEqual(engine1.reset()[0], {0: [1.0, 1.0]})
            self.assertEqual(engine2.reset()[0], {0: [1.0, 1.0]})

            foreign_reset = lambda: ({0: ["foreign"]}, {})
            engine1.reset = foreign_reset
            patch.restore_native_game_patch(engine1)
            self.assertIs(engine1.reset, foreign_reset)
            patch.restore_native_game_patch(engine2)
            self.assertIs(getattr(engine2.reset, "__self__", None), getattr(original2, "__self__", None))
            self.assertIs(getattr(engine2.reset, "__func__", None), getattr(original2, "__func__", None))
        finally:
            observation_swap.restore()
            sys.modules.pop(module_name, None)

    def test_observation_requires_explicit_helper_when_enabled(self):
        import ai2.simulation.native_game_patch as patch

        Player = make_player_class()
        player, objects, engine = self.make_world(player_cls=Player, player_pos=(4.5, 1.0))
        player.get_obs = lambda: ["original"]
        player.PLATFORM = objects["platform1"]
        engine.players = {0: player}
        original_get_obs = player.get_obs

        with self.assertRaisesRegex(ValueError, "observation_helper is required"):
            patch.install_native_game_patch(
                engine,
                source_path=self.frozen_env,
                helper=self.geometry_helper,
                enable_observation=True,
                allow_synthetic_class=True,
            )
        self.assertIs(player.get_obs, original_get_obs)


if __name__ == "__main__":
    unittest.main()
