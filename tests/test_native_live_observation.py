"""CPython hotpath for frozen Player.get_obs."""

from __future__ import annotations

import ast
from enum import Enum
import gc
import hashlib
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock
import weakref

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ENV_SHA256 = "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"


class Facing(Enum):
    RIGHT = 1
    LEFT = -1


class DroppedWeaponSpawner:
    pass


class WeakPlayer:
    pass


def resolve_frozen_env() -> Path:
    candidates = []
    if os.environ.get("AI2_OBSERVATION_ORACLE"):
        candidates.append(Path(os.environ["AI2_OBSERVATION_ORACLE"]))
    if os.environ.get("AI2_REPLAY_SOURCE"):
        candidates.append(Path(os.environ["AI2_REPLAY_SOURCE"]) / "UTMIST-AI2-main/environment/environment.py")
    candidates.append(
        ROOT.parent / "AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/"
        "payload/variants/corrected/UTMIST-AI2-main/environment/environment.py"
    )
    for path in candidates:
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != FROZEN_ENV_SHA256:
                raise AssertionError(f"frozen get_obs oracle hash mismatch for {path}: {digest}")
            return path
    raise AssertionError(f"frozen get_obs oracle not found; checked {[str(p) for p in candidates]}")


def load_frozen_get_obs():
    source_path = resolve_frozen_env()
    tree = ast.parse(source_path.read_text())
    player = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Player")
    method = next(node for node in player.body if isinstance(node, ast.FunctionDef) and node.name == "get_obs")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"Facing": Facing, "DroppedWeaponSpawner": DroppedWeaponSpawner}
    exec(compile(module, str(source_path), "exec"), ns)
    return ns["get_obs"]


try:
    import pymunk
except Exception:  # pragma: no cover - dependency-light hosts skip.
    pymunk = None


@unittest.skipIf(pymunk is None, "pymunk unavailable; observation tests require real body Vec2d values")
class NativeLiveObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ai2.simulation.native_live_observation as obs

        cls.obs = obs
        cls.helper = obs.load_c_extension(facing_right=Facing.RIGHT, dropped_spawner_type=DroppedWeaponSpawner)
        cls.oracle_method = load_frozen_get_obs()

    def make_player(self, *, pos=(0, 0), vel=(0, 0), facing=Facing.RIGHT, grounded=True, weapon="Punch",
                    state=None, spawners=(), platform_pos=(1, 2), platform_vel=(3, -4)):
        space = pymunk.Space()
        body = pymunk.Body(1, 1e9)
        shape = pymunk.Poly.create_box(body, size=(0.87, 1.0))
        body.position = pos
        body.velocity = vel
        platform_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        platform_body.position = platform_pos
        platform_body.velocity = platform_vel
        space.add(body, shape, platform_body)
        if state is None:
            state = types.SimpleNamespace(jumps_left=2, recoveries_left=1, dodge_timer=0, stun_frames=0, move_type=0)
        env = types.SimpleNamespace(
            space=space,
            weapon_controller=types.SimpleNamespace(spawners=list(spawners)),
        )
        player = WeakPlayer()
        player.body = body
        player.shape = shape
        player.env = env
        player.facing = facing
        player.state = state
        player.state_mapping = {
                "WalkingState": 0,
                "StandingState": 1,
                "TurnaroundState": 2,
                "AirTurnaroundState": 3,
                "SprintingState": 4,
                "StunState": 5,
                "InAirState": 6,
                "DodgeState": 7,
                "AttackState": 8,
                "DashState": 9,
                "BackDashState": 10,
                "KOState": 11,
                "TauntState": 12,
            }
        player.damage = 140
        player.stocks = 3
        player.weapon = weapon
        player.weapon_mapping = {"Punch": 0, "Spear": 1, "Hammer": 2}
        player.PLATFORM = types.SimpleNamespace(body=platform_body)
        player.is_on_floor = lambda: grounded
        player.get_obs = lambda: ["python"]
        return player

    def oracle(self, player):
        return list(types.MethodType(type(self).oracle_method, player)())

    def native(self, player):
        return self.helper.get_obs(player)

    def assertObsEqual(self, native, oracle, msg=None):
        self.assertEqual(len(native), 32)
        self.assertEqual(len(oracle), 32)
        self.assertTrue(all(type(v) is float for v in native))
        self.assertEqual(native, [float(v) for v in oracle], msg)

    def test_matches_frozen_ast_for_random_live_pymunk_values(self):
        rng = random.Random(20260906)
        for index in range(200):
            state_name = rng.choice(["InAirState", "DodgeState", "AttackState", "MysteryState"])
            state_type = type(state_name, (), {})
            state = state_type()
            if rng.random() < 0.7:
                state.jumps_left = rng.randint(0, 2)
            if rng.random() < 0.7:
                state.recoveries_left = rng.randint(0, 1)
            if rng.random() < 0.7:
                state.dodge_timer = rng.randint(0, 12)
            if rng.random() < 0.7:
                state.stun_frames = rng.randint(0, 30)
            if rng.random() < 0.7:
                state.move_type = rng.randint(0, 10)
            player = self.make_player(
                pos=(rng.uniform(-30, 30), rng.uniform(-12, 12)),
                vel=(rng.uniform(-20, 20), rng.uniform(-20, 20)),
                facing=rng.choice([Facing.RIGHT, Facing.LEFT]),
                grounded=rng.choice([True, False]),
                weapon=rng.choice(["Punch", "Spear", "Hammer"]),
                state=state,
                spawners=[
                    types.SimpleNamespace(active_weapon=object(), world_pos=[rng.uniform(-30, 30), rng.uniform(-12, 12)]),
                    DroppedWeaponSpawner(),
                ],
                platform_pos=(rng.uniform(-30, 30), rng.uniform(-12, 12)),
                platform_vel=(rng.uniform(-20, 20), rng.uniform(-20, 20)),
            )
            player.env.weapon_controller.spawners[1].active_weapon = object()
            player.env.weapon_controller.spawners[1].world_pos = [rng.uniform(-30, 30), rng.uniform(-12, 12)]
            player.env.weapon_controller.spawners[1].weapon_name = rng.choice(["Spear", "Hammer", "Sword"])
            self.assertObsEqual(self.native(player), self.oracle(player), index)

    def test_dropped_weapon_spawners_and_missing_slots_match_oracle(self):
        spear = DroppedWeaponSpawner()
        spear.active_weapon = object()
        spear.world_pos = [99, -99]
        spear.weapon_name = "Spear"
        hammer = DroppedWeaponSpawner()
        hammer.active_weapon = object()
        hammer.world_pos = [-3, 4]
        hammer.weapon_name = "Hammer"
        inactive = types.SimpleNamespace(active_weapon=None, world_pos=[5, 6])
        player = self.make_player(spawners=[spear, hammer, inactive])
        self.assertObsEqual(self.native(player), self.oracle(player))
        self.assertEqual(self.native(player)[16:28], [18.0, -7.0, 2.0, -3.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_actual_dropped_weapon_spawner_type_is_used_live(self):
        source_root = resolve_frozen_env().parents[1]
        sys.path.insert(0, str(source_root))
        from environment.environment import DroppedWeaponSpawner as ActualDroppedWeaponSpawner
        from environment.environment import Facing as ActualFacing

        dropped = ActualDroppedWeaponSpawner.__new__(ActualDroppedWeaponSpawner)
        dropped.active_weapon = object()
        dropped.world_pos = [2.5, 3.5]
        dropped.weapon_name = "Hammer"
        player = self.make_player(facing=ActualFacing.RIGHT, spawners=[dropped])
        helper = self.obs.load_c_extension(
            facing_right=ActualFacing.RIGHT,
            dropped_spawner_type=ActualDroppedWeaponSpawner,
        )
        self.assertEqual(helper.get_obs(player)[16:19], [2.5, 3.5, 3.0])

    def test_compile_extension_passes_fpic_for_linux_shared_linking(self):
        with tempfile.TemporaryDirectory(prefix="ai2-live-obs-build-") as tmp:
            with mock.patch.object(self.obs.tempfile, "mkdtemp", return_value=tmp):
                with mock.patch.object(self.obs.subprocess, "run") as run:
                    self.obs._compile_extension()
        cmd = run.call_args.args[0]
        self.assertIn("-fPIC", cmd)
        self.assertLess(cmd.index("-fPIC"), cmd.index(str(self.obs.NATIVE_DIR / "live_observation.c")))

    def test_reset_mutable_attrs_are_read_live_not_cached(self):
        player = self.make_player(pos=(0, 0), vel=(0, 0), grounded=True, weapon="Punch")
        first = self.native(player)
        player.body.position = (40, -20)
        player.body.velocity = (-40, 40)
        player.damage = 350
        player.stocks = 1
        player.weapon = "Hammer"
        player.PLATFORM.body.position = (-40, 20)
        player.PLATFORM.body.velocity = (40, -40)
        second = self.native(player)
        self.assertNotEqual(first, second)
        self.assertObsEqual(second, self.oracle(player))

    def test_prepared_callable_reads_same_live_player_after_mutation(self):
        player = self.make_player(pos=(0, 0), vel=(0, 0), grounded=True)
        prepared = self.helper.prepare_get_obs(player)
        self.assertObsEqual(prepared(), self.oracle(player))
        player.body.position = (-40, 40)
        player.body.velocity = (40, -40)
        player.damage = 210
        self.assertObsEqual(prepared(), self.oracle(player))

    def test_missing_required_attrs_fail_closed_but_optional_state_attrs_default_zero(self):
        state = type("StandingState", (), {})()
        player = self.make_player(state=state)
        self.assertObsEqual(self.native(player), self.oracle(player))
        del player.weapon_mapping
        with self.assertRaisesRegex((AttributeError, ValueError, RuntimeError), "weapon_mapping"):
            self.native(player)

    def test_patch_restores_after_exception_and_does_not_overwrite_foreign_hook(self):
        player = self.make_player()
        original = player.get_obs = lambda: ["old"]
        with self.obs.patch_players_get_obs([player], helper=self.helper):
            self.assertObsEqual(player.get_obs(), self.oracle(player))
        self.assertIs(player.get_obs, original)
        foreign = lambda: ["foreign"]
        with self.obs.patch_players_get_obs([player], helper=self.helper):
            player.get_obs = foreign
        self.assertIs(player.get_obs, foreign)
        player.get_obs = original
        with self.assertRaises(RuntimeError):
            with self.obs.patch_players_get_obs([player], helper=self.helper):
                raise RuntimeError("boom")
        self.assertIs(player.get_obs, original)

    def test_patch_refreshes_prepared_body_pointers_after_reset_swap(self):
        player = self.make_player(pos=(0, 0), vel=(0, 0), platform_pos=(1, 1), platform_vel=(0, 0))
        original = player.get_obs = lambda: ["old"]
        with self.obs.patch_players_get_obs([player], helper=self.helper):
            self.assertObsEqual(player.get_obs(), self.oracle(player))
            new_body = pymunk.Body(1, 1e9)
            new_shape = pymunk.Poly.create_box(new_body, size=(0.87, 1.0))
            new_body.position = (7, -3)
            new_body.velocity = (-2, 4)
            new_platform_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            new_platform_body.position = (-6, 2)
            new_platform_body.velocity = (5, -5)
            player.env.space.add(new_body, new_shape, new_platform_body)
            player.body = new_body
            player.shape = new_shape
            player.PLATFORM.body = new_platform_body
            self.assertObsEqual(player.get_obs(), self.oracle(player))
        self.assertIs(player.get_obs, original)

    def test_unrestored_patched_player_cycle_is_gc_collectible(self):
        player = self.make_player()
        ref = weakref.ref(player)
        patch = self.obs.patch_players_get_obs([player], helper=self.helper).__enter__()
        self.assertObsEqual(player.get_obs(), self.oracle(player))
        del patch
        del player
        for _ in range(3):
            gc.collect()
        self.assertIsNone(ref())

    def test_patch_attach_rolls_back_earlier_players_on_partial_failure(self):
        first = self.make_player()
        first_original = first.get_obs = lambda: ["first"]
        second = self.make_player()
        second.get_obs = lambda: ["second"]
        second.body = None
        with self.assertRaisesRegex(ValueError, "body"):
            self.obs.patch_players_get_obs([first, second], helper=self.helper).__enter__()
        self.assertIs(first.get_obs, first_original)

    def test_pymunk_version_mismatch_fails_before_borrowing_body_pointers(self):
        player = self.make_player()
        original_expected = self.obs.EXPECTED_PYMUNK_VERSION
        try:
            self.obs.EXPECTED_PYMUNK_VERSION = "0.0.0"
            with self.assertRaisesRegex(RuntimeError, "Pymunk version"):
                self.helper.prepare_get_obs(player)
        finally:
            self.obs.EXPECTED_PYMUNK_VERSION = original_expected

    def test_chipmunk_version_mismatch_fails_before_borrowing_body_pointers(self):
        player = self.make_player()
        original_expected = self.obs.EXPECTED_CHIPMUNK_VERSION
        try:
            self.obs.EXPECTED_CHIPMUNK_VERSION = "7.0.30"
            with self.assertRaisesRegex(RuntimeError, "Chipmunk version"):
                self.helper.prepare_get_obs(player)
        finally:
            self.obs.EXPECTED_CHIPMUNK_VERSION = original_expected

    def test_source_hash_mismatch_and_bad_lifetime_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="ai2-live-obs-source-") as tmp:
            bad = Path(tmp) / "environment.py"
            bad.write_text("not frozen")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.obs.verify_source_hash(bad, expected_sha256=FROZEN_ENV_SHA256)
        player = self.make_player()
        player.body = None
        with self.assertRaisesRegex((AttributeError, ValueError, RuntimeError), "body"):
            self.native(player)


if __name__ == "__main__":
    unittest.main()
