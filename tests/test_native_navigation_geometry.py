"""Exact native helper for frozen Player.is_on_floor geometry."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import random
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ENV_SHA256 = "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"


def resolve_frozen_env() -> Path:
    candidates = []
    if os.environ.get("AI2_GEOMETRY_ORACLE"):
        candidates.append(Path(os.environ["AI2_GEOMETRY_ORACLE"]))
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
                raise AssertionError(f"frozen geometry oracle hash mismatch for {path}: {digest}")
            return path
    raise AssertionError(f"frozen geometry oracle not found; checked {[str(p) for p in candidates]}")


def load_frozen_is_on_floor():
    source = resolve_frozen_env().read_text()
    tree = ast.parse(source)
    player = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Player")
    method = next(node for node in player.body if isinstance(node, ast.FunctionDef) and node.name == "is_on_floor")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {}
    exec(compile(module, str(resolve_frozen_env()), "exec"), ns)
    return ns["is_on_floor"]


try:
    import pymunk
except Exception:  # pragma: no cover - dependency-light hosts skip.
    pymunk = None


@unittest.skipIf(pymunk is None, "pymunk unavailable; geometry tests require real Chipmunk objects")
class NativeNavigationGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import native_navigation_geometry as geom

        cls.geom = geom
        cls.helper = geom.load_c_helper()
        cls.oracle_method = load_frozen_is_on_floor()

    def make_world(self, *, player_pos=(0, 0), ground1_pos=(4.5, 1), ground2_pos=(-4.5, 3),
                   platform_pos=(1, 0), player_angle=0.0, platform_angle=0.0):
        space = pymunk.Space()
        player_body = pymunk.Body(1, 1e9)
        player_shape = pymunk.Poly.create_box(player_body, size=(0.87, 1.0))
        player_body.position = player_pos
        player_body.angle = player_angle
        ground1_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        ground1_shape = pymunk.Poly.create_box(ground1_body, size=(10, 1))
        ground1_body.position = ground1_pos
        ground2_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        ground2_shape = pymunk.Poly.create_box(ground2_body, size=(10, 1))
        ground2_body.position = ground2_pos
        platform_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        platform_shape = pymunk.Poly.create_box(platform_body, size=(2, 1))
        platform_body.position = platform_pos
        platform_body.angle = platform_angle
        space.add(player_body, player_shape, ground1_body, ground1_shape, ground2_body, ground2_shape, platform_body, platform_shape)
        objects = {
            "ground1": types.SimpleNamespace(body=ground1_body, shape=ground1_shape),
            "ground2": types.SimpleNamespace(body=ground2_body, shape=ground2_shape),
            "platform1": types.SimpleNamespace(body=platform_body, shape=platform_shape),
        }
        player = types.SimpleNamespace(body=player_body, shape=player_shape, env=types.SimpleNamespace(objects=objects, space=space))
        return player, objects

    def oracle(self, player):
        return bool(types.MethodType(type(self).oracle_method, player)())

    def native(self, player):
        return self.helper.is_on_floor(
            player.shape,
            player.body,
            player.env.objects["ground1"].shape,
            player.env.objects["ground1"].body,
            player.env.objects["ground2"].shape,
            player.env.objects["ground2"].body,
            player.env.objects["platform1"].shape,
            player.env.objects["platform1"].body,
        )

    def add_player_body(self, player, *, position):
        body = pymunk.Body(1, 1e9)
        shape = pymunk.Poly.create_box(body, size=(0.87, 1.0))
        body.position = position
        player.env.space.add(body, shape)
        return body, shape

    def test_helper_is_compiled_once_per_process(self):
        self.assertIs(self.geom.load_c_helper(), self.geom.load_c_helper())

    def test_prepared_binding_matches_frozen_ast_and_holds_strong_refs(self):
        player, _ = self.make_world(player_pos=(4.5, 1.0))
        binding = self.helper.prepare_is_on_floor(
            player.shape,
            player.body,
            player.env.objects["ground1"].shape,
            player.env.objects["ground1"].body,
            player.env.objects["ground2"].shape,
            player.env.objects["ground2"].body,
            player.env.objects["platform1"].shape,
            player.env.objects["platform1"].body,
        )
        self.assertIs(binding._strong_refs[0], player.shape)
        self.assertEqual(binding.is_on_floor(), self.oracle(player))
        player.body.position = (20.0, 20.0)
        self.assertEqual(binding.is_on_floor(), self.oracle(player))

    def test_patch_refreshes_cached_binding_when_reset_swaps_body_and_shape(self):
        player, _ = self.make_world(player_pos=(4.5, 1.0))
        original_body = player.body
        original_shape = player.shape
        player.is_on_floor = lambda: False
        with self.geom.patch_players_is_on_floor([player], helper=self.helper):
            self.assertTrue(player.is_on_floor())
            player.body, player.shape = self.add_player_body(player, position=(20.0, 20.0))
            self.assertIsNot(player.body, original_body)
            self.assertIsNot(player.shape, original_shape)
            self.assertFalse(player.is_on_floor())

    def test_restore_does_not_overwrite_foreign_hook_installed_during_patch(self):
        player, _ = self.make_world(player_pos=(4.5, 1.0))
        original = player.is_on_floor = lambda: False
        foreign = lambda: "foreign"
        with self.geom.patch_players_is_on_floor([player], helper=self.helper):
            self.assertTrue(player.is_on_floor())
            player.is_on_floor = foreign
        self.assertIs(player.is_on_floor, foreign)
        player.is_on_floor = original

    def test_matches_frozen_ast_for_random_real_pymunk_rotated_boxes(self):
        rng = random.Random(20260906)
        fixed = [
            ((4.5, 1.0), (4.5, 1.0), (-4.5, 3.0), (1.0, 0.0), 0.0, 0.0),
            ((4.5, 1.5001), (4.5, 1.0), (-4.5, 3.0), (1.0, 0.0), 0.01, 0.0),
            ((1.0, -0.01), (4.5, 1.0), (-4.5, 3.0), (1.0, 0.0), 0.0, 0.02),
            ((2.251, -0.01), (4.5, 1.0), (-4.5, 3.0), (1.0, 0.0), 0.0, 0.0),
        ]
        for case in fixed:
            player, _ = self.make_world(player_pos=case[0], ground1_pos=case[1], ground2_pos=case[2],
                                        platform_pos=case[3], player_angle=case[4], platform_angle=case[5])
            self.assertEqual(self.native(player), self.oracle(player), case)
        for index in range(500):
            player, _ = self.make_world(
                player_pos=(rng.uniform(-7, 7), rng.uniform(-1, 4)),
                ground1_pos=(4.5, 1.0),
                ground2_pos=(-4.5, 3.0),
                platform_pos=(rng.uniform(-1.5, 1.5), rng.uniform(-0.5, 2.5)),
                player_angle=rng.uniform(-0.2, 0.2),
                platform_angle=rng.uniform(-0.15, 0.15),
            )
            self.assertEqual(self.native(player), self.oracle(player), index)

    def test_cache_bb_side_effect_is_preserved_by_native_order(self):
        player, _ = self.make_world(player_pos=(20.0, 20.0))
        self.assertFalse(self.native(player))
        player.body.position = (4.5, 1.0)
        self.assertTrue(self.native(player))

    def test_patch_installs_restores_idempotently_and_restores_after_exception(self):
        player, _ = self.make_world(player_pos=(4.5, 1.0))
        original = player.is_on_floor = lambda: False
        with self.geom.patch_players_is_on_floor([player], helper=self.helper) as patch:
            self.assertTrue(player.is_on_floor())
            patch.attach()
            self.assertTrue(player.is_on_floor())
        self.assertIs(player.is_on_floor, original)

        with self.assertRaises(RuntimeError):
            with self.geom.patch_players_is_on_floor([player], helper=self.helper):
                raise RuntimeError("boom")
        self.assertIs(player.is_on_floor, original)

    def test_source_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="ai2-geom-source-") as tmp:
            bad = Path(tmp) / "environment.py"
            bad.write_text("definitely not frozen")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.geom.verify_source_hash(bad, expected_sha256=FROZEN_ENV_SHA256)

    def test_pymunk_or_chipmunk_version_mismatch_fails_before_borrowing_pointers(self):
        original_pymunk = self.geom.EXPECTED_PYMUNK_VERSION
        original_chipmunk = self.geom.EXPECTED_CHIPMUNK_VERSION
        try:
            self.geom.EXPECTED_PYMUNK_VERSION = "0.0.0"
            with self.assertRaisesRegex(RuntimeError, "Pymunk version"):
                self.geom.NavigationGeometryHelper(self.geom._compile_shared())
            self.geom.EXPECTED_PYMUNK_VERSION = original_pymunk
            self.geom.EXPECTED_CHIPMUNK_VERSION = "7.0.30"
            with self.assertRaisesRegex(RuntimeError, "Chipmunk version"):
                self.geom.NavigationGeometryHelper(self.geom._compile_shared())
        finally:
            self.geom.EXPECTED_PYMUNK_VERSION = original_pymunk
            self.geom.EXPECTED_CHIPMUNK_VERSION = original_chipmunk

    def test_rejects_missing_pointers_and_unsafe_locked_space(self):
        player, _ = self.make_world(player_pos=(4.5, 1.0))
        with self.assertRaisesRegex(ValueError, "required shape/body"):
            self.helper.is_on_floor(None, player.body, player.env.objects["ground1"].shape, player.env.objects["ground1"].body,
                                    player.env.objects["ground2"].shape, player.env.objects["ground2"].body,
                                    player.env.objects["platform1"].shape, player.env.objects["platform1"].body)
        player.body.space._locked = True
        with self.assertRaisesRegex(RuntimeError, "locked"):
            self.geom.patch_players_is_on_floor([player], helper=self.helper).__enter__()


if __name__ == "__main__":
    unittest.main()
