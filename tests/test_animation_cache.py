"""Regression tests for shared animation asset decoding."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="ai2-mpl-"))

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "UTMIST-AI2-main"))

import pygame
from PIL import Image, ImageSequence


pygame.display.init()
pygame.display.set_mode((1, 1))

env_mod = importlib.import_module("environment.environment")


def _clear_animation_cache() -> None:
    loader = getattr(env_mod, "_load_animation_assets", None)
    if loader is not None and hasattr(loader, "cache_clear"):
        loader.cache_clear()


def _make_gif(path: Path, specs: list[tuple[tuple[int, int, int, int], int]]) -> None:
    frames = []
    durations = []
    for color, duration in specs:
        image = Image.new("RGBA", (3, 2), color)
        frames.append(image)
        durations.append(duration)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )


def _decode_reference(path: Path) -> tuple[list[bytes], list[int]]:
    pixels = []
    durations = []
    with Image.open(path) as gif:
        for frame in ImageSequence.Iterator(gif):
            rgba = frame.convert("RGBA")
            pixels.append(rgba.tobytes())
            durations.append(frame.info.get("duration", 100))
    return pixels, durations


def _surface_bytes(surface: pygame.Surface) -> bytes:
    return pygame.image.tostring(surface, "RGBA")


class DummyCamera:
    def __init__(self) -> None:
        self.canvas = pygame.Surface((32, 32), pygame.SRCALPHA)

    def gtp(self, pos):
        return (16, 16)

    def scale_gtp(self):
        return 10


class FakeWeapon:
    def __init__(self) -> None:
        self.name = "Spear"
        self.active = False
        self.world_pos = [0.0, 0.0]
        self.image = pygame.Surface((8, 8), pygame.SRCALPHA)

    def activate(self, camera, world_pos, current_frame):
        self.active = True
        self.world_pos = list(world_pos)
        self.spawn_frame = current_frame

    def frames_alive(self, current_frame):
        return current_frame - self.spawn_frame

    def deactivate(self):
        self.active = False


class FakePool:
    def __init__(self) -> None:
        self.weapon = FakeWeapon()

    def get_weapon(self, env, name, active):
        return self.weapon

    def return_weapon(self, weapon):
        weapon.deactivate()


class AnimationCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        _clear_animation_cache()
        self.tmp = tempfile.TemporaryDirectory(prefix="ai2-animation-cache-")
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()
        _clear_animation_cache()

    def _sprite(self, fps: int = 30) -> env_mod.AnimationSprite2D:
        sprite = env_mod.AnimationSprite2D(
            camera=None,
            scale=1.0,
            animation_folder=str(self.tmp_path),
            agent_id=0,
        )
        sprite.ENV_FPS = fps
        return sprite

    def test_two_sprites_decode_same_file_once_and_match_reference(self):
        gif_path = self.tmp_path / "move.gif"
        _make_gif(gif_path, [((255, 0, 0, 255), 80), ((0, 255, 0, 255), 140)])
        expected_pixels, expected_durations = _decode_reference(gif_path)
        real_open = env_mod.Image.open

        with mock.patch.object(env_mod.Image, "open", wraps=real_open) as open_mock:
            first = self._sprite().load_animation(str(gif_path))
            second = self._sprite().load_animation(str(gif_path))

        self.assertEqual(open_mock.call_count, 1)
        self.assertEqual(first.frame_durations, expected_durations)
        self.assertEqual(second.frame_durations, expected_durations)
        self.assertEqual([_surface_bytes(frame) for frame in first.frames], expected_pixels)
        self.assertEqual([_surface_bytes(frame) for frame in second.frames], expected_pixels)
        self.assertEqual(first.frames_per_step, [2, 4])
        self.assertEqual(second.frames_per_step, [2, 4])

    def test_cached_animations_share_read_only_surfaces_but_not_metadata_lists_or_playback(self):
        gif_path = self.tmp_path / "walk.gif"
        _make_gif(gif_path, [((10, 20, 30, 255), 100), ((40, 50, 60, 255), 100)])

        first_sprite = self._sprite()
        second_sprite = self._sprite()
        first = first_sprite.load_animation(str(gif_path))
        second = second_sprite.load_animation(str(gif_path))

        self.assertIsNot(first.frames, second.frames)
        self.assertIsNot(first.frame_durations, second.frame_durations)
        self.assertIsNot(first.frames_per_step, second.frames_per_step)
        self.assertIs(first.frames[0], second.frames[0])

        first.frame_durations.append(999)
        first.frames_per_step.append(999)
        self.assertEqual(second.frame_durations, [100, 100])
        self.assertEqual(second.frames_per_step, [3, 3])

        first_sprite.animations["walk"] = first
        second_sprite.animations["walk"] = second
        first_sprite.play("walk")
        second_sprite.play("walk")
        first_sprite.process((0, 0))
        first_sprite.process((0, 0))
        first_sprite.process((0, 0))
        self.assertEqual(first_sprite.current_frame_index, 1)
        self.assertEqual(second_sprite.current_frame_index, 0)

    def test_realpath_alias_reuses_cached_decode(self):
        gif_path = self.tmp_path / "alias.gif"
        alias_path = os.path.join(self.tmp.name, ".", "alias.gif")
        _make_gif(gif_path, [((1, 2, 3, 255), 100), ((4, 5, 6, 255), 100)])
        real_open = env_mod.Image.open

        with mock.patch.object(env_mod.Image, "open", wraps=real_open) as open_mock:
            self._sprite().load_animation(str(gif_path))
            self._sprite().load_animation(str(alias_path))

        self.assertEqual(open_mock.call_count, 1)

    def test_mtime_fingerprint_change_reloads_animation(self):
        gif_path = self.tmp_path / "changed.gif"
        _make_gif(gif_path, [((100, 0, 0, 255), 100)])
        first = self._sprite().load_animation(str(gif_path))
        first_bytes = _surface_bytes(first.frames[0])

        _make_gif(gif_path, [((0, 0, 200, 255), 200)])
        stat = gif_path.stat()
        os.utime(gif_path, ns=(stat.st_atime_ns + 2_000_000_000, stat.st_mtime_ns + 2_000_000_000))
        real_open = env_mod.Image.open

        with mock.patch.object(env_mod.Image, "open", wraps=real_open) as open_mock:
            second = self._sprite().load_animation(str(gif_path))

        self.assertEqual(open_mock.call_count, 1)
        self.assertNotEqual(_surface_bytes(second.frames[0]), first_bytes)
        self.assertEqual(second.frame_durations, [200])

    def test_frames_per_step_uses_sprite_fps_not_cache_key(self):
        gif_path = self.tmp_path / "timing.gif"
        _make_gif(gif_path, [((9, 8, 7, 255), 100), ((6, 5, 4, 255), 250)])

        low_fps = self._sprite(fps=20).load_animation(str(gif_path))
        high_fps = self._sprite(fps=60).load_animation(str(gif_path))

        self.assertEqual(low_fps.frame_durations, [100, 250])
        self.assertEqual(high_fps.frame_durations, [100, 250])
        self.assertEqual(low_fps.frames_per_step, [2, 5])
        self.assertEqual(high_fps.frames_per_step, [6, 15])
        self.assertIs(low_fps.frames[0], high_fps.frames[0])

    def test_failed_decode_is_retried_not_cached(self):
        bad_path = self.tmp_path / "bad.gif"
        bad_path.write_bytes(b"not a gif")
        real_open = env_mod.Image.open

        with mock.patch.object(env_mod.Image, "open", wraps=real_open) as open_mock:
            with self.assertRaises(Exception):
                self._sprite().load_animation(str(bad_path))
            with self.assertRaises(Exception):
                self._sprite().load_animation(str(bad_path))

        self.assertEqual(open_mock.call_count, 2)

    def test_weapon_spawner_update_unlocks_pickup_at_spawn_animation_duration(self):
        vfx_dir = self.tmp_path / "vfx"
        vfx_dir.mkdir()
        _make_gif(vfx_dir / "spawn.gif", [((255, 0, 0, 255), 100), ((0, 255, 0, 255), 200)])
        _make_gif(vfx_dir / "idle.gif", [((0, 0, 255, 255), 100)])

        env = type("FakeEnv", (), {"objects": {}})()
        pool = FakePool()
        camera = DummyCamera()
        with mock.patch.object(env_mod, "_get_asset_path", return_value=str(vfx_dir)):
            spawner = env_mod.WeaponSpawner(camera, 0, env, pool, [1.0, 0.7], cooldown_frames=0, despawn_frames=100)

        spawner.last_spawn_frame = -1
        spawner.update(0, 2)
        spawn_steps = spawner.vfx._steps("spawn")
        self.assertEqual(spawn_steps, 9)
        self.assertFalse(spawner.flag)

        spawner.update(spawn_steps - 1, 2)
        self.assertFalse(spawner.flag)
        spawner.update(spawn_steps, 2)
        self.assertTrue(spawner.flag)

    def test_warmed_warehouse_brawl_reset_does_not_reopen_cached_vfx_gifs(self):
        game = env_mod.WarehouseBrawl()
        real_open = env_mod.Image.open

        with mock.patch.object(env_mod.Image, "open", wraps=real_open) as open_mock:
            game.reset()
            game.reset()

        self.assertEqual(open_mock.call_count, 0)

    def test_animation_render_leaves_source_surface_unchanged(self):
        source = pygame.Surface((4, 4), pygame.SRCALPHA)
        source.fill((11, 22, 33, 255))
        before = _surface_bytes(source)
        sprite = env_mod.AnimationSprite2D(DummyCamera(), 1.0, str(self.tmp_path), 0)
        sprite.loaded = True
        sprite.position = (0, 0)
        sprite.animations["idle"] = env_mod.Animation([source], [100], [3])
        sprite.play("idle")

        sprite.render(DummyCamera(), flipped=True)

        self.assertEqual(_surface_bytes(source), before)


if __name__ == "__main__":
    unittest.main()
