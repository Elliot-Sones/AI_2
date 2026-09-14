"""Host binding for raw full-game C/CUDA-compatible simulation.

The CPU reference is a correctness oracle and environment benchmark, not a
CPU policy-training path. No Python game executes inside step_generated().
"""

from __future__ import annotations

import ctypes as C
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import native_navigation_episode as nav

ROOT = Path(__file__).resolve().parent
D, INT32, U = C.c_double, C.c_int32, C.c_uint64


class PlayerEvent(C.Structure):
    _fields_ = [
        (n, INT32)
        for n in (
            "kind",
            "source",
            "target",
            "move",
            "power",
            "cast",
            "hitbox",
            "phase_player",
            "event_sequence",
        )
    ] + [("attack_id", U), ("amount", D)]


class Attack(C.Structure):
    _fields_ = (
        [
            (n, INT32)
            for n in (
                "move_idx",
                "power_idx",
                "cast_idx",
                "move_frame",
                "total_frame_count",
                "frames_into_recovery",
            )
        ]
        + [("cast_frame", INT32 * 9)]
        + [(n, INT32) for n in ("cooldown", "move_facing", "first_hit_agent")]
        + [("all_hit_mask", C.c_uint32), ("power_hit_mask", C.c_uint32)]
        + [
            (n, INT32)
            for n in (
                "valid",
                "hit_anyone",
                "is_switching_casts",
                "in_recovery",
                "dealt_target_exists",
                "in_recovery_present",
                "dealt_target_components",
                "past_point_count",
            )
        ]
        + [
            (n, D)
            for n in (
                "dealt_target_x",
                "dealt_target_y",
                "dealt_target_mult",
                "past_point_x",
                "past_point_y",
            )
        ]
        + [("attack_id", U)]
    )


class CombatPlayer(C.Structure):
    _fields_ = (
        [
            (n, D)
            for n in (
                "damage_taken_this_stock",
                "damage_taken_total",
                "damage_taken_this_frame",
                "damage_done",
                "hurtbox_x",
                "hurtbox_y",
                "hurtbox_width",
                "hurtbox_height",
                "hurtbox_offset_x",
                "hurtbox_offset_y",
                "hurtbox_size_x",
                "hurtbox_size_y",
            )
        ]
        + [
            (n, INT32)
            for n in (
                "hurtbox_attributes_present",
                "gravity_scale",
                "just_got_hit",
                "stun_frames",
                "taunt_timer",
                "taunt_seed",
                "attack_seed",
            )
        ]
        + [("attribute_presence", C.c_uint32), ("attack", Attack)]
        + [
            (n, U)
            for n in (
                "attack_starts",
                "attack_contacts",
                "damage_events",
                "hit_during_stun",
            )
        ]
        + [
            ("last_damage_source", INT32),
            ("last_damage_attack_id", U),
            ("event_count", INT32),
            ("event_overflow", INT32),
            ("total_tick_events", INT32),
            ("events", PlayerEvent * 32),
        ]
    )


class WeaponAux(C.Structure):
    _fields_ = [
        (n, D) for n in ("hurtbox_x", "hurtbox_y", "hurtbox_width", "hurtbox_height")
    ] + [("pickup_lock_until", INT32), ("pad", INT32)]


class Weapon(C.Structure):
    _fields_ = [("world_pos", D * 2)] + [
        (n, INT32)
        for n in ("weapon", "active", "body", "shape", "spawn_frame", "physics_on")
    ]


class Spawner(C.Structure):
    _fields_ = [("world_pos", D * 2)] + [
        (n, INT32)
        for n in (
            "id",
            "kind",
            "og_cooldown_frames",
            "cooldown_frames",
            "last_spawn_frame",
            "despawn_frames",
            "weapon_slot",
            "flag",
            "spawned_once",
            "done",
            "flipped",
            "weapon",
        )
    ]


class WeaponEvent(C.Structure):
    _fields_ = [(n, INT32) for n in ("kind", "agent", "weapon", "spawner_id")]


class Weapons(C.Structure):
    _fields_ = (
        [("px_per_world", D)]
        + [
            (n, INT32)
            for n in (
                "initialized",
                "next_spawner_id",
                "spawner_count",
                "weapon_count",
                "pool_count",
                "event_count",
                "last_error",
                "pad",
            )
        ]
        + [
            ("spawners", Spawner * 8),
            ("weapons", Weapon * 8),
            ("pool", INT32 * 8),
            ("events", WeaponEvent * 8),
        ]
    )


class State(C.Structure):
    _fields_ = [
        ("scene", nav.Episode),
        ("combat", CombatPlayer * 2),
        ("weapons", Weapons),
        ("weapon_aux", WeaponAux * 2),
    ] + [(n, INT32) for n in ("initialized", "error", "win_mask", "max_timesteps")]


class Result(C.Structure):
    _fields_ = [("obs", (D * 64) * 2), ("reward", D * 2)] + [
        (n, INT32) for n in ("terminated", "truncated", "error", "win_mask")
    ]


class PlatformTable(C.Structure):
    _fields_ = (
        [("count", INT32), ("pad", INT32)]
        + [(n, D * 9001) for n in ("px", "py", "vx", "vy")]
        + [("dir_after", INT32 * 9001)]
    )


def compile_reference(build_dir: Path | None = None) -> Path:
    """Content-addressed artifact prevents stale in-process ctypes library reuse."""
    sources = sorted((ROOT / "native").glob("*.h")) + [
        ROOT / "native/game_episode.c",
        ROOT / "native/navigation_episode.c",
    ]
    contract = nav.contact.default_contraction()
    compiler = os.environ.get("CC", "cc")
    digest = hashlib.sha256((contract + compiler + platform.platform()).encode())
    for path in sources:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    folder = build_dir or ROOT / ".native-build"
    folder.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = folder / f"libgame-{digest.hexdigest()[:20]}{suffix}"
    if library.is_file():
        return library
    with tempfile.TemporaryDirectory(prefix="game-build-", dir=folder) as temporary:
        candidate = Path(temporary) / library.name
        command = [
            compiler,
            "-std=c99",
            "-O2",
            f"-ffp-contract={contract}",
            "-fno-fast-math",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-shared",
            "-fPIC",
            str(ROOT / "native/game_episode.c"),
            str(ROOT / "native/navigation_episode.c"),
            "-lm",
            "-o",
            str(candidate),
        ]
        subprocess.run(command, check=True)
        os.replace(candidate, library)
    return library


class GameReference:
    def __init__(self, library: Path | None = None):
        self.library = library or compile_reference()
        self.nav = nav.EpisodeReference(self.library)
        self.lib = self.nav.lib
        lib = self.lib
        for name, cls in (
            ("state", State),
            ("result", Result),
            ("table", PlatformTable),
            ("combat", CombatPlayer),
            ("weapons", Weapons),
        ):
            fn = getattr(lib, f"ai2g_{name}_size")
            fn.restype = C.c_size_t
            if fn() != C.sizeof(cls):
                raise RuntimeError(
                    f"full-game ABI size mismatch for {name}: native={fn()}, python={C.sizeof(cls)}"
                )
        lib.ai2g_state_offset.argtypes = [INT32]
        lib.ai2g_state_offset.restype = C.c_size_t
        for index, name in enumerate(
            (
                "scene",
                "combat",
                "weapons",
                "weapon_aux",
                "initialized",
                "error",
                "win_mask",
            )
        ):
            if lib.ai2g_state_offset(index) != getattr(State, name).offset:
                raise RuntimeError(f"full-game ABI offset mismatch: {name}")
        lib.ai2g_reset_export.argtypes = [
            C.POINTER(State),
            C.POINTER(nav.Config),
            C.POINTER(nav.Rng),
            D,
            C.POINTER(Result),
        ]
        lib.ai2g_reset_export.restype = INT32
        lib.ai2g_step_export.argtypes = [
            C.POINTER(State),
            C.POINTER(D),
            C.POINTER(nav.Rng),
            C.POINTER(PlatformTable),
            C.POINTER(Result),
        ]
        lib.ai2g_step_export.restype = INT32
        lib.ai2g_observe_export.argtypes = [C.POINTER(State), C.POINTER(D)]
        lib.ai2g_observe_export.restype = INT32
        lib.ai2g_table_build.argtypes = [C.POINTER(PlatformTable), D]
        lib.ai2g_table_build.restype = None
        lib.ai2g_place_player.argtypes = [C.POINTER(State), INT32, D, D, D, D]
        lib.ai2g_place_player.restype = INT32
        lib.ai2g_fixture_player_position.argtypes = [C.POINTER(State), INT32, D, D]
        lib.ai2g_fixture_player_position.restype = INT32
        self.cfg = self.nav.make_config(
            {
                "completion_mode": "horizon",
                "control_mode": "full",
                "max_targets": 1,
                "time_limit": 9000,
                "settle_steps": 1,
                "post_goal_required_steps": 1,
                "settle_speed": 1.0,
                "fall_penalty": 0.0,
                "min_dist": 0.0,
                "max_dist": 0.0,
            }
        )
        self.state = State()
        self.rng = nav.Rng()
        self.table = PlatformTable()
        lib.ai2g_table_build(C.byref(self.table), self.cfg.dt)

    @property
    def scene(self):
        return self.state.scene

    @property
    def combat(self):
        return self.state.combat

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"native full-game error {code}")

    def reset_generated(self, seed: int, lane: int = 0, pixels_per_world: float = 48.0):
        self.lib.ai2n_rng_seed_export(C.byref(self.rng), seed, lane)
        result = Result()
        self._check(
            self.lib.ai2g_reset_export(
                C.byref(self.state),
                C.byref(self.cfg),
                C.byref(self.rng),
                pixels_per_world,
                C.byref(result),
            )
        )
        return result

    def observe(self):
        out = np.empty((2, 64), dtype=np.float64)
        self._check(
            self.lib.ai2g_observe_export(
                C.byref(self.state), out.ctypes.data_as(C.POINTER(D))
            )
        )
        return out

    def place_player(self, player, x, y, vx=0.0, vy=0.0):
        self._check(
            self.lib.ai2g_place_player(C.byref(self.state), player, x, y, vx, vy)
        )

    def fixture_player_position(self, player, x, y):
        """Explicit reset fixture; does not overwrite previous-position history."""
        self._check(
            self.lib.ai2g_fixture_player_position(C.byref(self.state), player, x, y)
        )

    def step_generated(self, actions):
        action = np.asarray(actions, dtype=np.float64)
        if action.shape != (2, 10) or not np.isfinite(action).all():
            raise ValueError("actions must have shape (2,10) and finite values")
        action = np.ascontiguousarray(action)
        result = Result()
        self._check(
            self.lib.ai2g_step_export(
                C.byref(self.state),
                action.ctypes.data_as(C.POINTER(D)),
                C.byref(self.rng),
                C.byref(self.table),
                C.byref(result),
            )
        )
        return result
