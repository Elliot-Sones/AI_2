#!/usr/bin/env python3
"""Python driver for the closed-loop native navigation tick (native/navigation_episode.h).

Replays a capture written by ``navigation_episode_capture.py`` through the C
reference with no per-tick forcing: each episode is reset from the recorded
reset draws and then stepped with the recorded actions and the recorded random
draws only.  Every wrapper output (observation bits, reward bits, terminated,
truncated, navigation info) and every hidden-state snapshot field (player
state machines, platform, spawners, wrapper counters, Chipmunk bodies keyed by
shape hashid, broadphase leaves, arbiter cache) is compared bit for bit with the
real game.  The first divergent tick and field are reported per episode.

This validates the closed loop on the CPU host it runs on.  It is not a
throughput claim and not the GPU-resident environment.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
NATIVE_DIR = ROOT / "native"
if sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))

import native_navigation_contact as contact  # noqa: E402

SOURCE_FILES = tuple(NATIVE_DIR / name for name in (
    "navigation_episode.h", "navigation_episode.c", "navigation_input.h", "navigation_movement.h", "navigation_platform.h",
    "navigation_contact.h", "navigation_observation.h", "navigation_positions.h"))
MOVEMENT_STATE_COUNT = 13
SLOT_FIELD_COUNT = 14
INPUT_WIDTH = 35
OBS_WIDTH = 64
SLOT_NAMES = {1: "invincible_timer", 2: "dodge_cooldown", 3: "stun_time_stored", 4: "jump_timer", 5: "jumps_left", 6: "recoveries_left",
              7: "is_base", 8: "turnaround_timer", 9: "dash_timer", 10: "backdash_timer", 11: "dodge_timer", 12: "is_grounded", 13: "timer"}
WEAPON_NAMES = {0: "Spear", 1: "Hammer"}
NAV_INFO_KEYS = ("nav_touched", "nav_reached", "nav_fall", "nav_success", "nav_terminal_horizon", "nav_hold_steps", "nav_post_goal_steps")


class MovementSlot(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int32 * SLOT_FIELD_COUNT), ("present", ctypes.c_uint8 * SLOT_FIELD_COUNT)]


class MovementState(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("vx", ctypes.c_double), ("vy", ctypes.c_double),
                ("start_x", ctypes.c_double), ("start_y", ctypes.c_double), ("damage_vx", ctypes.c_double), ("damage_vy", ctypes.c_double),
                ("target_vx", ctypes.c_double), ("target_vy", ctypes.c_double), ("prev_x", ctypes.c_double), ("prev_y", ctypes.c_double),
                ("facing", ctypes.c_int32), ("state_id", ctypes.c_int32), ("stocks", ctypes.c_int32), ("velocity_func", ctypes.c_int32),
                ("body_type", ctypes.c_int32), ("bank", MovementSlot * MOVEMENT_STATE_COUNT)]


class MovementConfig(ctypes.Structure):
    _fields_ = [("stage_width_tiles", ctypes.c_double), ("stage_height_tiles", ctypes.c_double), ("move_speed", ctypes.c_double),
                ("jump_speed", ctypes.c_double), ("max_fall_speed", ctypes.c_double), ("fast_fall_ease", ctypes.c_double),
                ("in_air_ease", ctypes.c_double), ("run_speed", ctypes.c_double), ("dash_speed", ctypes.c_double), ("backdash_speed", ctypes.c_double),
                ("turnaround_time", ctypes.c_int32), ("backdash_time", ctypes.c_int32), ("dodge_time", ctypes.c_int32),
                ("grounded_dodge_cooldown", ctypes.c_int32), ("air_dodge_cooldown", ctypes.c_int32), ("invincible_time", ctypes.c_int32),
                ("jump_cooldown", ctypes.c_int32), ("dash_time", ctypes.c_int32)]


class Config(ctypes.Structure):
    _fields_ = [("completion_mode", ctypes.c_int32), ("control_mode", ctypes.c_int32), ("max_targets", ctypes.c_int32),
                ("time_limit_steps", ctypes.c_int32), ("settle_steps", ctypes.c_int32), ("post_goal_required_steps", ctypes.c_int32),
                ("pad", ctypes.c_int32 * 2), ("settle_speed", ctypes.c_double), ("fall_penalty", ctypes.c_double), ("min_dist", ctypes.c_double),
                ("max_dist", ctypes.c_double), ("dt", ctypes.c_double), ("collision_bias", ctypes.c_double), ("bias_coef", ctypes.c_double),
                ("damping_dt", ctypes.c_double)]


class Player(ctypes.Structure):
    _fields_ = [("ms", MovementState), ("input_state", ctypes.c_int32 * INPUT_WIDTH), ("body", ctypes.c_int32), ("shape", ctypes.c_int32),
                ("weapon", ctypes.c_int32), ("pad", ctypes.c_int32), ("damage", ctypes.c_double)]


class Spawner(ctypes.Structure):
    _fields_ = [("world_pos", ctypes.c_double * 2), ("og_cooldown_frames", ctypes.c_int32), ("cooldown_frames", ctypes.c_int32),
                ("last_spawn_frame", ctypes.c_int32), ("active", ctypes.c_int32), ("weapon_name", ctypes.c_int32), ("spawn_frame", ctypes.c_int32),
                ("body", ctypes.c_int32), ("shape", ctypes.c_int32), ("flag", ctypes.c_int32), ("pad", ctypes.c_int32)]


class Wrapper(ctypes.Structure):
    _fields_ = [("target_x", ctypes.c_double), ("target_y", ctypes.c_double), ("last_distance", ctypes.c_double), ("target_valid", ctypes.c_int32),
                ("last_distance_valid", ctypes.c_int32), ("episode_steps", ctypes.c_int32), ("targets_this_episode", ctypes.c_int32),
                ("hold_steps", ctypes.c_int32), ("post_goal_steps", ctypes.c_int32), ("goal_reached_step", ctypes.c_int32),
                ("goal_reached_valid", ctypes.c_int32), ("last_stocks", ctypes.c_int32), ("fell_this_episode", ctypes.c_int32),
                ("target_touched", ctypes.c_int32), ("targets_reached", ctypes.c_int32), ("falls", ctypes.c_int32), ("pad", ctypes.c_int32)]


class StepResult(ctypes.Structure):
    _fields_ = [("obs", ctypes.c_float * OBS_WIDTH), ("reward", ctypes.c_double), ("nav_reward", ctypes.c_double), ("terminated", ctypes.c_int32),
                ("truncated", ctypes.c_int32), ("underlying_done", ctypes.c_int32), ("touched", ctypes.c_int32), ("reached", ctypes.c_int32),
                ("fall", ctypes.c_int32), ("handoff", ctypes.c_int32), ("success", ctypes.c_int32), ("horizon", ctypes.c_int32), ("error", ctypes.c_int32)]


class Episode(ctypes.Structure):
    _fields_ = [("world", contact.World), ("mcfg", MovementConfig), ("cfg", Config), ("players", Player * 2), ("spawners", Spawner * 2),
                ("wrap", Wrapper), ("waypoint1", ctypes.c_double * 2), ("waypoint2", ctypes.c_double * 2), ("platform_body", ctypes.c_int32),
                ("platform_shape", ctypes.c_int32), ("moving_to_w2", ctypes.c_int32), ("steps", ctypes.c_int32), ("raw_terminated", ctypes.c_int32),
                ("raw_truncated", ctypes.c_int32), ("p1_right", ctypes.c_int32), ("shape_id_counter", ctypes.c_int32), ("last_error", ctypes.c_int32),
                ("pad", ctypes.c_int32 * 3), ("obs", ctypes.c_float * OBS_WIDTH)]


PLATFORM_TABLE_SIZE = 1024


class PlatformTable(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int32), ("pad", ctypes.c_int32), ("px", ctypes.c_double * PLATFORM_TABLE_SIZE), ("py", ctypes.c_double * PLATFORM_TABLE_SIZE),
                ("vx", ctypes.c_double * PLATFORM_TABLE_SIZE), ("vy", ctypes.c_double * PLATFORM_TABLE_SIZE), ("dir_after", ctypes.c_int32 * PLATFORM_TABLE_SIZE)]


class Draw(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32), ("pad", ctypes.c_int32), ("a", ctypes.c_double), ("b", ctypes.c_double), ("value", ctypes.c_double)]


class Rng(ctypes.Structure):
    _fields_ = [("mode", ctypes.c_int32), ("error", ctypes.c_int32), ("cursor", ctypes.c_int32), ("count", ctypes.c_int32),
                ("draws", ctypes.POINTER(Draw)), ("s", ctypes.c_ulonglong * 4), ("error_kind", ctypes.c_int32), ("error_cursor", ctypes.c_int32),
                ("error_a", ctypes.c_double), ("error_b", ctypes.c_double)]


EXPECTED_OFFSETS = (
    Episode.world.offset, Episode.mcfg.offset, Episode.cfg.offset, Episode.players.offset, Episode.spawners.offset, Episode.wrap.offset,
    Episode.waypoint1.offset, Episode.platform_body.offset, Episode.steps.offset, Episode.shape_id_counter.offset, Episode.obs.offset,
    Player.input_state.offset, Player.body.offset, Player.damage.offset, Wrapper.target_valid.offset, Wrapper.falls.offset,
    StepResult.reward.offset, StepResult.error.offset, Rng.draws.offset, Rng.s.offset, Rng.error_b.offset, Config.settle_speed.offset,
    Config.damping_dt.offset,
)


def source_hashes() -> dict[str, str]:
    return {path.name: contact.sha256_file(path) for path in SOURCE_FILES}


def _up_to_date(library: Path) -> bool:
    if not library.is_file():
        return False
    newest = max(path.stat().st_mtime for path in SOURCE_FILES if path.is_file())
    return library.stat().st_mtime > newest and library.stat().st_size > 0


def _atomic_build(library: Path, command: list[str]) -> Path:
    """Compile to a private temporary name and rename, so parallel verifiers never load a half-written library."""
    if _up_to_date(library):
        return library
    library.parent.mkdir(parents=True, exist_ok=True)
    temp = library.with_name(f".{library.name}.{os.getpid()}.tmp")
    subprocess.run(command + ["-o", str(temp)], check=True, cwd=str(NATIVE_DIR))
    os.replace(temp, library)
    return library


def compile_cpu_reference(build_dir: Path, contract: str | None = None) -> Path:
    build_dir = Path(build_dir)
    contract = contract or contact.default_contraction()
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = build_dir / f"libnavigation_episode-{contract}{suffix}"
    compiler = os.environ.get("CC") or "cc"
    command = [compiler, "-std=c99", "-O2", f"-ffp-contract={contract}", "-fno-fast-math", "-Wall", "-Wextra", "-Werror", "-pedantic",
               "-shared", "-fPIC", str(NATIVE_DIR / "navigation_episode.c")]
    return _atomic_build(library, command)


class EpisodeReference:
    def __init__(self, library_path: Path) -> None:
        self.lib = ctypes.CDLL(str(library_path))
        lib = self.lib
        for name in ("ai2n_episode_size", "ai2n_result_size", "ai2n_draw_size", "ai2n_rng_size", "ai2n_config_size", "ai2n_player_size",
                     "ai2n_spawner_size", "ai2n_wrapper_size", "ai2n_movement_state_size", "ai2n_movement_config_size", "ai2n_positions_count"):
            getattr(lib, name).restype = ctypes.c_int
        lib.ai2n_positions_sha256.restype = ctypes.c_char_p
        lib.ai2n_offsets.restype = ctypes.c_int
        lib.ai2n_offsets.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.ai2n_config_finalize.restype = ctypes.c_int
        lib.ai2n_config_finalize.argtypes = [ctypes.POINTER(Config), ctypes.c_int]
        lib.ai2n_reset_export.restype = ctypes.c_int
        lib.ai2n_reset_export.argtypes = [ctypes.POINTER(Episode), ctypes.POINTER(Config), ctypes.POINTER(Rng)]
        lib.ai2n_step_export.restype = ctypes.c_int
        lib.ai2n_step_export.argtypes = [ctypes.POINTER(Episode), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(Rng), ctypes.POINTER(PlatformTable), ctypes.POINTER(StepResult)]
        lib.ai2n_platform_table_size.restype = ctypes.c_int
        lib.ai2n_platform_table_build_export.restype = None
        lib.ai2n_platform_table_build_export.argtypes = [ctypes.POINTER(PlatformTable), ctypes.c_double]
        if lib.ai2n_platform_table_size() != ctypes.sizeof(PlatformTable):
            raise RuntimeError("ctypes/C platform table size mismatch")
        self.table: PlatformTable | None = None
        lib.ai2n_observe_export.restype = ctypes.c_int
        lib.ai2n_observe_export.argtypes = [ctypes.POINTER(Episode), ctypes.POINTER(ctypes.c_float)]
        lib.ai2n_is_on_floor_export.restype = ctypes.c_int
        lib.ai2n_is_on_floor_export.argtypes = [ctypes.POINTER(Episode), ctypes.c_int]
        lib.ai2n_rng_replay_export.restype = None
        lib.ai2n_rng_replay_export.argtypes = [ctypes.POINTER(Rng), ctypes.POINTER(Draw), ctypes.c_int]
        lib.ai2n_rng_seed_export.restype = None
        lib.ai2n_rng_seed_export.argtypes = [ctypes.POINTER(Rng), ctypes.c_ulonglong, ctypes.c_ulonglong]
        lib.ai2n_rng_request_export.restype = ctypes.c_double
        lib.ai2n_rng_request_export.argtypes = [ctypes.POINTER(Rng), ctypes.c_int, ctypes.c_double, ctypes.c_double]
        expected = {"episode": ctypes.sizeof(Episode), "result": ctypes.sizeof(StepResult), "draw": ctypes.sizeof(Draw), "rng": ctypes.sizeof(Rng),
                    "config": ctypes.sizeof(Config), "player": ctypes.sizeof(Player), "spawner": ctypes.sizeof(Spawner), "wrapper": ctypes.sizeof(Wrapper),
                    "movement_state": ctypes.sizeof(MovementState), "movement_config": ctypes.sizeof(MovementConfig)}
        actual = {"episode": lib.ai2n_episode_size(), "result": lib.ai2n_result_size(), "draw": lib.ai2n_draw_size(), "rng": lib.ai2n_rng_size(),
                  "config": lib.ai2n_config_size(), "player": lib.ai2n_player_size(), "spawner": lib.ai2n_spawner_size(), "wrapper": lib.ai2n_wrapper_size(),
                  "movement_state": lib.ai2n_movement_state_size(), "movement_config": lib.ai2n_movement_config_size()}
        if expected != actual:
            raise RuntimeError(f"ctypes/C struct size mismatch: python {expected} c {actual}")
        buf = (ctypes.c_int * 64)()
        n = lib.ai2n_offsets(buf, 64)
        offsets = tuple(buf[i] for i in range(n))
        if offsets != EXPECTED_OFFSETS:
            raise RuntimeError(f"ctypes/C field offset mismatch: python {EXPECTED_OFFSETS} c {offsets}")
        self.positions_sha256 = lib.ai2n_positions_sha256().decode("ascii")
        self.positions_count = lib.ai2n_positions_count()

    def make_config(self, wrapper_cfg: dict[str, Any], *, dt: float = 1.0 / 30.0, collision_bias: float | None = None) -> Config:
        cfg = Config()
        cfg.completion_mode = 1 if wrapper_cfg["completion_mode"] == "sequence" else 0
        cfg.control_mode = 1 if wrapper_cfg["control_mode"] == "movement" else 0
        cfg.max_targets = int(wrapper_cfg["max_targets"])
        cfg.time_limit_steps = int(wrapper_cfg["time_limit"])
        cfg.settle_steps = int(wrapper_cfg["settle_steps"])
        cfg.post_goal_required_steps = int(wrapper_cfg["post_goal_required_steps"])
        cfg.settle_speed = float(wrapper_cfg["settle_speed"])
        cfg.fall_penalty = float(wrapper_cfg["fall_penalty"])
        cfg.min_dist = float(wrapper_cfg["min_dist"])
        cfg.max_dist = float(wrapper_cfg["max_dist"])
        cfg.dt = float(dt)
        if collision_bias is not None:
            cfg.collision_bias = float(collision_bias)
        rc = self.lib.ai2n_config_finalize(ctypes.byref(cfg), 1 if collision_bias is not None else 0)
        if rc != 0:
            raise RuntimeError(f"config finalize failed: {rc}")
        return cfg

    def replay_rng(self, draws: list[dict[str, Any]]) -> tuple[Rng, Any]:
        arr = (Draw * max(1, len(draws)))()
        for i, d in enumerate(draws):
            arr[i].kind = int(d["kind"]); arr[i].a = float(d["a"]); arr[i].b = float(d["b"]); arr[i].value = float(d["value"])
        rng = Rng()
        self.lib.ai2n_rng_replay_export(ctypes.byref(rng), arr, len(draws))
        return rng, arr

    def reset(self, ep: Episode, cfg: Config, rng: Rng) -> int:
        return int(self.lib.ai2n_reset_export(ctypes.byref(ep), ctypes.byref(cfg), ctypes.byref(rng)))

    def platform_table(self, dt: float = 1.0 / 30.0) -> PlatformTable:
        if self.table is None:
            table = PlatformTable()
            self.lib.ai2n_platform_table_build_export(ctypes.byref(table), float(dt))
            self.table = table
        return self.table

    def step(self, ep: Episode, action: list[float], rng: Rng, *, use_table: bool = True) -> tuple[int, StepResult]:
        arr = (ctypes.c_double * 10)(*[float(x) for x in action])
        out = StepResult()
        table = ctypes.byref(self.platform_table(ep.cfg.dt)) if use_table else None
        rc = int(self.lib.ai2n_step_export(ctypes.byref(ep), arr, ctypes.byref(rng), table, ctypes.byref(out)))
        return rc, out


def compile_cuda_library(build_dir: Path) -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is not available")
    build_dir = Path(build_dir)
    library = build_dir / "libnavigation_episode_cuda.so"
    command = [nvcc, "-O2", "-fmad=false", "--shared", "-Xcompiler", "-fPIC", "-Xcompiler", "-ffp-contract=off", str(NATIVE_DIR / "navigation_episode.cu")]
    return _atomic_build(library, command)


class CudaEpisodeReference:
    """Device lanes stepped by the same header (navigation_episode.cu)."""

    def __init__(self, library_path: Path) -> None:
        self.lib = ctypes.CDLL(str(library_path))
        lib = self.lib
        lib.ai2n_cuda_episode_size.restype = ctypes.c_int
        if lib.ai2n_cuda_episode_size() != ctypes.sizeof(Episode):
            raise RuntimeError("CUDA episode struct size differs from the Python mirror")
        lib.ai2n_cuda_release.restype = ctypes.c_int
        lib.ai2n_cuda_upload.restype = ctypes.c_int
        lib.ai2n_cuda_upload.argtypes = [ctypes.POINTER(Episode), ctypes.c_int]
        lib.ai2n_cuda_reset_replicated.restype = ctypes.c_int
        lib.ai2n_cuda_reset_replicated.argtypes = [ctypes.POINTER(Config), ctypes.POINTER(Draw), ctypes.c_int, ctypes.c_int, ctypes.POINTER(Episode),
                                                   ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        lib.ai2n_cuda_step.restype = ctypes.c_int
        lib.ai2n_cuda_step.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(Draw), ctypes.c_int, ctypes.c_int, ctypes.POINTER(Episode),
                                       ctypes.POINTER(StepResult), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                       ctypes.POINTER(ctypes.c_int)]
        lib.ai2n_cuda_bench.restype = ctypes.c_int
        lib.ai2n_cuda_bench.argtypes = [ctypes.POINTER(Config), ctypes.c_int, ctypes.c_int, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_int)]
        lib.ai2n_cuda_table_size.restype = ctypes.c_int
        if lib.ai2n_cuda_table_size() != ctypes.sizeof(PlatformTable):
            raise RuntimeError("CUDA platform table size differs from the Python mirror")
        lib.ai2n_cuda_set_table.restype = ctypes.c_int
        lib.ai2n_cuda_set_table.argtypes = [ctypes.POINTER(PlatformTable)]

    def set_table(self, table: PlatformTable) -> None:
        rc = self.lib.ai2n_cuda_set_table(ctypes.byref(table))
        if rc != 0:
            raise RuntimeError(f"cuda set_table failed: {rc}")

    @staticmethod
    def _draws(draws: list[dict[str, Any]]):
        arr = (Draw * max(1, len(draws)))()
        for i, d in enumerate(draws):
            arr[i].kind = int(d["kind"]); arr[i].a = float(d["a"]); arr[i].b = float(d["b"]); arr[i].value = float(d["value"])
        return arr

    def upload(self, ep: Episode, lanes: int) -> None:
        rc = self.lib.ai2n_cuda_upload(ctypes.byref(ep), lanes)
        if rc != 0:
            raise RuntimeError(f"cuda upload failed: {rc}")

    def reset_replicated(self, cfg: Config, draws: list[dict[str, Any]], lanes: int) -> tuple[Episode, dict[str, int]]:
        out = Episode()
        flags = [ctypes.c_int(0) for _ in range(4)]
        arr = self._draws(draws)
        rc = self.lib.ai2n_cuda_reset_replicated(ctypes.byref(cfg), arr, len(draws), lanes, ctypes.byref(out), *(ctypes.byref(f) for f in flags))
        if rc != 0:
            raise RuntimeError(f"cuda reset failed: {rc}")
        return out, {"all_equal": flags[0].value, "lane_error": flags[1].value, "rng_error": flags[2].value, "rng_cursor": flags[3].value}

    def step(self, action: list[float], draws: list[dict[str, Any]], lanes: int) -> tuple[Episode, StepResult, dict[str, int]]:
        out = Episode()
        result = StepResult()
        flags = [ctypes.c_int(0) for _ in range(4)]
        arr = self._draws(draws)
        act = (ctypes.c_double * 10)(*[float(x) for x in action])
        rc = self.lib.ai2n_cuda_step(act, arr, len(draws), lanes, ctypes.byref(out), ctypes.byref(result), *(ctypes.byref(f) for f in flags))
        if rc != 0:
            raise RuntimeError(f"cuda step failed: {rc}")
        return out, result, {"all_equal": flags[0].value, "lane_error": flags[1].value, "rng_error": flags[2].value, "rng_cursor": flags[3].value}

    def bench(self, cfg: Config, lanes: int, iterations: int, seed: int) -> dict[str, Any]:
        elapsed = ctypes.c_double(0.0)
        done = ctypes.c_ulonglong(0)
        err = ctypes.c_int(0)
        rc = self.lib.ai2n_cuda_bench(ctypes.byref(cfg), lanes, iterations, seed, ctypes.byref(elapsed), ctypes.byref(done), ctypes.byref(err))
        if rc != 0:
            raise RuntimeError(f"cuda bench failed: {rc}")
        steps = lanes * iterations
        return {"lanes": lanes, "iterations": iterations, "elapsed_ms": elapsed.value, "lane_steps_per_second": steps / (elapsed.value / 1000.0),
                "episodes_completed": int(done.value), "lane_error": err.value}

    def release(self) -> None:
        self.lib.ai2n_cuda_release()


def episode_bytes(ep: Episode) -> bytes:
    return ctypes.string_at(ctypes.byref(ep), ctypes.sizeof(Episode))


def result_bytes(result: StepResult) -> bytes:
    return ctypes.string_at(ctypes.byref(result), ctypes.sizeof(StepResult))


def result_fields(result: StepResult) -> dict[str, Any]:
    return {"obs": [float(result.obs[i]) for i in range(OBS_WIDTH)], "reward": result.reward, "nav_reward": result.nav_reward,
            "terminated": int(result.terminated), "truncated": int(result.truncated), "touched": int(result.touched), "reached": int(result.reached),
            "fall": int(result.fall), "handoff": int(result.handoff), "success": int(result.success), "horizon": int(result.horizon), "error": int(result.error)}


def device_diff(cpu_ep: Episode, dev_ep: Episode, cpu_res: "StepResult | None", dev_res: "StepResult | None") -> list[dict[str, Any]]:
    diffs = contact.diff_records(native_snapshot(cpu_ep), native_snapshot(dev_ep), "state")
    if cpu_res is not None and dev_res is not None:
        diffs = contact.diff_records(result_fields(cpu_res), result_fields(dev_res), "result") + diffs
    return diffs


def _field_group(field: str) -> str:
    parts = field.replace("[", ".").split(".")
    return ".".join(parts[:3]) if parts[0] == "state" else ".".join(parts[:2])


TOLERATED_GROUPS = ("result.reward", "result.nav_reward", "state.wrapper.last_distance")


def _strict(diffs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Differences that matter for physics or control flow (everything but last-bit reward and last_distance deviations)."""
    return [d for d in diffs if _field_group(d["field"]) not in TOLERATED_GROUPS]


def _float32(x: float) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


# ----------------------------------------------------------------------------- snapshots

def native_player(ep: Episode, i: int) -> dict[str, Any]:
    pl = ep.players[i]
    body = ep.world.bodies[pl.body]
    ms = pl.ms
    bank = {}
    for sid in range(MOVEMENT_STATE_COUNT):
        slot = ms.bank[sid]
        bank[str(sid)] = {SLOT_NAMES[f]: int(slot.value[f]) for f in range(1, SLOT_FIELD_COUNT) if slot.present[f]}
    return {"x": body.p.x, "y": body.p.y, "vx": body.v.x, "vy": body.v.y, "facing": int(ms.facing), "state_id": int(ms.state_id),
            "stocks": int(ms.stocks), "velocity_mode": int(body.velocity_mode), "damage": float(pl.damage), "weapon": WEAPON_NAMES.get(-1, "Punch"),
            "start": [ms.start_x, ms.start_y], "input": [int(pl.input_state[k]) for k in range(INPUT_WIDTH)], "bank": bank}


def native_snapshot(ep: Episode) -> dict[str, Any]:
    w = ep.world
    hashes = {k: int(w.shapes[k].hashid) for k in range(w.shape_count) if w.shape_active[k]}
    bodies = {}
    for k, h in hashes.items():
        shape = w.shapes[k]
        bodies[str(h)] = {"body": contact.body_fields(w.bodies[shape.body]),
                          "shape": {"verts": [[shape.v0[j].x, shape.v0[j].y] for j in range(shape.count)], "radius": shape.r, "elasticity": shape.e,
                                    "friction": shape.u, "collision_type": int(shape.collision_type), "sensor": bool(shape.sensor),
                                    "filter": {"group": int(shape.group), "categories": int(shape.categories), "mask": int(shape.mask)},
                                    "surface_velocity": [shape.surface_v.x, shape.surface_v.y]}}
    tree = contact.tree_fields(w)
    leaves = {}
    for key, leaf in tree["leaves"].items():
        leaves[str(hashes[int(key)])] = {"bb": leaf["bb"], "stamp": leaf["stamp"], "pairs": [[hashes.get(a, -2), hashes.get(b, -2), pid] for a, b, pid in leaf["pairs"]]}
    tree_out = {"stamp": tree["stamp"], "leaves": leaves, "order": [hashes[k] for k in tree["order"]]}

    def arbiter(arb: Any) -> dict[str, Any]:
        f = contact.arbiter_fields(arb)
        a, b = hashes.get(f["a"], -1), hashes.get(f["b"], -1)
        return {"a": a, "b": b, "e": f["e"], "u": f["u"], "surface_vr": f["surface_vr"], "n": f["n"], "count": f["count"], "swapped": f["swapped"],
                "stamp": f["stamp"], "state": f["state"], "contacts": f["contacts"]}

    cache = {}
    for j in range(w.arbiter_count):
        if w.arbiters[j].active:
            arb = arbiter(w.arbiters[j])
            cache[f"{min(arb['a'], arb['b'])}-{max(arb['a'], arb['b'])}"] = arb
    solve = [arbiter(w.arbiters[w.solve[j]]) for j in range(w.solve_count)]
    pb = w.bodies[ep.platform_body]
    spawners = []
    for i in range(2):
        sp = ep.spawners[i]
        entry: dict[str, Any] = {"world_pos": [sp.world_pos[0], sp.world_pos[1]], "og_cooldown_frames": int(sp.og_cooldown_frames),
                                 "cooldown_frames": int(sp.cooldown_frames), "last_spawn_frame": int(sp.last_spawn_frame), "active": bool(sp.active)}
        if sp.active:
            entry["weapon_name"] = WEAPON_NAMES[int(sp.weapon_name)]
            entry["spawn_frame"] = int(sp.spawn_frame)
            entry["hashid"] = int(w.shapes[sp.shape].hashid)
        spawners.append(entry)
    wr = ep.wrap
    return {
        "players": [native_player(ep, 0), native_player(ep, 1)],
        "platform": {"x": pb.p.x, "y": pb.p.y, "vx": pb.v.x, "vy": pb.v.y, "moving_to_w2": bool(ep.moving_to_w2),
                     "waypoint1": [ep.waypoint1[0], ep.waypoint1[1]], "waypoint2": [ep.waypoint2[0], ep.waypoint2[1]]},
        "spawners": spawners,
        "wrapper": {"target": [wr.target_x, wr.target_y] if wr.target_valid else None,
                    "last_distance": wr.last_distance if wr.last_distance_valid else None,
                    "episode_steps": int(wr.episode_steps), "targets_this_episode": int(wr.targets_this_episode), "hold_steps": int(wr.hold_steps),
                    "post_goal_steps": int(wr.post_goal_steps), "goal_reached_step": int(wr.goal_reached_step) if wr.goal_reached_valid else None,
                    "target_touched": bool(wr.target_touched), "fell_this_episode": bool(wr.fell_this_episode), "last_stocks": int(wr.last_stocks)},
        "raw": {"steps": int(ep.steps), "terminated": bool(ep.raw_terminated)},
        "world": {"bodies": bodies, "tree": tree_out, "cache": cache, "solve": solve, "stamp": int(w.stamp), "shape_id_counter": int(ep.shape_id_counter)},
    }


def _expected_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Project the recorded snapshot onto the fields the native lane models."""
    out = json.loads(json.dumps(state))
    for player in out["players"]:
        player.pop("state", None)
        player["velocity_mode"] = int(player["velocity_mode"])
    for entry in out["world"]["bodies"].values():
        entry.pop("label", None)
        body = entry["body"]
        entry["body"] = {k: body[k] for k in ("p", "v", "f", "a", "w", "t", "transform", "v_bias", "w_bias")}
    return out


def _actual_snapshot(ep: Episode) -> dict[str, Any]:
    snap = native_snapshot(ep)
    for player in snap["players"]:
        player.pop("weapon", None)
    return snap


def _expected_players_strip(expected: dict[str, Any]) -> None:
    for player in expected["players"]:
        player.pop("weapon", None)


def float32_bits(values: list[float]) -> list[int]:
    import struct

    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def compare_outputs(expected: dict[str, Any], actual: StepResult) -> list[dict[str, Any]]:
    mismatches = []
    exp_obs = float32_bits(expected["obs"])
    act_obs = float32_bits([actual.obs[i] for i in range(OBS_WIDTH)])
    for i, (e, a) in enumerate(zip(exp_obs, act_obs)):
        if e != a:
            mismatches.append({"field": f"obs[{i}]", "expected": expected["obs"][i], "actual": float(actual.obs[i]), "kind": "float32"})
    if contact.bits(expected["reward"]) != contact.bits(actual.reward):
        mismatches.append({"field": "reward", "expected": expected["reward"], "actual": actual.reward, "kind": "float",
                           "abs_diff": abs(expected["reward"] - actual.reward)})
    for key, value in (("terminated", actual.terminated), ("truncated", actual.truncated)):
        if bool(expected[key]) != bool(value):
            mismatches.append({"field": key, "expected": expected[key], "actual": bool(value), "kind": "value"})
    info = expected.get("info", {})
    actual_info = {"nav_touched": actual.touched, "nav_reached": actual.reached, "nav_fall": actual.fall, "nav_success": actual.success,
                   "nav_terminal_horizon": actual.horizon}
    for key, value in actual_info.items():
        if int(info.get(key, 0)) != int(value):
            mismatches.append({"field": f"info.{key}", "expected": info.get(key, 0), "actual": int(value), "kind": "value"})
    return mismatches


def compare_state(expected_state: dict[str, Any], ep: Episode) -> list[dict[str, Any]]:
    expected = _expected_snapshot(expected_state)
    _expected_players_strip(expected)
    actual = _actual_snapshot(ep)
    # The recorded info counters are compared through outputs; wrapper holds are compared here.
    return contact.diff_records(expected, actual, "state")


def load_capture(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = None
    episodes: list[dict[str, Any]] = []
    with contact.open_capture(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "metadata" in row:
                metadata = row["metadata"]
            elif "header" in row:
                episodes.append({"header": row["header"], "ticks": []})
            elif "tick" in row:
                if not episodes:
                    raise ValueError("tick before any episode header")
                episodes[-1]["ticks"].append(row["tick"])
    if metadata is None or not episodes:
        raise ValueError("capture is missing metadata or episodes")
    return metadata, episodes


def verify_capture(ref: EpisodeReference, capture_path: Path, *, max_mismatch_ticks: int = 25, cuda: "CudaEpisodeReference | None" = None,
                   cuda_lanes: int = 256, cuda_mode: str = "per-tick") -> dict[str, Any]:
    metadata, episodes = load_capture(Path(capture_path))
    if metadata["source"].get("positions_sha256") != ref.positions_sha256:
        raise RuntimeError("positions.json of the capture differs from the compiled AI2N_POSITIONS table")
    wrapper_cfg = metadata["wrapper"]
    compared = 0
    exact_episodes = 0
    mismatch_ticks: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    episode_reports = []
    draws_total = 0
    cuda_stats: dict[str, Any] | None = None
    if cuda is not None:
        if cuda_mode not in ("per-tick", "resident"):
            raise ValueError("cuda_mode must be per-tick or resident")
        cuda_stats = {"mode": cuda_mode, "lanes": cuda_lanes, "ticks_compared": 0, "resets_compared": 0, "device_equal_ticks": 0, "device_mismatch_ticks": 0,
                      "strict_mismatch_ticks": 0, "tolerated_only_ticks": 0, "reward_float32_mismatch_ticks": 0, "reward_max_abs_diff": 0.0,
                      "lanes_identical_every_time": True, "lane_error_max": 0, "rng_error_ticks": 0, "mismatch_field_counts": {}, "first_mismatch": None,
                      "first_strict_mismatch": None, "resident_first_divergence": [],
                      "tolerated_groups": list(TOLERATED_GROUPS),
                      "note": "strict = physics, state machines, spawners, wrapper counters, observations, done flags; tolerated = last-bit double deviations of reward and last_distance from the device square (x*x) versus host libm pow"}
        cuda.set_table(ref.platform_table())
    for entry in episodes:
        header = entry["header"]
        seed, index = header["seed"], header["episode"]
        cfg = ref.make_config(wrapper_cfg, dt=float(header["space"]["dt"]), collision_bias=float(header["space"]["collision_bias"]))
        ep = Episode()
        rng, keep = ref.replay_rng(header["draws"])
        rc = ref.reset(ep, cfg, rng)
        report: dict[str, Any] = {"seed": seed, "episode": index, "ticks": len(entry["ticks"]), "compared": 0, "exact": True, "first": None}
        problems: list[dict[str, Any]] = []
        if rc != 0:
            problems.append({"field": "reset.rc", "expected": 0, "actual": rc, "kind": "value", "rng_error": int(rng.error), "rng_cursor": int(rng.cursor)})
        elif rng.cursor != rng.count:
            problems.append({"field": "reset.draws_consumed", "expected": int(rng.count), "actual": int(rng.cursor), "kind": "value"})
        else:
            obs_bits = float32_bits(header["obs"])
            act_bits = float32_bits([ep.obs[i] for i in range(OBS_WIDTH)])
            for i, (e, a) in enumerate(zip(obs_bits, act_bits)):
                if e != a:
                    problems.append({"field": f"reset.obs[{i}]", "expected": header["obs"][i], "actual": float(ep.obs[i]), "kind": "float32"})
            problems.extend(compare_state(header["state"], ep))
        draws_total += len(header["draws"])
        dev_ep = None
        if cuda is not None and rc == 0:
            if cuda_mode == "resident":
                dev_ep, flags = cuda.reset_replicated(cfg, header["draws"], cuda_lanes)
            else:
                cuda.upload(ep, cuda_lanes)
                dev_ep = Episode()
                ctypes.memmove(ctypes.byref(dev_ep), ctypes.byref(ep), ctypes.sizeof(Episode))
                flags = {"all_equal": 1, "lane_error": 0, "rng_error": 0, "rng_cursor": int(rng.cursor)}
            cuda_stats["resets_compared"] += 1
            cuda_stats["lanes_identical_every_time"] = cuda_stats["lanes_identical_every_time"] and bool(flags["all_equal"])
            cuda_stats["lane_error_max"] = max(cuda_stats["lane_error_max"], flags["lane_error"])
            if episode_bytes(dev_ep) != episode_bytes(ep) or flags["rng_cursor"] != int(rng.cursor) or flags["rng_error"] != 0:
                cuda_stats["device_mismatch_ticks"] += 1
                diffs = device_diff(ep, dev_ep, None, None)
                strict = _strict(diffs)
                for d in diffs:
                    g = _field_group(d["field"]); cuda_stats["mismatch_field_counts"][g] = cuda_stats["mismatch_field_counts"].get(g, 0) + 1
                if cuda_stats["first_mismatch"] is None:
                    cuda_stats["first_mismatch"] = {"seed": seed, "episode": index, "tick": "reset", "diffs": diffs[:8], "flags": flags}
                if strict or flags["rng_error"] != 0 or flags["rng_cursor"] != int(rng.cursor):
                    cuda_stats["strict_mismatch_ticks"] += 1
                    if cuda_stats["first_strict_mismatch"] is None:
                        cuda_stats["first_strict_mismatch"] = {"seed": seed, "episode": index, "tick": "reset", "diffs": strict[:8], "flags": flags}
                    if cuda_mode == "resident":
                        cuda_stats["resident_first_divergence"].append({"seed": seed, "episode": index, "tick": "reset", "first": strict[0] if strict else None})
                else:
                    cuda_stats["tolerated_only_ticks"] += 1
            else:
                cuda_stats["device_equal_ticks"] += 1
        dev_diverged = False if cuda is None else (cuda_stats["resident_first_divergence"][-1:] == [{"seed": seed, "episode": index, "tick": "reset", "first": None}] or
                                                   (cuda_stats["resident_first_divergence"] and cuda_stats["resident_first_divergence"][-1].get("episode") == index and
                                                    cuda_stats["resident_first_divergence"][-1].get("seed") == seed and cuda_stats["resident_first_divergence"][-1].get("tick") == "reset"))
        if problems:
            report["exact"] = False
            report["first"] = {"tick": "reset", "first": problems[0], "count": len(problems)}
            if first_divergence is None:
                first_divergence = {"seed": seed, "episode": index, "tick": "reset", "first": problems[0], "count": len(problems)}
            if len(mismatch_ticks) < max_mismatch_ticks:
                mismatch_ticks.append({"seed": seed, "episode": index, "tick": "reset", "mismatches": problems[:12], "count": len(problems)})
            episode_reports.append(report)
            continue
        for tick in entry["ticks"]:
            rng, keep = ref.replay_rng(tick["draws"])
            draws_total += len(tick["draws"])
            if cuda is not None and cuda_mode == "per-tick":
                cuda.upload(ep, cuda_lanes)
            rc, out = ref.step(ep, tick["action"], rng)
            if cuda is not None and rc == 0 and not (cuda_mode == "resident" and dev_diverged):
                dev_ep, dev_out, flags = cuda.step(tick["action"], tick["draws"], cuda_lanes)
                cuda_stats["ticks_compared"] += 1
                cuda_stats["lanes_identical_every_time"] = cuda_stats["lanes_identical_every_time"] and bool(flags["all_equal"])
                cuda_stats["lane_error_max"] = max(cuda_stats["lane_error_max"], flags["lane_error"])
                if flags["rng_error"] != 0:
                    cuda_stats["rng_error_ticks"] += 1
                same = episode_bytes(dev_ep) == episode_bytes(ep) and result_bytes(dev_out) == result_bytes(out) and flags["rng_cursor"] == int(rng.cursor)
                if same:
                    cuda_stats["device_equal_ticks"] += 1
                else:
                    cuda_stats["device_mismatch_ticks"] += 1
                    diffs = device_diff(ep, dev_ep, out, dev_out)
                    strict = _strict(diffs)
                    for d in diffs:
                        g = _field_group(d["field"]); cuda_stats["mismatch_field_counts"][g] = cuda_stats["mismatch_field_counts"].get(g, 0) + 1
                    if cuda_stats["first_mismatch"] is None:
                        cuda_stats["first_mismatch"] = {"seed": seed, "episode": index, "tick": tick["tick"], "label": tick.get("label"), "diffs": diffs[:8], "flags": flags}
                    if _float32(dev_out.reward) != _float32(out.reward):
                        cuda_stats["reward_float32_mismatch_ticks"] += 1
                    cuda_stats["reward_max_abs_diff"] = max(cuda_stats["reward_max_abs_diff"], abs(dev_out.reward - out.reward))
                    if strict or flags["rng_error"] != 0 or flags["rng_cursor"] != int(rng.cursor):
                        cuda_stats["strict_mismatch_ticks"] += 1
                        if cuda_stats["first_strict_mismatch"] is None:
                            cuda_stats["first_strict_mismatch"] = {"seed": seed, "episode": index, "tick": tick["tick"], "label": tick.get("label"), "diffs": strict[:8], "flags": flags}
                        if cuda_mode == "resident":
                            cuda_stats["resident_first_divergence"].append({"seed": seed, "episode": index, "tick": tick["tick"], "label": tick.get("label"),
                                                                            "first": strict[0] if strict else None})
                            dev_diverged = True
                    else:
                        cuda_stats["tolerated_only_ticks"] += 1
            problems = []
            if rc != 0:
                problems.append({"field": "step.rc", "expected": 0, "actual": rc, "kind": "value", "rng_error": int(rng.error), "rng_cursor": int(rng.cursor),
                                 "rng_error_kind": int(rng.error_kind), "rng_error_a": rng.error_a, "rng_error_b": rng.error_b})
            else:
                if rng.cursor != rng.count:
                    problems.append({"field": "step.draws_consumed", "expected": int(rng.count), "actual": int(rng.cursor), "kind": "value"})
                problems.extend(compare_outputs(tick, out))
                problems.extend(compare_state(tick["state"], ep))
            compared += 1
            report["compared"] += 1
            if problems:
                report["exact"] = False
                report["first"] = {"tick": tick["tick"], "label": tick.get("label"), "first": problems[0], "count": len(problems)}
                if first_divergence is None:
                    first_divergence = {"seed": seed, "episode": index, "tick": tick["tick"], "label": tick.get("label"), "first": problems[0], "count": len(problems)}
                if len(mismatch_ticks) < max_mismatch_ticks:
                    mismatch_ticks.append({"seed": seed, "episode": index, "tick": tick["tick"], "label": tick.get("label"), "mismatches": problems[:16], "count": len(problems)})
                break
        if report["exact"]:
            exact_episodes += 1
        episode_reports.append(report)
    return {
        "schema": "ai2.navigation_episode_verify.v1",
        "created_unix": time.time(),
        "capture": str(Path(capture_path).resolve()),
        "capture_sha256": contact.sha256_file(capture_path),
        "capture_metadata": {k: metadata.get(k) for k in ("controller", "seeds", "steps_per_seed", "ticks", "episodes", "pymunk", "chipmunk", "source", "wrapper")},
        "source_hashes": source_hashes(),
        "positions_sha256": ref.positions_sha256,
        "compared_ticks": compared,
        "episodes": len(episodes),
        "exact_episodes": exact_episodes,
        "draws_replayed": draws_total,
        "exact": first_divergence is None,
        "first_divergence": first_divergence,
        "mismatch_ticks": mismatch_ticks,
        "episode_reports": episode_reports,
        "tolerance": "none; bitwise equality required for every output and snapshot field",
        "mode": "closed loop: reset from recorded reset draws, then recorded actions and recorded draws only; no per-tick forcing",
        "cuda": cuda_stats,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="replay a closed-loop capture through the C reference")
    verify.add_argument("capture", type=Path)
    verify.add_argument("--build-dir", type=Path, default=ROOT / "build" / "navigation_episode")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--fp-contract", choices=("on", "off", "fast"), default=None)
    verify.add_argument("--cuda", action="store_true", help="also step device lanes and compare them with the CPU reference")
    verify.add_argument("--cuda-lanes", type=int, default=256)
    verify.add_argument("--cuda-mode", choices=("per-tick", "resident"), default="per-tick",
                        help="per-tick: device lanes re-seeded from the CPU state before every step; resident: device lanes carry their own state")
    bench = sub.add_parser("bench", help="whole-tick device throughput with generator-mode lanes (no policy)")
    bench.add_argument("--build-dir", type=Path, default=ROOT / "build" / "navigation_episode")
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--lanes", default="256,1024,4096,16384")
    bench.add_argument("--iterations", type=int, default=300)
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--cpu-steps", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    if args.command == "bench":
        return _bench_main(args)
    ref = EpisodeReference(compile_cpu_reference(args.build_dir, args.fp_contract))
    cuda = CudaEpisodeReference(compile_cuda_library(args.build_dir)) if args.cuda else None
    result = verify_capture(ref, args.capture, cuda=cuda, cuda_lanes=args.cuda_lanes, cuda_mode=args.cuda_mode)
    result["fp_contract"] = args.fp_contract or contact.default_contraction()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {k: result[k] for k in ("compared_ticks", "episodes", "exact_episodes", "draws_replayed", "exact")}
    summary["first_divergence"] = result["first_divergence"]
    if result["cuda"] is not None:
        summary["cuda"] = {k: result["cuda"][k] for k in ("mode", "lanes", "ticks_compared", "device_equal_ticks", "device_mismatch_ticks", "strict_mismatch_ticks",
                                                            "tolerated_only_ticks", "reward_float32_mismatch_ticks", "reward_max_abs_diff",
                                                            "lanes_identical_every_time", "lane_error_max", "mismatch_field_counts")}
    print(json.dumps(summary, sort_keys=True, default=str))
    ok = result["exact"] and (result["cuda"] is None or (result["cuda"]["strict_mismatch_ticks"] == 0 and result["cuda"]["lanes_identical_every_time"]
                                                          and result["cuda"]["reward_float32_mismatch_ticks"] == 0))
    return 0 if ok else 1


PHASE_0B = {"min_dist": 2.0, "max_dist": 6.0, "fall_penalty": -3.0, "completion_mode": "sequence", "control_mode": "movement", "settle_speed": 1.0,
            "max_targets": 3, "time_limit": 600, "settle_steps": 9, "post_goal_required_steps": 15}


def _bench_main(args: argparse.Namespace) -> int:
    import random

    ref = EpisodeReference(compile_cpu_reference(args.build_dir))
    cuda = CudaEpisodeReference(compile_cuda_library(args.build_dir))
    cfg = ref.make_config(PHASE_0B)
    cuda.set_table(ref.platform_table(cfg.dt))
    results = []
    for lanes in [int(x) for x in args.lanes.split(",") if x.strip()]:
        warm = cuda.bench(cfg, lanes, 20, args.seed)
        timed = cuda.bench(cfg, lanes, args.iterations, args.seed + 1)
        timed["warmup"] = warm
        results.append(timed)
        print(json.dumps({k: timed[k] for k in ("lanes", "iterations", "elapsed_ms", "lane_steps_per_second", "episodes_completed", "lane_error")}))
    # CPU single lane through ctypes, generator mode, same action policy shape.
    rng = Rng()
    ref.lib.ai2n_rng_seed_export(ctypes.byref(rng), args.seed, 0)
    ep = Episode()
    assert ref.reset(ep, cfg, rng) == 0
    actions = random.Random(args.seed)
    t0 = time.perf_counter()
    done = 0
    for _ in range(args.cpu_steps):
        act = [1.0 if actions.random() < 0.25 else 0.0 for _ in range(10)]
        rc, out = ref.step(ep, act, rng)
        if rc != 0:
            raise RuntimeError(f"cpu step failed: {rc}")
        if out.terminated or out.truncated:
            done += 1
            assert ref.reset(ep, cfg, rng) == 0
    cpu_elapsed = time.perf_counter() - t0
    payload = {"schema": "ai2.navigation_episode_bench.v1", "created_unix": time.time(), "source_hashes": source_hashes(), "config": PHASE_0B,
               "note": "whole environment tick per lane with generator-mode RNG, random actions and in-kernel resets; no policy, no host transfer per step; not environment SPS with a policy, not PPO SPS",
               "cuda": results, "cpu_single_lane": {"steps": args.cpu_steps, "elapsed_s": cpu_elapsed, "steps_per_second": args.cpu_steps / cpu_elapsed, "episodes_completed": done,
                                                   "note": "one lane through ctypes from Python; includes call overhead"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cpu_single_lane_steps_per_second": payload["cpu_single_lane"]["steps_per_second"]}))
    cuda.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
