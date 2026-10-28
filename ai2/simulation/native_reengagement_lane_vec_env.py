"""Finite-schedule, resident reengagement episodes for the curriculum TensorPPO.

One lane is one learner-controlled game against a passive opponent. The CPU
backend is a correctness reference, never an automatic training fallback.
Geometry is sampled once at setup exactly as the original B2 wrapper does.
Native steps, autoresets, observations and four-frame stacks remain on CUDA;
episode/error records cross to the host only at explicit rollout drains.
"""

from __future__ import annotations

from ai2.paths import REPO_ROOT as PROJECT_ROOT

import ctypes as C
import hashlib
import json
import numbers
import os
import platform
import shutil
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

import ai2.simulation.native_reengagement_episode as reengagement
from ai2.simulation.native_lane_vec_env import NativeLaneEnv, frozen_spaces
from ai2.simulation.native_navigation_episode import Rng

ROOT = PROJECT_ROOT
INT32, U, F, D, V = C.c_int32, C.c_uint64, C.c_float, C.c_double, C.c_void_p
SCENARIO_DTYPE = np.dtype(
    [
        ("seed", np.uint64),
        ("player_slot", np.int32),
        ("direction", np.int32),
        ("gap", np.float64),
        ("midpoint", np.float64),
    ],
    align=True,
)
SCHEDULE_GENERATOR = "b2-reengagement-vector-env-gym-seeds-v1"
EVALUATION_RANGES = (range(134000, 134020), range(144000, 144100))
EVALUATION_SEEDS = frozenset(seed for group in EVALUATION_RANGES for seed in group)
CONTROL_MASK = (True, True, True, True, True, False, True, True, False, False)
CURRICULUM_VERSION = "bridge_b2_reengagement-v1"


class Lane(C.Structure):
    _fields_ = [
        ("episode", reengagement.State),
        ("rng", Rng),
        ("ret", D),
        ("last_obs", F * 64),
        ("valid_steps", U),
    ] + [
        (name, INT32)
        for name in ("length", "episodes", "error", "active", "scenario_index")
    ]


class EpisodeRecord(C.Structure):
    _fields_ = [
        (name, INT32)
        for name in (
            "lane",
            "length",
            "terminated",
            "truncated",
            "episode_index",
            "scenario_index",
        )
    ] + [("ret", D), ("diagnostics", reengagement.Diagnostics)]


ABI_LAYOUTS = (("lane", Lane), ("record", EpisodeRecord))


def _copied_scenario_rows(rows):
    copied = []
    for row in rows:
        copied.append(
            {
                "lane": row["lane"],
                "episode": row["episode"],
                "curriculum_task": row["curriculum_task"],
                "curriculum_version": row["curriculum_version"],
                "generator": row["generator"],
                "scenario": dict(row["scenario"]),
            }
        )
    return copied


def _positive_int(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or not 0 < value <= np.iinfo(np.int32).max
    ):
        raise ValueError(f"{name} must be a positive int32 integer")
    return int(value)


def _initial_seeds(lanes, seed):
    if (
        isinstance(seed, bool)
        or not isinstance(seed, numbers.Integral)
        or not 0 <= seed <= (1 << 64) - lanes
    ):
        raise ValueError("seed plus lane index must fit an unsigned 64-bit integer")
    return np.asarray([int(seed) + lane for lane in range(lanes)], dtype=np.uint64)


def make_scenario_schedule(lanes, episodes_per_lane, *, seed=0):
    """Reproduce make_vector_env + seed_rollout + B2Env.reset(seed=None).

    The player's slot is rank % 2 for its lifetime. The first reset uses the
    paired seed plus rank. Later resets sample uint32 episode seeds from that
    lane's persistent Gymnasium default_rng stream. Geometry uses a separate
    default_rng(episode_seed); it never consumes the persistent seed stream.
    """
    lanes = _positive_int(lanes, "lanes")
    episodes_per_lane = _positive_int(episodes_per_lane, "episodes_per_lane")
    seeds = _initial_seeds(lanes, seed)
    schedule = np.empty((lanes, episodes_per_lane), dtype=SCENARIO_DTYPE)
    for lane, initial_seed in enumerate(seeds):
        generator = np.random.default_rng(int(initial_seed))
        for episode in range(episodes_per_lane):
            episode_seed = (
                int(initial_seed) if episode == 0 else int(generator.integers(0, 2**32))
            )
            while episode_seed in EVALUATION_SEEDS:
                episode_seed = int(generator.integers(0, 2**32))
            scenario = reengagement.scenario_from_seed(episode_seed, player_slot=lane % 2)
            schedule[lane, episode] = (
                scenario.seed,
                scenario.player_slot,
                scenario.direction,
                scenario.gap,
                scenario.midpoint,
            )
    return schedule


def _validate_schedule(schedule, lanes):
    if not isinstance(schedule, np.ndarray) or schedule.dtype != SCENARIO_DTYPE:
        raise ValueError("scenario_schedule must be an array with SCENARIO_DTYPE")
    if schedule.ndim != 2 or schedule.shape[0] != lanes:
        raise ValueError("scenario_schedule must have shape (lanes, episodes_per_lane)")
    _positive_int(schedule.shape[1], "episodes_per_lane")
    if (
        not np.isin(schedule["player_slot"], (0, 1)).all()
        or not np.isin(schedule["direction"], (-1, 1)).all()
        or not np.isfinite(schedule["gap"]).all()
        or not ((schedule["gap"] >= 3.1) & (schedule["gap"] <= 3.4)).all()
        or not np.isfinite(schedule["midpoint"]).all()
        or not ((schedule["midpoint"] >= 4.35) & (schedule["midpoint"] <= 4.65)).all()
    ):
        raise ValueError(
            "scenario_schedule contains invalid B2 geometry, slot or direction"
        )
    overlap = EVALUATION_SEEDS.intersection(int(seed) for seed in schedule["seed"].ravel())
    if overlap:
        raise ValueError(f"scenario_schedule contains evaluation seed(s): {sorted(overlap)[:8]}")
    # Independent, immutable setup copy. Native creation makes its own copy.
    result = np.array(schedule, dtype=SCENARIO_DTYPE, order="C", copy=True)
    result.setflags(write=False)
    return result


def _scenario_rows(schedule):
    rows = []
    for lane in range(schedule.shape[0]):
        for episode in range(schedule.shape[1]):
            row = schedule[lane, episode]
            rows.append(
                {
                    "lane": int(lane),
                    "episode": int(episode),
                    "curriculum_task": "combat",
                    "curriculum_version": CURRICULUM_VERSION,
                    "generator": SCHEDULE_GENERATOR,
                    "scenario": {
                        "seed": int(row["seed"]),
                        "player_slot": int(row["player_slot"]),
                        "direction": int(row["direction"]),
                        "gap": float(row["gap"]),
                        "midpoint": float(row["midpoint"]),
                    },
                }
            )
    return rows


def compile_lanes(kind, build_dir=None):
    if kind not in {"cpu", "cuda"}:
        raise ValueError("kind must be cpu or cuda")
    from ai2.simulation.native_navigation_contact import default_contraction

    folder = Path(build_dir or ROOT / ".native-build")
    folder.mkdir(parents=True, exist_ok=True)
    contract = default_contraction() if kind == "cpu" else "off"
    if kind == "cpu":
        requested_compiler = os.environ.get("CC", "cc")
        compiler = shutil.which(requested_compiler) or requested_compiler
        sources = [
            ROOT / "native" / name
            for name in (
                "reengagement_lanes.c",
                "reengagement_episode.c",
                "game_episode.c",
                "navigation_episode.c",
            )
        ]
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
            *(str(path) for path in sources),
            "-lm",
        ]
    else:
        compiler = shutil.which("nvcc")
        if not compiler:
            raise RuntimeError(
                "nvcc unavailable: no CPU fallback for CUDA reengagement lanes"
            )
        sources = [ROOT / "native/reengagement_lanes.cu"]
        command = [
            compiler,
            "-std=c++17",
            "-O2",
            "--fmad=false",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            "-Xcompiler",
            "-ffp-contract=off",
            *(str(path) for path in sources),
        ]
    identity = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    )
    digest = hashlib.sha256(
        repr(
            (
                kind,
                contract,
                platform.platform(),
                command,
                identity.stdout,
                identity.stderr,
            )
        ).encode()
    )
    for path in sorted((ROOT / "native").glob("*.h")) + sources:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    suffix = ".dylib" if kind == "cpu" and platform.system() == "Darwin" else ".so"
    output = folder / f"libreengagement-lanes-{kind}-{digest.hexdigest()[:20]}{suffix}"
    if not output.is_file():
        with tempfile.TemporaryDirectory(
            prefix="reengagement-lanes-build-", dir=folder
        ) as temporary:
            candidate = Path(temporary) / output.name
            subprocess.run(command + ["-o", str(candidate)], check=True)
            os.replace(candidate, output)
    return output


class ReengagementLaneBackend:
    def __init__(
        self,
        lanes,
        device,
        *,
        seed=0,
        episodes_per_lane=4096,
        scenario_schedule=None,
        record_capacity=None,
    ):
        import torch

        self.handle = None
        self.torch = torch
        self.lanes = self.games = _positive_int(lanes, "lanes")
        episodes_per_lane = _positive_int(episodes_per_lane, "episodes_per_lane")
        self.initial_seeds = _initial_seeds(self.lanes, seed)
        self.record_capacity = _positive_int(
            max(64, self.lanes * 8) if record_capacity is None else record_capacity,
            "record_capacity",
        )
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("reengagement lanes require cpu reference or cuda")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA unavailable; reengagement lanes will not silently fall back"
                )
            self.device = torch.device(
                "cuda",
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device(),
            )
        self.kind = self.device.type
        generated = scenario_schedule is None
        if generated:
            scenario_schedule = make_scenario_schedule(
                self.lanes, episodes_per_lane, seed=seed
            )
        self._schedule = _validate_schedule(scenario_schedule, self.lanes)
        self.episodes_per_lane = self._schedule.shape[1]
        if SCENARIO_DTYPE.itemsize != C.sizeof(reengagement.Scenario) or any(
            SCENARIO_DTYPE.fields[name][1] != getattr(reengagement.Scenario, name).offset
            for name in SCENARIO_DTYPE.names
        ):
            raise RuntimeError("scenario native/Python ABI mismatch")
        scenario_rows = _scenario_rows(self._schedule)
        self._scenario_metadata = {
            "generator": SCHEDULE_GENERATOR if generated else "explicit-schedule-v1",
            "seed": int(seed),
            "lanes": self.lanes,
            "episodes_per_lane": self.episodes_per_lane,
            "sha256": hashlib.sha256(self._schedule.tobytes()).hexdigest(),
            "shape": list(self._schedule.shape),
            "dtype": SCENARIO_DTYPE.descr,
            "slot_rule": "lane % 2" if generated else "explicit",
            "direction_rule": "-1 if (seed // 2) % 2 == 0 else +1"
            if generated
            else "explicit",
            "reset_rule": "consume next scenario; no rewind; exhaustion fails closed",
            "curriculum_version": CURRICULUM_VERSION,
            "curriculum_task": "combat",
            "allowed_control_mask": CONTROL_MASK,
            "excluded_seed_ranges": [[134000, 134019], [144000, 144099]],
            "scenarios": scenario_rows,
        }
        # CPU exports provide the exact shared config/table, including when
        # the transport itself is CUDA. No Python game is used in lane steps.
        self.reference = reengagement.ReengagementReference(compile_lanes("cpu"))
        self.library = compile_lanes(self.kind)
        self.lib = C.CDLL(str(self.library))
        prefix = f"ai2b2_lanes_{self.kind}_"
        extra = [V] if self.kind == "cuda" else []
        specs = {
            "create": ([V, V, INT32, INT32, D, V, INT32, C.POINTER(INT32)], V),
            "destroy": ([V], INT32),
            "reset": ([V, V, V] + extra, INT32),
            "step": ([V] * 7 + extra, INT32),
            "drain": ([V, V, INT32, V, V, V, V] + extra, INT32),
            "lane_get": ([V, INT32, V] + extra, INT32),
            "lane_set": ([V, INT32, V] + extra, INT32),
        }
        functions = {}
        for name, (args, result) in specs.items():
            fn = getattr(self.lib, prefix + name)
            fn.argtypes, fn.restype = args, result
            functions[name] = fn
        self.create, self.destroy = functions["create"], functions["destroy"]
        self.reset_fn, self.step_fn, self.drain_fn = (
            functions["reset"],
            functions["step"],
            functions["drain"],
        )
        self.get_fn, self.set_fn = functions["lane_get"], functions["lane_set"]
        for name, cls in ABI_LAYOUTS:
            fn = getattr(self.lib, prefix + name + "_size")
            fn.restype = C.c_size_t
            if fn() != C.sizeof(cls):
                raise RuntimeError(f"reengagement {name} native/Python ABI mismatch")
            count_fn = getattr(self.lib, prefix + name + "_field_count")
            offset_fn = getattr(self.lib, prefix + name + "_field_offset")
            count_fn.restype = offset_fn.restype = C.c_size_t
            offset_fn.argtypes = [C.c_size_t]
            if count_fn() != len(cls._fields_) or any(
                offset_fn(index) != getattr(cls, field_name).offset
                for index, (field_name, *_rest) in enumerate(cls._fields_)
            ):
                raise RuntimeError(f"reengagement {name} native/Python ABI mismatch")
        self.records = (EpisodeRecord * self.record_capacity)()
        self.errors = np.zeros(self.lanes, dtype=np.int32)
        self.valid_steps = np.zeros(self.lanes, dtype=np.uint64)
        self.last_drain = {"count": 0, "overflow": 0}
        error = INT32()
        with self._guard():
            self.handle = self.create(
                C.byref(self.reference.cfg),
                C.byref(self.reference.table),
                self.lanes,
                self.record_capacity,
                48.0,
                V(self._schedule.ctypes.data),
                self.episodes_per_lane,
                C.byref(error),
            )
        if not self.handle or error.value:
            raise RuntimeError(f"native reengagement context creation failed: {error.value}")

    @property
    def scenario_metadata(self):
        return {
            **self._scenario_metadata,
            "shape": list(self._scenario_metadata["shape"]),
            "allowed_control_mask": tuple(self._scenario_metadata["allowed_control_mask"]),
            "excluded_seed_ranges": [
                list(pair) for pair in self._scenario_metadata["excluded_seed_ranges"]
            ],
            "scenarios": _copied_scenario_rows(self._scenario_metadata["scenarios"]),
        }

    def write_scenarios_json(self, path):
        payload = self.scenario_metadata
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return target

    def _guard(self):
        return (
            self.torch.cuda.device(self.device)
            if self.kind == "cuda"
            else nullcontext()
        )

    def _args(self, *args):
        if not self.handle:
            raise RuntimeError("reengagement lane context is closed")
        return [self.handle, *args] + (
            [V(self.torch.cuda.current_stream(self.device).cuda_stream)]
            if self.kind == "cuda"
            else []
        )

    @staticmethod
    def _check(code, where):
        if code:
            raise RuntimeError(f"reengagement lanes {where}: error {code}")

    def _ptr(self, tensor, shape, dtype):
        if (
            not self.torch.is_tensor(tensor)
            or tensor.device != self.device
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ValueError(
                f"expected contiguous {dtype} {shape} on {self.device}, without gradients"
            )
        return V(tensor.data_ptr())

    def _record_stream(self, tensors):
        if self.kind == "cuda":
            stream = self.torch.cuda.current_stream(self.device)
            for tensor in tensors:
                tensor.record_stream(stream)

    def reset(self, seeds, obs):
        if not np.array_equal(np.asarray(seeds), self.initial_seeds):
            raise ValueError(
                "scenario schedule is fixed at context creation; construct a new context to reseed"
            )
        pointer = self._ptr(obs, (self.lanes, 64), self.torch.float32)
        with self._guard():
            self._record_stream((obs,))
            self._check(self.reset_fn(*self._args(None, pointer)), "reset")
        self.errors.fill(0)

    def step(self, actions, obs, terminal, rewards, terminated, truncated):
        t = self.torch
        arrays = (actions, obs, terminal, rewards, terminated, truncated)
        shapes = (
            (self.lanes, 10),
            (self.lanes, 64),
            (self.lanes, 64),
            (self.lanes,),
            (self.lanes,),
            (self.lanes,),
        )
        pointers = [
            self._ptr(array, shape, t.float32 if i < 4 else t.bool)
            for i, (array, shape) in enumerate(zip(arrays, shapes))
        ]
        with self._guard():
            self._record_stream(arrays)
            self._check(self.step_fn(*self._args(*pointers)), "step")

    def drain(self):
        count, overflow = INT32(), INT32()
        with self._guard():
            self._check(
                self.drain_fn(
                    *self._args(
                        self.records,
                        self.record_capacity,
                        C.byref(count),
                        C.byref(overflow),
                        V(self.errors.ctypes.data),
                        V(self.valid_steps.ctypes.data),
                    )
                ),
                "drain",
            )
        self.last_drain = {"count": count.value, "overflow": overflow.value}
        if self.errors.any() or overflow.value:
            raise RuntimeError(
                f"invalid rollout: lane_errors={self.errors.tolist()}, record_overflow={overflow.value}"
            )
        return [
            EpisodeRecord.from_buffer_copy(self.records[i]) for i in range(count.value)
        ], 0

    def error(self):
        # Refreshed only at explicit rollout drains; no CUDA host polling.
        bad = np.flatnonzero(self.errors)
        return (int(self.errors[bad[0]]), int(bad[0])) if len(bad) else (0, -1)

    def lane_state(self, lane):
        lane = self._lane_index(lane)
        state = Lane()
        with self._guard():
            self._check(self.get_fn(*self._args(lane, C.byref(state))), "lane_get")
        return state

    def set_lane_state(self, lane, state):
        """Explicit diagnostic fixture import; never called by training."""
        lane = self._lane_index(lane)
        if not isinstance(state, Lane):
            raise TypeError("state must be a reengagement Lane")
        with self._guard():
            self._check(self.set_fn(*self._args(lane, C.byref(state))), "lane_set")

    def _lane_index(self, value):
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or not 0 <= value < self.lanes
        ):
            raise ValueError("lane index is out of range")
        return int(value)

    def close(self):
        if self.handle:
            with self._guard():
                handle, self.handle = self.handle, None
                self._check(self.destroy(handle), "destroy")


class NativeReengagementEnv(NativeLaneEnv):
    @property
    def curriculum_task(self):
        return "combat"

    @property
    def allowed_control_mask(self):
        return CONTROL_MASK

    @property
    def curriculum_version(self):
        return CURRICULUM_VERSION

    def host_has_timeouts(self):
        return False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._collector_event = None
        self._collector_stream = None
        if self.device.type == "cuda":
            with self.backend._guard():
                self._collector_stream = self._torch.cuda.current_stream(self.device)
                self._collector_event = self._torch.cuda.Event()
                self._collector_event.record(self._collector_stream)

    def seed(self, seeds):
        if not np.array_equal(np.asarray(seeds), self.backend.initial_seeds):
            raise ValueError(
                "scenario schedule is fixed at context creation; construct a new context to reseed"
            )
        return super().seed(seeds)

    def _collector_begin(self):
        if self._collector_event is not None:
            stream = self._torch.cuda.current_stream(self.device)
            # Include prior-stream consumers queued since the previous return,
            # as well as native state and frame-stack writes. No host wait.
            self._collector_event.record(self._collector_stream)
            stream.wait_event(self._collector_event)
            self._collector_stream = stream
            self.backend._record_stream(
                (
                    self._obs,
                    self._terminal,
                    self._rewards,
                    self._terminated,
                    self._truncated,
                    self._stack,
                )
            )

    def _collector_end(self):
        if self._collector_event is not None:
            self._collector_event.record(self._collector_stream)

    def reset_tensor(self, seeds=None):
        with self.backend._guard():
            self._collector_begin()
            result = super().reset_tensor(seeds)
            self._collector_end()
        return result

    def step_tensor(self, clipped_actions):
        with self.backend._guard():
            self._collector_begin()
            result = super().step_tensor(clipped_actions)
            self._collector_end()
        return result

    def drain_episode_infos(self):
        records, _ = self.backend.drain()
        elapsed = time.time() - self._t_start
        return [
            {
                "episode": {
                    "r": round(record.ret, 6),
                    "l": record.length,
                    "t": elapsed,
                },
                "curriculum_task": self.curriculum_task,
                "allowed_control_mask": self.allowed_control_mask,
                "curriculum_version": self.curriculum_version,
                "reengagement": {
                    "lane": record.lane,
                    "episode_index": record.episode_index,
                    "scenario_index": record.scenario_index,
                    "terminated": bool(record.terminated),
                    "truncated": bool(record.truncated),
                    "diagnostics": reengagement.diagnostics_projection(record.diagnostics),
                },
            }
            for record in records
        ]


def build_reengagement_env(
    lanes,
    *,
    device="cuda",
    seed=0,
    episodes_per_lane=4096,
    scenario_schedule=None,
    record_capacity=None,
):
    backend = ReengagementLaneBackend(
        lanes,
        device,
        seed=seed,
        episodes_per_lane=episodes_per_lane,
        scenario_schedule=scenario_schedule,
        record_capacity=record_capacity,
    )
    try:
        observation_space, action_space = frozen_spaces()
        return NativeReengagementEnv(
            backend,
            observation_space=observation_space,
            action_space=action_space,
            seed=seed,
        )
    except BaseException:
        backend.close()
        raise
