"""Native navigation lanes as a device-resident environment for ``TensorPPO``.

``NativeLaneEnv`` implements the ``TensorVecEnv`` contract (``reset_tensor``,
``step_tensor``, ``drain_episode_infos``) on top of ``native/navigation_lanes``:
every lane is one SB3 environment slot whose whole tick (input decode,
platform, state machines, contact physics, spawner, navigation wrapper,
observation) runs in the native code, with generator-mode random streams and
in-kernel autoreset.  The env owns the frame stack as a tensor operation that
mirrors ``VecFrameStack`` and reports completed episodes the way ``Monitor``
does, once per rollout.

Two backends share the same C function bodies: ``CpuLaneBackend`` (host
library, the reference and the local test backend) and ``CudaLaneBackend``
(one thread per lane, buffers are torch CUDA tensors, kernels run on the
collector's stream, no host transfer per step).  ``TensorVecEnv`` itself is
CUDA-only, so the CPU backend is exercised directly by the tests.
"""
from __future__ import annotations

from ai2.paths import REPO_ROOT as PROJECT_ROOT

import ctypes
import json
import numbers
import os
import platform
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

import ai2.simulation.native_navigation_episode as episode

ROOT = PROJECT_ROOT
NATIVE_DIR = ROOT / "native"
LANE_SOURCE_FILES = (NATIVE_DIR / "navigation_lanes.h", NATIVE_DIR / "navigation_lanes.c", NATIVE_DIR / "navigation_lanes.cu") + tuple(episode.SOURCE_FILES)
SPACES_PATH = NATIVE_DIR / "navigation_spaces.json"
OBS_WIDTH = 64
ACTION_WIDTH = 10
FRAME_STACK = 4
STACKED_WIDTH = OBS_WIDTH * FRAME_STACK
HISTORY_WIDTH = STACKED_WIDTH - OBS_WIDTH
PHASE_0B = {"min_dist": 2.0, "max_dist": 6.0, "fall_penalty": -3.0, "completion_mode": "sequence", "control_mode": "movement", "settle_speed": 1.0,
            "max_targets": 3, "time_limit": 600, "settle_steps": 9, "post_goal_required_steps": 15}
DEFAULT_RECORDS_PER_LANE = 64
UINT64_MASK = (1 << 64) - 1

ERROR_NAMES = {0: "none", 1: "null", 2: "config", 3: "rng", 4: "movement", 5: "contact", 6: "capacity", 7: "observation", 8: "target", 9: "input", 10: "platform"}


class LaneError(RuntimeError):
    def __init__(self, code: int, where: str, lane: int = -1) -> None:
        self.code = int(code)
        self.lane = int(lane)
        name = ERROR_NAMES.get(self.code, str(self.code))
        suffix = f" (lane {self.lane})" if self.lane >= 0 else ""
        super().__init__(f"native lanes {where} failed: error {self.code} {name}{suffix}")


class Lane(ctypes.Structure):
    _fields_ = [("ep", episode.Episode), ("rng", episode.Rng), ("ep_return", ctypes.c_double), ("ep_length", ctypes.c_int32),
                ("episodes", ctypes.c_int32), ("error", ctypes.c_int32), ("active", ctypes.c_int32), ("pad", ctypes.c_int32 * 2)]


class EpisodeRecord(ctypes.Structure):
    _fields_ = [("lane", ctypes.c_int32), ("length", ctypes.c_int32), ("ret", ctypes.c_double), ("terminated", ctypes.c_int32),
                ("truncated", ctypes.c_int32), ("success", ctypes.c_int32), ("fell", ctypes.c_int32), ("targets", ctypes.c_int32),
                ("horizon", ctypes.c_int32), ("episode_index", ctypes.c_int32), ("pad", ctypes.c_int32)]


def source_hashes() -> dict[str, str]:
    return {path.name: episode.contact.sha256_file(path) for path in LANE_SOURCE_FILES if path.is_file()}


def _up_to_date(library: Path, sources: Sequence[Path]) -> bool:
    if not library.is_file():
        return False
    newest = max(path.stat().st_mtime for path in sources if path.is_file())
    return library.stat().st_mtime > newest and library.stat().st_size > 0


def _atomic_build(library: Path, command: list[str], sources: Sequence[Path]) -> Path:
    """Compile to a private temporary name and rename, so parallel users never load a half-written library."""
    if _up_to_date(library, sources):
        return library
    library.parent.mkdir(parents=True, exist_ok=True)
    temp = library.with_name(f".{library.name}.{os.getpid()}.tmp")
    subprocess.run(command + ["-o", str(temp)], check=True, cwd=str(NATIVE_DIR))
    os.replace(temp, library)
    return library


def compile_lanes_cpu(build_dir: Path, contract: str | None = None) -> Path:
    """Host library: the episode reference exports plus the lane runtime, unfused tick arithmetic."""
    build_dir = Path(build_dir)
    contract = contract or episode.contact.default_contraction()
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = build_dir / f"libnavigation_lanes-{contract}{suffix}"
    compiler = os.environ.get("CC") or "cc"
    command = [compiler, "-std=c99", "-O2", f"-ffp-contract={contract}", "-fno-fast-math", "-Wall", "-Wextra", "-Werror", "-pedantic",
               "-shared", "-fPIC", str(NATIVE_DIR / "navigation_episode.c"), str(NATIVE_DIR / "navigation_lanes.c")]
    return _atomic_build(library, command, LANE_SOURCE_FILES)


def compile_lanes_cuda(build_dir: Path) -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is not available")
    build_dir = Path(build_dir)
    library = build_dir / "libnavigation_lanes_cuda.so"
    command = [nvcc, "-O2", "-fmad=false", "--shared", "-Xcompiler", "-fPIC", "-Xcompiler", "-ffp-contract=off", str(NATIVE_DIR / "navigation_lanes.cu")]
    return _atomic_build(library, command, LANE_SOURCE_FILES)


def load_spaces(path: Path = SPACES_PATH) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    obs, act = data["observation"], data["action"]
    if len(obs["low"]) != OBS_WIDTH or len(obs["high"]) != OBS_WIDTH or int(obs.get("frame_stack", FRAME_STACK)) != FRAME_STACK:
        raise ValueError("navigation_spaces.json does not describe a 64-wide, 4-frame observation")
    if len(act["low"]) != ACTION_WIDTH or len(act["high"]) != ACTION_WIDTH:
        raise ValueError("navigation_spaces.json does not describe a 10-wide action")
    return data


def frozen_spaces(path: Path = SPACES_PATH) -> tuple[Any, Any]:
    """The frozen training stack's stacked observation space and action space (gymnasium Boxes)."""
    from gymnasium import spaces

    data = load_spaces(path)
    low = np.asarray(data["observation"]["low"], dtype=np.float32)
    high = np.asarray(data["observation"]["high"], dtype=np.float32)
    base = spaces.Box(low=low, high=high, dtype=np.float32)
    try:
        from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

        stacked = StackedObservations(1, FRAME_STACK, base).stacked_observation_space
    except Exception:  # pragma: no cover - SB3 unavailable: same rule (element-wise repeat)
        stacked = spaces.Box(low=np.repeat(low, FRAME_STACK, axis=-1), high=np.repeat(high, FRAME_STACK, axis=-1), dtype=np.float32)
    action = spaces.Box(low=np.asarray(data["action"]["low"], dtype=np.float32), high=np.asarray(data["action"]["high"], dtype=np.float32),
                        dtype=np.float32)
    return stacked, action


def _check_spaces(observation_space: Any, action_space: Any) -> None:
    if type(observation_space).__name__ != "Box" or type(action_space).__name__ != "Box":
        raise ValueError("NativeLaneEnv supports Box observation/action spaces only")
    if tuple(observation_space.shape) != (STACKED_WIDTH,):
        raise ValueError(f"observation space must be ({STACKED_WIDTH},), got {tuple(observation_space.shape)}")
    if tuple(action_space.shape) != (ACTION_WIDTH,):
        raise ValueError(f"action space must be ({ACTION_WIDTH},), got {tuple(action_space.shape)}")
    if np.dtype(observation_space.dtype) != np.float32 or np.dtype(action_space.dtype) != np.float32:
        raise ValueError("observation and action spaces must be float32")


def _draw_array(draws: Sequence[dict[str, Any]]) -> Any:
    arr = (episode.Draw * max(1, len(draws)))()
    for i, d in enumerate(draws):
        arr[i].kind = int(d["kind"])
        arr[i].a = float(d["a"])
        arr[i].b = float(d["b"])
        arr[i].value = float(d["value"])
    return arr


class CpuLaneBackend:
    """Host lanes: the reference the CUDA lanes are compared against and the local test backend."""

    kind = "cpu"

    def __init__(self, library_path: Path, cfg: Any, table: Any, lanes: int, record_capacity: int) -> None:
        import torch

        self.library_path = Path(library_path)
        self.lib = ctypes.CDLL(str(self.library_path))
        lib = self.lib
        p = ctypes.c_void_p
        lib.ai2n_lanes_cpu_lane_size.restype = ctypes.c_int
        lib.ai2n_lanes_cpu_record_size.restype = ctypes.c_int
        if lib.ai2n_lanes_cpu_lane_size() != ctypes.sizeof(Lane) or lib.ai2n_lanes_cpu_record_size() != ctypes.sizeof(EpisodeRecord):
            raise RuntimeError("native lane struct layout differs from the ctypes mirror")
        lib.ai2n_lanes_cpu_create.restype = p
        lib.ai2n_lanes_cpu_create.argtypes = [ctypes.POINTER(episode.Config), ctypes.POINTER(episode.PlatformTable), ctypes.c_int, ctypes.c_int]
        lib.ai2n_lanes_cpu_destroy.restype = None
        lib.ai2n_lanes_cpu_destroy.argtypes = [p]
        lib.ai2n_lanes_cpu_reset.restype = ctypes.c_int
        lib.ai2n_lanes_cpu_reset.argtypes = [p, p, p]
        lib.ai2n_lanes_cpu_step.restype = ctypes.c_int
        lib.ai2n_lanes_cpu_step.argtypes = [p] * 7
        lib.ai2n_lanes_cpu_drain.restype = ctypes.c_int
        lib.ai2n_lanes_cpu_drain.argtypes = [p, p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        lib.ai2n_lanes_cpu_error.restype = ctypes.c_int
        lib.ai2n_lanes_cpu_error.argtypes = [p, ctypes.POINTER(ctypes.c_int)]
        lib.ai2n_lanes_cpu_lane_get.restype = ctypes.c_int
        lib.ai2n_lanes_cpu_lane_get.argtypes = [p, ctypes.c_int, ctypes.POINTER(Lane)]
        lib.ai2n_lanes_cpu_lane_set.restype = ctypes.c_int
        lib.ai2n_lanes_cpu_lane_set.argtypes = [p, ctypes.c_int, ctypes.POINTER(Lane)]
        lib.ai2n_lanes_cpu_replay_reset.restype = ctypes.c_int
        lib.ai2n_lanes_cpu_replay_reset.argtypes = [p, ctypes.c_int, ctypes.POINTER(episode.Draw), ctypes.c_int, p]
        lib.ai2n_lanes_cpu_replay_step.restype = ctypes.c_int
        lib.ai2n_lanes_cpu_replay_step.argtypes = [p, ctypes.c_int, ctypes.POINTER(episode.Draw), ctypes.c_int, p, p, p, p, p, p]
        lib.ai2n_lanes_cpu_rng_status.restype = ctypes.c_int
        lib.ai2n_lanes_cpu_rng_status.argtypes = [p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.lanes = int(lanes)
        self.record_capacity = int(record_capacity)
        self.cfg = cfg
        self.table = table
        self.device = torch.device("cpu")
        self._records = (EpisodeRecord * self.record_capacity)()
        self.handle = lib.ai2n_lanes_cpu_create(ctypes.byref(cfg), ctypes.byref(table), self.lanes, self.record_capacity)
        if not self.handle:
            raise RuntimeError("native lane creation failed")

    @staticmethod
    def _ptr(tensor: Any) -> ctypes.c_void_p:
        return ctypes.c_void_p(tensor.data_ptr())

    def reset(self, seeds: np.ndarray, obs: Any) -> None:
        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
        rc = self.lib.ai2n_lanes_cpu_reset(self.handle, ctypes.c_void_p(seeds.ctypes.data), self._ptr(obs))
        if rc:
            raise LaneError(rc, "reset", self.error()[1])

    def step(self, actions: Any, obs: Any, terminal: Any, rewards: Any, terminated: Any, truncated: Any) -> None:
        rc = self.lib.ai2n_lanes_cpu_step(self.handle, self._ptr(actions), self._ptr(obs), self._ptr(terminal), self._ptr(rewards),
                                          self._ptr(terminated), self._ptr(truncated))
        if rc:
            raise LaneError(rc, "step", self.error()[1])

    def drain(self) -> tuple[list[EpisodeRecord], int]:
        count = ctypes.c_int(0)
        overflow = ctypes.c_int(0)
        rc = self.lib.ai2n_lanes_cpu_drain(self.handle, ctypes.cast(self._records, ctypes.c_void_p), self.record_capacity, ctypes.byref(count), ctypes.byref(overflow))
        if rc:
            raise LaneError(rc, "drain")
        records = [EpisodeRecord.from_buffer_copy(self._records[i]) for i in range(int(count.value))]
        return records, int(overflow.value)

    def error(self) -> tuple[int, int]:
        lane = ctypes.c_int(-1)
        code = int(self.lib.ai2n_lanes_cpu_error(self.handle, ctypes.byref(lane)))
        return code, int(lane.value)

    def lane_state(self, lane: int) -> Lane:
        out = Lane()
        rc = self.lib.ai2n_lanes_cpu_lane_get(self.handle, int(lane), ctypes.byref(out))
        if rc:
            raise LaneError(rc, "lane_get", lane)
        return out

    def set_lane_state(self, lane: int, state: Lane) -> None:
        rc = self.lib.ai2n_lanes_cpu_lane_set(self.handle, int(lane), ctypes.byref(state))
        if rc:
            raise LaneError(rc, "lane_set", lane)

    def replay_reset(self, lane: int, draws: Sequence[dict[str, Any]], obs_row: Any) -> None:
        arr = _draw_array(draws)
        rc = self.lib.ai2n_lanes_cpu_replay_reset(self.handle, int(lane), arr, len(draws), self._ptr(obs_row))
        if rc:
            raise LaneError(rc, "replay_reset", lane)

    def replay_step(self, lane: int, draws: Sequence[dict[str, Any]], action_row: Any, obs_row: Any, terminal_row: Any, reward_row: Any,
                    terminated_row: Any, truncated_row: Any) -> int:
        arr = _draw_array(draws)
        return int(self.lib.ai2n_lanes_cpu_replay_step(self.handle, int(lane), arr, len(draws), self._ptr(action_row), self._ptr(obs_row),
                                                        self._ptr(terminal_row), self._ptr(reward_row), self._ptr(terminated_row), self._ptr(truncated_row)))

    def rng_status(self, lane: int) -> dict[str, int]:
        error, cursor, count = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
        rc = self.lib.ai2n_lanes_cpu_rng_status(self.handle, int(lane), ctypes.byref(error), ctypes.byref(cursor), ctypes.byref(count))
        if rc:
            raise LaneError(rc, "rng_status", lane)
        return {"error": int(error.value), "cursor": int(cursor.value), "count": int(count.value)}

    def close(self) -> None:
        if self.handle:
            self.lib.ai2n_lanes_cpu_destroy(self.handle)
            self.handle = None


class CudaLaneBackend:
    """Device lanes: one thread per lane, kernels queued on the collector's current stream."""

    kind = "cuda"

    def __init__(self, library_path: Path, cfg: Any, table: Any, lanes: int, record_capacity: int, *, device: Any = "cuda") -> None:
        import torch

        self.library_path = Path(library_path)
        self._torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudaLaneBackend requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.index != torch.cuda.current_device():
            raise ValueError(f"CudaLaneBackend device {self.device} must be the current CUDA device")
        self.lib = ctypes.CDLL(str(self.library_path))
        lib = self.lib
        p = ctypes.c_void_p
        lib.ai2n_lanes_cuda_lane_size.restype = ctypes.c_int
        lib.ai2n_lanes_cuda_record_size.restype = ctypes.c_int
        if lib.ai2n_lanes_cuda_lane_size() != ctypes.sizeof(Lane) or lib.ai2n_lanes_cuda_record_size() != ctypes.sizeof(EpisodeRecord):
            raise RuntimeError("CUDA lane struct layout differs from the ctypes mirror")
        lib.ai2n_lanes_cuda_create.restype = p
        lib.ai2n_lanes_cuda_create.argtypes = [ctypes.POINTER(episode.Config), ctypes.POINTER(episode.PlatformTable), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.ai2n_lanes_cuda_destroy.restype = ctypes.c_int
        lib.ai2n_lanes_cuda_destroy.argtypes = [p]
        lib.ai2n_lanes_cuda_last_cuda_error.restype = ctypes.c_int
        lib.ai2n_lanes_cuda_last_cuda_error.argtypes = [p]
        lib.ai2n_lanes_cuda_reset.restype = ctypes.c_int
        lib.ai2n_lanes_cuda_reset.argtypes = [p, p, p, p]
        lib.ai2n_lanes_cuda_step.restype = ctypes.c_int
        lib.ai2n_lanes_cuda_step.argtypes = [p] * 8
        lib.ai2n_lanes_cuda_drain.restype = ctypes.c_int
        lib.ai2n_lanes_cuda_drain.argtypes = [p, p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), p]
        lib.ai2n_lanes_cuda_error.restype = ctypes.c_int
        lib.ai2n_lanes_cuda_error.argtypes = [p, ctypes.POINTER(ctypes.c_int), p]
        lib.ai2n_lanes_cuda_synchronize.restype = ctypes.c_int
        lib.ai2n_lanes_cuda_synchronize.argtypes = [p, p]
        lib.ai2n_lanes_cuda_lane_get.restype = ctypes.c_int
        lib.ai2n_lanes_cuda_lane_get.argtypes = [p, ctypes.c_int, ctypes.POINTER(Lane), p]
        lib.ai2n_lanes_cuda_lane_set.restype = ctypes.c_int
        lib.ai2n_lanes_cuda_lane_set.argtypes = [p, ctypes.c_int, ctypes.POINTER(Lane), p]
        self.lanes = int(lanes)
        self.record_capacity = int(record_capacity)
        self.cfg = cfg
        self.table = table
        self._records = (EpisodeRecord * self.record_capacity)()
        rc = ctypes.c_int(0)
        self.handle = lib.ai2n_lanes_cuda_create(ctypes.byref(cfg), ctypes.byref(table), self.lanes, self.record_capacity, ctypes.byref(rc))
        if not self.handle or rc.value:
            raise LaneError(rc.value or 3, "create")

    def _stream(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(self._torch.cuda.current_stream(self.device).cuda_stream)

    def _ptr(self, tensor: Any) -> ctypes.c_void_p:
        if tensor.device != self.device:
            raise ValueError(f"tensor must be on {self.device}")
        return ctypes.c_void_p(tensor.data_ptr())

    def _fail(self, rc: int, where: str) -> LaneError:
        cuda_err = int(self.lib.ai2n_lanes_cuda_last_cuda_error(self.handle))
        err = LaneError(rc, f"{where} (cuda error {cuda_err})")
        return err

    def reset(self, seeds: np.ndarray, obs: Any) -> None:
        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
        rc = self.lib.ai2n_lanes_cuda_reset(self.handle, ctypes.c_void_p(seeds.ctypes.data), self._ptr(obs), self._stream())
        if rc:
            raise self._fail(rc, "reset")

    def step(self, actions: Any, obs: Any, terminal: Any, rewards: Any, terminated: Any, truncated: Any) -> None:
        rc = self.lib.ai2n_lanes_cuda_step(self.handle, self._ptr(actions), self._ptr(obs), self._ptr(terminal), self._ptr(rewards),
                                           self._ptr(terminated), self._ptr(truncated), self._stream())
        if rc:
            raise self._fail(rc, "step")

    def drain(self) -> tuple[list[EpisodeRecord], int]:
        count = ctypes.c_int(0)
        overflow = ctypes.c_int(0)
        rc = self.lib.ai2n_lanes_cuda_drain(self.handle, ctypes.cast(self._records, ctypes.c_void_p), self.record_capacity, ctypes.byref(count),
                                            ctypes.byref(overflow), self._stream())
        if rc:
            raise self._fail(rc, "drain")
        records = [EpisodeRecord.from_buffer_copy(self._records[i]) for i in range(int(count.value))]
        return records, int(overflow.value)

    def error(self) -> tuple[int, int]:
        code = ctypes.c_int(0)
        rc = self.lib.ai2n_lanes_cuda_error(self.handle, ctypes.byref(code), self._stream())
        if rc:
            raise self._fail(rc, "error")
        return int(code.value), -1

    def synchronize(self) -> None:
        rc = self.lib.ai2n_lanes_cuda_synchronize(self.handle, self._stream())
        if rc:
            raise self._fail(rc, "synchronize")

    def lane_state(self, lane: int) -> Lane:
        out = Lane()
        rc = self.lib.ai2n_lanes_cuda_lane_get(self.handle, int(lane), ctypes.byref(out), self._stream())
        if rc:
            raise self._fail(rc, "lane_get")
        return out

    def set_lane_state(self, lane: int, state: Lane) -> None:
        rc = self.lib.ai2n_lanes_cuda_lane_set(self.handle, int(lane), ctypes.byref(state), self._stream())
        if rc:
            raise self._fail(rc, "lane_set")

    def close(self) -> None:
        if self.handle:
            self.lib.ai2n_lanes_cuda_destroy(self.handle)
            self.handle = None


class NativeLaneEnv:
    """``TensorVecEnv`` input backed by native lanes (frame stack included).

    Returned observations alias persistent storage and must be consumed before
    the next step, the same contract as ``TensorFrameStack``.  Completed
    episodes are reported through ``drain_episode_infos()`` once per rollout,
    so no per-step host synchronization happens on the CUDA backend.
    """

    def __init__(self, backend: Any, *, observation_space: Any, action_space: Any, seed: int = 0) -> None:
        import torch

        from ai2.training.tensor_ppo import TensorStep

        _check_spaces(observation_space, action_space)
        self._torch = torch
        self._step_type = TensorStep
        self.backend = backend
        self.device = backend.device
        self.num_envs = int(backend.lanes)
        self.observation_space = observation_space
        self.action_space = action_space
        n, dev = self.num_envs, self.device
        self._obs = torch.zeros((n, OBS_WIDTH), dtype=torch.float32, device=dev)
        self._terminal = torch.zeros((n, OBS_WIDTH), dtype=torch.float32, device=dev)
        self._rewards = torch.zeros((n,), dtype=torch.float32, device=dev)
        self._terminated = torch.zeros((n,), dtype=torch.bool, device=dev)
        self._truncated = torch.zeros((n,), dtype=torch.bool, device=dev)
        self._stack = torch.zeros((n, STACKED_WIDTH), dtype=torch.float32, device=dev)
        self._seeds = np.zeros((n,), dtype=np.uint64)
        self.seed([int(seed) + idx for idx in range(n)])
        self._t_start = time.time()
        self._initialized = False
        self._overflow_warned = False

    @property
    def frame_stack(self) -> Any:
        return self._stack

    def seed(self, seeds: Sequence[int]) -> list[int]:
        if len(seeds) != self.num_envs:
            raise ValueError(f"seeds length must be {self.num_envs}")
        values = []
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
                raise ValueError("seeds must be integers")
            values.append(int(seed) & UINT64_MASK)
        self._seeds = np.asarray(values, dtype=np.uint64)
        return values

    def _raise_if_error(self, where: str) -> None:
        code, lane = self.backend.error()
        if code:
            raise LaneError(code, where, lane)

    def reset_tensor(self, seeds: Optional[Sequence[int]] = None) -> Any:
        if seeds is not None:
            self.seed(seeds)
        self.backend.reset(self._seeds, self._obs)
        self._raise_if_error("reset")
        with self._torch.no_grad():
            self._stack.zero_()
            self._stack[:, -OBS_WIDTH:].copy_(self._obs)
        self._initialized = True
        self._t_start = time.time()
        return self._stack

    def _require_actions(self, actions: Any) -> Any:
        torch = self._torch
        if not torch.is_tensor(actions):
            raise ValueError("clipped_actions must be a torch.Tensor")
        if actions.device != self.device:
            raise ValueError(f"clipped_actions must be on {self.device}")
        if actions.dtype != torch.float32:
            raise ValueError("clipped_actions must be float32")
        if tuple(actions.shape) != (self.num_envs, ACTION_WIDTH):
            raise ValueError(f"clipped_actions shape must be ({self.num_envs}, {ACTION_WIDTH})")
        if actions.requires_grad:
            raise ValueError("clipped_actions must not require grad")
        return actions.contiguous()

    def step_tensor(self, clipped_actions: Any) -> Any:
        if not self._initialized:
            raise RuntimeError("reset_tensor must be called before step_tensor")
        actions = self._require_actions(clipped_actions)
        self.backend.step(actions, self._obs, self._terminal, self._rewards, self._terminated, self._truncated)
        if self.backend.kind == "cpu":
            self._raise_if_error("step")
        return self._advance_stack()

    def _advance_stack(self) -> Any:
        """VecFrameStack semantics: roll, zero the history of done lanes, append the new frame."""
        torch = self._torch
        with torch.no_grad():
            dones = self._terminated | self._truncated
            terminal = torch.cat((self._stack[:, OBS_WIDTH:], self._terminal), dim=1)
            history = terminal[:, :HISTORY_WIDTH].masked_fill(dones[:, None], 0)
            self._stack[:, :HISTORY_WIDTH].copy_(history)
            self._stack[:, -OBS_WIDTH:].copy_(self._obs)
        return self._step_type(observations=self._stack, rewards=self._rewards, terminated=self._terminated, truncated=self._truncated,
                               terminal_observations=terminal)

    # --- replay mode (CPU backend, one lane): drive the lane with the real game's recorded draws ---

    def replay_reset_tensor(self, draws: Sequence[dict[str, Any]]) -> Any:
        if self.backend.kind != "cpu" or self.num_envs != 1:
            raise RuntimeError("replay mode needs the CPU backend with a single lane")
        self.backend.replay_reset(0, draws, self._obs[0])
        self._raise_if_error("replay_reset")
        with self._torch.no_grad():
            self._stack.zero_()
            self._stack[:, -OBS_WIDTH:].copy_(self._obs)
        self._initialized = True
        self._t_start = time.time()
        return self._stack

    def replay_step_tensor(self, clipped_actions: Any, draws: Sequence[dict[str, Any]]) -> Any:
        if self.backend.kind != "cpu" or self.num_envs != 1:
            raise RuntimeError("replay mode needs the CPU backend with a single lane")
        if not self._initialized:
            raise RuntimeError("replay_reset_tensor must be called before replay_step_tensor")
        actions = self._require_actions(clipped_actions)
        rc = self.backend.replay_step(0, draws, actions[0], self._obs[0], self._terminal[0], self._rewards[0:1], self._terminated[0:1], self._truncated[0:1])
        if rc:
            raise LaneError(rc, "replay_step", 0)
        status = self.backend.rng_status(0)
        if status["error"] or status["cursor"] != status["count"]:
            raise LaneError(3, f"replay_step consumed {status['cursor']} of {status['count']} recorded draws (rng error {status['error']})", 0)
        return self._advance_stack()

    def drain_episode_infos(self) -> list[dict[str, Any]]:
        records, overflow = self.backend.drain()
        self._raise_if_error("rollout")
        if overflow and not self._overflow_warned:
            self._overflow_warned = True
            warnings.warn(f"{overflow} completed episodes exceeded the record capacity; episode statistics are incomplete")
        now = time.time()
        infos = []
        for r in records:
            infos.append({
                "episode": {"r": round(float(r.ret), 6), "l": int(r.length), "t": round(now - self._t_start, 6)},
                "navigation": {"lane": int(r.lane), "success": bool(r.success), "targets": int(r.targets), "fell": bool(r.fell),
                               "horizon": bool(r.horizon), "terminated": bool(r.terminated), "truncated": bool(r.truncated),
                               "episode_index": int(r.episode_index)},
            })
        return infos

    def close(self) -> None:
        self.backend.close()


def make_reference(library_path: Path) -> Any:
    """The episode reference API (config finalize, platform table) on the lane library."""
    return episode.EpisodeReference(Path(library_path))


def make_config(reference: Any, wrapper_cfg: dict[str, Any] | None = None, *, dt: float = 1.0 / 30.0, collision_bias: float | None = None) -> Any:
    return reference.make_config(dict(wrapper_cfg or PHASE_0B), dt=dt, collision_bias=collision_bias)


def build_cpu_env(lanes: int, *, build_dir: Path, seed: int = 0, wrapper_cfg: dict[str, Any] | None = None, record_capacity: int | None = None,
                  contract: str | None = None, observation_space: Any = None, action_space: Any = None, dt: float = 1.0 / 30.0,
                  collision_bias: float | None = None) -> NativeLaneEnv:
    library = compile_lanes_cpu(Path(build_dir), contract)
    reference = make_reference(library)
    cfg = make_config(reference, wrapper_cfg, dt=dt, collision_bias=collision_bias)
    table = reference.platform_table(dt)
    if observation_space is None or action_space is None:
        observation_space, action_space = frozen_spaces()
    backend = CpuLaneBackend(library, cfg, table, int(lanes), int(record_capacity or DEFAULT_RECORDS_PER_LANE * int(lanes)))
    return NativeLaneEnv(backend, observation_space=observation_space, action_space=action_space, seed=seed)


def build_cuda_env(lanes: int, *, build_dir: Path, device: Any = "cuda", seed: int = 0, wrapper_cfg: dict[str, Any] | None = None,
                   record_capacity: int | None = None, observation_space: Any = None, action_space: Any = None, dt: float = 1.0 / 30.0,
                   collision_bias: float | None = None) -> NativeLaneEnv:
    """CUDA lanes; the config and platform table come from the host reference library so both hosts agree bit for bit."""
    cpu_library = compile_lanes_cpu(Path(build_dir), "off")
    reference = make_reference(cpu_library)
    cfg = make_config(reference, wrapper_cfg, dt=dt, collision_bias=collision_bias)
    table = reference.platform_table(dt)
    if observation_space is None or action_space is None:
        observation_space, action_space = frozen_spaces()
    library = compile_lanes_cuda(Path(build_dir))
    backend = CudaLaneBackend(library, cfg, table, int(lanes), int(record_capacity or DEFAULT_RECORDS_PER_LANE * int(lanes)), device=device)
    return NativeLaneEnv(backend, observation_space=observation_space, action_space=action_space, seed=seed)
