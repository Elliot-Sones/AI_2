#!/usr/bin/env python3
"""Source-faithful moving-platform reducer prototype for AI_2 navigation.

This module ports only the velocity/direction math from the frozen
``Stage.physics_process`` method plus an optional kinematic position transition
for that same platform slice. It deliberately does not step Pymunk, resolve
contacts, or claim full-GPU simulator behavior.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
NATIVE_DIR = ROOT / "native"
DEFAULT_FROZEN_STAGE_SOURCE_PATH = Path(
    "/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/"
    "baselines/navigation-repair-20260905/payload/variants/corrected/"
    "UTMIST-AI2-main/environment/environment.py"
)
FROZEN_STAGE_LINE_RANGE = "1553-1595"
SCOPE = "platform_velocity_direction_and_optional_kinematic_position_transition_only"
RESIDENT_TIMING_LABEL = "resident_call_dispatch_alloc"
RESIDENT_TIMING_SCOPE = (
    "Resident tensors only; timing includes Python loop, ctypes dispatch, CUDA launch, "
    "and output allocation, not pure kernel time."
)
VELOCITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PlatformStep:
    velocity: tuple[float, float]
    moving_to_w2: bool


@dataclass(frozen=True)
class PlatformKinematicStep:
    position: tuple[float, float]
    velocity: tuple[float, float]
    moving_to_w2: bool


def _pair(value: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    x = float(value[0])
    y = float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{name} must be finite")
    return x, y


def step_platform_scalar(
    *,
    position: Sequence[float],
    waypoint1: Sequence[float],
    waypoint2: Sequence[float],
    moving_to_w2: bool,
) -> PlatformStep:
    """Match frozen ``Stage.physics_process`` velocity and direction update.

    Inputs are canonical current position and waypoints. The output velocity is
    what the frozen method writes to the kinematic Pymunk body for this tick.
    Position integration remains owned by Pymunk in the current contract.
    """

    current_x, current_y = _pair(position, name="position")
    waypoint1_x, waypoint1_y = _pair(waypoint1, name="waypoint1")
    waypoint2_x, waypoint2_y = _pair(waypoint2, name="waypoint2")

    target_x, target_y = (waypoint2_x, waypoint2_y) if moving_to_w2 else (waypoint1_x, waypoint1_y)
    dx = target_x - current_x
    dy = target_y - current_y
    dist = math.sqrt(dx * dx + dy * dy)

    if dist < 0.1:
        return PlatformStep(velocity=(0.0, 0.0), moving_to_w2=not moving_to_w2)

    dir_x = dx / dist
    dir_y = dy / dist

    seg_x = waypoint2_x - waypoint1_x
    seg_y = waypoint2_y - waypoint1_y
    seg_len = max(math.sqrt(seg_x * seg_x + seg_y * seg_y), 0.05)

    rel_x = current_x - waypoint1_x
    rel_y = current_y - waypoint1_y
    progress = (rel_x * seg_x + rel_y * seg_y) / (seg_len * seg_len)
    progress = max(0.0, min(1.0, progress))

    speed = math.sin(progress * math.pi) + 0.03
    return PlatformStep(velocity=(dir_x * speed, dir_y * speed), moving_to_w2=moving_to_w2)


def step_platform_kinematic_scalar(
    *,
    position: Sequence[float],
    waypoint1: Sequence[float],
    waypoint2: Sequence[float],
    moving_to_w2: bool,
    dt: float,
) -> PlatformKinematicStep:
    """Match frozen ordering: Stage sets velocity, then Pymunk integrates it.

    The AI_2 tick calls ``Stage.physics_process(self.dt)`` before
    ``space.step(self.dt)``. E0 traces confirm platform position advances by the
    velocity produced during that same tick, not by the previous velocity.
    """

    current_x, current_y = _pair(position, name="position")
    delta = float(dt)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("dt must be a positive finite value")
    step = step_platform_scalar(
        position=(current_x, current_y),
        waypoint1=waypoint1,
        waypoint2=waypoint2,
        moving_to_w2=moving_to_w2,
    )
    next_position = (
        current_x + step.velocity[0] * delta,
        current_y + step.velocity[1] * delta,
    )
    return PlatformKinematicStep(
        position=next_position,
        velocity=step.velocity,
        moving_to_w2=step.moving_to_w2,
    )


def step_platform_batch_python(
    positions: Sequence[Sequence[float]],
    waypoint1: Sequence[Sequence[float]],
    waypoint2: Sequence[Sequence[float]],
    moving_to_w2: Sequence[bool],
) -> list[PlatformStep]:
    lanes = len(positions)
    if not (len(waypoint1) == len(waypoint2) == len(moving_to_w2) == lanes):
        raise ValueError("positions, waypoint1, waypoint2, and moving_to_w2 must have the same length")
    return [
        step_platform_scalar(
            position=positions[index],
            waypoint1=waypoint1[index],
            waypoint2=waypoint2[index],
            moving_to_w2=bool(moving_to_w2[index]),
        )
        for index in range(lanes)
    ]


def step_platform_kinematic_batch_python(
    positions: Sequence[Sequence[float]],
    waypoint1: Sequence[Sequence[float]],
    waypoint2: Sequence[Sequence[float]],
    moving_to_w2: Sequence[bool],
    *,
    dt: float,
) -> list[PlatformKinematicStep]:
    lanes = len(positions)
    if not (len(waypoint1) == len(waypoint2) == len(moving_to_w2) == lanes):
        raise ValueError("positions, waypoint1, waypoint2, and moving_to_w2 must have the same length")
    return [
        step_platform_kinematic_scalar(
            position=positions[index],
            waypoint1=waypoint1[index],
            waypoint2=waypoint2[index],
            moving_to_w2=bool(moving_to_w2[index]),
            dt=dt,
        )
        for index in range(lanes)
    ]


def _shared_library_suffix() -> str:
    return ".dylib" if platform.system() == "Darwin" else ".so"


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_frozen_source_path(path: Path | None = None) -> Path:
    raw = path or Path(os.environ.get("AI2_FROZEN_STAGE_SOURCE", str(DEFAULT_FROZEN_STAGE_SOURCE_PATH)))
    resolved = raw.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"frozen Stage source file not found: {resolved}")
    return resolved


def source_record(frozen_source_path: Path) -> dict[str, Any]:
    return {
        "frozen_stage_physics_process": f"{frozen_source_path}:{FROZEN_STAGE_LINE_RANGE}",
        "frozen_stage_file": str(frozen_source_path),
        "frozen_stage_line_range": FROZEN_STAGE_LINE_RANGE,
        "frozen_stage_sha256": file_sha256(frozen_source_path),
        "position_integration": "not_owned_pymunk_remains_source_of_truth",
    }


def can_compile_cpu_reference() -> bool:
    compiler = os.environ.get("CC") or "cc"
    return shutil.which(compiler) is not None


def compile_cpu_reference(build_dir: Path) -> dict[str, Any]:
    compiler = os.environ.get("CC") or "cc"
    if shutil.which(compiler) is None:
        raise RuntimeError("no C compiler available for CPU reference")
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / f"libnavigation_platform_cpu{_shared_library_suffix()}"
    command = [
        compiler,
        "-O3",
        "-std=c99",
        "-ffp-contract=off",
        "-fPIC",
        str(NATIVE_DIR / "navigation_platform.c"),
        "-o",
        str(output),
    ]
    if platform.system() == "Darwin":
        command.insert(3, "-dynamiclib")
    else:
        command.insert(3, "-shared")
        command.append("-lm")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"CPU reference compilation failed:\n{completed.stdout}")
    return {"path": str(output), "command": command, "compiler_output": completed.stdout}


class CpuReference:
    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.lib = ctypes.CDLL(str(library_path))
        self._step = self.lib.navigation_platform_step_one
        self._step.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self._step.restype = None
        self._transition = self.lib.navigation_platform_step_kinematic_one
        self._transition.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_uint8,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self._transition.restype = None

    def step(
        self,
        *,
        position: Sequence[float],
        waypoint1: Sequence[float],
        waypoint2: Sequence[float],
        moving_to_w2: bool,
    ) -> PlatformStep:
        current_x, current_y = _pair(position, name="position")
        waypoint1_x, waypoint1_y = _pair(waypoint1, name="waypoint1")
        waypoint2_x, waypoint2_y = _pair(waypoint2, name="waypoint2")
        velocity_x = ctypes.c_double()
        velocity_y = ctypes.c_double()
        next_direction = ctypes.c_uint8()
        self._step(
            ctypes.c_double(current_x),
            ctypes.c_double(current_y),
            ctypes.c_double(waypoint1_x),
            ctypes.c_double(waypoint1_y),
            ctypes.c_double(waypoint2_x),
            ctypes.c_double(waypoint2_y),
            ctypes.c_uint8(1 if moving_to_w2 else 0),
            ctypes.byref(velocity_x),
            ctypes.byref(velocity_y),
            ctypes.byref(next_direction),
        )
        return PlatformStep(
            velocity=(float(velocity_x.value), float(velocity_y.value)),
            moving_to_w2=bool(next_direction.value),
        )

    def transition(
        self,
        *,
        position: Sequence[float],
        waypoint1: Sequence[float],
        waypoint2: Sequence[float],
        moving_to_w2: bool,
        dt: float,
    ) -> PlatformKinematicStep:
        current_x, current_y = _pair(position, name="position")
        waypoint1_x, waypoint1_y = _pair(waypoint1, name="waypoint1")
        waypoint2_x, waypoint2_y = _pair(waypoint2, name="waypoint2")
        delta = float(dt)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("dt must be a positive finite value")
        next_x = ctypes.c_double()
        next_y = ctypes.c_double()
        velocity_x = ctypes.c_double()
        velocity_y = ctypes.c_double()
        next_direction = ctypes.c_uint8()
        self._transition(
            ctypes.c_double(current_x),
            ctypes.c_double(current_y),
            ctypes.c_double(waypoint1_x),
            ctypes.c_double(waypoint1_y),
            ctypes.c_double(waypoint2_x),
            ctypes.c_double(waypoint2_y),
            ctypes.c_uint8(1 if moving_to_w2 else 0),
            ctypes.c_double(delta),
            ctypes.byref(next_x),
            ctypes.byref(next_y),
            ctypes.byref(velocity_x),
            ctypes.byref(velocity_y),
            ctypes.byref(next_direction),
        )
        return PlatformKinematicStep(
            position=(float(next_x.value), float(next_y.value)),
            velocity=(float(velocity_x.value), float(velocity_y.value)),
            moving_to_w2=bool(next_direction.value),
        )


def load_cpu_reference(build_dir: Path) -> CpuReference:
    result = compile_cpu_reference(build_dir)
    return CpuReference(Path(result["path"]))


def _nvcc() -> str | None:
    return shutil.which("nvcc")


def _torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def can_run_cuda_benchmark() -> bool:
    return _nvcc() is not None and _torch_cuda_available()


def normalize_cuda_device(device: str = "cuda", *, current_index: int | None = None) -> str:
    """Return an explicit CUDA device string; reject CPU fallback."""

    if device == "cuda":
        if current_index is None:
            try:
                import torch
            except Exception as exc:
                raise ValueError("bare cuda requires torch to resolve current device") from exc
            current_index = int(torch.cuda.current_device())
        return f"cuda:{int(current_index)}"
    if device.startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        if not suffix.isdigit():
            raise ValueError(f"CUDA device must use an explicit numeric index, got {device!r}")
        return device
    raise ValueError(f"device must be CUDA-only; got {device!r}")


def compile_cuda_library(build_dir: Path) -> dict[str, Any]:
    nvcc = _nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc is not available")
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / "libnavigation_platform_cuda.so"
    command = [
        nvcc,
        "-O3",
        "--fmad=false",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        str(NATIVE_DIR / "navigation_platform.cu"),
        "-o",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"CUDA library compilation failed:\n{completed.stdout}")
    return {"path": str(output), "command": command, "compiler_output": completed.stdout}


class CudaLibrary:
    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.lib = ctypes.CDLL(str(library_path))
        self._step = self.lib.navigation_platform_step_cuda
        self._step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._step.restype = ctypes.c_int
        self._transition = self.lib.navigation_platform_step_kinematic_cuda
        self._transition.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._transition.restype = ctypes.c_int

    def step(self, torch: Any, positions, waypoint1, waypoint2, moving_to_w2):
        input_device = str(getattr(positions, "device", ""))
        for name, tensor in (
            ("positions", positions),
            ("waypoint1", waypoint1),
            ("waypoint2", waypoint2),
        ):
            if not tensor.is_cuda or tensor.dtype != torch.float64 or tensor.ndim != 2 or tensor.shape[1] != 2:
                raise ValueError(f"{name} must be a CUDA float64 tensor with shape [lanes, 2]")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if str(tensor.device) != input_device:
                raise ValueError("all CUDA inputs must be on the same device")
        if not moving_to_w2.is_cuda or moving_to_w2.dtype != torch.uint8 or moving_to_w2.ndim != 1:
            raise ValueError("moving_to_w2 must be a CUDA uint8 tensor with shape [lanes]")
        if not moving_to_w2.is_contiguous():
            raise ValueError("moving_to_w2 must be contiguous")
        if str(moving_to_w2.device) != input_device:
            raise ValueError("all CUDA inputs must be on the same device")
        lanes = int(positions.shape[0])
        if lanes <= 0:
            raise ValueError("lanes must be positive")
        if not (waypoint1.shape[0] == waypoint2.shape[0] == moving_to_w2.shape[0] == lanes):
            raise ValueError("all CUDA inputs must have the same lane count")
        velocities = torch.empty_like(positions)
        next_direction = torch.empty_like(moving_to_w2)
        with torch.cuda.device(input_device):
            stream = ctypes.c_void_p(int(torch.cuda.current_stream(input_device).cuda_stream))
            error = self._step(
                ctypes.c_void_p(positions.data_ptr()),
                ctypes.c_void_p(waypoint1.data_ptr()),
                ctypes.c_void_p(waypoint2.data_ptr()),
                ctypes.c_void_p(moving_to_w2.data_ptr()),
                ctypes.c_void_p(velocities.data_ptr()),
                ctypes.c_void_p(next_direction.data_ptr()),
                ctypes.c_int(lanes),
                stream,
            )
        if error != 0:
            raise RuntimeError(f"navigation_platform_step_cuda failed with cuda error {error}")
        return velocities, next_direction

    def transition(self, torch: Any, positions, waypoint1, waypoint2, moving_to_w2, *, dt: float):
        delta = float(dt)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("dt must be a positive finite value")
        input_device = str(getattr(positions, "device", ""))
        for name, tensor in (
            ("positions", positions),
            ("waypoint1", waypoint1),
            ("waypoint2", waypoint2),
        ):
            if not tensor.is_cuda or tensor.dtype != torch.float64 or tensor.ndim != 2 or tensor.shape[1] != 2:
                raise ValueError(f"{name} must be a CUDA float64 tensor with shape [lanes, 2]")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if str(tensor.device) != input_device:
                raise ValueError("all CUDA inputs must be on the same device")
        if not moving_to_w2.is_cuda or moving_to_w2.dtype != torch.uint8 or moving_to_w2.ndim != 1:
            raise ValueError("moving_to_w2 must be a CUDA uint8 tensor with shape [lanes]")
        if not moving_to_w2.is_contiguous():
            raise ValueError("moving_to_w2 must be contiguous")
        if str(moving_to_w2.device) != input_device:
            raise ValueError("all CUDA inputs must be on the same device")
        lanes = int(positions.shape[0])
        if lanes <= 0:
            raise ValueError("lanes must be positive")
        if not (waypoint1.shape[0] == waypoint2.shape[0] == moving_to_w2.shape[0] == lanes):
            raise ValueError("all CUDA inputs must have the same lane count")
        next_positions = torch.empty_like(positions)
        velocities = torch.empty_like(positions)
        next_direction = torch.empty_like(moving_to_w2)
        with torch.cuda.device(input_device):
            stream = ctypes.c_void_p(int(torch.cuda.current_stream(input_device).cuda_stream))
            error = self._transition(
                ctypes.c_void_p(positions.data_ptr()),
                ctypes.c_void_p(waypoint1.data_ptr()),
                ctypes.c_void_p(waypoint2.data_ptr()),
                ctypes.c_void_p(moving_to_w2.data_ptr()),
                ctypes.c_double(delta),
                ctypes.c_void_p(next_positions.data_ptr()),
                ctypes.c_void_p(velocities.data_ptr()),
                ctypes.c_void_p(next_direction.data_ptr()),
                ctypes.c_int(lanes),
                stream,
            )
        if error != 0:
            raise RuntimeError(f"navigation_platform_step_kinematic_cuda failed with cuda error {error}")
        return next_positions, velocities, next_direction


def load_cuda_library(build_dir: Path) -> CudaLibrary:
    result = compile_cuda_library(build_dir)
    return CudaLibrary(Path(result["path"]))


def generate_fixture_arrays(lanes: int, *, seed: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]], list[bool]]:
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    rng = random.Random(seed)
    positions: list[tuple[float, float]] = []
    waypoint1: list[tuple[float, float]] = []
    waypoint2: list[tuple[float, float]] = []
    moving_to_w2: list[bool] = []
    seeds = [
        ((0.0, 1.0), (1.0, 0.0), (-1.0, 2.0), True),
        ((-0.95, 1.95), (1.0, 0.0), (-1.0, 2.0), True),
        ((0.96, 0.04), (1.0, 0.0), (-1.0, 2.0), False),
        ((1.0, 0.0), (0.0, 0.0), (0.001, 0.0), False),
    ]
    for lane in range(lanes):
        if lane < len(seeds):
            pos, w1, w2, direction = seeds[lane]
        else:
            w1 = (rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 3.0))
            w2 = (rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 3.0))
            pos = (rng.uniform(-8.0, 8.0), rng.uniform(-4.0, 5.0))
            direction = bool(rng.getrandbits(1))
        positions.append(pos)
        waypoint1.append(w1)
        waypoint2.append(w2)
        moving_to_w2.append(direction)
    return positions, waypoint1, waypoint2, moving_to_w2


def _max_abs_velocity_error(actual: Sequence[Sequence[float]], expected: Sequence[PlatformStep]) -> float:
    max_error = 0.0
    for lane, step in enumerate(expected):
        max_error = max(max_error, abs(float(actual[lane][0]) - step.velocity[0]))
        max_error = max(max_error, abs(float(actual[lane][1]) - step.velocity[1]))
    return max_error


def _max_abs_kinematic_velocity_error(
    actual: Sequence[Sequence[float]], expected: Sequence[PlatformKinematicStep]
) -> float:
    max_error = 0.0
    for lane, step in enumerate(expected):
        max_error = max(max_error, abs(float(actual[lane][0]) - step.velocity[0]))
        max_error = max(max_error, abs(float(actual[lane][1]) - step.velocity[1]))
    return max_error


def _max_abs_position_error(actual: Sequence[Sequence[float]], expected: Sequence[PlatformKinematicStep]) -> float:
    max_error = 0.0
    for lane, step in enumerate(expected):
        max_error = max(max_error, abs(float(actual[lane][0]) - step.position[0]))
        max_error = max(max_error, abs(float(actual[lane][1]) - step.position[1]))
    return max_error


def _cuda_parity_for_arrays(
    torch: Any,
    cuda: CudaLibrary,
    *,
    positions: Sequence[Sequence[float]],
    waypoint1: Sequence[Sequence[float]],
    waypoint2: Sequence[Sequence[float]],
    directions: Sequence[bool],
    device: str,
) -> dict[str, Any]:
    expected = step_platform_batch_python(positions, waypoint1, waypoint2, directions)
    cuda_device = normalize_cuda_device(device, current_index=int(torch.cuda.current_device()))
    torch_device = torch.device(cuda_device)
    positions_gpu = torch.tensor(positions, dtype=torch.float64, device=torch_device).contiguous()
    waypoint1_gpu = torch.tensor(waypoint1, dtype=torch.float64, device=torch_device).contiguous()
    waypoint2_gpu = torch.tensor(waypoint2, dtype=torch.float64, device=torch_device).contiguous()
    directions_gpu = torch.tensor([1 if item else 0 for item in directions], dtype=torch.uint8, device=torch_device)
    velocities_gpu, next_gpu = cuda.step(torch, positions_gpu, waypoint1_gpu, waypoint2_gpu, directions_gpu)
    torch.cuda.synchronize(torch_device)
    finite = bool(torch.isfinite(velocities_gpu).all().item())
    velocities = velocities_gpu.cpu().tolist()
    next_direction = [bool(item) for item in next_gpu.cpu().tolist()]
    mismatches = sum(
        1 for lane, step in enumerate(expected) if next_direction[lane] != step.moving_to_w2
    )
    return {
        "finite": finite,
        "max_abs_velocity_error": _max_abs_velocity_error(velocities, expected),
        "direction_mismatches": mismatches,
    }


def _cuda_kinematic_parity_for_arrays(
    torch: Any,
    cuda: CudaLibrary,
    *,
    positions: Sequence[Sequence[float]],
    waypoint1: Sequence[Sequence[float]],
    waypoint2: Sequence[Sequence[float]],
    directions: Sequence[bool],
    device: str,
    dt: float,
) -> dict[str, Any]:
    expected = step_platform_kinematic_batch_python(positions, waypoint1, waypoint2, directions, dt=dt)
    cuda_device = normalize_cuda_device(device, current_index=int(torch.cuda.current_device()))
    torch_device = torch.device(cuda_device)
    positions_gpu = torch.tensor(positions, dtype=torch.float64, device=torch_device).contiguous()
    waypoint1_gpu = torch.tensor(waypoint1, dtype=torch.float64, device=torch_device).contiguous()
    waypoint2_gpu = torch.tensor(waypoint2, dtype=torch.float64, device=torch_device).contiguous()
    directions_gpu = torch.tensor([1 if item else 0 for item in directions], dtype=torch.uint8, device=torch_device)
    next_positions_gpu, velocities_gpu, next_gpu = cuda.transition(
        torch,
        positions_gpu,
        waypoint1_gpu,
        waypoint2_gpu,
        directions_gpu,
        dt=dt,
    )
    torch.cuda.synchronize(torch_device)
    finite = bool(torch.isfinite(next_positions_gpu).all().item() and torch.isfinite(velocities_gpu).all().item())
    next_positions = next_positions_gpu.cpu().tolist()
    velocities = velocities_gpu.cpu().tolist()
    next_direction = [bool(item) for item in next_gpu.cpu().tolist()]
    mismatches = sum(
        1 for lane, step in enumerate(expected) if next_direction[lane] != step.moving_to_w2
    )
    return {
        "finite": finite,
        "max_abs_velocity_error": _max_abs_kinematic_velocity_error(velocities, expected),
        "max_abs_position_error": _max_abs_position_error(next_positions, expected),
        "direction_mismatches": mismatches,
    }


def run_cuda_parity_once(
    lanes: int,
    *,
    seed: int = 20260906,
    build_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    if not can_run_cuda_benchmark():
        raise RuntimeError("CUDA torch and nvcc are required; refusing CPU fallback for GPU benchmark")
    import torch

    owned_temp = None
    if build_dir is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="ai2-nav-platform-cuda-")
        build_dir = Path(owned_temp.name)
    try:
        cuda = load_cuda_library(build_dir)
        positions, waypoint1, waypoint2, directions = generate_fixture_arrays(lanes, seed=seed)
        parity = _cuda_parity_for_arrays(
            torch,
            cuda,
            positions=positions,
            waypoint1=waypoint1,
            waypoint2=waypoint2,
            directions=directions,
            device=device,
        )
        order = list(range(lanes))
        random.Random(seed + 1).shuffle(order)
        permutation = _cuda_parity_for_arrays(
            torch,
            cuda,
            positions=[positions[index] for index in order],
            waypoint1=[waypoint1[index] for index in order],
            waypoint2=[waypoint2[index] for index in order],
            directions=[directions[index] for index in order],
            device=device,
        )
        return {
            "lanes": lanes,
            **parity,
            "permutation": permutation,
        }
    finally:
        if owned_temp is not None:
            owned_temp.cleanup()


def run_cuda_kinematic_parity_once(
    lanes: int,
    *,
    seed: int = 20260906,
    build_dir: Path | None = None,
    device: str = "cuda",
    dt: float = 1.0 / 30.0,
) -> dict[str, Any]:
    if not can_run_cuda_benchmark():
        raise RuntimeError("CUDA torch and nvcc are required; refusing CPU fallback for GPU benchmark")
    import torch

    owned_temp = None
    if build_dir is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="ai2-nav-platform-cuda-")
        build_dir = Path(owned_temp.name)
    try:
        cuda = load_cuda_library(build_dir)
        positions, waypoint1, waypoint2, directions = generate_fixture_arrays(lanes, seed=seed)
        parity = _cuda_kinematic_parity_for_arrays(
            torch,
            cuda,
            positions=positions,
            waypoint1=waypoint1,
            waypoint2=waypoint2,
            directions=directions,
            device=device,
            dt=dt,
        )
        order = list(range(lanes))
        random.Random(seed + 1).shuffle(order)
        permutation = _cuda_kinematic_parity_for_arrays(
            torch,
            cuda,
            positions=[positions[index] for index in order],
            waypoint1=[waypoint1[index] for index in order],
            waypoint2=[waypoint2[index] for index in order],
            directions=[directions[index] for index in order],
            device=device,
            dt=dt,
        )
        return {
            "lanes": lanes,
            "dt": dt,
            **parity,
            "permutation": permutation,
        }
    finally:
        if owned_temp is not None:
            owned_temp.cleanup()


def run_cuda_benchmark(
    lanes: int,
    *,
    iterations: int,
    warmup: int,
    seed: int,
    build_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    if not can_run_cuda_benchmark():
        raise RuntimeError("CUDA torch and nvcc are required; refusing CPU fallback for GPU benchmark")
    import torch

    cuda = load_cuda_library(build_dir)
    positions, waypoint1, waypoint2, directions = generate_fixture_arrays(lanes, seed=seed)
    expected = step_platform_batch_python(positions, waypoint1, waypoint2, directions)
    directions_u8 = [1 if item else 0 for item in directions]
    cuda_device = normalize_cuda_device(device, current_index=int(torch.cuda.current_device()))
    torch_device = torch.device(cuda_device)

    positions_cpu = torch.tensor(positions, dtype=torch.float64).contiguous()
    waypoint1_cpu = torch.tensor(waypoint1, dtype=torch.float64).contiguous()
    waypoint2_cpu = torch.tensor(waypoint2, dtype=torch.float64).contiguous()
    directions_cpu = torch.tensor(directions_u8, dtype=torch.uint8)

    positions_gpu = positions_cpu.to(torch_device)
    waypoint1_gpu = waypoint1_cpu.to(torch_device)
    waypoint2_gpu = waypoint2_cpu.to(torch_device)
    directions_gpu = directions_cpu.to(torch_device)

    for _ in range(warmup):
        resident_velocities, resident_direction = cuda.step(
            torch, positions_gpu, waypoint1_gpu, waypoint2_gpu, directions_gpu
        )
    torch.cuda.synchronize(torch_device)
    torch.cuda.reset_peak_memory_stats(torch_device)
    start = time.perf_counter()
    for _ in range(iterations):
        resident_velocities, resident_direction = cuda.step(
            torch, positions_gpu, waypoint1_gpu, waypoint2_gpu, directions_gpu
        )
    torch.cuda.synchronize(torch_device)
    resident_seconds = time.perf_counter() - start
    resident_peak_memory = torch.cuda.max_memory_allocated(torch_device)

    resident_finite = bool(torch.isfinite(resident_velocities).all().item())
    resident_velocities_cpu = resident_velocities.cpu().tolist()
    resident_direction_cpu = [bool(item) for item in resident_direction.cpu().tolist()]

    for _ in range(warmup):
        positions_gpu_loop = positions_cpu.to(torch_device)
        waypoint1_gpu_loop = waypoint1_cpu.to(torch_device)
        waypoint2_gpu_loop = waypoint2_cpu.to(torch_device)
        directions_gpu_loop = directions_cpu.to(torch_device)
        roundtrip_velocities, roundtrip_direction = cuda.step(
            torch,
            positions_gpu_loop,
            waypoint1_gpu_loop,
            waypoint2_gpu_loop,
            directions_gpu_loop,
        )
        roundtrip_velocities.cpu()
        roundtrip_direction.cpu()
    torch.cuda.synchronize(torch_device)
    torch.cuda.reset_peak_memory_stats(torch_device)
    start = time.perf_counter()
    for _ in range(iterations):
        positions_gpu_loop = positions_cpu.to(torch_device)
        waypoint1_gpu_loop = waypoint1_cpu.to(torch_device)
        waypoint2_gpu_loop = waypoint2_cpu.to(torch_device)
        directions_gpu_loop = directions_cpu.to(torch_device)
        roundtrip_velocities, roundtrip_direction = cuda.step(
            torch,
            positions_gpu_loop,
            waypoint1_gpu_loop,
            waypoint2_gpu_loop,
            directions_gpu_loop,
        )
        roundtrip_velocities_host = roundtrip_velocities.cpu()
        roundtrip_direction_host = roundtrip_direction.cpu()
    torch.cuda.synchronize(torch_device)
    roundtrip_seconds = time.perf_counter() - start
    roundtrip_peak_memory = torch.cuda.max_memory_allocated(torch_device)

    roundtrip_finite = bool(torch.isfinite(roundtrip_velocities_host).all().item())
    roundtrip_velocities_cpu = roundtrip_velocities_host.tolist()
    roundtrip_direction_cpu = [bool(item) for item in roundtrip_direction_host.tolist()]

    resident_mismatches = sum(
        1 for lane, step in enumerate(expected) if resident_direction_cpu[lane] != step.moving_to_w2
    )
    roundtrip_mismatches = sum(
        1 for lane, step in enumerate(expected) if roundtrip_direction_cpu[lane] != step.moving_to_w2
    )
    report = {
        "lanes": lanes,
        "device": cuda_device,
        "iterations": iterations,
        "warmup": warmup,
        RESIDENT_TIMING_LABEL: {
            "timing_scope": RESIDENT_TIMING_SCOPE,
            "seconds": resident_seconds,
            "calls_per_second": iterations / resident_seconds if resident_seconds else math.inf,
            "lanes_per_second": (iterations * lanes) / resident_seconds if resident_seconds else math.inf,
            "peak_memory_bytes": int(resident_peak_memory),
            "finite": resident_finite,
            "max_abs_velocity_error": _max_abs_velocity_error(resident_velocities_cpu, expected),
            "direction_mismatches": resident_mismatches,
        },
        "host_roundtrip": {
            "seconds": roundtrip_seconds,
            "calls_per_second": iterations / roundtrip_seconds if roundtrip_seconds else math.inf,
            "lanes_per_second": (iterations * lanes) / roundtrip_seconds if roundtrip_seconds else math.inf,
            "peak_memory_bytes": int(roundtrip_peak_memory),
            "finite": roundtrip_finite,
            "max_abs_velocity_error": _max_abs_velocity_error(roundtrip_velocities_cpu, expected),
            "direction_mismatches": roundtrip_mismatches,
        },
    }
    validate_cuda_benchmark_report(report, velocity_tolerance=VELOCITY_TOLERANCE)
    return report


def validate_cuda_benchmark_report(report: dict[str, Any], *, velocity_tolerance: float) -> None:
    for section in (RESIDENT_TIMING_LABEL, "host_roundtrip"):
        metrics = report.get(section)
        if not isinstance(metrics, dict):
            raise RuntimeError(f"missing GPU benchmark section {section!r}")
        if not metrics.get("finite", False):
            raise RuntimeError(f"non-finite CUDA platform output in {section}")
        velocity_error = float(metrics.get("max_abs_velocity_error", math.inf))
        if velocity_error > velocity_tolerance:
            raise RuntimeError(
                f"CUDA platform velocity error {velocity_error} exceeds tolerance {velocity_tolerance} in {section}"
            )
        direction_mismatches = int(metrics.get("direction_mismatches", -1))
        if direction_mismatches != 0:
            raise RuntimeError(f"CUDA platform direction mismatches in {section}: {direction_mismatches}")


def _transition_is_reset_boundary(record: dict[str, Any]) -> bool:
    transition = record.get("transition") or {}
    return bool(transition.get("auto_reset") or transition.get("post_reset_state"))


def _transition_is_terminal_boundary(record: dict[str, Any]) -> bool:
    transition = record.get("transition") or {}
    return _transition_is_reset_boundary(record) or bool(transition.get("terminated") or transition.get("truncated"))


def _platforms_by_id(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    state = record.get("state")
    if not isinstance(state, dict):
        return {}
    platforms = state.get("moving_platforms")
    if not isinstance(platforms, list):
        return {}
    return {str(platform.get("id")): platform for platform in platforms if isinstance(platform, dict)}


def validate_replay_parity_report(report: dict[str, Any], *, velocity_tolerance: float) -> None:
    if int(report.get("checked_transitions", 0)) <= 0:
        raise RuntimeError(f"replay parity found no comparable platform transitions in {report.get('replay_file')}")
    velocity_error = float(report.get("max_abs_velocity_error", math.inf))
    if velocity_error > velocity_tolerance:
        raise RuntimeError(
            f"replay platform velocity error {velocity_error} exceeds tolerance "
            f"{velocity_tolerance} in {report.get('replay_file')}"
        )
    direction_mismatches = int(report.get("direction_mismatches", -1))
    if direction_mismatches != 0:
        raise RuntimeError(
            f"replay platform direction mismatches in {report.get('replay_file')}: {direction_mismatches}"
        )


def validate_kinematic_replay_parity_report(
    report: dict[str, Any],
    *,
    velocity_tolerance: float,
    position_tolerance: float,
) -> None:
    validate_replay_parity_report(report, velocity_tolerance=velocity_tolerance)
    position_error = float(report.get("max_abs_position_error", math.inf))
    if position_error > position_tolerance:
        raise RuntimeError(
            f"replay platform position error {position_error} exceeds tolerance "
            f"{position_tolerance} in {report.get('replay_file')}"
        )


def replay_file_scalar_parity(
    replay_file: Path,
    *,
    velocity_tolerance: float = VELOCITY_TOLERANCE,
) -> dict[str, Any]:
    """Compare the scalar reducer against consecutive E0 replay platform states.

    For each comparable transition, the previous record's platform
    position/waypoints/direction are treated as the reducer input and compared
    with the current record's captured platform velocity/direction. Reset,
    auto-reset, non-consecutive, and post-terminal boundaries are skipped.
    """

    path = replay_file.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"replay file not found: {path}")

    records_seen = 0
    checked = 0
    skipped_boundaries = 0
    skipped_missing_platform = 0
    max_abs_velocity_error = 0.0
    direction_mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    prev: dict[str, Any] | None = None

    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") != "record":
                continue
            records_seen += 1
            tick = record.get("tick")
            platforms = _platforms_by_id(record)
            if prev is None:
                skipped_boundaries += 1
            else:
                comparable = (
                    isinstance(tick, int)
                    and isinstance(prev["tick"], int)
                    and tick == prev["tick"] + 1
                    and not _transition_is_reset_boundary(record)
                    and not prev["terminal_boundary"]
                )
                if not comparable:
                    skipped_boundaries += 1
                else:
                    for platform_id, previous_platform in prev["platforms"].items():
                        current_platform = platforms.get(platform_id)
                        if current_platform is None:
                            skipped_missing_platform += 1
                            continue
                        expected = step_platform_scalar(
                            position=previous_platform["position"],
                            waypoint1=previous_platform["waypoint1"],
                            waypoint2=previous_platform["waypoint2"],
                            moving_to_w2=bool(previous_platform["moving_to_w2"]),
                        )
                        current_velocity = _pair(current_platform["velocity"], name="current velocity")
                        error = max(
                            abs(expected.velocity[0] - current_velocity[0]),
                            abs(expected.velocity[1] - current_velocity[1]),
                        )
                        direction_match = expected.moving_to_w2 == bool(current_platform["moving_to_w2"])
                        checked += 1
                        max_abs_velocity_error = max(max_abs_velocity_error, error)
                        if not direction_match:
                            direction_mismatches += 1
                        if first_mismatch is None and (error > velocity_tolerance or not direction_match):
                            first_mismatch = {
                                "line_number": line_number,
                                "tick": tick,
                                "platform_id": platform_id,
                                "velocity_error": error,
                                "expected_velocity": expected.velocity,
                                "actual_velocity": current_velocity,
                                "expected_moving_to_w2": expected.moving_to_w2,
                                "actual_moving_to_w2": bool(current_platform["moving_to_w2"]),
                            }
            prev = {
                "tick": tick,
                "platforms": platforms,
                "terminal_boundary": _transition_is_terminal_boundary(record),
            }

    report = {
        "replay_file": str(path),
        "replay_file_sha256": file_sha256(path),
        "records_seen": records_seen,
        "checked_transitions": checked,
        "skipped_boundaries": skipped_boundaries,
        "skipped_missing_platform": skipped_missing_platform,
        "velocity_tolerance": velocity_tolerance,
        "max_abs_velocity_error": max_abs_velocity_error,
        "direction_mismatches": direction_mismatches,
        "first_mismatch": first_mismatch,
        "scope": "scalar_reducer_vs_consecutive_e0_platform_records",
        "boundary_policy": "skip_reset_auto_reset_nonconsecutive_and_post_terminal_records",
    }
    validate_replay_parity_report(report, velocity_tolerance=velocity_tolerance)
    return report


def replay_file_kinematic_parity(
    replay_file: Path,
    *,
    dt: float,
    velocity_tolerance: float = VELOCITY_TOLERANCE,
    position_tolerance: float = VELOCITY_TOLERANCE,
) -> dict[str, Any]:
    """Compare reducer+same-tick kinematic integration against E0 traces."""

    delta = float(dt)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("dt must be a positive finite value")
    path = replay_file.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"replay file not found: {path}")

    records_seen = 0
    checked = 0
    skipped_boundaries = 0
    skipped_missing_platform = 0
    max_abs_velocity_error = 0.0
    max_abs_position_error = 0.0
    direction_mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    prev: dict[str, Any] | None = None

    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") != "record":
                continue
            records_seen += 1
            tick = record.get("tick")
            platforms = _platforms_by_id(record)
            if prev is None:
                skipped_boundaries += 1
            else:
                comparable = (
                    isinstance(tick, int)
                    and isinstance(prev["tick"], int)
                    and tick == prev["tick"] + 1
                    and not _transition_is_reset_boundary(record)
                    and not prev["terminal_boundary"]
                )
                if not comparable:
                    skipped_boundaries += 1
                else:
                    for platform_id, previous_platform in prev["platforms"].items():
                        current_platform = platforms.get(platform_id)
                        if current_platform is None:
                            skipped_missing_platform += 1
                            continue
                        expected = step_platform_kinematic_scalar(
                            position=previous_platform["position"],
                            waypoint1=previous_platform["waypoint1"],
                            waypoint2=previous_platform["waypoint2"],
                            moving_to_w2=bool(previous_platform["moving_to_w2"]),
                            dt=delta,
                        )
                        current_velocity = _pair(current_platform["velocity"], name="current velocity")
                        current_position = _pair(current_platform["position"], name="current position")
                        velocity_error = max(
                            abs(expected.velocity[0] - current_velocity[0]),
                            abs(expected.velocity[1] - current_velocity[1]),
                        )
                        position_error = max(
                            abs(expected.position[0] - current_position[0]),
                            abs(expected.position[1] - current_position[1]),
                        )
                        direction_match = expected.moving_to_w2 == bool(current_platform["moving_to_w2"])
                        checked += 1
                        max_abs_velocity_error = max(max_abs_velocity_error, velocity_error)
                        max_abs_position_error = max(max_abs_position_error, position_error)
                        if not direction_match:
                            direction_mismatches += 1
                        if first_mismatch is None and (
                            velocity_error > velocity_tolerance
                            or position_error > position_tolerance
                            or not direction_match
                        ):
                            first_mismatch = {
                                "line_number": line_number,
                                "tick": tick,
                                "platform_id": platform_id,
                                "velocity_error": velocity_error,
                                "position_error": position_error,
                                "expected_position": expected.position,
                                "actual_position": current_position,
                                "expected_velocity": expected.velocity,
                                "actual_velocity": current_velocity,
                                "expected_moving_to_w2": expected.moving_to_w2,
                                "actual_moving_to_w2": bool(current_platform["moving_to_w2"]),
                            }
            prev = {
                "tick": tick,
                "platforms": platforms,
                "terminal_boundary": _transition_is_terminal_boundary(record),
            }

    report = {
        "replay_file": str(path),
        "replay_file_sha256": file_sha256(path),
        "records_seen": records_seen,
        "checked_transitions": checked,
        "skipped_boundaries": skipped_boundaries,
        "skipped_missing_platform": skipped_missing_platform,
        "dt": delta,
        "velocity_tolerance": velocity_tolerance,
        "position_tolerance": position_tolerance,
        "max_abs_velocity_error": max_abs_velocity_error,
        "max_abs_position_error": max_abs_position_error,
        "direction_mismatches": direction_mismatches,
        "first_mismatch": first_mismatch,
        "scope": "kinematic_position_velocity_direction_vs_consecutive_e0_platform_records",
        "ordering": "Stage.physics_process_sets_velocity_before_space_step_integrates_position",
        "boundary_policy": "skip_reset_auto_reset_nonconsecutive_and_post_terminal_records",
    }
    validate_kinematic_replay_parity_report(
        report,
        velocity_tolerance=velocity_tolerance,
        position_tolerance=position_tolerance,
    )
    return report


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git_record() -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    return {"commit": run_git(["rev-parse", "HEAD"]), "dirty": bool(run_git(["status", "--short"]))}


def provenance(seed: int, frozen_source_path: Path) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_record(),
        "versions": {
            "torch": _package_version("torch"),
        },
        "source_sha256": {
            "native_navigation_platform.py": file_sha256(ROOT / "native_navigation_platform.py"),
            "native/navigation_platform.h": file_sha256(NATIVE_DIR / "navigation_platform.h"),
            "native/navigation_platform.c": file_sha256(NATIVE_DIR / "navigation_platform.c"),
            "native/navigation_platform.cu": file_sha256(NATIVE_DIR / "navigation_platform.cu"),
            "frozen_stage_source": file_sha256(frozen_source_path),
        },
    }


def scalar_payload(seed: int, *, frozen_source_path: Path | None = None) -> dict[str, Any]:
    resolved_source = resolve_frozen_source_path(frozen_source_path)
    fixtures = generate_fixture_arrays(8, seed=seed)
    outputs = step_platform_batch_python(*fixtures)
    cpu_reference: dict[str, Any]
    if can_compile_cpu_reference():
        with tempfile.TemporaryDirectory(prefix="ai2-nav-platform-cpu-") as tmp:
            try:
                compile_result = compile_cpu_reference(Path(tmp))
                cpu_reference = {
                    "available": True,
                    "compiled": True,
                    "command": compile_result["command"],
                    "compiler_output": compile_result["compiler_output"],
                }
            except RuntimeError as exc:
                cpu_reference = {"available": True, "compiled": False, "error": str(exc)}
    else:
        cpu_reference = {"available": False, "compiled": False}
    return {
        "mode": "scalar",
        "scope": SCOPE,
        "source": source_record(resolved_source),
        "provenance": provenance(seed, resolved_source),
        "cpu_reference": cpu_reference,
        "fixtures": [
            {
                "position": fixtures[0][index],
                "waypoint1": fixtures[1][index],
                "waypoint2": fixtures[2][index],
                "moving_to_w2": fixtures[3][index],
                "velocity": outputs[index].velocity,
                "next_moving_to_w2": outputs[index].moving_to_w2,
            }
            for index in range(len(outputs))
        ],
    }


def parse_lanes(raw: str) -> list[int]:
    try:
        lanes = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("lanes must be comma-separated positive integers") from exc
    if not lanes or any(item <= 0 for item in lanes):
        raise argparse.ArgumentTypeError("lanes must be comma-separated positive integers")
    return lanes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Source-faithful AI_2 moving-platform velocity reducer. "
            "This is not a full GPU game engine or training benchmark."
        )
    )
    parser.add_argument("--output", required=True, type=Path, help="JSON output path")
    parser.add_argument("--mode", choices=("scalar", "gpu"), default="scalar")
    parser.add_argument("--lanes", type=parse_lanes, default=parse_lanes("256,1024,4096"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--dt", type=float, default=1.0 / 30.0, help="Tick dt for kinematic replay parity")
    parser.add_argument("--device", default="cuda", help="CUDA device; bare cuda is normalized to cuda:<current>")
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument(
        "--frozen-source",
        type=Path,
        default=None,
        help="Override frozen environment.py source path for provenance/hash on remote runners",
    )
    parser.add_argument(
        "--replay-file",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional E0 replay JSONL trace to validate consecutive moving-platform "
            "velocity/direction transitions with the scalar reducer"
        ),
    )
    parser.add_argument(
        "--replay-velocity-tolerance",
        type=float,
        default=VELOCITY_TOLERANCE,
        help="Absolute velocity tolerance for --replay-file parity checks",
    )
    return parser


def write_payload(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.iterations <= 0 or args.warmup <= 0:
        raise RuntimeError("iterations and warmup must be positive")
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise RuntimeError("dt must be a positive finite value")

    if args.mode == "scalar":
        frozen_source_path = resolve_frozen_source_path(args.frozen_source)
        payload = scalar_payload(args.seed, frozen_source_path=frozen_source_path)
    else:
        frozen_source_path = resolve_frozen_source_path(args.frozen_source)
        if not can_run_cuda_benchmark():
            raise RuntimeError("CUDA torch and nvcc are required; refusing CPU fallback for GPU benchmark")
        owned_temp = None
        build_dir = args.build_dir
        if build_dir is None:
            owned_temp = tempfile.TemporaryDirectory(prefix="ai2-nav-platform-gpu-")
            build_dir = Path(owned_temp.name)
        try:
            payload = {
                "mode": "gpu",
                "scope": SCOPE,
                "source": source_record(frozen_source_path),
                "provenance": provenance(args.seed, frozen_source_path),
                "gpu": {
                    "lanes": args.lanes,
                    "iterations": args.iterations,
                    "warmup": args.warmup,
                    "benchmark_kind": "platform_velocity_reducer_only_not_training",
                    "benchmarks": [
                        run_cuda_benchmark(
                            lanes=lanes,
                            iterations=args.iterations,
                            warmup=args.warmup,
                            seed=args.seed + lanes,
                            build_dir=build_dir,
                            device=args.device,
                        )
                        for lanes in args.lanes
                    ],
                },
            }
        finally:
            if owned_temp is not None:
                owned_temp.cleanup()
    if args.replay_file:
        payload["replay_parity"] = [
            replay_file_scalar_parity(path, velocity_tolerance=args.replay_velocity_tolerance)
            for path in args.replay_file
        ]
        payload["kinematic_replay_parity"] = [
            replay_file_kinematic_parity(
                path,
                dt=args.dt,
                velocity_tolerance=args.replay_velocity_tolerance,
                position_tolerance=args.replay_velocity_tolerance,
            )
            for path in args.replay_file
        ]
    write_payload(payload, args.output)
    print(f"native_navigation_platform {payload['mode']} wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
