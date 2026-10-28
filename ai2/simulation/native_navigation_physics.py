#!/usr/bin/env python3
"""First source-faithful native physics foundation for AI_2 navigation.

This module ports only Chipmunk/Pymunk 6.2.1 body integration:
``cpSpaceStep`` integrates body positions before velocities, and the formulas
match ``cpBodyUpdatePosition`` / ``cpBodyUpdateVelocity``. It deliberately does
not implement convex contact generation, arbiter warm-starting, collision
callbacks, or the sequential impulse solver.
"""

from __future__ import annotations

from ai2.paths import REPO_ROOT as PROJECT_ROOT

import ctypes
import math
import os
import platform
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = PROJECT_ROOT
NATIVE_DIR = ROOT / "native"

BODY_STATE_FIELD_COUNT = 18

BODY_TYPE_DYNAMIC = 0
BODY_TYPE_KINEMATIC = 1
BODY_TYPE_STATIC = 2

VELOCITY_MODE_DEFAULT = 0
VELOCITY_MODE_NO_GRAVITY = 1

OK = 0
ERR_NULL = 1
ERR_INVALID_LANES = 2
ERR_NONFINITE = 3
ERR_INVALID_BODY_TYPE = 4
ERR_INVALID_VELOCITY_MODE = 5
ERR_INVALID_DYNAMIC_MASS = 6
ERR_INVALID_TIMESTEP = 7


@dataclass(frozen=True)
class BodyState:
    body_type: int = BODY_TYPE_DYNAMIC
    velocity_mode: int = VELOCITY_MODE_DEFAULT
    position: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    force: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    angular_velocity: float = 0.0
    torque: float = 0.0
    mass: float = 1.0
    moment: float = 1e9
    center_of_gravity: tuple[float, float] = (0.0, 0.0)
    velocity_bias: tuple[float, float] = (0.0, 0.0)
    angular_velocity_bias: float = 0.0


def _pair(value: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    x = float(value[0])
    y = float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{name} must be finite")
    return x, y


def _validate_state(state: BodyState) -> None:
    body_type_float = float(state.body_type)
    velocity_mode_float = float(state.velocity_mode)
    if not math.isfinite(body_type_float) or body_type_float != float(int(body_type_float)):
        raise ValueError(f"invalid body_type: {state.body_type!r}")
    if not math.isfinite(velocity_mode_float) or velocity_mode_float != float(int(velocity_mode_float)):
        raise ValueError(f"invalid velocity_mode: {state.velocity_mode!r}")
    if int(body_type_float) not in (BODY_TYPE_DYNAMIC, BODY_TYPE_KINEMATIC, BODY_TYPE_STATIC):
        raise ValueError(f"invalid body_type: {state.body_type!r}")
    if int(velocity_mode_float) not in (VELOCITY_MODE_DEFAULT, VELOCITY_MODE_NO_GRAVITY):
        raise ValueError(f"invalid velocity_mode: {state.velocity_mode!r}")
    _pair(state.position, name="position")
    _pair(state.velocity, name="velocity")
    _pair(state.force, name="force")
    center_of_gravity = _pair(state.center_of_gravity, name="center_of_gravity")
    if center_of_gravity != (0.0, 0.0):
        raise ValueError("nonzero center_of_gravity is not ported by this body integration slice")
    _pair(state.velocity_bias, name="velocity_bias")
    for name in ("angle", "angular_velocity", "torque", "angular_velocity_bias"):
        if not math.isfinite(float(getattr(state, name))):
            raise ValueError(f"{name} must be finite")
    mass = float(state.mass)
    moment = float(state.moment)
    if math.isnan(mass) or math.isnan(moment):
        raise ValueError("mass and moment must not be NaN")
    if int(state.body_type) == BODY_TYPE_DYNAMIC and (
        not math.isfinite(mass) or not math.isfinite(moment) or mass <= 0.0 or moment <= 0.0
    ):
        raise ValueError("dynamic mass and moment must be positive finite values")


def _validate_step(
    gravity: Sequence[float],
    damping_factor: float,
    dt: float,
) -> tuple[tuple[float, float], float, float]:
    gravity_pair = _pair(gravity, name="gravity")
    damping_float = float(damping_factor)
    dt_float = float(dt)
    if not math.isfinite(damping_float) or damping_float < 0.0:
        raise ValueError("damping_factor must be finite and non-negative")
    if not math.isfinite(dt_float) or dt_float < 0.0:
        raise ValueError("dt must be finite and non-negative")
    return gravity_pair, damping_float, dt_float


def step_body_scalar(
    state: BodyState,
    *,
    gravity: Sequence[float],
    damping_factor: float | None = None,
    dt: float,
    space_damping: float | None = None,
) -> BodyState:
    """Apply Chipmunk 7.0.3 body integration for one body.

    This follows ``cpSpaceStep`` ordering: position first, then velocity. Bias
    velocities affect only the position update and are reset after use. Dynamic
    bodies then receive default velocity integration; no-gravity mode is the
    frozen Dodge/KO callback behavior of calling the default updater with zero
    gravity. Kinematic bodies skip velocity integration.
    """

    if damping_factor is None:
        if space_damping is None:
            raise TypeError("step_body_scalar requires damping_factor or space_damping")
        if not math.isfinite(float(space_damping)) or float(space_damping) < 0.0:
            raise ValueError("space_damping must be finite and non-negative")
        damping_factor = math.pow(float(space_damping), float(dt))
    elif space_damping is not None:
        raise TypeError("provide either damping_factor or space_damping, not both")

    _validate_state(state)
    (gravity_x, gravity_y), damping_float, dt_float = _validate_step(gravity, damping_factor, dt)

    if int(state.body_type) == BODY_TYPE_STATIC:
        return state

    position_x, position_y = _pair(state.position, name="position")
    velocity_x, velocity_y = _pair(state.velocity, name="velocity")
    force_x, force_y = _pair(state.force, name="force")
    velocity_bias_x, velocity_bias_y = _pair(state.velocity_bias, name="velocity_bias")
    angular_velocity = float(state.angular_velocity)
    angular_velocity_bias = float(state.angular_velocity_bias)

    next_position = (
        position_x + (velocity_x + velocity_bias_x) * dt_float,
        position_y + (velocity_y + velocity_bias_y) * dt_float,
    )
    next_angle = float(state.angle) + (angular_velocity + angular_velocity_bias) * dt_float

    if int(state.body_type) == BODY_TYPE_KINEMATIC:
        return BodyState(
            body_type=int(state.body_type),
            velocity_mode=int(state.velocity_mode),
            position=next_position,
            velocity=(velocity_x, velocity_y),
            force=(force_x, force_y),
            angle=next_angle,
            angular_velocity=angular_velocity,
            torque=float(state.torque),
            mass=float(state.mass),
            moment=float(state.moment),
            center_of_gravity=(0.0, 0.0),
            velocity_bias=(0.0, 0.0),
            angular_velocity_bias=0.0,
        )

    if int(state.velocity_mode) == VELOCITY_MODE_NO_GRAVITY:
        gravity_x = 0.0
        gravity_y = 0.0
    mass = float(state.mass)
    moment = float(state.moment)
    next_velocity = (
        velocity_x * damping_float + (gravity_x + force_x / mass) * dt_float,
        velocity_y * damping_float + (gravity_y + force_y / mass) * dt_float,
    )
    next_angular_velocity = angular_velocity * damping_float + float(state.torque) / moment * dt_float

    return BodyState(
        body_type=int(state.body_type),
        velocity_mode=int(state.velocity_mode),
        position=next_position,
        velocity=next_velocity,
        force=(0.0, 0.0),
        angle=next_angle,
        angular_velocity=next_angular_velocity,
        torque=0.0,
        mass=mass,
        moment=moment,
        center_of_gravity=(0.0, 0.0),
        velocity_bias=(0.0, 0.0),
        angular_velocity_bias=0.0,
    )


def pack_state(state: BodyState) -> list[float]:
    _validate_state(state)
    position_x, position_y = _pair(state.position, name="position")
    velocity_x, velocity_y = _pair(state.velocity, name="velocity")
    force_x, force_y = _pair(state.force, name="force")
    bias_x, bias_y = _pair(state.velocity_bias, name="velocity_bias")
    return [
        float(int(state.body_type)),
        float(int(state.velocity_mode)),
        position_x,
        position_y,
        velocity_x,
        velocity_y,
        force_x,
        force_y,
        float(state.angle),
        float(state.angular_velocity),
        float(state.torque),
        float(state.mass),
        float(state.moment),
        float(state.center_of_gravity[0]),
        float(state.center_of_gravity[1]),
        bias_x,
        bias_y,
        float(state.angular_velocity_bias),
    ]


def unpack_state(values: Sequence[float]) -> BodyState:
    if len(values) != BODY_STATE_FIELD_COUNT:
        raise ValueError(f"packed state must have {BODY_STATE_FIELD_COUNT} fields")
    return BodyState(
        body_type=int(values[0]),
        velocity_mode=int(values[1]),
        position=(float(values[2]), float(values[3])),
        velocity=(float(values[4]), float(values[5])),
        force=(float(values[6]), float(values[7])),
        angle=float(values[8]),
        angular_velocity=float(values[9]),
        torque=float(values[10]),
        mass=float(values[11]),
        moment=float(values[12]),
        center_of_gravity=(float(values[13]), float(values[14])),
        velocity_bias=(float(values[15]), float(values[16])),
        angular_velocity_bias=float(values[17]),
    )


def _shared_library_suffix() -> str:
    return ".dylib" if platform.system() == "Darwin" else ".so"


def can_compile_cpu_reference() -> bool:
    compiler = os.environ.get("CC") or "cc"
    return shutil.which(compiler) is not None


def compile_cpu_reference(build_dir: Path) -> dict[str, Any]:
    compiler = os.environ.get("CC") or "cc"
    if shutil.which(compiler) is None:
        raise RuntimeError("no C compiler available for CPU reference")
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / f"libnavigation_physics_cpu{_shared_library_suffix()}"
    command = [
        compiler,
        "-O3",
        "-std=c99",
        "-ffp-contract=off",
        "-fPIC",
        str(NATIVE_DIR / "navigation_physics.c"),
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
        raise RuntimeError(f"CPU physics compilation failed:\n{completed.stdout}")
    return {"path": str(output), "command": command, "compiler_output": completed.stdout}


class CpuReference:
    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.lib = ctypes.CDLL(str(library_path))
        self._step_one = self.lib.navigation_physics_step_one_cpu
        self._step_one.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self._step_one.restype = ctypes.c_int
        self._step_batch = self.lib.navigation_physics_step_batch_cpu
        self._step_batch.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self._step_batch.restype = ctypes.c_int

    def step(self, state: BodyState, *, gravity: Sequence[float], damping_factor: float, dt: float) -> BodyState:
        return self.step_batch([state], gravity=gravity, damping_factor=damping_factor, dt=dt)[0]

    def step_batch(
        self,
        states: Sequence[BodyState],
        *,
        gravity: Sequence[float],
        damping_factor: float,
        dt: float,
    ) -> list[BodyState]:
        if len(states) <= 0:
            raise ValueError("states must be non-empty")
        (gravity_x, gravity_y), damping_float, dt_float = _validate_step(gravity, damping_factor, dt)
        packed_values: list[float] = []
        for state in states:
            packed_values.extend(pack_state(state))
        input_array = (ctypes.c_double * len(packed_values))(*packed_values)
        output_array = (ctypes.c_double * len(packed_values))()
        result = self._step_batch(
            input_array,
            output_array,
            ctypes.c_size_t(len(states)),
            ctypes.c_double(gravity_x),
            ctypes.c_double(gravity_y),
            ctypes.c_double(damping_float),
            ctypes.c_double(dt_float),
        )
        if result != OK:
            raise RuntimeError(f"navigation_physics_step_batch_cpu failed with error {result}")
        return [
            unpack_state(
                output_array[
                    lane * BODY_STATE_FIELD_COUNT : (lane + 1) * BODY_STATE_FIELD_COUNT
                ]
            )
            for lane in range(len(states))
        ]


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


def can_run_cuda() -> bool:
    return _nvcc() is not None and _torch_cuda_available()


def normalize_cuda_device(device: str = "cuda", *, current_index: int | None = None) -> str:
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
    output = build_dir / "libnavigation_physics_cuda.so"
    command = [
        nvcc,
        "-O3",
        "--fmad=false",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        str(NATIVE_DIR / "navigation_physics.cu"),
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
        raise RuntimeError(f"CUDA physics compilation failed:\n{completed.stdout}")
    return {"path": str(output), "command": command, "compiler_output": completed.stdout}


class CudaLibrary:
    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.lib = ctypes.CDLL(str(library_path))
        self._step_batch = self.lib.navigation_physics_step_batch_cuda
        self._step_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._step_batch.restype = ctypes.c_int

    def step_batch(self, torch: Any, states, *, gravity: Sequence[float], damping_factor: float, dt: float):
        (gravity_x, gravity_y), damping_float, dt_float = _validate_step(gravity, damping_factor, dt)
        if not states.is_cuda or states.dtype != torch.float64 or states.ndim != 2:
            raise ValueError("states must be a CUDA float64 tensor with shape [lanes, BODY_STATE_FIELD_COUNT]")
        if states.shape[1] != BODY_STATE_FIELD_COUNT:
            raise ValueError(f"states must have {BODY_STATE_FIELD_COUNT} fields per lane")
        if not states.is_contiguous():
            raise ValueError("states must be contiguous")
        lanes = int(states.shape[0])
        if lanes <= 0:
            raise ValueError("lanes must be positive")
        device = normalize_cuda_device(str(states.device), current_index=int(torch.cuda.current_device()))
        out = torch.empty_like(states)
        errors = torch.empty((lanes,), dtype=torch.int32, device=torch.device(device))
        with torch.cuda.device(device):
            stream = ctypes.c_void_p(int(torch.cuda.current_stream(device).cuda_stream))
            launch_error = self._step_batch(
                ctypes.c_void_p(states.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int(lanes),
                ctypes.c_double(gravity_x),
                ctypes.c_double(gravity_y),
                ctypes.c_double(damping_float),
                ctypes.c_double(dt_float),
                ctypes.c_void_p(errors.data_ptr()),
                stream,
            )
        if launch_error != 0:
            raise RuntimeError(f"navigation_physics_step_batch_cuda failed with cuda error {launch_error}")
        torch.cuda.synchronize(torch.device(device))
        host_errors = errors.cpu()
        if bool((host_errors != OK).any().item()):
            first = int((host_errors != OK).nonzero()[0].item())
            raise RuntimeError(f"navigation_physics CUDA lane {first} failed with error {int(host_errors[first])}")
        return out


def load_cuda_library(build_dir: Path) -> CudaLibrary:
    result = compile_cuda_library(build_dir)
    return CudaLibrary(Path(result["path"]))


def generate_fixture_states(lanes: int, *, seed: int) -> list[BodyState]:
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    rng = random.Random(seed)
    seeds = [
        BodyState(position=(0.0, 0.0), velocity=(1.0, 2.0), force=(3.0, 4.0), mass=1.0, moment=1e9),
        BodyState(
            position=(2.0, -2.0),
            velocity=(-1.0, 0.5),
            force=(0.0, 1.0),
            angle=0.4,
            angular_velocity=0.2,
            torque=3.0,
            mass=2.0,
            moment=5.0,
        ),
        BodyState(
            body_type=BODY_TYPE_KINEMATIC,
            position=(-1.0, 1.5),
            velocity=(2.0, -3.0),
            force=(10.0, 20.0),
            torque=99.0,
            mass=math.inf,
            moment=math.inf,
        ),
        BodyState(
            velocity_mode=VELOCITY_MODE_NO_GRAVITY,
            position=(1.25, -1.5),
            velocity=(0.25, -0.75),
            force=(-0.5, 1.25),
            angle=0.125,
            angular_velocity=-0.25,
            torque=0.5,
            mass=1.0,
            moment=1e9,
        ),
    ]
    states: list[BodyState] = []
    for lane in range(lanes):
        if lane < len(seeds):
            states.append(seeds[lane])
            continue
        mass = rng.uniform(0.5, 4.0)
        moment = rng.uniform(1.0, 1e9)
        states.append(
            BodyState(
                body_type=BODY_TYPE_DYNAMIC,
                velocity_mode=VELOCITY_MODE_NO_GRAVITY if rng.random() < 0.25 else VELOCITY_MODE_DEFAULT,
                position=(rng.uniform(-10.0, 10.0), rng.uniform(-8.0, 8.0)),
                velocity=(rng.uniform(-6.0, 6.0), rng.uniform(-6.0, 6.0)),
                force=(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)),
                angle=rng.uniform(-1.0, 1.0),
                angular_velocity=rng.uniform(-3.0, 3.0),
                torque=rng.uniform(-4.0, 4.0),
                mass=mass,
                moment=moment,
                velocity_bias=(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)),
                angular_velocity_bias=rng.uniform(-0.1, 0.1),
            )
        )
    return states


def _max_abs_error(actual: Sequence[Sequence[float]], expected: Sequence[BodyState]) -> float:
    max_error = 0.0
    for lane, expected_state in enumerate(expected):
        expected_values = pack_state(expected_state)
        for index, expected_value in enumerate(expected_values):
            actual_value = float(actual[lane][index])
            if math.isfinite(actual_value) and math.isfinite(expected_value):
                max_error = max(max_error, abs(actual_value - expected_value))
            elif actual_value != expected_value:
                return math.inf
    return max_error


def _mismatch_count(actual: Sequence[Sequence[float]], expected: Sequence[BodyState], *, fields: Sequence[int]) -> int:
    mismatches = 0
    for lane, expected_state in enumerate(expected):
        expected_values = pack_state(expected_state)
        for field in fields:
            if abs(float(actual[lane][field]) - expected_values[field]) > 1e-9:
                mismatches += 1
                break
    return mismatches


def run_cuda_parity_once(
    lanes: int,
    *,
    seed: int = 20260906,
    build_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    if not can_run_cuda():
        raise RuntimeError("CUDA torch and nvcc are required; refusing CPU fallback")
    import torch

    owned_temp = None
    if build_dir is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="ai2-nav-physics-cuda-")
        build_dir = Path(owned_temp.name)
    try:
        cuda = load_cuda_library(build_dir)
        states = generate_fixture_states(lanes, seed=seed)
        expected = [
            step_body_scalar(state, gravity=(0.0, 17.808), damping_factor=0.97, dt=1.0 / 30.0)
            for state in states
        ]
        torch_device = torch.device(normalize_cuda_device(device, current_index=int(torch.cuda.current_device())))
        states_gpu = torch.tensor([pack_state(state) for state in states], dtype=torch.float64, device=torch_device)
        actual_gpu = cuda.step_batch(
            torch,
            states_gpu.contiguous(),
            gravity=(0.0, 17.808),
            damping_factor=0.97,
            dt=1.0 / 30.0,
        )
        actual = actual_gpu.cpu().tolist()

        order = list(range(lanes))
        random.Random(seed + 1).shuffle(order)
        permuted_states = [states[index] for index in order]
        permuted_expected = [
            step_body_scalar(state, gravity=(0.0, 17.808), damping_factor=0.97, dt=1.0 / 30.0)
            for state in permuted_states
        ]
        permuted_gpu = torch.tensor(
            [pack_state(state) for state in permuted_states], dtype=torch.float64, device=torch_device
        )
        permuted_actual_gpu = cuda.step_batch(
            torch,
            permuted_gpu.contiguous(),
            gravity=(0.0, 17.808),
            damping_factor=0.97,
            dt=1.0 / 30.0,
        )
        permuted_actual = permuted_actual_gpu.cpu().tolist()
        return {
            "lanes": lanes,
            "max_abs_error": _max_abs_error(actual, expected),
            "position_mismatches": _mismatch_count(actual, expected, fields=(2, 3, 8)),
            "velocity_mismatches": _mismatch_count(actual, expected, fields=(4, 5, 9)),
            "permutation": {
                "max_abs_error": _max_abs_error(permuted_actual, permuted_expected),
                "position_mismatches": _mismatch_count(permuted_actual, permuted_expected, fields=(2, 3, 8)),
                "velocity_mismatches": _mismatch_count(permuted_actual, permuted_expected, fields=(4, 5, 9)),
            },
        }
    finally:
        if owned_temp is not None:
            owned_temp.cleanup()
