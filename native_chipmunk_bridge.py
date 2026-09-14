#!/usr/bin/env python3
"""ctypes bridge to the exact Chipmunk solver already loaded by Pymunk.

This is a CPU reference/orchestration bridge, not a speed claim. It preserves
the installed Chipmunk solver by calling function pointers from Pymunk
6.2.1's ``_chipmunk.abi3.so`` with raw Pymunk-owned ``cpSpace*`` and
``cpBody*`` pointers.

The bridge is loaded with ``ctypes.PyDLL`` so the GIL remains held during
scalar stepping. That is deliberate: Pymunk collision callbacks can call back
into Python. Parallel/off-GIL stepping is unsupported until callbacks,
deferred queues and post-step processing are fully native.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
NATIVE_DIR = ROOT / "native"
BODY_STATE_OUTPUT_FIELDS = 11
EXPECTED_PYMUNK_VERSION = "6.2.1"
EXPECTED_CHIPMUNK_VERSION = "7.0.3-0593976ef47fcb3957166bd342f6b2bafe4d0e44 "

OK = 0


def _shared_library_suffix() -> str:
    return ".dylib" if platform.system() == "Darwin" else ".so"


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _pymunk_modules():
    import pymunk
    from pymunk import _chipmunk_cffi

    if getattr(pymunk, "version", None) != EXPECTED_PYMUNK_VERSION:
        raise RuntimeError(f"expected Pymunk {EXPECTED_PYMUNK_VERSION}, got {getattr(pymunk, 'version', None)!r}")
    if getattr(pymunk, "chipmunk_version", None) != EXPECTED_CHIPMUNK_VERSION:
        raise RuntimeError(
            f"expected bundled Chipmunk {EXPECTED_CHIPMUNK_VERSION!r}, "
            f"got {getattr(pymunk, 'chipmunk_version', None)!r}"
        )
    return pymunk, _chipmunk_cffi


def installed_chipmunk_info() -> dict[str, Any]:
    pymunk, cffi = _pymunk_modules()
    extension_path = Path(cffi.__file__).with_name("_chipmunk.abi3.so")
    version_file = Path(pymunk.__file__).with_name("_version.py")
    return {
        "pymunk_version": pymunk.version,
        "chipmunk_version": pymunk.chipmunk_version,
        "extension_path": str(extension_path),
        "extension_sha256": file_sha256(extension_path),
        "pymunk_version_file": str(version_file),
        "pymunk_version_file_sha256": file_sha256(version_file),
        "bridge_scope": "CPU exact-solver bridge using installed Pymunk-owned Chipmunk objects",
        "gil_contract": "ctypes.PyDLL retains the GIL; no OpenMP/off-GIL stepping while Python callbacks exist",
    }


def _function_address(name: str) -> int:
    _, cffi = _pymunk_modules()
    return int(cffi.ffi.cast("uintptr_t", cffi.ffi.addressof(cffi.lib, name)))


def _space_pointer(space: Any) -> int:
    _, cffi = _pymunk_modules()
    return int(cffi.ffi.cast("uintptr_t", space._space))


def _body_pointer(body: Any) -> int:
    _, cffi = _pymunk_modules()
    return int(cffi.ffi.cast("uintptr_t", body._body))


def can_compile_bridge() -> bool:
    compiler = os.environ.get("CC") or "cc"
    return shutil.which(compiler) is not None


def compile_bridge(build_dir: Path) -> dict[str, Any]:
    compiler = os.environ.get("CC") or "cc"
    if shutil.which(compiler) is None:
        raise RuntimeError("no C compiler available")
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / f"libchipmunk_bridge{_shared_library_suffix()}"
    command = [
        compiler,
        "-O3",
        "-std=c99",
        "-ffp-contract=off",
        "-fPIC",
        str(NATIVE_DIR / "chipmunk_bridge.c"),
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
        raise RuntimeError(f"Chipmunk bridge compilation failed:\n{completed.stdout}")
    return {"path": str(output), "command": command, "compiler_output": completed.stdout}


class ChipmunkBridgeLibrary:
    def __init__(self, library_path: Path):
        self.library_path = library_path
        # PyDLL intentionally holds the GIL while bridge C invokes Chipmunk.
        self.lib = ctypes.PyDLL(str(library_path))
        self._step_spaces = self.lib.chipmunk_bridge_step_spaces
        self._step_spaces.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_double,
        ]
        self._step_spaces.restype = ctypes.c_int
        self._get_body = self.lib.chipmunk_bridge_get_body_state
        self._get_body.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._get_body.restype = ctypes.c_int
        self._set_body = self.lib.chipmunk_bridge_set_body_state
        self._set_body.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self._set_body.restype = ctypes.c_int


def load_bridge(build_dir: Path | None = None) -> tuple[ChipmunkBridgeLibrary, tempfile.TemporaryDirectory[str] | None]:
    owned_temp = None
    if build_dir is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="ai2-chipmunk-bridge-")
        build_dir = Path(owned_temp.name)
    compiled = compile_bridge(build_dir)
    return ChipmunkBridgeLibrary(Path(compiled["path"])), owned_temp


@dataclass
class BridgeWorld:
    space: Any
    bodies: Sequence[Any]
    keepalive: Any | None = None


class BridgeBatch:
    def __init__(self, worlds: Sequence[BridgeWorld], library: ChipmunkBridgeLibrary, tempdir):
        if len(worlds) <= 0:
            raise ValueError("worlds must be non-empty")
        self.worlds = [
            BridgeWorld(space=world.space, bodies=tuple(world.bodies), keepalive=world.keepalive)
            for world in worlds
        ]
        self.library = library
        self._tempdir = tempdir
        self.space_pointers = [_space_pointer(world.space) for world in self.worlds]
        self.body_pointers = [[_body_pointer(body) for body in world.bodies] for world in self.worlds]
        self._addresses = {
            name: _function_address(name)
            for name in (
                "cpSpaceStep",
                "cpBodyGetPosition",
                "cpBodyGetVelocity",
                "cpBodyGetForce",
                "cpBodyGetAngle",
                "cpBodyGetAngularVelocity",
                "cpBodyGetTorque",
                "cpBodyGetMass",
                "cpBodyGetMoment",
                "cpBodySetPosition",
                "cpBodySetVelocity",
                "cpBodySetAngle",
                "cpBodySetAngularVelocity",
            )
        }

    @classmethod
    def from_worlds(cls, worlds: Sequence[BridgeWorld], build_dir: Path | None = None) -> "BridgeBatch":
        library, tempdir = load_bridge(build_dir)
        return cls(worlds, library, tempdir)

    def _check_world_index(self, world_index: int) -> None:
        if world_index < 0 or world_index >= len(self.worlds):
            raise IndexError(f"world_index {world_index} out of range")

    def _check_body_index(self, world_index: int, body_index: int) -> None:
        self._check_world_index(world_index)
        if body_index < 0 or body_index >= len(self.body_pointers[world_index]):
            raise IndexError(f"body_index {body_index} out of range")

    def _validate_space_identity(self, world_index: int) -> None:
        try:
            current = _space_pointer(self.worlds[world_index].space)
        except Exception as exc:
            raise RuntimeError(f"world {world_index} space pointer changed or became invalid") from exc
        if current != self.space_pointers[world_index]:
            raise RuntimeError(f"world {world_index} space pointer changed")

    def _validate_body_identity(self, world_index: int, body_index: int) -> None:
        self._check_body_index(world_index, body_index)
        try:
            current = _body_pointer(self.worlds[world_index].bodies[body_index])
        except Exception as exc:
            raise RuntimeError(f"world {world_index} body pointer changed or became invalid at index {body_index}") from exc
        if current != self.body_pointers[world_index][body_index]:
            raise RuntimeError(f"world {world_index} body pointer changed at index {body_index}")

    def _validate_all_identities(self) -> None:
        for world_index, world in enumerate(self.worlds):
            self._validate_space_identity(world_index)
            for body_index in range(len(world.bodies)):
                self._validate_body_identity(world_index, body_index)

    def _reject_unsafe_pymunk_queues(self) -> None:
        for lane, world in enumerate(self.worlds):
            if getattr(world.space, "_locked", False):
                raise RuntimeError(f"world {lane} is already locked by Pymunk; bridge cannot safely re-enter cpSpaceStep")
            if getattr(world.space, "_handlers", None):
                raise RuntimeError(
                    f"world {lane} has collision handlers; bridge stepping would bypass Pymunk "
                    "callback/queue semantics"
                )
            if getattr(world.space, "_add_later", None):
                raise RuntimeError(f"world {lane} has deferred add queue; bridge would bypass Pymunk queue drain")
            if getattr(world.space, "_remove_later", None):
                raise RuntimeError(f"world {lane} has deferred remove queue; bridge would bypass Pymunk queue drain")
            if getattr(world.space, "_post_step_callbacks", None):
                raise RuntimeError(f"world {lane} has post-step callbacks; bridge cannot drain Pymunk callback queue")

    def step_all(self, dt: float) -> None:
        if not math.isfinite(float(dt)) or float(dt) < 0.0:
            raise ValueError("dt must be finite and non-negative")
        self._validate_all_identities()
        self._reject_unsafe_pymunk_queues()
        spaces = (ctypes.c_size_t * len(self.space_pointers))(*self.space_pointers)
        error = self.library._step_spaces(
            ctypes.c_size_t(self._addresses["cpSpaceStep"]),
            spaces,
            ctypes.c_size_t(len(self.space_pointers)),
            ctypes.c_double(float(dt)),
        )
        if error != OK:
            raise RuntimeError(f"chipmunk_bridge_step_spaces failed with error {error}")

    def read_body(self, world_index: int, body_index: int) -> dict[str, Any]:
        self._validate_body_identity(world_index, body_index)
        out = (ctypes.c_double * BODY_STATE_OUTPUT_FIELDS)()
        body_ptr = self.body_pointers[world_index][body_index]
        error = self.library._get_body(
            ctypes.c_size_t(self._addresses["cpBodyGetPosition"]),
            ctypes.c_size_t(self._addresses["cpBodyGetVelocity"]),
            ctypes.c_size_t(self._addresses["cpBodyGetForce"]),
            ctypes.c_size_t(self._addresses["cpBodyGetAngle"]),
            ctypes.c_size_t(self._addresses["cpBodyGetAngularVelocity"]),
            ctypes.c_size_t(self._addresses["cpBodyGetTorque"]),
            ctypes.c_size_t(self._addresses["cpBodyGetMass"]),
            ctypes.c_size_t(self._addresses["cpBodyGetMoment"]),
            ctypes.c_size_t(body_ptr),
            out,
        )
        if error != OK:
            raise RuntimeError(f"chipmunk_bridge_get_body_state failed with error {error}")
        return {
            "position": (float(out[0]), float(out[1])),
            "velocity": (float(out[2]), float(out[3])),
            "force": (float(out[4]), float(out[5])),
            "angle": float(out[6]),
            "angular_velocity": float(out[7]),
            "torque": float(out[8]),
            "mass": float(out[9]),
            "moment": float(out[10]),
        }

    def set_body_state(
        self,
        world_index: int,
        body_index: int,
        *,
        position: Sequence[float],
        velocity: Sequence[float],
        angle: float,
        angular_velocity: float,
    ) -> None:
        self._validate_body_identity(world_index, body_index)
        position_x, position_y = _pair(position, "position")
        velocity_x, velocity_y = _pair(velocity, "velocity")
        body_ptr = self.body_pointers[world_index][body_index]
        error = self.library._set_body(
            ctypes.c_size_t(self._addresses["cpBodySetPosition"]),
            ctypes.c_size_t(self._addresses["cpBodySetVelocity"]),
            ctypes.c_size_t(self._addresses["cpBodySetAngle"]),
            ctypes.c_size_t(self._addresses["cpBodySetAngularVelocity"]),
            ctypes.c_size_t(body_ptr),
            ctypes.c_double(position_x),
            ctypes.c_double(position_y),
            ctypes.c_double(velocity_x),
            ctypes.c_double(velocity_y),
            ctypes.c_double(float(angle)),
            ctypes.c_double(float(angular_velocity)),
        )
        if error != OK:
            raise RuntimeError(f"chipmunk_bridge_set_body_state failed with error {error}")


def _pair(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    x = float(value[0])
    y = float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{name} must be finite")
    return x, y
