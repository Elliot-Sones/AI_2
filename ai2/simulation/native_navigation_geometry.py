#!/usr/bin/env python3
"""One-call native geometry helper for frozen ``Player.is_on_floor``.

This is a narrow hotpath only: it calls the real Chipmunk
``cpShapeCacheBB``/``cpBodyGetPosition`` functions borrowed from the active
Pymunk module, preserving cache-bb side effects and the frozen Python
short-circuit order.  It does not replace physics or infer support contacts.
"""

from __future__ import annotations

from ai2.paths import REPO_ROOT as PROJECT_ROOT

from contextlib import AbstractContextManager
import hashlib
import os
import platform
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

ROOT = PROJECT_ROOT
NATIVE_DIR = ROOT / "native"
FROZEN_ENV_SHA256 = "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"
EXPECTED_PYMUNK_VERSION = "6.2.1"
EXPECTED_CHIPMUNK_VERSION = "7.0.3-0593976ef47fcb3957166bd342f6b2bafe4d0e44"
_C_HELPER_SINGLETON = None


GEOMETRY_CDEF = """
typedef struct { double x; double y; } AI2CpVect;
typedef struct { double l; double b; double r; double t; } AI2CpBB;
typedef AI2CpBB (*AI2CpShapeCacheBBFn)(void *shape);
typedef AI2CpVect (*AI2CpBodyGetPositionFn)(void *body);
typedef struct {
    AI2CpShapeCacheBBFn cp_shape_cache_bb;
    AI2CpBodyGetPositionFn cp_body_get_position;
} AI2NavigationGeometryFns;
typedef struct {
    void *player_shape;
    void *player_body;
    void *ground1_shape;
    void *ground1_body;
    void *ground2_shape;
    void *ground2_body;
    void *platform_shape;
    void *platform_body;
} AI2NavigationFloorPointers;
int ai2_navigation_is_on_floor(
    const AI2NavigationGeometryFns *fns,
    const AI2NavigationFloorPointers *ptrs,
    uint8_t *out_on_floor);
"""


def verify_source_hash(path: str | os.PathLike[str], *, expected_sha256: str = FROZEN_ENV_SHA256) -> str:
    source_path = Path(path)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"frozen environment source hash mismatch for {source_path}: {digest}")
    return digest


def verify_pymunk_version(
    *,
    expected_pymunk: str | None = None,
    expected_chipmunk: str | None = None,
) -> tuple[str, str]:
    try:
        import pymunk
    except Exception as exc:  # pragma: no cover - hosts without pymunk skip integration tests.
        raise RuntimeError("pymunk unavailable") from exc
    pymunk_version = str(getattr(pymunk, "version", "")).strip()
    chipmunk_version = str(getattr(pymunk, "chipmunk_version", "")).strip()
    expected_pymunk = EXPECTED_PYMUNK_VERSION if expected_pymunk is None else expected_pymunk
    expected_chipmunk = EXPECTED_CHIPMUNK_VERSION if expected_chipmunk is None else expected_chipmunk
    if pymunk_version != expected_pymunk:
        raise RuntimeError(f"Pymunk version mismatch: expected {expected_pymunk}, got {pymunk_version}")
    if chipmunk_version != expected_chipmunk:
        raise RuntimeError(f"Chipmunk version mismatch: expected {expected_chipmunk}, got {chipmunk_version}")
    return pymunk_version, chipmunk_version


def _compile_shared() -> Path:
    build = Path(tempfile.mkdtemp(prefix="ai2-navigation-geometry-"))
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    out = build / f"navigation_geometry{suffix}"
    cmd = [
        os.environ.get("CC", "cc"),
        "-std=c99",
        "-O2",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fPIC",
        "-shared",
        str(NATIVE_DIR / "navigation_geometry.c"),
        "-o",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    return out


def _load_pymunk_cffi():
    verify_pymunk_version()
    try:
        from pymunk import _chipmunk_cffi
    except Exception as exc:  # pragma: no cover - hosts without pymunk skip integration tests.
        raise RuntimeError("pymunk CFFI bindings unavailable") from exc
    return _chipmunk_cffi.ffi, _chipmunk_cffi.lib


def _require_handle(obj, attr: str, label: str, abi):
    if obj is None or not hasattr(obj, attr):
        raise ValueError(f"required shape/body pointer missing for {label}")
    try:
        handle = abi.cast("void *", getattr(obj, attr))
    except Exception as exc:
        raise ValueError(f"required shape/body pointer missing for {label}") from exc
    if handle == abi.NULL:
        raise ValueError(f"required shape/body pointer missing for {label}")
    return handle


class NavigationGeometryHelper:
    def __init__(self, shared_library: Path):
        try:
            import cffi
        except ImportError as exc:  # pragma: no cover - pymunk hosts provide CFFI.
            raise RuntimeError("cffi unavailable") from exc
        self._abi = cffi.FFI()
        self._abi.cdef(GEOMETRY_CDEF)
        self.lib = self._abi.dlopen(str(shared_library))
        pymunk_ffi, chipmunk = _load_pymunk_cffi()
        self._fns = self._abi.new("AI2NavigationGeometryFns *")
        self._fns.cp_shape_cache_bb = self._abi.cast(
            "AI2CpShapeCacheBBFn",
            pymunk_ffi.addressof(chipmunk, "cpShapeCacheBB"),
        )
        self._fns.cp_body_get_position = self._abi.cast(
            "AI2CpBodyGetPositionFn",
            pymunk_ffi.addressof(chipmunk, "cpBodyGetPosition"),
        )

    def prepare_is_on_floor(
        self,
        player_shape,
        player_body,
        ground1_shape,
        ground1_body,
        ground2_shape,
        ground2_body,
        platform_shape,
        platform_body,
    ) -> "NavigationGeometryBinding":
        ptrs = self._abi.new("AI2NavigationFloorPointers *")
        ptrs.player_shape = _require_handle(player_shape, "_shape", "player shape", self._abi)
        ptrs.player_body = _require_handle(player_body, "_body", "player body", self._abi)
        ptrs.ground1_shape = _require_handle(ground1_shape, "_shape", "ground1 shape", self._abi)
        ptrs.ground1_body = _require_handle(ground1_body, "_body", "ground1 body", self._abi)
        ptrs.ground2_shape = _require_handle(ground2_shape, "_shape", "ground2 shape", self._abi)
        ptrs.ground2_body = _require_handle(ground2_body, "_body", "ground2 body", self._abi)
        ptrs.platform_shape = _require_handle(platform_shape, "_shape", "platform shape", self._abi)
        ptrs.platform_body = _require_handle(platform_body, "_body", "platform body", self._abi)
        refs = (
            player_shape,
            player_body,
            ground1_shape,
            ground1_body,
            ground2_shape,
            ground2_body,
            platform_shape,
            platform_body,
        )
        return NavigationGeometryBinding(self, ptrs, refs)

    def is_on_floor(
        self,
        player_shape,
        player_body,
        ground1_shape,
        ground1_body,
        ground2_shape,
        ground2_body,
        platform_shape,
        platform_body,
    ) -> bool:
        return self.prepare_is_on_floor(
            player_shape,
            player_body,
            ground1_shape,
            ground1_body,
            ground2_shape,
            ground2_body,
            platform_shape,
            platform_body,
        ).is_on_floor()


class NavigationGeometryBinding:
    """Prepared CFFI binding for one current player/ground/platform identity."""

    def __init__(self, helper: NavigationGeometryHelper, ptrs, strong_refs: tuple[object, ...]):
        self.helper = helper
        self._ptrs = ptrs
        self._strong_refs = strong_refs
        self._out = helper._abi.new("uint8_t *")

    def is_on_floor(self) -> bool:
        status = self.helper.lib.ai2_navigation_is_on_floor(self.helper._fns, self._ptrs, self._out)
        if status == 3:
            raise ValueError("required shape/body pointer missing")
        if status != 0:
            raise ValueError(f"native geometry helper failed status={status}")
        return bool(self._out[0])


def load_c_helper() -> NavigationGeometryHelper:
    global _C_HELPER_SINGLETON
    if _C_HELPER_SINGLETON is None:
        _C_HELPER_SINGLETON = NavigationGeometryHelper(_compile_shared())
    return _C_HELPER_SINGLETON


def _objects_for_player(player):
    objects = getattr(getattr(player, "env", None), "objects", None)
    if not isinstance(objects, dict):
        raise ValueError("player.env.objects must provide ground1/ground2/platform1")
    try:
        return objects["ground1"], objects["ground2"], objects["platform1"]
    except KeyError as exc:
        raise ValueError("player.env.objects must provide ground1/ground2/platform1") from exc


def _space_is_locked(player) -> bool:
    body = getattr(player, "body", None)
    space = getattr(body, "space", None)
    return bool(getattr(space, "_locked", False))


def _current_floor_refs(player) -> tuple[object, ...]:
    ground1, ground2, platform1 = _objects_for_player(player)
    return (
        getattr(player, "shape", None),
        getattr(player, "body", None),
        getattr(ground1, "shape", None),
        getattr(ground1, "body", None),
        getattr(ground2, "shape", None),
        getattr(ground2, "body", None),
        getattr(platform1, "shape", None),
        getattr(platform1, "body", None),
    )


class _InstalledFloorHook:
    def __init__(self, player, helper: NavigationGeometryHelper):
        self.player = player
        self.helper = helper
        self._refs: tuple[object, ...] | None = None
        self._binding: NavigationGeometryBinding | None = None

    def _refresh_if_needed(self) -> None:
        refs = _current_floor_refs(self.player)
        if refs != self._refs:
            self._binding = self.helper.prepare_is_on_floor(*refs)
            self._refs = refs

    def __call__(self) -> bool:
        self._refresh_if_needed()
        if self._binding is None:  # pragma: no cover - defensive, impossible after refresh.
            raise RuntimeError("native geometry binding was not prepared")
        return self._binding.is_on_floor()


class PlayerFloorPatch(AbstractContextManager):
    def __init__(
        self,
        players: Iterable[object],
        *,
        helper: NavigationGeometryHelper | None = None,
        source_path: str | os.PathLike[str] | None = None,
        expected_sha256: str = FROZEN_ENV_SHA256,
    ):
        self.players = list(players)
        self.helper = helper or load_c_helper()
        self.source_path = Path(source_path) if source_path is not None else None
        self.expected_sha256 = expected_sha256
        self._originals: dict[int, tuple[object, object, object]] = {}
        self._attached = False

    def attach(self):
        if self._attached:
            return self
        if self.source_path is not None:
            verify_source_hash(self.source_path, expected_sha256=self.expected_sha256)
        for player in self.players:
            if _space_is_locked(player):
                raise RuntimeError("refusing to patch is_on_floor while Pymunk space is locked")
        originals: dict[int, tuple[object, object, object]] = {}
        for player in self.players:
            original = getattr(player, "is_on_floor")
            patched = _InstalledFloorHook(player, self.helper)
            patched._refresh_if_needed()
            originals[id(player)] = (player, original, patched)
            setattr(player, "is_on_floor", patched)
        self._originals = originals
        self._attached = True
        return self

    def restore(self) -> None:
        if not self._attached:
            return
        for player, original, patched in self._originals.values():
            if getattr(player, "is_on_floor", None) is patched:
                setattr(player, "is_on_floor", original)
        self._originals = {}
        self._attached = False

    def __enter__(self):
        return self.attach()

    def __exit__(self, exc_type, exc, tb):
        self.restore()
        return False


def patch_players_is_on_floor(players: Iterable[object], **kwargs) -> PlayerFloorPatch:
    return PlayerFloorPatch(players, **kwargs)
