#!/usr/bin/env python3
"""CPython hotpath wrapper for frozen live ``Player.get_obs``.

The extension reads the active game objects and calls the currently installed
``player.is_on_floor`` method.  It preserves the original Pymunk ``Space.step``
and does not own physics or reward semantics.
"""

from __future__ import annotations

from ai2.paths import REPO_ROOT as PROJECT_ROOT

from contextlib import AbstractContextManager
import hashlib
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sysconfig
import tempfile
from typing import Iterable

ROOT = PROJECT_ROOT
NATIVE_DIR = ROOT / "native"
FROZEN_ENV_SHA256 = "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"
EXPECTED_PYMUNK_VERSION = "6.2.1"
EXPECTED_CHIPMUNK_VERSION = "7.0.3-0593976ef47fcb3957166bd342f6b2bafe4d0e44"
_EXTENSION_SINGLETON = None
_CHIPMUNK_BODY_FNS = None


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
    except Exception as exc:  # pragma: no cover - dependency-light hosts skip integration tests.
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


def _compile_extension() -> Path:
    build = Path(tempfile.mkdtemp(prefix="ai2-live-observation-"))
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = build / f"_ai2_native_live_observation{ext_suffix}"
    ldshared = sysconfig.get_config_var("LDSHARED")
    if not ldshared:
        raise RuntimeError("Python sysconfig LDSHARED unavailable")
    includes = sysconfig.get_paths().get("include")
    cmd = shlex.split(ldshared)
    if includes:
        cmd.append(f"-I{includes}")
    cmd.extend([
        "-std=c99",
        "-O2",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(NATIVE_DIR / "live_observation.c"),
        "-o",
        str(out),
    ])
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    return out


def _load_module():
    global _EXTENSION_SINGLETON
    if _EXTENSION_SINGLETON is not None:
        return _EXTENSION_SINGLETON
    path = _compile_extension()
    spec = importlib.util.spec_from_file_location("_ai2_native_live_observation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import compiled live observation extension at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EXTENSION_SINGLETON = module
    return module


class LiveObservationHelper:
    def __init__(self, module, *, facing_right, dropped_spawner_type: type):
        if not isinstance(dropped_spawner_type, type):
            raise TypeError("dropped_spawner_type must be a type")
        self.module = module
        self.facing_right = facing_right
        self.dropped_spawner_type = dropped_spawner_type

    def get_obs(self, player) -> list[float]:
        return self.prepare_get_obs(player)()

    def prepare_get_obs(self, player):
        player_body, platform_body = _current_bodies(player)
        position_addr, velocity_addr = _chipmunk_body_function_addresses()
        return self.module.prepare(
            player,
            self.dropped_spawner_type,
            self.facing_right,
            player_body,
            platform_body,
            _body_address(player_body),
            _body_address(platform_body),
            position_addr,
            velocity_addr,
        )


def _load_pymunk_cffi():
    verify_pymunk_version()
    try:
        from pymunk import _chipmunk_cffi
    except Exception as exc:  # pragma: no cover - dependency-light hosts skip integration tests.
        raise RuntimeError("pymunk CFFI bindings unavailable") from exc
    return _chipmunk_cffi.ffi, _chipmunk_cffi.lib


def _chipmunk_body_function_addresses() -> tuple[int, int]:
    global _CHIPMUNK_BODY_FNS
    if _CHIPMUNK_BODY_FNS is None:
        ffi, lib = _load_pymunk_cffi()
        _CHIPMUNK_BODY_FNS = (
            int(ffi.cast("uintptr_t", ffi.addressof(lib, "cpBodyGetPosition"))),
            int(ffi.cast("uintptr_t", ffi.addressof(lib, "cpBodyGetVelocity"))),
        )
    return _CHIPMUNK_BODY_FNS


def _body_address(body) -> int:
    if body is None or not hasattr(body, "_body"):
        raise ValueError("body must expose a Pymunk _body pointer")
    ffi, _lib = _load_pymunk_cffi()
    value = int(ffi.cast("uintptr_t", body._body))
    if value <= 0:
        raise ValueError("body pointer is null")
    return value


def _current_bodies(player):
    body = getattr(player, "body", None)
    platform = getattr(player, "PLATFORM", None)
    platform_body = getattr(platform, "body", None)
    if body is None:
        raise ValueError("player body is required")
    if platform_body is None:
        raise ValueError("platform body is required")
    return body, platform_body


def load_c_extension(*, facing_right, dropped_spawner_type: type) -> LiveObservationHelper:
    return LiveObservationHelper(_load_module(), facing_right=facing_right, dropped_spawner_type=dropped_spawner_type)


class _InstalledGetObsHook:
    def __init__(self, player, helper: LiveObservationHelper):
        self.player = player
        self.helper = helper
        self._refs = None
        self._prepared = None

    def _refresh_if_needed(self) -> None:
        refs = _current_bodies(self.player)
        if refs != self._refs:
            self._prepared = self.helper.prepare_get_obs(self.player)
            self._refs = refs

    def __call__(self) -> list[float]:
        self._refresh_if_needed()
        return self._prepared()


class PlayerGetObsPatch(AbstractContextManager):
    def __init__(
        self,
        players: Iterable[object],
        *,
        helper: LiveObservationHelper,
        source_path: str | os.PathLike[str] | None = None,
        expected_sha256: str = FROZEN_ENV_SHA256,
    ):
        self.players = list(players)
        self.helper = helper
        self.source_path = Path(source_path) if source_path is not None else None
        self.expected_sha256 = expected_sha256
        self._originals: dict[int, tuple[object, object, object]] = {}
        self._attached = False

    def attach(self):
        if self._attached:
            return self
        if self.source_path is not None:
            verify_source_hash(self.source_path, expected_sha256=self.expected_sha256)
        originals: dict[int, tuple[object, object, object]] = {}
        try:
            for player in self.players:
                original = getattr(player, "get_obs")
                patched = _InstalledGetObsHook(player, self.helper)
                patched._refresh_if_needed()
                originals[id(player)] = (player, original, patched)
                setattr(player, "get_obs", patched)
        except Exception:
            for player, original, patched in originals.values():
                if getattr(player, "get_obs", None) is patched:
                    setattr(player, "get_obs", original)
            self._originals = {}
            self._attached = False
            raise
        self._originals = originals
        self._attached = True
        return self

    def restore(self) -> None:
        if not self._attached:
            return
        for player, original, patched in self._originals.values():
            if getattr(player, "get_obs", None) is patched:
                setattr(player, "get_obs", original)
        self._originals = {}
        self._attached = False

    def __enter__(self):
        return self.attach()

    def __exit__(self, exc_type, exc, tb):
        self.restore()
        return False


def patch_players_get_obs(players: Iterable[object], **kwargs) -> PlayerGetObsPatch:
    return PlayerGetObsPatch(players, **kwargs)
