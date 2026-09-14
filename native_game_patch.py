#!/usr/bin/env python3
"""Opt-in native hotpath patching for the frozen game.

This module deliberately does not patch ``pymunk.Space.step``. Exact physics,
callbacks, deferred queues, exceptions and contact state remain owned by the
original Pymunk/Chipmunk runtime. The only installed hook is a reversible
``Player.is_on_floor`` class method wrapper that delegates to the already
validated native geometry helper. Optionally, ``PlayerInputHandler.update`` and
per-player ``Player.get_obs`` can be patched with separately validated native
live helpers. By default native helper failures raise with context; explicit
``strict=False`` fallback mode calls the original method and records the
fallback count.

No module-level side effects are performed on import. Grouped workers can opt
in by importing this module and calling ``install_native_game_patch`` with an
explicit frozen source path inside each worker process.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Any


FROZEN_ENV_SHA256 = "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"
ENABLE_ENV_VAR = "AI2_ENABLE_NATIVE_GAME_PATCH"
ENABLE_INPUT_ENV_VAR = "AI2_NATIVE_INPUT"
ENABLE_OBSERVATION_ENV_VAR = "AI2_NATIVE_OBSERVATION"


@dataclass
class _ClassPatchState:
    player_class: type
    original: Any
    hook: Any
    helper: Any
    strict: bool
    refcount: int = 0
    native_failures: int = 0
    last_native_error: str | None = None


_CLASS_PATCH_REGISTRY: dict[type, _ClassPatchState] = {}


@dataclass
class _InputPatchState:
    input_class: type
    original: Any
    hook: Any
    helper: Any
    strict: bool
    refcount: int = 0
    native_failures: int = 0
    last_native_error: str | None = None


_INPUT_PATCH_REGISTRY: dict[type, _InputPatchState] = {}


def file_sha256(path: str | os.PathLike[str]) -> str:
    source_path = Path(path)
    hasher = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_source_hash(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str = FROZEN_ENV_SHA256,
) -> str:
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError(f"source hash mismatch for {Path(path)}: {digest}")
    return digest


def _verify_player_class_source(
    player_class: type,
    source_path: str | os.PathLike[str] | None,
    *,
    expected_sha256: str,
    allow_synthetic_class: bool,
) -> str | None:
    if source_path is None:
        return None
    source = Path(source_path).resolve()
    digest = verify_source_hash(source, expected_sha256=expected_sha256)
    if not allow_synthetic_class:
        module = sys.modules.get(getattr(player_class, "__module__", ""))
        actual_file = getattr(module, "__file__", None)
        if actual_file is None:
            raise ValueError("Player class source mismatch: class module has no __file__")
        actual = Path(actual_file).resolve()
        if actual != source:
            raise ValueError(f"Player class source mismatch: {actual} != {source}")
    return digest


def _load_default_geometry_helper():
    import native_navigation_geometry

    return native_navigation_geometry.load_c_helper()


def _load_default_input_helper():
    import native_live_input

    return native_live_input.load_live_input()


def _load_observation_patch_factory():
    import native_live_observation

    return native_live_observation.patch_players_get_obs


def _load_default_observation_helper(player_class: type):
    import native_live_observation

    module = sys.modules.get(getattr(player_class, "__module__", ""))
    if module is None:
        raise ValueError("observation_helper is required when Player module is unavailable")
    facing = getattr(getattr(module, "Facing", None), "RIGHT", None)
    dropped_spawner_type = getattr(module, "DroppedWeaponSpawner", None)
    if facing is None or dropped_spawner_type is None:
        raise ValueError(
            "observation_helper is required unless Player module provides Facing.RIGHT and DroppedWeaponSpawner"
        )
    return native_live_observation.load_c_extension(
        facing_right=facing,
        dropped_spawner_type=dropped_spawner_type,
    )


def _objects_for_player(player: Any) -> tuple[Any, Any, Any]:
    objects = getattr(getattr(player, "env", None), "objects", None)
    if not isinstance(objects, dict):
        raise ValueError("player.env.objects must provide ground1/ground2/platform1")
    try:
        return objects["ground1"], objects["ground2"], objects["platform1"]
    except KeyError as exc:
        raise ValueError("player.env.objects must provide ground1/ground2/platform1") from exc


def _floor_refs(player: Any) -> tuple[Any, ...]:
    ground1, ground2, platform1 = _objects_for_player(player)
    return (
        player.shape,
        player.body,
        ground1.shape,
        ground1.body,
        ground2.shape,
        ground2.body,
        platform1.shape,
        platform1.body,
    )


def _same_identity_tuple(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


def _native_is_on_floor(player: Any, helper: Any) -> bool:
    refs = _floor_refs(player)
    cached = getattr(player, "_ai2_floor_binding", None)
    if cached is not None:
        cached_refs, binding = cached
        if _same_identity_tuple(cached_refs, refs):
            return bool(binding.is_on_floor())

    if hasattr(helper, "prepare_is_on_floor"):
        binding = helper.prepare_is_on_floor(*refs)
        setattr(player, "_ai2_floor_binding", (refs, binding))
        return bool(binding.is_on_floor())

    return bool(helper.is_on_floor(*refs))


def describe_native_game_patch() -> dict[str, Any]:
    """Return scope/provenance for the opt-in patch without mutating gameplay."""

    try:
        import native_chipmunk_bridge

        chipmunk_bridge = native_chipmunk_bridge.installed_chipmunk_info()
    except Exception as exc:  # pragma: no cover - diagnostic path for hosts without Pymunk.
        chipmunk_bridge = {
            "available": False,
            "error": repr(exc),
            "bridge_scope": "CPU exact-solver bridge unavailable on this host",
        }
    else:
        chipmunk_bridge = {"available": True, **chipmunk_bridge}

    return {
        "patched_methods": [
            "Player.is_on_floor",
            "PlayerInputHandler.update optional via enable_input=True",
            "Player.get_obs optional via enable_observation=True",
        ],
        "space_step": "original Pymunk Space.step is not patched",
        "fallback": "strict=True raises on native helper failure; strict=False explicitly falls back and counts failures",
        "source_sha256": FROZEN_ENV_SHA256,
        "enable_env_var": ENABLE_ENV_VAR,
        "enable_input_env_var": ENABLE_INPUT_ENV_VAR,
        "enable_observation_env_var": ENABLE_OBSERVATION_ENV_VAR,
        "chipmunk_bridge": chipmunk_bridge,
    }


def _attach_input_patch(input_class: type, helper: Any | None, *, strict: bool) -> _InputPatchState:
    state = _INPUT_PATCH_REGISTRY.get(input_class)
    if state is not None:
        if getattr(input_class, "update") is not state.hook:
            raise RuntimeError("PlayerInputHandler.update changed while native input patch registry is active")
        state.refcount += 1
        return state

    original = getattr(input_class, "update")
    input_helper = helper if helper is not None else _load_default_input_helper()
    state = _InputPatchState(
        input_class=input_class,
        original=original,
        hook=None,
        helper=input_helper,
        strict=bool(strict),
        refcount=1,
    )

    def patched_update(handler, action):
        try:
            return input_helper.update(handler, action)
        except Exception as exc:
            state.native_failures += 1
            state.last_native_error = f"{type(exc).__name__}: {exc}"
            if state.strict:
                raise RuntimeError("native input update failed; strict fallback disabled") from exc
            return state.original(handler, action)

    state.hook = patched_update
    _INPUT_PATCH_REGISTRY[input_class] = state
    setattr(input_class, "update", patched_update)
    return state


def _restore_input_patch(input_class: type, state: _InputPatchState | None) -> None:
    if state is None:
        return
    state.refcount -= 1
    if state.refcount <= 0:
        if getattr(input_class, "update") is state.hook:
            setattr(input_class, "update", state.original)
        if _INPUT_PATCH_REGISTRY.get(input_class) is state:
            del _INPUT_PATCH_REGISTRY[input_class]


@dataclass
class NativeGamePatch(AbstractContextManager):
    """Reversible class-level patch for the frozen game's Player class."""

    player_class: type
    helper: Any | None = None
    source_path: str | os.PathLike[str] | None = None
    expected_sha256: str = FROZEN_ENV_SHA256
    strict: bool = True
    allow_synthetic_class: bool = False
    enable_input: bool = False
    input_class: type | None = None
    input_helper: Any | None = None
    game_env: Any | None = None
    players: tuple[Any, ...] = ()
    enable_observation: bool = False
    observation_helper: Any | None = None

    def __post_init__(self) -> None:
        self.original_is_on_floor = None
        self._attached = False
        self._helper = self.helper
        self._state: _ClassPatchState | None = None
        self._input_state: _InputPatchState | None = None
        self._observation_patch = None
        self._original_reset = None
        self._reset_hook = None

    def attach(self) -> "NativeGamePatch":
        if self._attached:
            return self
        _verify_player_class_source(
            self.player_class,
            self.source_path,
            expected_sha256=self.expected_sha256,
            allow_synthetic_class=self.allow_synthetic_class,
        )
        state = _CLASS_PATCH_REGISTRY.get(self.player_class)
        try:
            if state is not None:
                if getattr(self.player_class, "is_on_floor") is not state.hook:
                    raise RuntimeError("Player.is_on_floor changed while native patch registry is active")
                state.refcount += 1
                self.original_is_on_floor = state.original
                self._helper = state.helper
                self._state = state
                self._attached = True
                if self.enable_input:
                    if self.input_class is None:
                        raise ValueError("input_class is required when enable_input=True")
                    self._input_state = _attach_input_patch(
                        self.input_class,
                        self.input_helper,
                        strict=bool(self.strict),
                    )
                if self.enable_observation:
                    self._ensure_reset_hook()
                    self._refresh_observation_patch()
                return self

            original = getattr(self.player_class, "is_on_floor")
            helper = self._helper if self._helper is not None else _load_default_geometry_helper()
            state = _ClassPatchState(
                player_class=self.player_class,
                original=original,
                hook=None,
                helper=helper,
                strict=bool(self.strict),
            )

            def patched_is_on_floor(player):
                try:
                    return _native_is_on_floor(player, helper)
                except Exception as exc:
                    state.native_failures += 1
                    state.last_native_error = f"{type(exc).__name__}: {exc}"
                    if state.strict:
                        raise RuntimeError("native is_on_floor failed; strict fallback disabled") from exc
                    return state.original(player)

            state.hook = patched_is_on_floor
            state.refcount = 1
            self.original_is_on_floor = original
            self._helper = helper
            self._state = state
            _CLASS_PATCH_REGISTRY[self.player_class] = state
            setattr(self.player_class, "is_on_floor", patched_is_on_floor)
            self._attached = True
            if self.enable_input:
                if self.input_class is None:
                    raise ValueError("input_class is required when enable_input=True")
                self._input_state = _attach_input_patch(
                    self.input_class,
                    self.input_helper,
                    strict=bool(self.strict),
                )
            if self.enable_observation:
                self._ensure_reset_hook()
                self._refresh_observation_patch()
            return self
        except Exception:
            self.restore()
            raise

    def _current_players(self) -> tuple[Any, ...]:
        if self.game_env is not None:
            players = getattr(self.game_env, "players", None)
            if isinstance(players, dict):
                return tuple(players.values())
            return tuple(players or ())
        return tuple(self.players)

    def _resolve_observation_helper(self) -> Any:
        if self.observation_helper is None:
            self.observation_helper = _load_default_observation_helper(self.player_class)
        return self.observation_helper

    def _refresh_observation_patch(self) -> None:
        if self._observation_patch is not None:
            self._observation_patch.restore()
            self._observation_patch = None
        self.players = self._current_players()
        if not self.players:
            raise ValueError("players are required when enable_observation=True")
        helper = self._resolve_observation_helper()
        patch_factory = _load_observation_patch_factory()
        observation_patch = patch_factory(
            self.players,
            helper=helper,
            source_path=self.source_path,
            expected_sha256=self.expected_sha256,
        )
        observation_patch.attach()
        self._observation_patch = observation_patch

    def _ensure_reset_hook(self) -> None:
        if self.game_env is None or self._reset_hook is not None:
            return
        original_reset = getattr(self.game_env, "reset", None)
        if original_reset is None:
            return

        patch = self

        def reset_hook(*args, **kwargs):
            original_observe_all = getattr(patch.game_env, "_observe_all")

            def observe_all_hook(*observe_args, **observe_kwargs):
                patch._refresh_observation_patch()
                return original_observe_all(*observe_args, **observe_kwargs)

            setattr(patch.game_env, "_observe_all", observe_all_hook)
            try:
                return original_reset(*args, **kwargs)
            finally:
                if getattr(patch.game_env, "_observe_all", None) is observe_all_hook:
                    setattr(patch.game_env, "_observe_all", original_observe_all)

        self._original_reset = original_reset
        self._reset_hook = reset_hook
        setattr(self.game_env, "reset", reset_hook)

    def restore(self) -> None:
        if not self._attached:
            return
        if self._observation_patch is not None:
            self._observation_patch.restore()
            self._observation_patch = None
        if self.enable_input and self.input_class is not None:
            _restore_input_patch(self.input_class, self._input_state)
            self._input_state = None
        if self._reset_hook is not None and self.game_env is not None:
            if getattr(self.game_env, "reset", None) is self._reset_hook:
                setattr(self.game_env, "reset", self._original_reset)
            self._reset_hook = None
            self._original_reset = None
        state = self._state
        if state is not None:
            state.refcount -= 1
            if state.refcount <= 0:
                if getattr(self.player_class, "is_on_floor") is state.hook:
                    setattr(self.player_class, "is_on_floor", state.original)
                if _CLASS_PATCH_REGISTRY.get(self.player_class) is state:
                    del _CLASS_PATCH_REGISTRY[self.player_class]
        self.original_is_on_floor = None
        self._state = None
        self._attached = False

    def receipt(self) -> dict[str, Any]:
        state = self._state
        input_state = self._input_state
        if state is None:
            return {
                "attached": False,
                "strict": bool(self.strict),
                "native_failures": 0,
                "last_native_error": None,
                "input_enabled": bool(self.enable_input),
                "input_native_failures": 0,
                "input_last_native_error": None,
                "observation_enabled": bool(self.enable_observation),
                "observation_players": 0,
                "reset_hooked": False,
                "enable_input_env_var": ENABLE_INPUT_ENV_VAR,
                "enable_input_env_value": os.environ.get(ENABLE_INPUT_ENV_VAR),
                "enable_observation_env_var": ENABLE_OBSERVATION_ENV_VAR,
                "enable_observation_env_value": os.environ.get(ENABLE_OBSERVATION_ENV_VAR),
            }
        return {
            "attached": self._attached,
            "strict": state.strict,
            "native_failures": state.native_failures,
            "last_native_error": state.last_native_error,
            "refcount": state.refcount,
            "patched_class": f"{state.player_class.__module__}.{state.player_class.__qualname__}",
            "input_enabled": input_state is not None,
            "input_native_failures": 0 if input_state is None else input_state.native_failures,
            "input_last_native_error": None if input_state is None else input_state.last_native_error,
            "input_refcount": None if input_state is None else input_state.refcount,
            "observation_enabled": self._observation_patch is not None,
            "observation_players": len(self.players) if self._observation_patch is not None else 0,
            "reset_hooked": self._reset_hook is not None,
            "enable_input_env_var": ENABLE_INPUT_ENV_VAR,
            "enable_input_env_value": os.environ.get(ENABLE_INPUT_ENV_VAR),
            "enable_observation_env_var": ENABLE_OBSERVATION_ENV_VAR,
            "enable_observation_env_value": os.environ.get(ENABLE_OBSERVATION_ENV_VAR),
        }

    def __enter__(self) -> "NativeGamePatch":
        return self.attach()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.restore()
        return False


def patch_frozen_player_class(player_class: type, **kwargs) -> NativeGamePatch:
    """Create a reversible opt-in patch for ``player_class.is_on_floor``."""

    return NativeGamePatch(player_class, **kwargs)


def _find_raw_game_env(env: Any) -> Any:
    seen: set[int] = set()
    current = env
    for _ in range(8):
        if id(current) in seen:
            break
        seen.add(id(current))
        if hasattr(current, "players"):
            return current
        for attr in ("raw_env", "env", "unwrapped"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                current = child
                break
        else:
            break
    raise ValueError("could not find raw game env with players")


def _player_class_from_game_env(game_env: Any) -> type:
    players = getattr(game_env, "players", None)
    if isinstance(players, dict):
        values = list(players.values())
    else:
        values = list(players or [])
    if not values:
        raise ValueError("raw game env players must be non-empty before patching")
    return type(values[0])


def _input_class_from_game_env(game_env: Any) -> type:
    players = getattr(game_env, "players", None)
    if isinstance(players, dict):
        values = list(players.values())
    else:
        values = list(players or [])
    for player in values:
        input_handler = getattr(player, "input", None)
        if input_handler is not None:
            return type(input_handler)
    raise ValueError("raw game env players must have input handlers when enable_input=True")


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_enabled_flag(
    *,
    preferred: bool | None,
    legacy: bool | None = None,
    env_var: str,
    flag_name: str,
    legacy_name: str | None = None,
) -> bool:
    if preferred is not None and legacy is not None and bool(preferred) != bool(legacy):
        other = legacy_name or "legacy flag"
        raise ValueError(f"conflicting {flag_name} and {other} values")
    if preferred is not None:
        return bool(preferred)
    if legacy is not None:
        return bool(legacy)
    return _env_enabled(env_var)


def install_native_game_patch(
    env: Any,
    *,
    source_path: str | os.PathLike[str] | None = None,
    helper: Any | None = None,
    enabled: bool = True,
    expected_sha256: str = FROZEN_ENV_SHA256,
    strict: bool = True,
    allow_synthetic_class: bool = False,
    enable_input: bool | None = None,
    native_input: bool | None = None,
    input_helper: Any | None = None,
    enable_observation: bool | None = None,
    observation_helper: Any | None = None,
) -> NativeGamePatch | None:
    """Install the safe native game patch on a live env and retain it strongly.

    This is the grouped-worker-friendly entry point: fork/spawn factories pass
    only ``source_path`` and let each worker construct native handles locally.
    """

    if not enabled:
        return None
    if source_path is None:
        raise ValueError("source_path is required when installing native game patch")
    game_env = _find_raw_game_env(env)
    player_class = _player_class_from_game_env(game_env)
    native_input_enabled = _resolve_enabled_flag(
        preferred=enable_input,
        legacy=native_input,
        env_var=ENABLE_INPUT_ENV_VAR,
        flag_name="enable_input",
        legacy_name="native_input",
    )
    native_observation_enabled = _resolve_enabled_flag(
        preferred=enable_observation,
        env_var=ENABLE_OBSERVATION_ENV_VAR,
        flag_name="enable_observation",
    )
    _verify_player_class_source(
        player_class,
        source_path,
        expected_sha256=expected_sha256,
        allow_synthetic_class=allow_synthetic_class,
    )
    existing = getattr(env, "_ai2_native_game_patch", None) or getattr(game_env, "_ai2_native_game_patch", None)
    if existing is not None:
        if native_input_enabled and not existing.receipt().get("input_enabled"):
            raise RuntimeError("existing native game patch was installed without enable_input")
        if native_observation_enabled and not existing.receipt().get("observation_enabled"):
            raise RuntimeError("existing native game patch was installed without enable_observation")
        setattr(env, "_ai2_native_game_patch", existing)
        setattr(game_env, "_ai2_native_game_patch", existing)
        return existing
    input_class = _input_class_from_game_env(game_env) if native_input_enabled else None
    players = getattr(game_env, "players", None)
    player_values = tuple(players.values()) if isinstance(players, dict) else tuple(players or ())
    patch = NativeGamePatch(
        player_class,
        helper=helper,
        source_path=source_path,
        expected_sha256=expected_sha256,
        strict=strict,
        allow_synthetic_class=allow_synthetic_class,
        enable_input=native_input_enabled,
        input_class=input_class,
        input_helper=input_helper,
        game_env=game_env,
        players=player_values,
        enable_observation=native_observation_enabled,
        observation_helper=observation_helper,
    )
    patch.attach()
    setattr(env, "_ai2_native_game_patch", patch)
    setattr(game_env, "_ai2_native_game_patch", patch)
    return patch


def restore_native_game_patch(env: Any) -> None:
    """Restore and remove a patch installed by ``install_native_game_patch``."""

    game_env = None
    try:
        game_env = _find_raw_game_env(env)
    except ValueError:
        pass
    patch = getattr(env, "_ai2_native_game_patch", None)
    if patch is None and game_env is not None:
        patch = getattr(game_env, "_ai2_native_game_patch", None)
    if patch is not None:
        patch.restore()
    for target in (env, game_env):
        if target is not None and hasattr(target, "_ai2_native_game_patch"):
            delattr(target, "_ai2_native_game_patch")


def install_from_environment_module(
    environment_module: Any,
    *,
    enabled: bool | None = None,
    helper: Any | None = None,
    source_path: str | os.PathLike[str] | None = None,
    expected_sha256: str = FROZEN_ENV_SHA256,
    strict: bool = True,
    allow_synthetic_class: bool = False,
    enable_input: bool | None = None,
    native_input: bool | None = None,
    input_helper: Any | None = None,
    enable_observation: bool | None = None,
    observation_helper: Any | None = None,
) -> NativeGamePatch | None:
    """Optionally patch an imported frozen environment module.

    ``enabled=None`` reads ``AI2_ENABLE_NATIVE_GAME_PATCH``. False is a strict
    no-op, which keeps ordinary imports and benchmarks behavior-identical.
    """

    if enabled is None:
        enabled = os.environ.get(ENABLE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None
    if source_path is None and not allow_synthetic_class:
        module_file = getattr(environment_module, "__file__", None)
        if not module_file:
            raise ValueError("environment module source path is required when enabled")
        source_path = module_file
    if source_path is not None:
        try:
            verify_source_hash(source_path, expected_sha256=expected_sha256)
        except Exception as exc:
            raise ValueError(f"environment module source verification failed for {source_path}") from exc
    native_input_enabled = _resolve_enabled_flag(
        preferred=enable_input,
        legacy=native_input,
        env_var=ENABLE_INPUT_ENV_VAR,
        flag_name="enable_input",
        legacy_name="native_input",
    )
    native_observation_enabled = _resolve_enabled_flag(
        preferred=enable_observation,
        env_var=ENABLE_OBSERVATION_ENV_VAR,
        flag_name="enable_observation",
    )
    player_class = getattr(environment_module, "Player")
    input_class = getattr(environment_module, "PlayerInputHandler", None) if native_input_enabled else None
    patch = NativeGamePatch(
        player_class,
        helper=helper,
        source_path=source_path,
        expected_sha256=expected_sha256,
        strict=strict,
        allow_synthetic_class=allow_synthetic_class,
        enable_input=native_input_enabled,
        input_class=input_class,
        input_helper=input_helper,
        enable_observation=native_observation_enabled,
        observation_helper=observation_helper,
    )
    patch.attach()
    return patch
