"""Navigation-scoped grouped VecEnv with per-lane RNG objects.

This is an opt-in experimental variant of :mod:`grouped_vec_env`.  It keeps
the original grouped worker construction semantics: environment factories are
still built under the legacy process-global ``random``/``np.random`` state
swap, so constructor-time seeding and state capture match the baseline worker.

After construction, reset/step/autoreset run under lane-local ``random.Random``
and legacy ``np.random.RandomState`` objects dispatched through patched frozen
game modules.  The steady path therefore avoids copying full MT19937 states on
every lane step while preserving the same generators and draw APIs.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import importlib
import random
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


_MISSING = object()
FROZEN_TRAIN_SHA256 = "b93585d97ab00a603f492b3d42b3f4334732c25756651d8fc21f2a78f99f3fc6"
FROZEN_ENVIRONMENT_SHA256 = "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"


@dataclass
class EnvRngBundle:
    """One lane's mutable Python and NumPy legacy RNG objects."""

    python_random: random.Random
    numpy_random: np.random.RandomState

    @classmethod
    def from_states(cls, python_state: object, numpy_state: tuple[Any, ...]) -> "EnvRngBundle":
        py_rng = random.Random()
        py_rng.setstate(python_state)
        np_rng = np.random.RandomState()
        np_rng.set_state(numpy_state)
        return cls(py_rng, np_rng)


class _CurrentRngBundle:
    def __init__(self) -> None:
        self._current: contextvars.ContextVar[tuple[EnvRngBundle, ...]] = contextvars.ContextVar(
            "isolated_rng_current_bundle",
            default=(),
        )

    def get(self) -> EnvRngBundle:
        stack = self._current.get()
        if not stack:
            raise RuntimeError("RNG call outside an active isolated RNG lane")
        return stack[-1]

    @contextlib.contextmanager
    def use(self, bundle: EnvRngBundle) -> Iterator[None]:
        stack = self._current.get()
        token = self._current.set((*stack, bundle))
        try:
            yield
        finally:
            current_stack = self._current.get()
            if not current_stack or current_stack[-1] is not bundle:
                self._current.reset(token)
                raise RuntimeError("isolated RNG lane context stack corrupted")
            self._current.reset(token)


current_rng_bundle = _CurrentRngBundle()


@contextlib.contextmanager
def use_rng_bundle(bundle: EnvRngBundle) -> Iterator[None]:
    """Activate ``bundle`` for patched RNG module calls in this thread."""

    with current_rng_bundle.use(bundle):
        yield


class RandomModuleProxy:
    """Fail-closed proxy for the subset of stdlib ``random`` used by the game."""

    _allowed = {
        "choice",
        "choices",
        "getrandbits",
        "getstate",
        "randint",
        "random",
        "seed",
        "setstate",
        "uniform",
    }

    def __init__(self, current: _CurrentRngBundle) -> None:
        self._current = current

    def _rng(self) -> random.Random:
        return self._current.get().python_random

    def random(self) -> float:
        return self._rng().random()

    def uniform(self, a: float, b: float) -> float:
        return self._rng().uniform(a, b)

    def randint(self, a: int, b: int) -> int:
        return self._rng().randint(a, b)

    def getrandbits(self, k: int) -> int:
        return self._rng().getrandbits(k)

    def choice(self, seq: Sequence[Any]) -> Any:
        return self._rng().choice(seq)

    def choices(self, population: Sequence[Any], weights: Any = None, *, cum_weights: Any = None, k: int = 1) -> list[Any]:
        return self._rng().choices(population, weights=weights, cum_weights=cum_weights, k=k)

    def seed(self, a: Any = None, version: int = 2) -> None:
        self._rng().seed(a, version=version)

    def getstate(self) -> object:
        return self._rng().getstate()

    def setstate(self, state: object) -> None:
        self._rng().setstate(state)

    def __getattr__(self, name: str) -> Any:
        if name in self._allowed:
            return getattr(self, name)
        raise AttributeError(f"random.{name} is not allowed in isolated RNG navigation scope")


class NumpyRandomProxy:
    """Fail-closed proxy for legacy ``np.random`` calls used by navigation."""

    _allowed = {"get_state", "randint", "seed", "set_state", "uniform"}

    def __init__(self, current: _CurrentRngBundle) -> None:
        self._current = current

    def _rng(self) -> np.random.RandomState:
        return self._current.get().numpy_random

    def randint(self, *args: Any, **kwargs: Any) -> Any:
        return self._rng().randint(*args, **kwargs)

    def uniform(self, *args: Any, **kwargs: Any) -> Any:
        return self._rng().uniform(*args, **kwargs)

    def seed(self, seed: Any = None) -> None:
        self._rng().seed(seed)

    def get_state(self) -> tuple[Any, ...]:
        return self._rng().get_state()

    def set_state(self, state: tuple[Any, ...]) -> None:
        self._rng().set_state(state)

    def __getattr__(self, name: str) -> Any:
        if name in self._allowed:
            return getattr(self, name)
        raise AttributeError(f"np.random.{name} is not allowed in isolated RNG navigation scope")


class NumpyModuleProxy:
    """Proxy for a module-local ``np`` binding without mutating global NumPy."""

    def __init__(self, numpy_module: Any, random_proxy: NumpyRandomProxy) -> None:
        self._numpy_module = numpy_module
        self.random = random_proxy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._numpy_module, name)


class ScopedRngPatch:
    """Patch selected module globals to dispatch through lane-local RNGs."""

    def __init__(
        self,
        *,
        random_modules: Sequence[Any] | None = None,
        numpy_modules: Sequence[Any] | None = None,
    ) -> None:
        self.random_modules = list(random_modules or [])
        self.numpy_modules = list(numpy_modules or [])
        self.random_proxy = RandomModuleProxy(current_rng_bundle)
        self.numpy_random_proxy = NumpyRandomProxy(current_rng_bundle)
        self.numpy_proxy = NumpyModuleProxy(np, self.numpy_random_proxy)
        self._originals: list[tuple[Any, str, Any]] = []
        self._installed = False

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        self.install()
        try:
            yield
        finally:
            self.uninstall()

    def install(self) -> None:
        if self._installed:
            return
        try:
            for module in self.random_modules:
                self._patch_attr(module, "random", self.random_proxy, expected=random)
            for module in self.numpy_modules:
                self._patch_attr(module, "np", self.numpy_proxy, expected=np)
        except BaseException:
            self.uninstall()
            raise
        self._installed = True

    def uninstall(self) -> None:
        while self._originals:
            module, name, original = self._originals.pop()
            if original is _MISSING:
                try:
                    delattr(module, name)
                except AttributeError:
                    pass
            else:
                setattr(module, name, original)
        self._installed = False

    def _patch_attr(self, module: Any, name: str, replacement: Any, *, expected: Any) -> None:
        original = getattr(module, name, _MISSING)
        if original is not expected:
            raise RuntimeError(
                f"Refusing isolated RNG patch for {module!r}.{name}: expected the original module binding"
            )
        setattr(module, name, replacement)
        self._originals.append((module, name, original))


class IsolatedRngStates:
    """Legacy construction state-swap, then steady lane-local RNG dispatch."""

    def __init__(self, size: int) -> None:
        self._legacy_states: list[tuple[object, tuple[Any, ...]]] | None = [
            (random.getstate(), np.random.get_state()) for _ in range(size)
        ]
        self._bundles: list[EnvRngBundle] | None = None
        self._fatal_lanes: set[int] = set()
        self._patch: ScopedRngPatch | None = None

    def enable_isolated_dispatch(self, patch: ScopedRngPatch) -> None:
        if self._bundles is not None:
            return
        if self._legacy_states is None:
            raise RuntimeError("isolated RNG states already converted")
        self._bundles = [EnvRngBundle.from_states(py_state, np_state) for py_state, np_state in self._legacy_states]
        self._legacy_states = None
        self._patch = patch
        self._patch.install()

    def close(self) -> None:
        if self._patch is not None:
            self._patch.uninstall()
            self._patch = None

    def run(self, index: int, func: Callable[[], Any], *, seed: int | None = None) -> Any:
        del seed  # SB3 passes reset seeds to the env; RNG semantics come from the called code.
        if self._bundles is None:
            if self._legacy_states is None:
                raise RuntimeError("isolated RNG states not initialized")
            outer_random_state = random.getstate()
            outer_np_state = np.random.get_state()
            try:
                random.setstate(self._legacy_states[index][0])
                np.random.set_state(self._legacy_states[index][1])
                result = func()
                self._legacy_states[index] = (random.getstate(), np.random.get_state())
                return result
            finally:
                random.setstate(outer_random_state)
                np.random.set_state(outer_np_state)

        if index in self._fatal_lanes:
            raise RuntimeError(f"Lane {index} has fatal RNG exception state; refusing silent reuse")
        try:
            with use_rng_bundle(self._bundles[index]):
                return func()
        except BaseException:
            # The generator objects have already advanced and there is no cheap
            # snapshot rollback in the optimized path. Treat worker exceptions
            # as fatal for that lane so callers do not continue from a drifted
            # RNG stream after an interrupted step/reset.
            self._fatal_lanes.add(index)
            raise


def _unwrap_chain(env: Any) -> Iterator[Any]:
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "env", None)


def assert_navigation_constant_opponent_scope(envs: Sequence[Any]) -> None:
    """Fail closed unless each env is a FrozenOpponentWrapper navigation stack."""

    for idx, env in enumerate(envs):
        chain_names = [obj.__class__.__name__ for obj in _unwrap_chain(env)]
        if "FrozenOpponentWrapper" not in chain_names:
            raise RuntimeError(
                "Isolated RNG grouped env is navigation-only; expected FrozenOpponentWrapper "
                f"in env {idx} wrapper chain, got {chain_names}"
            )
        if "DamageTrackingWrapper" in chain_names:
            raise RuntimeError("Isolated RNG grouped env does not cover combat DamageTrackingWrapper scope")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_path(module: Any) -> Path:
    value = getattr(module, "__file__", None)
    if not value:
        raise RuntimeError(f"Loaded module {module!r} has no __file__ for frozen source verification")
    return Path(value).resolve()


def _verify_loaded_path(module: Any, expected_path: Path, label: str) -> None:
    actual_path = _module_path(module)
    if actual_path != expected_path:
        raise RuntimeError(f"{label} source path mismatch: {actual_path} != {expected_path}")


def _verify_file_hash(path: Path, expected_sha256: str, label: str) -> str:
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise RuntimeError(f"{label} source hash mismatch for {path}: {digest} != {expected_sha256}")
    return digest


def verify_frozen_sources(source_root: str | Path | None = None) -> dict[str, str]:
    """Verify loaded frozen modules match the expected train/env files and hashes."""

    root = Path.cwd() if source_root is None else Path(source_root)
    root = root.resolve()
    train_path = (root / "train.py").resolve()
    environment_path = (root / "UTMIST-AI2-main" / "environment" / "environment.py").resolve()
    train_module = sys.modules.get("train")
    env_module = sys.modules.get("environment.environment")
    if train_module is None:
        raise RuntimeError("Loaded train.py module not found as sys.modules['train']")
    if env_module is None:
        raise RuntimeError("Loaded environment.py module not found as sys.modules['environment.environment']")
    _verify_loaded_path(train_module, train_path, "train.py")
    _verify_loaded_path(env_module, environment_path, "environment.py")
    train_sha = _verify_file_hash(train_path, FROZEN_TRAIN_SHA256, "train.py")
    env_sha = _verify_file_hash(environment_path, FROZEN_ENVIRONMENT_SHA256, "environment.py")
    return {
        "source_root": str(root),
        "train_path": str(train_path),
        "train_sha256": train_sha,
        "environment_path": str(environment_path),
        "environment_sha256": env_sha,
    }


def discover_frozen_rng_modules(source_root: str | Path | None = None) -> ScopedRngPatch:
    """Find already-imported frozen modules whose globals contain RNG calls."""

    verify_frozen_sources(source_root)
    random_module_names = {
        "environment.environment",
        "environment.agent",
        "environment.WarehouseBrawl",
    }
    random_modules: list[Any] = []
    numpy_modules: list[Any] = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if name in random_module_names or name.endswith(".environment.environment") or name.endswith(".environment.agent"):
            if getattr(module, "random", None) is random:
                random_modules.append(module)
        if name == "train" or name.endswith(".train"):
            if getattr(module, "np", None) is np:
                numpy_modules.append(module)
    if not random_modules:
        raise RuntimeError("No frozen game random modules are loaded for isolated RNG patching")
    if not numpy_modules:
        raise RuntimeError("No frozen train.py numpy module is loaded for isolated RNG patching")
    return ScopedRngPatch(random_modules=random_modules, numpy_modules=numpy_modules)


class _InstallOnFirstRecvRemote:
    def __init__(self, remote: Any, rng_states: IsolatedRngStates, rng_scope: str, source_root: Path | None) -> None:
        self.remote = remote
        self.rng_states = rng_states
        self.rng_scope = rng_scope
        self.source_root = source_root
        self.installed = False

    def _install_if_needed(self) -> None:
        if self.installed:
            return
        base = importlib.import_module("grouped_vec_env")
        envs = getattr(base, "_isolated_rng_current_envs", None)
        if envs is None:
            raise RuntimeError("isolated RNG worker did not expose constructed envs")
        if self.rng_scope == "navigation":
            assert_navigation_constant_opponent_scope(envs)
            patch = discover_frozen_rng_modules(self.source_root)
        elif self.rng_scope == "test":
            patch = getattr(base, "_isolated_rng_test_patch", None)
            if patch is None:
                modules = []
                for env in envs:
                    module = sys.modules.get(env.__class__.__module__)
                    if module is not None and module not in modules:
                        modules.append(module)
                patch = ScopedRngPatch(
                    random_modules=[module for module in modules if getattr(module, "random", None) is random],
                    numpy_modules=[module for module in modules if getattr(module, "np", None) is np],
                )
        else:
            raise RuntimeError(f"Unsupported isolated RNG scope: {self.rng_scope!r}")
        self.rng_states.enable_isolated_dispatch(patch)
        self.installed = True

    def recv(self) -> Any:
        self._install_if_needed()
        return self.remote.recv()

    def close(self) -> Any:
        return self.remote.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.remote, name)


def _isolated_worker(
    remote: Any,
    parent_remote: Any,
    env_fns: list[Any],
    obs_raw: Any,
    rewards_raw: Any,
    dones_raw: Any,
    obs_shape: tuple[int, ...],
    rng_scope: str = "navigation",
    test_patch: ScopedRngPatch | None = None,
    source_root: str | Path | None = None,
) -> None:
    base = importlib.import_module("grouped_vec_env")
    base_worker = getattr(base, "_isolated_rng_original_worker", base._worker)
    original_rng_class = base._GlobalRngStates
    original_current_envs = getattr(base, "_isolated_rng_current_envs", _MISSING)
    original_test_patch = getattr(base, "_isolated_rng_test_patch", _MISSING)
    state_holder: dict[str, IsolatedRngStates] = {}
    resolved_source_root = Path(source_root).resolve() if source_root is not None else None

    class ControlledIsolatedRngStates(IsolatedRngStates):
        def __init__(self, size: int) -> None:
            super().__init__(size)
            state_holder["rng_states"] = self

    base._GlobalRngStates = ControlledIsolatedRngStates
    if test_patch is not None:
        base._isolated_rng_test_patch = test_patch

    original_array_from_raw = base._array_from_raw

    def exposing_array_from_raw(raw_array: Any, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        # The first array conversion happens immediately after env construction
        # in the base worker, so this is the earliest stable hook for exposing
        # constructed envs without copying the whole worker loop.
        frame = sys._getframe(1)
        envs = frame.f_locals.get("envs")
        if envs is not None:
            base._isolated_rng_current_envs = envs
        return original_array_from_raw(raw_array, shape, dtype)

    base._array_from_raw = exposing_array_from_raw
    try:
        # ``base._worker`` instantiates ``ControlledIsolatedRngStates`` before
        # constructing envs, so this proxy resolves and installs lazily at the
        # first command receive while preserving installer state thereafter.
        class RemoteProxy:
            def __init__(self) -> None:
                self._wrapped: _InstallOnFirstRecvRemote | None = None

            def _target(self) -> Any:
                rng_states = state_holder.get("rng_states")
                if rng_states is None:
                    return remote
                if self._wrapped is None:
                    self._wrapped = _InstallOnFirstRecvRemote(remote, rng_states, rng_scope, resolved_source_root)
                return self._wrapped

            def __getattr__(self, name: str) -> Any:
                return getattr(self._target(), name)

        return base_worker(
            RemoteProxy(),
            parent_remote,
            env_fns,
            obs_raw,
            rewards_raw,
            dones_raw,
            obs_shape,
        )
    finally:
        rng_states = state_holder.get("rng_states")
        if rng_states is not None:
            rng_states.close()
        base._GlobalRngStates = original_rng_class
        base._array_from_raw = original_array_from_raw
        if original_current_envs is _MISSING:
            try:
                delattr(base, "_isolated_rng_current_envs")
            except AttributeError:
                pass
        else:
            base._isolated_rng_current_envs = original_current_envs
        if original_test_patch is _MISSING:
            try:
                delattr(base, "_isolated_rng_test_patch")
            except AttributeError:
                pass
        else:
            base._isolated_rng_test_patch = original_test_patch


def _isolated_worker_navigation(
    remote: Any,
    parent_remote: Any,
    env_fns: list[Any],
    obs_raw: Any,
    rewards_raw: Any,
    dones_raw: Any,
    obs_shape: tuple[int, ...],
) -> None:
    return _isolated_worker(
        remote,
        parent_remote,
        env_fns,
        obs_raw,
        rewards_raw,
        dones_raw,
        obs_shape,
        rng_scope="navigation",
    )


def _isolated_worker_navigation_cwd(
    remote: Any,
    parent_remote: Any,
    env_fns: list[Any],
    obs_raw: Any,
    rewards_raw: Any,
    dones_raw: Any,
    obs_shape: tuple[int, ...],
) -> None:
    return _isolated_worker(
        remote,
        parent_remote,
        env_fns,
        obs_raw,
        rewards_raw,
        dones_raw,
        obs_shape,
        rng_scope="navigation",
        source_root=Path.cwd(),
    )


def _isolated_worker_test(
    remote: Any,
    parent_remote: Any,
    env_fns: list[Any],
    obs_raw: Any,
    rewards_raw: Any,
    dones_raw: Any,
    obs_shape: tuple[int, ...],
) -> None:
    return _isolated_worker(
        remote,
        parent_remote,
        env_fns,
        obs_raw,
        rewards_raw,
        dones_raw,
        obs_shape,
        rng_scope="test",
    )


class IsolatedRngGroupedSubprocVecEnv:
    """Drop-in grouped VecEnv class for navigation-only isolated RNG tests."""

    def __new__(
        cls,
        env_fns: Sequence[Callable[[], Any]],
        *,
        group_size: int = 2,
        start_method: Optional[str] = None,
        rng_scope: str = "navigation",
        test_patch: ScopedRngPatch | None = None,
    ) -> Any:
        base = importlib.import_module("grouped_vec_env")
        original_worker = base._worker
        original_saved_worker = getattr(base, "_isolated_rng_original_worker", _MISSING)

        if test_patch is not None:
            raise RuntimeError("test_patch is only supported for direct _isolated_worker tests, not spawn workers")
        if rng_scope == "navigation":
            worker = _isolated_worker_navigation_cwd
        elif rng_scope == "test":
            worker = _isolated_worker_test
        else:
            raise RuntimeError(f"Unsupported isolated RNG scope: {rng_scope!r}")

        base._isolated_rng_original_worker = original_worker
        base._worker = worker
        try:
            return base.GroupedSubprocVecEnv(env_fns, group_size=group_size, start_method=start_method)
        finally:
            base._worker = original_worker
            if original_saved_worker is _MISSING:
                try:
                    delattr(base, "_isolated_rng_original_worker")
                except AttributeError:
                    pass
            else:
                base._isolated_rng_original_worker = original_saved_worker


GroupedSubprocVecEnv = IsolatedRngGroupedSubprocVecEnv
