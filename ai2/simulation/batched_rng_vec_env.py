"""Experimental grouped VecEnv variant with batched outer RNG restoration.

This module intentionally leaves ``grouped_vec_env.py`` unchanged. It reuses the
original grouped worker loop, but runs that worker with a batch-aware
``_GlobalRngStates`` implementation so process-global ``random`` and
``np.random`` outer states are saved/restored once per group ``step``/``reset``
command instead of once per lane. Per-lane restore/capture order is preserved.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any, Optional

import gymnasium as gym
import numpy as np

import ai2.simulation.grouped_vec_env as _base


class _BatchController:
    def __init__(self) -> None:
        self.active = False
        self.command: str | None = None
        self.outer_random_state: object | None = None
        self.outer_np_state: tuple[Any, ...] | None = None
        self.outer_save_count = 0
        self.outer_restore_count = 0

    def begin(self, command: str) -> None:
        self.end()
        self.active = command in {"step", "reset"}
        self.command = command if self.active else None

    def save_outer_once(self) -> None:
        if not self.active or self.outer_np_state is not None:
            return
        self.outer_random_state = random.getstate()
        self.outer_np_state = np.random.get_state()
        self.outer_save_count += 1

    def end(self) -> None:
        if self.outer_np_state is not None:
            random.setstate(self.outer_random_state)
            np.random.set_state(self.outer_np_state)
            self.outer_restore_count += 1
        self.active = False
        self.command = None
        self.outer_random_state = None
        self.outer_np_state = None

    def abort(self) -> None:
        self.end()


class _BatchedGlobalRngStates:
    def __init__(self, size: int, controller: _BatchController | None = None) -> None:
        self.controller = controller or _BatchController()
        self.states: list[tuple[object, tuple[Any, ...]]] = [
            (random.getstate(), np.random.get_state()) for _ in range(size)
        ]

    def run(self, index: int, func: Callable[[], Any], *, seed: int | None = None) -> Any:
        if not self.controller.active:
            outer_random_state = random.getstate()
            outer_np_state = np.random.get_state()
            try:
                random.setstate(self.states[index][0])
                np.random.set_state(self.states[index][1])
                result = func()
                self.states[index] = (random.getstate(), np.random.get_state())
                return result
            finally:
                random.setstate(outer_random_state)
                np.random.set_state(outer_np_state)

        self.controller.save_outer_once()
        try:
            random.setstate(self.states[index][0])
            np.random.set_state(self.states[index][1])
            result = func()
            self.states[index] = (random.getstate(), np.random.get_state())
            return result
        except BaseException:
            self.controller.abort()
            raise


class _BatchRemote:
    def __init__(self, remote: Any, controller: _BatchController) -> None:
        self.remote = remote
        self.controller = controller

    def recv(self) -> Any:
        item = self.remote.recv()
        command = item[0] if isinstance(item, tuple) and item else None
        self.controller.begin(str(command))
        return item

    def send(self, value: Any) -> Any:
        self.controller.end()
        return self.remote.send(value)

    def close(self) -> Any:
        self.controller.end()
        return self.remote.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.remote, name)


def _batched_worker(
    remote: Any,
    parent_remote: Any,
    env_fns: list[Any],
    obs_raw: Any,
    rewards_raw: Any,
    dones_raw: Any,
    obs_shape: tuple[int, ...],
) -> None:
    controller = _BatchController()
    original_rng_class = _base._GlobalRngStates

    class ControlledBatchedGlobalRngStates(_BatchedGlobalRngStates):
        def __init__(self, size: int) -> None:
            super().__init__(size, controller)

    _base._GlobalRngStates = ControlledBatchedGlobalRngStates
    try:
        return _base._worker(
            _BatchRemote(remote, controller),
            parent_remote,
            env_fns,
            obs_raw,
            rewards_raw,
            dones_raw,
            obs_shape,
        )
    finally:
        controller.end()
        _base._GlobalRngStates = original_rng_class


class BatchedRngGroupedSubprocVecEnv(_base.GroupedSubprocVecEnv):
    """Drop-in experimental variant of ``GroupedSubprocVecEnv``.

    Only the grouped worker's outer RNG preservation strategy changes. Public
    command protocol, shared buffers, env methods, and per-lane RNG streams are
    inherited from the original grouped implementation.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        *,
        group_size: int = 2,
        start_method: Optional[str] = None,
    ) -> None:
        original_worker = _base._worker
        _base._worker = _batched_worker
        try:
            super().__init__(env_fns, group_size=group_size, start_method=start_method)
        finally:
            _base._worker = original_worker


GroupedSubprocVecEnv = BatchedRngGroupedSubprocVecEnv
