"""Grouped VecEnv variant that sends actions through shared memory.

This experimental transport keeps ``grouped_vec_env`` semantics intact while
removing per-step pickling of the action array.  The parent writes a float32
``num_envs x 10`` action batch into per-worker shared buffers and sends a cached
``("step_shared", None)`` token; the worker-side remote adapter translates that
token back into the original ``("step", actions)`` command using a defensive
copy of its shared slice.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import pickle
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv

import ai2.simulation.grouped_vec_env as _base


ACTION_DIMENSION = 10
_STEP_SHARED_BYTES = pickle.dumps(("step_shared", None), protocol=pickle.HIGHEST_PROTOCOL)


def _shared_actions_from_raw(raw_array: Any, rows: int) -> np.ndarray:
    return np.ctypeslib.as_array(raw_array).view(np.float32).reshape((int(rows), ACTION_DIMENSION))


class _SharedActionRemote:
    def __init__(self, remote: Any, shared_actions: np.ndarray) -> None:
        self.remote = remote
        self.shared_actions = shared_actions

    def recv(self) -> Any:
        item = self.remote.recv()
        if item == ("step_shared", None):
            return ("step", np.array(self.shared_actions, copy=True))
        return item

    def send(self, value: Any) -> Any:
        return self.remote.send(value)

    def send_bytes(self, value: bytes) -> Any:
        return self.remote.send_bytes(value)

    def close(self) -> Any:
        return self.remote.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.remote, name)


def _shared_action_worker(
    action_raw: Any,
    action_rows: int,
    remote: Any,
    parent_remote: Any,
    env_fns: list[CloudpickleWrapper],
    obs_raw: Any,
    rewards_raw: Any,
    dones_raw: Any,
    obs_shape: tuple[int, ...],
) -> None:
    shared_actions = _shared_actions_from_raw(action_raw, action_rows)
    return _base._worker(
        _SharedActionRemote(remote, shared_actions),
        parent_remote,
        env_fns,
        obs_raw,
        rewards_raw,
        dones_raw,
        obs_shape,
    )


def _shared_action_batched_worker(
    action_raw: Any,
    action_rows: int,
    remote: Any,
    parent_remote: Any,
    env_fns: list[CloudpickleWrapper],
    obs_raw: Any,
    rewards_raw: Any,
    dones_raw: Any,
    obs_shape: tuple[int, ...],
) -> None:
    from ai2.simulation.batched_rng_vec_env import _batched_worker

    shared_actions = _shared_actions_from_raw(action_raw, action_rows)
    return _batched_worker(
        _SharedActionRemote(remote, shared_actions),
        parent_remote,
        env_fns,
        obs_raw,
        rewards_raw,
        dones_raw,
        obs_shape,
    )


class SharedActionGroupedSubprocVecEnv(_base.GroupedSubprocVecEnv):
    """Drop-in experimental grouped VecEnv with shared-memory action transport."""

    _worker_target = staticmethod(_shared_action_worker)

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        *,
        group_size: int = 2,
        start_method: Optional[str] = None,
    ) -> None:
        if not env_fns:
            raise ValueError("SharedActionGroupedSubprocVecEnv needs at least one environment")
        if group_size <= 0:
            raise ValueError("group_size must be positive")

        probe_env = env_fns[0]()
        try:
            observation_space = _base._validate_box_float32(probe_env.observation_space, "observation")
            action_space = _base._validate_box_float32(probe_env.action_space, "action")
            if tuple(action_space.shape) != (ACTION_DIMENSION,):
                raise ValueError(
                    f"SharedActionGroupedSubprocVecEnv requires 10D action space, got {action_space.shape}"
                )
        finally:
            probe_env.close()

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self._shared_action_raws = []
        self.shared_actions = []
        self._step_shared_payload = _STEP_SHARED_BYTES
        worker_target = self._worker_target

        group_sizes = [
            min(int(group_size), len(env_fns) - start)
            for start in range(0, len(env_fns), int(group_size))
        ]
        for local_count in group_sizes:
            action_raw = ctx.RawArray(ctypes.c_float, int(local_count * ACTION_DIMENSION))
            self._shared_action_raws.append(action_raw)
            self.shared_actions.append(_shared_actions_from_raw(action_raw, local_count))

        try:
            self.num_envs = len(env_fns)
            self.waiting = False
            self.closed = False
            self.group_size = int(group_size)
            self.group_slices = []
            self.group_sizes = []
            for start in range(0, len(env_fns), self.group_size):
                stop = min(start + self.group_size, len(env_fns))
                self.group_slices.append(slice(start, stop))
                self.group_sizes.append(stop - start)
            self.num_groups = len(self.group_slices)
            self.shared_observations = []
            self.shared_rewards = []
            self.shared_dones = []
            self.remotes = []
            self.work_remotes = []
            self.processes = []

            for group_index, (group_slice, local_count) in enumerate(zip(self.group_slices, self.group_sizes)):
                obs_raw = ctx.RawArray(ctypes.c_float, int(local_count * np.prod(observation_space.shape)))
                rewards_raw = ctx.RawArray(ctypes.c_double, local_count)
                dones_raw = ctx.RawArray(ctypes.c_bool, local_count)
                self.shared_observations.append(
                    _base._array_from_raw(obs_raw, (local_count, *observation_space.shape), np.dtype(np.float32))
                )
                self.shared_rewards.append(_base._array_from_raw(rewards_raw, (local_count,), np.dtype(np.float64)))
                self.shared_dones.append(_base._array_from_raw(dones_raw, (local_count,), np.dtype(np.bool_)))

                remote, work_remote = ctx.Pipe()
                wrapped_env_fns = [CloudpickleWrapper(env_fn) for env_fn in env_fns[group_slice]]
                process = ctx.Process(
                    target=partial(
                        worker_target,
                        self._shared_action_raws[group_index],
                        local_count,
                    ),
                    args=(work_remote, remote, wrapped_env_fns, obs_raw, rewards_raw, dones_raw, observation_space.shape),
                    daemon=True,
                )
                process.start()
                work_remote.close()
                self.remotes.append(remote)
                self.work_remotes.append(work_remote)
                self.processes.append(process)

            self._validate_worker_spaces(observation_space, action_space)
            VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        except Exception:
            self.close()
            raise

    def step_async(self, actions: np.ndarray) -> None:
        if self.waiting:
            raise RuntimeError("Already waiting for grouped step results")
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_envs, ACTION_DIMENSION)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected action batch shape {expected_shape}, got {actions.shape}")
        for group_slice, shared_action, remote in zip(self.group_slices, self.shared_actions, self.remotes):
            shared_action[...] = actions[group_slice]
            remote.send_bytes(self._step_shared_payload)
        self.waiting = True


class SharedBatchedRngGroupedSubprocVecEnv(SharedActionGroupedSubprocVecEnv):
    """Shared-action transport plus batched grouped-worker RNG restoration."""

    _worker_target = staticmethod(_shared_action_batched_worker)


GroupedSubprocVecEnv = SharedActionGroupedSubprocVecEnv
