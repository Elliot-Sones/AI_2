"""Grouped subprocess VecEnv for Box(float32) environments."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import random
from collections.abc import Callable, Sequence
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _array_from_raw(raw_array: Any, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return np.ctypeslib.as_array(raw_array).view(dtype).reshape(shape)


def _get_wrapper_attr(env: gym.Env, name: str) -> Any:
    if hasattr(env, "get_wrapper_attr"):
        return env.get_wrapper_attr(name)
    return getattr(env, name)


def _set_wrapper_attr(env: gym.Env, name: str, value: Any) -> None:
    if hasattr(env, "set_wrapper_attr"):
        env.set_wrapper_attr(name, value)
    else:
        setattr(env, name, value)


class _GlobalRngStates:
    def __init__(self, size: int) -> None:
        self.states: list[tuple[object, tuple[Any, ...]]] = [
            (random.getstate(), np.random.get_state()) for _ in range(size)
        ]

    def run(self, index: int, func: Callable[[], Any], *, seed: int | None = None) -> Any:
        outer_random_state = random.getstate()
        outer_np_state = np.random.get_state()
        try:
            # ``seed`` is passed through to env.reset(seed=...), but stock
            # SubprocVecEnv does not reseed process-global random/np.random
            # around reset.  Keep the per-lane global streams produced by env
            # construction/autoresets so grouped workers match stock workers.
            random.setstate(self.states[index][0])
            np.random.set_state(self.states[index][1])
            result = func()
            self.states[index] = (random.getstate(), np.random.get_state())
            return result
        finally:
            random.setstate(outer_random_state)
            np.random.set_state(outer_np_state)


def _worker(
    remote: Any,
    parent_remote: Any,
    env_fns: list[CloudpickleWrapper],
    obs_raw: Any,
    rewards_raw: Any,
    dones_raw: Any,
    obs_shape: tuple[int, ...],
) -> None:
    parent_remote.close()
    rng_states = _GlobalRngStates(len(env_fns))
    envs = [rng_states.run(local_idx, lambda env_fn=env_fn: _patch_env(env_fn.var())) for local_idx, env_fn in enumerate(env_fns)]
    obs_buffer = _array_from_raw(obs_raw, (len(envs), *obs_shape), np.dtype(np.float32))
    rewards_buffer = _array_from_raw(rewards_raw, (len(envs),), np.dtype(np.float64))
    dones_buffer = _array_from_raw(dones_raw, (len(envs),), np.dtype(np.bool_))
    reset_infos: list[dict[str, Any]] = [{} for _ in envs]

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "get_spaces":
                remote.send([(env.observation_space, env.action_space) for env in envs])
            elif cmd == "reset":
                for local_idx, (seed, options) in enumerate(data):
                    kwargs = {"options": options} if options else {}
                    obs, reset_infos[local_idx] = rng_states.run(
                        local_idx,
                        lambda local_idx=local_idx, kwargs=kwargs, seed=seed: envs[local_idx].reset(
                            seed=seed, **kwargs
                        ),
                        seed=seed,
                    )
                    obs_buffer[local_idx] = obs
                    rewards_buffer[local_idx] = 0.0
                    dones_buffer[local_idx] = False
                remote.send(list(reset_infos))
            elif cmd == "step":
                infos = []
                for local_idx, action in enumerate(data):
                    obs, reward, terminated, truncated, info = rng_states.run(
                        local_idx,
                        lambda local_idx=local_idx, action=action: envs[local_idx].step(action),
                    )
                    done = bool(terminated or truncated)
                    info = dict(info)
                    info["TimeLimit.truncated"] = bool(truncated and not terminated)
                    if done:
                        info["terminal_observation"] = np.array(obs, copy=True)
                        obs, reset_infos[local_idx] = rng_states.run(
                            local_idx,
                            lambda local_idx=local_idx: envs[local_idx].reset(),
                        )
                    obs_buffer[local_idx] = obs
                    rewards_buffer[local_idx] = reward
                    dones_buffer[local_idx] = done
                    infos.append(info)
                remote.send((infos, list(reset_infos)))
            elif cmd == "get_attr":
                remote.send([_get_wrapper_attr(envs[local_idx], data[1]) for local_idx in data[0]])
            elif cmd == "set_attr":
                for local_idx in data[0]:
                    _set_wrapper_attr(envs[local_idx], data[1], data[2])
                remote.send(None)
            elif cmd == "env_method":
                local_indices, method_name, method_args, method_kwargs = data
                remote.send(
                    [
                        _get_wrapper_attr(envs[local_idx], method_name)(*method_args, **method_kwargs)
                        for local_idx in local_indices
                    ]
                )
            elif cmd == "is_wrapped":
                remote.send([is_wrapped(envs[local_idx], data[1]) for local_idx in data[0]])
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the grouped worker")
    except EOFError:
        pass


def _validate_box_float32(space: spaces.Space, label: str) -> spaces.Box:
    if not isinstance(space, spaces.Box) or space.dtype != np.float32:
        raise ValueError(f"GroupedSubprocVecEnv requires Box float32 {label} spaces")
    return space


class GroupedSubprocVecEnv(VecEnv):
    """Run multiple Gymnasium environments in each subprocess.

    This is intentionally narrower than SB3's general SubprocVecEnv: it supports
    only Box(float32) observations/actions so workers can write fixed-size
    results into preallocated shared arrays.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        *,
        group_size: int = 2,
        start_method: Optional[str] = None,
    ) -> None:
        if not env_fns:
            raise ValueError("GroupedSubprocVecEnv needs at least one environment")
        if group_size <= 0:
            raise ValueError("group_size must be positive")

        probe_env = env_fns[0]()
        try:
            observation_space = _validate_box_float32(probe_env.observation_space, "observation")
            action_space = _validate_box_float32(probe_env.action_space, "action")
            obs_shape = observation_space.shape
        finally:
            probe_env.close()

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

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

        try:
            for group_slice, local_count in zip(self.group_slices, self.group_sizes):
                obs_raw = ctx.RawArray(ctypes.c_float, int(local_count * np.prod(obs_shape)))
                rewards_raw = ctx.RawArray(ctypes.c_double, local_count)
                dones_raw = ctx.RawArray(ctypes.c_bool, local_count)
                self.shared_observations.append(
                    _array_from_raw(obs_raw, (local_count, *obs_shape), np.dtype(np.float32))
                )
                self.shared_rewards.append(_array_from_raw(rewards_raw, (local_count,), np.dtype(np.float64)))
                self.shared_dones.append(_array_from_raw(dones_raw, (local_count,), np.dtype(np.bool_)))

                remote, work_remote = ctx.Pipe()
                wrapped_env_fns = [CloudpickleWrapper(env_fn) for env_fn in env_fns[group_slice]]
                process = ctx.Process(
                    target=_worker,
                    args=(work_remote, remote, wrapped_env_fns, obs_raw, rewards_raw, dones_raw, obs_shape),
                    daemon=True,
                )
                process.start()
                work_remote.close()
                self.remotes.append(remote)
                self.work_remotes.append(work_remote)
                self.processes.append(process)

            self._validate_worker_spaces(observation_space, action_space)
            super().__init__(len(env_fns), observation_space, action_space)
        except Exception:
            self.close()
            raise

    def _validate_worker_spaces(self, observation_space: spaces.Box, action_space: spaces.Box) -> None:
        for remote in self.remotes:
            remote.send(("get_spaces", None))
        for remote in self.remotes:
            for obs_space, act_space in remote.recv():
                _validate_box_float32(obs_space, "observation")
                _validate_box_float32(act_space, "action")
                if obs_space != observation_space or act_space != action_space:
                    raise ValueError("All grouped environments must have the same observation/action spaces")

    def _obs_from_shared(self) -> np.ndarray:
        return np.concatenate([np.copy(buffer) for buffer in self.shared_observations], axis=0)

    def _rewards_from_shared(self) -> np.ndarray:
        return np.concatenate([np.copy(buffer) for buffer in self.shared_rewards], axis=0)

    def _dones_from_shared(self) -> np.ndarray:
        return np.concatenate([np.copy(buffer) for buffer in self.shared_dones], axis=0)

    def _normalize_indices(self, indices: Any = None) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def _indices_by_group(self, indices: Any = None) -> list[tuple[int, list[int], list[int]]]:
        normalized = self._normalize_indices(indices)
        by_group: dict[int, tuple[list[int], list[int]]] = {}
        for output_idx, env_idx in enumerate(normalized):
            if env_idx < 0:
                env_idx = self.num_envs + env_idx
            if not 0 <= env_idx < self.num_envs:
                raise IndexError(f"Environment index out of range: {env_idx}")
            group_idx = env_idx // self.group_size
            local_idx = env_idx - self.group_slices[group_idx].start
            group_local, group_output = by_group.setdefault(group_idx, ([], []))
            group_local.append(local_idx)
            group_output.append(output_idx)
        return [(group_idx, local, output) for group_idx, (local, output) in by_group.items()]

    def reset(self) -> np.ndarray:
        for group_idx, remote in enumerate(self.remotes):
            group_slice = self.group_slices[group_idx]
            reset_args = [
                (self._seeds[env_idx], self._options[env_idx])
                for env_idx in range(group_slice.start, group_slice.stop)
            ]
            remote.send(("reset", reset_args))

        reset_infos: list[dict[str, Any]] = []
        for remote in self.remotes:
            reset_infos.extend(remote.recv())
        self.reset_infos = reset_infos
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_shared()

    def step_async(self, actions: np.ndarray) -> None:
        if self.waiting:
            raise RuntimeError("Already waiting for grouped step results")
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_envs, *self.action_space.shape)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected action batch shape {expected_shape}, got {actions.shape}")
        for group_slice, remote in zip(self.group_slices, self.remotes):
            remote.send(("step", actions[group_slice]))
        self.waiting = True

    def step_wait(self):
        infos: list[dict[str, Any]] = []
        reset_infos: list[dict[str, Any]] = []
        for remote in self.remotes:
            group_infos, group_reset_infos = remote.recv()
            infos.extend(group_infos)
            reset_infos.extend(group_reset_infos)
        self.waiting = False
        self.reset_infos = reset_infos
        return self._obs_from_shared(), self._rewards_from_shared(), self._dones_from_shared(), infos

    def close(self) -> None:
        if getattr(self, "closed", True):
            return
        if self.waiting:
            for remote in self.remotes:
                if remote.poll(1.0):
                    remote.recv()
            self.waiting = False
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        self.closed = True

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        normalized = self._normalize_indices(indices)
        results = [None] * len(normalized)
        for group_idx, local_indices, output_indices in self._indices_by_group(normalized):
            self.remotes[group_idx].send(("get_attr", (local_indices, attr_name)))
            values = self.remotes[group_idx].recv()
            for output_idx, value in zip(output_indices, values):
                results[output_idx] = value
        return results

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        for group_idx, local_indices, _output_indices in self._indices_by_group(indices):
            self.remotes[group_idx].send(("set_attr", (local_indices, attr_name, value)))
        for group_idx, _local_indices, _output_indices in self._indices_by_group(indices):
            self.remotes[group_idx].recv()

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Any = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        normalized = self._normalize_indices(indices)
        results = [None] * len(normalized)
        for group_idx, local_indices, output_indices in self._indices_by_group(normalized):
            self.remotes[group_idx].send(("env_method", (local_indices, method_name, method_args, method_kwargs)))
            values = self.remotes[group_idx].recv()
            for output_idx, value in zip(output_indices, values):
                results[output_idx] = value
        return results

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: Any = None) -> list[bool]:
        normalized = self._normalize_indices(indices)
        results = [False] * len(normalized)
        for group_idx, local_indices, output_indices in self._indices_by_group(normalized):
            self.remotes[group_idx].send(("is_wrapped", (local_indices, wrapper_class)))
            values = self.remotes[group_idx].recv()
            for output_idx, value in zip(output_indices, values):
                results[output_idx] = value
        return results
