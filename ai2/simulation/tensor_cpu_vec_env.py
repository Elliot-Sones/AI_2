"""Hybrid CPU-game bridge for TensorPPO.

``CPUGameTensorAdapter`` is intentionally not a GPU simulator.  It wraps an
existing SB3 VecEnv whose game, wrappers, frame stack, autoreset, and monitor
info handling still run on CPU.  Each tensor step performs one batched D2H
action copy, calls the original CPU ``VecEnv.step()``, then uploads observations,
rewards, and done flags back to CUDA.  CUDA terminal observations are uploaded
only when host ``TimeLimit.truncated`` metadata shows that timeout bootstrapping
can use them; host callback ``infos`` still preserve terminal observations.

This is a hybrid CPU-game path, not GPU simulation.
"""

from __future__ import annotations

import numbers
from typing import Any, Optional, Sequence

import numpy as np


class CPUGameTensorAdapter:
    """Expose an existing 4-frame/256-observation SB3 VecEnv as TensorVecEnv input."""

    def __init__(self, vec_env: Any, *, device: Any = "cuda") -> None:
        import torch

        self.vec_env = vec_env
        self.device = self._canonical_cuda_device(torch.device(device), torch)
        if self.device.type != "cuda":
            raise ValueError("CPUGameTensorAdapter requires CUDA")
        raw_num_envs = getattr(vec_env, "num_envs")
        if isinstance(raw_num_envs, bool) or not isinstance(raw_num_envs, numbers.Integral) or int(raw_num_envs) <= 0:
            raise ValueError("num_envs must be a strict positive integer")
        self.num_envs = int(raw_num_envs)
        self.observation_space = getattr(vec_env, "observation_space")
        self.action_space = getattr(vec_env, "action_space")
        self._torch = torch
        self._last_infos: list[dict[str, Any]] = []
        self._last_dones = np.zeros((self.num_envs,), dtype=bool)
        self._last_timeout_indices = np.empty((0,), dtype=np.int64)
        self._episode_infos: list[dict[str, Any]] = []
        self._validate_spaces()

    def _canonical_cuda_device(self, device: Any, torch: Any) -> Any:
        if device.type != "cuda" or device.index is not None:
            return device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cuda", 0)

    def _validate_spaces(self) -> None:
        if type(self.observation_space).__name__ != "Box" or type(self.action_space).__name__ != "Box":
            raise ValueError("CPUGameTensorAdapter supports Box observation/action spaces only")
        if np.dtype(self.observation_space.dtype) != np.float32 or np.dtype(self.action_space.dtype) != np.float32:
            raise ValueError("CPUGameTensorAdapter supports float32 observation/action spaces only")
        if tuple(self.observation_space.shape) != (256,):
            raise ValueError("CPUGameTensorAdapter expects existing stacked 256D observations")
        if tuple(self.action_space.shape) != (10,):
            raise ValueError("CPUGameTensorAdapter expects 10D actions")

    def _require_cuda_actions(self, actions: Any) -> Any:
        torch = self._torch
        if not torch.is_tensor(actions):
            raise ValueError("clipped_actions must be a CUDA torch.Tensor")
        if not actions.is_cuda:
            raise ValueError("clipped_actions must be CUDA")
        if actions.device != self.device:
            raise ValueError(f"clipped_actions must be on {self.device}")
        if actions.dtype != torch.float32:
            raise ValueError("clipped_actions must be float32")
        if tuple(actions.shape) != (self.num_envs, 10):
            raise ValueError(f"clipped_actions shape must be ({self.num_envs}, 10)")
        if not actions.is_contiguous():
            raise ValueError("clipped_actions must be contiguous")
        if actions.requires_grad:
            raise ValueError("clipped_actions must not requires_grad")
        return actions

    def _validate_obs_array(self, name: str, value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype != np.float32:
            raise ValueError(f"{name} must be float32")
        if tuple(array.shape) != (self.num_envs, 256):
            raise ValueError(f"{name} shape must be ({self.num_envs}, 256)")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return np.ascontiguousarray(array)

    def reset_tensor(self, seeds: Optional[Sequence[int]] = None) -> Any:
        if seeds is not None:
            self.seed(seeds)
        obs = self._validate_obs_array("reset observations", self.vec_env.reset())
        self._last_infos = []
        self._last_dones = np.zeros((self.num_envs,), dtype=bool)
        self._last_timeout_indices = np.empty((0,), dtype=np.int64)
        self._episode_infos = []
        return self._torch.as_tensor(obs, dtype=self._torch.float32, device=self.device)

    def seed(self, seeds: Sequence[int]) -> list[int]:
        if len(seeds) != self.num_envs:
            raise ValueError(f"seeds length must be {self.num_envs}")
        seed_values = [self._validate_seed(seed) for seed in seeds]
        seed_method = getattr(self.vec_env, "seed", None)
        if seed_method is None:
            raise ValueError("underlying VecEnv does not support seed(seeds)")
        if seed_values == list(range(seed_values[0], seed_values[0] + self.num_envs)):
            seed_method(seed_values[0])
        elif hasattr(self.vec_env, "_seeds"):
            self.vec_env._seeds = seed_values  # SB3 VecEnv reset consumes this slot.
        else:
            raise ValueError("underlying VecEnv only supports contiguous base seeds")
        return seed_values

    def _validate_seed(self, seed: Any) -> int:
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise ValueError("seeds must be integers")
        return int(seed)

    def step_tensor(self, clipped_actions: Any) -> Any:
        from ai2.training.tensor_ppo import TensorStep

        actions_t = self._require_cuda_actions(clipped_actions)
        actions_np = actions_t.detach().to("cpu").numpy()
        if not np.isfinite(actions_np).all():
            raise ValueError("clipped_actions must be finite")
        obs, rewards, dones, infos = self.vec_env.step(actions_np)
        infos = [dict(info) for info in infos]
        if len(infos) != self.num_envs:
            raise ValueError(f"infos length must be {self.num_envs}")
        obs_np = self._validate_obs_array("observations", obs)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=bool)
        if tuple(rewards_np.shape) != (self.num_envs,):
            raise ValueError(f"rewards shape must be ({self.num_envs},)")
        if not np.isfinite(rewards_np).all():
            raise ValueError("rewards must be finite")
        if tuple(dones_np.shape) != (self.num_envs,):
            raise ValueError(f"dones shape must be ({self.num_envs},)")

        truncated_np = np.asarray([bool(info.get("TimeLimit.truncated", False)) for info in infos], dtype=bool)
        terminated_np = dones_np & ~truncated_np
        timeout_indices = np.flatnonzero(truncated_np & ~terminated_np).astype(np.int64, copy=False)
        terminal_obs = obs_np.copy() if timeout_indices.size else None
        for idx, done in enumerate(dones_np):
            if done:
                if "terminal_observation" not in infos[idx]:
                    raise ValueError("done lanes must include terminal_observation")
                lane_terminal_obs = np.asarray(infos[idx]["terminal_observation"], dtype=np.float32)
                if tuple(lane_terminal_obs.shape) != (256,):
                    raise ValueError("terminal_observation shape must be (256,)")
                if not np.isfinite(lane_terminal_obs).all():
                    raise ValueError("terminal_observation must be finite")
                if terminal_obs is not None:
                    terminal_obs[idx] = lane_terminal_obs

        self._last_infos = [dict(info) for info in infos]
        self._last_dones = dones_np.copy()
        self._last_timeout_indices = timeout_indices.copy()
        self._episode_infos.extend([dict(info) for info in infos if info.get("episode") is not None])

        return TensorStep(
            observations=self._torch.as_tensor(obs_np, dtype=self._torch.float32, device=self.device),
            rewards=self._torch.as_tensor(rewards_np, dtype=self._torch.float32, device=self.device),
            terminated=self._torch.as_tensor(terminated_np, dtype=self._torch.bool, device=self.device),
            truncated=self._torch.as_tensor(truncated_np, dtype=self._torch.bool, device=self.device),
            terminal_observations=None
            if terminal_obs is None
            else self._torch.as_tensor(terminal_obs, dtype=self._torch.float32, device=self.device),
        )

    def host_step_infos(self) -> list[dict[str, Any]]:
        infos = [dict(info) for info in self._last_infos]
        self._episode_infos = []
        return infos

    def host_step_dones(self) -> np.ndarray:
        return self._last_dones.copy()

    def host_has_timeouts(self) -> bool:
        return bool(self._last_timeout_indices.size)

    def host_timeout_indices(self) -> np.ndarray:
        return self._last_timeout_indices.copy()

    def drain_episode_infos(self) -> list[dict[str, Any]]:
        infos = self._episode_infos
        self._episode_infos = []
        return infos

    def close(self) -> None:
        close = getattr(self.vec_env, "close", None)
        if close is not None:
            close()
