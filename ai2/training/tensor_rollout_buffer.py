"""CUDA-resident rollout buffer compatible with SB3 PPO training samples.

This buffer is intentionally narrower than Stable-Baselines3's default
``RolloutBuffer``: it supports Box float32 observations/actions and stores the
rollout directly in preallocated CUDA tensors.  It does not subclass SB3's
NumPy-backed buffer because that parent allocates host trajectory arrays.
"""

from __future__ import annotations

from collections.abc import Generator
import math
import numbers
from typing import Any, Optional

import numpy as np


def _gae_scan(
    rewards: Any,
    episode_starts: Any,
    values: Any,
    last_values: Any,
    dones: Any,
    advantages: Any,
    returns: Any,
    gamma: float,
    gae_lambda: float,
) -> None:
    """Fixed-shape tensor-only GAE, retaining the eager operation order.

    Compilation can fuse floating-point operations, so the compiled experiment
    requires measured numerical comparison rather than an exactness claim.
    Validation and host scalar reads belong outside this function.
    """
    import torch

    buffer_size, n_envs = rewards.shape
    last_gae_lam = torch.zeros((n_envs,), dtype=torch.float32, device=rewards.device)
    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - dones
            next_values = last_values
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[step].copy_(last_gae_lam)
    returns.copy_(advantages + values)


def _compile_gae_scan() -> Any:
    """Initialize Dynamo without changing global distribution validation.

    The first compiler import in Torch 2.4.1 disables that global setting.
    Restoring it is necessary even when compiler initialization raises.
    """
    import torch
    from torch.distributions import Distribution

    validate_args = Distribution._validate_args
    try:
        return torch.compile(_gae_scan, fullgraph=True, dynamic=False)
    finally:
        Distribution.set_default_validate_args(validate_args)


class TensorRolloutBuffer:
    """Device-resident PPO rollout storage for CUDA/native collection paths.

    ``validation_mode="step"`` retains the original input-time numerical error
    latch. The optional ``"rollout"`` mode checks all stored inputs before GAE
    and each training epoch, plus computed targets before sampling. Compiled
    GAE also checks computed targets before sampling. Metadata checks still run for every add. Repairing
    an invalid stored value before its first boundary check is consequently
    visible only in rollout mode. Once observed, errors remain latched until
    reset in both modes, including explicit timeout reward checks.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: Any,
        action_space: Any,
        device: Any,
        gae_lambda: float,
        gamma: float,
        n_envs: int = 1,
        validation_mode: str = "step",
        compile_gae: bool = False,
        **_: Any,
    ) -> None:
        import torch

        self.torch = torch
        self.buffer_size = self._strict_positive_int("buffer_size", buffer_size)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = self._canonical_cuda_device(torch.device(device))
        self.gae_lambda = self._probability_scalar("gae_lambda", gae_lambda)
        self.gamma = self._probability_scalar("gamma", gamma)
        self.n_envs = self._strict_positive_int("n_envs", n_envs)
        self.validation_mode = validation_mode
        self.compile_gae = compile_gae
        self._compiled_gae = None
        self.pos = 0
        self.full = False
        self.generator_ready = False
        self._computed_returns_and_advantage = False
        self._flat_tensors: dict[str, Any] = {}

        self._validate_constructor()
        self.obs_shape = tuple(int(dim) for dim in observation_space.shape)
        self.action_dim = int(np.prod(action_space.shape))

        self.observations = torch.empty(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.empty(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.rewards = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self.episode_starts = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self.values = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self.log_probs = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self.advantages = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self.returns = torch.empty((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self._invalid_data = torch.zeros((), dtype=torch.bool, device=self.device)

    @property
    def validation_mode(self) -> str:
        return self._validation_mode

    @validation_mode.setter
    def validation_mode(self, value: str) -> None:
        if value not in ("step", "rollout"):
            raise ValueError("validation_mode must be 'step' or 'rollout'")
        self._validation_mode = value

    @property
    def compile_gae(self) -> bool:
        return self._compile_gae

    @compile_gae.setter
    def compile_gae(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("compile_gae must be a bool")
        self._compile_gae = value

    def _strict_positive_int(self, name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"{name} must be a strict positive integer")
        value_i = int(value)
        if value_i <= 0:
            raise ValueError(f"{name} must be positive")
        return value_i

    def _probability_scalar(self, name: str, value: Any) -> float:
        value_f = float(value)
        if not math.isfinite(value_f) or value_f < 0.0 or value_f > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return value_f

    def _canonical_cuda_device(self, device: Any) -> Any:
        torch = self.torch
        if device.type != "cuda":
            return device
        if device.index is not None:
            return device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cuda", 0)

    def _validate_constructor(self) -> None:
        if self.device.type != "cuda":
            raise ValueError("TensorRolloutBuffer is CUDA-only; no CPU fallback")
        if type(self.observation_space).__name__ != "Box" or type(self.action_space).__name__ != "Box":
            raise ValueError("TensorRolloutBuffer supports Box observation/action spaces only")
        if np.dtype(self.observation_space.dtype) != np.float32 or np.dtype(self.action_space.dtype) != np.float32:
            raise ValueError("TensorRolloutBuffer supports float32 observation/action spaces only")
        if not tuple(getattr(self.observation_space, "shape", ())):
            raise ValueError("observation_space must have a non-empty shape")
        if not tuple(getattr(self.action_space, "shape", ())):
            raise ValueError("action_space must have a non-empty shape")

    def reset(self) -> None:
        """Mark the existing device storage ready for the next rollout."""

        self.pos = 0
        self.full = False
        self.generator_ready = False
        self._computed_returns_and_advantage = False
        self._flat_tensors = {}
        self._invalid_data.zero_()

    def _require_cuda_float32(self, name: str, tensor: Any, shape: tuple[int, ...]) -> Any:
        torch = self.torch
        if not torch.is_tensor(tensor):
            raise ValueError(f"{name} must be a CUDA torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA; no CPU fallback")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.requires_grad:
            raise ValueError(f"{name} must not requires_grad")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        return tensor

    def _require_vector_or_column(self, name: str, tensor: Any) -> Any:
        torch = self.torch
        if not torch.is_tensor(tensor):
            raise ValueError(f"{name} must be a CUDA torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA; no CPU fallback")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.requires_grad:
            raise ValueError(f"{name} must not requires_grad")
        if tuple(tensor.shape) == (self.n_envs, 1):
            tensor = tensor.reshape(self.n_envs)
        elif tuple(tensor.shape) != (self.n_envs,):
            raise ValueError(f"{name} shape must be ({self.n_envs},) or ({self.n_envs}, 1), got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor

    def _accumulate_finite_check(self, tensor: Any) -> None:
        self._invalid_data.logical_or_(~self.torch.isfinite(tensor).all())

    def _accumulate_binary_check(self, tensor: Any) -> None:
        self._invalid_data.logical_or_(~((tensor == 0.0) | (tensor == 1.0)).all())

    def _validate_rollout_data(self) -> None:
        """Accumulate checks over stored snapshots without clearing prior errors."""
        for tensor in (
            self.observations, self.actions, self.rewards,
            self.episode_starts, self.values, self.log_probs,
        ):
            self._accumulate_finite_check(tensor)
        self._accumulate_binary_check(self.episode_starts)

    def _validate_training_targets(self) -> None:
        self._accumulate_finite_check(self.advantages)
        self._accumulate_finite_check(self.returns)

    def _raise_if_invalid_data(self) -> None:
        if bool(self._invalid_data.item()):
            raise ValueError("rollout buffer contains nonfinite or non-binary data")

    def add(
        self,
        obs: Any,
        action: Any,
        reward: Any,
        episode_start: Any,
        value: Any,
        log_prob: Any,
    ) -> None:
        if self.full or self.pos >= self.buffer_size:
            raise AssertionError("rollout buffer is full; call reset() before adding")
        obs_t = self._require_cuda_float32("obs", obs, (self.n_envs, *self.obs_shape))
        action_t = self._require_cuda_float32("action", action, (self.n_envs, self.action_dim))
        reward_t = self._require_vector_or_column("reward", reward)
        episode_start_t = self._require_vector_or_column("episode_start", episode_start)
        value_t = self._require_vector_or_column("value", value)
        log_prob_t = self._require_vector_or_column("log_prob", log_prob)
        if self.validation_mode == "step":
            self._accumulate_finite_check(obs_t)
            self._accumulate_finite_check(action_t)
            self._accumulate_finite_check(reward_t)
            self._accumulate_finite_check(episode_start_t)
            self._accumulate_finite_check(value_t)
            self._accumulate_finite_check(log_prob_t)
            self._accumulate_binary_check(episode_start_t)

        self.observations[self.pos].copy_(obs_t)
        self.actions[self.pos].copy_(action_t)
        self.rewards[self.pos].copy_(reward_t)
        self.episode_starts[self.pos].copy_(episode_start_t)
        self.values[self.pos].copy_(value_t)
        self.log_probs[self.pos].copy_(log_prob_t)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        self.generator_ready = False
        self._computed_returns_and_advantage = False
        self._flat_tensors = {}

    def compute_returns_and_advantage(self, last_values: Any, dones: Any) -> None:
        if not self.full:
            raise RuntimeError("rollout buffer must be full before computing returns")
        torch = self.torch
        if self.validation_mode == "rollout":
            self._validate_rollout_data()
        self._raise_if_invalid_data()
        last_values_t = self._require_vector_or_column("last_values", last_values)
        self._accumulate_finite_check(last_values_t)
        if not torch.is_tensor(dones):
            raise ValueError("dones must be a CUDA torch.Tensor")
        if not dones.is_cuda:
            raise ValueError("dones must be CUDA; no CPU fallback")
        if dones.device != self.device:
            raise ValueError(f"dones must be on {self.device}")
        if getattr(dones, "requires_grad", False):
            raise ValueError("dones must not require_grad")
        if tuple(dones.shape) == (self.n_envs, 1):
            dones_t = dones.reshape(self.n_envs).to(dtype=torch.float32)
        elif tuple(dones.shape) == (self.n_envs,):
            dones_t = dones.to(dtype=torch.float32)
        else:
            raise ValueError(f"dones shape must be ({self.n_envs},) or ({self.n_envs}, 1), got {tuple(dones.shape)}")
        if dones_t.dtype != torch.float32:
            dones_t = dones_t.float()
        self._accumulate_finite_check(dones_t)
        self._accumulate_binary_check(dones_t)
        self._raise_if_invalid_data()

        scan = _gae_scan
        if self.compile_gae:
            if self._compiled_gae is None:
                self._compiled_gae = _compile_gae_scan()
            scan = self._compiled_gae
        scan(
            self.rewards, self.episode_starts, self.values,
            last_values_t, dones_t, self.advantages, self.returns,
            self.gamma, self.gae_lambda,
        )
        self.generator_ready = False
        self._computed_returns_and_advantage = True
        self._flat_tensors = {}

    def _prepare_flattened_rollout(self) -> None:
        if self.generator_ready:
            return
        self._flat_tensors = {
            "observations": self._env_major_flatten(self.observations),
            "actions": self._env_major_flatten(self.actions),
            "values": self._env_major_flatten(self.values),
            "log_probs": self._env_major_flatten(self.log_probs),
            "advantages": self._env_major_flatten(self.advantages),
            "returns": self._env_major_flatten(self.returns),
        }
        self.generator_ready = True

    def _env_major_flatten(self, tensor: Any) -> Any:
        return tensor.swapaxes(0, 1).reshape(self.buffer_size * self.n_envs, *tensor.shape[2:])

    def get(self, batch_size: Optional[int] = None) -> Generator[Any, None, None]:
        from stable_baselines3.common.buffers import RolloutBufferSamples

        if not self.full:
            raise RuntimeError("rollout buffer must be full before sampling")
        if not self._computed_returns_and_advantage:
            raise RuntimeError("rollout returns and advantages must be computed before sampling")
        if self.validation_mode == "rollout":
            self._validate_rollout_data()
        if self.validation_mode == "rollout" or self.compile_gae:
            self._validate_training_targets()
        self._raise_if_invalid_data()
        total = self.buffer_size * self.n_envs
        indices = np.random.permutation(total)
        self._prepare_flattened_rollout()
        if batch_size is None:
            batch_size = total
        batch_size = self._strict_positive_int("batch_size", batch_size)

        indices_t = self.torch.as_tensor(indices, dtype=self.torch.long, device=self.device)
        start_idx = 0
        while start_idx < total:
            batch_inds = indices_t[start_idx : start_idx + batch_size]
            yield self._get_samples_from_device_indices(batch_inds, RolloutBufferSamples)
            start_idx += batch_size

    def _get_samples_from_device_indices(self, batch_inds: Any, samples_cls: Any) -> Any:
        data = (
            self._flat_tensors["observations"].index_select(0, batch_inds),
            self._flat_tensors["actions"].index_select(0, batch_inds),
            self._flat_tensors["values"].index_select(0, batch_inds).flatten(),
            self._flat_tensors["log_probs"].index_select(0, batch_inds).flatten(),
            self._flat_tensors["advantages"].index_select(0, batch_inds).flatten(),
            self._flat_tensors["returns"].index_select(0, batch_inds).flatten(),
        )
        return samples_cls(*data)
