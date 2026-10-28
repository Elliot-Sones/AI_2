"""Optional device-resident sampling cache for SB3 PPO rollout buffers."""

from __future__ import annotations

from collections.abc import Generator
from typing import Optional

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class CachedRolloutBuffer(RolloutBuffer):
    """RolloutBuffer variant that uploads rollout arrays once per PPO update.

    SB3's default ``RolloutBuffer.get()`` preserves a flattened host rollout and
    converts every sampled minibatch with ``to_torch``. This subclass keeps the
    same NumPy permutation semantics, but caches the six flattened rollout
    tensors on ``self.device`` for the duration of a ``PPO.train()`` call.
    """

    tensor_names = (
        "observations",
        "actions",
        "values",
        "log_probs",
        "advantages",
        "returns",
    )

    def __init__(self, *args, **kwargs) -> None:
        self.cached_tensors: dict[str, th.Tensor] = {}
        self.cache_valid = False
        self.cache_upload_counts = dict.fromkeys(self.tensor_names, 0)
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.clear_cache()

    def clear_cache(self) -> None:
        """Drop device-resident rollout tensors without unflattening host data."""

        self.cached_tensors.clear()
        self.cache_valid = False

    def _prepare_flattened_rollout(self) -> None:
        if self.generator_ready:
            return
        for tensor_name in self.tensor_names:
            self.__dict__[tensor_name] = self.swap_and_flatten(self.__dict__[tensor_name])
        self.generator_ready = True

    def _upload_cache(self) -> None:
        self._prepare_flattened_rollout()
        self.cached_tensors = {}
        for tensor_name in self.tensor_names:
            tensor = th.as_tensor(self.__dict__[tensor_name], device=self.device)
            if tensor.device.type == "cpu":
                tensor = tensor.clone()
            self.cached_tensors[tensor_name] = tensor
            self.cache_upload_counts[tensor_name] += 1
        self.cache_valid = True

    def _ensure_cache(self) -> None:
        if not self.cache_valid:
            self._upload_cache()

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        self._ensure_cache()
        indices_t = th.as_tensor(indices, device=self.device, dtype=th.long)

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples_from_device_indices(indices_t[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        self._ensure_cache()
        batch_inds_t = th.as_tensor(batch_inds, device=self.device, dtype=th.long)
        return self._get_samples_from_device_indices(batch_inds_t)

    def _get_samples_from_device_indices(self, batch_inds_t: th.Tensor) -> RolloutBufferSamples:
        data = (
            self.cached_tensors["observations"].index_select(0, batch_inds_t),
            self.cached_tensors["actions"].index_select(0, batch_inds_t),
            self.cached_tensors["values"].index_select(0, batch_inds_t).flatten(),
            self.cached_tensors["log_probs"].index_select(0, batch_inds_t).flatten(),
            self.cached_tensors["advantages"].index_select(0, batch_inds_t).flatten(),
            self.cached_tensors["returns"].index_select(0, batch_inds_t).flatten(),
        )
        return RolloutBufferSamples(*data)
