"""CPU-native navigation simulation with CUDA policy/rollout tensors.

The wrapped ``NativeLaneEnv`` owns the C simulation, frame stack, autoreset,
terminal observations and episode records. Each step copies the action batch
D2H, then uploads the complete ``TensorStep`` H2D. Blocking copies ensure the
native runtime has received actions and cannot overwrite buffers still being
uploaded. This preserves the same TensorPPO collection path as CUDA lanes.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence


class NativeLaneCPUAdapter:
    """Expose CPU-native lanes to the CUDA-only ``TensorVecEnv`` contract."""

    def __init__(self, env: Any, *, device: Any = "cuda") -> None:
        import torch

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("NativeLaneCPUAdapter requires a CUDA policy device")
        if torch.device(env.device).type != "cpu" or env.backend.kind != "cpu":
            raise ValueError("NativeLaneCPUAdapter requires CPU-native lanes")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0)
        self.env = env
        self._torch = torch
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def seed(self, seeds: Sequence[int]) -> list[int]:
        return self.env.seed(seeds)

    def reset_tensor(self, seeds: Optional[Sequence[int]] = None) -> Any:
        return self.env.reset_tensor(seeds=seeds).to(self.device, non_blocking=False)

    def _require_cuda_actions(self, actions: Any) -> Any:
        torch = self._torch
        if not torch.is_tensor(actions) or not actions.is_cuda:
            raise ValueError("clipped_actions must be a CUDA torch.Tensor")
        if actions.device != self.device:
            raise ValueError(f"clipped_actions must be on {self.device}")
        if actions.dtype != torch.float32:
            raise ValueError("clipped_actions must be float32")
        if tuple(actions.shape) != (self.num_envs, 10):
            raise ValueError(f"clipped_actions shape must be ({self.num_envs}, 10)")
        if actions.requires_grad:
            raise ValueError("clipped_actions must not require grad")
        return actions.contiguous()

    def step_tensor(self, clipped_actions: Any) -> Any:
        from ai2.training.tensor_ppo import TensorStep

        actions = self._require_cuda_actions(clipped_actions)
        step = self.env.step_tensor(actions.detach().to("cpu", non_blocking=False))
        return TensorStep(
            observations=step.observations.to(self.device, non_blocking=False),
            rewards=step.rewards.to(self.device, non_blocking=False),
            terminated=step.terminated.to(self.device, non_blocking=False),
            truncated=step.truncated.to(self.device, non_blocking=False),
            terminal_observations=step.terminal_observations.to(self.device, non_blocking=False),
        )

    def drain_episode_infos(self) -> list[dict[str, Any]]:
        return self.env.drain_episode_infos()

    def close(self) -> None:
        self.env.close()
