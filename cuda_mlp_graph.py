"""Reversible CUDA Graph capture for SB3's fixed-shape MLP extractor.

This helper is intentionally narrow: it targets the deterministic Linear/Tanh
``MlpExtractor`` used by the native-GPU PPO update benchmark. It captures only
the train-mode MLP extractor forward/backward path for one static minibatch
shape. Collection, evaluation, checkpointing, and mismatched inputs keep using
the original eager ``forward``.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class MlpGraphHandle:
    """Metadata for one temporary MLP graph capture."""

    capture_seconds: float
    batch_size: int
    features_dim: int


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _has_hooks(module: Any) -> bool:
    hook_attrs = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
    )
    return any(bool(getattr(module, attr, None)) for attr in hook_attrs)


def _validate_supported_mlp_extractor(mlp_extractor: Any) -> Any:
    import torch

    if not isinstance(mlp_extractor, torch.nn.Module):
        raise TypeError("mlp_extractor must be a torch.nn.Module")
    if mlp_extractor.__class__.__name__ != "MlpExtractor" or not mlp_extractor.__class__.__module__.startswith(
        "stable_baselines3."
    ):
        raise ValueError("unsupported MLP extractor: expected Stable-Baselines3 MlpExtractor")
    if not callable(getattr(mlp_extractor, "forward", None)):
        raise TypeError("mlp_extractor must expose a callable forward")
    if not hasattr(mlp_extractor, "policy_net") or not hasattr(mlp_extractor, "value_net"):
        raise ValueError("unsupported MLP extractor: expected SB3 policy_net and value_net")
    if not hasattr(mlp_extractor, "forward_actor") or not hasattr(mlp_extractor, "forward_critic"):
        raise ValueError("unsupported MLP extractor: expected actor and critic forward helpers")
    if _has_hooks(mlp_extractor):
        raise ValueError("unsupported MLP extractor: hooks are not graph-safe")

    allowed = (torch.nn.Linear, torch.nn.Tanh, torch.nn.Sequential)
    for child in mlp_extractor.modules():
        if child is mlp_extractor:
            continue
        if _has_hooks(child):
            raise ValueError("unsupported MLP extractor: hooks are not graph-safe")
        if not isinstance(child, allowed):
            raise ValueError(f"unsupported MLP extractor module: {child.__class__.__name__}")

    params = list(mlp_extractor.parameters())
    if not params:
        raise ValueError("unsupported MLP extractor: no parameters to graph")
    devices = {param.device for param in params}
    if len(devices) != 1:
        raise ValueError("MLP graph capture requires all parameters on one CUDA device")
    device = next(iter(devices))
    if device.type != "cuda":
        raise ValueError("MLP graph capture requires CUDA parameters")
    if any(param.dtype != torch.float32 for param in params):
        raise ValueError("MLP graph capture currently requires float32 parameters")
    if any(buffer.numel() for buffer in mlp_extractor.buffers()):
        raise ValueError("unsupported MLP extractor: buffers are not part of this narrow graph contract")
    return device


def _tensor_matches_sample(tensor: Any, sample: Any) -> bool:
    return (
        tensor.shape == sample.shape
        and tensor.dtype == sample.dtype
        and tensor.device == sample.device
        and tensor.requires_grad == sample.requires_grad
        and tensor.stride() == sample.stride()
    )


@contextmanager
def graph_mlp_extractor(
    mlp_extractor: Any,
    *,
    batch_size: int,
    features_dim: int,
    requires_grad: bool = False,
    num_warmup_iters: int = 3,
) -> Iterator[MlpGraphHandle]:
    """Temporarily graph an SB3 MLP extractor for one CUDA train minibatch shape.

    The context mutates ``mlp_extractor.forward`` while active and restores it on
    exit, even if capture or training raises. Inputs that do not exactly match
    the captured CUDA sample fall back to the original eager forward.
    """
    import torch

    batch = _validate_positive_int("batch_size", batch_size)
    width = _validate_positive_int("features_dim", features_dim)
    warmup = int(num_warmup_iters)
    if warmup < 0:
        raise ValueError("num_warmup_iters must be nonnegative")

    device = _validate_supported_mlp_extractor(mlp_extractor)

    original_forward = mlp_extractor.forward
    original_forward_is_instance_attr = "forward" in mlp_extractor.__dict__
    original_forward_attr = mlp_extractor.__dict__.get("forward")
    original_training = {module: bool(module.training) for module in mlp_extractor.modules()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    sample = torch.zeros((batch, width), device=device, dtype=torch.float32, requires_grad=bool(requires_grad))
    capture_seconds = 0.0

    try:
        mlp_extractor.train(True)
        try:
            torch.cuda.synchronize()
            started = time.perf_counter()
            torch.cuda.make_graphed_callables(
                mlp_extractor,
                sample_args=(sample,),
                num_warmup_iters=warmup,
            )
            torch.cuda.synchronize()
            capture_seconds = time.perf_counter() - started
            graphed_forward = mlp_extractor.forward
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)

        def guarded_forward(features: Any) -> Any:
            if (
                mlp_extractor.training
                and torch.is_grad_enabled()
                and torch.is_tensor(features)
                and _tensor_matches_sample(features, sample)
            ):
                return graphed_forward(features)
            return original_forward(features)

        mlp_extractor.forward = guarded_forward
        yield MlpGraphHandle(capture_seconds=capture_seconds, batch_size=batch, features_dim=width)
    finally:
        if original_forward_is_instance_attr:
            mlp_extractor.forward = original_forward_attr
        elif "forward" in mlp_extractor.__dict__:
            delattr(mlp_extractor, "forward")
        for module, training in original_training.items():
            module.training = training
