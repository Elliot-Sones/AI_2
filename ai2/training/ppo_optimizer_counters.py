"""Optional scalar-state placement for checkpoint-loaded Adam optimizers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def normalize_adam_step_devices(optimizer: torch.optim.Optimizer) -> int:
    """Move eligible Adam/AdamW scalar step counters to CPU; return the count.

    Call after checkpoint loading, before updates. Noncapturable, nonfused Adam
    reads these counters on the host; loading all state onto CUDA can introduce
    scalar readback synchronization at each minibatch. CPU placement leaves the
    counter dtype/value, moments, parameters, and optimizer options unchanged.

    Capturable and fused groups retain their required device-local state. Fresh
    state and existing CPU counters are left alone. All eligible counters are
    validated and copied before any state is replaced, so malformed state or a
    failed copy cannot leave a partially normalized optimizer.
    """
    import torch

    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        raise TypeError("step-device normalization requires an Adam or AdamW optimizer")

    pending = []
    for group in optimizer.param_groups:
        if group.get("capturable", False) or group.get("fused", False):
            continue
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            if "step" not in state:
                raise ValueError("initialized Adam state is missing its step counter")
            step = state["step"]
            if not isinstance(step, torch.Tensor):
                raise TypeError("Adam step counter must be a tensor")
            if step.ndim != 0:
                raise ValueError("Adam step counter must be a scalar tensor")
            if step.device.type == "meta":
                raise ValueError("Adam step counter cannot be a meta tensor")
            if step.device.type != "cpu":
                pending.append((state, step))

    replacements = [(state, step.cpu()) for state, step in pending]
    for state, step in replacements:
        state["step"] = step
    return len(replacements)
