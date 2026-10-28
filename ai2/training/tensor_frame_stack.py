"""Four-frame game observations, resident on CUDA and SB3-reset compatible.

The returned observation aliases persistent storage and must be consumed before
the next update. Calls enqueue work on the current stream; a caller switching
streams must establish dependencies with wait_stream/event before consumption.
Terminal observations are separate snapshots, including the final pre-reset
frame. Non-terminal lanes also have a defined terminal stack, which lets a
collector evaluate terminal values in one batch and mask them on device.

Finite-value validation is accumulated on device during reset/update. Call
``check_valid()`` at an explicit rollout/control boundary to synchronize and
raise if any observation or terminal observation contained NaN/Inf.
"""
from __future__ import annotations


class TensorFrameStack:
    def __init__(self, n_envs: int, *, device="cuda"):
        if type(n_envs) is not int or not 1 <= n_envs <= 4096:
            raise ValueError("n_envs must be an integer in [1, 4096]")
        import torch

        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("TensorFrameStack requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device, self.n_envs, self._torch = device, n_envs, torch
        self._stack = torch.zeros((n_envs, 256), dtype=torch.float32, device=device)
        self._invalid_data = torch.zeros((), dtype=torch.bool, device=device)
        self._initialized = False

    def _validate(self, value, shape, dtype, name):
        torch = self._torch
        if (not isinstance(value, torch.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape
                or not value.is_contiguous() or value.requires_grad):
            raise ValueError(f"{name} must be a contiguous {dtype} CUDA tensor with shape {shape}, without gradients")
        # Keep externally owned inputs alive for work queued on this stream.
        value.record_stream(torch.cuda.current_stream(self.device))

    def _accumulate_finite_check(self, value):
        self._invalid_data.logical_or_(~self._torch.isfinite(value).all())

    def check_valid(self):
        if bool(self._invalid_data.item()):
            raise ValueError("TensorFrameStack observed nonfinite input data")

    def reset(self, observations):
        torch = self._torch
        self._validate(observations, (self.n_envs, 64), torch.float32, "observations")
        with torch.no_grad():
            self._invalid_data.zero_()
            self._accumulate_finite_check(observations)
            self._stack.zero_()
            self._stack[:, -64:].copy_(observations)
        self._initialized = True
        return self._stack

    def update(self, observations, terminal_observations, dones):
        if not self._initialized:
            raise RuntimeError("reset must be called before update")
        torch = self._torch
        self._validate(observations, (self.n_envs, 64), torch.float32, "observations")
        self._validate(terminal_observations, (self.n_envs, 64), torch.float32, "terminal_observations")
        self._validate(dones, (self.n_envs,), torch.bool, "dones")
        with torch.no_grad():
            self._accumulate_finite_check(observations)
            self._accumulate_finite_check(terminal_observations)
            # cat snapshots history before persistent storage is overwritten.
            terminal = torch.cat((self._stack[:, 64:], terminal_observations), dim=1)
            history = terminal[:, :192].masked_fill(dones[:, None], 0)
            self._stack[:, :192].copy_(history)
            self._stack[:, -64:].copy_(observations)
        return self._stack, terminal
