"""Opt-in shared C/CUDA observation packing; not a simulator or speed claim.

The input schema is documented in native/navigation_observation.h. Caller owns
support detection, state attributes (missing attributes supplied as zero), held
weapon mapping and ordered active-spawner facts. CUDA returns device error flags
without synchronizing; consumers MUST check flags before accepting a rollout.
Invalid lanes contain NaNs and cannot silently become training observations.
Player rows must be ordered controlled player then opponent (swap for agent 1).
Outputs are produced on the current stream. A consumer on another CUDA stream
must wait on that producing stream before reading either output or error flags.
"""
from __future__ import annotations

from ai2.paths import REPO_ROOT as PROJECT_ROOT
import ctypes
import os
from pathlib import Path
import platform
import shutil
import subprocess
import numpy as np

ROOT = PROJECT_ROOT


def _shapes(p, s, platform):
    n = p.shape[0] if p.ndim == 3 else 0
    if not 1 <= n <= 4096 or tuple(p.shape) != (n, 2, 16) or tuple(s.shape) != (n, 4, 3) or tuple(platform.shape) != (n, 4):
        raise ValueError("expected players[N,2,16], spawners[N,4,3], platform[N,4]; N in 1..4096")
    return n


class ObservationReference:
    def __init__(self, library: Path, *, cuda: bool):
        self.cuda = cuda
        self.library = ctypes.CDLL(str(library))
        self.fn = getattr(self.library, "ai2_observation_cuda" if cuda else "ai2_observation_cpu")
        self.fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_size_t] + ([ctypes.c_void_p] if cuda else [])
        self.fn.restype = ctypes.c_int

    def pack_numpy(self, players, spawners, platform):
        if self.cuda:
            raise ValueError("CUDA reference requires pack_cuda")
        for value in (players, spawners, platform):
            if not isinstance(value, np.ndarray) or value.dtype != np.float64 or not value.flags.c_contiguous:
                raise ValueError("inputs must be contiguous float64 NumPy arrays")
        n = _shapes(players, spawners, platform)
        out = np.empty((n, 64), dtype=np.float32)
        errors = np.empty(n, dtype=np.int32)
        rc = self.fn(*(ctypes.c_void_p(a.ctypes.data) for a in (players, spawners, platform, out, errors)), n)
        if rc:
            raise RuntimeError(f"observation C call failed: {rc}")
        bad = np.flatnonzero(errors)
        if len(bad):
            raise ValueError(f"invalid observation input in lane {bad[0]}: {errors[bad[0]]}")
        return out

    def pack_cuda(self, players, spawners, platform):
        import torch
        if not self.cuda:
            raise ValueError("CPU reference requires pack_numpy")
        for value in (players, spawners, platform):
            if not torch.is_tensor(value) or not value.is_cuda or value.dtype != torch.float64 or not value.is_contiguous() or value.requires_grad:
                raise ValueError("inputs must be contiguous non-differentiable CUDA float64 tensors")
            if value.device != players.device:
                raise ValueError("all inputs must use the same CUDA device")
        n = _shapes(players, spawners, platform)
        with torch.cuda.device(players.device):
            stream = torch.cuda.current_stream(players.device)
            out = torch.empty((n, 64), dtype=torch.float32, device=players.device)
            errors = torch.empty(n, dtype=torch.int32, device=players.device)
            for value in (players, spawners, platform):
                value.record_stream(stream)
            rc = self.fn(*(ctypes.c_void_p(a.data_ptr()) for a in (players, spawners, platform, out, errors)), n, ctypes.c_void_p(stream.cuda_stream))
        if rc:
            raise RuntimeError(f"observation CUDA launch failed: {rc}")
        return out, errors


def load_reference(build_dir: Path, *, cuda=False):
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("nvcc" if cuda else os.environ.get("CC", "cc"))
    if not compiler:
        raise RuntimeError("required native compiler is unavailable")
    source = ROOT / "native" / ("navigation_observation.cu" if cuda else "navigation_observation.c")
    output = build_dir / ("observation_cuda.so" if cuda else "observation_cpu.dylib" if platform.system() == "Darwin" else "observation_cpu.so")
    if cuda:
        flags = ["-O3", "--fmad=false", "--shared", "-Xcompiler", "-fPIC"]
    else:
        flags = ["-O3", "-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-ffp-contract=off", "-fPIC", "-dynamiclib" if platform.system() == "Darwin" else "-shared"]
    subprocess.run([compiler, *flags, str(source), "-o", str(output), *([] if cuda else ["-lm"])], check=True, capture_output=True, text=True, timeout=120)
    return ObservationReference(output, cuda=cuda)
