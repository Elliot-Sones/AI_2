"""Full-game lanes for TensorPPO; two learner slots share each game.

One environment tick advances BOTH players. Report game ticks/s separately
from player transitions/s (two transitions per game tick). This raw reward
adapter is for infrastructure validation, not a fighting curriculum claim.
"""

from __future__ import annotations

from contextlib import nullcontext
import ctypes as C
import hashlib
import numbers
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

import numpy as np

from native_game_episode import GameReference, ROOT, D, INT32
from native_lane_vec_env import NativeLaneEnv, frozen_spaces


class EpisodeRecord(C.Structure):
    _fields_ = [
        (n, INT32)
        for n in (
            "lane",
            "length",
            "terminated",
            "truncated",
            "win_mask",
            "episode_index",
        )
    ] + [
        ("stocks", INT32 * 2),
        ("returns", D * 2),
        ("damage_done", D * 2),
        ("damage_taken", D * 2),
    ]


def compile_lanes(kind: str, build_dir: Path | None = None) -> Path:
    if kind not in {"cpu", "cuda"}:
        raise ValueError("kind must be cpu or cuda")
    folder = build_dir or ROOT / ".native-build"
    folder.mkdir(parents=True, exist_ok=True)
    from native_navigation_contact import default_contraction

    contract = default_contraction() if kind == "cpu" else "off"
    if kind == "cpu":
        requested_compiler = os.environ.get("CC", "cc")
        compiler = shutil.which(requested_compiler) or requested_compiler
        compiled_sources = [ROOT / "native" / name for name in (
            "game_lanes.c", "game_episode.c", "navigation_episode.c")]
        command = [
            compiler,
            "-std=c99",
            "-O2",
            f"-ffp-contract={contract}",
            "-fno-fast-math",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            *(str(path) for path in compiled_sources),
            "-lm",
        ]
    else:
        compiler = shutil.which("nvcc")
        if not compiler:
            raise RuntimeError(
                "nvcc unavailable: no CPU fallback for CUDA lane compilation"
            )
        compiled_sources = [ROOT / "native/game_lanes.cu"]
        command = [
            compiler,
            "-std=c++17",
            "-O2",
            "--fmad=false",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            "-Xcompiler",
            "-ffp-contract=off",
            *(str(path) for path in compiled_sources),
        ]
    identity = subprocess.run([compiler, "--version"], check=True, capture_output=True, text=True)
    h = hashlib.sha256(repr((kind, contract, platform.platform(), command, identity.stdout, identity.stderr)).encode())
    for path in sorted((ROOT / "native").glob("*.h")) + compiled_sources:
        h.update(path.name.encode() + b"\0" + path.read_bytes())
    suffix = ".dylib" if kind == "cpu" and platform.system() == "Darwin" else ".so"
    output = folder / f"libgame-lanes-{kind}-{h.hexdigest()[:20]}{suffix}"
    if output.is_file():
        return output
    with tempfile.TemporaryDirectory(
        prefix="game-lanes-build-", dir=folder
    ) as temporary:
        candidate = Path(temporary) / output.name
        subprocess.run(command + ["-o", str(candidate)], check=True)
        os.replace(candidate, output)
    return output


class GameLaneBackend:
    """Validate tensor buffers before passing raw addresses to C/CUDA."""

    def __init__(self, games: int, device, record_capacity: int | None = None):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("game lanes require cpu reference or cuda")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA unavailable; policy and lanes will not silently fall back"
                )
            self.device = torch.device(
                "cuda",
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device(),
            )
        self.kind = self.device.type
        self.games, self.lanes = games, games * 2
        self.record_capacity = int(
            record_capacity if record_capacity is not None else max(64, games * 8)
        )
        if self.record_capacity < 1:
            raise ValueError("record_capacity must be positive")
        self.reference = GameReference()
        self.library = compile_lanes(self.kind)
        self.lib = C.CDLL(str(self.library))
        prefix = f"ai2g_lanes_{self.kind}_"
        self.create = getattr(self.lib, prefix + "create")
        self.create.restype = C.c_void_p
        self.create.argtypes = [
            C.c_void_p,
            C.c_void_p,
            INT32,
            INT32,
            D,
            C.POINTER(INT32),
        ]
        self.destroy = getattr(self.lib, prefix + "destroy")
        self.destroy.restype = INT32
        self.destroy.argtypes = [C.c_void_p]
        self.reset_fn = getattr(self.lib, prefix + "reset")
        self.reset_fn.restype = INT32
        self.reset_fn.argtypes = [C.c_void_p] * (5 if self.kind == "cuda" else 4)
        self.step_fn = getattr(self.lib, prefix + "step")
        self.step_fn.restype = INT32
        self.step_fn.argtypes = [C.c_void_p] * (8 if self.kind == "cuda" else 7)
        self.drain_fn = getattr(self.lib, prefix + "drain")
        self.drain_fn.restype = INT32
        self.drain_fn.argtypes = [
            C.c_void_p,
            C.c_void_p,
            INT32,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
        ] + ([C.c_void_p] if self.kind == "cuda" else [])
        size_fn = getattr(self.lib, prefix + "record_size")
        size_fn.restype = C.c_size_t
        if size_fn() != C.sizeof(EpisodeRecord):
            raise RuntimeError("episode-record native/Python ABI mismatch")
        self.records = (EpisodeRecord * self.record_capacity)()
        self.errors = np.zeros(games, dtype=np.int32)
        self.valid_steps = np.zeros(games, dtype=np.uint64)
        self._terminated = torch.zeros(games, dtype=torch.uint8, device=self.device)
        self._truncated = torch.zeros_like(self._terminated)
        error = INT32()
        with self._guard():
            self.handle = self.create(
                C.byref(self.reference.cfg),
                C.byref(self.reference.table),
                games,
                self.record_capacity,
                48.0,
                C.byref(error),
            )
        if not self.handle or error.value:
            raise RuntimeError(
                f"native full-game context creation failed: {error.value}"
            )

    def _guard(self):
        return (
            self.torch.cuda.device(self.device)
            if self.kind == "cuda"
            else nullcontext()
        )

    def _args(self, *args):
        if not self.handle:
            raise RuntimeError("full-game lane context is closed")
        return [self.handle, *args] + (
            [C.c_void_p(self.torch.cuda.current_stream(self.device).cuda_stream)]
            if self.kind == "cuda"
            else []
        )

    @staticmethod
    def _check(error, where):
        if error:
            raise RuntimeError(f"full-game lanes {where}: error {error}")

    def _ptr(self, tensor, shape, dtype):
        if (
            not self.torch.is_tensor(tensor)
            or tensor.device != self.device
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ValueError(
                f"expected contiguous {dtype} {shape} on {self.device}, without gradients"
            )
        return C.c_void_p(tensor.data_ptr())

    def reset(self, seeds, obs):
        seeds = np.asarray(seeds)
        if seeds.shape != (self.lanes,):
            raise ValueError("need one seed per player slot")
        # Each match has one random stream; the first slot selects its seed.
        game_seeds = np.ascontiguousarray(seeds[::2], dtype=np.uint64)
        op = self._ptr(obs, (self.lanes, 64), self.torch.float32)
        with self._guard():
            self._check(
                self.reset_fn(
                    *self._args(C.c_void_p(game_seeds.ctypes.data), None, op)
                ),
                "reset",
            )
        self.errors.fill(0)

    def step(self, actions, obs, terminal, rewards, terminated, truncated):
        t = self.torch
        pointers = [
            self._ptr(actions, (self.lanes, 10), t.float32),
            self._ptr(obs, (self.lanes, 64), t.float32),
            self._ptr(terminal, (self.lanes, 64), t.float32),
            self._ptr(rewards, (self.lanes,), t.float32),
            C.c_void_p(self._terminated.data_ptr()),
            C.c_void_p(self._truncated.data_ptr()),
        ]
        self._ptr(terminated, (self.lanes,), t.bool)
        self._ptr(truncated, (self.lanes,), t.bool)
        with self._guard():
            if self.kind == "cuda":
                # C launches bypass PyTorch's allocator. Track side-stream use
                # so temporary input/output storage cannot be reused too early.
                stream = t.cuda.current_stream(self.device)
                for tensor in (actions, obs, terminal, rewards, terminated, truncated,
                               self._terminated, self._truncated):
                    tensor.record_stream(stream)
            self._check(self.step_fn(*self._args(*pointers)), "step")
            terminated.copy_(self._terminated[:, None].expand(-1, 2).reshape(-1))
            truncated.copy_(self._truncated[:, None].expand(-1, 2).reshape(-1))

    def drain(self):
        count, overflow = INT32(), INT32()
        with self._guard():
            self._check(
                self.drain_fn(
                    *self._args(
                        self.records,
                        self.record_capacity,
                        C.byref(count),
                        C.byref(overflow),
                        C.c_void_p(self.errors.ctypes.data),
                        C.c_void_p(self.valid_steps.ctypes.data),
                    )
                ),
                "drain",
            )
        if self.errors.any() or overflow.value:
            raise RuntimeError(
                f"invalid rollout: lane_errors={self.errors.tolist()}, record_overflow={overflow.value}"
            )
        return list(self.records[: count.value]), 0

    def error(self):
        # CUDA errors are checked by drain(), once per rollout. No host polling
        # on the production step path; CPU step() reports errors immediately.
        bad = np.flatnonzero(self.errors)
        return (int(self.errors[bad[0]]), int(bad[0])) if len(bad) else (0, -1)

    def close(self):
        if self.handle:
            with self._guard():
                handle, self.handle = self.handle, None
                # Native destroy can free successfully and still report a
                # preceding asynchronous CUDA error. Never retain that pointer.
                self._check(self.destroy(handle), "destroy")


class NativeGameEnv(NativeLaneEnv):
    """Reuse the verified four-frame stacking, not the navigation simulator."""

    def drain_episode_infos(self):
        records, _ = self.backend.drain()
        now = time.time()
        infos = []
        for record in records:
            for player in range(2):
                infos.append(
                    {
                        "episode": {
                            "r": round(record.returns[player], 6),
                            "l": record.length,
                            "t": now - self._t_start,
                        },
                        "combat": {
                            "game": record.lane,
                            "player": player,
                            "episode_index": record.episode_index,
                            "damage_done": record.damage_done[player],
                            "damage_taken": record.damage_taken[player],
                            "stocks": record.stocks[player],
                            "win_mask": record.win_mask,
                        },
                    }
                )
        return infos


def build_game_env(games: int, *, device="cuda", seed=0, record_capacity=None):
    if isinstance(games, bool) or not isinstance(games, numbers.Integral) or games <= 0:
        raise ValueError("games must be a positive integer")
    backend = GameLaneBackend(int(games), device, record_capacity)
    try:
        obs, action = frozen_spaces()
        return NativeGameEnv(
            backend, observation_space=obs, action_space=action, seed=seed
        )
    except BaseException:
        backend.close()
        raise
