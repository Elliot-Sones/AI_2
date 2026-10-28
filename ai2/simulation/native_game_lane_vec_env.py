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

from ai2.simulation.native_game_episode import GameReference, ROOT, D, INT32
import ai2.simulation.native_navigation_episode as nav
from ai2.simulation.native_lane_vec_env import NativeLaneEnv, frozen_spaces

AI2CR_TERMS = 14


class RewardEpisodeRecord(C.Structure):
    _fields_ = [
        (n, INT32)
        for n in (
            "version",
            "learner_slot",
            "length",
            "terminated",
            "truncated",
            "win_mask",
        )
    ] + [
        ("stocks", INT32 * 2),
        ("returns", D * 2),
        ("term_totals", (D * AI2CR_TERMS) * 2),
        ("damage_done", D * 2),
        ("damage_taken", D * 2),
        ("clean_win", INT32 * 2),
        ("timeout", INT32),
    ]


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
        ("reward", RewardEpisodeRecord),
    ]


def compile_lanes(kind: str, build_dir: Path | None = None) -> Path:
    if kind not in {"cpu", "cuda"}:
        raise ValueError("kind must be cpu or cuda")
    folder = build_dir or ROOT / ".native-build"
    folder.mkdir(parents=True, exist_ok=True)
    from ai2.simulation.native_navigation_contact import default_contraction

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
        self.replay_reset_fn = getattr(self.lib, prefix + "replay_reset")
        self.replay_reset_fn.restype = INT32
        self.replay_reset_fn.argtypes = [
            C.c_void_p,
            INT32,
            C.c_void_p,
            INT32,
            C.c_void_p,
        ] + ([C.c_void_p] if self.kind == "cuda" else [])
        self.configure_episode_fn = getattr(self.lib, prefix + "configure_episode")
        self.configure_episode_fn.restype = INT32
        self.configure_episode_fn.argtypes = [
            C.c_void_p,
            INT32,
            INT32,
            INT32,
            INT32,
        ] + ([C.c_void_p] if self.kind == "cuda" else [])
        self.configure_reward_fn = getattr(self.lib, prefix + "configure_reward")
        self.configure_reward_fn.restype = INT32
        self.configure_reward_fn.argtypes = [
            C.c_void_p,
            INT32,
            C.c_uint32,
            D,
            C.c_void_p,
        ] + ([C.c_void_p] if self.kind == "cuda" else [])
        self.step_fn = getattr(self.lib, prefix + "step")
        self.step_fn.restype = INT32
        self.step_fn.argtypes = [C.c_void_p] * (8 if self.kind == "cuda" else 7)
        self.replay_step_fn = getattr(self.lib, prefix + "replay_step")
        self.replay_step_fn.restype = INT32
        self.replay_step_fn.argtypes = [
            C.c_void_p,
            INT32,
            C.c_void_p,
            C.c_void_p,
            INT32,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
        ] + ([C.c_void_p] if self.kind == "cuda" else [])
        self.rng_status_fn = getattr(self.lib, prefix + "rng_status")
        self.rng_status_fn.restype = INT32
        self.rng_status_fn.argtypes = [C.c_void_p, INT32] + [C.c_void_p] * 4 + (
            [C.c_void_p] if self.kind == "cuda" else []
        )
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

    def _draw_array(self, draws):
        rows = []
        for draw in draws:
            if isinstance(draw, dict):
                row = (draw["kind"], draw.get("a", 0.0), draw.get("b", 0.0), draw["value"])
            else:
                row = draw
            if len(row) == 5:
                kind, _pad, a, b, value = row
            else:
                kind, a, b, value = row
            if any(not np.isfinite(float(value)) for value in (a, b, value)):
                raise ValueError("replay draws must be finite")
            rows.append(nav.Draw(int(kind), 0, float(a), float(b), float(value)))
        return (nav.Draw * len(rows))(*rows)

    def _game_index(self, index):
        if type(index) is not int or index < 0 or index >= self.games:
            raise ValueError("game index out of range")
        return index

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

    def replay_reset(self, game_index, draws, obs):
        game_index = self._game_index(game_index)
        draw_array = self._draw_array(draws)
        op = self._ptr(obs, (2, 64), self.torch.float32)
        with self._guard():
            if self.kind == "cuda":
                self.torch.cuda.current_stream(self.device)
                obs.record_stream(self.torch.cuda.current_stream(self.device))
            self._check(
                self.replay_reset_fn(
                    *self._args(
                        game_index,
                        C.c_void_p(C.addressof(draw_array)) if len(draw_array) else C.c_void_p(),
                        len(draw_array),
                        op,
                    )
                ),
                "replay_reset",
            )

    def configure_episode(self, game_index, *, stocks=None, max_timesteps=None):
        game_index = self._game_index(game_index)
        if stocks is None:
            player0_stocks, player1_stocks = -1, -1
        else:
            if (
                not isinstance(stocks, (list, tuple))
                or len(stocks) != 2
                or any(type(value) is not int or value < -1 for value in stocks)
            ):
                raise ValueError("stocks must be two integers >= -1")
            player0_stocks, player1_stocks = stocks
        if max_timesteps is None:
            limit = -1
        elif type(max_timesteps) is not int or max_timesteps <= 0:
            raise ValueError("max_timesteps must be a positive integer")
        else:
            limit = max_timesteps
        with self._guard():
            self._check(
                self.configure_episode_fn(
                    *self._args(game_index, player0_stocks, player1_stocks, limit)
                ),
                "configure_episode",
            )

    def configure_combat_reward(self, terms, enabled_mask, *, fade=1.0, learner_slot=0):
        values = np.ascontiguousarray(terms, dtype=np.float64)
        if values.shape != (AI2CR_TERMS,):
            raise ValueError(f"terms must contain {AI2CR_TERMS} coefficients")
        if type(learner_slot) is not int or learner_slot not in (0, 1):
            raise ValueError("learner_slot must be 0 or 1")
        if (
            not isinstance(enabled_mask, numbers.Integral)
            or enabled_mask < 0
            or enabled_mask > (1 << AI2CR_TERMS) - 1
        ):
            raise ValueError("enabled_mask must be a valid combat-reward term mask")
        if not np.isfinite(float(fade)):
            raise ValueError("fade must be finite")
        with self._guard():
            self._check(
                self.configure_reward_fn(
                    *self._args(
                        learner_slot,
                        int(enabled_mask),
                        float(fade),
                        C.c_void_p(values.ctypes.data),
                    )
                ),
                "configure_reward",
            )

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

    def replay_step(self, game_index, actions, draws, obs, terminal, rewards, terminated, truncated):
        t = self.torch
        game_index = self._game_index(game_index)
        draw_array = self._draw_array(draws)
        action_ptr = self._ptr(actions, (2, 10), t.float32)
        pointers = [
            action_ptr,
            C.c_void_p(C.addressof(draw_array)) if len(draw_array) else C.c_void_p(),
            len(draw_array),
            self._ptr(obs, (2, 64), t.float32),
            self._ptr(terminal, (2, 64), t.float32),
            self._ptr(rewards, (2,), t.float32),
        ]
        self._ptr(terminated, (2,), t.bool)
        self._ptr(truncated, (2,), t.bool)
        if self.kind == "cuda":
            done = self._terminated[game_index:game_index + 1]
            trunc = self._truncated[game_index:game_index + 1]
            pointers.extend([C.c_void_p(done.data_ptr()), C.c_void_p(trunc.data_ptr())])
        else:
            done_byte, trunc_byte = C.c_uint8(), C.c_uint8()
            pointers.extend([C.byref(done_byte), C.byref(trunc_byte)])
        with self._guard():
            if self.kind == "cuda":
                stream = t.cuda.current_stream(self.device)
                for tensor in (actions, obs, terminal, rewards, terminated, truncated):
                    tensor.record_stream(stream)
            self._check(
                self.replay_step_fn(*self._args(game_index, *pointers)),
                "replay_step",
            )
            if self.kind == "cuda":
                terminated.copy_(done.expand(2).to(dtype=t.bool))
                truncated.copy_(trunc.expand(2).to(dtype=t.bool))
            else:
                terminated.fill_(bool(done_byte.value))
                truncated.fill_(bool(trunc_byte.value))

    def rng_status(self, game_index):
        game_index = self._game_index(game_index)
        mode, cursor, count, error = INT32(), INT32(), INT32(), INT32()
        with self._guard():
            self._check(
                self.rng_status_fn(
                    *self._args(
                        game_index,
                        C.byref(mode),
                        C.byref(cursor),
                        C.byref(count),
                        C.byref(error),
                    )
                ),
                "rng_status",
            )
        return {
            "mode": int(mode.value),
            "cursor": int(cursor.value),
            "count": int(count.value),
            "error": int(error.value),
        }

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

    def configure_combat_reward(self, terms, enabled_mask, *, fade=1.0, learner_slot=0):
        self.backend.configure_combat_reward(
            terms, enabled_mask, fade=fade, learner_slot=learner_slot
        )

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
                if record.reward.version:
                    infos[-1]["combat_reward"] = {
                        "version": record.reward.version,
                        "learner_slot": record.reward.learner_slot,
                        "return": record.reward.returns[player],
                        "terms": list(record.reward.term_totals[player]),
                        "clean_win": bool(record.reward.clean_win[player]),
                        "timeout": bool(record.reward.timeout),
                    }
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
