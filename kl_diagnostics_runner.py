#!/usr/bin/env python3
"""Bounded KL-diagnostic replay for the native contact PPO update.

The default ``--dry-run`` path writes a replay plan only. A real run stays local:
it loads the supplied checkpoint, collects a finite number of contact rollouts,
and writes opt-in per-minibatch update diagnostics to JSONL. It does not create
cloud instances, change checkpoint defaults, or disable the target-KL safeguard.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = "ai2.kl_diagnostics_replay_plan.v1"
TRACE_SCHEMA_VERSION = "ai2.ppo_update_minibatch_trace.v1"


@dataclass(frozen=True)
class KLDiagnosticConfig:
    checkpoint: str
    expected_sha256: str
    output: str
    trace: str | None = None
    lanes: int = 64
    n_steps: int = 512
    batch_size: int = 8192
    n_epochs: int = 4
    seed: int = 145
    device: str = "cuda"
    update_mode: str = "deferred_metrics"
    target_kl: float = 0.01
    rollouts: int = 1
    max_diagnostics: int = 64
    dry_run: bool = False

    def validate(self) -> None:
        if len(self.expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.expected_sha256.lower()):
            raise ValueError("expected_sha256 must be a 64-character hex digest")
        if (self.lanes, self.n_steps, self.n_epochs) != (64, 512, 4):
            raise ValueError("Sept-9 replay geometry is fixed at 64 lanes x 512 steps x 4 epochs")
        if self.batch_size not in (8192, 16384):
            raise ValueError("batch_size must be one of the qualified contact minibatch arms")
        if self.update_mode != "deferred_metrics":
            raise ValueError("KL diagnostics only instrument update_mode=deferred_metrics; sb3 has no validated trace path")
        if self.target_kl != 0.01:
            raise ValueError("target_kl must remain the contact safeguard value 0.01")
        if self.rollouts <= 0 or self.rollouts > 4:
            raise ValueError("rollouts must be in [1, 4] for this bounded diagnostic runner")
        if self.max_diagnostics <= 0 or self.max_diagnostics > 512:
            raise ValueError("max_diagnostics must be in [1, 512]")

    @property
    def nominal_optimizer_steps(self) -> int:
        return self.n_epochs * self.n_steps * self.lanes // self.batch_size

    @property
    def trace_path(self) -> Path:
        if self.trace is not None:
            return Path(self.trace)
        output = Path(self.output)
        return output.with_name(output.stem + ".minibatches.jsonl")


class JSONLDiagnosticSink:
    def __init__(self, path: str | Path, *, limit: int | None = None) -> None:
        self.path = Path(path)
        self.limit = limit
        self.count = 0
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"refusing to mix KL diagnostics into existing trace: {self.path}")
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        os.close(fd)
        self._tmp = Path(tmp_name)
        self._stream = self._tmp.open("w", encoding="utf8")

    def __call__(self, row: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed KL diagnostic sink")
        if self.limit is not None and self.count >= self.limit:
            return
        self._stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        try:
            os.link(self._tmp, self.path)
        finally:
            self._tmp.unlink(missing_ok=True)
            self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self._stream.close()
        self._tmp.unlink(missing_ok=True)
        self._closed = True


def replay_plan(config: KLDiagnosticConfig) -> dict[str, Any]:
    config.validate()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run_unverified" if config.dry_run else "configured",
        "dry_run": bool(config.dry_run),
        "checkpoint": {
            "path": config.checkpoint,
            "expected_sha256": config.expected_sha256,
        },
        "geometry": {
            "lanes": config.lanes,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "nominal_optimizer_steps": config.nominal_optimizer_steps,
        },
        "ppo": {
            "target_kl": config.target_kl,
            "update_mode": config.update_mode,
        },
        "diagnostics": {
            "enabled": True,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "trace_path": str(config.trace_path),
            "max_rows": config.max_diagnostics,
        },
        "runner": {
            "device": config.device,
            "seed": config.seed,
            "rollouts": config.rollouts,
        },
        "verification": {
            "checkpoint_loaded": False,
            "rollouts_collected": False,
            "update_executed": False,
        },
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf8") + b"\n"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing KL diagnostic artifact: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def run_replay(config: KLDiagnosticConfig) -> dict[str, Any]:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    from native_contact_lane_vec_env import build_contact_env
    from native_curriculum_ppo import NativeCurriculumPPO
    from tensor_ppo import TensorVecEnv

    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for non-dry-run native KL diagnostics")

    raw_env = build_contact_env(config.lanes, device=config.device, seed=config.seed)
    env = TensorVecEnv(raw_env, device=config.device)
    model = NativeCurriculumPPO.load_contact_checkpoint(
        config.checkpoint,
        env,
        expected_sha256=config.expected_sha256,
        batch_size=config.batch_size,
        update_mode=config.update_mode,
    )
    sink = JSONLDiagnosticSink(config.trace_path, limit=config.max_diagnostics)
    model.update_diagnostic_sink = sink

    class Callback(BaseCallback):
        def _on_step(self) -> bool:
            return True

    total_timesteps = config.rollouts * config.n_steps * config.lanes
    _total, callback = model._setup_learn(total_timesteps=total_timesteps, callback=Callback(), reset_num_timesteps=False)
    try:
        for _ in range(config.rollouts):
            if not model.collect_rollouts(model.env, callback, model.rollout_buffer, model.n_steps):
                raise RuntimeError("rollout collection stopped before diagnostics completed")
            model.train()
        sink.close()
    except Exception:
        sink.abort()
        raise

    return {
        **replay_plan(config),
        "dry_run": False,
        "status": "replay_complete",
        "verification": {
            "checkpoint_loaded": True,
            "rollouts_collected": True,
            "update_executed": True,
        },
        "result": {
            "trace_rows": sink.count,
            "last_update_diagnostics": dict(model.last_update_diagnostics),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=145)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--update-mode", choices=("deferred_metrics", "sb3"), default="deferred_metrics")
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--max-diagnostics", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> KLDiagnosticConfig:
    return KLDiagnosticConfig(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256.lower(),
        output=args.output,
        trace=args.trace,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        update_mode=args.update_mode,
        rollouts=args.rollouts,
        max_diagnostics=args.max_diagnostics,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    payload = replay_plan(config) if config.dry_run else run_replay(config)
    write_json(config.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
