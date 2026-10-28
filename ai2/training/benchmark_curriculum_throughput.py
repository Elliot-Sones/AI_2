#!/usr/bin/env python3
"""Scaled native-curriculum throughput qualification.

This runner measures the same contact curriculum learner as
benchmark_native_contact_training.py, but removes the old 64-lane diagnostic
ceiling and avoids per-rollout checkpoint writes. It is a throughput screen,
not a claim that the resulting policy is good.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import traceback

import ai2.training.benchmark_native_contact_training as base
from ai2.training.benchmark_native_contact_training import (
    ACTION_EXPERIMENT_ARMS,
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    CHECKPOINT_STEPS,
    CONTACT_VERSION,
)

ROOT = PROJECT_ROOT
SUPPORTED_LANES = (64, 256, 1024, 2048)
OPTIONAL_LANES = (4096,)
EXTREME_LANES = (8192,)
SUPPORTED_BATCHES = (8192, 16384)
SUPPORTED_ARMS = ("G_keep", "B_projected")
SUPPORTED_LEARNING_RATES = (0.0001, 0.00003, 0.00001, 0.000003)
CANONICAL_LEARNING_RATE = 0.0001
SCHEMA_VERSION = "ai2.curriculum_throughput_qualification.v1"


@dataclass(frozen=True)
class ThroughputBenchmarkConfig:
    lanes: int = 64
    n_steps: int = 512
    batch_size: int = 8192
    n_epochs: int = 4
    warmup_rollouts: int = 1
    measured_rollouts: int = 6
    seed: int = 145
    device: str = "cuda"
    update_mode: str = "deferred_metrics"
    action_ab_arm: str = "G_keep"
    learning_rate: float = CANONICAL_LEARNING_RATE
    clip_range: float = 0.1
    target_kl: float = 0.01
    ent_coef: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    episodes_per_lane: int = 128
    allow_4096: bool = False
    allow_8192: bool = False

    def __post_init__(self):
        if any(type(getattr(self, name)) is not int for name in (
            "lanes", "n_steps", "batch_size", "n_epochs", "warmup_rollouts",
            "measured_rollouts", "seed", "episodes_per_lane",
        )):
            raise ValueError("rollout geometry and seed require integer values")
        lanes_allowed = (
            SUPPORTED_LANES
            + (OPTIONAL_LANES if self.allow_4096 else ())
            + (EXTREME_LANES if self.allow_8192 else ())
        )
        if self.lanes not in lanes_allowed:
            raise ValueError(f"lanes must be one of {lanes_allowed}")
        if self.n_steps != 512 or self.n_epochs != 4:
            raise ValueError("n_steps=512 and n_epochs=4 are fixed for this qualification")
        if self.batch_size not in SUPPORTED_BATCHES:
            raise ValueError("batch_size must be 8192 or 16384")
        if self.transitions_per_rollout % self.batch_size:
            raise ValueError("rollout size must be divisible by minibatch size")
        if self.warmup_rollouts < 0 or self.measured_rollouts <= 0:
            raise ValueError("warmup_rollouts >= 0 and measured_rollouts > 0 are required")
        if not 0 <= self.seed <= 2**32 - self.lanes:
            raise ValueError("seed and lane seeds must fit uint32")
        if self.device != "cuda" and not (self.device.startswith("cuda:") and self.device[5:].isdigit()):
            raise ValueError("throughput qualification requires CUDA; no CPU fallback")
        if self.update_mode not in ("sb3", "deferred_metrics"):
            raise ValueError("unsupported update mode")
        if self.action_ab_arm not in SUPPORTED_ARMS:
            raise ValueError("throughput screen only permits the qualified G_keep/B_projected arms")
        if self.learning_rate not in SUPPORTED_LEARNING_RATES:
            raise ValueError(f"learning_rate must be one of {SUPPORTED_LEARNING_RATES}")
        if (self.clip_range, self.target_kl, self.ent_coef, self.gamma, self.gae_lambda) != (
            0.1, 0.01, 0.001, 0.99, 0.95
        ):
            raise ValueError("approved PPO hyperparameters other than labeled LR diagnostics must not change")
        if self.episodes_per_lane < self.required_episodes_per_lane:
            raise ValueError("episodes_per_lane must be at least total rollouts plus one; native exhaustion remains fatal")

    @property
    def action_head(self):
        return ACTION_EXPERIMENT_ARMS[self.action_ab_arm]["action_head"]

    @property
    def transitions_per_rollout(self):
        return self.lanes * self.n_steps

    @property
    def nominal_optimizer_steps(self):
        return self.n_epochs * self.transitions_per_rollout // self.batch_size

    @property
    def total_rollouts(self):
        return self.warmup_rollouts + self.measured_rollouts

    @property
    def required_episodes_per_lane(self):
        return math.ceil(self.total_rollouts * self.n_steps / 300) + 1

    @property
    def record_capacity(self):
        return self.transitions_per_rollout

    @property
    def learning_contract_variant(self):
        return "canonical" if self.learning_rate == CANONICAL_LEARNING_RATE else "lr_diagnostic"

    @property
    def not_lossless_optimization(self):
        return self.learning_rate != CANONICAL_LEARNING_RATE


def sha256(path):
    return base.sha256(path)


def atomic_json(path, value):
    return base.atomic_json(path, value)


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def validate_deadline(deadline, *, now=None, reserve_seconds=120):
    return base.validate_deadline(deadline, now=now, reserve_seconds=reserve_seconds)


def validate_update_diagnostics(diagnostics, config):
    keys = ("attempted_epochs", "optimizer_steps", "kl_early_stop")
    if not isinstance(diagnostics, dict) or any(key not in diagnostics for key in keys):
        raise ValueError("actual PPO update work was not reported")
    epochs, steps, stopped = (diagnostics[key] for key in keys)
    if (
        type(epochs) is not int or not 1 <= epochs <= config.n_epochs
        or type(steps) is not int or not 0 <= steps <= config.nominal_optimizer_steps
        or type(stopped) is not bool
        or (not stopped and (epochs != config.n_epochs or steps != config.nominal_optimizer_steps))
    ):
        raise ValueError("invalid actual minibatch/epoch/KL-stop diagnostics")
    enriched = dict(diagnostics)
    enriched["nominal_optimizer_steps"] = config.nominal_optimizer_steps
    enriched["optimizer_step_fraction"] = steps / config.nominal_optimizer_steps
    enriched["low_update"] = bool(stopped or steps < config.nominal_optimizer_steps)
    return enriched


def summarize_measurements(rows, *, measured_window_seconds=None):
    if not rows:
        raise ValueError("no rollout rows")
    measured = [row for row in rows if not row.get("warmup")]
    warmups = [row for row in rows if row.get("warmup")]
    if not measured:
        raise ValueError("no measured rollouts")
    for row in rows:
        if row["transitions"] != row["game_ticks"]:
            raise ValueError("contact has exactly one learner transition per game tick")
        for key in ("collection_seconds", "update_seconds", "total_seconds"):
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError("invalid timing receipt")
    totals = {
        "measured_transitions": sum(row["transitions"] for row in measured),
        "measured_game_ticks": sum(row["game_ticks"] for row in measured),
        "measured_collection_seconds": sum(row["collection_seconds"] for row in measured),
        "measured_update_seconds": sum(row["update_seconds"] for row in measured),
        "measured_total_seconds": sum(row["total_seconds"] for row in measured),
        "warmup_rollouts": len(warmups),
        "measured_rollouts": len(measured),
        "low_update_measured_rollouts": sum(1 for row in measured if row["actual_update"]["low_update"]),
    }
    summary = {
        **totals,
        "measured_training_steps_per_second": totals["measured_transitions"] / totals["measured_total_seconds"],
        "measured_collection_steps_per_second": totals["measured_transitions"] / totals["measured_collection_seconds"],
        "measured_update_steps_per_second": totals["measured_transitions"] / totals["measured_update_seconds"],
        "learner_transitions_per_game_tick": 1,
        "timing_window": "measured_rollouts_only",
        "trainingwork_includes": "actual PPO optimizer steps reported by NativeCurriculumPPO, not nominal steps when KL stops early",
        "timing_excludes": "startup, scenario construction, logging writes, checkpoint writes and evaluation",
        "mastery_established": False,
        "full_update_candidate": totals["low_update_measured_rollouts"] == 0,
    }
    if measured_window_seconds is not None:
        if not math.isfinite(measured_window_seconds) or measured_window_seconds <= 0:
            raise ValueError("invalid continuous measured window")
        summary["measured_continuous_window_seconds"] = measured_window_seconds
        summary["measured_window_steps_per_second"] = totals["measured_transitions"] / measured_window_seconds
        summary["continuous_window_includes"] = "measured collection, update, host checks, rollout JSON writes and logger dumps"
    return summary


def model_parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.policy.parameters()))


def policy_parameter_digest(model):
    digest = hashlib.sha256()
    for index, parameter in enumerate(model.policy.parameters()):
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(str(index).encode("ascii") + b"\0")
        digest.update(str(getattr(array, "shape", "")).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def optimizer_state_summary(model):
    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    if optimizer is None:
        return {"available": False}
    state = getattr(optimizer, "state", {})
    tensors = 0
    elements = 0
    step_values = []
    for entry in state.values():
        if not isinstance(entry, dict):
            continue
        for name, value in entry.items():
            if hasattr(value, "numel"):
                tensors += 1
                elements += int(value.numel())
                if name == "step" and value.numel() == 1:
                    try:
                        step_values.append(float(value.detach().cpu().item()))
                    except Exception:
                        pass
            elif name == "step" and isinstance(value, (int, float)):
                step_values.append(float(value))
    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    return {
        "available": True,
        "state_entries": len(state),
        "state_tensors": tensors,
        "state_tensor_elements": elements,
        "lr_groups": [float(group.get("lr")) for group in getattr(optimizer, "param_groups", [])],
        "step_min": min(step_values) if step_values else None,
        "step_max": max(step_values) if step_values else None,
        "note": "records actual loaded optimizer state evidence; does not infer from provenance strings",
    }


def verified_action_initialization(model, optimizer_state):
    actual = dict(base.action_initialization_actual(model))
    if optimizer_state.get("available") and optimizer_state.get("state_tensor_elements", 0) > 0:
        actual["optimizer_provenance_verified"] = (
            "optimizer state tensors are present at load; provenance strings are logged as source metadata only"
        )
    else:
        actual["optimizer_provenance_verified"] = "no optimizer state tensors were detected at load"
    return actual


def apply_learning_rate_variant(model, cfg):
    if cfg.learning_rate != CANONICAL_LEARNING_RATE:
        model.learning_rate = cfg.learning_rate
        model.lr_schedule = lambda _: cfg.learning_rate
        optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
        if optimizer is None or not hasattr(optimizer, "param_groups"):
            raise ValueError("learning-rate diagnostic requires a policy optimizer with param_groups")
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate
    return {
        "learning_rate": cfg.learning_rate,
        "canonical_learning_rate": CANONICAL_LEARNING_RATE,
        "learning_contract_variant": cfg.learning_contract_variant,
        "not_lossless_optimization": cfg.not_lossless_optimization,
        "note": (
            "canonical checkpoint, architecture, actions, reward and target_kl are preserved; "
            "lane count, minibatch and rollout geometry are explicit sweep variables; "
            "non-default LR changes learning and is a diagnostic, not a lossless speed optimization"
        ),
    }


def final_model_counters(model):
    return {
        "num_timesteps": int(getattr(model, "num_timesteps")),
        "n_updates": int(getattr(model, "_n_updates", -1)),
        "episode_num": int(getattr(model, "_episode_num", -1)),
        "total_timesteps": int(getattr(model, "_total_timesteps", -1)),
    }


def validate_runtime(model, cfg):
    for key in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "target_kl", "ent_coef"):
        if getattr(model, key) != getattr(cfg, key):
            raise ValueError(f"loaded contact learner runtime drift: {key}")
    if model.n_envs != cfg.lanes or model.num_timesteps != CHECKPOINT_STEPS:
        raise ValueError("canonical checkpoint counter or lane count differs")
    if model.calibration or model.task != "combat" or model.arm != "E1":
        raise ValueError("the contact comparison requires joint, non-calibration E1 learning")
    for progress in (0.0, 0.5, 1.0):
        if model.lr_schedule(progress) != cfg.learning_rate or model.clip_range(progress) != cfg.clip_range:
            raise ValueError("approved constant learning rate/clip schedules differ")
    if any(parameter.device.type != "cuda" for parameter in model.policy.parameters()):
        raise RuntimeError("policy contains non-CUDA parameters")
    if getattr(model, "action_head", None) != cfg.action_head:
        raise ValueError("model action head differs from requested benchmark arm")
    if getattr(model, "action_ab_arm", None) != cfg.action_ab_arm:
        raise ValueError("model action A/B arm differs from requested benchmark arm")


def completed_episode_infos(callback):
    episodes = getattr(callback, "episodes", None)
    if episodes is None:
        episodes = []
    for info in episodes:
        if not isinstance(info, dict) or "episode" not in info:
            raise RuntimeError("native collector did not provide complete episode diagnostics")
    return list(episodes)


def write_episode_infos(output, rollout_index, infos):
    path = Path(output) / f"episodes-{rollout_index:03d}.json.gz"
    payload = json.dumps(infos, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
    atomic_bytes(path, gzip.compress(payload, mtime=0))
    return {
        "path": path.name,
        "count": len(infos),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "format": "json.gz",
        "compression": "gzip",
        "lossless": True,
    }


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def build_contact_env(*args, **kwargs):
    from ai2.simulation.native_contact_lane_vec_env import build_contact_env as real_build

    return real_build(*args, **kwargs)


def tensor_vec_env(env, device):
    from ai2.training.tensor_ppo import TensorVecEnv

    return TensorVecEnv(env, device=device)


def load_fresh_model(checkpoint, env, cfg):
    from ai2.training.native_curriculum_ppo import NativeCurriculumPPO

    return NativeCurriculumPPO.load_contact_checkpoint(
        checkpoint,
        env,
        expected_sha256=CHECKPOINT_SHA256,
        batch_size=cfg.batch_size,
        update_mode=cfg.update_mode,
        joint=True,
        action_ab_arm=cfg.action_ab_arm,
    )


def configure_logger(path):
    from stable_baselines3.common.logger import configure

    return configure(str(path), ["json", "csv"])


def synchronize_cuda(device):
    import torch

    torch.cuda.synchronize(device)


def cuda_peak_memory(device):
    import torch

    return int(torch.cuda.max_memory_allocated(device))


def cuda_device_name(device):
    import torch

    return torch.cuda.get_device_name(device)


def torch_precision_state():
    try:
        import torch
    except ImportError:
        return {"available": False}

    return {
        "available": True,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def reset_peak_memory(device):
    import torch

    torch.cuda.reset_peak_memory_stats(device)


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def perf_counter_sequence():
    return time.perf_counter()


def source_hashes(reference_root, frozen_root):
    hashes = dict(base.qualification_sources(reference_root, frozen_root))
    hashes["benchmark_curriculum_throughput.py"] = sha256(__file__)
    return hashes


def receipt_callback_base():
    try:
        from stable_baselines3.common.callbacks import BaseCallback

        return BaseCallback
    except ImportError:
        class MinimalCallback:
            def __init__(self):
                self.locals = {}
                self.episodes = []

            def update_locals(self, locals_):
                self.locals = dict(locals_)

            def on_training_start(self, *args):
                return None

            def on_training_end(self):
                return None

            def on_rollout_start(self):
                return None

            def on_step(self):
                return self._on_step()

            def on_rollout_end(self):
                return self._on_rollout_end()

        return MinimalCallback


def write_optional_checkpoint_and_actor(model, output, *, save_checkpoint, save_actor):
    artifacts = {}
    if save_checkpoint:
        temporary = output / "final.tmp.zip"
        final = output / "final.zip"
        model.save(temporary)
        os.link(temporary, final)
        temporary.unlink()
        artifacts["final_checkpoint"] = {
            "path": final.name,
            "bytes": final.stat().st_size,
            "sha256": sha256(final),
        }
        atomic_json(output / "final-checkpoint.json", artifacts["final_checkpoint"])
    if save_actor:
        if not hasattr(model, "export_actor"):
            raise RuntimeError("requested actor export, but model has no export_actor support")
        temporary = output / "actor.tmp.pt"
        actor = output / "actor.pt"
        model.export_actor(temporary)
        os.link(temporary, actor)
        temporary.unlink()
        artifacts["actor_export"] = {
            "path": actor.name,
            "bytes": actor.stat().st_size,
            "sha256": sha256(actor),
        }
        atomic_json(output / "actor-export.json", artifacts["actor_export"])
    return artifacts


def run_block(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a new output directory; never overwrite a throughput block")
    cfg = ThroughputBenchmarkConfig(
        lanes=args.lanes,
        batch_size=args.batch_size,
        warmup_rollouts=args.warmup_rollouts,
        measured_rollouts=args.measured_rollouts,
        seed=args.seed,
        device=args.device,
        update_mode=args.update_mode,
        action_ab_arm=args.action_ab_arm,
        allow_4096=args.allow_4096,
        allow_8192=args.allow_8192,
        episodes_per_lane=args.episodes_per_lane,
        learning_rate=args.learning_rate,
    )
    validate_deadline(args.deadline_unix)
    checkpoint = base.validate_checkpoint(args.checkpoint)
    preservation = base.require_checkpoint_preservation(args.checkpoint, getattr(args, "checkpoint_ack", None))
    qualification = base.require_qualification(args.qualification, args.reference_root, args.frozen_root)
    hashes = source_hashes(args.reference_root, args.frozen_root)
    device = base.require_cuda(cfg.device)

    os.environ["AI2_FROZEN_SOURCE_ROOT"] = str(Path(args.frozen_root).resolve())
    os.environ["WANDB_MODE"] = "offline"
    seed_everything(cfg.seed)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter_sequence()
    rows = []
    measured_window_start = None
    measured_window_end = None
    raw = None
    logger = None
    model = None

    try:
        reset_peak_memory(device)
    except Exception:
        pass

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(cfg),
        "checkpoint": checkpoint,
        "checkpoint_preservation": preservation,
        "qualification_sha256": sha256(args.qualification),
        "qualification": qualification,
        "source_sha256": hashes,
        "deadline_unix": args.deadline_unix,
        "started_unix": time.time(),
        "fresh_parent_per_block": True,
        "scenario_schedule_note": "finite bounded schedule; episode exhaustion is a fatal backend error, not a wraparound",
        "learning_contract_variant": cfg.learning_contract_variant,
        "not_lossless_optimization": cfg.not_lossless_optimization,
        "save_checkpoint": bool(args.save_checkpoint),
        "save_actor": bool(args.save_actor),
        "timing_excludes": "startup, scenario construction, logging writes, checkpoint writes and evaluation",
        "hardware": {
            "device": str(device),
            "gpu": cuda_device_name(device),
            "python": platform.python_version(),
            "packages": package_versions(("stable-baselines3", "numpy", "gymnasium", "pymunk")),
            "torch_precision": torch_precision_state(),
        },
        "claim": "throughput qualification only; original-game evaluation and learning-quality comparison remain separate",
    }

    try:
        BaseCallback = receipt_callback_base()

        class ReceiptCallback(BaseCallback):
            def _on_step(self):
                validate_deadline(args.deadline_unix, reserve_seconds=60)
                return True

            def _on_rollout_end(self):
                self.episodes = base.completed_episode_infos(self.locals)

        raw = build_contact_env(
            cfg.lanes,
            device=device,
            seed=cfg.seed,
            episodes_per_lane=cfg.episodes_per_lane,
            record_capacity=cfg.record_capacity,
        )
        env = tensor_vec_env(raw, device=device)
        model = load_fresh_model(args.checkpoint, env, cfg)
        learning_variant = apply_learning_rate_variant(model, cfg)
        validate_runtime(model, cfg)
        policy_digest_before = policy_parameter_digest(model)
        manifest["model_parameter_count"] = model_parameter_count(model)
        manifest["optimizer_state_at_load"] = optimizer_state_summary(model)
        manifest["learning_rate_variant"] = learning_variant
        manifest["action_initialization_actual"] = verified_action_initialization(
            model, manifest["optimizer_state_at_load"]
        )
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "scenarios.json", raw.backend.scenario_metadata)
        logger = configure_logger(output / "ppo")
        model.set_logger(logger)
        _, callback = model._setup_learn(
            cfg.total_rollouts * cfg.transitions_per_rollout,
            callback=ReceiptCallback(),
            reset_num_timesteps=False,
        )
        callback.on_training_start(locals(), globals())
        for index in range(cfg.total_rollouts):
            if index == cfg.warmup_rollouts:
                measured_window_start = perf_counter_sequence()
            validate_deadline(args.deadline_unix)
            before_count = model.num_timesteps
            before_steps = list(raw.backend.valid_steps)
            synchronize_cuda(device)
            begin = perf_counter_sequence()
            if not model.collect_rollouts(env, callback, model.rollout_buffer, cfg.n_steps):
                raise RuntimeError("cancelled/incomplete rollout is not a training result")
            synchronize_cuda(device)
            collected = perf_counter_sequence()
            delta = [int(after) - int(before) for after, before in zip(raw.backend.valid_steps, before_steps)]
            if (
                model.num_timesteps - before_count != cfg.transitions_per_rollout
                or len(delta) != cfg.lanes
                or any(value != cfg.n_steps for value in delta)
                or any(raw.backend.errors)
            ):
                raise RuntimeError("native lane error or incorrect learner transition count")
            validate_deadline(args.deadline_unix, reserve_seconds=60)
            model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
            model.train()
            synchronize_cuda(device)
            updated = perf_counter_sequence()
            actual_update = validate_update_diagnostics(model.last_update_diagnostics, cfg)
            episode_infos = completed_episode_infos(callback)
            episode_receipt = write_episode_infos(output, index, episode_infos)
            row = {
                "rollout": index,
                "warmup": index < cfg.warmup_rollouts,
                "transitions": cfg.transitions_per_rollout,
                "game_ticks": int(sum(delta)),
                "cumulative_timesteps": model.num_timesteps,
                "collection_seconds": collected - begin,
                "update_seconds": updated - collected,
                "total_seconds": updated - begin,
                "actual_update": actual_update,
                "metrics": base.loss_metrics(getattr(model.logger, "name_to_value", {})),
                "episode_info": episode_receipt,
                "episode_count": episode_receipt["count"],
            }
            atomic_json(output / f"rollout-{index:03d}.json", row)
            model.logger.dump(step=model.num_timesteps)
            if index >= cfg.warmup_rollouts:
                measured_window_end = perf_counter_sequence()
            rows.append(row)
            print(json.dumps({
                "rollout": index,
                "warmup": row["warmup"],
                "training_steps_per_second": row["transitions"] / row["total_seconds"],
                "actual_update": actual_update,
            }), flush=True)
        callback.on_training_end()
        policy_digest_after = policy_parameter_digest(model)
        artifacts = write_optional_checkpoint_and_actor(
            model, output, save_checkpoint=args.save_checkpoint, save_actor=args.save_actor
        )
        summary = summarize_measurements(
            rows,
            measured_window_seconds=None if measured_window_start is None or measured_window_end is None
            else measured_window_end - measured_window_start,
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "trained_not_evaluated" if args.save_checkpoint else "measured_not_evaluated",
            "summary": summary,
            "rollouts": rows,
            "model_parameter_count": manifest["model_parameter_count"],
            "learning_rate_variant": learning_variant,
            "policy_parameter_digest_before": policy_digest_before,
            "policy_parameter_digest_after": policy_digest_after,
            "policy_parameters_changed": policy_digest_before != policy_digest_after,
            "optimizer_state_at_load": manifest["optimizer_state_at_load"],
            "optimizer_state_at_end": optimizer_state_summary(model),
            "final_model_counters": final_model_counters(model),
            "peak_gpu_memory_bytes": cuda_peak_memory(device),
            "optional_artifacts": artifacts,
            "checkpoint_sha256": artifacts.get("final_checkpoint", {}).get("sha256"),
            "wall_seconds_including_setup_logging_optional_saves": perf_counter_sequence() - started,
            "finished_unix": time.time(),
            "next_required_action": "retrieve and hash-verify this block locally",
        }
        atomic_json(output / "result.json", result)
        if logger is not None:
            logger.close()
            logger = None
        base.write_inventory(output, block_id=output.name)
        return result
    except BaseException as error:
        if logger is not None:
            logger.close()
            logger = None
        if output.exists():
            base.preserve_failed_block(output, error, completed_rollouts=len(rows), last_completed_checkpoint=None)
        raise
    finally:
        if logger is not None:
            logger.close()
        if raw is not None:
            raw.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "frozen-root", "reference-root", "qualification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--checkpoint-ack", type=Path)
    parser.add_argument("--lanes", type=int, choices=SUPPORTED_LANES + OPTIONAL_LANES + EXTREME_LANES, default=64)
    parser.add_argument("--allow-4096", action="store_true")
    parser.add_argument("--allow-8192", action="store_true")
    parser.add_argument("--batch-size", type=int, choices=SUPPORTED_BATCHES, default=8192)
    parser.add_argument("--warmup-rollouts", type=int, default=1)
    parser.add_argument("--measured-rollouts", type=int, default=6)
    parser.add_argument("--episodes-per-lane", type=int, default=128)
    parser.add_argument("--seed", type=int, default=145)
    parser.add_argument("--learning-rate", type=float, choices=SUPPORTED_LEARNING_RATES, default=CANONICAL_LEARNING_RATE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--update-mode", choices=("sb3", "deferred_metrics"), default="deferred_metrics")
    parser.add_argument("--action-ab-arm", choices=SUPPORTED_ARMS, default="G_keep")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--save-actor", action="store_true")
    args = parser.parse_args(argv)
    ThroughputBenchmarkConfig(
        lanes=args.lanes,
        batch_size=args.batch_size,
        warmup_rollouts=args.warmup_rollouts,
        measured_rollouts=args.measured_rollouts,
        seed=args.seed,
        device=args.device,
        update_mode=args.update_mode,
        action_ab_arm=args.action_ab_arm,
        allow_4096=args.allow_4096,
        allow_8192=args.allow_8192,
        episodes_per_lane=args.episodes_per_lane,
        learning_rate=args.learning_rate,
    )
    return args


if __name__ == "__main__":
    run_block(parse_args())
