#!/usr/bin/env python3
"""Run one immutable native contact/navigation retention training block."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

import benchmark_curriculum_throughput as throughput
import benchmark_native_contact_training as base
import train_contact_campaign_block as contact_block
from benchmark_native_contact_training import ACTION_EXPERIMENT_ARMS
from native_navigation_curriculum_env import build_navigation_curriculum_env, NAVIGATION_VERSION


SCHEMA_VERSION = "ai2.retention_campaign_block.v1"
SCHEDULES = {
    "R4C1N": ("combat", "combat", "combat", "combat", "navigation"),
    "R3C2N": ("combat", "combat", "combat", "navigation", "navigation"),
}
TASK_VERSIONS = {
    "combat": base.CONTACT_VERSION,
    "navigation": NAVIGATION_VERSION,
}
SUPPORTED_ARMS = ("G_keep", "B_projected")
SUPPORTED_LEARNING_RATES = (0.00003,)
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RetentionCampaignConfig:
    schedule_id: str = "R4C1N"
    lanes: int = 4096
    n_steps: int = 512
    batch_size: int = 16384
    n_epochs: int = 4
    seed: int = 145
    device: str = "cuda"
    update_mode: str = "deferred_metrics"
    action_ab_arm: str = "G_keep"
    learning_rate: float = 0.00003
    clip_range: float = 0.1
    target_kl: float = 0.01
    ent_coef: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    episodes_per_lane: int = 1024

    def __post_init__(self):
        for name in ("lanes", "n_steps", "batch_size", "n_epochs", "seed", "episodes_per_lane"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.schedule_id not in SCHEDULES:
            raise ValueError("unsupported retention schedule")
        if self.lanes != 4096:
            raise ValueError("retention campaign blocks are bounded to 4096 lanes")
        if self.n_steps != 512 or self.batch_size != 16384 or self.n_epochs != 4:
            raise ValueError("retention campaign geometry is fixed at 4096x512, batch 16384, 4 epochs")
        if self.transitions_per_rollout % self.batch_size:
            raise ValueError("rollout size must be divisible by minibatch size")
        if self.learning_rate not in SUPPORTED_LEARNING_RATES:
            raise ValueError("retention campaign blocks use the verified 3e-5 learning rate")
        if self.device != "cuda" and not (self.device.startswith("cuda:") and self.device[5:].isdigit()):
            raise ValueError("native retention campaign blocks require CUDA; no CPU fallback")
        if self.update_mode not in ("sb3", "deferred_metrics"):
            raise ValueError("unsupported update mode")
        if self.action_ab_arm not in SUPPORTED_ARMS:
            raise ValueError("unsupported action A/B arm")
        if (self.clip_range, self.target_kl, self.ent_coef, self.gamma, self.gae_lambda) != (
            0.1, 0.01, 0.001, 0.99, 0.95
        ):
            raise ValueError("approved PPO hyperparameters must not change")
        if not 0 <= self.seed <= 2**32 - self.lanes:
            raise ValueError("seed and lane seeds must fit uint32")
        if self.episodes_per_lane < self.n_steps + 2:
            raise ValueError("episodes_per_lane must leave at least one active plus one full-rollout margin")

    @property
    def schedule(self):
        return SCHEDULES[self.schedule_id]

    @property
    def action_head(self):
        return ACTION_EXPERIMENT_ARMS[self.action_ab_arm]["action_head"]

    @property
    def transitions_per_rollout(self):
        return self.lanes * self.n_steps

    @property
    def total_transitions(self):
        return len(self.schedule) * self.transitions_per_rollout

    @property
    def task_transition_totals(self):
        return {
            task: self.schedule.count(task) * self.transitions_per_rollout
            for task in ("combat", "navigation")
        }

    @property
    def nominal_optimizer_steps(self):
        return self.n_epochs * self.transitions_per_rollout // self.batch_size

    @property
    def record_capacity(self):
        return self.transitions_per_rollout

    @property
    def learning_contract_variant(self):
        return "campaign_lr_3e-5"

    @property
    def not_lossless_optimization(self):
        return True


def sha256(path):
    return base.sha256(path)


def atomic_json(path, value):
    return base.atomic_json(path, value)


def verify_input_checkpoint(args):
    return contact_block.verify_input_checkpoint(args)


def source_hashes(reference_root, frozen_root):
    hashes = dict(throughput.source_hashes(reference_root, frozen_root))
    hashes["train_retention_campaign_block.py"] = sha256(__file__)
    return hashes


def receipt_callback_base():
    return throughput.receipt_callback_base()


def write_task_episode_infos(output, rollout_index, task, infos):
    for info in infos:
        if info.get("curriculum_task") not in (None, task):
            raise RuntimeError("episode info task does not match the active rollout")
        payload_key = "contact" if task == "combat" else "navigation"
        if payload_key not in info:
            raise RuntimeError(f"{task} episode info is missing {payload_key} payload")
    path = Path(output) / f"episodes-{task}-{rollout_index:03d}.json.gz"
    payload = json.dumps(infos, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
    throughput.atomic_bytes(path, gzip.compress(payload, mtime=0))
    return {
        "path": path.name,
        "task": task,
        "count": len(infos),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "format": "json.gz",
        "compression": "gzip",
        "lossless": True,
    }


def save_checkpoint(path, model):
    return contact_block.save_checkpoint(path, model)


def save_actor(path, model):
    return contact_block.save_actor(path, model)


def save_failure_checkpoint(output, model):
    return contact_block.save_failure_checkpoint(output, model)


def _close_all(environments):
    for env in environments.values():
        close = getattr(env, "close", None)
        if close is not None:
            close()


def build_task_environments(args, cfg, device):
    navigation_build_dir = getattr(args, "navigation_build_dir", ROOT / "native" / "build")
    contact_raw = throughput.build_contact_env(
        cfg.lanes,
        device=device,
        seed=cfg.seed,
        episodes_per_lane=cfg.episodes_per_lane,
        record_capacity=cfg.record_capacity,
    )
    navigation_raw = build_navigation_curriculum_env(
        cfg.lanes,
        build_dir=navigation_build_dir,
        device=device,
        seed=cfg.seed + cfg.lanes,
        record_capacity=cfg.record_capacity,
    )
    return {
        "combat": throughput.tensor_vec_env(contact_raw, device=device),
        "navigation": throughput.tensor_vec_env(navigation_raw, device=device),
    }


def _validate_rollout_progress(env, cfg, *, task, before_steps, before_count, model):
    after_steps = list(getattr(env.backend, "valid_steps", []))
    delta = [int(after) - int(before) for after, before in zip(after_steps, before_steps)]
    if (
        model.num_timesteps - before_count != cfg.transitions_per_rollout
        or len(delta) != cfg.lanes
        or any(value != cfg.n_steps for value in delta)
        or any(getattr(env.backend, "errors", []))
    ):
        raise RuntimeError(f"native {task} lane error or incorrect learner transition count")
    return int(sum(delta))


def run_block(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a new output directory; never overwrite a retention block")
    cfg = RetentionCampaignConfig(
        schedule_id=args.schedule,
        lanes=args.lanes,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        update_mode=args.update_mode,
        action_ab_arm=args.action_ab_arm,
        learning_rate=args.learning_rate,
        episodes_per_lane=args.episodes_per_lane,
    )
    throughput.validate_deadline(args.deadline_unix)
    checkpoint, preservation = verify_input_checkpoint(args)
    qualification = base.require_qualification(args.qualification, args.reference_root, args.frozen_root)
    hashes = source_hashes(args.reference_root, args.frozen_root)
    device = base.require_cuda(cfg.device)

    os.environ["AI2_FROZEN_SOURCE_ROOT"] = str(Path(args.frozen_root).resolve())
    os.environ["WANDB_MODE"] = "offline"
    throughput.seed_everything(cfg.seed)
    output.mkdir(parents=True, exist_ok=False)
    started = throughput.perf_counter_sequence()
    rows = []
    task_transitions = {"combat": 0, "navigation": 0}
    environments = {}
    logger = None
    model = None

    try:
        try:
            throughput.reset_peak_memory(device)
        except Exception:
            pass
        BaseCallback = receipt_callback_base()

        class ReceiptCallback(BaseCallback):
            def _on_step(self):
                throughput.validate_deadline(args.deadline_unix, reserve_seconds=60)
                return True

            def _on_rollout_end(self):
                self.episodes = base.completed_episode_infos(self.locals)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config": asdict(cfg),
            "schedule_id": cfg.schedule_id,
            "schedule": list(cfg.schedule),
            "task_transition_totals": cfg.task_transition_totals,
            "checkpoint": checkpoint,
            "checkpoint_preservation": preservation,
            "resume": bool(args.resume),
            "fresh_parent_per_block": not bool(args.resume),
            "qualification_sha256": sha256(args.qualification),
            "qualification": qualification,
            "source_sha256": hashes,
            "deadline_unix": args.deadline_unix,
            "started_unix": time.time(),
            "rng_and_environment_note": (
                "model/critic/optimizer/RNG counters resume from the checkpoint; "
                "live environment state does not resume across immutable blocks, so each block builds fresh task environments"
            ),
            "hardware": {
                "device": str(device),
                "gpu": throughput.cuda_device_name(device),
                "python": platform.python_version(),
                "packages": throughput.package_versions(("stable-baselines3", "numpy", "gymnasium", "pymunk")),
                "torch_precision": throughput.torch_precision_state(),
            },
            "claim": "bounded native contact/navigation retention block; quality requires separate original-game evaluation",
        }

        environments = build_task_environments(args, cfg, device)
        env = environments["combat"]
        if args.resume:
            model = contact_block.load_resume_campaign_model(args.checkpoint, env, cfg, args.checkpoint_sha256)
            learning_variant = {
                "learning_rate": cfg.learning_rate,
                "learning_contract_variant": "resume_preserved",
                "not_lossless_optimization": True,
                "note": "resume path preserves checkpoint policy, critics, LR schedule and optimizer state",
            }
        else:
            model = contact_block.load_fresh_campaign_model(args.checkpoint, env, cfg)
            learning_variant = throughput.apply_learning_rate_variant(model, cfg)
        optimizer_at_load = contact_block.validate_campaign_runtime(model, cfg, fresh=not args.resume)
        policy_digest_before = throughput.policy_parameter_digest(model)
        manifest.update({
            "model_parameter_count": throughput.model_parameter_count(model),
            "optimizer_state_at_load": optimizer_at_load,
            "learning_rate_variant": learning_variant,
            "start_model_counters": throughput.final_model_counters(model),
        })
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "scenarios-combat.json", environments["combat"].backend.scenario_metadata)
        atomic_json(output / "scenarios-navigation.json", environments["navigation"].backend.scenario_metadata)
        logger = throughput.configure_logger(output / "ppo")
        model.set_logger(logger)
        _, callback = model._setup_learn(
            cfg.total_transitions,
            callback=ReceiptCallback(),
            reset_num_timesteps=False,
        )
        callback.on_training_start(locals(), globals())
        measured_window_start = throughput.perf_counter_sequence()
        measured_window_end = measured_window_start
        for index, task in enumerate(cfg.schedule):
            throughput.validate_deadline(args.deadline_unix)
            env = environments[task]
            model.set_task_environment(task, env, calibration=False)
            before_count = model.num_timesteps
            before_steps = list(env.backend.valid_steps)
            throughput.synchronize_cuda(device)
            begin = throughput.perf_counter_sequence()
            if not model.collect_rollouts(env, callback, model.rollout_buffer, cfg.n_steps):
                raise RuntimeError("cancelled/incomplete rollout is not a training result")
            throughput.synchronize_cuda(device)
            collected = throughput.perf_counter_sequence()
            game_ticks = _validate_rollout_progress(
                env, cfg, task=task, before_steps=before_steps, before_count=before_count, model=model
            )
            throughput.validate_deadline(args.deadline_unix, reserve_seconds=60)
            model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
            model.train()
            throughput.synchronize_cuda(device)
            updated = throughput.perf_counter_sequence()
            actual_update = throughput.validate_update_diagnostics(model.last_update_diagnostics, cfg)
            episode_infos = throughput.completed_episode_infos(callback)
            episode_receipt = write_task_episode_infos(output, index, task, episode_infos)
            task_transitions[task] += cfg.transitions_per_rollout
            row = {
                "rollout": index,
                "task": task,
                "curriculum_version": TASK_VERSIONS[task],
                "transitions": cfg.transitions_per_rollout,
                "task_transitions": cfg.transitions_per_rollout,
                "game_ticks": game_ticks,
                "cumulative_timesteps": model.num_timesteps,
                "cumulative_task_transitions": dict(task_transitions),
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
            rows.append(row)
            measured_window_end = throughput.perf_counter_sequence()
            print(json.dumps({
                "rollout": index,
                "task": task,
                "training_steps_per_second": row["transitions"] / row["total_seconds"],
                "actual_update": actual_update,
                "cumulative_timesteps": model.num_timesteps,
            }), flush=True)
        callback.on_training_end()
        policy_digest_after = throughput.policy_parameter_digest(model)
        final_checkpoint = save_checkpoint(output / "final.zip", model)
        actor = save_actor(output / "actor.pt", model)
        atomic_json(output / "final-checkpoint.json", final_checkpoint)
        atomic_json(output / "actor-export.json", actor)
        summary = throughput.summarize_measurements(
            [{**row, "warmup": False} for row in rows],
            measured_window_seconds=measured_window_end - measured_window_start,
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "trained_not_evaluated",
            "schedule_id": cfg.schedule_id,
            "schedule": list(cfg.schedule),
            "summary": summary,
            "rollouts": rows,
            "task_transition_totals": dict(task_transitions),
            "model_parameter_count": manifest["model_parameter_count"],
            "learning_rate_variant": learning_variant,
            "policy_parameter_digest_before": policy_digest_before,
            "policy_parameter_digest_after": policy_digest_after,
            "policy_parameters_changed": policy_digest_before != policy_digest_after,
            "optimizer_state_at_load": optimizer_at_load,
            "optimizer_state_at_end": throughput.optimizer_state_summary(model),
            "final_model_counters": throughput.final_model_counters(model),
            "peak_gpu_memory_bytes": throughput.cuda_peak_memory(device),
            "checkpoint_sha256": final_checkpoint["sha256"],
            "actor_sha256": actor["sha256"],
            "resume_from_checkpoint_sha256": args.checkpoint_sha256 if args.resume else None,
            "wall_seconds_including_setup_logging_final_saves": throughput.perf_counter_sequence() - started,
            "finished_unix": time.time(),
            "next_required_action": "retrieve, hash-verify and evaluate this checkpoint in the original Python game",
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
            last = save_failure_checkpoint(output, model)
            base.preserve_failed_block(
                output,
                error,
                completed_rollouts=len(rows),
                last_completed_checkpoint=last,
                traceback=traceback.format_exc(),
            )
        raise
    finally:
        if logger is not None:
            logger.close()
        _close_all(environments)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "checkpoint-ack", "frozen-root", "reference-root", "qualification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--schedule", choices=tuple(SCHEDULES), required=True)
    parser.add_argument("--navigation-build-dir", type=Path, default=ROOT / "native" / "build")
    parser.add_argument("--lanes", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, choices=(16384,), default=16384)
    parser.add_argument("--episodes-per-lane", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=145)
    parser.add_argument("--learning-rate", type=float, choices=SUPPORTED_LEARNING_RATES, default=0.00003)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--update-mode", choices=("sb3", "deferred_metrics"), default="deferred_metrics")
    parser.add_argument("--action-ab-arm", choices=SUPPORTED_ARMS, default="G_keep")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    RetentionCampaignConfig(
        schedule_id=args.schedule,
        lanes=args.lanes,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        update_mode=args.update_mode,
        action_ab_arm=args.action_ab_arm,
        learning_rate=args.learning_rate,
        episodes_per_lane=args.episodes_per_lane,
    )
    return args


if __name__ == "__main__":
    run_block(parse_args())
