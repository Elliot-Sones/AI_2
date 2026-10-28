#!/usr/bin/env python3
"""Run one bounded native-contact curriculum training block.

This is the production block runner for staged contact training, not a
throughput screen.  It always writes one final checkpoint plus one actor export
for original-game evaluation, and it never writes per-rollout full checkpoints
unless a failure requires preserving completed work.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
from dataclasses import asdict, dataclass
import json
import math
import numbers
import os
from pathlib import Path
import platform
import time
import traceback

import ai2.training.benchmark_curriculum_throughput as throughput
import ai2.training.benchmark_native_contact_training as base
from ai2.training.benchmark_native_contact_training import ACTION_EXPERIMENT_ARMS


SCHEMA_VERSION = "ai2.contact_campaign_block.v1"
SUPPORTED_ARMS = ("G_keep", "B_projected")
SUPPORTED_LEARNING_RATES = (0.00003,)


@dataclass(frozen=True)
class ContactCampaignConfig:
    lanes: int = 4096
    rollouts: int = 5
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
        for name in ("lanes", "rollouts", "n_steps", "batch_size", "n_epochs", "seed", "episodes_per_lane"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.lanes != 4096:
            raise ValueError("campaign blocks are bounded to 4096 lanes")
        if not 1 <= self.rollouts <= 5:
            raise ValueError("rollouts must be between 1 and 5")
        if self.n_steps != 512 or self.n_epochs != 4:
            raise ValueError("n_steps=512 and n_epochs=4 are fixed")
        if self.batch_size not in (8192, 16384):
            raise ValueError("batch_size must be 8192 or 16384")
        if self.transitions_per_rollout % self.batch_size:
            raise ValueError("rollout size must be divisible by minibatch size")
        if self.learning_rate not in SUPPORTED_LEARNING_RATES:
            raise ValueError("campaign blocks currently use the verified 3e-5 learning rate")
        if self.device != "cuda" and not (self.device.startswith("cuda:") and self.device[5:].isdigit()):
            raise ValueError("native campaign blocks require CUDA; no CPU fallback")
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
    def action_head(self):
        return ACTION_EXPERIMENT_ARMS[self.action_ab_arm]["action_head"]

    @property
    def transitions_per_rollout(self):
        return self.lanes * self.n_steps

    @property
    def total_transitions(self):
        return self.rollouts * self.transitions_per_rollout

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


def empty_lane_episode_counts(cfg):
    return [0] * cfg.lanes


def accumulate_episode_counts(counts, infos):
    for info in infos:
        lane = (((info or {}).get("contact") or {}).get("lane"))
        if isinstance(lane, bool) or not isinstance(lane, numbers.Integral) or not 0 <= int(lane) < len(counts):
            raise RuntimeError("episode info is missing a valid contact lane")
        counts[int(lane)] += 1
    return counts


def require_schedule_capacity(cfg, completed_counts, *, next_rollout):
    del next_rollout  # documented in caller rows; check is current-lane state.
    required_remaining = cfg.n_steps + 1
    unsafe = []
    for lane, completed in enumerate(completed_counts):
        # The initial scenario is already active at environment creation/reset,
        # so only episodes_per_lane - 1 future resets exist before exhaustion.
        remaining = cfg.episodes_per_lane - 1 - int(completed)
        if remaining < required_remaining:
            unsafe.append({"lane": lane, "completed": int(completed), "remaining": remaining})
            if len(unsafe) >= 8:
                break
    if unsafe:
        raise RuntimeError(
            f"scenario capacity unsafe before next rollout: need {required_remaining} remaining per lane; "
            f"examples={unsafe}"
        )


def verify_input_checkpoint(args):
    checkpoint = Path(args.checkpoint)
    expected_sha = str(args.checkpoint_sha256)
    if len(expected_sha) != 64:
        raise ValueError("checkpoint-sha256 must be a 64-character digest")
    if args.resume:
        actual = sha256(checkpoint)
        if actual != expected_sha:
            raise ValueError(f"resume checkpoint SHA256 mismatch: expected {expected_sha}, got {actual}")
        canonical = False
        checkpoint_meta = {"path": str(checkpoint), "sha256": actual, "canonical_parent": False}
    else:
        if expected_sha != throughput.CHECKPOINT_SHA256:
            raise ValueError("fresh blocks must start from the canonical calibrated parent SHA256")
        checkpoint_meta = dict(base.validate_checkpoint(checkpoint))
        checkpoint_meta.update({"path": str(checkpoint), "sha256": throughput.CHECKPOINT_SHA256, "canonical_parent": True})
        canonical = True
    preservation = base.require_checkpoint_preservation(checkpoint, args.checkpoint_ack)
    return {**checkpoint_meta, "canonical_parent": canonical}, preservation


def load_fresh_campaign_model(checkpoint, env, cfg):
    return throughput.load_fresh_model(checkpoint, env, cfg)


def load_resume_campaign_model(checkpoint, env, cfg, expected_sha256):
    from ai2.training.native_curriculum_ppo import NativeCurriculumPPO

    del cfg
    return NativeCurriculumPPO.load(checkpoint, env=env, expected_sha256=expected_sha256)


def validate_campaign_runtime(model, cfg, *, fresh):
    for key in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "target_kl", "ent_coef"):
        if getattr(model, key) != getattr(cfg, key):
            raise ValueError(f"loaded contact learner runtime drift: {key}")
    if model.n_envs != cfg.lanes:
        raise ValueError("loaded contact learner lane count differs")
    if fresh and model.num_timesteps != throughput.CHECKPOINT_STEPS:
        raise ValueError("fresh canonical parent counter differs")
    if model.calibration or model.task != "combat" or model.arm != "E1":
        raise ValueError("campaign blocks require joint, non-calibration E1 combat learning")
    for progress in (0.0, 0.5, 1.0):
        if model.lr_schedule(progress) != cfg.learning_rate or model.clip_range(progress) != cfg.clip_range:
            raise ValueError("learning-rate or clip schedule differs")
    if any(parameter.device.type != "cuda" for parameter in model.policy.parameters()):
        raise RuntimeError("policy contains non-CUDA parameters")
    if getattr(model, "action_head", None) != cfg.action_head:
        raise ValueError("model action head differs from campaign config")
    if getattr(model, "action_ab_arm", None) != cfg.action_ab_arm:
        raise ValueError("model action A/B arm differs from campaign config")
    optimizer_state = throughput.optimizer_state_summary(model)
    empty_optimizer = (
        not optimizer_state.get("available") or optimizer_state.get("state_tensor_elements", 0) <= 0
    )
    if empty_optimizer and (not fresh or cfg.action_ab_arm == "G_keep"):
        raise ValueError("campaign training requires preserved optimizer state at load")
    return optimizer_state


def source_hashes(reference_root, frozen_root):
    hashes = dict(throughput.source_hashes(reference_root, frozen_root))
    hashes["train_contact_campaign_block.py"] = sha256(__file__)
    return hashes


def receipt_callback_base():
    return throughput.receipt_callback_base()


def save_checkpoint(path, model):
    temporary = path.with_name(path.stem + ".tmp.zip")
    model.save(temporary)
    saved = temporary
    if not saved.exists():
        appended = Path(str(temporary) + ".zip")
        if not appended.exists():
            raise FileNotFoundError(f"model.save did not create {temporary} or {appended}")
        saved = appended
    os.link(saved, path)
    saved.unlink()
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def save_actor(path, model):
    if not hasattr(model, "export_actor"):
        raise RuntimeError("model has no actor export support")
    temporary = path.with_name(path.name + ".tmp")
    model.export_actor(temporary)
    os.link(temporary, path)
    temporary.unlink()
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def save_failure_checkpoint(output, model):
    if model is None or not hasattr(model, "save"):
        return None
    destination = Path(output) / "failure-last.zip"
    if destination.exists():
        return destination.name
    save_checkpoint(destination, model)
    return destination.name


def run_block(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a new output directory; never overwrite a campaign block")
    cfg = ContactCampaignConfig(
        lanes=args.lanes,
        rollouts=args.rollouts,
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
    completed_counts = empty_lane_episode_counts(cfg)
    raw = None
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
            "checkpoint": checkpoint,
            "checkpoint_preservation": preservation,
            "resume": bool(args.resume),
            "fresh_parent_per_block": not bool(args.resume),
            "qualification_sha256": sha256(args.qualification),
            "qualification": qualification,
            "source_sha256": hashes,
            "deadline_unix": args.deadline_unix,
            "started_unix": time.time(),
            "scenario_schedule_note": (
                "finite bounded schedule; before every rollout require each lane has at least n_steps+1 "
                "future reset scenarios; exhaustion is fatal, never wrapped"
            ),
            "rng_and_environment_note": (
                "each block constructs a fresh environment and restarts RNG streams from the block seed; "
                "resume preserves model/optimizer state but makes no bitwise resumed trajectory claim"
            ),
            "hardware": {
                "device": str(device),
                "gpu": throughput.cuda_device_name(device),
                "python": platform.python_version(),
                "packages": throughput.package_versions(("stable-baselines3", "numpy", "gymnasium", "pymunk")),
                "torch_precision": throughput.torch_precision_state(),
            },
            "claim": "bounded native-contact training block; quality requires separate original-game evaluation",
        }

        raw = throughput.build_contact_env(
            cfg.lanes,
            device=device,
            seed=cfg.seed,
            episodes_per_lane=cfg.episodes_per_lane,
            record_capacity=cfg.record_capacity,
        )
        env = throughput.tensor_vec_env(raw, device=device)
        if args.resume:
            model = load_resume_campaign_model(args.checkpoint, env, cfg, args.checkpoint_sha256)
            learning_variant = {
                "learning_rate": cfg.learning_rate,
                "learning_contract_variant": "resume_preserved",
                "not_lossless_optimization": True,
                "note": "resume path preserves checkpoint policy, action head, LR schedule and optimizer state",
            }
        else:
            model = load_fresh_campaign_model(args.checkpoint, env, cfg)
            learning_variant = throughput.apply_learning_rate_variant(model, cfg)
        optimizer_at_load = validate_campaign_runtime(model, cfg, fresh=not args.resume)
        policy_digest_before = throughput.policy_parameter_digest(model)
        manifest.update({
            "model_parameter_count": throughput.model_parameter_count(model),
            "optimizer_state_at_load": optimizer_at_load,
            "learning_rate_variant": learning_variant,
            "action_initialization_actual": throughput.verified_action_initialization(model, optimizer_at_load),
            "start_model_counters": throughput.final_model_counters(model),
        })
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "scenarios.json", raw.backend.scenario_metadata)
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
        for index in range(cfg.rollouts):
            require_schedule_capacity(cfg, completed_counts, next_rollout=index)
            throughput.validate_deadline(args.deadline_unix)
            before_count = model.num_timesteps
            before_steps = list(raw.backend.valid_steps)
            throughput.synchronize_cuda(device)
            begin = throughput.perf_counter_sequence()
            if not model.collect_rollouts(env, callback, model.rollout_buffer, cfg.n_steps):
                raise RuntimeError("cancelled/incomplete rollout is not a training result")
            throughput.synchronize_cuda(device)
            collected = throughput.perf_counter_sequence()
            delta = [int(after) - int(before) for after, before in zip(raw.backend.valid_steps, before_steps)]
            if (
                model.num_timesteps - before_count != cfg.transitions_per_rollout
                or len(delta) != cfg.lanes
                or any(value != cfg.n_steps for value in delta)
                or any(raw.backend.errors)
            ):
                raise RuntimeError("native lane error or incorrect learner transition count")
            throughput.validate_deadline(args.deadline_unix, reserve_seconds=60)
            model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
            model.train()
            throughput.synchronize_cuda(device)
            updated = throughput.perf_counter_sequence()
            actual_update = throughput.validate_update_diagnostics(model.last_update_diagnostics, cfg)
            episode_infos = throughput.completed_episode_infos(callback)
            accumulate_episode_counts(completed_counts, episode_infos)
            episode_receipt = throughput.write_episode_infos(output, index, episode_infos)
            row = {
                "rollout": index,
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
                "lane_episode_counts_min": int(min(completed_counts)),
                "lane_episode_counts_max": int(max(completed_counts)),
            }
            atomic_json(output / f"rollout-{index:03d}.json", row)
            model.logger.dump(step=model.num_timesteps)
            rows.append(row)
            measured_window_end = throughput.perf_counter_sequence()
            print(json.dumps({
                "rollout": index,
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
            "summary": summary,
            "rollouts": rows,
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
        if raw is not None:
            raw.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "checkpoint-ack", "frozen-root", "reference-root", "qualification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--lanes", type=int, default=4096)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--batch-size", type=int, choices=(8192, 16384), default=16384)
    parser.add_argument("--episodes-per-lane", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=145)
    parser.add_argument("--learning-rate", type=float, choices=SUPPORTED_LEARNING_RATES, default=0.00003)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--update-mode", choices=("sb3", "deferred_metrics"), default="deferred_metrics")
    parser.add_argument("--action-ab-arm", choices=SUPPORTED_ARMS, default="G_keep")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    ContactCampaignConfig(
        lanes=args.lanes,
        rollouts=args.rollouts,
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
