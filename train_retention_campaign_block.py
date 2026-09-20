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
from types import SimpleNamespace
import time
import traceback

import numpy as np

import benchmark_curriculum_throughput as throughput
import benchmark_native_contact_training as base
import train_contact_campaign_block as contact_block
from benchmark_native_contact_training import ACTION_EXPERIMENT_ARMS
from native_navigation_curriculum_env import TASK_MASKS, build_navigation_curriculum_env, NAVIGATION_VERSION


SCHEMA_VERSION = "ai2.retention_campaign_block.v1"
SCHEDULES = {
    "R4C1N": ("combat", "combat", "combat", "combat", "navigation"),
    "R3C2N": ("combat", "combat", "combat", "navigation", "navigation"),
}
TASK_VERSIONS = {
    "combat": base.CONTACT_VERSION,
    "navigation": NAVIGATION_VERSION,
}
SUPPORTED_ARMS = ("B_projected",)
SUPPORTED_LEARNING_RATES = (0.00003,)
ROOT = Path(__file__).resolve().parent


class NativeCurriculumPPO:
    @staticmethod
    def load(*args, **kwargs):
        from native_curriculum_ppo import NativeCurriculumPPO as implementation

        return implementation.load(*args, **kwargs)


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
    action_ab_arm: str = "B_projected"
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
            raise ValueError("retention campaign is frozen to the B_projected Bernoulli action arm")
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


def _raw_env(env):
    return getattr(env, "env", env)


def _to_numpy(value):
    for name in ("detach", "cpu", "contiguous"):
        method = getattr(value, name, None)
        if callable(method):
            value = method()
    method = getattr(value, "numpy", None)
    if callable(method):
        return method()
    return np.asarray(value)


def _strict_bool_mask(value, task):
    expected = np.asarray(TASK_MASKS[task], dtype=bool)
    actual = np.asarray(_to_numpy(value))
    if actual.dtype != np.dtype(bool) or actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise RuntimeError("episode info task/mask/version does not match the active rollout")
    return actual.tolist()


def completed_retention_episode_infos(callback, task):
    if task not in TASK_VERSIONS:
        raise RuntimeError("unsupported retention task")
    episodes = getattr(callback, "episodes", None)
    if episodes is None:
        episodes = []
    payload_key = "contact" if task == "combat" else "navigation"
    expected_version = TASK_VERSIONS[task]
    infos = []
    for info in episodes:
        if not isinstance(info, dict) or "episode" not in info:
            raise RuntimeError("native collector did not provide complete episode diagnostics")
        if info.get("curriculum_task") != task or info.get("curriculum_version") != expected_version:
            raise RuntimeError("episode info task/mask/version does not match the active rollout")
        _strict_bool_mask(info.get("allowed_control_mask"), task)
        payload = info.get(payload_key)
        if not isinstance(payload, dict):
            raise RuntimeError(f"{task} episode info is missing {payload_key} payload")
        if task == "combat" and not isinstance(payload.get("diagnostics"), dict):
            raise RuntimeError("combat episode info is missing contact diagnostics")
        infos.append(info)
    return infos


def write_task_episode_infos(output, rollout_index, task, infos):
    for info in infos:
        if info.get("curriculum_task") != task:
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


def task_scenario_metadata(env, task):
    raw = _raw_env(env)
    backend = getattr(raw, "backend", None)
    metadata = getattr(backend, "scenario_metadata", None)
    if metadata is not None:
        return metadata
    return {
        "curriculum_task": getattr(raw, "curriculum_task", task),
        "curriculum_version": getattr(raw, "curriculum_version", TASK_VERSIONS[task]),
        "allowed_control_mask": _strict_bool_mask(getattr(raw, "allowed_control_mask"), task),
        "host_has_timeouts": bool(raw.host_has_timeouts()),
        "backend_class": type(backend).__name__ if backend is not None else None,
    }


def _validate_rollout_progress(env, cfg, *, task, before_steps, before_count, model):
    learner_delta = model.num_timesteps - before_count
    expected_transitions = _transitions_per_rollout(cfg)
    if learner_delta != expected_transitions:
        raise RuntimeError(f"native {task} learner transition count is incorrect")
    if task == "navigation":
        return int(learner_delta)
    raw = _raw_env(env)
    after_steps = list(getattr(raw.backend, "valid_steps", []))
    delta = [int(after) - int(before) for after, before in zip(after_steps, before_steps)]
    if (
        len(delta) != cfg.lanes
        or any(value != cfg.n_steps for value in delta)
        or any(getattr(raw.backend, "errors", []))
    ):
        raise RuntimeError("native combat lane error or incorrect learner transition count")
    return int(sum(delta))


def validate_retention_runtime(model, cfg, *, fresh, requested_task):
    if requested_task not in TASK_VERSIONS:
        raise ValueError("unsupported retention task")
    for key in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "target_kl", "ent_coef"):
        if getattr(model, key) != getattr(cfg, key):
            raise ValueError(f"loaded retention learner runtime drift: {key}")
    if model.n_envs != cfg.lanes:
        raise ValueError("loaded retention learner lane count differs")
    if fresh and model.num_timesteps != throughput.CHECKPOINT_STEPS:
        raise ValueError("fresh canonical parent counter differs")
    if model.calibration or model.task != requested_task or getattr(model.policy, "task", None) != requested_task:
        raise ValueError("retention blocks require joint, non-calibration learning on the requested task")
    expected_mask = np.asarray(TASK_MASKS[requested_task], dtype=bool)
    actual_mask = np.asarray(_to_numpy(getattr(model.policy, "allowed_control_mask", None)))
    if actual_mask.dtype != np.dtype(bool) or not np.array_equal(actual_mask, expected_mask):
        raise ValueError("model policy mask differs from the requested retention task")
    if getattr(model, "arm", None) != "E1":
        raise ValueError("retention campaign requires the E1 curriculum learner")
    for progress in (0.0, 0.5, 1.0):
        if model.lr_schedule(progress) != cfg.learning_rate or model.clip_range(progress) != cfg.clip_range:
            raise ValueError("learning-rate or clip schedule differs")
    if any(parameter.device.type != "cuda" for parameter in model.policy.parameters()):
        raise RuntimeError("policy contains non-CUDA parameters")
    if getattr(model, "action_head", None) != cfg.action_head:
        raise ValueError("model action head differs from retention campaign config")
    if getattr(model, "action_ab_arm", None) != cfg.action_ab_arm:
        raise ValueError("model action A/B arm differs from retention campaign config")
    optimizer_state = throughput.optimizer_state_summary(model)
    empty_optimizer = (
        not optimizer_state.get("available") or optimizer_state.get("state_tensor_elements", 0) <= 0
    )
    if empty_optimizer:
        raise ValueError("retention campaign requires preserved optimizer state at load")
    return optimizer_state


def _tensor_digest(values):
    import hashlib

    digest = hashlib.sha256()
    for index, value in enumerate(values):
        array = _to_numpy(value)
        digest.update(str(index).encode("ascii") + b"\0")
        digest.update(str(getattr(array, "shape", "")).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _module_parameters(*modules):
    for module in modules:
        if module is None:
            continue
        parameters = getattr(module, "parameters", None)
        if parameters is not None:
            yield from parameters()


def actor_digest(model):
    custom = getattr(model, "actor_digest", None)
    if callable(custom):
        return custom()
    policy = model.policy
    return _tensor_digest(_module_parameters(
        getattr(getattr(policy, "mlp_extractor", None), "policy_net", None),
        getattr(policy, "action_net", None),
    )) + ":" + _tensor_digest([getattr(policy, "log_std")])


def critic_digests(model):
    custom = getattr(model, "critic_digests", None)
    if callable(custom):
        return custom()
    policy = model.policy
    extractor = getattr(policy, "mlp_extractor", None)
    return {
        "navigation": _tensor_digest(_module_parameters(
            getattr(extractor, "value_net", None),
            getattr(policy, "value_net", None),
        )),
        "combat": _tensor_digest(_module_parameters(
            getattr(policy, "combat_value_features", None),
            getattr(policy, "combat_value_net", None),
        )),
    }


def _digest_structure(value):
    import hashlib

    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, dict):
            digest.update(b"d{")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                digest.update(repr(key).encode("utf8") + b":")
                update(item[key])
            digest.update(b"}")
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"l[")
            for child in item:
                update(child)
            digest.update(b"]")
            return
        try:
            array = _to_numpy(item)
        except Exception:
            array = None
        if array is not None and hasattr(array, "tobytes") and getattr(array, "shape", None) is not None:
            digest.update(b"a")
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(str(array.shape).encode("ascii") + b"\0")
            digest.update(array.tobytes())
            return
        digest.update(b"s")
        digest.update(repr(item).encode("utf8"))

    update(value)
    return digest.hexdigest()


def optimizer_exact_digest(model):
    custom = getattr(model, "optimizer_state_exact", None)
    if callable(custom):
        return _digest_structure(custom())
    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    if optimizer is None or not callable(getattr(optimizer, "state_dict", None)):
        raise ValueError("optimizer exact state is unavailable")
    return _digest_structure(optimizer.state_dict())


def _checkpoint_invariant_state(model):
    return {
        "actor": actor_digest(model),
        "critics": critic_digests(model),
        "optimizer_exact": optimizer_exact_digest(model),
        "counters": throughput.final_model_counters(model),
        "task_history": list(getattr(model, "task_history", [])),
    }


def _states_match(source_state, loaded_model):
    return source_state == _checkpoint_invariant_state(loaded_model)


def _ready_for_requested_task(model, task):
    try:
        actual_mask = np.asarray(_to_numpy(model.policy.allowed_control_mask))
    except Exception:
        return False
    expected_mask = np.asarray(TASK_MASKS[task], dtype=bool)
    return (
        getattr(model, "task", None) == task
        and getattr(model.policy, "task", None) == task
        and getattr(model, "calibration", None) is False
        and actual_mask.dtype == np.dtype(bool)
        and actual_mask.shape == expected_mask.shape
        and np.array_equal(actual_mask, expected_mask)
    )


def _make_smoke_config(cfg):
    data = dict(cfg)
    data.setdefault("schedule_id", "R4C1N")
    data.setdefault("episodes_per_lane", max(int(data["n_steps"]) + 2, 16))
    return RetentionCampaignConfig(
        schedule_id=data["schedule_id"],
        lanes=int(data["lanes"]),
        n_steps=int(data["n_steps"]),
        batch_size=int(data["batch_size"]),
        n_epochs=int(data["n_epochs"]),
        seed=int(data["seed"]),
        device=data.get("device", "cuda"),
        learning_rate=float(data.get("learning_rate", 3e-5)),
        episodes_per_lane=int(data["episodes_per_lane"]),
    )


def _transitions_per_rollout(cfg):
    return int(getattr(cfg, "transitions_per_rollout", int(cfg.lanes) * int(cfg.n_steps)))


def _nominal_optimizer_steps(cfg):
    return int(getattr(cfg, "nominal_optimizer_steps", int(cfg.n_epochs) * _transitions_per_rollout(cfg) // int(cfg.batch_size)))


def run_qualification_smoke(*, schedule, cfg, source, config, build_dir, output_dir, checkpoint):
    del source, config
    if tuple(schedule) != ("navigation", "combat"):
        raise ValueError("qualification smoke must run navigation then combat")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    smoke_cfg = _make_smoke_config(cfg)
    device = base.require_cuda(smoke_cfg.device)
    args = SimpleNamespace(navigation_build_dir=Path(build_dir))
    environments = {}
    logger = None
    started = throughput.perf_counter_sequence()
    try:
        environments = build_task_environments(args, smoke_cfg, device)
        checkpoint_sha = sha256(checkpoint)
        model = NativeCurriculumPPO.load(checkpoint, env=environments["navigation"], expected_sha256=checkpoint_sha)
        if model.n_steps != smoke_cfg.n_steps:
            raise ValueError("qualification smoke n_steps must match the loaded native model")
        if model.batch_size != smoke_cfg.batch_size or model.n_epochs != smoke_cfg.n_epochs:
            raise ValueError("qualification smoke update geometry must match the loaded native model")
        model.set_logger(throughput.configure_logger(output / "ppo"))
        logger = model.logger
        BaseCallback = receipt_callback_base()
        callback = BaseCallback()
        _, callback = model._setup_learn(
            len(schedule) * _transitions_per_rollout(smoke_cfg),
            callback=callback,
            reset_num_timesteps=False,
        )
        callback.on_training_start(locals(), globals())
        actor_changed = {}
        critic_change_matrix = {}
        update_steps = 0
        timings = {}
        rows = []
        for index, task in enumerate(schedule):
            env = environments[task]
            model.set_task_environment(task, env, calibration=False)
            validate_retention_runtime(model, smoke_cfg, fresh=False, requested_task=task)
            before_actor = actor_digest(model)
            before_critics = critic_digests(model)
            before_count = model.num_timesteps
            before_steps = list(_raw_env(env).backend.valid_steps) if task == "combat" else None
            throughput.synchronize_cuda(device)
            collect_started = throughput.perf_counter_sequence()
            if not model.collect_rollouts(env, callback, model.rollout_buffer, smoke_cfg.n_steps):
                raise RuntimeError("qualification smoke collection did not complete")
            throughput.synchronize_cuda(device)
            collected = throughput.perf_counter_sequence()
            game_ticks = _validate_rollout_progress(
                env, smoke_cfg, task=task, before_steps=before_steps, before_count=before_count, model=model
            )
            model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
            model.train()
            throughput.synchronize_cuda(device)
            updated = throughput.perf_counter_sequence()
            collection_seconds = collected - collect_started
            update_seconds = updated - collected
            timings[f"{task}_collection_seconds"] = collection_seconds
            timings[f"{task}_update_seconds"] = update_seconds
            diagnostics = throughput.validate_update_diagnostics(model.last_update_diagnostics, smoke_cfg)
            after_actor = actor_digest(model)
            after_critics = critic_digests(model)
            actor_changed[task] = before_actor != after_actor
            critic_change_matrix[task] = {
                name: before_critics[name] != after_critics[name]
                for name in ("navigation", "combat")
            }
            update_steps += diagnostics["optimizer_steps"]
            metrics = base.loss_metrics(getattr(model.logger, "name_to_value", {}))
            if any(not math.isfinite(float(value)) for value in metrics.values()):
                raise RuntimeError("qualification smoke produced non-finite PPO metrics")
            rows.append({
                "rollout": index,
                "task": task,
                "game_ticks": game_ticks,
                "collection_seconds": collection_seconds,
                "update_seconds": update_seconds,
                "actual_update": diagnostics,
                "metrics": metrics,
            })
        callback.on_training_end()
        saved_state = _checkpoint_invariant_state(model)
        save_load_started = throughput.perf_counter_sequence()
        checkpoint_receipt = save_checkpoint(output / "final.zip", model)
        continuation = {}
        exact = True
        for task in ("navigation", "combat"):
            loaded = NativeCurriculumPPO.load(
                output / "final.zip",
                env=environments[task],
                expected_sha256=checkpoint_receipt["sha256"],
            )
            continuation[task] = _ready_for_requested_task(loaded, task)
            exact = exact and _states_match(saved_state, loaded)
        load_finished = throughput.perf_counter_sequence()
        timings["save_load_seconds"] = load_finished - save_load_started
        timings["total_seconds"] = load_finished - started
        result = {
            "status": "passed",
            "schedule": list(schedule),
            "checkpoint_sha256": checkpoint_sha,
            "actor_changed": actor_changed,
            "critic_change_matrix": critic_change_matrix,
            "optimizer_steps": update_steps,
            "nominal_optimizer_steps": len(schedule) * _nominal_optimizer_steps(smoke_cfg),
            "kl_early_stop": any(row["actual_update"]["kl_early_stop"] for row in rows),
            "save_load_exact": bool(exact),
            "next_task_continuation": continuation,
            "timings": timings,
            "roundtrip_checkpoint": checkpoint_receipt,
            "rollouts": rows,
        }
        atomic_json(output / "alternating-smoke.json", result)
        return result
    except BaseException as error:
        atomic_json(output / "alternating-smoke-failure.json", {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        })
        raise
    finally:
        if logger is not None:
            logger.close()
        _close_all(environments)


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
                infos = self.locals.get("infos")
                self.episodes = [] if infos is None else list(infos)

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
        validate_retention_runtime(model, cfg, fresh=not args.resume, requested_task="combat")
        optimizer_at_load = throughput.optimizer_state_summary(model)
        policy_digest_before = throughput.policy_parameter_digest(model)
        manifest.update({
            "model_parameter_count": throughput.model_parameter_count(model),
            "optimizer_state_at_load": optimizer_at_load,
            "learning_rate_variant": learning_variant,
            "start_model_counters": throughput.final_model_counters(model),
        })
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "scenarios-combat.json", task_scenario_metadata(environments["combat"], "combat"))
        atomic_json(output / "scenarios-navigation.json", task_scenario_metadata(environments["navigation"], "navigation"))
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
            before_steps = list(_raw_env(env).backend.valid_steps) if task == "combat" else None
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
            episode_infos = completed_retention_episode_infos(callback, task)
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
    parser.add_argument("--action-ab-arm", choices=SUPPORTED_ARMS, default="B_projected")
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
