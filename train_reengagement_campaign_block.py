"""Run one immutable native B2 reengagement/navigation training block."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import benchmark_curriculum_throughput as throughput
import benchmark_native_contact_training as base
import train_contact_campaign_block as contact_block
from benchmark_native_contact_training import ACTION_EXPERIMENT_ARMS
from native_navigation_curriculum_env import (
    NAVIGATION_VERSION,
    TASK_MASKS,
    build_navigation_curriculum_env,
)
from native_reengagement_lane_vec_env import CURRICULUM_VERSION as REENGAGEMENT_VERSION
from native_reengagement_lane_vec_env import build_reengagement_env

SCHEMA_VERSION = "ai2.reengagement_campaign_block.v1"
QUALIFICATION_SCHEMA_VERSION = "ai2.native_reengagement_qualification.v1"
PARENT_CHECKPOINT_SHA256 = "d6e487c22f73913cd88a428683c7ca721e23b1f9cb5a3a561ad24d1f5bf410db"
PARENT_LIFETIME_TRANSITIONS = 60_391_424
SCHEDULES = {
    "R4E1N": ("reengagement", "reengagement", "reengagement", "reengagement", "navigation"),
}
MODEL_TASKS = {
    "reengagement": "combat",
    "navigation": "navigation",
}
TASK_VERSIONS = {
    "reengagement": REENGAGEMENT_VERSION,
    "navigation": NAVIGATION_VERSION,
}
EXPECTED_TASK_VERSIONS = {
    "combat": REENGAGEMENT_VERSION,
    "navigation": NAVIGATION_VERSION,
}
SUPPORTED_ARMS = ("B_projected",)
SUPPORTED_LEARNING_RATES = (0.00003,)
ROOT = Path(__file__).resolve().parent
DIRECT_SOURCE_FILES = (
    "train_reengagement_campaign_block.py",
    "native_reengagement_lane_vec_env.py",
    "native_reengagement_episode.py",
    "native_navigation_curriculum_env.py",
    "native_curriculum_ppo.py",
    "benchmark_curriculum_throughput.py",
    "benchmark_native_contact_training.py",
    "train_contact_campaign_block.py",
)
QUALIFICATION_SOURCE_FILES = (
    "train_reengagement_campaign_block.py",
    "reengagement_curriculum.py",
    "native_reengagement_episode.py",
    "native_reengagement_lane_vec_env.py",
    "native_navigation_curriculum_env.py",
    "native_curriculum_ppo.py",
    "curriculum_ppo.py",
    "tensor_ppo.py",
    "tensor_rollout_buffer.py",
    "ppo_update_optimizations.py",
    "benchmark_curriculum_throughput.py",
    "benchmark_native_contact_training.py",
    "train_contact_campaign_block.py",
    "native_game_episode.py",
    "native_navigation_episode.py",
    "native_navigation_contact.py",
    "native_lane_vec_env.py",
    "native/reengagement_episode.h",
    "native/reengagement_episode.c",
    "native/reengagement_lanes.h",
    "native/reengagement_lanes.c",
    "native/reengagement_lanes.cu",
    "native/navigation_episode.h",
    "native/navigation_episode.c",
    "native/navigation_lanes.h",
    "native/navigation_lanes.c",
    "native/navigation_lanes.cu",
    "native/game_episode.h",
    "native/game_episode.c",
)


class NativeCurriculumPPO:
    @staticmethod
    def load_contact_checkpoint(*args, **kwargs):
        from native_curriculum_ppo import NativeCurriculumPPO as implementation

        return implementation.load_contact_checkpoint(*args, **kwargs)


@dataclass(frozen=True)
class ReengagementCampaignConfig:
    schedule_id: str = "R4E1N"
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
            raise ValueError("unsupported reengagement schedule")
        if self.lanes != 4096:
            raise ValueError("reengagement campaign blocks are bounded to 4096 lanes")
        if self.n_steps != 512 or self.batch_size != 16384 or self.n_epochs != 4:
            raise ValueError("reengagement campaign geometry is fixed at 4096x512, batch 16384, 4 epochs")
        if self.transitions_per_rollout % self.batch_size:
            raise ValueError("rollout size must be divisible by minibatch size")
        if self.learning_rate not in SUPPORTED_LEARNING_RATES:
            raise ValueError("reengagement campaign blocks use the verified 3e-5 learning rate")
        if self.device != "cuda" and not (self.device.startswith("cuda:") and self.device[5:].isdigit()):
            raise ValueError("native reengagement campaign blocks require CUDA; no CPU fallback")
        if self.update_mode not in ("sb3", "deferred_metrics"):
            raise ValueError("unsupported update mode")
        if self.action_ab_arm not in SUPPORTED_ARMS:
            raise ValueError("reengagement campaign is frozen to the B_projected Bernoulli action arm")
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
            for task in ("reengagement", "navigation")
        }

    @property
    def nominal_optimizer_steps(self):
        return self.n_epochs * self.transitions_per_rollout // self.batch_size

    @property
    def record_capacity(self):
        return self.transitions_per_rollout

    @property
    def learning_contract_variant(self):
        return "b2_reengagement_parent_preserved_lr_3e-5"

    @property
    def not_lossless_optimization(self):
        return True


def sha256(path):
    return base.sha256(path)


def atomic_json(path, value):
    return base.atomic_json(path, value)


def verify_input_checkpoint(args):
    checkpoint = Path(args.checkpoint)
    expected_sha = str(args.checkpoint_sha256)
    if expected_sha != PARENT_CHECKPOINT_SHA256:
        raise ValueError("B2 reengagement blocks must start from the exact B_projected-050m parent SHA256")
    actual = sha256(checkpoint)
    if actual != PARENT_CHECKPOINT_SHA256:
        raise ValueError(f"parent checkpoint SHA256 mismatch: expected {PARENT_CHECKPOINT_SHA256}, got {actual}")
    preservation = base.require_checkpoint_preservation(checkpoint, args.checkpoint_ack)
    return {
        "path": str(checkpoint),
        "sha256": actual,
        "canonical_parent": True,
        "num_timesteps": PARENT_LIFETIME_TRANSITIONS,
    }, preservation


def source_hashes(reference_root, frozen_root):
    hashes = dict(throughput.source_hashes(reference_root, frozen_root))
    for name in DIRECT_SOURCE_FILES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"required trainer source is missing: {name}")
        hashes[name] = sha256(path)
    return hashes


def qualification_source_hashes(reference_root=None, frozen_root=None):
    try:
        import qualify_native_reengagement as qualification
    except ImportError:
        qualification = None
    if qualification is not None:
        exported = getattr(qualification, "source_hashes", None)
        if callable(exported):
            try:
                return dict(exported(reference_root, frozen_root))
            except TypeError:
                try:
                    return dict(exported())
                except TypeError:
                    pass
    hashes = {}
    for name in QUALIFICATION_SOURCE_FILES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"required qualification source is missing: {name}")
        hashes[name] = sha256(path)
    return hashes


def _resolved_path_matches(actual, expected):
    if actual is None:
        return False
    return Path(actual).resolve() == Path(expected).resolve()


def _qualification_root(proof, key):
    provenance = proof.get("provenance") or {}
    return proof.get(key) or provenance.get(key)


def _validate_stage_artifact(receipt_path, stage, evidence_key, expected_sha):
    artifact = receipt_path.parent / f"{stage}.{evidence_key}"
    if not artifact.is_file() or artifact.is_symlink():
        raise ValueError(f"B2 qualification evidence artifact is missing: {artifact.name}")
    if sha256(artifact) != expected_sha:
        raise ValueError(f"B2 qualification evidence artifact changed: {artifact.name}")


def require_b2_qualification(path, reference_root, frozen_root):
    path = Path(path)
    proof = json.loads(path.read_text())
    expected_mask = [bool(value) for value in TASK_MASKS["combat"]]
    contract = proof.get("version_contract") or {}
    checkpoint = proof.get("checkpoint") or {}
    if (
        proof.get("schema_version") != QUALIFICATION_SCHEMA_VERSION
        or proof.get("qualified") is not True
        or contract.get("curriculum_task") != "combat"
        or contract.get("curriculum_version") != REENGAGEMENT_VERSION
        or [bool(value) for value in contract.get("allowed_control_mask", [])] != expected_mask
        or checkpoint.get("sha256") != PARENT_CHECKPOINT_SHA256
        or not _resolved_path_matches(_qualification_root(proof, "reference_root"), reference_root)
        or not _resolved_path_matches(_qualification_root(proof, "frozen_root"), frozen_root)
    ):
        raise ValueError("B2 reengagement qualification failed, stale, incomplete, or for a different contract")
    expected_sources = qualification_source_hashes(reference_root, frozen_root)
    if proof.get("source_hashes") != expected_sources:
        raise ValueError("B2 reengagement qualification source hashes do not match this source tree")
    stages = proof.get("stage_status") or {}
    for stage in ("python_oracle_native_cpu", "native_cpu_cuda", "collector_learner_smoke"):
        if (stages.get(stage) or {}).get("status") != "passed":
            raise ValueError(f"B2 reengagement qualification did not pass required stage: {stage}")
    for stage in ("python_oracle_native_cpu", "native_cpu_cuda"):
        evidence = (stages.get(stage) or {}).get("evidence") or {}
        for artifact_key, sha_key in (("log", "log_sha256"), ("xml", "xml_sha256")):
            digest = evidence.get(sha_key)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"B2 qualification evidence is missing {sha_key} for {stage}")
            _validate_stage_artifact(path, stage, artifact_key, digest)
    return proof


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


def _strict_bool_mask(value, model_task):
    expected = np.asarray(TASK_MASKS[model_task], dtype=bool)
    actual = np.asarray(_to_numpy(value))
    if actual.dtype != np.dtype(bool) or actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise RuntimeError("episode info task/mask/version does not match the active rollout")
    return actual.tolist()


def _dtype_name(value):
    dtype = getattr(value, "dtype", value)
    name = str(dtype)
    if name.startswith("<class 'numpy."):
        name = name.removeprefix("<class 'numpy.").removesuffix("'>")
    return name


def _require_float32_tensor(name, value):
    if value is None or not hasattr(value, "dtype"):
        return
    dtype_name = _dtype_name(value)
    if dtype_name not in {"float32", "torch.float32", "numpy.float32"}:
        raise RuntimeError(f"native reengagement training requires FP32; {name} has dtype {dtype_name}")


def _autocast_enabled():
    try:
        import torch
    except ImportError:
        return False
    checks = []
    try:
        checks.append(bool(torch.is_autocast_enabled()))
    except TypeError:
        pass
    try:
        checks.append(bool(torch.is_autocast_enabled("cuda")))
    except TypeError:
        pass
    except RuntimeError:
        pass
    try:
        checks.append(bool(torch.is_autocast_enabled("cpu")))
    except TypeError:
        cpu_check = getattr(torch, "is_autocast_cpu_enabled", None)
        if callable(cpu_check):
            try:
                checks.append(bool(cpu_check()))
            except RuntimeError:
                pass
    except RuntimeError:
        pass
    return any(checks)


def validate_fp32_runtime(model, precision_state):
    if _autocast_enabled():
        raise RuntimeError("native reengagement training requires FP32; autocast/mixed precision is enabled")
    if precision_state.get("cuda_matmul_allow_tf32") or precision_state.get("cudnn_allow_tf32"):
        raise RuntimeError("native reengagement training requires FP32; TF32 is not allowed")
    if precision_state.get("float32_matmul_precision") not in (None, "highest"):
        raise RuntimeError("native reengagement training requires highest float32 matmul precision")
    for index, parameter in enumerate(model.policy.parameters()):
        _require_float32_tensor(f"policy parameter {index}", parameter)
    rollout_buffer = getattr(model, "rollout_buffer", None)
    if rollout_buffer is not None:
        for name in (
            "observations",
            "actions",
            "executed_actions",
            "rewards",
            "episode_starts",
            "values",
            "log_probs",
            "advantages",
            "returns",
        ):
            _require_float32_tensor(f"rollout_buffer.{name}", getattr(rollout_buffer, name, None))
    return precision_state


def completed_reengagement_episode_infos(callback, rollout_task):
    if rollout_task not in TASK_VERSIONS:
        raise RuntimeError("unsupported reengagement campaign task")
    episodes = getattr(callback, "episodes", None)
    if episodes is None:
        episodes = []
    model_task = MODEL_TASKS[rollout_task]
    expected_version = TASK_VERSIONS[rollout_task]
    payload_key = "reengagement" if rollout_task == "reengagement" else "navigation"
    infos = []
    for info in episodes:
        if not isinstance(info, dict) or "episode" not in info:
            raise RuntimeError("native collector did not provide complete episode diagnostics")
        if info.get("curriculum_task") != model_task or info.get("curriculum_version") != expected_version:
            raise RuntimeError("episode info task/mask/version does not match the active rollout")
        _strict_bool_mask(info.get("allowed_control_mask"), model_task)
        payload = info.get(payload_key)
        if not isinstance(payload, dict):
            raise RuntimeError(  # noqa: TRY004 - invalid collector diagnostics are runtime evidence failures.
                f"{rollout_task} episode info is missing {payload_key} payload"
            )
        if rollout_task == "reengagement" and not isinstance(payload.get("diagnostics"), dict):
            raise RuntimeError("reengagement episode info is missing diagnostics")
        infos.append(info)
    return infos


def _episode_json_value(value):
    if isinstance(value, np.ndarray):
        return [_episode_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _episode_json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _episode_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_episode_json_value(item) for item in value]
    return value


def write_task_episode_infos(output, rollout_index, rollout_task, infos):
    payload_key = "reengagement" if rollout_task == "reengagement" else "navigation"
    for info in infos:
        if payload_key not in info:
            raise RuntimeError(f"{rollout_task} episode info is missing {payload_key} payload")
    path = Path(output) / f"episodes-{rollout_task}-{rollout_index:03d}.json.gz"
    payload = json.dumps(_episode_json_value(infos), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
    throughput.atomic_bytes(path, gzip.compress(payload, mtime=0))
    return {
        "path": path.name,
        "task": rollout_task,
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
    reengagement_raw = build_reengagement_env(
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
        "reengagement": throughput.tensor_vec_env(reengagement_raw, device=device),
        "navigation": throughput.tensor_vec_env(navigation_raw, device=device),
    }


def task_scenario_metadata(env, rollout_task):
    raw = _raw_env(env)
    backend = getattr(raw, "backend", None)
    metadata = getattr(backend, "scenario_metadata", None)
    if metadata is not None:
        return metadata
    model_task = MODEL_TASKS[rollout_task]
    return {
        "curriculum_task": getattr(raw, "curriculum_task", model_task),
        "curriculum_version": getattr(raw, "curriculum_version", TASK_VERSIONS[rollout_task]),
        "allowed_control_mask": _strict_bool_mask(raw.allowed_control_mask, model_task),
        "host_has_timeouts": bool(raw.host_has_timeouts()),
        "backend_class": type(backend).__name__ if backend is not None else None,
    }


def _validate_rollout_progress(env, cfg, *, rollout_task, before_steps, before_count, model):
    learner_delta = model.num_timesteps - before_count
    if learner_delta != cfg.transitions_per_rollout:
        raise RuntimeError(f"native {rollout_task} learner transition count is incorrect")
    if rollout_task == "navigation":
        return int(learner_delta)
    raw = _raw_env(env)
    after_steps = list(getattr(raw.backend, "valid_steps", []))
    delta = [int(after) - int(before) for after, before in zip(after_steps, before_steps)]
    if (
        len(delta) != cfg.lanes
        or any(value != cfg.n_steps for value in delta)
        or any(getattr(raw.backend, "errors", []))
    ):
        raise RuntimeError("native reengagement lane error or incorrect learner transition count")
    return int(sum(delta))


def validate_reengagement_runtime(model, cfg, *, requested_rollout_task):
    model_task = MODEL_TASKS[requested_rollout_task]
    for key in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "target_kl", "ent_coef"):
        if getattr(model, key) != getattr(cfg, key):
            raise ValueError(f"loaded reengagement learner runtime drift: {key}")
    if model.n_envs != cfg.lanes:
        raise ValueError("loaded reengagement learner lane count differs")
    if (
        model.num_timesteps != PARENT_LIFETIME_TRANSITIONS
        and requested_rollout_task == "reengagement"
        and not getattr(model, "_ai2_b2_parent_counter_checked", False)
    ):
        raise ValueError("B2 parent lifetime transition counter differs")
    model._ai2_b2_parent_counter_checked = True
    if model.calibration or model.task != model_task or getattr(model.policy, "task", None) != model_task:
        raise ValueError("reengagement blocks require joint, non-calibration learning on the requested task")
    expected_mask = np.asarray(TASK_MASKS[model_task], dtype=bool)
    actual_mask = np.asarray(_to_numpy(getattr(model.policy, "allowed_control_mask", None)))
    if actual_mask.dtype != np.dtype(bool) or not np.array_equal(actual_mask, expected_mask):
        raise ValueError("model policy mask differs from the requested reengagement task")
    if getattr(model, "arm", None) != "E1":
        raise ValueError("reengagement campaign requires the E1 curriculum learner")
    for progress in (0.0, 0.5, 1.0):
        if model.lr_schedule(progress) != cfg.learning_rate or model.clip_range(progress) != cfg.clip_range:
            raise ValueError("learning-rate or clip schedule differs")
    if any(parameter.device.type != "cuda" for parameter in model.policy.parameters()):
        raise RuntimeError("policy contains non-CUDA parameters")
    if getattr(model, "action_head", None) != cfg.action_head:
        raise ValueError("model action head differs from reengagement campaign config")
    if getattr(model, "action_ab_arm", None) != cfg.action_ab_arm:
        raise ValueError("model action A/B arm differs from reengagement campaign config")
    optimizer_state = throughput.optimizer_state_summary(model)
    empty_optimizer = (
        not optimizer_state.get("available") or optimizer_state.get("state_tensor_elements", 0) <= 0
    )
    if empty_optimizer:
        raise ValueError("reengagement campaign requires preserved optimizer state at load")
    return optimizer_state


def set_model_task_environment(model, rollout_task, env):
    model_task = MODEL_TASKS[rollout_task]
    try:
        model.set_task_environment(
            model_task,
            env,
            calibration=False,
            expected_task_versions=EXPECTED_TASK_VERSIONS,
        )
    except TypeError as error:
        if "expected_task_versions" not in str(error):
            raise
        model.set_task_environment(model_task, env, calibration=False)


def load_parent_model(checkpoint, env, cfg):
    return NativeCurriculumPPO.load_contact_checkpoint(
        checkpoint,
        env=env,
        expected_sha256=PARENT_CHECKPOINT_SHA256,
        batch_size=cfg.batch_size,
        update_mode=cfg.update_mode,
        joint=True,
        expected_task_versions=EXPECTED_TASK_VERSIONS,
    )


def run_block(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a new output directory; never overwrite a reengagement block")
    cfg = ReengagementCampaignConfig(
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
    qualification = require_b2_qualification(args.qualification, args.reference_root, args.frozen_root)
    hashes = source_hashes(args.reference_root, args.frozen_root)
    device = base.require_cuda(cfg.device)

    os.environ["AI2_FROZEN_SOURCE_ROOT"] = str(Path(args.frozen_root).resolve())
    os.environ["WANDB_MODE"] = "offline"
    throughput.seed_everything(cfg.seed)
    output.mkdir(parents=True, exist_ok=False)
    started = throughput.perf_counter_sequence()
    rows = []
    task_transitions = {"reengagement": 0, "navigation": 0}
    environments = {}
    logger = None
    model = None

    try:
        throughput.reset_peak_memory(device)
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
            "parent_lifetime_transitions": PARENT_LIFETIME_TRANSITIONS,
            "expected_task_versions": dict(EXPECTED_TASK_VERSIONS),
            "qualification_sha256": sha256(args.qualification),
            "qualification": qualification,
            "source_sha256": hashes,
            "deadline_unix": args.deadline_unix,
            "started_unix": time.time(),
            "rng_and_environment_note": (
                "model, actor, critics, optimizer, RNG counters, and lifetime counters resume from the parent checkpoint; "
                "live task environments are rebuilt fresh for this immutable block"
            ),
            "hardware": {
                "device": str(device),
                "gpu": throughput.cuda_device_name(device),
                "python": platform.python_version(),
                "packages": throughput.package_versions(("stable-baselines3", "numpy", "gymnasium", "pymunk")),
                "torch_precision": validate_fp32_runtime(model, throughput.torch_precision_state()) if model else throughput.torch_precision_state(),
            },
            "claim": "B2 reengagement/native-navigation training block; evaluation and promotion are separate",
        }

        environments = build_task_environments(args, cfg, device)
        env = environments["reengagement"]
        model = load_parent_model(args.checkpoint, env, cfg)
        learning_variant = {
            "learning_rate": cfg.learning_rate,
            "learning_contract_variant": cfg.learning_contract_variant,
            "not_lossless_optimization": True,
            "note": "B2 block preserves the parent policy, critics, optimizer, action head, LR schedule and counters",
        }
        optimizer_at_load = validate_reengagement_runtime(model, cfg, requested_rollout_task="reengagement")
        precision_state = validate_fp32_runtime(model, throughput.torch_precision_state())
        policy_digest_before = throughput.policy_parameter_digest(model)
        manifest.update({
            "model_parameter_count": throughput.model_parameter_count(model),
            "optimizer_state_at_load": optimizer_at_load,
            "learning_rate_variant": learning_variant,
            "action_initialization_actual": throughput.verified_action_initialization(model, optimizer_at_load),
            "start_model_counters": throughput.final_model_counters(model),
            "torch_precision": precision_state,
        })
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "scenarios-reengagement.json", task_scenario_metadata(environments["reengagement"], "reengagement"))
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
        for index, rollout_task in enumerate(cfg.schedule):
            throughput.validate_deadline(args.deadline_unix)
            env = environments[rollout_task]
            set_model_task_environment(model, rollout_task, env)
            if index == 0:
                validate_reengagement_runtime(model, cfg, requested_rollout_task=rollout_task)
            before_count = model.num_timesteps
            before_steps = list(_raw_env(env).backend.valid_steps) if rollout_task == "reengagement" else None
            throughput.synchronize_cuda(device)
            begin = throughput.perf_counter_sequence()
            if not model.collect_rollouts(env, callback, model.rollout_buffer, cfg.n_steps):
                raise RuntimeError("cancelled/incomplete rollout is not a training result")
            throughput.synchronize_cuda(device)
            collected = throughput.perf_counter_sequence()
            game_ticks = _validate_rollout_progress(
                env,
                cfg,
                rollout_task=rollout_task,
                before_steps=before_steps,
                before_count=before_count,
                model=model,
            )
            throughput.validate_deadline(args.deadline_unix, reserve_seconds=60)
            model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)
            episode_infos = completed_reengagement_episode_infos(callback, rollout_task)
            validate_fp32_runtime(model, throughput.torch_precision_state())
            model.train()
            throughput.synchronize_cuda(device)
            updated = throughput.perf_counter_sequence()
            actual_update = throughput.validate_update_diagnostics(model.last_update_diagnostics, cfg)
            episode_receipt = write_task_episode_infos(output, index, rollout_task, episode_infos)
            task_transitions[rollout_task] += cfg.transitions_per_rollout
            row = {
                "rollout": index,
                "task": rollout_task,
                "model_task": MODEL_TASKS[rollout_task],
                "curriculum_version": TASK_VERSIONS[rollout_task],
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
                "task": rollout_task,
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
            "resume_from_checkpoint_sha256": None,
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
    parser.add_argument("--checkpoint-sha256", choices=(PARENT_CHECKPOINT_SHA256,), required=True)
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
    args = parser.parse_args(argv)
    ReengagementCampaignConfig(
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
