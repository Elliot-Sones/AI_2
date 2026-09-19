#!/usr/bin/env python3
"""Fail-closed CUDA gate for production contact action-head safety.

This is deliberately not a training runner.  It constructs the same production
policy/PPO/action-adapter surface used by ``benchmark_native_contact_training``
and proves that raw policy tensors, decoded/executed actions, log-probabilities
and entropy remain finite CUDA tensors for both the Gaussian and Bernoulli
contact action heads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback


class _MissingCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _MissingTorch:
    cuda = _MissingCuda()
    __version__ = "not-installed"
    version = type("_Version", (), {"cuda": None})()


try:  # Keep dependency-light local contract tests importable.
    import torch  # type: ignore[assignment]
except Exception:  # pragma: no cover - exercised on local Python 3.14 host
    torch = _MissingTorch()  # type: ignore[assignment]


SOURCE_FILES = (
    "validate_production_action_cuda_gate.py",
    "curriculum_ppo.py",
    "native_curriculum_ppo.py",
    "benchmark_native_contact_training.py",
    "tensor_ppo.py",
    "tensor_rollout_buffer.py",
)
PACKAGE_NAMES = ("torch", "stable-baselines3", "numpy", "gymnasium", "pymunk")
CANONICAL_CHECKPOINT_SHA256 = "301ce3c72924f3e5c62f265f88292b8952e86e6d71d9a0552213cb8f3c6784b6"
ACTION_ARMS = ("G_keep", "G_neutral", "B_neutral", "B_projected")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def reject_existing_or_symlink_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("output receipt path must be new and must not be a symlink")
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def write_new_json(path: Path, payload: dict) -> None:
    reject_existing_or_symlink_output(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent or Path(".")))
    tmp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        if path.exists() or path.is_symlink():
            raise FileExistsError("output receipt path appeared during write")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def base_receipt(args: argparse.Namespace, root: Path, status: str, *, gating: bool) -> dict:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "schema": "ai2.production_action_cuda_gate.v1",
        "status": status,
        "gating": gating,
        "created_unix": time.time(),
        "command": {"argv": sys.argv if args._argv is None else args._argv, "cwd": str(root)},
        "source_hashes": source_hashes(root),
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "packages": package_versions(),
        "cuda": {"available": cuda_available},
    }


def _require_real_torch():
    if isinstance(torch, _MissingTorch) or getattr(torch, "__version__", "not-installed") == "not-installed":
        raise RuntimeError("torch is not installed; production action gate requires CUDA torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; production action gate has no CPU fallback")
    return torch


def _assert_cuda_finite(name: str, tensor) -> None:
    if not hasattr(tensor, "device") or tensor.device.type != "cuda":
        raise AssertionError(f"{name} is not a CUDA tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise AssertionError(f"{name} contains non-finite values")


def _tensor_hash(tensor) -> str:
    contiguous = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(contiguous).hexdigest()


def bind_checkpoint(checkpoint: Path | None, expected_sha256: str) -> dict:
    if checkpoint is None:
        raise ValueError("successful production action gate requires --checkpoint")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("checkpoint expected SHA256 must be explicit")
    path = Path(checkpoint)
    if not path.is_file() or path.is_symlink():
        raise ValueError("checkpoint must be an existing regular file")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"checkpoint SHA256 mismatch: expected {expected_sha256}, got {digest}")
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def load_production_checkpoint(learner_cls, *, checkpoint: Path, expected_sha256: str, env, arm: str, update_mode: str):
    return learner_cls.load_contact_checkpoint(
        checkpoint,
        env,
        expected_sha256=expected_sha256,
        update_mode=update_mode,
        joint=True,
        action_ab_arm=arm,
    )


def benchmark_default_update_mode() -> str:
    import benchmark_native_contact_training

    return benchmark_native_contact_training.ContactBenchmarkConfig().update_mode


def validate_loaded_runtime(model, benchmark_module, *, arm: str, update_mode: str, validate_runtime_fn=None) -> dict:
    expected_head = benchmark_module.ACTION_EXPERIMENT_ARMS[arm]["action_head"]
    cfg = benchmark_module.ContactBenchmarkConfig(
        update_mode=update_mode,
        action_head=expected_head,
        action_ab_arm=arm,
    )
    runtime_validator = validate_runtime_fn or benchmark_module.validate_runtime
    runtime_validator(model, cfg)
    return {
        "update_mode": cfg.update_mode,
        "action_ab_arm": cfg.action_ab_arm,
        "action_head": cfg.action_head,
        "lanes": cfg.lanes,
        "n_steps": cfg.n_steps,
        "batch_size": cfg.batch_size,
        "n_epochs": cfg.n_epochs,
        "expected_checkpoint_steps": benchmark_module.CHECKPOINT_STEPS,
    }


def export_and_verify_actor(model, load_actor_fn, *, arm: str, expected_head: str, device, work_dir: Path) -> dict:
    actor_path = work_dir / f"{arm}-actor.pt"
    metadata = model.export_actor(actor_path)
    actor, loaded = load_actor_fn(
        actor_path,
        device=device,
        expected_action_head=expected_head,
        expected_action_ab_arm=arm,
    )
    if actor.action_head != expected_head or loaded["action_ab_arm"] != arm or metadata["action_ab_arm"] != arm:
        raise AssertionError(f"{arm}.export/load metadata did not round-trip")
    mismatch_rejections: list[str] = []
    other_head = "bernoulli_10" if expected_head == "gaussian" else "gaussian"
    try:
        load_actor_fn(actor_path, device=device, expected_action_head=other_head)
    except ValueError as exc:
        mismatch_rejections.append(str(exc))
    else:
        raise AssertionError(f"{arm}.export load accepted wrong expected action head")
    try:
        load_actor_fn(actor_path, device=device, expected_action_ab_arm="B_projected" if arm != "B_projected" else "G_keep")
    except ValueError as exc:
        mismatch_rejections.append(str(exc))
    else:
        raise AssertionError(f"{arm}.export load accepted wrong expected action arm")
    return {
        "actor_sha256": sha256_file(actor_path),
        "metadata_action_head": loaded["action_head"],
        "metadata_action_ab_arm": loaded["action_ab_arm"],
        "mismatch_rejections": mismatch_rejections,
    }


def _build_fixture_env(device):
    import numpy as np
    from benchmark_native_contact_training import ContactBenchmarkConfig
    from gymnasium import spaces
    from native_curriculum_ppo import COMBAT_MASK, CONTACT_VERSION
    from tensor_ppo import TensorStep

    class Fixture:
        num_envs = ContactBenchmarkConfig().lanes
        observation_space = spaces.Box(-10, 10, (256,), dtype=np.float32)
        action_space = spaces.Box(0, 1, (10,), dtype=np.float32)
        curriculum_task = "combat"
        curriculum_version = CONTACT_VERSION
        allowed_control_mask = COMBAT_MASK

        def __init__(self):
            self.device = device
            self.tick = 0
            self.last_action = None

        def reset_tensor(self, seeds=None):
            del seeds
            self.tick = 0
            return torch.zeros((self.num_envs, 256), dtype=torch.float32, device=self.device)

        def step_tensor(self, action):
            self.last_action = action.clone()
            self.tick += 1
            observations = torch.full(
                (self.num_envs, 256),
                0.01 * (self.tick % 20),
                dtype=torch.float32,
                device=self.device,
            )
            terminated = torch.full((self.num_envs,), self.tick % 32 == 0, dtype=torch.bool, device=self.device)
            truncated = torch.zeros_like(terminated)
            rewards = -action[:, 0].contiguous()
            return TensorStep(observations, rewards, terminated, truncated, observations.clone())

        def host_has_timeouts(self):
            return False

        def drain_episode_infos(self):
            return []

    return Fixture()


def _validate_arm(arm: str, device, work_dir: Path, *, checkpoint: Path, expected_sha256: str, update_mode: str) -> dict:
    import benchmark_native_contact_training
    from curriculum_ppo import ACTION_EXPERIMENT_ARMS, load_actor
    from native_curriculum_ppo import COMBAT_MASK, NativeCurriculumPPO
    from stable_baselines3.common.callbacks import BaseCallback
    from tensor_ppo import TensorVecEnv

    expected_head = ACTION_EXPERIMENT_ARMS[arm]["action_head"]
    raw_env = _build_fixture_env(device)
    env = TensorVecEnv(raw_env, device=device)
    model = load_production_checkpoint(
        NativeCurriculumPPO,
        checkpoint=checkpoint,
        expected_sha256=expected_sha256,
        env=env,
        arm=arm,
        update_mode=update_mode,
    )
    runtime = validate_loaded_runtime(
        model,
        benchmark_native_contact_training,
        arm=arm,
        update_mode=update_mode,
    )

    class Callback(BaseCallback):
        def _on_step(self):
            return True

    torch.manual_seed(173)
    _, callback = model._setup_learn(8192, callback=Callback(), reset_num_timesteps=False)
    callback.on_training_start(locals(), globals())
    collected = model.collect_rollouts(model.env, callback, model.rollout_buffer, model.rollout_buffer.buffer_size)
    if not collected:
        raise RuntimeError("production action gate rollout collection did not complete")

    buffer = model.rollout_buffer
    flat_observations = buffer.observations.reshape(-1, 256)
    flat_actions = buffer.actions.reshape(-1, 10)
    _assert_cuda_finite(f"{arm}.buffer.observations", buffer.observations)
    _assert_cuda_finite(f"{arm}.buffer.raw_actions", buffer.actions)
    _assert_cuda_finite(f"{arm}.buffer.executed_actions", buffer.executed_actions)
    _assert_cuda_finite(f"{arm}.buffer.log_probs", buffer.log_probs)
    if buffer.allowed_control_masks.device.type != "cuda":
        raise AssertionError(f"{arm}.buffer.allowed_control_masks is not CUDA")

    with torch.no_grad():
        sampled_actions, sampled_values, sampled_log_probs = model.policy(flat_observations, deterministic=False)
        deterministic_actions, deterministic_values, deterministic_log_probs = model.policy(
            flat_observations, deterministic=True
        )
        executed_sampled = model.policy.decode_actions(sampled_actions.reshape(-1, 10))
        evaluated_values, evaluated_log_probs, entropy = model.policy.evaluate_actions(flat_observations, flat_actions)

    for name, tensor in (
        ("sampled_actions", sampled_actions),
        ("sampled_values", sampled_values),
        ("sampled_log_probs", sampled_log_probs),
        ("deterministic_actions", deterministic_actions),
        ("deterministic_values", deterministic_values),
        ("deterministic_log_probs", deterministic_log_probs),
        ("executed_sampled", executed_sampled),
        ("evaluated_values", evaluated_values),
        ("evaluated_log_probs", evaluated_log_probs),
        ("entropy", entropy),
    ):
        _assert_cuda_finite(f"{arm}.{name}", tensor)

    mask = torch.as_tensor(COMBAT_MASK, dtype=torch.bool, device=device)
    if not bool((buffer.allowed_control_masks == mask).all()):
        raise AssertionError(f"{arm}.active-control masks differ from contact mask")
    if expected_head == "gaussian":
        expected_executed = buffer.actions.clamp(0, 1) * mask.to(buffer.actions.dtype)
    else:
        if not bool(((buffer.actions == 0) | (buffer.actions == 1)).all()):
            raise AssertionError("Bernoulli rollout raw keys are not exact binary buttons")
        if not bool(((buffer.executed_actions == 0) | (buffer.executed_actions == 1)).all()):
            raise AssertionError("Bernoulli executed keys are not exact binary buttons")
        expected_executed = buffer.actions * mask.to(buffer.actions.dtype)
    if not torch.equal(buffer.executed_actions, expected_executed):
        raise AssertionError(f"{arm}.raw-vs-executed rollout semantics differ from production decode")
    if not torch.equal(raw_env.last_action, buffer.executed_actions[-1]):
        raise AssertionError(f"{arm}.environment did not receive decoded executed actions")

    metrics = benchmark_native_contact_training.action_press_metrics(model)
    if metrics.get("action_head") != expected_head:
        raise AssertionError(f"{arm}.benchmark action metrics did not observe expected action head")

    export_load = export_and_verify_actor(
        model,
        load_actor,
        arm=arm,
        expected_head=expected_head,
        device=device,
        work_dir=work_dir,
    )

    wrong_checkpoint = work_dir / f"{arm}-wrong-checkpoint.zip"
    wrong_checkpoint.write_bytes(b"wrong checkpoint bytes\n")
    checkpoint_mismatch_rejections: list[str] = []
    try:
        NativeCurriculumPPO.load_contact_checkpoint(
            wrong_checkpoint,
            env,
            expected_sha256="0" * 64,
            update_mode="sb3",
            joint=True,
            action_ab_arm=arm,
        )
    except ValueError as exc:
        checkpoint_mismatch_rejections.append(str(exc))
    else:
        raise AssertionError(f"{arm}.checkpoint hash gate accepted wrong checkpoint bytes")

    return {
        "action_ab_arm": arm,
        "action_head": expected_head,
        "policy_class": type(model.policy).__name__,
        "learner_class": type(model).__name__,
        "checkpoint_load": "NativeCurriculumPPO.load_contact_checkpoint",
        "update_mode": runtime["update_mode"],
        "runtime_validation": runtime,
        "forward": {
            "sampled_actions_cuda": sampled_actions.device.type == "cuda",
            "deterministic_actions_cuda": deterministic_actions.device.type == "cuda",
            "sampled_actions_sha256": _tensor_hash(sampled_actions),
            "deterministic_actions_sha256": _tensor_hash(deterministic_actions),
        },
        "evaluate_actions": {
            "logprob_cuda": evaluated_log_probs.device.type == "cuda",
            "entropy_cuda": entropy.device.type == "cuda",
            "logprob_sha256": _tensor_hash(evaluated_log_probs),
            "entropy_sha256": _tensor_hash(entropy),
        },
        "rollout": {
            "buffer_size": int(buffer.buffer_size),
            "n_envs": int(buffer.n_envs),
            "raw_actions_cuda": buffer.actions.device.type == "cuda",
            "executed_actions_cuda": buffer.executed_actions.device.type == "cuda",
            "log_probs_cuda": buffer.log_probs.device.type == "cuda",
            "allowed_control_masks_cuda": buffer.allowed_control_masks.device.type == "cuda",
            "raw_actions_sha256": _tensor_hash(buffer.actions),
            "executed_actions_sha256": _tensor_hash(buffer.executed_actions),
        },
        "active_control_mask": [bool(value) for value in COMBAT_MASK],
        "benchmark_action_metrics": metrics,
        "export_load": export_load,
        "checkpoint_mismatch_rejections": checkpoint_mismatch_rejections,
    }


def run_cuda_gate(args: argparse.Namespace, root: Path) -> dict:
    real_torch = _require_real_torch()
    checkpoint = bind_checkpoint(args.checkpoint, args.expected_sha256)
    device = real_torch.device("cuda", real_torch.cuda.current_device())
    real_torch.cuda.set_device(device)
    receipt = base_receipt(args, root, "passed", gating=True)
    receipt["checkpoint"] = checkpoint
    receipt["update_mode"] = args.update_mode
    receipt["cuda"].update({
        "device": str(device),
        "device_name": real_torch.cuda.get_device_name(device),
        "torch_cuda": real_torch.version.cuda,
    })
    with tempfile.TemporaryDirectory(prefix="ai2-production-action-gate-") as directory:
        work_dir = Path(directory)
        receipt["arms"] = [
            _validate_arm(
                arm,
                device,
                work_dir,
                checkpoint=Path(checkpoint["path"]),
                expected_sha256=checkpoint["sha256"],
                update_mode=args.update_mode,
            )
            for arm in ACTION_ARMS
        ]
    receipt["claims"] = {
        "no_training_performed": True,
        "policy_and_action_tensors_cuda_finite": True,
        "raw_vs_executed_rollout_semantics_checked": True,
        "bernoulli_exact_binary_keys_checked": True,
        "export_and_checkpoint_mismatch_rejection_checked": True,
    }
    return receipt


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new JSON receipt path; must not exist or be a symlink")
    parser.add_argument("--checkpoint", type=Path, help="canonical calibrated production checkpoint zip")
    parser.add_argument("--expected-sha256", default=CANONICAL_CHECKPOINT_SHA256, help="expected checkpoint SHA256")
    parser.add_argument("--update-mode", default=benchmark_default_update_mode(), choices=("sb3", "deferred_metrics"))
    parser.add_argument(
        "--allow-cuda-skip",
        action="store_true",
        help="write a non-gating skipped receipt instead of failing when CUDA is unavailable",
    )
    args = parser.parse_args(argv)
    args._argv = [Path(sys.argv[0]).name, *(argv or sys.argv[1:])]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        reject_existing_or_symlink_output(args.output)
        if not torch.cuda.is_available():
            if not args.allow_cuda_skip:
                raise RuntimeError("CUDA is unavailable; production action gate has no CPU fallback")
            receipt = base_receipt(args, root, "skipped", gating=False)
            receipt["reason"] = "CUDA unavailable and --allow-cuda-skip was explicitly provided"
            write_new_json(args.output, receipt)
            return 0
        receipt = run_cuda_gate(args, root)
        write_new_json(args.output, receipt)
        return 0
    except Exception as exc:
        print(f"production action CUDA gate failed: {exc}", file=sys.stderr)
        if os.environ.get("AI2_GATE_TRACEBACK") == "1":
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
