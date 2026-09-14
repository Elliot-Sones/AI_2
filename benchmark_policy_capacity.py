#!/usr/bin/env python3
"""GPU-only SB3 policy capacity diagnostic for the AI_2 PPO policy shape.

This is a finite compute-scaling microbenchmark. It measures fresh SB3
ActorCriticPolicy instances on synthetic Box observations/actions and labels the
backward pass as a PPO-like synthetic update, not an SB3 training ceiling and
not human-quality evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


DEFAULT_OBS_DIM = 256
DEFAULT_ACTION_DIM = 10
DEFAULT_ARCHITECTURES = ([512, 512, 256], [1024, 1024, 1024], [1536, 1536, 1536])
DEFAULT_INFERENCE_BATCHES = [64, 128, 256, 512]
EXPECTED_PARAM_COUNTS = {
    (512, 512, 256): 1_053_973,
    (1024, 1024, 1024): 4_736_021,
    (1536, 1536, 1536): 10_249_749,
}


def count_actor_critic_policy_params(
    *, obs_dim: int, action_dim: int, net_arch: list[int]
) -> int:
    """Count SB3 ActorCriticPolicy MLP params for Box obs and Box actions.

    SB3 builds separate policy/value MLP extractors for a plain list net_arch.
    Continuous Box actions add an action mean layer plus one learned log_std per
    action dimension. The flatten extractor has no parameters for this input.
    """

    if obs_dim <= 0 or action_dim <= 0:
        raise ValueError("obs_dim and action_dim must be positive")
    if not net_arch or any(width <= 0 for width in net_arch):
        raise ValueError("net_arch must contain positive layer widths")

    def mlp_params(output_width: int) -> int:
        total = 0
        width_in = obs_dim
        for width_out in net_arch:
            total += width_in * width_out + width_out
            width_in = width_out
        total += width_in * output_width + output_width
        return total

    policy_branch = mlp_params(action_dim)
    value_branch = mlp_params(1)
    log_std = action_dim
    return policy_branch + value_branch + log_std


def parse_int_list(raw: str, *, label: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{label} must contain positive integers")
    return values


def parse_architectures(raw: str) -> list[list[int]]:
    return [parse_int_list(part, label="architecture") for part in raw.split(";") if part.strip()]


def architecture_record(net_arch: list[int], *, obs_dim: int, action_dim: int) -> dict[str, Any]:
    arch_tuple = tuple(net_arch)
    params = count_actor_critic_policy_params(
        obs_dim=obs_dim,
        action_dim=action_dim,
        net_arch=net_arch,
    )
    return {
        "net_arch": list(net_arch),
        "param_count": params,
        "expected_param_count": EXPECTED_PARAM_COUNTS.get(arch_tuple),
        "param_formula": "separate pi/vf MLPs + continuous-action mean head + value head + log_std",
    }


def build_plan(
    *,
    architectures: list[list[int]] | None = None,
    batches: list[int] | None = None,
    obs_dim: int = DEFAULT_OBS_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    warmup_calls: int = 20,
    measured_calls: int = 200,
    update_batch_size: int = 8192,
    update_warmup: int = 5,
    update_iterations: int = 20,
    seed: int = 20260906,
    device: str = "cuda",
) -> dict[str, Any]:
    arches = architectures or [list(item) for item in DEFAULT_ARCHITECTURES]
    inference_batches = batches or list(DEFAULT_INFERENCE_BATCHES)
    return {
        "label": "policy_capacity_gpu_only",
        "diagnostic_scope": "GPU-only policy compute scaling; synthetic observations/actions.",
        "no_cpu_fallback": True,
        "device": device,
        "seed": seed,
        "inference_policy_mode": "stochastic",
        "obs_space": {"type": "Box", "shape": [obs_dim]},
        "action_space": {"type": "Box", "shape": [action_dim]},
        "architectures": [
            architecture_record(arch, obs_dim=obs_dim, action_dim=action_dim) for arch in arches
        ],
        "inference_batches": inference_batches,
        "warmup_calls": warmup_calls,
        "measured_calls": measured_calls,
        "update_batch_size": update_batch_size,
        "update_warmup": update_warmup,
        "update_iterations": update_iterations,
        "notes": [
            "PPO-like evaluate_actions+loss+backward+optimizer is a synthetic update diagnostic.",
            "This does not step AI_2 environments, collect SB3 rollouts, resume checkpoints, or assess policy quality.",
            "Inference uses stochastic policy sampling to match rollout collection policy calls.",
            "Device-resident and CPU-roundtrip inference are reported separately.",
            "Finite-output validation runs outside timed loops after the timing boundary sync.",
        ],
    }


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def git_record(root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(run_git(["status", "--short"])),
    }


def provenance(root: Path) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_record(root),
        "versions": {
            "torch": _package_version("torch"),
            "stable_baselines3": _package_version("stable-baselines3"),
            "gymnasium": _package_version("gymnasium"),
            "numpy": _package_version("numpy"),
        },
    }


@dataclass
class Runtime:
    torch: Any
    np: Any
    gym: Any
    ActorCriticPolicy: Any


def require_gpu_runtime(device: str) -> Runtime:
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from stable_baselines3.common.policies import ActorCriticPolicy
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_policy_capacity requires torch, gymnasium, and stable_baselines3 for measurement"
        ) from exc

    if not device.startswith("cuda"):
        raise RuntimeError(f"device must be CUDA-only; got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU fallback")
    selected_device = torch.device(device)
    torch.cuda.set_device(torch.cuda.current_device() if selected_device.index is None else selected_device.index)
    torch.empty(1, device=device)
    return Runtime(torch=torch, np=np, gym=gym, ActorCriticPolicy=ActorCriticPolicy)


def set_runtime_seed(rt: Runtime, seed: int) -> None:
    random.seed(seed)
    rt.np.random.seed(seed)
    rt.torch.manual_seed(seed)
    rt.torch.cuda.manual_seed_all(seed)


def build_policy(rt: Runtime, *, net_arch: list[int], obs_dim: int, action_dim: int, device: str):
    obs_space = rt.gym.spaces.Box(
        low=-10.0,
        high=10.0,
        shape=(obs_dim,),
        dtype=rt.np.float32,
    )
    action_space = rt.gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(action_dim,),
        dtype=rt.np.float32,
    )
    policy = rt.ActorCriticPolicy(
        obs_space,
        action_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=list(net_arch),
    )
    policy.to(device)
    actual_params = sum(param.numel() for param in policy.parameters())
    expected_params = count_actor_critic_policy_params(
        obs_dim=obs_dim,
        action_dim=action_dim,
        net_arch=list(net_arch),
    )
    if actual_params != expected_params:
        raise RuntimeError(
            f"SB3 parameter count mismatch for {net_arch}: actual {actual_params}, formula {expected_params}"
        )
    return policy


def sync(rt: Runtime) -> None:
    rt.torch.cuda.synchronize()


def rate_record(
    *,
    label: str,
    calls: int,
    batch_size: int,
    seconds: float,
    peak_memory_bytes: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples = calls * batch_size
    record = {
        "label": label,
        "calls": calls,
        "batch_size": batch_size,
        "samples": samples,
        "seconds": seconds,
        "calls_per_second": calls / seconds if seconds else math.inf,
        "samples_per_second": samples / seconds if seconds else math.inf,
        "mean_ms_per_call": (seconds / calls) * 1000.0 if calls else None,
        "peak_memory_bytes": peak_memory_bytes,
    }
    if extra:
        record.update(extra)
    return record


def assert_finite_tensor(rt: Runtime, label: str, tensor: Any) -> None:
    if not bool(rt.torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"non-finite tensor in {label}")


def assert_finite_outputs(rt: Runtime, label: str, outputs: tuple[Any, Any, Any]) -> None:
    actions, values, log_prob = outputs
    assert_finite_tensor(rt, f"{label}_actions", actions)
    assert_finite_tensor(rt, f"{label}_values", values)
    assert_finite_tensor(rt, f"{label}_log_prob", log_prob)


def measure_inference(
    rt: Runtime,
    policy: Any,
    *,
    batch_size: int,
    obs_dim: int,
    warmup_calls: int,
    measured_calls: int,
    device: str,
) -> list[dict[str, Any]]:
    torch = rt.torch
    policy.eval()
    device_obs = torch.randn(batch_size, obs_dim, device=device)
    cpu_obs = torch.randn(batch_size, obs_dim, device="cpu")
    results = []

    def policy_forward(obs: Any) -> tuple[Any, Any, Any]:
        with torch.inference_mode():
            return policy(obs, deterministic=False)

    last_outputs = None
    for _ in range(warmup_calls):
        last_outputs = policy_forward(device_obs)
    sync(rt)
    assert_finite_outputs(rt, "inference_device_resident_warmup", last_outputs)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(measured_calls):
        last_outputs = policy_forward(device_obs)
    sync(rt)
    seconds = time.perf_counter() - start
    assert_finite_outputs(rt, "inference_device_resident_measured", last_outputs)
    results.append(
        rate_record(
            label="inference_device_resident",
            calls=measured_calls,
            batch_size=batch_size,
            seconds=seconds,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            extra={
                "policy_mode": "stochastic",
                "validation": "captured_outputs_after_boundary_sync",
            },
        )
    )

    last_outputs = None
    for _ in range(warmup_calls):
        last_outputs = tuple(tensor.cpu() for tensor in policy_forward(cpu_obs.to(device)))
    sync(rt)
    assert_finite_outputs(rt, "inference_cpu_roundtrip_warmup", last_outputs)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(measured_calls):
        last_outputs = tuple(tensor.cpu() for tensor in policy_forward(cpu_obs.to(device)))
    sync(rt)
    seconds = time.perf_counter() - start
    assert_finite_outputs(rt, "inference_cpu_roundtrip_measured", last_outputs)
    results.append(
        rate_record(
            label="inference_cpu_roundtrip",
            calls=measured_calls,
            batch_size=batch_size,
            seconds=seconds,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            extra={
                "policy_mode": "stochastic",
                "validation": "captured_outputs_after_boundary_sync",
            },
        )
    )
    return results


def synthetic_ppo_loss(
    rt: Runtime,
    policy: Any,
    obs: Any,
    actions: Any,
    old_log_prob: Any,
    advantages: Any,
    returns: Any,
    *,
    validate: bool = False,
) -> Any:
    values, log_prob, entropy = policy.evaluate_actions(obs, actions)
    if validate:
        assert_finite_tensor(rt, "update_values", values)
        assert_finite_tensor(rt, "update_log_prob", log_prob)
        if entropy is not None:
            assert_finite_tensor(rt, "update_entropy", entropy)
    ratio = rt.torch.exp(log_prob - old_log_prob)
    clipped = rt.torch.clamp(ratio, 0.8, 1.2)
    policy_loss = -rt.torch.min(advantages * ratio, advantages * clipped).mean()
    value_loss = rt.torch.nn.functional.mse_loss(values.flatten(), returns)
    entropy_term = entropy.mean() if entropy is not None else -log_prob.mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_term
    if validate:
        assert_finite_tensor(rt, "synthetic_update_loss", loss)
    return loss


def measure_synthetic_update(
    rt: Runtime,
    policy: Any,
    *,
    batch_size: int,
    obs_dim: int,
    action_dim: int,
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    torch = rt.torch
    policy.train()
    obs = torch.randn(batch_size, obs_dim, device=device)
    actions = torch.empty(batch_size, action_dim, device=device).uniform_(-1.0, 1.0)
    with torch.no_grad():
        _, old_log_prob, _ = policy.evaluate_actions(obs, actions)
        old_log_prob = old_log_prob.detach()
    advantages = torch.randn_like(old_log_prob)
    returns = torch.randn_like(old_log_prob)
    assert_finite_tensor(rt, "synthetic_update_old_log_prob", old_log_prob)

    last_loss = None
    last_grad_norm = None
    for _ in range(warmup):
        policy.optimizer.zero_grad(set_to_none=True)
        loss = synthetic_ppo_loss(
            rt,
            policy,
            obs,
            actions,
            old_log_prob,
            advantages,
            returns,
            validate=False,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        policy.optimizer.step()
    sync(rt)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iterations):
        policy.optimizer.zero_grad(set_to_none=True)
        loss = synthetic_ppo_loss(
            rt,
            policy,
            obs,
            actions,
            old_log_prob,
            advantages,
            returns,
            validate=False,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        policy.optimizer.step()
        last_loss = loss.detach()
        last_grad_norm = grad_norm.detach()
    sync(rt)
    seconds = time.perf_counter() - start
    assert_finite_tensor(rt, "synthetic_update_loss", last_loss)
    assert_finite_tensor(rt, "synthetic_update_grad_norm", last_grad_norm)
    return rate_record(
        label="synthetic_update_evaluate_actions_backward_optimizer",
        calls=iterations,
        batch_size=batch_size,
        seconds=seconds,
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
        extra={
            "updates": iterations,
            "updates_per_second": iterations / seconds if seconds else math.inf,
            "warmup_updates": warmup,
            "advantage_return_generation": "precreated_outside_timed_loop",
            "validation": "loss_and_grad_norm_after_boundary_sync",
            "last_loss": float(last_loss.item()) if last_loss is not None else None,
            "last_grad_norm": float(last_grad_norm.item()) if last_grad_norm is not None else None,
            "not_sb3_training_ceiling": True,
        },
    )


def run_measurement(plan: dict[str, Any], *, root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    rt = require_gpu_runtime(plan["device"])
    set_runtime_seed(rt, plan["seed"])
    results = []
    for arch in plan["architectures"]:
        policy = build_policy(
            rt,
            net_arch=arch["net_arch"],
            obs_dim=plan["obs_space"]["shape"][0],
            action_dim=plan["action_space"]["shape"][0],
            device=plan["device"],
        )
        phases = []
        for batch_size in plan["inference_batches"]:
            phases.extend(
                measure_inference(
                    rt,
                    policy,
                    batch_size=batch_size,
                    obs_dim=plan["obs_space"]["shape"][0],
                    warmup_calls=plan["warmup_calls"],
                    measured_calls=plan["measured_calls"],
                    device=plan["device"],
                )
            )
        phases.append(
            measure_synthetic_update(
                rt,
                policy,
                batch_size=plan["update_batch_size"],
                obs_dim=plan["obs_space"]["shape"][0],
                action_dim=plan["action_space"]["shape"][0],
                warmup=plan["update_warmup"],
                iterations=plan["update_iterations"],
                device=plan["device"],
            )
        )
        results.append({**arch, "phases": phases})
        del policy
        rt.torch.cuda.empty_cache()

    total_seconds = time.perf_counter() - started
    return {
        "mode": "measurement",
        "plan": plan,
        "provenance": provenance(root),
        "total_seconds": total_seconds,
        "cuda": {
            "device_name": rt.torch.cuda.get_device_name(plan["device"]),
            "device_count": rt.torch.cuda.device_count(),
            "torch_cuda": rt.torch.version.cuda,
            "cudnn": rt.torch.backends.cudnn.version(),
        },
        "results": results,
    }


def write_payload(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GPU-only finite SB3 ActorCriticPolicy compute diagnostic. "
            "This is not full AI_2 training or a policy-quality benchmark."
        )
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument("--dry-run", action="store_true", help="Write plan/provenance only; no Torch/SB3 imports")
    parser.add_argument(
        "--architectures",
        default=";".join(",".join(str(width) for width in arch) for arch in DEFAULT_ARCHITECTURES),
        help="Semicolon-separated MLP architectures, e.g. '512,512,256;1024,1024,1024'",
    )
    parser.add_argument("--batches", default="64,128,256,512", help="Comma-separated inference batch sizes")
    parser.add_argument("--obs-dim", type=int, default=DEFAULT_OBS_DIM)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--warmup-calls", type=int, default=20)
    parser.add_argument("--measured-calls", type=int, default=200)
    parser.add_argument("--update-batch-size", type=int, default=8192)
    parser.add_argument("--update-warmup", type=int, default=5)
    parser.add_argument("--update-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda", help="CUDA device string; CPU fallback is refused")
    return parser


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise SystemExit(f"{name} must be positive")


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    args = build_arg_parser().parse_args(argv)
    architectures = parse_architectures(args.architectures)
    batches = parse_int_list(args.batches, label="batches")
    for name in (
        "obs_dim",
        "action_dim",
        "warmup_calls",
        "measured_calls",
        "update_batch_size",
        "update_warmup",
        "update_iterations",
    ):
        validate_positive(name, int(getattr(args, name)))

    plan = build_plan(
        architectures=architectures,
        batches=batches,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        warmup_calls=args.warmup_calls,
        measured_calls=args.measured_calls,
        update_batch_size=args.update_batch_size,
        update_warmup=args.update_warmup,
        update_iterations=args.update_iterations,
        seed=args.seed,
        device=args.device,
    )
    if args.dry_run:
        payload = {"mode": "dry_run", "plan": plan, "provenance": provenance(root)}
    else:
        payload = run_measurement(plan, root=root)
    write_payload(payload, args.output)
    print(f"{plan['label']} {payload['mode']} wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
