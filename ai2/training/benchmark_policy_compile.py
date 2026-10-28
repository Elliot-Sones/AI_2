#!/usr/bin/env python3
"""E4 GPU-only static Gaussian ActorCriticPolicy compile diagnostic.

This is a finite policy-forward microbenchmark for the current AI_2 PPO policy
shape. It compares eager against ``torch.compile(fullgraph=True, dynamic=False)``
for a tensor-only Gaussian actor-critic helper ``(obs, eps) -> (actions, values,
log_prob)``. Unsupported compile is reported explicitly and never treated as a
successful fallback.
"""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import source_path

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = PROJECT_ROOT
DEFAULT_NAV_ROOT = (
    Path("/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905")
    / "baselines"
    / "navigation-repair-20260905"
)
DEFAULT_SOURCE = DEFAULT_NAV_ROOT / "payload" / "variants" / "corrected"
DEFAULT_CONFIG = DEFAULT_SOURCE / "control.yaml"
DEFAULT_CHECKPOINT = DEFAULT_NAV_ROOT / "payload" / "checkpoints" / "selected_16001056.zip"
DEFAULT_BATCHES = [64, 256]
DEFAULT_NET_ARCH = [512, 512, 256]
DEFAULT_OBS_DIM = 256
DEFAULT_ACTION_DIM = 10
DEFAULT_ITERATIONS = 200
DEFAULT_WARMUP = 20
EXPECTED_PARAM_COUNTS = {(512, 512, 256): 1_053_973}
DEFAULT_TOLERANCE = 1e-5


def count_actor_critic_policy_params(*, obs_dim: int, action_dim: int, net_arch: list[int]) -> int:
    if obs_dim <= 0 or action_dim <= 0:
        raise ValueError("obs_dim and action_dim must be positive")
    if not net_arch or any(width <= 0 for width in net_arch):
        raise ValueError("net_arch must contain positive widths")

    def branch_params(output_dim: int) -> int:
        total = 0
        width_in = obs_dim
        for width in net_arch:
            total += width_in * width + width
            width_in = width
        return total + width_in * output_dim + output_dim

    return branch_params(action_dim) + branch_params(1) + action_dim


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if dataclasses.is_dataclass(value):
            return json_safe(dataclasses.asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    return {"path": str(resolved), "exists": True, "size": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_plan(
    *,
    device: str = "cuda",
    batches: list[int] | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
    obs_dim: int = DEFAULT_OBS_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    net_arch: list[int] | None = None,
    seed: int = 20260906,
) -> dict[str, Any]:
    if not str(device).startswith("cuda"):
        raise ValueError("E4 policy compile diagnostic is GPU-only; refusing CPU fallback")
    batch_values = list(DEFAULT_BATCHES if batches is None else batches)
    arch = list(DEFAULT_NET_ARCH if net_arch is None else net_arch)
    if not batch_values or any(int(batch) <= 0 for batch in batch_values):
        raise ValueError("batch sizes must be positive")
    if int(iterations) < 200:
        raise ValueError("iterations must be >= 200 for the E4 timing loop")
    if int(warmup) < 0:
        raise ValueError("warmup must be nonnegative")
    param_count = count_actor_critic_policy_params(obs_dim=obs_dim, action_dim=action_dim, net_arch=arch)
    return {
        "label": "e4_policy_compile_static_gaussian",
        "device": str(device),
        "gpu_only": True,
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "net_arch": arch,
        "param_count_formula": param_count,
        "expected_param_count": EXPECTED_PARAM_COUNTS.get(tuple(arch)),
        "batches": [int(batch) for batch in batch_values],
        "iterations": int(iterations),
        "warmup": int(warmup),
        "seed": int(seed),
        "compile": {"fullgraph": True, "dynamic": False},
        "rng_generation": "inside_timed_loop",
        "notes": [
            "Tensor-only helper implements SB3 2.5 diagonal Gaussian policy math.",
            "Eager and compiled arms generate fresh epsilon inside timed loops for equal semantics.",
            "Compile cost is recorded separately from timed steady-state calls.",
            "CPU-roundtrip phases are labeled separately from device-resident phases.",
        ],
    }


def rate_record(
    *,
    label: str,
    arm: str,
    batch_size: int,
    iterations: int,
    seconds: float,
    rng_included: bool,
    cpu_roundtrip: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples = int(batch_size) * int(iterations)
    record = {
        "label": label,
        "arm": arm,
        "batch_size": int(batch_size),
        "iterations": int(iterations),
        "samples": samples,
        "seconds": float(seconds),
        "iterations_per_second": float(iterations) / float(seconds) if seconds else math.inf,
        "samples_per_second": samples / float(seconds) if seconds else math.inf,
        "mean_ms_per_iteration": (float(seconds) / float(iterations)) * 1000.0 if iterations else None,
        "rng_generation": "inside_timed_loop" if rng_included else "outside_timed_loop",
        "transfer_mode": "cpu_roundtrip" if cpu_roundtrip else "device_resident",
    }
    if extra:
        record.update(extra)
    return record


def tensor_ptr(value: Any) -> int | None:
    if hasattr(value, "data_ptr") and callable(value.data_ptr):
        return int(value.data_ptr())
    if hasattr(value, "untyped_storage"):
        try:
            return int(value.untyped_storage().data_ptr())
        except Exception:
            return None
    return None


def assert_no_storage_alias(named_tensors: dict[str, Any]) -> None:
    seen: dict[int, str] = {}
    for name, tensor in named_tensors.items():
        ptr = tensor_ptr(tensor)
        if ptr is None:
            continue
        if ptr in seen:
            raise RuntimeError(f"tensor storage alias detected: {name} aliases {seen[ptr]}")
        seen[ptr] = name


def clone_output_tuple(outputs: tuple[Any, ...]) -> tuple[Any, ...]:
    cloned = []
    for tensor in outputs:
        if hasattr(tensor, "detach") and callable(tensor.detach):
            tensor = tensor.detach()
        if hasattr(tensor, "clone") and callable(tensor.clone):
            cloned.append(tensor.clone())
        else:
            cloned.append(tensor)
    return tuple(cloned)


def assert_no_cross_call_storage_alias(
    first_outputs: tuple[Any, ...],
    second_outputs: tuple[Any, ...],
    *,
    first_label: str,
    second_label: str,
) -> None:
    seen: dict[int, str] = {}
    for index, tensor in enumerate(first_outputs):
        ptr = tensor_ptr(tensor)
        if ptr is not None:
            seen[ptr] = f"{first_label}[{index}]"
    for index, tensor in enumerate(second_outputs):
        ptr = tensor_ptr(tensor)
        if ptr is not None and ptr in seen:
            raise RuntimeError(f"reused output storage detected: {second_label}[{index}] aliases {seen[ptr]}")


def assert_output_snapshot_unchanged(torch: Any, snapshot: tuple[Any, ...], current: tuple[Any, ...], *, label: str, tol: float = 0.0) -> None:
    if len(snapshot) != len(current):
        raise RuntimeError(f"{label} output arity changed after follow-up call")
    for index, (before, after) in enumerate(zip(snapshot, current)):
        diff = max_abs_diff(torch, before, after)
        if diff > tol:
            raise RuntimeError(f"{label} output overwritten after follow-up call at index {index}: {diff}")


def assert_fresh_action_response(torch: Any, previous_actions: Any, fresh_actions: Any, *, label: str) -> float:
    diff = max_abs_diff(torch, previous_actions, fresh_actions)
    if diff <= 0.0:
        raise RuntimeError(f"{label} fresh epsilon did not change actions")
    return diff


def compile_unsupported_record(exc: BaseException) -> dict[str, Any]:
    return {
        "arm": "compiled",
        "status": "unsupported",
        "fallback_success": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def write_summary(
    output: Path,
    *,
    plan: dict[str, Any],
    phases: list[dict[str, Any]],
    compile_costs: list[dict[str, Any]],
    parity: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": "e4_policy_compile_static_gaussian",
        "status": "completed",
        "plan": plan,
        "phases": phases,
        "compile_costs": compile_costs,
        "parity": parity,
        "provenance": provenance,
    }
    write_json(output / "benchmark_policy_compile.json", payload)
    return payload


def normalize_cuda_device(device: str) -> str:
    raw = str(device)
    if raw == "cuda":
        return "cuda:0"
    if raw.startswith("cuda:"):
        suffix = raw.split(":", 1)[1]
        if suffix.isdigit():
            return raw
        raise RuntimeError("E4 policy compile diagnostic is GPU-only; refusing CPU fallback")
    raise RuntimeError("E4 policy compile diagnostic is GPU-only; refusing CPU fallback")


def require_gpu_runtime(device: str):
    normalized = normalize_cuda_device(device)
    try:
        import numpy as np
        import torch
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("benchmark_policy_compile requires torch and stable_baselines3 at runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU fallback")
    torch.cuda.set_device(torch.device(normalized))
    torch.empty(1, device=normalized)
    return torch, np, PPO


def configure_seed(torch: Any, np: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_policy(torch: Any, PPO: Any, checkpoint: Path, device: str):
    model = PPO.load(str(checkpoint), env=None, device=device)
    policy = model.policy.to(device)
    policy.eval()
    return policy


def static_gaussian_actorcritic(policy: Any, torch: Any) -> Callable[[Any, Any], tuple[Any, Any, Any]]:
    half_log_two_pi = 0.5 * math.log(2.0 * math.pi)

    def helper(obs: Any, eps: Any):
        features = policy.extract_features(obs)
        if getattr(policy, "share_features_extractor", True):
            latent_pi, latent_vf = policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = policy.mlp_extractor.forward_critic(vf_features)
        mean_actions = policy.action_net(latent_pi)
        log_std = policy.log_std.to(mean_actions.device)
        std = torch.exp(log_std)
        actions = mean_actions + eps * std
        values = policy.value_net(latent_vf)
        log_prob = gaussian_log_prob_from_actions(
            torch,
            actions=actions,
            mean_actions=mean_actions,
            log_std=log_std,
            half_log_two_pi=half_log_two_pi,
        )
        return actions, values, log_prob

    return helper


def gaussian_log_prob_from_actions(torch: Any, *, actions: Any, mean_actions: Any, log_std: Any, half_log_two_pi: float):
    std = torch.exp(log_std)
    variance = std**2
    std_log = std.log()
    element_log_prob = -((actions - mean_actions) ** 2) / (2.0 * variance) - std_log - half_log_two_pi
    return element_log_prob.sum(dim=1)


def assert_finite(torch: Any, label: str, tensor: Any) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"non-finite tensor in {label}")


def max_abs_diff(torch: Any, left: Any, right: Any) -> float:
    result = float(torch.max(torch.abs(left - right)).detach().cpu().item())
    if not math.isfinite(result):
        raise RuntimeError("non-finite parity difference")
    return result


def parity_tolerances(label: str, dtype_epsilon: float, base_atol: float) -> tuple[float, float]:
    """Four dtype epsilons, not a scale-blind widening of the absolute gate."""
    return (min(base_atol, 1e-6) if label == "actions" else base_atol, 4 * dtype_epsilon)


def tensor_parity_report(torch: Any, left: Any, right: Any, label: str, *, base_atol: float) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise RuntimeError(f"parity shape/dtype mismatch for {label}")
    assert_finite(torch, f"{label}_left", left)
    assert_finite(torch, f"{label}_right", right)
    finfo = torch.finfo(left.dtype)
    atol, rtol = parity_tolerances(label, float(finfo.eps), base_atol)
    error = torch.abs(left - right)
    scale = torch.maximum(torch.abs(left), torch.abs(right))
    return {
        "max_abs_diff": float(error.max().item()),
        "max_rel_diff": float((error / scale.clamp_min(finfo.tiny)).max().item()),
        "max_normalized_error": float((error / (atol + rtol * scale)).max().item()),
        "max_abs_left": float(left.abs().max().item()),
        "max_abs_right": float(right.abs().max().item()),
        "dtype": str(left.dtype),
        "atol": atol,
        "rtol": rtol,
        "rule": "abs(left-right) <= atol + rtol * max(abs(left), abs(right))",
    }


def require_parity_reports(reports: dict[str, dict[str, Any]]) -> None:
    for label, report in reports.items():
        error = float(report["max_normalized_error"])
        if not math.isfinite(error) or error > 1.0:
            raise RuntimeError(f"tensor parity failed for {label}: {report}")


def validate_helper_against_policy(torch: Any, policy: Any, helper: Callable[[Any, Any], tuple[Any, Any, Any]], obs: Any, eps: Any, *, tol: float) -> dict[str, Any]:
    with torch.no_grad():
        zero_eps = torch.zeros_like(eps)
        helper_actions, helper_values, helper_log_prob = helper(obs, zero_eps)
        policy_actions, policy_values, policy_log_prob = policy(obs, deterministic=True)
        deterministic = {
            "actions": tensor_parity_report(torch, helper_actions, policy_actions, "actions", base_atol=tol),
            "values": tensor_parity_report(torch, helper_values, policy_values, "values", base_atol=tol),
            "log_prob": tensor_parity_report(torch, helper_log_prob, policy_log_prob, "log_prob", base_atol=tol),
        }

        actions, values, log_prob = helper(obs, eps)
        eval_values, eval_log_prob, _entropy = policy.evaluate_actions(obs, actions)
        stochastic = {
            "values": tensor_parity_report(torch, values.flatten(), eval_values.flatten(), "values", base_atol=tol),
            "log_prob": tensor_parity_report(torch, log_prob, eval_log_prob, "log_prob", base_atol=tol),
        }
        for key, tensor in {"actions": actions, "values": values, "log_prob": log_prob}.items():
            assert_finite(torch, f"helper_{key}", tensor)
        assert_no_storage_alias({"obs": obs, "eps": eps, "actions": actions, "values": values, "log_prob": log_prob})
        output_snapshot = clone_output_tuple((actions, values, log_prob))
        fresh_eps = torch.randn_like(eps)
        fresh_outputs = helper(obs, fresh_eps)
        for key, tensor in zip(("actions", "values", "log_prob"), fresh_outputs):
            assert_finite(torch, f"helper_fresh_{key}", tensor)
        fresh_action_diff = assert_fresh_action_response(torch, actions, fresh_outputs[0], label="helper")
        assert_no_cross_call_storage_alias((actions, values, log_prob), fresh_outputs, first_label="helper", second_label="helper_fresh")
        assert_output_snapshot_unchanged(torch, output_snapshot, (actions, values, log_prob), label="helper")
        perm = torch.randperm(obs.shape[0], device=obs.device)
        p_actions, p_values, p_log_prob = helper(obs.index_select(0, perm), eps.index_select(0, perm))
        permutation = {
            "actions": tensor_parity_report(torch, p_actions, actions.index_select(0, perm), "actions", base_atol=tol),
            "values": tensor_parity_report(torch, p_values, values.index_select(0, perm), "values", base_atol=tol),
            "log_prob": tensor_parity_report(torch, p_log_prob, log_prob.index_select(0, perm), "log_prob", base_atol=tol),
        }
    for reports in (deterministic, stochastic, permutation):
        require_parity_reports(reports)
    maxima = [report["max_abs_diff"] for reports in (deterministic, stochastic, permutation) for report in reports.values()]
    return {
        "deterministic_zero_eps_vs_policy": deterministic,
        "stochastic_action_evaluate_actions": stochastic,
        "permutation_equivariance": permutation,
        "fresh_epsilon_action_diff": fresh_action_diff,
        "max_abs_diff": max(maxima) if maxima else 0.0,
        "base_atol": tol,
        "relative_tolerance_dtype_eps": 4,
    }


def compile_helper(torch: Any, helper: Callable[[Any, Any], tuple[Any, Any, Any]], obs: Any, eps: Any, device: str) -> tuple[Callable[[Any, Any], tuple[Any, Any, Any]] | None, dict[str, Any]]:
    sync = torch.cuda.synchronize
    started = time.perf_counter()
    try:
        with torch.no_grad():
            compiled = torch.compile(helper, fullgraph=True, dynamic=False)
            compiled(obs, eps)
        sync()
    except Exception as exc:
        return None, {**compile_unsupported_record(exc), "seconds": time.perf_counter() - started, "device": device}
    return compiled, {"arm": "compiled", "status": "compiled", "fallback_success": False, "seconds": time.perf_counter() - started, "device": device}


def measure_arm(torch: Any, helper: Callable[[Any, Any], tuple[Any, Any, Any]], obs: Any, *, batch_size: int, action_dim: int, iterations: int, warmup: int, device: str, arm: str, cpu_roundtrip: bool) -> dict[str, Any]:
    sync = torch.cuda.synchronize
    base_obs = obs.detach().clone()
    cpu_obs = base_obs.detach().cpu() if cpu_roundtrip else None
    with torch.no_grad():
        for _ in range(warmup):
            eps = torch.randn(batch_size, action_dim, device=device)
            if cpu_roundtrip:
                tuple(item.cpu() for item in helper(cpu_obs.to(device), eps))
            else:
                helper(base_obs, eps)
    sync()
    torch.cuda.reset_peak_memory_stats()
    last = None
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            eps = torch.randn(batch_size, action_dim, device=device)
            if cpu_roundtrip:
                last = tuple(item.cpu() for item in helper(cpu_obs.to(device), eps))
            else:
                last = helper(base_obs, eps)
    sync()
    seconds = time.perf_counter() - started
    for index, tensor in enumerate(last or ()):
        target = tensor.to(device) if getattr(tensor, "device", None).type == "cpu" else tensor
        assert_finite(torch, f"{arm}_{'cpu_roundtrip' if cpu_roundtrip else 'device'}_{index}", target)
    return rate_record(
        label=f"{arm}_{'cpu_roundtrip' if cpu_roundtrip else 'device_resident'}",
        arm=arm,
        batch_size=batch_size,
        iterations=iterations,
        seconds=seconds,
        rng_included=True,
        cpu_roundtrip=cpu_roundtrip,
        extra={"peak_memory_bytes": int(torch.cuda.max_memory_allocated())},
    )


def provenance(args: argparse.Namespace, actual_param_count: int) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "pid": os.getpid(),
        "versions": {
            "torch": package_version("torch"),
            "stable_baselines3": package_version("stable-baselines3"),
            "numpy": package_version("numpy"),
        },
        "files": {
            "checkpoint": file_record(args.checkpoint),
            "config": file_record(args.config),
            "source_train": file_record(source_path(args.source, "train.py")),
            "harness": file_record(Path(__file__)),
        },
        "actual_policy_param_count": actual_param_count,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.device = normalize_cuda_device(args.device)
    plan = build_plan(
        device=args.device,
        batches=args.batches,
        iterations=args.iterations,
        warmup=args.warmup,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        net_arch=args.net_arch,
        seed=args.seed,
    )
    torch, np, PPO = require_gpu_runtime(args.device)
    configure_seed(torch, np, args.seed)
    policy = load_policy(torch, PPO, args.checkpoint, args.device)
    actual_param_count = sum(param.numel() for param in policy.parameters())
    if actual_param_count != plan["param_count_formula"]:
        raise RuntimeError(f"checkpoint policy parameter count {actual_param_count} != plan formula {plan['param_count_formula']}")
    helper = static_gaussian_actorcritic(policy, torch)
    phases: list[dict[str, Any]] = []
    compile_costs: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    for batch_size in plan["batches"]:
        obs = torch.randn(batch_size, plan["obs_dim"], device=args.device)
        eps = torch.randn(batch_size, plan["action_dim"], device=args.device)
        parity.append({"batch_size": batch_size, "eager": validate_helper_against_policy(torch, policy, helper, obs, eps, tol=args.tolerance)})
        compiled, compile_record = compile_helper(torch, helper, obs, eps, args.device)
        compile_costs.append({"batch_size": batch_size, **compile_record})
        phases.append(
            measure_arm(
                torch,
                helper,
                obs,
                batch_size=batch_size,
                action_dim=plan["action_dim"],
                iterations=plan["iterations"],
                warmup=plan["warmup"],
                device=args.device,
                arm="eager",
                cpu_roundtrip=False,
            )
        )
        phases.append(
            measure_arm(
                torch,
                helper,
                obs,
                batch_size=batch_size,
                action_dim=plan["action_dim"],
                iterations=plan["iterations"],
                warmup=plan["warmup"],
                device=args.device,
                arm="eager",
                cpu_roundtrip=True,
            )
        )
        if compiled is None:
            phases.append({"label": "compiled_device_resident", "arm": "compiled", "batch_size": batch_size, "status": "unsupported", "fallback_success": False})
            continue
        with torch.no_grad():
            compiled_outputs = compiled(obs, eps)
            eager_outputs = helper(obs, eps)
        for name, tensor in zip(("compiled_actions", "compiled_values", "compiled_log_prob"), compiled_outputs):
            assert_finite(torch, name, tensor)
        for name, tensor in zip(("eager_actions", "eager_values", "eager_log_prob"), eager_outputs):
            assert_finite(torch, name, tensor)
        compile_parity = {name: tensor_parity_report(torch, left, right, name, base_atol=args.tolerance) for name, left, right in zip(("actions", "values", "log_prob"), compiled_outputs, eager_outputs)}
        assert_no_storage_alias({"obs": obs, "eps": eps, "actions": compiled_outputs[0], "values": compiled_outputs[1], "log_prob": compiled_outputs[2]})
        require_parity_reports(compile_parity)
        parity[-1]["compiled_vs_eager"] = compile_parity
        compiled_snapshot = clone_output_tuple(compiled_outputs)
        with torch.no_grad():
            compiled_fresh = compiled(obs, torch.randn_like(eps))
        for name, tensor in zip(("compiled_fresh_actions", "compiled_fresh_values", "compiled_fresh_log_prob"), compiled_fresh):
            assert_finite(torch, name, tensor)
        parity[-1]["compiled_fresh_epsilon_action_diff"] = assert_fresh_action_response(
            torch,
            compiled_outputs[0],
            compiled_fresh[0],
            label="compiled",
        )
        assert_no_cross_call_storage_alias(compiled_outputs, compiled_fresh, first_label="compiled", second_label="compiled_fresh")
        assert_output_snapshot_unchanged(torch, compiled_snapshot, compiled_outputs, label="compiled")
        perm = torch.randperm(obs.shape[0], device=obs.device)
        with torch.no_grad():
            compiled_perm = compiled(obs.index_select(0, perm), eps.index_select(0, perm))
        compiled_permutation = {
            "actions": tensor_parity_report(torch, compiled_perm[0], compiled_outputs[0].index_select(0, perm), "actions", base_atol=args.tolerance),
            "values": tensor_parity_report(torch, compiled_perm[1], compiled_outputs[1].index_select(0, perm), "values", base_atol=args.tolerance),
            "log_prob": tensor_parity_report(torch, compiled_perm[2], compiled_outputs[2].index_select(0, perm), "log_prob", base_atol=args.tolerance),
        }
        require_parity_reports(compiled_permutation)
        parity[-1]["compiled_permutation_equivariance"] = compiled_permutation
        parity[-1]["compiled_direct_sb3"] = validate_helper_against_policy(
            torch, policy, compiled, obs, eps, tol=args.tolerance
        )
        phases.append(
            measure_arm(
                torch,
                compiled,
                obs,
                batch_size=batch_size,
                action_dim=plan["action_dim"],
                iterations=plan["iterations"],
                warmup=plan["warmup"],
                device=args.device,
                arm="compiled",
                cpu_roundtrip=False,
            )
        )
        phases.append(
            measure_arm(
                torch,
                compiled,
                obs,
                batch_size=batch_size,
                action_dim=plan["action_dim"],
                iterations=plan["iterations"],
                warmup=plan["warmup"],
                device=args.device,
                arm="compiled",
                cpu_roundtrip=True,
            )
        )
    return write_summary(args.output, plan=plan, phases=phases, compile_costs=compile_costs, parity=parity, provenance=provenance(args, actual_param_count))


def parse_net_arch(raw: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--net-arch must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--net-arch must contain positive integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--obs-dim", type=int, default=DEFAULT_OBS_DIM)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--net-arch", type=parse_net_arch, default=list(DEFAULT_NET_ARCH))
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.source = args.source.resolve()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    build_plan(
        device=args.device,
        batches=args.batches,
        iterations=args.iterations,
        warmup=args.warmup,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        net_arch=args.net_arch,
        seed=args.seed,
    )
    if args.tolerance <= 0 or not math.isfinite(args.tolerance):
        raise SystemExit("--tolerance must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    print(json.dumps({"output": str(args.output / "benchmark_policy_compile.json"), "phases": len(payload["phases"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
