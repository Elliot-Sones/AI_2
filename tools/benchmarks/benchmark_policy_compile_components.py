#!/usr/bin/env python3
"""E4 component isolation diagnostic for policy compile numerical drift.

This script is GPU-only at runtime. It splits the static Gaussian policy helper
into deterministic subgraphs so a failed full-helper compile can be attributed
before any tolerance change or campaign optimization is accepted.
"""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import ai2.training.benchmark_policy_compile as e4


def _coords_from_flat(shape: tuple[int, ...], flat_index: int) -> list[int]:
    coords = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        size = int(shape[axis])
        coords[axis] = flat_index % size
        flat_index //= size
    return coords


def _value_at(tensor: Any, coords: list[int]) -> float:
    value = tensor
    for coord in coords:
        value = value[coord]
    return float(value.detach().cpu().item() if hasattr(value, "detach") else value)


def located_parity_report(
    torch: Any,
    left: Any,
    right: Any,
    label: str,
    *,
    base_atol: float,
    operands: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = e4.tensor_parity_report(torch, left, right, label, base_atol=base_atol)
    finfo = torch.finfo(left.dtype)
    error = torch.abs(left - right)
    scale = torch.maximum(torch.abs(left), torch.abs(right))
    denom = report["atol"] + report["rtol"] * scale
    normalized = error / denom.clamp_min(finfo.tiny)
    flat = int(torch.argmax(normalized).detach().cpu().item())
    coords = _coords_from_flat(tuple(int(item) for item in left.shape), flat)
    worst = {
        "flat_index": flat,
        "coordinates": coords,
        "left": _value_at(left, coords),
        "right": _value_at(right, coords),
        "abs_diff": _value_at(error, coords),
        "scale": _value_at(scale, coords),
        "denominator": _value_at(denom, coords),
        "normalized_error": _value_at(normalized, coords),
    }
    if operands:
        worst["operands"] = {name: _value_at(tensor, coords) for name, tensor in operands.items()}
    report["worst"] = worst
    return report


def _as_tuple(outputs: Any) -> tuple[Any, ...]:
    return outputs if isinstance(outputs, tuple) else (outputs,)


def compare_component(
    torch: Any,
    *,
    name: str,
    output_names: tuple[str, ...],
    eager_call: Callable[[], Any],
    compiled_call: Callable[[], Any],
    tolerance: float,
    operand_maps: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    reports: dict[str, Any] = {}
    errors: list[str] = []
    try:
        with torch.no_grad():
            eager_outputs = _as_tuple(eager_call())
            compiled_outputs = _as_tuple(compiled_call())
    except Exception as exc:
        return {
            "name": name,
            "status": "error",
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "reports": reports,
        }
    if len(eager_outputs) != len(compiled_outputs) or len(eager_outputs) != len(output_names):
        errors.append(f"output arity mismatch: eager={len(eager_outputs)} compiled={len(compiled_outputs)} names={len(output_names)}")
    for label, eager_tensor, compiled_tensor in zip(output_names, eager_outputs, compiled_outputs):
        try:
            reports[label] = located_parity_report(
                torch,
                compiled_tensor,
                eager_tensor,
                label,
                base_atol=tolerance,
                operands=(operand_maps or {}).get(label),
            )
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    try:
        if errors:
            raise RuntimeError("; ".join(errors))
        e4.require_parity_reports(reports)
    except Exception as exc:
        return {
            "name": name,
            "status": "failed",
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "reports": reports,
        }
    return {"name": name, "status": "passed", "seconds": time.perf_counter() - started, "reports": reports}


def compile_callable(torch: Any, fn: Callable[..., Any], example_args: tuple[Any, ...], device: str) -> tuple[Callable[..., Any] | None, dict[str, Any]]:
    started = time.perf_counter()
    try:
        with torch.no_grad():
            compiled = torch.compile(fn, fullgraph=True, dynamic=False)
            compiled(*example_args)
        torch.cuda.synchronize()
    except Exception as exc:
        return None, {**e4.compile_unsupported_record(exc), "seconds": time.perf_counter() - started, "device": device}
    return compiled, {"arm": "compiled", "status": "compiled", "fallback_success": False, "seconds": time.perf_counter() - started, "device": device}


def summarize_results(components: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [item["name"] for item in components if item.get("status") != "passed"]
    return {
        "status": "passed" if not failed else "failed",
        "component_count": len(components),
        "passed_components": sum(1 for item in components if item.get("status") == "passed"),
        "failed_components": failed,
    }


def write_summary(output: Path, *, plan: dict[str, Any], components: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": "e4_policy_compile_component_isolation",
        **summarize_results(components),
        "plan": plan,
        "components": components,
        "provenance": provenance,
    }
    e4.write_json(output / "benchmark_policy_compile_components.json", payload)
    return payload


def _build_component_functions(policy: Any, torch: Any):
    half_log_two_pi = 0.5 * math.log(2.0 * math.pi)

    def feature_latents(obs: Any):
        features = policy.extract_features(obs)
        if getattr(policy, "share_features_extractor", True):
            latent_pi, latent_vf = policy.mlp_extractor(features)
            return features, latent_pi, latent_vf
        pi_features, vf_features = features
        return pi_features, vf_features, policy.mlp_extractor.forward_actor(pi_features), policy.mlp_extractor.forward_critic(vf_features)

    def means(obs: Any):
        outputs = feature_latents(obs)
        latent_pi = outputs[-2]
        return policy.action_net(latent_pi)

    def values(obs: Any):
        outputs = feature_latents(obs)
        latent_vf = outputs[-1]
        return policy.value_net(latent_vf)

    def sampling(mean_actions: Any, std: Any, eps: Any):
        return mean_actions + eps * std

    def logprob(actions: Any, mean_actions: Any, log_std: Any):
        return e4.gaussian_log_prob_from_actions(
            torch,
            actions=actions,
            mean_actions=mean_actions,
            log_std=log_std,
            half_log_two_pi=half_log_two_pi,
        )

    def tensor_trunk(obs: Any):
        return means(obs), values(obs)

    return feature_latents, means, values, sampling, logprob, tensor_trunk


def _compile_or_component_error(torch: Any, name: str, fn: Callable[..., Any], args: tuple[Any, ...], device: str) -> tuple[Callable[..., Any] | None, dict[str, Any] | None]:
    compiled, record = compile_callable(torch, fn, args, device)
    if compiled is None:
        return None, {"name": name, "status": "error", "compile": record, "reports": {}, "error": record.get("error")}
    return compiled, record


def _stock_gaussian_from_trunk(torch: Any, trunk_outputs: tuple[Any, Any], eps: Any, log_std: Any, sampling: Callable[..., Any], logprob: Callable[..., Any]):
    mean_actions, values = trunk_outputs
    std = torch.exp(log_std)
    actions = sampling(mean_actions, std, eps)
    return actions, values, logprob(actions, mean_actions, log_std)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.device = e4.normalize_cuda_device(args.device)
    plan = e4.build_plan(device=args.device, batches=args.batches, iterations=200, warmup=0, seed=args.seed)
    torch, np, PPO = e4.require_gpu_runtime(args.device)
    e4.configure_seed(torch, np, args.seed)
    policy = e4.load_policy(torch, PPO, args.checkpoint, args.device)
    actual_param_count = sum(param.numel() for param in policy.parameters())
    if actual_param_count != plan["param_count_formula"]:
        raise RuntimeError(f"checkpoint policy parameter count {actual_param_count} != plan formula {plan['param_count_formula']}")

    feature_latents, means, values, sampling, logprob, tensor_trunk = _build_component_functions(policy, torch)
    components: list[dict[str, Any]] = []
    compile_costs: list[dict[str, Any]] = []
    for batch_size in plan["batches"]:
        obs = torch.randn(batch_size, plan["obs_dim"], device=args.device)
        eps = torch.randn(batch_size, plan["action_dim"], device=args.device)
        with torch.no_grad():
            eager_feature_latents = feature_latents(obs)
            eager_mean = means(obs)
            eager_value = values(obs)
            log_std = policy.log_std.to(eager_mean.device)
            std = torch.exp(log_std)
            eps_times_std = eps * std
            eager_actions = sampling(eager_mean, std, eps)
            eager_log_prob = logprob(eager_actions, eager_mean, log_std)

        specs = [
            (
                "features_latents",
                feature_latents,
                (obs,),
                ("features", "latent_pi", "latent_vf") if len(eager_feature_latents) == 3 else ("pi_features", "vf_features", "latent_pi", "latent_vf"),
                None,
            ),
            ("means", means, (obs,), ("actions",), None),
            ("critic_values", values, (obs,), ("values",), None),
            (
                "sampling_fixed_eager_operands",
                sampling,
                (eager_mean, std, eps),
                ("actions",),
                {"actions": {"mean_actions": eager_mean, "std": std.expand_as(eps), "eps": eps, "eps_times_std": eps_times_std}},
            ),
            ("logprob_fixed_actions", logprob, (eager_actions, eager_mean, log_std), ("log_prob",), None),
            ("mixed_compiled_trunk_stock_gaussian", tensor_trunk, (obs,), ("actions", "values", "log_prob"), None),
        ]
        eager_by_name = {
            "features_latents": lambda: eager_feature_latents,
            "means": lambda: eager_mean,
            "critic_values": lambda: eager_value,
            "sampling_fixed_eager_operands": lambda: eager_actions,
            "logprob_fixed_actions": lambda: eager_log_prob,
            "mixed_compiled_trunk_stock_gaussian": lambda: (eager_actions, eager_value, eager_log_prob),
        }
        for name, fn, fn_args, output_names, operand_maps in specs:
            compiled, compile_record = _compile_or_component_error(torch, name, fn, fn_args, args.device)
            compile_costs.append({"batch_size": batch_size, "component": name, **(compile_record or {})})
            if compiled is None:
                error_record = compile_record or {"name": name, "status": "error", "reports": {}}
                error_record["batch_size"] = batch_size
                components.append(error_record)
                continue
            result = compare_component(
                torch,
                name=name,
                output_names=output_names,
                eager_call=eager_by_name[name],
                compiled_call=(
                    (lambda compiled=compiled: _stock_gaussian_from_trunk(torch, compiled(obs), eps, log_std, sampling, logprob))
                    if name == "mixed_compiled_trunk_stock_gaussian"
                    else (lambda compiled=compiled, fn_args=fn_args: compiled(*fn_args))
                ),
                tolerance=args.tolerance,
                operand_maps=operand_maps,
            )
            result["batch_size"] = batch_size
            result["compile"] = compile_record
            components.append(result)
    provenance = e4.provenance(args, actual_param_count)
    provenance["source_sha256"] = {"benchmark_policy_compile_components.py": e4.sha256_file(Path(__file__))}
    payload = write_summary(args.output, plan=plan, components=components, provenance=provenance)
    payload["compile_costs"] = compile_costs
    e4.write_json(args.output / "benchmark_policy_compile_components.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=e4.DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=e4.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=e4.DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, nargs="+", default=list(e4.DEFAULT_BATCHES))
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--tolerance", type=float, default=e4.DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)
    args.source = args.source.resolve()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    try:
        e4.build_plan(device=args.device, batches=args.batches, iterations=200, warmup=0, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if args.tolerance <= 0 or not math.isfinite(args.tolerance):
        raise SystemExit("--tolerance must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    print(json.dumps({"output": str(args.output / "benchmark_policy_compile_components.json"), **summarize_results(payload["components"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
