#!/usr/bin/env python3
"""E4 standalone CUDA-graph policy inference diagnostic.

This GPU-only probe captures the existing static Gaussian SB3 policy helper for
fixed-shape inference. It deliberately does not patch PPO, does not enter
training, and does not claim game SPS. A timing result is accepted only after
graph replay matches eager actions, values, and log probabilities under the
same strict gate used by ``benchmark_policy_compile.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import benchmark_policy_compile as e4


DEFAULT_BATCHES = [64]
DEFAULT_ITERATIONS = 200
DEFAULT_WARMUP = 20
DEFAULT_CAPTURE_WARMUP = 3
REQUIRED_PARITY_SECTIONS = (
    "replay_vs_eager",
    "fresh_inputs",
    "permutation",
    "weight_update",
    "finite",
    "output_lifetime",
    "helper_direct_sb3",
    "graph_direct_sb3",
)


def build_plan(
    *,
    device: str = "cuda",
    batches: list[int] | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
    capture_warmup: int = DEFAULT_CAPTURE_WARMUP,
    obs_dim: int = e4.DEFAULT_OBS_DIM,
    action_dim: int = e4.DEFAULT_ACTION_DIM,
    net_arch: list[int] | None = None,
    seed: int = 20260906,
) -> dict[str, Any]:
    if not str(device).startswith("cuda"):
        raise ValueError("E4 CUDA graph diagnostic is GPU-only; refusing CPU fallback")
    batch_values = list(DEFAULT_BATCHES if batches is None else batches)
    arch = list(e4.DEFAULT_NET_ARCH if net_arch is None else net_arch)
    if not batch_values or any(int(batch) <= 0 for batch in batch_values):
        raise ValueError("batch sizes must be positive")
    if int(iterations) < 200:
        raise ValueError("iterations must be >= 200 for timing")
    if int(warmup) < 0 or int(capture_warmup) < 0:
        raise ValueError("warmup counts must be nonnegative")
    first_batch = int(batch_values[0])
    return {
        "label": "e4_policy_cuda_graph_static_gaussian",
        "device": str(device),
        "gpu_only": True,
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "net_arch": arch,
        "param_count_formula": e4.count_actor_critic_policy_params(obs_dim=obs_dim, action_dim=action_dim, net_arch=arch),
        "expected_param_count": e4.EXPECTED_PARAM_COUNTS.get(tuple(arch)),
        "batches": [int(batch) for batch in batch_values],
        "iterations": int(iterations),
        "warmup": int(warmup),
        "capture_warmup": int(capture_warmup),
        "seed": int(seed),
        "capture": {
            "api": "torch.cuda.CUDAGraph",
            "input_shapes": {"obs": [first_batch, int(obs_dim)], "eps": [first_batch, int(action_dim)]},
            "fixed_shape": True,
            "public_replay_outputs": "detached_clones",
        },
        "integration": "standalone_inference_only",
        "claims_game_sps": False,
        "notes": [
            "CUDA graph captures only the tensor-only policy helper, not PPO rollout or train loops.",
            "Resident replay and host-input pipeline timings are reported separately.",
            "Graph setup/capture cost is excluded from steady-state replay timing.",
            "No tolerance relaxation from the E4 compile parity gate.",
        ],
    }


def phase_record(
    *,
    label: str,
    batch_size: int,
    iterations: int,
    seconds: float,
    transfer_mode: str,
    includes_graph_replay: bool,
    includes_output_clone: bool,
    cuda_event_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(float(seconds)) or float(seconds) < 0:
        raise RuntimeError(f"invalid phase seconds for {label}: {seconds}")
    samples = int(batch_size) * int(iterations)
    record = {
        "label": label,
        "status": "completed",
        "batch_size": int(batch_size),
        "iterations": int(iterations),
        "samples": samples,
        "seconds": float(seconds),
        "iterations_per_second": float(iterations) / float(seconds) if seconds else math.inf,
        "samples_per_second": samples / float(seconds) if seconds else math.inf,
        "mean_ms_per_iteration": (float(seconds) / float(iterations)) * 1000.0 if iterations else None,
        "timing": {
            "synchronized_host_seconds": float(seconds),
            "cuda_event_seconds": float(cuda_event_seconds) if cuda_event_seconds is not None else None,
        },
        "transfer_mode": transfer_mode,
        "includes_graph_replay": bool(includes_graph_replay),
        "includes_output_clone": bool(includes_output_clone),
    }
    if extra:
        record.update(extra)
    return record


def parameter_storage_signature(policy: Any) -> list[dict[str, Any]]:
    signature = []
    for name, param in policy.named_parameters():
        signature.append(
            {
                "name": str(name),
                "data_ptr": int(param.data_ptr()),
                "shape": [int(item) for item in param.shape],
                "stride": [int(item) for item in param.stride()],
                "dtype": str(param.dtype),
                "device": str(param.device),
            }
        )
    return signature


def required_file_snapshot(*, checkpoint: Path, config: Path, source: Path, harness: Path, helper: Path) -> dict[str, dict[str, Any]]:
    files = {
        "checkpoint": Path(checkpoint),
        "config": Path(config),
        "source_train": Path(source) / "train.py",
        "harness": Path(harness),
        "policy_helper": Path(helper),
    }
    snapshot = {name: e4.file_record(path) for name, path in files.items()}
    missing = [name for name, record in snapshot.items() if record.get("exists") is not True]
    if missing:
        raise RuntimeError(f"missing required file(s): {', '.join(missing)}")
    return snapshot


def assert_file_snapshot_unchanged(snapshot: dict[str, dict[str, Any]]) -> None:
    for name, record in snapshot.items():
        current = e4.file_record(Path(record["path"]))
        if current != record:
            raise RuntimeError(f"required file changed during run: {name}")


def cuda_event_seconds(torch: Any, body: Callable[[], Any]) -> tuple[float, float, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    host_started = time.perf_counter()
    start.record()
    result = body()
    end.record()
    end.synchronize()
    host_seconds = time.perf_counter() - host_started
    return host_seconds, float(start.elapsed_time(end)) / 1000.0, result


def clone_outputs(outputs: tuple[Any, ...]) -> tuple[Any, ...]:
    cloned = []
    for tensor in outputs:
        cloned.append(tensor.detach().clone())
    return tuple(cloned)


class CudaGraphPolicyRunner:
    """Static-input CUDA graph wrapper that returns lifetime-safe clones."""

    def __init__(self, torch: Any, helper: Callable[[Any, Any], tuple[Any, Any, Any]], obs: Any, eps: Any, *, device: str, capture_warmup: int) -> None:
        self.torch = torch
        self.helper = helper
        self.device = device
        self.static_obs = obs.detach().clone()
        self.static_eps = eps.detach().clone()
        self.graph = torch.cuda.CUDAGraph()
        self.graph_outputs: tuple[Any, ...] | None = None
        self.capture_seconds = self._capture(int(capture_warmup))

    def _capture(self, capture_warmup: int) -> float:
        torch = self.torch
        torch.cuda.synchronize()
        side_stream = torch.cuda.Stream(device=self.device)
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            with torch.no_grad():
                for _ in range(capture_warmup):
                    self.helper(self.static_obs, self.static_eps)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            with torch.cuda.graph(self.graph):
                self.graph_outputs = tuple(self.helper(self.static_obs, self.static_eps))
        torch.cuda.synchronize()
        return time.perf_counter() - started

    def replay(self, obs: Any, eps: Any) -> tuple[Any, ...]:
        with self.torch.no_grad():
            self.static_obs.copy_(obs, non_blocking=True)
            self.static_eps.copy_(eps, non_blocking=True)
            self.graph.replay()
            return clone_outputs(self.graph_outputs or ())

    def replay_static(self) -> tuple[Any, ...]:
        with self.torch.no_grad():
            self.graph.replay()
            return clone_outputs(self.graph_outputs or ())


def assert_all_finite(torch: Any, label: str, outputs: tuple[Any, ...]) -> None:
    for index, tensor in enumerate(outputs):
        e4.assert_finite(torch, f"{label}_{index}", tensor)


def tuple_parity(torch: Any, left: tuple[Any, ...], right: tuple[Any, ...], *, tolerance: float) -> dict[str, Any]:
    names = ("actions", "values", "log_prob")
    if len(left) != 3 or len(right) != 3:
        raise RuntimeError(f"expected three policy outputs, got graph={len(left)} eager={len(right)}")
    reports = {
        "actions": e4.tensor_parity_report(torch, left[0], right[0], "actions", base_atol=tolerance),
        "values": e4.tensor_parity_report(torch, left[1], right[1], "values", base_atol=tolerance),
        "log_prob": e4.tensor_parity_report(torch, left[2], right[2], "log_prob", base_atol=tolerance),
    }
    e4.require_parity_reports(reports)
    return {name: reports[name] for name in names}


def validate_graph_runner(torch: Any, policy: Any, helper: Callable[[Any, Any], tuple[Any, Any, Any]], runner: CudaGraphPolicyRunner, obs: Any, eps: Any, *, tolerance: float) -> dict[str, Any]:
    parity: dict[str, Any] = {}
    graph_direct = e4.validate_helper_against_policy(torch, policy, runner.replay, obs, eps, tol=tolerance)
    parity["graph_direct_sb3"] = {"passed": True, "reports": graph_direct}
    with torch.no_grad():
        eager = tuple(helper(obs, eps))
    graph = runner.replay(obs, eps)
    assert_all_finite(torch, "graph_replay", graph)
    assert_all_finite(torch, "eager", eager)
    parity["replay_vs_eager"] = tuple_parity(torch, graph, eager, tolerance=tolerance)

    snapshot = e4.clone_output_tuple(graph)
    fresh_obs = torch.randn_like(obs)
    fresh_eps = torch.randn_like(eps)
    with torch.no_grad():
        eager_fresh = tuple(helper(fresh_obs, fresh_eps))
    graph_fresh = runner.replay(fresh_obs, fresh_eps)
    parity["fresh_inputs"] = {
        "graph_vs_eager": tuple_parity(torch, graph_fresh, eager_fresh, tolerance=tolerance),
        "action_diff": e4.assert_fresh_action_response(torch, graph[0], graph_fresh[0], label="cuda_graph"),
    }
    e4.assert_no_cross_call_storage_alias(graph, graph_fresh, first_label="graph", second_label="graph_fresh")
    e4.assert_output_snapshot_unchanged(torch, snapshot, graph, label="cuda_graph")
    parity["output_lifetime"] = {"passed": True, "public_outputs": "detached_clones"}

    perm = torch.randperm(obs.shape[0], device=obs.device)
    graph_perm = runner.replay(obs.index_select(0, perm), eps.index_select(0, perm))
    parity["permutation"] = tuple_parity(
        torch,
        graph_perm,
        tuple(item.index_select(0, perm) for item in graph),
        tolerance=tolerance,
    )

    signature_before = parameter_storage_signature(policy)
    state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    graph_before = runner.replay(obs, eps)
    try:
        with torch.no_grad():
            first_param = next(policy.parameters())
            first_param.add_(1e-4)
        signature_after_update = parameter_storage_signature(policy)
        if signature_after_update != signature_before:
            raise RuntimeError("parameter storage identity changed after in-place perturbation")
        with torch.no_grad():
            eager_after = tuple(helper(obs, eps))
        graph_after = runner.replay(obs, eps)
        after_parity = tuple_parity(torch, graph_after, eager_after, tolerance=tolerance)
        changed = e4.max_abs_diff(torch, graph_before[0], graph_after[0]) > 0.0 or e4.max_abs_diff(torch, graph_before[1], graph_after[1]) > 0.0
        if not changed:
            raise RuntimeError("CUDA graph replay did not reflect in-place parameter update")
    finally:
        policy.load_state_dict(state)
    signature_after_restore = parameter_storage_signature(policy)
    if signature_after_restore != signature_before:
        raise RuntimeError("parameter storage identity changed after state restore")
    graph_restored = runner.replay(obs, eps)
    restored = tuple_parity(torch, graph_restored, graph_before, tolerance=tolerance)
    parity["weight_update"] = {
        "changed_after_update": True,
        "restored_after_state_restore": True,
        "parameter_storage_stable": True,
        "parameter_signature": signature_before,
        "after_update_graph_vs_eager": after_parity,
        "restored_vs_before": restored,
    }
    parity["finite"] = {"passed": True}
    return parity


def measure_eager(torch: Any, helper: Callable[[Any, Any], tuple[Any, Any, Any]], obs: Any, eps: Any, *, batch_size: int, iterations: int, warmup: int) -> dict[str, Any]:
    with torch.no_grad():
        for _ in range(warmup):
            clone_outputs(tuple(helper(obs, eps)))
    torch.cuda.synchronize()
    def body():
        last = None
        with torch.no_grad():
            for _ in range(iterations):
                last = clone_outputs(tuple(helper(obs, eps)))
        return last

    host_seconds, event_seconds, last = cuda_event_seconds(torch, body)
    assert_all_finite(torch, "eager_timed", last or ())
    return phase_record(
        label="eager_device_resident_fixed_inputs",
        batch_size=batch_size,
        iterations=iterations,
        seconds=host_seconds,
        transfer_mode="device_resident_fixed_inputs",
        includes_graph_replay=False,
        includes_output_clone=True,
        cuda_event_seconds=event_seconds,
    )


def measure_graph_resident(torch: Any, runner: CudaGraphPolicyRunner, *, batch_size: int, iterations: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        runner.replay_static()
    torch.cuda.synchronize()
    def body():
        last = None
        for _ in range(iterations):
            last = runner.replay_static()
        return last

    host_seconds, event_seconds, last = cuda_event_seconds(torch, body)
    assert_all_finite(torch, "graph_resident_timed", last or ())
    return phase_record(
        label="graph_resident_replay_fixed_inputs",
        batch_size=batch_size,
        iterations=iterations,
        seconds=host_seconds,
        transfer_mode="device_resident_static_inputs",
        includes_graph_replay=True,
        includes_output_clone=True,
        cuda_event_seconds=event_seconds,
    )


def measure_graph_host_pipeline(torch: Any, runner: CudaGraphPolicyRunner, obs: Any, eps: Any, *, batch_size: int, iterations: int, warmup: int, device: str) -> dict[str, Any]:
    cpu_obs = obs.detach().cpu()
    cpu_eps = eps.detach().cpu()
    for _ in range(warmup):
        runner.replay(cpu_obs.to(device, non_blocking=True), cpu_eps.to(device, non_blocking=True))
    torch.cuda.synchronize()
    def body():
        last = None
        for _ in range(iterations):
            last = runner.replay(cpu_obs.to(device, non_blocking=True), cpu_eps.to(device, non_blocking=True))
        return last

    host_seconds, event_seconds, last = cuda_event_seconds(torch, body)
    assert_all_finite(torch, "graph_host_pipeline_timed", last or ())
    return phase_record(
        label="graph_host_input_pipeline_fixed_inputs",
        batch_size=batch_size,
        iterations=iterations,
        seconds=host_seconds,
        transfer_mode="host_to_static_device_inputs",
        includes_graph_replay=True,
        includes_output_clone=True,
        cuda_event_seconds=event_seconds,
    )


def summarize_status(*, phases: list[dict[str, Any]], parity: dict[str, Any]) -> dict[str, Any]:
    for section in REQUIRED_PARITY_SECTIONS:
        if section not in parity:
            raise RuntimeError(f"missing parity gate section: {section}")

    def gate_value(section: str, key: str) -> bool:
        record = parity.get(section, {})
        if key in record:
            return record.get(key) is True
        batches = record.get("batches") if isinstance(record, dict) else None
        if isinstance(batches, dict) and batches:
            return all(isinstance(item, dict) and item.get(key) is True for item in batches.values())
        return False

    if not gate_value("weight_update", "changed_after_update"):
        raise RuntimeError("weight update gate did not prove changed replay outputs")
    if not gate_value("weight_update", "restored_after_state_restore"):
        raise RuntimeError("weight update gate did not prove restored replay outputs")
    if not gate_value("finite", "passed"):
        raise RuntimeError("finite gate did not pass")
    if not gate_value("output_lifetime", "passed"):
        raise RuntimeError("output lifetime gate did not pass")
    if not gate_value("helper_direct_sb3", "passed"):
        raise RuntimeError("helper direct SB3 gate did not pass")
    if not gate_value("graph_direct_sb3", "passed"):
        raise RuntimeError("graph direct SB3 gate did not pass")
    failed_phases = [item.get("label") for item in phases if item.get("status") != "completed"]
    return {
        "status": "failed" if failed_phases else "completed",
        "failed_phases": failed_phases,
        "phase_count": len(phases),
        "parity_sections": list(parity.keys()),
    }


def provenance(args: argparse.Namespace, actual_param_count: int) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "checkpoint": e4.file_record(args.checkpoint),
            "config": e4.file_record(args.config),
            "source_train": e4.file_record(args.source / "train.py"),
            "harness": e4.file_record(Path(__file__)),
            "policy_helper": e4.file_record(Path(e4.__file__)),
        },
        "actual_policy_param_count": int(actual_param_count),
        "runtime": {
            "python": e4.sys.version,
            "executable": e4.sys.executable,
            "torch": e4.package_version("torch"),
            "stable_baselines3": e4.package_version("stable-baselines3"),
        },
    }


def write_summary(output: Path, *, plan: dict[str, Any], phases: list[dict[str, Any]], parity: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": "e4_policy_cuda_graph_static_gaussian",
        **summarize_status(phases=phases, parity=parity),
        "plan": plan,
        "phases": phases,
        "parity": parity,
        "provenance": provenance,
    }
    e4.write_json(output / "benchmark_policy_cuda_graph.json", payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.device = e4.normalize_cuda_device(args.device)
    plan = build_plan(device=args.device, batches=args.batches, iterations=args.iterations, warmup=args.warmup, capture_warmup=args.capture_warmup, seed=args.seed)
    file_snapshot = required_file_snapshot(
        checkpoint=args.checkpoint,
        config=args.config,
        source=args.source,
        harness=Path(__file__),
        helper=Path(e4.__file__),
    )
    torch, np, PPO = e4.require_gpu_runtime(args.device)
    e4.configure_seed(torch, np, args.seed)
    policy = e4.load_policy(torch, PPO, args.checkpoint, args.device)
    actual_param_count = sum(param.numel() for param in policy.parameters())
    if actual_param_count != plan["param_count_formula"]:
        raise RuntimeError(f"checkpoint policy parameter count {actual_param_count} != plan formula {plan['param_count_formula']}")
    helper = e4.static_gaussian_actorcritic(policy, torch)
    phases: list[dict[str, Any]] = []
    parity_by_batch: dict[str, Any] = {}
    for batch_size in plan["batches"]:
        obs = torch.randn(batch_size, plan["obs_dim"], device=args.device)
        eps = torch.randn(batch_size, plan["action_dim"], device=args.device)
        helper_direct = e4.validate_helper_against_policy(torch, policy, helper, obs, eps, tol=args.tolerance)
        runner = CudaGraphPolicyRunner(torch, helper, obs, eps, device=args.device, capture_warmup=args.capture_warmup)
        phases.append(
            phase_record(
                label="capture_setup",
                batch_size=batch_size,
                iterations=1,
                seconds=runner.capture_seconds,
                transfer_mode="capture_setup",
                includes_graph_replay=False,
                includes_output_clone=False,
            )
        )
        parity = validate_graph_runner(torch, policy, helper, runner, obs, eps, tolerance=args.tolerance)
        parity["helper_direct_sb3"] = {"passed": True, "reports": helper_direct}
        parity_by_batch[str(batch_size)] = parity
        phases.append(measure_eager(torch, helper, obs, eps, batch_size=batch_size, iterations=args.iterations, warmup=args.warmup))
        phases.append(measure_graph_resident(torch, runner, batch_size=batch_size, iterations=args.iterations, warmup=args.warmup))
        phases.append(measure_graph_host_pipeline(torch, runner, obs, eps, batch_size=batch_size, iterations=args.iterations, warmup=args.warmup, device=args.device))
    aggregate_parity = {section: {"batches": {batch: parity[section] for batch, parity in parity_by_batch.items()}} for section in REQUIRED_PARITY_SECTIONS}
    assert_file_snapshot_unchanged(file_snapshot)
    run_provenance = provenance(args, actual_param_count)
    run_provenance["files"] = file_snapshot
    return write_summary(args.output, plan=plan, phases=phases, parity=aggregate_parity, provenance=run_provenance)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=e4.DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=e4.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=e4.DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--capture-warmup", type=int, default=DEFAULT_CAPTURE_WARMUP)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--tolerance", type=float, default=e4.DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)
    args.source = args.source.resolve()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    try:
        e4.normalize_cuda_device(args.device)
        build_plan(device=args.device, batches=args.batches, iterations=args.iterations, warmup=args.warmup, capture_warmup=args.capture_warmup, seed=args.seed)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if args.tolerance <= 0 or not math.isfinite(args.tolerance):
        parser.error("--tolerance must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    print(json.dumps({"output": str(args.output / "benchmark_policy_cuda_graph.json"), "status": payload["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
