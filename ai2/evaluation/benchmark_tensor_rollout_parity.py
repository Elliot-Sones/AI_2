#!/usr/bin/env python3
"""One-update real-game parity gate for TensorPPO's hybrid CPU-game collector.

This verifier runs the frozen training job twice from the same checkpoint and
seed: stock SB3 PPO versus TensorPPO wrapping the existing CPU SB3 VecEnv with
``CPUGameTensorAdapter``.  The game remains CPU simulated; only collection and
rollout storage differ.  It saves rollout arrays and reports the first
divergence plus explicit tolerances instead of claiming byte identity.
"""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import source_path

import argparse
from contextlib import ExitStack
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any

import numpy as np

import ai2.runtime.benchmark_campaign as benchmark_campaign
import ai2.evaluation.benchmark_graph_rollout_parity as graph_parity


ROOT = PROJECT_ROOT
REQUIRED_INPUTS = (
    "source_train",
    "source_environment",
    "config",
    "train_job",
    "checkpoint",
    "verifier",
    "campaign",
    "grouped_env",
    "tensor_ppo",
    "tensor_rollout_buffer",
    "tensor_cpu_vec_env",
)
ROLLOUT_FIELDS = (
    "observations",
    "actions",
    "episode_starts",
    "rewards",
    "values",
    "log_probs",
    "advantages",
    "returns",
)
EXACT_ARRAY_FIELDS = {"observations", "actions", "episode_starts"}
TOLERANCES = {
    "rewards": 1e-6,
    "values": 1e-6,
    "log_probs": 1e-6,
    "advantages": 1e-5,
    "returns": 1e-5,
    "policy": 1e-6,
    "optimizer": 1e-6,
    "losses": 1e-6,
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(benchmark_campaign.json_safe(strip_internal_arrays(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def strip_internal_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_internal_arrays(item) for key, item in value.items() if key not in {"_arrays", "_policy", "_optimizer"}}
    if isinstance(value, list):
        return [strip_internal_arrays(item) for item in value]
    return value


def file_record(path: Path) -> dict[str, Any]:
    return benchmark_campaign.file_record(Path(path))


def require_file_record(name: str, path: Path) -> dict[str, Any]:
    record = file_record(path)
    if record.get("exists") is not True:
        raise RuntimeError(f"missing required input file {name}: {record.get('path')}")
    if not isinstance(record.get("sha256"), str):
        raise RuntimeError(f"missing required input sha256 {name}: {record.get('path')}")
    return record


def input_file_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).resolve()
    snapshot = {
        "source_train": require_file_record("source_train", source_path(source, "train.py")),
        "source_environment": require_file_record(
            "source_environment",
            source / "UTMIST-AI2-main" / "environment" / "environment.py",
        ),
        "config": require_file_record("config", Path(args.config).resolve()),
        "train_job": require_file_record("train_job", Path(args.train_job).resolve()),
        "checkpoint": require_file_record("checkpoint", Path(args.checkpoint).resolve()),
        "verifier": require_file_record("verifier", Path(__file__).resolve()),
        "campaign": require_file_record("campaign", Path(benchmark_campaign.__file__).resolve()),
        "grouped_env": require_file_record("grouped_env", Path(args.grouped_env_path).resolve()),
        "tensor_ppo": require_file_record("tensor_ppo", source_path(ROOT, "tensor_ppo.py")),
        "tensor_rollout_buffer": require_file_record("tensor_rollout_buffer", source_path(ROOT, "tensor_rollout_buffer.py")),
        "tensor_cpu_vec_env": require_file_record("tensor_cpu_vec_env", source_path(ROOT, "tensor_cpu_vec_env.py")),
    }
    return snapshot


def expected_steps(args: argparse.Namespace) -> int:
    return int(args.n_envs) * int(args.n_steps)


def expected_optimizer_steps(args: argparse.Namespace) -> int:
    return int(args.epochs) * math.ceil(expected_steps(args) / int(args.batch_size))


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.ascontiguousarray(np.asarray(value))


def capture_rollout_arrays(buffer: Any) -> dict[str, np.ndarray]:
    return {field: to_numpy(getattr(buffer, field)) for field in ROLLOUT_FIELDS}


def flatten_state(value: Any, prefix: str = "") -> dict[str, Any]:
    if hasattr(value, "detach"):
        return {prefix: to_numpy(value)}
    if isinstance(value, np.ndarray):
        return {prefix: np.ascontiguousarray(value)}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_state(value[key], child))
        return out
    if isinstance(value, (list, tuple)):
        out = {}
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            out.update(flatten_state(item, child))
        return out
    return {prefix: value}


def capture_policy(agent: Any) -> dict[str, Any]:
    return flatten_state(agent.policy.state_dict())


def capture_optimizer(agent: Any) -> dict[str, Any]:
    return flatten_state(agent.policy.optimizer.state_dict())


def capture_losses(agent: Any) -> dict[str, float]:
    values = getattr(getattr(agent, "logger", None), "name_to_value", {})
    losses: dict[str, float] = {}
    for key, value in values.items():
        if "loss" in str(key) or str(key) in {"train/approx_kl", "train/clip_fraction", "train/explained_variance"}:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                losses[str(key)] = number
    return losses


def rng_snapshot(torch: Any, device: str) -> dict[str, str]:
    record = {
        "python": graph_parity.digest_any(random.getstate()),
        "numpy": graph_parity.digest_any(np.random.get_state()),
        "torch_cpu": graph_parity.digest_any(torch.get_rng_state()),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        cuda_device = torch.device(device)
        index = torch.cuda.current_device() if cuda_device.index is None else int(cuda_device.index)
        record["torch_cuda"] = graph_parity.digest_any(torch.cuda.get_rng_state(index))
    return record


def array_descriptors(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path.resolve()),
            "key": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": graph_parity.digest_any(array),
        }
        for name, array in arrays.items()
    }


def state_descriptors(state: dict[str, Any]) -> dict[str, Any]:
    descriptors: dict[str, Any] = {}
    for name, value in state.items():
        if isinstance(value, np.ndarray):
            descriptors[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": graph_parity.digest_any(value),
            }
        else:
            descriptors[name] = value
    return descriptors


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return array_descriptors(path, arrays)


def arrays_for_compare(receipt: dict[str, Any]) -> dict[str, np.ndarray]:
    direct = receipt.get("arrays")
    if isinstance(direct, dict) and all(isinstance(value, np.ndarray) for value in direct.values()):
        return direct
    internal = receipt.get("_arrays")
    if isinstance(internal, dict):
        return internal
    path = receipt.get("rollout_arrays_path")
    if not isinstance(path, str):
        raise RuntimeError(f"{receipt.get('label', 'receipt')} missing rollout arrays")
    with np.load(path) as loaded:
        return {field: np.ascontiguousarray(loaded[field]) for field in ROLLOUT_FIELDS}


def state_for_compare(receipt: dict[str, Any], field: str) -> dict[str, Any]:
    internal = receipt.get(f"_{field}")
    if isinstance(internal, dict):
        return internal
    direct = receipt.get(field)
    if isinstance(direct, dict):
        return direct
    return {}


def first_array_divergence(control: np.ndarray, tensor: np.ndarray, *, atol: float) -> dict[str, Any] | None:
    if control.shape != tensor.shape:
        return {"kind": "shape", "control": list(control.shape), "tensor": list(tensor.shape)}
    if control.dtype != tensor.dtype and atol == 0:
        return {"kind": "dtype", "control": str(control.dtype), "tensor": str(tensor.dtype)}
    finite = np.isfinite(control) & np.isfinite(tensor)
    if not bool(np.all(finite)):
        index = list(map(int, np.argwhere(~finite)[0]))
        return {
            "kind": "nonfinite",
            "index": index,
            "control": float(control[tuple(index)]),
            "tensor": float(tensor[tuple(index)]),
        }
    diff = np.abs(control.astype(np.float64) - tensor.astype(np.float64))
    bad = diff > float(atol)
    if not np.any(bad):
        return None
    index = list(map(int, np.argwhere(bad)[0]))
    return {
        "kind": "value",
        "index": index,
        "control": float(control[tuple(index)]),
        "tensor": float(tensor[tuple(index)]),
        "abs_diff": float(diff[tuple(index)]),
    }


def array_metrics(control: np.ndarray, tensor: np.ndarray) -> dict[str, Any]:
    if control.shape != tensor.shape:
        return {"shape_mismatch": [list(control.shape), list(tensor.shape)]}
    diff = np.abs(control.astype(np.float64) - tensor.astype(np.float64))
    return {
        "shape": list(control.shape),
        "dtype_control": str(control.dtype),
        "dtype_tensor": str(tensor.dtype),
        "max_abs_diff": float(diff.max(initial=0.0)),
        "l2_norm_diff": float(np.linalg.norm(diff.ravel())) if diff.size else 0.0,
    }


def compare_named_values(control: dict[str, Any], tensor: dict[str, Any], *, atol: float, namespace: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    mismatches: list[dict[str, Any]] = []
    first = None
    if not control or not tensor:
        mismatch = {
            "field": namespace,
            "kind": "empty",
            "control_count": len(control),
            "tensor_count": len(tensor),
        }
        return [mismatch], mismatch
    keys = sorted(set(control) | set(tensor))
    for key in keys:
        if key not in control or key not in tensor:
            mismatch = {"field": namespace, "name": key, "kind": "missing"}
        elif isinstance(control[key], np.ndarray) or isinstance(tensor[key], np.ndarray):
            mismatch = first_array_divergence(to_numpy(control[key]), to_numpy(tensor[key]), atol=atol)
            if mismatch is not None:
                mismatch = {"field": namespace, "name": key, **mismatch}
        else:
            mismatch = None if control[key] == tensor[key] else {"field": namespace, "name": key, "control": control[key], "tensor": tensor[key]}
        if mismatch is not None:
            mismatches.append(mismatch)
            if first is None:
                first = mismatch
    return mismatches, first


def compare_parity(control: dict[str, Any], tensor: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    first = None
    metrics: dict[str, Any] = {}
    control_arrays = arrays_for_compare(control)
    tensor_arrays = arrays_for_compare(tensor)
    for field in ROLLOUT_FIELDS:
        atol = 0.0 if field in EXACT_ARRAY_FIELDS else float(TOLERANCES.get(field, 1e-6))
        metrics[field] = array_metrics(control_arrays[field], tensor_arrays[field])
        divergence = first_array_divergence(control_arrays[field], tensor_arrays[field], atol=atol)
        if divergence is not None:
            mismatch = {"field": field, "tolerance": atol, **divergence}
            mismatches.append(mismatch)
            if first is None:
                first = mismatch

    exact_sections = ("counters", "rng")
    for section in exact_sections:
        if control.get(section) != tensor.get(section):
            mismatch = {"field": section, "control": control.get(section), "tensor": tensor.get(section)}
            mismatches.append(mismatch)
            if first is None:
                first = mismatch

    for section in ("policy", "optimizer", "losses"):
        section_mismatches, section_first = compare_named_values(
            state_for_compare(control, section),
            state_for_compare(tensor, section),
            atol=float(TOLERANCES[section]),
            namespace=section,
        )
        mismatches.extend(section_mismatches)
        if first is None and section_first is not None:
            first = section_first

    return {
        "parity_passed": not mismatches,
        "first_divergence": first,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:25],
        "array_metrics": metrics,
        "tolerances": {**{field: 0.0 for field in EXACT_ARRAY_FIELDS}, **TOLERANCES},
    }


def actual_device_record(agent: Any, torch: Any, requested: str) -> dict[str, Any]:
    policy = getattr(agent, "policy", None)
    parameters = list(policy.parameters()) if policy is not None and hasattr(policy, "parameters") else []
    return {
        "requested": str(requested),
        "cuda_available": bool(torch.cuda.is_available()),
        "policy_device": str(getattr(policy, "device", None)),
        "parameter_devices": sorted({str(param.device) for param in parameters}),
        "parameters_all_cuda": bool(parameters) and all(str(param.device).startswith("cuda") for param in parameters),
    }


def make_parity_ppo_class(base_cls: type, *, receipt: dict[str, Any], output: Path, args: argparse.Namespace) -> type:
    class ParityPPO(base_cls):
        def collect_rollouts(self, *collect_args: Any, **collect_kwargs: Any) -> Any:
            import torch

            receipt["collect_calls"] = int(receipt.get("collect_calls", 0)) + 1
            receipt.setdefault("rng", {})["pre_collect"] = rng_snapshot(torch, str(args.device))
            ok = super().collect_rollouts(*collect_args, **collect_kwargs)
            if not ok:
                raise RuntimeError("collect_rollouts did not complete")
            arrays = capture_rollout_arrays(self.rollout_buffer)
            arrays_path = output / "rollout_arrays.npz"
            receipt["_arrays"] = arrays
            receipt["arrays"] = save_arrays(arrays_path, arrays)
            receipt["rollout_arrays_path"] = str(arrays_path.resolve())
            receipt["rollout_buffer_sha256"] = graph_parity.digest_any(arrays)
            receipt["pre_update_policy_sha256"] = graph_parity.digest_any(capture_policy(self))
            receipt["pre_update_optimizer_sha256"] = graph_parity.digest_any(capture_optimizer(self))
            receipt["rng"]["post_collect"] = rng_snapshot(torch, str(args.device))
            return ok

        def train(self) -> Any:
            import torch

            receipt["train_calls"] = int(receipt.get("train_calls", 0)) + 1
            receipt["counters"]["n_updates_before"] = int(getattr(self, "_n_updates", 0))
            optimizer = self.policy.optimizer
            original_step = optimizer.step
            step_calls = {"count": 0}

            def counted_step(*step_args: Any, **step_kwargs: Any) -> Any:
                step_calls["count"] += 1
                return original_step(*step_args, **step_kwargs)

            optimizer.step = counted_step
            try:
                result = super().train()
            finally:
                optimizer.step = original_step
            receipt["optimizer_step_calls"] = int(receipt.get("optimizer_step_calls", 0)) + int(step_calls["count"])
            receipt["counters"]["n_updates_after"] = int(getattr(self, "_n_updates", 0))
            receipt["n_updates_delta"] = receipt["counters"]["n_updates_after"] - receipt["counters"]["n_updates_before"]
            policy = capture_policy(self)
            optimizer = capture_optimizer(self)
            receipt["_policy"] = policy
            receipt["_optimizer"] = optimizer
            receipt["policy"] = state_descriptors(policy)
            receipt["optimizer"] = state_descriptors(optimizer)
            receipt["losses"] = capture_losses(self)
            receipt["policy_sha256"] = graph_parity.digest_any(policy)
            receipt["optimizer_sha256"] = graph_parity.digest_any(optimizer)
            receipt["rng"]["post_train"] = rng_snapshot(torch, str(args.device))
            receipt["num_timesteps"] = int(getattr(self, "num_timesteps", 0))
            receipt["counters"]["num_timesteps"] = receipt["num_timesteps"]
            receipt["actual_device"] = actual_device_record(self, torch, str(args.device))
            return result

    ParityPPO.__name__ = "TensorParityPPO" if "TensorPPO" in base_cls.__name__ else "StockParityPPO"
    return ParityPPO


def wrap_build_env_for_tensor(train_job: Any, *, device: str) -> Any:
    from ai2.simulation.tensor_cpu_vec_env import CPUGameTensorAdapter
    from ai2.training.tensor_ppo import TensorVecEnv

    original = train_job.build_env

    def build_env(*args: Any, **kwargs: Any) -> TensorVecEnv:
        bound = inspect.signature(original).bind(*args, **kwargs)
        call_device = bound.arguments.get("device", device)
        env = original(*args, **kwargs)
        try:
            return TensorVecEnv(CPUGameTensorAdapter(env, device=call_device), device=call_device)
        except BaseException:
            close = getattr(env, "close", None)
            if close is not None:
                close()
            raise

    train_job.build_env = build_env
    return original


def run_arm(args: argparse.Namespace, label: str, *, use_tensor: bool) -> dict[str, Any]:
    output = Path(args.output).resolve() / label
    output.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "label": label,
        "status": "running",
        "use_tensor_ppo": bool(use_tensor),
        "started_unix": time.time(),
        "expected_steps": expected_steps(args),
        "expected_epochs": int(args.epochs),
        "expected_optimizer_steps": expected_optimizer_steps(args),
        "collect_calls": 0,
        "train_calls": 0,
        "optimizer_step_calls": 0,
        "counters": {},
        "rng": {},
        "provenance_before": {
            **input_file_snapshot(args),
            "controls": {
                "n_envs": int(args.n_envs),
                "n_steps": int(args.n_steps),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "seed": int(args.seed),
                "phase": str(args.phase),
                "device": str(args.device),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "group_size": int(args.group_size),
            },
        },
    }
    train_job = benchmark_campaign.import_train_job_module(Path(args.train_job))
    original_ppo = train_job.PPO
    original_build_env = None
    grouped_cleanup = None
    try:
        grouped_cleanup = benchmark_campaign.patch_grouped_vec_env(train_job, args)
        if use_tensor:
            from ai2.training.tensor_ppo import TensorPPO

            original_build_env = wrap_build_env_for_tensor(train_job, device=str(args.device))
            train_job.PPO = make_parity_ppo_class(TensorPPO, receipt=receipt, output=output, args=args)
        else:
            train_job.PPO = make_parity_ppo_class(original_ppo, receipt=receipt, output=output, args=args)
        previous_cwd = Path.cwd()
        try:
            os.chdir(Path(args.source).resolve())
            plan = benchmark_campaign.plan_run(
                n_envs=args.n_envs,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                epochs=args.epochs,
                warmup=0,
                cycles=1,
            )
            train_args = benchmark_campaign.build_train_job_args(args, plan)
            train_args.output = str(output)
            train_job.run_job(train_args)
        finally:
            os.chdir(previous_cwd)
        result_path = output / "result.json"
        metadata_path = output / "metadata.json"
        receipt["train_result"] = json.loads(result_path.read_text(encoding="utf-8"))
        receipt["train_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        receipt["actual_steps"] = int(receipt["train_result"].get("actual_steps", -1))
        receipt["status"] = "completed"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
    finally:
        train_job.PPO = original_ppo
        if original_build_env is not None:
            train_job.build_env = original_build_env
        if grouped_cleanup is not None:
            grouped_cleanup()
        receipt["ended_unix"] = time.time()
        receipt["provenance_after"] = input_file_snapshot(args)
        write_json(output / "tensor_rollout_parity_receipt.json", receipt)
    if receipt["status"] != "completed":
        raise RuntimeError(f"{label} arm failed: {receipt.get('error')}")
    return receipt


def validate_training_receipt(receipt: dict[str, Any], *, label: str) -> None:
    if receipt.get("status") != "completed":
        raise RuntimeError(f"{label} status is not completed: {receipt.get('status')}")
    for field in ("expected_steps", "expected_epochs", "expected_optimizer_steps", "actual_steps", "collect_calls", "train_calls", "n_updates_delta", "optimizer_step_calls"):
        if field not in receipt:
            raise RuntimeError(f"{label} missing field: {field}")
    if int(receipt["actual_steps"]) != int(receipt["expected_steps"]):
        raise RuntimeError(f"{label} actual_steps mismatch")
    if int(receipt["collect_calls"]) != 1 or int(receipt["train_calls"]) != 1:
        raise RuntimeError(f"{label} expected one collect/train")
    if int(receipt["n_updates_delta"]) != int(receipt["expected_epochs"]):
        raise RuntimeError(f"{label} epoch count mismatch")
    if int(receipt["optimizer_step_calls"]) != int(receipt["expected_optimizer_steps"]):
        raise RuntimeError(f"{label} optimizer step count mismatch")
    device = receipt.get("actual_device")
    if not isinstance(device, dict) or device.get("parameters_all_cuda") is not True:
        raise RuntimeError(f"{label} missing CUDA device attestation")
    train_result = receipt.get("train_result")
    if not isinstance(train_result, dict) or train_result.get("status") != "completed":
        raise RuntimeError(f"{label} train_result not completed")
    if train_result.get("error") is not None or train_result.get("stop_reason") is not None:
        raise RuntimeError(f"{label} train_result ended early")


def validate_shared_inputs(control: dict[str, Any], tensor: dict[str, Any]) -> None:
    for section in ("provenance_before", "provenance_after"):
        left = control.get(section, {})
        right = tensor.get(section, {})
        for field in REQUIRED_INPUTS:
            if left.get(field, {}).get("sha256") != right.get(field, {}).get("sha256"):
                raise RuntimeError(f"input hash mismatch for {section}.{field}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"--output already exists; refusing to overwrite: {output}")
    output.mkdir(parents=True)
    benchmark_campaign.configure_environment(args)
    control = run_arm(args, "control", use_tensor=False)
    tensor = run_arm(args, "tensor", use_tensor=True)
    validate_training_receipt(control, label="control")
    validate_training_receipt(tensor, label="tensor")
    validate_shared_inputs(control, tensor)
    parity = compare_parity(control, tensor)
    summary = {
        "status": "completed" if parity["parity_passed"] else "failed",
        "diagnostic": "tensor_rollout_one_update_parity",
        "scope": "real CPU game, grouped VecEnv, TensorPPO hybrid CPU-game adapter; not GPU-native simulation",
        "control": strip_internal_arrays(control),
        "tensor": strip_internal_arrays(tensor),
        "parity": parity,
    }
    write_json(output / "tensor_rollout_parity_summary.json", summary)
    if not parity["parity_passed"]:
        raise RuntimeError(f"tensor rollout parity failed: {parity['first_divergence']}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=benchmark_campaign.DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=benchmark_campaign.DEFAULT_TRAIN_JOB)
    parser.add_argument("--config", type=Path, default=benchmark_campaign.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=benchmark_campaign.DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--phase", default="0b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maxseconds", type=float, default=600.0)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--opponents")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--grouped-env-path", type=Path, default=source_path(ROOT, "grouped_vec_env.py"))
    return parser


def hydrate_campaign_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Add benchmark_campaign default fields required by reused helpers."""

    defaults_parser = benchmark_campaign.build_parser()
    default_argv = ["--output", str(Path(args.output).resolve())]
    for option, name in (
        ("--source", "source"),
        ("--train-job", "train_job"),
        ("--config", "config"),
        ("--checkpoint", "checkpoint"),
        ("--device", "device"),
    ):
        if hasattr(args, name):
            default_argv.extend([option, str(getattr(args, name))])
    defaults = benchmark_campaign.validate_args(
        defaults_parser,
        defaults_parser.parse_args(default_argv),
    )
    for name, value in vars(defaults).items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if not Path(args.output).is_absolute():
        raise SystemExit("--output must be absolute")
    args = hydrate_campaign_defaults(args)
    if not str(args.device).startswith("cuda"):
        raise SystemExit("tensor rollout parity is CUDA-only")
    for name in ("source", "train_job", "config", "checkpoint", "output", "grouped_env_path"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    for name in ("n_envs", "n_steps", "batch_size", "epochs", "group_size"):
        value = getattr(args, name)
        if isinstance(value, bool) or int(value) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        setattr(args, name, int(value))
    if int(args.n_envs) * int(args.n_steps) != 32768:
        raise SystemExit("--n-envs * --n-steps must equal 32768 for matched one-update gate")
    return args


def main(argv: list[str] | None = None) -> int:
    args = validate_args(build_parser().parse_args(argv))
    summary = run(args)
    print(json.dumps({"status": summary["status"], "summary": str(Path(args.output) / "tensor_rollout_parity_summary.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
