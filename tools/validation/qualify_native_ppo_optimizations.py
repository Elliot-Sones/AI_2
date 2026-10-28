#!/usr/bin/env python3
"""Qualify selected-checkpoint PPO optimizations using actual CUDA updates.

This is a deterministic synthetic-environment integration qualification, not a
navigation-quality or timeout-batching test. No compiled arm is included. Every
arm loads through the experimental class's load -> from_sb3 path. Baseline CPU
snapshots live on temporary disk; only one is loaded for each exact comparison.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import source_path

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time
from typing import Any
import zipfile

import numpy as np

ROOT = PROJECT_ROOT
SCHEMA = "ai2.native_ppo_optimizations.qualification.v1"
ROLLOUT_FIELDS = ("observations", "actions", "rewards", "episode_starts", "values",
                  "log_probs", "advantages", "returns")
SOURCE_FILES = ("qualify_native_ppo_optimizations.py", "experimental_native_lane_training.py",
                "tensor_ppo.py", "tensor_rollout_buffer.py", "ppo_optimizer_counters.py",
                "ppo_update_optimizations.py")


class QualificationError(RuntimeError):
    """A required numerical, placement, or execution contract failed."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(*, root: Path = ROOT, names: tuple[str, ...] | None = None) -> dict[str, str]:
    return {name: file_sha256(source_path(root, name)) for name in (SOURCE_FILES if names is None else names)}


def compare_snapshots(expected: Any, actual: Any) -> dict[str, Any]:
    """Compare host snapshots exactly, rejecting missing fields and nonfinite data."""
    result = {"arrays_checked": 0, "elements_checked": 0, "max_abs_difference": 0.0}

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, np.ndarray):
            if not isinstance(right, np.ndarray) or left.shape != right.shape or left.dtype != right.dtype:
                raise QualificationError(f"{path}: array shape/dtype mismatch")
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise QualificationError(f"{path}: nonfinite values")
            if not np.array_equal(left, right):
                maximum = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                raise QualificationError(f"{path}: exact comparison failed, max_abs_difference={maximum}")
            result["arrays_checked"] += 1
            result["elements_checked"] += int(left.size)
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                raise QualificationError(f"{path}: dictionary fields differ")
            for key in left:
                compare(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                raise QualificationError(f"{path}: sequence type/length differs")
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{path}[{index}]")
        else:
            if isinstance(left, (float, np.floating)) and not np.isfinite(left):
                raise QualificationError(f"{path}: nonfinite scalar")
            if type(left) is not type(right) or left != right:
                raise QualificationError(f"{path}: scalar differs: {left!r} vs {right!r}")

    compare(expected, actual, "snapshot")
    return result


def optimization_arms() -> dict[str, Any]:
    from ai2.training.experimental_native_lane_training import OptimizationOptions

    return {
        "baseline": OptimizationOptions(),
        "optimizer_cpu_steps": OptimizationOptions(optimizer_cpu_steps=True),
        "deferred_metrics": OptimizationOptions(deferred_metrics=True),
        "boundary_validation": OptimizationOptions(boundary_validation=True),
        "combined": OptimizationOptions(optimizer_cpu_steps=True, deferred_metrics=True, boundary_validation=True),
    }


def validate_census(census: dict[str, Any], *, normalized: bool) -> None:
    if not census["all_parameters_cuda"] or not census["all_moments_cuda"]:
        raise QualificationError("policy parameters and Adam moments must all remain CUDA")
    if census["step_count"] < 1:
        raise QualificationError("selected checkpoint must contain initialized Adam counters")
    if normalized and census["step_devices"] != {"cpu": census["step_count"]}:
        raise QualificationError("normalized Adam scalar counters must all be CPU, including after reload")


def validate_completed_arms(arms: dict[str, Any], *, updates: int) -> None:
    if arms.keys() != optimization_arms().keys():
        raise QualificationError("qualification did not execute every required arm")
    for name, arm in arms.items():
        if not arm.get("initial_exact") or len(arm["updates"]) != updates:
            raise QualificationError(f"{name}: initial comparison or required update count incomplete")
        if not all(update.get("exact") for update in arm["updates"]) or not arm.get("save_reload", {}).get("exact"):
            raise QualificationError(f"{name}: update or save/reload parity incomplete")


def _host_tree(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return np.array(value)
    if isinstance(value, dict):
        return {key: _host_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_host_tree(item) for item in value)
    return value


def _rng_state() -> dict[str, Any]:
    import torch

    return _host_tree({"python": random.getstate(), "numpy": np.random.get_state(),
                       "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()})


def _rng_hashes(rng: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": hashlib.sha256(repr(rng["python"]).encode()).hexdigest(),
        "numpy": hashlib.sha256(repr(rng["numpy"][0]).encode() + rng["numpy"][1].tobytes()
                                 + repr(rng["numpy"][2:]).encode()).hexdigest(),
        "torch_cpu": hashlib.sha256(rng["torch_cpu"].tobytes()).hexdigest(),
        "torch_cuda": [hashlib.sha256(state.tobytes()).hexdigest() for state in rng["torch_cuda"]],
    }


def _model_snapshot(model: Any) -> dict[str, Any]:
    return _host_tree({"policy": model.policy.state_dict(), "optimizer": model.policy.optimizer.state_dict(),
                       "n_updates": model._n_updates, "num_timesteps": model.num_timesteps,
                       "progress_remaining": model._current_progress_remaining})


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def optimizer_census(model: Any) -> dict[str, Any]:
    import torch

    names = {parameter: name for name, parameter in model.policy.named_parameters()}
    parameters = list(model.policy.parameters())
    moments, steps, groups = [], [], []
    for group in model.policy.optimizer.param_groups:
        groups.append(_json_value(_host_tree({key: value for key, value in group.items() if key != "params"})))
        for parameter in group["params"]:
            state = model.policy.optimizer.state.get(parameter, {})
            if not state:
                continue
            step = state.get("step")
            if not torch.is_tensor(step) or step.ndim != 0 or not torch.isfinite(step).item():
                raise QualificationError("initialized Adam step must be a finite scalar tensor")
            steps.append({"parameter": names[parameter], "device": str(step.device),
                          "dtype": str(step.dtype), "value": step.item(),
                          "eligible_cpu": not group.get("capturable", False) and not group.get("fused", False)})
            for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if field in state:
                    tensor = state[field]
                    moments.append({"parameter": names[parameter], "field": field,
                                    "device": str(tensor.device), "dtype": str(tensor.dtype),
                                    "shape": list(tensor.shape), "elements": tensor.numel()})
    return {
        "parameter_tensors": len(parameters), "parameter_elements": sum(p.numel() for p in parameters),
        "parameter_devices": dict(Counter(str(p.device) for p in parameters)),
        "all_parameters_cuda": bool(parameters) and all(p.is_cuda for p in parameters),
        "moment_tensors": len(moments), "moment_devices": dict(Counter(m["device"] for m in moments)),
        "all_moments_cuda": bool(moments) and all(m["device"].startswith("cuda:") for m in moments),
        "step_count": len(steps), "step_devices": dict(Counter(s["device"] for s in steps)),
        "eligible_cuda_steps": sum(s["eligible_cpu"] and s["device"].startswith("cuda:") for s in steps),
        "normalized_moved_count": int(getattr(model, "_normalized_optimizer_counters", 0)),
        "steps": steps, "moments": moments, "param_groups": groups,
    }


class SyntheticCudaEnv:
    """Deterministic, action-dependent CUDA transitions with genuine terminations.

    No timeout occurs. Continuous features vary across lanes and ticks. Terminal
    flags exercise GAE masking while fixed autoreset observations keep the
    recurrence bounded; this environment makes no claims about navigation.
    """
    def __init__(self, observation_space: Any, action_space: Any, *, n_envs: int, device: Any):
        import torch

        self.observation_space, self.action_space = observation_space, action_space
        self.num_envs, self.device = n_envs, torch.device(device)
        if self.device.type != "cuda":
            raise QualificationError("synthetic qualification environment is CUDA-only")
        if len(observation_space.shape) != 1 or action_space.shape != (10,):
            raise QualificationError("checkpoint must use 1D Box observations and ten Box actions")
        self.lanes = torch.arange(n_envs, device=self.device, dtype=torch.int64)
        features = torch.arange(observation_space.shape[0], device=self.device, dtype=torch.float32)
        self.initial = (((self.lanes[:, None].float() * 7 + features[None, :] * 3) % 31) - 15) / 32
        self.observations = self.initial.clone()
        self.ticks = 0
        self.no_timeouts = torch.zeros(n_envs, dtype=torch.bool, device=self.device)

    def reset_tensor(self, seeds=None):
        self.ticks = 0
        self.observations.copy_(self.initial)
        return self.observations

    def step_tensor(self, clipped_actions):
        import torch
        from ai2.training.tensor_ppo import TensorStep

        self.ticks += 1
        influence = torch.tanh(clipped_actions).mean(dim=1)
        next_observations = self.observations * 0.91 + self.initial * 0.07 + influence[:, None] * 0.02
        reward = (0.1 * influence - 0.01 * clipped_actions.square().mean(dim=1)
                  + (self.lanes % 7).float() * 0.001)
        terminated = ((self.lanes + self.ticks) % 37) == 0
        self.observations.copy_(torch.where(terminated[:, None], self.initial, next_observations))
        return TensorStep(self.observations, reward, terminated, self.no_timeouts, next_observations)

    def host_has_timeouts(self):
        return False

    def close(self):
        pass


def _require_cuda() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise QualificationError("actual CUDA device required; CPU training cannot qualify this pipeline")
    device = torch.device("cuda", torch.cuda.current_device())
    check = torch.arange(4, dtype=torch.float32, device=device).square()
    torch.cuda.synchronize(device)
    if not check.is_cuda or check.sum().item() != 14.0:
        raise QualificationError("actual CUDA tensor computation failed")
    return device


def _checkpoint_data(checkpoint: Path) -> dict[str, Any]:
    from stable_baselines3.common.save_util import json_to_data

    with zipfile.ZipFile(checkpoint) as archive:
        return json_to_data(archive.read("data").decode())


def _load_arm(options: Any, checkpoint: Path, data: dict[str, Any], args: Any, device: Any) -> Any:
    from ai2.training.experimental_native_lane_training import make_ppo_class
    from ai2.training.tensor_ppo import TensorVecEnv

    environment = SyntheticCudaEnv(data["observation_space"], data["action_space"], n_envs=args.n_envs, device=device)
    tensor_env = TensorVecEnv(environment, device=device)
    model = make_ppo_class(options).load(checkpoint, env=tensor_env, device=str(device),
                                         n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.epochs)
    if model.device.type != "cuda":
        raise QualificationError("loaded policy is not CUDA")
    return model


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_receipt(output: Path, receipt: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(args: Any, receipt: dict[str, Any]) -> None:
    import torch
    import stable_baselines3
    from stable_baselines3.common.logger import configure
    from ai2.training.ppo_update_optimizations import sb3_train_source_metadata

    device = _require_cuda()
    metadata = sb3_train_source_metadata()
    if not metadata["supported"]:
        raise QualificationError(f"unsupported SB3 train source: {metadata}")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    data = _checkpoint_data(checkpoint)
    properties = torch.cuda.get_device_properties(device)
    receipt.update({
        "source_hashes": source_manifest(), "sb3_train_source": metadata,
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint),
                       "bytes": checkpoint.stat().st_size, "saved_num_timesteps": data.get("num_timesteps"),
                       "observation_shape": list(data["observation_space"].shape),
                       "action_shape": list(data["action_space"].shape)},
        "runtime": {"torch": torch.__version__, "sb3": stable_baselines3.__version__,
                    "cuda_runtime": torch.version.cuda, "device": str(device), "gpu": properties.name,
                    "gpu_memory_bytes": properties.total_memory,
                    "distribution_validation": bool(torch.distributions.Distribution._validate_args)},
        "snapshot_storage": "temporary CPU snapshots on disk; one comparison loaded at a time",
    })
    if not receipt["runtime"]["distribution_validation"]:
        raise QualificationError("existing distribution argument validation must be enabled")

    with tempfile.TemporaryDirectory(prefix="ai2-ppo-qualification-") as temporary:
        snapshot_dir = Path(temporary)
        baseline_eligible_steps = None
        for name, options in optimization_arms().items():
            _seed_all(args.seed)
            model = _load_arm(options, checkpoint, data, args, device)
            census = optimizer_census(model)
            validate_census(census, normalized=options.optimizer_cpu_steps)
            if name == "baseline":
                baseline_eligible_steps = census["eligible_cuda_steps"]
            if options.optimizer_cpu_steps and census["normalized_moved_count"] != baseline_eligible_steps:
                raise QualificationError(f"{name}: moved counter count differs from baseline eligible CUDA counters")
            arm = {"options": asdict(options), "loaded_optimizer": census, "updates": []}
            receipt["arms"][name] = arm
            model.set_logger(configure(folder=None, format_strings=[]))
            total_timesteps, callback = model._setup_learn(
                args.n_envs * args.n_steps * args.updates, callback=None, reset_num_timesteps=False)
            _seed_all(args.seed)
            initial = {"model": _model_snapshot(model), "rng": _rng_state()}
            if name == "baseline":
                torch.save(initial, snapshot_dir / "initial.pt")
            else:
                compare_snapshots(torch.load(snapshot_dir / "initial.pt", map_location="cpu", weights_only=False), initial)
            arm["initial_exact"] = True
            arm["initial_rng"] = _rng_hashes(initial["rng"])
            del initial
            print(json.dumps({"arm": name, "phase": "loaded", "moved_count": census["normalized_moved_count"],
                              "step_count": census["step_count"], "step_devices": census["step_devices"]}), flush=True)

            for update in range(args.updates):
                if not model.collect_rollouts(model.env, callback, model.rollout_buffer, args.n_steps):
                    raise QualificationError(f"{name}: rollout callback cancelled")
                after_rollout_rng = _rng_state()
                model._update_current_progress_remaining(model.num_timesteps, total_timesteps)
                model.train()
                torch.cuda.synchronize(device)
                current = _model_snapshot(model)
                current.update({"rollout": _host_tree({field: getattr(model.rollout_buffer, field) for field in ROLLOUT_FIELDS}),
                                "last_observations": _host_tree(model._last_obs),
                                "last_episode_starts": _host_tree(model._last_episode_starts),
                                "rng_after_rollout": after_rollout_rng, "rng_after_update": _rng_state(),
                                "logs": _host_tree(dict(model.logger.name_to_value))})
                path = snapshot_dir / f"update-{update}.pt"
                if name == "baseline":
                    comparison = compare_snapshots(current, current)
                    torch.save(current, path)
                else:
                    comparison = compare_snapshots(torch.load(path, map_location="cpu", weights_only=False), current)
                updated_census = optimizer_census(model)
                validate_census(updated_census, normalized=options.optimizer_cpu_steps)
                if not all(after["value"] > before["value"]
                           for before, after in zip(census["steps"], updated_census["steps"])):
                    raise QualificationError(f"{name}: optimizer counters did not advance during actual CUDA training")
                census = updated_census
                arm["updates"].append({"update": update + 1, "exact": True, "comparison": comparison,
                                       "optimizer": census, "n_updates": model._n_updates,
                                       "rng_after_rollout": _rng_hashes(after_rollout_rng),
                                       "rng_after_update": _rng_hashes(current["rng_after_update"]),
                                       "logs": _json_value(current["logs"])})
                del current
                _write_receipt(Path(args.output), receipt)
                print(json.dumps({"arm": name, "update": update + 1, "exact": True,
                                  "step_values": [step["value"] for step in census["steps"]]}), flush=True)

            saved = snapshot_dir / f"{name}-roundtrip.zip"
            before_reload = _model_snapshot(model)
            model.save(saved)
            model.tensor_vec_env.close()
            del callback, model
            gc.collect()
            torch.cuda.empty_cache()
            reloaded = _load_arm(options, saved, data, args, device)
            reload_comparison = compare_snapshots(before_reload, _model_snapshot(reloaded))
            reloaded_census = optimizer_census(reloaded)
            validate_census(reloaded_census, normalized=options.optimizer_cpu_steps)
            arm["save_reload"] = {"exact": True, "comparison": reload_comparison,
                                   "optimizer": reloaded_census, "saved_sha256": file_sha256(saved),
                                   "temporary_checkpoint_removed_after_qualification": True}
            reloaded.tensor_vec_env.close()
            del reloaded, before_reload
            gc.collect()
            torch.cuda.empty_cache()
            _write_receipt(Path(args.output), receipt)
    validate_completed_arms(receipt["arms"], updates=args.updates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for field in ("n_envs", "n_steps", "updates", "batch_size", "epochs"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.batch_size < 2 or args.n_envs * args.n_steps < 2:
        parser.error("PPO requires at least two samples and batch-size at least two")
    if Path(args.output).resolve() == Path(args.checkpoint).resolve():
        parser.error("output must not overwrite the input checkpoint")
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new receipt path")
    started = time.perf_counter()
    receipt = {"schema": SCHEMA, "status": "running", "skipped": 0, "arms": {},
               "started_at": datetime.now(timezone.utc).isoformat(), "configuration": vars(args),
               "scope": "selected checkpoint, synthetic CUDA env, exact rollout/update parity and optimizer save/reload",
               "not_qualified": ["navigation quality", "actual game dynamics", "timeout batching", "compiled arms", "performance"]}
    try:
        run(args, receipt)
        receipt.update(status="passed", exit_code=0)
    except Exception as exc:
        receipt.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    receipt["wall_seconds"] = time.perf_counter() - started
    _write_receipt(output, receipt)
    print(json.dumps({"status": receipt["status"], "output": str(output),
                      "error": receipt.get("error"), "exit_code": receipt["exit_code"]}), flush=True)
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
