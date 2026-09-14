#!/usr/bin/env python3
"""Policy-driven rollout parity diagnostic for stock vs grouped VecEnv.

This is a diagnostic harness only. It loads the frozen phase-0b environment
through ``train_job.build_env``, samples CUDA policy actions from a checkpoint,
applies the same action batch to stock and grouped envs, and records where the
two trajectories first diverge before any PPO update is run.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import benchmark_campaign


PHASE_KEY = "0b"
DEFAULT_SEED = 45
DEFAULT_STEPS = 1000
DEFAULT_GROUP_SIZE = 4


@contextmanager
def cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    return {
        "path": str(resolved),
        "exists": True,
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def array_record(value: Any) -> dict[str, Any]:
    """Return a stable, JSON-safe digest record for numpy arrays/action batches."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        payload = value.tolist()
    else:
        payload = value
    shape = list(getattr(value, "shape", []))
    dtype = str(getattr(value, "dtype", type(value).__name__))
    inferred_shape = _infer_shape(payload)
    if inferred_shape and (not shape or _shape_product(shape) != _shape_product(inferred_shape)):
        shape = inferred_shape
    encoded = _compact_json_bytes(payload)
    record = {
        "dtype": dtype,
        "shape": shape,
        "sha256": sha256_bytes(encoded),
        "nbytes_json": len(encoded),
    }
    try:
        flat = list(_flatten(payload))
        numeric = [float(item) for item in flat if isinstance(item, (int, float))]
        if numeric:
            record.update(
                {
                    "min": min(numeric),
                    "max": max(numeric),
                    "sum": sum(numeric),
                    "preview": numeric[:8],
                }
            )
    except Exception:
        record["preview"] = repr(payload)[:160]
    return record


def _infer_shape(value: Any) -> list[int]:
    shape = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else None
    return shape


def _shape_product(shape: list[int]) -> int:
    product = 1
    for item in shape:
        product *= int(item)
    return product


def _flatten(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def sanitize(value: Any, np_module: Any, path: tuple[str, ...] = ()) -> Any:
    """Make info/reset-info JSON-safe, excluding only Monitor episode wallclock t."""

    if isinstance(value, getattr(np_module, "ndarray", ())):
        return array_record(value)
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "t" and path and path[-1] == "episode":
                continue
            cleaned[str(key)] = sanitize(item, np_module, (*path, str(key)))
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize(item, np_module, path) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return repr(value)
    return copy.deepcopy(value)


def _as_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_as_list(item) for item in value]
    return value


def _compare_scalar(left: Any, right: Any) -> str | None:
    if left == right:
        return None
    return f"expected={left!r} actual={right!r}"


def _compare_nested(left: Any, right: Any, field: str, env: Any = "all") -> dict[str, Any] | None:
    left = _as_list(left)
    right = _as_list(right)
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            return {"env": env, "field": field, "detail": f"type expected={type(left).__name__} actual={type(right).__name__}"}
        if len(left) != len(right):
            return {"env": env, "field": field, "detail": f"length expected={len(left)} actual={len(right)}"}
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            child_env = index if env == "all" and field in {"obs", "reward", "done"} else env
            child_field = field if field in {"obs", "reward", "done"} else f"{field}[{index}]"
            diff = _compare_nested(left_item, right_item, child_field, child_env)
            if diff is not None:
                if field in {"obs", "reward", "done"}:
                    diff["detail"] = f"index={_index_detail(left, right)} {diff['detail']}"
                return diff
        return None
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return {"env": env, "field": field, "detail": f"type expected={type(left).__name__} actual={type(right).__name__}"}
        if set(left) != set(right):
            return {"env": env, "field": field, "detail": f"keys expected={sorted(left)} actual={sorted(right)}"}
        for key in sorted(left):
            if key == "t" and field.endswith("episode"):
                continue
            diff = _compare_nested(left[key], right[key], f"{field}.{key}", env)
            if diff is not None:
                return diff
        return None
    detail = _compare_scalar(left, right)
    if detail is not None:
        return {"env": env, "field": field, "detail": detail}
    return None


def _index_detail(left: Any, right: Any) -> tuple[int, ...] | str:
    def walk(a: Any, b: Any, prefix: tuple[int, ...] = ()) -> tuple[int, ...] | None:
        if isinstance(a, list) and isinstance(b, list):
            for idx, (ai, bi) in enumerate(zip(a, b)):
                found = walk(ai, bi, (*prefix, idx))
                if found is not None:
                    return found
            return None
        return prefix if a != b else None

    return walk(left, right) or "unknown"


def _sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "tick": record["tick"],
        "obs": _as_list(record["obs"]),
        "reward": _as_list(record.get("reward", [])),
        "done": _as_list(record.get("done", [])),
        "infos": _drop_episode_t(copy.deepcopy(record.get("infos", []))),
        "reset_infos": _drop_episode_t(copy.deepcopy(record.get("reset_infos", []))),
    }


def _drop_episode_t(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_episode_t(item, (*path, str(key)))
            for key, item in value.items()
            if not (key == "t" and path and path[-1] == "episode")
        }
    if isinstance(value, (list, tuple)):
        return [_drop_episode_t(item, path) for item in value]
    return value


def first_divergence(left_records: list[dict[str, Any]], right_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(left_records) != len(right_records):
        return {"tick": "all", "env": "all", "field": "trace_length", "detail": f"expected={len(left_records)} actual={len(right_records)}"}
    for left_raw, right_raw in zip(left_records, right_records):
        left = _sanitize_record(left_raw)
        right = _sanitize_record(right_raw)
        tick = left["tick"]
        if tick != right["tick"]:
            return {"tick": tick, "env": "all", "field": "tick", "detail": f"expected={tick!r} actual={right['tick']!r}"}
        for field in ("obs", "reward", "done", "infos", "reset_infos"):
            if tick == "reset" and field in {"reward", "done", "infos"}:
                continue
            diff = _compare_nested(left[field], right[field], field)
            if diff is not None:
                diff["tick"] = tick
                return diff
    return None


def action_sampling_summary(natural_stock: Any, natural_grouped: Any, shared_action: Any) -> dict[str, Any]:
    diff = _compare_nested(natural_stock, natural_grouped, "natural_action")
    return {
        "natural_policy_actions_equal": diff is None,
        "natural_action_diff": None if diff is None else {
            "index": list(_index_detail(_as_list(natural_stock), _as_list(natural_grouped)))
            if isinstance(_index_detail(_as_list(natural_stock), _as_list(natural_grouped)), tuple)
            else _index_detail(_as_list(natural_stock), _as_list(natural_grouped)),
            "detail": diff["detail"],
        },
        "shared_action": array_record(shared_action),
    }


def build_provenance(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "diagnostic": "policy_rollout_parity",
        "created_unix": time.time(),
        "argv": list(sys.argv),
        "runtime": {
            "device": str(args.device),
            "phase": str(args.phase),
            "n_envs": int(args.n_envs),
            "steps": int(args.steps),
            "group_size": int(args.group_size),
            "seed": int(args.seed),
        },
        "files": {
            "config": file_record(Path(args.config)),
            "checkpoint": file_record(Path(args.checkpoint)),
            "source_train": file_record(Path(args.source) / "train.py"),
            "train_job": file_record(Path(args.train_job)),
            "grouped_env": file_record(Path(args.grouped_env_path)),
        },
        "notes": [
            "No PPO update is run.",
            "Policy actions are sampled on CUDA from the checkpoint policy.",
            "The shared action batch is sampled once and applied identically to both env arms each step.",
            "Monitor episode.t is excluded from info comparison; other info/reset_info fields are compared.",
        ],
    }


def seed_vec_envs(stock_env: Any, grouped_env: Any, seed: int) -> None:
    stock_env.seed(int(seed))
    grouped_env.seed(int(seed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=benchmark_campaign.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=benchmark_campaign.DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=benchmark_campaign.DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=benchmark_campaign.DEFAULT_TRAIN_JOB)
    parser.add_argument("--grouped-env-path", type=Path, default=Path(__file__).resolve().parent / "grouped_vec_env.py")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", default=PHASE_KEY)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--time-limit-seconds", type=float)
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if not str(args.device).startswith("cuda"):
        raise ValueError("policy rollout parity requires --device cuda*")
    if args.n_envs <= 0 or args.steps <= 0 or args.group_size <= 1:
        raise ValueError("--n-envs and --steps must be positive; --group-size must be > 1")
    args.output = Path(args.output).resolve()
    for attr in ("config", "checkpoint", "source", "train_job", "grouped_env_path"):
        setattr(args, attr, Path(getattr(args, attr)).resolve())
    return args


def load_train_job_module(path: Path) -> ModuleType:
    """Load train_job through the campaign importer so spawn can re-import it."""

    return benchmark_campaign.import_train_job_module(Path(path))


def load_grouped_env_class(path: Path) -> type:
    """Load GroupedSubprocVecEnv under its real filename for spawn workers."""

    grouped_module = benchmark_campaign.load_grouped_env_module(Path(path))
    grouped_cls = getattr(grouped_module, "GroupedSubprocVecEnv", None)
    if not isinstance(grouped_cls, type):
        raise TypeError(f"{path} did not define GroupedSubprocVecEnv")
    return grouped_cls


def phase_lookup(params: dict[str, Any], phase_key: str) -> dict[str, Any]:
    phases = params.get("phases", {})
    if phase_key in phases:
        return phases[phase_key]
    for key, value in phases.items():
        if str(key) == str(phase_key):
            return value
    raise KeyError(f"phase {phase_key!r} not found")


def prepare_params(params: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    adjusted = copy.deepcopy(params)
    adjusted.setdefault("curriculum", {})["phase"] = str(args.phase)
    adjusted["frame_stack"] = 4
    adjusted.setdefault("environment_settings", {})["n_envs"] = int(args.n_envs)
    adjusted["environment_settings"]["resolution"] = adjusted["environment_settings"].get("resolution", "LOW")
    phase = phase_lookup(adjusted, str(args.phase))
    if phase.get("type") != "navigation":
        raise ValueError(f"phase {args.phase!r} must be navigation, got {phase.get('type')!r}")
    if args.time_limit_seconds is not None:
        phase.setdefault("navigation", {})["time_limit_seconds"] = float(args.time_limit_seconds)
    phase["opponents"] = {"constant_agent": 1.0}
    adjusted.setdefault("opponent_history", {})["enabled"] = bool(
        adjusted.get("opponent_history", {}).get("enabled", True)
    )
    return adjusted


def build_env(train_job: ModuleType, train_module: ModuleType, params: dict[str, Any], args: argparse.Namespace, *, grouped: bool):
    phase_config = train_module.PhaseConfig(params, str(args.phase))
    if not phase_config.is_navigation:
        raise ValueError(f"phase {args.phase!r} unexpectedly built as non-navigation")
    if not grouped:
        return train_job.build_env(train_module, phase_config, params, int(args.n_envs), int(args.seed), str(args.device))

    grouped_cls = load_grouped_env_class(args.grouped_env_path)
    original = train_job.SubprocVecEnv

    def grouped_subproc_vec_env(env_fns: Any, start_method: str | None = None):
        return grouped_cls(env_fns, group_size=int(args.group_size), start_method=start_method)

    train_job.SubprocVecEnv = grouped_subproc_vec_env
    try:
        return train_job.build_env(train_module, phase_config, params, int(args.n_envs), int(args.seed), str(args.device))
    finally:
        train_job.SubprocVecEnv = original


def set_parent_seeds(seed: int, torch_module: Any, np_module: Any) -> None:
    random.seed(int(seed))
    np_module.random.seed(int(seed))
    torch_module.manual_seed(int(seed))
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(int(seed))


def torch_rng_state(torch_module: Any) -> dict[str, Any]:
    state = {
        "cpu": base64.b64encode(torch_module.get_rng_state().cpu().numpy().tobytes()).decode("ascii"),
        "cuda": [],
    }
    if torch_module.cuda.is_available():
        state["cuda"] = [
            base64.b64encode(item.cpu().numpy().tobytes()).decode("ascii")
            for item in torch_module.cuda.get_rng_state_all()
        ]
    return state


def restore_torch_rng_state(torch_module: Any, state: dict[str, Any]) -> None:
    cpu_bytes = base64.b64decode(state["cpu"].encode("ascii"))
    cpu_tensor = torch_module.ByteTensor(list(cpu_bytes))
    torch_module.set_rng_state(cpu_tensor)
    cuda_states = []
    for item in state.get("cuda", []):
        cuda_states.append(torch_module.ByteTensor(list(base64.b64decode(item.encode("ascii")))))
    if cuda_states and torch_module.cuda.is_available():
        torch_module.cuda.set_rng_state_all(cuda_states)


def torch_rng_fingerprint(torch_module: Any) -> dict[str, Any]:
    state = torch_rng_state(torch_module)
    return {
        "cpu_sha256": sha256_bytes(state["cpu"].encode("ascii")),
        "cuda_sha256": [sha256_bytes(item.encode("ascii")) for item in state.get("cuda", [])],
    }


def sample_policy_action(model: Any, obs: Any, action_space: Any, torch_module: Any, np_module: Any, device: str) -> Any:
    obs_tensor = torch_module.as_tensor(obs, device=device)
    with torch_module.no_grad():
        actions, _values, _log_probs = model.policy(obs_tensor)
    action_np = actions.detach().cpu().numpy()
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is not None and high is not None:
        action_np = np_module.clip(action_np, low, high)
    return action_np.astype(getattr(action_space, "dtype", action_np.dtype), copy=False)


def capture_env_record(env: Any, tick: int | str, obs: Any, np_module: Any, rewards: Any = None, dones: Any = None, infos: Any = None) -> dict[str, Any]:
    record = {
        "tick": tick,
        "obs": array_record(obs),
        "reset_infos": sanitize(getattr(env, "reset_infos", []), np_module),
    }
    if rewards is not None:
        record["reward"] = array_record(rewards)
    if dones is not None:
        record["done"] = array_record(dones)
    if infos is not None:
        record["infos"] = sanitize(infos, np_module)
    return record


def compare_runtime_records(left: dict[str, Any], right: dict[str, Any], raw_left: dict[str, Any], raw_right: dict[str, Any]) -> dict[str, Any] | None:
    diff = first_divergence([raw_left], [raw_right])
    if diff is not None:
        return diff
    for field in ("obs", "reward", "done"):
        if field in left and left[field]["sha256"] != right[field]["sha256"]:
            return {
                "tick": left["tick"],
                "env": "all",
                "field": field,
                "detail": f"sha256 expected={left[field]['sha256']} actual={right[field]['sha256']}",
            }
    if left.get("infos") != right.get("infos"):
        return {"tick": left["tick"], "env": "all", "field": "infos", "detail": "sanitized infos differ"}
    if left.get("reset_infos") != right.get("reset_infos"):
        return {"tick": left["tick"], "env": "all", "field": "reset_infos", "detail": "sanitized reset_infos differ"}
    return None


def run_diagnostic(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for policy rollout parity diagnostic")

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "provenance.json", build_provenance(args))
    trace_path = args.output / "policy_rollout_trace.jsonl"
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("WANDB_MODE", "offline")

    train_job = load_train_job_module(args.train_job)
    if str(args.source) not in sys.path:
        sys.path.insert(0, str(args.source))

    with cwd(args.source):
        train_module = train_job.load_runtime_train_module()
        with Path(args.config).open("r", encoding="utf-8") as handle:
            params = prepare_params(yaml.safe_load(handle), args)

        stock_env = None
        grouped_env = None
        first_diff = None
        status = "matched"
        try:
            set_parent_seeds(args.seed, torch, np)
            rng_before_env_build = torch_rng_fingerprint(torch)
            stock_env = build_env(train_job, train_module, params, args, grouped=False)
            rng_after_stock_build = torch_rng_fingerprint(torch)
            grouped_env = build_env(train_job, train_module, params, args, grouped=True)
            rng_after_grouped_build = torch_rng_fingerprint(torch)
            rng_before_model = torch_rng_fingerprint(torch)
            model = train_job.PPO.load(str(args.checkpoint), device=str(args.device))
            model.policy.set_training_mode(False)
            rng_after_model = torch_rng_fingerprint(torch)
            set_parent_seeds(args.seed, torch, np)
            if hasattr(model, "set_random_seed"):
                model.set_random_seed(int(args.seed))
            seed_vec_envs(stock_env, grouped_env, int(args.seed))
            rng_after_explicit_seed = torch_rng_fingerprint(torch)

            stock_obs = stock_env.reset()
            stock_reset_record = capture_env_record(stock_env, "reset", stock_obs, np)
            grouped_obs = grouped_env.reset()
            grouped_reset_record = capture_env_record(grouped_env, "reset", grouped_obs, np)
            write_json(
                args.output / "initial_reset.json",
                {
                    "stock": stock_reset_record,
                    "grouped": grouped_reset_record,
                    "divergence": compare_runtime_records(
                        stock_reset_record,
                        grouped_reset_record,
                        {"tick": "reset", "obs": stock_obs, "reset_infos": getattr(stock_env, "reset_infos", [])},
                        {"tick": "reset", "obs": grouped_obs, "reset_infos": getattr(grouped_env, "reset_infos", [])},
                    ),
                },
            )

            forced_state = torch_rng_state(torch)
            natural_before = torch_rng_fingerprint(torch)
            natural_stock = sample_policy_action(model, stock_obs, stock_env.action_space, torch, np, str(args.device))
            natural_after_stock = torch_rng_fingerprint(torch)
            natural_grouped = sample_policy_action(model, grouped_obs, grouped_env.action_space, torch, np, str(args.device))
            natural_after_grouped = torch_rng_fingerprint(torch)
            restore_torch_rng_state(torch, forced_state)
            forced_stock = sample_policy_action(model, stock_obs, stock_env.action_space, torch, np, str(args.device))
            restore_torch_rng_state(torch, forced_state)
            forced_grouped = sample_policy_action(model, grouped_obs, grouped_env.action_space, torch, np, str(args.device))
            shared_action = forced_stock
            write_json(
                args.output / "action_probe.json",
                {
                    **action_sampling_summary(natural_stock, natural_grouped, shared_action),
                    "forced_same_rng_stock_action": array_record(forced_stock),
                    "forced_same_rng_grouped_obs_action": array_record(forced_grouped),
                    "forced_same_rng_actions_equal": _compare_nested(forced_stock, forced_grouped, "forced_action") is None,
                    "rng": {
                        "before_env_build": rng_before_env_build,
                        "after_stock_build": rng_after_stock_build,
                        "after_grouped_build": rng_after_grouped_build,
                        "before_model_load": rng_before_model,
                        "after_model_load": rng_after_model,
                        "after_explicit_parent_model_and_vecenv_seed": rng_after_explicit_seed,
                        "before_natural_policy_sampling": natural_before,
                        "after_natural_stock_policy_sampling": natural_after_stock,
                        "after_natural_grouped_policy_sampling": natural_after_grouped,
                        "forced_state_reused_for_both_arms": {
                            "cpu_sha256": sha256_bytes(forced_state["cpu"].encode("ascii")),
                            "cuda_sha256": [sha256_bytes(item.encode("ascii")) for item in forced_state.get("cuda", [])],
                        },
                    },
                },
            )

            reset_diff = compare_runtime_records(
                stock_reset_record,
                grouped_reset_record,
                {"tick": "reset", "obs": stock_obs, "reset_infos": getattr(stock_env, "reset_infos", [])},
                {"tick": "reset", "obs": grouped_obs, "reset_infos": getattr(grouped_env, "reset_infos", [])},
            )
            if reset_diff is not None and first_diff is None:
                first_diff = reset_diff

            for tick in range(int(args.steps)):
                if tick > 0:
                    shared_action = sample_policy_action(model, stock_obs, stock_env.action_space, torch, np, str(args.device))
                stock_next = stock_env.step(shared_action)
                grouped_next = grouped_env.step(shared_action)
                stock_obs, stock_reward, stock_done, stock_infos = stock_next
                grouped_obs, grouped_reward, grouped_done, grouped_infos = grouped_next
                stock_record = capture_env_record(stock_env, tick, stock_obs, np, stock_reward, stock_done, stock_infos)
                grouped_record = capture_env_record(grouped_env, tick, grouped_obs, np, grouped_reward, grouped_done, grouped_infos)
                raw_stock = {
                    "tick": tick,
                    "obs": stock_obs,
                    "reward": stock_reward,
                    "done": stock_done,
                    "infos": stock_infos,
                    "reset_infos": getattr(stock_env, "reset_infos", []),
                }
                raw_grouped = {
                    "tick": tick,
                    "obs": grouped_obs,
                    "reward": grouped_reward,
                    "done": grouped_done,
                    "infos": grouped_infos,
                    "reset_infos": getattr(grouped_env, "reset_infos", []),
                }
                divergence = compare_runtime_records(stock_record, grouped_record, raw_stock, raw_grouped)
                if divergence is not None and first_diff is None:
                    first_diff = divergence
                append_jsonl(
                    trace_path,
                    {
                        "tick": tick,
                        "action": array_record(shared_action),
                        "stock": stock_record,
                        "grouped": grouped_record,
                        "divergence": divergence,
                    },
                )
            if first_diff is not None:
                status = "diverged"
        finally:
            for env in (stock_env, grouped_env):
                if env is not None:
                    env.close()

    write_json(
        args.output / "summary.json",
        {
            "status": status,
            "first_divergence": first_diff,
            "steps_requested": int(args.steps),
            "trace_path": str(trace_path),
        },
    )
    print(f"summary={args.output / 'summary.json'}")
    return 0 if status == "matched" else 2


def main(argv: list[str] | None = None) -> int:
    args = validate_args(build_parser().parse_args(argv))
    return run_diagnostic(args)


if __name__ == "__main__":
    raise SystemExit(main())
