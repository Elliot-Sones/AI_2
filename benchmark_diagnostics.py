#!/usr/bin/env python3
"""Standalone E1 diagnostic ceilings for the frozen AI_2 PPO workload.

This script is separate from the A/B campaign harness. It runs finite diagnostic
workloads against the frozen navigation-repair train_job/checkpoint: fixed-action
vector-env stepping, policy inference batches, paired real rollout+update passes,
and an optional short torch.profiler CPU/CUDA trace. Results are labeled as
diagnostic ceilings, not full campaign baselines.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import types
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import benchmark_training as bt


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_NAV_ROOT = (
    Path("/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905")
    / "baselines"
    / "navigation-repair-20260905"
)
DEFAULT_SOURCE = DEFAULT_NAV_ROOT / "payload" / "variants" / "corrected"
DEFAULT_TRAIN_JOB = DEFAULT_NAV_ROOT / "payload" / "scripts" / "train_job.py"
DEFAULT_CONFIG = DEFAULT_SOURCE / "control.yaml"
DEFAULT_CHECKPOINT = DEFAULT_NAV_ROOT / "payload" / "checkpoints" / "selected_16001056.zip"
GPU_TRACE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
ROLLOUT_ARRAY_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "episode_starts",
    "values",
    "log_probs",
    "advantages",
    "returns",
)
ROLLOUT_META_FIELDS = ("pos", "full", "buffer_size", "n_envs")


def union_intervals(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged = []
    for start, end in sorted((float(s), float(e)) for s, e in intervals if e >= s):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def summarize_torch_trace_events(trace: dict[str, Any]) -> dict[str, Any]:
    events = trace.get("traceEvents", [])
    gpu_intervals: list[tuple[float, float]] = []
    starts: list[float] = []
    ends: list[float] = []
    kernel_us_sum = 0.0
    copy_us_sum = 0.0
    counts: dict[str, int] = {}
    for event in events:
        if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
            continue
        cat = str(event.get("cat", ""))
        counts[cat] = counts.get(cat, 0) + 1
        start = float(event.get("ts", 0.0))
        end = start + float(event["dur"])
        starts.append(start)
        ends.append(end)
        if cat in GPU_TRACE_CATEGORIES:
            gpu_intervals.append((start, end))
            if cat == "kernel":
                kernel_us_sum += float(event["dur"])
            else:
                copy_us_sum += float(event["dur"])
    active = union_intervals(gpu_intervals)
    span = max(ends) - min(starts) if starts and ends else 0.0
    return {
        "kernel_us_sum": kernel_us_sum,
        "copy_us_sum": copy_us_sum,
        "gpu_active_union_us": active,
        "trace_span_us": span,
        "gpu_active_fraction_of_trace_span": active / span if span else None,
        "event_counts": counts,
        "note": "GPU active time is a union of overlapping kernel/memcpy/memset trace spans, not a sum.",
    }


def phase_rate(phase: dict[str, Any]) -> dict[str, Any]:
    result = dict(phase)
    seconds = float(result.get("seconds") or 0.0)
    if "steps" in result:
        result["steps_per_second"] = float(result["steps"]) / seconds if seconds else 0.0
    if "batches" in result:
        result["batches_per_second"] = float(result["batches"]) / seconds if seconds else 0.0
    if "updates" in result:
        result["updates_per_second"] = float(result["updates"]) / seconds if seconds else 0.0
    return result


def build_summary(*, phases: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": "e1_diagnostic_ceilings",
        "phase_count": len(phases),
        "phases": [phase_rate(phase) for phase in phases],
        "provenance": provenance,
        "notes": [
            "Diagnostic ceilings are finite microbenchmarks, not full-training throughput claims.",
            "Torch trace GPU active time is a union of overlapping GPU spans.",
        ],
    }


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


def import_real_module(path: Path, alias: str | None = None):
    resolved = Path(path).resolve()
    parent = str(resolved.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(resolved.stem, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[resolved.stem] = module
    if alias:
        sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@dataclass
class ComponentTiming:
    count: int = 0
    seconds: float = 0.0

    @property
    def mean_seconds(self) -> float:
        return self.seconds / self.count if self.count else 0.0

    def to_json(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "seconds": self.seconds,
            "mean_seconds": self.mean_seconds,
        }


@dataclass
class ComponentStats:
    enabled: bool = False
    by_label: dict[str, ComponentTiming] = field(default_factory=dict)

    def record(self, label: str, seconds: float) -> None:
        if not self.enabled:
            return
        item = self.by_label.setdefault(label, ComponentTiming())
        item.count += 1
        item.seconds += seconds

    @contextmanager
    def measure_phase(self):
        previous = self.enabled
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = previous

    def to_json(self) -> dict[str, dict[str, float | int]]:
        return {label: timing.to_json() for label, timing in sorted(self.by_label.items())}


class TimedPatch:
    """Patch a method for scoped inclusive timing, then restore it exactly."""

    def __init__(
        self,
        target: Any,
        method_name: str,
        label: str,
        stats: ComponentStats,
        sync: Callable[[], None] | None = None,
    ) -> None:
        self.target = target
        self.method_name = method_name
        self.label = label
        self.stats = stats
        self.sync = sync or (lambda: None)
        self._original: Any = None
        self._had_instance_attr = False
        self._instance_attr: Any = None

    def __enter__(self):
        self.stats.by_label.setdefault(self.label, ComponentTiming())
        self._original = getattr(self.target, self.method_name)
        self._had_instance_attr = self.method_name in getattr(self.target, "__dict__", {})
        if self._had_instance_attr:
            self._instance_attr = self.target.__dict__[self.method_name]

        def timed(instance, *args, **kwargs):
            if not self.stats.enabled:
                return self._original(*args, **kwargs)
            self.sync()
            start = time.perf_counter()
            try:
                return self._original(*args, **kwargs)
            finally:
                self.sync()
                self.stats.record(self.label, time.perf_counter() - start)

        setattr(self.target, self.method_name, types.MethodType(timed, self.target))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._had_instance_attr:
            setattr(self.target, self.method_name, self._instance_attr)
        else:
            try:
                delattr(self.target, self.method_name)
            except AttributeError:
                pass


class RolloutTiming:
    def __init__(self, n_envs: int) -> None:
        self.n_envs = n_envs
        self.rollout_seconds: list[float] = []
        self.episodes: list[dict[str, Any]] = []
        self.episode_steps = [0 for _ in range(n_envs)]
        self.episode_start = [time.perf_counter() for _ in range(n_envs)]
        self._rollout_start: float | None = None

    def make_callback(self):
        owner = self
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except Exception:
            BaseCallback = object

        class _Callback(BaseCallback):
            def __init__(self) -> None:
                if BaseCallback is not object:
                    super().__init__(verbose=0)
                self.owner = owner
                self.locals: dict[str, Any] = {}

            def _on_training_start(self) -> None:
                now = time.perf_counter()
                self.owner.episode_start = [now for _ in range(self.owner.n_envs)]
                self.owner.episode_steps = [0 for _ in range(self.owner.n_envs)]

            def _on_rollout_start(self) -> None:
                self.owner._rollout_start = time.perf_counter()

            def _on_rollout_end(self) -> None:
                if self.owner._rollout_start is not None:
                    self.owner.rollout_seconds.append(
                        time.perf_counter() - self.owner._rollout_start
                    )
                    self.owner._rollout_start = None

            def _on_step(self) -> bool:
                now = time.perf_counter()
                dones = self.locals.get("dones", [])
                infos = self.locals.get("infos", [])
                for idx, done in enumerate(dones):
                    self.owner.episode_steps[idx] += 1
                    if bool(done):
                        info = infos[idx] if idx < len(infos) else {}
                        self.owner.episodes.append(
                            {
                                "env_index": idx,
                                "episode_steps": self.owner.episode_steps[idx],
                                "wall_seconds": now - self.owner.episode_start[idx],
                                "info": bt.json_safe(
                                    {
                                        key: value
                                        for key, value in info.items()
                                        if key != "terminal_observation"
                                    }
                                ),
                            }
                        )
                        self.owner.episode_steps[idx] = 0
                        self.owner.episode_start[idx] = now
                return True

        return _Callback()


def default_raw_env_getter(vec_env: Any) -> Any | None:
    current = vec_env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "envs") and current.envs:
            current = current.envs[0]
            continue
        if hasattr(current, "raw_env"):
            return current.raw_env
        if hasattr(current, "env"):
            current = current.env
            continue
        if hasattr(current, "venv"):
            current = current.venv
            continue
        break
    return None


def _linear_or_value(configured: Any):
    if isinstance(configured, list):
        try:
            from train import linear_schedule
        except ModuleNotFoundError:
            def linear_schedule(initial, final):
                return lambda progress_remaining: progress_remaining * (float(initial) - float(final)) + float(final)

        return linear_schedule(configured[0], configured[1])
    return configured


def build_model(model_cls: type, env: Any, params: dict[str, Any], args: argparse.Namespace):
    ppo = params.get("ppo_settings", {})
    n_steps = int(args.n_steps or ppo.get("n_steps", 1024))
    batch_size = min(int(args.batch_size or ppo.get("batch_size", 8192)), n_steps)
    return model_cls(
        "MlpPolicy",
        env,
        learning_rate=_linear_or_value(ppo.get("learning_rate", [3e-4, 1e-6])),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(ppo.get("n_epochs", 8)),
        gamma=float(ppo.get("gamma", 0.99)),
        gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        clip_range=_linear_or_value(ppo.get("clip_range", [0.2, 0.05])),
        ent_coef=float(ppo.get("nav_ent_coef", ppo.get("ent_coef", 0.01))),
        vf_coef=float(ppo.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        verbose=0,
        tensorboard_log=None,
        device=args.device,
        seed=args.seed,
        policy_kwargs={"net_arch": params.get("policy_kwargs", {}).get("net_arch", [512, 512, 256])},
    )


def schedule_parity_model_class(model_cls: type, configured_budget: int) -> type:
    class ScheduleParityModel(model_cls):
        def __init__(self, *model_args, **model_kwargs) -> None:
            self.diagnostic_progress_total_timesteps: list[int] = []
            super().__init__(*model_args, **model_kwargs)

        def _update_current_progress_remaining(
            self, num_timesteps: int, total_timesteps: int
        ) -> None:
            schedule_total = max(configured_budget, total_timesteps)
            self.diagnostic_progress_total_timesteps.append(schedule_total)
            super()._update_current_progress_remaining(num_timesteps, schedule_total)

    return ScheduleParityModel


def _sync_for(device: str) -> Callable[[], None]:
    def sync() -> None:
        try:
            bt.sync_device(device)
        except ModuleNotFoundError:
            return

    return sync


def run_diagnostic(
    args: argparse.Namespace,
    output_dir: Path,
    env_builder: Callable[[Path, int, int, int], tuple[Any, dict[str, Any], Any]] | None = None,
    model_cls: type | None = None,
    raw_env_getter: Callable[[Any], Any | None] = default_raw_env_getter,
) -> dict[str, Any]:
    bt.set_headless_environment()
    env_builder = env_builder or bt.build_wrapped_nav_env
    if model_cls is None:
        from stable_baselines3 import PPO

        model_cls = PPO
    output_dir.mkdir(parents=True, exist_ok=True)

    bt.seed_everything(args.seed)
    sync = _sync_for(args.device)
    env, params, phase_config = env_builder(args.config, args.seed, 1, args.torch_threads)
    raw_env = raw_env_getter(env)
    stats = ComponentStats()
    timing = RolloutTiming(n_envs=1)
    n_steps = int(args.n_steps or params.get("ppo_settings", {}).get("n_steps", 1024))
    requested_batch_size = int(
        args.batch_size or params.get("ppo_settings", {}).get("batch_size", 8192)
    )
    effective_batch_size = min(requested_batch_size, n_steps)
    measured_timesteps = int(args.ppo_rollouts * n_steps)
    configured_budget = int(
        params.get("ppo_settings", {}).get(
            "time_steps", max(n_steps + measured_timesteps, 1)
        )
    )
    profile_path = output_dir / "benchmark_diagnostics.prof" if args.profile else None

    try:
        model = build_model(
            schedule_parity_model_class(model_cls, configured_budget), env, params, args
        )
        with ExitStack() as stack:
            if hasattr(env, "step_wait"):
                stack.enter_context(TimedPatch(env, "step_wait", "vec_step_wait", stats, sync))
            if hasattr(env, "reset"):
                stack.enter_context(TimedPatch(env, "reset", "vec_reset", stats, sync))
            if raw_env is not None:
                if hasattr(raw_env, "step"):
                    stack.enter_context(TimedPatch(raw_env, "step", "game_step", stats, sync))
                if hasattr(raw_env, "reset"):
                    stack.enter_context(TimedPatch(raw_env, "reset", "game_reset", stats, sync))
            if hasattr(model, "policy") and hasattr(model.policy, "forward"):
                stack.enter_context(
                    TimedPatch(model.policy, "forward", "policy_forward", stats, sync)
                )
            stack.enter_context(TimedPatch(model, "train", "ppo_update", stats, sync))

            warmup_callback = RolloutTiming(n_envs=1).make_callback()
            model.learn(
                total_timesteps=n_steps,
                callback=warmup_callback,
                progress_bar=False,
                reset_num_timesteps=True,
            )

            measured_callback = timing.make_callback()
            profiler = cProfile.Profile() if profile_path is not None else None
            sync()
            start = time.perf_counter()
            with stats.measure_phase():
                if profiler is not None:
                    profiler.enable()
                try:
                    model.learn(
                        total_timesteps=measured_timesteps,
                        callback=measured_callback,
                        progress_bar=False,
                        reset_num_timesteps=True,
                    )
                finally:
                    if profiler is not None:
                        profiler.disable()
            sync()
            measured_seconds = time.perf_counter() - start
            if profiler is not None and profile_path is not None:
                profiler.dump_stats(str(profile_path))
    finally:
        env.close()

    update_count = stats.by_label.get("ppo_update", ComponentTiming()).count
    if len(timing.rollout_seconds) != args.ppo_rollouts:
        raise RuntimeError(
            f"Expected {args.ppo_rollouts} measured rollouts, got {len(timing.rollout_seconds)}"
        )
    if update_count != args.ppo_rollouts:
        raise RuntimeError(f"Expected {args.ppo_rollouts} measured PPO updates, got {update_count}")

    components = stats.to_json()
    result = {
        "label": "instrumented_single_env_ppo",
        "seed": args.seed,
        "device": args.device,
        "n_envs": 1,
        "n_steps": n_steps,
        "batch_size": effective_batch_size,
        "ppo_rollouts": args.ppo_rollouts,
        "configured_training_budget": configured_budget,
        "schedule_total_timesteps": max(configured_budget, measured_timesteps),
        "progress_total_timesteps": getattr(model, "diagnostic_progress_total_timesteps", []),
        "measured_timesteps": measured_timesteps,
        "measured_seconds": measured_seconds,
        "steps_per_second": measured_timesteps / measured_seconds if measured_seconds else 0.0,
        "rollout_count": len(timing.rollout_seconds),
        "rollout_seconds": timing.rollout_seconds,
        "rollout_seconds_total": sum(timing.rollout_seconds),
        "update_count": update_count,
        "update_seconds_total": components.get("ppo_update", {}).get("seconds", 0.0),
        "components": components,
        "completed_episodes": len(timing.episodes),
        "partial_episodes": sum(1 for steps in timing.episode_steps if steps),
        "profile_path": str(profile_path) if profile_path is not None else None,
        "phase": str(getattr(phase_config, "phase_key", "unknown")),
        "phase_name": getattr(phase_config, "name", "unknown"),
        "notes": [
            "Warmup rollout and model/env initialization are excluded from measured_seconds.",
            "Learning-rate and clip schedules use max(configured_training_budget, total_timesteps), matching benchmark_training.py short-run semantics.",
            "Component method timers are inclusive and may nest; do not sum them as exclusive wall time.",
            "vec_reset/game_reset counts are only reset calls that occur inside the measured learn window.",
        ],
    }
    payload = {"result": bt.json_safe(result), "episodes": bt.json_safe(timing.episodes)}
    (output_dir / "benchmark_diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def configure_runtime_env() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def seed_all(seed: int, device: str) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def load_training_context(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    import yaml

    train_job = import_real_module(args.train_job, alias="diagnostic_train_job")
    source = Path(args.source).resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    previous_cwd = Path.cwd()
    os.chdir(source)
    try:
        with Path(args.config).open(encoding="utf-8") as handle:
            raw_params = yaml.safe_load(handle) or {}
        train_module = train_job.load_runtime_train_module()
        params = train_job.prepare_params(
            raw_params,
            args.phase,
            output_dir,
            args.n_envs,
            args.steps,
            None,
            False,
            args.n_steps,
            args.batch_size,
            args.epochs,
        )
        checkpoint_path = train_job.resolve_checkpoint(params, str(args.checkpoint), output_dir)
        params.setdefault("ppo_settings", {})["model_checkpoint"] = str(checkpoint_path)
        phase_config = train_module.PhaseConfig(params, str(args.phase))
    finally:
        os.chdir(previous_cwd)
    return {
        "train_job": train_job,
        "train_module": train_module,
        "source": source,
        "params": params,
        "phase_config": phase_config,
        "checkpoint": checkpoint_path,
    }


def build_frozen_env(context: dict[str, Any], args: argparse.Namespace):
    with working_directory(context["source"]):
        return context["train_job"].build_env(
            context["train_module"],
            context["phase_config"],
            context["params"],
            args.n_envs,
            args.seed,
            args.device,
        )


def ppo_kwargs_for_context(context: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return context["train_job"].build_ppo_kwargs(
        context["params"],
        context["phase_config"].is_navigation,
        args.n_envs,
        args.device,
        output_dir / "tb",
    )


def load_agent(context: dict[str, Any], env: Any, args: argparse.Namespace, output_dir: Path, *, cache_class: type | None = None):
    ppo_kwargs = ppo_kwargs_for_context(context, args, output_dir)
    custom_objects = {
        "learning_rate": ppo_kwargs["learning_rate"],
        "clip_range": ppo_kwargs["clip_range"],
        "n_steps": ppo_kwargs["n_steps"],
        "batch_size": ppo_kwargs["batch_size"],
        "n_epochs": ppo_kwargs["n_epochs"],
    }
    if cache_class is not None:
        custom_objects["rollout_buffer_class"] = cache_class
    agent = context["train_job"].PPO.load(
        str(context["checkpoint"]),
        env=env,
        device=args.device,
        custom_objects=custom_objects,
    )
    if cache_class is not None:
        agent.rollout_buffer_class = cache_class
    for key in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef"):
        setattr(agent, key, ppo_kwargs[key])
    if hasattr(agent, "set_random_seed"):
        agent.set_random_seed(int(args.seed))
    return agent


def run_fixed_action_env(context: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    env = build_frozen_env(context, args)
    sync = _sync_for(args.device)
    try:
        env.reset()
        action_shape = getattr(env.action_space, "shape", ())
        action = np.zeros((args.n_envs, *action_shape), dtype=np.float32)
        for _ in range(args.env_warmup_steps):
            env.step(action)
        sync()
        start = time.perf_counter()
        for _ in range(args.env_steps):
            env.step(action)
        sync()
        seconds = time.perf_counter() - start
        return {
            "label": "fixed_action_env",
            "steps": int(args.env_steps * args.n_envs),
            "vector_steps": int(args.env_steps),
            "seconds": seconds,
            "n_envs": int(args.n_envs),
            "warmup_vector_steps": int(args.env_warmup_steps),
        }
    finally:
        env.close()


def batched_observation(obs: Any, batch_size: int):
    import numpy as np

    array = np.asarray(obs)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    first = array[:1]
    return np.repeat(first, int(batch_size), axis=0)


def run_inference_batches(context: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    import torch
    from stable_baselines3.common.utils import obs_as_tensor

    env = build_frozen_env(context, args)
    sync = _sync_for(args.device)
    try:
        agent = load_agent(context, env, args, output_dir / "inference")
        obs = env.reset()
        results = []
        for batch_size in args.inference_batches:
            batch = batched_observation(obs, batch_size)
            tensor = obs_as_tensor(batch, agent.device)
            with torch.no_grad():
                for _ in range(args.inference_warmup):
                    agent.policy(tensor)
                sync()
                start = time.perf_counter()
                for _ in range(args.inference_iters):
                    agent.policy(tensor)
                sync()
            seconds = time.perf_counter() - start
            results.append(
                {
                    "label": f"inference_{batch_size}",
                    "batch_size": int(batch_size),
                    "batches": int(args.inference_iters),
                    "items": int(batch_size * args.inference_iters),
                    "seconds": seconds,
                    "device": args.device,
                }
            )
        return results
    finally:
        env.close()


def load_cache_class(args: argparse.Namespace) -> type | None:
    if not args.rollout_cache_path:
        return None
    module = import_real_module(args.rollout_cache_path, alias="diagnostic_rollout_cache")
    cls = getattr(module, "CachedRolloutBuffer", None)
    if not isinstance(cls, type):
        raise TypeError(f"{args.rollout_cache_path} did not define CachedRolloutBuffer")
    return cls


def maybe_trace_profiler(args: argparse.Namespace):
    if not args.torch_trace:
        return None
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )


def clone_numeric_payload(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().clone()
    if hasattr(value, "copy") and callable(value.copy):
        try:
            return value.copy()
        except TypeError:
            pass
    return copy.deepcopy(value)


def iter_numeric_values(value: Any):
    if hasattr(value, "detach") and callable(value.detach):
        try:
            for item in value.detach().cpu().reshape(-1).tolist():
                yield float(item)
            return
        except Exception:
            return
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        try:
            for item in value.reshape(-1).tolist():
                yield float(item)
            return
        except Exception:
            return
    if isinstance(value, dict):
        for key in sorted(value):
            yield from iter_numeric_values(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_numeric_values(item)
    elif isinstance(value, (int, float, bool)):
        yield float(value)


def sample_norm(payload: dict[str, Any]) -> float:
    total = 0.0
    for field in ROLLOUT_ARRAY_FIELDS:
        for item in iter_numeric_values(payload.get(field)):
            total += item * item
    return total ** 0.5


def digest_any(digest: Any, value: Any) -> None:
    if hasattr(value, "detach") and callable(value.detach):
        tensor = value.detach().cpu().contiguous()
        try:
            array = tensor.numpy()
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
            return
        except Exception:
            digest.update(repr(tensor.tolist()).encode())
            return
    if isinstance(value, dict):
        digest.update(b"{")
        for key in sorted(value):
            digest.update(repr(key).encode())
            digest_any(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            digest_any(digest, item)
        digest.update(b"]")
    else:
        digest.update(repr(value).encode())


def agent_state_fingerprint(agent: Any) -> str:
    digest = hashlib.sha256()
    policy = getattr(agent, "policy", None)
    if policy is not None and hasattr(policy, "state_dict"):
        digest.update(b"policy")
        digest_any(digest, policy.state_dict())
    optimizer = getattr(policy, "optimizer", None) if policy is not None else None
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        digest.update(b"optimizer")
        digest_any(digest, optimizer.state_dict())
    return digest.hexdigest()


def capture_rollout_buffer_snapshot(buffer: Any, agent: Any) -> dict[str, Any]:
    arrays = {field: clone_numeric_payload(getattr(buffer, field)) for field in ROLLOUT_ARRAY_FIELDS if hasattr(buffer, field)}
    meta = {field: clone_numeric_payload(getattr(buffer, field)) for field in ROLLOUT_META_FIELDS if hasattr(buffer, field)}
    snapshot = {
        "arrays": arrays,
        "meta": meta,
        "sample_norm": sample_norm(arrays),
        "agent_state_fingerprint": agent_state_fingerprint(agent),
    }
    return snapshot


def restore_rollout_buffer_snapshot(buffer: Any, snapshot: dict[str, Any]) -> None:
    for field, value in snapshot["arrays"].items():
        setattr(buffer, field, clone_numeric_payload(value))
    for field, value in snapshot["meta"].items():
        setattr(buffer, field, clone_numeric_payload(value))


def rollout_buffer_norm(buffer: Any, fields: tuple[str, ...] = ROLLOUT_ARRAY_FIELDS) -> float:
    arrays = {field: getattr(buffer, field) for field in fields if hasattr(buffer, field)}
    return sample_norm(arrays)


def maybe_call_callback(callback: Any, name: str, *args: Any) -> None:
    if callback is None:
        return
    method = getattr(callback, name, None)
    if callable(method):
        try:
            method(*args)
        except TypeError:
            method()


def setup_collect_callback(agent: Any, callback: Any, total_timesteps: int):
    setup_learn = getattr(agent, "_setup_learn", None)
    if not callable(setup_learn):
        return total_timesteps, callback
    return setup_learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=False,
        tb_log_name="diagnostic",
        progress_bar=False,
    )


def collect_rollout_snapshot_from_agent(
    agent: Any,
    *,
    callback: Any,
    total_timesteps: int,
    n_rollout_steps: int,
    sync: Callable[[], None],
    label: str,
    profiler: Any = None,
) -> dict[str, Any]:
    total_timesteps, prepared_callback = setup_collect_callback(agent, callback, total_timesteps)
    maybe_call_callback(prepared_callback, "on_training_start", locals(), globals())
    sync()
    if profiler is not None:
        profiler.__enter__()
    start = time.perf_counter()
    try:
        ok = agent.collect_rollouts(agent.env, prepared_callback, agent.rollout_buffer, n_rollout_steps=n_rollout_steps)
    finally:
        sync()
        seconds = time.perf_counter() - start
        if profiler is not None:
            profiler.__exit__(None, None, None)
    if not ok:
        raise RuntimeError(f"{label} collect_rollouts stopped before a complete rollout")
    current_timesteps = getattr(agent, "num_timesteps", total_timesteps)
    if hasattr(agent, "_update_current_progress_remaining"):
        agent._update_current_progress_remaining(current_timesteps, total_timesteps)
    snapshot = capture_rollout_buffer_snapshot(agent.rollout_buffer, agent)
    snapshot["progress_num_timesteps"] = int(current_timesteps)
    snapshot["progress_total_timesteps"] = int(total_timesteps)
    return {
        "label": label,
        "steps": int(n_rollout_steps * getattr(agent.rollout_buffer, "n_envs", 1)),
        "vector_steps": int(n_rollout_steps),
        "seconds": seconds,
        "completed": True,
        "sample_norm": snapshot["sample_norm"],
        "snapshot": snapshot,
    }


def time_buffered_update_from_snapshot(
    agent: Any,
    snapshot: dict[str, Any],
    *,
    sync: Callable[[], None],
    label: str,
    reference_fingerprint: str | None = None,
    profiler: Any = None,
) -> dict[str, Any]:
    restore_rollout_buffer_snapshot(agent.rollout_buffer, snapshot)
    if hasattr(agent, "_update_current_progress_remaining"):
        agent._update_current_progress_remaining(
            int(snapshot.get("progress_num_timesteps", getattr(agent, "num_timesteps", 0))),
            int(snapshot.get("progress_total_timesteps", getattr(agent, "num_timesteps", 0))),
        )
    initial_fingerprint = agent_state_fingerprint(agent)
    if reference_fingerprint is not None and initial_fingerprint != reference_fingerprint:
        raise RuntimeError(f"{label} initial weights/optimizer fingerprint differed from rollout reference")
    restored_sample = rollout_buffer_norm(agent.rollout_buffer)
    if abs(float(restored_sample) - float(snapshot["sample_norm"])) > 1e-9:
        raise RuntimeError(f"{label} rollout sample norm changed while restoring snapshot")
    sync()
    if profiler is not None:
        profiler.__enter__()
    start = time.perf_counter()
    final_fingerprint = None
    try:
        agent.train()
    finally:
        sync()
        seconds = time.perf_counter() - start
        if profiler is not None:
            profiler.__exit__(None, None, None)
        try:
            final_fingerprint = agent_state_fingerprint(agent)
        finally:
            clear_cache = getattr(getattr(agent, "rollout_buffer", None), "clear_cache", None)
            if clear_cache is not None:
                clear_cache()
    return {
        "label": label,
        "updates": 1,
        "seconds": seconds,
        "sample_norm": float(snapshot["sample_norm"]),
        "restored_sample_norm": restored_sample,
        "initial_state_fingerprint": initial_fingerprint,
        "final_state_fingerprint": final_fingerprint,
        "state_changed_after_train": final_fingerprint != initial_fingerprint,
    }


def prepare_update_agent_for_train(train_job: Any, agent: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_logger = getattr(train_job, "_set_model_logger", None)
    if not callable(set_logger):
        raise RuntimeError("train_job does not expose _set_model_logger required for direct PPO.train()")
    set_logger(agent, output_dir)


def assert_and_record_pair_parity(control: dict[str, Any], cache: dict[str, Any], *, pair_index: int) -> None:
    for arm in (control, cache):
        initial = arm.get("initial_state_fingerprint")
        final = arm.get("final_state_fingerprint")
        if not initial or not final or initial == final or arm.get("state_changed_after_train") is not True:
            raise RuntimeError(f"pair {pair_index} arm {arm.get('label')} lacks evidence of an actual update")
    if control["initial_state_fingerprint"] != cache["initial_state_fingerprint"]:
        raise RuntimeError(f"control/cache initial fingerprint mismatch for pair {pair_index}")
    control_final = control.get("final_state_fingerprint")
    cache_final = cache.get("final_state_fingerprint")
    if control_final != cache_final:
        raise RuntimeError(
            f"control/cache final fingerprint mismatch for pair {pair_index}: "
            f"{control_final} != {cache_final}"
        )
    control["pair_final_fingerprint_parity"] = True
    cache["pair_final_fingerprint_parity"] = True
    control["paired_with"] = cache.get("label")
    cache["paired_with"] = control.get("label")
    control["pair_index"] = pair_index
    cache["pair_index"] = pair_index


def collect_real_rollout_snapshot(
    context: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_all(args.seed, args.device)
    env = build_frozen_env(context, args)
    sync = _sync_for(args.device)
    try:
        agent = load_agent(context, env, args, output_dir / "rollout_snapshot")
        callback = context["train_job"].JobCallback(output_dir / "rollout_snapshot", args.update_max_seconds)
        collected = collect_rollout_snapshot_from_agent(
            agent,
            callback=callback,
            total_timesteps=int(args.n_envs * args.n_steps),
            n_rollout_steps=int(args.n_steps),
            sync=sync,
            label="real_rollout_collect_snapshot",
        )
        phase = {key: value for key, value in collected.items() if key != "snapshot"}
        phase.update({"device": args.device, "n_envs": int(args.n_envs), "n_steps": int(args.n_steps)})
        return phase, collected["snapshot"]
    finally:
        env.close()


def run_buffered_update_on_env(
    context: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    snapshot: dict[str, Any],
    *,
    label: str,
    cache_class: type | None,
    env: Any,
) -> dict[str, Any]:
    seed_all(args.seed, args.device)
    arm_output = output_dir / label
    agent = load_agent(context, env, args, arm_output, cache_class=cache_class)
    prepare_update_agent_for_train(context["train_job"], agent, arm_output)
    result = time_buffered_update_from_snapshot(
        agent,
        snapshot,
        sync=_sync_for(args.device),
        label=label,
        reference_fingerprint=snapshot["agent_state_fingerprint"],
    )
    result.update(
        {
            "steps": int(args.n_envs * args.n_steps),
            "device": args.device,
            "cache_enabled": cache_class is not None,
            "num_timesteps": int(getattr(agent, "num_timesteps", 0)),
        }
    )
    return result


def run_buffered_update_arm(
    context: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    snapshot: dict[str, Any],
    *,
    label: str,
    cache_class: type | None,
) -> dict[str, Any]:
    env = build_frozen_env(context, args)
    try:
        return run_buffered_update_on_env(
            context,
            args,
            output_dir,
            snapshot,
            label=label,
            cache_class=cache_class,
            env=env,
        )
    finally:
        env.close()


def run_trace_collect_update_diagnostic(context: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    if not args.torch_trace:
        return []
    seed_all(args.seed, args.device)
    env = build_frozen_env(context, args)
    sync = _sync_for(args.device)
    label = "trace_collect_update_control"
    trace_path = output_dir / f"{label}_torch_trace.json"
    profiler = maybe_trace_profiler(args)
    try:
        arm_output = output_dir / label
        agent = load_agent(context, env, args, arm_output)
        prepare_update_agent_for_train(context["train_job"], agent, arm_output)
        callback = context["train_job"].JobCallback(output_dir / label, args.update_max_seconds)
        sync()
        if profiler is not None:
            profiler.__enter__()
        start = time.perf_counter()
        try:
            total_timesteps, prepared_callback = setup_collect_callback(
                agent,
                callback,
                int(args.n_envs * args.n_steps),
            )
            maybe_call_callback(prepared_callback, "on_training_start", locals(), globals())
            ok = agent.collect_rollouts(agent.env, prepared_callback, agent.rollout_buffer, n_rollout_steps=int(args.n_steps))
            if not ok:
                raise RuntimeError(f"{label} collect_rollouts stopped before a complete rollout")
            if hasattr(agent, "_update_current_progress_remaining"):
                agent._update_current_progress_remaining(getattr(agent, "num_timesteps", total_timesteps), total_timesteps)
            agent.train()
        finally:
            sync()
            seconds = time.perf_counter() - start
            trace_summary = None
            if profiler is not None:
                profiler.__exit__(None, None, None)
                profiler.export_chrome_trace(str(trace_path))
                trace_summary = summarize_torch_trace_events(json.loads(trace_path.read_text()))
        return [
            {
                "label": label,
                "steps": int(args.n_envs * args.n_steps),
                "updates": 1,
                "seconds": seconds,
                "device": args.device,
                "trace_path": str(trace_path) if trace_summary is not None else None,
                "trace_summary": trace_summary,
                "note": "Traceful collect+update diagnostic; excluded from fixed-rollout update ceiling.",
            }
        ]
    finally:
        env.close()


def run_update_pairs(context: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    cache_class = load_cache_class(args)
    collect_phase, snapshot = collect_real_rollout_snapshot(context, args, output_dir)
    results = [collect_phase]
    results.extend(run_trace_collect_update_diagnostic(context, args, output_dir))
    env = build_frozen_env(context, args)
    try:
        for repeat in range(args.update_repeats):
            control = run_buffered_update_on_env(
                context,
                args,
                output_dir,
                snapshot,
                label=f"real_rollout_update_control_r{repeat}",
                cache_class=None,
                env=env,
            )
            results.append(control)
            if cache_class is not None:
                cache = run_buffered_update_on_env(
                    context,
                    args,
                    output_dir,
                    snapshot,
                    label=f"real_rollout_update_cache_r{repeat}",
                    cache_class=cache_class,
                    env=env,
                )
                assert_and_record_pair_parity(control, cache, pair_index=repeat)
                results.append(cache)
    finally:
        env.close()
    return results


def provenance_for(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": sys.version,
        "executable": sys.executable,
        "pid": os.getpid(),
        "files": {
            "config": file_record(args.config),
            "checkpoint": file_record(args.checkpoint),
            "source_train": file_record(args.source / "train.py"),
            "train_job": file_record(args.train_job),
            "rollout_cache": file_record(args.rollout_cache_path) if args.rollout_cache_path else None,
            "harness": file_record(Path(__file__)),
        },
    }


def run_e1_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    configure_runtime_env()
    args.output.mkdir(parents=True, exist_ok=True)
    context = load_training_context(args, args.output)
    phases = [run_fixed_action_env(context, args)]
    phases.extend(run_inference_batches(context, args, args.output))
    phases.extend(run_update_pairs(context, args, args.output))
    payload = build_summary(phases=phases, provenance=provenance_for(args))
    payload["phase"] = str(args.phase)
    payload["n_envs"] = int(args.n_envs)
    payload["n_steps"] = int(args.n_steps)
    payload["batch_size"] = int(args.batch_size)
    payload["epochs"] = int(args.epochs)
    (args.output / "benchmark_diagnostics.json").write_text(json.dumps(bt.json_safe(payload), indent=2, sort_keys=True) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E1 diagnostic ceilings and short CPU/CUDA profiler trace for frozen navigation PPO."
    )
    parser.add_argument(
        "--legacy-single-env",
        action="store_true",
        help="Run the pre-existing single-env instrumented PPO diagnostic entry path.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=DEFAULT_TRAIN_JOB)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--phase", default="0b")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64 * 512)
    parser.add_argument("--env-steps", type=int, default=1000)
    parser.add_argument("--env-warmup-steps", type=int, default=50)
    parser.add_argument("--inference-batches", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--inference-iters", type=int, default=200)
    parser.add_argument("--inference-warmup", type=int, default=20)
    parser.add_argument("--update-repeats", type=int, default=1)
    parser.add_argument("--update-max-seconds", type=float, default=None)
    parser.add_argument("--rollout-cache-path", type=Path)
    parser.add_argument("--torch-trace", action="store_true")
    parser.add_argument("--ppo-rollouts", type=int, default=None, help="Legacy diagnostic rollout count.")
    parser.add_argument("--profile", action="store_true", help="Legacy diagnostic cProfile output.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "diagnostic_results")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.n_steps <= 1:
        parser.error("--n-steps must exceed one")
    if args.batch_size <= 1:
        parser.error("--batch-size must exceed one")
    if args.n_envs <= 0:
        parser.error("--n-envs must be positive")
    if args.env_steps <= 0 or args.env_warmup_steps < 0:
        parser.error("--env-steps must be positive and --env-warmup-steps nonnegative")
    if args.inference_iters <= 0 or args.inference_warmup < 0:
        parser.error("--inference-iters must be positive and --inference-warmup nonnegative")
    if any(batch <= 0 for batch in args.inference_batches):
        parser.error("--inference-batches entries must be positive")
    if args.ppo_rollouts is not None and args.ppo_rollouts <= 0:
        parser.error("--ppo-rollouts must be positive")
    if args.update_repeats <= 0:
        parser.error("--update-repeats must be positive")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    args.source = args.source.resolve()
    args.train_job = args.train_job.resolve()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.rollout_cache_path is not None:
        args.rollout_cache_path = args.rollout_cache_path.resolve()
    args.output = args.output.resolve()
    return args


def main() -> int:
    args = parse_args()
    configure_runtime_env()
    bt.require_runtime_dependencies()
    bt.configure_torch(args.device, args.torch_threads)
    if args.legacy_single_env or args.ppo_rollouts is not None or args.profile:
        if args.ppo_rollouts is None:
            args.ppo_rollouts = 2
        result = run_diagnostic(args, args.output)
        print(f"Saved instrumented diagnostic: {args.output / 'benchmark_diagnostics.json'}")
        print(
            f"instrumented_single_env_ppo: {result['steps_per_second']:.1f} env-steps/s, "
            f"{result['rollout_count']} rollouts, {result['update_count']} updates"
        )
        return 0
    result = run_e1_diagnostics(args)
    print(json.dumps({"output": str(args.output / "benchmark_diagnostics.json"), "phase_count": result["phase_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
