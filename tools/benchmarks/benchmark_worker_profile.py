#!/usr/bin/env python3
"""Worker/parent environment timing profile for the frozen PPO benchmark path."""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import source_path

import argparse
import cProfile
import contextvars
import dataclasses
import hashlib
import importlib
import json
import os
import pstats
import random
import sys
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import ai2.runtime.benchmark_campaign as benchmark_campaign

if __name__ == "__main__":
    sys.modules.setdefault("tools.benchmarks.benchmark_worker_profile", sys.modules[__name__])

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None


DEFAULT_CYCLES = 10
DEFAULT_WARMUP = 0
_GymWrapperBase = gym.Wrapper if gym is not None else object
_GROUPED_RNG_SPAN_TARGETS: list["Spans"] = []
_ACTIVE_WORKER_SPANS: contextvars.ContextVar["Spans | None"] = contextvars.ContextVar(
    "ai2_worker_profile_active_spans",
    default=None,
)


class Spans:
    def __init__(self) -> None:
        self.enabled = False
        self.stack: list[list[float]] = []
        self.rows: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])

    def wrap(self, function: Callable[..., Any], label: str) -> Callable[..., Any]:
        @wraps(function)
        def timed(*args: Any, **kwargs: Any) -> Any:
            if not self.enabled:
                return function(*args, **kwargs)
            frame = [time.perf_counter(), 0.0, time.thread_time(), 0.0]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                wall_elapsed = time.perf_counter() - frame[0]
                cpu_elapsed = time.thread_time() - frame[2]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += wall_elapsed
                    self.stack[-1][3] += cpu_elapsed
                row = self.rows[label]
                row[0] += 1
                row[1] += wall_elapsed
                row[2] += wall_elapsed - frame[1]
                row[3] += cpu_elapsed
                row[4] += cpu_elapsed - frame[3]

        return timed

    def record(self, label: str, wall_elapsed: float, cpu_elapsed: float) -> None:
        if not self.enabled:
            return
        row = self.rows[label]
        row[0] += 1
        row[1] += wall_elapsed
        row[2] += wall_elapsed
        row[3] += cpu_elapsed
        row[4] += cpu_elapsed

    def patch(self, target: Any, method: str, label: str) -> None:
        if target is None or not hasattr(target, method):
            return
        original = getattr(target, method)
        if (
            getattr(original, "_ai2_worker_profile_wrapped", False)
            and getattr(original, "_ai2_worker_profile_label", None) == label
            and getattr(original, "_ai2_worker_profile_context_var", None) is _ACTIVE_WORKER_SPANS
        ):
            return
        unwrapped = getattr(original, "_ai2_worker_profile_original", original)

        @wraps(unwrapped)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            spans = _ACTIVE_WORKER_SPANS.get() or self
            if not spans.enabled:
                return unwrapped(*args, **kwargs)
            return spans.wrap(unwrapped, label)(*args, **kwargs)

        setattr(wrapped, "_ai2_worker_profile_wrapped", True)
        setattr(wrapped, "_ai2_worker_profile_original", unwrapped)
        setattr(wrapped, "_ai2_worker_profile_label", label)
        setattr(wrapped, "_ai2_worker_profile_context_var", _ACTIVE_WORKER_SPANS)
        setattr(target, method, wrapped)

    def report(self) -> dict[str, dict[str, float | int]]:
        return {
            label: {
                "count": int(row[0]),
                "inclusive_wall_seconds": float(row[1]),
                "exclusive_wall_seconds": float(row[2]),
                "inclusive_thread_cpu_seconds": float(row[3]),
                "exclusive_thread_cpu_seconds": float(row[4]),
            }
            for label, row in sorted(self.rows.items())
        }


class WorkerTimingWrapper(_GymWrapperBase):
    def __init__(self, env: Any, spans: Spans, worker_index: int) -> None:
        if gym is not None and isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env
        self.spans = spans
        self.worker_index = int(worker_index)
        self.step_events: list[list[float]] = []
        patch_worker_components(env, spans)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(type(self), name) or name in self.__dict__:
            return getattr(self, name)
        getter = getattr(self.env, "get_wrapper_attr", None)
        if getter is not None:
            return getter(name)
        return getattr(self.env, name)

    def start_worker_measurement(self) -> int:
        self.spans.rows.clear()
        self.step_events.clear()
        self.spans.enabled = True
        return self.worker_index

    def finish_worker_measurement(self) -> dict[str, Any]:
        self.spans.enabled = False
        return {
            "worker_index": self.worker_index,
            "step_events": self.step_events,
            "timing_note": (
                "Inclusive rows nest; exclusive rows omit nested spans. "
                "Wall and thread CPU are reported separately; worker CPU totals are not additive "
                "to parent wall time because workers overlap."
            ),
            "components": self.spans.report(),
        }

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        timed = self.spans.wrap(self.env.reset, "worker.env_reset")
        token = _ACTIVE_WORKER_SPANS.set(self.spans)
        try:
            return timed(*args, **kwargs)
        finally:
            _ACTIVE_WORKER_SPANS.reset(token)

    def step(self, action: Any) -> Any:
        timed = self.spans.wrap(self.env.step, "worker.env_step")
        start = time.perf_counter()
        token = _ACTIVE_WORKER_SPANS.set(self.spans)
        try:
            return timed(action)
        finally:
            _ACTIVE_WORKER_SPANS.reset(token)
            self.step_events.append([start, time.perf_counter()])

    def close(self) -> Any:
        return self.env.close()


class ParentVecTimer:
    def __init__(self) -> None:
        self.rows: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])

    def wrap(self, function: Callable[..., Any], label: str) -> Callable[..., Any]:
        @wraps(function)
        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                row = self.rows[label]
                row[0] += 1
                row[1] += time.perf_counter() - start

        return timed

    def report(self) -> dict[str, Any]:
        components = {
            label: {
                "count": int(row[0]),
                "seconds": float(row[1]),
                "mean_seconds": float(row[1]) / int(row[0]) if row[0] else 0.0,
            }
            for label, row in sorted(self.rows.items())
        }
        return {
            **components,
            "note": (
                "Parent step_async/step_wait timing includes Python dispatch, pipe send/recv, "
                "blocking for workers, and wrapper stacking. It does not decompose worker CPU."
            ),
        }


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    resolved = Path(path)
    if not resolved.is_file():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


def patch_worker_components(env: Any, spans: Spans) -> None:
    game = safe_import("environment.environment")
    agents = safe_import("environment.agent")
    train = safe_import("train")
    pymunk = safe_import("pymunk")
    monitor_module = safe_import("stable_baselines3.common.monitor")

    patch_specs = []
    if game is not None:
        patch_specs.extend(
            [
                (getattr(game, "WarehouseBrawl", None), "step", "game.step"),
                (getattr(game, "WarehouseBrawl", None), "reset", "game.reset"),
                (getattr(game, "WarehouseBrawl", None), "observe", "game.observation"),
                (getattr(game, "WarehouseBrawl", None), "_observe_all", "game.observe_all"),
                (getattr(game, "Player", None), "get_obs", "game.player_observation"),
                (getattr(game, "Player", None), "pre_process", "game.player_pre_process"),
                (getattr(game, "Player", None), "physics_process", "game.player_physics_state"),
                (getattr(game, "Player", None), "process", "game.player_input"),
                (getattr(game, "Player", None), "is_on_floor", "game.floor_check"),
                (getattr(game, "Stage", None), "physics_process", "game.stage_platform_physics"),
                (getattr(game, "Camera", None), "process", "game.camera_process"),
            ]
        )
    if pymunk is not None:
        patch_specs.append((getattr(pymunk, "Space", None), "step", "game.native_physics_with_callbacks"))
    if agents is not None:
        patch_specs.extend(
            [
                (getattr(agents, "ConstantAgent", None), "predict", "wrapper.constant_opponent"),
                (getattr(agents, "RewardManager", None), "process", "wrapper.base_reward"),
                (getattr(agents, "SelfPlayWarehouseBrawl", None), "step", "wrapper.selfplay_step"),
                (getattr(agents, "SelfPlayWarehouseBrawl", None), "reset", "wrapper.selfplay_reset"),
            ]
        )
    if train is not None:
        patch_specs.extend(
            [
                (getattr(train, "FrozenOpponentWrapper", None), "step", "wrapper.navigation_reward_and_end_checks"),
                (getattr(train, "FrozenOpponentWrapper", None), "reset", "wrapper.navigation_reset"),
                (getattr(train, "Float32Wrapper", None), "observation", "wrapper.float32_observation"),
                (getattr(train, "OpponentHistoryWrapper", None), "step", "wrapper.history_step"),
                (getattr(train, "OpponentHistoryWrapper", None), "_augment_obs", "wrapper.history_augmentation"),
            ]
        )
    if monitor_module is not None:
        patch_specs.extend(
            [
                (getattr(monitor_module, "Monitor", None), "step", "wrapper.monitor_step"),
                (getattr(monitor_module, "Monitor", None), "reset", "wrapper.monitor_reset"),
            ]
        )

    for target, method, label in patch_specs:
        spans.patch(target, method, label)

    raw = getattr(getattr(env, "unwrapped", None), "raw_env", None)
    if raw is None:
        raw = getattr(env, "raw_env", None)
    if raw is not None:
        spans.patch(raw.__class__, "step", "game.step")
        spans.patch(raw.__class__, "reset", "game.reset")
        spans.patch(raw.__class__, "observe", "game.observation")
        controller = getattr(raw, "weapon_controller", None)
        if controller is not None:
            spans.patch(controller.__class__, "try_pick_up_all", "game.weapon_pickups")
            spans.patch(controller.__class__, "update", "game.weapon_update")


def _profile_worker_env_impl(env: Any, worker_index: int) -> WorkerTimingWrapper:
    install_grouped_rng_state_profile()
    spans = Spans()
    _GROUPED_RNG_SPAN_TARGETS.append(spans)
    return WorkerTimingWrapper(env, spans, worker_index)


def profile_worker_env(env: Any, worker_index: int) -> WorkerTimingWrapper:
    module_name = globals().get("__name__")
    module = sys.modules[module_name] if module_name == "tools.benchmarks.benchmark_worker_profile" else importlib.import_module(
        'tools.benchmarks.benchmark_worker_profile'
    )
    return getattr(module, "_profile_worker_env_impl")(env, worker_index)


class _ProfiledEnvFactory:
    def __init__(self, factory: Callable[[], Any], worker_index: int) -> None:
        self.factory = factory
        self.worker_index = int(worker_index)

    def __call__(self) -> Any:
        profile_module = importlib.import_module('tools.benchmarks.benchmark_worker_profile')
        return profile_module._profile_worker_env_impl(self.factory(), self.worker_index)

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[Callable[[], Any], int]]:
        rebuild = importlib.import_module('tools.benchmarks.benchmark_worker_profile')._rebuild_profiled_env_factory
        return (rebuild, (self.factory, self.worker_index))


_ProfiledEnvFactory.__module__ = "tools.benchmarks.benchmark_worker_profile"


def _rebuild_profiled_env_factory(factory: Callable[[], Any], worker_index: int) -> "_ProfiledEnvFactory":
    return _ProfiledEnvFactory(factory, worker_index)


_rebuild_profiled_env_factory.__module__ = "tools.benchmarks.benchmark_worker_profile"


def _grouped_rng_target(index: int) -> Spans | None:
    if 0 <= int(index) < len(_GROUPED_RNG_SPAN_TARGETS):
        return _GROUPED_RNG_SPAN_TARGETS[int(index)]
    return None


def _record_grouped_rng_span(spans: Spans | None, label: str, start_wall: float, start_cpu: float) -> None:
    if spans is None:
        return
    spans.record(label, time.perf_counter() - start_wall, time.thread_time() - start_cpu)


def install_grouped_rng_state_profile() -> None:
    """Patch imported grouped workers to time RNG state swaps inside each lane.

    The patch is installed from profiled env factory construction, which runs in
    spawned grouped workers before measured reset/step calls. It records only the
    explicit save/set/get/restore RNG operations, not the nested env step body.
    """

    for module in list(sys.modules.values()):
        cls = getattr(module, "_GlobalRngStates", None)
        if not isinstance(cls, type) or getattr(cls, "_ai2_worker_profile_rng_wrapped", False):
            continue

        def profiled_run(self: Any, index: int, func: Callable[[], Any], *, seed: int | None = None) -> Any:
            try:
                import numpy as np
            except ModuleNotFoundError:
                return func()

            spans = _grouped_rng_target(int(index))
            start_wall = time.perf_counter()
            start_cpu = time.thread_time()
            outer_random_state = random.getstate()
            outer_np_state = np.random.get_state()
            _record_grouped_rng_span(spans, "grouped_rng.save_outer_state", start_wall, start_cpu)
            try:
                start_wall = time.perf_counter()
                start_cpu = time.thread_time()
                random.setstate(self.states[index][0])
                np.random.set_state(self.states[index][1])
                _record_grouped_rng_span(spans, "grouped_rng.restore_lane_state", start_wall, start_cpu)

                result = func()

                start_wall = time.perf_counter()
                start_cpu = time.thread_time()
                self.states[index] = (random.getstate(), np.random.get_state())
                _record_grouped_rng_span(spans, "grouped_rng.capture_lane_state", start_wall, start_cpu)
                return result
            finally:
                start_wall = time.perf_counter()
                start_cpu = time.thread_time()
                random.setstate(outer_random_state)
                np.random.set_state(outer_np_state)
                _record_grouped_rng_span(spans, "grouped_rng.restore_outer_state", start_wall, start_cpu)

        cls.run = profiled_run
        setattr(cls, "_ai2_worker_profile_rng_wrapped", True)


def _wrap_factories(fns: list[Callable[[], Any]]) -> list[Callable[[], Any]]:
    profile_module = importlib.import_module('tools.benchmarks.benchmark_worker_profile')
    factory_cls = getattr(profile_module, "_ProfiledEnvFactory")
    return [factory_cls(fn, index) for index, fn in enumerate(fns)]


def patch_parent_vec_timing(env: Any, timer: ParentVecTimer) -> Any:
    if hasattr(env, "step_async"):
        env.step_async = timer.wrap(env.step_async, "parent.step_async_send")
    if hasattr(env, "step_wait"):
        env.step_wait = timer.wrap(env.step_wait, "parent.step_wait_recv_stack")
    return env


def attach_close_report(env: Any, output: Path, parent_timer: ParentVecTimer) -> Any:
    original_close = env.close

    def close_with_report(*args: Any, **kwargs: Any) -> Any:
        close_result = None
        try:
            worker_reports = finish_worker_measurement_or_raise(env)
            write_json(
                output / "worker_profile.json",
                {
                    "label": "worker_env_parent_profile",
                    "created_unix": time.time(),
                    "parent": parent_timer.report(),
                    "worker_reports": worker_reports,
                    "notes": [
                        "Worker CPU spans are inclusive/exclusive diagnostics and should not be summed as wall time.",
                        "Parent wait includes worker blocking, IPC receive, and vec/frame-stack assembly.",
                    ],
                },
            )
        finally:
            close_result = original_close(*args, **kwargs)
        return close_result

    env.close = close_with_report
    return env


def start_worker_measurement_or_raise(env: Any) -> list[Any]:
    if not hasattr(env, "env_method"):
        raise RuntimeError("profiled vec env has no env_method for start_worker_measurement")
    try:
        started = env.env_method("start_worker_measurement")
    except Exception as exc:
        raise RuntimeError(f"start_worker_measurement failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(started, list) or not started:
        raise RuntimeError(f"start_worker_measurement returned no worker rows: {started!r}")
    return started


def finish_worker_measurement_or_raise(env: Any) -> list[Any]:
    if not hasattr(env, "env_method"):
        raise RuntimeError("profiled vec env has no env_method for finish_worker_measurement")
    try:
        reports = env.env_method("finish_worker_measurement")
    except Exception as exc:
        raise RuntimeError(f"finish_worker_measurement failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(reports, list) or not reports:
        raise RuntimeError(f"finish_worker_measurement returned no worker rows: {reports!r}")
    return reports


def install_build_env_profile(train_job: Any, *, output: Path | None = None) -> None:
    original_build_env = train_job.build_env

    def build_env_profile(*args: Any, **kwargs: Any) -> Any:
        current_dummy = getattr(train_job, "DummyVecEnv")
        current_subproc = getattr(train_job, "SubprocVecEnv")

        def dummy_vec_env(fns: list[Callable[[], Any]], *ctor_args: Any, **ctor_kwargs: Any) -> Any:
            return current_dummy(_wrap_factories(list(fns)), *ctor_args, **ctor_kwargs)

        def subproc_vec_env(fns: list[Callable[[], Any]], *ctor_args: Any, **ctor_kwargs: Any) -> Any:
            return current_subproc(_wrap_factories(list(fns)), *ctor_args, **ctor_kwargs)

        train_job.DummyVecEnv = dummy_vec_env
        train_job.SubprocVecEnv = subproc_vec_env
        try:
            env = original_build_env(*args, **kwargs)
        finally:
            train_job.DummyVecEnv = current_dummy
            train_job.SubprocVecEnv = current_subproc
        parent_timer = ParentVecTimer()
        patch_parent_vec_timing(env, parent_timer)
        start_worker_measurement_or_raise(env)
        if output is not None:
            attach_close_report(env, output, parent_timer)
        return env

    train_job.build_env = build_env_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", default="0b")
    parser.add_argument("--maxseconds", type=float, default=None)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--grouped-env-path", type=Path, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.n_envs <= 0 or args.n_steps <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("n-envs, n-steps, batch-size, and epochs must be positive")
    if args.cycles <= 0 or args.warmup < 0:
        raise ValueError("cycles must be positive and warmup must be nonnegative")
    if args.group_size <= 0:
        raise ValueError("group-size must be positive")
    for name in ("config", "checkpoint", "source", "train_job", "output"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if args.grouped_env_path is None:
        args.grouped_env_path = source_path(args.source, "grouped_vec_env.py")
    args.grouped_env_path = Path(args.grouped_env_path).resolve()
    return args


def build_campaign_args(args: argparse.Namespace) -> argparse.Namespace:
    campaign_args = benchmark_campaign.build_parser().parse_args(["--output", str(args.output / "campaign")])
    overrides = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "source": args.source,
        "train_job": args.train_job,
        "output": args.output / "campaign",
        "n_envs": int(args.n_envs),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "cycles": int(args.cycles),
        "warmup": int(args.warmup),
        "device": str(args.device),
        "maxseconds": args.maxseconds,
        "phase": str(args.phase),
        "history": False,
        "opponents": None,
        "mode": "baseline",
        "profile_timing": "natural",
        "resident_buffer": False,
        "rollout_cache_path": source_path(args.source, "rollout_cache.py"),
        "group_size": int(args.group_size),
        "grouped_env_path": args.grouped_env_path,
    }
    for name, value in overrides.items():
        setattr(campaign_args, name, value)
    return campaign_args


def _write_cprofile_text(profile_path: Path, text_path: Path) -> None:
    with text_path.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(str(profile_path), stream=handle)
        stats.strip_dirs().sort_stats("cumtime").print_stats(80)


def run_profile(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    original_import = benchmark_campaign.import_train_job_module

    def import_with_worker_profile(path: Path) -> Any:
        train_job = original_import(path)
        install_build_env_profile(train_job, output=args.output)
        return train_job

    cprofile_path = args.output / "parent_cprofile.prof"
    cprofile_text = args.output / "parent_cprofile_top.txt"
    campaign_args = build_campaign_args(args)
    profiler = cProfile.Profile()
    code = 1
    started = time.time()
    try:
        benchmark_campaign.import_train_job_module = import_with_worker_profile
        profiler.enable()
        code = benchmark_campaign.run_campaign(campaign_args)
    finally:
        profiler.disable()
        benchmark_campaign.import_train_job_module = original_import
        profiler.dump_stats(str(cprofile_path))
        _write_cprofile_text(cprofile_path, cprofile_text)
    write_json(
        args.output / "worker_profile_run.json",
        {
            "label": "worker_profile_run",
            "status": "completed" if code == 0 else "failed",
            "return_code": code,
            "started_unix": started,
            "ended_unix": time.time(),
            "config": args.config,
            "checkpoint": args.checkpoint,
            "source": args.source,
            "train_job": args.train_job,
            "source_hashes": {
                "benchmark_worker_profile": sha256_file(Path(__file__).resolve()),
                "benchmark_campaign": sha256_file(Path(benchmark_campaign.__file__).resolve()),
                "train_job": sha256_file(args.train_job),
                "config": sha256_file(args.config),
                "checkpoint": sha256_file(args.checkpoint),
                "grouped_env": sha256_file(args.grouped_env_path) if args.group_size > 1 else None,
            },
            "grouping": {
                "group_size": int(args.group_size),
                "grouped_env_path": args.grouped_env_path,
            },
            "campaign_output": campaign_args.output,
            "worker_profile": args.output / "worker_profile.json",
            "parent_cprofile": cprofile_path,
            "parent_cprofile_text": cprofile_text,
            "notes": [
                "This diagnostic wraps exact frozen train_job.build_env factories and benchmark_campaign run semantics.",
                "Worker CPU spans and parent wall spans are related diagnostics, not additive totals.",
            ],
        },
    )
    return code


def main(argv: list[str] | None = None) -> int:
    args = validate_args(build_parser().parse_args(argv))
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    return run_profile(args)


if __name__ == "__main__":
    raise SystemExit(main())
