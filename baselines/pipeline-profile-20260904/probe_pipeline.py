#!/usr/bin/env python3
"""Bounded PPO pipeline profiler for the AI_2 navigation stack.

This file is intentionally isolated under baselines/. It does not modify SB3,
train.py, or benchmark_training.py; instead it builds a runtime PPO subclass
whose collect_rollouts method is AST-cloned from the installed SB3 version with
small timing context managers inserted around existing statements.
"""

from __future__ import annotations

import argparse
import ast
import cProfile
import importlib
import inspect
import json
import pstats
import sys
import tempfile
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FunctionType
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPO_ROOT / "baselines" / "pipeline-profile-20260904" / "artifacts"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROFILE_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILE_DIR))

import benchmark_training as bt


@dataclass(frozen=True)
class RolloutShape:
    n_envs: int
    n_steps: int
    requested_batch_size: int
    batch_size: int
    transitions_per_rollout: int


@dataclass
class StageStats:
    count: int = 0
    inclusive_seconds: float = 0.0
    exclusive_seconds: float = 0.0

    def to_json(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "inclusive_seconds": self.inclusive_seconds,
            "exclusive_seconds": self.exclusive_seconds,
            "mean_inclusive_seconds": self.inclusive_seconds / self.count if self.count else 0.0,
            "mean_exclusive_seconds": self.exclusive_seconds / self.count if self.count else 0.0,
        }


@dataclass
class RolloutRecord:
    index: int
    seconds: float
    transitions: int
    vector_cycles: int

    def to_json(self) -> dict[str, float | int]:
        return {
            "index": self.index,
            "seconds": self.seconds,
            "transitions": self.transitions,
            "vector_cycles": self.vector_cycles,
            "transitions_per_second": self.transitions / self.seconds if self.seconds else 0.0,
            "vector_cycles_per_second": self.vector_cycles / self.seconds if self.seconds else 0.0,
        }


class PipelineTimer:
    def __init__(self, *, enabled: bool = False, sync: Callable[[], None] | None = None) -> None:
        self.enabled = enabled
        self.sync = sync or (lambda: None)
        self.stages: dict[str, StageStats] = {}
        self.env_step_events: list[list[float]] = []
        self.stack: list[dict[str, float | str]] = []
        self.rollouts: list[RolloutRecord] = []
        self._rollout_start: float | None = None
        self._rollout_cycle_start = 0
        self.vector_cycle_count = 0

    @contextmanager
    def measure(self):
        previous = self.enabled
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = previous

    def begin_rollout(self) -> None:
        if not self.enabled:
            return
        self._rollout_start = time.perf_counter()
        self._rollout_cycle_start = self.vector_cycle_count

    def finish_rollout(self, transitions: int | None = None) -> None:
        if not self.enabled or self._rollout_start is None:
            return
        now = time.perf_counter()
        cycles = self.vector_cycle_count - self._rollout_cycle_start
        self.rollouts.append(
            RolloutRecord(
                index=len(self.rollouts),
                seconds=now - self._rollout_start,
                transitions=int(transitions if transitions is not None else cycles),
                vector_cycles=cycles,
            )
        )
        self._rollout_start = None

    def record_vector_cycle(self) -> None:
        if self.enabled:
            self.vector_cycle_count += 1

    @contextmanager
    def stage(self, label: str):
        if not self.enabled:
            yield
            return
        frame: dict[str, float | str] = {"label": label, "start": time.perf_counter(), "child": 0.0}
        self.stack.append(frame)
        try:
            yield
        finally:
            end = time.perf_counter()
            popped = self.stack.pop()
            if label == "env_step":
                self.env_step_events.append([float(popped["start"]), end])
            inclusive = end - float(popped["start"])
            exclusive = max(0.0, inclusive - float(popped["child"]))
            if self.stack:
                self.stack[-1]["child"] = float(self.stack[-1]["child"]) + inclusive
            stats = self.stages.setdefault(label, StageStats())
            stats.count += 1
            stats.inclusive_seconds += inclusive
            stats.exclusive_seconds += exclusive

    def report(
        self,
        *,
        envelope_seconds: float,
        transitions: int,
        n_envs: int,
        rollouts: int,
        update_seconds: float,
    ) -> dict[str, Any]:
        stage_total = sum(stage.exclusive_seconds for stage in self.stages.values())
        residual = max(0.0, envelope_seconds - stage_total - update_seconds)
        return {
            "envelope_seconds": envelope_seconds,
            "transitions": transitions,
            "n_envs": n_envs,
            "rollouts": rollouts,
            "vector_cycle_count": self.vector_cycle_count,
            "vector_cycles_per_rollout_expected": transitions // max(n_envs * rollouts, 1),
            "transitions_per_second": transitions / envelope_seconds if envelope_seconds else 0.0,
            "vector_cycles_per_second": self.vector_cycle_count / envelope_seconds
            if envelope_seconds
            else 0.0,
            "stages": {label: stats.to_json() for label, stats in sorted(self.stages.items())},
            "rollout_records": [record.to_json() for record in self.rollouts],
            "env_step_events": self.env_step_events,
            "stage_exclusive_seconds_total": stage_total,
            "ppo_train_seconds_total": update_seconds,
            "residual_seconds": residual,
            "notes": [
                "Stage timings are exclusive within instrumented collect_rollouts nesting.",
                "Residual is measured envelope minus instrumented rollout stages and PPO train; it contains Python loop overhead, logger/progress work, and any uninstrumented SB3 code.",
                "env_step is parent critical-path wall time. Worker process CPU timings, when present, are reported separately and must not be summed into parent wall time.",
                "No per-stage CUDA synchronization is inserted. GPU stages are natural enqueue/copy-wait timings; action copy and buffer transfer stages may include pending GPU work.",
            ],
        }


def rollout_shape(
    *, n_envs: int, rollout_transitions: int = 8192, requested_batch_size: int = 8192
) -> RolloutShape:
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if rollout_transitions <= 0:
        raise ValueError("rollout_transitions must be positive")
    if requested_batch_size <= 1:
        raise ValueError("requested_batch_size must exceed one")
    if rollout_transitions % n_envs != 0:
        raise ValueError(
            f"rollout_transitions ({rollout_transitions}) must be divisible by n_envs ({n_envs})"
        )
    n_steps = max(1, rollout_transitions // n_envs)
    transitions = n_steps * n_envs
    return RolloutShape(
        n_envs=n_envs,
        n_steps=n_steps,
        requested_batch_size=requested_batch_size,
        batch_size=min(requested_batch_size, transitions),
        transitions_per_rollout=transitions,
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    if isinstance(node, ast.Subscript):
        return _call_name(node.value)
    return ""


def _contains_text(node: ast.AST, text: str) -> bool:
    return text in ast.unparse(node)


def _stage_with(label: str, node: ast.stmt) -> ast.With:
    wrapped = ast.With(
        items=[
            ast.withitem(
                context_expr=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="pipeline_timer", ctx=ast.Load()),
                        attr="stage",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant(label)],
                    keywords=[],
                ),
                optional_vars=None,
            )
        ],
        body=[node],
        type_comment=None,
    )
    ast.copy_location(wrapped, node)
    ast.fix_missing_locations(wrapped)
    return wrapped


def _record_cycle_stmt() -> ast.stmt:
    return ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="pipeline_timer", ctx=ast.Load()),
                attr="record_vector_cycle",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )
    )


def _rollout_begin_stmt() -> ast.stmt:
    return ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="pipeline_timer", ctx=ast.Load()),
                attr="begin_rollout",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )
    )


def _rollout_finish_stmt() -> ast.stmt:
    return ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="pipeline_timer", ctx=ast.Load()),
                attr="finish_rollout",
                ctx=ast.Load(),
            ),
            args=[
                ast.BinOp(
                    left=ast.Name(id="n_rollout_steps", ctx=ast.Load()),
                    op=ast.Mult(),
                    right=ast.Attribute(value=ast.Name(id="env", ctx=ast.Load()), attr="num_envs", ctx=ast.Load()),
                )
            ],
            keywords=[],
        )
    )


class CollectRolloutsInstrumenter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.labels: set[str] = set()

    def wrap(self, label: str, node: ast.stmt) -> ast.With:
        self.labels.add(label)
        return _stage_with(label, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name != "collect_rollouts":
            return node
        self.generic_visit(node)
        node.body.insert(0, _rollout_begin_stmt())
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        self.generic_visit(node)
        if _contains_text(node.test, "n_steps") and _contains_text(node.test, "n_rollout_steps"):
            node.body.insert(0, _record_cycle_stmt())
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        rhs = _call_name(node.value)
        lhs = ", ".join(ast.unparse(target) for target in node.targets)
        text = ast.unparse(node)
        if "obs_as_tensor" in rhs and "obs_tensor" in lhs:
            return self.wrap("observations_to_tensor", node)
        if ".cpu.numpy" in rhs or ".cpu().numpy" in text:
            return self.wrap("actions_gpu_to_cpu", node)
        if "np.clip" in rhs:
            return self.wrap("clip_actions", node)
        if rhs.endswith(".step") or ".step(" in text and "env.step" in text:
            return self.wrap("env_step", node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        self.generic_visit(node)
        text = ast.unparse(node)
        if "callback.on_rollout_start" in text:
            return self.wrap("callback_rollout_start", node)
        if "callback.on_rollout_end" in text:
            self.labels.add("callback_rollout_end")
            return [self.wrap("callback_rollout_end", node), _rollout_finish_stmt()]
        if "callback.update_locals" in text or "callback.on_step" in text:
            return self.wrap("callback", node)
        if "rollout_buffer.add" in text:
            return self.wrap("buffer_add", node)
        if "compute_returns_and_advantage" in text:
            return self.wrap("compute_returns_advantage", node)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        if _contains_text(node.test, "callback.on_step"):
            self.generic_visit(node)
            return self.wrap("callback", node)
        if _contains_text(node.test, "TimeLimit.truncated") or _contains_text(
            node.test, "terminal_observation"
        ):
            node.body = [self.wrap("timeout_bootstrap", stmt) for stmt in node.body]
            self.labels.add("timeout_bootstrap")
            return node
        self.generic_visit(node)
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        original = ast.unparse(node)
        self.generic_visit(node)
        if "infos" in original and "dones" in original:
            return self.wrap("info", node)
        return node

    def visit_With(self, node: ast.With) -> ast.AST:
        original = ast.unparse(node)
        self.generic_visit(node)
        if "self.policy" in original and "actions" in original and "log_probs" in original:
            return self.wrap("policy_forward", node)
        if "predict_values" in original and "terminal" in original:
            return self.wrap("timeout_bootstrap", node)
        if "predict_values" in original:
            return self.wrap("final_value", node)
        return node


def instrument_collect_rollouts_source(source: str) -> ast.Module:
    tree = ast.parse(textwrap.dedent(source))
    instrumenter = CollectRolloutsInstrumenter()
    tree = instrumenter.visit(tree)
    ast.fix_missing_locations(tree)
    required = {
        "callback_rollout_start",
        "observations_to_tensor",
        "policy_forward",
        "actions_gpu_to_cpu",
        "env_step",
        "clip_actions",
        "callback",
        "info",
        "buffer_add",
        "final_value",
        "compute_returns_advantage",
        "callback_rollout_end",
    }
    missing = sorted(required - instrumenter.labels)
    if missing:
        raise RuntimeError(
            "Could not identify expected SB3 collect_rollouts statements: "
            + ", ".join(missing)
        )
    return tree


def instrumented_ppo_class(base_cls: type, timer: PipelineTimer) -> type:
    source = inspect.getsource(base_cls.collect_rollouts)
    tree = instrument_collect_rollouts_source(source)
    namespace = dict(base_cls.collect_rollouts.__globals__)
    exec(compile(tree, filename="<instrumented_collect_rollouts>", mode="exec"), namespace)
    collect_rollouts = namespace["collect_rollouts"]
    if not isinstance(collect_rollouts, FunctionType):
        raise RuntimeError("Instrumented collect_rollouts did not compile to a function")

    class PipelineProfilePPO(base_cls):
        def __init__(self, *args, **kwargs) -> None:
            self.pipeline_timer = timer
            self.ppo_train_seconds: list[float] = []
            self.diagnostic_progress_total_timesteps: list[int] = []
            self._configured_training_budget = int(kwargs.pop("configured_training_budget"))
            super().__init__(*args, **kwargs)

        def _update_current_progress_remaining(
            self, num_timesteps: int, total_timesteps: int
        ) -> None:
            schedule_total = max(self._configured_training_budget, total_timesteps)
            self.diagnostic_progress_total_timesteps.append(schedule_total)
            super()._update_current_progress_remaining(num_timesteps, schedule_total)

        def train(self) -> None:
            timer.sync()
            start = time.perf_counter()
            try:
                return super().train()
            finally:
                timer.sync()
                self.ppo_train_seconds.append(time.perf_counter() - start)

    PipelineProfilePPO.collect_rollouts = collect_rollouts
    return PipelineProfilePPO


def _worker_timing_module(use_worker_timing: bool):
    if not use_worker_timing:
        return None
    try:
        module = importlib.import_module("worker_timing")
    except ModuleNotFoundError as exc:
        if exc.name != "worker_timing":
            raise
        return None
    if hasattr(module, "install"):
        module.install()
    return module


def build_env(
    config_path: Path, seed: int, n_envs: int, torch_threads: int, *, use_worker_timing: bool = True
):
    module = _worker_timing_module(use_worker_timing)
    if module is not None and hasattr(module, "build_env"):
        env, params, phase = module.build_env(config_path, seed, n_envs, torch_threads)
        return env, params, phase, True
    env, params, phase = bt.build_wrapped_nav_env(config_path, seed, n_envs, torch_threads)
    return env, params, phase, False


def env_method_optional(env: Any, method_name: str) -> list[Any]:
    if not hasattr(env, "env_method"):
        return []
    return env.env_method(method_name)


def _linear_or_value(configured: Any):
    from train import linear_schedule

    if isinstance(configured, list):
        return linear_schedule(configured[0], configured[1])
    return configured


def make_model(model_cls: type, env: Any, params: dict[str, Any], args: argparse.Namespace):
    ppo = params.get("ppo_settings", {})
    shape = rollout_shape(
        n_envs=args.n_envs,
        rollout_transitions=args.rollout_transitions,
        requested_batch_size=args.batch_size,
    )
    return model_cls(
        "MlpPolicy",
        env,
        learning_rate=_linear_or_value(ppo.get("learning_rate", [3e-4, 1e-6])),
        n_steps=shape.n_steps,
        batch_size=shape.batch_size,
        n_epochs=int(args.epochs),
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
        policy_kwargs={"net_arch": [512, 512, 256]},
        configured_training_budget=int(
            ppo.get(
                "time_steps",
                max(args.warmup_transitions + args.rollouts * shape.transitions_per_rollout, 1),
            )
        ),
    )


def sync_for(device: str) -> Callable[[], None]:
    def sync() -> None:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

    return sync


def cprofile_summary(profile_path: Path, limit: int = 40) -> list[dict[str, Any]]:
    stats = pstats.Stats(str(profile_path)).sort_stats("cumtime")
    rows = []
    for func, data in stats.stats.items():
        cc, nc, tt, ct, _ = data
        filename, lineno, name = func
        rows.append(
            {
                "function": f"{Path(filename).name}:{lineno}:{name}",
                "primitive_calls": cc,
                "total_calls": nc,
                "total_seconds": tt,
                "cumulative_seconds": ct,
            }
        )
    return sorted(rows, key=lambda item: item["cumulative_seconds"], reverse=True)[:limit]


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    bt.set_headless_environment()
    bt.require_runtime_dependencies()
    bt.configure_torch(args.device, args.torch_threads)
    bt.add_repo_imports()

    import torch
    from stable_baselines3 import PPO

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = rollout_shape(
        n_envs=args.n_envs,
        rollout_transitions=args.rollout_transitions,
        requested_batch_size=args.batch_size,
    )
    env, params, phase_config, worker_timing_active = build_env(
        args.config,
        args.seed,
        args.n_envs,
        args.torch_threads,
        use_worker_timing=not args.no_worker_timing,
    )
    timer = PipelineTimer(enabled=False, sync=sync_for(args.device))
    profiled_cls = instrumented_ppo_class(PPO, timer)
    profile_path = output_dir / "pipeline_profile.prof" if args.cprofile else None
    torch_profile_path = output_dir / "torch_profile.json" if args.torch_profile else None
    worker_measurement_started: list[Any] = []
    worker_rows: list[Any] = []
    try:
        model = make_model(profiled_cls, env, params, args)
        warmup_transitions = args.warmup_transitions
        model.learn(
            total_timesteps=warmup_transitions,
            callback=None,
            progress_bar=False,
            reset_num_timesteps=True,
        )
        model.ppo_train_seconds.clear()
        timer.stages.clear()
        timer.env_step_events.clear()
        timer.rollouts.clear()
        timer.vector_cycle_count = 0
        measured_timesteps = args.rollouts * shape.transitions_per_rollout
        if worker_timing_active:
            worker_measurement_started = env_method_optional(env, "start_worker_measurement")

        profiler = cProfile.Profile() if profile_path is not None else None
        torch_profiler = None
        if torch_profile_path is not None:
            torch_profiler = torch.profiler.profile(
                activities=(
                    [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                    if args.device == "cuda"
                    else [torch.profiler.ProfilerActivity.CPU]
                ),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            torch_profiler.__enter__()
        timer.sync()
        start = time.perf_counter()
        with timer.measure():
            if profiler is not None:
                profiler.enable()
            try:
                model.learn(
                    total_timesteps=measured_timesteps,
                    callback=None,
                    progress_bar=False,
                    reset_num_timesteps=True,
                )
            finally:
                if profiler is not None:
                    profiler.disable()
        timer.sync()
        envelope_seconds = time.perf_counter() - start
        if torch_profiler is not None:
            torch_profiler.__exit__(None, None, None)
            torch_profiler.export_chrome_trace(str(torch_profile_path))
        if profiler is not None and profile_path is not None:
            profiler.dump_stats(str(profile_path))
        if worker_timing_active:
            worker_rows = env_method_optional(env, "finish_worker_measurement")
    finally:
        env.close()

    if len(timer.rollouts) != args.rollouts:
        raise RuntimeError(f"Expected {args.rollouts} rollouts, got {len(timer.rollouts)}")
    if len(model.ppo_train_seconds) != args.rollouts:
        raise RuntimeError(
            f"Expected {args.rollouts} PPO train calls, got {len(model.ppo_train_seconds)}"
        )
    update_seconds = float(sum(model.ppo_train_seconds))
    pipeline = timer.report(
        envelope_seconds=envelope_seconds,
        transitions=measured_timesteps,
        n_envs=args.n_envs,
        rollouts=args.rollouts,
        update_seconds=update_seconds,
    )
    payload = {
        "result": {
            "label": "ppo_pipeline_profile",
            "seed": args.seed,
            "device": args.device,
            "torch_threads": torch.get_num_threads(),
            "n_envs": args.n_envs,
            "n_steps": shape.n_steps,
            "rollouts": args.rollouts,
            "rollout_transitions": shape.transitions_per_rollout,
            "warmup_transitions": args.warmup_transitions,
            "measured_timesteps": int(model.num_timesteps),
            "measured_transitions_requested": measured_timesteps,
            "batch_size": shape.batch_size,
            "requested_batch_size": shape.requested_batch_size,
            "epochs": args.epochs,
            "phase": str(getattr(phase_config, "phase_key", "unknown")),
            "phase_name": getattr(phase_config, "name", "unknown"),
            "configured_training_budget": getattr(model, "_configured_training_budget", None),
            "progress_total_timesteps": getattr(model, "diagnostic_progress_total_timesteps", []),
            "pipeline": pipeline,
            "worker_measurement_start": bt.json_safe(worker_measurement_started),
            "worker_rows": bt.json_safe(worker_rows),
            "worker_timing_requested": not args.no_worker_timing,
            "worker_timing_active": worker_timing_active,
            "profile_path": str(profile_path) if profile_path is not None else None,
            "profile_top_cumulative": cprofile_summary(profile_path)
            if profile_path is not None
            else [],
            "torch_profile_path": str(torch_profile_path) if torch_profile_path is not None else None,
            "source": {
                "sb3_collect_rollouts_file": inspect.getsourcefile(PPO.collect_rollouts),
                "repo_root": str(REPO_ROOT),
                "git": bt.git_metadata(),
                "package_versions": bt.package_versions(),
                "source_hashes": bt.collect_hashes(),
            },
        }
    }
    output_path = output_dir / "pipeline_profile.json"
    output_path.write_text(json.dumps(bt.json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    return payload["result"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile PPO rollout pipeline stages.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--n-envs", type=int, choices=[1, 16], default=16)
    parser.add_argument("--rollout-transitions", type=int, default=8192)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--warmup-transitions", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--torch-profile", action="store_true")
    parser.add_argument("--no-worker-timing", action="store_true")
    args = parser.parse_args()
    if args.rollouts <= 0:
        parser.error("--rollouts must be positive")
    if args.rollout_transitions <= 0:
        parser.error("--rollout-transitions must be positive")
    if args.warmup_transitions <= 0:
        parser.error("--warmup-transitions must be positive")
    if args.batch_size <= 1:
        parser.error("--batch-size must exceed one")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    return args


def main() -> int:
    args = parse_args()
    result = run_probe(args)
    print(f"Saved pipeline profile: {args.output / 'pipeline_profile.json'}")
    print(
        f"{result['label']}: {result['pipeline']['transitions_per_second']:.1f} transitions/s, "
        f"{result['pipeline']['vector_cycle_count']} vector cycles, "
        f"{result['rollouts']} rollouts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
