#!/usr/bin/env python3
"""E1 current-workload PPO throughput harness.

This script intentionally delegates real training semantics to the frozen
navigation-repair ``train_job.py`` and only wraps its module-local ``PPO``
symbol. Baseline mode records completed rollout/update cycles without internal
stage profiling. Profile mode additionally loads the prior AST-based rollout
and PPO-update profilers by explicit file path.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


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
DEFAULT_PIPELINE_HOOKS = (
    Path("/Users/elliot18/Projects/ml-research/AI_2")
    / "baselines"
    / "pipeline-profile-20260904"
    / "probe_pipeline.py"
)
DEFAULT_UPDATE_HOOKS = (
    Path("/Users/elliot18/Projects/ml-research/AI_2-ppo-update-profile-20260905")
    / "baselines"
    / "ppo-update-profile-20260905"
    / "update_hooks.py"
)
DEFAULT_ROLLOUT_POLICY_GRAPH = REPO_ROOT / "rollout_policy_graph.py"


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if dataclasses.is_dataclass(value):
            return json_safe(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


@dataclasses.dataclass(frozen=True)
class RunPlan:
    n_envs: int
    n_steps: int
    batch_size: int
    epochs: int
    warmup: int
    cycles: int
    transitions_per_cycle: int
    steps: int
    minibatches_per_epoch: int
    minibatches_per_cycle: int


def plan_run(*, n_envs: int, n_steps: int, batch_size: int, epochs: int, warmup: int, cycles: int) -> RunPlan:
    for name, value in {
        "n_envs": n_envs,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "epochs": epochs,
        "warmup": warmup,
        "cycles": cycles,
    }.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if n_envs <= 0 or n_steps <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("n_envs, n_steps, batch_size, and epochs must be positive")
    transitions = int(n_envs) * int(n_steps)
    minibatches_per_epoch = (transitions + int(batch_size) - 1) // int(batch_size)
    return RunPlan(
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        epochs=int(epochs),
        warmup=int(warmup),
        cycles=int(cycles),
        transitions_per_cycle=transitions,
        steps=(int(warmup) + int(cycles)) * transitions,
        minibatches_per_epoch=minibatches_per_epoch,
        minibatches_per_cycle=minibatches_per_epoch * int(epochs),
    )


def validate_output_path(path: Path) -> Path:
    if not Path(path).is_absolute():
        raise ValueError("--output must be absolute")
    return Path(path).resolve()


def import_module_from_path(name: str, path: Path) -> ModuleType:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_train_job_module(path: Path) -> ModuleType:
    resolved = Path(path).resolve()
    parent = str(resolved.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = import_module_from_path(resolved.stem, resolved)
    sys.modules["campaign_target_train_job"] = module
    return module


def import_runtime_helper_module(path: Path, alias: str) -> ModuleType:
    resolved = Path(path).resolve()
    parent = str(resolved.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = import_module_from_path(resolved.stem, resolved)
    sys.modules[alias] = module
    return module


def sync_for(device: str) -> Callable[[], None]:
    def sync() -> None:
        if str(device).startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ModuleNotFoundError:
                return

    return sync


@dataclasses.dataclass
class ProfileHooks:
    pipeline_module: ModuleType
    update_module: ModuleType
    pipeline_timer: Any
    update_timer: Any


def load_profile_hooks(
    pipeline_hooks: Path,
    update_hooks: Path,
    *,
    device: str,
    sync: Callable[[], None],
    profile_timing: str = "natural",
) -> ProfileHooks:
    pipeline = import_module_from_path("campaign_pipeline_hooks", pipeline_hooks)
    update = import_module_from_path("campaign_update_hooks", update_hooks)
    pipeline_timer = pipeline.PipelineTimer(enabled=True, sync=sync)
    update_timer = update.UpdateTimer(device=device, synchronize_stages=profile_timing == "sync")
    return ProfileHooks(
        pipeline_module=pipeline,
        update_module=update,
        pipeline_timer=pipeline_timer,
        update_timer=update_timer,
    )


def load_resident_buffer(path: Path) -> type:
    module = import_runtime_helper_module(path, "campaign_rollout_cache")
    cls = getattr(module, "CachedRolloutBuffer", None)
    if not isinstance(cls, type):
        raise TypeError(f"{path} did not define CachedRolloutBuffer")
    return cls


def load_grouped_env_module(path: Path) -> ModuleType:
    return import_runtime_helper_module(path, "campaign_grouped_vec_env")


def load_rollout_policy_graph_module(path: Path) -> ModuleType:
    return import_runtime_helper_module(path, "campaign_rollout_policy_graph")


def patch_grouped_vec_env(train_job: ModuleType, args: argparse.Namespace) -> Callable[[], None] | None:
    group_size = int(getattr(args, "group_size", 1))
    if group_size <= 1:
        return None
    grouped_module = load_grouped_env_module(Path(args.grouped_env_path))
    grouped_cls = getattr(grouped_module, "GroupedSubprocVecEnv", None)
    if not isinstance(grouped_cls, type):
        raise TypeError(f"{args.grouped_env_path} did not define GroupedSubprocVecEnv")
    original = train_job.SubprocVecEnv

    def grouped_subproc_vec_env(env_fns: Any, start_method: str | None = None):
        return grouped_cls(env_fns, group_size=group_size, start_method=start_method)

    train_job.SubprocVecEnv = grouped_subproc_vec_env

    def cleanup() -> None:
        train_job.SubprocVecEnv = original

    return cleanup


@dataclasses.dataclass
class CampaignState:
    output: Path
    plan: RunPlan
    sync: Callable[[], None]
    mode: str = "baseline"
    profile_timing: str = "natural"
    resident_buffer_class: type | None = None
    rollout_policy_graph_module: ModuleType | None = None
    rollout_policy_graph_capture_warmup: int = 3
    rollout_policy_graph_tolerance: float = 1e-5
    rollout_policy_graph_adapter: Any | None = None
    profile_hooks: ProfileHooks | None = None
    started_at: float = dataclasses.field(default_factory=time.perf_counter)
    learn_started_at: float | None = None
    cycle_boundary_at: float | None = None
    cycle_index: int = 0
    measured_cycle_count: int = 0
    pending_rollout_seconds: float | None = None
    pending_rollout_ok: bool | None = None
    records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    rollout_policy_graph_records: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    @property
    def metrics_path(self) -> Path:
        return self.output / "campaign_metrics.jsonl"

    def mark_learn_started(self) -> None:
        if self.learn_started_at is None:
            self.sync()
            self.learn_started_at = time.perf_counter()
            self.cycle_boundary_at = self.learn_started_at

    def record_rollout(self, seconds: float, ok: Any) -> None:
        self.pending_rollout_seconds = float(seconds)
        self.pending_rollout_ok = bool(ok)

    def record_update(self, seconds: float, model: Any) -> None:
        if not self.pending_rollout_ok or self.cycle_boundary_at is None:
            raise RuntimeError("cannot count an update without a completed rollout")
        ended_at = time.perf_counter()
        cycle_seconds = ended_at - self.cycle_boundary_at
        self.cycle_boundary_at = ended_at
        self.cycle_index += 1
        is_measured = self.cycle_index > self.plan.warmup
        measured_index = self.cycle_index - self.plan.warmup if is_measured else 0
        if is_measured:
            self.measured_cycle_count += 1
        record = {
            "cycle_index": self.cycle_index,
            "phase": "measured" if is_measured else "warmup",
            "measured_cycle": measured_index,
            "rollout_seconds": self.pending_rollout_seconds,
            "update_seconds": float(seconds),
            "cycle_seconds": cycle_seconds,
            "boundary_seconds": cycle_seconds - float(self.pending_rollout_seconds) - float(seconds),
            "rollout_completed": self.pending_rollout_ok,
            "transitions": self.plan.transitions_per_cycle,
            "transitions_per_second": self.plan.transitions_per_cycle
            / max(cycle_seconds, 1e-12),
            "num_timesteps": int(getattr(model, "num_timesteps", 0)),
            "timestamp_unix": time.time(),
        }
        if self.mode == "profile" and self.profile_hooks is not None:
            if hasattr(self.profile_hooks.update_timer, "report"):
                record["update_profile"] = self.profile_hooks.update_timer.report()
                if hasattr(self.profile_hooks.update_timer, "reset"):
                    self.profile_hooks.update_timer.reset()
            pipeline_timer = self.profile_hooks.pipeline_timer
            if hasattr(pipeline_timer, "stages"):
                record["collect_profile_stages"] = {
                    str(label): json_safe(stats.to_json() if hasattr(stats, "to_json") else stats)
                    for label, stats in getattr(pipeline_timer, "stages", {}).items()
                }
                getattr(pipeline_timer, "stages", {}).clear()
        self.records.append(record)
        append_jsonl(self.metrics_path, record)
        self.pending_rollout_seconds = None
        self.pending_rollout_ok = None

    def measured_records(self) -> list[dict[str, Any]]:
        return [record for record in self.records if record["phase"] == "measured"]

    def summary(self, *, status: str, error: str | None = None) -> dict[str, Any]:
        measured = self.measured_records()
        measured_seconds = sum(float(record["cycle_seconds"]) for record in measured)
        total_transitions = self.plan.transitions_per_cycle * len(measured)
        learn_elapsed = None
        if self.learn_started_at is not None:
            self.sync()
            learn_elapsed = (self.cycle_boundary_at or self.learn_started_at) - self.learn_started_at
        return {
            "status": status,
            "error": error,
            "mode": self.mode,
            "completed_cycles_total": self.cycle_index,
            "completed_measured_cycles": len(measured),
            "requested_measured_cycles": self.plan.cycles,
            "warmup_cycles": self.plan.warmup,
            "plan": dataclasses.asdict(self.plan),
            "startup_seconds": None if self.learn_started_at is None else self.learn_started_at - self.started_at,
            "learning_wall_seconds": learn_elapsed,
            "measured_cycle_seconds": measured_seconds,
            "measured_transitions": total_transitions,
            "measured_transitions_per_second": total_transitions / measured_seconds if measured_seconds else 0.0,
            "measured_collect_seconds": sum(float(r["rollout_seconds"]) for r in measured),
            "measured_update_seconds": sum(float(r["update_seconds"]) for r in measured),
            "measured_boundary_seconds": sum(float(r["boundary_seconds"]) for r in measured),
            "profile_timing": (
                ("natural host spans; instrumented diagnostic, not headline throughput"
                 if self.profile_timing == "natural" else
                 "synchronized update stages; instrumented diagnostic, not natural throughput")
                if self.mode == "profile" else "boundary timing only"
            ),
            "rollout_policy_graph": {
                "enabled": self.rollout_policy_graph_module is not None,
                "records": self.rollout_policy_graph_records,
            },
            "residual_note": (
                "Primary SPS uses consecutive completed-update boundaries after warmup, "
                "including between-call logging, callbacks, periodic saves and measurement overhead. "
                "Boundary seconds reconcile elapsed time against collect and update spans. "
                "Startup and final checkpoint/tracking closeout are outside that steady-state window."
            ),
            "records_path": str(self.metrics_path),
        }


def make_measured_ppo_class(base_cls: type, state: CampaignState) -> type:
    hooks = state.profile_hooks
    profile_collect = None
    profile_train = None
    if state.mode == "profile":
        if hooks is None:
            raise ValueError("profile mode requires profile hooks")
        import inspect
        from types import FunctionType

        tree = hooks.pipeline_module.instrument_collect_rollouts_source(inspect.getsource(base_cls.collect_rollouts))
        namespace = dict(base_cls.collect_rollouts.__globals__)
        namespace["self"] = None
        exec(compile(tree, filename="<campaign_collect_rollouts>", mode="exec"), namespace)
        profile_collect = namespace.get("collect_rollouts")
        if not isinstance(profile_collect, FunctionType):
            raise RuntimeError("profile collect_rollouts did not compile")
        profile_train = hooks.update_module.instrument_train(base_cls.train)

    class CampaignPPO(base_cls):
        __campaign_base_cls__ = base_cls

        def _excluded_save_params(self) -> list[str]:
            return super()._excluded_save_params() + [
                "campaign_state",
                "pipeline_timer",
                "update_timer",
                "rollout_policy_graph_adapter",
            ]

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if state.resident_buffer_class is not None:
                kwargs.setdefault("rollout_buffer_class", state.resident_buffer_class)
            super().__init__(*args, **kwargs)
            self.campaign_state = state
            if hooks is not None:
                self.pipeline_timer = hooks.pipeline_timer
                self.update_timer = hooks.update_timer

        @classmethod
        def load(cls, *args: Any, **kwargs: Any) -> Any:
            if state.resident_buffer_class is not None:
                custom = dict(kwargs.get("custom_objects") or {})
                custom["rollout_buffer_class"] = state.resident_buffer_class
                kwargs["custom_objects"] = custom
            model = super(CampaignPPO, cls).load(*args, **kwargs)
            if state.resident_buffer_class is not None and not hasattr(model, "rollout_buffer_class"):
                model.rollout_buffer_class = state.resident_buffer_class
            return model

        def collect_rollouts(self, *args: Any, **kwargs: Any) -> Any:
            state.mark_learn_started()
            state.sync()
            started = time.perf_counter()
            if hooks is not None and hasattr(hooks.pipeline_timer, "enabled"):
                hooks.pipeline_timer.enabled = True
            try:
                if state.rollout_policy_graph_module is not None:
                    with state.rollout_policy_graph_module.policy_forward_graph_context(
                        self.policy,
                        adapter=state.rollout_policy_graph_adapter,
                        expected_batch_size=state.plan.n_envs,
                        capture_warmup=state.rollout_policy_graph_capture_warmup,
                        tolerance=state.rollout_policy_graph_tolerance,
                    ) as graph_adapter:
                        state.rollout_policy_graph_adapter = graph_adapter
                        if profile_collect is not None:
                            ok = profile_collect(self, *args, **kwargs)
                        else:
                            ok = super().collect_rollouts(*args, **kwargs)
                    receipt = getattr(graph_adapter, "preflight_receipt", None)
                    if not isinstance(receipt, dict) or receipt.get("status") != "passed":
                        raise RuntimeError("rollout policy graph preflight did not pass before collect output")
                    if hasattr(graph_adapter, "summary"):
                        state.rollout_policy_graph_records.append(json_safe(graph_adapter.summary()))
                else:
                    if profile_collect is not None:
                        ok = profile_collect(self, *args, **kwargs)
                    else:
                        ok = super().collect_rollouts(*args, **kwargs)
                return ok
            finally:
                state.sync()
                state.record_rollout(time.perf_counter() - started, locals().get("ok", False))

        def train(self) -> Any:
            state.sync()
            started = time.perf_counter()
            cleanup = None
            succeeded = False
            if hooks is not None and hasattr(hooks.update_module, "instrument_buffer"):
                cleanup = hooks.update_module.instrument_buffer(self.rollout_buffer, hooks.update_timer)
            try:
                if profile_train is not None:
                    result = profile_train(self)
                else:
                    result = super().train()
                succeeded = True
                return result
            finally:
                if cleanup is not None:
                    cleanup()
                clear_cache = getattr(getattr(self, "rollout_buffer", None), "clear_cache", None)
                if clear_cache is not None:
                    clear_cache()
                state.sync()
                elapsed = time.perf_counter() - started
                if succeeded:
                    state.record_update(elapsed, self)
                else:
                    append_jsonl(state.output / "campaign_failures.jsonl", {
                        "event": "incomplete_update", "seconds": elapsed,
                        "completed_cycles": state.cycle_index,
                    })

    CampaignPPO.__name__ = "CampaignPPO"
    return CampaignPPO


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault(
        "WANDB_NOTES",
        (
            "Goal: E1 current-workload PPO throughput campaign. "
            f"Change: mode={args.mode}, resident_buffer={bool(args.resident_buffer)}, "
            f"rollout_policy_graph={bool(getattr(args, 'rollout_policy_graph', False))}. "
            "Expect: completed rollout/update timing cycles, not gameplay-quality proof. "
            f"Setup: source={Path(args.source).resolve()} checkpoint={Path(args.checkpoint).resolve()}."
        ),
    )


def copy_source_snapshot(source_path: Path, output: Path) -> None:
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    for name in ("train_job.py", "train.py", "config.yaml", "control.yaml"):
        candidate = Path(source_path) / name
        if candidate.is_file():
            shutil.copy2(candidate, source_dir / name)


def write_provenance(
    output: Path,
    *,
    args: argparse.Namespace,
    train_job_module: ModuleType,
    state: CampaignState,
) -> None:
    import inspect

    records = {
        "created_unix": time.time(),
        "argv": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "which": {"python": shutil.which("python"), "python3": shutil.which("python3")},
        "files": {
            "config": file_record(Path(args.config)),
            "checkpoint": file_record(Path(args.checkpoint)),
            "source_train": file_record(Path(args.source) / "train.py"),
            "train_job": file_record(Path(args.train_job)),
            "pipeline_hooks": file_record(Path(args.pipeline_hooks)),
            "update_hooks": file_record(Path(args.update_hooks)),
            "resident_buffer": file_record(Path(args.rollout_cache_path)) if args.resident_buffer else None,
            "grouped_env": file_record(Path(args.grouped_env_path)) if int(getattr(args, "group_size", 1)) > 1 else None,
            "rollout_policy_graph": file_record(Path(args.rollout_policy_graph_path))
            if bool(getattr(args, "rollout_policy_graph", False))
            else None,
        },
        "plan": dataclasses.asdict(state.plan),
        "mode": state.mode,
        "profile_timing": getattr(args, "profile_timing", "natural"),
        "grouping": {
            "group_size": int(getattr(args, "group_size", 1)),
            "grouped_env_path": str(Path(args.grouped_env_path).resolve()),
        },
        "rollout_policy_graph": {
            "enabled": bool(getattr(args, "rollout_policy_graph", False)),
            "path": str(Path(args.rollout_policy_graph_path).resolve()),
            "capture_warmup": int(getattr(args, "rollout_policy_graph_capture_warmup", 3)),
            "tolerance": float(getattr(args, "rollout_policy_graph_tolerance", 1e-5)),
            "runtime_records": state.rollout_policy_graph_records,
        },
    }
    ppo = getattr(train_job_module, "PPO", None)
    if ppo is not None:
        base_ppo = getattr(ppo, "__campaign_base_cls__", ppo)
        collect = getattr(ppo, "collect_rollouts", None)
        train = getattr(ppo, "train", None)
        base_collect = getattr(base_ppo, "collect_rollouts", None)
        base_train = getattr(base_ppo, "train", None)
        records["ppo_source"] = {
            "class": repr(ppo),
            "collect_rollouts_file": inspect.getsourcefile(collect) if collect is not None else None,
            "train_file": inspect.getsourcefile(train) if train is not None else None,
            "base_class": repr(base_ppo),
            "base_collect_rollouts_file": inspect.getsourcefile(base_collect) if base_collect is not None else None,
            "base_train_file": inspect.getsourcefile(base_train) if base_train is not None else None,
            "base_collect_rollouts_sha256": hashlib.sha256(
                inspect.getsource(base_collect).encode("utf-8")
            ).hexdigest()
            if base_collect is not None
            else None,
            "base_train_sha256": hashlib.sha256(inspect.getsource(base_train).encode("utf-8")).hexdigest()
            if base_train is not None
            else None,
        }
    write_json(output / "campaign_provenance.json", records)


def build_train_job_args(args: argparse.Namespace, plan: RunPlan) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(Path(args.config).resolve()),
        phase=str(args.phase),
        seed=int(args.seed),
        steps=int(plan.steps),
        n_envs=int(args.n_envs),
        device=str(args.device),
        output=str(Path(args.output).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()),
        max_seconds=args.maxseconds,
        history=bool(args.history),
        opponents=args.opponents,
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
    )


def run_campaign(args: argparse.Namespace) -> int:
    if getattr(args, "rollout_cache_path", None) is None:
        args.rollout_cache_path = Path(args.source) / "rollout_cache.py"
    if getattr(args, "grouped_env_path", None) is None:
        args.grouped_env_path = Path(args.source) / "grouped_vec_env.py"
    output = validate_output_path(Path(args.output))
    output.mkdir(parents=True, exist_ok=True)
    configure_environment(args)
    plan = plan_run(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup=args.warmup,
        cycles=args.cycles,
    )
    sync = sync_for(args.device)
    hooks = None
    if args.mode == "profile":
        hooks = load_profile_hooks(
            Path(args.pipeline_hooks),
            Path(args.update_hooks),
            device=args.device,
            sync=sync,
            profile_timing=args.profile_timing,
        )
    resident_class = load_resident_buffer(Path(args.rollout_cache_path)) if args.resident_buffer else None
    rollout_policy_graph_module = (
        load_rollout_policy_graph_module(Path(args.rollout_policy_graph_path))
        if getattr(args, "rollout_policy_graph", False)
        else None
    )
    state = CampaignState(
        output=output,
        plan=plan,
        sync=sync,
        mode=args.mode,
        profile_timing=args.profile_timing,
        resident_buffer_class=resident_class,
        rollout_policy_graph_module=rollout_policy_graph_module,
        rollout_policy_graph_capture_warmup=args.rollout_policy_graph_capture_warmup,
        rollout_policy_graph_tolerance=args.rollout_policy_graph_tolerance,
        profile_hooks=hooks,
    )
    write_json(output / "campaign_plan.json", dataclasses.asdict(plan))
    copy_source_snapshot(Path(args.source).resolve(), output)

    train_job = import_train_job_module(Path(args.train_job))
    original_ppo = getattr(train_job, "PPO")
    train_job.PPO = make_measured_ppo_class(original_ppo, state)
    grouped_cleanup = patch_grouped_vec_env(train_job, args)
    if str(Path(args.source).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(args.source).resolve()))
    previous_cwd = Path.cwd()
    status = "completed"
    error = None
    try:
        os.chdir(Path(args.source).resolve())
        train_job_args = build_train_job_args(args, plan)
        write_provenance(output, args=args, train_job_module=train_job, state=state)
        train_job.run_job(train_job_args)
    except Exception as exc:
        status = "exception"
        error = f"{type(exc).__name__}: {exc}"
        write_json(output / "campaign_exception.json", {"error": error, "traceback": traceback.format_exc()})
    finally:
        os.chdir(previous_cwd)
        train_job.PPO = original_ppo
        if grouped_cleanup is not None:
            grouped_cleanup()

    if status == "completed" and state.measured_cycle_count != plan.cycles:
        status = "incomplete"
        error = (
            f"expected {plan.cycles} measured cycles after {plan.warmup} warmup, "
            f"got {state.measured_cycle_count}"
        )
    write_json(output / "campaign_summary.json", state.summary(status=status, error=error))
    if status == "completed":
        return 0
    if status == "incomplete":
        return 2
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E1 current-workload PPO throughput harness")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=DEFAULT_TRAIN_JOB)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maxseconds", type=float, default=None)
    parser.add_argument("--phase", default="0b")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--opponents")
    parser.add_argument("--mode", choices=["baseline", "profile"], default="baseline")
    parser.add_argument("--pipeline-hooks", type=Path, default=DEFAULT_PIPELINE_HOOKS)
    parser.add_argument("--update-hooks", type=Path, default=DEFAULT_UPDATE_HOOKS)
    parser.add_argument("--profile-timing", choices=["natural", "sync"], default="natural")
    parser.add_argument("--resident-buffer", action="store_true")
    parser.add_argument("--rollout-cache-path", type=Path)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--grouped-env-path", type=Path)
    parser.add_argument("--rollout-policy-graph", action="store_true")
    parser.add_argument("--rollout-policy-graph-path", type=Path, default=DEFAULT_ROLLOUT_POLICY_GRAPH)
    parser.add_argument("--rollout-policy-graph-capture-warmup", type=int, default=3)
    parser.add_argument("--rollout-policy-graph-tolerance", type=float, default=1e-5)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    if args.cycles <= 0:
        parser.error("--cycles must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.n_envs <= 0 or args.n_steps <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        parser.error("--n-envs, --n-steps, --batch-size, and --epochs must be positive")
    if args.group_size <= 0:
        parser.error("--group-size must be positive")
    if args.rollout_policy_graph_capture_warmup < 0:
        parser.error("--rollout-policy-graph-capture-warmup must be nonnegative")
    if args.rollout_policy_graph_tolerance <= 0:
        parser.error("--rollout-policy-graph-tolerance must be positive")
    if not Path(args.output).is_absolute():
        parser.error("--output must be absolute")
    args.config = Path(args.config).resolve()
    args.checkpoint = Path(args.checkpoint).resolve()
    args.source = Path(args.source).resolve()
    args.train_job = Path(args.train_job).resolve()
    args.output = Path(args.output).resolve()
    args.pipeline_hooks = Path(args.pipeline_hooks).resolve()
    args.update_hooks = Path(args.update_hooks).resolve()
    if args.rollout_cache_path is None:
        args.rollout_cache_path = args.source / "rollout_cache.py"
    args.rollout_cache_path = Path(args.rollout_cache_path).resolve()
    if args.grouped_env_path is None:
        args.grouped_env_path = args.source / "grouped_vec_env.py"
    args.grouped_env_path = Path(args.grouped_env_path).resolve()
    args.rollout_policy_graph_path = Path(args.rollout_policy_graph_path).resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = validate_args(parser, parser.parse_args(argv))
    code = run_campaign(args)
    summary_path = Path(args.output) / "campaign_summary.json"
    print(f"campaign_summary={summary_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
