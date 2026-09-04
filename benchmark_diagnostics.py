#!/usr/bin/env python3
"""Standalone instrumented PPO diagnostic for the AI_2 training stack.

This benchmark is intentionally labeled as instrumented output. It measures one
fresh-policy PPO run on one vectorized environment and breaks the measured
region into rollout, update, policy-forward, vector-env, and raw-game timings.
Warmup and initialization are excluded from the headline measured wall time.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import time
import types
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import benchmark_training as bt


REPO_ROOT = Path(__file__).resolve().parent


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
    from train import linear_schedule

    if isinstance(configured, list):
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
    return lambda: bt.sync_device(device)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instrument one fresh-policy single-env PPO diagnostic run."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ppo-rollouts", type=int, default=2)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "diagnostic_results")
    parser.add_argument("--profile", action="store_true", help="Write cProfile for measured region only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.n_steps <= 1:
        parser.error("--n-steps must exceed one")
    if args.batch_size <= 1:
        parser.error("--batch-size must exceed one")
    if args.ppo_rollouts <= 0:
        parser.error("--ppo-rollouts must be positive")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    return args


def main() -> int:
    bt.set_headless_environment()
    args = parse_args()
    bt.require_runtime_dependencies()
    bt.configure_torch(args.device, args.torch_threads)
    result = run_diagnostic(args, args.output)
    print(f"Saved instrumented diagnostic: {args.output / 'benchmark_diagnostics.json'}")
    print(
        f"instrumented_single_env_ppo: {result['steps_per_second']:.1f} env-steps/s, "
        f"{result['rollout_count']} rollouts, {result['update_count']} updates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
