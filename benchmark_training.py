#!/usr/bin/env python3
"""Benchmark the current Python training stack before replacing game code.

The script intentionally avoids checkpoint/video/tensorboard writes. It records
raw environment timing, wrapped navigation-env timing, and optional short PPO
rollout/update timing into a timestamped output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
UTMIST_ROOT = REPO_ROOT / "UTMIST-AI2-main"
DEFAULT_HASH_PATHS = [
    "benchmark_training.py",
    "train.py",
    "config.yaml",
    "positions.json",
    "UTMIST-AI2-main/environment/environment.py",
    "UTMIST-AI2-main/environment/agent.py",
    "UTMIST-AI2-main/environment/constants.py",
    "UTMIST-AI2-main/environment/unarmed_attacks",
    "UTMIST-AI2-main/environment/spear_attacks",
    "UTMIST-AI2-main/environment/hammer_attacks",
]


@dataclass
class RepeatResult:
    mode: str
    repeat: int
    seed: int
    n_envs: int
    requested_steps: int
    measured_steps: int
    total_seconds: float
    reset_seconds: float | None
    init_seconds: float
    steps_per_second: float
    completed_episodes: int
    partial_episodes: int
    aggregate_episodes_per_second: float
    mean_episode_steps: float | None
    mean_episode_wall_seconds: float | None
    extra: dict[str, Any]


class TimingCallback:
    """Small SB3 callback that records rollout and episode boundaries."""

    def __init__(
        self,
        mode: str = "ppo",
        repeat: int = 0,
        n_envs: int = 1,
    ) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, owner: TimingCallback) -> None:
                super().__init__(verbose=0)
                self.owner = owner

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
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                for idx, done in enumerate(dones):
                    self.owner.episode_steps[idx] += 1
                    if bool(done):
                        info = infos[idx] if idx < len(infos) else {}
                        self.owner.episodes.append(
                            {
                                "mode": self.owner.mode,
                                "repeat": self.owner.repeat,
                                "env_index": idx,
                                "episode_steps": self.owner.episode_steps[idx],
                                "wall_seconds": now - self.owner.episode_start[idx],
                                "termination": "done",
                                "info": json_safe(
                                    {k: v for k, v in info.items() if k != "terminal_observation"}
                                ),
                            }
                        )
                        self.owner.episode_steps[idx] = 0
                        self.owner.episode_start[idx] = now
                return True

        self.rollout_seconds: list[float] = []
        self._rollout_start: float | None = None
        self.mode = mode
        self.repeat = repeat
        self.n_envs = n_envs
        self.episodes: list[dict[str, Any]] = []
        self.episode_steps = [0 for _ in range(n_envs)]
        self.episode_start = [time.perf_counter() for _ in range(n_envs)]
        self.callback = _Callback(self)


def set_headless_environment() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="ai2-mpl-"))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ai2-cache-"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def configure_torch(device: str, torch_threads: int) -> None:
    import torch

    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(max(1, min(torch_threads, 4)))
    except RuntimeError:
        pass
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested --device mps but MPS is unavailable")


def sync_device(device: str) -> None:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def require_runtime_dependencies() -> None:
    missing = []
    for module in ("yaml", "numpy", "torch", "gymnasium", "stable_baselines3"):
        try:
            __import__(module)
        except Exception as exc:
            missing.append(f"{module}: {exc}")
    if missing:
        joined = "\n  - ".join(missing)
        raise RuntimeError(
            "Missing training dependencies. Install the repo requirements or run "
            "inside the training virtualenv.\n  - " + joined
        )


def load_config(config_path: Path) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def add_repo_imports() -> None:
    root = str(UTMIST_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def make_output_dir(base: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = base / f"baseline_{stamp}"
    suffix = 1
    while out.exists():
        out = base / f"baseline_{stamp}_{suffix}"
        suffix += 1
    out.mkdir(parents=True, exist_ok=False)
    return out


def action_plan(space: Any, steps: int, seed: int, n_agents: int = 1) -> list[Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    plan = []
    shape = tuple(space.shape)
    low = np.asarray(space.low)
    high = np.asarray(space.high)
    is_binary_box = np.all(low == 0) and np.all(high == 1)

    for _ in range(steps):
        if is_binary_box:
            if n_agents == 1:
                plan.append(rng.integers(0, 2, size=shape).astype(np.float32))
            else:
                plan.append(
                    {
                        agent: rng.integers(0, 2, size=shape).astype(np.float32)
                        for agent in range(n_agents)
                    }
                )
        else:
            sample = rng.uniform(low, high, size=shape).astype(space.dtype)
            plan.append(sample)
    return plan


def hash_actions(actions: Iterable[Any]) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for item in actions:
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                digest.update(np.asarray(item[key]).tobytes())
        else:
            digest.update(np.asarray(item).tobytes())
    return digest.hexdigest()


def build_raw_env(seed: int):
    add_repo_imports()
    seed_everything(seed)
    from environment.agent import CameraResolution
    from environment.environment import WarehouseBrawl

    return WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=True)


def build_wrapped_nav_env(config_path: Path, seed: int, n_envs: int, torch_threads: int):
    add_repo_imports()
    seed_everything(seed)
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
    from train import PhaseConfig, get_resolution, make_nav_env

    params = load_config(config_path)
    phase_key = params.get("curriculum", {}).get("phase", 1)
    phase_config = PhaseConfig(params, phase_key)
    if not phase_config.is_navigation:
        raise ValueError(
            f"Wrapped benchmark currently targets navigation phases; config selects {phase_key!r}"
        )
    resolution = get_resolution(
        params.get("environment_settings", {}).get("resolution", "LOW")
    )

    def make_one(worker_index: int):
        def _factory():
            seed_everything(seed + worker_index)
            configure_torch("cpu", torch_threads)
            return make_nav_env(phase_config, resolution, params)

        return _factory

    env_fns = [make_one(i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn") if n_envs > 1 else DummyVecEnv(env_fns)
    frame_stack = int(params.get("frame_stack", 1))
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return vec_env, params, phase_config


def summarize_episodes(records: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if not records:
        return None, None
    return (
        float(statistics.mean(item["episode_steps"] for item in records)),
        float(statistics.mean(item["wall_seconds"] for item in records)),
    )


def run_raw_repeat(args: argparse.Namespace, repeat: int, output_dir: Path) -> tuple[RepeatResult, list[dict[str, Any]]]:
    seed = args.seed + repeat
    t0 = time.perf_counter()
    env = build_raw_env(seed)
    init_seconds = time.perf_counter() - t0
    actions = action_plan(env.action_space, args.steps, seed, n_agents=2)
    warmup_actions = action_plan(
        env.action_space, args.warmup_steps, seed + 100_000, n_agents=2
    )
    episodes: list[dict[str, Any]] = []
    episode_steps = 0
    reset_seconds = 0.0
    initial_reset_seconds = 0.0
    measured_steps = 0
    try:
        reset_t0 = time.perf_counter()
        env.reset(seed=seed)
        initial_reset_seconds += time.perf_counter() - reset_t0
        for action in warmup_actions:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset(seed=seed)
        reset_t0 = time.perf_counter()
        env.reset(seed=seed)
        initial_reset_seconds += time.perf_counter() - reset_t0
        episode_start = time.perf_counter()
        t1 = time.perf_counter()
        for action in actions:
            _, _, terminated, truncated, _ = env.step(action)
            measured_steps += 1
            episode_steps += 1
            if terminated or truncated:
                now = time.perf_counter()
                episodes.append(
                    {
                        "mode": "raw",
                        "repeat": repeat,
                        "env_index": 0,
                        "episode_steps": episode_steps,
                        "wall_seconds": now - episode_start,
                        "termination": "terminated" if terminated else "truncated",
                    }
                )
                reset_t0 = time.perf_counter()
                env.reset(seed=seed + measured_steps)
                reset_seconds += time.perf_counter() - reset_t0
                episode_start = time.perf_counter()
                episode_steps = 0
        total_seconds = time.perf_counter() - t1
    finally:
        env.close()

    mean_steps, mean_wall = summarize_episodes(episodes)
    result = RepeatResult(
        mode="raw",
        repeat=repeat,
        seed=seed,
        n_envs=1,
        requested_steps=args.steps,
        measured_steps=measured_steps,
        total_seconds=total_seconds,
        reset_seconds=reset_seconds,
        init_seconds=init_seconds,
        steps_per_second=measured_steps / total_seconds if total_seconds else 0.0,
        completed_episodes=len(episodes),
        partial_episodes=1 if episode_steps else 0,
        aggregate_episodes_per_second=len(episodes) / total_seconds if total_seconds else 0.0,
        mean_episode_steps=mean_steps,
        mean_episode_wall_seconds=mean_wall,
        extra={
            "action_seed": seed,
            "action_sha256": hash_actions(actions),
            "warmup_steps": args.warmup_steps,
            "initial_reset_seconds": initial_reset_seconds,
            "max_timesteps": getattr(env, "max_timesteps", None),
            "episode_wall_note": "Raw episode wall time excludes reset; total throughput includes natural resets.",
            "timing_note": "init_seconds and initial_reset_seconds exclude measured loop; reset_seconds is natural terminal reset cost included inside total_seconds.",
        },
    )
    write_repeat(output_dir, result, episodes)
    return result, episodes


def run_wrapped_repeat(args: argparse.Namespace, repeat: int, output_dir: Path) -> tuple[RepeatResult, list[dict[str, Any]]]:
    seed = args.seed + repeat
    t0 = time.perf_counter()
    env, params, phase_config = build_wrapped_nav_env(args.config, seed, args.n_envs, args.torch_threads)
    init_seconds = time.perf_counter() - t0
    actions = action_plan(env.action_space, args.steps, seed, n_agents=1)
    warmup_actions = action_plan(
        env.action_space, args.warmup_steps, seed + 100_000, n_agents=1
    )
    episodes: list[dict[str, Any]] = []
    episode_steps = [0 for _ in range(args.n_envs)]
    reset_seconds = 0.0
    initial_reset_seconds = 0.0
    measured_steps = 0
    try:
        reset_t0 = time.perf_counter()
        env.reset()
        initial_reset_seconds += time.perf_counter() - reset_t0
        for action in warmup_actions:
            env.step([action for _ in range(args.n_envs)])
        reset_t0 = time.perf_counter()
        env.reset()
        initial_reset_seconds += time.perf_counter() - reset_t0
        episode_start = [time.perf_counter() for _ in range(args.n_envs)]
        t1 = time.perf_counter()
        for action in actions:
            _, _, dones, infos = env.step([action for _ in range(args.n_envs)])
            measured_steps += args.n_envs
            for idx in range(args.n_envs):
                episode_steps[idx] += 1
                if bool(dones[idx]):
                    now = time.perf_counter()
                    episodes.append(
                        {
                            "mode": "wrapped",
                            "repeat": repeat,
                            "env_index": idx,
                            "episode_steps": episode_steps[idx],
                            "wall_seconds": now - episode_start[idx],
                            "termination": "done",
                            "info": json_safe(
                                {k: v for k, v in infos[idx].items() if k != "terminal_observation"}
                            ),
                        }
                    )
                    episode_start[idx] = now
                    episode_steps[idx] = 0
        total_seconds = time.perf_counter() - t1
    finally:
        env.close()

    mean_steps, mean_wall = summarize_episodes(episodes)
    result = RepeatResult(
        mode="wrapped",
        repeat=repeat,
        seed=seed,
        n_envs=args.n_envs,
        requested_steps=args.steps,
        measured_steps=measured_steps,
        total_seconds=total_seconds,
        reset_seconds=None,
        init_seconds=init_seconds,
        steps_per_second=measured_steps / total_seconds if total_seconds else 0.0,
        completed_episodes=len(episodes),
        partial_episodes=sum(1 for steps in episode_steps if steps),
        aggregate_episodes_per_second=len(episodes) / total_seconds if total_seconds else 0.0,
        mean_episode_steps=mean_steps,
        mean_episode_wall_seconds=mean_wall,
        extra={
            "phase": str(phase_config.phase_key),
            "phase_name": phase_config.name,
            "frame_stack": params.get("frame_stack"),
            "opponent_history": params.get("opponent_history", {}),
            "action_seed": seed,
            "action_sha256": hash_actions(actions),
            "warmup_steps": args.warmup_steps,
            "initial_reset_seconds": initial_reset_seconds,
            "episode_wall_note": "Vector episode wall time includes automatic terminal reset; reset cost is not measured separately.",
        },
    )
    write_repeat(output_dir, result, episodes)
    return result, episodes


def run_ppo_repeat(args: argparse.Namespace, repeat: int, output_dir: Path) -> tuple[RepeatResult, list[dict[str, Any]]]:
    import torch
    from stable_baselines3 import PPO

    add_repo_imports()
    from train import linear_schedule

    seed = args.seed + repeat
    seed_everything(seed)
    env, params, phase_config = build_wrapped_nav_env(
        args.config, seed, args.n_envs, args.torch_threads
    )
    ppo = params.get("ppo_settings", {})
    n_steps = int(args.n_steps or ppo.get("n_steps", 1024))
    requested_batch_size = int(args.batch_size or ppo.get("batch_size", 8192))
    rollout_buffer_size = n_steps * args.n_envs
    batch_size = min(requested_batch_size, rollout_buffer_size)
    warmup_timesteps = int(n_steps * args.n_envs)
    measured_timesteps = int(args.ppo_rollouts * n_steps * args.n_envs)
    configured_budget = int(ppo.get("time_steps", max(warmup_timesteps + measured_timesteps, 1)))
    lr_config = ppo.get("learning_rate", [3e-4, 1e-6])
    learning_rate = (
        linear_schedule(lr_config[0], lr_config[1])
        if isinstance(lr_config, list)
        else lr_config
    )
    clip_config = ppo.get("clip_range", [0.2, 0.05])
    clip_range = (
        linear_schedule(clip_config[0], clip_config[1])
        if isinstance(clip_config, list)
        else clip_config
    )

    class TimedPPO(PPO):
        def __init__(self, *model_args, **model_kwargs) -> None:
            self.update_seconds: list[float] = []
            super().__init__(*model_args, **model_kwargs)

        def _update_current_progress_remaining(
            self, num_timesteps: int, total_timesteps: int
        ) -> None:
            super()._update_current_progress_remaining(
                num_timesteps, max(configured_budget, total_timesteps)
            )

        def train(self) -> None:
            sync_device(args.device)
            start = time.perf_counter()
            super().train()
            sync_device(args.device)
            self.update_seconds.append(time.perf_counter() - start)

    init_t0 = time.perf_counter()
    try:
        model = TimedPPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=int(ppo.get("n_epochs", 8)),
            gamma=float(ppo.get("gamma", 0.99)),
            gae_lambda=float(ppo.get("gae_lambda", 0.95)),
            clip_range=clip_range,
            ent_coef=float(ppo.get("nav_ent_coef", ppo.get("ent_coef", 0.01))),
            vf_coef=float(ppo.get("vf_coef", 0.5)),
            max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
            verbose=0,
            tensorboard_log=None,
            device=args.device,
            seed=seed,
            policy_kwargs={"net_arch": params.get("policy_kwargs", {}).get("net_arch", [512, 512, 256])},
        )
        init_seconds = time.perf_counter() - init_t0
        warmup = TimingCallback(
            mode="ppo_warmup",
            repeat=repeat,
            n_envs=args.n_envs,
        )
        model.learn(
            total_timesteps=warmup_timesteps,
            callback=warmup.callback,
            progress_bar=False,
            reset_num_timesteps=True,
        )
        model.update_seconds.clear()
        timing = TimingCallback(
            mode="ppo",
            repeat=repeat,
            n_envs=args.n_envs,
        )
        sync_device(args.device)
        t0 = time.perf_counter()
        model.learn(
            total_timesteps=measured_timesteps,
            callback=timing.callback,
            progress_bar=False,
            reset_num_timesteps=True,
        )
        sync_device(args.device)
        total_seconds = time.perf_counter() - t0
    finally:
        env.close()

    episodes = timing.episodes
    mean_steps, mean_wall = summarize_episodes(episodes)
    update_total = float(sum(model.update_seconds))
    rollout_total = float(sum(timing.rollout_seconds))
    if len(timing.rollout_seconds) != args.ppo_rollouts:
        raise RuntimeError(
            f"Expected {args.ppo_rollouts} measured rollouts, got {len(timing.rollout_seconds)}"
        )
    if len(model.update_seconds) != args.ppo_rollouts:
        raise RuntimeError(
            f"Expected {args.ppo_rollouts} measured PPO updates, got {len(model.update_seconds)}"
        )
    result = RepeatResult(
        mode="ppo",
        repeat=repeat,
        seed=seed,
        n_envs=args.n_envs,
        requested_steps=measured_timesteps,
        measured_steps=int(model.num_timesteps),
        total_seconds=total_seconds,
        reset_seconds=None,
        init_seconds=init_seconds,
        steps_per_second=measured_timesteps / total_seconds if total_seconds else 0.0,
        completed_episodes=len(episodes),
        partial_episodes=sum(1 for steps in timing.episode_steps if steps),
        aggregate_episodes_per_second=len(episodes) / total_seconds if total_seconds else 0.0,
        mean_episode_steps=mean_steps,
        mean_episode_wall_seconds=mean_wall,
        extra={
            "phase": str(phase_config.phase_key),
            "phase_name": phase_config.name,
            "rollouts": args.ppo_rollouts,
            "warmup_timesteps": warmup_timesteps,
            "configured_training_budget": configured_budget,
            "warmup_updates_retained": True,
            "episode_wall_note": "Includes automatic resets and any PPO updates between episode steps; incomplete final episodes are excluded from episode means.",
            "n_steps": n_steps,
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": batch_size,
            "rollout_seconds": timing.rollout_seconds,
            "update_seconds": model.update_seconds,
            "rollout_seconds_total": rollout_total,
            "update_seconds_total": update_total,
            "other_seconds_total": max(0.0, total_seconds - rollout_total - update_total),
            "device": args.device,
            "torch_threads": torch.get_num_threads(),
        },
    )
    write_repeat(output_dir, result, episodes)
    return result, episodes


def json_safe(value: Any) -> Any:
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_repeat(output_dir: Path, result: RepeatResult, episodes: list[dict[str, Any]]) -> None:
    stem = f"{result.mode}_repeat{result.repeat}"
    (output_dir / f"{stem}.json").write_text(
        json.dumps({"result": asdict(result), "episodes": episodes}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if episodes:
        keys = sorted({key for item in episodes for key in item})
        with (output_dir / f"{stem}_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for item in episodes:
                writer.writerow({key: json.dumps(item.get(key)) if key == "info" else item.get(key) for key in keys})


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in DEFAULT_HASH_PATHS:
        path = REPO_ROOT / rel
        if path.is_file():
            hashes[rel] = file_digest(path)
        elif path.is_dir():
            digest = hashlib.sha256()
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                digest.update(str(child.relative_to(REPO_ROOT)).encode("utf-8"))
                digest.update(file_digest(child).encode("utf-8"))
            hashes[rel + "/"] = digest.hexdigest()
    return hashes


def git_metadata() -> dict[str, Any]:
    def run_git(*parts: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *parts], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_short": status,
    }


def package_versions() -> dict[str, str | None]:
    names = [
        "gymnasium",
        "matplotlib",
        "numpy",
        "Pillow",
        "pygame",
        "pymunk",
        "PyYAML",
        "scikit-image",
        "stable_baselines3",
        "tensorboard",
        "torch",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def sysctl_value(name: str) -> str | None:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def write_run_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    config_snapshot = args.config.read_text(encoding="utf-8") if args.config.exists() else None
    metadata = {
        "command": sys.argv,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "repo_root": str(REPO_ROOT),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "os_cpu_count": os.cpu_count(),
            "cpu_brand": sysctl_value("machdep.cpu.brand_string"),
            "memsize_bytes": sysctl_value("hw.memsize"),
        },
        "git": git_metadata(),
        "package_versions": package_versions(),
        "source_hashes": collect_hashes(),
        "config_snapshot": config_snapshot,
        "seed_note": "Repository reset methods do not fully honor Gymnasium seed; benchmark seeds Python, NumPy, and Torch before env construction/reset.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def aggregate_results(results: list[RepeatResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {"repeats": [asdict(item) for item in results], "by_mode": {}}
    for mode in sorted({item.mode for item in results}):
        group = [item for item in results if item.mode == mode]
        fps = [item.steps_per_second for item in group]
        eps = [item.aggregate_episodes_per_second for item in group]
        summary["by_mode"][mode] = {
            "repeats": len(group),
            "steps_per_second_median": statistics.median(fps),
            "steps_per_second_min": min(fps),
            "steps_per_second_max": max(fps),
            "episodes_per_second_median": statistics.median(eps),
            "completed_episodes_total": sum(item.completed_episodes for item in group),
        }
    return summary


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    print(f"\nSaved benchmark output: {output_dir}")
    for mode, item in summary["by_mode"].items():
        print(
            f"{mode}: median {item['steps_per_second_median']:.1f} env-steps/s "
            f"(range {item['steps_per_second_min']:.1f}-{item['steps_per_second_max']:.1f}), "
            f"completed episodes {item['completed_episodes_total']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible performance baseline for AI_2 training."
    )
    parser.add_argument("--mode", choices=["raw", "wrapped", "ppo", "all"], default="all")
    parser.add_argument("--steps", type=int, default=10_000, help="Measured action-loop steps per raw/wrapped repeat")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup action-loop steps excluded from raw/wrapped timing")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "baseline_results")
    parser.add_argument("--ppo-rollouts", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=None, help="Override PPO rollout length")
    parser.add_argument("--batch-size", type=int, default=None, help="Override PPO batch size")
    parser.add_argument("--profile", action="store_true", help="Run cProfile around raw mode only")
    args = parser.parse_args()
    if args.steps <= 0 or args.repeats <= 0 or args.n_envs <= 0:
        parser.error("--steps, --repeats, and --n-envs must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be zero or positive")
    if args.ppo_rollouts <= 0:
        parser.error("--ppo-rollouts must be positive")
    if args.torch_threads <= 0 or (args.n_steps is not None and args.n_steps <= 1):
        parser.error("--torch-threads must be positive and --n-steps must exceed one")
    if args.batch_size is not None and args.batch_size <= 1:
        parser.error("--batch-size must exceed one")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    return args


def main() -> int:
    set_headless_environment()
    args = parse_args()
    require_runtime_dependencies()
    configure_torch(args.device, args.torch_threads)
    output_dir = make_output_dir(args.output)
    write_run_metadata(output_dir, args)

    selected = ["raw", "wrapped", "ppo"] if args.mode == "all" else [args.mode]
    if args.profile and selected != ["raw"]:
        raise ValueError("--profile is supported only with --mode raw")

    results: list[RepeatResult] = []
    for repeat in range(args.repeats):
        for mode in selected:
            if mode == "raw":
                if args.profile:
                    import cProfile

                    profile_path = output_dir / f"raw_repeat{repeat}.prof"
                    profiler = cProfile.Profile()
                    profiler.enable()
                    result, _ = run_raw_repeat(args, repeat, output_dir)
                    profiler.disable()
                    profiler.dump_stats(str(profile_path))
                    result.extra["profile_path"] = str(profile_path)
                else:
                    result, _ = run_raw_repeat(args, repeat, output_dir)
            elif mode == "wrapped":
                result, _ = run_wrapped_repeat(args, repeat, output_dir)
            elif mode == "ppo":
                result, _ = run_ppo_repeat(args, repeat, output_dir)
            else:
                raise AssertionError(mode)
            results.append(result)
            print(
                f"{mode} repeat {repeat}: {result.steps_per_second:.1f} env-steps/s, "
                f"{result.completed_episodes} completed episodes"
            )

    summary = aggregate_results(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print_summary(summary, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
