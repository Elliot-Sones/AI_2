#!/usr/bin/env python3
"""Real-dependency semantic smoke for classic WarehouseBrawl combat wiring."""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import source_path

import argparse
import contextlib
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any


ROOT = PROJECT_ROOT
UTMIST_ROOT = ROOT / "UTMIST-AI2-main"
ENV_DIR = UTMIST_ROOT / "environment"
SCHEMA = "classic-real-semantic-smoke/v1"
DEFAULT_SEEDS = (101, 202, 303)
TIME_LIMIT_STEPS = 90 * 30


class SmokeFailure(RuntimeError):
    pass


@contextlib.contextmanager
def scoped_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError as exc:
        raise SmokeFailure(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc


def git_dirty_files() -> list[dict[str, str]]:
    status = git_output("status", "--porcelain=v1", "--untracked-files=all")
    dirty = []
    for line in status.splitlines():
        if line:
            dirty.append({"status": line[:2], "path": line[3:]})
    return dirty


def dependency_version(dist_name: str, module) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def source_binding(stack) -> dict[str, Any]:
    source_paths = [
        "classic_real_semantic_smoke.py",
        "train.py",
        "config.yaml",
        "UTMIST-AI2-main/environment/environment.py",
        "UTMIST-AI2-main/requirements.txt",
    ]
    dependency_dist_names = {
        "gymnasium": "gymnasium",
        "numpy": "numpy",
        "pygame": "pygame",
        "pymunk": "pymunk",
        "torch": "torch",
        "yaml": "PyYAML",
    }
    dirty_files = git_dirty_files()
    return {
        "schema": "classic-real-semantic-smoke/source-binding/v1",
        "git": {
            "head_commit": git_output("rev-parse", "HEAD"),
            "head_tree": git_output("rev-parse", "HEAD^{tree}"),
            "dirty": bool(dirty_files),
            "dirty_files": dirty_files,
        },
        "source_files": {
            relative: {"sha256": file_sha256(source_path(ROOT, relative))}
            for relative in source_paths
        },
        "dependencies": {
            name: {
                "version": dependency_version(dependency_dist_names[name], module),
                "file": str(Path(module.__file__).resolve()),
            }
            for name, module in stack["dependency_modules"].items()
        },
    }


def validate_output_dir(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise SmokeFailure(f"refusing symlink output path: {output_dir}")
    if output_dir.exists():
        raise SmokeFailure(f"refusing existing output path: {output_dir}")


def create_output_dir(output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(mode=0o755)


def install_paths() -> None:
    for entry in (str(ROOT), str(UTMIST_ROOT)):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path[:0] = [str(ROOT), str(UTMIST_ROOT)]


def import_real_stack():
    if sys.version_info[:2] != (3, 12):
        raise SmokeFailure(f"Python 3.12 required, got {platform.python_version()}")

    install_paths()
    try:
        import gymnasium
        import numpy as np
        import pygame
        import pymunk
        import torch
        import yaml
        import ai2.training.train as train
        from environment.environment import AttackState, CameraResolution, WarehouseBrawl
    except Exception as exc:
        raise SmokeFailure(f"missing real dependency stack: {exc}") from exc

    modules = {
        "gymnasium": gymnasium,
        "numpy": np,
        "pygame": pygame,
        "pymunk": pymunk,
        "torch": torch,
        "yaml": yaml,
    }
    stubbed = []
    for name, module in modules.items():
        module_file = getattr(module, "__file__", None)
        if module.__class__.__name__ == "DummyModule" or not isinstance(module_file, (str, bytes, os.PathLike)):
            stubbed.append(name)
    if stubbed:
        raise SmokeFailure(f"refusing stubbed dependency modules: {stubbed}")
    if WarehouseBrawl.__name__ != "WarehouseBrawl" or WarehouseBrawl.__module__ != "environment.environment":
        raise SmokeFailure("WarehouseBrawl did not resolve to the real environment.environment class")

    return {
        "AttackState": AttackState,
        "CameraResolution": CameraResolution,
        "WarehouseBrawl": WarehouseBrawl,
        "np": np,
        "pygame": pygame,
        "pymunk": pymunk,
        "torch": torch,
        "train": train,
        "yaml": yaml,
        "dependency_modules": modules,
    }


def load_real_config(yaml_module) -> dict[str, Any]:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        params = yaml_module.safe_load(handle)
    settings = params.get("environment_settings", {})
    if settings.get("time_limit_seconds") != 90 or settings.get("max_timesteps") != TIME_LIMIT_STEPS:
        raise SmokeFailure("config.yaml must declare combat environment_settings as 90 seconds / 2700 steps")
    return params


def make_combat_params(base: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    phases = copy.deepcopy(base["phases"])
    for key in (1, "1"):
        if key in phases:
            phases[key]["opponents"] = {"constant_agent": 1.0}
    return {
        "folders": {
            "parent_dir": str(output_dir / "models"),
            "model_name": "classic_real_semantic_smoke",
        },
        "self_play_settings": dict(base["self_play_settings"]),
        "environment_settings": {
            "max_timesteps": int(base["environment_settings"]["max_timesteps"]),
            "time_limit_seconds": int(base["environment_settings"]["time_limit_seconds"]),
            "resolution": "LOW",
            "n_envs": 1,
        },
        "opponent_history": {"enabled": False},
        "phases": phases,
    }


def phase_config(train, params: dict[str, Any], phase_key: int = 1):
    return train.PhaseConfig(params, phase_key)


def unwrap_self_play(env):
    cursor = env
    while cursor is not None:
        if hasattr(cursor, "raw_env") and hasattr(cursor, "reward_manager"):
            return cursor
        cursor = getattr(cursor, "env", None)
    raise SmokeFailure("SelfPlayWarehouseBrawl not found in wrapper chain")


def unwrap_reward_reset(env):
    cursor = env
    while cursor is not None:
        if cursor.__class__.__name__ == "RewardEpisodeResetWrapper":
            return cursor
        cursor = getattr(cursor, "env", None)
    raise SmokeFailure("RewardEpisodeResetWrapper not found in wrapper chain")


def stabilize_no_combat_state(raw_env) -> None:
    """Production state setup: place players on separate grounded spawn islands."""
    placements = {0: (-4.5, 0.0), 1: (4.5, 0.0)}
    for agent, position in placements.items():
        player = raw_env.players[agent]
        player.body.position = position
        player.body.velocity = (0.0, 0.0)
        player.damage = 0.0
        player.damage_taken_total = 0.0
        player.damage_done = 0.0
        player.stocks = 99
    raw_env.rewards = {0: 0.0, 1: 0.0}
    raw_env.terminated = False


def observation_hash(np_module, obs: Any) -> str:
    array = np_module.asarray(obs, dtype=np_module.float32)
    return sha256_bytes(array.tobytes())


def info_digest(info: dict[str, Any]) -> dict[str, Any]:
    digest = {}
    for key, value in sorted(info.items()):
        if key == "opponent_action":
            digest[key] = sha256_bytes(bytes(memoryview(value.astype("float32"))))
        else:
            digest[key] = value
    return digest


def run_trace_once(stack, params: dict[str, Any], seed: int) -> dict[str, Any]:
    np_module = stack["np"]
    train = stack["train"]
    phase = phase_config(train, params, 1)
    events = []
    with scoped_cwd(ENV_DIR):
        env = train.make_combat_env(phase, params, device="cpu")
        try:
            self_play = unwrap_self_play(env)
            raw = self_play.raw_env
            obs, reset_info = env.reset(seed=seed)
            stabilize_no_combat_state(raw)
            action = np_module.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            configured = {
                "seed": seed,
                "fps": int(raw.fps),
                "configured_steps": int(raw.max_timesteps),
                "wrapper_chain": wrapper_chain(env),
                "reset_info": reset_info,
                "scenario": "after real reset(seed), players are placed on separate grounded spawn islands with 99 stocks and zero velocity; zero learner action plus the production constant_agent opponent is stepped until the configured time limit",
            }
            if configured["fps"] != 30 or configured["configured_steps"] != TIME_LIMIT_STEPS:
                raise SmokeFailure(f"unexpected time limit configuration: {configured}")
            events.append({"kind": "reset", **configured, "obs_sha256": observation_hash(np_module, obs)})

            truncation_step = None
            sample_steps = {1, 2, 30, 300, 900, 1800, 2699, 2700}
            for _ in range(TIME_LIMIT_STEPS):
                obs, reward, terminated, truncated, info = env.step(action)
                step = int(raw.steps)
                if terminated and not truncated:
                    raise SmokeFailure(f"natural termination before timeout for seed {seed} at step {step}")
                if step in sample_steps or truncated:
                    events.append(
                        {
                            "kind": "step",
                            "seed": seed,
                            "step": step,
                            "obs_sha256": observation_hash(np_module, obs),
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "info_digest": info_digest(info),
                            "raw_rewards": {str(k): float(v) for k, v in raw.rewards.items()},
                            "raw_terminated": bool(raw.terminated),
                        }
                    )
                if truncated:
                    truncation_step = step
                    break
            if truncation_step != TIME_LIMIT_STEPS:
                raise SmokeFailure(f"seed {seed} truncated at {truncation_step}, expected {TIME_LIMIT_STEPS}")
            _, post_reset_info = env.reset(seed=seed)
            events.append(
                {
                    "kind": "post_timeout_reset",
                    "seed": seed,
                    "raw_steps": int(raw.steps),
                    "post_reset_info": post_reset_info,
                    "obs_sha256": observation_hash(np_module, raw.observe(0)),
                }
            )
            trace_hash = sha256_json(events)
            return {"seed": seed, "trace_hash": trace_hash, "events": events}
        finally:
            env.close()


def wrapper_chain(env) -> list[str]:
    names = []
    cursor = env
    while cursor is not None:
        names.append(cursor.__class__.__name__)
        cursor = getattr(cursor, "env", None)
    return names


def run_seed_traces(stack, params: dict[str, Any], seeds: tuple[int, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_summaries = []
    all_events = []
    for seed in seeds:
        first = run_trace_once(stack, params, seed)
        second = run_trace_once(stack, params, seed)
        if first["trace_hash"] != second["trace_hash"]:
            raise SmokeFailure(f"seed {seed} trace was not reproducible")
        trace_summaries.append(
            {
                "seed": seed,
                "trace_hash": first["trace_hash"],
                "replay_trace_hash": second["trace_hash"],
                "sampled_events": len(first["events"]),
            }
        )
        all_events.extend(first["events"])
    return trace_summaries, all_events


def reward_semantics(stack, params: dict[str, Any]) -> dict[str, Any]:
    train = stack["train"]
    AttackState = stack["AttackState"]
    with scoped_cwd(ENV_DIR):
        phase_1 = phase_config(train, params, 1)
        env_phase_1 = train.make_combat_env(phase_1, params, device="cpu")
        try:
            phase_1_self_play = unwrap_self_play(env_phase_1)
            phase_1_raw = phase_1_self_play.raw_env
            env_phase_1.reset(seed=8181)
            phase_1_raw.knockout_signal.emit(agent="player")
            phase_1_learner_ko = float(phase_1_self_play.reward_manager.process(phase_1_raw, 1 / 30.0))
            phase_1_self_play.reward_manager.reset()
            phase_1_raw.knockout_signal.emit(agent="opponent")
            phase_1_opponent_ko = float(phase_1_self_play.reward_manager.process(phase_1_raw, 1 / 30.0))
        finally:
            env_phase_1.close()

        phase = phase_config(train, params, 3)
        env = train.make_combat_env(phase, params, device="cpu")
        try:
            self_play = unwrap_self_play(env)
            reward_reset = unwrap_reward_reset(env)
            raw = self_play.raw_env
            env.reset(seed=9191)
            stabilize_no_combat_state(raw)
            manager = reward_reset.reward_manager
            signal_manager = self_play.reward_manager

            raw.win_signal.emit(agent="player")
            learner_win = float(signal_manager.process(raw, 1 / 30.0))
            signal_manager.reset()

            raw.knockout_signal.emit(agent="player")
            learner_ko = float(signal_manager.process(raw, 1 / 30.0))
            signal_manager.reset()

            raw.knockout_signal.emit(agent="opponent")
            opponent_ko = float(signal_manager.process(raw, 1 / 30.0))
            signal_manager.reset()

            player = raw.players[0]
            player.state = player.states["attack"]
            if not isinstance(player.state, AttackState):
                raise SmokeFailure("player.states['attack'] is not an actual AttackState")
            aggression_reward = float(manager.compute_reward(raw))
            player.state = player.states["standing"]
            manager.reset_episode()
            non_attack_reward = float(manager.compute_reward(raw))
            expected_attack_bonus = float(phase.rewards["aggression"] * 0.1)

            before_reset = {
                "episode_damage_dealt": manager.episode_damage_dealt,
                "episode_damage_received": manager.episode_damage_received,
            }
            raw.players[0].damage_done = 7.0
            damage_reward = float(manager.compute_reward(raw))
            env.reset(seed=9292)
            after_reset_process = float(manager.compute_reward(unwrap_self_play(env).raw_env))
            after_reset = {
                "episode_damage_dealt": manager.episode_damage_dealt,
                "episode_damage_received": manager.episode_damage_received,
            }

            env2 = train.make_combat_env(phase, params, device="cpu")
            try:
                env2.reset(seed=9393)
                manager2 = unwrap_reward_reset(env2).reward_manager
                raw2 = unwrap_self_play(env2).raw_env
                raw2.players[0].damage_done = 5.0
                isolated_reward = float(manager2.compute_reward(raw2))
                env2.reset(seed=9494)
                isolated_after_reset = float(manager2.compute_reward(unwrap_self_play(env2).raw_env))
            finally:
                env2.close()

            leakage_absent = (
                after_reset_process == 0.0
                and isolated_after_reset == 0.0
                and after_reset["episode_damage_dealt"] == 0.0
                and manager is not manager2
            )
            return {
                "learner_win_reward": learner_win,
                "learner_knockout_reward": learner_ko,
                "opponent_knockout_reward": opponent_ko,
                "attack_state_aggression_reward": aggression_reward,
                "standing_aggression_reward": non_attack_reward,
                "phase_1": {
                    "phase_id": 1,
                    "configured_weights": dict(phase_1.rewards),
                    "learner_knockout_reward": phase_1_learner_ko,
                    "opponent_knockout_reward": phase_1_opponent_ko,
                },
                "aggression": {
                    "phase_id": 3,
                    "configured_weights": dict(phase.rewards),
                    "expected_attack_bonus": expected_attack_bonus,
                    "attack_state_reward": aggression_reward,
                    "standing_reward": non_attack_reward,
                },
                "damage_reward_before_reset": damage_reward,
                "episode_state_before_damage": before_reset,
                "episode_state_after_reset": after_reset,
                "second_env_damage_reward": isolated_reward,
                "second_env_after_reset_reward": isolated_after_reset,
                "reward_leakage_absent": bool(leakage_absent),
                "attack_state_class": player.states["attack"].__class__.__name__,
            }
        finally:
            env.close()


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for index, payload in enumerate(events):
            record = {
                "index": index,
                "schema": SCHEMA,
                "payload": payload,
                "payload_sha256": sha256_json(payload),
            }
            handle.write(canonical_json(record) + "\n")


def chmod_immutable_enough(output_dir: Path) -> None:
    for child in output_dir.iterdir():
        child.chmod(0o444)
    output_dir.chmod(0o555)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        if args.output_dir.is_symlink():
            raise SmokeFailure(f"refusing symlink output path: {args.output_dir}")
        output_dir = args.output_dir.resolve(strict=False)
        validate_output_dir(output_dir)
        stack = import_real_stack()
        binding = source_binding(stack)
        params = make_combat_params(load_real_config(stack["yaml"]), output_dir)
        seeds = tuple(args.seeds)
        if len(seeds) != 3:
            raise SmokeFailure("exactly 3 explicit seeds are required")
        seed_traces, events = run_seed_traces(stack, params, seeds)
        rewards = reward_semantics(stack, params)
        if not (
            rewards["learner_win_reward"] > 0
            and rewards["learner_knockout_reward"] < 0
            and rewards["opponent_knockout_reward"] > 0
            and rewards["phase_1"]["learner_knockout_reward"] == -rewards["phase_1"]["configured_weights"]["knockout"]
            and rewards["phase_1"]["opponent_knockout_reward"] == rewards["phase_1"]["configured_weights"]["knockout"]
            and rewards["aggression"]["attack_state_reward"] == rewards["aggression"]["expected_attack_bonus"]
            and rewards["aggression"]["standing_reward"] == 0.0
            and rewards["reward_leakage_absent"]
        ):
            raise SmokeFailure(f"reward semantic checks failed: {rewards}")

        create_output_dir(output_dir)
        events_path = output_dir / "events.jsonl"
        write_jsonl(events_path, events)
        receipt = {
            "schema": SCHEMA,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.platform(),
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            "seeds": list(seeds),
            "real_dependency_checks": {
                "WarehouseBrawl": True,
                "WarehouseBrawl_module": "environment.environment",
                "stubbed_dependency_modules": [],
                "dependency_files": {
                    name: str(Path(module.__file__).resolve())
                    for name, module in stack["dependency_modules"].items()
                },
            },
            "time_limit": {
                "fps": 30,
                "configured_steps": TIME_LIMIT_STEPS,
                "configured_seconds": 90,
                "truncation_step": TIME_LIMIT_STEPS,
                "reset_steps_after_truncation": 0,
                "scenario": "real reset(seed), production constant_agent opponent, and explicit production state setup to prevent combat/fall termination; no environment reimplementation",
            },
            "seed_traces": seed_traces,
            "reward_semantics": rewards,
            "source_binding": binding,
        }
        receipt["receipt_payload_sha256"] = sha256_json(receipt)
        receipt_path = output_dir / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt["events_jsonl_sha256"] = sha256_bytes(events_path.read_bytes())
        receipt["receipt_payload_sha256"] = sha256_json({k: v for k, v in receipt.items() if k != "receipt_payload_sha256"})
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        chmod_immutable_enough(output_dir)
        print(str(receipt_path))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
