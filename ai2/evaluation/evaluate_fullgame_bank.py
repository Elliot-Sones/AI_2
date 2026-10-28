#!/usr/bin/env python3
"""Evidence-grade full-game evaluation bank for AI_2 policies.

This file deliberately does not modify the legacy eval scripts.  It provides a
small, fail-closed evaluator that can be exercised with fakes in unit tests and
with the real WarehouseBrawl environment from the command line.
"""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "fullgame-eval-bank-v1"
FRAME_STACK = 4
OBS_WIDTH = 64
STACKED_OBS_WIDTH = OBS_WIDTH * FRAME_STACK
ACTION_WIDTH = 10
ACTION_KEYS = ("w", "a", "s", "d", "space", "h", "l", "j", "k", "g")
ATTACK_INDICES = (7, 8)
SUPPORTED_ACTION_HEADS = ("gaussian", "bernoulli_10")
VIDEO_FPS = 30


class VideoUnavailableError(RuntimeError):
    """Raised when an evaluation requested video but no real nonempty capture was produced."""


@dataclasses.dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    seed: int
    opponent: str
    max_ticks: int
    player_slot: int = 0
    mirroring: str = "player0_left_reset"
    placement_profile: str = "reset_only"

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class FrameStack:
    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []

    def reset(self, obs: Any) -> np.ndarray:
        frame = _validate_observation(obs, "reset observation")
        zero = np.zeros_like(frame, dtype=np.float32)
        self._frames = [zero.copy() for _ in range(FRAME_STACK - 1)] + [frame.copy()]
        return self.get()

    def step(self, obs: Any) -> np.ndarray:
        frame = _validate_observation(obs, "step observation")
        self._frames.pop(0)
        self._frames.append(frame.copy())
        return self.get()

    def get(self) -> np.ndarray:
        stacked = np.concatenate(self._frames, axis=0).astype(np.float32, copy=False)
        if stacked.shape != (STACKED_OBS_WIDTH,):
            raise ValueError(f"stacked observation must have shape {(STACKED_OBS_WIDTH,)}, got {stacked.shape}")
        if not np.isfinite(stacked).all():
            raise ValueError("stacked observation contains nonfinite values")
        return stacked


class SB3PolicyAdapter:
    def __init__(self, model: Any, *, action_head: str = "gaussian", source_path: str | Path | None = None) -> None:
        self.model = model
        self.action_head = _resolve_action_head(action_head, model)
        self.source_path = Path(source_path) if source_path is not None else None
        self.expected_observation_width = _resolve_expected_observation_width(model)
        _require_supported_observation_width(self.expected_observation_width, "SB3 policy")

    @classmethod
    def load(cls, path: str | Path, *, device: str, action_head: str = "auto") -> "SB3PolicyAdapter":
        from stable_baselines3 import PPO

        model = PPO.load(str(path), device=device)
        return cls(model, action_head=action_head, source_path=path)

    def predict(self, obs: np.ndarray, *, deterministic: bool) -> np.ndarray:
        _validate_stacked_observation(obs, self.expected_observation_width)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return _validate_action(action, "policy action")


class ActorPolicyAdapter:
    def __init__(
        self,
        actor: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        action_head: str = "auto",
        source_path: str | Path | None = None,
    ) -> None:
        self.actor = actor
        self.metadata = dict(metadata or {})
        self.action_head = _resolve_action_head(action_head, actor, self.metadata)
        self.source_path = Path(source_path) if source_path is not None else None
        self.expected_observation_width = _resolve_expected_observation_width(actor, self.metadata)
        _require_supported_observation_width(self.expected_observation_width, "exported actor")

    @classmethod
    def load(cls, path: str | Path, *, device: str, action_head: str = "auto") -> "ActorPolicyAdapter":
        from ai2.training.curriculum_ppo import load_actor

        expected = None if action_head == "auto" else action_head
        actor, metadata = load_actor(path, device=device, expected_action_head=expected)
        return cls(actor, metadata=metadata, action_head=action_head, source_path=path)

    def predict(self, obs: np.ndarray, *, deterministic: bool) -> np.ndarray:
        _validate_stacked_observation(obs, self.expected_observation_width)
        if hasattr(self.actor, "predict"):
            action, _ = self.actor.predict(obs, deterministic=deterministic)
            return _validate_action(action, "actor action")
        raise TypeError("actor adapter requires an actor with predict(obs, deterministic=...)")


def build_default_bank() -> list[EvaluationCase]:
    opponents = [
        "Constant",
        "Random",
        "Based",
        "Clockwork",
        "NoisyBased",
        "EdgeGuard",
        "WeaponHunter",
        "RecoveryPunish",
    ]
    cases: list[EvaluationCase] = []
    for idx, name in enumerate(opponents, start=1):
        cases.append(
            EvaluationCase(
                case_id=f"reset_{idx:02d}_{name.lower()}_p0",
                seed=2026091500 + idx * 2,
                opponent=name,
                max_ticks=30 * 90,
                player_slot=0,
                mirroring="player0_reset_distribution",
                placement_profile="reset_only",
            )
        )
        cases.append(
            EvaluationCase(
                case_id=f"reset_{idx:02d}_{name.lower()}_p1_mirror",
                seed=2026091500 + idx * 2 + 1,
                opponent=name,
                max_ticks=30 * 90,
                player_slot=1,
                mirroring="mirror_player1_reset_distribution",
                placement_profile="reset_only",
            )
        )
    return cases


def run_evaluation(
    *,
    policy: Any,
    cases: Sequence[EvaluationCase],
    modes: Sequence[str],
    output_dir: str | Path,
    env_factory: Callable[[EvaluationCase], Any] | None = None,
    opponent_factory: Callable[[str], tuple[Any, str]] | None = None,
    overwrite: bool = False,
    episode_max: int | None = None,
    video_subset: Sequence[str] = (),
    video_capture: Callable[[Sequence[np.ndarray], Path], None] | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    _validate_case_bank(cases)
    _validate_modes(modes)
    _validate_video_subset(video_subset, cases, modes)
    checkpoint_record = _checkpoint_record(
        checkpoint_path if checkpoint_path is not None else getattr(policy, "source_path", None),
        required=checkpoint_path is not None,
    )
    output_path = Path(output_dir)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    _prepare_output_paths(output_path, temp_path, overwrite=overwrite)
    env_factory = env_factory or create_real_env
    opponent_factory = opponent_factory or create_opponent
    video_keys = set(video_subset)
    manifest_cases = (
        [dataclasses.replace(case, max_ticks=episode_max) for case in cases]
        if episode_max is not None
        else list(cases)
    )

    episodes: list[dict[str, Any]] = []
    summary = _empty_summary(len(cases), tuple(modes))

    try:
        temp_path.mkdir(parents=True, exist_ok=False)
        trace_path = temp_path / "trace.jsonl"
        episodes_path = temp_path / "episodes.jsonl"
        with trace_path.open("w", encoding="utf-8") as trace_file, episodes_path.open("w", encoding="utf-8") as episode_file:
            for case in cases:
                for mode in modes:
                    episode = _run_one_episode(
                        policy=policy,
                        case=case,
                        mode=mode,
                        env_factory=env_factory,
                        opponent_factory=opponent_factory,
                        trace_file=trace_file,
                        max_ticks=episode_max if episode_max is not None else case.max_ticks,
                        video_enabled=f"{case.case_id}:{mode}" in video_keys,
                        video_dir=temp_path / "videos",
                        video_capture=video_capture,
                    )
                    episodes.append(episode)
                    episode_file.write(_json_line(episode))
                    _accumulate_summary(summary, episode)

        summary["win_rate"] = summary["wins"] / summary["episodes"] if summary["episodes"] else 0.0
        summary["avg_damage_done"] = summary["damage_done"] / summary["episodes"] if summary["episodes"] else 0.0
        (temp_path / "summary.json").write_text(_json_block(summary), encoding="utf-8")

        manifest = _build_manifest(
            cases=manifest_cases,
            modes=modes,
            checkpoint_record=checkpoint_record,
            output_dir=temp_path,
        )
        (temp_path / "manifest.json").write_text(_json_block(manifest), encoding="utf-8")
        manifest["artifact_sha256"] = _hash_artifacts(temp_path)
        (temp_path / "manifest.json").write_text(_json_block(manifest), encoding="utf-8")

        temp_path.replace(output_path)
        return {"summary": summary, "episodes": episodes, "manifest": manifest}
    except Exception:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise


def create_real_env(case: EvaluationCase) -> Any:
    _ensure_game_import_path()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env_cwd = _game_environment_dir()
    with _pushd(env_cwd):
        from environment.environment import CameraResolution, WarehouseBrawl

        env = WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=True)
    env.max_timesteps = case.max_ticks
    return _ScopedCwdEnv(env, env_cwd)


def create_opponent(name: str) -> tuple[Any, str]:
    key = name.replace("-", "_").replace(" ", "_").lower()
    registry_names = {
        "constant": "ConstantAgent",
        "random": "RandomAgent",
        "based": "BasedAgent",
        "clockwork": "ClockworkAgent",
        "noisybased": "NoisyBasedAgent",
        "noisy_based": "NoisyBasedAgent",
        "edgeguard": "EdgeGuardAgent",
        "edge_guard": "EdgeGuardAgent",
        "weaponhunter": "WeaponHunterAgent",
        "weapon_hunter": "WeaponHunterAgent",
        "recoverypunish": "RecoveryPunishAgent",
        "recovery_punish": "RecoveryPunishAgent",
    }
    if key not in registry_names:
        raise ValueError(f"unsupported opponent {name!r}; supported={sorted(registry_names)}")
    _ensure_game_import_path()
    with _pushd(_game_environment_dir()):
        from environment.agent import (
            BasedAgent,
            ClockworkAgent,
            ConstantAgent,
            EdgeGuardAgent,
            NoisyBasedAgent,
            RandomAgent,
            RecoveryPunishAgent,
            WeaponHunterAgent,
        )

    registry = {
        "constant": ConstantAgent,
        "random": RandomAgent,
        "based": BasedAgent,
        "clockwork": ClockworkAgent,
        "noisybased": NoisyBasedAgent,
        "noisy_based": NoisyBasedAgent,
        "edgeguard": EdgeGuardAgent,
        "edge_guard": EdgeGuardAgent,
        "weaponhunter": WeaponHunterAgent,
        "weapon_hunter": WeaponHunterAgent,
        "recoverypunish": RecoveryPunishAgent,
        "recovery_punish": RecoveryPunishAgent,
    }
    opponent = registry[key]()
    return opponent, type(opponent).__name__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a policy on the fixed full-game bank.")
    parser.add_argument("--policy", required=True, help="Path to SB3 zip or exported actor checkpoint")
    parser.add_argument("--policy-kind", choices=("sb3", "actor"), required=True)
    parser.add_argument("--action-head", choices=("auto", "gaussian", "bernoulli_10"), default="auto")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--modes", default="deterministic,sampled", help="Comma-separated: deterministic,sampled")
    parser.add_argument("--cases", default="all", help="Comma-separated case IDs or 'all'")
    parser.add_argument("--episode-max", type=int, default=None)
    parser.add_argument("--video-subset", default="", help="Comma-separated case:mode keys to record")
    args = parser.parse_args(argv)

    if args.policy_kind == "sb3":
        policy = SB3PolicyAdapter.load(args.policy, device=args.device, action_head=args.action_head)
    else:
        policy = ActorPolicyAdapter.load(args.policy, device=args.device, action_head=args.action_head)

    bank = build_default_bank()
    if args.cases != "all":
        wanted = set(_split_csv(args.cases))
        bank = [case for case in bank if case.case_id in wanted]
        missing = wanted - {case.case_id for case in bank}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")
    modes = tuple(_split_csv(args.modes))
    video_subset = tuple(_split_csv(args.video_subset)) if args.video_subset else ()
    result = run_evaluation(
        policy=policy,
        cases=bank,
        modes=modes,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        episode_max=args.episode_max,
        video_subset=video_subset,
        checkpoint_path=args.policy,
    )
    print(_json_block(result["summary"]))
    return 0


def _run_one_episode(
    *,
    policy: Any,
    case: EvaluationCase,
    mode: str,
    env_factory: Callable[[EvaluationCase], Any],
    opponent_factory: Callable[[str], tuple[Any, str]],
    trace_file: Any,
    max_ticks: int,
    video_enabled: bool,
    video_dir: Path,
    video_capture: Callable[[Sequence[np.ndarray], Path], None] | None,
) -> dict[str, Any]:
    deterministic = mode == "deterministic"
    video_recorder: Any | None = None
    video_path: Path | None = None
    video_closed = False
    with _seeded_global_rng(case.seed):
        env = env_factory(dataclasses.replace(case, max_ticks=max_ticks))
        try:
            opponent, realized_opponent = opponent_factory(case.opponent)
            if hasattr(env, "max_timesteps"):
                env.max_timesteps = max_ticks
            observations, _ = env.reset(seed=case.seed)
            learner_obs = _obs_for_slot(observations, case.player_slot)
            opponent_obs = _obs_for_slot(observations, 1 - case.player_slot)
            if hasattr(opponent, "get_env_info"):
                opponent.get_env_info(getattr(env, "unwrapped", env))
            stack = FrameStack()
            stacked = stack.reset(learner_obs)
            episode = _new_episode_record(case, mode, realized_opponent, max_ticks)
            last_stats = (_stats(env, case.player_slot), _stats(env, 1 - case.player_slot))
            start_distance = _distance_between_players(env)
            if video_enabled:
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_dir / f"{case.case_id}_{mode}.mp4"
                video_recorder = _open_video_recorder(video_path, video_capture)
                video_recorder.append(_render_video_frame(env))

            for tick in range(max_ticks):
                before_players = _players_snapshot(env)
                raw_action = _validate_action(policy.predict(stacked, deterministic=deterministic), "policy action")
                decoded_buttons = decode_executed_buttons(raw_action, getattr(policy, "action_head", "gaussian"))
                opponent_action = _validate_action(opponent.predict(opponent_obs), "opponent action")
                observations, _rewards, terminated, truncated, _info = env.step(
                    {case.player_slot: raw_action, 1 - case.player_slot: opponent_action}
                )
                learner_obs = _obs_for_slot(observations, case.player_slot)
                opponent_obs = _obs_for_slot(observations, 1 - case.player_slot)
                stacked = stack.step(learner_obs)
                current_stats = (_stats(env, case.player_slot), _stats(env, 1 - case.player_slot))
                after_players = _players_snapshot(env)
                trace = _trace_row(
                    case=case,
                    mode=mode,
                    tick=tick,
                    realized_opponent=realized_opponent,
                    raw_action=raw_action,
                    decoded_buttons=decoded_buttons,
                    before_players=before_players,
                    after_players=after_players,
                    last_stats=last_stats,
                    current_stats=current_stats,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                trace_file.write(_json_line(trace))
                _fold_trace_into_episode(episode, trace)
                last_stats = current_stats
                if video_enabled:
                    assert video_recorder is not None
                    video_recorder.append(_render_video_frame(env))
                if terminated or truncated:
                    episode["terminal_cause"] = _terminal_cause(trace, current_stats)
                    break
            else:
                episode["terminal_cause"] = "episode_max"

            final_stats = (_stats(env, case.player_slot), _stats(env, 1 - case.player_slot))
            episode.update(_final_episode_fields(env, final_stats, start_distance))
            episode["diagnoses"] = _diagnose_episode(episode)
            if video_enabled:
                assert video_recorder is not None and video_path is not None
                video_recorder.close()
                video_closed = True
                _verify_video_artifact(video_path)
                episode["video_path"] = str(Path("videos") / video_path.name)
            return episode
        finally:
            if video_recorder is not None and not video_closed:
                abort = getattr(video_recorder, "abort", None)
                if callable(abort):
                    abort()
            if hasattr(env, "close"):
                env.close()


def decode_executed_buttons(raw_action: Any, action_head: str) -> list[int]:
    action = _validate_action(raw_action, "raw action")
    head = _resolve_action_head(action_head)
    if head == "gaussian":
        return [int(v > 0.5) for v in action.tolist()]
    if not np.isin(action, [0.0, 1.0]).all():
        raise ValueError("bernoulli_10 action must be exact binary buttons")
    return [int(v) for v in action.tolist()]


def _validate_observation(obs: Any, label: str) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape != (OBS_WIDTH,):
        raise ValueError(f"{label} must have shape {(OBS_WIDTH,)}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains nonfinite values")
    return arr


def _validate_stacked_observation(obs: Any, expected_width: int = STACKED_OBS_WIDTH) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape != (expected_width,):
        raise ValueError(f"policy input must have shape {(expected_width,)}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("policy input contains nonfinite values")
    return arr


def _validate_action(action: Any, label: str) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape != (ACTION_WIDTH,):
        raise ValueError(f"{label} must have shape {(ACTION_WIDTH,)}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains nonfinite values")
    return arr


@contextlib.contextmanager
def _seeded_global_rng(seed: int) -> Iterator[None]:
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch = _try_import_torch()
    torch_state = torch.random.get_rng_state().clone() if torch is not None else None
    cuda_states = None
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if cuda_states is not None:
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None and torch_state is not None:
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


def _resolve_action_head(action_head: str, obj: Any | None = None, metadata: Mapping[str, Any] | None = None) -> str:
    if action_head == "auto":
        if metadata and metadata.get("action_head") in SUPPORTED_ACTION_HEADS:
            return str(metadata["action_head"])
        for candidate in (obj, getattr(obj, "policy", None)):
            found = getattr(candidate, "action_head", None)
            if found in SUPPORTED_ACTION_HEADS:
                return str(found)
        return "gaussian"
    if action_head not in SUPPORTED_ACTION_HEADS:
        raise ValueError(f"unsupported action head {action_head!r}")
    return action_head


def _resolve_expected_observation_width(obj: Any | None = None, metadata: Mapping[str, Any] | None = None) -> int:
    if metadata:
        for key in ("observation_size", "expected_observation_width"):
            if key in metadata:
                return int(metadata[key])
    for candidate in (obj, getattr(obj, "policy", None)):
        width = getattr(candidate, "expected_observation_width", None)
        if width is not None:
            return int(width)
        space = getattr(candidate, "observation_space", None)
        shape = getattr(space, "shape", None)
        if shape:
            return int(np.prod(shape))
    return STACKED_OBS_WIDTH


def _require_supported_observation_width(width: int, label: str) -> None:
    if width != STACKED_OBS_WIDTH:
        raise ValueError(
            f"{label} observation width {width} is not supported by this evaluator; "
            f"expected {STACKED_OBS_WIDTH}. Classic OpponentHistoryWrapper-expanded "
            "SB3 checkpoints must be evaluated by a compatible wrapper-specific evaluator."
        )


def _validate_case_bank(cases: Sequence[EvaluationCase]) -> None:
    if not cases:
        raise ValueError("at least one evaluation case is required")
    ids = [case.case_id for case in cases]
    seeds = [case.seed for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case IDs must be unique")
    if len(seeds) != len(set(seeds)):
        raise ValueError("evaluation case seeds must be unique")
    for case in cases:
        if case.player_slot not in (0, 1):
            raise ValueError("player_slot must be 0 or 1")
        if case.max_ticks <= 0:
            raise ValueError("max_ticks must be positive")
        if case.placement_profile != "reset_only":
            raise ValueError("this evaluator version only supports reset_only placement profiles")


def _validate_modes(modes: Sequence[str]) -> None:
    if not modes:
        raise ValueError("at least one mode is required")
    bad = set(modes) - {"deterministic", "sampled"}
    if bad:
        raise ValueError(f"unsupported evaluation modes: {sorted(bad)}")


def _validate_video_subset(video_subset: Sequence[str], cases: Sequence[EvaluationCase], modes: Sequence[str]) -> None:
    if len(video_subset) != len(set(video_subset)):
        raise ValueError("video_subset keys must be unique")
    valid = {f"{case.case_id}:{mode}" for case in cases for mode in modes}
    unknown = set(video_subset) - valid
    if unknown:
        raise ValueError(f"unknown video_subset keys: {sorted(unknown)}")


def _prepare_output_paths(output_path: Path, temp_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is nonempty: {output_path}")
        shutil.rmtree(output_path)
    elif output_path.exists():
        output_path.rmdir()
    if temp_path.exists():
        shutil.rmtree(temp_path)


def _empty_summary(case_count: int, modes: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_count": case_count,
        "modes": list(modes),
        "episodes": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "damage_done": 0.0,
        "damage_taken": 0.0,
        "falls": 0,
        "timeouts": 0,
        "attack_requested": 0,
        "attack_starts": 0,
        "hits": 0,
    }


def _new_episode_record(case: EvaluationCase, mode: str, realized_opponent: str, max_ticks: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        **case.to_json(),
        "mode": mode,
        "requested_opponent": case.opponent,
        "realized_opponent": realized_opponent,
        "max_ticks": max_ticks,
        "ticks": 0,
        "terminal_cause": "unknown",
        "attack_requested": 0,
        "attack_starts": 0,
        "hits": 0,
        "damage_done": 0.0,
        "damage_taken": 0.0,
        "self_stock_losses": 0,
        "opponent_stock_losses": 0,
        "video_path": None,
    }


def _trace_row(
    *,
    case: EvaluationCase,
    mode: str,
    tick: int,
    realized_opponent: str,
    raw_action: np.ndarray,
    decoded_buttons: list[int],
    before_players: list[dict[str, Any]],
    after_players: list[dict[str, Any]],
    last_stats: tuple[Mapping[str, Any], Mapping[str, Any]],
    current_stats: tuple[Mapping[str, Any], Mapping[str, Any]],
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    learner_before, opponent_before = before_players[case.player_slot], before_players[1 - case.player_slot]
    learner_after, opponent_after = after_players[case.player_slot], after_players[1 - case.player_slot]
    learner_stats, opponent_stats = current_stats
    last_learner_stats, last_opponent_stats = last_stats
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "seed": case.seed,
        "opponent": case.opponent,
        "realized_opponent": realized_opponent,
        "mode": mode,
        "tick": tick,
        "learner": learner_after,
        "opponent_state": opponent_after,
        "executed_action_values": [float(v) for v in raw_action.tolist()],
        "decoded_buttons": decoded_buttons,
        "decoded_button_names": [name for name, pressed in zip(ACTION_KEYS, decoded_buttons) if pressed],
        "attack_press": bool(decoded_buttons[7] or decoded_buttons[8]),
        "attack_state_start": (not _is_attack_state(learner_before)) and _is_attack_state(learner_after),
        "damage_delta": float(learner_stats["damage_done"] - last_learner_stats["damage_done"]),
        "damage_taken_delta": float(learner_stats["damage_taken"] - last_learner_stats["damage_taken"]),
        "self_stock_delta": int(learner_stats["lives_left"] - last_learner_stats["lives_left"]),
        "opponent_stock_delta": int(opponent_stats["lives_left"] - last_opponent_stats["lives_left"]),
        "distance": _distance(learner_after, opponent_after),
        "distance_before": _distance(learner_before, opponent_before),
        "termination": bool(terminated),
        "truncation": bool(truncated),
    }


def _fold_trace_into_episode(episode: dict[str, Any], trace: Mapping[str, Any]) -> None:
    episode["ticks"] = int(trace["tick"]) + 1
    episode["attack_requested"] += int(trace["attack_press"])
    episode["attack_starts"] += int(trace["attack_state_start"])
    episode["hits"] += int(trace["damage_delta"] > 0)
    episode["damage_done"] += float(trace["damage_delta"])
    episode["damage_taken"] += float(trace["damage_taken_delta"])
    episode["self_stock_losses"] += int(trace["self_stock_delta"] < 0)
    episode["opponent_stock_losses"] += int(trace["opponent_stock_delta"] < 0)
    episode["_last_distance"] = float(trace["distance"])
    episode.setdefault("_first_distance", float(trace["distance_before"]))


def _final_episode_fields(env: Any, final_stats: tuple[Mapping[str, Any], Mapping[str, Any]], start_distance: float | None) -> dict[str, Any]:
    learner, opponent = final_stats
    result = _result_from_stats(learner, opponent)
    current_distance = _distance_between_players(env)
    first_distance = start_distance if start_distance is not None else current_distance
    return {
        "result": result,
        "learner_lives": learner["lives_left"],
        "opponent_lives": opponent["lives_left"],
        "final_distance": current_distance,
        "approach_progress": (first_distance - current_distance) if first_distance is not None and current_distance is not None else None,
    }


def _diagnose_episode(episode: Mapping[str, Any]) -> list[str]:
    diagnoses: list[str] = []
    progress = episode.get("approach_progress")
    if progress is not None and progress <= 0.1:
        diagnoses.append("no_approach")
    if int(episode.get("attack_requested", 0)) == 0:
        diagnoses.append("no_attack_request")
    elif int(episode.get("attack_starts", 0)) == 0:
        diagnoses.append("requested_but_no_attack_start")
    elif float(episode.get("damage_done", 0.0)) <= 0:
        diagnoses.append("attack_started_but_no_hit")
    cause = episode.get("terminal_cause")
    if cause == "self_fall":
        diagnoses.append("self_fall")
    if cause in {"timeout", "episode_max"}:
        diagnoses.append("timeout_or_stuck")
        if progress is not None and progress < -0.1:
            diagnoses.append("runaway")
    return diagnoses


def _terminal_cause(trace: Mapping[str, Any], final_stats: tuple[Mapping[str, Any], Mapping[str, Any]]) -> str:
    if trace["self_stock_delta"] < 0:
        return "self_fall"
    if trace["opponent_stock_delta"] < 0:
        return "opponent_fall"
    if trace["termination"]:
        return _result_from_stats(final_stats[0], final_stats[1])
    if trace["truncation"]:
        return "timeout"
    return "unknown"


def _result_from_stats(learner: Mapping[str, Any], opponent: Mapping[str, Any]) -> str:
    if learner["lives_left"] > opponent["lives_left"]:
        return "win"
    if learner["lives_left"] < opponent["lives_left"]:
        return "loss"
    if learner["damage_done"] > opponent["damage_done"]:
        return "win"
    if learner["damage_done"] < opponent["damage_done"]:
        return "loss"
    return "draw"


def _accumulate_summary(summary: dict[str, Any], episode: Mapping[str, Any]) -> None:
    summary["episodes"] += 1
    result_key = {"win": "wins", "loss": "losses", "draw": "draws"}[str(episode["result"])]
    summary[result_key] += 1
    summary["damage_done"] += float(episode["damage_done"])
    summary["damage_taken"] += float(episode["damage_taken"])
    summary["falls"] += int(episode["self_stock_losses"] > 0)
    summary["timeouts"] += int(episode["terminal_cause"] in {"timeout", "episode_max"})
    summary["attack_requested"] += int(episode["attack_requested"])
    summary["attack_starts"] += int(episode["attack_starts"])
    summary["hits"] += int(episode["hits"])


def _players_snapshot(env: Any) -> list[dict[str, Any]]:
    players = list(getattr(env, "players", []))
    return [_player_snapshot(player) for player in players[:2]]


def _player_snapshot(player: Any) -> dict[str, Any]:
    body = getattr(player, "body", None)
    position = getattr(body, "position", (None, None))
    velocity = getattr(body, "velocity", (None, None))
    grounded_value = getattr(player, "is_on_floor", None)
    grounded = bool(grounded_value()) if callable(grounded_value) else bool(getattr(player, "grounded", False))
    return {
        "x": _float_or_none(_indexable(position, 0)),
        "y": _float_or_none(_indexable(position, 1)),
        "vx": _float_or_none(_indexable(velocity, 0)),
        "vy": _float_or_none(_indexable(velocity, 1)),
        "state": type(getattr(player, "state", None)).__name__,
        "facing": str(getattr(player, "facing", "")),
        "stocks": int(getattr(player, "stocks", 0)),
        "damage": float(getattr(player, "damage_taken_total", 0.0)),
        "grounded": grounded,
        "weapon": type(getattr(player, "weapon", None)).__name__ if getattr(player, "weapon", None) is not None else None,
    }


def _stats(env: Any, slot: int) -> dict[str, Any]:
    stats = env.get_stats(slot)
    return {
        "damage_done": float(getattr(stats, "damage_done")),
        "damage_taken": float(getattr(stats, "damage_taken")),
        "lives_left": int(getattr(stats, "lives_left")),
    }


def _distance_between_players(env: Any) -> float | None:
    players = _players_snapshot(env)
    if len(players) < 2:
        return None
    return _distance(players[0], players[1])


def _distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float | None:
    if a.get("x") is None or b.get("x") is None:
        return None
    return float(((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5)


def _is_attack_state(player_snapshot: Mapping[str, Any]) -> bool:
    return "Attack" in str(player_snapshot.get("state", ""))


def _obs_for_slot(observations: Any, slot: int) -> np.ndarray:
    if isinstance(observations, Mapping):
        return _validate_observation(observations[slot], f"observation[{slot}]")
    return _validate_observation(observations[slot], f"observation[{slot}]")


def _write_video(
    video_dir: Path,
    case: EvaluationCase,
    mode: str,
    frames: Sequence[np.ndarray],
    video_capture: Callable[[Sequence[np.ndarray], Path], None] | None,
) -> Path:
    if video_capture is None:
        raise VideoUnavailableError("video requested but no capture function was provided")
    if not frames:
        raise VideoUnavailableError("video capture had no frames to write")
    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / f"{case.case_id}_{mode}.mp4"
    video_capture(frames, path)
    if not path.exists() or path.stat().st_size <= 0:
        raise VideoUnavailableError(f"video capture did not produce a nonempty artifact: {path}")
    return Path("videos") / path.name


def _open_video_recorder(path: Path, video_capture: Callable[[Sequence[np.ndarray], Path], None] | None) -> Any:
    if video_capture is not None:
        return _CallbackVideoRecorder(path, video_capture)
    return _open_default_video_recorder(path)


def _open_default_video_recorder(path: Path) -> Any:
    return _ImageioVideoRecorder(path)


class _CallbackVideoRecorder:
    def __init__(self, path: Path, video_capture: Callable[[Sequence[np.ndarray], Path], None]) -> None:
        self.path = path
        self.video_capture = video_capture
        self.frames: list[np.ndarray] = []
        self.closed = False

    def append(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame, dtype=np.uint8).copy())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.video_capture(self.frames, self.path)
        _verify_video_artifact(self.path)

    def abort(self) -> None:
        self.closed = True


class _ImageioVideoRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: Any | None = None
        self.frames = 0
        self.closed = False

    def append(self, frame: np.ndarray) -> None:
        if self.closed:
            raise VideoUnavailableError("cannot append to a closed video recorder")
        if self.writer is None:
            try:
                import imageio
            except Exception as exc:
                raise VideoUnavailableError("imageio is required for default MP4 video capture") from exc
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.writer = imageio.get_writer(str(self.path), fps=VIDEO_FPS, codec="libx264", quality=8)
            except Exception as exc:
                raise VideoUnavailableError(f"failed to open MP4 video writer: {self.path}: {exc}") from exc
        try:
            self.writer.append_data(np.asarray(frame, dtype=np.uint8))
        except Exception as exc:
            raise VideoUnavailableError(f"failed to append MP4 frame: {self.path}: {exc}") from exc
        self.frames += 1

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.frames <= 0:
            raise VideoUnavailableError("video capture had no frames to write")
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as exc:
                raise VideoUnavailableError(f"failed to close MP4 video writer: {self.path}: {exc}") from exc
        _verify_default_video_decodable(self.path)

    def abort(self) -> None:
        self.closed = True
        if self.writer is not None:
            with contextlib.suppress(Exception):
                self.writer.close()


def _render_video_frame(env: Any) -> np.ndarray:
    render = getattr(env, "render", None)
    if not callable(render):
        raise VideoUnavailableError("video requested but environment does not expose render()")
    try:
        raw = render()
    except Exception as exc:
        raise VideoUnavailableError(f"environment render() failed during video capture: {exc}") from exc
    frame = np.asarray(raw)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise VideoUnavailableError(f"render() must return an RGB frame with shape HxWx3, got {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.swapaxes(frame, 0, 1))


def _verify_video_artifact(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise VideoUnavailableError(f"video capture did not produce a nonempty artifact: {path}")


def _verify_default_video_decodable(path: Path) -> None:
    try:
        import imageio

        reader = imageio.get_reader(str(path))
        try:
            reader.get_data(0)
        finally:
            reader.close()
    except Exception as exc:
        raise VideoUnavailableError(f"video artifact is not decodable: {path}: {exc}") from exc


def _build_manifest(
    *,
    cases: Sequence[EvaluationCase],
    modes: Sequence[str],
    checkpoint_record: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "git_revision": _git_revision(),
        "checkpoint": dict(checkpoint_record),
        "cases": [case.to_json() for case in cases],
        "modes": list(modes),
        "placement_limitations": "reset_only: this first evaluator does not mutate game internals for exact starting placements.",
        "artifact_sha256": _hash_artifacts(output_dir),
    }


def _hash_artifacts(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            hashes[str(path.relative_to(output_dir))] = _sha256_file(path)
    return hashes


def _checkpoint_record(path: str | Path | None, *, required: bool = False) -> dict[str, Any]:
    if path is None:
        return {"path": None, "sha256": None, "note": "injected policy object"}
    checkpoint = Path(path)
    if not checkpoint.exists() or not checkpoint.is_file():
        if required:
            raise FileNotFoundError(f"checkpoint_path does not exist or is not a file: {checkpoint}")
        return {"path": str(checkpoint), "sha256": None, "note": "policy source path not available as a local file"}
    return {
        "path": str(checkpoint),
        "sha256": _sha256_file(checkpoint),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _ensure_game_import_path() -> None:
    root = PROJECT_ROOT
    game_root = root / "UTMIST-AI2-main"
    if str(game_root) not in sys.path:
        sys.path.insert(0, str(game_root))


def _game_environment_dir() -> Path:
    return PROJECT_ROOT / "UTMIST-AI2-main" / "environment"


@contextlib.contextmanager
def _pushd(path: Path) -> Iterator[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class _ScopedCwdEnv:
    def __init__(self, env: Any, cwd: Path) -> None:
        self._env = env
        self._cwd = cwd

    @property
    def unwrapped(self) -> Any:
        return getattr(self._env, "unwrapped", self._env)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        with _pushd(self._cwd):
            return self._env.reset(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        with _pushd(self._cwd):
            return self._env.step(*args, **kwargs)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        with _pushd(self._cwd):
            return self._env.render(*args, **kwargs)

    def close(self) -> None:
        with _pushd(self._cwd):
            if hasattr(self._env, "close"):
                self._env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def _try_import_torch() -> Any | None:
    try:
        import torch

        return torch
    except Exception:
        return None


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _json_line(data: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(data), sort_keys=True, separators=(",", ":")) + "\n"


def _json_block(data: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _indexable(value: Any, index: int) -> Any:
    try:
        return value[index]
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
