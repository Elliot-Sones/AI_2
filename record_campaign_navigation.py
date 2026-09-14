#!/usr/bin/env python3
"""Record one validated E6 navigation-policy video from a trusted checkpoint.

This is a small, campaign-local wrapper around the proven navigation replay
recorder shape from
`AI_2-learning-validation-20260905/baselines/fresh-curriculum-20260905/record_navigation.py`.
It keeps the no-render/render trajectory parity guard, but makes policy loading
CUDA-only and records explicit device attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any, Callable
from unittest.mock import patch


FPS = 30
LABEL = "E6 campaign navigation replay from trained checkpoint"
OPPONENT_LABEL = "frozen navigation target, not combat opponent"
REQUESTED_DEVICE = "cuda"
SOURCE_FILES = (
    "config.yaml",
    "train.py",
    "learning_eval.py",
    "UTMIST-AI2-main/environment/environment.py",
    "UTMIST-AI2-main/environment/agent.py",
)


def _np():
    import numpy as numpy

    return numpy


# Copied with provenance from:
# AI_2-learning-validation-20260905/baselines/learning-validation-20260905/
# rendered-matches/record_match.py
# Kept local on purpose: the isolated GPU runtime does not carry that historical
# helper path, and verifier behavior should not vary by optional local files.
def orient_frame(raw):
    numpy = _np()
    return numpy.ascontiguousarray(numpy.swapaxes(raw, 0, 1), dtype=numpy.uint8)


def comparable(episode):
    return {key: value for key, value in episode.items() if key != "elapsed"}


class FrameTap:
    def __init__(self, capture):
        self.capture = capture
        self.started = False
        self.done = False

    def after_reset(self):
        if not self.started:
            self.started = True
            self.capture("reset")

    def after_step(self, terminated, truncated):
        if not self.done:
            self.done = bool(terminated or truncated)
            self.capture("terminal" if self.done else "step")


def digest(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _jsonable_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(part) for part in value]


def _first_attr(wrapper: Any, name: str, default: Any = None) -> Any:
    current = wrapper
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, name):
            return getattr(current, name)
        current = getattr(current, "env", None)
    return default


def _phase_config(params: dict, phase: str) -> dict:
    phases = params.get("phases", {})
    if phase in phases:
        return phases[phase]
    for key, value in phases.items():
        if str(key) == str(phase):
            return value
    raise ValueError(f"phase {phase!r} not found in params")


def phase_max_steps(params: dict, phase: str) -> int:
    config = _phase_config(params, phase)
    seconds = config.get("navigation", {}).get("time_limit_seconds")
    if seconds is None:
        raise ValueError(f"phase {phase!r} has no navigation.time_limit_seconds")
    return int(float(seconds) * FPS)


def device_text(device) -> str:
    return str(device)


def compatible_cuda(actual: str) -> bool:
    return actual == "cuda" or actual.startswith("cuda:")


def policy_device(policy) -> str:
    device = getattr(policy, "device", None)
    if device is not None:
        return device_text(device)
    try:
        return device_text(next(policy.parameters()).device)
    except (AttributeError, StopIteration):
        return "unknown"


def load_cuda_policy_model(PPO, torch, checkpoint: Path):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")

    model = PPO.load(str(checkpoint), device=REQUESTED_DEVICE)
    actual_model_device = device_text(getattr(model, "device", "unknown"))
    actual_policy_device = policy_device(model.policy)
    if not compatible_cuda(actual_model_device) or not compatible_cuda(actual_policy_device):
        raise RuntimeError(
            "device mismatch: "
            f"requested={REQUESTED_DEVICE} "
            f"model={actual_model_device} policy={actual_policy_device}"
        )
    return model, {
        "requested_device": REQUESTED_DEVICE,
        "actual_model_device": actual_model_device,
        "actual_policy_device": actual_policy_device,
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        numpy = _np()
    except ModuleNotFoundError:
        pass
    else:
        numpy.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_repo_modules(repo: Path):
    os.chdir(repo)
    for path in (repo, repo / "UTMIST-AI2-main"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    import torch
    import yaml
    from stable_baselines3 import PPO
    import learning_eval as le

    return torch, yaml, PPO, le


def _source_hashes(repo: Path, config: Path) -> dict[str, str]:
    hashes = {}
    for name in SOURCE_FILES:
        path = config if name == "config.yaml" else repo / name
        if path.is_file():
            hashes[name] = digest(path)
    return hashes


def _video_path(output: Path, phase: str, seed: int, mode: str) -> Path:
    return output / f"campaign-navigation-phase{phase}-{mode}-seed{int(seed)}.mp4"


def resolve_encoder() -> dict[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return {"path": ffmpeg, "source": "PATH"}
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg binary required; neither PATH ffmpeg nor imageio_ffmpeg bundled ffmpeg is available"
        ) from exc

    path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"imageio_ffmpeg returned unusable ffmpeg binary: {path}")
    return {"path": str(path), "source": "imageio_ffmpeg"}


def load_expected_episode(path: Path | None, seed: int) -> dict | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    if payload.get("status") not in (None, "completed"):
        raise ValueError(f"expected eval shard is not completed: {path}")
    matches = []
    for episode in payload.get("episodes", []):
        if int(episode.get("seed", -1)) == int(seed):
            matches.append(episode)
    for result in payload.get("results", []):
        for episode in result.get("episodes", []):
            if int(episode.get("seed", -1)) == int(seed):
                matches.append(episode)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one episode for seed {seed}; found {len(matches)}")
    return matches[0]


def trace_row(*, stage: str, raw: Any, wrapper: Any, metrics: dict) -> dict:
    players = [raw.players[index] for index in (0, 1)]
    target_position = _first_attr(wrapper, "target_position")
    return {
        "stage": stage,
        "step": int(raw.steps),
        "observation_sha256": hashlib.sha256(
            _np().asarray(wrapper.last_observation).tobytes()
        ).hexdigest(),
        "metrics": metrics,
        "navigation": {
            "target_position": _jsonable_vector(target_position),
            "targets_this_episode": int(_first_attr(wrapper, "targets_this_episode", 0)),
            "total_targets": int(_first_attr(wrapper, "total_targets", 0)),
            "falls": int(_first_attr(wrapper, "falls", 0)),
            "fell_this_episode": bool(_first_attr(wrapper, "fell_this_episode", False)),
        },
        "positions": [_jsonable_vector(player.body.position) for player in players],
        "velocities": [_jsonable_vector(player.body.velocity) for player in players],
        "actions": {
            str(key): _np().asarray(value).tolist()
            for key, value in getattr(raw, "cur_action", {}).items()
        },
    }


def record_episode(
    *,
    le,
    model,
    params: dict,
    phase: str,
    seed: int,
    max_steps: int,
    deterministic: bool,
    video: Path | None,
    encoder_path: str | None = None,
):
    trace: list[dict] = []
    encoder = None
    frame_count = 0
    base_wrapper = le.TerminalMetricsWrapper

    class CapturedMetrics(base_wrapper):
        def __init__(self, env, seed=0):
            super().__init__(env, seed=seed)
            self.last_observation = None
            self.tap = FrameTap(self.capture)

        def capture(self, stage):
            nonlocal encoder, frame_count
            raw = le._raw_env_from(self)
            metrics = le._snapshot_env(self)
            nav_goal = le._nav_goal_from_wrapper(self)
            metrics.update(
                {
                    "nav_targets": int(getattr(self, "_nav_targets", 0)),
                    "nav_falls": int(getattr(self, "_nav_falls", 0)),
                    "nav_goal": int(nav_goal or 0),
                    "nav_completed": bool(getattr(self, "_nav_completed", False)),
                }
            )
            trace.append(trace_row(stage=stage, raw=raw, wrapper=self, metrics=metrics))
            if video is not None:
                with le._preserve_rng():
                    frame = orient_frame(raw.render())
                if encoder is None:
                    if not encoder_path:
                        raise RuntimeError("encoder_path is required when rendering")
                    height, width, channels = frame.shape
                    if channels != 3:
                        raise RuntimeError(f"expected RGB frame, got shape {frame.shape}")
                    encoder = subprocess.Popen(
                        [
                            encoder_path,
                            "-v",
                            "error",
                            "-nostdin",
                            "-n",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "-s",
                            f"{width}x{height}",
                            "-r",
                            str(FPS),
                            "-i",
                            "pipe:0",
                            "-an",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "fast",
                            "-crf",
                            "20",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            str(video),
                        ],
                        stdin=subprocess.PIPE,
                    )
                assert encoder.stdin is not None
                encoder.stdin.write(frame.tobytes())
                frame_count += 1

        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            self.last_observation = obs
            self.tap.after_reset()
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            self.last_observation = obs
            self.tap.after_step(terminated, truncated)
            return obs, reward, terminated, truncated, info

    try:
        with patch.object(le, "TerminalMetricsWrapper", CapturedMetrics):
            result = le.evaluate_policy(
                model,
                params,
                str(phase),
                [int(seed)],
                opponent=None,
                max_steps=int(max_steps),
                deterministic=bool(deterministic),
            )
    finally:
        if encoder is not None:
            assert encoder.stdin is not None
            encoder.stdin.close()
            code = encoder.wait(timeout=30)
            if code:
                raise RuntimeError(f"ffmpeg exited {code}")
    return result["episodes"][0], trace, frame_count


def record_validated_pair(
    *,
    record_once: Callable[..., tuple[dict, list[dict], int]] = record_episode,
    le,
    model,
    params: dict,
    phase: str,
    seed: int,
    max_steps: int,
    mode: str,
    video: Path,
    encoder_path: str | None = None,
):
    if mode not in {"deterministic", "sampled"}:
        raise ValueError(f"unsupported mode: {mode!r}")
    deterministic = mode == "deterministic"

    _seed_all(int(seed))
    control, control_trace, _ = record_once(
        le=le,
        model=model,
        params=params,
        phase=str(phase),
        seed=int(seed),
        max_steps=int(max_steps),
        deterministic=deterministic,
        video=None,
        encoder_path=encoder_path,
    )
    _seed_all(int(seed))
    rendered, rendered_trace, frame_count = record_once(
        le=le,
        model=model,
        params=params,
        phase=str(phase),
        seed=int(seed),
        max_steps=int(max_steps),
        deterministic=deterministic,
        video=video,
        encoder_path=encoder_path,
    )

    trace_equal = control_trace == rendered_trace
    result_equal = comparable(control) == comparable(rendered)
    if not trace_equal or not result_equal:
        raise RuntimeError("Rendering changed the local trajectory; recording is not validated")
    return {
        "control_episode": control,
        "rendered_episode": rendered,
        "control_trace": control_trace,
        "rendered_trace": rendered_trace,
        "trace_equal": trace_equal,
        "result_equal": result_equal,
        "frame_count": frame_count,
    }


def build_proof(
    *,
    repo: Path,
    checkpoint: Path,
    config: Path,
    phase: str,
    seed: int,
    mode: str,
    max_steps: int,
    model_timesteps: int,
    device_attestation: dict,
    expected_episode: dict | None,
    control_episode: dict,
    rendered_episode: dict,
    trace_equal: bool,
    result_equal: bool,
    remote_equal: bool | None,
    video: Path,
    frame_count: int,
    encoder: dict,
    versions: dict,
) -> dict:
    return {
        "label": LABEL,
        "opponent": OPPONENT_LABEL,
        "phase": str(phase),
        "mode": str(mode),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_sha256": digest(checkpoint),
        "config": str(Path(config)),
        "config_sha256": digest(config),
        "stored_training_steps": int(model_timesteps),
        "source": str(Path(repo)),
        "source_hashes": _source_hashes(repo, config),
        "requested_device": str(device_attestation["requested_device"]),
        "device": str(device_attestation["actual_policy_device"]),
        "actual_model_device": str(device_attestation["actual_model_device"]),
        "actual_policy_device": str(device_attestation["actual_policy_device"]),
        "model_player": "P1 / slot 0",
        "opponent_player": "P2 / slot 1, frozen target marker",
        "expected_eval_episode": expected_episode,
        "local_control_episode": control_episode,
        "rendered_episode": rendered_episode,
        "outcome": rendered_episode.get("outcome"),
        "local_render_trace_equal": bool(trace_equal),
        "local_render_outcome_equal": bool(result_equal),
        "expected_eval_equal": remote_equal,
        "video": Path(video).name,
        "frames": int(frame_count),
        "fps": FPS,
        "video_sha256": digest(video),
        "encoder": {"path": str(encoder["path"]), "source": str(encoder["source"])},
        "versions": versions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record one validated campaign navigation replay")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("deterministic", "sampled"), default="deterministic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    config = repo / "config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"config not found: {config}")
    output.mkdir(parents=True, exist_ok=True)
    video = _video_path(output, args.phase, args.seed, args.mode)
    if video.exists():
        raise FileExistsError(video)
    encoder = resolve_encoder()
    for key in ("SDL_VIDEODRIVER", "SDL_AUDIODRIVER"):
        os.environ[key] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["WANDB_MODE"] = "disabled"

    torch, yaml, PPO, le = _load_repo_modules(repo)
    torch.set_num_threads(1)
    params = yaml.safe_load(config.read_text()) or {}
    params.setdefault("curriculum", {})["phase"] = str(args.phase)
    params.setdefault("opponent_history", {})["enabled"] = False
    max_steps = phase_max_steps(params, str(args.phase))
    checkpoint_hash = digest(checkpoint)
    expected_episode = load_expected_episode(args.expected, seed=args.seed) if args.expected else None
    model, device_attestation = load_cuda_policy_model(PPO, torch, checkpoint)

    print("Running no-render navigation control; no training", flush=True)
    pair = record_validated_pair(
        le=le,
        model=model,
        params=params,
        phase=str(args.phase),
        seed=args.seed,
        max_steps=max_steps,
        mode=args.mode,
        video=video,
        encoder_path=encoder["path"],
    )
    remote_equal = (
        comparable(pair["rendered_episode"]) == comparable(expected_episode)
        if expected_episode
        else None
    )
    if digest(checkpoint) != checkpoint_hash:
        raise RuntimeError("checkpoint changed during recording")
    if remote_equal is False:
        raise RuntimeError("Rendered episode does not match expected evaluate_job episode")

    proof = build_proof(
        repo=repo,
        checkpoint=checkpoint,
        config=config,
        phase=str(args.phase),
        seed=args.seed,
        mode=args.mode,
        max_steps=max_steps,
        model_timesteps=int(getattr(model, "num_timesteps", 0)),
        device_attestation=device_attestation,
        expected_episode=expected_episode,
        control_episode=pair["control_episode"],
        rendered_episode=pair["rendered_episode"],
        trace_equal=pair["trace_equal"],
        result_equal=pair["result_equal"],
        remote_equal=remote_equal,
        video=video,
        frame_count=pair["frame_count"],
        encoder=encoder,
        versions={"python": sys.version, "torch": torch.__version__},
    )
    video.with_suffix(".json").write_text(json.dumps(proof, indent=2) + "\n")
    video.with_suffix(".trace.json").write_text(json.dumps(pair["rendered_trace"]) + "\n")
    print(json.dumps(proof, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
