#!/usr/bin/env python3
"""Record classic CPU navigation policies using the configured frozen targets."""

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import os
from pathlib import Path

import numpy as np

from ai2.paths import REPO_ROOT
from tools.visualization.watch_games import FrameStack, load_training

CONFIG_PATH = REPO_ROOT / "config.yaml"
FRAME_STACK = 4
OUTPUT_DIR = REPO_ROOT / "videos"


def run_navigation_game(model, phase_config, game_num, save_video=True,
                        frame_stack=FRAME_STACK, output_dir=OUTPUT_DIR):
    train = load_training()
    cfg = train.OpponentsCfg(opponents={"constant_agent": (1.0, train.ConstantAgent)})
    env = train.SelfPlayWarehouseBrawl(
        opponent_cfg=cfg, save_handler=None, resolution=train.CameraResolution.LOW,
    )
    max_steps = int(phase_config.navigation.get("time_limit_seconds", 90) * 30)
    env.raw_env.max_timesteps = max_steps
    env = train.Float32Wrapper(env)
    env = train.FrozenOpponentWrapper(env, nav_config=phase_config.navigation, debug_logs=False)
    env = train.OpponentHistoryWrapper(env, zero_history=True)
    writer = None
    video_path = None
    try:
        obs, _ = env.reset()
        stack = FrameStack(frame_stack, obs.shape)
        stacked_obs = stack.reset(obs)
        if save_video:
            import imageio
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            video_path = str(Path(output_dir) / f"nav_game_{game_num}.mp4")
            writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=8)
        targets_reached = falls = steps = 0
        for _ in range(max_steps):
            action, _ = model.predict(stacked_obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            stacked_obs = stack.step(obs)
            steps += 1
            targets_reached += int(info.get("nav_reached", 0) > 0)
            falls += int(info.get("nav_fall", 0) > 0)
            if writer is not None:
                writer.append_data(np.fliplr(np.rot90(env.render(), k=-1)))
            if terminated or truncated:
                break
        return {"game": game_num, "targets_reached": targets_reached,
                "falls": falls, "steps": steps, "video": video_path}
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record a classic CPU navigation policy")
    parser.add_argument("--model", "-m", required=True, help="Path to PPO checkpoint")
    parser.add_argument("--config", "-c", type=Path, default=CONFIG_PATH)
    parser.add_argument("--phase", "-p", default="0c")
    parser.add_argument("--games", "-g", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--frame-stack", type=int, help="Override config frame_stack")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.games < 1 or (args.frame_stack is not None and args.frame_stack < 1):
        parser.error("--games and --frame-stack must be positive")
    if not Path(args.model).is_file():
        parser.error(f"model not found: {args.model}")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import yaml
    with args.config.open() as stream:
        params = yaml.safe_load(stream)
    train = load_training()
    try:
        phase = train.PhaseConfig(params, args.phase)
    except ValueError as error:
        parser.error(str(error))
    if not phase.is_navigation:
        parser.error(f"phase {args.phase} is not a navigation phase")
    frame_stack = args.frame_stack if args.frame_stack is not None else params.get("frame_stack", FRAME_STACK)
    if frame_stack < 1 or phase.navigation.get("time_limit_seconds", 90) <= 0:
        parser.error("frame_stack and navigation time limit must be positive")
    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    for number in range(1, args.games + 1):
        result = run_navigation_game(model, phase, number, not args.no_video,
                                     frame_stack, args.output_dir)
        print(f"Game {number}: {result['targets_reached']} targets, {result['falls']} falls, "
              f"{result['steps']} steps")
    if not args.no_video:
        print(f"Videos: {args.output_dir}")


if __name__ == "__main__":
    main()
