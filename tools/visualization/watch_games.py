#!/usr/bin/env python3
"""Record a classic CPU policy against built-in opponents.

Run ``python -m tools.visualization.watch_games --model checkpoint.zip``.
"""

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import os
from pathlib import Path
import sys

import numpy as np

from ai2.paths import REPO_ROOT

FRAME_STACK = 4
OUTPUT_DIR = REPO_ROOT / "videos"
OPPONENTS = ("random", "based", "constant", "clockwork")


class FrameStack:
    """Match SB3 VecFrameStack: zero padding until earlier frames exist."""

    def __init__(self, n_frames, obs_shape):
        if n_frames < 1:
            raise ValueError("frame stack must be positive")
        self.frames = [np.zeros(obs_shape, dtype=np.float32) for _ in range(n_frames)]

    def reset(self, obs):
        for frame in self.frames:
            frame.fill(0)
        self.frames[-1] = np.asarray(obs, dtype=np.float32).copy()
        return self.get()

    def step(self, obs):
        self.frames.pop(0)
        self.frames.append(np.asarray(obs, dtype=np.float32).copy())
        return self.get()

    def get(self):
        return np.concatenate(self.frames, axis=0)


def load_training():
    """Locate the vendored environment independently of the working directory."""
    vendor = str(REPO_ROOT / "UTMIST-AI2-main")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from ai2.training import train
    return train


def get_opponent(opponent_type):
    train = load_training()
    opponents = {
        "random": train.RandomAgent,
        "based": train.BasedAgent,
        "constant": train.ConstantAgent,
        "clockwork": train.ClockworkAgent,
    }
    return opponents[opponent_type.lower()]()


def make_game_env(opponent, *, zero_history=False, max_steps=2700):
    """Use the same scalar-action and opponent-history path as training."""
    train = load_training()
    cfg = train.OpponentsCfg(opponents={"viewer": (1.0, lambda: opponent)})
    env = train.SelfPlayWarehouseBrawl(
        opponent_cfg=cfg, save_handler=None, resolution=train.CameraResolution.LOW,
    )
    env.raw_env.max_timesteps = max_steps
    return train.OpponentHistoryWrapper(
        train.Float32Wrapper(env), zero_history=zero_history,
    )


def run_game(model, opponent, game_num, opponent_name, save_video=True,
             frame_stack=FRAME_STACK, output_dir=OUTPUT_DIR):
    env = make_game_env(opponent, zero_history=opponent_name == "constant")
    writer = None
    video_path = None
    try:
        obs, _ = env.reset()
        stack = FrameStack(frame_stack, obs.shape)
        stacked_obs = stack.reset(obs)
        if save_video:
            import imageio
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            video_path = str(Path(output_dir) / f"game_{game_num}_vs_{opponent_name}.mp4")
            writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=8)
        for _ in range(2700):
            action, _ = model.predict(stacked_obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            stacked_obs = stack.step(obs)
            if writer is not None:
                writer.append_data(np.fliplr(np.rot90(env.render(), k=-1)))
            if terminated or truncated:
                break
        raw_env = env.unwrapped.raw_env
        stats, opp_stats = raw_env.get_stats(0), raw_env.get_stats(1)
        if stats.lives_left != opp_stats.lives_left:
            result = "WIN" if stats.lives_left > opp_stats.lives_left else "LOSS"
        else:
            result = "WIN" if stats.damage_done > opp_stats.damage_done else "DRAW"
        return {
            "game": game_num, "result": result, "lives": stats.lives_left,
            "opp_lives": opp_stats.lives_left, "damage_dealt": stats.damage_done,
            "damage_taken": stats.damage_taken, "video": video_path,
        }
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record a classic CPU policy against built-in opponents")
    parser.add_argument("--model", "-m", required=True, help="Path to PPO checkpoint")
    parser.add_argument("--opponent", "-o", choices=OPPONENTS, default="random")
    parser.add_argument("--games", "-g", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--frame-stack", type=int, default=FRAME_STACK, help="Match the training configuration")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.games < 1 or args.frame_stack < 1:
        parser.error("--games and --frame-stack must be positive")
    if not Path(args.model).is_file():
        parser.error(f"model not found: {args.model}")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    load_training()
    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    results = []
    for number in range(1, args.games + 1):
        result = run_game(model, get_opponent(args.opponent), number, args.opponent,
                          not args.no_video, args.frame_stack, args.output_dir)
        results.append(result)
        print(f"Game {number}: {result['result']}; lives {result['lives']}-{result['opp_lives']}; "
              f"damage {result['damage_dealt']:.0f} dealt, {result['damage_taken']:.0f} taken")
    wins = sum(row["result"] == "WIN" for row in results)
    print(f"Win rate: {100 * wins / len(results):.0f}% ({wins}/{len(results)})")
    if not args.no_video:
        print(f"Videos: {args.output_dir}")


if __name__ == "__main__":
    main()
