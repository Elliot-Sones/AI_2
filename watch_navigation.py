#!/usr/bin/env python3
"""
Watch your trained navigation agent navigate to frozen targets.
Shows the actual navigation training behavior.

Usage:
    python watch_navigation.py                                    # Default model
    python watch_navigation.py --model ./path/to/model.zip       # Specific model
    python watch_navigation.py --games 5                          # Multiple games
    python watch_navigation.py --no-video                         # No recording
"""

import sys
sys.path.insert(0, 'UTMIST-AI2-main')

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import argparse
import numpy as np
from functools import partial
import torch
from environment.environment import CameraResolution, WarehouseBrawl
from environment.agent import ConstantAgent, SelfPlayWarehouseBrawl, RewardManager, RewTerm, OpponentsCfg
from train_utmist_v2 import Float32Wrapper, OpponentHistoryWrapper, FrozenOpponentWrapper, PhaseConfig
from stable_baselines3 import PPO
import imageio
import yaml

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "./results/ppo_utmist_v2/model/nav_0c_interrupted.zip"
CONFIG_PATH = "./utmist_config_v2.yaml"
FRAME_STACK = 4
OUTPUT_DIR = "./videos"
# ============================================================================


class FrameStack:
    def __init__(self, n_frames, obs_shape):
        self.n_frames = n_frames
        self.frames = [np.zeros(obs_shape, dtype=np.float32) for _ in range(n_frames)]
    
    def reset(self, obs):
        for i in range(self.n_frames):
            self.frames[i] = obs.copy()
        return self.get()
    
    def step(self, obs):
        self.frames.pop(0)
        self.frames.append(obs.copy())
        return self.get()
    
    def get(self):
        return np.concatenate(self.frames, axis=0)


def run_navigation_game(model, phase_config, game_num, save_video=True):
    """Run a single navigation game with frozen opponent."""
    # Create environment MATCHING training setup exactly
    def null_reward_func(env, **kwargs):
        return torch.tensor([0.0, 0.0])
    
    reward_manager = RewardManager(
        reward_functions={"null": RewTerm(func=null_reward_func, weight=0.0)},
        signal_subscriptions={}
    )
    
    opponent_cfg = OpponentsCfg(
        opponents={'constant_agent': (1.0, partial(ConstantAgent))}
    )
    
    # USE SelfPlayWarehouseBrawl - SAME AS TRAINING!
    env = SelfPlayWarehouseBrawl(
        opponent_cfg=opponent_cfg,
        save_handler=None,
        resolution=CameraResolution.LOW,
        reward_manager=reward_manager
    )
    env.raw_env.max_timesteps = 30 * 30  # 30 seconds
    
    # Apply same wrappers as training
    env = Float32Wrapper(env)
    env = FrozenOpponentWrapper(env, nav_config=phase_config.navigation, debug_logs=False)
    env = OpponentHistoryWrapper(env, zero_history=True)  # Navigation uses zero history
    
    obs, _ = env.reset()  # SelfPlayWarehouseBrawl returns single obs, not dict
    
    # Initialize frame stack
    obs_shape = (obs.shape[0],)
    frame_stack = FrameStack(FRAME_STACK, obs_shape)
    stacked_obs = frame_stack.reset(obs)
    
    # Video setup
    video_path = None
    writer = None
    if save_video:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        video_path = os.path.join(OUTPUT_DIR, f"nav_game_{game_num}.mp4")
        writer = imageio.get_writer(video_path, fps=30, codec='libx264', quality=8)
    
    # Tracking
    targets_reached = 0
    falls = 0
    
    # Run game
    for step in range(env.unwrapped.raw_env.max_timesteps):
        action, _ = model.predict(stacked_obs, deterministic=True)
        
        # SelfPlayWarehouseBrawl expects scalar action, not dict!
        obs, reward, terminated, truncated, info = env.step(action)
        stacked_obs = frame_stack.step(obs)
        
        # Track navigation events
        if info.get('nav_reached', 0) > 0:
            targets_reached += 1
        if info.get('nav_fall', 0) > 0:
            falls += 1
        
        # Capture frame
        if writer:
            img = env.render()
            img = np.rot90(img, k=-1)
            img = np.fliplr(img)
            writer.append_data(img)
        
        if terminated or truncated:
            break
    
    if writer:
        writer.close()
    
    env.close()
    
    return {
        'game': game_num,
        'targets_reached': targets_reached,
        'falls': falls,
        'steps': step + 1,
        'video': video_path
    }


def main():
    parser = argparse.ArgumentParser(description="Watch navigation agent reach targets")
    parser.add_argument('--model', '-m', type=str, default=MODEL_PATH,
                        help='Path to model file')
    parser.add_argument('--config', '-c', type=str, default=CONFIG_PATH,
                        help='Path to config file')
    parser.add_argument('--games', '-g', type=int, default=1,
                        help='Number of games to run')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--phase', '-p', type=str, default='0c',
                        help='Navigation phase (0a, 0b, 0c, 0d, 0e)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧭 WATCH NAVIGATION AGENT")
    print("=" * 60)
    print(f"📂 Model: {args.model}")
    print(f"📄 Config: {args.config}")
    print(f"🎯 Phase: {args.phase}")
    print(f"🎲 Games: {args.games}")
    print(f"📹 Video: {'Disabled' if args.no_video else 'Enabled'}")
    print("=" * 60)
    
    # Load config
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
    
    # Get phase config
    try:
        phase_config = PhaseConfig(params, args.phase)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    if not phase_config.is_navigation:
        print(f"❌ Phase {args.phase} is not a navigation phase!")
        sys.exit(1)
    
    print(f"📍 Nav Config: dist={phase_config.navigation['min_dist']}-{phase_config.navigation['max_dist']}")
    
    # Load model
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    model = PPO.load(args.model, device='cpu')
    print(f"✅ Model loaded successfully\n")
    
    # Run games
    results = []
    for game_num in range(1, args.games + 1):
        print(f"🧭 Navigation Game {game_num}/{args.games}...")
        result = run_navigation_game(model, phase_config, game_num, save_video=not args.no_video)
        results.append(result)
        
        print(f"   🎯 Targets: {result['targets_reached']} | 💀 Falls: {result['falls']} | ⏱️ Steps: {result['steps']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 NAVIGATION SUMMARY")
    print("=" * 60)
    
    total_targets = sum(r['targets_reached'] for r in results)
    total_falls = sum(r['falls'] for r in results)
    avg_steps = sum(r['steps'] for r in results) / len(results)
    
    print(f"🎯 Total Targets Reached: {total_targets}")
    print(f"💀 Total Falls: {total_falls}")
    print(f"⏱️  Avg Steps per Game: {avg_steps:.0f}")
    
    if not args.no_video:
        print(f"\n📹 Videos saved to: {OUTPUT_DIR}/")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
