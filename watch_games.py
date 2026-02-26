#!/usr/bin/env python3
"""
Watch your trained agent fight against various opponents.
Saves videos and reports results.

Usage:
    python watch_games.py                           # Default: 1 game vs RandomAgent
    python watch_games.py --opponent based          # 1 game vs BasedAgent
    python watch_games.py --opponent random --games 5   # 5 games vs RandomAgent
    python watch_games.py --opponent constant       # 1 game vs ConstantAgent (does nothing)
    python watch_games.py --opponent clockwork      # 1 game vs ClockworkAgent (patterns)

Opponents:
    random    - RandomAgent (random actions)
    based     - BasedAgent (rule-based, hardest built-in)
    constant  - ConstantAgent (does nothing)
    clockwork - ClockworkAgent (preset patterns)
"""

import sys
sys.path.insert(0, 'UTMIST-AI2-main')

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import argparse
import numpy as np
from environment.environment import CameraResolution, WarehouseBrawl
from environment.agent import RandomAgent, BasedAgent, ConstantAgent, ClockworkAgent
from train_utmist_v2 import Float32Wrapper, OpponentHistoryWrapper
from stable_baselines3 import PPO
from tqdm import tqdm
import imageio

# ============================================================================
# CONFIGURATION - CHANGE THESE
# ============================================================================
MODEL_PATH = "./results/ppo_utmist/model/phase4_final.zip"
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


def get_opponent(opponent_type: str):
    """Get opponent agent by name."""
    opponents = {
        "random": RandomAgent,
        "based": BasedAgent,
        "constant": ConstantAgent,
        "clockwork": ClockworkAgent,
    }
    if opponent_type.lower() not in opponents:
        print(f"❌ Unknown opponent: {opponent_type}")
        print(f"   Available: {', '.join(opponents.keys())}")
        sys.exit(1)
    return opponents[opponent_type.lower()]()


def run_game(model, opponent, game_num, opponent_name, save_video=True):
    """Run a single game and return results."""
    env = WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=True)
    env.max_timesteps = 30 * 90  # 90 second match
    
    # Apply matching wrappers
    env = Float32Wrapper(env)
    zero_history = (opponent_name == "constant")
    env = OpponentHistoryWrapper(env, zero_history=zero_history)
    
    observations, _ = env.reset()
    obs = observations[0]
    opponent.get_env_info(env.unwrapped)
    opponent_obs = observations[1]
    
    # Initialize frame stack
    obs_shape = (obs.shape[0],)
    frame_stack = FrameStack(FRAME_STACK, obs_shape)
    stacked_obs = frame_stack.reset(obs)
    
    # Video setup
    video_path = None
    writer = None
    if save_video:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        video_path = os.path.join(OUTPUT_DIR, f"game_{game_num}_vs_{opponent_name}.mp4")
        writer = imageio.get_writer(video_path, fps=30, codec='libx264', quality=8)
    
    # Run game
    for step in range(env.unwrapped.max_timesteps):
        action, _ = model.predict(stacked_obs, deterministic=True)
        opp_action = opponent.predict(opponent_obs)
        
        full_action = {0: action, 1: opp_action}
        observations, rewards, terminated, truncated, _ = env.step(full_action)
        
        obs = observations[0]
        opponent_obs = observations[1]
        stacked_obs = frame_stack.step(obs)
        
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
    
    # Get stats
    stats = env.unwrapped.get_stats(0)
    opp_stats = env.unwrapped.get_stats(1)
    
    # Determine result
    if stats.lives_left > opp_stats.lives_left:
        result = "WIN"
    elif stats.lives_left < opp_stats.lives_left:
        result = "LOSS"
    else:
        if stats.damage_done > opp_stats.damage_done:
            result = "WIN"
        else:
            result = "DRAW"
    
    env.close()
    
    return {
        'game': game_num,
        'result': result,
        'lives': stats.lives_left,
        'opp_lives': opp_stats.lives_left,
        'damage_dealt': stats.damage_done,
        'damage_taken': stats.damage_taken,
        'video': video_path
    }


def main():
    parser = argparse.ArgumentParser(description="Watch your agent play against opponents")
    parser.add_argument('--opponent', '-o', type=str, default='random',
                        help='Opponent type: random, based, constant, clockwork')
    parser.add_argument('--games', '-g', type=int, default=1,
                        help='Number of games to run')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording (faster)')
    parser.add_argument('--model', '-m', type=str, default=MODEL_PATH,
                        help='Path to model file')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"🎮 WATCH AGENT VS {args.opponent.upper()}")
    print("=" * 60)
    print(f"📂 Model: {args.model}")
    print(f"🎯 Opponent: {args.opponent}")
    print(f"🎲 Games: {args.games}")
    print(f"📹 Video: {'Disabled' if args.no_video else 'Enabled'}")
    print("=" * 60)
    
    # Load model
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    model = PPO.load(args.model, device='cpu')
    print(f"✅ Model loaded successfully\n")
    
    # Run games
    results = []
    for game_num in range(1, args.games + 1):
        print(f"🎮 Game {game_num}/{args.games}...")
        opponent = get_opponent(args.opponent)
        result = run_game(model, opponent, game_num, args.opponent, save_video=not args.no_video)
        results.append(result)
        
        marker = "✅" if result['result'] == "WIN" else ("❌" if result['result'] == "LOSS" else "➖")
        print(f"   {marker} {result['result']}: Lives {result['lives']}-{result['opp_lives']} | "
              f"Dmg: {result['damage_dealt']:.0f} dealt, {result['damage_taken']:.0f} taken")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    
    wins = sum(1 for r in results if r['result'] == 'WIN')
    losses = sum(1 for r in results if r['result'] == 'LOSS')
    draws = sum(1 for r in results if r['result'] == 'DRAW')
    win_rate = (wins / len(results)) * 100
    
    avg_dealt = sum(r['damage_dealt'] for r in results) / len(results)
    avg_taken = sum(r['damage_taken'] for r in results) / len(results)
    
    print(f"🏆 Win Rate: {win_rate:.0f}% ({wins}W / {losses}L / {draws}D)")
    print(f"⚔️  Avg Damage Dealt: {avg_dealt:.1f}")
    print(f"🛡️  Avg Damage Taken: {avg_taken:.1f}")
    print(f"📈 Net Damage: {avg_dealt - avg_taken:+.1f}")
    
    if not args.no_video:
        print(f"\n📹 Videos saved to: {OUTPUT_DIR}/")
        
        # Find best game
        best = max(results, key=lambda x: (x['lives'] * 10 + x['damage_dealt']))
        if best['result'] == 'WIN':
            print(f"⭐ Best game: Game {best['game']} - {best['video']}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
