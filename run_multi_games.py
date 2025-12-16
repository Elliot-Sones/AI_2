#!/usr/bin/env python3
"""
Run multiple games vs BasedAgent, save all videos, report best result.
"""

import sys
sys.path.insert(0, 'UTMIST-AI2-main')

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import numpy as np
from environment.environment import CameraResolution, WarehouseBrawl
from environment.agent import BasedAgent
from stable_baselines3 import PPO
from tqdm import tqdm
import skvideo.io

# Config
MODEL_PATH = "./results/ppo_utmist_v2/model/phase2_final.zip"
NUM_GAMES = 5
FRAME_STACK = 4

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

print("=" * 60)
print(f"🎮 RUNNING {NUM_GAMES} GAMES VS BASEDAGENT")
print("=" * 60)
print(f"📂 Model: {MODEL_PATH}")
print("=" * 60)

model = PPO.load(MODEL_PATH, device='cpu')
results = []

for game_num in range(1, NUM_GAMES + 1):
    print(f"\n🎮 Game {game_num}/{NUM_GAMES}...")
    
    opponent = BasedAgent()
    env = WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=True)
    env.max_timesteps = 30 * 90
    
    observations, _ = env.reset()
    obs = observations[0]
    opponent.get_env_info(env)
    opponent_obs = observations[1]
    
    obs_shape = (obs.shape[0],)
    frame_stack = FrameStack(FRAME_STACK, obs_shape)
    stacked_obs = frame_stack.reset(obs)
    
    video_path = f"./game_{game_num}.mp4"
    writer = skvideo.io.FFmpegWriter(video_path, outputdict={
        '-vcodec': 'libx264',
        '-pix_fmt': 'yuv420p',
        '-preset': 'fast',
        '-crf': '20',
        '-r': '30'
    })
    
    for step in range(env.max_timesteps):
        action, _ = model.predict(stacked_obs, deterministic=True)
        opp_action = opponent.predict(opponent_obs)
        
        full_action = {0: action, 1: opp_action}
        observations, rewards, terminated, truncated, _ = env.step(full_action)
        
        obs = observations[0]
        opponent_obs = observations[1]
        stacked_obs = frame_stack.step(obs)
        
        img = env.render()
        img = np.rot90(img, k=-1)
        img = np.fliplr(img)
        writer.writeFrame(img)
        
        if terminated or truncated:
            break
    
    writer.close()
    
    stats = env.get_stats(0)
    opp_stats = env.get_stats(1)
    
    if stats.lives_left > opp_stats.lives_left:
        result = "WIN"
        score = stats.lives_left * 10 + stats.damage_done
    elif stats.lives_left < opp_stats.lives_left:
        result = "LOSS"
        score = -((3 - stats.lives_left) * 10) + stats.damage_done
    else:
        result = "DRAW"
        score = stats.damage_done - opp_stats.damage_done
    
    results.append({
        'game': game_num,
        'result': result,
        'lives': stats.lives_left,
        'opp_lives': opp_stats.lives_left,
        'damage_dealt': stats.damage_done,
        'damage_taken': stats.damage_taken,
        'score': score,
        'video': video_path
    })
    
    print(f"   {result}: Lives {stats.lives_left}-{opp_stats.lives_left}, Dmg {stats.damage_done:.0f}/{stats.damage_taken:.0f}")
    env.close()

# Find best game
best = max(results, key=lambda x: x['score'])

print("\n" + "=" * 60)
print("📊 ALL RESULTS")
print("=" * 60)
for r in results:
    marker = "⭐" if r == best else "  "
    print(f"{marker} Game {r['game']}: {r['result']} | Lives: {r['lives']}-{r['opp_lives']} | Dmg: {r['damage_dealt']:.0f} dealt, {r['damage_taken']:.0f} taken")

print("\n" + "=" * 60)
print(f"🏆 BEST GAME: Game {best['game']} ({best['result']})")
print(f"📹 Video: {best['video']}")
print("=" * 60)
