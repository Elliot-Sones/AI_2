#!/usr/bin/env python3
"""
Watch your trained agent fight against BasedAgent.
Saves a video file you can download and view.
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

# Path to your trained model (latest checkpoint from current training)
MODEL_PATH = "./results/ppo_utmist_v2/model/phase2_final.zip"
VIDEO_PATH = "./vs_basedagent.mp4"
FRAME_STACK = 4

print("=" * 60)
print("🎬 AGENT VS BASEDAGENT")
print("=" * 60)
print(f"📂 Loading model: {MODEL_PATH}")
print(f"📹 Video will be saved to: {VIDEO_PATH}")
print("=" * 60)

# Frame stacking helper
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

# Load model
model = PPO.load(MODEL_PATH, device='cpu')
frame_stack = None

# Create opponent
opponent = BasedAgent()

print("\n🚀 Running match against BasedAgent...")

# Run match
env = WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=True)
env.max_timesteps = 30 * 90  # 90 second match

observations, _ = env.reset()
obs = observations[0]
opponent.get_env_info(env)
opponent_obs = observations[1]

# Initialize frame stack
obs_shape = (obs.shape[0],)
frame_stack = FrameStack(FRAME_STACK, obs_shape)
stacked_obs = frame_stack.reset(obs)

# Video writer
writer = skvideo.io.FFmpegWriter(VIDEO_PATH, outputdict={
    '-vcodec': 'libx264',
    '-pix_fmt': 'yuv420p',
    '-preset': 'fast',
    '-crf': '20',
    '-r': '30'
})

for step in tqdm(range(env.max_timesteps), desc="Recording"):
    # Get actions
    action, _ = model.predict(stacked_obs, deterministic=True)
    opp_action = opponent.predict(opponent_obs)
    
    full_action = {0: action, 1: opp_action}
    observations, rewards, terminated, truncated, _ = env.step(full_action)
    
    obs = observations[0]
    opponent_obs = observations[1]
    stacked_obs = frame_stack.step(obs)
    
    # Capture frame
    img = env.render()
    img = np.rot90(img, k=-1)
    img = np.fliplr(img)
    writer.writeFrame(img)
    
    if terminated or truncated:
        break

writer.close()

# Get stats
stats = env.get_stats(0)
opp_stats = env.get_stats(1)

if stats.lives_left > opp_stats.lives_left:
    result = "WIN"
elif stats.lives_left < opp_stats.lives_left:
    result = "LOSS"
else:
    result = "DRAW"

env.close()

print("\n" + "=" * 60)
print("📊 MATCH RESULT vs BasedAgent")
print("=" * 60)
print(f"Result: {result}")
print(f"Your Agent - Lives: {stats.lives_left}, Damage Dealt: {stats.damage_done:.0f}, Damage Taken: {stats.damage_taken:.0f}")
print(f"BasedAgent - Lives: {opp_stats.lives_left}, Damage Dealt: {opp_stats.damage_done:.0f}, Damage Taken: {opp_stats.damage_taken:.0f}")
print("=" * 60)
print(f"\n✅ Video saved to: {VIDEO_PATH}")
