#!/usr/bin/env python3
"""
Watch your trained agent fight against a random opponent.
Saves a video file you can download and view.
"""

import sys
sys.path.insert(0, 'UTMIST-AI2-main')

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import numpy as np
from environment.environment import CameraResolution, WarehouseBrawl
from environment.agent import RandomAgent, MatchStats, Result
from stable_baselines3 import PPO
from tqdm import tqdm
import skvideo.io

# Path to your trained model (latest available)
MODEL_PATH = "./results/ppo_utmist_v2/model/phase2_final.zip"
VIDEO_PATH = "./agent_demo.mp4"
FRAME_STACK = 4  # Must match training config

print("=" * 60)
print("🎬 WATCH YOUR AI AGENT FIGHT")
print("=" * 60)
print(f"📂 Loading model: {MODEL_PATH}")
print(f"📹 Video will be saved to: {VIDEO_PATH}")
print(f"📚 Frame stacking: {FRAME_STACK}")
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

# Load your trained agent
class TrainedAgent:
    def __init__(self, model_path, n_frames=4):
        self.model = PPO.load(model_path, device='cpu')
        self.n_frames = n_frames
        self.frame_stack = None
        self.initialized = False
        
    def get_env_info(self, env):
        self.action_space = env.action_space
        self.obs_helper = env.obs_helper
        self.act_helper = env.act_helper
        obs_shape = (env.observation_space.shape[0],)
        self.frame_stack = FrameStack(self.n_frames, obs_shape)
        self.initialized = True
        
    def reset(self, obs):
        return self.frame_stack.reset(obs)
    
    def predict(self, obs, is_reset=False):
        if is_reset:
            stacked_obs = self.frame_stack.reset(obs)
        else:
            stacked_obs = self.frame_stack.step(obs)
        action, _ = self.model.predict(stacked_obs, deterministic=True)
        return action

# Create agents  
trained_agent = TrainedAgent(MODEL_PATH, FRAME_STACK)
random_opponent = RandomAgent()

print("\n🚀 Running match (this may take a minute)...")

# Run match manually with frame stacking
env = WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=True)
env.max_timesteps = 30 * 90  # 90 second match

observations, infos = env.reset()
obs_1 = observations[0]
obs_2 = observations[1]

# Initialize agents
trained_agent.get_env_info(env)
random_opponent.get_env_info(env)

# Initialize frame stack
stacked_obs_1 = trained_agent.reset(obs_1)

# Video writer
writer = skvideo.io.FFmpegWriter(VIDEO_PATH, outputdict={
    '-vcodec': 'libx264',
    '-pix_fmt': 'yuv420p',
    '-preset': 'fast',
    '-crf': '20',
    '-r': '30'
})

for step in tqdm(range(env.max_timesteps), total=env.max_timesteps):
    # Get actions
    action_1 = trained_agent.predict(obs_1)
    action_2 = random_opponent.predict(obs_2)
    
    full_action = {0: action_1, 1: action_2}
    
    observations, rewards, terminated, truncated, info = env.step(full_action)
    obs_1 = observations[0]
    obs_2 = observations[1]
    
    # Capture frame
    img = env.render()
    img = np.rot90(img, k=-1)
    img = np.fliplr(img)
    writer.writeFrame(img)
    
    if terminated or truncated:
        break

writer.close()

# Get stats
player_1_stats = env.get_stats(0)
player_2_stats = env.get_stats(1)

if player_1_stats.lives_left > player_2_stats.lives_left:
    result = "WIN"
elif player_1_stats.lives_left < player_2_stats.lives_left:
    result = "LOSS"
else:
    result = "DRAW"

env.close()

print("\n" + "=" * 60)
print("📊 MATCH RESULT")
print("=" * 60)
print(f"Result: {result}")
print(f"Your Agent - Damage Dealt: {player_1_stats.damage_done:.1f}")
print(f"Your Agent - Damage Taken: {player_1_stats.damage_taken:.1f}")
print(f"Your Agent - Lives Left: {player_1_stats.lives_left}")
print(f"Random - Damage Dealt: {player_2_stats.damage_done:.1f}")
print(f"Random - Damage Taken: {player_2_stats.damage_taken:.1f}")
print(f"Random - Lives Left: {player_2_stats.lives_left}")
print("=" * 60)
print(f"\n✅ Video saved to: {VIDEO_PATH}")
print("Download it to watch your agent in action!")
