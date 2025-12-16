#!/usr/bin/env python3
"""
Run 10 games against each opponent for comprehensive evaluation.
"""

import sys
sys.path.insert(0, 'UTMIST-AI2-main')

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import numpy as np
from environment.environment import CameraResolution, WarehouseBrawl
from environment.agent import RandomAgent, BasedAgent, ConstantAgent, ClockworkAgent
from stable_baselines3 import PPO

# Config
MODEL_PATH = "./results/ppo_utmist_v2/model/phase2_final.zip"
NUM_GAMES = 10
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
print(f"📊 COMPREHENSIVE EVALUATION ({NUM_GAMES} GAMES EACH)")
print("=" * 60)
print(f"📂 Model: {MODEL_PATH}")
print("=" * 60)

model = PPO.load(MODEL_PATH, device='cpu')

# All opponents to test
opponents_config = {
    "ConstantAgent": ConstantAgent,
    "RandomAgent": RandomAgent,
    "ClockworkAgent": ClockworkAgent,
    "BasedAgent": BasedAgent,
}

# Try to add old models
for phase_name in ["phase1_final", "phase2_final"]:
    model_path = f"./results/ppo_utmist_v2/model/{phase_name}.zip"
    if os.path.exists(model_path):
        opponents_config[phase_name] = ("model", model_path)

all_results = {}

for opp_name, opp_config in opponents_config.items():
    print(f"\n🎮 Testing vs {opp_name}...")
    
    wins = 0
    losses = 0
    draws = 0
    total_dmg_dealt = 0
    total_dmg_received = 0
    
    for game in range(NUM_GAMES):
        # Create opponent
        if isinstance(opp_config, tuple) and opp_config[0] == "model":
            opp_model = PPO.load(opp_config[1], device='cpu')
            class ModelOpponent:
                def __init__(self, m):
                    self.model = m
                    self.frame_stack = None
                def get_env_info(self, env):
                    self.frame_stack = [np.zeros(64,) for _ in range(4)]
                def predict(self, obs):
                    if self.frame_stack is None:
                        return np.zeros(10)
                    self.frame_stack.pop(0)
                    self.frame_stack.append(obs.copy())
                    stacked = np.concatenate(self.frame_stack, axis=0)
                    action, _ = self.model.predict(stacked, deterministic=True)
                    return action
            opponent = ModelOpponent(opp_model)
        else:
            opponent = opp_config()
        
        env = WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=True)
        env.max_timesteps = 30 * 90
        
        observations, _ = env.reset()
        obs = observations[0]
        opponent.get_env_info(env)
        opponent_obs = observations[1]
        
        obs_shape = (obs.shape[0],)
        frame_stack = FrameStack(FRAME_STACK, obs_shape)
        stacked_obs = frame_stack.reset(obs)
        
        for step in range(env.max_timesteps):
            action, _ = model.predict(stacked_obs, deterministic=True)
            opp_action = opponent.predict(opponent_obs)
            
            full_action = {0: action, 1: opp_action}
            observations, rewards, terminated, truncated, _ = env.step(full_action)
            
            obs = observations[0]
            opponent_obs = observations[1]
            stacked_obs = frame_stack.step(obs)
            
            if terminated or truncated:
                break
        
        stats = env.get_stats(0)
        opp_stats = env.get_stats(1)
        
        total_dmg_dealt += stats.damage_done
        total_dmg_received += stats.damage_taken
        
        if stats.lives_left > opp_stats.lives_left:
            wins += 1
            result = "W"
        elif stats.lives_left < opp_stats.lives_left:
            losses += 1
            result = "L"
        else:
            if stats.damage_done > opp_stats.damage_done:
                wins += 1
                result = "W"
            else:
                draws += 1
                result = "D"
        
        print(f"   Game {game+1}: {result} | Lives: {stats.lives_left}-{opp_stats.lives_left} | Dmg: {stats.damage_done:.0f}/{stats.damage_taken:.0f}")
        env.close()
    
    win_rate = (wins / NUM_GAMES) * 100
    avg_dealt = total_dmg_dealt / NUM_GAMES
    avg_received = total_dmg_received / NUM_GAMES
    
    all_results[opp_name] = {
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_dealt": avg_dealt,
        "avg_received": avg_received
    }

print("\n" + "=" * 60)
print("📊 FINAL RESULTS (10 GAMES EACH)")
print("=" * 60)
print(f"{'Opponent':<20} {'Win Rate':>10} {'Record':>12} {'Dmg Dealt':>10} {'Dmg Recv':>10}")
print("-" * 60)

for opp_name, result in all_results.items():
    record = f"{result['wins']}W/{result['losses']}L/{result['draws']}D"
    print(f"{opp_name:<20} {result['win_rate']:>9.0f}% {record:>12} {result['avg_dealt']:>10.0f} {result['avg_received']:>10.0f}")

print("=" * 60)
