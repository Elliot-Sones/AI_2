#!/usr/bin/env python3
"""
Play against your trained agent!

Controls:
  WASD - Move
  SPACE - Jump
  H - Light attack (horizontal)
  J - Heavy attack
  K - Special attack
  L - Alternative attack
  G - Grab/Throw
"""

import sys
sys.path.insert(0, 'UTMIST-AI2-main')

from environment.environment import CameraResolution
from environment.agent import run_real_time_match, SB3Agent
from stable_baselines3 import PPO
import pygame

pygame.init()

# Path to your trained model
MODEL_PATH = "./results/ppo_utmist_v2/model/phase2_final.zip"

print("=" * 60)
print("🎮 PLAY AGAINST YOUR AI AGENT")
print("=" * 60)
print(f"📂 Loading model: {MODEL_PATH}")
print()
print("Controls:")
print("  WASD  - Move")
print("  SPACE - Jump")  
print("  H     - Light attack")
print("  J     - Heavy attack")
print("  K     - Special attack")
print("  L     - Alternative attack")
print("  G     - Grab/Throw")
print("=" * 60)

# Load your trained agent
class TrainedAgent:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
        self.initialized = False
        
    def get_env_info(self, env):
        self.action_space = env.action_space
        self.obs_helper = env.obs_helper
        self.act_helper = env.act_helper
        self.initialized = True
        
    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

# User input agent
class UserInputAgent:
    def __init__(self):
        self.initialized = False
        
    def get_env_info(self, env):
        self.action_space = env.action_space
        self.obs_helper = env.obs_helper
        self.act_helper = env.act_helper
        self.initialized = True
        
    def predict(self, obs):
        action = self.act_helper.zeros()
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)
            
        return action

# Create agents
player = UserInputAgent()
opponent = TrainedAgent(MODEL_PATH)

print("\n🚀 Starting match...")
print("Press ESC or close window to quit\n")

# Run the match
try:
    match_result = run_real_time_match(
        agent_1=player,
        agent_2=opponent,
        max_timesteps=30 * 180,  # 3 minute match
        resolution=CameraResolution.LOW
    )
    
    print("\n" + "=" * 60)
    print("📊 MATCH RESULT")
    print("=" * 60)
    print(f"Result: {match_result.player1_result}")
    print(f"Your damage dealt: {match_result.player1.damage_done:.1f}")
    print(f"Your damage taken: {match_result.player1.damage_taken:.1f}")
    print(f"AI damage dealt: {match_result.player2.damage_done:.1f}")
    print(f"AI damage taken: {match_result.player2.damage_taken:.1f}")
    print("=" * 60)
    
except KeyboardInterrupt:
    print("\nMatch cancelled.")
finally:
    pygame.quit()
