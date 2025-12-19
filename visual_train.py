#!/usr/bin/env python3
"""
Visual Training Script - Watch agent train in real-time
For short visual debugging sessions on hosted GPU

Usage:
    python visual_train.py --steps 10000  # Train for 10k steps with visual
"""

import sys
sys.path.insert(0, 'UTMIST-AI2-main')

import argparse
import yaml
import numpy as np
from environment.environment import CameraResolution, WarehouseBrawl
from train_utmist_v2 import (
    Float32Wrapper, OpponentHistoryWrapper, FrozenOpponentWrapper, 
    PhaseConfig, linear_schedule
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
import os

import gymnasium as gym

class ActionAdapterWrapper(gym.Wrapper):
    """
    Adapts multi-agent environment (WarehouseBrawl) to single-agent PPO interface.
    1. Input: Scalar action -> Dict {0: action, 1: dummy}
    2. Output: List obs -> Scalar obs[0]
    3. Output: Dict reward -> Scalar reward[0]
    """
    def step(self, action):
        if not isinstance(action, dict):
            action_dict = {0: action, 1: np.zeros(10, dtype=np.float32)}
        else:
            action_dict = action
            
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        
        # Adapt Observation (List/Dict -> Single)
        if isinstance(obs, (list, dict)):
            my_obs = obs[0]
        else:
            my_obs = obs
            
        # Adapt Reward (Dict -> Single)
        if isinstance(reward, dict):
            my_reward = reward[0]
        else:
            my_reward = reward
            
        return my_obs, my_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Adapt Observation on reset
        if isinstance(obs, (list, dict)):
            return obs[0], info
        return obs, info

def make_nav_env(phase_config, render=True):
    """Create single navigation environment"""
    def _init():
        # train_mode=True prevents window opening (headless friendly)
        env = WarehouseBrawl(
            resolution=CameraResolution.LOW, 
            train_mode=not render
        )
        # FORCE render_mode for VecVideoRecorder
        env.render_mode = "rgb_array"
        # Ensure metadata is present
        if not hasattr(env, 'metadata'):
            env.metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}
        
        env = Float32Wrapper(env)
        
        # Adapter converts Multi-Agent -> Single Agent interface
        env = ActionAdapterWrapper(env)
        
        # FrozenOpponentWrapper works on Single Agent interface (expects scalar reward/obs)
        env = FrozenOpponentWrapper(env, nav_config=phase_config.navigation, debug_logs=True)
        env = OpponentHistoryWrapper(env, zero_history=True)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Visual training for debugging")
    parser.add_argument('--steps', type=int, default=5000, help='Training steps')
    parser.add_argument('--phase', type=str, default='0d', help='Phase to train')
    parser.add_argument('--config', type=str, default='utmist_config_v2.yaml', help='Config file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎥 VISUAL TRAINING SESSION")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Steps: {args.steps:,}")
    print("=" * 60)
    
    # Load config
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
    
    phase_config = PhaseConfig(params, args.phase)
    
    # Create single environment (render mode)
    env = DummyVecEnv([make_nav_env(phase_config, render=True)])
    
    # Frame stack
    frame_stack = params.get('frame_stack', 4)
    env = VecFrameStack(env, n_stack=frame_stack)
    
    # Add Video Recorder to capture training behavior
    video_folder = "./videos"
    os.makedirs(video_folder, exist_ok=True)
    video_length = args.steps
    
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0, # Record from start
        video_length=video_length,
        name_prefix=f"training_session_{args.phase}"
    )
    
    # Model settings
    ppo = params.get("ppo_settings", {})
    model_checkpoint = ppo.get("model_checkpoint", "0")
    model_folder = f"./results/{params['folders']['model_name']}/model"
    
    # Load or create model
    if model_checkpoint != "0":
        checkpoint_path = os.path.join(model_folder, model_checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"📂 Loading: {checkpoint_path}")
            model = PPO.load(checkpoint_path, env=env, device='cpu')
        else:
            print(f"⚠️ Checkpoint not found, creating new agent")
            model_checkpoint = "0"
    
    if model_checkpoint == "0":
        print("🆕 Creating new agent")
        lr_config = ppo.get("learning_rate", [3e-4, 1e-6])
        learning_rate = linear_schedule(lr_config[0], lr_config[1])
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=256,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            ent_coef=0.1,
            verbose=1,
            device='cpu'
        )
    
    print(f"\n🎮 Training {args.steps:,} steps...")
    print(f"📹 Recording to {video_folder}/training_session_{args.phase}.mp4")
    
    # Train (video recorder will capture frames)
    model.learn(total_timesteps=args.steps)
    
    # Close env to save video
    env.close()
    
    print(f"\n✅ Training complete! Video saved to: {video_folder}/training_session_{args.phase}-step-0-to-step-{args.steps}.mp4")
    print("Note: This video shows the STOCHASTIC training behavior (trying new things), not the final deterministic policy.")

if __name__ == "__main__":
    main()
