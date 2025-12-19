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
import torch
from functools import partial
from environment.environment import CameraResolution, WarehouseBrawl
from environment.agent import (
    SelfPlayWarehouseBrawl, OpponentsCfg, RewardManager, ConstantAgent, RewTerm
)
from train_utmist_v2 import (
    Float32Wrapper, OpponentHistoryWrapper, FrozenOpponentWrapper, 
    PhaseConfig, linear_schedule
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
import os


def make_nav_env(phase_config, render=True):
    """Create single navigation environment - EXACTLY MATCHING TRAINING"""
    def _init():
        # NULL reward function (navigation uses custom rewards)
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
        
        # Set render mode for VecVideoRecorder
        env.render_mode = "rgb_array"
        if not hasattr(env, 'metadata'):
            env.metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}
        
        # Apply SAME wrappers as training
        env = Float32Wrapper(env)
        env = FrozenOpponentWrapper(env, nav_config=phase_config.navigation, debug_logs=False)
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
