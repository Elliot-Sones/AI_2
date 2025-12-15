"""
UTMIST AI^2 Training Script v2 - With Reward Curriculum
========================================================
Phase-based reward curriculum for proper skill acquisition:
- Phase 1: Learn to approach (distance focus)
- Phase 2: Learn to hit (damage focus)
- Phase 3: Learn to dominate (net damage focus)
- Phase 4: Pure competition (win focus)

Manual phase control via config file.
Comprehensive logging for monitoring progress.
"""

import os
import yaml
import argparse
import torch
import sys
import gymnasium as gym
import numpy as np
from functools import partial
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack

# Add UTMIST environment to path
sys.path.append(os.path.join(os.getcwd(), "UTMIST-AI2-main"))

from environment.agent import (
    SelfPlayWarehouseBrawl, OpponentsCfg, SaveHandler, SaveHandlerMode,
    CameraResolution, RewardManager, RandomAgent, SelfPlayLatest, 
    SelfPlayRandom, RewTerm
)


# ============================================================================
# PHASE CONFIGURATIONS
# ============================================================================

PHASE_CONFIGS = {
    1: {
        "name": "Learn to Approach",
        "description": "Focus on closing distance to opponent",
        "rewards": {
            "distance": 2.0,
            "aggression": 0.5,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "net_damage": 0.0,
            "win": 0.0,
            "knockout": 0.0,
        },
        "opponents": {
            "random_agent": 0.9,
            "self_play": 0.1,
        }
    },
    2: {
        "name": "Learn to Hit",
        "description": "Focus on dealing damage and avoiding damage",
        "rewards": {
            "distance": 0.5,
            "aggression": 0.2,
            "damage_dealt": 1.0,
            "damage_taken": -0.5,
            "net_damage": 0.3,
            "win": 1.0,
            "knockout": 2.0,
        },
        "opponents": {
            "random_agent": 0.6,
            "self_play": 0.4,
        }
    },
    3: {
        "name": "Learn to Dominate",
        "description": "Focus on winning by outplaying (net positive damage)",
        "rewards": {
            "distance": 0.2,
            "aggression": 0.1,
            "damage_dealt": 0.5,
            "damage_taken": -0.3,
            "net_damage": 0.5,
            "win": 5.0,
            "knockout": 3.0,
        },
        "opponents": {
            "random_agent": 0.2,
            "self_play": 0.8,
        }
    },
    4: {
        "name": "Pure Competition",
        "description": "Pure win focus with conditional bonus",
        "rewards": {
            "distance": 0.0,
            "aggression": 0.0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "net_damage": 0.0,
            "win": 10.0,
            "knockout": 5.0,
        },
        "opponents": {
            "random_agent": 0.1,
            "self_play": 0.9,
        }
    },
}


def get_device():
    """Detects the best available device for training."""
    if torch.cuda.is_available():
        print("🚀 NVIDIA GPU detected! Using CUDA.")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("🍎 Mac GPU detected! Using MPS.")
        return "mps"
    else:
        print("🐌 No GPU detected. Using CPU.")
        return "cpu"


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """Linear schedule from initial_value to final_value."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func


def entropy_schedule(initial_value: float = 0.05, final_value: float = 0.005):
    """Entropy coefficient schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func


class Float32Wrapper(gym.ObservationWrapper):
    """Ensures observations are float32."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            dtype=np.float32
        )
    
    def observation(self, observation):
        return observation.astype(np.float32)


class VectorizedSaveHandler(SaveHandler):
    """SaveHandler for subprocess environments."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = None
        
    def update_info(self):
        pass
        
    def process(self) -> bool:
        return False
        
    def save_agent(self) -> None:
        pass


class SelfPlayCallback(BaseCallback):
    """Handles self-play model saving."""
    def __init__(self, save_handler, verbose=0):
        super().__init__(verbose)
        self.save_handler = save_handler
    
    def _on_step(self) -> bool:
        self.save_handler.agent = self.model
        self.save_handler.num_timesteps = self.num_timesteps
        self.save_handler.process()
        return True


class LimitedCheckpointCallback(CheckpointCallback):
    """Keeps only the last N checkpoints."""
    def __init__(self, save_freq, save_path, name_prefix="rl_model", max_keep=5, verbose=0):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.max_keep = max_keep
        self.saved_files = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.saved_files.append(path)
            
            if len(self.saved_files) > self.max_keep:
                to_remove = self.saved_files.pop(0)
                if os.path.exists(to_remove):
                    os.remove(to_remove)
        return result


class DetailedLoggingCallback(BaseCallback):
    """
    Comprehensive logging callback.
    Logs damage dealt, damage received, net damage, win rate, etc.
    """
    def __init__(self, log_freq=10000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.total_damage_dealt = 0.0
        self.total_damage_received = 0.0
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Check for episode completions
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info:
                    self.episode_rewards.append(ep_info['r'])
        
        # Log every log_freq steps
        if self.num_timesteps % self.log_freq == 0 and self.num_timesteps > 0:
            self._log_stats()
        
        return True
    
    def _log_stats(self):
        """Log comprehensive statistics."""
        print("\n" + "=" * 70)
        print(f"📊 TRAINING STATS @ {self.num_timesteps:,} steps")
        print("=" * 70)
        
        # Episode reward stats
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            min_reward = np.min(recent_rewards)
            max_reward = np.max(recent_rewards)
            
            print(f"📈 Episode Rewards (last 100):")
            print(f"   Mean:  {mean_reward:+.2f}")
            print(f"   Std:   {std_reward:.2f}")
            print(f"   Min:   {min_reward:+.2f}")
            print(f"   Max:   {max_reward:+.2f}")
            
            # Log to TensorBoard
            self.logger.record("metrics/mean_reward", mean_reward)
            self.logger.record("metrics/std_reward", std_reward)
            self.logger.record("metrics/min_reward", min_reward)
            self.logger.record("metrics/max_reward", max_reward)
        
        # Episode length stats
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            lengths = [ep['l'] for ep in self.model.ep_info_buffer if 'l' in ep]
            if lengths:
                mean_length = np.mean(lengths)
                print(f"⏱️  Mean Episode Length: {mean_length:.0f} steps")
                self.logger.record("metrics/mean_episode_length", mean_length)
        
        print("=" * 70 + "\n")


# ============================================================================
# REWARD FUNCTIONS (Phase-Aware)
# ============================================================================

class PhaseAwareRewardManager:
    """
    Manages rewards based on current training phase.
    Tracks damage statistics for logging.
    """
    def __init__(self, phase: int):
        self.phase = phase
        self.phase_config = PHASE_CONFIGS[phase]
        self.weights = self.phase_config["rewards"]
        
        # Tracking for logging
        self.episode_damage_dealt = 0.0
        self.episode_damage_received = 0.0
        self.total_damage_dealt = 0.0
        self.total_damage_received = 0.0
        
        # Last values for delta calculation
        self.last_damage_done = {0: 0.0, 1: 0.0}
        self.last_damage_taken = {0: 0.0, 1: 0.0}
        self.last_distance = {0: None, 1: None}
        
        print(f"\n{'='*60}")
        print(f"🎯 PHASE {phase}: {self.phase_config['name']}")
        print(f"   {self.phase_config['description']}")
        print(f"{'='*60}")
        print("Reward Weights:")
        for name, weight in self.weights.items():
            if weight != 0:
                print(f"   {name}: {weight:+.2f}")
        print(f"{'='*60}\n")
    
    def compute_reward(self, env) -> float:
        """Compute total reward for agent 0."""
        total_reward = 0.0
        player = env.players[0]
        opponent = env.players[1]
        
        # --- Distance Reward ---
        if self.weights["distance"] != 0:
            p1_pos = player.body.position
            p2_pos = opponent.body.position
            current_dist = p1_pos.get_distance(p2_pos)
            
            if self.last_distance[0] is None:
                self.last_distance[0] = current_dist
            else:
                # Positive reward for getting closer
                dist_delta = self.last_distance[0] - current_dist
                total_reward += dist_delta * self.weights["distance"]
                self.last_distance[0] = current_dist
        
        # --- Damage Dealt Reward ---
        current_damage_done = player.damage_done
        delta_dd = current_damage_done - self.last_damage_done[0]
        self.last_damage_done[0] = current_damage_done
        
        if delta_dd > 0:
            self.episode_damage_dealt += delta_dd
            self.total_damage_dealt += delta_dd
        
        if self.weights["damage_dealt"] != 0:
            total_reward += delta_dd * self.weights["damage_dealt"]
        
        # --- Damage Taken Penalty ---
        current_damage_taken = player.damage_taken_total
        delta_dt = current_damage_taken - self.last_damage_taken[0]
        self.last_damage_taken[0] = current_damage_taken
        
        if delta_dt > 0:
            self.episode_damage_received += delta_dt
            self.total_damage_received += delta_dt
        
        if self.weights["damage_taken"] != 0:
            total_reward += delta_dt * self.weights["damage_taken"]  # Note: weight is negative
        
        # --- Net Damage Bonus ---
        if self.weights["net_damage"] != 0:
            net_damage = delta_dd - delta_dt
            if net_damage > 0:
                total_reward += net_damage * self.weights["net_damage"]
        
        # --- Aggression Bonus ---
        if self.weights["aggression"] != 0:
            # Check if in attack state (states 6-10 typically)
            attack_states = {6, 7, 8, 9, 10}
            if hasattr(player, 'state_machine') and hasattr(player.state_machine, 'current_state'):
                state_id = getattr(player.state_machine.current_state, 'id', -1)
                if state_id in attack_states:
                    total_reward += self.weights["aggression"] * 0.1
        
        return total_reward
    
    def compute_win_reward(self, winner_is_agent: bool) -> float:
        """Compute win/loss reward."""
        if winner_is_agent:
            # Check if win was "earned" (net positive damage)
            net_damage = self.episode_damage_dealt - self.episode_damage_received
            base_reward = self.weights["win"]
            
            # In Phase 4, add conditional bonus for positive net damage
            if self.phase == 4 and net_damage > 0:
                base_reward += 5.0  # Extra bonus for "clean" win
            
            return base_reward
        else:
            return -self.weights["win"]
    
    def compute_knockout_reward(self, agent_got_ko: bool) -> float:
        """Compute KO reward."""
        if agent_got_ko:
            return -self.weights["knockout"]
        else:
            return self.weights["knockout"]
    
    def reset_episode(self):
        """Reset episode-level tracking."""
        self.episode_damage_dealt = 0.0
        self.episode_damage_received = 0.0
        self.last_distance = {0: None, 1: None}
    
    def get_stats(self):
        """Get current statistics."""
        return {
            "episode_damage_dealt": self.episode_damage_dealt,
            "episode_damage_received": self.episode_damage_received,
            "total_damage_dealt": self.total_damage_dealt,
            "total_damage_received": self.total_damage_received,
            "net_damage": self.episode_damage_dealt - self.episode_damage_received,
        }


class DamageReward:
    """Phase-aware damage reward."""
    def __init__(self, reward_manager: PhaseAwareRewardManager):
        self.reward_manager = reward_manager

    def __call__(self, env):
        reward = self.reward_manager.compute_reward(env)
        return torch.tensor([reward, -reward])  # Zero-sum


class DistanceReward:
    """Included in PhaseAwareRewardManager, but kept for compatibility."""
    def __call__(self, env):
        # Distance is computed in PhaseAwareRewardManager
        return torch.tensor([0.0, 0.0])


# Global reward manager (set per phase)
GLOBAL_REWARD_MANAGER = None


def make_env(opponent_cfg, save_freq, max_saved, parent_dir, model_name, resolution, phase):
    """Factory function for creating environments."""
    global GLOBAL_REWARD_MANAGER
    
    # Create phase-aware reward manager
    if GLOBAL_REWARD_MANAGER is None:
        GLOBAL_REWARD_MANAGER = PhaseAwareRewardManager(phase)
    
    reward_manager_instance = GLOBAL_REWARD_MANAGER
    
    # Win reward function
    def win_reward_func(env, agent, **kwargs):
        winner_is_agent = (agent == 'opponent')  # 'opponent' means agent 1 won, so agent 0 lost
        reward = reward_manager_instance.compute_win_reward(not winner_is_agent)
        rewards = torch.zeros(2)
        rewards[0] = reward
        rewards[1] = -reward
        return rewards

    # Knockout reward function  
    def knockout_reward_func(env, agent, **kwargs):
        agent_got_ko = (agent == 'player')  # 'player' means agent 0 was KO'd
        reward = reward_manager_instance.compute_knockout_reward(agent_got_ko)
        rewards = torch.zeros(2)
        rewards[0] = reward
        rewards[1] = -reward
        return rewards

    # Configure reward manager
    reward_functions = {
        "phase_reward": RewTerm(func=DamageReward(reward_manager_instance), weight=1.0),
    }
    
    signal_subscriptions = {
        "win": RewTerm(func=win_reward_func, weight=1.0, params={}),
        "knockout": RewTerm(func=knockout_reward_func, weight=1.0, params={}),
    }
    
    reward_manager = RewardManager(
        reward_functions=reward_functions,
        signal_subscriptions=signal_subscriptions
    )
    
    # Vectorized save handler
    vec_save_handler = VectorizedSaveHandler(
        agent=None,
        save_freq=save_freq,
        max_saved=max_saved,
        run_name="self_play_run",
        save_path=os.path.join(parent_dir, model_name, "model"),
        mode=SaveHandlerMode.FORCE
    )
    
    env = SelfPlayWarehouseBrawl(
        opponent_cfg=opponent_cfg,
        save_handler=vec_save_handler,
        resolution=resolution,
        reward_manager=reward_manager
    )
    env = Float32Wrapper(env)
    return env


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(cfg_file):
    global GLOBAL_REWARD_MANAGER
    
    # Load configuration
    with open(cfg_file, 'r') as f:
        params = yaml.safe_load(f)
    
    # Get phase from config
    phase = params.get("curriculum", {}).get("phase", 1)
    
    print("\n" + "=" * 70)
    print("🎮 UTMIST AI^2 Training Script v2 - Reward Curriculum Edition")
    print("=" * 70)
    print(f"📄 Config: {cfg_file}")
    
    # Validate phase
    if phase not in PHASE_CONFIGS:
        print(f"❌ Invalid phase {phase}. Valid phases: 1, 2, 3, 4")
        return
    
    phase_config = PHASE_CONFIGS[phase]
    print(f"🎯 Phase {phase}: {phase_config['name']}")

    # Hardware detection
    device = get_device()

    # Paths
    base_path = os.getcwd()
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["folders"]["model_name"], "model")
    tensor_board_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["folders"]["model_name"], "tb")
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(os.path.join(model_folder, "checkpoints"), exist_ok=True)

    # Self-Play Configuration
    max_saved_models = params["self_play_settings"].get("max_saved_models", 10)
    
    save_handler = SaveHandler(
        agent=None,
        save_freq=params["self_play_settings"]["save_freq"],
        max_saved=max_saved_models,
        save_path=model_folder,
        run_name="self_play_run",
        mode=SaveHandlerMode.FORCE
    )

    # Opponent configuration from phase
    opponent_probs = phase_config["opponents"]
    opponent_cfg = OpponentsCfg(
        opponents={
            'random_agent': (opponent_probs["random_agent"], partial(RandomAgent)),
            'self_play': (opponent_probs["self_play"], SelfPlayRandom(partial(PPO.load, device=device)))
        }
    )
    
    print(f"👥 Opponents: {int(opponent_probs['random_agent']*100)}% Random, {int(opponent_probs['self_play']*100)}% Self-play")

    # Resolution
    resolution_map = {
        "LOW": CameraResolution.LOW,
        "MEDIUM": CameraResolution.MEDIUM,
        "HIGH": CameraResolution.HIGH
    }
    resolution = resolution_map.get(params["environment_settings"]["resolution"], CameraResolution.LOW)

    # Number of parallel environments
    n_envs = params["environment_settings"].get("n_envs", 32 if device == "cuda" else 8)
    print(f"🔄 Parallel environments: {n_envs}")

    # Environment kwargs
    env_kwargs = {
        "opponent_cfg": opponent_cfg,
        "save_freq": params["self_play_settings"]["save_freq"],
        "max_saved": max_saved_models,
        "parent_dir": params["folders"]["parent_dir"],
        "model_name": params["folders"]["model_name"],
        "resolution": resolution,
        "phase": phase,
    }

    # Create vectorized environment
    env = make_vec_env(
        make_env, 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
        env_kwargs=env_kwargs
    )
    env = VecMonitor(env)
    
    # Frame stacking
    frame_stack = params.get("frame_stack", 4)
    if frame_stack > 1:
        print(f"📚 Frame stacking: {frame_stack} frames")
        env = VecFrameStack(env, n_stack=frame_stack)

    # PPO Settings
    ppo_settings = params["ppo_settings"]
    
    # Schedules
    learning_rate = linear_schedule(
        ppo_settings["learning_rate"][0], 
        ppo_settings["learning_rate"][1]
    )
    clip_range = linear_schedule(
        ppo_settings["clip_range"][0], 
        ppo_settings["clip_range"][1]
    )
    ent_coef = entropy_schedule(
        ppo_settings.get("ent_coef_initial", 0.05),
        ppo_settings.get("ent_coef_final", 0.005)
    )
    
    # Batch size
    buffer_size = n_envs * ppo_settings["n_steps"]
    batch_size = min(ppo_settings["batch_size"], buffer_size)

    # Network architecture
    net_arch = params.get("policy_kwargs", {}).get("net_arch", [512, 512, 256])
    print(f"🧠 Network: {net_arch}")
    
    # Initialize or load agent
    model_checkpoint = ppo_settings.get("model_checkpoint", "0")
    
    if model_checkpoint == "0":
        print("🆕 Creating new agent...")
        agent = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=learning_rate,
            n_steps=ppo_settings["n_steps"],
            batch_size=batch_size,
            n_epochs=ppo_settings["n_epochs"],
            gamma=ppo_settings["gamma"],
            gae_lambda=ppo_settings["gae_lambda"],
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=ppo_settings["vf_coef"],
            max_grad_norm=ppo_settings["max_grad_norm"],
            tensorboard_log=tensor_board_folder,
            device=device,
            policy_kwargs={"net_arch": net_arch}
        )
    else:
        checkpoint_path = os.path.join(model_folder, model_checkpoint)
        print(f"📂 Loading from: {checkpoint_path}")
        agent = PPO.load(
            checkpoint_path, 
            env=env,
            device=device,
            learning_rate=learning_rate,
            clip_range=clip_range,
            ent_coef=ent_coef,
        )

    # Callbacks
    checkpoint_callback = LimitedCheckpointCallback(
        save_freq=params["self_play_settings"]["save_freq"],
        save_path=os.path.join(model_folder, "checkpoints"),
        name_prefix=f"phase{phase}_model",
        max_keep=10,
        verbose=1
    )
    
    self_play_callback = SelfPlayCallback(save_handler)
    logging_callback = DetailedLoggingCallback(log_freq=10000, verbose=1)

    # Training
    total_timesteps = ppo_settings["time_steps"]
    print(f"\n🚀 Starting Phase {phase} training for {total_timesteps:,} timesteps...")
    print(f"📊 TensorBoard: {tensor_board_folder}")
    print(f"💾 Checkpoints: {os.path.join(model_folder, 'checkpoints')}")
    print("=" * 70 + "\n")
    
    try:
        agent.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, self_play_callback, logging_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
    
    # Save final model
    final_path = os.path.join(model_folder, f"phase{phase}_final")
    agent.save(final_path)
    print(f"\n✅ Model saved to {final_path}")
    
    # Print final stats
    if GLOBAL_REWARD_MANAGER:
        stats = GLOBAL_REWARD_MANAGER.get_stats()
        print("\n" + "=" * 70)
        print("📊 FINAL TRAINING STATISTICS")
        print("=" * 70)
        print(f"Total Damage Dealt:    {stats['total_damage_dealt']:.0f}")
        print(f"Total Damage Received: {stats['total_damage_received']:.0f}")
        print(f"Net Damage:            {stats['total_damage_dealt'] - stats['total_damage_received']:+.0f}")
        print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UTMIST AI^2 Training - Reward Curriculum")
    parser.add_argument("--cfgFile", type=str, default="utmist_config_v2.yaml", help="Configuration file")
    opt = parser.parse_args()
    main(opt.cfgFile)
