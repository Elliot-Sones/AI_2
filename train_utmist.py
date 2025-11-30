import os
import yaml
import argparse
import torch
import sys
import gymnasium as gym
import numpy as np
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Add UTMIST environment to path
sys.path.append(os.path.join(os.getcwd(), "UTMIST-AI2-main"))

from environment.agent import SelfPlayWarehouseBrawl, OpponentsCfg, SaveHandler, SaveHandlerMode, CameraResolution, RewardManager, RandomAgent, SelfPlayLatest

def get_device():
    """Detects the best available device for training."""
    if torch.cuda.is_available():
        print("🚀 NVIDIA GPU detected! Using CUDA.")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("🍎 Mac GPU detected! Using MPS (Metal Performance Shaders).")
        return "mps"
    else:
        print("🐌 No GPU detected. Using CPU.")
        return "cpu"

def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return progress_remaining * (initial_value - final_value) + final_value

    return func

class Float32Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            dtype=np.float32
        )
    
    def observation(self, observation):
        return observation.astype(np.float32)

# Vectorized SaveHandler for subprocesses
# It disables saving (handled by main process) but allows loading opponents
class VectorizedSaveHandler(SaveHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = None # Agent is not available in subprocess
        
    def update_info(self):
        pass
        
    def process(self) -> bool:
        # Do nothing in subprocess
        return False
        
    def save_agent(self) -> None:
        # Do nothing in subprocess
        pass

# Callback to handle self-play saving in the main process
class SelfPlayCallback(BaseCallback):
    def __init__(self, save_handler, verbose=0):
        super().__init__(verbose)
        self.save_handler = save_handler
    
    def _on_step(self) -> bool:
        # Update num_timesteps in save_handler
        self.save_handler.agent = self.model
        self.save_handler.num_timesteps = self.num_timesteps
        self.save_handler.process()
        return True

# Callback to limit the number of checkpoints saved
class LimitedCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq, save_path, name_prefix="rl_model", max_keep=5, verbose=0):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.max_keep = max_keep
        self.saved_files = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            # Track the newly saved file
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.saved_files.append(path)
            
            # Remove old files if we exceed the limit
            if len(self.saved_files) > self.max_keep:
                to_remove = self.saved_files.pop(0)
                if os.path.exists(to_remove):
                    if self.verbose > 0:
                        print(f"Removing old checkpoint: {to_remove}")
                    os.remove(to_remove)
        return result

# Factory function for environment creation (must be picklable)
def make_env(opponent_cfg, save_freq, max_saved, parent_dir, model_name, resolution):
    # Define Reward Functions
    class DamageReward:
        def __init__(self):
            self.last_damage_done = {0: 0.0, 1: 0.0}
            self.last_damage_taken = {0: 0.0, 1: 0.0}

        def __call__(self, env):
            rewards = []
            for i in range(2):
                player = env.players[i]
                # Damage Done Reward
                dd = player.damage_done
                delta_dd = dd - self.last_damage_done[i]
                self.last_damage_done[i] = dd
                
                # Damage Taken Penalty
                dt = player.damage_taken_total
                delta_dt = dt - self.last_damage_taken[i]
                self.last_damage_taken[i] = dt
                
                # Net reward: +1 per damage dealt, -0.5 per damage taken
                reward = (delta_dd * 0.1) - (delta_dt * 0.05) 
                rewards.append(reward)
            return torch.tensor(rewards)

    def win_reward_func(env, agent, **kwargs):
        rewards = torch.zeros(2)
        if agent == 'player': # Agent 0 won
            rewards[0] = 10.0
            rewards[1] = -10.0
        else: # Agent 1 (opponent) won
            rewards[0] = -10.0
            rewards[1] = 10.0
        return rewards

    def knockout_reward_func(env, agent, **kwargs):
        rewards = torch.zeros(2)
        if agent == 'player': # Agent 0 was KO'd
            rewards[0] = -5.0
            rewards[1] = 5.0
        else: # Agent 1 was KO'd
            rewards[0] = 5.0
            rewards[1] = -5.0
        return rewards

    # Configure RewardManager
    from environment.agent import RewTerm
    
    damage_reward = DamageReward()
    
    reward_functions = {
        "damage": RewTerm(func=damage_reward, weight=1.0)
    }
    
    signal_subscriptions = {
        "win": RewTerm(func=win_reward_func, weight=1.0, params={}),
        "knockout": RewTerm(func=knockout_reward_func, weight=1.0, params={})
    }
    
    reward_manager = RewardManager(
        reward_functions=reward_functions,
        signal_subscriptions=signal_subscriptions
    )
    
    # Use VectorizedSaveHandler in the environment
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

def main(cfg_file):
    # Read the cfg file
    with open(cfg_file, 'r') as f:
        params = yaml.safe_load(f)
    
    print(f"Loaded config from {cfg_file}")

    # Hardware detection
    device = get_device()
    print(f"Training on device: {device}")

    # Paths
    base_path = os.getcwd()
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["folders"]["model_name"], "model")
    tensor_board_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["folders"]["model_name"], "tb")
    os.makedirs(model_folder, exist_ok=True)

    # Self-Play Configuration (Main Process)
    save_handler = SaveHandler(
        agent=None, # Will be set after agent creation
        save_freq=params["self_play_settings"]["save_freq"],
        max_saved=params["self_play_settings"]["max_saved_models"],
        save_path=model_folder,
        run_name="self_play_run",
        mode=SaveHandlerMode.FORCE # Force to create new run if not exists
    )

    opponent_cfg = OpponentsCfg(
        opponents={
            'random_agent': (params["self_play_settings"]["opponent_selection"]["random_agent"], partial(RandomAgent)),
            'self_play': (params["self_play_settings"]["opponent_selection"]["self_play"], SelfPlayLatest(partial(PPO.load, device=device))) 
        }
    )

    # Create Environment
    resolution_map = {
        "LOW": CameraResolution.LOW,
        "MEDIUM": CameraResolution.MEDIUM,
        "HIGH": CameraResolution.HIGH
    }
    resolution = resolution_map.get(params["environment_settings"]["resolution"], CameraResolution.LOW)

    # Auto-scale n_envs based on device if not specified in config
    if "n_envs" in params["environment_settings"]:
        n_envs = params["environment_settings"]["n_envs"]
        print(f"Using configured n_envs: {n_envs}")
    else:
        if device == "cuda":
            n_envs = 16
            print("🚀 CUDA detected! Auto-scaling to 16 parallel environments.")
        elif device == "mps":
            n_envs = 4
            print("🍎 MPS detected! Auto-scaling to 4 parallel environments.")
        else:
            n_envs = 1
            print("Using default n_envs: 1")

    print(f"Creating {n_envs} vectorized environments...")
    
    # Create vectorized environment
    # We use SubprocVecEnv for true parallelism (separate processes)
    # We wrap with VecMonitor to log stats (reward, length)
    
    # Prepare arguments for make_env factory
    env_kwargs = {
        "opponent_cfg": opponent_cfg,
        "save_freq": params["self_play_settings"]["save_freq"],
        "max_saved": params["self_play_settings"]["max_saved_models"],
        "parent_dir": params["folders"]["parent_dir"],
        "model_name": params["folders"]["model_name"],
        "resolution": resolution
    }

    # Determine start method
    start_method = "spawn" if device == "cuda" else "fork"
    # Force spawn on Mac/Linux for consistency if desired, but fork is faster on Mac.
    # However, for CUDA, spawn is REQUIRED.
    # Let's use spawn for everything to be safe and consistent with "100% confidence".
    start_method = "spawn" 
    print(f"Using multiprocessing start method: {start_method}")

    env = make_vec_env(
        make_env, 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": start_method},
        env_kwargs=env_kwargs
    )
    env = VecMonitor(env)
    
    print("Vectorized environment created.")

    # PPO Settings
    ppo_settings = params["ppo_settings"]
    learning_rate = linear_schedule(ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    clip_range = linear_schedule(ppo_settings["clip_range"][0], ppo_settings["clip_range"][1])
    
    # Initialize Agent
    model_checkpoint = ppo_settings["model_checkpoint"]
    
    if model_checkpoint == "0":
        print("Initializing new agent...")
        agent = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=learning_rate,
            n_steps=ppo_settings["n_steps"],
            batch_size=ppo_settings["batch_size"],
            n_epochs=ppo_settings["n_epochs"],
            gamma=ppo_settings["gamma"],
            gae_lambda=ppo_settings["gae_lambda"],
            clip_range=clip_range,
            ent_coef=ppo_settings["ent_coef"],
            vf_coef=ppo_settings["vf_coef"],
            max_grad_norm=ppo_settings["max_grad_norm"],
            tensorboard_log=tensor_board_folder,
            device=device,
            policy_kwargs=params.get("policy_kwargs", None)
        )
    else:
        print(f"Loading agent from checkpoint {model_checkpoint}...")
        agent = PPO.load(
            os.path.join(model_folder, model_checkpoint), 
            env=env,
            device=device,
            # Update hyperparameters if needed
            learning_rate=learning_rate,
            clip_range=clip_range
        )

    # Callbacks
    # Use LimitedCheckpointCallback to avoid filling up disk
    checkpoint_callback = LimitedCheckpointCallback(
        save_freq=params["self_play_settings"]["save_freq"],
        save_path=os.path.join(model_folder, "checkpoints"),
        name_prefix="rl_model",
        max_keep=5 # Keep only last 5 permanent checkpoints
    )
    
    # Create the self-play callback with the REAL save_handler
    self_play_callback = SelfPlayCallback(save_handler)

    print(f"Starting training for {ppo_settings['time_steps']} timesteps...")
    
    agent.learn(
        total_timesteps=ppo_settings["time_steps"],
        callback=[checkpoint_callback, self_play_callback],
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(model_folder, "ppo_utmist_final")
    agent.save(final_path)
    print(f"Training finished. Model saved to {final_path}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, default="utmist_config.yaml", help="Configuration file")
    opt = parser.parse_args()
    main(opt.cfgFile)
