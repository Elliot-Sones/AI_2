import os
import yaml
import argparse
import torch
import sys
import gymnasium as gym
import numpy as np
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecVideoRecorder, DummyVecEnv

# Add UTMIST environment to path
sys.path.append(os.path.join(os.getcwd(), "UTMIST-AI2-main"))

from environment.agent import SelfPlayWarehouseBrawl, OpponentsCfg, SaveHandler, SaveHandlerMode, CameraResolution, RewardManager, RandomAgent, SelfPlayLatest, RewTerm

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

class StatsWrapper(gym.Wrapper):
    """
    Wrapper to track and log custom statistics:
    - Damage Dealt
    - Win/Loss (with Damage Tie-Breaker)
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            # Extract stats from the underlying environment
            # We need to access the raw environment to get player objects
            # The env might be wrapped multiple times, so we unwrap to find WarehouseBrawl
            raw_env = self.env.unwrapped
            
            # Handle different environment structures
            players = None
            if hasattr(raw_env, 'players'):
                players = raw_env.players
            elif hasattr(raw_env, 'raw_env') and hasattr(raw_env.raw_env, 'players'):
                players = raw_env.raw_env.players
            
            if players:
                p0 = players[0] # Agent
                p1 = players[1] # Opponent
                
                # 1. Damage Dealt
                damage_dealt = p0.damage_done
                info['damage_dealt'] = damage_dealt

                # 1.5 Damage Taken
                damage_taken = p0.damage_taken_total
                info['damage_taken'] = damage_taken
                
                # 2. Win/Loss/Draw Check
                is_win = 0
                is_draw = 0
                
                if p0.stocks > p1.stocks:
                    is_win = 1
                elif p1.stocks > p0.stocks:
                    is_win = 0
                else:
                    # Tie-breaker: Damage Dealt
                    if p0.damage_done > p1.damage_done:
                        is_win = 1
                    elif p1.damage_done > p0.damage_done:
                        is_win = 0
                    else:
                        # True Draw (Equal stocks, equal damage)
                        is_draw = 1
                        is_win = 0 # Count draw as not-win
                
                info['is_win'] = is_win
                info['is_draw'] = is_draw
                
        return obs, reward, terminated, truncated, info
                
        return obs, reward, terminated, truncated, info

# Vectorized SaveHandler for subprocesses
# It disables saving (handled by main process) but allows loading opponents
class VectorizedSaveHandler(SaveHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = None # Agent is not available in subprocess
        self.last_update_step = 0
        self.update_freq = 10000 # Check for new models every 10k steps
        
    def update_info(self):
        pass
        
    def process(self) -> bool:
        # In subprocess, we don't save, but we SHOULD update our knowledge of what exists on disk
        # so that SelfPlayLatest/Random can find new opponents.
        self.num_timesteps += 1
        
        if self.num_timesteps - self.last_update_step >= self.update_freq:
            self.last_update_step = self.num_timesteps
            self._refresh_history()
            
        return False
        
    def save_agent(self) -> None:
        # Do nothing in subprocess
        pass

    def _refresh_history(self):
        """Re-scans the save directory for new .zip files."""
        if not os.path.exists(self.save_path):
            return

        try:
            # Get all model paths
            files = [os.path.join(self.save_path, f) for f in os.listdir(self.save_path) if os.path.isfile(os.path.join(self.save_path, f))]
            # Filter for .zip
            new_history = [f for f in files if f.endswith('.zip')]
            
            if len(new_history) > 0:
                # Sort by step count (assuming format name_steps.zip)
                # We use the same logic as SaveHandler to ensure consistency
                try:
                    new_history.sort(key=lambda x: int(os.path.basename(x).split('_')[-2]))
                    self.history = new_history
                    # print(f"Subprocess updated history: {len(self.history)} models found.")
                except Exception:
                    # Fallback if naming convention fails
                    self.history = new_history
        except Exception as e:
            pass # Suppress errors in subprocess to avoid spamming logs

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

class CustomLoggingCallback(BaseCallback):
    """
    Callback for logging custom metrics (Damage, Win Rate) to TensorBoard and Console.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.damage_history = []
        self.damage_taken_history = []
        self.win_history = []
        self.draw_history = []
        self.games_won_total = 0
        
    def _on_step(self) -> bool:
        # Access the 'infos' list from the locals (provided by SB3)
        # This contains the info dicts from all parallel environments
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if "damage_dealt" in info:
                self.damage_history.append(info["damage_dealt"])
            
            if "damage_taken" in info:
                self.damage_taken_history.append(info["damage_taken"])

            if "is_win" in info:
                is_win = info["is_win"]
                self.win_history.append(is_win)
                if is_win == 1:
                    self.games_won_total += 1
            
            if "is_draw" in info:
                self.draw_history.append(info["is_draw"])
                    
        # Log every n_steps (sync with PPO updates)
        # Use a fallback of 512 if model.n_steps is not available for some reason
        log_freq = self.model.n_steps if hasattr(self.model, "n_steps") else 512
        
        if self.n_calls % log_freq == 0:
            if len(self.damage_history) > 0:
                avg_damage = np.mean(self.damage_history)
                self.logger.record("custom/damage_dealt_mean", avg_damage)
                # Keep history short to reflect recent performance
                self.damage_history = self.damage_history[-log_freq:] 

            if len(self.damage_taken_history) > 0:
                avg_damage_taken = np.mean(self.damage_taken_history)
                self.logger.record("custom/damage_taken_mean", avg_damage_taken)
                self.damage_taken_history = self.damage_taken_history[-log_freq:]
                
            if len(self.win_history) > 0:
                win_rate = np.mean(self.win_history)
                self.logger.record("custom/win_rate", win_rate)
                self.win_history = self.win_history[-log_freq:]
            
            if len(self.draw_history) > 0:
                draw_rate = np.mean(self.draw_history)
                self.logger.record("custom/draw_rate", draw_rate)
                self.draw_history = self.draw_history[-log_freq:]
                
            self.logger.record("custom/games_won_total", self.games_won_total)
            
        return True

# Factory function for environment creation (must be picklable)
class DamageReward:
    def __init__(self):
        self.last_damage_done = {0: 0.0, 1: 0.0}
        self.last_damage_taken = {0: 0.0, 1: 0.0}

    def __call__(self, env):
        # Reset state if it's the first step of an episode
        if hasattr(env, 'steps') and env.steps == 0:
            self.last_damage_done = {0: 0.0, 1: 0.0}
            self.last_damage_taken = {0: 0.0, 1: 0.0}

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
            reward = (delta_dd * 1.0) - (delta_dt * 0.5)
            rewards.append(reward)
        return torch.tensor(rewards)

class DistanceReward:
    def __init__(self):
        self.last_distance = {0: None, 1: None}

    def __call__(self, env):
        # Reset state if it's the first step of an episode
        if hasattr(env, 'steps') and env.steps == 0:
            self.last_distance = {0: None, 1: None}

        rewards = []
        
        # Calculate current distance between players
        p1_pos = env.players[0].body.position
        p2_pos = env.players[1].body.position
        # Pymunk Vec2d distance
        current_dist = p1_pos.get_distance(p2_pos)
        
        for i in range(2):
            if self.last_distance[i] is None:
                self.last_distance[i] = current_dist
                rewards.append(0.0)
                continue
            
            # Reward = (Old Distance - New Distance)
            # Positive if got closer, Negative if moved away
            diff = self.last_distance[i] - current_dist
            self.last_distance[i] = current_dist
            
            # Scale factor: 1.0 (1 unit closer = +1 reward)
            rewards.append(diff * 1.0)
            
        return torch.tensor(rewards)

class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def update_reward_weights(self, new_weights):
        """
        Updates the weights of the reward functions in the RewardManager.
        :param new_weights: Dictionary mapping reward names to new weights.
        """
        # Access the inner environment's reward manager
        if hasattr(self.env, 'reward_manager'):
            for name, weight in new_weights.items():
                if name in self.env.reward_manager.reward_functions:
                    self.env.reward_manager.reward_functions[name].weight = weight
                elif name in self.env.reward_manager.signal_subscriptions:
                    self.env.reward_manager.signal_subscriptions[name].weight = weight

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        return True

def make_env(opponent_cfg, save_freq, max_saved, parent_dir, model_name, resolution, stage=1):
    # Define Reward Functions

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
    
    # Stage 1: Distance Reward
    # Stage 2: Full Game Rewards (Damage, Win, KO)
    
    reward_functions = {}
    signal_subscriptions = {}
    
    if stage == 1:
        distance_reward = DistanceReward()
        reward_functions = {
            "distance": RewTerm(func=distance_reward, weight=1.0)
        }
        signal_subscriptions = {
            "win": RewTerm(func=win_reward_func, weight=1.0, params={}),
            "knockout": RewTerm(func=knockout_reward_func, weight=1.0, params={})
        }
    elif stage == 2:
        # Full game rewards
        reward_functions = {
            "damage": RewTerm(func=DamageReward(), weight=1.0),
            "distance": RewTerm(func=DistanceReward(), weight=0.5) # Increased from 0.1 to 0.5
        }
        signal_subscriptions = {
            "win": RewTerm(func=win_reward_func, weight=2.0, params={}), 
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
        reward_manager=reward_manager,
        render_mode="rgb_array"
    )
    env = CurriculumWrapper(env) # Wrap with CurriculumWrapper
    env = StatsWrapper(env) # Wrap with StatsWrapper for custom metrics
    env = Float32Wrapper(env)
    return env

def main(cfg_file, stage=1, n_envs_override=None):
    # Read the cfg file
    with open(cfg_file, 'r') as f:
        params = yaml.safe_load(f)
    
    print(f"Loaded config from {cfg_file}")
    print(f"Stage: {stage}")

    # Hardware detection
    device = get_device()
    print(f"Training on device: {device}")

    # Paths
    base_path = os.getcwd()
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["folders"]["model_name"], "model")
    tensor_board_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["folders"]["model_name"], "tb")
    video_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["folders"]["model_name"], "videos")
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

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

    # FORCE n_envs to 1 for debugging
    n_envs = 1
    print(f"DEBUG MODE: Forcing n_envs to {n_envs}")
    
    # Prepare arguments for make_env factory
    env_kwargs = {
        "opponent_cfg": opponent_cfg,
        "save_freq": params["self_play_settings"]["save_freq"],
        "max_saved": params["self_play_settings"]["max_saved_models"],
        "parent_dir": params["folders"]["parent_dir"],
        "model_name": params["folders"]["model_name"],
        "resolution": resolution,
        "stage": stage
    }

    start_method = "spawn" 
    print(f"Using multiprocessing start method: {start_method}")

    # USE DummyVecEnv for debugging
    print("DEBUG MODE: Using DummyVecEnv")
    env = make_vec_env(
        make_env, 
        n_envs=n_envs, 
        vec_env_cls=DummyVecEnv, # CHANGED FROM SubprocVecEnv
        # vec_env_kwargs={"start_method": start_method}, # Not needed for DummyVecEnv
        env_kwargs=env_kwargs
    )
    env = VecMonitor(env)
    
    print("Vectorized environment created.")

    # --- Video Recording Setup (EvalCallback) ---
    # Create a separate evaluation environment for video recording
    # This avoids rendering overhead in the main training loop
    print("Creating evaluation environment for video recording...")
    eval_env = make_vec_env(
        make_env,
        n_envs=1, # Single environment for evaluation
        vec_env_cls=DummyVecEnv, # DummyVecEnv is sufficient for single process
        env_kwargs=env_kwargs
    )
    
    # Wrap eval_env with VecVideoRecorder
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder,
        record_video_trigger=lambda x: x == 0, # Record every episode triggered by callback
        video_length=2000,
        name_prefix=f"stage_{stage}_video"
    )
    
    # Create EvalCallback
    # Evaluate every 500,000 steps (adjusted for n_envs)
    eval_freq = 500000 // n_envs
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_folder, "best_model"),
        log_path=tensor_board_folder,
        eval_freq=eval_freq,
        n_eval_episodes=1, # Record 1 episode per evaluation
        deterministic=True,
        render=False
    )

    # PPO Settings
    ppo_settings = params["ppo_settings"]
    learning_rate = linear_schedule(ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    clip_range = linear_schedule(ppo_settings["clip_range"][0], ppo_settings["clip_range"][1])
    
    buffer_size = n_envs * ppo_settings["n_steps"]
    batch_size = ppo_settings["batch_size"]
    
    if buffer_size < batch_size:
        print(f"⚠️ Warning: Buffer size ({buffer_size}) is smaller than configured batch_size ({batch_size}).")
        batch_size = buffer_size
        print(f"📉 Auto-adjusting batch_size to {batch_size} to match buffer size.")
    
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
            batch_size=batch_size,
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
            learning_rate=learning_rate,
            clip_range=clip_range,
            tensorboard_log=tensor_board_folder # Ensure tensorboard logging continues
        )

    # Callbacks
    checkpoint_callback = LimitedCheckpointCallback(
        save_freq=params["self_play_settings"]["save_freq"],
        save_path=os.path.join(model_folder, "checkpoints"),
        name_prefix="rl_model",
        max_keep=5
    )
    
    self_play_callback = SelfPlayCallback(save_handler)
    curriculum_callback = CurriculumCallback()
    logging_callback = CustomLoggingCallback()

    # --- Manual Step 0 Video ---
    print("🎥 Recording initial video at step 0...")
    obs = eval_env.reset()
    # We need to step the environment to trigger the video recorder
    # The EvalCallback will handle periodic recording, but we want one NOW.
    # Since eval_env is wrapped with VecVideoRecorder, stepping it will record.
    done = False
    step_count = 0
    while step_count < 1000: # Short initial video
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        step_count += 1
        if done.any():
            break
    eval_env.close()
    
    # Re-create eval_env for the callback because we closed it
    eval_env = make_vec_env(
        make_env,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs=env_kwargs
    )
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=2000,
        name_prefix=f"stage_{stage}_video"
    )
    # Update callback with new env
    eval_callback.eval_env = eval_env

    print(f"Starting training for {ppo_settings['time_steps']} timesteps...")
    
    try:
        agent.learn(
            total_timesteps=ppo_settings["time_steps"],
            callback=[checkpoint_callback, self_play_callback, curriculum_callback, eval_callback, logging_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    # Save final model
    final_path = os.path.join(model_folder, "ppo_utmist_final")
    agent.save(final_path)
    print(f"Training finished. Model saved to {final_path}")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, default="utmist_config.yaml", help="Configuration file")
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage")
    parser.add_argument("--n_envs", type=int, default=None, help="Number of environments")
    opt = parser.parse_args()
    main(opt.cfgFile, stage=opt.stage, n_envs_override=opt.n_envs)
