"""
UTMIST AI^2 Training Script v2 - Config-Driven Edition
=======================================================
All configuration loaded from YAML file.
Phase-based reward curriculum for proper skill acquisition.

Usage:
    python train_utmist_v2.py --cfgFile utmist_config_v2.yaml
"""

import os
import yaml
import argparse
import torch
import sys
import signal
import gymnasium as gym
import numpy as np
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add UTMIST environment to path
sys.path.append(os.path.join(os.getcwd(), "UTMIST-AI2-main"))

from environment.agent import (
    SelfPlayWarehouseBrawl, OpponentsCfg, SaveHandler, SaveHandlerMode,
    CameraResolution, RewardManager, RandomAgent, BasedAgent, ConstantAgent,
    ClockworkAgent, SelfPlayLatest, SelfPlayRandom, RewTerm
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


def get_opponent_class(name: str):
    """Get opponent class by name."""
    opponents = {
        "random_agent": RandomAgent,
        "based_agent": BasedAgent,
        "constant_agent": ConstantAgent,
        "clockwork_agent": ClockworkAgent,
    }
    return opponents.get(name)


def get_resolution(resolution_str: str):
    """Get CameraResolution enum from string."""
    resolution_map = {
        "LOW": CameraResolution.LOW,
        "MEDIUM": CameraResolution.MEDIUM,
        "HIGH": CameraResolution.HIGH
    }
    return resolution_map.get(resolution_str.upper(), CameraResolution.LOW)


# ============================================================================
# PHASE CONFIG LOADER
# ============================================================================

class PhaseConfig:
    """Loads and validates phase configuration from YAML."""
    
    def __init__(self, params: dict, phase_key):
        self.params = params
        self.phase_key = phase_key
        
        # Get phase definition from YAML
        phases = params.get("phases", {})
        
        # Handle both string and int keys
        if phase_key in phases:
            self.config = phases[phase_key]
        elif str(phase_key) in phases:
            self.config = phases[str(phase_key)]
        else:
            raise ValueError(f"Phase '{phase_key}' not found in config. Available: {list(phases.keys())}")
        
        self.name = self.config.get("name", f"Phase {phase_key}")
        self.description = self.config.get("description", "")
        self.type = self.config.get("type", "combat")
        self.is_navigation = self.type == "navigation"
        
        # Rewards (for combat phases)
        self.rewards = self.config.get("rewards", {
            "distance": 0.0, "aggression": 0.0, "damage_dealt": 0.0,
            "damage_taken": 0.0, "net_damage": 0.0, "win": 0.0,
            "knockout": 0.0, "edge_penalty": 0.0
        })
        
        # Navigation settings (for nav phases)
        self.navigation = self.config.get("navigation", {})
        
        # Opponents
        self.opponents = self.config.get("opponents", {"random_agent": 1.0})
    
    def print_summary(self):
        """Print phase configuration summary."""
        print(f"\n{'='*60}")
        print(f"🎯 PHASE {self.phase_key}: {self.name}")
        print(f"   {self.description}")
        print(f"{'='*60}")
        
        if self.is_navigation:
            nav = self.navigation
            print(f"📍 Navigation Settings:")
            print(f"   Distance: {nav.get('min_dist', 1.0)} - {nav.get('max_dist', 12.0)}")
            print(f"   Fall Penalty: {nav.get('fall_penalty', -5.0)}")
            print(f"   Targets: {nav.get('multi_target', 1)}")
            print(f"   Time Limit: {nav.get('time_limit_seconds', 90)}s")
        else:
            print("⚖️ Reward Weights:")
            for name, weight in self.rewards.items():
                if weight != 0:
                    print(f"   {name}: {weight:+.2f}")
        
        print("👥 Opponents:")
        for name, prob in self.opponents.items():
            if prob > 0:
                print(f"   {name}: {prob*100:.0f}%")
        
        print(f"{'='*60}\n")


# ============================================================================
# ENVIRONMENT WRAPPERS
# ============================================================================

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


class FrozenOpponentWrapper(gym.Wrapper):
    """
    Wrapper for Phase 0 Navigation Training.
    Freezes opponent at random positions and rewards navigation.
    """
    
    # Stage bounds (approximate)
    STAGE_X_MIN = -7.0
    STAGE_X_MAX = 7.0
    STAGE_Y_MIN = -1.0  # Ground level
    STAGE_Y_MAX = 5.0   # Jump height
    
    def __init__(self, env, nav_config: dict, debug_logs=True):
        super().__init__(env)
        self.nav_config = nav_config
        self.debug_logs = debug_logs
        
        # Load from config
        self.min_dist = nav_config.get("min_dist", 1.0)
        self.max_dist = nav_config.get("max_dist", 12.0)
        self.fall_penalty = nav_config.get("fall_penalty", -5.0)
        self.time_limit = nav_config.get("time_limit_seconds", 90) * 30  # Convert to frames
        self.max_targets = nav_config.get("multi_target", 1)
        
        # Tracking
        self.target_position = None
        self.last_distance = None
        self.targets_reached = 0
        self.total_targets = 0
        self.falls = 0
        self.episode_steps = 0
        self.targets_this_episode = 0
        
        # Logging
        self.log_interval = 10000
        self.steps_since_log = 0
        self.interval_targets_reached = 0
        self.interval_falls = 0
        self.interval_episodes = 0
        
    def _get_player(self):
        """Get player object from environment."""
        env = self.env
        if hasattr(env, 'raw_env') and hasattr(env.raw_env, 'players'):
            return env.raw_env.players[0]
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'raw_env'):
            return env.unwrapped.raw_env.players[0]
        return None
    
    def _get_opponent(self):
        """Get opponent object from environment."""
        env = self.env
        if hasattr(env, 'raw_env') and hasattr(env.raw_env, 'players'):
            return env.raw_env.players[1]
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'raw_env'):
            return env.unwrapped.raw_env.players[1]
        return None
    
    def _spawn_target(self, player_pos):
        """Spawn a new target position based on config."""
        for _ in range(100):
            x = np.random.uniform(self.STAGE_X_MIN, self.STAGE_X_MAX)
            y = np.random.uniform(self.STAGE_Y_MIN, self.STAGE_Y_MAX)
            dist = np.sqrt((x - player_pos[0])**2 + (y - player_pos[1])**2)
            if self.min_dist <= dist <= self.max_dist:
                self.target_position = (x, y)
                self.total_targets += 1
                return
        self.target_position = (0.0, 0.0)
        self.total_targets += 1
    
    def _freeze_opponent(self):
        """Freeze opponent at target position."""
        opponent = self._get_opponent()
        if opponent is not None and self.target_position is not None:
            try:
                opponent.body.position = self.target_position
                opponent.body.velocity = (0, 0)
            except:
                pass
    
    def _get_distance_to_target(self, player_pos):
        """Calculate distance from player to target."""
        if self.target_position is None:
            return 0.0
        return np.sqrt(
            (player_pos[0] - self.target_position[0])**2 + 
            (player_pos[1] - self.target_position[1])**2
        )
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.steps_since_log += 1
        
        player = self._get_player()
        if player is None:
            return obs, reward, terminated, truncated, info
        
        player_pos = (player.body.position.x, player.body.position.y)
        self._freeze_opponent()
        
        # Calculate navigation reward
        nav_reward = 0.0
        current_dist = self._get_distance_to_target(player_pos)
        
        # Approach reward
        if self.last_distance is not None:
            dist_reduction = self.last_distance - current_dist
            nav_reward += dist_reduction * 2.0
        
        # Target reached bonus
        if current_dist < 1.0:
            nav_reward += 5.0
            self.targets_reached += 1
            self.interval_targets_reached += 1
            self.targets_this_episode += 1
            
            if self.targets_this_episode < self.max_targets:
                self._spawn_target(player_pos)
            else:
                truncated = True
        
        self.last_distance = current_dist
        
        # Edge penalty
        x_pos = abs(player_pos[0])
        if x_pos > 5.0:
            edge_factor = (x_pos - 5.0) / 2.0
            nav_reward -= 0.5 * min(edge_factor, 1.0)
        
        # Fall detection
        if terminated:
            my_lives = getattr(player, 'lives_left', 3)
            if my_lives < 3:
                nav_reward += self.fall_penalty
                self.falls += 1
                self.interval_falls += 1
        
        # Time limit
        if self.episode_steps >= self.time_limit:
            truncated = True
        
        # Logging
        if self.steps_since_log >= self.log_interval and self.debug_logs:
            self._log_stats()
        
        return obs, reward + nav_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        player = self._get_player()
        if player is not None:
            player_pos = (player.body.position.x, player.body.position.y)
            self._spawn_target(player_pos)
            self._freeze_opponent()
            self.last_distance = self._get_distance_to_target(player_pos)
        
        self.episode_steps = 0
        self.targets_this_episode = 0
        self.interval_episodes += 1
        
        return obs, info
    
    def _log_stats(self):
        """Log navigation statistics."""
        if self.interval_episodes > 0:
            reach_rate = (self.interval_targets_reached / max(1, self.interval_episodes)) * 100
            fall_rate = (self.interval_falls / max(1, self.interval_episodes)) * 100
        else:
            reach_rate = fall_rate = 0
        
        print(f"\n[NAV] {self.steps_since_log:,} steps | "
              f"Reach: {reach_rate:.0f}% | Falls: {fall_rate:.0f}% | "
              f"Episodes: {self.interval_episodes}")
        
        self.steps_since_log = 0
        self.interval_targets_reached = 0
        self.interval_falls = 0
        self.interval_episodes = 0


class DamageTrackingWrapper(gym.Wrapper):
    """Tracks damage dealt/received and death types (SD vs KO)."""
    
    def __init__(self, env):
        super().__init__(env)
        self.reset_tracking()
    
    def reset_tracking(self):
        self.total_damage_dealt = 0.0
        self.total_damage_received = 0.0
        self.episode_damage_dealt = 0.0
        self.episode_damage_received = 0.0
        self.last_damage_done = 0.0
        self.last_damage_taken = 0.0
        self.wins = 0
        self.losses = 0
        self.last_lives = 3
        self.recent_damage_taken = 0.0
        self.damage_history = []
        self.episode_sds = 0
        self.episode_kos = 0
        self.total_sds = 0
        self.total_kos = 0
    
    def _get_player(self):
        env = self.env
        if hasattr(env, 'raw_env') and hasattr(env.raw_env, 'players'):
            return env.raw_env.players[0]
        if hasattr(env, 'unwrapped'):
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'raw_env') and hasattr(unwrapped.raw_env, 'players'):
                return unwrapped.raw_env.players[0]
        return None
    
    def _get_opponent(self):
        env = self.env
        if hasattr(env, 'raw_env') and hasattr(env.raw_env, 'players'):
            return env.raw_env.players[1]
        if hasattr(env, 'unwrapped'):
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'raw_env') and hasattr(unwrapped.raw_env, 'players'):
                return unwrapped.raw_env.players[1]
        return None
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        try:
            player = self._get_player()
            if player is not None:
                current_dealt = getattr(player, 'damage_done', 0.0)
                current_taken = getattr(player, 'damage_taken_total', 0.0)
                
                delta_dealt = current_dealt - self.last_damage_done
                delta_taken = current_taken - self.last_damage_taken
                
                if delta_dealt > 0:
                    self.episode_damage_dealt += delta_dealt
                    self.total_damage_dealt += delta_dealt
                if delta_taken > 0:
                    self.episode_damage_received += delta_taken
                    self.total_damage_received += delta_taken
                
                self.damage_history.append(delta_taken)
                if len(self.damage_history) > 30:
                    self.damage_history.pop(0)
                self.recent_damage_taken = sum(self.damage_history)
                
                self.last_damage_done = current_dealt
                self.last_damage_taken = current_taken
                
                current_lives = getattr(player, 'lives_left', 3)
                if current_lives < self.last_lives:
                    if self.recent_damage_taken < 20:
                        self.episode_sds += 1
                        self.total_sds += 1
                    else:
                        self.episode_kos += 1
                        self.total_kos += 1
                self.last_lives = current_lives
                
                info['damage_dealt'] = self.episode_damage_dealt
                info['damage_received'] = self.episode_damage_received
        except Exception:
            pass
        
        if terminated or truncated:
            try:
                player = self._get_player()
                opponent = self._get_opponent()
                
                if player and opponent:
                    my_lives = getattr(player, 'lives_left', 0)
                    opp_lives = getattr(opponent, 'lives_left', 0)
                    my_damage = getattr(player, 'damage_done', 0)
                    opp_damage = getattr(opponent, 'damage_done', 0)
                    
                    if my_lives > opp_lives:
                        self.wins += 1
                        info['win'] = 1
                    elif my_lives < opp_lives:
                        self.losses += 1
                        info['win'] = 0
                    else:
                        if my_damage > opp_damage:
                            self.wins += 1
                            info['win'] = 1
                        else:
                            self.losses += 1
                            info['win'] = 0
                else:
                    if self.episode_damage_dealt > self.episode_damage_received:
                        self.wins += 1
                        info['win'] = 1
                    else:
                        self.losses += 1
                        info['win'] = 0
            except:
                info['win'] = 0
            
            info['episode_damage_dealt'] = self.episode_damage_dealt
            info['episode_damage_received'] = self.episode_damage_received
            info['episode_sds'] = self.episode_sds
            info['episode_kos'] = self.episode_kos
            
            # Reset episode stats
            self.episode_damage_dealt = 0.0
            self.episode_damage_received = 0.0
            self.episode_sds = 0
            self.episode_kos = 0
            self.last_damage_done = 0.0
            self.last_damage_taken = 0.0
            self.last_lives = 3
            self.damage_history = []
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.episode_damage_dealt = 0.0
        self.episode_damage_received = 0.0
        self.episode_sds = 0
        self.episode_kos = 0
        self.last_damage_done = 0.0
        self.last_damage_taken = 0.0
        self.last_lives = 3
        self.damage_history = []
        return self.env.reset(**kwargs)


# ============================================================================
# CALLBACKS
# ============================================================================

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
        self.last_save_timestep = 0

    def _on_step(self) -> bool:
        result = super()._on_step()
        
        if self.num_timesteps >= self.last_save_timestep + self.save_freq:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.saved_files.append(path)
            self.last_save_timestep = self.num_timesteps
            
            if len(self.saved_files) > self.max_keep:
                to_remove = self.saved_files.pop(0)
                if os.path.exists(to_remove):
                    os.remove(to_remove)
        return result


class DetailedLoggingCallback(BaseCallback):
    """Logs damage stats, win rate, SD/KO breakdown."""
    
    def __init__(self, log_freq=10000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.reset_interval()
        self.total_wins = 0
        self.total_losses = 0
        self.total_sds = 0
        self.total_kos = 0
    
    def reset_interval(self):
        self.interval_damage_dealt = 0.0
        self.interval_damage_received = 0.0
        self.interval_wins = 0
        self.interval_losses = 0
        self.interval_episodes = 0
        self.interval_sds = 0
        self.interval_kos = 0
        
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode_damage_dealt' in info:
                self.interval_damage_dealt += info.get('episode_damage_dealt', 0)
                self.interval_damage_received += info.get('episode_damage_received', 0)
                self.interval_sds += info.get('episode_sds', 0)
                self.interval_kos += info.get('episode_kos', 0)
                self.interval_episodes += 1
            
            if 'win' in info:
                if info['win'] == 1:
                    self.interval_wins += 1
                    self.total_wins += 1
                else:
                    self.interval_losses += 1
                    self.total_losses += 1
        
        if self.num_timesteps % self.log_freq == 0 and self.num_timesteps > 0:
            self._log_stats()
        
        return True
    
    def _log_stats(self):
        print("\n" + "=" * 60)
        print(f"📊 STATS @ {self.num_timesteps:,} steps")
        print("=" * 60)
        
        if self.interval_episodes > 0:
            avg_dealt = self.interval_damage_dealt / self.interval_episodes
            avg_received = self.interval_damage_received / self.interval_episodes
            avg_net = avg_dealt - avg_received
            
            print(f"⚔️  Avg Damage Dealt:    {avg_dealt:.1f}")
            print(f"🛡️  Avg Damage Received: {avg_received:.1f}")
            print(f"📈 Avg Net Damage:      {avg_net:+.1f}")
            
            self.logger.record("damage/avg_dealt", avg_dealt)
            self.logger.record("damage/avg_received", avg_received)
            self.logger.record("damage/avg_net", avg_net)
        
        interval_games = self.interval_wins + self.interval_losses
        if interval_games > 0:
            win_rate = (self.interval_wins / interval_games) * 100
            print(f"🏆 Win Rate (recent):   {win_rate:.1f}% ({self.interval_wins}W / {self.interval_losses}L)")
            self.logger.record("winrate/recent", win_rate)
        
        total_games = self.total_wins + self.total_losses
        if total_games > 0:
            overall_win_rate = (self.total_wins / total_games) * 100
            print(f"📊 Win Rate (overall):  {overall_win_rate:.1f}%")
            self.logger.record("winrate/overall", overall_win_rate)
        
        total_deaths = self.interval_sds + self.interval_kos
        if total_deaths > 0:
            sd_pct = (self.interval_sds / total_deaths) * 100
            print(f"💀 Deaths: {self.interval_sds} SD / {self.interval_kos} KO ({sd_pct:.0f}% self-destruct)")
            self.logger.record("deaths/sd_pct", sd_pct)
        
        print("=" * 60 + "\n")
        self.reset_interval()


class EvaluationCallback(BaseCallback):
    """Periodically evaluate agent against diverse opponents."""
    
    def __init__(self, eval_freq=500000, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_step = 0
        
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self._run_evaluation()
        return True
    
    def _run_evaluation(self):
        from environment.environment import WarehouseBrawl
        
        print("\n" + "=" * 50)
        print(f"🧪 EVALUATION @ {self.num_timesteps:,} steps")
        print("=" * 50)
        
        test_opponents = {
            "ConstantAgent": ConstantAgent(),
            "RandomAgent": RandomAgent(),
            "ClockworkAgent": ClockworkAgent(),
            "BasedAgent": BasedAgent(),
        }
        
        for opponent_name, opponent in test_opponents.items():
            wins, losses, draws = 0, 0, 0
            total_damage_dealt = 0
            
            for _ in range(self.n_eval_episodes):
                try:
                    env = WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=True)
                    env.max_timesteps = 30 * 90
                    
                    observations, _ = env.reset()
                    obs = observations[0]
                    opponent.get_env_info(env)
                    opponent_obs = observations[1]
                    
                    frame_stack = [obs.copy() for _ in range(4)]
                    
                    done = False
                    while not done:
                        stacked_obs = np.concatenate(frame_stack, axis=0)
                        action, _ = self.model.predict(stacked_obs, deterministic=True)
                        opp_action = opponent.predict(opponent_obs)
                        
                        observations, _, terminated, truncated, _ = env.step({0: action, 1: opp_action})
                        
                        obs = observations[0]
                        opponent_obs = observations[1]
                        frame_stack.pop(0)
                        frame_stack.append(obs.copy())
                        
                        done = terminated or truncated
                    
                    stats = env.get_stats(0)
                    opp_stats = env.get_stats(1)
                    
                    if stats.lives_left > opp_stats.lives_left:
                        wins += 1
                    elif stats.lives_left < opp_stats.lives_left:
                        losses += 1
                    else:
                        if stats.damage_done > opp_stats.damage_done:
                            wins += 1
                        else:
                            draws += 1
                    
                    total_damage_dealt += stats.damage_done
                    env.close()
                    
                except Exception as e:
                    if self.verbose:
                        print(f"  Eval error: {e}")
            
            total_games = wins + losses + draws
            win_rate = (wins / total_games) * 100 if total_games > 0 else 0
            avg_dealt = total_damage_dealt / max(1, total_games)
            
            print(f"  vs {opponent_name}: {win_rate:.0f}% ({wins}W/{losses}L/{draws}D) | Avg Dmg: {avg_dealt:.0f}")
            self.logger.record(f"eval/{opponent_name}/win_rate", win_rate)
        
        print("=" * 50 + "\n")


# ============================================================================
# REWARD SYSTEM
# ============================================================================

class PhaseAwareRewardManager:
    """Manages rewards based on phase config from YAML."""
    
    def __init__(self, phase_config: PhaseConfig):
        self.phase_config = phase_config
        self.weights = phase_config.rewards
        
        # Tracking
        self.episode_damage_dealt = 0.0
        self.episode_damage_received = 0.0
        self.total_damage_dealt = 0.0
        self.total_damage_received = 0.0
        
        self.last_damage_done = {0: 0.0, 1: 0.0}
        self.last_damage_taken = {0: 0.0, 1: 0.0}
        self.last_distance = {0: None, 1: None}
    
    def compute_reward(self, env) -> float:
        total_reward = 0.0
        player = env.players[0]
        opponent = env.players[1]
        
        # Distance Reward
        if self.weights.get("distance", 0) != 0:
            p1_pos = player.body.position
            p2_pos = opponent.body.position
            current_dist = p1_pos.get_distance(p2_pos)
            
            if self.last_distance[0] is None:
                self.last_distance[0] = current_dist
            else:
                dist_delta = self.last_distance[0] - current_dist
                total_reward += dist_delta * self.weights["distance"]
                self.last_distance[0] = current_dist
        
        # Damage Dealt
        current_damage_done = player.damage_done
        delta_dd = current_damage_done - self.last_damage_done[0]
        self.last_damage_done[0] = current_damage_done
        
        if delta_dd > 0:
            self.episode_damage_dealt += delta_dd
            self.total_damage_dealt += delta_dd
        
        if self.weights.get("damage_dealt", 0) != 0:
            total_reward += delta_dd * self.weights["damage_dealt"]
        
        # Damage Taken
        current_damage_taken = player.damage_taken_total
        delta_dt = current_damage_taken - self.last_damage_taken[0]
        self.last_damage_taken[0] = current_damage_taken
        
        if delta_dt > 0:
            self.episode_damage_received += delta_dt
            self.total_damage_received += delta_dt
        
        if self.weights.get("damage_taken", 0) != 0:
            total_reward += delta_dt * self.weights["damage_taken"]
        
        # Net Damage
        if self.weights.get("net_damage", 0) != 0:
            net_damage = delta_dd - delta_dt
            if net_damage > 0:
                total_reward += net_damage * self.weights["net_damage"]
        
        # Edge Penalty
        edge_weight = self.weights.get("edge_penalty", 0.0)
        if edge_weight != 0:
            x_pos = abs(player.body.position.x)
            if x_pos > 8.0:
                edge_factor = (x_pos - 8.0) / 4.0
                total_reward += edge_weight * min(edge_factor, 1.0)
        
        # Aggression
        if self.weights.get("aggression", 0) != 0:
            attack_states = {6, 7, 8, 9, 10}
            if hasattr(player, 'state_machine') and hasattr(player.state_machine, 'current_state'):
                state_id = getattr(player.state_machine.current_state, 'id', -1)
                if state_id in attack_states:
                    total_reward += self.weights["aggression"] * 0.1
        
        return total_reward
    
    def compute_win_reward(self, winner_is_agent: bool) -> float:
        win_weight = self.weights.get("win", 0.0)
        if winner_is_agent:
            base_reward = win_weight
            clean_bonus = self.weights.get("clean_win_bonus", 0.0)
            if clean_bonus > 0 and self.episode_damage_dealt > self.episode_damage_received:
                base_reward += clean_bonus
            return base_reward
        else:
            return -win_weight
    
    def compute_knockout_reward(self, agent_got_ko: bool) -> float:
        ko_weight = self.weights.get("knockout", 0.0)
        return -ko_weight if agent_got_ko else ko_weight
    
    def reset_episode(self):
        self.episode_damage_dealt = 0.0
        self.episode_damage_received = 0.0
        self.last_distance = {0: None, 1: None}


# Global reward manager
GLOBAL_REWARD_MANAGER = None


class DamageReward:
    """Phase-aware damage reward."""
    def __init__(self, reward_manager: PhaseAwareRewardManager):
        self.reward_manager = reward_manager

    def __call__(self, env):
        reward = self.reward_manager.compute_reward(env)
        return torch.tensor([reward, -reward])


# ============================================================================
# ENVIRONMENT FACTORIES
# ============================================================================

def make_nav_env(phase_config: PhaseConfig, resolution):
    """Factory for navigation Phase 0 environments."""
    
    def null_reward_func(env, **kwargs):
        return torch.tensor([0.0, 0.0])
    
    reward_manager = RewardManager(
        reward_functions={"null": RewTerm(func=null_reward_func, weight=0.0)},
        signal_subscriptions={}
    )
    
    opponent_cfg = OpponentsCfg(
        opponents={'constant_agent': (1.0, partial(ConstantAgent))}
    )
    
    env = SelfPlayWarehouseBrawl(
        opponent_cfg=opponent_cfg,
        save_handler=None,
        resolution=resolution,
        reward_manager=reward_manager
    )
    env = Float32Wrapper(env)
    env = FrozenOpponentWrapper(env, nav_config=phase_config.navigation, debug_logs=True)
    env = Monitor(env)
    
    return env


def make_combat_env(phase_config: PhaseConfig, params: dict, device: str):
    """Factory for combat Phases 1-4 environments."""
    global GLOBAL_REWARD_MANAGER
    
    if GLOBAL_REWARD_MANAGER is None:
        GLOBAL_REWARD_MANAGER = PhaseAwareRewardManager(phase_config)
    
    reward_manager_instance = GLOBAL_REWARD_MANAGER
    
    def win_reward_func(env, agent, **kwargs):
        winner_is_agent = (agent != 'player')
        reward = reward_manager_instance.compute_win_reward(winner_is_agent)
        rewards = torch.zeros(2)
        rewards[0] = reward
        rewards[1] = -reward
        return rewards

    def knockout_reward_func(env, agent, **kwargs):
        agent_got_ko = (agent == 'player')
        reward = reward_manager_instance.compute_knockout_reward(agent_got_ko)
        rewards = torch.zeros(2)
        rewards[0] = reward
        rewards[1] = -reward
        return rewards

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
    
    # Build opponent config from YAML
    opponents_dict = {}
    for opp_name, prob in phase_config.opponents.items():
        if prob <= 0:
            continue
        if opp_name == "self_play":
            opponents_dict[opp_name] = (prob, SelfPlayRandom(partial(PPO.load, device=device)))
        else:
            opp_class = get_opponent_class(opp_name)
            if opp_class:
                opponents_dict[opp_name] = (prob, partial(opp_class))
    
    opponent_cfg = OpponentsCfg(opponents=opponents_dict)
    
    resolution = get_resolution(params.get("environment_settings", {}).get("resolution", "LOW"))
    
    vec_save_handler = VectorizedSaveHandler(
        agent=None,
        save_freq=params["self_play_settings"]["save_freq"],
        max_saved=params["self_play_settings"]["max_saved_models"],
        run_name="self_play_run",
        save_path=os.path.join(params["folders"]["parent_dir"], params["folders"]["model_name"], "model"),
        mode=SaveHandlerMode.FORCE
    )
    
    env = SelfPlayWarehouseBrawl(
        opponent_cfg=opponent_cfg,
        save_handler=vec_save_handler,
        resolution=resolution,
        reward_manager=reward_manager
    )
    env = Float32Wrapper(env)
    env = DamageTrackingWrapper(env)
    
    return env


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def run_navigation_training(params: dict, phase_config: PhaseConfig):
    """Run navigation training for Phase 0."""
    print(f"\n🧭 Starting Navigation Training - {phase_config.name}")
    
    device = get_device()
    resolution = get_resolution(params.get("environment_settings", {}).get("resolution", "LOW"))
    
    # Paths
    model_folder = os.path.join(params["folders"]["parent_dir"], params["folders"]["model_name"], "model")
    tb_folder = os.path.join(params["folders"]["parent_dir"], params["folders"]["model_name"], "tb_nav")
    checkpoint_folder = os.path.join(model_folder, "nav_checkpoints")
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(tb_folder, exist_ok=True)
    os.makedirs(checkpoint_folder, exist_ok=True)
    
    # Settings
    n_envs = params.get("environment_settings", {}).get("n_envs", 8)
    total_timesteps = params.get("ppo_settings", {}).get("time_steps", 3_000_000)
    
    print(f"📊 Environments: {n_envs} parallel")
    print(f"🎯 Total timesteps: {total_timesteps:,}")
    
    # Create environments
    env_fns = [lambda: make_nav_env(phase_config, resolution) for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    
    # PPO settings
    ppo = params.get("ppo_settings", {})
    lr_config = ppo.get("learning_rate", [3e-4, 1e-6])
    learning_rate = linear_schedule(lr_config[0], lr_config[1]) if isinstance(lr_config, list) else lr_config
    
    clip_config = ppo.get("clip_range", [0.2, 0.05])
    clip_range = linear_schedule(clip_config[0], clip_config[1]) if isinstance(clip_config, list) else clip_config
    
    # Check for checkpoint
    model_checkpoint = ppo.get("model_checkpoint", "0")
    
    if model_checkpoint != "0":
        checkpoint_path = os.path.join(model_folder, model_checkpoint)
        if os.path.exists(checkpoint_path) or os.path.exists(checkpoint_path + ".zip"):
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            agent = PPO.load(checkpoint_path, env=vec_env, device=device)
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}, creating new agent")
            model_checkpoint = "0"
    
    if model_checkpoint == "0":
        print("🆕 Creating new PPO agent for navigation")
        net_arch = params.get("policy_kwargs", {}).get("net_arch", [512, 512, 256])
        
        agent = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=ppo.get("n_steps", 1024),
            batch_size=ppo.get("batch_size", 2048),
            n_epochs=ppo.get("n_epochs", 8),
            gamma=ppo.get("gamma", 0.99),
            gae_lambda=ppo.get("gae_lambda", 0.95),
            clip_range=clip_range,
            ent_coef=ppo.get("nav_ent_coef", 0.1),
            vf_coef=ppo.get("vf_coef", 0.5),
            max_grad_norm=ppo.get("max_grad_norm", 0.5),
            verbose=1,
            tensorboard_log=tb_folder,
            device=device,
            policy_kwargs={"net_arch": net_arch}
        )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=500000 // n_envs,
        save_path=checkpoint_folder,
        name_prefix=f"nav_{phase_config.phase_key}"
    )
    
    # Signal handler
    def signal_handler(sig, frame):
        print("\n⚠️ Training interrupted! Saving model...")
        interrupt_path = os.path.join(model_folder, f"nav_{phase_config.phase_key}_interrupted.zip")
        agent.save(interrupt_path)
        print(f"💾 Model saved to: {interrupt_path}")
        vec_env.close()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\n🚀 Training navigation phase {phase_config.phase_key}...")
    
    agent.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        progress_bar=True
    )
    
    final_path = os.path.join(model_folder, f"nav_{phase_config.phase_key}_final.zip")
    agent.save(final_path)
    print(f"✅ Navigation training complete! Model saved to: {final_path}")
    
    vec_env.close()


def run_combat_training(params: dict, phase_config: PhaseConfig):
    """Run combat training for Phases 1-4."""
    global GLOBAL_REWARD_MANAGER
    
    device = get_device()
    
    # Paths
    model_folder = os.path.join(params["folders"]["parent_dir"], params["folders"]["model_name"], "model")
    tb_folder = os.path.join(params["folders"]["parent_dir"], params["folders"]["model_name"], "tb")
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(os.path.join(model_folder, "checkpoints"), exist_ok=True)
    
    # Settings
    n_envs = params.get("environment_settings", {}).get("n_envs", 32)
    ppo = params.get("ppo_settings", {})
    
    print(f"🔄 Parallel environments: {n_envs}")
    
    # Print opponent distribution
    print("👥 Opponent Distribution:")
    for name, prob in phase_config.opponents.items():
        if prob > 0:
            print(f"   {name}: {prob*100:.0f}%")
    
    # Create environments
    env_kwargs = {
        "phase_config": phase_config,
        "params": params,
        "device": device,
    }
    
    env = make_vec_env(
        make_combat_env,
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
    lr_config = ppo.get("learning_rate", [3e-4, 1e-6])
    learning_rate = linear_schedule(lr_config[0], lr_config[1]) if isinstance(lr_config, list) else lr_config
    
    clip_config = ppo.get("clip_range", [0.2, 0.05])
    clip_range = linear_schedule(clip_config[0], clip_config[1]) if isinstance(clip_config, list) else clip_config
    
    ent_coef = ppo.get("ent_coef", 0.01)
    buffer_size = n_envs * ppo.get("n_steps", 1024)
    batch_size = min(ppo.get("batch_size", 8192), buffer_size)
    
    net_arch = params.get("policy_kwargs", {}).get("net_arch", [512, 512, 256])
    print(f"🧠 Network: {net_arch}")
    
    # Create or load agent
    model_checkpoint = ppo.get("model_checkpoint", "0")
    
    if model_checkpoint == "0":
        print("🆕 Creating new agent...")
        agent = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=ppo.get("n_steps", 1024),
            batch_size=batch_size,
            n_epochs=ppo.get("n_epochs", 8),
            gamma=ppo.get("gamma", 0.99),
            gae_lambda=ppo.get("gae_lambda", 0.95),
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=ppo.get("vf_coef", 0.5),
            max_grad_norm=ppo.get("max_grad_norm", 0.5),
            tensorboard_log=tb_folder,
            device=device,
            policy_kwargs={"net_arch": net_arch}
        )
    else:
        checkpoint_path = os.path.join(model_folder, model_checkpoint)
        if not checkpoint_path.endswith(".zip"):
            checkpoint_path += ".zip"
        
        print(f"📂 Loading from: {checkpoint_path}")
        agent = PPO.load(
            checkpoint_path,
            env=env,
            device=device,
            learning_rate=learning_rate,
            clip_range=clip_range,
            ent_coef=ent_coef,
        )
    
    # Save handler for self-play
    save_handler = SaveHandler(
        agent=None,
        save_freq=params["self_play_settings"]["save_freq"],
        max_saved=params["self_play_settings"]["max_saved_models"],
        save_path=model_folder,
        run_name="self_play_run",
        mode=SaveHandlerMode.FORCE
    )
    
    # Callbacks
    log_freq = params.get("logging", {}).get("log_freq", 10000)
    eval_freq = params.get("logging", {}).get("eval_freq", 500000)
    eval_episodes = params.get("logging", {}).get("eval_episodes", 10)
    
    callbacks = [
        LimitedCheckpointCallback(
            save_freq=params["self_play_settings"]["save_freq"],
            save_path=os.path.join(model_folder, "checkpoints"),
            name_prefix=f"phase{phase_config.phase_key}_model",
            max_keep=10,
            verbose=1
        ),
        SelfPlayCallback(save_handler),
        DetailedLoggingCallback(log_freq=log_freq, verbose=1),
    ]
    
    # Enable evaluation for Phase 3+
    phase_num = int(phase_config.phase_key) if str(phase_config.phase_key).isdigit() else 0
    if phase_num >= 3:
        callbacks.append(EvaluationCallback(eval_freq=eval_freq, n_eval_episodes=eval_episodes, verbose=1))
        print(f"🧪 Evaluation: Every {eval_freq:,} steps")
    else:
        print(f"🧪 Evaluation: Disabled for Phase {phase_config.phase_key}")
    
    # Training
    total_timesteps = ppo.get("time_steps", 5_000_000)
    print(f"\n🚀 Starting Phase {phase_config.phase_key} training for {total_timesteps:,} timesteps...")
    print(f"📊 TensorBoard: {tb_folder}")
    print(f"💾 Checkpoints: {os.path.join(model_folder, 'checkpoints')}")
    print("=" * 70 + "\n")
    
    try:
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
    
    # Save final model
    final_path = os.path.join(model_folder, f"phase{phase_config.phase_key}_final")
    agent.save(final_path)
    print(f"\n✅ Model saved to {final_path}")
    
    # Print final stats
    if GLOBAL_REWARD_MANAGER:
        print("\n" + "=" * 70)
        print("📊 FINAL TRAINING STATISTICS")
        print("=" * 70)
        print(f"Total Damage Dealt:    {GLOBAL_REWARD_MANAGER.total_damage_dealt:.0f}")
        print(f"Total Damage Received: {GLOBAL_REWARD_MANAGER.total_damage_received:.0f}")
        net = GLOBAL_REWARD_MANAGER.total_damage_dealt - GLOBAL_REWARD_MANAGER.total_damage_received
        print(f"Net Damage:            {net:+.0f}")
        print("=" * 70)
    
    env.close()


# ============================================================================
# MAIN
# ============================================================================

def main(cfg_file: str):
    # Load configuration
    with open(cfg_file, 'r') as f:
        params = yaml.safe_load(f)
    
    # Get phase from config
    phase_key = params.get("curriculum", {}).get("phase", 1)
    
    print("\n" + "=" * 70)
    print("🎮 UTMIST AI^2 Training Script v2 - Config-Driven Edition")
    print("=" * 70)
    print(f"📄 Config: {cfg_file}")
    
    # Load phase configuration
    try:
        phase_config = PhaseConfig(params, phase_key)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    phase_config.print_summary()
    
    # Run appropriate training
    if phase_config.is_navigation:
        run_navigation_training(params, phase_config)
    else:
        run_combat_training(params, phase_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UTMIST AI^2 Training - Config-Driven")
    parser.add_argument("--cfgFile", type=str, default="utmist_config_v2.yaml", help="Configuration file")
    opt = parser.parse_args()
    main(opt.cfgFile)
