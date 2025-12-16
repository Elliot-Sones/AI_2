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
    CameraResolution, RewardManager, RandomAgent, BasedAgent, ConstantAgent,
    ClockworkAgent, SelfPlayLatest, SelfPlayRandom, RewTerm
)


# ============================================================================
# PHASE CONFIGURATIONS
# ============================================================================

PHASE_CONFIGS = {
    1: {
        "name": "Learn to Approach",
        "description": "Focus on closing distance while staying on stage",
        "rewards": {
            "distance": 2.0,
            "aggression": 0.5,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "net_damage": 0.0,
            "win": 0.0,
            "knockout": -3.0,            # Death penalty from day 1!
            "edge_penalty": -1.0,        # Strong edge penalty
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
            "edge_penalty": -1.0,        # Consistent edge awareness
        },
        "opponents": {
            "random_agent": 0.6,
            "self_play": 0.4,
        }
    },
    3: {
        "name": "Learn to Dominate",
        "description": "Focus on winning with balanced rewards",
        "rewards": {
            "distance": 0.1,
            "aggression": 0.5,          # Reward for attacking
            "damage_dealt": 0.3,        # Keep fighting incentive
            "damage_taken": -0.3,
            "net_damage": 0.3,          # Reward advantage
            "win": 8.0,                 # Win reward
            "knockout": 6.0,            # KO opponent = +6, Get KO'd = -6 (death penalty)
            "edge_penalty": -1.5,       # Increased to reduce SDs
        },
        "opponents": {
            "random_agent": 0.7,        # 70% diverse agents for generalization
            "self_play": 0.3,           # 30% self-play
        }
    },
    4: {
        "name": "Pure Competition",
        "description": "Win-focused with diverse opponents",
        "rewards": {
            "distance": 0.0,
            "aggression": 0.2,           # Some aggression reward
            "damage_dealt": 0.2,         # Small damage incentive
            "damage_taken": -0.1,
            "net_damage": 0.1,
            "win": 10.0,                 # High win reward
            "knockout": 5.0,
            "edge_penalty": -1.0,        # Reduced for more aggressive play
        },
        "opponents": {
            "random_agent": 0.7,         # 70% diverse (prevent overfitting)
            "self_play": 0.3,            # 30% self-play
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


class DamageTrackingWrapper(gym.Wrapper):
    """Tracks damage dealt/received and death types (SD vs KO)."""
    def __init__(self, env):
        super().__init__(env)
        self.total_damage_dealt = 0.0
        self.total_damage_received = 0.0
        self.episode_damage_dealt = 0.0
        self.episode_damage_received = 0.0
        self.last_damage_done = 0.0
        self.last_damage_taken = 0.0
        self.wins = 0
        self.losses = 0
        # Death type tracking
        self.last_lives = 3
        self.recent_damage_taken = 0.0  # Damage taken in last N frames
        self.damage_history = []  # Track recent damage for SD detection
        self.episode_sds = 0  # Self-destructs this episode
        self.episode_kos = 0  # KO deaths this episode
        self.total_sds = 0
        self.total_kos = 0
    
    def _get_player(self):
        """Get player object from the environment, handling various wrapper levels."""
        env = self.env
        # Try multiple paths to find the player
        # Path 1: Direct raw_env access (SelfPlayWarehouseBrawl)
        if hasattr(env, 'raw_env') and hasattr(env.raw_env, 'players'):
            return env.raw_env.players[0]
        # Path 2: Through unwrapped
        if hasattr(env, 'unwrapped'):
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'raw_env') and hasattr(unwrapped.raw_env, 'players'):
                return unwrapped.raw_env.players[0]
            if hasattr(unwrapped, 'players'):
                return unwrapped.players[0]
        return None
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Try to extract damage from underlying environment
        try:
            player = self._get_player()
            if player is not None:
                # Calculate delta damage
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
                
                # Track damage history for SD detection (last 30 frames = 1 second)
                self.damage_history.append(delta_taken)
                if len(self.damage_history) > 30:
                    self.damage_history.pop(0)
                self.recent_damage_taken = sum(self.damage_history)
                
                self.last_damage_done = current_dealt
                self.last_damage_taken = current_taken
                
                # Check for death (lost a life)
                current_lives = getattr(player, 'lives_left', 3)
                if current_lives < self.last_lives:
                    # Player lost a life - determine if SD or KO
                    # SD = died with low recent damage (self-destruct)
                    # KO = died with significant recent damage
                    if self.recent_damage_taken < 20:
                        self.episode_sds += 1
                        self.total_sds += 1
                    else:
                        self.episode_kos += 1
                        self.total_kos += 1
                self.last_lives = current_lives
                
                # Add to info
                info['damage_dealt'] = self.episode_damage_dealt
                info['damage_received'] = self.episode_damage_received
                info['total_damage_dealt'] = self.total_damage_dealt
                info['total_damage_received'] = self.total_damage_received
        except Exception:
            pass
        
        # Track wins/losses on episode end
        if terminated or truncated:
            # Use lives comparison for accurate win detection
            try:
                player = self._get_player()
                if player is not None:
                    my_lives = getattr(player, 'lives_left', 0)
                    # Get opponent lives from environment
                    env = self.env
                    if hasattr(env, 'raw_env') and hasattr(env.raw_env, 'players'):
                        opp_lives = getattr(env.raw_env.players[1], 'lives_left', 0)
                    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'raw_env'):
                        opp_lives = getattr(env.unwrapped.raw_env.players[1], 'lives_left', 0)
                    else:
                        opp_lives = 0
                    
                    if my_lives > opp_lives:
                        self.wins += 1
                        info['win'] = 1
                    else:
                        self.losses += 1
                        info['win'] = 0
                else:
                    # Fallback to reward-based detection
                    if reward > 0:
                        self.wins += 1
                        info['win'] = 1
                    else:
                        self.losses += 1
                        info['win'] = 0
            except:
                # Fallback to reward-based detection
                if reward > 0:
                    self.wins += 1
                    info['win'] = 1
                else:
                    self.losses += 1
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
    Simplified logging callback.
    Shows: Avg damage dealt/received per episode, Net damage, Win rate, SD/KO breakdown.
    """
    def __init__(self, log_freq=20000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        # Interval tracking (reset each log)
        self.interval_damage_dealt = 0.0
        self.interval_damage_received = 0.0
        self.interval_wins = 0
        self.interval_losses = 0
        self.interval_episodes = 0
        self.interval_sds = 0
        self.interval_kos = 0
        # Overall tracking
        self.total_wins = 0
        self.total_losses = 0
        self.total_sds = 0
        self.total_kos = 0
        
    def _on_step(self) -> bool:
        # Get infos from the VecEnv
        infos = self.locals.get('infos', [])
        
        for info in infos:
            # Check for episode completion
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
        
        # Log every log_freq steps
        if self.num_timesteps % self.log_freq == 0 and self.num_timesteps > 0:
            self._log_stats()
        
        return True
    
    def _log_stats(self):
        """Log damage and win rate stats."""
        print("\n" + "=" * 60)
        print(f"📊 STATS @ {self.num_timesteps:,} steps (last {self.log_freq:,} steps)")
        print("=" * 60)
        
        # Average damage per episode (recent interval)
        if self.interval_episodes > 0:
            avg_dealt = self.interval_damage_dealt / self.interval_episodes
            avg_received = self.interval_damage_received / self.interval_episodes
            avg_net = avg_dealt - avg_received
            
            print(f"⚔️  Avg Damage Dealt:    {avg_dealt:.1f} per episode")
            print(f"🛡️  Avg Damage Received: {avg_received:.1f} per episode")
            print(f"📈 Avg Net Damage:      {avg_net:+.1f} per episode")
            
            # Log to TensorBoard
            self.logger.record("damage/avg_dealt", avg_dealt)
            self.logger.record("damage/avg_received", avg_received)
            self.logger.record("damage/avg_net", avg_net)
        
        # Win rate (recent interval)
        interval_games = self.interval_wins + self.interval_losses
        if interval_games > 0:
            win_rate = (self.interval_wins / interval_games) * 100
            print(f"🏆 Win Rate (recent):   {win_rate:.1f}% ({self.interval_wins}W / {self.interval_losses}L)")
            self.logger.record("winrate/recent", win_rate)
        
        # Overall win rate
        total_games = self.total_wins + self.total_losses
        if total_games > 0:
            overall_win_rate = (self.total_wins / total_games) * 100
            print(f"📊 Win Rate (overall):  {overall_win_rate:.1f}% ({self.total_wins}W / {self.total_losses}L)")
            self.logger.record("winrate/overall", overall_win_rate)
        
        # SD/KO breakdown
        total_deaths = self.interval_sds + self.interval_kos
        if total_deaths > 0:
            sd_pct = (self.interval_sds / total_deaths) * 100
            print(f"💀 Deaths: {self.interval_sds} SD / {self.interval_kos} KO ({sd_pct:.0f}% self-destruct)")
            self.logger.record("deaths/sds", self.interval_sds)
            self.logger.record("deaths/kos", self.interval_kos)
            self.logger.record("deaths/sd_pct", sd_pct)
            self.total_sds += self.interval_sds
            self.total_kos += self.interval_kos
        
        print("=" * 60 + "\n")
        
        # Reset interval counters
        self.interval_damage_dealt = 0.0
        self.interval_damage_received = 0.0
        self.interval_wins = 0
        self.interval_losses = 0
        self.interval_episodes = 0
        self.interval_sds = 0
        self.interval_kos = 0


class EvaluationCallback(BaseCallback):
    """
    Periodically evaluate agent against diverse opponents.
    Runs every eval_freq steps (default 500k).
    """
    def __init__(self, eval_freq=500000, n_eval_episodes=5, verbose=1, run_initial=True):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_step = 0
        self.run_initial = run_initial
        self.did_initial = False
        
    def _on_step(self) -> bool:
        # Run initial evaluation at start
        if self.run_initial and not self.did_initial:
            self.did_initial = True
            self._run_evaluation()
        # Regular periodic evaluation
        elif self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self._run_evaluation()
        return True
    
    def _run_evaluation(self):
        """Run evaluation matches against test opponents."""
        print("\n" + "=" * 50)
        print(f"EVALUATION @ {self.num_timesteps:,} steps")
        print("=" * 50)
        
        try:
            from environment.environment import CameraResolution, WarehouseBrawl
            from environment.agent import RandomAgent, BasedAgent, ConstantAgent, ClockworkAgent
            from stable_baselines3 import PPO
            import os
            
            # Create a simple evaluation function
            results = {}
            
            # Test opponents - built-in agents
            test_opponents = {
                "ConstantAgent": ConstantAgent(),   # Does nothing (easiest)
                "RandomAgent": RandomAgent(),       # Random actions
                "ClockworkAgent": ClockworkAgent(), # Preset patterns
                "BasedAgent": BasedAgent(),         # Rule-based (hardest built-in)
            }
            
            # Try to add previous phase models as opponents
            model_folder = "./results/ppo_utmist_v2/model"
            for phase_name in ["phase1_final", "phase2_final"]:
                model_path = os.path.join(model_folder, f"{phase_name}.zip")
                if os.path.exists(model_path):
                    try:
                        class ModelOpponent:
                            def __init__(self, model_path):
                                self.model = PPO.load(model_path, device='cpu')
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
                        test_opponents[phase_name] = ModelOpponent(model_path)
                    except Exception as e:
                        print(f"  Could not load {phase_name}: {e}")
            
            for opponent_name, opponent in test_opponents.items():
                wins = 0
                losses = 0
                draws = 0
                total_damage_dealt = 0
                total_damage_received = 0
                
                for ep in range(self.n_eval_episodes):
                    try:
                        # Run a quick evaluation episode
                        env = WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=True)
                        env.max_timesteps = 30 * 90  # 90 second match (longer for decisive outcomes)
                        
                        observations, _ = env.reset()
                        obs = observations[0]
                        opponent.get_env_info(env)
                        opponent_obs = observations[1]
                        
                        # Frame stacking for evaluation
                        frame_stack = [obs.copy() for _ in range(4)]
                        
                        done = False
                        step_count = 0
                        while not done:
                            # Stack frames
                            stacked_obs = np.concatenate(frame_stack, axis=0)
                            
                            # Get actions
                            action, _ = self.model.predict(stacked_obs, deterministic=True)
                            opp_action = opponent.predict(opponent_obs)
                            
                            full_action = {0: action, 1: opp_action}
                            observations, rewards, terminated, truncated, _ = env.step(full_action)
                            
                            obs = observations[0]
                            opponent_obs = observations[1]
                            
                            # Update frame stack
                            frame_stack.pop(0)
                            frame_stack.append(obs.copy())
                            
                            done = terminated or truncated
                            step_count += 1
                        
                        # Get stats
                        stats = env.get_stats(0)
                        opp_stats = env.get_stats(1)
                        
                        # Determine outcome
                        if stats.lives_left > opp_stats.lives_left:
                            wins += 1
                        elif stats.lives_left < opp_stats.lives_left:
                            losses += 1
                        else:
                            # Tie-breaker: who dealt more damage?
                            if stats.damage_done > opp_stats.damage_done:
                                wins += 1  # Count as win if dominated damage
                            else:
                                draws += 1
                        
                        total_damage_dealt += stats.damage_done
                        total_damage_received += stats.damage_taken
                        
                        env.close()
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"  Eval episode error: {e}")
                        continue
                
                # Calculate results
                total_games = wins + losses + draws
                win_rate = (wins / total_games) * 100 if total_games > 0 else 0
                avg_dealt = total_damage_dealt / max(1, total_games)
                avg_received = total_damage_received / max(1, total_games)
                
                results[opponent_name] = {
                    "win_rate": win_rate,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "games": total_games,
                    "avg_damage_dealt": avg_dealt,
                    "avg_damage_received": avg_received,
                }
                
                # Log results - clean format for terminal
                print(f"\n  ┌─ vs {opponent_name} ─────────────────────")
                print(f"  │ WIN_RATE: {win_rate:.0f}%  ({wins}W/{losses}L/{draws}D)")
                print(f"  │ DMG_DEALT: {avg_dealt:.0f}  DMG_RECV: {avg_received:.0f}")
                print(f"  └────────────────────────────────────")
                
                # Log to TensorBoard
                self.logger.record(f"eval/{opponent_name}/win_rate", win_rate)
                self.logger.record(f"eval/{opponent_name}/avg_damage_dealt", avg_dealt)
                self.logger.record(f"eval/{opponent_name}/avg_damage_received", avg_received)
            
            print("\n" + "=" * 50)
            print("EVAL COMPLETE")
            print("=" * 50 + "\n")
            
            # Record a video every 1M steps
            if self.num_timesteps > 0 and self.num_timesteps % 1000000 < self.eval_freq:
                self._record_demo_video()
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            import traceback
            traceback.print_exc()
    
    def _record_demo_video(self):
        """Record a demo video against BasedAgent."""
        try:
            import skvideo.io
            from environment.environment import CameraResolution, WarehouseBrawl
            from environment.agent import BasedAgent
            
            print("\n🎬 Recording demo video vs BasedAgent...")
            
            # Ensure videos folder exists
            os.makedirs("./videos", exist_ok=True)
            video_path = f"./videos/demo_{self.num_timesteps//1000000}M.mp4"
            
            opponent = BasedAgent()
            env = WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=True)
            env.max_timesteps = 30 * 60  # 60 second match
            
            observations, _ = env.reset()
            obs = observations[0]
            opponent.get_env_info(env)
            opponent_obs = observations[1]
            
            # Frame stacking
            frame_stack = [obs.copy() for _ in range(4)]
            
            writer = skvideo.io.FFmpegWriter(video_path, outputdict={
                '-vcodec': 'libx264',
                '-pix_fmt': 'yuv420p',
                '-preset': 'fast',
                '-crf': '23',
                '-r': '30'
            })
            
            for step in range(env.max_timesteps):
                stacked_obs = np.concatenate(frame_stack, axis=0)
                action, _ = self.model.predict(stacked_obs, deterministic=True)
                opp_action = opponent.predict(opponent_obs)
                
                full_action = {0: action, 1: opp_action}
                observations, rewards, terminated, truncated, _ = env.step(full_action)
                
                obs = observations[0]
                opponent_obs = observations[1]
                frame_stack.pop(0)
                frame_stack.append(obs.copy())
                
                img = env.render()
                img = np.rot90(img, k=-1)
                img = np.fliplr(img)
                writer.writeFrame(img)
                
                if terminated or truncated:
                    break
            
            writer.close()
            
            stats = env.get_stats(0)
            opp_stats = env.get_stats(1)
            result = "WIN" if stats.lives_left > opp_stats.lives_left else "LOSS" if stats.lives_left < opp_stats.lives_left else "DRAW"
            
            print(f"   📹 Video saved: {video_path}")
            print(f"   Result: {result} | Lives: {stats.lives_left}-{opp_stats.lives_left} | Dmg: {stats.damage_done:.0f}/{stats.damage_taken:.0f}")
            
            env.close()
            
        except Exception as e:
            print(f"   Video recording error: {e}")


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
        
        # --- Edge Position Penalty ---
        edge_weight = self.weights.get("edge_penalty", 0.0)
        if edge_weight != 0:
            # Stage edges are around x = ±10-12 units
            # Penalize when beyond ±8 (approaching edge)
            x_pos = abs(player.body.position.x)
            edge_threshold = 8.0
            if x_pos > edge_threshold:
                # Penalty scales with how far past threshold
                edge_factor = (x_pos - edge_threshold) / 4.0  # Max ~1.0 at edge
                total_reward += edge_weight * min(edge_factor, 1.0)  # weight is negative
        
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
    env = DamageTrackingWrapper(env)
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
    random_pct = opponent_probs["random_agent"]
    self_play_pct = opponent_probs["self_play"]
    
    # Phase 1+2: Simple opponent mix (only Random + Self-play)
    # Phase 3+4: Diverse opponents for generalization
    if phase <= 2:
        opponent_cfg = OpponentsCfg(
            opponents={
                'random_agent': (random_pct, partial(RandomAgent)),
                'self_play': (self_play_pct, SelfPlayRandom(partial(PPO.load, device=device)))
            }
        )
        print(f"👥 Opponents: {int(random_pct*100)}% Random, {int(self_play_pct*100)}% Self-play (simple)")
    else:
        # Distribute random percentage across diverse opponents
        opponent_cfg = OpponentsCfg(
            opponents={
                'random_agent': (random_pct * 0.4, partial(RandomAgent)),
                'based_agent': (random_pct * 0.3, partial(BasedAgent)),
                'constant_agent': (random_pct * 0.2, partial(ConstantAgent)),
                'clockwork_agent': (random_pct * 0.1, partial(ClockworkAgent)),
                'self_play': (self_play_pct, SelfPlayRandom(partial(PPO.load, device=device)))
            }
        )
        print(f"👥 Opponents: {int(self_play_pct*100)}% Self-play, {int(random_pct*100)}% Diverse (Random/Based/Const/Clock)")

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
    # Note: PPO doesn't support callable schedules for ent_coef, only for learning_rate and clip_range
    ent_coef = ppo_settings.get("ent_coef", 0.01)
    
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
    
    # Only enable evaluation for Phase 3+
    callbacks = [checkpoint_callback, self_play_callback, logging_callback]
    if phase >= 3:
        eval_callback = EvaluationCallback(eval_freq=500000, n_eval_episodes=10, verbose=1)
        callbacks.append(eval_callback)
        print(f"🧪 Evaluation: Every 500k steps against diverse opponents")
    else:
        print(f"🧪 Evaluation: Disabled for Phase {phase} (will enable in Phase 3)")

    # Training
    total_timesteps = ppo_settings["time_steps"]
    print(f"\n🚀 Starting Phase {phase} training for {total_timesteps:,} timesteps...")
    print(f"📊 TensorBoard: {tensor_board_folder}")
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
