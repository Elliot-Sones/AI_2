import os
import yaml
import torch
import sys
import numpy as np
import gymnasium as gym
import imageio
from functools import partial
from stable_baselines3 import PPO

# Add UTMIST environment to path
sys.path.append(os.path.join(os.getcwd(), "UTMIST-AI2-main"))

from environment.agent import SelfPlayWarehouseBrawl, OpponentsCfg, SaveHandler, SaveHandlerMode, CameraResolution, RewardManager, RandomAgent, RewTerm

# Re-define DistanceReward (copied from train_utmist.py)
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

# Re-define StrikingRangeReward (copied from train_utmist.py)
class StrikingRangeReward:
    def __init__(self, threshold=2.0, reward_value=5.0):
        self.threshold = threshold
        self.reward_value = reward_value

    def __call__(self, env):
        rewards = []
        # Calculate current distance between players
        p1_pos = env.players[0].body.position
        p2_pos = env.players[1].body.position
        # Pymunk Vec2d distance
        current_dist = p1_pos.get_distance(p2_pos)
        
        for i in range(2):
            if current_dist < self.threshold:
                rewards.append(self.reward_value)
            else:
                rewards.append(0.0)
        return torch.tensor(rewards)

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

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def main():
    # 1. Load Configuration
    with open("utmist_config.yaml", "r") as f:
        params = yaml.safe_load(f)

    # 2. Setup Environment (Stage 1 Settings)
    resolution_map = {
        "LOW": CameraResolution.LOW,
        "MEDIUM": CameraResolution.MEDIUM,
        "HIGH": CameraResolution.HIGH
    }
    resolution = resolution_map.get(params["environment_settings"]["resolution"], CameraResolution.LOW)

    # Force Random Agent for visualization (as in Stage 1)
    opponent_cfg = OpponentsCfg(
        opponents={
            'random_agent': (1.0, partial(RandomAgent)),
            'self_play': (0.0, None) 
        }
    )

    # Reward Manager (Stage 1)
    distance_reward = DistanceReward()
    striking_reward = StrikingRangeReward(threshold=2.0, reward_value=5.0)
    
    reward_functions = {
        "distance": RewTerm(func=distance_reward, weight=1.0),
        "striking": RewTerm(func=striking_reward, weight=1.0)
    }

    reward_manager = RewardManager(
        reward_functions=reward_functions,
        signal_subscriptions={}
    )

    # Create Environment
    env = SelfPlayWarehouseBrawl(
        opponent_cfg=opponent_cfg,
        save_handler=None,
        reward_manager=reward_manager,
        resolution=resolution
    )
    env = Float32Wrapper(env)

    # 3. Load Model
    model_path = "results/ppo_utmist/model/ppo_utmist_final.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device=get_device())

    # 4. Run Episode
    print("Starting visualization episode...")
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    frames = []

    while not done and not truncated and steps < 200:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Accumulate reward (Agent 0 is our trained agent)
        if isinstance(reward, (list, np.ndarray, torch.Tensor)):
            total_reward += float(reward[0])
        else:
            total_reward += float(reward)
        # Debug Prints
        p1_pos = env.unwrapped.raw_env.players[0].body.position
        p2_pos = env.unwrapped.raw_env.players[1].body.position
        dist = p1_pos.get_distance(p2_pos)
        
        if steps % 30 == 0: # Print every second (approx)
            print(f"Step {steps}: P1={p1_pos}, P2={p2_pos}, Dist={dist:.2f}, Reward={reward}")

        steps += 1
        
        # Capture frame
        frame = env.render()
        frames.append(frame)

        # Print progress every 100 steps
        if steps % 100 == 0:
            print(f"Step {steps}: Reward so far = {total_reward:.2f}")

    print(f"Episode finished! Total Steps: {steps}, Total Reward: {total_reward:.2f}")

    # 5. Save Video
    print("Saving video to replay.mp4...")
    imageio.mimsave("replay.mp4", frames, fps=30)
    print("Video saved successfully!")

    env.close()

if __name__ == "__main__":
    main()
