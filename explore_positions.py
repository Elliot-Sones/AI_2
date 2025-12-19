#!/usr/bin/env python3
"""
Automated Position Explorer - Runs an agent that explores the map and records all reachable positions.
Works on headless servers (no display needed).

Usage:
    python explore_positions.py

Output:
    positions.json - All recorded positions with valid spawn bounds
"""

import sys
sys.path.insert(0, 'UTMIST-AI2-main')

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import json
import numpy as np
from environment.environment import CameraResolution, WarehouseBrawl


def explore_map(num_episodes=50, steps_per_episode=5000):
    """Run random exploration to discover all reachable positions (aerial + grounded)."""
    
    print("=" * 60)
    print("🗺️ AUTOMATED POSITION EXPLORER (EXTENSIVE AERIAL SCAN)")
    print("=" * 60)
    print(f"Running {num_episodes} episodes with {steps_per_episode} steps each...")
    print("Recording ALL reachable positions (including mid-air)...")
    
    env = WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=True)
    
    # Use sets for efficient unique storage during runtime
    all_unique_positions = set()
    grounded_positions = set()
    aerial_positions = set()
    
    x_positions_grounded = []
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        
        # Randomize start position heavily if possible (handled in reset if train_mode=True)
        # But we can also force some variety by random actions
        
        for step in range(steps_per_episode):
            # Random exploration with bias toward movement
            action = np.zeros(10, dtype=np.float32)
            
            # Movement strategy: cycle through different actions
            # fast switching to cover more ground/air
            phase = (step // 20) % 12 
            
            if phase == 0:  # Walk right
                action[3] = 1.0  # D
            elif phase == 1:  # Walk left
                action[1] = 1.0  # A
            elif phase == 2:  # Jump right
                action[3] = 1.0  # D
                action[4] = 1.0  # Space
            elif phase == 3:  # Jump left
                action[1] = 1.0  # A
                action[4] = 1.0  # Space
            elif phase == 4:  # Jump in place
                action[4] = 1.0  # Space
            elif phase == 5:  # High jump (hold W + Space)
                action[0] = 1.0  # W
                action[4] = 1.0  # Space
            elif phase == 6:  # Random spam
                action[np.random.randint(0, 5)] = 1.0
            elif phase == 7:  # Double jump right
                action[3] = 1.0
                if step % 15 < 5: action[4] = 1.0
            elif phase == 8:  # Double jump left
                action[1] = 1.0
                if step % 15 < 5: action[4] = 1.0
            elif phase == 9: # Drop down / Fast fall
                action[2] = 1.0 # S
            elif phase == 10: # Aerial drift right
                action[3] = 1.0
                if step % 10 == 0: action[4] = 1.0 # Occasional flutter
            elif phase == 11: # Aerial drift left
                action[1] = 1.0
                if step % 10 == 0: action[4] = 1.0 

            
            # Opponent stands still
            opp_action = np.zeros(10, dtype=np.float32)
            
            observations, rewards, terminated, truncated, info = env.step({0: action, 1: opp_action})
            
            # Record position
            player = env.players[0]
            x = float(player.body.position.x)
            y = float(player.body.position.y)
            
            grounded = player.is_on_floor()
            
            # Round for storage efficiency (1 decimal place = ~0.1 unit precision)
            pos_key = (round(x, 1), round(y, 1))
            
            all_unique_positions.add(pos_key)
            if grounded:
                grounded_positions.add(pos_key)
                x_positions_grounded.append(x)
            else:
                aerial_positions.add(pos_key)
            
            if terminated:
                observations, _ = env.reset()
        
        progress = (episode + 1) / num_episodes * 100
        print(f"Episode {episode + 1}/{num_episodes} complete ({progress:.0f}%) - Unique positions so far: {len(all_unique_positions)}")
    
    env.close()
    
    # Process results
    if not all_unique_positions:
        print("⚠️ No positions recorded!")
        return

    # Calculate bounds
    grounded_list = list(grounded_positions)
    x_coords_g = [p[0] for p in grounded_list]
    y_coords_g = [p[1] for p in grounded_list]
    
    all_list = list(all_unique_positions)
    x_coords_all = [p[0] for p in all_list]
    y_coords_all = [p[1] for p in all_list]
    
    results = {
        'bounds': {
            'x_min': float(min(x_coords_all)),
            'x_max': float(max(x_coords_all)),
            'y_min': float(min(y_coords_all)), # Visual Top
            'y_max': float(max(y_coords_all))  # Visual Bottom
        },
        'stats': {
            'total_samples': num_episodes * steps_per_episode,
            'unique_total': len(all_unique_positions),
            'unique_grounded': len(grounded_positions),
            'unique_aerial': len(aerial_positions)
        },
        'grounded_positions': [{'x': x, 'y': y} for x, y in sorted(grounded_list)],
        'all_positions': [{'x': x, 'y': y, 'g': 1 if (x,y) in grounded_positions else 0} for x, y in sorted(all_list)]
    }
    
    # Save to file
    with open('positions.json', 'w') as f:
        json.dump(results, f, indent=None) # Compact save to save space
    
    print("\n" + "=" * 60)
    print("📊 EXTENSIVE EXPLORATION RESULTS")
    print("=" * 60)
    print(f"Total Unique Positions: {len(all_unique_positions)}")
    print(f"  - Grounded: {len(grounded_positions)}")
    print(f"  - Aerial:   {len(aerial_positions)}")
    print(f"\n🎯 FULL REACHABLE BOUNDS:")
    print(f"  X: [{min(x_coords_all):.2f}, {max(x_coords_all):.2f}]")
    print(f"  Y: [{min(y_coords_all):.2f}, {max(y_coords_all):.2f}]")
    print(f"\n💾 Saved to: positions.json")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    explore_map(num_episodes=50, steps_per_episode=5000)
