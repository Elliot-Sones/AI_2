#!/usr/bin/env python3
"""
Position Recording Tool - Play the game manually and record all reachable positions.
Run this script, move around using keyboard, and it will save all positions you visit.

Usage:
    python record_positions.py

Controls:
    A/D - Move left/right
    W - Aim up
    S - Crouch/Fall through
    Space - Jump
    Q - Quit and save positions

Output:
    positions.json - All recorded positions with min/max bounds
"""

import sys
sys.path.insert(0, 'UTMIST-AI2-main')

import os
import json
import numpy as np
import pygame
from environment.environment import CameraResolution, WarehouseBrawl

# Initialize pygame for keyboard input
pygame.init()

def main():
    print("=" * 60)
    print("🎮 POSITION RECORDING TOOL")
    print("=" * 60)
    print("Controls:")
    print("  A/D    - Move left/right")
    print("  W      - Aim up") 
    print("  S      - Crouch/Fall through")
    print("  Space  - Jump")
    print("  Q      - Quit and save positions")
    print("=" * 60)
    
    # Create environment
    env = WarehouseBrawl(resolution=CameraResolution.LOW, train_mode=False)
    env.max_timesteps = 30 * 300  # 5 minutes
    
    observations, _ = env.reset()
    
    # Position tracking
    positions = []
    x_positions = []
    y_positions = []
    
    # Create a small pygame window for keyboard input
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption("Position Recorder - Press Q to quit")
    font = pygame.font.Font(None, 24)
    
    running = True
    step = 0
    
    while running and step < env.max_timesteps:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Check for quit
        if keys[pygame.K_q]:
            running = False
            break
        
        # Build action vector: [W, A, S, D, Space, H, L, J, K, G]
        action = np.zeros(10, dtype=np.float32)
        action[0] = 1.0 if keys[pygame.K_w] else 0.0  # W
        action[1] = 1.0 if keys[pygame.K_a] else 0.0  # A
        action[2] = 1.0 if keys[pygame.K_s] else 0.0  # S
        action[3] = 1.0 if keys[pygame.K_d] else 0.0  # D
        action[4] = 1.0 if keys[pygame.K_SPACE] else 0.0  # Space
        
        # Opponent does nothing
        opp_action = np.zeros(10, dtype=np.float32)
        
        # Step environment
        observations, rewards, terminated, truncated, info = env.step({0: action, 1: opp_action})
        
        # Get player position
        player = env.players[0]
        x = player.body.position.x
        y = player.body.position.y
        
        positions.append({'x': float(x), 'y': float(y), 'step': step})
        x_positions.append(x)
        y_positions.append(y)
        
        # Update display
        screen.fill((30, 30, 30))
        text = font.render(f"Pos: ({x:.2f}, {y:.2f}) | Points: {len(positions)}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        bounds_text = font.render(f"X: [{min(x_positions):.2f}, {max(x_positions):.2f}] Y: [{min(y_positions):.2f}, {max(y_positions):.2f}]", True, (200, 200, 200))
        screen.blit(bounds_text, (10, 40))
        quit_text = font.render("Press Q to quit and save", True, (100, 255, 100))
        screen.blit(quit_text, (10, 70))
        pygame.display.flip()
        
        step += 1
        
        if terminated or truncated:
            observations, _ = env.reset()
        
        # Limit to 30 fps
        pygame.time.delay(33)
    
    # Save results
    pygame.quit()
    env.close()
    
    # Calculate bounds
    x_min, x_max = min(x_positions), max(x_positions)
    y_min, y_max = min(y_positions), max(y_positions)
    
    results = {
        'bounds': {
            'x_min': float(x_min),
            'x_max': float(x_max),
            'y_min': float(y_min),
            'y_max': float(y_max)
        },
        'total_points': len(positions),
        'positions': positions
    }
    
    # Save to file
    with open('positions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 POSITION RECORDING RESULTS")
    print("=" * 60)
    print(f"Total positions recorded: {len(positions)}")
    print(f"\nBounds discovered:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"\n💾 Saved to: positions.json")
    print("=" * 60)
    
    print("\n🔧 Suggested spawn bounds for FrozenOpponentWrapper:")
    print(f"    STAGE_X_MIN = {x_min:.1f}")
    print(f"    STAGE_X_MAX = {x_max:.1f}")
    print(f"    STAGE_Y_MIN = {y_min:.1f}")
    print(f"    STAGE_Y_MAX = {y_max:.1f}")

if __name__ == "__main__":
    main()
