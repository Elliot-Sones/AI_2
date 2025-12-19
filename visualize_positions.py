#!/usr/bin/env python3
"""
Visualize all possible agent positions as a heatmap overlaid on the actual game capture.
Uses the exact same coordinate transformation as the game camera.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image


def load_positions():
    """Load positions from JSON file."""
    with open('positions.json', 'r') as f:
        return json.load(f)


def game_to_pixel(x, y, window_width, window_height):
    """
    Convert game coordinates to pixel coordinates.
    Replicates the Camera.gtp() method from the game.
    """
    screen_width_tiles = 29.8
    pixels_per_tile = window_width / screen_width_tiles
    zoom = 2.0
    scale = pixels_per_tile * zoom
    
    # Camera position (centered at origin)
    pos_x, pos_y = 0, 0
    
    # Apply the game's coordinate transformation
    px = window_width / 2 + (x - pos_x) * scale
    # Y increases downward in game, so this maps directly to pixel Y
    py = window_height / 2 + (y - pos_y) * scale
    
    return px, py


def create_heatmap_overlay():
    """Create a heatmap of reachable positions overlaid on the background."""
    
    # Load the positions data
    data = load_positions()
    
    # Check if we have the new format (all_positions list)
    if 'all_positions' in data:
        positions = data['all_positions']
        print(f"Loaded {len(positions)} positions (Aerial + Grounded)")
    else:
        positions = data['grounded_positions']
        print(f"Loaded {len(positions)} grounded positions (Old format)")
    
    bounds = data['bounds']
    
    # Use the captured background
    try:
        bg_pil = Image.open('game_background_capture.png')
    except FileNotFoundError:
        print("Warning: game_background_capture.png not found, using placeholder")
        bg_pil = Image.new('RGB', (720, 480), color='black')
        
    window_width, window_height = bg_pil.size
    print(f"Background size: {window_width} x {window_height}")
    
    bg_array = np.array(bg_pil)
    
    # Extract coordinates and type
    x_coords = []
    y_coords = []
    is_grounded = []
    
    for p in positions:
        x_coords.append(p['x'])
        y_coords.append(p['y'])
        # New format has 'g' for grounded, old has 'grounded' boolean
        if 'g' in p:
            is_grounded.append(bool(p['g']))
        elif 'grounded' in p:
            is_grounded.append(p['grounded'])
        else:
            is_grounded.append(True) # Assume grounded if unknown
            
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    is_grounded = np.array(is_grounded)
    
    # Convert positions to pixels
    pixel_x = []
    pixel_y = []
    for x, y in zip(x_coords, y_coords):
        px, py = game_to_pixel(x, y, window_width, window_height)
        pixel_x.append(px)
        pixel_y.append(py)
    
    pixel_x = np.array(pixel_x)
    pixel_y = np.array(pixel_y)
    
    # === 1. Reachability Cloud Overlay ===
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 9.33))
    ax1.imshow(bg_array)
    
    # Plot Aerial positions (Blue, semi-transparent)
    aerial_mask = ~is_grounded
    if np.any(aerial_mask):
        ax1.scatter(pixel_x[aerial_mask], pixel_y[aerial_mask], 
                   c='deepskyblue', s=15, alpha=0.15, label='Aerial Reachable', marker='o')
    
    # Plot Grounded positions (Green, solid)
    grounded_mask = is_grounded
    if np.any(grounded_mask):
        ax1.scatter(pixel_x[grounded_mask], pixel_y[grounded_mask], 
                   c='lime', s=30, alpha=0.8, label='Grounded Positions', edgecolors='darkgreen', linewidths=0.5)
    
    ax1.legend(loc='upper right', fontsize=12)
    ax1.set_title(f'Full Agent Reachability Map ({len(positions)} positions)', 
                  fontsize=14, fontweight='bold', color='white')
    ax1.axis('off')
    
    # Limit view to background size to keep aspect ratio
    ax1.set_xlim(0, window_width)
    ax1.set_ylim(window_height, 0)
    
    plt.tight_layout()
    fig1.patch.set_facecolor('black')
    plt.savefig('aerial_reachability_cloud.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    print(f"✅ Saved: aerial_reachability_cloud.png")
    plt.close(fig1)
    
    # === 2. Heatmap (Aerial + Grounded) ===
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 9.33))
    ax2.imshow(bg_array)
    
    # Create a density heatmap
    heatmap = np.zeros((window_height, window_width), dtype=np.float32)
    
    # Use smaller radius and tighter smoothing to match cloud
    for px, py in zip(pixel_x, pixel_y):
        px_int, py_int = int(px), int(py)
        if 0 <= px_int < window_width and 0 <= py_int < window_height:
             # Just add value to pixel and immediate neighbors (3x3)
             y_min = max(0, py_int-1)
             y_max = min(window_height, py_int+2)
             x_min = max(0, px_int-1)
             x_max = min(window_width, px_int+2)
             heatmap[y_min:y_max, x_min:x_max] += 1.0
             
    # Smooth the heatmap LESS (sigma=2.5 instead of 4)
    heatmap = gaussian_filter(heatmap, sigma=2.5)
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Hot colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0, 0, 0, 0),         # Transparent
        (0, 0, 1, 0.2),       # Blue (aerial trace)
        (0, 1, 0, 0.4),       # Green (common aerial)
        (1, 1, 0, 0.6),       # Yellow (high traffic)
        (1, 0, 0, 0.8),       # Red (very high traffic)
    ]
    cmap = LinearSegmentedColormap.from_list('reachability', colors)
    
    im = ax2.imshow(heatmap, cmap=cmap, alpha=0.9)
    
    cbar = plt.colorbar(im, ax=ax2, shrink=0.7, pad=0.02)
    cbar.set_label('Reachability Density', fontsize=11, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    
    ax2.set_title('Reachability Density Heatmap (Refined)', 
                  fontsize=14, fontweight='bold', color='white')
    ax2.axis('off')
    ax2.set_xlim(0, window_width)
    ax2.set_ylim(window_height, 0)
    
    plt.tight_layout()
    fig2.patch.set_facecolor('black')
    plt.savefig('aerial_heatmap.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    print(f"✅ Saved: aerial_heatmap.png")
    plt.close(fig2)
    
    # === 3. Combined Verification Plot ===
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 9.33))
    ax3.imshow(bg_array)
    # Background: Heatmap
    ax3.imshow(heatmap, cmap=cmap, alpha=0.6)
    # Foreground: Cloud points (very small)
    if np.any(aerial_mask):
         ax3.scatter(pixel_x[aerial_mask], pixel_y[aerial_mask], 
                    c='white', s=1, alpha=0.1, label='Aerial Points')
    ax3.set_title('Verification: Heatmap + Point Cloud Alignment', 
                  fontsize=14, fontweight='bold', color='white')
    ax3.axis('off')
    ax3.set_xlim(0, window_width)
    ax3.set_ylim(window_height, 0)
    plt.tight_layout()
    fig3.patch.set_facecolor('black')
    plt.savefig('verification_combined.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    print(f"✅ Saved: verification_combined.png")
    plt.close(fig3)
    
    print("\n" + "=" * 50)
    print("Generated EXTERNSIVE visualization files!")
    print("=" * 50)


if __name__ == "__main__":
    create_heatmap_overlay()
