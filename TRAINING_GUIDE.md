# UTMIST AI^2 Training Guide

> A comprehensive guide to understanding and using the training system for the UTMIST fighting game agent.

---

## Table of Contents

1. [Overview](#overview)
2. [Training Philosophy](#training-philosophy)
3. [Phase System](#phase-system)
4. [Configuration](#configuration)
5. [Running Training](#running-training)
6. [Key Components](#key-components)
7. [Reward System](#reward-system)
8. [Video Recording](#video-recording)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The training system uses **PPO (Proximal Policy Optimization)** from Stable Baselines 3 to train a reinforcement learning agent to fight in the UTMIST AI^2 game. The system is designed with a **curriculum learning** approach, progressing from basic navigation to complex combat.

### File Structure

```
AI_2/
├── train_utmist_v2.py      # Main training script
├── utmist_config_v2.yaml   # All configuration in one place
├── watch_games.py          # Watch/record trained agent
├── eval_10_games.py        # Evaluate against opponents
└── results/
    └── ppo_utmist_v2/
        ├── model/          # Saved models & checkpoints
        ├── videos/         # Demo videos
        ├── tb/             # TensorBoard logs (combat)
        └── tb_nav/         # TensorBoard logs (navigation)
```

---

## Training Philosophy

### Curriculum Learning

Instead of throwing the agent into full combat immediately, we teach skills progressively:

```
Phase 0 (Navigation) → Phase 1-4 (Combat)
```

**Why this matters:**
- An agent that can't move reliably will waste training time falling off the stage
- Navigation skills transfer directly to combat (approach, retreat, positioning)
- Reduces self-destruct (SD) rate significantly

### Opponent Modeling (Simple Context)

The agent uses a rolling buffer of the **last 60 opponent actions** appended to observations:
- Converts temporal patterns → spatial features
- No recurrent networks needed (faster training)
- Agent can "see" opponent patterns directly

---

## Phase System

### Phase 0: Navigation (0a → 0e)

**Goal:** Learn to move around the stage without dying.

| Phase | Name | Distance | Targets | Time | Description |
|-------|------|----------|---------|------|-------------|
| **0a** | Ground Navigation | 1-2 | 1 | 90s | Walk to nearby targets |
| **0b** | Platform Navigation | 3-5 | 1 | 90s | Jump to reach targets |
| **0c** | Full Stage Navigation | 5-12 | 1 | 90s | Navigate anywhere |
| **0d** | Multi-Target Navigation | 1-12 | 3 | 60s | Sequential targets |
| **0e** | Speed Navigation | 1-12 | 3 | 30s | Time pressure |

**How it works:**
1. Opponent is **frozen** at a random position (acts as target marker)
2. Agent is rewarded for approaching the target
3. Bonus reward when reaching target (distance < 1.0)
4. Penalty for falling off stage
5. Penalty for staying near edges

### Phase 1-4: Combat

| Phase | Name | Focus | Opponents |
|-------|------|-------|-----------|
| **1** | Learn to Approach | Distance, avoid death | 90% Random, 10% Self-play |
| **2** | Learn to Hit | Damage dealing | 60% Random, 40% Self-play |
| **3** | Learn to Dominate | Win rate | Mixed + 30% Self-play |
| **4** | Pure Competition | Winning | 40% Based, 40% Self-play |

---

## Configuration

All settings are in `utmist_config_v2.yaml`. Here's what each section does:

### Curriculum Control

```yaml
curriculum:
  phase: "0a"  # Current phase to train
```

Change this to switch phases. Valid values: `"0a"`, `"0b"`, `"0c"`, `"0d"`, `"0e"`, `1`, `2`, `3`, `4`

### PPO Settings

```yaml
ppo_settings:
  gamma: 0.99              # Discount factor (how much future rewards matter)
  gae_lambda: 0.95         # GAE lambda for advantage estimation
  learning_rate: [3.0e-4, 1.0e-6]  # Linear schedule: start → end
  clip_range: [0.2, 0.05]  # PPO clip range schedule
  ent_coef: 0.01           # Entropy coefficient (exploration)
  nav_ent_coef: 0.1        # Higher entropy for navigation
  batch_size: 8192         # Batch size for updates
  n_epochs: 8              # Epochs per update
  n_steps: 1024            # Steps per environment before update
  time_steps: 5000000      # Total training steps
  model_checkpoint: "0"    # "0" = new, or filename to resume
```

### Environment Settings

```yaml
environment_settings:
  n_envs: 32        # Parallel environments (32 for GPU, 8 for CPU)
  resolution: "LOW" # Camera resolution
```

### Opponent History (Simple Context)

```yaml
opponent_history:
  enabled: true       # Enable opponent action tracking
  history_length: 60  # ~2 seconds of actions at 30fps
  action_dim: 10      # UTMIST action space size
```

---

## Running Training

### Basic Usage

```bash
# Train current phase (from config)
python train_utmist_v2.py

# Use custom config file
python train_utmist_v2.py --cfgFile my_config.yaml
```

### Switching Phases

1. Edit `utmist_config_v2.yaml`:
   ```yaml
   curriculum:
     phase: "0b"  # Change from "0a" to "0b"
   ```
2. Run training again

### Resuming Training

1. Edit `utmist_config_v2.yaml`:
   ```yaml
   ppo_settings:
     model_checkpoint: "nav_0a_final"  # or "phase1_final"
   ```
2. Run training

### Stopping Training

- **Ctrl+C**: Saves model + records demo video (safe to use)
- Wait for "Model saved" message before closing terminal

---

## Key Components

### Environment Wrappers

```
Base Environment (SelfPlayWarehouseBrawl)
    ↓
Float32Wrapper           # Ensure observations are float32
    ↓
FrozenOpponentWrapper    # [Phase 0 only] Freeze opponent, add nav rewards
    --- OR ---
DamageTrackingWrapper    # [Combat] Track damage dealt/received
    ↓
OpponentHistoryWrapper   # Add 60-frame opponent action history
    ↓
Monitor                  # Episode stats tracking
    ↓
VecFrameStack            # Stack 4 frames for temporal info
```

### Callbacks

| Callback | Purpose | Phase |
|----------|---------|-------|
| `CheckpointCallback` | Save model every 500k steps | All |
| `LimitedCheckpointCallback` | Keep only last N checkpoints | Combat |
| `SelfPlayCallback` | Save models for self-play pool | Combat |
| `DetailedLoggingCallback` | Log damage/winrate stats | Combat |
| `EvaluationCallback` | Test vs all opponents | Phase 3+ |
| `VideoRecordingCallback` | Record demo every 1M steps | Combat |

### Neural Network

```yaml
policy_kwargs:
  net_arch: [512, 512, 256]  # 3-layer MLP
```

Input: `(observation_size × 4 frames) + (60 × 10 opponent history)` = flattened vector

---

## Reward System

### Phase 0 Navigation Rewards

| Reward | Value | Description |
|--------|-------|-------------|
| Approach | +2.0 × distance_reduction | Moving toward target |
| Target Reached | +5.0 | Within 1.0 units of target |
| Edge Penalty | -0.5 | When x > 10.0 |
| Fall | -3.0 to -8.0 | Falling off stage (increases with phase) |

### Combat Rewards (Phase-dependent)

Configured per phase in YAML:

```yaml
rewards:
  distance: 2.0        # Closing distance
  aggression: 0.5      # Being in attack states
  damage_dealt: 0.0    # Per point of damage dealt
  damage_taken: 0.0    # Per point of damage taken (negative)
  net_damage: 0.0      # Damage dealt - damage taken
  win: 0.0             # End-of-game win bonus
  knockout: -3.0       # Losing a life (negative = penalty)
  edge_penalty: -1.0   # Being near stage edges
```

**Phase Progression Example:**
- Phase 1: High distance reward, no damage rewards, high death penalty
- Phase 4: No distance reward, high win reward, balanced damage

---

## Video Recording

### When Videos Are Recorded

| Situation | Video Name | Phase |
|-----------|------------|-------|
| Ctrl+C interrupt | `nav_0a_interrupted.mp4` | Navigation |
| Training complete | `nav_0a_final.mp4` | Navigation |
| Every 1M steps | `phase1_step1000k.mp4` | Combat |

### Video Location

```
./results/ppo_utmist_v2/videos/
```

### Manual Recording

Use `watch_games.py` for more control:

```bash
# Watch 1 game vs RandomAgent
python watch_games.py

# 5 games vs BasedAgent
python watch_games.py --opponent based --games 5

# No video (faster evaluation)
python watch_games.py --opponent based --games 10 --no-video
```

---

## Troubleshooting

### Common Issues

**Training is very slow**
- Reduce `n_envs` if low on RAM/VRAM
- Use `resolution: "LOW"` in config
- Check GPU is being used (should print "🚀 NVIDIA GPU detected!")

**Agent keeps falling off stage**
- Train Phase 0 longer
- Increase `edge_penalty` in combat phases
- Increase `knockout` penalty

**High self-destruct rate in combat**
- Phase 0 might not be trained enough
- Increase `edge_penalty`
- The agent might be too aggressive - reduce `aggression` reward

**Model not improving**
- Check TensorBoard: `tensorboard --logdir ./results/ppo_utmist_v2/tb`
- Try adjusting learning rate
- Ensure opponent mix includes enough easy opponents

### Useful Commands

```bash
# Monitor training with TensorBoard
tensorboard --logdir ./results/ppo_utmist_v2/tb

# Quick evaluation
python eval_10_games.py --model ./results/ppo_utmist_v2/model/phase4_final.zip

# Watch specific model
python watch_games.py --model ./results/ppo_utmist_v2/model/nav_0a_final.zip --opponent constant
```

---

## Quick Reference

### Training Phase 0

```bash
# Edit config: set phase to "0a"
python train_utmist_v2.py
# Press Ctrl+C when satisfied
# Edit config: set phase to "0b", set model_checkpoint to "nav_0a_interrupted" or "nav_0a_final"
# Repeat for 0c, 0d, 0e
```

### Training Combat Phases

```bash
# Edit config: set phase to 1
# Optionally set model_checkpoint to nav_0e_final to transfer navigation skills
python train_utmist_v2.py
# Let it run for full time_steps or interrupt when satisfied
# Progress through phases 2, 3, 4
```

### Final Evaluation

```bash
python eval_10_games.py --model ./results/ppo_utmist_v2/model/phase4_final.zip
python watch_games.py --model ./results/ppo_utmist_v2/model/phase4_final.zip --opponent based --games 5
```
