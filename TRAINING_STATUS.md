# UTMIST Training Status

## Current State: Phase 4 (Win Focus) @ ~1M steps

### Training Progress Summary

| Phase | Status | Key Achievement |
|-------|--------|-----------------|
| Phase 1 | ✅ Complete | Agent learned to approach + stay on stage |
| Phase 2 | ✅ Complete | Agent learned to deal damage, +44 net damage |
| Phase 3 | ✅ Complete | 90% vs ConstantAgent, 40% vs BasedAgent |
| Phase 4 | 🔄 In Progress | Win-focused training |

---

## Latest Eval Results (Phase 4 @ 1M steps)

| Opponent | Win Rate | Damage Dealt |
|----------|----------|--------------|
| **phase1_final** | **100%** | 43 |
| **ConstantAgent** | **70%** | 194 |
| **ClockworkAgent** | **60%** | 152 |
| **RandomAgent** | **60%** | 55 |
| **phase2_final** | **40%** | 38 |
| **BasedAgent** | **20%** | 97 |

---

## Key Metrics

- **Net Damage:** +20 to +63 per episode
- **Damage Dealt:** 64-93 per episode
- **Mean Reward:** -3.6 (adjusting to win focus)

---

## Model Checkpoints to Resume

| File | Description |
|------|-------------|
| `results/ppo_utmist_v2/model/phase3_final.zip` | Best Phase 3 model |
| `results/ppo_utmist_v2/model/checkpoints/` | Latest checkpoints |

---

## Configuration for Resume

**Config file:** `utmist_config_v2.yaml`
- Phase: 4
- Checkpoint: phase3_final
- Opponent mix: 70% diverse, 30% self-play
- Edge penalty: -1.0

---

## To Resume Training

```bash
source venv/bin/activate
python3 train_utmist_v2.py --cfgFile utmist_config_v2.yaml
```

---

## Remaining Work

1. **Continue Phase 4 training** (~30M steps total)
2. **Monitor win rates** - goal is 60%+ vs diverse opponents
3. **Watch for SDs** - death tracking now available in logs
4. **BasedAgent** - main challenge, currently at 20%

---

- ✅ Fixed win detection (was using reward, now uses lives comparison)
- ✅ Added SD/KO death tracking
- ✅ Increased edge penalty for stage awareness
- 🔄 Implemented "Operation Giant Slayer" (Phase 4 opponent mix optimization)

## Next Steps (Operation Giant Slayer)

Current Focus: **Beating BasedAgent**
- **Opponent Mix**:
  - 40% Self-Play (Continual learning)
  - 40% BasedAgent (Hard teacher)
  - 10% Clockwork (Pattern recognition)
  - 10% Random (Noise robustness)
  - 0% Constant (Dropped - too easy)

**Command to Resume:**
```bash
source venv/bin/activate
python3 train_utmist_v2.py --cfgFile utmist_config_v2.yaml
```
