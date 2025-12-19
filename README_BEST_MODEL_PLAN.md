README_BEST_MODEL_PLAN

Purpose
This is a concrete, step-by-step plan to train the strongest possible model
using the current codebase, rewards, and environment logic.

Phase 0: Navigation Foundations (0a -> 0e)
- Train 0a through 0e in order; only advance after:
  - targets_reached > ~80% and falls are low (goal: <2% per episode).
- Use positions.json for target spawning:
  - Include aerial targets in later nav phases so recovery routes are learned.
  - Avoid unreachable targets by sampling only from grounded/aerial lists.

Critical Reward Fixes (Do These First)
1) Fix win reward sign (train_utmist_v2.py).
   Current logic flips win/loss. Use:
   winner_is_agent = (agent == 'player')

2) Fix Phase 1 KO reward direction (utmist_config_v2.yaml).
   The reward manager treats knockout as "agent got KO?" so:
   - Set Phase 1 knockout to a positive value (e.g., +3.0), not -3.0.

3) Aggression reward is currently inactive.
   It checks player.state_machine, which does not exist.
   Options:
   - Remove aggression weights from the YAML, or
   - Update the check to use player.state (AttackState) in train_utmist_v2.py.

Match Length Consistency (Train/Eval)
- Decide your match length target (90s or full 5 min).
- Set it consistently in:
  - SelfPlayWarehouseBrawl raw_env.max_timesteps after creation
  - EvaluationCallback and record_demo_video
- Note: utmist_config_v2.yaml has environment_settings.max_timesteps but it is
  not currently wired into the training env; it should be.

Combat Phases: Reward Shaping That Matches Winning
Phase 1 (Approach)
- High distance reward, high edge penalty, positive KO reward
- No damage reward yet (keeps focus on stage control + survival)

Phase 2 (Learn to Hit)
- Add modest damage_dealt and damage_taken penalties
- Keep edge penalty high to reduce self-destructs

Phase 3 (Dominate)
- Reduce distance reward to near zero
- Increase win and KO weights to prioritize match wins

Phase 4 (Pure Competition)
- Distance reward = 0
- Maximize win reward and KO reward
- Keep damage terms small (damage is only a proxy for KO)

Use positions.json to Improve Edge Penalty
- positions.json gives real bounds:
  x_min ~ -12.4, x_max ~ 11.8, y_min ~ -7.5, y_max ~ 8.5
- Start edge penalty around 80-85% of these bounds to reduce SDs without
  discouraging edge-guarding.

Opponent Mix and Self-Play
- Keep a diverse opponent pool in later phases:
  - Self-play for adaptation
  - Random/constant/clockwork to avoid overfitting
- Increase self-play weight in Phase 3/4 but keep 10-20% easy opponents.

Evaluation and Model Selection
- Rank checkpoints by:
  1) Win rate
  2) KO speed (faster wins)
  3) SD rate (lower is better)
- Always test on multiple opponents and seeds.

Recommended Next Actions
1) Apply the three critical reward fixes above.
2) Wire environment_settings.max_timesteps into train/eval for consistency.
3) Update FrozenOpponentWrapper to sample aerial targets in later nav phases.
4) Run Phase 0 to completion, then Phase 1->4 with the reward schedule above.
