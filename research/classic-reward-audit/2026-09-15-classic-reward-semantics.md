# Classic Reward Semantics Audit - 2026-09-15

## Scope

Audited the classic combat reward path in `train.py`, `config.yaml`, and `UTMIST-AI2-main/environment/agent.py`.

## Findings

- `make_combat_env` built `RewardManager.signal_subscriptions` with bare `RewTerm` values, but `RewardManager.subscribe_signals` expects `(signal_name, RewTerm)` pairs.
- The combat factory never called `reward_manager.subscribe_signals(env.raw_env)`, so win and knockout signal rewards were never connected.
- The win callback inverted the learner winner check. `win_signal` uses `agent="player"` for the learner/player0 win case.
- Phase 1 configured `knockout: -3.0`, but `compute_knockout_reward` already negates the reward when the learner is KO'd.
- Aggression checked a nonexistent `player.state_machine.current_state.id`; the simulator exposes the current state as `player.state`, and attack state should be detected with `AttackState`.
- Combat env construction reused a module-level `GLOBAL_REWARD_MANAGER`, causing later envs with different phase configs to inherit stale weights and episode state.
- `PhaseAwareRewardManager.reset_episode` cleared only episode damage and distance, leaving stale damage baselines that could leak first-step rewards and clean-win bonuses.
- `RewardManager.reset` referenced `self.collected_signal_rewards` without assigning it, leaving pending signal rewards uncleared.

## Test Receipt

Red evidence:

`python3 tests/test_classic_reward_semantics.py -v` failed before implementation with 8 semantic failures covering subscription shape, signal rewards, phase 1 KO sign, aggression state detection, episode reset, reset typo, and global manager reuse.

Green evidence:

`python3 tests/test_classic_reward_semantics.py -v` passes 9 tests after the fix.

## Review Follow-Up

- Split damage accounting into explicit `episode_damage_*` and `lifetime_damage_*` counters; `total_damage_*` remains a compatibility alias for lifetime totals.
- Added a real-dependency integration smoke for actual `WarehouseBrawl` wrapper ordering and 2-env `SubprocVecEnv` signal isolation. It skips only when the full simulator dependency stack is unavailable.

Updated local evidence:

`python3 tests/test_classic_reward_semantics.py -v` passes 10 tests.

`python3 tests/test_classic_reward_integration.py -v` skips locally because this Python has no real `torch`; the test refuses to run against stubs.
