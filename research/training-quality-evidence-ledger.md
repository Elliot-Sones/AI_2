# AI_2 Training-Quality Evidence Ledger

This is the working source of truth for deciding what to fix, what to test, and
what may be promoted into a long training run. It deliberately keeps the classic
Python/SB3 path, the contact-lesson native path, and the resident native full-game
path separate. A result in one lane is not evidence for another lane.

Status vocabulary:

- **confirmed-live**: reproduced in the current executable path;
- **latent**: incorrect code exists but an earlier disconnected path prevents it
  from affecting current runs;
- **measured**: backed by a preserved run receipt;
- **hypothesis**: plausible, but not yet isolated by a controlled experiment;
- **fixed-local**: corrected on an isolated branch and covered by local tests;
- **GPU-verified**: tested on the named GPU with downloaded receipts;
- **learning-verified**: fixed-transition and held-out evaluation show the desired
  learning effect over multiple seeds.

## Current findings

| Finding | Pipeline | Status | Evidence | Required promotion test |
| --- | --- | --- | --- | --- |
| Classic win and knockout signals are not subscribed | classic `train.py` combat | confirmed-live on main; fixed-local and independently approved in `78746dd` | `make_combat_env` constructed invalid subscriptions and never called `subscribe_signals`; isolated branch now subscribes the raw env and passes focused plus order-independent real-env signal smoke | Same-seed phase-2 learning-direction check |
| Learner win is mapped as a loss after signal wiring | classic `train.py` combat | latent on main; fixed-local and independently approved in `78746dd` | game emits `agent="player"` for player-0 win; real-env smoke on the isolated fix observed learner win `[+1,-1]` and opponent win `[-1,+1]` | Same-seed phase-2 plus phases 3-4 clean-win learning checks |
| Phase-1 knockout weight double-negates learner death | classic `train.py` combat | latent on main; fixed-local and independently approved in `78746dd` | isolated config uses positive `3.0`; real-env smoke observed learner KO `[-3,+3]` and opponent KO `[+3,-3]`, including two-env SubprocVecEnv checks | Same-seed phase-1 learning-direction check |
| Aggression reward reads nonexistent `state_machine` | classic `train.py` combat | confirmed-live on main; fixed-local and independently approved in `78746dd` | isolated branch checks real `player.state` against `AttackState`; focused attack/non-attack test passes | Rollout component-frequency logging and learning-direction check |
| Phase reward baselines and clean-win totals leak across episodes | classic `train.py` combat | confirmed-live on main; fixed-local and independently approved in `78746dd` | isolated branch adds per-env manager and reset wrapper and clears damage/distance episode baselines; cross-episode and real-env tests pass in either test order | Multi-episode GPU rollout smoke |
| `RewardManager.reset` does not clear pending signal reward | generic classic wrapper | confirmed-code on main; fixed-local and independently approved in `78746dd` | isolated branch assigns `collected_signal_rewards = 0.0`; focused reset test passes | Included in same-seed classic learning check |
| `settings.action_space: discrete` does not control any audited trainer | classic, contact native, full-game native | confirmed-live | env exposes `Box(10)`; trainers require Box and do not read this setting | Delete or wire the knob after action A/B decision |
| Button actions are Gaussian values clipped to `[0,1]` and pressed only above `0.5` | classic, contact native, full-game native | confirmed-live | Python and native input paths use the same threshold | Three-arm action-representation A/B |
| Contact checkpoint attacks sampled but not deterministic | contact lesson | measured | preserved evaluation: sampled 20/20 hit episodes, deterministic 0/20; max deterministic light mean 0.02688 | Re-run preserved panel, then Gaussian vs 10 Bernoulli heads vs factored MultiDiscrete over at least 3 seeds |
| A ten-head Bernoulli policy is a safe executable alternative to Gaussian threshold buttons | contact native | fixed-local and independently approved in `7c3e3f8`; learning effect remains a hypothesis | exact binary execution, masked log-probability/entropy/gradient behavior, raw-policy versus executed-action storage, checkpoint schema, neutral-head conversion, truthful fresh-optimizer provenance, and fail-before-step nonfinite guards are covered by 45 focused local tests; 11 CUDA tests are pending | Matched 3-seed Gaussian-versus-Bernoulli CUDA A/B; deterministic initiation >=80%, hit rate >= sampled control, navigation/survival within seed spread |
| PPO target-KL stops after 1/16 steps | Sept-9 native contact profile only | measured; diagnostic runner fixed-local and independently approved in `6c81cce` | all 12 preserved control rollouts executed one optimizer step, then stopped when a subsequent minibatch KL of about 0.025-0.027 exceeded the 0.015 guard; the diagnostic runner now writes fail-closed per-minibatch pre-step KL/step-attempt rows without mixing artifacts, and local update-diagnostic tests passed; CUDA replay is still pending | Run the bounded exact-checkpoint GPU replay, then isolate batch geometry, LR, and threshold one variable at a time |
| Navigation entropy is configured five times higher than the base setting despite a “lower entropy” comment | classic `train.py` navigation | confirmed-live configuration mismatch | base `ent_coef` is `0.01`, `nav_ent_coef` is `0.05`, and `train.py` applies `nav_ent_coef`; this is higher exploration pressure, not lower | After action/reward correctness, compare a fixed or annealed lower-entropy schedule against the current value over matched seeds |
| Late configured opponent mix is 68-72% scripted and 28-32% self-play | classic curriculum | confirmed-design | phase 3 YAML is 72/28 and phase 4 is 68/32; snapshots are uniformly sampled from at most seven recent files | Current-mix vs deeper PFSP pool on frozen held-out opponents and checkpoint round robin |
| Vectorized classic self-play archive remains empty and falls back to `ConstantAgent` | classic `train.py` combat | measured on the real unpatched reset path; telemetry implementation independently reviewed in `61901d7` | four 1,000-reset phase audits matched configured-key rates within +/-3 percentage points, but all 1,050 requested self-play selections became `ConstantAgent`; the main configured `results/ppo_utmist/model/self_play_run` archive was separately inventoried read-only and contains zero files. Compressed receipts and hashes are under `baselines/classic-opponent-audit/unpatched-empty-archive/` | Integrate runtime telemetry, implement a shared immutable archive, then require realized policy/checkpoint mix within +/-3 percentage points and evaluate opponent difficulty rather than only selection frequency |
| Resident native full-game CUDA matches the original Python game end to end | full-game native | CPU replay and autoreset semantics measured; CUDA gate blocked | final producer `e6c84c3` passed independent review. A fresh 10-action x 3-seed x 120-tick matrix matched Python/serial-native/resident-CPU exactly. Forced termination and forced truncation also matched the terminal observation, same-step reset observation, and a healthy subsequent step. Compressed receipt is preserved under `baselines/native-fullgame-mvp/fixed-action-trace/local-final/`; uncompressed SHA-256 `28823021285bf27ae7bfd1b159ad2609a349ddb1d23cb2ab645a12641f38d107`. The receipt truthfully reports `promotion_status=blocked` and `three_way_exact=false` because CUDA did not run | Run all 10 actions x 3 seeds plus forced terminal/truncation probes on resident CUDA using a non-default stream; require exact traces and RNG status |
| Resident native full-game is ready to learn combat from its current reward | full-game native and current Puffer smoke | confirmed false by design | `native_game_lane_vec_env.py` explicitly labels the adapter raw/infrastructure-only; `game_episode.h` mirrors the original game's temporary center-movement delta reward and does not add damage, hit, knockout, win, aggression, or clean-win terms | After three-way environment parity, implement a separately tested resident GPU reward/curriculum layer and prove every component's sign, reset boundary, frequency, scale, and learning direction before long training |
| A native combat reward wrapper can reproduce the intended corrected reward signs | full-game native | fixed-local and independently approved in `392f778`; lane/CUDA integration pending | separately tested wrapper uses symmetric closing-distance approach for both shared-policy slots, positive-only net damage, positive raw event facts with signed coefficients, corrected clean-win semantics, legacy phase-1 KO normalization, validate-before-mutate behavior, and real native KO-then-terminal ordering; 9 focused tests pass with one CUDA probe skipped | Wire the wrapper after raw game step and before autoreset in resident CPU/CUDA lanes; compare component-by-component against an independent oracle, prove reset/no-leak behavior and measure cost on GPU |
| Puffer throughput is directly comparable to AI_2 throughput | Puffer vs AI_2 | not yet proven | the fail-closed receipt contract is independently approved at `8666d80` and 65 local contract/adapter tests pass; no same-semantics Puffer producer receipt has passed it | Same semantics hash, model/action/rollout/algorithm, included stages, RTX 4090, fixed-transition quality |
| Puffer can represent AI_2's ten buttons as ten binary categorical heads | Puffer 5 learner | confirmed-code, integration untested | Puffer uses `ACT_SIZES`; all sizes greater than one select its MultiDiscrete PPO path, so `{2,2,2,2,2,2,2,2,2,2}` is a natural Bernoulli-equivalent layout | Unit parity for sampled actions/log-probs/entropy, then the same contact A/B and full-game trace gates |
| Puffer GPU backend can directly run the required frozen-opponent league | Puffer 5 learner | currently blocked by stock capability | current initialization asserts that GPU envs do not enable Puffer's multi-policy/self-play mode; shared-policy two-player collection may still fit, but frozen checkpoint opponents require an explicit design | Prove shared-policy two-player semantics first; separately implement/test frozen-policy or environment-internal opponent execution before league claims |
| Stock Puffer's rollout/GAE is equivalent to TensorPPO/SB3 | Puffer vs full-game native | confirmed false in code | Puffer stores the post-prior-step reward/done at slot `t`, computes GAE from `R[t+1],D[t+1]`, and leaves the final horizon advantage zero; TensorPPO stores same-step rewards and bootstraps the final transition from post-rollout values/dones | Frozen-rollout return/advantage parity including the last transition after adding sentinel-row or SB3-style GAE inputs |
| Stock Puffer handles AI_2 time-limit truncation correctly | Puffer vs full-game native | confirmed false in code | Puffer exposes one terminal channel and no terminal observation; TensorPPO distinguishes terminated/truncated and adds discounted terminal value for timeouts | Separate timeout mask plus terminal stacked observation, with true-terminal versus truncation return tests |
| `--float` plus 10 continuous heads makes Puffer's learner otherwise equivalent | Puffer vs TensorPPO | confirmed false in code | FP32 and Gaussian Box10 are representable, but stock Puffer uses homogeneous MinGRU, clipped rewards, shifted GAE, fused PPO, and Muon; AI_2 uses `[512,512,256]` feed-forward actor/value networks and SB3 PPO/Adam semantics | Frozen-rollout forward/loss/gradient/post-update tensor comparison after a dedicated AI_2 compatibility mode |
| The locally downloaded Puffer headline rate proves AI_2 will train about 20x faster | Puffer vs AI_2 | unsupported | the compared workload/model/update/counter semantics are not yet contract-matched, and the stock Puffer learner differences above materially change work and learning | Contract-valid same-RTX-4090 end-to-end receipts plus fixed-transition and fixed-wall-clock quality gates |
| AI_2 can be launched as a stock Puffer GPU smoke environment | Puffer AI_2 ocean env | fixed-local and independently approved in `f0d749d`; CUDA unverified | `ocean/ai2/ai2.cu` wraps the pinned native full-game lane runtime, uses adapter kernels for action clamp and stacked observations, preserves terminated/truncated internally while stock Puffer still collapses them, sizes episode-record storage from games times rollout horizon, and drains records/errors/valid-steps once after each synchronized rollout. Seven wrapper-contract tests pass locally (one expected `nvcc` skip), build-flag guards fail closed, and CUDA graphs are disabled by default | RTX 4090 compile, reset/step smoke, explicit CUDA-graph capture check, and trace/receipt gates before speed or learning claims |

## Controlled experiment order

1. Repair and test classic reward plumbing, but do not use classic speed to choose
   the production trainer.
2. Finish all-action Python/native-CPU/native-CUDA trace parity. No native
   curriculum training is promoted before `three_way_exact=true`.
3. Establish random, scripted, frozen-checkpoint, and walk-off baselines using a
   fixed evaluation bank.
4. Run the contact action-head A/B with matched initialization, seeds, transitions,
   PPO settings, and evaluation modes.
5. Diagnose KL behavior with per-minibatch evidence; keep the safeguard until the
   learning comparison says otherwise.
6. Port the winning action and reward contract into the resident full-game lane.
7. Compare current opponent mix against deeper difficulty-weighted self-play.
8. Only then scale model size and transitions. Select on held-out play quality,
   exploitability, retention, and human-play video—not training reward alone.

## Final-agent gate

A candidate is not called "very good" until it passes all of these:

- deterministic and sampled behavior contracts are both reported;
- strong win/damage/control results across frozen scripted styles and historical
  checkpoints, with confidence intervals over multiple seeds;
- low self-destruct/walk-off rate and retained navigation/recovery;
- a checkpoint round robin shows no obvious cycling;
- a fresh short-budget exploiter cannot cheaply discover a dominant counter;
- representative videos cover spawn positions, moving ledges, recovery, weapon
  interactions, close/far combat, and adverse states;
- the checkpoint, config, source hashes, W&B run, evaluation JSON, and videos are
  downloaded locally before the GPU instance is released.
