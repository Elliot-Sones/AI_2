# Native contact curriculum batch comparison

> **For agentic workers:** Use superpowers:subagent-driven-development, test-driven-development, and verification-before-completion. This is the second stage of the user-approved combat update experiment, not a replacement curriculum.

**Goal:** Measure minibatch effects on the already-qualified `bridge_b1_contact-v1` lesson using CUDA simulation and the real calibrated large curriculum policy.

**Architecture:** Add a C/CUDA-compatible contact task around the unchanged raw `AI2GState` game. A separate lane runtime exposes one learner per game with device-resident autoresets, exact task rewards and diagnostic records. Preserve the real masked-action curriculum policy, independent critics, protected optimizer and KL early-stop rules when adapting the collector.

**Tech stack:** Existing C99/CUDA, ctypes, PyTorch, SB3 2.5.0 and immutable Python curriculum reference. No new dependency or gameplay rule.

## Frozen contract

Reference: `/private/tmp/ai2-curriculum-20260907.Lyif6J/combat_curriculum.py`, `B1ContactEnv`. Freeze an unchanged source copy with SHA256; leave that worktree untouched.

- Reset the raw scene and replace learner then opponent with fresh players, preserving exact shape/body removal/addition/reindex order. Gap and midpoint are `np.random.default_rng(seed)` draws U[3.1,3.4] and floor-center+U[-.15,.15]. Both bodies require clearance >=.2. Setup precomputes a finite scenario schedule; exhaustion fails closed rather than changing the distribution or repeating seeds silently.
- Mask controls to 0,1,2,3,4,6,7 after clipping raw action to [0,1]. Opponent inputs zero; its ordinary physics and knockback remain active.
- One actual attributed positive light-damage attack yields +1 once. On resolution of both players' attack/stun/stored stun, award +2 for approach-at-first-hit >=.25, both alive/grounded/fully supported, and learner clearance >=.2.
- Cost -.001/tick. Own stock loss first priority and -5; opponent stock loss fails without additional stock penalty. At300 ticks timeout costs -1. All task endings are terminal, never time-limit bootstrap truncations. No teleports after reset.
- Contact observation64 and four-frame stack256 retain historical checkpoint feature order/bounds. Terminal frame is preserved before reset. One learner transition per game tick, unlike the two-player raw benchmark.
- Every episode record includes scenario/slot/direction, outcome, return components, first-hit time, approach, attack requests/starts/contacts/damage, both players' stocks/support/clearance and deadline. Bounded validation traces retain XY/actions/events/termination for diagnosing errors.

## Starting policy and experimental scope

Use the canonical calibrated checkpoint `AI_2/baselines/combat-curriculum-session13/fresh-run/approved-resume/preserved/calibration/final.zip`, 102673333bytes, SHA256 `301ce3c72924f3e5c62f265f88292b8952e86e6d71d9a0552213cb8f3c6784b6`. It descends from the selected [1536,1536,1536] 10,027,008-step navigation parent and adds32,768 calibration transitions. Reject the incomplete `native-copy` duplicate. Load corresponding curriculum policy/critic/optimizer semantics; no fresh-policy substitution.

The parent hash is `0e2f64e02cd0f7d300554f7a6cb641065c75771f69f55c0885258dda32afc6cd`. Keep E1 entropy .001, lr .0001, clip .1, targetKL .01, gamma .99, lambda .95, four epochs. End calibration for the joint contact learning test while preserving protected inactive rows and separate navigation critic. If adapting to native TensorPPO cannot preserve these contracts, block learning rather than running ordinary raw-reward PPO under a curriculum label.

Compare minibatches8192 and16384 with the same environment count,512 rollout ticks, checkpoint and evaluation seeds; predeclare any necessary environment-count scaling. Report actual KL early stops and optimizer steps, since nominal epochs alone do not establish equal update work. Use short paired multi-seed blocks within provider budget, paired nearby-retention evaluations, and original-game contact evaluation. No B2 advancement or claim of mastery without existing gates.

## Ownership and steps

### Task A — shared native contact task and exact reference comparison

Owner: contact task implementer. Files `native/contact_episode.h`, `native/contact_episode.c`, `native_contact_episode.py`, `tests/test_native_contact_episode.py`.

- [ ] Write failing reset/step/reward/outcome tests against real B1ContactEnv, including both slots/directions and idle/attack-only/walk-off/unsafe-hit controls.
- [ ] Implement fresh-player reset with preserved body/shape ordering, native event attribution, task rewards and explicit outcome metrics using unchanged ai2g_step.
- [ ] Exact replay comparison consumes Python recorded game RNG and setup scenarios, never copied mid-trajectory state. Compare observations, float32 training rewards, term/trunc, underlying game state and diagnostic projection each tick.
- [ ] Run full existing raw-game regression subset and contact positives/negatives. State finite coverage honestly.

Interface: `AI2B1Scenario {uint64_t seed; int32_t player_slot,direction; double gap,midpoint;}`, `AI2B1State` includes `AI2GState game` plus task bookkeeping; `AI2B1StepResult` includes learner `double obs[64],reward`, int32 terminated/truncated/error and task snapshot/metrics. Shared HD functions `ai2b1_reset(state,cfg,rng,scale,scenario,out)` and `ai2b1_step(state,learner_action,rng,table,out)`; reconcile exact names with lane implementer before writing code.

### Task B — separate contact lanes and collector adapter

Owner: lane implementer. Files `native/contact_lanes.h`, `.c`, `.cu`, `native_contact_lane_vec_env.py`, `tests/test_native_contact_lanes.py`.

- [ ] Test lane autoreset, terminal frame, one-learner accounting, independent RNG/scenario schedule, source errors, record overflow and finite schedule exhaustion first.
- [ ] Reuse transport conventions, not raw game_lanes ABI changes. Kernels run on the caller Torch stream; no host synchronization/action/observation transfer per tick. Setup scenario arrays and diagnostic drains may cross host boundary.
- [ ] CUDA and CPU reference replay equal with identical scenarios/RNG. Verify non-default stream, lifecycle/error handling, true termination and zero-padded stack.
- [ ] Expose terminal outcomes/metrics and optional bounded traces; never invent hit credit from proximity.

### Task C — faithful learner adapter and bounded experiment

Owner: coordinator/assigned learner implementer. Reuse unchanged curriculum policy source in this worktree, add a small native collector adapter and a dedicated batch comparison runner/test.

- [ ] Test CUDA-only checkpoint load/hash gate, raw sampled action storage, active-control log probabilities, independent critic/protected rows, task at rollout boundaries and exact one-update parity against original curriculum learner from the same buffer/RNG.
- [ ] Require source-bound contact parity and checkpoint preservation receipts before training. The runner must reject output reuse, missing CUDA, bad checkpoint, unqualified stage, or expired finite deadline.
- [ ] Run paired same-checkpoint seeded learning blocks only after GPU parity. Download/checkpoint after every block, then evaluate original Python game with CUDA policy and full diagnostic traces.
- [ ] Contact gates unchanged: two development panels >=18/20 successes and <=1 own fall in each action mode, followed by fresh100/mode >=90successes, <=5falls, >=40/50 per slot; require paired nearby retention. Report bounded/inconclusive result if not reached.

## Review and closeout

- [ ] Independent spec then quality review; fix material issues and rerun.
- [ ] Separate throughput, time-to-behavior, finite correctness and hardware/version limits. Preserve local artifacts/hash manifests and failed attempts.
- [ ] Coordinate one shared GPU window with raw update qualification, obey cumulative $12 and stop-only deadline. No W&B upload/credential transfer or instance destruction.
