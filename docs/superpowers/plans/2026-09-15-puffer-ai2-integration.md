# Puffer AI_2 Native-GPU Integration Plan

> **Execution rule:** Implement task-by-task in isolated branches. Every speed
> result must pass the canonical benchmark contract and every learning change
> must have a matched control. A fast run with different work is not a speedup.

**Goal:** Integrate the exact AI_2 native full-game simulator with Puffer's GPU
trainer, measure a trustworthy same-workload throughput result, then use the
faster system to test action representation and training-quality hypotheses
without silently changing game or PPO semantics.

**Architecture:** One Puffer environment contains `N` complete two-player AI_2
games. Each game directly embeds the existing `AI2GLane`; the two player slots
map to adjacent Puffer agents. The first implementation keeps the current
shared-policy, FP32, diagonal-Gaussian `Box(10)` behavior. Four-frame history is
maintained on device with the same reset/autoreset contract as
`TensorFrameStack`. Puffer must emit the existing fail-closed benchmark receipt.
Discrete actions and opponent leagues are separate experiments, never folded
into the baseline speed claim.

**Pinned inputs:**

- AI_2 branch: `codex/puffer-native-contract-20260914`
- Puffer reference: commit `6ffa5b1`
- AI_2 game: one `AI2GLane` per game, two players, raw obs `(2, 64)`, action
  `(2, 10)`, reward `(2,)`
- Policy input: four frames, `(256,)` per player
- Baseline action contract: sampled Gaussian float, clamp to `[0, 1]`, key down
  iff value `> 0.5`
- Headline unit: **agent transitions**; one two-player game tick is two agent
  transitions

---

## Task 0: Close prerequisite correctness gates

**Files:**

- `fixed_action_trace_parity.py`
- `native/game_lanes.h`
- `native/game_lanes.c`
- `native/game_lanes.cu`
- `tests/test_fixed_action_trace_parity.py`
- native-lane parity tests

- [ ] Make the fixed-action harness fail closed when any requested backend or
  comparison is missing. `status=completed` is forbidden unless the requested
  trace matrix actually ran and matched.
- [ ] Implement resident CPU and CUDA replay step/status ABIs. Replay state must
  be per game, consume the exact recorded RNG tape, and expose overrun/underrun.
- [ ] Compare Python, serial native, resident CPU, and resident CUDA for all ten
  single-button actions, at least three seeds, at least 120 ticks, reset,
  terminal, truncation, and autoreset.
- [ ] Compare every observation element, float32 reward, termination,
  truncation, win mask, episode return/length, terminal observation, RNG draw
  count, and lane error. No aggregate-only comparison.
- [ ] Run CUDA replay on a non-default PyTorch stream and prove tape lifetime is
  valid until the stream finishes.
- [ ] Prove ordinary non-replay collection remains byte-identical after adding
  replay support.

**Gate:** no Puffer learning or throughput claim until the downloaded receipt
reports every requested backend present and `three_way_exact=true` (or an
equivalent stronger four-way result).

## Task 1: Freeze semantic and benchmark identities

**Files:**

- `training_benchmark_contract.py`
- `benchmark_native_combat_training.py`
- `research/puffer-benchmark-contract/*`
- new manifest generator and tests

- [ ] Record exact AI_2 and Puffer commits, compiler/CUDA/PyTorch versions,
  source-file manifest, game-semantics SHA-256, observation/action schemas,
  policy architecture/parameter count, numerical precision, PPO geometry,
  terminal/bootstrap behavior, RNG contract, and timing stages.
- [ ] Generate a non-placeholder semantics hash from canonical source and config
  bytes; never accept `null`, `TODO`, repeated-character digests, or user-entered
  labels.
- [ ] Validate every receipt before exclusive atomic publication. Refuse to
  overwrite an existing artifact.
- [ ] Add a negative fixture for each mismatch: counter unit, hardware, omitted
  PPO update, model size, action distribution, frame stack, rollout geometry,
  optimizer, terminal semantics, actor staleness, and missing parity evidence.

**Gate:** contract tests pass and an independent reviewer approves the receipt
producer. Commit `8666d80` is the reviewed starting point.

## Task 2: Build the minimal Puffer Ocean AI_2 environment

**Ownership boundary:** new Puffer `ocean/ai2` files and the thinnest build glue.
Do not copy or reinterpret physics/reward logic.

- [ ] Write compile-time contract tests for two agents per game, raw obs `64`,
  stacked obs `256`, ten actions, and an even `total_agents` requirement.
- [ ] Embed the exact `AI2GLane`/`ai2g_lane_step` implementation in the Puffer
  GPU environment. Map game `g` to agent slots `2*g` and `2*g+1`.
- [ ] Bind the Puffer CUDA stream and keep observations, actions, rewards,
  termination metadata, RNG, frame history, and episode accounting resident on
  device.
- [ ] Implement reset and step without Python callbacks or per-step host copies.
  Episode summaries may be copied once per rollout.
- [ ] Propagate lane errors and non-finite observations as hard failures with
  game/agent indices; never replace them with zeros.
- [ ] Add sanitizer-friendly CPU/header tests for buffer sizes, indices, odd
  agent counts, reset idempotence, and bounds.

**Gate:** the environment compiles against pinned Puffer and runs deterministic
CPU/header tests before CUDA benchmarking.

## Task 3: Preserve frame-stack and episode semantics

- [ ] Add tests that initially fail unless reset produces three zero frames plus
  the initial raw frame, matching `TensorFrameStack` exactly.
- [ ] On every ordinary step, shift the prior three frames and append the new
  raw observation on device.
- [ ] On termination, preserve the final stacked observation for rollout/value
  logic before autoreset; after autoreset, clear history and append only the
  new initial observation.
- [ ] Carry `terminated` and `truncated` separately through the environment/
  rollout boundary. If stock Puffer lacks this ABI, extend it rather than
  conflating time limits with deaths.
- [ ] Match AI_2 timeout bootstrapping: truncation adds the discounted terminal
  value computed from the terminal stacked observation; true termination does
  not bootstrap.
- [ ] Add targeted tests where identical trajectories end once by stock loss and
  once by time limit; expected returns must match TensorPPO bit-for-bit within
  the declared floating-point tolerance.

**Gate:** no quality training while time-limit bootstrap is missing. A short
speed-only measurement may exclude truncations, but its receipt must say so and
is not promotion-eligible.

## Task 4: Match the existing Gaussian policy/update baseline

- [ ] Express the exact AI_2 actor and critic layer geometry `[512, 512, 256]`,
  activations, initialization, log-standard-deviation parameterization, FP32
  precision, optimizer, schedules, advantage normalization, clipping, entropy,
  value loss, GAE, and gradient clipping in Puffer.
- [ ] Load a canonical initial state into both TensorPPO and Puffer. Export and
  compare pre-update actions, values, log probabilities, entropy, advantages,
  returns, losses, gradients, and post-update parameters on a frozen rollout.
- [ ] Test action post-processing at values below, equal to, and above `0.5`,
  plus out-of-range values requiring clamp.
- [ ] Prove one full PPO update consumes the same samples the same number of
  times. Log actual optimizer-step count, not configured epochs alone.
- [ ] Keep CUDA graphs, fused kernels, alternate optimizers, TF32, reduced
  precision, and asynchronous actors disabled for this equivalence gate.

**Gate:** matched frozen-rollout update passes declared per-tensor tolerances,
and any unavoidable difference is a named learning blocker in the receipt.

## Task 5: Add a canonical Puffer benchmark producer

- [ ] Emit raw `agent_transitions` and elapsed wall time; derive rate in the
  validator. Never publish a bare dashboard `SPS` value.
- [ ] Include policy inference, action sampling, environment step, rollout
  storage, returns/advantages, PPO update, and synchronization in the end-to-end
  timed scope.
- [ ] Warm compilation/caches outside the measured samples, then use at least
  three timed repetitions and report each raw repetition plus median and spread.
- [ ] Capture CUDA event/Nsight evidence for any overlap claim. A config flag is
  not proof of overlap.
- [ ] Emit source manifests and correctness evidence after timing, then validate
  and atomically publish the receipt.
- [ ] Add fault-injection tests proving omitted stages, fake hashes, partial
  transitions, failed parity, non-finite loss, unchanged parameters, and
  overwritten paths all fail closed.

**Gate:** `training_benchmark_contract.py validate` succeeds on the Puffer
receipt, and `compare` says `throughput_comparable=true` against the matched
AI_2 native receipt before a speedup ratio is reported.

## Task 6: Same-GPU RTX 4090 benchmark matrix

**Lifecycle rule:** inventory before creation; use one owned instance; never
touch another agent's instance; do not replace an instance in loading/stopped/
unknown state; download and hash-verify all artifacts before destroying it;
inventory again at closeout.

- [ ] Run both stacks on the same RTX 4090 image and pinned clocks/software.
- [ ] Sweep game counts while keeping rollout transitions and PPO sample reuse
  explicit: 256, 1,024, 4,096, and the largest memory-safe point.
- [ ] For each point, capture end-to-end transitions/s, collection/update split,
  GPU utilization, memory, SM occupancy, kernel launch count, launch gaps,
  policy matmul shapes, environment-kernel time, and synchronization time.
- [ ] Separate throughput improvements from changed batch/rollout geometry.
- [ ] Re-run the winning point after a clean process start to catch cache/warmup
  artifacts.
- [ ] Download receipts, raw JSON/CSV, profiler traces, stdout/stderr, configs,
  source manifests, W&B identifiers, and checksums locally before teardown.

**Gate:** a speedup is real only if the contract says comparable, parity remains
passed, parameter changes/losses are finite, and repeat variance is acceptable.

## Task 7: Diagnose PPO KL/update efficiency without weakening learning

- [ ] Reproduce the Sept-9 contact checkpoint where 12/12 rollouts stopped after
  one of 16 nominal optimizer steps.
- [ ] Log per-minibatch pre-step approximate KL, threshold, whether the step ran,
  clip fraction, entropy, policy/value losses, gradient norm, learning rate,
  log-prob/action statistics, and sample indices.
- [ ] First run a shadow arm that records the threshold decision but does not
  change behavior; prove diagnostics-off is byte-identical to current training.
- [ ] Test batch size 8,192 vs 16,384, shuffled order, learning rate, and
  target-KL threshold one variable at a time over matched seeds.
- [ ] Never remove the KL safeguard merely to increase optimizer-step count.
  Select by held-out learning per transition and per wall-clock second.

**Gate:** explain the one-step stop from measured minibatch traces and show the
chosen configuration improves held-out quality without instability.

## Task 8: Controlled action-representation experiment

Run three arms from matched trunk/value initialization:

1. current diagonal-Gaussian `Box(10)` plus threshold;
2. ten independent binary categorical heads (`MultiBinary` semantics);
3. a factored `MultiDiscrete` movement/attack representation with an explicit
   reversible mapping to legal game controls.

- [ ] Unit-test sampled action, deterministic action, log-probability, entropy,
  masking, and PPO ratio calculations for every head.
- [ ] Keep seeds, transitions, model capacity, PPO settings, reward, opponent,
  observation stack, and evaluator fixed.
- [ ] Evaluate sampled and deterministic modes separately on a fixed contact
  bank and original-game scenarios.
- [ ] Track attack initiation, attempted attacks, landed hits, time-to-first-hit,
  damage, distance/orientation at attack, movement, recovery, walk-offs, returns,
  and episode length.
- [ ] Require at least three seeds and confidence intervals. Save videos for
  representative successes and failures, not only the best episode.

**Promotion gate:** deterministic initiation at least 80%, deterministic hit
rate no worse than the sampled Gaussian control, and navigation/survival within
control seed spread. This experiment is a learning change, not part of the
same-semantics speed ratio.

## Task 9: Fix and verify reward/curriculum wiring

- [ ] Promote the independently reviewed classic reward fix: actual raw signal
  subscriptions, correct learner win sign, positive phase-1 KO magnitude with
  direction applied in code, real `AttackState`, per-environment manager, and
  complete episode reset.
- [ ] Test real `WarehouseBrawl` wrapper ordering and a two-process
  `SubprocVecEnv`: one subscription per signal, no duplicate on reset, no
  cross-environment reward leakage.
- [ ] Add reward-component frequency/magnitude telemetry so configured-but-dead
  terms fail an automated smoke test.
- [ ] Run fixed-versus-broken same-seed short phase-1/phase-2 direction checks;
  do not infer learning benefit from unit tests alone.
- [ ] Port only verified semantic fixes into the native/Puffer reward path, with
  parity tests. Keep classic and native evidence explicitly separated.

## Task 10: Build the opponent system required for a strong agent

Stock Puffer GPU currently rejects multiple policies/self-play. Shared-policy
self-play is useful for the MVP but cannot create a stable historical league.

- [ ] Establish fixed evaluator baselines: random, scripted styles, walk-off,
  frozen navigation/contact checkpoints, and held-out historical policies.
- [ ] Implement resident frozen-opponent inference or extend Puffer GPU policy
  assignment so each two-player game can map learner and opponent to different
  policy weights without host round-trips.
- [ ] Publish immutable checkpoint records with step, source/config hashes,
  policy architecture, action contract, rating, and checksum.
- [ ] Implement a deeper pool plus prioritized fictitious self-play: sample
  opponents near the learner's competence boundary while retaining older
  strategies and scripted styles.
- [ ] Record configured opponent key/probability and realized policy class,
  snapshot hash/step, pool size, fallback reason, game, and reset. Fail if
  realized mix differs by more than three percentage points over a sufficient
  window.
- [ ] Add checkpoint round robins and short-budget exploiters to detect cycling
  and cheap dominant counters.

**Gate:** no “human-level” claim from shared-policy mirror matches alone.

## Task 11: Scale model and curriculum only after the gates

- [ ] Compare model sizes with matched transition budgets and wall-clock budgets.
  Capacity is selected by held-out performance, not training reward.
- [ ] Use mastery-gated phases with retention probes: movement/recovery, contact,
  attack timing, moving opponents, weapons/ledge states, historical self-play,
  exploiters, and mixed full game.
- [ ] Fade shaping rather than switching it off abruptly; keep a rehearsal mix
  from earlier phases to detect forgetting.
- [ ] Stop a phase when multiple held-out metrics plateau across evaluation
  intervals, not at an arbitrary transition count.
- [ ] Preserve every promoted checkpoint with config/source/receipt/evaluation
  provenance and local copies.

## Final promotion criteria

The final agent must satisfy all of the following before it is called “very
good”:

- deterministic and sampled evaluation are both reported;
- low walk-off/self-destruction rate and retained navigation/recovery;
- strong win, damage, contact, and stage-control metrics across held-out spawn
  positions, moving ledges, weapons, scripted styles, and historical policies;
- confidence intervals over multiple training seeds;
- no obvious cycling in checkpoint round robin;
- no cheap dominant counter from a fresh short-budget exploiter;
- representative failure as well as success videos reviewed in the original
  Python game;
- checkpoint, W&B export, config, source/semantics hashes, raw evaluation JSON,
  benchmark receipts, profiler traces, and videos downloaded and checksum-
  verified locally before releasing the owned GPU instance.

This plan intentionally separates three claims: **the simulator is correct**,
**training is faster**, and **the learned policy is better**. None implies the
next without its own evidence gate.
