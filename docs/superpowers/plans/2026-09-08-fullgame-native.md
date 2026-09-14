# Full-game native simulation implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development, with independent file ownership and separate specification and quality review. The user approved implementation and rented-GPU testing; no further generic permission handoff is required.

**Goal:** Run the original two-player combat gameplay entirely in native C/CUDA and measure matched end-to-end combat training on a rented GPU, without confusing navigation throughput with combat throughput.

**Architecture:** Keep the existing navigation ABI and collector intact. Compose a raw full-game state from the proven contact world and input/movement bank, additional combat fields, and an ordered weapon system. Shared host/device functions execute player 0 before player 1 inside each independent lane. Rendering remains in the Python game for evaluation.

**Tech stack:** Existing C99/CUDA, ctypes, PyTorch/SB3 PPO, unittest, frozen Python/Pymunk reference; no new dependency.

## Gate snapshot — 2026-09-09 verified CUDA and benchmark update

This supersedes the 02:27 UTC host-only checkpoint and includes the **03:20:30 UTC cloud closeout**. The raw full-game core now has real CUDA parity, sanitizer, TensorPPO training/benchmark, original-Python-game checkpoint-reload, and short-horizon collector evidence. The **declared finite closed-loop qualification matrix is complete**: four exact host corpora total **1,487 cases / 54,051 ticks**, covering **264/264** actual move/slot/facing/hit-or-whiff outcomes, **10/10** mechanic labels, and **2,048/2,048** binary slot/input combinations. This is not an exhaustive state-space proof. The current combat curriculum wrapper is **not ported**. These are **raw two-player full-game TensorPPO infrastructure results**, not a completed curriculum experiment or evidence of strong or human-level play. Final artifacts are preserved and verified locally, both instances are stopped without destruction, and zero instances are running. Only external W&B publication is blocked on separate payload approval; the offline bundle is preserved and this is not a GPU-work blocker.

| Gate | Confirmed evidence | Status |
|---|---|---|
| Source-exact host components | Raw scene, capsule oracle, attack constructor tables, player state/move fixtures, and ordered weapon lifecycle checks | Host PASS |
| Player component matrix | 264 force-start fixtures: 33 moves × 2 slots × 2 facings × actual hit/whiff assertions | Component PASS; not the closed-loop coverage matrix |
| Raw game and tensor binding | 11 raw-game tests; 8 binding tests | Host PASS |
| Resident lanes on the real GPU host | 38 tests: 37 pass, one explicit two-GPU skip; final lane run 50.592s | CUDA/host PASS for executed tests |
| Snapshot/acceptance boundary | 13 original-capture and negative-control tests; independent specification and quality review | Host PASS |
| Closed-loop comprehensive parity | 1,301 fresh-construction cases, 35,691 actual ticks exact; 209/264 outcomes and all 2,048 binary slot/input cases | Exact for recorded cases; preserved intermediate coverage |
| Targeted parity extension | Six additional cases, 360 ticks exact; initial combined outcome coverage 212/264, including observed invulnerable overlap | Exact for recorded cases |
| Finite coverage completion | Additional 156 cases / 15,600 ticks and 24 cases / 2,400 ticks; four-run union 264/264 outcomes, 10/10 mechanics, 2,048/2,048 inputs | Complete finite suite; not exhaustive state-space proof |
| CUDA sanitizer | `compute-sanitizer` smoke completed with zero reported errors | PASS for recorded smoke |
| Combined Linux suite | 129 tests: 127 pass, two explicit skips, 114.343s | PASS; coordinator-confirmed final output |
| Combined local suite | 129 tests: 109 pass, 20 CUDA skips, 63.838s | Host PASS; skips are not GPU passes |
| Previously recorded harness scope checks | Benchmark / validator / profiler-related suites: 10 / 9 / 24 tests pass within their declared scopes | PASS; not a substitute for real runtime receipts |
| Final focused suites on GPU host | Benchmark 10; attack tables 11; actual native collector 3; profiler 7; validator 9 | PASS; coordinator-verified final remote logs |
| Raw-game PPO and throughput | Matched 128-game runs, 512/2,048-game scale runs, separate optimizer-batch experiment | PASS; rates and comparability limits below |
| Original-game checkpoint reload | CUDA-loaded policy hash matches saved benchmark policy; seeds 3/7/101, 768 game ticks / 1,536 player transitions | Infrastructure PASS; no damage observed |
| Additional short-horizon collector | Three-tick horizon, seven-step real CUDA collection, true bootstrap and negative control | PASS; three tests in 3.977s |
| Artifact preservation and cloud closeout | 121 remote files / 302,066,380 bytes verified in both local archives; both instances stopped, zero running | Verified at 03:20:30 UTC; none destroyed |
| W&B publication | Offline bundle preserved; specific external-payload publication was rejected for missing authority | BLOCKED pending user approval; no effect on completed GPU gates |

Current receipts: [finite coverage union](../../../.qualification/coverage-completion-union-v1.json), [comprehensive summary](../../../.qualification/comprehensive-v1/summary.json), [targeted summary](../../../.qualification/targeted-v1/summary.json), [completion inventory](../../../.qualification/coverage-completion-v1/receipt.json), [final completion inventory](../../../.qualification/coverage-completion-final-v1/receipt.json), [final GPU lane log](../../../baselines/fullgame-20260908/downloads/lane-tests-final.log), [final Linux combined log](../../../baselines/fullgame-20260908/downloads/final-native-suite.log), [actual collector log](../../../baselines/fullgame-20260908/downloads/final-test_native_game_collector.py.log), [sanitizer log](../../../baselines/fullgame-20260908/downloads/cuda-memcheck.log), [benchmark aggregation](../../../baselines/fullgame-20260908/summary.json), [original-game reload summary](../../../baselines/fullgame-20260908/downloads/checkpoint-original-v1/summary.json), and [profiler receipt](../../../baselines/fullgame-20260908/downloads/profile-128-v1/profile.json). The final combined Linux/local counts and focused GPU-host logs were independently confirmed by the coordinator. Final focused timings: benchmark 10 in 8.489s; attack tables 11 in 1.966s; actual native collector 3 in 3.977s; profiler 7 in 0.079s; validator 9 in 2.451s. The coordinator confirms that the gameplay core remained unchanged through these GPU tests.

The original comprehensive receipt remains `exact=true` and `full_combat_coverage=false`: its 209 observed outcomes are not retroactively changed to 264. Targeted brought the intermediate union to 212/264; the two completion runs then closed the remaining outcomes. The verified [four-run union](../../../.qualification/coverage-completion-union-v1.json), SHA256 `a221bd7a0095d34d1ba575ad6593f4c9d4a7908f128971fc135be5fca2441eef`, records `exact_for_recorded_cases=true` and `declared_finite_suite_coverage_complete=true`, with no missing move cases or mechanic labels. The qualification owner rehashed all 4,473 receipt-listed files and verified identical non-qualifier source hashes and native library across the four corpora. Actual observed hit/whiff witnesses, not fixture intent, establish coverage. Each independent case constructs fresh original/native environments; neither every possible state nor repeated-reset equivalence across reused parity adapters is claimed.

Historical evidence remains preserved: [integration-v2 summary](../../../.qualification/integration-v2/summary.json), [inventory receipt](../../../.qualification/integration-v2/receipt.json), and [receipt hash](../../../.qualification/integration-v2/receipt.sha256.json). That receipt contains 39 artifacts and has SHA256 `a86fa8ae04cf0e9f1dc0760d7fdcc8d6a2d02c51d0c236effa4a029d44e780cf`. Its 12 cases requested 2,986 ticks and executed 2,985 because the last-stock case ended early; coverage then was 40/264 outcomes and 539/2,048 inputs. [Integration-v1](../../../.qualification/integration-v1/summary.json) stopped fail-closed after 1,793 exact ticks when native source changed. The failed [first GPU lane attempt](../../../baselines/fullgame-20260908/downloads/lane-tests-v1.log) (compiler errors, zero tests) and [first Linux original-parity attempt](../../../baselines/fullgame-20260908/downloads/original-parity-v1.log) (unresolved local source path, zero tests) are retained alongside the successful reruns, not rewritten as passes.

Historical host timings confirmed by coordinator tool output, not separate JSON receipts: raw scene 1 test in 1.129s; post-extraction legacy navigation 6 tests in 40.675s; raw game 11 tests in 3.137s. Snapshot 13 tests passed locally in 7.086s and on Linux in [15.260s](../../../baselines/fullgame-20260908/downloads/original-parity-v2.log). Local sandbox attempts that stopped with OpenMP error 179 remain failed attempts; matching outside-sandbox controls established the sandbox shared-memory restriction, not a gameplay mismatch. No sandbox failure or CUDA skip is counted as a passing executed test.

## Source and operating boundaries

- Isolated branch: `codex/fullgame-native-20260908`; worktree `/private/tmp/ai2-fullgame-20260908.LtE6Ou`.
- Inherited source snapshot: `AI_2-native-ppo-optimizations-20260907`, base c154f1a; its uncommitted root Python/native/tests copied without changing that checkout.
- Correctness oracle: `AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected/UTMIST-AI2-main/environment/environment.py`, SHA256 `c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230`.
- Native baseline: six existing episode tests passed in 105.581 seconds. This proves navigation only.
- GPU authorization: one running instance maximum, cumulative campaign ceiling $12. Historical audits recorded $8.770332723225753 spent before the later pre-rental audit of **$8.778417940895746**. Replacement 50337302 was rented at about $0.318/hour with finite deadline 2026-09-09T04:15:00Z. The coordinator's strict **03:20:30 UTC** final audit confirms **zero running instances**, credit **$3.1836025611042373**, and cumulative spend **$9.131434438895763**. Both 50337302 and retained 50301573 report `actual_status=exited` and `cur_state=intended_status=next_state=stopped`; neither was destroyed. The **$0.353016498** additional account debit since the later pre-rental audit is not a pure-compute invoice. The stop-only watchdog timed out while verifying after issuing stop; its failed verification is not erased, and the independent strict provider-state audit resolved the final state. Re-audit before any new cloud action; no destruction without user confirmation.
- Preserve artifacts locally after every block with verified SHA256; stop compute at the finite budget/time guard. Final [retrieval manifest](../../../baselines/fullgame-20260908/retrieval-manifest.json), SHA256 `a6e05137d65d4673d8003ad10c17bb109fff2bec9d16e9d44c91f66c1a83d1a6`, contains **121 files / 302,066,380 bytes**, independently verified by the coordinator in both this worktree and durable `/Users/elliot18/Projects/ml-research/AI_2/baselines/fullgame-20260908`. The qualification captures/union and hash-pinned frozen oracle are also preserved there. No cloud-backup requirement.
- Never silently fix frozen gameplay quirks while claiming parity. A mismatch blocks promotion, not source modification.

## Shared interfaces and ownership

| Owner | Files and interface |
|---|---|
| Leader | `native/game_episode.h/.c`, `native_game_episode.py`, `tests/test_native_game_scene.py`, `tests/test_native_game_episode.py`, shared raw-reset extraction in `native/navigation_episode.h/.c`; integrates all gates |
| Player implementation | `native/game_player_types.h`, `native/game_player.h/.c`, `tests/test_native_game_player.py`; `ai2g_player_init(scene,extra[2],slot)` and `ai2g_player_physics(scene,extra[2],slot,rng)` |
| Tables | `native_attack_tables.py`, `native/game_attack_data.h`, `tests/test_native_attack_tables.py`; exact 33 move/107 power/206 cast/207 hitbox/4 point/194 frame-change tables, with provenance recorded by the qualification runner |
| Weapons | `native/game_weapons.h/.c`, `tests/test_native_game_weapons.py`; initialize from scene, mirror pre-physics, drop, pickup, update/prune |
| Independent oracle | `native/game_capsule.h/.c`, `tests/test_native_game_capsule.py`, `combat_episode_capture.py`, `combat_episode_parity.py`, corresponding tests |
| Gameplay snapshot adapter | `native_game_snapshot.py`, `tests/test_native_game_snapshot.py`; `GameParityAdapter.reset(draws,scenario)` and `.step(actions,draws)` expose only the declared gameplay projection |
| Qualification runner | `native_game_qualification.py`, `tests/test_native_game_qualification.py`; immutable capture, comparison, provenance, and first-divergence receipts |
| Device/learner integration | `native/game_lanes.h/.c/.cu`, `native_game_lane_vec_env.py`, `benchmark_native_combat_training.py`, corresponding tests; real CUDA lanes, raw-game TensorPPO benchmarks, and short-horizon collector gate verified |
| Checkpoint validation / profiling | `validate_native_combat_checkpoint.py`, profiler driver and corresponding tests; CUDA-only policy reload/forward into original Python game, hash-bound traces, diagnostic CPU/CUDA attribution |

`AI2NEpisode *scene` provides existing world, movement config, players, platform and clock, but full-game code must never run its navigation wrapper. Extra combat and weapon types are composed into `AI2GState`; no legacy struct resizing. `AI2GCapsule {double x,y,width,height;}` and `ai2g_capsule_intersects(a,b)` preserve Python integer-rectangle collision behavior. Attack state ID is 8; stun is 5; taunt is 12.

## Task 1: establish isolated source and raw reset

- [x] Copy inherited source without touching peer/main checkouts; run existing episode suite.
- [x] Write failing `RawSceneTests.test_reset_consumes_only_raw_draws_and_leaves_both_players_at_spawn`: expect seven RNG draws, both original spawn coordinates, no target/freeze.
- [x] Extract `ai2n_reset_scene(ep,cfg,rng)` ending before the navigation wrapper; original `ai2n_reset` calls this then its unchanged wrapper.
- [x] Export raw reset, run `python -B -m unittest discover -s tests -p test_native_game_scene.py -v`, then rerun six legacy episode tests. Coordinator-confirmed outputs: 1 raw-scene test and 6 legacy tests passed; timings above.
- [x] Record inherited and foundation changes separately with Lore trailers: `23b3bff` preserves the inherited navigation snapshot and `65d5d0a` records the new collision/attack foundations.

## Task 2: exact collision and attack data

- [x] Differential-test capsules against AST-loaded frozen `CapsuleCollider`, including negative coordinates, integer rounding, degenerate dimensions, tangency and nonfinite inputs.
- [x] Generate exact table schema from the three runtime-loaded attack directories, retaining frame order, halved timings, defaults and global power references.
- [x] Test source-constructor equivalence, exact record counts, reproducible regeneration, absent/unknown data and capacity failures. Separate host/device declarations resolve in their address spaces; compiler evidence does not qualify device execution.
- [x] Run host `test_native_game_capsule.py` and `test_native_attack_tables.py` suites. Local CUDA-unavailable skips remain skips, not device passes.

## Task 3: all player states and moves

- [x] Tests first for unmasked input/move selection, attack initiation, charge/release, hit/whiff, chained casts, damage/knockback/stun, dodge vulnerability, taunt and KO/respawn.
- [x] Implement `ai2g_player_physics` against both actual bodies and persistent state banks. Preserve reset(old), exit(old), assign, enter(next) order and same-tick slot asymmetry.
- [x] Component comparison covers 11 moves × 3 weapons × both players × both facings × hit/whiff (264 force-start fixtures), with positive damage or no-damage assertions and persistent state-bank comparison.
- [x] Complete the separately observed closed-loop 264-case move/outcome matrix: the verified four-run union records all 264 actual move/slot/facing/hit-or-whiff outcomes, with witness captures and unchanged gameplay core. This is finite-suite coverage, not exhaustive state-space proof.
- [x] Test all 1,024 binary single-player action combinations for each slot: comprehensive receipt observes all 2,048 slot/input combinations. Simultaneous attacks, pending impulses, attack/cast timers and damage counters also appear in recorded host cases; this finite matrix is not exhaustive state-space coverage.

## Task 4: weapons and raw match lifecycle

- [x] Test regular pickup ages 35/36/37 and expiry350/351; held-H versus edge-H; attacking exclusions and player-order ties.
- [x] Test drop offset, same-tick creation, lock14/15, expiry249/250, pool reuse, controller suppression/pruning and first-four observation order.
- [x] Implement source-exact drop/pickup/update phases with replay RNG; include asset-derived geometry/timing in provenance. Capacity exhaustion returns an explicit error, never drops entities silently.
- [x] Raw tick: clear per-frame counters → both input handlers/process rewards → platform → player0 → player1 → weapon position mirrors → contact solve → increment clock → drops → pickups → controller update/prune → truncation → both observations.
- [x] Host tests cover strict KO bounds, 90-tick respawn, one-tick stock-exhaustion terminal delay, short truncation and all 9,000 ticks of a match.
- [x] Retain a dedicated simultaneous-terminal-and-timeout qualification case: `comprehensive-v1/terminal_timeout-012` executes two exact ticks, separately from the ordinary timeout and terminal cases.

## Task 5: independent closed-loop parity

- [x] Capture original raw game with explicit Python/NumPy seeds, ordered RNG tape, both actions, observations, rewards, events and hidden state.
- [x] Native adapter resets only once per fresh-construction case and advances solely from actions/RNG; reference-state injection during replay is forbidden.
- [x] Compare every declared `gameplay_projection_v1` field, including weapons and attack internals; no navigation field-stripping or blanket float tolerance. See the precise scope below.
- [x] Negative controls reject native observation corruption in both slots, native state-bank/world/timeout mutations, weapon/timer/event changes, missing hurtboxes, signed-zero differences, and wrong/unused RNG draws. Native `Result.obs` is serialized directly on ordinary steps; recomputing it to hide output corruption is forbidden.
- [x] Twelve initial closed-loop host cases exercise mixed inputs, contact caches, pooled bodies, combat and ordered events, with immutable captures and receipts in `integration-v2`; the later comprehensive run adds 1,301 cases / 35,691 exact ticks, and targeted adds six / 360.
- [x] Complete the recorded binary-input matrix and observe invulnerable overlap; preserve exact state/event/RNG comparison and explicit finite-corpus limitations.
- [x] Complete the remaining closed-loop move/outcome coverage: completion adds 156 cases / 15,600 exact ticks and final completion adds 24 / 2,400. The four-run union records all 264 outcomes, 10 mechanics and 2,048 binary slot/input cases across 1,487 cases / 54,051 ticks.

### Exact scope: `gameplay_projection_v1`

The accepted original-vs-native host gate compares end-of-tick dynamic gameplay/world state, both float64[64] observations and rewards, flags, complete ordered semantic events, and exact RNG requests/consumption. Separate resident CPU-vs-CUDA replay tests establish their own device-runtime gate; the 1,487-case original-game host union is not mislabeled as 1,487 GPU replay cases. Player state-bank attribute presence, attack manager/power/cast state, damage and pending impulses, hurtboxes, weapon identity, spawner/pool/object order, and all captured body/contact-cache/broadphase-tree fields remain in scope. Attack constructor metadata is reconstructed only from hash-verified frozen assets. Reset-only declared fixtures are allowed; no corrective state upload occurs between ticks.

Explicit exclusions, retained in the unmodified raw diagnostic capture:

- Player constructor constants are pinned by frozen source/reset provenance rather than compared as mutable fields.
- `hitboxes_to_draw` and `points_to_draw` are render arrays; actual attack/collision semantics and geometry remain compared.
- Hurtbox `rect`, `cap1`, and `cap2` are derived collision caches; base center/width/height/radius/is_circle and offset/size attribute presence remain compared.
- `SpawnerVFX*`/`DroppedVFX*` rendering-only object names are excluded; physical object order remains compared.
- Intermediate phase snapshots and verbose collision-attempt diagnostics remain in raw receipts, but are not the accepted end-of-tick gate. No claim of intermediate full-snapshot parity is made.

The semantic stream is not discarded: player event kinds 1–9 and weapon kinds 101–102 retain source/target, move/power/cast/hitbox, attack ID, amount, phase player, and global event sequence. Named gameplay coordinate vectors and integral state counters have explicit type canonicalization only; float values are not rounded and comparison still checks IEEE-754 signed zero. Missing/extra fields, changed event order, and actual native output corruption must fail.

## Task 6: device lanes and real collector

- [x] Implement resident heterogeneous lanes with both action rows and separate raw-game versus autoreset contracts; host lane and tensor binding gates pass.
- [x] Host tests cover per-lane RNG, true terminal observations, double reward accumulation, four-frame stacking, selective reset and isolation. Mocked stream-lifetime checks are binding tests, not CUDA runtime evidence.
- [x] Verify real CUDA stream/device lifetime handling and end-to-end TensorPPO updates in the recorded lane/training gates.
- [x] Record the additional real native-CUDA collector truncation/bootstrap regression: three-tick horizon, seven collected steps, true bootstrap and a negative control. All three focused tests passed on the GPU host in 3.977s; this is actual collector evidence beyond lane-only autoreset tests.
- [x] Execute resident CPU-vs-CUDA replay/parity tests without uploading corrected state between ticks. Real GPU lane suite: 37 pass and one two-GPU skip. Local no-CUDA tests skip explicitly, never count as CUDA passes.
- [x] Complete the raw-game TensorPPO benchmark with batched CUDA policy inference and no per-step host action/observation transfers in the native-GPU lane; preserve raw both-player observations/rewards and matched optimizer settings for engine comparisons.
- [ ] Port/connect the current combat curriculum wrapper separately. It is not implemented in this raw-game lane, and raw-game benchmark results cannot be labeled curriculum-equivalent.

## Task 7: bounded rented-GPU qualification

- [x] Coordinator re-audited/rented the single replacement CUDA host 50337302 after the recorded host gate, with the finite deadline above. This does not complete GPU qualification.
- [x] Record the pinned runtime/source/capture and final retrieval manifests; the coordinator verified remote-file hashes against both local archives.
- [ ] Historical preparation record: a separately timed upload/download throughput check before long work is not recorded here. Successful checksum-verified final delivery is recorded below; no transfer-bandwidth number is inferred from it.
- [x] Run CUDA parity and compute-sanitizer smoke; downloaded lane and sanitizer logs are linked above. Zero sanitizer errors qualifies that smoke, not every possible execution.
- [x] Matched raw-game baseline: original Python CPU simulator vs native CPU vs native GPU, policy always CUDA, same initial parameter hash/architecture/game count/rollout/PPO batches/epochs/raw rewards/both-player policy setup. Recorded runs freeze warm-up exclusions; Python/native generators need not produce identical learned trajectories. Current-curriculum equivalence remains a separate unimplemented gate.
- [x] Record collection, update and total wall time, useful transitions/s and coarse GPU telemetry; diagnostic profiling separately attributes action/value forward, native environment step, buffer insertion, GAE and PPO update. Host launch/wait overhead is visible, but nested regions and CUDA annotation spans are not additive timing partitions.
- [x] Short training smoke validates finite losses and optimizer progress; CUDA checkpoint reload into the original CPU game passes 768 held-out game ticks. Scale runs are recorded below and remain infrastructure/performance evidence only.
- [x] Download and SHA256-verify models/logs/configs/telemetry: 121 files / 302,066,380 bytes match the remote final inventory in both local archives. Qualification captures and frozen oracle are preserved separately with verified hashes.
- [x] Confirm provider reports the instances stopped, not destroyed, and record exact final spend/state: strict 03:20:30 UTC audit reports zero running, both instance states stopped, $9.131434438895763 cumulative spend. Monitor PID 1519 stopped cleanly. The watchdog's verification timeout is retained; the independent provider-state audit, not merely the stop request, establishes closeout.
- [ ] Publish the preserved offline W&B bundle only after explicit approval for the specific external payload. The local publication attempt was rejected for missing authority; do not retry automatically. This publication block does not reopen or invalidate the completed GPU qualification/benchmark work.

### Throughput accounting

Report **game ticks/s** and **player transitions/s** separately. One valid raw-game tick advances both player slots, so `player_transitions = 2 × game_ticks`; a rollout with `G` games and `T` ticks has `G × T` game ticks and `2 × G × T` player transitions. This factor of two is accounting, not a speedup. Reset, failed, or padding steps are not useful game ticks. Compare like-for-like denominators across Python/native CPU/native GPU; distinguish environment-only, collection, and total collection-plus-update rates.

The following measured rates are **player transitions/s, total collection plus PPO update**, from the unprofiled [benchmark aggregation](../../../baselines/fullgame-20260908/summary.json). Divide by two for game ticks/s. Policies always execute on CUDA, including Python-simulation and native-CPU rows. Matched architecture is `[512,512,256]`, 1,053,973 parameters, 128 rollout steps, eight PPO epochs, and optimizer batch 4,096 except the explicitly separate final row.

| Run | Python CPU simulation | Native CPU simulation | Native CUDA simulation |
|---|---:|---:|---:|
| 128 games, eight Python workers | 8,475.68 | 29,866.55 | 32,298.66 |
| 128 games, 14-worker repeat | 8,287.76 | 29,479.54 | 30,679.54 |
| 512 games, native scale comparison | — | 36,529.71 | 39,452.31 |
| 2,048 games, native scale comparison | — | 38,174.61 | 41,661.51 |
| 2,048 games, optimizer batch **16,384** | — | — | 160,935.69 |

The 16,384-batch result changes the optimizer minibatch count and is **performance-only**, not an engine-only speedup or a matched training-equivalence comparison against batch 4,096. At 2,048 games / batch 4,096, PPO update time accounts for **97.4%** of native-CUDA collection-plus-update wall time, limiting the benefit of faster simulation.

The passed [diagnostic profiler](../../../baselines/fullgame-20260908/downloads/profile-128-v1/profile.json) records **63,372 CUDA events**, complete CPU user-range attribution and unchanged source hashes. Large host-region durations relative to attributed kernel durations, together with the update share, provide strong evidence of host launch/synchronization and learner overhead; they do not show that native environment kernels dominate runtime. The trace includes profiler overhead and warm-up work, and its nested inclusive times / same-name CUDA annotation spans must not be summed or used as headline throughput.

The [original-game checkpoint validation](../../../baselines/fullgame-20260908/downloads/checkpoint-original-v1/summary.json) reloads the saved 1,053,973-parameter policy on CUDA with matching parameter SHA256 `54e215c27865bb2b0d4fb4047ebd5f2ec55618bfc64a96c8129f0d2dd7635141`. Seeds 3, 7 and 101 each execute 256 game ticks: **768 game ticks / 1,536 player transitions** total, finite parameters/actions and preserved checkpoint bytes. No damage increase was observed; player-0/player-1 stock losses were **2/0**, **2/0**, and **0/2** respectively. The trace proves reload/interaction infrastructure, not a strong bot, successful fighting, or human-level play.

## Review and completion contract

- [x] Independent host specification and quality reviews completed; acceptance-boundary fixes were regression-tested, including native timeout-state corruption and both-slot `Result.obs` corruption.
- [x] Confirmed focused host suites, legacy navigation regression, host compiler checks and Python lint pass at the recorded gate.
- [x] Real CUDA runtime, sanitizer, raw-game training/benchmark and original-game checkpoint-reload gates have passing receipts; focused benchmark/validator/profiler suites also pass within their stated scopes.
- [x] Accept the verified finite coverage-completion union and actual short-horizon real collector regression; all requested GPU gates have passing coordinator-confirmed results.
- [x] Finish artifact preservation with coordinator-confirmed hashes in the isolated and durable local archives.
- [x] Finish cloud closeout with exact spend and independently provider-confirmed stopped/not-destroyed state at 03:20:30 UTC; zero instances running.
- [ ] External W&B publication awaits the user's specific payload approval; offline data is preserved, and no retry is authorized by the completed GPU task alone.
- [x] Status distinguishes the complete declared finite qualification suite from an exhaustive state-space proof, and retains all historical failed receipts and prior incomplete coverage summaries.
- [x] Human-level playing strength remains a separate learning/evaluation outcome; neither exact gameplay parity nor measured PPO throughput establishes it. Current-curriculum wrapper parity remains unimplemented.
