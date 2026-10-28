# Skill-gated navigation-to-combat curriculum

Status, updated 2026-09-08: the B0/B1 trainer, telemetry, gates and replay infrastructure are implemented and locally tested. The original two-hit B1 geometry FAILED scripted safety qualification; CUDA validation and model training have NOT run. See [implementation receipt](2026-09-08-combat-curriculum-implementation.md). The lesson definition below remains the approved proposal, not a qualified training task. The user selected the preserved Large-145 checkpoint at approximately 10M steps and requested moderate exploration and a complete progression. Numeric pilot settings below are proposed experimental defaults, not measured optima. This document does not authorize a new rental, raise the existing budget, or permit instance destruction.

## 1. Objective and the decision we are changing

Produce a fighter that deliberately approaches, lands attacks, responds to opponents, recovers and wins real matches, ultimately against humans on the user's website. Navigation completion, damage farming, GPU utilization and wins against self-destructing opponents are not substitutes for that objective.

Advance this individual model when its current skill is demonstrated. Do not wait for the other eight model-size comparison arms. Do not assign every phase another 5M/10M/20M steps. A plateau or budget stop triggers diagnosis, not automatic promotion or endless repetition.

Three options were considered:

1. **Continue from Large-145 at 10M with selective combat introduction (selected).** Keeps demonstrated movement, honors the user's checkpoint choice and exposes the missing skills directly.
2. Continue the 25M navigation checkpoint through the same bridge. A legitimate later transfer comparison, but no evidence currently establishes that it learns combat faster. Preserve it; do not run a second full curriculum by default.
3. Restart everything. Rejected for this experiment: measured navigation improved rather than deteriorated, and the demonstrated combat failure has a specific action-unmasking mechanism.

## 2. Verified starting material

Local parent checkpoint:

```text
/Users/elliot18/Projects/ml-research/AI_2-native-ppo-optimizations-20260907/baselines/native-capacity-session11/retrieved/artifacts/0a-steps10027008-dev20/arch-1536-1536-1536_seed145/train/final.zip
SHA256: 0e2f64e02cd0f7d300554f7a6cb641065c75771f69f55c0885258dda32afc6cd
Bytes: 123047421
Training transitions: 10027008
Architecture: [1536, 1536, 1536], separate actor/value MLPs
Parameters: 10249749
SB3: 2.5.0; PyTorch: 2.4.1+cu121
Observation: 64 raw values, four zero-padded frames, 256 float32 inputs
Action: ten independent Gaussian outputs, clipped to [0,1], pressed if >0.5
Saved entropy coefficient: 0.0; learning rate: 0.0003; target_kl: null
Saved rollout: 64 environments x 512 steps; batch_size 8192; eight epochs
```

Read the exact checkpoint and source hashes before execution; never substitute a same-named final.zip. Preserve the parent unchanged. New weights, optimizer state, calibration and counters belong to a new lineage.

Existing 100-episode development evidence per action mode:

| Large-145 checkpoint | Nearby success D/S | Nearby falls D/S | Wider success D/S | Wider falls D/S |
|---|---|---|---|---|
| 10,027,008 | 98% / 97% | 1% / 2% | 81% / 73% | 14% / 24% |
| 20,021,248 | 99% / 100% | 1% / 0% | 87% / 84% | 8% / 12% |

The 10M model qualified for harder lessons, not unrestricted combat. Approximately 10M further near-navigation steps were beyond its first observed near-skill gate; they were not demonstrated to damage it. The 25M checkpoint includes approximately 5M wider navigation after 20M nearby navigation.

The session12 matches used the 25M checkpoint with zero training updates. Attack outputs previously ignored by navigation became effective. In the Large phase4 video's first 300 frames, left was requested 300 times and AttackState occupied 287 frames. This is the failure the new curriculum must explicitly prevent.

## 3. Environment and interface contract

The original corrected game is the combat reference:

```text
/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected
```

Its environment/environment.py hash is `c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230`. Snapshot the full game, attack assets, train.py, evaluation code and dependencies in the launch manifest.

- Keep input shape/order, float32 casting and zero-padded four-frame history. Do not add observation normalization or repeat the initial frame four times.
- Keep ten output dimensions for checkpoint compatibility. Store raw sampled actions separately from clipped/executed actions and the per-transition allowed-control mask.
- A passive target issues zero inputs but remains a real physics body: damage, stun, displacement and knockback must work normally. Do not freeze its position after every step.
- Curriculum repositioning occurs only at explicit encounter boundaries, after attack/stun resolution, never during a live hit. Record each reposition event. Clear all relevant body/contact/state bookkeeping through a tested reset helper.
- Early starts use validated stable floor locations away from edges, both player slots and both facing directions. The target must be farther away than measured light-attack reach when a movement response is required.
- Use real collision/hit detection. Navigation retention counts body/capsule overlap, not exact center alignment; if changing the historical center-based success predicate, version the metric and reevaluate the unchanged 10M control before comparing percentages.
- Episodes use game ticks at 30 Hz. Deadlines are simulated time, separate from wall-clock launch limits. Stock loss ends early lessons immediately; final-stock elimination takes priority over timeout on the same tick.
- Multi-stock episodes are introduced explicitly later. They retain frame history across respawns, clear history only at episode reset and include recovery/reengagement measurements.

## 4. Complete progression

Stage identifiers here are new `bridge_*` identifiers; do not overload old phases 1-4 and silently assume their opponent mixtures implement these drills.

| Stage | Lesson and task | Controls newly effective | Initial task window | Evidence needed to leave |
|---|---|---|---|---|
| B0: verify parent | Reproduce nearby/wider navigation; record active-button probabilities; qualify passive combat physics and masks | None | Existing navigation: 20 s | Hash/schema/runtime checks and matched movement control pass; no extra near training by default |
| B1: hit and reengage | On stable ground, hit a nearby passive target. Once the attack and stun resolve, relocate the target safely beyond attack reach; approach and land a second distinct light attack | Light only (7) | 10 s total; response within 2 s after relocation | Actual hits in both encounters, measurable pursuit after relocation, no stock loss; retained navigation |
| B2: approach across distance | Approach passive targets outside attack reach, from both sides and eventually across qualified gaps; hit and reengage | None | 20 s, two encounters | Approach/contact on both sides; wider-navigation mastery and nearby retention |
| B3: intercept a moving target | Nonattacking target follows safe patrols with reversals; speed curriculum starts near 25% ordinary walking speed, then 50%, then full speed | None | 20 s; first hit by 8 s | Two distinct attack contacts after different approach opportunities, target actually moves; no collapse on held-out patrols |
| B4: respond to retaliation | Safe opponent approaches and returns a telegraphed, rate-limited light attack; reduce telegraph and increase cadence only after qualification | No new learner control; dodge already exists | 30 s | Land damage, avoid at least one real attack opportunity and reengage; positive net damage in qualifying exchanges, not inactivity |
| B5: recover and finish | Alternate known-recoverable offstage resets with single-stock fights against opponents that can stay onstage; teach attack-assisted recovery separately before harder edge fights | Heavy/recovery (8), selectively initialized and qualified in a separate substage | Recovery 5 s; fight 45 s | Successful recovery followed by pursuit; genuine stock eliminations and acceptable own-stock-loss rate |
| B6: weapon and match transitions | First one weapon type, then another; pickup, fight, disarm and resume unarmed play. Introduce full starts, moving ledges and three-stock respawns in separate substeps | Pickup/throw (5); never unlock together with an unqualified heavy output | Drill 20 s; match 90 s | Each weapon and unarmed fallback pass; reengagement after respawn; timed-out games reported separately |
| B7: varied opponents and human validation | Frozen opponent league, held-out scripts, then actual website playtests | No additional controls; taunt remains disabled | 90 s development matches; final website rules must match deployment | Broad skill retention and actual contest wins, followed by separately reported human and deployment evidence |

Timings and safe placements are starting specifications. Before training a stage, run a scripted competent controller against its measured attack durations, geometry and motion. If a task is physically impossible, revise/version the stage before its first learning run; never silently loosen a failed model's test.

### B1 exact success sequence

1. Start on the same stable platform, separated enough that an actual approach is needed, with collision-free spawns validated against measured attack reach.
2. Detect a learner-caused damage event from a light attack. Count the attack instance once, not each multi-hit frame. Do not count merely entering AttackState.
3. After both combatants finish that attack/stun interaction, relocate the passive target to a safe point at least 1 world unit beyond the measured light-attack contact range. Validate enough continuous supported floor to close that distance before a hit is possible. The position change is the observable pursuit cue. Record the boundary and continue the clock.
4. Within 60 ticks, close the target-distance gap by at least 1 world unit. Require a second distinct real attack contact within 90 ticks after relocation.
5. After the second attack resolves, the learner must still be alive and supported by valid floor. No reward for standing still or waiting. Deadline misses and falling fail the drill.

This is a narrow drill, not proof of general fighting. Holding attack near one dummy cannot satisfy the reposition-and-reengage requirement. A geometry audit can lengthen an impossible timing contract before launch; it cannot reinterpret a completed failing run as a pass.

### Retention without delaying the new lesson

Start B1 with four combat rollouts followed by one on-policy navigation rollout: 80% new skill / 20% maintenance by transitions. Alternate nearby/wider maintenance, shifting maintenance to wider/recovery scenarios as nearby skill remains mastered. Use separate homogeneous rollouts and the appropriate action mask; do not combine differing raw reward scales and unlabeled task masks inside a rollout.

Maintain a task-specific critic/value head for navigation versus combat while keeping the same shared actor interface. Initialize the navigation critic from the checkpoint; initialize the new combat value head separately. Log task-specific losses. This avoids applying the combat value estimate to a different navigation reward objective without an explicit task indicator. Deployment needs only the actor, not these training critics.

Retention uses fresh on-policy experience from the current model, not stale PPO transitions from the old checkpoint. A 20% ratio is a pilot default, not a proven best mix. If retention deteriorates, stop the learning block and diagnose before increasing difficulty. Do not force navigation-only updates indefinitely.

## 5. Exploration and transfer policy

Moderate exploration means the agent tries alternatives without continuously holding attack or destroying movement. The saved `ent_coef=0` does not mean deterministic training: the policy still has learned Gaussian standard deviations.

### Protect learned movement; calibrate only newly activated controls

- First evaluate the unchanged 10M actor with its original movement mask. Audit per-dimension mean, learned standard deviation, press probability, simultaneous presses and hold/run lengths on recorded states and closed-loop scenarios.
- Preserve actor feature layers, movement output rows 0/1/2/3/4/6 and their standard deviations at transfer. Do not globally reinitialize the model or its noise.
- For B1, create a derived checkpoint with only the light-output row initialized to weights zero, mean bias 0 and standard deviation 0.30. Its initial Gaussian threshold-crossing probability is about 4.8% per sampled decision; this is a starting calibration, not a final target rate or a claim about the original checkpoint. Confirm empirically on CUDA.
- Keep pickup5/heavy8/taunt9 disabled. When later enabling 8 or 5, repeat isolated calibration rather than exposing their old unused means. No taunt curriculum is needed for competence.
- During an initial one-rollout calibration block, freeze the learned actor features, movement rows and their log standard deviations; fit the new attack row and combat critic. Use entropy0.001 and the other E1 settings below. Keep a separate pre-calibration actor control. After transfer/objective/safety qualification, unfreeze actor features for the joint curriculum under the conservative PPO controls below. This qualification does not require B1 behavioral mastery; learning B1 is the subsequent pilot's purpose.
- Use fresh optimizer state for the new combat objective. Retain a task-specific navigation critic, initialize the final combat value head afresh and record all parameter changes. Do not inherit navigation Adam moments or silently treat navigation value targets as combat targets.

### PPO accounting is part of correctness

For a fixed allowed-control mask `m`, calculate old/new action log probabilities, entropy and KL from the same allowed dimensions. Retain the sampled, unclipped action for likelihood calculations. Inactive dimensions must not contribute to PPO ratios or receive entropy updates. Carry the mask and task identity with every stored transition. A mask may change only at a logged task/stage boundary, never mid-rollout. Freezing masked parameters alone does not make stock all-dimension entropy an executed-button metric.

This requires a tested learner extension, not a claim that the current stock trainer already does it. Do not run the new stages on an unmodified all-dimension loss and describe them as equivalent. Preserve ten-dimensional actor exports for the existing gameplay adapter; export the allowed-controls/stage metadata too.

### Initial bounded exploration experiment

After source/unit/physics checks, compare two cloned B1 starts with the same initialized weights, seed schedule, task mix and transition budget:

| Setting | Arm E1 (default candidate) | Arm E2 (slightly more exploration) |
|---|---|---|
| B1 active-dimension entropy coefficient | 0.001 | 0.003 |
| Constant actor learning rate | 0.0001 | 0.0001 |
| PPO clip range | 0.10 | 0.10 |
| target_kl | 0.01 | 0.01 |
| Epochs | 4 | 4 |
| Environments / rollout | 64 / 512 | 64 / 512 |
| Minibatch | 8192 | 8192 |
| gamma / gae_lambda | 0.99 / 0.95 | 0.99 / 0.95 |
| max_grad_norm / vf_coef | 0.5 / 0.5 | 0.5 / 0.5 |

Use homogeneous task rollouts and report entropy as both active sum and per-active-dimension mean. Navigation maintenance uses entropy0.001 in both arms; only B1 entropy differs. Three paired rounds of four B1 rollouts plus one navigation rollout give 491,520 additional transitions per arm, including maintenance. Alternate maintenance near/wide/near for both arms. Run E1 and E2 sequentially on the same GPU; preserve each round before continuing. The one-rollout calibration is separately counted and shared by cloning its result, not hidden warmup training.

Prefer the lower-exploration arm when development success differs by fewer than two successes in a 20-episode panel and neither fails retention. Such a coarse tie-break is an operational choice, not evidence of superiority. If E2 improves sampled play but deterministic play remains attack-spam or nonattacking, it has not passed. Do not assume success within this budget; a pilot can conclude neither setting is ready.

Record effective learning rate every update. Use explicit constant schedules for this transfer; saved schedules and cumulative counters must not silently shrink the intended rate. Never claim `log_std_init` overwrites learned checkpoint variances. `target_kl` is an update guard, not a guarantee against forgetting.

## 6. Reward and anti-shortcut contract

Training reward and promotion metrics are separate. Do not use the legacy aggression/AttackState reward. Never pay for pressing attack, an attack animation, facing the target, idle survival, or opponent self-destruction.

For B1, start with a deliberately bounded reward:

- +1 for the first valid learner-caused light contact in each of the two encounters, maximum +2 per episode.
- +2 for completing the entire hit/reengage sequence safely.
- -5 for learner stock loss; -1 for timeout without stock loss. Stock loss suppresses same-tick completion reward.
- -0.001 per tick. No uncapped distance-shaping term in the first pilot; later potential-based shaping needs its own boundary/reset tests.
- Dummy reset damage, stock bookkeeping and curriculum teleports are never damage/hit rewards. Clear per-encounter credit only at an explicit boundary.

In later stages, introduce measured damage taken, defense and genuine eliminations with separate capped terms. For elimination training, log actual match outcome regardless of cause, but require a learner-caused hit/knockback event in the victim's final 90 ticks before paying an attributed-KO bonus. This is an explicitly limited attribution window, not proof of counterfactual causation. Unknown attribution stays unknown. Do not penalize all edge positions uniformly: legitimate edgeguards and recoveries require those positions.

Before B4/B5 learning, instrument controlled hits to verify damage/knockout reward signs for both player slots and simultaneous KOs. Do not inherit the older global reward manager or legacy timeout-as-win callback.

## 7. Measurement, gates and stopping

Every development panel is balanced by player slot and covers deterministic and sampled actions separately. Training, development and final holdout scenario IDs are disjoint. Keep the old final holdout banks untouched. Repeated use of development cases is model selection, not an unbiased final test.

- **Fast check:** after every 163,840 additional transitions (four combat plus one maintenance rollout), evaluate 20 episodes per mode for the current stage and 20 per mode for nearby retention. Keep wider/recovery diagnostics appropriate to the current stage. This interval is a check, not a minimum amount to train after mastery.
- **Promotion candidate:** at least 18/20 successes and at most one own-stock-loss failure in both modes on two consecutive fast checks. No non-finite values, interface failures or observed systemic movement lock.
- **Confirmation:** fresh 100-episode stage panels per mode, at least 90/100 success, no more than five own-stock-loss failures, and at least 40/50 successes from each player slot for B1-B3. Recheck nearby retention on the same baseline metric: no more than five percentage points below the unchanged parent's freshly measured success, at most 5% falls. B2 also requires wider-navigation success >=90% and falls <=5% in both modes.
- **Later-stage gates:** B4 uses the same 90/100 trial success criterion for attack/avoid/reengage sequences. B5 requires >=90/100 recoveries on qualified recoverable starts and >=70/100 genuine elimination wins against each prequalified weak opponent, with zero-damage wins reported separately and excluded from the credited-skill count. B6 repeats these gates per weapon and requires >=90/100 timely reengagements after respawn. These are provisional competence thresholds, not statistical guarantees.
- **Regression response:** a >=10-point success drop or >10% nearby falls on a fast check pauses updates for a fresh 100-case check. Confirmed retention failure restores the last qualified checkpoint and optimizer/task state. Rollback is a recorded new branch, not overwriting an artifact.
- **Plateau review:** after three fast checks with fewer than two additional successes/20 in the weaker mode and no improvement in time-to-hit/reengagement, inspect trajectories and reward/action metrics. This coarse flag is not proof of convergence. Increase diagnostic sample size before claiming a plateau.
- **Finite review boundary:** after 2,621,440 new transitions in one stage (16 fast-check blocks), stop automatic extension for a reviewed decision. Continue only with an evidence-backed next finite block and budget. No fixed 10M allocation and no promotion merely because time/credit ran out.

Keep best-qualified, latest and final checkpoints separate. A new phase starts from the best qualified checkpoint, never automatically from the largest timestep counter. Each model advances independently.

### Mandatory behavioral receipts

Per episode: scenario/stage, seed, side, outcome, every reward component, distinct hit events and their attack IDs, time-to-first-hit, target-relocation response, target/opponent trajectory, stock events and attribution category, time limit and actual terminal reason.

Per rollout: raw versus executed actions, per-control allowed mask/mean/std/press rate, consecutive attack-hold distribution, attack-state occupancy, jump/dodge usage, active entropy, KL, clipping fraction, task-specific value errors, navigation retention, true end-to-end steps/s, collection/update/evaluation time and GPU telemetry. Nonconstant float vectors are not evidence that button choices vary.

Videos at every proposed promotion: four preselected cases covering both sides and both action modes, plus the two worst failures by a predeclared failure ranking. If there are no failures, label that fact instead of substituting a showcase. Include frame/tick, player identity, allowed/executed buttons, hit count and stage condition. Validate rendered versus recorded action/state trajectories. Videos supplement numeric gates and remain a user-visible review requirement, not cherry-picked proof.

## 8. Later league and human evidence

Start B7 with an immutable opponent distribution: 50% recent qualified snapshots, 30% older/hard snapshots, 20% stage-qualified scripted opponents. Draw a fixed opponent at episode reset. Do not evaluate only against the current training opponent. Prove adapter/frame-history correctness separately for each player; the old self-play loader is not ready for these checkpoints as-is.

Qualification includes unseen patrol timings, both sides, larger initial separation, moving-platform phases, weapons, edge recovery, opponent knockback and multi-stock reengagement. Report per-opponent performance, not just one pooled number. An opponent must demonstrate engagement and staying onstage before wins against it support a skill claim.

Human readiness is a separate endpoint: actual website matches with the user, then additional willing players and varied playstyles. Predeclare match rules and sample size before measuring wins; report opponent experience and uncertainty. Proposed initial review is 20 user matches, not a universal human-strength claim. Website action fidelity, timing, end-to-end latency, reset handling and exploit checks must pass before attributing losses to model skill. No external human coordination or website deployment is authorized by this planning document.

## 9. Runtime, scope and existing-code traps

- The current native learning runner supports movement-only phases 0a/0b and requires nav entropy0. It cannot execute this attack curriculum. Do not label B1 training GPU-native or assume navigation throughput persists.
- First validate the combat curriculum with the corrected Python/Pymunk game on the GPU host's CPU and batched policy inference/PPO updates on CUDA. This is not local CPU model training. Native combat acceleration is a separate parity-gated project, not a prerequisite for a small correctness pilot.
- Reuse controlled `payload/scripts/train_job.py` concepts and frozen `make_combat_env`, not the optimization branch's older global reward state/win-sign behavior. Do not enable self-play merely by adding YAML: controlled jobs intentionally prohibit it, and legacy opponent loading lacks the validated checkpoint adapter.
- Combat bridge implementation is new work: stage reset/controller logic, per-task value handling, selective action distribution, reliable metric gates and staged runner. None of these is claimed complete by this document.
- Keep learning/inference on CUDA, one active GPU instance, existing budget ceiling and no destruction without confirmation. Audit live credit/rate/identity before launching; old quoted prices and credit snapshots are not current authorization.
- Preserve checkpoints/logs/config/source and per-round results locally with checksum verification immediately after each round and before another round. Use atomic checkpoints and a local downloader while training. If verified local copies lag one fast-check block, pause training; never depend on remote-only results. No new cloud-storage setup is required.
- This deliverable changes documentation only on `codex/combat-curriculum-20260907`. It does not merge other agents' changes, change reward code, train, rent, restart or destroy an instance.

## 10. Evidence and technical references

Local evidence, relative to the optimization checkout named in section 2:

- `docs/superpowers/specs/2026-09-07-model-size-training-design.md:29`: old all-nine shared advancement rule.
- `baselines/native-capacity-session11/retrieved/artifacts/0a-steps10027008-dev100/arch-1536-1536-1536_seed145/eval/summary.json`: 10M nearby/wider measurements.
- `baselines/native-capacity-session11/retrieved/artifacts/0a-steps20021248-dev100/arch-1536-1536-1536_seed145/eval/summary.json`: matched 20M results.
- `native_learning_runner.py:45`: movement-only phase/config restriction.

Frozen reference boundaries: `train.py:327` (per-step target freezing), `train.py:1378` (corrected combat factory), `UTMIST-AI2-main/environment/agent.py:852` (passive and scripted agents), `UTMIST-AI2-main/environment/environment.py:2991` (attack control), and sibling `payload/scripts/train_job.py:67` (no self-play).

The saved checkpoint uses SB3 2.5.0. Official [PPO documentation](https://stable-baselines3.readthedocs.io/en/v2.5.0/modules/ppo.html) defines entropy weighting, KL limits and progress-dependent schedules; it does not establish good coefficients for this game. [Distribution implementation](https://stable-baselines3.readthedocs.io/en/v2.5.0/_modules/stable_baselines3/common/distributions.html) shows learned Gaussian standard deviations and dimension-summed likelihood/entropy. [Loading/training setup](https://stable-baselines3.readthedocs.io/en/v2.5.0/_modules/stable_baselines3/common/base_class.html) explains restored parameters and timestep continuation. For thresholded Gaussian actions, the proposed initial press probability is derived as `1 - NormalCDF((0.5 - mean) / std)`; verify executed behavior, not only this calculation.
