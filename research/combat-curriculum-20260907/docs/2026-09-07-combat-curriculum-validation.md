# Combat Curriculum Qualification and Pilot Plan

> **For agentic implementers:** Use superpowers:subagent-driven-development or superpowers:executing-plans after the curriculum design is approved for implementation. This is the experiment/validation plan; it does not claim the new trainer exists.

**2026-09-08 implementation update:** B0/B1 infrastructure is implemented on the isolated branch with 142 new tests plus 34 existing regressions passing. Scripted qualification rejected the original two-hit geometry. CUDA checks, training, live monitoring and promotion remain unexecuted. See [current receipt](../specs/2026-09-08-combat-curriculum-implementation.md); the planning receipt at the end is historical, not the current implementation status. Do not check off a combined local/CUDA acceptance criterion using only local evidence.

**Goal:** Qualify a gradual combat curriculum from the preserved 10M Large-145 parent without losing demonstrated movement or mistaking attack spam for learning.

**Architecture:** Keep the original corrected combat game as reference, the ten-dimensional actor interface and independent four-frame histories. Add stage-specific passive/scripted opponents, selectively enabled action distributions, task-specific training value heads and objective behavioral gates before extending the frozen-opponent league.

**Tech Stack:** Python 3.11, saved SB3 2.5.0/PyTorch 2.4.1 CUDA stack, original Pymunk/Pygame game, existing isolated combat adapter and trace runner. No new dependency is assumed.

---

## Scope and dependency order

Deliver the B0/B1 controlled bridge first. B2-B7 are the ordered curriculum roadmap, not permission to implement every subsystem before seeing whether the first transfer works.

Implementation must remain on an isolated branch and must not overwrite frozen payloads or another agent's dirty files. Use immutable source/input snapshots and explicit paths in a run manifest. Existing code surfaces are reference/reuse points, not a request to mutate their archived copies.

Suggested new ownership boundaries for the subsequent implementation:

| Unit | Responsibility | Required tests |
|---|---|---|
| `combat_curriculum.py` | Versioned stage definitions, mask, reset/encounter state machine, passive/scripted controller parameters and reward events | `tests/test_combat_curriculum.py` |
| `curriculum_ppo.py` | Ten-output actor transfer, allowed-dimension likelihood/entropy/KL, task-specific critics, parameter freeze/optimizer provenance | `tests/test_curriculum_ppo.py` |
| `curriculum_evaluation.py` | Scenario banks, success/retention/plateau gates, per-attack metrics and no-timeout-win semantics | `tests/test_curriculum_evaluation.py` |
| `train_combat_curriculum.py` | Finite CUDA-only runner, manifests, continuation counters, snapshots, task-round schedule and local-preservation gates | `tests/test_train_combat_curriculum.py` |
| Existing `combat_policy_adapter.py` and `combat_comparison.py` | Reuse strict checkpoint/stacking and trace/replay mechanics; extend only with regression coverage | Existing 34 focused tests plus new scripted-opponent cases |

No implementation code is included or claimed in this planning-only deliverable. A subsequent code-level implementation plan must resolve interfaces before parallel edits; do not treat filenames above as already importable modules.

## Task 1: Freeze and qualify the parent

- [ ] Verify the parent path, SHA256 `0e2f64e02cd0f7d300554f7a6cb641065c75771f69f55c0885258dda32afc6cd`, 123047421 bytes and 10027008 transitions against the design.
- [ ] Capture source/asset/runtime hashes and installed library versions. Confirm whether the runtime matches the saved checkpoint; record any tested compatibility difference.
- [ ] On CUDA, inspect actual architecture, policy device, finite parameters and learned mean/std/press probabilities on the recorded observation bank. Do not run real checkpoint inference on local CPU.
- [ ] Reproduce historical near/wide navigation controls using the original metric and seeds; record a separate new-metric control if collision-overlap completion replaces center-based completion.
- [ ] Verify player-slot order, initial zero padding, resets and raw64-to256 stacking against the real environment. Fail closed if the unchanged parent fails the control without explanation.

Acceptance: immutable parent plus verified runtime/input contract, baseline behavior and calibration data. It is acceptable to discover a mismatch and stop; it is not acceptable to infer that the parent is bad from a mismatched wrapper.

## Task 2: Qualify the B1 game task without learning

- [ ] Test allowed controls exactly `{0,1,2,3,4,6,7}`. Values exactly0.5 are not pressed;5/8/9 remain zero despite large raw outputs.
- [ ] Use zero-action opponent inputs while leaving physics, damage, stun and knockback intact. Compare a deterministic light-hit fixture against unwrapped original combat through attack resolution.
- [ ] Prove no per-tick target teleport occurs during a hit. Verify that explicit encounter relocation clears target contact/body/state bookkeeping and creates an observable target-position change.
- [ ] Measure attack reach, animation duration and safe spawn envelopes for both sides. Prove a scripted controller can approach, hit, reengage after relocation and hit again within the declared time limits.
- [ ] Stationary held-light controller must fail the full B1 success predicate even if it earns the first contact reward. Walking-off, idle, taunt, wrong-facing, simultaneous-key and attack-lock controllers must not pass.
- [ ] Count one rewardable hit per encounter/attack ID; repeated damage frames, dummy reset and health reset must not add credit. Maximum positive B1 reward is4, learner stock loss costs5, timeout1 and the tick cost remains explicit.
- [ ] Test coincident hit/fall/deadline events; own stock loss suppresses completion, and an actual last-stock elimination is never a timeout win.
- [ ] Test collision-overlap navigation completion with an off-center valid overlap and a nonoverlap negative case; preserve a versioned historical evaluator.

Acceptance: positive controller succeeds from both sides; exploit controllers fail; ordinary game hit trajectory is preserved outside explicit curriculum encounter boundaries.

## Task 3: Qualify transfer and PPO accounting

- [ ] Compare parent and derived actor state dicts. Only the newly enabled light row and its log_std may differ before learning; actor feature layers and movement means/variances must match exactly.
- [ ] Sample the calibrated head on CUDA and verify the observed initial press rate is consistent with mean0/std0.30 within a predeclared binomial tolerance; verify deterministic behavior separately. Do not pretend this is the original parent's measured attack rate.
- [ ] Test old/new log-probability and entropy against a hand-computed active-dimension Gaussian reference. Alter only disabled dimensions and require unchanged PPO ratios, KL and objective gradients on active dimensions.
- [ ] Save raw unclipped action separately from executed clipped action and mask. Verify training uses raw samples for log probabilities, while execution uses clipped/thresholded inputs.
- [ ] Reject mask changes mid-rollout, incompatible task IDs and rollouts evaluated with the wrong mask or critic. Save and reload both task critics and their optimizer state without silently selecting the wrong one.
- [ ] During one calibration rollout, prove movement output rows/features remain unchanged while the light row/combat critic can update. Check fresh-optimizer identity; preserve navigation value parameters at initialization.
- [ ] Verify joint unfreezing updates the actor only under the configured PPO objective, and no disabled-row optimizer momentum changes its parameters.
- [ ] Test one update and save/reload continuity on CUDA; verify loaded actor outputs and value heads before and after export/import. The deployed actor export retains256 inputs and10 outputs.
- [ ] Verify constant learning rate0.0001, entropy selection, target_kl0.01, rollout/batch/epoch settings and cumulative versus additional counters from runtime values, not merely YAML.

Acceptance: objective parity, parameter-preservation evidence, correct task-specific values and verified reload. If this extension is not yet correct, do not run a larger stock-PPO experiment and label it the proposed curriculum.

## Task 4: Qualify evaluation, promotion and data preservation

- [ ] Seed-bank collisions among train/development/final are rejected. Preserve original final holdout seeds and label historical development reuse.
- [ ] Unit-test exact promotion boundaries:18/20 success with1fall passes the fast threshold,17/20 or2falls fails;89/100 success or6falls fails confirmation. Both action modes and both sides must satisfy the design.
- [ ] Retention uses a freshly verified unchanged-parent baseline under the same metric. A five-point permitted drop is not compared between old center-based and new overlap-based definitions.
- [ ] Confirm wider-navigation mastery is required at B2, not used to block every B1 light-contact lesson. Verify paused/failed/plateau/budget statuses never become completed/mastered.
- [ ] Verify no-damage opponent self-destruction is visible in outcomes but cannot satisfy contact or credited-elimination skill gates. Keep unknown attribution separate.
- [ ] Record meaningful active-button run lengths and state transitions; a changing float vector with constant pressed buttons fails the variation sanity check.
- [ ] Render preselected cases and worst failures; compare replay state/action trajectories to original traces and label any numerical tolerance across runtimes. Preserve full videos, not only thumbnails.
- [ ] Test interruption/resume and atomic checkpoint writes, missing local downloader, checksum mismatch, stale instance identity and insufficient remaining credit. At most one unverified fast-check block may be outstanding before pausing.

Acceptance: synthetic good/bad episode fixtures produce the intended gate decisions, replay fidelity is measured, and intentional preservation failures stop continuation.

## Task 5: First finite GPU experiment (not launched by this document)

- [ ] Use the current vastai-training skill, live instance/credit/rate audit and one-instance guard. Keep the existing total ceiling; do not infer new funds from this document. Do not destroy anything.
- [ ] Set a fresh cost and wall-clock envelope with enough reserve for evaluations and verified local download. If the proposed full pilot cannot fit, reduce the experiment to a diagnostic smoke and report that limitation before calling it a learning comparison.
- [ ] Run B0 CUDA control and one-rollout B1 transfer calibration:32768 transitions. Preserve the calibrated checkpoint and clone it into E1/E2; both arms must start from identical calibrated weights.
- [ ] For each arm, run at most three paired rounds of four B1 rollouts and one navigation maintenance rollout:163840 transitions per round,491520 per arm. Start E1 first in odd rounds and E2 first in even rounds to reduce a simple order effect. One job runs at a time.
- [ ] Alternate the maintenance rollouts near/wide/near in both arms. Keep their critics and masks separate; use the same navigation entropy default0.001 in both arms so the only exploration difference is B1 entropy0.001 versus0.003.
- [ ] After each round, save latest/best-qualified state, run fast development panels, create videos for proposed promotion, download and verify locally before another round.
- [ ] Report 32768 calibration +983040 arm transitions separately from the parent's10027008 historical transitions; total new training budget1015808 if all planned rounds run. Evaluation transitions are separate and do not count as learning.
- [ ] Compare stage success, sampled/deterministic gap, movement retention, time-to-hit, reengagement, attack occupancy and credit cost. No claim about architecture size follows from two entropy arms of one seed.
- [ ] If a candidate passes fast checks, run the independent100-case confirmation panels. If neither does, report failure/diagnosis and the last safe artifact; do not extend automatically or skip to B2.

Acceptance: either a behaviorally qualified B1 checkpoint with preserved movement, or a bounded, reproducible negative result identifying the next discriminating test. Both are legitimate pilot outcomes; only the former advances.

## Task 6: Promote through the rest of the curriculum

- [ ] Implement/qualify B2 wider approach using original-game fixtures and the same action contract; gate wider navigation before B3's moving cross-platform opponents.
- [ ] Add B3 patrols with explicit movement-only scripts and safe turn points. Verify they genuinely move and cannot self-destruct as a default behavior. Do not reuse ClockworkAgent's attack-containing default sequence.
- [ ] Add B4 telegraphed retaliation with explicit cadence, range and cooldown. Verify positive avoidance/contact controls and that running away forever cannot pass.
- [ ] Qualify B5 heavy/recovery output introduction independently. Define recoverable starts through original-game scripted solutions and exclude impossible placements from the recovery denominator.
- [ ] Qualify B6 one weapon at a time, then respawn/multi-stock histories. Test actual pickups, disarms, unarmed fallback and full-start distances, not merely a weapon field in the observation.
- [ ] Repair/qualify frozen-policy opponent loading with the per-player adapter before B7. Controlled job self-play exclusions may be lifted only in the new validated path; do not alter the frozen reference or silently bypass its guard.
- [ ] Preserve every stage's scenario bank and best-qualified checkpoint. Reuse prior gates as retention tests; advance this lineage independently of other architecture arms.
- [ ] Treat website deployment and human playtesting as a separate authorized task with its own timing/action/latency tests and human evaluation receipt.

Acceptance: each stage adds one bounded skill/mechanic and demonstrates retention before expanding difficulty. Stage count and wall-clock time do not certify human readiness.

## Review and deliverable boundary

This turn produces only the curriculum design and this qualification/pilot plan. Validation now consists of document consistency, source/checkpoint evidence, independent critique and existing-code baseline tests. Future test cases above are unchecked because no new curriculum implementation or GPU experiment has run.

Planning receipt, 2026-09-07:

- [x] Parent checkpoint presence, SHA256, size, saved configuration and library version checked without model inference.
- [x] Curriculum source boundaries and original-game/native-runner limitations independently inspected.
- [x] Independent design critique completed; clarified relocation geometry and calibration-versus-mastery qualification.
- [x] Pilot arithmetic, stage coverage, Markdown fence balance, whitespace and placeholder checks passed.
- [x] Existing baseline suite passed: `PYTHONDONTWRITEBYTECODE=1 /private/tmp/ai2-baseline.XWm6Cd/venv/bin/python -m unittest discover -s tests -p 'test_combat*.py' -q` — 34 tests in10.302seconds. These are existing adapter/runner tests, not tests of the proposed curriculum.

No new trainer code, policy updates, GPU inference, instance lifecycle changes, deployment or new spend occurred in this planning task.
