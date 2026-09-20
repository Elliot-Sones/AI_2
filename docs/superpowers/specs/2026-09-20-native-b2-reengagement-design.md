# Native B2 Re-engagement Curriculum Design

Date: 2026-09-20

Status: approved in conversation for specification; implementation requires this written-spec review gate

Owner: Codex root lane

Implementation branch: `codex/throughput-qualification-20260920`

## Purpose

Advance the learned Bernoulli-button policy beyond one safe contact. The new
lesson must teach the actor to finish one interaction, deliberately pursue the
opponent again, and land a second distinct damaging attack without falling.

The existing `bridge_b1_contact-v1` lesson ends after one safe hit. Repeating it
cannot teach continued play. Imperfect wide navigation is therefore a retention
guardrail for this work, not a prerequisite that blocks advancement.

## Scope

This design adds one new, independently versioned native-GPU lesson:

`bridge_b2_reengagement-v1`

It includes:

- two distinct attributed light-attack hits;
- a clean encounter boundary after the first hit;
- measured renewed pursuit before the second hit;
- a stationary, non-retaliating target in both encounters;
- safe recovery and full-body floor support at completion;
- navigation rehearsal during training;
- deterministic and sampled evaluation in both learner slots;
- immutable evidence, preservation, and promotion receipts.

It does not include continuous target movement, retaliation, heavy attacks,
weapons, pickups, full-arena combat, self-play, or changes to the existing B1
lesson. Those are later curriculum stages, not hidden additions to B2.

## Parent policy

The initial B2 campaign resumes from the preserved `B_projected-050m`
checkpoint:

- checkpoint SHA-256:
  `d6e487c22f73913cd88a428683c7ca721e23b1f9cb5a3a561ad24d1f5bf410db`;
- lifetime learner-transition counter: `60,391,424`;
- action head: projected Bernoulli buttons;
- model: the existing 15.4-million-parameter navigation/contact policy;
- optimizer and model counters are preserved on resume.

This parent is selected because it is the strongest tested first-contact model:
20/20 deterministic and 16/20 sampled contact at its last development check.
Its imperfect wide navigation is recorded as a matched parent baseline and is
not described as mastery.

## Lesson state machine

### Encounter 1: first contact

1. Generate a seed-derived scenario balanced across learner slot and direction.
2. Spawn both complete bodies on the fixed floor with a horizontal gap sampled
   from `[3.1, 3.4]`, midpoint sampled from `[4.35, 4.65]`, and at least `0.2`
   units of full-body horizontal floor clearance.
   Every B2 gap and gap-closed measurement is the absolute horizontal distance
   between the two body centers, not the distance between body edges.
3. The learner controls the same B1 mask: movement, jump, dodge, and light
   attack. The target submits an all-zero action and cannot retaliate.
4. Credit the first contact only when a permitted light attack from the learner
   produces positive, attributed damage with a nonzero attack ID.
5. Multiple damage frames from the same attack ID count as one hit.
6. Wait until both players have left attack/stun states and all stored stun has
   resolved. Relocation is illegal before this clean boundary.

### Encounter boundary

1. Both players must be alive, grounded, and fully supported before relocation.
2. Keep the learner's complete physical and combat state unchanged.
3. Define safe anchors from the actual floor bounds and player half-width:
   `left + half_width + 0.2` and `right - half_width - 0.2`.
4. Compute both center gaps from the learner's post-step body center. Discard
   any anchor whose recreated target would have less than `0.2` clearance or a
   center gap less than `2.0`. Choose the remaining anchor with the larger
   center gap. On an exact tie, choose the right anchor for scenario direction
   `+1` and the left anchor for direction `-1`.
5. Recreate only the target at the selected anchor and at the same standing
   height used by B1 (`floor.b - 0.5`). Set its velocity to zero, standing
   state, and facing toward the learner. Preserve its stocks, current damage
   percentage, and cumulative damage-dealt/damage-taken counters. The learner
   is not recreated or modified.
6. Relocation requires a horizontal learner-target gap of at least `2.0`. If
   neither anchor qualifies, keep the clean encounter boundary pending while
   the 300-tick episode clock continues; do not recreate or teleport the target
   to an unsafe or trivial position. The ordinary tick cost still applies.
7. Record the exact pre/post target snapshots, relocation tick, initial second-
   encounter gap, learner position, target position, and direction of pursuit.

### Encounter 2: renewed pursuit and second contact

1. Renewed pursuit is credited once, within 60 ticks after relocation, only
   when both conditions hold:
   - the learner has closed at least `1.0` unit of the relocation gap;
   - the learner itself has progressed at least `1.0` unit toward the target.
2. The second contact must occur within 90 ticks after relocation.
3. It must use a different nonzero attack ID and produce new positive,
   attributed learner damage.
4. A held button, repeated collision frames, or lingering hitbox from the first
   encounter cannot satisfy the second-contact requirement.
5. Success is emitted only after the second attack resolves and both players
   are alive, grounded, and fully supported, with the learner retaining at
   least `0.2` horizontal floor clearance.

The total episode deadline remains 300 ticks. All task failures are true
terminals rather than time-limit truncations, so PPO must not bootstrap through
them.

### Tick and terminal ordering

One successful call to the raw game step advances the game and B2 episode clock
by exactly one tick. Each B2 step uses this order:

1. snapshot the pre-step state and apply the learner and all-zero target action;
2. run one raw game tick, then snapshot the post-step state and event stream;
3. validate state and evidence, with invalid evidence raising a hard run error
   before a transition can enter PPO;
4. attribute eligible damage and pursuit events for this tick;
5. resolve stock-loss failures, then clean-boundary support failure or success;
6. perform an eligible target relocation;
7. apply response, second-hit, and total deadlines in that order;
8. build the returned reward, observation, terminal flag, and diagnostics.

Learner stock loss has priority if both players lose stock on the same tick;
otherwise target stock loss is the terminal outcome. Positive event rewards
earned earlier or on the terminal tick remain additive with the selected
terminal penalty and ordinary tick cost. `unsafe_support` is considered only at
a clean attack/stun resolution boundary when both players are grounded: it is
terminal if either body is not fully supported or learner clearance is below
`0.2`. Ordinary airborne movement is not itself unsafe support.

If relocation happens after processing game tick `T`, `relocation_tick = T`.
The observation returned for tick `T` contains the relocated target, but no
learner action is applied against that target until game tick `T + 1`. Define
second-encounter elapsed time as `current_tick - relocation_tick`:

- pursuit can be credited on elapsed ticks 1 through 60 inclusive; after all
  tick-60 events are processed, absence of pursuit terminates that same step as
  `response_deadline`;
- the distinct second hit can be credited on elapsed ticks 1 through 90
  inclusive; after all tick-90 events are processed, absence of that hit
  terminates that same step as `second_hit_deadline`;
- tick 300 is processed normally, including any eligible success. If the
  episode remains ongoing after its tick-300 events, it terminates as
  `total_deadline`. Tick 301 can never execute.

## Rewards

The positive-return scale remains close to B1 while shifting credit toward the
new behavior:

| Event | Reward | Frequency |
| --- | ---: | --- |
| First distinct attributed hit | `+0.5` | once |
| Qualified renewed pursuit | `+0.5` | once |
| Second distinct attributed hit | `+1.0` | once |
| Safe sequence completion | `+2.0` | once |
| Own stock loss | `-5.0` | terminal |
| Unsafe support | `-2.0` | terminal |
| Opponent stock loss or self-loss | `-1.0` | terminal |
| Response, second-hit, or total deadline failure | `-1.0` | terminal |
| Ordinary elapsed tick | `-0.001` | each tick |

Opponent stock loss, unsafe support, invalid evidence, and opponent self-loss
are failures and never successes. Invalid evidence fails the run rather than
becoming a trainable transition. The episode record reports every reward
component separately. These penalties ensure that collecting a partial hit and
then leaving the floor has negative return rather than remaining a profitable
shortcut.

## Native architecture

B2 is added beside B1 rather than changing B1 in place.

- A dedicated native episode state owns the two-encounter state machine,
  scenario, reward totals, snapshots, and diagnostics.
- Host and CUDA lane implementations expose the same tensor-owned interface as
  the current contact lanes.
- The lane step runs entirely on the selected PyTorch CUDA stream. Actions are
  read from collector-owned tensors and observations, rewards, and terminal
  flags are written back without per-step host copies.
- Completed episode records are drained once per rollout, preserving the
  current lossless diagnostic contract.
- The Python adapter identifies the new task/version explicitly and fails
  closed if its action mask, task label, version, or diagnostic schema differs.
- A separately versioned Python reference environment implements the same B2
  contract against the frozen original game. It is the behavioral oracle for
  differential tests and original-game evaluation; the older unqualified B1
  re-engagement prototype is evidence to reuse, not an oracle to copy blindly.
- The existing B1 ABI and behavior remain unchanged and must continue to pass
  exact regression tests.

The first implementation may share low-level game helpers with B1, but B2 must
have distinct version identifiers, scenario validation, outcome codes, reward
term count, diagnostics, and evaluation metrics.

### Concrete module and runner boundary

B1 remains a frozen regression surface. In particular,
`benchmark_native_contact_training.py`, `native_contact_episode.py`,
`native_contact_lane_vec_env.py`, and the existing `native/contact_*` files are
not repurposed for B2. The 64-lane `ContactBenchmarkConfig` remains the B1
qualification comparison and is not the B2 campaign runner.

B2 adds these sibling boundaries:

- `reengagement_curriculum.py`: original-Python-game reference environment;
- `evaluate_reengagement_curriculum.py`: trace-first B2 evaluator and pinned
  development/confirmation bank generator;
- `native_reengagement_episode.py`: ctypes CPU reference and projections;
- `native_reengagement_lane_vec_env.py`: finite scenario schedule and tensor
  environment adapter;
- `native/reengagement_episode.h` plus `native/reengagement_episode.c`: shared
  state machine and CPU entry points;
- `native/reengagement_lanes.h`, `native/reengagement_lanes.c`, and
  `native/reengagement_lanes.cu`: CPU and CUDA lane runtimes;
- `qualify_native_reengagement.py`: original-game/native-CPU/CUDA parity proof;
- `train_reengagement_campaign_block.py`: immutable 4,096-lane mixed block
  runner, modeled on `train_retention_campaign_block.py`, that constructs the
  B2 environment plus the existing `native_navigation_curriculum_env.py`
  rehearsal environment and executes one complete `R4E1N` schedule while
  preserving the same actor, both critics, optimizer, and counters;
- `continue_reengagement_campaign.py`: `R4E1N` controller, matched evaluation,
  retention alarms, stop rules, and preservation.

`native_curriculum_ppo.py` keeps task ID `combat == 1`, the existing combat
critic, and the existing combat action mask. It gains an explicit expected
combat-version contract. Existing call sites default to
`bridge_b1_contact-v1`; B2 callers must pass `bridge_b2_reengagement-v1`.
Environment validation, task switching, and every drained episode-info row
must match that expected version or fail closed. The module-level B1 default is
never globally changed to B2.

`native_curriculum_runner.py` adds an explicit `native_reengagement` block and
a dispatch table. `native_contact` continues to invoke the old B1 runner with
the same accepted arguments and guards; `native_reengagement` invokes
`train_reengagement_campaign_block.py`. Neither block can override conductor-
owned checkpoint or output paths. The B2 block requires the explicit
`B_projected-050m` SHA and its verified preservation acknowledgement, rather
than the older calibrated-parent constant. Every new source boundary is added
to qualification/source hashes.

## Required diagnostics

Every completed B2 episode records at least:

- seed, scenario ID, learner slot, direction, and episode length;
- initial positions, gap, midpoint, stocks, and clearance;
- first and second attack IDs and hit ticks;
- attributed damage events and cumulative damage dealt/taken;
- attack requests, attack starts, attack-state occupancy, and button holds;
- relocation tick and exact pre/post target snapshots;
- relocation gap, gap closed, learner progress, and pursuit-credit tick;
- minimum clearance, unsupported ticks, falls, and stock changes for both slots;
- reward components and terminal reason;
- whether first contact, pursuit, second contact, and safe resolution were each
  achieved.

Failure summaries distinguish at least: no approach, no attack request,
requested but no attack started, attack started but missed, first hit only,
failed renewed pursuit, pursued but missed second hit, unsafe support, own stock
loss, opponent stock loss, and deadline.

## Evaluation contract

### Pinned case banks

The retained B1/navigation evaluator is pinned to its existing case generator:

| Panel | Development seeds per mode | Confirmation seeds per mode |
| --- | --- | --- |
| B1 contact | `131000..131019` | `141000..141099` |
| Nearby navigation | `132000..132019` | `142000..142099` |
| Wide navigation | `133000..133019` | `143000..143099` |

For each mode and zero-based case index `i`, player slot is `i % 2`, direction
is `-1` for index pairs 0/1, 4/5, and so on and `+1` for pairs 2/3, 6/7, and so
on. Deterministic and sampled modes use the same scenario seeds rather than two
different scenario sets. Case IDs remain
`{task}-{mode}-p{slot}-s{seed}`.

B2 uses the same slot/direction formula but a disjoint versioned bank:

- development: seeds `134000..134019` per mode;
- confirmation: seeds `144000..144099` per mode;
- case IDs: `reengagement-{mode}-p{slot}-s{seed}`;
- metric and bank generator version: `bridge_b2_reengagement-v1` and
  `b2-reengagement-bank-v1`.

The B2 finite training schedule must reject and redraw any episode seed in
either B2 evaluation range, record every scheduled seed in `scenarios.json`,
and assert zero overlap before CUDA context creation. Development seeds are
used for repeated measurement but never training; confirmation seeds remain
untouched until the two-checkpoint development gate passes.

The matched B1 parent reference artifacts are pinned under
`baselines/staged-training-20260920`: contact and nearby results under
`eval-B_projected-050m/`, and wide results under
`eval-wide-B_projected-050m/`. Their scenario-bank SHA-256 values are,
respectively, `6ed0da7d7172027987519d545480384c625a0a38e2fbadafef5138fc509d6dea`,
`70eb82f3e7adf561329418f8e61914ad622b848670a2b9f7ebdf659f5d2c46a9`, and
`277f725e4b964d9ecf1f2d300c1e264ccd639d39887b00a0f8f9a3b4f087f7b4`.
These historical artifacts are context only: the parent is rerun on every
pinned development bank before the first update, and that new immutable result
is the operative comparison baseline.

### Development checks

Each checkpoint is evaluated on fixed, held-out, disjoint seeds:

- 20 deterministic B2 episodes;
- 20 sampled B2 episodes;
- 10 episodes per learner slot in each mode;
- balanced initial direction and scenario strata;
- reconstructed success from trace evidence, never from the environment's
  success label alone.

The authoritative development and confirmation panels run through the frozen
original Python game reference. Native CPU/CUDA panels establish parity and
debug failures but cannot by themselves promote a checkpoint.

Report full-sequence success as well as first-hit, pursuit, second-hit, safe-
resolution, fall, timeout, damage, and timing rates. Deterministic and sampled
results and both player slots remain separate.

### Promotion

B2 development mastery requires two consecutive, distinct checkpoints with
increasing lifetime counters. Each checkpoint must achieve, in both modes:

- at least 18/20 reconstructable full-sequence successes;
- no more than one learner fall;
- no invalid or ambiguous success evidence.

Only then consume the untouched confirmation bank. Confirmation requires, per
mode:

- at least 90/100 full-sequence successes;
- no more than five learner falls;
- at least 40/50 successes from each learner slot.

### Retention guardrails

Run the existing B1 contact, nearby-navigation, and wide-navigation development
panels after every B2 training cycle, on the exact parent comparison cases.

Before the first B2 update, rerun and preserve the parent on the exact B2 and
retention development banks. Historical scores provide context but do not
substitute for this matched baseline.

- Pause further updates if any mode loses more than two successes out of 20 or
  adds more than two falls relative to its matched parent panel.
- Promotion confirmation uses the existing parent-relative rule: no more than
  a five-percentage-point success drop and no more than a 5% fall rate on the
  corresponding 100-case confirmation panels.
- Wide navigation is judged relative to the imperfect parent baseline; B2 is
  not required to retroactively make the parent perfect.

## Training campaign

The initial campaign schedule is `R4E1N`, where `E` means a complete B2
re-engagement rollout:

1. four B2 re-engagement rollouts;
2. one native navigation-rehearsal rollout.

At 4,096 lanes and 512 steps, one cycle is exactly `10,485,760` learner
transitions. Use the already qualified PPO geometry unless a separate controlled
experiment changes it:

- batch size `16,384`;
- four PPO epochs;
- learning rate `3e-5`;
- clip range `0.1`;
- target KL `0.01`;
- entropy coefficient `0.001`;
- FP32;
- no CPU training fallback.

Evaluate and preserve after every cycle. Continue toward five cycles
(`52,428,800` added transitions) only while B2 improves and retention alarms
remain clear. Stop early on mastery, numerical failure, invalid evidence,
severe retention regression, or a three-check plateau that shows neither
success nor time-to-reengagement improvement.

The existing `$12` campaign ceiling remains a ceiling, not a target. Before
renting, run a fresh live Vast inventory and own at most one labeled instance.
Every cloud run must have W&B tracking (online when available, otherwise a
preserved offline run), GPU/CPU monitoring, immutable commands, and immediate
local artifact retrieval with hash verification. Destroy the owned instance
only after verified closeout.

## Test and qualification plan

Implementation follows test-driven development.

1. Write reference behavior tests first and observe them fail because B2 does
   not exist.
2. Add CPU native tests for scenario validity, reward/outcome boundaries,
   distinct attack IDs, pursuit thresholds, exact deadlines, terminal semantics,
   private RNG use, and serialization.
3. Prove shortcut controllers fail: idle, held light, one hit then idle, repeated
   frames from one attack, movement without a second hit, walkoff, and opponent
   self-loss.
4. Prove a scripted hit-resolve-pursue-hit-resolve controller succeeds across
   both slots, both directions, and boundary scenarios.
5. Run differential CPU-reference versus native-CPU traces for exact gameplay,
   reward, outcome, observation, diagnostic, and reset agreement.
6. Run native CPU versus CUDA lane parity on the rented GPU across scripted,
   adversarial, and seeded random action streams.
7. Run the real tensor collector and learner-update exactness tests with the B2
   adapter before learning experiments.
8. Run a short CUDA learning smoke, then the first full `R4E1N` cycle.
9. Preserve source, build information, qualification receipts, checkpoint,
   traces, logs, monitoring, W&B data, and evaluations before any downstream
   cycle consumes the checkpoint.

Qualification fails closed. A passing unit suite on the Mac is not CUDA
qualification, and environment throughput is not training throughput.

## Error handling and immutability

- Invalid scenarios, actions, state, support, diagnostics, or task/version masks
  produce explicit errors, never approximate results.
- An episode cannot remain ongoing after processing tick 300, and tick 301
  cannot execute.
- A success outcome without reconstructable two-hit evidence invalidates the
  evaluation block.
- Training and evaluation output directories are immutable and uniquely named.
  Interrupted jobs are recovered from verified artifacts or retried under a new
  name; they are never overwritten.
- A checkpoint cannot be used downstream until its inventory and preservation
  acknowledgement are verified locally.

## Subsequent stages

After B2 re-engagement passes its development, confirmation, and retention
gates:

1. `bridge_b3_moving_passive-v1`: pursue targets moving both directions at
   varied speeds and starting positions, still without attacks.
2. `bridge_b4_limited_retaliation-v1`: face a qualified restricted attacker and
   learn damage exchange, interruption recovery, and survival.
3. Full-arena opponent diversity and verified self-play, with held-out scripted
   and archived-policy identities.

Each stage receives its own lesson version, evaluator, qualification, and
retention gates. Passing B2 is not described as full fighting mastery.
