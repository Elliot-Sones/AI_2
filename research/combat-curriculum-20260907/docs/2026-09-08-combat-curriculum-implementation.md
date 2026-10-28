# Combat curriculum implementation and negative qualification

Status: implemented infrastructure, locally verified; original B1 lesson unqualified.
No actual model checkpoint inference, policy updates, CUDA validation, new W&B
run, instance start/stop/destruction, or deployment occurred during this task.

## Implemented scope

- `combat_curriculum.py`: original-game passive opponent and two-encounter
  prototype, light-only action extension, finite task deadlines, attributed hit
  events, private random streams, original observation contract and full-body
  floor support. The prototype exports `B1_TRAINING_QUALIFIED=False`.
- `curriculum_ppo.py`: shared ten-output actor, selective light-head
  initialization, task-specific critics, masked Gaussian PPO accounting,
  protected calibration parameters, optimizer/save/reload checks. Synthetic
  tests are not evidence of CUDA correctness or model learning.
- `train_combat_curriculum.py`: finite CUDA-only calibration and paired E1/E2
  blocks, fixed PPO geometry, separate arm lineage, per-rollout checkpoints,
  raw/executed actions, timings and metrics. It refuses the unqualified lesson
  before any CUDA/model work. Paired environment seeds are restored after SB3
  load, which otherwise overwrites the subprocess reset seeds.
- `curriculum_navigation.py`: unchanged historical navigation retention,
  separate versioned body-overlap mode, actual player-slot mapping, private RNG.
- `curriculum_telemetry.py`, `curriculum_evaluation.py` and
  `evaluate_combat_curriculum.py`: event reconstruction, diagnostic reports,
  deterministic/sampled development panels, distinct confirmation cases,
  source/checkpoint provenance, retention and fail-closed promotion checks.
- `replay_combat_curriculum.py`: policy-free game replay, bitwise observations,
  state/event/identity comparison, immutable video and verification receipt.
- `curriculum_cloud.py`, `curriculum_monitor.py`: sanitized read-only provider
  audit, budget/identity gates, immutable checksum verification, local scalar
  W&B bridge. Monitor behavior is fake-SDK tested, not live authenticated-run
  validation. No credentials, trajectories, media or checkpoints are sent by
  this scalar bridge.
- `qualify_curriculum_cuda.py`: transfer/calibration/export verification entry
  points prepared for the actual CUDA checkpoint; CPU synthetic tests only.

## What the logs can establish

Each combat tick includes both players' pre/post x/y, velocity, facing, damage,
stocks, grounded/full-support state, edge clearance, attack/state identifiers,
observations, raw actions, executed controls and reward terms. Attack requests,
starts, active hitboxes, collision/vulnerability, damage and attack ends are
separate events. One attack with multiple damage frames is counted once as a
connected attack. Unknown damage attribution stays unknown.

Aggregates identify failure to approach, no attack request, a request that never
started, a started attack that missed, failure to reengage, sustained attack
movement lock, unsafe support, deadline and stock loss. Their evidence includes
the exact ticks and positions. These are observed behavior labels, not claims
about the network's internal intent. A later model run is still needed to learn
what the actual model is good at.

## Qualification result: do not train this lesson yet

Measured 2,604 ground fixtures and 1,698 aerial fixtures in the frozen original
game. These are finite test grids, not a proof of a global attack-reach bound.
Four scripted cases cover both player slots and both directions. All four are
rejected at ticks 30, 31, 31 and 32 as `unsafe_floor_support`.

The two-hit reset rule requires the target at least one unit beyond the measured
attack envelope. On the original five-unit platforms, the scripted learner
backs toward the edge to make room. Its body overhangs by 0.128–0.228 world units
while the game's ordinary `grounded` flag can still be true. A former apparent
completion therefore was not a safe positive control. No completion reward is
now issued. No claim is made that every imaginable two-hit controller is
impossible; this drill lacks a qualified safe positive controller.

Example from the scripted slot-0/leftward trace (not a trained model):

| Tick | Simulated time | Recorded event |
|---|---|---|
| 9 | 0.30 s | One down-light attack requested and started |
| 15 | 0.50 s | Actual collision and 8 damage; one connected attack |
| 17 | 0.57 s | Attack ended; target still resolving stun |
| 30 | 1.00 s | Learner x=2.30665, y=0.35000, grounded=true but full support=false; task fails |

The recommended design change is to split the first lesson: approach, land one
real light attack and finish safely; qualify reengagement as a separate next
lesson. A user choice between this split and a redesigned two-hit lesson is
pending. Do not flip the qualification flag or silently weaken evaluation.

## Verification and preservation

Fresh root verification: 142 curriculum tests passed in 28.511 s; 23 existing
adapter tests passed in 0.752 s; 11 existing comparison tests passed in 0.002 s.
Total: 176 passing tests. Focused Ruff passed. Final source is preserved in the
isolated `codex/combat-curriculum-20260907` branch, not merged into main.
Independent PPO and replay reviews passed after identified issues were fixed.

Durable local receipts are in
`/Users/elliot18/Projects/ml-research/AI_2/baselines/combat-curriculum-session13/`.
Use `local-qualification-v2/`: its inventory SHA256 is
`4f2110f4a80b8d59e0b33916793a67997d9a8ab6caa0abdbf85c59afb6a5f566`,
verified locally. The first `local-qualification/` remains immutable for audit
but is superseded: its action-source label incorrectly said Gaussian sample
for a scripted controller. V2 fixes the label; the physical outcomes agree.
The real policy has never been loaded for these scripted qualifications.

Known unverified work: actual GPU transfer/calibration/reload, throughput with
full tracing, fresh parent behavior, learning/retention, live W&B delivery,
remote interruption/retrieval under load, and later B2–B7 stages. Inactive
optimizer moments are cleared by the protection mechanism; light-head momentum
does not carry through navigation maintenance. Measure this tradeoff rather
than presenting it as a neutral performance optimization.

Next execution requires a qualified lesson. Then refresh provider budget,
verify source/checkpoint hashes, run CUDA qualification and the finite pilot,
evaluate both action modes and sides, preserve each block locally before the
next, and advance only through behavioral and retention gates.
