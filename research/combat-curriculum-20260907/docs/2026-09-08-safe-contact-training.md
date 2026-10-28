# Approved safe-contact split and finite CUDA pilot

User approved the prior recommendation on 2026-09-08: implement and train
approach -> one actual light hit -> safe finish; use the logging infrastructure
to evaluate it. Reengagement is a later separately qualified lesson. This
updates the B1 objective only, not the preserved parent, budget or PPO geometry.

## Contact lesson contract

- Add version `bridge_b1_contact-v1`, leaving `bridge_b1-v1` and its rejected
  two-hit qualification untouched for regression/replay.
- Original frozen physics, raw64/stack256, ten output dimensions, same seven
  active controls, passive real opponent. Never freeze the target or teleport
  either character during a live episode; no relocation in this lesson.
- Seeded starts on the original static five-unit floor. Both player slots and
  approach directions, initial separation sampled in [3.1,3.4], midpoint
  jitter in [-0.15,0.15] from floor center, with >=0.2 full-body horizontal
  clearance for both players. Assert geometry instead of silently clipping.
- Success requires one attributed positive-damage light attack, >=0.25 units
  of learner progress toward the initial target before that first hit, attack
  and stun resolved, both players alive and fully supported, and learner floor
  clearance >=0.2. This is encounter completion, not combat mastery.
- Deadline300ticks/10simulated seconds. Stock loss takes priority. One contact
  credit maximum(+1), safe completion+2, tick cost-0.001, own stock loss-5,
  timeout-1. No survival/idle reward and no second-hit/relocation bonus.
- Qualification: scripted approach/release/down-light/settle succeeds across
  both slots, both directions, and multiple jitter seeds; idle, stationary
  held-light and walking off do not pass. Preserve actual damage trajectories,
  random-stream isolation, action masks, terminal semantics and trace parity.

## Implementation and validation sequence

1. Environment/tests: additive contact lesson and bounded scripted qualification.
2. Telemetry/gates/tests: reconstruct contact success separately by version;
   preserve two-hit reconstruction and provenance. No relocation or unqualified
   geometry can pass contact validation. All existing diagnostics remain.
3. Runner/evaluator/replay/tests: explicitly select contact version and require
   its qualification receipt; preserve CUDA-only actual-model operations,
   parent/arm lineage, fixed scheduling, stage-specific metric version and
   source hashes. Exact replay supports either lesson by recorded version.
4. Independent spec and quality review, complete local regression suite, source
   archive and scripted trace receipts before the GPU is started.
5. Fresh one-instance/budget audit; reuse50149729. Start the bounded guard and
   scalar W&B monitors, verify CUDA operation and remote frozen-source hashes.
   Qualify actual parent transfer/calibration and obtain fresh navigation
   controls on CUDA. No actual policy inference on local CPU.
6. Run the previously approved finite schedule:32768 calibration; up to three
   paired E1/E2 rounds of163840 each (four contact rollouts plus one navigation
   maintenance rollout), total at most1015808 new transitions. Same seed and
   scenario schedule, E1 entropy0.001/E2 entropy0.003, alternating order.
7. Evaluate deterministic and sampled behavior on balanced fresh development
   scenarios, compare navigation to matched unchanged-parent controls, inspect
   exact approach/hit/failure traces, and render representative/worst cases.
   Promotion needs two distinct fast checks and independent confirmation plus
   retention. A failed pilot is a measured result, never permission to skip ahead.
8. Download/checksum each completed block before another block. Preserve full
   logs, source, checkpoints and videos locally. Close monitors and stop compute
   at completion/deadline with saved receipts; do not destroy the instance.

Known limits: native GPU lanes implement navigation, not this original-game
combat task; combat environments run on host CPUs while policy/update run on
CUDA. Full tracing overhead and end-to-end throughput must be measured.
No human-opponent strength claim follows from this first contact lesson.
