# Native Navigation-Retention Curriculum Design

Date: 2026-09-20

## Purpose

Train the current binary-button contact policy without silently erasing its
navigation ability. The first deliverable is the strongest evidence-backed B1
safe-contact candidate, not a claim of full fighting ability. Re-engagement,
moving targets, retaliation, full-arena combat, and opponent diversity remain
separate, subsequent curriculum projects because the current native CUDA task
does not implement them.

The practical campaign target is 50 million learner transitions per viable
recipe. Ten million is an evaluation boundary inside that campaign, not the
intended stopping point. A candidate continues automatically while learning
evidence improves, artifact integrity holds, navigation does not regress past
the declared alarm, and the remaining budget covers preservation and teardown.

## Starting evidence

- `B_projected-040m` is the common experimental parent. Checkpoint SHA-256:
  `6d7b83a287b1a29bf8741295d769178764fd8b4beaf101a678b62cdc41bd73f0`.
- The existing contact-only continuation from B40 to B50 is the historical
  control. B50 checkpoint SHA-256:
  `d6e487c22f73913cd88a428683c7ca721e23b1f9cb5a3a561ad24d1f5bf410db`.
- B50 reached 20/20 deterministic and 16/20 sampled B1 safe completions.
  Nearby navigation was 19/20 in each mode. Wide navigation was 13/20
  deterministic and 9/20 sampled, versus B40 at 13/20 in both modes.
- One native rollout contains `4,096 * 512 = 2,097,152` learner transitions.
  Five rollouts contain 10,485,760 learner transitions.

The prior action-head comparison is a practical transfer comparison, not a
pure causal test: action representation and optimizer initialization differed.
The new comparison fixes the parent, policy/action head, optimizer checkpoint,
rollout geometry, learning rate, evaluator, and development banks.

## Selected architecture

Use **alternating homogeneous native rollouts**. Every rollout contains only
one task: either B1 contact or navigation. The same actor is updated by both
tasks, while the existing navigation and combat value heads remain separate.
Task and action-mask changes occur only after the previous rollout has been
fully consumed by PPO.

This fits the current policy contract: task selection and value-head selection
are scalar per rollout. A same-rollout mixture of navigation and contact lanes
would require per-sample task IDs, per-lane action masks, per-sample value-head
selection, and a redesigned buffer. That larger design is excluded until the
alternating approach is measured and shown insufficient.

The implementation consists of four bounded units:

1. A native navigation curriculum adapter exposing the existing CUDA navigation
   environment through the same task metadata contract as contact.
2. A task-general native tensor rollout buffer that accepts navigation or
   contact but rejects mixing them within one rollout.
3. A task-general native curriculum PPO collector that switches the environment,
   control mask, and value head only at rollout boundaries.
4. A campaign runner/controller that executes an immutable schedule, evaluates
   every cycle, preserves each block locally, and stops or continues from
   explicit gates.

Both environments retain their own reward and terminal semantics. Separate
critics prevent the contact and navigation return scales from being forced into
one value target. Reward definitions are not changed in this experiment.

## Experiment matrix

All new candidates resume independently from the exact B40 checkpoint and use
the same CUDA device, 4,096 lanes, 512 steps, batch size 16,384, four PPO passes,
FP32, learning rate `3e-5`, policy architecture, and disjoint training seeds.

| Candidate | Schedule / observed block | Contact transitions | Navigation transitions | Interpretation |
| --- | --- | ---: | ---: | --- |
| Historical control | 4 contact, 0 navigation | 8,388,608 | 0 | Existing B40 to B50 evidence |
| R4C1N | 4 contact, 1 navigation | 8,388,608 | 2,097,152 | Same contact exposure plus light rehearsal |
| R3C2N | 3 contact, 2 navigation | 6,291,456 | 4,194,304 | Fixed total budget with stronger rehearsal |

R4C1N is the closest matched intervention because its contact exposure equals
the historical control. R3C2N is a practical fixed-total-transition competitor,
not a pure rehearsal-ratio estimate because it receives less contact data. Every
report must preserve that distinction.

After the first 10,485,760-transition cycle, select candidates by the declared
multimetric gates; do not collapse contact and navigation into a single score.
Continue the strongest eligible recipe in additional immutable cycles toward
approximately 50 million new transitions. Preserve checkpoints at every cycle.

## Evaluation and automatic gates

Every checkpoint is evaluated in the original Python game, independently from
the native training engine, in deterministic and sampled modes. The existing
20-case development banks are model-selection alarms, not independent repeated
trials. The reserved 100-case confirmation banks remain untouched until the
promotion contract is satisfied.

### Integrity gate

Stop the affected candidate on any of the following:

- non-finite model, optimizer, loss, gradient, reward, or observation value;
- wrong checkpoint/source/scenario hash or broken resume chain;
- missing PPO pass/update accounting, unexpected KL behavior, scenario reuse,
  evaluator failure, or artifact-preservation failure;
- task switch while a rollout is partially collected or not yet trained.

### Ten-million diagnostic gate

A candidate is eligible for continuation only if:

- deterministic contact is at least 18/20;
- sampled contact is at least 14/20 and does not fall below its B40 baseline;
- nearby navigation is at least 19/20 in each mode with at most one fall;
- wide navigation is at least 13/20 in each mode and does not exceed the B40
  fall count in either mode;
- no measured regression is hidden by opponent self-loss, timeout, or ambiguous
  terminal attribution.

If both candidates are eligible, prefer the Pareto-superior candidate. If each
wins a different metric, run one additional matched cycle or comparison rather
than inventing an aggregate score. If neither is eligible, retain B40/B50 and
revise the intervention before further scaling.

### Fifty-million continuation gate

Continue cycle by cycle while the candidate remains eligible and at least one
unmastered target improves. Stop an individual lineage after two consecutive
cycles with no improvement, or immediately for severe retention/integrity
failure. A stop means the recipe needs review; it does not prove the policy can
never learn with a different curriculum.

### Promotion gate

B1 promotion requires two consecutive increasing-step development checkpoints,
each with at least 18/20 safe completions and no more than one fall in **both**
deterministic and sampled modes. Then run the untouched 100-case confirmation:
at least 90/100 and at most five falls per mode, with each player slot at least
40/50. Navigation confirmation must be matched to a fresh unchanged-parent
baseline using the exact same predicate, source, and cases.

The historical `active_sequence_v3` navigation metric and the requested
body-overlap predicate are different contracts. Body-overlap and targeted
moving-platform recovery require their own qualification and parent baselines;
they must not inherit old headline scores silently.

## Correctness tests before cloud learning

Implementation follows test-first development. Required evidence:

1. Task-buffer tests prove one homogeneous navigation or contact rollout is
   accepted and any within-rollout task/mask change fails closed.
2. Environment-switch tests prove a switch is rejected before PPO consumes the
   current rollout and accepted afterward.
3. Navigation-adapter tests prove observation/action ABI, task metadata, masks,
   episode records, truncation, autoreset, and device residency.
4. Resume tests prove actor, both critics, optimizer tensors, masks, counters,
   and next update survive every block boundary.
5. CPU reference and CUDA parity tests compare fixed scenarios/actions for each
   supported task. Existing B1 behavior remains unchanged.
6. Controller tests prove immutable job names, uncertain-launch adoption,
   deadline/budget enforcement, immediate download, checksum verification,
   and no downstream run before a preservation acknowledgement.
7. A short CUDA learner smoke proves both task rollouts update the intended
   shared actor and only the selected critic, with finite metrics and complete
   optimizer-step accounting.

## Cloud execution and evidence

- Existing total authorization: $5.00. Observed prior account-credit decrease:
  about $0.61825. Treat the remaining amount as a ceiling, not a spending target.
- Reserve at least $0.75 for transfers and closeout uncertainty.
- Inventory Vast.ai live immediately before creation. Own at most one instance.
- Label the instance with project, task, owner, and date; record ID and hourly
  price locally immediately.
- Use offline W&B monitoring because credential transfer/sync is not authorized.
- Download and hash-verify every completed block before its checkpoint can feed
  another block. Preserve full episode traces, source/runtime manifests, job
  receipts, utilization records, and representative success/failure videos.
- When no longer needed, stop monitoring cleanly, verify the full local mirror,
  destroy the owned instance, and confirm a fresh zero-unexplained-instance
  inventory.

## Boundaries and subsequent curriculum

This design produces a navigation-retaining B1 candidate and a trustworthy
decision about scaling it. It cannot, by itself, produce a human-playable fighter.
The current native B1 episode ends after one safe hit against a passive target.

After B1 promotion, subsequent designs and qualification gates proceed in order:

1. B2 re-engagement: two distinct damaging attack IDs separated by renewed
   pursuit and safe resolution.
2. Moving passive targets with balanced direction, speed, slot, and spawn strata.
3. Restricted retaliation with damage exchange, survival, recovery, and attack
   interruption telemetry.
4. Full-arena matches against fixed, hashed scripted opponents.
5. Verified archived-policy opponents/self-play, held-out opponent identities,
   full regression over earlier lessons, representative videos, and human play.

Deterministic policy execution is the default deployment candidate because it
is reproducible. Sampled play may be offered as a varied mode only after it
independently passes the same combat, retention, and safety gates.

No step count alone establishes a good agent. The final claim requires a playable
checkpoint, attributed full-match performance against held-out opponents, earlier
lesson retention, failure videos, and human playtesting.
