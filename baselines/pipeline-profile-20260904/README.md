# RTX 4090 PPO pipeline diagnostic — 2026-09-04 EDT

## Conclusion and evidence limits

This run identifies concrete costs beyond "choose an action": environment
execution and coordination, observation construction, rollout storage, and
training-batch preparation. A game-only C rewrite would target environment-side
work, not the existing PyTorch policy or SB3 rollout-buffer sampling.

The retrieved 16-environment detailed profile has consistent counts and timing
accounting. **It is a diagnostic, not a stable timing baseline.** Its learning
updates take 0.938 seconds, whereas nearby comparison runs take 3.817–4.412
seconds for the same number of updates. The cause of this discrepancy is not
established. Do not extrapolate a C speedup or universal percentage allocation
from this one profile.

## Workload and provenance

- Retained Vast instance: `49902188`, verified RTX 4090, 24 GB VRAM.
- Game/worker code executes on CPU; policy/update device is CUDA.
- Cached-animation source: `/workspace/AI_2/animation-cache-20260904`.
- Fresh MLP `[512, 512, 256]`; navigation phase `0d`; no resumed checkpoint.
- 16 environments, 512 vector steps/rollout = 8,192 transitions/rollout.
- Batch size 8,192; 8 epochs/update; seed 42; Torch threads 1.
- One 8,192-transition warmup, then two measured rollouts = 16,384 transitions.
- Frame stack 4; existing opponent-history wrapper retained; headless execution.
- Versions and source hashes are in the detailed JSON's `result.source`.
  Local source-file hashes were checked against that metadata. Remote uploads
  exclude `.git`, so absent git metadata is not evidence of a clean checkout.
- This run only adds isolated diagnostic files under this directory. It does
  not implement a C port or change production training/gameplay.

Primary artifact:
`artifacts/natural_16env/pipeline_profile.json`

SHA-256:
`71516cea7648405c14dfc0c36682d321fc792cec807f0e8838f6f14fa3b39b94`

## Measured parent-process breakdown

The measured window took **9.426802 s**, or **1,738.02 aggregate transitions/s**.
There were 1,024 rounds in which all 16 environments advanced once. A round
therefore costs 9.206 ms on average, including amortized learning updates.
Dividing by 16 gives 0.5754 ms/transition of aggregate throughput cost, NOT an
individual environment's step latency.

All stage rows below are exclusive: nested observation conversion is removed
from the policy-forward row. Seconds add to the measured window.

| Operation | Total seconds | ms per 16-environment round | Share of this diagnostic |
|---|---:|---:|---:|
| Convert/send observations to tensors | 0.095964 | 0.093714 | 1.02% |
| Policy forward, distribution construction, action sampling | 1.818785 | 1.776157 | 19.29% |
| Return actions from GPU to CPU | 0.073429 | 0.071708 | 0.78% |
| Clip actions to allowed bounds | 0.052624 | 0.051391 | 0.56% |
| Environment send/wait/receive and vector/frame-stack work | 5.634554 | 5.502495 | 59.77% |
| Store observations/actions/rewards/values/log probabilities | 0.586591 | 0.572843 | 6.22% |
| Callback updates and calls | 0.044535 | 0.043491 | 0.47% |
| Episode/done checks in rollout loop | 0.006491 | 0.006339 | 0.07% |
| Compute returns and advantages at rollout end | 0.015174 | 0.014818 | 0.16% |
| Final value estimate | 0.000874 | 0.000853 | 0.01% |
| Rollout start/end callbacks | 0.000062 | 0.000061 | <0.01% |
| PPO learning updates, amortized | 0.937602 | 0.915627 | 9.95% |
| Not individually attributed | 0.160116 | 0.156364 | 1.70% |

Learning is not performed once per round. The last-but-one row represents two
updates, averaging **468.801 ms per update**, after each 8,192-transition rollout.

The remaining 0.160116 s can be located, but not fully assigned to functions:

- 0.103218 s inside rollout collection but outside individual stage timers.
- 0.056898 s outside rollout collection and timed learning updates.

These include instrumentation and uninstrumented control/setup work; their
individual contributions were not measured. They are not claimed as pure
Python, GPU, or C-rewrite opportunity.

There is no per-stage CUDA synchronization. These are CPU-observed elapsed
stages: policy forward includes launch/orchestration, and GPU-to-CPU copies can
wait for outstanding GPU work. **No kernel-only GPU trace was captured.**

## What is inside the environment interval?

Every one of the 16,384 worker-step intervals falls inside its matching parent
environment round. Aligning those timestamps gives the following nonoverlapping
partition of the 5.634554-second parent interval:

| Timestamp interval | Total seconds | ms per round |
|---|---:|---:|
| Parent start to first worker starting | 0.182980 | 0.178691 |
| First worker starting to last worker's step returning | 4.219529 | 4.120634 |
| Last worker step return to parent receiving complete result | 1.232046 | 1.203169 |

This is **not a pure simulation/IPC split**. The middle interval includes worker
start staggering and OS scheduling. The tail can include automatic resets,
serialization, pipe transfer, result collection, and frame stacking. Worker
step timestamps end before SubprocVecEnv's automatic reset. Transport can also
overlap other workers' computation.

Parent environment-round latency: mean 5.50 ms, median 4.76 ms, p95 10.90 ms,
p99 30.80 ms, maximum 62.97 ms. The two rollout durations also varied markedly:
2.846 s and 5.586 s. These are further reasons not to present one stable latency.

## Concrete worker-side costs

These are summed worker **elapsed** costs divided by 16,384 transitions. They
include possible descheduling and instrumentation. They run in parallel, so
they MUST NOT be added to the parent table or interpreted as end-to-end shares.
The rows are exclusive with respect to the other instrumented worker methods.

| Worker operation | Mean exclusive ms/transition |
|---|---:|
| Frozen-opponent action creation | 0.3121 |
| Player physics/state handling, excluding timed floor queries | 0.2469 |
| Player observation construction, excluding timed floor queries | 0.2249 |
| Floor queries, across their callers | 0.2100 |
| Remaining game-step body | 0.1590 |
| Opponent-history augmentation | 0.1280 |
| Player input processing | 0.1092 |
| Weapon pickup checks | 0.0836 |
| Observation assembly outside player-observation calls | 0.0794 |
| Pymunk native physics call, including callbacks | 0.0769 |
| Navigation reward/termination wrapper body | 0.0715 |

"Remaining game-step body" is not a claim that its contents are fully profiled:
it is the exclusive body of `WarehouseBrawl.step`, covering its untimed object
loops, preprocessing, allocations and housekeeping plus timer overhead.

There are 65,676 player-observation calls: four per game transition plus four
per reset. `observe(agent)` constructs both players' observations, and the
game calls it for each player. Computing each player's state once and reusing
it across the two views is a concrete optimization candidate, subject to
behavior/parity tests.

`ConstantAgent.predict` calls `np.zeros_like(self.action_space.sample())`:
it generates a random action and then discards its values. Replacing that with
correctly shaped zero construction is another candidate; verify dtype, shape,
ownership, and relevant RNG behavior rather than assuming perfect equivalence.

The 35 measured reset calls comprise 16 measured-start resets plus 19 episode
auto-resets. All wrapper-inclusive reset time totals 0.640549 worker-seconds;
game-core reset totals 0.557877 worker-seconds. These are NOT parent critical-path
reset costs and must not all be subtracted from parent `env_step`.

## Independent cProfile findings

`artifacts/native_1env/full.prof` and `artifacts/native_16env/full.prof` use
stdlib cProfile on the existing benchmark. They cover startup, one warmup and
two measured rollouts. The function totals below cover **24,576 transitions**,
not the detailed JSON's 16,384-transition window. Nested columns are explicitly
identified; do not sum parents with children.

| Direct child of collect_rollouts | 1 environment, seconds | 16 environments, seconds |
|---|---:|---:|
| Environment step | 20.277 | 4.647 |
| Policy call | 20.427 | 2.166 |
| Rollout-buffer add | 3.303 | 0.535 |
| Observation tensor conversion | 0.706 | 0.103 |
| Action CPU/NumPy conversion | 0.725 | 0.063 |
| Action clipping | 0.358 | 0.038 |
| Callbacks | 0.368 | 0.081 |
| Returns/advantages | 0.213 | 0.018 |

Within the 1-env policy-forward function, direct child edges include 7.997 s
in MLP/module calls, 5.087 s building action distributions, 4.685 s computing
log probabilities, 1.348 s obtaining actions, and 0.649 s feature extraction.
Thus "choose an action" is not synonymous with matrix multiplication alone.
These remain CPU-observed profiler values, not GPU kernel times.

The 16-env PPO `train` function totals 6.707 s over three updates. Nested within
it, rollout-buffer `get` totals 6.152 s and `_get_samples` totals **5.762 s**.
The latter has **5.408 s self time**. It performs NumPy indexed array gathering,
flattening and conversion of the training batch. Native array operations can
be charged to a Python function's self time; this is not evidence of 5.408 s
of Python bytecode. The direct tensor-conversion child totals only 0.353 s.

This identifies a training-data preparation hotspot independent of game code.
Memory layout, copying, allocation and host scheduling are hypotheses for
further investigation, not established explanations of the inter-run variation.

## Comparisons and missing evidence

| Run | Measured seconds | Aggregate steps/s | Measured update seconds |
|---|---:|---:|---:|
| Existing benchmark + whole-process cProfile, 16 env | 9.429864 | 1737.46 | 4.411687 |
| Existing benchmark, 16 env | 9.507784 | 1723.22 | 3.817140 |
| Detailed stage/worker probe, 16 env | 9.426802 | 1738.02 | 0.937602 |
| Existing benchmark + whole-process cProfile, 1 env | 32.287861 | 507.44 | 0.770874 |

The nearby comparison job and detailed probe were launched close together;
strict isolation across startup/warmup was not established. A later process
snapshot showed only the detailed probe. Do not use these runs to estimate
profiler overhead or causal speedup. There is only one repetition per setup.

The detailed 1-env run completed remotely (console reported 687.9 steps/s),
but its JSON was not downloaded before the automatic stop. It is intentionally
excluded from the tables. The final retrieval attempt failed because the
instance had stopped. Its remote directory is retained, but recovery would
require further authorized access. No additional restart was performed.

Retrieved JSON/profile files parse successfully and their internal counts were
audited. A final remote checksum comparison could not be completed before stop.

## Where C would and would not help

1. A targeted native implementation can address repeated observation building,
   game state/input handling and object-loop overhead inside workers. First test
   cheap duplicate-work removal without changing gameplay timing.
2. A batched native environment interface could additionally reduce per-worker
   Python/serialization boundaries. Merely replacing internal game methods
   with C does not automatically remove those boundaries.
3. Pymunk already calls the native `cpSpaceStep`; porting that wrapper alone
   does not turn a Python physics solver into a C solver.
4. A game-only port does not fix SB3 `_get_samples` or policy distribution/
   tensor handling. Those require separate data-layout or model-execution work.

No C implementation or C speedup was measured. The next evidence needed for a
stable baseline is isolated repeated runs, with matching instrumentation on/off
and explicit timing of batch gathering plus a short kernel-level trace.

## Verification and instance lifecycle

- Seven isolated diagnostic unit tests passed.
- All 17 existing repository tests passed after granting OS shared-memory access
  needed by local OpenMP subprocesses. The first sandboxed attempt failed on
  `OMP: Error #179: Function Can't open SHM failed`; no code fix was needed.
- Checked parent timing partition, worker exclusive partition, rollout/update
  counts, all 16,384 worker/parent interval matches, and local source hashes.
- Retained instance started at 2026-09-05 00:06:57 UTC and was verified stopped
  at 00:16:03 UTC. It was **not deleted**. Exact status is in `session.json`.
- Elapsed guard interval: 546.43 s; estimated running cost **$0.0481** at the
  recorded $0.316667/hour. This is an estimate, not a provider invoice.
- Retained storage was quoted separately at approximately **$0.40/day**.

The full multi-job `run_cloud.sh` was not executed. Only the completed artifacts
listed above should be treated as run evidence.
