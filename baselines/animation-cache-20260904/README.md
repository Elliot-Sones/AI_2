# Animation cache: matched local and RTX 4090 before/after

## Result

Reusing decoded animation frames removes the reset bottleneck without changing
animation durations or weapon pickup timing. On the same RTX 4090 host,
configured-model PPO is **3.06x faster with 8 environments** and **4.18x faster
with 16 environments**. The matched small-model CPU benchmark on this Apple
M5 Pro is **2.59x faster**. These are throughput results for fresh navigation
policies, not evidence of faster learning to a target score.

Local headline results come from `isolated-repeat/`, run sequentially after all
review/test workloads finished. The first pass remains in the top-level
`local-before/` and `local-after/`; it is not the headline because review tests
may have overlapped. No benchmark/model/config/reward changes were made.

## Matched small-model measurements

Same machine, Python 3.11.14, Torch 2.4.1, SB3 2.5.0, CPU device, one Torch
thread, one environment, seeds 42/43/44, phase 0d, `[64,64]` policy,
1,024-step rollout and batch, eight PPO epochs. Each repeat measures 10,000
raw and wrapped transitions, and eight complete PPO rollouts (8,192 steps)
after warmup. Startup is excluded. All values are aggregate transitions/s.

| Workload | Before median (range) | Cached median (range) | Ratio |
| --- | ---: | ---: | ---: |
| Raw game, no policy | 6,681 (6,042–7,188) | 18,759 (18,524–19,352) | 2.81x |
| Navigation wrappers, no policy | 2,405 (2,127–2,997) | 10,249 (9,914–10,264) | 4.26x |
| Full small-model PPO | 1,892 (1,599–2,072) | 4,894 (4,720–4,920) | 2.59x |

Each pair completed the same total episodes: raw 24, wrapped 75, PPO 63.
Per-episode step counts, termination flags, navigation outcomes and rewards
match exactly after excluding wall-clock fields. Fixed-action hashes also
match. Metadata confirms identical config, dependency versions, machine and
source hashes except `environment.py`.

## Where the time goes now

A separate, instrumented CPU diagnostic uses the configured `[512,512,256]`
policy, one environment, 8,192-step rollout/batch, two measured rollouts and
seed 42. It is not the small-model headline benchmark.

| Component | Before | Cached |
| --- | ---: | ---: |
| Total time for 16,384 transitions | 11.281 s | 7.349 s |
| 33 game resets, total | 4.076 s | 0.0287 s |
| Average game reset | 123.51 ms | 0.871 ms |
| Game stepping | 1.156 s | 1.171 s |
| Policy inference | 2.428 s | 2.519 s |
| PPO updates | 2.367 s | 2.379 s |

Reset time fell **99.29%**; total diagnostic throughput improved **1.54x**.
The 32 completed episode records match, excluding top-level `wall_seconds`
and nested `info.episode.t` wall-clock fields. After caching,
game reset is 0.39%, game stepping 15.94%, policy inference 34.28%, and PPO
updates 32.38% of measured time. Inclusive vector-env timing also contains
game stepping/resets; do not add it to those components.

Next priorities:

1. Profile parallel CUDA collection/inference in more detail. The matched
   8/16-environment rebenchmark is now complete below, but local CPU fractions
   cannot establish the remaining parallel 4090 bottleneck. Neural-network
   operations already execute in native Torch kernels; rewriting the game in C
   does not accelerate those operations.
2. For a narrow C candidate, inspect observation assembly, state/floor checks,
   and collision/pickup loops. In the separate cached CPU cProfile run, 8,192
   game steps take 1.492 s inclusively; `observe` is 0.552 s and its nested
   `Player.get_obs` is 0.510 s (32,840 calls). These are overlapping profiler
   timings, not additional exclusive costs. Consider reducing duplicate work
   before a port; use batched arenas if a port is justified.

In the **non-cProfile instrumented CPU diagnostic**, even zero-cost game
stepping alone would yield at most `1/(1-0.1594) = 1.19x` further speedup,
with everything else fixed. This is a workload-specific upper bound, not a
prediction for CUDA, more workers, combat, or a batched implementation.

## Behavior and implementation verification

- 17 unittest cases pass, including nine new cache/timing regression tests.
  The pre-change tests exposed two decodes instead of one and 16 GIF reopens
  across two warmed resets; cached code performs zero warmed-reset reopens.
- Independent spec and code-quality reviews approved the change.
- Three 10,000-step old/new random-input replays match exactly: observations,
  rewards, player position/velocity/damage/stocks/weapons, spawner state and
  boundary flags. There are 60 forced resets; no natural terminations occur in
  this forced-reset replay. Natural episode boundaries are additionally covered
  by the matched raw/wrapped/PPO benchmark records.
- Explicit tests preserve RGBA pixels, GIF durations, per-sprite FPS rounding,
  independent playback/metadata, canonical-path reuse, file invalidation,
  failed-decode retry, source pixels after rendering, and the nine-step test
  spawner's pickup unlock boundary.
- Cache capacity is 128 decoded files **per process**, not shared across
  subprocess workers. First use in each worker still decodes the file. Modified
  mtime/size reloads it. Source surfaces are shared and must remain read-only;
  current renderers create transformed copies. Animation metadata lists and
  playback counters remain private.
- The unused legacy `Particle` GIF loader is out of scope; no current training
  instantiation was found. Existing reset/seed limitations and production
  combat/self-play behavior were not rewritten.

## Matched RTX 4090 results

The original and cached implementations ran sequentially on instance
**49902188**, an RTX 4090 with AMD EPYC 7K62 and 24 effective CPU cores.
These runs use the configured `[512,512,256]` policy, not the small model.

| CUDA workload | Before median steps/s (range) | Cached median steps/s (range) | Ratio |
| --- | ---: | ---: | ---: |
| 8 environments | 707.69 (661.90–726.11) | 2,165.00 (2,160.12–2,229.65) | 3.06x |
| 16 environments | 704.86 (694.05–775.38) | 2,945.71 (2,869.89–3,232.04) | 4.18x |

Each configuration uses three seeds and six measured 8,192-transition rollouts
per repeat, batch 8,192, eight PPO epochs and one Torch thread. That is 49,152
transitions per seed and 147,456 per configuration/version. Before and after
config/dependency snapshots match, and source hashes differ only in the cached
environment file. All six updates per seed and every recorded episode outcome
match after excluding wall-clock fields. Total completed episodes remain 287
for 8 environments and 286 for 16 environments.

Before caching, 8 and 16 environments had similar throughput. After caching,
16 is **36.1% faster** than 8 by median, with non-overlapping observed ranges.
Removing repeated reset work therefore improved useful parallel scaling, not
just the speed of an isolated reset. Sixteen is the best tested count, not a
claim that it is globally optimal. No 24/32/48-environment cached sweep was run.

Median per-repeat collection/update shares are 91.27%/8.64% at 8 environments
and 87.19%/12.64% at 16 environments. Collection includes policy inference,
simulation, wrappers and process communication; it is not synonymous with
Python game simulation.

The same host's small-model CPU benchmark improved from 459.31 to 893.40 PPO
steps/s (**1.95x**). Its raw/wrapped medians improved 1,803.15 to 5,008.12
(2.78x) and 669.85 to 2,756.14 (4.11x). These use the same small-model shape
as the Mac test, but do not mix these CPU figures with configured-model CUDA.

### Remaining cloud bottleneck

A separate synchronized, single-environment CUDA diagnostic used the configured
model, seed 42, two measured 8,192-step rollouts and the same batch/epochs.

| Component | Before | Cached |
| --- | ---: | ---: |
| Total time for 16,384 transitions | 43.283 s | 23.293 s |
| 44 game resets, total | 19.077 s | 0.137 s |
| Average reset | 433.58 ms | 3.108 ms |
| Game stepping | 3.823 s | 3.741 s |
| Policy inference | 12.022 s | 11.304 s |
| PPO updates | 0.689 s | 0.665 s |

Reset cost fell **99.28%** and total diagnostic throughput rose **1.86x**.
All 43 completed episode outcomes match. Cached reset time is 0.59%, game
stepping 16.06%, policy inference 48.53% and updates 2.86% of measured wall
time. Inclusive vector-env time is 30.48% and contains game stepping/reset;
do not add those overlapping percentages. Synchronization/instrumentation can
affect timings, and these single-environment fractions are **not** the
component breakdown of the 8/16-environment headline runs.

The next priorities are batching/inference and profiling parallel collection
and process communication, alongside targeted game improvements. A full C port
does not accelerate the neural-network kernels. For a narrow game target,
the separate CPU cProfile on this same cloud host shows 8,192 game steps taking
5.692 s inclusively: `observe` 1.810 s, nested `Player.get_obs` 1.629 s across
32,840 calls, `Player.physics_process` 1.640 s, `is_on_floor` 0.814 s, and
pickup loops 0.432 s. These nested profiler costs are not additive and are
not GPU wall-time shares. Investigate duplicate observation assembly and
state/collision checks before deciding what to port to C.

### Verification and retained instance

The upload permission blocker was resolved by explicit user approval. Only the
approved source/config/test/asset payload was uploaded into the separate
optimized-run directory; the original source was preserved. Environment/test
uploads passed checksum comparison, CUDA passed a real tensor-operation check,
and all 17 tests passed remotely. Old/new cloud gameplay replays match across
30,000 steps and 60 forced resets. Benchmark and diagnostic episode traces also
match, excluding top-level `wall_seconds` and nested `info.episode.t`.

`artifacts/after.exit` is 0, all requested after runs/profile files are present,
and final downloads passed checksum-mode rsync verification. Source/config
hashes, package versions and the remote dependency freeze are retained.

Instance 49902188 was stopped and verified `actual_status=exited`,
`cur_state=stopped`; it was **not deleted**. Its stop-only guard was cancelled
after verification. The original-control session used about 13 minutes
(~$0.07), and the approved optimized continuation used about 13 minutes
(~$0.07), at the quoted $0.3167/hour. These are estimates, not invoices.
Retained storage remains about $0.40/day. Original remote source
`/workspace/AI_2/benchmark-20260904b` is intact.

## Reproduction and evidence

- `run_local.sh` records exact sequential commands; its optional first argument
  selects a separate result directory. The retained old local snapshot was
  extracted from commit `bf7b00d` into `/tmp/ai2-animation-before.m31OQE`, with
  unchanged benchmark scripts and the root `assets.zip` copied in. This
  temporary path must be recreated if cleaned up.
- `isolated-repeat/local-before/` and `isolated-repeat/local-after/` contain
  per-repeat JSON, per-episode CSV, complete config snapshots, package versions,
  source hashes and command lines.
- `isolated-repeat/local-diagnostics-{before,after}/` contains component timing;
  `isolated-repeat/local-profile-after/` contains the separately profiled run.
- `replay_trace.py`, `local-replay-before.json`, `local-replay-after.json` retain
  the behavior comparison. `tests-final.log` retains the final test output.
- `run_before.sh`, `run_after.sh`, `session.json`, `artifacts/before/`,
  `artifacts/after/`, `artifacts/diagnostics/` and `artifacts/parity/` retain
  the remote commands, controls, measurements, behavior checks and lifecycle.
- Cloud small-CPU and diagnostic before controls are in
  `../cloud-4090-20260904b/artifacts/screen/small_cpu_local_shape/` and
  `../cloud-4090-20260904b/artifacts/diagnostics/`. Primary CUDA 8/16 controls
  are this run's `artifacts/before/`, not the earlier short screening runs.
- Old environment SHA256:
  `e6aacc4e80f817f6b4895184cef7be7b8eecdd5ba23eecbd442fd7fcb12e1b25`.
  Cached environment SHA256:
  `1bc31afc1319613c7d7f81a3280e87c2a036ca3c97801bc1579c766c5c39077e`.

No new dependencies were added. Syntax and whitespace checks pass. There is
no configured Python lint/typecheck runner in this repository. This short,
fresh-policy navigation test does not establish long-run stability, trained
combat throughput, or learning-quality parity.
