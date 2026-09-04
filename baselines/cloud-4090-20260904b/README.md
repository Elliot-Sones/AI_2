# RTX 4090 training baseline — September 4, 2026

## Main finding

The first concrete simulator optimization to test is **reset-time animation
loading**, before committing to a full C rewrite. The GPU greatly accelerates
the configured network's PPO updates, but then rollout collection dominates.
Collection includes inference, game steps, resets, wrappers, communication,
and advantage calculation; it is not synonymous with simulation.

The game and training implementation were not optimized for these measurements.
This is a fresh **phase 0d navigation** throughput baseline, not trained combat,
self-play, a learning-quality comparison, or a maximum GPU stress test.

## Machine and measurement contract

- Vast instance `49902188`, label `ai2-benchmark-4090-20260904b`.
- RTX 4090, 24,564 MiB reported VRAM; AMD EPYC 7K62 host. The offer allocated
  24 vCPUs/about 64 GB RAM. The measured cgroup CPU quota was 23.04 CPU
  equivalents; the host's 192 logical CPUs are **not** our allocation.
- Linux x86-64, Python 3.11.9, PyTorch 2.4.1+cu121, SB3 2.5.0, Gymnasium 1.0.0,
  NumPy 2.1.1, Pymunk 6.2.1, Pygame 2.6.1. See
  [hardware](artifacts/hardware.txt) and [installed dependencies](artifacts/requirements-linux.txt).
- Headless execution, one Torch/BLAS thread, no video/evaluation/checkpoint
  callbacks. One learner uses multiple environment copies, not one model per
  environment. There is no visible scene rendering during these benchmarks.
- Timed PPO windows exclude environment/model construction and one warmup
  rollout/update. Warmup-trained weights are retained. Every measured rollout
  includes its optimizer update, including the final one.
  The PPO `init_seconds` field measures model construction only, not full
  process/environment startup; do not use it as total launch latency.
- Schedules retain the configured 5,000,000-step denominator. The benchmark
  constructs fresh policies and does not load `model_checkpoint`; no
  checkpoints were uploaded.
- Observations: 64 game features + 600 opponent-history features = 664 values
  before parent-side four-frame stacking, yielding 2,656 policy inputs.
- Headline numbers are unprofiled. Detailed component timers and cProfile run
  separately; their inclusive timings must not be summed as disjoint costs.

## 1. Simple model baseline

Architecture `[64, 64]`, one environment, rollout/batch 1,024, eight epochs,
eight measured PPO rollouts per repeat, three seeds (42–44).

| Execution | Median training steps/s | Range |
| --- | ---: | ---: |
| This cloud host, CPU | 459.3 | 390.7–466.7 |
| This cloud host, CUDA | 408.2 | 397.0–435.2 |
| Local Apple M5 Pro, CPU, matched small-model test | 1,928.9 | 1,573.5–2,088.2 |

The tiny one-environment model did not benefit from CUDA on this host. CPU
and CUDA trajectories differ, so this is not an isolated kernel comparison.
The matched cloud CPU run was about 4.2 times slower than the local CPU run;
that comparison includes CPU architecture, OS, libraries, and host/container
conditions, not just GPU choice. Seeds and completed-episode counts matched
for the local/cloud CPU raw, wrapped, and PPO workloads.

The cloud small-model suite also recorded raw-game median 1,803.1 steps/s and
wrapped-navigation median 669.9 steps/s (10,000 transitions per repeat).
Those workloads differ from PPO, so their ratios do not isolate wrapper cost.

Evidence: `artifacts/screen/small_{cpu,cuda}_local_shape/`, and the matched
[local small-model run](../small-model-local/baseline_20260904_164950/summary.json).
Do not confuse it with the earlier, differently configured `baselines/local/`.

## 2. Configured model: environment-count screening

Architecture `[512, 512, 256]`, 8,192 aggregate samples per rollout, batch 8,192,
eight epochs, two measured rollouts (16,384 transitions), one seed per case.
`n_steps = 8192 / n_envs`. One environment uses `DummyVecEnv`; multiple
environments use spawned `SubprocVecEnv` workers.

| Environments | CPU steps/s | CUDA steps/s |
| ---: | ---: | ---: |
| 1 | 189.6 | 401.8 |
| 4 | 241.3 | 502.9 |
| 8 | 256.7 | 678.5 |
| 16 | 261.0 | 628.2 |

CUDA cut configured-model optimization from roughly 33 seconds to 0.7–1.1
seconds per two-update test. At eight environments, CPU spent about 52.5% of
wall time updating the network; CUDA spent 4.6%. CUDA collection took 92.7%.
The stronger GPU therefore moves the bottleneck toward collecting experience.

This short sweep does not establish that eight environments beats sixteen.
Changing rollout length per worker changes trajectory/episode mix and can
affect learning. This is a controlled throughput screen with fixed aggregate
rollout size, not proof of equivalent training quality.

### Longer confirmation

Three seeds per setting, six measured rollouts per seed (49,152 transitions),
for eight and sixteen CUDA environments. Warmup remains excluded.

| CUDA environments | Median steps/s | Range | Collection share | Update share |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 702.4 | 688.4–714.7 | 96.1% | 3.0% |
| 16 | 702.5 | 700.7–794.6 | 92.9% | 5.0% |

Shares are aggregated by measured wall time across the three repeats. Each
repeat lasted 62–71 measured seconds. Both settings completed 147,456 measured
transitions across their three repeats, with 287/286 completed episodes.
The median rates are practically identical; three seeds are not enough to
establish a statistically reliable winner. The short screen's apparent
eight-worker advantage did not persist as a clear median difference.
Use eight as a compact reference point, not a universally optimal setting.

Evidence: [eight-worker summary](artifacts/confirmation/configured_cuda_8env/baseline_20260904_221807/summary.json)
and [sixteen-worker summary](artifacts/confirmation/configured_cuda_16env/baseline_20260904_222242/summary.json).
See [runner](run_confirmation.sh).

## 3. Where collection time goes

Single-environment configured CUDA diagnostic, 16,384 transitions:

| Component | Seconds | Share of 43.28-second measured window |
| --- | ---: | ---: |
| Game resets, 44 calls | 19.08 | 44.1% |
| Raw game steps, 16,384 calls | 3.82 | 8.8% |
| Policy forward calls | 12.02 | 27.8% |
| PPO optimizer updates, 2 calls | 0.69 | 1.6% |

Each reset averaged 0.434 seconds; there were 44 measured reset calls and 43
completed episodes. The small-model
CUDA diagnostic similarly spent 45.7% on resets and 8.9% on raw game steps.
These are instrumented, single-environment measurements, not a decomposition
of the eight/sixteen-worker runs. CUDA synchronization changes timing, and
policy forward excludes some surrounding tensor conversion/transfer work.

The separate configured CPU profile identifies the reset path:

`WarehouseBrawl.reset -> _setup -> WeaponSpawner.initialize_vfx -> SpawnerVFX
-> load_animations -> load_animation -> PIL/GIF decoding and Pygame surfaces`

Across 18 resets, `game_reset` took 8.214 seconds and the 36 spawner
`initialize_vfx` calls took 8.144 seconds cumulatively. All 47 `load_animations`
calls across the profile took 8.157 seconds, including other VFX construction.
These are nested, not additive.
The source unconditionally constructs spawner VFX during this setup path even when
training is headless. Source anchors:
[reset](../../UTMIST-AI2-main/environment/environment.py#L1135),
[animation loading](../../UTMIST-AI2-main/environment/environment.py#L3152),
[spawner VFX setup](../../UTMIST-AI2-main/environment/environment.py#L4207).

Evidence: [configured CUDA timers](artifacts/diagnostics/configured_cuda/benchmark_diagnostics.json),
[CPU profile timers](artifacts/diagnostics/configured_cpu_profile/benchmark_diagnostics.json),
[CPU call profile](artifacts/diagnostics/configured_cpu_profile/benchmark_diagnostics.prof).

## 4. Cheap-environment communication control

The control replaced the game with constant preallocated observations and
empty info dictionaries, preserving 664-value worker observations, four-frame
parent stacking, ten-value actions, and Dummy/Subproc vectorization. Each case
measured 65,536 aggregate transitions after 4,096 warmup transitions.

| Workers | Control steps/s | Measured seconds |
| ---: | ---: | ---: |
| 1, DummyVecEnv | 30,921 | 2.12 |
| 4, SubprocVecEnv | 15,912 | 4.12 |
| 8, SubprocVecEnv | 20,979 | 3.12 |
| 16, SubprocVecEnv | 24,951 | 2.63 |

One short control repeat per setting, not a sustained IPC benchmark. The
roughly 21,000–25,000 steps/s at 8–16 workers is far above full training's
roughly 702 steps/s. Simple transport and frame stacking alone do not explain
the low full-training rate. Evidence: `artifacts/transport/*env/summary.json`.

This omits actual game logic, policy inference, history/reward wrappers, and
real info payloads. It tests whether transport/framing alone can explain the
observed rates; it is **not** an exact measurement of real-game IPC overhead.
Natural cheap-environment resets occur every 1,800 steps per worker; counters
separate warmup resets from measured resets.

## What to optimize next

1. Cache decoded immutable animation assets per worker, or avoid visual asset
   loading during headless training while preserving game state and behavior.
   Regression-test reset/step parity, then rerun this unchanged benchmark.
   Do not simply remove VFX objects: `WeaponSpawner.update` uses the spawn
   animation's duration to decide when a weapon can be picked up. Preserve
   that timing metadata and keep mutable animation state separate from cached
   image surfaces.
2. Re-profile after that change. Investigate observation/history construction,
   action/tensor transfers, policy batching, and synchronous worker waiting.
   The best next target may change once resets are cheap.
3. If simulation still dominates, put the game-state/step core behind a native
   batched interface while keeping PPO and PyTorch in Python. Pymunk physics
   already calls a native engine; translating every Python file is not an
   automatic gain. Validate numerical/gameplay parity before speed comparisons.
4. Add frozen-policy combat and a fixed opponent pool before projecting real
   self-play training time or choosing production hardware/environment counts.

These measurements do not predict an achieved C speedup. Eliminating all raw
game-step work would only remove about 9% of the single-environment CUDA
diagnostic window; that bound does not include reset improvements and cannot
be extrapolated to different combat or parallel workloads.

## Reproduction and artifacts

- [Screening commands](run_screen.sh), [diagnostic commands](run_diagnostics.sh),
  [longer confirmations](run_confirmation.sh), [transport/tests](run_transport.sh).
- Every main run saves command arguments, full configuration, source SHA256s,
  package versions, per-repeat JSON, and per-completed-episode CSV. Wall-time
  episode measurements include intervening updates/resets and overlap between
  workers; aggregate steps/s is not individual episode latency.
- Source snapshots are identified by hashes rather than remote git status:
  `.git` was intentionally not uploaded. Local original commit was `bf7b00d`.
- Remote setup reused CUDA Torch in an isolated venv. `build-essential` was
  needed to build pinned Pymunk. The existing root `assets.zip` was uploaded
  because the environment checks for it before using extracted assets.
  Initial setup failures are retained; successful retry/test logs supersede them.
- Only needed game code/config/assets and benchmark helpers were uploaded;
  no checkpoints, credentials, `.git`, or unrelated project data were uploaded.

## Instance lifecycle and cost

The user explicitly requires confirmation before deletion. Instance `49902188`
was **stopped, not destroyed**, after testing. At approximately 22:32 UTC Vast
reported `actual_status=exited`, `cur_state=stopped`, and both intended/next
states `stopped`, with the 30 GB disk retained. The local stop-only deadline
guard was cancelled after this verification. See [session ledger](session.json).

The quoted running rate including the 30 GB disk is $0.3167/hour. Retained
stopped storage is $0.40/GB/month, approximately $0.40/day for 30 GB. The $1
benchmark budget is not a perpetual cap on storage retained at the user's
request. Rates/elapsed-time estimates are not a reconciled provider invoice.
This instance's roughly 50-minute running period costs approximately $0.26 at
the quoted rate, excluding transfer, other instances, and subsequent storage.

Verification: all four stage exit markers are zero; all eight remote benchmark
tests passed. The six longer PPO repeats contain 294,912 measured transitions
and 36 complete rollout/update pairs, each repeat over 60 seconds. Screening
transition/timing counts, source hashes, and all transport reset/shape/count
checks passed. Game/training/config files have no modifications.

The complete remote `artifacts/` tree matched the local copy in a final
checksum-based rsync dry run with no missing or different files. The final
uploaded benchmark helpers also matched local checksums. No benchmark
processes remained before stopping, and the preserved instance is verified
stopped. No further remote work or deletion is scheduled.
