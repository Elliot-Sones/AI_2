# Training performance baseline

The first milestone is to measure the existing Python game before changing its
implementation. `benchmark_training.py` is an isolated benchmark; it does not
change the game, training configuration, or saved policies.

Measured results: [local baseline](baselines/local/README.md) and
[RTX 4090 bottleneck study](baselines/cloud-4090-20260904b/README.md).
The cloud study includes a matched small-model test, CPU/CUDA environment
scaling, longer repeats, and reset/animation profiling.

Follow-up: [animation cache before/after](baselines/animation-cache-20260904/README.md)
measures 2.59x faster small-model PPO on the local CPU, and 3.06x/4.18x faster
configured-model CUDA PPO with 8/16 environments on the same 4090 host. Gameplay
traces match. Both comparisons are complete; the retained instance is stopped.

Local follow-up: [player observation reuse](baselines/observation-reuse-20260905/README.md)
halves repeated observation construction. Three matched seeds measured 1.171x
raw-game throughput, 1.054x small-model PPO and 1.180x configured-network PPO
on the local CPU. Episode records and gameplay replays match. Gain sizes vary
between runs; no GPU comparison or learning-quality claim accompanies this change.

## Measurements

1. **Raw game:** replay seeded random inputs for both fighters without rendering.
   Measure initialization, stepping, resets, completed episode lengths, and wall
   time. This is the initial comparison target for a C simulator.
2. **Wrapped navigation environment:** include the existing navigation rewards,
   opponent history, frame stacking, and vector environment transport. Use the
   current navigation phase, with an explicitly recorded environment count.
3. **PPO:** measure rollout collection and optimization separately, and count
   only complete training iterations after warmup. Use a fresh policy and record
   every override. This is a throughput measurement, not a learning-quality test.

Use three repeated runs and report the median and range. Keep the seeds,
machine, software versions, environment count, policy architecture, rollout
length, batch size, update epochs, thread count, and episode rules fixed when
comparing implementations. Record startup separately from sustained throughput.

Episode length in simulation steps, elapsed wall time per completed episode,
and aggregate completed episodes per second are different measurements. With
parallel environments, wall times overlap. Faster episode completion can also
mean worse play, so environment transitions per second is the primary speed
metric. One raw game transition advances both players once; do not double it.

## Existing artifacts and reproducibility limits

- Historical measurements from November 25, 2025:

  | Run | Logged steps | Final aggregate steps/s | Final mean episode steps |
  | --- | ---: | ---: | ---: |
  | PPO_2 | 28,672 | 340 | 1,284.81 |
  | PPO_5 | 49,152 | 819 | 1,448.78 |
  | PPO_6 | 49,152 | 789 | 1,437.36 |
  | PPO_7 | 16,384 | 676 | 1,451.50 |

  These are short runs; PPO_7 has only one scalar sample. PPO_1, PPO_3,
  and PPO_4 contain no scalar measurements. Complete extracted series and
  source hashes are in `baselines/historical_tensorboard.json`. Do not divide
  mean episode length by aggregate FPS to infer individual episode latency
  without knowing concurrency. First-to-last event timestamps omit startup,
  the first collection period, and potentially the last optimizer update.
- Historical TensorBoard events are under
  `UTMIST-AI2-main/results/ppo_utmist/tb/PPO_*/`, with filenames identifying
  `Elliots-MacBook-Air.local`.
- They contain SB3 `time/fps` and `rollout/ep_len_mean`, but are not individual
  episode wall-time logs. `time/fps` includes learning overhead.
- The top-level saved combat policies are in `results/ppo_utmist/model/`.
- `config.yaml` currently selects navigation phase `0d`, 48 environments, and
  a navigation checkpoint that is absent from the top-level model directory.
- The simulator advances at 30 Hz. A raw match permits 9,000 steps (300 simulated
  seconds); phase `0d` uses a 1,800-step navigation limit. Some config comments
  describe 60 Hz and must not be used to interpret episode duration.
- The game's reset methods do not fully honor Gymnasium's seed argument.
  The benchmark explicitly seeds Python, NumPy, and Torch before constructing
  each worker. This improves repeatability within the Python implementation;
  exact replay across a C port will also require matching RNG state/algorithms.
- Fixed random actions do not represent a trained fighter or a self-play pool.
  Add frozen-policy combat and an immutable opponent pool before claiming a
  production training speedup.

## C migration boundary

Keep PPO and the policy network in Python/PyTorch initially. Expose a headless
C game core through the same observation, action, reward, termination, and
reset contracts. Batch multiple arenas per call when the measurements justify
it, so Python does not make one foreign-function call per game object.

Pymunk already wraps Chipmunk's C physics engine. Profile Python state changes,
attack/hitbox processing, observation construction, allocations, wrapper work,
process communication, and reset/asset loading before deciding what to port.
The C version should initially reuse the same physics and attack data.

An initial diagnostic profile on the local Apple M5 Pro replayed 5,000 seeded
random binary action pairs. Its five natural resets consumed about 0.70 seconds
of the 1.53-second profiled loop, mostly loading/decoding VFX animation images.
Observation construction was also prominent. These are profiler timings from
one workload, not uninstrumented benchmark results or a claimed C speedup.
They identify a concrete opportunity to investigate headless asset loading
before undertaking the full port.

The raw game accepts two ten-element input vectors and returns one 64-element
observation per player. The current navigation training adds 600 opponent-history
features and stacks four frames, producing 2,656 policy inputs. Preserve the
actual action thresholds and observation layout when constructing the C boundary;
the config's `action_space: discrete` label does not describe the actual Box
action space used by the code.

Before accepting a C implementation, replay matching states and inputs through
both implementations. Compare positions/velocities with explicit tolerances;
compare stocks, damage, attacks, rewards, observations, and episode boundaries.
Then repeat the same benchmark and evaluate learning quality over multiple
seeds. Do not mix reward/gameplay corrections into the speed comparison.

A faster simulator does not imply the same improvement in full training. For
example, if simulation is 70% of runtime and becomes 10 times faster, total
training becomes approximately `1 / (0.30 + 0.70 / 10) = 2.7` times faster.
Those percentages are an illustration, not measurements from this repository.

## Verification plan

- Import and step the unchanged environment using the repository's pinned
  Torch, SB3, Gymnasium, NumPy, Pygame, and Pymunk versions.
- Smoke-test all three benchmark modes and validate timing/count accounting.
- Run the baseline repeatedly without a profiler, then profile a separate run
  so instrumentation does not contaminate headline throughput.
- Save the exact command, dependency versions, configuration and source hashes,
  raw measurements, and a concise interpretation alongside the results.

## Running the benchmark

From a Python environment with the project's training dependencies installed:

```sh
python benchmark_training.py --mode all --steps 10000 --repeats 3 \
  --n-envs 1 --device cpu --torch-threads 1 --batch-size 1024 \
  --ppo-rollouts 3 --output baselines/local
```

This deliberately uses one environment and a 1,024-sample optimization batch.
The configured rollout length (1,024), network, and eight optimization epochs
remain in use. The original configuration uses 48 environments and an
8,192-sample batch. The local run is a new reference point for later local
comparisons, not a reproduction of the old cloud run.

For the cloud machine, use the same script and explicitly select its settings:

```sh
python benchmark_training.py --mode all --steps 10000 --repeats 3 \
  --n-envs 48 --device cuda --torch-threads 1 --batch-size 8192 \
  --ppo-rollouts 3 --output baselines/cloud
```

This still measures a fresh navigation policy with warmup and no periodic
evaluation, video recording, or self-play checkpoint writes. Benchmark frozen
combat policies separately before projecting production combat-training cost.

The benchmark generates a unique directory per invocation, with metadata,
per-repeat JSON, per-episode CSV, and a summary. Preserve that directory when
moving between machines. Run profilers separately from the reported baseline.

References: [SB3 2.5.0 logging definitions](https://stable-baselines3.readthedocs.io/en/v2.5.0/common/logger.html),
[Pymunk's native physics boundary](https://www.pymunk.org/en/latest/advanced.html),
and the [Gymnasium environment contract](https://gymnasium.farama.org/api/env/).
