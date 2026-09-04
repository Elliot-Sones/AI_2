# Local baseline — September 4, 2026

Source code: git `bf7b00d`, with the standalone benchmark added; game and training
code unchanged. Results are in `baseline_20260904_154924/`.

Hardware: Apple M5 Pro, 18 logical CPUs, 64 GiB RAM, macOS 26.6.2 arm64.
CPU/RAM were verified separately with `sysctl`; the sandbox may leave these
fields unavailable in generated metadata. Python 3.11.14, Torch 2.4.1,
SB3 2.5.0, Gymnasium 1.0.0, NumPy 2.1.1, Pygame 2.6.1, Pymunk 6.2.1.
The full installed environment is captured in the run's `requirements.txt`.

These are short initial measurements, not a long-duration stability test.
Seeds were 42, 43, and 44. Each repeat used 10,000 measured raw steps,
10,000 wrapped navigation steps, and 3,072 PPO training steps, following warmup.
There was one environment, one Torch thread, CPU inference/training,
1,024 steps per PPO rollout, a 1,024-sample batch, and eight update epochs.
PPO retained the weights from one unmeasured warmup rollout/update.

| Workload | Median steps/s | Range across three seeds | Completed episodes |
| --- | ---: | ---: | ---: |
| Raw game, random inputs to both fighters | 6,666.5 | 6,183.8–7,115.4 | 24 |
| Navigation phase 0d with training wrappers | 2,367.5 | 2,146.0–3,032.5 | 75 |
| Navigation PPO collection plus updates | 1,262.0 | 1,137.6–1,313.2 | 22 |

The workloads differ, so ratios between rows do not isolate wrapper overhead.
Raw throughput includes natural resets. Navigation wraps a frozen opponent,
navigation rewards, opponent history, and frame stacking. The PPO agent is a
fresh policy, not any saved combat model. Shorter navigation episodes cause
more resets, which can substantially affect throughput.

Across the three PPO repeats, 7.474 seconds of measured time comprised 5.477
seconds collecting rollouts (73.3%), 1.617 seconds updating the network (21.6%),
and about 0.380 seconds of setup/other work. Rollout collection includes policy
inference, Python wrappers, simulator work, episode resets, and return/advantage
calculation. It must not be interpreted as simulator time alone.

Per-repeat mean PPO episode lengths were 314, 466, and 414 game steps, taking
0.259, 0.327, and 0.298 wall seconds respectively. These wall times include any
intervening learning updates and automatic resets; incomplete final episodes
are excluded. The JSON/CSV files contain every completed episode measurement.
These episode durations describe a short fresh-policy benchmark, not the old
trained agents.

## Reproduce

Create an isolated Python 3.11 environment, install the captured
`baseline_20260904_154924/requirements.txt`, then run from the repository:

```sh
python benchmark_training.py --mode all --steps 10000 --repeats 3 \
  --n-envs 1 --device cpu --torch-threads 1 --batch-size 1024 \
  --ppo-rollouts 3 --output baselines/local
```

For a more sustained comparison, use `--steps 100000 --ppo-rollouts 32` on both
implementations. Keep machine load and all other arguments fixed. The current
numbers cannot be compared directly with the historical MacBook Air logs or a
48-environment cloud run.

Verified: all three modes completed three repeats; stored source hashes match;
PPO recorded three complete rollout/update pairs per repeat; a separate
integration check verifies transition counts and inclusion of the final PPO
update. No performance optimization or C port has been applied yet.

A separate two-environment subprocess smoke run also passed: 128 wrapped
transitions and 32 PPO transitions with two complete optimizer updates. This
checks the parallel code path, not its sustained speed. OpenMP shared memory
required running that check outside the tool sandbox. Its tiny-workload timing
is intentionally excluded from the reported baseline.
