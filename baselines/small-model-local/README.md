# Small-model baseline — September 4, 2026

Status: local CPU benchmark completed; Vast.ai CPU/CUDA comparison not yet run.
No instance was created. The authenticated instance list was empty. Cloud work
is waiting for a rental spending limit and teardown authorization.

## Measured local result

Run: `baseline_20260904_164950/`. Config: `../configs/small_mlp.yaml`.
The policy uses two hidden layers of 64 units. Phase 0d, the 2,656-feature
stacked/history observation, rewards, and eight PPO optimization epochs are
unchanged. This is a fresh navigation policy, not trained combat/self-play.

One environment, CPU device, one Torch thread, seeds 42–44, 1,024 steps per
rollout and 1,024-sample batches. Each repeat includes one unmeasured warmup
rollout/update followed by eight complete measured rollout/update pairs.
Raw and wrapped modes each measure 10,000 action-loop steps per repeat.

| Workload | Median steps/s | Three-repeat range |
| --- | ---: | ---: |
| Raw game | 6,812.8 | 6,041.3–7,138.9 |
| Wrapped navigation | 2,327.8 | 2,101.8–3,060.8 |
| Small-model PPO | 1,928.9 | 1,573.5–2,088.2 |

Across 24,576 measured PPO transitions, the measured training loops took
13.376 seconds: 12.236 seconds collecting rollouts (91.5%), 0.756 seconds
updating the network (5.7%), and 0.384 seconds other work (2.9%). Rollout time
includes action inference, simulation, reset, wrappers, and advantage/return
calculation. These results identify collection as the dominant stage, not
simulation alone. Making only optimizer updates instantaneous would improve
this workload by at most about 6%; GPU inference could affect collection too.

Natural resets consumed 64.4% of the separate raw-game measured loop. This
supports inspecting reset/asset loading, but is NOT the reset share of PPO.
Workloads and episode behavior differ; do not subtract row times to infer
wrapper overhead or call the difference from the older local run a measured
model-size speedup. The older run used fewer PPO iterations.

This is a short diagnostic baseline, not a learning-quality or sustained-load
test. The run includes metadata, configuration snapshot, source hashes,
dependency freeze, per-repeat JSON, and episode CSV files. All nine repeats,
eight complete optimizer updates per PPO repeat, and individual source-file
hashes were verified. The simulator and main training config were not edited.

## Reproduce

```sh
python benchmark_training.py --config baselines/configs/small_mlp.yaml \
  --mode all --steps 10000 --repeats 3 --n-envs 1 --device cpu \
  --torch-threads 1 --batch-size 1024 --ppo-rollouts 8 \
  --output baselines/small-model-local
```

## Pending cloud comparison

Use the same small-model config on one GPU host. Measure CPU versus CUDA
sequentially at 1, 4, and 8 environments, subject to verified CPU/RAM allocation.
Keep 1,024 transitions per rollout and a 1,024-sample batch: use `--n-steps`
1024, 256, and 128 respectively. Keep seeds, eight update epochs, eight measured
rollouts, and three repeats fixed. Changing rollout length affects trajectories;
this matrix measures throughput, not equivalent learning quality.

Profile a separate single-environment PPO run to split collection time further;
keep instrumented numbers separate from headline throughput. Parent-process
profiling cannot reveal simulator call stacks inside subprocess environments.

Offer snapshot from the live CLI on September 4 (not reserved): offer 45602170,
RTX 3060 12 GiB, Ryzen 7 5800X host, 8 effective vCPUs, approximately 16 GB
allocated RAM, and 30 GiB requested disk. Quoted compute plus storage was
US$0.056222/hour; transfer was US$0.008789 per GB in each direction. This is a
shared CPU allocation, not eight dedicated physical cores. Recheck availability,
price, platform compatibility, and actual resources before renting.

Suggested authorization: up to US$1 total, at most one hour, retrieve results
and destroy the exact benchmark instance afterward. This is only a proposal,
not permission to rent. Storage and transfer are separate billing components:
[Vast.ai billing documentation](https://docs.vast.ai/guides/reference/billing).
