---
name: vastai-training
description: Manage Vast.ai GPU instances for this AI_2 repository's reinforcement-learning benchmarks and training. Use for GPU offer searches, instance setup, code upload, remote benchmarking, training, monitoring, result retrieval, and teardown.
---

# Vast.ai training for AI_2

Adapted from the SportsNGEN `vastai-training` skill in SMT-nf-recovery.
Use the `vastai` CLI and SSH. Resolve the project root from this skill's
repository, not a hardcoded path from the source project.

## Enter at the requested stage

Installing or inspecting this skill does not launch a cloud workflow. When the
user requests cloud work, inspect existing instances before creating another:

```sh
command -v vastai
vastai show instances --raw
```

Use current CLI help for offer fields, creation options, image selection, and
instance lifecycle commands. CLI presence does not prove authentication. Some
CLI versions fetch GPU names even for `--help`; a network failure there is not
proof of missing credentials. Keep credentials in the existing credential
mechanism; do not print or copy them.

Reuse an instance already selected in this conversation when appropriate.
Track its exact ID, hourly cost, SSH host/port, remote directory, run ID,
and process/log paths. Do not stop or modify unrelated instances.

Before a new rental, resolve the machine, image, disk allocation, and estimated
cost against the user's existing authorization. Ask for the missing choice or
budget only when needed; do not repeat approval requests for already authorized
setup, upload, or training. Installing this skill alone authorizes no rental.

## Read the actual workload

- `config.yaml`: curriculum phase, environment count, policy architecture, PPO
  settings, checkpoint selection, and output paths.
- `train.py`: training entry point, wrappers, and device selection.
- `benchmark_training.py`, `BASELINE.md`, and `baselines/local/README.md`:
  benchmark commands, timing definitions, and local comparison limits.
- `UTMIST-AI2-main/requirements.txt` and `setup_vast.sh`: dependency/setup
  references. Inspect before using the setup script; its CUDA installation
  choices may not match the selected image.

The game simulation runs on the CPU even when PPO uses CUDA. Compare offers
using CPU allocation/performance, RAM, disk performance, GPU, reliability,
network transfer costs, and total hourly price. Do not select solely by GPU
model or assume the source skill's 24 GB VRAM requirement applies here.

## Provision and upload

After an authorized rental, resolve the returned instance ID. Poll readiness
and verify a real SSH command succeeds; populated SSH fields alone are not a
working connection. Keep waits short enough to report progress. Stop retries
when the instance fails or the authorized time/cost bound is reached.

Use normal SSH host-key verification; verify unexpected key changes. Upload
into a dedicated run directory, such as `/workspace/AI_2/<run-id>`. Use the
resolved host, port, and directory in commands, with proper shell quoting.

Transfer the current working source, including uncommitted benchmark files:

- `train.py`, `config.yaml`, `benchmark_training.py`, and `positions.json`.
- `UTMIST-AI2-main/`, including environment code, attack JSON, and game assets.
- Benchmark documentation and tests when verifying or reproducing a baseline.
- Only the checkpoint files needed for the selected run.

Use `rsync` without deletion flags. Exclude local virtual environments,
`.git`, credentials, caches, old videos/logs, and unrelated results. Verify the
remote source/config hashes match the intended local snapshot. This game
currently loads some graphical assets even in headless mode; omitting all
assets will not produce a valid baseline.

## Verify the remote runtime

Create an isolated environment with a Python/CUDA combination compatible with
the repository's core versions. Install the actual imports needed by `train.py`
and the environment, including PyYAML and sb3-contrib. Treat the local saved
dependency freeze as provenance; resolve CUDA wheels for the remote platform
and record its installed versions rather than blindly copying a Mac runtime.

Verify CPU allocation, RAM, `nvidia-smi`, Torch's CUDA availability, and an actual
CUDA tensor operation. Run an environment reset/step and a tiny benchmark before
the sustained workload. Use `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`.
No W&B account or external training dataset is required by this project.

## Benchmark before optimizing

Run commands from the uploaded project root, using its isolated Python.
First smoke-test the selected device, then measure CPU and CUDA sequentially on
the same machine with identical settings. Example fixed-workload comparison:

```sh
python benchmark_training.py --mode all --steps 10000 --repeats 3 \
  --n-envs 1 --device cpu --torch-threads 1 --batch-size 1024 \
  --ppo-rollouts 3 --output baselines/cloud-cpu

python benchmark_training.py --mode all --steps 10000 --repeats 3 \
  --n-envs 1 --device cuda --torch-threads 1 --batch-size 1024 \
  --ppo-rollouts 3 --output baselines/cloud-cuda
```

Scale environment count within the allocated CPU/RAM budget and preserve the
effective rollout/batch settings in each comparison. Use longer measurements
for sustained claims, for example `--steps 100000 --ppo-rollouts 32`. Keep
profiling separate from headline throughput runs and avoid concurrent workloads.

The current benchmark's wrapped/PPO modes support navigation phases and fresh
policies with warmup. They do not reproduce trained combat or self-play cost.
Raw mode measures a two-player random-input game; its reset-time fraction is
not the reset fraction of full PPO. Check the current config rather than
assuming a historical phase or checkpoint is still selected.

## Train and monitor

For a full training request, prepare a run-specific config with the selected
phase, environment count, output directory, and verified checkpoint path.
Do not silently start fresh if the user requested resuming a missing checkpoint.

```sh
python -u train.py --cfgFile config.yaml
```

Use the prepared config path instead of `config.yaml` when it differs. Run
long jobs in a persistent terminal or `nohup`, recording the PID, log path,
exit status, and exact command. Confirm that steps advance and expected output
files appear; a background PID alone does not establish successful training.

Monitor the selected process, recent log output, CPU/GPU utilization, memory,
and output timestamps. Report aggregate steps/s, episode lengths/wall times
where actually recorded, rollout/update timing for benchmarks, and gameplay
metrics for training. Do not infer episode latency from aggregate FPS without
accounting for concurrency. A quiet GPU may be waiting on CPU simulation,
resets, inference batches, or transfers; profile before diagnosing.

After connection failures, check the exact instance's status before retrying.
Report relevant active billing at handoff, and follow any authorized stopping
policy. Do not create replacement rentals or extend a bounded run silently.

## Retrieve results and teardown

Download benchmark run directories with metadata, summary JSON, per-episode
records, logs, and the remote dependency freeze. For training, include selected
checkpoints and TensorBoard directories (`tb_nav` or `tb` under the configured
results folder). Keep remote run IDs separate from existing local results.

Verify transfer completeness using file sizes/hashes and inspect the result
files before claiming completion. Report hardware and workload differences when
comparing cloud measurements with the local baseline.

Destroy only the resolved instance when the user has authorized destruction or
an applicable auto-teardown condition. First verify required results are safely
downloaded; then confirm the instance's resulting state through the CLI. If
teardown is not authorized, report that it remains active and its known cost.
