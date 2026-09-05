---
name: vastai-training
description: Manage Vast.ai GPU instances for this AI_2 repository's reinforcement-learning benchmarks and training. Use for GPU offer searches, instance setup, code upload, remote benchmarking, training, Weights & Biases dashboards and run descriptions, monitoring, result retrieval, and teardown.
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
- `tracking.py`, `monitor.py`, and the `wandb` section of `config.yaml`:
  Weights & Biases run creation, run descriptions, checkpoint and video
  upload, and the instance monitor. `tests/test_tracking.py` shows the contract.

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
No external training dataset is required by this project.

Weights & Biases is required for every cloud run. Install `wandb` in the remote
environment, then copy the local login into the remote `~/.netrc` without
printing it:

```sh
python - <<'EOF' | ssh -p <port> root@<host> 'umask 077; cat >> ~/.netrc'
import netrc
host = "api.wandb.ai"
login, _, password = netrc.netrc().authenticators(host)
print(f"machine {host}\n  login {login}\n  password {password}")
EOF
ssh -p <port> root@<host> 'cd <run-dir> && python -c "import wandb; print(wandb.Api().default_entity)"'
```

The second command must print the entity name. If the instance has no
outbound network, set `WANDB_MODE=offline` for every command below and sync
the run directories at retrieval time.

## Experiment tracking and run descriptions (Weights & Biases)

Every cloud run must be visible on the W&B dashboard with a description of
what it is for. Three mechanisms make this hold without relying on memory.

1. **Instance monitor.** Right after the runtime is verified, start one
   monitor run per instance and keep it running until teardown. It logs GPU
   utilization, GPU memory, power, CPU, and RAM for the whole instance
   lifetime, whatever script is using the GPU (benchmarks included).

   ```sh
   cd <run-dir> && nohup python monitor.py --name <instance-label> \
     --group <run-id> --record <run-dir>/monitor_run.json \
     > <run-dir>/monitor.log 2>&1 &
   echo $!   # record this PID with the instance
   ```

   Confirm `monitor_run.json` contains a `url` before proceeding, and report
   that URL at handoff.

2. **Training runs describe themselves.** `train.py` starts a W&B run per
   phase (TensorBoard scalars, config, checkpoints as artifacts, demo videos)
   and writes `<results>/wandb_run.json` (default `results/ppo_utmist/`) plus
   a history in `wandb_runs.jsonl`. Before launching, set the environment:

   ```sh
   export WANDB_RUN_GROUP=<run-id>
   export WANDB_NAME=<run-id>-phase<phase>
   export WANDB_TAGS=vast,<gpu-model>,phase-<phase>
   export WANDB_NOTES="$(cat <<'EOF'
   Goal: <one sentence: the question this run answers>
   Change: <what differs from the previous run: config keys, code, checkpoint>
   Expect: <the metric and value that counts as success, and by when>
   Setup: instance <id>, <gpu>, <n_envs> envs, phase <phase>, checkpoint <name>, commit <sha>
   EOF
   )"
   ```

   Write the notes yourself, in full sentences, from the actual config and
   the previous run's outcome. Never leave template placeholders. A run
   started without `WANDB_NOTES` is tagged `notes-missing` and must be fixed
   with `append-notes` before handoff.

3. **Outcome is appended when the run ends.** `train.py` appends a one-line
   completion note automatically. After the run, add the real result:

   ```sh
   python tracking.py append-notes --run <results>/wandb_run.json --text "$(cat <<'EOF'
   Outcome: <steps completed, steps/s, final win rates or reach rate, SD rate>
   Verdict: <did Expect hold? what to change next>
   EOF
   )"
   ```

   `python tracking.py info --run <results>/wandb_run.json` prints the record.
   Pass a run path such as `entity/project/id` instead of a file to fetch a
   remote run, including its state.

Benchmarks (`benchmark_training.py`) do not create W&B runs. The monitor run
covers their GPU and CPU usage. Attach the benchmark summary and the headline
numbers to the monitor run:

```sh
python tracking.py upload-file --run <run-dir>/monitor_run.json --file <output>/baseline_*/summary.json
python tracking.py append-notes --run <run-dir>/monitor_run.json --text "Benchmark: <mode, envs, device> -> <steps/s>"
```

Without outbound network, run everything with `WANDB_MODE=offline` and, at
retrieval time, run `wandb sync <run-dir>/wandb/offline-run-*` from a machine
that has the login. Never print or copy the API key into logs, configs, or
notes.

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

Confirm the training run's W&B URL appears in the log (`📈 W&B run:`) and in
`<results>/wandb_run.json`; a run that starts untracked on a cloud instance is
a setup failure, not an acceptable outcome. Use the W&B pages (training run
and monitor run) as the dashboard; `nvidia-smi` and log tails remain the
ground truth when the dashboard lags.

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

Close out the dashboard before teardown: append the outcome notes to every
training run and to the monitor run, stop the monitor with `kill -TERM <pid>`
and confirm its log ends cleanly, sync any offline run directories, and check
that no run is left in state `running` (`python tracking.py info --run
<entity/project/id>`). List every run URL in the handoff report.

Destroy only the resolved instance when the user has authorized destruction or
an applicable auto-teardown condition. First verify required results are safely
downloaded; then confirm the instance's resulting state through the CLI. If
teardown is not authorized, report that it remains active and its known cost.
