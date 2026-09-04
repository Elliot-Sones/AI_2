#!/usr/bin/env bash
# Fixed benchmark commands; no simulator or training configuration changes.
set -euo pipefail
cd /workspace/AI_2/benchmark-20260904b
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/workspace/AI_2/venv/bin/python
mkdir -p artifacts/screen
trap 'status=$?; printf "%s\n" "$status" > artifacts/screen.exit' EXIT
run_case() {
  local name="$1"
  shift
  date -u +%FT%TZ
  printf 'START %s\n' "$name"
  printf '%q ' "$PY" benchmark_training.py "$@"
  printf '\n'
  timeout 420 "$PY" -u benchmark_training.py "$@" --output "artifacts/screen/$name" > "artifacts/screen/$name.log" 2>&1
  printf 'DONE %s\n' "$name"
}
# Match the local small-network baseline first, including raw/wrapped modes.
run_case small_cpu_local_shape --config baselines/configs/small_mlp.yaml --mode all --steps 10000 --repeats 3 --n-envs 1 --n-steps 1024 --batch-size 1024 --ppo-rollouts 8 --device cpu --torch-threads 1
run_case small_cuda_local_shape --config baselines/configs/small_mlp.yaml --mode ppo --repeats 3 --n-envs 1 --n-steps 1024 --batch-size 1024 --ppo-rollouts 8 --device cuda --torch-threads 1
# Hold total rollout and optimization batch fixed while changing worker count.
# This changes per-environment trajectory length; it is a throughput screen.
for n in 1 4 8 16; do
  steps=$((8192 / n))
  for device in cpu cuda; do
    run_case "configured_${device}_${n}env" --config config.yaml --mode ppo --repeats 1 --n-envs "$n" --n-steps "$steps" --batch-size 8192 --ppo-rollouts 2 --device "$device" --torch-threads 1
  done
done
date -u +%FT%TZ
