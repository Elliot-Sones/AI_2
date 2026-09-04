#!/usr/bin/env bash
# Run only after headline throughput cases; profiling changes measured speed.
set -euo pipefail
cd /workspace/AI_2/benchmark-20260904b
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/workspace/AI_2/venv/bin/python
mkdir -p artifacts/diagnostics
trap 'status=$?; printf "%s\n" "$status" > artifacts/diagnostics.exit' EXIT
for size in small configured; do
  config=config.yaml
  steps=8192
  rollouts=2
  if [ "$size" = small ]; then
    config=baselines/configs/small_mlp.yaml
    steps=1024
    rollouts=8
  fi
  for device in cpu cuda; do
    name="${size}_${device}"
    date -u +%FT%TZ
    printf 'START %s\n' "$name"
    timeout 420 "$PY" -u benchmark_diagnostics.py --config "$config" --device "$device" --n-steps "$steps" --batch-size "$steps" --ppo-rollouts "$rollouts" --torch-threads 1 --output "artifacts/diagnostics/$name" > "artifacts/diagnostics/$name.log" 2>&1
    printf 'DONE %s\n' "$name"
  done
done
# Profile the same game/learner separately to identify expensive call paths.
timeout 420 "$PY" -u benchmark_diagnostics.py --config config.yaml --device cpu --n-steps 8192 --batch-size 8192 --ppo-rollouts 1 --torch-threads 1 --profile --output artifacts/diagnostics/configured_cpu_profile > artifacts/diagnostics/configured_cpu_profile.log 2>&1
date -u +%FT%TZ
