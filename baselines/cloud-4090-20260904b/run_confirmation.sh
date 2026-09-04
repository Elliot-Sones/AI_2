#!/usr/bin/env bash
# Longer repeat runs of the two strongest neighboring CUDA configurations.
set -euo pipefail
cd /workspace/AI_2/benchmark-20260904b
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/workspace/AI_2/venv/bin/python
mkdir -p artifacts/confirmation
trap 'status=$?; printf "%s\n" "$status" > artifacts/confirmation.exit' EXIT
for n in 8 16; do
  date -u +%FT%TZ
  printf 'START configured_cuda_%senv\n' "$n"
  timeout 600 "$PY" -u benchmark_training.py --config config.yaml --mode ppo --repeats 3 --n-envs "$n" --n-steps "$((8192 / n))" --batch-size 8192 --ppo-rollouts 6 --device cuda --torch-threads 1 --output "artifacts/confirmation/configured_cuda_${n}env" > "artifacts/confirmation/configured_cuda_${n}env.log" 2>&1
  printf 'DONE configured_cuda_%senv\n' "$n"
done
date -u +%FT%TZ
