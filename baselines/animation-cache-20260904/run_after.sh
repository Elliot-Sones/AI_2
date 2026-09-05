#!/usr/bin/env bash
set -euo pipefail
cd /workspace/AI_2/animation-cache-20260904
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/workspace/AI_2/venv/bin/python
mkdir -p artifacts/after artifacts/diagnostics artifacts/parity
trap 'status=$?; printf "%s\n" "$status" > artifacts/after.exit' EXIT
test "$(cat artifacts/before.exit)" = 0
timeout 180 "$PY" -m unittest discover -s tests > artifacts/tests.log 2>&1
timeout 180 "$PY" replay_trace.py --source-root /workspace/AI_2/benchmark-20260904b --output artifacts/parity/before.json > artifacts/parity/before.log 2>&1
timeout 180 "$PY" replay_trace.py --source-root /workspace/AI_2/animation-cache-20260904 --output artifacts/parity/after.json > artifacts/parity/after.log 2>&1
"$PY" -c 'import json; a=json.load(open("artifacts/parity/before.json")); b=json.load(open("artifacts/parity/after.json")); assert a["results"] == b["results"], "Gameplay replay differs"; print("Identical 30000-step replay across 3 seeds including forced resets, boundary flags and weapon pickup state")' > artifacts/parity/verification.log
for n in 8 16; do
  date -u +%FT%TZ
  printf 'START after_cuda_%senv\n' "$n"
  timeout 420 "$PY" -u benchmark_training.py --config config.yaml --mode ppo --repeats 3 --n-envs "$n" --n-steps "$((8192 / n))" --batch-size 8192 --ppo-rollouts 6 --device cuda --torch-threads 1 --output "artifacts/after/configured_cuda_${n}env" > "artifacts/after/configured_cuda_${n}env.log" 2>&1
  printf 'DONE after_cuda_%senv\n' "$n"
done
timeout 240 "$PY" -u benchmark_training.py --config baselines/configs/small_mlp.yaml --mode all --steps 10000 --repeats 3 --n-envs 1 --n-steps 1024 --batch-size 1024 --ppo-rollouts 8 --device cpu --torch-threads 1 --output artifacts/after/small_cpu > artifacts/after/small_cpu.log 2>&1
timeout 180 "$PY" -u benchmark_diagnostics.py --config config.yaml --device cuda --n-steps 8192 --batch-size 8192 --ppo-rollouts 2 --torch-threads 1 --output artifacts/diagnostics/configured_cuda > artifacts/diagnostics/configured_cuda.log 2>&1
timeout 180 "$PY" -u benchmark_diagnostics.py --config config.yaml --device cpu --n-steps 8192 --batch-size 8192 --ppo-rollouts 1 --torch-threads 1 --profile --output artifacts/diagnostics/configured_cpu_profile > artifacts/diagnostics/configured_cpu_profile.log 2>&1
date -u +%FT%TZ
