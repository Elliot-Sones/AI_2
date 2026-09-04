#!/usr/bin/env bash
# Run after the learning benchmarks so test/transport workers do not contend.
set -euo pipefail
cd /workspace/AI_2/benchmark-20260904b
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/workspace/AI_2/venv/bin/python
mkdir -p artifacts/transport
trap 'status=$?; printf "%s\n" "$status" > artifacts/transport.exit' EXIT
test "$(cat artifacts/confirmation.exit)" = 0
timeout 180 "$PY" -m unittest discover -s tests -p 'test_benchmark*.py' > artifacts/tests-complete.log 2>&1
for n in 1 4 8 16; do
  date -u +%FT%TZ
  printf 'START transport_%senv\n' "$n"
  timeout 180 "$PY" -u benchmark_transport.py --n-envs "$n" --steps 65536 --warmup-steps 4096 --repeats 1 --obs-dim 664 --frame-stack 4 --output "artifacts/transport/${n}env" > "artifacts/transport/${n}env.log" 2>&1
  printf 'DONE transport_%senv\n' "$n"
done
date -u +%FT%TZ
