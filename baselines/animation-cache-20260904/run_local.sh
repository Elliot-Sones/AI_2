#!/usr/bin/env bash
# Sequential old/new local comparisons; no cloud or external writes.
set -euo pipefail
ROOT=/Users/elliot18/Projects/ml-research/AI_2
OLD=/tmp/ai2-animation-before.m31OQE
PY=/tmp/ai2-baseline.XWm6Cd/venv/bin/python
OUT="${1:-$ROOT/baselines/animation-cache-20260904}"
mkdir -p "$OUT"
for variant in before after; do
  source_root="$ROOT"
  if [[ "$variant" == before ]]; then source_root="$OLD"; fi
  cd "$source_root"
  echo "START local_${variant}"
  "$PY" benchmark_training.py --config "$ROOT/baselines/configs/small_mlp.yaml" \
    --mode all --steps 10000 --repeats 3 --n-envs 1 --n-steps 1024 \
    --batch-size 1024 --ppo-rollouts 8 --device cpu --torch-threads 1 \
    --output "$OUT/local-$variant" > "$OUT/local-$variant.log" 2>&1
  echo "DONE local_${variant}"
done
for variant in before after; do
  source_root="$ROOT"
  if [[ "$variant" == before ]]; then source_root="$OLD"; fi
  cd "$source_root"
  "$PY" benchmark_diagnostics.py --config config.yaml --device cpu \
    --n-steps 8192 --batch-size 8192 --ppo-rollouts 2 --torch-threads 1 \
    --output "$OUT/local-diagnostics-$variant" > "$OUT/local-diagnostics-$variant.log" 2>&1
  echo "DONE diagnostic_${variant}"
done
cd "$ROOT"
"$PY" benchmark_diagnostics.py --config config.yaml --device cpu \
  --n-steps 8192 --batch-size 8192 --ppo-rollouts 1 --torch-threads 1 --profile \
  --output "$OUT/local-profile-after" > "$OUT/local-profile-after.log" 2>&1
echo "DONE local_profile_after"
