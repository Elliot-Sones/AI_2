#!/usr/bin/env bash
# Sequential, local-only control/optimized measurements. No source mutation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONTROL="${1:?Pass the preserved control source directory}"
PY="${2:-/tmp/ai2-baseline.XWm6Cd/venv/bin/python}"
OUT="${3:-$ROOT/baselines/observation-reuse-20260905}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export MPLCONFIGDIR="$(mktemp -d /tmp/ai2-observation-mpl.XXXXXX)"

# Alternate the ordering by seed to reduce systematic before/after drift.
# Each timed process runs alone; metadata records every benchmark argument.
for seed in 42 43 44; do
  variants=(before after)
  if [[ "$seed" == 43 ]]; then variants=(after before); fi
  for variant in "${variants[@]}"; do
    source_root="$ROOT"
    if [[ "$variant" == before ]]; then source_root="$CONTROL"; fi
    cd "$source_root"
    "$PY" benchmark_training.py --config "$ROOT/baselines/configs/small_mlp.yaml" \
      --mode all --steps 100000 --repeats 1 --seed "$seed" --n-envs 1 \
      --n-steps 1024 --batch-size 1024 --ppo-rollouts 32 \
      --device cpu --torch-threads 1 \
      --output "$OUT/small-$variant/seed-$seed" > "$OUT/small-$variant-$seed.log" 2>&1
    printf 'Completed small-model %s seed %s\n' "$variant" "$seed"
    "$PY" benchmark_training.py --config "$ROOT/config.yaml" \
      --mode ppo --repeats 1 --seed "$seed" --n-envs 1 \
      --n-steps 1024 --batch-size 1024 --ppo-rollouts 32 \
      --device cpu --torch-threads 1 \
      --output "$OUT/configured-$variant/seed-$seed" > "$OUT/configured-$variant-$seed.log" 2>&1
    printf 'Completed configured-model %s seed %s\n' "$variant" "$seed"
  done
done

# Profiling is separate from all headline measurements.
for variant in before after; do
  source_root="$ROOT"
  if [[ "$variant" == before ]]; then source_root="$CONTROL"; fi
  cd "$source_root"
  "$PY" benchmark_training.py --mode raw --steps 10000 --repeats 1 \
    --seed 42 --profile --device cpu --torch-threads 1 \
    --output "$OUT/profile-$variant" > "$OUT/profile-$variant.log" 2>&1
  printf 'Completed raw profile %s\n' "$variant"
done
