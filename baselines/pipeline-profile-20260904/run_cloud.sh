#!/usr/bin/env bash
# Diagnostic artifacts only; retained cached game/model configuration unchanged.
set -euo pipefail
cd /workspace/AI_2/animation-cache-20260904
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/ai2-pipeline-mpl-20260904
export PYTHONDONTWRITEBYTECODE=1
PY=/workspace/AI_2/venv/bin/python
PROBE=baselines/pipeline-profile-20260904/probe_pipeline.py
OUT=baselines/pipeline-profile-20260904/artifacts
mkdir -p "$OUT" "$MPLCONFIGDIR"
trap 'status=$?; printf "%s\n" "$status" > "$OUT/run.exit"' EXIT
date -u +%FT%TZ
printf 'START natural_16env\n'
timeout 120 "$PY" -u "$PROBE" --n-envs 16 --rollouts 2 --output "$OUT/natural_16env" > "$OUT/natural_16env.log" 2>&1
printf 'START control_16env\n'
timeout 120 "$PY" -u benchmark_training.py --config config.yaml --mode ppo --repeats 1 --n-envs 16 --n-steps 512 --batch-size 8192 --ppo-rollouts 2 --device cuda --torch-threads 1 --output "$OUT/control_16env" > "$OUT/control_16env.log" 2>&1
printf 'START natural_1env\n'
timeout 120 "$PY" -u "$PROBE" --n-envs 1 --rollouts 2 --output "$OUT/natural_1env" > "$OUT/natural_1env.log" 2>&1
printf 'START cprofile_16env\n'
timeout 120 "$PY" -u "$PROBE" --n-envs 16 --rollouts 2 --cprofile --output "$OUT/cprofile_16env" > "$OUT/cprofile_16env.log" 2>&1
printf 'START cprofile_1env\n'
timeout 120 "$PY" -u "$PROBE" --n-envs 1 --rollouts 1 --cprofile --output "$OUT/cprofile_1env" > "$OUT/cprofile_1env.log" 2>&1
printf 'START kernel_trace_16env_small_rollout\n'
timeout 90 "$PY" -u "$PROBE" --n-envs 16 --rollout-transitions 256 --warmup-transitions 256 --batch-size 256 --rollouts 1 --epochs 1 --torch-profile --output "$OUT/kernel_trace_16env" > "$OUT/kernel_trace_16env.log" 2>&1
printf 'START kernel_trace_1env_small_rollout\n'
timeout 90 "$PY" -u "$PROBE" --n-envs 1 --rollout-transitions 128 --warmup-transitions 128 --batch-size 128 --rollouts 1 --epochs 1 --torch-profile --output "$OUT/kernel_trace_1env" > "$OUT/kernel_trace_1env.log" 2>&1
date -u +%FT%TZ
printf 'DONE all_pipeline_probes\n'
