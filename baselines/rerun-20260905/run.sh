#!/bin/bash
set -euo pipefail
cd /workspace/AI_2/rerun-20260905
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=offline
export MPLCONFIGDIR=/tmp/ai2-rerun-mpl
PY=/workspace/AI_2/venv/bin/python
mkdir -p artifacts "$MPLCONFIGDIR"
trap 'printf "%s\n" "$?" > artifacts/run.exit' EXIT
"$PY" monitor.py --name ai2-rerun-49976003 --group rerun-20260905 --interval 1 --record artifacts/monitor_run.json --notes 'Rerun the missing one-environment PPO profile on a new 4090 host. Offline monitoring and local-only archival requested. This is a new diagnostic, not a controlled speed comparison.' > artifacts/monitor.log 2>&1 &
MONITOR=$!
trap 'status=$?; kill -TERM "$MONITOR" 2>/dev/null || true; wait "$MONITOR" || true; printf "%s\n" "$status" > artifacts/run.exit' EXIT
for attempt in {1..30}; do
  test -s artifacts/monitor_run.json && break
  sleep 1
done
test -s artifacts/monitor_run.json
"$PY" -m unittest discover -s baselines/pipeline-profile-20260904 -p 'test_*.py' -v > artifacts/tests.log 2>&1
timeout 180 "$PY" -u baselines/pipeline-profile-20260904/probe_pipeline.py --n-envs 1 --rollouts 2 --output artifacts/natural_1env > artifacts/natural_1env.log 2>&1
