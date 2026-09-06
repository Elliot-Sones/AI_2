# Local player-observation reuse experiment — September 5, 2026

`WarehouseBrawl` now computes each player's observation once when returning a
step/reset result. Both returned arrays retain their original values, dtype,
64-element shape, player/opponent ordering, and independent ownership.
Standalone `observe(agent)` still reads current state every time. There is no
persistent observation cache, C rewrite, reward change, or training change.

## Measured results

All rates are aggregate environment transitions per second. Medians and ranges
are across three seeds; startup and warmup are excluded.

| Workload | Before median (range) | After median (range) | Throughput ratio | Time reduction at median rate |
| --- | ---: | ---: | ---: | ---: |
| Raw game, no policy | 17,763 (15,097–18,082) | 20,798 (20,633–21,298) | 1.171x | 14.6% |
| Wrapped navigation, no policy | 8,457 (8,322–9,576) | 10,488 (10,184–10,716) | 1.240x | 19.4% |
| Full small-model PPO | 4,309 (4,181–4,547) | 4,544 (4,372–4,851) | 1.054x | 5.2% |
| Full configured-network PPO | 1,662 (1,531–1,922) | 1,960 (1,925–1,989) | 1.180x | 15.2% |

Every matched seed was faster after the change, but the size varied. Paired
configured-network throughput gains were **19.7%, 25.7%, and 2.0%**; small-model
gains were **1.4%, 8.7%, and 6.7%**. The unchanged optimizer also ran at different
speeds across comparisons, so these results do not isolate every saved
millisecond to observation construction. Treat the medians as observed local
results, not a guaranteed 18% throughput gain on other runs or hardware.
No long-run learning-quality or GPU performance improvement is established.

The separate raw-game profile counted **40,444 → 20,222** `Player.get_obs`
calls, exactly halving them. Those counts include construction/reset, warmup,
and 10,000 measured transitions. Profiled cumulative observation time fell
from 0.608 s to 0.341 s, but profiler timings are excluded from the table.

The completed verifier confirmed **1,335 matching natural episode records**
across the before/after workloads, identical replay hashes, unchanged
configurations/dependencies, only the intended environment source difference,
and all 32 rollout/update pairs in every PPO run. Raw evidence and seed-level
timings are in [comparison.json](comparison.json); the verification transcript
is [verification.log](verification.log).

## Measurement contract

- Apple M5 Pro, 18 reported CPU cores, 64 GiB RAM, macOS 26.6.2, Python 3.11.14,
  Torch 2.4.1, SB3 2.5.0, NumPy 2.1.1.
- CPU device, one environment, one Torch/BLAS thread; headless, fresh phase 0d
  policies, no online tracking, video, evaluation, or checkpoint callbacks.
- Three seeds (42, 43, 44), each with a separate before/after process. Order is
  before/after for 42 and 44, after/before for 43. Each process runs alone;
  this task's tests and independent reviews finished before measurements.
  Normal desktop activity was not controlled.
- Small model `[64,64]`: 100,000 raw-game transitions, 100,000 wrapped
  navigation transitions, and 32 complete 1,024-transition PPO rollouts and
  optimizer updates per seed. Configured model `[512,512,256]`: the same PPO
  rollout/batch settings, separately measured. Eight epochs per update.
- Warmup and startup are excluded. PPO warmup weights are retained identically
  in both variants. Each model/version has 98,304 measured PPO transitions;
  each raw/wrapped version has 300,000 measured transitions.
- These are longer fresh comparisons, not a rerun of the earlier animation
  cache benchmark. Both variants already include the animation cache.

Raw game and wrapped navigation use fixed random inputs. PPO uses a fresh
learning policy. Their speed ratios do not isolate wrapper overhead because
their trajectories differ. The large network uses local benchmark overrides
(one environment and batch 1,024), not the production configuration's 48
environments and batch 8,192. No GPU or cloud benchmark was run for this change.

## Correctness and verification

- Eight observation integration tests pass. Before the change, the two
  intended call-count tests failed with `4 != 2`; the other six passed.
  The new code satisfies exactly two real `Player.get_obs` calls for each
  step/reset. Tests cover fresh state reads, reset replacement of players,
  dtype/shape/perspectives, independent arrays, and terminal/truncated results.
- All 36 repository tests and seven pipeline diagnostic tests pass. Syntax
  checks and `git diff --check` pass. No Python lint/typecheck runner is
  configured. Independent code review approved the production change;
  see [review notes](review.md).
- Old/new seeded gameplay replays match exactly across 30,000 steps and 60
  forced resets. The hashes include observations, rewards, player states,
  weapons/spawners, and boundary flags. Those forced-reset runs have no natural
  endings; natural episodes are checked separately in the benchmark records.
- Initial full-suite attempts exposed a missing existing W&B dependency and
  macOS sandbox denial of OpenMP shared memory. Installed `wandb==0.29.0` in
  the temporary test virtualenv and granted local OS access. No source fix
  was needed. W&B's shutdown hook rejects unittest's boolean exit code;
  the successful final runner supplies integer 0/1. A stdin-based runner was
  unsuitable for spawned workers; the final runner uses `python -c`.
  Failed-attempt logs are retained, and `tests-full-final.log` is the clean
  full-suite result.

The clean full-suite command (with the environment variables from the
reproduction section, but `WANDB_MODE=offline`) was:

```sh
"$PY" -c 'import sys, unittest
suite = unittest.defaultTestLoader.discover("tests")
result = unittest.TextTestRunner().run(suite)
sys.exit(0 if result.wasSuccessful() else 1)'
"$PY" -m unittest discover -s baselines/pipeline-profile-20260904 -p 'test_*.py'
```

## Source provenance and reproduction

The control copies the pre-change files from this workspace into
`/tmp/ai2-observation-before-fbkldjuj`. `control-manifest.json` records 154
source/config/asset files with SHA-256 and sizes. The git reference is
`12414651c6f0b41a38f77d6d21ba7767dee1ee88`; unrelated concurrent rerun work is
outside this change. The optimized production diff is only `environment.py`.

- Before environment SHA-256:
  `1bc31afc1319613c7d7f81a3280e87c2a036ca3c97801bc1579c766c5c39077e`.
- After environment SHA-256:
  `c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230`.

The benchmark writes exact arguments, source/config hashes, package versions,
hardware metadata, per-repeat JSON, episode CSVs, and timing summaries. The
result verifier rejects unexpected source differences, changed dependencies
or settings, episode/replay differences, and missing PPO rollouts/updates.
It removes only episode `wall_seconds` and nested `info.episode.t` timestamps
when comparing behavior. Run it with normal Python or `python -O`; its checks
remain enabled.

From the repository root, using the existing compatible test virtualenv:

```sh
PY=/tmp/ai2-baseline.XWm6Cd/venv/bin/python
CONTROL=$(mktemp -d /tmp/ai2-observation-control.XXXXXX)
OUT=$(mktemp -d /tmp/ai2-observation-results.XXXXXX)
git archive 12414651c6f0b41a38f77d6d21ba7767dee1ee88 \
  benchmark_training.py benchmark_diagnostics.py train.py tracking.py \
  config.yaml positions.json baselines/configs/small_mlp.yaml \
  UTMIST-AI2-main/environment | tar -x -C "$CONTROL"
cp assets.zip "$CONTROL/"

export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
export MPLCONFIGDIR=$(mktemp -d /tmp/ai2-observation-mpl.XXXXXX)
"$PY" baselines/animation-cache-20260904/replay_trace.py \
  --source-root "$CONTROL" --output "$OUT/replay-before.json"
"$PY" baselines/animation-cache-20260904/replay_trace.py \
  --source-root "$PWD" --output "$OUT/replay-after.json"
bash baselines/observation-reuse-20260905/run_local.sh "$CONTROL" "$PY" "$OUT"
"$PY" baselines/observation-reuse-20260905/summarize.py "$OUT"
```

The temporary virtualenv/control paths must be recreated if removed. Normal
local process/shared-memory access is required for PyTorch subprocess tests.
The historical pipeline worker timer instruments `observe`; this change uses
`_observe_all` at step/reset boundaries, so that historical instrumentation
would need adjustment before a new detailed pipeline comparison. The separate
raw cProfile here captures the new helper and the actual `Player.get_obs`
calls directly.
