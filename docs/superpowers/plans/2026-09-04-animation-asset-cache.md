# Animation Asset Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate repeated animation decoding across resets without changing gameplay, then measure the real training gain.

**Architecture:** Keep a 128-entry process-local decoded-file LRU next to AnimationSprite2D. Share original image surfaces, copy each Animation's metadata lists, and preserve the exact timing formula and sprite state.

**Tech Stack:** Python 3.11, functools.lru_cache, Pillow, Pygame, unittest, existing SB3 benchmarks and Vast.ai RTX 4090.

---

## Task 1: Test and implement asset reuse

Files: modify `UTMIST-AI2-main/environment/environment.py`; create
`tests/test_animation_cache.py`. No other game/config/training changes.

- [ ] Write tests using real temporary GIFs and existing spawner assets. Use a
  wraps spy around Image.open only to count real decoding. First demonstrate
  that two sprite loads decode twice instead of once on the old implementation.
- [ ] Assert equal RGBA bytes, frame durations and frames_per_step; separately
  mutable metadata/frame lists; independent play/process counters; canonical
  path reuse; file fingerprint invalidation; fps-specific timing; failed-load
  retry; and the actual WeaponSpawner.update pickup boundary before/at spawn
  animation duration. Include repeated WarehouseBrawl resets with no re-decode
  after warmup, and verify cached frames survive a render call unchanged.
- [ ] Run `/tmp/ai2-baseline.XWm6Cd/venv/bin/python -m unittest tests.test_animation_cache`
  and retain expected pre-implementation failure evidence.
- [ ] Implement the minimal loader shape:

```python
@lru_cache(maxsize=128)
def _load_animation_assets(path, mtime_ns, size):
    frames, durations = [], []
    with Image.open(path) as gif:
        for frame in ImageSequence.Iterator(gif):
            frames.append(pygame.image.fromstring(
                frame.convert("RGBA").tobytes(), frame.size, "RGBA"))
            durations.append(frame.info.get("duration", 100))
    return tuple(frames), tuple(durations)

def load_animation(self, file_path):
    path = os.path.realpath(file_path)
    stat = os.stat(path)
    frames, durations = _load_animation_assets(path, stat.st_mtime_ns, stat.st_size)
    return Animation(list(frames), list(durations),
                     [max(1, round(d / 1000 * self.ENV_FPS)) for d in durations])
```

  Keep the decode's RGBA conversion, default 100 ms duration, and Pygame surface
  construction identical. Do not change timing/state/render behavior.
- [ ] Run the targeted tests to green and all tests with
  `/tmp/ai2-baseline.XWm6Cd/venv/bin/python -m unittest discover -s tests`.
- [ ] Run spec compliance review, then independent code-quality review. Resolve
  findings. Preserve uncommitted work; do not commit or change branches here.

## Task 2: Benchmark matched old and new code

Files: create operational scripts/results under `baselines/animation-cache-20260904/`;
leave `baselines/cloud-4090-20260904b/` and original remote source unchanged.

- [ ] Resume exact instance 49902188, record new endpoint/state and establish a
  stop-only 30-minute deadline guard with exact instance/label checking.
- [ ] Verify old remote source hash against the saved baseline. Run old code
  sequentially at 8 and 16 CUDA environments, 3 seeds and 6 measured rollouts:
  `benchmark_training.py --config config.yaml --mode ppo --repeats 3 --n-envs N --n-steps $((8192/N)) --batch-size 8192 --ppo-rollouts 6 --device cuda --torch-threads 1`.
- [ ] Upload new source/tests into a distinct run directory and checksum-verify.
  Run remote tests and verify CUDA before performance measurements.
- [ ] Repeat those exact CUDA commands on new code without overlapping loads.
  Repeat the old small CPU `--mode all --steps 10000 --repeats 3 --n-envs 1
  --n-steps 1024 --batch-size 1024 --ppo-rollouts 8` configuration.
- [ ] Run separate configured single-environment CUDA diagnostics and CPU
  cProfile using `benchmark_diagnostics.py`, not in headline measurements.
- [ ] Verify transition/update/episode accounting, matching action/episode
  sequences where deterministic, hashes and downloaded artifacts. Report
  medians/ranges, reset reduction, remaining bottleneck and learning-quality limits.
- [ ] Stop, never delete, instance 49902188; verify provider state; cancel only
  this run's deadline guard. Record continuing storage cost.

## Completion evidence

- [ ] Red/green tests, integration tests, syntax/diff checks and review complete.
- [ ] Old/new artifacts retained separately, report written, instance stopped.
