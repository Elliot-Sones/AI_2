# Observation Reuse Implementation Plan

**Goal:** Remove repeated player observation construction and measure its local
training throughput impact without changing gameplay.

**Architecture:** Build each player's observation once per step/reset and
assemble both ordered views from those lists. Keep standalone observations
fresh and avoid any persistent cache. Execute in the current workspace under
the user's approval, preserving unrelated edits; use a separate temporary
source snapshot as the benchmark control.

**Tech Stack:** Existing Python 3.11 virtualenv, NumPy, Pymunk, unittest,
Stable Baselines3/PyTorch CPU benchmarks. No new dependencies.

## Task 1: Lock behavior and reproduce duplication

Files: `tests/test_observation_reuse.py` and a new
`baselines/observation-reuse-20260905/` evidence directory.

- [x] Preserve current game, training, benchmark, config and asset files in a
  temporary control source tree; record the original environment hash.
- [x] Write integration tests against real WarehouseBrawl/Player objects.
  Count actual `get_obs` executions through a wraps spy: step/reset should
  each call it twice. Check values against `observe(0)`/`observe(1)`, dtype,
  64-value shape, reversed halves, independent arrays, live state mutations,
  repeated resets, and terminal/truncated observations.
- [x] Run the new test file with the existing virtualenv and retain expected
  red output proving four real calls before implementation. Existing value
  and behavior assertions must pass against the original implementation.

## Task 2: Implement the narrow change

File: `UTMIST-AI2-main/environment/environment.py`.

- [x] Replace the two step/reset observation comprehensions with
  `self._observe_all()` and add:

```python
def _observe_all(self) -> dict[int, np.ndarray]:
    player_obs = [player.get_obs() for player in self.players]
    return {
        agent: np.array(player_obs[agent] + player_obs[1 - agent])
        for agent in self.agents
    }
```

- [x] Run targeted tests to green, then the existing test suite and relevant
  pipeline diagnostics tests; run syntax and `git diff --check` checks.
- [x] Obtain an independent code review while preparing benchmark scripts;
  resolve concrete findings before measured runs.

## Task 3: Verify behavior and measure speed locally

Files: scripts, raw measurements and README under
`baselines/observation-reuse-20260905/`. Keep existing benchmarks unchanged.

- [x] Run the existing animation-cache `replay_trace.py` against control and
  optimized source trees: three seeds, 10,000 steps each, 60 forced resets.
  Compare all action and trajectory hashes exactly.
- [x] Run unprofiled control/optimized benchmarks sequentially without
  concurrent test or review workloads. Use three seeds, CPU, one Torch
  thread, one environment, phase 0d, 32 measured 1,024-step PPO rollouts
  with 1,024 batch size, and 100,000 raw/wrapped steps. Test both the small
  `[64,64]` network and the configured `[512,512,256]` network. Alternate
  before/after order by seed. Longer windows reduce noise for this smaller
  optimization; these are fresh comparisons, not the previous cache runs.
- [x] Run a separate matched raw-game cProfile diagnostic to verify reduced
  `get_obs` calls; never include profiler timings in headline throughput.
- [x] Compare source/config/dependency hashes, action hashes, episode lengths,
  rewards, termination flags and complete update counts. Exclude only
  recorded wall-clock fields from behavior comparisons.
- [x] Report medians/ranges, observed speedup or lack of speedup, test and
  parity evidence, and the limits of local fresh-policy navigation runs.
  Preserve raw output and exact reproduction commands.

## Completion evidence

43 tests passed (36 repository and 7 diagnostic). Old/new gameplay hashes match
for 30,000 steps and 60 forced resets; 1,335 natural episode records match.
`Player.get_obs` profile calls fell from 40,444 to 20,222. Local medians improved
1.171x raw, 1.240x wrapped, 1.054x small PPO, and 1.180x configured-network PPO.
All three seed pairs improved, with substantial timing variation; see the
experiment README for ranges, optimizer timing caveats, and reproduction.
Independent review approved the change; its low verifier-assertion concern
was fixed. No GPU run or C implementation was performed.
