# Combat update qualification implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development and test-driven-development. Keep all edits in this isolated worktree; do not modify main or the curriculum owner's worktree.

**Goal:** Qualify the existing deferred-metric PPO update on real native combat, then compare batch sizes on a faithfully ported existing curriculum stage using time-to-behavior evidence.

**Architecture:** Keep raw combat, its Gaussian policy, rollout buffer, optimizer, epochs, and collection unchanged for the matched update experiment. Add a default-off update choice that reuses `deferred_metrics_train`. Curriculum integration is a separate, explicitly qualified surface; raw-reward throughput never substitutes for learning results.

**Tech stack:** Existing Python, PyTorch, SB3 2.5.0, C/CUDA native game lanes, unittest; no new dependencies.

## Approved design and constraints

The user approved the two-step recommendation on 2026-09-09. Preserve exact PPO updates before changing minibatch size. Prefer existing deferred-metric code over a new algorithm. Do not enable compiled loss, change model size, reduce PPO epochs, or remove validation to manufacture a speedup.

- Source baseline: `a8aa5c0`, branch `codex/fullgame-native-20260908`, worktree `/private/tmp/ai2-fullgame-20260908.LtE6Ou`.
- One running GPU maximum, cumulative campaign ceiling $12, current provider audit required before start/rental. At most two hours per authorized compute window with retrieval reserve and stop-only guard. Never destroy an instance.
- Policy execution and learning must use CUDA. Local configuration/unit tests and native game references are not training runs.
- W&B stays offline: earlier security review blocked credential transfer/publication. Do not retry either operation without specific authorization.
- Download and hash-verify each completed cloud block immediately. Retain failures and checkpoints. New durable destination: `AI_2/baselines/combat-update-20260909/`.

## Task 1 — opt-in update routing

Files: modify `benchmark_native_combat_training.py`, `tests/test_benchmark_native_combat_training.py`; add a focused CUDA integration test in `tests/test_native_combat_update_parity.py` if separate coverage improves clarity. Reuse `ppo_update_optimizations.py`; do not rewrite its objective.

- [ ] Run the existing driver test suite and record baseline status.
- [ ] Add failing tests for `BenchmarkConfig(update_mode='deferred_metrics')`, invalid choices, unchanged default `sb3`, parser propagation, and mode/source provenance.
- [ ] Add `update_mode` validation/CLI choices, default `sb3`; construct an opt-in TensorPPO subclass whose `train` invokes the existing deferred update on the actual tensor buffer. All-off must remain the original TensorPPO class, including checkpoint behavior.
- [ ] Include the update implementation's hash and source guard metadata in receipts. Keep game/player step denominators and timed boundaries unchanged.
- [ ] Verify the new routing tests and all original driver tests; run Ruff and diff checks.

## Task 2 — real combat update parity

- [ ] Add a CUDA-required test: collect actual native combat observations/rewards into the TensorRolloutBuffer; clone identical policy/optimizer and update-start RNG into plain/deferred learners; compare multiple updates.
- [ ] Require exact policy tensors, optimizer state, minibatch/sample ordering, final RNG, loss statistics, and update counts. Cover timeout/autoreset and source-version refusal; preserve existing KL early-stop tests, including their intentional synchronization.
- [ ] Ensure unavailable CUDA reports a skip, never a fabricated pass or CPU training fallback.
- [ ] Run the CUDA test on the rented host and the existing PPO optimization regression suite. Run instrumented diagnostic counts separately from headline timing.

## Task 3 — controlled GPU timing

- [ ] Audit instance ownership/status/price and remaining ceiling. Reuse selected 50337302 if it is available and suitable, otherwise resolve one replacement only within existing authority; do not destroy retained disks.
- [ ] Verify SSH, source hashes, CUDA tensor work, software versions, and offline utilization monitor before tests.
- [ ] Run both update modes sequentially, identical seeds/configs: 128 games/batch4096 and 2048 games/batch4096; also test 2048 games/batch16384 separately. Use two excluded warmups and at least ten measured rollouts, then reverse order for a repeat if budget permits.
- [ ] Compare initial/final parameter hashes and per-rollout losses for each same-config pair; label changed-batch arms as learning experiments, not semantics-preserving engine wins.
- [ ] Preserve per-rollout collection/update/wall times, actual optimizer steps, GPU utilization/memory/power, checkpoints, versions and code hashes. Report median/pooled measured rates plus individual runs; never pool different workloads.

## Task 4 — curriculum learning comparison

- [ ] Inspect the current staged curriculum and checkpoint contract in the existing curriculum worktree; record the selected next stage and its reset/reward/opponent/action/observation/evaluation definitions before porting.
- [ ] Write a separate faithful curriculum-bridge design and test matrix after that discovery. No raw-reward substitute, silent checkpoint replacement, altered promotion gate, or CPU training fallback.
- [ ] Prove original-vs-native curriculum parity before any learning comparison. Then run matched checkpoint/seed/rollout/epoch arms with different minibatches on CUDA; evaluate held-out original-game episodes and record XY, contact, attacks, landed damage, falls and timeouts.
- [ ] Judge by time to declared behavioral gates and retention, not steps/s or shaped reward alone. Report inconclusive if bounded runs do not establish a winner.

## Task 5 — review and closeout

- [ ] Independent spec review, then code-quality review; fix material findings and rerun affected tests.
- [ ] Recompute results from raw receipts, verify every local artifact checksum, preserve complete source and frozen oracle, and document any unqualified claims.
- [ ] Stop compute, independently verify provider state and cumulative cost, leave instances undestroyed and source isolated. Do not imply W&B is online.

## Verification commands

Run with the recovered pinned local Python for local checks and verified remote Python on CUDA:

```sh
python -B -m unittest discover -s tests -p test_benchmark_native_combat_training.py -v
python -B -m unittest discover -s tests -p test_native_combat_update_parity.py -v
python -B -m unittest discover -s tests -p test_ppo_update_optimizations.py -v
ruff check benchmark_native_combat_training.py tests/test_benchmark_native_combat_training.py tests/test_native_combat_update_parity.py
git diff --check
```

Cloud benchmark commands use `--backends native_cuda --games 128 --n-steps 128 --batch-size 4096 --n-epochs 8 --net-arch 512 512 256 --warmup-rollouts 2 --measured-rollouts 10 --update-mode sb3` and an otherwise identical `--update-mode deferred_metrics`, always with unique output paths and the hash-pinned frozen root.
