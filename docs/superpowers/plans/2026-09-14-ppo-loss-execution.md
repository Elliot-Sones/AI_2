# PPO Loss Execution Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for bounded implementation and independent review. Keep all changes in the existing isolated worktree.

**Goal:** Test whether recording the PPO loss forward/backward or compiling its tensor math improves native-GPU end-to-end training without removing learning work or weakening safeguards.

**Architecture:** The user approved broader GPU replay and kernel fusion after the checked-validation experiment proved inconclusive. Keep the already-graphed MLP and unchanged eager Normal validation in every control/candidate. Inject a callable into the existing `_experimental_loss_function` seam; do not capture the optimizer, policy validation, or the target-KL decision. The CUDA-graph candidate records the original loss arithmetic; the Inductor candidate uses the existing `compile_ppo_loss`. Production defaults remain unchanged.

**Tech Stack:** Python, PyTorch 2.5.1+cu124 on RTX4090, SB3 2.5.0, existing native CUDA game and TensorPPO collector. Local CPU tests validate interfaces/algebra only, never training speed.

## Fixed acceptance contract

- Native fullgame, 2048 games / 4096 player slots, 128 ticks, batch16384, 8 epochs, FP32/TF32off, actor and critic [512,512,256]. No reduced passes, changed model, changed precision or changed game.
- Preserve distribution rejection and target-KL stop before backward/Adam. Do not silently remove validation as a capture workaround.
- Exact graph-loss outputs/gradients/weights/Adam/RNG/metrics are required on replay tests. Compilation is numerically experimental: report max differences, never lower the existing exact-comparison gate or claim equivalent learning from tolerance alone.
- Detect unsupported shape/dtype/stride/grad/signature/scalar changes; no silent eager fallback or undisclosed recapture. Exclude capture/compile warmup from clean time and record startup separately.
- Compare only source-matched sequential runs. Short screening uses seeds149/150/151 with alternating order and at least16 measured rollouts. A qualifying candidate then needs128 measured rollouts per side; at least5% pooled short gain and a positive long confirmation before any speedup recommendation. Negative or insufficient evidence leaves defaults unchanged.

## Budget and cloud boundary

Fresh read: account empty, credit10.692771892364078, effective historical accounting anchor22.32106959549408; conservative campaign spending11.628297703130002 of12, leaving0.3717022968699979. Do not reinterpret account credit as new authority. One instance only, guard installed before create, at most30minutes with compute+disk at most0.40/hour and0.15 transfer/headroom reserve (maximum0.35). Recheck live offers/credit before creation. No paid run until source/tests/qualification commands are ready. If the required experiment cannot fit safely, finish local work and report the budget blocker rather than overrun or claim GPU verification. Download each block immediately. W&B monitor and exact-ID teardown after verified local artifacts are required by existing authorization.

## Task 1 — Source/trace and official documentation review

- [ ] Read existing loss, MLP capture, native driver and comparator; inspect saved control trace without summing overlapping CPU/GPU durations.
- [ ] Check official pinned PyTorch capture/compile restrictions, including nondifferentiable outputs and global validation state.
- [ ] Record actionable attribution and distinguish observed facts from future speed hypotheses.

## Task 2 — Opt-in loss execution adapters (executor-owned)

Files: create `research/update-profile-20260913/optimization-research/loss_execution.py` and `test_loss_execution.py`.

- [ ] Write failing tests for the callable protocol (`adapter(*ppo_loss_arguments)`), metadata, default unchanged, rejection of unsupported modes/signatures, and preservation of Distribution validation on failure.
- [ ] Run `python3 -m unittest discover -s research/update-profile-20260913/optimization-research -p test_loss_execution.py -v`; require intended red failures.
- [ ] Implement a lazy CUDA loss graph around the original `ppo_loss`, with detached cloned tensor samples, captured scalar configuration, replay signature guards, capture timing and preserved RNG. Preserve differentiable loss outputs and nondifferentiable metric outputs; reuse PyTorch's graph APIs rather than hand-written gradients.
- [ ] Implement the separate Inductor wrapper using existing `compile_ppo_loss`, preserving global validation state through construction and lazy invocation. Report compile/startup metadata and errors; no fallback.
- [ ] Test eager algebra, gradients, metadata and failure cleanup locally; add CUDA-only exact replay/gradient and signature tests which explicitly skip without CUDA.

## Task 3 — Driver and qualification integration (root-owned)

Files: modify only research `update_experiments.py`, `run_gpu_block.py`; add research `test_loss_experiment_dispatch.py`, `qualify_loss_execution.py`, and qualification tests.

- [ ] Add failing parser/dispatch tests: `--loss-mode eager|cuda-graph|inductor`; non-eager only with graph arm; reject validation bypass/coalesced candidate combinations and incompatible parity mode.
- [ ] Inject adapter through existing `_experimental_loss_function` and record mode, source hashes, compile/capture costs/counters. Preserve unchanged control commands and source/config equivalence.
- [ ] Add isolated replay qualification comparing one starting state and actual collected native rollout; verify outputs, gradients/parameters, optimizer work/state, RNG and minibatch order. Test NaN actions/scale and forced KL stop before mutation; invalidate flattened action cache when injecting faults.
- [ ] Save receipts on passing and failing runs; never treat a local algebra test as CUDA proof.

## Task 4 — Verify, review, then bounded GPU gate

- [ ] Run existing128 optimization tests, new tests, related loss/MLP suites and32 lifecycle tests. Critical Ruff check E9,F63,F7,F82. Independently review changed files for safety and accurate measurement scope.
- [ ] Assemble and hash a new source snapshot without overwriting previous bundles/receipts.
- [ ] If cost plan fits, rent at most one instance, verify pinned runtime and W&B, execute CUDA qualification before clean comparisons. Stop on correctness failure; preserve negative evidence too.
- [ ] Run only qualified candidates, use separate diagnostic profiles, then matched longer confirmation when budget/time reserves permit. Do not merge unequal-numerics results into exact claims.
- [ ] Download/verify all outputs, finish W&B, stop and destroy the now-unneeded exact instance, confirm inventory empty, report actual findings and remaining verification gaps.
