# KL Diagnostics GPU Runbook - 2026-09-15

Purpose: replay the canonical contact checkpoint for four native rollouts and record PPO minibatch KL diagnostics without disabling the target-KL stop-before-step safeguard.

Canonical checkpoint SHA-256:

```text
301ce3c72924f3e5c62f265f88292b8952e86e6d71d9a0552213cb8f3c6784b6
```

Preflight:

```bash
test ! -e /workspace/AI_2/artifacts/kl-diagnostics/20260915-b8192-receipt.json
test ! -e /workspace/AI_2/artifacts/kl-diagnostics/20260915-b8192-minibatches.jsonl
test ! -e /workspace/AI_2/artifacts/kl-diagnostics/20260915-b16384-receipt.json
test ! -e /workspace/AI_2/artifacts/kl-diagnostics/20260915-b16384-minibatches.jsonl
```

Batch 8192:

```bash
python kl_diagnostics_runner.py \
  --checkpoint /workspace/AI_2/experiment-inputs/checkpoint/final.zip \
  --expected-sha256 301ce3c72924f3e5c62f265f88292b8952e86e6d71d9a0552213cb8f3c6784b6 \
  --output /workspace/AI_2/artifacts/kl-diagnostics/20260915-b8192-receipt.json \
  --trace /workspace/AI_2/artifacts/kl-diagnostics/20260915-b8192-minibatches.jsonl \
  --batch-size 8192 \
  --rollouts 4 \
  --max-diagnostics 512 \
  --device cuda
```

Batch 16384:

```bash
python kl_diagnostics_runner.py \
  --checkpoint /workspace/AI_2/experiment-inputs/checkpoint/final.zip \
  --expected-sha256 301ce3c72924f3e5c62f265f88292b8952e86e6d71d9a0552213cb8f3c6784b6 \
  --output /workspace/AI_2/artifacts/kl-diagnostics/20260915-b16384-receipt.json \
  --trace /workspace/AI_2/artifacts/kl-diagnostics/20260915-b16384-minibatches.jsonl \
  --batch-size 16384 \
  --rollouts 4 \
  --max-diagnostics 512 \
  --device cuda
```

Expected trace semantics:

- `pre_step_approx_kl` is the SB3 safeguard value measured before `zero_grad`, `backward`, and `optimizer.step`.
- `post_step_approx_kl` is populated only for rows where `optimizer_step_executed` is true.
- Early-stop rows keep `optimizer_step_executed=false` and `post_step_approx_kl=null`.
- The runner refuses existing output or trace paths before loading CUDA, the checkpoint, or the environment.
