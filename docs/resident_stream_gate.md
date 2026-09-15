# Resident CUDA Stream Gate

This gate verifies resident full-game native CUDA replay against the CPU/native
oracle on a non-default PyTorch CUDA stream. It fails closed: any mismatch,
missing CUDA runtime, missing replay support, stale source hash, incomplete
forced terminal/autoreset evidence, or missing observation/reward/done evidence
returns a non-zero exit code.

Run on a CUDA host from the repository root:

```bash
python -B fixed_action_trace_parity.py \
  --frozen-root . \
  --backends python,native_cpu,native_cuda \
  --actions 0,1,2,3,4,5,6,7,8,9 \
  --seeds 101,102 \
  --horizon 12 \
  --output artifacts/resident-stream-gate-$(date -u +%Y%m%dT%H%M%SZ)
```

Expected artifacts:

- `trace-report.json` - exact case matrix, blockers, CPU/CUDA parity evidence,
  forced termination evidence, and forced time-limit truncation/autoreset
  evidence.
- `resident-stream-gate-manifest.json` - command, artifact paths, backend
  controls, explicit CUDA skip/block status, and SHA256 hashes for the parity
  runner and native full-game lane sources.

Passing promotion requires `promotion_status: "passed"`,
`native_cuda_nondefault_stream_exact: true`, and
`native_cuda_observation_checks_exact: true`. A local host without CUDA must
produce a blocked manifest status such as `blocked_no_cuda`; that is an explicit
skip/block receipt, not a passing CUDA result.
