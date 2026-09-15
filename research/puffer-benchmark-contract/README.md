# Puffer Benchmark Contract

This folder is the handoff contract for the Puffer lane. Do not compare AI2 and Puffer by copying a dashboard `SPS` number. A valid comparison needs a JSON receipt with raw counts, elapsed wall time, workload semantics, model/update geometry, correctness evidence, and quality evidence.

The Puffer producer must provide:

- Exact command, Puffer commit, AI2 commit, and CUDA/PyTorch/runtime versions.
- GPU model and count from the measured host.
- Concrete CPU model from the measured host. Sentinel values such as `unknown`,
  `unknown-cpu-model`, `n/a`, or `generic` are invalid because two unknown CPUs
  cannot prove a fair hardware comparison.
- Raw `agent_transitions` and `elapsed_seconds` from the measured wall-clock interval.
- Timing scope and included stages, including graph warmups and anything excluded from timing.
- Async overlap evidence from a CUDA trace or event timeline when rollout/update overlap is claimed.
- Actor staleness and replay/update semantics.
- Environment parity evidence against AI2, including the actual game semantics
  SHA-256 hash as a 64-character hex digest, observation/action contracts, and
  tick semantics.
- Model architecture, parameter count, precision, rollout geometry, minibatches, PPO epochs, and sample reuse.
- Quality evidence for fixed-transition and fixed-wall-clock gates in the original AI2 game.
- Non-empty source, command, and runtime provenance under both `system.software`
  and `workload.source_provenance`. Runner commands and commits only need to be
  present; they are not required to equal AI2's values for throughput comparison.

Use `puffer-receipt.template.json` as a starting point. Fields left as `null` are intentionally invalid: validation should fail until the producer replaces every placeholder with measured evidence.

Useful commands:

```bash
python training_benchmark_contract.py validate RECEIPT.json
python training_benchmark_contract.py compare ai2.json puffer.json --output comparison.json
```

The comparison result can show a throughput speedup only when the workload is comparable. Promotion remains blocked until parity and quality gates are true.
