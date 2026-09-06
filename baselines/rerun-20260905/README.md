# One-environment profile rerun — 2026-09-05

Completed and downloaded locally before stopping instance **49976003**.
Instance was verified stopped, not destroyed. No deletion is authorized.

## Result

- RTX 4090, Ryzen 9 7950X3D host, advertised 16 effective CPU cores.
- Phase 0d, one environment, fresh policy, seed 42, one Torch thread.
- 8,192 warmup transitions; two measured rollouts of 8,192 transitions each.
- Batch size 8,192, eight epochs per update.
- **16,384 measured transitions in 10.338770 seconds: 1,584.7146 steps/s.**
- PPO updates: 0.276518 seconds total within that measurement.
- Seven profiler/worker/guard tests passed through unittest in 0.015 seconds.
- This is a short instrumented diagnostic, not sustained training throughput.

## Local evidence and integrity

- `artifacts/natural_1env/pipeline_profile.json`: raw timing/event data.
- `artifacts/`: setup and failed-attempt logs, final tests, profiler output,
  CUDA check, hardware details, package freeze and monitoring logs.
- `wandb/`: offline monitoring runs, not uploaded or synced to W&B.
- `source.tar.gz`: exact uploaded source/config/assets snapshot.
- `results.tar.gz`: downloaded remote artifacts and offline monitoring archive.
- `verification.json`: profile invariants plus per-file SHA256 inventory.
- `lease.json`: instance lifecycle and verified stop evidence.

Source archive local/remote SHA256:
`696e74ae0aee77472f4b0cce3656a0b39d9a9c011be14124ff27579660a713f5`

Results archive local/remote SHA256:
`93123cb89ebc7ff00648d805ce2ed6b4debdd5f9b226c9c013092746367c9341`

## Reproduction and setup differences

The original 4090 host was destroyed by the user. This run uses a different CPU
and host and cannot establish a speedup over the old profile. Game engine and
benchmark script hashes match the old diagnostic. Current train/config include
tracking additions, and several supporting dependency versions differ; the full
freeze and raw source hashes are retained. Torch 2.4.1+cu121, SB3 2.5.0,
Gymnasium 1.0.0 and Pymunk 6.2.1 were used.

Run from `/workspace/AI_2/rerun-20260905` using the isolated interpreter
`/workspace/AI_2/venv/bin/python`. The final command was the local `run.sh`
sent to `bash -s` over SSH. The source archive contains its earlier pytest
version; the local final script uses unittest. No gameplay code was changed.

Setup corrections, all applied before successful measurement:

1. Added `gcc` and `libc6-dev` because Pymunk's source build lacked a compiler.
2. Used unittest for the same seven tests after pytest crashed importing the
   image's native readline module. This avoids the crash; it does not repair
   the underlying native-module incompatibility.
3. Removed generated `._*` AppleDouble metadata from the extracted game tree.
   These files remain recoverable in source.tar.gz; exclude Mac metadata in
   future source packages.
4. Created `assets -> UTMIST-AI2-main/environment/assets` in the run root.
   Without this expected path the game tried gdown's incompatible download
   API. No external asset download was needed for the successful run.

## Monitoring and cost

W&B monitoring ran **offline only**. There is no live dashboard URL. The user
asked whether W&B was set up; an optional question about enabling live upload
remained unanswered, so no credentials or monitoring records were sent to W&B.

Whole rental session: about 29.1 minutes, mostly provider initialization and
package installation. Quoted-rate elapsed-time estimate is approximately $0.16,
not a reconciled invoice; transfers and retained storage are separate.
Retained 20 GB disk at quoted $0.20/GB/month is about $0.13/day, assuming a
30-day month. The compute instance is stopped. User confirmation is required
before destruction. This archive is local only, not independently backed up.
