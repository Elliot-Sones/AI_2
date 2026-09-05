# Deletion readiness audit — instance 49902188

Status: **NOT READY TO DELETE**. No destruction was requested or performed.

The user approved a bounded archival-only restart. On 2026-09-05 UTC,
Vast responded: `Required resources are currently unavailable, state change queued.`
The CLI returned zero despite not starting the instance. Repeated status checks
reported `actual_status=exited`, `cur_state=stopped`, `intended_status=stopped`.
Direct SSH to the previously verified endpoint closed without access.
The queued restart was cancelled with a stop request; the lifecycle guard
then verified the same stopped state. See `session.json`.

## Local evidence

`local-manifest.json` records SHA256 and size for 390 existing baseline files,
11,058,739 bytes total, excluding Python bytecode and this audit directory.
It includes the original 4090, animation-cache, and retrieved pipeline results.
This verifies an inventory of local evidence, NOT completeness against remote
storage, and is not an independent off-device backup.

## Outstanding before deletion

- Retrieve `/workspace/AI_2/animation-cache-20260904/baselines/pipeline-profile-20260904/artifacts/natural_1env/pipeline_profile.json`.
- Inventory both remote experiment source trees, including source/assets,
  configs, logs, raw profiles, episode records, dependency snapshots and any
  newer checkpoints, TensorBoard or W&B output. Preserve these locally,
  excluding credentials, then compare remote/local checksums.
- Check existing tracking state and close out any applicable runs; no new
  training, monitor, or W&B run was launched during this archival attempt.
- Obtain explicit destruction confirmation after the archive is complete.

The instance remains retained. Last provider storage rate was approximately
$0.40/day. The guard's `estimated_running_cost_usd` is elapsed wall time times
the quoted running rate, not an invoice: the instance was never observed
running during this attempt. Historical GPU utilization/kernel traces cannot
be reconstructed if they were not recorded originally.
