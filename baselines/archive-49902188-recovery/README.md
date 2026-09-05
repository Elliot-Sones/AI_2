# Stopped-instance recovery attempt

Instance 49902188 remains exited/stopped; no start, stop or destroy commands
were issued during this attempt. No benchmark files were downloaded.

Vast execute successfully listed /workspace/AI_2 and measured its directories:
original experiment approximately 22 MB, animation-cache experiment 31 MB,
and virtual environment 668 MB (not selected for download).

The outstanding file is confirmed present, 2,692,417 bytes:
/workspace/AI_2/animation-cache-20260904/baselines/pipeline-profile-20260904/artifacts/natural_1env/pipeline_profile.json

The installed Vast CLI's copy operation resolved the host transfer endpoint
38.246.237.140:32498. Authentication with the existing ed25519 identity
succeeded far enough to reach rsync, but both experiment downloads failed:
`@ERROR: Unknown module '49902188'`. Retrying after initialization also failed.
Logs are stored alongside download.py. The wrapper removes sudo and uses
accept-new host-key checking instead of disabling checking; changed host keys
remain rejected. It excludes credentials and caches and never deletes files.

An upstream report has the same error:
https://github.com/vast-ai/vast-cli/issues/326
Its normal-container-SSH workaround requires a running instance and is not
available while this GPU is occupied.

## Support request draft (not sent)

Please help recover data from stopped Docker instance 49902188 on host
38.246.237.140. GPU restart is unavailable because its resources are occupied.
`vastai execute 49902188 'ls -la /workspace/AI_2/'` works and the data exists,
but `vastai copy` using your returned rsync endpoint 38.246.237.140:32498 fails
with `@ERROR: Unknown module '49902188'` after SSH authentication.
Please restore the instance's rsync export or provide a supported export of
/workspace/AI_2/benchmark-20260904b/ and
/workspace/AI_2/animation-cache-20260904/.
Please preserve the instance and its files; do not destroy it.

An alternative is Cloud Sync to a user-approved connected storage destination;
none was selected or configured during this attempt.
