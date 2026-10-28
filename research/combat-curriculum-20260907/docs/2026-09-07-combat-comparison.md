# Combat Comparison Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development for assigned independent slices; the root owns integration and cloud lifecycle.

**Goal:** Produce an identity-verified GPU combat baseline and replay videos for Medium-145 and Large-145 without altering trained weights.

**Architecture:** A reusable SB3 opponent adapter supplies independent, zero-padded four-frame inputs to CUDA models. A separate comparison driver controls both original-game slots, records honest terminal outcomes, and replays selected matches. Existing frozen game files remain untouched.

**Tech Stack:** Python3.11, unittest, NumPy, SB3/PyTorch CUDA, frozen Pymunk/Pygame game, FFmpeg, SSH/rsync and Vast CLI.

## Task 1 — Opponent adapter (assigned implementer)

Files: create combat_policy_adapter.py and tests/test_combat_policy_adapter.py only.

- [x] Write failing tests for zero-padded reset, oldest-to-newest stacking, independent player history, tuple-unpacked clipped finite actions, shape rejection, missing checkpoint failure and CUDA enforcement.
- [x] Run /private/tmp/ai2-baseline.XWm6Cd/venv/bin/python -m unittest discover -s tests -p test_combat_policy_adapter.py -v and preserve the expected red result.
- [x] Implement SB3CombatPolicy(model, deterministic=True, require_cuda=True), reset(), get_env_info(env), predict(raw_obs), and load_cuda_checkpoint(path). All actual checkpoint inference must require CUDA. For example, the first input must satisfy np.array_equal(stacked[:192], np.zeros(192)) and np.array_equal(stacked[-64:], obs).
- [x] Compare the real stack helper with installed SB3 StackedObservations, including a fresh episode after prior history; run green tests.
- [x] Root checks specification compliance, then independent code review checks interface and failure behavior.

## Task 2 — Honest match runner (root)

Files: create combat_comparison.py and tests/test_combat_comparison.py.

- [x] Write failing unit tests for schedule cardinality320, paired seeds/slots, strict timeout classification, simultaneous elimination, candidate-relative metrics, finite episode data and deterministic trace digests.
- [x] Implement pure scheduling/outcome helpers and a lazy-import runtime loading the frozen original game via --repo. Check all inputs before starting cases, seed Python/NumPy/Torch per episode, use independent adapters, and record exact executed actions/state.
- [x] Write JSON output atomically per completed match and refuse clobbering a nonempty output path. Include requested/actual device and checkpoint/source hashes.
- [x] Implement a single selected-case render mode using normal-speed30fps video; compare replay outcomes and trace digest to saved JSON before calling it valid.
- [x] Run new unit tests and existing light baseline suites. Tests do not train or infer real models on CPU.

## Task 3 — Cloud qualification and bounded baseline (root)

- [x] Prepare source inventory and checksum candidate/legacy checkpoint files locally. Reuse already-present identical remote files where validated.
- [x] Read existing guard/monitor contracts, adapt only run-specific notes and paths; persist an explicit source/run/deadline/cost manifest locally before startup.
- [x] Fresh-audit exact retained instance and credit, start it only when prepared, verify SSH host key/runtime/CUDA and actual tensor operation.
- [x] Start scalar-only W&B hardware monitoring locally; upload only run source/checkpoints/tests and verify remote hashes.
- [x] Run CUDA smoke and original-game adapter parity before the320-match panel. Check model identities, real nonconstant actions and terminal metrics.
- [x] Retrieve each completed match promptly; verify hashes/readability and retain partial results on any interruption.
- [x] Render the four preselected replays, verify equivalence, download and fully decode them.

## Task 4 — Review and closeout (root plus independent reviewer)

- [x] Review outcome accounting, paired schedule, no-fallback/device enforcement, source/checkpoint immutability and replay proof.
- [x] Rehash local artifacts against remote manifest, preserve source archive and reproducible commands in baselines/combat-comparison-session12, summarize model results with limitations.
- [x] Close hardware monitor, stop only instance50149729 compute, verify stopped status, retain the disk.
- [x] Report which checkpoint is the better observed starting point; neither navigation nor a small combat baseline alone proves human-ready skill.

## Verified closeout

2026-09-07:34focused tests passed locally and on the4090;4CUDA smoke cases passed;320paired matches completed in666.0247seconds;239,735game ticks and every full trace verified;4preselected replays matched exact traces and fully decoded at30fps. All678remote artifact files(74,708,342bytes)matched locally. Independent code/result reviews approved. Large won106/160 vsMedium61/160. All checkpoints unchanged. W&B monitor3hfvut11 finished; provider reports50149729exited/stopped, disk retained. Main-worktree code was not edited. Baseline project benchmark/monitor tests remained environment-blocked; no full-suite passing claim is made. Durable evidence: /Users/elliot18/Projects/ml-research/AI_2/baselines/combat-comparison-session12/.
