# Native B2 Re-engagement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, locally verify, CUDA-qualify, and run the first measured cycle of the independently versioned `bridge_b2_reengagement-v1` curriculum without changing the frozen B1 contact lesson.

**Architecture:** B2 is a sibling of B1 with its own Python oracle, C reference ABI, CPU/CUDA lane runtime, tensor environment, evaluator, qualification receipt, mixed `R4E1N` trainer, and campaign controller. The existing shared PPO keeps task ID `combat == 1`, but gains a model-owned expected-version map so B1 and B2 records can never mix silently. Local tests establish the state-machine and CPU parity contracts before one labeled RTX 4090 is rented for CUDA parity, collector/update smoke, and the first immutable campaign cycle.

**Tech Stack:** Python 3, pytest/unittest, C99, CUDA C++17, PyTorch, Gymnasium, SB3/PPO, Vast.ai, Weights & Biases

---

## File Structure

New production files:

- `reengagement_curriculum.py` — frozen-original-game B2 behavioral oracle.
- `native_reengagement_episode.py` — ctypes ABI and native CPU reference wrapper.
- `native/reengagement_episode.h` — shared host/device B2 state machine.
- `native/reengagement_episode.c` — exported CPU reset/step/ABI functions.
- `native_reengagement_lane_vec_env.py` — finite scenario schedule, native backend, tensor environment, and lossless episode projection.
- `native/reengagement_lanes.h` — lane/episode-record ABI and shared autoreset logic.
- `native/reengagement_lanes.c` — CPU lane runtime exports.
- `native/reengagement_lanes.cu` — CUDA kernels and stream-aware exports.
- `evaluate_reengagement_curriculum.py` — pinned trace-first original-game evaluator.
- `qualify_native_reengagement.py` — oracle/native-CPU/CUDA qualification driver and receipt.
- `train_reengagement_campaign_block.py` — one immutable 4,096-lane `R4E1N` train block.
- `continue_reengagement_campaign.py` — cycle/evaluation/preservation controller.

Shared files changed narrowly:

- `native_curriculum_ppo.py` — explicit expected task-version contract; B1 remains the default.
- `native_curriculum_runner.py` — explicit `native_reengagement` dispatch; `native_contact` remains unchanged.

New or extended tests:

- `tests/test_reengagement_curriculum.py`
- `tests/test_native_reengagement_episode.py`
- `tests/test_native_reengagement_lanes.py`
- `tests/test_evaluate_reengagement_curriculum.py`
- `tests/test_qualify_native_reengagement.py`
- `tests/test_train_reengagement_campaign_block.py`
- `tests/test_continue_reengagement_campaign.py`
- `tests/test_native_curriculum_ppo.py`
- `tests/test_native_curriculum_runner.py`

Frozen regression files are not repurposed: `native_contact_episode.py`, `native_contact_lane_vec_env.py`, `native/contact_*`, and `benchmark_native_contact_training.py`.

## Frozen B2 Codes and Ordering

The native/Python ABI uses eight reward terms in this exact order:

```python
REWARD_NAMES = (
    "first_hit",
    "pursuit",
    "second_hit",
    "completion",
    "stock_loss",
    "unsafe_support",
    "deadline",
    "tick",
)
```

Outcome codes are frozen before implementation:

```python
OUTCOMES = (
    None,
    "success",
    "own_stock_loss",
    "opponent_stock_loss",
    "unsafe_support",
    "response_deadline",
    "second_hit_deadline",
    "total_deadline",
)
```

Every step applies events in this order: raw tick, evidence validation, hit and pursuit attribution, stock failures, clean-boundary support/success, relocation, response deadline, second-hit deadline, total deadline, reward/observation projection. Positive events and the ordinary `-0.001` tick term remain additive on terminal ticks.

## Task 1: Lock the Original-Game B2 Oracle

**Files:**

- Create: `tests/test_reengagement_curriculum.py`
- Create: `reengagement_curriculum.py`
- Read-only reference: `baselines/staged-training-20260920/remote-closeout/inputs/reference/combat_curriculum.py`
- Read-only game: the frozen original-game source selected by the test fixture

- [ ] Write failing tests for module/version/schema presence.

```python
def test_contract_identity_is_frozen():
    assert curriculum.CURRICULUM_VERSION == "bridge_b2_reengagement-v1"
    assert curriculum.BANK_VERSION == "b2-reengagement-bank-v1"
    assert curriculum.REWARD_NAMES == (
        "first_hit", "pursuit", "second_hit", "completion",
        "stock_loss", "unsafe_support", "deadline", "tick",
    )
```

- [ ] Write failing tests for scenario validation and deterministic reset across both learner slots and directions.
- [ ] Write failing tests for the exact relocation rule: both anchors, left-only, right-only, neither valid, and exact-tie direction choice.
- [ ] Write failing tests proving relocation preserves every learner field and the target's stocks, current damage, cumulative damage dealt, and cumulative damage taken.
- [ ] Write failing controller tests: idle, held-light, one-hit-then-idle, repeated first attack ID, movement without second hit, walkoff, and opponent self-loss must fail.
- [ ] Write a scripted hit-resolve-pursue-hit-resolve success test for both slots and directions.
- [ ] Write boundary tests for pursuit elapsed ticks 60/61, second-hit elapsed ticks 90/91, and total ticks 300/301.
- [ ] Run the new test once and confirm the import failure.

Run:

```bash
python -m pytest tests/test_reengagement_curriculum.py -q
```

Expected: FAIL because `reengagement_curriculum` does not exist.

- [ ] Implement the oracle using a private RNG scope and the frozen original game.

The relocation helper must have an independently testable pure decision boundary:

```python
def choose_target_anchor(*, floor_left, floor_right, target_half_width,
                         learner_x, direction):
    anchors = (
        floor_left + target_half_width + 0.2,
        floor_right - target_half_width - 0.2,
    )
    valid = [x for x in anchors if abs(x - learner_x) >= 2.0]
    if not valid:
        return None
    gaps = {x: abs(x - learner_x) for x in valid}
    best = max(gaps.values())
    tied = [x for x in valid if gaps[x] == best]
    return max(tied) if direction == 1 else min(tied)
```

The implementation must restore module-global Python and NumPy RNG states after every reset/step call, and must project a structured trace sufficient to reconstruct success without trusting the environment's label.

- [ ] Run the oracle tests until green.
- [ ] Run the existing B1 Python curriculum tests to ensure no shared behavior changed.

Run:

```bash
python -m pytest tests/test_reengagement_curriculum.py tests/test_combat_curriculum.py -q
```

Expected: all selected tests pass.

## Task 2: Implement the Native CPU B2 Episode ABI

**Files:**

- Create: `tests/test_native_reengagement_episode.py`
- Create: `native_reengagement_episode.py`
- Create: `native/reengagement_episode.h`
- Create: `native/reengagement_episode.c`
- Regression: `tests/test_native_contact_episode.py`

- [ ] Write the ctypes ABI test first. It must assert field sizes/order, eight reward terms, outcome codes, invalid-action rejection, invalid-scenario rejection, post-terminal rejection, and corrupt-state rejection.
- [ ] Add differential tests that feed the exact same scenarios, actions, and recorded raw-game random draws to the original-game oracle and native CPU reference.
- [ ] Assert exact agreement for float32 observations, rewards, reward components, terminal flags, attack IDs, relocation snapshots, pursuit metrics, outcomes, and reset state.
- [ ] Run once and observe missing-module failure.

Run:

```bash
python -m pytest tests/test_native_reengagement_episode.py -q
```

Expected: FAIL at import.

- [ ] Implement `native_reengagement_episode.py` by defining `Scenario`, `PlayerSnapshot`, `Diagnostics`, `State`, `StepResult`, `scenario_from_seed`, `compile_reference`, `diagnostics_projection`, and `ReengagementReference` with B2-specific names and symbols.
- [ ] Implement the native ABI with explicit constants:

```c
#define AI2B2_REWARD_TERMS 8
#define AI2B2_TIME_LIMIT 300
enum {
    AI2B2_ONGOING = 0,
    AI2B2_SUCCESS = 1,
    AI2B2_OWN_STOCK_LOSS = 2,
    AI2B2_OPPONENT_STOCK_LOSS = 3,
    AI2B2_UNSAFE_SUPPORT = 4,
    AI2B2_RESPONSE_DEADLINE = 5,
    AI2B2_SECOND_HIT_DEADLINE = 6,
    AI2B2_TOTAL_DEADLINE = 7
};
```

- [ ] Preserve the learner byte-for-byte across relocation. Recreate only the target, restoring target stocks, current damage, `damage_done`, and `damage_taken_total` after initialization.
- [ ] Store nonzero `first_attack_id` and require nonzero `second_attack_id != first_attack_id`; repeated damage frames from either attack are not new hits.
- [ ] Make tick-60 pursuit and tick-90 second-hit eligible before their respective deadline checks. Process tick 300 normally and never permit tick 301.
- [ ] Run B2 native episode tests until green.
- [ ] Run frozen B1 regression tests.

Run:

```bash
python -m pytest tests/test_native_reengagement_episode.py tests/test_native_contact_episode.py -q
```

Expected: all selected tests pass with no B1 source changes.

## Task 3: Implement CPU and CUDA B2 Lanes

**Files:**

- Create: `tests/test_native_reengagement_lanes.py`
- Create: `native_reengagement_lane_vec_env.py`
- Create: `native/reengagement_lanes.h`
- Create: `native/reengagement_lanes.c`
- Create: `native/reengagement_lanes.cu`
- Regression: `tests/test_native_contact_lanes.py`

- [ ] Write failing lane tests for deterministic lane-major schedules, seed exclusion, 4-frame stacks, terminal observation before autoreset, exact-once drain records, schedule exhaustion, counter overflow, and B2 metadata.
- [ ] Assert the training schedule contains no seeds in `134000..134019` or `144000..144099` and writes every chosen scenario into `scenarios.json`.
- [ ] Assert completed records contain `curriculum_task == "combat"`, `curriculum_version == "bridge_b2_reengagement-v1"`, the existing combat control mask, and the full `reengagement` diagnostics payload.
- [ ] Add a CUDA parity test marked skip only when CUDA is unavailable. It must run scripted, adversarial, and seeded-random streams.
- [ ] Run once and observe missing-module failure.

Run:

```bash
python -m pytest tests/test_native_reengagement_lanes.py -q
```

Expected: FAIL at import.

- [ ] Implement the finite schedule and tensor adapter using the same owned-tensor API as B1, but distinct C symbols and source hash.

```python
CURRICULUM_VERSION = "bridge_b2_reengagement-v1"
SCHEDULE_GENERATOR = "b2-reengagement-vector-env-gym-seeds-v1"
EVALUATION_RANGES = (range(134000, 134020), range(144000, 144100))
```

- [ ] Implement shared lane autoreset logic in `reengagement_lanes.h`, CPU exports in `.c`, and stream-aware CUDA kernels in `.cu`. Per-step data must remain resident; only completed episode records drain once per rollout.
- [ ] Run B2 lane tests and B1 lane regressions.

Run:

```bash
python -m pytest tests/test_native_reengagement_lanes.py tests/test_native_contact_lanes.py -q
```

Expected: CPU tests pass; CUDA-only parity is skipped on the Mac with an explicit reason.

## Task 4: Make PPO Lesson Version Explicit Without Changing B1

**Files:**

- Modify: `tests/test_native_curriculum_ppo.py`
- Modify: `native_curriculum_ppo.py`

- [ ] Add failing tests proving default construction accepts B1 and rejects B2, explicit B2 construction accepts B2 and rejects B1, task switching preserves the expected version, save/load preserves it, and a wrong-version drained record fails before optimizer update.
- [ ] Run only those new tests and confirm failure against the global B1-only mapping.
- [ ] Add a copied default version map owned by each model:

```python
DEFAULT_TASK_VERSIONS = {
    "navigation": NAVIGATION_VERSION,
    "combat": CONTACT_VERSION,
}

def normalized_task_versions(expected_task_versions=None):
    versions = dict(DEFAULT_TASK_VERSIONS)
    if expected_task_versions:
        versions.update(expected_task_versions)
    return versions
```

- [ ] Thread `expected_task_versions` through `__init__`, `from_curriculum`, `load_contact_checkpoint`, and `load`; store it in save data; default old checkpoints/callers to B1.
- [ ] Update environment validation, CUDA requirement, task switching, and rollout-drain validation to use `self.task_versions[task]`.
- [ ] Keep task IDs, critic selection, masks, optimizer state, and B1 constants unchanged.
- [ ] Run focused and regression suites.

Run:

```bash
python -m pytest tests/test_native_curriculum_ppo.py tests/test_benchmark_native_contact_training.py -q
```

Expected: all selected tests pass; old checkpoints load with B1 defaults.

## Task 5: Build the Immutable Mixed `R4E1N` Train Block

**Files:**

- Create: `tests/test_train_reengagement_campaign_block.py`
- Create: `train_reengagement_campaign_block.py`
- Read-only template: `train_retention_campaign_block.py`

- [ ] Write failing configuration tests that freeze `R4E1N`, 4,096 lanes, 512 steps, batch 16,384, four epochs, learning rate `3e-5`, clip `0.1`, target KL `0.01`, entropy `0.001`, FP32, and no CPU fallback.
- [ ] Write mocked block tests for the exact schedule `("reengagement",) * 4 + ("navigation",)`, homogeneous rollout metadata, task switching on one model, counter continuity, optimizer preservation, lossless gzip episode rows, source hashes, and immutable output rejection.
- [ ] Require the exact parent SHA and a verified preservation acknowledgement before opening CUDA or loading a model.
- [ ] Run once and observe missing-module failure.
- [ ] Implement the runner from the existing retention block's artifact and failure-preservation machinery, with B2 passed as the expected combat version.

```python
SCHEDULES = {
    "R4E1N": ("reengagement", "reengagement", "reengagement",
               "reengagement", "navigation"),
}
PARENT_SHA256 = "d6e487c22f73913cd88a428683c7ca721e23b1f9cb5a3a561ad24d1f5bf410db"
PARENT_LIFETIME_TRANSITIONS = 60_391_424
```

- [ ] Mark a completed train block as `trained_not_evaluated`; it cannot promote itself.
- [ ] Run focused tests.

Run:

```bash
python -m pytest tests/test_train_reengagement_campaign_block.py tests/test_train_retention_campaign_block.py -q
```

Expected: all selected tests pass.

## Task 6: Add Versioned Conductor Dispatch

**Files:**

- Modify: `tests/test_native_curriculum_runner.py`
- Modify: `native_curriculum_runner.py`

- [ ] Add failing tests that `native_contact` still invokes the old B1 module, `native_reengagement` invokes the new B2 block, and neither accepts phase-owned checkpoint/output overrides.
- [ ] Add `native_reengagement` to supported block types and replace the hard-coded train function with an explicit dispatch table.

```python
TRAIN_BLOCKS = {
    "native_contact": "benchmark_native_contact_training",
    "native_reengagement": "train_reengagement_campaign_block",
}
```

- [ ] Preserve every existing B1 parser argument and guard.
- [ ] Run the conductor suite and B1 regression.

Run:

```bash
python -m pytest tests/test_native_curriculum_runner.py tests/test_benchmark_native_contact_training.py -q
```

Expected: all selected tests pass.

## Task 7: Implement Pinned Trace-First Evaluation

**Files:**

- Create: `tests/test_evaluate_reengagement_curriculum.py`
- Create: `evaluate_reengagement_curriculum.py`
- Read-only template: `baselines/staged-training-20260920/remote-closeout/inputs/reference/evaluate_combat_curriculum.py`

- [ ] Write failing tests for development seeds `134000..134019`, confirmation seeds `144000..144099`, identical deterministic/sampled scenarios, alternating slots, pairwise direction formula, unique case IDs, and bank SHA recording.
- [ ] Write reconstruction tests that reject success labels unless the trace proves two distinct nonzero attack IDs, a clean relocation, qualified pursuit, second hit, and safe resolution.
- [ ] Write summary tests separating modes and slots and reporting full sequence, first hit, pursuit, second hit, safe resolution, falls, deadlines, damage, and timing.
- [ ] Implement policy loading/inference on CUDA and original-game stepping on the authoritative frozen source. Evaluation may use host game stepping, but policy inference must not silently fall back from requested CUDA.
- [ ] Write immutable per-episode JSON/traces, bank manifest/hash, summary, environment inventory, command receipt, checkpoint hash, and source hashes.
- [ ] Run focused tests.

Run:

```bash
python -m pytest tests/test_evaluate_reengagement_curriculum.py -q
```

Expected: all selected tests pass.

## Task 8: Implement Three-Way Qualification

**Files:**

- Create: `tests/test_qualify_native_reengagement.py`
- Create: `qualify_native_reengagement.py`

- [ ] Write failing tests for a qualification manifest containing Python-oracle/native-CPU exact parity, native-CPU/CUDA parity, collector smoke, source hashes, compiler/CUDA/PyTorch inventory, scenario hashes, and a fail-closed overall result.
- [ ] Write tests that any mismatch, skipped required CUDA stage on a GPU host, seed overlap, wrong B2 version/mask, or ambiguous trace makes qualification fail.
- [ ] Implement local stages so the Mac can prove oracle/native-CPU parity and record CUDA as pending rather than passed.
- [ ] Implement the GPU stage using the real tensor collector and one learner update, checking finite losses, counter increments, optimizer steps, version-pure episode rows, and unchanged B1 behavior.
- [ ] Run local qualification tests.

Run:

```bash
python -m pytest tests/test_qualify_native_reengagement.py -q
```

Expected: all local tests pass; actual CUDA proof remains pending until Task 12.

## Task 9: Implement Campaign Continuation and Promotion Gates

**Files:**

- Create: `tests/test_continue_reengagement_campaign.py`
- Create: `continue_reengagement_campaign.py`
- Read-only template: `continue_retention_campaign.py`

- [ ] Write failing tests for parent matched-bank reruns before update, preservation before downstream use, exactly one cycle at a time, no overwrite, and no confirmation-bank consumption before two qualifying development checkpoints.
- [ ] Encode development mastery: two distinct increasing counters; deterministic and sampled each at least 18/20; at most one learner fall; no invalid/ambiguous evidence.
- [ ] Encode confirmation: each mode at least 90/100, at most five falls, and each learner slot at least 40/50.
- [ ] Encode retention alarms: pause on more than two lost successes or more than two added falls out of 20 versus matched parent; confirmation permits no more than five percentage points of success loss and no more than 5% falls.
- [ ] Encode stop conditions: mastery, invalid evidence, numerical failure, severe retention regression, budget/deadline, or three checks with neither success nor time-to-reengagement improvement.
- [ ] Implement immediate preservation with checkpoint hash, inventory, local acknowledgement, evaluation receipt, W&B receipt/offline directory, logs, monitor data, and commands before any checkpoint is reused.
- [ ] Run controller tests.

Run:

```bash
python -m pytest tests/test_continue_reengagement_campaign.py -q
```

Expected: all selected tests pass.

## Task 10: Full Local Regression and Static Verification

**Files:** all files above

- [ ] Run formatting/static syntax checks without rewriting unrelated files.

```bash
python -m compileall -q \
  reengagement_curriculum.py \
  native_reengagement_episode.py \
  native_reengagement_lane_vec_env.py \
  evaluate_reengagement_curriculum.py \
  qualify_native_reengagement.py \
  train_reengagement_campaign_block.py \
  continue_reengagement_campaign.py
git diff --check
```

- [ ] Run the complete B2 local suite.

```bash
python -m pytest \
  tests/test_reengagement_curriculum.py \
  tests/test_native_reengagement_episode.py \
  tests/test_native_reengagement_lanes.py \
  tests/test_native_curriculum_ppo.py \
  tests/test_train_reengagement_campaign_block.py \
  tests/test_native_curriculum_runner.py \
  tests/test_evaluate_reengagement_curriculum.py \
  tests/test_qualify_native_reengagement.py \
  tests/test_continue_reengagement_campaign.py -q
```

- [ ] Run frozen B1/native-navigation regressions.

```bash
python -m pytest \
  tests/test_native_contact_episode.py \
  tests/test_native_contact_lanes.py \
  tests/test_benchmark_native_contact_training.py \
  tests/test_train_retention_campaign_block.py \
  tests/test_native_navigation_curriculum_env.py -q
```

- [ ] Inspect `git status --short`, confirm only intended files are tracked/staged, and leave all pre-existing untracked files untouched.

## Task 11: Independent Code and Contract Review

- [ ] Request an independent code review focused on state-machine correctness, ABI alignment, CUDA synchronization, version isolation, seed leakage, artifact immutability, and B1 regressions.
- [ ] Request an independent verifier to compare every design requirement with tests and executable evidence.
- [ ] Fix every confirmed issue test-first, rerun the focused test, then rerun Task 10.

## Task 12: One-Instance RTX 4090 Qualification and First Cycle

**Precondition:** Tasks 1-11 pass locally. No GPU is created earlier.

- [ ] Run a fresh live inventory immediately before rental.

```bash
vastai show instances --raw
```

- [ ] Reuse/coordinate any owned instance. If this lane owns none and no reusable instance is available, search for one reliable RTX 4090 offer within the remaining $12 campaign ceiling, create exactly one labeled `AI_2-b2-reengagement-root-20260920`, and record its ID and hourly cost locally immediately.
- [ ] Upload the committed source plus verified parent checkpoint and acknowledgement. Do not upload unrelated worktree files.
- [ ] Establish W&B online tracking or a preserved offline run before training; start `nvidia-smi` and CPU monitoring with timestamped logs.
- [ ] Run the CUDA qualification:

```bash
python -B qualify_native_reengagement.py \
  --output-dir /workspace/AI_2/artifacts/reengagement-qualification \
  --reference-root /workspace/AI_2/inputs/reference \
  --frozen-root /workspace/AI_2/inputs/frozen \
  --checkpoint /workspace/AI_2/inputs/B_projected-050m/final.zip
```

Expected: exact native CPU/CUDA parity, real collector/update smoke pass, version-pure diagnostics, and `qualified: true`.

- [ ] Run a short CUDA training smoke in a unique output directory and verify finite losses, actual optimizer steps, expected counters, B2 completions, navigation rehearsal, and no per-step host copies.
- [ ] Run the matched parent development panels before the first full update and preserve them locally.
- [ ] Run one full `R4E1N` cycle only after qualification and smoke pass:

```bash
python -B train_reengagement_campaign_block.py \
  --schedule R4E1N \
  --checkpoint /workspace/AI_2/inputs/B_projected-050m/final.zip \
  --checkpoint-sha256 d6e487c22f73913cd88a428683c7ca721e23b1f9cb5a3a561ad24d1f5bf410db \
  --checkpoint-ack /workspace/AI_2/inputs/B_projected-050m/preservation_ack.json \
  --qualification /workspace/AI_2/artifacts/reengagement-qualification/qualification.json \
  --frozen-root /workspace/AI_2/inputs/frozen \
  --reference-root /workspace/AI_2/inputs/reference \
  --output /workspace/AI_2/artifacts/b2-cycle-001-R4E1N \
  --deadline-unix "$AI2_B2_DEADLINE"
```

- [ ] Immediately download the qualification, source/build receipts, train output, checkpoint, actor export, scenarios, episodes, W&B data, command/stdout/stderr logs, GPU/CPU monitor logs, and matched evaluations. Verify every hash locally before any further cycle.
- [ ] Evaluate B2 plus B1/near/wide retention on the pinned development banks. Continue to another cycle only if B2 improves and all alarms remain clear; stop according to the controller gates, not an arbitrary transition count.
- [ ] Before closeout, retrieve and verify every remaining useful artifact, then destroy this lane's unused instance under the standing authorization.
- [ ] Run a fresh final inventory and report every remaining instance by ID, label, state, owner/purpose, and hourly cost. Zero unexplained instances is required.

## Commit Sequence

Each cohesive commit must use the Lore protocol and exact co-author trailer. Example:

```text
Teach the native learner to re-engage after safe contact

Constraint: Preserve the frozen B1 lesson and its evidence byte-for-byte.
Rejected: Add re-engagement as an in-place B1 mode | It would mix qualification evidence and task versions.
Confidence: high
Scope-risk: broad
Directive: Keep combat task ID 1, but always validate the explicit lesson version.
Tested: B2 oracle/native/lane/PPO/runner/evaluator/controller suites and frozen B1 regressions.
Not-tested: CUDA parity is recorded separately until the RTX 4090 gate.
Co-authored-by: OmX <omx@oh-my-codex.dev>
```

Do not stage, modify, remove, or commit unrelated pre-existing untracked files.
