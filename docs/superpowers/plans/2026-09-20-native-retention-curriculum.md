# Native Navigation-Retention Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the existing binary-button B1 contact policy with alternating native CUDA navigation and contact rollouts, then run an artifact-safe 10M candidate screen and gated continuation toward 50M.

**Architecture:** Keep one rollout homogeneous and switch tasks only after PPO consumes it. Reuse the shared actor and the existing independent navigation/combat critics. Wrap the qualified phase-0b navigation lanes with explicit curriculum metadata, generalize the native buffer/collector without changing either task's reward semantics, and drive immutable R4C1N/R3C2N schedules from the exact B40 checkpoint.

**Tech Stack:** Python 3, PyTorch CUDA, Stable-Baselines3 PPO, native C/CUDA lane engines, `unittest`, Vast.ai, offline W&B-compatible logs.

---

## Preconditions and fixed evidence

- Work only in `/private/tmp/ai2-throughput-qualification-20260920` on `codex/throughput-qualification-20260920`.
- Preserve the existing uncommitted learning-rate/resume edits in `native_curriculum_ppo.py` and all prior campaign files.
- Common parent is B40 SHA-256 `6d7b83a287b1a29bf8741295d769178764fd8b4beaf101a678b62cdc41bd73f0`.
- Historical control is B50 SHA-256 `d6e487c22f73913cd88a428683c7ca721e23b1f9cb5a3a561ad24d1f5bf410db`.
- Fixed geometry is 4,096 lanes, 512 steps, batch 16,384, four epochs, FP32, learning rate `3e-5`.
- One rollout is 2,097,152 learner transitions; one five-rollout cycle is 10,485,760.
- Never mix navigation and contact lanes inside one rollout.
- Do not change reward definitions, episode termination definitions, action head, or evaluator predicates in this implementation.

## Task 1: Give native navigation the curriculum metadata contract

**Files:**

- Create: `native_navigation_curriculum_env.py`
- Create: `tests/test_native_navigation_curriculum_env.py`

- [ ] **Step 1: Write the failing metadata and episode-record tests**

The tests must construct a lightweight fake raw navigation environment and assert:

```python
self.assertEqual(env.curriculum_task, "navigation")
self.assertEqual(env.curriculum_version, NAVIGATION_VERSION)
np.testing.assert_array_equal(env.allowed_control_mask, TASK_MASKS["navigation"])
self.assertTrue(env.host_has_timeouts())
self.assertIs(env.reset_tensor(), raw.reset_result)
self.assertIs(env.step_tensor(actions), raw.step_result)
self.assertEqual(info["curriculum_task"], "navigation")
self.assertEqual(info["curriculum_version"], NAVIGATION_VERSION)
np.testing.assert_array_equal(info["allowed_control_mask"], TASK_MASKS["navigation"])
```

Also assert that missing/malformed native navigation records fail closed rather than being silently relabeled.

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
PYTHONPATH=. python3 -m unittest -v tests.test_native_navigation_curriculum_env
```

Expected: import failure because `native_navigation_curriculum_env.py` does not exist.

- [ ] **Step 3: Implement the thin ownership-preserving adapter**

Implement these exact public contracts:

```python
NAVIGATION_VERSION = "bridge_0b_navigation-v1"
NAVIGATION_MASK = np.asarray(TASK_MASKS["navigation"], dtype=bool)

class NativeNavigationCurriculumEnv:
    curriculum_task = "navigation"
    curriculum_version = NAVIGATION_VERSION

    def __init__(self, env): ...
    @property
    def allowed_control_mask(self): ...
    def host_has_timeouts(self): return True
    def reset_tensor(self, seeds=None): ...
    def step_tensor(self, clipped_actions): ...
    def drain_episode_infos(self): ...
    def close(self): ...
    def __getattr__(self, name): ...

def build_navigation_curriculum_env(lanes, *, build_dir, device="cuda", seed=0,
                                    wrapper_cfg=None, record_capacity=None,
                                    observation_space=None, action_space=None,
                                    dt=1.0 / 30.0, collision_bias=None): ...
```

`build_navigation_curriculum_env` must call the existing `native_lane_vec_env.build_cuda_env`; it must not duplicate the native backend. `drain_episode_infos` must preserve the original `episode` and `navigation` payloads and add task, mask, and version at top level.

- [ ] **Step 4: Run focused and regression tests GREEN**

```bash
PYTHONPATH=. python3 -m unittest -v \
  tests.test_native_navigation_curriculum_env \
  tests.test_native_lane_vec_env \
  tests.test_native_lane_cpu_adapter
```

- [ ] **Step 5: Commit only Task 1 files with a Lore-format commit**

## Task 2: Generalize the native rollout buffer without weakening validation

**Files:**

- Modify: `native_curriculum_ppo.py`
- Modify: `tests/test_native_curriculum_ppo.py`

- [ ] **Step 1: Add failing homogeneous-buffer tests**

Add tests proving:

1. A full navigation rollout with `TASK_MASKS["navigation"]` validates.
2. A full contact rollout with `TASK_MASKS["combat"]` validates exactly as before.
3. Changing task or mask after the first transition raises.
4. Navigation execution forces attack index 7 to zero, while contact permits it.
5. Bernoulli actions remain exact 0/1 and Gaussian actions remain clipped/masked.

Use task IDs from a frozen mapping:

```python
TASK_IDS = {"navigation": 0, "combat": 1}
```

- [ ] **Step 2: Run the focused test and observe RED**

```bash
PYTHONPATH=. python3 -m unittest -v tests.test_native_curriculum_ppo
```

Expected: the current `TaskTensorRolloutBuffer.add()` rejects navigation.

- [ ] **Step 3: Generalize `TaskTensorRolloutBuffer`**

Replace contact-only checks with canonical task-derived checks:

```python
def _canonical_task_mask(task):
    if task not in TASK_IDS:
        raise ValueError(f"Unsupported native curriculum task: {task!r}")
    return np.asarray(TASK_MASKS[task], dtype=bool)

def add(self, *args, task, mask, executed_actions):
    expected_mask = _canonical_task_mask(task)
    if self.task is not None and self.task != task:
        raise ValueError("Cannot mix tasks within one rollout")
    if not np.array_equal(mask, expected_mask):
        raise ValueError("Wrong allowed-control mask for rollout task")
    ...

def validate(self, task, mask):
    expected_mask = _canonical_task_mask(task)
    ...
```

The task/mask arrays remain CUDA resident. Preserve the class name as an import-compatible alias if introducing `NativeTaskTensorRolloutBuffer`.

- [ ] **Step 4: Run the focused test GREEN and the existing action/update parity tests**

```bash
PYTHONPATH=. python3 -m unittest -v \
  tests.test_native_curriculum_ppo \
  tests.test_contact_action_update_parity \
  tests.test_native_combat_update_parity
```

- [ ] **Step 5: Commit only Task 2 files with a Lore-format commit**

## Task 3: Switch native task, environment, mask, and critic only at consumed-rollout boundaries

**Files:**

- Modify: `native_curriculum_ppo.py`
- Modify: `tests/test_native_curriculum_ppo.py`
- Modify: `tests/test_campaign_resume.py`

- [ ] **Step 1: Add failing switch and resume tests**

The tests must prove:

- a task/environment switch is rejected during collection;
- a switch is rejected while a full rollout is unconsumed;
- after `train()` marks the rollout consumed, switching succeeds;
- the new tensor environment must match CUDA device, lane count, observation space, and action space;
- switching restores that task's own saved `_last_obs` and `_last_episode_starts`, never frames from the other task; only a task's first use in a newly constructed block resets its environment;
- save/load preserves the shared actor, navigation critic, combat critic, protected optimizer tensors, counters, task history, action contract, and learning-rate schedule;
- a loaded checkpoint can select either task for its next rollout.

- [ ] **Step 2: Run focused tests and observe RED**

```bash
PYTHONPATH=. python3 -m unittest -v \
  tests.test_native_curriculum_ppo \
  tests.test_campaign_resume
```

- [ ] **Step 3: Implement task/environment validation and switching**

Add exact public seams:

```python
TASK_VERSIONS = {
    "navigation": "bridge_0b_navigation-v1",
    "combat": CONTACT_VERSION,
}

def validate_task_environment(env, task): ...

class NativeCurriculumPPO(TensorPPO):
    def set_task_environment(self, task, tensor_env, *, calibration=False): ...
```

`set_task_environment` must:

1. refuse switching during collection or with an unconsumed rollout;
2. validate task metadata/version/mask and identical tensor ABI;
3. save the outgoing task's live observation and episode-start tensors in a per-task state map;
4. update `tensor_vec_env` and the SB3 shim consistently;
5. call `policy.set_task(task, calibration)`;
6. restore the incoming task's live state, or reset it and create CUDA float32 episode starts only on first use in this process;
7. append exactly one task-history entry.

Generalize collection so contact keeps its “no truncations” assertion, while navigation applies SB3-compatible terminal-value bootstrapping only where `truncated & ~terminated` is true. Every drained record must match the active task/version/mask.

Relax `from_curriculum()`/`load()` only enough to accept a saved navigation or combat task from a native alternating checkpoint. Environment state itself remains deliberately excluded from SB3 checkpoint ZIPs, so every new immutable block constructs fresh task environments; the manifest must say that model/critic/optimizer/RNG counters resume but live episode state does not.

`_policy_contract()` and `train()` must derive their mask from `TASK_MASKS[self.task]`; task-specific metrics remain `train/navigation/*` and `train/combat/*`.

- [ ] **Step 4: Run focused, resume, and parity tests GREEN**

```bash
PYTHONPATH=. python3 -m unittest -v \
  tests.test_native_curriculum_ppo \
  tests.test_campaign_resume \
  tests.test_throughput_checkpoint_roundtrip \
  tests.test_contact_action_update_parity \
  tests.test_native_combat_update_parity
```

- [ ] **Step 5: Commit only Task 3 files with a Lore-format commit**

## Task 4: Add the immutable alternating retention block runner

**Files:**

- Create: `train_retention_campaign_block.py`
- Create: `tests/test_train_retention_campaign_block.py`

- [ ] **Step 1: Write failing config, schedule, artifact, and failure tests**

Freeze these recipes:

```python
SCHEDULES = {
    "R4C1N": ("combat", "combat", "combat", "combat", "navigation"),
    "R3C2N": ("combat", "combat", "combat", "navigation", "navigation"),
}
```

Tests must assert exact transition counts per task, one homogeneous task per row, one final checkpoint/actor per block, task-specific episode files, policy and optimizer continuity, immutable output paths, failure checkpoint preservation, and no CPU fallback.

- [ ] **Step 2: Run the focused test and observe RED**

```bash
PYTHONPATH=. python3 -m unittest -v tests.test_train_retention_campaign_block
```

- [ ] **Step 3: Implement the runner by reusing existing artifact helpers**

Use `train_contact_campaign_block.py` and `benchmark_curriculum_throughput.py` helpers rather than copying checkpoint/hash/logger logic. Build one contact tensor env and one navigation tensor env, then for every scheduled rollout:

```python
model.set_task_environment(task, environments[task], calibration=False)
model.collect_rollouts(environments[task], callback, model.rollout_buffer, cfg.n_steps)
model.train()
```

Every rollout receipt must include task, version, task transitions, cumulative total transitions, collection/update time, optimizer-step accounting, metrics, and the compressed episode trace path/hash. The final result must separately total navigation and contact transitions.

- [ ] **Step 4: Run focused and existing campaign regressions GREEN**

```bash
PYTHONPATH=. python3 -m unittest -v \
  tests.test_train_retention_campaign_block \
  tests.test_contact_campaign_block \
  tests.test_campaign_resume
```

- [ ] **Step 5: Commit only Task 4 files with a Lore-format commit**

## Task 5: Add a preservation-gated 10M-to-50M controller

**Files:**

- Create: `continue_retention_campaign.py`
- Create: `tests/test_continue_retention_campaign.py`

- [ ] **Step 1: Write failing plan, gate, adoption, and preservation tests**

Tests must prove:

- both 10M candidates independently start from the exact B40 SHA;
- job names include recipe, cumulative added transitions, seed, and source digest;
- an uncertain remote launch is adopted rather than duplicated;
- no eval or continuation begins before local hash verification and remote preservation ACK agree;
- the diagnostic gate uses the separate contact, nearby, wide, fall, and attribution fields without an aggregate score;
- an ineligible lineage stops independently;
- Pareto dominance selects a winner; a tradeoff schedules one matched comparison;
- the winner continues one immutable five-rollout cycle at a time toward about 50M;
- two consecutive no-improvement cycles stop continuation;
- deadline and remaining-cost reserve fail closed.

- [ ] **Step 2: Run the focused test and observe RED**

```bash
PYTHONPATH=. python3 -m unittest -v tests.test_continue_retention_campaign
```

- [ ] **Step 3: Implement using the proven contact controller primitives**

Reuse the validated `RemoteClient`, checksum verification, immutable JSON writing, background launch receipt, polling, and local-preservation verification from `continue_contact_campaign.py`. Do not weaken its ACK contract.

Expose pure functions for testability:

```python
def diagnostic_gate(result, b40_baseline): ...
def pareto_relation(left, right): ...
def improved(previous, current): ...
def build_initial_plan(parent_sha256=B40_SHA256): ...
def build_continuation_step(winner, cycle): ...
```

Every decision receipt must preserve raw component metrics and a textual reason. No weighted score is permitted.

- [ ] **Step 4: Run controller and artifact regressions GREEN**

```bash
PYTHONPATH=. python3 -m unittest -v \
  tests.test_continue_retention_campaign \
  tests.test_continue_contact_campaign \
  tests.test_record_throughput_qualification_offline
```

- [ ] **Step 5: Commit only Task 5 files with a Lore-format commit**

## Task 6: Complete local integration and independent review

**Files:**

- Modify only files required by review findings.

- [ ] **Step 1: Run the complete relevant local suite**

```bash
PYTHONPATH=. python3 -m unittest -v \
  tests.test_native_navigation_curriculum_env \
  tests.test_native_lane_vec_env \
  tests.test_native_lane_cpu_adapter \
  tests.test_native_contact_lanes \
  tests.test_native_curriculum_ppo \
  tests.test_contact_action_update_parity \
  tests.test_native_combat_update_parity \
  tests.test_campaign_resume \
  tests.test_throughput_checkpoint_roundtrip \
  tests.test_train_retention_campaign_block \
  tests.test_contact_campaign_block \
  tests.test_continue_retention_campaign \
  tests.test_continue_contact_campaign
```

- [ ] **Step 2: Run syntax/static import validation**

```bash
python3 -m py_compile \
  native_navigation_curriculum_env.py \
  native_curriculum_ppo.py \
  train_retention_campaign_block.py \
  continue_retention_campaign.py
```

- [ ] **Step 3: Request specification review, then code-quality review**

Fix all correctness findings and rerun the affected tests. Preserve unrelated uncommitted work.

- [ ] **Step 4: Commit review fixes with a Lore-format commit**

## Task 7: Qualify on one RTX 4090 before campaign spend

**Files:**

- Create: `tests/test_retention_cuda_qualification.py`
- Create: `run_retention_cuda_qualification.py`
- Create locally after run: `baselines/native-retention-20260920/qualification/**`

- [ ] **Step 1: Inventory Vast.ai live and write a local ownership receipt before creation**

Run `vastai show instances --raw` with stderr suppressed from user-visible output. Reuse/coordinate any existing owned instance; create only if inventory is empty. Record ID, exact label, owner, purpose, state, and hourly price immediately.

- [ ] **Step 2: Upload a clean source snapshot and preserved B40 checkpoint/ACK**

Verify the uploaded source archive SHA and B40 SHA remotely before execution.

- [ ] **Step 3: Run CUDA qualification gates**

The qualification must prove:

1. navigation CPU/CUDA fixed-action parity for qualified phase 0b;
2. existing contact CUDA parity still passes;
3. one navigation rollout/update followed by one contact rollout/update is finite;
4. the shared actor changes on each task;
5. only the selected critic changes in each update;
6. optimizer steps equal the nominal count with no unexpected KL stop;
7. save/load exactness covers actor, both critics, optimizer tensors, counters, task history, masks, and next-task continuation.

Run with offline W&B mode and record `nvidia-smi dmon`/utilization samples plus per-rollout timing.

- [ ] **Step 4: Download immediately, then verify every artifact hash locally**

Do not start candidate training until the qualification block, inventory, and preservation ACK are locally verified.

## Task 8: Run the 10M screen and gated continuation

**Files:**

- Create locally after run: `baselines/native-retention-20260920/campaign/**`
- Update: `baselines/native-retention-20260920/RESULTS.md`

- [ ] **Step 1: Launch R4C1N and preserve it locally**

Run five scheduled rollouts from B40, download immediately, and verify source/checkpoint/episode/result hashes.

- [ ] **Step 2: Evaluate R4C1N in the original Python game**

Run deterministic and sampled contact, nearby navigation, and wide navigation development banks. Preserve terminal attribution and failure traces.

- [ ] **Step 3: Launch and evaluate R3C2N only after R4C1N preservation is acknowledged**

Use a distinct immutable seed and the same evaluator banks.

- [ ] **Step 4: Apply the declared diagnostic/Pareto gate**

Continue no ineligible candidate. If candidates trade off, run the one additional matched comparison specified by the controller rather than choosing by intuition.

- [ ] **Step 5: Continue the eligible winner cycle-by-cycle toward 50M**

After every cycle: download, hash-verify, evaluate, update no-improvement state, and check remaining cost plus the $0.75 reserve.

- [ ] **Step 6: Run confirmation only if promotion conditions are met**

Use untouched 100-case contact confirmation and a fresh matched parent navigation confirmation. Do not call the model promoted from development panels alone.

- [ ] **Step 7: Final closeout**

Verify the full local mirror, destroy the owned instance, run a fresh live instance inventory, and report every remaining instance by ID, label, state, owner/purpose, and hourly cost. Update `RESULTS.md` with tested facts, failures, exact spend, and the next curriculum decision.
