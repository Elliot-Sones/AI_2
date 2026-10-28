# AI_2 repository architecture

AI_2 trains a reinforcement learning agent for the UTMIST AI^2 fighting game.
The original game is Python. The project rewrote the game loop in C and CUDA so
training runs on the GPU. This document says what each folder holds, who writes
to it, and the rules for adding to it.

Read [README.md](README.md) for the training approach and
[docs/python-layout.md](docs/python-layout.md) for the command entry points.
[AGENTS.md](AGENTS.md) holds the cloud instance rules.

## Map

| Folder | Holds | Written by | In git |
| --- | --- | --- | --- |
| `ai2/` | The Python package: training, simulation, evaluation, runtime | People and agents, by hand | Yes |
| `tools/` | Standalone commands: benchmarks, profiling, validation, analysis, playback | People and agents, by hand | Yes |
| `native/` | C and CUDA sources for the game, lanes, and rewards | People and agents, by hand | Yes |
| `tests/` | Regression tests | People and agents, by hand | Yes |
| `docs/` | Guides, contracts, design plans, the file map | People and agents, by hand | Yes |
| `research/` | Investigation notes and their evidence | People and agents, by hand | Partly |
| `baselines/` | One record per finished experiment session | Retrieval scripts, then closeout | No |
| `results/` | Output of classic training: checkpoints, TensorBoard, run record | `ai2.training.train` | Model folder only |
| `wandb/` | Local Weights & Biases logs and offline runs | The `wandb` library | No |
| `videos/` | Demo recordings | Playback tools, then committed by hand | Yes |
| `images/` | Analysis figures | Analysis tools, then committed by hand | Yes |
| `UTMIST-AI2-main/` | The vendored tournament game | Upstream only | Yes |

Root files: `train.py` and `curriculum_ppo.py` are compatibility shims that
import from `ai2/`. `config.yaml` configures classic training. `setup_vast.sh`
prepares a cloud GPU host. `positions.json` is the reachable-position table that
navigation targets are sampled from. `assets.zip` is a copy of the game assets
and is byte-identical to `UTMIST-AI2-main/assets.zip`. `ENV_INFO` summarizes the
game rules as read from the environment source. `BASELINE.md`,
`TRAINING_GUIDE.md`, and `README_BEST_MODEL_PLAN.md` are long-form guides.

## ai2/

The importable package. Run every command from the repository root as
`python -m ai2.<subpackage>.<module>`.

| Subpackage | Holds |
| --- | --- |
| `ai2/training/` | PPO, curriculum phases, rollout collection, training entry points |
| `ai2/simulation/` | Environment adapters, vector environments, native bindings |
| `ai2/evaluation/` | Evaluators, reference captures, parity comparisons |
| `ai2/runtime/` | Campaign coordination, job execution, Weights & Biases tracking |
| `ai2/paths.py` | Maps old root filenames to current paths. Declares the source layout version |

Rules:

- Put code that other code imports in `ai2/`. Put a command that nothing imports in `tools/`.
- A module name that starts with `benchmark_` can still belong in `ai2/` when training or evaluation imports it.
- Import by the package path. Do not import through `train.py` or `curriculum_ppo.py`.
- When you move or rename a module, update `SOURCE_LOCATIONS` in `ai2/paths.py` and `docs/python-file-map.json`.
- Do not replace a frozen experiment source with live code. `ai2.paths` resolves names inside the checkout you give it.

## tools/

Standalone commands. Three training modules in `ai2/` still import validators
from `tools/validation/`. New code should not add to that list. Move a validator
into `ai2/evaluation/` when package code needs it.

| Subfolder | Holds |
| --- | --- |
| `tools/benchmarks/` | Throughput and capacity measurements |
| `tools/profiling/` | Profilers for native training |
| `tools/validation/` | Correctness gates, qualification receipts, source audits |
| `tools/analysis/` | Campaign summaries, monitors, checkpoint utilities |
| `tools/visualization/` | Watch games, watch navigation, play against the agent |

Rules:

- Every tool accepts `--help` and takes its inputs as arguments. No hardcoded checkpoint paths.
- A tool writes to the output directory you pass it. It does not write into `baselines/` or `research/` on its own.
- A script that only makes sense for one past experiment goes in that experiment's `scripts/` folder under `baselines/`, not here.

## native/

The C and CUDA game. Twenty-two `.c` files, twelve `.cu` files, and their
headers. `navigation_spaces.json` declares the navigation observation layout.

Rules:

- Each `.c` file has a `.cu` twin when the GPU path differs. Keep them in step.
- The Python side compiles these on demand into a build directory the caller passes. Do not commit compiled objects.
- A change to game logic needs a parity test against the frozen Python game. See `tests/`.

## tests/

Unit tests, one file per module under test, named `test_<module>.py`. Forty-four
of them cover the native engine. `native_sanitizer_smoke.c` is a compiled
smoke test.

Rules:

- Run the tests with a Python that has `torch` installed. A test that needs CUDA, a C compiler, or the frozen game source skips. A skip is not a pass. Read the summary line.
- Native parity tests read the frozen Python game from the path in `AI2_FROZEN_SOURCE_ROOT`. A local copy is inside `research/update-profile-20260913/oracle.tar.gz`. Git does not track it.
- Add a test in the same change as the behavior. A gate without a test is a comment.
- Test fixtures stay small. A fixture over one megabyte belongs in `baselines/` with a receipt, and the test reads it from there.

## docs/

Guides and contracts that stay true across sessions.

| Item | Holds |
| --- | --- |
| `docs/python-layout.md` | Entry points, where code belongs, how to find an old filename |
| `docs/python-file-map.json` | Every former root script: destination, reason, original hash |
| `docs/fullgame-eval-bank-contract.md` | The fixed full-game evaluator contract |
| `docs/resident_stream_gate.md` | The resident CUDA replay gate |
| `docs/training-flow/` | A self-contained HTML explainer of the training loop |
| `docs/superpowers/specs/` | Dated design documents |
| `docs/superpowers/plans/` | Dated implementation plans |

Rules:

- A document here describes how something works now. Dated history goes in `research/`.
- Name a spec or plan `YYYY-MM-DD-<topic>.md`.
- When behavior changes, change the contract in the same pull request.

## research/

Investigations. Each one answers a question and keeps its evidence.

| Item | Holds |
| --- | --- |
| `research/00-...md` to `research/03-...md` | Option surveys written before the native rewrite |
| `research/training-quality-evidence-ledger.md` | The running ledger of what has been shown and what has not |
| `research/<topic>-<date>/` | One investigation: `REPORT` or `RESULTS`, receipts, scripts, and downloads |

Rules:

- Start a folder with a report file that states the question, the status, and the evidence limits in the first ten lines.
- Name the folder `<topic>-<YYYYMMDD>`.
- Keep raw downloads under a `downloads/` subfolder. Git ignores that name.
- Apply the same retention rule as `baselines/` once the investigation closes.
- `research/update-profile-20260913/` stays on disk only. Git ignores it, and no commit contains it. It is about 3 GB, mostly downloads.
- Link to a `baselines/` folder instead of copying its files.

## baselines/

One folder per finished experiment session. A session is one cloud rental or
one local campaign. The folder is the proof behind a results claim, and the
place a later session resumes from.

The name is historical. The first folders measured the Python game before the
native rewrite, so they were the baseline. A better name is `experiments/`.
Renaming touches tests, tools, docs, and research links, so do it as one change
with `ai2/paths.py` and the file map updated.

Git ignores this folder, and no commit contains it. It was removed from all
history on 2026-10-18. The folders live only on disk, so back them up yourself.

Two kinds of folder live here today:

- The small September 4 and 5 measurement folders that `BASELINE.md` links to. Keep them.
- Session folders from native training. These follow the retention rule below.

Layout of a session folder:

```text
baselines/<topic>-<YYYYMMDD>/
├── README.md or RESULTS.md   Question, status, result, evidence limits
├── receipts/                 Instance, retrieval, and verification JSON
├── scripts/                  Scripts that only this session used
├── final/                    The final checkpoint and its metrics
└── payload/                  Frozen inputs the session trained from, if any
```

Rules for populating a session folder:

- Name it `<topic>-<YYYYMMDD>` with the start date. Record the Vast.ai instance ID in a receipt on the day you create it.
- Write the result file before you destroy the instance. State what was measured, what was not, and what remains unproven.
- Retrieve from the instance once. If a retrieval fails, retry into the same folder. Do not keep `batch1`, `batch2`, `retrieved`, and `guard` copies of the same tree.
- Keep one final checkpoint per run. Delete periodic `stage_step_*.zip` files after closeout unless the result file names one of them.
- Delete a tarball after you extract it and verify the extracted files.
- Send metrics to Weights & Biases during the run. The dashboard is the metric record. The folder holds the checkpoint and the receipts.
- Do not store a second copy of the frozen game source. Point at the existing payload.
- A closed session folder is under one gigabyte. If it is larger, something above was skipped.

Before the October 2026 cleanup this folder was 136 GB. Duplicate retrievals
were 31 GB of that and periodic checkpoints were 45 GB.

## results/

Where classic training writes. `config.yaml` sets `folders.parent_dir` and
`folders.model_name`, so the default is `results/ppo_utmist/`.

| Path | Holds |
| --- | --- |
| `results/ppo_utmist/model/` | Phase checkpoints such as `phase4_final.zip`. Git tracks this folder |
| `results/ppo_utmist/tb/` | TensorBoard logs. Ignored |
| `results/ppo_utmist/wandb_run.json` | The run record that `ai2.runtime.tracking` reads |

Rules:

- Training writes here. Nothing else does.
- Commit a checkpoint only when a guide refers to it by name.

## wandb/

Local state from the `wandb` library: debug logs and offline runs waiting for
`wandb sync`. Git ignores it. The cloud project is the record. Delete this
folder when the offline runs are synced.

## videos/ and images/

Demo recordings and analysis figures that the README shows. Both are small and
tracked. Add a file here only when a guide or the README links to it.

## UTMIST-AI2-main/

The tournament game as published. The native engine is checked against it, so
it must not drift.

Rules:

- Do not edit files here. A game change goes in `native/` or in an adapter under `ai2/simulation/`.
- The frozen copies under `baselines/` are the reference for parity tests, not this folder, because this folder can be updated from upstream.
- `UTMIST-AI2-main/=0.2.1` is a stray pip log from an unquoted version pin. It can be deleted.

## Cloud sessions

Every GPU session follows [AGENTS.md](AGENTS.md): one instance per agent, a
label that names the project, task, owner, and date, retrieval and
verification before destroy, and a live inventory before the session ends. The
retention rules above say what to keep after that retrieval.

## Keep this document true

When you add a top-level folder, add a row to the map and a section with its
rules. Regenerate the folder sizes with:

```sh
du -sh */ | sort -h
```
