# Python layout and commands

Run module commands from the repository root with the Python environment that
contains the project dependencies. The classic training walkthrough remains in
[TRAINING_GUIDE.md](../TRAINING_GUIDE.md).

| Task | Entry point | Notes |
| --- | --- | --- |
| Classic training | `python -m ai2.training.train --cfgFile config.yaml` | Curriculum, rewards and checkpoint selection come from the config. |
| Native training | `python -m ai2.training.native_curriculum_runner --help` | Coordinates native blocks and the fixed full-game evaluator. |
| Full-game evaluation | `python -m ai2.evaluation.evaluate_fullgame_bank --help` | Select an exact policy and a fresh output directory. Requires 256-value observations. |
| Checkpoint comparison | `python -m ai2.evaluation.compare_training_checkpoints --help` | Compares saved checkpoint contracts and state. |
| Combat playback | `python -m tools.visualization.watch_games --help` | Select the classic checkpoint, opponent and number of games. |
| Navigation playback | `python -m tools.visualization.watch_navigation --help` | Select the checkpoint, navigation phase and config. |
| Human play | `python -m tools.visualization.play_vs_agent --help` | Interactive match; select a checkpoint with `--model`. |
| Benchmarking | `python -m tools.benchmarks.benchmark_policy_capacity --help` | Policy capacity and PPO update measurements; the actual benchmark requires CUDA. |
| Profiling | `python -m tools.profiling.profile_native_combat_training --help` | Profiles a specified frozen source checkout. |
| Validation | `python -m tools.validation.qualify_native_lanes --help` | Native-lane correctness and qualification receipts. |
| Analysis | `python -m tools.analysis.summarize_campaign --help` | Summarizes campaign artifacts. |

For example, a compatible SB3 policy can be evaluated on the fixed full-game bank
with:

```sh
python -m ai2.evaluation.evaluate_fullgame_bank \
  --policy /path/to/compatible-policy.zip --policy-kind sb3 \
  --device cpu --output-dir /path/to/new-evaluation-directory
```

This bank uses four 64-value frames. It rejects classic checkpoints whose
observation width includes opponent history. Use the classic playback tools for
those checkpoints; a playback result is distinct from a fixed-bank evaluation
receipt. The CPU evaluation command does not qualify CUDA training performance.

## Where code belongs

- `ai2/training/`: learning implementations, collectors and training entry points.
- `ai2/simulation/`: environment adapters, native bindings and simulation helpers.
- `ai2/evaluation/`: evaluators, reference captures and parity comparisons.
- `ai2/runtime/`: reusable campaign coordination, job execution and tracking.
- `tools/`: standalone benchmarks, profiling, validation, analysis and visualization.
- `baselines/<experiment>/scripts/`: scripts tied to an existing historical experiment.
- `research/`, `native/`, `tests/` and `docs/`: retain their existing roles.

A `benchmark_` filename does not necessarily mean a standalone tool: some of
these modules implement training or shared evaluation behavior and therefore
live in `ai2/`.

## Finding an old filename

[python-file-map.json](python-file-map.json) records all 146 former root Python
files, including their destination, disposition, reason and original SHA-256.
Of these, 143 implementations moved and three obsolete scripts were retired.
The two root compatibility modules below are included in the 143 moves.

| Root compatibility module | Implementation | Reason retained |
| --- | --- | --- |
| `train.py` | [`ai2/training/train.py`](../ai2/training/train.py) | Existing training commands and imports. |
| `curriculum_ppo.py` | [`ai2/training/curriculum_ppo.py`](../ai2/training/curriculum_ppo.py) | Saved checkpoints reference `curriculum_ppo.CurriculumPolicy`. |

New code should import the canonical package path. Search the file map's
`original` field to locate an old filename. For example, `watch_navigation.py`
now lives at [`tools/visualization/watch_navigation.py`](../tools/visualization/watch_navigation.py).

| Retired file | Reason |
| --- | --- |
| `eval_10_games.py` | Hardcoded checkpoint and import-time execution; its previously documented `--model` option did not exist. Select a model explicitly with the compatible evaluation/playback entry point. |
| `visual_train.py` | Broken one-off script importing the missing `train_utmist_v2` module. |
| `record_positions.py` | Overwrote `positions.json` using a `positions` field that did not match the required `grounded_positions` schema. |

## Historical experiments and source identity

The existing baseline folders are reused:

- [Throughput scaling scripts](../baselines/throughput-scaling-20260906/scripts/README.md)
- [Native PPO session 10 scripts](../baselines/native-ppo-session10/scripts/README.md)
- [Native capacity session 11 scripts](../baselines/native-capacity-session11/scripts/README.md)

Only the associated scripts move into these folders. Existing model files,
measurements, receipts, manifests and frozen source snapshots keep their original
paths and contents. Historical receipts can therefore still name old root
filenames. The file map locates their current counterparts without rewriting the
historical evidence. There is no second baseline archive or copy of the datasets.

Relocating imports and launch paths changes source identity. New runs must record
the current source inventory, including package initializers, path resolution and
compatibility modules. [`ai2.paths`](../ai2/paths.py) identifies this as source
layout version 2. Its resolver maps legacy logical filenames inside the chosen
checkout: a new checkout uses canonical paths; a frozen flat checkout uses its
own original files. Missing frozen training/game inputs are never replaced with
live inputs. Replay payloads without their own benchmark helper use current
orchestration code with the explicitly selected game source; a supplied frozen
helper remains the selected helper.

Existing benchmark or qualification receipts continue to describe their original
source hashes. The cleanup itself is not a new GPU benchmark or CUDA
qualification, and old receipts must not be treated as evidence for the relocated
source without the relevant checks being rerun.
