# Full-Game Evaluation Bank v1 Contract

`evaluate_fullgame_bank.py` is the first immutable full-game evaluator for comparing AI_2 policies in the original Python game. It intentionally does not edit `eval_10_games.py`.

## What v1 Measures

- Fixed reset-only cases with unique case IDs and seeds.
- Both learner slots: player 0 reset distribution and player 1 mirrored reset distribution.
- Both deterministic and sampled execution panels.
- Requested opponent name and realized opponent class.
- Four-frame stacked observations shaped as 256 float32 values, with SB3/VecFrameStack-compatible reset semantics: three zero frames followed by the initial observation.
- Raw policy action values sent to `env.step` as `executed_action_values`, plus decoded keyboard buttons as `decoded_buttons`.
- Per-tick player position, velocity, state, facing, stocks, damage, grounded flag, and weapon when those fields exist. Grounded status uses `is_on_floor()` when the player exposes that callable.
- Episode outcomes, terminal cause from stock loss or timeout, approach progress, attack requests, attack starts, hits, damage, falls, timeouts, and trace-derived diagnoses.

## What v1 Does Not Claim

- It does not place players at handcrafted coordinates; all cases use `placement_profile=reset_only`.
- It does not support classic SB3 checkpoints whose observations were expanded by `OpponentHistoryWrapper`; those fail closed instead of being evaluated under the wrong input contract.
- It does not infer walkoffs from arbitrary x/y bounds. The summary reports stock-loss falls and timeouts only.
- It does not validate GPU policy performance by itself. `--device cuda` is available for adapters, but CUDA evaluation is a later gate.
- It does not prove a policy is human-good. It creates repeatable evidence and representative traces for the larger training-quality decision.
- Video capture is optional and off by default. If video is requested without a real capture callback, or the callback creates an empty artifact, evaluation fails closed.

## Output Discipline

The evaluator refuses a nonempty output directory unless `--overwrite` is supplied. It writes into a sibling temporary directory first, then atomically promotes the completed artifact set. An explicitly supplied `checkpoint_path` must exist and hash before episode work begins. `manifest.json` records the schema version, git revision, checkpoint hash when available, case bank, modes, placement limitation, and hashes of output artifacts.

## Real-Game Runtime

The real WarehouseBrawl environment is constructed and stepped under a scoped cwd of `UTMIST-AI2-main/environment` so the existing checked-in `assets/` directory is visible to game code that still checks cwd-relative assets. The cwd is restored after every call, and the evaluator must not call `gdown.download` during no-render evaluation.
