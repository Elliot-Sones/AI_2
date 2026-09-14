#!/usr/bin/env python3
"""Run the six frozen-game grouped parity gates and bind results to source hashes.

Scripted simulator correctness only: no policy inference or training. The stock
process-isolated game is the independent reference, including terminal metadata.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parent


def snapshot(paths):
    return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}


def result_passed(result):
    return result.testsRun == 6 and not (result.failures or result.errors or result.skipped)


def required_sources(source, job, candidate):
    # The frozen game is the whole environment package: environment.py imports environment.constants,
    # and train.py imports the scripted opponents from environment.agent. Hash them all.
    paths = [Path(__file__), job, candidate, source / "UTMIST-AI2-main/environment/environment.py",
             source / "UTMIST-AI2-main/environment/__init__.py", source / "UTMIST-AI2-main/environment/constants.py",
             source / "UTMIST-AI2-main/environment/agent.py",
             source / "train.py", source / "control.yaml", ROOT / "benchmark_campaign.py",
             ROOT / "grouped_vec_env.py", ROOT / "batched_rng_vec_env.py", ROOT / "isolated_rng_vec_env.py", ROOT / "shared_action_vec_env.py",
             ROOT / "tests/test_grouped_candidate_game.py", ROOT / "tests/test_grouped_game.py"]
    paths += sorted(ROOT.glob("native_*.py"))
    for pattern in ("*.c", "*.h", "*.cu"):
        paths += sorted((ROOT / "native").glob(pattern))
    return list(dict.fromkeys(Path(path).resolve() for path in paths))


def runtime_environment(source, job, candidate, *, input_enabled, observation_enabled):
    return {
        "AI2_CAMPAIGN_SOURCE": str(source), "TRAIN_JOB": str(job),
        "AI2_CONFIG": str(source / "control.yaml"),
        "AI2_GROUPED_CANDIDATE_PATH": str(candidate), "AI2_GROUPED_GAME_STEPS": "1000",
        "AI2_NATIVE_INPUT": str(int(input_enabled)),
        "AI2_NATIVE_OBSERVATION": str(int(observation_enabled)),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
        "MPLCONFIGDIR": "/tmp/ai2-mpl", "PYGAME_HIDE_SUPPORT_PROMPT": "1",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-job", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-input", action="store_true")
    parser.add_argument("--native-observation", action="store_true")
    args = parser.parse_args(argv)
    source, job, candidate, output = (getattr(args, key).resolve() for key in ("source", "train_job", "candidate", "output"))
    output.mkdir(parents=True, exist_ok=False)
    environment = source / "UTMIST-AI2-main/environment/environment.py"
    sources = required_sources(source, job, candidate)
    before = snapshot(sources)
    runtime = runtime_environment(source, job, candidate, input_enabled=args.native_input, observation_enabled=args.native_observation)
    previous = {key: os.environ.get(key) for key in runtime}
    started = time.time()
    os.environ.update(runtime)
    try:
        with (output / "tests.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_grouped_candidate_game.CandidateGroupedGameParityTest")
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    after = snapshot(sources)
    passed = result_passed(result) and before == after
    receipt = {
        "status": "passed" if passed else "failed", "exit_code": 0 if passed else 1,
        "testsRun": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "candidate_grouped_path": str(candidate), "runner_sha256": before[str(candidate)],
        "frozen_environment_sha256": before[str(environment)],
        "sourcehashes_before": before, "sourcehashes_after": after,
        "runtime_env": runtime, "steps_per_case": 1000, "group_sizes": [2, 4, 8],
        "started_unix": started, "ended_unix": time.time(),
    }
    (output / "parity_summary.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "exit_code", "testsRun", "failures", "errors", "skipped")}))
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
