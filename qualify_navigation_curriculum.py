#!/usr/bin/env python3
"""Qualify corrected phase 0a/0b against the frozen Python/SB3 game, then CUDA.

CPU mode is physics and wrapper parity only. CUDA mode also requires all four
64/256-lane default/nondefault-stream cases; neither mode performs training.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from qualify_native_lanes import QualificationError, _require_cuda, run_case, source_manifest as native_source_manifest

ROOT = Path(__file__).resolve().parent
SCHEMA = "ai2.navigation_curriculum.parity.v1"
STEPS = 1200
DISTANCES = {"0a": (1.5, 3.0), "0b": (2.0, 6.0)}
COMMON_NAVIGATION = {"fall_penalty": -3.0, "time_limit_seconds": 20, "multi_target": 3,
                     "completion_mode": "sequence", "control_mode": "movement",
                     "settle_seconds": 0.3, "post_goal_seconds": 0.5}


def select_phase(params: dict[str, Any], phase: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep both corrected navigation definitions intact; change only the selector."""
    if phase not in DISTANCES:
        raise ValueError("phase must be 0a or 0b")
    selected = copy.deepcopy(params)
    cfg = selected["phases"][phase]
    if cfg.get("type") != "navigation":
        raise ValueError(f"phase {phase} must have type navigation")
    nav = cfg["navigation"]
    minimum, maximum = DISTANCES[phase]
    required = dict(COMMON_NAVIGATION, min_dist=minimum, max_dist=maximum)
    for key, value in required.items():
        if nav.get(key) != value:
            raise ValueError(f"corrected phase {phase} requires {key}={value!r}; got {nav.get(key)!r}")
    if nav.get("settle_speed", 1.0) != 1.0:
        raise ValueError("corrected navigation requires settle_speed=1.0")
    selected.setdefault("curriculum", {})["phase"] = phase
    wrapper = {"min_dist": minimum, "max_dist": maximum, "fall_penalty": -3.0,
               "completion_mode": "sequence", "control_mode": "movement", "settle_speed": 1.0,
               "max_targets": 3, "time_limit": 600, "settle_steps": 9, "post_goal_required_steps": 15}
    return selected, wrapper


def require_wrapper_config(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise QualificationError(f"live wrapper config {key}: expected {value!r}, got {actual.get(key)!r}")


def frozen_source_manifest(source: Path) -> dict[str, str]:
    required = ("train.py", "control.yaml", "positions.json", "UTMIST-AI2-main/environment/environment.py")
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(f"frozen corrected source missing {source / name}")
    files = [p for p in source.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".yaml")
             and not any(part in ("__pycache__", ".git", "results", "wandb") for part in p.relative_to(source).parts)]
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def run_python_cases(*, source: Path, config: Path, wrapper_cfg: dict[str, Any],
                     build_dir: Path, steps: int = STEPS) -> list[dict[str, Any]]:
    """Reuse the original exact SB3 replay assertions without unittest skip gates."""
    import pymunk
    import torch
    import native_lane_vec_env as native

    if str(getattr(pymunk, "version", "")) != "6.2.1":
        raise QualificationError(f"frozen parity requires pymunk 6.2.1, got {getattr(pymunk, 'version', '?')}")
    spec = importlib.util.spec_from_file_location("ai2_curriculum_lane_tests", ROOT / "tests/test_native_lane_vec_env.py")
    if spec is None or spec.loader is None:
        raise QualificationError("frozen SB3 pipeline parity helper unavailable")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    class SelectedPhasePipeline(helpers.FrozenGameLaneTests):
        def _build_game(self, seed: int):
            from benchmark_replay import build_navigation_env
            from navigation_episode_capture import wrapper_config
            from validate_native_navigation import find_frozen_opponent_wrapper

            env, _params, phase = build_navigation_env(source, config, source / "train.py", seed, frame_stack=4)
            try:
                wrapper = find_frozen_opponent_wrapper(env)
                actual = wrapper_config(wrapper)
                require_wrapper_config(wrapper_cfg, actual)
                wrapper_source = Path(inspect.getfile(type(wrapper))).resolve()
                if wrapper_source != (source / "train.py").resolve():
                    raise QualificationError(f"wrapper loaded from unexpected source: {wrapper_source}")
                game_module = sys.modules.get("environment.environment")
                game_file = Path(getattr(game_module, "__file__", "")).resolve()
                if game_file != (source / "UTMIST-AI2-main/environment/environment.py").resolve():
                    raise QualificationError(f"game loaded from unexpected source: {game_file}")
                self.actual_wrapper_config = actual
                self.actual_phase = str(phase.phase_key)
                return env, wrapper
            except BaseException:
                env.close()
                raise

    case = SelectedPhasePipeline()
    case.torch, case.m, case.tmp = torch, native, build_dir
    case.library = native.compile_lanes_cpu(build_dir)
    original_cwd, original_path = Path.cwd(), list(sys.path)
    original_bytecode = sys.dont_write_bytecode
    results = []
    try:
        sys.dont_write_bytecode = True
        for seed, controller in ((45, "adaptive"), (46, "random")):
            result = case._run_pipeline(seed, controller, steps, min_episodes=1)
            result.update(status="passed", seed=seed, controller=controller, phase=case.actual_phase,
                          wrapper_config=dict(case.actual_wrapper_config))
            results.append(result)
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode
    return results


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(DISTANCES), required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args(argv)
    source, config, output, parent_build = (p.resolve() for p in (args.source, args.config, args.output, args.build_dir))
    for path in (output, parent_build):
        if path == source or source in path.parents:
            parser.error("output and build-dir must be outside the frozen source")
    if output == config or output.exists():
        parser.error("output must be a new receipt path and cannot overwrite the input config")
    receipt: dict[str, Any] = {"schema": SCHEMA, "status": "failed", "exit_code": 1, "skipped": 0,
                              "started_at": datetime.now(timezone.utc).isoformat(), "phase": args.phase,
                              "device": args.device, "source": str(source), "config": str(config),
                              "qualification_scope": "python_to_native_cpu" if args.device == "cpu" else "python_to_native_cpu_to_cuda",
                              "cuda_qualified": False, "python_cases": [], "cuda_cases": []}
    try:
        import yaml

        config_bytes = config.read_bytes()
        selected, wrapper = select_phase(yaml.safe_load(config_bytes), args.phase)
        receipt.update(config_sha256=hashlib.sha256(config_bytes).hexdigest(), wrapper_config=wrapper,
                       selected_navigation=selected["phases"][args.phase]["navigation"],
                       frozen_source_hashes=frozen_source_manifest(source), native_sources=native_source_manifest())
        receipt["qualification_source_hashes"] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / "tests/test_native_lane_vec_env.py", ROOT / "benchmark_replay.py",
                      ROOT / "navigation_episode_capture.py", ROOT / "navigation_contact_capture.py")}
        if args.device == "cuda":
            _require_cuda()
        parent_build.mkdir(parents=True, exist_ok=True)
        build_dir = Path(tempfile.mkdtemp(prefix=f"phase-{args.phase}-qualification-", dir=parent_build))
        selected_path = build_dir / "selected-config.yaml"
        selected_path.write_text(yaml.safe_dump(selected, sort_keys=False))
        receipt.update(build_dir=str(build_dir), selected_config=str(selected_path),
                       selected_config_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest())
        cases = run_python_cases(source=source, config=selected_path, wrapper_cfg=wrapper, build_dir=build_dir, steps=STEPS)
        receipt["python_cases"] = cases
        if len(cases) != 2 or [c.get("controller") for c in cases] != ["adaptive", "random"]:
            raise QualificationError("both adaptive and random Python parity cases are required")
        for case in cases:
            if (case.get("status") != "passed" or case.get("skipped", 0) != 0
                    or case.get("ticks") != STEPS or case.get("episodes", 0) < 1):
                raise QualificationError("Python parity case must pass all 1200 ticks with at least one episode")
            require_wrapper_config(wrapper, case.get("wrapper_config", {}))
        receipt["counts"] = {"python_frames": sum(c["ticks"] for c in cases),
                             "python_episodes": sum(c["episodes"] for c in cases)}
        if args.device == "cuda":
            for lanes in (64, 256):
                for stream in ("default", "nondefault"):
                    result = run_case(lanes=lanes, steps=STEPS, seed=45, build_dir=build_dir,
                                      stream=stream, wrapper_cfg=wrapper)
                    receipt["cuda_cases"].append(result)
                    if (result.get("status") != "passed" or result.get("skipped", 0) != 0
                            or result.get("counts", {}).get("episode_records", 0) < 1):
                        raise QualificationError("CUDA parity case did not pass with completed episodes")
                    require_wrapper_config(wrapper, result.get("wrapper_config", {}))
        if frozen_source_manifest(source) != receipt["frozen_source_hashes"]:
            raise QualificationError("frozen source changed during qualification")
        if hashlib.sha256(config.read_bytes()).hexdigest() != receipt["config_sha256"]:
            raise QualificationError("input config changed during qualification")
        receipt.update(status="passed", exit_code=0, cuda_qualified=args.device == "cuda")
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "status": receipt["status"], "phase": args.phase,
                      "cuda_qualified": receipt["cuda_qualified"], "error": receipt.get("error")}))
    return int(receipt["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
