#!/usr/bin/env python3
"""Qualify actual CUDA native lanes against the unfused CPU lane reference.

All actions are uploaded before rollout. Each block queues steps and clones
their outputs on the device, then drains records and copies the block to the
host. A CPU-only machine emits a failed receipt; it cannot qualify CUDA.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from types import SimpleNamespace
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parent
SCHEMA = "ai2.native_lanes.parity.v1"
ATOL = 1e-6
OBS_WIDTH, STACKED_WIDTH, HISTORY_WIDTH, ACTION_WIDTH = 64, 256, 192, 10
FIELDS = ("observations", "terminal_observations", "rewards", "terminated", "truncated")
CANDIDATE_FILES = (
    "qualify_native_lanes.py", "native_lane_vec_env.py", "benchmark_native_lane_training.py",
    "native_lane_cpu_adapter.py",
    "tensor_ppo.py", "tensor_rollout_buffer.py", "native_navigation_episode.py",
    "native_navigation_contact.py", "native/navigation_spaces.json",
)


class QualificationError(RuntimeError):
    """The candidate did not satisfy an exact or explicitly bounded contract."""


def source_manifest() -> dict[str, Any]:
    import native_lane_vec_env as native

    files = [ROOT / name for name in CANDIDATE_FILES] + list(native.LANE_SOURCE_FILES)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"required qualification source missing: {path}")
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    return {"source_hashes": hashes, "native_sources": native.source_hashes(),
            "candidate_sources": {name: hashes[name] for name in CANDIDATE_FILES}}


def _difference(name: str, expected: np.ndarray, actual: np.ndarray, *, atol: float = 0) -> float:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise QualificationError(f"{name} shape/dtype mismatch: {expected.shape}/{expected.dtype} vs {actual.shape}/{actual.dtype}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise QualificationError(f"{name} contains nonfinite values")
    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    maximum = float(delta.max(initial=0))
    if maximum > atol:
        index = tuple(int(x) for x in np.unravel_index(int(delta.argmax()), delta.shape))
        raise QualificationError(f"{name} differs at {index}: max_abs={maximum}, allowed={atol}")
    return maximum


def _validate_observation(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise QualificationError(f"{name} requires finite float32 observations with shape {shape}")


def compare_reset(cpu: np.ndarray, gpu: np.ndarray) -> dict[str, Any]:
    if cpu.ndim != 2 or cpu.shape[0] < 1:
        raise QualificationError("reset must contain at least one lane")
    for label, values in (("cpu", cpu), ("cuda", gpu)):
        _validate_observation(f"{label} reset", values, (cpu.shape[0], STACKED_WIDTH))
        if np.any(values[:, :HISTORY_WIDTH] != 0):
            raise QualificationError(f"{label} reset history is not zero")
    return {"max_differences": {"reset": _difference("reset", cpu, gpu)},
            "counts": {"reset_lanes": int(cpu.shape[0])}}


def _validate_stack(block: dict[str, np.ndarray], previous: np.ndarray, label: str) -> None:
    obs, terminal = block["observations"], block["terminal_observations"]
    if obs.ndim != 3 or obs.shape[0] < 1 or obs.shape[1] < 1:
        raise QualificationError(f"{label} block must contain steps and lanes")
    steps, lanes = obs.shape[:2]
    _validate_observation(f"{label} observations", obs, (steps, lanes, STACKED_WIDTH))
    _validate_observation(f"{label} terminal_observations", terminal, obs.shape)
    _validate_observation(f"{label} previous observations", previous, (lanes, STACKED_WIDTH))
    for field in ("rewards", "terminated", "truncated"):
        values = block[field]
        dtype = np.float32 if field == "rewards" else np.bool_
        if values.shape != (steps, lanes) or values.dtype != dtype or not np.isfinite(values).all():
            raise QualificationError(f"{label} {field} has invalid shape, dtype, or nonfinite values")
    if np.any(block["terminated"] & block["truncated"]):
        raise QualificationError(f"{label} terminated and truncated overlap")
    for tick in range(steps):
        done = block["terminated"][tick] | block["truncated"][tick]
        history = previous[:, OBS_WIDTH:]
        _difference(f"{label} terminal history step {tick}", history, terminal[tick, :, :HISTORY_WIDTH])
        expected_history = np.where(done[:, None], np.float32(0), history)
        _difference(f"{label} live history step {tick}", expected_history, obs[tick, :, :HISTORY_WIDTH])
        _difference(f"{label} live final frame step {tick}", terminal[tick, ~done, -OBS_WIDTH:], obs[tick, ~done, -OBS_WIDTH:])
        previous = obs[tick]


def compare_block(cpu: dict[str, np.ndarray], gpu: dict[str, np.ndarray],
                  previous_cpu: np.ndarray, previous_gpu: np.ndarray) -> dict[str, Any]:
    if set(cpu) != set(FIELDS) or set(gpu) != set(FIELDS):
        raise QualificationError("step output fields are incomplete or unexpected")
    maxima = {name: _difference(name, cpu[name], gpu[name], atol=ATOL if name == "rewards" else 0) for name in FIELDS}
    _validate_stack(cpu, previous_cpu, "cpu")
    _validate_stack(gpu, previous_gpu, "cuda")
    return {"max_differences": maxima, "counts": {
        "transitions": int(cpu["rewards"].size),
        "terminated": int(np.count_nonzero(cpu["terminated"])),
        "truncated": int(np.count_nonzero(cpu["truncated"])),
        "frame_stack_checks": int(cpu["rewards"].size),
        "reward_values_different": int(np.count_nonzero(cpu["rewards"] != gpu["rewards"])),
    }}


def _indexed_infos(infos: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    indexed = {}
    required = {"lane", "episode_index", "success", "fell", "targets", "horizon", "terminated", "truncated"}
    for info in infos:
        if not isinstance(info, dict) or not isinstance(info.get("navigation"), dict) or not isinstance(info.get("episode"), dict):
            raise QualificationError("episode record is incomplete")
        nav, ep = info["navigation"], info["episode"]
        if not required.issubset(nav) or not {"r", "l"}.issubset(ep):
            raise QualificationError("episode record fields are incomplete")
        if not np.isfinite(ep["r"]) or ep["l"] < 1 or nav["lane"] < 0 or nav["episode_index"] < 0:
            raise QualificationError("episode record contains invalid values")
        key = (nav["lane"], nav["episode_index"])
        if key in indexed:
            raise QualificationError(f"duplicate episode record: {key}")
        indexed[key] = info
    return indexed


def compare_episode_infos(cpu: list[dict[str, Any]], gpu: list[dict[str, Any]]) -> dict[str, Any]:
    left, right = _indexed_infos(cpu), _indexed_infos(gpu)
    if left.keys() != right.keys():
        raise QualificationError("episode record identities differ")
    maximum, different = 0.0, 0
    for key in left:
        c, g = left[key], right[key]
        if c["navigation"] != g["navigation"] or c["episode"]["l"] != g["episode"]["l"]:
            raise QualificationError(f"episode record flags/length differ: {key}")
        delta = abs(float(c["episode"]["r"]) - float(g["episode"]["r"]))
        if delta > ATOL:
            raise QualificationError(f"episode return differs: {key}, max_abs={delta}, allowed={ATOL}")
        maximum = max(maximum, delta)
        different += int(delta != 0)
    return {"max_differences": {"episode_returns": maximum},
            "counts": {"episode_records": len(left), "episode_returns_different": different}}


def require_episode_count(block_result: dict[str, Any], infos: list[dict[str, Any]]) -> None:
    counts = block_result["counts"]
    if len(infos) != counts["terminated"] + counts["truncated"]:
        raise QualificationError("episode record count does not match terminal events (possible overflow)")


def _merge_metrics(total: dict[str, Any], result: dict[str, Any]) -> None:
    for name, value in result["max_differences"].items():
        total["max_differences"][name] = max(total["max_differences"].get(name, 0), value)
    for name, value in result["counts"].items():
        total["counts"][name] = total["counts"].get(name, 0) + value


def _queue_block(env: Any, actions: Any) -> dict[str, Any]:
    """No host reads, copies, error queries or synchronization inside this loop."""
    import torch

    snapshots = {name: [] for name in FIELDS}
    with torch.no_grad():
        for tick in range(actions.shape[0]):
            output = env.step_tensor(actions[tick])
            for name in FIELDS:
                snapshots[name].append(getattr(output, name).clone())
        return {name: torch.stack(values) for name, values in snapshots.items()}


def _host_block(block: dict[str, Any]) -> dict[str, np.ndarray]:
    return {name: value.cpu().numpy() for name, value in block.items()}


def _require_cuda() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise QualificationError("actual CUDA device required; CPU fallback cannot qualify native lanes")
    if shutil.which("nvcc") is None:
        raise QualificationError("nvcc required to compile the CUDA lane candidate")
    return torch


def run_case(*, lanes: int, steps: int, seed: int, build_dir: Path, block_size: int = 128,
             stream: str = "default", wrapper_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    import native_lane_vec_env as native

    torch = _require_cuda()
    if min(lanes, steps, block_size) < 1 or stream not in ("default", "nondefault"):
        raise ValueError("lanes, steps and block_size must be positive; stream must be default or nondefault")
    effective_wrapper = dict(native.PHASE_0B if wrapper_cfg is None else wrapper_cfg)
    if not effective_wrapper:
        raise ValueError("wrapper_cfg must specify the complete wrapper configuration")
    started = time.perf_counter()
    selected = torch.cuda.default_stream() if stream == "default" else torch.cuda.Stream()
    result = {"status": "running", "lanes": lanes, "steps": steps, "seed": seed, "stream": stream,
              "block_size": block_size, "wrapper_config": effective_wrapper, "max_differences": {},
              "counts": {"rollout_blocks": 0, "per_step_host_transfers": 0}}
    # Construction and reset belong to this stream too; this catches initialization
    # that accidentally relies on default-stream ordering.
    with torch.cuda.stream(selected), ExitStack() as cleanup:
        cpu = native.build_cpu_env(lanes, build_dir=build_dir, seed=seed, contract="off", record_capacity=lanes * block_size,
                                   wrapper_cfg=wrapper_cfg)
        cleanup.callback(cpu.close)
        gpu = native.build_cuda_env(lanes, build_dir=build_dir, seed=seed, record_capacity=lanes * block_size,
                                    wrapper_cfg=wrapper_cfg)
        cleanup.callback(gpu.close)
        expected_cfg = native.make_config(native.make_reference(cpu.backend.library_path), effective_wrapper)
        expected = {name: getattr(expected_cfg, name) for name, _ctype in expected_cfg._fields_ if name != "pad"}
        for backend, env in (("cpu", cpu), ("cuda", gpu)):
            actual = {name: getattr(env.backend.cfg, name) for name in expected}
            if actual != expected:
                raise QualificationError(f"{backend} effective config differs from requested wrapper config: {actual} != {expected}")
        result["effective_config"] = expected
        previous_cpu = cpu.reset_tensor().clone().numpy()
        previous_gpu = gpu.reset_tensor().clone().cpu().numpy()
        _merge_metrics(result, compare_reset(previous_cpu, previous_gpu))
        rng = np.random.default_rng(seed)
        actions = (rng.random((steps, lanes, ACTION_WIDTH)) < 0.35).astype(np.float32)
        # Keep one idle lane to exercise the configured horizon as well as random
        # action lanes' falls/autoresets. The same action bytes drive both backends.
        actions[:, 0, :] = 0
        cpu_actions = torch.from_numpy(actions)
        cuda_actions = cpu_actions.to(gpu.device)
        for start in range(0, steps, block_size):
            stop = min(start + block_size, steps)
            cpu_block = _host_block(_queue_block(cpu, cpu_actions[start:stop]))
            device_block = _queue_block(gpu, cuda_actions[start:stop])
            # Exactly one drain per backend/block. The native drain synchronizes
            # and reports errors here, after all block work has been submitted.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                cpu_infos = cpu.drain_episode_infos()
                gpu_infos = gpu.drain_episode_infos()
            gpu_block = _host_block(device_block)
            comparison = compare_block(cpu_block, gpu_block, previous_cpu, previous_gpu)
            require_episode_count(comparison, cpu_infos)
            require_episode_count(comparison, gpu_infos)
            _merge_metrics(result, comparison)
            _merge_metrics(result, compare_episode_infos(cpu_infos, gpu_infos))
            previous_cpu = cpu_block["observations"][-1].copy()
            previous_gpu = gpu_block["observations"][-1].copy()
            result["counts"]["rollout_blocks"] += 1
        # This final explicit query also covers a partial final rollout block.
        error, lane = gpu.backend.error()
        if error:
            raise native.LaneError(error, "qualification rollout end", lane)
        result["rollout_end_error"] = error
        result["libraries"] = {"cpu": str(cpu.backend.library_path), "cuda": str(gpu.backend.library_path)}
        result["library_hashes"] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                    for name, path in result["libraries"].items()}
    result.update(status="passed", wall_seconds=time.perf_counter() - started)
    return result


def check_async_error(*, build_dir: Path) -> dict[str, Any]:
    """A device-side nonfinite action error must survive until the rollout drain."""
    import native_lane_vec_env as native

    torch = _require_cuda()
    with torch.cuda.stream(torch.cuda.Stream()), ExitStack() as cleanup:
        env = native.build_cuda_env(2, build_dir=build_dir, seed=45, record_capacity=256)
        cleanup.callback(env.close)
        env.reset_tensor()
        actions = torch.zeros((8, 2, ACTION_WIDTH), dtype=torch.float32)
        actions[0, 1, 0] = float("nan")
        actions = actions.to(env.device)
        _queue_block(env, actions)
        try:
            env.drain_episode_infos()
        except native.LaneError as exc:
            if exc.code != 9:
                raise
            return {"status": "passed", "stream": "nondefault", "queued_steps": 8,
                    "observed_error_code": exc.code, "observed_at": "rollout_drain"}
        raise QualificationError("nonfinite device actions did not fail at the rollout drain")


def _version(command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, check=False, text=True, capture_output=True, timeout=20)
        return {"command": command, "exit_code": proc.returncode, "output": (proc.stdout + proc.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def runtime_manifest(torch: Any) -> dict[str, Any]:
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "device": {"index": index, "name": props.name, "capability": list(torch.cuda.get_device_capability(index)),
                       "total_memory": props.total_memory},
            "compiler": {"cpu": _version([os.environ.get("CC") or "cc", "--version"]),
                         "cuda": _version([shutil.which("nvcc"), "--version"])},
            "driver": _version(["nvidia-smi", "--query-gpu=driver_version,name,uuid", "--format=csv,noheader"]),
            "cpu_flags": ["-O2", "-ffp-contract=off", "-fno-fast-math"],
            "cuda_flags": ["-O2", "-fmad=false", "-Xcompiler", "-ffp-contract=off"]}


def compile_qualification_libraries(build_dir: Path) -> list[dict[str, Any]]:
    """Capture commands from the existing builders while producing fresh binaries."""
    import native_lane_vec_env as native

    commands = []
    original = native.subprocess

    def recorded_run(command, *args, **kwargs):
        entry = {"command": list(command), "cwd": str(kwargs.get("cwd", Path.cwd()))}
        commands.append(entry)
        result = original.run(command, *args, **kwargs)
        entry["exit_code"] = result.returncode
        return result

    # Restrict instrumentation to this module and synchronous compilation only;
    # restore its subprocess reference before constructing either environment.
    native.subprocess = SimpleNamespace(run=recorded_run)
    try:
        native.compile_lanes_cpu(build_dir, "off")
        native.compile_lanes_cuda(build_dir)
    finally:
        native.subprocess = original
    if len(commands) != 2:
        raise QualificationError("qualification must freshly compile both CPU and CUDA libraries")
    return commands


def _positive_csv(value: str) -> list[int]:
    values = [int(part) for part in value.split(",")]
    if not values or min(values) < 1 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("provide distinct positive comma-separated integers")
    return values


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON qualification receipt")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build" / "native-lanes")
    parser.add_argument("--lanes", type=_positive_csv, default=[64, 256])
    parser.add_argument("--seeds", type=_positive_csv, default=[45, 46, 47])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.block_size < 1:
        parser.error("steps and block-size must be positive")
    receipt = {"schema": SCHEMA, "status": "failed", "exit_code": 1, "skipped": 0,
               "started_at": datetime.now(timezone.utc).isoformat(), "cases": [],
               "configuration": {"lanes": args.lanes, "seeds": args.seeds, "steps": args.steps,
                                 "block_size": args.block_size, "reward_atol": ATOL, "return_atol": ATOL,
                                 "observations_exact": True, "flags_exact": True, "cpu_contract": "off",
                                 "actions": "seeded Bernoulli(0.35), lane 0 idle", "cuda_required": True}}
    try:
        receipt.update(source_manifest())
        torch = _require_cuda()
        receipt["runtime"] = runtime_manifest(torch)
        # A private empty build directory makes the receipt independent of an
        # older shared library's timestamp cache. Keep binaries for audit.
        args.build_dir.mkdir(parents=True, exist_ok=True)
        build_dir = Path(tempfile.mkdtemp(prefix="qualification-", dir=args.build_dir))
        receipt["build_dir"] = str(build_dir)
        receipt["compile_commands"] = compile_qualification_libraries(build_dir)
        for lanes in args.lanes:
            for seed in args.seeds:
                case = run_case(lanes=lanes, steps=args.steps, seed=seed, build_dir=build_dir, block_size=args.block_size)
                if not case["counts"].get("episode_records", 0):
                    raise QualificationError("no completed episodes; increase --steps to cover terminal/reset/record behavior")
                receipt["cases"].append(case)
        # Repeat construction, reset and a rollout on independent nondefault
        # streams, with no default-stream wait added by this harness.
        for repeat in range(2):
            case = run_case(lanes=min(args.lanes), steps=min(args.steps, max(650, 2 * args.block_size)),
                            seed=args.seeds[0], build_dir=build_dir, block_size=args.block_size, stream="nondefault")
            case["repeat"] = repeat
            receipt["cases"].append(case)
        receipt["async_error_check"] = check_async_error(build_dir=build_dir)
        after = source_manifest()
        if after != {name: receipt[name] for name in after}:
            raise QualificationError("qualification sources changed during the run")
        receipt.update(status="passed", exit_code=0)
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output), "error": receipt.get("error")}))
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
