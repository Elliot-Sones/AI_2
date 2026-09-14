#!/usr/bin/env python3
"""Capture and validate frozen navigation reducer cases against native CPU/CUDA.

This tool validates only the post-physics navigation rule reducer: the frozen
Python game still owns physics, contacts, wrappers, action execution, and reset
semantics.  The native code is compared against scalar reducer inputs captured
from the real FrozenOpponentWrapper immediately after each underlying env step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Sequence


CONTRACT_CLI_SCOPE = "postphysics_navigation_rule_reducer_only"
DEFAULT_STEPS = 1200
DEFAULT_SEEDS = "45,46,47"
DEFAULT_CUDA_BATCH_SIZES = (256, 1024, 4096)
REQUIRED_CAPTURE_HELPERS = (
    "read_jsonl",
    "compare_native_results",
    "native_case_from_record",
    "native_result_for_record",
)


class ValidationError(RuntimeError):
    """Fail-closed validation error that should be written to the receipt."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Capture reducer cases from the frozen navigation wrapper")
    capture.add_argument("--source-root", type=Path, required=True)
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--wrapper-source", type=Path)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Steps per seed; hard bounded to 1200")
    capture.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated seeds")
    capture.add_argument("--frame-stack", type=int, default=1)
    capture.add_argument("--controller", choices=("adaptive", "scripted"), default="adaptive")

    verify = subparsers.add_parser("verify", help="Verify a captured JSONL against native CPU and CUDA")
    verify.add_argument("capture", type=Path)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--build-dir", type=Path, default=Path("artifacts/native-navigation-validation/build"))
    verify.add_argument("--device", default="cuda")
    verify.add_argument("--cuda-batch-sizes", default=",".join(str(item) for item in DEFAULT_CUDA_BATCH_SIZES))

    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(label: str, path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists() or not resolved.is_file():
        raise RuntimeError(f"required {label} file is missing: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def source_provenance(*, source_root: Path, config: Path, wrapper_source: Path | None = None) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    if not source_root.exists():
        raise RuntimeError(f"source_root does not exist: {source_root}")
    wrapper_source = source_root / "train.py" if wrapper_source is None else Path(wrapper_source)
    env_path = source_root / "UTMIST-AI2-main" / "environment" / "environment.py"
    positions_path = source_root / "positions.json"
    here = Path(__file__).resolve().parent
    engine_candidates = {
        "warehouse_brawl": source_root / "UTMIST-AI2-main" / "environment" / "WarehouseBrawl.py",
        "constants": source_root / "UTMIST-AI2-main" / "environment" / "constants.py",
    }
    engine_hashes = {
        name: _required_file(name, path) if path.exists() else {"path": str(path.resolve()), "exists": False}
        for name, path in engine_candidates.items()
    }
    return {
        "source_root": str(source_root),
        "config": _required_file("config", Path(config)),
        "wrapper_source": _required_file("wrapper_source", wrapper_source),
        "positions": _required_file("positions", positions_path),
        "frozen_environment": _required_file("frozen_environment", env_path),
        "engine": engine_hashes,
        "native_navigation_contract": _required_file("native_navigation_contract", here / "native_navigation_contract.py"),
        "navigation_contract_capture": _required_file("navigation_contract_capture", here / "navigation_contract_capture.py"),
    }


def native_source_provenance() -> dict[str, Any]:
    native_dir = Path(__file__).resolve().parent / "native"
    names = (
        "navigation_contract.h",
        "navigation_contract.c",
        "navigation_contract.cu",
    )
    return {name: _required_file(name, native_dir / name) for name in names}


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not seeds:
        raise ValidationError("at least one seed is required")
    return seeds


def parse_batch_sizes(raw: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        sizes = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    else:
        sizes = tuple(int(item) for item in raw)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValidationError("cuda batch sizes must be positive")
    return sizes


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("non-finite float in output")
        return value
    return str(value)


def _assert_finite(value: Any, path: str = "value") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"non-finite value at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _ensure_capture_helpers(capture_module: Any) -> None:
    missing = [name for name in REQUIRED_CAPTURE_HELPERS if not hasattr(capture_module, name)]
    if missing:
        raise ValidationError(f"capture helper module missing required helpers: {', '.join(missing)}")


def _truthy_observed_event_names(raw_events: Any) -> list[str]:
    if raw_events is None:
        return []
    if isinstance(raw_events, list):
        names = []
        for index, name in enumerate(raw_events):
            if not isinstance(name, str):
                raise ValidationError(f"observed_events[{index}] must be a string event name")
            if name:
                names.append(name)
        return names
    if isinstance(raw_events, dict):
        return [str(name) for name, value in raw_events.items() if bool(value)]
    raise ValidationError("observed_events must be a list of names")


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    payload = _json_safe(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _failure_receipt(reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "status": "failed",
        "scope": CONTRACT_CLI_SCOPE,
        "reason": reason,
    }
    payload.update(extra)
    return payload


def _cpu_results(
    records: Sequence[dict[str, Any]],
    *,
    native_module: Any,
    capture_module: Any,
    build_dir: Path,
) -> tuple[list[Any], list[Any]]:
    cpu = native_module.load_cpu_reference(build_dir)
    cases = []
    results = []
    for record in records:
        config, state, post = capture_module.native_case_from_record(record)
        cases.append((config, state, post))
        result = capture_module.native_result_for_record(record, cpu.step(config, state, post))
        _assert_finite(result, f"cpu.{record.get('case_id', record.get('tick', '?'))}")
        results.append(result)
    return cases, results


def _torch_cuda_provenance(torch_module: Any, device: str, *, nvcc_path: str | None = None) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        raise ValidationError("Torch CUDA is unavailable; refusing CPU fallback")
    nvcc_path = nvcc_path or shutil.which("nvcc")
    if nvcc_path is None:
        raise ValidationError("nvcc is unavailable; refusing CPU fallback")
    device_name = None
    try:
        index = torch_module.device(device).index if hasattr(torch_module, "device") else None
        device_name = cuda.get_device_name(index if index is not None else 0)
    except Exception:
        device_name = "unknown"
    return {
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "device": str(device),
        "cuda_device_name": device_name,
        "nvcc_path": nvcc_path,
    }


def _cuda_results_for_batch_size(
    records: Sequence[dict[str, Any]],
    cases: Sequence[tuple[Any, Any, Any]],
    *,
    native_module: Any,
    capture_module: Any,
    cuda: Any,
    torch_module: Any,
    device: str,
    batch_size: int,
) -> list[Any]:
    if not records:
        return []
    all_results: list[Any] = []
    for start in range(0, len(records), batch_size):
        chunk_records = list(records[start : start + batch_size])
        chunk_cases = list(cases[start : start + batch_size])
        pad = batch_size - len(chunk_cases)
        if pad:
            chunk_records.extend([chunk_records[-1]] * pad)
            chunk_cases.extend([chunk_cases[-1]] * pad)
        config = chunk_cases[0][0]
        for other_config, _state, _post in chunk_cases[1:]:
            if other_config != config:
                raise ValidationError("CUDA batch contains mixed reducer configs")
        states = [state for _config, state, _post in chunk_cases]
        posts = [post for _config, _state, post in chunk_cases]
        tensors = native_module.tensors_from_cases(config, states, posts, torch=torch_module, device=device)
        native_results = native_module.unpack_cuda_results(cuda.step(tensors))
        if len(native_results) < len(chunk_cases):
            raise ValidationError("CUDA unpack returned fewer lanes than requested")
        actual_lanes = len(chunk_records) - pad
        for record, native_result in zip(chunk_records[:actual_lanes], native_results[:actual_lanes]):
            normalized = capture_module.native_result_for_record(record, native_result)
            _assert_finite(normalized, f"cuda.{record.get('case_id', record.get('tick', '?'))}")
            all_results.append(normalized)
    return all_results


def verify_capture(
    capture_path: Path,
    receipt_path: Path,
    *,
    native_module: Any | None = None,
    capture_module: Any | None = None,
    build_dir: Path | None = None,
    require_cuda: bool = True,
    cuda_batch_sizes: Iterable[int] = DEFAULT_CUDA_BATCH_SIZES,
    device: str = "cuda",
    torch_module: Any | None = None,
    nvcc_path: str | None = None,
) -> int:
    original_cwd = Path.cwd()
    capture_path = Path(capture_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    build_dir = (Path("artifacts/native-navigation-validation/build") if build_dir is None else Path(build_dir)).resolve()
    audit_context: dict[str, Any] = {
        "scope": CONTRACT_CLI_SCOPE,
        "capture_path": str(capture_path),
        "build_dir": str(build_dir),
    }
    _write_receipt(receipt_path, {"status": "running", **audit_context})
    try:
        audit_context["native_sources"] = native_source_provenance()
        native_module = native_module or importlib.import_module("native_navigation_contract")
        capture_module = capture_module or importlib.import_module("navigation_contract_capture")
        _ensure_capture_helpers(capture_module)
        metadata, records = capture_module.read_jsonl(capture_path)
        records = list(records)
        audit_context.update({"metadata": metadata, "records": len(records)})
        if not records:
            _write_receipt(receipt_path, _failure_receipt("zero_records", **audit_context))
            return 1

        cases, cpu_outputs = _cpu_results(records, native_module=native_module, capture_module=capture_module, build_dir=build_dir / "cpu")
        cpu_summary = capture_module.compare_native_results(records, cpu_outputs)
        if not cpu_summary.get("passed"):
            payload = _failure_receipt(
                str(cpu_summary.get("reason", "cpu_mismatch")),
                **audit_context,
                cpu=cpu_summary,
                first_difference={"backend": "cpu", **(cpu_summary.get("first_difference") or {})},
            )
            _write_receipt(receipt_path, payload)
            return 1

        cuda_summaries: list[dict[str, Any]] = []
        torch_provenance = None
        if require_cuda:
            if not bool(native_module.can_run_cuda_contract()):
                raise ValidationError("CUDA native contract is unavailable; refusing CPU fallback")
            if torch_module is None:
                torch_module = importlib.import_module("torch")
            torch_provenance = _torch_cuda_provenance(torch_module, device, nvcc_path=nvcc_path)
            cuda = native_module.load_cuda_contract(build_dir / "cuda")
            for batch_size in parse_batch_sizes(cuda_batch_sizes):
                cuda_outputs = _cuda_results_for_batch_size(
                    records,
                    cases,
                    native_module=native_module,
                    capture_module=capture_module,
                    cuda=cuda,
                    torch_module=torch_module,
                    device=device,
                    batch_size=batch_size,
                )
                summary = capture_module.compare_native_results(records, cuda_outputs)
                summary = {"batch_size": batch_size, **summary}
                cuda_summaries.append(summary)
                if not summary.get("passed"):
                    payload = _failure_receipt(
                        str(summary.get("reason", "cuda_mismatch")),
                        **audit_context,
                        cpu=cpu_summary,
                        cuda=cuda_summaries,
                        torch_cuda=torch_provenance,
                        first_difference={"backend": "cuda", "batch_size": batch_size, **(summary.get("first_difference") or {})},
                    )
                    _write_receipt(receipt_path, payload)
                    return 1

        payload = {
            "status": "passed",
            **audit_context,
            "cpu": cpu_summary,
            "cuda": cuda_summaries,
            "torch_cuda": torch_provenance,
        }
        _write_receipt(receipt_path, payload)
        return 0
    except Exception as exc:
        _write_receipt(receipt_path, _failure_receipt(str(exc), **audit_context, exception_type=type(exc).__name__))
        return 1
    finally:
        os.chdir(original_cwd)


def find_frozen_opponent_wrapper(env: Any) -> Any:
    seen: set[int] = set()
    queue = [env]
    while queue:
        item = queue.pop(0)
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if item.__class__.__name__ == "FrozenOpponentWrapper":
            return item
        for attr in ("env", "unwrapped"):
            try:
                child = getattr(item, attr)
            except Exception:
                child = None
            if child is not None and id(child) not in seen and child is not item:
                queue.append(child)
        for attr in ("envs", "venv"):
            try:
                child = getattr(item, attr)
            except Exception:
                child = None
            if isinstance(child, (list, tuple)):
                queue.extend(child)
            elif child is not None and child is not item:
                queue.append(child)
    raise RuntimeError("FrozenOpponentWrapper not found in navigation environment")


def _reset_env(env: Any) -> None:
    try:
        env.reset()
    except TypeError:
        env.reset(seed=None)


def _adaptive_actions(wrapper: Any, steps: int) -> Iterable[tuple[str, Any]]:
    replay = importlib.import_module("benchmark_replay")
    fixtures = importlib.import_module("benchmark_contact_fixtures")
    controller = fixtures.AdaptiveContactController("navigation", wrapper.action_space)
    for tick in range(steps):
        try:
            state = replay.capture_env_state(wrapper)
        except Exception:
            state = {}
        yield controller.next_action(state, tick)


def _scripted_actions(wrapper: Any, steps: int) -> Iterable[tuple[str, Any]]:
    replay = importlib.import_module("benchmark_replay")
    labels = replay.navigation_fixture_labels(steps)
    actions = replay.scripted_navigation_actions(wrapper.action_space, labels)
    yield from zip(labels, actions)


def capture_validation_records(args: argparse.Namespace) -> dict[str, Any]:
    original_cwd = Path.cwd()
    if args.steps <= 0 or args.steps > DEFAULT_STEPS:
        raise ValidationError(f"--steps must be between 1 and {DEFAULT_STEPS}")
    seeds = parse_seed_list(args.seeds)
    if len(seeds) > 3:
        raise ValidationError("capture supports at most 3 seeds")
    source_root = Path(args.source_root).resolve()
    config = Path(args.config).resolve()
    wrapper_source = Path(args.wrapper_source).resolve() if args.wrapper_source else source_root / "train.py"
    output = Path(args.output).resolve()
    try:
        provenance = source_provenance(source_root=source_root, config=config, wrapper_source=wrapper_source)
        replay = importlib.import_module("benchmark_replay")
        capture = importlib.import_module("navigation_contract_capture")

        records: list[dict[str, Any]] = []
        observed_events: set[str] = set()
        case_labels: set[str] = set()
        for seed in seeds:
            env = None
            try:
                env, _params, phase_config = replay.build_navigation_env(source_root, config, wrapper_source, seed, frame_stack=args.frame_stack)
                wrapper = find_frozen_opponent_wrapper(env)
                _reset_env(env)
                actions = _adaptive_actions(wrapper, args.steps) if args.controller == "adaptive" else _scripted_actions(wrapper, args.steps)
                for tick, (label, action) in enumerate(actions):
                    case_labels.add(label)
                    case_id = f"seed{seed}:tick{tick}:{label}"
                    record = capture.capture_step(wrapper, action, tick=tick, case_id=case_id)
                    record["seed"] = seed
                    record["case_label"] = label
                    truthy_events = _truthy_observed_event_names(record.get("observed_events"))
                    record["observed_events"] = truthy_events
                    observed_events.update(truthy_events)
                    records.append(record)
                    expected = record.get("expected", {})
                    if expected.get("terminated") or expected.get("truncated"):
                        _reset_env(env)
            finally:
                if env is not None:
                    env.close()

        if not records:
            raise ValidationError("capture produced zero records")
        action_count = sum(1 for record in records if record.get("input", {}).get("executed_action") is not None)
        metadata = capture.default_metadata(
            source_train_path=wrapper_source,
            case_source=f"validate_native_navigation:{args.controller}",
            config_path=config,
            extra={
                "contract_cli_scope": CONTRACT_CLI_SCOPE,
                "provenance": provenance,
                "seeds": seeds,
                "steps_per_seed": args.steps,
                "frame_stack": args.frame_stack,
                "controller": args.controller,
                "action_count": action_count,
                "case_labels": sorted(case_labels),
                "truthy_observed_events": sorted(observed_events),
                "capture_note": "Captured from real frozen wrapper before AutoReset; validates reducer only.",
            },
        )
        written = capture.write_jsonl(output, metadata, records)
        return {"status": "captured", "output": str(output), "records": written, "metadata": metadata}
    finally:
        os.chdir(original_cwd)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "capture":
            print(json.dumps(capture_validation_records(args), indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "verify":
            return verify_capture(
                args.capture,
                args.output,
                build_dir=args.build_dir,
                require_cuda=True,
                cuda_batch_sizes=parse_batch_sizes(args.cuda_batch_sizes),
                device=args.device,
            )
        raise ValidationError(f"unknown command {args.command!r}")
    except Exception as exc:
        print(f"validate_native_navigation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
