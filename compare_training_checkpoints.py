#!/usr/bin/env python3
"""Compare SB3 checkpoint behavior state without loading or running a policy.

The comparison intentionally ignores whole-zip byte equality because zip member
metadata can differ without changing policy/optimizer tensors or PPO scalar
metadata. It loads only checkpoint members on CPU.

Fail-closed rules (v2): every scalar metadata key must be present on both
sides; a state section with no tensors is a failure, not a vacuous pass;
numerically equal tensors must also be bitwise identical (signed zeros and
NaN payloads count); non-numeric array dtypes are rejected; comparing a file
with itself or with a byte-identical copy is an error, not a result; and
``--expect KEY=VALUE`` pins metadata so the wrong checkpoints (for example two
untrained ``initial.zip`` files) cannot pass as a trained-state comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_graph_rollout_parity import digest_any
from benchmark_tensor_rollout_parity import flatten_state


SCHEMA = "ai2.checkpoint_behavior_compare.v2"
STATE_MEMBERS = {
    "policy": "policy.pth",
    "optimizer": "policy.optimizer.pth",
}
SCALAR_METADATA_KEYS = (
    "num_timesteps",
    "_n_updates",
    "n_steps",
    "n_envs",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "ent_coef",
)


class CheckpointCompareError(RuntimeError):
    """Raised for malformed or unsupported checkpoint inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    record: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        stat = resolved.stat()
        record.update({"size": stat.st_size, "sha256": sha256_file(resolved)})
    return record


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_zip_member(zip_path: Path, member: str) -> Any:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            try:
                payload = archive.read(member)
            except KeyError as exc:
                raise CheckpointCompareError(f"checkpoint {zip_path} missing required member {member}") from exc
    except zipfile.BadZipFile as exc:
        raise CheckpointCompareError(f"checkpoint is not a valid zip: {zip_path}") from exc

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise CheckpointCompareError("torch is required to load checkpoint tensors") from exc

    try:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exact torch exception varies by version.
        raise CheckpointCompareError(f"failed to torch.load {member} from {zip_path}: {exc}") from exc


def load_metadata(zip_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            try:
                raw = archive.read("data")
            except KeyError as exc:
                raise CheckpointCompareError(f"checkpoint {zip_path} missing required member data") from exc
    except zipfile.BadZipFile as exc:
        raise CheckpointCompareError(f"checkpoint is not a valid zip: {zip_path}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointCompareError(f"checkpoint {zip_path} has non-JSON data member") from exc
    if not isinstance(data, dict):
        raise CheckpointCompareError(f"checkpoint {zip_path} data member must be a JSON object")
    return data


def load_checkpoint(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": resolved,
        "policy": load_zip_member(resolved, STATE_MEMBERS["policy"]),
        "optimizer": load_zip_member(resolved, STATE_MEMBERS["optimizer"]),
        "metadata": load_metadata(resolved),
    }


def _is_array(value: Any) -> bool:
    return isinstance(value, np.ndarray)


def _is_numeric_array(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)


def _nonfinite_mismatch(name: str, left: np.ndarray, right: np.ndarray) -> dict[str, Any] | None:
    if not _is_numeric_array(left) or not _is_numeric_array(right):
        return None
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    if bool(np.all(left_finite)) and bool(np.all(right_finite)):
        return None
    if left.shape == right.shape:
        bad = ~(left_finite & right_finite)
        index = list(map(int, np.argwhere(bad)[0])) if bad.size else []
    else:
        index = []
    return {
        "name": name,
        "kind": "nonfinite",
        "index": index,
        "baseline_finite": bool(np.all(left_finite)),
        "candidate_finite": bool(np.all(right_finite)),
    }


def _array_value_mismatch(name: str, left: np.ndarray, right: np.ndarray) -> tuple[dict[str, Any] | None, float]:
    if left.shape != right.shape:
        return {"name": name, "kind": "shape", "baseline": list(left.shape), "candidate": list(right.shape)}, math.inf
    if left.dtype != right.dtype:
        return {"name": name, "kind": "dtype", "baseline": str(left.dtype), "candidate": str(right.dtype)}, 0.0
    if not _is_numeric_array(left):
        # Object/string/void arrays have no meaningful elementwise numeric contract; fail closed.
        return {"name": name, "kind": "unsupported_dtype", "dtype": str(left.dtype)}, math.inf
    nonfinite = _nonfinite_mismatch(name, left, right)
    if nonfinite is not None:
        return nonfinite, math.inf
    if left.dtype == np.bool_:
        bad = left != right
        if bool(np.any(bad)):
            index = list(map(int, np.argwhere(bad)[0]))
            return {"name": name, "kind": "value", "index": index, "baseline": bool(left[tuple(index)]), "candidate": bool(right[tuple(index)]), "abs_diff": 1.0}, 1.0
        return None, 0.0
    diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
    max_diff = float(diff.max(initial=0.0))
    bad = diff != 0
    if bool(np.any(bad)):
        index = list(map(int, np.argwhere(bad)[0]))
        return {
            "name": name,
            "kind": "value",
            "index": index,
            "baseline": float(left[tuple(index)]),
            "candidate": float(right[tuple(index)]),
            "abs_diff": float(diff[tuple(index)]),
        }, max_diff
    # Numerically equal is not enough for an "exact" claim: signed zeros, NaN payloads and
    # wide integers that survive the float64 cast unchanged must also match bit for bit.
    if left.tobytes() != right.tobytes():
        return {"name": name, "kind": "bits", "abs_diff": 0.0}, 0.0
    return None, max_diff


def _values_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def compare_state_section(name: str, baseline_state: Any, candidate_state: Any) -> dict[str, Any]:
    baseline = flatten_state(baseline_state)
    candidate = flatten_state(candidate_state)
    keys = sorted(set(baseline) | set(candidate))
    missing_baseline = [key for key in keys if key not in baseline]
    missing_candidate = [key for key in keys if key not in candidate]
    mismatches: list[dict[str, Any]] = []
    first_mismatch: dict[str, Any] | None = None
    max_abs_diff = 0.0
    finite = True
    tensor_count = 0
    param_count = 0

    for key in keys:
        if key in missing_baseline or key in missing_candidate:
            continue
        left = baseline[key]
        right = candidate[key]
        if _is_array(left) or _is_array(right):
            if not _is_array(left) or not _is_array(right):
                mismatch = {
                    "name": key,
                    "kind": "type",
                    "baseline": type(left).__name__,
                    "candidate": type(right).__name__,
                }
                mismatches.append(mismatch)
                first_mismatch = first_mismatch or mismatch
                finite = False
                continue
            tensor_count += 1
            if name == "policy":
                param_count += int(left.size)
            mismatch, diff = _array_value_mismatch(key, left, right)
            if math.isfinite(max_abs_diff) and math.isfinite(diff):
                max_abs_diff = max(max_abs_diff, diff)
            elif not math.isfinite(diff):
                max_abs_diff = diff
            if mismatch is not None:
                if mismatch["kind"] in ("nonfinite", "unsupported_dtype"):
                    finite = False
                mismatches.append(mismatch)
                first_mismatch = first_mismatch or mismatch
        elif not _values_equal(left, right):
            mismatch = {
                "name": key,
                "kind": "value",
                "baseline": repr(left),
                "candidate": repr(right),
                "type_baseline": type(left).__name__,
                "type_candidate": type(right).__name__,
            }
            mismatches.append(mismatch)
            first_mismatch = first_mismatch or mismatch

    if tensor_count == 0 or (name == "policy" and param_count == 0):
        # Nothing was actually compared; an empty section must never read as a pass.
        mismatch = {"name": name, "kind": "empty", "tensor_count": tensor_count, "param_count": param_count}
        mismatches.append(mismatch)
        first_mismatch = first_mismatch or mismatch

    if missing_baseline and first_mismatch is None:
        first_mismatch = {"kind": "missing_baseline", "name": missing_baseline[0]}
    if missing_candidate and first_mismatch is None:
        first_mismatch = {"kind": "missing_candidate", "name": missing_candidate[0]}

    digest_baseline = digest_any(baseline)
    digest_candidate = digest_any(candidate)
    if digest_baseline != digest_candidate and not mismatches and not missing_baseline and not missing_candidate:
        # Elementwise checks found nothing, yet the canonical byte digests differ (for example a
        # scalar 0.0 versus -0.0 outside any array). Report it rather than trust the weaker check.
        mismatch = {"name": name, "kind": "digest", "baseline": digest_baseline, "candidate": digest_candidate}
        mismatches.append(mismatch)
        first_mismatch = first_mismatch or mismatch

    exact = not missing_baseline and not missing_candidate and not mismatches
    return {
        "member": STATE_MEMBERS[name],
        "exact": exact,
        "finite": finite,
        "digest_baseline": digest_baseline,
        "digest_candidate": digest_candidate,
        "item_count_baseline": len(baseline),
        "item_count_candidate": len(candidate),
        "tensor_count": tensor_count,
        "param_count": param_count if name == "policy" else None,
        "missing_baseline": missing_baseline,
        "missing_candidate": missing_candidate,
        "mismatch_count": len(mismatches) + len(missing_baseline) + len(missing_candidate),
        "first_mismatch": first_mismatch,
        "mismatches": mismatches[:25],
        "max_abs_diff": max_abs_diff,
    }


def _metadata_finite(key: str, value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def compare_metadata(
    baseline: dict[str, Any], candidate: dict[str, Any], expect: dict[str, Any] | None = None
) -> dict[str, Any]:
    expected = dict(expect or {})
    keys = list(SCALAR_METADATA_KEYS) + [key for key in expected if key not in SCALAR_METADATA_KEYS]
    mismatches: list[dict[str, Any]] = []
    values: dict[str, dict[str, Any]] = {}
    missing_baseline: list[str] = []
    missing_candidate: list[str] = []
    finite = True
    for key in keys:
        left_present = key in baseline
        right_present = key in candidate
        left = baseline.get(key)
        right = candidate.get(key)
        values[key] = {
            "baseline": left,
            "candidate": right,
            "baseline_present": left_present,
            "candidate_present": right_present,
        }
        if not left_present:
            missing_baseline.append(key)
        if not right_present:
            missing_candidate.append(key)
        if not (left_present and right_present):
            # A key absent from both checkpoints must not compare as None == None.
            mismatches.append(
                {"key": key, "kind": "missing", "baseline_present": left_present, "candidate_present": right_present}
            )
            continue
        finite = finite and _metadata_finite(key, left) and _metadata_finite(key, right)
        if left != right or type(left) is not type(right):
            mismatches.append(
                {
                    "key": key,
                    "kind": "value",
                    "baseline": left,
                    "candidate": right,
                    "type_baseline": type(left).__name__,
                    "type_candidate": type(right).__name__,
                }
            )
            continue
        if key in expected and (left != expected[key] or type(left) is not type(expected[key])):
            mismatches.append(
                {"key": key, "kind": "expected", "expected": expected[key], "baseline": left, "candidate": right}
            )
    return {
        "keys": keys,
        "values": values,
        "expected": expected,
        "missing_baseline": missing_baseline,
        "missing_candidate": missing_candidate,
        "mismatches": mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "finite": finite,
        "exact": not mismatches and finite,
    }


def compare_checkpoints(
    baseline_path: Path | str, candidate_path: Path | str, expect: dict[str, Any] | None = None
) -> dict[str, Any]:
    baseline_path = Path(baseline_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    if baseline_path == candidate_path:
        raise CheckpointCompareError(f"baseline and candidate are the same file: {baseline_path}")
    baseline_record = file_record(baseline_path)
    candidate_record = file_record(candidate_path)
    if baseline_record.get("sha256") is not None and baseline_record.get("sha256") == candidate_record.get("sha256"):
        raise CheckpointCompareError(
            "baseline and candidate are byte-identical files; the comparison would be vacuous: "
            f"{baseline_path} == {candidate_path}"
        )
    baseline = load_checkpoint(baseline_path)
    candidate = load_checkpoint(candidate_path)
    policy = compare_state_section("policy", baseline["policy"], candidate["policy"])
    optimizer = compare_state_section("optimizer", baseline["optimizer"], candidate["optimizer"])
    metadata = compare_metadata(baseline["metadata"], candidate["metadata"], expect=expect)
    finite = bool(policy["finite"] and optimizer["finite"] and metadata["finite"])
    exact_match = bool(policy["exact"] and optimizer["exact"] and metadata["exact"] and finite)
    return {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "operation": "cpu_torch_load_weights_only_no_policy_execution",
        "comparison_config": {
            "state_members": STATE_MEMBERS,
            "scalar_metadata_keys": list(SCALAR_METADATA_KEYS),
            "expected_metadata": dict(expect or {}),
            "map_location": "cpu",
            "weights_only": True,
            "exact_tensor_values": True,
            "bitwise_tensor_equality": True,
            "require_all_metadata_keys": True,
            "reject_empty_sections": True,
            "reject_identical_inputs": True,
            "compare_zip_container_hash_as_behavior": False,
        },
        "inputs": {
            "baseline": baseline_record,
            "candidate": candidate_record,
        },
        "source_hashes": {
            "compare_training_checkpoints.py": sha256_file(Path(__file__).resolve()),
        },
        "policy": policy,
        "optimizer": optimizer,
        "metadata": metadata,
        "finite": finite,
        "exact_match": exact_match,
        "exit_code": 0 if exact_match else 1,
    }


def parse_expect(items: list[str] | None, parser: argparse.ArgumentParser | None = None) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for item in items or []:
        key, separator, raw_value = item.partition("=")
        key = key.strip()
        if not separator or not key:
            message = f"--expect requires KEY=VALUE, got {item!r}"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        try:
            value: Any = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        expected[key] = value
    return expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline SB3 checkpoint zip")
    parser.add_argument("--candidate", type=Path, required=True, help="candidate SB3 checkpoint zip")
    parser.add_argument("--output", type=Path, required=True, help="fresh JSON report path")
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="require this metadata value on both sides (JSON-parsed; repeatable), e.g. num_timesteps=19343392",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    expect = parse_expect(args.expect, parser)
    output = Path(args.output).resolve()
    if output.exists():
        print(f"refusing to overwrite existing output: {output}", file=sys.stderr)
        return 2
    try:
        result = compare_checkpoints(args.baseline, args.candidate, expect=expect)
        write_json(output, result)
        return int(result["exit_code"])
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "created_unix": time.time(),
            "status": "error",
            "error": str(exc),
            "expected_metadata": expect,
            "inputs": {
                "baseline": file_record(Path(args.baseline)),
                "candidate": file_record(Path(args.candidate)),
            },
            "exit_code": 2,
        }
        write_json(output, failure)
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
