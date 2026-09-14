"""Summarize paired throughput campaign artifacts and audit SB3 checkpoints."""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = [45, 46, 47]
ARMS = ["control", "cache"]


class SummaryError(RuntimeError):
    """Raised when campaign artifacts are missing or internally incomplete."""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return clean[int(rank)]
    return clean[low] + (clean[high] - clean[low]) * (rank - low)


def geometric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    if any(value <= 0 for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_run_name(path: Path) -> tuple[int, str] | None:
    prefix = "paired-cache-seed"
    name = path.name
    if not name.startswith(prefix):
        return None
    try:
        seed_text, arm = name[len(prefix) :].rsplit("-", 1)
        seed = int(seed_text)
    except ValueError:
        return None
    if arm not in ARMS:
        return None
    return seed, arm


def measured_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("phase") == "measured"]


def profile_breakdown(run: dict[str, Any]) -> list[dict[str, Any]]:
    total = float(run["measured_cycle_seconds"])
    stages = [
        ("collect", float(run.get("measured_collect_seconds", 0.0))),
        ("update", float(run.get("measured_update_seconds", 0.0))),
        ("boundary", float(run.get("measured_boundary_seconds", 0.0))),
    ]
    residual = total - sum(seconds for _name, seconds in stages)
    stages.append(("residual", residual))
    return [
        {
            "stage": name,
            "seconds": seconds,
            "share_of_measured_cycle": seconds / total if total else None,
        }
        for name, seconds in stages
    ]


def _as_finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise SummaryError(f"{label} must be finite")
    return number


def _validate_run_metrics(summary: dict[str, Any], measured: list[dict[str, Any]], run_dir: Path) -> None:
    plan_cycles = int(summary.get("plan", {}).get("cycles", -1))
    completed_cycles = int(summary.get("completed_measured_cycles", -1))
    row_count = len(measured)
    if plan_cycles < 0 or completed_cycles < 0 or row_count != plan_cycles or row_count != completed_cycles:
        raise SummaryError(
            f"measured row count mismatch for {run_dir.name}: "
            f"rows={row_count} plan.cycles={plan_cycles} completed={completed_cycles}"
        )

    indices = [row.get("cycle_index") for row in measured]
    if any(not isinstance(index, int) for index in indices):
        raise SummaryError(f"measured cycle indices must be contiguous unique integers for {run_dir.name}")
    expected_indices = list(range(min(indices), min(indices) + len(indices))) if indices else []
    if sorted(indices) != expected_indices or len(set(indices)) != len(indices):
        raise SummaryError(f"measured cycle indices must be contiguous unique integers for {run_dir.name}")

    summary_transitions = _as_finite_float(summary.get("measured_transitions"), "measured_transitions")
    row_transitions = sum(_as_finite_float(row.get("transitions"), "row transitions") for row in measured)
    if not math.isclose(row_transitions, summary_transitions, rel_tol=0.0, abs_tol=1e-6):
        raise SummaryError(
            f"transitions mismatch for {run_dir.name}: rows={row_transitions} summary={summary_transitions}"
        )

    summary_seconds = _as_finite_float(summary.get("measured_cycle_seconds"), "measured_cycle_seconds")
    row_seconds = sum(_as_finite_float(row.get("cycle_seconds"), "row cycle_seconds") for row in measured)
    if not math.isclose(row_seconds, summary_seconds, rel_tol=1e-9, abs_tol=1e-6):
        raise SummaryError(f"timing mismatch for {run_dir.name}: rows={row_seconds} summary={summary_seconds}")

    if summary_seconds <= 0:
        raise SummaryError(f"timing mismatch for {run_dir.name}: measured_cycle_seconds must be positive")
    summary_sps = _as_finite_float(summary.get("measured_transitions_per_second"), "measured_transitions_per_second")
    reconciled_sps = summary_transitions / summary_seconds
    if not math.isclose(reconciled_sps, summary_sps, rel_tol=1e-9, abs_tol=1e-6):
        raise SummaryError(f"SPS reconciliation mismatch for {run_dir.name}: computed={reconciled_sps} summary={summary_sps}")

    for row in measured:
        transitions = _as_finite_float(row.get("transitions"), "row transitions")
        seconds = _as_finite_float(row.get("cycle_seconds"), "row cycle_seconds")
        row_sps = _as_finite_float(row.get("transitions_per_second"), "row transitions_per_second")
        if seconds <= 0:
            raise SummaryError(f"timing mismatch for {run_dir.name}: row cycle_seconds must be positive")
        computed = transitions / seconds
        if not math.isclose(computed, row_sps, rel_tol=1e-9, abs_tol=1e-6):
            raise SummaryError(f"SPS row mismatch for {run_dir.name}: cycle_index={row.get('cycle_index')}")


def load_run(run_dir: Path, *, allow_incomplete: bool) -> dict[str, Any]:
    parsed = parse_run_name(run_dir)
    if parsed is None:
        raise SummaryError(f"unexpected run directory name: {run_dir.name}")
    seed, arm = parsed
    summary_path = run_dir / "campaign_summary.json"
    metrics_path = run_dir / "campaign_metrics.jsonl"
    if not summary_path.is_file():
        if allow_incomplete:
            return {
                "seed": seed,
                "arm": arm,
                "name": run_dir.name,
                "path": str(run_dir),
                "status": "missing_summary",
                "metrics": read_jsonl(metrics_path),
            }
        raise SummaryError(f"missing run summary: {summary_path}")

    summary = read_json(summary_path)
    if summary.get("status") != "completed":
        if not allow_incomplete:
            raise SummaryError(f"incomplete run summary: {summary_path} status={summary.get('status')!r}")

    metrics = read_jsonl(metrics_path)
    measured = measured_rows(metrics)
    try:
        _validate_run_metrics(summary, measured, run_dir)
        validation_error = None
        status = summary.get("status")
    except SummaryError as exc:
        if not allow_incomplete:
            raise
        validation_error = str(exc)
        status = "invalid"
    run = {
        "seed": seed,
        "arm": arm,
        "name": run_dir.name,
        "path": str(run_dir),
        "status": status,
        "sps": summary.get("measured_transitions_per_second"),
        "elapsed_seconds": summary.get("measured_cycle_seconds"),
        "learning_wall_seconds": summary.get("learning_wall_seconds"),
        "collect_seconds": summary.get("measured_collect_seconds"),
        "update_seconds": summary.get("measured_update_seconds"),
        "boundary_seconds": summary.get("measured_boundary_seconds"),
        "startup_seconds": summary.get("startup_seconds"),
        "measured_transitions": summary.get("measured_transitions"),
        "completed_measured_cycles": summary.get("completed_measured_cycles"),
        "plan": summary.get("plan", {}),
        "metrics_path": str(metrics_path),
        "measured_window": measured_window(measured),
        "per_cycle_sps": [row.get("transitions_per_second") for row in measured if row.get("transitions_per_second") is not None],
        "profile_breakdown": profile_breakdown(summary),
    }
    if validation_error is not None:
        run["validation_error"] = validation_error
    return run


def measured_window(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    starts = []
    ends = []
    for row in rows:
        end = row.get("timestamp_unix")
        cycle_seconds = row.get("cycle_seconds")
        if end is None or cycle_seconds is None:
            continue
        end = float(end)
        starts.append(end - float(cycle_seconds))
        ends.append(end)
    if not starts or not ends:
        return None
    return {"start_unix": min(starts), "end_unix": max(ends)}


def discover_runs(root: Path) -> dict[tuple[int, str], Path]:
    runs: dict[tuple[int, str], Path] = {}
    for child in sorted(root.iterdir() if root.is_dir() else []):
        if not child.is_dir():
            continue
        parsed = parse_run_name(child)
        if parsed is not None:
            runs[parsed] = child
    return runs


def validate_expected_runs(run_paths: dict[tuple[int, str], Path], expected_seeds: list[int]) -> None:
    for seed in expected_seeds:
        for arm in ARMS:
            if (seed, arm) not in run_paths:
                raise SummaryError(f"missing run for seed={seed} arm={arm}")


def pair_runs(runs: list[dict[str, Any]], expected_seeds: list[int]) -> list[dict[str, Any]]:
    by_key = {(run["seed"], run["arm"]): run for run in runs if run.get("status") == "completed"}
    pairs = []
    for seed in expected_seeds:
        control = by_key.get((seed, "control"))
        cache = by_key.get((seed, "cache"))
        if control is None or cache is None:
            continue
        control_sps = float(control["sps"])
        cache_sps = float(cache["sps"])
        pairs.append(
            {
                "seed": seed,
                "control_sps": control_sps,
                "cache_sps": cache_sps,
                "cache_over_control_sps": cache_sps / control_sps,
                "control_run": control["name"],
                "cache_run": cache["name"],
            }
        )
    return pairs


def aggregate_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [float(pair["cache_over_control_sps"]) for pair in pairs]
    return {
        "n_pairs": len(pairs),
        "median_cache_over_control_sps": statistics.median(ratios) if ratios else None,
        "geomean_cache_over_control_sps": geometric_mean(ratios),
        "ratios": ratios,
        "note": "No confidence interval is reported for this small paired sample.",
    }


def parse_monitor_start(run_path: Path) -> float | None:
    if not run_path.is_file():
        return None
    data = read_json(run_path)
    started_at = data.get("started_at")
    if not started_at:
        return None
    dt = datetime.fromisoformat(str(started_at))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def monitor_samples_for_window(root: Path, windows: list[dict[str, float] | None]) -> dict[str, Any]:
    monitor_root = root.parent
    started_unix = parse_monitor_start(monitor_root / "monitor_run.json")
    samples = read_jsonl(monitor_root / "monitor.log")
    valid_windows = [window for window in windows if window is not None]
    if started_unix is None or not samples or not valid_windows:
        return {"sample_count": 0, "util_avg": None, "util_p50": None, "util_p95": None, "util_max": None, "vram_peak_mb": None}

    filtered = []
    for sample in samples:
        uptime = sample.get("monitor/uptime_s")
        if uptime is None:
            continue
        sample_time = started_unix + float(uptime)
        if any(window["start_unix"] <= sample_time <= window["end_unix"] for window in valid_windows):
            filtered.append(sample)

    utils = [float(sample["gpu/util"]) for sample in filtered if sample.get("gpu/util") is not None]
    vram = [float(sample["gpu/mem_used_mb"]) for sample in filtered if sample.get("gpu/mem_used_mb") is not None]
    return {
        "sample_count": len(filtered),
        "util_avg": statistics.fmean(utils) if utils else None,
        "util_p50": percentile(utils, 50),
        "util_p95": percentile(utils, 95),
        "util_max": max(utils) if utils else None,
        "vram_peak_mb": max(vram) if vram else None,
    }


def summarize_campaign(root: Path | str, *, expected_seeds: list[int] | None = None, allow_incomplete: bool = False) -> dict[str, Any]:
    root = Path(root)
    expected_seeds = list(DEFAULT_SEEDS if expected_seeds is None else expected_seeds)
    run_paths = discover_runs(root)
    if not allow_incomplete:
        validate_expected_runs(run_paths, expected_seeds)

    runs = []
    for key in sorted(run_paths):
        seed, _arm = key
        if seed in expected_seeds:
            runs.append(load_run(run_paths[key], allow_incomplete=allow_incomplete))

    if not allow_incomplete:
        for run in runs:
            if run.get("status") != "completed":
                raise SummaryError(f"incomplete run: {run.get('name')} status={run.get('status')}")

    pairs = pair_runs(runs, expected_seeds)
    if not allow_incomplete and len(pairs) != len(expected_seeds):
        raise SummaryError(f"expected {len(expected_seeds)} complete pairs, found {len(pairs)}")

    windows = [run.get("measured_window") for run in runs if run.get("status") == "completed"]
    return {
        "root": str(root),
        "expected_seeds": expected_seeds,
        "runs": runs,
        "pairs": pairs,
        "aggregate": aggregate_pairs(pairs),
        "monitor": {"measured_window_gpu": monitor_samples_for_window(root, windows)},
    }


def format_markdown(result: dict[str, Any]) -> str:
    lines = ["# Campaign Summary", ""]
    aggregate = result["aggregate"]
    lines.append(f"- Complete pairs: {aggregate['n_pairs']}")
    if aggregate["median_cache_over_control_sps"] is not None:
        lines.append(f"- Median cache/control SPS: {aggregate['median_cache_over_control_sps']:.4f}x")
    if aggregate["geomean_cache_over_control_sps"] is not None:
        lines.append(f"- Geomean cache/control SPS: {aggregate['geomean_cache_over_control_sps']:.4f}x")
    lines.append("- Note: no n=3 confidence interval reported.")
    lines.append("")
    lines.append("## Pairs")
    for pair in result["pairs"]:
        lines.append(
            f"- seed {pair['seed']}: control {pair['control_sps']:.2f} SPS, "
            f"cache {pair['cache_sps']:.2f} SPS, ratio {pair['cache_over_control_sps']:.4f}x"
        )
    gpu = result["monitor"]["measured_window_gpu"]
    lines.append("")
    lines.append("## GPU monitor measured window")
    lines.append(
        "- samples {sample_count}, util avg/p50/p95/max {avg}/{p50}/{p95}/{maxv}, VRAM peak {vram} MB".format(
            sample_count=gpu["sample_count"],
            avg=_fmt(gpu["util_avg"]),
            p50=_fmt(gpu["util_p50"]),
            p95=_fmt(gpu["util_p95"]),
            maxv=_fmt(gpu["util_max"]),
            vram=_fmt(gpu["vram_peak_mb"]),
        )
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _load_checkpoint_member(zip_path: Path, member: str):
    import torch

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            payload = handle.read()
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def _state_items(value: Any, prefix: str = "") -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SummaryError("torch is required for comparecheckpoints") from exc

    if torch.is_tensor(value):
        return {prefix: value}
    if isinstance(value, dict):
        items = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(_state_items(child, child_prefix))
        return items
    if isinstance(value, (list, tuple)):
        items = {}
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            items.update(_state_items(child, child_prefix))
        return items
    return {prefix: value}


def _is_tensor(value: Any) -> bool:
    import torch

    return torch.is_tensor(value)


def _same_non_tensor(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _tensor_max_abs_diff(left: Any, right: Any) -> tuple[float, bool]:
    import torch

    if tuple(left.shape) != tuple(right.shape):
        return math.inf, False
    if left.numel() == 0:
        return 0.0, True
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    try:
        if left_cpu.dtype == torch.bool or right_cpu.dtype == torch.bool:
            mismatch = left_cpu != right_cpu
            max_abs = 1.0 if bool(mismatch.any().item()) else 0.0
        else:
            diff = (left_cpu.to(torch.float64) - right_cpu.to(torch.float64)).abs()
            max_abs = float(diff.max().item())
    except RuntimeError:
        return math.inf, False
    return max_abs, math.isfinite(max_abs)


def compare_checkpoints(left: Path | str, right: Path | str, *, tolerance: float = 0.0) -> dict[str, Any]:
    import torch

    tolerance = _as_finite_float(tolerance, "tolerance")
    if tolerance < 0:
        raise SummaryError("tolerance must be nonnegative and finite")

    left = Path(left)
    right = Path(right)
    members = ["policy.pth", "policy.optimizer.pth"]
    compared = []
    exact = True
    finite = True
    global_max = 0.0
    issue_count = 0
    for member in members:
        left_items = _state_items(_load_checkpoint_member(left, member))
        right_items = _state_items(_load_checkpoint_member(right, member))
        if not left_items or not right_items:
            issue_count += 1
            exact = False
        names = sorted(set(left_items) | set(right_items))
        missing_left = [name for name in names if name not in left_items]
        missing_right = [name for name in names if name not in right_items]
        issue_count += len(missing_left) + len(missing_right)
        tensor_reports = []
        non_tensor_differences = []
        for name in names:
            if name in missing_left or name in missing_right:
                exact = False
                continue
            a = left_items[name]
            b = right_items[name]
            if not _is_tensor(a) or not _is_tensor(b):
                if _is_tensor(a) != _is_tensor(b) or not _same_non_tensor(a, b):
                    exact = False
                    issue_count += 1
                    non_tensor_differences.append(
                        {
                            "name": name,
                            "left": repr(a),
                            "right": repr(b),
                            "type_left": type(a).__name__,
                            "type_right": type(b).__name__,
                        }
                    )
                continue
            same_shape = tuple(a.shape) == tuple(b.shape)
            same_dtype = a.dtype == b.dtype
            max_abs, diff_finite = _tensor_max_abs_diff(a, b)
            finite = finite and diff_finite
            equal = bool(torch.equal(a, b)) if same_shape else False
            if math.isfinite(global_max) and math.isfinite(max_abs):
                global_max = max(global_max, max_abs)
            elif not math.isfinite(max_abs):
                global_max = max_abs
            tensor_exact = same_shape and same_dtype and equal
            if not tensor_exact:
                issue_count += 1
            exact = exact and tensor_exact
            tensor_reports.append(
                {
                    "name": name,
                    "shape_left": list(a.shape),
                    "shape_right": list(b.shape),
                    "dtype_left": str(a.dtype),
                    "dtype_right": str(b.dtype),
                    "exact": tensor_exact,
                    "max_abs_diff": max_abs,
                    "within_tolerance": math.isfinite(max_abs) and max_abs <= tolerance and same_shape and same_dtype,
                }
            )
        compared.append(
            {
                "member": member,
                "missing_left": missing_left,
                "missing_right": missing_right,
                "tensors": tensor_reports,
                "non_tensor_differences": non_tensor_differences,
            }
        )
    within_tolerance = issue_count == 0 and finite and global_max <= tolerance
    return {
        "left": str(left),
        "right": str(right),
        "members": compared,
        "exact": exact,
        "max_abs_diff": global_max,
        "finite": finite,
        "tolerance": tolerance,
        "issue_count": issue_count,
        "within_tolerance": within_tolerance,
    }


def parse_seeds(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser("summarize", help="summarize a paired campaign directory")
    summarize_parser.add_argument("root", type=Path)
    summarize_parser.add_argument("--expected-seeds", default="45,46,47")
    summarize_parser.add_argument("--allow-incomplete", action="store_true")
    summarize_parser.add_argument("--json-out", type=Path)
    summarize_parser.add_argument("--md-out", type=Path)

    compare_parser = subparsers.add_parser("comparecheckpoints", help="compare SB3 zip tensor payloads")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--tolerance", type=float, default=0.0)
    compare_parser.add_argument("--json-out", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "summarize":
            result = summarize_campaign(
                args.root,
                expected_seeds=parse_seeds(args.expected_seeds),
                allow_incomplete=args.allow_incomplete,
            )
            json_text = json.dumps(result, indent=2, sort_keys=True)
            md_text = format_markdown(result)
            if args.json_out:
                args.json_out.write_text(json_text + "\n", encoding="utf-8")
            if args.md_out:
                args.md_out.write_text(md_text, encoding="utf-8")
            print(json_text)
            print(md_text, file=sys.stderr)
            return 0
        if args.command == "comparecheckpoints":
            result = compare_checkpoints(args.left, args.right, tolerance=args.tolerance)
            json_text = json.dumps(result, indent=2, sort_keys=True)
            if args.json_out:
                args.json_out.write_text(json_text + "\n", encoding="utf-8")
            print(json_text)
            return 0 if result["within_tolerance"] else 1
    except SummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
