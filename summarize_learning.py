#!/usr/bin/env python3
"""Summarize E6 learning campaign artifacts without importing training deps.

The summarizer is intentionally read-only and stdlib-only. It consumes either a
campaign artifact root containing ``smoke_results.json`` or
``artifacts/campaign_learning_results.json``, or a direct results JSON file, and
fails closed on incomplete/invalid metrics unless ``--partial`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_EVAL_KEYS = ("0a-deterministic", "0a-sampled", "0b-deterministic", "0b-sampled")
HISTORY_HYDRATED_STATUSES = {"skipped_completed", "skipped_existing"}
EXPLAINED_SKIP_STATUSES = {"skipped_not_candidate", "skipped_after_mastery"}


class SummaryError(RuntimeError):
    """Raised when campaign artifacts are incomplete or internally inconsistent."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SummaryError(f"invalid JSON: {path}: {exc}") from exc


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def finite_number(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SummaryError(f"non-finite {label}: {value!r}")
    return float(value)


def finite_nonnegative(value: Any, *, label: str) -> float:
    number = finite_number(value, label=label)
    if number < 0:
        raise SummaryError(f"negative {label}: {number}")
    return number


def parse_label(label: str) -> tuple[str, int | None, str]:
    if "_chunk" not in label:
        raise SummaryError(f"unrecognized result label: {label}")
    run_label, tail = label.rsplit("_chunk", 1)
    digits = ""
    for char in tail:
        if char.isdigit():
            digits += char
        else:
            break
    if not digits:
        raise SummaryError(f"missing chunk index in label: {label}")
    suffix = tail[len(digits):].lstrip("_")
    return run_label, int(digits), suffix


def process_wall(result: dict[str, Any]) -> float:
    started = result.get("started_unix")
    ended = result.get("ended_unix")
    if started is None or ended is None:
        return 0.0
    wall = finite_number(ended, label=f"{result.get('label')} ended_unix") - finite_number(
        started, label=f"{result.get('label')} started_unix"
    )
    if wall < 0:
        raise SummaryError(f"negative process wall for {result.get('label')}: {wall}")
    return wall


def require_completed(result: dict[str, Any]) -> None:
    if result.get("status") != "completed":
        raise SummaryError(f"result is not completed: {result.get('label')} status={result.get('status')}")


def validate_checkpoint_record(record: dict[str, Any], *, label: str, key: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise SummaryError(f"missing checkpoint evidence {key} for {label}")
    if record.get("exists") is not True:
        raise SummaryError(f"checkpoint evidence not marked existing {key} for {label}")
    path = record.get("path")
    sha = record.get("sha256")
    size = record.get("size")
    if not isinstance(path, str) or not path:
        raise SummaryError(f"checkpoint evidence missing path {key} for {label}")
    if not isinstance(sha, str) or not sha:
        raise SummaryError(f"checkpoint evidence missing sha256 {key} for {label}")
    if not isinstance(size, int) or size < 0:
        raise SummaryError(f"checkpoint evidence invalid size {key} for {label}")
    return {"path": path, "sha256": sha, "size": size, "exists": True}


def summarize_train(result: dict[str, Any]) -> dict[str, Any]:
    require_completed(result)
    label = str(result.get("label", ""))
    run_label, chunk, suffix = parse_label(label)
    if suffix != "train":
        raise SummaryError(f"train result has unexpected label suffix: {label}")
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        raise SummaryError(f"missing train evidence for {label}")
    start = int(finite_nonnegative(evidence.get("starting_steps"), label=f"{label} starting_steps"))
    end = int(finite_nonnegative(evidence.get("ending_steps"), label=f"{label} ending_steps"))
    actual = int(finite_nonnegative(evidence.get("actual_steps"), label=f"{label} actual_steps"))
    if end - start != actual:
        raise SummaryError(f"train step mismatch for {label}: {start}->{end}, actual={actual}")
    learner_wall = finite_nonnegative(evidence.get("wall_seconds"), label=f"{label} learner wall_seconds")
    parameter_count = int(finite_nonnegative(evidence.get("parameter_count"), label=f"{label} parameter_count"))
    checkpoint_evidence = evidence.get("checkpoint_evidence")
    if not isinstance(checkpoint_evidence, dict):
        raise SummaryError(f"missing checkpoint evidence for {label}")
    checkpoints = {
        key: validate_checkpoint_record(checkpoint_evidence.get(key), label=label, key=key)
        for key in ("initial_checkpoint", "latest_checkpoint", "final_checkpoint")
    }
    return {
        "label": label,
        "run_label": run_label,
        "chunk": chunk,
        "starting_steps": start,
        "ending_steps": end,
        "actual_steps": actual,
        "process_wall_seconds": process_wall(result),
        "learner_wall_seconds": learner_wall,
        "parameter_count": parameter_count,
        "input_checkpoint": evidence.get("input_checkpoint"),
        "input_checkpoint_sha256": evidence.get("input_checkpoint_sha256"),
        "initial_checkpoint": checkpoints["initial_checkpoint"],
        "latest_checkpoint": checkpoints["latest_checkpoint"],
        "final_checkpoint": checkpoints["final_checkpoint"],
    }


def summarize_eval(result: dict[str, Any]) -> dict[str, Any]:
    require_completed(result)
    label = str(result.get("label", ""))
    run_label, chunk, suffix = parse_label(label)
    if not suffix.startswith("eval_"):
        raise SummaryError(f"eval result has unexpected label suffix: {label}")
    eval_kind = suffix.removeprefix("eval_")
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        raise SummaryError(f"missing eval evidence for {label}")
    summaries = evidence.get("summaries")
    if not isinstance(summaries, dict):
        raise SummaryError(f"missing eval summaries for {label}")
    summarized = {}
    for key in REQUIRED_EVAL_KEYS:
        summary = summaries.get(key)
        if not isinstance(summary, dict):
            raise SummaryError(f"missing eval summary {key} for {label}")
        episodes = int(finite_nonnegative(summary.get("episodes"), label=f"{label} {key} episodes"))
        if episodes <= 0:
            raise SummaryError(f"empty eval summary {key} for {label}")
        success_rate = finite_number(summary.get("success_rate"), label=f"{label} {key} success_rate")
        fall_rate = finite_number(summary.get("fall_rate"), label=f"{label} {key} fall_rate")
        timeout_rate = finite_number(summary.get("timeout_rate", 0.0), label=f"{label} {key} timeout_rate")
        for rate_name, rate in (("success_rate", success_rate), ("fall_rate", fall_rate), ("timeout_rate", timeout_rate)):
            if rate < 0.0 or rate > 1.0:
                raise SummaryError(f"{label} {key} {rate_name} out of range: {rate}")
        summarized[key] = {
            "episodes": episodes,
            "success_rate": success_rate,
            "fall_rate": fall_rate,
            "timeout_rate": timeout_rate,
            "nav_falls": int(finite_nonnegative(summary.get("nav_falls"), label=f"{label} {key} nav_falls")),
            "timeouts": int(finite_nonnegative(summary.get("timeouts", round(timeout_rate * episodes)), label=f"{label} {key} timeouts")),
            "avg_steps": finite_nonnegative(summary.get("avg_steps", 0.0), label=f"{label} {key} avg_steps"),
        }
    return {
        "label": label,
        "run_label": run_label,
        "chunk": chunk,
        "eval_kind": eval_kind,
        "process_wall_seconds": process_wall(result),
        "summaries": summarized,
    }


def validate_lineage(trains: list[dict[str, Any]]) -> None:
    trains.sort(key=lambda item: item["chunk"])
    previous = None
    for item in trains:
        if previous is None:
            if item["starting_steps"] != 0:
                raise SummaryError(f"first train chunk did not start at zero for {item['run_label']}")
            if item["input_checkpoint"] is not None or item["input_checkpoint_sha256"] is not None:
                raise SummaryError(f"fresh train unexpectedly has input checkpoint for {item['label']}")
        else:
            if item["starting_steps"] != previous["ending_steps"]:
                raise SummaryError(f"lineage step mismatch for {item['label']}")
            if item["input_checkpoint"] != previous["final_checkpoint"]["path"]:
                raise SummaryError(f"lineage checkpoint path mismatch for {item['label']}")
            if item["input_checkpoint_sha256"] != previous["final_checkpoint"]["sha256"]:
                raise SummaryError(f"lineage sha mismatch for {item['label']}")
        previous = item


def expected_labels_from_plan(plan: dict[str, Any] | None) -> set[str]:
    if not isinstance(plan, dict):
        return set()
    commands = plan.get("commands")
    if not isinstance(commands, list):
        return set()
    labels = set()
    for item in commands:
        if isinstance(item, dict) and isinstance(item.get("label"), str):
            labels.add(item["label"])
    return labels


def run_label_from_any_label(label: str) -> str:
    if "_chunk" not in label:
        return label
    return label.rsplit("_chunk", 1)[0]


def normalize_results(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"], payload.get("status")
    raise SummaryError("results JSON must be a list or object with results list")


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))


def extract_completed_history(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and payload.get("status") == "completed" and isinstance(payload.get("label"), str):
        return [payload]
    results, _status = normalize_results(payload)
    completed = []
    for result in results:
        if isinstance(result, dict) and result.get("status") == "completed" and isinstance(result.get("label"), str):
            completed.append(result)
    return completed


def find_execution_history_files(root_or_file: Path, results_path: Path) -> list[Path]:
    path = Path(root_or_file)
    root = path.parent if path.is_file() else path
    candidates = [
        root / "artifacts" / "execution_history",
        root / "execution_history",
        results_path.parent / "execution_history",
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for directory in candidates:
        if not directory.is_dir():
            continue
        for file_path in sorted(directory.glob("*.json")):
            resolved = file_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(file_path)
    return files


def load_execution_history(files: list[Path]) -> dict[str, dict[str, Any]]:
    history: dict[str, dict[str, Any]] = {}
    for file_path in files:
        for result in extract_completed_history(read_json(file_path)):
            label = str(result["label"])
            existing = history.get(label)
            if existing is not None and canonical_json(existing) != canonical_json(result):
                raise SummaryError(f"conflicting execution history for {label}: {file_path}")
            history[label] = result
    return history


def evidence_conflicts_with_history(current: Any, historical: Any, *, label: str) -> bool:
    if not isinstance(current, dict) or not isinstance(historical, dict):
        raise SummaryError(f"missing execution history evidence for {label}")
    for key, value in current.items():
        if key not in historical or canonical_json(historical[key]) != canonical_json(value):
            return True
    return False


def hydrate_skipped_results(results: list[dict[str, Any]], history: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    hydrated = []
    hydrated_labels = []
    for result in results:
        if not isinstance(result, dict):
            hydrated.append(result)
            continue
        status = result.get("status")
        if status not in HISTORY_HYDRATED_STATUSES:
            hydrated.append(result)
            continue
        label = str(result.get("label", ""))
        historical = history.get(label)
        if historical is None:
            raise SummaryError(f"missing execution history for skipped result: {label}")
        if historical.get("kind") != result.get("kind"):
            raise SummaryError(f"execution history kind mismatch for {label}")
        if evidence_conflicts_with_history(result.get("evidence"), historical.get("evidence"), label=label):
            raise SummaryError(f"skipped result evidence conflicts with execution history for {label}")
        hydrated.append(historical)
        hydrated_labels.append(label)
    return hydrated, hydrated_labels


def summarize_results(results: list[dict[str, Any]], *, allow_partial: bool, expected_labels: set[str] | None = None, campaign_status: str | None = None) -> dict[str, Any]:
    expected_labels = set(expected_labels or ())
    completed_labels = {str(item.get("label")) for item in results if isinstance(item, dict) and item.get("status") == "completed"}
    all_seen_labels = {str(item.get("label")) for item in results if isinstance(item, dict) and item.get("label") is not None}
    pending_labels = sorted(expected_labels - all_seen_labels)
    explained_skips = [
        {"label": item.get("label"), "kind": item.get("kind"), "status": item.get("status"), "dev_verdict": item.get("dev_verdict"), "mastery_verdict": item.get("mastery_verdict")}
        for item in results
        if isinstance(item, dict) and item.get("status") in EXPLAINED_SKIP_STATUSES
    ]
    noncompleted = [
        item for item in results
        if isinstance(item, dict)
        and item.get("status") not in {"completed", "skipped_not_candidate", "skipped_after_mastery", "skipped_completed", "skipped_existing"}
    ]
    if (pending_labels or noncompleted or (campaign_status not in (None, "completed") and campaign_status != "running")) and not allow_partial:
        raise SummaryError("pending commands or non-completed results present; rerun with --partial for an explicit partial report")

    arms: dict[str, dict[str, Any]] = {}
    totals = {
        "actual_steps": 0,
        "train_process_wall_seconds": 0.0,
        "train_learning_wall_seconds": 0.0,
        "eval_process_wall_seconds": 0.0,
        "train_jobs": 0,
        "eval_jobs": 0,
    }

    for result in results:
        if not isinstance(result, dict):
            raise SummaryError("result entry is not an object")
        if result.get("status") != "completed":
            continue
        kind = result.get("kind")
        if kind == "train":
            row = summarize_train(result)
            arm = arms.setdefault(
                row["run_label"],
                {"trains": [], "evals": {}, "parameter_count": None, "train_process_wall_seconds": 0.0, "train_learning_wall_seconds": 0.0, "eval_process_wall_seconds": 0.0},
            )
            arm["trains"].append(row)
            arm["train_process_wall_seconds"] += row["process_wall_seconds"]
            arm["train_learning_wall_seconds"] += row["learner_wall_seconds"]
            totals["actual_steps"] += row["actual_steps"]
            totals["train_process_wall_seconds"] += row["process_wall_seconds"]
            totals["train_learning_wall_seconds"] += row["learner_wall_seconds"]
            totals["train_jobs"] += 1
            if arm["parameter_count"] is None:
                arm["parameter_count"] = row["parameter_count"]
            elif arm["parameter_count"] != row["parameter_count"]:
                raise SummaryError(f"parameter count changed for {row['run_label']}")
        elif kind == "eval":
            row = summarize_eval(result)
            arm = arms.setdefault(
                row["run_label"],
                {"trains": [], "evals": {}, "parameter_count": None, "train_process_wall_seconds": 0.0, "train_learning_wall_seconds": 0.0, "eval_process_wall_seconds": 0.0},
            )
            arm["evals"][f"chunk{row['chunk']:02d}_eval_{row['eval_kind']}"] = row["summaries"]
            arm["eval_process_wall_seconds"] += row["process_wall_seconds"]
            totals["eval_process_wall_seconds"] += row["process_wall_seconds"]
            totals["eval_jobs"] += 1
        else:
            raise SummaryError(f"unknown completed result kind: {kind!r}")

    for arm_label, arm in arms.items():
        validate_lineage(arm["trains"])
        latest = arm["trains"][-1] if arm["trains"] else None
        arm["lineage"] = [
            {
                "chunk": item["chunk"],
                "starting_steps": item["starting_steps"],
                "ending_steps": item["ending_steps"],
                "actual_steps": item["actual_steps"],
                "input_checkpoint": item["input_checkpoint"],
                "input_checkpoint_sha256": item["input_checkpoint_sha256"],
                "final_checkpoint": item["final_checkpoint"],
            }
            for item in arm["trains"]
        ]
        arm["cumulative_steps"] = latest["ending_steps"] if latest else 0
        arm["latest_checkpoint"] = latest["final_checkpoint"] if latest else None
        arm.pop("trains", None)

    partial = bool(pending_labels or noncompleted or (campaign_status not in (None, "completed")))
    return {
        "status": "partial" if partial else "completed",
        "totals": totals,
        "arms": arms,
        "pending": {
            "count": len(pending_labels),
            "labels": pending_labels[:100],
            "arms": sorted({run_label_from_any_label(label) for label in pending_labels}),
            "truncated": len(pending_labels) > 100,
            "completed_labels": len(completed_labels),
        },
        "noncompleted": [
            {"label": item.get("label"), "kind": item.get("kind"), "status": item.get("status"), "error": item.get("error")}
            for item in noncompleted
        ],
        "skips": explained_skips,
    }


def find_results_file(root_or_file: Path) -> Path:
    path = Path(root_or_file)
    if path.is_file():
        return path
    candidates = [
        path / "artifacts" / "campaign_learning_results.json",
        path / "smoke_results.json",
        path / "campaign_learning_results.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SummaryError(f"could not find campaign results JSON under {path}")


def find_plan(root_or_file: Path) -> tuple[dict[str, Any] | None, Path | None]:
    path = Path(root_or_file)
    root = path.parent if path.is_file() else path
    for candidate in (root / "artifacts" / "campaign_learning_plan.json", root / "campaign_learning_plan.json"):
        if candidate.is_file():
            return read_json(candidate), candidate
    return None, None


def summarize_campaign(root_or_file: Path | str, *, allow_partial: bool = False) -> dict[str, Any]:
    root_or_file = Path(root_or_file)
    results_path = find_results_file(root_or_file)
    results, campaign_status = normalize_results(read_json(results_path))
    history_files = find_execution_history_files(root_or_file, results_path)
    history = load_execution_history(history_files)
    results, hydrated_labels = hydrate_skipped_results(results, history)
    plan, plan_path = find_plan(root_or_file)
    summary = summarize_results(
        results,
        allow_partial=allow_partial,
        expected_labels=expected_labels_from_plan(plan),
        campaign_status=campaign_status,
    )
    summary["source"] = {"results": str(results_path), "plan": str(plan_path) if plan_path else None}
    summary["history"] = {
        "files": [str(path) for path in history_files],
        "completed_receipts": len(history),
        "hydrated_labels": hydrated_labels,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_or_results", type=Path, help="campaign root or results JSON")
    parser.add_argument("--partial", action="store_true", help="allow pending/incomplete campaign entries and mark report partial")
    parser.add_argument("--output", type=Path, help="write JSON summary here instead of stdout only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize_campaign(args.root_or_results, allow_partial=args.partial)
    except SummaryError as exc:
        print(f"ERROR: {exc}")
        return 1
    text = json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
