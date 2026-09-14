#!/usr/bin/env python3
"""Exact-byte hardlink deduplication for completed learning checkpoints.

Dry-run is the default.  ``--apply`` only replaces same-chunk ``latest.zip``
with a hardlink to byte-identical ``final.zip`` after validating a fixed dry-run
manifest and an independently verified local-backup receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable


SCHEMA_VERSION = 1
TEMP_NAME = ".latest.zip.hardlink-tmp"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HardlinkError(RuntimeError):
    """Raised when checkpoint hardlinking is unsafe or invalid."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HardlinkError(f"invalid JSON: {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_root(root: Path | str) -> Path:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise HardlinkError(f"--root must be an existing directory: {root}")
    return resolved


def require_contained(root: Path, path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HardlinkError(f"{label} escapes root: {path}") from exc
    return resolved


def require_absolute(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise HardlinkError(f"{label} must be absolute, not relative: {path}")
    return path


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise HardlinkError(f"{label} must be a 64-character lowercase hex sha256")
    return value


def scan_escaping_symlink_dirs(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink() and path.is_dir():
            require_contained(root, path, label="symlink directory")


def is_completed_chunk(result_path: Path) -> tuple[bool, str | None]:
    if result_path.is_symlink() or not result_path.is_file():
        return False, "result_not_regular"
    result = read_json(result_path)
    if not isinstance(result, dict):
        return False, "result_not_object"
    if result.get("status") != "completed":
        return False, "result_not_completed"
    if result.get("error") is not None:
        return False, "result_error"
    if result.get("stop_reason") is not None:
        return False, "result_stop_reason"
    if (result_path.parent / "stop.request").exists():
        return False, "chunk_stop_request"
    return True, None


def stat_regular_checkpoint(path: Path) -> tuple[os.stat_result | None, str | None]:
    if path.is_symlink():
        return None, "symlink_checkpoint"
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None, "missing_checkpoint"
    if not path.is_file():
        return None, "checkpoint_not_regular"
    return stat, None


def restoration_recipe() -> str:
    return (
        "Restoration: because only latest.zip is replaced by a hardlink to "
        "same-byte final.zip, restore by copying latest.zip from the independently "
        "verified local backup receipt for the affected path if a distinct inode is required."
    )


def pair_record(root: Path, result_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    chunk = require_contained(root, result_path.parent, label="chunk")
    latest = chunk / "latest.zip"
    final = chunk / "final.zip"
    latest_stat, latest_error = stat_regular_checkpoint(latest)
    final_stat, final_error = stat_regular_checkpoint(final)
    if latest_error or final_error:
        return None, {"chunk": str(chunk), "reason": latest_error or final_error}
    assert latest_stat is not None and final_stat is not None
    if latest_stat.st_ino == final_stat.st_ino and latest_stat.st_dev == final_stat.st_dev:
        return None, {"chunk": str(chunk), "reason": "already_same_inode"}
    if latest_stat.st_size != final_stat.st_size:
        return None, {"chunk": str(chunk), "reason": "size_mismatch"}
    latest_sha = sha256_file(latest)
    final_sha = sha256_file(final)
    if latest_sha != final_sha:
        return None, {"chunk": str(chunk), "reason": "content_mismatch"}
    return (
        {
            "result_path": str(result_path),
            "result_sha256": sha256_file(result_path),
            "chunk_dir": str(chunk),
            "latest_path": str(latest),
            "final_path": str(final),
            "size": int(latest_stat.st_size),
            "latest_sha256": latest_sha,
            "final_sha256": final_sha,
            "latest_inode_before": int(latest_stat.st_ino),
            "final_inode_before": int(final_stat.st_ino),
            "device_before": int(latest_stat.st_dev),
            "action": "hardlink_latest_to_final",
        },
        None,
    )


def plan_dedup(root: Path | str) -> dict[str, Any]:
    root = resolve_root(root)
    scan_escaping_symlink_dirs(root)
    pairs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for result_path in sorted(root.rglob("result.json")):
        result_path = require_contained(root, result_path, label="result.json")
        completed, reason = is_completed_chunk(result_path)
        if not completed:
            skipped.append({"chunk": str(result_path.parent), "reason": reason})
            continue
        pair, skip = pair_record(root, result_path)
        if pair is not None:
            pairs.append(pair)
        elif skip is not None:
            skipped.append(skip)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run",
        "root": str(root),
        "generated_unix": time.time(),
        "candidate_count": len(pairs),
        "recoverable_bytes": sum(int(pair["size"]) for pair in pairs),
        "pairs": pairs,
        "skipped": skipped,
        "restoration_recipe": restoration_recipe(),
    }


def load_manifest(manifest_path: Path | str | None) -> dict[str, Any]:
    if manifest_path is None:
        raise HardlinkError("--apply requires a fixed dry-run manifest")
    manifest = read_json(Path(manifest_path))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise HardlinkError("manifest schema/version is invalid")
    if manifest.get("status") != "dry_run":
        raise HardlinkError("manifest status must be dry_run")
    if not isinstance(manifest.get("pairs"), list):
        raise HardlinkError("manifest pairs are missing")
    return manifest


def backup_receipt_entries(
    receipt_path: Path | str | None,
    *,
    manifest_sha256: str,
) -> tuple[dict[str, tuple[str, int, str]], Path, str, int]:
    if receipt_path is None:
        raise HardlinkError("--apply requires an independently verified local-backup receipt")
    receipt_file = Path(receipt_path).resolve()
    receipt_sha256 = sha256_file(receipt_file)
    receipt = read_json(receipt_file)
    if not isinstance(receipt, dict):
        raise HardlinkError("backup receipt must be a JSON object")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise HardlinkError("backup receipt schema_version must be 1")
    if receipt.get("status") != "verified":
        raise HardlinkError("backup receipt status must be verified")
    if receipt.get("independently_verified_local_backup") is not True:
        raise HardlinkError("backup receipt is not marked independently verified")
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise HardlinkError("backup receipt manifest_sha256 does not match input manifest")
    verified_unix = receipt.get("verified_unix")
    if not isinstance(verified_unix, (int, float)) or not math.isfinite(float(verified_unix)):
        raise HardlinkError("backup receipt verified_unix must be finite")
    entries = receipt.get("files")
    if not isinstance(entries, list):
        raise HardlinkError("backup receipt files are missing")
    covered: dict[str, tuple[str, int, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise HardlinkError("backup receipt file entries must be objects")
        try:
            remote_path = Path(entry["remote_path"])
            backup_path = Path(entry["backup_path"])
            entry_sha256 = require_sha256(entry["sha256"], label="backup receipt file sha256")
            size = int(entry["size"])
        except KeyError as exc:
            raise HardlinkError(f"backup receipt file entry missing {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise HardlinkError("backup receipt file entry has malformed fields") from exc
        require_absolute(remote_path, label="backup receipt remote_path")
        require_absolute(backup_path, label="backup receipt backup_path")
        if size < 0:
            raise HardlinkError("backup receipt file size must be nonnegative")
        remote_key = str(remote_path)
        if remote_key in covered:
            raise HardlinkError(f"duplicate backup receipt entry for {remote_key}")
        covered[remote_key] = (entry_sha256, size, str(backup_path))
    return covered, receipt_file, receipt_sha256, SCHEMA_VERSION


def validate_manifest_pair(root: Path, pair: dict[str, Any], receipt_entries: dict[str, tuple[str, int, str]]) -> dict[str, Any]:
    required = ("result_path", "result_sha256", "latest_path", "final_path", "size", "latest_sha256", "final_sha256")
    missing = [key for key in required if key not in pair]
    if missing:
        raise HardlinkError(f"manifest pair missing keys: {missing}")
    if pair.get("action") != "hardlink_latest_to_final":
        raise HardlinkError("manifest action is not hardlink_latest_to_final")
    result_input = require_absolute(Path(pair["result_path"]), label="manifest result_path")
    latest_input = require_absolute(Path(pair["latest_path"]), label="manifest latest_path")
    final_input = require_absolute(Path(pair["final_path"]), label="manifest final_path")
    if result_input.name != "result.json":
        raise HardlinkError("manifest result_path must name result.json")
    result_parent = require_contained(root, result_input.parent, label="manifest result_path parent")
    if latest_input.name != "latest.zip" or final_input.name != "final.zip":
        raise HardlinkError("manifest pair must target same-dir latest.zip and final.zip")
    latest_parent = require_contained(root, latest_input.parent, label="manifest latest_path parent")
    final_parent = require_contained(root, final_input.parent, label="manifest final_path parent")
    if result_parent != latest_parent or result_parent != final_parent:
        raise HardlinkError("manifest result.json must be in same chunk as latest.zip and final.zip")
    result_path = result_parent / "result.json"
    latest = latest_parent / "latest.zip"
    final = final_parent / "final.zip"
    if latest.parent != final.parent:
        raise HardlinkError("manifest pair must target same-dir latest.zip and final.zip")
    completed, reason = is_completed_chunk(result_path)
    if not completed:
        raise HardlinkError(f"completion drift for {result_path}: {reason}")
    planned_result_sha = require_sha256(pair["result_sha256"], label="manifest result_sha256")
    current_result_sha = sha256_file(result_path)
    if current_result_sha != planned_result_sha:
        raise HardlinkError(f"result.json drift for {result_path}: sha256 changed")
    temp = latest.parent / TEMP_NAME
    if temp.exists() or temp.is_symlink():
        raise HardlinkError(f"temporary hardlink path already exists: {temp}")
    latest_stat, latest_error = stat_regular_checkpoint(latest)
    final_stat, final_error = stat_regular_checkpoint(final)
    if latest_error or final_error or latest_stat is None or final_stat is None:
        raise HardlinkError(f"input drift for {latest.parent}: {latest_error or final_error}")
    expected_size = int(pair["size"])
    if latest_stat.st_size != expected_size or final_stat.st_size != expected_size:
        raise HardlinkError(f"input drift for {latest.parent}: size changed")
    latest_sha = sha256_file(latest)
    final_sha = sha256_file(final)
    if latest_sha != pair["latest_sha256"] or final_sha != pair["final_sha256"] or latest_sha != final_sha:
        raise HardlinkError(f"input drift for {latest.parent}: sha256 changed")
    for path, expected_sha in ((latest, str(pair["latest_sha256"])), (final, str(pair["final_sha256"]))):
        receipt_entry = receipt_entries.get(str(path))
        if receipt_entry is None or receipt_entry[:2] != (expected_sha, expected_size):
            raise HardlinkError(f"backup receipt does not cover both paths for {latest.parent}")
    return {
        "result_path": result_path,
        "latest": latest,
        "final": final,
        "size": expected_size,
        "sha256": latest_sha,
        "latest_inode_before": int(latest_stat.st_ino),
        "final_inode_before": int(final_stat.st_ino),
        "already_same_inode": latest_stat.st_ino == final_stat.st_ino and latest_stat.st_dev == final_stat.st_dev,
    }


def apply_dedup(
    root: Path | str,
    manifest_path: Path | str | None,
    backup_receipt_path: Path | str | None,
    *,
    replace: Callable[[Path, Path], None] = os.replace,
) -> dict[str, Any]:
    root = resolve_root(root)
    manifest_file = Path(manifest_path).resolve() if manifest_path is not None else None
    manifest_sha256 = sha256_file(manifest_file) if manifest_file is not None else ""
    manifest = load_manifest(manifest_path)
    if Path(str(manifest.get("root"))).resolve() != root:
        raise HardlinkError("manifest root does not match --root")
    receipt_entries, receipt_file, receipt_sha256, receipt_schema_version = backup_receipt_entries(
        backup_receipt_path,
        manifest_sha256=manifest_sha256,
    )
    pairs = [validate_manifest_pair(root, pair, receipt_entries) for pair in manifest["pairs"]]

    applied: list[dict[str, Any]] = []
    for pair in pairs:
        if pair["already_same_inode"]:
            continue
        latest = pair["latest"]
        final = pair["final"]
        temp = latest.parent / TEMP_NAME
        try:
            os.link(final, temp)
            replace(temp, latest)
        except OSError as exc:
            try:
                if temp.exists() or temp.is_symlink():
                    temp.unlink()
            finally:
                raise HardlinkError(f"atomic hardlink replace failed for {latest}: {exc}. {restoration_recipe()}") from exc
        latest_stat = latest.stat()
        final_stat = final.stat()
        latest_sha = sha256_file(latest)
        final_sha = sha256_file(final)
        if latest_sha != pair["sha256"] or final_sha != pair["sha256"]:
            raise HardlinkError(f"post-apply sha mismatch for {latest}. {restoration_recipe()}")
        if latest_stat.st_ino != final_stat.st_ino or latest_stat.st_dev != final_stat.st_dev:
            raise HardlinkError(f"post-apply inode mismatch for {latest}. {restoration_recipe()}")
        applied.append(
            {
                "latest_path": str(latest),
                "final_path": str(final),
                "size": pair["size"],
                "sha256": pair["sha256"],
                "latest_inode_before": pair["latest_inode_before"],
                "final_inode_before": pair["final_inode_before"],
                "latest_inode_after": int(latest_stat.st_ino),
                "final_inode_after": int(final_stat.st_ino),
                "same_inode_after": latest_stat.st_ino == final_stat.st_ino and latest_stat.st_dev == final_stat.st_dev,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "applied",
        "root": str(root),
        "manifest_path": str(manifest_file),
        "manifest_sha256": manifest_sha256,
        "backup_receipt_path": str(receipt_file),
        "backup_receipt_sha256": receipt_sha256,
        "backup_receipt_schema_version": receipt_schema_version,
        "applied_count": len(applied),
        "recovered_bytes": sum(int(item["size"]) for item in applied),
        "applied": applied,
        "restoration_recipe": restoration_recipe(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write JSON report/manifest here")
    parser.add_argument("--apply", action="store_true", help="perform guarded hardlink replacement; default is dry-run")
    parser.add_argument("--manifest", type=Path, help="fixed dry-run manifest required with --apply")
    parser.add_argument("--backup-receipt", type=Path, help="independently verified local-backup receipt required with --apply")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.apply:
            report = apply_dedup(args.root, args.manifest, args.backup_receipt)
        else:
            report = plan_dedup(args.root)
    except HardlinkError as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
