#!/usr/bin/env python3
"""Local block preservation and sanitized read-only Vast.ai preconditions.

No create/start/stop/destroy operations exist here. ``audit`` is an explicit
read-only provider request; ``verify-local`` never accesses the network.
Inventories describe immutable, quiescent downloaded blocks, not live logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import time

INSTANCE_ID = 50149729
INSTANCE_LABEL = "ai2-native-ppo-session10-20260907"
MAX_HOURLY_PRICE = 0.40
CAMPAIGN_START_CREDIT = 12.315037
CAMPAIGN_CEILING = 12.0
AUDIT_MAX_AGE_SECONDS = 600.0
RETRIEVAL_RESERVE_SECONDS = 300.0
LOCAL_OUTPUTS = {"preservation_receipt.json", "preservation_ack.json"}
RESERVED_FILES = LOCAL_OUTPUTS | {"inventory.json"}

# This installed CLI version sends an obsolete owner=me query for show__user.
# Keep its credential loading and response handler, but route only this read
# through its authorization-header helper without any query-string credential.
# A child process contains CLI output/exceptions and is bounded by run(timeout).
USER_CREDIT_SCRIPT = '''import sys
import vast

original_get = vast.http_get
endpoint = "https://console.vast.ai/api/v0/users/current"

def credit_url(args, subpath, query_args=None):
    if args.url != "https://console.vast.ai" or subpath != "/users/current":
        raise RuntimeError("unsupported credit endpoint")
    return endpoint

def credit_get(args, req_url, headers=None, json=None):
    if req_url != endpoint:
        raise RuntimeError("unsupported credit request")
    authorization = vast.apiheaders(args)
    if not authorization.get("Authorization"):
        raise RuntimeError("provider authentication unavailable")
    return original_get(args, req_url, headers=authorization, json=json)

vast.apiurl = credit_url
vast.http_get = credit_get
sys.argv = ["vastai", "show", "user", "--raw"]
vast.main()
'''


class SafetyError(RuntimeError):
    """Fixed, non-sensitive rejection; never interpolate provider payloads."""


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SafetyError("invalid numeric safety field")
    try:
        value = float(value)
    except OverflowError:
        raise SafetyError("invalid numeric safety field") from None
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise SafetyError("invalid numeric safety field")
    return float(value)


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise SafetyError("unsafe inventory path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or part.endswith(".tmp") or ".tmp." in part for part in parts):
        raise SafetyError("unsafe or temporary inventory path")
    if any(ord(char) < 32 or ord(char) == 127 for char in value) or PurePosixPath(value).is_absolute():
        raise SafetyError("unsafe inventory path")
    return parts


def _root(block_dir):
    path = Path(block_dir).absolute()
    if path.is_symlink() or path.resolve() != path or not path.is_dir():
        raise SafetyError("block directory must be a real nonsymlink directory")
    return path


def _read_regular(root, relative, *, collect=False):
    """Open every component without following links; reject special files."""
    parts = _relative_path(relative)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise SafetyError("inventory entry must be a regular file")
            digest, size, chunks = hashlib.sha256(), 0, []
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                if collect:
                    chunks.append(chunk)
            after = os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise SafetyError("inventory file changed during verification")
            return size, digest.hexdigest(), b"".join(chunks)
    finally:
        os.close(directory)


def verify_inventory(block_dir):
    """Return a receipt and hash-bound ACK without writing any files.

Every regular file must be inventoried, except inventory.json and this helper's
two generated receipts. Symlinks, temporary files and incomplete copies fail.
"""
    try:
        root = _root(block_dir)
        _, inventory_hash, raw = _read_regular(root, "inventory.json", collect=True)
        inventory = json.loads(raw)
        if not isinstance(inventory, dict) or inventory.get("schema_version") != "curriculum-inventory-v1":
            raise SafetyError("unsupported inventory schema")
        block_id = inventory.get("block_id", root.name)
        if not isinstance(block_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", block_id):
            raise SafetyError("invalid block identity")
        entries = inventory.get("files")
        if not isinstance(entries, list) or not entries:
            raise SafetyError("inventory must contain files")
        expected, total = set(), 0
        for entry in entries:
            if not isinstance(entry, dict):
                raise SafetyError("invalid inventory entry")
            name = entry.get("path")
            _relative_path(name)
            if name in expected or name in RESERVED_FILES:
                raise SafetyError("duplicate or reserved inventory entry")
            expected.add(name)
            length, digest = entry.get("bytes"), entry.get("sha256")
            if type(length) is not int or length < 0 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise SafetyError("invalid inventory size or digest")
            actual_size, actual_hash, _ = _read_regular(root, name)
            if (actual_size, actual_hash) != (length, digest):
                raise SafetyError("inventory size or checksum mismatch")
            total += length
        actual = set()
        for directory, directories, files in os.walk(root, followlinks=False):
            for name in directories + files:
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                _relative_path(relative)
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise SafetyError("symlink or special file in block")
                if stat.S_ISREG(mode) and relative not in RESERVED_FILES:
                    actual.add(relative)
        if actual != expected:
            raise SafetyError("inventory does not cover the complete local block")
        if _read_regular(root, "inventory.json")[1] != inventory_hash:
            raise SafetyError("inventory changed during verification")
        ack = {"block_id": block_id, "inventory_sha256": inventory_hash, "verified": True}
        return {"schema_version": "curriculum-preservation-v1", **ack,
                "files_verified": len(expected), "bytes_verified": total, "ack": ack}
    except (OSError, ValueError, TypeError):
        raise SafetyError("local block verification failed") from None


def _write_immutable(root, name, payload):
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    destination = root / name
    if destination.exists() or destination.is_symlink():
        if _read_regular(root, name, collect=True)[2] != encoded:
            raise SafetyError("refusing to overwrite preservation evidence")
        return
    fd, temporary = tempfile.mkstemp(prefix=".preservation-", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)  # Atomic publish, never overwrite.
        except FileExistsError:
            if _read_regular(root, name, collect=True)[2] != encoded:
                raise SafetyError("refusing to overwrite preservation evidence") from None
    finally:
        os.unlink(temporary)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def preserve_block(block_dir):
    """Verify first, then atomically publish immutable local receipt and ACK."""
    receipt = verify_inventory(block_dir)
    try:
        root = _root(block_dir)
        _write_immutable(root, "preservation_receipt.json", receipt)
        _write_immutable(root, "preservation_ack.json", receipt["ack"])
    except OSError:
        raise SafetyError("preservation receipt could not be published") from None
    return receipt


def _provider_json(kind, runner):
    # Deliberately no arbitrary command argument or lifecycle operation.
    if kind not in {"instances", "user"}:
        raise SafetyError("unsupported provider read")
    try:
        command = (["vastai", "show", "instances", "--raw"] if kind == "instances"
                   else [sys.executable, "-c", USER_CREDIT_SCRIPT])
        result = runner(command, capture_output=True,
                        text=True, check=False, timeout=30)
        if result.returncode != 0:
            raise SafetyError("provider request failed")
        return json.loads(result.stdout)
    except Exception:
        # Provider exceptions can contain API-key query strings in cmd/stderr.
        # Suppress exception chaining as well as the raw captured streams.
        raise SafetyError("provider request failed") from None


def _status(row):
    values = [row[key] for key in ("actual_status", "cur_state", "state", "intended_status", "next_state") if key in row]
    states = {"running": "running", "stopped": "stopped", "exited": "stopped"}
    if not values or any(not isinstance(value, str) or value not in states for value in values):
        raise SafetyError("provider instance status is not allowlisted")
    normalized = {states[value] for value in values}
    if len(normalized) != 1:
        raise SafetyError("provider instance states disagree")
    return normalized.pop()


def audit_provider(*, now=None, runner=None):
    """Capture two read-only CLI calls; return ONLY validated safe fields."""
    timestamp = _number(time.time() if now is None else now)
    runner = subprocess.run if runner is None else runner
    instances = _provider_json("instances", runner)
    user = _provider_json("user", runner)
    if not isinstance(instances, list) or not isinstance(user, dict):
        raise SafetyError("unsupported provider response schema")
    ids, selected, running = set(), None, 0
    for row in instances:
        if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] <= 0 or row["id"] in ids:
            raise SafetyError("invalid or duplicate provider instance identity")
        ids.add(row["id"])
        status = _status(row)
        running += status == "running"
        if row["id"] != INSTANCE_ID and status != "stopped":
            raise SafetyError("another instance is active")
        if row["id"] == INSTANCE_ID:
            if row.get("label") != INSTANCE_LABEL:
                raise SafetyError("retained instance label mismatch")
            price = _number(row.get("dph_total"))
            if price > MAX_HOURLY_PRICE:
                raise SafetyError("retained instance price exceeds ceiling")
            selected = {"instance_id": INSTANCE_ID, "label": INSTANCE_LABEL,
                        "status": status, "hourly_price": price}
    if selected is None or running > 1:
        raise SafetyError("retained instance missing or account overlap")
    credit = _number(user.get("credit"))
    if credit > CAMPAIGN_START_CREDIT:
        raise SafetyError("credit increased; cumulative spending requires reconciliation")
    return {"schema_version": "curriculum-provider-audit-v1", "sanitized": True,
            "audit_unix": timestamp, **selected, "running_instances": running,
            "credit": credit, "cumulative_spent": CAMPAIGN_START_CREDIT - credit,
            "credit_method": "installed_cli_show_user_without_legacy_owner_query"}


def validate_budget(audit, *, now, deadline_unix, job_seconds, reserve_seconds=RETRIEVAL_RESERVE_SECONDS):
    """Fail closed before a job; include the entire remaining guard window.

Credit deltas are valid only without intervening top-ups/refunds; an increased
balance is rejected. The existing lifecycle guard remains externally owned.
"""
    if not isinstance(audit, dict) or audit.get("sanitized") is not True or audit.get("schema_version") != "curriculum-provider-audit-v1":
        raise SafetyError("missing sanitized provider audit")
    if type(audit.get("instance_id")) is not int or audit["instance_id"] != INSTANCE_ID or audit.get("label") != INSTANCE_LABEL:
        raise SafetyError("retained instance identity mismatch")
    if audit.get("status") not in {"running", "stopped"} or type(audit.get("running_instances")) is not int or audit["running_instances"] != (audit["status"] == "running"):
        raise SafetyError("invalid audited account concurrency")
    now, timestamp = _number(now), _number(audit.get("audit_unix"))
    deadline, job, reserve = _number(deadline_unix), _number(job_seconds, positive=True), _number(reserve_seconds, positive=True)
    if not 0 <= now - timestamp <= AUDIT_MAX_AGE_SECONDS:
        raise SafetyError("provider audit is stale or from the future")
    if reserve < RETRIEVAL_RESERVE_SECONDS or deadline - now < job + reserve:
        raise SafetyError("deadline cannot cover job and retrieval reserve")
    price, credit = _number(audit.get("hourly_price")), _number(audit.get("credit"))
    spent = _number(audit.get("cumulative_spent"))
    if price > MAX_HOURLY_PRICE or credit > CAMPAIGN_START_CREDIT or not math.isclose(spent, CAMPAIGN_START_CREDIT - credit, rel_tol=0, abs_tol=1e-9):
        raise SafetyError("audited price or cumulative spending is inconsistent")
    # The balance was observed at audit time, not at this later launch check.
    planned_cost = price * (deadline - timestamp) / 3600
    remaining = CAMPAIGN_CEILING - spent
    if remaining < 0 or planned_cost > remaining or planned_cost > credit:
        raise SafetyError("guard window exceeds remaining campaign credit")
    return {"instance_id": INSTANCE_ID, "deadline_unix": deadline,
            "cumulative_spent": spent, "remaining_campaign_credit": remaining,
            "planned_cost_ceiling": planned_cost, "reserve_seconds": reserve}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    verify = subparsers.add_parser("verify-local", help="verify a downloaded block and publish its ACK")
    verify.add_argument("block_dir")
    subparsers.add_parser("audit", help="sanitized read-only provider identity/status/credit audit")
    args = parser.parse_args(argv)
    try:
        result = preserve_block(args.block_dir)["ack"] if args.operation == "verify-local" else audit_provider()
    except Exception:
        # Even an unexpected CLI/JSON/runtime exception may embed credentials.
        print("curriculum safety check failed; no continuation authorized", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
