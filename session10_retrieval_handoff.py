#!/usr/bin/env python3
"""Explicit, audited, running-only session10 retrieval watchdog.

Never starts, rents, destroys, trains, or copies. The root owns manual retrieval
and retirement of the old guard. ``armed.json`` proves initial readiness only;
also check the live PID and ``status.json`` before retiring the old watchdog.
The 30-minute monotonic deadline requests stop; bounded provider calls and stop
verification can add closeout latency. Any failure after admission requests stop.
Original creation time is preserved; this is a separate handoff admission mode.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import campaign_cloud


INSTANCE_ID = 50149729
LABEL = "ai2-native-ppo-session10-20260907"
REMOTE_ROOT = "/workspace/AI_2/native-ppo-session10"
MAX_SECONDS = 1800
AUDIT_MAX_AGE = 300
INITIAL_CREDIT = 12.315037


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f"invalid {name}")
    return float(value)


def validate_live(instances):
    require(isinstance(instances, list), "instances must be a list")
    selected = [row for row in instances if row.get("id", row.get("instance_id")) == INSTANCE_ID]
    require(len(selected) == 1, "selected instance must appear exactly once")
    row = selected[0]
    require(row.get("label") == LABEL, "selected label mismatch")
    require(str(row.get("actual_status", "")).lower() == "running"
            and str(row.get("cur_state", "running")).lower() == "running",
            "selected instance must already be running; never restart")
    require(str(row.get("intended_status", "running")).lower() == "running",
            "selected instance is pending stop")
    require(row.get("num_gpus") == 1 and "4090" in str(row.get("gpu_name")), "GPU mismatch")
    require(0 < numeric(row.get("dph_total"), "hourly price") <= 0.40, "price ceiling exceeded")
    for other in instances:
        if other is row:
            continue
        states = {str(other.get(key, "")).lower()
                  for key in ("actual_status", "cur_state", "intended_status", "next_state")}
        require(not states.intersection({"running", "loading", "creating"}), "unrelated active instance")
    return row


def validate_audit(audit, original_bytes, *, now, max_seconds):
    require(type(max_seconds) is int and 1 <= max_seconds <= MAX_SECONDS, "max seconds must be 1..1800")
    require(audit.get("handoff_audit") is True, "explicit handoff audit required")
    checked = numeric(audit.get("audit_unix"), "audit timestamp")
    require(0 <= now - checked <= AUDIT_MAX_AGE, "handoff audit is stale or future")
    require(audit.get("original_receipt_sha256") == hashlib.sha256(original_bytes).hexdigest(),
            "original receipt hash mismatch")
    original = json.loads(original_bytes)
    require(original.get("created_by_campaign") is True and original.get("id") == INSTANCE_ID
            and original.get("label") == LABEL, "original ownership mismatch")
    require(numeric(original.get("created_unix"), "original creation") <= checked, "future original creation")
    require(0 < numeric(original.get("hourly_price"), "original price") <= 0.40, "original price mismatch")
    require(bool(original.get("ssh_host")), "missing original SSH host")
    campaign_cloud.valid_port(original.get("ssh_port"), "original SSH port")
    live = audit.get("live_account_audit", {})
    require(live.get("sanitized") is True, "sanitized live account audit required")
    validate_live(live.get("instances"))
    credit = audit.get("credit_audit", {})
    require(credit.get("ok") is True, "successful credit audit required")
    fetched = datetime.datetime.fromisoformat(credit.get("fetched_at_utc", ""))
    require(fetched.tzinfo is not None and 0 <= now - fetched.timestamp() <= AUDIT_MAX_AGE,
            "credit audit is stale or future")
    balance = numeric(credit.get("balance_fields", {}).get("credit"), "credit balance")
    spent = numeric(audit.get("cumulative_spent"), "cumulative spend")
    ceiling = numeric(audit.get("planned_batch_ceiling"), "planned batch ceiling", 0.50)
    require(audit.get("user_credit_ceiling") == 12, "user ceiling must remain 12")
    require(spent + 1e-6 >= max(0, INITIAL_CREDIT - balance), "cumulative spend underreports credit audit")
    require(spent + ceiling <= 12 and ceiling <= balance, "authorized budget exhausted")
    manual = audit.get("manual_retrieval", {})
    require(type(manual.get("pid")) is int and manual["pid"] > 1, "manual retrieval PID required")
    require(isinstance(manual.get("destination"), str) and Path(manual["destination"]).is_absolute(),
            "absolute manual retrieval destination required")
    return original


class Clock:
    now = staticmethod(time.time)
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


def run_watchdog(audit, original_bytes, state_dir, *, max_seconds=1800, poll_seconds=15,
                 provider=None, clock=None, install_signals=True):
    clock = clock or Clock()
    original = validate_audit(audit, original_bytes, now=clock.now(), max_seconds=max_seconds)
    require(0 < numeric(poll_seconds, "poll seconds") <= 15, "poll seconds must be <=15")
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=False)
    record = campaign_cloud.InstanceRecord(
        instance_id=INSTANCE_ID, label=LABEL, ssh_host=original["ssh_host"],
        ssh_port=original["ssh_port"], hourly_price=original["hourly_price"],
        created_unix=original["created_unix"])
    guard = campaign_cloud.load_guard_from_record(record)
    provider = provider or guard.VastCliProvider(state_dir)
    start = clock.now()
    deadline = clock.monotonic() + max_seconds
    session = dict(instance_id=INSTANCE_ID, label=LABEL, pid=os.getpid(),
                   started_unix=start, deadline_unix=start + max_seconds,
                   max_seconds=max_seconds, original_created_unix=record.created_unix,
                   remote_root=REMOTE_ROOT, deletion_authorized=False,
                   manual_retrieval=audit["manual_retrieval"],
                   manual_retrieval_status="externally_managed_not_verified",
                   status="admitted_not_armed", ready=False)
    previous = {}
    interrupted = []

    def handle_signal(signum, frame):
        interrupted.append(signum)

    try:
        if install_signals:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.signal(signum, handle_signal)
        guard.write_json_atomic(state_dir / "audit.json", audit, mode=0o600)
        guard.write_json_atomic(state_dir / "status.json", session, mode=0o600)
        row = validate_live(provider.list_instances())
        host = row.get("ssh_host") or record.ssh_host
        port = row.get("ssh_port") or record.ssh_port
        if not provider.check_readiness(host, port, "/workspace", "/opt/conda/bin/python"):
            raise RuntimeError("SSH readiness failed")
        if interrupted or clock.monotonic() >= deadline:
            raise RuntimeError("interrupted or deadline elapsed before arming")
        session.update(status="armed", ready=True, checked_unix=clock.now())
        guard.write_json_atomic(state_dir / "armed.json", session, mode=0o600)
        guard.write_json_atomic(state_dir / "status.json", session, mode=0o600)
        while not interrupted and not (state_dir / "stop.request").exists() and clock.monotonic() < deadline:
            validate_live(provider.list_instances())
            session["checked_unix"] = clock.now()
            guard.write_json_atomic(state_dir / "status.json", session, mode=0o600)
            clock.sleep(min(poll_seconds, max(0, deadline - clock.monotonic())))
        session["stop_reason"] = "signal" if interrupted else "stop.request" if (state_dir / "stop.request").exists() else "deadline"
    except BaseException as exc:
        session.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        try:
            session["status"] = "stopping"
            guard.write_json_atomic(state_dir / "status.json", session, mode=0o600)
        finally:
            try:
                guard.stop_and_verify(provider, state_dir, clock, poll_seconds)
                session.update(status="stopped", stopped_unix=clock.now())
                guard.write_json_atomic(state_dir / "status.json", session, mode=0o600)
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
    return session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--instance-record", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args(argv)
    try:
        result = run_watchdog(json.loads(args.audit.read_bytes()), args.instance_record.read_bytes(),
                              args.state_dir, max_seconds=args.max_seconds, poll_seconds=args.poll_seconds)
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
