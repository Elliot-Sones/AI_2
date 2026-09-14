#!/usr/bin/env python3
"""Reuse the verified retained-instance guard for this campaign's first batch.

Local orchestration only: intentionally depends on the archived guard in the
learning-validation sibling checkout. User approved $12 total on 2026-09-06;
this batch is capped at two hours. A replacement instance may be adopted only
from an explicit recent receipt written by the root driver.
No pricing or overlap waiver. Subsequent batches require a cumulative-cost audit.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


REMOTE_ROOT = "/workspace/AI_2/throughput-scaling-20260906"
NEW_IMAGE_REMOTE_BASE_ROOT = "/workspace"
NEW_IMAGE_REMOTE_PYTHON = "/opt/conda/bin/python"
ADOPTED_PRICE_CEILING = 0.40
ADOPTED_READINESS_TIMEOUT_SECONDS = 900
MAX_SECONDS = 7200
RETRIEVAL_RESERVE_SECONDS = 300
STOP_VERIFY_SECONDS = 120
RECEIPT_MAX_AGE_SECONDS = 600
RESUME_AUDIT_MAX_AGE_SECONDS = 600
USER_CREDIT_CEILING = 12.0
RESUME_BATCH_CEILING_FLOOR = ADOPTED_PRICE_CEILING * (
    MAX_SECONDS + RETRIEVAL_RESERVE_SECONDS + STOP_VERIFY_SECONDS
) / 3600.0
RETRIEVAL_COMMAND_TIMEOUT_SECONDS = 25
RETRIEVAL_KILL_GRACE_SECONDS = 5.0


@dataclasses.dataclass(frozen=True)
class InstanceRecord:
    instance_id: int
    label: str
    ssh_host: str
    ssh_port: int
    hourly_price: float
    created_unix: float

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, now: float | None = None) -> "InstanceRecord":
        now = finite_float(time.time() if now is None else now, "now", minimum=0.0)
        if data.get("created_by_campaign") is not True:
            raise RuntimeError("instance record missing created_by_campaign:true")
        for key in ("id", "label", "ssh_host", "ssh_port", "created_unix"):
            if key not in data or data[key] in (None, ""):
                raise RuntimeError(f"instance record missing {key}")
        instance_id = valid_positive_int(data["id"], "id")
        ssh_port = valid_port(data["ssh_port"], "ssh_port")
        created_unix = finite_float(data["created_unix"], "created_unix", minimum=0.0, maximum=now)
        if now - created_unix > RECEIPT_MAX_AGE_SECONDS:
            raise RuntimeError("instance record is not recent")
        price = data.get("hourly_price", data.get("dph_total"))
        if price is None:
            raise RuntimeError("instance record missing hourly_price")
        hourly_price = finite_float(price, "hourly_price", minimum=0.0)
        return cls(
            instance_id=instance_id,
            label=str(data["label"]),
            ssh_host=str(data["ssh_host"]),
            ssh_port=ssh_port,
            hourly_price=hourly_price,
            created_unix=created_unix,
        )


@dataclasses.dataclass(frozen=True)
class ResumeAudit:
    instance_id: int
    label: str
    created_unix: float
    quoted_hourly_price: float
    audit_unix: float
    cumulative_spent: float
    planned_batch_ceiling: float

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, now: float | None = None) -> "ResumeAudit":
        now = finite_float(time.time() if now is None else now, "now", minimum=0.0)
        if data.get("resume_audit") is not True:
            raise RuntimeError("resume audit missing resume_audit:true")
        receipt = data.get("original_campaign_receipt")
        live_audit = data.get("live_account_audit")
        if not isinstance(receipt, dict):
            raise RuntimeError("resume audit missing original_campaign_receipt")
        if not isinstance(live_audit, dict):
            raise RuntimeError("resume audit missing live_account_audit")
        if live_audit.get("sanitized") is not True:
            raise RuntimeError("resume audit live_account_audit must be sanitized")
        for key in ("id", "label", "created_unix", "quoted_hourly_price"):
            if key not in receipt or receipt[key] in (None, ""):
                raise RuntimeError(f"original campaign receipt missing {key}")
        audit_unix = finite_float(
            data.get("audit_unix", live_audit.get("audit_unix", 0.0)),
            "audit_unix",
            minimum=0.0,
            maximum=now,
        )
        if now - audit_unix > RESUME_AUDIT_MAX_AGE_SECONDS:
            raise RuntimeError("resume audit is stale")
        cumulative_spent = finite_float(data.get("cumulative_spent", 0.0), "cumulative_spent", minimum=0.0)
        planned_batch_ceiling = finite_float(
            data.get("planned_batch_ceiling", 0.0),
            "planned_batch_ceiling",
            minimum=RESUME_BATCH_CEILING_FLOOR,
        )
        credit_ceiling = finite_float(
            data.get("user_credit_ceiling", USER_CREDIT_CEILING),
            "user_credit_ceiling",
            minimum=0.01,
            maximum=USER_CREDIT_CEILING,
        )
        if credit_ceiling > USER_CREDIT_CEILING:
            raise RuntimeError("resume audit cannot raise approved budget ceiling")
        if cumulative_spent + planned_batch_ceiling > credit_ceiling:
            raise RuntimeError("resume audit would exceed approved budget")
        instance_id = valid_positive_int(receipt["id"], "id")
        label = str(receipt["label"])
        instances = live_audit.get("instances")
        if not isinstance(instances, list):
            raise RuntimeError("resume audit live_account_audit.instances must be a list")
        selected = [item for item in instances if isinstance(item, dict) and _raw_instance_id(item) == instance_id]
        if len(selected) != 1:
            raise RuntimeError(f"resume audit selected id {instance_id} not found exactly once")
        selected_instance = selected[0]
        if str(selected_instance.get("label") or "") != label:
            raise RuntimeError("resume audit selected label does not match original receipt")
        if not _raw_instance_stopped(selected_instance):
            raise RuntimeError("resume audit selected instance is not stopped")
        for item in instances:
            if isinstance(item, dict) and _raw_instance_id(item) != instance_id and _raw_instance_running(item):
                raise RuntimeError(f"resume audit unrelated running instance present: {_raw_instance_id(item)}")
        created_unix = finite_float(receipt["created_unix"], "created_unix", minimum=0.0, maximum=now)
        quoted = finite_float(receipt["quoted_hourly_price"], "quoted_hourly_price", minimum=0.0)
        if quoted > ADOPTED_PRICE_CEILING:
            raise RuntimeError("original campaign quoted price exceeds campaign price ceiling")
        return cls(
            instance_id=instance_id,
            label=label,
            created_unix=created_unix,
            quoted_hourly_price=quoted,
            audit_unix=audit_unix,
            cumulative_spent=cumulative_spent,
            planned_batch_ceiling=planned_batch_ceiling,
        )


def finite_float(value: Any, field: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise RuntimeError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise RuntimeError(f"{field} must be <= {maximum}")
    return result


def valid_positive_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise RuntimeError(f"{field} must be a positive integer")
    return result


def valid_port(value: Any, field: str) -> int:
    result = valid_positive_int(value, field)
    if result > 65535:
        raise RuntimeError(f"{field} must be in range 1..65535")
    return result


def _raw_instance_id(item: dict[str, Any]) -> int | None:
    value = item.get("id", item.get("instance_id"))
    if value in (None, ""):
        return None
    return valid_positive_int(value, "instance id")


def _raw_instance_running(item: dict[str, Any]) -> bool:
    actual = str(item.get("actual_status", "")).lower()
    state = str(item.get("cur_state", item.get("state", ""))).lower()
    return actual == "running" or state == "running"


def _raw_instance_stopped(item: dict[str, Any]) -> bool:
    if _raw_instance_running(item):
        return False
    actual = str(item.get("actual_status", "")).lower()
    state = str(item.get("cur_state", item.get("state", ""))).lower()
    return actual in {"exited", "stopped"} or state in {"stopped", "exited"}


def load_instance_record(path: Path) -> InstanceRecord:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("instance record must be a JSON object")
    return InstanceRecord.from_mapping(data)


def load_resume_audit(path: Path) -> ResumeAudit:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("resume audit must be a JSON object")
    return ResumeAudit.from_mapping(data)


def load_guard(instance_record: Path | None = None, resume_audit_path: Path | None = None):
    if instance_record is not None and resume_audit_path is not None:
        raise RuntimeError("use either --instance-record or --resume-audit, not both")
    receipt = load_instance_record(instance_record) if instance_record is not None else None
    resume_audit = load_resume_audit(resume_audit_path) if resume_audit_path is not None else None
    if resume_audit is not None:
        return load_guard_from_resume_audit(resume_audit)
    return load_guard_from_record(receipt)


def load_guard_from_record(receipt: InstanceRecord | None):
    base = Path(__file__).resolve().parent.parent / "AI_2-learning-validation-20260905"
    path = base / "baselines/active-movement-20260905/cloud_guard.py"
    spec = importlib.util.spec_from_file_location("_campaign_retained_guard", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load archived lifecycle guard: {path}")
    guard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)
    guard.REMOTE_ROOT = guard._base.REMOTE_ROOT = REMOTE_ROOT
    guard.MAX_SECONDS = guard._base.MAX_SECONDS = MAX_SECONDS
    guard.RETRIEVAL_RESERVE_SECONDS = guard._base.RETRIEVAL_RESERVE_SECONDS = RETRIEVAL_RESERVE_SECONDS
    guard.stop_and_verify = _stop_and_verify_for(guard)
    guard._base.stop_and_verify = guard.stop_and_verify
    if receipt is not None:
        _bind_receipt(guard, receipt)
    return guard


def _bind_receipt(guard: Any, receipt: InstanceRecord) -> None:
    original_run_start_guard = guard._base.run_start_guard
    for module in (guard, guard._base):
        module.INSTANCE_ID = receipt.instance_id
        module.INSTANCE_LABEL = receipt.label
        module.PRICE_CEILING = ADOPTED_PRICE_CEILING
        module.DEFAULT_SSH_HOST = receipt.ssh_host
        module.DEFAULT_SSH_PORT = receipt.ssh_port
        module.REMOTE_ROOT = REMOTE_ROOT
        module.REMOTE_BASE_ROOT = NEW_IMAGE_REMOTE_BASE_ROOT
        module.REMOTE_PYTHON = NEW_IMAGE_REMOTE_PYTHON
        module.MAX_SECONDS = MAX_SECONDS
        module.RETRIEVAL_RESERVE_SECONDS = RETRIEVAL_RESERVE_SECONDS
        module.READINESS_TIMEOUT_SECONDS = ADOPTED_READINESS_TIMEOUT_SECONDS
    guard.validate_account = _validate_account_for(guard)
    guard._base.validate_account = guard.validate_account
    guard._base._ack_result = _strict_ack_result_for(guard)
    guard.run_start_guard = _run_start_guard_for(guard, receipt, original_run_start_guard)


def load_guard_from_resume_audit(audit: ResumeAudit):
    guard = load_guard_from_record(None)
    original_run_start_guard = guard._base.run_start_guard
    for module in (guard, guard._base):
        module.INSTANCE_ID = audit.instance_id
        module.INSTANCE_LABEL = audit.label
        module.PRICE_CEILING = ADOPTED_PRICE_CEILING
        module.REMOTE_ROOT = REMOTE_ROOT
        module.REMOTE_BASE_ROOT = NEW_IMAGE_REMOTE_BASE_ROOT
        module.REMOTE_PYTHON = NEW_IMAGE_REMOTE_PYTHON
        module.MAX_SECONDS = MAX_SECONDS
        module.RETRIEVAL_RESERVE_SECONDS = RETRIEVAL_RESERVE_SECONDS
        module.READINESS_TIMEOUT_SECONDS = ADOPTED_READINESS_TIMEOUT_SECONDS
    guard.validate_account = _validate_account_for(guard)
    guard._base.validate_account = guard.validate_account
    guard._base._ack_result = _strict_ack_result_for(guard)
    guard.run_start_guard = _run_resume_start_guard_for(guard, audit, original_run_start_guard)
    return guard


def _validate_account_for(guard: Any):
    def validate_account(instances: list[dict[str, Any]]) -> dict[str, Any]:
        selected = [item for item in instances if guard.instance_id(item) == guard.INSTANCE_ID]
        if not selected:
            raise guard.GuardError(f"selected instance {guard.INSTANCE_ID} was not found")
        if len(selected) > 1:
            raise guard.GuardError(f"multiple records for selected instance {guard.INSTANCE_ID}")
        for item in instances:
            if guard.instance_id(item) != guard.INSTANCE_ID and guard.instance_is_running(item):
                raise guard.GuardError(f"unrelated running instance present: {guard.instance_id(item)}")
        instance = selected[0]
        gpu_name = str(instance.get("gpu_name") or "")
        num_gpus = int(instance.get("num_gpus") or 0)
        if "4090" not in gpu_name or num_gpus != 1:
            raise guard.GuardError(f"selected instance must have one RTX 4090, got {num_gpus} {gpu_name}")
        try:
            price = finite_float(
                instance.get("dph_total") or instance.get("hourly_price") or 999.0,
                "selected instance price",
                minimum=0.0,
            )
        except RuntimeError as exc:
            raise guard.GuardError(str(exc)) from exc
        if price > guard.PRICE_CEILING:
            raise guard.GuardError(f"selected instance price {price:.6f}/hr exceeds {guard.PRICE_CEILING:.2f}/hr")
        if str(instance.get("label") or "") != guard.INSTANCE_LABEL:
            raise guard.GuardError(f"selected instance label does not match {guard.INSTANCE_LABEL}")
        return instance

    return validate_account


def _stop_and_verify_for(guard: Any):
    def stop_and_verify(provider: Any, state_dir: Path, clock: Any, poll_seconds: float) -> None:
        provider.stop_instance(guard.INSTANCE_ID)
        deadline = clock.now() + STOP_VERIFY_SECONDS
        last_state: dict[str, Any] | None = None
        while True:
            instances = provider.list_instances()
            target = next((item for item in instances if guard.instance_id(item) == guard.INSTANCE_ID), None)
            if target is None:
                raise guard.GuardError(f"stop verification failed: selected instance {guard.INSTANCE_ID} not found")
            last_state = target
            if guard.instance_is_stopped(target):
                guard.write_event(state_dir, "stop_verified", instance_id=guard.INSTANCE_ID)
                return
            if clock.now() >= deadline:
                break
            clock.sleep(min(poll_seconds, max(0.0, deadline - clock.now())))
        raise guard.GuardError(
            "stop verification failed after 120 seconds: "
            f"actual_status={last_state.get('actual_status') if last_state else None} "
            f"cur_state={last_state.get('cur_state') if last_state else None}"
        )

    return stop_and_verify


class AdoptedProvider:
    def __init__(self, provider: Any, receipt: InstanceRecord, guard: Any):
        self.provider = provider
        self.receipt = receipt
        self.guard = guard

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    def list_instances(self) -> list[dict[str, Any]]:
        return self.provider.list_instances()

    def start_instance(self, instance_id: int) -> dict[str, Any]:
        if int(instance_id) != self.receipt.instance_id:
            raise self.guard.GuardError(f"refusing to start unrecorded instance {instance_id}")
        instance = self.guard.validate_account(self.provider.list_instances())
        if self.guard.instance_is_running(instance):
            return {"acknowledged": True, "already_running": True}
        result = self.provider.start_instance(instance_id)
        return classify_start_result(result, self.guard)

    def check_readiness(self, host: str, port: int, remote_root: str, interpreter: str) -> bool:
        return self.provider.check_readiness(
            host,
            port,
            NEW_IMAGE_REMOTE_BASE_ROOT,
            NEW_IMAGE_REMOTE_PYTHON,
        )

    def retrieve(self, host: str, port: int, remote_root: str, local_dir: Path) -> dict[str, bool]:
        return resumable_retrieve(
            host=host,
            port=port,
            remote_root=remote_root,
            local_dir=local_dir,
            state_dir=Path(getattr(self.provider, "state_dir", Path(local_dir).parent)),
            popen_factory=getattr(self, "popen_factory", subprocess.Popen),
        )


class ResumeProvider:
    def __init__(self, provider: Any, audit: ResumeAudit, guard: Any):
        self.provider = provider
        self.audit = audit
        self.guard = guard

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    def list_instances(self) -> list[dict[str, Any]]:
        return self.provider.list_instances()

    def start_instance(self, instance_id: int) -> dict[str, Any]:
        if int(instance_id) != self.audit.instance_id:
            raise self.guard.GuardError(f"refusing to start unrecorded instance {instance_id}")
        instance = self.guard.validate_account(self.provider.list_instances())
        if self.guard.instance_is_running(instance):
            return {"acknowledged": True, "already_running": True}
        result = self.provider.start_instance(instance_id)
        return classify_start_result(result, self.guard)

    def check_readiness(self, host: str, port: int, remote_root: str, interpreter: str) -> bool:
        return self.provider.check_readiness(
            host,
            port,
            NEW_IMAGE_REMOTE_BASE_ROOT,
            NEW_IMAGE_REMOTE_PYTHON,
        )

    def retrieve(self, host: str, port: int, remote_root: str, local_dir: Path) -> dict[str, bool]:
        return resumable_retrieve(
            host=host,
            port=port,
            remote_root=remote_root,
            local_dir=local_dir,
            state_dir=Path(getattr(self.provider, "state_dir", Path(local_dir).parent)),
            popen_factory=getattr(self, "popen_factory", subprocess.Popen),
        )


def resumable_retrieve(
    *,
    host: str,
    port: int,
    remote_root: str,
    local_dir: Path,
    state_dir: Path,
    timeout_seconds: float = RETRIEVAL_COMMAND_TIMEOUT_SECONDS,
    kill_grace_seconds: float = RETRIEVAL_KILL_GRACE_SECONDS,
    popen_factory: Any = subprocess.Popen,
    killpg: Any = os.killpg,
) -> dict[str, bool]:
    """Retrieve artifacts and W&B with rsync partials that survive timeouts.

    The retained guard calls ``provider.retrieve`` inside its normal stop/budget
    loop. Keeping the implementation here changes only future campaign adapter
    instances and leaves the archived sibling guard untouched.
    """

    _validate_ssh_target(host, port)
    local_dir = Path(local_dir)
    state_dir = Path(state_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    private_log = state_dir / "retrieval-private.jsonl"
    results: dict[str, bool] = {}
    for name in ("artifacts", "wandb"):
        target = local_dir / name
        target.mkdir(parents=True, exist_ok=True)
        partial_dir = target / ".rsync-partial"
        partial_dir.mkdir(parents=True, exist_ok=True)
        stale_complete_flag = target / ".retrieval-complete.json"
        if stale_complete_flag.exists():
            stale_complete_flag.unlink()
        command = build_resumable_rsync_command(
            host=host,
            port=port,
            remote_root=remote_root,
            name=name,
            target=target,
        )
        status, stdout, stderr = _run_retrieval_command(
            command,
            timeout_seconds=timeout_seconds,
            kill_grace_seconds=kill_grace_seconds,
            popen_factory=popen_factory,
            killpg=killpg,
        )
        _write_retrieval_private(private_log, name, command, status, stdout, stderr)
        ok = status == 0
        results[name] = ok
        _write_retrieval_status(target / ".retrieval-status.json", name=name, remote_root=remote_root, status=status)
    return results


def build_resumable_rsync_command(
    *,
    host: str,
    port: int,
    remote_root: str,
    name: str,
    target: Path,
) -> list[str]:
    _validate_retrieval_name(name)
    _validate_ssh_target(host, port)
    ssh_command = " ".join(
        [
            "ssh",
            "-p",
            str(port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
        ]
    )
    return [
        "rsync",
        "-az",
        "--partial",
        "--partial-dir=.rsync-partial",
        "-e",
        ssh_command,
        f"root@{host}:{sh_quote(remote_root)}/{name}/",
        f"{Path(target)}/",
    ]


def _run_retrieval_command(
    command: list[str],
    *,
    timeout_seconds: float,
    kill_grace_seconds: float,
    popen_factory: Any,
    killpg: Any,
) -> tuple[int | str, str | None, str | None]:
    proc = popen_factory(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return int(proc.returncode), stdout, stderr
    except subprocess.TimeoutExpired as exc:
        try:
            _kill_process_group(killpg, proc.pid, signal.SIGTERM)
            proc.communicate(timeout=kill_grace_seconds)
            status: int | str = "timeout_terminated"
        except ProcessLookupError:
            status = "timeout_process_gone"
        except subprocess.TimeoutExpired:
            try:
                _kill_process_group(killpg, proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                status = "timeout_process_gone"
            else:
                status = "timeout_killed"
            try:
                proc.communicate(timeout=kill_grace_seconds)
            except subprocess.TimeoutExpired:
                pass
        return status, _decode_maybe(exc.output), _decode_maybe(exc.stderr)


def _kill_process_group(killpg: Any, pid: int, signum: int) -> None:
    killpg(pid, signum)


def _write_retrieval_status(path: Path, *, name: str, remote_root: str, status: int | str) -> None:
    data = {
        "name": name,
        "remote_root": remote_root,
        "attempted_unix": time.time(),
        "status": status,
        "success": status == 0,
        "note": "rsync success is not a final checksum verification",
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_retrieval_private(
    private_log: Path,
    name: str,
    command: list[str],
    status: int | str,
    stdout: str | None,
    stderr: str | None,
) -> None:
    record = {
        "ts": time.time(),
        "name": name,
        "command": redact_command(command),
        "status": status,
        "stdout_len": len(stdout or ""),
        "stderr_len": len(stderr or ""),
    }
    with private_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.chmod(private_log, 0o600)


def _validate_ssh_target(host: str, port: int) -> None:
    if not host or any(ch.isspace() for ch in host) or "/" in host or "\\" in host:
        raise RuntimeError("ssh host is not safe for retrieval")
    valid_port(port, "ssh_port")


def _validate_retrieval_name(name: str) -> None:
    if name not in {"artifacts", "wandb"}:
        raise RuntimeError(f"unsupported retrieval directory: {name}")


def _decode_maybe(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def redact_command(command: list[str]) -> list[str]:
    redacted = []
    for part in command:
        lowered = str(part).lower()
        if "key" in lowered or "token" in lowered:
            redacted.append("<redacted>")
        else:
            redacted.append(str(part))
    return redacted


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def classify_start_result(result: Any, guard: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        if result.get("queued_unavailable"):
            raise guard.GuardError("start queued because resources are unavailable")
        if result.get("acknowledged") is True:
            return result
    if isinstance(result, str):
        lowered = result.lower()
        if "required resources are currently unavailable" in lowered or "state change queued" in lowered:
            raise guard.GuardError("start queued because resources are unavailable")
        if f"starting instance {guard.INSTANCE_ID}" in lowered:
            return {"acknowledged": True, "text_ack": True}
    raise guard.GuardError("start did not return an accepted acknowledgement")


def _strict_ack_result_for(guard: Any):
    def strict_ack_result(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            lowered = result.lower()
            if "required resources are currently unavailable" in lowered or "state change queued" in lowered:
                raise guard.GuardError("CLI acknowledgement reported queued unavailable resources")
            if lowered.strip() == "null":
                raise guard.GuardError("CLI acknowledgement was null only")
            if f"starting instance {guard.INSTANCE_ID}" in lowered or f"stopping instance {guard.INSTANCE_ID}" in lowered:
                return {"acknowledged": True, "text_ack": True}
        raise guard.GuardError("CLI acknowledgement was not accepted")

    return strict_ack_result


def _run_start_guard_for(guard: Any, receipt: InstanceRecord, original_run_start_guard: Any):
    def run_start_guard(
        *,
        state_dir: Path,
        provider: Any | None = None,
        clock: Any | None = None,
        once: bool = False,
        poll_seconds: float = 5.0,
        retrieval_seconds: float = guard.RETRIEVAL_SECONDS,
        max_wait_iterations: int | None = None,
    ) -> dict[str, Any]:
        base_provider = provider or guard.VastCliProvider(state_dir)
        return original_run_start_guard(
            state_dir=state_dir,
            provider=AdoptedProvider(base_provider, receipt, guard),
            clock=clock,
            once=once,
            poll_seconds=poll_seconds,
            retrieval_seconds=retrieval_seconds,
            max_wait_iterations=max_wait_iterations,
        )

    return run_start_guard


def _run_resume_start_guard_for(guard: Any, audit: ResumeAudit, original_run_start_guard: Any):
    def run_start_guard(
        *,
        state_dir: Path,
        provider: Any | None = None,
        clock: Any | None = None,
        once: bool = False,
        poll_seconds: float = 5.0,
        retrieval_seconds: float = guard.RETRIEVAL_SECONDS,
        max_wait_iterations: int | None = None,
    ) -> dict[str, Any]:
        base_provider = provider or guard.VastCliProvider(state_dir)
        return original_run_start_guard(
            state_dir=state_dir,
            provider=ResumeProvider(base_provider, audit, guard),
            clock=clock,
            once=once,
            poll_seconds=poll_seconds,
            retrieval_seconds=retrieval_seconds,
            max_wait_iterations=max_wait_iterations,
        )

    return run_start_guard


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--start", action="store_true", help="start/watch/retrieve/stop selected instance")
    action.add_argument("--stop-request", action="store_true", help="write stop.request in the state dir")
    parser.add_argument("--state-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--instance-record", type=Path, help="recent root-created replacement instance receipt")
    parser.add_argument("--resume-audit", type=Path, help="recent sanitized subsequent-batch resume audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    guard = load_guard(args.instance_record, args.resume_audit)
    try:
        if args.stop_request:
            path = guard.request_stop_file(args.state_dir)
            print(json.dumps({"stop_request": str(path)}, sort_keys=True))
            return 0
        result = guard.run_start_guard(state_dir=args.state_dir, poll_seconds=args.poll_seconds)
    except Exception as exc:
        print(f"throughput campaign cloud guard failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
