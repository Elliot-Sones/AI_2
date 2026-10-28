#!/usr/bin/env python3
"""Continue the staged native-contact campaign without duplicating GPU jobs.

This controller does not create or destroy providers.  It talks only to the
already-approved staged host and launches named jobs through the existing
remote tools/run_job.py wrapper, which owns the single-GPU lock and deadline.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time


SSH_HOST = "root@70.69.192.6"
SSH_PORT = "26861"
SSH_KEY = Path("/Users/elliot18/.ssh/id_ed25519")
REMOTE_ROOT = Path("/workspace/AI_2/staged-20260920")
REMOTE_ARTIFACTS = REMOTE_ROOT / "artifacts"
LOCAL_ROOT = Path("/Users/elliot18/Projects/ml-research/AI_2/baselines/staged-training-20260920")
DEADLINE_UNIX = 1789888388.520737
POLL_SECONDS = 15.0
RESERVE_SECONDS = 180.0
CANONICAL_PARENT_STEPS = 10_059_776
LANES = 4096
N_STEPS = 512
TRANSITIONS_PER_ROLLOUT = LANES * N_STEPS
OPTIMIZER_STEPS_PER_ROLLOUT = 512
STAGE_TOTAL_ROLLOUTS = {10: 5, 20: 10, 30: 15, 40: 20, 50: 24, 60: 29, 70: 34, 80: 39, 90: 44, 100: 48}
STAGE_SEEDS = {20: 20145, 30: 30145, 40: 40145, 50: 50145, 60: 60145, 70: 70145, 80: 80145, 90: 90145, 100: 100145}
QUALIFICATION = REMOTE_ROOT / "artifacts/contact-qualification-v3/qualification.json"
FROZEN_ROOT = REMOTE_ROOT / "inputs/frozen"
REFERENCE_ROOT = REMOTE_ROOT / "inputs/reference"
NAVIGATION_PARENT = REMOTE_ROOT / "inputs/navigation_parent.zip"


@dataclass(frozen=True)
class PlanStep:
    kind: str
    name: str
    arm: str
    stage: int
    seed: int | None = None
    rollouts: int | None = None
    resume_from: str | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def expected_counter_for_stage(stage: int) -> int:
    if stage not in STAGE_TOTAL_ROLLOUTS:
        raise ValueError(f"unsupported campaign stage {stage}")
    return CANONICAL_PARENT_STEPS + STAGE_TOTAL_ROLLOUTS[stage] * TRANSITIONS_PER_ROLLOUT


def previous_stage(stage: int) -> int:
    if stage == 20:
        return 10
    if stage in (30, 40, 50, 60, 70, 80, 90, 100):
        return stage - 10
    raise ValueError(f"unsupported continuation stage {stage}")


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def atomic_json(path: Path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(tmp, path)
    tmp.unlink()


def append_event(path: Path | None, event):
    if path is None:
        print(json.dumps(event, sort_keys=True), flush=True)
        return
    with Path(path).open("a", encoding="utf8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()


def build_plan():
    plan = [PlanStep(kind="eval", name="eval-G_keep-020m", arm="G_keep", stage=20)]
    stages = [
        ("B_projected", 20, 20145, 5, "B_projected-010m"),
        ("G_keep", 30, 30145, 5, "G_keep-020m"),
        ("B_projected", 30, 30145, 5, "B_projected-020m"),
        ("G_keep", 40, 40145, 5, "G_keep-030m"),
        ("B_projected", 40, 40145, 5, "B_projected-030m"),
        ("G_keep", 50, 50145, 4, "G_keep-040m"),
        ("B_projected", 50, 50145, 4, "B_projected-040m"),
    ]
    for arm, stage, seed, rollouts, parent in stages:
        name = f"{arm}-{stage:03d}m"
        plan.append(PlanStep("train", name, arm, stage, seed=seed, rollouts=rollouts, resume_from=parent))
        plan.append(PlanStep("eval", f"eval-{name}", arm, stage))
    return plan


def validate_name(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name or Path(name).name != name:
        raise ValueError(f"unsafe job/artifact name: {name!r}")
    if not all(char.isalnum() or char in "-_" for char in name):
        raise ValueError(f"unsafe job/artifact name: {name!r}")
    return name


def verify_ack_for_bytes(inventory_bytes: bytes, ack_bytes: bytes, *, expected_name: str) -> dict:
    inventory = json.loads(inventory_bytes)
    ack = json.loads(ack_bytes)
    problems = []
    inventory_hash = bytes_sha256(inventory_bytes)
    if inventory.get("schema_version") != "curriculum-inventory-v1":
        problems.append("bad inventory schema_version")
    if inventory.get("block_id") != expected_name:
        problems.append("inventory block_id mismatch")
    if ack.get("verified") is not True:
        problems.append("ack verified is not true")
    if ack.get("block_id") != inventory.get("block_id"):
        problems.append("ack block_id mismatch")
    if ack.get("inventory_sha256") != inventory_hash:
        problems.append("ack inventory_sha256 mismatch")
    if problems:
        raise ValueError(f"{expected_name} preservation invalid: {problems}")
    return {"inventory": inventory, "ack": ack, "inventory_sha256": inventory_hash}


def verify_local_preservation(local_root: Path, name: str, *, require_final_zip: bool) -> dict:
    validate_name(name)
    block = Path(local_root) / name
    inventory_path = block / "inventory.json"
    ack_path = block / "preservation_ack.json"
    if inventory_path.is_symlink() or ack_path.is_symlink():
        raise ValueError(f"{name} preservation metadata is symlinked")
    inventory_bytes = inventory_path.read_bytes()
    ack_bytes = ack_path.read_bytes()
    verified = verify_ack_for_bytes(inventory_bytes, ack_bytes, expected_name=name)
    files = {row.get("path"): row for row in verified["inventory"].get("files", [])}
    required = ["result.json", "manifest.json"]
    if require_final_zip:
        required.append("final.zip")
    evidence = {}
    for relative in required:
        row = files.get(relative)
        if row is None:
            raise ValueError(f"{name} inventory missing {relative}")
        path = block / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name} unsafe or missing {relative}")
        actual = sha256(path)
        if row.get("bytes") != path.stat().st_size or row.get("sha256") != actual:
            raise ValueError(f"{name} inventory mismatch for {relative}")
        evidence[relative] = {"path": path, "sha256": actual, "bytes": path.stat().st_size}
    result = read_json(block / "result.json")
    manifest = read_json(block / "manifest.json")
    if require_final_zip:
        if result.get("status") != "trained_not_evaluated":
            raise ValueError(f"{name} is not a completed training block")
        if result.get("checkpoint_sha256") != evidence["final.zip"]["sha256"]:
            raise ValueError(f"{name} result checkpoint_sha256 does not match local final.zip")
        summary = result.get("summary") or {}
        counters = result.get("final_model_counters") or {}
        rollouts = result.get("rollouts") or []
        if not isinstance(summary.get("measured_transitions"), int) or summary["measured_transitions"] <= 0:
            raise ValueError(f"{name} training result missing positive measured_transitions")
        if not isinstance(counters.get("num_timesteps"), int) or counters["num_timesteps"] <= 0:
            raise ValueError(f"{name} training result missing positive final num_timesteps")
        if not rollouts:
            raise ValueError(f"{name} training result missing rollout receipts")
    else:
        if result.get("status") != "evaluated_not_promoted":
            raise ValueError(f"{name} is not a completed evaluation block")
    return {"name": name, "directory": block, "files": evidence, "result": result, "manifest": manifest, **verified}


def build_background_launch_shell(
    *, remote_root: Path, deadline_unix: float, name: str, command: list[str],
    python_executable: Path | str = "python3",
) -> str:
    validate_name(name)
    quoted = " ".join(shlex.quote(part) for part in command)
    log_path = Path(remote_root) / "jobs" / (name + ".controller-launch.log")
    return (
        f"cd {shlex.quote(str(remote_root))} "
        f"&& mkdir -p {shlex.quote(str(Path(remote_root) / 'jobs'))} "
        "&& if [ -f source/ai2/paths.py ]; then "
        "ai2_job_runner=source/ai2/runtime/run_campaign_job.py; "
        "else ai2_job_runner=source/run_campaign_job.py; fi "
        f'&& {{ nohup {shlex.quote(str(python_executable))} "$ai2_job_runner" '
        f"--root {shlex.quote(str(remote_root))} "
        f"--deadline-unix {shlex.quote(str(float(deadline_unix)))} "
        f"{shlex.quote(name)} -- {quoted} "
        f"> {shlex.quote(str(log_path))} 2>&1 < /dev/null & echo $!; }}"
    )


class RemoteClient:
    def __init__(self, *, ssh_host=SSH_HOST, ssh_port=SSH_PORT, ssh_key=SSH_KEY,
                 remote_root=REMOTE_ROOT, deadline_unix=DEADLINE_UNIX):
        self.ssh_host = ssh_host
        self.ssh_port = str(ssh_port)
        self.ssh_key = Path(ssh_key)
        self.remote_root = Path(remote_root)
        self.deadline_unix = float(deadline_unix)

    def ssh_base(self):
        return [
            "ssh",
            "-i",
            str(self.ssh_key),
            "-p",
            self.ssh_port,
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=3",
            self.ssh_host,
        ]

    def run_shell(self, command: str, *, timeout=120):
        return subprocess.run(
            self.ssh_base() + [command],
            capture_output=True,
            text=False,
            check=False,
            timeout=timeout,
        )

    def remote_json(self, path: Path):
        command = "cat " + shlex.quote(str(path))
        result = self.run_shell(command, timeout=60)
        if result.returncode:
            if result.returncode == 1:
                return None
            raise RuntimeError(f"remote read failed for {path}: {result.stderr.decode(errors='replace')}")
        return json.loads(result.stdout)

    def remote_file(self, path: Path):
        command = "cat " + shlex.quote(str(path))
        result = self.run_shell(command, timeout=60)
        if result.returncode:
            return None
        return result.stdout

    def job_receipt(self, name):
        validate_name(name)
        return self.remote_json(self.remote_root / "jobs" / f"{name}.json")

    def job_started(self, name):
        validate_name(name)
        return self.remote_json(self.remote_root / "jobs" / f"{name}.started.json")

    def launch_job(self, name, command):
        validate_name(name)
        remote = build_background_launch_shell(
            remote_root=self.remote_root,
            deadline_unix=self.deadline_unix,
            name=name,
            command=command,
        )
        try:
            result = self.run_shell(remote, timeout=30)
        except subprocess.TimeoutExpired as exc:
            receipt = self.job_receipt(name)
            if receipt is not None:
                return receipt
            started = self.job_started(name)
            if started is not None:
                return started
            raise TimeoutError(f"remote launch timed out and no started receipt was found for {name}") from exc
        if result.returncode:
            raise RuntimeError(f"remote launch failed for {name}: {result.stderr.decode(errors='replace')}")
        return {"pid": result.stdout.decode(errors="replace").strip() or None, "name": name}

    def read_artifact_file(self, name, filename):
        validate_name(name)
        if filename not in {"inventory.json", "preservation_ack.json"}:
            raise ValueError("only preservation metadata is read remotely")
        data = self.remote_file(REMOTE_ARTIFACTS / name / filename)
        if data is None:
            raise FileNotFoundError(f"remote artifact missing: {name}/{filename}")
        return data


class Controller:
    def __init__(
        self,
        *,
        local_root=LOCAL_ROOT,
        remote=None,
        deadline_unix=DEADLINE_UNIX,
        sleeper=time.sleep,
        now=time.time,
        poll_seconds=POLL_SECONDS,
        max_wait_seconds=None,
        event_log=None,
    ):
        self.local_root = Path(local_root)
        self.deadline_unix = float(deadline_unix)
        self.remote = remote or RemoteClient(deadline_unix=self.deadline_unix)
        self.sleeper = sleeper
        self.now = now
        self.poll_seconds = float(poll_seconds)
        self.max_wait_seconds = max_wait_seconds
        self.event_log = Path(event_log) if event_log else None
        self.stopped_arms = set()
        self.events = []

    def emit(self, **event):
        event = {"time_unix": self.now(), **event}
        self.events.append(event)
        append_event(self.event_log, event)

    def validate_deadline(self):
        if self.now() > self.deadline_unix - RESERVE_SECONDS:
            raise TimeoutError("deadline reserve exhausted")

    def verify_remote_preservation(self, name):
        inventory = self.remote.read_artifact_file(name, "inventory.json")
        ack = self.remote.read_artifact_file(name, "preservation_ack.json")
        return verify_ack_for_bytes(inventory, ack, expected_name=name)

    def wait_for_preservation(self, name, *, require_final_zip):
        started = self.now()
        while True:
            self.validate_deadline()
            try:
                local = verify_local_preservation(self.local_root, name, require_final_zip=require_final_zip)
                remote = self.verify_remote_preservation(name)
                if remote["inventory_sha256"] != local["inventory_sha256"]:
                    raise ValueError(
                        f"{name} remote inventory_sha256 {remote['inventory_sha256']} "
                        f"does not match local {local['inventory_sha256']}"
                    )
                self.emit(event="preserved", name=name, inventory_sha256=local["inventory_sha256"])
                return {"local": local, "remote": remote}
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
                if self.max_wait_seconds is not None and self.now() - started >= self.max_wait_seconds:
                    raise TimeoutError(f"timed out waiting for preservation of {name}: {exc}") from exc
                self.emit(event="waiting_preservation", name=name, reason=str(exc))
                self.sleeper(self.poll_seconds)

    def preservation_once(self, name, *, require_final_zip):
        local = verify_local_preservation(self.local_root, name, require_final_zip=require_final_zip)
        remote = self.verify_remote_preservation(name)
        if remote["inventory_sha256"] != local["inventory_sha256"]:
            raise ValueError(
                f"{name} remote inventory_sha256 {remote['inventory_sha256']} "
                f"does not match local {local['inventory_sha256']}"
            )
        return {"local": local, "remote": remote}

    def local_preservation_once(self, name, *, require_final_zip):
        return verify_local_preservation(self.local_root, name, require_final_zip=require_final_zip)

    def verify_remote_job_success_if_present(self, name):
        receipt = self.remote.job_receipt(name)
        if receipt is None:
            return None
        if receipt.get("returncode") != 0:
            raise RuntimeError(f"{name} prior receipt returncode is {receipt.get('returncode')}")
        self.emit(event="remote_job_rc0_verified", name=name)
        return receipt

    def validate_train_step(self, step: PlanStep):
        if step.kind != "train":
            raise ValueError("expected train step")
        if step.arm not in {"G_keep", "B_projected"}:
            raise ValueError(f"{step.name} unsupported arm")
        expected_name = f"{step.arm}-{step.stage:03d}m"
        if step.name != expected_name:
            raise ValueError(f"{step.name} does not match exact train plan name {expected_name}")
        if step.seed != STAGE_SEEDS.get(step.stage):
            raise ValueError(f"{step.name} seed {step.seed} != expected {STAGE_SEEDS.get(step.stage)}")
        expected_rollouts = 4 if step.stage in (50, 100) else 5
        if step.rollouts != expected_rollouts:
            raise ValueError(f"{step.name} rollout count {step.rollouts} != expected {expected_rollouts}")
        expected_parent = f"{step.arm}-{previous_stage(step.stage):03d}m"
        if step.resume_from != expected_parent:
            raise ValueError(f"{step.name} resume parent {step.resume_from} != expected {expected_parent}")

    def validate_train_receipt(self, step: PlanStep, preserved: dict):
        self.validate_train_step(step)
        parent = verify_local_preservation(self.local_root, step.resume_from, require_final_zip=True)
        result = preserved["result"]
        manifest = preserved["manifest"]
        summary = result.get("summary") or {}
        rollouts = result.get("rollouts") or []
        expected_added = step.rollouts * TRANSITIONS_PER_ROLLOUT
        expected_final = expected_counter_for_stage(step.stage)
        expected_start = expected_counter_for_stage(previous_stage(step.stage))
        if len(rollouts) != step.rollouts:
            raise ValueError(f"{step.name} rollout count {len(rollouts)} != expected {step.rollouts}")
        if summary.get("measured_transitions") != expected_added:
            raise ValueError(f"{step.name} measured_transitions != exact planned total")
        if summary.get("measured_game_ticks") not in (None, expected_added):
            raise ValueError(f"{step.name} measured_game_ticks != exact planned total")
        if summary.get("measured_rollouts") not in (None, step.rollouts):
            raise ValueError(f"{step.name} measured_rollouts != exact planned rollout count")
        for key in ("measured_training_steps_per_second", "measured_window_steps_per_second"):
            if key in summary and not finite_number(summary[key]):
                raise ValueError(f"{step.name} summary metric {key} is not finite")
        if (result.get("final_model_counters") or {}).get("num_timesteps") != expected_final:
            raise ValueError(f"{step.name} final counter does not match stage target")
        if result.get("resume_from_checkpoint_sha256") != parent["files"]["final.zip"]["sha256"]:
            raise ValueError(f"{step.name} resumeSHA does not match parent actual checkpoint")
        if result.get("policy_parameter_digest_before") != parent["result"].get("policy_parameter_digest_after"):
            raise ValueError(f"{step.name} policybefore does not match parent policyafter")
        config = manifest.get("config") or {}
        expected_config = {
            "action_ab_arm": step.arm,
            "seed": step.seed,
            "rollouts": step.rollouts,
            "lanes": LANES,
            "n_steps": N_STEPS,
            "episodes_per_lane": 640,
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise ValueError(f"{step.name} manifest config {key}={config.get(key)} != expected {expected}")
        checkpoint = manifest.get("checkpoint") or {}
        if checkpoint.get("sha256") != parent["files"]["final.zip"]["sha256"]:
            raise ValueError(f"{step.name} manifest checkpoint sha256 does not match parent actual checkpoint")
        for index, row in enumerate(rollouts):
            if row.get("transitions") != TRANSITIONS_PER_ROLLOUT:
                raise ValueError(f"{step.name} rollout {index} transitions != {TRANSITIONS_PER_ROLLOUT}")
            if row.get("game_ticks") not in (None, TRANSITIONS_PER_ROLLOUT):
                raise ValueError(f"{step.name} rollout {index} game_ticks != {TRANSITIONS_PER_ROLLOUT}")
            expected_counter = expected_start + (index + 1) * TRANSITIONS_PER_ROLLOUT
            if row.get("cumulative_timesteps") != expected_counter:
                raise ValueError(f"{step.name} rollout {index} cumulative counter != expected")
            actual_update = row.get("actual_update") or {}
            if actual_update.get("optimizer_steps") != OPTIMIZER_STEPS_PER_ROLLOUT:
                raise ValueError(f"{step.name} rollout {index} optimizer_steps != {OPTIMIZER_STEPS_PER_ROLLOUT}")
            if actual_update.get("nominal_optimizer_steps") != OPTIMIZER_STEPS_PER_ROLLOUT:
                raise ValueError(f"{step.name} rollout {index} nominal_optimizer_steps != {OPTIMIZER_STEPS_PER_ROLLOUT}")
            if actual_update.get("kl_early_stop") not in (False, None):
                raise ValueError(f"{step.name} rollout {index} KL early-stopped")
            if actual_update.get("low_update") not in (False, None):
                raise ValueError(f"{step.name} rollout {index} low_update is true")
            if not finite_number(actual_update.get("optimizer_step_fraction")):
                raise ValueError(f"{step.name} rollout {index} optimizer_step_fraction is not finite")
            for metric, value in (row.get("metrics") or {}).items():
                if not finite_number(value):
                    raise ValueError(f"{step.name} rollout {index} metric {metric} is not finite")
        return True

    def validate_eval_receipt(self, step: PlanStep, preserved: dict):
        train_name = step.name[len("eval-"):]
        trained = verify_local_preservation(self.local_root, train_name, require_final_zip=True)
        candidate = (preserved["manifest"].get("candidate") or {}).get("checkpoint_sha256")
        actual = trained["files"]["final.zip"]["sha256"]
        if not candidate:
            raise ValueError(f"{step.name} candidate checkpoint_sha256 is missing")
        if candidate != actual:
            raise ValueError(f"{step.name} candidate checkpoint_sha256 does not match {train_name}")
        return True

    def wait_for_job_rc0(self, name):
        started = self.now()
        while True:
            self.validate_deadline()
            receipt = self.remote.job_receipt(name)
            if receipt is not None:
                if receipt.get("returncode") != 0:
                    raise RuntimeError(f"{name} finished with returncode {receipt.get('returncode')}")
                self.emit(event="job_finished", name=name, returncode=0)
                return receipt
            if self.max_wait_seconds is not None and self.now() - started >= self.max_wait_seconds:
                raise TimeoutError(f"timed out waiting for job receipt {name}")
            self.emit(event="waiting_job", name=name)
            self.sleeper(self.poll_seconds)

    def ensure_job_started_or_finished(self, step: PlanStep, command):
        receipt = self.remote.job_receipt(step.name)
        if receipt is not None:
            if receipt.get("returncode") != 0:
                raise RuntimeError(f"{step.name} prior receipt returncode is {receipt.get('returncode')}")
            self.emit(event="skip_completed_job", name=step.name)
            return receipt
        started = self.remote.job_started(step.name)
        if started is not None:
            self.emit(event="job_already_started", name=step.name, pid=started.get("pid"))
            return self.wait_for_job_rc0(step.name)
        self.emit(event="launch_job", name=step.name, argv=command)
        self.remote.launch_job(step.name, command)
        return self.wait_for_job_rc0(step.name)

    def train_command(self, step: PlanStep):
        self.validate_train_step(step)
        if not step.resume_from:
            raise ValueError("continuation blocks must resume from a preserved prior block")
        parent = self.preservation_once(step.resume_from, require_final_zip=True)["local"]
        checkpoint = REMOTE_ARTIFACTS / step.resume_from / "final.zip"
        ack = REMOTE_ARTIFACTS / step.resume_from / "preservation_ack.json"
        return [
            "python3",
            "-B",
            "train_contact_campaign_block.py",
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-sha256",
            parent["files"]["final.zip"]["sha256"],
            "--checkpoint-ack",
            str(ack),
            "--frozen-root",
            str(FROZEN_ROOT),
            "--reference-root",
            str(REFERENCE_ROOT),
            "--qualification",
            str(QUALIFICATION),
            "--output",
            str(REMOTE_ARTIFACTS / step.name),
            "--deadline-unix",
            str(self.deadline_unix),
            "--lanes",
            "4096",
            "--rollouts",
            str(step.rollouts),
            "--seed",
            str(step.seed),
            "--learning-rate",
            "3e-5",
            "--action-ab-arm",
            step.arm,
            "--episodes-per-lane",
            "640",
            "--resume",
        ]

    def eval_command(self, step: PlanStep):
        train_name = step.name[len("eval-"):]
        candidate = verify_local_preservation(self.local_root, train_name, require_final_zip=True)
        return [
            "python3",
            "-B",
            "evaluate_native_contact_training.py",
            "--checkpoint",
            str(REMOTE_ARTIFACTS / train_name / "final.zip"),
            "--navigation-parent",
            str(NAVIGATION_PARENT),
            "--verified-ack",
            str(REMOTE_ARTIFACTS / train_name / "preservation_ack.json"),
            "--frozen-root",
            str(FROZEN_ROOT),
            "--reference-root",
            str(REFERENCE_ROOT),
            "--qualification",
            str(QUALIFICATION),
            "--output",
            str(REMOTE_ARTIFACTS / step.name),
            "--deadline-unix",
            str(self.deadline_unix),
            "--episodes",
            "20",
        ]

    def check_eval_gate(self, eval_name, *, arm):
        preserved = verify_local_preservation(self.local_root, eval_name, require_final_zip=False)
        if eval_name.startswith("eval-"):
            self.validate_eval_receipt(PlanStep(kind="eval", name=eval_name, arm=arm, stage=0), preserved)
        result = preserved["result"]
        retention = result.get("nearby_retention") or {}
        pause = retention.get("pause_updates")
        if pause is not False:
            self.stopped_arms.add(arm)
            self.emit(event="arm_stopped_retention_alarm", name=eval_name, arm=arm, pause_updates=pause)
            return False
        self.emit(event="eval_gate_clear", name=eval_name, arm=arm)
        return True

    def run(self, plan=None):
        completed = []
        for step in plan or build_plan():
            if step.arm in self.stopped_arms:
                self.emit(event="skip_stopped_arm", name=step.name, arm=step.arm)
                continue
            self.validate_deadline()
            if step.kind == "train":
                try:
                    local = self.local_preservation_once(step.name, require_final_zip=True)
                    self.wait_for_preservation(step.name, require_final_zip=True)
                    self.wait_for_job_rc0(step.name)
                    self.validate_train_receipt(step, local)
                    self.emit(event="skip_completed_local_train", name=step.name)
                    completed.append(asdict(step))
                    continue
                except (FileNotFoundError, KeyError, json.JSONDecodeError):
                    pass
                command = self.train_command(step)
                self.ensure_job_started_or_finished(step, command)
                preserved = self.wait_for_preservation(step.name, require_final_zip=True)["local"]
                self.validate_train_receipt(step, preserved)
            elif step.kind == "eval":
                train_name = step.name[len("eval-"):]
                self.wait_for_preservation(train_name, require_final_zip=True)
                try:
                    local = self.local_preservation_once(step.name, require_final_zip=False)
                    self.wait_for_preservation(step.name, require_final_zip=False)
                    self.wait_for_job_rc0(step.name)
                    self.validate_eval_receipt(step, local)
                    self.emit(event="skip_completed_local_eval", name=step.name)
                    self.check_eval_gate(step.name, arm=step.arm)
                    completed.append(asdict(step))
                    continue
                except (FileNotFoundError, KeyError, json.JSONDecodeError):
                    pass
                command = self.eval_command(step)
                self.ensure_job_started_or_finished(step, command)
                preserved = self.wait_for_preservation(step.name, require_final_zip=False)["local"]
                self.validate_eval_receipt(step, preserved)
                self.check_eval_gate(step.name, arm=step.arm)
            else:
                raise ValueError(f"unknown step kind: {step.kind}")
            completed.append(asdict(step))
        return {"status": "complete", "completed": completed, "stopped_arms": sorted(self.stopped_arms)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT)
    parser.add_argument("--deadline-unix", type=float, default=DEADLINE_UNIX)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    event_log = args.event_log or (args.local_root / "contact-continuation-events.jsonl")
    receipt = args.receipt or (args.local_root / "contact-continuation-receipt.json")
    if Path(event_log).exists() or Path(receipt).exists():
        raise FileExistsError("controller event log and receipt are immutable")
    controller = Controller(
        local_root=args.local_root,
        deadline_unix=args.deadline_unix,
        event_log=event_log,
        poll_seconds=args.poll_seconds,
    )
    started = time.time()
    result = controller.run()
    result.update(
        schema_version="ai2.contact_campaign_continuation.v1",
        started_unix=started,
        finished_unix=time.time(),
        event_log=str(event_log),
    )
    atomic_json(receipt, result)
    print(json.dumps({"status": result["status"], "stopped_arms": result["stopped_arms"]}, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
