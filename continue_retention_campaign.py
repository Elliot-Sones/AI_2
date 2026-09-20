#!/usr/bin/env python3
"""Preservation-gated controller for native retention curriculum campaigns.

This module plans and gates immutable retention blocks.  It deliberately keeps
provider mutation outside the controller: remote execution is injectable and
uses the same preservation ACK contract as the contact campaign controller.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

import continue_contact_campaign as contact


B40_SHA256 = "6d7b83a287b1a29bf8741295d769178764fd8b4beaf101a678b62cdc41bd73f0"
B40_BLOCK_NAME = "B_projected-040m"
REMOTE_ROOT = Path("/workspace/AI_2/staged-20260920")
REMOTE_ARTIFACTS = REMOTE_ROOT / "artifacts"
LOCAL_ROOT = Path("/Users/elliot18/Projects/ml-research/AI_2/baselines/native-retention-20260920/campaign")
CONTACT_QUALIFICATION_BLOCK = "contact-qualification-current"
QUALIFICATION = REMOTE_ARTIFACTS / CONTACT_QUALIFICATION_BLOCK / "qualification.json"
CONTACT_QUALIFICATION_CHECKPOINT = REMOTE_ROOT / "inputs/calibration/final.zip"
FROZEN_ROOT = contact.FROZEN_ROOT
REFERENCE_ROOT = contact.REFERENCE_ROOT
DEADLINE_UNIX = contact.DEADLINE_UNIX
RESERVE_SECONDS = contact.RESERVE_SECONDS
COST_RESERVE_DOLLARS = 0.75
LANES = 4096
N_STEPS = 512
ROLLOUTS_PER_CYCLE = 5
TRANSITIONS_PER_ROLLOUT = LANES * N_STEPS
CYCLE_TRANSITIONS = ROLLOUTS_PER_CYCLE * TRANSITIONS_PER_ROLLOUT
TARGET_ADDED_TRANSITIONS = 5 * CYCLE_TRANSITIONS
INITIAL_RECIPES = ("R4C1N", "R3C2N")
INITIAL_SEEDS = {"R4C1N": 404101, "R3C2N": 302102}
ACTION_AB_ARM = "B_projected"
SCHEDULES = {
    "R4C1N": ("combat", "combat", "combat", "combat", "navigation"),
    "R3C2N": ("combat", "combat", "combat", "navigation", "navigation"),
}
B40_BASELINE = {
    "schema_version": "ai2.retention_evaluation.v1",
    "status": "evaluated_not_promoted",
    "panels": {
        "contact": {
            "deterministic": {"episodes": 20, "successes": 19, "self_losses": 0, "opponent_self_losses": 0, "timeouts": 1, "ambiguous": 0},
            "sampled": {"episodes": 20, "successes": 14, "self_losses": 0, "opponent_self_losses": 0, "timeouts": 0, "ambiguous": 2},
        },
        "nearby": {
            "deterministic": {"episodes": 20, "successes": 20, "falls": 0},
            "sampled": {"episodes": 20, "successes": 20, "falls": 0},
        },
        "wide": {
            "deterministic": {"episodes": 20, "successes": 13, "falls": 7},
            "sampled": {"episodes": 20, "successes": 13, "falls": 7},
        },
    },
}
B40_BASELINE_PROVENANCE = {
    "source": "preserved B_projected B40 original-game development panels",
    "checkpoint_sha256": B40_SHA256,
    "regenerated_in_this_controller": False,
    "counts": "contact det/sample 19/14 with det timeout=1 and sampled ambiguous=2; nearby det/sample 20/20 with 0 falls; wide det/sample 13/13 with 7 falls",
}

sha256 = contact.sha256
bytes_sha256 = contact.bytes_sha256
read_json = contact.read_json
atomic_json = contact.atomic_json
verify_ack_for_bytes = contact.verify_ack_for_bytes
verify_local_preservation = contact.verify_local_preservation


def build_bounded_background_launch_shell(
    *, remote_root: Path, remote_python: Path, deadline_unix: float, name: str, command: list[str]
) -> str:
    """Launch one immutable job through the current campaign deadline wrapper."""
    contact.validate_name(name)
    remote_root = Path(remote_root)
    runner = remote_root / "source" / "run_campaign_job.py"
    argv = [
        str(remote_python),
        str(runner),
        "--root",
        str(remote_root),
        "--deadline-unix",
        str(float(deadline_unix)),
        name,
        "--",
        *command,
    ]
    quoted = " ".join(shlex.quote(str(part)) for part in argv)
    log_path = remote_root / "jobs" / f"{name}.controller-launch.log"
    return (
        f"cd {shlex.quote(str(remote_root))} "
        f"&& mkdir -p {shlex.quote(str(remote_root / 'jobs'))} "
        f"&& {{ nohup {quoted} > {shlex.quote(str(log_path))} "
        f"2>&1 < /dev/null & echo $!; }}"
    )


class RemoteClient(contact.RemoteClient):
    def __init__(self, *, remote_artifacts=REMOTE_ARTIFACTS, remote_python=None,
                 deadline_unix=DEADLINE_UNIX, **kwargs):
        super().__init__(**kwargs)
        self.remote_artifacts = Path(remote_artifacts)
        self.python_executable = Path(remote_python) if remote_python is not None else self.remote_root / "venv/bin/python"
        self.runner_deadline_unix = float(deadline_unix)

    def launch_job(self, name, command):
        contact.validate_name(name)
        launch = build_bounded_background_launch_shell(
            remote_root=self.remote_root,
            remote_python=self.python_executable,
            deadline_unix=self.runner_deadline_unix,
            name=name,
            command=command,
        )
        try:
            result = self.run_shell(launch, timeout=30)
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

    def _artifact_path(self, name, filename):
        contact.validate_name(name)
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe artifact path: {filename!r}")
        return self.remote_artifacts / name / relative

    def read_artifact_file(self, name, filename):
        data = self.remote_file(self._artifact_path(name, filename))
        if data is None:
            raise FileNotFoundError(f"remote artifact missing: {name}/{filename}")
        return data

    def write_artifact_file(self, name, filename, data):
        path = self._artifact_path(name, filename)
        existing = self.remote_file(path)
        if existing is not None:
            if existing != data:
                raise FileExistsError(f"remote artifact already exists with different bytes: {name}/{filename}")
            return {"path": str(path), "bytes": len(data), "sha256": bytes_sha256(data), "already_exists": True}
        payload = base64.b64encode(data).decode("ascii")
        tmp = str(path) + ".tmp"
        command = (
            f"mkdir -p {shlex.quote(str(path.parent))} && "
            f"printf %s {shlex.quote(payload)} | base64 -d > {shlex.quote(tmp)} && "
            f"test ! -e {shlex.quote(str(path))} && "
            f"mv {shlex.quote(tmp)} {shlex.quote(str(path))}"
        )
        result = self.run_shell(command, timeout=60)
        if result.returncode:
            raise RuntimeError(f"remote artifact upload failed for {name}/{filename}: {result.stderr.decode(errors='replace')}")
        return {"path": str(path), "bytes": len(data), "sha256": bytes_sha256(data)}

    def read_job_file(self, filename):
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError(f"unsafe remote job filename: {filename!r}")
        return self.remote_file(self.remote_root / "jobs" / relative)


@dataclass(frozen=True)
class PlanStep:
    kind: str
    name: str
    recipe: str
    seed: int
    cumulative_added_transitions: int
    parent_sha256: str
    resume_from: str | None = None
    rollouts: int = ROLLOUTS_PER_CYCLE
    cycle: int = 1
    source_digest: str = B40_SHA256
    result_sha256: str | None = None


@dataclass(frozen=True)
class GateResult:
    eligible: bool
    reason: str
    components: dict
    raw: dict


@dataclass(frozen=True)
class ParetoResult:
    winner: str | None
    relation: str
    matched_comparisons: tuple[PlanStep, ...] = ()


@dataclass(frozen=True)
class ContinuationDecision:
    improved: bool
    consecutive_no_improvement: int
    stop: bool
    reason: str


@dataclass(frozen=True)
class ElapsedBudgetMeter:
    initial_remaining_dollars: float
    hourly_cost_dollars: float
    started_unix: float
    now: object = time.time

    def remaining(self):
        elapsed_hours = max(0.0, (float(self.now()) - float(self.started_unix)) / 3600.0)
        return float(self.initial_remaining_dollars) - elapsed_hours * float(self.hourly_cost_dollars)


def _source_prefix(source: str) -> str:
    if not isinstance(source, str) or len(source) < 12:
        raise ValueError("source digest must be a sha256-like string")
    return source[:12]


def _step_name(recipe: str, cumulative_added: int, seed: int, source_digest: str, *, cycle: int = 1) -> str:
    return f"{recipe}-cycle{cycle:02d}-add{cumulative_added}-s{seed}-src{_source_prefix(source_digest)}"


def build_initial_plan(parent_sha256=B40_SHA256, source_digest=None):
    source_digest = source_digest or parent_sha256
    steps = []
    for recipe in INITIAL_RECIPES:
        seed = INITIAL_SEEDS[recipe]
        steps.append(
            PlanStep(
                kind="train",
                name=_step_name(recipe, CYCLE_TRANSITIONS, seed, source_digest, cycle=1),
                recipe=recipe,
                seed=seed,
                cumulative_added_transitions=CYCLE_TRANSITIONS,
                parent_sha256=parent_sha256,
                source_digest=source_digest,
                cycle=1,
            )
        )
    return steps


def eval_step(train_step: PlanStep) -> PlanStep:
    return PlanStep(
        kind="eval",
        name=f"eval-{train_step.name}",
        recipe=train_step.recipe,
        seed=train_step.seed,
        cumulative_added_transitions=train_step.cumulative_added_transitions,
        parent_sha256=train_step.parent_sha256,
        resume_from=train_step.name,
        rollouts=0,
        cycle=train_step.cycle,
        source_digest=train_step.source_digest,
        result_sha256=train_step.result_sha256,
    )


def build_continuation_step(winner, cycle):
    if not isinstance(cycle, int) or cycle < 2:
        raise ValueError("continuation cycle must be an integer >= 2")
    parent_sha = getattr(winner, "result_sha256", None)
    if not parent_sha:
        raise ValueError("winner must carry result_sha256 before continuation")
    cumulative = cycle * CYCLE_TRANSITIONS
    if cumulative > TARGET_ADDED_TRANSITIONS:
        raise ValueError("continuation would exceed declared target transitions")
    seed = int(getattr(winner, "seed")) + cycle
    source_digest = getattr(winner, "source_digest", B40_SHA256)
    recipe = getattr(winner, "recipe")
    return PlanStep(
        kind="train",
        name=_step_name(recipe, cumulative, seed, source_digest, cycle=cycle),
        recipe=recipe,
        seed=seed,
        cumulative_added_transitions=cumulative,
        parent_sha256=parent_sha,
        resume_from=getattr(winner, "name"),
        rollouts=ROLLOUTS_PER_CYCLE,
        cycle=cycle,
        source_digest=source_digest,
    )


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _panel_mode(result, panel, mode):
    try:
        row = result["panels"][panel][mode]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"result must include {panel}/{mode} panel") from exc
    episodes = row.get("episodes")
    if episodes != 20:
        raise ValueError(f"{panel}/{mode} must contain exactly 20 episodes")
    return row


def _component_values(result):
    contact_det = _panel_mode(result, "contact", "deterministic")
    contact_sampled = _panel_mode(result, "contact", "sampled")
    nearby_det = _panel_mode(result, "nearby", "deterministic")
    nearby_sampled = _panel_mode(result, "nearby", "sampled")
    wide_det = _panel_mode(result, "wide", "deterministic")
    wide_sampled = _panel_mode(result, "wide", "sampled")
    values = {
        "contact_deterministic": contact_det.get("successes"),
        "contact_sampled": contact_sampled.get("successes"),
        "nearby_deterministic": nearby_det.get("successes"),
        "nearby_sampled": nearby_sampled.get("successes"),
        "wide_deterministic": wide_det.get("successes"),
        "wide_sampled": wide_sampled.get("successes"),
        "nearby_deterministic_falls": nearby_det.get("falls", 0),
        "nearby_sampled_falls": nearby_sampled.get("falls", 0),
        "wide_deterministic_falls": wide_det.get("falls", 0),
        "wide_sampled_falls": wide_sampled.get("falls", 0),
        "contact_deterministic_self_losses": contact_det.get("self_losses", 0),
        "contact_deterministic_opponent_self_losses": contact_det.get("opponent_self_losses", 0),
        "contact_deterministic_timeouts": contact_det.get("timeouts", 0),
        "contact_deterministic_ambiguous": contact_det.get("ambiguous", 0),
        "contact_sampled_self_losses": contact_sampled.get("self_losses", 0),
        "contact_sampled_opponent_self_losses": contact_sampled.get("opponent_self_losses", 0),
        "contact_sampled_timeouts": contact_sampled.get("timeouts", 0),
        "contact_sampled_ambiguous": contact_sampled.get("ambiguous", 0),
    }
    if not all(_finite(value) for value in values.values()):
        raise ValueError("diagnostic components must be finite numbers")
    return values


def diagnostic_gate(result, b40_baseline):
    if "score" in result:
        raise ValueError("aggregate score is not permitted for retention gating")
    current = _component_values(result)
    baseline = _component_values(b40_baseline)
    components = {
        "contact_deterministic": {
            "candidate": current["contact_deterministic"],
            "baseline": baseline["contact_deterministic"],
            "passed": current["contact_deterministic"] >= 18,
        },
        "contact_sampled": {
            "candidate": current["contact_sampled"],
            "baseline": baseline["contact_sampled"],
            "passed": current["contact_sampled"] >= max(14, baseline["contact_sampled"]),
        },
        "nearby_deterministic": {
            "candidate": current["nearby_deterministic"],
            "baseline": baseline["nearby_deterministic"],
            "passed": current["nearby_deterministic"] >= 19 and current["nearby_deterministic_falls"] <= 1,
        },
        "nearby_sampled": {
            "candidate": current["nearby_sampled"],
            "baseline": baseline["nearby_sampled"],
            "passed": current["nearby_sampled"] >= 19 and current["nearby_sampled_falls"] <= 1,
        },
        "wide_deterministic": {
            "candidate": current["wide_deterministic"],
            "baseline": baseline["wide_deterministic"],
            "passed": current["wide_deterministic"] >= 13,
        },
        "wide_sampled": {
            "candidate": current["wide_sampled"],
            "baseline": baseline["wide_sampled"],
            "passed": current["wide_sampled"] >= 13,
        },
        "wide_deterministic_falls": {
            "candidate": current["wide_deterministic_falls"],
            "baseline": baseline["wide_deterministic_falls"],
            "passed": current["wide_deterministic_falls"] <= baseline["wide_deterministic_falls"],
        },
        "wide_sampled_falls": {
            "candidate": current["wide_sampled_falls"],
            "baseline": baseline["wide_sampled_falls"],
            "passed": current["wide_sampled_falls"] <= baseline["wide_sampled_falls"],
        },
        "contact_deterministic_self_losses": {
            "candidate": current["contact_deterministic_self_losses"],
            "baseline": baseline["contact_deterministic_self_losses"],
            "passed": current["contact_deterministic_self_losses"] == 0,
        },
        "contact_deterministic_opponent_self_losses": {
            "candidate": current["contact_deterministic_opponent_self_losses"],
            "baseline": baseline["contact_deterministic_opponent_self_losses"],
            "passed": current["contact_deterministic_opponent_self_losses"] <= baseline["contact_deterministic_opponent_self_losses"],
        },
        "contact_deterministic_timeouts": {
            "candidate": current["contact_deterministic_timeouts"],
            "baseline": baseline["contact_deterministic_timeouts"],
            "passed": current["contact_deterministic_timeouts"] <= baseline["contact_deterministic_timeouts"],
        },
        "contact_deterministic_ambiguous": {
            "candidate": current["contact_deterministic_ambiguous"],
            "baseline": baseline["contact_deterministic_ambiguous"],
            "passed": current["contact_deterministic_ambiguous"] <= baseline["contact_deterministic_ambiguous"],
        },
        "contact_sampled_self_losses": {
            "candidate": current["contact_sampled_self_losses"],
            "baseline": baseline["contact_sampled_self_losses"],
            "passed": current["contact_sampled_self_losses"] == 0,
        },
        "contact_sampled_opponent_self_losses": {
            "candidate": current["contact_sampled_opponent_self_losses"],
            "baseline": baseline["contact_sampled_opponent_self_losses"],
            "passed": current["contact_sampled_opponent_self_losses"] <= baseline["contact_sampled_opponent_self_losses"],
        },
        "contact_sampled_timeouts": {
            "candidate": current["contact_sampled_timeouts"],
            "baseline": baseline["contact_sampled_timeouts"],
            "passed": current["contact_sampled_timeouts"] <= baseline["contact_sampled_timeouts"],
        },
        "contact_sampled_ambiguous": {
            "candidate": current["contact_sampled_ambiguous"],
            "baseline": baseline["contact_sampled_ambiguous"],
            "passed": current["contact_sampled_ambiguous"] <= baseline["contact_sampled_ambiguous"],
        },
    }
    failed = [name for name, row in components.items() if not row["passed"]]
    return GateResult(
        eligible=not failed,
        reason="eligible" if not failed else "failed components: " + ", ".join(failed),
        components=components,
        raw=dict(result),
    )


def _pareto_vectors(left, right):
    left_values = _component_values(left)
    right_values = _component_values(right)
    keys = tuple(left_values)
    left_better = []
    right_better = []
    for key in keys:
        lv = left_values[key]
        rv = right_values[key]
        if _lower_is_better(key):
            if lv < rv:
                left_better.append(key)
            elif rv < lv:
                right_better.append(key)
        else:
            if lv > rv:
                left_better.append(key)
            elif rv > lv:
                right_better.append(key)
    return left_better, right_better


def _lower_is_better(key):
    return key.endswith("_falls") or key.endswith("_self_losses") or key.endswith("_timeouts") or key.endswith("_ambiguous")


def pareto_relation(left, right):
    left_better, right_better = _pareto_vectors(left, right)
    if left_better and not right_better:
        return ParetoResult(winner="left", relation="left_dominates")
    if right_better and not left_better:
        return ParetoResult(winner="right", relation="right_dominates")
    if not left_better and not right_better:
        return ParetoResult(winner=None, relation="tie")
    return ParetoResult(winner=None, relation="tradeoff")


def improved(previous, current):
    left_better, right_better = _pareto_vectors(current, previous)
    return bool(left_better and not right_better)


def _safe_artifact_relative(relative):
    if not isinstance(relative, str):
        raise ValueError("remote inventory contains invalid path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe artifact path: {relative!r}")
    return path


class ContinuationState:
    def __init__(self):
        self.consecutive_no_improvement = 0

    def update(self, previous, current):
        did_improve = improved(previous, current)
        self.consecutive_no_improvement = 0 if did_improve else self.consecutive_no_improvement + 1
        return ContinuationDecision(
            improved=did_improve,
            consecutive_no_improvement=self.consecutive_no_improvement,
            stop=self.consecutive_no_improvement >= 2,
            reason="improved" if did_improve else "no improvement",
        )


class Controller(contact.Controller):
    def __init__(self, *, remaining_cost=lambda: float("inf"), cost_reserve=COST_RESERVE_DOLLARS,
                 qualification=None, frozen_root=FROZEN_ROOT, reference_root=REFERENCE_ROOT,
                 contact_qualification_checkpoint=CONTACT_QUALIFICATION_CHECKPOINT,
                 contact_qualification_block=CONTACT_QUALIFICATION_BLOCK,
                 plan_source_digest=None,
                 initial_checkpoint=None,
                 initial_checkpoint_ack=None,
                 remote_artifacts=REMOTE_ARTIFACTS,
                 preserve_after_job=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.remaining_cost = remaining_cost
        self.cost_reserve = float(cost_reserve)
        self.stopped_lineages = set()
        self.frozen_root = Path(frozen_root)
        self.reference_root = Path(reference_root)
        self.contact_qualification_checkpoint = Path(contact_qualification_checkpoint)
        contact.validate_name(contact_qualification_block)
        self.contact_qualification_block = contact_qualification_block
        self.plan_source_digest = plan_source_digest or B40_SHA256
        _source_prefix(self.plan_source_digest)
        self.remote_artifacts = Path(remote_artifacts)
        if qualification is None or Path(qualification) == QUALIFICATION:
            self.qualification = self.remote_artifacts / self.contact_qualification_block / "qualification.json"
        else:
            self.qualification = Path(qualification)
        self.initial_checkpoint = Path(initial_checkpoint) if initial_checkpoint is not None else self.remote_artifacts / B40_BLOCK_NAME / "final.zip"
        self.initial_checkpoint_ack = Path(initial_checkpoint_ack) if initial_checkpoint_ack is not None else self.remote_artifacts / B40_BLOCK_NAME / "preservation_ack.json"
        self.preserve_after_job_callback = preserve_after_job

    def validate_budget(self):
        if self.now() > self.deadline_unix - RESERVE_SECONDS:
            raise TimeoutError("deadline reserve exhausted")
        remaining = float(self.remaining_cost())
        if remaining < self.cost_reserve:
            raise TimeoutError("remaining-cost reserve exhausted")

    def validate_deadline(self):
        self.validate_budget()

    def retrieve_and_preserve_remote_block(self, name, *, require_final_zip):
        inventory_bytes = self.remote.read_artifact_file(name, "inventory.json")
        inventory = json.loads(inventory_bytes)
        if inventory.get("schema_version") != "curriculum-inventory-v1" or inventory.get("block_id") != name:
            raise ValueError(f"{name} remote inventory is invalid")
        block = self.local_root / name
        block.mkdir(parents=True, exist_ok=True)
        files = {}
        for row in inventory.get("files", []):
            relative = row.get("path")
            _safe_artifact_relative(relative)
            files[relative] = row
        required = {"result.json", "manifest.json"}
        if require_final_zip:
            required.add("final.zip")
        missing = sorted(required - set(files))
        if missing:
            raise ValueError(f"{name} remote inventory missing required artifacts: {missing}")
        for relative, row in files.items():
            destination = block / _safe_artifact_relative(relative)
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise ValueError(f"{name} unsafe local artifact path: {relative}")
                if destination.stat().st_size == row.get("bytes") and sha256(destination) == row.get("sha256"):
                    continue
                raise ValueError(f"{name} local artifact differs from remote inventory: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = self.remote.read_artifact_file(name, relative)
            if len(data) != row.get("bytes") or bytes_sha256(data) != row.get("sha256"):
                raise ValueError(f"{name} remote artifact hash mismatch: {relative}")
            destination.write_bytes(data)
        inventory_path = block / "inventory.json"
        if not inventory_path.exists():
            inventory_path.write_bytes(inventory_bytes)
        elif inventory_path.read_bytes() != inventory_bytes:
            raise ValueError(f"{name} local inventory differs from remote inventory")
        ack = {"verified": True, "block_id": name, "inventory_sha256": bytes_sha256(inventory_bytes)}
        ack_path = block / "preservation_ack.json"
        if not ack_path.exists():
            atomic_json(ack_path, ack)
        elif read_json(ack_path) != ack:
            raise ValueError(f"{name} local preservation ACK differs from expected ACK")
        if hasattr(self.remote, "write_artifact_file"):
            try:
                self.remote.write_artifact_file(name, "preservation_ack.json", ack_path.read_bytes())
            except FileExistsError:
                pass
        return self.preservation_once(name, require_final_zip=require_final_zip)

    def preserve_after_job(self, name, *, require_final_zip):
        if self.preserve_after_job_callback is not None:
            self.preserve_after_job_callback(name, require_final_zip=require_final_zip)
            return self.preservation_once(name, require_final_zip=require_final_zip)
        return self.retrieve_and_preserve_remote_block(name, require_final_zip=require_final_zip)

    def _preserve_failure_inventory_block(self, name):
        contact.validate_name(name)
        inventory_bytes = self.remote.read_artifact_file(name, "inventory.json")
        inventory = json.loads(inventory_bytes)
        if inventory.get("schema_version") != "curriculum-inventory-v1" or inventory.get("block_id") != name:
            raise ValueError(f"{name} remote failure inventory is invalid")
        block = self.local_root / name
        block.mkdir(parents=True, exist_ok=True)
        copied = []
        for row in inventory.get("files", []):
            relative = row.get("path")
            destination = block / _safe_artifact_relative(relative)
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise ValueError(f"{name} unsafe local failure artifact path: {relative}")
                if destination.stat().st_size == row.get("bytes") and sha256(destination) == row.get("sha256"):
                    copied.append(relative)
                    continue
                raise ValueError(f"{name} local failure artifact differs from remote inventory: {relative}")
            data = self.remote.read_artifact_file(name, relative)
            if len(data) != row.get("bytes") or bytes_sha256(data) != row.get("sha256"):
                raise ValueError(f"{name} remote failure artifact hash mismatch: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            copied.append(relative)
        inventory_path = block / "inventory.json"
        if not inventory_path.exists():
            inventory_path.write_bytes(inventory_bytes)
        elif inventory_path.read_bytes() != inventory_bytes:
            raise ValueError(f"{name} local failure inventory differs from remote inventory")
        receipt = {
            "schema_version": "ai2.remote_failure_preservation.v1",
            "block_id": name,
            "inventory_sha256": bytes_sha256(inventory_bytes),
            "files": sorted(copied),
        }
        receipt_path = block / "failure_preservation.json"
        if not receipt_path.exists():
            atomic_json(receipt_path, receipt)
        elif read_json(receipt_path) != receipt:
            raise ValueError(f"{name} local failure preservation receipt differs")
        return receipt

    def _preserve_failure_job_files(self, name):
        contact.validate_name(name)
        jobs_dir = self.local_root / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for suffix in ("started.json", "json", "log"):
            filename = f"{name}.{suffix}"
            data = None
            if hasattr(self.remote, "read_job_file"):
                data = self.remote.read_job_file(filename)
            elif hasattr(self.remote, "remote_file") and hasattr(self.remote, "remote_root"):
                data = self.remote.remote_file(Path(self.remote.remote_root) / "jobs" / filename)
            if data is None:
                continue
            destination = jobs_dir / filename
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise ValueError(f"{name} unsafe local job evidence path: {filename}")
                if destination.read_bytes() == data:
                    copied.append(filename)
                    continue
                raise ValueError(f"{name} local job evidence differs: {filename}")
            destination.write_bytes(data)
            copied.append(filename)
        return copied

    def preserve_failure_evidence(self, name):
        errors = []
        block = None
        jobs = []
        try:
            block = self._preserve_failure_inventory_block(name)
        except Exception as exc:  # Best effort: keep the original job failure as the raised error.
            errors.append(f"block: {type(exc).__name__}: {exc}")
        try:
            jobs = self._preserve_failure_job_files(name)
        except Exception as exc:
            errors.append(f"jobs: {type(exc).__name__}: {exc}")
        self.emit(event="failure_evidence_preserved", name=name, block=block is not None, jobs=jobs, errors=errors)
        return {"block": block, "jobs": jobs, "errors": errors}

    def wait_for_job_rc0(self, name):
        try:
            return super().wait_for_job_rc0(name)
        except (RuntimeError, TimeoutError) as error:
            self.preserve_failure_evidence(name)
            setattr(error, "_ai2_failure_evidence_preserved", True)
            raise

    def ensure_job_started_or_finished(self, step, command):
        try:
            return super().ensure_job_started_or_finished(step, command)
        except (RuntimeError, TimeoutError) as error:
            if not getattr(error, "_ai2_failure_evidence_preserved", False):
                self.preserve_failure_evidence(step.name)
            raise

    def verify_local_qualification_preservation(self):
        name = self.contact_qualification_block
        block = self.local_root / name
        inventory_path = block / "inventory.json"
        ack_path = block / "preservation_ack.json"
        if inventory_path.is_symlink() or ack_path.is_symlink():
            raise ValueError("contact qualification preservation metadata is symlinked")
        verified = verify_ack_for_bytes(inventory_path.read_bytes(), ack_path.read_bytes(), expected_name=name)
        files = {row.get("path"): row for row in verified["inventory"].get("files", [])}
        row = files.get("qualification.json")
        if row is None:
            raise ValueError("contact qualification inventory missing qualification.json")
        receipt_path = block / "qualification.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("contact qualification receipt is unsafe or missing")
        if row.get("bytes") != receipt_path.stat().st_size or row.get("sha256") != sha256(receipt_path):
            raise ValueError("contact qualification inventory mismatch for qualification.json")
        receipt = read_json(receipt_path)
        if receipt.get("schema_version") != "ai2.native_contact_qualification.v1" or receipt.get("qualified") is not True:
            raise ValueError("contact qualification receipt is not qualified")
        checks = receipt.get("checks") or {}
        for check in ("original_cpu_exact", "cpu_cuda_exact", "learner_update_exact"):
            if checks.get(check) is not True:
                raise ValueError(f"contact qualification missing passing check {check}")
        return {"name": name, "directory": block, "files": {"qualification.json": row}, "receipt": receipt, **verified}

    def retrieve_and_preserve_contact_qualification(self):
        name = self.contact_qualification_block
        inventory_bytes = self.remote.read_artifact_file(name, "inventory.json")
        inventory = json.loads(inventory_bytes)
        if inventory.get("schema_version") != "curriculum-inventory-v1" or inventory.get("block_id") != name:
            raise ValueError("contact qualification remote inventory is invalid")
        block = self.local_root / name
        block.mkdir(parents=True, exist_ok=True)
        files = {}
        for row in inventory.get("files", []):
            relative = row.get("path")
            _safe_artifact_relative(relative)
            files[relative] = row
        if "qualification.json" not in files:
            raise ValueError("contact qualification remote inventory missing qualification.json")
        for relative, row in files.items():
            destination = block / _safe_artifact_relative(relative)
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise ValueError(f"contact qualification unsafe local artifact path: {relative}")
                if destination.stat().st_size == row.get("bytes") and sha256(destination) == row.get("sha256"):
                    continue
                raise ValueError(f"contact qualification local artifact differs: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = self.remote.read_artifact_file(name, relative)
            if len(data) != row.get("bytes") or bytes_sha256(data) != row.get("sha256"):
                raise ValueError(f"contact qualification remote artifact hash mismatch: {relative}")
            destination.write_bytes(data)
        inventory_path = block / "inventory.json"
        if not inventory_path.exists():
            inventory_path.write_bytes(inventory_bytes)
        elif inventory_path.read_bytes() != inventory_bytes:
            raise ValueError("contact qualification local inventory differs from remote")
        ack = {"verified": True, "block_id": name, "inventory_sha256": bytes_sha256(inventory_bytes)}
        ack_path = block / "preservation_ack.json"
        if not ack_path.exists():
            atomic_json(ack_path, ack)
        elif read_json(ack_path) != ack:
            raise ValueError("contact qualification local ACK differs from expected")
        if hasattr(self.remote, "write_artifact_file"):
            try:
                self.remote.write_artifact_file(name, "preservation_ack.json", ack_path.read_bytes())
            except FileExistsError:
                pass
        return self.verify_local_qualification_preservation()

    def qualification_command(self):
        return [
            str(getattr(self.remote, "python_executable", "python3")),
            "-B",
            "qualify_native_contact_training.py",
            "--output-dir",
            str(self.remote_artifacts / self.contact_qualification_block),
            "--reference-root",
            str(self.reference_root),
            "--frozen-root",
            str(self.frozen_root),
            "--checkpoint",
            str(self.contact_qualification_checkpoint),
        ]

    def ensure_contact_qualification(self):
        try:
            local = self.verify_local_qualification_preservation()
            remote = self.verify_remote_preservation(self.contact_qualification_block)
            if remote["inventory_sha256"] != local["inventory_sha256"]:
                raise ValueError("contact qualification remote inventory does not match local")
            self.emit(event="skip_completed_contact_qualification", name=self.contact_qualification_block)
            return {"local": local, "remote": remote}
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass
        step = PlanStep(
            kind="qualification",
            name=self.contact_qualification_block,
            recipe="contact",
            seed=0,
            cumulative_added_transitions=0,
            parent_sha256="0" * 64,
        )
        self.ensure_job_started_or_finished(step, self.qualification_command())
        local = self.retrieve_and_preserve_contact_qualification()
        remote = self.verify_remote_preservation(self.contact_qualification_block)
        if remote["inventory_sha256"] != local["inventory_sha256"]:
            raise ValueError("contact qualification remote inventory does not match local")
        return {"local": local, "remote": remote}

    def validate_train_step(self, step: PlanStep):
        if step.kind != "train":
            raise ValueError("expected train step")
        if step.recipe not in INITIAL_RECIPES:
            raise ValueError(f"unsupported retention recipe {step.recipe!r}")
        if step.rollouts != ROLLOUTS_PER_CYCLE:
            raise ValueError("retention blocks must be immutable five-rollout cycles")
        expected = _step_name(step.recipe, step.cumulative_added_transitions, step.seed, step.source_digest, cycle=step.cycle)
        if step.name != expected:
            raise ValueError(f"{step.name} does not match immutable retention name {expected}")
        if step.cycle == 1:
            if step.parent_sha256 != B40_SHA256 or step.resume_from is not None:
                raise ValueError("initial candidates must start independently from exact B40")
        elif not step.resume_from:
            raise ValueError("continuation cycles must name a preserved parent block")

    def train_command(self, step: PlanStep):
        self.validate_train_step(step)
        command = [
            str(getattr(self.remote, "python_executable", "python3")),
            "-B",
            "train_retention_campaign_block.py",
            "--schedule",
            step.recipe,
            "--seed",
            str(step.seed),
            "--lanes",
            str(LANES),
            "--batch-size",
            str(16384),
            "--action-ab-arm",
            ACTION_AB_ARM,
            "--checkpoint",
            str(self.initial_checkpoint),
            "--checkpoint-sha256",
            B40_SHA256,
            "--checkpoint-ack",
            str(self.initial_checkpoint_ack),
            "--frozen-root",
            str(self.frozen_root),
            "--reference-root",
            str(self.reference_root),
            "--qualification",
            str(self.qualification),
            "--output",
            str(self.remote_artifacts / step.name),
            "--deadline-unix",
            str(self.deadline_unix),
            "--resume",
        ]
        if step.resume_from:
            parent = self.preservation_once(step.resume_from, require_final_zip=True)["local"]
            command[command.index("--checkpoint") + 1] = str(self.remote_artifacts / step.resume_from / "final.zip")
            command[command.index("--checkpoint-sha256") + 1] = parent["files"]["final.zip"]["sha256"]
            command[command.index("--checkpoint-ack") + 1] = str(self.remote_artifacts / step.resume_from / "preservation_ack.json")
        return command

    def eval_command(self, step: PlanStep):
        if step.kind != "eval" or not step.resume_from:
            raise ValueError("expected eval step with train parent")
        candidate = self.preservation_once(step.resume_from, require_final_zip=True)
        return [
            str(getattr(self.remote, "python_executable", "python3")),
            "-B",
            "evaluate_retention_campaign.py",
            "--checkpoint",
            str(self.remote_artifacts / step.resume_from / "final.zip"),
            "--checkpoint-sha256",
            candidate["local"]["files"]["final.zip"]["sha256"],
            "--verified-ack",
            str(self.remote_artifacts / step.resume_from / "preservation_ack.json"),
            "--frozen-root",
            str(self.frozen_root),
            "--reference-root",
            str(self.reference_root),
            "--qualification",
            str(self.qualification),
            "--output",
            str(self.remote_artifacts / step.name),
            "--deadline-unix",
            str(self.deadline_unix),
        ]

    def validate_train_receipt(self, step: PlanStep, preserved: dict):
        self.validate_train_step(step)
        result = preserved["result"]
        manifest = preserved["manifest"]
        expected_schedule = list(SCHEDULES[step.recipe])
        if result.get("status") != "trained_not_evaluated":
            raise ValueError(f"{step.name} did not finish as a training block")
        if result.get("schema_version") != "ai2.retention_campaign_block.v1":
            raise ValueError(f"{step.name} schema_version mismatch")
        if result.get("checkpoint_sha256") != preserved["files"]["final.zip"]["sha256"]:
            raise ValueError(f"{step.name} checkpoint sha mismatch")
        if result.get("schedule_id") != step.recipe or (manifest.get("config") or {}).get("schedule_id") != step.recipe:
            raise ValueError(f"{step.name} schedule mismatch")
        if result.get("schedule") != expected_schedule or manifest.get("schedule") != expected_schedule:
            raise ValueError(f"{step.name} schedule lineage mismatch")
        if (manifest.get("config") or {}).get("action_ab_arm") != ACTION_AB_ARM or result.get("action_ab_arm") not in (None, ACTION_AB_ARM):
            raise ValueError(f"{step.name} action_ab_arm contract mismatch")
        if (manifest.get("config") or {}).get("seed") != step.seed:
            raise ValueError(f"{step.name} seed mismatch")
        if (manifest.get("checkpoint") or {}).get("sha256") != step.parent_sha256:
            raise ValueError(f"{step.name} lineage parent checkpoint sha mismatch")
        if (result.get("summary") or {}).get("measured_transitions") != CYCLE_TRANSITIONS:
            raise ValueError(f"{step.name} measured transitions mismatch")
        rollouts = result.get("rollouts") or []
        if len(rollouts) != ROLLOUTS_PER_CYCLE:
            raise ValueError(f"{step.name} rollout count mismatch")
        for index, row in enumerate(rollouts):
            if row.get("task") != expected_schedule[index]:
                raise ValueError(f"{step.name} rollout {index} task mismatch")
            if row.get("transitions") != TRANSITIONS_PER_ROLLOUT or row.get("game_ticks") != TRANSITIONS_PER_ROLLOUT:
                raise ValueError(f"{step.name} rollout {index} transition/game tick mismatch")
            for key in ("collection_seconds", "update_seconds", "total_seconds"):
                if not _finite(row.get(key)):
                    raise ValueError(f"{step.name} rollout {index} non-finite {key}")
            actual = row.get("actual_update") or {}
            if actual.get("optimizer_steps") != 512 or actual.get("nominal_optimizer_steps") != 512:
                raise ValueError(f"{step.name} rollout {index} optimizer steps mismatch")
            if actual.get("kl_early_stop") is not False or actual.get("low_update") is not False:
                raise ValueError(f"{step.name} rollout {index} KL early stop or low update")
            for value in (row.get("metrics") or {}).values():
                if not _finite(value):
                    raise ValueError(f"{step.name} rollout {index} non-finite metric")
        return True

    def validate_eval_receipt(self, step: PlanStep, preserved: dict):
        if not step.resume_from:
            raise ValueError("eval step is missing train parent")
        trained = verify_local_preservation(self.local_root, step.resume_from, require_final_zip=True)
        candidate = (preserved["manifest"].get("candidate") or {}).get("checkpoint_sha256")
        if candidate != trained["files"]["final.zip"]["sha256"]:
            raise ValueError(f"{step.name} candidate checkpoint_sha256 does not match train block")
        return True

    def apply_eval_gate(self, train_steps, *, b40_baseline):
        outcomes = {}
        for step in train_steps:
            evaluation = eval_step(step)
            preserved = self.preservation_once(evaluation.name, require_final_zip=False)
            self.validate_eval_receipt(evaluation, preserved["local"])
            gate = diagnostic_gate(preserved["local"]["result"], b40_baseline)
            outcomes[step.name] = asdict(gate)
            if not gate.eligible:
                self.stopped_lineages.add(step.name)
                self.emit(event="lineage_stopped", name=step.name, reason=gate.reason)
        return {"outcomes": outcomes, "stopped_lineages": sorted(self.stopped_lineages)}

    def _run_steps(self, plan):
        completed = []
        for step in plan:
            self.validate_budget()
            if step.name in self.stopped_lineages:
                self.emit(event="skip_stopped_lineage", name=step.name)
                continue
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
                self.preserve_after_job(step.name, require_final_zip=True)
                preserved = self.wait_for_preservation(step.name, require_final_zip=True)["local"]
                self.validate_train_receipt(step, preserved)
            elif step.kind == "eval":
                self.wait_for_preservation(step.resume_from, require_final_zip=True)
                try:
                    local = self.local_preservation_once(step.name, require_final_zip=False)
                    self.wait_for_preservation(step.name, require_final_zip=False)
                    self.wait_for_job_rc0(step.name)
                    self.validate_eval_receipt(step, local)
                    self.emit(event="skip_completed_local_eval", name=step.name)
                    completed.append(asdict(step))
                    continue
                except (FileNotFoundError, KeyError, json.JSONDecodeError):
                    pass
                command = self.eval_command(step)
                self.ensure_job_started_or_finished(step, command)
                self.preserve_after_job(step.name, require_final_zip=False)
                preserved = self.wait_for_preservation(step.name, require_final_zip=False)["local"]
                self.validate_eval_receipt(step, preserved)
            else:
                raise ValueError(f"unknown retention step kind: {step.kind}")
            completed.append(asdict(step))
        return completed

    def _step_with_result_sha(self, step):
        preserved = self.local_preservation_once(step.name, require_final_zip=True)
        return PlanStep(**{**asdict(step), "result_sha256": preserved["files"]["final.zip"]["sha256"]})

    def _select_after_matched_cycle(self, candidates, *, b40_baseline):
        matched_sequence = []
        for candidate in candidates:
            matched_sequence.extend([candidate, eval_step(candidate)])
        completed = self._run_steps(matched_sequence)
        gate_summary = self.apply_eval_gate(candidates, b40_baseline=b40_baseline)
        eligible = [candidate for candidate in candidates if candidate.name not in self.stopped_lineages]
        comparison = {
            "cycle": 2,
            "status": "completed",
            "candidates": [candidate.name for candidate in candidates],
            "gates": gate_summary["outcomes"],
        }
        if not eligible:
            return None, None, None, completed, {**comparison, "status": "all_failed_gate", "relation": "gate_eliminated"}
        if len(eligible) == 1:
            winner = eligible[0]
            return (
                winner,
                self._step_with_result_sha(winner),
                self.local_preservation_once(eval_step(winner).name, require_final_zip=False)["result"],
                completed,
                {**comparison, "status": "selected", "relation": "gate_eliminated", "winner": winner.name},
            )

        left_eval = self.local_preservation_once(eval_step(eligible[0]).name, require_final_zip=False)["result"]
        right_eval = self.local_preservation_once(eval_step(eligible[1]).name, require_final_zip=False)["result"]
        relation = pareto_relation(left_eval, right_eval)
        if relation.winner == "left":
            winner = eligible[0]
        elif relation.winner == "right":
            winner = eligible[1]
        else:
            return None, None, None, completed, {**comparison, "status": "unresolved", "relation": relation.relation}
        return (
            winner,
            self._step_with_result_sha(winner),
            self.local_preservation_once(eval_step(winner).name, require_final_zip=False)["result"],
            completed,
            {**comparison, "status": "selected", "relation": relation.relation, "winner": winner.name},
        )

    def run(self, plan=None, *, b40_baseline=B40_BASELINE, max_cycles=5):
        if plan is not None:
            self.ensure_contact_qualification()
            completed = self._run_steps(plan)
            return {"status": "complete", "completed": completed, "stopped_lineages": sorted(self.stopped_lineages)}

        self.ensure_contact_qualification()
        completed = []
        initial = build_initial_plan(source_digest=self.plan_source_digest)
        initial_sequence = []
        for step in initial:
            initial_sequence.extend([step, eval_step(step)])
        completed.extend(self._run_steps(initial_sequence))
        gate_summary = self.apply_eval_gate(initial, b40_baseline=b40_baseline)
        eligible = [step for step in initial if step.name not in self.stopped_lineages]
        if not eligible:
            return {
                "status": "stopped_no_eligible_initial",
                "completed": completed,
                "initial_gates": gate_summary["outcomes"],
                "baseline_provenance": B40_BASELINE_PROVENANCE,
                "stopped_lineages": sorted(self.stopped_lineages),
            }

        winner_step = eligible[0]
        matched_comparison = None
        winner = None
        previous_eval = None
        next_cycle = 2
        if len(eligible) == 2:
            left_eval = self.local_preservation_once(eval_step(eligible[0]).name, require_final_zip=False)["result"]
            right_eval = self.local_preservation_once(eval_step(eligible[1]).name, require_final_zip=False)["result"]
            relation = pareto_relation(left_eval, right_eval)
            if relation.winner == "right":
                winner_step = eligible[1]
            elif relation.winner is None:
                if max_cycles < 2:
                    return {
                        "status": "matched_cycle_not_run_max_cycles_exhausted",
                        "completed": completed,
                        "initial_gates": gate_summary["outcomes"],
                        "baseline_provenance": B40_BASELINE_PROVENANCE,
                        "matched_comparison": {
                            "cycle": 2,
                            "status": "not_run",
                            "relation": relation.relation,
                            "candidates": [step.name for step in eligible],
                        },
                        "stopped_lineages": sorted(self.stopped_lineages),
                    }
                matched_candidates = [
                    build_continuation_step(self._step_with_result_sha(candidate), 2)
                    for candidate in eligible
                ]
                selected_step, selected_winner, selected_eval, matched_completed, matched_comparison = self._select_after_matched_cycle(
                    matched_candidates,
                    b40_baseline=b40_baseline,
                )
                completed.extend(matched_completed)
                if selected_step is None:
                    return {
                        "status": "stopped_matched_cycle_all_failed_gate" if matched_comparison["status"] == "all_failed_gate" else "matched_cycle_unresolved_tradeoff",
                        "completed": completed,
                        "initial_gates": gate_summary["outcomes"],
                        "baseline_provenance": B40_BASELINE_PROVENANCE,
                        "matched_comparison": matched_comparison,
                        "stopped_lineages": sorted(self.stopped_lineages),
                    }
                winner_step = selected_step
                winner = selected_winner
                previous_eval = selected_eval
                next_cycle = 3

        if winner is None:
            winner = self._step_with_result_sha(winner_step)
            previous_eval = self.local_preservation_once(eval_step(winner_step).name, require_final_zip=False)["result"]
        continuation_state = ContinuationState()
        last_decision = None
        for cycle in range(next_cycle, max_cycles + 1):
            self.validate_budget()
            continuation = build_continuation_step(winner, cycle)
            continuation_eval = eval_step(continuation)
            completed.extend(self._run_steps([continuation, continuation_eval]))
            gate = self.apply_eval_gate([continuation], b40_baseline=b40_baseline)
            if continuation.name in self.stopped_lineages:
                return {
                    "status": "stopped_failed_gate",
                    "winner": winner.name,
                    "failed_checkpoint": continuation.name,
                    "selected_candidate": winner_step.name,
                    "completed": completed,
                    "last_gate": gate["outcomes"].get(continuation.name),
                    "baseline_provenance": B40_BASELINE_PROVENANCE,
                    "stopped_lineages": sorted(self.stopped_lineages),
                }
            current_eval = self.local_preservation_once(continuation_eval.name, require_final_zip=False)["result"]
            last_decision = continuation_state.update(previous_eval, current_eval)
            winner = self._step_with_result_sha(continuation)
            previous_eval = current_eval
            if last_decision.stop:
                break

        return {
            "status": "complete",
            "winner": winner.name,
            "selected_candidate": winner_step.name,
            "completed": completed,
            "initial_gates": gate_summary["outcomes"],
            "baseline_provenance": B40_BASELINE_PROVENANCE,
            "last_continuation_decision": asdict(last_decision) if last_decision is not None else None,
            "matched_comparison": matched_comparison,
            "stopped_lineages": sorted(self.stopped_lineages),
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=contact.POLL_SECONDS)
    parser.add_argument("--remaining-budget-dollars", type=float, required=True)
    parser.add_argument("--hourly-cost-dollars", type=float, required=True)
    parser.add_argument("--ssh-host", default=contact.SSH_HOST)
    parser.add_argument("--ssh-port", default=contact.SSH_PORT)
    parser.add_argument("--ssh-key", type=Path, default=contact.SSH_KEY)
    parser.add_argument("--remote-root", type=Path, default=REMOTE_ROOT)
    parser.add_argument("--remote-artifacts", type=Path, default=REMOTE_ARTIFACTS)
    parser.add_argument("--remote-python", type=Path)
    parser.add_argument("--qualification", type=Path, default=QUALIFICATION)
    parser.add_argument("--contact-qualification-block", default=CONTACT_QUALIFICATION_BLOCK)
    parser.add_argument("--plan-source-digest")
    parser.add_argument("--frozen-root", type=Path, default=FROZEN_ROOT)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint-ack", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    event_log = args.event_log or (args.local_root / "retention-continuation-events.jsonl")
    receipt = args.receipt or (args.local_root / "retention-continuation-receipt.json")
    if Path(event_log).exists() or Path(receipt).exists():
        raise FileExistsError("controller event log and receipt are immutable")
    started = time.time()
    remote = RemoteClient(
        ssh_host=args.ssh_host,
        ssh_port=args.ssh_port,
        ssh_key=args.ssh_key,
        remote_root=args.remote_root,
        remote_artifacts=args.remote_artifacts,
        remote_python=args.remote_python or (args.remote_root / "venv/bin/python"),
        deadline_unix=args.deadline_unix,
    )
    remote_artifacts = args.remote_artifacts
    controller = Controller(
        local_root=args.local_root,
        remote=remote,
        deadline_unix=args.deadline_unix,
        event_log=event_log,
        poll_seconds=args.poll_seconds,
        remaining_cost=ElapsedBudgetMeter(
            initial_remaining_dollars=args.remaining_budget_dollars,
            hourly_cost_dollars=args.hourly_cost_dollars,
            started_unix=started,
        ).remaining,
        remote_artifacts=remote_artifacts,
        qualification=args.qualification,
        contact_qualification_block=args.contact_qualification_block,
        plan_source_digest=args.plan_source_digest,
        frozen_root=args.frozen_root,
        reference_root=args.reference_root,
        initial_checkpoint=args.initial_checkpoint or (remote_artifacts / B40_BLOCK_NAME / "final.zip"),
        initial_checkpoint_ack=args.initial_checkpoint_ack or (remote_artifacts / B40_BLOCK_NAME / "preservation_ack.json"),
    )
    result = controller.run()
    result.update(
        schema_version="ai2.retention_campaign_continuation.v1",
        started_unix=started,
        finished_unix=time.time(),
        event_log=str(event_log),
    )
    atomic_json(receipt, result)
    print(json.dumps({"status": result["status"], "stopped_lineages": result["stopped_lineages"]}, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
