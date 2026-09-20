#!/usr/bin/env python3
"""Preservation-gated controller for native retention curriculum campaigns.

This module plans and gates immutable retention blocks.  It deliberately keeps
provider mutation outside the controller: remote execution is injectable and
uses the same preservation ACK contract as the contact campaign controller.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import time

import continue_contact_campaign as contact


B40_SHA256 = "6d7b83a287b1a29bf8741295d769178764fd8b4beaf101a678b62cdc41bd73f0"
B40_BLOCK_NAME = "B_projected-040m"
REMOTE_ROOT = Path("/workspace/AI_2/staged-20260920")
REMOTE_ARTIFACTS = REMOTE_ROOT / "artifacts"
LOCAL_ROOT = Path("/Users/elliot18/Projects/ml-research/AI_2/baselines/native-retention-20260920/campaign")
QUALIFICATION = contact.QUALIFICATION
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

sha256 = contact.sha256
bytes_sha256 = contact.bytes_sha256
read_json = contact.read_json
atomic_json = contact.atomic_json
verify_ack_for_bytes = contact.verify_ack_for_bytes
verify_local_preservation = contact.verify_local_preservation
RemoteClient = contact.RemoteClient


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


def _source_prefix(source: str) -> str:
    if not isinstance(source, str) or len(source) < 12:
        raise ValueError("source digest must be a sha256-like string")
    return source[:12]


def _step_name(recipe: str, cumulative_added: int, seed: int, source_digest: str, *, cycle: int = 1) -> str:
    return f"{recipe}-cycle{cycle:02d}-add{cumulative_added}-s{seed}-src{_source_prefix(source_digest)}"


def build_initial_plan(parent_sha256=B40_SHA256):
    steps = []
    for recipe in INITIAL_RECIPES:
        seed = INITIAL_SEEDS[recipe]
        steps.append(
            PlanStep(
                kind="train",
                name=_step_name(recipe, CYCLE_TRANSITIONS, seed, parent_sha256, cycle=1),
                recipe=recipe,
                seed=seed,
                cumulative_added_transitions=CYCLE_TRANSITIONS,
                parent_sha256=parent_sha256,
                source_digest=parent_sha256,
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


def _require_contact_terminal_attribution(row, name):
    bad = {
        "self_losses": row.get("self_losses", 0),
        "opponent_self_losses": row.get("opponent_self_losses", 0),
        "timeouts": row.get("timeouts", 0),
        "ambiguous": row.get("ambiguous", 0),
    }
    if any(value != 0 for value in bad.values()):
        raise ValueError(f"{name} terminal attribution must have no self-loss, opponent self-loss, timeout, or ambiguous endings")


def _component_values(result):
    contact_det = _panel_mode(result, "contact", "deterministic")
    contact_sampled = _panel_mode(result, "contact", "sampled")
    nearby_det = _panel_mode(result, "nearby", "deterministic")
    nearby_sampled = _panel_mode(result, "nearby", "sampled")
    wide_det = _panel_mode(result, "wide", "deterministic")
    wide_sampled = _panel_mode(result, "wide", "sampled")
    _require_contact_terminal_attribution(contact_det, "contact/deterministic")
    _require_contact_terminal_attribution(contact_sampled, "contact/sampled")
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
        "terminal_attribution": 1,
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
            "passed": current["contact_deterministic"] >= max(18, baseline["contact_deterministic"]),
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
        "terminal_attribution": {"candidate": 1, "baseline": 1, "passed": True},
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
        if key.endswith("_falls"):
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


def pareto_relation(left, right):
    left_better, right_better = _pareto_vectors(left, right)
    if left_better and not right_better:
        return ParetoResult(winner="left", relation="left_dominates")
    if right_better and not left_better:
        return ParetoResult(winner="right", relation="right_dominates")
    if not left_better and not right_better:
        return ParetoResult(winner=None, relation="tie")
    comparison = PlanStep(
        kind="matched_comparison",
        name=f"matched-comparison-{int(time.time())}",
        recipe="matched",
        seed=919191,
        cumulative_added_transitions=0,
        parent_sha256=B40_SHA256,
        rollouts=0,
    )
    return ParetoResult(winner=None, relation="tradeoff", matched_comparisons=(comparison,))


def improved(previous, current):
    left_better, right_better = _pareto_vectors(current, previous)
    return bool(left_better and not right_better)


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
                 qualification=QUALIFICATION, frozen_root=FROZEN_ROOT, reference_root=REFERENCE_ROOT,
                 initial_checkpoint=REMOTE_ARTIFACTS / B40_BLOCK_NAME / "final.zip",
                 initial_checkpoint_ack=REMOTE_ARTIFACTS / B40_BLOCK_NAME / "preservation_ack.json",
                 **kwargs):
        super().__init__(**kwargs)
        self.remaining_cost = remaining_cost
        self.cost_reserve = float(cost_reserve)
        self.stopped_lineages = set()
        self.qualification = Path(qualification)
        self.frozen_root = Path(frozen_root)
        self.reference_root = Path(reference_root)
        self.initial_checkpoint = Path(initial_checkpoint)
        self.initial_checkpoint_ack = Path(initial_checkpoint_ack)

    def validate_budget(self):
        if self.now() > self.deadline_unix - RESERVE_SECONDS:
            raise TimeoutError("deadline reserve exhausted")
        remaining = float(self.remaining_cost())
        if remaining < self.cost_reserve:
            raise TimeoutError("remaining-cost reserve exhausted")

    def validate_deadline(self):
        self.validate_budget()

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
            "python3",
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
            str(REMOTE_ARTIFACTS / step.name),
            "--deadline-unix",
            str(self.deadline_unix),
        ]
        if step.resume_from:
            parent = self.preservation_once(step.resume_from, require_final_zip=True)["local"]
            command.extend([
                "--resume",
            ])
            command[command.index("--checkpoint") + 1] = str(REMOTE_ARTIFACTS / step.resume_from / "final.zip")
            command[command.index("--checkpoint-sha256") + 1] = parent["files"]["final.zip"]["sha256"]
            command[command.index("--checkpoint-ack") + 1] = str(REMOTE_ARTIFACTS / step.resume_from / "preservation_ack.json")
        return command

    def eval_command(self, step: PlanStep):
        if step.kind != "eval" or not step.resume_from:
            raise ValueError("expected eval step with train parent")
        candidate = self.preservation_once(step.resume_from, require_final_zip=True)
        return [
            "python3",
            "-B",
            "evaluate_retention_campaign.py",
            "--checkpoint",
            str(REMOTE_ARTIFACTS / step.resume_from / "final.zip"),
            "--checkpoint-sha256",
            candidate["local"]["files"]["final.zip"]["sha256"],
            "--verified-ack",
            str(REMOTE_ARTIFACTS / step.resume_from / "preservation_ack.json"),
            "--frozen-root",
            str(self.frozen_root),
            "--reference-root",
            str(self.reference_root),
            "--qualification",
            str(self.qualification),
            "--output",
            str(REMOTE_ARTIFACTS / step.name),
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

    def run(self, plan=None):
        completed = []
        for step in plan or build_initial_plan():
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
                preserved = self.wait_for_preservation(step.name, require_final_zip=False)["local"]
                self.validate_eval_receipt(step, preserved)
            else:
                raise ValueError(f"unknown retention step kind: {step.kind}")
            completed.append(asdict(step))
        return {"status": "complete", "completed": completed, "stopped_lineages": sorted(self.stopped_lineages)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT)
    parser.add_argument("--deadline-unix", type=float, default=DEADLINE_UNIX)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=contact.POLL_SECONDS)
    parser.add_argument("--remaining-budget-dollars", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    event_log = args.event_log or (args.local_root / "retention-continuation-events.jsonl")
    receipt = args.receipt or (args.local_root / "retention-continuation-receipt.json")
    if Path(event_log).exists() or Path(receipt).exists():
        raise FileExistsError("controller event log and receipt are immutable")
    controller = Controller(
        local_root=args.local_root,
        deadline_unix=args.deadline_unix,
        event_log=event_log,
        poll_seconds=args.poll_seconds,
        remaining_cost=lambda: args.remaining_budget_dollars,
    )
    started = time.time()
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
