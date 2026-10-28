#!/usr/bin/env python3
"""Run immutable one-rollout causal probes for retention curriculum diagnosis."""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import ai2.runtime.continue_retention_campaign as retention


SCHEMA_VERSION = "ai2.causal_retention_campaign.v1"
REMOTE_ROOT = retention.REMOTE_ROOT
REMOTE_ARTIFACTS = retention.REMOTE_ARTIFACTS
LOCAL_ROOT = retention.LOCAL_ROOT
LANES = retention.LANES
N_STEPS = retention.N_STEPS
BATCH_SIZE = 16384
TRANSITIONS_PER_ROLLOUT = retention.TRANSITIONS_PER_ROLLOUT
OPTIMIZER_STEPS_PER_ROLLOUT = 512
B40_SHA256 = retention.B40_SHA256
B40_STAGE = 40
B40_TIMESTEPS = retention.contact.expected_counter_for_stage(B40_STAGE)
B40_BLOCK_NAME = retention.B40_BLOCK_NAME
PLAN_SOURCE_DIGEST = retention.PLAN_SOURCE_DIGEST
ACTION_AB_ARM = retention.ACTION_AB_ARM
CONTACT_QUALIFICATION_BLOCK = retention.CONTACT_QUALIFICATION_BLOCK
QUALIFICATION = retention.QUALIFICATION
FROZEN_ROOT = retention.FROZEN_ROOT
REFERENCE_ROOT = retention.REFERENCE_ROOT
CONTACT_QUALIFICATION_CHECKPOINT = retention.CONTACT_QUALIFICATION_CHECKPOINT
DEADLINE_UNIX = retention.DEADLINE_UNIX

CAUSAL_SCHEDULES = {
    "P1C": ("combat",),
    "P1N": ("navigation",),
}
PROBE_SEEDS = {
    "P1C": 510101,
    "P1N_INITIAL": 520102,
    "P1N_REPAIR": 510103,
}

sha256 = retention.sha256
read_json = retention.read_json
atomic_json = retention.atomic_json


@dataclass(frozen=True)
class CausalStep:
    kind: str
    name: str
    recipe: str
    seed: int
    parent_sha256: str
    resume_from: str | None = None
    rollouts: int = 1
    cumulative_added_transitions: int = TRANSITIONS_PER_ROLLOUT
    source_digest: str = PLAN_SOURCE_DIGEST
    result_sha256: str | None = None


def _source_prefix(source: str) -> str:
    return retention._source_prefix(source)


def _step_name(label: str, recipe: str, seed: int, source_digest: str, added: int) -> str:
    return f"{label}-{recipe}-add{added}-s{seed}-src{_source_prefix(source_digest)}"


def build_p1c_step(*, source_digest=PLAN_SOURCE_DIGEST, parent_sha256=B40_SHA256):
    return CausalStep(
        kind="train",
        name=_step_name("probe", "P1C", PROBE_SEEDS["P1C"], source_digest, TRANSITIONS_PER_ROLLOUT),
        recipe="P1C",
        seed=PROBE_SEEDS["P1C"],
        parent_sha256=parent_sha256,
        source_digest=source_digest,
    )


def build_p1n_step(*, source_digest=PLAN_SOURCE_DIGEST, parent_sha256=B40_SHA256):
    return CausalStep(
        kind="train",
        name=_step_name("probe", "P1N", PROBE_SEEDS["P1N_INITIAL"], source_digest, TRANSITIONS_PER_ROLLOUT),
        recipe="P1N",
        seed=PROBE_SEEDS["P1N_INITIAL"],
        parent_sha256=parent_sha256,
        source_digest=source_digest,
    )


def build_repair_step(parent: CausalStep, *, source_digest=PLAN_SOURCE_DIGEST):
    if parent.recipe != "P1C" or not getattr(parent, "result_sha256", None):
        raise ValueError("repair probe must resume from a preserved P1C step with result_sha256")
    return CausalStep(
        kind="train",
        name=_step_name("repair-after-p1c", "P1N", PROBE_SEEDS["P1N_REPAIR"], source_digest, 2 * TRANSITIONS_PER_ROLLOUT),
        recipe="P1N",
        seed=PROBE_SEEDS["P1N_REPAIR"],
        parent_sha256=parent.result_sha256,
        resume_from=parent.name,
        cumulative_added_transitions=2 * TRANSITIONS_PER_ROLLOUT,
        source_digest=source_digest,
    )


def eval_step(train_step: CausalStep):
    return CausalStep(
        kind="eval",
        name=f"eval-{train_step.name}",
        recipe=train_step.recipe,
        seed=train_step.seed,
        parent_sha256=train_step.parent_sha256,
        resume_from=train_step.name,
        rollouts=0,
        cumulative_added_transitions=train_step.cumulative_added_transitions,
        source_digest=train_step.source_digest,
    )


def build_initial_plan(*, source_digest=PLAN_SOURCE_DIGEST):
    return (build_p1c_step(source_digest=source_digest), build_p1n_step(source_digest=source_digest))


def _finite(value) -> bool:
    return retention._finite(value)


def _metric_deltas(result, baseline):
    current = retention._component_values(result)
    base = retention._component_values(baseline)
    return {
        key: {
            "candidate": current[key],
            "baseline": base[key],
            "delta": current[key] - base[key],
            "lower_is_better": retention._lower_is_better(key),
        }
        for key in current
    }


def _pairwise_relation(left_name, left_result, right_name, right_result):
    relation = retention.pareto_relation(left_result, right_result)
    left_better, right_better = retention._pareto_vectors(left_result, right_result)
    return {
        "left": left_name,
        "right": right_name,
        "relation": relation.relation,
        "winner": {"left": left_name, "right": right_name}.get(relation.winner),
        "left_better_components": left_better,
        "right_better_components": right_better,
    }


class Controller(retention.Controller):
    """Causal probe controller that reuses the retention preservation contract."""

    def validate_train_step(self, step: CausalStep):
        if step.kind != "train":
            raise ValueError("expected train step")
        if step.recipe not in CAUSAL_SCHEDULES:
            raise ValueError(f"unsupported causal recipe {step.recipe!r}")
        if step.rollouts != 1:
            raise ValueError("causal probes must be exactly one rollout")
        if step.source_digest != self.plan_source_digest:
            raise ValueError("causal probe source digest mismatch")
        expected_seed = PROBE_SEEDS["P1C"] if step.recipe == "P1C" else PROBE_SEEDS["P1N_REPAIR" if step.resume_from else "P1N_INITIAL"]
        if step.seed != expected_seed:
            raise ValueError(f"{step.name} seed mismatch")
        label = "repair-after-p1c" if step.resume_from else "probe"
        expected_name = _step_name(label, step.recipe, step.seed, step.source_digest, step.cumulative_added_transitions)
        if step.name != expected_name:
            raise ValueError(f"{step.name} does not match immutable causal name {expected_name}")
        if step.resume_from is None:
            if step.parent_sha256 != B40_SHA256 or step.cumulative_added_transitions != TRANSITIONS_PER_ROLLOUT:
                raise ValueError("initial causal probes must start from exact B40")
        else:
            if step.recipe != "P1N" or step.seed != PROBE_SEEDS["P1N_REPAIR"]:
                raise ValueError("only the generic P1N repair probe may resume")
            parent = self.local_preservation_once(step.resume_from, require_final_zip=True)
            if step.parent_sha256 != parent["files"]["final.zip"]["sha256"]:
                raise ValueError("repair parent_sha256 does not match preserved P1C final.zip")

    def train_command(self, step: CausalStep):
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
            str(BATCH_SIZE),
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

    def eval_command(self, step: CausalStep):
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

    def validate_train_receipt(self, step: CausalStep, preserved: dict):
        self.validate_train_step(step)
        result = preserved["result"]
        manifest = preserved["manifest"]
        expected_schedule = list(CAUSAL_SCHEDULES[step.recipe])
        if result.get("schema_version") != "ai2.retention_campaign_block.v1":
            raise ValueError(f"{step.name} schema_version mismatch")
        if result.get("status") != "trained_not_evaluated":
            raise ValueError(f"{step.name} did not finish as a training block")
        if result.get("checkpoint_sha256") != preserved["files"]["final.zip"]["sha256"]:
            raise ValueError(f"{step.name} checkpoint sha mismatch")
        if result.get("schedule_id") != step.recipe or (manifest.get("config") or {}).get("schedule_id") != step.recipe:
            raise ValueError(f"{step.name} schedule mismatch")
        if result.get("schedule") != expected_schedule or manifest.get("schedule") != expected_schedule:
            raise ValueError(f"{step.name} schedule lineage mismatch")
        config = manifest.get("config") or {}
        if config.get("seed") != step.seed or config.get("action_ab_arm") != ACTION_AB_ARM:
            raise ValueError(f"{step.name} seed/action contract mismatch")
        if (manifest.get("checkpoint") or {}).get("sha256") != step.parent_sha256:
            raise ValueError(f"{step.name} lineage parent checkpoint sha mismatch")
        sources = manifest.get("source_sha256") or {}
        for key, expected in {
            "native_curriculum_ppo.py": retention.CONTACT_SOURCE_DIGEST,
            "train_retention_campaign_block.py": retention.RETENTION_RUNNER_DIGEST,
        }.items():
            if sources.get(key) != expected:
                raise ValueError(f"{step.name} source digest mismatch for {key}")
        if (result.get("summary") or {}).get("measured_transitions") != TRANSITIONS_PER_ROLLOUT:
            raise ValueError(f"{step.name} measured transitions mismatch")
        expected_timesteps = B40_TIMESTEPS + step.cumulative_added_transitions
        if (result.get("final_model_counters") or {}).get("num_timesteps") != expected_timesteps:
            raise ValueError(f"{step.name} final model timestep mismatch")
        rollouts = result.get("rollouts") or []
        if len(rollouts) != 1:
            raise ValueError(f"{step.name} rollout count mismatch")
        row = rollouts[0]
        if row.get("task") != expected_schedule[0]:
            raise ValueError(f"{step.name} rollout task mismatch")
        if row.get("transitions") != TRANSITIONS_PER_ROLLOUT or row.get("game_ticks") != TRANSITIONS_PER_ROLLOUT:
            raise ValueError(f"{step.name} rollout transition/game tick mismatch")
        for key in ("collection_seconds", "update_seconds", "total_seconds"):
            if not _finite(row.get(key)):
                raise ValueError(f"{step.name} rollout non-finite {key}")
        actual = row.get("actual_update") or {}
        if actual.get("optimizer_steps") != OPTIMIZER_STEPS_PER_ROLLOUT:
            raise ValueError(f"{step.name} optimizer steps mismatch")
        if actual.get("nominal_optimizer_steps") != OPTIMIZER_STEPS_PER_ROLLOUT:
            raise ValueError(f"{step.name} nominal optimizer steps mismatch")
        if actual.get("kl_early_stop") is not False or actual.get("low_update") is not False:
            raise ValueError(f"{step.name} KL early stop or low update")
        for value in (row.get("metrics") or {}).values():
            if not _finite(value):
                raise ValueError(f"{step.name} non-finite metric")
        return True

    def validate_eval_receipt(self, step: CausalStep, preserved: dict):
        if not step.resume_from:
            raise ValueError("eval step is missing train parent")
        trained = retention.verify_local_preservation(self.local_root, step.resume_from, require_final_zip=True)
        candidate = (preserved["manifest"].get("candidate") or {}).get("checkpoint_sha256")
        if candidate != trained["files"]["final.zip"]["sha256"]:
            raise ValueError(f"{step.name} candidate checkpoint_sha256 does not match train block")
        return True

    def _step_with_result_sha(self, step: CausalStep):
        preserved = self.local_preservation_once(step.name, require_final_zip=True)
        return replace(step, result_sha256=preserved["files"]["final.zip"]["sha256"])

    def _run_train_and_eval(self, step: CausalStep):
        completed = self._run_steps([step, eval_step(step)])
        train = self.local_preservation_once(step.name, require_final_zip=True)
        evaluation = self.local_preservation_once(eval_step(step).name, require_final_zip=False)
        return completed, train, evaluation

    def _gate_and_compare(self, steps, baseline):
        evaluations = {
            step.name: self.local_preservation_once(eval_step(step).name, require_final_zip=False)["result"]
            for step in steps
        }
        gates = {
            name: asdict(retention.diagnostic_gate(result, baseline))
            for name, result in evaluations.items()
        }
        deltas = {
            name: _metric_deltas(result, baseline)
            for name, result in evaluations.items()
        }
        comparisons = {
            "vs_b40": deltas,
            "p1c_vs_p1n": _pairwise_relation(steps[0].name, evaluations[steps[0].name], steps[1].name, evaluations[steps[1].name]),
            "p1c_vs_repair": _pairwise_relation(steps[0].name, evaluations[steps[0].name], steps[2].name, evaluations[steps[2].name]),
            "p1n_vs_repair": _pairwise_relation(steps[1].name, evaluations[steps[1].name], steps[2].name, evaluations[steps[2].name]),
        }
        return gates, comparisons

    def run(self, *, b40_baseline=retention.B40_BASELINE):
        self.ensure_contact_qualification()
        completed = []
        p1c, p1n = build_initial_plan(source_digest=self.plan_source_digest)
        for step in (p1c, p1n):
            rows, _, _ = self._run_train_and_eval(step)
            completed.extend(rows)
        repair = build_repair_step(self._step_with_result_sha(p1c), source_digest=self.plan_source_digest)
        rows, _, _ = self._run_train_and_eval(repair)
        completed.extend(rows)
        gates, comparisons = self._gate_and_compare((p1c, p1n, repair), b40_baseline)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_no_promotion",
            "completed": completed,
            "probes": [asdict(step) for step in (p1c, p1n, repair)],
            "gates": gates,
            "paired_comparisons": comparisons,
            "baseline": b40_baseline,
            "baseline_provenance": retention.B40_BASELINE_PROVENANCE,
            "source_digest": self.plan_source_digest,
            "next_action": "inspect causal deltas before authorizing any scale-up; no checkpoint is promoted by this controller",
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=retention.contact.POLL_SECONDS)
    parser.add_argument("--remaining-budget-dollars", type=float, required=True)
    parser.add_argument("--hourly-cost-dollars", type=float, required=True)
    parser.add_argument("--ssh-host", default=retention.contact.SSH_HOST)
    parser.add_argument("--ssh-port", default=retention.contact.SSH_PORT)
    parser.add_argument("--ssh-key", type=Path, default=retention.contact.SSH_KEY)
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
    event_log = args.event_log or (args.local_root / "causal-retention-events.jsonl")
    receipt = args.receipt or (args.local_root / "causal-retention-receipt.json")
    if Path(event_log).exists() or Path(receipt).exists():
        raise FileExistsError("controller event log and receipt are immutable")
    started = time.time()
    remote = retention.RemoteClient(
        ssh_host=args.ssh_host,
        ssh_port=args.ssh_port,
        ssh_key=args.ssh_key,
        remote_root=args.remote_root,
        remote_artifacts=args.remote_artifacts,
        remote_python=args.remote_python or (args.remote_root / "venv/bin/python"),
        deadline_unix=args.deadline_unix,
    )
    controller = Controller(
        local_root=args.local_root,
        remote=remote,
        deadline_unix=args.deadline_unix,
        event_log=event_log,
        poll_seconds=args.poll_seconds,
        remaining_cost=retention.ElapsedBudgetMeter(
            initial_remaining_dollars=args.remaining_budget_dollars,
            hourly_cost_dollars=args.hourly_cost_dollars,
            started_unix=started,
        ).remaining,
        remote_artifacts=args.remote_artifacts,
        qualification=args.qualification,
        contact_qualification_block=args.contact_qualification_block,
        plan_source_digest=args.plan_source_digest,
        frozen_root=args.frozen_root,
        reference_root=args.reference_root,
        initial_checkpoint=args.initial_checkpoint or (args.remote_artifacts / B40_BLOCK_NAME / "final.zip"),
        initial_checkpoint_ack=args.initial_checkpoint_ack or (args.remote_artifacts / B40_BLOCK_NAME / "preservation_ack.json"),
    )
    result = controller.run()
    result.update(started_unix=started, finished_unix=time.time(), event_log=str(event_log))
    atomic_json(receipt, result)
    print(json.dumps({"status": result["status"], "probes": [row["name"] for row in result["probes"]]}, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
