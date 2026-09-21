"""Pure continuation gates for native B2 reengagement campaigns.

The controller logic in this module is deliberately local and side-effect
light. It decides whether a preserved checkpoint may advance, but it does not
rent GPUs, manage Vast instances, or infer remaining budget from a ceiling.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import hashlib
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "ai2.reengagement_continuation.v1"
SCHEDULE = "R4E1N"
LANES = 4096
N_STEPS = 512
ROLLOUTS_PER_CYCLE = 5
TRANSITIONS_PER_ROLLOUT = LANES * N_STEPS
CYCLE_TRANSITIONS = ROLLOUTS_PER_CYCLE * TRANSITIONS_PER_ROLLOUT
CAMPAIGN_CEILING_DOLLARS = 12.0

DEVELOPMENT_EPISODES = 20
DEVELOPMENT_MIN_SUCCESSES = 18
DEVELOPMENT_MAX_LEARNER_FALLS = 1
CONFIRMATION_EPISODES = 100
CONFIRMATION_MIN_SUCCESSES = 90
CONFIRMATION_MAX_FALLS = 5
CONFIRMATION_MIN_SLOT_SUCCESSES = 40

REQUIRED_RECEIPT_FIELDS = (
    "checkpoint",
    "inventory",
    "local_acknowledgement",
    "evaluation_receipt",
    "wandb",
    "logs",
    "monitor",
    "commands",
)


@dataclass(frozen=True)
class CheckpointRef:
    name: str
    checkpoint_hash: str
    counter: int
    cycle_index: int


@dataclass(frozen=True)
class MatchedDevelopmentRerun:
    parent: CheckpointRef
    result: dict


@dataclass(frozen=True)
class CycleStep:
    parent_checkpoint: CheckpointRef
    cycle_index: int
    schedule: str = SCHEDULE
    rollouts: int = ROLLOUTS_PER_CYCLE
    added_transitions: int = CYCLE_TRANSITIONS


@dataclass(frozen=True)
class GateVerdict:
    eligible: bool
    stop: bool
    reason: str
    checkpoint_counter: int | None
    checkpoint_hash: str | None
    components: dict
    raw: dict


@dataclass(frozen=True)
class MasteryVerdict:
    mastered: bool
    reason: str
    checkpoints: tuple[str, ...]


@dataclass(frozen=True)
class AlarmVerdict:
    pause: bool
    reason: str
    components: dict


@dataclass(frozen=True)
class ConfirmationPermission:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class StopVerdict:
    stop: bool
    reason: str


@dataclass(frozen=True)
class CampaignBudget:
    ceiling_dollars: float = CAMPAIGN_CEILING_DOLLARS
    actual_spend_dollars: float | None = None

    def __post_init__(self):
        ceiling = float(self.ceiling_dollars)
        if ceiling > CAMPAIGN_CEILING_DOLLARS:
            raise ValueError("campaign ceiling support is capped at $12")
        if ceiling < 0 or not math.isfinite(ceiling):
            raise ValueError("campaign ceiling must be finite and non-negative")

    def remaining(self) -> float:
        if self.actual_spend_dollars is None:
            raise ValueError("actual spend must be supplied; remaining budget is never inferred")
        spend = float(self.actual_spend_dollars)
        if spend < 0 or not math.isfinite(spend):
            raise ValueError("actual spend must be finite and non-negative")
        return float(self.ceiling_dollars) - spend


class NoImprovementTracker:
    def __init__(self):
        self.checks = 0

    def update(self, current: dict, parent: dict) -> StopVerdict:
        if _has_success_or_time_improvement(current, parent):
            self.checks = 0
            return StopVerdict(False, "improved")
        self.checks += 1
        if self.checks >= 3:
            return StopVerdict(True, "no_improvement_for_3_checks")
        return StopVerdict(False, "no_improvement")


def build_next_cycle(
    *,
    current: CheckpointRef,
    matched_parent_rerun: MatchedDevelopmentRerun | None,
    current_result: dict,
    cycle_index: int,
) -> CycleStep:
    if matched_parent_rerun is None:
        raise ValueError("matched parent development-bank rerun is required before update")
    if not isinstance(cycle_index, int) or isinstance(cycle_index, bool) or cycle_index != current.cycle_index + 1:
        raise ValueError("advance exactly one cycle at a time")
    _assert_result_matches_checkpoint(matched_parent_rerun.result, matched_parent_rerun.parent, "matched parent")
    _assert_result_matches_checkpoint(current_result, current, "current checkpoint")
    if development_gate(matched_parent_rerun.result).stop:
        raise ValueError("matched parent development rerun evidence is invalid")
    if development_gate(current_result).stop:
        raise ValueError("current checkpoint evidence is invalid")
    return CycleStep(parent_checkpoint=current, cycle_index=cycle_index)


def development_gate(result: dict) -> GateVerdict:
    raw = dict(result)
    panels = _panels(result)
    counter = result.get("checkpoint_counter")
    checkpoint_hash = _checkpoint_hash(result)
    components = {}
    stop = False
    failed = []
    if not isinstance(counter, int) or counter < 0:
        failed.append("checkpoint_counter")
        stop = True
    if not _sha_like(checkpoint_hash):
        failed.append("checkpoint_hash")
        stop = True
    for mode in ("deterministic", "sampled"):
        row = _mode_row(panels, mode, DEVELOPMENT_EPISODES)
        try:
            successes = _finite_number(row.get("successes"), f"{mode}.successes")
            falls = _finite_number(row.get("falls", 0), f"{mode}.falls")
            invalid = _finite_number(row.get("invalid", 0), f"{mode}.invalid")
            ambiguous = _finite_number(row.get("ambiguous", 0), f"{mode}.ambiguous")
            time_to_reengagement = _finite_number(row.get("time_to_reengagement", 0.0), f"{mode}.time_to_reengagement")
        except ValueError:
            failed.append(f"{mode}.numerical_failure")
            stop = True
            components[mode] = {"numerical_failure": True}
            continue
        mode_components = {
            "successes": successes,
            "falls": falls,
            "invalid": invalid,
            "ambiguous": ambiguous,
            "time_to_reengagement": time_to_reengagement,
        }
        components[mode] = mode_components
        if invalid or ambiguous:
            failed.append(f"{mode}.invalid_or_ambiguous")
            stop = True
        if successes < DEVELOPMENT_MIN_SUCCESSES:
            failed.append(f"{mode}.successes")
        if falls > DEVELOPMENT_MAX_LEARNER_FALLS:
            failed.append(f"{mode}.falls")
    return GateVerdict(
        eligible=not failed,
        stop=stop,
        reason="eligible" if not failed else "failed: " + ", ".join(failed),
        checkpoint_counter=counter if isinstance(counter, int) else None,
        checkpoint_hash=checkpoint_hash if _sha_like(checkpoint_hash) else None,
        components=components,
        raw=raw,
    )


def development_mastery(verdicts) -> MasteryVerdict:
    qualified = [row for row in verdicts if row.eligible and row.checkpoint_counter is not None and row.checkpoint_hash]
    if len(qualified) < 2:
        return MasteryVerdict(False, "need_two_qualified_development_checkpoints", ())
    last_two = qualified[-2:]
    if last_two[0].checkpoint_hash == last_two[1].checkpoint_hash:
        return MasteryVerdict(False, "checkpoints_must_be_distinct", tuple(row.checkpoint_hash for row in last_two))
    if last_two[1].checkpoint_counter <= last_two[0].checkpoint_counter:
        return MasteryVerdict(False, "checkpoint_counters_must_increase", tuple(row.checkpoint_hash for row in last_two))
    return MasteryVerdict(True, "mastery", tuple(row.checkpoint_hash for row in last_two))


def confirmation_allowed(development_verdicts) -> ConfirmationPermission:
    mastery = development_mastery(development_verdicts)
    return ConfirmationPermission(mastery.mastered, "allowed" if mastery.mastered else mastery.reason)


def confirmation_gate(result: dict) -> GateVerdict:
    panels = _panels(result)
    counter = result.get("checkpoint_counter")
    checkpoint_hash = _checkpoint_hash(result)
    components = {}
    failed = []
    stop = False
    if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
        failed.append("checkpoint_counter")
        stop = True
    if not _sha_like(checkpoint_hash):
        failed.append("checkpoint_hash")
        stop = True
    for mode in ("deterministic", "sampled"):
        row = _mode_row(panels, mode, CONFIRMATION_EPISODES)
        try:
            successes = _finite_number(row.get("successes"), f"{mode}.successes")
            falls = _finite_number(row.get("falls", 0), f"{mode}.falls")
            invalid = _finite_number(row.get("invalid", 0), f"{mode}.invalid")
            ambiguous = _finite_number(row.get("ambiguous", 0), f"{mode}.ambiguous")
        except ValueError:
            failed.append(f"{mode}.numerical_failure")
            stop = True
            components[mode] = {"numerical_failure": True}
            continue
        slots = row.get("learner_slots")
        if not isinstance(slots, dict):
            raise TypeError(f"{mode} learner slots are required")
        if set(slots) != {"slot0", "slot1"}:
            raise ValueError(f"{mode} learner slots must be exactly slot0 and slot1")
        slot_values = {}
        for name, slot in slots.items():
            if not isinstance(slot, dict):
                raise TypeError(f"{mode}.{name} must be an object")
            if slot.get("episodes") != 50:
                raise ValueError(f"{mode}.{name} must contain exactly 50 episodes")
            slot_values[name] = {
                "episodes": 50,
                "successes": _finite_number(slot.get("successes"), f"{mode}.{name}.successes"),
            }
        components[mode] = {
            "successes": successes,
            "falls": falls,
            "invalid": invalid,
            "ambiguous": ambiguous,
            "learner_slots": slot_values,
        }
        if invalid or ambiguous:
            failed.append(f"{mode}.invalid_or_ambiguous")
            stop = True
        if successes < CONFIRMATION_MIN_SUCCESSES:
            failed.append(f"{mode}.successes")
        if falls > CONFIRMATION_MAX_FALLS:
            failed.append(f"{mode}.falls")
        low_slots = [name for name, value in slot_values.items() if value["successes"] < CONFIRMATION_MIN_SLOT_SUCCESSES]
        if low_slots:
            failed.append(f"{mode}.learner_slots")
    return GateVerdict(
        eligible=not failed,
        stop=stop,
        reason="eligible" if not failed else "failed: " + ", ".join(failed),
        checkpoint_counter=counter if isinstance(counter, int) and not isinstance(counter, bool) else None,
        checkpoint_hash=checkpoint_hash if _sha_like(checkpoint_hash) else None,
        components=components,
        raw=dict(result),
    )


def retention_alarm(candidate: dict, matched_parent: dict) -> AlarmVerdict:
    components = {}
    reasons = []
    for domain in ("contact", "nearby", "wide"):
        components[domain] = {}
        for mode in ("deterministic", "sampled"):
            current = _retention_mode(candidate, domain, mode, DEVELOPMENT_EPISODES)
            parent = _retention_mode(matched_parent, domain, mode, DEVELOPMENT_EPISODES)
            lost_successes = _finite_number(parent.get("successes"), f"{domain}.{mode}.parent_successes") - _finite_number(
                current.get("successes"), f"{domain}.{mode}.successes"
            )
            added_falls = _retention_falls(current, domain, mode) - _retention_falls(parent, domain, mode, parent=True)
            components[domain][mode] = {"lost_successes": lost_successes, "added_falls": added_falls}
            if lost_successes > 2:
                reasons.append(f"{domain}.{mode}.lost_successes")
            if added_falls > 2:
                reasons.append(f"{domain}.{mode}.added_falls")
    return AlarmVerdict(bool(reasons), "ok" if not reasons else "pause: " + ", ".join(reasons), components)


def confirmation_retention_alarm(candidate: dict, matched_parent: dict) -> AlarmVerdict:
    components = {}
    reasons = []
    for domain in ("contact", "nearby", "wide"):
        components[domain] = {}
        for mode in ("deterministic", "sampled"):
            current = _retention_mode(candidate, domain, mode, CONFIRMATION_EPISODES)
            parent = _retention_mode(matched_parent, domain, mode, CONFIRMATION_EPISODES)
            success_loss_points = _finite_number(
                parent.get("successes"), f"{domain}.{mode}.parent_successes"
            ) - _finite_number(current.get("successes"), f"{domain}.{mode}.successes")
            fall_rate = _retention_falls(current, domain, mode) / CONFIRMATION_EPISODES
            components[domain][mode] = {
                "success_loss_points": success_loss_points,
                "fall_rate": fall_rate,
            }
            if success_loss_points > 5:
                reasons.append(f"{domain}.{mode}.success_loss")
            if fall_rate > 0.05:
                reasons.append(f"{domain}.{mode}.fall_rate")
    return AlarmVerdict(bool(reasons), "ok" if not reasons else "pause: " + ", ".join(reasons), components)


def stop_decision(
    *,
    mastered: bool = False,
    invalid_evidence: bool = False,
    numerical_failure: bool = False,
    severe_retention_regression: bool = False,
    budget_remaining_dollars: float | None = None,
    deadline_remaining_seconds: float | None = None,
    no_improvement_checks: int = 0,
) -> StopVerdict:
    if mastered:
        return StopVerdict(True, "mastery")
    if invalid_evidence:
        return StopVerdict(True, "invalid_evidence")
    if numerical_failure:
        return StopVerdict(True, "numerical_failure")
    if severe_retention_regression:
        return StopVerdict(True, "severe_retention_regression")
    if budget_remaining_dollars is None:
        return StopVerdict(True, "missing_actual_budget")
    if not _is_finite_number(budget_remaining_dollars) or budget_remaining_dollars < 0:
        return StopVerdict(True, "budget_exhausted")
    if deadline_remaining_seconds is None or not _is_finite_number(deadline_remaining_seconds) or deadline_remaining_seconds <= 0:
        return StopVerdict(True, "deadline")
    if no_improvement_checks >= 3:
        return StopVerdict(True, "no_improvement_for_3_checks")
    return StopVerdict(False, "continue")


def preserve_receipt(block_dir: Path, payload: dict, *, expected_checkpoint_hash: str) -> dict:
    block_dir = Path(block_dir)
    block_dir.mkdir(parents=True, exist_ok=True)
    receipt = _validate_receipt(payload, expected_checkpoint_hash=expected_checkpoint_hash, base_dir=block_dir)
    atomic_output(block_dir / "preservation_receipt.json", receipt)
    return receipt


def verify_reuse_receipt(block_dir: Path, *, expected_checkpoint_hash: str) -> dict:
    path = Path(block_dir) / "preservation_receipt.json"
    if not path.is_file():
        raise FileNotFoundError("preservation receipt is required before checkpoint reuse")
    return _validate_receipt(json.loads(path.read_text()), expected_checkpoint_hash=expected_checkpoint_hash, base_dir=Path(block_dir))


def atomic_output(path: Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        tmp.unlink(missing_ok=True)


def _validate_receipt(payload: dict, *, expected_checkpoint_hash: str, base_dir: Path | None = None) -> dict:
    if not _sha_like(expected_checkpoint_hash):
        raise ValueError("expected checkpoint hash must be sha256-like")
    if not isinstance(payload, dict):
        raise TypeError("preservation receipt must be an object")
    missing = [field for field in REQUIRED_RECEIPT_FIELDS if field not in payload]
    if missing:
        raise ValueError("preservation receipt missing fields: " + ", ".join(missing))
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != expected_checkpoint_hash:
        raise ValueError("preservation receipt checkpoint hash mismatch")
    wandb = payload.get("wandb")
    if not isinstance(wandb, dict) or not wandb.get("receipt_sha256"):
        raise ValueError("preservation receipt requires W&B receipt")
    if not wandb.get("offline_dir") and not (wandb.get("url") or wandb.get("run_path")):
        raise ValueError("preservation receipt requires W&B online receipt or offline dir")
    if not isinstance(payload.get("logs"), list) or not payload["logs"]:
        raise ValueError("preservation receipt requires logs")
    if not isinstance(payload.get("commands"), list) or not payload["commands"]:
        raise ValueError("preservation receipt requires commands")
    if base_dir is not None:
        _validate_artifact_object(base_dir, checkpoint, "checkpoint")
        for section in ("logs", "commands"):
            _validate_artifact_rows(base_dir, payload[section])
        for section in ("inventory", "local_acknowledgement", "evaluation_receipt", "monitor"):
            _validate_artifact_object(base_dir, payload[section], section)
        inventory_files = payload["inventory"].get("files")
        if not isinstance(inventory_files, list) or not any(
            isinstance(row, dict)
            and row.get("path") == checkpoint.get("path")
            and row.get("sha256") == expected_checkpoint_hash
            for row in inventory_files
        ):
            raise ValueError("preservation inventory does not bind the checkpoint")
        _validate_wandb_receipt(base_dir, wandb)
    return dict(payload)


def _validate_artifact_object(base_dir: Path, value, label: str) -> None:
    if not isinstance(value, dict) or "path" not in value or "sha256" not in value:
        raise ValueError(f"preservation receipt requires {label} object")
    _validate_artifact_rows(base_dir, [value])


def _validate_artifact_rows(base_dir: Path, rows: list[dict]) -> None:
    for row in rows:
        if not isinstance(row, dict) or not row.get("path") or not _sha_like(row.get("sha256")):
            raise ValueError("preservation artifact rows require path and sha256")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("preservation artifact paths must be receipt-relative")
        artifact = base_dir / relative
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"preserved artifact is missing: {relative}")
        if _sha256(artifact) != row["sha256"]:
            raise ValueError(f"preserved artifact hash mismatch: {relative}")


def _assert_result_matches_checkpoint(result: dict, checkpoint: CheckpointRef, label: str) -> None:
    if result.get("checkpoint_counter") != checkpoint.counter:
        raise ValueError(f"{label} result counter mismatch")
    if _checkpoint_hash(result) != checkpoint.checkpoint_hash:
        raise ValueError(f"{label} result hash mismatch")


def _has_success_or_time_improvement(current: dict, parent: dict) -> bool:
    current_panels = _panels(current)
    parent_panels = _panels(parent)
    for mode in ("deterministic", "sampled"):
        cur = _mode_row(current_panels, mode, DEVELOPMENT_EPISODES)
        old = _mode_row(parent_panels, mode, DEVELOPMENT_EPISODES)
        if _finite_number(cur.get("successes"), f"{mode}.successes") > _finite_number(old.get("successes"), f"{mode}.parent_successes"):
            return True
        cur_time = _finite_number(cur.get("time_to_reengagement", math.inf), f"{mode}.time_to_reengagement")
        old_time = _finite_number(old.get("time_to_reengagement", math.inf), f"{mode}.parent_time_to_reengagement")
        if cur_time < old_time:
            return True
    return False


def _panels(result: dict) -> dict:
    if not isinstance(result, dict):
        raise TypeError("result must be an object")
    if isinstance(result.get("panels"), dict):
        return result["panels"]
    summary = result.get("summary")
    modes = summary.get("modes") if isinstance(summary, dict) else None
    if not isinstance(modes, dict):
        raise TypeError("result must include panels or summary.modes")
    return {mode: _panel_from_evaluator_mode(row) for mode, row in modes.items()}


def _mode_row(panels: dict, mode: str, episodes: int) -> dict:
    try:
        row = panels[mode]
    except KeyError as exc:
        raise ValueError(f"missing {mode} panel") from exc
    if not isinstance(row, dict):
        raise TypeError(f"{mode} panel must be an object")
    if row.get("episodes") != episodes:
        raise ValueError(f"{mode} panel must contain exactly {episodes} episodes")
    return row


def _panel_from_evaluator_mode(row: dict) -> dict:
    if not isinstance(row, dict):
        raise TypeError("evaluator mode summary must be an object")
    timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
    second_hit_ticks = timing.get("second_hit_ticks") if isinstance(timing.get("second_hit_ticks"), list) else []
    if second_hit_ticks:
        time_to_reengagement = sum(float(value) for value in second_hit_ticks) / len(second_hit_ticks)
    else:
        time_to_reengagement = float(timing.get("average_ticks", math.inf))
    slots = row.get("slots") if isinstance(row.get("slots"), dict) else {}
    return {
        "episodes": row.get("episodes"),
        "successes": row.get("full_sequence_successes"),
        "falls": row.get("falls", 0),
        "invalid": row.get("invalid_or_ambiguous", 0),
        "ambiguous": 0,
        "time_to_reengagement": time_to_reengagement,
        "learner_slots": {
            f"slot{slot}": {
                "episodes": slot_row.get("episodes"),
                "successes": slot_row.get("full_sequence_successes"),
            }
            for slot, slot_row in slots.items()
            if isinstance(slot_row, dict)
        },
    }


def _retention_mode(result: dict, domain: str, mode: str, episodes: int) -> dict:
    try:
        row = result["panels"][domain][mode]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"result must include {domain}/{mode} retention panel") from exc
    if not isinstance(row, dict):
        raise TypeError(f"{domain}/{mode} retention panel must be an object")
    if row.get("episodes") != episodes:
        raise ValueError(f"{domain}/{mode} retention panel must contain exactly {episodes} episodes")
    return row


def _retention_falls(row: dict, domain: str, mode: str, *, parent: bool = False) -> float:
    field = "self_losses" if domain == "contact" else "falls"
    prefix = "parent_" if parent else ""
    return _finite_number(row.get(field, 0), f"{domain}.{mode}.{prefix}{field}")


def _validate_wandb_receipt(base_dir: Path, wandb: dict) -> None:
    receipt_path = wandb.get("receipt_path")
    receipt_sha256 = wandb.get("receipt_sha256")
    _validate_artifact_rows(base_dir, [{"path": receipt_path, "sha256": receipt_sha256}])
    offline_dir = wandb.get("offline_dir")
    if offline_dir:
        relative = _safe_relative_path(offline_dir, "W&B offline directory")
        directory = base_dir / relative
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("preserved W&B offline directory is missing")
        if not any(path.is_file() and not path.is_symlink() for path in directory.rglob("*")):
            raise ValueError("preserved W&B offline directory is empty")


def _safe_relative_path(value, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is required")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be receipt-relative")
    return relative


def _checkpoint_hash(result: dict):
    for value in (
        result.get("checkpoint_hash"),
        (result.get("checkpoint") or {}).get("sha256") if isinstance(result.get("checkpoint"), dict) else None,
        (result.get("checkpoint_hash_receipt") or {}).get("checkpoint_sha256")
        if isinstance(result.get("checkpoint_hash_receipt"), dict)
        else None,
    ):
        if value is not None:
            return value
    return None


def _finite_number(value, label: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sha_like(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
