import json
import hashlib
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


def subject():
    import continue_reengagement_campaign

    return continue_reengagement_campaign


def panel(successes=18, falls=1, *, episodes=20, ambiguous=0, invalid=0, time_to_reengagement=10.0, learner_slots=None):
    row = {
        "episodes": episodes,
        "successes": successes,
        "falls": falls,
        "ambiguous": ambiguous,
        "invalid": invalid,
        "time_to_reengagement": time_to_reengagement,
    }
    if learner_slots is not None:
        row["learner_slots"] = learner_slots
    return row


def development_result(det=18, sampled=18, falls=1, *, counter=1, hash_value=None, ambiguous=0, invalid=0, time=10.0):
    return {
        "checkpoint_counter": counter,
        "checkpoint_hash": hash_value or f"{counter:064x}",
        "panels": {
            "deterministic": panel(det, falls, ambiguous=ambiguous, invalid=invalid, time_to_reengagement=time),
            "sampled": panel(sampled, falls, ambiguous=ambiguous, invalid=invalid, time_to_reengagement=time),
        },
    }


def confirmation_result(det=90, sampled=90, *, falls=5, slots=(40, 50), episodes=100):
    return {
        "panels": {
            "deterministic": panel(det, falls, episodes=episodes, learner_slots={"slot0": slots[0], "slot1": slots[1]}),
            "sampled": panel(sampled, falls, episodes=episodes, learner_slots={"slot0": slots[0], "slot1": slots[1]}),
        }
    }


def preservation_payload(checkpoint_hash="a" * 64):
    return {
        "checkpoint": {"sha256": checkpoint_hash},
        "inventory": {"sha256": "b" * 64, "files": [{"path": "final.zip", "sha256": checkpoint_hash}]},
        "local_acknowledgement": {"sha256": "c" * 64},
        "evaluation_receipt": {"sha256": "d" * 64},
        "wandb": {"receipt_sha256": "e" * 64, "offline_dir": "wandb/offline-run"},
        "logs": [{"path": "run.log", "sha256": "f" * 64}],
        "monitor": {"path": "monitor.jsonl", "sha256": "1" * 64},
        "commands": [{"argv": ["python", "train_reengagement_campaign_block.py"], "sha256": "2" * 64}],
    }


def write_preservation_payload(block_dir, checkpoint_hash="a" * 64):
    block_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "inventory.json": b"inventory",
        "ack.json": b"ack",
        "eval.json": b"eval",
        "run.log": b"log",
        "monitor.jsonl": b"monitor",
        "command.json": b"command",
    }
    for name, data in files.items():
        (block_dir / name).write_bytes(data)
    digest = lambda name: hashlib.sha256((block_dir / name).read_bytes()).hexdigest()
    payload = preservation_payload(checkpoint_hash)
    payload["inventory"] = {"path": "inventory.json", "sha256": digest("inventory.json"), "files": [{"path": "final.zip", "sha256": checkpoint_hash}]}
    payload["local_acknowledgement"] = {"path": "ack.json", "sha256": digest("ack.json")}
    payload["evaluation_receipt"] = {"path": "eval.json", "sha256": digest("eval.json")}
    payload["logs"] = [{"path": "run.log", "sha256": digest("run.log")}]
    payload["monitor"] = {"path": "monitor.jsonl", "sha256": digest("monitor.jsonl")}
    payload["commands"] = [{"path": "command.json", "sha256": digest("command.json")}]
    return payload


def test_build_next_cycle_requires_matched_parent_development_rerun_and_exactly_one_cycle():
    m = subject()
    parent = m.CheckpointRef(name="parent", checkpoint_hash="a" * 64, counter=10)
    current = m.CheckpointRef(name="current", checkpoint_hash="b" * 64, counter=11)
    parent_rerun = m.MatchedDevelopmentRerun(parent=parent, result=development_result(counter=10, hash_value=parent.checkpoint_hash))
    current_result = development_result(counter=11, hash_value=current.checkpoint_hash)

    step = m.build_next_cycle(current=current, matched_parent_rerun=parent_rerun, current_result=current_result, cycle_index=2)

    assert step.parent_checkpoint == current
    assert step.cycle_index == 2
    assert step.rollouts == m.ROLLOUTS_PER_CYCLE
    assert step.schedule == "R4E1N"

    with pytest.raises(ValueError, match="matched parent"):
        m.build_next_cycle(current=current, matched_parent_rerun=None, current_result=current_result, cycle_index=2)
    with pytest.raises(ValueError, match="one cycle"):
        m.build_next_cycle(current=current, matched_parent_rerun=parent_rerun, current_result=current_result, cycle_index=4)


def test_development_mastery_requires_two_distinct_increasing_qualified_checkpoints():
    m = subject()
    first = m.development_gate(development_result(counter=3, hash_value="3" * 64))
    second = m.development_gate(development_result(counter=4, hash_value="4" * 64))
    assert first.eligible
    assert second.eligible
    assert m.development_mastery([first, second]).mastered

    duplicate_counter = m.development_gate(development_result(counter=3, hash_value="5" * 64))
    assert not m.development_mastery([first, duplicate_counter]).mastered
    weak = m.development_gate(development_result(det=18, sampled=17, counter=5))
    assert not weak.eligible
    bad_evidence = m.development_gate(development_result(counter=6, ambiguous=1))
    assert bad_evidence.stop
    assert "invalid_or_ambiguous" in bad_evidence.reason


def test_development_gate_consumes_real_task7_evaluator_schema_without_invented_panels():
    m = subject()
    result = {
        "checkpoint_counter": 7,
        "checkpoint_hash_receipt": {"checkpoint_sha256": "7" * 64},
        "summary": {
            "modes": {
                "deterministic": {
                    "episodes": 20,
                    "full_sequence_successes": 18,
                    "falls": 1,
                    "invalid_or_ambiguous": 0,
                    "timing": {"second_hit_ticks": [9, 11]},
                    "slots": {
                        "0": {"full_sequence_successes": 9},
                        "1": {"full_sequence_successes": 9},
                    },
                },
                "sampled": {
                    "episodes": 20,
                    "full_sequence_successes": 19,
                    "falls": 0,
                    "invalid_or_ambiguous": 0,
                    "timing": {"average_ticks": 12},
                    "slots": {
                        "0": {"full_sequence_successes": 10},
                        "1": {"full_sequence_successes": 9},
                    },
                },
            }
        },
    }

    verdict = m.development_gate(result)

    assert verdict.eligible
    assert not verdict.stop
    assert verdict.checkpoint_hash == "7" * 64
    assert verdict.components["deterministic"]["time_to_reengagement"] == 10.0


def test_confirmation_gate_requires_100_case_modes_falls_and_per_slot_successes():
    m = subject()
    assert m.confirmation_gate(confirmation_result()).eligible
    assert not m.confirmation_gate(confirmation_result(det=89)).eligible
    assert not m.confirmation_gate(confirmation_result(falls=6)).eligible
    assert not m.confirmation_gate(confirmation_result(slots=(39, 50))).eligible
    with pytest.raises(ValueError, match="100 episodes"):
        m.confirmation_gate(confirmation_result(episodes=99))


def test_retention_alarms_pause_on_matched_parent_regression_thresholds():
    m = subject()
    parent = development_result(det=20, sampled=20, falls=0)
    assert not m.retention_alarm(development_result(det=18, sampled=18, falls=2), parent).pause
    lost = m.retention_alarm(development_result(det=17, sampled=20, falls=0), parent)
    assert lost.pause
    assert "lost_successes" in lost.reason
    falls = m.retention_alarm(development_result(det=20, sampled=20, falls=3), parent)
    assert falls.pause
    assert "added_falls" in falls.reason


def test_confirmation_retention_allows_only_five_percentage_points_success_loss_and_five_percent_falls():
    m = subject()
    parent = confirmation_result(det=95, sampled=96, falls=0)
    assert not m.confirmation_retention_alarm(confirmation_result(det=90, sampled=91, falls=5), parent).pause
    assert m.confirmation_retention_alarm(confirmation_result(det=89, sampled=91, falls=5), parent).pause
    assert m.confirmation_retention_alarm(confirmation_result(det=90, sampled=91, falls=6), parent).pause


def test_stop_conditions_include_no_improvement_budget_deadline_and_numerical_failure():
    m = subject()
    tracker = m.NoImprovementTracker()
    parent = development_result(det=18, sampled=18, time=10.0)
    same = development_result(det=18, sampled=18, time=10.0)
    assert not tracker.update(same, parent).stop
    assert not tracker.update(same, parent).stop
    stopped = tracker.update(same, parent)
    assert stopped.stop
    assert "no_improvement" in stopped.reason

    assert m.stop_decision(budget_remaining_dollars=None, deadline_remaining_seconds=100).stop
    assert m.stop_decision(budget_remaining_dollars=1.0, deadline_remaining_seconds=0).stop
    numerical = m.development_gate(development_result(counter=8))
    corrupted = dict(numerical.raw)
    corrupted["panels"]["sampled"]["successes"] = float("nan")
    assert m.development_gate(corrupted).stop


def test_preservation_receipt_is_required_before_reuse_and_written_atomically_without_overwrite():
    m = subject()
    checkpoint_hash = "a" * 64
    with TemporaryDirectory() as directory:
        root = Path(directory)
        block = root / "block"
        receipt = m.preserve_receipt(block, write_preservation_payload(block, checkpoint_hash), expected_checkpoint_hash=checkpoint_hash)
        assert receipt["checkpoint"]["sha256"] == checkpoint_hash
        assert (block / "preservation_receipt.json").is_file()
        with pytest.raises(FileExistsError):
            m.preserve_receipt(block, write_preservation_payload(block, "b" * 64), expected_checkpoint_hash="b" * 64)
        assert m.verify_reuse_receipt(block, expected_checkpoint_hash=checkpoint_hash)["checkpoint"]["sha256"] == checkpoint_hash

        missing = root / "missing"
        missing.mkdir()
        with pytest.raises(FileNotFoundError):
            m.verify_reuse_receipt(missing, expected_checkpoint_hash=checkpoint_hash)


def test_preservation_receipt_accepts_online_wandb_and_hashes_real_files():
    m = subject()
    checkpoint_hash = "a" * 64
    with TemporaryDirectory() as directory:
        block = Path(directory) / "block"
        payload = write_preservation_payload(block, checkpoint_hash)
        payload["wandb"] = {
            "receipt_sha256": "e" * 64,
            "url": "https://wandb.ai/entity/project/runs/run-id",
        }
        assert m.preserve_receipt(block, payload, expected_checkpoint_hash=checkpoint_hash)["wandb"]["url"]

    with TemporaryDirectory() as directory:
        block = Path(directory) / "block"
        payload = write_preservation_payload(block, checkpoint_hash)
        payload["inventory"] = {"sha256": "b" * 64, "files": [{"path": "final.zip", "sha256": checkpoint_hash}]}
        with pytest.raises(ValueError, match="inventory object"):
            m.preserve_receipt(block, payload, expected_checkpoint_hash=checkpoint_hash)

    with TemporaryDirectory() as directory:
        block = Path(directory) / "block"
        payload = write_preservation_payload(block, checkpoint_hash)
        payload["logs"][0]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="hash mismatch"):
            m.preserve_receipt(block, payload, expected_checkpoint_hash=checkpoint_hash)


def test_confirmation_bank_is_blocked_until_two_development_checkpoints_qualify():
    m = subject()
    first = m.development_gate(development_result(counter=1, hash_value="1" * 64))
    second = m.development_gate(development_result(counter=2, hash_value="2" * 64))
    assert not m.confirmation_allowed([first]).allowed
    assert m.confirmation_allowed([first, second]).allowed
    assert not m.confirmation_allowed([first, replace(second, checkpoint_hash=first.checkpoint_hash)]).allowed


def test_budget_ceiling_tracks_actual_spend_without_inferred_remaining_budget():
    m = subject()
    budget = m.CampaignBudget(ceiling_dollars=12.0, actual_spend_dollars=11.5)
    assert budget.remaining() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="actual spend"):
        m.CampaignBudget(ceiling_dollars=12.0, actual_spend_dollars=None).remaining()
    with pytest.raises(ValueError, match="ceiling"):
        m.CampaignBudget(ceiling_dollars=12.01, actual_spend_dollars=1.0)


def test_cycle_summary_json_is_immutable_and_atomic():
    m = subject()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "summary.json"
        payload = {"status": "paused", "reason": "retention_alarm"}
        m.atomic_output(path, payload)
        assert json.loads(path.read_text()) == payload
        with pytest.raises(FileExistsError):
            m.atomic_output(path, {"status": "overwrite"})
