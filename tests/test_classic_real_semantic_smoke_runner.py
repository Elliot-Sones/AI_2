"""Artifact contract tests for the real classic combat semantic smoke."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "classic_real_semantic_smoke.py"


def test_runner_writes_hash_bound_real_dependency_artifacts(tmp_path):
    output_dir = tmp_path / "classic-real-smoke"

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--output-dir",
            str(output_dir),
            "--seeds",
            "101",
            "202",
            "303",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    receipt_path = output_dir / "receipt.json"
    events_path = output_dir / "events.jsonl"
    assert receipt_path.exists()
    assert events_path.exists()

    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema"] == "classic-real-semantic-smoke/v1"
    assert receipt["python"] == "3.12"
    assert receipt["real_dependency_checks"]["WarehouseBrawl"] is True
    assert receipt["real_dependency_checks"]["stubbed_dependency_modules"] == []
    assert receipt["time_limit"]["configured_steps"] == 2700
    assert receipt["time_limit"]["fps"] == 30
    assert receipt["time_limit"]["truncation_step"] == 2700
    assert receipt["time_limit"]["reset_steps_after_truncation"] == 0
    assert [item["seed"] for item in receipt["seed_traces"]] == [101, 202, 303]
    assert len({item["trace_hash"] for item in receipt["seed_traces"]}) == 3
    assert all(item["replay_trace_hash"] == item["trace_hash"] for item in receipt["seed_traces"])
    assert receipt["reward_semantics"]["learner_win_reward"] > 0
    assert receipt["reward_semantics"]["learner_knockout_reward"] < 0
    assert receipt["reward_semantics"]["opponent_knockout_reward"] > 0
    assert receipt["reward_semantics"]["attack_state_aggression_reward"] > 0
    assert receipt["reward_semantics"]["reward_leakage_absent"] is True

    event_lines = events_path.read_text().splitlines()
    assert event_lines
    assert receipt["events_jsonl_sha256"] == hashlib.sha256(events_path.read_bytes()).hexdigest()


def test_runner_fails_closed_when_output_path_exists(tmp_path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    result = subprocess.run(
        [sys.executable, str(RUNNER), "--output-dir", str(output_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing existing output path" in result.stderr


def test_runner_fails_closed_when_output_path_is_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    output_dir = tmp_path / "link"
    output_dir.symlink_to(target, target_is_directory=True)

    result = subprocess.run(
        [sys.executable, str(RUNNER), "--output-dir", str(output_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing symlink output path" in result.stderr
