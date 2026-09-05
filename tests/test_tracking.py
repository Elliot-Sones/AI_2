"""Tests for the Weights & Biases tracking helpers (offline, no network)."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tracking  # noqa: E402


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def offline_env(tmp: str) -> dict:
    return {
        "WANDB_MODE": "offline",
        "WANDB_DIR": tmp,
        "WANDB_SILENT": "true",
        "WANDB_CONSOLE": "off",
        "WANDB_NOTES": "",
        "WANDB_TAGS": "",
        "WANDB_NAME": "",
        "WANDB_RUN_GROUP": "",
    }


def base_params(**wandb_overrides) -> dict:
    cfg = {"enabled": True, "project": "ai2-test", "entity": None, "tags": ["cfgtag"]}
    cfg.update(wandb_overrides)
    return {
        "wandb": cfg,
        "curriculum": {"phase": "1"},
        "ppo_settings": {"time_steps": 10, "model_checkpoint": "0"},
        "environment_settings": {"n_envs": 2},
    }


class LoginDetectionTest(unittest.TestCase):
    def test_has_login_from_env_key_or_netrc_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "no-netrc")
            self.assertFalse(tracking.has_login(environ={}, netrc_path=missing))
            self.assertTrue(tracking.has_login(environ={"WANDB_API_KEY": "k"}, netrc_path=missing))
            netrc = os.path.join(tmp, "netrc")
            with open(netrc, "w") as f:
                f.write("machine api.wandb.ai\n  login user\n  password key\n")
            self.assertTrue(tracking.has_login(environ={}, netrc_path=netrc))

    def test_tracking_enabled_requires_config_flag_and_login_or_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "no-netrc")
            off = base_params(enabled=False)
            on = base_params(enabled=True)
            self.assertFalse(tracking.tracking_enabled(off, environ={"WANDB_API_KEY": "k"}, netrc_path=missing))
            self.assertFalse(tracking.tracking_enabled(on, environ={}, netrc_path=missing))
            self.assertTrue(tracking.tracking_enabled(on, environ={"WANDB_MODE": "offline"}, netrc_path=missing))
            self.assertTrue(tracking.tracking_enabled(on, environ={"WANDB_API_KEY": "k"}, netrc_path=missing))
            self.assertFalse(tracking.tracking_enabled(on, environ={"WANDB_API_KEY": "k", "WANDB_MODE": "disabled"}, netrc_path=missing))


class DisabledTrackingTest(unittest.TestCase):
    def test_helpers_are_noops_without_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, offline_env(tmp)):
                run = tracking.init_run(base_params(enabled=False), phase_key="1", kind="combat", results_dir=tmp)
                self.assertIsNone(run)
                self.assertFalse(tracking.log_video(os.path.join(tmp, "missing.mp4"), step=1))
                self.assertEqual(tracking.log_eval_table([], step=1), 0)
                self.assertIsNone(tracking.log_checkpoint(os.path.join(tmp, "missing.zip"), step=1))
                self.assertFalse(tracking.append_notes("nothing"))
                tracking.log_summary(final_steps=1)
                tracking.finish()
                self.assertFalse(os.path.exists(os.path.join(tmp, "wandb_run.json")))


class OfflineRunTest(unittest.TestCase):
    def test_init_records_run_with_notes_tags_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = offline_env(tmp)
            env.update({"WANDB_NOTES": "Goal: smoke test", "WANDB_TAGS": "smoke,vast", "WANDB_RUN_GROUP": "run-42"})
            cfg_path = os.path.join(tmp, "config.yaml")
            with open(cfg_path, "w") as f:
                f.write("curriculum:\n  phase: '1'\n")
            with mock.patch.dict(os.environ, env):
                run = tracking.init_run(
                    base_params(), phase_key="1", kind="combat", results_dir=tmp,
                    config_path=cfg_path, extra_config={"device": "cpu"},
                )
                try:
                    self.assertIsNotNone(run)
                    self.assertEqual(run.notes, "Goal: smoke test")
                    for tag in ("phase-1", "combat", "smoke", "vast", "cfgtag"):
                        self.assertIn(tag, run.tags)
                    self.assertNotIn("notes-missing", run.tags)
                    self.assertEqual(run.group, "run-42")
                    self.assertEqual(run.config["phase"], "1")
                    self.assertEqual(run.config["kind"], "combat")
                    self.assertEqual(run.config["device"], "cpu")
                    self.assertEqual(run.config["environment_settings"]["n_envs"], 2)
                    record = read_json(os.path.join(tmp, "wandb_run.json"))
                    self.assertEqual(record["id"], run.id)
                    self.assertEqual(record["project"], "ai2-test")
                    self.assertEqual(record["notes"], "Goal: smoke test")
                    self.assertIn("path", record)
                    history = read_jsonl(os.path.join(tmp, "wandb_runs.jsonl"))
                    self.assertEqual(history[-1]["id"], run.id)
                finally:
                    tracking.finish()

    def test_missing_notes_are_flagged_and_auto_described(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, offline_env(tmp)):
                run = tracking.init_run(base_params(), phase_key="0a", kind="navigation", results_dir=tmp)
                try:
                    self.assertIsNotNone(run)
                    self.assertIn("notes-missing", run.tags)
                    self.assertIn("phase 0a", run.notes)
                    self.assertIn("navigation", run.notes)
                finally:
                    tracking.finish()

    def test_append_notes_extends_live_run_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = offline_env(tmp)
            env["WANDB_NOTES"] = "Goal: A"
            with mock.patch.dict(os.environ, env):
                run = tracking.init_run(base_params(), phase_key="1", kind="combat", results_dir=tmp)
                try:
                    self.assertTrue(tracking.append_notes("Outcome: B"))
                    self.assertIn("Goal: A", run.notes)
                    self.assertIn("Outcome: B", run.notes)
                    record = read_json(os.path.join(tmp, "wandb_run.json"))
                    self.assertIn("Outcome: B", record["notes"])
                finally:
                    tracking.finish()

    def test_log_checkpoint_builds_model_artifact_with_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "phase1_model_1000_steps.zip")
            with open(ckpt, "wb") as f:
                f.write(b"PK\x05\x06" + b"\x00" * 18)
            with mock.patch.dict(os.environ, offline_env(tmp)):
                tracking.init_run(base_params(), phase_key="1", kind="combat", results_dir=tmp)
                try:
                    artifact = tracking.log_checkpoint(ckpt, step=1000, aliases=["final"])
                    self.assertIsNotNone(artifact)
                    self.assertEqual(artifact.type, "model")
                    self.assertIn("phase1_model_1000_steps.zip", artifact.manifest.entries)
                    self.assertEqual(artifact.metadata["step"], 1000)
                    self.assertIsNone(tracking.log_checkpoint(os.path.join(tmp, "missing.zip"), step=1))
                finally:
                    tracking.finish()

    def test_log_video_and_eval_table_report_what_was_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "demo.mp4")
            with open(video, "wb") as f:
                f.write(b"\x00" * 64)
            rows = [
                {"opponent": "ConstantAgent", "win_rate": 70.0, "wins": 7, "losses": 3, "draws": 0, "avg_damage": 194.0},
                {"opponent": "BasedAgent", "win_rate": 20.0, "wins": 2, "losses": 8, "draws": 0, "avg_damage": 97.0},
            ]
            with mock.patch.dict(os.environ, offline_env(tmp)):
                tracking.init_run(base_params(), phase_key="3", kind="combat", results_dir=tmp)
                try:
                    self.assertTrue(tracking.log_video(video, step=5, caption="demo"))
                    self.assertFalse(tracking.log_video(os.path.join(tmp, "missing.mp4"), step=5))
                    self.assertEqual(tracking.log_eval_table(rows, step=5), 2)
                finally:
                    tracking.finish()

    def test_finish_clears_active_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, offline_env(tmp)):
                tracking.init_run(base_params(), phase_key="1", kind="combat", results_dir=tmp)
                tracking.finish()
                self.assertIsNone(tracking.active_run())
                tracking.finish()  # second finish must be harmless

    def test_cli_info_prints_run_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = offline_env(tmp)
            env["WANDB_NOTES"] = "Goal: cli"
            with mock.patch.dict(os.environ, env):
                run = tracking.init_run(base_params(), phase_key="1", kind="combat", results_dir=tmp)
                run_id = run.id
                tracking.finish()
            result = subprocess.run(
                [sys.executable, str(ROOT / "tracking.py"), "info", "--run", os.path.join(tmp, "wandb_run.json")],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(run_id, result.stdout)
            self.assertIn("Goal: cli", result.stdout)


if __name__ == "__main__":
    unittest.main()
