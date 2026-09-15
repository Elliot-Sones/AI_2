import json
import os
import random
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_classic_reward_semantics import train  # noqa: E402


class FakeAgent:
    def __init__(self):
        self.saved_paths = []
        self.on_save = None

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(f"fake checkpoint {len(self.saved_paths)}".encode("utf-8"))
        self.saved_paths.append(path)
        if self.on_save is not None:
            self.on_save(path)


class FakeOpponent:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.env = None

    def get_env_info(self, env):
        self.env = env


class FakeEnv:
    action_space = object()
    observation_space = object()
    obs_helper = object()
    act_helper = object()


def make_parent_handler(tmp, *, save_freq=100, max_saved=5, name_prefix="rl_model"):
    return train.ClassicSelfPlaySaveHandler(
        agent=None,
        save_freq=save_freq,
        max_saved=max_saved,
        run_name="self_play_run",
        save_path=tmp,
        name_prefix=name_prefix,
        mode=train.SaveHandlerMode.FORCE,
    )


def make_worker_handler(tmp, *, max_saved=5, name_prefix="rl_model"):
    return train.VectorizedSaveHandler(
        agent=None,
        save_freq=100,
        max_saved=max_saved,
        run_name="self_play_run",
        save_path=tmp,
        name_prefix=name_prefix,
        mode=train.SaveHandlerMode.FORCE,
    )


def write_manifest(tmp, entries, *, version=None):
    archive_dir = os.path.join(tmp, "self_play_run")
    os.makedirs(archive_dir, exist_ok=True)
    with open(os.path.join(archive_dir, train.SELF_PLAY_ARCHIVE_MANIFEST), "w", encoding="utf-8") as f:
        json.dump({"version": train.SELF_PLAY_ARCHIVE_VERSION if version is None else version, "checkpoints": entries}, f)


class SelfPlayArchiveContractTest(unittest.TestCase):
    def setUp(self):
        self._previous_monitor = train.environment_agent_module.Monitor
        train.environment_agent_module.Monitor = type("FakeMonitor", (), {})

    def tearDown(self):
        train.environment_agent_module.Monitor = self._previous_monitor

    def test_worker_constructed_before_parent_publish_refreshes_new_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-refresh-") as tmp:
            worker = make_worker_handler(tmp)
            self.assertEqual(worker.history, [])

            parent = make_parent_handler(tmp)
            parent.agent = FakeAgent()
            parent.num_timesteps = 100
            self.assertTrue(parent.process())

            selected = worker.get_random_model_path()
            self.assertIsNotNone(selected)
            self.assertTrue(selected.endswith("rl_model_100_steps.zip"))
            self.assertEqual(worker.history, [selected])

    def test_worker_force_mode_is_read_only_and_never_deletes_archive(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-readonly-") as tmp:
            parent = make_parent_handler(tmp)
            parent.agent = FakeAgent()
            parent.num_timesteps = 100
            parent.process()
            checkpoint = parent.history[-1]

            worker = make_worker_handler(tmp)
            worker.process()
            worker.save_agent()

            self.assertTrue(os.path.exists(checkpoint))
            self.assertEqual(worker.history, [checkpoint])

    def test_parent_atomic_publish_keeps_partial_checkpoint_unselectable(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-atomic-") as tmp:
            worker = make_worker_handler(tmp)
            parent = make_parent_handler(tmp)
            agent = FakeAgent()
            seen_during_save = []

            def inspect_during_save(path):
                seen_during_save.extend(worker.refresh_history())
                self.assertTrue(path.endswith(".zip"))
                self.assertIn(".tmp-", os.path.basename(path))

            agent.on_save = inspect_during_save
            parent.agent = agent
            parent.num_timesteps = 100
            self.assertTrue(parent.process())

            self.assertEqual(seen_during_save, [])
            self.assertEqual(worker.refresh_history(), parent.history)

    def test_parent_save_cadence_uses_absolute_timesteps_without_duplicates(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-cadence-") as tmp:
            parent = make_parent_handler(tmp, save_freq=100)
            parent.agent = FakeAgent()

            for step in [48, 96]:
                parent.num_timesteps = step
                self.assertFalse(parent.process())
            parent.num_timesteps = 100
            self.assertTrue(parent.process())
            parent.num_timesteps = 100
            self.assertFalse(parent.process())
            parent.num_timesteps = 149
            self.assertFalse(parent.process())

            self.assertEqual([os.path.basename(path) for path in parent.history], ["rl_model_100_steps.zip"])
            self.assertEqual(len(parent.agent.saved_paths), 1)

    def test_initial_and_final_snapshots_are_explicit_and_not_duplicated(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-initial-final-") as tmp:
            parent = make_parent_handler(tmp, save_freq=100)
            parent.agent = FakeAgent()
            parent.num_timesteps = 0
            self.assertTrue(parent.save_initial_snapshot())
            self.assertFalse(parent.save_initial_snapshot())
            parent.num_timesteps = 250
            self.assertTrue(parent.save_final_snapshot())

            self.assertEqual(
                [os.path.basename(path) for path in parent.history],
                ["rl_model_0_steps.zip", "rl_model_250_steps.zip"],
            )

    def test_selection_record_includes_requested_realized_checkpoint_hash_and_reason(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-selection-") as tmp:
            parent = make_parent_handler(tmp, save_freq=100)
            parent.agent = FakeAgent()
            parent.num_timesteps = 100
            parent.process()

            worker = make_worker_handler(tmp)
            telemetry = train.OpponentSelectionTelemetry({"self_play": 1.0}, "unit")
            self_play = train.SelfPlayRandom(partial(FakeOpponent))
            cfg = train.TelemetryOpponentsCfg(opponents={"self_play": (1.0, self_play)}, telemetry=telemetry)
            fake_env = FakeEnv()
            cfg.env = fake_env
            self_play.env = fake_env
            self_play.save_handler = worker
            random.seed(11)

            opponent = cfg.on_env_reset()
            record = telemetry.records[0]

            self.assertIsInstance(opponent, FakeOpponent)
            self.assertEqual(record["requested_opponent"], "self_play")
            self.assertEqual(record["realized_opponent_class"], "FakeOpponent")
            self.assertEqual(record["checkpoint_step"], 100)
            self.assertEqual(record["checkpoint_archive_version"], train.SELF_PLAY_ARCHIVE_VERSION)
            self.assertEqual(record["selection_reason"], "selected_checkpoint")
            self.assertIsNotNone(record["checkpoint_path_sha256"])
            self.assertIsNotNone(record["checkpoint_content_sha256"])
            self.assertIsNone(record["fallback_reason"])

    def test_corrupt_or_incompatible_checkpoint_fails_closed_with_explicit_reason(self):
        def broken_loader(file_path=None):
            raise RuntimeError("not a compatible PPO checkpoint")

        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-corrupt-") as tmp:
            parent = make_parent_handler(tmp, save_freq=100)
            parent.agent = FakeAgent()
            parent.num_timesteps = 100
            parent.process()

            worker = make_worker_handler(tmp)
            telemetry = train.OpponentSelectionTelemetry({"self_play": 1.0}, "unit")
            self_play = train.SelfPlayRandom(partial(broken_loader))
            cfg = train.TelemetryOpponentsCfg(opponents={"self_play": (1.0, self_play)}, telemetry=telemetry)
            fake_env = FakeEnv()
            cfg.env = fake_env
            self_play.env = fake_env
            self_play.save_handler = worker

            opponent = cfg.on_env_reset()
            record = telemetry.records[0]

            self.assertIsInstance(opponent, train.ConstantAgent)
            self.assertEqual(record["realized_opponent_class"], "ConstantAgent")
            self.assertEqual(record["selection_reason"], "selected_checkpoint")
            self.assertEqual(record["fallback_reason"], "checkpoint_load_failed:RuntimeError")
            self.assertEqual(record["fallback_class"], "ConstantAgent")

    def test_incompatible_manifest_fails_closed_even_if_zip_exists(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-bad-manifest-") as tmp:
            archive_dir = os.path.join(tmp, "self_play_run")
            os.makedirs(archive_dir)
            checkpoint_path = os.path.join(archive_dir, "rl_model_100_steps.zip")
            with open(checkpoint_path, "wb") as f:
                f.write(b"checkpoint")
            with open(os.path.join(archive_dir, train.SELF_PLAY_ARCHIVE_MANIFEST), "w", encoding="utf-8") as f:
                json.dump({"version": 999, "checkpoints": [{"filename": "rl_model_100_steps.zip"}]}, f)

            worker = make_worker_handler(tmp)

            self.assertEqual(worker.history, [])
            self.assertIsNone(worker.get_random_model_path())
            self.assertEqual(worker.selection_reason, "incompatible_manifest_version")

    def test_manifest_rejects_path_escapes_and_symlink_escapes(self):
        cases = [
            ("absolute", "/tmp/rl_model_100_steps.zip", "invalid_manifest_path"),
            ("parent", "../rl_model_100_steps.zip", "invalid_manifest_path"),
            ("separator", "nested/rl_model_100_steps.zip", "invalid_manifest_path"),
            ("backslash", "nested\\rl_model_100_steps.zip", "invalid_manifest_path"),
        ]
        for _, filename, reason in cases:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory(prefix="ai2-selfplay-escape-") as tmp:
                write_manifest(tmp, [{"filename": filename, "content_sha256": "0" * 64}])

                worker = make_worker_handler(tmp)

                self.assertEqual(worker.history, [])
                self.assertIsNone(worker.get_random_model_path())
                self.assertEqual(worker.selection_reason, reason)

        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-symlink-") as tmp:
            outside_fd, outside_path = tempfile.mkstemp(prefix="ai2-outside-checkpoint-", suffix=".zip")
            os.close(outside_fd)
            try:
                with open(outside_path, "wb") as f:
                    f.write(b"outside")
                archive_dir = os.path.join(tmp, "self_play_run")
                os.makedirs(archive_dir)
                link_path = os.path.join(archive_dir, "rl_model_100_steps.zip")
                os.symlink(outside_path, link_path)
                write_manifest(tmp, [{
                    "filename": "rl_model_100_steps.zip",
                    "content_sha256": train._sha256_file(outside_path),
                }])

                worker = make_worker_handler(tmp)

                self.assertEqual(worker.history, [])
                self.assertIsNone(worker.get_random_model_path())
                self.assertEqual(worker.selection_reason, "manifest_path_escape")
            finally:
                os.remove(outside_path)

    def test_manifest_requires_valid_matching_sha_for_every_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-invalid-sha-") as tmp:
            archive_dir = os.path.join(tmp, "self_play_run")
            os.makedirs(archive_dir)
            with open(os.path.join(archive_dir, "rl_model_100_steps.zip"), "wb") as f:
                f.write(b"checkpoint")
            write_manifest(tmp, [{"filename": "rl_model_100_steps.zip", "content_sha256": "not-a-sha"}])

            worker = make_worker_handler(tmp)

            self.assertEqual(worker.history, [])
            self.assertIsNone(worker.get_random_model_path())
            self.assertEqual(worker.selection_reason, "invalid_manifest_hash")

        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-sha-mismatch-") as tmp:
            archive_dir = os.path.join(tmp, "self_play_run")
            os.makedirs(archive_dir)
            with open(os.path.join(archive_dir, "rl_model_100_steps.zip"), "wb") as f:
                f.write(b"checkpoint")
            write_manifest(tmp, [{"filename": "rl_model_100_steps.zip", "content_sha256": "0" * 64}])

            worker = make_worker_handler(tmp)

            self.assertEqual(worker.history, [])
            self.assertIsNone(worker.get_random_model_path())
            self.assertEqual(worker.selection_reason, "checkpoint_hash_mismatch")

    def test_missing_archive_reports_empty_archive(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-missing-") as tmp:
            worker = make_worker_handler(tmp)

            self.assertEqual(worker.history, [])
            self.assertIsNone(worker.get_random_model_path())
            self.assertEqual(worker.selection_reason, "empty_archive")

    def test_empty_manifest_does_not_scan_stale_zip_files(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-empty-manifest-") as tmp:
            archive_dir = os.path.join(tmp, "self_play_run")
            os.makedirs(archive_dir)
            with open(os.path.join(archive_dir, "rl_model_100_steps.zip"), "wb") as f:
                f.write(b"stale checkpoint")
            with open(os.path.join(archive_dir, train.SELF_PLAY_ARCHIVE_MANIFEST), "w", encoding="utf-8") as f:
                json.dump({"version": train.SELF_PLAY_ARCHIVE_VERSION, "checkpoints": []}, f)

            worker = make_worker_handler(tmp)

            self.assertEqual(worker.history, [])
            self.assertIsNone(worker.get_random_model_path())
            self.assertEqual(worker.selection_reason, "empty_archive")

    def test_manifest_lists_only_published_retained_checkpoints(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-manifest-") as tmp:
            parent = make_parent_handler(tmp, save_freq=100, max_saved=2)
            parent.agent = FakeAgent()
            for step in [100, 200, 300]:
                parent.num_timesteps = step
                parent.process()

            manifest_path = os.path.join(tmp, "self_play_run", train.SELF_PLAY_ARCHIVE_MANIFEST)
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["version"], train.SELF_PLAY_ARCHIVE_VERSION)
            self.assertEqual(
                [entry["filename"] for entry in manifest["checkpoints"]],
                ["rl_model_200_steps.zip", "rl_model_300_steps.zip"],
            )
            self.assertFalse(os.path.exists(os.path.join(tmp, "self_play_run", "rl_model_100_steps.zip")))

    def test_parent_force_construction_is_noninteractive_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-no-force-delete-") as tmp:
            archive_dir = os.path.join(tmp, "self_play_run")
            os.makedirs(archive_dir)
            sentinel = os.path.join(archive_dir, "do-not-delete.txt")
            with open(sentinel, "wb") as f:
                f.write(b"keep")

            with mock.patch("builtins.input", side_effect=AssertionError("must not prompt")):
                parent = make_parent_handler(tmp)

            self.assertTrue(os.path.exists(sentinel))
            self.assertEqual(parent.history, [])

    def test_phase_specific_prefix_prevents_initial_snapshot_collisions(self):
        with tempfile.TemporaryDirectory(prefix="ai2-selfplay-phase-prefix-") as tmp:
            phase1_prefix = train._classic_self_play_name_prefix(SimpleNamespace(phase_key=1))
            phase2_prefix = train._classic_self_play_name_prefix(SimpleNamespace(phase_key=2))
            phase1 = make_parent_handler(tmp, name_prefix=phase1_prefix)
            phase1.agent = FakeAgent()
            phase1.num_timesteps = 0
            self.assertTrue(phase1.save_initial_snapshot())

            phase2 = make_parent_handler(tmp, name_prefix=phase2_prefix)
            phase2.agent = FakeAgent()
            phase2.num_timesteps = 0
            self.assertTrue(phase2.save_initial_snapshot())

            self.assertNotEqual(phase1_prefix, phase2_prefix)
            self.assertEqual([os.path.basename(path) for path in phase1.history], [f"{phase1_prefix}_0_steps.zip"])
            self.assertEqual([os.path.basename(path) for path in phase2.history], [f"{phase2_prefix}_0_steps.zip"])
            self.assertTrue(os.path.exists(os.path.join(tmp, "self_play_run", f"{phase1_prefix}_0_steps.zip")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "self_play_run", f"{phase2_prefix}_0_steps.zip")))


if __name__ == "__main__":
    unittest.main()
