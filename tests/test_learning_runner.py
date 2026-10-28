"""Tests for the thin grouped-throughput learning runner wrapper."""

from __future__ import annotations

from ai2.paths import source_path

import sys
import types
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class LearningRunnerTests(unittest.TestCase):
    def test_runtime_cuda_probe_and_policy_device_hook_write_proof(self):
        import ai2.training.learning_runner as learning_runner

        with TemporaryDirectory(prefix="runner-cuda-") as tmp:
            output = Path(tmp) / "out"
            output.mkdir()

            class FakeTensor:
                device = "cuda:0"

                def numel(self):
                    return 1

            class FakePolicy:
                def parameters(self):
                    return [FakeTensor()]

            class FakeTorch:
                __version__ = "fake"

                class cuda:
                    @staticmethod
                    def is_available():
                        return True

                @staticmethod
                def empty(*args, **kwargs):
                    return object()

            fake_train_job = types.SimpleNamespace(
                _parameter_count=lambda agent: 1,
                torch=FakeTorch,
            )
            agent = types.SimpleNamespace(policy=FakePolicy())

            cleanup = learning_runner.install_runtime_device_probe(fake_train_job, output)
            try:
                self.assertEqual(fake_train_job._parameter_count(agent), 1)
            finally:
                cleanup()

            proof = output / "runtime_device.json"
            self.assertTrue(proof.is_file())
            self.assertIn("cuda:0", proof.read_text(encoding="utf-8"))

    def test_metadata_hook_records_config_and_input_checkpoint_sha_before_training(self):
        import ai2.training.learning_runner as learning_runner

        with TemporaryDirectory(prefix="runner-meta-") as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            checkpoint = root / "previous.zip"
            metadata = root / "out" / "metadata.json"
            config.write_text("config\n", encoding="utf-8")
            checkpoint.write_text("checkpoint-v1\n", encoding="utf-8")
            writes = []

            def fake_write(path, payload):
                writes.append((Path(path), payload))

            fake_train_job = types.SimpleNamespace(_write_json=fake_write)
            cleanup = learning_runner.install_metadata_probe(fake_train_job, config, checkpoint)
            try:
                fake_train_job._write_json(metadata, {"ppo_controls": {"n_steps": 256}})
            finally:
                cleanup()
            expected_sha = learning_runner.sha256_file(config)
            expected_checkpoint_sha = learning_runner.sha256_file(checkpoint)

        self.assertEqual(writes[0][1]["config_sha256"], expected_sha)
        self.assertEqual(writes[0][1]["input_checkpoint_sha256"], expected_checkpoint_sha)
        self.assertEqual(writes[0][1]["ppo_controls"]["n_steps"], 256)

    def test_metadata_hook_records_fresh_null_checkpoint_sha(self):
        import ai2.training.learning_runner as learning_runner

        writes = []
        fake_train_job = types.SimpleNamespace(_write_json=lambda path, payload: writes.append((Path(path), payload)))

        with TemporaryDirectory(prefix="runner-fresh-meta-") as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            config.write_text("config\n", encoding="utf-8")
            cleanup = learning_runner.install_metadata_probe(fake_train_job, config, None)
            try:
                fake_train_job._write_json(root / "metadata.json", {})
            finally:
                cleanup()

        self.assertIsNone(writes[0][1]["input_checkpoint_sha256"])

    def test_runner_patches_grouped_env_and_cleans_up_while_forwarding_train_args(self):
        import ai2.training.learning_runner as learning_runner

        calls = []

        class FakeTrainJob(types.SimpleNamespace):
            def __init__(self):
                super().__init__()
                self.PPO = object
                self.SubprocVecEnv = "original"
                self._parameter_count = lambda agent: 1
                self._write_json = lambda path, payload: None
                self.torch = types.SimpleNamespace(
                    cuda=types.SimpleNamespace(is_available=lambda: True),
                    empty=lambda *args, **kwargs: object(),
                )

            def build_parser(self):
                import argparse

                parser = argparse.ArgumentParser()
                parser.add_argument("--config")
                parser.add_argument("--phase")
                parser.add_argument("--seed", type=int)
                parser.add_argument("--steps", type=int)
                parser.add_argument("--n-envs", type=int)
                parser.add_argument("--n-steps", type=int)
                parser.add_argument("--batch-size", type=int)
                parser.add_argument("--epochs", type=int)
                parser.add_argument("--device")
                parser.add_argument("--output")
                parser.add_argument("--max-seconds")
                parser.add_argument("--checkpoint")
                return parser

            def run_job(self, args):
                self._parameter_count(types.SimpleNamespace(policy=types.SimpleNamespace(parameters=lambda: [types.SimpleNamespace(device="cuda:0")])))
                calls.append(("run", args.device, args.n_envs, args.n_steps, args.checkpoint))

        fake_train_job = FakeTrainJob()
        cleanup_called = []

        def fake_import(path):
            calls.append(("import", Path(path).name))
            return fake_train_job

        def fake_patch(train_job, args):
            calls.append(("patch", args.group_size, Path(args.grouped_env_path).name))
            train_job.SubprocVecEnv = "grouped"

            def cleanup():
                cleanup_called.append(True)
                train_job.SubprocVecEnv = "original"

            return cleanup

        with TemporaryDirectory(prefix="runner-forward-") as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            config.write_text("config\n", encoding="utf-8")
            argv = [
                "--train-job",
                str(root / "train_job.py"),
                "--grouped-env-path",
                str(source_path(root, "grouped_vec_env.py")),
                "--group-size",
                "4",
                "--config",
                str(config),
                "--phase",
                "0a",
                "--seed",
                "145",
                "--steps",
                "500000",
                "--n-envs",
                "128",
                "--n-steps",
                "256",
                "--batch-size",
                "8192",
                "--epochs",
                "8",
                "--device",
                "cuda",
                "--output",
                str(root / "out"),
                "--max-seconds",
                "1200",
            ]

            with patch.object(learning_runner, "import_train_job_module", side_effect=fake_import), patch.object(
                learning_runner, "patch_grouped_vec_env", side_effect=fake_patch
            ):
                self.assertEqual(learning_runner.main(argv), 0)

        self.assertIn(("patch", 4, "grouped_vec_env.py"), calls)
        self.assertIn(("run", "cuda", 128, 256, None), calls)
        self.assertEqual(cleanup_called, [True])
        self.assertEqual(fake_train_job.SubprocVecEnv, "original")

    def test_runner_refuses_cpu_before_calling_train_job(self):
        import ai2.training.learning_runner as learning_runner

        with patch.object(learning_runner, "import_train_job_module") as import_train:
            with self.assertRaisesRegex(SystemExit, "CUDA"):
                learning_runner.main(["--train-job", "/tmp/train.py", "--device", "cpu"])

        import_train.assert_not_called()

    def test_campaign_learning_uses_learning_runner_and_grouped_throughput_opt_in(self):
        import ai2.runtime.campaign_learning as e6

        with TemporaryDirectory(prefix="e6-wrapper-") as tmp:
            root = Path(tmp)
            runner = source_path(root, "learning_runner.py")
            train_job = root / "train_job.py"
            grouped = source_path(root, "grouped_vec_env.py")
            source = root / "source"
            config = root / "config.yaml"
            for path in (runner, train_job, grouped, config):
                path.write_text("# file\n", encoding="utf-8")
            source.mkdir()

            command = e6.build_train_command(
                python="/venv/bin/python",
                runner=runner,
                train_job=train_job,
                grouped_env_path=grouped,
                source=source,
                config=config,
                spec=e6.RunSpec(net_arch=(512, 512, 256), seed=145),
                chunk_index=1,
                previous_checkpoint=None,
                output=root / "out",
                steps=500_000,
                max_seconds=900,
                controls=e6.TrainControls(n_envs=128, n_steps=256, group_size=4),
            )

        self.assertEqual(command.argv[1], str(runner.resolve()))
        self.assertEqual(command.argv[command.argv.index("--group-size") + 1], "4")
        self.assertEqual(command.argv[command.argv.index("--n-envs") + 1], "128")
        self.assertEqual(command.argv[command.argv.index("--n-steps") + 1], "256")
        self.assertNotIn("--checkpoint", command.argv)
        self.assertEqual(command.metadata["runner"], "learning_runner.py")

    def test_campaign_learning_defaults_remain_stock_until_grouped_checkpoint_parity_resolves(self):
        import ai2.runtime.campaign_learning as e6

        controls = e6.TrainControls()

        self.assertEqual(controls.n_envs, 64)
        self.assertEqual(controls.n_steps, 512)
        self.assertEqual(controls.group_size, 1)


if __name__ == "__main__":
    unittest.main()
