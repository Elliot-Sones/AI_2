
from ai2.paths import source_path
import importlib
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FakePolicy:
    def __init__(self, device="cuda:0"):
        self.device = device


class FakeModel:
    def __init__(self, *, model_device="cuda", policy_device="cuda:0", num_timesteps=1234):
        self.device = model_device
        self.policy = FakePolicy(policy_device)
        self.num_timesteps = num_timesteps


class RecordCampaignNavigationTests(unittest.TestCase):
    def setUp(self):
        self.module = self.import_module()

    def import_module(self):
        sys.modules.pop("record_campaign_navigation", None)
        return importlib.import_module('tools.visualization.record_campaign_navigation')

    def test_module_import_does_not_require_numpy(self):
        sys.modules.pop("record_campaign_navigation", None)
        saved_numpy = sys.modules.get("numpy", None)
        had_numpy = "numpy" in sys.modules
        sys.modules["numpy"] = None
        try:
            module = importlib.import_module('tools.visualization.record_campaign_navigation')
        finally:
            sys.modules.pop("record_campaign_navigation", None)
            if had_numpy:
                sys.modules["numpy"] = saved_numpy
            else:
                sys.modules.pop("numpy", None)

        self.assertTrue(callable(module.orient_frame))

    def test_orient_frame_matches_legacy_golden_orientation(self):
        import numpy as np

        frame = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            ],
            dtype=np.uint8,
        )
        expected = np.array(
            [
                [[1, 2, 3], [10, 11, 12]],
                [[4, 5, 6], [13, 14, 15]],
                [[7, 8, 9], [16, 17, 18]],
            ],
            dtype=np.uint8,
        )

        oriented = self.module.orient_frame(frame)

        self.assertTrue(np.array_equal(oriented, expected))
        self.assertTrue(oriented.flags["C_CONTIGUOUS"])
        self.assertEqual(oriented.dtype, np.uint8)
        self.assertEqual(
            self.module.comparable({"seed": 1, "elapsed": 3.2, "outcome": "success"}),
            {"seed": 1, "outcome": "success"},
        )

    def test_load_cuda_policy_refuses_cpu_fallback_when_cuda_unavailable(self):
        class FakeTorch:
            cuda = SimpleNamespace(is_available=lambda: False)

        class FakePPO:
            @staticmethod
            def load(_path, _device):
                raise AssertionError("PPO.load should not be called without CUDA")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.zip"
            checkpoint.write_text("checkpoint")

            with self.assertRaisesRegex(RuntimeError, "CUDA requested but unavailable"):
                self.module.load_cuda_policy_model(FakePPO, FakeTorch, checkpoint)

    def test_load_cuda_policy_attests_model_and_policy_devices(self):
        loaded = []

        class FakeTorch:
            cuda = SimpleNamespace(is_available=lambda: True)

        class FakePPO:
            @staticmethod
            def load(path, device):
                loaded.append((path, device))
                return FakeModel(model_device="cpu", policy_device="cuda:0")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.zip"
            checkpoint.write_text("checkpoint")

            with self.assertRaisesRegex(RuntimeError, "device mismatch"):
                self.module.load_cuda_policy_model(FakePPO, FakeTorch, checkpoint)

        self.assertEqual(loaded, [(str(checkpoint.resolve()), "cuda")])

    def test_load_cuda_policy_returns_attestation_for_cuda_model(self):
        class FakeTorch:
            cuda = SimpleNamespace(is_available=lambda: True)

        class FakePPO:
            @staticmethod
            def load(path, device):
                return FakeModel(model_device="cuda", policy_device="cuda:0", num_timesteps=99)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.zip"
            checkpoint.write_text("checkpoint")

            model, attestation = self.module.load_cuda_policy_model(FakePPO, FakeTorch, checkpoint)

        self.assertEqual(model.num_timesteps, 99)
        self.assertEqual(attestation["requested_device"], "cuda")
        self.assertEqual(attestation["actual_model_device"], "cuda")
        self.assertEqual(attestation["actual_policy_device"], "cuda:0")

    def test_seed_all_propagates_torch_rng_failures(self):
        class FakeTorch:
            cuda = SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda _seed: None)

            @staticmethod
            def manual_seed(_seed):
                raise RuntimeError("rng seed failed")

        saved_torch = sys.modules.get("torch", None)
        had_torch = "torch" in sys.modules
        sys.modules["torch"] = FakeTorch
        try:
            with self.assertRaisesRegex(RuntimeError, "rng seed failed"):
                self.module._seed_all(123)
        finally:
            if had_torch:
                sys.modules["torch"] = saved_torch
            else:
                sys.modules.pop("torch", None)

    def test_load_expected_episode_accepts_direct_evaluate_modes_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expected = Path(tmpdir) / "modes.json"
            expected.write_text(
                '{"episodes":[{"seed":83001,"outcome":"success"},{"seed":83002,"outcome":"timeout"}],'
                '"summary":{"ok":true}}'
            )

            episode = self.module.load_expected_episode(expected, seed=83001)

        self.assertEqual(episode["seed"], 83001)
        self.assertEqual(episode["outcome"], "success")

    def test_load_expected_episode_rejects_duplicate_seed_in_direct_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expected = Path(tmpdir) / "modes.json"
            expected.write_text(
                '{"episodes":[{"seed":83001,"outcome":"success"},{"seed":83001,"outcome":"timeout"}]}'
            )

            with self.assertRaisesRegex(ValueError, "expected exactly one episode"):
                self.module.load_expected_episode(expected, seed=83001)

    def test_resolve_encoder_prefers_path_ffmpeg(self):
        with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            resolved = self.module.resolve_encoder()

        self.assertEqual(resolved["path"], "/usr/bin/ffmpeg")
        self.assertEqual(resolved["source"], "PATH")

    def test_resolve_encoder_uses_imageio_ffmpeg_when_path_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            binary = Path(tmpdir) / "ffmpeg"
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)

            fake_imageio_ffmpeg = SimpleNamespace(get_ffmpeg_exe=lambda: str(binary))
            saved_module = sys.modules.get("imageio_ffmpeg", None)
            had_module = "imageio_ffmpeg" in sys.modules
            sys.modules["imageio_ffmpeg"] = fake_imageio_ffmpeg
            try:
                with patch.object(shutil, "which", return_value=None):
                    resolved = self.module.resolve_encoder()
            finally:
                if had_module:
                    sys.modules["imageio_ffmpeg"] = saved_module
                else:
                    sys.modules.pop("imageio_ffmpeg", None)

        self.assertEqual(resolved["path"], str(binary))
        self.assertEqual(resolved["source"], "imageio_ffmpeg")

    def test_resolve_encoder_fails_without_path_or_imageio_binary(self):
        saved_module = sys.modules.get("imageio_ffmpeg", None)
        had_module = "imageio_ffmpeg" in sys.modules
        sys.modules["imageio_ffmpeg"] = None
        try:
            with patch.object(shutil, "which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "ffmpeg binary required"):
                    self.module.resolve_encoder()
        finally:
            if had_module:
                sys.modules["imageio_ffmpeg"] = saved_module
            else:
                sys.modules.pop("imageio_ffmpeg", None)

    def test_sampled_pair_reseeds_control_and_render_for_matching_actions(self):
        calls = []

        def fake_record_once(**kwargs):
            calls.append(kwargs["deterministic"])
            value = random.random()
            episode = {"seed": kwargs["seed"], "outcome": "success", "elapsed": value}
            trace = [{"stage": "step", "rng": value}]
            return episode, trace, 1 if kwargs["video"] else 0

        result = self.module.record_validated_pair(
            record_once=fake_record_once,
            le=SimpleNamespace(),
            model=FakeModel(),
            params={},
            phase="0b",
            seed=83001,
            max_steps=600,
            mode="sampled",
            video=Path("out.mp4"),
            encoder_path="ffmpeg",
        )

        self.assertEqual(calls, [False, False])
        self.assertTrue(result["trace_equal"])
        self.assertTrue(result["result_equal"])
        self.assertEqual(result["frame_count"], 1)

    def test_record_validated_pair_fails_when_render_changes_trajectory(self):
        call_count = 0

        def fake_record_once(**kwargs):
            nonlocal call_count
            call_count += 1
            episode = {"seed": kwargs["seed"], "outcome": "success", "elapsed": 1.0}
            trace = [{"run": call_count}]
            return episode, trace, 0

        with self.assertRaisesRegex(RuntimeError, "Rendering changed"):
            self.module.record_validated_pair(
                record_once=fake_record_once,
                le=SimpleNamespace(),
                model=FakeModel(),
                params={},
                phase="0a",
                seed=84001,
                max_steps=600,
                mode="deterministic",
                video=Path("out.mp4"),
                encoder_path="ffmpeg",
            )

    def test_build_proof_records_hashes_device_mode_outcome_and_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "repo"
            repo.mkdir()
            checkpoint = Path(tmpdir) / "checkpoint.zip"
            video = Path(tmpdir) / "episode.mp4"
            checkpoint.write_text("checkpoint")
            video.write_text("video")
            (repo / "config.yaml").write_text("phase: 0b\n")
            (source_path(repo, "train.py")).write_text("train\n")

            proof = self.module.build_proof(
                repo=repo,
                checkpoint=checkpoint,
                config=repo / "config.yaml",
                phase="0b",
                seed=83001,
                mode="sampled",
                max_steps=600,
                model_timesteps=123,
                device_attestation={
                    "requested_device": "cuda",
                    "actual_model_device": "cuda",
                    "actual_policy_device": "cuda:0",
                },
                expected_episode=None,
                control_episode={"seed": 83001, "outcome": "success", "elapsed": 1.0},
                rendered_episode={"seed": 83001, "outcome": "success", "elapsed": 2.0},
                trace_equal=True,
                result_equal=True,
                remote_equal=None,
                video=video,
                frame_count=12,
                encoder={"path": "/usr/bin/ffmpeg", "source": "PATH"},
                versions={"python": "x", "torch": "y"},
            )

        self.assertEqual(proof["device"], "cuda:0")
        self.assertEqual(proof["requested_device"], "cuda")
        self.assertEqual(proof["phase"], "0b")
        self.assertEqual(proof["mode"], "sampled")
        self.assertEqual(proof["seed"], 83001)
        self.assertEqual(proof["outcome"], "success")
        self.assertEqual(proof["encoder"]["path"], "/usr/bin/ffmpeg")
        self.assertEqual(proof["encoder"]["source"], "PATH")
        self.assertRegex(proof["checkpoint_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(proof["video_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("config.yaml", proof["source_hashes"])
        self.assertIn("train.py", proof["source_hashes"])


if __name__ == "__main__":
    unittest.main()
