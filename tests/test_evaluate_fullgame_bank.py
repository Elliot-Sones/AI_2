import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np


class FakeStats:
    def __init__(self, *, lives_left=3, damage_done=0.0, damage_taken=0.0):
        self.lives_left = lives_left
        self.damage_done = damage_done
        self.damage_taken = damage_taken


class FakeBody:
    def __init__(self, x, y, vx=0.0, vy=0.0):
        self.position = (x, y)
        self.velocity = (vx, vy)


class FakePlayer:
    def __init__(self, x, y, *, grounded=True):
        self.body = FakeBody(x, y)
        self.state = type("StandingState", (), {})()
        self.facing = "right"
        self.stocks = 3
        self.damage_taken_total = 0.0
        self.damage_done = 0.0
        self.grounded = grounded
        self.weapon = None


class FakeOpponent:
    def __init__(self):
        self.action_space = type("ActionSpace", (), {"sample": lambda _: np.zeros(10, dtype=np.float32)})()

    def get_env_info(self, env):
        self.env = env

    def predict(self, obs):
        return np.zeros(10, dtype=np.float32)


class FakeEnv:
    def __init__(self, *, terminal="truncated", nonfinite_after_step=False, positions=None, grounded=True):
        self.max_timesteps = 3
        self.steps = 0
        self.step_count = 0
        self.closed = False
        self.terminal = terminal
        self.nonfinite_after_step = nonfinite_after_step
        positions = positions or ((0.0, 0.0), (10.0, 0.0))
        self.players = [
            FakePlayer(positions[0][0], positions[0][1], grounded=grounded),
            FakePlayer(positions[1][0], positions[1][1], grounded=grounded),
        ]
        self.action_space = type("ActionSpace", (), {"shape": (10,), "dtype": np.float32})()

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None, options=None):
        self.seed_seen = seed
        obs = np.arange(64, dtype=np.float32) + (seed or 0)
        return {0: obs.copy(), 1: obs.copy() + 1000}, {}

    def step(self, action):
        self.step_count += 1
        self.steps += 1
        learner = np.asarray(action[0], dtype=np.float32)
        attack_pressed = bool(learner[7] > 0.5 or learner[8] > 0.5)
        if learner[0] > 0.5:
            self.players[0].body.position = (self.players[0].body.position[0] + 2.0, 0.0)
        if attack_pressed:
            self.players[0].state = type("AttackState", (), {})()
        if attack_pressed and self.step_count >= 2:
            self.players[0].damage_done += 4.0
            self.players[1].damage_taken_total += 4.0
        if self.terminal == "self_fall" and self.step_count == 1:
            self.players[0].stocks = 2
            terminated, truncated = True, False
        elif self.step_count >= self.max_timesteps:
            terminated, truncated = False, True
        else:
            terminated, truncated = False, False
        obs = np.arange(64, dtype=np.float32) + self.step_count
        if self.nonfinite_after_step:
            obs[0] = np.nan
        return {0: obs, 1: obs + 1000}, {0: 0.0, 1: 0.0}, terminated, truncated, {}

    def render(self):
        value = np.uint8(self.step_count)
        return np.full((3, 2, 3), value, dtype=np.uint8)

    def get_stats(self, agent_id):
        p = self.players[agent_id]
        return FakeStats(
            lives_left=p.stocks,
            damage_done=p.damage_done,
            damage_taken=p.damage_taken_total,
        )

    def close(self):
        self.closed = True


class DummyPolicy:
    def __init__(self, action, sampled_action=None):
        self.action = np.asarray(action, dtype=np.float32)
        self.sampled_action = np.asarray(sampled_action if sampled_action is not None else action, dtype=np.float32)
        self.calls = []

    def predict(self, obs, *, deterministic):
        self.calls.append((obs.shape, deterministic))
        return (self.action if deterministic else self.sampled_action).copy()


def missing_real_game_dependencies():
    missing = []
    for module in ("gdown", "pygame", "pymunk", "gymnasium", "skvideo", "cv2", "skimage", "PIL", "imageio"):
        try:
            __import__(module)
        except Exception as exc:
            missing.append(f"{module}:{type(exc).__name__}")
    return missing


class EvaluateFullGameBankTests(unittest.TestCase):
    def test_bank_has_unique_case_ids_and_seeds(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        bank = ev.build_default_bank()
        ids = [case.case_id for case in bank]
        seeds = [case.seed for case in bank]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue({"Constant", "Random", "Based", "Clockwork"}.issubset({c.opponent for c in bank}))
        self.assertTrue(all(case.placement_profile == "reset_only" for case in bank))
        self.assertTrue({0, 1}.issubset({case.player_slot for case in bank}))
        self.assertTrue(any("mirror" in case.mirroring for case in bank if case.player_slot == 1))

    def test_frame_stack_reset_matches_zero_prefill_reference(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        obs = np.arange(64, dtype=np.float32)
        expected = np.concatenate([np.zeros(64, dtype=np.float32)] * 3 + [obs])
        stack = ev.FrameStack()
        np.testing.assert_array_equal(stack.reset(obs), expected)
        next_obs = obs + 100
        np.testing.assert_array_equal(stack.step(next_obs), np.concatenate([np.zeros(64, dtype=np.float32)] * 2 + [obs, next_obs]))

    def test_repeatable_evaluation_labels_modes_and_decodes_actions(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        action = np.array([0.2, 0, 0, 0.7, 0, 0, 0, 0.51, 0, 0], dtype=np.float32)
        sampled = np.array([0, 0, 0, 0, 0, 0, 0, 0.49, 0.75, 0], dtype=np.float32)
        policy = DummyPolicy(action, sampled_action=sampled)
        case = ev.EvaluationCase("case_a", seed=11, opponent="Constant", max_ticks=2)

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            out1 = Path(d1) / "out"
            out2 = Path(d2) / "out"
            r1 = ev.run_evaluation(
                policy=policy,
                cases=[case],
                modes=("deterministic", "sampled"),
                output_dir=out1,
                env_factory=lambda c: FakeEnv(),
                opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
            )
            r2 = ev.run_evaluation(
                policy=policy,
                cases=[case],
                modes=("deterministic", "sampled"),
                output_dir=out2,
                env_factory=lambda c: FakeEnv(),
                opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
            )

            self.assertEqual(r1["summary"]["wins"], r2["summary"]["wins"])
            self.assertEqual(r1["summary"]["episodes"], 2)
            traces = [json.loads(line) for line in (out1 / "trace.jsonl").read_text().splitlines()]
            self.assertEqual({row["mode"] for row in traces}, {"deterministic", "sampled"})
            det_first = next(row for row in traces if row["mode"] == "deterministic" and row["tick"] == 0)
            sampled_first = next(row for row in traces if row["mode"] == "sampled" and row["tick"] == 0)
            self.assertAlmostEqual(det_first["executed_action_values"][3], 0.7, places=6)
            self.assertEqual(det_first["decoded_buttons"][3], 1)
            self.assertEqual(det_first["decoded_buttons"][7], 1)
            self.assertEqual(sampled_first["decoded_buttons"][7], 0)
            self.assertEqual(sampled_first["decoded_buttons"][8], 1)
            self.assertEqual(policy.calls[0][0], (256,))

    def test_manifest_cases_record_effective_episode_max_override(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        case = ev.EvaluationCase("case_a", seed=11, opponent="Constant", max_ticks=2700)

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            result = ev.run_evaluation(
                policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                cases=[case],
                modes=("deterministic",),
                output_dir=out,
                env_factory=lambda c: FakeEnv(),
                opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
                episode_max=180,
            )

            written_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(result["manifest"]["cases"][0]["max_ticks"], 180)
            self.assertEqual(written_manifest["cases"][0]["max_ticks"], 180)

    def test_slot_one_case_sends_policy_action_to_slot_one(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        class SlotRecordingEnv(FakeEnv):
            def step(self, action):
                self.last_action_keys = sorted(action.keys())
                self.slot1_action = np.asarray(action[1], dtype=np.float32).copy()
                return super().step({0: action[0], 1: action[1]})

        env = SlotRecordingEnv(positions=((10.0, 0.0), (0.0, 0.0)))
        with tempfile.TemporaryDirectory() as d:
            ev.run_evaluation(
                policy=DummyPolicy(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
                cases=[ev.EvaluationCase("slot1", seed=11, opponent="Constant", max_ticks=1, player_slot=1, mirroring="mirror_player1_right")],
                modes=("deterministic",),
                output_dir=Path(d) / "out",
                env_factory=lambda c: env,
                opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
            )
        self.assertEqual(env.last_action_keys, [0, 1])
        self.assertEqual(float(env.slot1_action[3]), 1.0)

    def test_rng_state_is_restored_even_when_env_construction_fails(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        try:
            import torch
        except Exception:
            torch = None

        random.seed(123)
        np.random.seed(123)
        if torch is not None:
            torch.manual_seed(123)
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone() if torch is not None else None

        def fail_factory(case):
            random.random()
            np.random.random()
            if torch is not None:
                torch.rand(1)
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                ev.run_evaluation(
                    policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                    cases=[ev.EvaluationCase("case_a", seed=7, opponent="Constant", max_ticks=1)],
                    modes=("deterministic",),
                    output_dir=Path(d) / "out",
                    env_factory=fail_factory,
                    opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
                )

        self.assertEqual(random.getstate(), py_state)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        if torch is not None:
            self.assertTrue(bool(torch.equal(torch.random.get_rng_state(), torch_state)))

    def test_terminal_cause_and_diagnosis_are_derived_from_trace(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        with tempfile.TemporaryDirectory() as d:
            result = ev.run_evaluation(
                policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                cases=[ev.EvaluationCase("fall_case", seed=5, opponent="Constant", max_ticks=4)],
                modes=("deterministic",),
                output_dir=Path(d) / "out",
                env_factory=lambda c: FakeEnv(terminal="self_fall"),
                opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
            )
            episode = json.loads((Path(d) / "out" / "episodes.jsonl").read_text())
            self.assertEqual(episode["terminal_cause"], "self_fall")
            self.assertIn("no_attack_request", episode["diagnoses"])
            self.assertIn("self_fall", episode["diagnoses"])
            self.assertEqual(result["summary"]["losses"], 1)

    def test_requested_attack_without_hit_is_diagnosed(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        with tempfile.TemporaryDirectory() as d:
            episode_dir = Path(d) / "out"
            ev.run_evaluation(
                policy=DummyPolicy(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32)),
                cases=[ev.EvaluationCase("attack_case", seed=5, opponent="Constant", max_ticks=1)],
                modes=("deterministic",),
                output_dir=episode_dir,
                env_factory=lambda c: FakeEnv(),
                opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
            )
            episode = json.loads((episode_dir / "episodes.jsonl").read_text())
            self.assertIn("attack_started_but_no_hit", episode["diagnoses"])

    def test_nonfinite_policy_output_fails_before_env_step_and_output_commit(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        env = FakeEnv()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            with self.assertRaises(ValueError):
                ev.run_evaluation(
                    policy=DummyPolicy(np.array([np.nan] + [0] * 9, dtype=np.float32)),
                    cases=[ev.EvaluationCase("bad", seed=1, opponent="Constant", max_ticks=1)],
                    modes=("deterministic",),
                    output_dir=out,
                    env_factory=lambda c: env,
                    opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
                )
            self.assertEqual(env.step_count, 0)
            self.assertFalse((out / "trace.jsonl").exists())

    def test_output_refuses_nonempty_directory_and_records_video_metadata(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            out.mkdir()
            (out / "existing.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                ev.run_evaluation(
                    policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                    cases=[ev.EvaluationCase("case", seed=1, opponent="Constant", max_ticks=1)],
                    modes=("deterministic",),
                    output_dir=out,
                    env_factory=lambda c: FakeEnv(),
                    opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
                )

    def test_video_request_records_episode_frames_and_hashes_manifest_video(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            captured_frames = []

            def fake_capture(frames, path):
                captured_frames.extend(frame.copy() for frame in frames)
                path.write_bytes(b"fake mp4 bytes")

            result = ev.run_evaluation(
                policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                cases=[ev.EvaluationCase("case", seed=1, opponent="Constant", max_ticks=2)],
                modes=("deterministic",),
                output_dir=out,
                env_factory=lambda c: FakeEnv(),
                opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
                video_subset=("case:deterministic",),
                video_capture=fake_capture,
            )

            self.assertEqual([int(frame[0, 0, 0]) for frame in captured_frames], [0, 1, 2])
            episode = result["episodes"][0]
            video_rel = Path(episode["video_path"])
            self.assertEqual(video_rel.as_posix(), "videos/case_deterministic.mp4")
            self.assertIn(video_rel.as_posix(), result["manifest"]["artifact_sha256"])
            self.assertEqual((out / video_rel).read_bytes(), b"fake mp4 bytes")

    def test_default_video_capture_streams_frames_to_recorder(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        opened_recorders = []

        class FakeRecorder:
            def __init__(self, path):
                self.path = path
                self.frames = []
                self.closed = False
                opened_recorders.append(self)

            def append(self, frame):
                self.frames.append(int(frame[0, 0, 0]))

            def close(self):
                self.closed = True
                self.path.write_bytes(b"streamed mp4 bytes")

        original = ev._open_default_video_recorder
        ev._open_default_video_recorder = FakeRecorder
        try:
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "out"
                result = ev.run_evaluation(
                    policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                    cases=[ev.EvaluationCase("case", seed=1, opponent="Constant", max_ticks=2)],
                    modes=("deterministic",),
                    output_dir=out,
                    env_factory=lambda c: FakeEnv(),
                    opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
                    video_subset=("case:deterministic",),
                )
                self.assertEqual(result["episodes"][0]["video_path"], "videos/case_deterministic.mp4")
        finally:
            ev._open_default_video_recorder = original

        self.assertEqual(len(opened_recorders), 1)
        self.assertEqual(opened_recorders[0].frames, [0, 1, 2])
        self.assertTrue(opened_recorders[0].closed)

    def test_video_request_fails_closed_when_default_capture_cannot_render(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        class NoRenderEnv(FakeEnv):
            render = None

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            with self.assertRaises(ev.VideoUnavailableError):
                ev.run_evaluation(
                    policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                    cases=[ev.EvaluationCase("case", seed=1, opponent="Constant", max_ticks=1)],
                    modes=("deterministic",),
                    output_dir=out,
                    env_factory=lambda c: NoRenderEnv(),
                    opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
                    video_subset=("case:deterministic",),
                )
            self.assertFalse(out.exists())

    def test_video_capture_must_create_nonempty_artifact(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ev.VideoUnavailableError):
                ev.run_evaluation(
                    policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                    cases=[ev.EvaluationCase("case", seed=1, opponent="Constant", max_ticks=1)],
                    modes=("deterministic",),
                    output_dir=Path(d) / "out",
                    env_factory=lambda c: FakeEnv(),
                    opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
                    video_subset=("case:deterministic",),
                    video_capture=lambda frames, path: path.write_bytes(b""),
                )

    def test_unknown_video_subset_key_fails_before_env_work(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        env = FakeEnv()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            with self.assertRaises(ValueError):
                ev.run_evaluation(
                    policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                    cases=[ev.EvaluationCase("case", seed=1, opponent="Constant", max_ticks=1)],
                    modes=("deterministic",),
                    output_dir=out,
                    env_factory=lambda c: env,
                    opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
                    video_subset=("missing:deterministic",),
                )
            self.assertEqual(env.step_count, 0)
            self.assertFalse(out.exists())

    def test_policy_adapters_validate_action_shape(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        class BadModel:
            def predict(self, obs, deterministic):
                return np.zeros(9, dtype=np.float32), None

        adapter = ev.SB3PolicyAdapter(BadModel(), action_head="gaussian")
        with self.assertRaises(ValueError):
            adapter.predict(np.zeros(256, dtype=np.float32), deterministic=True)

    def test_sb3_adapter_rejects_unsupported_observation_width(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        class ClassicHistoryModel:
            observation_space = type("ObsSpace", (), {"shape": (2656,)})()

            def predict(self, obs, deterministic):
                return np.zeros(10, dtype=np.float32), None

        with self.assertRaises(ValueError):
            ev.SB3PolicyAdapter(ClassicHistoryModel(), action_head="gaussian")

    def test_actor_adapter_accepts_export_like_actor_with_256_obs(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        class ExportLikeActor:
            action_head = "bernoulli_10"
            expected_observation_width = 256

            def predict(self, obs, deterministic):
                self.observed_shape = np.asarray(obs).shape
                return np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32), None

        actor = ExportLikeActor()
        adapter = ev.ActorPolicyAdapter(actor, metadata={"action_head": "bernoulli_10", "observation_size": 256})
        action = adapter.predict(np.zeros(256, dtype=np.float32), deterministic=True)
        self.assertEqual(actor.observed_shape, (256,))
        self.assertEqual(action[7], 1)

    def test_grounded_snapshot_uses_callable_is_on_floor(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        class CallableGroundPlayer(FakePlayer):
            grounded = True

            def is_on_floor(self):
                return False

        player = CallableGroundPlayer(0.0, 0.0)
        self.assertFalse(ev._player_snapshot(player)["grounded"])

    def test_explicit_checkpoint_path_must_exist_before_env_work(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        env = FakeEnv()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                ev.run_evaluation(
                    policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                    cases=[ev.EvaluationCase("case", seed=1, opponent="Constant", max_ticks=1)],
                    modes=("deterministic",),
                    output_dir=Path(d) / "out",
                    env_factory=lambda c: env,
                    opponent_factory=lambda name: (FakeOpponent(), "FakeOpponent"),
                    checkpoint_path=Path(d) / "missing.zip",
                )
            self.assertEqual(env.step_count, 0)

    def test_unsupported_opponent_fails_closed_before_output_commit(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            with self.assertRaises(ValueError):
                ev.run_evaluation(
                    policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                    cases=[ev.EvaluationCase("case", seed=1, opponent="DefinitelyUnsupported", max_ticks=1)],
                    modes=("deterministic",),
                    output_dir=out,
                    env_factory=lambda c: FakeEnv(),
                )
            self.assertFalse(out.exists())

    def test_real_environment_no_render_smoke_is_opt_in(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        if os.environ.get("AI2_RUN_REAL_EVAL_SMOKE") != "1":
            self.skipTest("opt-in real game smoke")
        missing = missing_real_game_dependencies()
        if missing:
            self.skipTest("missing real game deps: " + ",".join(missing))
        import gdown

        original_cwd = os.getcwd()
        download_calls = []

        def forbidden_download(*args, **kwargs):
            download_calls.append((args, kwargs))
            raise AssertionError("real smoke must use existing assets and never call gdown.download")

        original_download = gdown.download
        gdown.download = forbidden_download
        with tempfile.TemporaryDirectory() as d:
            try:
                out = Path(d) / "out"
                result = ev.run_evaluation(
                    policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                    cases=[ev.EvaluationCase("real_constant_smoke", seed=20260915, opponent="Constant", max_ticks=1)],
                    modes=("deterministic",),
                    output_dir=out,
                )
                self.assertEqual(result["summary"]["episodes"], 1)
                self.assertTrue((out / "trace.jsonl").exists())
                self.assertTrue((out / "manifest.json").exists())
                self.assertEqual(download_calls, [])
            finally:
                gdown.download = original_download
        self.assertEqual(os.getcwd(), original_cwd)

    def test_real_environment_video_smoke_is_opt_in(self):
        import ai2.evaluation.evaluate_fullgame_bank as ev

        if os.environ.get("AI2_RUN_REAL_EVAL_VIDEO_SMOKE") != "1":
            self.skipTest("opt-in real game video smoke")
        missing = missing_real_game_dependencies()
        if missing:
            self.skipTest("missing real game deps: " + ",".join(missing))
        import gdown
        import imageio

        original_cwd = os.getcwd()
        download_calls = []

        def forbidden_download(*args, **kwargs):
            download_calls.append((args, kwargs))
            raise AssertionError("real video smoke must use existing assets and never call gdown.download")

        original_download = gdown.download
        gdown.download = forbidden_download
        with tempfile.TemporaryDirectory() as d:
            try:
                out = Path(d) / "video-out"
                case = ev.EvaluationCase("real_video_smoke", seed=20260915, opponent="Constant", max_ticks=2)
                result = ev.run_evaluation(
                    policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                    cases=[case],
                    modes=("deterministic",),
                    output_dir=out,
                    video_subset=("real_video_smoke:deterministic",),
                )
                self.assertEqual(result["summary"]["episodes"], 1)
                episode = json.loads((out / "episodes.jsonl").read_text())
                self.assertEqual(episode["video_path"], "videos/real_video_smoke_deterministic.mp4")
                video = out / episode["video_path"]
                self.assertGreater(video.stat().st_size, 0)
                self.assertIn(episode["video_path"], result["manifest"]["artifact_sha256"])
                reader = imageio.get_reader(str(video))
                try:
                    frame = reader.get_data(0)
                finally:
                    reader.close()
                self.assertEqual(frame.ndim, 3)
                self.assertEqual(frame.shape[2], 3)
                self.assertEqual(download_calls, [])

                bad_out = Path(d) / "bad-video-out"
                with self.assertRaises(ValueError):
                    ev.run_evaluation(
                        policy=DummyPolicy(np.zeros(10, dtype=np.float32)),
                        cases=[case],
                        modes=("deterministic",),
                        output_dir=bad_out,
                        video_subset=("missing:deterministic",),
                    )
                self.assertFalse(bad_out.exists())
            finally:
                gdown.download = original_download
        self.assertEqual(os.getcwd(), original_cwd)


if __name__ == "__main__":
    unittest.main()
