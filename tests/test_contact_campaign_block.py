import gzip
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from contextlib import ExitStack
import unittest
from unittest.mock import patch


class ContactCampaignBlockTests(unittest.TestCase):
    def subject(self):
        self.assertIsNotNone(
            importlib.util.find_spec("ai2.training.train_contact_campaign_block"),
            "contact campaign block runner has not been implemented",
        )
        import ai2.training.train_contact_campaign_block as train_contact_campaign_block

        return train_contact_campaign_block

    def test_config_is_bounded_to_one_4096_lane_training_block(self):
        m = self.subject()
        cfg = m.ContactCampaignConfig(rollouts=5, episodes_per_lane=1024)
        self.assertEqual(cfg.lanes, 4096)
        self.assertEqual(cfg.n_steps, 512)
        self.assertEqual(cfg.n_epochs, 4)
        self.assertEqual(cfg.batch_size, 16384)
        self.assertEqual(cfg.learning_rate, 3e-5)
        self.assertEqual(cfg.transitions_per_rollout, 2_097_152)
        self.assertEqual(cfg.total_transitions, 10_485_760)
        self.assertEqual(cfg.nominal_optimizer_steps, 512)
        for changes in (
            {"lanes": 2048},
            {"rollouts": 0},
            {"rollouts": 6},
            {"batch_size": 32768},
            {"learning_rate": 1e-4},
            {"action_ab_arm": "bad"},
            {"episodes_per_lane": 512},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.ContactCampaignConfig(**changes)

    def test_schedule_gate_counts_completed_episode_records_by_lane(self):
        m = self.subject()
        cfg = m.ContactCampaignConfig(lanes=4096, rollouts=1, episodes_per_lane=1024)
        completed = m.empty_lane_episode_counts(cfg)
        m.require_schedule_capacity(cfg, completed, next_rollout=0)
        m.accumulate_episode_counts(completed, [
            {"contact": {"lane": 3, "episode_index": 0}},
            {"contact": {"lane": 3, "episode_index": 1}},
            {"contact": {"lane": 7, "episode_index": 0}},
        ])
        self.assertEqual(completed[3], 2)
        self.assertEqual(completed[7], 1)
        completed[3] = 510
        m.require_schedule_capacity(cfg, completed, next_rollout=1)
        completed[3] = 511
        with self.assertRaisesRegex(RuntimeError, "scenario capacity"):
            m.require_schedule_capacity(cfg, completed, next_rollout=1)
        with self.assertRaisesRegex(RuntimeError, "lane"):
            m.accumulate_episode_counts(completed, [{"contact": {}}])

    def test_parse_requires_explicit_checkpoint_sha_and_ack(self):
        m = self.subject()
        root = Path("/tmp/root")
        base = [
            "--checkpoint", str(root / "checkpoint.zip"),
            "--frozen-root", str(root),
            "--reference-root", str(root),
            "--qualification", str(root / "proof.json"),
            "--output", str(root / "out"),
            "--deadline-unix", "9999999999",
        ]
        with self.assertRaises(SystemExit):
            m.parse_args(base)
        args = m.parse_args(base + [
            "--checkpoint-sha256", "a" * 64,
            "--checkpoint-ack", str(root / "preservation_ack.json"),
            "--resume",
            "--rollouts", "4",
        ])
        self.assertTrue(args.resume)
        self.assertEqual(args.rollouts, 4)

    def test_fresh_b_projected_can_start_with_empty_optimizer_state_but_resume_cannot(self):
        m = self.subject()
        cfg = m.ContactCampaignConfig(action_ab_arm="B_projected")
        model = self.fake_model(cfg, start_timesteps=m.throughput.CHECKPOINT_STEPS, empty_optimizer=True)
        state = m.validate_campaign_runtime(model, cfg, fresh=True)
        self.assertEqual(state["state_tensor_elements"], 0)
        with self.assertRaisesRegex(ValueError, "preserved optimizer state"):
            m.validate_campaign_runtime(model, cfg, fresh=False)

    def test_save_checkpoint_accepts_sb3_zip_suffix_append(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "final.zip"
            model = SimpleNamespace(save=lambda path: Path(str(path) + ".zip").write_bytes(b"checkpoint"))
            receipt = m.save_checkpoint(output, model)
            self.assertTrue(output.is_file())
            self.assertFalse((Path(directory) / "final.zip.tmp.zip").exists())
            self.assertEqual(receipt["sha256"], m.sha256(output))

    def test_fresh_block_validates_canonical_parent_and_applies_lr_variant(self):
        m = self.subject()
        cfg = m.ContactCampaignConfig(lanes=4096, rollouts=1, episodes_per_lane=1024)
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), resume=False)
            raw = self.fake_raw(cfg)
            model = self.fake_model(cfg, start_timesteps=m.throughput.CHECKPOINT_STEPS)
            with self.common_patches(m, cfg, raw, model), \
                 patch.object(m.base, "validate_checkpoint", return_value={"sha256": m.throughput.CHECKPOINT_SHA256}) as validate, \
                 patch.object(m, "load_fresh_campaign_model", return_value=model) as load_fresh, \
                 patch.object(m, "load_resume_campaign_model") as load_resume, \
                 patch.object(m.throughput, "apply_learning_rate_variant", wraps=m.throughput.apply_learning_rate_variant) as apply_lr:
                result = m.run_block(args)
            validate.assert_called_once_with(args.checkpoint)
            load_fresh.assert_called_once()
            load_resume.assert_not_called()
            apply_lr.assert_called_once()
            self.assertEqual(result["status"], "trained_not_evaluated")
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertTrue(manifest["fresh_parent_per_block"])
            self.assertFalse(manifest["resume"])
            self.assertEqual(manifest["checkpoint"]["sha256"], m.throughput.CHECKPOINT_SHA256)

    def test_resume_block_uses_noncanonical_sha_without_reinitializing_lr_or_optimizer(self):
        m = self.subject()
        cfg = m.ContactCampaignConfig(lanes=4096, rollouts=1, episodes_per_lane=1024)
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), resume=True, checkpoint_sha="b" * 64)
            args.checkpoint.write_bytes(b"resume checkpoint")
            raw = self.fake_raw(cfg)
            model = self.fake_model(cfg, start_timesteps=m.throughput.CHECKPOINT_STEPS + cfg.transitions_per_rollout)
            with self.common_patches(m, cfg, raw, model), \
                 patch.object(m.base, "validate_checkpoint") as validate, \
                 patch.object(m, "sha256", return_value="b" * 64), \
                 patch.object(m, "load_fresh_campaign_model") as load_fresh, \
                 patch.object(m, "load_resume_campaign_model", return_value=model) as load_resume, \
                 patch.object(m.throughput, "apply_learning_rate_variant") as apply_lr:
                result = m.run_block(args)
            validate.assert_not_called()
            load_fresh.assert_not_called()
            load_resume.assert_called_once()
            apply_lr.assert_not_called()
            self.assertEqual(result["resume_from_checkpoint_sha256"], "b" * 64)
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertFalse(manifest["fresh_parent_per_block"])
            self.assertTrue(manifest["resume"])
            self.assertIn("no bitwise resumed trajectory claim", manifest["rng_and_environment_note"])

    def test_failure_before_next_rollout_saves_recovery_checkpoint_and_inventory(self):
        m = self.subject()
        cfg = m.ContactCampaignConfig(lanes=4096, rollouts=2, episodes_per_lane=514)
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), resume=False)
            raw = self.fake_raw(cfg)
            model = self.fake_model(cfg, start_timesteps=m.throughput.CHECKPOINT_STEPS, completed_lane=11)
            with self.common_patches(m, cfg, raw, model), \
                 patch.object(m.base, "validate_checkpoint", return_value={"sha256": m.throughput.CHECKPOINT_SHA256}), \
                 patch.object(m, "load_fresh_campaign_model", return_value=model), \
                 patch.object(m.throughput, "apply_learning_rate_variant", wraps=m.throughput.apply_learning_rate_variant), \
                 self.assertRaisesRegex(RuntimeError, "scenario capacity"):
                m.run_block(args)
            self.assertTrue((args.output / "failure-last.zip").is_file())
            self.assertTrue((args.output / "failure.json").is_file())
            failure = json.loads((args.output / "failure.json").read_text())
            self.assertEqual(failure["last_completed_checkpoint"], "failure-last.zip")

    def test_mocked_success_writes_one_final_checkpoint_actor_episode_gz_and_evaluable_result(self):
        m = self.subject()
        cfg = m.ContactCampaignConfig(lanes=4096, rollouts=2, episodes_per_lane=1024)
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), resume=False)
            raw = self.fake_raw(cfg)
            model = self.fake_model(cfg, start_timesteps=m.throughput.CHECKPOINT_STEPS)
            with self.common_patches(m, cfg, raw, model), \
                 patch.object(m.base, "validate_checkpoint", return_value={"sha256": m.throughput.CHECKPOINT_SHA256}), \
                 patch.object(m, "load_fresh_campaign_model", return_value=model), \
                 patch.object(m.throughput, "apply_learning_rate_variant", wraps=m.throughput.apply_learning_rate_variant):
                result = m.run_block(args)
            self.assertEqual(result["status"], "trained_not_evaluated")
            self.assertEqual(result["checkpoint_sha256"], m.sha256(args.output / "final.zip"))
            self.assertTrue((args.output / "final.zip").is_file())
            self.assertTrue((args.output / "actor.pt").is_file())
            self.assertTrue((args.output / "inventory.json").is_file())
            paths = {p.relative_to(args.output).as_posix() for p in args.output.rglob("*") if p.is_file()}
            self.assertIn("episodes-000.json.gz", paths)
            self.assertIn("episodes-001.json.gz", paths)
            self.assertNotIn("rollout-000.zip", paths)
            row = json.loads((args.output / "rollout-000.json").read_text())
            self.assertEqual(row["episode_info"]["count"], 1)
            payload = gzip.decompress((args.output / row["episode_info"]["path"]).read_bytes())
            self.assertEqual(json.loads(payload)[0]["contact"]["lane"], 0)
            self.assertEqual(json.loads(payload)[0]["episode"], {"r": 1.0, "l": 1})
            self.assertEqual(result["final_model_counters"]["num_timesteps"], m.throughput.CHECKPOINT_STEPS + cfg.total_transitions)

    def args(self, root, *, resume, checkpoint_sha=None):
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "checkpoint.zip"
        checkpoint.write_bytes(b"canonical checkpoint")
        ack = root / "preservation_ack.json"
        ack.write_text("{}")
        (root / "proof.json").write_text("{}")
        return SimpleNamespace(
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha or "301ce3c72924f3e5c62f265f88292b8952e86e6d71d9a0552213cb8f3c6784b6",
            checkpoint_ack=ack,
            frozen_root=root,
            reference_root=root,
            qualification=root / "proof.json",
            output=root / "run",
            deadline_unix=10**12,
            lanes=4096,
            rollouts=1,
            batch_size=16384,
            episodes_per_lane=1024,
            seed=145,
            learning_rate=3e-5,
            device="cuda",
            update_mode="deferred_metrics",
            action_ab_arm="G_keep",
            resume=resume,
        )

    def common_patches(self, m, cfg, raw, model):
        patches = [
            patch.object(m, "ContactCampaignConfig", return_value=cfg),
            patch.object(m.base, "require_checkpoint_preservation", return_value={"verified": True}),
            patch.object(m.base, "require_qualification", return_value={"qualified": True}),
            patch.object(m.base, "qualification_sources", return_value={}),
            patch.object(m.base, "require_cuda", return_value="cuda:0"),
            patch.object(m.throughput, "seed_everything"),
            patch.object(m.throughput, "build_contact_env", return_value=raw),
            patch.object(m.throughput, "tensor_vec_env", side_effect=lambda env, device: env),
            patch.object(m.throughput, "synchronize_cuda"),
            patch.object(m.throughput, "cuda_peak_memory", return_value=5678),
            patch.object(m.throughput, "cuda_device_name", return_value="mock-gpu"),
            patch.object(m.throughput, "model_parameter_count", return_value=15_000_000),
            patch.object(m.throughput, "configure_logger", return_value=self.fake_logger()),
            patch.object(m.throughput, "perf_counter_sequence", side_effect=self.clock()),
        ]
        class Stack:
            def __enter__(self_nonlocal):
                self_nonlocal.stack = ExitStack()
                self_nonlocal.entered = [self_nonlocal.stack.enter_context(p) for p in patches]
                return self_nonlocal.entered
            def __exit__(self_nonlocal, *exc):
                return self_nonlocal.stack.__exit__(*exc)
        return Stack()

    def fake_raw(self, cfg):
        backend = SimpleNamespace(
            valid_steps=[0] * cfg.lanes,
            errors=[0] * cfg.lanes,
            scenario_metadata={"lanes": cfg.lanes, "episodes_per_lane": cfg.episodes_per_lane},
        )
        return SimpleNamespace(backend=backend, close=lambda: None)

    def fake_logger(self):
        return SimpleNamespace(
            name_to_value={
                "train/loss": 0.1,
                "train/policy_gradient_loss": 0.1,
                "train/value_loss": 0.1,
                "train/entropy_loss": -0.1,
            },
            dump=lambda **kwargs: None,
            close=lambda: None,
        )

    def fake_model(self, cfg, *, start_timesteps, completed_lane=0, empty_optimizer=False):
        optimizer_state = {} if empty_optimizer else {1: {"step": 2, "exp_avg": self.fake_parameter([1, 2, 3])}}
        model = SimpleNamespace(
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            target_kl=cfg.target_kl,
            ent_coef=cfg.ent_coef,
            n_envs=cfg.lanes,
            num_timesteps=start_timesteps,
            _n_updates=0,
            _episode_num=0,
            _total_timesteps=start_timesteps,
            task="combat",
            calibration=False,
            arm="E1",
            action_head=cfg.action_head,
            action_ab_arm=cfg.action_ab_arm,
            action_init_strategy="keep",
            action_init_metadata={"arm": cfg.action_ab_arm, "source": "test"},
            optimizer_provenance="protected Adam restored",
            policy=SimpleNamespace(
                parameters=lambda: [self.fake_parameter([1, 2, model.num_timesteps % 251])],
                optimizer=SimpleNamespace(
                    param_groups=[{"lr": cfg.learning_rate}],
                    state=optimizer_state,
                ),
            ),
            lr_schedule=lambda _: cfg.learning_rate,
            clip_range=lambda _: cfg.clip_range,
            logger=SimpleNamespace(name_to_value={}, dump=lambda **kwargs: None),
            rollout_buffer=SimpleNamespace(actions=None, executed_actions=None),
            last_update_diagnostics={
                "attempted_epochs": 4,
                "optimizer_steps": cfg.nominal_optimizer_steps,
                "kl_early_stop": False,
            },
        )

        def setup(total, **kwargs):
            model._total_timesteps = model.num_timesteps + total
            callback = SimpleNamespace(
                locals={},
                episodes=[],
                on_training_start=lambda *a: None,
                on_training_end=lambda: None,
            )
            return model._total_timesteps, callback

        def collect(env, callback, buffer, n_steps):
            model.num_timesteps += cfg.transitions_per_rollout
            env.backend.valid_steps = [value + n_steps for value in env.backend.valid_steps]
            callback.episodes = [
                {"contact": {"lane": completed_lane, "episode_index": model.num_timesteps}, "episode": {"r": 1.0, "l": 1}}
            ]
            return True

        model.set_logger = lambda logger: setattr(model, "logger", logger)
        model._setup_learn = setup
        model.collect_rollouts = collect
        model._update_current_progress_remaining = lambda *args: None
        model.train = lambda: None
        model.save = lambda path: Path(path).write_bytes(b"checkpoint")
        model.export_actor = lambda path: Path(path).write_bytes(b"actor")
        return model

    def fake_parameter(self, values):
        class Parameter:
            device = SimpleNamespace(type="cuda")
            def __init__(self, values):
                self.values = values
            def numel(self):
                return len(self.values)
            def detach(self):
                return self
            def cpu(self):
                return self
            def contiguous(self):
                return self
            def numpy(self):
                return self
            def tobytes(self):
                return bytes(self.values)
        return Parameter(values)

    def clock(self):
        value = 0.0
        while True:
            value += 1.0
            yield value


if __name__ == "__main__":
    unittest.main()
