from dataclasses import replace
import gzip
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class CurriculumThroughputHarnessTests(unittest.TestCase):
    def subject(self):
        self.assertIsNotNone(
            importlib.util.find_spec("benchmark_curriculum_throughput"),
            "curriculum throughput qualification harness has not been implemented",
        )
        import benchmark_curriculum_throughput

        return benchmark_curriculum_throughput

    def test_config_allows_scaling_lanes_without_changing_training_hyperparams(self):
        m = self.subject()
        cfg = m.ThroughputBenchmarkConfig(lanes=1024, batch_size=16384, warmup_rollouts=2, measured_rollouts=5)
        self.assertEqual(cfg.n_steps, 512)
        self.assertEqual(cfg.n_epochs, 4)
        self.assertEqual(cfg.learning_rate, 0.0001)
        self.assertEqual(cfg.target_kl, 0.01)
        self.assertEqual(cfg.transitions_per_rollout, 524288)
        self.assertEqual(cfg.nominal_optimizer_steps, 128)
        self.assertEqual(cfg.total_rollouts, 7)
        self.assertEqual(cfg.learning_contract_variant, "canonical")
        for changes in (
            {"lanes": 128},
            {"lanes": 4096, "allow_4096": False},
            {"lanes": 8192, "allow_4096": True, "allow_8192": False},
            {"batch_size": 32768},
            {"warmup_rollouts": -1},
            {"measured_rollouts": 0},
            {"seed": -1},
            {"device": "cpu"},
            {"action_ab_arm": "G_neutral"},
            {"episodes_per_lane": 1},
            {"learning_rate": 2e-5},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(cfg, **changes)
        diagnostic = replace(cfg, learning_rate=3e-5)
        self.assertEqual(diagnostic.learning_contract_variant, "lr_diagnostic")
        self.assertTrue(diagnostic.not_lossless_optimization)
        extreme = m.ThroughputBenchmarkConfig(lanes=8192, allow_4096=True, allow_8192=True, batch_size=16384)
        self.assertEqual(extreme.transitions_per_rollout, 4194304)
        self.assertEqual(extreme.nominal_optimizer_steps, 1024)

    def test_episode_schedule_is_bounded_and_documented(self):
        m = self.subject()
        cfg = m.ThroughputBenchmarkConfig(lanes=2048, warmup_rollouts=1, measured_rollouts=3)
        self.assertEqual(cfg.required_episodes_per_lane, 8)
        self.assertEqual(cfg.episodes_per_lane, 128)
        self.assertEqual(cfg.record_capacity, cfg.transitions_per_rollout)
        smaller = replace(cfg, episodes_per_lane=8)
        self.assertEqual(smaller.episodes_per_lane, 8)

    def test_warmup_rollouts_are_excluded_from_throughput_summary(self):
        m = self.subject()
        rows = [
            self.row(0, warmup=True, total_seconds=100.0),
            self.row(1, warmup=False, total_seconds=2.0, collection_seconds=0.5, update_seconds=1.5),
            self.row(2, warmup=False, total_seconds=3.0, collection_seconds=1.0, update_seconds=2.0),
        ]
        summary = m.summarize_measurements(rows)
        self.assertEqual(summary["measured_rollouts"], 2)
        self.assertEqual(summary["warmup_rollouts"], 1)
        self.assertEqual(summary["measured_transitions"], 65536)
        self.assertEqual(summary["measured_training_steps_per_second"], 65536 / 5.0)
        self.assertEqual(summary["measured_collection_steps_per_second"], 65536 / 1.5)
        self.assertEqual(summary["timing_window"], "measured_rollouts_only")
        self.assertTrue(summary["full_update_candidate"])
        self.assertNotIn("promote_as_fastest_candidate", summary)

    def test_early_kl_stop_is_honest_and_disqualifies_speed_claim(self):
        m = self.subject()
        cfg = m.ThroughputBenchmarkConfig(lanes=64, batch_size=8192)
        full = {"attempted_epochs": 4, "optimizer_steps": 16, "kl_early_stop": False}
        early = {"attempted_epochs": 1, "optimizer_steps": 3, "kl_early_stop": True}
        self.assertFalse(m.validate_update_diagnostics(full, cfg)["low_update"])
        self.assertTrue(m.validate_update_diagnostics(early, cfg)["low_update"])
        for bad in (
            {"attempted_epochs": 4, "optimizer_steps": 15, "kl_early_stop": False},
            {"attempted_epochs": 5, "optimizer_steps": 16, "kl_early_stop": False},
            {"attempted_epochs": 1, "optimizer_steps": 17, "kl_early_stop": True},
            {"attempted_epochs": 1, "optimizer_steps": 3, "kl_early_stop": 1},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.validate_update_diagnostics(bad, cfg)

    def test_learning_rate_override_marks_variant_and_updates_optimizer_groups(self):
        m = self.subject()
        cfg = m.ThroughputBenchmarkConfig(learning_rate=1e-5)
        model = SimpleNamespace(
            learning_rate=0.0001,
            lr_schedule=lambda _: 0.0001,
            policy=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 0.0001}, {"lr": 0.0001}])),
        )
        receipt = m.apply_learning_rate_variant(model, cfg)
        self.assertEqual(receipt["learning_contract_variant"], "lr_diagnostic")
        self.assertTrue(receipt["not_lossless_optimization"])
        self.assertIn("lane count, minibatch and rollout geometry are explicit sweep variables", receipt["note"])
        self.assertNotIn("PPO geometry are preserved", receipt["note"])
        self.assertEqual(model.learning_rate, 1e-5)
        self.assertEqual(model.lr_schedule(0.5), 1e-5)
        self.assertEqual([group["lr"] for group in model.policy.optimizer.param_groups], [1e-5, 1e-5])

    def test_policy_digest_changes_after_training_when_weights_move(self):
        m = self.subject()
        class Parameter:
            def __init__(self, values):
                self.values = values
                self.device = SimpleNamespace(type="cuda")
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
            def numel(self):
                return len(self.values)
        model = SimpleNamespace(policy=SimpleNamespace(parameters=lambda: [Parameter([1, 2, 3])]))
        before = m.policy_parameter_digest(model)
        model.policy.parameters = lambda: [Parameter([1, 2, 4])]
        after = m.policy_parameter_digest(model)
        self.assertNotEqual(before, after)

    def test_episode_infos_are_preserved_once_as_deterministic_gzip(self):
        m = self.subject()
        infos = [{"episode": {"r": 1.0, "l": 200}, "contact": {"diagnostics": {"hit": True}}}]
        with TemporaryDirectory() as directory:
            receipt = m.write_episode_infos(Path(directory), 3, infos)
            payload = gzip.decompress((Path(directory) / receipt["path"]).read_bytes()).decode("utf8")
            self.assertEqual(json.loads(payload), infos)
            self.assertEqual(receipt["count"], 1)
            self.assertEqual(receipt["sha256"], m.sha256(Path(directory) / receipt["path"]))

    def test_source_hashes_and_checkpoint_ack_are_required_before_cuda_setup(self):
        m = self.subject()
        with TemporaryDirectory() as directory, patch.object(m.base, "validate_checkpoint", return_value={"sha256": m.CHECKPOINT_SHA256}), patch.object(m.base, "require_checkpoint_preservation") as ack, patch.object(m.base, "require_cuda") as cuda:
            args = self.args(Path(directory))
            ack.side_effect = ValueError("missing preservation ack")
            with self.assertRaisesRegex(ValueError, "missing preservation ack"):
                m.run_block(args)
            cuda.assert_not_called()

    def test_mocked_run_uses_fresh_parent_per_block_and_saves_no_rollout_checkpoints(self):
        m = self.subject()
        cfg = m.ThroughputBenchmarkConfig(lanes=64, warmup_rollouts=1, measured_rollouts=2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, save_checkpoint=False, save_actor=False)
            saved = []
            raw = self.fake_raw(cfg)
            model = self.fake_model(cfg, saved)
            with patch.object(m, "ThroughputBenchmarkConfig", return_value=cfg), \
                 patch.object(m.base, "validate_checkpoint", return_value={"sha256": m.CHECKPOINT_SHA256}), \
                 patch.object(m.base, "require_checkpoint_preservation", return_value={"verified": True}), \
                 patch.object(m.base, "require_qualification", return_value={"qualified": True}), \
                 patch.object(m.base, "qualification_sources", return_value={"benchmark_curriculum_throughput.py": "a" * 64}), \
                 patch.object(m.base, "require_cuda", return_value="cuda:0"), \
                 patch.object(m, "build_contact_env", return_value=raw), \
                 patch.object(m, "tensor_vec_env", side_effect=lambda env, device: env), \
                 patch.object(m, "load_fresh_model", return_value=model), \
                 patch.object(m, "synchronize_cuda"), \
                 patch.object(m, "cuda_peak_memory", return_value=1234), \
                 patch.object(m, "cuda_device_name", return_value="mock-gpu"), \
                 patch.object(m, "model_parameter_count", return_value=999), \
                 patch.object(m, "seed_everything"), \
                 patch.object(m, "configure_logger", return_value=self.fake_logger()), \
                 patch.object(m, "perf_counter_sequence", side_effect=self.clock()):
                result = m.run_block(args)
            self.assertEqual(result["summary"]["measured_rollouts"], 2)
            self.assertEqual(result["status"], "measured_not_evaluated")
            self.assertEqual(result["model_parameter_count"], 999)
            self.assertEqual(result["peak_gpu_memory_bytes"], 1234)
            self.assertEqual(saved, [])
            paths = {p.relative_to(args.output).as_posix() for p in args.output.rglob("*") if p.is_file()}
            self.assertIn("result.json", paths)
            self.assertIn("manifest.json", paths)
            self.assertIn("rollout-000.json", paths)
            self.assertIn("episodes-001.json.gz", paths)
            self.assertNotIn("rollout-000.zip", paths)
            self.assertNotIn("final.zip", paths)
            row = json.loads((args.output / "rollout-001.json").read_text())
            self.assertNotIn("episodes", row)
            self.assertEqual(row["episode_info"]["count"], 1)
            self.assertEqual(gzip.decompress((args.output / row["episode_info"]["path"]).read_bytes()).decode("utf8").count("mock"), 1)
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["fresh_parent_per_block"], True)
            self.assertEqual(manifest["learning_contract_variant"], "canonical")
            self.assertEqual(manifest["timing_excludes"], "startup, scenario construction, logging writes, checkpoint writes and evaluation")
            self.assertNotEqual(result["policy_parameter_digest_before"], result["policy_parameter_digest_after"])
            self.assertEqual(result["optimizer_state_at_end"]["lr_groups"], [cfg.learning_rate])
            self.assertEqual(result["final_model_counters"]["num_timesteps"], m.CHECKPOINT_STEPS + cfg.total_rollouts * cfg.transitions_per_rollout)

    def test_mocked_finalist_can_save_one_checkpoint_and_actor(self):
        m = self.subject()
        cfg = m.ThroughputBenchmarkConfig(lanes=64, warmup_rollouts=0, measured_rollouts=1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, save_checkpoint=True, save_actor=True)
            saved = []
            raw = self.fake_raw(cfg)
            model = self.fake_model(cfg, saved)
            with patch.object(m, "ThroughputBenchmarkConfig", return_value=cfg), \
                 patch.object(m.base, "validate_checkpoint", return_value={"sha256": m.CHECKPOINT_SHA256}), \
                 patch.object(m.base, "require_checkpoint_preservation", return_value={"verified": True}), \
                 patch.object(m.base, "require_qualification", return_value={"qualified": True}), \
                 patch.object(m.base, "qualification_sources", return_value={}), \
                 patch.object(m.base, "require_cuda", return_value="cuda:0"), \
                 patch.object(m, "build_contact_env", return_value=raw), \
                 patch.object(m, "tensor_vec_env", side_effect=lambda env, device: env), \
                 patch.object(m, "load_fresh_model", return_value=model), \
                 patch.object(m, "synchronize_cuda"), \
                 patch.object(m, "cuda_peak_memory", return_value=0), \
                 patch.object(m, "cuda_device_name", return_value="mock-gpu"), \
                 patch.object(m, "model_parameter_count", return_value=10), \
                 patch.object(m, "seed_everything"), \
                 patch.object(m, "configure_logger", return_value=self.fake_logger()), \
                 patch.object(m, "perf_counter_sequence", side_effect=self.clock()):
                m.run_block(args)
            self.assertEqual(saved, ["final.tmp.zip", "actor.tmp.pt"])
            self.assertTrue((args.output / "final.zip").is_file())
            self.assertTrue((args.output / "actor.pt").is_file())
            self.assertTrue((args.output / "final-checkpoint.json").is_file())
            self.assertTrue((args.output / "actor-export.json").is_file())

    def row(self, rollout, *, warmup, total_seconds, collection_seconds=1.0, update_seconds=1.0):
        return {
            "rollout": rollout,
            "warmup": warmup,
            "transitions": 32768,
            "game_ticks": 32768,
            "collection_seconds": collection_seconds,
            "update_seconds": update_seconds,
            "total_seconds": total_seconds,
            "actual_update": {"low_update": False},
        }

    def args(self, root, *, save_checkpoint=False, save_actor=False):
        root.mkdir(parents=True, exist_ok=True)
        (root / "proof.json").write_text("{}")
        (root / "calibrated.zip").write_bytes(b"mock checkpoint")
        return SimpleNamespace(
            output=root / "run",
            checkpoint=root / "calibrated.zip",
            checkpoint_ack=None,
            frozen_root=root,
            reference_root=root,
            qualification=root / "proof.json",
            deadline_unix=10**12,
            lanes=64,
            batch_size=8192,
            warmup_rollouts=1,
            measured_rollouts=2,
            seed=145,
            device="cuda",
            update_mode="deferred_metrics",
            action_ab_arm="G_keep",
            allow_4096=False,
            allow_8192=False,
            episodes_per_lane=128,
            save_checkpoint=save_checkpoint,
            save_actor=save_actor,
            learning_rate=0.0001,
        )

    def fake_raw(self, cfg):
        backend = SimpleNamespace(
            valid_steps=[0] * cfg.lanes,
            errors=[0] * cfg.lanes,
            scenario_metadata={"generator": "test"},
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

    def fake_model(self, cfg, saved):
        model = SimpleNamespace(
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            target_kl=cfg.target_kl,
            ent_coef=cfg.ent_coef,
            n_envs=cfg.lanes,
            num_timesteps=m.CHECKPOINT_STEPS if False else 10059776,
            task="combat",
            calibration=False,
            arm="E1",
            action_head=cfg.action_head,
            action_ab_arm=cfg.action_ab_arm,
            action_init_strategy="keep",
            action_init_metadata={"arm": cfg.action_ab_arm},
            optimizer_provenance="fresh Adam",
            policy=SimpleNamespace(
                parameters=lambda: [self.fake_parameter([1, 2, model.num_timesteps % 251])],
                optimizer=SimpleNamespace(param_groups=[{"lr": cfg.learning_rate}], state={}),
            ),
            lr_schedule=lambda _: cfg.learning_rate,
            clip_range=lambda _: cfg.clip_range,
            logger=SimpleNamespace(name_to_value={}, dump=lambda **kwargs: None),
            rollout_buffer=SimpleNamespace(actions=None, executed_actions=None),
            last_update_diagnostics={"attempted_epochs": 4, "optimizer_steps": cfg.nominal_optimizer_steps, "kl_early_stop": False},
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
            callback.episodes = [{"episode": {"r": 1.0, "l": 1}, "contact": {"diagnostics": {"outcome": "mock"}}}]
            return True

        model.set_logger = lambda logger: setattr(model, "logger", logger)
        model._setup_learn = setup
        model.collect_rollouts = collect
        model._update_current_progress_remaining = lambda *args: None
        model.train = lambda: None
        model.save = lambda path: saved.append(Path(path).name) or Path(path).write_bytes(b"checkpoint")
        model.export_actor = lambda path: saved.append(Path(path).name) or Path(path).write_bytes(b"actor")
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
