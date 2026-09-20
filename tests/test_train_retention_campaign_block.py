import gzip
import importlib.util
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from contextlib import ExitStack
import unittest
from unittest.mock import patch


class RetentionCampaignBlockTests(unittest.TestCase):
    def subject(self):
        self.assertIsNotNone(
            importlib.util.find_spec("train_retention_campaign_block"),
            "retention campaign block runner has not been implemented",
        )
        import train_retention_campaign_block

        return train_retention_campaign_block

    def test_config_freezes_the_two_alternating_cuda_schedules(self):
        m = self.subject()
        self.assertEqual(m.SCHEDULES["R4C1N"], ("combat", "combat", "combat", "combat", "navigation"))
        self.assertEqual(m.SCHEDULES["R3C2N"], ("combat", "combat", "combat", "navigation", "navigation"))
        cfg = m.RetentionCampaignConfig(schedule_id="R3C2N")
        self.assertEqual(cfg.lanes, 4096)
        self.assertEqual(cfg.n_steps, 512)
        self.assertEqual(cfg.batch_size, 16384)
        self.assertEqual(cfg.n_epochs, 4)
        self.assertEqual(cfg.learning_rate, 3e-5)
        self.assertEqual(cfg.transitions_per_rollout, 2_097_152)
        self.assertEqual(cfg.total_transitions, 10_485_760)
        self.assertEqual(cfg.task_transition_totals, {"combat": 6_291_456, "navigation": 4_194_304})
        self.assertEqual(cfg.nominal_optimizer_steps, 512)
        for changes in (
            {"schedule_id": "bad"},
            {"lanes": 2048},
            {"n_steps": 256},
            {"batch_size": 8192},
            {"n_epochs": 3},
            {"learning_rate": 1e-4},
            {"device": "cpu"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.RetentionCampaignConfig(**changes)

    def test_parse_requires_schedule_and_explicit_parent_receipts(self):
        m = self.subject()
        root = Path("/tmp/root")
        base = [
            "--checkpoint", str(root / "checkpoint.zip"),
            "--checkpoint-sha256", "a" * 64,
            "--checkpoint-ack", str(root / "preservation_ack.json"),
            "--frozen-root", str(root),
            "--reference-root", str(root),
            "--qualification", str(root / "proof.json"),
            "--output", str(root / "out"),
            "--deadline-unix", "9999999999",
        ]
        with self.assertRaises(SystemExit):
            m.parse_args(base)
        args = m.parse_args(base + ["--schedule", "R4C1N", "--resume"])
        self.assertEqual(args.schedule, "R4C1N")
        self.assertTrue(args.resume)

    def test_mocked_success_writes_homogeneous_rollout_rows_task_episode_files_and_one_final_model(self):
        m = self.subject()
        cfg = m.RetentionCampaignConfig(schedule_id="R3C2N")
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), schedule="R3C2N", resume=False)
            contact_env = self.fake_env(cfg, "combat")
            navigation_env = self.fake_env(cfg, "navigation")
            model = self.fake_model(cfg, start_timesteps=m.throughput.CHECKPOINT_STEPS)
            with self.common_patches(m, cfg, contact_env, navigation_env, model), \
                 patch.object(m.contact_block, "load_fresh_campaign_model", return_value=model) as load_fresh, \
                 patch.object(m.contact_block, "load_resume_campaign_model") as load_resume, \
                 patch.object(m.throughput, "apply_learning_rate_variant", wraps=m.throughput.apply_learning_rate_variant):
                result = m.run_block(args)

            load_fresh.assert_called_once()
            load_resume.assert_not_called()
            self.assertEqual(model.load_generation, 1)
            self.assertEqual(model.switches, list(cfg.schedule))
            self.assertEqual(model.collect_tasks, list(cfg.schedule))
            self.assertEqual(model.train_count, len(cfg.schedule))
            self.assertEqual(model.save_count, 1)
            self.assertEqual(model.export_count, 1)
            self.assertEqual(result["status"], "trained_not_evaluated")
            self.assertEqual(result["task_transition_totals"], cfg.task_transition_totals)
            self.assertEqual(result["final_model_counters"]["num_timesteps"], m.throughput.CHECKPOINT_STEPS + cfg.total_transitions)
            self.assertTrue((args.output / "final.zip").is_file())
            self.assertTrue((args.output / "actor.pt").is_file())
            self.assertTrue((args.output / "inventory.json").is_file())

            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["schedule"], list(cfg.schedule))
            self.assertIn("model/critic/optimizer/RNG counters resume", manifest["rng_and_environment_note"])
            self.assertIn("live environment state does not resume across immutable blocks", manifest["rng_and_environment_note"])

            rows = [json.loads((args.output / f"rollout-{index:03d}.json").read_text()) for index in range(5)]
            self.assertEqual([row["task"] for row in rows], list(cfg.schedule))
            self.assertTrue(all(row["task"] == row["episode_info"]["task"] for row in rows))
            self.assertTrue(all(row["transitions"] == cfg.transitions_per_rollout for row in rows))
            self.assertEqual(rows[0]["curriculum_version"], m.TASK_VERSIONS["combat"])
            self.assertEqual(rows[-1]["curriculum_version"], m.TASK_VERSIONS["navigation"])
            self.assertEqual(rows[-1]["cumulative_task_transitions"]["navigation"], 2 * cfg.transitions_per_rollout)
            self.assertEqual(rows[-1]["cumulative_task_transitions"]["combat"], 3 * cfg.transitions_per_rollout)
            self.assertEqual(rows[0]["episode_info"]["path"], "episodes-combat-000.json.gz")
            self.assertEqual(rows[-1]["episode_info"]["path"], "episodes-navigation-004.json.gz")
            payload = gzip.decompress((args.output / rows[-1]["episode_info"]["path"]).read_bytes())
            self.assertEqual(json.loads(payload)[0]["navigation"]["lane"], 0)
            self.assertNotIn("rollout-000.zip", {p.name for p in args.output.iterdir()})

    def test_retention_episode_extractor_accepts_active_task_records_without_contact_helper(self):
        m = self.subject()
        mask = [bool(value) for value in m.TASK_MASKS["navigation"]]
        callback = SimpleNamespace(episodes=[{
            "episode": {"r": 1.0, "l": 4},
            "navigation": {"lane": 0, "valid_steps": 4},
            "curriculum_task": "navigation",
            "curriculum_version": m.TASK_VERSIONS["navigation"],
            "allowed_control_mask": mask,
        }])
        with patch.object(m.base, "completed_episode_infos", side_effect=AssertionError("contact helper called")):
            infos = m.completed_retention_episode_infos(callback, "navigation")
        self.assertEqual(infos, callback.episodes)

        bad_mask = dict(callback.episodes[0])
        bad_mask["allowed_control_mask"] = [int(value) for value in mask]
        callback.episodes = [bad_mask]
        with self.assertRaisesRegex(RuntimeError, "task/mask/version"):
            m.completed_retention_episode_infos(callback, "navigation")

        bad_payload = dict(bad_mask)
        bad_payload["allowed_control_mask"] = mask
        bad_payload.pop("navigation")
        callback.episodes = [bad_payload]
        with self.assertRaisesRegex(RuntimeError, "navigation payload"):
            m.completed_retention_episode_infos(callback, "navigation")

    def test_retention_runtime_validator_accepts_requested_task_without_contact_only_rejection(self):
        m = self.subject()
        cfg = m.RetentionCampaignConfig(schedule_id="R4C1N")
        model = self.fake_model(cfg, start_timesteps=m.throughput.CHECKPOINT_STEPS)
        model.policy.allowed_control_mask = [bool(value) for value in m.TASK_MASKS["navigation"]]
        model.task = "navigation"
        model.policy.task = "navigation"
        optimizer = m.validate_retention_runtime(model, cfg, fresh=False, requested_task="navigation")
        self.assertEqual(optimizer["state_tensor_elements"], 3)

        model.calibration = True
        with self.assertRaisesRegex(ValueError, "non-calibration"):
            m.validate_retention_runtime(model, cfg, fresh=False, requested_task="navigation")

    def test_resume_uses_existing_checkpoint_once_without_reconstructing_policy_or_optimizer(self):
        m = self.subject()
        cfg = m.RetentionCampaignConfig(schedule_id="R4C1N")
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), schedule="R4C1N", resume=True, checkpoint_sha="b" * 64)
            args.checkpoint.write_bytes(b"resume checkpoint")
            contact_env = self.fake_env(cfg, "combat")
            navigation_env = self.fake_env(cfg, "navigation")
            model = self.fake_model(cfg, start_timesteps=m.throughput.CHECKPOINT_STEPS + cfg.transitions_per_rollout)
            with self.common_patches(m, cfg, contact_env, navigation_env, model), \
                 patch.object(m, "sha256", return_value="b" * 64), \
                 patch.object(m.contact_block, "load_fresh_campaign_model") as load_fresh, \
                 patch.object(m.contact_block, "load_resume_campaign_model", return_value=model) as load_resume, \
                 patch.object(m.throughput, "apply_learning_rate_variant") as apply_lr:
                result = m.run_block(args)
            load_fresh.assert_not_called()
            load_resume.assert_called_once()
            apply_lr.assert_not_called()
            self.assertEqual(model.load_generation, 1)
            self.assertEqual(result["resume_from_checkpoint_sha256"], "b" * 64)
            self.assertGreater(result["optimizer_state_at_end"]["state_tensor_elements"], 0)

    def test_resume_requests_first_rollout_combat_env_even_after_navigation_saved_checkpoint(self):
        m = self.subject()
        cfg = m.RetentionCampaignConfig(schedule_id="R4C1N")
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), schedule="R4C1N", resume=True, checkpoint_sha="c" * 64)
            args.checkpoint.write_bytes(b"navigation-saved checkpoint")
            contact_env = self.fake_env(cfg, "combat")
            navigation_env = self.fake_env(cfg, "navigation")
            model = self.fake_model(cfg, start_timesteps=m.throughput.CHECKPOINT_STEPS + cfg.transitions_per_rollout)
            load_envs = []

            def load_resume(checkpoint, env, cfg_arg, expected_sha256):
                load_envs.append(env)
                self.assertIs(env, contact_env)
                self.assertEqual(env.task, "combat")
                model.task = "combat"
                model.policy.task = "combat"
                return model

            with self.common_patches(m, cfg, contact_env, navigation_env, model), \
                 patch.object(m, "sha256", return_value="c" * 64), \
                 patch.object(m.contact_block, "load_resume_campaign_model", side_effect=load_resume):
                m.run_block(args)
            self.assertEqual(load_envs, [contact_env])
            self.assertEqual(model.collect_tasks[0], "combat")

    def test_qualification_smoke_validates_real_alternating_contract_from_runner_hooks(self):
        m = self.subject()
        cfg_dict = {
            "lanes": 4,
            "n_steps": 2,
            "batch_size": 4,
            "n_epochs": 1,
            "seed": 9,
            "device": "cuda",
            "learning_rate": 3e-5,
        }
        smoke_cfg = SimpleNamespace(
            lanes=4,
            n_steps=2,
            batch_size=4,
            n_epochs=1,
            seed=9,
            device="cuda",
            learning_rate=3e-5,
            update_mode="deferred_metrics",
            action_ab_arm="G_keep",
            action_head=m.ACTION_EXPERIMENT_ARMS["G_keep"]["action_head"],
            clip_range=0.1,
            target_kl=0.01,
            ent_coef=0.001,
            gamma=0.99,
            gae_lambda=0.95,
            episodes_per_lane=16,
            record_capacity=8,
            nominal_optimizer_steps=2,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.zip"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "smoke"
            contact_env = self.fake_env(smoke_cfg, "combat")
            navigation_env = self.fake_env(smoke_cfg, "navigation")
            model = self.fake_smoke_model(smoke_cfg)
            loaded_navigation = self.fake_smoke_model(smoke_cfg)
            loaded_combat = self.fake_smoke_model(smoke_cfg)
            load_calls = []

            def load_model(path, *, env, expected_sha256):
                load_calls.append(env.task)
                if len(load_calls) == 1:
                    return model
                return loaded_navigation if env.task == "navigation" else loaded_combat

            with patch.object(m, "RetentionCampaignConfig", return_value=smoke_cfg), \
                 patch.object(m.base, "require_cuda", return_value="cuda:0"), \
                 patch.object(m.throughput, "build_contact_env", return_value=contact_env), \
                 patch.object(m, "build_navigation_curriculum_env", return_value=navigation_env), \
                 patch.object(m.throughput, "tensor_vec_env", side_effect=lambda env, device: env), \
                 patch.object(m.throughput, "configure_logger", return_value=self.fake_logger()), \
                 patch.object(m.throughput, "synchronize_cuda"), \
                 patch.object(m.throughput, "optimizer_state_summary", side_effect=lambda ppo: ppo.optimizer_summary()), \
                 patch.object(m.throughput, "policy_parameter_digest", side_effect=lambda ppo: ppo.actor_digest()), \
                 patch.object(m, "sha256", return_value="d" * 64), \
                 patch.object(m, "save_checkpoint", side_effect=lambda path, ppo: ppo.save(path) or {"path": str(path), "sha256": "e" * 64}), \
                 patch.object(m.NativeCurriculumPPO, "load", side_effect=load_model):
                result = m.run_qualification_smoke(
                    schedule=("navigation", "combat"),
                    cfg=cfg_dict,
                    source=root,
                    config=root / "config.yaml",
                    build_dir=root / "build",
                    output_dir=output,
                    checkpoint=checkpoint,
                )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["schedule"], ["navigation", "combat"])
            self.assertEqual(result["actor_changed"], {"navigation": True, "combat": True})
            self.assertEqual(result["critic_change_matrix"]["navigation"], {"navigation": True, "combat": False})
            self.assertEqual(result["critic_change_matrix"]["combat"], {"navigation": False, "combat": True})
            self.assertEqual(result["optimizer_steps"], 4)
            self.assertEqual(result["nominal_optimizer_steps"], 4)
            self.assertFalse(result["kl_early_stop"])
            self.assertTrue(result["save_load_exact"])
            self.assertEqual(result["next_task_continuation"], {"navigation": True, "combat": True})
            self.assertEqual(load_calls, ["navigation", "navigation", "combat"])
            self.assertTrue((output / "alternating-smoke.json").is_file())

    def test_existing_output_directory_is_never_overwritten(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), schedule="R4C1N", resume=False)
            args.output.mkdir()
            with self.assertRaises(FileExistsError):
                m.run_block(args)

    def test_failure_preserves_recovery_checkpoint_and_failure_receipt(self):
        m = self.subject()
        cfg = m.RetentionCampaignConfig(schedule_id="R4C1N")
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), schedule="R4C1N", resume=False)
            contact_env = self.fake_env(cfg, "combat")
            navigation_env = self.fake_env(cfg, "navigation")
            model = self.fake_model(cfg, start_timesteps=m.throughput.CHECKPOINT_STEPS, fail_on_collect=1)
            with self.common_patches(m, cfg, contact_env, navigation_env, model), \
                 patch.object(m.contact_block, "load_fresh_campaign_model", return_value=model), \
                 self.assertRaisesRegex(RuntimeError, "injected collection failure"):
                m.run_block(args)
            self.assertTrue((args.output / "failure-last.zip").is_file())
            self.assertTrue((args.output / "failure.json").is_file())
            failure = json.loads((args.output / "failure.json").read_text())
            self.assertEqual(failure["completed_rollouts"], 1)
            self.assertEqual(failure["last_completed_checkpoint"], "failure-last.zip")

    def args(self, root, *, schedule, resume, checkpoint_sha=None):
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
            schedule=schedule,
            batch_size=16384,
            episodes_per_lane=1024,
            seed=145,
            learning_rate=3e-5,
            device="cuda",
            update_mode="deferred_metrics",
            action_ab_arm="G_keep",
            resume=resume,
        )

    def common_patches(self, m, cfg, contact_env, navigation_env, model):
        patches = [
            patch.object(m, "RetentionCampaignConfig", return_value=cfg),
            patch.object(m.base, "require_checkpoint_preservation", return_value={"verified": True}),
            patch.object(m.base, "require_qualification", return_value={"qualified": True}),
            patch.object(m.base, "qualification_sources", return_value={}),
            patch.object(m.base, "require_cuda", return_value="cuda:0"),
            patch.object(m.throughput, "seed_everything"),
            patch.object(m.throughput, "build_contact_env", return_value=contact_env),
            patch.object(m, "build_navigation_curriculum_env", return_value=navigation_env),
            patch.object(m.throughput, "tensor_vec_env", side_effect=lambda env, device: env),
            patch.object(m.throughput, "synchronize_cuda"),
            patch.object(m.throughput, "cuda_peak_memory", return_value=5678),
            patch.object(m.throughput, "cuda_device_name", return_value="mock-gpu"),
            patch.object(m.throughput, "model_parameter_count", return_value=15_000_000),
            patch.object(m.throughput, "configure_logger", return_value=self.fake_logger()),
            patch.object(m.throughput, "perf_counter_sequence", side_effect=self.clock()),
            patch.object(m, "source_hashes", return_value={"source.py": "a" * 64}),
            patch.object(m, "verify_input_checkpoint", return_value=(
                {"path": "checkpoint.zip", "sha256": "301ce3c72924f3e5c62f265f88292b8952e86e6d71d9a0552213cb8f3c6784b6"},
                {"verified": True},
            )),
            patch.object(m.contact_block, "validate_campaign_runtime", return_value={"available": True, "state_tensor_elements": 6}),
        ]
        class Stack:
            def __enter__(self_nonlocal):
                self_nonlocal.stack = ExitStack()
                self_nonlocal.entered = [self_nonlocal.stack.enter_context(p) for p in patches]
                return self_nonlocal.entered
            def __exit__(self_nonlocal, *exc):
                return self_nonlocal.stack.__exit__(*exc)
        return Stack()

    def fake_env(self, cfg, task):
        task_masks = self.subject().TASK_MASKS
        task_versions = self.subject().TASK_VERSIONS
        backend = SimpleNamespace(
            valid_steps=[0] * cfg.lanes,
            errors=[0] * cfg.lanes,
            scenario_metadata={"task": task, "lanes": cfg.lanes, "episodes_per_lane": cfg.episodes_per_lane},
        )
        return SimpleNamespace(
            task=task,
            curriculum_task=task,
            curriculum_version=task_versions[task],
            allowed_control_mask=[bool(value) for value in task_masks[task]],
            backend=backend,
            host_has_timeouts=lambda: task == "navigation",
            close=lambda: None,
        )

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

    def fake_model(self, cfg, *, start_timesteps, fail_on_collect=None):
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
            _total_timesteps=start_timesteps,
            task="combat",
            calibration=False,
            arm="E1",
            action_head=cfg.action_head,
            action_ab_arm=cfg.action_ab_arm,
            policy=SimpleNamespace(
                task="combat",
                calibration=False,
                allowed_control_mask=[True] * 10,
                parameters=lambda: [self.fake_parameter([1, 2, model.num_timesteps % 251])],
                optimizer=SimpleNamespace(
                    param_groups=[{"lr": cfg.learning_rate}],
                    state={1: {"step": 2, "exp_avg": self.fake_parameter([1, 2, 3])}},
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
            switches=[],
            collect_tasks=[],
            train_count=0,
            save_count=0,
            export_count=0,
            load_generation=1,
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

        def switch(task, env, calibration=False):
            self.assertIs(env.task, task)
            self.assertFalse(calibration)
            model.task = task
            model.switches.append(task)

        def collect(env, callback, buffer, n_steps):
            index = len(model.collect_tasks)
            if fail_on_collect is not None and index == fail_on_collect:
                raise RuntimeError("injected collection failure")
            self.assertEqual(env.task, model.task)
            model.collect_tasks.append(model.task)
            model.num_timesteps += cfg.transitions_per_rollout
            env.backend.valid_steps = [value + n_steps for value in env.backend.valid_steps]
            payload_key = "contact" if model.task == "combat" else "navigation"
            callback.episodes = [
                {
                    payload_key: {"lane": 0, "episode_index": model.num_timesteps},
                    "episode": {"r": 1.0, "l": 1},
                    "curriculum_task": model.task,
                    "curriculum_version": "bridge_b1_contact-v1" if model.task == "combat" else "bridge_0b_navigation-v1",
                }
            ]
            return True

        def train():
            model.train_count += 1

        def save(path):
            model.save_count += 1
            Path(path).write_bytes(b"checkpoint")

        def export(path):
            model.export_count += 1
            Path(path).write_bytes(b"actor")

        model.set_logger = lambda logger: setattr(model, "logger", logger)
        model._setup_learn = setup
        model.set_task_environment = switch
        model.collect_rollouts = collect
        model._update_current_progress_remaining = lambda *args: None
        model.train = train
        model.save = save
        model.export_actor = export
        return model

    def fake_smoke_model(self, cfg):
        model = self.fake_model(cfg, start_timesteps=1000)
        model.actor_value = 1
        model.navigation_critic_value = 10
        model.combat_critic_value = 20
        model.optimizer_steps_total = 0
        model.task_history = []
        model.saved_state = None

        def actor_digest():
            return f"actor:{model.actor_value}"

        def critic_digests():
            return {
                "navigation": f"nav:{model.navigation_critic_value}",
                "combat": f"combat:{model.combat_critic_value}",
            }

        def optimizer_summary():
            return {"available": True, "state_tensor_elements": 3, "optimizer_steps_total": model.optimizer_steps_total}

        def train():
            model.train_count += 1
            model.actor_value += 1
            if model.task == "navigation":
                model.navigation_critic_value += 1
            else:
                model.combat_critic_value += 1
            model.optimizer_steps_total += cfg.nominal_optimizer_steps
            model.last_update_diagnostics = {
                "attempted_epochs": cfg.n_epochs,
                "optimizer_steps": cfg.nominal_optimizer_steps,
                "kl_early_stop": False,
            }

        def save(path):
            model.saved_state = {
                "actor": model.actor_digest(),
                "critics": model.critic_digests(),
                "optimizer": model.optimizer_summary(),
                "counters": {"num_timesteps": model.num_timesteps},
                "task_history": list(model.task_history),
                "masks": [bool(value) for value in model.policy.allowed_control_mask],
            }
            Path(path).write_bytes(json.dumps(model.saved_state, sort_keys=True).encode())

        def switch(task, env, calibration=False):
            self.assertEqual(env.task, task)
            model.task = task
            model.policy.task = task
            model.policy.calibration = calibration
            model.policy.allowed_control_mask = [bool(value) for value in self.subject().TASK_MASKS[task]]
            model.switches.append(task)
            model.task_history.append({"task": task, "num_timesteps": model.num_timesteps})

        model.actor_digest = actor_digest
        model.critic_digests = critic_digests
        model.optimizer_summary = optimizer_summary
        model.train = train
        model.save = save
        model.set_task_environment = switch
        model.loaded_exact_state = lambda state: state == model.saved_state if model.saved_state is not None else True
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
