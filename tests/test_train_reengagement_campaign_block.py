import gzip
import importlib.util
import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


class ReengagementCampaignBlockTests(unittest.TestCase):
    def subject(self):
        self.assertIsNotNone(
            importlib.util.find_spec("train_reengagement_campaign_block"),
            "B2 reengagement campaign block runner has not been implemented",
        )
        import train_reengagement_campaign_block

        return train_reengagement_campaign_block

    def test_config_freezes_r4e1n_schedule_parent_and_ppo_contract(self):
        m = self.subject()
        self.assertEqual(
            m.SCHEDULES,
            {"R4E1N": ("reengagement", "reengagement", "reengagement", "reengagement", "navigation")},
        )
        self.assertEqual(
            m.PARENT_CHECKPOINT_SHA256,
            "d6e487c22f73913cd88a428683c7ca721e23b1f9cb5a3a561ad24d1f5bf410db",
        )
        self.assertEqual(m.PARENT_LIFETIME_TRANSITIONS, 60_391_424)
        cfg = m.ReengagementCampaignConfig()
        self.assertEqual(cfg.schedule_id, "R4E1N")
        self.assertEqual(cfg.lanes, 4096)
        self.assertEqual(cfg.n_steps, 512)
        self.assertEqual(cfg.batch_size, 16384)
        self.assertEqual(cfg.n_epochs, 4)
        self.assertEqual(cfg.learning_rate, 3e-5)
        self.assertEqual(cfg.clip_range, 0.1)
        self.assertEqual(cfg.target_kl, 0.01)
        self.assertEqual(cfg.ent_coef, 0.001)
        self.assertEqual(cfg.gamma, 0.99)
        self.assertEqual(cfg.gae_lambda, 0.95)
        self.assertEqual(cfg.action_ab_arm, "B_projected")
        self.assertEqual(tuple(m.SUPPORTED_ARMS), ("B_projected",))
        self.assertEqual(cfg.transitions_per_rollout, 2_097_152)
        self.assertEqual(cfg.total_transitions, 10_485_760)
        self.assertEqual(
            cfg.task_transition_totals,
            {"reengagement": 8_388_608, "navigation": 2_097_152},
        )
        self.assertEqual(cfg.nominal_optimizer_steps, 512)
        for changes in (
            {"schedule_id": "R3C2N"},
            {"lanes": 2048},
            {"n_steps": 256},
            {"batch_size": 8192},
            {"n_epochs": 3},
            {"learning_rate": 1e-4},
            {"device": "cpu"},
            {"action_ab_arm": "G_keep"},
            {"clip_range": 0.2},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.ReengagementCampaignConfig(**changes)

    def test_parse_requires_explicit_schedule_and_parent_sha(self):
        m = self.subject()
        root = Path("/tmp/root")
        base = [
            "--checkpoint", str(root / "checkpoint.zip"),
            "--checkpoint-sha256", m.PARENT_CHECKPOINT_SHA256,
            "--checkpoint-ack", str(root / "preservation_ack.json"),
            "--frozen-root", str(root),
            "--reference-root", str(root),
            "--qualification", str(root / "proof.json"),
            "--output", str(root / "out"),
            "--deadline-unix", "9999999999",
        ]
        with self.assertRaises(SystemExit):
            m.parse_args(base)
        args = m.parse_args(base + ["--schedule", "R4E1N"])
        self.assertEqual(args.schedule, "R4E1N")
        with self.assertRaises(SystemExit):
            m.parse_args([*base[:-11], "--checkpoint-sha256", "a" * 64, *base[-10:], "--schedule", "R4E1N"])

    def test_parent_hash_and_preservation_are_verified_before_cuda_env_or_model(self):
        m = self.subject()
        order = []
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            args.checkpoint.write_bytes(b"canonical parent")
            with patch.object(m, "sha256", side_effect=lambda path: order.append("sha256") or "bad"), \
                 patch.object(m.base, "require_checkpoint_preservation", side_effect=lambda *a: order.append("ack") or {}), \
                 patch.object(m.base, "require_cuda", side_effect=lambda *a: order.append("cuda") or "cuda:0"), \
                 patch.object(m, "build_task_environments", side_effect=lambda *a: order.append("env") or {}), \
                 patch.object(m.NativeCurriculumPPO, "load_contact_checkpoint", side_effect=AssertionError("model load reached")), \
                 self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                m.run_block(args)
            self.assertEqual(order, ["sha256"])

            order.clear()
            with patch.object(m, "sha256", side_effect=lambda path: order.append("sha256") or m.PARENT_CHECKPOINT_SHA256), \
                 patch.object(m.base, "require_checkpoint_preservation", side_effect=RuntimeError("ack missing")), \
                 patch.object(m.base, "require_cuda", side_effect=lambda *a: order.append("cuda") or "cuda:0"), \
                 patch.object(m, "build_task_environments", side_effect=lambda *a: order.append("env") or {}), \
                 patch.object(m.NativeCurriculumPPO, "load_contact_checkpoint", side_effect=AssertionError("model load reached")), \
                 self.assertRaisesRegex(RuntimeError, "ack missing"):
                m.run_block(args)
            self.assertEqual(order, ["sha256"])

    def test_mocked_success_runs_exact_schedule_and_preserves_single_model_across_switches(self):
        m = self.subject()
        cfg = m.ReengagementCampaignConfig()
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            reengagement_env = self.fake_env(m, cfg, "reengagement")
            navigation_env = self.fake_env(m, cfg, "navigation")
            model = self.fake_model(m, cfg, start_timesteps=m.PARENT_LIFETIME_TRANSITIONS)
            with self.common_patches(m, cfg, reengagement_env, navigation_env, model), \
                 patch.object(m.NativeCurriculumPPO, "load_contact_checkpoint", return_value=model) as load_parent:
                result = m.run_block(args)

            load_parent.assert_called_once()
            _, kwargs = load_parent.call_args
            self.assertEqual(kwargs["expected_sha256"], m.PARENT_CHECKPOINT_SHA256)
            self.assertEqual(kwargs["expected_task_versions"]["combat"], m.REENGAGEMENT_VERSION)
            self.assertEqual(kwargs["expected_task_versions"]["navigation"], m.NAVIGATION_VERSION)
            self.assertEqual(model.switches, [
                ("combat", m.REENGAGEMENT_VERSION),
                ("combat", m.REENGAGEMENT_VERSION),
                ("combat", m.REENGAGEMENT_VERSION),
                ("combat", m.REENGAGEMENT_VERSION),
                ("navigation", m.NAVIGATION_VERSION),
            ])
            self.assertEqual(model.collect_tasks, ["reengagement", "reengagement", "reengagement", "reengagement", "navigation"])
            self.assertEqual(model.train_count, 5)
            self.assertEqual(result["status"], "trained_not_evaluated")
            self.assertEqual(result["schedule"], list(cfg.schedule))
            self.assertEqual(result["task_transition_totals"], cfg.task_transition_totals)
            self.assertEqual(
                result["final_model_counters"]["num_timesteps"],
                m.PARENT_LIFETIME_TRANSITIONS + cfg.total_transitions,
            )
            self.assertTrue((args.output / "final.zip").is_file())
            self.assertTrue((args.output / "actor.pt").is_file())
            self.assertTrue((args.output / "inventory.json").is_file())

            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["checkpoint"]["sha256"], m.PARENT_CHECKPOINT_SHA256)
            self.assertEqual(manifest["expected_task_versions"]["combat"], m.REENGAGEMENT_VERSION)
            self.assertEqual(manifest["parent_lifetime_transitions"], m.PARENT_LIFETIME_TRANSITIONS)
            self.assertEqual(manifest["claim"], "B2 reengagement/native-navigation training block; evaluation and promotion are separate")

            rows = [json.loads((args.output / f"rollout-{index:03d}.json").read_text()) for index in range(5)]
            self.assertEqual([row["task"] for row in rows], list(cfg.schedule))
            self.assertEqual(rows[0]["curriculum_version"], m.REENGAGEMENT_VERSION)
            self.assertEqual(rows[3]["curriculum_version"], m.REENGAGEMENT_VERSION)
            self.assertEqual(rows[4]["curriculum_version"], m.NAVIGATION_VERSION)
            self.assertEqual(rows[4]["cumulative_task_transitions"], cfg.task_transition_totals)
            self.assertEqual(rows[0]["episode_info"]["path"], "episodes-reengagement-000.json.gz")
            self.assertEqual(rows[4]["episode_info"]["path"], "episodes-navigation-004.json.gz")
            payload = gzip.decompress((args.output / rows[0]["episode_info"]["path"]).read_bytes())
            self.assertEqual(json.loads(payload)[0]["reengagement"]["lane"], 0)
            self.assertNotIn("promoted", json.dumps(result).lower())

    def test_episode_extractor_rejects_wrong_task_version_mask_or_payload(self):
        m = self.subject()
        mask = [bool(value) for value in m.TASK_MASKS["combat"]]
        callback = SimpleNamespace(episodes=[{
            "episode": {"r": 1.0, "l": 4},
            "reengagement": {"lane": 0, "diagnostics": {"success": False}},
            "curriculum_task": "combat",
            "curriculum_version": m.REENGAGEMENT_VERSION,
            "allowed_control_mask": mask,
        }])
        infos = m.completed_reengagement_episode_infos(callback, "reengagement")
        self.assertEqual(infos, callback.episodes)
        for bad in (
            {**callback.episodes[0], "curriculum_version": "bridge_b1_contact-v1"},
            {**callback.episodes[0], "allowed_control_mask": [int(value) for value in mask]},
            {key: value for key, value in callback.episodes[0].items() if key != "reengagement"},
        ):
            callback.episodes = [bad]
            with self.assertRaisesRegex(RuntimeError, "task/mask/version|reengagement payload"):
                m.completed_reengagement_episode_infos(callback, "reengagement")

    def test_failure_preserves_recovery_checkpoint_and_receipt(self):
        m = self.subject()
        cfg = m.ReengagementCampaignConfig()
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            reengagement_env = self.fake_env(m, cfg, "reengagement")
            navigation_env = self.fake_env(m, cfg, "navigation")
            model = self.fake_model(m, cfg, start_timesteps=m.PARENT_LIFETIME_TRANSITIONS, fail_on_collect=1)
            with self.common_patches(m, cfg, reengagement_env, navigation_env, model), \
                 patch.object(m.NativeCurriculumPPO, "load_contact_checkpoint", return_value=model), \
                 self.assertRaisesRegex(RuntimeError, "injected collection failure"):
                m.run_block(args)
            self.assertTrue((args.output / "failure-last.zip").is_file())
            failure = json.loads((args.output / "failure.json").read_text())
            self.assertEqual(failure["completed_rollouts"], 1)
            self.assertEqual(failure["last_completed_checkpoint"], "failure-last.zip")

    def args(self, root):
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "checkpoint.zip"
        checkpoint.write_bytes(b"canonical checkpoint")
        ack = root / "preservation_ack.json"
        ack.write_text("{}")
        qualification = root / "proof.json"
        qualification.write_text("{}")
        m = self.subject()
        return SimpleNamespace(
            checkpoint=checkpoint,
            checkpoint_sha256=m.PARENT_CHECKPOINT_SHA256,
            checkpoint_ack=ack,
            frozen_root=root,
            reference_root=root,
            qualification=qualification,
            output=root / "run",
            deadline_unix=10**12,
            lanes=4096,
            schedule="R4E1N",
            batch_size=16384,
            episodes_per_lane=1024,
            seed=145,
            learning_rate=3e-5,
            device="cuda",
            update_mode="deferred_metrics",
            action_ab_arm="B_projected",
        )

    def common_patches(self, m, cfg, reengagement_env, navigation_env, model):
        wrappers = {"reengagement": self.fake_wrapper(reengagement_env), "navigation": self.fake_wrapper(navigation_env)}
        patches = [
            patch.object(m, "ReengagementCampaignConfig", return_value=cfg),
            patch.object(m, "verify_input_checkpoint", return_value=(
                {"path": "checkpoint.zip", "sha256": m.PARENT_CHECKPOINT_SHA256, "canonical_parent": True},
                {"verified": True},
            )),
            patch.object(m.base, "require_qualification", return_value={"qualified": True}),
            patch.object(m.base, "qualification_sources", return_value={}),
            patch.object(m.base, "require_cuda", return_value="cuda:0"),
            patch.object(m.throughput, "seed_everything"),
            patch.object(m, "build_task_environments", return_value=wrappers),
            patch.object(m.throughput, "synchronize_cuda"),
            patch.object(m.throughput, "reset_peak_memory"),
            patch.object(m.throughput, "cuda_peak_memory", return_value=5678),
            patch.object(m.throughput, "cuda_device_name", return_value="mock-gpu"),
            patch.object(m.throughput, "model_parameter_count", return_value=15_000_000),
            patch.object(m.throughput, "configure_logger", return_value=self.fake_logger()),
            patch.object(m.throughput, "perf_counter_sequence", side_effect=self.clock()),
            patch.object(m, "source_hashes", return_value={"source.py": "a" * 64}),
            patch.object(m.throughput, "optimizer_state_summary", side_effect=lambda ppo: ppo.optimizer_summary()),
            patch.object(m.throughput, "verified_action_initialization", return_value={"arm": "B_projected"}),
        ]

        class Stack:
            def __enter__(self_nonlocal):
                self_nonlocal.stack = ExitStack()
                return [self_nonlocal.stack.enter_context(p) for p in patches]

            def __exit__(self_nonlocal, *exc):
                return self_nonlocal.stack.__exit__(*exc)

        return Stack()

    def fake_env(self, m, cfg, task):
        model_task = "combat" if task == "reengagement" else "navigation"
        version = m.REENGAGEMENT_VERSION if task == "reengagement" else m.NAVIGATION_VERSION
        backend = SimpleNamespace(
            valid_steps=[0] * cfg.lanes,
            errors=[0] * cfg.lanes,
            scenario_metadata={"task": task, "version": version, "lanes": cfg.lanes},
        )
        return SimpleNamespace(
            task=task,
            model_task=model_task,
            curriculum_task=model_task,
            curriculum_version=version,
            allowed_control_mask=[bool(value) for value in m.TASK_MASKS[model_task]],
            backend=backend,
            host_has_timeouts=lambda: task == "navigation",
            close=lambda: None,
        )

    def fake_wrapper(self, raw):
        return SimpleNamespace(
            env=raw,
            task=raw.task,
            model_task=raw.model_task,
            close=lambda: raw.close(),
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

    def fake_model(self, m, cfg, *, start_timesteps, fail_on_collect=None):
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
            _n_updates=0,
            _episode_num=0,
            task="combat",
            calibration=False,
            arm="E1",
            action_head=cfg.action_head,
            action_ab_arm=cfg.action_ab_arm,
            policy=SimpleNamespace(
                task="combat",
                calibration=False,
                allowed_control_mask=[bool(value) for value in m.TASK_MASKS["combat"]],
                parameters=lambda: [self.fake_parameter([1, 2, model.num_timesteps % 251])],
                optimizer=SimpleNamespace(param_groups=[{"lr": cfg.learning_rate}], state={1: {"step": 2, "exp_avg": self.fake_parameter([1, 2, 3])}}),
            ),
            lr_schedule=lambda _: cfg.learning_rate,
            clip_range=lambda _: cfg.clip_range,
            logger=SimpleNamespace(name_to_value={}, dump=lambda **kwargs: None),
            rollout_buffer=SimpleNamespace(actions=None, executed_actions=None),
            last_update_diagnostics={"attempted_epochs": 4, "optimizer_steps": cfg.nominal_optimizer_steps, "kl_early_stop": False},
            switches=[],
            collect_tasks=[],
            train_count=0,
            save_count=0,
            export_count=0,
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

        def switch(model_task, env, calibration=False, expected_task_versions=None):
            raw_env = getattr(env, "env", env)
            self.assertEqual(model_task, raw_env.model_task)
            self.assertFalse(calibration)
            self.assertEqual(expected_task_versions[model_task], raw_env.curriculum_version)
            model.task = model_task
            model.policy.task = model_task
            model.policy.allowed_control_mask = [bool(value) for value in m.TASK_MASKS[model_task]]
            model.switches.append((model_task, expected_task_versions[model_task]))

        def collect(env, callback, buffer, n_steps):
            index = len(model.collect_tasks)
            if fail_on_collect is not None and index == fail_on_collect:
                raise RuntimeError("injected collection failure")
            raw_env = getattr(env, "env", env)
            model.collect_tasks.append(raw_env.task)
            model.num_timesteps += cfg.transitions_per_rollout
            raw_env.backend.valid_steps = [value + n_steps for value in raw_env.backend.valid_steps]
            payload_key = "reengagement" if raw_env.task == "reengagement" else "navigation"
            payload = {"lane": 0, "episode_index": model.num_timesteps}
            if raw_env.task == "reengagement":
                payload["diagnostics"] = {"success": False}
            callback.episodes = [{
                payload_key: payload,
                "episode": {"r": 1.0, "l": 1},
                "curriculum_task": raw_env.model_task,
                "curriculum_version": raw_env.curriculum_version,
                "allowed_control_mask": [bool(value) for value in m.TASK_MASKS[raw_env.model_task]],
            }]
            return True

        model.set_logger = lambda logger: setattr(model, "logger", logger)
        model._setup_learn = setup
        model.set_task_environment = switch
        model.collect_rollouts = collect
        model._update_current_progress_remaining = lambda *args: None
        model.train = lambda: setattr(model, "train_count", model.train_count + 1)
        model.save = lambda path: Path(path).write_bytes(b"checkpoint")
        model.export_actor = lambda path: Path(path).write_bytes(b"actor")
        model.optimizer_summary = lambda: {"available": True, "state_tensor_elements": 3}
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
