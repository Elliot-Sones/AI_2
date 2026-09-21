import gzip
import importlib.util
import json
import sys
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

    def test_parse_requires_explicit_schedule_and_distinguishes_fresh_from_resume_sha(self):
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
        self.assertFalse(args.resume)
        alternate = list(base)
        alternate[alternate.index("--checkpoint-sha256") + 1] = "a" * 64
        with self.assertRaises(SystemExit):
            m.parse_args(alternate + ["--schedule", "R4E1N"])
        resumed = m.parse_args(alternate + [
            "--schedule", "R4E1N",
            "--resume",
            "--cycle-index", "2",
            "--resume-parent-result", str(root / "cycle-001-result.json"),
        ])
        self.assertTrue(resumed.resume)
        self.assertEqual(resumed.cycle_index, 2)
        self.assertEqual(resumed.checkpoint_sha256, "a" * 64)
        with self.assertRaises(SystemExit):
            m.parse_args(alternate + ["--schedule", "R4E1N", "--resume"])
        with self.assertRaises(SystemExit):
            m.parse_args(base + ["--schedule", "R4E1N", "--cycle-index", "2"])

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

    def test_resume_hash_and_preservation_are_verified_without_requiring_canonical_parent(self):
        m = self.subject()
        order = []
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), resume=True, checkpoint_sha="b" * 64)
            args.checkpoint.write_bytes(b"previous B2 cycle")
            with patch.object(m, "sha256", side_effect=lambda path: order.append("sha256") or "b" * 64), \
                 patch.object(m.base, "require_checkpoint_preservation", side_effect=lambda *a: order.append("ack") or {"verified": True}):
                checkpoint, preservation = m.verify_input_checkpoint(args)
            self.assertEqual(order, ["sha256", "sha256", "ack"])
            self.assertFalse(checkpoint["canonical_parent"])
            self.assertEqual(checkpoint["sha256"], "b" * 64)
            self.assertTrue(preservation["verified"])

            order.clear()
            with patch.object(m, "sha256", side_effect=lambda path: order.append("sha256") or "c" * 64), \
                 patch.object(m.base, "require_checkpoint_preservation", side_effect=lambda *a: order.append("ack") or {}), \
                 self.assertRaisesRegex(ValueError, "resume checkpoint SHA256 mismatch"):
                m.verify_input_checkpoint(args)
            self.assertEqual(order, ["sha256"])

    def test_resume_requires_exact_previous_cycle_result_lineage(self):
        m = self.subject()
        cfg = m.ReengagementCampaignConfig()
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), resume=True, checkpoint_sha="b" * 64)
            args.checkpoint.write_bytes(b"previous B2 cycle")
            with patch.object(m, "sha256", return_value="b" * 64), \
                 patch.object(m.base, "require_checkpoint_preservation", return_value={"verified": True}):
                checkpoint, _ = m.verify_input_checkpoint(args, cfg)
            self.assertEqual(checkpoint["resume_parent_result"]["cycle_index"], 1)
            self.assertEqual(
                checkpoint["expected_start_timesteps"],
                m.PARENT_LIFETIME_TRANSITIONS + cfg.total_transitions,
            )

            previous = json.loads(args.resume_parent_result.read_text())
            previous["checkpoint_sha256"] = "c" * 64
            args.resume_parent_result.write_text(json.dumps(previous))
            with patch.object(m, "sha256", return_value="b" * 64), \
                 patch.object(m.base, "require_checkpoint_preservation", return_value={"verified": True}), \
                 self.assertRaisesRegex(ValueError, "previous cycle checkpoint hash"):
                m.verify_input_checkpoint(args, cfg)

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

    def test_resume_loads_previous_b2_checkpoint_once_and_preserves_lineage(self):
        m = self.subject()
        cfg = m.ReengagementCampaignConfig()
        resume_sha = "b" * 64
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory), resume=True, checkpoint_sha=resume_sha)
            reengagement_env = self.fake_env(m, cfg, "reengagement")
            navigation_env = self.fake_env(m, cfg, "navigation")
            model = self.fake_model(
                m,
                cfg,
                start_timesteps=m.PARENT_LIFETIME_TRANSITIONS + cfg.total_transitions,
            )
            with self.common_patches(
                    m, cfg, reengagement_env, navigation_env, model, checkpoint_sha=resume_sha
                 ), \
                 patch.object(m.NativeCurriculumPPO, "load_contact_checkpoint") as load_parent, \
                 patch.object(m.NativeCurriculumPPO, "load", return_value=model) as load_resume:
                result = m.run_block(args)

            load_parent.assert_not_called()
            load_resume.assert_called_once()
            _, kwargs = load_resume.call_args
            self.assertEqual(kwargs["expected_sha256"], resume_sha)
            self.assertEqual(kwargs["expected_task_versions"], m.EXPECTED_TASK_VERSIONS)
            self.assertEqual(kwargs["update_mode"], cfg.update_mode)
            self.assertEqual(result["resume_from_checkpoint_sha256"], resume_sha)
            self.assertEqual(result["cycle_index"], 2)
            self.assertEqual(result["root_parent_checkpoint_sha256"], m.PARENT_CHECKPOINT_SHA256)
            self.assertGreater(result["final_model_counters"]["num_timesteps"], m.PARENT_LIFETIME_TRANSITIONS)
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertTrue(manifest["resume"])
            self.assertFalse(manifest["fresh_parent_per_block"])
            self.assertFalse(manifest["checkpoint"]["canonical_parent"])
            self.assertEqual(manifest["cycle_index"], 2)
            self.assertEqual(
                manifest["expected_start_timesteps"],
                m.PARENT_LIFETIME_TRANSITIONS + cfg.total_transitions,
            )

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

    def test_bad_rollout_diagnostics_stop_before_optimizer_update(self):
        m = self.subject()
        cfg = m.ReengagementCampaignConfig()
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            reengagement_env = self.fake_env(m, cfg, "reengagement")
            navigation_env = self.fake_env(m, cfg, "navigation")
            model = self.fake_model(
                m,
                cfg,
                start_timesteps=m.PARENT_LIFETIME_TRANSITIONS,
                bad_episode_version=True,
            )
            with self.common_patches(m, cfg, reengagement_env, navigation_env, model), \
                 patch.object(m.NativeCurriculumPPO, "load_contact_checkpoint", return_value=model), \
                 self.assertRaisesRegex(RuntimeError, "task/mask/version"):
                m.run_block(args)
            self.assertEqual(model.train_count, 0)
            self.assertEqual(model._n_updates, 0)

    def test_runtime_precision_must_remain_fp32_and_disallow_tf32(self):
        m = self.subject()
        cfg = m.ReengagementCampaignConfig()
        model = self.fake_model(m, cfg, start_timesteps=m.PARENT_LIFETIME_TRANSITIONS)
        self.assertEqual(m.validate_fp32_runtime(model, {"available": True}), {"available": True})

        model.policy.parameters = lambda: [self.fake_parameter([1, 2, 3], dtype="torch.float16")]
        with self.assertRaisesRegex(RuntimeError, "FP32"):
            m.validate_fp32_runtime(model, {"available": True})

        model = self.fake_model(m, cfg, start_timesteps=m.PARENT_LIFETIME_TRANSITIONS)
        model.rollout_buffer.rewards = SimpleNamespace(dtype="torch.float64")
        with self.assertRaisesRegex(RuntimeError, "FP32"):
            m.validate_fp32_runtime(model, {"available": True})

        model = self.fake_model(m, cfg, start_timesteps=m.PARENT_LIFETIME_TRANSITIONS)
        with self.assertRaisesRegex(RuntimeError, "TF32"):
            m.validate_fp32_runtime(model, {
                "available": True,
                "cuda_matmul_allow_tf32": True,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
            })

        model = self.fake_model(m, cfg, start_timesteps=m.PARENT_LIFETIME_TRANSITIONS)
        model.policy.optimizer.state[1]["exp_avg"] = self.fake_parameter(
            [1, 2, 3], dtype="torch.float16"
        )
        with self.assertRaisesRegex(RuntimeError, "optimizer"):
            m.validate_fp32_runtime(model, {"available": True})

    def test_fp32_runtime_is_configured_before_validation(self):
        m = self.subject()
        matmul = SimpleNamespace(allow_tf32=True)
        cudnn = SimpleNamespace(allow_tf32=True)
        calls = []
        fake_torch = SimpleNamespace(
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn),
            set_float32_matmul_precision=lambda value: calls.append(value),
        )
        expected = {
            "available": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        }
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
            m.throughput, "torch_precision_state", return_value=expected
        ):
            self.assertEqual(m.configure_fp32_runtime(), expected)
        self.assertFalse(matmul.allow_tf32)
        self.assertFalse(cudnn.allow_tf32)
        self.assertEqual(calls, ["highest"])

    def test_task_switch_requires_exact_model_versions_without_signature_fallback(self):
        m = self.subject()
        calls = []
        model = SimpleNamespace(
            task_versions=dict(m.EXPECTED_TASK_VERSIONS),
            set_task_environment=lambda task, env, *, calibration=False: calls.append(
                (task, env, calibration)
            ),
        )
        env = object()
        m.set_model_task_environment(model, "reengagement", env)
        self.assertEqual(calls, [("combat", env, False)])
        model.task_versions = {**m.EXPECTED_TASK_VERSIONS, "combat": "bridge_b1_contact-v1"}
        with self.assertRaisesRegex(ValueError, "task version"):
            m.set_model_task_environment(model, "reengagement", env)
        self.assertEqual(len(calls), 1)

    def test_resume_runtime_rejects_version_action_counter_or_optimizer_drift(self):
        m = self.subject()
        cfg = m.ReengagementCampaignConfig()
        expected_counter = m.PARENT_LIFETIME_TRANSITIONS + cfg.total_transitions

        model = self.fake_model(m, cfg, start_timesteps=expected_counter)
        model.task_versions = {**m.EXPECTED_TASK_VERSIONS, "combat": "bridge_b1_contact-v1"}
        with self.assertRaisesRegex(ValueError, "task version"):
            m.validate_reengagement_runtime(
                model, cfg, requested_rollout_task="reengagement", expected_start_timesteps=expected_counter
            )

        model = self.fake_model(m, cfg, start_timesteps=expected_counter)
        model.policy.action_head = "gaussian"
        with self.assertRaisesRegex(ValueError, "policy action head"):
            m.validate_reengagement_runtime(
                model, cfg, requested_rollout_task="reengagement", expected_start_timesteps=expected_counter
            )

        model = self.fake_model(m, cfg, start_timesteps=expected_counter - 1)
        with self.assertRaisesRegex(ValueError, "cycle lineage"):
            m.validate_reengagement_runtime(
                model, cfg, requested_rollout_task="reengagement", expected_start_timesteps=expected_counter
            )

        model = self.fake_model(m, cfg, start_timesteps=expected_counter)
        model.policy.optimizer.state = {}
        with self.assertRaisesRegex(ValueError, "optimizer"):
            m.validate_reengagement_runtime(
                model, cfg, requested_rollout_task="reengagement", expected_start_timesteps=expected_counter
            )

    def test_required_wandb_must_create_a_hashable_run_receipt_before_training(self):
        m = self.subject()
        cfg = m.ReengagementCampaignConfig()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root)
            args.require_wandb = True
            args.wandb_project = "utmist-ai2"
            checkpoint = {"sha256": m.PARENT_CHECKPOINT_SHA256}
            with patch.object(m.tracking, "init_run", return_value=None), \
                 self.assertRaisesRegex(RuntimeError, "W&B"):
                m.start_wandb_run(args, cfg, root / "out", checkpoint)

            receipt = root / "out" / "wandb_run.json"

            def initialize(*call_args, **call_kwargs):
                receipt.parent.mkdir(parents=True, exist_ok=True)
                receipt.write_text(json.dumps({
                    "id": "run-1",
                    "path": "entity/utmist-ai2/run-1",
                    "url": "https://wandb.ai/entity/utmist-ai2/runs/run-1",
                    "mode": "online",
                }))
                return SimpleNamespace(log=lambda *a, **k: None)

            with patch.object(m.tracking, "init_run", side_effect=initialize) as init_run:
                run, run_receipt = m.start_wandb_run(args, cfg, root / "out", checkpoint)
            self.assertIsNotNone(run)
            self.assertEqual(run_receipt["path"], str(receipt))
            self.assertRegex(run_receipt["sha256"], r"^[0-9a-f]{64}$")
            params = init_run.call_args.args[0]
            self.assertEqual(params["wandb"]["project"], "utmist-ai2")
            self.assertEqual(init_run.call_args.kwargs["phase_key"], "b2-cycle-001")
            self.assertEqual(init_run.call_args.kwargs["extra_config"]["checkpoint_sha256"], m.PARENT_CHECKPOINT_SHA256)

    def test_source_hash_inventory_includes_direct_trainer_imports_and_fails_missing(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(m.throughput, "source_hashes", return_value={}):
                hashes = m.source_hashes(root, root)
            for key in (
                "train_reengagement_campaign_block.py",
                "native_reengagement_lane_vec_env.py",
                "native_reengagement_episode.py",
                "native_navigation_curriculum_env.py",
                "native_curriculum_ppo.py",
                "benchmark_curriculum_throughput.py",
                "benchmark_native_contact_training.py",
                "train_contact_campaign_block.py",
            ):
                self.assertRegex(hashes[key], r"^[0-9a-f]{64}$")
            with patch.object(m.throughput, "source_hashes", return_value={}), \
                 patch.object(m, "DIRECT_SOURCE_FILES", ("missing-source.py",)), \
                 self.assertRaisesRegex(FileNotFoundError, "missing-source"):
                m.source_hashes(root, root)

    def test_b2_qualification_consumer_rejects_b1_or_stale_receipts(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proof = root / "qualification.json"
            (root / "python_oracle_native_cpu.log").write_text("cpu passed")
            (root / "python_oracle_native_cpu.xml").write_text("<testsuite><testcase name='a'/></testsuite>")
            (root / "native_cpu_cuda.log").write_text("cuda passed")
            (root / "native_cpu_cuda.xml").write_text("<testsuite><testcase name='b'/></testsuite>")
            (root / "b1_behavior_regression.log").write_text("b1 passed")
            (root / "b1_behavior_regression.xml").write_text("<testsuite><testcase name='c'/></testsuite>")
            source_hashes = {"native_reengagement_lane_vec_env.py": "a" * 64}
            payload = {
                "schema_version": m.QUALIFICATION_SCHEMA_VERSION,
                "qualification_version": m.QUALIFICATION_VERSION,
                "qualified": True,
                "overall_status": "passed",
                "checkpoint": {"sha256": m.PARENT_CHECKPOINT_SHA256},
                "provenance": {
                    "reference_root": str(root.resolve()),
                    "frozen_root": str(root.resolve()),
                },
                "version_contract": {
                    "curriculum_task": "combat",
                    "curriculum_version": m.REENGAGEMENT_VERSION,
                    "allowed_control_mask": [bool(value) for value in m.TASK_MASKS["combat"]],
                },
                "source_hashes": source_hashes,
                "root_hashes": {
                    "source_root": "b" * 64,
                    "reference_root": "c" * 64,
                    "frozen_root": "d" * 64,
                },
                "stage_status": {
                    "source_build_inventory": {
                        "status": "passed", "required": True,
                        "evidence": {"source_files": 1, "missing_sources": []},
                    },
                    "b1_behavior_regression": {
                        "status": "passed", "required": True,
                        "evidence": {
                            "cases": 1, "skips": [],
                            "log_sha256": m.sha256(root / "b1_behavior_regression.log"),
                            "xml_sha256": m.sha256(root / "b1_behavior_regression.xml"),
                        },
                    },
                    "python_oracle_native_cpu": {
                        "status": "passed", "required": True,
                        "evidence": {
                            "cases": 1, "skips": [],
                            "log_sha256": m.sha256(root / "python_oracle_native_cpu.log"),
                            "xml_sha256": m.sha256(root / "python_oracle_native_cpu.xml"),
                        },
                    },
                    "native_cpu_cuda": {
                        "status": "passed", "required": True,
                        "evidence": {
                            "cases": 1, "skips": [], "optional_skips": [],
                            "log_sha256": m.sha256(root / "native_cpu_cuda.log"),
                            "xml_sha256": m.sha256(root / "native_cpu_cuda.xml"),
                        },
                    },
                    "collector_learner_smoke": {
                        "status": "passed", "required": True,
                        "evidence": {
                            "optimizer_steps": 1,
                            "train_invocations": 1,
                            "attempted_epochs": 4,
                            "update_counter_delta": 4,
                            "checkpoint_sha256": m.PARENT_CHECKPOINT_SHA256,
                            "training_metrics": {
                                "train/value_loss": 0.25,
                                "train/entropy_loss": -0.5,
                            },
                            "parent_metadata": {
                                "action_ab_arm": "B_projected",
                                "action_head": "bernoulli_10",
                                "policy_action_head": "bernoulli_10",
                                "task_versions": dict(m.EXPECTED_TASK_VERSIONS),
                            },
                        },
                    },
                },
            }
            proof.write_text(json.dumps(payload))
            expected_inventory = {
                "source_sha256": source_hashes,
                "root_hashes": payload["root_hashes"],
            }
            with patch.object(m, "qualification_source_inventory", return_value=expected_inventory):
                accepted = m.require_b2_qualification(proof, root, root)
            self.assertTrue(accepted["qualified"])

            missing_b1 = json.loads(json.dumps(payload))
            missing_b1["stage_status"].pop("b1_behavior_regression")
            proof.write_text(json.dumps(missing_b1))
            with patch.object(m, "qualification_source_inventory", return_value=expected_inventory), \
                 self.assertRaisesRegex(ValueError, "b1_behavior_regression"):
                m.require_b2_qualification(proof, root, root)

            wrong_mask = json.loads(json.dumps(payload))
            wrong_mask["version_contract"]["allowed_control_mask"] = [
                int(value) for value in m.TASK_MASKS["combat"]
            ]
            proof.write_text(json.dumps(wrong_mask))
            with patch.object(m, "qualification_source_inventory", return_value=expected_inventory), \
                 self.assertRaisesRegex(ValueError, "qualification"):
                m.require_b2_qualification(proof, root, root)

            payload["schema_version"] = "ai2.native_contact_qualification.v1"
            proof.write_text(json.dumps(payload))
            with patch.object(m, "qualification_source_inventory", return_value=expected_inventory), \
                 self.assertRaisesRegex(ValueError, "B2 reengagement qualification"):
                m.require_b2_qualification(proof, root, root)

            payload["schema_version"] = m.QUALIFICATION_SCHEMA_VERSION
            payload["source_hashes"] = {"native_reengagement_lane_vec_env.py": "b" * 64}
            proof.write_text(json.dumps(payload))
            with patch.object(m, "qualification_source_inventory", return_value=expected_inventory), \
                 self.assertRaisesRegex(ValueError, "source"):
                m.require_b2_qualification(proof, root, root)

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

    def args(self, root, *, resume=False, checkpoint_sha=None, cycle_index=None):
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "checkpoint.zip"
        checkpoint.write_bytes(b"canonical checkpoint")
        ack = root / "preservation_ack.json"
        ack.write_text("{}")
        qualification = root / "proof.json"
        qualification.write_text("{}")
        m = self.subject()
        selected_cycle = cycle_index if cycle_index is not None else (2 if resume else 1)
        resume_parent_result = root / "previous-result.json"
        if resume:
            resume_parent_result.write_text(json.dumps({
                "schema_version": m.SCHEMA_VERSION,
                "cycle_index": selected_cycle - 1,
                "schedule_id": "R4E1N",
                "schedule": list(m.SCHEDULES["R4E1N"]),
                "expected_task_versions": dict(m.EXPECTED_TASK_VERSIONS),
                "checkpoint_sha256": checkpoint_sha,
                "root_parent_checkpoint_sha256": m.PARENT_CHECKPOINT_SHA256,
                "final_model_counters": {
                    "num_timesteps": m.PARENT_LIFETIME_TRANSITIONS
                    + (selected_cycle - 1) * m.ReengagementCampaignConfig().total_transitions,
                },
            }))
        return SimpleNamespace(
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha or m.PARENT_CHECKPOINT_SHA256,
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
            resume=resume,
            cycle_index=selected_cycle,
            resume_parent_result=resume_parent_result if resume else None,
            require_wandb=False,
            wandb_project="utmist-ai2",
        )

    def common_patches(self, m, cfg, reengagement_env, navigation_env, model, *, checkpoint_sha=None):
        wrappers = {"reengagement": self.fake_wrapper(reengagement_env), "navigation": self.fake_wrapper(navigation_env)}
        input_sha = checkpoint_sha or m.PARENT_CHECKPOINT_SHA256
        expected_start = m.PARENT_LIFETIME_TRANSITIONS + (
            cfg.total_transitions if input_sha != m.PARENT_CHECKPOINT_SHA256 else 0
        )
        patches = [
            patch.object(m, "ReengagementCampaignConfig", return_value=cfg),
            patch.object(m, "verify_input_checkpoint", return_value=(
                {
                    "path": "checkpoint.zip",
                    "sha256": input_sha,
                    "canonical_parent": input_sha == m.PARENT_CHECKPOINT_SHA256,
                    "expected_start_timesteps": expected_start,
                },
                {"verified": True},
            )),
            patch.object(m, "require_b2_qualification", return_value={"qualified": True}),
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
            patch.object(m.throughput, "torch_precision_state", return_value={"available": True, "float32_matmul_precision": "highest"}),
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

    def fake_model(self, m, cfg, *, start_timesteps, fail_on_collect=None, bad_episode_version=False):
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
            task_versions=dict(m.EXPECTED_TASK_VERSIONS),
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
            rollout_buffer=SimpleNamespace(
                actions=self.fake_tensor(dtype="torch.float32"),
                executed_actions=self.fake_tensor(dtype="torch.float32"),
                rewards=self.fake_tensor(dtype="torch.float32"),
                observations=self.fake_tensor(dtype="torch.float32"),
                values=self.fake_tensor(dtype="torch.float32"),
                log_probs=self.fake_tensor(dtype="torch.float32"),
                advantages=self.fake_tensor(dtype="torch.float32"),
                returns=self.fake_tensor(dtype="torch.float32"),
                episode_starts=self.fake_tensor(dtype="torch.float32"),
            ),
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

        def switch(model_task, env, calibration=False):
            raw_env = getattr(env, "env", env)
            self.assertEqual(model_task, raw_env.model_task)
            self.assertFalse(calibration)
            self.assertEqual(model.task_versions[model_task], raw_env.curriculum_version)
            model.task = model_task
            model.policy.task = model_task
            model.policy.allowed_control_mask = [bool(value) for value in m.TASK_MASKS[model_task]]
            model.switches.append((model_task, model.task_versions[model_task]))

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
            version = "bridge_b1_contact-v1" if bad_episode_version else raw_env.curriculum_version
            callback.episodes = [{
                payload_key: payload,
                "episode": {"r": 1.0, "l": 1},
                "curriculum_task": raw_env.model_task,
                "curriculum_version": version,
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

    def fake_tensor(self, *, dtype):
        return SimpleNamespace(dtype=dtype)

    def fake_parameter(self, values, *, dtype="torch.float32"):
        class Parameter:
            device = SimpleNamespace(type="cuda")

            def __init__(self, values, dtype):
                self.values = values
                self.dtype = dtype

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

        return Parameter(values, dtype)

    def clock(self):
        value = 0.0
        while True:
            value += 1.0
            yield value


if __name__ == "__main__":
    unittest.main()
