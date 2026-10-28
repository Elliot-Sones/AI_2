"""Pure local resume-contract tests for the native contact checkpoint adapter."""
from __future__ import annotations

from ai2.paths import source_path
import ai2.training

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch


def load_subject():
    """Import native_curriculum_ppo with minimal stubs; no Torch/SB3 required."""
    root = Path(__file__).resolve().parents[1]
    saved = {}
    fake_curriculum = ModuleType("curriculum_ppo")
    fake_curriculum.__file__ = str(source_path(root, "curriculum_ppo.py"))
    fake_curriculum.ACTION_EXPERIMENT_ARMS = {
        "G_keep": {"action_head": "gaussian", "action_init_strategy": "keep"},
        "B_projected": {"action_head": "bernoulli_10", "action_init_strategy": "projected"},
    }
    fake_curriculum.ACTION_EXPERIMENT_INTERPRETATION = {
        "G_keep": "keep gaussian",
        "B_projected": "project bernoulli",
    }
    fake_curriculum.ACTION_HEADS = ("gaussian", "bernoulli_10")
    fake_curriculum.PPO_DEFAULTS = dict(
        learning_rate=1e-4, n_steps=512, batch_size=8192, n_epochs=4,
        gamma=.99, gae_lambda=.95, clip_range=.1, target_kl=.01,
        vf_coef=.5, max_grad_norm=.5,
    )
    fake_curriculum.TASK_MASKS = {
        "navigation": (True, True, True, True, True, False, True, False, False, False),
        "combat": (True, True, True, True, True, False, True, True, False, False)
    }

    class CurriculumPolicy:
        action_head = "gaussian"

    class CurriculumPPO:
        @classmethod
        def load(cls, *args, **kwargs):
            raise AssertionError("patched in individual tests")
        def export_actor(self, *args, **kwargs):
            raise AssertionError("not used by resume contract tests")

    class ProtectedAdam:
        def __init__(self, policy):
            self._policy = policy
            self.param_groups = [{"lr": 3e-5}]
        def _policy_ref(self):
            return self._policy

    def resolve_action_arm(action_head=None, action_init_strategy=None, action_ab_arm=None):
        if action_ab_arm == "B_projected":
            return "bernoulli_10", "projected", "B_projected"
        return action_head or "gaussian", action_init_strategy or "keep", action_ab_arm or "G_keep"

    fake_curriculum.CurriculumPolicy = CurriculumPolicy
    fake_curriculum.CurriculumPPO = CurriculumPPO
    fake_curriculum._ProtectedAdam = ProtectedAdam
    fake_curriculum._resolve_action_arm = resolve_action_arm
    fake_curriculum._apply_action_initialization = lambda *args, **kwargs: {}
    fake_curriculum._copy_compatible_non_action_state = lambda *args, **kwargs: None
    fake_curriculum._set_neutral_action_head = lambda *args, **kwargs: None

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_rng_state_all=lambda: [],
        set_rng_state_all=lambda state: None,
    )
    fake_torch.get_rng_state = lambda: object()
    fake_torch.set_rng_state = lambda state: None
    fake_torch.device = lambda value: value

    fake_optimizations = ModuleType("ppo_update_optimizations")
    fake_optimizations.deferred_metrics_train = lambda model: None
    fake_optimizations.sb3_train_source_metadata = lambda: {"supported": True}

    fake_tensor_ppo = ModuleType("tensor_ppo")
    fake_tensor_ppo.TensorPPO = object
    fake_tensor_ppo.TensorVecEnv = type("TensorVecEnv", (), {})

    fake_tensor_rollout = ModuleType("tensor_rollout_buffer")
    fake_tensor_rollout.TensorRolloutBuffer = object

    replacements = {
        "ai2.training.curriculum_ppo": fake_curriculum,
        "torch": fake_torch,
        "ai2.training.ppo_update_optimizations": fake_optimizations,
        "ai2.training.tensor_ppo": fake_tensor_ppo,
        "ai2.training.tensor_rollout_buffer": fake_tensor_rollout,
    }
    for name, module in replacements.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = module
    name = "native_curriculum_ppo_campaign_resume_subject"
    saved[name] = sys.modules.get(name)
    try:
        spec = importlib.util.spec_from_file_location(name, source_path(root, "native_curriculum_ppo.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        with patch.object(ai2.training, "curriculum_ppo", fake_curriculum, create=True):
            spec.loader.exec_module(module)
        module._test_stubs = SimpleNamespace(
            CurriculumPPO=CurriculumPPO,
            CurriculumPolicy=CurriculumPolicy,
            ProtectedAdam=ProtectedAdam,
        )
        return module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class CampaignResumeContractTests(unittest.TestCase):
    def tensor_env(self, module, task):
        return SimpleNamespace(
            device="cuda",
            num_envs=16,
            env=SimpleNamespace(curriculum_task=task),
        )

    def source_checkpoint(self, module, *, learning_rate=3e-5, task="combat"):
        stubs = module._test_stubs
        source = stubs.CurriculumPPO()
        policy = stubs.CurriculumPolicy()
        policy.action_head = "gaussian"
        policy.optimizer = stubs.ProtectedAdam(policy)
        source.policy = policy
        source.arm = "E1"
        source.task = task
        source.batch_size = 8192
        source.learning_rate = learning_rate
        source.lr_schedule = lambda progress: learning_rate
        settings_learning_rate = learning_rate if learning_rate in (1e-4, 3e-5) else 1e-4
        for name, value in module.contact_settings(batch_size=8192,
                                                   learning_rate=settings_learning_rate).items():
            if name not in ("learning_rate", "batch_size"):
                setattr(source, name, value)
        source.policy_kwargs = {"net_arch": [8]}
        source.calibration = False
        source.verbose = 0
        source._stats_window_size = 100
        source.tensorboard_log = None
        source.num_timesteps = 123
        source._n_updates = 4
        source._episode_num = 5
        source._total_timesteps = 6
        source._num_timesteps_at_start = 7
        source._current_progress_remaining = .25
        source.parent_num_timesteps = 99
        source.task_history = [{"task": "combat"}]
        source.seed = 17
        source.action_head = "gaussian"
        source.action_init_strategy = "keep"
        source.action_ab_arm = "G_keep"
        source.action_init_metadata = {"schema": "ai2-action-ab-init-v1", "source": "saved"}
        source.get_parameters = lambda: {"policy": "weights", "policy.optimizer": "adam"}
        return source

    def test_task_versions_and_environment_validation_cover_navigation_and_combat(self):
        module = load_subject()
        self.assertEqual(module.TASK_VERSIONS, {
            "navigation": "bridge_0b_navigation-v1",
            "combat": module.CONTACT_VERSION,
        })
        for task in ("navigation", "combat"):
            with self.subTest(task=task):
                env = SimpleNamespace(
                    curriculum_task=task,
                    curriculum_version=module.TASK_VERSIONS[task],
                    allowed_control_mask=module.TASK_MASKS[task],
                    host_has_timeouts=lambda task=task: task == "navigation",
                )
                module.validate_task_environment(env, task)

        bad = SimpleNamespace(
            curriculum_task="navigation",
            curriculum_version=module.TASK_VERSIONS["navigation"],
            allowed_control_mask=module.TASK_MASKS["combat"],
            host_has_timeouts=lambda: True,
        )
        with self.assertRaisesRegex(ValueError, "mask"):
            module.validate_task_environment(bad, "navigation")

    def test_contact_settings_allows_only_canonical_and_staged_resume_learning_rates(self):
        module = load_subject()
        self.assertEqual(module.contact_settings()["learning_rate"], 1e-4)
        self.assertEqual(module.contact_settings(learning_rate=3e-5)["learning_rate"], 3e-5)
        for learning_rate in (True, 0, 1e-5, 3e-4, "3e-5"):
            with self.subTest(learning_rate=learning_rate), self.assertRaises(ValueError):
                module.contact_settings(learning_rate=learning_rate)

    def test_from_curriculum_derives_resume_learning_rate_and_keeps_optimizer_metadata_honest(self):
        module = load_subject()
        source = self.source_checkpoint(module, learning_rate=3e-5)
        env = self.tensor_env(module, "combat")
        captured = {}

        def fake_init(model, **kwargs):
            captured.update(kwargs)
            model.policy = SimpleNamespace(
                optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}, {"lr": 1e-4}]),
                action_head=kwargs["action_head"],
            )
            def set_parameters(params, **kw):
                captured.setdefault("set_parameters", (params, kw))
                for group in model.policy.optimizer.param_groups:
                    group["lr"] = params["policy.optimizer"]["param_groups"][0]["lr"]
            model.set_parameters = set_parameters

        parameters = {"policy": "weights", "policy.optimizer": {"param_groups": [{"lr": 3e-5}]}}
        source.get_parameters = lambda: copy.deepcopy(parameters)

        with patch.object(module, "_require_cuda"), patch.object(module, "_require_sources"), \
             patch.object(module.NativeCurriculumPPO, "__init__", fake_init):
            model = module.NativeCurriculumPPO.from_curriculum(source, env)

        self.assertEqual(captured["learning_rate"], 3e-5)
        self.assertEqual(model.learning_rate, 3e-5)
        self.assertEqual(model.lr_schedule(.5), 3e-5)
        self.assertEqual([group["lr"] for group in model.policy.optimizer.param_groups], [3e-5, 3e-5])
        self.assertEqual(model.action_init_metadata["source"], "saved")
        self.assertNotIn("optimizer state not transferred", model.optimizer_provenance)
        self.assertIn("optimizer state copied", model.optimizer_provenance)
        self.assertEqual(captured["set_parameters"][0], source.get_parameters())

    def test_from_curriculum_uses_env_task_as_the_requested_next_task(self):
        module = load_subject()
        source = self.source_checkpoint(module, learning_rate=3e-5, task="navigation")
        env = self.tensor_env(module, "combat")
        captured = {}

        def fake_init(model, **kwargs):
            captured.update(kwargs)
            model.task = kwargs["task"]
            model.policy = SimpleNamespace(
                optimizer=SimpleNamespace(param_groups=[{"lr": kwargs["learning_rate"]}]),
                action_head=kwargs["action_head"],
            )
            model.set_parameters = lambda params, **kw: captured.setdefault("set_parameters", (params, kw))

        with patch.object(module, "_require_cuda"), patch.object(module, "_require_sources"), \
             patch.object(module.NativeCurriculumPPO, "__init__", fake_init):
            model = module.NativeCurriculumPPO.from_curriculum(source, env)

        self.assertEqual(captured["task"], "combat")
        self.assertEqual(model.task, "combat")
        self.assertEqual(model.task_history, [{"task": "combat"}])
        self.assertEqual(captured["set_parameters"][0], source.get_parameters())

    def test_from_curriculum_can_resume_combat_checkpoint_into_navigation_next(self):
        module = load_subject()
        source = self.source_checkpoint(module, learning_rate=3e-5, task="combat")
        env = self.tensor_env(module, "navigation")
        captured = {}

        def fake_init(model, **kwargs):
            captured.update(kwargs)
            model.task = kwargs["task"]
            model.policy = SimpleNamespace(
                optimizer=SimpleNamespace(param_groups=[{"lr": kwargs["learning_rate"]}]),
                action_head=kwargs["action_head"],
            )
            model.set_parameters = lambda params, **kw: captured.setdefault("set_parameters", (params, kw))

        with patch.object(module, "_require_cuda"), patch.object(module, "_require_sources"), \
             patch.object(module.NativeCurriculumPPO, "__init__", fake_init):
            model = module.NativeCurriculumPPO.from_curriculum(source, env)

        self.assertEqual(captured["task"], "navigation")
        self.assertFalse(captured["calibration"])
        self.assertEqual(model.task, "navigation")
        self.assertEqual(captured["set_parameters"][0], source.get_parameters())

    def test_from_curriculum_accepts_explicit_next_task_when_env_task_is_not_introspectable(self):
        module = load_subject()
        source = self.source_checkpoint(module, learning_rate=3e-5, task="navigation")
        env = SimpleNamespace(device="cuda", num_envs=16)
        captured = {}

        def fake_init(model, **kwargs):
            captured.update(kwargs)
            model.task = kwargs["task"]
            model.policy = SimpleNamespace(
                optimizer=SimpleNamespace(param_groups=[{"lr": kwargs["learning_rate"]}]),
                action_head=kwargs["action_head"],
            )
            model.set_parameters = lambda params, **kw: captured.setdefault("set_parameters", (params, kw))

        with patch.object(module, "_require_cuda"), patch.object(module, "_require_sources"), \
             patch.object(module.NativeCurriculumPPO, "__init__", fake_init):
            model = module.NativeCurriculumPPO.from_curriculum(source, env, task="combat")

        self.assertEqual(captured["task"], "combat")
        self.assertEqual(model.task, "combat")

    def test_set_task_environment_rejects_collection_and_unconsumed_rollouts(self):
        module = load_subject()
        model = object.__new__(module.NativeCurriculumPPO)
        model.rollout_buffer = SimpleNamespace(pos=0, full=False, consumed=True)
        model._task_live_state = {}
        model._collecting = True
        with self.assertRaisesRegex(ValueError, "during collection"):
            model.set_task_environment("navigation", SimpleNamespace())

        model._collecting = False
        for pos, full in ((1, False), (0, True)):
            with self.subTest(pos=pos, full=full):
                model.rollout_buffer = SimpleNamespace(pos=pos, full=full, consumed=False)
                with self.assertRaisesRegex(ValueError, "consumed"):
                    model.set_task_environment("navigation", SimpleNamespace())

    def test_set_task_environment_swaps_shim_and_keeps_live_state_per_task(self):
        module = load_subject()

        class FakeTensor:
            def __init__(self, label):
                self.label = label
            def detach(self):
                return self
            def clone(self):
                return FakeTensor(self.label + ".clone")

        class FakeEnv:
            num_envs = 16
            def __init__(self, label):
                self.label = label
                self.reset_count = 0
            def as_sb3_vec_env(self):
                return f"{self.label}-shim"
            def reset_tensor(self):
                self.reset_count += 1
                return FakeTensor(f"{self.label}.reset")

        model = object.__new__(module.NativeCurriculumPPO)
        model.task = "combat"
        model.calibration = False
        model.num_timesteps = 123
        model.rollout_buffer = SimpleNamespace(pos=0, full=True, consumed=True)
        model._task_live_state = {}
        model._collecting = False
        model._last_obs = FakeTensor("combat.obs")
        model._last_episode_starts = FakeTensor("combat.starts")
        model.task_history = []
        model.device = "cuda"
        calls = []
        model.policy = SimpleNamespace(set_task=lambda task, calibration=False: calls.append((task, calibration)))
        nav_env, combat_env = FakeEnv("navigation"), FakeEnv("combat")
        model.tensor_vec_env = combat_env

        with patch.object(module, "_require_cuda"), \
             patch.object(module.NativeCurriculumPPO, "_same_tensor_abi"), \
             patch.object(module.th, "ones", return_value=FakeTensor("navigation.starts"), create=True), \
             patch.object(module.th, "float32", "float32", create=True):
            model.set_task_environment("navigation", nav_env)

            self.assertEqual(model.tensor_vec_env, nav_env)
            self.assertEqual(model.env, "navigation-shim")
            self.assertEqual(nav_env.reset_count, 1)
            self.assertEqual(model._last_obs.label, "navigation.reset.clone")
            self.assertEqual(model._last_episode_starts.label, "navigation.starts")
            self.assertTrue(any(
                state[0] is combat_env and state[1].label == "combat.obs.clone"
                for state in model._task_live_state.values()
            ))
            self.assertEqual(calls[-1], ("navigation", False))

            model._last_obs = FakeTensor("navigation.obs")
            model._last_episode_starts = FakeTensor("navigation.starts.live")
            model.set_task_environment("combat", combat_env, calibration=True)

        self.assertEqual(model.tensor_vec_env, combat_env)
        self.assertEqual(model.env, "combat-shim")
        self.assertEqual(combat_env.reset_count, 0)
        self.assertEqual(model._last_obs.label, "combat.obs.clone.clone")
        self.assertEqual(model._last_episode_starts.label, "combat.starts.clone.clone")
        self.assertTrue(any(
            state[0] is nav_env and state[1].label == "navigation.obs.clone"
            for state in model._task_live_state.values()
        ))
        self.assertEqual(calls[-1], ("combat", True))

    def test_replacing_same_task_environment_resets_instead_of_reusing_stale_live_state(self):
        module = load_subject()

        class FakeTensor:
            def __init__(self, label):
                self.label = label
            def detach(self):
                return self
            def clone(self):
                return FakeTensor(self.label + ".clone")

        class FakeEnv:
            num_envs = 16
            def __init__(self, label):
                self.label = label
                self.reset_count = 0
            def as_sb3_vec_env(self):
                return f"{self.label}-shim"
            def reset_tensor(self):
                self.reset_count += 1
                return FakeTensor(f"{self.label}.reset")

        current_env = FakeEnv("combat.current")
        replacement_env = FakeEnv("combat.replacement")
        model = object.__new__(module.NativeCurriculumPPO)
        model.task = "combat"
        model.calibration = False
        model.num_timesteps = 123
        model.rollout_buffer = SimpleNamespace(pos=0, full=True, consumed=True)
        model._task_live_state = {}
        model._collecting = False
        model._last_obs = FakeTensor("combat.live")
        model._last_episode_starts = FakeTensor("combat.starts")
        model.task_history = []
        model.device = "cuda"
        model.tensor_vec_env = current_env
        model.policy = SimpleNamespace(set_task=lambda *args, **kwargs: None)

        with patch.object(module, "_require_cuda"), \
             patch.object(module.NativeCurriculumPPO, "_same_tensor_abi"), \
             patch.object(module.th, "ones", return_value=FakeTensor("starts.reset"), create=True), \
             patch.object(module.th, "float32", "float32", create=True):
            model.set_task_environment("combat", replacement_env)

        self.assertEqual(replacement_env.reset_count, 1)
        self.assertEqual(model._last_obs.label, "combat.replacement.reset.clone")
        self.assertEqual(model._last_episode_starts.label, "starts.reset")
        self.assertTrue(any(state[0] is current_env for state in model._task_live_state.values()))

    def test_from_curriculum_rejects_unsupported_resume_learning_rate(self):
        module = load_subject()
        source = self.source_checkpoint(module, learning_rate=1e-5)
        env = self.tensor_env(module, "combat")
        with patch.object(module, "_require_cuda"), patch.object(module, "_require_sources"), \
             patch.object(module.NativeCurriculumPPO, "__init__", side_effect=AssertionError("must not construct")):
            with self.assertRaisesRegex(ValueError, "learning_rate"):
                module.NativeCurriculumPPO.from_curriculum(source, env)

    def test_from_curriculum_rejects_resume_learning_rate_schedule_that_is_not_constant(self):
        module = load_subject()
        source = self.source_checkpoint(module, learning_rate=3e-5)
        source.lr_schedule = lambda progress: 3e-5 if progress in (0.0, 1.0) else 1e-4
        env = self.tensor_env(module, "combat")
        with patch.object(module, "_require_cuda"), patch.object(module, "_require_sources"), \
             patch.object(module.NativeCurriculumPPO, "__init__", side_effect=AssertionError("must not construct")):
            with self.assertRaisesRegex(ValueError, "lr_schedule"):
                module.NativeCurriculumPPO.from_curriculum(source, env)

    def test_from_curriculum_rejects_optimizer_lr_that_was_not_preserved_exactly(self):
        module = load_subject()
        source = self.source_checkpoint(module, learning_rate=3e-5)
        source.get_parameters = lambda: {"policy.optimizer": {"param_groups": [{"lr": 1e-4}]}}
        env = self.tensor_env(module, "combat")

        def fake_init(model, **kwargs):
            model.policy = SimpleNamespace(
                optimizer=SimpleNamespace(param_groups=[{"lr": kwargs["learning_rate"]}]),
                action_head=kwargs["action_head"],
            )
            def set_parameters(params, **kw):
                for group in model.policy.optimizer.param_groups:
                    group["lr"] = params["policy.optimizer"]["param_groups"][0]["lr"]
            model.set_parameters = set_parameters

        with patch.object(module, "_require_cuda"), patch.object(module, "_require_sources"), \
             patch.object(module.NativeCurriculumPPO, "__init__", fake_init):
            with self.assertRaisesRegex(ValueError, "optimizer learning_rate"):
                module.NativeCurriculumPPO.from_curriculum(source, env)

    def test_load_rejects_learning_rate_override_instead_of_generalizing_resume(self):
        module = load_subject()
        env = SimpleNamespace(device="cuda")
        with patch.object(module, "_require_cuda"), patch.object(module, "_require_sources"), \
             patch.object(module, "verify_checkpoint"):
            with self.assertRaisesRegex(ValueError, "hyperparameter overrides"):
                module.NativeCurriculumPPO.load("checkpoint.zip", env=env, expected_sha256="0"*64,
                                                learning_rate=3e-5)


class CampaignResumeCudaTests(unittest.TestCase):
    def setUp(self):
        checkpoint = os.environ.get("AI2_CONTACT_CHECKPOINT")
        if not checkpoint:
            self.skipTest("set AI2_CONTACT_CHECKPOINT to the canonical calibrated checkpoint")
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local contract environment")
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable; no CPU policy construction, inference or training")
        self.checkpoint = checkpoint
        self.torch = torch
        self.module = __import__("native_curriculum_ppo")
        self.assertTrue(self.module.curriculum_source_metadata()["supported"])
        from ai2.training.ppo_update_optimizations import sb3_train_source_metadata
        self.assertTrue(sb3_train_source_metadata()["supported"])
        self.addCleanup(self.restore_rng, self.rng_state())

    def rng_state(self):
        return (self.torch.get_rng_state().clone(),
                [value.clone() for value in self.torch.cuda.get_rng_state_all()])

    def restore_rng(self, state):
        self.torch.set_rng_state(state[0])
        self.torch.cuda.set_rng_state_all(state[1])

    def environment(self, seed):
        from ai2.simulation.native_contact_lane_vec_env import build_contact_env
        from ai2.training.tensor_ppo import TensorVecEnv
        raw = build_contact_env(64, device="cuda", seed=seed, episodes_per_lane=2048,
                                record_capacity=64*512)
        self.addCleanup(raw.close)
        return TensorVecEnv(raw, device="cuda")

    def attach_logger(self, model):
        from stable_baselines3.common.logger import configure
        logger = configure(folder=None, format_strings=[])
        self.addCleanup(logger.close)
        model.set_logger(logger)
        return model

    def collect_and_train_once(self, model):
        from stable_baselines3.common.callbacks import BaseCallback
        class Callback(BaseCallback):
            def _on_step(self):
                return True
        _, callback = model._setup_learn(64*512, callback=Callback(), reset_num_timesteps=False)
        try:
            self.assertTrue(model.collect_rollouts(model.env, callback, model.rollout_buffer, 512))
        finally:
            callback.locals.clear()
        model.train()
        self.assertFalse(model.last_update_diagnostics["kl_early_stop"])
        self.assertGreater(model.last_update_diagnostics["optimizer_steps"], 0)
        for parameter in model.policy.parameters():
            self.assertTrue(bool(self.torch.isfinite(parameter).all()))

    def set_lower_learning_rate(self, model):
        model.learning_rate = 3e-5
        model.lr_schedule = lambda _progress_remaining: 3e-5
        for group in model.policy.optimizer.param_groups:
            group["lr"] = 3e-5

    def assert_tree_exact(self, left, right):
        torch = self.torch
        if torch.is_tensor(left) or torch.is_tensor(right):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_exact(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for actual, expected in zip(left, right):
                self.assert_tree_exact(actual, expected)
        else:
            self.assertEqual(left, right)

    def assert_optimizer_exact(self, left, right):
        left_state, right_state = left.policy.optimizer.state_dict(), right.policy.optimizer.state_dict()
        self.assert_tree_exact(left_state["param_groups"], right_state["param_groups"])
        self.assertEqual(left_state["state"].keys(), right_state["state"].keys())
        for key, values in left_state["state"].items():
            for name, value in values.items():
                other = right_state["state"][key][name]
                if self.torch.is_tensor(value):
                    value = value.detach().cpu()
                    other = other.detach().cpu()
                self.assert_tree_exact(value, other)

    def assert_resume_exact(self, source, restored, *, arm):
        torch, module = self.torch, self.module
        self.assert_tree_exact(source.policy.state_dict(), restored.policy.state_dict())
        self.assert_optimizer_exact(source, restored)
        for name in ("num_timesteps", "_n_updates", "_episode_num", "_total_timesteps",
                     "_num_timesteps_at_start", "_current_progress_remaining",
                     "parent_num_timesteps", "task_history"):
            self.assert_tree_exact(getattr(source, name), getattr(restored, name))
        self.assertEqual(restored.learning_rate, 3e-5)
        for progress in (0.0, 0.5, 1.0):
            self.assertEqual(restored.lr_schedule(progress), 3e-5)
        self.assertEqual([group["lr"] for group in restored.policy.optimizer.param_groups], [3e-5])
        self.assertEqual(restored.policy.allowed_control_mask.detach().cpu().tolist(),
                         list(module.COMBAT_MASK))
        self.assertEqual(restored.action_ab_arm, arm)
        self.assertEqual(restored.action_head, source.action_head)
        self.assertEqual(restored.policy.action_head, source.policy.action_head)
        self.assertIs(restored.policy.optimizer._policy_ref(), restored.policy)
        self.assertFalse(restored.calibration)
        self.assertTrue(all(parameter.device.type == "cuda" and bool(torch.isfinite(parameter).all())
                            for parameter in restored.policy.parameters()))

    def test_lower_lr_resume_roundtrip_preserves_state_and_next_cuda_update_for_both_arms(self):
        module = self.module
        module.verify_checkpoint(self.checkpoint, module.CALIBRATED_CHECKPOINT_SHA256)
        with tempfile.TemporaryDirectory(prefix="campaign-resume-cuda-") as temporary:
            for index, arm in enumerate(("G_keep", "B_projected")):
                with self.subTest(arm=arm):
                    model = self.attach_logger(module.NativeCurriculumPPO.load_contact_checkpoint(
                        self.checkpoint, self.environment(910 + index), action_ab_arm=arm,
                        update_mode="deferred_metrics",
                    ))
                    self.set_lower_learning_rate(model)
                    self.collect_and_train_once(model)
                    path = Path(temporary) / f"{arm}.zip"
                    model.save(path)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()

                    restored = self.attach_logger(module.NativeCurriculumPPO.load(
                        path, env=self.environment(950 + index), expected_sha256=digest,
                    ))
                    self.assert_resume_exact(model, restored, arm=arm)
                    self.collect_and_train_once(restored)

                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        module.NativeCurriculumPPO.load(path, env=self.environment(990 + index),
                                                        expected_sha256="0"*64)
                    with self.assertRaisesRegex(ValueError, "hyperparameter overrides"):
                        module.NativeCurriculumPPO.load(path, env=self.environment(1030 + index),
                                                        expected_sha256=digest, learning_rate=3e-5)


if __name__ == "__main__":
    unittest.main()
