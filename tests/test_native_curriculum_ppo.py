"""Native curriculum contracts; policy execution is CUDA-only, including tests."""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np


class NativeTaskTensorRolloutBufferTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('native_curriculum_ppo'),
                             'the CUDA curriculum adapter is not implemented')
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch is not installed in this local contract environment')
        return importlib.import_module('native_curriculum_ppo')

    def make_buffer(self, *, action_head='gaussian', buffer_size=2, n_envs=3):
        module = self.module()
        torch = module.th
        buffer = object.__new__(module.TaskTensorRolloutBuffer)
        buffer.torch = torch
        buffer.buffer_size = buffer_size
        buffer.n_envs = n_envs
        buffer.device = torch.device('cpu')
        buffer.obs_shape = (4,)
        buffer.action_dim = 10
        buffer.pos = 0
        buffer.full = False
        buffer.generator_ready = False
        buffer._computed_returns_and_advantage = False
        buffer.validation_mode = 'step'
        buffer._flat_tensors = {}
        buffer.observations = torch.empty((buffer_size, n_envs, 4), dtype=torch.float32)
        buffer.actions = torch.empty((buffer_size, n_envs, 10), dtype=torch.float32)
        buffer.rewards = torch.empty((buffer_size, n_envs), dtype=torch.float32)
        buffer.episode_starts = torch.empty((buffer_size, n_envs), dtype=torch.float32)
        buffer.values = torch.empty((buffer_size, n_envs), dtype=torch.float32)
        buffer.log_probs = torch.empty((buffer_size, n_envs), dtype=torch.float32)
        buffer.advantages = torch.zeros((buffer_size, n_envs), dtype=torch.float32)
        buffer.returns = torch.zeros((buffer_size, n_envs), dtype=torch.float32)
        buffer._invalid_data = torch.zeros((), dtype=torch.bool)
        buffer.executed_actions = torch.empty_like(buffer.actions)
        buffer.task_ids = torch.empty((buffer_size, n_envs), dtype=torch.int8)
        buffer.allowed_control_masks = torch.empty(buffer.actions.shape, dtype=torch.bool)
        buffer.task = None
        buffer.consumed = False
        buffer.action_head = action_head
        buffer._task_masks = {
            task: torch.tensor(module.TASK_MASKS[task], dtype=torch.bool)
            for task in module.TASK_IDS
        }

        def require_float32(_name, tensor, shape):
            self.assertEqual(tuple(tensor.shape), shape)
            self.assertEqual(tensor.dtype, torch.float32)
            return tensor.contiguous()

        def require_vector(_name, tensor):
            self.assertIn(tuple(tensor.shape), ((n_envs,), (n_envs, 1)))
            return tensor.reshape(n_envs).to(dtype=torch.float32).contiguous()

        buffer._require_cuda_float32 = require_float32
        buffer._require_vector_or_column = require_vector
        return module, torch, buffer

    def fill_rollout(self, buffer, torch, module, *, task, actions, executed_actions):
        mask = np.asarray(module.TASK_MASKS[task], dtype=bool)
        for step in range(buffer.buffer_size):
            buffer.add(
                torch.full((buffer.n_envs, *buffer.obs_shape), float(step), dtype=torch.float32),
                actions.clone(),
                torch.zeros(buffer.n_envs, dtype=torch.float32),
                torch.zeros(buffer.n_envs, dtype=torch.float32),
                torch.zeros(buffer.n_envs, dtype=torch.float32),
                torch.zeros(buffer.n_envs, dtype=torch.float32),
                task=task,
                mask=mask,
                executed_actions=executed_actions.clone(),
            )
        buffer._computed_returns_and_advantage = True

    def test_task_ids_and_masks_are_canonical(self):
        module = self.module()
        self.assertEqual(module.TASK_IDS, {'navigation': 0, 'combat': 1})
        for task, task_id in module.TASK_IDS.items():
            with self.subTest(task=task):
                mask = module._canonical_task_mask(task)
                self.assertEqual(mask.dtype, np.dtype(bool))
                np.testing.assert_array_equal(mask, np.asarray(module.TASK_MASKS[task], dtype=bool))
                self.assertEqual(task_id, int(module.TASK_IDS[task]))
        with self.assertRaisesRegex(ValueError, 'Unsupported native curriculum task'):
            module._canonical_task_mask('full_combat')

    def test_full_navigation_and_contact_rollouts_validate(self):
        for task in ('navigation', 'combat'):
            with self.subTest(task=task):
                module, torch, buffer = self.make_buffer()
                actions = torch.tensor([[1.5, -.25, .25, .75, 1.0, .6, .8, .9, .7, .4]],
                                       dtype=torch.float32).repeat(buffer.n_envs, 1)
                executed = actions.clamp(0, 1) * torch.tensor(module.TASK_MASKS[task], dtype=torch.float32)
                self.fill_rollout(buffer, torch, module, task=task, actions=actions,
                                  executed_actions=executed)
                buffer.validate(task, np.asarray(module.TASK_MASKS[task], dtype=bool))
                self.assertEqual(buffer.task, task)
                self.assertTrue(bool((buffer.task_ids == module.TASK_IDS[task]).all()))
                self.assertTrue(bool((buffer.allowed_control_masks ==
                                      torch.tensor(module.TASK_MASKS[task], dtype=torch.bool)).all()))

    def test_changing_task_or_mask_after_first_transition_raises(self):
        module, torch, buffer = self.make_buffer()
        actions = torch.ones((buffer.n_envs, 10), dtype=torch.float32)
        combat_mask = np.asarray(module.TASK_MASKS['combat'], dtype=bool)
        combat_executed = actions * torch.tensor(module.TASK_MASKS['combat'], dtype=torch.float32)
        buffer.add(torch.zeros((buffer.n_envs, *buffer.obs_shape), dtype=torch.float32), actions,
                   torch.zeros(buffer.n_envs), torch.zeros(buffer.n_envs), torch.zeros(buffer.n_envs),
                   torch.zeros(buffer.n_envs), task='combat', mask=combat_mask,
                   executed_actions=combat_executed)
        with self.assertRaisesRegex(ValueError, 'Cannot mix tasks'):
            buffer.add(torch.zeros((buffer.n_envs, *buffer.obs_shape), dtype=torch.float32), actions,
                       torch.zeros(buffer.n_envs), torch.zeros(buffer.n_envs), torch.zeros(buffer.n_envs),
                       torch.zeros(buffer.n_envs), task='navigation',
                       mask=np.asarray(module.TASK_MASKS['navigation'], dtype=bool),
                       executed_actions=actions * torch.tensor(module.TASK_MASKS['navigation'], dtype=torch.float32))
        wrong_mask = combat_mask.copy()
        wrong_mask[7] = False
        with self.assertRaisesRegex(ValueError, 'Wrong allowed-control mask'):
            buffer.add(torch.zeros((buffer.n_envs, *buffer.obs_shape), dtype=torch.float32), actions,
                       torch.zeros(buffer.n_envs), torch.zeros(buffer.n_envs), torch.zeros(buffer.n_envs),
                       torch.zeros(buffer.n_envs), task='combat', mask=wrong_mask,
                       executed_actions=combat_executed)

    def test_numeric_masks_are_not_coerced_to_bool(self):
        module, torch, buffer = self.make_buffer()
        actions = torch.ones((buffer.n_envs, 10), dtype=torch.float32)
        numeric_mask = np.asarray(module.TASK_MASKS['combat'], dtype=np.int8)
        executed = actions * torch.tensor(module.TASK_MASKS['combat'], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, 'Wrong allowed-control mask'):
            buffer.add(torch.zeros((buffer.n_envs, *buffer.obs_shape), dtype=torch.float32), actions,
                       torch.zeros(buffer.n_envs), torch.zeros(buffer.n_envs), torch.zeros(buffer.n_envs),
                       torch.zeros(buffer.n_envs), task='combat', mask=numeric_mask,
                       executed_actions=executed)

        self.fill_rollout(buffer, torch, module, task='combat', actions=actions,
                          executed_actions=executed)
        with self.assertRaisesRegex(ValueError, 'Rollout mask'):
            buffer.validate('combat', numeric_mask)
        with self.assertRaisesRegex(ValueError, 'Rollout mask'):
            buffer.validate('combat', torch.tensor(module.TASK_MASKS['combat'], dtype=torch.int8))

    def test_task_specific_gaussian_executed_action_decode(self):
        module, torch, nav_buffer = self.make_buffer(action_head='gaussian')
        actions = torch.tensor([[1.25, -.5, .25, .75, 1.0, .6, .9, .95, .8, .4]],
                               dtype=torch.float32).repeat(nav_buffer.n_envs, 1)
        nav_executed = actions.clamp(0, 1) * torch.tensor(module.TASK_MASKS['navigation'], dtype=torch.float32)
        self.fill_rollout(nav_buffer, torch, module, task='navigation', actions=actions,
                          executed_actions=nav_executed)
        nav_buffer.validate('navigation', np.asarray(module.TASK_MASKS['navigation'], dtype=bool))
        self.assertTrue(bool((nav_buffer.executed_actions[..., 7] == 0).all()))

        _module, torch, combat_buffer = self.make_buffer(action_head='gaussian')
        combat_executed = actions.clamp(0, 1) * torch.tensor(module.TASK_MASKS['combat'], dtype=torch.float32)
        self.fill_rollout(combat_buffer, torch, module, task='combat', actions=actions,
                          executed_actions=combat_executed)
        combat_buffer.validate('combat', np.asarray(module.TASK_MASKS['combat'], dtype=bool))
        self.assertTrue(bool((combat_buffer.executed_actions[..., 7] > 0).all()))

    def test_bernoulli_actions_must_be_exact_binary_and_are_masked(self):
        module, torch, buffer = self.make_buffer(action_head='bernoulli_10')
        actions = torch.ones((buffer.n_envs, 10), dtype=torch.float32)
        executed = actions * torch.tensor(module.TASK_MASKS['navigation'], dtype=torch.float32)
        self.fill_rollout(buffer, torch, module, task='navigation', actions=actions,
                          executed_actions=executed)
        buffer.validate('navigation', np.asarray(module.TASK_MASKS['navigation'], dtype=bool))
        self.assertTrue(bool((buffer.executed_actions[..., 7] == 0).all()))
        buffer.actions[0, 0, 0] = .5
        with self.assertRaisesRegex(ValueError, 'Bernoulli rollout actions'):
            buffer.validate('navigation', np.asarray(module.TASK_MASKS['navigation'], dtype=bool))


class NativeCurriculumContractTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('native_curriculum_ppo'),
                             'the CUDA curriculum adapter is not implemented')
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch is not installed in this local contract environment')
        return importlib.import_module('native_curriculum_ppo')

    def test_frozen_curriculum_source_is_unchanged(self):
        module = self.module()
        self.assertEqual(module.curriculum_source_metadata()['sha256'],
                         '380dcafe717366ea3c94fb82f028a329baab0db3b2833b5c9145218ef834eb40')
        self.assertTrue(module.curriculum_source_metadata()['supported'])

    def test_experiment_settings_retain_curriculum_not_raw_combat_defaults(self):
        module = self.module()
        settings = module.contact_settings(batch_size=16384, update_mode='deferred_metrics')
        self.assertEqual(settings['n_steps'], 512)
        self.assertEqual(settings['n_epochs'], 4)
        self.assertEqual(settings['target_kl'], .01)
        self.assertEqual(settings['ent_coef'], .001)
        self.assertEqual(settings['batch_size'], 16384)
        for batch in (True, 0, 4096, 8192.0):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                module.contact_settings(batch_size=batch)
        with self.assertRaises(ValueError):
            module.contact_settings(update_mode='compiled')

    def test_environment_requires_explicit_contact_identity_and_mask(self):
        module = self.module()
        env = SimpleNamespace(curriculum_task='combat', curriculum_version=module.CONTACT_VERSION,
                              allowed_control_mask=module.COMBAT_MASK,
                              host_has_timeouts=lambda: False)
        module.validate_contact_environment(env)
        for name, value in (('curriculum_task', 'navigation'), ('curriculum_version', 'raw'),
                            ('allowed_control_mask', (True,)*10), ('host_has_timeouts', lambda: True)):
            original = getattr(env, name)
            setattr(env, name, value)
            with self.subTest(field=name), self.assertRaises(ValueError):
                module.validate_contact_environment(env)
            setattr(env, name, original)

    def test_task_version_map_defaults_to_b1_and_can_accept_b2_combat(self):
        module = self.module()
        self.assertEqual(module.DEFAULT_TASK_VERSIONS['combat'], module.CONTACT_VERSION)
        self.assertEqual(module.TASK_VERSIONS['combat'], module.CONTACT_VERSION)
        self.assertIsNot(module.TASK_VERSIONS, module.DEFAULT_TASK_VERSIONS)
        versions = module.normalized_task_versions()
        self.assertIsNot(versions, module.DEFAULT_TASK_VERSIONS)
        for invalid in ({'combat': ''}, {'combat': None}, {'full_combat': module.CONTACT_VERSION}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                module.normalized_task_versions(invalid)
        env_b1 = SimpleNamespace(curriculum_task='combat', curriculum_version=module.CONTACT_VERSION,
                                 allowed_control_mask=module.COMBAT_MASK,
                                 host_has_timeouts=lambda: False)
        env_b2 = SimpleNamespace(curriculum_task='combat', curriculum_version='bridge_b2_reengagement-v1',
                                 allowed_control_mask=module.COMBAT_MASK,
                                 host_has_timeouts=lambda: False)
        module.validate_task_environment(env_b1, 'combat')
        with self.assertRaisesRegex(ValueError, 'curriculum version'):
            module.validate_task_environment(env_b2, 'combat')
        b2_versions = module.normalized_task_versions({'combat': 'bridge_b2_reengagement-v1'})
        module.validate_task_environment(env_b2, 'combat', b2_versions)
        with self.assertRaisesRegex(ValueError, 'curriculum version'):
            module.validate_task_environment(env_b1, 'combat', b2_versions)
        self.assertEqual(module.DEFAULT_TASK_VERSIONS['combat'], module.CONTACT_VERSION)

    def test_model_owned_task_versions_survive_copy_and_task_switch_checks(self):
        module = self.module()
        expected = {'combat': 'bridge_b2_reengagement-v1'}
        model = object.__new__(module.NativeCurriculumPPO)
        model.task_versions = module.normalized_task_versions(expected)
        model.task = 'combat'
        copied = module.normalized_task_versions(model.task_versions)
        self.assertEqual(copied['combat'], 'bridge_b2_reengagement-v1')
        copied['combat'] = module.CONTACT_VERSION
        self.assertEqual(model.task_versions['combat'], 'bridge_b2_reengagement-v1')
        env_b2 = SimpleNamespace(curriculum_task='combat', curriculum_version='bridge_b2_reengagement-v1',
                                 allowed_control_mask=module.COMBAT_MASK,
                                 host_has_timeouts=lambda: False)
        module.validate_task_environment(env_b2, 'combat', model.task_versions)
        tensor_env = SimpleNamespace(env=env_b2)
        with patch.object(module, '_require_cuda', wraps=module._require_cuda) as require_cuda, \
             patch.object(module.NativeCurriculumPPO, '_same_tensor_abi'), \
             patch.object(module.NativeCurriculumPPO, '_save_live_task_state'), \
             patch.object(module.NativeCurriculumPPO, '_restore_or_reset_task_state'):
            model.rollout_buffer = SimpleNamespace(pos=0, full=False, consumed=True)
            model._collecting = False
            model.tensor_vec_env = SimpleNamespace(as_sb3_vec_env=lambda: object())
            model.policy = SimpleNamespace(set_task=lambda *_args: None)
            model.num_timesteps = 0
            model.task_history = []
            tensor_env.device = SimpleNamespace(type='cuda')
            tensor_env.observation_space = SimpleNamespace(shape=(256,), dtype=np.float32)
            tensor_env.action_space = SimpleNamespace(shape=(10,), dtype=np.float32)
            tensor_env.num_envs = 1
            tensor_env.as_sb3_vec_env = lambda: object()
            with patch.object(module.th.cuda, 'is_available', return_value=True), \
                 patch.object(module, 'TensorVecEnv', (SimpleNamespace,)):
                model.set_task_environment('combat', tensor_env)
            require_cuda.assert_called_once_with(tensor_env, 'combat', model.task_versions)

    def test_drain_records_validate_against_model_owned_task_versions(self):
        module = self.module()
        versions = module.normalized_task_versions({'combat': 'bridge_b2_reengagement-v1'})
        good = [{
            'curriculum_task': 'combat',
            'curriculum_version': 'bridge_b2_reengagement-v1',
            'allowed_control_mask': module.COMBAT_MASK,
        }]
        module.validate_episode_infos(good, 'combat', module.COMBAT_MASK, versions)
        bad = [dict(good[0], curriculum_version=module.CONTACT_VERSION)]
        with self.assertRaisesRegex(ValueError, 'curriculum task/mask/version'):
            module.validate_episode_infos(bad, 'combat', module.COMBAT_MASK, versions)

    def test_wrong_drained_version_fails_before_optimizer_update(self):
        module = self.module()
        model = object.__new__(module.NativeCurriculumPPO)
        model.task = 'combat'
        model.task_versions = module.normalized_task_versions({'combat': 'bridge_b2_reengagement-v1'})
        model._policy_contract = lambda: None
        model.rollout_buffer = SimpleNamespace(
            validate=lambda *args: None,
            consumed=False,
            drained_episode_infos=[{
                'curriculum_task': 'combat',
                'curriculum_version': module.CONTACT_VERSION,
                'allowed_control_mask': module.COMBAT_MASK,
            }],
        )
        model.policy = SimpleNamespace(optimizer=SimpleNamespace(step=lambda: None))
        model.update_mode = 'deferred_metrics'
        with self.assertRaisesRegex(ValueError, 'curriculum task/mask/version'), \
             patch.object(module, 'deferred_metrics_train', side_effect=AssertionError('optimizer update reached')), \
             patch.object(module.TensorPPO, 'train', side_effect=AssertionError('optimizer update reached')):
            model.train()
        self.assertFalse(model.rollout_buffer.consumed)

    def test_load_reads_b2_task_versions_before_environment_validation_and_legacy_defaults_b1(self):
        module = self.module()
        b2_versions = module.normalized_task_versions({'combat': 'bridge_b2_reengagement-v1'})
        with patch.object(module.TensorPPO, '_excluded_save_params', return_value=[]):
            model = object.__new__(module.NativeCurriculumPPO)
            model.task_versions = b2_versions
            self.assertNotIn('task_versions', model._excluded_save_params())
        env = SimpleNamespace(
            device='cuda',
            env=SimpleNamespace(curriculum_task='combat', curriculum_version='bridge_b2_reengagement-v1',
                                allowed_control_mask=module.COMBAT_MASK,
                                host_has_timeouts=lambda: False),
        )
        source = SimpleNamespace(task_versions=b2_versions, update_mode='deferred_metrics')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.zip'
            path.touch()
            digest = hashlib.sha256(b'').hexdigest()
            with patch.object(module, '_require_sources'), \
                 patch.object(module, '_require_cuda') as require_cuda, \
                 patch.object(module.CurriculumPPO, 'load', return_value=source), \
                 patch.object(module.NativeCurriculumPPO, 'from_curriculum',
                              return_value=SimpleNamespace()) as from_curriculum, \
                 patch.object(module.th.cuda, 'get_rng_state_all', return_value=[]), \
                 patch.object(module.th.cuda, 'set_rng_state_all'):
                module.NativeCurriculumPPO.load(path, env=env, expected_sha256=digest)
        require_cuda.assert_called_once_with(env, 'combat', b2_versions)
        self.assertEqual(from_curriculum.call_args.kwargs['expected_task_versions'], b2_versions)

        legacy_source = SimpleNamespace(update_mode='deferred_metrics')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'legacy.zip'
            path.touch()
            with patch.object(module, '_require_sources'), \
                 patch.object(module, '_require_cuda') as require_cuda, \
                 patch.object(module.CurriculumPPO, 'load', return_value=legacy_source), \
                 patch.object(module.NativeCurriculumPPO, 'from_curriculum',
                              return_value=SimpleNamespace()) as from_curriculum, \
                 patch.object(module.th.cuda, 'get_rng_state_all', return_value=[]), \
                 patch.object(module.th.cuda, 'set_rng_state_all'):
                module.NativeCurriculumPPO.load(path, env=env, expected_sha256=digest)
        require_cuda.assert_called_once_with(env, 'combat', module.DEFAULT_TASK_VERSIONS)
        self.assertEqual(from_curriculum.call_args.kwargs['expected_task_versions'],
                         module.DEFAULT_TASK_VERSIONS)

    def test_cpu_load_fails_before_checkpoint_read_or_policy_construction(self):
        module = self.module()
        with self.assertRaisesRegex(ValueError, 'CUDA'), \
             patch.object(module.th.cuda, 'is_available', return_value=False), \
             patch.object(module.CurriculumPPO, 'load', side_effect=AssertionError('must not load')):
            module.NativeCurriculumPPO.load_contact_checkpoint('missing.zip', env=None)

    def test_checkpoint_hash_gate_checks_bytes(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.zip'
            # A tiny file is a hash-gate fixture, never loaded as a policy.
            path.touch()
            self.assertEqual(module.verify_checkpoint(path, hashlib.sha256(b'').hexdigest()),
                             hashlib.sha256(b'').hexdigest())
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                module.verify_checkpoint(path, '0'*64)

    def test_explained_variance_uses_original_sb3_env_major_reduction_order(self):
        module = self.module()
        from stable_baselines3.common.utils import explained_variance
        generator = np.random.default_rng(3)
        values = generator.normal(size=(512, 16))
        returns = values + generator.normal(size=(512, 16))
        values, returns = values.astype(np.float32), returns.astype(np.float32)
        time_major = explained_variance(values.flatten(), returns.flatten())
        expected = explained_variance(values.swapaxes(0, 1).flatten(), returns.swapaxes(0, 1).flatten())
        self.assertNotEqual(time_major, expected)
        self.assertTrue(hasattr(module, 'curriculum_explained_variance'),
                        'native metrics must reproduce the original env-major reduction order')
        self.assertEqual(module.curriculum_explained_variance(values, returns), expected)

    def test_public_load_methods_preserve_caller_rng_around_checkpoint_deserialization(self):
        module = self.module()
        torch = module.th
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        def simulated_checkpoint_load(*args, **kwargs):
            # Model construction is forbidden locally; emulate only the
            # deserializer's documented RNG side effect at its external seam.
            random.seed(93)
            np.random.seed(93)
            torch.random.default_generator.manual_seed(93)
            return SimpleNamespace()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/'empty.zip'
                path.touch()
                with patch.object(module, '_require_cuda'), \
                     patch.object(module, '_require_sources'), \
                     patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                     patch.object(torch.cuda, 'set_rng_state_all'), \
                     patch.object(module.CurriculumPPO, 'load', side_effect=simulated_checkpoint_load), \
                     patch.object(module.NativeCurriculumPPO, 'from_curriculum', return_value=SimpleNamespace()):
                    for method in ('load_contact_checkpoint', 'load'):
                        with self.subTest(method=method):
                            getattr(module.NativeCurriculumPPO, method)(path,
                                env=SimpleNamespace(device='cuda'), expected_sha256=hashlib.sha256(b'').hexdigest())
                            self.assertEqual(random.getstate(), before[0])
                            np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
                            self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        finally:
            random.setstate(before[0])
            np.random.set_state(before[1])
            torch.set_rng_state(before[2])

    def test_update_diagnostics_count_outer_protected_adam_calls_not_nested_hooks(self):
        """Tiny optimizer-only reproduction; no policy network or inference."""
        module = self.module()
        torch = module.th
        # CurriculumPolicy constructs ordinary Adam before replacing it with
        # protected Adam. That makes both parent and overriding step hookable.
        torch.optim.Adam([torch.nn.Parameter(torch.ones(1))])

        class TensorOwner:
            def __init__(self):
                self.action_net = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(10, 1)),
                                                   bias=torch.nn.Parameter(torch.ones(10)))
                self.log_std = torch.nn.Parameter(torch.ones(10))
                self.trainable_rows = torch.tensor(module.COMBAT_MASK)
                self.active_indices = tuple(np.flatnonzero(module.COMBAT_MASK))
                self.optimizer = module._ProtectedAdam(self.parameters(), policy=self, lr=.0001)
            def parameters(self):
                return [self.action_net.weight, self.action_net.bias, self.log_std]

        def step(owner):
            for parameter in owner.parameters():
                parameter.grad = torch.ones_like(parameter)
            owner.optimizer.step()

        for mode in ('sb3', 'deferred_metrics'):
            cases = ((16, 4, False, False), (3, 1, False, False), (0, 1, False, False),
                     (1, 1, True, False), (3, 1, False, True), (1, 1, True, True))
            for completed, epochs, fail, preexisting_step in cases:
                with self.subTest(mode=mode, completed_steps=completed, failure=fail,
                                  preexisting_step=preexisting_step):
                    owner, reference = TensorOwner(), TensorOwner()
                    if preexisting_step:
                        bound_step = owner.optimizer.step
                        def existing_step(*args, bound_step=bound_step, **kwargs):
                            return bound_step(*args, **kwargs)
                        owner.optimizer.step = existing_step
                    model = object.__new__(module.NativeCurriculumPPO)
                    model.policy = owner
                    model._policy_contract = lambda: None
                    model.rollout_buffer = SimpleNamespace(validate=lambda *args: None, consumed=False,
                        values=np.zeros((2, 2), dtype=np.float32),
                        returns=np.arange(4, dtype=np.float32).reshape(2, 2))
                    metrics = {'train/value_loss': .25, 'train/entropy_loss': -.5}
                    model._logger = SimpleNamespace(name_to_value=metrics,
                                                    record=lambda key, value, metrics=metrics: metrics.__setitem__(key, value))
                    model.task, model.update_mode, model._n_updates = 'combat', mode, 0
                    model.task_versions = module.normalized_task_versions()
                    model.n_epochs, model.n_steps, model.n_envs, model.batch_size = 4, 512, 64, 8192
                    model.ent_coef = .001
                    calls = []
                    hook = owner.optimizer.register_step_post_hook(lambda *args, calls=calls: calls.append(1))
                    def update(
                        *args,
                        completed=completed,
                        owner=owner,
                        fail=fail,
                        preexisting_step=preexisting_step,
                        model=model,
                        epochs=epochs,
                    ):
                        for _ in range(completed):
                            step(owner)
                        if fail:
                            if preexisting_step:
                                # ProtectedAdam rejects closures before updating.
                                # Exercise an exception inside step itself too.
                                owner.optimizer.step(closure=lambda: None)
                            raise RuntimeError('injected update failure')
                        model._n_updates += epochs
                    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
                    try:
                        with patch.object(module, 'deferred_metrics_train', side_effect=update), \
                             patch.object(module.TensorPPO, 'train', side_effect=update):
                            if fail:
                                error = ValueError if preexisting_step else RuntimeError
                                message = 'closures' if preexisting_step else 'injected update failure'
                                with self.assertRaisesRegex(error, message):
                                    model.train()
                            else:
                                model.train()
                    finally:
                        hook.remove()
                    # Existing PyTorch hooks keep their exact behavior; only our
                    # work counter must represent one outer call per minibatch.
                    self.assertEqual(len(calls), 2*completed)
                    if fail:
                        self.assertFalse(model.rollout_buffer.consumed)
                    else:
                        self.assertEqual(model.last_update_diagnostics, {
                            'attempted_epochs': epochs, 'optimizer_steps': completed,
                            'kl_early_stop': completed < 16})
                    if preexisting_step:
                        self.assertIs(owner.optimizer.__dict__['step'], existing_step)
                    else:
                        self.assertNotIn('step', owner.optimizer.__dict__)
                    self.assertEqual(random.getstate(), before[0])
                    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
                    self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
                    for _ in range(completed):
                        step(reference)
                    for actual, expected in zip(owner.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        for key, value in owner.optimizer.state[actual].items():
                            torch.testing.assert_close(value, reference.optimizer.state[expected][key], rtol=0, atol=0)


class NativeCurriculumCudaTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch is not installed in this local contract environment')
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable; no CPU policy construction, inference or training')
        self.torch = torch
        self.module = importlib.import_module('native_curriculum_ppo')
        self.assertTrue(self.module.curriculum_source_metadata()['supported'])
        from ppo_update_optimizations import sb3_train_source_metadata
        self.assertTrue(sb3_train_source_metadata()['supported'])
        self.addCleanup(self.restore_rng, self.rng_state())

    def rng_state(self):
        return (random.getstate(), np.random.get_state(), self.torch.get_rng_state().clone(),
                [value.clone() for value in self.torch.cuda.get_rng_state_all()])

    def restore_rng(self, state):
        random.setstate(state[0])
        np.random.set_state(state[1])
        self.torch.set_rng_state(state[2])
        self.torch.cuda.set_rng_state_all(state[3])

    def assert_tree_equal(self, left, right, *, exact=True):
        from tests.test_ppo_update_optimizations import PPOUpdateParityTests
        PPOUpdateParityTests.assert_tree_equal(self, left, right, exact=exact)

    def environment(self, task='combat'):
        """Tensor fixture tests collector contracts, not native game fidelity."""
        from gymnasium import spaces

        from tensor_ppo import TensorStep, TensorVecEnv
        torch, module = self.torch, self.module
        mask = module.TASK_MASKS[task]
        version = module.TASK_VERSIONS[task]
        reset_value = .05 if task == 'combat' else .15

        class Fixture:
            num_envs = 16
            observation_space = spaces.Box(-10, 10, (256,), dtype=np.float32)
            action_space = spaces.Box(0, 1, (10,), dtype=np.float32)
            curriculum_task = task
            curriculum_version = version
            allowed_control_mask = mask
            device = torch.device('cuda', torch.cuda.current_device())
            def reset_tensor(self, seeds=None):
                self.tick = 0
                return torch.full((16, 256), reset_value, device=self.device)
            def step_tensor(self, action):
                self.last_action = action.clone()
                self.tick += 1
                obs = torch.full((16, 256), .01*(self.tick % 20), device=self.device)
                done = torch.full((16,), self.tick % 32 == 0, device=self.device, dtype=torch.bool)
                truncated = torch.zeros_like(done)
                if task == 'navigation' and self.tick % 17 == 0:
                    truncated[::3] = True
                return TensorStep(obs, -action[:, 0].contiguous(), done,
                                  truncated, obs.clone())
            def host_has_timeouts(self):
                return task == 'navigation'
            def drain_episode_infos(self):
                return [{
                    'episode': {'r': 1.0, 'l': self.tick},
                    'curriculum_task': task,
                    'curriculum_version': version,
                    'allowed_control_mask': mask,
                }] if self.tick else []
        return TensorVecEnv(Fixture(), device='cuda')

    def model(self, mode='sb3'):
        from stable_baselines3.common.logger import configure
        model = self.module.NativeCurriculumPPO(env=self.environment(), update_mode=mode,
            policy_kwargs={'net_arch': {'pi': [8], 'vf': [8]}}, seed=741)
        model.set_logger(configure(folder=None, format_strings=[]))
        self.addCleanup(model.logger.close)
        return model

    def collect(self, model):
        from stable_baselines3.common.callbacks import BaseCallback
        class Callback(BaseCallback):
            def _on_step(self):
                return True
        _, callback = model._setup_learn(8192*4, callback=Callback(), reset_num_timesteps=False)
        self.assertTrue(model.collect_rollouts(model.env, callback, model.rollout_buffer, 512))
        callback.locals.clear()
        return model.rollout_buffer

    def test_task_environment_switch_updates_shim_mask_and_resets_new_env_instances(self):
        model = self.model()
        from stable_baselines3.common.callbacks import BaseCallback
        class Callback(BaseCallback):
            def _on_step(self):
                return True
        _, callback = model._setup_learn(8192*4, callback=Callback(), reset_num_timesteps=False)
        callback.locals.clear()
        model.rollout_buffer.full = True
        model.rollout_buffer.consumed = True

        nav_env = self.environment('navigation')
        model.set_task_environment('navigation', nav_env)
        self.assertIs(model.tensor_vec_env, nav_env)
        self.assertEqual(model.env.num_envs, nav_env.num_envs)
        self.assertEqual(model.task, 'navigation')
        self.assertEqual(model.policy.task, 'navigation')
        self.torch.testing.assert_close(model.policy.allowed_control_mask.detach().cpu(),
                                        self.torch.as_tensor(self.module.TASK_MASKS['navigation']))
        self.assertTrue(bool((model._last_obs == .15).all()))
        model._last_obs.fill_(.77)
        model._last_episode_starts.zero_()
        model.rollout_buffer.full = True
        model.rollout_buffer.consumed = True

        combat_env = self.environment('combat')
        model.set_task_environment('combat', combat_env, calibration=True)
        self.assertIs(model.tensor_vec_env, combat_env)
        self.assertEqual(model.policy.task, 'combat')
        self.assertTrue(model.policy.calibration)
        self.assertTrue(bool((model._last_obs == .05).all()))
        self.assertTrue(bool((model._last_episode_starts == 1).all()))

        model._last_obs.fill_(.33)
        model._last_episode_starts.zero_()
        model.rollout_buffer.full = True
        model.rollout_buffer.consumed = True
        model.set_task_environment('navigation', nav_env)
        model.rollout_buffer.full = True
        model.rollout_buffer.consumed = True
        model.set_task_environment('combat', combat_env, calibration=True)
        self.assertTrue(bool((model._last_obs == .33).all()))
        self.assertTrue(bool((model._last_episode_starts == 0).all()))

    def copy_buffer(self, source, target):
        target.reset()
        for name in ('observations', 'actions', 'rewards', 'episode_starts', 'values', 'log_probs',
                     'advantages', 'returns', 'executed_actions', 'task_ids', 'allowed_control_masks',
                     '_invalid_data'):
            getattr(target, name).copy_(getattr(source, name))
        target.pos, target.full = source.pos, source.full
        target.task, target.consumed = source.task, False
        target._computed_returns_and_advantage = True

    def test_collector_raw_samples_and_execution_mask_guards(self):
        model = self.model()
        buffer = self.collect(model)
        self.assertTrue((buffer.actions[..., 5].abs() > 0).any())
        self.assertTrue((buffer.executed_actions[..., [5, 8, 9]] == 0).all())
        expected = buffer.actions.clamp(0, 1)*model.policy.allowed_control_mask
        self.assertTrue(self.torch.equal(expected, buffer.executed_actions))
        observed = buffer.observations.reshape(-1, 256)
        actions = buffer.actions.reshape(-1, 10)
        # Match collection's per-step batch geometry. One giant matrix multiply
        # can differ by a final FP32 bit even when every stored log probability
        # is correct because cuBLAS selects a different reduction geometry.
        log_probs = self.torch.cat([
            model.policy.evaluate_actions(
                observed[offset:offset + model.n_envs],
                actions[offset:offset + model.n_envs],
            )[1]
            for offset in range(0, observed.shape[0], model.n_envs)
        ])
        self.torch.testing.assert_close(log_probs, buffer.log_probs.reshape(-1), rtol=0, atol=0)
        self.torch.testing.assert_close(model.tensor_vec_env.env.last_action, buffer.executed_actions[-1], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'consumed'):
            model.set_task('combat', calibration=True)
        with self.assertRaisesRegex(ValueError, 'consumed'):
            self.collect(model)
        buffer.allowed_control_masks[0, 0, 7] = False
        with self.assertRaisesRegex(ValueError, 'mask'):
            model.train()

    def test_native_collector_rejects_nonfinite_policy_outputs_before_env_step_or_buffer_write(self):
        for field in ('actions', 'values', 'log_probs'):
            with self.subTest(field=field):
                model = self.model()
                from stable_baselines3.common.callbacks import BaseCallback
                class Callback(BaseCallback):
                    def _on_step(self):
                        return True
                _, callback = model._setup_learn(8192*4, callback=Callback(), reset_num_timesteps=False)

                def bad_forward(policy, observations, deterministic=False, field=field):
                    batch = observations.shape[0]
                    actions = self.torch.zeros((batch, 10), device=observations.device)
                    values = self.torch.zeros((batch, 1), device=observations.device)
                    log_probs = self.torch.zeros((batch,), device=observations.device)
                    if field == 'actions':
                        actions[0, 0] = float('nan')
                    elif field == 'values':
                        values[0, 0] = float('inf')
                    else:
                        log_probs[0] = float('-inf')
                    return actions, values, log_probs

                model.policy.forward = MethodType(bad_forward, model.policy)
                with self.assertRaisesRegex(ValueError, 'finite'):
                    model.collect_rollouts(model.env, callback, model.rollout_buffer, 512)
                self.assertEqual(model.tensor_vec_env.env.tick, 0)
                self.assertEqual(model.rollout_buffer.pos, 0)
                self.assertFalse(model.rollout_buffer.full)
                callback.locals.clear()

    def test_plain_deferred_repeated_updates_protect_critics_rows_and_roundtrip(self):
        plain, deferred = self.model(), self.model('deferred_metrics')
        self.assert_tree_equal(plain.policy.state_dict(), deferred.policy.state_dict())
        protected = {name: value.clone() for name, value in plain.policy.state_dict().items()
                     if name.startswith(('mlp_extractor.value_net.', 'value_net.'))}
        output_rows = {name: value[[5, 8, 9]].detach().clone()
                       for name, value in (('log_std', plain.policy.log_std),
                                           ('weight', plain.policy.action_net.weight),
                                           ('bias', plain.policy.action_net.bias))}
        for _ in range(2):
            buffer = self.collect(plain)
            self.copy_buffer(buffer, deferred.rollout_buffer)
            before = self.rng_state()
            plain.train()
            after = self.rng_state()
            self.restore_rng(before)
            deferred.train()
            self.assert_tree_equal(after, self.rng_state())
            self.assert_tree_equal(plain.policy.state_dict(), deferred.policy.state_dict())
            self.assert_tree_equal(plain.policy.optimizer.state_dict(), deferred.policy.optimizer.state_dict())
            self.assert_tree_equal(dict(plain.logger.name_to_value), dict(deferred.logger.name_to_value))
            self.assertEqual(plain.last_update_diagnostics, deferred.last_update_diagnostics)
            self.assertTrue(buffer.consumed)
            with self.assertRaisesRegex(ValueError, 'unconsumed'):
                plain.train()
        for name, value in protected.items():
            self.assertTrue(self.torch.equal(value, plain.policy.state_dict()[name]))
        for name, value in (('log_std', plain.policy.log_std), ('weight', plain.policy.action_net.weight),
                            ('bias', plain.policy.action_net.bias)):
            self.assertTrue(self.torch.equal(output_rows[name], value[[5, 8, 9]]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'joint.zip'
            plain.save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            restored = self.module.NativeCurriculumPPO.load(path, env=self.environment(), expected_sha256=digest)
            self.assert_tree_equal(plain.policy.state_dict(), restored.policy.state_dict())
            for key, values in plain.policy.optimizer.state_dict()['state'].items():
                for name, value in values.items():
                    other = restored.policy.optimizer.state_dict()['state'][key][name]
                    self.assert_tree_equal(value.cpu() if self.torch.is_tensor(value) else value,
                                           other.cpu() if self.torch.is_tensor(other) else other)
            self.assertIs(restored.policy.optimizer._policy_ref(), restored.policy)
            self.assertFalse(restored.calibration)
            self.assertEqual(restored.num_timesteps, plain.num_timesteps)

    def test_canonical_checkpoint_conversion_keeps_moments_and_ends_calibration(self):
        checkpoint = os.environ.get('AI2_CONTACT_CHECKPOINT')
        if not checkpoint:
            self.skipTest('Set AI2_CONTACT_CHECKPOINT to the canonical calibrated checkpoint')
        module = self.module
        module.verify_checkpoint(checkpoint, module.CALIBRATED_CHECKPOINT_SHA256)
        source = module.CurriculumPPO.load(checkpoint, device='cuda')
        self.assertTrue(source.calibration)
        before = self.rng_state()
        model = module.NativeCurriculumPPO.from_curriculum(source, self.environment(), joint=True)
        self.assert_tree_equal(before, self.rng_state())
        self.assert_tree_equal(source.policy.state_dict(), model.policy.state_dict())
        self.assertFalse(model.policy.optimizer.state_dict()['state'])
        self.assertIn('policy/action tensors copied', model.optimizer_provenance)
        self.assertIn('optimizer state not transferred', model.optimizer_provenance)
        self.assertIs(model.policy.optimizer._policy_ref(), model.policy)
        self.assertFalse(model.policy.calibration)
        self.assertTrue(all(p.requires_grad for p in model.policy.mlp_extractor.policy_net.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.policy.mlp_extractor.value_net.parameters()))
        self.assertEqual(model.num_timesteps, 10059776)

    def test_source_conversion_does_not_reseed_the_fixed_scenario_schedule(self):
        from curriculum_ppo import CurriculumPPO
        env = self.environment()
        source = CurriculumPPO(env=env.as_sb3_vec_env(), device='cuda', calibration=False,
            policy_kwargs={'net_arch': {'pi': [8], 'vf': [8]}}, seed=17)
        def forbidden_reseed(*args, **kwargs):
            raise AssertionError('Conversion must not replace caller-owned scenario seeds')
        env.env.seed = forbidden_reseed
        model = self.module.NativeCurriculumPPO.from_curriculum(source, env)
        self.assertEqual(model.seed, source.seed)

    def test_source_conversion_reapplies_requested_task_after_parameter_restore(self):
        from curriculum_ppo import CurriculumPPO
        combat_env = self.environment('combat')
        source = CurriculumPPO(env=combat_env.as_sb3_vec_env(), device='cuda', calibration=False,
            policy_kwargs={'net_arch': {'pi': [8], 'vf': [8]}}, seed=29)
        source.task_history.append({'task': 'combat', 'num_timesteps': 0})
        source.policy.optimizer.zero_grad(set_to_none=True)
        for parameter in source.policy.parameters():
            if parameter.requires_grad:
                parameter.grad = self.torch.ones_like(parameter)
        source.policy.optimizer.step()
        source.policy.optimizer.zero_grad(set_to_none=True)
        source_optimizer = copy.deepcopy(source.policy.optimizer.state_dict())
        self.assertTrue(source_optimizer['state'])

        model = self.module.NativeCurriculumPPO.from_curriculum(
            source, self.environment('navigation'), task='navigation',
        )

        self.assertEqual(model.task, 'navigation')
        self.assertEqual(model.policy.task, 'navigation')
        self.assertFalse(model.calibration)
        self.assertFalse(model.policy.calibration)
        self.torch.testing.assert_close(
            model.policy.allowed_control_mask.detach().cpu(),
            self.torch.as_tensor(self.module.TASK_MASKS['navigation']),
        )
        self.torch.testing.assert_close(
            model.policy.action_dist.mask.detach().cpu(),
            self.torch.as_tensor(self.module.TASK_MASKS['navigation']),
        )
        self.assertTrue(all(parameter.requires_grad
                            for parameter in model.policy.mlp_extractor.value_net.parameters()))
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in model.policy.combat_value_features.parameters()))
        self.assertEqual(model.task_history, source.task_history)
        self.assert_tree_equal(source_optimizer, model.policy.optimizer.state_dict())
        model._policy_contract()

    def test_bernoulli_conversion_provenance_survives_metadata_copy_and_uses_fresh_optimizer(self):
        from curriculum_ppo import CurriculumPPO
        env = self.environment()
        source = CurriculumPPO(env=env.as_sb3_vec_env(), device='cuda', calibration=False,
            policy_kwargs={'net_arch': {'pi': [8], 'vf': [8]}}, seed=19)
        source.optimizer_provenance = 'source optimizer metadata must not survive Bernoulli conversion'
        model = self.module.NativeCurriculumPPO.from_curriculum(source, env, action_head='bernoulli_10')
        self.assertEqual(
            model.optimizer_provenance,
            'fresh neutral bernoulli_10 action head; source trunk/value copied; fresh Adam optimizer not transferred',
        )
        self.assertEqual(model.policy.action_head, 'bernoulli_10')
        self.assertIs(model.policy.optimizer._policy_ref(), model.policy)
        self.assertFalse(model.policy.optimizer.state_dict()['state'])

    def test_no_arg_from_curriculum_preserves_nondefault_action_arm_and_optimizer_claims(self):
        from curriculum_ppo import CurriculumPPO
        env = self.environment()
        source = CurriculumPPO(env=env.as_sb3_vec_env(), device='cuda', calibration=False,
            action_head='bernoulli_10', action_ab_arm='B_neutral',
            policy_kwargs={'net_arch': {'pi': [8], 'vf': [8]}}, seed=23)
        source.action_ab_arm = 'B_projected'
        source.action_init_strategy = 'projected'
        source.action_init_metadata = {'schema': 'ai2-action-ab-init-v1', 'arm': 'B_projected',
            'action_head': 'bernoulli_10', 'strategy': 'projected', 'source': 'test-parent'}
        source.optimizer_provenance = 'source optimizer provenance must not be copied'
        model = self.module.NativeCurriculumPPO.from_curriculum(source, env)
        self.assertEqual(model.action_head, 'bernoulli_10')
        self.assertEqual(model.policy.action_head, 'bernoulli_10')
        self.assertEqual(model.action_ab_arm, 'B_projected')
        self.assertEqual(model.action_init_strategy, 'projected')
        self.assertEqual(model.action_init_metadata['arm'], 'B_projected')
        self.assertEqual(model.action_init_metadata['strategy'], 'projected')
        self.assertIn('policy/action tensors copied', model.optimizer_provenance)
        self.assertIn('source optimizer state copied', model.optimizer_provenance)
        self.assertNotEqual(model.optimizer_provenance, source.optimizer_provenance)
        self.assertFalse(model.policy.optimizer.state_dict()['state'])
        self.assert_tree_equal(source.policy.state_dict(), model.policy.state_dict())

    def test_original_curriculum_update_matches_native_from_the_same_saved_rollout(self):
        from stable_baselines3.common.logger import configure

        from curriculum_ppo import CurriculumPPO
        candidate = self.model('deferred_metrics')
        reference = CurriculumPPO(env=self.environment().as_sb3_vec_env(), device='cuda',
            calibration=False, policy_kwargs={'net_arch': {'pi': [8], 'vf': [8]}}, seed=741)
        reference.set_parameters(candidate.get_parameters(), exact_match=True, device='cuda')
        reference.set_logger(configure(folder=None, format_strings=[]))
        self.addCleanup(reference.logger.close)
        source = self.collect(candidate)
        target = reference.rollout_buffer
        target.reset()
        for name in ('observations', 'actions', 'rewards', 'episode_starts', 'values', 'log_probs',
                     'advantages', 'returns', 'executed_actions'):
            getattr(target, name)[:] = getattr(source, name).cpu().numpy()
        target.tasks[:] = 'combat'
        target.allowed_control_masks[:] = self.module.COMBAT_MASK
        target.task, target.pos, target.full = 'combat', 512, True
        before = self.rng_state()
        reference.train()
        after = self.rng_state()
        self.restore_rng(before)
        candidate.train()
        self.assert_tree_equal(after, self.rng_state())
        self.assert_tree_equal(reference.policy.state_dict(), candidate.policy.state_dict())
        self.assert_tree_equal(reference.policy.optimizer.state_dict(), candidate.policy.optimizer.state_dict())
        self.assert_tree_equal(dict(reference.logger.name_to_value), dict(candidate.logger.name_to_value))

    def test_nonfinite_buffer_and_changed_task_fail_before_optimizer_step(self):
        model = self.model()
        buffer = self.collect(model)
        buffer.advantages[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            model.train()
        self.assertEqual(model._n_updates, 0)
        self.assertFalse(buffer.consumed)

    def test_kl_stop_rejects_minibatch_before_any_optimizer_step_in_both_modes(self):
        plain, deferred = self.model(), self.model('deferred_metrics')
        source = self.collect(plain)
        # Deliberately stale likelihoods exercise SB3's exact stop-before-step rule.
        source.log_probs.add_(2)
        self.copy_buffer(source, deferred.rollout_buffer)
        initial = {name: value.clone() for name, value in plain.policy.state_dict().items()}
        before = self.rng_state()
        plain.train()
        after = self.rng_state()
        self.restore_rng(before)
        deferred.train()
        self.assert_tree_equal(after, self.rng_state())
        self.assert_tree_equal(initial, plain.policy.state_dict())
        self.assert_tree_equal(initial, deferred.policy.state_dict())
        self.assertEqual(plain.last_update_diagnostics,
                         {'attempted_epochs': 1, 'optimizer_steps': 0, 'kl_early_stop': True})
        self.assertEqual(plain.last_update_diagnostics, deferred.last_update_diagnostics)
        self.assert_tree_equal(dict(plain.logger.name_to_value), dict(deferred.logger.name_to_value))

    def test_no_kl_full_update_count_matches_real_adam_parameter_counters(self):
        for mode in ('sb3', 'deferred_metrics'):
            with self.subTest(mode=mode):
                model = self.model(mode)
                # Controlled optimizer diagnostic test only. The production
                # contact runner and checkpoint defaults retain target_kl=.01.
                model.target_kl = None
                self.collect(model)
                model.train()
                expected = model.n_epochs*model.n_steps*model.n_envs//model.batch_size
                self.assertEqual(model.last_update_diagnostics, {
                    'attempted_epochs': model.n_epochs, 'optimizer_steps': expected,
                    'kl_early_stop': False})
                parameter = model.policy.combat_value_net.bias
                self.assertEqual(model.policy.optimizer.state[parameter]['step'].item(), expected)
                self.assertNotIn('step', model.policy.optimizer.__dict__)

    def test_canonical_real_contact_saved_rollouts_match_original_and_both_native_updates(self):
        """Qualification, not a speed benchmark: real lanes and the real large policy."""
        checkpoint = os.environ.get('AI2_CONTACT_CHECKPOINT')
        if not checkpoint:
            self.skipTest('Set AI2_CONTACT_CHECKPOINT to qualify the real canonical contact policy')
        from stable_baselines3.common.logger import configure

        from curriculum_ppo import CurriculumPPO, TaskRolloutBuffer
        from native_contact_lane_vec_env import build_contact_env
        from tensor_ppo import TensorVecEnv

        module, torch = self.module, self.torch
        module.verify_checkpoint(checkpoint, module.CALIBRATED_CHECKPOINT_SHA256)
        artifact_path = os.environ.get('AI2_CONTACT_PARITY_ARTIFACTS')
        if artifact_path:
            artifacts = Path(artifact_path)
            artifacts.mkdir(parents=True, exist_ok=False)
        else:
            temporary = tempfile.TemporaryDirectory(prefix='contact-canonical-parity-')
            self.addCleanup(temporary.cleanup)
            artifacts = Path(temporary.name)

        def environment():
            raw = build_contact_env(16, device='cuda', seed=145, episodes_per_lane=2048,
                                    record_capacity=16*512)
            self.addCleanup(raw.close)
            return TensorVecEnv(raw, device='cuda')

        def attach_logger(model):
            logger = configure(folder=None, format_strings=[])
            self.addCleanup(logger.close)
            model.set_logger(logger)
            return model

        def original_model(path):
            model = CurriculumPPO.load(path, device='cuda')
            model.set_task('combat', calibration=False)
            # Same original NumPy buffer implementation, matched 16-lane geometry.
            model.n_envs = 16
            model.rollout_buffer = TaskRolloutBuffer(512, model.observation_space,
                model.action_space, device='cuda', gamma=model.gamma, gae_lambda=model.gae_lambda,
                n_envs=16)
            return attach_logger(model)

        reference = original_model(checkpoint)
        natives = []
        for mode in ('sb3', 'deferred_metrics'):
            env = environment()
            before = self.rng_state()
            model = module.NativeCurriculumPPO.load_contact_checkpoint(checkpoint, env,
                                                                       update_mode=mode)
            self.assert_tree_equal(before, self.rng_state())
            self.assertEqual(model.policy.net_arch, reference.policy.net_arch)
            # The canonical checkpoint uses SB3's list form, which builds
            # independent actor/value stacks with identical layer widths.
            self.assertEqual(model.policy.net_arch, [1536, 1536, 1536])
            for trunk in (model.policy.mlp_extractor.policy_net,
                          model.policy.mlp_extractor.value_net,
                          model.policy.combat_value_features):
                dimensions = [(layer.in_features, layer.out_features)
                              for layer in trunk if isinstance(layer, torch.nn.Linear)]
                self.assertEqual(dimensions, [(256, 1536), (1536, 1536), (1536, 1536)])
            for head, outputs in ((model.policy.action_net, 10), (model.policy.value_net, 1),
                                  (model.policy.combat_value_net, 1)):
                self.assertEqual((head.in_features, head.out_features), (1536, outputs))
            self.assertFalse(model.calibration)
            self.assert_tree_equal(reference.policy.state_dict(), model.policy.state_dict())
            self.assert_tree_equal(reference.policy.optimizer.state_dict(), model.policy.optimizer.state_dict())
            self.assertIs(model.policy.optimizer._policy_ref(), model.policy)
            natives.append(attach_logger(model))

        tensor_names = ('observations', 'actions', 'rewards', 'episode_starts', 'values', 'log_probs',
                        'advantages', 'returns', 'executed_actions', 'task_ids', 'allowed_control_masks',
                        '_invalid_data')
        receipts = []

        def exact_update(index, original, native_models):
            plain, deferred = native_models
            backend = plain.tensor_vec_env.env.backend
            valid_before = np.asarray(backend.valid_steps).copy()
            collected = self.collect(plain)
            np.testing.assert_array_equal(np.asarray(backend.valid_steps)-valid_before,
                                          np.full(16, 512))
            self.assertFalse(np.asarray(backend.errors).any())
            self.assertTrue(plain.ep_info_buffer)
            self.assertTrue(all(value.is_cuda for value in
                                (collected.observations, collected.actions, collected.rewards)))
            rollout_path = artifacts/f'rollout-{index}.pt'
            torch.save({name: getattr(collected, name) for name in tensor_names}, rollout_path)
            payload = torch.load(rollout_path, map_location='cuda', weights_only=True)
            for name in tensor_names:
                self.assert_tree_equal(getattr(collected, name), payload[name])
            saved = SimpleNamespace(**payload, pos=512, full=True, task='combat')
            for model in native_models:
                self.copy_buffer(saved, model.rollout_buffer)
                model.num_timesteps = plain.num_timesteps
                with self.assertRaisesRegex(ValueError, 'consumed'):
                    model.set_task('combat', calibration=True)
                model.rollout_buffer.validate('combat', module.COMBAT_MASK)
            original.num_timesteps = plain.num_timesteps
            target = original.rollout_buffer
            target.reset()
            for name in tensor_names[:9]:
                getattr(target, name)[:] = payload[name].cpu().numpy()
            target.tasks[:] = 'combat'
            target.allowed_control_masks[:] = module.COMBAT_MASK
            target.task, target.pos, target.full = 'combat', 512, True
            with self.assertRaisesRegex(ValueError, 'consumed'):
                original.set_task('combat', calibration=True)

            traces = [[], [], []]
            sample_original = target._get_samples
            def original_sample(indices, env=None):
                traces[0].append(np.array(indices, copy=True))
                return sample_original(indices, env)
            target._get_samples = original_sample
            restorations = []
            for model_index, model in enumerate(native_models, start=1):
                buffer = model.rollout_buffer
                sample = buffer._get_samples_from_device_indices
                def native_sample(indices, samples_cls, *, owner=model_index, original_sample=sample):
                    traces[owner].append(indices.detach().cpu().numpy().copy())
                    return original_sample(indices, samples_cls)
                buffer._get_samples_from_device_indices = native_sample
                restorations.append((buffer, sample))
            before = self.rng_state()
            try:
                original.train()
                after = self.rng_state()
                for model in native_models:
                    self.restore_rng(before)
                    model.train()
                    self.assert_tree_equal(after, self.rng_state())
                    self.assert_tree_equal(original.policy.state_dict(), model.policy.state_dict())
                    self.assert_tree_equal(original.policy.optimizer.state_dict(), model.policy.optimizer.state_dict())
                    self.assert_tree_equal(dict(original.logger.name_to_value), dict(model.logger.name_to_value))
                    self.assertTrue(model.rollout_buffer.consumed)
                    self.assertEqual(original._n_updates, model._n_updates)
                self.assert_tree_equal(traces[0], traces[1])
                self.assert_tree_equal(traces[0], traces[2])
                self.assertEqual(plain.last_update_diagnostics, deferred.last_update_diagnostics)
            finally:
                target._get_samples = sample_original
                for buffer, sample in restorations:
                    buffer._get_samples_from_device_indices = sample
            receipts.append({'index': index, 'transitions': 8192, 'native_game_ticks': 8192,
                             'rollout_sha256': hashlib.sha256(rollout_path.read_bytes()).hexdigest(),
                             'sampled_minibatches': len(traces[0]),
                             'update': plain.last_update_diagnostics,
                             'parameters_optimizer_rng_samples_metrics': 'exact'})

        exact_update(0, reference, natives)
        exact_update(1, reference, natives)
        paths = [artifacts/f'{name}.zip' for name in ('original', 'sb3', 'deferred_metrics')]
        for model, path in zip((reference, *natives), paths):
            model.save(path)
        restored_reference = original_model(paths[0])
        restored_natives = []
        for model, path in zip(natives, paths[1:]):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            env = environment()
            before = self.rng_state()
            restored = module.NativeCurriculumPPO.load(path, env=env, expected_sha256=digest)
            self.assert_tree_equal(before, self.rng_state())
            self.assert_tree_equal(model.policy.state_dict(), restored.policy.state_dict())
            # PyTorch may relocate optimizer step counters at load; their exact
            # values and all parameter moments must survive serialization.
            left, right = model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict()
            self.assert_tree_equal(left['param_groups'], right['param_groups'])
            self.assertEqual(left['state'].keys(), right['state'].keys())
            for key, values in left['state'].items():
                for name, value in values.items():
                    other = right['state'][key][name]
                    self.assert_tree_equal(value.cpu() if torch.is_tensor(value) else value,
                                           other.cpu() if torch.is_tensor(other) else other)
            self.assertIs(restored.policy.optimizer._policy_ref(), restored.policy)
            self.assertFalse(restored.calibration)
            restored_natives.append(attach_logger(restored))
        exact_update(2, restored_reference, restored_natives)
        print(json.dumps({'qualification': 'canonical-native-contact-update-v1',
                          'checkpoint_sha256': module.CALIBRATED_CHECKPOINT_SHA256,
                          'net_arch': reference.policy.net_arch, 'updates': receipts,
                          'artifacts': str(artifacts)}, sort_keys=True), flush=True)


if __name__ == '__main__':
    unittest.main()
