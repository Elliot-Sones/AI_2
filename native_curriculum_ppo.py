"""CUDA collector for the frozen B1 contact curriculum and its real policy.

The unchanged ``curriculum_ppo.py`` reference owns masking, the two critics and
protected Adam. This adapter only changes rollout storage/transport and offers
the source-bound deferred-metric update. Contact has true terminal endings, not
time-limit truncations. It deliberately cannot train a different task or on CPU.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import copy
import hashlib
import numbers
from pathlib import Path
import random

import numpy as np
import torch as th

import curriculum_ppo
from curriculum_ppo import CurriculumPPO, CurriculumPolicy, PPO_DEFAULTS, TASK_MASKS, _ProtectedAdam
from ppo_update_optimizations import deferred_metrics_train, sb3_train_source_metadata
from tensor_ppo import TensorPPO, TensorVecEnv
from tensor_rollout_buffer import TensorRolloutBuffer

CONTACT_VERSION = 'bridge_b1_contact-v1'
COMBAT_MASK = TASK_MASKS['combat']
CURRICULUM_SOURCE_SHA256 = '991cda3d086e47eb8d6fddf6cbc0f870aa002e7632e7c93daf6dd2d56e20fe6b'
CALIBRATED_CHECKPOINT_SHA256 = '301ce3c72924f3e5c62f265f88292b8952e86e6d71d9a0552213cb8f3c6784b6'


def curriculum_source_metadata():
    path = Path(curriculum_ppo.__file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'path': str(path), 'sha256': digest, 'supported': digest == CURRICULUM_SOURCE_SHA256}


def verify_checkpoint(path, expected_sha256):
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError('An explicit checkpoint SHA256 is required')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    result = digest.hexdigest()
    if result != expected_sha256:
        raise ValueError(f'Checkpoint SHA256 mismatch: expected {expected_sha256}, got {result}')
    return result


def contact_settings(*, batch_size=8192, update_mode='deferred_metrics'):
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) or batch_size not in (8192, 16384):
        raise ValueError('Contact comparison batch_size must be 8192 or 16384')
    if update_mode not in ('sb3', 'deferred_metrics'):
        raise ValueError('update_mode must be sb3 or deferred_metrics')
    return {**PPO_DEFAULTS, 'batch_size': int(batch_size), 'ent_coef': .001,
            'normalize_advantage': True, 'clip_range_vf': None, 'use_sde': False}


def curriculum_explained_variance(values, returns):
    """Reproduce SB3's post-sampling env-major NumPy logging reduction.

    SB3 mutates its stored arrays to env-major when get() first samples them;
    TensorRolloutBuffer keeps time-major storage. The order matters even for a
    scalar variance because float32 reductions are not associative.
    """
    from stable_baselines3.common.utils import explained_variance
    arrays = [value.detach().cpu().numpy() if th.is_tensor(value) else np.asarray(value)
              for value in (values, returns)]
    if len(arrays[0].shape) != 2 or arrays[0].shape != arrays[1].shape:
        raise ValueError('Curriculum logging requires matching time-major [steps, lanes] arrays')
    return explained_variance(*(value.swapaxes(0, 1).flatten() for value in arrays))


def validate_contact_environment(env):
    if getattr(env, 'curriculum_task', None) != 'combat':
        raise ValueError('Environment must explicitly identify the combat curriculum task')
    if getattr(env, 'curriculum_version', None) != CONTACT_VERSION:
        raise ValueError('Environment must explicitly identify the qualified contact curriculum version')
    if not np.array_equal(getattr(env, 'allowed_control_mask', None), COMBAT_MASK):
        raise ValueError('Environment allowed-control mask differs from the contact curriculum')
    has_timeouts = getattr(env, 'host_has_timeouts', None)
    if has_timeouts is None or has_timeouts() is not False:
        raise ValueError('Contact environment must declare no time-limit truncations')


def _require_cuda(env):
    if not th.cuda.is_available():
        raise ValueError('Native curriculum training requires CUDA; no CPU fallback')
    if not isinstance(env, TensorVecEnv) or env.device.type != 'cuda':
        raise ValueError('Native curriculum requires a CUDA TensorVecEnv')
    validate_contact_environment(env.env)
    if tuple(env.observation_space.shape) != (256,):
        raise ValueError('Contact curriculum requires a four-frame 256-value observation')


def _require_sources():
    if not curriculum_source_metadata()['supported']:
        raise ValueError('Frozen curriculum policy source does not match its qualified SHA256')
    metadata = sb3_train_source_metadata()
    if not metadata['supported']:
        raise ValueError(f'Native curriculum requires the pinned SB3 train source: {metadata}')


@contextmanager
def _preserve_rng():
    """Checkpoint initialization must not reseed the caller's experiment stream."""
    state = (random.getstate(), np.random.get_state(), th.get_rng_state(), th.cuda.get_rng_state_all())
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        th.set_rng_state(state[2])
        th.cuda.set_rng_state_all(state[3])


class TaskTensorRolloutBuffer(TensorRolloutBuffer):
    """Homogeneous contact rollout, including the actions actually executed.

    Labels and masks live beside each transition on CUDA. Numeric errors remain
    latched by TensorRolloutBuffer and are checked before any optimizer update.
    The additional full-array boundary checks also catch callback corruption.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_actions = th.empty_like(self.actions)
        self.task_ids = th.empty((self.buffer_size, self.n_envs), dtype=th.int8, device=self.device)
        self.allowed_control_masks = th.empty(self.actions.shape, dtype=th.bool, device=self.device)
        self._contact_mask = th.tensor(COMBAT_MASK, dtype=th.bool, device=self.device)
        self.task, self.consumed = None, False

    def reset(self):
        super().reset()
        self.task, self.consumed = None, False

    def add(self, *args, task, mask, executed_actions):
        if task != 'combat' or (self.task is not None and self.task != task):
            raise ValueError('Cannot mix tasks within a contact rollout')
        if not np.array_equal(mask, COMBAT_MASK):
            raise ValueError('Wrong allowed-control mask for rollout task')
        executed = self._require_cuda_float32('executed_actions', executed_actions,
                                              (self.n_envs, self.action_dim))
        position = self.pos
        super().add(*args)
        self._accumulate_finite_check(executed)
        self.executed_actions[position].copy_(executed)
        self.task_ids[position].fill_(1)
        self.allowed_control_masks[position].copy_(self._contact_mask)
        self.task = task

    def validate(self, task, mask):
        if not self.full or self.consumed:
            raise ValueError('A full, unconsumed rollout is required')
        if task != 'combat' or self.task != task or not bool((self.task_ids == 1).all()):
            raise ValueError('Wrong task or mixed task identities in rollout')
        if not np.array_equal(mask, COMBAT_MASK) or not bool((self.allowed_control_masks == self._contact_mask).all()):
            raise ValueError('Rollout mask differs from the current policy mask')
        if not self._computed_returns_and_advantage:
            raise ValueError('Rollout returns and advantages must be computed')
        self._validate_rollout_data()
        self._validate_training_targets()
        self._accumulate_finite_check(self.executed_actions)
        self._raise_if_invalid_data()
        if not th.equal(self.executed_actions, self.actions.clamp(0, 1)*self._contact_mask):
            raise ValueError('Executed actions do not match the clipped masked raw samples')


class NativeCurriculumPPO(TensorPPO):
    """E1 contact learner with CUDA transport and unchanged curriculum policy."""
    def __init__(self, policy=CurriculumPolicy, env=None, *, batch_size=8192,
                 update_mode='deferred_metrics', calibration=False, arm='E1', **kwargs):
        _require_cuda(env)
        _require_sources()
        settings = contact_settings(batch_size=batch_size, update_mode=update_mode)
        if arm != 'E1' or policy is not CurriculumPolicy:
            raise ValueError('This experiment requires the E1 CurriculumPolicy')
        # Do not silently accept ordinary raw-combat PPO settings in this run.
        for name, value in settings.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f'Contact curriculum setting {name} must remain {value!r}')
            kwargs.pop(name, None)
        forbidden = {'rollout_buffer_class', 'rollout_buffer_kwargs', 'defer_timeout_values',
                     'terminal_value_batch_size'} & kwargs.keys()
        if forbidden:
            raise ValueError(f'Unsupported contact collector overrides: {sorted(forbidden)}')
        if settings['n_steps']*env.num_envs < batch_size or settings['n_steps']*env.num_envs % batch_size:
            raise ValueError('Contact rollout size must be a positive multiple of minibatch size')
        self.arm, self.task, self.calibration = arm, 'combat', bool(calibration)
        self.update_mode = update_mode
        self.parent_num_timesteps = 0
        self.optimizer_provenance = 'unchanged protected curriculum Adam'
        self.task_history = []
        self._collecting = False
        self.last_update_diagnostics = {}
        super().__init__(policy, env, **settings, **kwargs)
        self.policy.set_task(self.task, self.calibration)
        self.rollout_buffer_class = TaskTensorRolloutBuffer
        self.rollout_buffer = TaskTensorRolloutBuffer(self.n_steps, self.observation_space,
            self.action_space, device=self.device, gamma=self.gamma, gae_lambda=self.gae_lambda,
            n_envs=self.n_envs)

    @classmethod
    def from_curriculum(cls, source, env, *, batch_size=None, update_mode='deferred_metrics', joint=False):
        _require_cuda(env)
        _require_sources()
        if not isinstance(source, CurriculumPPO) or type(source.policy) is not CurriculumPolicy:
            raise ValueError('Expected the original CurriculumPPO checkpoint and specialized policy')
        if source.arm != 'E1' or source.task != 'combat':
            raise ValueError('Expected an E1 contact curriculum checkpoint')
        if not isinstance(source.policy.optimizer, _ProtectedAdam) or source.policy.optimizer._policy_ref() is not source.policy:
            raise ValueError('Checkpoint optimizer is not the policy-owned protected Adam')
        for name, expected in contact_settings(batch_size=source.batch_size).items():
            actual = getattr(source, name)
            if callable(actual):
                actual = actual(1.0)
            if actual != expected:
                raise ValueError(f'Checkpoint curriculum setting {name} differs: {actual!r}')
        with _preserve_rng():
            model = cls(env=env, batch_size=source.batch_size if batch_size is None else batch_size,
                        update_mode=update_mode, calibration=source.calibration, arm=source.arm,
                        # The immutable scenario schedule belongs to the caller.
                        # SB3 seeds its env in __init__ when seed is not None.
                        policy_kwargs=copy.deepcopy(source.policy_kwargs), seed=None,
                        verbose=source.verbose, device=env.device,
                        stats_window_size=source._stats_window_size, tensorboard_log=source.tensorboard_log)
            model.set_parameters(source.get_parameters(), exact_match=True, device=env.device)
        for name in ('num_timesteps', '_n_updates', '_episode_num', '_total_timesteps',
                     '_num_timesteps_at_start', '_current_progress_remaining', 'parent_num_timesteps',
                     'optimizer_provenance', 'task_history', 'seed'):
            if hasattr(source, name):
                setattr(model, name, copy.deepcopy(getattr(source, name)))
        for name in ('ep_info_buffer', 'ep_success_buffer'):
            value = getattr(source, name, None)
            if value is not None:
                setattr(model, name, deque(value, maxlen=value.maxlen))
        if joint:
            model.set_task('combat', calibration=False)
        return model

    @classmethod
    def load_contact_checkpoint(cls, path, env=None, *, expected_sha256=CALIBRATED_CHECKPOINT_SHA256,
                                batch_size=8192, update_mode='deferred_metrics', joint=True):
        _require_cuda(env)
        _require_sources()
        digest = verify_checkpoint(path, expected_sha256)
        with _preserve_rng():
            source = CurriculumPPO.load(path, device=env.device)
            model = cls.from_curriculum(source, env, batch_size=batch_size, update_mode=update_mode, joint=joint)
        model.starting_checkpoint_sha256 = digest
        return model

    @classmethod
    def load(cls, path, env=None, device='auto', *, expected_sha256, update_mode=None, **kwargs):
        _require_cuda(env)
        _require_sources()
        if kwargs or (device != 'auto' and th.device(device) not in (env.device, th.device('cuda'))):
            raise ValueError('Native contact load does not accept policy or hyperparameter overrides')
        verify_checkpoint(path, expected_sha256)
        with _preserve_rng():
            source = CurriculumPPO.load(path, device=env.device)
            model = cls.from_curriculum(source, env,
                update_mode=update_mode or getattr(source, 'update_mode', 'deferred_metrics'))
        model.starting_checkpoint_sha256 = getattr(source, 'starting_checkpoint_sha256', expected_sha256)
        model.last_update_diagnostics = copy.deepcopy(getattr(source, 'last_update_diagnostics', {}))
        return model

    def _policy_contract(self):
        if self.policy.task != self.task or self.policy.calibration != self.calibration:
            raise ValueError('Policy task/calibration differs from learner task')
        if self.task != 'combat' or not np.array_equal(self.policy.allowed_control_mask.cpu(), COMBAT_MASK):
            raise ValueError('Policy mask differs from the canonical contact task mask')
        if not isinstance(self.policy.optimizer, _ProtectedAdam) or self.policy.optimizer._policy_ref() is not self.policy:
            raise ValueError('Protected optimizer must reference this policy')

    def set_task(self, task, calibration=False):
        if task != 'combat':
            raise ValueError('Native contact adapter cannot train a different curriculum task')
        buffer = self.rollout_buffer
        if self._collecting or (buffer.pos and not buffer.consumed):
            raise ValueError('Cannot change task or mask before the current rollout is consumed')
        self.policy.set_task(task, calibration)
        self.task, self.calibration = task, bool(calibration)
        self.ent_coef = .001
        self.task_history.append(dict(task=task, calibration=calibration, num_timesteps=self.num_timesteps))

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        del env
        if self._last_obs is None:
            raise ValueError('Call learn/_setup_learn before collection')
        if not isinstance(rollout_buffer, TaskTensorRolloutBuffer) or n_rollout_steps != rollout_buffer.buffer_size:
            raise ValueError('Collection requires one full curriculum rollout')
        if rollout_buffer.pos and not rollout_buffer.consumed:
            raise ValueError('Previous rollout has not been consumed')
        self._policy_contract()
        validate_contact_environment(self.tensor_vec_env.env)
        task, calibration = self.task, self.calibration
        mask_version = self.policy.allowed_control_mask._version
        action_distribution = self.policy.action_dist
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        self._collecting = True
        try:
            callback.on_rollout_start()
            for _ in range(n_rollout_steps):
                if (self.task != task or self.policy.task != task or self.policy.calibration != calibration
                        or self.calibration != calibration or self.policy.allowed_control_mask._version != mask_version
                        or self.policy.action_dist is not action_distribution):
                    raise ValueError('Task/mask changed during rollout')
                validate_contact_environment(self.tensor_vec_env.env)
                observations = self._last_obs.clone()
                with th.no_grad():
                    actions, values, log_probs = self.policy(observations)
                raw_actions = actions.reshape(self.n_envs, 10)
                executed_actions = th.clamp(raw_actions, self._action_low_tensor,
                                            self._action_high_tensor)*self.policy.allowed_control_mask
                step = self.tensor_vec_env.step_tensor(executed_actions)
                # Contact's deadline is terminal, never a bootstrap truncation.
                rollout_buffer._invalid_data.logical_or_(step.truncated.any())
                dones = step.terminated | step.truncated
                self.num_timesteps += self.n_envs
                callback.update_locals(locals())
                if not callback.on_step():
                    return False
                rollout_buffer.add(observations, raw_actions, step.rewards, self._last_episode_starts,
                                   values, log_probs, task=task, mask=COMBAT_MASK,
                                   executed_actions=executed_actions)
                self._last_obs = step.observations.clone()
                self._last_episode_starts = dones.to(dtype=th.float32)
            self._policy_contract()
            if self.policy.allowed_control_mask._version != mask_version:
                raise ValueError('Task/mask changed during rollout')
            with th.no_grad():
                last_values = self.policy.predict_values(self._last_obs)
            rollout_buffer.compute_returns_and_advantage(last_values, dones.to(dtype=th.float32))
            # The native backend validates lane errors and overflow in this drain.
            infos = self.tensor_vec_env.drain_episode_infos()
            for info in infos:
                if (info.get('curriculum_task') != task or info.get('curriculum_version') != CONTACT_VERSION
                        or not np.array_equal(info.get('allowed_control_mask'), COMBAT_MASK)):
                    raise ValueError('Native episode records have a missing or wrong curriculum task/mask/version')
            if infos:
                self._ensure_info_buffers()
                self._update_info_buffer(infos, np.ones(len(infos), dtype=bool))
            callback.update_locals(locals())
            callback.on_rollout_end()
            return True
        finally:
            self._collecting = False

    def train(self):
        self._policy_contract()
        self.rollout_buffer.validate(self.task, COMBAT_MASK)
        updates_before = self._n_updates
        optimizer_steps = 0
        optimizer = self.policy.optimizer
        had_instance_step = 'step' in optimizer.__dict__
        previous_instance_step = optimizer.__dict__.get('step')
        original_step = optimizer.step
        def count_step(*args, **kwargs):
            nonlocal optimizer_steps
            result = original_step(*args, **kwargs)
            optimizer_steps += 1
            return result
        # _ProtectedAdam.step calls Adam.step; PyTorch wraps both methods and
        # dispatches optimizer hooks twice. Count completed outer invocations,
        # preserving the original optimizer and all of its existing hooks.
        optimizer.step = count_step
        try:
            if self.update_mode == 'deferred_metrics':
                deferred_metrics_train(self)
            elif self.update_mode == 'sb3':
                super().train()
            else:
                raise ValueError('Unsupported native curriculum update mode')
        finally:
            if had_instance_step:
                optimizer.step = previous_instance_step
            else:
                del optimizer.step
        self.rollout_buffer.consumed = True
        self.last_update_diagnostics = {
            'attempted_epochs': self._n_updates-updates_before,
            'optimizer_steps': optimizer_steps,
            'kl_early_stop': optimizer_steps < self.n_epochs*(self.n_steps*self.n_envs//self.batch_size),
        }
        self.logger.record('train/explained_variance', curriculum_explained_variance(
            self.rollout_buffer.values, self.rollout_buffer.returns))
        self.logger.record(f'train/{self.task}/value_loss', self.logger.name_to_value['train/value_loss'])
        self.logger.record(f'train/{self.task}/explained_variance', self.logger.name_to_value['train/explained_variance'])
        entropy = -float(self.logger.name_to_value['train/entropy_loss'])
        self.logger.record('train/active_entropy_sum', entropy)
        self.logger.record('train/active_entropy_mean', entropy/len(self.policy.active_indices))
        self.logger.record('train/active_entropy_coefficient', self.ent_coef)

    export_actor = CurriculumPPO.export_actor
