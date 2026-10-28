"""CUDA collector for the frozen native navigation/contact curriculum and policy.

The unchanged ``curriculum_ppo.py`` reference owns masking, the two critics and
protected Adam. This adapter only changes rollout storage/transport and offers
the source-bound deferred-metric update. Contact has true terminal endings;
navigation uses time-limit truncations. It deliberately cannot train on CPU.
"""
from __future__ import annotations

import copy
import hashlib
import numbers
import random
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch as th

import ai2.training.curriculum_ppo as curriculum_ppo
from ai2.training.curriculum_ppo import (
    ACTION_EXPERIMENT_INTERPRETATION,
    ACTION_HEADS,
    PPO_DEFAULTS,
    TASK_MASKS,
    CurriculumPolicy,
    CurriculumPPO,
    _apply_action_initialization,
    _copy_compatible_non_action_state,
    _ProtectedAdam,
    _resolve_action_arm,
    _set_neutral_action_head,
)
from ai2.training.ppo_update_optimizations import deferred_metrics_train, sb3_train_source_metadata
from ai2.training.tensor_ppo import TensorPPO, TensorVecEnv
from ai2.training.tensor_rollout_buffer import TensorRolloutBuffer

NAVIGATION_VERSION = 'bridge_0b_navigation-v1'
CONTACT_VERSION = 'bridge_b1_contact-v1'
TASK_IDS = {'navigation': 0, 'combat': 1}
DEFAULT_TASK_VERSIONS = {
    'navigation': NAVIGATION_VERSION,
    'combat': CONTACT_VERSION,
}
TASK_VERSIONS = dict(DEFAULT_TASK_VERSIONS)
COMBAT_MASK = TASK_MASKS['combat']
CURRICULUM_SOURCE_SHA256 = '380dcafe717366ea3c94fb82f028a329baab0db3b2833b5c9145218ef834eb40'
CALIBRATED_CHECKPOINT_SHA256 = '301ce3c72924f3e5c62f265f88292b8952e86e6d71d9a0552213cb8f3c6784b6'
CONTACT_LEARNING_RATES = (1e-4, 3e-5)


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


def _validate_contact_learning_rate(learning_rate):
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, numbers.Real):
        raise ValueError('Contact learning_rate must be 1e-4 or 3e-5')  # noqa: TRY004
    learning_rate = float(learning_rate)
    if learning_rate not in CONTACT_LEARNING_RATES:
        raise ValueError('Contact learning_rate must be 1e-4 or 3e-5')
    return learning_rate


def normalized_task_versions(expected_task_versions=None):
    versions = dict(DEFAULT_TASK_VERSIONS)
    if expected_task_versions:
        for task, version in dict(expected_task_versions).items():
            if task not in TASK_IDS:
                raise ValueError(f'Unsupported native curriculum task: {task!r}')
            if not isinstance(version, str) or not version:
                raise ValueError('Expected curriculum version values must be non-empty strings')
            versions[task] = version
    return versions


def _constant_schedule_value(value):
    return lambda _progress_remaining: value


def _apply_contact_learning_rate(model, learning_rate):
    learning_rate = _validate_contact_learning_rate(learning_rate)
    model.learning_rate = learning_rate
    model.lr_schedule = _constant_schedule_value(learning_rate)
    optimizer = getattr(getattr(model, 'policy', None), 'optimizer', None)
    if optimizer is not None:
        for group in optimizer.param_groups:
            if group.get('lr') != learning_rate:
                raise ValueError('Checkpoint optimizer learning_rate differs from source learning_rate')


def _source_learning_rate(source):
    learning_rate = _validate_contact_learning_rate(getattr(source, 'learning_rate', None))
    schedule = getattr(source, 'lr_schedule', None)
    if not callable(schedule):
        raise ValueError('Checkpoint lr_schedule must be callable and match learning_rate')  # noqa: TRY004
    for progress in (0.0, 0.5, 1.0):
        if schedule(progress) != learning_rate:
            raise ValueError('Checkpoint lr_schedule must be constant and match learning_rate')
    return learning_rate


def contact_settings(*, batch_size=8192, update_mode='deferred_metrics', learning_rate=1e-4):
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) or batch_size not in (8192, 16384):
        raise ValueError('Contact comparison batch_size must be 8192 or 16384')
    if update_mode not in ('sb3', 'deferred_metrics'):
        raise ValueError('update_mode must be sb3 or deferred_metrics')
    return {**PPO_DEFAULTS, 'learning_rate': _validate_contact_learning_rate(learning_rate),
            'batch_size': int(batch_size), 'ent_coef': .001,
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


def _canonical_task_mask(task):
    if task not in TASK_IDS:
        raise ValueError(f'Unsupported native curriculum task: {task!r}')
    return np.asarray(TASK_MASKS[task], dtype=bool)


def _mask_matches(actual, expected):
    is_tensor = getattr(th, 'is_tensor', lambda _value: False)
    if is_tensor(actual):
        if actual.dtype != th.bool:
            return False
        actual = actual.detach().cpu().numpy()
    actual = np.asarray(actual)
    return actual.dtype == np.dtype(bool) and np.array_equal(actual, expected)


def validate_task_environment(env, task, expected_task_versions=None):
    task_versions = normalized_task_versions(expected_task_versions)
    expected_mask = _canonical_task_mask(task)
    if getattr(env, 'curriculum_task', None) != task:
        raise ValueError(f'Environment must explicitly identify the {task} curriculum task')
    if getattr(env, 'curriculum_version', None) != task_versions[task]:
        raise ValueError(f'Environment must explicitly identify the qualified {task} curriculum version')
    if not _mask_matches(getattr(env, 'allowed_control_mask', None), expected_mask):
        raise ValueError(f'Environment allowed-control mask differs from the {task} curriculum')
    has_timeouts = getattr(env, 'host_has_timeouts', None)
    if has_timeouts is None:
        raise ValueError(f'{task} environment must declare its timeout behavior')
    timeout_value = has_timeouts()
    if not isinstance(timeout_value, (bool, np.bool_)):
        raise ValueError(f'{task} environment timeout declaration must be bool')  # noqa: TRY004
    if task == 'combat' and bool(timeout_value) is not False:
        raise ValueError('Contact environment must declare no time-limit truncations')
    if task == 'navigation' and bool(timeout_value) is not True:
        raise ValueError('Navigation environment must declare time-limit truncations')


def validate_contact_environment(env):
    validate_task_environment(env, 'combat')


def validate_episode_infos(infos, task, task_mask, expected_task_versions=None):
    task_versions = normalized_task_versions(expected_task_versions)
    for info in infos:
        if (
            info.get('curriculum_task') != task
            or info.get('curriculum_version') != task_versions[task]
            or not _mask_matches(info.get('allowed_control_mask'), task_mask)
        ):
            raise ValueError('Native episode records have a missing or wrong curriculum task/mask/version')


def _require_cuda(env, task='combat', expected_task_versions=None):
    if not th.cuda.is_available():
        raise ValueError('Native curriculum training requires CUDA; no CPU fallback')
    if not isinstance(env, TensorVecEnv) or env.device.type != 'cuda':
        raise ValueError('Native curriculum requires a CUDA TensorVecEnv')
    validate_task_environment(env.env, task, expected_task_versions)
    if tuple(env.observation_space.shape) != (256,):
        raise ValueError('Native curriculum requires a four-frame 256-value observation')


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
    """Homogeneous native-task rollout, including the actions actually executed.

    Labels and masks live beside each transition on CUDA. Numeric errors remain
    latched by TensorRolloutBuffer and are checked before any optimizer update.
    The additional full-array boundary checks also catch callback corruption.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_actions = th.empty_like(self.actions)
        self.task_ids = th.empty((self.buffer_size, self.n_envs), dtype=th.int8, device=self.device)
        self.allowed_control_masks = th.empty(self.actions.shape, dtype=th.bool, device=self.device)
        self._task_masks = {
            task: th.tensor(_canonical_task_mask(task), dtype=th.bool, device=self.device)
            for task in TASK_IDS
        }
        self.task, self.consumed = None, False
        self.action_head = 'gaussian'
        self.drained_episode_infos = []

    def reset(self):
        super().reset()
        self.task, self.consumed = None, False
        self.drained_episode_infos = []

    def add(self, *args, task, mask, executed_actions):
        expected_mask = _canonical_task_mask(task)
        if self.task is not None and self.task != task:
            raise ValueError('Cannot mix tasks within one rollout')
        if not _mask_matches(mask, expected_mask):
            raise ValueError('Wrong allowed-control mask for rollout task')
        executed = self._require_cuda_float32('executed_actions', executed_actions,
                                              (self.n_envs, self.action_dim))
        position = self.pos
        super().add(*args)
        self._accumulate_finite_check(executed)
        self.executed_actions[position].copy_(executed)
        self.task_ids[position].fill_(TASK_IDS[task])
        self.allowed_control_masks[position].copy_(self._task_masks[task])
        self.task = task

    def validate(self, task, mask):
        expected_mask = _canonical_task_mask(task)
        expected_task_id = TASK_IDS[task]
        expected_mask_tensor = self._task_masks[task]
        if not self.full or self.consumed:
            raise ValueError('A full, unconsumed rollout is required')
        if self.task != task or not bool((self.task_ids == expected_task_id).all()):
            raise ValueError('Wrong task or mixed task identities in rollout')
        if not _mask_matches(mask, expected_mask) or not bool((self.allowed_control_masks == expected_mask_tensor).all()):
            raise ValueError('Rollout mask differs from the current policy mask')
        if not self._computed_returns_and_advantage:
            raise ValueError('Rollout returns and advantages must be computed')
        self._validate_rollout_data()
        self._validate_training_targets()
        self._accumulate_finite_check(self.executed_actions)
        self._raise_if_invalid_data()
        if self.action_head == 'gaussian':
            expected = self.actions.clamp(0, 1)*expected_mask_tensor
        elif self.action_head == 'bernoulli_10':
            if not bool(((self.actions == 0) | (self.actions == 1)).all()):
                raise ValueError('Bernoulli rollout actions must be exact 0/1 buttons')
            expected = self.actions*expected_mask_tensor
        else:
            raise ValueError(f'Unsupported rollout action head: {self.action_head!r}')
        if not th.equal(self.executed_actions, expected):
            raise ValueError('Executed actions do not match the policy action-head decode')


class NativeCurriculumPPO(TensorPPO):
    """E1 contact learner with CUDA transport and unchanged curriculum policy."""
    def __init__(self, policy=CurriculumPolicy, env=None, *, batch_size=8192,
                 update_mode='deferred_metrics', calibration=False, arm='E1',
                 action_head='gaussian', action_init_strategy=None, action_ab_arm=None,
                 learning_rate=1e-4, task='combat', expected_task_versions=None, **kwargs):
        if task not in TASK_IDS:
            raise ValueError(f'Unsupported native curriculum task: {task!r}')
        self.task_versions = normalized_task_versions(expected_task_versions)
        _require_cuda(env, task, self.task_versions)
        _require_sources()
        settings = contact_settings(batch_size=batch_size, update_mode=update_mode,
                                    learning_rate=learning_rate)
        action_head, action_init_strategy, action_ab_arm = _resolve_action_arm(
            action_head=action_head, action_init_strategy=action_init_strategy, action_ab_arm=action_ab_arm,
        )
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
        if calibration and task != 'combat':
            raise ValueError('Light calibration requires the combat task')
        self.arm, self.task, self.calibration = arm, task, bool(calibration)
        self.action_head = action_head
        self.action_init_strategy = action_init_strategy
        self.action_ab_arm = action_ab_arm
        self.action_init_metadata = {
            'schema': 'ai2-action-ab-init-v1',
            'arm': action_ab_arm,
            'action_head': action_head,
            'strategy': action_init_strategy,
            'interpretation': ACTION_EXPERIMENT_INTERPRETATION.get(action_ab_arm),
            'source': 'native_constructor',
        }
        self.update_mode = update_mode
        self.parent_num_timesteps = 0
        self.optimizer_provenance = 'unchanged protected curriculum Adam'
        self.task_history = []
        self._task_live_state = {}
        self._collecting = False
        self.environment_state_provenance = (
            'Model, critic, optimizer, counters, and task history resume from checkpoint; '
            'live environment observations and episode-start flags are process-local and reset per immutable block.'
        )
        self.last_update_diagnostics = {}
        policy_kwargs = dict(kwargs.pop('policy_kwargs', {}) or {})
        if 'action_head' in policy_kwargs and policy_kwargs['action_head'] != action_head:
            raise ValueError('policy_kwargs action_head disagrees with learner action_head')
        policy_kwargs['action_head'] = action_head
        super().__init__(policy, env, **settings, policy_kwargs=policy_kwargs, **kwargs)
        self.policy.set_task(self.task, self.calibration)
        self.rollout_buffer_class = TaskTensorRolloutBuffer
        self.rollout_buffer = TaskTensorRolloutBuffer(self.n_steps, self.observation_space,
            self.action_space, device=self.device, gamma=self.gamma, gae_lambda=self.gae_lambda,
            n_envs=self.n_envs)
        self.rollout_buffer.action_head = action_head
        if action_init_strategy == 'neutral':
            _set_neutral_action_head(self.policy)

    @classmethod
    def from_curriculum(cls, source, env, *, batch_size=None, update_mode='deferred_metrics',
                        joint=False, action_head=None, action_init_strategy=None, action_ab_arm=None,
                        task=None, expected_task_versions=None):
        task_versions = normalized_task_versions(
            expected_task_versions if expected_task_versions is not None else getattr(source, 'task_versions', None)
        )
        source_task = getattr(source, 'task', None)
        if source_task not in TASK_IDS:
            raise ValueError('Expected an E1 native curriculum checkpoint')
        requested_task = task
        if requested_task is None:
            requested_task = getattr(getattr(env, 'env', None), 'curriculum_task', None)
        if requested_task not in TASK_IDS:
            raise ValueError('A supported native curriculum task is required for the next block')
        _require_cuda(env, requested_task, task_versions)
        _require_sources()
        if not isinstance(source, CurriculumPPO) or type(source.policy) is not CurriculumPolicy:
            raise ValueError('Expected the original CurriculumPPO checkpoint and specialized policy')
        if source.arm != 'E1':
            raise ValueError('Expected an E1 native curriculum checkpoint')
        if not isinstance(source.policy.optimizer, _ProtectedAdam) or source.policy.optimizer._policy_ref() is not source.policy:
            raise ValueError('Checkpoint optimizer is not the policy-owned protected Adam')
        source_learning_rate = _source_learning_rate(source)
        for name, expected in contact_settings(batch_size=source.batch_size,
                                               learning_rate=source_learning_rate).items():
            actual = getattr(source, name)
            if callable(actual):
                actual = actual(1.0)
            if actual != expected:
                raise ValueError(f'Checkpoint curriculum setting {name} differs: {actual!r}')
        preserve_source_action_contract = (
            action_head is None and action_init_strategy is None and action_ab_arm is None
        )
        source_action_head = getattr(source, 'action_head', source.policy.action_head)
        source_action_init = getattr(source, 'action_init_strategy', 'keep')
        source_action_arm = getattr(source, 'action_ab_arm', None)
        if preserve_source_action_contract:
            constructor_action_head = source_action_head
            constructor_action_init = 'keep'
            constructor_action_arm = None
            actual_action_head = source_action_head
            actual_action_init = source_action_init
            actual_action_arm = source_action_arm
        else:
            constructor_action_head, constructor_action_init, constructor_action_arm = _resolve_action_arm(
                action_head=action_head, action_init_strategy=action_init_strategy, action_ab_arm=action_ab_arm,
            )
            actual_action_head = constructor_action_head
            actual_action_init = constructor_action_init
            actual_action_arm = constructor_action_arm
        with _preserve_rng():
            policy_kwargs = copy.deepcopy(source.policy_kwargs)
            policy_kwargs['action_head'] = constructor_action_head
            requested_calibration = source.calibration if requested_task == 'combat' else False
            model = cls(env=env, batch_size=source.batch_size if batch_size is None else batch_size,
                        update_mode=update_mode, calibration=requested_calibration, arm=source.arm,
                        action_head=constructor_action_head, action_init_strategy=constructor_action_init,
                        action_ab_arm=constructor_action_arm,
                        learning_rate=source_learning_rate,
                        task=requested_task,
                        expected_task_versions=task_versions,
                        # The immutable scenario schedule belongs to the caller.
                        # SB3 seeds its env in __init__ when seed is not None.
                        policy_kwargs=policy_kwargs, seed=None,
                        verbose=source.verbose, device=env.device,
                        stats_window_size=source._stats_window_size, tensorboard_log=source.tensorboard_log)
            if constructor_action_init == 'keep':
                model.set_parameters(source.get_parameters(), exact_match=True, device=env.device)
                if preserve_source_action_contract and isinstance(getattr(source, 'action_init_metadata', None), dict):
                    model.action_init_metadata = copy.deepcopy(source.action_init_metadata)
                    model.action_init_metadata.update({
                        'schema': 'ai2-action-ab-init-v1',
                        'arm': actual_action_arm,
                        'action_head': actual_action_head,
                        'strategy': actual_action_init,
                        'interpretation': ACTION_EXPERIMENT_INTERPRETATION.get(actual_action_arm),
                        'source': model.action_init_metadata.get('source', 'curriculum_checkpoint_resume'),
                        'resume_tensor_copy': 'source policy/action tensors and optimizer state copied',
                    })
                else:
                    model.action_init_metadata = {
                        'schema': 'ai2-action-ab-init-v1',
                        'arm': actual_action_arm,
                        'action_head': actual_action_head,
                        'strategy': actual_action_init,
                        'interpretation': ACTION_EXPERIMENT_INTERPRETATION.get(actual_action_arm),
                        'source': 'curriculum_checkpoint',
                        'source_action_head': source.policy.action_head,
                        'tensor_copy': 'source policy/action tensors copied',
                    }
                model.optimizer_provenance = (
                    'source policy/action tensors copied; source optimizer state copied'
                )
            else:
                _copy_compatible_non_action_state(model.policy, source.policy)
                init_metadata = _apply_action_initialization(
                    model.policy, source.policy, action_init_strategy=constructor_action_init,
                )
                model.action_init_metadata = {
                    'schema': 'ai2-action-ab-init-v1',
                    'arm': actual_action_arm,
                    'action_head': actual_action_head,
                    'strategy': actual_action_init,
                    'interpretation': ACTION_EXPERIMENT_INTERPRETATION.get(actual_action_arm),
                    'source': 'curriculum_checkpoint',
                    'source_action_head': source.policy.action_head,
                    **init_metadata,
                }
                model.optimizer_provenance = (
                    f'fresh {constructor_action_init} {constructor_action_head} action head; source trunk/value copied; '
                    'fresh Adam optimizer not transferred'
                )
        for name in ('num_timesteps', '_n_updates', '_episode_num', '_total_timesteps',
                     '_num_timesteps_at_start', '_current_progress_remaining', 'parent_num_timesteps',
                     'task_history', 'seed'):
            if hasattr(source, name):
                setattr(model, name, copy.deepcopy(getattr(source, name)))
        model.task = requested_task
        model.calibration = requested_calibration
        model.action_head = actual_action_head
        model.action_init_strategy = actual_action_init
        model.action_ab_arm = actual_action_arm
        # set_parameters() restores the checkpoint's saved task mask buffer.
        # Reapply the caller's requested task without recording a live task switch.
        model.policy.set_task(requested_task, requested_calibration)
        _apply_contact_learning_rate(model, source_learning_rate)
        for name in ('ep_info_buffer', 'ep_success_buffer'):
            value = getattr(source, name, None)
            if value is not None:
                setattr(model, name, deque(value, maxlen=value.maxlen))
        if joint:
            model.set_task(requested_task, calibration=False)
        return model

    @classmethod
    def load_contact_checkpoint(cls, path, env=None, *, expected_sha256=CALIBRATED_CHECKPOINT_SHA256,
                                batch_size=8192, update_mode='deferred_metrics', joint=True,
                                action_head=None, action_init_strategy=None, action_ab_arm=None,
                                expected_task_versions=None):
        task_versions = normalized_task_versions(expected_task_versions)
        _require_cuda(env, 'combat', task_versions)
        _require_sources()
        digest = verify_checkpoint(path, expected_sha256)
        with _preserve_rng():
            source = CurriculumPPO.load(path, device=env.device)
            model = cls.from_curriculum(source, env, batch_size=batch_size, update_mode=update_mode,
                                        joint=joint, action_head=action_head,
                                        action_init_strategy=action_init_strategy,
                                        action_ab_arm=action_ab_arm,
                                        expected_task_versions=task_versions)
        model.starting_checkpoint_sha256 = digest
        return model

    @classmethod
    def load(cls, path, env=None, device='auto', *, expected_sha256, update_mode=None,
             expected_task_versions=None, **kwargs):
        task = getattr(getattr(env, 'env', None), 'curriculum_task', 'combat')
        _require_sources()
        action_head = kwargs.pop('action_head', None)
        action_init_strategy = kwargs.pop('action_init_strategy', None)
        action_ab_arm = kwargs.pop('action_ab_arm', None)
        if kwargs or (device != 'auto' and th.device(device) not in (env.device, th.device('cuda'))):
            raise ValueError('Native contact load does not accept policy or hyperparameter overrides')
        verify_checkpoint(path, expected_sha256)
        with _preserve_rng():
            source = CurriculumPPO.load(path, device=env.device)
            task_versions = normalized_task_versions(
                expected_task_versions if expected_task_versions is not None else getattr(source, 'task_versions', None)
            )
            _require_cuda(env, task, task_versions)
            model = cls.from_curriculum(source, env,
                update_mode=update_mode or getattr(source, 'update_mode', 'deferred_metrics'),
                action_head=action_head, action_init_strategy=action_init_strategy,
                action_ab_arm=action_ab_arm,
                expected_task_versions=task_versions)
        model.starting_checkpoint_sha256 = getattr(source, 'starting_checkpoint_sha256', expected_sha256)
        model.last_update_diagnostics = copy.deepcopy(getattr(source, 'last_update_diagnostics', {}))
        return model

    def _excluded_save_params(self):
        excluded = super()._excluded_save_params()
        return excluded + ['_task_live_state', '_task_env_generations', '_next_task_env_generation',
                           '_current_env_live_key', '_collecting']

    def _policy_contract(self):
        if self.policy.task != self.task or self.policy.calibration != self.calibration:
            raise ValueError('Policy task/calibration differs from learner task')
        if self.policy.action_head != self.action_head:
            raise ValueError('Policy action head differs from learner action head')
        expected_mask = _canonical_task_mask(self.task)
        if not np.array_equal(self.policy.allowed_control_mask.detach().cpu().numpy(), expected_mask):
            raise ValueError(f'Policy mask differs from the canonical {self.task} task mask')
        if not isinstance(self.policy.optimizer, _ProtectedAdam) or self.policy.optimizer._policy_ref() is not self.policy:
            raise ValueError('Protected optimizer must reference this policy')

    def set_task(self, task, calibration=False):
        if task not in TASK_IDS:
            raise ValueError(f'Unsupported native curriculum task: {task!r}')
        if task != self.task:
            raise ValueError('Use set_task_environment() to change native curriculum tasks')
        buffer = self.rollout_buffer
        if self._collecting or ((buffer.pos or buffer.full) and not buffer.consumed):
            raise ValueError('Cannot change task or mask before the current rollout is consumed')
        self.policy.set_task(task, calibration)
        self.task, self.calibration = task, bool(calibration)
        self.ent_coef = .001
        self.task_history.append({'task': task, 'calibration': calibration, 'num_timesteps': self.num_timesteps})

    def _same_tensor_abi(self, tensor_env):
        if not isinstance(tensor_env, TensorVecEnv):
            raise ValueError('Task environment must be a CUDA TensorVecEnv')  # noqa: TRY004
        if tensor_env.device != self.device:
            raise ValueError('Task environment CUDA device must match the learner device')
        if tensor_env.num_envs != self.n_envs:
            raise ValueError('Task environment lane count must match the learner')
        for name in ('observation_space', 'action_space'):
            current = getattr(self, name)
            candidate = getattr(tensor_env, name)
            if tuple(current.shape) != tuple(candidate.shape):
                raise ValueError(f'Task environment {name} shape must match the learner')
            if np.dtype(current.dtype) != np.dtype(candidate.dtype):
                raise ValueError(f'Task environment {name} dtype must match the learner')
            for bound in ('low', 'high'):
                if (
                    hasattr(current, bound)
                    and hasattr(candidate, bound)
                    and not np.array_equal(getattr(current, bound), getattr(candidate, bound))
                ):
                    raise ValueError(f'Task environment {name} {bound} bound must match the learner')

    def _save_live_task_state(self):
        if not hasattr(self, '_task_live_state'):
            self._task_live_state = {}
        if getattr(self, '_last_obs', None) is None:
            return
        key = self._env_live_key(self.task, self.tensor_vec_env)
        self._current_env_live_key = key
        self._task_live_state[key] = (
            self.tensor_vec_env,
            self._last_obs.detach().clone(),
            self._last_episode_starts.detach().clone(),
        )

    def _ensure_env_generation(self, tensor_env):
        if not hasattr(self, '_task_env_generations'):
            self._task_env_generations = {}
        if not hasattr(self, '_next_task_env_generation'):
            self._next_task_env_generation = 0
        env_id = id(tensor_env)
        entry = self._task_env_generations.get(env_id)
        if entry is not None and entry[0] is tensor_env:
            return entry[1]
        generation = self._next_task_env_generation
        self._next_task_env_generation += 1
        self._task_env_generations[env_id] = (tensor_env, generation)
        return generation

    def _env_live_key(self, task, tensor_env):
        return (task, id(tensor_env), self._ensure_env_generation(tensor_env))

    def _restore_or_reset_task_state(self, task):
        if not hasattr(self, '_task_live_state'):
            self._task_live_state = {}
        key = self._env_live_key(task, self.tensor_vec_env)
        self._current_env_live_key = key
        state = self._task_live_state.get(key)
        if state is not None and state[0] is self.tensor_vec_env:
            self._last_obs = state[1].detach().clone()
            self._last_episode_starts = state[2].detach().clone()
        else:
            self._last_obs = self.tensor_vec_env.reset_tensor().detach().clone()
            self._last_episode_starts = th.ones((self.tensor_vec_env.num_envs,), dtype=th.float32, device=self.device)

    def set_task_environment(self, task, tensor_env, *, calibration=False):
        if not hasattr(self, '_task_live_state'):
            self._task_live_state = {}
        if task not in TASK_IDS:
            raise ValueError(f'Unsupported native curriculum task: {task!r}')
        if calibration and task != 'combat':
            raise ValueError('Light calibration requires the combat task')
        buffer = self.rollout_buffer
        if self._collecting:
            raise ValueError('Cannot switch task environment during collection')
        if (buffer.pos or buffer.full) and not buffer.consumed:
            raise ValueError('Cannot switch task environment before the current rollout is consumed')
        _require_cuda(tensor_env, task, self.task_versions)
        self._same_tensor_abi(tensor_env)
        self._save_live_task_state()
        self.tensor_vec_env = tensor_env
        self.env = tensor_env.as_sb3_vec_env()
        self.policy.set_task(task, calibration)
        self.task, self.calibration = task, bool(calibration)
        self._restore_or_reset_task_state(task)
        self.task_history.append({'task': task, 'calibration': calibration, 'num_timesteps': self.num_timesteps})

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        del env
        if self._last_obs is None:
            raise ValueError('Call learn/_setup_learn before collection')
        if not isinstance(rollout_buffer, TaskTensorRolloutBuffer) or n_rollout_steps != rollout_buffer.buffer_size:
            raise ValueError('Collection requires one full curriculum rollout')
        if (rollout_buffer.pos or rollout_buffer.full) and not rollout_buffer.consumed:
            raise ValueError('Previous rollout has not been consumed')
        self._policy_contract()
        validate_task_environment(self.tensor_vec_env.env, self.task, self.task_versions)
        task, calibration = self.task, self.calibration
        task_mask = _canonical_task_mask(task)
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
                validate_task_environment(self.tensor_vec_env.env, task, self.task_versions)
                observations = self._last_obs.clone()
                with th.no_grad():
                    actions, values, log_probs = self.policy(observations)
                raw_actions = actions.reshape(self.n_envs, 10)
                if not (th.isfinite(raw_actions).all()
                        and th.isfinite(values).all()
                        and th.isfinite(log_probs).all()):
                    raise ValueError('Policy actions, values and log probabilities must be finite')
                executed_actions = self.policy.decode_actions(raw_actions)
                if not th.isfinite(executed_actions).all():
                    raise ValueError('Executed actions must be finite')
                step = self.tensor_vec_env.step_tensor(executed_actions)
                rewards = step.rewards
                if task == 'combat':
                    # Contact's deadline is terminal, never a bootstrap truncation.
                    rollout_buffer._invalid_data.logical_or_(step.truncated.any())
                else:
                    timeout_mask = step.truncated & ~step.terminated
                    if bool(timeout_mask.any()):
                        with th.no_grad():
                            terminal_values = self.policy.predict_values(step.terminal_observations).flatten()
                        rewards = th.where(timeout_mask, rewards + self.gamma*terminal_values, rewards)
                dones = step.terminated | step.truncated
                self.num_timesteps += self.n_envs
                callback.update_locals(locals())
                if not callback.on_step():
                    return False
                rollout_buffer.add(observations, raw_actions, rewards, self._last_episode_starts,
                                   values, log_probs, task=task, mask=task_mask,
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
            validate_episode_infos(infos, task, task_mask, self.task_versions)
            rollout_buffer.drained_episode_infos = copy.deepcopy(infos)
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
        task_mask = _canonical_task_mask(self.task)
        self.rollout_buffer.validate(self.task, task_mask)
        validate_episode_infos(
            getattr(self.rollout_buffer, 'drained_episode_infos', []),
            self.task,
            task_mask,
            self.task_versions,
        )
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
        normalizer = getattr(self.policy, 'normalized_entropy', None)
        normalized_entropy = None if normalizer is None else normalizer(
            th.as_tensor(entropy, device=getattr(self, 'device', 'cpu')))
        self.logger.record('train/active_entropy_normalized',
                           -1.0 if normalized_entropy is None else float(normalized_entropy.item()))
        action_head = getattr(self.policy, 'action_head', 'gaussian')
        self.logger.record('train/action_head_id', ACTION_HEADS.index(action_head) if action_head in ACTION_HEADS else -1)
        self.logger.record('train/action_entropy_scale_id', 0 if normalized_entropy is None else 1)
        self.logger.record('train/active_entropy_coefficient', self.ent_coef)

    export_actor = CurriculumPPO.export_actor
