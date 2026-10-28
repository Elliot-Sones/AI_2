"""Frozen navigation maintenance and explicitly versioned retention metrics.

The reported session11 development panels used 0a=1.5..3 and 0b=2..6
world units. They used player zero, not a balanced player-slot panel. This
module keeps that factory and completion predicate as its default; slot-one
panels and body-overlap panels must be reported separately.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import sys
from types import MethodType

import gymnasium as gym
import numpy as np


NAVIGATION_ACTIVE_INDICES = (0, 1, 2, 3, 4, 6)
HISTORICAL_METRIC = 'active_sequence_v3'
BODY_OVERLAP_METRIC = 'body_overlap_sequence_v1'
FROZEN_TRAIN_SHA256 = 'b93585d97ab00a603f492b3d42b3f4334732c25756651d8fc21f2a78f99f3fc6'
FROZEN_ENVIRONMENT_SHA256 = 'c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230'
HISTORICAL_NAVIGATION_BANK = {
    'near': {'phase_key': '0a', 'min_dist': 1.5, 'max_dist': 3.0, 'seed_start': 84001},
    'wide': {'phase_key': '0b', 'min_dist': 2.0, 'max_dist': 6.0, 'seed_start': 83001},
}


@contextmanager
def _working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def load_frozen_train(frozen_root):
    """Import the pinned source without importing a worktree's ``train``.

The original source uses absolute ``environment`` imports. A process with a
different environment already imported is rejected rather than silently
reusing it. B1 may safely import the same frozen environment first.
"""
    root = Path(frozen_root).resolve(strict=True)
    game_root = root / 'UTMIST-AI2-main'
    for path, expected in (
        (root / 'train.py', FROZEN_TRAIN_SHA256),
        (game_root / 'environment/environment.py', FROZEN_ENVIRONMENT_SHA256),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source hash mismatch: {path}')
    for name, module in tuple(sys.modules.items()):
        if name == 'environment' or name.startswith('environment.'):
            origin = getattr(module, '__file__', None)
            if origin and not Path(origin).resolve().is_relative_to(game_root):
                raise RuntimeError(f'Conflicting environment import: {origin}')
    name = '_ai2_curriculum_frozen_train'
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != root / 'train.py':
            raise RuntimeError('Frozen train module already belongs to another source root')
        return existing
    tracking = sys.modules.get('tracking')
    if tracking is not None and Path(tracking.__file__).resolve() != root / 'tracking.py':
        raise RuntimeError(f'Conflicting tracking import: {tracking.__file__}')
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    previous_path = sys.path[:]
    try:
        sys.path[:0] = [str(root), str(game_root)]
        spec = importlib.util.spec_from_file_location(name, root / 'train.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        with _working_directory(root):
            spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.path[:] = previous_path
    return module


def navigation_params(variant):
    """Return the historical evaluation settings, without opponent history."""
    if variant not in HISTORICAL_NAVIGATION_BANK:
        raise ValueError(f'Unknown navigation variant: {variant}')
    bank = HISTORICAL_NAVIGATION_BANK[variant]
    nav = {
        'min_dist': bank['min_dist'], 'max_dist': bank['max_dist'],
        'fall_penalty': -3.0, 'time_limit_seconds': 20, 'multi_target': 3,
        'completion_mode': 'sequence', 'control_mode': 'movement',
        'settle_seconds': .3, 'post_goal_seconds': .5, 'settle_speed': 1.0,
    }
    return {
        'phases': {bank['phase_key']: {
            'name': f'Historical {variant} navigation', 'type': 'navigation',
            'navigation': nav, 'opponents': {'constant_agent': 1.0},
        }},
        'opponent_history': {'enabled': False}, 'frame_stack': 4,
    }


def historical_scenarios(variant, count=100):
    """The reused historical development bank; never label this a holdout."""
    navigation_params(variant)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 100:
        raise ValueError('Historical development count must be between 1 and 100')
    start = HISTORICAL_NAVIGATION_BANK[variant]['seed_start']
    return [{
        'id': f'historical-development-{variant}-{seed}', 'seed': seed,
        'variant': variant, 'player_slot': 0, 'metric_version': HISTORICAL_METRIC,
        'bank': 'historical-development-reuse',
    } for seed in range(start, start + count)]


def bodies_overlap(first, second):
    """Use the actual Pymunk body polygons, including off-center overlap."""
    first.shape.cache_bb()
    second.shape.cache_bb()
    return any(point.distance <= 0 for point in first.shape.shapes_collide(second.shape).points)


def _overlap_settled(wrapper, player, distance):
    opponent = wrapper._get_opponent()
    if opponent is None or not bodies_overlap(player, opponent) or not player.is_on_floor():
        return False
    support = getattr(player, 'on_platform', None)
    support_velocity = support.velocity if support is not None else (0.0, 0.0)
    velocity = player.body.velocity
    return bool(np.hypot(velocity[0] - support_velocity[0], velocity[1] - support_velocity[1]) <= wrapper.settle_speed)


class _PlayerSlotView:
    """Present physical player one as learner zero to the unchanged wrapper.

    This swaps indexing only. It does not mirror positions, reorder physics
    processing, or change collision IDs. Raw game state stays accessible on
    NavigationEnv.raw_env for truthful traces and slot-based evaluation.
    """
    def __init__(self, raw_env, player_slot):
        self.raw_env = raw_env
        self.player_slot = player_slot

    def __getattr__(self, name):
        return getattr(self.raw_env, name)

    @property
    def players(self):
        return [self.raw_env.players[self.player_slot], self.raw_env.players[1 - self.player_slot]]

    def observe(self, agent):
        return self.raw_env.observe(self.player_slot if agent == 0 else 1 - self.player_slot)

    def reset(self, **kwargs):
        obs, info = self.raw_env.reset(**kwargs)
        return {0: obs[self.player_slot], 1: obs[1 - self.player_slot]}, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.raw_env.step({
            self.player_slot: actions[0], 1 - self.player_slot: actions[1],
        })
        return ({0: obs[self.player_slot], 1: obs[1 - self.player_slot]},
                {0: rewards[self.player_slot], 1: rewards[1 - self.player_slot]},
                terminated, truncated, info)


class NavigationEnv(gym.Wrapper):
    """Raw64 Gym interface with private game RNG and terminal diagnostics."""
    def __init__(self, env, frozen_root, seed, variant, player_slot, metric_version):
        super().__init__(env)
        self.frozen_root = Path(frozen_root).resolve()
        self.variant = variant
        self.player_slot = player_slot
        self.metric_version = metric_version
        self.raw_env = env.unwrapped.raw_env
        self._perspective = _PlayerSlotView(self.raw_env, player_slot)
        env.unwrapped.raw_env = self._perspective
        self.navigation_wrapper = env.env
        if metric_version == BODY_OVERLAP_METRIC:
            self.navigation_wrapper._is_settled_at_target = MethodType(_overlap_settled, self.navigation_wrapper)
        self.allowed_control_mask = np.zeros(10, dtype=bool)
        self.allowed_control_mask[list(NAVIGATION_ACTIVE_INDICES)] = True
        self._set_seed(seed)
        self.scenario_id = f'{variant}-{seed}-slot{player_slot}'
        self._done = True

    def _set_seed(self, seed):
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or not 0 <= seed < 2**32:
            raise ValueError('seed must be a uint32 integer')
        self.seed_value = int(seed)
        self._np_state = np.random.RandomState(seed).get_state()
        self._py_state = random.Random(int(seed)).getstate()

    @contextmanager
    def _game_rng(self):
        previous_np, previous_py = np.random.get_state(), random.getstate()
        try:
            np.random.set_state(self._np_state)
            random.setstate(self._py_state)
            yield
        finally:
            self._np_state, self._py_state = np.random.get_state(), random.getstate()
            np.random.set_state(previous_np)
            random.setstate(previous_py)

    def _info(self, info):
        if 'opponent_action' in info:
            info['opponent_action'] = np.asarray(info['opponent_action']).tolist()
        info.update({
            'curriculum_task': 'navigation',
            'allowed_control_mask': self.allowed_control_mask.tolist(),
            'metric_version': self.metric_version, 'variant': self.variant,
            'player_slot': self.player_slot, 'seed': self.seed_value,
            'scenario_id': self.scenario_id,
        })
        if self.metric_version != HISTORICAL_METRIC:
            info['nav_contract'] = self.metric_version
        return info

    def reset(self, *, seed=None, options=None):
        scenario = dict((options or {}).get('scenario', {}))
        if scenario.get('variant', self.variant) != self.variant or scenario.get('metric_version', self.metric_version) != self.metric_version:
            raise ValueError('Scenario variant/metric does not match this environment')
        slot = scenario.get('player_slot', self.player_slot)
        if not isinstance(slot, (int, np.integer)) or slot not in (0, 1) or isinstance(slot, bool):
            raise ValueError('player_slot must be zero or one')
        if seed is not None and 'seed' in scenario and seed != scenario['seed']:
            raise ValueError('Reset seed and scenario seed disagree')
        requested_seed = scenario.get('seed', seed)
        if requested_seed is not None:
            self._set_seed(requested_seed)
        self.player_slot = slot
        self._perspective.player_slot = slot
        self.scenario_id = scenario.get('id', f'{self.variant}-{self.seed_value}-slot{slot}')
        with self._game_rng(), _working_directory(self.frozen_root / 'UTMIST-AI2-main/environment'):
            obs, info = self.env.reset(seed=self.seed_value)
        self.action_space.seed(self.seed_value)
        self.observation_space.seed(self.seed_value)
        self._done = False
        return obs, self._info(info)

    def step(self, action):
        if self._done:
            raise RuntimeError('reset is required before stepping a completed episode')
        raw = np.asarray(action)
        if raw.shape != (10,) or not np.isfinite(raw).all():
            raise ValueError('Navigation actions must be ten finite values')
        executed = np.where(self.allowed_control_mask, np.clip(raw, 0, 1), 0).astype(np.float32)
        with self._game_rng():
            obs, reward, terminated, truncated, info = self.env.step(executed)
        info = self._info(info)
        nav = self.navigation_wrapper
        info['navigation_trace'] = {
            'tick': nav.episode_steps, 'player_slot': self.player_slot,
            'raw_action': raw.tolist(), 'executed_action': executed.tolist(),
            'opponent_action': [0.] * 10,
            'target_position': list(nav.target_position),
            'post_players': [{
                'slot': slot, 'position': list(player.body.position),
                'velocity': list(player.body.velocity), 'stocks': int(player.stocks),
                'grounded': bool(player.is_on_floor()),
            } for slot, player in enumerate(self.raw_env.players)],
        }
        if terminated or truncated:
            fall = bool(nav.fell_this_episode)
            success = bool(info.get('nav_success') and terminated and not truncated and not fall)
            info['episode_diagnostics'] = {
                'success': success, 'fall': fall,
                'outcome': 'fall' if fall else 'success' if success else 'timeout' if truncated else 'terminated',
                'metric_version': self.metric_version, 'variant': self.variant,
                'player_slot': self.player_slot, 'scenario_id': self.scenario_id,
                'seed': self.seed_value, 'steps': nav.episode_steps,
                'targets_reached': nav.targets_this_episode,
                'hold_steps': nav.hold_steps, 'post_goal_steps': nav.post_goal_steps,
            }
            self._done = True
        return obs, float(reward), terminated, truncated, info


def make_navigation_env(frozen_root, seed, variant='near', player_slot=0, *, metric_version=HISTORICAL_METRIC):
    """Build one environment; wrap outside with zero-padded VecFrameStack(4).

    Suitable for a functools.partial passed to SubprocVecEnv. Game physics
    alone executes here; this module never loads or evaluates a policy.
    """
    params = navigation_params(variant)
    if not isinstance(player_slot, (int, np.integer)) or player_slot not in (0, 1) or isinstance(player_slot, bool):
        raise ValueError('player_slot must be zero or one')
    if metric_version not in (HISTORICAL_METRIC, BODY_OVERLAP_METRIC):
        raise ValueError('Unknown navigation metric version')
    train = load_frozen_train(frozen_root)
    phase_key = HISTORICAL_NAVIGATION_BANK[variant]['phase_key']
    with _working_directory(Path(frozen_root) / 'UTMIST-AI2-main/environment'):
        env = train.make_nav_env(train.PhaseConfig(params, phase_key), train.get_resolution('LOW'), params)
    return NavigationEnv(env, frozen_root, seed, variant, player_slot, metric_version)
