"""Versioned combat lessons over the immutable, corrected original game.

Only encounter boundaries replace a passive body. Every ordinary tick delegates
unchanged to WarehouseBrawl; instance hooks observe original collision/damage.
The public observation is raw64; frame history belongs to the policy adapter.
The versioned geometry receipt rejects this two-hit layout; it must not train.
The additive contact lesson requires its own bounded, source-bound receipt.
"""

from contextlib import contextmanager, redirect_stdout
from functools import lru_cache
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import sys
import uuid

import gymnasium as gym
import numpy as np


B1_ACTIVE_INDICES = (0, 1, 2, 3, 4, 6, 7)
B1_ALLOWED_MASK = np.array([i in B1_ACTIVE_INDICES for i in range(10)], dtype=bool)
B1_VERSION = 'bridge_b1-v1'
CONTACT_VERSION = 'bridge_b1_contact-v1'
# The bounded 80-positive/240-negative panel passes. The runner must still
# verify a fresh receipt bound to this adapter and the frozen game source.
CONTACT_TRAINING_QUALIFIED = True
TRACE_VERSION = 'b1-trace-v1'
ORIGINAL_GAME_SHA256 = 'c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230'
B1_TRAINING_QUALIFIED = False
B1_QUALIFICATION_BLOCKER = (
    'Original two-hit drill requires near-edge relocation: scripted learner bodies '
    'overhang the five-unit floor by 0.128 to 0.228 units; target reset clearance '
    'is only 0.015 units. Aerial SAIR contact at 3.01 also exceeds the 3.0 reach '
    'bracket. The drill needs a versioned design decision before training.'
)


@contextmanager
def _asset_directory(module):
    """The original constructor checks ./assets before its obsolete downloader."""
    original = Path.cwd()
    asset_parent = Path(module.__file__).parent
    if not (asset_parent / 'assets').is_dir():
        raise FileNotFoundError('Frozen original-game assets are required')
    try:
        os.chdir(asset_parent)
        yield
    finally:
        os.chdir(original)


def load_original_game(frozen_root):
    root = Path(frozen_root).resolve()
    source = root / 'UTMIST-AI2-main/environment/environment.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != ORIGINAL_GAME_SHA256:
        raise ValueError('Frozen original-game source hash mismatch')
    loaded = sys.modules.get('environment.environment')
    if loaded is not None and Path(loaded.__file__).resolve() != source:
        raise RuntimeError('Another environment.environment source is already loaded')
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(root / 'UTMIST-AI2-main'))
        module = importlib.import_module('environment.environment')
    finally:
        sys.path[:] = previous
    if Path(module.__file__).resolve() != source:
        raise RuntimeError('Original-game import provenance mismatch')
    return module


def scripted_b1_action(env):
    """Physics qualification controller: walk, release movement, down-light."""
    player, target = env.raw_env.players[env.player_slot], env.raw_env.players[1-env.player_slot]
    action = np.zeros(10, dtype=np.float32)
    if type(player.state).__name__ == 'AttackState':
        return action
    if env.hit_attack_id is not None:
        if env.encounter == 1:
            # Down-light advances the caster. Recover room for the next safe
            # target placement through ordinary movement, never a learner warp.
            edge = min((env.safe_min, env.safe_max), key=lambda x: abs(x-player.body.position.x))
            if max(abs(x-player.body.position.x) for x in (env.safe_min, env.safe_max)) < env.measured_light_reach+1:
                action[3 if edge > player.body.position.x else 1] = 1
        return action
    gap = target.body.position.x - player.body.position.x
    direction = 1 if gap >= 0 else -1
    if abs(gap) > 2.1 or int(player.facing) != direction:
        action[3 if direction == 1 else 1] = 1
    elif type(player.state).__name__ == 'StandingState' and abs(player.body.velocity.x) < .01:
        action[2] = action[7] = 1
    return action


class B1Env(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}

    def __init__(self, frozen_root, *, seed=0, player_slot=0, trace_dir=None):
        if player_slot not in (0, 1):
            raise ValueError('player_slot must be 0 or 1')
        self.game = load_original_game(frozen_root)
        self._rng_depth = 0
        self._python_random_state = random.Random(seed).getstate()
        self._numpy_random_state = np.random.RandomState(seed).get_state()
        with self._rng_scope(), _asset_directory(self.game), redirect_stdout(io.StringIO()):
            self.raw_env = self.game.WarehouseBrawl(train_mode=True)
        self.player_slot = player_slot
        self.action_space = gym.spaces.Box(0, 1, shape=(10,), dtype=np.float32)
        # Keep the historical advertised bounds for SB3 checkpoint/set_env
        # compatibility; actual raw coordinates are deliberately never clipped.
        self.observation_space = gym.spaces.Box(
            np.asarray(self.raw_env.observation_space.low, dtype=np.float32),
            np.asarray(self.raw_env.observation_space.high, dtype=np.float32), dtype=np.float32)
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self._trace_handle = None
        self._instance_id = uuid.uuid4().hex
        self._episode_number = 0
        # Conservative upper bracket from the versioned safe-ground sweep.
        # Includes caster movement; the earlier coarse 2.2 estimate failed the
        # expanded grid (a leftward DLIGHT contacted from initial gap 2.7).
        self.measured_light_reach = 3.0
        self.reset(seed=seed)

    @contextmanager
    def _rng_scope(self):
        """Isolate legacy global RNG calls, including nested encounter resets."""
        if self._rng_depth:
            yield
            return
        previous_python, previous_numpy = random.getstate(), np.random.get_state()
        random.setstate(self._python_random_state)
        np.random.set_state(self._numpy_random_state)
        self._rng_depth = 1
        try:
            yield
        finally:
            self._python_random_state = random.getstate()
            self._numpy_random_state = np.random.get_state()
            self._rng_depth = 0
            random.setstate(previous_python)
            np.random.set_state(previous_numpy)

    def _event(self, kind, **fields):
        self.events.append({'type': kind, 'tick': self.tick, 'encounter': self.encounter, **fields})

    def _snapshot(self, player):
        floor = self.raw_env.objects['ground1'].shape.cache_bb()
        body = player.shape.cache_bb()
        clearance = float(min(body.left-floor.left, floor.right-body.right))
        grounded = bool(player.is_on_floor())
        return {'player_slot': player.agent_id, 'x': float(player.body.position.x),
                'y': float(player.body.position.y), 'vx': float(player.body.velocity.x),
                'vy': float(player.body.velocity.y), 'facing': int(player.facing),
                'stocks': int(player.stocks), 'damage': float(player.damage),
                'state': type(player.state).__name__, 'grounded': grounded,
                'fully_supported': bool(grounded and clearance >= 0.),
                'horizontal_floor_clearance': clearance,
                'stun_time_stored': int(player.state.stun_time_stored),
                'attack_id': getattr(player.state, '_b1_attack_id', None) if isinstance(player.state, self.game.AttackState) else None}

    def _snapshots(self):
        return [self._snapshot(p) for p in self.raw_env.players]

    def _instrument(self, player):
        original_give = player.states['attack'].give_move

        def give_move(move_type):
            original_give(move_type)
            self.attack_counter += 1
            attack_id = f'{self.episode_id}:attack:{self.attack_counter}'
            attack = player.states['attack']
            attack._b1_attack_id = attack_id
            attack._b1_attack_type = move_type.name
            self._event('attack_started', player_slot=player.agent_id, attack_id=attack_id, attack_type=move_type.name)
            original_move = attack.move_manager.do_move

            def do_move(*args, **kwargs):
                self._damage_source = (player.agent_id, attack_id, move_type.name)
                try:
                    return original_move(*args, **kwargs)
                finally:
                    self._damage_source = None
            attack.move_manager.do_move = do_move
        player.states['attack'].give_move = give_move

        original_damage = player.apply_damage

        def apply_damage(amount, stun_dealt=0, velocity_dealt=(0, 0)):
            before = float(player.damage)
            original_damage(amount, stun_dealt, velocity_dealt)
            source = self._damage_source
            self._event('damage', source_slot=source[0] if source else None,
                        target_slot=player.agent_id, attack_id=source[1] if source else None,
                        attack_type=source[2] if source else None, amount=float(amount),
                        damage_before=before, damage_after=float(player.damage),
                        source='original_game.Power.do_power' if source else 'unknown')
        player.apply_damage = apply_damage

        original_intersects = player.hurtbox_collider.intersects

        def intersects(other):
            result = original_intersects(other)
            source = self._damage_source
            if result and source is not None:
                self._event('collision', source_slot=source[0], target_slot=player.agent_id,
                            attack_id=source[1], intersects=True, vulnerable=bool(player.state.vulnerable()))
            return result
        player.hurtbox_collider.intersects = intersects

        original_boxes = player.set_hitboxes_to_draw

        def set_boxes(hitboxes=None, points=None, move_facing=None):
            original_boxes(hitboxes, points, move_facing)
            source = self._damage_source
            if hitboxes and source is not None:
                self._event('attack_active', player_slot=player.agent_id, attack_id=source[1], attack_type=source[2])
        player.set_hitboxes_to_draw = set_boxes

    def _place_fresh_player(self, slot, x, y, facing):
        """Discard old contacts, body impulses, inputs and every state at a boundary."""
        old = self.raw_env.players[slot]
        self.raw_env.space.remove(old.shape, old.body)
        with self._rng_scope(), _asset_directory(self.game):
            player = self.game.Player(self.raw_env, slot, start_position=[x, y], color=old.shape.color)
        player.stocks = old.stocks
        player.damage_taken_total = old.damage_taken_total
        player.damage_done = old.damage_done
        player.facing = self.game.Facing.RIGHT if facing == 1 else self.game.Facing.LEFT
        player.state = player.states['standing']
        player.state.dash_timer = 0
        player.hurtbox_collider.center = player.body.position
        self.raw_env.players[slot] = player
        self.raw_env.objects['player' if slot == 0 else 'opponent'] = player
        self.raw_env.space.reindex_shapes_for_body(player.body)
        for p in self.raw_env.players:
            p.opponent = self.raw_env.players[1-p.agent_id]
        self._instrument(player)
        return player

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = dict((options or {}).get('scenario') or {})
        self.player_slot = int(scenario.get('player_slot', self.player_slot))
        if self.player_slot not in (0, 1):
            raise ValueError('player_slot must be 0 or 1')
        self.seed_value = int(seed if seed is not None else self.np_random.integers(0, 2**32))
        self._python_random_state = random.Random(self.seed_value).getstate()
        self._numpy_random_state = np.random.RandomState(self.seed_value).get_state()
        self.direction = int(scenario.get('direction', 1 if self.seed_value % 2 else -1))
        if self.direction not in (-1, 1):
            raise ValueError('direction must be -1 or 1')
        self.scenario_id = str(scenario.get('id', f'b1-s{self.seed_value}-p{self.player_slot}-d{self.direction}'))
        self._episode_number += 1
        self.episode_id = f'{self._instance_id}:{self._episode_number}'
        self.tick, self.encounter, self.attack_counter = 0, 1, 0
        self.events, self.hit_ids = [], []
        self._damage_source = None
        self.hit_attack_id = None
        self.relocation_tick = None
        self.pursuit_tick = None
        self.outcome = None
        self.reward_totals = {'contact': 0., 'completion': 0., 'stock_loss': 0., 'timeout': 0., 'tick': 0.}
        self.previous_light = False
        self.first_hit_tick = None
        self.second_hit_tick = None
        with self._rng_scope(), _asset_directory(self.game), redirect_stdout(io.StringIO()):
            self.raw_env.reset(seed=self.seed_value)
        self.raw_env.max_timesteps = 300
        ground = self.raw_env.objects['ground1']
        bb = ground.shape.cache_bb()
        self.safe_min, self.safe_max = float(bb.left + .45), float(bb.right - .45)
        self.floor_y = float(bb.bottom - .5)
        self._spawn_players()
        self.initial_stocks = [p.stocks for p in self.raw_env.players]
        if not all(p.is_on_floor() for p in self.raw_env.players):
            raise RuntimeError('B1 spawn has no original-game floor support')
        if self._trace_handle is not None:
            self._trace_handle.close()
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_handle = (self.trace_dir / f'{self.episode_id.replace(":", "-")}.jsonl').open('x')
        snapshots = self._snapshots()
        self._observation_before = self._observation()
        return self._observation(), self._info(snapshots, snapshots, np.zeros(10), np.zeros(10), {})

    def _spawn_players(self):
        # Both orientations on the same immobile five-unit floor. Body halfwidth
        # is .435; the entire .87-wide body lies on floor at these reset points.
        learner_x = self.safe_min if self.direction == 1 else self.safe_max
        target_x = learner_x + self.direction * 3.2
        self._place_fresh_player(self.player_slot, learner_x, self.floor_y, self.direction)
        self._place_fresh_player(1-self.player_slot, target_x, self.floor_y, -self.direction)

    def _observation(self):
        return np.asarray(self.raw_env.observe(self.player_slot), dtype=np.float32)

    def _resolved(self):
        return all(not isinstance(p.state, (self.game.AttackState, self.game.StunState))
                   and p.state.stun_time_stored == 0 for p in self.raw_env.players)

    def _relocate(self):
        player, target = self.raw_env.players[self.player_slot], self.raw_env.players[1-self.player_slot]
        if not all(self._snapshot(p)['fully_supported'] for p in (player, target)):
            self.outcome = 'unsafe_floor_support'
            return
        destinations = [self.safe_min, self.safe_max]
        destination = max(destinations, key=lambda x: abs(x-player.body.position.x))
        if abs(destination-player.body.position.x) < self.measured_light_reach + 1:
            # Wait for ordinary player movement to create a safe encounter
            # boundary; the total 300-tick deadline keeps running.
            return
        before = self._snapshot(target)
        self.encounter = 2
        self.hit_attack_id = None
        self.relocation_tick = self.tick
        self.relocation_learner_x = float(player.body.position.x)
        self.relocation_target_x = float(destination)
        self.relocation_distance = abs(destination-player.body.position.x)
        target = self._place_fresh_player(1-self.player_slot, destination, self.floor_y,
                                         1 if player.body.position.x > destination else -1)
        self._event('relocation', player_slot=target.agent_id,
                    **{'from': {'x': before['x'], 'y': before['y']},
                       'to': {'x': float(destination), 'y': self.floor_y}},
                    reason='encounter_boundary', gap=self.relocation_distance)

    def step(self, action):
        with self._rng_scope():
            return self._step(action)

    def _game_step(self, action):
        """Run one untouched original-game tick with shared causal telemetry."""
        if self.outcome is not None:
            raise RuntimeError('reset is required after the episode ends')
        raw = np.asarray(action, dtype=np.float32)
        if raw.shape != (10,) or not np.isfinite(raw).all():
            raise ValueError('Expected ten finite action values')
        executed = np.clip(raw, 0, 1) * B1_ALLOWED_MASK
        self.tick += 1
        self.events = []
        pre = self._snapshots()
        self._observation_before = self._observation()
        if executed[7] > .5 and not self.previous_light:
            self._event('attack_press', player_slot=self.player_slot, control=7)
        self.previous_light = bool(executed[7] > .5)
        self.raw_env.step({self.player_slot: executed, 1-self.player_slot: np.zeros(10, dtype=np.float32)})
        post = self._snapshots()
        for before, after in zip(pre, post):
            if before['attack_id'] is not None and before['attack_id'] != after['attack_id']:
                self._event('attack_ended', player_slot=before['player_slot'], attack_id=before['attack_id'])
            if after['stocks'] < before['stocks']:
                self._event('stock_loss', player_slot=after['player_slot'], before=before['stocks'], after=after['stocks'], attribution='unknown')
        return raw, executed, pre, post

    def _step(self, action):
        raw, executed, pre, post = self._game_step(action)
        terms = {'contact': 0., 'completion': 0., 'stock_loss': 0., 'timeout': 0., 'tick': -.001}
        own_loss = post[self.player_slot]['stocks'] < self.initial_stocks[self.player_slot]
        opponent_loss = post[1-self.player_slot]['stocks'] < self.initial_stocks[1-self.player_slot]
        if own_loss:
            self.outcome, terms['stock_loss'] = 'own_stock_loss', -5.
        elif opponent_loss:
            self.outcome = 'opponent_stock_loss'
        else:
            for event in list(self.events):
                if (event['type'] == 'damage' and event['source_slot'] == self.player_slot
                        and event['target_slot'] == 1-self.player_slot and event['amount'] > 0
                        and event['attack_type'] in ('NLIGHT', 'DLIGHT', 'SLIGHT', 'NAIR', 'DAIR', 'SAIR')
                        and event['attack_id'] not in self.hit_ids and self.hit_attack_id is None):
                    self.hit_attack_id = event['attack_id']
                    self.hit_ids.append(event['attack_id'])
                    terms['contact'] = 1.
                    if self.encounter == 1:
                        self.first_hit_tick = self.tick
                    else:
                        self.second_hit_tick = self.tick
                    self._event('hit_credit', player_slot=self.player_slot, attack_id=event['attack_id'])
            if self.relocation_tick is not None:
                player, target = post[self.player_slot], post[1-self.player_slot]
                gap_closed = self.relocation_distance - abs(target['x']-player['x'])
                toward = 1 if self.relocation_target_x > self.relocation_learner_x else -1
                learner_progress = toward * (player['x']-self.relocation_learner_x)
                elapsed = self.tick-self.relocation_tick
                if self.pursuit_tick is None and gap_closed >= 1. and learner_progress >= 1. and elapsed <= 60:
                    self.pursuit_tick = self.tick
                    self._event('pursuit', gap_closed=gap_closed, learner_progress=learner_progress, elapsed_ticks=elapsed)
                if elapsed >= 60 and self.pursuit_tick is None:
                    self.outcome = 'response_deadline'
                elif elapsed >= 90 and self.second_hit_tick is None:
                    self.outcome = 'second_hit_deadline'
            if self.outcome is None and self.hit_attack_id is not None and self._resolved():
                if self.encounter == 1:
                    if all(p['grounded'] for p in post):
                        self._relocate()
                elif self.pursuit_tick is not None and post[self.player_slot]['grounded']:
                    if post[self.player_slot]['fully_supported']:
                        self.outcome, terms['completion'] = 'success', 2.
                    else:
                        self.outcome = 'unsafe_floor_support'
            if self.outcome is None and self.tick >= 300:
                self.outcome = 'timeout'
            if self.outcome in ('timeout', 'response_deadline', 'second_hit_deadline', 'unsafe_relocation'):
                terms['timeout'] = -1.
        return self._finish_step(pre, raw, executed, terms)

    def _finish_step(self, pre, raw, executed, terms):
        for key, value in terms.items():
            self.reward_totals[key] += value
        if self.outcome is not None:
            self._event('episode_ended', outcome=self.outcome)
        post = self._snapshots()
        info = self._info(pre, post, raw, executed, terms)
        # These are finite-task failures, not external interruptions: SB3 must
        # not add a TimeLimit terminal-value bootstrap to their bounded reward.
        return self._observation(), float(sum(terms.values())), bool(self.outcome is not None), False, info

    def _info(self, pre, post, raw, executed, terms):
        trace = {'schema_version': TRACE_VERSION, 'episode_id': self.episode_id, 'seed': self.seed_value,
                 'scenario_id': self.scenario_id, 'player_slot': self.player_slot, 'tick': self.tick,
                 'pre_players': pre, 'post_players': post, 'raw_action': raw.tolist(),
                 'observation_before': self._observation_before.tolist(),
                 'observation_after': self._observation().tolist(),
                 'executed_action': executed.tolist(), 'action_source': 'environment_input',
                 'opponent_action': [0.]*10, 'allowed_mask': B1_ALLOWED_MASK.tolist(),
                 'events': list(self.events), 'reward_terms': terms,
                 'stage': 'bridge_b1', 'stage_version': B1_VERSION, 'encounter': self.encounter,
                 'training_qualified': B1_TRAINING_QUALIFIED,
                 'qualification_blocker': B1_QUALIFICATION_BLOCKER,
                 'outcome': self.outcome, 'measured_light_reach': self.measured_light_reach,
                 'reengagement_metric': 'horizontal_gap_and_learner_progress_v1'}
        info = {'trace': trace, 'curriculum_task': 'combat', 'allowed_control_mask': B1_ALLOWED_MASK.tolist()}
        if self.outcome is not None:
            info['episode_diagnostics'] = {'episode_id': self.episode_id, 'scenario_id': self.scenario_id,
                'seed': self.seed_value, 'player_slot': self.player_slot, 'stage': 'bridge_b1',
                'outcome': self.outcome, 'success': self.outcome == 'success', 'ticks': self.tick,
                'sequence_completed': len(self.hit_ids) == 2 and self._resolved(),
                'training_qualified': B1_TRAINING_QUALIFIED, 'qualification_blocker': B1_QUALIFICATION_BLOCKER,
                'simulated_seconds': self.tick/30., 'distinct_hit_ids': list(self.hit_ids),
                'first_hit_tick': self.first_hit_tick, 'second_hit_tick': self.second_hit_tick,
                'relocation_tick': self.relocation_tick, 'pursuit_tick': self.pursuit_tick,
                'positive_reward': self.reward_totals['contact']+self.reward_totals['completion'],
                'reward_terms': dict(self.reward_totals), 'own_stock_loss': self.outcome == 'own_stock_loss',
                'time_limit_ticks': 300, 'terminal_reason': self.outcome}
        self._stage_info(info)
        if self._trace_handle is not None:
            self._trace_handle.write(json.dumps(trace, allow_nan=False, separators=(',', ':'))+'\n')
            self._trace_handle.flush()
        return info

    def _stage_info(self, info):
        """Additive lessons specialize metadata before the exact trace is saved."""

    def render(self):
        return self.raw_env.render()

    def close(self):
        if self._trace_handle is not None:
            self._trace_handle.close()
            self._trace_handle = None
        self.raw_env.close()


def make_b1_env(frozen_root, *, seed=0, player_slot=0, trace_dir=None):
    """Build physics diagnostics; the runner must enforce B1_TRAINING_QUALIFIED."""
    return B1Env(frozen_root, seed=seed, player_slot=player_slot, trace_dir=trace_dir)


class B1ContactEnv(B1Env):
    """Approach, one real light hit, and a resolved safe finish; no relocation."""

    def _spawn_players(self):
        floor = self.raw_env.objects['ground1'].shape.cache_bb()
        # Sampling has a separate seed-derived stream and never consumes caller
        # RNG or the legacy game's private stream. Do not clip sampled geometry.
        rng = np.random.default_rng(self.seed_value)
        self.spawn_gap = float(rng.uniform(3.1, 3.4))
        self.spawn_midpoint = float((floor.left+floor.right)/2+rng.uniform(-.15, .15))
        self.initial_learner_x = self.spawn_midpoint-self.direction*self.spawn_gap/2
        self.initial_target_x = self.spawn_midpoint+self.direction*self.spawn_gap/2
        self.approach_progress_at_first_hit = None
        self._place_fresh_player(self.player_slot, self.initial_learner_x, self.floor_y, self.direction)
        self._place_fresh_player(1-self.player_slot, self.initial_target_x, self.floor_y, -self.direction)
        snapshots = self._snapshots()
        if not all(p['fully_supported'] and p['horizontal_floor_clearance'] >= .2 for p in snapshots):
            raise RuntimeError('Contact spawn violates the required full-body .2 floor clearance')

    def _step(self, action):
        raw, executed, pre, post = self._game_step(action)
        terms = {'contact': 0., 'completion': 0., 'stock_loss': 0., 'timeout': 0., 'tick': -.001}
        own_loss = post[self.player_slot]['stocks'] < self.initial_stocks[self.player_slot]
        opponent_loss = post[1-self.player_slot]['stocks'] < self.initial_stocks[1-self.player_slot]
        if own_loss:
            self.outcome, terms['stock_loss'] = 'own_stock_loss', -5.
        elif opponent_loss:
            self.outcome = 'opponent_stock_loss'
        else:
            if self.hit_attack_id is None:
                for event in list(self.events):
                    if (event['type'] == 'damage' and event['source_slot'] == self.player_slot
                            and event['target_slot'] == 1-self.player_slot and event['amount'] > 0
                            and event['attack_type'] in ('NLIGHT', 'DLIGHT', 'SLIGHT', 'NAIR', 'DAIR', 'SAIR')
                            and event['attack_id'] is not None):
                        self.hit_attack_id = event['attack_id']
                        self.hit_ids.append(event['attack_id'])
                        self.first_hit_tick = self.tick
                        self.approach_progress_at_first_hit = self.direction*(post[self.player_slot]['x']-self.initial_learner_x)
                        terms['contact'] = 1.
                        self._event('hit_credit', player_slot=self.player_slot, attack_id=event['attack_id'],
                                    approach_progress=self.approach_progress_at_first_hit)
                        break
            if self.hit_attack_id is not None and self._resolved():
                if self.approach_progress_at_first_hit < .25:
                    self.outcome = 'insufficient_approach'
                elif all(p['grounded'] for p in post):
                    if (all(p['stocks'] > 0 and p['fully_supported'] for p in post)
                            and post[self.player_slot]['horizontal_floor_clearance'] >= .2):
                        self.outcome, terms['completion'] = 'success', 2.
                    else:
                        self.outcome = 'unsafe_floor_support'
            if self.outcome is None and self.tick >= 300:
                self.outcome, terms['timeout'] = 'timeout', -1.
        return self._finish_step(pre, raw, executed, terms)

    def _stage_info(self, info):
        trace = info['trace']
        trace.update(stage='bridge_b1_contact', stage_version=CONTACT_VERSION,
                     training_qualified=CONTACT_TRAINING_QUALIFIED, qualification_blocker=None,
                     approach_metric='initial_target_horizontal_progress_v1',
                     initial_learner_x=self.initial_learner_x, initial_target_x=self.initial_target_x,
                     initial_direction=self.direction, spawn_gap=self.spawn_gap, spawn_midpoint=self.spawn_midpoint,
                     approach_progress=self.direction*(trace['post_players'][self.player_slot]['x']-self.initial_learner_x),
                     approach_progress_at_first_hit=self.approach_progress_at_first_hit,
                     required_approach_progress=.25, required_spawn_clearance=.2,
                     required_final_learner_clearance=.2)
        trace.pop('reengagement_metric')
        trace.pop('measured_light_reach')
        if 'episode_diagnostics' in info:
            info['episode_diagnostics'].update(
                stage='bridge_b1_contact', stage_version=CONTACT_VERSION,
                sequence_completed=self.hit_attack_id is not None and self._resolved(),
                training_qualified=CONTACT_TRAINING_QUALIFIED, qualification_blocker=None,
                approach_progress_at_first_hit=self.approach_progress_at_first_hit,
                final_learner_clearance=trace['post_players'][self.player_slot]['horizontal_floor_clearance'])


def make_b1_contact_env(frozen_root, *, seed=0, player_slot=0, trace_dir=None):
    """Build the contact lesson; training additionally requires its code-bound receipt."""
    return B1ContactEnv(frozen_root, seed=seed, player_slot=player_slot, trace_dir=trace_dir)


def scripted_contact_action(env):
    """Walk toward the passive target, brake, down-light once, then settle."""
    player, target = env.raw_env.players[env.player_slot], env.raw_env.players[1-env.player_slot]
    action = np.zeros(10, dtype=np.float32)
    if env.hit_attack_id is not None or isinstance(player.state, env.game.AttackState):
        return action
    gap = target.body.position.x-player.body.position.x
    direction = 1 if gap >= 0 else -1
    if abs(gap) > 2.1 or int(player.facing) != direction:
        action[3 if direction == 1 else 1] = 1
    elif isinstance(player.state, env.game.StandingState) and abs(player.body.velocity.x) < .01:
        action[2] = action[7] = 1
    return action


def qualify_contact_geometry(frozen_root):
    """Bounded behavioral panel, not a global reach or policy-learning guarantee."""
    env = make_b1_contact_env(frozen_root, seed=0)
    positive, negative = [], []
    try:
        for seed in range(20):
            for slot in (0, 1):
                for direction in (-1, 1):
                    for controller in ('scripted_contact', 'idle', 'stationary_held_light', 'walkoff'):
                        _, reset_info = env.reset(seed=seed, options={'scenario': {
                            'player_slot': slot, 'direction': direction,
                            'id': f'contact-qualification-{seed}-{slot}-{direction}-{controller}'}})
                        spawn_clearance = min(p['horizontal_floor_clearance'] for p in reset_info['trace']['post_players'])
                        min_clearance, relocations = spawn_clearance, 0
                        for _ in range(300):
                            action = np.zeros(10, dtype=np.float32)
                            if controller == 'scripted_contact':
                                action = scripted_contact_action(env)
                            elif controller == 'stationary_held_light':
                                action[7] = 1
                            elif controller == 'walkoff':
                                action[3 if direction == 1 else 1] = 1
                            _, _, terminated, truncated, info = env.step(action)
                            trace = info['trace']
                            min_clearance = min(min_clearance, *(p['horizontal_floor_clearance'] for p in trace['post_players']))
                            relocations += sum(event['type'] == 'relocation' for event in trace['events'])
                            if terminated or truncated:
                                break
                        case = {'seed': seed, 'player_slot': slot, 'direction': direction, 'controller': controller,
                                'outcome': env.outcome, 'ticks': env.tick, 'first_hit_tick': env.first_hit_tick,
                                'approach_progress_at_first_hit': env.approach_progress_at_first_hit,
                                'spawn_gap': env.spawn_gap, 'spawn_midpoint': env.spawn_midpoint,
                                'spawn_min_clearance': spawn_clearance, 'min_horizontal_floor_clearance': min_clearance,
                                'final_learner_clearance': trace['post_players'][slot]['horizontal_floor_clearance'],
                                'terminal_players': trace['post_players'], 'relocations': relocations,
                                'reward_terms': dict(env.reward_totals)}
                        (positive if controller == 'scripted_contact' else negative).append(case)
        qualified = (all(case['outcome'] == 'success' and case['spawn_min_clearance'] >= .2
                         and case['final_learner_clearance'] >= .2 for case in positive)
                     and all(case['outcome'] != 'success' for case in negative)
                     and all(case['relocations'] == 0 for case in positive+negative))
        return {'qualification_version': 'contact_safe_spawn_panel_v1', 'stage_version': CONTACT_VERSION,
                'source_sha256': ORIGINAL_GAME_SHA256,
                'adapter_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'qualified': qualified, 'qualification_blocker': None if qualified else 'Bounded contact panel failed',
                'seed_panel': {'seeds': list(range(20)), 'player_slots': [0, 1], 'directions': [-1, 1],
                               'spawn_gap_uniform': [3.1, 3.4], 'midpoint_jitter_uniform': [-.15, .15],
                               'required_spawn_clearance': .2, 'required_final_learner_clearance': .2,
                               'required_approach_progress': .25, 'time_limit_ticks': 300},
                'positive_cases': positive, 'negative_cases': negative}
    finally:
        env.close()


def _measure_original_light(env, slot, direction, learner_x, gap, move, air_velocity=None):
    env.reset(seed=17, options={'scenario': {'player_slot': slot, 'direction': direction}})
    height = .4 if air_velocity is not None else 0.
    player = env._place_fresh_player(slot, learner_x, env.floor_y-height, direction)
    if air_velocity is not None:
        player.body.velocity = (0., air_velocity)
        player.state = player.states['in_air']
        player.state.refresh()
    target_x = learner_x+direction*float(gap)
    target = env._place_fresh_player(1-slot, target_x, env.floor_y, -direction)
    action = np.zeros(10, dtype=np.float32)
    action[7] = 1
    if move in ('DLIGHT', 'DAIR'):
        action[2] = 1
    elif move in ('SLIGHT', 'SAIR'):
        action[3 if direction == 1 else 1] = 1
    hits, duration = [], 0
    with env._rng_scope():
        for tick in range(1, 49):
            before = target.damage_taken_total
            env.raw_env.step({slot: action if tick == 1 else np.zeros(10), 1-slot: np.zeros(10)})
            if target.damage_taken_total > before:
                hits.append(tick)
            duration += isinstance(player.state, env.game.AttackState)
    return {'player_slot': slot, 'direction': direction, 'learner_x': learner_x, 'target_x': target_x,
            'gap': float(gap), 'move_type': move, 'height_above_floor': height,
            'vertical_velocity': air_velocity, 'hit_ticks': hits, 'attack_ticks': int(duration)}


@lru_cache(maxsize=2)
def qualify_light_geometry(frozen_root):
    """Measure real stationary-start light attacks over a declared safe grid.

    This is a finite qualification panel, not a mathematical global reach proof.
    Includes original caster movement, collision implementation and both update
    orders. A failing upper bracket blocks qualification instead of changing B1.
    """
    env = make_b1_env(frozen_root, seed=17)
    records = []
    try:
        for slot in (0, 1):
            for direction in (-1, 1):
                for anchor in (2.45, 2.55, 2.6, 2.75, 3., 3.2, 3.25, 3.5, 3.75, 4., 4.25, 4.5):
                    learner_x = anchor if direction == 1 else 9.-anchor
                    for gap in np.round(np.arange(1.5, 4.11, .1), 2):
                        target_x = learner_x + direction*float(gap)
                        if not env.safe_min <= target_x <= env.safe_max:
                            continue
                        for move in ('NLIGHT', 'DLIGHT', 'SLIGHT'):
                            records.append(_measure_original_light(env, slot, direction, learner_x, gap, move))
        aerial = []
        for slot in (0, 1):
            for direction in (-1, 1):
                for anchor in (2.45, 2.6, 3., 3.5):
                    learner_x = anchor if direction == 1 else 9.-anchor
                    for gap in np.round(np.arange(2., 4.11, .1), 2):
                        if not env.safe_min <= learner_x+direction*float(gap) <= env.safe_max:
                            continue
                        for move in ('NAIR', 'DAIR', 'SAIR'):
                            for velocity in (0., -4.):
                                aerial.append(_measure_original_light(env, slot, direction, learner_x, gap, move, velocity))
            # Preserve the already-run boundary refinement; do not expand the
            # search indefinitely to force this narrow-floor drill to qualify.
            for gap in np.round(np.arange(2.95, 3.151, .01), 2):
                aerial.append(_measure_original_light(env, slot, 1, 2.45, gap, 'SAIR', -4.))
        scripted = []
        for slot in (0, 1):
            for direction in (-1, 1):
                env.reset(seed=17, options={'scenario': {'player_slot': slot, 'direction': direction}})
                minimum_clearance = float('inf')
                for _ in range(300):
                    _, _, terminated, truncated, info = env.step(scripted_b1_action(env))
                    clearance = info['trace']['post_players'][slot]['horizontal_floor_clearance']
                    minimum_clearance = min(minimum_clearance, clearance)
                    if terminated or truncated:
                        break
                scripted.append({'player_slot': slot, 'direction': direction,
                                 'outcome': env.outcome, 'ticks': env.tick,
                                 'first_hit_tick': env.first_hit_tick,
                                 'relocation_tick': env.relocation_tick,
                                 'min_horizontal_floor_clearance': minimum_clearance,
                                 'terminal_players': info['trace']['post_players']})
        maximum = max(r['gap'] for r in records if r['hit_ticks'])
        return {'qualification_version': 'ground_light_safe_grid_v2',
                'source_sha256': ORIGINAL_GAME_SHA256, 'directions': [-1, 1], 'player_slots': [0, 1],
                'fixtures': len(records), 'grid_step': .1, 'max_observed_hit_gap': maximum,
                'stage_reach_upper_bound': env.measured_light_reach,
                'qualified': False, 'qualification_blocker': B1_QUALIFICATION_BLOCKER,
                'airborne': {'fixtures': len(aerial), 'coarse_grid_step': .1, 'boundary_grid_step': .01,
                             'max_observed_hit_gap': max(r['gap'] for r in aerial if r['hit_ticks']), 'records': aerial},
                'scripted_safety_panel': scripted,
                'floor': {'left': 2., 'right': 7., 'player_width': .87,
                          'reset_center_margin': .45, 'reset_body_clearance': .015},
                'superseded_coarse_estimate': {'upper_bound': 2.2, 'reason': 'Expanded position grid observed left DLIGHT contact from 2.7'},
                'max_attack_ticks': {move: max(r['attack_ticks'] for r in records if r['move_type'] == move)
                                     for move in ('NLIGHT', 'DLIGHT', 'SLIGHT')},
                'records': records}
    finally:
        env.close()
