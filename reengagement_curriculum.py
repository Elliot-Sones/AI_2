"""B2 re-engagement oracle over the frozen original Python game.

This module is deliberately separate from the B1 contact lesson.  It keeps the
old game untouched, instruments causal damage, and owns the two-encounter
state machine required by ``bridge_b2_reengagement-v1``.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import random
import sys
import uuid
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import ClassVar

import numpy as np

CURRICULUM_VERSION = "bridge_b2_reengagement-v1"
BANK_VERSION = "b2-reengagement-bank-v1"
TRACE_VERSION = "b2-reengagement-trace-v1"
REQUIRED_PYMUNK_VERSION = "6.2.1"
OBSERVATION_RESIDUAL_VELOCITY_EPSILON = 1e-15
OBSERVATION_VELOCITY_INDICES = (2, 3, 34, 35)
REWARD_NAMES = (
    "first_hit",
    "pursuit",
    "second_hit",
    "completion",
    "stock_loss",
    "unsafe_support",
    "deadline",
    "tick",
)
OUTCOMES = (
    None,
    "success",
    "own_stock_loss",
    "opponent_stock_loss",
    "unsafe_support",
    "response_deadline",
    "second_hit_deadline",
    "total_deadline",
)


def canonicalize_public_observation(observation):
    """Return the public float32 observation with contact residual velocities zeroed.

    The frozen observation schema stores controlled-player vx/vy in slots 2/3 and
    opponent vx/vy in slots 34/35. Values with magnitude strictly below
    OBSERVATION_RESIDUAL_VELOCITY_EPSILON are platform contact-solver residue,
    many orders below meaningful game motion, and are exposed as exact +0.0.
    """

    raw = np.asarray(observation)
    result = raw.astype(np.float32, copy=True)
    for index in OBSERVATION_VELOCITY_INDICES:
        if abs(float(raw[index])) < OBSERVATION_RESIDUAL_VELOCITY_EPSILON:
            result[index] = np.float32(0.0)
    return result

ACTIVE_INDICES = (0, 1, 2, 3, 4, 6, 7)
ALLOWED_MASK = np.array([i in ACTIVE_INDICES for i in range(10)], dtype=bool)
ORIGINAL_GAME_SHA256 = "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"
LIGHT_ATTACK_TYPES = ("NLIGHT", "DLIGHT", "SLIGHT", "NAIR", "DAIR", "SAIR")


@contextmanager
def _asset_directory(module):
    original = Path.cwd()
    asset_parent = Path(module.__file__).parent
    if not (asset_parent / "assets").is_dir():
        raise FileNotFoundError("Frozen original-game assets are required")
    try:
        os.chdir(asset_parent)
        yield
    finally:
        os.chdir(original)


def load_original_game(frozen_root):
    try:
        import pymunk
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"B2 re-engagement oracle requires pymunk=={REQUIRED_PYMUNK_VERSION}"
        ) from exc
    if getattr(pymunk, "version", None) != REQUIRED_PYMUNK_VERSION:
        raise RuntimeError(
            f"B2 re-engagement oracle requires pymunk=={REQUIRED_PYMUNK_VERSION}, "
            f"got {getattr(pymunk, 'version', None)!r}"
        )
    root = Path(frozen_root).resolve()
    source = root / "UTMIST-AI2-main/environment/environment.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != ORIGINAL_GAME_SHA256:
        raise ValueError("Frozen original-game source hash mismatch")
    loaded = sys.modules.get("environment.environment")
    if loaded is not None and Path(loaded.__file__).resolve() != source:
        raise RuntimeError("Another environment.environment source is already loaded")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(root / "UTMIST-AI2-main"))
        module = importlib.import_module("environment.environment")
    finally:
        sys.path[:] = previous
    if Path(module.__file__).resolve() != source:
        raise RuntimeError("Original-game import provenance mismatch")
    return module


def choose_target_anchor(floor_left, floor_right, target_half_width, learner_x, direction):
    """Choose the safe target center using actual floor bounds.

    The recreated target needs 0.2 horizontal clearance on both sides and a
    center gap of at least 2.0 from the learner.  When both anchors are equally
    far, scenario direction chooses the intended side.
    """

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    values = (floor_left, floor_right, target_half_width, learner_x)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("anchor inputs must be finite")
    left_anchor = float(floor_left) + float(target_half_width) + 0.2
    right_anchor = float(floor_right) - float(target_half_width) - 0.2
    if left_anchor > right_anchor:
        return None
    candidates = []
    for name, anchor in (("left", left_anchor), ("right", right_anchor)):
        if abs(anchor - learner_x) >= 2.0:
            candidates.append((name, anchor, abs(anchor - learner_x)))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][1]
    left = candidates[0]
    right = candidates[1]
    if left[2] == right[2]:
        return right[1] if direction == 1 else left[1]
    return max(candidates, key=lambda item: item[2])[1]


def _validate_seed(seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _validate_player_slot(player_slot):
    if not isinstance(player_slot, int) or isinstance(player_slot, bool) or player_slot not in (0, 1):
        raise ValueError("player_slot must be 0 or 1")


def _validate_direction(direction):
    if not isinstance(direction, int) or isinstance(direction, bool) or direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")


def _validate_gap(gap):
    if (
        not isinstance(gap, (int, float))
        or isinstance(gap, bool)
        or not math.isfinite(gap)
        or not 3.1 <= gap <= 3.4
    ):
        raise ValueError("gap must be in [3.1, 3.4]")


def _validate_midpoint(midpoint):
    if (
        not isinstance(midpoint, (int, float))
        or isinstance(midpoint, bool)
        or not math.isfinite(midpoint)
        or not 4.35 <= midpoint <= 4.65
    ):
        raise ValueError("midpoint must be in [4.35, 4.65]")


def _validate_scenario(seed, player_slot, direction, gap, midpoint):
    _validate_seed(seed)
    _validate_player_slot(player_slot)
    _validate_direction(direction)
    _validate_gap(gap)
    _validate_midpoint(midpoint)


def scenario_from_seed(seed, player_slot=None, direction=None):
    _validate_seed(seed)
    rng = np.random.default_rng(seed)
    slot = seed % 2 if player_slot is None else player_slot
    dir_value = -1 if (seed // 2) % 2 == 0 else 1
    if direction is not None:
        dir_value = direction
    gap = float(rng.uniform(3.1, 3.4))
    midpoint = float(rng.uniform(4.35, 4.65))
    _validate_scenario(seed, slot, dir_value, gap, midpoint)
    return {
        "seed": seed,
        "player_slot": slot,
        "direction": dir_value,
        "gap": gap,
        "midpoint": midpoint,
        "id": f"reengagement-s{seed}-p{slot}-d{dir_value}",
    }


class ReengagementEnv:
    metadata: ClassVar[dict[str, object]] = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, frozen_root, *, seed=0, player_slot=0, trace_dir=None):
        self.game = load_original_game(frozen_root)
        self._rng_depth = 0
        self._python_random_state = random.Random(seed).getstate()
        self._numpy_random_state = np.random.RandomState(seed).get_state()
        with self._rng_scope(), _asset_directory(self.game), redirect_stdout(io.StringIO()):
            self.raw_env = self.game.WarehouseBrawl(train_mode=True)
        self.player_slot = player_slot
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self._trace_handle = None
        self._instance_id = uuid.uuid4().hex
        self._episode_number = 0
        self._damage_source = None
        self.reset(seed=seed, options={"scenario": {"player_slot": player_slot}})

    @contextmanager
    def _rng_scope(self):
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

    def reset(self, *, seed=None, options=None):
        scenario_options = dict((options or {}).get("scenario") or {})
        seed_value = seed if seed is not None else scenario_options.get("seed", 0)
        _validate_seed(seed_value)
        generated = scenario_from_seed(seed_value)
        player_slot = scenario_options.get("player_slot", generated["player_slot"])
        direction = scenario_options.get("direction", generated["direction"])
        gap = scenario_options.get("gap", generated["gap"])
        midpoint = scenario_options.get("midpoint", generated["midpoint"])
        scenario_id = str(scenario_options.get("id", f"reengagement-s{seed_value}-p{player_slot}-d{direction}"))
        _validate_scenario(seed_value, player_slot, direction, gap, midpoint)
        gap = float(gap)
        midpoint = float(midpoint)

        self.seed_value = seed_value
        self.player_slot = player_slot
        self.direction = direction
        self.scenario = {
            "seed": seed_value,
            "player_slot": player_slot,
            "direction": direction,
            "gap": gap,
            "midpoint": midpoint,
            "id": scenario_id,
        }
        self._python_random_state = random.Random(seed_value).getstate()
        self._numpy_random_state = np.random.RandomState(seed_value).get_state()
        self._episode_number += 1
        self.episode_id = f"{self._instance_id}:{self._episode_number}"
        self.tick = 0
        self.encounter = 1
        self.attack_counter = 0
        self.events = []
        self.hit_ids = []
        self.first_hit_tick = None
        self.second_hit_tick = None
        self.hit_attack_id = None
        self.relocation_tick = None
        self.relocation_gap = None
        self.relocation_learner_x = None
        self.relocation_target_x = None
        self.pursuit_direction = None
        self.pursuit_tick = None
        self.outcome = None
        self.previous_light = False
        self.reward_totals = {name: 0.0 for name in REWARD_NAMES}
        self._damage_source = None
        self._reset_diagnostics()

        with self._rng_scope(), _asset_directory(self.game), redirect_stdout(io.StringIO()):
            self.raw_env.reset(seed=seed_value)
        self.raw_env.max_timesteps = 300
        ground = self.raw_env.objects["ground1"]
        floor = ground.shape.cache_bb()
        self.floor_left = float(floor.left)
        self.floor_right = float(floor.right)
        self.floor_y = float(floor.bottom - 0.5)
        self._spawn_players(gap, midpoint)
        self.initial_stocks = [p.stocks for p in self.raw_env.players]

        if self._trace_handle is not None:
            self._trace_handle.close()
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_handle = (self.trace_dir / f"{self.episode_id.replace(':', '-')}.jsonl").open("x")
        snapshots = self._snapshots()
        if min(p["horizontal_floor_clearance"] for p in snapshots) < 0.2:
            raise RuntimeError("B2 spawn violates required full-body .2 floor clearance")
        self.min_clearance = min(p["horizontal_floor_clearance"] for p in snapshots)
        self.initial_gap = abs(snapshots[1 - self.player_slot]["x"] - snapshots[self.player_slot]["x"])
        self.initial_learner_x = snapshots[self.player_slot]["x"]
        self._observation_before = self._observation()
        terms = {name: 0.0 for name in REWARD_NAMES}
        return self._observation(), self._info(snapshots, snapshots, np.zeros(10), np.zeros(10), terms)

    def _reset_diagnostics(self):
        self.relocation_pre_target = None
        self.relocation_post_target = None
        self.gap_closed = None
        self.learner_progress = None
        self.attributed_damage_events = []
        self.min_clearance = math.inf
        self.unsupported_ticks = {0: 0, 1: 0}
        self.falls = {0: 0, 1: 0}
        self.stock_changes = {0: 0, 1: 0}
        self.light_request_ticks = 0
        self.movement_request_ticks = 0
        self.renewed_movement_request_ticks = 0
        self.attack_start_count = 0
        self.max_initial_approach_progress = 0.0

    def _ensure_diagnostics(self):
        defaults = {
            "relocation_pre_target": None,
            "relocation_post_target": None,
            "gap_closed": None,
            "learner_progress": None,
            "attributed_damage_events": [],
            "min_clearance": math.inf,
            "unsupported_ticks": {0: 0, 1: 0},
            "falls": {0: 0, 1: 0},
            "stock_changes": {0: 0, 1: 0},
            "light_request_ticks": 0,
            "movement_request_ticks": 0,
            "renewed_movement_request_ticks": 0,
            "attack_start_count": 0,
            "max_initial_approach_progress": 0.0,
            "initial_gap": None,
            "initial_learner_x": None,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value.copy() if isinstance(value, (dict, list)) else value)

    def _spawn_players(self, gap, midpoint):
        learner_x = midpoint - self.direction * gap / 2.0
        target_x = midpoint + self.direction * gap / 2.0
        self._place_fresh_player(self.player_slot, learner_x, self.floor_y, self.direction, preserve_from=None)
        self._place_fresh_player(1 - self.player_slot, target_x, self.floor_y, -self.direction, preserve_from=None)

    def _event(self, kind, **fields):
        self.events.append({"type": kind, "tick": self.tick, "encounter": self.encounter, **fields})

    def _observation(self):
        return canonicalize_public_observation(self.raw_env.observe(self.player_slot))

    def _target_half_width(self, target):
        bb = target.shape.cache_bb()
        return float((bb.right - bb.left) / 2.0)

    def _snapshot(self, player):
        floor = self.raw_env.objects["ground1"].shape.cache_bb()
        body = player.shape.cache_bb()
        clearance = float(min(body.left - floor.left, floor.right - body.right))
        grounded = bool(player.is_on_floor())
        return {
            "player_slot": player.agent_id,
            "x": float(player.body.position.x),
            "y": float(player.body.position.y),
            "vx": float(player.body.velocity.x),
            "vy": float(player.body.velocity.y),
            "facing": int(player.facing),
            "stocks": int(player.stocks),
            "damage": float(player.damage),
            "damage_done": float(player.damage_done),
            "damage_taken_total": float(player.damage_taken_total),
            "state": type(player.state).__name__,
            "grounded": grounded,
            "fully_supported": bool(grounded and clearance >= 0.0),
            "horizontal_floor_clearance": clearance,
            "stun_time_stored": int(player.state.stun_time_stored),
            "attack_id": getattr(player.state, "_b2_attack_id", None)
            if isinstance(player.state, self.game.AttackState)
            else None,
        }

    def _snapshots(self):
        return [self._snapshot(p) for p in self.raw_env.players]

    def _instrument(self, player):
        original_give = player.states["attack"].give_move

        def give_move(move_type):
            original_give(move_type)
            self.attack_counter += 1
            attack_id = f"{self.episode_id}:attack:{self.attack_counter}"
            attack = player.states["attack"]
            attack._b2_attack_id = attack_id
            attack._b2_attack_type = move_type.name
            self._event("attack_started", player_slot=player.agent_id, attack_id=attack_id, attack_type=move_type.name)
            original_move = attack.move_manager.do_move

            def do_move(*args, **kwargs):
                self._damage_source = (player.agent_id, attack_id, move_type.name)
                try:
                    return original_move(*args, **kwargs)
                finally:
                    self._damage_source = None

            attack.move_manager.do_move = do_move

        player.states["attack"].give_move = give_move

        original_damage = player.apply_damage

        def apply_damage(amount, stun_dealt=0, velocity_dealt=(0, 0)):
            before = float(player.damage)
            original_damage(amount, stun_dealt, velocity_dealt)
            source = self._damage_source
            self._event(
                "damage",
                source_slot=source[0] if source else None,
                target_slot=player.agent_id,
                attack_id=source[1] if source else None,
                attack_type=source[2] if source else None,
                amount=float(amount),
                damage_before=before,
                damage_after=float(player.damage),
                source="original_game.Power.do_power" if source else "unknown",
            )

        player.apply_damage = apply_damage

    def _place_fresh_player(self, slot, x, y, facing, *, preserve_from):
        old = self.raw_env.players[slot]
        preserved = {}
        if preserve_from is not None:
            for name in ("stocks", "damage", "damage_done", "damage_taken_total"):
                preserved[name] = getattr(preserve_from, name)
        self.raw_env.space.remove(old.shape, old.body)
        with self._rng_scope(), _asset_directory(self.game):
            player = self.game.Player(self.raw_env, slot, start_position=[x, y], color=old.shape.color)
        for name, value in preserved.items():
            setattr(player, name, value)
        player.facing = self.game.Facing.RIGHT if facing == 1 else self.game.Facing.LEFT
        player.body.velocity = (0.0, 0.0)
        player.state = player.states["standing"]
        player.state.dash_timer = 0
        player.hurtbox_collider.center = player.body.position
        self.raw_env.players[slot] = player
        self.raw_env.objects["player" if slot == 0 else "opponent"] = player
        self.raw_env.space.reindex_shapes_for_body(player.body)
        for p in self.raw_env.players:
            p.opponent = self.raw_env.players[1 - p.agent_id]
        self._instrument(player)
        return player

    def _resolved(self):
        return all(
            not isinstance(p.state, (self.game.AttackState, self.game.StunState)) and p.state.stun_time_stored == 0
            for p in self.raw_env.players
        )

    def step(self, action):
        with self._rng_scope():
            return self._step(action)

    def _game_step(self, action):
        if self.outcome is not None:
            raise RuntimeError("reset is required after the episode ends")
        raw = np.asarray(action, dtype=np.float32)
        if raw.shape != (10,) or not np.isfinite(raw).all():
            raise ValueError("Expected ten finite action values")
        executed = np.clip(raw, 0, 1) * ALLOWED_MASK
        self.tick += 1
        self.events = []
        pre = self._snapshots()
        self._observation_before = self._observation()
        if executed[7] > 0.5:
            self.light_request_ticks += 1
        if executed[1] > 0.5 or executed[3] > 0.5:
            self.movement_request_ticks += 1
            if self.relocation_tick is not None:
                self.renewed_movement_request_ticks += 1
        if executed[7] > 0.5 and not self.previous_light:
            self._event("attack_press", player_slot=self.player_slot, control=7)
        self.previous_light = bool(executed[7] > 0.5)
        self.raw_env.step({self.player_slot: executed, 1 - self.player_slot: np.zeros(10, dtype=np.float32)})
        post = self._snapshots()
        for before, after in zip(pre, post):
            if before["attack_id"] is not None and before["attack_id"] != after["attack_id"]:
                self._event("attack_ended", player_slot=before["player_slot"], attack_id=before["attack_id"])
            if after["stocks"] < before["stocks"]:
                self._event("stock_loss", player_slot=after["player_slot"], before=before["stocks"], after=after["stocks"])
        return raw, executed, pre, post

    def _validate_evidence(self):
        for event in self.events:
            if event.get("type") != "damage":
                continue
            if event.get("amount", 0.0) <= 0:
                continue
            if event.get("source_slot") != self.player_slot or event.get("target_slot") != 1 - self.player_slot:
                raise RuntimeError("invalid damage evidence")
            attack_id = event.get("attack_id")
            attack_type = event.get("attack_type")
            if not attack_id or attack_type not in LIGHT_ATTACK_TYPES:
                raise RuntimeError("invalid damage evidence")
            if self.encounter == 2 and attack_id in self.hit_ids:
                credited_second_id = self.hit_ids[1] if self.second_hit_tick is not None and len(self.hit_ids) >= 2 else None
                if attack_id == credited_second_id:
                    continue
                if self.hit_ids and attack_id == self.hit_ids[0]:
                    raise RuntimeError("invalid damage evidence: replayed first encounter attack id after relocation")
                raise RuntimeError("invalid damage evidence: duplicate credited attack id")

    def _track_step_diagnostics(self, pre, post):
        self._ensure_diagnostics()
        for event in self.events:
            if event.get("type") == "attack_started" and event.get("player_slot") == self.player_slot:
                self.attack_start_count += 1
        for snapshot in post:
            slot = snapshot["player_slot"]
            clearance = snapshot["horizontal_floor_clearance"]
            if math.isclose(clearance, 0.2, rel_tol=0.0, abs_tol=1e-9):
                clearance = 0.2
            self.min_clearance = min(self.min_clearance, clearance)
            if snapshot["grounded"] and not snapshot["fully_supported"]:
                self.unsupported_ticks[slot] = self.unsupported_ticks.get(slot, 0) + 1
            initial = self.initial_stocks[slot]
            previous_change = self.stock_changes.get(slot, 0)
            self.stock_changes[slot] = max(previous_change, max(0, initial - snapshot["stocks"]))
        for before, after in zip(pre, post):
            slot = after["player_slot"]
            if after["stocks"] < before["stocks"]:
                self.falls[slot] = self.falls.get(slot, 0) + before["stocks"] - after["stocks"]
        if self.initial_gap is not None and self.initial_learner_x is not None and self.relocation_tick is None:
            learner = post[self.player_slot]
            target = post[1 - self.player_slot]
            gap = abs(target["x"] - learner["x"])
            progress = self.direction * (learner["x"] - self.initial_learner_x)
            self.max_initial_approach_progress = max(
                self.max_initial_approach_progress,
                min(self.initial_gap - gap, progress),
            )

    def _credit_hits(self, terms):
        for event in list(self.events):
            if (
                event["type"] == "damage"
                and event["source_slot"] == self.player_slot
                and event["target_slot"] == 1 - self.player_slot
                and event["amount"] > 0
                and event["attack_type"] in LIGHT_ATTACK_TYPES
                and event["attack_id"] not in self.hit_ids
            ):
                if self.encounter == 1 and self.first_hit_tick is None:
                    self.hit_ids.append(event["attack_id"])
                    self.hit_attack_id = event["attack_id"]
                    self.first_hit_tick = self.tick
                    self.attributed_damage_events.append(dict(event))
                    terms["first_hit"] += 0.5
                    self._event("hit_credit", ordinal=1, player_slot=self.player_slot, attack_id=event["attack_id"])
                elif self.encounter == 2 and self.second_hit_tick is None:
                    self.hit_ids.append(event["attack_id"])
                    self.hit_attack_id = event["attack_id"]
                    self.second_hit_tick = self.tick
                    self.attributed_damage_events.append(dict(event))
                    terms["second_hit"] += 1.0
                    self._event("hit_credit", ordinal=2, player_slot=self.player_slot, attack_id=event["attack_id"])
                break

    def _credit_pursuit(self, post, terms):
        if self.relocation_tick is None or self.pursuit_tick is not None:
            return
        elapsed = self.tick - self.relocation_tick
        if elapsed < 1 or elapsed > 60:
            return
        player = post[self.player_slot]
        target = post[1 - self.player_slot]
        gap_closed = self.relocation_gap - abs(target["x"] - player["x"])
        learner_progress = self.pursuit_direction * (player["x"] - self.relocation_learner_x)
        self.gap_closed = gap_closed
        self.learner_progress = learner_progress
        if gap_closed >= 1.0 and learner_progress >= 1.0:
            self.pursuit_tick = self.tick
            terms["pursuit"] += 0.5
            self._event("pursuit", gap_closed=gap_closed, learner_progress=learner_progress, elapsed_ticks=elapsed)

    def _relocate(self, post):
        player = self.raw_env.players[self.player_slot]
        target = self.raw_env.players[1 - self.player_slot]
        floor = self.raw_env.objects["ground1"].shape.cache_bb()
        anchor = choose_target_anchor(
            float(floor.left),
            float(floor.right),
            self._target_half_width(target),
            float(player.body.position.x),
            self.direction,
        )
        if anchor is None:
            self._event("relocation_pending", reason="no_safe_anchor")
            return False
        before = self._snapshot(target)
        self.encounter = 2
        self.hit_attack_id = None
        self.relocation_tick = self.tick
        self.relocation_learner_x = float(player.body.position.x)
        self.relocation_target_x = float(anchor)
        self.relocation_gap = abs(self.relocation_target_x - self.relocation_learner_x)
        self.pursuit_direction = 1 if self.relocation_target_x >= self.relocation_learner_x else -1
        recreated = self._place_fresh_player(
            1 - self.player_slot,
            self.relocation_target_x,
            self.floor_y,
            -self.pursuit_direction,
            preserve_from=target,
        )
        after = self._snapshot(recreated)
        self.relocation_pre_target = before
        self.relocation_post_target = after
        self._event(
            "relocation",
            reason="encounter_boundary",
            player_slot=recreated.agent_id,
            pre_target=before,
            post_target=after,
            relocation_gap=self.relocation_gap,
            learner_x=self.relocation_learner_x,
            target_x=self.relocation_target_x,
            pursuit_direction=self.pursuit_direction,
        )
        return True

    def _step(self, action):
        self._ensure_diagnostics()
        if self.outcome is not None:
            raise RuntimeError("reset is required after the episode ends")
        raw, executed, pre, post = self._game_step(action)
        self._validate_evidence()
        terms = {name: 0.0 for name in REWARD_NAMES}
        terms["tick"] = -0.001
        self._track_step_diagnostics(pre, post)

        self._credit_hits(terms)
        self._credit_pursuit(post, terms)

        own_loss = post[self.player_slot]["stocks"] < self.initial_stocks[self.player_slot]
        opponent_loss = post[1 - self.player_slot]["stocks"] < self.initial_stocks[1 - self.player_slot]
        if own_loss:
            self.outcome = "own_stock_loss"
            terms["stock_loss"] += -5.0
        elif opponent_loss:
            self.outcome = "opponent_stock_loss"
            terms["stock_loss"] += -1.0

        if self.outcome is None and self.hit_attack_id is not None and self._resolved():
            post = self._snapshots()
            if all(p["grounded"] for p in post):
                learner_clearance = post[self.player_slot]["horizontal_floor_clearance"]
                if not all(p["fully_supported"] for p in post) or learner_clearance < 0.2:
                    self.outcome = "unsafe_support"
                    terms["unsafe_support"] += -2.0
                elif self.encounter == 1:
                    self._relocate(post)
                    post = self._snapshots()
                elif self.second_hit_tick is not None and self.pursuit_tick is not None:
                    self.outcome = "success"
                    terms["completion"] += 2.0

        if self.outcome is None and self.relocation_tick is not None:
            elapsed = self.tick - self.relocation_tick
            if elapsed >= 60 and self.pursuit_tick is None:
                self.outcome = "response_deadline"
                terms["deadline"] += -1.0
            elif elapsed >= 90 and self.second_hit_tick is None:
                self.outcome = "second_hit_deadline"
                terms["deadline"] += -1.0
        if self.outcome is None and self.tick >= 300:
            self.outcome = "total_deadline"
            terms["deadline"] += -1.0
        return self._finish_step(pre, raw, executed, terms)

    def _failure_category(self):
        if self.outcome == "success":
            return "success"
        if self.outcome in ("own_stock_loss", "opponent_stock_loss", "unsafe_support"):
            return self.outcome
        if self.outcome in ("response_deadline", "second_hit_deadline", "total_deadline"):
            if self.relocation_tick is not None:
                if self.pursuit_tick is None:
                    if self.first_hit_tick is not None and self.renewed_movement_request_ticks == 0:
                        return "first_hit_only"
                    return "failed_renewed_pursuit"
                if self.second_hit_tick is None:
                    return "pursued_missed_second"
            if self.first_hit_tick is None:
                if self.attack_start_count > 0:
                    return "started_missed"
                if self.light_request_ticks > 0:
                    return "requested_no_start"
                if self.movement_request_ticks > 0 and self.max_initial_approach_progress < 1.0:
                    return "no_approach"
                return "no_attack_request"
            if self.relocation_tick is None:
                return "first_hit_only"
        return "deadline"

    def _finish_step(self, pre, raw, executed, terms):
        for name, value in terms.items():
            self.reward_totals[name] += value
        if self.outcome is not None:
            self._event("episode_ended", outcome=self.outcome)
        post = self._snapshots()
        info = self._info(pre, post, raw, executed, terms)
        return self._observation(), float(sum(terms.values())), bool(self.outcome is not None), False, info

    def _info(self, pre, post, raw, executed, terms):
        trace = {
            "schema_version": TRACE_VERSION,
            "episode_id": self.episode_id,
            "seed": self.seed_value,
            "scenario_id": self.scenario["id"],
            "scenario": dict(self.scenario),
            "player_slot": self.player_slot,
            "direction": self.direction,
            "tick": self.tick,
            "pre_players": pre,
            "post_players": post,
            "raw_action": raw.tolist(),
            "executed_action": executed.tolist(),
            "opponent_action": [0.0] * 10,
            "allowed_mask": ALLOWED_MASK.tolist(),
            "observation_before": self._observation_before.tolist(),
            "observation_after": self._observation().tolist(),
            "events": list(self.events),
            "reward_terms": dict(terms),
            "reward_totals": dict(self.reward_totals),
            "curriculum_task": "combat",
            "curriculum_version": CURRICULUM_VERSION,
            "stage": "bridge_b2_reengagement",
            "stage_version": CURRICULUM_VERSION,
            "encounter": self.encounter,
            "outcome": self.outcome,
            "spawn_gap": self.scenario["gap"],
            "spawn_midpoint": self.scenario["midpoint"],
            "first_hit_tick": self.first_hit_tick,
            "second_hit_tick": self.second_hit_tick,
            "relocation_tick": self.relocation_tick,
            "relocation_gap": self.relocation_gap,
            "relocation_learner_x": self.relocation_learner_x,
            "relocation_target_x": self.relocation_target_x,
            "pursuit_tick": self.pursuit_tick,
            "pursuit_direction": self.pursuit_direction,
            "distinct_hit_ids": list(self.hit_ids),
            "relocation_pre_target": self.relocation_pre_target,
            "relocation_post_target": self.relocation_post_target,
            "gap_closed": self.gap_closed,
            "learner_progress": self.learner_progress,
            "attributed_damage_events": list(self.attributed_damage_events),
            "min_clearance": self.min_clearance,
            "unsupported_ticks": dict(self.unsupported_ticks),
            "falls": dict(self.falls),
            "stock_changes": dict(self.stock_changes),
        }
        info = {
            "trace": trace,
            "curriculum_task": "combat",
            "curriculum_version": CURRICULUM_VERSION,
            "allowed_control_mask": ALLOWED_MASK.tolist(),
        }
        if self.outcome is not None:
            info["episode_diagnostics"] = {
                "episode_id": self.episode_id,
                "scenario_id": self.scenario["id"],
                "seed": self.seed_value,
                "player_slot": self.player_slot,
                "direction": self.direction,
                "stage": "bridge_b2_reengagement",
                "stage_version": CURRICULUM_VERSION,
                "outcome": self.outcome,
                "success": self.outcome == "success",
                "ticks": self.tick,
                "sequence_completed": len(self.hit_ids) == 2 and self.outcome == "success",
                "distinct_hit_ids": list(self.hit_ids),
                "first_hit_tick": self.first_hit_tick,
                "second_hit_tick": self.second_hit_tick,
                "relocation_tick": self.relocation_tick,
                "relocation_pre_target": self.relocation_pre_target,
                "relocation_post_target": self.relocation_post_target,
                "relocation_gap": self.relocation_gap,
                "pursuit_tick": self.pursuit_tick,
                "gap_closed": self.gap_closed,
                "learner_progress": self.learner_progress,
                "attributed_damage_events": list(self.attributed_damage_events),
                "min_clearance": self.min_clearance,
                "unsupported_ticks": dict(self.unsupported_ticks),
                "falls": dict(self.falls),
                "stock_changes": dict(self.stock_changes),
                "failure_category": self._failure_category(),
                "positive_reward": (
                    self.reward_totals["first_hit"]
                    + self.reward_totals["pursuit"]
                    + self.reward_totals["second_hit"]
                    + self.reward_totals["completion"]
                ),
                "reward_terms": dict(self.reward_totals),
                "terminal_reason": self.outcome,
                "time_limit_ticks": 300,
            }
        if self._trace_handle is not None:
            self._trace_handle.write(json.dumps(trace, allow_nan=False, separators=(",", ":")) + "\n")
            self._trace_handle.flush()
        return info

    def render(self):
        return self.raw_env.render()

    def close(self):
        if self._trace_handle is not None:
            self._trace_handle.close()
            self._trace_handle = None
        self.raw_env.close()


def make_reengagement_env(frozen_root, *, seed=0, player_slot=0, trace_dir=None):
    return ReengagementEnv(frozen_root, seed=seed, player_slot=player_slot, trace_dir=trace_dir)


def scripted_reengagement_action(env):
    """Walk, hit, wait for relocation, pursue, hit again, then settle."""

    learner = env.raw_env.players[env.player_slot]
    target = env.raw_env.players[1 - env.player_slot]
    action = np.zeros(10, dtype=np.float32)
    if isinstance(learner.state, env.game.AttackState):
        return action
    if env.hit_attack_id is not None and (env.encounter == 1 or env.second_hit_tick is not None):
        return action
    gap = target.body.position.x - learner.body.position.x
    direction = 1 if gap >= 0 else -1
    if abs(gap) > 2.1 or int(learner.facing) != direction:
        action[3 if direction == 1 else 1] = 1
    elif isinstance(learner.state, env.game.StandingState) and abs(learner.body.velocity.x) < 0.01:
        action[2] = 1
        action[7] = 1
    return action
