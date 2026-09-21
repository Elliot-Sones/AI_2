"""CPU reference for the shared C/CUDA B2 re-engagement task."""

from __future__ import annotations

import ctypes as C
import hashlib
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import native_navigation_episode as nav
from native_game_episode import ROOT, GameReference, PlatformTable
from native_game_episode import State as GameState

D, INT32, U = C.c_double, C.c_int32, C.c_uint64
VERSION = "bridge_b2_reengagement-v1"
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
FAILURE_CATEGORIES = (
    None,
    "success",
    "own_stock_loss",
    "opponent_stock_loss",
    "unsafe_support",
    "response_deadline",
    "second_hit_deadline",
    "total_deadline",
    "first_hit_only",
    "failed_renewed_pursuit",
    "pursued_missed_second",
    "started_missed",
    "requested_no_start",
    "no_approach",
    "no_attack_request",
    "deadline",
)
STATE_NAMES = (
    "WalkingState",
    "StandingState",
    "TurnaroundState",
    "AirTurnaroundState",
    "SprintingState",
    "StunState",
    "InAirState",
    "DodgeState",
    "AttackState",
    "DashState",
    "BackDashState",
    "KOState",
    "TauntState",
)
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


class Scenario(C.Structure):
    _fields_ = [
        ("seed", U),
        ("player_slot", INT32),
        ("direction", INT32),
        ("gap", D),
        ("midpoint", D),
    ]


class PlayerSnapshot(C.Structure):
    _fields_ = (
        [
            (n, D)
            for n in (
                "x",
                "y",
                "vx",
                "vy",
                "damage",
                "damage_done",
                "damage_taken",
                "clearance",
            )
        ]
        + [
            (n, INT32)
            for n in (
                "player_slot",
                "facing",
                "stocks",
                "state_id",
                "grounded",
                "fully_supported",
                "stun_time_stored",
            )
        ]
        + [("attack_id", U)]
    )


class PlayerEvent(C.Structure):
    _fields_ = [
        ("kind", INT32),
        ("source", INT32),
        ("target", INT32),
        ("move", INT32),
        ("power", INT32),
        ("cast", INT32),
        ("hitbox", INT32),
        ("phase_player", INT32),
        ("event_sequence", INT32),
        ("attack_id", U),
        ("amount", D),
    ]


class Diagnostics(C.Structure):
    _fields_ = (
        [
            ("seed", U),
            ("scenario_id_hash", U),
            ("first_hit_attack_id", U),
            ("second_hit_attack_id", U),
            ("hit_attack_id", U),
        ]
        + [
            (n, INT32)
            for n in (
                "player_slot",
                "direction",
                "tick",
                "outcome",
                "first_hit_tick",
                "second_hit_tick",
                "relocation_tick",
                "pursuit_tick",
                "pursuit_direction",
                "terminal_reason",
                "success",
                "sequence_completed",
                "failure_category",
                "achieved_first_hit",
                "achieved_relocation",
                "achieved_pursuit",
                "achieved_second_hit",
                "achieved_safe_resolution",
                "attributed_damage_event_count",
            )
        ]
        + [
            ("attributed_damage_attack_ids", U * 2),
            ("attributed_damage_ticks", INT32 * 2),
            ("attributed_damage_sources", INT32 * 2),
            ("attributed_damage_targets", INT32 * 2),
            ("attributed_damage_moves", INT32 * 2),
            ("attributed_damage_amounts", D * 2),
        ]
        + [("initial_stocks", INT32 * 2)]
        + [
            (n, INT32)
            for n in (
                "light_request_ticks",
                "light_press_count",
                "max_light_hold",
                "attack_ticks",
                "movement_request_ticks",
                "renewed_movement_request_ticks",
                "attack_start_count",
            )
        ]
        + [(n, U) for n in ("attack_starts", "attack_contacts", "damage_events")]
        + [
            (n, D)
            for n in (
                "initial_learner_x",
                "initial_target_x",
                "spawn_gap",
                "spawn_midpoint",
                "initial_gap",
                "max_initial_approach_progress",
                "relocation_gap",
                "relocation_learner_x",
                "relocation_target_x",
                "gap_closed",
                "learner_progress",
            )
        ]
        + [
            ("reward_totals", D * 8),
            ("min_clearance", D * 2),
            ("unsupported_ticks", INT32 * 2),
            ("falls", INT32 * 2),
            ("stock_changes", INT32 * 2),
            ("relocation_pre_target", PlayerSnapshot),
            ("relocation_post_target", PlayerSnapshot),
            ("players", PlayerSnapshot * 2),
        ]
    )


class State(C.Structure):
    _fields_ = [
        ("game", GameState),
        ("scenario", Scenario),
        ("diagnostics", Diagnostics),
        ("initial_stocks", INT32 * 2),
    ] + [
        (n, INT32)
        for n in (
            "initialized",
            "error",
            "previous_light",
            "current_light_hold",
            "second_stage",
            "relocated",
        )
    ] + [
        (n, U)
        for n in (
            "first_attack_id",
            "second_attack_id",
            "last_second_duplicate_id",
        )
    ]


class StepResult(C.Structure):
    _fields_ = (
        [
            ("obs", D * 64),
            ("reward", D),
            ("reward_terms", D * 8),
            ("raw_action", D * 10),
            ("executed_action", D * 10),
        ]
        + [(n, INT32) for n in ("terminated", "truncated", "error")]
        + [("pre_players", PlayerSnapshot * 2), ("diagnostics", Diagnostics)]
    )


ABI_LAYOUTS = (
    ("scenario", Scenario),
    ("player", PlayerSnapshot),
    ("event", PlayerEvent),
    ("diagnostics", Diagnostics),
    ("state", State),
    ("result", StepResult),
)
ABI_OFFSETS = {
    cls: {
        field_name: (
            f"ai2b2_{prefix}_field_offset",
            index,
        )
        for index, (field_name, *_rest) in enumerate(cls._fields_)
    }
    for prefix, cls in ABI_LAYOUTS
}


def scenario_from_seed(seed, player_slot=None, direction=None):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if player_slot is None:
        player_slot = seed % 2
    if type(player_slot) is not int or player_slot not in (0, 1):
        raise ValueError("player_slot must be 0 or 1")
    if direction is None:
        direction = -1 if (seed // 2) % 2 == 0 else 1
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    generator = np.random.default_rng(seed)
    return Scenario(
        seed,
        player_slot,
        direction,
        float(generator.uniform(3.1, 3.4)),
        float(generator.uniform(4.35, 4.65)),
    )


def compile_reference(build_dir=None):
    sources = sorted((ROOT / "native").glob("*.h")) + [
        ROOT / "native" / name
        for name in ("reengagement_episode.c", "game_episode.c", "navigation_episode.c")
    ]
    contract = nav.contact.default_contraction()
    compiler = os.environ.get("CC", "cc")
    digest = hashlib.sha256((contract + compiler + platform.platform()).encode())
    for path in sources:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    directory = Path(build_dir or ROOT / ".native-build")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = directory / f"libreengagement-{digest.hexdigest()[:20]}{suffix}"
    if not library.exists():
        with tempfile.TemporaryDirectory(prefix="reengagement-build-", dir=directory) as temporary:
            candidate = Path(temporary) / library.name
            subprocess.run(
                [
                    compiler,
                    "-std=c99",
                    "-O2",
                    f"-ffp-contract={contract}",
                    "-fno-fast-math",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-shared",
                    "-fPIC",
                    *[str(p) for p in sources if p.suffix == ".c"],
                    "-lm",
                    "-o",
                    str(candidate),
                ],
                check=True,
            )
            os.replace(candidate, library)
    return library


def _snapshot_projection(player):
    row = {name: getattr(player, name) for name, _ in PlayerSnapshot._fields_}
    row["state"] = STATE_NAMES[row.pop("state_id")]
    row["horizontal_floor_clearance"] = row.pop("clearance")
    row["grounded"] = bool(row["grounded"])
    row["fully_supported"] = bool(row["fully_supported"])
    row["attack_id"] = row["attack_id"] or None
    return row


def diagnostics_projection(d):
    players = [_snapshot_projection(player) for player in d.players]
    pre = _snapshot_projection(d.relocation_pre_target)
    post = _snapshot_projection(d.relocation_post_target)
    if d.relocation_tick < 0:
        pre = post = None
    count = max(0, min(int(d.attributed_damage_event_count), 2))
    damage_events = []
    for i in range(count):
        damage_events.append(
            {
                "type": "damage",
                "source_slot": d.attributed_damage_sources[i],
                "target_slot": d.attributed_damage_targets[i],
                "attack_id": d.attributed_damage_attack_ids[i] or None,
                "attack_type": "LIGHT",
                "move": d.attributed_damage_moves[i],
                "amount": d.attributed_damage_amounts[i],
                "tick": d.attributed_damage_ticks[i],
            }
        )
    return {
        "version": VERSION,
        "trace_status": "reconstructable",
        "seed": d.seed,
        "scenario_id": f"reengagement-s{d.seed}-p{d.player_slot}-d{d.direction}",
        "scenario_id_hash": d.scenario_id_hash,
        "player_slot": d.player_slot,
        "direction": d.direction,
        "tick": d.tick,
        "outcome": OUTCOMES[d.outcome],
        "terminal_reason": OUTCOMES[d.terminal_reason],
        "success": bool(d.success),
        "sequence_completed": bool(d.sequence_completed),
        "failure_category": FAILURE_CATEGORIES[d.failure_category],
        "achieved_first_hit": bool(d.achieved_first_hit),
        "achieved_relocation": bool(d.achieved_relocation),
        "achieved_pursuit": bool(d.achieved_pursuit),
        "achieved_second_hit": bool(d.achieved_second_hit),
        "achieved_safe_resolution": bool(d.achieved_safe_resolution),
        "first_hit_tick": d.first_hit_tick if d.first_hit_tick >= 0 else None,
        "second_hit_tick": d.second_hit_tick if d.second_hit_tick >= 0 else None,
        "relocation_tick": d.relocation_tick if d.relocation_tick >= 0 else None,
        "pursuit_tick": d.pursuit_tick if d.pursuit_tick >= 0 else None,
        "first_hit_attack_id": d.first_hit_attack_id or None,
        "second_hit_attack_id": d.second_hit_attack_id or None,
        "hit_attack_id": d.hit_attack_id or None,
        "initial_learner_x": d.initial_learner_x,
        "initial_target_x": d.initial_target_x,
        "spawn_gap": d.spawn_gap,
        "spawn_midpoint": d.spawn_midpoint,
        "relocation_gap": None if d.relocation_tick < 0 else d.relocation_gap,
        "relocation_learner_x": None if d.relocation_tick < 0 else d.relocation_learner_x,
        "relocation_target_x": None if d.relocation_tick < 0 else d.relocation_target_x,
        "pursuit_direction": None if d.relocation_tick < 0 else d.pursuit_direction,
        "gap_closed": None if d.pursuit_tick < 0 and d.gap_closed == 0.0 else d.gap_closed,
        "learner_progress": None
        if d.pursuit_tick < 0 and d.learner_progress == 0.0
        else d.learner_progress,
        "reward_terms": dict(zip(REWARD_NAMES, d.reward_totals)),
        "players": players,
        "relocation_pre_target": pre,
        "relocation_post_target": post,
        "light_request_ticks": d.light_request_ticks,
        "light_press_count": d.light_press_count,
        "max_light_hold": d.max_light_hold,
        "attack_ticks": d.attack_ticks,
        "movement_request_ticks": d.movement_request_ticks,
        "renewed_movement_request_ticks": d.renewed_movement_request_ticks,
        "attack_starts": d.attack_starts,
        "attack_contacts": d.attack_contacts,
        "damage_events": d.damage_events,
        "attributed_damage_event_count": count,
        "attributed_damage_events": damage_events,
        "min_clearance": list(d.min_clearance),
        "unsupported_ticks": list(d.unsupported_ticks),
        "falls": list(d.falls),
        "stock_changes": list(d.stock_changes),
    }


class ReengagementReference:
    scenario = staticmethod(scenario_from_seed)

    def __init__(self, library=None):
        self.library = Path(library) if library else compile_reference()
        game = GameReference(self.library)
        self.lib, self.nav, self.cfg, self.table = (
            game.lib,
            game.nav,
            game.cfg,
            game.table,
        )
        self.cfg.time_limit_steps = 300
        self.state, self.rng = State(), nav.Rng()
        for name, cls in (
            ("scenario", Scenario),
            ("state", State),
            ("result", StepResult),
            ("diagnostics", Diagnostics),
            ("player", PlayerSnapshot),
        ):
            fn = getattr(self.lib, f"ai2b2_{name}_size")
            fn.restype = C.c_size_t
            if fn() != C.sizeof(cls):
                raise RuntimeError(f"reengagement ABI size mismatch: {name}")
        for prefix, cls in ABI_LAYOUTS:
            count = getattr(self.lib, f"ai2b2_{prefix}_field_count")
            count.restype = C.c_size_t
            offset = getattr(self.lib, f"ai2b2_{prefix}_field_offset")
            offset.argtypes = [C.c_size_t]
            offset.restype = C.c_size_t
            if count() != len(cls._fields_):
                raise RuntimeError(f"reengagement ABI field-count mismatch: {prefix}")
        self.lib.ai2b2_player_event_size.restype = C.c_size_t
        self.lib.ai2b2_reset_export.argtypes = [
            C.POINTER(State),
            C.POINTER(nav.Config),
            C.POINTER(nav.Rng),
            D,
            C.POINTER(Scenario),
            C.POINTER(StepResult),
        ]
        self.lib.ai2b2_reset_export.restype = INT32
        self.lib.ai2b2_step_export.argtypes = [
            C.POINTER(State),
            C.POINTER(D),
            C.POINTER(nav.Rng),
            C.POINTER(PlatformTable),
            C.POINTER(StepResult),
        ]
        self.lib.ai2b2_step_export.restype = INT32
        self.lib.ai2b2_step_injected_event_export.argtypes = [
            C.POINTER(State),
            C.POINTER(D),
            C.POINTER(nav.Rng),
            C.POINTER(PlatformTable),
            INT32,
            C.POINTER(PlayerEvent),
            C.POINTER(StepResult),
        ]
        self.lib.ai2b2_step_injected_event_export.restype = INT32
        self.lib.ai2b2_debug_prepare_boundary_export.argtypes = [
            C.POINTER(State),
            C.POINTER(PlatformTable),
            INT32,
            INT32,
            INT32,
        ]
        self.lib.ai2b2_debug_prepare_boundary_export.restype = INT32
        self.lib.ai2b2_validate_event_export.argtypes = [
            C.POINTER(State),
            C.POINTER(PlayerEvent),
        ]
        self.lib.ai2b2_validate_event_export.restype = INT32

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"native reengagement error {code}")

    @staticmethod
    def diagnostics(result):
        return diagnostics_projection(result.diagnostics)

    def _reset(self, scenario, rng):
        result = StepResult()
        self._check(
            self.lib.ai2b2_reset_export(
                C.byref(self.state),
                C.byref(self.cfg),
                C.byref(rng),
                48.0,
                C.byref(scenario),
                C.byref(result),
            )
        )
        return result

    def reset_generated(self, scenario, stream=0):
        self.lib.ai2n_rng_seed_export(C.byref(self.rng), scenario.seed, stream)
        return self._reset(scenario, self.rng)

    def reset_replay(self, scenario, draws):
        rng, _keep = self.nav.replay_rng(draws)
        result = self._reset(scenario, rng)
        if rng.cursor != len(draws) or rng.error:
            raise RuntimeError("reengagement reset did not consume the exact random tape")
        return result

    def _step(self, action, rng):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (10,) or not np.isfinite(action).all():
            raise ValueError("action must contain ten finite values")
        action = np.ascontiguousarray(action)
        result = StepResult()
        self._check(
            self.lib.ai2b2_step_export(
                C.byref(self.state),
                action.ctypes.data_as(C.POINTER(D)),
                C.byref(rng),
                C.byref(self.table),
                C.byref(result),
            )
        )
        return result

    def step_generated(self, action):
        return self._step(action, self.rng)

    def step_replay(self, action, draws):
        rng, _keep = self.nav.replay_rng(draws)
        result = self._step(action, rng)
        if rng.cursor != len(draws) or rng.error:
            raise RuntimeError("reengagement step did not consume the exact random tape")
        return result
