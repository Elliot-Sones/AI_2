"""CPU reference for the shared C/CUDA contact task; no policy inference."""

from __future__ import annotations

import ctypes as C
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np

import native_navigation_episode as nav
from native_game_episode import GameReference, State as GameState, PlatformTable, ROOT

D, INT32, U = C.c_double, C.c_int32, C.c_uint64
OUTCOMES = (
    None,
    "success",
    "own_stock_loss",
    "opponent_stock_loss",
    "unsafe_floor_support",
    "insufficient_approach",
    "timeout",
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
REWARD_NAMES = ("contact", "completion", "stock_loss", "timeout", "tick")


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


class Diagnostics(C.Structure):
    _fields_ = (
        [("seed", U), ("hit_attack_id", U)]
        + [
            (n, INT32)
            for n in ("player_slot", "direction", "tick", "outcome", "first_hit_tick")
        ]
        + [("initial_stocks", INT32 * 2)]
        + [
            (n, INT32)
            for n in (
                "light_request_ticks",
                "light_press_count",
                "max_light_hold",
                "attack_ticks",
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
                "approach_progress",
                "approach_progress_at_first_hit",
            )
        ]
        + [
            ("reward_totals", D * 5),
            ("min_clearance", D * 2),
            ("unsupported_ticks", INT32 * 2),
            ("players", PlayerSnapshot * 2),
        ]
    )


class State(C.Structure):
    _fields_ = [
        ("game", GameState),
        ("scenario", Scenario),
        ("diagnostics", Diagnostics),
    ] + [
        (n, INT32)
        for n in ("initialized", "error", "previous_light", "current_light_hold")
    ]


class StepResult(C.Structure):
    _fields_ = (
        [
            ("obs", D * 64),
            ("reward", D),
            ("reward_terms", D * 5),
            ("raw_action", D * 10),
            ("executed_action", D * 10),
        ]
        + [(n, INT32) for n in ("terminated", "truncated", "error")]
        + [("pre_players", PlayerSnapshot * 2), ("diagnostics", Diagnostics)]
    )


def scenario_from_seed(seed, player_slot=0, direction=None):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if type(player_slot) is not int or player_slot not in (0, 1):
        raise ValueError("player_slot must be 0 or 1")
    if direction is None:
        direction = 1 if seed % 2 else -1
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    generator = np.random.default_rng(seed)
    return Scenario(
        seed,
        player_slot,
        direction,
        float(generator.uniform(3.1, 3.4)),
        float(4.5 + generator.uniform(-0.15, 0.15)),
    )


def compile_reference(build_dir=None):
    sources = sorted((ROOT / "native").glob("*.h")) + [
        ROOT / "native" / name
        for name in ("contact_episode.c", "game_episode.c", "navigation_episode.c")
    ]
    contract = nav.contact.default_contraction()
    compiler = os.environ.get("CC", "cc")
    digest = hashlib.sha256((contract + compiler + platform.platform()).encode())
    for path in sources:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    directory = Path(build_dir or ROOT / ".native-build")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = directory / f"libcontact-{digest.hexdigest()[:20]}{suffix}"
    if not library.exists():
        with tempfile.TemporaryDirectory(
            prefix="contact-build-", dir=directory
        ) as temporary:
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


def diagnostics_projection(d):
    players = []
    for player in d.players:
        row = {name: getattr(player, name) for name, _ in PlayerSnapshot._fields_}
        row["state"] = STATE_NAMES[row.pop("state_id")]
        row["horizontal_floor_clearance"] = row.pop("clearance")
        row["grounded"] = bool(row["grounded"])
        row["fully_supported"] = bool(row["fully_supported"])
        row["attack_id"] = row["attack_id"] or None
        players.append(row)
    return {
        "seed": d.seed,
        "player_slot": d.player_slot,
        "direction": d.direction,
        "tick": d.tick,
        "outcome": OUTCOMES[d.outcome],
        "first_hit_tick": d.first_hit_tick if d.first_hit_tick >= 0 else None,
        "hit_attack_id": d.hit_attack_id or None,
        "approach_progress": d.approach_progress,
        "approach_progress_at_first_hit": d.approach_progress_at_first_hit
        if d.hit_attack_id
        else None,
        "initial_learner_x": d.initial_learner_x,
        "initial_target_x": d.initial_target_x,
        "spawn_gap": d.spawn_gap,
        "spawn_midpoint": d.spawn_midpoint,
        "reward_terms": dict(zip(REWARD_NAMES, d.reward_totals)),
        "players": players,
        "light_request_ticks": d.light_request_ticks,
        "light_press_count": d.light_press_count,
        "max_light_hold": d.max_light_hold,
        "attack_ticks": d.attack_ticks,
        "attack_starts": d.attack_starts,
        "attack_contacts": d.attack_contacts,
        "damage_events": d.damage_events,
        "min_clearance": list(d.min_clearance),
        "unsupported_ticks": list(d.unsupported_ticks),
    }


class ContactReference:
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
            fn = getattr(self.lib, f"ai2b1_{name}_size")
            fn.restype = C.c_size_t
            if fn() != C.sizeof(cls):
                raise RuntimeError(f"contact ABI size mismatch: {name}")
        self.lib.ai2b1_reset_export.argtypes = [
            C.POINTER(State),
            C.POINTER(nav.Config),
            C.POINTER(nav.Rng),
            D,
            C.POINTER(Scenario),
            C.POINTER(StepResult),
        ]
        self.lib.ai2b1_reset_export.restype = INT32
        self.lib.ai2b1_step_export.argtypes = [
            C.POINTER(State),
            C.POINTER(D),
            C.POINTER(nav.Rng),
            C.POINTER(PlatformTable),
            C.POINTER(StepResult),
        ]
        self.lib.ai2b1_step_export.restype = INT32

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"native contact error {code}")

    @staticmethod
    def diagnostics(result):
        return diagnostics_projection(result.diagnostics)

    def _reset(self, scenario, rng):
        result = StepResult()
        self._check(
            self.lib.ai2b1_reset_export(
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
        rng, keep = self.nav.replay_rng(draws)
        result = self._reset(scenario, rng)
        if rng.cursor != len(draws) or rng.error:
            raise RuntimeError("contact reset did not consume the exact random tape")
        return result

    def _step(self, action, rng):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (10,) or not np.isfinite(action).all():
            raise ValueError("action must contain ten finite values")
        action = np.ascontiguousarray(action)
        result = StepResult()
        self._check(
            self.lib.ai2b1_step_export(
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
        rng, keep = self.nav.replay_rng(draws)
        result = self._step(action, rng)
        if rng.cursor != len(draws) or rng.error:
            raise RuntimeError("contact step did not consume the exact random tape")
        return result
