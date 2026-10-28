"""CPU oracle tests for the native full-game combat reward wrapper.

These tests deliberately exercise reward bookkeeping as a layer above the raw
game state. They never call ai2g_step and therefore cannot mask a parity
regression in physics, RNG, observations, or raw rewards.
"""

from __future__ import annotations

import ctypes as C
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = C.c_double
I = C.c_int32
U = C.c_uint64

AI2CR_TERMS = 14
AI2CR_VERSION = 1

TERM_APPROACH = 0
TERM_ATTACK_START = 1
TERM_CONTACT = 2
TERM_DAMAGE_DEALT = 3
TERM_DAMAGE_TAKEN = 4
TERM_NET_DAMAGE = 5
TERM_KO = 6
TERM_WIN = 7
TERM_CLEAN_WIN = 8
TERM_TIMEOUT = 9
TERM_OFFSTAGE = 10
TERM_WALKOFF = 11
TERM_RECOVERY = 12
TERM_WEAPON = 13


class Coefficients(C.Structure):
    _fields_ = [("term", D * AI2CR_TERMS), ("enabled_mask", C.c_uint32), ("fade", D)]


class State(C.Structure):
    _fields_ = [
        ("version", I),
        ("initialized", I),
        ("learner_slot", I),
        ("tick", I),
        ("initial_stocks", I * 2),
        ("last_stocks", I * 2),
        ("last_weapon", I * 2),
        ("last_offstage", I * 2),
        ("clean_stock", I * 2),
        ("last_x", D * 2),
        ("last_y", D * 2),
        ("last_distance", D),
        ("last_damage_done", D * 2),
        ("last_damage_taken", D * 2),
        ("base_damage_done", D * 2),
        ("base_damage_taken", D * 2),
        ("last_attack_starts", U * 2),
        ("last_attack_contacts", U * 2),
        ("last_damage_events", U * 2),
        ("returns", D * 2),
        ("term_totals", (D * AI2CR_TERMS) * 2),
    ]


class Breakdown(C.Structure):
    _fields_ = [
        ("version", I),
        ("learner_slot", I),
        ("tick", I),
        ("terminated", I),
        ("truncated", I),
        ("win_mask", I),
        ("terms", (D * AI2CR_TERMS) * 2),
        ("total", D * 2),
        ("approach_delta", D),
        ("distance", D),
        ("attack_starts", I * 2),
        ("contacts", I * 2),
        ("damage_events", I * 2),
        ("ko_events", I * 2),
        ("weapon_pickups", I * 2),
        ("weapon_drops", I * 2),
        ("offstage", I * 2),
        ("walkoff", I * 2),
        ("recovery", I * 2),
    ]


class EpisodeRecord(C.Structure):
    _fields_ = [
        ("version", I),
        ("learner_slot", I),
        ("length", I),
        ("terminated", I),
        ("truncated", I),
        ("win_mask", I),
        ("stocks", I * 2),
        ("returns", D * 2),
        ("term_totals", (D * AI2CR_TERMS) * 2),
        ("damage_done", D * 2),
        ("damage_taken", D * 2),
        ("clean_win", I * 2),
        ("timeout", I),
    ]


class NativeCombatRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="combat-reward-")
        cls.source = Path(cls.build.name) / "probe.c"
        cls.library = Path(cls.build.name) / "probe.so"
        cls.source.write_text(
            textwrap.dedent(
                """
                #include "native/combat_reward.h"
                #include <string.h>

                void probe_zero_game(AI2GState *g) {
                    memset(g, 0, sizeof(*g));
                    g->initialized = 1;
                    g->scene.mcfg.stage_width_tiles = 29.8;
                    g->scene.mcfg.stage_height_tiles = 16.8;
                    for (int i = 0; i < 2; ++i) {
                        g->scene.players[i].ms.stocks = 3;
                        g->scene.players[i].weapon = 0;
                        g->combat[i].last_damage_source = -1;
                    }
                }

                void probe_place(AI2GState *g, int i, double x, double y) {
                    g->scene.players[i].ms.x = x;
                    g->scene.players[i].ms.y = y;
                }

                void probe_damage(AI2GState *g, int source, int target, double amount) {
                    g->combat[source].damage_done += amount;
                    g->combat[source].damage_events += 1;
                    g->combat[target].damage_taken_total += amount;
                    g->combat[source].events[g->combat[source].event_count].kind = AI2G_EVENT_DAMAGE;
                    g->combat[source].events[g->combat[source].event_count].source = source;
                    g->combat[source].events[g->combat[source].event_count].target = target;
                    g->combat[source].events[g->combat[source].event_count].amount = amount;
                    g->combat[source].event_count += 1;
                }

                void probe_attack(AI2GState *g, int source) {
                    g->combat[source].attack_starts += 1;
                    g->combat[source].events[g->combat[source].event_count].kind = AI2G_EVENT_ATTACK_STARTED;
                    g->combat[source].events[g->combat[source].event_count].source = source;
                    g->combat[source].event_count += 1;
                }

                void probe_contact(AI2GState *g, int source, int target) {
                    g->combat[source].attack_contacts += 1;
                    g->combat[source].events[g->combat[source].event_count].kind = AI2G_EVENT_ATTACK_CONTACT;
                    g->combat[source].events[g->combat[source].event_count].source = source;
                    g->combat[source].events[g->combat[source].event_count].target = target;
                    g->combat[source].event_count += 1;
                }

                void probe_ko(AI2GState *g, int target) {
                    g->scene.players[target].ms.stocks -= 1;
                    g->combat[target].events[g->combat[target].event_count].kind = AI2G_EVENT_KO;
                    g->combat[target].events[g->combat[target].event_count].source = target;
                    g->combat[target].events[g->combat[target].event_count].target = target;
                    g->combat[target].event_count += 1;
                }

                void probe_weapon(AI2GState *g, int agent, int weapon) {
                    g->scene.players[agent].weapon = weapon;
                }

                int probe_reset(AI2CRState *s, const AI2GState *g, int learner) {
                    return ai2cr_reset(s, g, learner);
                }

                int probe_step(AI2CRState *s, const AI2GState *g, const AI2GStepResult *raw,
                               const AI2CRCoefficients *c, AI2CRBreakdown *b) {
                    return ai2cr_step(s, g, raw, c, b);
                }

                int probe_record(const AI2CRState *s, const AI2GState *g, const AI2GStepResult *raw,
                                 AI2CREpisodeRecord *r) {
                    return ai2cr_episode_record(s, g, raw, r);
                }

                double probe_normalize_legacy_term(int term, double value) {
                    return ai2cr_normalize_legacy_term(term, value);
                }

                int probe_term_requires_magnitude(int term) {
                    return ai2cr_term_requires_magnitude(term);
                }

                size_t probe_state_size(void) { return sizeof(AI2CRState); }
                size_t probe_breakdown_size(void) { return sizeof(AI2CRBreakdown); }
                size_t probe_record_size(void) { return sizeof(AI2CREpisodeRecord); }
                """
            )
        )
        subprocess.run(
            [
                "cc",
                "-std=c99",
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                "-shared",
                "-fPIC",
                "-I",
                str(ROOT),
                str(cls.source),
                str(ROOT / "native/game_episode.c"),
                str(ROOT / "native/navigation_episode.c"),
                "-lm",
                "-o",
                str(cls.library),
            ],
            check=True,
        )
        cls.lib = C.CDLL(str(cls.library))
        import ai2.simulation.native_game_episode as game

        cls.GameState = game.State
        cls.RawResult = game.Result
        cls.GameReference = game.GameReference
        cls.lib.probe_zero_game.argtypes = [C.POINTER(cls.GameState)]
        cls.lib.probe_place.argtypes = [C.POINTER(cls.GameState), I, D, D]
        cls.lib.probe_damage.argtypes = [C.POINTER(cls.GameState), I, I, D]
        cls.lib.probe_attack.argtypes = [C.POINTER(cls.GameState), I]
        cls.lib.probe_contact.argtypes = [C.POINTER(cls.GameState), I, I]
        cls.lib.probe_ko.argtypes = [C.POINTER(cls.GameState), I]
        cls.lib.probe_weapon.argtypes = [C.POINTER(cls.GameState), I, I]
        cls.lib.probe_reset.argtypes = [C.POINTER(State), C.POINTER(cls.GameState), I]
        cls.lib.probe_step.argtypes = [
            C.POINTER(State),
            C.POINTER(cls.GameState),
            C.POINTER(cls.RawResult),
            C.POINTER(Coefficients),
            C.POINTER(Breakdown),
        ]
        cls.lib.probe_record.argtypes = [
            C.POINTER(State),
            C.POINTER(cls.GameState),
            C.POINTER(cls.RawResult),
            C.POINTER(EpisodeRecord),
        ]
        for name in ("probe_reset", "probe_step", "probe_record"):
            getattr(cls.lib, name).restype = I
        cls.lib.probe_normalize_legacy_term.argtypes = [I, D]
        cls.lib.probe_normalize_legacy_term.restype = D
        cls.lib.probe_term_requires_magnitude.argtypes = [I]
        cls.lib.probe_term_requires_magnitude.restype = I
        cls.lib.probe_state_size.restype = C.c_size_t
        cls.lib.probe_breakdown_size.restype = C.c_size_t
        cls.lib.probe_record_size.restype = C.c_size_t

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.game = self.GameState()
        self.raw = self.RawResult()
        self.state = State()
        self.breakdown = Breakdown()
        self.coeff = Coefficients()
        self.coeff.enabled_mask = (1 << AI2CR_TERMS) - 1
        self.coeff.fade = 1.0
        for i in range(AI2CR_TERMS):
            self.coeff.term[i] = 1.0
        self.lib.probe_zero_game(C.byref(self.game))
        self.lib.probe_place(C.byref(self.game), 0, -4.0, 0.0)
        self.lib.probe_place(C.byref(self.game), 1, 4.0, 0.0)
        self.assertEqual(self.lib.probe_reset(C.byref(self.state), C.byref(self.game), 0), 0)
        self.assertEqual(self.lib.probe_state_size(), C.sizeof(State))
        self.assertEqual(self.lib.probe_breakdown_size(), C.sizeof(Breakdown))
        self.assertEqual(self.lib.probe_record_size(), C.sizeof(EpisodeRecord))

    def step(self):
        rc = self.lib.probe_step(
            C.byref(self.state),
            C.byref(self.game),
            C.byref(self.raw),
            C.byref(self.coeff),
            C.byref(self.breakdown),
        )
        self.assertEqual(rc, 0)
        return self.breakdown

    @staticmethod
    def load_combat_phase_rewards():
        phases: dict[str, dict[str, float]] = {}
        current_phase = None
        in_rewards = False
        for raw_line in (ROOT / "config.yaml").read_text().splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            phase_match = re.match(r"^  ([1-4]):\s*$", line)
            if phase_match:
                current_phase = phase_match.group(1)
                phases[current_phase] = {}
                in_rewards = False
                continue
            if current_phase is None:
                continue
            if re.match(r"^    rewards:\s*$", line):
                in_rewards = True
                continue
            if in_rewards:
                if not line.startswith("      "):
                    in_rewards = False
                    continue
                reward_match = re.match(r"^\s+([a-z_]+):\s+(-?[0-9]+(?:\.[0-9]+)?)\s*$", line)
                if reward_match:
                    phases[current_phase][reward_match.group(1)] = float(reward_match.group(2))
        return phases

    def coefficients_from_legacy_rewards(self, rewards: dict[str, float]) -> Coefficients:
        coeff = Coefficients()
        coeff.fade = 1.0
        mapping = {
            "distance": (TERM_APPROACH,),
            "aggression": (TERM_ATTACK_START,),
            "damage_dealt": (TERM_DAMAGE_DEALT,),
            "damage_taken": (TERM_DAMAGE_TAKEN,),
            "net_damage": (TERM_NET_DAMAGE,),
            "win": (TERM_WIN,),
            "knockout": (TERM_KO,),
            "edge_penalty": (TERM_OFFSTAGE, TERM_WALKOFF),
            "clean_win_bonus": (TERM_CLEAN_WIN,),
        }
        for name, value in rewards.items():
            self.assertIn(name, mapping)
            for term in mapping[name]:
                coeff.term[term] = self.lib.probe_normalize_legacy_term(term, value)
                coeff.enabled_mask |= 1 << term
        return coeff

    def test_approach_is_distance_delta_from_learner_perspective_and_mirrors(self):
        self.lib.probe_place(C.byref(self.game), 0, -3.0, 0.0)
        b = self.step()
        self.assertAlmostEqual(b.approach_delta, 1.0)
        self.assertAlmostEqual(b.terms[0][TERM_APPROACH], 1.0)
        self.assertAlmostEqual(b.terms[1][TERM_APPROACH], 1.0)

    def test_approach_perspective_follows_learner_slot_one(self):
        learner_one = State()
        self.assertEqual(self.lib.probe_reset(C.byref(learner_one), C.byref(self.game), 1), 0)
        self.lib.probe_place(C.byref(self.game), 1, 3.0, 0.0)
        self.assertEqual(
            self.lib.probe_step(
                C.byref(learner_one),
                C.byref(self.game),
                C.byref(self.raw),
                C.byref(self.coeff),
                C.byref(self.breakdown),
            ),
            0,
        )
        self.assertAlmostEqual(self.breakdown.approach_delta, 1.0)
        self.assertAlmostEqual(self.breakdown.terms[1][TERM_APPROACH], 1.0)
        self.assertAlmostEqual(self.breakdown.terms[0][TERM_APPROACH], 1.0)

    def test_attack_contact_damage_and_net_damage_are_counted_once_per_step(self):
        self.lib.probe_attack(C.byref(self.game), 0)
        self.lib.probe_contact(C.byref(self.game), 0, 1)
        self.lib.probe_damage(C.byref(self.game), 0, 1, 7.5)
        b = self.step()
        self.assertEqual(b.attack_starts[0], 1)
        self.assertEqual(b.contacts[0], 1)
        self.assertEqual(b.damage_events[0], 1)
        self.assertAlmostEqual(b.terms[0][TERM_ATTACK_START], 1.0)
        self.assertAlmostEqual(b.terms[0][TERM_CONTACT], 1.0)
        self.assertAlmostEqual(b.terms[0][TERM_DAMAGE_DEALT], 7.5)
        self.assertAlmostEqual(b.terms[0][TERM_NET_DAMAGE], 7.5)
        self.assertAlmostEqual(b.terms[1][TERM_DAMAGE_TAKEN], 7.5)
        self.assertAlmostEqual(b.terms[1][TERM_NET_DAMAGE], 0.0)
        b2 = self.step()
        self.assertEqual(b2.damage_events[0], 0)
        self.assertAlmostEqual(b2.terms[0][TERM_DAMAGE_DEALT], 0.0)

    def test_ko_win_loss_clean_win_timeout_and_episode_record(self):
        self.lib.probe_damage(C.byref(self.game), 0, 1, 10.0)
        self.game.scene.players[0].ms.stocks = 2
        self.lib.probe_ko(C.byref(self.game), 1)
        self.raw.terminated = 1
        self.raw.win_mask = 1
        b = self.step()
        self.assertEqual(b.ko_events[1], 1)
        self.assertAlmostEqual(b.terms[0][TERM_KO], 1.0)
        self.assertAlmostEqual(b.terms[1][TERM_KO], -1.0)
        self.assertAlmostEqual(b.terms[0][TERM_WIN], 1.0)
        self.assertAlmostEqual(b.terms[1][TERM_WIN], -1.0)
        self.assertAlmostEqual(b.terms[0][TERM_CLEAN_WIN], 1.0)
        record = EpisodeRecord()
        self.assertEqual(self.lib.probe_record(C.byref(self.state), C.byref(self.game), C.byref(self.raw), C.byref(record)), 0)
        self.assertEqual(record.version, AI2CR_VERSION)
        self.assertEqual(record.terminated, 1)
        self.assertEqual(record.clean_win[0], 1)
        self.assertAlmostEqual(record.term_totals[0][TERM_WIN], 1.0)

        no_damage_state = State()
        self.lib.probe_zero_game(C.byref(self.game))
        self.lib.probe_place(C.byref(self.game), 0, -4.0, 0.0)
        self.lib.probe_place(C.byref(self.game), 1, 4.0, 0.0)
        self.assertEqual(self.lib.probe_reset(C.byref(no_damage_state), C.byref(self.game), 0), 0)
        self.raw = self.RawResult()
        self.raw.terminated = 1
        self.raw.win_mask = 1
        self.assertEqual(
            self.lib.probe_step(C.byref(no_damage_state), C.byref(self.game), C.byref(self.raw), C.byref(self.coeff), C.byref(self.breakdown)),
            0,
        )
        self.assertAlmostEqual(self.breakdown.terms[0][TERM_CLEAN_WIN], 0.0)

        other = State()
        self.lib.probe_zero_game(C.byref(self.game))
        self.lib.probe_place(C.byref(self.game), 0, -4.0, 0.0)
        self.lib.probe_place(C.byref(self.game), 1, 4.0, 0.0)
        self.assertEqual(self.lib.probe_reset(C.byref(other), C.byref(self.game), 0), 0)
        self.raw = self.RawResult()
        self.raw.truncated = 1
        self.coeff.term[TERM_TIMEOUT] = -1.0
        self.assertEqual(
            self.lib.probe_step(C.byref(other), C.byref(self.game), C.byref(self.raw), C.byref(self.coeff), C.byref(self.breakdown)),
            0,
        )
        self.assertAlmostEqual(self.breakdown.terms[0][TERM_TIMEOUT], 1.0)
        self.assertAlmostEqual(self.breakdown.terms[1][TERM_TIMEOUT], 1.0)
        self.assertAlmostEqual(self.breakdown.total[0], -1.0)
        self.assertAlmostEqual(self.breakdown.total[1], -1.0)

    def test_offstage_walkoff_recovery_weapon_and_reset_no_leak(self):
        self.coeff.term[TERM_OFFSTAGE] = -1.0
        self.coeff.term[TERM_WALKOFF] = -1.0
        self.lib.probe_place(C.byref(self.game), 0, 20.0, 0.0)
        b = self.step()
        self.assertEqual(b.offstage[0], 1)
        self.assertEqual(b.walkoff[0], 1)
        self.assertAlmostEqual(b.terms[0][TERM_OFFSTAGE], 1.0)
        self.assertAlmostEqual(b.terms[0][TERM_WALKOFF], 1.0)
        self.assertLess(b.total[0], 0.0)
        self.lib.probe_place(C.byref(self.game), 0, 0.0, 0.0)
        self.lib.probe_weapon(C.byref(self.game), 0, 2)
        b = self.step()
        self.assertEqual(b.recovery[0], 1)
        self.assertEqual(b.weapon_pickups[0], 1)
        self.assertAlmostEqual(b.terms[0][TERM_RECOVERY], 1.0)
        self.assertAlmostEqual(b.terms[0][TERM_WEAPON], 1.0)
        self.lib.probe_weapon(C.byref(self.game), 0, 0)
        b = self.step()
        self.assertEqual(b.weapon_drops[0], 1)
        self.assertAlmostEqual(b.terms[0][TERM_WEAPON], -1.0)

        fresh = State()
        self.assertEqual(self.lib.probe_reset(C.byref(fresh), C.byref(self.game), 1), 0)
        self.assertEqual(fresh.learner_slot, 1)
        self.assertAlmostEqual(fresh.returns[0], 0.0)
        self.assertEqual(fresh.last_offstage[0], 0)

    def test_coefficients_validate_finite_masks_and_fades_fail_closed(self):
        self.coeff.fade = float("nan")
        before = bytes(self.state)
        rc = self.lib.probe_step(
            C.byref(self.state),
            C.byref(self.game),
            C.byref(self.raw),
            C.byref(self.coeff),
            C.byref(self.breakdown),
        )
        self.assertNotEqual(rc, 0)
        self.assertEqual(bytes(self.state), before)
        self.coeff.fade = 1.0
        self.coeff.enabled_mask = 1 << AI2CR_TERMS
        rc = self.lib.probe_step(
            C.byref(self.state),
            C.byref(self.game),
            C.byref(self.raw),
            C.byref(self.coeff),
            C.byref(self.breakdown),
        )
        self.assertNotEqual(rc, 0)
        self.assertEqual(bytes(self.state), before)
        self.coeff.enabled_mask = (1 << AI2CR_TERMS) - 1
        self.coeff.term[TERM_KO] = -3.0
        rc = self.lib.probe_step(
            C.byref(self.state),
            C.byref(self.game),
            C.byref(self.raw),
            C.byref(self.coeff),
            C.byref(self.breakdown),
        )
        self.assertNotEqual(rc, 0)
        self.assertEqual(bytes(self.state), before)
        self.coeff.term[TERM_KO] = 1.0
        self.coeff.term[TERM_WIN] = -1.0
        rc = self.lib.probe_step(
            C.byref(self.state),
            C.byref(self.game),
            C.byref(self.raw),
            C.byref(self.coeff),
            C.byref(self.breakdown),
        )
        self.assertNotEqual(rc, 0)
        self.assertEqual(bytes(self.state), before)
        self.coeff.term[TERM_WIN] = 1.0
        self.coeff.term[TERM_DAMAGE_TAKEN] = -0.5
        self.coeff.term[TERM_OFFSTAGE] = -1.0
        rc = self.lib.probe_step(
            C.byref(self.state),
            C.byref(self.game),
            C.byref(self.raw),
            C.byref(self.coeff),
            C.byref(self.breakdown),
        )
        self.assertEqual(rc, 0)

    def test_legacy_config_mapping_has_no_ambiguous_double_signs(self):
        rewards_by_phase = self.load_combat_phase_rewards()
        self.assertEqual(set(rewards_by_phase), {"1", "2", "3", "4"})
        for term in (TERM_APPROACH, TERM_KO, TERM_WIN, TERM_WEAPON):
            self.assertEqual(self.lib.probe_term_requires_magnitude(term), 1)
        for term in (TERM_DAMAGE_TAKEN, TERM_TIMEOUT, TERM_OFFSTAGE, TERM_WALKOFF):
            self.assertEqual(self.lib.probe_term_requires_magnitude(term), 0)

        for phase, rewards in rewards_by_phase.items():
            coeff = self.coefficients_from_legacy_rewards(rewards)
            if phase == "1":
                self.assertIn(rewards["knockout"], {-3.0, 3.0})
                self.assertAlmostEqual(coeff.term[TERM_KO], 3.0)
            self.assertLessEqual(coeff.term[TERM_DAMAGE_TAKEN], 0.0)
            self.assertLess(coeff.term[TERM_OFFSTAGE], 0.0)
            self.assertLess(coeff.term[TERM_WALKOFF], 0.0)
            before = bytes(self.state)
            rc = self.lib.probe_step(
                C.byref(self.state),
                C.byref(self.game),
                C.byref(self.raw),
                C.byref(coeff),
                C.byref(self.breakdown),
            )
            self.assertEqual(rc, 0, phase)
            self.assertNotEqual(bytes(self.state), before)

    def test_real_step_ko_precedes_terminal_win_and_record_precedes_reset(self):
        ref = self.GameReference(self.library)
        ref.cfg.time_limit_steps = 9000
        ref.reset_generated(17)
        reward_state = State()
        self.assertEqual(self.lib.probe_reset(C.byref(reward_state), C.byref(ref.state), 0), 0)

        ref.place_player(0, 15.0, 0.0)
        ref.scene.players[0].ms.stocks = 1
        first = ref.step_generated([[0.0] * 10, [0.0] * 10])
        self.assertFalse(first.terminated)
        self.assertEqual(ref.scene.players[0].ms.stocks, 0)
        first_breakdown = Breakdown()
        self.assertEqual(
            self.lib.probe_step(C.byref(reward_state), C.byref(ref.state), C.byref(first), C.byref(self.coeff), C.byref(first_breakdown)),
            0,
        )
        self.assertEqual(first_breakdown.ko_events[0], 1)
        self.assertAlmostEqual(first_breakdown.terms[1][TERM_KO], 1.0)
        self.assertAlmostEqual(first_breakdown.terms[0][TERM_WIN], 0.0)

        second = ref.step_generated([[0.0] * 10, [0.0] * 10])
        self.assertTrue(second.terminated)
        self.assertEqual(second.win_mask, 2)
        second_breakdown = Breakdown()
        self.assertEqual(
            self.lib.probe_step(C.byref(reward_state), C.byref(ref.state), C.byref(second), C.byref(self.coeff), C.byref(second_breakdown)),
            0,
        )
        self.assertAlmostEqual(second_breakdown.terms[1][TERM_WIN], 1.0)
        record = EpisodeRecord()
        self.assertEqual(
            self.lib.probe_record(C.byref(reward_state), C.byref(ref.state), C.byref(second), C.byref(record)),
            0,
        )
        self.assertEqual(record.length, 2)
        self.assertEqual(record.terminated, 1)
        self.assertEqual(record.win_mask, 2)
        self.assertAlmostEqual(record.term_totals[1][TERM_KO], 1.0)
        self.assertAlmostEqual(record.term_totals[1][TERM_WIN], 1.0)

        ref.reset_generated(19)
        reset_state = State()
        self.assertEqual(self.lib.probe_reset(C.byref(reset_state), C.byref(ref.state), 0), 0)
        self.assertEqual(reset_state.tick, 0)
        self.assertAlmostEqual(reset_state.returns[0], 0.0)
        self.assertAlmostEqual(reset_state.term_totals[1][TERM_WIN], 0.0)

    def test_cuda_header_scaffold_compiles_when_nvcc_available(self):
        nvcc = shutil.which("nvcc")
        if not nvcc:
            self.skipTest("nvcc not available")
        source = Path(self.build.name) / "combat_reward_cuda_probe.cu"
        obj = Path(self.build.name) / "combat_reward_cuda_probe.o"
        source.write_text(
            textwrap.dedent(
                """
                #include "native/combat_reward.h"

                __global__ void combat_reward_probe(AI2CRState *state, AI2GState *game,
                                                    AI2GStepResult *raw,
                                                    AI2CRCoefficients *coeff,
                                                    AI2CRBreakdown *breakdown) {
                    if (threadIdx.x == 0) {
                        (void)ai2cr_reset(state, game, 0);
                        (void)ai2cr_step(state, game, raw, coeff, breakdown);
                    }
                }
                """
            )
        )
        subprocess.run(
            [
                nvcc,
                "-std=c++17",
                "--device-c",
                "--fmad=false",
                "-I",
                str(ROOT),
                str(source),
                "-o",
                str(obj),
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
