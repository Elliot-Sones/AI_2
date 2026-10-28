"""Raw two-player resident lanes; CPU oracle and opt-in CUDA parity tests.

No policy or checkpoint is loaded. Declared fixtures are initialized once;
ordinary trajectories, combat, weapon interactions, and replay stay resident.
"""
import ctypes as C
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import ai2.simulation.native_game_episode as game
import ai2.simulation.native_navigation_episode as nav

ROOT = Path(__file__).resolve().parents[1]
INT32, U, B, F, D, V = C.c_int32, C.c_uint64, C.c_uint8, C.c_float, C.c_double, C.c_void_p
AI2CR_TERMS = 14
AI2CR_TERM_APPROACH = 0
AI2CR_TERM_ATTACK_START = 1
AI2CR_TERM_DAMAGE_DEALT = 3
AI2CR_TERM_DAMAGE_TAKEN = 4
AI2CR_TERM_NET_DAMAGE = 5
AI2CR_TERM_KO = 6
AI2CR_TERM_WIN = 7
AI2CR_TERM_CLEAN_WIN = 8
AI2CR_TERM_TIMEOUT = 9
AI2CR_TERM_OFFSTAGE = 10
AI2CR_TERM_WALKOFF = 11


class RewardCoefficients(C.Structure):
    _fields_ = [("term", D * AI2CR_TERMS), ("enabled_mask", C.c_uint32), ("fade", D)]


class RewardState(C.Structure):
    _fields_ = [
        (n, INT32) for n in ("version", "initialized", "learner_slot", "tick")
    ] + [
        ("initial_stocks", INT32 * 2), ("last_stocks", INT32 * 2),
        ("last_weapon", INT32 * 2), ("last_offstage", INT32 * 2),
        ("clean_stock", INT32 * 2), ("last_x", D * 2), ("last_y", D * 2),
        ("last_distance", D), ("last_damage_done", D * 2),
        ("last_damage_taken", D * 2), ("base_damage_done", D * 2),
        ("base_damage_taken", D * 2), ("last_attack_starts", U * 2),
        ("last_attack_contacts", U * 2), ("last_damage_events", U * 2),
        ("returns", D * 2), ("term_totals", (D * AI2CR_TERMS) * 2),
    ]


class RewardBreakdown(C.Structure):
    _fields_ = [
        (n, INT32) for n in (
            "version", "learner_slot", "tick", "terminated", "truncated", "win_mask"
        )
    ] + [
        ("terms", (D * AI2CR_TERMS) * 2), ("total", D * 2),
        ("approach_delta", D), ("distance", D),
        ("attack_starts", INT32 * 2), ("contacts", INT32 * 2),
        ("damage_events", INT32 * 2), ("ko_events", INT32 * 2),
        ("weapon_pickups", INT32 * 2), ("weapon_drops", INT32 * 2),
        ("offstage", INT32 * 2), ("walkoff", INT32 * 2),
        ("recovery", INT32 * 2),
    ]


class RewardEpisodeRecord(C.Structure):
    _fields_ = [
        (n, INT32) for n in (
            "version", "learner_slot", "length", "terminated", "truncated", "win_mask"
        )
    ] + [
        ("stocks", INT32 * 2), ("returns", D * 2),
        ("term_totals", (D * AI2CR_TERMS) * 2),
        ("damage_done", D * 2), ("damage_taken", D * 2),
        ("clean_win", INT32 * 2), ("timeout", INT32),
    ]


class Lane(C.Structure):
    _fields_ = [("game", game.State), ("rng", nav.Rng), ("returns", D * 2),
                ("last_obs", F * 128), ("valid_steps", U),
                ("reward_cfg", RewardCoefficients), ("reward_state", RewardState),
                ("last_reward", RewardBreakdown)] + [
                    (n, INT32) for n in ("reward_learner_slot", "length", "episodes", "error", "active")]


class Record(C.Structure):
    _fields_ = [(n, INT32) for n in ("lane", "length", "terminated", "truncated", "win_mask", "episode_index")] + [
        ("stocks", INT32 * 2), ("returns", D * 2), ("damage_done", D * 2), ("damage_taken", D * 2),
        ("reward", RewardEpisodeRecord)]


def ptr(array, kind=F):
    return array.ctypes.data_as(C.POINTER(kind))


class Runtime:
    def __init__(self, lib, ref, games=2, capacity=8, cuda=False):
        self.lib, self.n, self.cuda = lib, games, cuda
        prefix = "ai2g_lanes_cuda_" if cuda else "ai2g_lanes_cpu_"
        self.f = {name: getattr(lib, prefix + name) for name in (
            "create", "destroy", "reset", "step", "drain", "lane_get", "lane_set",
            "replay_reset", "replay_step", "rng_status", "configure_reward")}
        extra = [V] if cuda else []
        specs = {
            "create": [C.POINTER(nav.Config), C.POINTER(game.PlatformTable), INT32, INT32, D, C.POINTER(INT32)],
            "destroy": [V], "reset": [V, C.POINTER(U), C.POINTER(B), C.POINTER(F)] + extra,
            "step": [V] + [C.POINTER(F)] * 4 + [C.POINTER(B)] * 2 + extra,
            "drain": [V, C.POINTER(Record), INT32, C.POINTER(INT32), C.POINTER(INT32), C.POINTER(INT32), C.POINTER(U)] + extra,
            "lane_get": [V, INT32, C.POINTER(Lane)] + extra,
            "lane_set": [V, INT32, C.POINTER(Lane)] + extra,
            "replay_reset": [V, INT32, C.POINTER(nav.Draw), INT32, C.POINTER(F)] + extra,
            "replay_step": [V, INT32, C.POINTER(F), C.POINTER(nav.Draw), INT32] + [C.POINTER(F)] * 3 + [C.POINTER(B)] * 2 + extra,
            "rng_status": [V, INT32, C.POINTER(INT32), C.POINTER(INT32), C.POINTER(INT32), C.POINTER(INT32)] + extra,
            "configure_reward": [V, INT32, C.c_uint32, D, C.POINTER(D)] + extra,
        }
        for name, args in specs.items():
            self.f[name].argtypes = args
            self.f[name].restype = V if name == "create" else INT32
        for name, cls in (("lane", Lane), ("record", Record)):
            fn = getattr(lib, prefix + name + "_size")
            fn.restype = C.c_size_t
            assert fn() == C.sizeof(cls), name
        code = INT32()
        self.h = self.f["create"](C.byref(ref.cfg), C.byref(ref.table), games, capacity, 48, C.byref(code))
        assert self.h and not code.value, code.value
        self.obs = np.zeros((games, 2, 64), np.float32)
        self.terminal = np.zeros_like(self.obs)
        self.rewards = np.zeros((games, 2), np.float32)
        self.terminated = np.zeros(games, np.uint8)
        self.truncated = np.zeros(games, np.uint8)
        self.capacity = capacity
        if cuda:
            import torch
            self.torch = torch
            self.stream = torch.cuda.Stream()
            self.extra = [V(self.stream.cuda_stream)]
            self.buffers = [torch.empty(a.shape, dtype=torch.uint8 if a.dtype == np.uint8 else torch.float32,
                                        device="cuda") for a in self.outputs]
        else:
            self.extra = []

    @property
    def outputs(self):
        return [self.obs, self.terminal, self.rewards, self.terminated, self.truncated]

    def pointers(self):
        if self.cuda:
            return [C.cast(b.data_ptr(), C.POINTER(B if index > 2 else F)) for index, b in enumerate(self.buffers)]
        return [ptr(a, B if index > 2 else F) for index, a in enumerate(self.outputs)]

    def copy_outputs(self):
        if self.cuda:
            self.stream.synchronize()
            for target, source in zip(self.outputs, self.buffers):
                np.copyto(target, source.cpu().numpy())

    def reset(self, seeds=(7, 19), mask=None):
        seeds = np.asarray(seeds, np.uint64)
        mask = None if mask is None else np.asarray(mask, np.uint8)
        rc = self.f["reset"](self.h, ptr(seeds, U), None if mask is None else ptr(mask, B), self.pointers()[0], *self.extra)
        self.copy_outputs()
        return rc

    def step(self, actions=None):
        actions = np.zeros((self.n, 2, 10), np.float32) if actions is None else np.asarray(actions, np.float32)
        if self.cuda:
            with self.torch.cuda.stream(self.stream):
                device_actions = self.torch.as_tensor(actions, device="cuda")
            action_ptr = C.cast(device_actions.data_ptr(), C.POINTER(F))
        else:
            action_ptr = ptr(actions)
        rc = self.f["step"](self.h, action_ptr, *self.pointers(), *self.extra)
        self.copy_outputs()
        return rc

    def get(self, index):
        lane = Lane()
        assert self.f["lane_get"](self.h, index, C.byref(lane), *self.extra) == 0
        return lane

    def put(self, index, lane):
        return self.f["lane_set"](self.h, index, C.byref(lane), *self.extra)

    def drain(self, capacity=None):
        capacity = self.capacity if capacity is None else capacity
        rows = (Record * max(1, capacity))()
        count, overflow = INT32(), INT32()
        errors, steps = (INT32 * self.n)(), (U * self.n)()
        rc = self.f["drain"](self.h, rows, capacity, C.byref(count), C.byref(overflow), errors, steps, *self.extra)
        return rc, list(rows)[:count.value], overflow.value, list(errors), list(steps)

    def replay(self, index, draws):
        arr = (nav.Draw * len(draws))(*(nav.Draw(k, 0, a, b, value) for k, a, b, value in draws))
        out = self.pointers()[0]
        address = C.cast(out, V).value + index * 128 * C.sizeof(F)
        rc = self.f["replay_reset"](self.h, index, arr, len(arr), C.cast(address, C.POINTER(F)), *self.extra)
        self.copy_outputs()
        return rc

    def rng_status(self, index):
        mode, cursor, count, error = INT32(), INT32(), INT32(), INT32()
        rc = self.f["rng_status"](self.h, index, C.byref(mode), C.byref(cursor),
                                  C.byref(count), C.byref(error), *self.extra)
        return rc, (mode.value, cursor.value, count.value, error.value)

    def configure_reward(self, terms, enabled_mask, fade=1.0, learner_slot=0):
        terms = np.asarray(terms, np.float64)
        assert terms.shape == (AI2CR_TERMS,)
        return self.f["configure_reward"](self.h, learner_slot, enabled_mask, fade, ptr(terms, D), *self.extra)

    def replay_step(self, index, actions, draws):
        actions = np.asarray(actions, np.float32)
        arr = (nav.Draw * len(draws))(*(nav.Draw(k, 0, a, b, value) for k, a, b, value in draws))
        if self.cuda:
            with self.torch.cuda.stream(self.stream):
                device_actions = self.torch.as_tensor(actions, device="cuda")
            action_ptr = C.cast(device_actions.data_ptr(), C.POINTER(F))
        else:
            action_ptr = ptr(actions)
        offsets = [index * 128 * C.sizeof(F), index * 128 * C.sizeof(F), index * 2 * C.sizeof(F),
                   index * C.sizeof(B), index * C.sizeof(B)]
        outputs = [
            C.cast(C.cast(pointer, V).value + offset, C.POINTER(B if pos > 2 else F))
            for pos, (pointer, offset) in enumerate(zip(self.pointers(), offsets))
        ]
        rc = self.f["replay_step"](self.h, index, action_ptr, arr, len(arr), *outputs, *self.extra)
        self.copy_outputs()
        return rc

    def close(self):
        if self.h:
            rc = self.f["destroy"](self.h)
            self.h = None
            assert not rc, rc


def reset_tape(side, phase):
    return [(1, 1, 0, side)] + [(kind, a, b, value) for _ in range(2) for kind, a, b, value in (
        (2, 2.6, 6.5, 4.5), (3, 500, 700, 600), (3, 0, 350, phase))]


class HostLaneTests(unittest.TestCase):
    cuda = False

    @classmethod
    def setUpClass(cls):
        assert (ROOT / "native/game_lanes.h").is_file(), "raw full-game resident lane header is missing"
        cls.build = tempfile.TemporaryDirectory(prefix="game-lanes-test-")
        cls.library = Path(cls.build.name) / "lanes.so"
        subprocess.run(["cc", "-std=c99", "-O2", "-ffp-contract=off", "-fno-fast-math", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", str(ROOT / "native/game_lanes.c"), str(ROOT / "native/game_episode.c"),
                        str(ROOT / "native/navigation_episode.c"), "-lm", "-o", str(cls.library)], check=True)
        cls.ref = game.GameReference(cls.library)
        cls.lib = C.CDLL(str(cls.library))

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.runtime = Runtime(self.lib, self.ref, cuda=self.cuda)
        self.addCleanup(self.runtime.close)

    def test_independent_generated_lanes_match_game_reference_without_state_injection(self):
        h = self.runtime
        self.assertEqual(h.reset(), 0)
        refs = [game.GameReference(self.library) for _ in range(2)]
        for index, seed in enumerate((7, 19)):
            refs[index].reset_generated(seed, index)
            np.testing.assert_array_equal(h.obs[index], refs[index].observe().astype(np.float32))
        actions = np.zeros((2, 2, 10), np.float32)
        for tick in range(240):
            actions.fill(0)
            actions[0, 0, 3 if tick < 50 else 1] = 1
            actions[1, 1, 4] = tick % 40 < 10
            actions[0, 1, 7] = tick % 23 == 0
            self.assertEqual(h.step(actions), 0)
            for index, ref in enumerate(refs):
                out = ref.step_generated(actions[index])
                self.assertFalse(out.terminated or out.truncated)
                np.testing.assert_array_equal(h.obs[index], np.asarray(out.obs, np.float32))
                np.testing.assert_array_equal(h.rewards[index], np.asarray(out.reward, np.float32))
                self.assertEqual(bytes(h.get(index).game), bytes(ref.state))
        self.assertEqual(h.drain()[4], [240, 240])

    def test_resident_combat_weapons_and_ko_match_cpu_through_autoreset(self):
        # Six positive-contact lanes cover every weapon from both actor slots.
        # No hit, damage, attack runtime, or per-tick position is injected.
        fixtures = [(f"weapon-{weapon}-slot-{slot}", weapon, slot)
                    for weapon in range(3) for slot in range(2)]
        names = [row[0] for row in fixtures] + ["simultaneous", "drop-pickup", "ko-respawn", "last-stock-ko"]
        count, ticks = len(names), 200
        ref = game.GameReference(self.library)
        ref.cfg.time_limit_steps = 160
        host = Runtime(C.CDLL(str(self.library)), ref, games=count, capacity=count * 3)
        candidate = Runtime(self.lib, ref, games=count, capacity=count * 3, cuda=self.cuda)
        self.addCleanup(host.close)
        self.addCleanup(candidate.close)
        seeds = np.arange(17, 17 + count, dtype=np.uint64)
        self.assertEqual(host.reset(seeds), 0)
        self.assertEqual(candidate.reset(seeds), 0)
        for index, name in enumerate(names):
            lane = host.get(index)
            positions = [(4., .2), (4.9, .2)]
            if name == "drop-pickup":
                positions = [(4., .2), (-4., .2)]
            elif name in ("ko-respawn", "last-stock-ko"):
                positions[0] = (15., 0.)
            for slot, position in enumerate(positions):
                self.assertEqual(ref.lib.ai2g_fixture_player_position(C.byref(lane.game), slot, *position), 0)
                player = lane.game.scene.players[slot]
                player.ms.facing = 1 if slot == 0 else -1
                player.ms.state_id = 1
            if index < len(fixtures):
                _, weapon, actor = fixtures[index]
                lane.game.scene.players[actor].weapon = weapon
            elif name == "drop-pickup":
                for slot in range(2):
                    lane.game.scene.players[slot].weapon = slot + 1
            elif name == "last-stock-ko":
                lane.game.scene.players[0].ms.stocks = 1
            initial_obs = np.empty((2, 64), np.float64)
            ref.lib.ai2g_observe_export(C.byref(lane.game), ptr(initial_obs, D))
            lane.last_obs[:] = initial_obs.astype(np.float32).ravel()
            # Exactly one initial upload per lane, before the first game tick.
            self.assertEqual(host.put(index, lane), 0)
            self.assertEqual(candidate.put(index, lane), 0)

        actions = np.zeros((ticks, count, 2, 10), np.float32)
        for index, (_, _, actor) in enumerate(fixtures):
            actions[:, index, actor, 7] = 1
        actions[:, 6, :, 7] = 1  # Both players initiate on the same tick.
        actions[[0, 20], 7, :, 5] = 1  # Drop, release, then pick up after the lock.
        contacts = np.zeros((count, 2), np.uint64)
        damage_events = np.zeros_like(contacts)
        damage_done = np.zeros((count, 2), np.float64)
        events = [set() for _ in names]
        weapon_events = [set() for _ in names]
        simultaneous_starts = False
        for tick, batch in enumerate(actions):
            self.assertEqual(host.step(batch), 0, f"host tick {tick + 1}")
            self.assertEqual(candidate.step(batch), 0, f"candidate tick {tick + 1}")
            for label, actual, expected in zip(
                    ("obs", "terminal", "rewards", "terminated", "truncated"), candidate.outputs, host.outputs):
                self.assertEqual(actual.tobytes(), expected.tobytes(), f"{label}, tick {tick + 1}")
            for index, name in enumerate(names):
                actual, expected = candidate.get(index), host.get(index)
                left, right = bytes(actual), bytes(expected)
                if left != right:
                    offset = next(i for i, (a, b) in enumerate(zip(left, right)) if a != b)
                    self.fail(f"{name}, tick {tick + 1}: full lane state differs at byte {offset}")
                started_this_tick = set()
                for slot, combat in enumerate(actual.game.combat):
                    contacts[index, slot] = max(contacts[index, slot], combat.attack_contacts)
                    damage_events[index, slot] = max(damage_events[index, slot], combat.damage_events)
                    damage_done[index, slot] = max(damage_done[index, slot], combat.damage_done)
                    events[index].update((event.kind, event.source) for event in combat.events[:combat.event_count])
                    started_this_tick.update(event.source for event in combat.events[:combat.event_count]
                                             if event.kind == 1)
                weapons = actual.game.weapons
                weapon_events[index].update((event.kind, event.agent, event.weapon)
                                            for event in weapons.events[:weapons.event_count])
                if index == 6 and started_this_tick == {0, 1}:
                    simultaneous_starts = True

        host_result, result = host.drain(), candidate.drain()
        self.assertEqual((result[0], result[2], result[3], result[4]), (0, 0, [0] * count, [ticks] * count))
        self.assertEqual((result[0], result[2:]), (host_result[0], host_result[2:]))
        def record_key(record):
            return record.lane, record.episode_index
        records = sorted(result[1], key=record_key)
        self.assertEqual([bytes(row) for row in records],
                         [bytes(row) for row in sorted(host_result[1], key=record_key)])
        for index, (name, _, actor) in enumerate(fixtures):
            self.assertGreater(contacts[index, actor], 0, name)
            self.assertGreater(damage_events[index, actor], 0, name)
            self.assertGreater(damage_done[index, actor], 0, name)
            self.assertGreater(sum(row.damage_done[actor] for row in records if row.lane == index), 0, name)
        self.assertTrue(simultaneous_starts)
        self.assertGreater(int(contacts[6].sum()), 0, "simultaneous attacks must actually connect")
        self.assertGreater(damage_done[6].sum(), 0, "simultaneous attacks must actually deal damage")
        self.assertTrue({(1, 0, 1), (2, 0, 1), (1, 1, 2), (2, 1, 2)} <= weapon_events[7], weapon_events[7])
        self.assertTrue({(7, 0), (8, 0)} <= events[8], "KO fixture must lose a stock and actually respawn")
        self.assertIn((7, 0), events[9], "last stock must be lost through the game KO transition")
        self.assertTrue(any(row.lane == 9 and row.terminated and row.win_mask == 2
                            and list(row.stocks) == [0, 3] for row in records))
        self.assertTrue(all(any(row.lane == index and row.truncated and not row.terminated
                                for row in records) for index in range(count)), "every lane must exercise autoreset")

    def test_selective_reset_preserves_other_lane_and_pending_record(self):
        h = self.runtime
        self.assertEqual(h.reset(), 0)
        h.step()
        before = bytes(h.get(1))
        self.assertEqual(h.reset((99, 101), (1, 0)), 0)
        self.assertEqual(bytes(h.get(1)), before)
        self.ref.reset_generated(99, 0)
        np.testing.assert_array_equal(h.obs[0], self.ref.observe().astype(np.float32))
        self.assertEqual(h.drain()[4], [1, 1])

    def test_seed_values_keep_all_64_bits(self):
        h = self.runtime
        seeds = (2**63 + 7, 2**64 - 1)
        self.assertEqual(h.reset(seeds), 0)
        for index, seed in enumerate(seeds):
            self.ref.reset_generated(seed, index)
            np.testing.assert_array_equal(h.obs[index], self.ref.observe().astype(np.float32))
            self.assertEqual(list(h.get(index).rng.s), list(self.ref.rng.s))

    def test_true_terminal_observation_record_and_autoreset(self):
        h = self.runtime
        h.reset()
        lane = h.get(0)
        lane.game.scene.players[0].ms.stocks = 0
        lane.game.combat[0].damage_done = 13.5
        lane.game.combat[1].damage_taken_total = 13.5
        self.assertEqual(h.put(0, lane), 0)
        self.assertEqual(h.step(), 0)
        self.assertEqual(h.terminated.tolist(), [1, 0])
        self.assertEqual(h.truncated.tolist(), [0, 0])
        self.assertEqual(h.terminal[0, 0, 13], 0)
        self.assertEqual(h.obs[0, 0, 13], 3)
        rc, rows, overflow, errors, steps = h.drain()
        self.assertEqual((rc, overflow, errors, steps), (0, 0, [0, 0], [1, 1]))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row.lane, row.length, row.terminated, row.win_mask, row.episode_index), (0, 1, 1, 2, 0))
        self.assertEqual(list(row.stocks), [0, 3])
        self.assertEqual(list(row.damage_done), [13.5, 0])
        self.assertEqual(list(row.damage_taken), [0, 13.5])
        self.assertEqual(h.drain()[1], [])

    def test_record_returns_use_double_accumulation_and_autoreset_episode_indices(self):
        h = self.runtime
        h.reset()
        lane = h.get(0)
        lane.returns[0], lane.returns[1] = 1.234567890123, -2.345678901234
        lane.length = 7
        lane.game.scene.players[0].ms.stocks = 0
        # Reset-body previous x produces a known exact zero reward this tick.
        h.put(0, lane)
        h.step()
        row = h.drain()[1][0]
        self.assertEqual(list(row.returns), [1.234567890123, -2.345678901234])
        self.assertEqual((row.length, row.episode_index), (8, 0))
        lane = h.get(0)
        self.assertEqual(list(lane.returns), [0.0, 0.0])
        lane.game.scene.players[1].ms.stocks = 0
        h.put(0, lane)
        h.step()
        row = h.drain()[1][0]
        self.assertEqual((row.length, row.episode_index, row.win_mask), (1, 1, 1))

    def test_combat_reward_mode_shapes_records_and_resets_after_autoreset(self):
        h = self.runtime
        h.reset()
        terms = np.zeros(AI2CR_TERMS, np.float64)
        terms[AI2CR_TERM_APPROACH] = 1.0
        terms[AI2CR_TERM_KO] = -3.0  # legacy phase-1 sign must normalize to +3 magnitude
        terms[AI2CR_TERM_WIN] = 5.0
        terms[AI2CR_TERM_CLEAN_WIN] = 2.0
        terms[AI2CR_TERM_TIMEOUT] = -0.25
        enabled = ((1 << AI2CR_TERM_APPROACH) | (1 << AI2CR_TERM_KO) |
                   (1 << AI2CR_TERM_WIN) | (1 << AI2CR_TERM_CLEAN_WIN) |
                   (1 << AI2CR_TERM_TIMEOUT))
        self.assertEqual(h.configure_reward(terms, enabled, learner_slot=0), 0)
        lane = h.get(0)
        lane.game.scene.players[0].ms.stocks = 0
        lane.game.combat[1].damage_done = 12.0
        lane.game.combat[0].damage_taken_total = 12.0
        self.assertEqual(h.put(0, lane), 0)
        self.assertEqual(h.step(), 0)
        self.assertEqual(h.terminated.tolist(), [1, 0])
        self.assertEqual(h.truncated.tolist(), [0, 0])
        self.assertEqual(h.rewards[0, 1], 7.0)
        self.assertEqual(h.rewards[0, 0], -5.0)
        rows = h.drain()[1]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row.reward.version, row.reward.length, row.reward.terminated,
                          row.reward.truncated, row.reward.win_mask), (1, 1, 1, 0, 2))
        self.assertEqual(h.get(0).reward_cfg.term[AI2CR_TERM_KO], 3.0)
        self.assertEqual(row.reward.term_totals[0][AI2CR_TERM_KO], 0.0)
        self.assertEqual(row.reward.term_totals[1][AI2CR_TERM_KO], 0.0)
        self.assertEqual(row.reward.term_totals[0][AI2CR_TERM_WIN], -1.0)
        self.assertEqual(row.reward.term_totals[1][AI2CR_TERM_WIN], 1.0)
        self.assertEqual(row.reward.term_totals[1][AI2CR_TERM_CLEAN_WIN], 1.0)
        self.assertEqual(list(row.reward.returns), [-5.0, 7.0])
        lane = h.get(0)
        self.assertEqual((lane.reward_state.initialized, lane.reward_state.tick), (1, 0))
        self.assertEqual(list(lane.reward_state.returns), [0.0, 0.0])

    def test_combat_reward_mode_records_timeout_and_keeps_raw_mode_disabled(self):
        raw = Runtime(self.lib, self.ref, cuda=self.cuda)
        shaped = Runtime(self.lib, self.ref, cuda=self.cuda)
        self.addCleanup(raw.close)
        self.addCleanup(shaped.close)
        self.assertEqual(raw.reset(), 0)
        self.assertEqual(shaped.reset(), 0)
        for h in (raw, shaped):
            lane = h.get(0)
            lane.game.max_timesteps = 1
            h.put(0, lane)
        terms = np.zeros(AI2CR_TERMS, np.float64)
        terms[AI2CR_TERM_TIMEOUT] = -0.5
        self.assertEqual(shaped.configure_reward(terms, 1 << AI2CR_TERM_TIMEOUT), 0)
        self.assertEqual(raw.step(), 0)
        self.assertEqual(shaped.step(), 0)
        self.assertEqual(raw.truncated[0], 1)
        self.assertEqual(shaped.truncated[0], 1)
        self.assertEqual(raw.rewards[0].tolist(), [0.0, 0.0])
        self.assertEqual(shaped.rewards[0].tolist(), [-0.5, -0.5])
        raw_row = raw.drain()[1][0]
        shaped_row = shaped.drain()[1][0]
        self.assertEqual(raw_row.reward.version, 0)
        self.assertEqual(shaped_row.reward.timeout, 1)
        self.assertEqual(list(shaped_row.reward.returns), [-0.5, -0.5])

    def test_combat_reward_nonterminal_terms_match_transparent_oracle(self):
        h = self.runtime
        h.reset()
        terms = np.zeros(AI2CR_TERMS, np.float64)
        terms[AI2CR_TERM_ATTACK_START] = 0.5
        terms[AI2CR_TERM_DAMAGE_DEALT] = 1.0
        terms[AI2CR_TERM_DAMAGE_TAKEN] = -1.0
        terms[AI2CR_TERM_NET_DAMAGE] = 2.0
        terms[AI2CR_TERM_OFFSTAGE] = -0.25
        terms[AI2CR_TERM_WALKOFF] = -0.75
        enabled = ((1 << AI2CR_TERM_ATTACK_START) | (1 << AI2CR_TERM_DAMAGE_DEALT) |
                   (1 << AI2CR_TERM_DAMAGE_TAKEN) | (1 << AI2CR_TERM_NET_DAMAGE) |
                   (1 << AI2CR_TERM_OFFSTAGE) | (1 << AI2CR_TERM_WALKOFF))
        self.assertEqual(h.configure_reward(terms, enabled), 0)
        lane = h.get(0)
        self.assertEqual(self.ref.lib.ai2g_fixture_player_position(C.byref(lane.game), 0, 20.0, 0.0), 0)
        lane.game.combat[0].attack_starts += 1
        lane.game.combat[0].damage_done += 4.0
        lane.game.combat[1].damage_taken_total += 4.0
        self.assertEqual(h.put(0, lane), 0)
        self.assertEqual(h.step(), 0)
        lane = h.get(0)
        self.assertEqual(lane.last_reward.attack_starts[0], 1)
        self.assertEqual(lane.last_reward.offstage[0], 1)
        self.assertEqual(lane.last_reward.walkoff[0], 1)
        self.assertEqual(lane.last_reward.terms[0][AI2CR_TERM_DAMAGE_DEALT], 4.0)
        self.assertEqual(lane.last_reward.terms[1][AI2CR_TERM_DAMAGE_TAKEN], 4.0)
        self.assertEqual(lane.last_reward.terms[0][AI2CR_TERM_NET_DAMAGE], 4.0)
        self.assertEqual(lane.last_reward.terms[1][AI2CR_TERM_NET_DAMAGE], 0.0)
        self.assertEqual(h.rewards[0, 0], 0.5 + 4.0 + 8.0 - 0.25 - 0.75)
        self.assertEqual(h.rewards[0, 1], -4.0)

    def test_combat_reward_learner_slot_one_keeps_symmetric_approach_terms(self):
        h = self.runtime
        h.reset()
        terms = np.zeros(AI2CR_TERMS, np.float64)
        terms[AI2CR_TERM_APPROACH] = 1.0
        self.assertEqual(h.configure_reward(terms, 1 << AI2CR_TERM_APPROACH, learner_slot=1), 0)
        lane = h.get(0)
        self.assertEqual(lane.reward_state.learner_slot, 1)
        self.assertEqual(self.ref.lib.ai2g_fixture_player_position(C.byref(lane.game), 0, 4.0, 0.0), 0)
        self.assertEqual(self.ref.lib.ai2g_fixture_player_position(C.byref(lane.game), 1, 5.0, 0.0), 0)
        self.assertEqual(h.put(0, lane), 0)
        self.assertEqual(h.configure_reward(terms, 1 << AI2CR_TERM_APPROACH, learner_slot=1), 0)
        lane = h.get(0)
        self.assertEqual(self.ref.lib.ai2g_fixture_player_position(C.byref(lane.game), 0, 4.25, 0.0), 0)
        self.assertEqual(self.ref.lib.ai2g_fixture_player_position(C.byref(lane.game), 1, 4.75, 0.0), 0)
        self.assertEqual(h.put(0, lane), 0)
        self.assertEqual(h.step(), 0)
        lane = h.get(0)
        self.assertGreater(lane.last_reward.approach_delta, 0.0)
        self.assertEqual(lane.last_reward.learner_slot, 1)
        self.assertEqual(lane.last_reward.terms[0][AI2CR_TERM_APPROACH],
                         lane.last_reward.terms[1][AI2CR_TERM_APPROACH])
        self.assertEqual(h.rewards[0, 0], h.rewards[0, 1])

    def test_invalid_combat_reward_config_rejected_before_mutation(self):
        h = self.runtime
        h.reset()
        before = bytes(h.get(0))
        terms = np.zeros(AI2CR_TERMS, np.float64)
        terms[AI2CR_TERM_APPROACH] = float("nan")
        self.assertEqual(h.configure_reward(terms, 1 << AI2CR_TERM_APPROACH), 2)
        self.assertEqual(bytes(h.get(0)), before)
        terms[AI2CR_TERM_APPROACH] = 1.0
        self.assertEqual(h.configure_reward(terms, 1 << AI2CR_TERM_WIN, fade=float("nan")), 2)
        self.assertEqual(bytes(h.get(0)), before)

    def test_time_limit_is_truncated_not_terminated_and_full_platform_table_used(self):
        h = self.runtime
        h.reset()
        lane = h.get(0)
        scene = lane.game.scene
        scene.steps = 8999
        body = scene.world.bodies[scene.platform_body]
        body.p.x, body.p.y = self.ref.table.px[8999], self.ref.table.py[8999]
        self.assertEqual(h.put(0, lane), 0)
        self.assertEqual(h.step(), 0)
        self.assertEqual(h.truncated.tolist(), [1, 0])
        self.assertEqual(h.terminated.tolist(), [0, 0])
        row = h.drain()[1][0]
        self.assertEqual((row.terminated, row.truncated), (0, 1))
        self.assertEqual(h.get(0).game.scene.steps, 0)

    def test_nonfinite_action_freezes_only_error_lane_without_fake_frames(self):
        h = self.runtime
        h.reset()
        prior = h.obs.copy()
        action = np.zeros((2, 2, 10), np.float32)
        action[0, 1, 8] = np.nan
        rc = h.step(action)
        self.assertIn(rc, (0, 202) if self.cuda else (202,))
        np.testing.assert_array_equal(h.obs[0], prior[0])
        self.assertEqual(h.get(0).game.scene.steps, 0)
        h.step()
        self.assertEqual(h.drain()[3:], ([202, 0], [0, 2]))
        self.assertEqual(h.terminated[0], 0)
        self.assertEqual(h.rewards[0].tolist(), [0, 0])
        self.assertEqual(h.reset((1, 2), (1, 0)), 0)
        self.assertEqual(h.drain()[3], [0, 0])

    def test_capacity_error_is_explicit_and_sticky(self):
        h = Runtime(self.lib, self.ref, capacity=1, cuda=self.cuda)
        self.addCleanup(h.close)
        h.reset()
        for index in range(2):
            lane = h.get(index)
            lane.game.scene.players[0].ms.stocks = 0
            self.assertEqual(h.put(index, lane), 0)
        h.step()
        rc, rows, overflow, errors, steps = h.drain()
        self.assertEqual((rc, len(rows), overflow, steps), (0, 1, 1, [1, 1]))
        self.assertEqual(sorted(errors), [0, 301])
        failed = errors.index(301)
        h.step()
        self.assertEqual(h.drain()[4][failed], 1)

    def test_drain_too_small_does_not_discard_records(self):
        h = self.runtime
        h.reset()
        lane = h.get(0)
        lane.game.scene.players[0].ms.stocks = 0
        h.put(0, lane)
        h.step()
        self.assertEqual(h.drain(0)[0], 6)
        self.assertEqual(h.reset((30, 31), (1, 0)), 0)
        self.assertEqual(len(h.drain()[1]), 1)

    def test_two_resident_replay_tapes_do_not_alias_and_fail_on_exhaustion(self):
        h = self.runtime
        self.assertEqual(h.replay(0, reset_tape(0, 1)), 0)
        self.assertEqual(h.replay(1, reset_tape(1, 350)), 0)
        self.assertEqual(h.obs[:, 0, 0].tolist(), [-5, 5])
        h.step()  # lane0 needs a weapon draw; lane1 still has no RNG request.
        self.assertEqual(h.drain()[3:], ([105, 0], [0, 1]))
        for _ in range(10):
            h.step()
        self.assertEqual(h.get(1).rng.cursor, 7)
        self.assertEqual(h.get(0).rng.cursor, 7)
        self.assertEqual(h.drain()[4], [0, 11])

    def test_invalid_replay_value_rejected_before_reset(self):
        h = self.runtime
        h.reset()
        before = bytes(h.get(0))
        draws = reset_tape(0, 350)
        draws[0] = (1, 1, 0, float("nan"))
        self.assertEqual(h.replay(0, draws), 3)
        self.assertEqual(bytes(h.get(0)), before)

    def test_complete_heterogeneous_replay_tapes_stay_resident_and_match_reference(self):
        h = self.runtime
        refs = [game.GameReference(self.library) for _ in range(2)]
        keep = []
        for index in range(2):
            draws = reset_tape(index, 1 + index * 30) + [
                (2, 2.6, 6.5, 3.0 + index), (3, 0, 1, index),
                (2, 2.6, 6.5, 5.0 - index), (3, 0, 1, 1 - index)]
            self.assertEqual(h.replay(index, draws), 0)
            rng, array = refs[index].nav.replay_rng([
                {"kind": k, "a": a, "b": b, "value": v} for k, a, b, v in draws])
            keep.append(array)
            refs[index].rng = rng
            out = game.Result()
            self.assertEqual(refs[index].lib.ai2g_reset_export(
                C.byref(refs[index].state), C.byref(refs[index].cfg), C.byref(rng), 48, C.byref(out)), 0)
        for _ in range(200):
            self.assertEqual(h.step(), 0)
            for index, ref in enumerate(refs):
                out = ref.step_generated(np.zeros((2, 10)))
                np.testing.assert_array_equal(h.obs[index], np.asarray(out.obs, np.float32))
                self.assertEqual(bytes(h.get(index).game), bytes(ref.state))
        self.assertEqual([h.get(index).rng.cursor for index in range(2)], [11, 11])
        self.assertEqual(h.drain()[3:], ([0, 0], [200, 200]))

    def test_replay_step_consumes_exact_tape_and_restores_generated_mode(self):
        reference = Runtime(self.lib, self.ref, cuda=self.cuda)
        replay = Runtime(self.lib, self.ref, cuda=self.cuda)
        self.addCleanup(reference.close)
        self.addCleanup(replay.close)
        seeds = (77, 79)
        self.assertEqual(reference.reset(seeds), 0)
        self.assertEqual(replay.reset(seeds), 0)
        action = np.zeros((2, 10), np.float32)
        action[0, 7] = 1
        draw = [(3, 1, 12, 7)]
        self.assertEqual(reference.replay_step(0, action, draw), 0)
        self.assertEqual(reference.rng_status(0), (0, (0, 1, 1, 0)))
        self.assertEqual(replay.replay_step(0, action, draw), 0)
        self.assertEqual(replay.rng_status(0), (0, (0, 1, 1, 0)))
        np.testing.assert_array_equal(replay.obs[0], reference.obs[0])
        np.testing.assert_array_equal(replay.terminal[0], reference.terminal[0])
        np.testing.assert_array_equal(replay.rewards[0], reference.rewards[0])
        self.assertEqual((replay.terminated[0], replay.truncated[0]), (reference.terminated[0], reference.truncated[0]))
        self.assertEqual(replay.get(0).rng.mode, 1)
        self.assertEqual(replay.step(), 0)
        self.assertEqual(replay.get(0).rng.mode, 1)

    def test_replay_step_zero_draw_status_is_explicit_and_generated_rng_does_not_leak(self):
        h = self.runtime
        self.assertEqual(h.reset(), 0)
        before = h.get(0)
        action = np.zeros((2, 10), np.float32)
        self.assertEqual(h.replay_step(0, action, []), 0)
        after = h.get(0)
        self.assertEqual(h.rng_status(0), (0, (0, 0, 0, 0)))
        self.assertEqual(after.rng.mode, 1)
        self.assertEqual(list(after.rng.s), list(before.rng.s))
        self.assertEqual(after.rng.cursor, before.rng.cursor)
        self.assertEqual(after.rng.count, before.rng.count)

    def test_autoreset_rng_failure_keeps_valid_terminal_record_and_no_fake_reset(self):
        h = self.runtime
        self.assertEqual(h.replay(0, reset_tape(0, 350)), 0)
        self.assertEqual(h.replay(1, reset_tape(1, 350)), 0)
        lane = h.get(0)
        lane.game.scene.players[0].ms.stocks = 0
        h.put(0, lane)
        rc = h.step()
        self.assertIn(rc, (0, 3) if self.cuda else (3,))
        self.assertEqual(h.terminated.tolist(), [1, 0])
        np.testing.assert_array_equal(h.obs[0], h.terminal[0])
        rc, rows, overflow, errors, steps = h.drain()
        self.assertEqual((rc, len(rows), overflow, errors, steps), (0, 1, 0, [3, 0], [1, 1]))
        h.step()
        self.assertEqual(h.drain()[3:], ([3, 0], [1, 2]))

    def test_counter_exhaustion_does_not_advance_a_game(self):
        h = self.runtime
        h.reset()
        lane = h.get(0)
        lane.valid_steps = 2**64 - 1
        h.put(0, lane)
        h.step()
        self.assertEqual(h.get(0).game.scene.steps, 0)
        self.assertEqual(h.drain()[3:], ([302, 0], [2**64 - 1, 1]))

    def test_diagnostic_import_never_uses_foreign_replay_pointer(self):
        h = self.runtime
        h.replay(0, reset_tape(0, 350))
        h.replay(1, reset_tape(1, 350))
        lane = h.get(0)
        owned = C.cast(lane.rng.draws, V).value
        lane.rng.draws = C.cast(V(1), C.POINTER(nav.Draw))
        self.assertEqual(h.put(0, lane), 0)
        self.assertEqual(C.cast(h.get(0).rng.draws, V).value, owned)
        lane.rng.count = 1000
        self.assertEqual(h.put(0, lane), 3)
        self.assertEqual(h.step(), 0)

    def test_create_rejects_short_platform_table_and_invalid_configuration(self):
        rc = INT32()
        table = game.PlatformTable.from_buffer_copy(self.ref.table)
        table.count = 1024
        result = self.runtime.f['create'](C.byref(self.ref.cfg), C.byref(table), 2, 8, 48, C.byref(rc))
        self.assertFalse(result)
        self.assertEqual(rc.value, 2)
        for games, capacity, scale in ((0, 8, 48), (2, 0, 48), (2, 8, float('nan'))):
            result = self.runtime.f['create'](C.byref(self.ref.cfg), C.byref(self.ref.table), games, capacity, scale, C.byref(rc))
            self.assertFalse(result)
            self.assertEqual(rc.value, 2)

    def test_independent_context_lifetimes(self):
        h = self.runtime
        other = Runtime(self.lib, self.ref, cuda=self.cuda)
        self.addCleanup(other.close)
        h.reset()
        other.reset((987, 986))
        before = bytes(h.get(0))
        other.step()
        other.close()
        self.assertEqual(bytes(h.get(0)), before)
        self.assertEqual(h.step(), 0)


@unittest.skipUnless(shutil.which("nvcc"), "CUDA lane execution requires nvcc and an NVIDIA GPU; not qualified locally")
class CudaLaneTests(HostLaneTests):
    cuda = True

    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("NVIDIA CUDA device unavailable")
        super().setUpClass()
        output = Path(cls.build.name) / "cuda-lanes.so"
        subprocess.run(["nvcc", "-std=c++17", "-O2", "--fmad=false", "-shared", "-Xcompiler=-fPIC",
                        str(ROOT / "native/game_lanes.cu"), "-o", str(output)], check=True)
        cls.lib = C.CDLL(str(output))

    def test_context_orders_steps_across_nondefault_streams(self):
        h = self.runtime
        self.assertEqual(h.reset(), 0)
        streams = [h.torch.cuda.Stream(), h.torch.cuda.Stream()]
        actions = np.zeros((20, 2, 2, 10), np.float32)
        actions[::2, 0, 0, 3] = 1
        actions[1::2, 0, 0, 1] = 1
        actions[::3, 1, 1, 4] = 1
        device_actions = h.torch.as_tensor(actions, device='cuda')
        h.torch.cuda.synchronize()  # Setup only; no synchronization in loop.
        refs = [game.GameReference(self.library) for _ in range(2)]
        for index, seed in enumerate((7, 19)):
            refs[index].reset_generated(seed, index)
        for tick in range(len(actions)):
            h.stream = streams[tick % 2]
            h.extra = [V(h.stream.cuda_stream)]
            action_ptr = C.cast(device_actions[tick].data_ptr(), C.POINTER(F))
            self.assertEqual(h.f['step'](h.h, action_ptr, *h.pointers(), *h.extra), 0)
            for index, ref in enumerate(refs):
                ref.step_generated(actions[tick, index])
        self.assertEqual(h.drain()[4], [20, 20])  # First post-launch host sync.
        h.copy_outputs()
        for index, ref in enumerate(refs):
            np.testing.assert_array_equal(h.obs[index], ref.observe().astype(np.float32))
            self.assertEqual(bytes(h.get(index).game), bytes(ref.state))

    def test_wrong_device_rejected_without_freeing_context(self):
        h = self.runtime
        if h.torch.cuda.device_count() < 2:
            self.skipTest('wrong-device isolation requires two CUDA devices')
        with h.torch.cuda.device(1):
            self.assertEqual(h.f['destroy'](h.h), 304)
        self.assertEqual(h.reset(), 0)

    def test_cuda_reward_enabled_trace_matches_cpu_records_and_last_reward(self):
        cpu = Runtime(C.CDLL(str(self.library)), self.ref, games=3, capacity=12)
        cuda = Runtime(self.lib, self.ref, games=3, capacity=12, cuda=True)
        self.addCleanup(cpu.close)
        self.addCleanup(cuda.close)
        seeds = np.array([101, 202, 303], np.uint64)
        self.assertEqual(cpu.reset(seeds), 0)
        self.assertEqual(cuda.reset(seeds), 0)
        for index in range(3):
            lane = cpu.get(index)
            self.assertEqual(self.ref.lib.ai2g_fixture_player_position(C.byref(lane.game), 0, 4.0 + index, 0.0), 0)
            self.assertEqual(self.ref.lib.ai2g_fixture_player_position(C.byref(lane.game), 1, 5.0 + index, 0.0), 0)
            lane.game.max_timesteps = 4 if index == 0 else 12
            self.assertEqual(cpu.put(index, lane), 0)
            self.assertEqual(cuda.put(index, lane), 0)
        terms = np.zeros(AI2CR_TERMS, np.float64)
        terms[AI2CR_TERM_APPROACH] = 1.0
        terms[AI2CR_TERM_ATTACK_START] = 0.5
        terms[AI2CR_TERM_DAMAGE_DEALT] = 1.0
        terms[AI2CR_TERM_DAMAGE_TAKEN] = -1.0
        terms[AI2CR_TERM_NET_DAMAGE] = 2.0
        terms[AI2CR_TERM_TIMEOUT] = -0.25
        terms[AI2CR_TERM_OFFSTAGE] = -0.1
        enabled = sum(1 << term for term in (
            AI2CR_TERM_APPROACH, AI2CR_TERM_ATTACK_START, AI2CR_TERM_DAMAGE_DEALT,
            AI2CR_TERM_DAMAGE_TAKEN, AI2CR_TERM_NET_DAMAGE, AI2CR_TERM_TIMEOUT,
            AI2CR_TERM_OFFSTAGE,
        ))
        self.assertEqual(cpu.configure_reward(terms, enabled, learner_slot=1), 0)
        self.assertEqual(cuda.configure_reward(terms, enabled, learner_slot=1), 0)
        actions = np.zeros((6, 3, 2, 10), np.float32)
        actions[:, :, 0, 3] = 1.0
        actions[:, :, 1, 1] = 1.0
        actions[0::2, :, 0, 7] = 1.0
        actions[1::3, :, 1, 7] = 1.0
        for tick, batch in enumerate(actions):
            self.assertEqual(cpu.step(batch), 0, f"cpu tick {tick}")
            self.assertEqual(cuda.step(batch), 0, f"cuda tick {tick}")
            for label, expected, actual in zip(("obs", "terminal", "rewards", "terminated", "truncated"),
                                               cpu.outputs, cuda.outputs):
                self.assertEqual(actual.tobytes(), expected.tobytes(), f"{label} tick {tick}")
            for index in range(3):
                cpu_lane = cpu.get(index)
                cuda_lane = cuda.get(index)
                self.assertEqual(bytes(cuda_lane.last_reward), bytes(cpu_lane.last_reward), f"last_reward lane {index} tick {tick}")
                self.assertEqual(bytes(cuda_lane.reward_state), bytes(cpu_lane.reward_state), f"reward_state lane {index} tick {tick}")
        cpu_rows = sorted(cpu.drain()[1], key=lambda row: (row.lane, row.episode_index))
        cuda_rows = sorted(cuda.drain()[1], key=lambda row: (row.lane, row.episode_index))
        self.assertEqual([bytes(row) for row in cuda_rows], [bytes(row) for row in cpu_rows])
        self.assertTrue(cpu_rows, "trace must produce reward-enabled episode records")
        self.assertTrue(any(row.reward.version == 1 and row.reward.learner_slot == 1 for row in cpu_rows))

    def test_cuda_rejected_reward_config_does_not_mutate_lane_bytes(self):
        h = self.runtime
        self.assertEqual(h.reset(), 0)
        before = bytes(h.get(0))
        terms = np.zeros(AI2CR_TERMS, np.float64)
        terms[AI2CR_TERM_APPROACH] = float("nan")
        self.assertEqual(h.configure_reward(terms, 1 << AI2CR_TERM_APPROACH), 2)
        self.assertEqual(bytes(h.get(0)), before)


if __name__ == "__main__":
    unittest.main()
