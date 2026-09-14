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

import native_game_episode as game
import native_navigation_episode as nav

ROOT = Path(__file__).resolve().parents[1]
INT32, U, B, F, D, V = C.c_int32, C.c_uint64, C.c_uint8, C.c_float, C.c_double, C.c_void_p


class Lane(C.Structure):
    _fields_ = [("game", game.State), ("rng", nav.Rng), ("returns", D * 2),
                ("last_obs", F * 128), ("valid_steps", U)] + [
                    (n, INT32) for n in ("length", "episodes", "error", "active")]


class Record(C.Structure):
    _fields_ = [(n, INT32) for n in ("lane", "length", "terminated", "truncated", "win_mask", "episode_index")] + [
        ("stocks", INT32 * 2), ("returns", D * 2), ("damage_done", D * 2), ("damage_taken", D * 2)]


def ptr(array, kind=F):
    return array.ctypes.data_as(C.POINTER(kind))


class Runtime:
    def __init__(self, lib, ref, games=2, capacity=8, cuda=False):
        self.lib, self.n, self.cuda = lib, games, cuda
        prefix = "ai2g_lanes_cuda_" if cuda else "ai2g_lanes_cpu_"
        self.f = {name: getattr(lib, prefix + name) for name in (
            "create", "destroy", "reset", "step", "drain", "lane_get", "lane_set", "replay_reset")}
        extra = [V] if cuda else []
        specs = {
            "create": [C.POINTER(nav.Config), C.POINTER(game.PlatformTable), INT32, INT32, D, C.POINTER(INT32)],
            "destroy": [V], "reset": [V, C.POINTER(U), C.POINTER(B), C.POINTER(F)] + extra,
            "step": [V] + [C.POINTER(F)] * 4 + [C.POINTER(B)] * 2 + extra,
            "drain": [V, C.POINTER(Record), INT32, C.POINTER(INT32), C.POINTER(INT32), C.POINTER(INT32), C.POINTER(U)] + extra,
            "lane_get": [V, INT32, C.POINTER(Lane)] + extra,
            "lane_set": [V, INT32, C.POINTER(Lane)] + extra,
            "replay_reset": [V, INT32, C.POINTER(nav.Draw), INT32, C.POINTER(F)] + extra,
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


if __name__ == "__main__":
    unittest.main()
