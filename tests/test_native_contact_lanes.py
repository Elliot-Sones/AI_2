"""Policy-free contact transport evidence: CPU oracle and optional real CUDA.

No CPU policy training occurs here. CPU and CUDA run the same explicit,
finite scenario schedules; fixture state edits only test failure handling.
"""

import ctypes as C
import importlib
from pathlib import Path
import shutil
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class ContactTransportPresenceTests(unittest.TestCase):
    def test_contact_lane_public_module_exists(self):
        self.assertTrue(
            (ROOT / "native_contact_lane_vec_env.py").is_file(),
            "contact lane transport is not implemented",
        )


class ContactLaneTests(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        path = ROOT / "native_contact_lane_vec_env.py"
        assert path.is_file(), "contact lane transport is not implemented"
        cls.transport = importlib.import_module("native_contact_lane_vec_env")
        cls.core = importlib.import_module("native_contact_episode")

    def make_env(self, lanes=2, **kwargs):
        kwargs.setdefault("episodes_per_lane", 4)
        env = self.transport.build_contact_env(
            lanes, device=self.device, seed=7, **kwargs
        )
        self.addCleanup(env.close)
        return env

    def sync(self):
        if self.device == "cuda":
            torch.cuda.current_stream().synchronize()

    def test_schedule_is_reproducible_explicit_and_lane_major(self):
        schedule = self.transport.make_scenario_schedule(3, 4, seed=17)
        other = self.transport.make_scenario_schedule(3, 4, seed=17)
        self.assertEqual(schedule.shape, (3, 4))
        self.assertEqual(schedule.tobytes(), other.tobytes())
        self.assertEqual(
            schedule["seed"].tolist(),
            [
                [17, 3182414963, 3629568597, 461025355],
                [18, 3837977555, 1715005219, 915759487],
                [19, 2529908022, 1805513179, 1531970275],
            ],
        )
        for lane in range(3):
            for episode in range(4):
                scenario = self.core.scenario_from_seed(
                    int(schedule["seed"][lane, episode]), player_slot=lane % 2
                )
                self.assertEqual(schedule[lane, episode].tobytes(), bytes(scenario))

    def test_one_learner_per_game_and_zero_padded_four_frame_stack(self):
        env = self.make_env()
        initial = env.reset_tensor().clone()
        self.assertEqual(tuple(initial.shape), (2, 256))
        self.assertEqual(torch.count_nonzero(initial[:, :192]).item(), 0)
        action = torch.zeros((2, 10), device=self.device)
        result = env.step_tensor(action)
        torch.testing.assert_close(
            result.observations[:, 128:192], initial[:, 192:], rtol=0, atol=0
        )
        self.assertEqual(tuple(result.rewards.shape), (2,))
        self.assertEqual(result.terminated.tolist(), [False, False])
        self.assertEqual(result.truncated.tolist(), [False, False])
        self.assertEqual(env.drain_episode_infos(), [])
        self.assertEqual(env.backend.valid_steps.tolist(), [1, 1])

    def test_task_metadata_is_immutable_and_present_on_episode_records(self):
        env = self.make_env(lanes=1)
        self.assertEqual(env.curriculum_task, "combat")
        self.assertEqual(env.curriculum_version, "bridge_b1_contact-v1")
        self.assertEqual(
            env.allowed_control_mask,
            (True, True, True, True, True, False, True, True, False, False),
        )
        self.assertFalse(env.host_has_timeouts())
        with self.assertRaises(AttributeError):
            env.curriculum_task = "navigation"
        env.reset_tensor()
        for _ in range(300):
            env.step_tensor(torch.zeros((1, 10), device=self.device))
        (info,) = env.drain_episode_infos()
        for name in ("curriculum_task", "curriculum_version", "allowed_control_mask"):
            self.assertEqual(info[name], getattr(env, name))
        self.assertEqual(
            env.backend.scenario_metadata["generator"], "b1-vector-env-gym-seeds-v1"
        )

    def test_scripted_contacts_match_host_through_autoresets_for_both_slots_and_directions(
        self,
    ):
        schedule = self.transport.make_scenario_schedule(4, 8, seed=17)
        for lane in range(4):
            schedule["player_slot"][lane] = lane // 2
            schedule["direction"][lane] = 1 if lane % 2 else -1
        candidate = self.make_env(lanes=4, scenario_schedule=schedule)
        host = self.transport.build_contact_env(
            4, device="cpu", seed=7, scenario_schedule=schedule
        )
        self.addCleanup(host.close)
        candidate.reset_tensor()
        host.reset_tensor()
        actual_rows, expected_rows = [], []
        names = (
            "observations",
            "rewards",
            "terminated",
            "truncated",
            "terminal_observations",
        )
        for _ in range(180):
            actions = torch.zeros((4, 10))
            for lane in range(4):
                diagnostic = host.backend.lane_state(lane).episode.diagnostics
                learner = diagnostic.player_slot
                player, target = (
                    diagnostic.players[learner],
                    diagnostic.players[1 - learner],
                )
                if diagnostic.hit_attack_id or player.state_id == 8:
                    continue
                gap = target.x - player.x
                direction = 1 if gap >= 0 else -1
                if abs(gap) > 2.1 or player.facing != direction:
                    actions[lane, 3 if direction == 1 else 1] = 1
                elif player.state_id == 1 and abs(player.vx) < 0.01:
                    actions[lane, 2] = actions[lane, 7] = 1
            expected = host.step_tensor(actions)
            actual = candidate.step_tensor(actions.to(self.device))
            expected_rows.append([getattr(expected, name).clone() for name in names])
            actual_rows.append([getattr(actual, name).clone() for name in names])
        self.sync()
        for actual, expected in zip(actual_rows, expected_rows):
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left.cpu(), right, rtol=0, atol=0)
        records, _ = candidate.backend.drain()
        reference_records, _ = host.backend.drain()

        def key(row):
            return row.lane, row.episode_index

        self.assertEqual(
            [bytes(row) for row in sorted(records, key=key)],
            [bytes(row) for row in sorted(reference_records, key=key)],
        )
        self.assertGreaterEqual(len(records), 4)
        for lane in range(4):
            row = next(row for row in records if row.lane == lane)
            self.assertEqual(row.diagnostics.outcome, 1)
            self.assertGreater(row.diagnostics.first_hit_tick, 0)
            self.assertGreater(row.diagnostics.attack_starts, 0)
            self.assertGreater(row.diagnostics.damage_events, 0)
            self.assertGreaterEqual(
                row.diagnostics.approach_progress_at_first_hit, 0.25
            )
            self.assertEqual(list(row.diagnostics.reward_totals)[:2], [1, 2])
            self.assertTrue(
                all(
                    p.stocks == 3 and p.fully_supported for p in row.diagnostics.players
                )
            )

    def test_setup_schedule_is_owned_and_drain_records_are_stable_snapshots(self):
        schedule = self.transport.make_scenario_schedule(1, 4, seed=7)
        env = self.make_env(lanes=1, scenario_schedule=schedule)
        schedule["seed"][:] = 999
        env.reset_tensor()
        self.assertEqual(env.backend.lane_state(0).episode.scenario.seed, 7)
        for _ in range(300):
            env.step_tensor(torch.zeros((1, 10), device=self.device))
        first, _ = env.backend.drain()
        snapshot = bytes(first[0])
        for _ in range(300):
            env.step_tensor(torch.zeros((1, 10), device=self.device))
        second, _ = env.backend.drain()
        self.assertEqual(bytes(first[0]), snapshot)
        self.assertEqual(second[0].episode_index, 1)
        self.assertNotEqual(bytes(first[0]), bytes(second[0]))

    def test_small_drain_buffer_does_not_discard_pending_records(self):
        env = self.make_env(lanes=1)
        env.reset_tensor()
        for _ in range(300):
            env.step_tensor(torch.zeros((1, 10), device=self.device))
        backend = env.backend
        count, overflow = C.c_int32(), C.c_int32()
        with backend._guard():
            rc = backend.drain_fn(
                *backend._args(
                    None,
                    0,
                    C.byref(count),
                    C.byref(overflow),
                    C.c_void_p(backend.errors.ctypes.data),
                    C.c_void_p(backend.valid_steps.ctypes.data),
                )
            )
        self.assertEqual(rc, 6)
        records, _ = backend.drain()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].length, 300)

    def test_independent_lanes_match_standalone_contact_core(self):
        env = self.make_env()
        obs = env.reset_tensor()
        refs = [self.core.ContactReference() for _ in range(2)]
        for lane, ref in enumerate(refs):
            initial = ref.reset_generated(
                self.core.scenario_from_seed(7 + lane, player_slot=lane % 2),
                stream=lane,
            )
            np.testing.assert_array_equal(
                obs[lane, -64:].cpu().numpy(), np.asarray(initial.obs, np.float32)
            )
        actions = np.zeros((2, 10), np.float32)
        for tick in range(25):
            actions.fill(0)
            actions[0, 3] = 1
            actions[1, 1] = tick < 10
            actions[1, 7] = tick % 9 == 0
            result = env.step_tensor(torch.as_tensor(actions, device=self.device))
            for lane, ref in enumerate(refs):
                expected = ref.step_generated(actions[lane])
                self.assertFalse(expected.terminated or expected.truncated)
                np.testing.assert_array_equal(
                    result.observations[lane, -64:].cpu().numpy(),
                    np.asarray(expected.obs, np.float32),
                )
                self.assertEqual(
                    result.rewards[lane].item(), np.float32(expected.reward)
                )
                self.assertEqual(
                    bytes(env.backend.lane_state(lane).episode), bytes(ref.state)
                )
        env.drain_episode_infos()
        self.assertEqual(env.backend.valid_steps.tolist(), [25, 25])

    def test_timeout_record_preserves_terminal_frame_before_autoreset(self):
        env = self.make_env()
        previous = env.reset_tensor().clone()
        actions = torch.zeros((2, 10), device=self.device)
        for _ in range(300):
            result = env.step_tensor(actions)
            if bool(result.terminated.any()):
                break
            previous = result.observations.clone()
        self.assertEqual(result.terminated.tolist(), [True, True])
        self.assertEqual(result.truncated.tolist(), [False, False])
        self.assertEqual(torch.count_nonzero(result.observations[:, :192]).item(), 0)
        torch.testing.assert_close(
            result.terminal_observations[:, :192], previous[:, 64:], rtol=0, atol=0
        )
        self.assertFalse(
            torch.equal(
                result.observations[:, -64:], result.terminal_observations[:, -64:]
            )
        )
        infos = env.drain_episode_infos()
        self.assertEqual(len(infos), 2)
        for info in infos:
            self.assertEqual(info["episode"]["l"], 300)
            self.assertAlmostEqual(info["episode"]["r"], -1.3, places=6)
            contact = info["contact"]
            self.assertEqual(contact["episode_index"], 0)
            self.assertEqual(contact["scenario_index"], 0)
            self.assertTrue(contact["terminated"])
            self.assertFalse(contact["truncated"])
            self.assertEqual(contact["diagnostics"]["tick"], 300)
            self.assertAlmostEqual(
                sum(contact["diagnostics"]["reward_terms"].values()), -1.3, places=6
            )
            self.assertEqual(len(contact["diagnostics"]["players"]), 2)
        self.assertEqual(env.backend.valid_steps.tolist(), [300, 300])
        self.assertEqual(
            [env.backend.lane_state(i).scenario_index for i in range(2)], [1, 1]
        )

    def test_explicit_reset_consumes_schedule_and_preserves_undrained_records(self):
        env = self.make_env(lanes=1)
        env.reset_tensor()
        for _ in range(300):
            env.step_tensor(torch.zeros((1, 10), device=self.device))
        env.reset_tensor()
        self.assertEqual(env.backend.lane_state(0).scenario_index, 2)
        infos = env.drain_episode_infos()
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0]["contact"]["episode_index"], 0)
        self.assertEqual(env.backend.valid_steps.tolist(), [300])

    def test_schedule_exhaustion_latches_and_keeps_terminal_transition(self):
        env = self.make_env(lanes=1, episodes_per_lane=1)
        env.reset_tensor()
        for _ in range(299):
            env.step_tensor(torch.zeros((1, 10), device=self.device))
        if self.device == "cpu":
            with self.assertRaisesRegex(RuntimeError, "505"):
                env.step_tensor(torch.zeros((1, 10)))
        else:
            env.step_tensor(torch.zeros((1, 10), device=self.device))
        lane = env.backend.lane_state(0)
        self.assertEqual(lane.error, 505)
        self.assertEqual(lane.valid_steps, 300)
        self.assertEqual(lane.episodes, 1)
        self.assertEqual(lane.scenario_index, 0)
        self.assertTrue(env._terminated[0])
        self.assertAlmostEqual(env._rewards[0].item(), -1.001, places=6)
        with self.assertRaisesRegex(RuntimeError, "invalid rollout"):
            env.drain_episode_infos()
        with self.assertRaisesRegex(RuntimeError, "505"):
            env.reset_tensor()

    def test_capacity_overflow_fails_closed_and_counter_does_not_wrap(self):
        env = self.make_env(lanes=2, record_capacity=1, episodes_per_lane=8)
        env.reset_tensor()
        for _ in range(299):
            env.step_tensor(torch.zeros((2, 10), device=self.device))
        if self.device == "cpu":
            with self.assertRaisesRegex(RuntimeError, "501"):
                env.step_tensor(torch.zeros((2, 10)))
        else:
            env.step_tensor(torch.zeros((2, 10), device=self.device))
        with self.assertRaisesRegex(RuntimeError, "record_overflow=1"):
            env.drain_episode_infos()
        self.assertEqual(sum(env.backend.errors == 501), 1)
        for field, value in (
            ("valid_steps", (1 << 64) - 1),
            ("length", (1 << 31) - 1),
            ("episodes", (1 << 31) - 1),
        ):
            env.reset_tensor()
            lane = env.backend.lane_state(0)
            setattr(lane, field, value)
            if field != "valid_steps":
                # These counters cannot exceed the bounded current episode /
                # finite schedule. Reject inconsistent fixtures at import.
                original = bytes(env.backend.lane_state(0))
                with self.assertRaisesRegex(RuntimeError, "lane_set: error 2"):
                    env.backend.set_lane_state(0, lane)
                self.assertEqual(bytes(env.backend.lane_state(0)), original)
                continue
            env.backend.set_lane_state(0, lane)
            if self.device == "cpu":
                with self.assertRaisesRegex(RuntimeError, "502"):
                    env.step_tensor(torch.zeros((2, 10)))
            else:
                env.step_tensor(torch.zeros((2, 10), device=self.device))
            self.assertEqual(getattr(env.backend.lane_state(0), field), value)
            with self.assertRaisesRegex(RuntimeError, "invalid rollout"):
                env.drain_episode_infos()
            # Diagnostic fixture only: restore the invariant before next reset.
            lane = env.backend.lane_state(0)
            lane.valid_steps, lane.length, lane.episodes = 300, 0, 1
            env.backend.set_lane_state(0, lane)

    def test_nonfinite_action_freezes_lane_without_counting_invalid_tick(self):
        env = self.make_env(lanes=1)
        before = env.reset_tensor().clone()
        action = torch.full((1, 10), float("nan"), device=self.device)
        if self.device == "cpu":
            with self.assertRaises(RuntimeError):
                env.step_tensor(action)
        else:
            env.step_tensor(action)
        lane = env.backend.lane_state(0)
        self.assertNotEqual(lane.error, 0)
        self.assertEqual(lane.valid_steps, 0)
        torch.testing.assert_close(env._obs, before[:, -64:], rtol=0, atol=0)
        self.assertEqual(env._rewards.tolist(), [0.0])
        with self.assertRaisesRegex(RuntimeError, "invalid rollout"):
            env.drain_episode_infos()
        env.reset_tensor()
        env.step_tensor(torch.zeros((1, 10), device=self.device))
        env.drain_episode_infos()
        self.assertEqual(env.backend.valid_steps.tolist(), [1])

    def test_diagnostic_import_rejects_invalid_player_slot_without_mutating_lane(self):
        env = self.make_env(lanes=1)
        env.reset_tensor()
        original = env.backend.lane_state(0)
        malformed = self.transport.Lane.from_buffer_copy(original)
        malformed.episode.scenario.player_slot = 2
        with self.assertRaisesRegex(RuntimeError, "lane_set: error 2"):
            env.backend.set_lane_state(0, malformed)
        self.assertEqual(bytes(env.backend.lane_state(0)), bytes(original))
        env.step_tensor(torch.zeros((1, 10), device=self.device))
        self.assertEqual(env.backend.lane_state(0).valid_steps, 1)

    def test_diagnostic_import_rejects_foreign_and_rewound_valid_scenarios(self):
        env = self.make_env(lanes=1)
        env.reset_tensor()
        original = env.backend.lane_state(0)
        reference = self.core.ContactReference()
        result = reference.reset_generated(self.core.scenario_from_seed(999), stream=0)
        foreign = self.transport.Lane.from_buffer_copy(original)
        foreign.episode, foreign.rng = reference.state, reference.rng
        foreign.last_obs[:] = np.asarray(result.obs, np.float32)
        with self.assertRaisesRegex(RuntimeError, "lane_set: error 2"):
            env.backend.set_lane_state(0, foreign)
        self.assertEqual(bytes(env.backend.lane_state(0)), bytes(original))
        env.reset_tensor()
        current = bytes(env.backend.lane_state(0))
        with self.assertRaisesRegex(RuntimeError, "lane_set: error 2"):
            env.backend.set_lane_state(0, original)
        self.assertEqual(bytes(env.backend.lane_state(0)), current)

    def test_diagnostic_import_rejects_inconsistent_counters_rng_and_metadata(self):
        env = self.make_env(lanes=1)
        env.reset_tensor()
        env.step_tensor(torch.zeros((1, 10), device=self.device))
        original = env.backend.lane_state(0)
        mutations = [
            ("episode.scenario.player_slot", -1),
            ("episode.scenario.direction", 0),
            ("episode.scenario.gap", float("nan")),
            ("episode.scenario.midpoint", float("inf")),
            ("episode.diagnostics.player_slot", 2),
            ("episode.diagnostics.tick", 301),
            ("episode.initialized", 0),
            ("rng.mode", 0),
            ("rng.cursor", 1),
            ("rng.count", 1),
            ("rng.error", 1),
            ("rng.error_cursor", 1),
            ("rng.error_kind", 1),
            ("rng.error_a", 1.0),
            ("rng.error_b", float("nan")),
            ("length", 0),
            ("valid_steps", 0),
            ("episodes", 2),
            ("ret", float("inf")),
            ("active", 0),
        ]
        for path, value in mutations:
            with self.subTest(field=path):
                malformed = self.transport.Lane.from_buffer_copy(original)
                target = malformed
                components = path.split(".")
                for name in components[:-1]:
                    target = getattr(target, name)
                setattr(target, components[-1], value)
                with self.assertRaisesRegex(RuntimeError, "lane_set: error 2"):
                    env.backend.set_lane_state(0, malformed)
                self.assertEqual(bytes(env.backend.lane_state(0)), bytes(original))
        for field in ("rng.s", "last_obs"):
            with self.subTest(field=field):
                malformed = self.transport.Lane.from_buffer_copy(original)
                if field == "rng.s":
                    malformed.rng.s[:] = [0] * 4
                else:
                    malformed.last_obs[0] = float("nan")
                with self.assertRaisesRegex(RuntimeError, "lane_set: error 2"):
                    env.backend.set_lane_state(0, malformed)
                self.assertEqual(bytes(env.backend.lane_state(0)), bytes(original))

    def test_validation_and_closed_context(self):
        for count in (0, -1, True, 1.2):
            with self.assertRaises(ValueError):
                self.transport.build_contact_env(count, device=self.device)
        for count in (0, -1, True, 1.2):
            with self.assertRaises(ValueError):
                self.make_env(episodes_per_lane=count)
        env = self.make_env(lanes=1)
        env.reset_tensor()
        with self.assertRaises(ValueError):
            env.step_tensor(torch.zeros((2, 10), device=self.device))
        with self.assertRaisesRegex(ValueError, "fixed"):
            env.reset_tensor(seeds=[999])
        schedule = self.transport.make_scenario_schedule(1, 2, seed=7)
        schedule["gap"][0, 1] = float("nan")
        with self.assertRaises(ValueError):
            self.make_env(lanes=1, scenario_schedule=schedule)
        env.close()
        env.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            env.reset_tensor()


@unittest.skipUnless(
    torch.cuda.is_available() and shutil.which("nvcc"),
    "real CUDA device and nvcc required",
)
class ContactCudaTests(ContactLaneTests):
    device = "cuda"

    def test_nondefault_stream_cpu_parity_and_cross_stream_order(self):
        gpu = self.make_env()
        host = self.transport.build_contact_env(
            2, device="cpu", seed=7, episodes_per_lane=4
        )
        self.addCleanup(host.close)
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        with torch.cuda.stream(streams[0]):
            device_initial = gpu.reset_tensor().clone()
        streams[0].synchronize()
        torch.testing.assert_close(
            device_initial.cpu(), host.reset_tensor(), rtol=0, atol=0
        )
        queued = []
        for tick in range(305):
            actions = torch.zeros((2, 10))
            stream = streams[tick % 2]
            with torch.cuda.stream(stream):
                actual = gpu.step_tensor(actions.to("cuda"))
                snapshots = [
                    getattr(actual, name).clone()
                    for name in (
                        "observations",
                        "rewards",
                        "terminated",
                        "truncated",
                        "terminal_observations",
                    )
                ]
            expected = host.step_tensor(actions)
            expected_snapshots = [
                getattr(expected, name).clone()
                for name in (
                    "observations",
                    "rewards",
                    "terminated",
                    "truncated",
                    "terminal_observations",
                )
            ]
            queued.append((snapshots, expected_snapshots))
        for stream in streams:
            stream.synchronize()
        for snapshots, expected_snapshots in queued:
            for snapshot, expected in zip(snapshots, expected_snapshots):
                torch.testing.assert_close(snapshot.cpu(), expected, rtol=0, atol=0)
        records, overflow = gpu.backend.drain()
        host_records, host_overflow = host.backend.drain()
        self.assertEqual((overflow, host_overflow), (0, 0))

        def key(row):
            return row.lane, row.episode_index

        self.assertEqual(
            [bytes(row) for row in sorted(records, key=key)],
            [bytes(row) for row in sorted(host_records, key=key)],
        )
        for index in range(2):
            self.assertEqual(
                bytes(gpu.backend.lane_state(index)),
                bytes(host.backend.lane_state(index)),
            )

    @unittest.skipUnless(torch.cuda.device_count() > 1, "two CUDA devices required")
    def test_native_context_rejects_wrong_device(self):
        env = self.make_env(lanes=1)
        env.reset_tensor()
        backend = env.backend
        with torch.cuda.device(1):
            rc = backend.get_fn(
                backend.handle,
                0,
                C.byref(self.transport.Lane()),
                C.c_void_p(torch.cuda.current_stream().cuda_stream),
            )
        self.assertEqual(rc, 504)


if __name__ == "__main__":
    unittest.main()
