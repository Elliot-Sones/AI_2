"""Shared raw scene reset must not apply the navigation curriculum wrapper."""

import ctypes
import tempfile
import unittest
from pathlib import Path

import ai2.simulation.native_navigation_episode as nav


class RawSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="ai2-game-scene-")
        cls.ref = nav.EpisodeReference(
            nav.compile_cpu_reference(
                Path(cls.tmp.name), nav.contact.default_contraction()
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reset_consumes_only_raw_draws_and_leaves_both_players_at_spawn(self):
        reset = self.ref.lib.ai2n_reset_scene_export
        reset.argtypes = [
            ctypes.POINTER(nav.Episode),
            ctypes.POINTER(nav.Config),
            ctypes.POINTER(nav.Rng),
        ]
        reset.restype = ctypes.c_int
        draws = [{"kind": 1, "a": 1, "b": 0, "value": 1}]
        for x, cooldown, phase in ((3.0, 500, 10), (4.0, 600, 20)):
            draws.extend(
                [
                    {"kind": 2, "a": 2.6, "b": 6.5, "value": x},
                    {"kind": 3, "a": 500, "b": 700, "value": cooldown},
                    {"kind": 3, "a": 0, "b": 350, "value": phase},
                ]
            )
        rng, keepalive = self.ref.replay_rng(draws)
        cfg = self.ref.make_config(
            {
                "completion_mode": "sequence",
                "control_mode": "full",
                "max_targets": 3,
                "time_limit": 600,
                "settle_steps": 9,
                "post_goal_required_steps": 15,
                "settle_speed": 1.0,
                "fall_penalty": -3.0,
                "min_dist": 2.0,
                "max_dist": 6.0,
            }
        )
        state = nav.Episode()
        self.assertEqual(
            reset(ctypes.byref(state), ctypes.byref(cfg), ctypes.byref(rng)), 0
        )
        self.assertEqual(rng.cursor, 7)
        self.assertEqual(rng.error, 0)
        self.assertEqual(state.world.bodies[3].p.x, 5.0)
        self.assertEqual(state.world.bodies[4].p.x, -5.0)
        self.assertEqual(state.world.bodies[4].p.y, 0.0)
        self.assertEqual(state.wrap.target_valid, 0)
        self.assertEqual(state.steps, 0)
        self.assertEqual(state.spawners[0].last_spawn_frame, -490)
        self.assertEqual(state.spawners[1].last_spawn_frame, -580)
        self.assertEqual(len(keepalive), 7)


if __name__ == "__main__":
    unittest.main()
