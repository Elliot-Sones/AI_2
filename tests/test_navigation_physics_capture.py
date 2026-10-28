"""Physics capture records are observations, never restorable solver snapshots."""
import importlib
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


def body(*, angle=0.0):
    return NS(position=(1.0, 2.0), velocity=(3.0, 4.0), angle=angle,
              angular_velocity=2e-8, force=(0.0, 0.0), torque=0.0,
              mass=1.0, moment=1e9, body_type=0,
              center_of_gravity=(0.0, 0.0))


class _FakeFilter:
    group = 0
    categories = 0xFFFF_FFFF
    mask = 0xFFFF_FFFF


class _FakeBB:
    left = -0.5
    right = 0.5
    bottom = -0.5
    top = 0.5


class _FakeShape:
    collision_type = 1
    sensor = False
    friction = 0.7
    elasticity = 0.1
    filter = _FakeFilter()
    bb = _FakeBB()

    def __init__(self, owner_body):
        self.body = owner_body

    def get_vertices(self):
        return [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]


class _FakeSpace:
    iterations = 10
    gravity = (0.0, 17.808)
    damping = 1.0
    idle_speed_threshold = 0.0
    sleep_time_threshold = math.inf
    collision_slop = 0.1
    collision_bias = 0.5
    collision_persistence = 3
    threads = 1

    def __init__(self):
        self.dynamic = body()
        self.shape = _FakeShape(self.dynamic)
        self.bodies = [self.dynamic]
        self.shapes = [self.shape]
        self.calls = []

    def step(self, dt):
        self.calls.append(dt)
        self.dynamic.position = (self.dynamic.position[0] + 1.0, self.dynamic.position[1])


class _FakeGame:
    action_space = object()

    def __init__(self):
        self.space = _FakeSpace()
        self.objects = {"player": NS(body=self.space.dynamic, shape=self.space.shape)}
        self.closed = False

    def reset(self, *, seed):
        self.seed = seed

    def step(self, _action):
        self.space.step(1 / 30)

    def close(self):
        self.closed = True


class PhysicsCaptureTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('ai2.evaluation.navigation_physics_capture')

    def test_records_rotation_and_finite_inertia_without_rounding(self):
        row = self.module.capture_body(body(angle=1e-12), "player")
        self.assertEqual(row["angle"], 1e-12)
        self.assertEqual(row["angular_velocity"], 2e-8)
        self.assertEqual(row["inverse_moment"], 1e-9)
        self.assertEqual(row["position"], [1.0, 2.0])

    def test_static_infinite_mass_is_encoded_without_non_json_infinity(self):
        value = body()
        value.mass = value.moment = math.inf
        value.body_type = 2
        row = self.module.capture_body(value, "ground")
        self.assertEqual(row["inverse_mass"], 0.0)
        self.assertEqual(row["inverse_moment"], 0.0)

    def test_nonfinite_motion_fails_with_field_name(self):
        value = body(angle=math.nan)
        with self.assertRaisesRegex(ValueError, "player.angle"):
            self.module.capture_body(value, "player")

    def test_summary_reports_real_rotation_and_rejects_empty(self):
        rows = [{"tick": 0, "before": {"bodies": [self.module.capture_body(body(), "player")]},
                 "after": {"bodies": [self.module.capture_body(body(angle=1e-10), "player")]}}]
        result = self.module.summarize_physics_records(rows)
        self.assertEqual(result["physics_steps"], 1)
        self.assertEqual(result["max_abs_angle"], 1e-10)
        self.assertEqual(result["max_abs_angular_velocity"], 2e-8)
        self.assertEqual(result["rotating_body_ids"], ["player"])
        self.assertFalse(result["solver_snapshot_restore_supported"])
        with self.assertRaisesRegex(ValueError, "empty"):
            self.module.summarize_physics_records([])

    def test_recorder_restores_step_hook_on_error(self):
        calls = []
        def original(dt):
            calls.append(dt)
            raise RuntimeError("solver failed")
        raw = NS(space=NS(step=original))
        with self.module.PhysicsStepRecorder(raw, capture=lambda env: {"bodies": []}) as recorder:
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                raw.space.step(1 / 30)
        self.assertIs(raw.space.step, original)
        self.assertEqual(calls, [1 / 30])
        self.assertEqual(recorder.records, [])

    def test_recorder_captures_pre_post_exactly_once_and_handles_new_space(self):
        counter = [0]
        def original(dt):
            counter[0] += 1
            return "done"
        space = NS(step=original)
        raw = NS(space=space)
        with self.module.PhysicsStepRecorder(raw, capture=lambda env: {"count": counter[0]}) as recorder:
            self.assertEqual(raw.space.step(1 / 30), "done")
            self.assertEqual(recorder.records[0]["before"], {"count": 0})
            self.assertEqual(recorder.records[0]["after"], {"count": 1})
            new_space = NS(step=original)
            raw.space = new_space
            recorder.attach()
            raw.space.step(1 / 30)
            self.assertIs(space.step, original)
        self.assertIs(new_space.step, original)
        self.assertEqual([r["tick"] for r in recorder.records], [0, 1])

    def test_invalid_dt_is_rejected_before_advancing_solver(self):
        calls = []
        raw = NS(space=NS(step=lambda dt: calls.append(dt)))
        with self.module.PhysicsStepRecorder(raw, capture=lambda env: {}) as recorder:
            with self.assertRaisesRegex(ValueError, "dt"):
                raw.space.step(math.nan)
        self.assertEqual(calls, [])
        self.assertEqual(recorder.records, [])

    def test_game_capture_cli_is_bounded_writes_output_and_restores_cwd(self):
        def build(seed):
            self.assertEqual(seed, 12)
            return _FakeGame()

        def actions(space, steps, seed, n_agents):
            self.assertIsNotNone(space)
            self.assertEqual((steps, seed, n_agents), (2, 34, 2))
            return ["a", "b"]

        with tempfile.TemporaryDirectory(prefix="ai2-physics-capture-") as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            out = Path(tmp) / "out" / "capture.json"
            before_cwd = Path.cwd()
            payload = self.module.capture_game_physics(
                root,
                out,
                seed=12,
                action_seed=34,
                steps=2,
                prepare_source=lambda path: os.chdir(path),
                build_env=build,
                make_actions=actions,
            )
            self.assertEqual(Path.cwd(), before_cwd)
            written = json.loads(out.read_text())

        self.assertEqual(payload["physics_steps_recorded"], 2)
        self.assertEqual(written["physics_steps_recorded"], 2)
        self.assertEqual(written["records"][0]["before"]["bodies"][0]["position"], [1.0, 2.0])
        self.assertEqual(written["records"][0]["after"]["bodies"][0]["position"], [2.0, 2.0])
        self.assertFalse(written["solver_snapshot_restore_supported"])

    def test_game_capture_rejects_more_than_one_small_local_diagnostic_window(self):
        with tempfile.TemporaryDirectory(prefix="ai2-physics-capture-") as tmp:
            with self.assertRaisesRegex(ValueError, "between 0 and 500"):
                self.module.capture_game_physics(
                    Path(tmp),
                    Path(tmp) / "out.json",
                    steps=501,
                    prepare_source=lambda _path: None,
                    build_env=lambda _seed: _FakeGame(),
                    make_actions=lambda *_args: [],
                )


class RealPymunkCaptureTests(unittest.TestCase):
    def setUp(self):
        try:
            import pymunk
        except ImportError:
            self.skipTest("Pymunk runtime unavailable")
        self.pm = pymunk
        self.module = importlib.import_module('ai2.evaluation.navigation_physics_capture')

    def world(self):
        pm = self.pm
        space = pm.Space()
        space.gravity = (0, 17.808)
        dynamic = pm.Body(1, 1e9)
        dynamic.position = (0.2, -0.9)
        shape = pm.Poly.create_box(dynamic, (0.87, 1))
        shape.friction = 0.7
        ground = pm.Body(body_type=pm.Body.STATIC)
        floor = pm.Poly.create_box(ground, (10, 1))
        floor.friction = 0.7
        space.add(shape, dynamic, floor, ground)
        return NS(space=space, objects={"player": NS(body=dynamic, shape=shape),
                                      "ground": NS(body=ground, shape=floor)})

    def test_space_capture_has_shapes_settings_and_oriented_real_contacts(self):
        raw = self.world()
        for _ in range(3):
            raw.space.step(1 / 30)
        snapshot = self.module.capture_space(raw)
        self.assertEqual(snapshot["settings"]["iterations"], 10)
        self.assertEqual(len(snapshot["bodies"]), 2)
        self.assertEqual(len(snapshot["shapes"]), 2)
        self.assertTrue(snapshot["contacts"])
        self.assertEqual(set(snapshot["contacts"][0]["shape_ids"]), {"objects/player", "objects/ground"})
        self.assertEqual(snapshot["contacts"][0]["normal_direction"], "shape_a_to_shape_b")
        self.assertFalse(snapshot["solver_snapshot_restore_supported"])

    def test_unlabelled_shape_fails_instead_of_dropping_physics_body(self):
        raw = self.world()
        extra = self.pm.Body(1, 1e9)
        extra_shape = self.pm.Poly.create_box(extra, (1, 1))
        raw.space.add(extra, extra_shape)
        with self.assertRaisesRegex(ValueError, "unlabelled"):
            self.module.capture_space(raw)


if __name__ == "__main__":
    unittest.main()
