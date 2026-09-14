"""Tests for the exact installed-Chipmunk ctypes bridge."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))


DT = 1.0 / 30.0


def require_pymunk():
    try:
        import pymunk
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("Pymunk oracle unavailable") from exc
    if getattr(pymunk, "version", None) != "6.2.1":
        raise unittest.SkipTest(f"Pymunk oracle must be 6.2.1, got {getattr(pymunk, 'version', None)!r}")
    return pymunk


def make_world(*, dynamic_position, dynamic_velocity, floor_type="static", floor_velocity=(0.0, 0.0), angle=0.0):
    pymunk = require_pymunk()
    space = pymunk.Space()
    space.gravity = (0.0, 17.808)
    space.damping = 0.97
    player = pymunk.Body(1.0, 1e9)
    player.position = dynamic_position
    player.velocity = dynamic_velocity
    player.angle = angle
    player.angular_velocity = 0.25 if angle else 0.0
    player_shape = pymunk.Poly.create_box(player, (0.87, 1.0))
    player_shape.friction = 0.7
    player_shape.elasticity = 0.0

    if floor_type == "kinematic":
        floor = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    else:
        floor = pymunk.Body(body_type=pymunk.Body.STATIC)
    floor.position = (0.0, 1.0)
    floor.velocity = floor_velocity
    floor_shape = pymunk.Poly.create_box(floor, (20.0, 0.1))
    floor_shape.friction = 0.9
    floor_shape.elasticity = 0.0

    space.add(player, player_shape, floor, floor_shape)
    return space, player, floor, player_shape, floor_shape


def clone_world(spec):
    return make_world(**spec)


def assert_body_close(testcase, actual, expected, *, tolerance=1e-12):
    testcase.assertLessEqual(abs(actual["position"][0] - expected.position.x), tolerance)
    testcase.assertLessEqual(abs(actual["position"][1] - expected.position.y), tolerance)
    testcase.assertLessEqual(abs(actual["velocity"][0] - expected.velocity.x), tolerance)
    testcase.assertLessEqual(abs(actual["velocity"][1] - expected.velocity.y), tolerance)
    testcase.assertLessEqual(abs(actual["angle"] - expected.angle), tolerance)
    testcase.assertLessEqual(abs(actual["angular_velocity"] - expected.angular_velocity), tolerance)


class NativeChipmunkBridgeTest(unittest.TestCase):
    def test_bridge_reports_exact_pymunk_chipmunk_identity_and_source_hashes(self):
        import native_chipmunk_bridge as bridge
        pymunk = require_pymunk()

        info = bridge.installed_chipmunk_info()
        self.assertEqual(info["pymunk_version"], "6.2.1")
        self.assertEqual(
            info["chipmunk_version"],
            "7.0.3-0593976ef47fcb3957166bd342f6b2bafe4d0e44 ",
        )
        self.assertTrue(info["extension_path"].endswith("_chipmunk.abi3.so"))
        self.assertEqual(len(info["extension_sha256"]), 64)
        self.assertEqual(len(info["pymunk_version_file_sha256"]), 64)
        original = pymunk.chipmunk_version
        try:
            pymunk.chipmunk_version = "unexpected"
            with self.assertRaisesRegex(RuntimeError, "expected bundled Chipmunk"):
                bridge.installed_chipmunk_info()
        finally:
            pymunk.chipmunk_version = original

    def test_bridge_steps_two_independent_contact_worlds_like_scalar_pymunk(self):
        import native_chipmunk_bridge as bridge

        specs = [
            {"dynamic_position": (0.0, 0.44), "dynamic_velocity": (0.0, 2.0)},
            {
                "dynamic_position": (0.0, 0.44),
                "dynamic_velocity": (0.0, 2.0),
                "floor_type": "kinematic",
                "floor_velocity": (1.0, 0.0),
            },
        ]
        expected = [clone_world(spec) for spec in specs]
        bridged = [clone_world(spec) for spec in specs]

        for world in expected:
            world[0].step(DT)

        batch = bridge.BridgeBatch.from_worlds(
            [bridge.BridgeWorld(space=world[0], bodies=[world[1], world[2]], keepalive=world) for world in bridged]
        )
        batch.step_all(DT)

        for lane, world in enumerate(expected):
            player_state = batch.read_body(lane, 0)
            floor_state = batch.read_body(lane, 1)
            assert_body_close(self, player_state, world[1])
            assert_body_close(self, floor_state, world[2])

    def test_bridge_preserves_chipmunk_warm_start_over_200_steps(self):
        import native_chipmunk_bridge as bridge

        spec = {"dynamic_position": (0.3, 0.44), "dynamic_velocity": (2.0, 2.0)}
        expected = clone_world(spec)
        bridged = clone_world(spec)
        batch = bridge.BridgeBatch.from_worlds(
            [bridge.BridgeWorld(space=bridged[0], bodies=[bridged[1], bridged[2]], keepalive=bridged)]
        )

        for _ in range(200):
            expected[0].step(DT)
            batch.step_all(DT)

        assert_body_close(self, batch.read_body(0, 0), expected[1])
        assert_body_close(self, batch.read_body(0, 1), expected[2])

    def test_bridge_handles_finite_angles_without_axis_aligned_restriction(self):
        import native_chipmunk_bridge as bridge

        spec = {"dynamic_position": (0.0, 0.44), "dynamic_velocity": (0.0, 2.0), "angle": 0.125}
        expected = clone_world(spec)
        bridged = clone_world(spec)
        expected[0].step(DT)

        batch = bridge.BridgeBatch.from_worlds(
            [bridge.BridgeWorld(space=bridged[0], bodies=[bridged[1], bridged[2]], keepalive=bridged)]
        )
        batch.step_all(DT)
        assert_body_close(self, batch.read_body(0, 0), expected[1])

    def test_bridge_rejects_pymunk_deferred_queues_and_post_step_callbacks(self):
        import native_chipmunk_bridge as bridge

        world = clone_world({"dynamic_position": (0.0, 0.44), "dynamic_velocity": (0.0, 2.0)})
        space = world[0]
        space._post_step_callbacks["unsafe"] = lambda _space: None
        batch = bridge.BridgeBatch.from_worlds([bridge.BridgeWorld(space=space, bodies=[world[1]], keepalive=world)])
        with self.assertRaisesRegex(RuntimeError, "post-step callbacks"):
            batch.step_all(DT)

        world = clone_world({"dynamic_position": (0.0, 0.44), "dynamic_velocity": (0.0, 2.0)})
        space = world[0]
        space._add_later.add(world[1])
        batch = bridge.BridgeBatch.from_worlds([bridge.BridgeWorld(space=space, bodies=[world[1]], keepalive=world)])
        with self.assertRaisesRegex(RuntimeError, "deferred add queue"):
            batch.step_all(DT)

        world = clone_world({"dynamic_position": (0.0, 0.44), "dynamic_velocity": (0.0, 2.0)})
        space = world[0]
        space._remove_later.add(world[1])
        batch = bridge.BridgeBatch.from_worlds([bridge.BridgeWorld(space=space, bodies=[world[1]], keepalive=world)])
        with self.assertRaisesRegex(RuntimeError, "deferred remove queue"):
            batch.step_all(DT)

        world = clone_world({"dynamic_position": (0.0, 0.44), "dynamic_velocity": (0.0, 2.0)})
        space = world[0]
        space._locked = True
        batch = bridge.BridgeBatch.from_worlds([bridge.BridgeWorld(space=space, bodies=[world[1]], keepalive=world)])
        with self.assertRaisesRegex(RuntimeError, "already locked"):
            batch.step_all(DT)

        world = clone_world({"dynamic_position": (0.0, 0.44), "dynamic_velocity": (0.0, 2.0)})
        handler = world[0].add_collision_handler(1, 2)
        handler.begin = lambda arbiter, space, data: True
        batch = bridge.BridgeBatch.from_worlds([bridge.BridgeWorld(space=world[0], bodies=[world[1]], keepalive=world)])
        with self.assertRaisesRegex(RuntimeError, "collision handlers"):
            batch.step_all(DT)

    def test_bridge_revalidates_cached_pointer_identity_and_rejects_negative_indices(self):
        import native_chipmunk_bridge as bridge

        world = clone_world({"dynamic_position": (0.0, 0.44), "dynamic_velocity": (0.0, 2.0)})
        batch = bridge.BridgeBatch.from_worlds([bridge.BridgeWorld(space=world[0], bodies=[world[1]], keepalive=world)])
        with self.assertRaisesRegex(IndexError, "world_index"):
            batch.read_body(-1, 0)
        with self.assertRaisesRegex(IndexError, "body_index"):
            batch.set_body_state(0, -1, position=(0, 0), velocity=(0, 0), angle=0, angular_velocity=0)

        original_body_ptr = world[1]._body
        try:
            world[1]._body = None
            with self.assertRaisesRegex(RuntimeError, "body pointer changed"):
                batch.read_body(0, 0)
        finally:
            world[1]._body = original_body_ptr

        original_space_ptr = world[0]._space
        try:
            world[0]._space = None
            with self.assertRaisesRegex(RuntimeError, "space pointer changed"):
                batch.step_all(DT)
        finally:
            world[0]._space = original_space_ptr

    def test_bridge_can_set_and_get_body_state_without_reinitializing_world(self):
        import native_chipmunk_bridge as bridge

        world = clone_world({"dynamic_position": (0.0, 0.44), "dynamic_velocity": (0.0, 2.0)})
        batch = bridge.BridgeBatch.from_worlds([bridge.BridgeWorld(space=world[0], bodies=[world[1]], keepalive=world)])
        pointer_before = batch.space_pointers[0]
        batch.set_body_state(0, 0, position=(3.0, 4.0), velocity=(-1.0, 2.0), angle=0.75, angular_velocity=-0.25)
        state = batch.read_body(0, 0)

        self.assertEqual(batch.space_pointers[0], pointer_before)
        self.assertEqual(state["position"], (3.0, 4.0))
        self.assertEqual(state["velocity"], (-1.0, 2.0))
        self.assertEqual(state["angle"], 0.75)
        self.assertEqual(state["angular_velocity"], -0.25)
        self.assertEqual(world[1].position, (3.0, 4.0))

    def test_compiled_bridge_loads_without_linking_to_new_chipmunk_dependency(self):
        import native_chipmunk_bridge as bridge

        if not bridge.can_compile_bridge():
            self.skipTest("no C compiler available")

        with tempfile.TemporaryDirectory(prefix="ai2-chipmunk-bridge-") as tmp:
            compiled = bridge.compile_bridge(Path(tmp))
            self.assertTrue(Path(compiled["path"]).is_file())
            self.assertIn("chipmunk_bridge.c", " ".join(compiled["command"]))


if __name__ == "__main__":
    unittest.main()
