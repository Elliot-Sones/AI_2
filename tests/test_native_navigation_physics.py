"""Parity tests for the first faithful native Chipmunk body integration slice."""

from __future__ import annotations

import ctypes
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require_pymunk():
    try:
        import pymunk
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("Pymunk oracle is unavailable in this Python") from exc
    if getattr(pymunk, "version", None) != "6.2.1":
        raise unittest.SkipTest(f"Pymunk oracle must be 6.2.1, got {getattr(pymunk, 'version', None)!r}")
    return pymunk


def pymunk_body_after_step(
    *,
    gravity=(0.0, 17.808),
    damping=0.97,
    dt=1.0 / 30.0,
    no_gravity=False,
):
    pymunk = require_pymunk()
    body = pymunk.Body(1.0, 1e9)
    body.position = (2.25, -3.5)
    body.velocity = (4.0, -2.0)
    body.force = (0.5, -1.25)
    body.angle = 0.25
    body.angular_velocity = 3.0
    body.torque = 7.5

    # Chipmunk's Space.step() integrates positions before velocities.
    pymunk.Body.update_position(body, dt)
    pymunk.Body.update_velocity(body, (0.0, 0.0) if no_gravity else gravity, damping, dt)
    return body


def pymunk_space_after_step(*, space_damping=0.97, dt=1.0 / 30.0):
    pymunk = require_pymunk()
    space = pymunk.Space()
    space.gravity = (0.0, 17.808)
    space.damping = space_damping
    body = pymunk.Body(1.0, 1e9)
    body.position = (2.25, -3.5)
    body.velocity = (4.0, -2.0)
    body.force = (0.5, -1.25)
    body.angle = 0.25
    body.angular_velocity = 3.0
    body.torque = 7.5
    space.add(body)
    space.step(dt)
    return body


def pymunk_space_body_after_step(body, *, space_damping=0.97, dt=1.0 / 30.0, no_gravity=False):
    pymunk = require_pymunk()
    space = pymunk.Space()
    space.gravity = (0.0, 17.808)
    space.damping = space_damping
    if no_gravity:
        body.velocity_func = lambda wrapped_body, _gravity, damping, delta: pymunk.Body.update_velocity(
            wrapped_body,
            (0.0, 0.0),
            damping,
            delta,
        )
    space.add(body)
    space.step(dt)
    return body


def pack_invalid_dynamic_state(physics):
    values = physics.pack_state(
        physics.BodyState(position=(0.0, 0.0), velocity=(1.0, 1.0), mass=1.0, moment=1e9)
    )
    values[11] = -1.0
    return values


class NativeNavigationPhysicsTest(unittest.TestCase):
    def test_scalar_dynamic_body_step_matches_pymunk_default_position_then_velocity(self):
        import native_navigation_physics as physics

        expected = pymunk_body_after_step()
        actual = physics.step_body_scalar(
            physics.BodyState(
                position=(2.25, -3.5),
                velocity=(4.0, -2.0),
                force=(0.5, -1.25),
                angle=0.25,
                angular_velocity=3.0,
                torque=7.5,
                mass=1.0,
                moment=1e9,
            ),
            gravity=(0.0, 17.808),
            damping_factor=0.97,
            dt=1.0 / 30.0,
        )

        self.assertAlmostEqual(actual.position[0], expected.position.x, places=12)
        self.assertAlmostEqual(actual.position[1], expected.position.y, places=12)
        self.assertAlmostEqual(actual.velocity[0], expected.velocity.x, places=12)
        self.assertAlmostEqual(actual.velocity[1], expected.velocity.y, places=12)
        self.assertAlmostEqual(actual.angle, expected.angle, places=12)
        self.assertAlmostEqual(actual.angular_velocity, expected.angular_velocity, places=12)
        self.assertEqual(actual.force, (0.0, 0.0))
        self.assertEqual(actual.torque, 0.0)

    def test_scalar_dynamic_body_step_matches_real_space_step_with_space_damping_exponent(self):
        import native_navigation_physics as physics

        expected = pymunk_space_after_step(space_damping=0.97, dt=1.0 / 30.0)
        actual = physics.step_body_scalar(
            physics.BodyState(
                position=(2.25, -3.5),
                velocity=(4.0, -2.0),
                force=(0.5, -1.25),
                angle=0.25,
                angular_velocity=3.0,
                torque=7.5,
                mass=1.0,
                moment=1e9,
            ),
            gravity=(0.0, 17.808),
            space_damping=0.97,
            dt=1.0 / 30.0,
        )

        self.assertAlmostEqual(actual.position[0], expected.position.x, places=12)
        self.assertAlmostEqual(actual.position[1], expected.position.y, places=12)
        self.assertAlmostEqual(actual.velocity[0], expected.velocity.x, places=12)
        self.assertAlmostEqual(actual.velocity[1], expected.velocity.y, places=12)
        self.assertAlmostEqual(actual.angular_velocity, expected.angular_velocity, places=12)

    def test_scalar_kinematic_body_step_matches_real_space_step_orientation(self):
        import native_navigation_physics as physics

        pymunk = require_pymunk()
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = (-1.0, 1.5)
        body.velocity = (2.0, -3.0)
        body.angle = 0.1
        body.angular_velocity = 0.4
        expected = pymunk_space_body_after_step(body, space_damping=0.5, dt=0.25)

        actual = physics.step_body_scalar(
            physics.BodyState(
                body_type=physics.BODY_TYPE_KINEMATIC,
                position=(-1.0, 1.5),
                velocity=(2.0, -3.0),
                angle=0.1,
                angular_velocity=0.4,
                mass=float("inf"),
                moment=float("inf"),
            ),
            gravity=(0.0, 17.808),
            space_damping=0.5,
            dt=0.25,
        )

        self.assertAlmostEqual(actual.position[0], expected.position.x, places=12)
        self.assertAlmostEqual(actual.position[1], expected.position.y, places=12)
        self.assertAlmostEqual(actual.angle, expected.angle, places=12)
        self.assertAlmostEqual(actual.velocity[0], expected.velocity.x, places=12)
        self.assertAlmostEqual(actual.velocity[1], expected.velocity.y, places=12)

    def test_scalar_static_body_step_matches_real_space_step_no_motion(self):
        import native_navigation_physics as physics

        pymunk = require_pymunk()
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (5.0, 6.0)
        body.angle = 0.25
        expected = pymunk_space_body_after_step(body, space_damping=0.5, dt=0.25)

        actual = physics.step_body_scalar(
            physics.BodyState(
                body_type=physics.BODY_TYPE_STATIC,
                position=(5.0, 6.0),
                angle=0.25,
                mass=float("inf"),
                moment=float("inf"),
            ),
            gravity=(0.0, 17.808),
            space_damping=0.5,
            dt=0.25,
        )

        self.assertAlmostEqual(actual.position[0], expected.position.x, places=12)
        self.assertAlmostEqual(actual.position[1], expected.position.y, places=12)
        self.assertAlmostEqual(actual.angle, expected.angle, places=12)

    def test_scalar_no_gravity_body_step_matches_real_space_step_callback(self):
        import native_navigation_physics as physics

        pymunk = require_pymunk()
        body = pymunk.Body(1.0, 1e9)
        body.position = (2.25, -3.5)
        body.velocity = (4.0, -2.0)
        body.force = (0.5, -1.25)
        body.angle = 0.25
        body.angular_velocity = 3.0
        body.torque = 7.5
        expected = pymunk_space_body_after_step(body, space_damping=0.97, dt=1.0 / 30.0, no_gravity=True)

        actual = physics.step_body_scalar(
            physics.BodyState(
                velocity_mode=physics.VELOCITY_MODE_NO_GRAVITY,
                position=(2.25, -3.5),
                velocity=(4.0, -2.0),
                force=(0.5, -1.25),
                angle=0.25,
                angular_velocity=3.0,
                torque=7.5,
                mass=1.0,
                moment=1e9,
            ),
            gravity=(0.0, 17.808),
            space_damping=0.97,
            dt=1.0 / 30.0,
        )

        self.assertAlmostEqual(actual.position[0], expected.position.x, places=12)
        self.assertAlmostEqual(actual.position[1], expected.position.y, places=12)
        self.assertAlmostEqual(actual.velocity[0], expected.velocity.x, places=12)
        self.assertAlmostEqual(actual.velocity[1], expected.velocity.y, places=12)
        self.assertAlmostEqual(actual.angular_velocity, expected.angular_velocity, places=12)

    def test_scalar_no_gravity_mode_matches_frozen_dodge_velocity_callback(self):
        import native_navigation_physics as physics

        expected = pymunk_body_after_step(no_gravity=True)
        actual = physics.step_body_scalar(
            physics.BodyState(
                position=(2.25, -3.5),
                velocity=(4.0, -2.0),
                force=(0.5, -1.25),
                angle=0.25,
                angular_velocity=3.0,
                torque=7.5,
                mass=1.0,
                moment=1e9,
                velocity_mode=physics.VELOCITY_MODE_NO_GRAVITY,
            ),
            gravity=(0.0, 17.808),
            damping_factor=0.97,
            dt=1.0 / 30.0,
        )

        self.assertAlmostEqual(actual.velocity[0], expected.velocity.x, places=12)
        self.assertAlmostEqual(actual.velocity[1], expected.velocity.y, places=12)
        self.assertAlmostEqual(actual.angular_velocity, expected.angular_velocity, places=12)
        self.assertEqual(actual.force, (0.0, 0.0))
        self.assertEqual(actual.torque, 0.0)

    def test_bias_velocity_affects_position_and_is_reset_without_polluting_velocity(self):
        import native_navigation_physics as physics

        actual = physics.step_body_scalar(
            physics.BodyState(
                position=(1.0, 2.0),
                velocity=(3.0, 4.0),
                force=(0.0, 0.0),
                angle=0.5,
                angular_velocity=0.25,
                torque=0.0,
                mass=1.0,
                moment=1e9,
                velocity_bias=(0.75, -0.5),
                angular_velocity_bias=-0.125,
            ),
            gravity=(0.0, 0.0),
            damping_factor=1.0,
            dt=0.2,
        )

        self.assertAlmostEqual(actual.position[0], 1.0 + (3.0 + 0.75) * 0.2, places=12)
        self.assertAlmostEqual(actual.position[1], 2.0 + (4.0 - 0.5) * 0.2, places=12)
        self.assertAlmostEqual(actual.velocity[0], 3.0, places=12)
        self.assertAlmostEqual(actual.velocity[1], 4.0, places=12)
        self.assertAlmostEqual(actual.angle, 0.5 + (0.25 - 0.125) * 0.2, places=12)
        self.assertEqual(actual.velocity_bias, (0.0, 0.0))
        self.assertEqual(actual.angular_velocity_bias, 0.0)

    def test_kinematic_body_integrates_position_but_skips_velocity_update(self):
        import native_navigation_physics as physics

        actual = physics.step_body_scalar(
            physics.BodyState(
                body_type=physics.BODY_TYPE_KINEMATIC,
                position=(-1.0, 1.5),
                velocity=(2.0, -3.0),
                force=(10.0, 20.0),
                angle=0.1,
                angular_velocity=0.4,
                torque=99.0,
                mass=math.inf,
                moment=math.inf,
            ),
            gravity=(0.0, 17.808),
            damping_factor=0.5,
            dt=0.25,
        )

        self.assertAlmostEqual(actual.position[0], -0.5, places=12)
        self.assertAlmostEqual(actual.position[1], 0.75, places=12)
        self.assertEqual(actual.velocity, (2.0, -3.0))
        self.assertEqual(actual.force, (10.0, 20.0))
        self.assertEqual(actual.torque, 99.0)

    def test_python_api_rejects_fractional_enum_fields(self):
        import native_navigation_physics as physics

        with self.assertRaises(ValueError):
            physics.step_body_scalar(
                physics.BodyState(body_type=0.5),
                gravity=(0.0, 17.808),
                damping_factor=0.97,
                dt=1.0 / 30.0,
            )
        with self.assertRaises(ValueError):
            physics.step_body_scalar(
                physics.BodyState(velocity_mode=0.5),
                gravity=(0.0, 17.808),
                damping_factor=0.97,
                dt=1.0 / 30.0,
            )

    def test_python_api_rejects_nonzero_cog_and_negative_damping_factor(self):
        import native_navigation_physics as physics

        with self.assertRaises(ValueError):
            physics.step_body_scalar(
                physics.BodyState(center_of_gravity=(0.1, 0.0)),
                gravity=(0.0, 17.808),
                damping_factor=0.97,
                dt=1.0 / 30.0,
            )
        with self.assertRaises(ValueError):
            physics.step_body_scalar(
                physics.BodyState(),
                gravity=(0.0, 17.808),
                damping_factor=-0.97,
                dt=1.0 / 30.0,
            )

    def test_compiled_c_reference_matches_scalar_for_batch_and_preserves_outputs_on_invalid_lane(self):
        import native_navigation_physics as physics

        if not physics.can_compile_cpu_reference():
            self.skipTest("no C compiler available")

        states = [
            physics.BodyState(position=(0.0, 0.0), velocity=(1.0, 2.0), force=(3.0, 4.0), mass=1.0, moment=1e9),
            physics.BodyState(
                position=(2.0, -2.0),
                velocity=(-1.0, 0.5),
                force=(0.0, 1.0),
                angle=0.4,
                angular_velocity=0.2,
                torque=3.0,
                mass=2.0,
                moment=5.0,
            ),
        ]
        with tempfile.TemporaryDirectory(prefix="ai2-nav-physics-cpu-") as tmp:
            ref = physics.load_cpu_reference(Path(tmp))
            actual = ref.step_batch(states, gravity=(0.0, 17.808), damping_factor=0.97, dt=1.0 / 30.0)

            for lane, state in enumerate(states):
                expected = physics.step_body_scalar(
                    state,
                    gravity=(0.0, 17.808),
                    damping_factor=0.97,
                    dt=1.0 / 30.0,
                )
                self.assertAlmostEqual(actual[lane].position[0], expected.position[0], places=12)
                self.assertAlmostEqual(actual[lane].position[1], expected.position[1], places=12)
                self.assertAlmostEqual(actual[lane].velocity[0], expected.velocity[0], places=12)
                self.assertAlmostEqual(actual[lane].velocity[1], expected.velocity[1], places=12)
                self.assertAlmostEqual(actual[lane].angle, expected.angle, places=12)
                self.assertAlmostEqual(actual[lane].angular_velocity, expected.angular_velocity, places=12)

            sentinel = (ctypes.c_double * physics.BODY_STATE_FIELD_COUNT)(
                *([12345.0] * physics.BODY_STATE_FIELD_COUNT)
            )
            invalid = (ctypes.c_double * physics.BODY_STATE_FIELD_COUNT)(
                *pack_invalid_dynamic_state(physics)
            )
            error = ref._step_batch(
                invalid,
                sentinel,
                ctypes.c_size_t(1),
                ctypes.c_double(0.0),
                ctypes.c_double(17.808),
                ctypes.c_double(0.97),
                ctypes.c_double(1.0 / 30.0),
            )
            self.assertNotEqual(error, 0)
            self.assertEqual(list(sentinel), [12345.0] * physics.BODY_STATE_FIELD_COUNT)

    def test_compiled_c_reference_rejects_fractional_enum_fields(self):
        import native_navigation_physics as physics

        if not physics.can_compile_cpu_reference():
            self.skipTest("no C compiler available")

        with tempfile.TemporaryDirectory(prefix="ai2-nav-physics-enum-") as tmp:
            ref = physics.load_cpu_reference(Path(tmp))
            for index in (0, 1):
                with self.subTest(index=index):
                    sentinel = (ctypes.c_double * physics.BODY_STATE_FIELD_COUNT)(
                        *([54321.0] * physics.BODY_STATE_FIELD_COUNT)
                    )
                    invalid = physics.pack_state(physics.BodyState())
                    invalid[index] = 0.5
                    invalid_array = (ctypes.c_double * physics.BODY_STATE_FIELD_COUNT)(*invalid)
                    error = ref._step_batch(
                        invalid_array,
                        sentinel,
                        ctypes.c_size_t(1),
                        ctypes.c_double(0.0),
                        ctypes.c_double(17.808),
                        ctypes.c_double(0.97),
                        ctypes.c_double(1.0 / 30.0),
                    )
                    self.assertNotEqual(error, 0)
                    self.assertEqual(list(sentinel), [54321.0] * physics.BODY_STATE_FIELD_COUNT)

    def test_cuda_batch_matches_scalar_when_runtime_available(self):
        import native_navigation_physics as physics

        if not physics.can_run_cuda():
            self.skipTest("CUDA runtime or nvcc unavailable")

        report = physics.run_cuda_parity_once(lanes=256, seed=20260906)
        self.assertEqual(report["position_mismatches"], 0)
        self.assertEqual(report["velocity_mismatches"], 0)
        self.assertLessEqual(report["max_abs_error"], 1e-9)
        self.assertEqual(report["permutation"]["position_mismatches"], 0)
        self.assertEqual(report["permutation"]["velocity_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
