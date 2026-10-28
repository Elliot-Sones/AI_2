"""Parity tests for the source-faithful moving-platform reducer prototype."""

from __future__ import annotations

from ai2.paths import source_path

import json
import os
import subprocess
import sys
import tempfile
import unittest
import ctypes
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_FROZEN_ENV_PATH = Path(
    "/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/"
    "baselines/navigation-repair-20260905/payload/variants/corrected/"
    "UTMIST-AI2-main/environment/environment.py"
)
DEFAULT_E0_ADAPTIVE_NAV_TRACE = (
    ROOT
    / "baselines/throughput-scaling-20260906/cloud/replacement-session1/retrieved/artifacts/"
    "e0-replay/adaptive-navigation/traces/navigation_seed7_contact_fixture.jsonl"
)
FROZEN_ENV_PATH = Path(os.environ.get("AI2_FROZEN_STAGE_SOURCE", str(DEFAULT_FROZEN_ENV_PATH))).expanduser()
E0_ADAPTIVE_NAV_TRACE = Path(
    os.environ.get("AI2_NATIVE_REPLAY_TRACE", str(DEFAULT_E0_ADAPTIVE_NAV_TRACE))
).expanduser()


def actual_frozen_stage_step(position, waypoint1, waypoint2, moving_to_w2):
    """Execute the real frozen Stage.physics_process source on a tiny stub."""

    import ast
    import textwrap

    source = FROZEN_ENV_PATH.read_text()
    tree = ast.parse(source)
    stage = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Stage")
    method = next(
        node for node in stage.body if isinstance(node, ast.FunctionDef) and node.name == "physics_process"
    )
    method_source = ast.get_source_segment(source, method)
    namespace = {}
    exec("class FrozenStage:\n" + textwrap.indent(method_source, "    "), namespace)

    body = SimpleNamespace(position=tuple(position), velocity=(999.0, 999.0))
    instance = namespace["FrozenStage"]()
    instance.body = body
    instance.waypoint1 = tuple(waypoint1)
    instance.waypoint2 = tuple(waypoint2)
    instance.moving_to_w2 = bool(moving_to_w2)
    instance.physics_process(1.0 / 30.0)
    return tuple(body.velocity), bool(instance.moving_to_w2)


def first_comparable_e0_platform_transition():
    if not E0_ADAPTIVE_NAV_TRACE.is_file():
        return None
    previous = None
    with E0_ADAPTIVE_NAV_TRACE.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") != "record":
                continue
            platform = record["state"]["moving_platforms"][0]
            if previous is not None and record.get("tick") == previous["tick"] + 1:
                return previous, record, platform
            previous = {"tick": record.get("tick"), "platform": platform}
    return None


class NativeNavigationPlatformTest(unittest.TestCase):
    def test_scalar_reducer_matches_actual_frozen_stage_method_for_boundary_fixtures(self):
        import ai2.simulation.native_navigation_platform as nav

        fixtures = [
            ((0.0, 1.0), (1.0, 0.0), (-1.0, 2.0), True),
            ((0.0, 1.0), (1.0, 0.0), (-1.0, 2.0), False),
            ((-0.95, 1.95), (1.0, 0.0), (-1.0, 2.0), True),
            ((0.96, 0.04), (1.0, 0.0), (-1.0, 2.0), False),
            ((5.0, -3.0), (1.0, 0.0), (-1.0, 2.0), True),
            ((-7.5, 4.25), (1.0, 0.0), (-1.0, 2.0), False),
            ((1.0, 0.0), (0.0, 0.0), (0.001, 0.0), False),
        ]

        for position, waypoint1, waypoint2, moving_to_w2 in fixtures:
            with self.subTest(position=position, moving_to_w2=moving_to_w2):
                expected_velocity, expected_direction = actual_frozen_stage_step(
                    position, waypoint1, waypoint2, moving_to_w2
                )
                actual = nav.step_platform_scalar(
                    position=position,
                    waypoint1=waypoint1,
                    waypoint2=waypoint2,
                    moving_to_w2=moving_to_w2,
                )

                self.assertEqual(actual.moving_to_w2, expected_direction)
                self.assertAlmostEqual(actual.velocity[0], expected_velocity[0], places=7)
                self.assertAlmostEqual(actual.velocity[1], expected_velocity[1], places=7)

    def test_scalar_and_c_reference_preserve_double_precision_threshold_edges(self):
        import ai2.simulation.native_navigation_platform as nav

        if not nav.can_compile_cpu_reference():
            self.skipTest("no C compiler available for ctypes CPU reference")

        fixtures = [
            ((-0.900000001, 2.0), (1.0, 0.0), (-1.0, 2.0), True),
            ((-0.899999999, 2.0), (1.0, 0.0), (-1.0, 2.0), True),
            ((0.900000001, 0.0), (1.0, 0.0), (-1.0, 2.0), False),
            ((0.899999999, 0.0), (1.0, 0.0), (-1.0, 2.0), False),
        ]

        with tempfile.TemporaryDirectory(prefix="ai2-nav-platform-threshold-") as tmp:
            cpu_ref = nav.load_cpu_reference(Path(tmp))
            for position, waypoint1, waypoint2, moving_to_w2 in fixtures:
                with self.subTest(position=position, moving_to_w2=moving_to_w2):
                    expected_velocity, expected_direction = actual_frozen_stage_step(
                        position, waypoint1, waypoint2, moving_to_w2
                    )
                    scalar = nav.step_platform_scalar(
                        position=position,
                        waypoint1=waypoint1,
                        waypoint2=waypoint2,
                        moving_to_w2=moving_to_w2,
                    )
                    native = cpu_ref.step(
                        position=position,
                        waypoint1=waypoint1,
                        waypoint2=waypoint2,
                        moving_to_w2=moving_to_w2,
                    )
                    self.assertEqual(scalar.moving_to_w2, expected_direction)
                    self.assertEqual(native.moving_to_w2, expected_direction)
                    self.assertAlmostEqual(native.velocity[0], expected_velocity[0], places=12)
                    self.assertAlmostEqual(native.velocity[1], expected_velocity[1], places=12)

    def test_random_scalar_parity_uses_actual_frozen_source_not_formula_copy(self):
        import ai2.simulation.native_navigation_platform as nav

        positions, waypoint1, waypoint2, directions = nav.generate_fixture_arrays(64, seed=77)
        for index in range(64):
            expected_velocity, expected_direction = actual_frozen_stage_step(
                positions[index], waypoint1[index], waypoint2[index], directions[index]
            )
            actual = nav.step_platform_scalar(
                position=positions[index],
                waypoint1=waypoint1[index],
                waypoint2=waypoint2[index],
                moving_to_w2=directions[index],
            )
            self.assertEqual(actual.moving_to_w2, expected_direction)
            self.assertAlmostEqual(actual.velocity[0], expected_velocity[0], places=12)
            self.assertAlmostEqual(actual.velocity[1], expected_velocity[1], places=12)

    def test_compiled_cpu_reference_matches_scalar_reducer(self):
        import ai2.simulation.native_navigation_platform as nav

        if not nav.can_compile_cpu_reference():
            self.skipTest("no C compiler available for ctypes CPU reference")

        fixtures = [
            ((0.2, 0.8), (1.0, 0.0), (-1.0, 2.0), True),
            ((-0.2, 1.2), (1.0, 0.0), (-1.0, 2.0), False),
            ((-1.0, 2.0), (1.0, 0.0), (-1.0, 2.0), True),
            ((1.0, 0.0), (1.0, 0.0), (-1.0, 2.0), False),
        ]
        with tempfile.TemporaryDirectory(prefix="ai2-nav-platform-cpu-") as tmp:
            cpu_ref = nav.load_cpu_reference(Path(tmp))
            for position, waypoint1, waypoint2, moving_to_w2 in fixtures:
                with self.subTest(position=position, moving_to_w2=moving_to_w2):
                    actual = cpu_ref.step(
                        position=position,
                        waypoint1=waypoint1,
                        waypoint2=waypoint2,
                        moving_to_w2=moving_to_w2,
                    )
                    expected = nav.step_platform_scalar(
                        position=position,
                        waypoint1=waypoint1,
                        waypoint2=waypoint2,
                        moving_to_w2=moving_to_w2,
                    )
                    self.assertEqual(actual.moving_to_w2, expected.moving_to_w2)
                    self.assertAlmostEqual(actual.velocity[0], expected.velocity[0], places=6)
                    self.assertAlmostEqual(actual.velocity[1], expected.velocity[1], places=6)

    def test_compiled_c_reference_canonicalizes_nonzero_direction_uint8(self):
        import ai2.simulation.native_navigation_platform as nav

        if not nav.can_compile_cpu_reference():
            self.skipTest("no C compiler available for ctypes CPU reference")

        with tempfile.TemporaryDirectory(prefix="ai2-nav-platform-canon-") as tmp:
            cpu_ref = nav.load_cpu_reference(Path(tmp))
            velocity_x = ctypes.c_double()
            velocity_y = ctypes.c_double()
            next_direction = ctypes.c_uint8()
            cpu_ref._step(
                ctypes.c_double(0.0),
                ctypes.c_double(1.0),
                ctypes.c_double(1.0),
                ctypes.c_double(0.0),
                ctypes.c_double(-1.0),
                ctypes.c_double(2.0),
                ctypes.c_uint8(2),
                ctypes.byref(velocity_x),
                ctypes.byref(velocity_y),
                ctypes.byref(next_direction),
            )

        expected = nav.step_platform_scalar(
            position=(0.0, 1.0),
            waypoint1=(1.0, 0.0),
            waypoint2=(-1.0, 2.0),
            moving_to_w2=True,
        )
        self.assertEqual(next_direction.value, 1)
        self.assertAlmostEqual(velocity_x.value, expected.velocity[0], places=12)
        self.assertAlmostEqual(velocity_y.value, expected.velocity[1], places=12)

    def test_batch_reducer_preserves_permutation_order(self):
        import ai2.simulation.native_navigation_platform as nav

        positions = [(0.0, 1.0), (-0.95, 1.95), (0.96, 0.04), (5.0, -3.0)]
        waypoint1 = [(1.0, 0.0)] * len(positions)
        waypoint2 = [(-1.0, 2.0)] * len(positions)
        directions = [True, True, False, True]
        order = [2, 0, 3, 1]

        baseline = nav.step_platform_batch_python(positions, waypoint1, waypoint2, directions)
        permuted = nav.step_platform_batch_python(
            [positions[i] for i in order],
            [waypoint1[i] for i in order],
            [waypoint2[i] for i in order],
            [directions[i] for i in order],
        )

        for permuted_index, original_index in enumerate(order):
            self.assertEqual(permuted[permuted_index].moving_to_w2, baseline[original_index].moving_to_w2)
            self.assertAlmostEqual(
                permuted[permuted_index].velocity[0],
                baseline[original_index].velocity[0],
                places=7,
            )
            self.assertAlmostEqual(
                permuted[permuted_index].velocity[1],
                baseline[original_index].velocity[1],
                places=7,
            )

    def test_cuda_device_normalization_requires_explicit_index(self):
        import ai2.simulation.native_navigation_platform as nav

        self.assertEqual(nav.normalize_cuda_device("cuda", current_index=2), "cuda:2")
        self.assertEqual(nav.normalize_cuda_device("cuda:1", current_index=2), "cuda:1")
        self.assertEqual(nav.normalize_cuda_device("cuda:0", current_index=2), "cuda:0")
        with self.assertRaises(ValueError):
            nav.normalize_cuda_device("cpu", current_index=0)

    def test_resident_timing_label_discloses_dispatch_and_allocation(self):
        import ai2.simulation.native_navigation_platform as nav

        self.assertIn("optional kinematic position", nav.__doc__)
        self.assertEqual(nav.RESIDENT_TIMING_LABEL, "resident_call_dispatch_alloc")
        self.assertIn("ctypes", nav.RESIDENT_TIMING_SCOPE)
        self.assertIn("output allocation", nav.RESIDENT_TIMING_SCOPE)
        self.assertNotIn("pure kernel", nav.RESIDENT_TIMING_LABEL)

    def test_gpu_report_validation_fails_closed_on_parity_failures(self):
        import ai2.simulation.native_navigation_platform as nav

        valid = {
            "resident_call_dispatch_alloc": {
                "finite": True,
                "max_abs_velocity_error": 1e-12,
                "direction_mismatches": 0,
            },
            "host_roundtrip": {
                "finite": True,
                "max_abs_velocity_error": 1e-12,
                "direction_mismatches": 0,
            },
        }
        nav.validate_cuda_benchmark_report(valid, velocity_tolerance=1e-9)

        for patch in (
            {"resident_call_dispatch_alloc": {"finite": False}},
            {"resident_call_dispatch_alloc": {"max_abs_velocity_error": 1e-3}},
            {"host_roundtrip": {"direction_mismatches": 1}},
        ):
            invalid = json.loads(json.dumps(valid))
            for section, values in patch.items():
                invalid[section].update(values)
            with self.assertRaises(RuntimeError):
                nav.validate_cuda_benchmark_report(invalid, velocity_tolerance=1e-9)

    def test_replay_file_scalar_parity_uses_consecutive_e0_platform_states(self):
        import ai2.simulation.native_navigation_platform as nav

        if not E0_ADAPTIVE_NAV_TRACE.is_file():
            if "AI2_NATIVE_REPLAY_TRACE" in os.environ:
                self.fail(f"configured AI2_NATIVE_REPLAY_TRACE not found: {E0_ADAPTIVE_NAV_TRACE}")
            self.skipTest("adaptive E0 replay trace is not available")

        report = nav.replay_file_scalar_parity(E0_ADAPTIVE_NAV_TRACE)
        self.assertEqual(report["replay_file"], str(E0_ADAPTIVE_NAV_TRACE.resolve()))
        self.assertEqual(report["replay_file_sha256"], nav.file_sha256(E0_ADAPTIVE_NAV_TRACE))
        self.assertGreater(report["checked_transitions"], 1000)
        self.assertLessEqual(report["max_abs_velocity_error"], 1e-9)
        self.assertEqual(report["direction_mismatches"], 0)
        self.assertIsNone(report["first_mismatch"])

    def test_replay_validation_fails_closed_on_no_checks_or_mismatch(self):
        import ai2.simulation.native_navigation_platform as nav

        valid = {
            "replay_file": "fixture.jsonl",
            "checked_transitions": 1,
            "max_abs_velocity_error": 1e-12,
            "direction_mismatches": 0,
        }
        nav.validate_replay_parity_report(valid, velocity_tolerance=1e-9)

        for patch in (
            {"checked_transitions": 0},
            {"max_abs_velocity_error": 1e-3},
            {"direction_mismatches": 1},
        ):
            invalid = {**valid, **patch}
            with self.assertRaises(RuntimeError):
                nav.validate_replay_parity_report(invalid, velocity_tolerance=1e-9)

    def test_kinematic_scalar_transition_matches_e0_trace_with_new_velocity_order(self):
        import ai2.simulation.native_navigation_platform as nav

        transition = first_comparable_e0_platform_transition()
        if transition is None:
            if "AI2_NATIVE_REPLAY_TRACE" in os.environ:
                self.fail(f"configured AI2_NATIVE_REPLAY_TRACE not usable: {E0_ADAPTIVE_NAV_TRACE}")
            self.skipTest("adaptive E0 replay trace is not available")
        previous, _record, current_platform = transition
        previous_platform = previous["platform"]

        actual = nav.step_platform_kinematic_scalar(
            position=previous_platform["position"],
            waypoint1=previous_platform["waypoint1"],
            waypoint2=previous_platform["waypoint2"],
            moving_to_w2=previous_platform["moving_to_w2"],
            dt=1.0 / 30.0,
        )
        self.assertEqual(actual.moving_to_w2, current_platform["moving_to_w2"])
        self.assertAlmostEqual(actual.velocity[0], current_platform["velocity"][0], places=12)
        self.assertAlmostEqual(actual.velocity[1], current_platform["velocity"][1], places=12)
        self.assertAlmostEqual(actual.position[0], current_platform["position"][0], places=12)
        self.assertAlmostEqual(actual.position[1], current_platform["position"][1], places=12)

        previous_velocity_position = (
            previous_platform["position"][0] + previous_platform["velocity"][0] * (1.0 / 30.0),
            previous_platform["position"][1] + previous_platform["velocity"][1] * (1.0 / 30.0),
        )
        self.assertGreater(
            max(
                abs(previous_velocity_position[0] - current_platform["position"][0]),
                abs(previous_velocity_position[1] - current_platform["position"][1]),
            ),
            1e-6,
        )

    def test_replay_file_kinematic_parity_checks_captured_positions(self):
        import ai2.simulation.native_navigation_platform as nav

        if not E0_ADAPTIVE_NAV_TRACE.is_file():
            if "AI2_NATIVE_REPLAY_TRACE" in os.environ:
                self.fail(f"configured AI2_NATIVE_REPLAY_TRACE not found: {E0_ADAPTIVE_NAV_TRACE}")
            self.skipTest("adaptive E0 replay trace is not available")

        report = nav.replay_file_kinematic_parity(E0_ADAPTIVE_NAV_TRACE, dt=1.0 / 30.0)
        self.assertGreater(report["checked_transitions"], 1000)
        self.assertLessEqual(report["max_abs_velocity_error"], 1e-9)
        self.assertLessEqual(report["max_abs_position_error"], 1e-9)
        self.assertEqual(report["direction_mismatches"], 0)
        self.assertIsNone(report["first_mismatch"])

    def test_compiled_cpu_reference_matches_kinematic_scalar_transition(self):
        import ai2.simulation.native_navigation_platform as nav

        if not nav.can_compile_cpu_reference():
            self.skipTest("no C compiler available for ctypes CPU reference")

        with tempfile.TemporaryDirectory(prefix="ai2-nav-platform-kinematic-cpu-") as tmp:
            cpu_ref = nav.load_cpu_reference(Path(tmp))
            actual = cpu_ref.transition(
                position=(0.0, 1.0),
                waypoint1=(1.0, 0.0),
                waypoint2=(-1.0, 2.0),
                moving_to_w2=True,
                dt=1.0 / 30.0,
            )
        expected = nav.step_platform_kinematic_scalar(
            position=(0.0, 1.0),
            waypoint1=(1.0, 0.0),
            waypoint2=(-1.0, 2.0),
            moving_to_w2=True,
            dt=1.0 / 30.0,
        )
        self.assertEqual(actual.moving_to_w2, expected.moving_to_w2)
        self.assertAlmostEqual(actual.velocity[0], expected.velocity[0], places=12)
        self.assertAlmostEqual(actual.velocity[1], expected.velocity[1], places=12)
        self.assertAlmostEqual(actual.position[0], expected.position[0], places=12)
        self.assertAlmostEqual(actual.position[1], expected.position[1], places=12)

    def test_cuda_step_rejects_unsafe_pointer_inputs_and_uses_input_device_stream(self):
        import ai2.simulation.native_navigation_platform as nav

        class FakeTensor:
            ndim = 2

            def __init__(self, *, dtype, shape, device="cuda:0", contiguous=True, is_cuda=True):
                self.dtype = dtype
                self.shape = shape
                self.device = device
                self._contiguous = contiguous
                self.is_cuda = is_cuda

            def is_contiguous(self):
                return self._contiguous

            def data_ptr(self):
                return 1234

        class FakeCuda:
            def __init__(self):
                self.device_contexts = []
                self.stream_devices = []

            def device(self, device):
                outer = self

                class Context:
                    def __enter__(self_inner):
                        outer.device_contexts.append(str(device))

                    def __exit__(self_inner, exc_type, exc, tb):
                        return False

                return Context()

            def current_stream(self, device=None):
                self.stream_devices.append(str(device))
                return SimpleNamespace(cuda_stream=5678)

        class FakeTorch:
            float64 = "float64"
            uint8 = "uint8"

            def __init__(self):
                self.cuda = FakeCuda()

            def empty_like(self, tensor):
                ndim = 1 if len(tensor.shape) == 1 else 2
                result = FakeTensor(dtype=tensor.dtype, shape=tensor.shape, device=tensor.device)
                result.ndim = ndim
                return result

        fake_torch = FakeTorch()
        positions = FakeTensor(dtype=fake_torch.float64, shape=(4, 2), device="cuda:1")
        waypoint1 = FakeTensor(dtype=fake_torch.float64, shape=(4, 2), device="cuda:1")
        waypoint2 = FakeTensor(dtype=fake_torch.float64, shape=(4, 2), device="cuda:1")
        direction = FakeTensor(dtype=fake_torch.uint8, shape=(4,), device="cuda:1")
        direction.ndim = 1

        cuda = object.__new__(nav.CudaLibrary)
        calls = []

        def fake_step(*args):
            calls.append(args)
            return 0

        cuda._step = fake_step

        bad_direction = FakeTensor(
            dtype=fake_torch.uint8, shape=(4,), device="cuda:1", contiguous=False
        )
        bad_direction.ndim = 1
        with self.assertRaises(ValueError):
            cuda.step(fake_torch, positions, waypoint1, waypoint2, bad_direction)

        mixed_waypoint = FakeTensor(dtype=fake_torch.float64, shape=(4, 2), device="cuda:0")
        with self.assertRaises(ValueError):
            cuda.step(fake_torch, positions, mixed_waypoint, waypoint2, direction)

        zero_lanes = FakeTensor(dtype=fake_torch.float64, shape=(0, 2), device="cuda:1")
        with self.assertRaises(ValueError):
            cuda.step(fake_torch, zero_lanes, zero_lanes, zero_lanes, direction)

        velocities, next_direction = cuda.step(fake_torch, positions, waypoint1, waypoint2, direction)
        self.assertEqual(velocities.device, "cuda:1")
        self.assertEqual(next_direction.device, "cuda:1")
        self.assertEqual(fake_torch.cuda.device_contexts, ["cuda:1"])
        self.assertEqual(fake_torch.cuda.stream_devices, ["cuda:1"])
        self.assertEqual(len(calls), 1)

    def test_cuda_benchmark_memory_stats_use_requested_device(self):
        import ai2.simulation.native_navigation_platform as nav

        class FakeTensor:
            def __init__(self, data, *, dtype, device="cpu"):
                self.data = data
                self.dtype = dtype
                self.device = str(device)
                self.is_cuda = self.device.startswith("cuda")
                self.ndim = 2 if data and isinstance(data[0], (list, tuple)) else 1
                if self.ndim == 2:
                    self.shape = (len(data), len(data[0]) if data else 0)
                else:
                    self.shape = (len(data),)

            def contiguous(self):
                return self

            def to(self, device):
                return FakeTensor(self.data, dtype=self.dtype, device=str(device))

            def cpu(self):
                return FakeTensor(self.data, dtype=self.dtype, device="cpu")

            def tolist(self):
                return self.data

        class FakeFinite:
            def all(self):
                return self

            def item(self):
                return True

        class FakeCuda:
            def __init__(self):
                self.reset_devices = []
                self.max_devices = []
                self.synchronize_devices = []

            def is_available(self):
                return True

            def current_device(self):
                return 0

            def synchronize(self, device=None):
                self.synchronize_devices.append(None if device is None else str(device))

            def reset_peak_memory_stats(self, device=None):
                self.reset_devices.append(None if device is None else str(device))

            def max_memory_allocated(self, device=None):
                self.max_devices.append(None if device is None else str(device))
                return 0

        class FakeTorch:
            float64 = "float64"
            uint8 = "uint8"

            def __init__(self):
                self.cuda = FakeCuda()

            def device(self, device):
                return str(device)

            def tensor(self, data, *, dtype, device="cpu"):
                normalized = [list(item) if isinstance(item, tuple) else item for item in data]
                return FakeTensor(normalized, dtype=dtype, device=str(device))

            def isfinite(self, _tensor):
                return FakeFinite()

        fake_torch = FakeTorch()

        class FakeCudaLibrary:
            def step(self, _torch_mod, positions, waypoint1, waypoint2, moving_to_w2):
                expected = nav.step_platform_batch_python(
                    positions.data,
                    waypoint1.data,
                    waypoint2.data,
                    [bool(item) for item in moving_to_w2.data],
                )
                velocities = [list(step.velocity) for step in expected]
                directions = [1 if step.moving_to_w2 else 0 for step in expected]
                return (
                    FakeTensor(velocities, dtype=fake_torch.float64, device=positions.device),
                    FakeTensor(directions, dtype=fake_torch.uint8, device=positions.device),
                )

        original_torch = sys.modules.get("torch")
        original_can_run = nav.can_run_cuda_benchmark
        original_load_cuda = nav.load_cuda_library
        original_normalize = nav.normalize_cuda_device
        sys.modules["torch"] = fake_torch
        try:
            nav.can_run_cuda_benchmark = lambda: True
            nav.load_cuda_library = lambda _build_dir: FakeCudaLibrary()
            nav.normalize_cuda_device = lambda device="cuda", *, current_index=None: "cuda:1"
            report = nav.run_cuda_benchmark(
                2,
                iterations=1,
                warmup=1,
                seed=20260906,
                build_dir=Path("/tmp/unused-ai2-nav-fake"),
                device="cuda:1",
            )
        finally:
            nav.can_run_cuda_benchmark = original_can_run
            nav.load_cuda_library = original_load_cuda
            nav.normalize_cuda_device = original_normalize
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        self.assertEqual(report["device"], "cuda:1")
        self.assertEqual(fake_torch.cuda.reset_devices, ["cuda:1", "cuda:1"])
        self.assertEqual(fake_torch.cuda.max_devices, ["cuda:1", "cuda:1"])

    def test_gpu_batch_matches_cpu_for_required_lane_counts_when_available(self):
        import ai2.simulation.native_navigation_platform as nav

        if not nav.can_run_cuda_benchmark():
            self.skipTest("CUDA torch and nvcc are not available")

        for lanes in (256, 1024, 4096):
            with self.subTest(lanes=lanes):
                report = nav.run_cuda_parity_once(lanes=lanes, seed=20260906)
                self.assertEqual(report["lanes"], lanes)
                self.assertTrue(report["finite"])
                self.assertLessEqual(report["max_abs_velocity_error"], 1e-5)
                self.assertEqual(report["direction_mismatches"], 0)
                self.assertTrue(report["permutation"]["finite"])
                self.assertLessEqual(report["permutation"]["max_abs_velocity_error"], 1e-5)
                self.assertEqual(report["permutation"]["direction_mismatches"], 0)
                transition_report = nav.run_cuda_kinematic_parity_once(lanes=lanes, seed=20260906)
                self.assertTrue(transition_report["finite"])
                self.assertLessEqual(transition_report["max_abs_velocity_error"], 1e-9)
                self.assertLessEqual(transition_report["max_abs_position_error"], 1e-9)
                self.assertEqual(transition_report["direction_mismatches"], 0)

    def test_cuda_canonicalizes_nonzero_direction_uint8_when_available(self):
        import ai2.simulation.native_navigation_platform as nav

        if not nav.can_run_cuda_benchmark():
            self.skipTest("CUDA torch and nvcc are not available")
        import torch

        with tempfile.TemporaryDirectory(prefix="ai2-nav-platform-canon-cuda-") as tmp:
            cuda = nav.load_cuda_library(Path(tmp))
            device = torch.device(nav.normalize_cuda_device("cuda"))
            positions = torch.tensor([[0.0, 1.0]], dtype=torch.float64, device=device)
            waypoint1 = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=device)
            waypoint2 = torch.tensor([[-1.0, 2.0]], dtype=torch.float64, device=device)
            direction = torch.tensor([2], dtype=torch.uint8, device=device)
            velocities, next_direction = cuda.step(torch, positions, waypoint1, waypoint2, direction)
            torch.cuda.synchronize(device)

        self.assertEqual(next_direction.cpu().tolist(), [1])
        self.assertTrue(torch.isfinite(velocities).all().item())

    def test_cli_writes_scalar_provenance_and_refuses_cpu_fallback_for_gpu_mode(self):
        import ai2.simulation.native_navigation_platform as nav

        with tempfile.TemporaryDirectory(prefix="ai2-nav-platform-cli-") as tmp:
            output = Path(tmp) / "platform.json"
            scalar_args = [
                sys.executable,
                str(source_path(ROOT, "native_navigation_platform.py")),
                "--output",
                str(output),
                "--mode",
                "scalar",
                "--seed",
                "123",
                "--dt",
                str(1.0 / 30.0),
            ]
            if E0_ADAPTIVE_NAV_TRACE.is_file():
                scalar_args.extend(["--replay-file", str(E0_ADAPTIVE_NAV_TRACE)])
            elif "AI2_NATIVE_REPLAY_TRACE" in os.environ:
                self.fail(f"configured AI2_NATIVE_REPLAY_TRACE not found: {E0_ADAPTIVE_NAV_TRACE}")
            result = subprocess.run(
                scalar_args,
                cwd=ROOT,
                env={**os.environ, "AI2_FROZEN_STAGE_SOURCE": str(FROZEN_ENV_PATH)},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            payload = json.loads(output.read_text())
            self.assertEqual(
                payload["scope"],
                "platform_velocity_direction_and_optional_kinematic_position_transition_only",
            )
            self.assertEqual(payload["provenance"]["seed"], 123)
            self.assertIn("frozen_stage_physics_process", payload["source"])
            self.assertEqual(payload["source"]["frozen_stage_file"], str(FROZEN_ENV_PATH))
            self.assertEqual(
                payload["source"]["frozen_stage_sha256"],
                nav.file_sha256(FROZEN_ENV_PATH),
            )
            if E0_ADAPTIVE_NAV_TRACE.is_file():
                self.assertEqual(payload["replay_parity"][0]["direction_mismatches"], 0)
                self.assertGreater(payload["replay_parity"][0]["checked_transitions"], 1000)
                self.assertEqual(payload["kinematic_replay_parity"][0]["direction_mismatches"], 0)
                self.assertLessEqual(payload["kinematic_replay_parity"][0]["max_abs_position_error"], 1e-9)

            gpu_output = Path(tmp) / "gpu.json"
            gpu_result = subprocess.run(
                [
                    sys.executable,
                    str(source_path(ROOT, "native_navigation_platform.py")),
                    "--output",
                    str(gpu_output),
                    "--mode",
                    "gpu",
                    "--lanes",
                    "256",
                    "--iterations",
                    "1",
                    "--warmup",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            if gpu_result.returncode == 0:
                gpu_payload = json.loads(gpu_output.read_text())
                self.assertEqual(gpu_payload["mode"], "gpu")
                self.assertEqual(gpu_payload["gpu"]["benchmarks"][0]["lanes"], 256)
            else:
                self.assertEqual(gpu_result.returncode, 2, gpu_result.stdout)
                self.assertIn("refusing CPU fallback", gpu_result.stdout)


if __name__ == "__main__":
    unittest.main()
