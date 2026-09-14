"""Experimental profiler binding, provenance, and fail-closed artifact checks."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack, nullcontext
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def fake_classes():
    class Policy:
        def forward(self, value):
            return value

        def predict_values(self, value):
            return value

    class Env:
        def step_tensor(self, value):
            return value

    class Buffer:
        def add(self, value):
            return value

        def compute_returns_and_advantage(self, value):
            return value

    class BasePPO:
        def train(self, value):
            return value + 1

        def collect_rollouts(self, value):
            return Policy().forward(value)

    return Policy, Env, Buffer, BasePPO


class ExperimentalProfilerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("profile_native_ppo_optimizations"),
            "experimental PPO profiler is missing",
        )
        self.module = importlib.import_module("profile_native_ppo_optimizations")

    def test_parser_passes_every_experimental_flag_and_short_defaults(self):
        import experimental_native_lane_training as experiment

        flags = ["--" + name.replace("_", "-") for name in experiment.OptimizationOptions.__dataclass_fields__]
        args = self.module.build_parser().parse_args(["--output", "/tmp/profile-ppo-options", *flags])
        self.assertEqual((args.warmup, args.cycles), (1, 3))
        self.assertTrue(args.native_lanes)
        self.assertEqual((args.device, args.lane_device), ("cuda", "cuda"))
        for name in experiment.OptimizationOptions.__dataclass_fields__:
            self.assertIs(getattr(args, name), True)

    def test_factory_instruments_exact_dynamic_class_and_restores_all_methods(self):
        policy, env, buffer, base = fake_classes()
        created = []
        supplied = []

        def factory(options, *, on_loaded):
            class ActualPPO(base):
                pass

            created.append(ActualPPO)
            supplied.append((options, on_loaded))
            return ActualPPO

        experiment = SimpleNamespace(make_ppo_class=factory)
        callback, options = object(), object()
        original = policy.forward
        with self.module.instrument_ppo_factory(
            experiment, policy_class=policy, env_class=env, buffer_class=buffer,
            record_function=lambda _name: nullcontext(),
        ) as captures:
            actual = experiment.make_ppo_class(options, on_loaded=callback)
            self.assertIs(actual, created[0])
            self.assertEqual(supplied, [(options, callback)])
            self.assertEqual(actual().train(4), 5)
            self.assertEqual(actual().collect_rollouts(4), 4)
            self.assertEqual(captures[0]["stats"]["ppo_update"]["calls"], 1)
            self.assertEqual(captures[0]["stats"]["collect_rollout"]["calls"], 1)
        self.assertIs(experiment.make_ppo_class, factory)
        self.assertIs(policy.forward, original)
        self.assertNotIn("train", actual.__dict__)
        self.assertNotIn("collect_rollouts", actual.__dict__)

    def test_factory_and_partial_instrumentation_restore_on_failures(self):
        policy, env, buffer, base = fake_classes()
        failure = RuntimeError("training failed")

        def factory(*args, **kwargs):
            class ActualPPO(base):
                def train(self, value):
                    raise failure
            return ActualPPO

        experiment = SimpleNamespace(make_ppo_class=factory)
        original = policy.forward
        with self.assertRaises(RuntimeError) as caught:
            with self.module.instrument_ppo_factory(
                experiment, policy_class=policy, env_class=env, buffer_class=buffer,
                record_function=lambda _name: nullcontext(),
            ) as captures:
                experiment.make_ppo_class(None)().train(1)
        self.assertIs(caught.exception, failure)
        self.assertIs(experiment.make_ppo_class, factory)
        self.assertIs(policy.forward, original)
        self.assertEqual(captures[0]["stats"]["ppo_update"]["failed_calls"], 1)

        del env.step_tensor
        with self.assertRaises(AttributeError):
            with self.module.instrument_ppo_factory(
                experiment, policy_class=policy, env_class=env, buffer_class=buffer,
                record_function=lambda _name: nullcontext(),
            ):
                experiment.make_ppo_class(None)
        self.assertIs(experiment.make_ppo_class, factory)
        self.assertIs(policy.forward, original)

    def test_source_hashes_cover_all_experiment_and_native_inputs(self):
        import experimental_native_lane_training as experiment
        hashes = self.module.source_hashes()
        for name in (*experiment.EXPERIMENT_FILES, "profile_native_ppo_optimizations.py",
                     "profile_native_lane_training.py", "native/navigation_lanes.cu"):
            self.assertEqual(len(hashes[name]), 64)

    def test_compiler_counter_snapshot_does_not_import_or_reset_dynamo(self):
        marker = SimpleNamespace(counters={"stats": {"unique_graphs": 3}})
        with patch.dict(sys.modules, {"torch._dynamo.utils": marker}):
            self.assertEqual(self.module.compiler_counters(), {"stats": {"unique_graphs": 3}})
            self.assertEqual(marker.counters["stats"]["unique_graphs"], 3)
        with patch.dict(sys.modules, {"torch._dynamo.utils": None}):
            self.assertIsNone(self.module.compiler_counters())

    @contextmanager
    def mocked_run(self, output, *, cuda_events=1, drift=False, failure=None,
                   distribution_change=False, cuda_available=True, profiler_failure=None):
        import torch
        import experimental_native_lane_training as experiment

        module = self.module

        class Profiler:
            def __enter__(self):
                if profiler_failure is not None:
                    raise profiler_failure
                return self

            def __exit__(self, *exc):
                return False

            def key_averages(self):
                return []

            def events(self):
                return [SimpleNamespace(device_type="DeviceType.CUDA")] * cuda_events

            def export_chrome_trace(self, path):
                Path(path).write_text('{"traceEvents": []}')

        passed_args = []
        classes = []

        def factory(*args, **kwargs):
            _policy, _env, _buffer, base = fake_classes()
            class ActualPPO(base):
                pass
            classes.append(ActualPPO)
            return ActualPPO

        def train(args):
            passed_args.append(args)
            actual = experiment.make_ppo_class(experiment.options_from_args(args), on_loaded=None)
            actual().train(3)
            if distribution_change:
                torch.distributions.Distribution.set_default_validate_args(
                    not torch.distributions.Distribution._validate_args
                )
            if failure is not None:
                raise failure
            return 0

        before = {"source.py": "before"}
        after = {"source.py": "after"} if drift else before
        args = module.build_parser().parse_args(["--output", output, "--deferred-metrics", "--optimizer-cpu-steps"])
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=cuda_available))
            stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="mock CUDA boundary"))
            stack.enter_context(patch.object(torch.profiler, "supported_activities", return_value={torch.profiler.ProfilerActivity.CUDA}))
            profile_call = stack.enter_context(patch.object(torch.profiler, "profile", return_value=Profiler()))
            sync = stack.enter_context(patch.object(torch.cuda, "synchronize", side_effect=AssertionError("extra synchronization")))
            stack.enter_context(patch.object(experiment, "make_ppo_class", side_effect=factory))
            stack.enter_context(patch.object(experiment, "run", side_effect=train))
            stack.enter_context(patch.object(module, "source_hashes", side_effect=[before, after]))
            yield args, passed_args, classes, profile_call, sync

    def report(self, output):
        return json.loads((Path(output) / "native_ppo_profile.json").read_text())

    def test_success_uses_actual_class_options_both_activities_and_no_extra_sync(self):
        import torch
        with tempfile.TemporaryDirectory() as output:
            with self.mocked_run(output) as (args, passed, classes, profile_call, sync):
                self.assertEqual(self.module.run(args), 0)
                self.assertEqual(passed, [args])
                self.assertEqual(len(classes), 1)
                self.assertNotIn("train", classes[0].__dict__)
                sync.assert_not_called()
                self.assertEqual(profile_call.call_args.kwargs["activities"], [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            report = self.report(output)
            self.assertTrue((Path(output) / "native_ppo_profile.trace.json").is_file())
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["headline_sps_eligible"])
        self.assertEqual(report["regions"]["ppo_update"]["calls"], 1)
        self.assertTrue(report["source_hashes_unchanged"])
        self.assertTrue(report["distribution_validation_unchanged"])

    def test_missing_cuda_events_or_source_drift_fails_closed_with_artifacts(self):
        for cuda_events, drift, message in ((0, False, "No CUDA"), (1, True, "Source files changed")):
            with self.subTest(cuda_events=cuda_events, drift=drift), tempfile.TemporaryDirectory() as output:
                with self.mocked_run(output, cuda_events=cuda_events, drift=drift) as fixture:
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.module.run(fixture[0])
                report = self.report(output)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["exit_code"], 1)
                self.assertTrue((Path(output) / "native_ppo_profile.trace.json").is_file())

    def test_training_exception_preserves_identity_and_failure_artifact(self):
        failure = RuntimeError("original training exception")
        with tempfile.TemporaryDirectory() as output:
            with self.mocked_run(output, failure=failure) as fixture:
                with self.assertRaises(RuntimeError) as caught:
                    self.module.run(fixture[0])
            self.assertIs(caught.exception, failure)
            report = self.report(output)
            self.assertEqual(report["status"], "failed")
            self.assertIn("original training exception", report["error"])
            self.assertTrue((Path(output) / "native_ppo_profile.trace.json").is_file())

    def test_distribution_validation_drift_fails_and_restores_both_settings(self):
        import torch
        distribution = torch.distributions.Distribution
        initial = distribution._validate_args
        try:
            for enabled in (False, True):
                distribution.set_default_validate_args(enabled)
                with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as output:
                    with self.mocked_run(output, distribution_change=True) as fixture:
                        with self.assertRaisesRegex(RuntimeError, "Distribution validation"):
                            self.module.run(fixture[0])
                    self.assertIs(distribution._validate_args, enabled)
                    self.assertFalse(self.report(output)["distribution_validation_unchanged"])
        finally:
            distribution.set_default_validate_args(initial)

    def test_cpu_and_missing_cuda_fail_before_training_and_keep_failure_receipts(self):
        for device, cuda_available in (("cpu", True), ("cuda", False)):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as output:
                with self.mocked_run(output, cuda_available=cuda_available) as fixture:
                    fixture[0].device = device
                    with self.assertRaisesRegex((ValueError, RuntimeError), "CUDA"):
                        self.module.run(fixture[0])
                    self.assertEqual(fixture[1], [])
                report = self.report(output)
                self.assertEqual(report["status"], "failed")
                trace = json.loads((Path(output) / "native_ppo_profile.trace.json").read_text())
                self.assertFalse(trace["metadata"]["capture_available"])

    def test_profiler_enter_exception_still_writes_failure_receipt(self):
        failure = RuntimeError("profiler start failed")
        with tempfile.TemporaryDirectory() as output:
            with self.mocked_run(output, profiler_failure=failure) as fixture:
                with self.assertRaises(RuntimeError) as caught:
                    self.module.run(fixture[0])
            self.assertIs(caught.exception, failure)
            self.assertEqual(self.report(output)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
