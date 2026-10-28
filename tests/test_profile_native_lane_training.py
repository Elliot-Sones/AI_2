"""Profiler instrumentation contracts without policy training or CUDA."""
from __future__ import annotations

from ai2.paths import source_path

from contextlib import contextmanager, ExitStack
import importlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fake_classes():
    class Policy:
        def forward(self, value, *, bias=0):
            return value + bias

        def predict_values(self, value):
            return value * 2

    class Env:
        def step_tensor(self, value):
            return value + 1

    class Buffer:
        def add(self, value):
            return value

        def compute_returns_and_advantage(self, value):
            return value

    class BasePPO:
        def train(self, value):
            return value * 3

    class PPO(BasePPO):
        def collect_rollouts(self, value):
            action = Policy().forward(value, bias=4)
            terminal_value = Policy().predict_values(action)
            final_value = Policy().predict_values(action)
            step = Env().step_tensor(action)
            Buffer().add(step)
            Buffer().compute_returns_and_advantage(final_value)
            return step + terminal_value

    return {"policy_class": Policy, "env_class": Env, "buffer_class": Buffer, "ppo_class": PPO}


class ProfileRegionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("tools.profiling.profile_native_lane_training"), "profiler wrapper must exist")
        self.m = importlib.import_module('tools.profiling.profile_native_lane_training')
        self.classes = fake_classes()
        self.events = []

    @contextmanager
    def record(self, name):
        self.events.append(("enter", name))
        try:
            yield
        finally:
            self.events.append(("exit", name))

    def test_context_preserves_results_counts_nested_regions_and_restores_inheritance(self):
        policy, ppo = self.classes["policy_class"], self.classes["ppo_class"]
        original = policy.forward
        original_signature = inspect.signature(original)
        self.assertNotIn("train", ppo.__dict__)
        with self.m.instrument_regions(**self.classes, record_function=self.record) as stats:
            self.assertEqual(ppo().collect_rollouts(2), 19)
            self.assertEqual(ppo().train(7), 21)
            self.assertEqual(inspect.signature(policy.forward), original_signature)
            self.assertEqual(stats["action_forward"]["calls"], 1)
            self.assertEqual(stats["critic_only_forward"]["calls"], 2)
            self.assertEqual(stats["native_environment_step"]["calls"], 1)
            self.assertEqual(stats["rollout_buffer_add"]["calls"], 1)
            self.assertEqual(stats["gae"]["calls"], 1)
            self.assertEqual(stats["ppo_update"]["calls"], 1)
            self.assertEqual(stats["collect_rollout"]["calls"], 1)
        self.assertIs(policy.forward, original)
        self.assertNotIn("train", ppo.__dict__)
        outer = stats["collect_rollout"]
        self.assertGreater(outer["host_wall_inclusive_seconds"], outer["host_wall_exclusive_seconds"])
        self.assertGreaterEqual(outer["host_wall_exclusive_seconds"], 0)
        self.assertEqual(self.events[0], ("enter", "collect_rollout"))
        self.assertEqual(self.events[-1], ("exit", "ppo_update"))

    def test_exception_preserved_counted_and_all_class_methods_restored(self):
        env = self.classes["env_class"]
        failure = RuntimeError("same failure object")

        def fail(self, value):
            raise failure

        env.step_tensor = fail
        original_policy = self.classes["policy_class"].forward
        with self.assertRaises(RuntimeError) as caught:
            with self.m.instrument_regions(**self.classes, record_function=self.record) as stats:
                self.classes["ppo_class"]().collect_rollouts(1)
        self.assertIs(caught.exception, failure)
        self.assertIs(env.step_tensor, fail)
        self.assertIs(self.classes["policy_class"].forward, original_policy)
        self.assertEqual(stats["native_environment_step"]["failed_calls"], 1)
        self.assertEqual(stats["collect_rollout"]["failed_calls"], 1)
        self.assertEqual(stats["critic_only_forward"]["failed_calls"], 0)

    def test_partial_patch_failure_restores_already_patched_methods(self):
        policy = self.classes["policy_class"]
        original = policy.forward
        del self.classes["env_class"].step_tensor
        with self.assertRaises(AttributeError):
            with self.m.instrument_regions(**self.classes, record_function=self.record):
                self.fail("invalid instrumentation entered")
        self.assertIs(policy.forward, original)

    def test_unrelated_methods_and_arguments_unchanged(self):
        policy = self.classes["policy_class"]
        original_predict = policy.predict_values
        marker = object()
        with self.m.instrument_regions(**self.classes, record_function=self.record) as stats:
            self.assertIs(self.classes["buffer_class"]().add(marker), marker)
        self.assertIs(policy.predict_values, original_predict)
        self.assertEqual(stats["action_forward"]["calls"], 0)

    def test_aggregates_expose_cpu_cuda_metrics_without_summing_nested_totals(self):
        with self.m.instrument_regions(**self.classes, record_function=self.record) as stats:
            self.classes["policy_class"]().forward(1)
        event = SimpleNamespace(key="action_forward", count=1, cpu_time_total=18.0, self_cpu_time_total=11.0,
                                device_time_total=9.0, self_device_time_total=5.0,
                                device_type="DeviceType.CPU", is_user_annotation=True)
        result = self.m.aggregate_regions([event], stats)
        action = result["regions"]["action_forward"]
        self.assertEqual(action["calls"], 1)
        self.assertEqual(action["cpu_time_total_us"], 18)
        self.assertEqual(action["cuda_time_total_us"], 9)
        self.assertEqual(action["self_cuda_time_total_us"], 5)
        self.assertIsNone(result["regions"]["gae"]["cuda_time_total_us"])
        self.assertNotIn("total_cuda_time_us", result)
        self.assertIn("nested", " ".join(result["interpretation"]).lower())
        self.assertIn("checkpoint", " ".join(result["interpretation"]).lower())
        self.assertIn("overhead", " ".join(result["interpretation"]).lower())

    def test_legacy_cuda_fields_supported_but_unavailable_fields_are_null(self):
        with self.m.instrument_regions(**self.classes, record_function=self.record) as stats:
            pass
        legacy = SimpleNamespace(key="action_forward", count=1, cpu_time_total=10, self_cpu_time_total=10,
                                 cuda_time_total=4, self_cuda_time_total=3,
                                 device_type="DeviceType.CPU", is_user_annotation=True)
        unavailable = SimpleNamespace(key="gae", count=1, cpu_time_total=4, self_cpu_time_total=4,
                                      device_type="DeviceType.CPU", is_user_annotation=True)
        result = self.m.aggregate_regions([legacy, unavailable], stats)["regions"]
        self.assertEqual(result["action_forward"]["cuda_time_total_us"], 4)
        self.assertIsNone(result["gae"]["cuda_time_total_us"])

    def test_same_name_cpu_and_cuda_averages_keep_cpu_kernel_attribution_and_cuda_spans(self):
        # Real Torch metadata exercises its device-aware key_averages, without CUDA execution.
        from torch.autograd import DeviceType
        from torch.autograd.profiler_util import EventList, FunctionEvent

        cpu = FunctionEvent(1, "action_forward", 1, 0, 100, use_device="cuda",
                            device_type=DeviceType.CPU, stack=[])
        cpu.append_kernel("test_kernel", 0, 7)
        cuda = FunctionEvent(2, "action_forward", 1, 10, 90, use_device="cuda",
                             device_type=DeviceType.CUDA, stack=[])
        events = EventList([cpu, cuda], use_device="cuda")
        events._build_tree()
        averages = list(events.key_averages())
        # Torch 2.4 does not accept or propagate this optional metadata.
        for event in averages:
            event.is_user_annotation = True
        self.assertEqual(len(averages), 2)
        with self.m.instrument_regions(**self.classes, record_function=self.record) as stats:
            self.classes["policy_class"]().forward(1)
        for ordered in (averages, list(reversed(averages))):
            with self.subTest(order=[str(event.device_type) for event in ordered]):
                result = self.m.aggregate_regions(ordered, stats)
                action = result["regions"]["action_forward"]
                self.assertEqual(action["cpu_time_total_us"], 100)
                self.assertEqual(action["self_cpu_time_total_us"], 100)
                self.assertEqual(action["cuda_time_total_us"], 7)
                self.assertEqual(action["self_cuda_time_total_us"], 7)
                self.assertEqual(action["profiler_event_count"], 1)
                self.assertEqual(action["cpu_aggregate_status"], "matched")
                grouped = action["profiler_aggregates_by_device"]
                self.assertEqual(set(grouped), {"CPU", "CUDA"})
                self.assertEqual(len(grouped["CPU"]), 1)
                self.assertEqual(len(grouped["CUDA"]), 1)
                self.assertEqual(grouped["CUDA"][0]["span_time_total_us"], 80)
                self.assertEqual(grouped["CUDA"][0]["timing_semantics"], "annotation_span_not_kernel_time")
                self.assertNotIn("cuda_time_total_us", grouped["CUDA"][0])
                self.assertTrue(result["cpu_attribution_complete"])

    def test_legacy_missing_annotation_metadata_uses_unique_cpu_with_explicit_warning(self):
        cpu = SimpleNamespace(key="action_forward", count=1, device_type="DeviceType.CPU",
                              cpu_time_total=100, self_cpu_time_total=90, device_time_total=7, self_device_time_total=5)
        cuda = SimpleNamespace(key="action_forward", count=1, device_type="DeviceType.CUDA",
                               cpu_time_total=0, self_cpu_time_total=0, device_time_total=80, self_device_time_total=80)
        excluded = SimpleNamespace(**vars(cpu), is_user_annotation=False)
        with self.m.instrument_regions(**self.classes, record_function=self.record) as stats:
            self.classes["policy_class"]().forward(1)
        result = self.m.aggregate_regions([cpu, cuda, excluded], stats)
        action = result["regions"]["action_forward"]
        self.assertEqual(action["cpu_time_total_us"], 100)
        self.assertEqual(action["cuda_time_total_us"], 7)
        self.assertEqual(action["cpu_annotation_metadata"], "unavailable_legacy")
        self.assertIn("action_forward", " ".join(result["legacy_annotation_warnings"]))
        self.assertTrue(result["cpu_attribution_complete"])
        self.assertEqual(action["profiler_aggregates_by_device"]["CUDA"][0]["timing_semantics"],
                         "legacy_named_span_not_kernel_time")
        ambiguous = self.m.aggregate_regions([cpu, cpu, cuda], stats)
        self.assertEqual(ambiguous["regions"]["action_forward"]["cpu_aggregate_status"], "ambiguous")
        self.assertIsNone(ambiguous["regions"]["action_forward"]["cpu_time_total_us"])
        self.assertFalse(ambiguous["cpu_attribution_complete"])

    def test_missing_or_ambiguous_cpu_user_range_produces_null_not_false_zero_metrics(self):
        def event(device, *, annotation=True):
            return SimpleNamespace(key="action_forward", count=1, device_type=device,
                                   is_user_annotation=annotation, cpu_time_total=0, self_cpu_time_total=0,
                                   device_time_total=80, self_device_time_total=80)

        with self.m.instrument_regions(**self.classes, record_function=self.record) as stats:
            self.classes["policy_class"]().forward(1)
        cases = (
            ([], "missing", 0),
            ([event("DeviceType.CUDA")], "missing", 1),
            ([event("DeviceType.CPU", annotation=False), event("DeviceType.CUDA")], "missing", 2),
            ([event(None)], "missing", 1),
            ([event("DeviceType.CPU"), event("DeviceType.CPU")], "ambiguous", 2),
        )
        for events, expected_status, preserved_count in cases:
            with self.subTest(status=expected_status, events=events):
                result = self.m.aggregate_regions(events, stats)
                action = result["regions"]["action_forward"]
                self.assertEqual(action["cpu_aggregate_status"], expected_status)
                for metric in ("profiler_event_count", "cpu_time_total_us", "self_cpu_time_total_us",
                               "cuda_time_total_us", "self_cuda_time_total_us"):
                    self.assertIsNone(action[metric], metric)
                self.assertEqual(sum(map(len, action["profiler_aggregates_by_device"].values())), preserved_count)
                self.assertFalse(result["cpu_attribution_complete"])
                self.assertIn("action_forward", " ".join(result["attribution_warnings"]))

    def test_capture_fails_closed_for_missing_cuda_events_or_source_hash_drift(self):
        for cuda_events, source_unchanged, expected_error in (
            (0, True, "No CUDA device events"),
            (1, False, "Source files changed"),
            (0, False, "No CUDA device events"),
        ):
            with self.subTest(cuda_events=cuda_events, source_unchanged=source_unchanged):
                with tempfile.TemporaryDirectory() as tmp:
                    with self.mock_capture(tmp, cuda_events=cuda_events, source_unchanged=source_unchanged) as args:
                        with self.assertRaisesRegex(RuntimeError, expected_error):
                            self.m.run(args)
                    report = json.loads((Path(tmp) / "native_lane_profile.json").read_text())
                    self.assertEqual(report["status"], "failed")
                    self.assertEqual(report["exit_code"], 1)
                    self.assertEqual(report["cuda_trace_event_count"], cuda_events)
                    self.assertEqual(report["source_hashes_unchanged"], source_unchanged)
                    self.assertTrue((Path(tmp) / "native_lane_profile.trace.json").is_file())

    def test_valid_capture_completes_and_keeps_diagnostic_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.mock_capture(tmp, cuda_events=1, source_unchanged=True) as args:
                self.assertEqual(self.m.run(args), 0)
            report = json.loads((Path(tmp) / "native_lane_profile.json").read_text())
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["exit_code"], 0)
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["headline_sps_eligible"])

    @contextmanager
    def mock_capture(self, output, *, cuda_events, source_unchanged):
        # Only the external GPU profiler and training boundary are mocked. Report
        # construction, instrumentation restoration, aggregation and validation run normally.
        import torch
        import ai2.training.benchmark_native_lane_training as driver

        class Profiler:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def key_averages(self):
                return []

            def events(self):
                return [SimpleNamespace(device_type="DeviceType.CUDA")] * cuda_events

            def export_chrome_trace(self, path):
                Path(path).write_text('{"traceEvents": []}')

        before = {"profile_native_lane_training.py": "before"}
        after = before if source_unchanged else {"profile_native_lane_training.py": "after"}
        args = self.m.build_parser().parse_args(["--output", output])
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="mock CUDA boundary"))
            stack.enter_context(patch.object(torch.profiler, "supported_activities", return_value={torch.profiler.ProfilerActivity.CUDA}))
            stack.enter_context(patch.object(torch.profiler, "profile", return_value=Profiler()))
            stack.enter_context(patch.object(driver, "run", return_value=0))
            stack.enter_context(patch.object(self.m, "source_hashes", side_effect=[before, after]))
            yield args

    def test_parser_defaults_to_short_native_cuda_diagnostic(self):
        args = self.m.build_parser().parse_args(["--output", "/tmp/profile-native-test"])
        self.assertEqual((args.warmup, args.cycles), (1, 3))
        self.assertTrue(args.native_lanes)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.mode, "baseline")
        self.assertEqual(args.lane_device, "cuda")

    def test_cpu_policy_and_missing_cuda_rejected_before_driver(self):
        args = SimpleNamespace(device="cpu", native_lanes=True, mode="baseline", resident_buffer=False, rollout_policy_graph=False)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.m.run(args)
        args.device = "cuda"
        import torch

        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                self.m.run(args)

    def test_provenance_hashes_profiler_and_training_sources(self):
        hashes = self.m.source_hashes()
        for name in ("profile_native_lane_training.py", "benchmark_native_lane_training.py", "native_lane_vec_env.py",
                     "native_lane_cpu_adapter.py", "tensor_ppo.py", "tensor_rollout_buffer.py", "native/navigation_lanes.cu"):
            self.assertEqual(len(hashes[str(source_path(ROOT, name).relative_to(ROOT))]), 64)

    def test_installed_torch_profiler_exports_named_fake_class_regions(self):
        import torch

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            with self.m.instrument_regions(**self.classes, record_function=torch.profiler.record_function) as stats:
                self.classes["ppo_class"]().collect_rollouts(2)
                self.classes["ppo_class"]().train(3)
        result = self.m.aggregate_regions(profiler.key_averages(), stats)
        self.assertEqual(result["regions"]["critic_only_forward"]["profiler_event_count"], 2)
        self.assertEqual(result["regions"]["ppo_update"]["profiler_event_count"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            profiler.export_chrome_trace(str(path))
            names = {event.get("name") for event in json.loads(path.read_text())["traceEvents"]}
            self.assertTrue(set(self.m.REGIONS).issubset(names))


if __name__ == "__main__":
    unittest.main()
