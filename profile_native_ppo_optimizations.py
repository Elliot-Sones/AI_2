#!/usr/bin/env python3
"""Diagnostic CPU/CUDA trace of the independently selectable native PPO arms.

Uses the experimental driver's exact class and workload. Defaults to one warmup
and three measured cycles. Profiler overhead, startup, compilation and checkpoint
work are included; these timings are not eligible for headline throughput.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from functools import wraps
import hashlib
from pathlib import Path
import sys
import time
from typing import Any

import experimental_native_lane_training as experiment
import profile_native_lane_training as profile_helpers

ROOT = Path(__file__).resolve().parent


@contextmanager
def instrument_ppo_factory(driver: Any, *, policy_class: type, env_class: type,
                           buffer_class: type, record_function: Any):
    """Bind instrumentation when run() creates its actual PPO class.

    Each driver run creates one class. The ExitStack keeps its wrappers alive
    through training and restores both the factory and inherited methods even
    after partial instrumentation or training failures.
    """
    original = driver.make_ppo_class
    captures = []
    with ExitStack() as stack:
        @wraps(original)
        def factory(*args, **kwargs):
            if captures:
                raise RuntimeError("profiling requires one PPO class factory invocation per run")
            ppo_class = original(*args, **kwargs)
            stats = stack.enter_context(profile_helpers.instrument_regions(
                policy_class=policy_class, env_class=env_class,
                buffer_class=buffer_class, ppo_class=ppo_class,
                record_function=record_function,
            ))
            captures.append({"ppo_class": f"{ppo_class.__module__}.{ppo_class.__qualname__}",
                             "stats": stats})
            return ppo_class

        driver.make_ppo_class = factory
        stack.callback(setattr, driver, "make_ppo_class", original)
        yield captures


def source_hashes() -> dict[str, str]:
    """Include original native provenance plus every experimental source."""
    hashes = profile_helpers.source_hashes()
    for name in (*experiment.EXPERIMENT_FILES, Path(__file__).name):
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"required profiling source missing: {path}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def compiler_counters() -> dict[str, Any] | None:
    """Snapshot counters only if loaded; importing Dynamo can change validation."""
    module = sys.modules.get("torch._dynamo.utils")
    counters = getattr(module, "counters", None)
    if counters is None:
        return None
    return {str(group): dict(values) for group, values in counters.items() if values}


def build_parser():
    parser = experiment.build_parser()
    parser.description = __doc__
    parser.set_defaults(warmup=1, cycles=3)
    return parser


def run(args: Any) -> int:
    campaign = experiment.native.campaign
    output = campaign.validate_output_path(Path(args.output))
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "native_ppo_profile.json"
    trace_path = output / "native_ppo_profile.trace.json"
    report = {
        "schema": "ai2.native_ppo_ablations.profile.v1", "status": "running",
        "diagnostic_only": True, "headline_sps_eligible": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        "activities": ["CPU", "CUDA"], "trace_path": str(trace_path),
        "record_shapes": False, "with_stack": False, "profile_memory": False,
        "interpretation": list(profile_helpers.INTERPRETATION),
        "compiler_counters_before": None, "compiler_counters_after": None,
        "capture_validation_errors": [], "artifact_errors": [],
    }
    campaign.write_json(summary_path, report)
    profiler = None
    captures = []
    distribution = None
    validation_before = None
    failure = None
    code = 1
    trace_exported = False
    started = time.perf_counter()

    def remember_failure(exc: BaseException, *, artifact: bool = False) -> None:
        nonlocal failure
        if failure is None:
            failure = (exc, exc.__traceback__)
        if artifact:
            report["artifact_errors"].append(f"{type(exc).__name__}: {exc}")

    try:
        experiment.validate_experiment(args)
        if not args.native_lanes or args.lane_device != "cuda":
            raise ValueError("profiling requires native CUDA simulation lanes")
        import torch
        from torch.distributions import Distribution

        distribution = Distribution
        validation_before = Distribution._validate_args
        report["distribution_validation_before"] = validation_before
        report["compiler_counters_before"] = compiler_counters()
        report["source_hashes"] = source_hashes()
        if not torch.cuda.is_available():
            raise RuntimeError("actual CUDA is required for PPO profiling; no CPU policy fallback")
        activity = torch.profiler.ProfilerActivity
        if activity.CUDA not in torch.profiler.supported_activities():
            raise RuntimeError("Torch profiler CUDA activity support is required")
        report.update(torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                      device=str(args.device), lane_device=args.lane_device,
                      gpu_name=torch.cuda.get_device_name(torch.device(args.device)))
        from native_lane_vec_env import NativeLaneEnv
        from stable_baselines3.common.policies import ActorCriticPolicy
        from tensor_rollout_buffer import TensorRolloutBuffer

        profiler = torch.profiler.profile(
            activities=[activity.CPU, activity.CUDA], record_shapes=False,
            with_stack=False, profile_memory=False,
        )
        with profiler:
            with instrument_ppo_factory(
                experiment, policy_class=ActorCriticPolicy, env_class=NativeLaneEnv,
                buffer_class=TensorRolloutBuffer,
                record_function=torch.profiler.record_function,
            ) as captures:
                code = int(experiment.run(args))
        if len(captures) != 1:
            raise RuntimeError("experimental run did not create its instrumented PPO class")
    except BaseException as exc:
        remember_failure(exc)

    report["diagnostic_driver_wall_seconds"] = time.perf_counter() - started
    if profiler is not None:
        try:
            if captures:
                report["instrumented_ppo_class"] = captures[0]["ppo_class"]
                report.update(profile_helpers.aggregate_regions(profiler.key_averages(), captures[0]["stats"]))
        except BaseException as exc:
            remember_failure(exc, artifact=True)
        try:
            profiler.export_chrome_trace(str(trace_path))
            trace_exported = True
        except BaseException as exc:
            remember_failure(exc, artifact=True)
        try:
            report["cuda_trace_event_count"] = sum(
                profile_helpers._device_type(event) == "CUDA" for event in profiler.events()
            )
            if report["cuda_trace_event_count"] == 0:
                report["capture_validation_errors"].append("No CUDA device events were captured.")
        except BaseException as exc:
            remember_failure(exc, artifact=True)

    if "source_hashes" in report:
        try:
            report["source_hashes_after"] = source_hashes()
            report["source_hashes_unchanged"] = report["source_hashes_after"] == report["source_hashes"]
            if not report["source_hashes_unchanged"]:
                report["capture_validation_errors"].append("Source files changed during profiling.")
        except BaseException as exc:
            remember_failure(exc, artifact=True)
    if distribution is not None:
        report["distribution_validation_after"] = distribution._validate_args
        report["distribution_validation_unchanged"] = distribution._validate_args == validation_before
        if not report["distribution_validation_unchanged"]:
            report["capture_validation_errors"].append("Distribution validation changed during profiling.")
        distribution.set_default_validate_args(validation_before)
    report["compiler_counters_after"] = compiler_counters()
    if report["capture_validation_errors"]:
        report["interpretation"].extend(report["capture_validation_errors"])
        remember_failure(RuntimeError(" ".join(report["capture_validation_errors"])), artifact=True)
    if failure is not None:
        report["error"] = f"{type(failure[0]).__name__}: {failure[0]}"
        code = 1
    report["status"] = "completed" if code == 0 and failure is None else "failed"
    report["exit_code"] = code
    report["trace_exported"] = trace_exported
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    if not trace_exported:
        # An explicit failed-capture receipt is not a fabricated profiler trace.
        campaign.write_json(trace_path, {"traceEvents": [], "metadata": {
            "capture_available": False, "status": "failed",
            "error": report.get("error", "profiler trace unavailable"),
        }})
    campaign.write_json(summary_path, report)
    print(f"native_ppo_profile={summary_path}")
    print("Diagnostic only: profiler overhead is included; do not use as headline SPS.")
    if failure is not None:
        raise failure[0].with_traceback(failure[1])
    return code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = experiment.native.campaign.validate_args(parser, parser.parse_args(argv))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
