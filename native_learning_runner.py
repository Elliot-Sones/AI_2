#!/usr/bin/env python3
"""Run fresh or resumed CUDA native learning through the frozen training job.

Steps are additional transitions and round up to whole rollout/update cycles.
Every cycle is measured; there are no warmup updates or PPO optimization flags.
Only initial/latest/final checkpoints are saved, with final update counters
checked against the measured cycles before the receipt can report completion.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any
import zipfile

import benchmark_campaign as campaign
import benchmark_native_lane_training as native


def require_cuda_arg(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", required=True)
    args, _ = parser.parse_known_args(argv)
    if re.fullmatch(r"cuda(?::[0-9]+)?", args.device) is None:
        raise SystemExit("native learning requires explicit --device cuda; no CPU fallback")
    return args.device


def require_actual_cuda(job: Any, device: str) -> None:
    if not job.torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
    tensor = job.torch.empty(1, device=device)
    if not getattr(tensor, "is_cuda", False) or str(tensor.device).split(":")[0] != "cuda":
        raise RuntimeError("runtime probe did not allocate a CUDA tensor")
    job.torch.cuda.synchronize(device)


def validate_config(params: dict[str, Any], phase: str, expected_parameters: int) -> dict[str, Any]:
    if phase not in {"0a", "0b"}:
        raise ValueError("native learning supports phases 0a and 0b only")
    phase_config = params.get("phases", {}).get(phase, {})
    if phase_config.get("type") != "navigation":
        raise ValueError("phase must be navigation")
    if params.get("frame_stack", 4) != 4:
        raise ValueError("native learning requires frame_stack=4")
    if params.get("opponent_history", {}).get("enabled", False):
        raise ValueError("native learning requires opponent_history disabled")
    nav = phase_config.get("navigation", {})
    if nav.get("control_mode") != "movement" or nav.get("completion_mode") != "sequence":
        raise ValueError("native learning requires movement-only sequence navigation")
    targets = nav.get("multi_target", 1)
    if isinstance(targets, bool) or not isinstance(targets, (int, float)) or not math.isfinite(targets) or targets < 2 or int(targets) != targets:
        raise ValueError("sequence navigation requires at least two integer targets")
    seconds = float(nav.get("time_limit_seconds", 90))
    if not math.isfinite(seconds) or seconds <= 0 or math.ceil(seconds * 30) > 1024:
        raise ValueError("native horizon must be positive and at most 1024 ticks (ceil(seconds * 30))")
    arch = params.get("policy_kwargs", {}).get("net_arch", [512, 512, 256])
    if not isinstance(arch, list) or not arch or any(type(width) is not int or width <= 0 for width in arch):
        raise ValueError("net_arch must be a nonempty list of positive integers")
    # Separate actor/value MLPs, 256 stacked inputs, ten Gaussian actions,
    # one value output and ten learned log standard deviations.
    widths = [256, *arch]
    count = 2 * sum((a + 1) * b for a, b in zip(widths, widths[1:])) + (arch[-1] + 1) * 11 + 10
    if count != expected_parameters:
        raise ValueError(f"config parameter count {count} differs from expected {expected_parameters}")
    ppo = params.get("ppo_settings", {})
    for key, expected in (("learning_rate", 3e-4), ("clip_range", .2), ("nav_ent_coef", 0)):
        if ppo.get(key) != expected:
            raise ValueError(f"native learning requires constant {key}={expected}")
    return {"phase": phase, "net_arch": arch, "parameter_count": count,
            "horizon_steps": math.ceil(seconds * 30), "targets": int(targets)}


def step_expectations(steps: int, n_envs: int, n_steps: int, starting_steps: int) -> dict[str, int]:
    if any(type(value) is not int or value <= 0 for value in (steps, n_envs, n_steps)) or starting_steps < 0:
        raise ValueError("additional steps and rollout geometry must be positive; start must be nonnegative")
    transitions = n_envs * n_steps
    cycles = (steps + transitions - 1) // transitions
    additional = cycles * transitions
    return {"starting_steps": starting_steps, "requested_additional_steps": steps,
            "transitions_per_cycle": transitions, "cycles": cycles,
            "expected_additional_steps": additional, "expected_final_steps": starting_steps + additional,
            "checkpoint_interval": starting_steps + additional + 1}


def checkpoint_counters(path: Path) -> dict[str, int]:
    """Read plain JSON counters, without executing checkpoint pickle payloads."""
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    counters = {key: data.get(key) for key in ("num_timesteps", "_n_updates")}
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError(f"checkpoint has missing/invalid step or update counters: {path}")
    return counters


def validate_empty_output(path: Path) -> Path:
    output = campaign.validate_output_path(path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--output must be absent or an empty directory")
    return output


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=campaign.DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=campaign.DEFAULT_TRAIN_JOB)
    parser.add_argument("--lane-build-dir", type=Path, default=native.DEFAULT_BUILD_DIR)
    parser.add_argument("--expected-parameters", required=True, type=int)
    return parser


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    wrapper, train_argv = build_wrapper_parser().parse_known_args(argv)
    device = require_cuda_arg(train_argv)
    source, train_path, build_dir = wrapper.source.resolve(), wrapper.train_job.resolve(), wrapper.lane_build_dir.resolve()
    if not (source / "train.py").is_file() or not train_path.is_file():
        raise FileNotFoundError("--source/train.py and --train-job must exist")
    previous_cwd, previous_path = Path.cwd(), list(sys.path)
    aliases = {name: sys.modules.get(name) for name in (train_path.stem, "campaign_target_train_job", "train")}
    previous_env = {key: os.environ.get(key) for key in ("WANDB_MODE", "WANDB_NOTES", "SDL_VIDEODRIVER")}
    try:
        job = campaign.import_train_job_module(train_path)
        args = job.build_parser().parse_args(train_argv)
        output = validate_empty_output(Path(args.output))
        args.output, args.config = str(output), str(Path(args.config).resolve())
        if args.history:
            raise ValueError("native learning does not support --history")
        if args.opponents and args.opponents != "constant_agent":
            raise ValueError("native learning only supports the frozen constant_agent opponent")
        if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
            raise ValueError("--max-seconds must be positive and finite")
        with Path(args.config).open() as handle:
            params = job.yaml.safe_load(handle) or {}
        config_receipt = validate_config(params, args.phase, wrapper.expected_parameters)
        checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
        args.checkpoint = str(checkpoint) if checkpoint else None
        initial = checkpoint_counters(checkpoint) if checkpoint else {"num_timesteps": 0, "_n_updates": 0}
        expected = step_expectations(args.steps, args.n_envs, args.n_steps, initial["num_timesteps"])
        plan = campaign.plan_run(n_envs=args.n_envs, n_steps=args.n_steps, batch_size=args.batch_size,
                                 epochs=args.epochs, warmup=0, cycles=expected["cycles"])
        if args.batch_size > plan.transitions_per_cycle:
            raise ValueError("--batch-size must not exceed the rollout size")
        require_actual_cuda(job, device)
        output.mkdir(parents=True, exist_ok=True)
        files = {"config": campaign.file_record(Path(args.config)),
                 "checkpoint": campaign.file_record(checkpoint) if checkpoint else None,
                 "source_train": campaign.file_record(source / "train.py"),
                 "train_job": campaign.file_record(train_path),
                 "runner": campaign.file_record(Path(__file__))}
        receipt = {**expected, "config": config_receipt, "files": files, "source": str(source),
                   "lane_build_dir": str(build_dir), "device": device, "tracking_mode": "offline",
                   "periodic_checkpoints": False, "checkpoint_semantics": "initial/latest/final only; final must follow completed updates",
                   "warmup_cycles": 0, "optimization_flags": {}, "completed_cycles": 0,
                   "final_completed_update_verified": False, "status": "running",
                   "candidate_files": {name: campaign.file_record(native.ROOT / name) for name in native.CANDIDATE_FILES}}
        campaign.write_json(output / "native_learning_receipt.json", receipt)
        campaign.write_json(output / "campaign_plan.json", dataclasses.asdict(plan))
        os.environ["WANDB_MODE"] = "offline"
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        sys.path.insert(0, str(source))
        os.chdir(source)
        state = campaign.CampaignState(output=output, plan=plan, sync=lambda: job.torch.cuda.synchronize(device))
        runtime_agent = None
        status, error = "completed", None
        try:
            from tensor_ppo import TensorPPO

            measured_ppo = campaign.make_measured_ppo_class(TensorPPO, state)
            with ExitStack() as stack:
                stack.enter_context(native.native_lane_job_context(job, build_dir=build_dir, ppo_class=measured_ppo))
                original_callback, original_interval = job.JobCallback, job.CHECKPOINT_INTERVAL
                original_count, original_write = job._parameter_count, job._write_json

                def restore():
                    job.JobCallback, job.CHECKPOINT_INTERVAL = original_callback, original_interval
                    job._parameter_count, job._write_json = original_count, original_write

                stack.callback(restore)

                class FinalOnlyCallback(original_callback):
                    def __init__(self, output_dir, max_seconds, checkpoint_interval=None, verbose=0):
                        super().__init__(output_dir, max_seconds, checkpoint_interval=expected["checkpoint_interval"], verbose=verbose)

                def parameter_probe(agent):
                    nonlocal runtime_agent
                    runtime_agent = agent
                    count = original_count(agent)
                    parameters = list(agent.policy.parameters())
                    devices = sorted({str(param.device) for param in parameters})
                    all_cuda = bool(parameters) and all(value.split(":")[0] == "cuda" for value in devices)
                    actual_arch = getattr(agent.policy, "net_arch", None)
                    proof = {"parameter_count": count, "policy_devices": devices,
                             "policy_device": devices[0] if len(devices) == 1 else ",".join(devices),
                             "all_policy_parameters_cuda": all_cuda,
                             "net_arch": actual_arch, "starting_steps": int(agent.num_timesteps), "starting_updates": int(agent._n_updates)}
                    campaign.write_json(output / "runtime_device.json", proof)
                    receipt["runtime_policy"] = proof
                    if not all_cuda:
                        raise RuntimeError(f"policy parameters are not on CUDA: {devices}")
                    if count != wrapper.expected_parameters or sum(param.numel() for param in parameters) != count or actual_arch != config_receipt["net_arch"]:
                        raise RuntimeError("actual policy parameter count/architecture differs from config and expected parameters")
                    if agent.num_timesteps != initial["num_timesteps"] or agent._n_updates != initial["_n_updates"]:
                        raise RuntimeError("loaded policy counters differ from the input checkpoint/fresh expectation")
                    return count

                def metadata_probe(path, payload):
                    if Path(path).name == "metadata.json":
                        payload = {**payload, "config_sha256": files["config"]["sha256"],
                                   "input_checkpoint_sha256": files["checkpoint"]["sha256"] if checkpoint else None,
                                   "native_learning": {**expected, "warmup_cycles": 0, "periodic_checkpoints": False}}
                    return original_write(path, payload)

                job.JobCallback, job.CHECKPOINT_INTERVAL = FinalOnlyCallback, expected["checkpoint_interval"]
                job._parameter_count, job._write_json = parameter_probe, metadata_probe
                job.run_job(args)
            result = json.loads((output / "result.json").read_text())
            expected_updates = initial["_n_updates"] + expected["cycles"] * args.epochs
            archived_initial = checkpoint_counters(output / "initial.zip")
            archived_final = checkpoint_counters(output / "final.zip")
            archived_latest = checkpoint_counters(output / "latest.zip")
            final_counters = {"num_timesteps": expected["expected_final_steps"], "_n_updates": expected_updates}
            verified = (result.get("status") == "completed" and result.get("starting_steps") == initial["num_timesteps"]
                        and result.get("ending_steps") == expected["expected_final_steps"]
                        and result.get("actual_steps") == expected["expected_additional_steps"]
                        and state.cycle_index == expected["cycles"] and archived_initial == initial
                        and archived_final == final_counters and archived_latest == final_counters
                        and runtime_agent is not None and runtime_agent._n_updates == expected_updates
                        and all(row["num_timesteps"] == initial["num_timesteps"] + index * plan.transitions_per_cycle
                                for index, row in enumerate(state.records, 1)))
            receipt.update(final_completed_update_verified=verified, expected_final_updates=expected_updates,
                           initial_checkpoint_counters=archived_initial, final_checkpoint_counters=archived_final,
                           result=result, final_checkpoint=campaign.file_record(output / "final.zip"))
            if not verified:
                status, error = "incomplete", "final checkpoint/result did not match all expected completed updates"
        except Exception as exc:
            status, error = "exception", f"{type(exc).__name__}: {exc}"
        receipt.update(status=status, error=error, completed_cycles=state.cycle_index)
        campaign.write_json(output / "campaign_summary.json", state.summary(status=status, error=error))
        campaign.write_json(output / "native_learning_receipt.json", receipt)
        return {"completed": 0, "incomplete": 2, "exception": 1}[status]
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        for name, module in aliases.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
