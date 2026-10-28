#!/usr/bin/env python3
"""Thin grouped-throughput runner for E6 learning jobs.

The frozen ``train_job.py`` remains the algorithm owner. This wrapper only
loads it under an importable real module name, optionally patches its
``SubprocVecEnv`` factory to ``GroupedSubprocVecEnv``, forwards the original
train arguments, and restores the patch afterward.
"""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import source_path

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from ai2.runtime.benchmark_campaign import import_train_job_module, patch_grouped_vec_env


DEFAULT_NAV_ROOT = (
    Path("/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905")
    / "baselines"
    / "navigation-repair-20260905"
)
DEFAULT_TRAIN_JOB = DEFAULT_NAV_ROOT / "payload" / "scripts" / "train_job.py"
DEFAULT_GROUPED_ENV = source_path(PROJECT_ROOT, "grouped_vec_env.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_cuda_arg(argv: list[str]) -> None:
    if "--device" not in argv:
        raise SystemExit("learning_runner requires explicit --device cuda")
    device = argv[argv.index("--device") + 1]
    if not device.startswith("cuda"):
        raise SystemExit("learning_runner requires CUDA; refusing CPU fallback")


def require_actual_cuda(train_job: Any) -> None:
    torch = getattr(train_job, "torch", None)
    if torch is None:
        import torch as runtime_torch

        torch = runtime_torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing silent CPU fallback")
    torch.empty(1, device="cuda")


def install_runtime_device_probe(train_job: Any, output: Path):
    original = train_job._parameter_count
    output = Path(output)

    def wrapped(agent):
        count = original(agent)
        policy = getattr(agent, "policy", None)
        parameters = list(policy.parameters()) if policy is not None and hasattr(policy, "parameters") else []
        devices = sorted({str(getattr(param, "device", "unknown")) for param in parameters})
        all_cuda = bool(parameters) and all(device.startswith("cuda") for device in devices)
        payload = {
            "parameter_count": count,
            "policy_devices": devices,
            "policy_device": devices[0] if len(devices) == 1 else ",".join(devices),
            "all_policy_parameters_cuda": all_cuda,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "runtime_device.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not all_cuda:
            raise RuntimeError(f"policy parameters not on CUDA: {devices}")
        return count

    train_job._parameter_count = wrapped

    def cleanup() -> None:
        train_job._parameter_count = original

    return cleanup


def install_metadata_probe(train_job: Any, config: Path, checkpoint: Path | None):
    original = train_job._write_json
    config_sha = sha256_file(Path(config))
    input_checkpoint_sha = sha256_file(Path(checkpoint).resolve()) if checkpoint is not None else None

    def wrapped(path, payload):
        if Path(path).name == "metadata.json" and isinstance(payload, dict):
            payload = dict(payload)
            payload.setdefault("config_sha256", config_sha)
            payload.setdefault("input_checkpoint_sha256", input_checkpoint_sha)
        return original(path, payload)

    train_job._write_json = wrapped

    def cleanup() -> None:
        train_job._write_json = original

    return cleanup


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--train-job", type=Path, default=DEFAULT_TRAIN_JOB)
    parser.add_argument("--grouped-env-path", type=Path, default=DEFAULT_GROUPED_ENV)
    parser.add_argument("--group-size", type=int, default=4)
    return parser


def namespace_for_group_patch(args: argparse.Namespace) -> argparse.Namespace:
    if int(args.group_size) <= 0:
        raise SystemExit("--group-size must be positive")
    return argparse.Namespace(group_size=int(args.group_size), grouped_env_path=Path(args.grouped_env_path).resolve())


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    wrapper_args, train_argv = build_wrapper_parser().parse_known_args(argv)
    require_cuda_arg(train_argv)
    train_job = import_train_job_module(Path(wrapper_args.train_job))
    require_actual_cuda(train_job)
    train_args = train_job.build_parser().parse_args(train_argv)
    if not str(getattr(train_args, "device", "")).startswith("cuda"):
        raise SystemExit("learning_runner requires CUDA; refusing CPU fallback")
    cleanup = patch_grouped_vec_env(train_job, namespace_for_group_patch(wrapper_args))
    device_cleanup = install_runtime_device_probe(train_job, Path(train_args.output))
    checkpoint_path = Path(train_args.checkpoint).resolve() if getattr(train_args, "checkpoint", None) else None
    metadata_cleanup = install_metadata_probe(train_job, Path(train_args.config), checkpoint_path)
    try:
        train_job.run_job(train_args)
    finally:
        metadata_cleanup()
        device_cleanup()
        if cleanup is not None:
            cleanup()
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
