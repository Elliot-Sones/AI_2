#!/usr/bin/env python3
"""Opt-in real-game, CUDA-resident collector qualification harness.

Uses the unchanged campaign/train_job entrypoints, game, checkpoint, logging,
rollout geometry and PPO update schedule. The game is still CPU simulated.
Only the collector/storage and explicit CPU/CUDA boundary are changed. This
driver is not evidence of either a GPU-native game or a measured speed gain.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import inspect
from pathlib import Path

import benchmark_campaign as campaign

ROOT = Path(__file__).resolve().parent
CANDIDATE_FILES = ("benchmark_tensor_training.py", "tensor_ppo.py", "tensor_rollout_buffer.py", "tensor_cpu_vec_env.py")


def _adapt(env, *, device):
    from tensor_cpu_vec_env import CPUGameTensorAdapter
    from tensor_ppo import TensorVecEnv
    return TensorVecEnv(CPUGameTensorAdapter(env, device=device), device=device)


@contextmanager
def tensor_job_context(job, *, adapter=None, ppo_class=None):
    if adapter is None:
        adapter = _adapt
    if ppo_class is None:
        from tensor_ppo import TensorPPO
        ppo_class = TensorPPO
    original_build, original_ppo = job.build_env, job.PPO

    def build_env(*args, **kwargs):
        bound = inspect.signature(original_build).bind(*args, **kwargs)
        device = bound.arguments.get("device", args[5] if len(args) > 5 else kwargs.get("device"))
        env = original_build(*args, **kwargs)
        try:
            return adapter(env, device=device)
        except BaseException:
            env.close()
            raise

    job.build_env, job.PPO = build_env, ppo_class
    try:
        yield job
    finally:
        job.build_env, job.PPO = original_build, original_ppo


def run(args):
    if not args.tensor_collector:
        return campaign.run_campaign(args)
    if str(args.device).split(":", 1)[0] != "cuda":
        raise ValueError("tensor collector qualification requires CUDA")
    if getattr(args, "rollout_policy_graph", False):
        raise ValueError("qualify the tensor collector independently before combining policy graph changes")
    for name in CANDIDATE_FILES:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"required candidate source missing: {ROOT / name}")
    output = campaign.validate_output_path(Path(args.output))
    output.mkdir(parents=True, exist_ok=True)
    campaign.write_json(output / "tensor_candidate_provenance.json", {
        "scope": "CPU game with CUDA collector/storage; not GPU simulation",
        "candidate": {name: campaign.file_record(ROOT / name) for name in CANDIDATE_FILES},
        "training_algorithm": "installed stable_baselines3.PPO.train",
    })
    original_import = campaign.import_train_job_module
    with ExitStack() as stack:
        def import_job(path):
            return stack.enter_context(tensor_job_context(original_import(path)))
        campaign.import_train_job_module = import_job
        try:
            return campaign.run_campaign(args)
        finally:
            campaign.import_train_job_module = original_import


def main(argv=None):
    parser = campaign.build_parser()
    parser.description = __doc__
    parser.add_argument("--tensor-collector", action="store_true")
    args = campaign.validate_args(parser, parser.parse_args(argv))
    if args.tensor_collector and (args.resident_buffer or args.mode != "baseline"):
        parser.error("tensor collector owns its buffer; use baseline timing and no --resident-buffer")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
