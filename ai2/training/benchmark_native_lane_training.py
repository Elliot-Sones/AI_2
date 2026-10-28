#!/usr/bin/env python3
"""Opt-in native-lane training driver (milestone 4).

Runs the unchanged campaign/train_job entrypoints, checkpointing, logging,
rollout geometry and PPO update schedule with two substitutions: the rollout
environment is ``NativeLaneEnv`` on CUDA by default (the whole phase 0b tick
per lane in native code, generator-mode RNG, native autoreset) and the
collector is ``TensorPPO`` (rollouts on CUDA tensors). ``--lane-device cpu``
selects the same native simulation on CPU, with action/observation transfers
through ``NativeLaneCPUAdapter``; policy inference and PPO stay on CUDA. The
PPO update is still the installed ``stable_baselines3.PPO.train``.

The lane environment is configured from one real frozen navigation env built
in-process (wrapper settings, spaces and the Chipmunk collision bias are read
from it and it is closed). This driver requires a CUDA policy device and is
not itself evidence of a speed gain: matched measurements against the
Python/Pymunk CPU runner are separate campaign runs on one host.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import REPO_ROOT as PROJECT_ROOT
from ai2.paths import source_path

from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
from typing import Any

import ai2.runtime.benchmark_campaign as campaign

ROOT = PROJECT_ROOT
CANDIDATE_FILES = ("benchmark_native_lane_training.py", "native_lane_vec_env.py", "native_lane_cpu_adapter.py", "tensor_ppo.py", "tensor_rollout_buffer.py",
                   "native/navigation_lanes.h", "native/navigation_lanes.c", "native/navigation_lanes.cu", "native/navigation_spaces.json")
DEFAULT_BUILD_DIR = ROOT / "build" / "native-lanes"


def validate_params(params: dict[str, Any], phase_config: Any) -> None:
    if not getattr(phase_config, "is_navigation", False):
        raise ValueError("native lanes implement the navigation phase only")
    if int(params.get("frame_stack", 4)) != 4:
        raise ValueError("native lanes implement the 4-frame stack only")
    if bool(params.get("opponent_history", {}).get("enabled", False)):
        raise ValueError("native lanes do not implement the opponent history block; disable opponent_history")


def probe_frozen_env(train_module: Any, phase_config: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Build one real navigation env, read what the lanes must match, close it."""
    import ai2.evaluation.navigation_episode_capture as capture_mod
    import tools.validation.validate_native_navigation as validator
    from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

    resolution = train_module.get_resolution(params.get("environment_settings", {}).get("resolution", "LOW"))
    env = train_module.make_nav_env(phase_config, resolution, params)
    try:
        wrapper = validator.find_frozen_opponent_wrapper(env)
        wrapper_cfg = capture_mod.wrapper_config(wrapper)
        raw_env = capture_mod.find_raw_env(wrapper)
        collision_bias = float(capture_mod.SpaceRecorder(raw_env).space_config()["collision_bias"])
        frame_stack = int(params.get("frame_stack", 4))
        base_space = env.observation_space
        observation_space = StackedObservations(1, frame_stack, base_space).stacked_observation_space if frame_stack > 1 else base_space
        return {"wrapper_cfg": wrapper_cfg, "collision_bias": collision_bias, "observation_space": observation_space,
                "action_space": env.action_space, "frame_stack": frame_stack}
    finally:
        env.close()


def make_lane_build_env(build_dir: Path, *, lane_device: str = "cuda", record_capacity: int | None = None, probe: Any = probe_frozen_env):
    if lane_device not in {"cpu", "cuda"}:
        raise ValueError("lane_device must be cpu or cuda")

    def build_env(train_module: Any, phase_config: Any, params: dict[str, Any], n_envs: int, seed: int, device: Any):
        from ai2.simulation.native_lane_vec_env import build_cpu_env, build_cuda_env
        from ai2.training.tensor_ppo import TensorVecEnv

        if str(device).split(":", 1)[0] != "cuda":
            raise ValueError("native lane training requires a CUDA device")
        validate_params(params, phase_config)
        info = probe(train_module, phase_config, params)
        kwargs = dict(build_dir=Path(build_dir), seed=int(seed), wrapper_cfg=info["wrapper_cfg"],
                      observation_space=info["observation_space"], action_space=info["action_space"],
                      collision_bias=info["collision_bias"], record_capacity=record_capacity)
        if lane_device == "cpu":
            from ai2.simulation.native_lane_cpu_adapter import NativeLaneCPUAdapter

            env = NativeLaneCPUAdapter(build_cpu_env(int(n_envs), contract="off", **kwargs), device=device)
        else:
            env = build_cuda_env(int(n_envs), device=device, **kwargs)
        return TensorVecEnv(env, device=device)

    return build_env


def make_native_lane_callback(base_class: type) -> type:
    """Persist rollout-end native records without replaying per-step callbacks."""
    class NativeLaneJobCallback(base_class):
        def _on_training_start(self):
            super()._on_training_start()
            self._native_start_step = int(self.model.num_timesteps)
            self._native_num_envs = int(self.model.n_envs)
            self._native_lane_progress: dict[int, tuple[int, int | None]] = {}

        def _on_rollout_end(self):
            super()._on_rollout_end()
            infos = [info for info in self.locals.get("drained_infos", [])
                     if isinstance(info, dict) and info.get("episode")]
            # CUDA record reservations can interleave lanes. Each lane's index
            # establishes episode order before cumulative lengths become ticks.
            infos.sort(key=lambda info: (info.get("navigation", {}).get("lane", -1),
                                         info.get("navigation", {}).get("episode_index", -1)))
            rows = []
            for info in infos:
                episode, navigation = info["episode"], info.get("navigation", {})
                lane, index = navigation.get("lane"), navigation.get("episode_index")
                step, basis = int(self.num_timesteps), "rollout_end"
                if lane is not None and index is not None:
                    last_index, ticks = self._native_lane_progress.get(lane, (-1, 0))
                    if index <= last_index:
                        continue
                    length = int(episode["l"])
                    if index == last_index + 1 and ticks is not None and length > 0:
                        ticks += length
                        event_step = self._native_start_step + ticks * self._native_num_envs
                        if event_step <= step:
                            step, basis = event_step, "lane_episode_lengths"
                        else:
                            ticks = None
                    else:
                        # Overflow/missing records invalidate subsequent exact
                        # event ticks for this lane; never invent lost lengths.
                        ticks = None
                    self._native_lane_progress[lane] = (index, ticks)
                row = {**info, "step": step, "step_basis": basis, "recorded_at": "rollout_end",
                       "rollout_end_step": int(self.num_timesteps), "reward": episode.get("r"),
                       "length": episode.get("l"), "lane": lane, "episode_index": index}
                if "t" in episode:
                    # NativeLaneEnv reports time since reset when records drain,
                    # not an individual episode's wall-clock duration.
                    row["episode_time_basis"] = "rollout_drain_elapsed"
                rows.append(row)
            rows.sort(key=lambda row: (row["step"], -1 if row["lane"] is None else row["lane"]))
            if rows:
                with self.episodes_path.open("a", encoding="utf-8") as handle:
                    handle.write("".join(json.dumps(row) + "\n" for row in rows))

    return NativeLaneJobCallback


@contextmanager
def native_lane_job_context(job: Any, *, build_dir: Path, lane_device: str = "cuda", ppo_class: Any = None,
                            record_capacity: int | None = None, probe: Any = probe_frozen_env):
    if ppo_class is None:
        from ai2.training.tensor_ppo import TensorPPO

        ppo_class = TensorPPO
    original_build, original_ppo = job.build_env, job.PPO
    original_callback = getattr(job, "JobCallback", None)
    job.build_env = make_lane_build_env(Path(build_dir), lane_device=lane_device, record_capacity=record_capacity, probe=probe)
    job.PPO = ppo_class
    if original_callback is not None:
        job.JobCallback = make_native_lane_callback(original_callback)
    try:
        yield job
    finally:
        job.build_env, job.PPO = original_build, original_ppo
        if original_callback is not None:
            job.JobCallback = original_callback


def provenance(build_dir: Path, *, lane_device: str = "cuda", policy_device: Any = "cuda") -> dict[str, Any]:
    import ai2.simulation.native_lane_vec_env as lanes

    return {
        "scope": f"native lanes on {lane_device.upper()} with CUDA policy and the TensorPPO collector; PPO update is the installed stable_baselines3 train",
        "lane_device": lane_device,
        "policy_device": str(policy_device),
        "collector": "TensorPPO",
        "step_transfers": "blocking D2H actions; blocking H2D observations, rewards, flags and terminal observations" if lane_device == "cpu"
                          else "none; lane outputs remain on CUDA",
        "candidate": {name: campaign.file_record(source_path(ROOT, name)) for name in CANDIDATE_FILES},
        "native_sources": lanes.source_hashes(),
        "spaces": json.loads((ROOT / "native" / "navigation_spaces.json").read_text()).get("source"),
        "build_dir": str(Path(build_dir).resolve()),
        "training_algorithm": "installed stable_baselines3.PPO.train",
    }


def run(args: Any) -> int:
    if not args.native_lanes:
        return campaign.run_campaign(args)
    if str(args.device).split(":", 1)[0] != "cuda":
        raise ValueError("native lane training requires CUDA")
    lane_device = getattr(args, "lane_device", "cuda")
    if lane_device not in {"cpu", "cuda"}:
        raise ValueError("lane_device must be cpu or cuda")
    if getattr(args, "rollout_policy_graph", False):
        raise ValueError("qualify the native lanes independently before combining policy graph changes")
    for name in CANDIDATE_FILES:
        if not (source_path(ROOT, name)).is_file():
            raise FileNotFoundError(f"required candidate source missing: {source_path(ROOT, name)}")
    output = campaign.validate_output_path(Path(args.output))
    output.mkdir(parents=True, exist_ok=True)
    build_dir = Path(args.lane_build_dir)
    campaign.write_json(output / "native_lane_provenance.json", provenance(build_dir, lane_device=lane_device, policy_device=args.device))
    original_import = campaign.import_train_job_module
    with ExitStack() as stack:
        def import_job(path):
            return stack.enter_context(native_lane_job_context(original_import(path), build_dir=build_dir, lane_device=lane_device,
                                                                record_capacity=args.lane_record_capacity))

        campaign.import_train_job_module = import_job
        try:
            return campaign.run_campaign(args)
        finally:
            campaign.import_train_job_module = original_import


def main(argv: Any = None) -> int:
    parser = campaign.build_parser()
    parser.description = __doc__
    parser.add_argument("--native-lanes", action="store_true")
    parser.add_argument("--lane-device", choices=("cpu", "cuda"), default="cuda", help="native simulation device; policy/PPO still require --device cuda")
    parser.add_argument("--lane-build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--lane-record-capacity", type=int, default=None)
    args = campaign.validate_args(parser, parser.parse_args(argv))
    if args.native_lanes and (args.resident_buffer or args.mode != "baseline"):
        parser.error("native lanes own their rollout path; use baseline timing and no --resident-buffer")
    if args.native_lanes and str(args.device).split(":", 1)[0] != "cuda":
        parser.error("--native-lanes requires --device cuda")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
