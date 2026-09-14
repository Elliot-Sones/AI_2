#!/usr/bin/env python3
"""E6 fresh-architecture PPO learning campaign planner/supervisor.

This script builds a finite, resumable phase-0a-only pilot matrix for the
frozen navigation-repair payload. By default it writes plan/config artifacts
without launching jobs. Use ``--execute`` on the GPU host to run the generated
commands sequentially.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PAYLOAD = (
    Path("/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905")
    / "baselines"
    / "navigation-repair-20260905"
    / "payload"
)
DEFAULT_SOURCE = DEFAULT_PAYLOAD / "variants" / "corrected"
DEFAULT_TRAIN_JOB = DEFAULT_PAYLOAD / "scripts" / "train_job.py"
DEFAULT_EVALUATE_MODES = DEFAULT_PAYLOAD / "scripts" / "evaluate_modes.py"
DEFAULT_LEARNING_RUNNER = Path(__file__).resolve().parent / "learning_runner.py"
DEFAULT_GROUPED_ENV = Path(__file__).resolve().parent / "grouped_vec_env.py"
DEFAULT_ARCHITECTURES = ((512, 512, 256), (1024, 1024, 1024), (1536, 1536, 1536))
DEFAULT_SEEDS = (145, 146, 147)
DEFAULT_CHUNK_STEPS = 500_000
DEFAULT_CHUNKS = 10
DEFAULT_DEV_EPISODES = 20
DEFAULT_FULL_EPISODES = 100
DEFAULT_MASTERY_THRESHOLD = 0.90
DEFAULT_RESERVE_SECONDS = 300.0
DEFAULT_TRAIN_TIMEOUT_SECONDS = 1_200.0
DEFAULT_DEV_EVAL_TIMEOUT_SECONDS = 420.0
DEFAULT_FULL_EVAL_TIMEOUT_SECONDS = 1_800.0
MIN_FREE_DISK_BYTES_BEFORE_LAUNCH = 3 * 1024**3


@dataclasses.dataclass(frozen=True)
class RunSpec:
    net_arch: tuple[int, ...]
    seed: int

    @property
    def label(self) -> str:
        return f"arch-{'-'.join(str(value) for value in self.net_arch)}_seed{self.seed}"


@dataclasses.dataclass(frozen=True)
class TrainControls:
    n_envs: int = 64
    n_steps: int = 512
    batch_size: int = 8192
    epochs: int = 8
    max_seconds: float = DEFAULT_TRAIN_TIMEOUT_SECONDS
    group_size: int = 1


@dataclasses.dataclass(frozen=True)
class CommandSpec:
    label: str
    kind: str
    cwd: Path
    argv: list[str]
    output: Path
    timeout_seconds: float
    metadata: dict[str, Any]


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_bytes_if_new_or_same(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"existing file differs; refusing overwrite: {path}")
        return
    path.write_bytes(data)


def copy_file_if_new_or_same(source: Path, target: Path) -> None:
    write_bytes_if_new_or_same(target, Path(source).read_bytes())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "size": resolved.stat().st_size if resolved.is_file() else None,
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
    }


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def default_run_specs() -> list[RunSpec]:
    return [RunSpec(tuple(arch), int(seed)) for arch in DEFAULT_ARCHITECTURES for seed in DEFAULT_SEEDS]


def require_cuda_device(device: str) -> None:
    if not str(device).startswith("cuda"):
        raise ValueError("E6 learning campaign requires CUDA; refusing CPU fallback")


def assert_deadline_allows(*, now: float, deadline: float | None, timeout: float, reserve: float) -> None:
    if deadline is None:
        return
    if float(now) + float(timeout) + float(reserve) > float(deadline):
        raise RuntimeError("insufficient deadline for job timeout plus reserve")


def is_completed_output(output: Path) -> bool:
    result = Path(output) / "result.json"
    if not result.is_file():
        return False
    try:
        return json.loads(result.read_text(encoding="utf-8")).get("status") == "completed"
    except json.JSONDecodeError:
        return False


def assert_not_completed_output(output: Path) -> None:
    if is_completed_output(output):
        raise RuntimeError(f"refusing to overwrite completed output: {output}")


def nearest_existing_ancestor(path: Path) -> Path:
    current = Path(path).resolve()
    if not current.exists():
        current = current.parent
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise RuntimeError(f"could not find existing filesystem ancestor for {path}")
        current = parent
    return current


def output_root_for_command(command: CommandSpec) -> Path:
    explicit = command.metadata.get("output_root") if isinstance(command.metadata, dict) else None
    return Path(explicit).resolve() if explicit else Path(command.output).resolve().parent


def ensure_launch_safety(command: CommandSpec) -> None:
    output_root = output_root_for_command(command)
    stop_request = output_root / "stop.request"
    if stop_request.exists():
        raise RuntimeError(f"stop.request sentinel present; safe pause before launching {command.label}: {stop_request}")
    usage_path = nearest_existing_ancestor(command.output)
    free_bytes = int(shutil.disk_usage(usage_path).free)
    if free_bytes < MIN_FREE_DISK_BYTES_BEFORE_LAUNCH:
        gib = free_bytes / 1024**3
        need_gib = MIN_FREE_DISK_BYTES_BEFORE_LAUNCH / 1024**3
        raise RuntimeError(
            f"insufficient free disk before launching {command.label}: "
            f"need >= {need_gib:.1f} GiB at {usage_path}, found {gib:.2f} GiB; output={command.output}"
        )


def rounded_rollout_steps(requested_steps: int, *, n_envs: int, n_steps: int) -> int:
    rollout_size = int(n_envs) * int(n_steps)
    if int(requested_steps) <= 0 or rollout_size <= 0:
        raise ValueError("requested_steps, n_envs, and n_steps must be positive")
    return math.ceil(int(requested_steps) / rollout_size) * rollout_size


def replace_net_arch_block(text: str, arch: tuple[int, ...]) -> str:
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        out.append(line)
        if line.strip() == "net_arch:":
            key_indent = len(line) - len(line.lstrip())
            child_indent = line[:key_indent] + "  "
            index += 1
            while index < len(lines):
                stripped = lines[index].strip()
                candidate_indent = len(lines[index]) - len(lines[index].lstrip())
                if stripped == "" or stripped.startswith("- ") or candidate_indent > key_indent:
                    index += 1
                    continue
                break
            out.extend(f"{child_indent}- {int(width)}" for width in arch)
            continue
        index += 1
    return "\n".join(out) + "\n"


def replace_scalar_or_list_value(text: str, section: str, key: str, value: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_section = False
    section_indent = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped == f"{section}:":
            in_section = True
            section_indent = line[: len(line) - len(line.lstrip())]
            out.append(line)
            index += 1
            continue
        if in_section and stripped and not line.startswith(section_indent + " "):
            in_section = False
        if in_section and stripped.startswith(f"{key}:"):
            key_indent = len(line) - len(line.lstrip())
            indent = line[:key_indent]
            out.append(f"{indent}{key}: {value}")
            index += 1
            while index < len(lines):
                child = lines[index]
                child_stripped = child.strip()
                child_indent = len(child) - len(child.lstrip())
                if child_stripped == "" or child_stripped.startswith("- ") or child_indent > key_indent:
                    index += 1
                    continue
                break
            continue
        out.append(line)
        index += 1
    return "\n".join(out) + "\n"


def mutate_config_dict(config: dict[str, Any], arch: tuple[int, ...]) -> dict[str, Any]:
    mutated = copy.deepcopy(config)
    mutated.setdefault("policy_kwargs", {})["net_arch"] = [int(width) for width in arch]
    ppo = mutated.setdefault("ppo_settings", {})
    ppo["learning_rate"] = 0.0003
    ppo["clip_range"] = 0.2
    ppo["nav_ent_coef"] = 0.0
    return mutated


def write_arch_config(base_config: Path, output: Path, arch: tuple[int, ...]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ModuleNotFoundError:
        text = Path(base_config).read_text(encoding="utf-8")
        text = replace_net_arch_block(text, tuple(int(value) for value in arch))
        text = replace_scalar_or_list_value(text, "ppo_settings", "learning_rate", "0.0003")
        text = replace_scalar_or_list_value(text, "ppo_settings", "clip_range", "0.2")
        text = replace_scalar_or_list_value(text, "ppo_settings", "nav_ent_coef", "0.0")
        write_bytes_if_new_or_same(output, text.encode("utf-8"))
        return
    config = yaml.safe_load(Path(base_config).read_text(encoding="utf-8")) or {}
    text = yaml.safe_dump(mutate_config_dict(config, tuple(int(value) for value in arch)), sort_keys=False)
    write_bytes_if_new_or_same(output, text.encode("utf-8"))



def prepare_variant_source(frozen_source: Path, variant_source: Path, config: Path) -> None:
    frozen_source = Path(frozen_source).resolve()
    variant_source = Path(variant_source).resolve()
    if variant_source.exists():
        copy_file_if_new_or_same(config, variant_source / "config.yaml")
        return
    variant_source.mkdir(parents=True)
    for entry in frozen_source.iterdir():
        target = variant_source / entry.name
        if entry.name == "config.yaml":
            copy_file_if_new_or_same(config, target)
        else:
            target.symlink_to(entry, target_is_directory=entry.is_dir())
    if not (variant_source / "config.yaml").exists():
        copy_file_if_new_or_same(config, variant_source / "config.yaml")


def build_train_command(
    *,
    python: str,
    runner: Path,
    train_job: Path,
    grouped_env_path: Path,
    source: Path,
    config: Path,
    spec: RunSpec,
    chunk_index: int,
    previous_checkpoint: Path | None,
    output: Path,
    steps: int,
    max_seconds: float,
    controls: TrainControls,
) -> CommandSpec:
    argv = [
        python,
        str(Path(runner).resolve()),
        "--train-job",
        str(Path(train_job).resolve()),
        "--grouped-env-path",
        str(Path(grouped_env_path).resolve()),
        "--group-size",
        str(int(controls.group_size)),
        "--config",
        str(Path(config).resolve()),
        "--phase",
        "0a",
        "--seed",
        str(int(spec.seed)),
        "--steps",
        str(int(steps)),
        "--n-envs",
        str(int(controls.n_envs)),
        "--n-steps",
        str(int(controls.n_steps)),
        "--batch-size",
        str(int(controls.batch_size)),
        "--epochs",
        str(int(controls.epochs)),
        "--device",
        "cuda",
        "--output",
        str(Path(output).resolve()),
        "--max-seconds",
        str(float(max_seconds)),
    ]
    if previous_checkpoint is not None:
        argv.extend(["--checkpoint", str(Path(previous_checkpoint).resolve())])
    rounded_steps = rounded_rollout_steps(steps, n_envs=controls.n_envs, n_steps=controls.n_steps)
    return CommandSpec(
        label=f"{spec.label}_chunk{chunk_index:02d}_train",
        kind="train",
        cwd=Path(source).resolve(),
        argv=argv,
        output=Path(output).resolve(),
        timeout_seconds=float(max_seconds) + 60.0,
        metadata={
            "phase": "0a",
            "seed": int(spec.seed),
            "net_arch": list(spec.net_arch),
            "run_label": spec.label,
            "runner": Path(runner).name,
            "group_size": int(controls.group_size),
            "chunk_index": int(chunk_index),
            "steps_semantics": "additional_timesteps",
            "checkpoint": str(Path(previous_checkpoint).resolve()) if previous_checkpoint else None,
            "expected_parameter_count": architecture_parameter_count(256, 10, spec.net_arch),
            "requested_steps": int(steps),
            "rounded_steps": int(rounded_steps),
            "expected_starting_steps": int(chunk_index - 1) * int(rounded_steps),
            "expected_ending_steps": int(chunk_index) * int(rounded_steps),
            "controls": dataclasses.asdict(controls),
            "constant_learning_rate": 0.0003,
            "constant_clip_range": 0.2,
            "nav_ent_coef": 0.0,
        },
    )


def build_eval_command(
    *,
    python: str,
    evaluate_modes: Path,
    source: Path,
    spec: RunSpec,
    chunk_index: int,
    eval_kind: str,
    checkpoint: Path,
    output: Path,
    episodes: int,
    near_seed: int,
    wide_seed: int,
    timeout_seconds: float,
) -> CommandSpec:
    argv = [
        python,
        str(Path(evaluate_modes).resolve()),
        "--repo",
        str(Path(source).resolve()),
        "--checkpoint",
        str(Path(checkpoint).resolve()),
        "--output",
        str(Path(output).resolve()),
        "--near-seed",
        str(int(near_seed)),
        "--wide-seed",
        str(int(wide_seed)),
        "--episodes",
        str(int(episodes)),
        "--device",
        "cuda",
    ]
    return CommandSpec(
        label=f"{spec.label}_chunk{chunk_index:02d}_eval_{eval_kind}",
        kind="eval",
        cwd=Path(source).resolve(),
        argv=argv,
        output=Path(output).resolve(),
        timeout_seconds=float(timeout_seconds),
        metadata={
            "run_label": spec.label,
            "chunk_index": int(chunk_index),
            "eval_kind": eval_kind,
            "episodes": int(episodes),
            "mastery_phase": "0a",
            "modes": ["deterministic", "sampled"],
        },
    )


def finite_summary_number(summary: dict[str, Any], key: str) -> float | None:
    if key not in summary:
        return None
    value = summary[key]
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def evaluate_mastery(output: Path, *, required_episodes: int, threshold: float, max_fall_rate: float = 0.05) -> dict[str, Any]:
    modes: dict[str, dict[str, Any]] = {}
    for mode in ("deterministic", "sampled"):
        path = Path(output) / f"0a-{mode}.json"
        if not path.is_file():
            return {"candidate_mastery": False, "phase": "0a", "reason": f"missing {path.name}", "modes": modes}
        summary = json.loads(path.read_text(encoding="utf-8")).get("summary", {})
        modes[mode] = summary
        if int(summary.get("episodes", -1)) != int(required_episodes):
            return {"candidate_mastery": False, "phase": "0a", "reason": f"{mode} episode count mismatch", "modes": modes}
        success_rate = finite_summary_number(summary, "success_rate")
        fall_rate = finite_summary_number(summary, "fall_rate")
        dealt = finite_summary_number(summary, "avg_damage_dealt")
        taken = finite_summary_number(summary, "avg_damage_taken")
        if success_rate is None or success_rate < float(threshold):
            return {"candidate_mastery": False, "phase": "0a", "reason": f"{mode} below threshold", "modes": modes}
        if fall_rate is None or fall_rate > float(max_fall_rate):
            return {"candidate_mastery": False, "phase": "0a", "reason": f"{mode} fall rate above threshold", "modes": modes}
        if dealt is None or taken is None:
            return {"candidate_mastery": False, "phase": "0a", "reason": f"{mode} damage summary non-finite", "modes": modes}
    return {"candidate_mastery": True, "phase": "0a", "threshold": float(threshold), "max_fall_rate": float(max_fall_rate), "modes": modes}


def architecture_parameter_count(obs_dim: int, action_dim: int, arch: tuple[int, ...]) -> int:
    def branch(out_dim: int) -> int:
        total = 0
        width_in = int(obs_dim)
        for width in arch:
            total += width_in * int(width) + int(width)
            width_in = int(width)
        return total + width_in * int(out_dim) + int(out_dim)

    return branch(action_dim) + branch(1) + action_dim


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    require_cuda_device(args.device)
    output = Path(args.output).resolve()
    artifacts = output / "artifacts"
    config_dir = artifacts / "configs"
    source_dir = artifacts / "sources"
    run_dir = artifacts / "runs"
    commands: list[CommandSpec] = []
    specs = default_run_specs()
    controls = TrainControls(args.n_envs, args.n_steps, args.batch_size, args.epochs, args.train_max_seconds, args.group_size)
    previous_by_spec: dict[str, Path | None] = {spec.label: None for spec in specs}
    for chunk in range(1, int(args.chunks) + 1):
        for spec in specs:
            config_path = config_dir / f"{spec.label}.yaml"
            variant_source = source_dir / spec.label
            previous_checkpoint = previous_by_spec[spec.label]
            chunk_output = run_dir / spec.label / f"chunk{chunk:02d}"
            commands.append(
                build_train_command(
                    python=args.python,
                    runner=args.learning_runner,
                    train_job=args.train_job,
                    grouped_env_path=args.grouped_env_path,
                    source=variant_source,
                    config=config_path,
                    spec=spec,
                    chunk_index=chunk,
                    previous_checkpoint=previous_checkpoint,
                    output=chunk_output,
                    steps=args.chunk_steps,
                    max_seconds=args.train_max_seconds,
                    controls=controls,
                )
            )
            checkpoint = chunk_output / "final.zip"
            previous_by_spec[spec.label] = checkpoint
            dev_output = run_dir / spec.label / f"chunk{chunk:02d}_eval_dev20"
            commands.append(
                build_eval_command(
                    python=args.python,
                    evaluate_modes=args.evaluate_modes,
                    source=variant_source,
                    spec=spec,
                    chunk_index=chunk,
                    eval_kind="dev20",
                    checkpoint=checkpoint,
                    output=dev_output,
                    episodes=args.dev_episodes,
                    near_seed=84001,
                    wide_seed=83001,
                    timeout_seconds=args.dev_eval_timeout,
                )
            )
            full_output = run_dir / spec.label / f"chunk{chunk:02d}_eval_full100"
            commands.append(
                build_eval_command(
                    python=args.python,
                    evaluate_modes=args.evaluate_modes,
                    source=variant_source,
                    spec=spec,
                    chunk_index=chunk,
                    eval_kind="full100",
                    checkpoint=checkpoint,
                    output=full_output,
                    episodes=args.full_episodes,
                    near_seed=85001,
                    wide_seed=86001,
                    timeout_seconds=args.full_eval_timeout,
                )
            )
    for command in commands:
        command.metadata["output_root"] = str(output)
    return {
        "label": "e6_fresh_architecture_phase0a_pilots",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload": str(Path(args.payload).resolve()),
        "source": str(Path(args.source).resolve()),
        "train_job": str(Path(args.train_job).resolve()),
        "evaluate_modes": str(Path(args.evaluate_modes).resolve()),
        "learning_runner": str(Path(args.learning_runner).resolve()),
        "grouped_env_path": str(Path(args.grouped_env_path).resolve()),
        "output": str(output),
        "artifacts": str(artifacts),
        "device": args.device,
        "matrix": [dataclasses.asdict(spec) for spec in specs],
        "controls": dataclasses.asdict(controls),
        "chunk_steps": int(args.chunk_steps),
        "max_chunks": int(args.chunks),
        "mastery": {"phase": "0a", "dev_episodes": int(args.dev_episodes), "full_episodes": int(args.full_episodes), "threshold": float(args.mastery_threshold)},
        "experiment_diff_note": "Relative to the current frozen control config, generated configs change policy_kwargs.net_arch and ppo_settings.nav_ent_coef=0.0; learning_rate=0.0003 and clip_range=0.2 are pinned constants for all arms.",
        "schedule_note": "Chunking train_job restarts SB3 schedule horizons; generated configs pin learning_rate=0.0003 and clip_range=0.2 for controlled E6 pilot comparability.",
        "expected_parameter_counts": {
            "-".join(str(width) for width in arch): architecture_parameter_count(args.obs_dim, args.action_dim, tuple(arch))
            for arch in DEFAULT_ARCHITECTURES
        },
        "commands": [dataclasses.asdict(command) for command in commands],
    }


def materialize_plan_inputs(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    output = Path(plan["output"])
    for spec in default_run_specs():
        config = output / "artifacts" / "configs" / f"{spec.label}.yaml"
        write_arch_config(args.config, config, spec.net_arch)
        prepare_variant_source(args.source, output / "artifacts" / "sources" / spec.label, config)
    for command in plan["commands"]:
        if command["kind"] == "train":
            config_path = Path(command["argv"][command["argv"].index("--config") + 1])
            command["metadata"]["config_sha256"] = sha256_file(config_path)
        elif command["kind"] == "eval":
            source_config = Path(command["cwd"]) / "config.yaml"
            command["metadata"]["config_sha256"] = sha256_file(source_config)
            checkpoint_index = command["argv"].index("--checkpoint") + 1
            command["metadata"]["checkpoint"] = command["argv"][checkpoint_index]
    plan_sha = payload_sha256(plan)
    write_json(output / "artifacts" / "campaign_learning_plan.json", plan)
    provenance = {
        "payload_manifest": file_record(Path(args.payload) / "source-manifest.json"),
        "source_config": file_record(args.config),
        "train_job": file_record(args.train_job),
        "evaluate_modes": file_record(args.evaluate_modes),
        "learning_runner": file_record(args.learning_runner),
        "grouped_env_path": file_record(args.grouped_env_path),
        "harness": file_record(Path(__file__)),
    }
    write_json(output / "artifacts" / "campaign_learning_provenance.json", provenance)
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan_sha,
        "plan_path": str(output / "artifacts" / "campaign_learning_plan.json"),
        "provenance_path": str(output / "artifacts" / "campaign_learning_provenance.json"),
        "args": {key: json_safe(value) for key, value in vars(args).items()},
        "provenance": provenance,
    }
    receipt_dir = output / "artifacts" / "invocation_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    nonce = time.time_ns()
    receipt_path = receipt_dir / f"{nonce}-{plan_sha[:12]}.json"
    while receipt_path.exists():
        nonce += 1
        receipt_path = receipt_dir / f"{nonce}-{plan_sha[:12]}.json"
    write_json(receipt_path, receipt)
    write_json(output / "artifacts" / "campaign_learning_latest_receipt.json", {"receipt": str(receipt_path), "plan_sha256": plan_sha})


def command_env(command: CommandSpec) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "offline",
            "SDL_VIDEODRIVER": "dummy",
            "OMP_NUM_THREADS": "1",
            "WANDB_NOTES": (
                f"Goal: E6 fresh architecture phase0a learning pilot. "
                f"Change: {command.label}. Expect: real checkpoint/eval evidence, not synthetic mastery."
            ),
        }
    )
    return env


def validate_train_artifacts(command: CommandSpec) -> dict[str, Any]:
    result_path = command.output / "result.json"
    metadata_path = command.output / "metadata.json"
    runtime_device_path = command.output / "runtime_device.json"
    if not result_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"missing train result/metadata for {command.label}")
    if not runtime_device_path.is_file():
        raise RuntimeError(f"missing runtime CUDA device proof for {command.label}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    runtime_device = json.loads(runtime_device_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise RuntimeError(f"train did not complete for {command.label}: {result.get('status')}")
    if result.get("error") is not None:
        raise RuntimeError(f"train error was not clean for {command.label}: {result.get('error')}")
    if result.get("stop_reason") is not None:
        raise RuntimeError(f"train stop_reason was not clean for {command.label}: {result.get('stop_reason')}")
    wall_seconds = float(result.get("wall_seconds", math.nan))
    if not math.isfinite(wall_seconds) or wall_seconds < 0:
        raise RuntimeError(f"train wall_seconds invalid for {command.label}")
    rounded_steps = int(command.metadata.get("rounded_steps", 0))
    expected_starting_steps = int(command.metadata.get("expected_starting_steps", 0))
    expected_ending_steps = int(command.metadata.get("expected_ending_steps", expected_starting_steps + rounded_steps))
    starting_steps = int(result.get("starting_steps", 0))
    ending_steps = int(result.get("ending_steps", starting_steps))
    actual_steps = int(result.get("actual_steps", ending_steps - starting_steps))
    if starting_steps != expected_starting_steps or ending_steps != expected_ending_steps:
        raise RuntimeError(
            f"train cumulative step mismatch for {command.label}: "
            f"expected {expected_starting_steps}->{expected_ending_steps}, got {starting_steps}->{ending_steps}"
        )
    if ending_steps - starting_steps != rounded_steps or actual_steps != rounded_steps:
        raise RuntimeError(
            f"train rounded step mismatch for {command.label}: expected {rounded_steps}, "
            f"ending-starting={ending_steps - starting_steps}, actual={actual_steps}"
        )
    expected_checkpoints = {
        "initial_checkpoint": command.output / "initial.zip",
        "latest_checkpoint": command.output / "latest.zip",
        "final_checkpoint": command.output / "final.zip",
    }
    checkpoint_evidence: dict[str, dict[str, Any]] = {}
    for key, expected_path in expected_checkpoints.items():
        reported = Path(result.get(key) or "")
        if reported.resolve() != expected_path.resolve():
            raise RuntimeError(f"{key} path mismatch for {command.label}: expected {expected_path}, got {reported}")
        if not reported.is_file():
            raise RuntimeError(f"missing {key} for {command.label}: {reported}")
        checkpoint_evidence[key] = file_record(reported)
    final_checkpoint = Path(result["final_checkpoint"])
    if not runtime_device.get("all_policy_parameters_cuda"):
        raise RuntimeError(f"runtime device proof did not confirm CUDA parameters for {command.label}")
    if not str(runtime_device.get("policy_device", "")).startswith("cuda"):
        raise RuntimeError(f"runtime policy device is not CUDA for {command.label}")
    if metadata.get("seed") != command.metadata.get("seed"):
        raise RuntimeError(f"seed metadata mismatch for {command.label}")
    expected_checkpoint = command.metadata.get("checkpoint")
    metadata_checkpoint = metadata.get("checkpoint")
    if expected_checkpoint is None:
        if starting_steps != 0:
            raise RuntimeError(f"fresh train did not start at zero for {command.label}")
        if metadata_checkpoint is not None:
            raise RuntimeError(f"fresh train unexpectedly recorded checkpoint for {command.label}: {metadata_checkpoint}")
        if metadata.get("input_checkpoint_sha256") is not None:
            raise RuntimeError(f"fresh train unexpectedly recorded input checkpoint sha for {command.label}")
    else:
        expected_checkpoint_path = Path(expected_checkpoint).resolve()
        if metadata_checkpoint != str(expected_checkpoint_path):
            raise RuntimeError(f"checkpoint metadata mismatch for {command.label}: expected {expected_checkpoint_path}, got {metadata_checkpoint}")
        if not expected_checkpoint_path.is_file():
            raise RuntimeError(f"missing expected input checkpoint for {command.label}: {expected_checkpoint_path}")
        actual_input_sha = sha256_file(expected_checkpoint_path)
        if metadata.get("input_checkpoint_sha256") != actual_input_sha:
            raise RuntimeError(f"input checkpoint sha mismatch for {command.label}")
    expected_config_sha = command.metadata.get("config_sha256")
    if expected_config_sha is not None and metadata.get("config_sha256") != expected_config_sha:
        raise RuntimeError(f"config sha metadata mismatch for {command.label}")
    if metadata.get("parameter_count") != command.metadata.get("expected_parameter_count"):
        raise RuntimeError(f"parameter count mismatch for {command.label}")
    ppo_controls = metadata.get("ppo_controls", {})
    expected_controls = command.metadata.get("controls", {})
    expected_ppo_controls = {
        "net_arch": command.metadata.get("net_arch"),
        "n_steps": expected_controls.get("n_steps"),
        "batch_size": expected_controls.get("batch_size"),
        "n_epochs": expected_controls.get("epochs"),
        "ent_coef": command.metadata.get("nav_ent_coef"),
        "reset_num_timesteps": expected_checkpoint is None,
    }
    for key, expected in expected_ppo_controls.items():
        actual = ppo_controls.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"{key} metadata mismatch for {command.label}: expected {expected}, got {actual}")
        elif actual != expected:
            raise RuntimeError(f"{key} metadata mismatch for {command.label}: expected {expected}, got {actual}")
    if metadata.get("n_envs") is not None and metadata.get("n_envs") != expected_controls.get("n_envs"):
        raise RuntimeError(f"n_envs metadata mismatch for {command.label}: expected {expected_controls.get('n_envs')}, got {metadata.get('n_envs')}")
    return {
        "result": str(result_path),
        "metadata": str(metadata_path),
        "runtime_device": str(runtime_device_path),
        "final_checkpoint": str(final_checkpoint),
        "latest_checkpoint": str(expected_checkpoints["latest_checkpoint"]),
        "initial_checkpoint": str(expected_checkpoints["initial_checkpoint"]),
        "checkpoint_evidence": checkpoint_evidence,
        "input_checkpoint": expected_checkpoint,
        "input_checkpoint_sha256": metadata.get("input_checkpoint_sha256"),
        "parameter_count": metadata.get("parameter_count"),
        "actual_steps": actual_steps,
        "starting_steps": starting_steps,
        "ending_steps": ending_steps,
        "wall_seconds": wall_seconds,
    }


def validate_eval_artifacts(command: CommandSpec) -> dict[str, Any]:
    summaries = {}
    metadata_path = command.output / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"missing eval metadata for {command.label}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("training_performed") is not False:
        raise RuntimeError(f"eval metadata did not confirm read-only evaluation for {command.label}")
    if not str(metadata.get("actual_policy_device", "")).startswith("cuda"):
        raise RuntimeError(f"eval metadata did not confirm CUDA policy for {command.label}")
    if command.metadata.get("checkpoint") is not None and metadata.get("checkpoint") != command.metadata.get("checkpoint"):
        raise RuntimeError(f"eval checkpoint metadata mismatch for {command.label}")
    if command.metadata.get("checkpoint") is not None:
        checkpoint = Path(command.metadata["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise RuntimeError(f"eval checkpoint missing for {command.label}: {checkpoint}")
        if metadata.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise RuntimeError(f"eval checkpoint sha mismatch for {command.label}")
    if command.metadata.get("config_sha256") is not None and metadata.get("config_sha256") != command.metadata.get("config_sha256"):
        raise RuntimeError(f"eval config sha metadata mismatch for {command.label}")
    expected_episodes = int(command.metadata.get("episodes"))
    for phase in ("0a", "0b"):
        for mode in ("deterministic", "sampled"):
            path = command.output / f"{phase}-{mode}.json"
            if not path.is_file():
                raise RuntimeError(f"missing {phase} {mode} eval artifact for {command.label}")
            summary = json.loads(path.read_text(encoding="utf-8")).get("summary", {})
            for key in ("episodes", "success_rate", "fall_rate", "nav_falls"):
                if key not in summary:
                    raise RuntimeError(f"eval summary missing {key} for {command.label}")
            if int(summary.get("episodes", -1)) != expected_episodes:
                raise RuntimeError(f"eval episode count mismatch for {command.label}")
            for key, value in summary.items():
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    raise RuntimeError(f"eval summary non-finite {key} for {command.label}")
            summaries[f"{phase}-{mode}"] = summary
    return {"metadata": str(metadata_path), "summaries": summaries}


def run_command(command: CommandSpec, *, deadline: float | None, reserve: float) -> dict[str, Any]:
    if command.kind == "train" and is_completed_output(command.output):
        evidence = validate_train_artifacts(command)
        return {"label": command.label, "kind": command.kind, "status": "skipped_completed", "output": str(command.output), "evidence": evidence}
    if command.kind == "eval" and Path(command.output).is_dir() and (Path(command.output) / "summary.json").is_file():
        evidence = validate_eval_artifacts(command)
        return {"label": command.label, "kind": command.kind, "status": "skipped_existing", "output": str(command.output), "evidence": evidence}
    assert_not_completed_output(command.output)
    ensure_launch_safety(command)
    assert_deadline_allows(now=time.time(), deadline=deadline, timeout=command.timeout_seconds, reserve=reserve)
    command.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = command.output.with_suffix(".log")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command.argv,
            cwd=str(command.cwd),
            env=command_env(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=command.timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = proc.wait(timeout=10.0)
    status = "timeout" if timed_out else ("completed" if returncode == 0 else "failed")
    evidence = None
    if status == "completed":
        evidence = validate_train_artifacts(command) if command.kind == "train" else validate_eval_artifacts(command)
    return {
        "label": command.label,
        "kind": command.kind,
        "status": status,
        "returncode": returncode,
        "started_unix": started,
        "ended_unix": time.time(),
        "output": str(command.output),
        "log": str(log_path),
        "evidence": evidence,
    }


def archive_existing_execution_manifest(manifest_path: Path) -> Path | None:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return None
    previous_bytes = manifest_path.read_bytes()
    try:
        previous = json.loads(previous_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"malformed existing execution manifest; refusing overwrite: {manifest_path}") from exc
    if not isinstance(previous, dict) or not isinstance(previous.get("status"), str) or not isinstance(previous.get("results"), list):
        raise RuntimeError(f"invalid existing execution manifest; refusing overwrite: {manifest_path}")
    digest = hashlib.sha256(previous_bytes).hexdigest()
    archive_path = manifest_path.parent / "execution_history" / f"{digest}.json"
    write_bytes_if_new_or_same(archive_path, previous_bytes)
    return archive_path


def execute_plan(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    commands = [CommandSpec(**{**item, "cwd": Path(item["cwd"]), "output": Path(item["output"])}) for item in plan["commands"]]
    results: list[dict[str, Any]] = []
    manifest_path = Path(plan["output"]) / "artifacts" / "campaign_learning_results.json"
    archive_existing_execution_manifest(manifest_path)
    mastered: set[str] = set()
    for command in commands:
        label_root = str(command.metadata.get("run_label"))
        if label_root in mastered:
            results.append({"label": command.label, "kind": command.kind, "status": "skipped_after_mastery"})
            continue
        if command.kind == "eval" and "full100" in command.label:
            dev_output = command.output.parent / command.output.name.replace("full100", "dev20")
            dev_verdict = evaluate_mastery(dev_output, required_episodes=args.dev_episodes, threshold=args.mastery_threshold)
            if not dev_verdict["candidate_mastery"]:
                results.append({"label": command.label, "kind": command.kind, "status": "skipped_not_candidate", "dev_verdict": dev_verdict})
                write_json(manifest_path, {"status": "running", "results": results})
                continue
        try:
            result = run_command(command, deadline=args.deadline, reserve=args.reserve_seconds)
        except Exception as exc:
            result = {
                "label": command.label,
                "kind": command.kind,
                "status": "failed",
                "output": str(command.output),
                "error": str(exc),
            }
        results.append(result)
        if result["status"] not in {"completed", "skipped_completed", "skipped_existing"}:
            write_json(manifest_path, {"status": "failed", "results": results})
            return {"status": "failed", "results": results}
        if command.kind == "eval" and "full100" in command.label:
            full_verdict = evaluate_mastery(command.output, required_episodes=args.full_episodes, threshold=args.mastery_threshold)
            results[-1]["mastery_verdict"] = full_verdict
            if full_verdict["candidate_mastery"]:
                mastered.add(label_root)
        write_json(manifest_path, {"status": "running", "results": results})
    payload = {"status": "completed", "results": results, "mastered": sorted(mastered)}
    write_json(manifest_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_SOURCE / "control.yaml")
    parser.add_argument("--train-job", type=Path, default=DEFAULT_TRAIN_JOB)
    parser.add_argument("--evaluate-modes", type=Path, default=DEFAULT_EVALUATE_MODES)
    parser.add_argument("--learning-runner", type=Path, default=DEFAULT_LEARNING_RUNNER)
    parser.add_argument("--grouped-env-path", type=Path, default=DEFAULT_GROUPED_ENV)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-steps", type=int, default=DEFAULT_CHUNK_STEPS)
    parser.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--train-max-seconds", type=float, default=DEFAULT_TRAIN_TIMEOUT_SECONDS)
    parser.add_argument("--dev-episodes", type=int, default=DEFAULT_DEV_EPISODES)
    parser.add_argument("--full-episodes", type=int, default=DEFAULT_FULL_EPISODES)
    parser.add_argument("--mastery-threshold", type=float, default=DEFAULT_MASTERY_THRESHOLD)
    parser.add_argument("--dev-eval-timeout", type=float, default=DEFAULT_DEV_EVAL_TIMEOUT_SECONDS)
    parser.add_argument("--full-eval-timeout", type=float, default=DEFAULT_FULL_EVAL_TIMEOUT_SECONDS)
    parser.add_argument("--reserve-seconds", type=float, default=DEFAULT_RESERVE_SECONDS)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--obs-dim", type=int, default=256)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--execute", action="store_true", help="actually run the generated commands sequentially")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    require_cuda_device(args.device)
    if args.chunk_steps <= 0 or args.chunks <= 0:
        raise SystemExit("--chunk-steps and --chunks must be positive")
    if args.mastery_threshold <= 0 or args.mastery_threshold > 1 or not math.isfinite(args.mastery_threshold):
        raise SystemExit("--mastery-threshold must be finite in (0, 1]")
    for path_attr in ("payload", "source", "config", "train_job", "evaluate_modes", "learning_runner", "grouped_env_path"):
        path = getattr(args, path_attr)
        if not Path(path).exists():
            raise SystemExit(f"{path_attr} does not exist: {path}")
        setattr(args, path_attr, Path(path).resolve())
    args.output = Path(args.output).resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    materialize_plan_inputs(args, plan)
    result = {"plan": str(args.output / "artifacts" / "campaign_learning_plan.json"), "commands": len(plan["commands"]), "executed": False}
    if args.execute:
        result["run"] = execute_plan(args, plan)
        result["executed"] = True
    print(json.dumps(json_safe(result), sort_keys=True))
    if args.execute and result.get("run", {}).get("status") != "completed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
