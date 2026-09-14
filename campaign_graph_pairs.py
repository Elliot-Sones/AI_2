#!/usr/bin/env python3
"""Dry-run-first launcher for sustained E4 rollout-policy CUDA graph pairs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import benchmark_campaign
import campaign_suite


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HARNESS = REPO_ROOT / "benchmark_campaign.py"
DEFAULT_ROLLOUT_POLICY_GRAPH = REPO_ROOT / "rollout_policy_graph.py"
DEFAULT_PARITY_VERIFIER = REPO_ROOT / "benchmark_graph_rollout_parity.py"
DEFAULT_POLICY_COMPILE = REPO_ROOT / "benchmark_policy_compile.py"
DEFAULT_POLICY_CUDA_GRAPH = REPO_ROOT / "benchmark_policy_cuda_graph.py"

SEED_ARM_ORDER = (
    (45, ("control", "graph")),
    (46, ("graph", "control")),
    (47, ("control", "graph")),
)
N_ENVS = 64
N_STEPS = 512
GROUP_SIZE = 1
TRANSITIONS_PER_CYCLE = 32768
ALLOWED_GEOMETRIES = ((64, 512, 1), (256, 128, 4))
WARMUP_CYCLES = 2
MEASURED_CYCLES = 100
BATCH_SIZE = 8192
EPOCHS = 8
MAX_PER_JOB_TIMEOUT = 600
RESERVE_SECONDS = 300
MIN_FREE_DISK_BYTES = 3 * 1024**3
INVOCATION_PROVENANCE = "invocation_provenance.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--per-job-timeout", type=float, default=MAX_PER_JOB_TIMEOUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", default="0b")
    parser.add_argument("--config", type=Path, default=benchmark_campaign.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=benchmark_campaign.DEFAULT_CHECKPOINT)
    parser.add_argument("--source", type=Path, default=benchmark_campaign.DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=benchmark_campaign.DEFAULT_TRAIN_JOB)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--rollout-cache-path", type=Path, default=campaign_suite.DEFAULT_ROLLOUT_CACHE)
    parser.add_argument("--grouped-env-path", type=Path, default=campaign_suite.DEFAULT_GROUPED_ENV)
    parser.add_argument("--rollout-policy-graph-path", type=Path, default=DEFAULT_ROLLOUT_POLICY_GRAPH)
    parser.add_argument("--parity-summary", type=Path)
    parser.add_argument("--verifier-path", type=Path, default=DEFAULT_PARITY_VERIFIER)
    parser.add_argument("--policy-compile-path", type=Path, default=DEFAULT_POLICY_COMPILE)
    parser.add_argument("--policy-cuda-graph-path", type=Path, default=DEFAULT_POLICY_CUDA_GRAPH)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--execute", action="store_true", help="run CUDA jobs; default is dry-run only")
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.output = Path(args.output).resolve()
    for attr in (
        "config",
        "checkpoint",
        "source",
        "train_job",
        "harness",
        "rollout_cache_path",
        "grouped_env_path",
        "rollout_policy_graph_path",
        "verifier_path",
        "policy_compile_path",
        "policy_cuda_graph_path",
    ):
        setattr(args, attr, Path(getattr(args, attr)).resolve())
    args.parity_summary = (
        (args.output / "graph_rollout_parity_summary.json")
        if getattr(args, "parity_summary", None) is None
        else Path(args.parity_summary).resolve()
    )
    args.per_job_timeout = float(args.per_job_timeout)
    args.deadline_unix = float(args.deadline_unix)
    if not math.isfinite(args.per_job_timeout) or args.per_job_timeout <= 0:
        raise ValueError("--per-job-timeout must be positive")
    if args.per_job_timeout > MAX_PER_JOB_TIMEOUT:
        raise ValueError(f"--per-job-timeout must be <= {MAX_PER_JOB_TIMEOUT}")
    if not math.isfinite(args.deadline_unix):
        raise ValueError("--deadline-unix must be finite")
    if bool(args.execute) and args.deadline_unix <= time.time():
        raise ValueError("--deadline-unix must be in the future for --execute")
    if not hasattr(args, "n_envs"):
        args.n_envs = N_ENVS
    if not hasattr(args, "n_steps"):
        args.n_steps = N_STEPS
    if not hasattr(args, "group_size"):
        args.group_size = GROUP_SIZE
    geometry = (int(args.n_envs), int(args.n_steps), int(args.group_size))
    if geometry not in ALLOWED_GEOMETRIES:
        raise ValueError(f"unsupported graph-pairs geometry: {geometry}; allowed={ALLOWED_GEOMETRIES}")
    args.mode = "graph-pairs"
    return args


def plan_jobs(args: argparse.Namespace) -> list[campaign_suite.SuiteJob]:
    jobs: list[campaign_suite.SuiteJob] = []
    for seed, arms in SEED_ARM_ORDER:
        for arm in arms:
            jobs.append(
                campaign_suite.SuiteJob(
                    name=f"graph-pairs-seed{seed}-{arm}",
                    arm=arm,
                    seed=seed,
                    output=args.output / f"graph-pairs-seed{seed}-{arm}",
                    n_envs=int(args.n_envs),
                    n_steps=int(args.n_steps),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    warmup=WARMUP_CYCLES,
                    cycles=MEASURED_CYCLES,
                    group_size=int(args.group_size),
                    resident_buffer=False,
                )
            )
    return jobs


def command_maxseconds(per_job_timeout: float) -> float:
    timeout = float(per_job_timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--per-job-timeout must be positive")
    return max(0.001, timeout * 0.95)


def format_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def command_for_job(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> list[str]:
    command = campaign_suite.command_for_job(args, job)
    if job.arm == "graph":
        command.extend(["--rollout-policy-graph", "--rollout-policy-graph-path", str(args.rollout_policy_graph_path)])
    return [*command, "--maxseconds", format_seconds(command_maxseconds(float(args.per_job_timeout)))]


def ensure_disk_free(output: Path, *, disk_usage: Callable[[Path], Any] = shutil.disk_usage) -> int:
    free_bytes = int(disk_usage(Path(output)).free)
    if free_bytes <= MIN_FREE_DISK_BYTES:
        raise RuntimeError(f"--execute requires more than at least 3 GiB free disk at {output}; got {free_bytes} bytes")
    return free_bytes


def frozen_train_path(args: argparse.Namespace) -> Path:
    return args.source / "train.py"


def expected_file_records(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "harness": campaign_suite.file_record(args.harness),
        "train_job": campaign_suite.file_record(args.train_job),
        "config": campaign_suite.file_record(args.config),
        "checkpoint": campaign_suite.file_record(args.checkpoint),
        "source_train": campaign_suite.file_record(frozen_train_path(args)),
        "frozen_environment": campaign_suite.file_record(campaign_suite.frozen_environment_path(args)),
        "grouped_env": campaign_suite.file_record(args.grouped_env_path),
        "rollout_policy_graph": campaign_suite.file_record(args.rollout_policy_graph_path),
        "parity_summary": campaign_suite.file_record(args.parity_summary),
        "verifier": campaign_suite.file_record(args.verifier_path),
        "policy_compile": campaign_suite.file_record(args.policy_compile_path),
        "policy_cuda_graph": campaign_suite.file_record(args.policy_cuda_graph_path),
    }


def require_existing_file_records(records: dict[str, Any]) -> None:
    for name, record in records.items():
        if not isinstance(record, dict) or record.get("exists") is not True:
            path = record.get("path") if isinstance(record, dict) else None
            raise RuntimeError(f"required provenance file is missing: {name} path={path}")


def _sha(record: dict[str, Any], field: str, *, label: str) -> str:
    item = record.get(field)
    if not isinstance(item, dict) or item.get("exists") is not True:
        raise RuntimeError(f"{label} missing required file record: {field}")
    digest = item.get("sha256")
    if not isinstance(digest, str) or not digest:
        raise RuntimeError(f"{label} missing required sha256: {field}")
    return digest


def _require_same_hash(before: dict[str, Any], after: dict[str, Any], field: str, *, label: str) -> str:
    before_sha = _sha(before, field, label=f"{label} before")
    after_sha = _sha(after, field, label=f"{label} after")
    if before_sha != after_sha:
        raise RuntimeError(f"{label} input changed during parity run: {field}")
    return before_sha


def _validate_graph_summary(graph: Any) -> None:
    if not isinstance(graph, dict):
        raise RuntimeError("graph runtime summary must be an object")
    if int(graph.get("captures", 0)) <= 0:
        raise RuntimeError(f"graph runtime captures must be positive: {graph.get('captures')}")
    if int(graph.get("replays", 0)) <= 0:
        raise RuntimeError(f"graph runtime replays must be positive: {graph.get('replays')}")
    preflight = graph.get("preflight")
    if not isinstance(preflight, dict) or preflight.get("status") != "passed":
        raise RuntimeError("graph runtime preflight must be passed")
    repeats = int(preflight.get("stochastic_repeats", 0))
    stochastic = preflight.get("stochastic_exact")
    if repeats <= 0 or not isinstance(stochastic, list) or len(stochastic) != repeats:
        raise RuntimeError("graph runtime preflight stochastic_exact records must match stochastic_repeats")
    if any(not isinstance(item, dict) or item.get("passed") is not True for item in stochastic):
        raise RuntimeError("graph runtime stochastic preflight did not pass")
    deterministic = preflight.get("deterministic_no_rng")
    if preflight.get("setup_rng_restored") is not True:
        raise RuntimeError("graph runtime preflight setup_rng_restored must be true")
    if not isinstance(deterministic, dict) or deterministic.get("passed") is not True:
        raise RuntimeError("graph runtime deterministic preflight did not pass")
    for field in ("input_spec", "eps_spec"):
        spec = graph.get(field)
        if not isinstance(spec, dict) or not str(spec.get("device", "")).startswith("cuda"):
            raise RuntimeError(f"graph runtime {field} must attest CUDA device")
    for field in ("helper_direct_sb3", "graph_direct_sb3"):
        if not isinstance(graph.get(field), dict):
            raise RuntimeError(f"graph runtime {field} is required")


def _validate_parity_inputs(summary: dict[str, Any], records: dict[str, Any]) -> None:
    pairs = {
        "source_train": "source_train",
        "config": "config",
        "train_job": "train_job",
        "checkpoint": "checkpoint",
        "verifier": "verifier",
        "campaign": "harness",
        "grouped_env": "grouped_env",
        "rollout_policy_graph": "rollout_policy_graph",
        "policy_compile": "policy_compile",
        "policy_cuda_graph": "policy_cuda_graph",
    }
    control = summary.get("control")
    graph = summary.get("graph")
    if not isinstance(control, dict) or not isinstance(graph, dict):
        raise RuntimeError("parity summary missing control/graph receipts")
    for label, receipt in (("control", control), ("graph", graph)):
        before = receipt.get("provenance_before")
        after = receipt.get("provenance_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise RuntimeError(f"parity summary missing {label} provenance")
        for parity_field, record_field in pairs.items():
            parity_sha = _require_same_hash(before, after, parity_field, label=label)
            current_sha = records[record_field].get("sha256")
            if parity_sha != current_sha:
                raise RuntimeError(f"parity summary {parity_field} does not match current {record_field}")
    _validate_graph_summary(graph.get("rollout_policy_graph"))


def validate_parity_summary(args: argparse.Namespace, records: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(args.parity_summary)
    if not path.is_file():
        raise RuntimeError(f"graph pairs require completed graph rollout parity summary: {path}")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"graph rollout parity summary malformed: {path}") from exc
    if not isinstance(summary, dict):
        raise RuntimeError(f"graph rollout parity summary must be an object: {path}")
    if summary.get("status") != "completed":
        raise RuntimeError(f"graph rollout parity summary status must be completed: {summary.get('status')!r}")
    parity = summary.get("parity")
    if not isinstance(parity, dict):
        raise RuntimeError("graph rollout parity summary missing parity section")
    if parity.get("byte_parity") is not True:
        raise RuntimeError(f"graph rollout parity summary byte_parity must be true: {parity.get('byte_parity')!r}")
    expected_transitions = int(args.n_envs) * int(args.n_steps)
    if int(parity.get("expected_steps", -1)) != expected_transitions:
        raise RuntimeError("graph rollout parity summary expected_steps mismatch")
    if int(parity.get("expected_epochs", -1)) != EPOCHS:
        raise RuntimeError("graph rollout parity summary expected_epochs mismatch")
    if int(parity.get("expected_optimizer_steps", -1)) != EPOCHS * math.ceil(expected_transitions / BATCH_SIZE):
        raise RuntimeError("graph rollout parity summary expected_optimizer_steps mismatch")
    validate_parity_geometry_controls(summary, args)
    records = records or expected_file_records(args)
    require_existing_file_records(records)
    _validate_parity_inputs(summary, records)
    return summary


def validate_parity_geometry_controls(summary: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "n_envs": int(args.n_envs),
        "n_steps": int(args.n_steps),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "group_size": int(args.group_size),
    }
    for label in ("control", "graph"):
        receipt = summary.get(label)
        if not isinstance(receipt, dict):
            raise RuntimeError(f"graph rollout parity summary missing {label} receipt")
        controls = receipt.get("provenance_before", {}).get("controls")
        if not isinstance(controls, dict):
            raise RuntimeError(f"graph rollout parity summary {label} missing controls")
        for field, value in expected.items():
            if int(controls.get(field, -1)) != int(value):
                raise RuntimeError(
                    f"graph rollout parity summary {label} {field} mismatch: {controls.get(field)!r} != {value!r}"
                )


def preflight(
    args: argparse.Namespace,
    *,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    if str(args.device).lower() != "cuda":
        raise ValueError("--execute requires CUDA device")
    if not math.isfinite(float(args.per_job_timeout)) or float(args.per_job_timeout) <= 0:
        raise ValueError("--per-job-timeout must be positive")
    if float(args.per_job_timeout) > MAX_PER_JOB_TIMEOUT:
        raise ValueError(f"--per-job-timeout must be <= {MAX_PER_JOB_TIMEOUT}")
    free_bytes = ensure_disk_free(args.output, disk_usage=disk_usage)
    records = expected_file_records(args)
    parity = validate_parity_summary(args, records)
    return {"free_disk_bytes": free_bytes, "graph_rollout_parity_summary": parity}


def append_invocation_provenance(
    args: argparse.Namespace,
    *,
    jobs: list[campaign_suite.SuiteJob],
    status: str,
    created: float,
    preflight_record: dict[str, Any] | None = None,
) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    record = {
        "status": status,
        "created_unix": created,
        "execute": bool(args.execute),
        "planned_jobs": len(jobs),
        "transitions_per_cycle": TRANSITIONS_PER_CYCLE,
        "reserve_seconds": RESERVE_SECONDS,
        "per_job_timeout": float(args.per_job_timeout),
        "local_maxseconds": command_maxseconds(float(args.per_job_timeout)),
        "deadline_unix": float(args.deadline_unix),
        "provenance": expected_file_records(args),
    }
    if preflight_record is not None:
        record["preflight"] = preflight_record
    with (args.output / INVOCATION_PROVENANCE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(campaign_suite.json_safe(record), sort_keys=True) + "\n")


def dry_run(args: argparse.Namespace, jobs: list[campaign_suite.SuiteJob]) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    commands = [command_for_job(args, job) for job in jobs]
    append_invocation_provenance(args, jobs=jobs, status="dry_run", created=time.time())
    campaign_suite.write_json(
        args.output / "dry_run_plan.json",
        {
            "status": "dry_run",
            "execute": False,
            "planned_jobs": len(jobs),
            "transitions_per_cycle": TRANSITIONS_PER_CYCLE,
            "reserve_seconds": RESERVE_SECONDS,
            "per_job_timeout": args.per_job_timeout,
            "deadline_unix": args.deadline_unix,
            "provenance": expected_file_records(args),
            "jobs": [
                {
                    "name": job.name,
                    "arm": job.arm,
                    "seed": job.seed,
                    "n_envs": job.n_envs,
                    "n_steps": job.n_steps,
                    "batch_size": job.batch_size,
                    "epochs": job.epochs,
                    "warmup": job.warmup,
                    "cycles": job.cycles,
                    "group_size": job.group_size,
                    "transitions_per_cycle": job.n_envs * job.n_steps,
                    "command": command,
                }
                for job, command in zip(jobs, commands, strict=True)
            ],
        },
    )
    write_suite_report(args.output, status="dry_run", args=args, started=time.time(), ended=time.time(), planned_jobs=len(jobs), records=[])
    return 0


def write_job_manifest(
    args: argparse.Namespace,
    job: campaign_suite.SuiteJob,
    command: list[str],
    env: dict[str, str],
) -> None:
    campaign_suite.write_job_manifest(args, job, command, env)
    manifest_path = job.output / "job_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = expected_file_records(args)
    manifest["graph_pairs"] = {
        "transitions_per_cycle": TRANSITIONS_PER_CYCLE,
        "runner_timeout_seconds": float(args.per_job_timeout),
        "local_maxseconds": command_maxseconds(float(args.per_job_timeout)),
        "parity": validate_parity_summary(args, manifest["files"]).get("parity"),
    }
    campaign_suite.write_json(manifest_path, manifest)


def validate_graph_job_summary(job: campaign_suite.SuiteJob) -> str | None:
    summary_path = job.output / "campaign_summary.json"
    if not summary_path.is_file():
        return f"missing summary: {summary_path}"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"malformed summary: {exc}"
    if job.arm != "graph":
        graph = summary.get("rollout_policy_graph")
        if isinstance(graph, dict) and graph.get("enabled") is True:
            return "control job unexpectedly enabled rollout policy graph"
        return None
    graph = summary.get("rollout_policy_graph")
    if not isinstance(graph, dict) or graph.get("enabled") is not True:
        return "graph job missing enabled rollout_policy_graph summary"
    records = graph.get("records")
    if not isinstance(records, list) or not records:
        return "graph job missing rollout_policy_graph runtime records"
    try:
        for record in records:
            _validate_graph_summary(record)
    except RuntimeError as exc:
        return str(exc)
    return None


def validate_job_summary(job: campaign_suite.SuiteJob) -> str | None:
    error = campaign_suite.validate_job_summary(job)
    if error is not None:
        return error
    return validate_graph_job_summary(job)


def validate_existing_manifest(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> str | None:
    manifest_path = job.output / "job_manifest.json"
    if not manifest_path.is_file():
        return f"missing manifest: {manifest_path}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"malformed manifest: {exc}"
    if manifest.get("job") != campaign_suite.json_safe(dataclasses.asdict(job)):
        return "job controls do not match current plan"
    expected_command = command_for_job(args, job)
    if manifest.get("command") != expected_command:
        return "command does not match current plan"
    expected_env = campaign_suite.whitelisted_env(job=job, args=args)
    if manifest.get("env") != expected_env:
        return "env does not match current plan"
    files = manifest.get("files")
    if not isinstance(files, dict):
        return "files manifest is missing"
    expected_files = expected_file_records(args)
    for name, expected in expected_files.items():
        if expected.get("exists") is not True:
            return f"{name} current file record is missing"
        if files.get(name) != expected:
            return f"{name} file record does not match current inputs"
    graph_pairs = manifest.get("graph_pairs")
    if not isinstance(graph_pairs, dict):
        return "graph_pairs manifest section is missing"
    current_parity = validate_parity_summary(args, expected_files).get("parity")
    if graph_pairs.get("parity") != current_parity:
        return "parity summary does not match current inputs"
    return None


def existing_completed(args: argparse.Namespace, job: campaign_suite.SuiteJob) -> bool:
    if not job.output.exists():
        return False
    error = validate_job_summary(job)
    if error is not None:
        raise FileExistsError(f"refusing incomplete existing output: {job.output}: {error}")
    manifest_error = validate_existing_manifest(args, job)
    if manifest_error is not None:
        raise FileExistsError(f"refusing existing completed output with mismatched controls: {job.output}: {manifest_error}")
    return True


def stop_requested(output: Path) -> bool:
    return (output / "stop.request").exists()


def ensure_no_running_active_job(output: Path) -> None:
    active_path = output / "active_job.json"
    if not active_path.is_file():
        return
    try:
        active = json.loads(active_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"active job file is malformed: {active_path}") from exc
    if isinstance(active, dict) and active.get("status") == "running":
        raise RuntimeError(f"refusing to start while active job is running: {active.get('name')!r}")


def completed_existing_record(index: int, job: campaign_suite.SuiteJob, now: float) -> dict[str, Any]:
    record = running_record(index, job, now)
    record.update({"return_code": 0, "started_unix": None, "ended_unix": now, "status": "completed_existing", "error": None})
    return record


def running_record(index: int, job: campaign_suite.SuiteJob, now: float) -> dict[str, Any]:
    return {
        "name": job.name,
        "index": index,
        "arm": job.arm,
        "seed": job.seed,
        "n_envs": job.n_envs,
        "n_steps": job.n_steps,
        "group_size": job.group_size,
        "output": str(job.output),
        "return_code": None,
        "started_unix": now,
        "ended_unix": None,
        "status": "running",
        "error": None,
    }


def write_suite_report(
    output: Path,
    *,
    status: str,
    args: argparse.Namespace,
    started: float,
    ended: float,
    planned_jobs: int,
    records: list[dict[str, Any]],
) -> None:
    campaign_suite.write_suite_report(
        output,
        status=status,
        args=args,
        started=started,
        ended=ended,
        planned_jobs=planned_jobs,
        records=records,
    )
    report_path = output / "suite_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["completed_jobs"] = sum(
        1 for record in records if record.get("status") in {"completed", "completed_existing"}
    )
    campaign_suite.write_json(report_path, report)


def validate_plan(jobs: list[campaign_suite.SuiteJob]) -> None:
    if len(jobs) != 6:
        raise RuntimeError("graph pairs plan must contain exactly 6 jobs")
    for job in jobs:
        if job.arm not in {"control", "graph"} or job.resident_buffer:
            raise RuntimeError(f"invalid graph pairs job: {job}")
        if (int(job.n_envs), int(job.n_steps), int(job.group_size)) not in ALLOWED_GEOMETRIES:
            raise RuntimeError(f"invalid graph pairs geometry: {job}")
        if job.n_envs * job.n_steps != TRANSITIONS_PER_CYCLE:
            raise RuntimeError(f"job {job.name} does not have {TRANSITIONS_PER_CYCLE} transitions/cycle")


def run_suite(
    args: argparse.Namespace,
    *,
    runner: Callable[[campaign_suite.SuiteJob, list[str], dict[str, str], float], int] = campaign_suite.run_subprocess_job,
    clock: Any = time,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> int:
    jobs = plan_jobs(args)
    validate_plan(jobs)
    if not args.execute:
        return dry_run(args, jobs)

    args.output.mkdir(parents=True, exist_ok=True)
    ensure_no_running_active_job(args.output)
    started = clock.time()
    if float(args.deadline_unix) <= float(started):
        raise ValueError("--deadline-unix must be in the future for --execute")
    preflight_record = preflight(args, disk_usage=disk_usage)
    append_invocation_provenance(args, jobs=jobs, status="execute", created=started, preflight_record=preflight_record)
    records: list[dict[str, Any]] = []
    status = "completed"
    exit_code = 0

    for index, job in enumerate(jobs):
        if existing_completed(args, job):
            records.append(completed_existing_record(index, job, clock.time()))
            write_suite_report(args.output, status="running", args=args, started=started, ended=clock.time(), planned_jobs=len(jobs), records=records)
            continue
        if stop_requested(args.output):
            status = "stop_requested"
            exit_code = 4
            break
        remaining = float(args.deadline_unix) - float(clock.time())
        required = float(args.per_job_timeout) + RESERVE_SECONDS
        if remaining < required:
            status = "deadline_insufficient"
            exit_code = 3
            records.append(
                {
                    "name": job.name,
                    "status": "skipped_deadline",
                    "remaining_seconds": remaining,
                    "required_seconds": required,
                }
            )
            break
        ensure_disk_free(args.output, disk_usage=disk_usage)
        job.output.mkdir(parents=True, exist_ok=False)
        command = command_for_job(args, job)
        env = campaign_suite.whitelisted_env(job=job, args=args)
        write_job_manifest(args, job, command, env)
        record = running_record(index, job, clock.time())
        records.append(record)
        campaign_suite.write_active_job(
            args.output,
            {
                "name": job.name,
                "pid": None,
                "status": "running",
                "started_unix": record["started_unix"],
                "output": str(job.output),
                "log_path": str(job.output / "job.log"),
            },
        )
        write_suite_report(args.output, status="running", args=args, started=started, ended=clock.time(), planned_jobs=len(jobs), records=records)
        return_code = runner(job, command, env, float(args.per_job_timeout))
        error = validate_job_summary(job) if return_code == 0 else f"process exit code {return_code}"
        record.update(
            {
                "return_code": return_code,
                "ended_unix": clock.time(),
                "status": "completed" if error is None else "failed",
                "error": error,
            }
        )
        if error is not None:
            status = "aborted"
            exit_code = 2 if return_code == 0 else 1
            break

    campaign_suite.write_active_job(
        args.output,
        {"name": records[-1]["name"] if records else None, "pid": None, "status": status, "ended_unix": clock.time()},
    )
    write_suite_report(args.output, status=status, args=args, started=started, ended=clock.time(), planned_jobs=len(jobs), records=records)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = validate_args(parser.parse_args(argv))
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
