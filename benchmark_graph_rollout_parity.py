#!/usr/bin/env python3
"""One-update parity gate for opt-in rollout policy CUDA graph integration.

This benchmark-only verifier runs the frozen ``train_job.run_job`` twice from
the same checkpoint/seed: once with vanilla PPO forward and once with the
rollout-policy graph context. It records rollout buffer, policy, optimizer,
counters, RNG, and final checkpoint digests, then fails closed unless the two
arms are byte-identical on all required digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import benchmark_campaign


REQUIRED_RECEIPT_FIELDS = (
    "status",
    "input_checkpoint_sha256",
    "checkpoint_container_sha256",
    "rollout_buffer_sha256",
    "pre_update_policy_sha256",
    "pre_update_optimizer_sha256",
    "policy_sha256",
    "optimizer_sha256",
    "pre_update_rng_sha256",
    "post_update_rng_sha256",
    "num_timesteps",
    "expected_steps",
    "expected_epochs",
    "expected_optimizer_steps",
    "actual_steps",
    "collect_calls",
    "train_calls",
    "n_updates_before",
    "n_updates_after",
    "n_updates_delta",
    "optimizer_step_calls",
    "actual_device",
    "train_result",
    "train_metadata",
    "provenance_before",
    "provenance_after",
)

GATED_PARITY_FIELDS = (
    "input_checkpoint_sha256",
    "rollout_buffer_sha256",
    "pre_update_policy_sha256",
    "pre_update_optimizer_sha256",
    "policy_sha256",
    "optimizer_sha256",
    "pre_update_rng_sha256",
    "post_update_rng_sha256",
    "num_timesteps",
    "expected_steps",
    "expected_epochs",
    "expected_optimizer_steps",
    "actual_steps",
    "collect_calls",
    "train_calls",
    "n_updates_before",
    "n_updates_after",
    "n_updates_delta",
    "optimizer_step_calls",
)

RECORDED_NOT_GATED_FIELDS = ("checkpoint_container_sha256",)

REQUIRED_INPUT_HASH_FIELDS = (
    "source_train",
    "config",
    "train_job",
    "checkpoint",
    "verifier",
    "campaign",
    "grouped_env",
    "rollout_policy_graph",
    "policy_compile",
    "policy_cuda_graph",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(benchmark_campaign.json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return benchmark_campaign.sha256_file(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def update_digest_frame(digest: Any, tag: str, payload: bytes = b"") -> None:
    digest.update(tag.encode("utf-8"))
    digest.update(b":")
    digest.update(str(len(payload)).encode("ascii"))
    digest.update(b":")
    digest.update(payload)
    digest.update(b";")


def canonical_dict_key(key: Any) -> tuple[str, bytes]:
    if isinstance(key, str):
        return ("str", key.encode("utf-8"))
    if isinstance(key, int) and not isinstance(key, bool):
        return ("int", str(key).encode("ascii"))
    raise TypeError(f"unsupported digest key type: {type(key).__name__}")


def update_digest_for_value(digest: Any, value: Any) -> None:
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    if hasattr(value, "tobytes"):
        update_digest_frame(digest, "array_shape", json.dumps(list(getattr(value, "shape", ()))).encode("utf-8"))
        update_digest_frame(digest, "array_dtype", str(getattr(value, "dtype", None)).encode("utf-8"))
        update_digest_frame(digest, "array_bytes", value.tobytes())
    elif isinstance(value, dict):
        update_digest_frame(digest, "dict_len", str(len(value)).encode("ascii"))
        for key in sorted(value, key=canonical_dict_key):
            key_type, key_bytes = canonical_dict_key(key)
            update_digest_frame(digest, f"dict_key_{key_type}", key_bytes)
            update_digest_for_value(digest, value[key])
    elif isinstance(value, tuple):
        update_digest_frame(digest, "tuple_len", str(len(value)).encode("ascii"))
        for item in value:
            update_digest_for_value(digest, item)
    elif isinstance(value, list):
        update_digest_frame(digest, "list_len", str(len(value)).encode("ascii"))
        for item in value:
            update_digest_for_value(digest, item)
    elif value is None:
        update_digest_frame(digest, "none")
    elif isinstance(value, bool):
        update_digest_frame(digest, "bool", b"1" if value else b"0")
    elif isinstance(value, int):
        update_digest_frame(digest, "int", str(value).encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("unsupported non-finite float digest value")
        update_digest_frame(digest, "float", repr(value).encode("ascii"))
    elif isinstance(value, str):
        update_digest_frame(digest, "str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        update_digest_frame(digest, "bytes", value)
    else:
        raise TypeError(f"unsupported digest value type: {type(value).__name__}")


def digest_any(value: Any) -> str:
    digest = hashlib.sha256()
    update_digest_for_value(digest, value)
    return digest.hexdigest()


def rollout_buffer_digest(buffer: Any) -> str:
    fields = (
        "observations",
        "actions",
        "rewards",
        "episode_starts",
        "values",
        "log_probs",
        "advantages",
        "returns",
        "pos",
        "full",
        "buffer_size",
        "n_envs",
    )
    payload = {field: getattr(buffer, field, None) for field in fields}
    return digest_any(payload)


def policy_digest(agent: Any) -> str:
    return digest_any(agent.policy.state_dict())


def optimizer_digest(agent: Any) -> str:
    return digest_any(agent.policy.optimizer.state_dict())


def rng_digest(torch: Any, device: str) -> dict[str, str]:
    record = {"torch_cpu": digest_any(torch.random.get_rng_state())}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        index = 0 if torch.device(device).index is None else int(torch.device(device).index)
        record["torch_cuda"] = digest_any(torch.cuda.get_rng_state(index))
    return record


def file_provenance(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    record: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        record["sha256"] = sha256_file(resolved)
        record["size"] = resolved.stat().st_size
    return record


def module_file(name: str) -> Path:
    module = sys.modules.get(name)
    path = getattr(module, "__file__", None)
    if path:
        return Path(path)
    return Path(__file__).resolve().with_name(f"{name}.py")


def input_file_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    verifier = Path(__file__).resolve()
    repo_root = verifier.parent
    return {
        "source_train": file_provenance(Path(args.source) / "train.py"),
        "config": file_provenance(Path(args.config)),
        "train_job": file_provenance(Path(args.train_job)),
        "checkpoint": file_provenance(Path(args.checkpoint)),
        "verifier": file_provenance(verifier),
        "campaign": file_provenance(Path(getattr(benchmark_campaign, "__file__", repo_root / "benchmark_campaign.py"))),
        "grouped_env": file_provenance(Path(args.grouped_env_path)),
        "rollout_policy_graph": file_provenance(Path(args.rollout_policy_graph_path)),
        "policy_compile": file_provenance(repo_root / "benchmark_policy_compile.py"),
        "policy_cuda_graph": file_provenance(repo_root / "benchmark_policy_cuda_graph.py"),
    }


def expected_steps(args: argparse.Namespace) -> int:
    return int(args.n_envs) * int(args.n_steps)


def expected_optimizer_steps(args: argparse.Namespace) -> int:
    return int(args.epochs) * math.ceil(expected_steps(args) / int(args.batch_size))


def required_input_hash(record: dict[str, Any], field: str, *, label: str) -> str:
    item = record.get(field)
    if not isinstance(item, dict):
        raise RuntimeError(f"{label} missing input file record: {field}")
    if item.get("exists") is not True:
        raise RuntimeError(f"{label} missing required input file: {field}")
    digest = item.get("sha256")
    if not isinstance(digest, str) or not digest:
        raise RuntimeError(f"{label} missing required input sha256: {field}")
    return digest


def validate_input_hashes_unchanged(receipt: dict[str, Any], *, label: str) -> None:
    before = receipt.get("provenance_before")
    after = receipt.get("provenance_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise RuntimeError(f"{label} missing provenance before/after snapshots")
    for field in REQUIRED_INPUT_HASH_FIELDS:
        before_sha = required_input_hash(before, field, label=f"{label} before")
        after_sha = required_input_hash(after, field, label=f"{label} after")
        if before_sha != after_sha:
            raise RuntimeError(f"required input {field} changed during {label}: {before_sha} != {after_sha}")


def validate_shared_input_hashes(control: dict[str, Any], graph: dict[str, Any]) -> None:
    control_before = control.get("provenance_before")
    graph_before = graph.get("provenance_before")
    if not isinstance(control_before, dict) or not isinstance(graph_before, dict):
        raise RuntimeError("missing shared input provenance")
    for field in REQUIRED_INPUT_HASH_FIELDS:
        control_sha = required_input_hash(control_before, field, label="control before")
        graph_sha = required_input_hash(graph_before, field, label="graph before")
        if control_sha != graph_sha:
            raise RuntimeError(f"shared input hash mismatch for {field}: {control_sha} != {graph_sha}")


def read_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def actual_device_record(agent: Any, torch: Any, requested: str) -> dict[str, Any]:
    policy = getattr(agent, "policy", None)
    parameters = list(policy.parameters()) if policy is not None and hasattr(policy, "parameters") else []
    parameter_devices = sorted({str(param.device) for param in parameters})
    return {
        "requested": str(requested),
        "cuda_available": bool(torch.cuda.is_available()),
        "policy_device": str(getattr(policy, "device", None)),
        "parameter_devices": parameter_devices,
        "parameters_all_cuda": bool(parameters) and all(str(param.device).startswith("cuda") for param in parameters),
    }


def build_campaign_argv(args: argparse.Namespace, label: str, *, use_graph: bool) -> list[str]:
    output = Path(args.output) / label
    argv = [
        "--config",
        str(Path(args.config)),
        "--checkpoint",
        str(Path(args.checkpoint)),
        "--source",
        str(Path(args.source)),
        "--train-job",
        str(Path(args.train_job)),
        "--output",
        str(output),
        "--n-envs",
        str(int(args.n_envs)),
        "--n-steps",
        str(int(args.n_steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--epochs",
        str(int(args.epochs)),
        "--group-size",
        str(int(args.group_size)),
        "--grouped-env-path",
        str(Path(args.grouped_env_path)),
        "--seed",
        str(int(args.seed)),
        "--cycles",
        "1",
        "--warmup",
        "0",
        "--device",
        str(args.device),
        "--phase",
        str(args.phase),
    ]
    if args.maxseconds is not None:
        argv.extend(["--maxseconds", str(float(args.maxseconds))])
    if use_graph:
        argv.extend(["--rollout-policy-graph", "--rollout-policy-graph-path", str(Path(args.rollout_policy_graph_path))])
    return argv


def make_parity_ppo_class(base_cls: type, *, receipt: dict[str, Any], use_graph: bool, graph_module: Any, args: argparse.Namespace) -> type:
    class ParityPPO(base_cls):
        __campaign_base_cls__ = base_cls

        def _excluded_save_params(self):
            return super()._excluded_save_params() + ["parity_receipt"]

        def collect_rollouts(self, *collect_args: Any, **collect_kwargs: Any) -> Any:
            receipt["collect_calls"] = int(receipt.get("collect_calls", 0)) + 1
            if use_graph:
                context = graph_module.policy_forward_graph_context(
                    self.policy,
                    expected_batch_size=int(args.n_envs),
                    capture_warmup=int(args.rollout_policy_graph_capture_warmup),
                    tolerance=float(args.rollout_policy_graph_tolerance),
                )
            else:
                class NullContext:
                    def __enter__(self_inner):
                        return None
                    def __exit__(self_inner, exc_type, exc, tb):
                        return False
                context = NullContext()
            with context as adapter:
                ok = super().collect_rollouts(*collect_args, **collect_kwargs)
            if not ok:
                raise RuntimeError("collect_rollouts did not complete")
            receipt["rollout_buffer_sha256"] = rollout_buffer_digest(self.rollout_buffer)
            receipt["pre_update_policy_sha256"] = policy_digest(self)
            receipt["pre_update_optimizer_sha256"] = optimizer_digest(self)
            receipt["pre_update_rng_sha256"] = rng_digest(sys.modules["torch"], str(args.device))
            if adapter is not None and hasattr(adapter, "summary"):
                receipt["rollout_policy_graph"] = adapter.summary()
            return ok

        def train(self) -> Any:
            receipt["train_calls"] = int(receipt.get("train_calls", 0)) + 1
            receipt["n_updates_before"] = int(getattr(self, "_n_updates", 0))
            optimizer = self.policy.optimizer
            original_step = optimizer.step
            step_calls = {"count": 0}

            def counted_step(*step_args: Any, **step_kwargs: Any) -> Any:
                step_calls["count"] += 1
                return original_step(*step_args, **step_kwargs)

            optimizer.step = counted_step
            try:
                result = super().train()
            finally:
                optimizer.step = original_step
            receipt["optimizer_step_calls"] = int(receipt.get("optimizer_step_calls", 0)) + int(step_calls["count"])
            receipt["n_updates_after"] = int(getattr(self, "_n_updates", 0))
            receipt["n_updates_delta"] = int(receipt["n_updates_after"]) - int(receipt["n_updates_before"])
            receipt["policy_sha256"] = policy_digest(self)
            receipt["optimizer_sha256"] = optimizer_digest(self)
            receipt["post_update_rng_sha256"] = rng_digest(sys.modules["torch"], str(args.device))
            receipt["num_timesteps"] = int(getattr(self, "num_timesteps", 0))
            receipt["actual_device"] = actual_device_record(self, sys.modules["torch"], str(args.device))
            return result

    ParityPPO.__name__ = "ParityPPO"
    return ParityPPO


def run_arm(args: argparse.Namespace, label: str, *, use_graph: bool) -> dict[str, Any]:
    output = Path(args.output).resolve() / label
    output.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "label": label,
        "status": "running",
        "use_rollout_policy_graph": bool(use_graph),
        "started_unix": time.time(),
        "input_checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "expected_steps": expected_steps(args),
        "expected_epochs": int(args.epochs),
        "expected_optimizer_steps": expected_optimizer_steps(args),
        "collect_calls": 0,
        "train_calls": 0,
        "provenance_before": {
            **input_file_snapshot(args),
            "controls": {
                "n_envs": int(args.n_envs),
                "n_steps": int(args.n_steps),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "seed": int(args.seed),
                "phase": str(args.phase),
                "device": str(args.device),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "maxseconds": args.maxseconds,
                "group_size": int(args.group_size),
                "grouped_env_path": str(Path(args.grouped_env_path).resolve()),
                "rollout_policy_graph_capture_warmup": int(args.rollout_policy_graph_capture_warmup),
                "rollout_policy_graph_tolerance": float(args.rollout_policy_graph_tolerance),
            },
        },
    }
    graph_module = None
    if use_graph:
        graph_module = benchmark_campaign.load_rollout_policy_graph_module(Path(args.rollout_policy_graph_path))
    train_job = benchmark_campaign.import_train_job_module(Path(args.train_job))
    original_ppo = train_job.PPO
    grouped_cleanup = None
    train_job.PPO = make_parity_ppo_class(original_ppo, receipt=receipt, use_graph=use_graph, graph_module=graph_module, args=args)
    previous_cwd = Path.cwd()
    try:
        grouped_cleanup = benchmark_campaign.patch_grouped_vec_env(train_job, args)
        os.chdir(Path(args.source).resolve())
        train_job_args = benchmark_campaign.build_train_job_args(args, benchmark_campaign.plan_run(
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup=0,
            cycles=1,
        ))
        train_job_args.output = str(output)
        train_job.run_job(train_job_args)
        receipt["train_result"] = read_json_file(output / "result.json")
        receipt["train_metadata"] = read_json_file(output / "metadata.json")
        receipt["actual_steps"] = int(receipt["train_result"].get("actual_steps", -1))
        final_zip = output / "final.zip"
        if not final_zip.is_file():
            raise RuntimeError(f"missing final checkpoint: {final_zip}")
        receipt["checkpoint_container_sha256"] = sha256_file(final_zip)
        receipt["final_checkpoint"] = str(final_zip)
        receipt["status"] = "completed"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
    finally:
        os.chdir(previous_cwd)
        train_job.PPO = original_ppo
        if grouped_cleanup is not None:
            grouped_cleanup()
        receipt["ended_unix"] = time.time()
        receipt["provenance_after"] = {
            **input_file_snapshot(args),
            "final_checkpoint": file_provenance(output / "final.zip"),
            "result": file_provenance(output / "result.json"),
            "metadata": file_provenance(output / "metadata.json"),
        }
        write_json(output / "graph_rollout_parity_receipt.json", receipt)
    if receipt["status"] != "completed":
        raise RuntimeError(f"{label} parity arm failed: {receipt.get('error')}")
    return receipt


def read_parity_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_RECEIPT_FIELDS if field not in payload]
    if missing:
        raise RuntimeError(f"missing parity receipt fields: {', '.join(missing)}")
    if payload["status"] != "completed":
        raise RuntimeError(f"parity receipt status is not completed: {payload['status']}")
    return payload


def require_fields(record: dict[str, Any], fields: tuple[str, ...], *, label: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise RuntimeError(f"{label} missing required parity fields: {', '.join(missing)}")


def require_completed_status(receipt: dict[str, Any], *, label: str) -> None:
    if receipt.get("status") != "completed":
        raise RuntimeError(f"{label} receipt status is not completed: {receipt.get('status')}")


def validate_actual_device(receipt: dict[str, Any], *, label: str) -> None:
    device = receipt.get("actual_device")
    if not isinstance(device, dict):
        raise RuntimeError(f"{label} missing actual_device attestation")
    if not str(device.get("requested", "")).startswith("cuda"):
        raise RuntimeError(f"{label} requested device is not CUDA: {device.get('requested')}")
    if not str(device.get("policy_device", "")).startswith("cuda"):
        raise RuntimeError(f"{label} policy_device is not CUDA: {device.get('policy_device')}")
    if device.get("parameters_all_cuda") is not True:
        raise RuntimeError(f"{label} parameters_all_cuda attestation failed")
    if device.get("cuda_available") is not True:
        raise RuntimeError(f"{label} cuda_available attestation failed")


def validate_training_receipt(
    receipt: dict[str, Any],
    *,
    expected_steps: int | None = None,
    expected_epochs: int | None = None,
    expected_optimizer_steps: int | None = None,
    label: str = "receipt",
) -> None:
    require_fields(receipt, REQUIRED_RECEIPT_FIELDS, label=label)
    require_completed_status(receipt, label=label)
    train_result = receipt.get("train_result")
    if not isinstance(train_result, dict):
        raise RuntimeError(f"{label} train_result must be an object")
    if train_result.get("status") != "completed":
        raise RuntimeError(f"{label} train_result status is not completed: {train_result.get('status')}")
    if train_result.get("stop_reason") is not None:
        raise RuntimeError(f"{label} train_result stop_reason is set: {train_result.get('stop_reason')}")
    if train_result.get("error") is not None:
        raise RuntimeError(f"{label} train_result error is set: {train_result.get('error')}")

    if expected_steps is None:
        expected_steps = int(receipt["expected_steps"])
    if expected_epochs is None:
        expected_epochs = int(receipt["expected_epochs"])
    if expected_optimizer_steps is None:
        expected_optimizer_steps = int(receipt["expected_optimizer_steps"])

    if int(receipt.get("actual_steps", -1)) != int(expected_steps):
        raise RuntimeError(f"{label} actual_steps mismatch: {receipt.get('actual_steps')} != {expected_steps}")
    if int(train_result.get("actual_steps", -1)) != int(expected_steps):
        raise RuntimeError(f"{label} train_result actual_steps mismatch: {train_result.get('actual_steps')} != {expected_steps}")
    starting_steps = int(train_result.get("starting_steps", -1))
    ending_steps = int(train_result.get("ending_steps", -1))
    if ending_steps - starting_steps != int(expected_steps):
        raise RuntimeError(f"{label} ending_steps - starting_steps mismatch: {ending_steps} - {starting_steps} != {expected_steps}")
    if int(receipt.get("num_timesteps", -1)) != ending_steps:
        raise RuntimeError(f"{label} num_timesteps mismatch: {receipt.get('num_timesteps')} != ending_steps {ending_steps}")
    if int(receipt.get("collect_calls", -1)) != 1:
        raise RuntimeError(f"{label} collect_calls mismatch: {receipt.get('collect_calls')} != 1")
    if int(receipt.get("train_calls", -1)) != 1:
        raise RuntimeError(f"{label} train_calls mismatch: {receipt.get('train_calls')} != 1")
    if int(receipt.get("n_updates_delta", -1)) != int(expected_epochs):
        raise RuntimeError(f"{label} n_updates_delta mismatch: {receipt.get('n_updates_delta')} != {expected_epochs}")
    if int(receipt.get("optimizer_step_calls", -1)) != int(expected_optimizer_steps):
        raise RuntimeError(
            f"{label} optimizer_step_calls mismatch: {receipt.get('optimizer_step_calls')} != {expected_optimizer_steps}"
        )
    if int(receipt.get("n_updates_after", -1)) - int(receipt.get("n_updates_before", -1)) != int(expected_epochs):
        raise RuntimeError(f"{label} _n_updates before/after mismatch")
    if not isinstance(receipt.get("train_metadata"), dict):
        raise RuntimeError(f"{label} train_metadata must be an object")
    controls = receipt.get("provenance_before", {}).get("controls", {})
    if not isinstance(controls, dict):
        raise RuntimeError(f"{label} provenance_before.controls must be an object")
    metadata = receipt["train_metadata"]
    for field in ("device", "seed", "phase", "n_envs", "checkpoint"):
        if field in controls and metadata.get(field) != controls[field]:
            raise RuntimeError(f"{label} metadata {field} mismatch: {metadata.get(field)} != {controls[field]}")
    for field in ("device", "seed", "phase", "n_envs", "checkpoint", "ppo_controls"):
        if field not in metadata:
            raise RuntimeError(f"{label} metadata missing required field: {field}")
    ppo_controls = metadata.get("ppo_controls")
    if not isinstance(ppo_controls, dict):
        raise RuntimeError(f"{label} metadata ppo_controls must be an object")
    ppo_expected = {
        "n_steps": controls.get("n_steps"),
        "batch_size": controls.get("batch_size"),
        "n_epochs": controls.get("epochs"),
    }
    for field, expected in ppo_expected.items():
        if field not in ppo_controls:
            raise RuntimeError(f"{label} metadata ppo_controls missing required field: {field}")
        if expected is not None and int(ppo_controls[field]) != int(expected):
            raise RuntimeError(f"{label} metadata ppo_controls.{field} mismatch")
    if ppo_controls.get("reset_num_timesteps") is not False:
        raise RuntimeError(f"{label} metadata ppo_controls.reset_num_timesteps must be false for resumed checkpoint")
    validate_actual_device(receipt, label=label)
    validate_input_hashes_unchanged(receipt, label=label)


def validate_tensor_report(report: Any, *, label: str) -> None:
    if not isinstance(report, dict):
        raise RuntimeError(f"{label} tensor report must be an object")
    if "max_normalized_error" not in report:
        raise RuntimeError(f"{label} missing max_normalized_error")
    error = float(report["max_normalized_error"])
    if not math.isfinite(error) or error > 1.0:
        raise RuntimeError(f"{label} tensor parity failed: {error}")


def validate_helper_report(report: Any, *, label: str) -> None:
    if not isinstance(report, dict):
        raise RuntimeError(f"{label} helper report must be an object")
    required = (
        "deterministic_zero_eps_vs_policy",
        "stochastic_action_evaluate_actions",
        "permutation_equivariance",
        "fresh_epsilon_action_diff",
        "max_abs_diff",
        "base_atol",
        "relative_tolerance_dtype_eps",
    )
    for field in required:
        if field not in report:
            raise RuntimeError(f"{label} missing helper report field: {field}")
    for section in ("deterministic_zero_eps_vs_policy", "stochastic_action_evaluate_actions", "permutation_equivariance"):
        reports = report[section]
        if not isinstance(reports, dict) or not reports:
            raise RuntimeError(f"{label}.{section} must contain tensor reports")
        for tensor_name, tensor_report in reports.items():
            validate_tensor_report(tensor_report, label=f"{label}.{section}.{tensor_name}")
    for field in ("fresh_epsilon_action_diff", "max_abs_diff", "base_atol", "relative_tolerance_dtype_eps"):
        value = float(report[field])
        if not math.isfinite(value):
            raise RuntimeError(f"{label}.{field} must be finite")


def validate_graph_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("use_rollout_policy_graph") is not True:
        raise RuntimeError("graph receipt did not enable rollout policy graph")
    graph = receipt.get("rollout_policy_graph")
    if not isinstance(graph, dict):
        raise RuntimeError("graph receipt missing rollout_policy_graph summary")
    if int(graph.get("captures", 0)) <= 0:
        raise RuntimeError(f"graph rollout captures must be positive: {graph.get('captures')}")
    if int(graph.get("replays", 0)) <= 0:
        raise RuntimeError(f"graph rollout replays must be positive: {graph.get('replays')}")
    preflight = graph.get("preflight")
    if not isinstance(preflight, dict):
        raise RuntimeError("graph preflight receipt is required")
    if preflight.get("status") != "passed":
        raise RuntimeError(f"graph preflight status is not passed: {preflight.get('status')}")
    repeats = int(preflight.get("stochastic_repeats", 0))
    if repeats <= 0:
        raise RuntimeError(f"graph preflight stochastic_repeats must be positive: {preflight.get('stochastic_repeats')}")
    if preflight.get("setup_rng_restored") is not True:
        raise RuntimeError("graph preflight setup_rng_restored must be true")
    stochastic = preflight.get("stochastic_exact")
    if not isinstance(stochastic, list) or len(stochastic) != repeats:
        raise RuntimeError("graph preflight stochastic_exact records must match stochastic_repeats")
    for index, record in enumerate(stochastic):
        if not isinstance(record, dict) or record.get("passed") is not True:
            raise RuntimeError(f"graph preflight stochastic_exact[{index}] did not pass")
    deterministic = preflight.get("deterministic_no_rng")
    if not isinstance(deterministic, dict) or deterministic.get("passed") is not True:
        raise RuntimeError("graph preflight deterministic_no_rng did not pass")
    for field in ("input_spec", "eps_spec"):
        spec = graph.get(field)
        if not isinstance(spec, dict) or not str(spec.get("device", "")).startswith("cuda"):
            raise RuntimeError(f"graph {field} must attest CUDA device")
    for field in ("helper_direct_sb3", "graph_direct_sb3"):
        report = graph.get(field)
        validate_helper_report(report, label=f"graph {field}")


def compare_parity(control: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    expected_steps_value = int(control.get("expected_steps", graph.get("expected_steps", -1)))
    expected_epochs_value = int(control.get("expected_epochs", graph.get("expected_epochs", -1)))
    expected_optimizer_steps_value = int(
        control.get("expected_optimizer_steps", graph.get("expected_optimizer_steps", -1))
    )
    validate_training_receipt(
        control,
        expected_steps=expected_steps_value,
        expected_epochs=expected_epochs_value,
        expected_optimizer_steps=expected_optimizer_steps_value,
        label="control",
    )
    validate_training_receipt(
        graph,
        expected_steps=expected_steps_value,
        expected_epochs=expected_epochs_value,
        expected_optimizer_steps=expected_optimizer_steps_value,
        label="graph",
    )
    validate_graph_receipt(graph)
    validate_shared_input_hashes(control, graph)
    control_controls = control.get("provenance_before", {}).get("controls")
    graph_controls = graph.get("provenance_before", {}).get("controls")
    if control_controls != graph_controls:
        raise RuntimeError("graph rollout parity controls mismatch")
    fields = GATED_PARITY_FIELDS
    mismatches = []
    for field in fields:
        if control[field] != graph[field]:
            mismatches.append(field)
    if mismatches:
        raise RuntimeError(f"graph rollout parity mismatch: {', '.join(mismatches)}")
    return {
        "byte_parity": True,
        "matched_fields": list(fields),
        "recorded_not_gated": list(RECORDED_NOT_GATED_FIELDS),
        "expected_steps": expected_steps_value,
        "expected_epochs": expected_epochs_value,
        "expected_optimizer_steps": expected_optimizer_steps_value,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if Path(args.output).exists():
        raise RuntimeError(f"--output already exists; refusing to overwrite parity artifacts: {args.output}")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    control = run_arm(args, "control", use_graph=False)
    graph = run_arm(args, "graph", use_graph=True)
    parity = compare_parity(control, graph)
    summary = {
        "status": "completed",
        "diagnostic": "graph_rollout_one_update_parity",
        "control": control,
        "graph": graph,
        "parity": parity,
    }
    write_json(Path(args.output) / "graph_rollout_parity_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=benchmark_campaign.DEFAULT_SOURCE)
    parser.add_argument("--train-job", type=Path, default=benchmark_campaign.DEFAULT_TRAIN_JOB)
    parser.add_argument("--config", type=Path, default=benchmark_campaign.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=benchmark_campaign.DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--phase", default="0b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maxseconds", type=float, default=180.0)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--opponents")
    parser.add_argument("--mode", default="baseline")
    parser.add_argument("--profile-timing", default="natural")
    parser.add_argument("--resident-buffer", action="store_true")
    parser.add_argument("--rollout-cache-path", type=Path)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--grouped-env-path", type=Path)
    parser.add_argument("--rollout-policy-graph-path", type=Path, default=benchmark_campaign.DEFAULT_ROLLOUT_POLICY_GRAPH)
    parser.add_argument("--rollout-policy-graph-capture-warmup", type=int, default=3)
    parser.add_argument("--rollout-policy-graph-tolerance", type=float, default=1e-5)
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if not Path(args.output).is_absolute():
        raise SystemExit("--output must be absolute")
    if not str(args.device).startswith("cuda"):
        raise SystemExit("graph rollout parity is CUDA-only")
    if getattr(args, "grouped_env_path", None) is None:
        args.grouped_env_path = Path(args.source) / "grouped_vec_env.py"
    for name in ("source", "train_job", "config", "checkpoint", "output", "grouped_env_path", "rollout_policy_graph_path"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    return args


def main(argv: list[str] | None = None) -> int:
    args = validate_args(build_parser().parse_args(argv))
    summary = run(args)
    print(json.dumps({"status": summary["status"], "summary": str(Path(args.output) / "graph_rollout_parity_summary.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
