#!/usr/bin/env python3
"""Evaluate a retention-curriculum checkpoint in exact 20-case development panels."""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from ai2.paths import source_path

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ai2.training.benchmark_native_contact_training import (
    ROOT,
    atomic_json,
    preserve_failed_block,
    require_cuda,
    require_qualification,
    sha256,
    validate_deadline,
    write_inventory,
)


SCHEMA_VERSION = "ai2.retention_evaluation.v1"
TASKS = ("combat", "near", "wide")
TASK_TO_DOMAIN = {"combat": "contact", "near": "nearby", "wide": "wide"}
MODES = ("deterministic", "sampled")
SUPPORTED_EPISODES_PER_MODE = (20, 100)


def require_preserved_checkpoint(checkpoint, ack_path):
    checkpoint = Path(checkpoint)
    inventory_path = checkpoint.parent / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    ack = json.loads(Path(ack_path).read_text())
    if (
        inventory.get("schema_version") != "curriculum-inventory-v1"
        or ack.get("verified") is not True
        or ack.get("block_id") != inventory.get("block_id")
        or ack.get("inventory_sha256") != sha256(inventory_path)
    ):
        raise ValueError("candidate checkpoint requires its hash-matched verified-local preservation ACK")
    matches = [row for row in inventory.get("files", []) if row.get("path") == checkpoint.name]
    if (
        len(matches) != 1
        or checkpoint.is_symlink()
        or not checkpoint.is_file()
        or matches[0].get("bytes") != checkpoint.stat().st_size
        or matches[0].get("sha256") != sha256(checkpoint)
    ):
        raise ValueError("candidate checkpoint does not match the preserved inventory")
    return {
        "checkpoint_sha256": matches[0]["sha256"],
        "inventory_sha256": sha256(inventory_path),
        "preservation_ack_sha256": sha256(ack_path),
    }


def evaluation_jobs(args):
    return [
        {"name": task, "task": task, "domain": TASK_TO_DOMAIN[task], "checkpoint": args.checkpoint}
        for task in TASKS
    ]


def job_output(args, job):
    return Path(args.output) / f'{Path(args.output).name}-{job["name"]}'


def evaluation_command(args, job):
    return [
        sys.executable,
        "-B",
        str(Path(args.reference_root) / "evaluate_combat_curriculum.py"),
        "--checkpoint",
        str(job["checkpoint"]),
        "--frozen-root",
        str(args.frozen_root),
        "--output",
        str(job_output(args, job)),
        "--task",
        job["task"],
        "--episodes",
        str(args.episodes),
        "--deadline-unix",
        str(args.deadline_unix),
    ]


def _episode_mode(episode):
    return episode.get("mode") or episode.get("action_mode") or episode.get("policy_mode")


def _episode_player_slot(episode):
    value = episode.get("player_slot")
    if value is None and isinstance(episode.get("scenario"), dict):
        value = episode["scenario"].get("player_slot")
    if value is None and isinstance(episode.get("case"), dict):
        value = episode["case"].get("player_slot")
    if type(value) is not int or value not in (0, 1):
        raise ValueError("episode rows must include integer player_slot 0 or 1")
    return value


def _episode_success(episode):
    if type(episode.get("success")) is bool:
        return episode["success"]
    result = str(episode.get("result") or episode.get("outcome") or "").lower()
    return result in {"success", "win"}


def _episode_fall(episode):
    if any(episode.get(key) is True for key in ("fall", "fell", "nav_fall", "self_fall")):
        return True
    text = " ".join(str(episode.get(key, "")).lower() for key in ("result", "outcome", "terminal_reason", "terminal_cause", "cause"))
    return "fall" in text


def _mode_episode_rows(task_summary, mode, episodes, *, domain):
    rows = task_summary.get("episodes")
    if rows is None:
        return None
    if not isinstance(rows, list):
        raise ValueError(f"{domain} episode rows must be a list when present")
    selected = [row for row in rows if isinstance(row, dict) and _episode_mode(row) == mode]
    if len(selected) != episodes:
        raise ValueError(f"{domain} {mode} panel must include exactly {episodes} episode rows")
    return selected


def _slot_breakdown(task_summary, mode, episodes, *, domain):
    selected = _mode_episode_rows(task_summary, mode, episodes, domain=domain)
    if selected is None:
        return None
    by_slot = {0: [], 1: []}
    for row in selected:
        by_slot[_episode_player_slot(row)].append(row)
    if episodes % 2:
        raise ValueError(f"{domain} {mode} panel cannot balance an odd episode count")
    per_slot = episodes // 2
    if len(by_slot[0]) != per_slot or len(by_slot[1]) != per_slot:
        raise ValueError(
            f"{domain} {mode} panel must contain exactly {per_slot} slot0 and {per_slot} slot1 episodes"
        )
    breakdown = {}
    for slot, rows in by_slot.items():
        counts = {
            "episodes": len(rows),
            "successes": sum(1 for row in rows if _episode_success(row)),
            "falls": sum(1 for row in rows if _episode_fall(row)),
        }
        if domain == "contact":
            attribution = _derive_contact_attribution({"episodes": rows}, mode, len(rows))
            counts.update(
                {
                    "self_losses": attribution["self_losses"],
                    "opponent_self_losses": attribution["opponent_self_losses"],
                    "timeouts": attribution["timeouts"],
                    "ambiguous": attribution["ambiguous"],
                }
            )
            counts.pop("falls")
        breakdown[str(slot)] = counts
    return breakdown


def _derive_contact_attribution(task_summary, mode, episodes):
    rows = task_summary.get("episodes")
    if not isinstance(rows, list):
        return {"self_losses": 0, "opponent_self_losses": 0, "timeouts": 0, "ambiguous": episodes}
    selected = [row for row in rows if _episode_mode(row) == mode]
    if len(selected) != episodes:
        return {"self_losses": 0, "opponent_self_losses": 0, "timeouts": 0, "ambiguous": episodes}
    counts = {"self_losses": 0, "opponent_self_losses": 0, "timeouts": 0, "ambiguous": 0}
    for row in selected:
        result = str(row.get("result") or row.get("outcome") or "").lower()
        cause = str(row.get("terminal_reason") or row.get("terminal_cause") or row.get("cause") or row.get("terminal") or "").lower()
        diagnoses = " ".join(str(item).lower() for item in row.get("diagnoses", []))
        evidence = " ".join(str(row.get(key, "")).lower() for key in ("evidence", "events", "terminal_evidence"))
        stock_events = row.get("stock_events") or []
        stock_text = " ".join(json.dumps(event, sort_keys=True).lower() for event in stock_events if isinstance(event, dict))
        text = " ".join((result, cause, diagnoses, evidence))
        if "timeout" in text:
            counts["timeouts"] += 1
        elif row.get("own_stock_loss") is True or "own_stock_loss" in text or "self_fall" in text or "self_loss" in text or result == "loss":
            counts["self_losses"] += 1
        elif "unknown" in stock_text and ("opponent_stock_loss" in text or "stock_loss" in stock_text):
            counts["ambiguous"] += 1
        elif "opponent_self" in text:
            counts["opponent_self_losses"] += 1
        elif result in {"unsafe_floor_support", "no_attack"}:
            # These are explicit, attributed lesson failures rather than an
            # unexplained stock result.  They still reduce the success count.
            continue
        elif (
            result == "success"
            and row.get("success") is True
            and row.get("evidence_valid") is True
            and row.get("training_qualified") is True
            and not stock_events
        ):
            continue
        elif result in {"win", "opponent_stock_loss"} and any(token in " ".join((text, stock_text)) for token in ("opponent_ko", "hit_ko", "damage_ko", "agent_hit", "agent")):
            continue
        else:
            counts["ambiguous"] += 1
    return counts


def _summary_counts(summary, *, domain, expected_episodes=20, task_summary=None, mode=None):
    data = summary.get("summary", summary)
    episodes = data.get("episodes")
    successes = data.get("successes", data.get("wins"))
    if successes is None and data.get("success_rate") is not None and episodes is not None:
        successes = round(float(data["success_rate"]) * int(episodes))
    panel = {
        "episodes": episodes,
        "successes": successes,
        "falls": data.get("falls", data.get("nav_falls", 0)),
    }
    if domain == "contact":
        unknown_attribution = data.get("terminal_attribution") == "unknown" or data.get("attribution") == "unknown"
        explicit = all(key in data for key in ("self_losses", "opponent_self_losses", "timeouts", "ambiguous"))
        derived = (
            {
                "self_losses": data.get("self_losses"),
                "opponent_self_losses": data.get("opponent_self_losses"),
                "timeouts": data.get("timeouts"),
                "ambiguous": data.get("ambiguous"),
            }
            if explicit
            else _derive_contact_attribution(task_summary or {}, mode, panel["episodes"])
        )
        if unknown_attribution and not explicit:
            derived["ambiguous"] = panel["episodes"]
        panel.update(
            {
                "self_losses": derived["self_losses"],
                "opponent_self_losses": derived["opponent_self_losses"],
                "timeouts": derived["timeouts"],
                "ambiguous": derived["ambiguous"],
            }
        )
    for key, value in panel.items():
        if type(value) is not int:
            raise ValueError(f"{domain} panel summary field {key} must be an integer")
    if panel["episodes"] != expected_episodes:
        raise ValueError(f"{domain} panel must contain exactly {expected_episodes} episodes")
    breakdown = _slot_breakdown(task_summary or {}, mode, panel["episodes"], domain=domain)
    if breakdown is not None:
        panel["by_player_slot"] = breakdown
    return panel


def build_result(
    summaries,
    *,
    candidate,
    qualification,
    qualification_sha256,
    wall_seconds,
    episodes_per_mode=20,
):
    if episodes_per_mode not in SUPPORTED_EPISODES_PER_MODE:
        raise ValueError("retention evaluation supports only 20-case development or 100-case confirmation panels")
    panels = {}
    slot_breakdown = {}
    raw = {}
    for task in TASKS:
        domain = TASK_TO_DOMAIN[task]
        task_summary = summaries[task]
        raw[domain] = task_summary
        modes = task_summary.get("modes", task_summary.get("panels"))
        if not isinstance(modes, dict):
            raise ValueError(f"{task} summary must include deterministic and sampled modes")
        for mode in MODES:
            if mode not in modes:
                raise ValueError(f"{task} summary missing {mode} mode")
            panels.setdefault(domain, {})[mode] = _summary_counts(
                modes[mode],
                domain=domain,
                expected_episodes=episodes_per_mode,
                task_summary=task_summary,
                mode=mode,
            )
            if "by_player_slot" in panels[domain][mode]:
                slot_breakdown.setdefault(domain, {})[mode] = panels[domain][mode]["by_player_slot"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "evaluated_not_promoted",
        "mastery_established": False,
        "episodes_per_mode": episodes_per_mode,
        "candidate": candidate,
        "qualification": qualification,
        "qualification_sha256": qualification_sha256,
        "panels": panels,
        "slot_breakdown": slot_breakdown,
        "raw_panels": raw,
        "wall_seconds": wall_seconds,
        "finished_unix": time.time(),
    }


def evaluate(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("evaluation outputs must be new and immutable")
    if args.episodes not in SUPPORTED_EPISODES_PER_MODE:
        raise ValueError("retention evaluation supports only exact 20-case development or 100-case confirmation panels")
    validate_deadline(args.deadline_unix)
    qualification = require_qualification(args.qualification, args.reference_root, args.frozen_root)
    preserved = require_preserved_checkpoint(args.checkpoint, args.verified_ack)
    if preserved["checkpoint_sha256"] != args.checkpoint_sha256:
        raise ValueError("checkpoint sha256 does not match verified preservation")
    require_cuda("cuda")
    # Original evaluator imports curriculum_ppo. Any reference copy must be the
    # same frozen policy module the adapter used, not an unqualified replacement.
    reference_policy = source_path(Path(args.reference_root), "curriculum_ppo.py")
    if reference_policy.exists() and sha256(reference_policy) != sha256(source_path(ROOT, "curriculum_ppo.py")):
        raise ValueError("reference evaluation policy source differs from the qualified adapter source")
    output.mkdir(parents=True, exist_ok=False)
    jobs = evaluation_jobs(args)
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "candidate": preserved,
            "qualification_sha256": sha256(args.qualification),
            "episodes_per_mode": args.episodes,
            "player_slot_balance": {
                "required_when_episode_rows_present": True,
                "slot0": args.episodes // 2,
                "slot1": args.episodes // 2,
            },
            "panels": [{"task": job["task"], "domain": job["domain"], "output": str(job_output(args, job))} for job in jobs],
            "commands": [evaluation_command(args, job) for job in jobs],
            "started_unix": time.time(),
        },
    )
    environment = {
        **os.environ,
        "WANDB_MODE": "offline",
        "OMP_NUM_THREADS": "1",
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
        "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    summaries = {}
    started = time.perf_counter()
    try:
        for job in jobs:
            validate_deadline(args.deadline_unix)
            with (output / f'{job["name"]}.log').open("x", encoding="utf8") as log:
                completed = subprocess.run(
                    evaluation_command(args, job),
                    env=environment,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=max(1.0, args.deadline_unix - time.time() - 60),
                )
            if completed.returncode:
                raise RuntimeError(f'original-game {job["name"]} evaluation failed: exit {completed.returncode}')
            summaries[job["task"]] = json.loads((job_output(args, job) / "summary.json").read_text())
        result = build_result(
            summaries,
            candidate=preserved,
            qualification=qualification,
            qualification_sha256=sha256(args.qualification),
            wall_seconds=time.perf_counter() - started,
            episodes_per_mode=args.episodes,
        )
        atomic_json(output / "result.json", result)
        write_inventory(output, block_id=output.name)
        return result
    except BaseException as error:
        preserve_failed_block(output, error, completed_panels=list(summaries))
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "verified-ack", "frozen-root", "reference-root", "qualification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
