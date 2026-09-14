#!/usr/bin/env python3
"""Read-only synthesis of one nine-lineage native capacity training round.

Print JSON to stdout. No checkpoint loading, third-party imports, or remote calls.
Episode outcomes are checked against panel summaries; training seeds, not games,
are the replication unit. This is descriptive reporting, not a significance test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


ARCHITECTURES = {(512, 512, 256): 1053973, (1024, 1024, 1024): 4736021,
                 (1536, 1536, 1536): 10249749}
SEEDS = (145, 146, 147)
PANELS = tuple(f"{phase}-{mode}" for phase in ("0a", "0b")
               for mode in ("deterministic", "sampled"))
CONTROLS = {"n_envs": 64, "n_steps": 512, "batch_size": 8192, "epochs": 8}
ACCEPTED = {"train": {"completed", "skipped_completed"},
            "eval": {"completed", "skipped_existing"}}
TELEMETRY_SCOPE = ("WHOLE training-command window [started_unix, ended_unix), "
                   "including startup, any warmup, collection, updates, and checkpointing; "
                   "GPU busy percent is not FLOPS utilization or compute occupancy. "
                   "Remote sample time precedes nvidia-smi; boundary attribution is approximate.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"invalid {name}")
    return value


def integer(value, name):
    require(type(value) is int and value >= 0, f"invalid integer {name}")
    return value


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def panel_counts(data, panel, expected_episodes):
    phase, mode = panel.split("-")
    require(data["phase_key"] == phase and data["deterministic"] is (mode == "deterministic"),
            f"panel identity mismatch: {panel}")
    episodes = data["episodes"]
    require(len(episodes) == expected_episodes and len(episodes) > 0, f"episode count mismatch: {panel}")
    counts = dict(episodes=len(episodes), successes=0, falls=0, timeouts=0, targets=0)
    seeds = []
    for episode in episodes:
        success, fall, timeout = (episode[key] for key in ("nav_success", "nav_fall", "timeout"))
        require(type(success) is bool and type(timeout) is bool
                and type(fall) in (int, bool) and fall in (0, 1), f"invalid outcome counts: {panel}")
        require(int(success) + int(fall) + int(timeout) == 1, f"outcome counts do not partition episodes: {panel}")
        counts["successes"] += int(success)
        counts["falls"] += int(fall)
        counts["timeouts"] += int(timeout)
        counts["targets"] += integer(episode["nav_targets"], "nav_targets")
        seeds.append(integer(episode["seed"], "evaluation seed"))
    require(len(set(seeds)) == len(seeds), f"duplicate episode seed: {panel}")
    summary = data["summary"]
    for reported, actual in (("episodes", "episodes"), ("nav_successes", "successes"),
                             ("nav_falls", "falls"), ("timeouts", "timeouts")):
        require(integer(summary[reported], reported) == counts[actual], f"summary count mismatch: {panel}/{reported}")
    counts.update(success_rate=counts["successes"] / len(episodes),
                  fall_rate=counts["falls"] / len(episodes),
                  timeout_rate=counts["timeouts"] / len(episodes),
                  targets_per_episode=counts["targets"] / len(episodes), evaluation_seeds=seeds)
    require(math.isclose(number(summary["navigation_targets_per_episode"], "targets mean"),
                         counts["targets_per_episode"], rel_tol=1e-9, abs_tol=1e-9),
            f"target count mismatch: {panel}")
    return counts


def telemetry_rows(path, hashes):
    rows, digest = [], hashlib.sha256()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("monitor/telemetry_ok") == 0:
                continue
            require(row.get("monitor/telemetry_ok") == 1, f"invalid telemetry status at line {line_number}")
            timestamp = number(row.get("monitor/remote_unix"), "monitor/remote_unix", positive=True)
            busy = number(row.get("gpu/util"), "GPU busy percent")
            require(busy <= 100, "invalid GPU busy percent")
            memory = number(row.get("gpu/mem_used_mb"), "GPU memory used")
            rows.append((timestamp, busy, memory))
    hashes[str(path.resolve())] = digest.hexdigest()
    return rows


def summarize(round_dir, *, partial=False, telemetry=None):
    root = Path(round_dir).resolve()
    hashes = {}

    def read(path):
        path = root / path
        payload = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    plan, journal = read("plan.json"), read("round-result.json")
    require(plan.get("schema") == "native-capacity-round-v1" and not plan.get("evaluate_only"),
            "expected a native capacity training round")
    inputs = plan["inputs"]
    expected = {(arch, seed) for arch in ARCHITECTURES for seed in SEEDS}
    require(len(inputs) == 9 and {(tuple(row["net_arch"]), row["seed"]) for row in inputs} == expected,
            "plan must contain exactly nine architecture/seed training replicates")
    require(len({row["label"] for row in inputs}) == 9, "duplicate arm label")
    for row in inputs:
        arch = tuple(row["net_arch"])
        require(row["label"] == f"arch-{'-'.join(map(str, arch))}_seed{row['seed']}"
                and row["parameters"] == ARCHITECTURES[arch], "arm identity/parameter mismatch")
    start = integer(plan["starting_steps"], "starting_steps")
    end = integer(plan["target"], "target")
    require(end > start and start % 32768 == end % 32768 == 0, "invalid cumulative step boundary")
    require(partial or journal.get("status") == "completed", "round is not completed; use --partial explicitly")
    records = {}
    allowed = {f"{row['label']}-{kind}" for row in inputs for kind in ACCEPTED}
    for record in journal["results"]:
        key = record["label"]
        require(key in allowed and key not in records and record.get("kind") == key.rsplit("-", 1)[1],
                "unexpected, duplicate, or mismatched journal command")
        records[key] = record
    next_inputs = {row["label"]: row for row in journal["next_inputs"]}
    require(len(next_inputs) == len(journal["next_inputs"])
            and set(next_inputs) <= {row["label"] for row in inputs}, "invalid next-input lineage set")
    samples = telemetry_rows(Path(telemetry), hashes) if telemetry is not None else None
    arms, pending, panel_seeds = [], [], {}
    for row in inputs:
        label = row["label"]
        files = [f"{label}/train/{name}.json" for name in ("result", "campaign_summary", "native_learning_receipt")]
        files += [f"{label}/eval/metadata.json"] + [f"{label}/eval/{panel}.json" for panel in PANELS]
        ready = all(records.get(f"{label}-{kind}", {}).get("status") in statuses
                    for kind, statuses in ACCEPTED.items()) and label in next_inputs
        if not ready or not all((root / file).is_file() for file in files):
            require(partial, f"arm is not completed or downloaded: {label}")
            pending.append(label)
            continue
        result, timing, native, evaluation, *panels = [read(file) for file in files]
        require(all(data.get("status") == "completed" and data.get("error") is None
                    for data in (result, timing, native)), f"invalid completed training artifacts: {label}")
        require(row["steps"] == start and row["updates"] == start // 32768 * 8
                and result["starting_steps"] == start and result["ending_steps"] == end
                and result["actual_steps"] == end - start, f"actual step mismatch: {label}")
        require(native.get("final_completed_update_verified") is True
                and native["starting_steps"] == start and native["expected_final_steps"] == end
                and native["expected_final_updates"] == end // 32768 * 8
                and native["initial_checkpoint_counters"] == {"num_timesteps": start, "_n_updates": start // 32768 * 8}
                and native["final_checkpoint_counters"] == {"num_timesteps": end, "_n_updates": end // 32768 * 8},
                f"native checkpoint counter proof mismatch: {label}")
        require(native["config"]["net_arch"] == row["net_arch"]
                and native["config"]["parameter_count"] == row["parameters"]
                and native["runtime_policy"]["parameter_count"] == row["parameters"]
                and native["runtime_policy"].get("all_policy_parameters_cuda") is True,
                f"native policy identity mismatch: {label}")
        following = next_inputs[label]
        checkpoint_hash = native["final_checkpoint"]["sha256"]
        require(len(checkpoint_hash) == 64 and set(checkpoint_hash) <= set("0123456789abcdef")
                and evaluation["checkpoint_sha256"] == checkpoint_hash
                and following["checkpoint_sha256"] == checkpoint_hash
                and evaluation["cumulative_steps"] == following["steps"] == end
                and following["updates"] == end // 32768 * 8
                and following["seed"] == row["seed"] and following["net_arch"] == row["net_arch"]
                and evaluation.get("training_performed") is False, f"evaluation checkpoint mismatch: {label}")
        require(all(timing["plan"][key] == value for key, value in CONTROLS.items()), "training controls mismatch")
        cycles = integer(timing["completed_measured_cycles"], "measured cycles")
        warmup = integer(timing["warmup_cycles"], "warmup cycles")
        require(cycles > 0 and timing["completed_cycles_total"] == cycles + warmup
                and (cycles + warmup) * 32768 == end - start
                and timing["measured_transitions"] == cycles * 32768, "measured transition count mismatch")
        seconds = number(timing["measured_cycle_seconds"], "measured cycle seconds", positive=True)
        collect = number(timing["measured_collect_seconds"], "collection seconds")
        update = number(timing["measured_update_seconds"], "update seconds")
        require(collect + update <= seconds + 1e-6, "collection/update exceed measured duration")
        sps = timing["measured_transitions"] / seconds
        require(math.isclose(number(timing["measured_transitions_per_second"], "steady SPS"), sps, rel_tol=1e-8), "steady SPS mismatch")
        command = records[f"{label}-train"]
        window = None
        # Resumed commands may retain original timing despite a skipped status.
        if (command["status"] == "completed" or command.get("started_unix") is not None
                or command.get("ended_unix") is not None):
            begin = number(command.get("started_unix"), "remote train started_unix", positive=True)
            finish = number(command.get("ended_unix"), "remote train ended_unix", positive=True)
            require(finish > begin, "invalid remote train command window")
            window = [begin, finish]
        counts = {panel: panel_counts(data, panel, plan["episodes_per_panel"]) for panel, data in zip(PANELS, panels)}
        for panel, data in counts.items():
            seeds = data["evaluation_seeds"]
            require(panel_seeds.setdefault(panel, seeds) == seeds, f"evaluation seed panel mismatch: {panel}")
        arm = dict(label=label, net_arch=row["net_arch"], training_seed=row["seed"], parameters=row["parameters"],
                   starting_steps=start, ending_steps=end, actual_steps=result["actual_steps"],
                   learner_wall_seconds=number(result["wall_seconds"], "result wall seconds", positive=True),
                   process_wall_seconds=window[1] - window[0] if window else None,
                   remote_train_command_window=window, steady_sps=sps,
                   measured_cycles=cycles, warmup_cycles=warmup, measured_cycle_seconds=seconds,
                   collection_share=collect / seconds, update_share=update / seconds, panels=counts)
        if samples is not None:
            selected = [sample for sample in samples if window and window[0] <= sample[0] < window[1]]
            arm["telemetry"] = dict(scope=TELEMETRY_SCOPE, timestamp_field="monitor/remote_unix",
                                    available=window is not None, samples=len(selected),
                                    mean_gpu_busy_percent=mean([sample[1] for sample in selected]),
                                    peak_gpu_memory_used_mb=max((sample[2] for sample in selected), default=None))
        arms.append(arm)
    require(partial or len(arms) == 9, "all nine completed training replicates are required")
    groups = []
    for arch in ARCHITECTURES:
        members = [arm for arm in arms if tuple(arm["net_arch"]) == arch]
        group = dict(net_arch=list(arch), parameters=ARCHITECTURES[arch], training_replicates=len(members),
                     training_seeds=sorted(arm["training_seed"] for arm in members),
                     process_wall_training_replicates=sum(arm["process_wall_seconds"] is not None for arm in members))
        for metric in ("learner_wall_seconds", "process_wall_seconds", "steady_sps", "collection_share", "update_share"):
            group[f"mean_{metric}"] = mean([arm[metric] for arm in members])
        group["panels"] = {panel: {f"mean_{metric}": mean([arm["panels"][panel][metric] for arm in members])
                                  for metric in ("successes", "falls", "timeouts", "targets", "success_rate", "fall_rate", "timeout_rate", "targets_per_episode")}
                           for panel in PANELS}
        groups.append(group)
    return dict(schema="native-capacity-summary-v1", status="completed" if len(arms) == 9 and journal["status"] == "completed" else "partial",
                round=str(root), phase=plan["phase"], evaluation_panel=plan["panel"], target_steps=end,
                expected_training_replicates=9, completed_training_replicates=len(arms), pending_arms=pending,
                interpretation="Descriptive equal-training-seed means only; no statistical significance claim. Evaluation episodes are not training replicates.",
                timing_note="Learner wall is result.json wall_seconds; steady SPS/shares use campaign_summary measured cycles, not process wall. Shares need not sum to one. Preserved timestamps from resumed commands are used; missing skipped-command timestamps are not reconstructed.",
                validation_limit="JSON consistency only; checkpoint ZIP contents and source provenance were not revalidated here. Hashes identify exact input bytes read.",
                arms=arms, architectures=groups, source_sha256=hashes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=Path)
    parser.add_argument("--partial", action="store_true", help="report only fully evidenced arms of an incomplete round")
    parser.add_argument("--telemetry", type=Path, help="session11 scalar JSONL with actual remote timestamps")
    args = parser.parse_args(argv)
    try:
        result = summarize(args.round, partial=args.partial, telemetry=args.telemetry)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Summary rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
