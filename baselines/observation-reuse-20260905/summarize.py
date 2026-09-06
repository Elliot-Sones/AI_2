"""Validate matched local runs and summarize their measured throughput."""

import copy
import json
from pathlib import Path
import pstats
import statistics
import sys


ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
ENVIRONMENT = "UTMIST-AI2-main/environment/environment.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def one(pattern):
    paths = list(ROOT.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {pattern}; found {len(paths)}")
    return paths[0]


def read(path):
    return json.loads(path.read_text())


def without_episode_timing(episodes):
    episodes = copy.deepcopy(episodes)
    for episode in episodes:
        episode.pop("wall_seconds")
        if "episode" in episode.get("info", {}):
            episode["info"]["episode"].pop("t")
    return episodes


def verify_metadata(before, after):
    for key in ("platform", "package_versions", "config_snapshot"):
        require(before[key] == after[key], f"Mismatched {key}")
    for key in before["args"]:
        if key != "output":
            require(before["args"][key] == after["args"][key], f"Mismatched argument {key}")
    changed = sorted(
        key for key in before["source_hashes"].keys() | after["source_hashes"].keys()
        if before["source_hashes"].get(key) != after["source_hashes"].get(key)
    )
    require(changed == [ENVIRONMENT], f"Unexpected source changes: {changed}")


def main():
    evidence = {"workloads": {}, "matched_episode_records": 0}
    for model, modes in (("small", ("raw", "wrapped", "ppo")), ("configured", ("ppo",))):
        values = {mode: {"before": [], "after": [], "paired_ratios": []} for mode in modes}
        for seed in (42, 43, 44):
            paths = {
                variant: one(f"{model}-{variant}/seed-{seed}/baseline_*/metadata.json").parent
                for variant in ("before", "after")
            }
            verify_metadata(*(read(paths[v] / "metadata.json") for v in ("before", "after")))
            for mode in modes:
                runs = {v: read(paths[v] / f"{mode}_repeat0.json") for v in paths}
                before, after = runs["before"], runs["after"]
                require(without_episode_timing(before["episodes"]) == without_episode_timing(after["episodes"]), f"Episode mismatch: {model}, {seed}, {mode}")
                evidence["matched_episode_records"] += len(before["episodes"])
                for field in ("seed", "measured_steps", "completed_episodes", "partial_episodes", "n_envs"):
                    require(before["result"][field] == after["result"][field], f"Mismatched result {field}")
                if mode == "ppo":
                    for run in runs.values():
                        result = run["result"]
                        require(result["measured_steps"] == 32 * 1024, "Incorrect PPO step count")
                        require(len(result["extra"]["rollout_seconds"]) == 32, "Missing PPO rollout")
                        require(len(result["extra"]["update_seconds"]) == 32, "Missing PPO update")
                else:
                    require(before["result"]["measured_steps"] == 100000, "Incorrect action-loop step count")
                    require(before["result"]["extra"]["action_sha256"] == after["result"]["extra"]["action_sha256"], "Action mismatch")
                for variant in paths:
                    result = runs[variant]["result"]
                    values[mode][variant].append({
                        "seed": seed,
                        "steps_per_second": result["steps_per_second"],
                        "total_seconds": result["total_seconds"],
                        "completed_episodes": result["completed_episodes"],
                    })
                values[mode]["paired_ratios"].append(after["result"]["steps_per_second"] / before["result"]["steps_per_second"])
        for mode, rows in values.items():
            for variant in ("before", "after"):
                rates = [r["steps_per_second"] for r in rows[variant]]
                rows[f"{variant}_median"] = statistics.median(rates)
                rows[f"{variant}_range"] = [min(rates), max(rates)]
            rows["median_ratio"] = rows["after_median"] / rows["before_median"]
            rows["time_reduction_percent"] = 100 * (1 - 1 / rows["median_ratio"])
            evidence["workloads"][f"{model}_{mode}"] = rows
    replays = {v: read(ROOT / f"replay-{v}.json") for v in ("before", "after")}
    require(replays["before"]["results"] == replays["after"]["results"], "Replay mismatch")
    evidence["replay"] = replays["after"]["results"]
    evidence["profile_get_obs"] = {}
    for variant in ("before", "after"):
        stats = pstats.Stats(str(one(f"profile-{variant}/baseline_*/raw_repeat0.prof")))
        functions = [v for k, v in stats.stats.items() if k[2] == "get_obs" and k[0].endswith("environment.py")]
        require(len(functions) == 1, "Expected exactly one profiled get_obs function")
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _ = functions[0]
        evidence["profile_get_obs"][variant] = {
            "primitive_calls": primitive_calls, "total_calls": total_calls,
            "self_seconds": self_seconds, "cumulative_seconds": cumulative_seconds,
        }
    require(evidence["profile_get_obs"]["before"]["total_calls"] == 2 * evidence["profile_get_obs"]["after"]["total_calls"], "Expected exactly half as many get_obs calls")
    (ROOT / "comparison.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for name, row in evidence["workloads"].items():
        print(f"{name}: {row['before_median']:.1f} -> {row['after_median']:.1f} steps/s; {row['median_ratio']:.3f}x; time -{row['time_reduction_percent']:.1f}%")
    print(f"Matched {evidence['matched_episode_records']} episode records and all replay hashes.")


if __name__ == "__main__":
    main()
