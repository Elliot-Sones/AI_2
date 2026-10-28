"""Paired, CUDA-policy combat evaluation in an unchanged original game.

This is evaluation, not training. The game runs on CPU; both policies require
CUDA. A timeout is never scored as a win. Completed cases are saved immediately.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np


def build_cases(candidates, opponents, seeds):
    candidates, opponents, seeds = map(list, (candidates, opponents, seeds))
    for names in (candidates, opponents):
        if not names or len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
            raise ValueError("Model names must be unique, nonempty safe identifiers")
    if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be unique uint32 integers")
    return [
        {"id": f"{candidate}__{opponent}__{'d' if deterministic else 's'}__p{slot}__s{seed}",
         "candidate": candidate, "opponent": opponent, "deterministic": deterministic,
         "candidate_slot": slot, "seed": seed}
        for opponent in opponents for deterministic in (True, False)
        for seed in seeds for slot in (0, 1) for candidate in candidates
    ]


def classify_outcome(stocks, terminated, truncated, candidate_slot):
    if candidate_slot not in (0, 1) or len(stocks) != 2 or any(type(stock) is not int or stock < 0 for stock in stocks):
        raise ValueError("Invalid stocks or player slot")
    if 0 in stocks:
        if stocks == [0, 0]:
            return "draw"
        return "loss" if stocks[candidate_slot] == 0 else "win"
    if terminated:
        raise ValueError("Game terminated without stock elimination")
    if truncated:
        return "timeout"
    raise ValueError("Episode ended without termination or truncation")


def relative_stats(players, candidate_slot):
    if candidate_slot not in (0, 1) or len(players) != 2:
        raise ValueError("Expected two players and slot 0 or 1")
    return {"candidate": players[candidate_slot], "opponent": players[1-candidate_slot]}


def update_trace(digest, record, trace_sink=None):
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    digest.update(encoded)
    if trace_sink is not None:
        trace_sink(encoded)


def write_json(path, payload):
    path = Path(path)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def player_stats(env):
    result = []
    for slot in (0, 1):
        stats = env.get_stats(slot)
        result.append({"stocks": int(stats.lives_left), "damage_done": float(stats.damage_done), "damage_taken": float(stats.damage_taken)})
    return result


def seed_runtime(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_episode(env, policies, case, *, max_steps, deadline_unix, seed_fn=seed_runtime, frame_sink=None, trace_sink=None):
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if time.time() >= deadline_unix:
        raise TimeoutError("Combat process deadline reached")
    seed_fn(case["seed"])
    observations, _ = env.reset(seed=case["seed"])
    env.max_timesteps = max_steps
    for policy in policies:
        policy.get_env_info(env)
    digest = hashlib.sha256()
    stats = player_stats(env)
    stocks = [stats[slot]["stocks"] for slot in (0, 1)]
    stock_losses = [0, 0]
    stock_loss_events = []
    counts = np.zeros((2, 10), dtype=np.int64)
    action_min = np.ones((2, 10), dtype=np.float32)
    action_max = np.zeros((2, 10), dtype=np.float32)
    action_changes = [0, 0]
    previous_actions = None
    update_trace(digest, {"initial_obs": [np.asarray(observations[i]).tolist() for i in (0, 1)], "stats": stats}, trace_sink)
    started = time.monotonic()
    for step in range(1, max_steps + 1):
        if time.time() >= deadline_unix:
            raise TimeoutError("Combat process deadline reached; incomplete case excluded")
        actions = {slot: policies[slot].predict(observations[slot]) for slot in (0, 1)}
        action_array = np.stack([actions[0], actions[1]])
        counts += action_array > .5
        action_min = np.minimum(action_min, action_array)
        action_max = np.maximum(action_max, action_array)
        if previous_actions is not None:
            for slot in (0, 1):
                action_changes[slot] += int(not np.array_equal(action_array[slot], previous_actions[slot]))
        previous_actions = action_array.copy()
        observations, rewards, terminated, truncated, _ = env.step(actions)
        stats = player_stats(env)
        for slot in (0, 1):
            current = stats[slot]["stocks"]
            if current > stocks[slot]:
                raise RuntimeError("Unexpected stock increase inside episode")
            if current < stocks[slot]:
                stock_loss_events.append({"step": step, "slot": slot, "before": stocks[slot], "after": current})
            stock_losses[slot] += stocks[slot] - current
            stocks[slot] = current
        update_trace(digest, {"step": step, "actions": action_array.tolist(),
                              "obs": [np.asarray(observations[i]).tolist() for i in (0, 1)],
                              "stats": stats, "terminated": bool(terminated), "truncated": bool(truncated),
                              "rewards": [float(rewards[i]) for i in (0, 1)]}, trace_sink)
        if frame_sink is not None:
            frame_sink(np.ascontiguousarray(np.swapaxes(env.render(), 0, 1), dtype=np.uint8))
        if 0 in stocks or terminated or truncated:
            break
    outcome = classify_outcome(stocks, bool(terminated), bool(truncated), case["candidate_slot"])
    return {"case": case, "outcome": outcome, "steps": step, "simulated_seconds": step / env.fps,
            "wall_seconds": time.monotonic()-started, "terminated": bool(terminated), "truncated": bool(truncated),
            "players": stats, "relative": relative_stats(stats, case["candidate_slot"]),
            "stock_losses": stock_losses, "stock_loss_events": stock_loss_events,
            "end_reason": "post_step_stock_elimination" if 0 in stocks else "time_limit",
            "action_active_counts": counts.tolist(),
            "action_min": action_min.tolist(), "action_max": action_max.tolist(), "action_changes": action_changes,
            "trace_sha256": digest.hexdigest(), "max_steps": max_steps}


def verify_inputs(manifest, repo):
    sources = manifest["source_sha256"]
    if "UTMIST-AI2-main/environment/environment.py" not in sources or not sources:
        raise ValueError("Frozen game source inventory is required")
    for relative, expected in sources.items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo) or sha256(path) != expected:
            raise ValueError(f"Frozen source hash mismatch: {relative}")
    model_hashes = {}
    for name, entry in manifest["models"].items():
        actual = sha256(entry["path"])
        if actual != entry["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
        model_hashes[name] = actual
    cases = build_cases(manifest["candidates"], manifest["opponents"], manifest["seeds"])
    if any(case[key] not in model_hashes for case in cases for key in ("candidate", "opponent")):
        raise ValueError("Schedule references an absent model")
    return cases, model_hashes


def summarize(results):
    rows = {}
    for result in results:
        case = result["case"]
        key = f"{case['candidate']}/{case['opponent']}/{'deterministic' if case['deterministic'] else 'sampled'}"
        row = rows.setdefault(key, {"matches": 0, "win": 0, "loss": 0, "draw": 0, "timeout": 0,
                                    "damage_done_total": 0., "damage_taken_total": 0., "stock_losses_total": 0})
        row["matches"] += 1
        row[result["outcome"]] += 1
        row["damage_done_total"] += result["relative"]["candidate"]["damage_done"]
        row["damage_taken_total"] += result["relative"]["candidate"]["damage_taken"]
        row["stock_losses_total"] += result["stock_losses"][case["candidate_slot"]]
    return {"matches": len(results), "groups": rows, "timeout_is_win": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--max-steps", type=int, default=2700)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.deadline_unix) or args.deadline_unix <= time.time() or args.max_steps <= 0:
        parser.error("A future finite deadline and positive max-steps are required")
    if args.case_limit is not None and args.case_limit <= 0:
        parser.error("case-limit must be positive")
    repo, output = args.repo.resolve(), args.output.resolve()
    manifest = json.loads(args.manifest.read_text())
    cases, model_hashes = verify_inputs(manifest, repo)
    replay = json.loads(args.replay.read_text()) if args.replay else None
    if replay:
        if replay["case"] not in cases or replay["provenance"]["model_sha256"] != model_hashes or replay["provenance"]["source_sha256"] != manifest["source_sha256"]:
            raise ValueError("Replay identity does not match this manifest")
        cases = [replay["case"]]
        args.max_steps = replay["max_steps"]
    elif args.case_limit:
        cases = cases[:args.case_limit]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("WANDB_MODE", "disabled")
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU policy fallback")
    from combat_policy_adapter import SB3CombatPolicy, load_cuda_checkpoint
    models = {name: load_cuda_checkpoint(entry["path"]) for name, entry in manifest["models"].items()}
    provenance = {"model_sha256": model_hashes, "source_sha256": manifest["source_sha256"],
                  "requested_device": "cuda", "model_devices": {name: str(model.device) for name, model in models.items()},
                  "policy_devices": {name: str(model.policy.device) for name, model in models.items()},
                  "torch_version": torch.__version__, "gpu_name": torch.cuda.get_device_name(0),
                  "driver_sha256": sha256(__file__), "adapter_sha256": sha256(Path(__file__).with_name("combat_policy_adapter.py")),
                  "checkpoint_updates": 0, "opponent_deterministic": True}
    write_json(output / "provenance.json", provenance)
    os.chdir(repo)
    sys.path.insert(0, str(repo / "UTMIST-AI2-main"))
    from environment.environment import WarehouseBrawl, RenderMode, CameraResolution
    env = WarehouseBrawl(mode=RenderMode.RGB_ARRAY, resolution=CameraResolution.LOW, train_mode=True)
    results = []
    try:
        for case in cases:
            slot = case["candidate_slot"]
            policies = [None, None]
            policies[slot] = SB3CombatPolicy(models[case["candidate"]], deterministic=case["deterministic"])
            policies[1-slot] = SB3CombatPolicy(models[case["opponent"]], deterministic=True)
            writer = None
            if replay:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(output / "replay.mp4"), fps=env.fps, codec="libx264", quality=8, macro_block_size=None)
            try:
                trace_path = output / f"{case['id']}.trace.jsonl.gz"
                partial_trace = trace_path.with_suffix(".gz.part")
                with gzip.open(partial_trace, "wb", compresslevel=1) as trace:
                    result = run_episode(env, policies, case, max_steps=args.max_steps, deadline_unix=args.deadline_unix,
                                         frame_sink=writer.append_data if writer else None, trace_sink=trace.write)
                partial_trace.replace(trace_path)
                result["trace_file"] = trace_path.name
                result["trace_file_sha256"] = sha256(trace_path)
            finally:
                if writer:
                    writer.close()
            result["provenance"] = provenance
            if replay:
                if result["trace_sha256"] != replay["trace_sha256"] or result["outcome"] != replay["outcome"]:
                    raise RuntimeError("Rendered replay diverged from recorded match")
                result["replay_verified"] = True
                result["video_sha256"] = sha256(output / "replay.mp4")
            write_json(output / f"{case['id']}.json", result)
            results.append(result)
            print(json.dumps({"completed": len(results), "planned": len(cases), "id": case["id"], "outcome": result["outcome"],
                              "steps": result["steps"], "relative": result["relative"], "wall_seconds": result["wall_seconds"]}), flush=True)
    finally:
        env.close()
    verify_inputs(manifest, repo)
    write_json(output / "summary.json", summarize(results))
    print(json.dumps({"complete": True, "matches": len(results), "unchanged_checkpoint_hashes": model_hashes}), flush=True)


if __name__ == "__main__":
    main()
