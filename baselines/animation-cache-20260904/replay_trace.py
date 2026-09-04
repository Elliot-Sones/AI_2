"""Compare deterministic game traces across the preserved old/new source trees.

This is a behavior check, not a performance measurement. Run each source tree
in a separate process to isolate imports, RNG state, and asset caches.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    os.chdir(source_root)
    sys.path.insert(0, str(source_root))
    import benchmark_training as bench

    bench.set_headless_environment()
    import torch

    torch.set_num_threads(1)
    results = []
    for seed in (42, 43, 44):
        env = bench.build_raw_env(seed)
        actions = bench.action_plan(env.action_space, 10_000, seed, n_agents=2)
        digest = hashlib.sha256()
        resets, terminations, truncations = 0, 0, 0

        def record(value):
            digest.update(json.dumps(bench.json_safe(value), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode())
            digest.update(b"\n")

        try:
            record(env.reset(seed=seed))
            for index, action in enumerate(actions, 1):
                transition = env.step(action)
                record(transition)
                record([
                    {"weapon": p.weapon, "stocks": p.stocks, "damage": p.damage,
                     "position": list(p.body.position), "velocity": list(p.body.velocity)}
                    for p in env.players
                ])
                record([
                    {"id": obj.id, "kind": type(obj).__name__, "pickup_ready": obj.flag,
                     "spawn_frame": obj.last_spawn_frame,
                     "spawn_steps": obj.vfx._steps("spawn") if obj.vfx else None,
                     "position": obj.world_pos,
                     "active_weapon": obj.active_weapon.name if obj.active_weapon else None}
                    for obj in env.weapon_controller.spawners
                ])
                terminated, truncated = transition[2:4]
                terminations += int(terminated)
                truncations += int(truncated)
                if terminated or truncated or index % 500 == 0:
                    record(env.reset(seed=seed + index))
                    resets += 1
        finally:
            env.close()
        results.append({"seed": seed, "steps": len(actions), "resets": resets,
                        "terminations": terminations, "truncations": truncations,
                        "action_sha256": bench.hash_actions(actions),
                        "trajectory_sha256": digest.hexdigest()})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"source_root": str(source_root),
                                 "environment_sha256": bench.file_digest(
                                     source_root / "UTMIST-AI2-main/environment/environment.py"),
                                 "results": results}, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
