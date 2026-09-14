"""Policy-free original/native qualification with immutable per-case evidence.

Smoke is deliberately tiny; integration is the bounded lifecycle run. The much
larger comprehensive preset declares candidate hit/whiff fixtures, not coverage
claims. Only recorded events earn coverage. No oracle state enters the adapter.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
import tempfile

ROOT = Path(__file__).resolve().parent
SCHEMA = "ai2.native_game_qualification.v1"


def _idle(ticks):
    return [[[0.] * 10 for _ in range(2)] for _ in range(ticks)]


def build_cases(preset="smoke", *, seed=17):
    """Deterministic input programs; initial fixtures use only raw scenario API."""
    if preset not in {"smoke", "integration", "targeted", "terminal_timeout", "comprehensive", "coverage_completion", "coverage_completion_final"} or type(seed) is not int:
        raise ValueError("invalid preset or seed")
    cases = []

    def add(family, actions, scenario=None, intent=None):
        cases.append({"case_id": f"{family}-{len(cases):03d}", "family": family, "seed": seed + len(cases),
                      "actions": actions, "scenario": scenario or {}, "intent": intent or {}})

    if preset == "coverage_completion_final":
        for case in build_cases("coverage_completion", seed=seed):
            intent = case["intent"]
            if intent["weapon"] != "Spear" or intent["move"] not in (3, 7, 11) or intent["candidate_outcome"] != "hit":
                continue
            slot, variant, move = intent["slot"], intent["geometry_variant"], intent["move"]
            target = case["scenario"]["players"][1 - slot]
            if move == 3:
                target["position"][0] = (2.8, 3., 3.2)[variant]
                case["actions"][(13, 15, 17)[variant]][1 - slot][4] = 1.
            elif move == 7:
                target["position"][0] = 4.5 + intent["facing"] * (1.2, 1.85, 2.5)[variant]
                for row in case["actions"][15 + (1, 10, 20)[variant]:]:
                    row[slot] = [0.] * 10
            else:
                target["position"][0] = 4.5 + (1.05, 1.27, 1.49)[variant]
            intent["prior_evidence"] = "coverage-completion-v1 residual Spear hits"
            add("coverage_completion_final", case["actions"], case["scenario"], intent)
        return cases

    if preset == "coverage_completion":
        # Remaining actual event combinations in comprehensive-v1 + targeted-v1.
        # These are candidate programs, not assertions that the move hits/ends.
        patterns = [
            ("Hammer", 6, (0, 1), (-1,), ("hit",)),
            ("Hammer", 7, (0, 1), (-1,), ("hit",)),
            ("Hammer", 7, (0, 1), (1,), ("hit", "whiff")),
            ("Hammer", 9, (0, 1), (-1, 1), ("hit",)),
            ("Hammer", 11, (0, 1), (-1, 1), ("hit",)),
            ("Hammer", 12, (0, 1), (1,), ("hit",)),
            ("Punch", 2, (1,), (1,), ("hit",)),
            ("Punch", 6, (0, 1), (-1,), ("hit",)),
            ("Punch", 7, (0,), (1,), ("hit", "whiff")),
            ("Punch", 7, (1,), (1,), ("hit",)),
            ("Punch", 12, (0, 1), (1,), ("hit",)),
            ("Spear", 2, (0, 1), (-1, 1), ("hit",)),
            ("Spear", 3, (0, 1), (-1,), ("hit",)),
            ("Spear", 5, (0, 1), (-1,), ("hit",)),
            ("Spear", 6, (0, 1), (-1, 1), ("hit",)),
            ("Spear", 7, (0, 1), (-1, 1), ("hit", "whiff")),
            ("Spear", 9, (0, 1), (-1, 1), ("hit",)),
            ("Spear", 11, (0, 1), (1,), ("hit",)),
        ]
        for weapon, move, slots, facings, outcomes in patterns:
            for slot, facing, candidate, variant in product(slots, facings, outcomes, range(3)):
                airborne = move >= 8
                x, y = 4.5, -3. if airborne else .2
                if airborne:
                    dx = (-.58 if weapon == "Hammer" and move in (9, 11) else .35 if weapon == "Spear" else -.05) * facing
                    dy = (-1.25 if move == 11 else 1.15 if weapon == "Spear" else .8 if move == 9 else .45)
                    dx += facing * (-.25, 0., .25)[variant]
                else:
                    dx, dy = facing * (.55, 1.2, 1.85)[variant], 0.
                if candidate == "whiff":
                    dx, dy = -facing * 3.5, 0.
                players = [{}, {}]
                players[slot] = {"position": [x, y], "weapon": weapon, "facing": facing,
                                 "state": "in_air" if airborne else "standing"}
                players[1 - slot] = {"position": [x + dx, y + dy], "facing": -facing,
                                     "state": "in_air" if airborne else "standing"}
                actions = _idle(100)
                start = 0 if airborne else 15
                heavy = move not in (2, 3, 4, 8, 9, 10)
                for index in range(start, start + (50 if heavy else 1)):
                    actions[index][slot][8 if heavy else 7] = 1.
                    if move in (3, 6, 9, 12):
                        actions[index][slot][2] = 1.
                    elif move in (4, 7, 10) and index == start:
                        actions[index][slot][3 if facing == 1 else 1] = 1.
                add("coverage_completion", actions, {"players": players},
                    {"weapon": weapon, "move": move, "slot": slot, "facing": facing,
                     "candidate_outcome": candidate, "geometry_variant": variant,
                     "ground_settle_ticks": start, "prior_evidence": "comprehensive-v1+targeted-v1"})
        return cases

    if preset == "terminal_timeout":
        add("terminal_timeout", _idle(3), {"max_timesteps": 2, "players": [{"position": [15., 0.], "stocks": 1}, {}]})
        return cases

    if preset == "targeted":
        for attacker in (0, 1):
            players = [{"position": [4., .2], "state": "standing", "facing": 1},
                       {"position": [4.9, .2], "state": "standing", "facing": -1}]
            actions = _idle(35)
            for row in actions[15:35]:
                row[attacker][7] = 1.
                row[1 - attacker][6] = 1.
            add("dodge_overlap", actions, {"players": players}, {"attacker": attacker, "dodger": 1 - attacker})
        for mirrored in (False, True):
            players = [{"position": [4., .2], "state": "standing", "facing": 1},
                       {"position": [4.9, .2], "state": "standing", "facing": -1}]
            if mirrored:
                players.reverse()
            actions = _idle(45)
            for row in actions[15:45]:
                row[0][7] = row[1][7] = 1.
            add("ordered_simultaneous", actions, {"players": players}, {"physical_positions_swapped": mirrored})
        for attacker in (0, 1):
            actions = _idle(100)
            for row in actions[15:]:
                row[attacker][8] = 1.
            add("max_charge", actions, {"players": [
                {"position": [4., .2], "state": "standing", "facing": 1},
                {"position": [4.8, .2], "state": "standing", "facing": -1}]},
                {"attacker": attacker, "move": "Punch NSig", "heavy_held_through_charge_limit": True})
        return cases

    ticks = 3 if preset == "smoke" else 300
    add("idle", _idle(ticks))
    rng = random.Random(seed)
    add("random", [[[float(rng.randrange(2)) for _ in range(10)] for _ in range(2)] for _ in range(ticks)],
        intent={"both_players": True, "all_controls_unmasked": True})
    add("timeout", _idle(3), {"max_timesteps": 3})
    if preset == "smoke":
        return cases
    for weapon in ("Punch", "Spear", "Hammer"):
        actions = _idle(180)
        for tick, row in enumerate(actions):
            for slot in (0, 1):
                row[slot][7 if tick % 60 < 30 else 8] = 1.
                if tick % 45 >= 30:
                    row[slot][2] = 1.
        add("simultaneous", actions, {"players": [
            {"position": [4., .2], "state": "standing", "facing": 1, "weapon": weapon},
            {"position": [4.9, .2], "state": "standing", "facing": -1, "weapon": weapon}]}, {"weapon": weapon})
    for weapon in ("Spear", "Hammer"):
        actions = _idle(650)
        for tick in (0, 20, 40, 80, 320, 340, 600):
            for slot in (0, 1):
                actions[tick][slot][5] = 1.
        add("drop_pickup", actions, {"players": [
            {"position": [3.5, .49], "facing": 1, "weapon": weapon},
            {"position": [4.8, .49], "facing": -1, "weapon": weapon}]},
            {"weapon": weapon, "intended_events": ["drop", "pickup", "despawn", "respawn"]})
    add("ko_respawn", _idle(120), {"players": [{"position": [15., 0.]}, {}]})
    add("match_end", _idle(3), {"players": [{"position": [15., 0.], "stocks": 1}, {}]})
    actions = _idle(300)
    for tick, row in enumerate(actions):
        for slot in (0, 1):
            row[slot][1 if (tick // 40 + slot) % 2 else 3] = 1.
            if tick % 60 == 0:
                row[slot][4] = 1.
    add("moving_ledge", actions, {"players": [{"position": [1.9, -.99]}, {"position": [.1, -.99]}]})
    actions = _idle(120)
    for tick, row in enumerate(actions):
        row[0][6 if tick < 60 else 9] = 1.
        row[1][9 if tick < 60 else 6] = 1.
    add("dodge_taunt", actions)
    if preset == "integration":
        return cases
    add("terminal_timeout", _idle(3), {"max_timesteps": 2, "players": [{"position": [15., 0.], "stocks": 1}, {}]})
    for weapon, move, slot, facing, candidate in product(("Punch", "Spear", "Hammer"), range(2, 13),
                                                        (0, 1), (-1, 1), ("hit", "whiff")):
        airborne = move >= 8
        x = 4.5 - facing * .5
        other_x = x + facing * (.9 if candidate == "hit" else 4.)
        players = [{}, {}]
        players[slot] = {"position": [x, -.5 if airborne else .2], "weapon": weapon, "facing": facing,
                         "state": "in_air" if airborne else "standing"}
        players[1 - slot] = {"position": [other_x, -.5 if airborne else .2], "facing": -facing,
                             "state": "in_air" if airborne else "standing"}
        actions = _idle(120)
        for row in actions[:45]:
            row[slot][7 if move in (2, 3, 4, 8, 9, 10) else 8] = 1.
            if move in (3, 6, 9, 12):
                row[slot][2] = 1.
            elif move in (4, 7, 10):
                row[slot][3 if facing == 1 else 1] = 1.
        add("move_candidate", actions, {"players": players},
            {"weapon": weapon, "move": move, "slot": slot, "facing": facing, "candidate_outcome": candidate})
    for bits in product((0., 1.), repeat=10):
        add("binary_input", [[list(bits), list(bits)]])
    return cases


def targeted_evidence(document):
    """Observe original events; never infer immunity or a charged hit from input alone."""
    from combat_episode_capture import validate_capture
    validate_capture(document)
    result = {"dodge_invulnerable_overlaps": [], "simultaneous_attack_start_ticks": [], "same_tick_reciprocal_damage": [],
              "phase0_damage_before_phase1_stun": [], "max_charge_transitions": [], "max_charge_then_hits": [], "damage": []}
    charged = {}
    for row in document["ticks"]:
        phase_players = None
        tick_damage = []
        starts, stuns = set(), []
        for event in row["events"]:
            if event["kind"] == "player_physics_begin":
                phase_players = event["players"]
            if event["kind"] == "attack_started":
                starts.add(event["player_slot"])
            if event["kind"] == "gameplay" and event["event"]["kind"] == 6:
                stuns.append(event["event"])
            if event["kind"] == "attack_collision" and event["intersects"] and not event["vulnerable"]:
                target = event["target_slot"]
                if phase_players is not None and phase_players[target]["state"] == "DodgeState":
                    result["dodge_invulnerable_overlaps"].append({
                        "tick": row["tick"], "time_seconds": row["tick"] * row["state"]["raw"]["dt"],
                        "phase_player": event["phase_player"], "source": event["source_slot"], "target": target,
                        "attack_id": event["attack_id"], "cast_frame": event["cast_frame"],
                        "hurtbox": event["hurtbox"], "hitbox": event["hitbox"], "target_state": "DodgeState"})
            if event["kind"] == "power_step" and event["next_power_id"] is not None and phase_players is not None:
                slot = event["player_slot"]
                power = phase_players[slot]["bank"]["8"]["move_manager"]["current_power"]
                maximum = power["casts"][-1]["startup_frames"]
                if power["is_charge"] and power["total_frame_count"] > maximum and row["actions"][slot][8] > .5:
                    proof = {"tick": row["tick"], "source": slot, "attack_id": event["attack_id"],
                             "charge_counter_before_step": power["total_frame_count"], "max_charge_threshold": maximum,
                             "heavy_held": True, "next_power_id": event["next_power_id"]}
                    charged[(slot, event["attack_id"])] = proof
                    result["max_charge_transitions"].append(proof)
            if event["kind"] == "damage":
                damage = {"tick": row["tick"], "time_seconds": row["tick"] * row["state"]["raw"]["dt"],
                          "phase_player": event["phase_player"], "raw_event_sequence": event["sequence"],
                          "source": event["source_slot"], "target": event["target_slot"], "attack_id": event["attack_id"],
                          "damage_before": event["before"]["damage_taken_total"], "damage_after": event["after"]["damage_taken_total"],
                          "target_state_before": event["before"]["state"],
                          "target_xy_before": [event["before"]["x"], event["before"]["y"]]}
                result["damage"].append(damage)
                tick_damage.append(damage)
                key = (event["source_slot"], event["attack_id"])
                if key in charged and damage["damage_after"] > damage["damage_before"]:
                    result["max_charge_then_hits"].append({"charge": charged[key], "hit": damage})
        if starts == {0, 1}:
            result["simultaneous_attack_start_ticks"].append(row["tick"])
        if {event["source"] for event in tick_damage} == {0, 1}:
            result["same_tick_reciprocal_damage"].append({"tick": row["tick"], "damage": tick_damage})
        if (any(event["source"] == 0 and event["phase_player"] == 0 for event in tick_damage)
                and any(event["target"] == 1 and event["phase_player"] == 1 for event in stuns)
                and not any(event["source"] == 1 for event in tick_damage)):
            result["phase0_damage_before_phase1_stun"].append({"tick": row["tick"], "damage": tick_damage, "stun": stuns})
    result["damage_source_slots"] = sorted({event["source"] for event in result["damage"]})
    return result


def _digest(path, relative_to=None):
    payload = path.read_bytes()
    return {"path": str(path.relative_to(relative_to) if relative_to else path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def _write_json(path, value):
    """Atomic exclusive publication; an existing artifact is never replaced."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=".qualification-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(name, path)
    finally:
        os.unlink(name)


def _sources():
    paths = [ROOT / name for name in ("native_game_qualification.py", "native_game_snapshot.py", "native_game_episode.py",
             "native_attack_tables.py", "native_navigation_episode.py", "native_navigation_contact.py",
             "combat_episode_capture.py", "combat_episode_parity.py", "navigation_episode_capture.py", "navigation_contact_capture.py")]
    paths += sorted((ROOT / "native").glob("*.h")) + sorted((ROOT / "native").glob("*.c"))
    return [_digest(path, ROOT) for path in paths]


def run_qualification(source_root, output, *, preset="smoke", seed=17, adapter_factory=None):
    from combat_episode_capture import capture_raw, write_capture
    from combat_episode_parity import GAMEPLAY_EXCLUSIONS, GAMEPLAY_SCOPE, coverage, compare_capture
    from native_game_snapshot import GameParityAdapter, SPAWN_ASSETS

    cases = build_cases(preset, seed=seed)
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = _sources()
    config = {"schema": SCHEMA, "preset": preset, "seed": seed, "source_root": str(source_root),
              "scope": GAMEPLAY_SCOPE, "exclusions": GAMEPLAY_EXCLUSIONS, "sources": sources, "cases": cases,
              "lifecycle": "fresh original game and fresh native adapter per independent case"}
    _write_json(output / "config.json", config)
    summary = {"schema": SCHEMA, "preset": preset, "status": "error", "exact": False, "full_combat_coverage": False,
               "requested_cases": len(cases), "requested_ticks": sum(len(case["actions"]) for case in cases),
               "compared_ticks": 0, "scope": GAMEPLAY_SCOPE, "exclusions": GAMEPLAY_EXCLUSIONS, "cases": [],
               "provenance": {}, "coverage": coverage([]),
               "limits": "Host reference only, no GPU or policy inference. Scenario intent is not observed coverage. "
                         "Raw phase diagnostics remain preserved but uncompared under the explicit gameplay projection."}
    observed_inputs = set()
    stage = "adapter_initialization"
    try:
        adapter = (adapter_factory or GameParityAdapter)(source=source_root / "UTMIST-AI2-main/environment")
        summary["provenance"] = {"native_library": _digest(Path(adapter.library)),
                                 "environment_sha256": adapter.metadata.tables["source_sha256"],
                                 "attack_sha256": adapter.metadata.tables["asset_sha256"],
                                 "spawn_assets": json.loads(json.dumps(SPAWN_ASSETS)), "sources": sources}
        _write_json(output / "provenance.json", summary["provenance"])
        for index, case in enumerate(cases):
            if index:
                stage = f"adapter_initialization:{case['case_id']}"
                adapter = (adapter_factory or GameParityAdapter)(source=source_root / "UTMIST-AI2-main/environment")
                if _digest(Path(adapter.library)) != summary["provenance"]["native_library"]:
                    raise RuntimeError("native library changed during qualification")
            folder = output / case["case_id"]
            folder.mkdir()
            _write_json(folder / "inputs.json", case)
            stage = f"capture:{case['case_id']}"
            document = capture_raw(source_root, seed=case["seed"], actions=case["actions"], scenario=case["scenario"])
            capture_receipt = write_capture(document, folder / "capture.json.gz")
            capture_receipt["path"] = str((folder / "capture.json.gz").relative_to(output))
            stage = f"compare:{case['case_id']}"
            report = compare_capture(document, adapter, scope=GAMEPLAY_SCOPE)
            if preset == "targeted":
                report["targeted_evidence"] = targeted_evidence(document)
            _write_json(folder / "comparison.json", report)
            row = {"case_id": case["case_id"], "family": case["family"], "capture": capture_receipt, "report": report}
            summary["cases"].append(row)
            summary["compared_ticks"] += report["compared_ticks"]
            observed = report["coverage"]
            total = summary["coverage"]
            for key in ("ticks", "attack_starts", "damage_frames", "damaging_attacks", "completed_whiffs"):
                total[key] += observed[key]
            total["missing_move_cases"] = [item for item in total["missing_move_cases"] if item in observed["missing_move_cases"]]
            total["missing_mechanics"] = sorted(set(total["missing_mechanics"]) & set(observed["missing_mechanics"]))
            total["move_cases_observed"] = total["move_cases_required"] - len(total["missing_move_cases"])
            for tick in document["ticks"]:
                for slot, action in enumerate(tick["actions"]):
                    observed_inputs.add((slot, tuple(value > .5 for value in action)))
            total["binary_input_cases_observed"] = len(observed_inputs)
            total["full_combat_coverage"] = (not total["missing_move_cases"] and not total["missing_mechanics"]
                                             and len(observed_inputs) == total["binary_input_cases_required"])
            # Each case is its own compressed chunk; do not retain a full corpus in RAM.
            del document
            if not report["exact"]:
                summary["status"] = "mismatch"
                break
        else:
            summary["exact"] = True
            summary["full_combat_coverage"] = summary["coverage"]["full_combat_coverage"]
            summary["status"] = "full_coverage_exact" if summary["full_combat_coverage"] else "exact_for_recorded_cases"
    except Exception as exc:
        summary["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    if sources != _sources():
        summary.update(status="source_changed", exact=False, full_combat_coverage=False)
    _write_json(output / "summary.json", summary)
    files = [_digest(path, output) for path in sorted(output.rglob("*")) if path.is_file()]
    _write_json(output / "receipt.json", {"schema": SCHEMA, "files": files,
                                          "inventory_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()})
    _write_json(output / "receipt.sha256.json", _digest(output / "receipt.json", output))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "integration", "targeted", "terminal_timeout", "comprehensive", "coverage_completion", "coverage_completion_final"), default="smoke")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        cases = build_cases(args.preset, seed=args.seed)
        print(json.dumps({"preset": args.preset, "requested_ticks": sum(len(c["actions"]) for c in cases),
                          "cases": [{key: value for key, value in case.items() if key != "actions"} for case in cases]}, sort_keys=True))
        return 0
    if args.source_root is None or args.output is None:
        parser.error("--source-root and --output are required unless --list")
    result = run_qualification(args.source_root, args.output, preset=args.preset, seed=args.seed)
    print(json.dumps({key: result[key] for key in ("status", "exact", "full_combat_coverage", "compared_ticks")}, sort_keys=True))
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
