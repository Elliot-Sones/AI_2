"""Fail-closed raw-game replay comparison; no native engine or policy loading.

Adapter.reset(draws, scenario) and adapter.step(actions_2x10, draws) return
outputs/state/events/draws_consumed. The adapter never receives oracle state.
Protocol-test doubles prove comparator sensitivity, not simulator parity.
"""
from __future__ import annotations

import copy
from itertools import product
import math
import struct
from typing import Any

from combat_episode_capture import validate_capture

GAMEPLAY_SCOPE = "gameplay_projection_v1"
GAMEPLAY_EXCLUSIONS = {
    "players.constants": "Constructor constants are pinned by original source and reset provenance; not runtime-mutated.",
    "players.hitboxes_to_draw,players.points_to_draw": "Rendering arrays; actual attack/collision events and state remain compared.",
    "players.hurtbox.rect,cap1,cap2": "Derived pygame collision caches rebuilt by every intersects call; base geometry remains compared.",
    "objects.SpawnerVFX*,objects.DroppedVFX*": "Rendering-only object entries; physical object insertion order remains compared.",
    "diagnostic_events": "Verbose phase snapshots/collision attempts remain in raw receipt; only the explicitly typed semantic stream is gated.",
}


def gameplay_projection(row: dict[str, Any]) -> dict[str, Any]:
    result = {key: copy.deepcopy(row[key]) for key in ("outputs", "state")}
    result["outputs"]["rewards"] = {slot: float(value) for slot, value in result["outputs"]["rewards"].items()}
    for player in result["state"]["players"]:
        for key in ("constants", "hitboxes_to_draw", "points_to_draw"):
            del player[key]
        for key in ("rect", "cap1", "cap2"):
            player["hurtbox"].pop(key, None)
        for key in ("damage", "damage_taken_this_stock", "damage_taken_total", "damage_done", "damage_taken_this_frame",
                    "prev_x", "prev_y", "smoothXVel"):
            if key in player:
                player[key] = float(player[key])
        for key in ("damage_velocity", "target_vel", "start_position"):
            player[key] = [float(value) for value in player[key]]
        for key in ("center", "offset", "size"):
            if key in player["hurtbox"]:
                player["hurtbox"][key] = [float(value) for value in player["hurtbox"][key]]
        for bank in player["bank"].values():
            if "move_manager" in bank:
                power = bank["move_manager"]["current_power"]
                power["current_dealt_position_target"] = [float(value) for value in power["current_dealt_position_target"]]
                power["past_point_positions"] = [[float(value) for value in point] for point in power["past_point_positions"]]
            for key, value in bank.items():
                if key in {"is_base", "is_grounded"}:
                    bank[key] = bool(value)
                elif isinstance(value, (int, float)):
                    if int(value) != value:
                        raise ValueError("noninteger state-bank counter")
                    bank[key] = int(value)
    result["state"]["objects"] = [key for key in result["state"]["objects"] if not key.startswith(("SpawnerVFX", "DroppedVFX"))]
    for key in ("waypoint1", "waypoint2"):
        result["state"]["platform"][key] = [float(value) for value in result["state"]["platform"][key]]
    result["events"] = [copy.deepcopy(event["event"]) for event in row["events"] if event["kind"] == "gameplay"]
    return result


def diff_records(expected: Any, actual: Any, path: str = "record") -> list[dict[str, Any]]:
    """Compare all keys, order, scalar types and IEEE-754 signed-zero bits."""
    if type(expected) is not type(actual):
        return [{"field": path, "kind": "type", "expected": type(expected).__name__, "actual": type(actual).__name__}]
    if isinstance(expected, dict):
        result = [{"field": f"{path}.{key}", "kind": "missing"} for key in sorted(expected.keys() - actual.keys())]
        result += [{"field": f"{path}.{key}", "kind": "unexpected"} for key in sorted(actual.keys() - expected.keys())]
        for key in sorted(expected.keys() & actual.keys()):
            result.extend(diff_records(expected[key], actual[key], f"{path}.{key}"))
        return result
    if isinstance(expected, list):
        result = [] if len(expected) == len(actual) else [{"field": path, "kind": "length", "expected": len(expected), "actual": len(actual)}]
        for i, (left, right) in enumerate(zip(expected, actual)):
            result.extend(diff_records(left, right, f"{path}[{i}]"))
        return result
    if isinstance(expected, float):
        equal = math.isfinite(actual) and struct.pack("!d", expected) == struct.pack("!d", actual)
    else:
        equal = expected == actual
    return [] if equal else [{"field": path, "kind": "value", "expected": expected, "actual": actual}]


def coverage(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Credit only observed starts and positive damage or completed whiffs."""
    move_cases, mechanics, inputs = set(), set(), set()
    counts = {"ticks": 0, "attack_starts": 0, "damage_frames": 0, "damaging_attacks": 0, "completed_whiffs": 0}
    for document in documents:
        validate_capture(document)
        attacks: dict[tuple[int, int], dict[str, Any]] = {}
        for row in document["ticks"]:
            counts["ticks"] += 1
            for slot, action in enumerate(row["actions"]):
                inputs.add((slot, tuple(v > .5 for v in action)))
            for player in row["state"]["players"]:
                if player["state"] in {"DodgeState", "TauntState", "StunState"}:
                    mechanics.add(player["state"])
            if row["outputs"]["truncated"]:
                mechanics.add("timeout")
            started_slots = set()
            for event in row["events"]:
                kind = event["kind"]
                if kind == "attack_started":
                    key = (event["player_slot"], event["attack_id"])
                    attacks[key] = {**event, "hit": False, "ended": False}
                    started_slots.add(event["player_slot"])
                    counts["attack_starts"] += 1
                elif kind == "damage":
                    counts["damage_frames"] += 1
                    key = (event["source_slot"], event["attack_id"])
                    if key in attacks and event["after"]["damage_taken_total"] > event["before"]["damage_taken_total"]:
                        attacks[key]["hit"] = True
                elif kind == "attack_ended":
                    key = (event["player_slot"], event["attack_id"])
                    if key in attacks:
                        attacks[key]["ended"] = True
                elif kind == "attack_collision" and event["intersects"] and not event["vulnerable"]:
                    mechanics.add("invulnerable_overlap")
                elif kind in {"knockout_signal", "respawn", "weapon_drop_signal", "weapon_equip_signal"}:
                    mechanics.add(kind)
            if started_slots == {0, 1}:
                mechanics.add("simultaneous_attacks")
        for attack in attacks.values():
            if attack["hit"]:
                outcome = "hit"
                counts["damaging_attacks"] += 1
            elif attack["ended"]:
                outcome = "whiff"
                counts["completed_whiffs"] += 1
            else:
                continue
            move_cases.add((attack["weapon"], attack["move"], attack["player_slot"], attack["facing"], outcome))
    required_moves = set(product(("Punch", "Spear", "Hammer"), range(2, 13), (0, 1), (-1, 1), ("hit", "whiff")))
    required_mechanics = {"DodgeState", "TauntState", "StunState", "timeout", "knockout_signal", "respawn", "weapon_drop_signal",
                          "weapon_equip_signal", "simultaneous_attacks", "invulnerable_overlap"}
    missing = sorted(required_moves - move_cases)
    missing_mechanics = sorted(required_mechanics - mechanics)
    return {**counts, "move_cases_observed": len(required_moves & move_cases), "move_cases_required": 264,
            "missing_move_cases": [list(case) for case in missing], "missing_mechanics": missing_mechanics,
            "binary_input_cases_observed": len(inputs), "binary_input_cases_required": 2048,
            "full_combat_coverage": not missing and not missing_mechanics and len(inputs) == 2048,
            "note": "Coverage measures recorded behavior, not policy quality or native correctness; finite cases are not an exhaustive proof."}


def compare_capture(document: dict[str, Any], adapter: Any, *, scope="raw") -> dict[str, Any]:
    validate_capture(document)
    if scope not in ("raw", GAMEPLAY_SCOPE):
        raise ValueError("unknown comparison scope")
    observed = coverage([document])
    report = {"exact": False, "compared_ticks": 0, "reset_compared": False, "first_divergence": None,
              "coverage": observed, "full_combat_coverage": observed["full_combat_coverage"], "scope": scope,
              "exclusions": GAMEPLAY_EXCLUSIONS if scope == GAMEPLAY_SCOPE else {}}
    for tick, row in enumerate([document["reset"], *document["ticks"]]):
        if tick == 0:
            actual = adapter.reset(copy.deepcopy(row["draws"]), copy.deepcopy(document["metadata"]["scenario"]))
        else:
            actual = adapter.step(copy.deepcopy(row["actions"]), copy.deepcopy(row["draws"]))
        expected = gameplay_projection(row) if scope == GAMEPLAY_SCOPE else {key: row[key] for key in ("outputs", "state", "events")}
        expected["draws_consumed"] = len(row["draws"])
        differences = diff_records(expected, actual)
        if differences:
            report["first_divergence"] = {"tick": tick, "mismatch_count": len(differences), "differences": differences[:30]}
            return report
        if tick:
            report["compared_ticks"] += 1
        else:
            report["reset_compared"] = True
    report["exact"] = True
    return report
