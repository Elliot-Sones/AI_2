"""Read-only pre/post-Pymunk observations for native physics qualification.

This is not a restorable solver snapshot: private bias velocities, warm-start
impulses and broadphase caches are deliberately not claimed to be serialized.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _vector(value: Any, field: str) -> list[float]:
    if len(value) != 2:
        raise ValueError(f"{field} must have two components")
    return [_finite(value[index], f"{field}[{index}]") for index in range(2)]


def _positive_dt(value: Any, field: str = "dt") -> float:
    result = _finite(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _inverse(value: Any, field: str) -> float:
    number = float(value)
    if number == math.inf:
        return 0.0
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be positive or positive infinity")
    return 1.0 / number


def capture_body(body: Any, label: str) -> dict[str, Any]:
    result = {"id": label, "body_type": int(body.body_type)}
    for field in ("position", "velocity", "force", "center_of_gravity"):
        result[field] = _vector(getattr(body, field), f"{label}.{field}")
    for field in ("angle", "angular_velocity", "torque"):
        result[field] = _finite(getattr(body, field), f"{label}.{field}")
    result["inverse_mass"] = _inverse(body.mass, f"{label}.mass")
    result["inverse_moment"] = _inverse(body.moment, f"{label}.moment")
    return result


def _bb(bb: Any, field: str) -> dict[str, float]:
    return {name: _finite(getattr(bb, name), f"{field}.{name}") for name in ("left", "right", "bottom", "top")}


def _shape_filter(value: Any) -> dict[str, int]:
    return {
        "group": int(getattr(value, "group")),
        "categories": int(getattr(value, "categories")),
        "mask": int(getattr(value, "mask")),
    }


def _space_settings(space: Any) -> dict[str, Any]:
    def setting(value: Any, field: str) -> Any:
        number = float(value)
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        if not math.isfinite(number):
            raise ValueError(f"{field} must be finite or infinity")
        return number

    settings: dict[str, Any] = {}
    for field in (
        "iterations",
        "damping",
        "idle_speed_threshold",
        "sleep_time_threshold",
        "collision_slop",
        "collision_bias",
        "collision_persistence",
    ):
        if hasattr(space, field):
            value = getattr(space, field)
            settings[field] = int(value) if field in ("iterations", "collision_persistence") else setting(value, f"space.{field}")
    if hasattr(space, "gravity"):
        settings["gravity"] = _vector(getattr(space, "gravity"), "space.gravity")
    if hasattr(space, "threads"):
        settings["threads"] = int(getattr(space, "threads"))
    return settings


def _label_physics_members(raw: Any) -> tuple[dict[int, str], dict[int, str]]:
    body_labels: dict[int, str] = {}
    shape_labels: dict[int, str] = {}

    def add(label: str, owner: Any) -> None:
        body = getattr(owner, "body", None)
        shape = getattr(owner, "shape", None)
        if body is not None:
            body_labels[id(body)] = label
        if shape is not None:
            shape_labels[id(shape)] = label

    for key, owner in sorted(getattr(raw, "objects", {}).items(), key=lambda item: str(item[0])):
        add(f"objects/{key}", owner)

    for index, player in enumerate(getattr(raw, "players", []) or []):
        agent_id = getattr(player, "agent_id", index)
        add(f"players/{agent_id}", player)

    weapon_controller = getattr(raw, "weapon_controller", None)
    for index, spawner in enumerate(getattr(weapon_controller, "spawners", []) or []):
        spawner_id = getattr(spawner, "id", index)
        add(f"spawners/{spawner_id}", spawner)
        active_weapon = getattr(spawner, "active_weapon", None)
        if active_weapon is not None:
            add(f"spawners/{spawner_id}/active_weapon", active_weapon)

    return body_labels, shape_labels


def _label_for(mapping: dict[int, str], item: Any, kind: str) -> str:
    label = mapping.get(id(item))
    if label is None:
        raise ValueError(f"unlabelled {kind} in physics space")
    return label


def _vertices(shape: Any, label: str) -> dict[str, Any] | None:
    get_vertices = getattr(shape, "get_vertices", None)
    if not callable(get_vertices):
        return None
    local = [_vector(vertex, f"{label}.vertices[{index}]") for index, vertex in enumerate(get_vertices())]
    body = getattr(shape, "body", None)
    local_to_world = getattr(body, "local_to_world", None)
    if callable(local_to_world):
        world = [_vector(local_to_world(vertex), f"{label}.world_vertices[{index}]") for index, vertex in enumerate(get_vertices())]
    else:
        world = None
    return {"local": local, "world": world}


def capture_shape(shape: Any, label: str, body_label: str) -> dict[str, Any]:
    result = {
        "id": label,
        "body_id": body_label,
        "kind": type(shape).__name__,
        "collision_type": int(getattr(shape, "collision_type")),
        "sensor": bool(getattr(shape, "sensor")),
        "friction": _finite(getattr(shape, "friction"), f"{label}.friction"),
        "elasticity": _finite(getattr(shape, "elasticity"), f"{label}.elasticity"),
        "filter": _shape_filter(getattr(shape, "filter")),
        "bb": _bb(getattr(shape, "bb"), f"{label}.bb"),
    }
    vertices = _vertices(shape, label)
    if vertices is not None:
        result["vertices"] = vertices
    return result


def _arbiter_points(arbiter: Any, field: str) -> list[dict[str, Any]]:
    point_set = getattr(arbiter, "contact_point_set", None)
    points = []
    for index, point in enumerate(getattr(point_set, "points", []) or []):
        points.append(
            {
                "point_a": _vector(getattr(point, "point_a"), f"{field}.points[{index}].point_a"),
                "point_b": _vector(getattr(point, "point_b"), f"{field}.points[{index}].point_b"),
                "distance": _finite(getattr(point, "distance"), f"{field}.points[{index}].distance"),
            }
        )
    return points


def capture_contacts(space: Any, body_labels: dict[int, str], shape_labels: dict[int, str]) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def append_arbiter(arbiter: Any) -> None:
        shapes = list(getattr(arbiter, "shapes", ()) or ())
        if len(shapes) < 2:
            return
        shape_ids = tuple(_label_for(shape_labels, shape, "shape") for shape in shapes)
        key = tuple(sorted(shape_ids))
        if key in seen:
            return
        seen.add(key)
        field = f"contact[{len(contacts)}]"
        contacts.append(
            {
                "shape_ids": list(shape_ids),
                "body_ids": [_label_for(body_labels, getattr(shape, "body", None), "body") for shape in shapes],
                "normal_direction": "shape_a_to_shape_b",
                "normal": _vector(getattr(getattr(arbiter, "contact_point_set", None), "normal"), f"{field}.normal"),
                "points": _arbiter_points(arbiter, field),
                "total_impulse": _vector(getattr(arbiter, "total_impulse"), f"{field}.total_impulse"),
                "total_ke": _finite(getattr(arbiter, "total_ke"), f"{field}.total_ke"),
            }
        )

    for body in list(getattr(space, "bodies", []) or []):
        each_arbiter = getattr(body, "each_arbiter", None)
        if callable(each_arbiter):
            each_arbiter(append_arbiter)
    return contacts


def capture_space(raw: Any) -> dict[str, Any]:
    space = raw.space
    body_labels, shape_labels = _label_physics_members(raw)
    bodies = []
    for body in list(getattr(space, "bodies", []) or []):
        label = _label_for(body_labels, body, "body")
        bodies.append(capture_body(body, label))
    shapes = []
    for shape in list(getattr(space, "shapes", []) or []):
        label = _label_for(shape_labels, shape, "shape")
        body_label = _label_for(body_labels, getattr(shape, "body", None), "body")
        shapes.append(capture_shape(shape, label, body_label))
    return {
        "settings": _space_settings(space),
        "bodies": sorted(bodies, key=lambda item: item["id"]),
        "shapes": sorted(shapes, key=lambda item: item["id"]),
        "contacts": sorted(capture_contacts(space, body_labels, shape_labels), key=lambda item: item["shape_ids"]),
        "solver_snapshot_restore_supported": False,
        "solver_snapshot_restore_note": "Pymunk private warm-start impulses, bias velocities and broadphase internals are not serialized.",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def capture_game_physics(
    source_root: Path,
    output: Path,
    *,
    seed: int = 0,
    steps: int = 120,
    action_seed: int | None = None,
    prepare_source: Callable[[Path], None] | None = None,
    build_env: Callable[[int], Any] | None = None,
    make_actions: Callable[[Any, int, int, int], list[Any]] | None = None,
) -> dict[str, Any]:
    if steps < 0 or steps > 500:
        raise ValueError("steps must be between 0 and 500")
    source_root = source_root.resolve()
    output = output.resolve()
    action_seed = seed if action_seed is None else action_seed
    original_cwd = Path.cwd()
    env = None
    try:
        if prepare_source is None:
            import ai2.evaluation.benchmark_replay as benchmark_replay

            prepare_source = benchmark_replay.prepare_source_imports
        prepare_source(source_root)
        if build_env is None or make_actions is None:
            import ai2.evaluation.benchmark_training as bench

            bench.set_headless_environment()
            build_env = build_env or bench.build_raw_env
            make_actions = make_actions or bench.action_plan
            try:
                import torch

                torch.set_num_threads(1)
            except Exception:
                pass
        env = build_env(seed)
        actions = make_actions(getattr(env, "action_space", None), steps, action_seed, 2)
        if hasattr(env, "reset"):
            env.reset(seed=seed)
        with PhysicsStepRecorder(env, capture=capture_space) as recorder:
            for action in actions:
                recorder.attach()
                env.step(action)
        payload = {
            "contract": "actual_raw_game_space_step_observation_only",
            "source_root": str(source_root),
            "seed": seed,
            "action_seed": action_seed,
            "steps_requested": steps,
            "physics_steps_recorded": len(recorder.records),
            "solver_snapshot_restore_supported": False,
            "solver_snapshot_restore_note": "Pymunk private warm-start impulses, bias velocities and broadphase internals are not serialized.",
            "records": recorder.records,
            "summary": summarize_physics_records(recorder.records) if recorder.records else None,
        }
        write_json(output, payload)
        return payload
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        os.chdir(original_cwd)


def summarize_physics_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    max_angle = max_angular_velocity = 0.0
    rotating = set()
    for record in records:
        count += 1
        for side in ("before", "after"):
            for body in record[side]["bodies"]:
                angle = abs(_finite(body["angle"], "angle"))
                omega = abs(_finite(body["angular_velocity"], "angular_velocity"))
                max_angle = max(max_angle, angle)
                max_angular_velocity = max(max_angular_velocity, omega)
                if angle != 0.0 or omega != 0.0:
                    rotating.add(body["id"])
    if count == 0:
        raise ValueError("empty physics capture")
    return {"physics_steps": count, "max_abs_angle": max_angle,
            "max_abs_angular_velocity": max_angular_velocity,
            "rotating_body_ids": sorted(rotating),
            "solver_snapshot_restore_supported": False}


class PhysicsStepRecorder:
    """Temporarily bracket the real space.step, including spaces recreated by reset.

    Call attach() after a game reset. Before/after captures must only read the
    space; failed steps are not recorded as completed transitions. The recorder
    never patches frozen source and restores every installed hook on exit.
    """
    def __init__(self, raw: Any, *, capture: Callable[[Any], dict[str, Any]]):
        self.raw = raw
        self.capture = capture
        self.records: list[dict[str, Any]] = []
        self._space = None
        self._original = None
        self._hook = None

    def attach(self) -> None:
        if self._space is self.raw.space:
            return
        self._restore()
        self._space = self.raw.space
        self._original = self._space.step
        original = self._original

        def step(dt):
            clean_dt = _positive_dt(dt, "dt")
            before = self.capture(self.raw)
            result = original(clean_dt)
            after = self.capture(self.raw)
            self.records.append({"tick": len(self.records), "dt": clean_dt,
                                 "before": before, "after": after})
            return result

        self._hook = step
        self._space.step = step

    def _restore(self) -> None:
        if self._space is not None and self._space.step is self._hook:
            self._space.step = self._original
        self._space = self._original = self._hook = None

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *_):
        self._restore()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-seed", type=int)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args(argv)
    payload = capture_game_physics(
        args.source_root,
        args.output,
        seed=args.seed,
        action_seed=args.action_seed,
        steps=args.steps,
    )
    print(json.dumps({"output": str(args.output.resolve()), "physics_steps_recorded": payload["physics_steps_recorded"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
