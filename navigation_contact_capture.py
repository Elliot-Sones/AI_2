#!/usr/bin/env python3
"""Capture exact pre/post ``space.step`` state from the frozen navigation game.

Runs the real frozen game (same Pymunk 6.2.1 / Chipmunk 7.0.3 build as the
GPU host) under deterministic action streams and records, for every physics
tick, the state Chipmunk reads and writes: every body (including the private
bias velocities), every solved arbiter in ``space->arbiters`` order and every
cached arbiter with its contacts and accumulated impulses.  Hidden fields are
read directly from the live Chipmunk structs at offsets validated against the
public API in the same process (``validate_layout``).

This is an oracle for the native contact step (research/10).  It is not a
restorable solver snapshot for arbitrary mid-episode restarts, and it records
nothing about the broadphase tree.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parent
SCHEMA = "ai2.navigation_contact_capture.v4"
BODY_LABELS = ("ground1", "ground2", "platform1", "player0", "player1")

# Validated struct offsets (Chipmunk 7.0.3 double build, LP64).
BODY = dict(velocity_func=0, m=16, m_inv=24, i=32, i_inv=40, cog=48, p=64, v=80, f=96, a=112, w=120, t=128,
            transform=136, v_bias=192, w_bias=208)
SHAPE = dict(bb=64, sensor=96, e=104, u=112, surfaceV=120, type=144, filter_group=152, filter_categories=160, filter_mask=164, hashid=184)
SPACE = dict(stamp=80, curr_dt=88, shapeIDCounter=128, staticShapes=136, dynamicShapes=144, arbiters=160, cachedArbiters=176)
TREE = dict(leaves=40, root=48, stamp=80)
NODE = dict(obj=0, bb=8, parent=40, A=48, B=56, stamp=48, pairs=56)
PAIR = dict(a_prev=0, a_leaf=8, a_next=16, b_prev=24, b_leaf=32, b_next=40, id=48)
ARB = dict(e=0, u=8, surface_vr=16, a=40, b=48, body_a=56, body_b=64, count=104, contacts=112, n=120, swapped=160, stamp=164, state=168)
CONTACT = dict(size=96, r1=0, r2=16, nMass=32, tMass=40, bounce=48, jnAcc=56, jtAcc=64, jBias=72, bias=80, hash=88)


def _addr(cdata: Any) -> int:
    from pymunk._chipmunk_cffi import ffi

    return int(ffi.cast("uintptr_t", cdata))


def _d(base: int, off: int) -> float:
    return ctypes.cast(base + off, ctypes.POINTER(ctypes.c_double)).contents.value


def _i(base: int, off: int, kind: Any = ctypes.c_int) -> int:
    return ctypes.cast(base + off, ctypes.POINTER(kind)).contents.value


def _u64(base: int, off: int) -> int:
    return _i(base, off, ctypes.c_uint64)


def _u32(base: int, off: int) -> int:
    return _i(base, off, ctypes.c_uint32)


def _vec(base: int, off: int) -> list[float]:
    return [_d(base, off), _d(base, off + 8)]


def velocity_mode_of(body: Any) -> int:
    """0 = Chipmunk default gravity integration, 1 = frozen no-gravity variant.

    Pymunk always installs a CFFI trampoline as the C velocity function and keeps
    the Python callable on the body, so the mode is identified from that callable.
    """
    import pymunk

    func = getattr(body, "_velocity_func_base", None)
    if func is None:
        func = getattr(body, "_velocity_func", None)
    name = getattr(func, "__name__", "")
    if func is None or func is pymunk.Body.update_velocity or name == "update_velocity":
        return 0
    if name == "no_gravity_velocity_func":
        return 1
    raise RuntimeError(f"unsupported velocity function on body: {func!r}")


class SpaceRecorder:
    """Read-only snapshotter bound to one frozen raw env."""

    def __init__(self, raw_env: Any) -> None:
        self.raw_env = raw_env
        self.rebind()

    def rebind(self) -> None:
        env = self.raw_env
        bodies = [env.objects["ground1"].body, env.objects["ground2"].body, env.objects["platform1"].body,
                  env.players[0].body, env.players[1].body]
        shapes = [env.objects["ground1"].shape, env.objects["ground2"].shape, env.objects["platform1"].shape,
                  env.players[0].shape, env.players[1].shape]
        self.space_addr = _addr(env.space._space)
        self.body_addrs = [_addr(b._body) for b in bodies]
        self.shape_addrs = [_addr(s._shape) for s in shapes]
        self.body_index = {a: k for k, a in enumerate(self.body_addrs)}
        self.shape_index = {a: k for k, a in enumerate(self.shape_addrs)}
        self.bodies = list(bodies)
        self.shapes = list(shapes)
        self.labels = list(BODY_LABELS)
        self.present_shapes: set[int] = set(range(len(shapes)))
        self.present_bodies: set[int] = set(range(len(bodies)))
        self.sync_extras()

    def sync_extras(self) -> list[dict[str, Any]]:
        """Assign stable indexes to shapes/bodies that appear after reset (weapons)."""
        added = []
        for shape in self.raw_env.space.shapes:
            addr = _addr(shape._shape)
            if addr in self.shape_index:
                continue
            body = shape.body
            baddr = _addr(body._body)
            if baddr not in self.body_index:
                self.body_index[baddr] = len(self.bodies)
                self.body_addrs.append(baddr)
                self.bodies.append(body)
            k = len(self.shapes)
            self.shape_index[addr] = k
            self.shape_addrs.append(addr)
            self.shapes.append(shape)
            owner = getattr(shape, "owner", None)
            self.labels.append(f"{type(owner).__name__ if owner is not None else 'shape'}:{k}")
            added.append({"shape": k, "body": self.body_index[baddr]})
        present_s = {self.shape_index[_addr(s._shape)] for s in self.raw_env.space.shapes}
        present_b = {self.body_index[_addr(b._body)] for b in self.raw_env.space.bodies if _addr(b._body) in self.body_index}
        self.present_shapes = present_s
        self.present_bodies = present_b
        return added

    # --- static configuration ---------------------------------------------
    def shape_config(self) -> list[dict[str, Any]]:
        out = []
        for k, (shape, addr) in enumerate(zip(self.shapes, self.shape_addrs)):
            if k not in self.present_shapes:
                continue
            out.append({
                "index": k, "label": self.labels[k], "body": self.body_index[_addr(shape.body._body)],
                "verts": [[float(v.x), float(v.y)] for v in shape.get_vertices()],
                "radius": float(shape.radius),
                "elasticity": _d(addr, SHAPE["e"]), "friction": _d(addr, SHAPE["u"]),
                "surface_velocity": _vec(addr, SHAPE["surfaceV"]),
                "collision_type": _u64(addr, SHAPE["type"]),
                "filter": {"group": _u64(addr, SHAPE["filter_group"]), "categories": _u32(addr, SHAPE["filter_categories"]), "mask": _u32(addr, SHAPE["filter_mask"])},
                "hashid": _u64(addr, SHAPE["hashid"]),
                "sensor": bool(shape.sensor),
            })
        return out

    def space_config(self) -> dict[str, Any]:
        sp = self.raw_env.space
        return {"iterations": int(sp.iterations), "gravity": [float(sp.gravity.x), float(sp.gravity.y)], "damping": float(sp.damping),
                "collision_slop": float(sp.collision_slop), "collision_bias": float(sp.collision_bias),
                "collision_persistence": int(sp.collision_persistence), "shape_id_counter": _u64(self.space_addr, SPACE["shapeIDCounter"]),
                "dt": float(self.raw_env.dt)}

    # --- per-tick snapshots ----------------------------------------------
    def shape_props(self) -> dict[str, dict[str, Any]]:
        """Mutable shape properties; the frozen game sets filters after space.add, so record them per tick."""
        out = {}
        for k, addr in enumerate(self.shape_addrs):
            if k not in self.present_shapes:
                continue
            out[str(k)] = {
                "elasticity": _d(addr, SHAPE["e"]), "friction": _d(addr, SHAPE["u"]), "surface_velocity": _vec(addr, SHAPE["surfaceV"]),
                "collision_type": _u64(addr, SHAPE["type"]),
                "filter": {"group": _u64(addr, SHAPE["filter_group"]), "categories": _u32(addr, SHAPE["filter_categories"]), "mask": _u32(addr, SHAPE["filter_mask"])},
                "sensor": bool(_i(addr, SHAPE["sensor"], ctypes.c_ubyte)),
            }
        return out

    def shape_bbs(self) -> dict[str, list[float]]:
        out = {}
        for k, addr in enumerate(self.shape_addrs):
            if k in self.present_shapes:
                out[str(k)] = [_d(addr, SHAPE["bb"] + 8 * j) for j in range(4)]
        return out

    def body_state(self, k: int) -> dict[str, Any]:
        a = self.body_addrs[k]
        body = self.bodies[k]
        return {
            "index": k, "type": int(body.body_type),
            "velocity_mode": velocity_mode_of(body),
            "m": _d(a, BODY["m"]), "m_inv": _d(a, BODY["m_inv"]), "i": _d(a, BODY["i"]), "i_inv": _d(a, BODY["i_inv"]),
            "cog": _vec(a, BODY["cog"]), "p": _vec(a, BODY["p"]), "v": _vec(a, BODY["v"]), "f": _vec(a, BODY["f"]),
            "a": _d(a, BODY["a"]), "w": _d(a, BODY["w"]), "t": _d(a, BODY["t"]),
            "transform": [_d(a, BODY["transform"] + 8 * j) for j in range(6)],
            "v_bias": _vec(a, BODY["v_bias"]), "w_bias": _d(a, BODY["w_bias"]),
        }

    def arbiter_state(self, aa: int) -> dict[str, Any]:
        count = _i(aa, ARB["count"])
        contacts_ptr = _u64(aa, ARB["contacts"])
        contacts = []
        for j in range(count if contacts_ptr else 0):
            c = contacts_ptr + CONTACT["size"] * j
            contacts.append({
                "r1": _vec(c, CONTACT["r1"]), "r2": _vec(c, CONTACT["r2"]),
                "nMass": _d(c, CONTACT["nMass"]), "tMass": _d(c, CONTACT["tMass"]), "bounce": _d(c, CONTACT["bounce"]),
                "jnAcc": _d(c, CONTACT["jnAcc"]), "jtAcc": _d(c, CONTACT["jtAcc"]), "jBias": _d(c, CONTACT["jBias"]),
                "bias": _d(c, CONTACT["bias"]), "hash": _u64(c, CONTACT["hash"]),
            })
        return {
            "a": self.shape_index.get(_u64(aa, ARB["a"]), -1), "b": self.shape_index.get(_u64(aa, ARB["b"]), -1),
            "body_a": self.body_index.get(_u64(aa, ARB["body_a"]), -1), "body_b": self.body_index.get(_u64(aa, ARB["body_b"]), -1),
            "e": _d(aa, ARB["e"]), "u": _d(aa, ARB["u"]), "surface_vr": _vec(aa, ARB["surface_vr"]), "n": _vec(aa, ARB["n"]),
            "count": count, "swapped": _i(aa, ARB["swapped"], ctypes.c_ubyte), "stamp": _u32(aa, ARB["stamp"]), "state": _i(aa, ARB["state"]),
            "contacts": contacts,
        }

    def solve_list(self) -> list[dict[str, Any]]:
        arr = _u64(self.space_addr, SPACE["arbiters"])
        num = _i(arr, 0)
        base = _u64(arr, 8)
        return [self.arbiter_state(_u64(base, 8 * j)) for j in range(num)]

    def cache_list(self) -> list[dict[str, Any]]:
        hs = _u64(self.space_addr, SPACE["cachedArbiters"])
        size = _u32(hs, 4)
        table = _u64(hs, 24)
        out = []
        for b in range(size):
            binp = _u64(table, 8 * b)
            while binp:
                out.append(self.arbiter_state(_u64(binp, 0)))
                binp = _u64(binp, 16)
        out.sort(key=lambda a: (a["a"], a["b"]))
        return out

    def tree_state(self) -> dict[str, Any]:
        """Dynamic BB tree: stamp, each leaf's fat bound/stamp/pair list, and the root-order traversal."""
        tree = _u64(self.space_addr, SPACE["dynamicShapes"])
        out: dict[str, Any] = {"stamp": _u32(tree, TREE["stamp"]), "leaves": {}, "order": []}
        node_shape: dict[int, int] = {}
        dynamic_nodes: dict[int, int] = {}
        for which, index_offset in (("dynamic", SPACE["dynamicShapes"]), ("static", SPACE["staticShapes"])):
            t = _u64(self.space_addr, index_offset)
            hs = _u64(t, TREE["leaves"])
            size = _u32(hs, 4)
            table = _u64(hs, 24)
            for b in range(size):
                binp = _u64(table, 8 * b)
                while binp:
                    node = _u64(binp, 0)
                    node_shape[node] = self.shape_index.get(_u64(node, NODE["obj"]), -1)
                    if which == "dynamic":
                        dynamic_nodes[node] = node_shape[node]
                    binp = _u64(binp, 16)
        for node, k in dynamic_nodes.items():
            pairs = []
            pair = _u64(node, NODE["pairs"])
            guard = 0
            while pair and guard < 64:
                a_leaf = _u64(pair, PAIR["a_leaf"]); b_leaf = _u64(pair, PAIR["b_leaf"])
                pairs.append([node_shape.get(a_leaf, -2), node_shape.get(b_leaf, -2), _u32(pair, PAIR["id"])])
                pair = _u64(pair, PAIR["a_next"]) if a_leaf == node else _u64(pair, PAIR["b_next"])
                guard += 1
            out["leaves"][str(k)] = {"bb": [_d(node, NODE["bb"] + 8 * j) for j in range(4)], "stamp": _u32(node, NODE["stamp"]), "pairs": pairs}
        # In-order traversal of the tree (A before B), leaves only.
        root = _u64(tree, TREE["root"])
        stack = [root] if root else []
        while stack:
            node = stack.pop()
            if node in dynamic_nodes:
                out["order"].append(dynamic_nodes[node])
            else:
                stack.append(_u64(node, NODE["B"])); stack.append(_u64(node, NODE["A"]))
        return out

    def snapshot(self) -> dict[str, Any]:
        return {
            "tree": self.tree_state(),
            "stamp": _u32(self.space_addr, SPACE["stamp"]), "curr_dt": _d(self.space_addr, SPACE["curr_dt"]),
            "bodies": [self.body_state(k) for k in range(len(self.bodies)) if k in self.present_bodies],
            "shape_bbs": self.shape_bbs(), "shape_props": self.shape_props(),
            "solve": self.solve_list(), "cache": self.cache_list(),
            "player0_on_floor": bool(self.raw_env.players[0].is_on_floor()),
        }


def validate_layout(recorder: SpaceRecorder) -> dict[str, Any]:
    """Cross-check raw reads against the public API; raise on any disagreement."""
    env = recorder.raw_env
    problems = []
    for k, body in enumerate(recorder.bodies):
        st = recorder.body_state(k)
        if tuple(st["p"]) != (body.position.x, body.position.y):
            problems.append(f"body{k} position raw {st['p']} api {tuple(body.position)}")
        if tuple(st["v"]) != (body.velocity.x, body.velocity.y):
            problems.append(f"body{k} velocity")
        if st["a"] != body.angle or st["w"] != body.angular_velocity:
            problems.append(f"body{k} angle/w")
        if body.body_type == 0 and (st["m"] != body.mass or st["i"] != body.moment):
            problems.append(f"body{k} mass/moment")
    configs = {c["index"]: c for c in recorder.shape_config()}
    for k, shape in enumerate(recorder.shapes):
        if k not in configs:
            continue
        cfg = configs[k]
        if cfg["collision_type"] != shape.collision_type or cfg["friction"] != shape.friction or cfg["elasticity"] != shape.elasticity:
            problems.append(f"shape{k} type/friction/elasticity")
        if cfg["filter"]["categories"] != shape.filter.categories or cfg["filter"]["mask"] != shape.filter.mask:
            problems.append(f"shape{k} filter")
        bb = shape.cache_bb()
        if recorder.shape_bbs()[str(k)] != [bb.left, bb.bottom, bb.right, bb.top]:
            problems.append(f"shape{k} bb raw {recorder.shape_bbs()[str(k)]} api {(bb.left, bb.bottom, bb.right, bb.top)}")
    hashids = sorted(c["hashid"] for c in configs.values())
    if len(hashids) != len(set(hashids)):
        problems.append(f"duplicate hashids: {hashids}")
    for k in (3, 4):
        arbs: list[Any] = []
        recorder.bodies[k].each_arbiter(lambda arb: arbs.append(arb))
        for arb in arbs:
            raw = recorder.arbiter_state(_addr(arb._arbiter))
            cps = arb.contact_point_set
            if raw["count"] != len(cps.points):
                problems.append("arbiter contact count")
            normal = (-raw["n"][0], -raw["n"][1]) if raw["swapped"] else tuple(raw["n"])
            if normal != (cps.normal.x, cps.normal.y):
                problems.append("arbiter normal")
            total = [0.0, 0.0]
            for c in raw["contacts"]:
                # cpvrotate(n, cpv(jn, jt))
                total[0] += raw["n"][0] * c["jnAcc"] - raw["n"][1] * c["jtAcc"]
                total[1] += raw["n"][0] * c["jtAcc"] + raw["n"][1] * c["jnAcc"]
            if not raw["swapped"]:
                total = [-total[0], -total[1]]
            api = arb.total_impulse
            if any(abs(t - x) > 1e-12 * max(1.0, abs(x)) for t, x in zip(total, (api.x, api.y))):
                problems.append(f"arbiter total_impulse raw {total} api {tuple(api)}")
    if problems:
        raise RuntimeError("raw layout validation failed: " + "; ".join(problems))
    return {"validated": True, "bodies": len(recorder.bodies), "shapes": len(recorder.shapes)}


def find_raw_env(env: Any) -> Any:
    seen: set[int] = set()
    queue = [env]
    while queue:
        item = queue.pop(0)
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if hasattr(item, "space") and hasattr(item, "players") and hasattr(item, "objects"):
            return item
        for attr in ("env", "unwrapped", "venv", "raw_env", "game"):
            child = getattr(item, attr, None)
            if child is not None and child is not item:
                queue.append(child)
        envs = getattr(item, "envs", None)
        if isinstance(envs, (list, tuple)):
            queue.extend(envs)
        # Generic fallback: any attribute holding an object that looks like the raw game.
        for value in list(vars(item).values()) if hasattr(item, "__dict__") else []:
            if value is not None and value is not item and hasattr(value, "space") and hasattr(value, "players"):
                queue.append(value)
    raise RuntimeError("raw WarehouseBrawl env not found")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def random_actions(action_space: Any, steps: int, seed: int) -> Iterable[tuple[str, Any]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    for tick in range(steps):
        # Hold each random action for a few ticks so walks/jumps actually happen.
        if tick % 6 == 0:
            action = rng.uniform(0.0, 1.0, size=action_space.shape).astype(action_space.dtype)
        yield "random", action


def capture(args: argparse.Namespace) -> dict[str, Any]:
    import pymunk

    replay = importlib.import_module("benchmark_replay")
    validator = importlib.import_module("validate_native_navigation")
    source_root = Path(args.source_root).resolve()
    config = Path(args.config).resolve()
    wrapper_source = Path(args.wrapper_source).resolve() if args.wrapper_source else source_root / "train.py"
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    original_cwd = Path.cwd()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    records: list[dict[str, Any]] = []
    headers: list[dict[str, Any]] = []
    original_step = pymunk.Space.step
    original_add = pymunk.Space.add
    original_remove = pymunk.Space.remove
    original_init = pymunk.Space.__init__
    state: dict[str, Any] = {"recorder": None, "tick": 0, "seed": None, "label": None, "episode": 0}
    add_events: dict[int, list[dict[str, Any]]] = {}

    pending_events: list[dict[str, Any]] = []

    def recording_add(space: Any, *objs: Any) -> None:
        original_add(space, *objs)
        events = add_events.setdefault(_addr(space._space), [])
        rec: SpaceRecorder | None = state["recorder"]
        live = rec is not None and _addr(space._space) == rec.space_addr
        if live:
            rec.sync_extras()
        for obj in objs:
            if isinstance(obj, pymunk.Shape):
                bb = obj.cache_bb()
                event = {"shape_addr": _addr(obj._shape), "hashid": _u64(_addr(obj._shape), SHAPE["hashid"]),
                         "bb": [float(bb.left), float(bb.bottom), float(bb.right), float(bb.top)],
                         "body_v": [float(obj.body.velocity.x), float(obj.body.velocity.y)],
                         "body_p": [float(obj.body.position.x), float(obj.body.position.y)]}
                events.append(event)
                if live:
                    k = rec.shape_index[_addr(obj._shape)]
                    cfg = {c["index"]: c for c in rec.shape_config()}[k]
                    pending_events.append({"kind": "add", "shape": k, "body": cfg["body"], "hashid": event["hashid"], "bb": event["bb"],
                                           "body_v": event["body_v"], "shape_config": cfg, "body_state": rec.body_state(cfg["body"])})

    def recording_init(space: Any, *args: Any, **kwargs: Any) -> None:
        original_init(space, *args, **kwargs)
        # A fresh Space may reuse a freed address: start its add log from scratch.
        add_events[_addr(space._space)] = []

    def recording_remove(space: Any, *objs: Any) -> None:
        rec: SpaceRecorder | None = state["recorder"]
        live = rec is not None and _addr(space._space) == rec.space_addr
        if live:
            for obj in objs:
                if isinstance(obj, pymunk.Shape):
                    k = rec.shape_index.get(_addr(obj._shape), -1)
                    pending_events.append({"kind": "remove", "shape": k, "hashid": _u64(_addr(obj._shape), SHAPE["hashid"])})
                elif isinstance(obj, pymunk.Body):
                    pending_events.append({"kind": "remove_body", "body": rec.body_index.get(_addr(obj._body), -1)})
        original_remove(space, *objs)
        if live:
            rec.sync_extras()

    def header_for(rec: "SpaceRecorder", seed: int, episode: int) -> dict[str, Any]:
        events = add_events.get(rec.space_addr, [])
        adds = []
        for ev in events:
            adds.append({"shape": rec.shape_index.get(ev["shape_addr"], -1), "hashid": ev["hashid"], "bb": ev["bb"], "body_v": ev["body_v"], "body_p": ev["body_p"]})
        return {"seed": seed, "episode": episode, "layout": validate_layout(rec), "shapes": rec.shape_config(), "space": rec.space_config(),
                "body_labels": list(rec.labels), "add_order": adds, "initial": rec.snapshot()}

    def recording_step(space: Any, dt: float) -> None:
        rec: SpaceRecorder = state["recorder"]
        if rec is None or _addr(space._space) != rec.space_addr:
            return original_step(space, dt)
        rec.sync_extras()
        events = list(pending_events)
        pending_events.clear()
        pre = rec.snapshot()
        original_step(space, dt)
        post = rec.snapshot()
        records.append({"seed": state["seed"], "episode": state["episode"], "tick": state["tick"], "label": state["label"],
                        "dt": float(dt), "events_before": events, "pre": pre, "post": post})

    pymunk.Space.step = recording_step
    pymunk.Space.add = recording_add
    pymunk.Space.remove = recording_remove
    pymunk.Space.__init__ = recording_init
    try:
        for seed in seeds:
            env = None
            try:
                env, _params, _phase = replay.build_navigation_env(source_root, config, wrapper_source, seed, frame_stack=args.frame_stack)
                wrapper = validator.find_frozen_opponent_wrapper(env)
                validator._reset_env(env)
                raw_env = find_raw_env(wrapper)
                rec = SpaceRecorder(raw_env)
                headers.append(header_for(rec, seed, 0))
                state.update({"recorder": rec, "seed": seed, "episode": 0})
                if args.controller == "adaptive":
                    actions = validator._adaptive_actions(wrapper, args.steps)
                elif args.controller == "scripted":
                    actions = validator._scripted_actions(wrapper, args.steps)
                else:
                    actions = random_actions(wrapper.action_space, args.steps, seed)
                for tick, (label, action) in enumerate(actions):
                    state["tick"] = tick
                    state["label"] = label
                    result = wrapper.step(action)
                    terminated, truncated = bool(result[2]), bool(result[3])
                    if terminated or truncated:
                        pending_events.clear()
                        validator._reset_env(env)
                        state["episode"] += 1
                        rec.rebind()
                        headers.append(header_for(rec, seed, state["episode"]))
            finally:
                state["recorder"] = None
                if env is not None:
                    env.close()
    finally:
        pymunk.Space.step = original_step
        pymunk.Space.add = original_add
        pymunk.Space.remove = original_remove
        pymunk.Space.__init__ = original_init
        os.chdir(original_cwd)

    if not records:
        raise RuntimeError("capture produced zero records")
    metadata = {
        "schema": SCHEMA, "created_unix": time.time(), "controller": args.controller, "seeds": seeds, "steps_per_seed": args.steps,
        "records": len(records), "pymunk": importlib.metadata.version("pymunk"), "chipmunk": getattr(pymunk, "chipmunk_version", None),
        "source": {"root": str(source_root), "environment_sha256": sha256_file(source_root / "UTMIST-AI2-main/environment/environment.py"),
                   "train_sha256": sha256_file(wrapper_source), "config_sha256": sha256_file(config)},
        "capture_sha256": sha256_file(Path(__file__)),
        "offsets": {"body": BODY, "shape": SHAPE, "space": SPACE, "arbiter": ARB, "contact": CONTACT},
        "note": "Hidden solver fields read from live Chipmunk structs; layout validated per episode against the public API.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"metadata": metadata}) + "\n")
        for header in headers:
            handle.write(json.dumps({"header": header}) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")
    solved = sum(len(r["post"]["solve"]) for r in records)
    two_plus = sum(1 for r in records if len(r["post"]["solve"]) > 2)
    return {"output": str(output), "records": len(records), "episodes": len(headers), "solved_arbiters": solved, "ticks_with_more_than_two_arbiters": two_plus}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--wrapper-source", type=Path)
    parser.add_argument("--controller", choices=("adaptive", "scripted", "random"), default="adaptive")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", default="45")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = capture(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
