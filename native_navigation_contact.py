#!/usr/bin/env python3
"""Python driver for the native navigation contact step (native/navigation_contact.h).

Loads a capture written by ``navigation_contact_capture.py`` and replays it
through the C reference: bodies are forced from each tick's recorded
pre-step state, the reference owns its broadphase tree and arbiter cache, and
every post-step field is compared bit for bit with the real game.  The first
divergent tick and field are reported; nothing is rounded or tolerated unless
a tolerance is declared explicitly in the receipt.

This validates the contact solver only.  It does not step the player state
machine, produce observations or claim throughput.
"""
from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
NATIVE_DIR = ROOT / "native"
SOURCE_FILES = (NATIVE_DIR / "navigation_contact.h", NATIVE_DIR / "navigation_contact.c", NATIVE_DIR / "navigation_contact.cu")

MAX_BODIES = 12
MAX_SHAPES = 12
MAX_VERTS = 4
MAX_ARBITERS = 24
MAX_CONTACTS = 2
MAX_NODES = 32
MAX_PAIRS = 48
MAX_HANDLERS = 4
WILDCARD_TYPE = (1 << 64) - 1

STATE_NAMES = {0: "first_collision", 1: "normal", 2: "ignore", 3: "cached", 4: "invalidated"}


class Vec(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class BB(ctypes.Structure):
    _fields_ = [("l", ctypes.c_double), ("b", ctypes.c_double), ("r", ctypes.c_double), ("t", ctypes.c_double)]


class Body(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int), ("velocity_mode", ctypes.c_int),
        ("m", ctypes.c_double), ("m_inv", ctypes.c_double), ("i", ctypes.c_double), ("i_inv", ctypes.c_double),
        ("cog", Vec), ("p", Vec), ("v", Vec), ("f", Vec),
        ("a", ctypes.c_double), ("w", ctypes.c_double), ("t", ctypes.c_double),
        ("tf", ctypes.c_double * 6),
        ("v_bias", Vec), ("w_bias", ctypes.c_double),
    ]


class Shape(ctypes.Structure):
    _fields_ = [
        ("body", ctypes.c_int), ("count", ctypes.c_int),
        ("v0", Vec * MAX_VERTS), ("n0", Vec * MAX_VERTS), ("tv", Vec * MAX_VERTS), ("tn", Vec * MAX_VERTS),
        ("bb", BB),
        ("r", ctypes.c_double), ("e", ctypes.c_double), ("u", ctypes.c_double),
        ("surface_v", Vec),
        ("hashid", ctypes.c_ulonglong), ("collision_type", ctypes.c_ulonglong), ("group", ctypes.c_ulonglong),
        ("categories", ctypes.c_uint), ("mask", ctypes.c_uint),
        ("sensor", ctypes.c_int),
    ]


class Contact(ctypes.Structure):
    _fields_ = [
        ("r1", Vec), ("r2", Vec),
        ("nMass", ctypes.c_double), ("tMass", ctypes.c_double), ("bounce", ctypes.c_double),
        ("jnAcc", ctypes.c_double), ("jtAcc", ctypes.c_double), ("jBias", ctypes.c_double), ("bias", ctypes.c_double),
        ("hash", ctypes.c_ulonglong),
    ]


class Arbiter(ctypes.Structure):
    _fields_ = [
        ("active", ctypes.c_int), ("a", ctypes.c_int), ("b", ctypes.c_int), ("body_a", ctypes.c_int), ("body_b", ctypes.c_int),
        ("state", ctypes.c_int), ("stamp", ctypes.c_uint), ("swapped", ctypes.c_int),
        ("e", ctypes.c_double), ("u", ctypes.c_double), ("surface_vr", Vec), ("n", Vec),
        ("count", ctypes.c_int), ("contacts", Contact * MAX_CONTACTS),
    ]


class Node(ctypes.Structure):
    _fields_ = [("active", ctypes.c_int), ("obj", ctypes.c_int), ("bb", BB), ("parent", ctypes.c_int), ("A", ctypes.c_int), ("B", ctypes.c_int),
                ("stamp", ctypes.c_uint), ("pairs", ctypes.c_int)]


class Pair(ctypes.Structure):
    _fields_ = [("active", ctypes.c_int), ("a_leaf", ctypes.c_int), ("a_prev", ctypes.c_int), ("a_next", ctypes.c_int),
                ("b_leaf", ctypes.c_int), ("b_prev", ctypes.c_int), ("b_next", ctypes.c_int), ("id", ctypes.c_uint)]


class Tree(ctypes.Structure):
    _fields_ = [("root", ctypes.c_int), ("stamp", ctypes.c_uint), ("is_static", ctypes.c_int)]


class World(ctypes.Structure):
    _fields_ = [
        ("stamp", ctypes.c_uint), ("curr_dt", ctypes.c_double),
        ("iterations", ctypes.c_int), ("gravity", Vec), ("damping", ctypes.c_double), ("slop", ctypes.c_double), ("bias", ctypes.c_double),
        ("persistence", ctypes.c_uint),
        ("body_count", ctypes.c_int), ("bodies", Body * MAX_BODIES),
        ("shape_count", ctypes.c_int), ("shapes", Shape * MAX_SHAPES),
        ("arbiter_count", ctypes.c_int), ("arbiters", Arbiter * MAX_ARBITERS),
        ("solve_count", ctypes.c_int), ("solve", ctypes.c_int * MAX_ARBITERS),
        ("dyn_tree", Tree), ("static_tree", Tree),
        ("nodes", Node * MAX_NODES), ("pairs", Pair * MAX_PAIRS),
        ("leaf_of_shape", ctypes.c_int * MAX_SHAPES),
        ("shape_active", ctypes.c_int * MAX_SHAPES), ("body_active", ctypes.c_int * MAX_BODIES),
        ("leaf_set_size", ctypes.c_uint), ("leaf_set_entries", ctypes.c_uint),
        ("leaf_set_rank", ctypes.c_int * MAX_SHAPES), ("leaf_set_member", ctypes.c_int * MAX_SHAPES),
        ("handler_count", ctypes.c_int),
        ("handler_type_a", ctypes.c_ulonglong * MAX_HANDLERS), ("handler_type_b", ctypes.c_ulonglong * MAX_HANDLERS),
        ("handler_begin_false", ctypes.c_int * MAX_HANDLERS),
        ("last_error", ctypes.c_int),
        ("scratch_contacts", Contact * MAX_CONTACTS),
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256_file(path) for path in SOURCE_FILES}


def default_contraction() -> str:
    """The macOS Pymunk wheel is built with clang's in-expression contraction; manylinux wheels (x86-64 baseline) have no FMA."""
    return os.environ.get("AI2_CONTACT_FP_CONTRACT") or ("on" if platform.system() == "Darwin" else "off")


def compile_cpu_reference(build_dir: Path, contract: str | None = None) -> Path:
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    contract = contract or default_contraction()
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = build_dir / f"libnavigation_contact-{contract}{suffix}"
    compiler = os.environ.get("CC") or "cc"
    command = [compiler, "-std=c99", "-O2", f"-ffp-contract={contract}", "-fno-fast-math", "-Wall", "-Wextra", "-Werror", "-pedantic",
               "-shared", "-fPIC", "-o", str(library), str(SOURCE_FILES[1])]
    subprocess.run(command, check=True, cwd=str(NATIVE_DIR))
    return library


def compile_cuda_library(build_dir: Path) -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is not available")
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / "libnavigation_contact_cuda.so"
    command = [nvcc, "-O2", "-fmad=false", "--shared", "-Xcompiler", "-fPIC", "-Xcompiler", "-ffp-contract=off",
               "-o", str(library), str(SOURCE_FILES[2])]
    subprocess.run(command, check=True, cwd=str(NATIVE_DIR))
    return library


class CudaContactReference:
    def __init__(self, library_path: Path) -> None:
        self.lib = ctypes.CDLL(str(library_path))
        lib = self.lib
        lib.ai2c_cuda_world_size.restype = ctypes.c_int
        if lib.ai2c_cuda_world_size() != ctypes.sizeof(World):
            raise RuntimeError("CUDA world struct size differs from the Python mirror")
        lib.ai2c_cuda_step_replicated.restype = ctypes.c_int
        lib.ai2c_cuda_step_replicated.argtypes = [ctypes.POINTER(World), ctypes.POINTER(World), ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                                  ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        lib.ai2c_cuda_bench_resident.restype = ctypes.c_int
        lib.ai2c_cuda_bench_resident.argtypes = [ctypes.POINTER(World), ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                                 ctypes.POINTER(ctypes.c_double)]
        lib.ai2c_cuda_release.restype = ctypes.c_int

    def step_replicated(self, world: World, lanes: int, dt: float, bias_coef: float, damping_dt: float) -> tuple[World, bool, int]:
        out = World()
        all_equal = ctypes.c_int(0)
        lane_error = ctypes.c_int(0)
        rc = self.lib.ai2c_cuda_step_replicated(ctypes.byref(world), ctypes.byref(out), lanes, dt, bias_coef, damping_dt,
                                                ctypes.byref(all_equal), ctypes.byref(lane_error))
        if rc != 0:
            raise RuntimeError(f"cuda step failed: {rc}")
        return out, bool(all_equal.value), int(lane_error.value)

    def bench_resident(self, world: World, lanes: int, iterations: int, dt: float, bias_coef: float, damping_dt: float) -> float:
        elapsed = ctypes.c_double(0.0)
        rc = self.lib.ai2c_cuda_bench_resident(ctypes.byref(world), lanes, iterations, dt, bias_coef, damping_dt, ctypes.byref(elapsed))
        if rc != 0:
            raise RuntimeError(f"cuda bench failed: {rc}")
        return elapsed.value

    def release(self) -> None:
        self.lib.ai2c_cuda_release()


class ContactReference:
    def __init__(self, library_path: Path) -> None:
        self.lib = ctypes.CDLL(str(library_path))
        lib = self.lib
        lib.ai2c_world_size.restype = ctypes.c_int
        lib.ai2c_body_size.restype = ctypes.c_int
        lib.ai2c_shape_size.restype = ctypes.c_int
        lib.ai2c_arbiter_size.restype = ctypes.c_int
        lib.ai2c_hash_pair_export.restype = ctypes.c_ulonglong
        lib.ai2c_hash_pair_export.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong]
        lib.ai2c_coefficients_export.restype = None
        lib.ai2c_coefficients_export.argtypes = [ctypes.POINTER(World), ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        lib.ai2c_shape_set_verts_export.restype = ctypes.c_int
        lib.ai2c_shape_set_verts_export.argtypes = [ctypes.POINTER(World), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        lib.ai2c_shape_cache_export.restype = ctypes.c_int
        lib.ai2c_shape_cache_export.argtypes = [ctypes.POINTER(World), ctypes.c_int]
        lib.ai2c_body_set_transform_export.restype = ctypes.c_int
        lib.ai2c_body_set_transform_export.argtypes = [ctypes.POINTER(World), ctypes.c_int]
        lib.ai2c_world_build_trees_export.restype = ctypes.c_int
        lib.ai2c_world_build_trees_export.argtypes = [ctypes.POINTER(World), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.ai2c_world_step_export.restype = ctypes.c_int
        lib.ai2c_world_step_export.argtypes = [ctypes.POINTER(World), ctypes.c_double, ctypes.c_double, ctypes.c_double]
        for name in ("ai2c_world_add_shape_export", "ai2c_world_remove_shape_export", "ai2c_world_remove_body_export"):
            fn = getattr(lib, name)
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.POINTER(World), ctypes.c_int]
        expected = {"world": ctypes.sizeof(World), "body": ctypes.sizeof(Body), "shape": ctypes.sizeof(Shape), "arbiter": ctypes.sizeof(Arbiter)}
        actual = {"world": lib.ai2c_world_size(), "body": lib.ai2c_body_size(), "shape": lib.ai2c_shape_size(), "arbiter": lib.ai2c_arbiter_size()}
        if expected != actual:
            raise RuntimeError(f"ctypes/C struct size mismatch: python {expected} c {actual}")

    def hash_pair(self, a: int, b: int) -> int:
        return int(self.lib.ai2c_hash_pair_export(a, b))

    def coefficients(self, world: World, dt: float) -> tuple[float, float]:
        bias = ctypes.c_double()
        damping = ctypes.c_double()
        self.lib.ai2c_coefficients_export(ctypes.byref(world), dt, ctypes.byref(bias), ctypes.byref(damping))
        return bias.value, damping.value

    def step(self, world: World, dt: float, bias_coef: float, damping_dt: float) -> int:
        return int(self.lib.ai2c_world_step_export(ctypes.byref(world), dt, bias_coef, damping_dt))


# ----------------------------------------------------------------------------- world construction

def _set_vec(target: Vec, value: Any) -> None:
    target.x = float(value[0])
    target.y = float(value[1])


def load_body(body: Body, data: dict[str, Any]) -> None:
    body.type = int(data["type"])
    body.velocity_mode = int(data["velocity_mode"])
    body.m = float(data["m"]); body.m_inv = float(data["m_inv"]); body.i = float(data["i"]); body.i_inv = float(data["i_inv"])
    _set_vec(body.cog, data["cog"]); _set_vec(body.p, data["p"]); _set_vec(body.v, data["v"]); _set_vec(body.f, data["f"])
    body.a = float(data["a"]); body.w = float(data["w"]); body.t = float(data["t"])
    for j in range(6):
        body.tf[j] = float(data["transform"][j])
    _set_vec(body.v_bias, data["v_bias"]); body.w_bias = float(data["w_bias"])


def load_shape(ref: ContactReference, world: World, k: int, data: dict[str, Any]) -> None:
    shape = world.shapes[k]
    shape.body = int(data["body"])
    verts = data["verts"]
    flat = (ctypes.c_double * (2 * len(verts)))(*[float(c) for v in verts for c in v])
    rc = ref.lib.ai2c_shape_set_verts_export(ctypes.byref(world), k, len(verts), flat)
    if rc != 0:
        raise RuntimeError(f"shape {k} vertex setup failed: {rc}")
    shape.r = float(data["radius"]); shape.e = float(data["elasticity"]); shape.u = float(data["friction"])
    _set_vec(shape.surface_v, data["surface_velocity"])
    shape.hashid = int(data["hashid"]); shape.collision_type = int(data["collision_type"])
    shape.group = int(data["filter"]["group"]); shape.categories = int(data["filter"]["categories"]); shape.mask = int(data["filter"]["mask"])
    shape.sensor = 1 if data["sensor"] else 0


def apply_events(ref: ContactReference, world: World, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    problems = []
    for event in events:
        kind = event["kind"]
        if kind == "add":
            k = int(event["shape"])
            body = int(event["body"])
            if k >= MAX_SHAPES or body >= MAX_BODIES:
                raise RuntimeError("capture exceeds reference capacity during add")
            load_body(world.bodies[body], event["body_state"])
            world.body_count = max(world.body_count, body + 1)
            load_shape(ref, world, k, event["shape_config"])
            world.shape_count = max(world.shape_count, k + 1)
            rc = ref.lib.ai2c_world_add_shape_export(ctypes.byref(world), k)
            if rc != 0:
                raise RuntimeError(f"reference add_shape failed: {rc}")
            bb = world.shapes[k].bb
            if [bb.l, bb.b, bb.r, bb.t] != [float(x) for x in event["bb"]]:
                problems.append({"field": f"add.shape[{k}].bb", "expected": event["bb"], "actual": [bb.l, bb.b, bb.r, bb.t], "kind": "value"})
        elif kind == "remove":
            k = int(event["shape"])
            if k >= 0:
                rc = ref.lib.ai2c_world_remove_shape_export(ctypes.byref(world), k)
                if rc != 0:
                    problems.append({"field": f"remove.shape[{k}]", "expected": 0, "actual": rc, "kind": "value"})
        elif kind == "remove_body":
            body = int(event["body"])
            if body >= 0:
                ref.lib.ai2c_world_remove_body_export(ctypes.byref(world), body)
    return problems


def build_world(ref: ContactReference, header: dict[str, Any]) -> World:
    world = World()
    ctypes.memset(ctypes.byref(world), 0, ctypes.sizeof(World))
    space = header["space"]
    world.stamp = 0
    world.curr_dt = 0.0
    world.iterations = int(space["iterations"])
    _set_vec(world.gravity, space["gravity"])
    world.damping = float(space["damping"])
    world.slop = float(space["collision_slop"])
    world.bias = float(space["collision_bias"])
    world.persistence = int(space["collision_persistence"])
    initial = header["initial"]
    bodies = initial["bodies"]
    if len(bodies) > MAX_BODIES or len(header["shapes"]) > MAX_SHAPES:
        raise ValueError("capture exceeds reference capacity")
    world.body_count = len(bodies)
    for k, data in enumerate(bodies):
        load_body(world.bodies[k], data)
    world.shape_count = len(header["shapes"])
    for data in header["shapes"]:
        load_shape(ref, world, int(data["index"]), data)
    for k in range(world.shape_count):
        ref.lib.ai2c_shape_cache_export(ctypes.byref(world), k)
    # Collision handlers of the frozen game: player-player begin=False; player-platform normal.
    types = [int(s["collision_type"]) for s in header["shapes"]]
    labels = header.get("body_labels", [])
    by_label = {labels[k]: types[k] for k in range(len(types)) if k < len(labels)}
    handlers = []
    if "player0" in by_label and "player1" in by_label:
        handlers.append((by_label["player0"], by_label["player1"], 1))
    if "platform1" in by_label:
        for player in ("player0", "player1"):
            if player in by_label:
                handlers.append((by_label[player], by_label["platform1"], 0))
    world.handler_count = len(handlers)
    for h, (ta, tb, begin_false) in enumerate(handlers):
        world.handler_type_a[h] = ta; world.handler_type_b[h] = tb; world.handler_begin_false[h] = begin_false
    # Replay shape insertion with add-time bounds so the broadphase tree starts like the real space.
    adds = header["add_order"]
    order = (ctypes.c_int * len(adds))(*[int(a["shape"]) for a in adds])
    bbs = (ctypes.c_double * (4 * len(adds)))(*[float(c) for a in adds for c in a["bb"]])
    vs = (ctypes.c_double * (2 * len(adds)))(*[float(c) for a in adds for c in a["body_v"]])
    rc = ref.lib.ai2c_world_build_trees_export(ctypes.byref(world), order, bbs, vs, len(adds))
    if rc != 0:
        raise RuntimeError(f"tree construction failed: {rc}")
    return world


# ----------------------------------------------------------------------------- comparison

def bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def body_fields(body: Body) -> dict[str, Any]:
    return {
        "p": [body.p.x, body.p.y], "v": [body.v.x, body.v.y], "f": [body.f.x, body.f.y], "a": body.a, "w": body.w, "t": body.t,
        "transform": [body.tf[j] for j in range(6)], "v_bias": [body.v_bias.x, body.v_bias.y], "w_bias": body.w_bias,
    }


def contact_fields(con: Contact) -> dict[str, Any]:
    return {"r1": [con.r1.x, con.r1.y], "r2": [con.r2.x, con.r2.y], "nMass": con.nMass, "tMass": con.tMass, "bounce": con.bounce,
            "jnAcc": con.jnAcc, "jtAcc": con.jtAcc, "jBias": con.jBias, "bias": con.bias, "hash": int(con.hash)}


def arbiter_fields(arb: Arbiter) -> dict[str, Any]:
    return {"a": arb.a, "b": arb.b, "body_a": arb.body_a, "body_b": arb.body_b, "e": arb.e, "u": arb.u,
            "surface_vr": [arb.surface_vr.x, arb.surface_vr.y], "n": [arb.n.x, arb.n.y], "count": arb.count, "swapped": arb.swapped,
            "stamp": arb.stamp, "state": arb.state, "contacts": [contact_fields(arb.contacts[j]) for j in range(arb.count)]}


def tree_fields(world: World) -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    for k in range(world.shape_count):
        leaf = world.leaf_of_shape[k]
        if leaf < 0 or not world.shape_active[k] or world.bodies[world.shapes[k].body].type == 2:
            continue
        node = world.nodes[leaf]
        pairs = []
        pair = node.pairs
        guard = 0
        while pair >= 0 and guard < 64:
            p = world.pairs[pair]
            pairs.append([world.nodes[p.a_leaf].obj, world.nodes[p.b_leaf].obj, int(p.id)])
            pair = p.a_next if p.a_leaf == leaf else p.b_next
            guard += 1
        leaves[str(k)] = {"bb": [node.bb.l, node.bb.b, node.bb.r, node.bb.t], "stamp": int(node.stamp), "pairs": pairs}
    order: list[int] = []
    stack = [world.dyn_tree.root] if world.dyn_tree.root >= 0 else []
    while stack:
        n = stack.pop()
        node = world.nodes[n]
        if node.obj >= 0:
            order.append(node.obj)
        else:
            stack.append(node.B); stack.append(node.A)
    return {"stamp": int(world.dyn_tree.stamp), "leaves": leaves, "order": order}


def _flatten(prefix: str, value: Any, out: list[tuple[str, Any]]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten(f"{prefix}.{key}" if prefix else str(key), value[key], out)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _flatten(f"{prefix}[{idx}]", item, out)
    else:
        out.append((prefix, value))


def diff_records(expected: Any, actual: Any, prefix: str) -> list[dict[str, Any]]:
    left: list[tuple[str, Any]] = []
    right: list[tuple[str, Any]] = []
    _flatten(prefix, expected, left)
    _flatten(prefix, actual, right)
    lmap = dict(left)
    rmap = dict(right)
    mismatches = []
    for key in sorted(set(lmap) | set(rmap)):
        if key not in lmap or key not in rmap:
            mismatches.append({"field": key, "expected": lmap.get(key), "actual": rmap.get(key), "kind": "missing"})
            continue
        e, a = lmap[key], rmap[key]
        if isinstance(e, float) or isinstance(a, float):
            if bits(e) != bits(a):
                mismatches.append({"field": key, "expected": e, "actual": a, "kind": "float", "abs_diff": abs(float(e) - float(a))})
        elif e != a:
            mismatches.append({"field": key, "expected": e, "actual": a, "kind": "value"})
    return mismatches


def canonical_cache(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for entry in entries:
        key = f"{min(entry['a'], entry['b'])}-{max(entry['a'], entry['b'])}"
        out[key] = entry
    return out


def open_capture(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def verify_capture(ref: ContactReference, capture_path: Path, *, max_mismatch_ticks: int = 25,
                   cuda: "CudaContactReference | None" = None, cuda_lanes: int = 256) -> dict[str, Any]:
    metadata = None
    headers: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    with open_capture(capture_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "metadata" in row:
                metadata = row["metadata"]
            elif "header" in row:
                headers.append(row["header"])
            else:
                records.append(row)
    if metadata is None or not headers or not records:
        raise ValueError("capture is missing metadata, headers or records")
    header_by_episode = {(h["seed"], h["episode"]): h for h in headers}
    episodes: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for rec in records:
        episodes.setdefault((rec["seed"], rec["episode"]), []).append(rec)

    compared = 0
    mismatch_ticks: list[dict[str, Any]] = []
    field_max: dict[str, float] = {}
    solved_total = 0
    multi_contact_ticks = 0
    first_divergence = None
    cuda_lanes_identical = True
    cuda_cpu_identical = True
    cuda_first_mismatch = None
    for key in sorted(episodes):
        header = header_by_episode[key]
        world = build_world(ref, header)
        for rec in episodes[key]:
            pre = rec["pre"]
            event_problems = apply_events(ref, world, rec.get("events_before", []))
            for data in pre["bodies"]:
                load_body(world.bodies[int(data["index"])], data)
            for key_s, props in pre.get("shape_props", {}).items():
                shape = world.shapes[int(key_s)]
                shape.e = float(props["elasticity"]); shape.u = float(props["friction"])
                _set_vec(shape.surface_v, props["surface_velocity"])
                shape.collision_type = int(props["collision_type"])
                shape.group = int(props["filter"]["group"]); shape.categories = int(props["filter"]["categories"]); shape.mask = int(props["filter"]["mask"])
                shape.sensor = 1 if props["sensor"] else 0
            if world.stamp != int(pre["stamp"]):
                mismatch_ticks.append({"seed": key[0], "episode": key[1], "tick": rec["tick"], "mismatches": [{"field": "stamp_pre", "expected": pre["stamp"], "actual": world.stamp, "kind": "value"}]})
                world.stamp = int(pre["stamp"])
            world.curr_dt = float(pre["curr_dt"])
            dt = float(rec["dt"])
            bias_coef, damping_dt = ref.coefficients(world, dt)
            cuda_world = None
            if cuda is not None:
                before = World()
                ctypes.memmove(ctypes.byref(before), ctypes.byref(world), ctypes.sizeof(World))
                cuda_world, lanes_equal, lane_error = cuda.step_replicated(before, cuda_lanes, dt, bias_coef, damping_dt)
                if lane_error != 0:
                    raise RuntimeError(f"cuda lane error {lane_error} at seed {key[0]} tick {rec['tick']}")
                cuda_lanes_identical = cuda_lanes_identical and lanes_equal
            rc = ref.step(world, dt, bias_coef, damping_dt)
            if rc != 0:
                raise RuntimeError(f"reference step failed at seed {key[0]} tick {rec['tick']}: {rc}")
            if cuda_world is not None:
                same = ctypes.string_at(ctypes.byref(cuda_world), ctypes.sizeof(World)) == ctypes.string_at(ctypes.byref(world), ctypes.sizeof(World))
                if not same:
                    cuda_cpu_identical = False
                    if cuda_first_mismatch is None:
                        cuda_first_mismatch = {"seed": key[0], "episode": key[1], "tick": rec["tick"],
                                               "diffs": diff_records({"bodies": {str(k): body_fields(world.bodies[k]) for k in range(world.body_count)}},
                                                                     {"bodies": {str(k): body_fields(cuda_world.bodies[k]) for k in range(cuda_world.body_count)}}, "")[:8]}
            post = rec["post"]
            present_bodies = [int(b["index"]) for b in post["bodies"]]
            actual_post = {
                "stamp": world.stamp,
                "tree": tree_fields(world),
                "bodies": {str(k): body_fields(world.bodies[k]) for k in present_bodies},
                "shape_bbs": {k: [world.shapes[int(k)].bb.l, world.shapes[int(k)].bb.b, world.shapes[int(k)].bb.r, world.shapes[int(k)].bb.t] for k in post.get("shape_bbs", {})},
                "solve": [arbiter_fields(world.arbiters[world.solve[j]]) for j in range(world.solve_count)],
                "cache": canonical_cache([arbiter_fields(world.arbiters[j]) for j in range(world.arbiter_count) if world.arbiters[j].active]),
            }
            expected_post = {
                "stamp": post["stamp"],
                "tree": post.get("tree", {}),
                "bodies": {str(b["index"]): {k: v for k, v in b.items() if k in ("p", "v", "f", "a", "w", "t", "transform", "v_bias", "w_bias")} for b in post["bodies"]},
                "shape_bbs": post.get("shape_bbs", {}),
                "solve": [{k: v for k, v in a.items() if k in ("a", "b", "body_a", "body_b", "e", "u", "surface_vr", "n", "count", "swapped", "stamp", "state", "contacts")} for a in post["solve"]],
                "cache": canonical_cache([{k: v for k, v in a.items() if k in ("a", "b", "body_a", "body_b", "e", "u", "surface_vr", "n", "count", "swapped", "stamp", "state", "contacts")} for a in post["cache"]]),
            }
            solved_total += len(post["solve"])
            if any(len(a["contacts"]) > 0 for a in post["solve"]) and len(post["solve"]) > 1:
                multi_contact_ticks += 1
            mismatches = event_problems + diff_records(expected_post, actual_post, "")
            compared += 1
            for m in mismatches:
                if m["kind"] == "float":
                    field = m["field"].split("[")[0].split(".")[-1]
                    field_max[field] = max(field_max.get(field, 0.0), m["abs_diff"])
            if mismatches:
                if first_divergence is None:
                    first_divergence = {"seed": key[0], "episode": key[1], "tick": rec["tick"], "label": rec.get("label"), "first": mismatches[0], "count": len(mismatches)}
                if len(mismatch_ticks) < max_mismatch_ticks:
                    mismatch_ticks.append({"seed": key[0], "episode": key[1], "tick": rec["tick"], "label": rec.get("label"), "mismatches": mismatches[:12], "count": len(mismatches)})
    exact = first_divergence is None
    return {
        "schema": "ai2.navigation_contact_verify.v1",
        "created_unix": time.time(),
        "capture": str(Path(capture_path).resolve()),
        "capture_sha256": sha256_file(capture_path),
        "capture_metadata": {k: metadata.get(k) for k in ("controller", "seeds", "steps_per_seed", "records", "pymunk", "chipmunk", "source")},
        "source_hashes": source_hashes(),
        "compared_ticks": compared,
        "episodes": len(episodes),
        "solved_arbiters_total": solved_total,
        "ticks_with_multiple_solved_arbiters": multi_contact_ticks,
        "exact": exact,
        "first_divergence": first_divergence,
        "mismatch_ticks": mismatch_ticks,
        "mismatch_tick_count": len(mismatch_ticks),
        "max_abs_diff_by_field": field_max,
        "tolerance": "none; bitwise equality required for every compared field",
        "mode": "bodies forced from captured pre-state each tick; reference owns broadphase tree and arbiter cache",
        "cuda": None if cuda is None else {"lanes": cuda_lanes, "all_lanes_identical_every_tick": cuda_lanes_identical,
                                            "cuda_world_bitwise_equal_to_cpu_every_tick": cuda_cpu_identical, "first_mismatch": cuda_first_mismatch},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="replay a capture through the C reference")
    verify.add_argument("capture", type=Path)
    verify.add_argument("--build-dir", type=Path, default=ROOT / "build" / "navigation_contact")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--fp-contract", choices=("on", "off", "fast"), default=None, help="CPU reference contraction (default: platform rule)")
    verify.add_argument("--cuda", action="store_true", help="also step every tick on the GPU across --cuda-lanes replicated lanes")
    verify.add_argument("--cuda-lanes", type=int, default=256)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"refusing to overwrite {args.output}", file=sys.stderr)
        return 2
    library = compile_cpu_reference(args.build_dir, args.fp_contract)
    ref = ContactReference(library)
    cuda = CudaContactReference(compile_cuda_library(args.build_dir)) if args.cuda else None
    result = verify_capture(ref, args.capture, cuda=cuda, cuda_lanes=args.cuda_lanes)
    result["fp_contract"] = args.fp_contract or default_contraction()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {k: result[k] for k in ("compared_ticks", "episodes", "solved_arbiters_total", "ticks_with_multiple_solved_arbiters", "exact", "mismatch_tick_count")}
    summary["first_divergence"] = result["first_divergence"]
    summary["cuda"] = result["cuda"]
    print(json.dumps(summary, sort_keys=True, default=str))
    ok = result["exact"] and (result["cuda"] is None or (result["cuda"]["all_lanes_identical_every_tick"] and result["cuda"]["cuda_world_bitwise_equal_to_cpu_every_tick"]))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
