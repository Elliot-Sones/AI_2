"""Evidence-grade CPU-vs-CUDA gate for native combat reward components.

The probe is intentionally generated into a temporary build directory so the
gate cannot become a second production reward implementation. CPU always runs.
CUDA must be real nvcc + torch CUDA unless the caller explicitly allows a local
skip for non-CUDA hosts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
AI2CR_TERMS = 14


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Scenario:
    ident: int
    name: str
    components: tuple[str, ...]


SCENARIOS = (
    Scenario(0, "approach_proximity", ("approach_proximity",)),
    Scenario(1, "attack_contact_damage", ("attack_start", "landed_hit_damage")),
    Scenario(2, "terminal_ko_win_clean", ("ko_win_signs", "clean_win")),
    Scenario(3, "fall_walkoff", ("fall_walkoff",)),
    Scenario(4, "recovery", ("recovery",)),
    Scenario(5, "timeout_truncation", ("terminal_vs_truncation",)),
    Scenario(6, "episode_reset", ("episode_reset",)),
    Scenario(7, "invalid_config_preserves_state", ("invalid_config_byte_preservation",)),
)


class GateOutput(C.Structure):
    _fields_ = [
        (name, C.c_int32)
        for name in (
            "scenario",
            "reset_rc",
            "step_rc",
            "record_rc",
            "state_preserved",
            "state_tick",
            "terminated",
            "truncated",
            "win_mask",
            "record_length",
            "record_terminated",
            "record_truncated",
            "record_win_mask",
            "record_timeout",
        )
    ] + [
        ("before_hash", C.c_uint64),
        ("after_hash", C.c_uint64),
        ("approach_delta", C.c_double),
        ("distance", C.c_double),
        ("terms", (C.c_double * AI2CR_TERMS) * 2),
        ("total", C.c_double * 2),
        ("returns", C.c_double * 2),
        ("term_totals", (C.c_double * AI2CR_TERMS) * 2),
        ("damage_done", C.c_double * 2),
        ("damage_taken", C.c_double * 2),
        ("attack_starts", C.c_int32 * 2),
        ("contacts", C.c_int32 * 2),
        ("damage_events", C.c_int32 * 2),
        ("ko_events", C.c_int32 * 2),
        ("clean_win", C.c_int32 * 2),
        ("offstage", C.c_int32 * 2),
        ("walkoff", C.c_int32 * 2),
        ("recovery", C.c_int32 * 2),
    ]


@contextmanager
def temporary_output_dir(prefix: str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        yield Path(tmp)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes() -> dict[str, dict[str, Any]]:
    names = [
        "native_combat_reward_gate.py",
        "native/combat_reward.h",
        "native/game_lanes.h",
        "native/game_episode.h",
        "native/game_episode.c",
        "native/navigation_episode.h",
        "native/navigation_episode.c",
    ]
    hashes = {
        name: {"sha256": sha256_file(ROOT / name), "bytes": (ROOT / name).stat().st_size}
        for name in names
    }
    for name, text in {
        "generated/combat_reward_gate.c": _probe_source(False),
        "generated/combat_reward_gate.cu": _probe_source(True),
    }.items():
        encoded = text.encode("utf-8")
        hashes[name] = {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}
    return hashes


def cuda_available() -> bool:
    if not shutil.which("nvcc"):
        return False
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _probe_source(cuda: bool) -> str:
    cuda_prefix = '#include <cuda_runtime.h>\n' if cuda else ""
    cuda_wrapper = r"""
__global__ void ai2_gate_kernel(int scenario, AI2GateOutput *out, int *rc) {
    *rc = ai2_gate_run_impl(scenario, out);
}

extern "C" int ai2_gate_run(int scenario, AI2GateOutput *out) {
    AI2GateOutput *device_out = 0;
    int *device_rc = 0;
    int host_rc = -1;
    cudaError_t err;
    if (!out) return -2;
    err = cudaMalloc((void **)&device_out, sizeof(*device_out));
    if (err != cudaSuccess) return -3;
    err = cudaMalloc((void **)&device_rc, sizeof(*device_rc));
    if (err != cudaSuccess) {
        cudaFree(device_out);
        return -4;
    }
    ai2_gate_kernel<<<1, 1>>>(scenario, device_out, device_rc);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(out, device_out, sizeof(*out), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(&host_rc, device_rc, sizeof(host_rc), cudaMemcpyDeviceToHost);
    cudaFree(device_out);
    cudaFree(device_rc);
    return err == cudaSuccess ? host_rc : -5;
}
"""
    host_wrapper = r"""
#ifdef __cplusplus
extern "C" {
#endif
int ai2_gate_run(int scenario, AI2GateOutput *out) {
    return ai2_gate_run_impl(scenario, out);
}
#ifdef __cplusplus
}
#endif
"""
    wrapper = cuda_wrapper if cuda else host_wrapper
    return cuda_prefix + r"""
#include "native/combat_reward.h"
#include <stdint.h>
#include <string.h>

typedef struct {
    int32_t scenario, reset_rc, step_rc, record_rc, state_preserved, state_tick;
    int32_t terminated, truncated, win_mask;
    int32_t record_length, record_terminated, record_truncated, record_win_mask, record_timeout;
    uint64_t before_hash, after_hash;
    double approach_delta, distance;
    double terms[2][AI2CR_TERMS], total[2], returns[2], term_totals[2][AI2CR_TERMS];
    double damage_done[2], damage_taken[2];
    int32_t attack_starts[2], contacts[2], damage_events[2], ko_events[2];
    int32_t clean_win[2], offstage[2], walkoff[2], recovery[2];
} AI2GateOutput;

AI2N_HD static inline uint64_t gate_hash_bytes(const void *ptr, int count) {
    const unsigned char *p = (const unsigned char *)ptr;
    uint64_t h = 1469598103934665603ull;
    int i;
    for (i = 0; i < count; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

AI2N_HD static inline void gate_coeff(AI2CRCoefficients *c) {
    int i;
    ai2n_zero(c, sizeof(*c));
    c->enabled_mask = (1u << AI2CR_TERMS) - 1u;
    c->fade = 1.0;
    for (i = 0; i < AI2CR_TERMS; ++i) c->term[i] = 1.0;
    c->term[AI2CR_TERM_DAMAGE_TAKEN] = -1.0;
    c->term[AI2CR_TERM_TIMEOUT] = -1.0;
    c->term[AI2CR_TERM_OFFSTAGE] = -1.0;
    c->term[AI2CR_TERM_WALKOFF] = -1.0;
}

AI2N_HD static inline void gate_zero_game(AI2GState *g) {
    int i;
    ai2n_zero(g, sizeof(*g));
    g->initialized = 1;
    g->scene.mcfg.stage_width_tiles = 29.8;
    g->scene.mcfg.stage_height_tiles = 16.8;
    for (i = 0; i < 2; ++i) {
        g->scene.players[i].ms.stocks = 3;
        g->scene.players[i].weapon = 0;
        g->combat[i].last_damage_source = -1;
    }
    g->scene.players[0].ms.x = -4.0;
    g->scene.players[1].ms.x = 4.0;
}

AI2N_HD static inline void gate_place(AI2GState *g, int i, double x, double y) {
    g->scene.players[i].ms.x = x;
    g->scene.players[i].ms.y = y;
}

AI2N_HD static inline void gate_attack(AI2GState *g, int source) {
    AI2GCombatPlayer *c = &g->combat[source];
    c->attack_starts += 1;
    c->events[c->event_count].kind = AI2G_EVENT_ATTACK_STARTED;
    c->events[c->event_count].source = source;
    c->event_count += 1;
}

AI2N_HD static inline void gate_contact(AI2GState *g, int source, int target) {
    AI2GCombatPlayer *c = &g->combat[source];
    c->attack_contacts += 1;
    c->events[c->event_count].kind = AI2G_EVENT_ATTACK_CONTACT;
    c->events[c->event_count].source = source;
    c->events[c->event_count].target = target;
    c->event_count += 1;
}

AI2N_HD static inline void gate_damage(AI2GState *g, int source, int target, double amount) {
    AI2GCombatPlayer *c = &g->combat[source];
    c->damage_done += amount;
    c->damage_events += 1;
    g->combat[target].damage_taken_total += amount;
    c->events[c->event_count].kind = AI2G_EVENT_DAMAGE;
    c->events[c->event_count].source = source;
    c->events[c->event_count].target = target;
    c->events[c->event_count].amount = amount;
    c->event_count += 1;
}

AI2N_HD static inline void gate_ko(AI2GState *g, int target) {
    AI2GCombatPlayer *c = &g->combat[target];
    g->scene.players[target].ms.stocks -= 1;
    c->events[c->event_count].kind = AI2G_EVENT_KO;
    c->events[c->event_count].source = target;
    c->events[c->event_count].target = target;
    c->event_count += 1;
}

AI2N_HD static inline void gate_export(int scenario, const AI2CRState *before,
                                       const AI2CRState *after, const AI2CRBreakdown *b,
                                       const AI2CREpisodeRecord *r, AI2GateOutput *out) {
    int i, t;
    ai2n_zero(out, sizeof(*out));
    out->scenario = scenario;
    out->state_tick = after->tick;
    out->before_hash = gate_hash_bytes(before, (int)sizeof(*before));
    out->after_hash = gate_hash_bytes(after, (int)sizeof(*after));
    out->state_preserved = out->before_hash == out->after_hash;
    out->terminated = b->terminated;
    out->truncated = b->truncated;
    out->win_mask = b->win_mask;
    out->approach_delta = b->approach_delta;
    out->distance = b->distance;
    out->record_length = r->length;
    out->record_terminated = r->terminated;
    out->record_truncated = r->truncated;
    out->record_win_mask = r->win_mask;
    out->record_timeout = r->timeout;
    for (i = 0; i < 2; ++i) {
        out->total[i] = b->total[i];
        out->returns[i] = after->returns[i];
        out->damage_done[i] = r->damage_done[i];
        out->damage_taken[i] = r->damage_taken[i];
        out->attack_starts[i] = b->attack_starts[i];
        out->contacts[i] = b->contacts[i];
        out->damage_events[i] = b->damage_events[i];
        out->ko_events[i] = b->ko_events[i];
        out->clean_win[i] = r->clean_win[i];
        out->offstage[i] = b->offstage[i];
        out->walkoff[i] = b->walkoff[i];
        out->recovery[i] = b->recovery[i];
        for (t = 0; t < AI2CR_TERMS; ++t) {
            out->terms[i][t] = b->terms[i][t];
            out->term_totals[i][t] = after->term_totals[i][t];
        }
    }
}

AI2N_HD static inline int ai2_gate_run_impl(int scenario, AI2GateOutput *out) {
    AI2GState g;
    AI2GStepResult raw;
    AI2CRState state, before;
    AI2CRBreakdown breakdown;
    AI2CREpisodeRecord record;
    AI2CRCoefficients coeff;
    int reset_rc, step_rc, record_rc;
    if (!out) return -1;
    gate_zero_game(&g);
    ai2n_zero(&raw, sizeof(raw));
    ai2n_zero(&state, sizeof(state));
    ai2n_zero(&before, sizeof(before));
    ai2n_zero(&breakdown, sizeof(breakdown));
    ai2n_zero(&record, sizeof(record));
    gate_coeff(&coeff);
    if (scenario == 4) gate_place(&g, 0, 20.0, 0.0);
    reset_rc = ai2cr_reset(&state, &g, 0);
    if (scenario == 0) gate_place(&g, 0, -3.0, 0.0);
    else if (scenario == 1) {
        gate_attack(&g, 0);
        gate_contact(&g, 0, 1);
        gate_damage(&g, 0, 1, 7.5);
    } else if (scenario == 2) {
        gate_damage(&g, 0, 1, 10.0);
        gate_ko(&g, 1);
        raw.terminated = 1;
        raw.win_mask = 1;
    } else if (scenario == 3) gate_place(&g, 0, 20.0, 0.0);
    else if (scenario == 4) gate_place(&g, 0, 0.0, 0.0);
    else if (scenario == 5) raw.truncated = 1;
    else if (scenario == 6) {
        gate_damage(&g, 0, 1, 10.0);
        raw.terminated = 1;
        raw.win_mask = 1;
    } else if (scenario == 7) coeff.fade = 0.0 / 0.0;
    else return -7;
    before = state;
    step_rc = ai2cr_step(&state, &g, &raw, &coeff, &breakdown);
    record_rc = ai2cr_episode_record(&state, &g, &raw, &record);
    if (scenario == 6) {
        before = state;
        reset_rc = ai2cr_reset(&state, &g, 0);
        ai2n_zero(&breakdown, sizeof(breakdown));
        ai2n_zero(&record, sizeof(record));
    }
    gate_export(scenario, &before, &state, &breakdown, &record, out);
    out->reset_rc = reset_rc;
    out->step_rc = step_rc;
    out->record_rc = record_rc;
    return 0;
}
""" + wrapper


def _compile_probe(kind: str, build_dir: Path) -> Path:
    if kind not in {"cpu", "cuda"}:
        raise ValueError("kind must be cpu or cuda")
    build_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" and kind == "cpu" else ".so"
    source = build_dir / ("combat_reward_gate.cu" if kind == "cuda" else "combat_reward_gate.c")
    library = build_dir / f"libcombat-reward-gate-{kind}{suffix}"
    source.write_text(_probe_source(kind == "cuda"), encoding="utf-8")
    if kind == "cpu":
        compiler = shutil.which(os.environ.get("CC", "cc")) or os.environ.get("CC", "cc")
        command = [
            compiler,
            "-std=c99",
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-Wall",
            "-Wextra",
            "-shared",
            "-fPIC",
            "-I",
            str(ROOT),
            str(source),
            str(ROOT / "native/game_episode.c"),
            str(ROOT / "native/navigation_episode.c"),
            "-lm",
            "-o",
            str(library),
        ]
    else:
        compiler = shutil.which("nvcc")
        if not compiler:
            raise GateError("CUDA unavailable: nvcc not found")
        command = [
            compiler,
            "-std=c++17",
            "-O2",
            "--fmad=false",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            "-Xcompiler",
            "-ffp-contract=off",
            "-I",
            str(ROOT),
            str(source),
            "-o",
            str(library),
        ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return library


def _load_runner(library: Path):
    lib = C.CDLL(str(library))
    lib.ai2_gate_run.argtypes = [C.c_int32, C.POINTER(GateOutput)]
    lib.ai2_gate_run.restype = C.c_int32
    return lib.ai2_gate_run


def _list2(values) -> list[Any]:
    return [float(v) if isinstance(v, float) else int(v) for v in values]


def output_to_dict(out: GateOutput) -> dict[str, Any]:
    return {
        "scenario_id": int(out.scenario),
        "reset_rc": int(out.reset_rc),
        "step_rc": int(out.step_rc),
        "record_rc": int(out.record_rc),
        "state_bytes_preserved": bool(out.state_preserved),
        "state_tick": int(out.state_tick),
        "before_hash": f"{int(out.before_hash):016x}",
        "after_hash": f"{int(out.after_hash):016x}",
        "terminated": bool(out.terminated),
        "truncated": bool(out.truncated),
        "win_mask": int(out.win_mask),
        "record": {
            "length": int(out.record_length),
            "terminated": bool(out.record_terminated),
            "truncated": bool(out.record_truncated),
            "win_mask": int(out.record_win_mask),
            "timeout": bool(out.record_timeout),
        },
        "approach_delta": float(out.approach_delta),
        "distance": float(out.distance),
        "terms": [_list2(out.terms[i]) for i in range(2)],
        "total": _list2(out.total),
        "returns": _list2(out.returns),
        "term_totals": [_list2(out.term_totals[i]) for i in range(2)],
        "damage_done": _list2(out.damage_done),
        "damage_taken": _list2(out.damage_taken),
        "attack_starts": _list2(out.attack_starts),
        "contacts": _list2(out.contacts),
        "damage_events": _list2(out.damage_events),
        "ko_events": _list2(out.ko_events),
        "clean_win": _list2(out.clean_win),
        "offstage": _list2(out.offstage),
        "walkoff": _list2(out.walkoff),
        "recovery": _list2(out.recovery),
    }


def run_backend(kind: str, build_dir: Path) -> dict[str, Any]:
    runner = _load_runner(_compile_probe(kind, build_dir / kind))
    rows = []
    for scenario in SCENARIOS:
        out = GateOutput()
        rc = runner(scenario.ident, C.byref(out))
        if rc:
            raise GateError(f"{kind} probe failed for {scenario.name}: rc={rc}")
        rows.append(output_to_dict(out))
    return {"status": "passed", "scenario_count": len(rows), "rows": rows}


def first_mismatch(cpu: Any, cuda: Any, path: str = "") -> dict[str, Any] | None:
    if isinstance(cpu, dict) and isinstance(cuda, dict):
        if cpu.keys() != cuda.keys():
            return {
                "path": path or "$",
                "cpu": sorted(cpu.keys()),
                "cuda": sorted(cuda.keys()),
            }
        for key in cpu:
            child = f"{path}.{key}" if path else str(key)
            mismatch = first_mismatch(cpu[key], cuda[key], child)
            if mismatch:
                return mismatch
        return None
    if isinstance(cpu, list) and isinstance(cuda, list):
        if len(cpu) != len(cuda):
            return {"path": path, "cpu": len(cpu), "cuda": len(cuda)}
        for index, (left, right) in enumerate(zip(cpu, cuda)):
            mismatch = first_mismatch(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if isinstance(cpu, float) or isinstance(cuda, float):
        if abs(float(cpu) - float(cuda)) > 0.0:
            return {"path": path, "cpu": cpu, "cuda": cuda}
        return None
    return None if cpu == cuda else {"path": path, "cpu": cpu, "cuda": cuda}


def compare_rows(cpu_rows: list[dict[str, Any]], cuda_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches = []
    for scenario, cpu, cuda in zip(SCENARIOS, cpu_rows, cuda_rows):
        mismatch = first_mismatch(cpu, cuda)
        if mismatch:
            mismatches.append({"scenario": scenario.name, "first_mismatch": mismatch})
    if len(cpu_rows) != len(cuda_rows):
        mismatches.append({"scenario": "$row_count", "cpu": len(cpu_rows), "cuda": len(cuda_rows)})
    return mismatches


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _event_rows(
    cpu_rows: list[dict[str, Any]], cuda_rows: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    rows = []
    cuda_by_id = {row["scenario_id"]: row for row in cuda_rows or []}
    for scenario, cpu in zip(SCENARIOS, cpu_rows):
        cuda = cuda_by_id.get(scenario.ident)
        mismatch = first_mismatch(cpu, cuda) if cuda is not None else None
        rows.append(
            {
                "scenario": scenario.name,
                "components": list(scenario.components),
                "cpu": cpu,
                "cuda": cuda,
                "exact": cuda is None or mismatch is None,
                "first_mismatch": mismatch,
            }
        )
    return rows


def run_gate(
    *,
    output_dir: Path,
    require_cuda: bool = False,
    allow_cuda_skip: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    with tempfile.TemporaryDirectory(prefix="native-combat-reward-gate-build-") as tmp:
        build_dir = Path(tmp)
        cpu = run_backend("cpu", build_dir)
        cuda: dict[str, Any]
        mismatches: list[dict[str, Any]] = []
        if cuda_available():
            cuda = run_backend("cuda", build_dir)
            mismatches = compare_rows(cpu["rows"], cuda["rows"])
            status = "passed" if not mismatches else "failed"
        elif require_cuda and not allow_cuda_skip:
            cuda = {"status": "missing", "reason": "CUDA unavailable: nvcc and torch.cuda are required"}
            status = "failed"
        elif allow_cuda_skip:
            cuda = {"status": "skipped", "reason": "CUDA unavailable on this host; CPU oracle executed"}
            status = "skipped_cuda"
        else:
            cuda = {"status": "missing", "reason": "CUDA unavailable: rerun on CUDA host or allow explicit local skip"}
            status = "failed"

        summary = {
            "status": status,
            "gate": "native_combat_reward_gate",
            "config": {
                "fail_closed": True,
                "require_cuda": bool(require_cuda),
                "allow_cuda_skip": bool(allow_cuda_skip),
                "python": sys.version,
                "platform": platform.platform(),
            },
            "source_hashes": source_hashes(),
            "scenario_names": [case.name for case in SCENARIOS],
            "cpu": {k: v for k, v in cpu.items() if k != "rows"},
            "cuda": {k: v for k, v in cuda.items() if k != "rows"},
            "mismatches": mismatches,
        }
        events = _event_rows(cpu["rows"], cuda.get("rows"))
        _write_new(output_dir / "events.jsonl", "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in events))
        _write_new(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if status == "failed":
            raise GateError(summary["cuda"].get("reason") or json.dumps(mismatches[:1], sort_keys=True))
        return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true", help="fail if real CUDA cannot execute")
    parser.add_argument("--allow-cuda-skip", action="store_true", help="explicit local-only CUDA skip")
    args = parser.parse_args(argv)
    try:
        result = run_gate(
            output_dir=args.output_dir,
            require_cuda=args.require_cuda,
            allow_cuda_skip=args.allow_cuda_skip,
        )
    except GateError as exc:
        print(f"native combat reward gate failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "summary": str(args.output_dir / "summary.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
