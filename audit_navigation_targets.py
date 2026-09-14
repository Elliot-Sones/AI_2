#!/usr/bin/env python3
"""CPU-only navigation target geometry audit.

This diagnostic classifies historical ``positions.json`` coordinates against the
frozen game's explicit floor geometry and moving-platform sweep. It does not run
policy inference, training, CUDA, or mutate game code. A missing support match is
reported as unknown, not unreachable, because controller failure or coarse
geometry coverage is not a mathematical reachability proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_FROZEN_ROOT = (
    DEFAULT_ROOT.parent
    / "AI_2-learning-validation-20260905"
    / "baselines/navigation-repair-20260905/payload/variants/corrected"
)
DEFAULT_POSITIONS = DEFAULT_FROZEN_ROOT / "positions.json"
DEFAULT_STAGE_SOURCE = DEFAULT_FROZEN_ROOT / "UTMIST-AI2-main/environment/environment.py"
DEFAULT_TRACE_DIR = (
    DEFAULT_ROOT
    / "baselines/throughput-scaling-20260906/cloud/replacement-session2/retrieved/artifacts/e0-replay/adaptive-navigation/traces"
)
DEFAULT_REPORT = DEFAULT_ROOT / "baselines/throughput-scaling-20260906/NAVIGATION-TARGET-AUDIT.md"
DEFAULT_GEOMETRY_TOLERANCE = 1e-9

CLASS_ALWAYS = "always_geometrically_supported"
CLASS_PERIODIC = "periodically_supported"
CLASS_UNKNOWN = "no_support_found_unknown"


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    left: float
    right: float
    top: float
    bottom: float

    def intersects(self, other: "Rect", tol: float = 1e-9) -> bool:
        return not (
            self.right < other.left - tol
            or other.right < self.left - tol
            or self.bottom < other.top - tol
            or other.bottom < self.top - tol
        )


@dataclass(frozen=True)
class FrozenFloorGeometry:
    """Constants mirrored from frozen ``environment.py`` floor checks.

    Player shape: ``pymunk.Poly.create_box(size=(0.87, 1.0))``.
    ``Ground`` rectangles: x +/- ``width_ground/4`` and y top ``body_y - 0.15``.
    ``Stage`` platform: ``create_box(width=2, height=1*0.1+0.1)`` and
    ``Player.is_on_floor`` platform x-gate center +/- 1.25 with player above
    ``platform_y - 0.01``.
    """

    player_width: float = 0.87
    player_height: float = 1.0
    ground1_center: Point = Point(4.5, 1.0)
    ground2_center: Point = Point(-4.5, 3.0)
    ground_width: float = 10.0
    ground_top_offset: float = -0.15
    ground_bottom_offset: float = 4.0
    platform_waypoint1: Point = Point(1.0, 0.0)
    platform_waypoint2: Point = Point(-1.0, 2.0)
    platform_width: float = 2.0
    platform_height: float = 0.2
    platform_x_gate_half_width: float = 1.25
    platform_y_gate_offset: float = -0.01
    settle_speed_threshold: float = 1.0

    @property
    def player_half_width(self) -> float:
        return self.player_width / 2.0

    @property
    def player_half_height(self) -> float:
        return self.player_height / 2.0

    def player_bb(self, point: Point) -> Rect:
        return Rect(
            point.x - self.player_half_width,
            point.x + self.player_half_width,
            point.y - self.player_half_height,
            point.y + self.player_half_height,
        )

    def ground_bb(self, center: Point) -> Rect:
        half = self.ground_width / 4.0
        return Rect(
            center.x - half,
            center.x + half,
            center.y + self.ground_top_offset,
            center.y + self.ground_bottom_offset,
        )

    def platform_bb(self, center: Point) -> Rect:
        return Rect(
            center.x - self.platform_width / 2.0,
            center.x + self.platform_width / 2.0,
            center.y - self.platform_height / 2.0,
            center.y + self.platform_height / 2.0,
        )


@dataclass(frozen=True)
class PositionClassification:
    point: Point
    classification: str
    support_sources: list[str]
    platform_support_samples: int
    notes: list[str]
    proves_reachability: bool = False


@dataclass(frozen=True)
class PositionAuditResult:
    positions_file: str
    positions_sha256: str
    total_positions: int
    counts: dict[str, int]
    examples: dict[str, list[PositionClassification]]
    geometry: FrozenFloorGeometry
    platform_samples: int
    reachability_claim: str
    stage_source_file: str = ""
    stage_source_sha256: str = ""
    geometry_tolerance: float = DEFAULT_GEOMETRY_TOLERANCE


@dataclass
class TraceSupportSummary:
    records: int = 0
    trace_files: int = 0
    platform_support_records: int = 0
    platform_support_on_platform_none: int = 0
    platform_hold_records: int = 0
    target_handoff_records: int = 0
    target_handoff_on_platform_support: int = 0
    nav_touched_on_platform_support: int = 0
    platform_direction_counts: dict[str, int] = field(default_factory=dict)
    platform_progress_min: float | None = None
    platform_progress_max: float | None = None
    max_observed_platform_speed: float = 0.0
    examples: dict[str, dict[str, Any]] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ground_support(point: Point, geometry: FrozenFloorGeometry, *, tolerance: float = DEFAULT_GEOMETRY_TOLERANCE) -> list[str]:
    player_bb = geometry.player_bb(point)
    supports = []
    for name, center in (
        ("ground1", geometry.ground1_center),
        ("ground2", geometry.ground2_center),
    ):
        if player_bb.intersects(geometry.ground_bb(center), tol=tolerance) and point.y <= center.y + tolerance:
            supports.append(name)
    return supports


def _platform_support(
    point: Point,
    center: Point,
    geometry: FrozenFloorGeometry,
    *,
    tolerance: float = DEFAULT_GEOMETRY_TOLERANCE,
) -> bool:
    player_bb = geometry.player_bb(point)
    return bool(
        player_bb.intersects(geometry.platform_bb(center), tol=tolerance)
        and point.y <= center.y + geometry.platform_y_gate_offset + tolerance
        and (center.x - geometry.platform_x_gate_half_width - tolerance)
        <= point.x
        <= (center.x + geometry.platform_x_gate_half_width + tolerance)
    )


def platform_sweep_points(geometry: FrozenFloorGeometry, samples: int) -> list[Point]:
    if samples < 2:
        raise ValueError("platform_samples must be >= 2")
    w1 = geometry.platform_waypoint1
    w2 = geometry.platform_waypoint2
    return [
        Point(w1.x + (w2.x - w1.x) * index / (samples - 1), w1.y + (w2.y - w1.y) * index / (samples - 1))
        for index in range(samples)
    ]


def classify_position(
    point: Point,
    geometry: FrozenFloorGeometry | None = None,
    *,
    platform_samples: int = 401,
    tolerance: float = DEFAULT_GEOMETRY_TOLERANCE,
) -> PositionClassification:
    geometry = geometry or FrozenFloorGeometry()
    supports = _ground_support(point, geometry, tolerance=tolerance)
    notes = ["not_reachability_proof"]
    if supports:
        return PositionClassification(point, CLASS_ALWAYS, supports, 0, notes)

    platform_hits = sum(
        1
        for center in platform_sweep_points(geometry, platform_samples)
        if _platform_support(point, center, geometry, tolerance=tolerance)
    )
    if platform_hits:
        return PositionClassification(point, CLASS_PERIODIC, ["platform1_sweep"], platform_hits, notes)

    return PositionClassification(point, CLASS_UNKNOWN, [], 0, ["not_unreachable_proof", "not_reachability_proof"])


def load_grounded_positions(path: Path) -> list[Point]:
    payload = json.loads(path.read_text())
    positions = payload.get("grounded_positions")
    if not isinstance(positions, list):
        raise ValueError(f"{path} does not contain a grounded_positions list")
    result = []
    for item in positions:
        result.append(Point(float(item["x"]), float(item["y"])))
    return result


def audit_positions(
    path: Path,
    *,
    stage_source: Path = DEFAULT_STAGE_SOURCE,
    platform_samples: int = 401,
    example_limit: int = 8,
    tolerance: float = DEFAULT_GEOMETRY_TOLERANCE,
) -> PositionAuditResult:
    geometry = FrozenFloorGeometry()
    counts = {CLASS_ALWAYS: 0, CLASS_PERIODIC: 0, CLASS_UNKNOWN: 0}
    examples: dict[str, list[PositionClassification]] = {CLASS_ALWAYS: [], CLASS_PERIODIC: [], CLASS_UNKNOWN: []}
    positions = load_grounded_positions(path)
    for point in positions:
        result = classify_position(point, geometry, platform_samples=platform_samples, tolerance=tolerance)
        counts[result.classification] += 1
        if len(examples[result.classification]) < example_limit:
            examples[result.classification].append(result)
    return PositionAuditResult(
        positions_file=str(path),
        positions_sha256=sha256_file(path),
        total_positions=len(positions),
        counts=counts,
        examples=examples,
        geometry=geometry,
        platform_samples=platform_samples,
        reachability_claim="not_reachability_proof",
        stage_source_file=str(stage_source),
        stage_source_sha256=sha256_file(stage_source) if stage_source.exists() else "missing",
        geometry_tolerance=tolerance,
    )


def platform_progress(position: Iterable[float], waypoint1: Iterable[float], waypoint2: Iterable[float]) -> float:
    px, py = [float(v) for v in position]
    x1, y1 = [float(v) for v in waypoint1]
    x2, y2 = [float(v) for v in waypoint2]
    sx = x2 - x1
    sy = y2 - y1
    denom = sx * sx + sy * sy
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, ((px - x1) * sx + (py - y1) * sy) / denom))


def platform_speed_at_progress(progress: float) -> float:
    return math.sin(max(0.0, min(1.0, progress)) * math.pi) + 0.03


def theoretical_max_platform_speed() -> float:
    return 1.03


def _remember_example(summary: TraceSupportSummary, key: str, trace: Path, lineno: int, record: dict[str, Any]) -> None:
    if key in summary.examples:
        return
    state = record.get("state") or {}
    players = state.get("players") or [{}]
    player = players[0] if players else {}
    platforms = state.get("platforms") or state.get("moving_platforms") or []
    info = (record.get("transition") or {}).get("info") or {}
    summary.examples[key] = {
        "file": trace.name,
        "line": lineno,
        "tick": record.get("tick"),
        "player_position": player.get("position"),
        "player_velocity": player.get("velocity"),
        "support": player.get("support"),
        "on_platform": player.get("on_platform"),
        "platform": platforms[0] if platforms else None,
        "info": {name: value for name, value in info.items() if str(name).startswith("nav")},
    }


def summarize_traces(trace_dir: Path) -> TraceSupportSummary:
    summary = TraceSupportSummary()
    if not trace_dir.exists():
        return summary
    traces = sorted(trace_dir.glob("navigation_seed*_contact_fixture.jsonl"))
    summary.trace_files = len(traces)
    for trace in traces:
        with trace.open() as handle:
            for lineno, line in enumerate(handle, 1):
                record = json.loads(line)
                if record.get("type") == "metadata":
                    continue
                summary.records += 1
                state = record.get("state") or {}
                players = state.get("players") or []
                player = players[0] if players else {}
                support = player.get("support")
                platform_support = bool(isinstance(support, dict) and support.get("kind") == "platform")
                on_platform_none = player.get("on_platform") is None
                info = (record.get("transition") or {}).get("info") or {}
                if platform_support:
                    summary.platform_support_records += 1
                    _remember_example(summary, "platform_support", trace, lineno, record)
                    if on_platform_none:
                        summary.platform_support_on_platform_none += 1
                        _remember_example(summary, "platform_support_on_platform_none", trace, lineno, record)
                    if int(info.get("nav_hold_steps") or 0) > 0:
                        summary.platform_hold_records += 1
                        _remember_example(summary, "platform_hold", trace, lineno, record)
                    if int(info.get("nav_reached") or 0) == 1:
                        summary.target_handoff_on_platform_support += 1
                        _remember_example(summary, "target_handoff_on_platform_support", trace, lineno, record)
                    if int(info.get("nav_touched") or 0) == 1:
                        summary.nav_touched_on_platform_support += 1
                        _remember_example(summary, "nav_touched_on_platform_support", trace, lineno, record)
                if int(info.get("nav_reached") or 0) == 1:
                    summary.target_handoff_records += 1
                    _remember_example(summary, "target_handoff", trace, lineno, record)

                platforms = state.get("platforms") or state.get("moving_platforms") or []
                if platforms:
                    platform = platforms[0]
                    direction_key = str(bool(platform.get("moving_to_w2")))
                    summary.platform_direction_counts[direction_key] = summary.platform_direction_counts.get(direction_key, 0) + 1
                    if all(key in platform for key in ("position", "waypoint1", "waypoint2")):
                        progress = platform_progress(platform["position"], platform["waypoint1"], platform["waypoint2"])
                        summary.platform_progress_min = progress if summary.platform_progress_min is None else min(summary.platform_progress_min, progress)
                        summary.platform_progress_max = progress if summary.platform_progress_max is None else max(summary.platform_progress_max, progress)
                    velocity = platform.get("velocity")
                    if isinstance(velocity, list) and len(velocity) >= 2:
                        speed = math.hypot(float(velocity[0]), float(velocity[1]))
                        summary.max_observed_platform_speed = max(summary.max_observed_platform_speed, speed)
    return summary


def _classification_label(label: str) -> str:
    return {
        CLASS_ALWAYS: "always geometrically supported",
        CLASS_PERIODIC: "periodically supported",
        CLASS_UNKNOWN: "no-support-found/unknown",
    }[label]


def _format_examples(items: list[PositionClassification]) -> str:
    if not items:
        return "_none captured_"
    return "\n".join(
        f"- ({item.point.x:.3f}, {item.point.y:.3f}) — {', '.join(item.support_sources) or 'no support source'}, "
        f"platform_samples={item.platform_support_samples}, notes={','.join(item.notes)}"
        for item in items
    )


def _compact(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(f"{float(item):.6g}" for item in value[:2]) + "]"
    return str(value)


def _format_trace_examples(summary: TraceSupportSummary) -> str:
    if not summary.examples:
        return "_none captured_"
    lines = []
    for key in sorted(summary.examples):
        example = summary.examples[key]
        platform = example.get("platform") or {}
        info = example.get("info") or {}
        nav_flags = {
            name: info.get(name)
            for name in ("nav_hold_steps", "nav_reached", "nav_touched", "nav_success")
            if name in info
        }
        support = example.get("support") or {}
        support_label = support.get("kind") if isinstance(support, dict) else support
        lines.append(
            f"- `{key}`: `{example.get('file')}:{example.get('line')}`, tick=`{example.get('tick')}`, "
            f"player_pos=`{_compact(example.get('player_position'))}`, player_vel=`{_compact(example.get('player_velocity'))}`, "
            f"support=`{support_label}`, on_platform=`{example.get('on_platform')}`, "
            f"platform_pos=`{_compact(platform.get('position'))}`, platform_vel=`{_compact(platform.get('velocity'))}`, "
            f"nav_flags=`{json.dumps(nav_flags, sort_keys=True)}`"
        )
    return "\n".join(lines)


def render_markdown_report(positions: PositionAuditResult, traces: TraceSupportSummary) -> str:
    geometry = positions.geometry
    total = max(positions.total_positions, 1)
    max_platform_speed = theoretical_max_platform_speed()
    support_none_pct = (
        100.0 * traces.platform_support_on_platform_none / traces.platform_support_records
        if traces.platform_support_records
        else 0.0
    )
    lines = [
        "# Navigation target geometry audit",
        "",
        "Date: 2026-09-06",
        "",
        "Scope: CPU-only diagnostic of historical `positions.json` target coordinates against frozen static floor geometry, sampled moving-platform sweep, and existing adaptive E0 navigation traces. No policy calls, no training, no GPU, and no game/reward/action code changes.",
        "",
        "## Verdict",
        "",
        "This audit distinguishes geometry support classes only. `no-support-found/unknown` is not proof of unreachable; it means this bounded static/swept-geometry audit did not find support. Reachability still requires a controller/trajectory proof in the frozen game.",
        "",
        "## Provenance",
        "",
        f"- positions file: `{positions.positions_file}`",
        f"- positions.json SHA-256: `{positions.positions_sha256}`",
        f"- frozen environment source: `{positions.stage_source_file}`",
        f"- frozen environment SHA-256: `{positions.stage_source_sha256}`",
        f"- platform sweep samples: `{positions.platform_samples}`",
        f"- geometry tolerance: `{positions.geometry_tolerance}`",
        "- frozen source semantics used:",
        "  - `Ground.__init__`: `ground1=(4.5,1,width=10)`, `ground2=(-4.5,3,width=10)`, x half-width is `width/4`, top y is `body_y - 0.15`.",
        "  - `Stage.__init__`: platform box width `2.0`, height `0.2`, waypoints `(1,0)` to `(-1,2)`.",
        "  - `Player.is_on_floor`: support comes from player BB intersecting ground/platform BB, with platform y gate `player_y <= platform_y - 0.01` and x gate `platform_x +/- 1.25`.",
        "  - player shape: box `(0.87, 1.0)`.",
        "",
        "## Coordinate counts",
        "",
        "| classification | count | percent | interpretation |",
        "| --- | ---: | ---: | --- |",
    ]
    for label in (CLASS_ALWAYS, CLASS_PERIODIC, CLASS_UNKNOWN):
        count = positions.counts.get(label, 0)
        lines.append(f"| {_classification_label(label)} | {count} | {100.0 * count / total:.2f}% | {classification_interpretation(label)} |")
    lines.extend(
        [
            "",
            "## Examples",
            "",
            "### Always geometrically supported",
            "",
            _format_examples(positions.examples.get(CLASS_ALWAYS, [])),
            "",
            "### Periodically supported",
            "",
            _format_examples(positions.examples.get(CLASS_PERIODIC, [])),
            "",
            "### no-support-found/unknown",
            "",
            _format_examples(positions.examples.get(CLASS_UNKNOWN, [])),
            "",
            "## Moving-platform speed vs settle threshold",
            "",
            f"- theoretical maximum platform speed from `sin(progress*pi)+0.03`: `{max_platform_speed:.6f}`",
            f"- wrapper settle speed threshold: `{geometry.settle_speed_threshold:.6f}`",
            f"- platform max exceeds settle speed by `{max_platform_speed - geometry.settle_speed_threshold:.6f}` if `on_platform` is not used for support-relative subtraction.",
            "- Current source/trace evidence therefore matters: geometric riding can exist while `player.on_platform` remains `None`, causing settle checks to use world velocity rather than platform-relative velocity.",
            "",
            "## Existing adaptive E0 trace support / hold stats",
            "",
            f"- trace files scanned: `{traces.trace_files}`",
            f"- non-metadata records scanned: `{traces.records}`",
            f"- platform-support records: `{traces.platform_support_records}`",
            f"- platform-support records with `on_platform is None`: `{traces.platform_support_on_platform_none}` ({support_none_pct:.2f}%)",
            f"- platform-support records with `nav_hold_steps > 0`: `{traces.platform_hold_records}`",
            f"- total `nav_reached` target-handoff records: `{traces.target_handoff_records}`",
            f"- `nav_reached` while geometrically platform-supported: `{traces.target_handoff_on_platform_support}`",
            f"- `nav_touched` while geometrically platform-supported: `{traces.nav_touched_on_platform_support}`",
            f"- platform direction coverage: `{json.dumps(traces.platform_direction_counts, sort_keys=True)}`",
            f"- platform path progress coverage: `{traces.platform_progress_min}` to `{traces.platform_progress_max}`",
            f"- max observed platform speed in traces: `{traces.max_observed_platform_speed:.6f}`",
            "",
            "## Trace examples",
            "",
            _format_trace_examples(traces),
            "",
            "## Limits",
            "",
            "- This is not an unreachable-target proof.",
            "- This is not a policy success/failure diagnosis.",
            "- This does not replace Pymunk collision/contact/solver validation.",
            "- The platform sweep is sampled geometry over the frozen waypoint path; if a future gate needs mathematical continuity, solve the interval geometry analytically instead of widening labels.",
            "",
            "## Reusable takeaway",
            "",
            "Use these counts to separate target-coordinate support questions from policy-capacity diagnostics. Coordinates marked `always geometrically supported` or `periodically supported` are plausible target candidates under frozen floor geometry; coordinates marked `no-support-found/unknown` need a frozen-game trajectory diagnosis before anyone calls them unreachable.",
            "",
        ]
    )
    return "\n".join(lines)


def classification_interpretation(label: str) -> str:
    if label == CLASS_ALWAYS:
        return "Supported by static frozen ground geometry independent of platform phase."
    if label == CLASS_PERIODIC:
        return "Supported only for at least one sampled moving-platform phase."
    if label == CLASS_UNKNOWN:
        return "No static/swept support found; bounded diagnostic result, not unreachable proof."
    raise ValueError(f"unknown classification {label!r}")


def to_jsonable_positions(result: PositionAuditResult) -> dict[str, Any]:
    return {
        "positions_file": result.positions_file,
        "positions_sha256": result.positions_sha256,
        "total_positions": result.total_positions,
        "counts": result.counts,
        "platform_samples": result.platform_samples,
        "reachability_claim": result.reachability_claim,
        "examples": {
            key: [
                {
                    "x": item.point.x,
                    "y": item.point.y,
                    "classification": item.classification,
                    "support_sources": item.support_sources,
                    "platform_support_samples": item.platform_support_samples,
                    "notes": item.notes,
                    "proves_reachability": item.proves_reachability,
                }
                for item in value
            ]
            for key, value in result.examples.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions-file", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--stage-source", type=Path, default=DEFAULT_STAGE_SOURCE)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--platform-samples", type=int, default=401)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    positions = audit_positions(args.positions_file, stage_source=args.stage_source, platform_samples=args.platform_samples)
    traces = summarize_traces(args.trace_dir)
    report = render_markdown_report(positions, traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "positions": to_jsonable_positions(positions),
                    "traces": {
                        "records": traces.records,
                        "trace_files": traces.trace_files,
                        "platform_support_records": traces.platform_support_records,
                        "platform_support_on_platform_none": traces.platform_support_on_platform_none,
                        "platform_hold_records": traces.platform_hold_records,
                        "target_handoff_records": traces.target_handoff_records,
                        "target_handoff_on_platform_support": traces.target_handoff_on_platform_support,
                        "nav_touched_on_platform_support": traces.nav_touched_on_platform_support,
                        "platform_direction_counts": traces.platform_direction_counts,
                        "platform_progress_min": traces.platform_progress_min,
                        "platform_progress_max": traces.platform_progress_max,
                        "max_observed_platform_speed": traces.max_observed_platform_speed,
                        "examples": traces.examples,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    print(f"wrote {args.output}")
    if args.json_output is not None:
        print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
