import json
import tempfile
import unittest
from pathlib import Path

import tools.validation.audit_navigation_targets as audit


class NavigationTargetGeometryAuditTests(unittest.TestCase):
    def test_classifies_static_ground_as_always_supported(self):
        geometry = audit.FrozenFloorGeometry()

        result = audit.classify_position(audit.Point(4.5, 0.7), geometry, platform_samples=9)

        self.assertEqual("always_geometrically_supported", result.classification)
        self.assertEqual("ground1", result.support_sources[0])
        self.assertFalse(result.proves_reachability)

    def test_classifies_platform_path_point_as_periodically_supported(self):
        geometry = audit.FrozenFloorGeometry()

        result = audit.classify_position(audit.Point(0.0, 0.7), geometry, platform_samples=33)

        self.assertEqual("periodically_supported", result.classification)
        self.assertIn("platform1_sweep", result.support_sources)
        self.assertGreater(result.platform_support_samples, 0)

    def test_classifies_missing_geometry_as_unknown_not_unreachable(self):
        geometry = audit.FrozenFloorGeometry()

        result = audit.classify_position(audit.Point(20.0, -4.0), geometry, platform_samples=33)

        self.assertEqual("no_support_found_unknown", result.classification)
        self.assertFalse(result.proves_reachability)
        self.assertIn("not_unreachable_proof", result.notes)

    def test_audit_positions_counts_grounded_positions_by_conservative_label(self):
        payload = {
            "bounds": {"x_min": -1, "x_max": 5, "y_min": -4, "y_max": 3},
            "stats": {"total_samples": 3},
            "grounded_positions": [
                {"x": 4.5, "y": 0.7},
                {"x": 0.0, "y": 0.7},
                {"x": 20.0, "y": -4.0},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            positions = Path(tmp) / "positions.json"
            positions.write_text(json.dumps(payload))

            result = audit.audit_positions(positions, platform_samples=33)

        self.assertEqual(3, result.total_positions)
        self.assertEqual(1, result.counts["always_geometrically_supported"])
        self.assertEqual(1, result.counts["periodically_supported"])
        self.assertEqual(1, result.counts["no_support_found_unknown"])
        self.assertEqual("not_reachability_proof", result.reachability_claim)


class NavigationTargetTraceAuditTests(unittest.TestCase):
    def test_summarizes_platform_support_on_platform_none_and_hold_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "navigation_seed0_contact_fixture.jsonl"
            trace.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "metadata"}),
                        json.dumps(
                            {
                                "tick": 1,
                                "state": {
                                    "players": [
                                        {
                                            "support": {"kind": "platform", "source": "geometry"},
                                            "on_platform": None,
                                            "position": [0.0, 0.7],
                                            "velocity": [0.1, 0.0],
                                        }
                                    ],
                                    "platforms": [
                                        {
                                            "position": [0.0, 1.0],
                                            "velocity": [0.5, -0.5],
                                            "moving_to_w2": True,
                                            "waypoint1": [1.0, 0.0],
                                            "waypoint2": [-1.0, 2.0],
                                        }
                                    ],
                                },
                                "transition": {"info": {"nav_hold_steps": 1}},
                            }
                        ),
                        json.dumps(
                            {
                                "tick": 2,
                                "state": {
                                    "players": [
                                        {
                                            "support": None,
                                            "on_platform": None,
                                            "position": [4.5, 0.7],
                                            "velocity": [0.0, 0.0],
                                        }
                                    ],
                                    "platforms": [
                                        {
                                            "position": [0.1, 0.9],
                                            "velocity": [0.0, 0.0],
                                            "moving_to_w2": False,
                                            "waypoint1": [1.0, 0.0],
                                            "waypoint2": [-1.0, 2.0],
                                        }
                                    ],
                                },
                                "transition": {"info": {"nav_reached": 1}},
                            }
                        ),
                    ]
                )
            )

            summary = audit.summarize_traces(Path(tmp))

        self.assertEqual(2, summary.records)
        self.assertEqual(1, summary.platform_support_records)
        self.assertEqual(1, summary.platform_support_on_platform_none)
        self.assertEqual(1, summary.platform_hold_records)
        self.assertEqual(1, summary.target_handoff_records)
        self.assertEqual({"True": 1, "False": 1}, summary.platform_direction_counts)

    def test_report_keeps_unknown_label_and_records_provenance(self):
        positions_result = audit.PositionAuditResult(
            positions_file="positions.json",
            positions_sha256="abc",
            total_positions=1,
            counts={"no_support_found_unknown": 1},
            examples={"no_support_found_unknown": [audit.PositionClassification(audit.Point(9, 9), "no_support_found_unknown", [], 0, ["not_unreachable_proof"])]},
            geometry=audit.FrozenFloorGeometry(),
            platform_samples=17,
            reachability_claim="not_reachability_proof",
        )
        trace_summary = audit.TraceSupportSummary(records=0)

        report = audit.render_markdown_report(positions_result, trace_summary)

        self.assertIn("no-support-found/unknown", report)
        self.assertIn("not proof of unreachable", report)
        self.assertIn("settle speed threshold", report)
        self.assertIn("positions.json SHA-256", report)
        self.assertIn("geometry tolerance", report)


if __name__ == "__main__":
    unittest.main()
