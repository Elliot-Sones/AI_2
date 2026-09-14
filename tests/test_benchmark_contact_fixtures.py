"""Tests for contact fixture evidence summaries.

These tests intentionally exercise summary logic over replay records. They do
not teleport a live environment or certify fixture labels as physical events.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark_contact_fixtures as fixtures
import benchmark_replay as replay


class ContactFixtureEvidenceTests(unittest.TestCase):
    def test_labels_alone_do_not_satisfy_real_contact_evidence_gates(self):
        records = [
            replay_record(
                0,
                labels=["platformlanding", "platformriding", "platformreversal", "edge"],
                verified_events=["platformlanding", "platformriding", "platformreversal"],
                contacts=[on_platform_contact("inactive-support-field")],
                moving_to_w2=True,
            )
        ]

        summary = fixtures.summarize_records(records)

        self.assertEqual(summary["found_contact_evidence_gates"], [])
        self.assertEqual(
            summary["missing_contact_evidence_gates"],
            [
                "platformlanding",
                "platformriding",
                "platformreversal",
                "contact_underside",
                "contact_side",
                "contact_edge",
            ],
        )
        self.assertEqual(summary["contact_evidence"]["platformlanding"], [])

    def test_malformed_arbiter_contact_is_reported_and_cannot_satisfy_gate(self):
        malformed = arbiter_contact("side", normal=[1.0, 0.0])
        malformed.pop("platform_body_id")
        records = [replay_record(0, contacts=[malformed])]

        summary = fixtures.summarize_records(records)

        self.assertNotIn("contact_side", summary["found_contact_evidence_gates"])
        self.assertIn("contact_side", summary["missing_contact_evidence_gates"])
        self.assertEqual(summary["invalid_contact_evidence"][0]["reason"], "missing_field")
        self.assertEqual(summary["invalid_contact_evidence"][0]["field"], "platform_body_id")

    def test_opponent_arbiter_contacts_are_ignored_before_controlled_player_validation(self):
        opponent = arbiter_contact("opponent-side", normal=[1.0, 0.0], points=[{"point_a": [0.5, 0.0]}])
        opponent["player_agent_id"] = 1
        opponent["shape_roles"][0]["agent_id"] = 1
        records = [replay_record(0, contacts=[opponent])]

        summary = fixtures.summarize_records(records)

        self.assertEqual(summary["invalid_contact_evidence"], [])
        self.assertNotIn("contact_side", summary["found_contact_evidence_gates"])

    def test_edge_controller_targets_moving_platform_endpoint_not_center(self):
        state = {
            "players": [{"agent_id": 0, "grounded": True, "position": [10.1, 1.0], "support": None}],
            "moving_platforms": [
                {
                    "id": 1,
                    "position": [10.0, 0.0],
                    "velocity": [0.2, 0.0],
                    "moving_to_w2": True,
                    "geometry": {"world_bb": {"left": 9.5, "right": 10.5, "bottom": -0.2, "top": 0.2}},
                }
            ],
            "wrapper": {"chain": []},
        }
        controller = fixtures.AdaptiveContactController(
            "navigation",
            action_space=type("Space", (), {"shape": (10,), "dtype": int})(),
            target="edge",
        )

        label, action = controller.next_action(state, tick=12)

        self.assertEqual(label, "adaptive_edge_jump")
        self.assertEqual(list(action), [0, 0, 0, 1, 1, 0, 0, 0, 0, 0])

    def test_platform_events_require_contact_on_exact_supported_platform(self):
        records = [
            replay_record(
                0,
                contacts=[geometric_contact("wrong-platform", platform_id=2)],
                support_id=1,
                platforms=[
                    moving_platform(1, moving_to_w2=True, velocity=[0.2, 0.0]),
                    moving_platform(2, moving_to_w2=True, velocity=[0.2, 0.0]),
                ],
            )
        ]

        summary = fixtures.summarize_records(records)

        self.assertNotIn("platformlanding", summary["found_contact_evidence_gates"])
        self.assertIn("platformlanding", summary["missing_contact_evidence_gates"])

    def test_platform_reversal_evidence_requires_same_support_and_current_contact_platform(self):
        records = [
            replay_record(
                0,
                contacts=[geometric_contact("previous", platform_id=1)],
                support_id=1,
                moving_to_w2=True,
                velocity=[0.2, 0.0],
            ),
            replay_record(
                1,
                contacts=[geometric_contact("wrong-current-contact", platform_id=2)],
                support_id=1,
                platforms=[
                    moving_platform(1, moving_to_w2=False, velocity=[-0.2, 0.0]),
                    moving_platform(2, moving_to_w2=False, velocity=[-0.2, 0.0]),
                ],
            ),
        ]

        summary = fixtures.summarize_records(records)

        self.assertNotIn("platformreversal", summary["found_contact_evidence_gates"])

    def test_platform_events_do_not_cross_reset_boundary(self):
        records = [
            replay_record(
                0,
                contacts=[geometric_contact("previous")],
                support_id=1,
                moving_to_w2=True,
                velocity=[0.2, 0.0],
            ),
            replay_record(
                1,
                contacts=[geometric_contact("after-reset")],
                support_id=1,
                moving_to_w2=False,
                velocity=[-0.2, 0.0],
                post_reset_state=True,
            ),
        ]

        summary = fixtures.summarize_records(records)

        self.assertNotIn("platformreversal", summary["found_contact_evidence_gates"])

    def test_geometric_and_arbiter_contacts_prove_moving_platform_sequence(self):
        records = [
            replay_record(
                0,
                verified_events=["platformlanding"],
                contacts=[geometric_contact("landing")],
                moving_to_w2=True,
                velocity=[0.2, 0.0],
            ),
            replay_record(
                1,
                verified_events=["platformlanding", "platformriding"],
                contacts=[arbiter_contact("ride", normal=[0.0, 1.0])],
                moving_to_w2=True,
                velocity=[0.2, 0.0],
            ),
            replay_record(
                2,
                verified_events=["platformlanding", "platformriding", "platformreversal"],
                contacts=[arbiter_contact("reverse", normal=[0.0, 1.0])],
                moving_to_w2=False,
                velocity=[-0.2, 0.0],
            ),
        ]

        summary = fixtures.summarize_records(records)

        self.assertIn("platformlanding", summary["found_contact_evidence_gates"])
        self.assertIn("platformriding", summary["found_contact_evidence_gates"])
        self.assertIn("platformreversal", summary["found_contact_evidence_gates"])
        self.assertEqual(summary["contact_evidence"]["platformlanding"][0]["source"], "geometric")
        self.assertEqual(summary["contact_evidence"]["platformriding"][0]["source"], "arbiter")
        self.assertEqual(summary["contact_evidence"]["platformreversal"][0]["previous_moving_to_w2"], True)
        self.assertEqual(summary["contact_evidence"]["platformreversal"][0]["moving_to_w2"], False)

    def test_arbiter_normals_prove_underside_side_and_edge_contacts(self):
        records = [
            replay_record(0, contacts=[arbiter_contact("underside", normal=[0.0, -1.0], points=[{"point_a": [0.0, -0.45]}])]),
            replay_record(1, contacts=[arbiter_contact("side", normal=[1.0, 0.0], points=[{"point_a": [0.49, 0.0]}])]),
            replay_record(2, contacts=[arbiter_contact("edge", normal=[0.7, 0.7], points=[{"point_a": [0.5, 0.5]}])]),
        ]

        summary = fixtures.summarize_records(records)

        self.assertIn("contact_underside", summary["found_contact_evidence_gates"])
        self.assertIn("contact_side", summary["found_contact_evidence_gates"])
        self.assertIn("contact_edge", summary["found_contact_evidence_gates"])
        self.assertEqual(summary["contact_evidence"]["contact_underside"][0]["normal"], [0.0, -1.0])
        self.assertEqual(summary["contact_evidence"]["contact_side"][0]["normal"], [1.0, 0.0])
        self.assertEqual(summary["contact_evidence"]["contact_edge"][0]["contact_id"], "edge")

    def test_edge_contact_requires_endpoint_geometry_not_diagonal_normal_only(self):
        records = [replay_record(0, contacts=[arbiter_contact("center-diagonal", normal=[0.7, 0.7], points=[{"point_a": [0.0, 0.0]}])])]

        summary = fixtures.summarize_records(records)

        self.assertNotIn("contact_edge", summary["found_contact_evidence_gates"])
        self.assertIn("contact_edge", summary["missing_contact_evidence_gates"])

    def test_main_summary_missing_gates_include_contact_evidence(self):
        records = [replay_record(0, labels=["edge"], contacts=[arbiter_contact("side", normal=[1.0, 0.0], points=[{"point_a": [0.5, 0.0]}])])]

        summary = fixtures.summarize_records(records)

        self.assertIn("contact_side", summary["found_contact_evidence_gates"])
        self.assertIn("contact_edge", summary["missing_contact_evidence_gates"])
        self.assertIn("platformlanding", summary["missing_event_gates"])


def replay_record(
    tick: int,
    *,
    labels=(),
    verified_events=(),
    contacts=(),
    support_id=1,
    platforms=None,
    moving_to_w2=True,
    velocity=None,
    post_reset_state=False,
) -> dict:
    velocity = [0.2, 0.0] if velocity is None else velocity
    platform_records = platforms if platforms is not None else [moving_platform(1, moving_to_w2=moving_to_w2, velocity=velocity)]
    state = {
        "contacts": list(contacts),
        "players": [{"agent_id": 0, "grounded": True, "support": {"kind": "platform", "id": support_id}}],
        "moving_platforms": platform_records,
        "wrapper": {"chain": []},
        "coverage": replay.coverage_ledger(labels, verified_events=verified_events),
    }
    return {
        "type": "record",
        "schema": replay.REPLAY_SCHEMA,
        "tick": tick,
        "action": [0] * 10,
        "transition": {"observation": None, "reward": 0.0, "terminated": False, "truncated": False, "info": {}, "post_reset_state": post_reset_state},
        "state": state,
        "discrete_events": [],
    }


def moving_platform(platform_id: int, *, moving_to_w2=True, velocity=None) -> dict:
    velocity = [0.2, 0.0] if velocity is None else velocity
    return {
        "id": platform_id,
        "name": f"moving-{platform_id}",
        "body_id": f"platform:{platform_id}",
        "position": [0.0, 0.0],
        "velocity": velocity,
        "moving_to_w2": moving_to_w2,
        "waypoint1": [-0.5, -0.5],
        "waypoint2": [0.5, 0.5],
        "geometry": {
            "world_bb": {"left": -0.5, "right": 0.5, "bottom": -0.5, "top": 0.5},
            "endpoint_tolerance": 0.05,
        },
    }


def geometric_contact(contact_id: str, *, platform_id=1) -> dict:
    return {
        "kind": "geometric_platform_support",
        "source": "is_on_floor_bb",
        "player_agent_id": 0,
        "platform_id": platform_id,
        "platform_body_id": f"platform:{platform_id}",
        "contact_id": contact_id,
    }


def on_platform_contact(contact_id: str) -> dict:
    return {
        "kind": "player_platform_support",
        "source": "player.on_platform",
        "player_agent_id": 0,
        "platform_id": 1,
        "contact_id": contact_id,
    }


def arbiter_contact(contact_id: str, *, normal, points=None) -> dict:
    return {
        "kind": "pymunk_arbiter",
        "source": "body.each_arbiter",
        "player_agent_id": 0,
        "platform_id": 1,
        "platform_body_id": "platform:1",
        "contact_id": contact_id,
        "shape_roles": [
            {"order": 0, "role": "player", "agent_id": 0},
            {"order": 1, "role": "platform", "platform_id": 1, "platform_body_id": "platform:1"},
        ],
        "normal_role": "shape0_to_shape1_raw",
        "normal": normal,
        "points": list(points or []),
        "total_impulse": [0.0, 0.0],
        "total_ke": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
