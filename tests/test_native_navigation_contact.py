"""Tests for the native navigation contact step (native/navigation_contact.h).

Pure tests always run (compile the C reference with cc). The exactness test
against the real frozen game runs only when the frozen payload and a Torch-free
Pymunk 6.2.1 runtime are importable; it is skipped, not passed, otherwise.
"""
from __future__ import annotations

import ctypes
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))

FROZEN = Path(os.environ.get("AI2_FROZEN_SOURCE_ROOT") or "/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected")


def build_reference(tmp: Path, contract: str = "on"):
    import native_navigation_contact as m

    if shutil.which(os.environ.get("CC") or "cc") is None:
        raise unittest.SkipTest("no C compiler")
    lib = tmp / f"libcontact-{contract}.so"
    subprocess.run([os.environ.get("CC") or "cc", "-std=c99", "-O2", f"-ffp-contract={contract}", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-shared", "-fPIC", "-o", str(lib), str(ROOT / "native" / "navigation_contact.c")], check=True)
    return m, m.ContactReference(lib)


class ContactReferencePureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ai2-contact-"))
        self.m, self.ref = build_reference(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_struct_layouts_match_between_python_and_c(self):
        # The constructor raises on a size mismatch; assert the sizes are also non-trivial.
        self.assertEqual(self.ref.lib.ai2c_world_size(), ctypes.sizeof(self.m.World))
        self.assertGreater(ctypes.sizeof(self.m.World), 4096)

    def test_hash_pair_matches_chipmunk_macro(self):
        coef = 3344921057
        mask = (1 << 64) - 1
        for a, b in ((0, 1), (3, 0), (7, 12345678901234), (mask, 1)):
            expected = ((a * coef) & mask) ^ ((b * coef) & mask)
            self.assertEqual(self.ref.hash_pair(a, b), expected)
        # Symmetric, as in Chipmunk (documented source of rare false positives).
        self.assertEqual(self.ref.hash_pair(3, 1), self.ref.hash_pair(1, 3))

    def test_step_coefficients_use_space_constants(self):
        import math

        world = self.m.World()
        ctypes.memset(ctypes.byref(world), 0, ctypes.sizeof(self.m.World))
        world.bias = 0.0017970074436457143  # pow((double)0.9f, 60) as stored by Pymunk
        world.damping = 1.0
        bias_coef, damping_dt = self.ref.coefficients(world, 1.0 / 30.0)
        self.assertEqual(damping_dt, 1.0)
        self.assertEqual(bias_coef, 1.0 - math.pow(world.bias, 1.0 / 30.0))

    def _box_world(self):
        """A unit-mass box dropped onto a static ground polygon, mirroring the frozen scene's shapes."""
        m, ref = self.m, self.ref
        world = m.World()
        ctypes.memset(ctypes.byref(world), 0, ctypes.sizeof(m.World))
        world.iterations = 10
        world.gravity.x, world.gravity.y = 0.0, 17.808
        world.damping = 1.0
        world.slop = 0.10000000149011612
        world.bias = 0.0017970074436457143
        world.persistence = 3
        world.body_count = 2
        ground, box = world.bodies[0], world.bodies[1]
        ground.type = 2; ground.m = float("inf"); ground.i = float("inf"); ground.m_inv = 0.0; ground.i_inv = 0.0
        ground.p.x, ground.p.y = 4.5, 1.0
        box.type = 0; box.m = 1.0; box.m_inv = 1.0; box.i = 1e9; box.i_inv = 1e-9
        box.p.x, box.p.y = 5.0, 0.0
        for k in (0, 1):
            ref.lib.ai2c_body_set_transform_export(ctypes.byref(world), k)
        world.shape_count = 2
        gverts = [(-2.5, -0.15), (2.5, -0.15), (2.5, 4.0), (-2.5, 4.0)]
        bverts = [(0.435, -0.5), (0.435, 0.5), (-0.435, 0.5), (-0.435, -0.5)]
        for k, (body, verts, friction, ctype, hashid, cats, mask) in enumerate(((0, gverts, 0.7, 1, 0, 2, 0xFFFFFFFF), (1, bverts, 0.7, 3, 1, 1, 0xFFFFFFFB))):
            flat = (ctypes.c_double * 8)(*[c for v in verts for c in v])
            self.assertEqual(ref.lib.ai2c_shape_set_verts_export(ctypes.byref(world), k, 4, flat), 0)
            shape = world.shapes[k]
            shape.body = body; shape.u = friction; shape.e = 0.0; shape.collision_type = ctype; shape.hashid = hashid
            shape.categories = cats; shape.mask = mask
            ref.lib.ai2c_shape_cache_export(ctypes.byref(world), k)
        order = (ctypes.c_int * 2)(0, 1)
        bbs = (ctypes.c_double * 8)(*[world.shapes[0].bb.l, world.shapes[0].bb.b, world.shapes[0].bb.r, world.shapes[0].bb.t,
                                      world.shapes[1].bb.l, world.shapes[1].bb.b, world.shapes[1].bb.r, world.shapes[1].bb.t])
        vs = (ctypes.c_double * 4)(0.0, 0.0, 0.0, 0.0)
        self.assertEqual(ref.lib.ai2c_world_build_trees_export(ctypes.byref(world), order, bbs, vs, 2), 0)
        return world

    def test_box_falls_lands_and_rests_within_collision_slop(self):
        world = self._box_world()
        dt = 1.0 / 30.0
        bias_coef, damping_dt = self.ref.coefficients(world, dt)
        heights = []
        for _ in range(120):
            self.assertEqual(self.ref.step(world, dt, bias_coef, damping_dt), 0)
            heights.append(world.bodies[1].p.y)
        box = world.bodies[1]
        # Ground top is at y = 0.85 (+y is down); a resting half-height 0.5 box sits at 0.35 plus penetration <= slop.
        self.assertGreater(box.p.y, 0.35)
        self.assertLess(box.p.y, 0.35 + world.slop + 1e-9)
        self.assertLess(abs(box.v.y), 1e-6)
        self.assertEqual(world.solve_count, 1)
        arb = world.arbiters[world.solve[0]]
        self.assertEqual(arb.count, 2)
        self.assertEqual((arb.n.x, arb.n.y), (0.0, -1.0))
        # Static ground is body a, dynamic box is body b once the pair persists (Chipmunk role order).
        self.assertEqual((arb.body_a, arb.body_b), (0, 1))
        self.assertGreater(sum(arb.contacts[j].jnAcc for j in range(arb.count)), 0.5)
        self.assertTrue(all(h2 >= h1 - 1e-12 for h1, h2 in zip(heights, heights[1:])) or heights[-1] > heights[0])

    def test_player_pair_is_ignored_by_begin_handler(self):
        world = self._box_world()
        # Turn the "ground" into a second player-type box that overlaps the first.
        world.shape_count = 2
        world.handler_count = 1
        world.handler_type_a[0] = 3; world.handler_type_b[0] = 4; world.handler_begin_false[0] = 1
        world.shapes[0].collision_type = 4
        dt = 1.0 / 30.0
        bias_coef, damping_dt = self.ref.coefficients(world, dt)
        for _ in range(60):
            self.ref.step(world, dt, bias_coef, damping_dt)
        # With the begin handler rejecting the pair, nothing is solved and the box keeps falling through.
        self.assertEqual(world.solve_count, 0)
        self.assertGreater(world.bodies[1].p.y, 1.0)
        ignored = [world.arbiters[i] for i in range(world.arbiter_count) if world.arbiters[i].active]
        self.assertTrue(all(a.state == 2 for a in ignored))

    def test_default_contraction_is_unfused_even_on_darwin(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("platform.system", return_value="Darwin"):
            self.assertEqual(self.m.default_contraction(), "off")

    def test_valid_contraction_override_is_preserved(self):
        for value in ("on", "off", "fast"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"AI2_CONTACT_FP_CONTRACT": value}, clear=True):
                self.assertEqual(self.m.default_contraction(), value)

    def test_invalid_contraction_override_fails_closed(self):
        with mock.patch.dict(os.environ, {"AI2_CONTACT_FP_CONTRACT": "junk"}, clear=True):
            with self.assertRaisesRegex(ValueError, "AI2_CONTACT_FP_CONTRACT"):
                self.m.default_contraction()


class ContactReferenceFrozenGameTests(unittest.TestCase):
    def test_frozen_replay_uses_production_default_contraction(self):
        import native_navigation_contact as contact

        first = {"compared_ticks": 150, "exact": True, "first_divergence": None}
        second = {"exact": False}
        capture_row = json.dumps({"pre": {}, "post": {"solve": [0], "bodies": [{}, {}, {}, {"p": [0.0, 0.0]}]}})
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.dict(sys.modules, {"pymunk": types.ModuleType("pymunk")}), \
                mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(Path, "read_text", return_value=capture_row + "\n"), \
                mock.patch("importlib.import_module"), \
                mock.patch("subprocess.run"), \
                mock.patch.object(contact, "verify_capture", side_effect=(first, second)), \
                mock.patch(f"{__name__}.build_reference", return_value=(contact, object())) as build:
            self.test_short_capture_replays_bit_exactly()

        self.assertEqual(build.call_args.kwargs["contract"], contact.default_contraction())

    def test_short_capture_replays_bit_exactly(self):
        try:
            import pymunk  # noqa: F401
            importlib.import_module("benchmark_replay")
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise unittest.SkipTest(f"frozen runtime unavailable: {exc}")
        if not (FROZEN / "control.yaml").is_file():
            raise unittest.SkipTest("frozen navigation payload not present")
        if os.environ.get("AI2_CONTACT_FROZEN_TEST", "1") != "1":
            raise unittest.SkipTest("frozen-game test disabled")
        tmp = Path(tempfile.mkdtemp(prefix="ai2-contact-frozen-"))
        try:
            capture = tmp / "capture.jsonl"
            env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1", MPLCONFIGDIR=str(tmp / "mpl"))
            subprocess.run([sys.executable, str(ROOT / "navigation_contact_capture.py"), "--source-root", str(FROZEN), "--config", str(FROZEN / "control.yaml"),
                            "--controller", "scripted", "--steps", "150", "--seeds", "45", "--output", str(capture)], check=True, env=env, cwd=str(ROOT),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import native_navigation_contact as m
            m, ref = build_reference(tmp, contract=m.default_contraction())
            result = m.verify_capture(ref, capture)
            self.assertEqual(result["compared_ticks"], 150)
            self.assertTrue(result["exact"], json.dumps(result["first_divergence"], default=str))
            # The verifier must notice a perturbed expectation.
            rows = [json.loads(line) for line in capture.read_text().splitlines() if line.strip()]
            for row in rows:
                if "pre" in row and row["post"]["solve"]:
                    row["post"]["bodies"][3]["p"][0] += 1e-9
                    break
            perturbed = tmp / "perturbed.jsonl"
            perturbed.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            self.assertFalse(m.verify_capture(ref, perturbed)["exact"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
