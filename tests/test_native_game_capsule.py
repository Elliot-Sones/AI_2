"""Differential tests against the unchanged frozen CapsuleCollider class."""
from __future__ import annotations

import ast
import ctypes
import hashlib
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(os.environ.get("AI2_FROZEN_SOURCE_ROOT", "/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected"))
SOURCE_SHA = "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230"


class Capsule(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in ("x", "y", "width", "height")]


class CapsuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = FROZEN / "UTMIST-AI2-main/environment/environment.py"
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
            raise AssertionError("frozen collider source hash changed")
        tree = ast.parse(source)
        collider = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CapsuleCollider")
        scope = {"pygame": pygame, "Camera": object}
        exec(compile(ast.Module(body=[collider], type_ignores=[]), str(path), "exec"), scope)
        cls.original = scope["CapsuleCollider"]
        cls.temp = tempfile.TemporaryDirectory(prefix="ai2-game-capsule-")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def native(self, a, b):
        if not hasattr(type(self), "lib"):
            source = ROOT / "native/game_capsule.c"
            self.assertTrue(source.is_file(), "native capsule implementation is missing")
            libpath = Path(self.temp.name) / "capsule.so"
            result = subprocess.run([os.environ.get("CC", "cc"), "-shared", "-fPIC", "-O2", "-ffp-contract=off", str(source), "-o", str(libpath), "-lm"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            type(self).lib = ctypes.CDLL(str(libpath))
            type(self).lib.ai2g_capsule_intersects_export.argtypes = [ctypes.POINTER(Capsule)] * 2
            type(self).lib.ai2g_capsule_intersects_export.restype = ctypes.c_int
        return bool(self.lib.ai2g_capsule_intersects_export(ctypes.byref(Capsule(*a)), ctypes.byref(Capsule(*b))))

    def compare(self, a, b):
        expected = bool(self.original(a[:2], *a[2:]).intersects(self.original(b[:2], *b[2:])))
        self.assertEqual(self.native(a, b), expected, (a, b))

    def test_circles_and_strict_tangency(self):
        for x in (0, 1, math.nextafter(2.0, 0), 2, math.nextafter(2.0, 3), 3):
            self.compare((0, 0, 2, 2), (x, 0, 2, 2))

    def test_large_finite_circles_do_not_require_pygame_rect_coordinates(self):
        self.compare((3e9, 0, 2, 2), (3e9, 0, 2, 2))

    def test_negative_coordinates_rect_truncation_and_degenerate_sizes(self):
        cases = [(x, y, w, h) for x in (-2.9, -1.0, -.99, 0, .99, 1.0, 2.9)
                 for y in (-1.7, 0, 1.7) for w, h in ((.928, 1.024), (.1, 3.2), (3.2, .1), (0, 0), (0, 2), (2, 0), (-1, 2), (2, -1))]
        for i, a in enumerate(cases):
            for b in (cases[(i * 17) % len(cases)], (a[0] + 1, a[1], 1, 1), a):
                self.compare(a, b)
                self.compare(b, a)

    def test_randomized_frozen_differential(self):
        rng = random.Random(130913)
        for _ in range(12000):
            a, b = [(rng.uniform(-20, 20), rng.uniform(-10, 10), rng.uniform(0, 8), rng.uniform(0, 8)) for _ in range(2)]
            if rng.random() < .3:
                a = (*a[:3], a[2])
            if rng.random() < .3:
                b = (*b[:3], b[2])
            self.compare(a, b)

    def test_nonfinite_and_unrepresentable_rects_fail_closed(self):
        for value in (math.nan, math.inf, -math.inf, 1e100):
            self.assertFalse(self.native((value, 0, 1, 2), (0, 0, 1, 2)))


if __name__ == "__main__":
    unittest.main()
