"""Compile the live input hotpath and compare with actual frozen input classes."""
import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
import importlib
import os
from pathlib import Path
import types
import unittest
import numpy as np


def oracle():
    source = os.environ.get("AI2_REPLAY_SOURCE")
    root = Path(source) if source else Path(__file__).resolve().parents[2] / "AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected"
    path = root / "UTMIST-AI2-main/environment/environment.py"
    text = path.read_text()
    if hashlib.sha256(text.encode()).hexdigest() != "c0116fc7b4a46d45c37367d9aa4d7f44041296ad96ba01e8b1f3f1f158e24230":
        raise AssertionError("frozen source mismatch")
    tree = ast.parse(text)
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ("Facing", "KeyStatus", "HorizontalState", "PlayerInputHandler")]
    namespace = {"dataclass": dataclass, "Enum": Enum, "np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["PlayerInputHandler"]


class LiveInputTests(unittest.TestCase):
    def test_c_update_matches_actual_frozen_all_button_combinations_and_history(self):
        native = importlib.import_module("native_live_input").load_live_input()
        cls = oracle()
        reference, candidate = cls(), cls()
        rng = np.random.default_rng(1992)
        masks = rng.permutation(1024)
        for mask in masks:
            action = np.array([(mask >> bit) & 1 for bit in range(10)], dtype=np.float32)
            reference.update(action)
            native.update(candidate, action)
            self.assertEqual(reference.prev_state, candidate.prev_state)
            self.assertEqual(vars(reference)["horizontal_state"], vars(candidate)["horizontal_state"])
            for field in ("raw_horizontal", "raw_vertical", "no_horizontal", "last_direction"):
                self.assertEqual(getattr(reference, field), getattr(candidate, field), field)
            for key in reference.key_names:
                for field in ("held", "just_pressed", "just_released"):
                    left, right = getattr(reference.key_status[key], field), getattr(candidate.key_status[key], field)
                    self.assertEqual(left, right)
                    self.assertIs(type(left), type(right))

    def test_threshold_noncontiguous_and_invalid_action_preserve_source_semantics(self):
        native = importlib.import_module("native_live_input").load_live_input()
        cls = oracle()
        for action in ([0.5] * 10, np.linspace(-1, 1, 20)[::2], [float("nan")] * 10):
            reference, candidate = cls(), cls()
            reference.update(action)
            native.update(candidate, action)
            self.assertEqual(repr(reference), repr(candidate))
        for action in ([], [0] * 9, None):
            reference, candidate = cls(), cls()
            try:
                reference.update(action)
            except Exception as error:
                with self.assertRaises(type(error)):
                    native.update(candidate, action)
            else:
                self.fail("expected source exception")


if __name__ == "__main__":
    unittest.main()
