"""Candidate grouped-vector real-game parity gate.

This reuses the independent stock-oracle tests from ``test_grouped_game`` but
loads the grouped implementation from ``AI2_GROUPED_CANDIDATE_PATH``.  Without a
candidate path the runtime parity class skips; with an explicit bad path or bad
module it fails closed instead of silently falling back to the in-tree grouped
implementation.
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import ai2.runtime.benchmark_campaign as benchmark_campaign
from tests import test_grouped_game as grouped_game


def load_candidate_grouped_cls(env_var: str = "AI2_GROUPED_CANDIDATE_PATH"):
    raw_path = os.environ.get(env_var)
    if not raw_path:
        raise unittest.SkipTest(f"set {env_var} to run candidate grouped-game parity")
    candidate_path = Path(raw_path).expanduser().resolve()
    if not candidate_path.is_file():
        raise AssertionError(f"{env_var} does not exist: {candidate_path}")
    module = benchmark_campaign.load_grouped_env_module(candidate_path)
    grouped_cls = getattr(module, "GroupedSubprocVecEnv", None)
    if not isinstance(grouped_cls, type):
        raise TypeError(f"{candidate_path} did not define GroupedSubprocVecEnv")
    return grouped_cls


class CandidateGroupedGameParityTest(grouped_game.FrozenGroupedGameParityTest):
    @classmethod
    def setUpClass(cls):
        candidate_cls = load_candidate_grouped_cls()
        super().setUpClass()
        cls.grouped_cls = candidate_cls


class CandidateGroupedLoaderTest(unittest.TestCase):
    def test_absent_candidate_path_skips_candidate_runtime_gate(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(unittest.SkipTest):
                load_candidate_grouped_cls()

    def test_explicit_missing_candidate_path_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_grouped.py"
            with mock.patch.dict(os.environ, {"AI2_GROUPED_CANDIDATE_PATH": str(missing)}, clear=True):
                with self.assertRaisesRegex(AssertionError, "does not exist"):
                    load_candidate_grouped_cls()

    def test_candidate_module_must_define_grouped_subproc_vec_env_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate_grouped.py"
            candidate.write_text("GroupedSubprocVecEnv = None\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"AI2_GROUPED_CANDIDATE_PATH": str(candidate)}, clear=True):
                with self.assertRaisesRegex(TypeError, "GroupedSubprocVecEnv"):
                    load_candidate_grouped_cls()

    def test_candidate_module_is_loaded_through_campaign_helper(self):
        grouped_cls = type("CandidateGroupedSubprocVecEnv", (), {})
        fake_module = types.SimpleNamespace(GroupedSubprocVecEnv=grouped_cls)
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate_grouped.py"
            candidate.write_text("# candidate\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"AI2_GROUPED_CANDIDATE_PATH": str(candidate)}, clear=True), mock.patch(
                'ai2.runtime.benchmark_campaign.load_grouped_env_module', return_value=fake_module
            ) as load:
                resolved = load_candidate_grouped_cls()

        self.assertIs(resolved, grouped_cls)
        self.assertEqual(load.call_args.args, (candidate.resolve(),))


if __name__ == "__main__":
    unittest.main()
