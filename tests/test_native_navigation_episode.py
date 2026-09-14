"""Tests for the closed-loop native navigation tick (native/navigation_episode.h).

Pure tests always run (they compile the C reference with cc). The exactness
test against the real frozen game runs only when the frozen payload and a
Pymunk 6.2.1 runtime are importable; it is skipped, not passed, otherwise.
"""
from __future__ import annotations

import ctypes
import gzip
import hashlib
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))

FROZEN = Path(os.environ.get("AI2_FROZEN_SOURCE_ROOT") or "/Users/elliot18/Projects/ml-research/AI_2-learning-validation-20260905/baselines/navigation-repair-20260905/payload/variants/corrected")
PHASE_0B = {"min_dist": 2.0, "max_dist": 6.0, "fall_penalty": -3.0, "completion_mode": "sequence", "control_mode": "movement", "settle_speed": 1.0,
            "max_targets": 3, "time_limit": 600, "settle_steps": 9, "post_goal_required_steps": 15}


def build_reference(tmp: Path):
    import native_navigation_episode as m

    if shutil.which(os.environ.get("CC") or "cc") is None:
        raise unittest.SkipTest("no C compiler")
    return m, m.EpisodeReference(m.compile_cpu_reference(tmp, m.contact.default_contraction()))


class EpisodePureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ai2-episode-"))
        self.m, self.ref = build_reference(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_struct_layouts_and_positions_table(self):
        # The constructor raises on any size or offset mismatch; check the table provenance too.
        self.assertGreater(ctypes.sizeof(self.m.Episode), ctypes.sizeof(self.m.contact.World))
        header = (ROOT / "native" / "navigation_positions.h").read_text()
        self.assertIn(self.ref.positions_sha256, header)
        self.assertEqual(self.ref.positions_count, header.count("\n    {"))
        if (FROZEN / "positions.json").is_file():
            self.assertEqual(self.ref.positions_sha256, hashlib.sha256((FROZEN / "positions.json").read_bytes()).hexdigest())

    def test_tick_arithmetic_is_not_fused(self):
        probe = self.ref.lib.ai2n_fma_probe_export
        probe.restype = ctypes.c_double
        probe.argtypes = [ctypes.c_double] * 3
        rng = random.Random(3)
        for _ in range(20000):
            a, b, c = (rng.uniform(-10, 10) for _ in range(3))
            self.assertEqual(probe(a, b, c), a * b + c)

    def test_replay_rng_rejects_wrong_kind_arguments_and_exhaustion(self):
        m, ref = self.m, self.ref
        rng, keep = ref.replay_rng([{"kind": 2, "a": 2.6, "b": 6.5, "value": 3.25}])
        self.assertEqual(ref.lib.ai2n_rng_request_export(ctypes.byref(rng), 3, 2.6, 6.5), 0.0)
        self.assertEqual(rng.error, 2)
        rng, keep = ref.replay_rng([{"kind": 2, "a": 2.6, "b": 6.5, "value": 3.25}])
        self.assertEqual(ref.lib.ai2n_rng_request_export(ctypes.byref(rng), 2, 2.6, 6.0), 0.0)
        self.assertEqual(rng.error, 3)
        rng, keep = ref.replay_rng([{"kind": 2, "a": 2.6, "b": 6.5, "value": 3.25}])
        self.assertEqual(ref.lib.ai2n_rng_request_export(ctypes.byref(rng), 2, 2.6, 6.5), 3.25)
        self.assertEqual(ref.lib.ai2n_rng_request_export(ctypes.byref(rng), 2, 2.6, 6.5), 0.0)
        self.assertEqual(rng.error, 1)

    def test_generator_rng_draws_stay_in_range_and_differ_by_lane(self):
        ref = self.m
        lanes = []
        for lane in range(3):
            rng = self.m.Rng()
            self.ref.lib.ai2n_rng_seed_export(ctypes.byref(rng), 1234, lane)
            values = []
            for _ in range(2000):
                u = self.ref.lib.ai2n_rng_request_export(ctypes.byref(rng), 2, 2.6, 6.5)
                self.assertTrue(2.6 <= u < 6.5)
                r = self.ref.lib.ai2n_rng_request_export(ctypes.byref(rng), 3, 500.0, 700.0)
                self.assertTrue(500.0 <= r <= 700.0 and r == int(r))
                n = self.ref.lib.ai2n_rng_request_export(ctypes.byref(rng), 4, 649.0, 0.0)
                self.assertTrue(0.0 <= n < 649.0 and n == int(n))
                bit = self.ref.lib.ai2n_rng_request_export(ctypes.byref(rng), 1, 1.0, 0.0)
                self.assertIn(bit, (0.0, 1.0))
                values.append(u)
            lanes.append(values)
        self.assertNotEqual(lanes[0], lanes[1])
        self.assertNotEqual(lanes[1], lanes[2])

    def test_generator_mode_runs_full_episodes_without_error(self):
        m, ref = self.m, self.ref
        cfg = ref.make_config(PHASE_0B)
        rng = m.Rng()
        ref.lib.ai2n_rng_seed_export(ctypes.byref(rng), 7, 0)
        ep = m.Episode()
        self.assertEqual(ref.reset(ep, cfg, rng), 0)
        self.assertEqual(ep.wrap.target_valid, 1)
        actions = random.Random(11)
        dones = 0
        for tick in range(2000):
            act = [1.0 if actions.random() < 0.3 else 0.0 for _ in range(10)]
            rc, out = ref.step(ep, act, rng)
            self.assertEqual(rc, 0, f"tick {tick} error {out.error} last {ep.last_error}")
            self.assertTrue(all(abs(out.obs[i]) <= 18.0 for i in range(64)))
            if out.terminated or out.truncated:
                dones += 1
                self.assertEqual(ref.reset(ep, cfg, rng), 0)
        self.assertGreater(dones, 0)


class EpisodeFrozenGameTests(unittest.TestCase):
    def test_closed_loop_matches_frozen_game_bit_for_bit(self):
        try:
            import pymunk  # noqa: F401
            importlib.import_module("benchmark_replay")
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise unittest.SkipTest(f"frozen runtime unavailable: {exc}")
        if not (FROZEN / "control.yaml").is_file():
            raise unittest.SkipTest("frozen navigation payload not present")
        if os.environ.get("AI2_EPISODE_FROZEN_TEST", "1") != "1":
            raise unittest.SkipTest("frozen-game test disabled")
        tmp = Path(tempfile.mkdtemp(prefix="ai2-episode-frozen-"))
        try:
            capture = tmp / "capture.jsonl.gz"
            env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1", MPLCONFIGDIR=str(tmp / "mpl"))
            subprocess.run([sys.executable, str(ROOT / "navigation_episode_capture.py"), "--source-root", str(FROZEN), "--config", str(FROZEN / "control.yaml"),
                            "--controller", "adaptive", "--steps", "150", "--seeds", "45", "--output", str(capture)], check=True, env=env, cwd=str(ROOT),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            m, ref = build_reference(tmp)
            result = m.verify_capture(ref, capture)
            self.assertEqual(result["compared_ticks"], 150)
            self.assertTrue(result["exact"], json.dumps(result["first_divergence"], default=str))
            # The verifier must notice a perturbed expectation in outputs and in hidden state.
            rows = [json.loads(line) for line in gzip.open(capture, "rt").read().splitlines() if line.strip()]
            perturbed_obs = [json.loads(json.dumps(r)) for r in rows]
            perturbed_state = [json.loads(json.dumps(r)) for r in rows]
            for row in perturbed_obs:
                if "tick" in row and row["tick"]["tick"] == 40:
                    row["tick"]["obs"][0] += 1e-6
            for row in perturbed_state:
                if "tick" in row and row["tick"]["tick"] == 40:
                    row["tick"]["state"]["players"][0]["vx"] += 1e-12
            for name, rows_out in (("obs", perturbed_obs), ("state", perturbed_state)):
                path = tmp / f"perturbed-{name}.jsonl"
                path.write_text("\n".join(json.dumps(r) for r in rows_out) + "\n")
                verdict = m.verify_capture(ref, path)
                self.assertFalse(verdict["exact"], name)
                self.assertEqual(verdict["first_divergence"]["tick"], 40, name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
