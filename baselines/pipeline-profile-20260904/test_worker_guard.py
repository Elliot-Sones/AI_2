import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import instance_guard as guard
from worker_timing import Spans


class WorkerTimingTests(unittest.TestCase):
    def test_nested_times_partition_and_disabled_calls_do_not_count(self):
        spans = Spans()
        inner = spans.wrap(lambda: 42, "inner")
        outer = spans.wrap(inner, "outer")
        self.assertEqual(outer(), 42)
        self.assertEqual(dict(spans.rows), {})
        spans.enabled = True
        with patch("worker_timing.perf_counter", side_effect=[1.0, 2.0, 3.0, 5.0]):
            self.assertEqual(outer(), 42)
        self.assertEqual(spans.rows["inner"], [1, 1.0, 1.0])
        self.assertEqual(spans.rows["outer"], [1, 4.0, 3.0])
        self.assertEqual(sum(row[2] for row in spans.rows.values()), 4.0)

    def test_exception_unwinds_timing_stack(self):
        spans = Spans()
        spans.enabled = True
        def fail():
            raise ValueError("expected")
        with self.assertRaises(ValueError):
            spans.wrap(fail, "failure")()
        self.assertEqual(spans.stack, [])
        self.assertEqual(spans.rows["failure"][0], 1)


class InstanceGuardTests(unittest.TestCase):
    def test_exact_instance_start_then_stop_without_destroy(self):
        state = {"id": 49902188, "cur_state": "stopped", "actual_status": "exited", "dph_total": 0.3166666667}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def cli(*args):
                if args[:2] == ("start", "instance"):
                    (root / "stop.request").touch()
                return {}
            with patch.object(guard, "ROOT", root), patch.object(guard, "STATE", root / "session.json"), patch.object(guard, "current", return_value=state), patch.object(guard, "cli", side_effect=cli) as calls, contextlib.redirect_stdout(io.StringIO()):
                guard.main()
            self.assertEqual([c.args for c in calls.call_args_list], [("start", "instance", "49902188", "--raw"), ("stop", "instance", "49902188", "--raw")])
            self.assertIn("stopped_verified_not_deleted", (root / "session.json").read_text())

    def test_cli_failure_does_not_leak_raw_output(self):
        class Failure:
            returncode = 1
            stdout = "secret-example"
            stderr = "secret-example"
        with patch.object(guard.subprocess, "run", return_value=Failure()):
            with self.assertRaises(RuntimeError) as caught:
                guard.cli("show", "instances", "--raw")
        self.assertNotIn("secret-example", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
