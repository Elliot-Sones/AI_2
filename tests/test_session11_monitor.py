"""No-network tests for the local scalar-only session11 monitor."""

import builtins
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


MODULE = Path(__file__).resolve().parents[1] / "session11_monitor.py"
HEADER = "utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W], temperature.gpu"
SAMPLE = "1788800000.125\n" + HEADER + "\n42, 12, 8000, 24564, 210.5, 62\n"


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), "session11 scalar monitor must exist")
        spec = importlib.util.spec_from_file_location("session11_monitor_test", MODULE)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def args(self, *extra):
        return self.mod.parse_args(list(extra))

    def test_import_does_not_import_wandb_or_run_commands(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"wandb", "torch"}:
                raise AssertionError("heavy import during module import")
            return original(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("monitor_light_import", MODULE)
        with mock.patch("builtins.__import__", side_effect=guarded), mock.patch("subprocess.run", side_effect=AssertionError("command on import")):
            spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def test_defaults_and_cli_overrides(self):
        args = self.args()
        self.assertEqual((args.host, args.port, args.instance_id, args.group),
                         ("ssh9.vast.ai", 29728, 50149729, "native-capacity-session11"))
        self.assertEqual(args.output.name, "native-capacity-session11")
        args = self.args("--host", "gpu.example", "--port", "2222", "--instance-id", "9", "--group", "trial", "--interval", "2")
        self.assertEqual((args.host, args.port, args.instance_id, args.group, args.interval), ("gpu.example", 2222, 9, "trial", 2))

    def test_cli_rejects_invalid_values_before_wandb(self):
        for argv in [("--host", "-oProxyCommand=bad"), ("--port", "0"), ("--instance-id", "0"), ("--interval", "nan"), ("--interval", "-1")]:
            with self.subTest(argv=argv), self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                self.args(*argv)

    def test_parser_preserves_remote_timestamp_and_exact_six_scalars(self):
        row = self.mod.parse_sample(SAMPLE)
        self.assertEqual(row, {"monitor/remote_unix": 1788800000.125, "gpu/util": 42.0,
                              "gpu/mem_util": 12.0, "gpu/mem_used_mb": 8000.0,
                              "gpu/mem_total_mb": 24564.0, "gpu/power_w": 210.5, "gpu/temp_c": 62.0})

    def test_parser_rejects_wrong_header_rows_and_shape(self):
        for sample in [SAMPLE.replace("power.draw", "power.secret"), SAMPLE + "0,0,0,0,0,0\n",
                       SAMPLE.replace("42, 12,", "42,"), SAMPLE.replace("42, 12,", "42, 12, 1,"),
                       SAMPLE.replace(HEADER + "\n", ""), "warning\n" + SAMPLE]:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                self.mod.parse_sample(sample)

    def test_parser_rejects_nonfinite_negative_and_invalid_ranges(self):
        for value in ["NaN", "inf", "-1", "101"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.mod.parse_sample(SAMPLE.replace("42,", value + ","))
        for sample in [SAMPLE.replace("210.5", "-1"), SAMPLE.replace("8000", "25000"),
                       SAMPLE.replace("24564", "0"), SAMPLE.replace("1788800000.125", "nan"),
                       SAMPLE.replace("1788800000.125", "0"), SAMPLE.replace("62\n", "-1\n")]:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                self.mod.parse_sample(sample)

    def test_query_records_local_bracket_and_remote_timestamp(self):
        response = subprocess.CompletedProcess([], 0, SAMPLE, "")
        with mock.patch.object(self.mod.time, "time", side_effect=[200.0, 203.0]), \
             mock.patch.object(self.mod.time, "monotonic", side_effect=[10.0, 12.5]), \
             mock.patch.object(self.mod.subprocess, "run", return_value=response) as run:
            row = self.mod.query(self.args())
        self.assertEqual((row["monitor/local_query_start_unix"], row["monitor/local_query_end_unix"], row["monitor/query_seconds"]), (200.0, 203.0, 2.5))
        self.assertEqual(row["monitor/remote_unix"], 1788800000.125)
        self.assertEqual(row["monitor/telemetry_ok"], 1)
        command = run.call_args.args[0]
        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertIn("root@ssh9.vast.ai", command)
        self.assertEqual(command[-1], "date +%s.%N && nvidia-smi --id=0 --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,nounits")
        self.assertEqual(run.call_args.kwargs["timeout"], 15)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_failed_query_keeps_only_exception_type_no_stale_gpu_values(self):
        failure = subprocess.CalledProcessError(1, "secret-command", output="secret-key", stderr="secret-token")
        with mock.patch.object(self.mod.subprocess, "run", side_effect=failure):
            row = self.mod.query(self.args())
        self.assertEqual(row["monitor/error_type"], "CalledProcessError")
        self.assertEqual(row["monitor/telemetry_ok"], 0)
        self.assertNotIn("monitor/remote_unix", row)
        self.assertFalse(any(k.startswith("gpu/") for k in row))
        self.assertNotIn("secret", json.dumps(row))

    def test_main_logs_scalars_only_and_sigterm_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            handlers = {}
            def install(sig, handler):
                previous = handlers.get(sig, signal.SIG_DFL)
                handlers[sig] = handler
                return previous
            run = mock.Mock(id="abc", url="https://wandb.example/run", entity="owner", project="project")
            run.log.side_effect = lambda row: handlers[signal.SIGTERM](signal.SIGTERM, None)
            wandb = types.SimpleNamespace(init=mock.Mock(return_value=run), Settings=mock.Mock(return_value="settings"))
            with mock.patch.dict(sys.modules, {"wandb": wandb}), \
                 mock.patch.object(self.mod.signal, "signal", side_effect=install), \
                 mock.patch.object(self.mod.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, SAMPLE, "")):
                self.assertEqual(self.mod.main(["--output", directory]), 0)
            receipt = json.loads((Path(directory) / "monitor_run.json").read_text())
            self.assertEqual((receipt["id"], receipt["url"]), ("abc", "https://wandb.example/run"))
            self.assertIsInstance(receipt["pid"], int)
            stored = json.loads((Path(directory) / "gpu-telemetry.jsonl").read_text())
            self.assertEqual(stored, run.log.call_args.args[0])
            self.assertTrue(all(isinstance(v, (float, int)) for v in run.log.call_args.args[0].values()))
            run.finish.assert_called_once_with()
            run.save.assert_not_called()
            run.log_artifact.assert_not_called()
            self.assertEqual(handlers[signal.SIGTERM], signal.SIG_DFL)
            self.assertTrue(wandb.Settings.call_args.kwargs["x_disable_stats"])
            self.assertTrue(wandb.Settings.call_args.kwargs["disable_code"])
            self.assertEqual(wandb.Settings.call_args.kwargs["console"], "off")
            self.assertFalse(wandb.init.call_args.kwargs["save_code"])
            self.assertIn("64x512", wandb.init.call_args.kwargs["notes"])

    def test_failed_query_type_is_local_only_and_logging_failure_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = mock.Mock(id="abc", url="https://wandb.example/run", entity="owner", project="project")
            run.log.side_effect = RuntimeError("offline")
            wandb = types.SimpleNamespace(init=mock.Mock(return_value=run), Settings=mock.Mock())
            with mock.patch.dict(sys.modules, {"wandb": wandb}), \
                 mock.patch.object(self.mod.subprocess, "run", side_effect=RuntimeError("secret")), \
                 self.assertRaisesRegex(RuntimeError, "offline"):
                self.mod.main(["--output", directory])
            row = json.loads((Path(directory) / "gpu-telemetry.jsonl").read_text())
            self.assertEqual(row["monitor/error_type"], "RuntimeError")
            self.assertNotIn("monitor/error_type", run.log.call_args.args[0])
            self.assertNotIn("secret", json.dumps(row))
            run.finish.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
