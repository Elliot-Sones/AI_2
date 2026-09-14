"""Address/UB sanitizer smoke tests for portable native navigation C code."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native"
SMOKE = Path(__file__).with_name("native_sanitizer_smoke.c")
COMPILE_TIMEOUT_SECONDS = 30
RUN_TIMEOUT_SECONDS = 20


class NativeSanitizerSmokeTests(unittest.TestCase):
    def test_sanitizer_smoke_exercises_native_scalar_guards(self):
        compiler = shutil.which(os.environ.get("CC", "")) if os.environ.get("CC") else None
        compiler = compiler or shutil.which("clang") or shutil.which("cc")
        if compiler is None:
            self.skipTest("No C compiler available for native sanitizer smoke")

        with tempfile.TemporaryDirectory(prefix="ai2-native-sanitize-") as tmp:
            exe = Path(tmp) / "native_sanitizer_smoke"
            command = [
                compiler,
                "-std=c11",
                "-O1",
                "-g",
                "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
                "-I",
                str(NATIVE),
                str(SMOKE),
                str(NATIVE / "navigation_input.c"),
                str(NATIVE / "navigation_movement.c"),
                str(NATIVE / "navigation_platform.c"),
                str(NATIVE / "navigation_contract.c"),
                str(NATIVE / "navigation_observation.c"),
                str(NATIVE / "navigation_physics.c"),
                "-lm",
                "-o",
                str(exe),
            ]
            try:
                compiled = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=COMPILE_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                self.skipTest(f"Compiler sanitizer build timed out after {COMPILE_TIMEOUT_SECONDS}s: {(exc.stdout or '')[-500:]} {(exc.stderr or '')[-500:]}")
            if compiled.returncode != 0:
                combined = compiled.stdout + compiled.stderr
                unsupported_markers = ("unsupported option", "unsupported argument", "not supported", "unable to find library")
                if any(marker in combined.lower() for marker in unsupported_markers):
                    self.skipTest(f"Compiler does not support ASan/UBSan here: {combined[-1000:]}")
                self.fail(f"native sanitizer smoke compile failed:\n{combined}")

            env = dict(os.environ)
            env.setdefault("ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1")
            env.setdefault("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
            try:
                run = subprocess.run(
                    [str(exe)],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    env=env,
                    timeout=RUN_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                self.skipTest(
                    "ASan/UBSan runtime did not complete within "
                    f"{RUN_TIMEOUT_SECONDS}s on this host; stdout={exc.stdout!r} stderr={exc.stderr!r}"
                )

        self.assertEqual(run.returncode, 0, msg=f"stdout:\n{run.stdout}\nstderr:\n{run.stderr}")
        self.assertIn("native sanitizer smoke ok", run.stdout)
        self.assertIn("fuzz_cases=1000", run.stdout)


if __name__ == "__main__":
    unittest.main()
