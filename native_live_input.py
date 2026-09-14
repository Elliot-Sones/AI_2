"""Opt-in compiled input update; original live state and enums stay authoritative."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sysconfig
import tempfile

_MODULE = None
_BUILD = None


def load_live_input():
    global _MODULE, _BUILD
    if _MODULE is not None:
        return _MODULE
    build = tempfile.TemporaryDirectory(prefix="ai2-live-input-")
    output = Path(build.name) / ("_ai2_live_input" + sysconfig.get_config_var("EXT_SUFFIX"))
    source = Path(__file__).resolve().parent / "native/live_input.c"
    compiler = shlex.split(os.environ.get("CC") or "cc")
    command = [*compiler, "-std=c99", "-O3", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared", "-I" + sysconfig.get_path("include"), str(source), "-o", str(output)]
    if platform.system() == "Darwin":
        command.extend(["-undefined", "dynamic_lookup"])
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    spec = importlib.util.spec_from_file_location("_ai2_live_input", output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE, _BUILD = module, build
    return module
