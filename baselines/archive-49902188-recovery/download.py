"""Use installed Vast copy routing with unprivileged, host-key-checked rsync."""
import contextlib
import io
from pathlib import Path
import shlex
import subprocess
import sys
import vast

ROOT = Path(__file__).resolve().parent
run = subprocess.run
mode = sys.argv[1]
directory = sys.argv[2]
assert mode in ("download", "verify")
assert directory in ("animation-cache-20260904", "benchmark-20260904b")
destination = ROOT / "remote" / directory
destination.mkdir(parents=True, exist_ok=True)
results = []

def safe_run(command, *args, **kwargs):
    if isinstance(command, str) and command.startswith("mkdir -p "):
        return subprocess.CompletedProcess(command, 0)
    if isinstance(command, str) and command.startswith("sudo rsync "):
        argv = shlex.split(command)[1:]
        idx = argv.index("-e") + 1
        ssh = shlex.split(argv[idx])
        if ssh[0] == "sudo":
            ssh.pop(0)
        ssh = [x.replace("StrictHostKeyChecking=no", "StrictHostKeyChecking=accept-new") for x in ssh]
        ssh += ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        argv[idx] = shlex.join(ssh)
        argv[1:1] = ["--exclude=.git/", "--exclude=__pycache__/", "--exclude=.env",
                     "--exclude=.netrc", "--exclude=.ssh/", "--exclude=.cache/"]
        if mode == "verify":
            argv[1:1] = ["--dry-run", "--checksum", "--itemize-changes"]
        result = run(argv, capture_output=True, text=True, timeout=180)
        (ROOT / f"{directory}-{mode}.log").write_text(result.stdout + result.stderr)
        results.append(result.returncode)
        return result
    return run(command, *args, **kwargs)

subprocess.run = safe_run
sys.argv = ["vastai", "copy", "-i", str(Path.home() / ".ssh/id_ed25519"),
            f"49902188:/workspace/AI_2/{directory}/", str(destination)]
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        vast.main()
except BaseException as error:
    print(f"Transfer wrapper ended: {type(error).__name__}; provider output withheld")
print(f"{directory} {mode}: rsync exit codes={results}")
sys.exit(0 if results == [0] else 1)
