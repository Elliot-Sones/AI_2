"""Local-only credentials; explicitly launched, scalar-only remote GPU monitor.

GPU utilization is kernel-busy time, not compute occupancy or peak FLOPS.
Memory utilization is memory-access busy time; memory.used is device-wide MiB,
not torch.memory_allocated(). Remote time precedes the nvidia-smi query; local
start/end bracket the SSH request. No per-sample remote completion is implied.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time


GPU_FIELDS = ("utilization.gpu", "utilization.memory", "memory.used",
              "memory.total", "power.draw", "temperature.gpu")
GPU_HEADER = ("utilization.gpu [%]", "utilization.memory [%]", "memory.used [MiB]",
              "memory.total [MiB]", "power.draw [W]", "temperature.gpu")
GPU_KEYS = ("gpu/util", "gpu/mem_util", "gpu/mem_used_mb", "gpu/mem_total_mb",
            "gpu/power_w", "gpu/temp_c")
TIMING_KEYS = ("monitor/local_query_start_unix", "monitor/local_query_end_unix",
               "monitor/remote_unix", "monitor/query_seconds", "monitor/telemetry_ok")
REMOTE_COMMAND = ("date +%s.%N && nvidia-smi --id=0 --query-gpu="
                  + ",".join(GPU_FIELDS) + " --format=csv,nounits")
NOTES = (
    "Goal: compare 9 models in E6 continuation using native GPU navigation, "
    "64x512 rollout geometry. Credentials stay on this Mac. Only six GPU "
    "scalars, timestamp brackets, query duration and telemetry status are sent; "
    "no weights, source files or artifacts are uploaded. GPU busy percentage "
    "is not compute occupancy or peak-FLOPS utilization. Memory utilization is "
    "memory-access busy time; used memory is device-wide MiB, not tensor allocation."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "baselines/native-capacity-session11")
    parser.add_argument("--host", default="ssh9.vast.ai")
    parser.add_argument("--port", type=int, default=29728)
    parser.add_argument("--instance-id", type=int, default=50149729)
    parser.add_argument("--group", default="native-capacity-session11")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", args.host):
        parser.error("--host must be a hostname or IPv4 address")
    if not 1 <= args.port <= 65535 or args.instance_id <= 0:
        parser.error("--port must be 1..65535 and --instance-id must be positive")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be positive and finite")
    args.output = args.output.resolve()
    return args


def parse_sample(output):
    """Require one remote timestamp, the exact CSV schema, and one GPU row."""
    lines = output.strip().splitlines()
    if len(lines) != 3 or tuple(cell.strip() for cell in lines[1].split(",")) != GPU_HEADER:
        raise ValueError("invalid GPU telemetry schema")
    remote = float(lines[0])
    values = [float(cell.strip()) for cell in lines[2].split(",")]
    if len(values) != 6 or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("invalid GPU telemetry values")
    if not math.isfinite(remote) or remote <= 0:
        raise ValueError("invalid remote timestamp")
    if values[0] > 100 or values[1] > 100 or values[3] <= 0 or values[2] > values[3]:
        raise ValueError("GPU telemetry values outside valid ranges")
    return {"monitor/remote_unix": remote, **dict(zip(GPU_KEYS, values))}


def query(args):
    command = ["ssh", "-p", str(args.port), "-o", "BatchMode=yes", "-o",
               "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=5",
               "root@" + args.host, REMOTE_COMMAND]
    start = time.time()
    monotonic_start = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        row = {**parse_sample(result.stdout), "monitor/telemetry_ok": 1}
    except Exception as error:
        # Never persist or forward command, stdout, stderr, or exception message.
        row = {"monitor/telemetry_ok": 0, "monitor/error_type": type(error).__name__}
    row.update({"monitor/local_query_start_unix": start,
                "monitor/local_query_end_unix": time.time(),
                "monitor/query_seconds": time.monotonic() - monotonic_start})
    return row


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    previous_handlers = {}
    run = None

    def stopping(*_):
        stop.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, stopping)
        import wandb  # Credentials and SDK are used only by an explicit launch.

        run = wandb.init(project="utmist-ai2", name=f"{args.group}-instance{args.instance_id}",
                         group=args.group, job_type="monitor", dir=str(args.output), notes=NOTES,
                         config={"instance_id": args.instance_id, "credential_location": "local_only",
                                 "interval_seconds": args.interval, "telemetry_scope": "device0_scalars_only"},
                         save_code=False,
                         settings=wandb.Settings(x_disable_stats=True, disable_code=True, console="off"))
        receipt = {"id": run.id, "url": run.url, "path": f"{run.entity}/{run.project}/{run.id}",
                   "pid": os.getpid(), "instance_id": args.instance_id, "group": args.group}
        (args.output / "monitor_run.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("Monitor URL: " + str(run.url), flush=True)
        while not stop.is_set():
            started = time.monotonic()
            row = query(args)
            with (args.output / "gpu-telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            # A strict allowlist prevents local failure details entering W&B.
            run.log({key: row[key] for key in GPU_KEYS + TIMING_KEYS if key in row})
            print(json.dumps(row, allow_nan=False), flush=True)
            stop.wait(max(0.0, args.interval - (time.monotonic() - started)))
    finally:
        try:
            if run is not None:
                run.finish()
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
