#!/usr/bin/env python3
"""Instance-level GPU/CPU monitor as a Weights & Biases run.

Start it once per cloud instance so the W&B system panels (GPU utilization,
GPU memory, power, CPU, RAM) cover the whole instance lifetime, whatever
script is using the GPU. It also logs an explicit nvidia-smi and psutil sample
at a fixed interval, so the numbers appear as ordinary charts too.

    nohup python monitor.py --name <instance-label> --group <run-id> > monitor.log 2>&1 &
    python monitor.py --once      # one sample, then exit

Stop it with SIGTERM or SIGINT; the run is finished cleanly.
"""

import argparse
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import time

import wandb

try:
    import psutil
except Exception:  # psutil ships with wandb, but stay safe
    psutil = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracking import scrub_empty_wandb_env  # noqa: E402


def _num(value: str):
    value = value.strip()
    if value in ("", "N/A", "[N/A]", "[Not Supported]"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def gpu_sample() -> dict:
    """Per-GPU utilization, memory, power and temperature from nvidia-smi."""
    query = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return {}
    if out.returncode != 0:
        return {}
    metrics = {}
    for index, line in enumerate(out.stdout.strip().splitlines()):
        values = [_num(v) for v in line.split(",")]
        if len(values) != 6:
            continue
        prefix = "gpu" if index == 0 else f"gpu{index}"
        util, mem_util, mem_used, mem_total, power, temp = values
        metrics.update({
            f"{prefix}/util": util,
            f"{prefix}/mem_util": mem_util,
            f"{prefix}/mem_used_mb": mem_used,
            f"{prefix}/mem_total_mb": mem_total,
            f"{prefix}/power_w": power,
            f"{prefix}/temp_c": temp,
        })
    return metrics


def system_sample() -> dict:
    metrics = {}
    if psutil is not None:
        metrics["cpu/util"] = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        metrics["mem/used_pct"] = memory.percent
        metrics["mem/used_gb"] = round(memory.used / 1e9, 2)
    else:
        metrics["cpu/util"] = None
    try:
        metrics["cpu/load_1m"] = os.getloadavg()[0]
    except (OSError, AttributeError):
        pass
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Log instance GPU/CPU usage to Weights & Biases")
    parser.add_argument("--name", default=socket.gethostname(), help="run name (use the instance label)")
    parser.add_argument("--project", default="utmist-ai2")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default=None, help="group runs by cloud run id")
    parser.add_argument("--tags", default="", help="comma separated extra tags")
    parser.add_argument("--notes", default=None, help="what this instance is for")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between explicit samples")
    parser.add_argument("--stats-interval", type=float, default=10.0, help="seconds between W&B system samples")
    parser.add_argument("--record", default="monitor_run.json", help="where to write the run record")
    parser.add_argument("--once", action="store_true", help="log one sample and exit")
    args = parser.parse_args(argv)
    scrub_empty_wandb_env()

    try:
        settings = wandb.Settings(x_stats_sampling_interval=args.stats_interval)
    except Exception:
        settings = None

    tags = ["monitor"] + [t.strip() for t in args.tags.split(",") if t.strip()]
    notes = args.notes or (
        f"Instance monitor for {args.name}. System panels cover the whole instance lifetime; "
        f"explicit gpu/* and cpu/* samples every {args.interval:g}s."
    )
    run = wandb.init(
        project=args.project, entity=args.entity, name=args.name, group=args.group,
        job_type="monitor", tags=tags, notes=notes,
        config={"interval": args.interval, "hostname": socket.gethostname(), "pid": os.getpid()},
        settings=settings,
    )

    url = None
    try:
        url = run.url
    except Exception:
        pass
    record = {
        "id": run.id, "name": run.name, "project": run.project,
        "entity": getattr(run, "entity", None) or None,
        "path": "/".join([getattr(run, "entity", None) or "", run.project or "", run.id]),
        "url": url, "group": args.group, "pid": os.getpid(),
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
    with open(args.record, "w") as f:
        json.dump(record, f, indent=2)
    print(f"monitor run: {url or record['path']} (record: {args.record})", flush=True)

    stop = {"flag": False}

    def _stop(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    if psutil is not None:
        psutil.cpu_percent(interval=None)  # prime the counter
    started = time.time()
    try:
        while True:
            sample = {**system_sample(), **gpu_sample(), "monitor/uptime_s": round(time.time() - started, 1)}
            run.log(sample)
            print(json.dumps(sample), flush=True)
            if args.once or stop["flag"]:
                break
            deadline = time.time() + args.interval
            while time.time() < deadline and not stop["flag"]:
                time.sleep(0.2)
            if stop["flag"]:
                break
    finally:
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
