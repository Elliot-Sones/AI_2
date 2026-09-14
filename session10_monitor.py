"""Send only GPU scalar telemetry to W&B; credentials stay on this Mac."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import wandb

OUT = Path(__file__).resolve().parent / "baselines/native-ppo-session10"
stop = False


def stopping(*_):
    global stop
    stop = True


def main():
    signal.signal(signal.SIGINT, stopping)
    signal.signal(signal.SIGTERM, stopping)
    run = wandb.init(project="utmist-ai2", name="native-ppo-session10-instance50149729",
                     group="native-ppo-session10", job_type="monitor", dir=str(OUT),
                     notes="Goal: test independent and combined native PPO throughput optimizations against the unchanged CUDA-lane baseline. Instance 50149729, RTX 4090. Only six GPU telemetry scalars are forwarded. Credentials stay on the Mac; training records remain offline and are downloaded locally, not cloud-synchronized.",
                     config={"instance_id": 50149729, "gpu": "RTX 4090", "credential_location": "local_only", "interval_seconds": 10},
                     settings=wandb.Settings(x_disable_stats=True))
    (OUT / "monitor_run.json").write_text(json.dumps({"id": run.id, "url": run.url,
        "path": f"{run.entity}/{run.project}/{run.id}", "pid": os.getpid()}, indent=2) + "\n")
    print("Monitor URL: " + run.url, flush=True)
    command = ["ssh", "-p", "29728", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=5", "root@ssh9.vast.ai",
               "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits"]
    try:
        while not stop:
            tick = time.time()
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
                values = [float(v.strip()) for v in result.stdout.strip().splitlines()[-1].split(",")]
                if len(values) != 6:
                    raise ValueError("Invalid scalar telemetry shape")
                row = dict(zip(("gpu/util", "gpu/mem_util", "gpu/mem_used_mb", "gpu/mem_total_mb", "gpu/power_w", "gpu/temp_c"), values))
                row.update({"monitor/remote_unix": tick, "monitor/telemetry_ok": 1})
            except Exception as error:
                row = {"monitor/remote_unix": tick, "monitor/telemetry_ok": 0, "monitor/error_type": type(error).__name__}
            with (OUT / "gpu-telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            run.log(row)
            print(json.dumps(row), flush=True)
            while not stop and time.time() < tick + 10:
                time.sleep(0.2)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
