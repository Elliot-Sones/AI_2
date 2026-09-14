#!/usr/bin/env python3
"""One authorized, source-recorded 4090 rental with the existing cost guard."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "baselines/native-ppo-session10"
CLI = "/Library/Frameworks/Python.framework/Versions/3.13/bin/vastai"
CREDIT_HELPER = Path("/Users/elliot18/Projects/ml-research/AI_2-throughput-scaling-20260906/baselines/throughput-scaling-20260906/native-lanes-session9/session9_query_credit.py")
LABEL = "ai2-native-ppo-session10-20260907"
REMOTE = "/workspace/AI_2/native-ppo-session10"


def cli(*args):
    result = subprocess.run([CLI, *args], capture_output=True, text=True, timeout=45, check=True)
    return json.loads(result.stdout)


def watch():
    import campaign_cloud
    campaign_cloud.REMOTE_ROOT = REMOTE
    campaign_cloud.MAX_SECONDS = 7200
    campaign_cloud.RETRIEVAL_RESERVE_SECONDS = 300
    return campaign_cloud.main(["--start", "--instance-record", str(OUT / "instance.json"),
                                "--state-dir", str(OUT / "guard"), "--poll-seconds", "15"])


def start(machine_id):
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / "instance.json"
    if receipt_path.exists():
        raise RuntimeError("An instance receipt already exists; do not rent a second instance")
    instances = cli("show", "instances", "--raw")
    for item in instances:
        states = {str(item.get(key, "")).lower() for key in ("actual_status", "cur_state", "intended_status", "next_state")}
        if states & {"running", "loading", "creating"}:
            raise RuntimeError("A running or pending instance exists; refusing overlap")
    credit = json.loads(subprocess.run([sys.executable, str(CREDIT_HELPER)], capture_output=True, text=True, check=True, timeout=45).stdout)
    if not credit["ok"]:
        raise RuntimeError("Credit audit failed")
    spent = 12.315037 - credit["balance_fields"]["credit"]
    if not 0 <= spent <= 10:
        raise RuntimeError("Insufficient authorized budget for conservative $2 batch ceiling")
    offers = cli("search", "offers", "gpu_name=RTX_4090 num_gpus=1 rentable=true verified=true",
                 "--storage", "40", "--order", "dph_total", "--limit", "20", "--raw")
    offer = next((x for x in offers if int(x["machine_id"]) == machine_id), None)
    if offer is None or not 0 < offer["dph_total"] <= 0.40:
        raise RuntimeError("Selected offer unavailable or outside price ceiling")
    safe_keys = ("id", "machine_id", "gpu_name", "num_gpus", "cpu_name", "cpu_cores_effective", "cpu_ram", "dph_total", "dph_base", "storage_cost", "inet_down_cost", "inet_up_cost", "geolocation", "reliability2")
    receipt = {"offer": {k: offer.get(k) for k in safe_keys}, "cumulative_spent": spent,
               "credit_audit": credit, "user_credit_ceiling": 12, "planned_batch_ceiling": 2,
               "batch_max_seconds": 7200, "disk_gb": 40, "label": LABEL,
               "image": "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel",
               "hourly_price": offer["dph_total"], "created_unix": time.time(),
               "created_by_campaign": True, "retained_stopped_ids": [int(x["id"]) for x in instances]}
    (OUT / "pre-rental-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result = cli("create", "instance", str(offer["id"]), "--image", receipt["image"],
                 "--disk", "40", "--ssh", "--direct", "--label", LABEL, "--cancel-unavail", "--raw")
    if result.get("success") is not True or not result.get("new_contract"):
        raise RuntimeError("Provider did not acknowledge rental")
    receipt["id"] = int(result["new_contract"])
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        for _ in range(18):
            row = next(x for x in cli("show", "instances", "--raw") if int(x["id"]) == receipt["id"])
            if row.get("ssh_host") and row.get("ssh_port"):
                receipt.update(ssh_host=row["ssh_host"], ssh_port=int(row["ssh_port"]))
                break
            time.sleep(5)
        else:
            raise RuntimeError("Instance has no connection metadata after bounded wait")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        with (OUT / "guard-launch.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--watch"],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (OUT / "guard-pid.json").write_text(json.dumps({"pid": process.pid, "instance_id": receipt["id"]}) + "\n")
    except BaseException:
        subprocess.run([CLI, "stop", "instance", str(receipt["id"])], capture_output=True, text=True, timeout=45)
        raise
    print(json.dumps({k: receipt[k] for k in ("id", "label", "hourly_price", "ssh_host", "ssh_port", "cumulative_spent")}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--machine", type=int)
    action.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    raise SystemExit(watch() if args.watch else start(args.machine))
