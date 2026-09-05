"""Start exactly the authorized retained instance and guarantee stop requests.

All CLI output is captured: Vast CLI errors may contain its API credential.
No destroy operation is present. Local stop.request ends the run early.
"""
import datetime
import json
from pathlib import Path
import subprocess
import signal
import time

INSTANCE_ID = 49902188
ROOT = Path(__file__).resolve().parent
STATE = ROOT / "session.json"


def cli(*args):
    result = subprocess.run(["vastai", *args], capture_output=True, text=True, timeout=25)
    if result.returncode:
        raise RuntimeError("Vast command failed; output withheld to protect credentials")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"command_completed": True}


def current():
    payload = cli("show", "instances", "--raw")
    rows = payload if isinstance(payload, list) else payload.get("instances", [])
    matches = [row for row in rows if row.get("id") == INSTANCE_ID]
    if len(matches) != 1:
        raise RuntimeError("Exact retained instance is missing or ambiguous")
    return {key: matches[0].get(key) for key in ("id", "actual_status", "cur_state", "intended_status", "dph_total", "storage_cost", "ssh_host", "ssh_port", "public_ipaddr", "ports")}


def save(state):
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)


def main():
    if STATE.exists() or (ROOT / "stop.request").exists():
        raise RuntimeError("Refusing to overwrite an existing lifecycle run")
    initial = current()
    if initial["cur_state"] != "stopped":
        raise RuntimeError("Expected stopped retained instance; refusing ambiguous lifecycle")
    if float(initial["dph_total"]) > 0.32:
        raise RuntimeError("Running rate exceeds the approved estimate")
    start = time.time()
    state = {"instance_id": INSTANCE_ID, "deletion_authorized": False, "started_unix": start,
             "started_utc": datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat(),
             "maximum_seconds": 600, "stop_request_deadline_unix": start + 540,
             "quoted_running_rate": initial["dph_total"], "initial": initial, "status": "starting"}
    save(state)
    stop_signal = [False]
    def request_stop(signum, frame):
        stop_signal[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        cli("start", "instance", str(INSTANCE_ID), "--raw")
        state["status"] = "start_requested"
        save(state)
        while time.time() < start + 540 and not stop_signal[0] and not (ROOT / "stop.request").exists():
            time.sleep(2)
    except Exception:
        state["run_error"] = "Start or monitoring failed; command output withheld"
    finally:
        state["status"] = "stopping"
        save(state)
        for attempt in range(8):
            try:
                cli("stop", "instance", str(INSTANCE_ID), "--raw")
                final = current()
                state["last_provider_state"] = final
                if final["actual_status"] == "exited" and final["cur_state"] == "stopped":
                    state.update(status="stopped_verified_not_deleted", stopped_unix=time.time(), elapsed_seconds=time.time()-start)
                    state["estimated_running_cost_usd"] = state["elapsed_seconds"] / 3600 * float(initial["dph_total"])
                    save(state)
                    return
            except Exception:
                state["last_stop_error"] = "Status or stop command failed; credential-bearing output withheld"
            time.sleep(3)
        state["status"] = "stop_requested_verification_incomplete"
        save(state)
        raise RuntimeError("Must independently verify instance stopped")


if __name__ == "__main__":
    main()
