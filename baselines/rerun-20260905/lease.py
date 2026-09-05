"""Bounded authorized benchmark rental. Never destroys an instance."""
import json
from pathlib import Path
import runpy
import signal
import time

ROOT = Path(__file__).resolve().parent
helper = runpy.run_path(str(ROOT.parent / "pipeline-profile-20260904/instance_guard.py"))
cli = helper["cli"]
state_path = ROOT / "lease.json"
if state_path.exists():
    raise RuntimeError("Refusing duplicate rental")
start = time.time()
reply = cli("create", "instance", "24912145", "--image", "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime",
            "--disk", "20", "--ssh", "--direct", "--cancel-unavail", "--label", "ai2-rerun-20260905", "--raw")
if not reply.get("success") or not reply.get("new_contract"):
    raise RuntimeError("Rental not confirmed; inspect account before retrying")
instance_id = int(reply["new_contract"])
helper["current"].__globals__["INSTANCE_ID"] = instance_id
state = {"instance_id": instance_id, "started_unix": start, "deadline_unix": start+1740,
         "deletion_authorized": False, "status": "created"}
def save():
    state_path.write_text(json.dumps(state, indent=2)+"\n")
    print(json.dumps(state), flush=True)
save()
stop = [False]
signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
try:
    while time.time() < start+1740 and not stop[0] and not (ROOT/"stop.request").exists():
        time.sleep(2)
finally:
    for attempt in range(10):
        try:
            cli("stop", "instance", str(instance_id), "--raw")
            current = helper["current"]()
            state["provider"] = current
            if current["actual_status"] == "exited" and current["cur_state"] == "stopped":
                state.update(status="stopped_verified_not_deleted", elapsed_seconds=time.time()-start)
                save()
                break
        except Exception:
            state["stop_error"] = "Provider error withheld to protect credentials"
        time.sleep(3)
    else:
        state["status"] = "stop_verification_failed"
        save()
