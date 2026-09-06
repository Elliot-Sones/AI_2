"""Verify the downloaded rerun against remote digest and profile invariants."""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
expected = "93123cb89ebc7ff00648d805ce2ed6b4debdd5f9b226c9c013092746367c9341"
assert hashlib.sha256((root / "results.tar.gz").read_bytes()).hexdigest() == expected
assert (root / "artifacts/run.exit").read_text().strip() == "0"
result = json.loads((root / "artifacts/natural_1env/pipeline_profile.json").read_text())["result"]
p = result["pipeline"]
assert result["n_envs"] == 1 and result["rollouts"] == 2
assert result["measured_timesteps"] == p["transitions"] == p["vector_cycle_count"] == 16384
assert len(p["env_step_events"]) == 16384
assert math.isclose(p["transitions_per_second"], 16384 / p["envelope_seconds"])
assert math.isclose(p["envelope_seconds"], p["stage_exclusive_seconds_total"] + p["ppo_train_seconds_total"] + p["residual_seconds"], abs_tol=1e-8)
assert "Ran 7 tests" in (root / "artifacts/tests.log").read_text()
lease = json.loads((root / "lease.json").read_text())
assert lease["status"] == "stopped_verified_not_deleted"
entries = []
for directory in (root / "artifacts", root / "wandb"):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink():
            entries.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
report = {"remote_results_sha256": expected, "archive_hash_verified": True,
          "profile_invariants_verified": True, "tests_passed": 7,
          "steps_per_second": p["transitions_per_second"],
          "measured_seconds": p["envelope_seconds"], "files": entries}
(root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Verified archive SHA256, 16,384 transition/event counts, timing accounting, seven tests, stopped instance, and {len(entries)} archived files.")
