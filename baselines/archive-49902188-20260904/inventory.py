"""Inventory existing local baseline evidence without modifying original files."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
baselines = root.parent
entries = []
for path in sorted(baselines.rglob("*")):
    if not path.is_file() or path.is_symlink():
        continue
    if root in path.parents or "__pycache__" in path.parts:
        continue
    entries.append({"path": str(path.relative_to(baselines.parent)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
result = {"scope": "Existing local baseline files; NOT remote completeness verification",
          "files": len(entries), "bytes": sum(x["bytes"] for x in entries),
          "entries": entries}
(root / "local-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ("scope", "files", "bytes")}))
