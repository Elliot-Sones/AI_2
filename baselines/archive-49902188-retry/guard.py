"""Bounded archival retry; never destroy the retained instance."""
from pathlib import Path
import runpy

root = Path(__file__).resolve().parent
module = runpy.run_path(str(root.parent / "pipeline-profile-20260904/instance_guard.py"))
namespace = module["main"].__globals__
namespace["ROOT"] = root
namespace["STATE"] = root / "session.json"
module["main"]()
