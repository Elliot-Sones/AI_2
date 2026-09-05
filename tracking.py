#!/usr/bin/env python3
"""Experiment tracking for AI_2 training runs (Weights & Biases).

Every helper here is a no-op when tracking is off, so training never fails
because of the dashboard. Tracking is on when all of these hold:

* ``wandb.enabled`` is true in the config (default: true),
* the ``wandb`` package is installed,
* a login exists (``WANDB_API_KEY`` or an ``api.wandb.ai`` entry in ~/.netrc),
  or ``WANDB_MODE=offline`` is set for a later ``wandb sync``.

Run description (what this run is for) comes from ``WANDB_NOTES`` or
``wandb.notes`` in the config. Tags come from ``WANDB_TAGS`` (comma separated)
plus ``wandb.tags``. ``WANDB_NAME`` names the run and ``WANDB_RUN_GROUP``
groups runs (use the cloud run id). A JSON record of the run (id, path, url,
notes) is written next to the results so other tools can find it later.

CLI:
    python tracking.py info --run results/ppo_utmist/wandb_run.json
    python tracking.py append-notes --run <record.json | entity/project/id> --text "Outcome: ..."
    python tracking.py upload-file --run <record.json | entity/project/id> --file summary.json
"""

import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
import time

try:
    import wandb
except Exception:  # not installed or broken install: tracking stays off
    wandb = None


RECORD_NAME = "wandb_run.json"
HISTORY_NAME = "wandb_runs.jsonl"
EVAL_COLUMNS = ["opponent", "win_rate", "wins", "losses", "draws", "avg_damage"]
DEFAULT_PROJECT = "utmist-ai2"

_active = None          # live wandb run created by init_run
_record_path = None     # wandb_run.json for the live run
_settings = {}          # resolved "wandb" config section for the live run


# ---------------------------------------------------------------------------
# Enablement
# ---------------------------------------------------------------------------

def has_login(environ=None, netrc_path=None) -> bool:
    """True when a W&B API key is reachable via env var or ~/.netrc."""
    env = os.environ if environ is None else environ
    if env.get("WANDB_API_KEY"):
        return True
    path = netrc_path or env.get("NETRC") or os.path.join(os.path.expanduser("~"), ".netrc")
    try:
        with open(path) as f:
            return "api.wandb.ai" in f.read()
    except OSError:
        return False


def _wandb_config(params: dict) -> dict:
    return dict((params or {}).get("wandb") or {})


def tracking_enabled(params: dict, environ=None, netrc_path=None) -> bool:
    """Config flag on, wandb importable, and either a login or offline mode."""
    env = os.environ if environ is None else environ
    if not _wandb_config(params).get("enabled", True):
        return False
    if wandb is None:
        return False
    mode = (env.get("WANDB_MODE") or "").strip().lower()
    if mode == "disabled":
        return False
    if mode == "offline":
        return True
    return has_login(environ=env, netrc_path=netrc_path)


def active_run():
    return _active


def scrub_empty_wandb_env(environ=None) -> None:
    """Drop empty WANDB_* run variables. wandb parses them itself, and an
    empty WANDB_TAGS produces an invalid empty tag that aborts init."""
    env = os.environ if environ is None else environ
    for key in ("WANDB_TAGS", "WANDB_NOTES", "WANDB_NAME", "WANDB_RUN_GROUP", "WANDB_ENTITY", "WANDB_PROJECT"):
        if key in env and not env[key].strip():
            del env[key]


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _auto_notes(params: dict, phase: str, kind: str) -> str:
    ppo = (params or {}).get("ppo_settings", {}) or {}
    envs = (params or {}).get("environment_settings", {}) or {}
    return (
        "No run description was provided (set WANDB_NOTES, or wandb.notes in the config). "
        f"Auto summary: {kind} training, phase {phase}, "
        f"checkpoint {ppo.get('model_checkpoint', '0')}, "
        f"{envs.get('n_envs', '?')} environments, {ppo.get('time_steps', '?')} timesteps."
    )


def _split_tags(value) -> list:
    if not value:
        return []
    if isinstance(value, str):
        value = value.split(",")
    return [str(t).strip() for t in value if str(t).strip()]


def _run_path(run) -> str:
    return "/".join([getattr(run, "entity", None) or "", getattr(run, "project", None) or "", run.id])


def _record_dict(run, notes: str) -> dict:
    url = None
    try:
        url = run.url
    except Exception:
        pass
    return {
        "id": run.id,
        "name": run.name,
        "project": run.project,
        "entity": getattr(run, "entity", None) or None,
        "path": _run_path(run),
        "url": url,
        "group": getattr(run, "group", None) or None,
        "tags": list(run.tags or ()),
        "notes": notes,
        "mode": (os.environ.get("WANDB_MODE") or "online").lower(),
        "dir": run.dir,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _write_record(record: dict, results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, RECORD_NAME)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    with open(os.path.join(results_dir, HISTORY_NAME), "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def init_run(params: dict, phase_key, kind: str, results_dir: str,
             config_path: str = None, extra_config: dict = None):
    """Start a W&B run for one training phase. Returns the run, or None."""
    global _active, _record_path, _settings
    if not tracking_enabled(params):
        return None
    if _active is not None:
        finish()

    cfg = _wandb_config(params)
    scrub_empty_wandb_env()
    env = os.environ
    phase = str(phase_key)

    notes = (env.get("WANDB_NOTES") or "").strip() or str(cfg.get("notes") or "").strip()
    tags = [f"phase-{phase}", kind] + _split_tags(cfg.get("tags")) + _split_tags(env.get("WANDB_TAGS"))
    if not notes:
        notes = _auto_notes(params, phase, kind)
        tags.append("notes-missing")
    tags = list(dict.fromkeys(tags))

    name = (env.get("WANDB_NAME") or "").strip() or f"{kind}-phase{phase}-{time.strftime('%Y%m%d-%H%M%S')}"
    group = (env.get("WANDB_RUN_GROUP") or "").strip() or cfg.get("group") or None

    config = dict(params or {})
    config.update({
        "phase": phase,
        "kind": kind,
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
    })
    if extra_config:
        config.update(extra_config)

    try:
        run = wandb.init(
            project=cfg.get("project") or DEFAULT_PROJECT,
            entity=cfg.get("entity") or None,
            name=name,
            group=group,
            job_type=cfg.get("job_type") or "train",
            tags=tags,
            notes=notes,
            config=config,
            sync_tensorboard=True,
        )
    except Exception as e:
        print(f"⚠️ W&B init failed, continuing without tracking: {e}")
        return None

    if config_path and os.path.isfile(config_path):
        try:
            abs_path = os.path.abspath(config_path)
            wandb.save(abs_path, base_path=os.path.dirname(abs_path), policy="now")
        except Exception as e:
            print(f"⚠️ W&B could not save config file: {e}")

    _active = run
    _settings = cfg
    record = _record_dict(run, notes)
    _record_path = _write_record(record, results_dir)
    print(f"📈 W&B run: {record['url'] or record['path']} ({record['mode']})")
    if "notes-missing" in tags:
        print("⚠️ W&B run has no description. Set WANDB_NOTES to say what this run is for.")
    return run


def _update_record_notes(notes: str) -> None:
    if not _record_path or not os.path.isfile(_record_path):
        return
    try:
        with open(_record_path) as f:
            record = json.load(f)
        record["notes"] = notes
        with open(_record_path, "w") as f:
            json.dump(record, f, indent=2)
    except Exception:
        pass


def append_notes(text: str) -> bool:
    """Append a paragraph to the live run's description."""
    if _active is None or not text:
        return False
    try:
        current = _active.notes or ""
        _active.notes = f"{current}\n\n{text}".strip() if current else text.strip()
        _update_record_notes(_active.notes)
        return True
    except Exception as e:
        print(f"⚠️ W&B could not update notes: {e}")
        return False


def log_summary(**values) -> None:
    if _active is None or not values:
        return
    try:
        _active.summary.update(values)
    except Exception as e:
        print(f"⚠️ W&B could not update summary: {e}")


def finish(exit_code: int = 0) -> None:
    global _active, _record_path, _settings
    run, _active = _active, None
    _record_path, _settings = None, {}
    if run is None:
        return
    try:
        run.finish(exit_code=exit_code)
    except Exception as e:
        print(f"⚠️ W&B finish failed: {e}")


# ---------------------------------------------------------------------------
# Media, tables, artifacts
# ---------------------------------------------------------------------------

def _payload(values: dict, step) -> dict:
    # With TensorBoard syncing on, W&B keys charts by "global_step"; use the
    # same key so custom logs line up with the SB3 scalars.
    if step is not None:
        values["global_step"] = int(step)
    return values


def log_video(path: str, step=None, caption: str = None, key: str = "video/demo") -> bool:
    if _active is None or not _settings.get("upload_videos", True):
        return False
    if not path or not os.path.isfile(path):
        return False
    try:
        _active.log(_payload({key: wandb.Video(path, format="mp4", caption=caption)}, step))
        return True
    except Exception as e:
        print(f"⚠️ W&B could not log video: {e}")
        return False


def log_eval_table(rows: list, step=None) -> int:
    """rows: dicts with keys opponent, win_rate, wins, losses, draws, avg_damage."""
    if _active is None or not rows:
        return 0
    try:
        table = wandb.Table(columns=EVAL_COLUMNS)
        for row in rows:
            table.add_data(*[row.get(col) for col in EVAL_COLUMNS])
        _active.log(_payload({"eval/table": table}, step))
        return len(rows)
    except Exception as e:
        print(f"⚠️ W&B could not log eval table: {e}")
        return 0


def log_checkpoint(path: str, step=None, aliases: list = None, name: str = None, metadata: dict = None):
    """Upload a model checkpoint as a versioned W&B artifact. Returns the artifact."""
    if _active is None or not _settings.get("upload_checkpoints", True):
        return None
    if not path or not os.path.isfile(path):
        return None
    try:
        meta = {"step": step, "file": os.path.basename(path)}
        meta.update(metadata or {})
        artifact = wandb.Artifact(name or f"model-{_active.id}", type="model", metadata=meta)
        artifact.add_file(path)
        _active.log_artifact(artifact, aliases=list(dict.fromkeys(["latest"] + (aliases or []))))
        return artifact
    except Exception as e:
        print(f"⚠️ W&B could not upload checkpoint: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI (for use after a run has finished, e.g. by the cloud skill)
# ---------------------------------------------------------------------------

def _load_record(ref: str):
    if ref and os.path.isfile(ref):
        with open(ref) as f:
            return json.load(f)
    return None


def _resolve_run_path(ref: str) -> str:
    record = _load_record(ref)
    if record:
        return record["path"]
    return ref


def _cli_info(args) -> int:
    record = _load_record(args.run)
    if record is None:
        if wandb is None:
            print("wandb is not installed", file=sys.stderr)
            return 1
        run = wandb.Api().run(args.run)
        record = {
            "id": run.id, "name": run.name, "project": run.project, "entity": run.entity,
            "path": "/".join(run.path), "url": run.url, "state": run.state,
            "tags": list(run.tags), "notes": run.notes, "summary": dict(run.summary),
        }
    print(json.dumps(record, indent=2, default=str))
    return 0


def _cli_append_notes(args) -> int:
    if wandb is None:
        print("wandb is not installed", file=sys.stderr)
        return 1
    text = args.text
    if args.file:
        with open(args.file) as f:
            text = f.read()
    text = (text or "").strip()
    if not text:
        print("nothing to append: pass --text or --file", file=sys.stderr)
        return 1
    path = _resolve_run_path(args.run)
    run = wandb.Api().run(path)
    run.notes = f"{run.notes}\n\n{text}".strip() if run.notes else text
    run.update()
    record = _load_record(args.run)
    if record is not None:
        record["notes"] = run.notes
        with open(args.run, "w") as f:
            json.dump(record, f, indent=2)
    print(f"Notes updated for {run.url}")
    return 0


def _cli_upload_file(args) -> int:
    if wandb is None:
        print("wandb is not installed", file=sys.stderr)
        return 1
    if not os.path.isfile(args.file):
        print(f"no such file: {args.file}", file=sys.stderr)
        return 1
    run = wandb.Api().run(_resolve_run_path(args.run))
    run.upload_file(args.file, root=os.path.dirname(os.path.abspath(args.file)))
    print(f"Uploaded {os.path.basename(args.file)} to {run.url}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Weights & Biases helpers for AI_2 runs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="print the run record (or fetch the run by path)")
    p.add_argument("--run", required=True, help="wandb_run.json path, or entity/project/run_id")
    p.set_defaults(func=_cli_info)

    p = sub.add_parser("append-notes", help="append text to a run's description")
    p.add_argument("--run", required=True, help="wandb_run.json path, or entity/project/run_id")
    p.add_argument("--text", help="text to append")
    p.add_argument("--file", help="file whose contents to append")
    p.set_defaults(func=_cli_append_notes)

    p = sub.add_parser("upload-file", help="attach a file to a run")
    p.add_argument("--run", required=True, help="wandb_run.json path, or entity/project/run_id")
    p.add_argument("--file", required=True)
    p.set_defaults(func=_cli_upload_file)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
