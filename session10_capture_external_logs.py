"""Capture the exact session10 external-link targets, never recursive trees.

Linux/macOS, Python 3.11+, standard library only. The CLI requires a pinned
manifest digest. Credential-pattern screening is conservative, not a proof that
arbitrary diagnostic text contains no secrets. Suspect files are rejected whole.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile


SESSION_ROOT = Path("/workspace/AI_2/native-ppo-session10")
CORE_LOG_ROOT = Path("/root/.cache/wandb/logs")
EXPECTED_COUNTS = (99, 96, 49)
MAX_FILES = 50
MAX_BYTES = 100_000_000
LOG_NAME = re.compile(r"core-debug-[0-9]{8}_[0-9]{6}\.log\Z")
SECRET_PATTERN = re.compile(
    rb"(?i)(authorization[\"'\s]*[:=][\"'\s]*(?:bearer|basic)\s+\S+|"
    rb"(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)[\"'\s]*[:=]"
    rb"[\"'\s]*(?!redacted\b|none\b|null\b|\*|<)[a-z0-9_/+=.-]{12,}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


def _safe_path(value, *, absolute):
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("invalid path")
    path = PurePosixPath(value)
    if path.is_absolute() != absolute or ".." in path.parts or str(path) != value:
        raise ValueError("non-canonical or traversing path")
    if str(path) in ("/", "."):
        raise ValueError("empty path")
    return path


def _open_directory(path, *, create=False):
    """Walk from / using descriptor-relative, no-follow opens at every level."""
    parsed = _safe_path(str(path), absolute=True)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in parsed.parts[1:]:
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_regular(path, limit):
    path = Path(path)
    parent_fd = _open_directory(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise ValueError("target is not a bounded regular file")
            data = handle.read(limit + 1)
            after = os.fstat(handle.fileno())
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if len(data) > limit or len(data) != before.st_size:
            raise ValueError("target exceeds bound or changed while reading")
        if _signature(before) != _signature(after) or _signature(before) != _signature(current):
            raise ValueError("target changed while reading")
        return data, _signature(before)
    finally:
        os.close(parent_fd)


def capture(manifest_path, session_root=SESSION_ROOT, *, core_log_root=CORE_LOG_ROOT,
            expected_manifest_sha256, expected_counts=EXPECTED_COUNTS,
            max_files=MAX_FILES, max_bytes=MAX_BYTES):
    """Publish tar + mapping only after all exact targets pass fail-closed checks.

    Root injection is for local fixtures; the CLI never exposes a cache-root
    override. Input links are inventoried, not followed or rewritten.
    """
    if not 0 < max_files <= MAX_FILES or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError("capture bounds cannot exceed 50 files / 100 MB")
    session_root = Path(_safe_path(str(session_root), absolute=True))
    core_log_root = Path(_safe_path(str(core_log_root), absolute=True))
    manifest_bytes, manifest_signature = _read_regular(manifest_path, 10_000_000)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_digest != expected_manifest_sha256:
        raise ValueError("manifest digest mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("finalized") is not True:
        raise ValueError("manifest must be finalized")
    if "source_roots" in manifest and manifest["source_roots"] != {
        "artifacts": str(session_root / "artifacts"), "wandb": str(session_root / "wandb")
    }:
        raise ValueError("manifest source roots mismatch")
    config = str(session_root / "variants/corrected/control.yaml")
    links = []
    members = {}
    seen_paths = set()
    literal_count = 0
    for entry in manifest["files"]:
        path = entry["path"]
        _safe_path(path, absolute=False)
        if path in seen_paths:
            raise ValueError("duplicate manifest path")
        seen_paths.add(path)
        if entry["type"] != "symlink":
            continue
        literal_count += 1
        target = entry["target"]
        absolute = target.startswith("/")
        parsed = _safe_path(target, absolute=absolute)
        if not absolute:
            continue
        if target == config:
            member = "external-targets/control.yaml"
        elif str(parsed.parent) == str(core_log_root) and LOG_NAME.fullmatch(parsed.name):
            member = "external-targets/wandb-core/" + parsed.name
        else:
            raise ValueError("unallowlisted absolute target: " + target)
        members[target] = member
        links.append({"link_path": path, "original_target": target, "archive_member": member})
    if (literal_count, len(links), len(members)) != tuple(expected_counts):
        raise ValueError("manifest link/absolute/unique counts mismatch")
    if len(members) > max_files or not members:
        raise ValueError("capture target count exceeds bound or is empty")
    snapshots = {}
    total = 0
    for target in sorted(members):
        data, signature = _read_regular(target, max_bytes - total)
        if SECRET_PATTERN.search(data):
            raise ValueError("credential-like content rejected; no content echoed")
        snapshots[target] = (data, signature)
        total += len(data)
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as archive:
        for target, member in sorted(members.items()):
            data = snapshots[target][0]
            info = tarfile.TarInfo(member)
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
    for target, snapshot in snapshots.items():
        if _read_regular(target, max_bytes) != snapshot:
            raise ValueError("target changed before publication")
    if _read_regular(manifest_path, 10_000_000) != (manifest_bytes, manifest_signature):
        raise ValueError("manifest changed before publication")
    for link in links:
        data = snapshots[link["original_target"]][0]
        link.update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    archive_bytes = tar_buffer.getvalue()
    receipt = {
        "schema": "ai2.session10.external-link-targets.v1", "status": "completed",
        "manifest_sha256": manifest_digest, "literal_symlink_count": literal_count,
        "absolute_symlink_count": len(links), "unique_target_count": len(members),
        "total_bytes": total, "links": links,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_size": len(archive_bytes),
        "credential_screening": "known patterns only; not a guarantee of secret-free text",
        "consistency": "stable file identity/stat/content verified twice; snapshot, not a future-write lock",
    }
    output_fd = _open_directory(session_root / "artifacts/provenance", create=True)
    try:
        outputs = {"external-link-targets.tar.gz": archive_bytes,
                   "external-link-targets.json": (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()}
        for name in outputs:
            try:
                os.stat(name, dir_fd=output_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise FileExistsError(name)
        for name, data in outputs.items():
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=output_fd)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        os.close(output_fd)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--session-root", type=Path, default=SESSION_ROOT)
    args = parser.parse_args()
    receipt = capture(args.manifest, args.session_root,
                      expected_manifest_sha256=args.manifest_sha256)
    print(json.dumps({key: receipt[key] for key in (
        "status", "unique_target_count", "total_bytes", "archive_sha256")}))


if __name__ == "__main__":
    main()
