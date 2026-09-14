"""Preserve only pinned session11 inventory's exact config/core-log targets.

Standard library only, no remote operations and no recursive collection. The
single, never-overwritten tar contains mapping.json and hash-deduplicated regular
content files. Known credential patterns reject whole files without echoing data;
this conservative screen is not a guarantee that arbitrary text is secret-free.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import uuid

from session10_capture_external_logs import (
    CORE_LOG_ROOT, LOG_NAME, SECRET_PATTERN, _open_directory,
    _read_regular, _safe_path, _signature,
)


SESSION_ROOT = Path("/workspace/AI_2/native-capacity-session11")
MAX_FILES = 64
MAX_BYTES = 100_000_000
MANIFEST_MAX_BYTES = 10_000_000
VARIANTS = tuple(f"arch-{arch}_seed{seed}" for arch in
                 ("512-512-256", "1024-1024-1024", "1536-1536-1536")
                 for seed in (145, 146, 147))


def _link_snapshot(path, expected_target):
    """Read a literal symlink through no-follow parent directory descriptors."""
    parent_fd = _open_directory(path.parent)
    try:
        before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISLNK(before.st_mode):
            raise ValueError("inventoried link is no longer a symlink")
        target = os.readlink(path.name, dir_fd=parent_fd)
        after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if target != expected_target or _signature(before) != _signature(after):
            raise ValueError("literal link differs from inventory or changed while reading")
        return target, _signature(before)
    finally:
        os.close(parent_fd)


def capture(manifest_path, session_root=SESSION_ROOT, *, output_tar,
            expected_manifest_sha256, core_log_root=CORE_LOG_ROOT,
            max_files=MAX_FILES, max_bytes=MAX_BYTES):
    """Capture one stable, pinned inventory; root injection is for local fixtures.

    Public CLI roots are fixed. Limits may only be reduced. Absolute targets must
    be one of nine exact configs or an exact timestamp-named core-debug log.
    Relative links are verified literally but their targets are never followed.
    """
    if (type(max_files) is not int or not 0 < max_files <= MAX_FILES
            or type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES):
        raise ValueError("capture bounds cannot exceed 64 targets / 100 MB")
    root = Path(_safe_path(str(session_root), absolute=True))
    logs_root = Path(_safe_path(str(core_log_root), absolute=True))
    output = Path(_safe_path(str(output_tar), absolute=True))
    if not output.name.endswith(".tar.gz"):
        raise ValueError("output must be a unique .tar.gz path")
    manifest_bytes, manifest_signature = _read_regular(manifest_path, MANIFEST_MAX_BYTES)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != expected_manifest_sha256:
        raise ValueError("manifest digest mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "ai2.session9.download-integrity.v1" or manifest.get("finalized") is not True:
        raise ValueError("a finalized session9 inventory is required")
    if manifest.get("source_roots") != {"artifacts": str(root / "artifacts"), "wandb": str(root / "wandb")}:
        raise ValueError("manifest source roots mismatch")
    configs = {str(root / "variants" / variant / "config.yaml") for variant in VARIANTS}
    seen_paths, targets, literal_links, external_links = set(), set(), [], []
    for entry in manifest["files"]:
        relative = _safe_path(entry["path"], absolute=False)
        if relative.parts[0] not in ("artifacts", "wandb") or len(relative.parts) < 2:
            raise ValueError("manifest path is outside artifact/W&B roots")
        if str(relative) in seen_paths:
            raise ValueError("duplicate manifest path")
        seen_paths.add(str(relative))
        if entry["type"] != "symlink":
            continue
        target = entry["target"]
        parsed = _safe_path(target, absolute=isinstance(target, str) and target.startswith("/"))
        literal_links.append({"link_path": str(relative), "original_target": target})
        if not parsed.is_absolute():
            continue
        if target not in configs and not (str(parsed.parent) == str(logs_root) and LOG_NAME.fullmatch(parsed.name)):
            raise ValueError("unallowlisted absolute target")
        targets.add(target)
        external_links.append({"link_path": str(relative), "original_target": target})
    if manifest.get("symlink_count") != len(literal_links):
        raise ValueError("manifest symlink count mismatch")
    if not targets or len(targets) > max_files:
        raise ValueError("target count is empty or exceeds bound")
    link_snapshots = {link["link_path"]: _link_snapshot(root / link["link_path"], link["original_target"])
                      for link in literal_links}
    snapshots, contents, target_hashes = {}, {}, {}
    total = 0
    for target in sorted(targets):
        data, signature = _read_regular(target, max_bytes - total)
        if SECRET_PATTERN.search(data):
            raise ValueError("credential-like content rejected; no content echoed")
        target_digest = hashlib.sha256(data).hexdigest()
        if target_digest in contents and contents[target_digest] != data:
            raise ValueError("content digest collision")
        contents[target_digest] = data
        target_hashes[target] = target_digest
        snapshots[target] = (data, signature)
        total += len(data)
    for link in external_links:
        target = link["original_target"]
        link.update(sha256=target_hashes[target], size=len(snapshots[target][0]),
                    archive_member=f"content/sha256/{target_hashes[target]}")
    mapping = {
        "schema": "ai2.session11.external-link-targets.v1", "status": "completed",
        "manifest_sha256": digest, "source_roots": manifest["source_roots"],
        "literal_symlink_count": len(literal_links), "absolute_symlink_count": len(external_links),
        "unique_target_count": len(targets), "unique_content_count": len(contents),
        "total_bytes": total, "unique_content_bytes": sum(map(len, contents.values())),
        "links": external_links,
        "relative_links_not_followed": [link for link in literal_links if not link["original_target"].startswith("/")],
        "credential_screening": "known patterns only; not a guarantee of secret-free text",
        "consistency": "manifest, literal links and exact target bytes/stat checked twice before publication; snapshot, not a future-write lock",
    }
    archive_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=archive_buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            members = {"mapping.json": (json.dumps(mapping, indent=2, sort_keys=True) + "\n").encode()}
            members.update({f"content/sha256/{key}": data for key, data in contents.items()})
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o600
                archive.addfile(info, io.BytesIO(data))
    archive_bytes = archive_buffer.getvalue()
    output_fd = _open_directory(output.parent, create=True)
    temporary = None
    try:
        try:
            os.stat(output.name, dir_fd=output_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(output.name)
        name = f".{output.name}.tmp-{uuid.uuid4().hex}"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=output_fd)
        temporary = name
        with os.fdopen(fd, "wb") as handle:
            handle.write(archive_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        for target, snapshot in snapshots.items():
            if _read_regular(target, max_bytes) != snapshot:
                raise ValueError("target changed before publication")
        if _read_regular(manifest_path, MANIFEST_MAX_BYTES) != (manifest_bytes, manifest_signature):
            raise ValueError("manifest changed before publication")
        for link in literal_links:
            if _link_snapshot(root / link["link_path"], link["original_target"]) != link_snapshots[link["link_path"]]:
                raise ValueError("literal link changed before publication")
        # Hard-link publication is atomic and fails if the final name exists.
        os.link(temporary, output.name, src_dir_fd=output_fd, dst_dir_fd=output_fd, follow_symlinks=False)
        os.fsync(output_fd)
    finally:
        if temporary is not None:
            os.unlink(temporary, dir_fd=output_fd)
        os.close(output_fd)
    return {**mapping, "archive_path": str(output), "archive_size": len(archive_bytes),
            "archive_sha256": hashlib.sha256(archive_bytes).hexdigest()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="unique absolute .tar.gz path; never overwritten")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    receipt = capture(args.manifest, output_tar=args.output, expected_manifest_sha256=args.manifest_sha256)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
