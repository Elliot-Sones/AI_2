import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import checkpoint_hardlinks


class CheckpointHardlinkTests(unittest.TestCase):
    def test_dry_run_reports_exact_latest_final_duplicate_without_mutating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")

            report = checkpoint_hardlinks.plan_dedup(root)

            self.assertEqual(report["status"], "dry_run")
            self.assertEqual(report["candidate_count"], 1)
            self.assertEqual(report["recoverable_bytes"], 4)
            self.assertEqual((chunk / "latest.zip").read_bytes(), b"same")
            self.assertNotEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)
            self.assertIn("Restoration", report["restoration_recipe"])

    def test_dry_run_skips_already_same_inode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = root / "run" / "chunk01"
            chunk.mkdir(parents=True)
            (chunk / "final.zip").write_bytes(b"same")
            os.link(chunk / "final.zip", chunk / "latest.zip")
            write_result(chunk)

            report = checkpoint_hardlinks.plan_dedup(root)

            self.assertEqual(report["candidate_count"], 0)
            self.assertEqual(report["recoverable_bytes"], 0)
            self.assertEqual(report["skipped"][0]["reason"], "already_same_inode")

    def test_dry_run_skips_mismatched_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_completed_chunk(root / "run" / "chunk01", latest=b"left!", final=b"right")

            report = checkpoint_hardlinks.plan_dedup(root)

            self.assertEqual(report["candidate_count"], 0)
            self.assertEqual(report["skipped"][0]["reason"], "content_mismatch")

    def test_dry_run_skips_symlink_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = root / "run" / "chunk01"
            chunk.mkdir(parents=True)
            (chunk / "final.zip").write_bytes(b"same")
            (chunk / "actual_latest.zip").write_bytes(b"same")
            (chunk / "latest.zip").symlink_to("actual_latest.zip")
            write_result(chunk)

            report = checkpoint_hardlinks.plan_dedup(root)

            self.assertEqual(report["candidate_count"], 0)
            self.assertEqual(report["skipped"][0]["reason"], "symlink_checkpoint")

    def test_refuses_root_escape_from_symlinked_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            outside = Path(tmp) / "outside"
            outside.mkdir()
            root.mkdir()
            make_completed_chunk(outside / "chunk01", latest=b"same", final=b"same")
            (root / "linked_chunk").symlink_to(outside / "chunk01")

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "escapes root"):
                checkpoint_hardlinks.plan_dedup(root)

    def test_apply_requires_manifest_and_backup_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "manifest"):
                checkpoint_hardlinks.apply_dedup(root, manifest_path=None, backup_receipt_path=None)

    def test_apply_rejects_stale_manifest_hash_before_mutating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            manifest = write_manifest(root)
            receipt = write_receipt(root, manifest)
            (chunk / "latest.zip").write_bytes(b"changed")

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "input drift"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt)
            self.assertEqual((chunk / "latest.zip").read_bytes(), b"changed")
            self.assertNotEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)

    def test_apply_rejects_non_dry_run_manifest_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            manifest = write_manifest(root)
            receipt = write_receipt(root, manifest)
            data = json.loads(manifest.read_text())
            data["status"] = "applied"
            manifest.write_text(json.dumps(data))

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "dry_run"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt)
            self.assertNotEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)

    def test_apply_rejects_missing_backup_receipt_for_both_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            manifest = write_manifest(root)
            data = json.loads(manifest.read_text())
            receipt = root / "backup.json"
            receipt.write_text(
                json.dumps(
                    {
                        "independently_verified_local_backup": True,
                        "files": [
                            {
                                "path": data["pairs"][0]["latest_path"],
                                "sha256": data["pairs"][0]["latest_sha256"],
                                "size": data["pairs"][0]["size"],
                            }
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "backup receipt"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt)

    def test_apply_uses_temp_hardlink_and_replaces_latest_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            manifest = write_manifest(root)
            receipt = write_receipt(root, manifest)

            report = checkpoint_hardlinks.apply_dedup(root, manifest, receipt)

            self.assertEqual(report["status"], "applied")
            self.assertEqual(report["applied_count"], 1)
            self.assertEqual(report["recovered_bytes"], 4)
            self.assertEqual((chunk / "latest.zip").read_bytes(), b"same")
            self.assertEqual((chunk / "final.zip").read_bytes(), b"same")
            self.assertEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)
            self.assertFalse((chunk / ".latest.zip.hardlink-tmp").exists())
            self.assertIn("Restoration", report["restoration_recipe"])
            self.assertEqual(report["manifest_path"], str(manifest.resolve()))
            self.assertEqual(report["manifest_sha256"], checkpoint_hardlinks.sha256_file(manifest))
            self.assertEqual(report["backup_receipt_path"], str(receipt.resolve()))
            self.assertEqual(report["backup_receipt_sha256"], checkpoint_hardlinks.sha256_file(receipt))
            self.assertEqual(report["backup_receipt_schema_version"], 1)

    def test_apply_refuses_existing_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            (chunk / ".latest.zip.hardlink-tmp").write_bytes(b"old")
            manifest = write_manifest(root)
            receipt = write_receipt(root, manifest)

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "temporary"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt)

    def test_apply_rejects_completion_drift_before_mutating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            manifest = write_manifest(root)
            receipt = write_receipt(root, manifest)
            write_result(chunk, status="failed", error="late failure")

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "completion drift"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt)
            self.assertEqual((chunk / "latest.zip").read_bytes(), b"same")
            self.assertNotEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)

    def test_apply_rejects_result_path_from_different_completed_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            other = make_completed_chunk(root / "run" / "chunk02", latest=b"other", final=b"other")
            manifest = write_manifest(root)
            data = json.loads(manifest.read_text())
            data["pairs"] = [pair for pair in data["pairs"] if Path(pair["chunk_dir"]).resolve() == chunk.resolve()]
            data["pairs"][0]["result_path"] = str(other / "result.json")
            data["pairs"][0]["result_sha256"] = checkpoint_hardlinks.sha256_file(other / "result.json")
            manifest.write_text(json.dumps(data))
            receipt = write_receipt(root, manifest)

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "same chunk"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt)
            self.assertNotEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)

    def test_apply_rejects_completed_result_json_content_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            manifest = write_manifest(root)
            receipt = write_receipt(root, manifest)
            result = json.loads((chunk / "result.json").read_text())
            result["measured_cycles"] = 101
            (chunk / "result.json").write_text(json.dumps(result))

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "result.json drift"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt)
            self.assertNotEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)

    def test_apply_rejects_relative_manifest_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            manifest = write_manifest(root)
            data = json.loads(manifest.read_text())
            data["pairs"][0]["latest_path"] = os.path.relpath(chunk / "latest.zip", Path.cwd())
            manifest.write_text(json.dumps(data))
            receipt = write_receipt(root, manifest)

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "absolute"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt)
            self.assertNotEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)

    def test_apply_rejects_backup_receipt_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            manifest = write_manifest(root)
            receipt = write_receipt(root, manifest)
            receipt_data = json.loads(receipt.read_text())
            receipt_data["manifest_sha256"] = "0" * 64
            receipt.write_text(json.dumps(receipt_data))

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "manifest_sha256"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt)
            self.assertNotEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)

    def test_apply_rejects_malformed_backup_receipt_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            manifest = write_manifest(root)
            receipt = write_receipt(root, manifest)
            receipt_data = json.loads(receipt.read_text())
            receipt_data["files"][0]["sha256"] = "not-a-sha"
            receipt.write_text(json.dumps(receipt_data))

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "backup receipt"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt)
            self.assertNotEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)

    def test_apply_rejects_duplicate_backup_receipt_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            manifest = write_manifest(root)
            receipt = write_receipt(root, manifest)
            receipt_data = json.loads(receipt.read_text())
            receipt_data["files"].append(dict(receipt_data["files"][0]))
            receipt.write_text(json.dumps(receipt_data))

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "duplicate"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt)
            self.assertNotEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)

    def test_atomic_failure_leaves_latest_bytes_and_reports_restoration_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_completed_chunk(root / "run" / "chunk01", latest=b"same", final=b"same")
            manifest = write_manifest(root)
            receipt = write_receipt(root, manifest)

            def failing_replace(src, dst):
                raise OSError("simulated replace failure")

            with self.assertRaisesRegex(checkpoint_hardlinks.HardlinkError, "Restoration"):
                checkpoint_hardlinks.apply_dedup(root, manifest, receipt, replace=failing_replace)
            self.assertEqual((chunk / "latest.zip").read_bytes(), b"same")
            self.assertNotEqual(os.stat(chunk / "latest.zip").st_ino, os.stat(chunk / "final.zip").st_ino)
            self.assertFalse((chunk / ".latest.zip.hardlink-tmp").exists())


def make_completed_chunk(chunk: Path, *, latest: bytes, final: bytes) -> Path:
    chunk.mkdir(parents=True)
    (chunk / "latest.zip").write_bytes(latest)
    (chunk / "final.zip").write_bytes(final)
    write_result(chunk)
    return chunk


def write_result(chunk: Path, *, status="completed", error=None, stop_reason=None):
    (chunk / "result.json").write_text(
        json.dumps(
            {
                "status": status,
                "error": error,
                "stop_reason": stop_reason,
                "latest_checkpoint": str(chunk / "latest.zip"),
                "final_checkpoint": str(chunk / "final.zip"),
            }
        )
    )


def write_manifest(root: Path) -> Path:
    manifest = root / "manifest.json"
    report = checkpoint_hardlinks.plan_dedup(root)
    manifest.write_text(json.dumps(report))
    return manifest


def write_receipt(root: Path, manifest: Path) -> Path:
    data = json.loads(manifest.read_text())
    files = []
    for pair in data["pairs"]:
        files.extend(
            [
                {
                    "remote_path": pair["latest_path"],
                    "backup_path": str((root / "backup" / "latest.zip").resolve()),
                    "sha256": pair["latest_sha256"],
                    "size": pair["size"],
                },
                {
                    "remote_path": pair["final_path"],
                    "backup_path": str((root / "backup" / "final.zip").resolve()),
                    "sha256": pair["final_sha256"],
                    "size": pair["size"],
                },
            ]
        )
    receipt = root / "backup.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "verified",
                "independently_verified_local_backup": True,
                "manifest_sha256": checkpoint_hardlinks.sha256_file(manifest),
                "verified_unix": time.time(),
                "files": files,
            }
        )
    )
    return receipt


if __name__ == "__main__":
    unittest.main()
