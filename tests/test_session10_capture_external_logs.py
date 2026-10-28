"""CPU-only safety checks for the one-shot session10 external log capture."""

from ai2.paths import source_path

import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock


MODULE = source_path(Path(__file__).resolve().parents[1], "session10_capture_external_logs.py")


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), "the bounded capture helper must exist")
        spec = importlib.util.spec_from_file_location("capture_external", MODULE)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base / "session"
        self.logs = self.base / "core-logs"
        self.config = self.root / "variants/corrected/control.yaml"
        self.config.parent.mkdir(parents=True)
        self.config.write_bytes(b"training: phase0b\n")
        self.logs.mkdir()
        self.log = self.logs / "core-debug-20260907_112706.log"
        self.log.write_bytes(b"ordinary diagnostic log\n")
        self.entries = [
            {"type": "symlink", "path": "wandb/run/files/control.yaml", "target": str(self.config)},
            {"type": "symlink", "path": "wandb/run/logs/debug-core.log", "target": str(self.log)},
            {"type": "symlink", "path": "wandb/latest-run", "target": "run"},
        ]
        self.manifest = self.base / "manifest.json"

    def capture(self, **kwargs):
        payload = {"finalized": True, "files": self.entries, "source_roots": {
            "artifacts": str(self.root / "artifacts"), "wandb": str(self.root / "wandb")}}
        self.manifest.write_text(json.dumps(payload))
        digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        return self.mod.capture(self.manifest, self.root, core_log_root=self.logs,
                                expected_manifest_sha256=digest, expected_counts=(3, 2, 2), **kwargs)

    def test_regular_members_mapping_hashes_and_no_overwrite(self):
        receipt = self.capture()
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["unique_target_count"], 2)
        output = self.root / "artifacts/provenance"
        self.assertEqual(json.loads((output / "external-link-targets.json").read_text()), receipt)
        with tarfile.open(output / "external-link-targets.tar.gz") as archive:
            self.assertEqual(len(archive.getmembers()), 2)
            for link in receipt["links"]:
                member = archive.getmember(link["archive_member"])
                self.assertTrue(member.isfile())
                self.assertFalse(member.name.startswith("/"))
                data = archive.extractfile(member).read()
                self.assertEqual(hashlib.sha256(data).hexdigest(), link["sha256"])
                self.assertEqual(len(data), link["size"])
        with self.assertRaises(FileExistsError):
            self.capture()

    def test_duplicate_targets_are_captured_once(self):
        self.entries[2] = {"type": "symlink", "path": "wandb/other/files/control.yaml", "target": str(self.config)}
        with mock.patch.object(self.mod, "EXPECTED_COUNTS", (3, 3, 2)):
            self.manifest.write_text(json.dumps({"finalized": True, "files": self.entries}))
            receipt = self.mod.capture(self.manifest, self.root, core_log_root=self.logs,
                expected_manifest_sha256=hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                expected_counts=(3, 3, 2))
        self.assertEqual(len(receipt["links"]), 3)
        self.assertEqual(receipt["unique_target_count"], 2)

    def test_reject_missing_target(self):
        self.log.unlink()
        with self.assertRaises((ValueError, OSError)):
            self.capture()
        self.assertFalse((self.root / "artifacts/provenance/external-link-targets.tar.gz").exists())

    def test_reject_target_symlink_and_directory(self):
        self.log.unlink()
        self.log.symlink_to(self.config)
        with self.assertRaises((ValueError, OSError)):
            self.capture()
        self.log.unlink()
        self.log.mkdir()
        with self.assertRaises((ValueError, OSError)):
            self.capture()

    def test_reject_symlinked_ancestor(self):
        moved = self.base / "moved"
        self.logs.rename(moved)
        self.logs.symlink_to(moved, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            self.capture()

    def test_reject_unallowlisted_credentials_and_traversal(self):
        for target in (str(self.base / ".netrc"), str(self.logs / "../.netrc"),
                       str(self.logs / "settings"), str(self.logs / "core-debug-20260907_112706.log/extra")):
            with self.subTest(target=target):
                self.entries[1]["target"] = target
                with self.assertRaises(ValueError):
                    self.capture()

    def test_reject_link_path_traversal_and_duplicate_path(self):
        for path in ("../escape", "/absolute", self.entries[0]["path"]):
            with self.subTest(path=path):
                self.entries[1]["path"] = path
                with self.assertRaises(ValueError):
                    self.capture()

    def test_reject_limits_and_count_mismatch(self):
        with self.assertRaises(ValueError):
            self.capture(max_bytes=1)
        with self.assertRaises(ValueError):
            self.capture(max_files=1)
        self.entries.pop()
        with self.assertRaises(ValueError):
            self.capture()

    def test_reject_manifest_digest_mismatch(self):
        self.manifest.write_text('{"files": []}')
        with self.assertRaises(ValueError):
            self.mod.capture(self.manifest, self.root, expected_manifest_sha256="0" * 64)

    def test_reject_changes_after_first_read(self):
        original = self.mod._read_regular
        changed = False

        def changing(path, limit):
            nonlocal changed
            result = original(path, limit)
            if Path(path) == self.config and not changed:
                changed = True
                self.config.write_bytes(b"changed\n")
            return result

        with mock.patch.object(self.mod, "_read_regular", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "changed"):
                self.capture()
        self.assertFalse((self.root / "artifacts/provenance/external-link-targets.tar.gz").exists())

    def test_reject_obvious_embedded_credentials_without_echoing(self):
        secret = "this-is-a-private-token-123456"
        self.log.write_text("Authorization: Bearer " + secret)
        with self.assertRaises(ValueError) as error:
            self.capture()
        self.assertNotIn(secret, str(error.exception))

    def test_reject_symlinked_output_directory(self):
        (self.root / "artifacts").symlink_to(self.logs, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            self.capture()


if __name__ == "__main__":
    unittest.main()
