"""Local fixtures for bounded, literal-link session11 preservation."""

from ai2.paths import source_path
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock


MODULE = source_path(Path(__file__).resolve().parents[1], "session11_capture_external_logs.py")


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), "session11 capture utility must exist")
        spec = importlib.util.spec_from_file_location("session11_capture_test", MODULE)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base / "session11"
        self.logs = self.base / "core-logs"
        self.config = self.root / "variants/arch-512-512-256_seed145/config.yaml"
        self.config.parent.mkdir(parents=True)
        self.config.write_bytes(b"phase: 0a\n")
        self.logs.mkdir()
        self.log = self.logs / "core-debug-20260907_112706.log"
        self.log.write_bytes(b"ordinary diagnostic log\n")
        self.entries = [
            {"type": "symlink", "path": "wandb/run/files/config.yaml", "target": str(self.config)},
            {"type": "symlink", "path": "wandb/run/logs/debug-core.log", "target": str(self.log)},
            {"type": "symlink", "path": "wandb/latest-run", "target": "run"},
        ]
        self.manifest = self.base / "inventory.json"
        self.output = self.root / "artifacts/provenance/job-001.tar.gz"
        self.prepare()

    def prepare(self):
        for entry in self.entries:
            link = self.root / entry["path"]
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                link.unlink()
            link.symlink_to(entry["target"])
        payload = {"schema": "ai2.session9.download-integrity.v1", "finalized": True,
                   "symlink_count": len(self.entries), "files": self.entries,
                   "source_roots": {"artifacts": str(self.root / "artifacts"), "wandb": str(self.root / "wandb")}}
        self.manifest.write_text(json.dumps(payload))
        self.digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def capture(self, **kwargs):
        return self.mod.capture(self.manifest, self.root, core_log_root=self.logs,
                                output_tar=self.output, expected_manifest_sha256=self.digest, **kwargs)

    def test_exact_regular_content_mapping_and_archive_hash(self):
        receipt = self.capture()
        self.assertEqual((receipt["literal_symlink_count"], receipt["absolute_symlink_count"], receipt["unique_target_count"]), (3, 2, 2))
        self.assertEqual(receipt["archive_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        with tarfile.open(self.output) as archive:
            mapping = json.load(archive.extractfile("mapping.json"))
            self.assertEqual(mapping["manifest_sha256"], self.digest)
            self.assertEqual(len(archive.getmembers()), 3)
            self.assertTrue(all(member.isfile() and not member.name.startswith("/") for member in archive))
            for link in mapping["links"]:
                payload = archive.extractfile(link["archive_member"]).read()
                self.assertEqual(payload, Path(link["original_target"]).read_bytes())
                self.assertEqual(hashlib.sha256(payload).hexdigest(), link["sha256"])

    def test_distinct_targets_with_identical_content_are_hash_deduplicated(self):
        self.log.write_bytes(self.config.read_bytes())
        receipt = self.capture()
        self.assertEqual(receipt["unique_target_count"], 2)
        self.assertEqual(receipt["unique_content_count"], 1)
        with tarfile.open(self.output) as archive:
            self.assertEqual(len(archive.getmembers()), 2)

    def test_duplicate_targets_keep_each_literal_link_mapping(self):
        self.entries.append({"type": "symlink", "path": "wandb/other/files/config.yaml", "target": str(self.config)})
        self.prepare()
        receipt = self.capture()
        self.assertEqual((len(receipt["links"]), receipt["unique_target_count"]), (3, 2))

    def test_output_is_never_overwritten(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_bytes(b"existing archive")
        with self.assertRaises(FileExistsError):
            self.capture()
        self.assertEqual(self.output.read_bytes(), b"existing archive")

    def test_same_snapshot_has_reproducible_archive_hash(self):
        first = self.capture()
        self.output = self.output.with_name("job-002.tar.gz")
        second = self.capture()
        self.assertEqual(first["archive_sha256"], second["archive_sha256"])

    def test_manifest_pin_and_finalization_required(self):
        self.manifest.write_bytes(self.manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.capture()
        data = json.loads(self.manifest.read_bytes())
        data["finalized"] = False
        self.manifest.write_text(json.dumps(data))
        self.digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "finalized"):
            self.capture()
        self.assertFalse(self.output.exists())

    def test_reject_targets_outside_nine_exact_configs_and_timestamped_logs(self):
        for target in (self.root / "variants/arch-512-512-256_seed148/config.yaml",
                       self.root / "variants/arch-512-512-256_seed145/settings",
                       self.logs / "settings", self.base / ".netrc", self.logs / "../token"):
            with self.subTest(target=target):
                self.entries[1]["target"] = str(target)
                self.prepare()
                with self.assertRaises(ValueError):
                    self.capture()
        self.assertFalse(self.output.exists())

    def test_missing_and_symlink_target_rejected(self):
        self.log.unlink()
        with self.assertRaises((OSError, ValueError)):
            self.capture()
        self.log.symlink_to(self.config)
        with self.assertRaises((OSError, ValueError)):
            self.capture()
        self.assertFalse(self.output.exists())

    def test_link_must_still_match_manifest(self):
        link = self.root / self.entries[0]["path"]
        link.unlink()
        link.symlink_to(self.log)
        with self.assertRaisesRegex(ValueError, "link"):
            self.capture()

    def test_symlinked_target_ancestor_rejected(self):
        moved = self.base / "moved-logs"
        self.logs.rename(moved)
        self.logs.symlink_to(moved, target_is_directory=True)
        with self.assertRaises((OSError, ValueError)):
            self.capture()

    def test_symlinked_output_parent_rejected(self):
        destination = self.base / "unrelated"
        destination.mkdir()
        (self.root / "artifacts").symlink_to(destination, target_is_directory=True)
        with self.assertRaises((OSError, ValueError)):
            self.capture()
        self.assertEqual(list(destination.iterdir()), [])

    def test_manifest_paths_and_counts_are_validated(self):
        original = json.loads(self.manifest.read_bytes())
        for invalid in ("../escape", "/absolute", "other-root/file", self.entries[0]["path"]):
            with self.subTest(path=invalid):
                data = json.loads(json.dumps(original))
                data["files"][1]["path"] = invalid
                self.manifest.write_text(json.dumps(data))
                self.digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    self.capture()
        original["symlink_count"] = 99
        self.manifest.write_text(json.dumps(original))
        self.digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "count"):
            self.capture()

    def test_secret_like_bytes_rejected_without_content_echo(self):
        secret = "test-only-long-secret-123456789"
        self.log.write_text("Authorization: Bearer " + secret)
        with self.assertRaises(ValueError) as error:
            self.capture()
        self.assertNotIn(secret, str(error.exception))
        self.assertFalse(self.output.exists())

    def test_strict_bounds(self):
        for limits in ({"max_files": 1}, {"max_files": 65}, {"max_bytes": 1}, {"max_bytes": 100_000_001}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.capture(**limits)

    def test_link_changed_after_target_read_rejected_before_publish(self):
        original = self.mod._read_regular
        changed = False
        def changing(path, limit):
            nonlocal changed
            result = original(path, limit)
            if Path(path) == self.config and not changed:
                changed = True
                link = self.root / self.entries[0]["path"]
                link.unlink()
                link.symlink_to(self.log)
            return result
        with mock.patch.object(self.mod, "_read_regular", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "link"):
                self.capture()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.output.parent.glob("*.tmp-*")))

    def test_target_and_manifest_mutation_rejected_before_publish(self):
        for mutate_manifest in (False, True):
            with self.subTest(manifest=mutate_manifest):
                self.prepare()
                original = self.mod._read_regular
                changed = False
                def changing(path, limit):
                    nonlocal changed
                    result = original(path, limit)
                    if Path(path) == self.config and not changed:
                        changed = True
                        target = self.manifest if mutate_manifest else self.config
                        target.write_bytes(target.read_bytes() + b" ")
                    return result
                with mock.patch.object(self.mod, "_read_regular", side_effect=changing):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        self.capture()
                self.assertFalse(self.output.exists())
                self.assertFalse(list(self.output.parent.glob("*.tmp-*")))

    def test_cli_has_fixed_roots_and_requires_unique_output(self):
        args = self.mod.parse_args(["--manifest", str(self.manifest), "--manifest-sha256", self.digest, "--output", str(self.output)])
        self.assertEqual(args.output, self.output)
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            self.mod.parse_args(["--session-root", str(self.root)])


if __name__ == "__main__":
    unittest.main()
