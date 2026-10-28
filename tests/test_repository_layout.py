"""Regression contracts for moving source without changing frozen experiments."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def layout(self):
        self.assertTrue((ROOT / "ai2/paths.py").is_file(), "canonical source layout is missing")
        return importlib.import_module("ai2.paths")

    def test_frozen_roots_resolve_only_their_own_source(self):
        layout = self.layout()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "curriculum_ppo.py"
            frozen.write_text("frozen = True\n")
            self.assertEqual(layout.source_path(root, frozen.name), frozen)
            self.assertEqual(layout.source_path(root, "tensor_ppo.py"), root / "tensor_ppo.py")

    def test_current_layout_hashes_implementation_instead_of_checkpoint_shim(self):
        layout = self.layout()
        self.assertEqual(layout.source_path(ROOT, "curriculum_ppo.py"), ROOT / "ai2/training/curriculum_ppo.py")
        self.assertEqual(layout.source_path(ROOT, "train.py"), ROOT / "ai2/training/train.py")

    def test_missing_current_source_never_falls_back_to_old_file(self):
        layout = self.layout()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ai2").mkdir()
            (root / "ai2/paths.py").write_text("# new layout\n")
            (root / "tensor_ppo.py").write_text("stale = True\n")
            mapped = layout.source_path(root, "tensor_ppo.py")
            self.assertEqual(mapped, root / "ai2/training/tensor_ppo.py")
            self.assertFalse(mapped.exists())

    def test_source_inventory_includes_implementation_and_compatibility(self):
        inventory = self.layout().source_inventory(ROOT)
        for name in ("ai2/paths.py", "ai2/__init__.py", "ai2/training/curriculum_ppo.py",
                     "ai2/training/tensor_ppo.py", "train.py", "curriculum_ppo.py"):
            self.assertIn(name, inventory)
            self.assertTrue(inventory[name].is_file())
        self.assertEqual(len(inventory), len(set(inventory.values())))

    def test_inventory_refuses_missing_required_source(self):
        layout = self.layout()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ai2").mkdir()
            (root / "ai2/paths.py").write_text("# incomplete checkout\n")
            with self.assertRaises(FileNotFoundError):
                layout.source_inventory(root)

    def test_inventory_requires_nested_package_initializers(self):
        layout = self.layout()
        real_is_file = Path.is_file
        missing = ROOT / "ai2/training/__init__.py"
        with mock.patch.object(Path, "is_file", lambda path: False if path == missing else real_is_file(path)):
            with self.assertRaisesRegex(FileNotFoundError, "ai2/training/__init__.py"):
                layout.source_inventory(ROOT)

    def test_source_snapshots_copy_selected_implementation_with_logical_name(self):
        from ai2.runtime.benchmark_campaign import copy_source_snapshot
        for packaged in (False, True):
            with self.subTest(packaged=packaged), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "selected"
                root.mkdir()
                if packaged:
                    (root / "ai2/training").mkdir(parents=True)
                    (root / "ai2/paths.py").write_text("# selected layout\n")
                selected = self.layout().source_path(root, "train.py")
                selected.write_text("# selected implementation\n")
                output = Path(directory) / "output"
                output.mkdir()
                copy_source_snapshot(root, output)
                self.assertEqual((output / "source/train.py").read_bytes(), selected.read_bytes())

    def test_every_original_file_has_one_disposition(self):
        manifest = ROOT / "docs/python-file-map.json"
        self.assertTrue(manifest.is_file(), "the complete migration map is missing")
        rows = json.loads(manifest.read_text())
        self.assertEqual(len(rows), 146)
        self.assertEqual(len({row["original"] for row in rows}), 146)
        destinations = [row["destination"] for row in rows if row["destination"]]
        self.assertEqual(len(destinations), len(set(destinations)))
        for row in rows:
            if row["destination"]:
                self.assertTrue((ROOT / row["destination"]).is_file(), row["original"])
            elif row["action"] == "retire":
                self.assertFalse((ROOT / row["original"]).exists(), row["original"])

    def test_only_documented_compatibility_modules_remain_at_root(self):
        self.assertEqual({path.name for path in ROOT.glob("*.py")}, {"train.py", "curriculum_ppo.py"})

    def test_package_cli_can_show_help_without_running_work(self):
        self.layout()
        result = subprocess.run(
            [sys.executable, "-m", "tools.analysis.summarize_campaign", "--help"],
            cwd=ROOT, text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
