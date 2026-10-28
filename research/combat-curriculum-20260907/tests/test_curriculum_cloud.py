"""Offline preservation and provider-boundary tests; never contact Vast.ai."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import types
import unittest
from unittest.mock import patch

try:
    import curriculum_cloud as cloud
except ModuleNotFoundError:
    cloud = None


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(cloud, "curriculum_cloud preservation API is missing")
        self.temp = tempfile.TemporaryDirectory(
            prefix="curriculum-test-", dir="/private/tmp"
        )
        self.addCleanup(self.temp.cleanup)
        self.block = Path(self.temp.name) / "E1-round-01"
        self.block.mkdir()
        (self.block / "checkpoint.bin").write_bytes(b"checkpoint")
        (self.block / "logs").mkdir()
        (self.block / "logs" / "metrics.json").write_text('{"loss":1.0}\n')
        self.inventory = {
            "schema_version": "curriculum-inventory-v1",
            "block_id": "E1-round-01",
            "files": [self.entry("checkpoint.bin"), self.entry("logs/metrics.json")],
        }
        self.write_inventory()

    def entry(self, name):
        data = (self.block / name).read_bytes()
        return {
            "path": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def write_inventory(self):
        (self.block / "inventory.json").write_text(json.dumps(self.inventory))

    def test_verify_is_read_only_and_ack_binds_exact_inventory_bytes(self):
        receipt = cloud.verify_inventory(self.block)
        digest = hashlib.sha256(
            (self.block / "inventory.json").read_bytes()
        ).hexdigest()
        self.assertEqual(
            receipt["ack"],
            {"block_id": "E1-round-01", "inventory_sha256": digest, "verified": True},
        )
        self.assertEqual(receipt["files_verified"], 2)
        self.assertEqual(
            receipt["bytes_verified"], sum(e["bytes"] for e in self.inventory["files"])
        )
        self.assertFalse((self.block / "preservation_ack.json").exists())

    def test_missing_optional_block_id_uses_directory_name(self):
        self.inventory.pop("block_id")
        self.write_inventory()
        self.assertEqual(
            cloud.verify_inventory(self.block)["block_id"], self.block.name
        )

    def test_rejects_unsafe_paths(self):
        for name in [
            "../outside",
            "/absolute",
            "logs/../checkpoint.bin",
            "./checkpoint.bin",
            "logs//metrics.json",
            "logs\\metrics.json",
            "C:/file",
            "bad\x00name",
            "file.tmp",
            ".tmp/file",
            "checkpoint.tmp.part",
        ]:
            with self.subTest(path=name):
                self.inventory["files"][0]["path"] = name
                self.write_inventory()
                with self.assertRaises(cloud.SafetyError):
                    cloud.verify_inventory(self.block)

    def test_rejects_duplicates_and_reserved_outputs(self):
        self.inventory["files"].append(copy.deepcopy(self.inventory["files"][0]))
        self.write_inventory()
        with self.assertRaises(cloud.SafetyError):
            cloud.verify_inventory(self.block)
        self.inventory["files"].pop()
        for name in [
            "inventory.json",
            "preservation_receipt.json",
            "preservation_ack.json",
        ]:
            self.inventory["files"][0]["path"] = name
            self.write_inventory()
            with self.assertRaises(cloud.SafetyError):
                cloud.verify_inventory(self.block)

    def test_rejects_missing_corrupt_and_wrong_size_files(self):
        target = self.block / "checkpoint.bin"
        for data in [b"corruption", b"short"]:
            target.write_bytes(data)
            with self.assertRaises(cloud.SafetyError):
                cloud.verify_inventory(self.block)
        target.unlink()
        with self.assertRaises(cloud.SafetyError):
            cloud.verify_inventory(self.block)

    def test_rejects_malformed_inventory(self):
        for change in [
            {"schema_version": "other"},
            {"files": []},
            {"files": {}},
            {"block_id": "../secret"},
        ]:
            original = copy.deepcopy(self.inventory)
            self.inventory.update(change)
            self.write_inventory()
            with self.assertRaises(cloud.SafetyError):
                cloud.verify_inventory(self.block)
            self.inventory = original
        for key, value in [
            ("bytes", True),
            ("bytes", -1),
            ("bytes", 1.5),
            ("sha256", "not-a-hash"),
        ]:
            original = copy.deepcopy(self.inventory)
            self.inventory["files"][0][key] = value
            self.write_inventory()
            with self.assertRaises(cloud.SafetyError):
                cloud.verify_inventory(self.block)
            self.inventory = original

    def test_rejects_unlisted_files_and_temporary_directories(self):
        extra = self.block / "unlisted.log"
        extra.write_text("not in inventory")
        with self.assertRaises(cloud.SafetyError):
            cloud.verify_inventory(self.block)
        extra.unlink()
        (self.block / "unfinished.tmp").mkdir()
        with self.assertRaises(cloud.SafetyError):
            cloud.verify_inventory(self.block)

    def test_rejects_file_directory_inventory_and_root_symlinks(self):
        target = self.block / "checkpoint.bin"
        target.unlink()
        target.symlink_to("logs/metrics.json")
        with self.assertRaises(cloud.SafetyError):
            cloud.verify_inventory(self.block)
        target.unlink()
        target.write_bytes(b"checkpoint")
        link = self.block / "directory-link"
        link.symlink_to(self.block / "logs", target_is_directory=True)
        with self.assertRaises(cloud.SafetyError):
            cloud.verify_inventory(self.block)
        link.unlink()
        inventory = self.block / "inventory.json"
        saved = Path(self.temp.name) / "inventory-copy.json"
        saved.write_bytes(inventory.read_bytes())
        inventory.unlink()
        inventory.symlink_to(saved)
        with self.assertRaises(cloud.SafetyError):
            cloud.verify_inventory(self.block)
        root_link = Path(self.temp.name) / "block-link"
        root_link.symlink_to(self.block, target_is_directory=True)
        with self.assertRaises(cloud.SafetyError):
            cloud.verify_inventory(root_link)

    def test_preserve_is_idempotent_and_never_overwrites(self):
        receipt = cloud.preserve_block(self.block)
        for name, expected in [
            ("preservation_receipt.json", receipt),
            ("preservation_ack.json", receipt["ack"]),
        ]:
            self.assertEqual(json.loads((self.block / name).read_text()), expected)
        self.assertEqual(cloud.preserve_block(self.block), receipt)
        (self.block / "preservation_ack.json").write_text('{"verified":false}')
        with self.assertRaises(cloud.SafetyError):
            cloud.preserve_block(self.block)
        self.assertEqual(
            (self.block / "preservation_ack.json").read_text(), '{"verified":false}'
        )
        self.assertEqual(list(self.block.rglob("*.tmp")), [])

    def test_failed_verification_creates_no_ack(self):
        (self.block / "checkpoint.bin").write_bytes(b"bad")
        with self.assertRaises(cloud.SafetyError):
            cloud.preserve_block(self.block)
        self.assertFalse((self.block / "preservation_ack.json").exists())
        self.assertFalse((self.block / "preservation_receipt.json").exists())

    def test_verify_local_cli_prints_only_ack(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cloud.main(["verify-local", str(self.block)])
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(output.getvalue()),
            json.loads((self.block / "preservation_ack.json").read_text()),
        )

    def test_standalone_cli_verifies_then_fails_closed_on_modified_payload(self):
        command = [
            sys.executable,
            "-B",
            str(Path(cloud.__file__)),
            "verify-local",
            str(self.block),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["block_id"], self.inventory["block_id"]
        )
        self.assertEqual(result.stderr, "")
        (self.block / "checkpoint.bin").write_bytes(b"changed")
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)

    def test_ack_matches_actual_runner_preservation_contract(self):
        from train_combat_curriculum import require_preserved, write_inventory

        (self.block / "inventory.json").unlink()
        write_inventory(self.block, block_id="E1-round1")
        cloud.preserve_block(self.block)
        require_preserved(self.block, self.block / "preservation_ack.json")


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(cloud, "curriculum_cloud audit API is missing")
        self.instances = [
            {
                "id": 50149729,
                "label": "ai2-native-ppo-session10-20260907",
                "actual_status": "exited",
                "cur_state": "stopped",
                "dph_total": 0.30,
                "api_key": "SECRET",
            }
        ]
        self.user = {
            "credit": 4.315037,
            "api_key": "SECRET",
            "email": "private@example.org",
        }

    def fake_runner(self, argv, **kwargs):
        self.assertEqual(kwargs["capture_output"], True)
        self.assertEqual(kwargs["text"], True)
        self.assertEqual(kwargs["check"], False)
        self.assertGreater(kwargs["timeout"], 0)
        self.assertNotIn("shell", kwargs)
        instance_command = ["vastai", "show", "instances", "--raw"]
        credit_command = [sys.executable, "-c", cloud.USER_CREDIT_SCRIPT]
        self.assertIn(argv, [instance_command, credit_command])
        data = self.instances if argv == instance_command else self.user
        return subprocess.CompletedProcess(
            argv, 0, json.dumps(data), "api_key=SECRET stderr"
        )

    def audit(self):
        return cloud.audit_provider(now=1000.0, runner=self.fake_runner)

    def test_read_only_audit_captures_and_allowlists_provider_data(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            audit = self.audit()
        self.assertEqual(out.getvalue() + err.getvalue(), "")
        self.assertEqual(audit["instance_id"], 50149729)
        self.assertEqual(audit["status"], "stopped")
        self.assertEqual(audit["running_instances"], 0)
        self.assertAlmostEqual(audit["cumulative_spent"], 8.0)
        self.assertNotIn("SECRET", json.dumps(audit))
        self.assertNotIn("private@", json.dumps(audit))

    def test_running_selected_instance_is_allowed_but_overlap_is_not(self):
        self.instances[0].update(actual_status="running", cur_state="running")
        self.assertEqual(self.audit()["running_instances"], 1)
        self.instances.append(
            {"id": 7, "label": "other", "actual_status": "running", "dph_total": 0.1}
        )
        with self.assertRaises(cloud.SafetyError):
            self.audit()

    def test_pending_instance_start_does_not_count_as_safely_stopped(self):
        self.instances.append(
            {
                "id": 7,
                "actual_status": "exited",
                "cur_state": "stopped",
                "intended_status": "running",
            }
        )
        with self.assertRaises(cloud.SafetyError):
            self.audit()

    def test_rejects_identity_status_price_credit_and_duplicate_errors(self):
        original = copy.deepcopy(self.instances)
        for change in [
            {"id": 7},
            {"id": True},
            {"label": "other"},
            {"actual_status": "running", "cur_state": "stopped"},
            {"actual_status": "unknown", "cur_state": "stopped"},
            {"dph_total": 0.40001},
            {"dph_total": float("nan")},
            {"dph_total": -1},
        ]:
            with self.subTest(change=change):
                self.instances = copy.deepcopy(original)
                self.instances[0].update(change)
                with self.assertRaises(cloud.SafetyError):
                    self.audit()
        self.instances = original + original
        with self.assertRaises(cloud.SafetyError):
            self.audit()
        self.instances = original
        for credit in [float("nan"), -1, 12.315038, True]:
            self.user["credit"] = credit
            with self.assertRaises(cloud.SafetyError):
                self.audit()

    def test_provider_failures_never_expose_output_exception_or_command(self):
        for result in [
            subprocess.CompletedProcess(
                ["SECRET-command"], 1, "SECRET-stdout", "api_key=SECRET"
            ),
            subprocess.CompletedProcess([], 0, "SECRET-malformed-json", "SECRET"),
        ]:
            with self.subTest(result=result.returncode):
                output = io.StringIO()
                with (
                    contextlib.redirect_stdout(output),
                    contextlib.redirect_stderr(output),
                ):
                    try:
                        cloud.audit_provider(now=1000.0, runner=lambda *a, **kw: result)
                    except cloud.SafetyError:
                        traceback.print_exc()
                    else:
                        self.fail("provider failure accepted")
                self.assertNotIn("SECRET", output.getvalue())

        def failed(*args, **kwargs):
            raise subprocess.TimeoutExpired(
                "SECRET-command", 20, output="SECRET", stderr="api_key=SECRET"
            )

        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            try:
                cloud.audit_provider(now=1000.0, runner=failed)
            except cloud.SafetyError:
                traceback.print_exc()
            else:
                self.fail("timeout accepted")
        self.assertNotIn("SECRET", output.getvalue())

    def test_budget_includes_remaining_guard_window_and_retrieval_reserve(self):
        result = cloud.validate_budget(
            self.audit(),
            now=1010.0,
            deadline_unix=4610.0,
            job_seconds=300.0,
            reserve_seconds=300.0,
        )
        self.assertAlmostEqual(result["cumulative_spent"], 8.0)
        # Include billing between the credit observation and the launch check.
        self.assertAlmostEqual(result["planned_cost_ceiling"], 0.30 * 3610 / 3600)
        self.assertAlmostEqual(result["remaining_campaign_credit"], 4.0)
        self.assertEqual(result["deadline_unix"], 4610.0)

    def test_budget_rejects_stale_future_deadline_nonfinite_and_insufficient_time(self):
        audit = self.audit()
        for changes in [
            {"now": 1601.0},
            {"now": 999.0},
            {"deadline_unix": float("inf")},
            {"deadline_unix": 1200.0},
            {"job_seconds": -1},
            {"reserve_seconds": 0},
            {"now": float("nan")},
        ]:
            kwargs = {
                "now": 1010.0,
                "deadline_unix": 4610.0,
                "job_seconds": 300.0,
                "reserve_seconds": 300.0,
            }
            kwargs.update(changes)
            with self.subTest(changes=changes), self.assertRaises(cloud.SafetyError):
                cloud.validate_budget(audit, **kwargs)

    def test_budget_rejects_insufficient_credit_and_tampered_audit(self):
        self.user["credit"] = 0.32
        with self.assertRaises(cloud.SafetyError):
            cloud.validate_budget(
                self.audit(), now=1010.0, deadline_unix=4610.0, job_seconds=300.0
            )
        self.user["credit"] = 4.315037
        for change in [
            {"instance_id": 1},
            {"label": "wrong"},
            {"hourly_price": 0.5},
            {"running_instances": 2},
            {"cumulative_spent": 0},
            {"sanitized": False},
        ]:
            audit = self.audit()
            audit.update(change)
            with self.subTest(change=change), self.assertRaises(cloud.SafetyError):
                cloud.validate_budget(
                    audit, now=1010.0, deadline_unix=4610.0, job_seconds=300.0
                )

    def test_audit_cli_errors_are_sanitized(self):
        output = io.StringIO()
        with (
            patch.object(
                cloud,
                "audit_provider",
                side_effect=cloud.SafetyError("provider request failed"),
            ),
            contextlib.redirect_stderr(output),
        ):
            code = cloud.main(["audit"])
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", output.getvalue())

    def test_malformed_provider_status_and_huge_credit_fail_closed(self):
        for status in [{"secret": "SECRET"}, ["running"], None]:
            self.instances[0]["actual_status"] = status
            with self.subTest(status=status), self.assertRaises(cloud.SafetyError):
                self.audit()
        self.instances[0]["actual_status"] = "exited"
        self.user["credit"] = 10**1000
        with self.assertRaises(cloud.SafetyError):
            self.audit()

    def test_audit_age_cost_cannot_be_ignored_near_budget_limit(self):
        self.user["credit"] = 0.45  # Only $0.134963 remains under campaign ceiling.
        with self.assertRaises(cloud.SafetyError):
            cloud.validate_budget(
                self.audit(), now=1500.0, deadline_unix=2700.0, job_seconds=300.0
            )

    def test_unexpected_cli_exception_is_not_printed(self):
        output = io.StringIO()
        with (
            patch.object(
                cloud, "audit_provider", side_effect=RuntimeError("api_key=SECRET")
            ),
            contextlib.redirect_stderr(output),
        ):
            code = cloud.main(["audit"])
        self.assertEqual(code, 2)
        self.assertNotIn("SECRET", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())

    def test_installed_user_read_omits_legacy_query_and_uses_auth_header(self):
        self.assertTrue(
            hasattr(cloud, "USER_CREDIT_SCRIPT"),
            "compatible installed-CLI credit read is missing",
        )
        fake = types.ModuleType("vast")
        request = {}

        def get(args, url, headers=None, json=None):
            request.update(url=url, headers=headers)
            return {"credit": 4.315037}

        fake.http_get = get
        fake.apiheaders = lambda args: {"Authorization": "Bearer " + args.api_key}

        def main():
            self.assertEqual(sys.argv, ["vastai", "show", "user", "--raw"])
            args = types.SimpleNamespace(
                url="https://console.vast.ai", api_key="SECRET"
            )
            url = fake.apiurl(args, "/users/current", {"owner": "me"})
            print(json.dumps(fake.http_get(args, url)))

        fake.main = main
        output = io.StringIO()
        with (
            patch.dict(sys.modules, {"vast": fake}),
            patch.object(sys, "argv", []),
            contextlib.redirect_stdout(output),
        ):
            exec(cloud.USER_CREDIT_SCRIPT, {})
        self.assertEqual(request["url"], "https://console.vast.ai/api/v0/users/current")
        self.assertEqual(request["headers"], {"Authorization": "Bearer SECRET"})
        self.assertNotIn("SECRET", output.getvalue())
        self.assertEqual(json.loads(output.getvalue()), {"credit": 4.315037})


if __name__ == "__main__":
    unittest.main()
