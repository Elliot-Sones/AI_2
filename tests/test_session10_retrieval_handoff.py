import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "session10_retrieval_handoff.py"


def module():
    assert SOURCE.exists(), "retrieval-only handoff watchdog is missing"
    spec = importlib.util.spec_from_file_location("handoff_test_module", SOURCE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixtures():
    original = dict(id=50149729, label="ai2-native-ppo-session10-20260907",
                    created_by_campaign=True, created_unix=1000.0,
                    hourly_price=0.315, ssh_host="ssh.example.test", ssh_port=12345)
    raw = json.dumps(original).encode()
    row = dict(id=50149729, label=original["label"], actual_status="running",
               cur_state="running", intended_status="running", num_gpus=1,
               gpu_name="RTX 4090", dph_total=0.315)
    audit = dict(handoff_audit=True, audit_unix=10000.0,
                 original_receipt_sha256=hashlib.sha256(raw).hexdigest(),
                 cumulative_spent=5.0, planned_batch_ceiling=0.50,
                 user_credit_ceiling=12,
                 credit_audit=dict(ok=True, fetched_at_utc="1970-01-01T02:46:40+00:00",
                                   balance_fields=dict(credit=7.315037)),
                 live_account_audit=dict(sanitized=True, instances=[row]),
                 manual_retrieval=dict(pid=123, destination="/private/tmp/manual-copy"))
    return raw, audit, row


class Clock:
    def __init__(self):
        self.value = 10000.0

    def now(self):
        return self.value

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class Provider:
    def __init__(self, row, ready=True, fail_after=None):
        self.row = row
        self.ready = ready
        self.stops = []
        self.calls = 0
        self.fail_after = fail_after

    def list_instances(self):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after and not self.stops:
            raise RuntimeError("live check failed")
        return [dict(self.row, actual_status="exited", cur_state="stopped",
                     intended_status="stopped") if self.stops else self.row]

    def check_readiness(self, *args):
        return self.ready

    def stop_instance(self, instance):
        self.stops.append(instance)

    def start_instance(self, *args):
        raise AssertionError("MUST NEVER START")

    def retrieve(self, *args):
        raise AssertionError("MUST NEVER RETRIEVE")


class HandoffTests(unittest.TestCase):
    def test_preserves_original_creation_and_validates_fresh_audit(self):
        m = module()
        raw, audit, row = fixtures()
        record = m.validate_audit(audit, raw, now=10000, max_seconds=1800)
        self.assertEqual(record["created_unix"], 1000)

    def test_bad_audits_and_longer_deadline_rejected(self):
        m = module()
        raw, audit, row = fixtures()
        mutations = [dict(audit_unix=9000), dict(cumulative_spent=12),
                     dict(original_receipt_sha256="0" * 64), dict(handoff_audit=False),
                     dict(user_credit_ceiling=13), dict(planned_batch_ceiling=0.1),
                     dict(cumulative_spent=float("nan"))]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                m.validate_audit(dict(audit, **mutation), raw, now=10000, max_seconds=1800)
        with self.assertRaises(ValueError):
            m.validate_audit(audit, raw, now=10000, max_seconds=1801)

    def test_running_identity_and_overlap_required(self):
        m = module()
        raw, audit, row = fixtures()
        for rows in ([dict(row, actual_status="exited", cur_state="stopped")],
                     [dict(row, label="wrong")], [dict(row, dph_total=0.50)],
                     [row, dict(id=12, actual_status="loading")]):
            invalid = copy.deepcopy(audit)
            invalid["live_account_audit"]["instances"] = rows
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                m.validate_audit(invalid, raw, now=10000, max_seconds=1800)

    def test_stale_credit_and_underreported_spend_rejected(self):
        m = module()
        raw, audit, row = fixtures()
        audit["credit_audit"]["fetched_at_utc"] = "1970-01-01T00:00:00+00:00"
        with self.assertRaises(ValueError):
            m.validate_audit(audit, raw, now=10000, max_seconds=1800)
        raw, audit, row = fixtures()
        audit["cumulative_spent"] = 1
        with self.assertRaises(ValueError):
            m.validate_audit(audit, raw, now=10000, max_seconds=1800)

    def test_deadline_arms_then_stops_without_start_or_retrieve(self):
        m = module()
        raw, audit, row = fixtures()
        provider = Provider(row)
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "new"
            result = m.run_watchdog(audit, raw, state, max_seconds=30, poll_seconds=5,
                                    provider=provider, clock=Clock(), install_signals=False)
            self.assertEqual(provider.stops, [50149729])
            self.assertEqual(result["status"], "stopped")
            armed = json.loads((state / "armed.json").read_text())
            self.assertTrue(armed["ready"])
            self.assertEqual(armed["original_created_unix"], 1000)
            self.assertEqual(armed["deadline_unix"], 10030)
            self.assertEqual(armed["manual_retrieval_status"], "externally_managed_not_verified")

    def test_failed_readiness_never_arms_and_stops(self):
        m = module()
        raw, audit, row = fixtures()
        provider = Provider(row, ready=False)
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "new"
            with self.assertRaises(RuntimeError):
                m.run_watchdog(audit, raw, state, provider=provider, clock=Clock(), install_signals=False)
            self.assertFalse((state / "armed.json").exists())
            self.assertEqual(provider.stops, [50149729])

    def test_live_failure_after_arm_stops(self):
        m = module()
        raw, audit, row = fixtures()
        provider = Provider(row, fail_after=1)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                m.run_watchdog(audit, raw, Path(tmp) / "new", provider=provider,
                               clock=Clock(), install_signals=False)
            self.assertEqual(provider.stops, [50149729])

    def test_existing_state_refused_without_provider_action(self):
        m = module()
        raw, audit, row = fixtures()
        provider = Provider(row)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                m.run_watchdog(audit, raw, Path(tmp), provider=provider,
                               clock=Clock(), install_signals=False)
            self.assertEqual(provider.calls, 0)
            self.assertEqual(provider.stops, [])

    def test_stop_request_stops_early_without_waiting_for_deadline(self):
        m = module()
        raw, audit, row = fixtures()
        provider = Provider(row)
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "new"
            clock = Clock()
            def request_on_sleep(seconds):
                clock.value += seconds
                (state / "stop.request").touch()
            clock.sleep = request_on_sleep
            result = m.run_watchdog(audit, raw, state, provider=provider, clock=clock,
                                    install_signals=False)
            self.assertEqual(result["stop_reason"], "stop.request")
            self.assertEqual(provider.stops, [50149729])
            self.assertLess(clock.value, 10000 + 1800)

    def test_stopped_live_instance_is_never_armed_or_restarted(self):
        m = module()
        raw, audit, row = fixtures()
        provider = Provider(dict(row, actual_status="exited", cur_state="stopped"))
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "new"
            with self.assertRaises(ValueError):
                m.run_watchdog(audit, raw, state, provider=provider, clock=Clock(),
                               install_signals=False)
            self.assertFalse((state / "armed.json").exists())
            self.assertEqual(provider.stops, [50149729])

    def test_bad_audit_has_no_cloud_effects(self):
        m = module()
        raw, audit, row = fixtures()
        provider = Provider(row)
        audit["original_receipt_sha256"] = "wrong"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                m.run_watchdog(audit, raw, Path(tmp) / "new", provider=provider,
                               clock=Clock(), install_signals=False)
            self.assertEqual(provider.calls, 0)
            self.assertEqual(provider.stops, [])

    def test_stop_verification_failure_does_not_claim_stopped(self):
        m = module()
        raw, audit, row = fixtures()
        provider = Provider(row)
        def refuse_stop(instance):
            raise RuntimeError("provider stop unavailable")
        provider.stop_instance = refuse_stop
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "new"
            with self.assertRaises(RuntimeError):
                m.run_watchdog(audit, raw, state, max_seconds=1, provider=provider,
                               clock=Clock(), install_signals=False)
            self.assertEqual(json.loads((state / "status.json").read_text())["status"], "stopping")
