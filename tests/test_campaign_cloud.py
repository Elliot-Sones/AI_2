import importlib.util
import json
import math
import subprocess
from pathlib import Path
import sys
import tempfile
import time
import unittest


class CampaignCloudTests(unittest.TestCase):
    def test_retained_instance_guard_keeps_limits_and_refuses_overlap(self):
        path = Path(__file__).resolve().parents[1] / "campaign_cloud.py"
        self.assertTrue(path.exists(), "campaign adapter is missing")
        spec = importlib.util.spec_from_file_location("campaign_cloud", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["campaign_cloud"] = module
        spec.loader.exec_module(module)
        guard = module.load_guard()
        self.assertEqual(guard.REMOTE_ROOT, "/workspace/AI_2/throughput-scaling-20260906")
        self.assertEqual(guard.INSTANCE_ID, 49994285)
        self.assertEqual(guard.MAX_SECONDS, 7200)
        self.assertEqual(guard.RETRIEVAL_RESERVE_SECONDS, 300)
        self.assertEqual(guard.PRICE_CEILING, 0.30)
        selected = dict(id=49994285, actual_status="exited", cur_state="stopped",
                        gpu_name="RTX 4090", num_gpus=1, dph_total=0.28556,
                        label=guard.INSTANCE_LABEL)
        self.assertEqual(guard.validate_account([selected]), selected)
        with self.assertRaises(guard.GuardError):
            guard.validate_account([selected, dict(id=50022933, actual_status="running")])

    def test_instance_record_binds_guard_and_base_constants(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            record_path = Path(tmp) / "instance.json"
            record_path.write_text(json.dumps(instance_record(id=50012345, label="e1-replacement")))

            guard = module.load_guard(record_path)

            self.assertEqual(guard.INSTANCE_ID, 50012345)
            self.assertEqual(guard._base.INSTANCE_ID, 50012345)
            self.assertEqual(guard.INSTANCE_LABEL, "e1-replacement")
            self.assertEqual(guard._base.INSTANCE_LABEL, "e1-replacement")
            self.assertEqual(guard.PRICE_CEILING, 0.40)
            self.assertEqual(guard._base.PRICE_CEILING, 0.40)
            self.assertEqual(guard.DEFAULT_SSH_HOST, "ssh-new.vast.ai")
            self.assertEqual(guard._base.DEFAULT_SSH_HOST, "ssh-new.vast.ai")
            self.assertEqual(guard.DEFAULT_SSH_PORT, 40022)
            self.assertEqual(guard._base.DEFAULT_SSH_PORT, 40022)
            self.assertEqual(guard.REMOTE_ROOT, "/workspace/AI_2/throughput-scaling-20260906")
            self.assertEqual(guard._base.REMOTE_ROOT, "/workspace/AI_2/throughput-scaling-20260906")
            self.assertEqual(guard.REMOTE_PYTHON, "/opt/conda/bin/python")
            self.assertEqual(guard._base.REMOTE_PYTHON, "/opt/conda/bin/python")
            self.assertEqual(guard.READINESS_TIMEOUT_SECONDS, 900)
            self.assertEqual(guard._base.READINESS_TIMEOUT_SECONDS, 900)

    def test_instance_record_rejects_untrusted_stale_and_missing_fields(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            stale = instance_record(created_by_campaign=True, created_unix=time.time() - 601)
            path.write_text(json.dumps(stale))
            with self.assertRaisesRegex(RuntimeError, "not recent"):
                module.load_guard(path)

            bad = instance_record(created_by_campaign=False)
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError, "created_by_campaign"):
                module.load_guard(path)

            missing = instance_record()
            del missing["label"]
            path.write_text(json.dumps(missing))
            with self.assertRaisesRegex(RuntimeError, "label"):
                module.load_guard(path)

    def test_adopted_validate_account_allows_selected_running_and_rejects_old_running(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            record_path = Path(tmp) / "instance.json"
            record_path.write_text(json.dumps(instance_record(id=50012345, label="e1-replacement")))
            guard = module.load_guard(record_path)
            selected = dict(
                id=50012345,
                actual_status="running",
                cur_state="running",
                gpu_name="RTX 4090",
                num_gpus=1,
                dph_total=0.39,
                label="e1-replacement",
            )

            self.assertEqual(guard.validate_account([selected]), selected)
            with self.assertRaises(guard.GuardError):
                guard.validate_account([selected, dict(id=49994285, actual_status="running", cur_state="running")])

    def test_adopted_provider_does_not_start_already_running_instance(self):
        module = load_module()
        receipt = module.InstanceRecord.from_mapping(instance_record(id=50012345, label="e1-replacement"), now=time.time())
        guard = module.load_guard_from_record(receipt)
        selected = dict(
            id=50012345,
            actual_status="running",
            cur_state="running",
            gpu_name="RTX 4090",
            num_gpus=1,
            dph_total=0.39,
            label="e1-replacement",
        )
        provider = FakeProvider(guard, [selected])
        adopted = module.AdoptedProvider(provider, receipt, guard)

        result = adopted.start_instance(50012345)

        self.assertEqual(result["already_running"], True)
        self.assertEqual(provider.start_calls, [])

    def test_adopted_provider_starts_stopped_instance_once_and_rejects_queued_ack(self):
        module = load_module()
        receipt = module.InstanceRecord.from_mapping(instance_record(id=50012345, label="e1-replacement"), now=time.time())
        guard = module.load_guard_from_record(receipt)
        selected = dict(
            id=50012345,
            actual_status="exited",
            cur_state="stopped",
            gpu_name="RTX 4090",
            num_gpus=1,
            dph_total=0.39,
            label="e1-replacement",
        )
        provider = FakeProvider(guard, [selected], start_result={"acknowledged": True})
        adopted = module.AdoptedProvider(provider, receipt, guard)

        adopted.start_instance(50012345)

        self.assertEqual(provider.start_calls, [50012345])

        queued = FakeProvider(
            guard,
            [selected],
            start_result="Required resources are currently unavailable, state change queued.\nnull\n",
        )
        with self.assertRaises(guard.GuardError):
            module.AdoptedProvider(queued, receipt, guard).start_instance(50012345)

    def test_adopted_base_ack_classifier_rejects_cli_null_and_queued_text(self):
        module = load_module()
        receipt = module.InstanceRecord.from_mapping(instance_record(id=50012345, label="e1-replacement"), now=time.time())
        guard = module.load_guard_from_record(receipt)

        with self.assertRaises(guard.GuardError):
            guard._base._ack_result("Required resources are currently unavailable, state change queued.\nnull\n")
        with self.assertRaises(guard.GuardError):
            guard._base._ack_result("null\n")
        self.assertEqual(
            guard._base._ack_result("starting instance 50012345."),
            {"acknowledged": True, "text_ack": True},
        )

    def test_adopted_readiness_uses_workspace_and_conda_python(self):
        module = load_module()
        receipt = module.InstanceRecord.from_mapping(instance_record(id=50012345, label="e1-replacement"), now=time.time())
        guard = module.load_guard_from_record(receipt)
        provider = FakeProvider(guard, [])
        adopted = module.AdoptedProvider(provider, receipt, guard)

        self.assertTrue(adopted.check_readiness("ssh-new.vast.ai", 40022, "/ignored", "/ignored-python"))

        self.assertEqual(provider.readiness_checks, [("ssh-new.vast.ai", 40022, "/workspace", "/opt/conda/bin/python")])

    def test_resume_audit_binds_original_receipt_after_stopped_audit(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume-audit.json"
            path.write_text(json.dumps(resume_audit(instance_id=50012345, label="e1-replacement")))

            guard = module.load_guard(resume_audit_path=path)

            self.assertEqual(guard.INSTANCE_ID, 50012345)
            self.assertEqual(guard.INSTANCE_LABEL, "e1-replacement")
            self.assertEqual(guard.PRICE_CEILING, 0.40)
            self.assertEqual(guard.REMOTE_BASE_ROOT, "/workspace")
            self.assertEqual(guard._base.REMOTE_BASE_ROOT, "/workspace")
            self.assertEqual(guard.REMOTE_PYTHON, "/opt/conda/bin/python")
            self.assertEqual(guard._base.REMOTE_PYTHON, "/opt/conda/bin/python")
            self.assertEqual(guard.READINESS_TIMEOUT_SECONDS, 900)
            self.assertEqual(guard._base.READINESS_TIMEOUT_SECONDS, 900)
            selected = dict(
                id=50012345,
                actual_status="exited",
                cur_state="stopped",
                gpu_name="RTX 4090",
                num_gpus=1,
                dph_total=0.39,
                label="e1-replacement",
            )
            self.assertEqual(guard.validate_account([selected]), selected)

    def test_resume_audit_rejects_nonfinite_negative_and_future_numeric_values(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume-audit.json"

            for value in (math.nan, math.inf, -1.0):
                path.write_text(json.dumps(resume_audit(cumulative_spent=value), allow_nan=True))
                with self.assertRaisesRegex(RuntimeError, "cumulative_spent"):
                    module.load_guard(resume_audit_path=path)

            path.write_text(json.dumps(resume_audit(audit_unix=time.time() + 1.0)))
            with self.assertRaisesRegex(RuntimeError, "audit_unix"):
                module.load_guard(resume_audit_path=path)

            audit = resume_audit()
            audit["original_campaign_receipt"]["created_unix"] = time.time() + 1.0
            path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(RuntimeError, "created_unix"):
                module.load_guard(resume_audit_path=path)

            audit = resume_audit()
            audit["original_campaign_receipt"]["quoted_hourly_price"] = math.inf
            path.write_text(json.dumps(audit),)
            with self.assertRaisesRegex(RuntimeError, "quoted_hourly_price"):
                module.load_guard(resume_audit_path=path)

    def test_resume_audit_requires_conservative_batch_ceiling_floor(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume-audit.json"

            path.write_text(json.dumps(resume_audit(planned_batch_ceiling=0.80)))
            with self.assertRaisesRegex(RuntimeError, "planned_batch_ceiling"):
                module.load_guard(resume_audit_path=path)

            path.write_text(json.dumps(resume_audit(planned_batch_ceiling=0.90)))
            guard = module.load_guard(resume_audit_path=path)
            self.assertEqual(guard.INSTANCE_ID, 50012345)

    def test_instance_record_and_provider_price_reject_invalid_ranges(self):
        module = load_module()
        with self.assertRaisesRegex(RuntimeError, "hourly_price"):
            module.InstanceRecord.from_mapping(instance_record(hourly_price=math.nan), now=time.time())
        with self.assertRaisesRegex(RuntimeError, "created_unix"):
            module.InstanceRecord.from_mapping(instance_record(created_unix=time.time() + 1.0), now=time.time())
        with self.assertRaisesRegex(RuntimeError, "ssh_port"):
            module.InstanceRecord.from_mapping(instance_record(ssh_port=70000), now=time.time())

        receipt = module.InstanceRecord.from_mapping(instance_record(hourly_price=0.39), now=time.time())
        guard = module.load_guard_from_record(receipt)
        with self.assertRaises(guard.GuardError):
            guard.validate_account([
                dict(
                    id=50012345,
                    actual_status="running",
                    cur_state="running",
                    gpu_name="RTX 4090",
                    num_gpus=1,
                    dph_total=math.nan,
                    label="e1-replacement",
                )
            ])

    def test_resume_audit_rejects_wrong_id_label_stale_and_overspend(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume-audit.json"

            path.write_text(json.dumps(resume_audit(instance_id=50012345, selected_id=50099999)))
            with self.assertRaisesRegex(RuntimeError, "selected id"):
                module.load_guard(resume_audit_path=path)

            path.write_text(json.dumps(resume_audit(label="e1-replacement", selected_label="other-label")))
            with self.assertRaisesRegex(RuntimeError, "selected label"):
                module.load_guard(resume_audit_path=path)

            path.write_text(json.dumps(resume_audit(audit_unix=time.time() - 601)))
            with self.assertRaisesRegex(RuntimeError, "audit is stale"):
                module.load_guard(resume_audit_path=path)

            path.write_text(json.dumps(resume_audit(cumulative_spent=11.30, planned_batch_ceiling=0.90)))
            with self.assertRaisesRegex(RuntimeError, "approved budget"):
                module.load_guard(resume_audit_path=path)

    def test_resume_audit_rejects_unrelated_running_and_selected_not_stopped(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume-audit.json"

            path.write_text(json.dumps(resume_audit(selected_status="running", selected_state="running")))
            with self.assertRaisesRegex(RuntimeError, "not stopped"):
                module.load_guard(resume_audit_path=path)

            audit = resume_audit()
            audit["live_account_audit"]["instances"].append(
                {"id": 49994285, "actual_status": "running", "cur_state": "running", "label": "old"}
            )
            path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(RuntimeError, "unrelated running"):
                module.load_guard(resume_audit_path=path)

    def test_resume_provider_checks_current_identity_before_noop_or_start(self):
        module = load_module()
        record = module.ResumeAudit.from_mapping(resume_audit(instance_id=50012345, label="e1-replacement"), now=time.time())
        guard = module.load_guard_from_resume_audit(record)
        selected = dict(
            id=50012345,
            actual_status="running",
            cur_state="running",
            gpu_name="RTX 4090",
            num_gpus=1,
            dph_total=0.39,
            label="e1-replacement",
        )
        provider = FakeProvider(guard, [selected])
        adopted = module.ResumeProvider(provider, record, guard)

        result = adopted.start_instance(50012345)

        self.assertEqual(result["already_running"], True)
        self.assertEqual(provider.start_calls, [])
        self.assertEqual(provider.list_calls, 1)

        wrong = FakeProvider(guard, [dict(selected, label="wrong")])
        with self.assertRaises(guard.GuardError):
            module.ResumeProvider(wrong, record, guard).start_instance(50012345)

    def test_resume_provider_readiness_uses_workspace_and_conda_python(self):
        module = load_module()
        record = module.ResumeAudit.from_mapping(resume_audit(instance_id=50012345, label="e1-replacement"), now=time.time())
        guard = module.load_guard_from_resume_audit(record)
        provider = FakeProvider(guard, [])
        resumed = module.ResumeProvider(provider, record, guard)

        self.assertTrue(resumed.check_readiness("ssh-new.vast.ai", 40022, "/ignored", "/ignored-python"))

        self.assertEqual(provider.readiness_checks, [("ssh-new.vast.ai", 40022, "/workspace", "/opt/conda/bin/python")])

    def test_resumable_retrieval_builds_safe_rsync_commands_and_completion_flags(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "retrieved"
            state_dir = Path(tmp) / "state"
            popen = FakePopenFactory([FakeProcess(returncode=0), FakeProcess(returncode=0)])

            result = module.resumable_retrieve(
                host="ssh-new.vast.ai",
                port=40022,
                remote_root="/workspace/AI_2/throughput-scaling-20260906",
                local_dir=local_dir,
                state_dir=state_dir,
                timeout_seconds=25,
                popen_factory=popen,
            )

            self.assertEqual(result, {"artifacts": True, "wandb": True})
            self.assertEqual(len(popen.commands), 2)
            for command in popen.commands:
                self.assertEqual(command[0], "rsync")
                self.assertIn("--partial", command)
                self.assertIn("--partial-dir=.rsync-partial", command)
                self.assertNotIn("--delete", command)
                self.assertNotIn("--inplace", command)
                self.assertTrue(popen.kwargs_by_call[popen.commands.index(command)]["start_new_session"])
                remote = command[-2]
                target = command[-1]
                self.assertTrue(remote.startswith("root@ssh-new.vast.ai:"))
                self.assertIn("'/workspace/AI_2/throughput-scaling-20260906'/", remote)
                self.assertTrue(target.endswith("/"))
                ssh_command = command[command.index("-e") + 1]
                self.assertIn("-p 40022", ssh_command)
                self.assertIn("BatchMode=yes", ssh_command)
                self.assertIn("StrictHostKeyChecking=yes", ssh_command)
                self.assertNotIn("StrictHostKeyChecking=accept-new", ssh_command)
                self.assertNotIn("UserKnownHostsFile=", ssh_command)
            self.assertTrue(read_retrieval_status(local_dir, "artifacts")["success"])
            self.assertTrue(read_retrieval_status(local_dir, "wandb")["success"])
            private_log = state_dir / "retrieval-private.jsonl"
            self.assertEqual(private_log.stat().st_mode & 0o777, 0o600)
            for line in private_log.read_text().splitlines():
                record = json.loads(line)
                self.assertIn(record["name"], {"artifacts", "wandb"})
                self.assertIn(record["status"], {0})
                self.assertNotIn("stdout", record)
                self.assertNotIn("stderr", record)

    def test_resumable_retrieval_times_out_without_marking_complete_and_still_attempts_wandb(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "retrieved"
            state_dir = Path(tmp) / "state"
            artifacts = FakeProcess(timeout=True)
            wandb = FakeProcess(returncode=0)
            popen = FakePopenFactory([artifacts, wandb])

            result = module.resumable_retrieve(
                host="ssh-new.vast.ai",
                port=40022,
                remote_root="/workspace/AI_2/throughput-scaling-20260906",
                local_dir=local_dir,
                state_dir=state_dir,
                timeout_seconds=25,
                kill_grace_seconds=0.01,
                popen_factory=popen,
                killpg=lambda pgid, sig: artifacts.signals.append((pgid, sig)),
            )

            self.assertEqual(result, {"artifacts": False, "wandb": True})
            self.assertEqual(len(popen.commands), 2)
            self.assertFalse(read_retrieval_status(local_dir, "artifacts")["success"])
            self.assertTrue((local_dir / "artifacts" / ".rsync-partial").exists())
            self.assertTrue(read_retrieval_status(local_dir, "wandb")["success"])
            self.assertEqual([signal for _, signal in artifacts.signals], [module.signal.SIGTERM, module.signal.SIGKILL])
            records = [json.loads(line) for line in (state_dir / "retrieval-private.jsonl").read_text().splitlines()]
            self.assertEqual([record["name"] for record in records], ["artifacts", "wandb"])
            self.assertEqual(records[0]["status"], "timeout_killed")
            self.assertEqual(records[1]["status"], 0)

    def test_later_failed_retrieval_invalidates_previous_success_status(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "retrieved"
            state_dir = Path(tmp) / "state"

            ok_result = module.resumable_retrieve(
                host="ssh-new.vast.ai",
                port=40022,
                remote_root="/workspace/AI_2/throughput-scaling-20260906",
                local_dir=local_dir,
                state_dir=state_dir,
                popen_factory=FakePopenFactory([FakeProcess(returncode=0), FakeProcess(returncode=0)]),
            )
            self.assertEqual(ok_result, {"artifacts": True, "wandb": True})
            self.assertTrue(read_retrieval_status(local_dir, "artifacts")["success"])

            timeout_result = module.resumable_retrieve(
                host="ssh-new.vast.ai",
                port=40022,
                remote_root="/workspace/AI_2/throughput-scaling-20260906",
                local_dir=local_dir,
                state_dir=state_dir,
                kill_grace_seconds=0.01,
                popen_factory=FakePopenFactory([FakeProcess(timeout=True), FakeProcess(returncode=0)]),
                killpg=lambda _pgid, _sig: None,
            )

            self.assertEqual(timeout_result, {"artifacts": False, "wandb": True})
            artifacts_status = read_retrieval_status(local_dir, "artifacts")
            self.assertFalse(artifacts_status["success"])
            self.assertEqual(artifacts_status["status"], "timeout_killed")
            self.assertIn("not a final checksum verification", artifacts_status["note"])
            self.assertTrue(read_retrieval_status(local_dir, "wandb")["success"])

    def test_wrapped_providers_use_adapter_local_resumable_retrieval(self):
        module = load_module()
        receipt = module.InstanceRecord.from_mapping(instance_record(id=50012345, label="e1-replacement"), now=time.time())
        guard = module.load_guard_from_record(receipt)
        base = FakeProvider(guard, [])
        adopted = module.AdoptedProvider(base, receipt, guard)
        with tempfile.TemporaryDirectory() as tmp:
            popen = FakePopenFactory([FakeProcess(returncode=0), FakeProcess(returncode=0)])
            adopted.popen_factory = popen
            result = adopted.retrieve("ssh-new.vast.ai", 40022, "/workspace/AI_2/throughput-scaling-20260906", Path(tmp) / "retrieved")

        self.assertEqual(result, {"artifacts": True, "wandb": True})
        self.assertEqual(len(popen.commands), 2)
        self.assertFalse(base.retrieve_calls)

        audit = module.ResumeAudit.from_mapping(resume_audit(instance_id=50012345, label="e1-replacement"), now=time.time())
        guard = module.load_guard_from_resume_audit(audit)
        base = FakeProvider(guard, [])
        resumed = module.ResumeProvider(base, audit, guard)
        with tempfile.TemporaryDirectory() as tmp:
            popen = FakePopenFactory([FakeProcess(returncode=0), FakeProcess(returncode=0)])
            resumed.popen_factory = popen
            result = resumed.retrieve("ssh-new.vast.ai", 40022, "/workspace/AI_2/throughput-scaling-20260906", Path(tmp) / "retrieved")

        self.assertEqual(result, {"artifacts": True, "wandb": True})
        self.assertEqual(len(popen.commands), 2)
        self.assertFalse(base.retrieve_calls)

    def test_timed_out_retrieval_tolerates_process_group_already_gone(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            process = FakeProcess(timeout=True)

            status, stdout, stderr = module._run_retrieval_command(
                ["rsync", "-az"],
                timeout_seconds=25,
                kill_grace_seconds=0.01,
                popen_factory=FakePopenFactory([process]),
                killpg=lambda _pgid, _sig: (_ for _ in ()).throw(ProcessLookupError()),
            )

            self.assertEqual(status, "timeout_process_gone")
            self.assertEqual(stdout, "out")
            self.assertEqual(stderr, "err")


def load_module():
    path = Path(__file__).resolve().parents[1] / "campaign_cloud.py"
    spec = importlib.util.spec_from_file_location("campaign_cloud", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["campaign_cloud"] = module
    spec.loader.exec_module(module)
    return module


def read_retrieval_status(local_dir, name):
    return json.loads((Path(local_dir) / name / ".retrieval-status.json").read_text())


def instance_record(**overrides):
    record = {
        "created_by_campaign": True,
        "created_unix": time.time(),
        "id": 50012345,
        "label": "e1-replacement",
        "ssh_host": "ssh-new.vast.ai",
        "ssh_port": 40022,
        "hourly_price": 0.39,
    }
    record.update(overrides)
    return record


def resume_audit(**overrides):
    instance_id = overrides.pop("instance_id", 50012345)
    label = overrides.pop("label", "e1-replacement")
    selected_id = overrides.pop("selected_id", instance_id)
    selected_label = overrides.pop("selected_label", label)
    selected_status = overrides.pop("selected_status", "exited")
    selected_state = overrides.pop("selected_state", "stopped")
    audit_unix = overrides.pop("audit_unix", time.time())
    cumulative_spent = overrides.pop("cumulative_spent", 1.25)
    planned_batch_ceiling = overrides.pop("planned_batch_ceiling", 0.90)
    data = {
        "resume_audit": True,
        "audit_unix": audit_unix,
        "user_credit_ceiling": 12.0,
        "cumulative_spent": cumulative_spent,
        "planned_batch_ceiling": planned_batch_ceiling,
        "original_campaign_receipt": {
            "id": instance_id,
            "label": label,
            "created_unix": time.time() - 3600,
            "quoted_hourly_price": 0.344444,
        },
        "live_account_audit": {
            "sanitized": True,
            "audit_unix": audit_unix,
            "instances": [
                {
                    "id": selected_id,
                    "label": selected_label,
                    "actual_status": selected_status,
                    "cur_state": selected_state,
                    "gpu_name": "RTX 4090",
                    "num_gpus": 1,
                    "dph_total": 0.39,
                }
            ],
        },
    }
    data.update(overrides)
    return data


class FakeProvider:
    def __init__(self, guard, instances, start_result=None):
        self.guard = guard
        self.instances = [dict(item) for item in instances]
        self.start_result = start_result if start_result is not None else {"acknowledged": True}
        self.start_calls = []
        self.readiness_checks = []
        self.retrieve_calls = []
        self.list_calls = 0

    def list_instances(self):
        self.list_calls += 1
        return [dict(item) for item in self.instances]

    def start_instance(self, instance_id):
        self.start_calls.append(instance_id)
        return self.start_result

    def check_readiness(self, host, port, remote_root, interpreter):
        self.readiness_checks.append((host, port, remote_root, interpreter))
        return True

    def retrieve(self, host, port, remote_root, local_dir):
        self.retrieve_calls.append((host, port, remote_root, local_dir))
        return {"artifacts": True, "wandb": True}


class FakeProcess:
    def __init__(self, *, returncode=0, timeout=False, stdout="out", stderr="err"):
        self.returncode = returncode
        self.remaining_timeouts = 2 if timeout else 0
        self.stdout = stdout
        self.stderr = stderr
        self.pid = 12345
        self.signals = []

    def communicate(self, timeout=None):
        if self.remaining_timeouts:
            self.remaining_timeouts -= 1
            raise subprocess.TimeoutExpired(cmd=["rsync"], timeout=timeout, output=self.stdout, stderr=self.stderr)
        return self.stdout, self.stderr


class FakePopenFactory:
    def __init__(self, processes):
        self.processes = list(processes)
        self.commands = []
        self.kwargs_by_call = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        self.kwargs_by_call.append(dict(kwargs))
        return self.processes.pop(0)


if __name__ == "__main__":
    unittest.main()
