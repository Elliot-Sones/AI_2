
from ai2.paths import source_path
import json
from pathlib import Path
import subprocess
import time
from tempfile import TemporaryDirectory
import unittest


class ContinueContactCampaignTests(unittest.TestCase):
    def subject(self):
        import ai2.runtime.continue_contact_campaign as continue_contact_campaign
        return continue_contact_campaign

    def test_plan_order_seeds_rollouts_and_resume_parents(self):
        m = self.subject()
        plan = m.build_plan()
        train_jobs = [step for step in plan if step.kind == "train"]
        self.assertEqual(
            [job.name for job in train_jobs],
            [
                "B_projected-020m",
                "G_keep-030m",
                "B_projected-030m",
                "G_keep-040m",
                "B_projected-040m",
                "G_keep-050m",
                "B_projected-050m",
            ],
        )
        self.assertEqual([job.seed for job in train_jobs], [20145, 30145, 30145, 40145, 40145, 50145, 50145])
        self.assertEqual([job.rollouts for job in train_jobs], [5, 5, 5, 5, 5, 4, 4])
        self.assertEqual(train_jobs[0].resume_from, "B_projected-010m")
        self.assertEqual(train_jobs[1].resume_from, "G_keep-020m")

    def test_started_remote_job_is_waited_not_launched_again(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_train(root, "B_projected-010m")
            remote = FakeRemote(started={"B_projected-020m"})
            controller = m.Controller(
                local_root=root,
                remote=remote,
                sleeper=lambda _: None,
                now=StepClock([1, 2, 3, 4, 5, 6, 7]),
                deadline_unix=999,
                max_wait_seconds=1,
            )
            with self.assertRaises(TimeoutError):
                controller.ensure_job_started_or_finished(
                    m.PlanStep(
                        kind="train",
                        name="B_projected-020m",
                        arm="B_projected",
                        stage=20,
                        seed=20145,
                        rollouts=5,
                        resume_from="B_projected-010m",
                    ),
                    ["python3", "train_contact_campaign_block.py"],
                )
        self.assertEqual(remote.launches, [])
        self.assertIn(("started", "B_projected-020m"), remote.events)

    def test_launch_timeout_with_started_receipt_is_adopted_not_relaunched(self):
        m = self.subject()
        remote = TimeoutThenStartedRemote(m, started={"eval-G_keep-040m"})
        remote.launch_job("eval-G_keep-040m", ["python3", "evaluate_native_contact_training.py"])
        self.assertEqual(remote.launch_commands, 1)
        self.assertIn(("started", "eval-G_keep-040m"), remote.events)

    def test_local_grouped_background_launch_shell_returns_promptly(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "jobs"
            source = root / "source"
            jobs.mkdir()
            source.mkdir()
            (source_path(source, "run_campaign_job.py")).write_text(
                "import subprocess, sys\n"
                "separator = sys.argv.index('--')\n"
                "subprocess.Popen(sys.argv[separator + 1:], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()\n"
            )
            command = m.build_background_launch_shell(
                remote_root=root,
                deadline_unix=time.time() + 300,
                name="sleep_job",
                command=["python3", "-c", "import time; time.sleep(2)"],
            )
            self.assertIn("source/run_campaign_job.py", command)
            self.assertIn("--deadline-unix", command)
            self.assertNotIn("tools/run_job.py", command)
            started = time.monotonic()
            result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, timeout=1.0)
            elapsed = time.monotonic() - started
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertLess(elapsed, 1.0)
            self.assertRegex(result.stdout.strip(), r"^\d+$")

    def test_background_launch_selects_layout_when_shell_executes(self):
        m = self.subject()
        for packaged in (False, True):
            with self.subTest(packaged=packaged), TemporaryDirectory() as directory:
                root = Path(directory) / "remote root"
                source = root / "source"
                source.mkdir(parents=True)
                (root / "jobs").mkdir()
                # Construct before either layout exists: selection belongs to
                # the shell on the execution host, not the controller host.
                command = m.build_background_launch_shell(
                    remote_root=root, deadline_unix=time.time() + 300,
                    name="layout_check", command=["python3", "train_contact_campaign_block.py"],
                )
                script = source / "run_campaign_job.py"
                if packaged:
                    (source / "ai2/runtime").mkdir(parents=True)
                    (source / "ai2/paths.py").touch()
                    script = source / "ai2/runtime/run_campaign_job.py"
                marker = root / "launched.json"
                script.write_text(
                    "import json, pathlib, sys\n"
                    f"pathlib.Path({str(marker)!r}).write_text(json.dumps(sys.argv))\n"
                )
                result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, timeout=2)
                self.assertEqual(result.returncode, 0, result.stderr)
                until = time.monotonic() + 3
                while not marker.is_file() and time.monotonic() < until:
                    time.sleep(0.01)
                self.assertTrue(marker.is_file(), (root / "jobs/layout_check.controller-launch.log").read_text())
                argv = json.loads(marker.read_text())
                self.assertEqual((root / argv[0]).resolve(), script.resolve())
                self.assertEqual(argv[-2:], ["python3", "train_contact_campaign_block.py"])

    def test_training_command_uses_actual_preserved_checkpoint_sha(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected_sha = self.preserved_train(root, "G_keep-020m")
            controller = m.Controller(
                local_root=root,
                remote=FakeRemote(remote_artifacts=self.remote_artifacts_for(root, "G_keep-020m")),
                deadline_unix=999,
            )
            step = m.PlanStep(
                kind="train",
                name="G_keep-030m",
                arm="G_keep",
                stage=30,
                seed=30145,
                rollouts=5,
                resume_from="G_keep-020m",
            )
            command = controller.train_command(step)
        self.assertIn("--checkpoint-sha256", command)
        self.assertEqual(command[command.index("--checkpoint-sha256") + 1], expected_sha)
        self.assertIn("--resume", command)
        self.assertEqual(command[command.index("--seed") + 1], "30145")

    def test_local_completed_job_requires_remote_rc0_and_is_not_relaunched_when_missing(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parent_sha = self.preserved_train(root, "B_projected-010m", policy_after="parent-after")
            self.preserved_train(
                root,
                "B_projected-020m",
                resume_from="B_projected-010m",
                resume_sha=parent_sha,
                policy_before="parent-after",
            )
            remote = FakeRemote(remote_artifacts=self.remote_artifacts_for(root, "B_projected-020m"))
            controller = m.Controller(
                local_root=root,
                remote=remote,
                deadline_unix=10000,
                now=StepClock([0, 0, 2]),
                sleeper=lambda _: None,
                max_wait_seconds=1,
            )
            step = m.PlanStep(
                kind="train",
                name="B_projected-020m",
                arm="B_projected",
                stage=20,
                seed=20145,
                rollouts=5,
                resume_from="B_projected-010m",
            )
            with self.assertRaises(TimeoutError):
                controller.run([step])
        self.assertEqual(remote.launches, [])

    def test_local_completed_with_remote_rc0_is_skipped_not_relaunched(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parent_sha = self.preserved_train(root, "B_projected-010m", policy_after="parent-after")
            self.preserved_train(
                root,
                "B_projected-020m",
                resume_from="B_projected-010m",
                resume_sha=parent_sha,
                policy_before="parent-after",
            )
            remote = FakeRemote(
                finished={"B_projected-020m"},
                remote_artifacts=self.remote_artifacts_for(root, "B_projected-020m"),
            )
            controller = m.Controller(
                local_root=root,
                remote=remote,
                deadline_unix=10000,
                now=lambda: 0,
                sleeper=lambda _: None,
                max_wait_seconds=1,
            )
            step = m.PlanStep(
                kind="train",
                name="B_projected-020m",
                arm="B_projected",
                stage=20,
                seed=20145,
                rollouts=5,
                resume_from="B_projected-010m",
            )
            result = controller.run([step])
        self.assertEqual(result["completed"][0]["name"], "B_projected-020m")
        self.assertEqual(remote.launches, [])

    def test_exact_train_receipt_rejects_wrong_rollout_count(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parent_after = "parent-policy-after"
            parent_sha = self.preserved_train(root, "G_keep-020m", policy_after=parent_after)
            self.preserved_train(
                root,
                "G_keep-030m",
                rollouts=4,
                stage=30,
                arm="G_keep",
                seed=30145,
                resume_from="G_keep-020m",
                resume_sha=parent_sha,
                policy_before=parent_after,
            )
            controller = m.Controller(local_root=root, remote=FakeRemote(), deadline_unix=999)
            step = m.PlanStep(
                kind="train",
                name="G_keep-030m",
                arm="G_keep",
                stage=30,
                seed=30145,
                rollouts=5,
                resume_from="G_keep-020m",
            )
            preserved = m.verify_local_preservation(root, "G_keep-030m", require_final_zip=True)
            with self.assertRaisesRegex(ValueError, "rollout count"):
                controller.validate_train_receipt(step, preserved)

    def test_local_remote_inventory_mismatch_is_rejected(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_train(root, "G_keep-020m")
            remote_artifacts = self.remote_artifacts_for(root, "G_keep-020m")
            remote_inventory = json.loads(remote_artifacts[("G_keep-020m", "inventory.json")])
            remote_inventory["files"][0]["sha256"] = "0" * 64
            remote_artifacts[("G_keep-020m", "inventory.json")] = json.dumps(remote_inventory, sort_keys=True).encode()
            remote_artifacts[("G_keep-020m", "preservation_ack.json")] = json.dumps({
                "verified": True,
                "block_id": "G_keep-020m",
                "inventory_sha256": m.bytes_sha256(remote_artifacts[("G_keep-020m", "inventory.json")]),
            }).encode()
            controller = m.Controller(
                local_root=root,
                remote=FakeRemote(remote_artifacts=remote_artifacts),
                deadline_unix=999,
                now=StepClock([0, 0, 2]),
                sleeper=lambda _: None,
                max_wait_seconds=1,
            )
            with self.assertRaisesRegex(ValueError, "remote inventory"):
                controller.wait_for_preservation("G_keep-020m", require_final_zip=True)

    def test_retention_alarm_stops_only_that_arm(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_train(root, "G_keep-020m")
            self.preserved_train(root, "B_projected-020m")
            self.preserved_eval(root, "eval-B_projected-020m", pause=True)
            controller = m.Controller(local_root=root, remote=FakeRemote(), deadline_unix=999)
            controller.check_eval_gate("eval-B_projected-020m", arm="B_projected")
        self.assertIn("B_projected", controller.stopped_arms)
        self.assertNotIn("G_keep", controller.stopped_arms)

    def test_eval_candidate_checkpoint_mismatch_is_rejected(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_train(root, "G_keep-020m")
            self.preserved_eval(root, "eval-G_keep-020m", pause=False)
            manifest = root / "eval-G_keep-020m" / "manifest.json"
            manifest.write_text(json.dumps({"candidate": {"checkpoint_sha256": "wrong"}}))
            # Re-seal the tampered manifest so this test targets semantic linkage,
            # not preservation hash failure.
            self.inventory_and_ack(root / "eval-G_keep-020m", "eval-G_keep-020m", ["result.json", "manifest.json"])
            controller = m.Controller(local_root=root, remote=FakeRemote(), deadline_unix=999)
            with self.assertRaisesRegex(ValueError, "candidate checkpoint_sha256"):
                controller.check_eval_gate("eval-G_keep-020m", arm="G_keep")

    def preserved_train(
        self,
        root,
        name,
        *,
        rollouts=None,
        stage=None,
        arm=None,
        seed=None,
        resume_from=None,
        resume_sha=None,
        policy_before="policy-before",
        policy_after="policy-after",
    ):
        m = self.subject()
        arm = arm or ("B_projected" if name.startswith("B_projected") else "G_keep")
        stage = stage if stage is not None else int(name.rsplit("-", 1)[1].removesuffix("m"))
        rollouts = rollouts if rollouts is not None else (4 if stage == 50 else 5)
        seed = seed if seed is not None else {10: 145, 20: 20145, 30: 30145, 40: 40145, 50: 50145}.get(stage, 20145)
        path = source_path(root, name)
        path.mkdir()
        (path / "final.zip").write_bytes((name + "-checkpoint").encode())
        sha = m.sha256(path / "final.zip")
        added = rollouts * m.TRANSITIONS_PER_ROLLOUT
        final = m.expected_counter_for_stage(stage)
        rollout_rows = []
        start = final - added
        for index in range(rollouts):
            rollout_rows.append({
                "rollout": index,
                "transitions": m.TRANSITIONS_PER_ROLLOUT,
                "game_ticks": m.TRANSITIONS_PER_ROLLOUT,
                "cumulative_timesteps": start + (index + 1) * m.TRANSITIONS_PER_ROLLOUT,
                "actual_update": {
                    "optimizer_steps": 512,
                    "nominal_optimizer_steps": 512,
                    "optimizer_step_fraction": 1.0,
                    "kl_early_stop": False,
                    "low_update": False,
                },
                "metrics": {"train/loss": 1.0},
            })
        (path / "manifest.json").write_text(json.dumps({
            "config": {
                "action_ab_arm": arm,
                "seed": seed,
                "rollouts": rollouts,
                "lanes": 4096,
                "n_steps": 512,
                "episodes_per_lane": 640,
            },
            "checkpoint": {"sha256": resume_sha or "parent-sha"},
        }))
        (path / "result.json").write_text(json.dumps({
            "status": "trained_not_evaluated",
            "checkpoint_sha256": sha,
            "summary": {
                "measured_transitions": added,
                "measured_game_ticks": added,
                "measured_rollouts": rollouts,
                "measured_training_steps_per_second": 100.0,
                "measured_window_steps_per_second": 90.0,
            },
            "rollouts": rollout_rows,
            "final_model_counters": {"num_timesteps": final},
            "resume_from_checkpoint_sha256": resume_sha,
            "policy_parameter_digest_before": policy_before,
            "policy_parameter_digest_after": policy_after,
        }))
        self.inventory_and_ack(path, name, ["final.zip", "result.json", "manifest.json"])
        return sha

    def preserved_eval(self, root, name, *, pause):
        path = source_path(root, name)
        path.mkdir()
        train = name.removeprefix("eval-")
        candidate_sha = self.subject().sha256(root / train / "final.zip") if (root / train / "final.zip").exists() else "sha"
        (path / "manifest.json").write_text(json.dumps({
            "candidate": {"checkpoint_sha256": candidate_sha},
        }))
        (path / "result.json").write_text(json.dumps({
            "status": "evaluated_not_promoted",
            "nearby_retention": {"pause_updates": pause},
        }))
        self.inventory_and_ack(path, name, ["result.json", "manifest.json"])

    def remote_artifacts_for(self, root, name):
        path = Path(root) / name
        return {
            (name, "inventory.json"): (path / "inventory.json").read_bytes(),
            (name, "preservation_ack.json"): (path / "preservation_ack.json").read_bytes(),
        }

    def inventory_and_ack(self, path, name, files):
        m = self.subject()
        inventory = {
            "schema_version": "curriculum-inventory-v1",
            "block_id": name,
            "files": [
                {
                    "path": file,
                    "bytes": (path / file).stat().st_size,
                    "sha256": m.sha256(path / file),
                }
                for file in files
            ],
        }
        (path / "inventory.json").write_text(json.dumps(inventory, sort_keys=True))
        (path / "preservation_ack.json").write_text(json.dumps({
            "verified": True,
            "block_id": name,
            "inventory_sha256": m.sha256(path / "inventory.json"),
        }))


class FakeRemote:
    def __init__(self, *, started=(), finished=(), remote_artifacts=None):
        self.started = set(started)
        self.finished = set(finished)
        self.remote_artifacts = remote_artifacts or {}
        self.launches = []
        self.events = []

    def job_receipt(self, name):
        if name in self.finished:
            return {"returncode": 0, "name": name}
        return None

    def job_started(self, name):
        if name in self.started:
            self.events.append(("started", name))
            return {"pid": 123, "name": name}
        return None

    def launch_job(self, name, command):
        self.launches.append((name, list(command)))
        self.started.add(name)

    def read_artifact_file(self, name, filename):
        return self.remote_artifacts[(name, filename)]


class TimeoutThenStartedRemote:
    def __init__(self, module, *, started=()):
        self.module = module
        self.client = module.RemoteClient(remote_root=Path("/remote"))
        self.client.run_shell = self.run_shell
        self.started = set(started)
        self.events = []
        self.launch_attempts = 0
        self.launch_commands = 0

    def run_shell(self, command, *, timeout=120):
        self.launch_attempts += 1
        if "source/run_campaign_job.py" in command:
            self.launch_commands += 1
        if self.launch_attempts == 1:
            raise subprocess.TimeoutExpired(command, timeout)
        if "started.json" in command:
            name = command.rsplit("/", 1)[-1].split(".started.json", 1)[0].strip("'\"")
            if name in self.started:
                self.events.append(("started", name))
                return subprocess.CompletedProcess([], 0, json.dumps({"pid": 123, "name": name}).encode(), b"")
        return subprocess.CompletedProcess([], 1, b"", b"missing")

    def launch_job(self, name, command):
        return self.client.launch_job(name, command)


class StepClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0) if self.values else 9999


if __name__ == "__main__":
    unittest.main()
