import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class CausalRetentionCampaignTests(unittest.TestCase):
    def subject(self):
        self.assertIsNotNone(
            importlib.util.find_spec("causal_retention_campaign"),
            "causal retention campaign controller has not been implemented",
        )
        import causal_retention_campaign

        return causal_retention_campaign

    def test_plan_freezes_three_one_rollout_causal_probes(self):
        m = self.subject()
        p1c, p1n = m.build_initial_plan(source_digest=m.PLAN_SOURCE_DIGEST)
        self.assertEqual((p1c.recipe, p1c.seed, p1c.rollouts, p1c.parent_sha256), ("P1C", 510101, 1, m.B40_SHA256))
        self.assertEqual((p1n.recipe, p1n.seed, p1n.rollouts, p1n.parent_sha256), ("P1N", 520102, 1, m.B40_SHA256))
        self.assertEqual(p1c.cumulative_added_transitions, 2_097_152)
        self.assertEqual(p1n.cumulative_added_transitions, 2_097_152)
        self.assertIn("-src" + m.PLAN_SOURCE_DIGEST[:12], p1c.name)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_train(root, p1c.name, recipe="P1C", seed=510101, parent_sha=m.B40_SHA256)
            self.inventory_and_ack(root / m.CONTACT_QUALIFICATION_BLOCK, m.CONTACT_QUALIFICATION_BLOCK, self.qualification_files(root, m.CONTACT_QUALIFICATION_BLOCK))
            controller = m.Controller(local_root=root, remote=FakeRemote(), deadline_unix=1000, now=lambda: 0, remaining_cost=lambda: 5.0)
            repair = m.build_repair_step(controller._step_with_result_sha(p1c), source_digest=m.PLAN_SOURCE_DIGEST)
        self.assertEqual((repair.recipe, repair.seed, repair.rollouts, repair.resume_from), ("P1N", 510103, 1, p1c.name))
        self.assertEqual(repair.cumulative_added_transitions, 4_194_304)

    def test_run_records_train_eval_after_each_probe_and_never_promotes(self):
        m = self.subject()

        class RecordingController(m.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.run_batches = []

            def _run_steps(self, plan):
                self.run_batches.append([step.name for step in plan])
                return super()._run_steps(plan)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_qualification(root, m.CONTACT_QUALIFICATION_BLOCK)
            p1c, p1n = m.build_initial_plan(source_digest=m.PLAN_SOURCE_DIGEST)
            p1c_sha = self.preserved_train(root, p1c.name, recipe="P1C", seed=510101, parent_sha=m.B40_SHA256)
            self.preserved_eval(root, m.eval_step(p1c).name, train_name=p1c.name, result=self.metrics(contact_sampled=15, wide_deterministic_falls=7, wide_sampled_falls=7))
            self.preserved_train(root, p1n.name, recipe="P1N", seed=520102, parent_sha=m.B40_SHA256)
            self.preserved_eval(root, m.eval_step(p1n).name, train_name=p1n.name, result=self.metrics(nearby_sampled=20, wide_deterministic=14, wide_sampled=13, wide_deterministic_falls=6, wide_sampled_falls=7))
            repair = m.build_repair_step(m.replace(p1c, result_sha256=p1c_sha), source_digest=m.PLAN_SOURCE_DIGEST)
            self.preserved_train(root, repair.name, recipe="P1N", seed=510103, parent_sha=p1c_sha, cumulative_rollouts=2)
            self.preserved_eval(root, m.eval_step(repair).name, train_name=repair.name, result=self.metrics(contact_sampled=16, wide_deterministic=15, wide_sampled=14, wide_deterministic_falls=5, wide_sampled_falls=6))
            names = [
                m.CONTACT_QUALIFICATION_BLOCK,
                p1c.name,
                m.eval_step(p1c).name,
                p1n.name,
                m.eval_step(p1n).name,
                repair.name,
                m.eval_step(repair).name,
            ]
            remote = FakeRemote(
                finished=set(names),
                remote_artifacts={key: value for name in names for key, value in self.remote_artifacts_for(root, name).items()},
            )
            controller = RecordingController(local_root=root, remote=remote, deadline_unix=1000, now=lambda: 0, remaining_cost=lambda: 5.0)
            custom_baseline = self.metrics(wide_deterministic=10, wide_sampled=11)
            summary = controller.run(b40_baseline=custom_baseline)

        self.assertEqual(summary["status"], "complete_no_promotion")
        self.assertEqual(
            controller.run_batches,
            [
                [p1c.name, m.eval_step(p1c).name],
                [p1n.name, m.eval_step(p1n).name],
                [repair.name, m.eval_step(repair).name],
            ],
        )
        self.assertEqual([row["name"] for row in summary["probes"]], [p1c.name, p1n.name, repair.name])
        self.assertIn("p1c_vs_p1n", summary["paired_comparisons"])
        self.assertIn("p1n_vs_repair", summary["paired_comparisons"])
        self.assertEqual(
            summary["paired_comparisons"]["vs_b40"][p1c.name]["wide_deterministic"]["baseline"],
            10,
        )
        self.assertEqual(summary["baseline"], custom_baseline)
        self.assertEqual(summary["next_action"], "inspect causal deltas before authorizing any scale-up; no checkpoint is promoted by this controller")

    def test_train_command_uses_b40_for_initial_and_preserved_p1c_for_repair(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            p1c = m.build_p1c_step(source_digest=m.PLAN_SOURCE_DIGEST)
            p1c_sha = self.preserved_train(root, p1c.name, recipe="P1C", seed=510101, parent_sha=m.B40_SHA256)
            remote = FakeRemote(remote_artifacts=self.remote_artifacts_for(root, p1c.name))
            controller = m.Controller(local_root=root, remote=remote, deadline_unix=1000, now=lambda: 0, remaining_cost=lambda: 5.0)
            initial = controller.train_command(p1c)
            repair = m.build_repair_step(m.replace(p1c, result_sha256=p1c_sha), source_digest=m.PLAN_SOURCE_DIGEST)
            repair_command = controller.train_command(repair)

        self.assertEqual(initial[initial.index("--schedule") + 1], "P1C")
        self.assertEqual(initial[initial.index("--checkpoint-sha256") + 1], m.B40_SHA256)
        self.assertEqual(repair_command[repair_command.index("--schedule") + 1], "P1N")
        self.assertEqual(repair_command[repair_command.index("--checkpoint-sha256") + 1], p1c_sha)
        self.assertEqual(Path(repair_command[repair_command.index("--checkpoint-ack") + 1]).name, "preservation_ack.json")

    def test_train_receipt_rejects_kl_early_stop(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            step = m.build_p1c_step(source_digest=m.PLAN_SOURCE_DIGEST)
            self.preserved_train(root, step.name, recipe="P1C", seed=510101, parent_sha=m.B40_SHA256, kl_early_stop=True)
            controller = m.Controller(local_root=root, remote=FakeRemote(remote_artifacts=self.remote_artifacts_for(root, step.name)), deadline_unix=1000, now=lambda: 0, remaining_cost=lambda: 5.0)
            preserved = controller.local_preservation_once(step.name, require_final_zip=True)
            with self.assertRaisesRegex(ValueError, "KL early stop"):
                controller.validate_train_receipt(step, preserved)

    def test_train_receipt_rejects_missing_or_wrong_required_source_hashes(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            step = m.build_p1c_step(source_digest=m.PLAN_SOURCE_DIGEST)
            self.preserved_train(root, step.name, recipe="P1C", seed=510101, parent_sha=m.B40_SHA256)
            controller = m.Controller(
                local_root=root,
                remote=FakeRemote(remote_artifacts=self.remote_artifacts_for(root, step.name)),
                deadline_unix=1000,
                now=lambda: 0,
                remaining_cost=lambda: 5.0,
            )
            preserved = controller.local_preservation_once(step.name, require_final_zip=True)
            for key in ("native_curriculum_ppo.py", "train_retention_campaign_block.py"):
                with self.subTest(key=key, case="missing"):
                    changed = {**preserved, "manifest": json.loads(json.dumps(preserved["manifest"]))}
                    changed["manifest"]["source_sha256"].pop(key)
                    with self.assertRaisesRegex(ValueError, f"source digest mismatch for {key}"):
                        controller.validate_train_receipt(step, changed)
                with self.subTest(key=key, case="wrong"):
                    changed = {**preserved, "manifest": json.loads(json.dumps(preserved["manifest"]))}
                    changed["manifest"]["source_sha256"][key] = "0" * 64
                    with self.assertRaisesRegex(ValueError, f"source digest mismatch for {key}"):
                        controller.validate_train_receipt(step, changed)

    def test_repair_receipt_rejects_a_model_counter_that_only_advanced_one_rollout(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            p1c = m.build_p1c_step(source_digest=m.PLAN_SOURCE_DIGEST)
            p1c_sha = self.preserved_train(root, p1c.name, recipe="P1C", seed=510101, parent_sha=m.B40_SHA256)
            repair = m.build_repair_step(m.replace(p1c, result_sha256=p1c_sha), source_digest=m.PLAN_SOURCE_DIGEST)
            self.preserved_train(
                root,
                repair.name,
                recipe="P1N",
                seed=510103,
                parent_sha=p1c_sha,
                cumulative_rollouts=1,
            )
            controller = m.Controller(
                local_root=root,
                remote=FakeRemote(remote_artifacts=self.remote_artifacts_for(root, repair.name)),
                deadline_unix=1000,
                now=lambda: 0,
                remaining_cost=lambda: 5.0,
            )
            preserved = controller.local_preservation_once(repair.name, require_final_zip=True)
            with self.assertRaisesRegex(ValueError, "final model timestep mismatch"):
                controller.validate_train_receipt(repair, preserved)

    def metrics(
        self,
        *,
        contact_deterministic=19,
        contact_sampled=14,
        nearby_deterministic=20,
        nearby_sampled=19,
        wide_deterministic=13,
        wide_sampled=13,
        nearby_deterministic_falls=0,
        nearby_sampled_falls=1,
        wide_deterministic_falls=7,
        wide_sampled_falls=7,
    ):
        return {
            "schema_version": "ai2.retention_evaluation.v1",
            "status": "evaluated_not_promoted",
            "panels": {
                "contact": {
                    "deterministic": {"episodes": 20, "successes": contact_deterministic, "self_losses": 0, "opponent_self_losses": 0, "timeouts": 0, "ambiguous": 0},
                    "sampled": {"episodes": 20, "successes": contact_sampled, "self_losses": 0, "opponent_self_losses": 0, "timeouts": 0, "ambiguous": 0},
                },
                "nearby": {
                    "deterministic": {"episodes": 20, "successes": nearby_deterministic, "falls": nearby_deterministic_falls},
                    "sampled": {"episodes": 20, "successes": nearby_sampled, "falls": nearby_sampled_falls},
                },
                "wide": {
                    "deterministic": {"episodes": 20, "successes": wide_deterministic, "falls": wide_deterministic_falls},
                    "sampled": {"episodes": 20, "successes": wide_sampled, "falls": wide_sampled_falls},
                },
            },
        }

    def preserved_train(self, root, name, *, recipe, seed, parent_sha, kl_early_stop=False, cumulative_rollouts=1):
        m = self.subject()
        path = Path(root) / name
        path.mkdir(parents=True)
        task = m.CAUSAL_SCHEDULES[recipe][0]
        (path / "final.zip").write_bytes((name + "-checkpoint").encode())
        final_sha = m.sha256(path / "final.zip")
        manifest = {
            "schema_version": "ai2.retention_campaign_block.v1",
            "config": {"schedule_id": recipe, "recipe": recipe, "seed": seed, "rollouts": 1, "action_ab_arm": "B_projected"},
            "schedule_id": recipe,
            "schedule": [task],
            "checkpoint": {"sha256": parent_sha},
            "source_sha256": {
                "native_curriculum_ppo.py": m.retention.CONTACT_SOURCE_DIGEST,
                "train_retention_campaign_block.py": m.retention.RETENTION_RUNNER_DIGEST,
            },
        }
        result = {
            "schema_version": "ai2.retention_campaign_block.v1",
            "status": "trained_not_evaluated",
            "schedule_id": recipe,
            "schedule": [task],
            "checkpoint_sha256": final_sha,
            "summary": {"measured_transitions": m.TRANSITIONS_PER_ROLLOUT},
            "rollouts": [{
                "rollout": 0,
                "task": task,
                "transitions": m.TRANSITIONS_PER_ROLLOUT,
                "game_ticks": m.TRANSITIONS_PER_ROLLOUT,
                "collection_seconds": 1.0,
                "update_seconds": 1.0,
                "total_seconds": 2.0,
                "metrics": {"loss": 0.1},
                "actual_update": {
                    "optimizer_steps": 512,
                    "nominal_optimizer_steps": 512,
                    "kl_early_stop": kl_early_stop,
                    "low_update": False,
                },
            }],
            "task_transition_totals": {"combat": m.TRANSITIONS_PER_ROLLOUT if task == "combat" else 0, "navigation": m.TRANSITIONS_PER_ROLLOUT if task == "navigation" else 0},
            "parent_checkpoint_sha256": parent_sha,
            "final_model_counters": {
                "num_timesteps": m.retention.contact.CANONICAL_PARENT_STEPS
                + cumulative_rollouts * m.TRANSITIONS_PER_ROLLOUT
            },
        }
        (path / "manifest.json").write_text(json.dumps(manifest))
        (path / "result.json").write_text(json.dumps(result))
        self.inventory_and_ack(path, name, ["final.zip", "result.json", "manifest.json"])
        return final_sha

    def preserved_eval(self, root, name, *, train_name, result):
        m = self.subject()
        path = Path(root) / name
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps({
            "candidate": {"checkpoint_sha256": m.sha256(Path(root) / train_name / "final.zip")},
        }))
        (path / "result.json").write_text(json.dumps(result))
        self.inventory_and_ack(path, name, ["result.json", "manifest.json"])

    def preserved_qualification(self, root, name):
        path = Path(root) / name
        path.mkdir(parents=True)
        (path / "qualification.json").write_text(json.dumps({
            "schema_version": "ai2.native_contact_qualification.v1",
            "qualified": True,
            "checks": {"original_cpu_exact": True, "cpu_cuda_exact": True, "learner_update_exact": True},
        }))
        self.inventory_and_ack(path, name, ["qualification.json"])

    def qualification_files(self, root, name):
        path = Path(root) / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "qualification.json").write_text(json.dumps({
            "schema_version": "ai2.native_contact_qualification.v1",
            "qualified": True,
            "checks": {"original_cpu_exact": True, "cpu_cuda_exact": True, "learner_update_exact": True},
        }))
        return ["qualification.json"]

    def inventory_and_ack(self, path, name, files):
        m = self.subject()
        inventory = {
            "schema_version": "curriculum-inventory-v1",
            "block_id": name,
            "files": [
                {"path": file, "bytes": (Path(path) / file).stat().st_size, "sha256": m.sha256(Path(path) / file)}
                for file in files
            ],
        }
        (Path(path) / "inventory.json").write_text(json.dumps(inventory, sort_keys=True))
        (Path(path) / "preservation_ack.json").write_text(json.dumps({
            "verified": True,
            "block_id": name,
            "inventory_sha256": m.sha256(Path(path) / "inventory.json"),
        }))

    def remote_artifacts_for(self, root, name):
        path = Path(root) / name
        return {
            (name, "inventory.json"): (path / "inventory.json").read_bytes(),
            (name, "preservation_ack.json"): (path / "preservation_ack.json").read_bytes(),
        }


class FakeRemote:
    def __init__(self, *, finished=(), remote_artifacts=None):
        self.finished = set(finished)
        self.remote_artifacts = remote_artifacts or {}
        self.launches = []

    def job_receipt(self, name):
        if name in self.finished:
            return {"returncode": 0, "name": name}
        return None

    def job_started(self, name):
        return None

    def launch_job(self, name, command):
        self.launches.append((name, list(command)))
        self.finished.add(name)
        return {"pid": "1", "name": name}

    def read_artifact_file(self, name, filename):
        key = (name, filename)
        if key not in self.remote_artifacts:
            raise FileNotFoundError(f"missing {name}/{filename}")
        return self.remote_artifacts[key]

    def write_artifact_file(self, name, filename, data):
        self.remote_artifacts[(name, filename)] = data
        return {"bytes": len(data)}


if __name__ == "__main__":
    unittest.main()
