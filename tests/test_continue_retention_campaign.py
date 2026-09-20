import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class ContinueRetentionCampaignTests(unittest.TestCase):
    def subject(self):
        import continue_retention_campaign

        return continue_retention_campaign

    def test_initial_candidates_start_independently_from_exact_b40_and_name_the_recipe_seed_digest(self):
        m = self.subject()
        plan = m.build_initial_plan()
        self.assertEqual([step.recipe for step in plan], ["R4C1N", "R3C2N"])
        self.assertEqual({step.parent_sha256 for step in plan}, {m.B40_SHA256})
        self.assertEqual([step.resume_from for step in plan], [None, None])
        self.assertEqual([step.cumulative_added_transitions for step in plan], [m.CYCLE_TRANSITIONS, m.CYCLE_TRANSITIONS])
        self.assertNotEqual(plan[0].seed, plan[1].seed)
        for step in plan:
            self.assertIn(step.recipe, step.name)
            self.assertIn(str(step.cumulative_added_transitions), step.name)
            self.assertIn(f"s{step.seed}", step.name)
            self.assertIn(m.B40_SHA256[:12], step.name)

    def test_started_remote_job_is_adopted_not_duplicated(self):
        m = self.subject()
        step = m.build_initial_plan()[0]
        remote = FakeRemote(started={step.name})
        controller = m.Controller(
            local_root=Path("/unused"),
            remote=remote,
            now=StepClock([0, 1, 2, 3]),
            sleeper=lambda _: None,
            max_wait_seconds=1,
            deadline_unix=1000,
            remaining_cost=lambda: 5.0,
        )
        with self.assertRaises(TimeoutError):
            controller.ensure_job_started_or_finished(step, ["python3", "train_retention_campaign_block.py"])
        self.assertEqual(remote.launches, [])
        self.assertIn(("started", step.name), remote.events)

    def test_eval_waits_for_local_hash_verification_and_remote_preservation_ack(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = m.build_initial_plan()[0]
            self.preserved_train(root, train.name, recipe=train.recipe, seed=train.seed, parent_sha=m.B40_SHA256)
            remote = FakeRemote()
            controller = m.Controller(
                local_root=root,
                remote=remote,
                deadline_unix=1000,
                now=StepClock([0, 1, 2, 3]),
                sleeper=lambda _: None,
                max_wait_seconds=1,
                remaining_cost=lambda: 5.0,
            )
            with self.assertRaises(TimeoutError):
                controller.run([train, m.eval_step(train)])
        self.assertEqual(remote.launches, [])

    def test_eval_starts_after_local_and_remote_preservation_agree(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = m.build_initial_plan()[0]
            self.preserved_train(root, train.name, recipe=train.recipe, seed=train.seed, parent_sha=m.B40_SHA256)
            remote = FakeRemote(remote_artifacts=self.remote_artifacts_for(root, train.name))
            controller = m.Controller(
                local_root=root,
                remote=remote,
                deadline_unix=1000,
                now=StepClock([0, 0, 0]),
                remaining_cost=lambda: 5.0,
            )
            command = controller.eval_command(m.eval_step(train))
        self.assertIn("--checkpoint", command)
        self.assertIn(train.name, " ".join(command))

    def test_component_gate_uses_fields_separately_without_aggregate_score(self):
        m = self.subject()
        baseline = self.metrics(contact_sampled=16, wide_deterministic_falls=1, wide_sampled_falls=2)
        candidate = self.metrics(contact_sampled=16, wide_deterministic_falls=1, wide_sampled_falls=2)
        accepted = m.diagnostic_gate(candidate, baseline)
        self.assertTrue(accepted.eligible)
        self.assertEqual(
            set(accepted.components),
            {
                "contact_deterministic",
                "contact_sampled",
                "nearby_deterministic",
                "nearby_sampled",
                "wide_deterministic",
                "wide_sampled",
                "wide_deterministic_falls",
                "wide_sampled_falls",
                "terminal_attribution",
            },
        )
        self.assertNotIn("score", accepted.raw)
        failed = m.diagnostic_gate(self.metrics(nearby_sampled=18), baseline)
        self.assertFalse(failed.eligible)
        self.assertIn("nearby_sampled", failed.reason)
        with self.assertRaisesRegex(ValueError, "score"):
            m.diagnostic_gate({"score": 0.99, **candidate}, baseline)
        with self.assertRaisesRegex(ValueError, "20 episodes"):
            m.diagnostic_gate(self.metrics(contact_sampled_episodes=19), baseline)
        with self.assertRaisesRegex(ValueError, "terminal attribution"):
            m.diagnostic_gate(self.metrics(contact_sampled_self_losses=1), baseline)

    def test_train_command_roundtrips_through_actual_retention_runner_parser(self):
        m = self.subject()
        step = m.build_initial_plan()[0]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = root / "qualification.json"
            frozen = root / "frozen"
            reference = root / "reference"
            checkpoint = root / "b40.zip"
            ack = root / "b40-ack.json"
            controller = m.Controller(
                local_root=root,
                remote=FakeRemote(),
                deadline_unix=1000,
                remaining_cost=lambda: 5.0,
                qualification=qualification,
                frozen_root=frozen,
                reference_root=reference,
                initial_checkpoint=checkpoint,
                initial_checkpoint_ack=ack,
            )
            command = controller.train_command(step)
        self.assertIn("--schedule", command)
        self.assertNotIn("--recipe", command)
        values = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(values["--schedule"], step.recipe)
        self.assertEqual(values["--checkpoint-sha256"], m.B40_SHA256)
        self.assertEqual(Path(values["--checkpoint"]), checkpoint)
        self.assertEqual(Path(values["--checkpoint-ack"]), ack)
        self.assertEqual(Path(values["--qualification"]), qualification)
        self.assertEqual(Path(values["--frozen-root"]), frozen)
        self.assertEqual(Path(values["--reference-root"]), reference)
        self.assertEqual(values["--action-ab-arm"], "B_projected")

    def test_train_receipt_rejects_wrong_schedule_low_update_and_bad_lineage(self):
        m = self.subject()
        step = m.build_initial_plan()[0]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_train(root, step.name, recipe=step.recipe, seed=step.seed, parent_sha=m.B40_SHA256)
            preserved = m.verify_local_preservation(root, step.name, require_final_zip=True)
            controller = m.Controller(local_root=root, remote=FakeRemote(), remaining_cost=lambda: 5.0)
            self.assertTrue(controller.validate_train_receipt(step, preserved))
            preserved["result"]["schedule_id"] = "R3C2N"
            with self.assertRaisesRegex(ValueError, "schedule"):
                controller.validate_train_receipt(step, preserved)
            preserved = m.verify_local_preservation(root, step.name, require_final_zip=True)
            preserved["result"]["rollouts"][0]["actual_update"]["kl_early_stop"] = True
            with self.assertRaisesRegex(ValueError, "KL"):
                controller.validate_train_receipt(step, preserved)
            preserved = m.verify_local_preservation(root, step.name, require_final_zip=True)
            preserved["manifest"]["checkpoint"]["sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "lineage"):
                controller.validate_train_receipt(step, preserved)

    def test_ineligible_lineage_stops_independently(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            steps = m.build_initial_plan()
            for step in steps:
                self.preserved_train(root, step.name, recipe=step.recipe, seed=step.seed, parent_sha=m.B40_SHA256)
                self.preserved_eval(root, m.eval_step(step).name, train_name=step.name, result=self.metrics())
            bad_eval = root / m.eval_step(steps[0]).name / "result.json"
            bad_eval.write_text(json.dumps({"status": "evaluated_not_promoted", **self.metrics(nearby_sampled=18)}))
            self.inventory_and_ack(bad_eval.parent, m.eval_step(steps[0]).name, ["result.json", "manifest.json"])
            remote = FakeRemote(
                remote_artifacts={
                    **self.remote_artifacts_for(root, m.eval_step(steps[0]).name),
                    **self.remote_artifacts_for(root, m.eval_step(steps[1]).name),
                }
            )
            controller = m.Controller(local_root=root, remote=remote, deadline_unix=1000, remaining_cost=lambda: 5.0)
            summary = controller.apply_eval_gate(steps, b40_baseline=self.metrics())
        self.assertIn(steps[0].name, summary["stopped_lineages"])
        self.assertNotIn(steps[1].name, summary["stopped_lineages"])

    def test_pareto_winner_or_exactly_one_matched_comparison_on_tradeoff(self):
        m = self.subject()
        left = self.metrics(contact_sampled=16, wide_sampled_falls=1)
        right = self.metrics(contact_sampled=14, wide_sampled_falls=2)
        self.assertEqual(m.pareto_relation(left, right).winner, "left")
        tradeoff = m.pareto_relation(
            self.metrics(contact_sampled=16, nearby_sampled=19),
            self.metrics(contact_sampled=14, nearby_sampled=20),
        )
        self.assertIsNone(tradeoff.winner)
        self.assertEqual(len(tradeoff.matched_comparisons), 1)
        self.assertIn("matched-comparison", tradeoff.matched_comparisons[0].name)

    def test_winner_continues_immutable_five_rollout_cycles_toward_50m(self):
        m = self.subject()
        winner = replace(m.build_initial_plan()[0], result_sha256="a" * 64)
        cycle2 = m.build_continuation_step(winner, 2)
        cycle3 = m.build_continuation_step(replace(cycle2, result_sha256="b" * 64), 3)
        self.assertEqual(cycle2.rollouts, 5)
        self.assertEqual(cycle2.resume_from, winner.name)
        self.assertEqual(cycle2.parent_sha256, winner.result_sha256)
        self.assertEqual(cycle2.cumulative_added_transitions, 2 * m.CYCLE_TRANSITIONS)
        self.assertIn("cycle02", cycle2.name)
        self.assertIn(str(2 * m.CYCLE_TRANSITIONS), cycle2.name)
        self.assertEqual(cycle3.cumulative_added_transitions, 3 * m.CYCLE_TRANSITIONS)
        self.assertLessEqual(cycle3.cumulative_added_transitions, m.TARGET_ADDED_TRANSITIONS)

    def test_two_consecutive_no_improvement_cycles_stop_continuation(self):
        m = self.subject()
        previous = self.metrics(contact_sampled=14)
        same = self.metrics(contact_sampled=14)
        better = self.metrics(contact_sampled=15)
        state = m.ContinuationState()
        self.assertFalse(state.update(previous, same).stop)
        self.assertTrue(state.update(previous, same).stop)
        self.assertFalse(m.ContinuationState().update(previous, better).stop)

    def test_deadline_and_remaining_cost_reserve_fail_closed(self):
        m = self.subject()
        controller = m.Controller(
            local_root=Path("/unused"),
            remote=FakeRemote(),
            now=lambda: 950,
            deadline_unix=1000,
            remaining_cost=lambda: m.COST_RESERVE_DOLLARS - 0.01,
        )
        with self.assertRaises(TimeoutError):
            controller.validate_budget()

    def metrics(
        self,
        *,
        contact_deterministic=18,
        contact_sampled=14,
        nearby_deterministic=19,
        nearby_sampled=19,
        wide_deterministic=13,
        wide_sampled=13,
        nearby_deterministic_falls=1,
        nearby_sampled_falls=1,
        wide_deterministic_falls=1,
        wide_sampled_falls=1,
        contact_deterministic_episodes=20,
        contact_sampled_episodes=20,
        nearby_deterministic_episodes=20,
        nearby_sampled_episodes=20,
        wide_deterministic_episodes=20,
        wide_sampled_episodes=20,
        contact_deterministic_self_losses=0,
        contact_sampled_self_losses=0,
        contact_deterministic_timeouts=0,
        contact_sampled_timeouts=0,
        contact_deterministic_ambiguous=0,
        contact_sampled_ambiguous=0,
    ):
        return {
            "schema_version": "ai2.retention_evaluation.v1",
            "status": "evaluated_not_promoted",
            "panels": {
                "contact": {
                    "deterministic": {
                        "episodes": contact_deterministic_episodes,
                        "successes": contact_deterministic,
                        "self_losses": contact_deterministic_self_losses,
                        "opponent_self_losses": 0,
                        "timeouts": contact_deterministic_timeouts,
                        "ambiguous": contact_deterministic_ambiguous,
                    },
                    "sampled": {
                        "episodes": contact_sampled_episodes,
                        "successes": contact_sampled,
                        "self_losses": contact_sampled_self_losses,
                        "opponent_self_losses": 0,
                        "timeouts": contact_sampled_timeouts,
                        "ambiguous": contact_sampled_ambiguous,
                    },
                },
                "nearby": {
                    "deterministic": {"episodes": nearby_deterministic_episodes, "successes": nearby_deterministic, "falls": nearby_deterministic_falls},
                    "sampled": {"episodes": nearby_sampled_episodes, "successes": nearby_sampled, "falls": nearby_sampled_falls},
                },
                "wide": {
                    "deterministic": {"episodes": wide_deterministic_episodes, "successes": wide_deterministic, "falls": wide_deterministic_falls},
                    "sampled": {"episodes": wide_sampled_episodes, "successes": wide_sampled, "falls": wide_sampled_falls},
                },
            },
        }

    def preserved_train(self, root, name, *, recipe, seed, parent_sha):
        m = self.subject()
        path = Path(root) / name
        path.mkdir()
        (path / "final.zip").write_bytes((name + "-checkpoint").encode())
        final_sha = m.sha256(path / "final.zip")
        (path / "manifest.json").write_text(json.dumps({
            "config": {"schedule_id": recipe, "recipe": recipe, "seed": seed, "rollouts": 5, "action_ab_arm": "B_projected"},
            "schedule_id": recipe,
            "action_ab_arm": "B_projected",
            "schedule": ["combat", "combat", "combat", "combat", "navigation"] if recipe == "R4C1N" else ["combat", "combat", "combat", "navigation", "navigation"],
            "task_transition_totals": {"combat": 4 * m.TRANSITIONS_PER_ROLLOUT, "navigation": m.TRANSITIONS_PER_ROLLOUT},
            "checkpoint": {"sha256": parent_sha},
        }))
        (path / "result.json").write_text(json.dumps({
            "schema_version": "ai2.retention_campaign_block.v1",
            "status": "trained_not_evaluated",
            "schedule_id": recipe,
            "schedule": ["combat", "combat", "combat", "combat", "navigation"] if recipe == "R4C1N" else ["combat", "combat", "combat", "navigation", "navigation"],
            "checkpoint_sha256": final_sha,
            "summary": {"measured_transitions": m.CYCLE_TRANSITIONS},
            "rollouts": [
                {
                    "rollout": index,
                    "task": task,
                    "transitions": m.TRANSITIONS_PER_ROLLOUT,
                    "game_ticks": m.TRANSITIONS_PER_ROLLOUT,
                    "collection_seconds": 1.0,
                    "update_seconds": 1.0,
                    "total_seconds": 2.0,
                    "metrics": {"loss": 0.1},
                    "actual_update": {
                        "attempted_epochs": 4,
                        "optimizer_steps": 512,
                        "nominal_optimizer_steps": 512,
                        "kl_early_stop": False,
                        "low_update": False,
                    },
                    "cumulative_timesteps": 10_059_776 + (index + 1) * m.TRANSITIONS_PER_ROLLOUT,
                }
                for index, task in enumerate(["combat", "combat", "combat", "combat", "navigation"] if recipe == "R4C1N" else ["combat", "combat", "combat", "navigation", "navigation"])
            ],
            "task_transition_totals": {"combat": 4 * m.TRANSITIONS_PER_ROLLOUT, "navigation": m.TRANSITIONS_PER_ROLLOUT},
            "final_model_counters": {"num_timesteps": 10_059_776 + m.CYCLE_TRANSITIONS},
            "parent_checkpoint_sha256": parent_sha,
        }))
        self.inventory_and_ack(path, name, ["final.zip", "result.json", "manifest.json"])
        return final_sha

    def preserved_eval(self, root, name, *, train_name, result):
        m = self.subject()
        path = Path(root) / name
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({
            "candidate": {"checkpoint_sha256": m.sha256(Path(root) / train_name / "final.zip")},
        }))
        result = {"status": "evaluated_not_promoted", **result}
        (path / "result.json").write_text(json.dumps(result))
        self.inventory_and_ack(path, name, ["result.json", "manifest.json"])

    def inventory_and_ack(self, path, name, files):
        m = self.subject()
        inventory = {
            "schema_version": "curriculum-inventory-v1",
            "block_id": name,
            "files": [
                {"path": file, "bytes": (path / file).stat().st_size, "sha256": m.sha256(path / file)}
                for file in files
            ],
        }
        (path / "inventory.json").write_text(json.dumps(inventory, sort_keys=True))
        (path / "preservation_ack.json").write_text(json.dumps({
            "verified": True,
            "block_id": name,
            "inventory_sha256": m.sha256(path / "inventory.json"),
        }))

    def remote_artifacts_for(self, root, name):
        path = Path(root) / name
        return {
            (name, "inventory.json"): (path / "inventory.json").read_bytes(),
            (name, "preservation_ack.json"): (path / "preservation_ack.json").read_bytes(),
        }


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


class StepClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0) if self.values else 9999


if __name__ == "__main__":
    unittest.main()
