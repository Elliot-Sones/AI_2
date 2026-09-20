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
            self.assertIn(m.PLAN_SOURCE_DIGEST[:12], step.name)

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
            self.preserved_qualification(root, m.CONTACT_QUALIFICATION_BLOCK)
            train = m.build_initial_plan()[0]
            self.preserved_train(root, train.name, recipe=train.recipe, seed=train.seed, parent_sha=m.B40_SHA256)
            remote = FakeRemote(remote_artifacts=self.remote_artifacts_for(root, m.CONTACT_QUALIFICATION_BLOCK, include_files=True))
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
                "contact_deterministic_self_losses",
                "contact_deterministic_opponent_self_losses",
                "contact_deterministic_timeouts",
                "contact_deterministic_ambiguous",
                "contact_sampled_self_losses",
                "contact_sampled_opponent_self_losses",
                "contact_sampled_timeouts",
                "contact_sampled_ambiguous",
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
        failed_terminal = m.diagnostic_gate(self.metrics(contact_sampled_self_losses=1), baseline)
        self.assertFalse(failed_terminal.eligible)
        self.assertIn("contact_sampled_self_losses", failed_terminal.reason)

    def test_b40_baseline_uses_actual_attribution_counts_and_gate_rejects_regression(self):
        m = self.subject()
        baseline = m.B40_BASELINE
        self.assertEqual(baseline["panels"]["contact"]["deterministic"]["timeouts"], 1)
        self.assertEqual(baseline["panels"]["contact"]["sampled"]["ambiguous"], 2)
        candidate = self.metrics(
            contact_deterministic=19,
            contact_sampled=14,
            contact_deterministic_timeouts=1,
            contact_sampled_ambiguous=2,
            wide_deterministic_falls=7,
            wide_sampled_falls=7,
        )
        self.assertTrue(m.diagnostic_gate(candidate, baseline).eligible)
        timeout_regression = m.diagnostic_gate(
            self.metrics(
                contact_deterministic=19,
                contact_sampled=14,
                contact_deterministic_timeouts=2,
                contact_sampled_ambiguous=2,
                wide_deterministic_falls=7,
                wide_sampled_falls=7,
            ),
            baseline,
        )
        self.assertFalse(timeout_regression.eligible)
        self.assertIn("contact_deterministic_timeouts", timeout_regression.reason)
        ambiguous_regression = m.diagnostic_gate(
            self.metrics(
                contact_deterministic=19,
                contact_sampled=14,
                contact_deterministic_timeouts=1,
                contact_sampled_ambiguous=3,
                wide_deterministic_falls=7,
                wide_sampled_falls=7,
            ),
            baseline,
        )
        self.assertFalse(ambiguous_regression.eligible)
        self.assertIn("contact_sampled_ambiguous", ambiguous_regression.reason)

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
        self.assertIn("--resume", command)

    def test_remote_commands_use_the_configured_venv_python(self):
        m = self.subject()
        remote_python = Path("/workspace/run/venv/bin/python")
        remote = m.RemoteClient(
            remote_root=Path("/workspace/run"),
            remote_artifacts=Path("/workspace/run/artifacts"),
            remote_python=remote_python,
            deadline_unix=12345,
        )
        controller = m.Controller(
            local_root=Path("/unused"),
            remote=remote,
            deadline_unix=12345,
            remaining_cost=lambda: 5.0,
        )
        self.assertEqual(controller.train_command(m.build_initial_plan()[0])[0], str(remote_python))
        self.assertEqual(controller.qualification_command()[0], str(remote_python))

    def test_remote_launch_uses_configurable_deadline_job_wrapper(self):
        m = self.subject()
        command = m.build_bounded_background_launch_shell(
            remote_root=Path("/workspace/run"),
            remote_python=Path("/workspace/run/venv/bin/python"),
            deadline_unix=12345.5,
            name="safe-job",
            command=["/workspace/run/venv/bin/python", "-B", "train_retention_campaign_block.py"],
        )
        self.assertIn("source/run_campaign_job.py", command)
        self.assertIn("--root /workspace/run", command)
        self.assertIn("--deadline-unix 12345.5", command)
        self.assertIn("jobs/safe-job.controller-launch.log", command)

    def test_contact_qualification_command_uses_current_source_runner_and_new_receipt_block(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = m.Controller(
                local_root=root,
                remote=FakeRemote(),
                deadline_unix=1000,
                remaining_cost=lambda: 5.0,
                remote_artifacts=Path("/remote/artifacts"),
                frozen_root=Path("/remote/inputs/frozen"),
                reference_root=Path("/remote/inputs/reference"),
                contact_qualification_checkpoint=Path("/remote/inputs/calibration/final.zip"),
            )
            command = controller.qualification_command()
            qualification = controller.qualification
        values = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(command[:3], ["python3", "-B", "qualify_native_contact_training.py"])
        self.assertTrue(m.CONTACT_QUALIFICATION_BLOCK.startswith("contact-qualification-src"))
        self.assertEqual(Path(values["--output-dir"]), Path("/remote/artifacts") / m.CONTACT_QUALIFICATION_BLOCK)
        self.assertEqual(Path(values["--reference-root"]), Path("/remote/inputs/reference"))
        self.assertEqual(Path(values["--frozen-root"]), Path("/remote/inputs/frozen"))
        self.assertEqual(Path(values["--checkpoint"]), Path("/remote/inputs/calibration/final.zip"))
        self.assertEqual(Path(values["--output-dir"]) / "qualification.json", qualification)

    def test_custom_contact_qualification_block_is_used_for_command_preservation_and_training(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            block = "contact-qualification-425dbc61be7f"
            self.preserved_qualification(root, block)
            remote = FakeRemote(
                finished={block},
                remote_artifacts=self.remote_artifacts_for(root, block, include_files=True),
            )
            controller = m.Controller(
                local_root=root,
                remote=remote,
                deadline_unix=1000,
                now=StepClock([0, 0, 0]),
                remaining_cost=lambda: 5.0,
                remote_artifacts=Path("/remote/artifacts"),
                contact_qualification_block=block,
            )
            command = controller.qualification_command()
            train_command = controller.train_command(m.build_initial_plan()[0])
            preserved = controller.ensure_contact_qualification()
        qualification = Path("/remote/artifacts") / block / "qualification.json"
        values = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(Path(values["--output-dir"]), Path("/remote/artifacts") / block)
        self.assertEqual(Path(train_command[train_command.index("--qualification") + 1]), qualification)
        self.assertEqual(controller.qualification, qualification)
        self.assertEqual(preserved["local"]["name"], block)
        self.assertEqual(remote.launches, [])

    def test_missing_local_contact_qualification_is_retrieved_from_preserved_remote_block_without_launch(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            local = Path(directory) / "local"
            source.mkdir()
            local.mkdir()
            block = m.CONTACT_QUALIFICATION_BLOCK
            self.preserved_qualification(source, block)
            remote = FakeRemote(remote_artifacts=self.remote_artifacts_for(source, block, include_files=True))
            controller = m.Controller(
                local_root=local,
                remote=remote,
                deadline_unix=1000,
                now=StepClock([0, 0, 0]),
                remaining_cost=lambda: 5.0,
            )
            preserved = controller.ensure_contact_qualification()
            self.assertEqual(preserved["local"]["name"], block)
            self.assertTrue((local / block / "qualification.json").is_file())
            self.assertEqual(remote.launches, [])
            self.assertIn(
                {"time_unix": 0, "event": "retrieved_completed_contact_qualification", "name": block},
                controller.events,
            )

    def test_current_native_source_digest_changes_train_names_without_changing_checkpoint_lineage(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            block = "contact-qualification-425dbc61be7f"
            self.preserved_qualification(root, block)
            old = m.build_initial_plan(source_digest=m.B40_SHA256)[0]
            remote = FakeRemote(
                finished={block},
                failed={old.name: 1},
                remote_artifacts=self.remote_artifacts_for(root, block, include_files=True),
            )
            controller = m.Controller(
                local_root=root,
                remote=remote,
                deadline_unix=1000,
                now=StepClock([0, 1, 2, 3]),
                sleeper=lambda _: None,
                max_wait_seconds=1,
                remaining_cost=lambda: 5.0,
                contact_qualification_block=block,
            )
            with self.assertRaises(TimeoutError):
                controller.run(max_cycles=1)
        self.assertEqual(len(remote.launches), 1)
        launched_name = remote.launches[0][0]
        self.assertNotEqual(launched_name, old.name)
        self.assertIn(f"src{m.PLAN_SOURCE_DIGEST[:12]}", launched_name)
        command = remote.launches[0][1]
        self.assertEqual(command[command.index("--checkpoint-sha256") + 1], m.B40_SHA256)

    def test_full_campaign_run_preflights_current_contact_qualification_before_training(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            qualification_name = m.CONTACT_QUALIFICATION_BLOCK
            self.preserved_qualification(root, qualification_name)
            step = m.build_initial_plan()[0]
            remote = FakeRemote(finished={qualification_name}, remote_artifacts=self.remote_artifacts_for(root, qualification_name, include_files=True))
            controller = m.Controller(
                local_root=root,
                remote=remote,
                deadline_unix=1000,
                now=StepClock([0, 0, 0]),
                remaining_cost=lambda: 5.0,
            )
            with self.assertRaises(TimeoutError):
                controller.run([step])
            self.assertEqual(len(remote.launches), 1)
            self.assertNotEqual(remote.launches[0][0], qualification_name)
            self.assertIn(str(m.QUALIFICATION), remote.launches[0][1])
            self.assertTrue((root / qualification_name / "qualification.json").is_file())

    def test_custom_remote_artifact_root_derives_initial_b40_paths(self):
        m = self.subject()
        root = Path("/custom/remote/artifacts")
        controller = m.Controller(
            local_root=Path("/unused"),
            remote=FakeRemote(),
            remaining_cost=lambda: 5.0,
            remote_artifacts=root,
        )
        self.assertEqual(
            controller.initial_checkpoint,
            root / m.B40_BLOCK_NAME / "final.zip",
        )
        self.assertEqual(
            controller.initial_checkpoint_ack,
            root / m.B40_BLOCK_NAME / "preservation_ack.json",
        )

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
            controller = m.Controller(local_root=root, remote=remote, deadline_unix=1000, now=lambda: 0, remaining_cost=lambda: 5.0)
            summary = controller.apply_eval_gate(steps, b40_baseline=self.metrics())
        self.assertIn(steps[0].name, summary["stopped_lineages"])
        self.assertNotIn(steps[1].name, summary["stopped_lineages"])

    def test_pareto_winner_or_tradeoff_without_synthetic_comparison_step(self):
        m = self.subject()
        left = self.metrics(contact_sampled=16, wide_sampled_falls=1)
        right = self.metrics(contact_sampled=14, wide_sampled_falls=2)
        self.assertEqual(m.pareto_relation(left, right).winner, "left")
        tradeoff = m.pareto_relation(
            self.metrics(contact_sampled=16, nearby_sampled=19),
            self.metrics(contact_sampled=14, nearby_sampled=20),
        )
        self.assertIsNone(tradeoff.winner)
        self.assertEqual(tradeoff.relation, "tradeoff")
        self.assertEqual(tradeoff.matched_comparisons, ())

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

    def test_elapsed_budget_meter_subtracts_hourly_cost_before_each_external_action(self):
        m = self.subject()
        meter = m.ElapsedBudgetMeter(
            initial_remaining_dollars=5.00,
            hourly_cost_dollars=2.00,
            started_unix=1000,
            now=lambda: 1000 + 1800,
        )
        self.assertAlmostEqual(meter.remaining(), 4.00)
        controller = m.Controller(
            local_root=Path("/unused"),
            remote=FakeRemote(),
            now=lambda: 1000 + 2 * 3600,
            deadline_unix=1000 + 10 * 3600,
            remaining_cost=m.ElapsedBudgetMeter(
                initial_remaining_dollars=1.00,
                hourly_cost_dollars=0.20,
                started_unix=1000,
                now=lambda: 1000 + 2 * 3600,
            ).remaining,
        )
        with self.assertRaises(TimeoutError):
            controller.validate_budget()

    def test_cli_requires_hourly_cost_for_live_budget_meter(self):
        m = self.subject()
        with self.assertRaises(SystemExit):
            m.parse_args(["--deadline-unix", "123", "--remaining-budget-dollars", "5"])
        args = m.parse_args(["--deadline-unix", "123", "--remaining-budget-dollars", "5", "--hourly-cost-dollars", "0.8"])
        self.assertEqual(args.hourly_cost_dollars, 0.8)

    def test_remote_retrieval_rejects_inventory_paths_that_escape_local_block(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            name = "safe-block"
            self.preserved_train(root, name, recipe="R4C1N", seed=123, parent_sha=m.B40_SHA256)
            artifacts = self.remote_artifacts_for(root, name, include_files=True)
            inventory = json.loads(artifacts[(name, "inventory.json")])
            inventory["files"].append({"path": "../evil.txt", "bytes": 4, "sha256": m.bytes_sha256(b"evil")})
            artifacts[(name, "inventory.json")] = json.dumps(inventory, sort_keys=True).encode()
            artifacts[(name, "../evil.txt")] = b"evil"
            remote = FakeRemote(remote_artifacts=artifacts)
            controller = m.Controller(local_root=root / "download", remote=remote, remaining_cost=lambda: 5.0)
            with self.assertRaisesRegex(ValueError, "unsafe artifact path"):
                controller.retrieve_and_preserve_remote_block(name, require_final_zip=True)

    def test_remote_preservation_ack_upload_is_idempotent_when_matching_ack_already_exists(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            name = "safe-block"
            self.preserved_train(root, name, recipe="R4C1N", seed=123, parent_sha=m.B40_SHA256)
            artifacts = self.remote_artifacts_for(root, name, include_files=True)
            remote = FakeRemote(remote_artifacts=artifacts)
            controller = m.Controller(local_root=root / "download", remote=remote, remaining_cost=lambda: 5.0)
            controller.retrieve_and_preserve_remote_block(name, require_final_zip=True)
        self.assertEqual(remote.writes, [])

    def test_default_run_trains_evaluates_gates_and_continues_pareto_winner_once(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_qualification(root, m.CONTACT_QUALIFICATION_BLOCK)
            initial = m.build_initial_plan(source_digest=m.PLAN_SOURCE_DIGEST)
            for step in initial:
                final_sha = self.preserved_train(root, step.name, recipe=step.recipe, seed=step.seed, parent_sha=m.B40_SHA256)
                result = self.metrics(contact_deterministic=19, contact_sampled=14, wide_deterministic_falls=7, wide_sampled_falls=7)
                if step.recipe == "R4C1N":
                    result = self.metrics(contact_deterministic=20, contact_sampled=15, wide_deterministic_falls=6, wide_sampled_falls=6)
                self.preserved_eval(root, m.eval_step(step).name, train_name=step.name, result=result)
                if step.recipe == "R4C1N":
                    winner = replace(step, result_sha256=final_sha)
            continuation = m.build_continuation_step(winner, 2)
            self.preserved_train(root, continuation.name, recipe=continuation.recipe, seed=continuation.seed, parent_sha=continuation.parent_sha256)
            self.preserved_eval(root, m.eval_step(continuation).name, train_name=continuation.name, result=self.metrics(contact_deterministic=20, contact_sampled=16, wide_deterministic_falls=6, wide_sampled_falls=6))
            names = [step.name for step in initial]
            names += [m.eval_step(step).name for step in initial]
            names += [continuation.name, m.eval_step(continuation).name]
            remote_names = [m.CONTACT_QUALIFICATION_BLOCK, *names]
            remote = FakeRemote(
                finished=set(remote_names),
                remote_artifacts={key: value for name in remote_names for key, value in self.remote_artifacts_for(root, name, include_files=name == m.CONTACT_QUALIFICATION_BLOCK).items()},
            )
            controller = m.Controller(local_root=root, remote=remote, deadline_unix=1000, now=lambda: 0, remaining_cost=lambda: 5.0)
            summary = controller.run(max_cycles=2)
        self.assertEqual(summary["winner"], continuation.name)
        self.assertEqual(summary["selected_candidate"], initial[0].name)
        self.assertIn(continuation.name, [row["name"] for row in summary["completed"]])
        self.assertFalse(summary["baseline_provenance"]["regenerated_in_this_controller"])
        self.assertIn("B_projected B40", summary["baseline_provenance"]["source"])
        self.assertEqual(summary["status"], "complete")

    def test_failed_continuation_keeps_last_eligible_checkpoint_as_winner(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_qualification(root, m.CONTACT_QUALIFICATION_BLOCK)
            initial = m.build_initial_plan(source_digest=m.PLAN_SOURCE_DIGEST)
            winner = None
            for step in initial:
                final_sha = self.preserved_train(
                    root,
                    step.name,
                    recipe=step.recipe,
                    seed=step.seed,
                    parent_sha=m.B40_SHA256,
                )
                result = self.metrics(
                    contact_deterministic=19,
                    contact_sampled=14,
                    wide_deterministic_falls=7,
                    wide_sampled_falls=7,
                )
                if step.recipe == "R4C1N":
                    result = self.metrics(
                        contact_deterministic=20,
                        contact_sampled=15,
                        wide_deterministic_falls=6,
                        wide_sampled_falls=6,
                    )
                    winner = replace(step, result_sha256=final_sha)
                self.preserved_eval(root, m.eval_step(step).name, train_name=step.name, result=result)
            continuation = m.build_continuation_step(winner, 2)
            self.preserved_train(
                root,
                continuation.name,
                recipe=continuation.recipe,
                seed=continuation.seed,
                parent_sha=continuation.parent_sha256,
            )
            self.preserved_eval(
                root,
                m.eval_step(continuation).name,
                train_name=continuation.name,
                result=self.metrics(
                    nearby_sampled=18,
                    contact_deterministic=20,
                    contact_sampled=16,
                    wide_deterministic_falls=6,
                    wide_sampled_falls=6,
                ),
            )
            names = [step.name for step in initial]
            names += [m.eval_step(step).name for step in initial]
            names += [continuation.name, m.eval_step(continuation).name]
            remote_names = [m.CONTACT_QUALIFICATION_BLOCK, *names]
            remote = FakeRemote(
                finished=set(remote_names),
                remote_artifacts={
                    key: value
                    for name in remote_names
                    for key, value in self.remote_artifacts_for(root, name, include_files=name == m.CONTACT_QUALIFICATION_BLOCK).items()
                },
            )
            controller = m.Controller(
                local_root=root,
                remote=remote,
                deadline_unix=1000,
                now=lambda: 0,
                remaining_cost=lambda: 5.0,
            )
            summary = controller.run(max_cycles=2)

        self.assertEqual(summary["status"], "stopped_failed_gate")
        self.assertEqual(summary["winner"], winner.name)
        self.assertEqual(summary["failed_checkpoint"], continuation.name)
        self.assertEqual(summary["selected_candidate"], winner.name)

    def test_initial_pareto_tradeoff_runs_real_matched_cycle_then_continues_from_selected_cycle2_checkpoint(self):
        m = self.subject()

        class RecordingController(m.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.run_plans = []

            def _run_steps(self, plan):
                self.run_plans.append([step.name for step in plan])
                return super()._run_steps(plan)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_qualification(root, m.CONTACT_QUALIFICATION_BLOCK)
            initial = m.build_initial_plan(source_digest=m.PLAN_SOURCE_DIGEST)
            left_sha = self.preserved_train(root, initial[0].name, recipe=initial[0].recipe, seed=initial[0].seed, parent_sha=m.B40_SHA256)
            right_sha = self.preserved_train(root, initial[1].name, recipe=initial[1].recipe, seed=initial[1].seed, parent_sha=m.B40_SHA256)
            self.preserved_eval(
                root,
                m.eval_step(initial[0]).name,
                train_name=initial[0].name,
                result=self.metrics(contact_sampled=16, nearby_sampled=19, wide_deterministic_falls=7, wide_sampled_falls=7),
            )
            self.preserved_eval(
                root,
                m.eval_step(initial[1]).name,
                train_name=initial[1].name,
                result=self.metrics(contact_sampled=14, nearby_sampled=20, wide_deterministic_falls=7, wide_sampled_falls=7),
            )

            left_cycle2 = m.build_continuation_step(replace(initial[0], result_sha256=left_sha), 2)
            right_cycle2 = m.build_continuation_step(replace(initial[1], result_sha256=right_sha), 2)
            left_cycle2_sha = self.preserved_train(root, left_cycle2.name, recipe=left_cycle2.recipe, seed=left_cycle2.seed, parent_sha=left_cycle2.parent_sha256)
            right_cycle2_sha = self.preserved_train(root, right_cycle2.name, recipe=right_cycle2.recipe, seed=right_cycle2.seed, parent_sha=right_cycle2.parent_sha256)
            self.preserved_eval(
                root,
                m.eval_step(left_cycle2).name,
                train_name=left_cycle2.name,
                result=self.metrics(contact_sampled=17, nearby_sampled=20, wide_deterministic_falls=6, wide_sampled_falls=6),
            )
            self.preserved_eval(
                root,
                m.eval_step(right_cycle2).name,
                train_name=right_cycle2.name,
                result=self.metrics(contact_sampled=14, nearby_sampled=20, wide_deterministic_falls=7, wide_sampled_falls=7),
            )

            left_cycle3 = m.build_continuation_step(replace(left_cycle2, result_sha256=left_cycle2_sha), 3)
            self.preserved_train(root, left_cycle3.name, recipe=left_cycle3.recipe, seed=left_cycle3.seed, parent_sha=left_cycle2_sha)
            self.preserved_eval(
                root,
                m.eval_step(left_cycle3).name,
                train_name=left_cycle3.name,
                result=self.metrics(contact_sampled=17, nearby_sampled=20, wide_deterministic_falls=6, wide_sampled_falls=6),
            )

            names = [
                *(step.name for step in initial),
                *(m.eval_step(step).name for step in initial),
                left_cycle2.name,
                m.eval_step(left_cycle2).name,
                right_cycle2.name,
                m.eval_step(right_cycle2).name,
                left_cycle3.name,
                m.eval_step(left_cycle3).name,
            ]
            remote_names = [m.CONTACT_QUALIFICATION_BLOCK, *names]
            remote = FakeRemote(
                finished=set(remote_names),
                remote_artifacts={key: value for name in remote_names for key, value in self.remote_artifacts_for(root, name, include_files=name == m.CONTACT_QUALIFICATION_BLOCK).items()},
            )
            controller = RecordingController(local_root=root, remote=remote, deadline_unix=1000, now=lambda: 0, remaining_cost=lambda: 5.0)
            summary = controller.run(max_cycles=3)

        matched_cycle_names = {left_cycle2.name, m.eval_step(left_cycle2).name, right_cycle2.name, m.eval_step(right_cycle2).name}
        self.assertTrue(any(matched_cycle_names.issubset(set(plan)) for plan in controller.run_plans))
        completed_by_name = {row["name"]: row for row in summary["completed"]}
        self.assertIn(left_cycle3.name, completed_by_name)
        self.assertEqual(completed_by_name[left_cycle3.name]["resume_from"], left_cycle2.name)
        self.assertEqual(completed_by_name[left_cycle3.name]["parent_sha256"], left_cycle2_sha)
        self.assertEqual(summary["winner"], left_cycle3.name)
        self.assertEqual(summary["selected_candidate"], left_cycle2.name)
        self.assertEqual(summary["matched_comparison"]["winner"], left_cycle2.name)
        self.assertNotIn("needs_matched_comparison", summary["status"])
        self.assertEqual(summary["matched_comparison"]["relation"], "left_dominates")

    def test_remote_job_preservation_callback_runs_before_eval_command(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_qualification(root, m.CONTACT_QUALIFICATION_BLOCK)
            train = m.build_initial_plan()[0]
            evaluation = m.eval_step(train)
            remote = FakeRemote(
                finished={m.CONTACT_QUALIFICATION_BLOCK, train.name, evaluation.name},
                remote_artifacts=self.remote_artifacts_for(root, m.CONTACT_QUALIFICATION_BLOCK, include_files=True),
            )
            events = []

            def preserve(name, *, require_final_zip):
                events.append(("preserve", name, require_final_zip))
                if name == train.name:
                    self.preserved_train(root, train.name, recipe=train.recipe, seed=train.seed, parent_sha=m.B40_SHA256)
                elif name == evaluation.name:
                    self.assertIn(("preserve", train.name, True), events)
                    self.assertTrue((root / train.name / "preservation_ack.json").is_file())
                    self.preserved_eval(
                        root,
                        evaluation.name,
                        train_name=train.name,
                        result=self.metrics(contact_deterministic=20, contact_sampled=15, wide_deterministic_falls=6, wide_sampled_falls=6),
                    )
                remote.remote_artifacts.update(self.remote_artifacts_for(root, name))

            controller = m.Controller(
                local_root=root,
                remote=remote,
                deadline_unix=1000,
                now=lambda: 0,
                remaining_cost=lambda: 5.0,
                preserve_after_job=preserve,
            )
            summary = controller.run([train, evaluation])
        self.assertEqual(events, [("preserve", train.name, True), ("preserve", evaluation.name, False)])
        self.assertEqual([row["name"] for row in summary["completed"]], [train.name, evaluation.name])

    def test_failed_remote_job_preserves_inventory_block_and_job_receipts_before_reraising(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.preserved_qualification(root, m.CONTACT_QUALIFICATION_BLOCK)
            train = m.build_initial_plan()[0]
            failure_block = root / train.name
            failure_block.mkdir()
            (failure_block / "result.json").write_text(json.dumps({"status": "failed", "error": "cuda assert"}))
            (failure_block / "manifest.json").write_text(json.dumps({"config": {"seed": train.seed}}))
            self.inventory_and_ack(failure_block, train.name, ["result.json", "manifest.json"])
            remote = FakeRemote(
                finished={m.CONTACT_QUALIFICATION_BLOCK},
                failed={train.name: 7},
                remote_artifacts={
                    **self.remote_artifacts_for(root, m.CONTACT_QUALIFICATION_BLOCK, include_files=True),
                    **self.remote_artifacts_for(root, train.name, include_files=True),
                },
                job_files={
                    f"{train.name}.started.json": json.dumps({"pid": 123, "name": train.name}).encode(),
                    f"{train.name}.json": json.dumps({"returncode": 7, "name": train.name}).encode(),
                    f"{train.name}.log": b"traceback\ncuda assert\n",
                },
            )
            for child in failure_block.iterdir():
                child.unlink()
            failure_block.rmdir()
            controller = m.Controller(
                local_root=root,
                remote=remote,
                deadline_unix=1000,
                now=lambda: 0,
                remaining_cost=lambda: 5.0,
            )

            with self.assertRaisesRegex(RuntimeError, "returncode"):
                controller.run([train])

            self.assertEqual((root / train.name / "result.json").read_text(), json.dumps({"status": "failed", "error": "cuda assert"}))
            self.assertTrue((root / train.name / "failure_preservation.json").is_file())
            self.assertFalse((root / train.name / "preservation_ack.json").exists())
            self.assertEqual(json.loads((root / "jobs" / f"{train.name}.started.json").read_text())["pid"], 123)
            self.assertEqual(json.loads((root / "jobs" / f"{train.name}.json").read_text())["returncode"], 7)
            self.assertIn("cuda assert", (root / "jobs" / f"{train.name}.log").read_text())

    def test_timed_out_remote_job_preserves_available_failure_receipts_before_reraising(self):
        m = self.subject()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = m.build_initial_plan()[0]
            failure_block = root / train.name
            failure_block.mkdir()
            (failure_block / "result.json").write_text(json.dumps({"status": "running", "last_step": 12}))
            (failure_block / "manifest.json").write_text(json.dumps({"config": {"seed": train.seed}}))
            self.inventory_and_ack(failure_block, train.name, ["result.json", "manifest.json"])
            remote = FakeRemote(
                started={train.name},
                remote_artifacts=self.remote_artifacts_for(root, train.name, include_files=True),
                job_files={
                    f"{train.name}.started.json": json.dumps({"pid": 456, "name": train.name}).encode(),
                    f"{train.name}.log": b"still running\nlast step 12\n",
                },
            )
            for child in failure_block.iterdir():
                child.unlink()
            failure_block.rmdir()
            controller = m.Controller(
                local_root=root,
                remote=remote,
                deadline_unix=1000,
                now=StepClock([0, 0, 2]),
                sleeper=lambda _: None,
                max_wait_seconds=1,
                remaining_cost=lambda: 5.0,
            )

            with self.assertRaisesRegex(TimeoutError, "timed out waiting for job receipt"):
                controller.wait_for_job_rc0(train.name)

            self.assertEqual(json.loads((root / train.name / "result.json").read_text())["last_step"], 12)
            self.assertTrue((root / train.name / "failure_preservation.json").is_file())
            self.assertEqual(json.loads((root / "jobs" / f"{train.name}.started.json").read_text())["pid"], 456)
            self.assertFalse((root / "jobs" / f"{train.name}.json").exists())
            self.assertIn("last step 12", (root / "jobs" / f"{train.name}.log").read_text())

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
        contact_deterministic_opponent_self_losses=0,
        contact_sampled_opponent_self_losses=0,
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
                        "opponent_self_losses": contact_deterministic_opponent_self_losses,
                        "timeouts": contact_deterministic_timeouts,
                        "ambiguous": contact_deterministic_ambiguous,
                    },
                    "sampled": {
                        "episodes": contact_sampled_episodes,
                        "successes": contact_sampled,
                        "self_losses": contact_sampled_self_losses,
                        "opponent_self_losses": contact_sampled_opponent_self_losses,
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

    def preserved_qualification(self, root, name):
        m = self.subject()
        path = Path(root) / name
        path.mkdir()
        (path / "qualification.json").write_text(json.dumps({
            "schema_version": "ai2.native_contact_qualification.v1",
            "qualified": True,
            "checks": {"original_cpu_exact": True, "cpu_cuda_exact": True, "learner_update_exact": True},
            "evidence": [],
        }))
        self.inventory_and_ack(path, name, ["qualification.json"])

    def remote_artifacts_for(self, root, name, *, include_files=False):
        path = Path(root) / name
        artifacts = {
            (name, "inventory.json"): (path / "inventory.json").read_bytes(),
            (name, "preservation_ack.json"): (path / "preservation_ack.json").read_bytes(),
        }
        if include_files:
            inventory = json.loads((path / "inventory.json").read_text())
            for row in inventory["files"]:
                artifacts[(name, row["path"])] = (path / row["path"]).read_bytes()
        return artifacts


class FakeRemote:
    def __init__(self, *, started=(), finished=(), failed=None, remote_artifacts=None, job_files=None):
        self.started = set(started)
        self.finished = set(finished)
        self.failed = failed or {}
        self.remote_artifacts = remote_artifacts or {}
        self.job_files = job_files or {}
        self.launches = []
        self.events = []
        self.writes = []

    def job_receipt(self, name):
        if name in self.failed:
            return {"returncode": self.failed[name], "name": name}
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

    def read_job_file(self, filename):
        return self.job_files.get(filename)

    def write_artifact_file(self, name, filename, data):
        if (name, filename) in self.remote_artifacts:
            if self.remote_artifacts[(name, filename)] == data:
                return {"already_exists": True}
            raise FileExistsError(f"remote artifact differs: {name}/{filename}")
        self.remote_artifacts[(name, filename)] = data
        self.writes.append((name, filename, data))
        return {"bytes": len(data)}


class StepClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0) if self.values else 9999


if __name__ == "__main__":
    unittest.main()
