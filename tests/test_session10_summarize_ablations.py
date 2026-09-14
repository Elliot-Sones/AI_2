"""Synthetic artifact tests: no Torch, GPU, subprocess, or original source required."""
from __future__ import annotations

import copy
from contextlib import redirect_stderr
import importlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("optimizer_cpu_steps", "deferred_metrics", "deferred_timeouts",
         "boundary_validation", "compile_loss", "compile_gae")


class SummarizeAblationsTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "session10_summarize_ablations.py").exists(),
                        "standalone ablation artifact summarizer must exist")
        self.module = importlib.import_module("session10_summarize_ablations")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def change(self, path, update):
        value = json.loads(path.read_text())
        update(value)
        self.write(path, value)

    def suite(self, name="paired", *, mode="paired", seeds=(45, 46, 47),
              sizes=(256,), ratios=(1.1, 1.2, 1.3), arms=("baseline", "counters")):
        suite = self.root / name
        files = {key: {"exists": True, "path": f"/remote/{key}", "size": 12, "sha256": "a" * 64}
                 for key in ("checkpoint", "config", "train_job", "tensor_ppo.py", "source/train.py")}
        entries, records = [], []
        for seed_index, seed in enumerate(seeds):
            for n_envs in sizes:
                for arm in arms:
                    job_name = f"native-ppo-{mode}-env{n_envs}-seed{seed}-{arm}"
                    output = suite / job_name
                    job = {"name": job_name, "arm": arm, "seed": seed,
                           "output": f"/remote/archive/{job_name}", "n_envs": n_envs,
                           "n_steps": 128, "batch_size": 8192, "epochs": 8,
                           "warmup": 2, "cycles": 3, "group_size": 1, "resident_buffer": False}
                    flags = {flag: False for flag in FLAGS}
                    if arm == "counters":
                        flags["optimizer_cpu_steps"] = True
                    elif arm == "compiled_gae":
                        flags["compile_gae"] = True
                    entries.append({"job": job, "optimizations": flags})
                    scale = 1.0 if arm == "baseline" else 1 / ratios[seed_index % len(ratios)]
                    transition_count = n_envs * job["n_steps"]
                    rows = []
                    for i in range(1, 6):
                        seconds = 100.0 if i <= 2 else (i - 1) * scale
                        rows.append({"cycle_index": i, "phase": "warmup" if i <= 2 else "measured",
                                     "measured_cycle": max(0, i - 2), "rollout_completed": True,
                                     "transitions": transition_count, "num_timesteps": 16000000 + i * transition_count,
                                     "cycle_seconds": seconds, "rollout_seconds": .6 * seconds,
                                     "update_seconds": .3 * seconds, "boundary_seconds": .1 * seconds,
                                     "transitions_per_second": transition_count / seconds})
                    measured = rows[2:]
                    total = sum(row["cycle_seconds"] for row in measured)
                    plan = {key: job[key] for key in ("n_envs", "n_steps", "batch_size", "epochs", "warmup", "cycles")}
                    plan.update(transitions_per_cycle=transition_count,
                                steps=5 * transition_count,
                                minibatches_per_epoch=math.ceil(transition_count / job["batch_size"]),
                                minibatches_per_cycle=math.ceil(transition_count / job["batch_size"]) * 8)
                    summary = {"status": "completed", "mode": "baseline", "plan": plan,
                               "completed_cycles_total": 5, "completed_measured_cycles": 3,
                               "requested_measured_cycles": 3, "warmup_cycles": 2,
                               "measured_transitions": 3 * transition_count,
                               "measured_cycle_seconds": total, "measured_transitions_per_second": 3 * transition_count / total,
                               "measured_collect_seconds": sum(row["rollout_seconds"] for row in measured),
                               "measured_update_seconds": sum(row["update_seconds"] for row in measured),
                               "measured_boundary_seconds": sum(row["boundary_seconds"] for row in measured),
                               "startup_seconds": 2.0, "learning_wall_seconds": 200 + total}
                    self.write(output / "job_manifest.json", {"job": job, "files": files, "optimizations": flags,
                                                              "policy_device": "cuda"})
                    self.write(output / "campaign_summary.json", summary)
                    self.write(output / "native_lane_provenance.json", {
                        "schema": "ai2.native_ppo_ablations.v1", "lane_device": "cuda", "policy_device": "cuda:0",
                        "optimizations": flags})
                    history = [{"stats": {"unique_graphs": 1}}] * 5
                    self.write(output / "optimization_runtime.json", {
                        "compile_headline_eligible": True, "measured_recompile_cycles": [],
                        "compiler_history_per_update": [history]})
                    (output / "campaign_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                    records.append({"name": job_name, "arm": arm, "seed": seed, "n_envs": n_envs,
                                    "status": "completed", "return_code": 0, "error": None,
                                    "started_unix": 1000.0, "ended_unix": 1210.0 + total})
        self.write(suite / "source_manifest.json", files)
        self.write(suite / "native_ppo_ablations_plan.json", {
            "schema": "ai2.native_ppo_ablations.plan.v1", "mode": mode, "dry_run": False,
            "policy_device": "cuda", "lane_device": "cuda", "jobs": entries})
        self.write(suite / "suite_report.json", {"mode": mode, "status": "completed", "jobs": records,
                                                "planned_jobs": len(entries), "completed_jobs": len(records)})
        return suite

    def output(self, suite, arm="counters", seed=45, size=256):
        plan = json.loads((suite / "native_ppo_ablations_plan.json").read_text())
        job = next(entry["job"] for entry in plan["jobs"]
                   if entry["job"]["arm"] == arm and entry["job"]["seed"] == seed and entry["job"]["n_envs"] == size)
        return suite / job["name"]

    def test_completed_timings_exclude_warmup_and_use_ratio_of_sums(self):
        suite = self.suite()
        result = self.module.summarize_suites([suite])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["findings"], [])
        job = result["jobs"][0]
        self.assertTrue(job["eligible"])
        timing = job["timings"]
        self.assertEqual(timing["warmup_seconds"], 200)
        self.assertEqual(timing["full_job_wall_seconds"], 219)
        self.assertEqual(timing["measured_cycle_seconds"], 9)
        self.assertEqual(timing["cycle_seconds_mean"], 3)
        self.assertAlmostEqual(timing["collect_seconds_mean"], 1.8)
        self.assertAlmostEqual(timing["update_seconds_mean"], .9)
        self.assertAlmostEqual(timing["boundary_seconds_mean"], .3)
        self.assertAlmostEqual(timing["sps"], 3 * 32768 / 9)
        self.assertNotEqual(timing["sps"], sum(32768 / i for i in (2, 3, 4)) / 3)
        self.assertTrue(result["input_artifacts"])

    def test_paired_geomean_median_range_and_unique_seed_count(self):
        result = self.module.summarize_suites([self.suite()])
        group, = result["comparisons"]
        self.assertEqual(group["count_seeds"], 3)
        self.assertEqual(group["seeds"], [45, 46, 47])
        self.assertTrue(group["claim_eligible"])
        self.assertAlmostEqual(group["ratio_geomean"], (1.1 * 1.2 * 1.3) ** (1 / 3))
        self.assertAlmostEqual(group["ratio_median"], 1.2)
        self.assertAlmostEqual(group["ratio_min"], 1.1)
        self.assertAlmostEqual(group["ratio_max"], 1.3)

    def test_modes_suites_and_sizes_never_pool(self):
        paired = self.suite(sizes=(256, 4096))
        screen = self.suite("screen", mode="screen", seeds=(45,), sizes=(256, 4096))
        retry = self.suite("retry", seeds=(45,))
        result = self.module.summarize_suites([paired, screen, retry])
        self.assertEqual(len(result["comparisons"]), 5)
        for group in result["comparisons"]:
            self.assertEqual(group["claim_eligible"], group["suite"] == str(paired.resolve()))
            if group["mode"] == "screen":
                self.assertEqual(group["interpretation"], "screen_descriptive")

    def test_missing_partial_and_unstarted_jobs_remain_findings(self):
        suite = self.suite()
        self.output(suite).joinpath("campaign_summary.json").unlink()
        self.change(self.output(suite, seed=46) / "campaign_summary.json", lambda value: value.update(status="incomplete"))
        self.change(suite / "suite_report.json", lambda value: (value.update(status="aborted", completed_jobs=5), value["jobs"].pop()))
        result = self.module.summarize_suites([suite])
        self.assertEqual(result["status"], "findings")
        self.assertEqual(len(result["jobs"]), 6)
        self.assertEqual(sum(job["eligible"] for job in result["jobs"]), 3)
        self.assertGreaterEqual(len(result["findings"]), 3)
        self.assertTrue(all(not group["claim_eligible"] for group in result["comparisons"]))

    def test_malformed_jsonl_is_not_silently_skipped(self):
        suite = self.suite(seeds=(45,))
        metrics = self.output(suite) / "campaign_metrics.jsonl"
        metrics.write_text(metrics.read_text() + "{bad-json}\n")
        result = self.module.summarize_suites([suite])
        self.assertFalse(result["jobs"][1]["eligible"])
        self.assertIn("campaign_metrics.jsonl", str(result["findings"]))

    def test_nonfinite_invalid_counts_and_residuals_rejected(self):
        for field, value in (("measured_cycle_seconds", float("nan")),
                             ("completed_measured_cycles", 2), ("measured_transitions", 7),
                             ("measured_boundary_seconds", -1), ("measured_update_seconds", 1000)):
            with self.subTest(field=field):
                suite = self.suite(field, seeds=(45,))
                self.change(self.output(suite) / "campaign_summary.json", lambda payload: payload.update({field: value}))
                result = self.module.summarize_suites([suite])
                self.assertFalse(result["jobs"][1]["eligible"])

    def test_metrics_require_exact_warmup_indices_and_positive_finite_spans(self):
        for corruption in ("phase", "index", "negative", "infinite", "residual", "timesteps"):
            with self.subTest(corruption=corruption):
                suite = self.suite(corruption, seeds=(45,))
                path = self.output(suite) / "campaign_metrics.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if corruption == "phase": rows[2]["phase"] = "warmup"
                if corruption == "index": rows[2]["cycle_index"] = 4
                if corruption == "negative": rows[2]["rollout_seconds"] = -1
                if corruption == "infinite": rows[2]["update_seconds"] = float("inf")
                if corruption == "residual": rows[2]["boundary_seconds"] += .2
                if corruption == "timesteps": rows[2]["num_timesteps"] += 1
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                self.assertFalse(self.module.summarize_suites([suite])["jobs"][1]["eligible"])

    def test_flags_source_and_geometry_must_match_manifest(self):
        for corruption in ("flags", "source", "geometry", "missing_hash", "cuda", "job_identity"):
            with self.subTest(corruption=corruption):
                suite = self.suite(corruption, seeds=(45,))
                output = self.output(suite)
                if corruption == "flags":
                    self.change(output / "native_lane_provenance.json", lambda value: value["optimizations"].update(deferred_metrics=True))
                if corruption == "source":
                    self.change(output / "job_manifest.json", lambda value: value["files"]["config"].update(sha256="b" * 64))
                if corruption == "geometry":
                    self.change(output / "campaign_summary.json", lambda value: value["plan"].update(n_envs=4096))
                if corruption == "missing_hash":
                    self.change(output / "job_manifest.json", lambda value: value["files"].pop("checkpoint"))
                if corruption == "cuda":
                    self.change(output / "native_lane_provenance.json", lambda value: value.update(policy_device="cpu"))
                if corruption == "job_identity":
                    self.change(output / "job_manifest.json", lambda value: value["job"].update(seed=99))
                self.assertFalse(self.module.summarize_suites([suite])["jobs"][1]["eligible"])

    def test_compile_eligibility_requires_complete_unchanged_measured_history(self):
        for corruption in (None, "false", "recompiled", "short", "lying_guard"):
            with self.subTest(corruption=corruption):
                suite = self.suite(str(corruption), seeds=(45,), arms=("baseline", "compiled_gae"))
                path = self.output(suite, "compiled_gae") / "optimization_runtime.json"
                runtime = json.loads(path.read_text())
                if corruption == "false": runtime["compile_headline_eligible"] = False
                if corruption == "recompiled": runtime["measured_recompile_cycles"] = [2]
                if corruption == "short": runtime["compiler_history_per_update"][0].pop()
                if corruption == "lying_guard": runtime["compiler_history_per_update"][0][2] = {"stats": {"unique_graphs": 2}}
                self.write(path, runtime)
                job = self.module.summarize_suites([suite])["jobs"][1]
                self.assertEqual(job["eligible"], corruption is None)
                self.assertEqual(job["compile_required"], True)
                self.assertEqual(job["compile_headline_eligible"], corruption is None)

    def test_nested_malformed_records_cannot_abort_other_jobs_or_suites(self):
        for corruption in ("execution", "arm", "geometry", "seed", "return_code", "flags"):
            with self.subTest(corruption=corruption):
                suite = self.suite(corruption, seeds=(45,))
                if corruption == "execution":
                    self.change(suite / "suite_report.json", lambda value: value.update(jobs=[7]))
                elif corruption == "return_code":
                    self.change(suite / "suite_report.json", lambda value: value["jobs"][1].update(return_code=False))
                elif corruption == "flags":
                    self.change(suite / "native_ppo_ablations_plan.json", lambda value: value["jobs"][1]["optimizations"].update(optimizer_cpu_steps=1))
                else:
                    key = {"arm": "arm", "geometry": "n_envs", "seed": "seed"}[corruption]
                    self.change(suite / "native_ppo_ablations_plan.json", lambda value: value["jobs"][1]["job"].update({key: []}))
                good = self.suite("good-" + corruption, seeds=(45,))
                result = self.module.summarize_suites([suite, good])
                self.assertEqual(len(result["jobs"]), 4)
                self.assertFalse(result["jobs"][1]["eligible"])
                self.assertTrue(all(job["eligible"] for job in result["jobs"][2:]))

    def test_three_seed_four_arm_two_size_campaign_has_six_comparisons(self):
        suite = self.suite(arms=("baseline", "counters", "validation", "compiled_gae"), sizes=(256, 4096))
        result = self.module.summarize_suites([suite])
        self.assertEqual(len(result["jobs"]), 24)
        self.assertEqual(len(result["comparisons"]), 6)
        self.assertTrue(all(job["eligible"] for job in result["jobs"]))
        self.assertTrue(all(group["count_seeds"] == 3 and group["claim_eligible"] for group in result["comparisons"]))

    def test_cli_writes_findings_for_negative_inputs(self):
        output = self.root / "negative.json"
        self.assertEqual(self.module.main(["--suite", str(self.root / "missing"), "--output", str(output)]), 1)
        self.assertEqual(json.loads(output.read_text())["status"], "findings")

    def test_duplicate_json_keys_are_rejected_instead_of_last_value_winning(self):
        suite = self.suite(seeds=(45,))
        path = self.output(suite) / "campaign_summary.json"
        path.write_text(path.read_text().replace('"status": "completed"', '"status": "failed", "status": "completed"'))
        result = self.module.summarize_suites([suite])
        self.assertFalse(result["jobs"][1]["eligible"])
        self.assertIn("duplicate JSON key", str(result["findings"]))

    def test_duplicate_seed_arm_and_duplicate_suite_do_not_inflate_replication(self):
        suite = self.suite(seeds=(45,))
        result = self.module.summarize_suites([suite, suite])
        self.assertEqual(len(result["jobs"]), 2)
        self.assertEqual(result["comparisons"][0]["count_seeds"], 1)
        self.assertIn("duplicate suite", str(result["findings"]))
        self.change(suite / "native_ppo_ablations_plan.json", lambda value: value["jobs"].append(copy.deepcopy(value["jobs"][1])))
        result = self.module.summarize_suites([suite])
        self.assertEqual(len(result["jobs"]), 3)
        self.assertTrue(all(not job["eligible"] for job in result["jobs"] if job.get("arm") == "counters"))

    def test_valid_but_unequal_workload_is_not_paired(self):
        suite = self.suite(seeds=(45,))
        output = self.output(suite)
        for filename in ("native_ppo_ablations_plan.json",):
            self.change(suite / filename, lambda value: value["jobs"][1]["job"].update(epochs=4))
        self.change(output / "job_manifest.json", lambda value: value["job"].update(epochs=4))
        self.change(output / "campaign_summary.json", lambda value: value["plan"].update(epochs=4, minibatches_per_cycle=16))
        result = self.module.summarize_suites([suite])
        self.assertTrue(all(job["eligible"] for job in result["jobs"]))
        self.assertIn("unequal workload", str(result["findings"]))
        self.assertTrue(all(group["count_seeds"] == 0 for group in result["comparisons"]))

    def test_missing_or_malformed_suite_remains_a_negative_finding(self):
        missing = self.root / "missing"
        malformed = self.root / "malformed"
        malformed.mkdir()
        (malformed / "native_ppo_ablations_plan.json").write_text("[]")
        result = self.module.summarize_suites([missing, malformed])
        self.assertEqual(result["status"], "findings")
        self.assertEqual(len(result["suites"]), 2)
        self.assertGreaterEqual(len(result["findings"]), 2)

    def test_cli_is_reproducible_fresh_and_preserves_input_bytes(self):
        suite = self.suite(seeds=(45,))
        before = {str(path): path.read_bytes() for path in suite.rglob("*") if path.is_file()}
        first, second = self.root / "one.json", self.root / "two.json"
        for path in (first, second):
            self.assertEqual(self.module.main(["--suite", str(suite), "--output", str(path)]), 0)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(before, {str(path): path.read_bytes() for path in suite.rglob("*") if path.is_file()})
        with redirect_stderr(io.StringIO()) as error:
            self.assertEqual(self.module.main(["--suite", str(suite), "--output", str(first)]), 2)
        self.assertIn("already exists", error.getvalue())
        self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
