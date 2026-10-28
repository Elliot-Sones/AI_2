"""Tiny saved-artifact fixtures; no GPU, training, network, or third-party imports."""

from ai2.paths import source_path
import builtins
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE = source_path(Path(__file__).resolve().parents[1], "summarize_native_capacity.py")
ARCHES = [(512, 512, 256), (1024, 1024, 1024), (1536, 1536, 1536)]
PARAMS = [1053973, 4736021, 10249749]
PANELS = [f"{phase}-{mode}" for phase in ("0a", "0b") for mode in ("deterministic", "sampled")]


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), "summarizer must exist")
        spec = importlib.util.spec_from_file_location("capacity_summary_test", MODULE)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.make_round()

    def put(self, path, obj):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(obj))

    def edit(self, path, callback):
        value = json.loads((self.root / path).read_text())
        callback(value)
        self.put(path, value)

    def make_round(self):
        inputs, records, next_inputs = [], [], []
        for arch, parameters in zip(ARCHES, PARAMS):
            for seed in (145, 146, 147):
                label = f"arch-{'-'.join(map(str, arch))}_seed{seed}"
                row = dict(label=label, net_arch=list(arch), seed=seed, parameters=parameters, steps=32768, updates=8)
                inputs.append(row)
                next_inputs.append({**row, "steps": 65536, "updates": 16, "checkpoint_sha256": "a" * 64})
                start = 1000 + len(inputs) * 100
                for kind in ("train", "eval"):
                    records.append(dict(kind=kind, label=f"{label}-{kind}", status="completed", started_unix=start, ended_unix=start + 20))
                self.put(f"{label}/train/result.json", dict(status="completed", error=None, starting_steps=32768, ending_steps=65536, actual_steps=32768, wall_seconds=18))
                self.put(f"{label}/train/campaign_summary.json", dict(status="completed", error=None, completed_measured_cycles=1, completed_cycles_total=1, warmup_cycles=0, measured_transitions=32768, measured_transitions_per_second=2048, measured_cycle_seconds=16, measured_collect_seconds=10, measured_update_seconds=5, measured_boundary_seconds=1, plan=dict(n_envs=64, n_steps=512, batch_size=8192, epochs=8, warmup=0)))
                self.put(f"{label}/train/native_learning_receipt.json", dict(status="completed", error=None, final_completed_update_verified=True, starting_steps=32768, expected_final_steps=65536, expected_final_updates=16, config=dict(net_arch=list(arch), parameter_count=parameters), runtime_policy=dict(all_policy_parameters_cuda=True, parameter_count=parameters), initial_checkpoint_counters=dict(num_timesteps=32768, _n_updates=8), final_checkpoint_counters=dict(num_timesteps=65536, _n_updates=16), final_checkpoint=dict(sha256="a" * 64)))
                self.put(f"{label}/eval/metadata.json", dict(checkpoint_sha256="a" * 64, cumulative_steps=65536, training_performed=False))
                for panel in PANELS:
                    phase, mode = panel.split("-")
                    episodes = [dict(seed=84001 + i, nav_success=i == 0, nav_fall=int(i == 1), timeout=i == 2, nav_targets=[3, 1, 2][i], reward=999, outcome=["success", "fall", "timeout"][i]) for i in range(3)]
                    self.put(f"{label}/eval/{panel}.json", dict(phase_key=phase, deterministic=mode == "deterministic", summary=dict(episodes=3, nav_successes=1, nav_falls=1, timeouts=1, navigation_targets_per_episode=2, wins=99), episodes=episodes))
        self.label = inputs[0]["label"]
        self.put("plan.json", dict(schema="native-capacity-round-v1", phase="0a", panel="dev", target=65536, starting_steps=32768, episodes_per_panel=3, inputs=inputs, evaluate_only=False))
        self.put("round-result.json", dict(status="completed", results=records, next_inputs=next_inputs))

    def test_complete_preserves_training_replicates_actual_counts_and_shares(self):
        result = self.mod.summarize(self.root)
        self.assertEqual((result["status"], result["completed_training_replicates"]), ("completed", 9))
        arm = result["arms"][0]
        self.assertEqual((arm["starting_steps"], arm["ending_steps"], arm["parameters"]), (32768, 65536, 1053973))
        self.assertEqual((arm["learner_wall_seconds"], arm["process_wall_seconds"], arm["steady_sps"]), (18, 20, 2048))
        self.assertEqual((arm["collection_share"], arm["update_share"]), (10 / 16, 5 / 16))
        panel = arm["panels"]["0a-deterministic"]
        self.assertEqual((panel["successes"], panel["falls"], panel["timeouts"], panel["targets"]), (1, 1, 1, 6))
        self.assertEqual(len(result["architectures"]), 3)
        for group in result["architectures"]:
            self.assertEqual(group["training_replicates"], 3)
            self.assertEqual(group["training_seeds"], [145, 146, 147])
            self.assertEqual(group["panels"]["0a-deterministic"]["mean_success_rate"], 1 / 3)
        path = str((self.root / "plan.json").resolve())
        self.assertEqual(result["source_sha256"][path], hashlib.sha256((self.root / "plan.json").read_bytes()).hexdigest())

    def test_default_rejects_partial_explicit_partial_reports_missing_seed(self):
        self.edit("round-result.json", lambda x: x.update(status="running", results=x["results"][:-2], next_inputs=x["next_inputs"][:-1]))
        with self.assertRaisesRegex(ValueError, "completed"):
            self.mod.summarize(self.root)
        result = self.mod.summarize(self.root, partial=True)
        self.assertEqual((result["status"], result["completed_training_replicates"]), ("partial", 8))
        self.assertEqual(len(result["pending_arms"]), 1)
        self.assertEqual(result["architectures"][-1]["training_replicates"], 2)

    def test_complete_status_cannot_hide_missing_seed_or_files(self):
        self.edit("round-result.json", lambda x: x.update(results=x["results"][:-2]))
        with self.assertRaises(ValueError):
            self.mod.summarize(self.root)

    def test_missing_download_is_explicitly_partial_not_completed(self):
        (self.root / self.label / "eval/0b-sampled.json").unlink()
        with self.assertRaises(ValueError):
            self.mod.summarize(self.root)
        result = self.mod.summarize(self.root, partial=True)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["pending_arms"], [self.label])

    def test_matrix_must_be_exact_nine_unique_training_seeds_even_in_partial(self):
        self.edit("plan.json", lambda x: x["inputs"][-1].update(seed=145))
        with self.assertRaises(ValueError):
            self.mod.summarize(self.root, partial=True)

    def test_counts_crosschecked_against_episodes_not_rewards(self):
        self.edit(f"{self.label}/eval/0a-deterministic.json", lambda x: x["summary"].update(nav_successes=2))
        with self.assertRaisesRegex(ValueError, "count"):
            self.mod.summarize(self.root, partial=True)

    def test_native_counters_and_eval_checkpoint_must_match(self):
        self.edit(f"{self.label}/eval/metadata.json", lambda x: x.update(checkpoint_sha256="b" * 64))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.mod.summarize(self.root)

    def test_final_checkpoint_step_is_actual_not_planned_target(self):
        self.edit(f"{self.label}/train/native_learning_receipt.json", lambda x: x["final_checkpoint_counters"].update(num_timesteps=65535))
        with self.assertRaisesRegex(ValueError, "counter"):
            self.mod.summarize(self.root, partial=True)

    def test_journal_kind_must_match_its_command_label(self):
        self.edit("round-result.json", lambda x: x["results"][0].update(kind="eval"))
        with self.assertRaisesRegex(ValueError, "journal"):
            self.mod.summarize(self.root)

    def test_bad_measured_time_and_step_proof_rejected(self):
        self.edit(f"{self.label}/train/campaign_summary.json", lambda x: x.update(measured_cycle_seconds=float("nan")))
        with self.assertRaises(ValueError):
            self.mod.summarize(self.root)

    def test_telemetry_uses_remote_only_half_open_whole_command_window(self):
        rows = [dict(**{"monitor/telemetry_ok": 1, "monitor/remote_unix": remote, "monitor/local_query_start_unix": local, "gpu/util": busy, "gpu/mem_used_mb": mem}) for remote, local, busy, mem in [(1100, 9, 20, 100), (1119, 9, 80, 200), (1120, 1110, 99, 900), (9, 1110, 99, 900)]]
        rows.append({"monitor/telemetry_ok": 0, "monitor/local_query_start_unix": 1110})
        path = self.root / "telemetry.jsonl"
        path.write_text("\n".join(map(json.dumps, rows)))
        value = self.mod.summarize(self.root, telemetry=path)["arms"][0]["telemetry"]
        self.assertEqual((value["samples"], value["mean_gpu_busy_percent"], value["peak_gpu_memory_used_mb"]), (2, 50, 200))
        self.assertIn("WHOLE training-command", value["scope"])
        rows[0].pop("monitor/remote_unix")
        path.write_text("\n".join(map(json.dumps, rows)))
        with self.assertRaisesRegex(ValueError, "remote_unix"):
            self.mod.summarize(self.root, telemetry=path)

    def test_skipped_existing_train_has_no_fabricated_process_time(self):
        path = self.root / "telemetry.jsonl"
        path.write_text(json.dumps({"monitor/telemetry_ok": 1, "monitor/remote_unix": 1110,
                                    "gpu/util": 90, "gpu/mem_used_mb": 900}))
        for omit_keys in (False, True):
            with self.subTest(omit_keys=omit_keys):
                self.edit("round-result.json", lambda x: x["results"][0].update(status="skipped_completed", started_unix=None, ended_unix=None))
                if omit_keys:
                    self.edit("round-result.json", lambda x: [x["results"][0].pop(key) for key in ("started_unix", "ended_unix")])
                result = self.mod.summarize(self.root, telemetry=path)
                arm = result["arms"][0]
                self.assertIsNone(arm["process_wall_seconds"])
                self.assertIsNone(arm["remote_train_command_window"])
                self.assertFalse(arm["telemetry"]["available"])
                self.assertEqual(arm["telemetry"]["samples"], 0)
                self.assertIsNone(arm["telemetry"]["mean_gpu_busy_percent"])
                self.assertEqual(result["architectures"][0]["process_wall_training_replicates"], 2)

    def test_resumed_train_preserves_original_window_process_wall_and_telemetry(self):
        self.edit("round-result.json", lambda x: x["results"][0].update(status="skipped_completed"))
        path = self.root / "telemetry.jsonl"
        path.write_text("\n".join(json.dumps({"monitor/telemetry_ok": 1, "monitor/remote_unix": timestamp,
                                             "gpu/util": busy, "gpu/mem_used_mb": memory})
                                  for timestamp, busy, memory in ((1100, 20, 100), (1119, 80, 200), (1120, 99, 900))))
        result = self.mod.summarize(self.root, telemetry=path)
        arm = result["arms"][0]
        self.assertEqual(arm["remote_train_command_window"], [1100, 1120])
        self.assertEqual(arm["process_wall_seconds"], 20)
        self.assertTrue(arm["telemetry"]["available"])
        self.assertEqual((arm["telemetry"]["samples"], arm["telemetry"]["mean_gpu_busy_percent"],
                          arm["telemetry"]["peak_gpu_memory_used_mb"]), (2, 50, 200))
        self.assertEqual(result["architectures"][0]["process_wall_training_replicates"], 3)
        self.assertEqual(result["architectures"][0]["mean_process_wall_seconds"], 20)

    def test_present_timestamps_must_be_finite_positive_and_complete_for_resumed_train(self):
        for status in ("completed", "skipped_completed"):
            for begin, end in ((-1, 1120), (1100, -1), (0, 1120), (1100, 0), (float("nan"), 1120),
                               (1100, float("inf")), ("1100", 1120), (True, 1120), (1120, 1100),
                               (1100, 1100), (None, 1120), (1100, None)):
                with self.subTest(status=status, begin=begin, end=end):
                    self.edit("round-result.json", lambda x: x["results"][0].update(status=status, started_unix=begin, ended_unix=end))
                    with self.assertRaisesRegex(ValueError, "remote train"):
                        self.mod.summarize(self.root)

    def test_cli_and_import_are_stdlib_only_and_read_only(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "wandb", "native_capacity_campaign", "campaign_learning"}:
                raise AssertionError("non-stdlib import")
            return original(name, *args, **kwargs)
        before = sorted(str(p) for p in self.root.rglob("*"))
        with mock.patch("builtins.__import__", side_effect=guarded), contextlib.redirect_stdout(io.StringIO()) as output:
            spec = importlib.util.spec_from_file_location("capacity_summary_light", MODULE)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            self.assertEqual(self.mod.main([str(self.root)]), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "completed")
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob("*")))


if __name__ == "__main__":
    unittest.main()
