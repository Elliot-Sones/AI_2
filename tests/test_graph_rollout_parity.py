
from ai2.paths import source_path
import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = source_path(ROOT, "benchmark_graph_rollout_parity.py")


class FakeArray:
    def __init__(self, data, *, shape, dtype):
        self._data = data
        self.shape = shape
        self.dtype = dtype

    def tobytes(self):
        return self._data


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_graph_rollout_parity", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_graph_rollout_parity"] = module
    spec.loader.exec_module(module)
    return module


class GraphRolloutParityTests(unittest.TestCase):
    def helper_report(self):
        metric = {
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "max_normalized_error": 0.0,
            "dtype": "torch.float32",
        }
        return {
            "deterministic_zero_eps_vs_policy": {"actions": metric, "values": metric, "log_prob": metric},
            "stochastic_action_evaluate_actions": {"values": metric, "log_prob": metric},
            "permutation_equivariance": {"actions": metric, "values": metric, "log_prob": metric},
            "fresh_epsilon_action_diff": 1.0,
            "max_abs_diff": 0.0,
            "base_atol": 1e-5,
            "relative_tolerance_dtype_eps": 4,
        }

    def graph_summary(self, **overrides):
        summary = {
            "captures": 1,
            "replays": 512,
            "preflight": {
                "status": "passed",
                "stochastic_repeats": 3,
                "setup_rng_restored": True,
                "stochastic_exact": [{"passed": True}, {"passed": True}, {"passed": True}],
                "deterministic_no_rng": {"passed": True},
            },
            "input_spec": {"device": "cuda:0"},
            "eps_spec": {"device": "cuda:0"},
            "helper_direct_sb3": self.helper_report(),
            "graph_direct_sb3": self.helper_report(),
        }
        summary.update(overrides)
        return summary

    def file_record(self, sha):
        return {"path": f"/inputs/{sha}", "exists": True, "sha256": sha, "size": 10}

    def complete_receipt(self):
        return {
            "status": "completed",
            "input_checkpoint_sha256": "input",
            "checkpoint_container_sha256": "container-a",
            "rollout_buffer_sha256": "buf",
            "pre_update_policy_sha256": "pre-policy",
            "pre_update_optimizer_sha256": "pre-opt",
            "policy_sha256": "policy",
            "optimizer_sha256": "opt",
            "pre_update_rng_sha256": {"torch_cpu": "rng-pre-cpu", "torch_cuda": "rng-pre-cuda"},
            "post_update_rng_sha256": {"torch_cpu": "rng-post-cpu", "torch_cuda": "rng-post-cuda"},
            "num_timesteps": 16033824,
            "expected_steps": 32768,
            "expected_epochs": 8,
            "expected_optimizer_steps": 32,
            "actual_steps": 32768,
            "collect_calls": 1,
            "train_calls": 1,
            "n_updates_before": 0,
            "n_updates_after": 8,
            "n_updates_delta": 8,
            "optimizer_step_calls": 32,
            "actual_device": {
                "requested": "cuda",
                "policy_device": "cuda:0",
                "parameters_all_cuda": True,
                "cuda_available": True,
            },
            "train_result": {
                "status": "completed",
                "starting_steps": 16001056,
                "ending_steps": 16033824,
                "actual_steps": 32768,
                "stop_reason": None,
                "error": None,
            },
            "train_metadata": {
                "device": "cuda",
                "seed": 45,
                "phase": "0b",
                "n_envs": 64,
                "checkpoint": "/inputs/checkpoint.zip",
                "ppo_controls": {"n_steps": 512, "batch_size": 8192, "n_epochs": 8, "reset_num_timesteps": False},
                "output_files": {"final": "/outputs/final.zip"},
                "effective_params": {"ppo_settings": {"model_checkpoint": "/inputs/checkpoint.zip"}},
            },
            "provenance_before": {
                "source_train": self.file_record("source"),
                "config": self.file_record("config"),
                "train_job": self.file_record("trainjob"),
                "checkpoint": self.file_record("checkpoint"),
                "verifier": self.file_record("verifier"),
                "campaign": self.file_record("campaign"),
                "grouped_env": self.file_record("grouped-env"),
                "rollout_policy_graph": self.file_record("graph"),
                "policy_compile": self.file_record("compile"),
                "policy_cuda_graph": self.file_record("cuda-graph"),
                "controls": {
                    "n_envs": 64,
                    "n_steps": 512,
                    "batch_size": 8192,
                    "epochs": 8,
                    "seed": 45,
                    "phase": "0b",
                    "device": "cuda",
                    "checkpoint": "/inputs/checkpoint.zip",
                    "group_size": 1,
                    "grouped_env_path": "/inputs/grouped-env",
                }
            },
            "provenance_after": {
                "source_train": self.file_record("source"),
                "config": self.file_record("config"),
                "train_job": self.file_record("trainjob"),
                "checkpoint": self.file_record("checkpoint"),
                "verifier": self.file_record("verifier"),
                "campaign": self.file_record("campaign"),
                "grouped_env": self.file_record("grouped-env"),
                "rollout_policy_graph": self.file_record("graph"),
                "policy_compile": self.file_record("compile"),
                "policy_cuda_graph": self.file_record("cuda-graph"),
                "final_checkpoint": {"exists": True},
            },
        }

    def test_build_campaign_args_sets_one_update_with_and_without_graph(self):
        module = load_module()
        args = argparse.Namespace(
            source=Path("/src"),
            train_job=Path("/scripts/train_job.py"),
            config=Path("/src/control.yaml"),
            checkpoint=Path("/ckpt.zip"),
            output=Path("/out"),
            n_envs=64,
            n_steps=512,
            batch_size=8192,
            epochs=8,
            seed=45,
            phase="0b",
            device="cuda",
            maxseconds=180.0,
            group_size=4,
            grouped_env_path=Path("/scripts/grouped_vec_env.py"),
            rollout_policy_graph_path=Path("/scripts/rollout_policy_graph.py"),
        )

        control = module.build_campaign_argv(args, "control", use_graph=False)
        graph = module.build_campaign_argv(args, "graph", use_graph=True)

        self.assertIn("--cycles", control)
        self.assertIn("1", control)
        self.assertNotIn("--rollout-policy-graph", control)
        self.assertIn("--rollout-policy-graph", graph)
        self.assertIn(str(Path("/out/graph")), graph)
        self.assertEqual(control[control.index("--group-size") + 1], "4")
        self.assertEqual(graph[graph.index("--grouped-env-path") + 1], str(Path("/scripts/grouped_vec_env.py")))

    def test_run_arm_installs_grouped_vec_env_and_cleans_up(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source_path(source, "train.py")).write_text("train\n", encoding="utf-8")
            files = {}
            for name in ("train_job.py", "config.yaml", "checkpoint.zip", "rollout_policy_graph.py", "grouped_vec_env.py"):
                path = source_path(root, name)
                path.write_text(name, encoding="utf-8")
                files[name] = path

            class FakeTrainJob:
                PPO = type("BasePPO", (), {})
                SubprocVecEnv = object()

                @staticmethod
                def run_job(train_args):
                    out = Path(train_args.output)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "result.json").write_text(
                        json.dumps({"status": "completed", "actual_steps": 32768, "starting_steps": 0, "ending_steps": 32768, "stop_reason": None, "error": None}),
                        encoding="utf-8",
                    )
                    (out / "metadata.json").write_text(json.dumps({"device": "cuda"}), encoding="utf-8")
                    (out / "final.zip").write_bytes(b"final")

            calls = []

            def fake_patch(train_job, args):
                calls.append(("patch", train_job, args.group_size, args.grouped_env_path))

                def cleanup():
                    calls.append(("cleanup", train_job, args.group_size, args.grouped_env_path))

                return cleanup

            original_import = module.benchmark_campaign.import_train_job_module
            original_build = module.benchmark_campaign.build_train_job_args
            original_patch = module.benchmark_campaign.patch_grouped_vec_env
            try:
                module.benchmark_campaign.import_train_job_module = lambda path: FakeTrainJob
                module.benchmark_campaign.build_train_job_args = lambda args, plan: argparse.Namespace(output=str(Path(args.output) / "control"))
                module.benchmark_campaign.patch_grouped_vec_env = fake_patch
                args = argparse.Namespace(
                    source=source,
                    train_job=files["train_job.py"],
                    config=files["config.yaml"],
                    checkpoint=files["checkpoint.zip"],
                    output=root / "out",
                    n_envs=256,
                    n_steps=128,
                    batch_size=8192,
                    epochs=8,
                    seed=45,
                    phase="0b",
                    device="cuda",
                    maxseconds=180.0,
                    history=False,
                    opponents=None,
                    group_size=4,
                    grouped_env_path=files["grouped_vec_env.py"],
                    rollout_policy_graph_path=files["rollout_policy_graph.py"],
                    rollout_policy_graph_capture_warmup=3,
                    rollout_policy_graph_tolerance=1e-5,
                )

                receipt = module.run_arm(args, "control", use_graph=False)
            finally:
                module.benchmark_campaign.import_train_job_module = original_import
                module.benchmark_campaign.build_train_job_args = original_build
                module.benchmark_campaign.patch_grouped_vec_env = original_patch

            self.assertEqual([call[0] for call in calls], ["patch", "cleanup"])
            self.assertEqual(receipt["provenance_before"]["controls"]["group_size"], 4)
            self.assertEqual(receipt["provenance_before"]["grouped_env"]["sha256"], module.sha256_file(files["grouped_vec_env.py"]))

    def test_compare_parity_gates_state_but_records_zip_container_without_requiring_match(self):
        module = load_module()
        control = self.complete_receipt()
        graph = dict(control)
        graph["use_rollout_policy_graph"] = True
        graph["checkpoint_container_sha256"] = "container-b"
        graph["rollout_policy_graph"] = self.graph_summary()

        result = module.compare_parity(control, graph)

        self.assertTrue(result["byte_parity"])
        self.assertIn("checkpoint_container_sha256", result["recorded_not_gated"])
        graph["policy_sha256"] = "different"
        with self.assertRaisesRegex(RuntimeError, "policy_sha256"):
            module.compare_parity(control, graph)

    def test_compare_parity_fails_closed_for_missing_fields_and_rng_mismatch(self):
        module = load_module()
        control = self.complete_receipt()
        graph = dict(control)
        graph["use_rollout_policy_graph"] = True
        graph["rollout_policy_graph"] = self.graph_summary(replays=1)

        del control["optimizer_sha256"]
        del graph["optimizer_sha256"]
        with self.assertRaisesRegex(RuntimeError, "missing.*optimizer_sha256"):
            module.compare_parity(control, graph)

        control = self.complete_receipt()
        graph = dict(control)
        graph["use_rollout_policy_graph"] = True
        graph["rollout_policy_graph"] = self.graph_summary(replays=1)
        graph["post_update_rng_sha256"] = {"torch_cpu": "different", "torch_cuda": "rng-post-cuda"}
        with self.assertRaisesRegex(RuntimeError, "post_update_rng_sha256"):
            module.compare_parity(control, graph)

    def test_compare_parity_requires_true_graph_capture_replay_cuda_and_expected_counts(self):
        module = load_module()
        control = self.complete_receipt()
        graph = dict(control)
        graph["use_rollout_policy_graph"] = True
        graph["rollout_policy_graph"] = self.graph_summary(captures=0, replays=0)
        with self.assertRaisesRegex(RuntimeError, "captures"):
            module.compare_parity(control, graph)

        graph["rollout_policy_graph"]["captures"] = 1
        graph["rollout_policy_graph"]["replays"] = 1
        graph["actual_device"] = dict(graph["actual_device"], parameters_all_cuda=False)
        with self.assertRaisesRegex(RuntimeError, "parameters_all_cuda"):
            module.compare_parity(control, graph)

        graph = dict(self.complete_receipt())
        graph["use_rollout_policy_graph"] = True
        graph["rollout_policy_graph"] = self.graph_summary(replays=1)
        graph["optimizer_step_calls"] = 31
        with self.assertRaisesRegex(RuntimeError, "optimizer_step_calls"):
            module.compare_parity(control, graph)

    def test_validate_training_receipt_rejects_partial_or_wrong_step_results(self):
        module = load_module()
        receipt = self.complete_receipt()
        module.validate_training_receipt(receipt, expected_steps=32768, expected_epochs=8, expected_optimizer_steps=32)

        bad = self.complete_receipt()
        bad["train_result"] = {"status": "completed", "actual_steps": 1, "stop_reason": None, "error": None}
        with self.assertRaisesRegex(RuntimeError, "actual_steps"):
            module.validate_training_receipt(bad, expected_steps=32768, expected_epochs=8, expected_optimizer_steps=32)

        bad = self.complete_receipt()
        bad["train_calls"] = 2
        with self.assertRaisesRegex(RuntimeError, "train_calls"):
            module.validate_training_receipt(bad, expected_steps=32768, expected_epochs=8, expected_optimizer_steps=32)

        bad = self.complete_receipt()
        bad["train_metadata"] = {
            "device": "cpu",
            "seed": 45,
            "phase": "0b",
            "n_envs": 64,
            "checkpoint": "/inputs/checkpoint.zip",
            "ppo_controls": {"n_steps": 512, "batch_size": 8192, "n_epochs": 8, "reset_num_timesteps": False},
            "output_files": {"final": "/outputs/final.zip"},
            "effective_params": {"ppo_settings": {"model_checkpoint": "/inputs/checkpoint.zip"}},
        }
        with self.assertRaisesRegex(RuntimeError, "metadata.*device"):
            module.validate_training_receipt(bad, expected_steps=32768, expected_epochs=8, expected_optimizer_steps=32)

        bad = self.complete_receipt()
        bad["train_metadata"] = {
            "device": "cuda",
            "seed": 45,
            "phase": "0b",
            "n_envs": 64,
            "checkpoint": "/inputs/checkpoint.zip",
        }
        with self.assertRaisesRegex(RuntimeError, "ppo_controls"):
            module.validate_training_receipt(bad, expected_steps=32768, expected_epochs=8, expected_optimizer_steps=32)

        bad = self.complete_receipt()
        bad["train_result"] = dict(bad["train_result"], ending_steps=16033825)
        with self.assertRaisesRegex(RuntimeError, "ending_steps"):
            module.validate_training_receipt(bad, expected_steps=32768, expected_epochs=8, expected_optimizer_steps=32)

        bad = self.complete_receipt()
        bad["num_timesteps"] = 16033825
        with self.assertRaisesRegex(RuntimeError, "num_timesteps"):
            module.validate_training_receipt(bad, expected_steps=32768, expected_epochs=8, expected_optimizer_steps=32)

        bad = self.complete_receipt()
        bad["train_metadata"]["ppo_controls"]["reset_num_timesteps"] = True
        with self.assertRaisesRegex(RuntimeError, "reset_num_timesteps"):
            module.validate_training_receipt(bad, expected_steps=32768, expected_epochs=8, expected_optimizer_steps=32)

    def test_compare_parity_fails_closed_on_control_drift_and_run_refuses_existing_output(self):
        module = load_module()
        control = self.complete_receipt()
        graph = dict(control)
        graph["use_rollout_policy_graph"] = True
        graph["rollout_policy_graph"] = self.graph_summary(replays=1)
        graph["provenance_before"] = dict(graph["provenance_before"])
        graph["provenance_before"]["controls"] = dict(control["provenance_before"]["controls"], seed=46)
        graph["train_metadata"] = dict(graph["train_metadata"], seed=46)
        with self.assertRaisesRegex(RuntimeError, "controls"):
            module.compare_parity(control, graph)

        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(output=Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                module.run(args)

    def test_compare_parity_fails_closed_on_input_hash_drift(self):
        module = load_module()
        control = self.complete_receipt()
        graph = dict(control)
        graph["use_rollout_policy_graph"] = True
        graph["rollout_policy_graph"] = self.graph_summary()

        control["provenance_after"] = dict(control["provenance_after"], config=self.file_record("changed"))
        with self.assertRaisesRegex(RuntimeError, "changed during control"):
            module.compare_parity(control, graph)

        control = self.complete_receipt()
        graph = dict(control)
        graph["use_rollout_policy_graph"] = True
        graph["rollout_policy_graph"] = self.graph_summary()
        graph["provenance_before"] = dict(graph["provenance_before"], train_job=self.file_record("other-trainjob"))
        graph["provenance_after"] = dict(graph["provenance_after"], train_job=self.file_record("other-trainjob"))
        with self.assertRaisesRegex(RuntimeError, "shared input.*train_job"):
            module.compare_parity(control, graph)

        graph = dict(self.complete_receipt())
        graph["use_rollout_policy_graph"] = True
        graph["rollout_policy_graph"] = self.graph_summary()
        graph["provenance_after"] = dict(graph["provenance_after"], rollout_policy_graph=self.file_record("changed-graph"))
        with self.assertRaisesRegex(RuntimeError, "changed during graph"):
            module.compare_parity(self.complete_receipt(), graph)

    def test_digest_supports_typed_int_and_string_optimizer_state_keys(self):
        module = load_module()
        adam_like = {
            "state": {
                1: {
                    "step": 2,
                    "exp_avg": FakeArray(b"avg", shape=(2,), dtype="float32"),
                    "exp_avg_sq": FakeArray(b"sq", shape=(2,), dtype="float32"),
                },
                10: {"step": 3},
            },
            "param_groups": [{"params": [1, 10], "lr": 0.001}],
        }
        digest = module.digest_any(adam_like)

        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertNotEqual(module.digest_any({1: "same"}), module.digest_any({"1": "same"}))

    def test_validate_graph_receipt_requires_passed_preflight_and_exact_repeats(self):
        module = load_module()
        receipt = self.complete_receipt()
        receipt["use_rollout_policy_graph"] = True
        receipt["rollout_policy_graph"] = {
            "captures": 1,
            "replays": 1,
            "preflight": {
                "status": "failed",
                "stochastic_repeats": 3,
                "setup_rng_restored": True,
                "stochastic_exact": [{"passed": True}, {"passed": True}, {"passed": True}],
                "deterministic_no_rng": {"passed": True},
            },
            "input_spec": {"device": "cuda:0"},
            "eps_spec": {"device": "cuda:0"},
            "helper_direct_sb3": self.helper_report(),
            "graph_direct_sb3": self.helper_report(),
        }
        with self.assertRaisesRegex(RuntimeError, "preflight"):
            module.validate_graph_receipt(receipt)

        receipt["rollout_policy_graph"]["preflight"] = {
            "status": "passed",
            "stochastic_repeats": 3,
            "setup_rng_restored": True,
            "stochastic_exact": [{"passed": True}, {"passed": True}],
            "deterministic_no_rng": {"passed": True},
        }
        with self.assertRaisesRegex(RuntimeError, "stochastic_exact"):
            module.validate_graph_receipt(receipt)

        receipt["rollout_policy_graph"] = self.graph_summary(helper_direct_sb3={"passed": True})
        with self.assertRaisesRegex(RuntimeError, "helper_direct_sb3"):
            module.validate_graph_receipt(receipt)

    def test_digest_rejects_unsupported_arbitrary_objects(self):
        module = load_module()

        class Unsupported:
            pass

        with self.assertRaisesRegex(TypeError, "unsupported"):
            module.digest_any(Unsupported())

    def test_read_parity_receipt_fails_closed_on_missing_fields(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "missing"):
                module.read_parity_receipt(path)


if __name__ == "__main__":
    unittest.main()
