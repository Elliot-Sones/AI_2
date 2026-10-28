import importlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


class TensorTrainingDriverTests(unittest.TestCase):
    def test_patch_wraps_real_factory_and_restores_on_failure(self):
        module = importlib.import_module('tools.benchmarks.benchmark_tensor_training')
        factory_calls = []
        original_env = object()
        def build(*args, **kwargs):
            factory_calls.append((args, kwargs))
            return original_env
        original_ppo = type("OriginalPPO", (), {})
        candidate_ppo = type("CandidatePPO", (), {})
        job = types.SimpleNamespace(build_env=build, PPO=original_ppo)
        converted = []
        def adapt(env, *, device):
            converted.append((env, device))
            return (env, device)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with module.tensor_job_context(job, adapter=adapt, ppo_class=candidate_ppo):
                self.assertIs(job.PPO, candidate_ppo)
                result = job.build_env("train", "phase", {}, 8, 45, "cuda")
                self.assertEqual(result, (original_env, "cuda"))
                raise RuntimeError("injected")
        self.assertIs(job.PPO, original_ppo)
        self.assertIs(job.build_env, build)
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(converted, [(original_env, "cuda")])

    def test_control_does_not_import_or_patch_tensor_runtime(self):
        module = importlib.import_module('tools.benchmarks.benchmark_tensor_training')
        args = types.SimpleNamespace(tensor_collector=False)
        original = module.campaign.import_train_job_module
        with patch.object(module.campaign, "run_campaign", return_value=0) as run:
            self.assertEqual(module.run(args), 0)
        run.assert_called_once_with(args)
        self.assertIs(module.campaign.import_train_job_module, original)

    def test_cpu_candidate_rejected_before_running_game(self):
        module = importlib.import_module('tools.benchmarks.benchmark_tensor_training')
        args = types.SimpleNamespace(tensor_collector=True, device="cpu")
        with patch.object(module.campaign, "run_campaign") as run:
            with self.assertRaisesRegex(ValueError, "CUDA"):
                module.run(args)
        run.assert_not_called()

    def test_candidate_records_sources_and_restores_importer_on_failure(self):
        module = importlib.import_module('tools.benchmarks.benchmark_tensor_training')
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(tensor_collector=True, device="cuda", output=Path(tmp))
            original = module.campaign.import_train_job_module
            with patch.object(module.campaign, "run_campaign", side_effect=RuntimeError("injected")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    module.run(args)
            self.assertIs(module.campaign.import_train_job_module, original)
            self.assertTrue((Path(tmp) / "tensor_candidate_provenance.json").is_file())
            provenance = json.loads((Path(tmp) / "tensor_candidate_provenance.json").read_text())
            self.assertEqual(set(provenance["candidate"]), set(module.CANDIDATE_FILES))
            self.assertTrue(all(value["sha256"] for value in provenance["candidate"].values()))

    def test_missing_candidate_source_prevents_launch(self):
        module = importlib.import_module('tools.benchmarks.benchmark_tensor_training')
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(tensor_collector=True, device="cuda", output=Path(tmp))
            with patch.object(module, "CANDIDATE_FILES", ("not-present-input.py",)):
                with patch.object(module.campaign, "run_campaign") as run:
                    with self.assertRaises(FileNotFoundError):
                        module.run(args)
                    run.assert_not_called()

    def test_tensor_graph_combination_is_rejected(self):
        module = importlib.import_module('tools.benchmarks.benchmark_tensor_training')
        args = types.SimpleNamespace(tensor_collector=True, device="cuda", rollout_policy_graph=True)
        with self.assertRaisesRegex(ValueError, "graph"):
            module.run(args)

    def test_adapter_failure_closes_env(self):
        module = importlib.import_module('tools.benchmarks.benchmark_tensor_training')
        closed = []
        env = types.SimpleNamespace(close=lambda: closed.append(True))
        job = types.SimpleNamespace(build_env=lambda *a, **kw: env, PPO=object)
        def fail(*a, **kw):
            raise ValueError("adapter failure")
        with module.tensor_job_context(job, adapter=fail, ppo_class=dict):
            with self.assertRaisesRegex(ValueError, "adapter failure"):
                job.build_env("train", "phase", {}, 8, 45, "cuda")
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
