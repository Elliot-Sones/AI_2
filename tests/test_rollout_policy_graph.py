
from ai2.paths import source_path
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = source_path(ROOT, "rollout_policy_graph.py")


def load_module():
    spec = importlib.util.spec_from_file_location("rollout_policy_graph", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rollout_policy_graph"] = module
    spec.loader.exec_module(module)
    return module


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTensor:
    def __init__(self, shape=(2, 3), dtype="float32", device="cuda:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = device


class FakeTorch:
    def __init__(self, current_device=0):
        self.cuda = types.SimpleNamespace(current_device=lambda: current_device)

    def device(self, raw):
        text = str(raw)
        if text == "cuda":
            return types.SimpleNamespace(type="cuda", index=None)
        if text.startswith("cuda:"):
            return types.SimpleNamespace(type="cuda", index=int(text.split(":", 1)[1]))
        return types.SimpleNamespace(type=text, index=None)


class DiagGaussianDistribution:
    pass


class SquashedDiagGaussianDistribution:
    pass


class FlattenExtractor:
    pass


class CustomExtractor:
    pass


class FakePolicy:
    def __init__(self):
        self.forward_calls = []
        self.use_sde = False
        self.recurrent = False
        self.training = False
        self._rollout_graph_policy_name = "ActorCriticPolicy"
        self.action_dist = DiagGaussianDistribution()
        self.features_extractor = FlattenExtractor()
        self.share_features_extractor = True
        self.log_std = types.SimpleNamespace(numel=lambda: 10)

    def forward(self, obs, deterministic=False):
        self.forward_calls.append((obs, deterministic))
        return ("eager-actions", "eager-values", "eager-logprob")


class RolloutPolicyGraphContractTests(unittest.TestCase):
    def test_context_temporarily_replaces_and_restores_forward_after_exception(self):
        module = load_module()
        policy = FakePolicy()
        original = policy.forward
        adapter = types.SimpleNamespace(forward=lambda obs, deterministic=False: ("graph", deterministic))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with module.temporary_policy_forward(policy, adapter):
                self.assertIsNot(policy.forward, original)
                self.assertEqual(policy.forward("obs", deterministic=False), ("graph", False))
                raise RuntimeError("boom")

        self.assertEqual(policy.forward("obs2", deterministic=True), original("obs2", deterministic=True))
        self.assertNotIn("forward", policy.__dict__)
        self.assertEqual(policy.forward("obs2", deterministic=True), ("eager-actions", "eager-values", "eager-logprob"))

    def test_unsupported_policy_features_fail_closed(self):
        module = load_module()
        for attr in ("use_sde", "recurrent"):
            policy = FakePolicy()
            setattr(policy, attr, True)
            with self.subTest(attr=attr):
                with self.assertRaisesRegex(RuntimeError, attr):
                    module.validate_supported_policy(policy)

    def test_unsupported_distribution_feature_extractor_and_action_dim_fail_closed(self):
        module = load_module()
        policy = FakePolicy()
        policy.action_dist = SquashedDiagGaussianDistribution()
        with self.assertRaisesRegex(RuntimeError, "DiagGaussianDistribution"):
            module.validate_supported_policy(policy)

        policy = FakePolicy()
        policy.features_extractor = CustomExtractor()
        with self.assertRaisesRegex(RuntimeError, "FlattenExtractor"):
            module.validate_supported_policy(policy)

        policy = FakePolicy()
        policy.log_std = types.SimpleNamespace(numel=lambda: 9)
        with self.assertRaisesRegex(RuntimeError, "10 action"):
            module.validate_supported_policy(policy)

    def test_validate_static_inputs_rejects_cpu_dtype_and_shape_changes(self):
        module = load_module()
        ref = module.StaticInputSpec.from_tensor(FakeTensor(shape=(64, 256), dtype="float32", device="cuda:0"))

        module.validate_static_input(ref, FakeTensor(shape=(64, 256), dtype="float32", device="cuda:0"))
        with self.assertRaisesRegex(RuntimeError, "shape"):
            module.validate_static_input(ref, FakeTensor(shape=(32, 256), dtype="float32", device="cuda:0"))
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            module.validate_static_input(ref, FakeTensor(shape=(64, 256), dtype="float64", device="cuda:0"))
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            module.validate_static_input(ref, FakeTensor(shape=(64, 256), dtype="float32", device="cpu"))

    def test_rollout_input_contract_allows_supported_static_batches_with_256_obs_width(self):
        module = load_module()

        module.validate_rollout_input_contract(FakeTensor(shape=(64, 256), dtype="torch.float32", device="cuda:0"))
        module.validate_rollout_input_contract(FakeTensor(shape=(1, 256), dtype="torch.float32", device="cuda:0"))
        module.validate_rollout_input_contract(FakeTensor(shape=(256, 256), dtype="torch.float32", device="cuda:0"))
        module.validate_rollout_input_contract(FakeTensor(shape=(4096, 256), dtype="torch.float32", device="cuda:0"))
        with self.assertRaisesRegex(RuntimeError, "batch"):
            module.validate_rollout_input_contract(FakeTensor(shape=(0, 256), dtype="torch.float32", device="cuda:0"))
        with self.assertRaisesRegex(RuntimeError, "batch"):
            module.validate_rollout_input_contract(FakeTensor(shape=(4097, 256), dtype="torch.float32", device="cuda:0"))
        with self.assertRaisesRegex(RuntimeError, "observation width 256"):
            module.validate_rollout_input_contract(FakeTensor(shape=(64, 255), dtype="torch.float32", device="cuda:0"))
        with self.assertRaisesRegex(RuntimeError, "rank-2"):
            module.validate_rollout_input_contract(FakeTensor(shape=(64, 256, 1), dtype="torch.float32", device="cuda:0"))
        with self.assertRaisesRegex(RuntimeError, "float32"):
            module.validate_rollout_input_contract(FakeTensor(shape=(64, 256), dtype="torch.float64", device="cuda:0"))
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            module.validate_rollout_input_contract(FakeTensor(shape=(64, 256), dtype="torch.float32", device="cpu"))

    def test_expected_batch_size_is_still_enforced_inside_supported_range(self):
        module = load_module()
        adapter = object.__new__(module.RolloutPolicyGraphForward)
        adapter.expected_batch_size = 256
        adapter.input_spec = None
        adapter.torch = FakeTorch(current_device=0)

        adapter._validate_obs(FakeTensor(shape=(256, 256), dtype="torch.float32", device="cuda:0"))
        adapter = object.__new__(module.RolloutPolicyGraphForward)
        adapter.expected_batch_size = 256
        adapter.input_spec = None
        adapter.torch = FakeTorch(current_device=0)
        with self.assertRaisesRegex(RuntimeError, "expected batch 256"):
            adapter._validate_obs(FakeTensor(shape=(128, 256), dtype="torch.float32", device="cuda:0"))

        with self.assertRaisesRegex(RuntimeError, "expected batch size"):
            module.validate_rollout_input_contract(
                FakeTensor(shape=(64, 256), dtype="torch.float32", device="cuda:0"),
                expected_batch_size="64",
            )

    def test_configured_expected_batch_size_rejects_unsupported_bounds(self):
        module = load_module()
        for invalid_batch in (0, 4097, True, 64.5, "64"):
            with self.subTest(invalid_batch=invalid_batch):
                with self.assertRaisesRegex(RuntimeError, "expected batch size"):
                    module.RolloutPolicyGraphForward(FakePolicy(), object(), expected_batch_size=invalid_batch)

    def test_current_cuda_device_gate_accepts_current_and_rejects_noncurrent_device(self):
        module = load_module()

        module.validate_current_cuda_device(FakeTorch(current_device=0), "cuda")
        module.validate_current_cuda_device(FakeTorch(current_device=0), "cuda:0")
        module.validate_current_cuda_device(FakeTorch(current_device=1), "cuda:1")
        with self.assertRaisesRegex(RuntimeError, "current CUDA device"):
            module.validate_current_cuda_device(FakeTorch(current_device=0), "cuda:1")
        with self.assertRaisesRegex(RuntimeError, "CUDA-only"):
            module.validate_current_cuda_device(FakeTorch(current_device=0), "cpu")

    def test_first_capture_rejects_noncurrent_cuda_device_before_freezing_spec(self):
        module = load_module()
        adapter = object.__new__(module.RolloutPolicyGraphForward)
        adapter.expected_batch_size = 64
        adapter.input_spec = None
        adapter.torch = FakeTorch(current_device=0)

        with self.assertRaisesRegex(RuntimeError, "current CUDA device"):
            adapter._validate_obs(FakeTensor(shape=(64, 256), dtype="torch.float32", device="cuda:1"))
        self.assertIsNone(adapter.input_spec)

    def test_stochastic_forward_uses_exact_external_normal_path(self):
        module = load_module()
        source = inspect.getsource(module.RolloutPolicyGraphForward._external_epsilon)

        self.assertIn(".empty(", source)
        self.assertIn(".normal_()", source)
        self.assertNotIn("randn", source)

    def test_capture_setup_and_deterministic_validation_use_fork_rng(self):
        module = load_module()
        source = inspect.getsource(module.RolloutPolicyGraphForward._ensure_runner)
        deterministic_source = inspect.getsource(module.validate_deterministic_forward_no_rng)

        self.assertIn("fork_rng", source)
        self.assertIn("fork_rng", deterministic_source)

    def test_mandatory_preflight_records_repeated_stochastic_and_deterministic_gates(self):
        module = load_module()
        adapter = object.__new__(module.RolloutPolicyGraphForward)
        adapter.preflight_receipt = None
        adapter.preflight_repeats = 3
        adapter._ensure_runner = lambda obs: None
        adapter.torch = FakeTorch(current_device=0)
        adapter.torch.random = types.SimpleNamespace(fork_rng=lambda devices: _null_context())
        adapter._rng_device = lambda obs: "cuda:0"
        adapter._validate_parameter_storage = lambda: None
        adapter.summary = lambda: {"preflight": adapter.preflight_receipt}
        calls = []
        module.snapshot_rng_state = lambda torch, device: {"cpu": "c", "cuda": "g"}
        module.rng_states_equal = lambda torch, left, right: True
        module.validate_stochastic_forward_exact = lambda torch, adapter_arg, obs: calls.append("stochastic") or {"passed": True}
        module.validate_deterministic_forward_no_rng = lambda torch, adapter_arg, obs: calls.append("deterministic") or {"passed": True}

        adapter.run_mandatory_preflight("obs")

        self.assertEqual(calls, ["stochastic", "stochastic", "stochastic", "deterministic"])
        self.assertEqual(adapter.preflight_receipt["status"], "passed")
        self.assertEqual(adapter.preflight_receipt["stochastic_repeats"], 3)
        self.assertTrue(adapter.summary()["preflight"]["setup_rng_restored"])

    def test_runtime_cuda_contract_tests_have_real_assertions_or_skip_without_gpu_stack(self):
        module = load_module()
        source = inspect.getsource(module.RolloutPolicyGraphRuntimeTests)
        for name in (
            "test_setup_preserves_rng_state",
            "test_repeated_stochastic_forward_matches_sb3_and_cuda_rng",
            "test_deterministic_forward_does_not_advance_rng",
            "test_weight_update_visible_without_storage_change",
            "test_log_std_update_with_zero_epsilon_changes_logprob_only",
            "test_outputs_are_lifetime_safe",
            "test_policy_forward_restores_after_exception",
            "test_ppo_save_load_inside_context",
        ):
            self.assertIn(name, source)
        try:
            import torch  # noqa: F401
            from stable_baselines3 import PPO  # noqa: F401
        except Exception as exc:
            self.skipTest(f"runtime CUDA stack unavailable locally: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable locally")

    def test_runtime_weight_update_targets_action_net_bias_not_first_parameter(self):
        module = load_module()
        source = inspect.getsource(module.RolloutPolicyGraphRuntimeTests.test_weight_update_visible_without_storage_change)

        self.assertIn("named_parameters = dict(policy.named_parameters())", source)
        self.assertIn('named_parameters["action_net.bias"]', source)
        self.assertIn("add_(0.125)", source)
        self.assertNotIn("next(policy.parameters())", source)
        self.assertIn("tuple_parity", source)

    def test_runtime_log_std_regression_documents_zero_epsilon_root_cause(self):
        module = load_module()
        source = inspect.getsource(module.RolloutPolicyGraphRuntimeTests.test_log_std_update_with_zero_epsilon_changes_logprob_only)

        self.assertIn("named_parameters = dict(policy.named_parameters())", source)
        self.assertIn('named_parameters["log_std"]', source)
        self.assertIn("actions_diff", source)
        self.assertIn("values_diff", source)
        self.assertIn("log_prob_diff", source)


if __name__ == "__main__":
    unittest.main()
