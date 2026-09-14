#!/usr/bin/env python3
"""Opt-in CUDA-graph policy forward for benchmark rollout collection only.

The public integration point is a context manager that temporarily replaces an
SB3 ActorCriticPolicy ``forward`` method while ``PPO.collect_rollouts`` runs.
It deliberately leaves ``predict_values`` and ``evaluate_actions`` eager and
does not copy or fork SB3's rollout loop.
"""

from __future__ import annotations

import dataclasses
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator
import unittest

import benchmark_policy_compile as e4
import benchmark_policy_cuda_graph as graph_core


MIN_SUPPORTED_BATCH_SIZE = 1
MAX_SUPPORTED_BATCH_SIZE = 4096
SUPPORTED_OBSERVATION_WIDTH = 256


def validate_supported_batch_size(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"rollout policy graph {label} must be a strict int, got {value!r}")
    if not (MIN_SUPPORTED_BATCH_SIZE <= value <= MAX_SUPPORTED_BATCH_SIZE):
        raise RuntimeError(
            "rollout policy graph "
            f"{label} must be {MIN_SUPPORTED_BATCH_SIZE}..{MAX_SUPPORTED_BATCH_SIZE}, got {value}"
        )
    return value


@dataclasses.dataclass(frozen=True)
class StaticInputSpec:
    shape: tuple[int, ...]
    dtype: str
    device: str

    @classmethod
    def from_tensor(cls, tensor: Any) -> "StaticInputSpec":
        return cls(
            shape=tuple(int(item) for item in getattr(tensor, "shape")),
            dtype=str(getattr(tensor, "dtype")),
            device=str(getattr(tensor, "device")),
        )


def validate_static_input(expected: StaticInputSpec, tensor: Any) -> None:
    actual = StaticInputSpec.from_tensor(tensor)
    if not actual.device.startswith("cuda"):
        raise RuntimeError(f"rollout policy graph requires CUDA tensor input, got {actual.device}")
    if actual.shape != expected.shape:
        raise RuntimeError(f"rollout policy graph shape mismatch: expected {expected.shape}, got {actual.shape}")
    if actual.dtype != expected.dtype:
        raise RuntimeError(f"rollout policy graph dtype mismatch: expected {expected.dtype}, got {actual.dtype}")
    if actual.device != expected.device:
        raise RuntimeError(f"rollout policy graph device mismatch: expected {expected.device}, got {actual.device}")


def current_cuda_device_index(torch: Any) -> int:
    return int(torch.cuda.current_device())


def validate_current_cuda_device(torch: Any, device: str) -> int:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError(f"rollout policy graph is CUDA-only, got {device}")
    requested_index = current_cuda_device_index(torch) if resolved.index is None else int(resolved.index)
    current_index = current_cuda_device_index(torch)
    if requested_index != current_index:
        raise RuntimeError(
            "rollout policy graph only supports the current CUDA device before capture: "
            f"requested cuda:{requested_index}, current cuda:{current_index}"
        )
    return current_index


def infer_action_dim(policy: Any) -> int:
    log_std = getattr(policy, "log_std", None)
    if log_std is not None and hasattr(log_std, "numel"):
        return int(log_std.numel())
    action_net = getattr(policy, "action_net", None)
    if action_net is not None and hasattr(action_net, "out_features"):
        return int(action_net.out_features)
    action_space = getattr(policy, "action_space", None)
    shape = getattr(action_space, "shape", None)
    if shape:
        total = 1
        for item in shape:
            total *= int(item)
        return total
    raise RuntimeError("cannot infer rollout policy graph action dimension")


def validate_supported_policy(policy: Any) -> None:
    policy_name = str(getattr(policy, "_rollout_graph_policy_name", policy.__class__.__name__))
    if policy_name not in {"ActorCriticPolicy", "MlpPolicy", "MultiInputPolicy"}:
        raise RuntimeError(f"rollout policy graph supports only SB3 MlpPolicy/ActorCriticPolicy, got {policy_name}")
    if bool(getattr(policy, "use_sde", False)):
        raise RuntimeError("rollout policy graph does not support use_sde policies")
    for attr in ("recurrent", "is_recurrent", "lstm_actor", "lstm_critic"):
        if bool(getattr(policy, attr, False)):
            raise RuntimeError(f"rollout policy graph does not support {attr} policies")
    action_dist = getattr(policy, "action_dist", None)
    dist_name = "" if action_dist is None else action_dist.__class__.__name__
    if dist_name != "DiagGaussianDistribution":
        raise RuntimeError(f"rollout policy graph supports only unsquashed DiagGaussianDistribution, got {dist_name}")
    if bool(getattr(policy, "squash_output", False)):
        raise RuntimeError("rollout policy graph does not support squashed actions")
    extractor = getattr(policy, "features_extractor", None)
    extractor_name = "" if extractor is None else extractor.__class__.__name__
    if extractor_name != "FlattenExtractor":
        raise RuntimeError(f"rollout policy graph supports only FlattenExtractor, got {extractor_name}")
    if not bool(getattr(policy, "share_features_extractor", True)):
        raise RuntimeError("rollout policy graph requires shared feature extractor")
    action_dim = infer_action_dim(policy)
    if action_dim != 10:
        raise RuntimeError(f"rollout policy graph supports exactly 10 action dimensions, got {action_dim}")


def validate_rollout_input_contract(obs: Any, *, expected_batch_size: int | None = None) -> None:
    spec = StaticInputSpec.from_tensor(obs)
    if len(spec.shape) != 2:
        raise RuntimeError(f"rollout policy graph supports only rank-2 observations, got {spec.shape}")
    batch_size, obs_width = spec.shape
    validate_supported_batch_size(batch_size, label="batch size")
    if expected_batch_size is not None:
        expected_batch_size = validate_supported_batch_size(expected_batch_size, label="expected batch size")
        if batch_size != expected_batch_size:
            raise RuntimeError(f"rollout policy graph expected batch {expected_batch_size}, got {batch_size}")
    if obs_width != SUPPORTED_OBSERVATION_WIDTH:
        raise RuntimeError(
            f"rollout policy graph supports only observation width {SUPPORTED_OBSERVATION_WIDTH}, got {obs_width}"
        )
    if spec.dtype not in {"torch.float32", "float32"}:
        raise RuntimeError(f"rollout policy graph supports only float32 observations, got {spec.dtype}")
    if not spec.device.startswith("cuda"):
        raise RuntimeError(f"rollout policy graph requires CUDA observations, got {spec.device}")


def cuda_device_index(torch: Any, device: str) -> int:
    return validate_current_cuda_device(torch, device)


def _rng_devices(torch: Any, device: str) -> list[int]:
    return [cuda_device_index(torch, device)]


def snapshot_rng_state(torch: Any, device: str) -> dict[str, Any]:
    index = cuda_device_index(torch, device)
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state(index).clone(),
    }


def restore_rng_state(torch: Any, device: str, state: dict[str, Any]) -> None:
    index = cuda_device_index(torch, device)
    torch.random.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], index)


def rng_states_equal(torch: Any, left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(torch.equal(left["cpu"], right["cpu"])) and bool(torch.equal(left["cuda"], right["cuda"]))


def _tuple_exact_equal(torch: Any, left: tuple[Any, ...], right: tuple[Any, ...]) -> dict[str, Any]:
    names = ("actions", "values", "log_prob")
    if len(left) != 3 or len(right) != 3:
        raise RuntimeError(f"expected three forward outputs, got {len(left)} and {len(right)}")
    result: dict[str, Any] = {}
    for name, lhs, rhs in zip(names, left, right):
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
            raise RuntimeError(f"{name} shape/dtype mismatch: {lhs.shape}/{lhs.dtype} vs {rhs.shape}/{rhs.dtype}")
        e4.assert_finite(torch, f"{name}_left", lhs)
        e4.assert_finite(torch, f"{name}_right", rhs)
        equal = bool(torch.equal(lhs, rhs))
        if not equal:
            raise RuntimeError(f"{name} exact tensor parity failed")
        result[name] = {"exact_equal": True, "shape": [int(item) for item in lhs.shape], "dtype": str(lhs.dtype)}
    return result


class RolloutPolicyGraphForward:
    """Lazy CUDA-graph replacement for stochastic SB3 policy forward."""

    def __init__(
        self,
        policy: Any,
        torch: Any,
        *,
        original_forward: Callable[..., tuple[Any, Any, Any]] | None = None,
        expected_batch_size: int | None = None,
        capture_warmup: int = 3,
        tolerance: float = 1e-5,
    ) -> None:
        validate_supported_policy(policy)
        self.policy = policy
        self.torch = torch
        self.original_forward = original_forward or policy.forward
        self.expected_batch_size = (
            None
            if expected_batch_size is None
            else validate_supported_batch_size(expected_batch_size, label="expected batch size")
        )
        self.capture_warmup = int(capture_warmup)
        self.tolerance = float(tolerance)
        self.helper = e4.static_gaussian_actorcritic(policy, torch)
        self.input_spec: StaticInputSpec | None = None
        self.eps_spec: StaticInputSpec | None = None
        self.runner: graph_core.CudaGraphPolicyRunner | None = None
        self.parameter_signature: list[dict[str, Any]] | None = None
        self.capture_seconds: float | None = None
        self.replay_count = 0
        self.preflight_replay_count = 0
        self.deterministic_eager_count = 0
        self.preflight_repeats = 3
        self.preflight_receipt: dict[str, Any] | None = None

    def _action_dim(self) -> int:
        return infer_action_dim(self.policy)

    def _zero_epsilon(self, obs: Any) -> Any:
        return self.torch.zeros((int(obs.shape[0]), self._action_dim()), dtype=obs.dtype, device=obs.device)

    def _external_epsilon(self, obs: Any) -> Any:
        return self.torch.empty((int(obs.shape[0]), self._action_dim()), dtype=obs.dtype, device=obs.device).normal_()

    def _validate_obs(self, obs: Any) -> None:
        if self.input_spec is None:
            validate_rollout_input_contract(obs, expected_batch_size=self.expected_batch_size)
            validate_current_cuda_device(self.torch, str(obs.device))
            self.input_spec = StaticInputSpec.from_tensor(obs)
            if not self.input_spec.device.startswith("cuda"):
                raise RuntimeError(f"rollout policy graph requires CUDA tensor input, got {self.input_spec.device}")
        validate_static_input(self.input_spec, obs)

    def _validate_parameter_storage(self) -> None:
        current = graph_core.parameter_storage_signature(self.policy)
        if self.parameter_signature is None:
            self.parameter_signature = current
        elif current != self.parameter_signature:
            raise RuntimeError("rollout policy graph parameter storage changed; refusing unsafe replay")

    def _rng_device(self, obs: Any) -> str:
        return str(obs.device)

    def _ensure_runner(self, obs: Any) -> None:
        self._validate_obs(obs)
        self._validate_parameter_storage()
        if self.runner is not None:
            return
        device = str(obs.device)
        started = time.perf_counter()
        with self.torch.random.fork_rng(devices=_rng_devices(self.torch, device)):
            zero_eps = self._zero_epsilon(obs)
            self.eps_spec = StaticInputSpec.from_tensor(zero_eps)
            helper_direct = e4.validate_helper_against_policy(
                self.torch,
                self.policy,
                self.helper,
                obs,
                zero_eps,
                tol=self.tolerance,
            )
            self.runner = graph_core.CudaGraphPolicyRunner(
                self.torch,
                self.helper,
                obs,
                zero_eps,
                device=device,
                capture_warmup=self.capture_warmup,
            )
            graph_direct = e4.validate_helper_against_policy(
                self.torch,
                self.policy,
                self.runner.replay,
                obs,
                zero_eps,
                tol=self.tolerance,
            )
        self.capture_seconds = time.perf_counter() - started
        self.helper_direct_sb3 = helper_direct
        self.graph_direct_sb3 = graph_direct

    def _forward_graph_without_preflight(self, obs: Any, *, count_as_production: bool) -> tuple[Any, Any, Any]:
        self._ensure_runner(obs)
        eps = self._external_epsilon(obs)
        if self.eps_spec is not None:
            validate_static_input(self.eps_spec, eps)
        if count_as_production:
            self.replay_count += 1
        else:
            self.preflight_replay_count += 1
        return tuple(self.runner.replay(obs, eps))

    def run_mandatory_preflight(self, obs: Any) -> dict[str, Any]:
        if self.preflight_receipt is not None and self.preflight_receipt.get("status") == "passed":
            return self.preflight_receipt
        device = self._rng_device(obs)
        with self.torch.random.fork_rng(devices=_rng_devices(self.torch, device)):
            setup_before = snapshot_rng_state(self.torch, device)
            self._ensure_runner(obs)
            setup_after = snapshot_rng_state(self.torch, device)
            if not rng_states_equal(self.torch, setup_before, setup_after):
                raise RuntimeError("rollout policy graph setup changed RNG state")
            stochastic = [
                validate_stochastic_forward_exact(self.torch, self, obs)
                for _ in range(int(self.preflight_repeats))
            ]
            deterministic = validate_deterministic_forward_no_rng(self.torch, self, obs)
        self.preflight_receipt = {
            "status": "passed",
            "stochastic_repeats": int(self.preflight_repeats),
            "setup_rng_restored": True,
            "stochastic_exact": stochastic,
            "deterministic_no_rng": deterministic,
        }
        return self.preflight_receipt

    def forward(self, obs: Any, deterministic: bool = False):
        if deterministic:
            self.deterministic_eager_count += 1
            return self.original_forward(obs, deterministic=True)
        if self.preflight_receipt is None:
            self.run_mandatory_preflight(obs)
        return self._forward_graph_without_preflight(obs, count_as_production=True)

    def summary(self) -> dict[str, Any]:
        return {
            "captures": 1 if self.runner is not None else 0,
            "replays": int(self.replay_count),
            "preflight_replays": int(self.preflight_replay_count),
            "deterministic_eager_calls": int(self.deterministic_eager_count),
            "capture_seconds": self.capture_seconds,
            "preflight": self.preflight_receipt,
            "input_spec": dataclasses.asdict(self.input_spec) if self.input_spec is not None else None,
            "eps_spec": dataclasses.asdict(self.eps_spec) if self.eps_spec is not None else None,
            "helper_direct_sb3": getattr(self, "helper_direct_sb3", None),
            "graph_direct_sb3": getattr(self, "graph_direct_sb3", None),
        }


@contextmanager
def temporary_policy_forward(policy: Any, adapter: RolloutPolicyGraphForward) -> Iterator[RolloutPolicyGraphForward]:
    had_instance_forward = "forward" in getattr(policy, "__dict__", {})
    original = policy.forward
    policy.forward = adapter.forward
    try:
        yield adapter
    finally:
        if had_instance_forward:
            policy.forward = original
        else:
            delattr(policy, "forward")


@contextmanager
def policy_forward_graph_context(
    policy: Any,
    *,
    adapter: RolloutPolicyGraphForward | None = None,
    torch: Any | None = None,
    expected_batch_size: int | None = None,
    capture_warmup: int = 3,
    tolerance: float = 1e-5,
) -> Iterator[RolloutPolicyGraphForward]:
    if torch is None:
        import torch as torch_module

        torch = torch_module
    if adapter is None:
        adapter = RolloutPolicyGraphForward(
            policy,
            torch,
            original_forward=policy.forward,
            expected_batch_size=expected_batch_size,
            capture_warmup=capture_warmup,
            tolerance=tolerance,
        )
    with temporary_policy_forward(policy, adapter):
        yield adapter


def validate_stochastic_forward_exact(torch: Any, adapter: RolloutPolicyGraphForward, obs: Any) -> dict[str, Any]:
    device = str(obs.device)
    before = snapshot_rng_state(torch, device)
    restore_rng_state(torch, device, before)
    with torch.no_grad():
        eager = tuple(adapter.original_forward(obs, deterministic=False))
    eager_after = snapshot_rng_state(torch, device)
    restore_rng_state(torch, device, before)
    with torch.no_grad():
        graph = tuple(adapter._forward_graph_without_preflight(obs, count_as_production=False))
    graph_after = snapshot_rng_state(torch, device)
    if not rng_states_equal(torch, eager_after, graph_after):
        raise RuntimeError("CUDA RNG state diverged between eager SB3 forward and graph external-epsilon forward")
    outputs = _tuple_exact_equal(torch, graph, eager)
    return {"passed": True, "outputs": outputs, "cuda_rng_state_exact": True}


def validate_deterministic_forward_no_rng(torch: Any, adapter: RolloutPolicyGraphForward, obs: Any) -> dict[str, Any]:
    device = str(obs.device)
    with torch.random.fork_rng(devices=_rng_devices(torch, device)):
        before = snapshot_rng_state(torch, device)
        with torch.no_grad():
            eager = tuple(adapter.original_forward(obs, deterministic=True))
            graph = tuple(adapter.forward(obs, deterministic=True))
        after = snapshot_rng_state(torch, device)
        if not rng_states_equal(torch, before, after):
            raise RuntimeError("deterministic rollout policy graph path advanced RNG state")
        outputs = _tuple_exact_equal(torch, graph, eager)
    return {"passed": True, "outputs": outputs, "rng_unchanged": True}


class RolloutPolicyGraphRuntimeTests(unittest.TestCase):
    """CUDA/SB3 runtime contract tests for root's GPU runner.

    They intentionally skip on local machines without the exact torch/SB3 GPU
    stack, but they are real assertions when CUDA is available.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            from stable_baselines3 import PPO
        except Exception as exc:  # pragma: no cover - local dependency gate
            raise unittest.SkipTest(f"runtime CUDA stack unavailable: {exc}") from exc
        if not torch.cuda.is_available():  # pragma: no cover - local dependency gate
            raise unittest.SkipTest("CUDA unavailable")
        cls.torch = torch
        cls.PPO = PPO
        cls.device = "cuda:0"
        cls.checkpoint = os.environ.get("ROLLOUT_POLICY_GRAPH_CHECKPOINT", str(e4.DEFAULT_CHECKPOINT))
        torch.cuda.set_device(0)
        torch.empty(1, device=cls.device)

    def make_adapter(self):
        policy = e4.load_policy(self.torch, self.PPO, self.checkpoint, self.device)
        adapter = RolloutPolicyGraphForward(policy, self.torch, expected_batch_size=64)
        obs = self.torch.randn(64, 256, dtype=self.torch.float32, device=self.device)
        return policy, adapter, obs

    def test_setup_preserves_rng_state(self):
        _policy, adapter, obs = self.make_adapter()
        before = snapshot_rng_state(self.torch, self.device)
        adapter._ensure_runner(obs)
        after = snapshot_rng_state(self.torch, self.device)
        self.assertTrue(rng_states_equal(self.torch, before, after))

    def test_repeated_stochastic_forward_matches_sb3_and_cuda_rng(self):
        _policy, adapter, obs = self.make_adapter()
        for _ in range(3):
            receipt = validate_stochastic_forward_exact(self.torch, adapter, obs)
            self.assertTrue(receipt["passed"])
            self.assertTrue(receipt["cuda_rng_state_exact"])

    def test_deterministic_forward_does_not_advance_rng(self):
        _policy, adapter, obs = self.make_adapter()
        receipt = validate_deterministic_forward_no_rng(self.torch, adapter, obs)
        self.assertTrue(receipt["passed"])
        self.assertTrue(receipt["rng_unchanged"])

    def test_weight_update_visible_without_storage_change(self):
        policy, adapter, obs = self.make_adapter()
        adapter._ensure_runner(obs)
        eps = adapter._zero_epsilon(obs)
        before_signature = graph_core.parameter_storage_signature(policy)
        before_outputs = adapter.runner.replay(obs, eps)
        state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        try:
            named_parameters = dict(policy.named_parameters())
            target = named_parameters["action_net.bias"]
            with self.torch.no_grad():
                target.add_(0.125)
            self.assertEqual(before_signature, graph_core.parameter_storage_signature(policy))
            after_outputs = adapter.runner.replay(obs, eps)
            with self.torch.no_grad():
                eager_after = tuple(adapter.helper(obs, eps))
            graph_core.tuple_parity(self.torch, after_outputs, eager_after, tolerance=adapter.tolerance)
            self.assertGreater(e4.max_abs_diff(self.torch, before_outputs[0], after_outputs[0]), 0.0)
        finally:
            policy.load_state_dict(state)

    def test_log_std_update_with_zero_epsilon_changes_logprob_only(self):
        policy, adapter, obs = self.make_adapter()
        adapter._ensure_runner(obs)
        eps = adapter._zero_epsilon(obs)
        before_signature = graph_core.parameter_storage_signature(policy)
        before_outputs = adapter.runner.replay(obs, eps)
        state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        try:
            named_parameters = dict(policy.named_parameters())
            target = named_parameters["log_std"]
            with self.torch.no_grad():
                target.add_(0.125)
            self.assertEqual(before_signature, graph_core.parameter_storage_signature(policy))
            after_outputs = adapter.runner.replay(obs, eps)
            with self.torch.no_grad():
                eager_after = tuple(adapter.helper(obs, eps))
            graph_core.tuple_parity(self.torch, after_outputs, eager_after, tolerance=adapter.tolerance)
            actions_diff = e4.max_abs_diff(self.torch, before_outputs[0], after_outputs[0])
            values_diff = e4.max_abs_diff(self.torch, before_outputs[1], after_outputs[1])
            log_prob_diff = e4.max_abs_diff(self.torch, before_outputs[2], after_outputs[2])
            self.assertEqual(actions_diff, 0.0)
            self.assertEqual(values_diff, 0.0)
            self.assertGreater(log_prob_diff, 0.0)
        finally:
            policy.load_state_dict(state)

    def test_outputs_are_lifetime_safe(self):
        _policy, adapter, obs = self.make_adapter()
        adapter._ensure_runner(obs)
        eps = adapter._zero_epsilon(obs)
        first = adapter.runner.replay(obs, eps)
        snapshot = e4.clone_output_tuple(first)
        second = adapter.runner.replay(obs + 0.01, eps)
        e4.assert_no_cross_call_storage_alias(first, second, first_label="first", second_label="second")
        e4.assert_output_snapshot_unchanged(self.torch, snapshot, first, label="rollout_graph")

    def test_policy_forward_restores_after_exception(self):
        policy, adapter, _obs = self.make_adapter()
        original = policy.forward
        with self.assertRaisesRegex(RuntimeError, "restore-test"):
            with temporary_policy_forward(policy, adapter):
                raise RuntimeError("restore-test")
        self.assertNotIn("forward", policy.__dict__)
        self.assertEqual(policy.forward.__func__, original.__func__)

    def test_ppo_save_load_inside_context(self):
        import tempfile

        model = self.PPO.load(str(self.checkpoint), env=None, device=self.device)
        adapter = RolloutPolicyGraphForward(model.policy, self.torch, expected_batch_size=64)
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/context-save.zip"
            with temporary_policy_forward(model.policy, adapter):
                model.save(path)
                loaded = self.PPO.load(path, env=None, device=self.device)
        self.assertEqual(type(loaded.policy).__name__, type(model.policy).__name__)
