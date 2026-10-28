"""CUDA-only parity tests for the partial MLP extractor graph helper."""

from __future__ import annotations

import copy
import importlib
import inspect
import unittest


def _dependencies():
    try:
        import torch
        from stable_baselines3.common.torch_layers import MlpExtractor
    except ImportError as exc:
        raise unittest.SkipTest(f"PPO dependencies unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA unavailable; MLP graph capture is GPU-only")
    try:
        import ai2.training.cuda_mlp_graph as cuda_mlp_graph
    except ImportError as exc:
        raise AssertionError("cuda_mlp_graph module must exist") from exc
    return torch, MlpExtractor, cuda_mlp_graph


class CudaMlpGraphInterfaceTests(unittest.TestCase):
    def test_context_manager_interface_is_available(self):
        module = importlib.import_module('ai2.training.cuda_mlp_graph')

        helper = module.graph_mlp_extractor
        signature = inspect.signature(helper)

        self.assertIn("mlp_extractor", signature.parameters)
        self.assertIn("batch_size", signature.parameters)
        self.assertIn("features_dim", signature.parameters)


class CudaMlpGraphTests(unittest.TestCase):
    def setUp(self):
        self.torch, self.MlpExtractor, self.graphs = _dependencies()
        self.torch.manual_seed(20260913)

    def make_module(self):
        torch = self.torch
        return self.MlpExtractor(
            feature_dim=4,
            net_arch={"pi": [8, 3], "vf": [8, 2]},
            activation_fn=torch.nn.Tanh,
            device="cuda",
        ).train()

    def assert_state_close(self, left, right):
        for name, tensor in left.items():
            self.torch.testing.assert_close(tensor, right[name], rtol=0, atol=0)

    def clone_grads(self, module):
        return {
            name: None if param.grad is None else param.grad.detach().clone()
            for name, param in module.named_parameters()
        }

    def assert_same_bound_method(self, left, right):
        self.assertIs(getattr(left, "__self__", None), getattr(right, "__self__", None))
        self.assertIs(getattr(left, "__func__", left), getattr(right, "__func__", right))

    def test_graph_matches_forward_backward_and_optimizer_step(self):
        torch = self.torch
        eager = self.make_module()
        graphed = copy.deepcopy(eager)
        eager_opt = torch.optim.Adam(eager.parameters(), lr=0.003, eps=1e-5)
        graph_opt = torch.optim.Adam(graphed.parameters(), lr=0.003, eps=1e-5)
        inputs = torch.randn(4, 4, device="cuda")

        with self.graphs.graph_mlp_extractor(graphed, batch_size=4, features_dim=4) as graph:
            self.assertGreaterEqual(graph.capture_seconds, 0.0)
            for _ in range(3):
                eager_opt.zero_grad()
                graph_opt.zero_grad()
                eager_loss = sum(t.square().mean() for t in eager(inputs))
                graph_loss = sum(t.square().mean() for t in graphed(inputs))
                eager_loss.backward()
                graph_loss.backward()
                eager_opt.step()
                graph_opt.step()

                self.torch.testing.assert_close(graph_loss, eager_loss, rtol=0, atol=0)
                self.assert_state_close(graphed.state_dict(), eager.state_dict())

        self.assert_state_close(graphed.state_dict(), eager.state_dict())

    def test_input_gradients_match_eager_when_inputs_require_grad(self):
        torch = self.torch
        eager = self.make_module()
        graphed = copy.deepcopy(eager)
        eager_input = torch.randn(4, 4, device="cuda", requires_grad=True)
        graph_input = eager_input.detach().clone().requires_grad_(True)

        with self.graphs.graph_mlp_extractor(graphed, batch_size=4, features_dim=4, requires_grad=True):
            eager_loss = sum(t.sum() for t in eager(eager_input))
            graph_loss = sum(t.sum() for t in graphed(graph_input))
            eager_loss.backward()
            graph_loss.backward()

        self.torch.testing.assert_close(graph_input.grad, eager_input.grad, rtol=0, atol=0)
        for name, grad in self.clone_grads(graphed).items():
            self.torch.testing.assert_close(grad, self.clone_grads(eager)[name], rtol=0, atol=0)

    def test_eval_no_grad_wrong_shape_and_stride_fall_back_to_eager_forward(self):
        torch = self.torch
        module = self.make_module()
        calls = {"original": 0}
        original = module.forward

        def counted_forward(features):
            calls["original"] += 1
            return original(features)

        module.forward = counted_forward
        with self.graphs.graph_mlp_extractor(module, batch_size=4, features_dim=4):
            calls["original"] = 0
            module(torch.randn(4, 4, device="cuda"))
            self.assertEqual(calls["original"], 0)

            module.eval()
            module(torch.randn(4, 4, device="cuda"))
            self.assertEqual(calls["original"], 1)

            module.train()
            with torch.no_grad():
                module(torch.randn(4, 4, device="cuda"))
            self.assertEqual(calls["original"], 2)

            module(torch.randn(2, 4, device="cuda"))
            self.assertEqual(calls["original"], 3)

            non_contiguous = torch.randn(4, 8, device="cuda")[:, ::2]
            module(non_contiguous)
            self.assertEqual(calls["original"], 4)

    def test_context_restores_forward_and_training_mode_after_exception(self):
        module = self.make_module()
        original_forward = module.forward
        module.eval()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with self.graphs.graph_mlp_extractor(module, batch_size=4, features_dim=4):
                raise RuntimeError("boom")
        self.assert_same_bound_method(module.forward, original_forward)
        self.assertNotIn("forward", module.__dict__)
        self.assertFalse(module.training)

    def test_rejects_unsupported_modules_before_forward_mutation(self):
        torch = self.torch
        unsupported = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout()).cuda()
        original_forward = unsupported.forward

        with self.assertRaisesRegex(ValueError, "unsupported"):
            with self.graphs.graph_mlp_extractor(unsupported, batch_size=4, features_dim=4):
                pass
        self.assert_same_bound_method(unsupported.forward, original_forward)
        self.assertNotIn("forward", unsupported.__dict__)

    def test_rng_state_and_checkpoint_reload_are_unchanged(self):
        torch = self.torch
        module = self.make_module()
        reference = copy.deepcopy(module)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()

        with self.graphs.graph_mlp_extractor(module, batch_size=4, features_dim=4):
            pass

        self.torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        self.torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)

        fresh = self.make_module()
        fresh.load_state_dict(module.state_dict())
        inputs = torch.randn(4, 4, device="cuda")
        expected = reference(inputs)
        actual = fresh(inputs)
        for left, right in zip(actual, expected):
            self.torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_random_draws_inside_context_are_not_rewound_on_exit(self):
        torch = self.torch
        module = self.make_module()
        baseline = copy.deepcopy(module)
        torch.manual_seed(77)
        before_cpu = torch.get_rng_state()
        before_cuda = torch.cuda.get_rng_state()

        with self.graphs.graph_mlp_extractor(module, batch_size=4, features_dim=4):
            inside_cpu = torch.rand(3)
            inside_cuda = torch.rand(3, device="cuda")
        after_actual_cpu = torch.get_rng_state()
        after_actual_cuda = torch.cuda.get_rng_state()

        torch.set_rng_state(before_cpu)
        torch.cuda.set_rng_state(before_cuda)
        with self.graphs.graph_mlp_extractor(baseline, batch_size=4, features_dim=4):
            expected_cpu = torch.rand(3)
            expected_cuda = torch.rand(3, device="cuda")
        after_expected_cpu = torch.get_rng_state()
        after_expected_cuda = torch.cuda.get_rng_state()

        self.torch.testing.assert_close(inside_cpu, expected_cpu, rtol=0, atol=0)
        self.torch.testing.assert_close(inside_cuda, expected_cuda, rtol=0, atol=0)
        self.torch.testing.assert_close(after_actual_cpu, after_expected_cpu, rtol=0, atol=0)
        self.torch.testing.assert_close(after_actual_cuda, after_expected_cuda, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
