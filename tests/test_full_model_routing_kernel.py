"""Dependency-light CPU checks of the real routing/fusion kernel source.

These checks do NOT import or validate the complete architecture/registry.
The real core.py is imported directly, and only the custom autograd class AST
is executed from full_expert_network.py. This avoids importing Restormer's
einops dependency, without stubbing it or substituting its implementation.

Run after placing this file under repository tests/:
    python -B -m unittest discover -s tests -p test_full_model_routing_kernel.py -v

Both sources are always read from this test file's repository root. There is
no dependency stub, copied kernel, environment override, or Temp fallback.
"""

import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE = ROOT / "basicsr/models/archs/unirestorerpp/core.py"
ARCH_SOURCE = ROOT / "basicsr/models/archs/unirestorerpp/full_expert_network.py"
spec = importlib.util.spec_from_file_location("unirestorerpp_real_core_kernel_test", CORE_SOURCE)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def load_exact_source_class():
    tree = ast.parse(ARCH_SOURCE.read_text(encoding="utf-8"), filename=str(ARCH_SOURCE))
    nodes = [node for node in tree.body
             if isinstance(node, ast.ClassDef) and node.name == "_ExactExpertWithResidualGradient"]
    if len(nodes) != 1:
        raise AssertionError("the real custom autograd class must occur exactly once")
    isolated = ast.Module(body=nodes, type_ignores=[])
    namespace = {"torch": torch, "Tensor": Tensor}
    exec(compile(isolated, str(ARCH_SOURCE), "exec"), namespace)
    return namespace[nodes[0].name]


ExactExpert = load_exact_source_class()


class Zeros(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return x.new_zeros(x.shape[0], self.channels, *x.shape[-2:])


class Logits(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(value)

    def forward(self, x):
        return self.value.expand(x.shape[0], -1, -1, -1)


def controlled_router(height=3, width=4, noise=False):
    hierarchy = core.HierarchyIndex.perfect_binary_tree(2)
    der = torch.randn(1, 4, height, width, dtype=torch.double)
    ger = torch.randn(1, 3, height, width, dtype=torch.double)
    return core.DensePriorRouter(Zeros(4), Logits(der), Zeros(3), Logits(ger),
                                 hierarchy, router_noise=noise)


class RoutingKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(302)

    def test_custom_function_exact_value_despite_cancellation(self):
        base = torch.full((2, 3, 5, 7), 1e20, requires_grad=True)
        expert = torch.randn_like(base, requires_grad=True)
        gate = torch.ones(2, 1, 5, 7, requires_grad=True)
        output = ExactExpert.apply(base, expert, gate)
        self.assertTrue(torch.equal(output, expert))
        self.assertFalse(torch.equal(base + gate * (expert - base), expert))

    def test_backward_matches_old_surrogate_on_noncontiguous_rgb_crops(self):
        base = torch.randn(2, 3, 5, 9, dtype=torch.double)[..., :7].requires_grad_()
        expert = torch.randn(2, 3, 5, 9, dtype=torch.double)[..., :7].requires_grad_()
        gate = torch.ones(2, 1, 5, 7, dtype=torch.double, requires_grad=True)
        cotangent = torch.randn_like(expert)
        self.assertFalse(base.is_contiguous())
        output = ExactExpert.apply(base, expert, gate)
        actual = torch.autograd.grad((output * cotangent).sum(), (base, expert, gate))
        surrogate = base + gate * (expert - base)
        expected = torch.autograd.grad((surrogate * cotangent).sum(), (base, expert, gate))
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.equal(actual[0], torch.zeros_like(base)))
        self.assertTrue(torch.equal(actual[1], cotangent))

    def test_depth_two_ancestor_table_and_disjoint_root_inclusive_masks(self):
        tree = core.HierarchyIndex.perfect_binary_tree(2)
        leaves = torch.arange(4).view(1, 1, 4).expand(1, 3, 4)
        levels = torch.arange(3).view(1, 3, 1).expand(1, 3, 4)
        der = torch.zeros(1, 4, 3, 4).scatter_(1, leaves.unsqueeze(1), 9)
        ger = torch.zeros(1, 3, 3, 4).scatter_(1, levels.unsqueeze(1), 9)
        result = tree.hard_top1_assignment(der, ger)
        expected_groups = torch.tensor([[[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 2, 3]]])
        expected_experts = torch.tensor([[[-1, -1, -1, -1], [0, 0, 1, 1], [2, 3, 4, 5]]])
        self.assertTrue(torch.equal(result.leaf_indices, leaves))
        self.assertTrue(torch.equal(result.level_indices, levels))
        self.assertTrue(torch.equal(result.group_indices, expected_groups))
        self.assertTrue(torch.equal(result.expert_ids, expected_experts))
        self.assertEqual(result.masks.dtype, torch.bool)
        all_masks = torch.cat((result.expert_ids.unsqueeze(1) == -1, result.masks), dim=1)
        self.assertTrue(torch.equal(all_masks.sum(1), torch.ones(1, 3, 4, dtype=torch.long)))
        self.assertEqual(set(result.expert_ids.unique().tolist()), set(range(-1, 6)))

    def test_depth_zero_contains_only_root(self):
        tree = core.HierarchyIndex.perfect_binary_tree(0)
        output = tree(torch.randn(2, 1, 3, 5), torch.randn(2, 1, 3, 5))
        self.assertEqual(tree.num_experts, 0)
        self.assertEqual(tree.num_leaves, 1)
        self.assertEqual(tree.num_levels, 1)
        self.assertEqual(output.masks.shape, (2, 0, 3, 5))
        self.assertTrue(bool((output.expert_ids == -1).all()))

    def test_router_interpolates_only_dense_priors_to_feature_grid(self):
        tree = core.HierarchyIndex.perfect_binary_tree(2)
        modules = (nn.Conv2d(3, 4, 1), nn.Conv2d(2, 4, 1),
                   nn.Conv2d(3, 3, 1), nn.Conv2d(2, 3, 1))
        router = core.DensePriorRouter(*modules, tree, router_noise=False).double()
        h = torch.randn(2, 2, 5, 7, dtype=torch.double)
        e_d = torch.randn(2, 3, 2, 3, dtype=torch.double)
        e_g = torch.randn(2, 3, 2, 3, dtype=torch.double)
        der, ger = router.compute_logits(h, e_d, e_g)
        resized_d = F.interpolate(e_d, size=(5, 7), mode="bilinear", align_corners=False)
        resized_g = F.interpolate(e_g, size=(5, 7), mode="bilinear", align_corners=False)
        torch.testing.assert_close(der, modules[0](resized_d) + modules[1](h))
        torch.testing.assert_close(ger, modules[2](resized_g) + modules[3](h))
        output = router(h, e_d, e_g)
        self.assertEqual(output.masks.shape[-2:], (5, 7))
        self.assertEqual(output.masks.dtype, torch.bool)

    def test_probability_gate_forward_one_and_selected_log_probability_gradient(self):
        router = controlled_router(noise=False)
        h = torch.randn(1, 2, 3, 4, dtype=torch.double)
        output = router(h, h, h)
        self.assertTrue(torch.equal(output.normalized_gate, torch.ones_like(output.normalized_gate)))
        output.normalized_gate.sum().backward()
        for branch, indices in ((router.der_feature, output.leaf_indices),
                                (router.ger_feature, output.level_indices)):
            expected = torch.zeros_like(branch.value).scatter_(1, indices.unsqueeze(1), 1)
            expected = expected - branch.value.detach().softmax(dim=1)
            torch.testing.assert_close(branch.value.grad, expected, rtol=1e-12, atol=1e-12)

    def test_noise_scales_and_evaluation_disable_noise(self):
        router = controlled_router(noise=True)
        h = torch.randn(1, 2, 3, 4, dtype=torch.double)
        der, ger = router.compute_logits(h, h, h)
        der_noise = torch.randn_like(der)
        ger_noise = torch.randn_like(ger)
        router.train()
        with mock.patch.object(core.torch, "randn_like", side_effect=[der_noise, ger_noise]) as draw:
            noisy = router(h, h, h)
        self.assertEqual(draw.call_count, 2)
        expected_der = der + der_noise / 4
        expected_ger = ger + ger_noise / 3
        self.assertTrue(torch.equal(noisy.leaf_indices, expected_der.argmax(1)))
        self.assertTrue(torch.equal(noisy.level_indices, expected_ger.argmax(1)))
        torch.testing.assert_close(noisy.der_selected_probability,
                                   expected_der.softmax(1).gather(1, noisy.leaf_indices.unsqueeze(1)))
        torch.testing.assert_close(noisy.ger_selected_probability,
                                   expected_ger.softmax(1).gather(1, noisy.level_indices.unsqueeze(1)))
        router.eval()
        with mock.patch.object(core.torch, "randn_like", side_effect=AssertionError("eval must be deterministic")):
            clean = router(h, h, h)
        self.assertTrue(torch.equal(clean.leaf_indices, der.argmax(1)))
        self.assertTrue(torch.equal(clean.level_indices, ger.argmax(1)))
        router.train()
        router.router_noise = False
        with mock.patch.object(core.torch, "randn_like", side_effect=AssertionError("noise explicitly disabled")):
            router(h, h, h)

    def test_masked_full_output_kernel_matches_original_delta_gradients(self):
        """Fusion kernel only; these parameter tensors are not Restormer models."""
        router = controlled_router(height=3, width=7, noise=False)
        leaves = torch.tensor([0, 0, 2, 0, 1, 2, 3]).view(1, 1, 7).expand(1, 3, 7)
        levels = torch.tensor([0, 1, 1, 2, 2, 2, 2]).view(1, 1, 7).expand(1, 3, 7)
        with torch.no_grad():
            router.der_feature.value.zero_().scatter_(1, leaves.unsqueeze(1), 3.2)
            router.ger_feature.value.zero_().scatter_(1, levels.unsqueeze(1), 2.3)
        h = torch.randn(1, 2, 3, 7, dtype=torch.double)
        base = torch.randn(1, 3, 3, 7, dtype=torch.double, requires_grad=True)
        experts = [torch.randn_like(base, requires_grad=True) for _ in range(6)]
        decision = router(h, h, h)
        output = base
        reference = base
        for index, expert in enumerate(experts):
            mask = decision.masks[:, index:index + 1]
            output = torch.where(mask, ExactExpert.apply(base, expert, decision.normalized_gate), output)
            reference = reference + mask * decision.normalized_gate * (expert - base)
        expected_hard = base.detach()
        for index, expert in enumerate(experts):
            expected_hard = torch.where(decision.masks[:, index:index + 1], expert.detach(), expected_hard)
        self.assertTrue(torch.equal(output, expected_hard))
        cotangent = torch.randn_like(output)
        parameters = [base, *experts, router.der_feature.value, router.ger_feature.value]
        actual_gradients = torch.autograd.grad((output * cotangent).sum(), parameters, retain_graph=True)
        expected_gradients = torch.autograd.grad((reference * cotangent).sum(), parameters)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        # Root pixels use base only and do not acquire this reconstruction proxy.
        root = (decision.expert_ids == -1).unsqueeze(1)
        torch.testing.assert_close(actual_gradients[-2] * root,
                                   torch.zeros_like(actual_gradients[-2]), rtol=0, atol=0)
        torch.testing.assert_close(actual_gradients[-1] * root,
                                   torch.zeros_like(actual_gradients[-1]), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
