"""CPU checks for the independent full-model expert architecture.

Run from the repository root with an environment providing its declared deps:
    python -B -m unittest discover -s tests -p test_unirestorerpp_full_expert.py -v

Only ``test_real_dense_prior_native_restormer_smoke`` uses the actual ResNet18 D.
Other tests replace only D's encoder with a small Conv/BN encoder to isolate
routing/fusion cheaply; distribution heads and DensePriorRouter stay real.
Controlled-output tests explicitly replace complete restoration models with
affine functions. They validate fusion and gradients, not Restormer quality.
"""

from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.models.archs import build_network
from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.archs.unirestorerpp.core import DensePriorRouter, RoutingDecision
from basicsr.models.archs.unirestorerpp.full_expert_network import UniRestorerPPFullModel
from basicsr.utils.registry import ARCH_REGISTRY


def config(**overrides):
    result = dict(tree_depth=1, dim=2, num_blocks=(1, 1, 1, 1),
                  num_refinement_blocks=1, heads=(1, 1, 1, 1),
                  ffn_expansion_factor=1.5, p=2, router_hidden_channels=2,
                  router_noise=False)
    result.update(overrides)
    return result


class TinyEncoder(nn.Module):
    """Test-only D replacement; actual D is exercised by a separate smoke."""

    def __init__(self, p=2, output_stride=8):
        super().__init__()
        self.conv = nn.Conv2d(3, p, 3, 2, 1)
        self.bn = nn.BatchNorm2d(p)

    def forward(self, x):
        return self.bn(self.conv(x))


def make_model(**overrides):
    with mock.patch("basicsr.models.archs.unirestorerpp.dense_prior.DenseResNet18Encoder", TinyEncoder):
        return UniRestorerPPFullModel(**config(**overrides))


class ZeroLogits(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return x.new_zeros(x.shape[0], self.channels, *x.shape[-2:])


class SpatialLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits)

    def forward(self, x):
        if x.shape[-2:] != self.logits.shape[-2:]:
            raise AssertionError("router features must retain the original pixel grid")
        return self.logits.expand(x.shape[0], -1, -1, -1)


class AffineImage(nn.Module):
    """A complete image-to-image test function, including its full output."""

    def __init__(self, scale, bias):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.seen_inputs = []

    def forward(self, x):
        self.seen_inputs.append(x.detach().clone())
        return self.scale * x + self.bias


def pattern_routes(model, height, width, root_only=False):
    """Visit root, both internal nodes and all four leaves of depth two."""
    if model.tree_depth != 2:
        raise AssertionError("the controlled seven-node pattern needs depth two")
    count = height * width
    leaves = torch.tensor([0, 0, 2, 0, 1, 2, 3]).repeat((count + 6) // 7)[:count]
    levels = torch.tensor([0, 1, 1, 2, 2, 2, 2]).repeat((count + 6) // 7)[:count]
    if root_only:
        levels.zero_()
    leaves = leaves.view(1, height, width)
    levels = levels.view(1, height, width)
    der = torch.zeros(1, 4, height, width).scatter_(1, leaves.unsqueeze(1), 3.2)
    ger = torch.zeros(1, 3, height, width).scatter_(1, levels.unsqueeze(1), 2.3)
    model.router.der_prior = ZeroLogits(4)
    model.router.ger_prior = ZeroLogits(3)
    model.router.der_feature = SpatialLogits(der)
    model.router.ger_feature = SpatialLogits(ger)


def controlled_model(height=3, width=7, freeze_base=True, root_only=False):
    model = make_model(tree_depth=2, freeze_base=freeze_base)
    model.base_model = AffineImage(0.3, -0.4).requires_grad_(not freeze_base)
    model.experts = nn.ModuleList(AffineImage(0.7 + index / 7, 0.1 + index / 11)
                                 for index in range(6))
    pattern_routes(model, height, width, root_only)
    model.train()
    return model


def pad_like_native_wrapper(x):
    height, width = x.shape[-2:]
    pad_h, pad_w = (-height) % 8, (-width) % 8
    mode = "reflect" if pad_h < height and pad_w < width else "replicate"
    if pad_h or pad_w:
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x


def force_nonroot_zero(model):
    with torch.no_grad():
        for branch in (model.router.der_prior, model.router.der_feature,
                       model.router.ger_prior, model.router.ger_feature):
            for parameter in branch.parameters():
                parameter.zero_()
        model.router.der_feature.project_out.bias[0] = 5
        model.router.ger_feature.project_out.bias[1] = 5


class FullModelExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(1123)

    def test_explicit_registry_entries_and_old_alias_state_compatibility(self):
        for name in ("UniRestorerPP", "UniRestorerPPLoRA", "UniRestorerPPFullModel"):
            self.assertIn(name, ARCH_REGISTRY)
        old_config = config(rank=1)
        with mock.patch("basicsr.models.archs.unirestorerpp.dense_prior.DenseResNet18Encoder", TinyEncoder):
            torch.manual_seed(13)
            old = build_network(dict(type="UniRestorerPP", **old_config)).eval()
            torch.manual_seed(13)
            alias = build_network(dict(type="UniRestorerPPLoRA", **old_config)).eval()
            full_options = dict(type="UniRestorerPPFullModel", **config())
            unchanged_options = deepcopy(full_options)
            full = build_network(full_options).eval()
        self.assertIsInstance(full, UniRestorerPPFullModel)
        self.assertEqual(full_options, unchanged_options)
        self.assertEqual(tuple(old.state_dict()), tuple(alias.state_dict()))
        alias.load_state_dict(old.state_dict(), strict=True)
        x = torch.randn(1, 3, 9, 11)
        with torch.no_grad():
            old_y, old_aux = old(x, return_aux=True)
            alias_y = alias(x)
        self.assertTrue(torch.equal(old_y, alias_y))
        self.assertEqual(len(old_aux["routing"]), 8)

    def test_independent_complete_parameters_and_counts(self):
        model = make_model(tree_depth=2)
        owners = [model.base_model, *model.experts]
        self.assertEqual(len(owners), 7)
        self.assertTrue(all(isinstance(owner, Restormer) for owner in owners))
        parameter_ids = [{id(value) for value in owner.parameters()} for owner in owners]
        parameter_storages = [{value.data_ptr() for value in owner.parameters()} for owner in owners]
        for left in range(len(owners)):
            for right in range(left + 1, len(owners)):
                self.assertTrue(parameter_ids[left].isdisjoint(parameter_ids[right]))
                self.assertTrue(parameter_storages[left].isdisjoint(parameter_storages[right]))
        for expert in model.experts:
            for key, value in model.base_model.state_dict().items():
                self.assertTrue(torch.equal(value, expert.state_dict()[key]))
            self.assertTrue(all(value.requires_grad for value in expert.parameters()))
        self.assertTrue(all(not value.requires_grad for value in model.base_model.parameters()))
        before = next(model.base_model.parameters()).detach().clone()
        other_before = next(model.experts[1].parameters()).detach().clone()
        with torch.no_grad():
            next(model.experts[0].parameters()).add_(1)
        self.assertTrue(torch.equal(before, next(model.base_model.parameters())))
        self.assertTrue(torch.equal(other_before, next(model.experts[1].parameters())))
        self.assertFalse(any("lora" in name.lower() for name, _ in model.named_parameters()))
        base_count = sum(value.numel() for value in model.base_model.parameters())
        expert_count = sum(value.numel() for value in model.experts.parameters())
        counts = model.parameter_counts()
        self.assertEqual(counts["base"], dict(total=base_count, trainable=0))
        self.assertEqual(counts["experts"], dict(total=expert_count, trainable=expert_count))
        self.assertEqual(counts["total"], dict(total=base_count + expert_count, trainable=expert_count))

    def test_learnable_stem_is_separate_and_preserves_original_grid(self):
        model = make_model(router_feature_channels=5)
        x = torch.randn(1, 3, 9, 11, requires_grad=True)
        features = model.router_stem(x)
        self.assertEqual(features.shape, (1, 5, 9, 11))
        features.square().mean().backward()
        self.assertTrue(any(value.grad is not None and bool(value.grad.abs().sum() > 0)
                            for value in model.router_stem.parameters()))
        stem_ids = {id(value) for value in model.router_stem.parameters()}
        restoration_ids = {id(value) for value in model.base_model.parameters()}
        restoration_ids.update(id(value) for value in model.experts.parameters())
        self.assertTrue(stem_ids.isdisjoint(restoration_ids))
        default_model = make_model()
        self.assertEqual(default_model.router_stem(x.detach()).shape[1], 2)

    def test_every_tree_node_exact_hard_splice_and_complete_input(self):
        height, width = 7, 9
        model = controlled_model(height, width).eval()
        # An arithmetic base + (expert-base) loses the small selected value here.
        # Exact equality therefore detects accidental floating-point cancellation.
        with torch.no_grad():
            model.base_model.bias.fill_(1e20)
        x = torch.randn(2, 3, height, width)
        calls = []
        handle = model.router.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            actual, aux = model(x, return_aux=True)
        handle.remove()
        self.assertEqual(len(calls), 1)
        decision = aux["routing"]
        self.assertIsInstance(decision, RoutingDecision)
        self.assertEqual(set(decision.expert_ids.unique().tolist()), set(range(-1, 6)))
        self.assertEqual(decision.masks.dtype, torch.bool)
        root_mask = decision.expert_ids.unsqueeze(1) == -1
        masks = torch.cat((root_mask, decision.masks), 1)
        self.assertTrue(torch.equal(masks.sum(1), torch.ones_like(decision.expert_ids)))
        self.assertEqual(tuple(decision.expert_ids.shape), (2, height, width))
        if "hard_masks" in aux:
            self.assertTrue(torch.equal(aux["hard_masks"], masks))
        expected = model.base_model.scale * x + model.base_model.bias
        for index, expert in enumerate(model.experts):
            expected = torch.where(decision.masks[:, index:index + 1],
                                   expert.scale * x + expert.bias, expected)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(tuple(actual.shape), tuple(x.shape))
        padded = pad_like_native_wrapper(x)
        for image_model in (model.base_model, *model.experts):
            self.assertEqual(len(image_model.seen_inputs), 1)
            self.assertTrue(torch.equal(image_model.seen_inputs[0], padded))
        self.assertEqual(aux["input_size"], (height, width))
        self.assertEqual(aux["padded_size"], (8, 16))

    def test_surrogate_backward_matches_original_base_delta_expression(self):
        height, width = 3, 7
        model = controlled_model(height, width, freeze_base=False).double()
        x = torch.randn(1, 3, height, width, dtype=torch.double, requires_grad=True)
        cotangent = torch.randn_like(x)
        owners = (model.base_model, *model.experts,
                  model.router.der_feature, model.router.ger_feature)
        parameters = [value for owner in owners for value in owner.parameters()]
        actual = model(x)
        actual_gradients = torch.autograd.grad((actual * cotangent).sum(), [x, *parameters])

        features = model.router_stem(x)
        prior = model.degradation_prior.compute_dr(x)
        e_d, e_g = model.degradation_prior.compute_distribution(prior)
        decision = model.router(features, e_d, e_g)
        base = model.base_model(x)
        reference = base
        for index, expert in enumerate(model.experts):
            reference = reference + decision.masks[:, index:index + 1] * decision.normalized_gate * (expert(x) - base)
        reference_gradients = torch.autograd.grad((reference * cotangent).sum(), [x, *parameters])
        for actual_gradient, reference_gradient in zip(actual_gradients, reference_gradients):
            torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-10, atol=1e-11)
        self.assertGreater(float(actual_gradients[-2].abs().sum()), 0)
        self.assertGreater(float(actual_gradients[-1].abs().sum()), 0)
        self.assertFalse(decision.leaf_indices.requires_grad)
        self.assertFalse(decision.masks.requires_grad)
        self.assertTrue(torch.equal(decision.normalized_gate, torch.ones_like(decision.normalized_gate)))

    def test_frozen_base_and_dr_with_trainable_expert_heads_and_router(self):
        model = controlled_model(freeze_base=True)
        self.assertFalse(model.base_model.training)
        self.assertFalse(model.degradation_prior.dr_encoder.training)
        self.assertTrue(model.router.training)
        self.assertTrue(all(expert.training for expert in model.experts))
        self.assertTrue(all(not value.requires_grad for value in model.base_model.parameters()))
        self.assertTrue(all(not value.requires_grad for value in model.degradation_prior.dr_encoder.parameters()))
        self.assertTrue(all(value.requires_grad for value in model.degradation_prior.phi_mu.parameters()))
        self.assertTrue(all(value.requires_grad for value in model.degradation_prior.phi_rawvar.parameters()))
        model(torch.randn(1, 3, 3, 7)).square().mean().backward()
        self.assertTrue(all(value.grad is None for value in model.base_model.parameters()))
        self.assertTrue(all(value.grad is None for value in model.degradation_prior.dr_encoder.parameters()))
        self.assertTrue(all(expert.scale.grad is not None for expert in model.experts))
        self.assertGreater(float(model.router.der_feature.logits.grad.abs().sum()), 0)
        self.assertGreater(float(model.router.ger_feature.logits.grad.abs().sum()), 0)

    def test_native_backward_reaches_stem_and_heads_but_not_frozen_d(self):
        """Real small Restormers/routers/heads, with only D's encoder replaced."""
        model = make_model().train()
        with torch.no_grad():
            model.experts[0].output.weight.add_(0.07)
            # Keep feature/prior weights intact while selecting a stable node.
            model.router.der_feature.project_out.bias[0].add_(8)
            model.router.ger_feature.project_out.bias[1].add_(8)
        output, aux = model(torch.randn(1, 3, 9, 11), return_aux=True)
        self.assertTrue(bool((aux["routing"].expert_ids == 0).all()))
        output.square().mean().backward()
        for module in (model.router_stem, model.degradation_prior.phi_mu,
                       model.degradation_prior.phi_rawvar, model.router):
            self.assertTrue(any(value.grad is not None and bool(value.grad.abs().sum() > 0)
                                for value in module.parameters()))
        self.assertTrue(any(value.grad is not None and bool(value.grad.abs().sum() > 0)
                            for value in model.experts[0].parameters()))
        self.assertTrue(all(value.grad is None for value in model.base_model.parameters()))
        self.assertTrue(all(value.grad is None for value in model.degradation_prior.dr_encoder.parameters()))
        self.assertTrue(all(value.grad is None for value in model.experts[1].parameters()))

    def test_minimum_size_uses_replicate_padding_and_root_only_tree(self):
        model = make_model(tree_depth=0).eval()
        x = torch.randn(1, 3, 1, 2)
        with torch.no_grad():
            actual, aux = model(x, return_aux=True)
            expected = model.base_model(pad_like_native_wrapper(x))[..., :1, :2]
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(aux["padding_mode"], "replicate")
        self.assertEqual(len(model.experts), 0)
        self.assertEqual(aux["hard_masks"].shape, (1, 1, 1, 2))
        self.assertTrue(bool(aux["hard_masks"].all()))

    def test_root_only_has_no_reconstruction_router_gradient_or_expert_call(self):
        model = controlled_model(root_only=True)
        x = torch.randn(1, 3, 3, 7, requires_grad=True)
        output, aux = model(x, return_aux=True)
        self.assertTrue(bool((aux["routing"].expert_ids == -1).all()))
        expected = model.base_model.scale * x + model.base_model.bias
        self.assertTrue(torch.equal(output, expected))
        output.square().mean().backward()
        self.assertTrue(all(not expert.seen_inputs for expert in model.experts))
        for branch in (model.router.der_feature, model.router.ger_feature):
            gradient = branch.logits.grad
            self.assertTrue(gradient is None or bool((gradient == 0).all()))

    def test_all_root_frozen_base_batch_has_no_backward_graph(self):
        """Document the inherited boundary; do not invent a root proxy/loss."""
        model = controlled_model(root_only=True, freeze_base=True)
        output = model(torch.randn(1, 3, 3, 7))
        self.assertFalse(output.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "does not require grad|grad_fn"):
            output.square().mean().backward()

    def test_noise_uses_original_scale_and_only_training_flag(self):
        model = make_model(tree_depth=2, router_noise=True)
        router = model.router
        self.assertIsInstance(router, DensePriorRouter)
        h = torch.randn(1, 2, 3, 5)
        e_d = torch.randn(1, 2, 2, 2)
        e_g = torch.randn(1, 2, 2, 2)
        clean_der, clean_ger = router.compute_logits(h, e_d, e_g)
        der_noise = torch.randn_like(clean_der)
        ger_noise = torch.randn_like(clean_ger)
        router.train()
        with mock.patch("basicsr.models.archs.unirestorerpp.core.torch.randn_like",
                        side_effect=[der_noise, ger_noise]) as draw:
            decision = router(h, e_d, e_g)
        self.assertEqual(draw.call_count, 2)
        expected_der = clean_der + der_noise / 4
        expected_ger = clean_ger + ger_noise / 3
        self.assertTrue(torch.equal(decision.leaf_indices, expected_der.argmax(1)))
        self.assertTrue(torch.equal(decision.level_indices, expected_ger.argmax(1)))
        expected_p = expected_der.softmax(1).gather(1, decision.leaf_indices.unsqueeze(1))
        torch.testing.assert_close(decision.der_selected_probability, expected_p)
        router.eval()
        with mock.patch("basicsr.models.archs.unirestorerpp.core.torch.randn_like", side_effect=AssertionError("eval noise")):
            first = router(h, e_d, e_g)
            second = router(h, e_d, e_g)
        self.assertTrue(torch.equal(first.expert_ids, second.expert_ids))
        router.train()
        router.router_noise = False
        with mock.patch("basicsr.models.archs.unirestorerpp.core.torch.randn_like", side_effect=AssertionError("disabled noise")):
            router(h, e_d, e_g)

    def test_checkpoint_roundtrip_and_native_base_initialization(self):
        model = make_model().eval()
        force_nonroot_zero(model)
        with torch.no_grad():
            model.experts[0].output.weight.add_(0.02)
        x = torch.randn(1, 3, 9, 11)
        with torch.no_grad():
            expected = model(x)
        with tempfile.TemporaryDirectory(prefix="uni_full_model_test_") as directory:
            checkpoint_path = Path(directory) / "model.pt"
            torch.save({"params": model.state_dict()}, checkpoint_path)
            restored = make_model().eval()
            restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["params"], strict=True)
            with torch.no_grad():
                self.assertTrue(torch.equal(expected, restored(x)))
        native = model.native_base_state_dict()
        reinitialized = make_model(base_checkpoint={"params": native}, base_checkpoint_key="params")
        for owner in (reinitialized.base_model, *reinitialized.experts):
            for key, value in native.items():
                self.assertTrue(torch.equal(value, owner.state_dict()[key]))
        self.assertNotEqual(next(restored.base_model.parameters()).data_ptr(),
                            next(restored.experts[0].parameters()).data_ptr())

    def test_real_dense_prior_native_restormer_smoke(self):
        """No test doubles: actual ResNet18 prior and three small Restormers."""
        model = UniRestorerPPFullModel(**config()).eval()
        force_nonroot_zero(model)
        x = torch.randn(1, 3, 9, 11)
        calls = []
        original_d_inputs = []
        handles = [
            model.router.register_forward_hook(lambda *args: calls.append(1)),
            model.degradation_prior.dr_encoder.register_forward_pre_hook(
                lambda module, args: original_d_inputs.append(args[0].detach().clone())),
        ]
        with torch.no_grad():
            actual, aux = model(x, return_aux=True)
            native_selected = model.experts[0](pad_like_native_wrapper(x))[..., :9, :11]
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [1])
        self.assertEqual(len(original_d_inputs), 1)
        self.assertTrue(torch.equal(original_d_inputs[0], x))
        self.assertTrue(torch.equal(actual, native_selected))
        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(aux["E"].shape[1], 2)
        self.assertEqual(aux["routing"].expert_ids.shape[-2:], x.shape[-2:])


if __name__ == "__main__":
    unittest.main()
