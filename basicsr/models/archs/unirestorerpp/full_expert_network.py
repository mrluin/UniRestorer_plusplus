"""One input-space hard router over independent, complete Restormer models.

Reuses this repository's native Restormer and UniRestorer++ hierarchy/prior/router.
The LoRA network and its checkpoint layout are deliberately left unchanged.
"""

from copy import deepcopy
from collections import OrderedDict
from collections.abc import Sequence
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .core import HierarchyIndex
from .dense_prior import DenseDegradationPrior, build_dense_prior_router
from .network import _native_checkpoint_state
from ..restormer_arch import Restormer


class _ExactExpertWithResidualGradient(torch.autograd.Function):
    """Copy expert values exactly; differentiate base + gate * (expert - base).

    DensePriorRouter's gate has value one. Performing the subtraction/addition
    in forward could round away expert pixels; only backward needs that delta.
    The caller applies the discrete mask with torch.where.
    """

    @staticmethod
    def forward(ctx, base: Tensor, expert: Tensor, gate: Tensor) -> Tensor:
        ctx.save_for_backward(base, expert, gate)
        return expert.view_as(expert)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        base, expert, gate = ctx.saved_tensors
        return (
            grad_output * (1 - gate),
            grad_output * gate,
            (grad_output * (expert - base)).sum(dim=1, keepdim=True),
        )


class UniRestorerPPFullModel(nn.Module):
    """Complete RGB Restormer at every tree node, routed once before restoration.

    Root is ``base_model`` (frozen by default). Non-root ``experts`` follow the
    existing breadth-first hierarchy IDs, independently copy the base's initial
    values, and have trainable, unshared Parameters. Native Restormer already
    includes its image residual; this wrapper never adds X to its outputs.
    """

    def __init__(
        self,
        *,
        tree_depth: int,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "BiasFree",
        dual_pixel_task: bool = False,
        p: int = 512,
        router_hidden_channels: int = 256,
        router_feature_channels: Optional[int] = None,
        router_noise: bool = True,
        freeze_dr: bool = True,
        dr_output_stride: int = 8,
        variance_floor: float = 0.01,
        variance_init: float = 1.0,
        base_checkpoint=None,
        base_checkpoint_key: Optional[str] = None,
        freeze_base: bool = True,
        frozen_base_eval: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(tree_depth, bool) or not isinstance(tree_depth, int) or tree_depth < 0:
            raise ValueError("tree_depth must be a nonnegative integer")
        if inp_channels != 3 or out_channels != 3 or dual_pixel_task:
            raise ValueError("UniRestorerPPFullModel supports RGB-to-RGB restoration only")
        if router_feature_channels is None:
            router_feature_channels = dim
        if not isinstance(router_feature_channels, int) or router_feature_channels < 1:
            raise ValueError("router_feature_channels must be a positive integer")
        self.tree_depth = tree_depth
        self.p = p
        self.router_hidden_channels = router_hidden_channels
        self.router_feature_channels = router_feature_channels
        self.freeze_base = freeze_base
        self.frozen_base_eval = frozen_base_eval
        self.base_config = {
            "inp_channels": inp_channels, "out_channels": out_channels,
            "dim": dim, "num_blocks": list(num_blocks),
            "num_refinement_blocks": num_refinement_blocks, "heads": list(heads),
            "ffn_expansion_factor": ffn_expansion_factor, "bias": bias,
            "LayerNorm_type": LayerNorm_type, "dual_pixel_task": dual_pixel_task,
        }
        self.hierarchy = HierarchyIndex.perfect_binary_tree(tree_depth)
        self.base_model = Restormer(**self.base_config)
        native_state = _native_checkpoint_state(base_checkpoint, base_checkpoint_key)
        if native_state is not None:
            self.base_model.load_state_dict(native_state, strict=True)
        # No deepcopy memo: every full expert owns all of its Parameters/buffers.
        self.experts = nn.ModuleList(
            deepcopy(self.base_model) for _ in range(self.hierarchy.num_experts)
        )
        self.experts.requires_grad_(True)
        self.base_model.requires_grad_(not freeze_base)
        self.degradation_prior = DenseDegradationPrior(
            p=p, variance_floor=variance_floor, variance_init=variance_init,
            freeze_dr=freeze_dr, output_stride=dr_output_stride,
        )
        # Learnable input-space H; independent of every restoration expert.
        # Stride one keeps hard decisions on the exact original pixel grid.
        self.router_stem = nn.Sequential(
            nn.Conv2d(inp_channels, router_feature_channels, 3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )
        self.router = build_dense_prior_router(
            router_feature_channels, self.hierarchy, p=p,
            hidden_channels=router_hidden_channels, router_noise=router_noise,
        )
        self.train(self.training)

    def train(self, mode: bool = True) -> "UniRestorerPPFullModel":
        super().train(mode)
        if self.freeze_base and self.frozen_base_eval:
            self.base_model.eval()
        return self

    def native_base_state_dict(self) -> "OrderedDict[str, Tensor]":
        """Return native Restormer keys, without any router or expert prefix."""
        return self.base_model.state_dict()

    def parameter_counts(self) -> Dict[str, Dict[str, int]]:
        """Restoration Parameters only; exclude D, phi, routing stem and router."""
        counts = {}
        for name, module in (("base", self.base_model), ("experts", self.experts)):
            counts[name] = {
                "total": sum(value.numel() for value in module.parameters()),
                "trainable": sum(value.numel() for value in module.parameters() if value.requires_grad),
            }
        counts["total"] = {
            key: counts["base"][key] + counts["experts"][key]
            for key in ("total", "trainable")
        }
        return counts

    def forward(self, X: Tensor, return_aux: bool = False):
        if X.ndim != 4 or X.shape[1] != 3 or min(X.shape[0], *X.shape[-2:]) < 1:
            raise ValueError("X must be a nonempty NCHW RGB image batch")
        height, width = X.shape[-2:]
        E = self.degradation_prior.compute_dr(X)
        E_d, E_g = self.degradation_prior.compute_distribution(E)
        # Exactly one router invocation on the original image grid.
        routing = self.router(self.router_stem(X), E_d, E_g)

        pad_h, pad_w = (-height) % 8, (-width) % 8
        padding = (0, pad_w, 0, pad_h)
        padding_mode = None
        padded = X
        if pad_h or pad_w:
            padding_mode = "reflect" if pad_h < height and pad_w < width else "replicate"
            padded = F.pad(X, padding, mode=padding_mode)
        # Full, unmasked input for every model; crop before applying pixel masks.
        base_output = self.base_model(padded)[..., :height, :width]
        output = base_output
        gate = routing.normalized_gate
        if gate is not None:
            gate = gate.to(dtype=base_output.dtype)
        for expert_id, expert in enumerate(self.experts):
            mask = routing.masks[:, expert_id:expert_id + 1]
            if not bool(mask.any()):
                continue
            expert_output = expert(padded)[..., :height, :width]
            if gate is not None:
                expert_output = _ExactExpertWithResidualGradient.apply(base_output, expert_output, gate)
            output = torch.where(mask, expert_output, output)
        if return_aux:
            root_mask = routing.expert_ids.unsqueeze(1) == -1
            return output, {
                "E": E, "E_d": E_d, "E_g": E_g, "routing": routing,
                "root_mask": root_mask,
                "hard_masks": torch.cat((root_mask, routing.masks), dim=1),
                "routing_size": tuple(routing.expert_ids.shape[-2:]),
                "input_size": (height, width), "padded_size": tuple(padded.shape[-2:]),
                "padding": padding, "padding_mode": padding_mode,
            }
        return output


__all__ = ["UniRestorerPPFullModel"]
