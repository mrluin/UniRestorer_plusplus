"""Configurable RGB UniRestorer++ and independently usable architecture components.

The native ``basicsr.models.archs.unirestorerpp_arch`` entry registers the RGB
network and feature blocks with the local builder. Components use relative
imports; construction uses real torchvision/einops and the unchanged official
``basicsr.models.archs.restormer_arch`` source.
"""

from .core import DensePriorRouter, HierarchyIndex, RegionAwareExpertBlock, RoutingDecision
from .lora import LoRAConv2dResidual
from .lora_adapter import (
    LoRAAdaptedConv2d, replace_lora_convs,
    adapt_restormer_transformer_block, RESTORMER_TRANSFORMER_TARGETS,
)
from .restormer_expert import (
    LoRARestormerBlockExpert, LoRARestormerBlockResidual,
    RegionAwareLoRARestormerBlock,
)
from .dense_prior import (
    DenseResNet18Encoder, DenseDegradationPrior, PointwiseRouterBranch,
    build_dense_prior_router,
)
from .network import UniRestorerPP

__all__ = [
    'DensePriorRouter', 'HierarchyIndex', 'RegionAwareExpertBlock', 'RoutingDecision',
    'LoRAConv2dResidual', 'LoRAAdaptedConv2d', 'replace_lora_convs',
    'adapt_restormer_transformer_block', 'RESTORMER_TRANSFORMER_TARGETS',
    'LoRARestormerBlockExpert', 'LoRARestormerBlockResidual',
    'RegionAwareLoRARestormerBlock',
    'DenseResNet18Encoder', 'DenseDegradationPrior', 'PointwiseRouterBranch',
    'build_dense_prior_router', 'UniRestorerPP',
]
