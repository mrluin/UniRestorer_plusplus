from collections.abc import Mapping

from basicsr.utils.registry import ARCH_REGISTRY

from .unirestorerpp.core import DensePriorRouter, HierarchyIndex, RegionAwareExpertBlock
from .unirestorerpp.restormer_expert import RegionAwareLoRARestormerBlock
from .unirestorerpp.network import UniRestorerPP


__all__ = ["UniRestorerPP", "UniRestorerPPBlock", "UniRestorerPPRestormerBlock"]


ARCH_REGISTRY.register()(UniRestorerPP)


def _build_component(component):
    if isinstance(component, Mapping):
        from basicsr.models.archs import build_network
        return build_network(dict(component))
    return component


@ARCH_REGISTRY.register()
class UniRestorerPPBlock(RegionAwareExpertBlock):
    def __init__(self, parents_by_level, base_block, experts, der_prior,
                 der_feature, ger_prior, ger_feature, freeze_base=True,
                 frozen_base_eval=True, router_noise=True):
        hierarchy = HierarchyIndex(parents_by_level)
        router = DensePriorRouter(
            hierarchy=hierarchy,
            der_prior=_build_component(der_prior),
            der_feature=_build_component(der_feature),
            ger_prior=_build_component(ger_prior),
            ger_feature=_build_component(ger_feature),
            router_noise=router_noise,
        )
        super().__init__(
            base_block=_build_component(base_block),
            experts=[_build_component(expert) for expert in experts],
            router=router,
            freeze_base=freeze_base,
            frozen_base_eval=frozen_base_eval,
        )


@ARCH_REGISTRY.register()
class UniRestorerPPRestormerBlock(RegionAwareLoRARestormerBlock):
    def __init__(self, parents_by_level, base_block, der_prior, der_feature,
                 ger_prior, ger_feature, rank=8, lora_alpha=None,
                 frozen_base_eval=True, router_noise=True):
        hierarchy = HierarchyIndex(parents_by_level)
        router = DensePriorRouter(
            hierarchy=hierarchy,
            der_prior=_build_component(der_prior),
            der_feature=_build_component(der_feature),
            ger_prior=_build_component(ger_prior),
            ger_feature=_build_component(ger_feature),
            router_noise=router_noise,
        )
        super().__init__(
            base_block=_build_component(base_block), router=router,
            rank=rank, lora_alpha=lora_alpha,
            frozen_base_eval=frozen_base_eval,
        )
