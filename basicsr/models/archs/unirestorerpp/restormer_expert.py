from copy import deepcopy
from typing import Optional

from torch import Tensor, nn

from .core import DensePriorRouter, RegionAwareExpertBlock
from .lora_adapter import adapt_restormer_transformer_block


class LoRARestormerBlockExpert(nn.Module):
    def __init__(
        self, base_block: nn.Module, rank: int = 8,
        lora_alpha: Optional[float] = None,
    ) -> None:
        super().__init__()
        memo = {id(parameter): parameter for parameter in base_block.parameters()}
        memo.update({id(buffer): buffer for buffer in base_block.buffers()})
        adapted = deepcopy(base_block, memo)
        adapters = adapt_restormer_transformer_block(
            adapted, rank=rank, lora_alpha=lora_alpha, freeze_base=True,
        )
        # Keep the borrowed base outside this module's registered children.
        object.__setattr__(self, "_base_block", base_block)
        self.adapted_block = adapted
        self.target_paths = tuple(adapters)

    @property
    def base_block(self) -> nn.Module:
        return self._base_block

    def forward(self, H: Tensor) -> Tensor:
        """Return the complete adapted-block output, including its base."""
        return self.adapted_block(H)


class LoRARestormerBlockResidual(nn.Module):
    def __init__(self, expert: LoRARestormerBlockExpert) -> None:
        super().__init__()
        self.expert = expert

    @property
    def base_block(self) -> nn.Module:
        return self.expert.base_block

    def forward(self, H: Tensor) -> Tensor:
        """Evaluate both complete functions on H; no input mask is applied."""
        return self.expert(H) - self.base_block(H)


class RegionAwareLoRARestormerBlock(RegionAwareExpertBlock):
    def __init__(
        self, 
        base_block: nn.Module, 
        router: DensePriorRouter, 
        rank: int = 8,
        lora_alpha: Optional[float] = None, 
        frozen_base_eval: bool = True,
    ) -> None:
        residuals = [
            LoRARestormerBlockResidual(
                LoRARestormerBlockExpert(base_block, rank, lora_alpha)
            )
            for _ in range(router.hierarchy.num_experts)
        ]
        super().__init__(
            base_block, 
            residuals, 
            router,
            freeze_base=True, 
            frozen_base_eval=frozen_base_eval,
        )


__all__ = [
    "LoRARestormerBlockExpert", "LoRARestormerBlockResidual",
    "RegionAwareLoRARestormerBlock",
]
