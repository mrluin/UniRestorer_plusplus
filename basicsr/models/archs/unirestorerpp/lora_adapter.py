from typing import Dict, Optional, Sequence, Tuple

from torch import Tensor, nn

from .lora import LoRAConv2dResidual


RESTORMER_TRANSFORMER_TARGETS = (
    "attn.qkv", "attn.qkv_dwconv", "attn.project_out",
    "ffn.project_in", "ffn.dwconv", "ffn.project_out",
)


class LoRAAdaptedConv2d(nn.Module):
    def __init__(
        self, base_conv: nn.Conv2d, rank: int = 8,
        lora_alpha: Optional[float] = None, freeze_base: bool = True,
    ) -> None:
        super().__init__()
        residual = LoRAConv2dResidual(base_conv, rank=rank, lora_alpha=lora_alpha)
        self.base_conv = base_conv
        self.residual = residual
        self.freeze_base = freeze_base
        self.training = base_conv.training
        self.residual.train(base_conv.training)
        if freeze_base:
            self.base_conv.requires_grad_(False)

    def forward(self, H: Tensor) -> Tensor:
        residual = self.residual(H)
        return self.base_conv(H) + residual


def _resolve_targets(
    model: nn.Module, target_paths: Sequence[str],
) -> Tuple[Tuple[str, nn.Module, str, nn.Conv2d], ...]:
    resolved = []
    for path in target_paths:
        parent_path, _, name = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        resolved.append((path, parent, name, getattr(parent, name)))
    return tuple(resolved)


def replace_lora_convs(
    model: nn.Module, target_paths: Sequence[str], rank: int = 8,
    lora_alpha: Optional[float] = None, freeze_base: bool = True,
) -> Dict[str, LoRAAdaptedConv2d]:
    resolved = _resolve_targets(model, target_paths)
    adapters = {
        path: LoRAAdaptedConv2d(convolution, rank=rank, lora_alpha=lora_alpha, freeze_base=False)
        for path, _, _, convolution in resolved
    }
    for path, parent, name, _ in resolved:
        setattr(parent, name, adapters[path])
    for adapter in adapters.values():
        if freeze_base:
            adapter.base_conv.requires_grad_(False)
        adapter.freeze_base = freeze_base
    return adapters


def adapt_restormer_transformer_block(
    block: nn.Module, rank: int = 8, lora_alpha: Optional[float] = None,
    freeze_base: bool = True,
) -> Dict[str, LoRAAdaptedConv2d]:
    originals = tuple(block.parameters())
    adapters = replace_lora_convs(
        block, RESTORMER_TRANSFORMER_TARGETS, rank=rank,
        lora_alpha=lora_alpha, freeze_base=False,
    )
    if freeze_base:
        for parameter in originals:
            parameter.requires_grad_(False)
    for adapter in adapters.values():
        adapter.freeze_base = freeze_base
    return adapters


__all__ = [
    "LoRAAdaptedConv2d", "replace_lora_convs",
    "adapt_restormer_transformer_block", "RESTORMER_TRANSFORMER_TARGETS",
]
