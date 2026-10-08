import math
from typing import Optional

from torch import Tensor, nn
from torch.nn import functional as F


class LoRAConv2dResidual(nn.Module):
    """Convolve full H with the LoRA-IR kernel update, without the base bias."""

    def __init__(
        self, base_conv: nn.Conv2d, rank: int = 8, lora_alpha: Optional[float] = None
    ) -> None:
        super().__init__()
        rank = int(rank)
        alpha = float(2 * rank if lora_alpha is None else lora_alpha)
        kernel_size = tuple(base_conv.kernel_size)
        if kernel_size[0] != kernel_size[1] or base_conv.padding_mode != "zeros":
            raise NotImplementedError("LoRA-IR residuals support square kernels with zero padding")

        self.rank = rank
        self.lora_alpha = alpha
        self.scaling = alpha / rank
        self.in_channels = base_conv.in_channels
        self.out_channels = base_conv.out_channels
        self.kernel_size = kernel_size
        self.stride = tuple(base_conv.stride)
        self.padding = base_conv.padding if isinstance(base_conv.padding, str) else tuple(base_conv.padding)
        self.dilation = tuple(base_conv.dilation)
        self.groups = base_conv.groups
        self.padding_mode = base_conv.padding_mode
        self.weight_shape = tuple(base_conv.weight.shape)
        k = int(kernel_size[0])
        # LoRA-IR uses r*k rows, including grouped and depthwise convolutions.
        self.lora_A = nn.Parameter(base_conv.weight.new_empty((rank * k, self.in_channels * k)))
        self.lora_B = nn.Parameter(base_conv.weight.new_empty((self.out_channels // self.groups * k, rank * k)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def delta_weight(self) -> Tensor:
        """Return the scaled residual kernel without modifying base weights."""
        return (self.lora_B @ self.lora_A).view(self.weight_shape) * self.scaling

    def forward(self, H: Tensor) -> Tensor:
        """Compute one full-map residual; output masking belongs to the block."""
        weight = self.delta_weight()
        dilation = self.dilation
        if H.device.type == 'cpu' and dilation != (1, 1):
            # Zero insertion preserves dilated-conv math and A/B gradients on CPU.
            height = (weight.shape[-2] - 1) * dilation[0] + 1
            width = (weight.shape[-1] - 1) * dilation[1] + 1
            expanded = weight.new_zeros((*weight.shape[:2], height, width))
            expanded[..., ::dilation[0], ::dilation[1]] = weight
            weight = expanded
            dilation = (1, 1)
        return F.conv2d(
            H, weight, bias=None, stride=self.stride,
            padding=self.padding, dilation=dilation, groups=self.groups,
        )


__all__ = ["LoRAConv2dResidual"]
