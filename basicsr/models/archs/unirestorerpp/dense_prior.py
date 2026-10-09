import math
from typing import Tuple

from torch import Tensor, nn
from torch.nn import functional as F

from .core import DensePriorRouter, HierarchyIndex


class DenseResNet18Encoder(nn.Module):
    """ResNet18 trunk at stride 8, followed by a linear 512-to-p projection."""

    def __init__(self, p: int = 512, output_stride: int = 8) -> None:
        super().__init__()
        self.p = int(p)
        self.output_stride = int(output_stride)
        if self.output_stride != 8:
            raise NotImplementedError("DenseResNet18Encoder supports only output_stride=8")
        from torchvision.models import resnet18

        backbone = resnet18(weights=None)
        # Remove the last two downsamplings; keep dilation one.
        for stage in (backbone.layer3, backbone.layer4):
            first = stage[0]
            first.conv1.stride = (1, 1)
            first.downsample[0].stride = (1, 1)
            first.stride = 1
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            setattr(self, name, getattr(backbone, name))
        self.projection = nn.Conv2d(512, self.p, kernel_size=1, bias=True)

    def forward(self, X: Tensor) -> Tensor:
        features = self.maxpool(self.relu(self.bn1(self.conv1(X))))
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        return self.projection(features)


class DenseDegradationPrior(nn.Module):
    """Predict dense mean and variance, with optional freezing of D."""

    def __init__(
        self, p: int = 512, variance_floor: float = 0.01,
        variance_init: float = 1.0, freeze_dr: bool = True,
        output_stride: int = 8,
    ) -> None:
        super().__init__()
        self.p = int(p)
        self.variance_floor = float(variance_floor)
        self.variance_init = float(variance_init)
        self.dr_encoder = DenseResNet18Encoder(p=self.p, output_stride=output_stride)
        self.phi_mu = nn.Conv2d(self.p, self.p, kernel_size=1, bias=True)
        self.phi_rawvar = nn.Conv2d(self.p, self.p, kernel_size=1, bias=True)
        positive_raw_target = self.variance_init - self.variance_floor
        # Stable inverse of softplus(x), including large positive targets.
        raw_bias = positive_raw_target + math.log(-math.expm1(-positive_raw_target))
        nn.init.eye_(self.phi_mu.weight[:, :, 0, 0])
        nn.init.zeros_(self.phi_mu.bias)
        nn.init.zeros_(self.phi_rawvar.weight)
        nn.init.constant_(self.phi_rawvar.bias, raw_bias)
        self.set_freeze_dr(freeze_dr)

    def set_freeze_dr(self, freeze_dr: bool) -> "DenseDegradationPrior":
        """Explicitly freeze/unfreeze D; estimation-head flags are preserved."""
        self.freeze_dr = freeze_dr
        self.dr_encoder.requires_grad_(not freeze_dr)
        self.dr_encoder.train(False if freeze_dr else self.training)
        return self

    def train(self, mode: bool = True) -> "DenseDegradationPrior":
        super().train(mode)
        if self.freeze_dr:
            self.dr_encoder.eval()
        return self

    def compute_dr(self, X: Tensor) -> Tensor:
        """Return p-channel E on the native dense encoder grid."""
        return self.dr_encoder(X)

    def compute_distribution(self, E: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply p-to-p heads without changing E's spatial grid."""
        E_d = self.phi_mu(E)
        # E_g is variance, not a standard deviation or a sampled feature.
        E_g = F.softplus(self.phi_rawvar(E)) + self.variance_floor
        return E_d, E_g

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        return self.compute_distribution(self.compute_dr(X))


class PointwiseRouterBranch(nn.Module):
    """Conv1x1-GELU-Conv1x1 logits, independently instantiated per branch."""

    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.project_in = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=1, bias=True)
        self.activation = nn.GELU()
        self.project_out = nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1, bias=True)

    def forward(self, features: Tensor) -> Tensor:
        return self.project_out(self.activation(self.project_in(features)))


def build_dense_prior_router(
    feature_channels: int, 
    hierarchy: HierarchyIndex, 
    p: int = 512,
    hidden_channels: int = 256, 
    router_noise: bool = True,
) -> DensePriorRouter:
    """Create four fresh independent branches for exactly one restoration block."""
    return DensePriorRouter(
        der_prior=PointwiseRouterBranch(p, hierarchy.num_leaves, hidden_channels),
        der_feature=PointwiseRouterBranch(feature_channels, hierarchy.num_leaves, hidden_channels),
        ger_prior=PointwiseRouterBranch(p, hierarchy.num_levels, hidden_channels),
        ger_feature=PointwiseRouterBranch(feature_channels, hierarchy.num_levels, hidden_channels),
        hierarchy=hierarchy, router_noise=router_noise,
    )


__all__ = [
    "DenseResNet18Encoder", "DenseDegradationPrior", "PointwiseRouterBranch",
    "build_dense_prior_router",
]
