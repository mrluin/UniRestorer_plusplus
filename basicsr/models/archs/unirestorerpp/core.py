"""Dense top-1 routing and full-feature expert fusion."""

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RoutingDecision:
    leaf_indices: Tensor
    level_indices: Tensor
    group_indices: Tensor
    expert_ids: Tensor
    masks: Tensor
    der_selected_probability: Optional[Tensor] = None
    ger_selected_probability: Optional[Tensor] = None
    normalized_gate: Optional[Tensor] = None


class HierarchyIndex(nn.Module):
    """Map a selected leaf and level to its ancestor and non-root expert."""

    def __init__(self, parents_by_level: Sequence[Sequence[int]]) -> None:
        super().__init__()
        self.parents_by_level = tuple(tuple(row) for row in parents_by_level)
        self.level_sizes = (1,) + tuple(len(row) for row in self.parents_by_level[1:])
        paths = torch.empty((self.num_levels, self.num_leaves), dtype=torch.long)
        paths[-1] = torch.arange(self.num_leaves)
        for level in range(self.num_levels - 1, 0, -1):
            parents = torch.tensor(self.parents_by_level[level], dtype=torch.long)
            paths[level - 1] = parents[paths[level]]
        self.register_buffer("leaf_to_group", paths)

        # Root uses -1; other experts follow level/group order.
        offsets = [0]
        running = 0
        for size in self.level_sizes[1:]:
            offsets.append(running)
            running += size
        self.register_buffer("expert_offsets", torch.tensor(offsets, dtype=torch.long))

    @property
    def num_levels(self) -> int:
        return len(self.level_sizes)

    @property
    def num_leaves(self) -> int:
        return self.level_sizes[-1]

    @property
    def num_experts(self) -> int:
        return sum(self.level_sizes[1:])

    @classmethod
    def perfect_binary_tree(cls, depth: int) -> "HierarchyIndex":
        """Build depth non-root levels, with heap root 0 and children 2i+1/2i+2."""
        parents = [()]
        parents.extend(tuple(group // 2 for group in range(1 << level)) for level in range(1, depth + 1))
        hierarchy = cls(parents)
        hierarchy.perfect_binary_depth = depth
        device = hierarchy.leaf_to_group.device
        level_bases = torch.tensor([(1 << level) - 1 for level in range(depth + 1)], dtype=torch.long, device=device)
        hierarchy.register_buffer("leaf_to_heap_node", hierarchy.leaf_to_group + level_bases[:, None])
        num_nodes = (1 << (depth + 1)) - 1
        hierarchy.register_buffer("expert_heap_ids", torch.arange(1, num_nodes, dtype=torch.long, device=device))
        node_ids = torch.arange(num_nodes, dtype=torch.long, device=device)
        hierarchy.register_buffer("heap_parent_indices", (node_ids - 1) // 2)
        return hierarchy

    def hard_top1_assignment(self, der_logits: Tensor, ger_logits: Tensor) -> RoutingDecision:
        leaf_indices = der_logits.argmax(dim=1)
        level_indices = ger_logits.argmax(dim=1)
        group_indices = self.leaf_to_group[level_indices, leaf_indices]
        expert_ids = self.expert_offsets[level_indices] + group_indices
        expert_ids = torch.where(level_indices == 0, torch.full_like(expert_ids, -1), expert_ids)
        all_expert_ids = torch.arange(self.num_experts, device=expert_ids.device).view(1, -1, 1, 1)
        masks = expert_ids.unsqueeze(1) == all_expert_ids
        return RoutingDecision(leaf_indices, level_indices, group_indices, expert_ids, masks)

    def forward(self, der_logits: Tensor, ger_logits: Tensor) -> RoutingDecision:
        return self.hard_top1_assignment(der_logits, ger_logits)


class DensePriorRouter(nn.Module):
    def __init__(
        self,
        der_prior: nn.Module,
        der_feature: nn.Module,
        ger_prior: nn.Module,
        ger_feature: nn.Module,
        hierarchy: HierarchyIndex,
        *,
        router_noise: bool = True,
    ) -> None:
        super().__init__()
        self.der_prior = der_prior
        self.der_feature = der_feature
        self.ger_prior = ger_prior
        self.ger_feature = ger_feature
        self.hierarchy = hierarchy
        self.router_noise = router_noise

    def compute_logits(self, H: Tensor, E_d: Tensor, E_g: Tensor) -> Tuple[Tensor, Tensor]:
        grid = H.shape[-2:]
        if E_d.shape[-2:] != grid:
            E_d = F.interpolate(E_d, size=grid, mode="bilinear", align_corners=False)
        if E_g.shape[-2:] != grid:
            E_g = F.interpolate(E_g, size=grid, mode="bilinear", align_corners=False)
        der_logits = self.der_prior(E_d) + self.der_feature(H)
        ger_logits = self.ger_prior(E_g) + self.ger_feature(H)
        return der_logits, ger_logits

    def forward(self, H: Tensor, E_d: Tensor, E_g: Tensor) -> RoutingDecision:
        der_logits, ger_logits = self.compute_logits(H, E_d, E_g)
        
        if self.training and self.router_noise:
            der_logits = der_logits + torch.randn_like(der_logits) / self.hierarchy.num_leaves
            ger_logits = ger_logits + torch.randn_like(ger_logits) / self.hierarchy.num_levels
        routing = self.hierarchy.hard_top1_assignment(der_logits, ger_logits)
        
        probabilities = []
        for logits, indices in ((der_logits, routing.leaf_indices), (ger_logits, routing.level_indices)):
            probability_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
            full_probability = logits.to(dtype=probability_dtype).softmax(dim=1)
            probabilities.append(full_probability.gather(1, indices.unsqueeze(1)))
        der_selected, ger_selected = probabilities
        # Value one in forward; both selected probabilities retain their gradients.
        normalized_gate = (der_selected / der_selected.detach()) * (ger_selected / ger_selected.detach())
        return replace(
            routing,
            der_selected_probability=der_selected,
            ger_selected_probability=ger_selected,
            normalized_gate=normalized_gate,
        )


class RegionAwareExpertBlock(nn.Module):
    def __init__(
        self,
        base_block: nn.Module,
        experts: Sequence[nn.Module],
        router: DensePriorRouter,
        freeze_base: bool = True,
        frozen_base_eval: bool = True,
    ) -> None:
        super().__init__()
        self.base_block = base_block
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.freeze_base = freeze_base
        self.frozen_base_eval = frozen_base_eval
        if self.freeze_base:
            self.base_block.requires_grad_(False)
            if self.frozen_base_eval:
                self.base_block.eval()

    def train(self, mode: bool = True) -> "RegionAwareExpertBlock":
        super().train(mode)
        if self.freeze_base and self.frozen_base_eval:
            self.base_block.eval()
        return self

    def forward(
        self, 
        H: Tensor, 
        E_d: Tensor, 
        E_g: Tensor, 
        return_routing: bool = False
    ) -> Union[Tensor, Tuple[Tensor, RoutingDecision]]:
        routing = self.router(H, E_d, E_g)
        output = self.base_block(H)
        gate = routing.normalized_gate
        if gate is not None:
            gate = gate.to(dtype=output.dtype)
        for expert_id, expert in enumerate(self.experts):
            mask = routing.masks[:, expert_id:expert_id + 1]
            if not bool(mask.any()):
                continue
            # Experts see the complete H; apply the spatial mask to their output.
            masked_residual = mask.to(dtype=output.dtype) * expert(H)
            if gate is not None:
                masked_residual = masked_residual * gate
            output = output + masked_residual
        if return_routing:
            return output, routing
        return output


__all__ = ["RoutingDecision", "HierarchyIndex", "DensePriorRouter", "RegionAwareExpertBlock"]
