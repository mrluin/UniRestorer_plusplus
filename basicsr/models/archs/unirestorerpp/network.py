import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .core import HierarchyIndex, RoutingDecision
from .dense_prior import DenseDegradationPrior, build_dense_prior_router
from .lora import LoRAConv2dResidual
from .restormer_expert import RegionAwareLoRARestormerBlock
from ..restormer_arch import Restormer


def _native_checkpoint_state(checkpoint, key: Optional[str]):
    if checkpoint is None:
        if key is not None:
            raise ValueError("base_checkpoint_key requires base_checkpoint")
        return None
    if key is not None and (not isinstance(key, str) or not key):
        raise TypeError("base_checkpoint_key must be a nonempty string or None")
    if isinstance(checkpoint, (str, os.PathLike)):
        # Local, explicit checkpoint only: never a URL or a weight download.
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    elif not isinstance(checkpoint, Mapping):
        raise TypeError("base_checkpoint must be a local path or a state-dict mapping")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("base checkpoint must contain a mapping")
    if key is not None:
        if key not in checkpoint:
            raise KeyError("base checkpoint has no explicitly requested key {!r}".format(key))
        checkpoint = checkpoint[key]
    if not isinstance(checkpoint, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in checkpoint.items()
    ):
        raise TypeError(
            "native base state must map parameter names to tensors; select an envelope "
            "explicitly with base_checkpoint_key"
        )
    return checkpoint


class UniRestorerPP(nn.Module):
    stage_names = (
        "encoder_level1", "encoder_level2", "encoder_level3", "latent",
        "decoder_level3", "decoder_level2", "decoder_level1", "refinement",
    )

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
        rank: int = 8, 
        lora_alpha: Optional[float] = None,
        router_noise: bool = True, 
        freeze_dr: bool = True,
        dr_output_stride: int = 8, 
        variance_floor: float = 0.01,
        variance_init: float = 1.0,
        base_checkpoint=None, 
        base_checkpoint_key: Optional[str] = None,
        frozen_base_eval: bool = True,
    ) -> None:
        super().__init__()
        native_state = _native_checkpoint_state(base_checkpoint, base_checkpoint_key)

        self.base_config = {
            "inp_channels": inp_channels, 
            "out_channels": out_channels,
            "dim": dim, 
            "num_blocks": list(num_blocks),
            "num_refinement_blocks": num_refinement_blocks, 
            "heads": list(heads),
            "ffn_expansion_factor": ffn_expansion_factor, 
            "bias": bias,
            "LayerNorm_type": LayerNorm_type, 
            "dual_pixel_task": dual_pixel_task,
        }
        self.tree_depth = tree_depth
        self.p = p
        self.router_hidden_channels = router_hidden_channels
        self.frozen_base_eval = frozen_base_eval
        backbone = Restormer(**self.base_config)
        if native_state is not None:
            backbone.load_state_dict(native_state, strict=True)
        backbone.requires_grad_(False)
        
        self.backbone = backbone
        self.hierarchy = HierarchyIndex.perfect_binary_tree(tree_depth)
        self.degradation_prior = DenseDegradationPrior(
            p=p, 
            variance_floor=variance_floor, 
            variance_init=variance_init,
            freeze_dr=freeze_dr, 
            output_stride=dr_output_stride,
        )
        
        feature_widths = (dim, 
                          2 * dim, 
                          4 * dim, 
                          8 * dim,
                          4 * dim, 
                          2 * dim, 
                          2 * dim, 
                          2 * dim)
        
        for stage_name, feature_channels in zip(self.stage_names, feature_widths):
            native_stage = getattr(self.backbone, stage_name)
            wrapped = []
            for native_block in native_stage:
                router = build_dense_prior_router(
                    feature_channels, 
                    self.hierarchy, 
                    p=p,
                    hidden_channels=router_hidden_channels, 
                    router_noise=router_noise,
                )
                wrapped.append(RegionAwareLoRARestormerBlock(
                    native_block, 
                    router, 
                    rank=rank, 
                    lora_alpha=lora_alpha,
                    frozen_base_eval=frozen_base_eval,
                ))
            setattr(self.backbone, stage_name, nn.ModuleList(wrapped))
        self.train(self.training)

    def train(self, mode: bool = True) -> "UniRestorerPP":
        super().train(mode)
        if self.frozen_base_eval:
            # Keep frozen outer modules in eval without disabling routers.
            for name, module in self.backbone.named_children():
                if name not in self.stage_names:
                    module.eval()
        return self

    def routed_blocks(self) -> Iterator[Tuple[str, RegionAwareLoRARestormerBlock]]:
        """Iterate all independently routed blocks in official forward order."""
        for stage_name in self.stage_names:
            for index, block in enumerate(getattr(self.backbone, stage_name)):
                yield "{}.{}".format(stage_name, index), block

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if assign:
            raise ValueError(
                "assign=True would replace shared frozen base Parameters; "
                "load this architecture with assign=False"
            )
        return super().load_state_dict(state_dict, strict=strict)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Parent loads must also preserve shared frozen Parameters.
        if local_metadata.get("assign_to_params_buffers", False):
            raise ValueError(
                "assign=True would replace shared frozen base Parameters; "
                "load this architecture with assign=False"
            )
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def native_base_state_dict(self) -> "OrderedDict[str, Tensor]":
        state = OrderedDict()
        for name, module in self.backbone.named_children():
            if name in self.stage_names:
                for index, block in enumerate(module):
                    for key, value in block.base_block.state_dict().items():
                        state["{}.{}.{}".format(name, index, key)] = value
            else:
                for key, value in module.state_dict().items():
                    state["{}.{}".format(name, key)] = value
        return state

    def parameter_counts(self) -> Dict[str, Dict[str, int]]:
        """Count restoration base and LoRA A/B Parameters, shared bases once.

        Both total and trainable exclude DR, phi, routers and other parameters.
        Trainable counts reflect the selected Parameters' current requires_grad.
        """
        base_ids = set()
        for name, module in self.backbone.named_children():
            base_modules = (block.base_block for block in module) if name in self.stage_names else (module,)
            for base_module in base_modules:
                base_ids.update(id(value) for value in base_module.parameters())
        group_ids = {
            "base": frozenset(base_ids),
            "lora": frozenset(id(value) for module in self.backbone.modules()
                              if isinstance(module, LoRAConv2dResidual)
                              for value in (module.lora_A, module.lora_B)),
        }
        counts = {name: {"total": 0, "trainable": 0} for name in ("base", "lora", "total")}
        # Module.parameters() deduplicates shared Parameter objects by identity.
        for parameter in self.backbone.parameters():
            owner = next((name for name, ids in group_ids.items() if id(parameter) in ids), None)
            if owner is None:
                continue
            size = parameter.numel()
            counts[owner]["total"] += size
            counts["total"]["total"] += size
            if parameter.requires_grad:
                counts[owner]["trainable"] += size
                counts["total"]["trainable"] += size
        return counts

    def _run_stage(
        self, name: str, H: Tensor, E_d: Tensor, E_g: Tensor,
        routing: Optional[Dict[str, RoutingDecision]],
        stage_shapes: Optional[Dict[str, Tuple[int, ...]]],
    ) -> Tensor:
        for index, block in enumerate(getattr(self.backbone, name)):
            if routing is None:
                H = block(H, E_d, E_g)
            else:
                H, decision = block(H, E_d, E_g, return_routing=True)
                routing["{}.{}".format(name, index)] = decision
        if stage_shapes is not None:
            stage_shapes[name] = tuple(H.shape)
        return H

    def forward(self, X: Tensor, return_aux: bool = False):
        if self.backbone.dual_pixel_task:
            raise NotImplementedError("dual-pixel restoration is outside this RGB architecture")
        height, width = X.shape[-2:]
        pad_h, pad_w = (-height) % 8, (-width) % 8
        padding = (0, pad_w, 0, pad_h)
        padding_mode = None
        padded = X
        if pad_h or pad_w:
            padding_mode = "reflect" if pad_h < height and pad_w < width else "replicate"
            padded = F.pad(X, padding, mode=padding_mode)
        # Keep D's input identical to DR training/extraction; only the backbone
        # needs dimensions divisible by eight.
        E = self.degradation_prior.compute_dr(X)
        E_d, E_g = self.degradation_prior.compute_distribution(E)
        
        routing = OrderedDict() if return_aux else None
        stage_shapes = OrderedDict() if return_aux else None
        
        base = self.backbone

        inp_enc_level1 = base.patch_embed(padded)
        out_enc_level1 = self._run_stage(
            "encoder_level1", inp_enc_level1, E_d, E_g, routing, stage_shapes)
        out_enc_level2 = self._run_stage(
            "encoder_level2", base.down1_2(out_enc_level1), E_d, E_g, routing, stage_shapes)
        out_enc_level3 = self._run_stage(
            "encoder_level3", base.down2_3(out_enc_level2), E_d, E_g, routing, stage_shapes)
        latent = self._run_stage(
            "latent", base.down3_4(out_enc_level3), E_d, E_g, routing, stage_shapes)

        inp_dec_level3 = torch.cat([base.up4_3(latent), out_enc_level3], dim=1)
        out_dec_level3 = self._run_stage(
            "decoder_level3", base.reduce_chan_level3(inp_dec_level3),
            E_d, E_g, routing, stage_shapes)
        inp_dec_level2 = torch.cat([base.up3_2(out_dec_level3), out_enc_level2], dim=1)
        out_dec_level2 = self._run_stage(
            "decoder_level2", base.reduce_chan_level2(inp_dec_level2),
            E_d, E_g, routing, stage_shapes)
        inp_dec_level1 = torch.cat([base.up2_1(out_dec_level2), out_enc_level1], dim=1)
        out_dec_level1 = self._run_stage(
            "decoder_level1", inp_dec_level1, E_d, E_g, routing, stage_shapes)
        refined = self._run_stage(
            "refinement", out_dec_level1, E_d, E_g, routing, stage_shapes)
        restored = (base.output(refined) + padded)[..., :height, :width]
        if return_aux:
            return restored, {
                "E": E, "E_d": E_d, "E_g": E_g, "routing": routing,
                "stage_shapes": stage_shapes, "input_size": (height, width),
                "padded_size": tuple(padded.shape[-2:]), "padding": padding,
                "padding_mode": padding_mode,
            }
        return restored


__all__ = ["UniRestorerPP"]
