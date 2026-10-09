import math

import torch
from torch.nn import functional as F

from basicsr.models.archs.unirestorerpp.degradation_space import gaussian_degradation_nll


def level_load_balance_loss(clean_logits, noisy_logits, noise_std):
    """Balance GER importance and expected top-1 load across levels."""
    dtype = torch.float64 if clean_logits.dtype == torch.float64 else torch.float32
    clean = clean_logits.to(dtype=dtype)
    noisy = noisy_logits.to(dtype=dtype)
    levels = clean.shape[1]
    if levels == 1:
        return clean.sum() * 0

    importance = clean.softmax(dim=1).sum(dim=(0, 2, 3))
    if noise_std == 0:
        load = F.one_hot(noisy.argmax(dim=1), num_classes=levels).to(dtype=dtype).mean(dim=(0, 1, 2))
    else:
        # Keep the same draw's winning threshold in the gradient graph.
        threshold = noisy.max(dim=1, keepdim=True).values
        probability = 0.5 * (1 + torch.erf((clean - threshold) / (noise_std * math.sqrt(2))))
        load = probability.mean(dim=(0, 2, 3))
    importance_cv = (importance.std(unbiased=True) / (importance.mean() + 1e-8)).square()
    load_cv = (load.std(unbiased=True) / (load.mean() + 1e-8)).square()
    return 0.5 * importance_cv + 0.5 * load_cv


def joint_restoration_loss(restored, target, aux, space, leaf_map):
    """Return paper-weighted total and per-block losses from original-grid labels.

    DG predicts a finest group's raw-GAP centroid on the native prior grid.
    Only source leaf labels are resized; both phi outputs keep their own grid.
    """
    mu, var = aux["E_d"], aux["E_g"]
    aligned_leaf = F.interpolate(
        leaf_map[:, None].to(device=mu.device, dtype=torch.float64),
        size=mu.shape[-2:], mode="nearest",
    )[:, 0].long()
    shared_dg = gaussian_degradation_nll(mu, var, space.target_map(aligned_leaf))
    per_block = {
        key: {
            "dg": shared_dg,
            "aux": level_load_balance_loss(
                decision.ger_clean_logits, decision.ger_noisy_logits, decision.ger_noise_std,
            ),
        }
        for key, decision in aux["routing"].items()
    }
    dg_loss = torch.stack([losses["dg"] for losses in per_block.values()]).mean()
    aux_loss = torch.stack([losses["aux"] for losses in per_block.values()]).mean()
    l1_loss = F.l1_loss(restored, target)
    total = l1_loss + 0.1 * dg_loss + 0.01 * aux_loss
    return total, {"l1": l1_loss, "dg": dg_loss, "aux": aux_loss, "per_block": per_block}


__all__ = ["level_load_balance_loss", "joint_restoration_loss"]
