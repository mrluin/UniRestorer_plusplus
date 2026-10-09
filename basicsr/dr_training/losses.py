import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _dense_rows(features):
    return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])


def _mean_row_ce(queries, keys, labels, temperature, chunk_size):
    def summed_ce(rows, bank, targets):
        return F.cross_entropy(rows @ bank.T / temperature, targets, reduction="sum")

    total = queries.new_zeros(())
    for start in range(0, queries.shape[0], chunk_size):
        rows = queries[start:start + chunk_size]
        targets = labels[start:start + chunk_size]
        if torch.is_grad_enabled() and (rows.requires_grad or keys.requires_grad):
            # Recompute logits in backward: simply summing CE chunks would
            # otherwise retain every chunk's class-column softmax in autograd.
            total = total + checkpoint(summed_ce, rows, keys, targets,
                                       use_reentrant=False, preserve_rng_state=False)
        else:
            total = total + summed_ce(rows, keys, targets)
    return total / queries.shape[0]


def caption_similarity_logits(dense_features, caption_bank, *, temperature):
    """Return B,C,H,W cosine logits from B,p,H,W images and a C,p bank.

    The approved raw DR/text width is 512. This tensor operation keeps its
    channel width generic and creates no trainable projection or text head.
    """
    images, captions = F.normalize(dense_features, dim=1), F.normalize(caption_bank, dim=1)
    return torch.einsum("bphw,cp->bchw", images, captions) / float(temperature)


def unique_caption_pixel_ce(
    dense_features, caption_bank, caption_ids, *, temperature,
    chunk_size=None,
):
    """Mean segmentation CE against an explicitly supplied unique C,p bank.

    caption_ids must already match the dense feature grid in B,H,W order (or
    be its flattened equivalent). This function does not resize label maps.
    """
    if chunk_size is None:
        logits = caption_similarity_logits(dense_features, caption_bank, temperature=temperature)
        labels = caption_ids.reshape(dense_features.shape[0], *dense_features.shape[2:])
        return F.cross_entropy(logits, labels)
    images, labels = _dense_rows(dense_features), caption_ids.reshape(-1)
    images, captions = F.normalize(images, dim=1), F.normalize(caption_bank, dim=1)
    return _mean_row_ce(images, captions, labels, float(temperature), chunk_size)


DR_LOSSES = {
    "unique_caption_pixel_ce": unique_caption_pixel_ce,
}

__all__ = ["caption_similarity_logits", "unique_caption_pixel_ce", "DR_LOSSES"]
