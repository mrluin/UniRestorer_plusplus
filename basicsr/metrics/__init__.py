# Modified from BasicSR 1.4.2: PSNR/SSIM exports without NIQE dependencies.
# See docs/GENERAL_CORE_PROVENANCE.json for source hashes and changes.
from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .psnr_ssim import calculate_psnr, calculate_psnr_pt, calculate_ssim, calculate_ssim_pt

__all__ = ['calculate_metric', 'calculate_psnr', 'calculate_ssim', 'calculate_psnr_pt', 'calculate_ssim_pt']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
