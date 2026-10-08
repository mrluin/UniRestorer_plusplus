"""Architecture discovery and registry construction at Restormer's native path.

Modified from BasicSR 1.4.2's registry builder (Apache-2.0). The namespace follows
the official Restormer checkout, ``basicsr.models.archs``. ``define_network``
retains the earlier Restormer API and uses the same non-mutating builder.
"""

import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

from .restormer_arch import Restormer


__all__ = ["build_network", "define_network", "Restormer"]

# The retained official Restormer source has no registry decorator. Register
# its original class here, leaving that source and its hash unchanged.
if "Restormer" not in ARCH_REGISTRY:
    ARCH_REGISTRY.register()(Restormer)

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(value))[0]
    for value in scandir(arch_folder) if value.endswith("_arch.py")
]
_arch_modules = [
    importlib.import_module("basicsr.models.archs.{}".format(name))
    for name in arch_filenames
]


def build_network(opt):
    """Build the registered ``type`` while preserving the caller's config."""
    opt = deepcopy(opt)
    network_type = opt.pop("type")
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info("Network [{}] is created.".format(net.__class__.__name__))
    return net


define_network = build_network
