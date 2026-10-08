"""Pinned source metadata and a compatibility import; runtime does no source checks."""

from pathlib import Path
from types import ModuleType

from .. import restormer_arch


RESTORMER_COMMIT = "68dc6ac472db26f16361150cb7a96a1bc87da93f"
RESTORMER_SOURCE_SHA256 = "3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a"
RESTORMER_SOURCE_PATH = Path(restormer_arch.__file__).resolve()


def load_official_restormer() -> ModuleType:
    """Compatibility access to the normally imported local module."""
    return restormer_arch


__all__ = [
    "load_official_restormer", "RESTORMER_COMMIT", "RESTORMER_SOURCE_SHA256",
    "RESTORMER_SOURCE_PATH",
]
