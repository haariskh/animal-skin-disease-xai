"""Device selection."""

from __future__ import annotations

import torch

from aniderm.utils.logging import get_logger

logger = get_logger(__name__)


def select_device(preference: str = "auto") -> torch.device:
    """Resolve a device string into a concrete ``torch.device``.

    ``"auto"`` prefers CUDA, then Apple's MPS backend, then CPU. An explicit preference
    that is unavailable produces a warning and falls back rather than crashing, so the
    same config file works on a GPU workstation and on a laptop.
    """
    preference = (preference or "auto").lower()

    if preference == "auto":
        if torch.cuda.is_available():
            chosen = torch.device("cuda")
        elif torch.backends.mps.is_available():
            chosen = torch.device("mps")
        else:
            chosen = torch.device("cpu")
    elif preference.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            chosen = torch.device("cpu")
        else:
            chosen = torch.device(preference)
    elif preference == "mps":
        if not torch.backends.mps.is_available():
            logger.warning("MPS requested but unavailable; falling back to CPU")
            chosen = torch.device("cpu")
        else:
            chosen = torch.device("mps")
    else:
        chosen = torch.device("cpu")

    if chosen.type == "cuda":
        index = chosen.index or 0
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / 1024**3
        logger.info("Using CUDA device %d: %s (%.1f GB)", index, name, total)
    else:
        logger.info("Using device: %s", chosen.type)
    return chosen


def amp_enabled(requested: bool, device: torch.device) -> bool:
    """Mixed precision is only worth enabling on CUDA.

    On CPU it is slower than full precision for these model sizes, and on MPS the
    autocast implementation is still inconsistent across torch releases.
    """
    if requested and device.type != "cuda":
        logger.info("Mixed precision requested but device is %s; running in fp32", device.type)
        return False
    return requested and device.type == "cuda"
