"""Seeding helpers so a run can be repeated."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch.

    With ``deterministic=True`` cuDNN's autotuner is switched off and deterministic
    kernels are requested, which makes repeated runs match bit for bit at the cost of
    roughly 10-20% throughput. Some ops have no deterministic kernel, so the stricter
    ``use_deterministic_algorithms`` call is allowed to fail with a warning rather than
    bring the run down.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Required by several deterministic CUDA matmul kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - depends on the torch build
            pass
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a distinct but reproducible seed.

    Without this, forked workers share a NumPy seed and generate identical augmentation
    sequences, which quietly reduces the effective diversity of the training data.
    """
    base = torch.initial_seed() % 2**31
    seed = (base + worker_id) % 2**31
    random.seed(seed)
    np.random.seed(seed)
