"""Loss construction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from aniderm.utils.logging import get_logger

logger = get_logger(__name__)


def build_criterion(
    class_weighting: str,
    weights: np.ndarray | None,
    label_smoothing: float,
    device: torch.device,
) -> nn.Module:
    """Cross-entropy, optionally class-weighted and label-smoothed.

    Class weighting matters on this dataset because the largest class holds roughly 1.5x
    the images of the smallest; without it the model drifts toward the majority class and
    macro-F1 suffers even while accuracy looks acceptable.

    Label smoothing is kept small (0.05). Dermatological categories genuinely overlap -
    a bacterial infection secondary to an allergy can look like either - so refusing to
    let the model become fully confident is a reasonable prior, but smoothing too hard
    would flatten the probability output the UI relies on.
    """
    weight_tensor: torch.Tensor | None = None
    if class_weighting == "balanced":
        if weights is None:
            raise ValueError("class_weighting='balanced' requires class weights")
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        logger.info("Class weights: %s", np.round(weights, 3).tolist())

    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
