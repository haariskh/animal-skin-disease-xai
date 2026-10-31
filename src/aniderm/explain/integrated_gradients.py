"""Integrated Gradients, via Captum.

Grad-CAM++ answers "which region", at the resolution of the final feature map - 7x7 for
an EfficientNet-B0 at 224px, upsampled for display. Integrated Gradients answers "which
pixels", at full resolution, and it is the method with an axiomatic guarantee behind it:
the attributions of all pixels sum to the difference in model output between the image
and the baseline (completeness). The two are complementary, which is why both ship here
rather than one standing in for the other.

Reference: Sundararajan, Taly and Yan, "Axiomatic Attribution for Deep Networks",
ICML 2017.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from aniderm.explain.base import Explainer, make_baseline
from aniderm.utils.logging import get_logger

logger = get_logger(__name__)


class IntegratedGradientsExplainer(Explainer):
    name = "integrated_gradients"
    label = "Integrated Gradients"
    description = (
        "Accumulates gradients along a straight path from a reference image to the actual "
        "image, giving a per-pixel attribution with a completeness guarantee."
    )

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        steps: int = 64,
        baseline: str = "blur",
        internal_batch_size: int = 16,
    ) -> None:
        super().__init__(model, device)
        self.steps = max(8, int(steps))
        self.baseline_kind = baseline
        self.internal_batch_size = max(1, int(internal_batch_size))

    def explain(self, image_tensor: torch.Tensor, target_index: int) -> np.ndarray:
        from captum.attr import IntegratedGradients

        was_training = self.model.training
        self.model.eval()
        try:
            baseline = make_baseline(image_tensor, self.baseline_kind)
            explainer = IntegratedGradients(self.model)

            with torch.enable_grad():
                attribution = explainer.attribute(
                    inputs=image_tensor,
                    baselines=baseline,
                    target=int(target_index),
                    n_steps=self.steps,
                    internal_batch_size=self.internal_batch_size,
                )
        finally:
            self.model.train(was_training)

        # Sum the signed attribution over the three colour channels, then keep only the
        # positive part. A negative total means that pixel pushed the model AWAY from the
        # predicted class, and the question being asked is what supported the prediction.
        channel_sum = attribution[0].sum(dim=0)
        positive = torch.relu(channel_sum)
        return positive.detach().cpu().numpy().astype(np.float32)
