"""LIME for images - a model-agnostic second opinion.

Grad-CAM++ and Integrated Gradients both read the model's internals. LIME does not: it
segments the image into superpixels, switches groups of them off, watches how the
prediction moves, and fits a sparse linear model to that behaviour. Because it shares no
machinery with the gradient methods, agreement between them is genuine corroboration
rather than two views of the same computation.

The trade-off is cost - several hundred forward passes per image - and variance, since
the perturbations are sampled. The random seed is fixed so a given image yields the same
explanation twice.

Reference: Ribeiro, Singh and Guestrin, "Why Should I Trust You?", KDD 2016.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from aniderm.explain.base import Explainer
from aniderm.utils.imaging import IMAGENET_MEAN, IMAGENET_STD, denormalise
from aniderm.utils.logging import get_logger

logger = get_logger(__name__)


class LimeExplainer(Explainer):
    name = "lime"
    label = "LIME"
    description = (
        "Perturbs superpixel regions of the image and fits a sparse linear model to the "
        "resulting prediction changes. Treats the network as a black box."
    )

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_samples: int = 800,
        num_features: int = 8,
        batch_size: int = 32,
        segmentation: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(model, device)
        self.num_samples = max(50, int(num_samples))
        self.num_features = max(1, int(num_features))
        self.batch_size = max(1, int(batch_size))
        self.segmentation = segmentation or {"kernel_size": 4, "max_dist": 200, "ratio": 0.2}
        self.seed = seed

    def explain(self, image_tensor: torch.Tensor, target_index: int) -> np.ndarray:
        from lime.lime_image import LimeImageExplainer
        from lime.wrappers.scikit_image import SegmentationAlgorithm

        # LIME works on plain RGB images, so the normalised tensor is converted back to
        # [0, 1] pixels and re-normalised inside the prediction callback.
        rgb = denormalise(image_tensor).astype(np.double)

        segmenter = SegmentationAlgorithm(
            "quickshift",
            kernel_size=self.segmentation.get("kernel_size", 4),
            max_dist=self.segmentation.get("max_dist", 200),
            ratio=self.segmentation.get("ratio", 0.2),
            random_seed=self.seed,
        )
        explainer = LimeImageExplainer(random_state=self.seed)

        kwargs: dict[str, Any] = {
            "image": rgb,
            "classifier_fn": self._batch_predict,
            "labels": (int(target_index),),
            "top_labels": None,
            "hide_color": None,  # occlude with the mean colour of each segment
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "segmentation_fn": segmenter,
            "random_seed": self.seed,
        }
        # lime 0.2.0.1 has no progress-bar switch; later versions do. Passing it
        # unconditionally raises a TypeError on the released version, so it is added
        # only when the installed signature accepts it. Suppressing the bar keeps the
        # server log readable, since the API reports its own timing.
        if "progress_bar" in inspect.signature(explainer.explain_instance).parameters:
            kwargs["progress_bar"] = False

        was_training = self.model.training
        self.model.eval()
        try:
            explanation = explainer.explain_instance(**kwargs)
        finally:
            self.model.train(was_training)

        return self._weights_to_map(explanation, int(target_index), rgb.shape[:2])

    def _batch_predict(self, images: np.ndarray) -> np.ndarray:
        """Classify a batch of perturbed RGB images for LIME.

        ``images`` arrives as ``(N, H, W, 3)`` floats in ``[0, 1]``.
        """
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

        batch = torch.from_numpy(images.astype(np.float32)).permute(0, 3, 1, 2)
        batch = ((batch - mean) / std).to(self.device)

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, batch.shape[0], self.batch_size):
                chunk = batch[start : start + self.batch_size]
                logits = self.model(chunk)
                outputs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _weights_to_map(
        self,
        explanation: Any,
        target_index: int,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Paint each superpixel's regression weight back onto the image grid.

        LIME's own ``get_image_and_mask`` returns a binary mask of the top-k segments.
        Using the raw weights instead keeps the relative strength of each region, which
        renders as a graded heatmap comparable with the other two methods rather than a
        flat cut-out.
        """
        if target_index not in explanation.local_exp:
            available = list(explanation.local_exp.keys())
            raise KeyError(
                f"LIME produced no explanation for label {target_index}; it has {available}"
            )

        segments = explanation.segments
        weights = explanation.local_exp[target_index]
        ranked = sorted(weights, key=lambda item: abs(item[1]), reverse=True)
        ranked = ranked[: self.num_features]

        saliency = np.zeros(shape, dtype=np.float32)
        for segment_id, weight in ranked:
            # Only positive weights support the prediction; negative ones argue against
            # it and are dropped so the map means the same thing as the others.
            if weight > 0:
                saliency[segments == segment_id] = float(weight)
        return saliency
