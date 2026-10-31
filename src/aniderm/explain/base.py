"""The common interface every explainer implements.

Having one shape of result for Grad-CAM++, Integrated Gradients and LIME is what lets
the API and the UI treat them interchangeably, and lets the faithfulness check score
them on equal terms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class Explanation:
    """One saliency map plus everything derived from it."""

    method: str
    method_label: str
    target_index: int
    target_class_key: str
    # 2-D map normalised to [0, 1]; the same spatial size as the model input.
    saliency: np.ndarray
    runtime_ms: float
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    # Filled in later by the region analysis and the faithfulness check.
    regions: dict[str, Any] = field(default_factory=dict)
    faithfulness: dict[str, Any] = field(default_factory=dict)
    overlay_data_uri: str = ""
    heatmap_data_uri: str = ""

    def __post_init__(self) -> None:
        if self.saliency.ndim != 2:
            raise ValueError(
                f"Explanation saliency must be 2-D, got shape {self.saliency.shape}"
            )


class Explainer(ABC):
    """Base class for saliency methods.

    Subclasses implement :meth:`explain` and return a raw, unnormalised 2-D map; the
    normalisation, colouring and region analysis are handled once, centrally, so every
    method is rendered identically.
    """

    name: str = "explainer"
    label: str = "Explainer"
    description: str = ""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    @abstractmethod
    def explain(self, image_tensor: torch.Tensor, target_index: int) -> np.ndarray:
        """Return a raw 2-D saliency map for ``target_index``.

        ``image_tensor`` is a normalised ``(1, 3, H, W)`` tensor already on ``device``.
        """

    def supports(self, _model: torch.nn.Module) -> bool:
        return True


# Pixel values are normalised with the ImageNet statistics, so "zero" in tensor space is
# the dataset mean colour (a mid grey), not black. These constants let a true black or
# white be requested explicitly.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _constant_baseline(image_tensor: torch.Tensor, pixel_value: float) -> torch.Tensor:
    """A uniform image of ``pixel_value`` (in 0-1 pixel space), normalised like the input."""
    mean = torch.tensor(_MEAN, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=image_tensor.device).view(1, 3, 1, 1)
    constant = torch.full_like(image_tensor, pixel_value)
    return (constant - mean) / std


def make_baseline(image_tensor: torch.Tensor, kind: str = "blur") -> torch.Tensor:
    """Build the reference input that attribution methods measure against.

    The choice matters more than it first appears, and the right answer differs by use.

    * ``zero`` - zeros in normalised space, which is the ImageNet mean colour, a neutral
      mid grey. The conventional Integrated Gradients baseline and the conventional
      deletion reference.
    * ``blur`` - a blurred copy of the image. Asks "what if this region had no detail"
      while leaving brightness and colour intact, so it isolates structure. The right
      reference for insertion, and a better IG baseline than black on photographs of dark
      fur or shadowed skin, where a dark baseline understates dark pixels.
    * ``black`` / ``white`` - true black or white pixels. Blunt, but occasionally the
      honest question.

    Note that ``blur`` preserves colour by construction. If the signal a model relies on
    is colour rather than texture - which it partly is in dermatology - a blur reference
    will understate how much the model depends on the highlighted region. That is exactly
    why the deletion curve uses a flat reference and only insertion uses blur.
    """
    kind = (kind or "blur").lower()

    if kind == "zero":
        return torch.zeros_like(image_tensor)
    if kind == "black":
        return _constant_baseline(image_tensor, 0.0)
    if kind == "white":
        return _constant_baseline(image_tensor, 1.0)
    if kind == "gaussian_noise":
        generator = torch.Generator(device="cpu").manual_seed(0)
        noise = torch.randn(image_tensor.shape, generator=generator).to(image_tensor.device)
        return noise * 0.5
    if kind == "blur":
        from torchvision.transforms.functional import gaussian_blur

        # A kernel roughly 1/8 of the image side removes local texture while leaving the
        # overall composition recognisable.
        side = min(image_tensor.shape[-2:])
        kernel = max(3, int(side // 8) | 1)  # force odd, as gaussian_blur requires
        return gaussian_blur(image_tensor, kernel_size=[kernel, kernel], sigma=[kernel / 3.0])

    raise ValueError(
        f"Unknown baseline '{kind}'. Use zero, blur, black, white or gaussian_noise."
    )
