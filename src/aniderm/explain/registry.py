"""Construct explainers by name, from config."""

from __future__ import annotations

import torch

from aniderm.config import XAIConfig
from aniderm.explain.base import Explainer
from aniderm.explain.gradcam import GradCAMPlusPlus
from aniderm.explain.integrated_gradients import IntegratedGradientsExplainer
from aniderm.explain.lime_explainer import LimeExplainer
from aniderm.models.classifier import SkinDiseaseClassifier
from aniderm.utils.logging import get_logger

logger = get_logger(__name__)

AVAILABLE_METHODS = {
    GradCAMPlusPlus.name: GradCAMPlusPlus,
    IntegratedGradientsExplainer.name: IntegratedGradientsExplainer,
    LimeExplainer.name: LimeExplainer,
}


def method_catalogue(xai: XAIConfig) -> list[dict[str, str]]:
    """Describe the enabled methods, for the API's ``/api/metadata`` response."""
    catalogue = []
    for name in xai.enabled_methods:
        cls = AVAILABLE_METHODS.get(name)
        if cls is None:
            logger.warning("Config enables unknown XAI method '%s'; ignoring it", name)
            continue
        catalogue.append(
            {
                "name": cls.name,
                "label": cls.label,
                "description": cls.description,
                "is_default": name == xai.default_method,
            }
        )
    return catalogue


def build_explainer(
    name: str,
    model: SkinDiseaseClassifier,
    device: torch.device,
    xai: XAIConfig,
    seed: int = 42,
) -> Explainer:
    """Instantiate one explainer with its configured hyperparameters."""
    if name not in AVAILABLE_METHODS:
        raise KeyError(
            f"Unknown XAI method '{name}'. Available: {sorted(AVAILABLE_METHODS)}"
        )

    if name == GradCAMPlusPlus.name:
        target_layer = model.target_layer(xai.gradcam_target_layer)
        return GradCAMPlusPlus(model=model, device=device, target_layer=target_layer)

    if name == IntegratedGradientsExplainer.name:
        return IntegratedGradientsExplainer(
            model=model,
            device=device,
            steps=xai.ig_steps,
            baseline=xai.ig_baseline,
            internal_batch_size=xai.ig_internal_batch_size,
        )

    return LimeExplainer(
        model=model,
        device=device,
        num_samples=xai.lime_num_samples,
        num_features=xai.lime_num_features,
        batch_size=xai.lime_batch_size,
        segmentation=xai.lime_segmentation,
        seed=seed,
    )
