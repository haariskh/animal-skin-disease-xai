"""Explainability: Grad-CAM++, Integrated Gradients, LIME, and their evaluation."""

from aniderm.explain.base import Explainer, Explanation, make_baseline
from aniderm.explain.faithfulness import deletion_insertion
from aniderm.explain.gradcam import GradCAMPlusPlus
from aniderm.explain.integrated_gradients import IntegratedGradientsExplainer
from aniderm.explain.lime_explainer import LimeExplainer
from aniderm.explain.regions import analyse as analyse_regions
from aniderm.explain.registry import AVAILABLE_METHODS, build_explainer, method_catalogue

__all__ = [
    "AVAILABLE_METHODS",
    "Explainer",
    "Explanation",
    "GradCAMPlusPlus",
    "IntegratedGradientsExplainer",
    "LimeExplainer",
    "analyse_regions",
    "build_explainer",
    "deletion_insertion",
    "make_baseline",
    "method_catalogue",
]
