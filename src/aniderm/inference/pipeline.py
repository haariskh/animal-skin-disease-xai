"""Predict, explain, measure, render - the single call the API makes.

Keeping this orchestration in the library rather than in the route handler means the CLI
(``scripts/explain_image.py``) and the web API produce byte-identical results, and the
whole thing stays testable without starting a server.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aniderm.config import XAIConfig
from aniderm.explain.base import Explanation
from aniderm.explain.faithfulness import deletion_insertion
from aniderm.explain.regions import analyse as analyse_regions
from aniderm.explain.registry import AVAILABLE_METHODS, build_explainer
from aniderm.inference.predictor import Prediction, Predictor
from aniderm.utils.imaging import (
    array_to_image,
    denormalise,
    normalise_map,
    overlay_heatmap,
    resize_for_display,
    resize_map,
    to_png_data_uri,
)
from aniderm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """The complete answer for one uploaded image."""

    prediction: Prediction
    explanations: list[Explanation]
    source_data_uri: str
    model_input_data_uri: str
    explained_class_key: str
    explained_class_display_name: str
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        top = self.prediction.top
        return {
            "prediction": {
                "class_key": top.key,
                "display_name": top.display_name,
                "probability": round(top.probability, 6),
                "colour": top.colour,
                "summary": top.summary,
                "margin": round(self.prediction.margin, 6),
                "entropy": round(self.prediction.entropy, 6),
            },
            "scores": [score.to_dict() for score in self.prediction.scores],
            "explained_class": {
                "index": self.prediction.target_index,
                "key": self.explained_class_key,
                "display_name": self.explained_class_display_name,
                "is_top_prediction": self.explained_class_key == self.prediction.top.key,
            },
            "explanations": [
                {
                    "method": item.method,
                    "method_label": item.method_label,
                    "description": item.description,
                    "target_class_key": item.target_class_key,
                    "runtime_ms": round(item.runtime_ms, 1),
                    "parameters": item.parameters,
                    "regions": item.regions,
                    "faithfulness": item.faithfulness,
                    "overlay": item.overlay_data_uri,
                    "heatmap": item.heatmap_data_uri,
                }
                for item in self.explanations
            ],
            "images": {
                "source": self.source_data_uri,
                "model_input": self.model_input_data_uri,
            },
            "warnings": self.warnings,
            "timings_ms": {key: round(value, 1) for key, value in self.timings_ms.items()},
        }


class AnalysisPipeline:
    def __init__(
        self,
        predictor: Predictor,
        xai: XAIConfig,
        low_confidence_threshold: float = 0.55,
        seed: int = 42,
    ) -> None:
        self.predictor = predictor
        self.xai = xai
        self.low_confidence_threshold = low_confidence_threshold
        self.seed = seed

    def analyse(
        self,
        image: Image.Image,
        methods: list[str] | None = None,
        target_class: str | None = None,
        include_faithfulness: bool | None = None,
    ) -> AnalysisResult:
        timings: dict[str, float] = {}

        started = time.perf_counter()
        prediction = self.predictor.predict(image, target_class=target_class)
        timings["prediction"] = (time.perf_counter() - started) * 1000

        requested = methods or list(self.xai.enabled_methods)
        selected = self._validate_methods(requested)

        # The overlay base is the square-resized view the model actually saw. Overlaying
        # on the original aspect ratio would shift every heatmap pixel relative to the
        # photograph, which is exactly the kind of quiet error that makes an explanation
        # misleading.
        base_rgb = denormalise(prediction.image_tensor)
        target_key = self.predictor.class_specs[prediction.target_index].key

        explanations: list[Explanation] = []
        warnings: list[str] = []
        want_faithfulness = (
            self.xai.faithfulness_enabled if include_faithfulness is None else include_faithfulness
        )

        for name in selected:
            try:
                explanation = self._run_method(
                    name=name,
                    prediction=prediction,
                    base_rgb=base_rgb,
                    target_key=target_key,
                    include_faithfulness=want_faithfulness,
                )
            except Exception as exc:
                # One failing method must not sink the whole response; the user still
                # gets the prediction and whatever other explanations succeeded.
                logger.exception("Explainer '%s' failed", name)
                warnings.append(f"{name} could not be computed: {exc}")
                continue
            explanations.append(explanation)
            timings[name] = explanation.runtime_ms

        warnings.extend(self._confidence_warnings(prediction))
        if not explanations:
            warnings.append("No explanation method completed successfully for this image.")

        target_spec = self.predictor.class_specs[prediction.target_index]
        return AnalysisResult(
            prediction=prediction,
            explanations=explanations,
            source_data_uri=to_png_data_uri(resize_for_display(image)),
            model_input_data_uri=to_png_data_uri(array_to_image(base_rgb)),
            explained_class_key=target_spec.key,
            explained_class_display_name=target_spec.display_name,
            warnings=warnings,
            timings_ms=timings,
        )

    # -- internals ---------------------------------------------------------- #
    def _validate_methods(self, requested: list[str]) -> list[str]:
        selected = []
        for name in requested:
            if name not in AVAILABLE_METHODS:
                raise KeyError(
                    f"Unknown XAI method '{name}'. Available: {sorted(AVAILABLE_METHODS)}"
                )
            if name not in self.xai.enabled_methods:
                raise KeyError(f"Method '{name}' is not enabled in the config")
            selected.append(name)
        return selected or [self.xai.default_method]

    def _run_method(
        self,
        name: str,
        prediction: Prediction,
        base_rgb: np.ndarray,
        target_key: str,
        include_faithfulness: bool,
    ) -> Explanation:
        explainer = build_explainer(
            name=name,
            model=self.predictor.model,
            device=self.predictor.device,
            xai=self.xai,
            seed=self.seed,
        )

        started = time.perf_counter()
        raw = explainer.explain(prediction.image_tensor, prediction.target_index)
        runtime_ms = (time.perf_counter() - started) * 1000

        # Grad-CAM++ returns a feature-map-sized grid (7x7 for an EfficientNet-B0 at
        # 224px); the pixel methods return full resolution. Upsampling here means
        # everything downstream - regions, faithfulness, overlay - sees one geometry.
        height, width = prediction.image_tensor.shape[-2:]
        saliency = normalise_map(resize_map(raw, (height, width)))

        explanation = Explanation(
            method=explainer.name,
            method_label=explainer.label,
            target_index=prediction.target_index,
            target_class_key=target_key,
            saliency=saliency,
            runtime_ms=runtime_ms,
            description=explainer.description,
            parameters=self._parameters_for(name),
        )
        explanation.regions = analyse_regions(saliency)
        explanation.overlay_data_uri = to_png_data_uri(
            overlay_heatmap(
                base=base_rgb,
                saliency=saliency,
                alpha=self.xai.overlay_alpha,
                colormap=self.xai.colormap,
            )
        )
        explanation.heatmap_data_uri = to_png_data_uri(
            overlay_heatmap(
                base=np.zeros_like(base_rgb),
                saliency=saliency,
                alpha=1.0,
                colormap=self.xai.colormap,
            )
        )

        if include_faithfulness:
            try:
                explanation.faithfulness = deletion_insertion(
                    model=self.predictor.model,
                    image_tensor=prediction.image_tensor,
                    saliency=saliency,
                    target_index=prediction.target_index,
                    steps=self.xai.faithfulness_steps,
                    deletion_baseline=self.xai.faithfulness_deletion_baseline,
                    insertion_baseline=self.xai.faithfulness_insertion_baseline,
                    seed=self.seed,
                )
            except Exception as exc:
                logger.warning("Faithfulness check failed for %s: %s", name, exc)
                explanation.faithfulness = {"error": str(exc)}

        return explanation

    def _parameters_for(self, name: str) -> dict[str, Any]:
        if name == "integrated_gradients":
            return {"steps": self.xai.ig_steps, "baseline": self.xai.ig_baseline}
        if name == "lime":
            return {
                "num_samples": self.xai.lime_num_samples,
                "num_features": self.xai.lime_num_features,
                "segmentation": self.xai.lime_segmentation,
            }
        return {"target_layer": self.predictor.model.target_layer_name}

    def _confidence_warnings(self, prediction: Prediction) -> list[str]:
        """Say out loud when the output should not be read as a finding."""
        notes: list[str] = []
        top = prediction.top

        if top.probability < self.low_confidence_threshold:
            notes.append(
                f"Top probability is only {top.probability:.0%}, below the "
                f"{self.low_confidence_threshold:.0%} threshold. Treat this as inconclusive "
                f"rather than as a result."
            )
        if prediction.margin < 0.15 and len(prediction.scores) > 1:
            runner_up = prediction.scores[1]
            notes.append(
                f"'{top.display_name}' and '{runner_up.display_name}' are within "
                f"{prediction.margin:.0%} of each other, so the model is not separating them."
            )
        if prediction.entropy > 0.85:
            notes.append(
                "The probability distribution is close to uniform, which usually means the "
                "image is unlike anything in the training data."
            )
        return notes


def build_pipeline(
    checkpoint: str | Path,
    xai: XAIConfig,
    device_preference: str = "auto",
    low_confidence_threshold: float = 0.55,
    seed: int = 42,
) -> AnalysisPipeline:
    predictor = Predictor(checkpoint_path=checkpoint, device_preference=device_preference)
    return AnalysisPipeline(
        predictor=predictor,
        xai=xai,
        low_confidence_threshold=low_confidence_threshold,
        seed=seed,
    )
