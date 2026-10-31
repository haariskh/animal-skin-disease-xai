"""The end-to-end analysis pipeline: predict, explain, measure, render."""

from __future__ import annotations

import base64

import pytest
from PIL import Image

from aniderm.config import Config
from aniderm.inference.pipeline import AnalysisPipeline, build_pipeline
from aniderm.inference.predictor import Predictor


@pytest.fixture
def pipeline(checkpoint_path, config: Config) -> AnalysisPipeline:
    return build_pipeline(
        checkpoint=checkpoint_path,
        xai=config.xai,
        device_preference="cpu",
        low_confidence_threshold=config.serving.low_confidence_threshold,
        seed=config.project.seed,
    )


def test_probabilities_sum_to_one_and_are_ordered(pipeline, sample_image) -> None:
    prediction = pipeline.predictor.predict(sample_image)
    probabilities = [score.probability for score in prediction.scores]
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)
    assert probabilities == sorted(probabilities, reverse=True)


def test_margin_and_entropy_are_in_range(pipeline, sample_image) -> None:
    prediction = pipeline.predictor.predict(sample_image)
    assert 0.0 <= prediction.margin <= 1.0
    assert 0.0 <= prediction.entropy <= 1.0


def test_analysis_produces_one_explanation_per_requested_method(pipeline, sample_image) -> None:
    result = pipeline.analyse(sample_image, methods=["gradcam_pp", "integrated_gradients"])
    assert [item.method for item in result.explanations] == [
        "gradcam_pp",
        "integrated_gradients",
    ]


def test_every_saliency_map_is_normalised_to_the_model_input_size(pipeline, sample_image) -> None:
    """Grad-CAM starts at feature-map resolution; everything downstream needs one geometry."""
    result = pipeline.analyse(sample_image, methods=["gradcam_pp", "integrated_gradients"])
    size = pipeline.predictor.image_size
    for explanation in result.explanations:
        assert explanation.saliency.shape == (size, size)
        assert 0.0 <= explanation.saliency.min()
        assert explanation.saliency.max() <= 1.0


def test_images_come_back_as_decodable_pngs(pipeline, sample_image) -> None:
    import io

    result = pipeline.analyse(sample_image, methods=["gradcam_pp"])
    for uri in (
        result.source_data_uri,
        result.model_input_data_uri,
        result.explanations[0].overlay_data_uri,
        result.explanations[0].heatmap_data_uri,
    ):
        assert uri.startswith("data:image/png;base64,")
        Image.open(io.BytesIO(base64.b64decode(uri.split("base64,", 1)[1]))).verify()


def test_overlay_is_aligned_with_the_model_input(pipeline, sample_image) -> None:
    """A misaligned overlay would point at the wrong part of the animal."""
    import io

    result = pipeline.analyse(sample_image, methods=["gradcam_pp"])
    size = pipeline.predictor.image_size

    def dimensions(uri: str) -> tuple[int, int]:
        return Image.open(io.BytesIO(base64.b64decode(uri.split("base64,", 1)[1]))).size

    assert dimensions(result.model_input_data_uri) == (size, size)
    assert dimensions(result.explanations[0].overlay_data_uri) == (size, size)


def test_explaining_a_non_top_class_is_supported(pipeline, sample_image) -> None:
    result = pipeline.analyse(
        sample_image, methods=["gradcam_pp"], target_class="healthy"
    )
    assert result.explained_class_key == "healthy"
    assert result.to_dict()["explained_class"]["is_top_prediction"] == (
        result.prediction.top.key == "healthy"
    )


def test_unknown_target_class_raises(pipeline, sample_image) -> None:
    with pytest.raises(KeyError):
        pipeline.analyse(sample_image, methods=["gradcam_pp"], target_class="eczema")


def test_disabled_method_is_refused(pipeline, sample_image) -> None:
    with pytest.raises(KeyError):
        pipeline.analyse(sample_image, methods=["shap"])


def test_faithfulness_can_be_skipped(pipeline, sample_image) -> None:
    with_check = pipeline.analyse(sample_image, methods=["gradcam_pp"], include_faithfulness=True)
    without = pipeline.analyse(sample_image, methods=["gradcam_pp"], include_faithfulness=False)
    assert with_check.explanations[0].faithfulness
    assert without.explanations[0].faithfulness == {}


def test_low_confidence_is_reported_rather_than_hidden(checkpoint_path, config: Config) -> None:
    """An untrained model is near-uniform over 3 classes, which must produce a warning."""
    strict = AnalysisPipeline(
        predictor=Predictor(checkpoint_path, "cpu"),
        xai=config.xai,
        low_confidence_threshold=0.99,
    )
    import numpy as np

    noise = Image.fromarray(
        np.random.default_rng(1).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    )
    result = strict.analyse(noise, methods=["gradcam_pp"])
    assert any("inconclusive" in warning for warning in result.warnings)


def test_a_failing_explainer_does_not_sink_the_whole_response(
    pipeline, sample_image, monkeypatch
) -> None:
    import aniderm.inference.pipeline as pipeline_module

    real_builder = pipeline_module.build_explainer

    def flaky(name, *args, **kwargs):
        if name == "integrated_gradients":
            raise RuntimeError("simulated CUDA OOM")
        return real_builder(name, *args, **kwargs)

    monkeypatch.setattr(pipeline_module, "build_explainer", flaky)
    result = pipeline.analyse(sample_image, methods=["gradcam_pp", "integrated_gradients"])

    assert [item.method for item in result.explanations] == ["gradcam_pp"]
    assert any("integrated_gradients" in warning for warning in result.warnings)


def test_result_serialises_to_plain_json(pipeline, sample_image) -> None:
    import json

    payload = pipeline.analyse(sample_image, methods=["gradcam_pp"]).to_dict()
    json.dumps(payload)  # must not raise on numpy scalars
    assert payload["prediction"]["class_key"]
    assert payload["explanations"][0]["regions"]["summary"]
    assert payload["timings_ms"]
