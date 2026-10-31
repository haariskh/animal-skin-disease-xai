"""The explainability layer: correctness of the maps, not just that they run.

The sharpest test here is `test_gradcam_finds_the_patch_the_model_actually_uses`, which
builds a model that provably depends on one corner of the image and then checks the
heatmap points there. Shape assertions alone would pass on a method that returned noise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from aniderm.config import Config
from aniderm.explain.base import make_baseline
from aniderm.explain.faithfulness import deletion_insertion
from aniderm.explain.gradcam import GradCAMPlusPlus
from aniderm.explain.integrated_gradients import IntegratedGradientsExplainer
from aniderm.explain.lime_explainer import LimeExplainer
from aniderm.explain.regions import analyse, concentration, hotspots
from aniderm.explain.registry import AVAILABLE_METHODS, build_explainer, method_catalogue
from aniderm.utils.imaging import normalise_map, resize_map

DEVICE = torch.device("cpu")


class CornerModel(nn.Module):
    """A model whose class-1 score depends only on the top-left quadrant.

    Being able to state the ground truth is what makes the localisation assertions below
    meaningful rather than circular.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        with torch.no_grad():
            self.features.weight.fill_(0.05)
            self.features.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = self.act(self.features(x))
        half_h = maps.shape[2] // 2
        half_w = maps.shape[3] // 2
        corner = maps[:, :, :half_h, :half_w].mean(dim=(1, 2, 3))
        everything = maps.mean(dim=(1, 2, 3))
        # Class 1 responds to the top-left corner; class 0 is a flat reference.
        return torch.stack([everything * 0.0, corner * 10.0], dim=1)


def test_gradcam_finds_the_patch_the_model_actually_uses() -> None:
    model = CornerModel().eval()
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, 4:12, 4:12] = 2.0  # a bright square inside the top-left quadrant

    explainer = GradCAMPlusPlus(model=model, device=DEVICE, target_layer=model.act)
    cam = normalise_map(resize_map(explainer.explain(image, target_index=1), (32, 32)))

    top_left = cam[:16, :16].mean()
    bottom_right = cam[16:, 16:].mean()
    assert top_left > bottom_right
    assert np.unravel_index(int(cam.argmax()), cam.shape)[0] < 16


def test_gradcam_output_is_non_negative() -> None:
    """The final ReLU is the point of the method; evidence against a class must not leak in."""
    model = CornerModel().eval()
    image = torch.randn(1, 3, 32, 32)
    explainer = GradCAMPlusPlus(model=model, device=DEVICE, target_layer=model.act)
    assert explainer.explain(image, target_index=1).min() >= 0.0


def test_gradcam_is_deterministic(untrained_model, sample_tensor) -> None:
    explainer = GradCAMPlusPlus(
        model=untrained_model, device=DEVICE, target_layer=untrained_model.target_layer()
    )
    first = explainer.explain(sample_tensor, 0)
    second = explainer.explain(sample_tensor, 0)
    assert np.allclose(first, second)


def test_gradcam_removes_its_hooks(untrained_model, sample_tensor) -> None:
    """A leaked hook would slow every later request and hold the graph alive."""
    layer = untrained_model.target_layer()
    before = len(layer._forward_hooks)
    explainer = GradCAMPlusPlus(model=untrained_model, device=DEVICE, target_layer=layer)
    explainer.explain(sample_tensor, 0)
    assert len(layer._forward_hooks) == before


def test_gradcam_rejects_an_out_of_range_class(untrained_model, sample_tensor) -> None:
    explainer = GradCAMPlusPlus(
        model=untrained_model, device=DEVICE, target_layer=untrained_model.target_layer()
    )
    with pytest.raises(IndexError):
        explainer.explain(sample_tensor, 99)


def test_gradcam_leaves_the_model_in_its_original_mode(untrained_model, sample_tensor) -> None:
    untrained_model.train()
    explainer = GradCAMPlusPlus(
        model=untrained_model, device=DEVICE, target_layer=untrained_model.target_layer()
    )
    explainer.explain(sample_tensor, 0)
    assert untrained_model.training is True
    untrained_model.eval()


def test_integrated_gradients_shape_and_sign(untrained_model, sample_tensor) -> None:
    explainer = IntegratedGradientsExplainer(
        model=untrained_model, device=DEVICE, steps=8, internal_batch_size=4
    )
    attribution = explainer.explain(sample_tensor, 0)
    assert attribution.shape == tuple(sample_tensor.shape[-2:])
    assert attribution.min() >= 0.0  # only positive evidence is kept


def test_integrated_gradients_locates_the_corner() -> None:
    model = CornerModel().eval()
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, 2:14, 2:14] = 3.0

    explainer = IntegratedGradientsExplainer(
        model=model, device=DEVICE, steps=16, baseline="zero", internal_batch_size=8
    )
    attribution = normalise_map(explainer.explain(image, target_index=1))
    assert attribution[:16, :16].mean() > attribution[16:, 16:].mean()


def test_lime_returns_a_map_at_input_resolution(untrained_model, sample_tensor) -> None:
    explainer = LimeExplainer(
        model=untrained_model, device=DEVICE, num_samples=60, num_features=4, batch_size=20
    )
    saliency = explainer.explain(sample_tensor, 0)
    assert saliency.shape == tuple(sample_tensor.shape[-2:])
    assert saliency.min() >= 0.0


def test_lime_is_reproducible_for_a_fixed_seed(untrained_model, sample_tensor) -> None:
    kwargs = dict(model=untrained_model, device=DEVICE, num_samples=60, num_features=4, seed=5)
    first = LimeExplainer(**kwargs).explain(sample_tensor, 0)
    second = LimeExplainer(**kwargs).explain(sample_tensor, 0)
    assert np.allclose(first, second)


# --- baselines ------------------------------------------------------------------ #
@pytest.mark.parametrize("kind", ["zero", "blur", "black", "white", "gaussian_noise"])
def test_baselines_keep_the_input_shape(sample_tensor, kind: str) -> None:
    assert make_baseline(sample_tensor, kind).shape == sample_tensor.shape


def test_zero_baseline_is_not_the_same_as_black(sample_tensor) -> None:
    """Zero in normalised space is mid grey, not black; conflating them skews attributions."""
    assert not torch.allclose(
        make_baseline(sample_tensor, "zero"), make_baseline(sample_tensor, "black")
    )


def test_blur_baseline_preserves_average_colour(sample_tensor) -> None:
    blurred = make_baseline(sample_tensor, "blur")
    assert torch.allclose(blurred.mean(dim=(2, 3)), sample_tensor.mean(dim=(2, 3)), atol=0.25)


def test_unknown_baseline_is_rejected(sample_tensor) -> None:
    with pytest.raises(ValueError, match="Unknown baseline"):
        make_baseline(sample_tensor, "rainbow")


# --- faithfulness --------------------------------------------------------------- #
def test_an_honest_map_beats_a_random_one_on_deletion() -> None:
    model = CornerModel().eval()
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, 2:14, 2:14] = 3.0

    truthful = np.zeros((32, 32), dtype=np.float32)
    truthful[2:14, 2:14] = 1.0

    scores = deletion_insertion(
        model=model,
        image_tensor=image,
        saliency=truthful,
        target_index=1,
        steps=8,
        include_random_control=True,
    )
    assert scores["deletion_auc"] < scores["random_deletion_auc"]
    assert len(scores["deletion_curve"]) == 9
    assert scores["verdict"]


def test_faithfulness_rejects_a_mismatched_saliency_size(sample_tensor, untrained_model) -> None:
    with pytest.raises(ValueError, match="does not match"):
        deletion_insertion(
            model=untrained_model,
            image_tensor=sample_tensor,
            saliency=np.zeros((8, 8), dtype=np.float32),
            target_index=0,
            steps=3,
        )


# --- region analysis ------------------------------------------------------------ #
def test_regions_locate_a_known_blob() -> None:
    saliency = np.zeros((90, 90), dtype=np.float32)
    saliency[5:25, 5:25] = 1.0  # top-left
    spots = hotspots(saliency)
    assert len(spots) == 1
    assert spots[0]["location"] == "top left"
    assert spots[0]["evidence_share"] == pytest.approx(1.0, abs=1e-3)


def test_regions_separate_two_blobs() -> None:
    saliency = np.zeros((90, 90), dtype=np.float32)
    saliency[5:20, 5:20] = 1.0
    saliency[70:85, 70:85] = 0.8
    spots = hotspots(saliency, max_regions=3)
    assert len(spots) == 2
    assert {spots[0]["location"], spots[1]["location"]} == {"top left", "bottom right"}


def test_concentration_distinguishes_focused_from_flat() -> None:
    focused = np.zeros((64, 64), dtype=np.float32)
    focused[30:34, 30:34] = 1.0
    flat = np.full((64, 64), 0.5, dtype=np.float32)

    assert concentration(focused)["top_10pct_share"] > 0.9
    assert concentration(flat)["top_10pct_share"] == pytest.approx(0.1, abs=0.01)


def test_flat_map_is_described_as_uninformative() -> None:
    summary = analyse(np.zeros((40, 40), dtype=np.float32))["summary"]
    assert "flat" in summary.lower()


def test_small_regions_are_not_reported_as_zero_percent() -> None:
    saliency = np.zeros((128, 128), dtype=np.float32)
    saliency[60:64, 60:64] = 1.0
    assert "0% of the image" not in analyse(saliency)["summary"]


# --- registry --------------------------------------------------------------------- #
def test_registry_builds_every_enabled_method(config: Config, untrained_model) -> None:
    for name in config.xai.enabled_methods:
        explainer = build_explainer(name, untrained_model, DEVICE, config.xai)
        assert explainer.name == name
        assert explainer.label and explainer.description


def test_registry_rejects_an_unknown_method(config: Config, untrained_model) -> None:
    with pytest.raises(KeyError, match="Unknown XAI method"):
        build_explainer("shap", untrained_model, DEVICE, config.xai)


def test_catalogue_marks_exactly_one_default(config: Config) -> None:
    catalogue = method_catalogue(config.xai)
    assert len(catalogue) == len(config.xai.enabled_methods)
    assert sum(1 for entry in catalogue if entry["is_default"]) == 1


def test_all_registered_methods_are_documented() -> None:
    for cls in AVAILABLE_METHODS.values():
        assert cls.description.strip()
        assert cls.label.strip()
