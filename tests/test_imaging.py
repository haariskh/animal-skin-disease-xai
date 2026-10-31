"""Image decoding, normalisation and heatmap rendering."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from aniderm.utils.imaging import (
    IMAGENET_MEAN,
    ImageDecodeError,
    denormalise,
    image_to_tensor,
    load_image,
    normalise_map,
    overlay_heatmap,
    resize_for_display,
    resize_map,
    to_png_data_uri,
)


def _encode(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_loads_from_bytes_and_path(tmp_path, sample_image) -> None:
    assert load_image(_encode(sample_image)).mode == "RGB"
    path = tmp_path / "x.png"
    sample_image.save(path)
    assert load_image(path).size == sample_image.size


def test_greyscale_and_rgba_are_converted_to_rgb(tmp_path) -> None:
    grey = Image.new("L", (40, 40), color=128)
    assert load_image(_encode(grey, "PNG")).mode == "RGB"

    rgba = Image.new("RGBA", (40, 40), color=(10, 20, 30, 128))
    assert load_image(_encode(rgba, "PNG")).mode == "RGB"


def test_garbage_bytes_raise_a_typed_error() -> None:
    with pytest.raises(ImageDecodeError):
        load_image(b"this is not an image")


def test_one_pixel_image_is_rejected() -> None:
    with pytest.raises(ImageDecodeError, match="too small"):
        load_image(_encode(Image.new("RGB", (1, 1)), "PNG"))


def test_tensor_shape_and_normalisation(sample_image) -> None:
    tensor = image_to_tensor(sample_image, 64)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == torch.float32

    # A mid-grey image normalises to roughly zero on the mean channel values.
    grey = Image.new("RGB", (32, 32), color=tuple(int(c * 255) for c in IMAGENET_MEAN))
    assert torch.allclose(image_to_tensor(grey, 32).mean(), torch.tensor(0.0), atol=0.02)


def test_denormalise_inverts_image_to_tensor(sample_image) -> None:
    tensor = image_to_tensor(sample_image, 64)
    restored = denormalise(tensor)
    assert restored.shape == (64, 64, 3)
    assert restored.min() >= 0.0 and restored.max() <= 1.0

    resized = sample_image.resize((64, 64), Image.Resampling.BILINEAR)
    direct = np.asarray(resized, dtype=np.float32) / 255.0
    assert np.allclose(restored, direct, atol=0.02)


def test_denormalise_rejects_a_batch(sample_image) -> None:
    batch = torch.cat([image_to_tensor(sample_image, 32)] * 2)
    with pytest.raises(ValueError, match="single image"):
        denormalise(batch)


def test_aspect_ratio_is_preserved_when_resizing_for_display() -> None:
    tall = Image.new("RGB", (400, 1600))
    resized = resize_for_display(tall, max_side=800)
    assert max(resized.size) == 800
    assert resized.size == (200, 800)


def test_small_images_are_not_upscaled_for_display() -> None:
    small = Image.new("RGB", (100, 80))
    assert resize_for_display(small, max_side=800).size == (100, 80)


def test_normalise_map_spans_zero_to_one() -> None:
    normalised = normalise_map(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert normalised.min() == pytest.approx(0.0)
    assert normalised.max() == pytest.approx(1.0)


def test_normalise_map_clips_a_single_extreme_outlier() -> None:
    """Without clipping, one hot pixel flattens the rest of the map to near zero.

    Integrated Gradients in particular produces occasional extreme pixels, and an
    unclipped rescale would push every real structure into the bottom of the colour
    range, so the heatmap would look empty.
    """
    base = np.linspace(0.0, 1.0, 400, dtype=np.float32).reshape(20, 20)
    spiked = base.copy()
    spiked[0, 0] = 10_000.0

    unclipped = normalise_map(spiked, percentile_clip=None)
    clipped = normalise_map(spiked, percentile_clip=99.0)

    # The mid-range structure survives clipping but is destroyed without it.
    assert unclipped[10, 10] < 0.01
    assert clipped[10, 10] > 0.4


def test_flat_map_normalises_to_zeros_rather_than_noise() -> None:
    normalised = normalise_map(np.full((10, 10), 0.7, dtype=np.float32))
    assert np.allclose(normalised, 0.0)


def test_nan_and_inf_are_neutralised() -> None:
    saliency = np.array([[np.nan, np.inf], [-np.inf, 1.0]], dtype=np.float32)
    normalised = normalise_map(saliency)
    assert np.isfinite(normalised).all()


def test_resize_map_reaches_the_requested_size() -> None:
    assert resize_map(np.random.rand(7, 7).astype(np.float32), (64, 64)).shape == (64, 64)


def test_resize_map_rejects_a_3d_array() -> None:
    with pytest.raises(ValueError, match="2-D"):
        resize_map(np.zeros((4, 4, 3), dtype=np.float32), (8, 8))


def test_overlay_matches_the_base_image_size() -> None:
    base = np.zeros((32, 32, 3), dtype=np.float32)
    saliency = np.zeros((8, 8), dtype=np.float32)
    saliency[0, 0] = 1.0
    assert overlay_heatmap(base, saliency).size == (32, 32)


def test_cold_regions_stay_close_to_the_original_photograph() -> None:
    """A flat alpha would tint the whole image and bury the thing being explained."""
    base = np.full((32, 32, 3), 0.5, dtype=np.float32)
    saliency = np.zeros((32, 32), dtype=np.float32)
    saliency[:4, :4] = 1.0

    blended = np.asarray(overlay_heatmap(base, saliency, alpha=0.5), dtype=np.float32) / 255.0
    assert np.allclose(blended[20:, 20:], 0.5, atol=0.02)   # untouched
    assert not np.allclose(blended[:4, :4], 0.5, atol=0.02)  # tinted


def test_data_uri_is_a_decodable_png(sample_image) -> None:
    import base64

    uri = to_png_data_uri(sample_image)
    assert uri.startswith("data:image/png;base64,")
    decoded = base64.b64decode(uri.split("base64,", 1)[1])
    assert Image.open(io.BytesIO(decoded)).size == sample_image.size
