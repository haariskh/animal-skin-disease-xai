"""Image decoding, tensor conversion and heatmap rendering.

Everything the API needs to turn raw upload bytes into a model input, and a saliency
map back into a picture a human can read, lives here.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageFile, UnidentifiedImageError

# Phone photos are frequently saved with a slightly truncated final chunk. Without this,
# Pillow raises on files that every other viewer opens without complaint.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ImageNet statistics. The backbones are pretrained on ImageNet, so inputs must be
# normalised the same way the pretraining pipeline did.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MAX_SIDE_FOR_DISPLAY = 768


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes are not a readable image."""


def load_image(source: str | Path | bytes | io.BytesIO) -> Image.Image:
    """Open an image from a path or raw bytes and return it as RGB.

    ``exif_transpose`` is applied so a portrait phone photo is not silently rotated
    sideways, which would otherwise shift every saliency map relative to what the user
    sees on screen.
    """
    try:
        if isinstance(source, (str, Path)):
            with Image.open(source) as handle:
                image = handle.convert("RGB")
                # Re-open for EXIF because convert() drops the metadata.
                with Image.open(source) as original:
                    rotated = _apply_exif(original)
                image = rotated.convert("RGB") if rotated is not None else image
        else:
            buffer = source if isinstance(source, io.BytesIO) else io.BytesIO(source)
            buffer.seek(0)
            with Image.open(buffer) as handle:
                rotated = _apply_exif(handle)
                image = (rotated or handle).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError(f"Could not decode image: {exc}") from exc

    if image.width < 2 or image.height < 2:
        raise ImageDecodeError(f"Image is too small to use: {image.width}x{image.height}")
    return image


def _apply_exif(image: Image.Image) -> Image.Image | None:
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(image)
    except Exception:  # pragma: no cover - malformed EXIF should never be fatal
        return None


def resize_for_display(image: Image.Image, max_side: int = MAX_SIDE_FOR_DISPLAY) -> Image.Image:
    """Shrink an oversized image for the browser while keeping the aspect ratio."""
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def to_square(image: Image.Image, size: int) -> Image.Image:
    """Resize to ``size x size``.

    A plain resize is used rather than a centre crop so that nothing is thrown away:
    a lesion sitting near the edge of the frame must still reach the model.
    """
    return image.resize((size, size), Image.Resampling.BILINEAR)


def image_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    """Convert a PIL image into a normalised ``(1, 3, size, size)`` float tensor."""
    resized = to_square(image, size)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = (array - mean) / std
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation and return ``(H, W, 3)`` floats in ``[0, 1]``.

    Accepts either ``(3, H, W)`` or ``(1, 3, H, W)``.
    """
    working = tensor.detach().cpu()
    if working.dim() == 4:
        if working.shape[0] != 1:
            raise ValueError(f"Expected a single image, got a batch of {working.shape[0]}")
        working = working[0]
    if working.dim() != 3 or working.shape[0] != 3:
        raise ValueError(f"Expected a (3, H, W) tensor, got shape {tuple(working.shape)}")

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    restored = working * std + mean
    return restored.permute(1, 2, 0).clamp(0, 1).numpy()


def normalise_map(saliency: np.ndarray, percentile_clip: float = 99.0) -> np.ndarray:
    """Rescale a saliency map to ``[0, 1]``.

    The top percentile is clipped first because integrated gradients in particular tends
    to produce a handful of extreme pixels; without clipping those outliers compress
    everything else into the bottom of the colour range and the heatmap looks empty.
    """
    array = np.asarray(saliency, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Cannot normalise an empty saliency map")

    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if percentile_clip is not None and 0 < percentile_clip < 100:
        ceiling = float(np.percentile(array, percentile_clip))
        if ceiling > float(array.min()):
            array = np.clip(array, None, ceiling)

    low, high = float(array.min()), float(array.max())
    span = high - low
    if span < 1e-12:
        # A perfectly flat map carries no information; return zeros instead of dividing
        # by ~0 and amplifying floating-point noise into a fake pattern.
        return np.zeros_like(array)
    return (array - low) / span


def colourise(saliency: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """Map a ``[0, 1]`` saliency map to an RGB uint8 image."""
    try:
        cmap = colormaps[colormap]
    except KeyError:
        cmap = colormaps["turbo"]
    coloured = cmap(np.clip(saliency, 0.0, 1.0))[..., :3]
    return (coloured * 255).astype(np.uint8)


def overlay_heatmap(
    base: np.ndarray,
    saliency: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "turbo",
) -> Image.Image:
    """Blend a saliency map over the original image.

    ``base`` is ``(H, W, 3)`` in ``[0, 1]``. The per-pixel blend weight is scaled by the
    saliency value itself, so cold regions stay close to the original photograph and only
    the evidence the model used is tinted. A flat ``alpha`` everywhere would wash out the
    whole picture and make the result harder to interpret, not easier.
    """
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError(f"base must be (H, W, 3), got {base.shape}")

    target_hw = base.shape[:2]
    resized = resize_map(saliency, target_hw)
    heat = colourise(resized, colormap).astype(np.float32) / 255.0

    weight = (alpha * resized)[..., None]
    blended = base * (1.0 - weight) + heat * weight
    return Image.fromarray((np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8))


def resize_map(saliency: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinearly resize a 2-D map to ``(height, width)``."""
    array = np.asarray(saliency, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Saliency map must be 2-D, got shape {array.shape}")
    if array.shape == tuple(target_hw):
        return array

    tensor = torch.from_numpy(array)[None, None]
    resized = torch.nn.functional.interpolate(
        tensor, size=tuple(target_hw), mode="bilinear", align_corners=False
    )
    return resized[0, 0].numpy()


def to_png_data_uri(image: Image.Image) -> str:
    """Encode a PIL image as a ``data:image/png;base64,...`` string.

    The API returns images inline this way so the frontend needs no second request and
    the server needs no writable static directory or cache-eviction policy.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def array_to_image(array: np.ndarray) -> Image.Image:
    """Turn an ``(H, W, 3)`` float array in ``[0, 1]`` into a PIL image."""
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255).astype(np.uint8))
