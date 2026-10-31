"""Augmentation and evaluation transforms.

Design note on the photometric settings: in dermatology, colour carries diagnostic
signal - erythema is redness, a fungal lesion has a characteristic tone. So hue jitter is
kept very small (0.02 by default) while geometry is varied freely. Augmenting colour
aggressively here would teach the model to ignore the very cue a clinician uses.
"""

from __future__ import annotations

from torchvision import transforms

from aniderm.config import AugmentationConfig
from aniderm.utils.imaging import IMAGENET_MEAN, IMAGENET_STD


def build_train_transform(image_size: int, cfg: AugmentationConfig) -> transforms.Compose:
    """Training pipeline: geometry, then mild colour, then normalise, then erase.

    ``RandomErasing`` comes after normalisation because it operates on tensors.
    ``RandomRotation`` is wrapped in ``RandomApply`` so only half the samples pay the
    cost, and because rotation leaves zero-filled corners that are themselves a mild
    distribution shift - applying it every time would teach the model about black corners.
    """
    steps: list[object] = [
        transforms.RandomResizedCrop(
            image_size,
            scale=cfg.scale_jitter,
            ratio=(0.9, 1.1),
            antialias=True,
        ),
    ]

    if cfg.horizontal_flip > 0:
        steps.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip))
    # A skin lesion has no canonical "up", so a vertical flip is a valid view of the
    # same finding rather than an unnatural image.
    if cfg.vertical_flip > 0:
        steps.append(transforms.RandomVerticalFlip(p=cfg.vertical_flip))
    if cfg.rotation_degrees > 0:
        steps.append(
            transforms.RandomApply(
                [transforms.RandomRotation(cfg.rotation_degrees)],
                p=0.5,
            )
        )

    if any((cfg.brightness, cfg.contrast, cfg.saturation, cfg.hue)):
        steps.append(
            transforms.ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
                saturation=cfg.saturation,
                hue=cfg.hue,
            )
        )

    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    if cfg.random_erasing > 0:
        steps.append(transforms.RandomErasing(p=cfg.random_erasing, value="random"))

    return transforms.Compose(steps)


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic pipeline for validation, test and single-image inference.

    A straight resize to a square is used instead of resize-then-centre-crop so that a
    lesion near the frame edge is never cropped away. This must stay identical to
    ``aniderm.utils.imaging.image_to_tensor``, otherwise the served model would see a
    different preprocessing than the one it was evaluated with.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
