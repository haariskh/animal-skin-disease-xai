#!/usr/bin/env python3
"""Generate a synthetic stand-in dataset, for verifying the pipeline only.

This exists so you can run prepare -> train -> evaluate -> explain -> serve end to end
before the real download finishes, and so the test suite has something deterministic to
work against. It is NOT a substitute for the real dataset and any accuracy measured on it
is meaningless - the classes here are separated by a synthetic coloured lesion, which is
a far easier problem than canine dermatology.

Each image is a skin-like background plus one lesion blob whose colour and shape encode
the class. The blob's centre is recorded in ``lesion_positions.csv``, which lets the XAI
tests check that the saliency map actually lands on the lesion rather than merely looking
plausible.

    python scripts/make_synthetic_dataset.py --output data/raw/synthetic --per-class 60
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# (folder name, background tint, lesion colour, shape)
SYNTHETIC_CLASSES = [
    ("Bacterial_dermatosis", (206, 170, 150), (188, 62, 48), "circle"),
    ("Fungal_infections", (198, 176, 152), (140, 96, 176), "ring"),
    ("Hypersensitivity_allergic_dermatosis", (214, 178, 156), (214, 170, 40), "patch"),
    ("Healthy", (202, 174, 148), None, "none"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", default="data/raw/synthetic", help="destination directory")
    parser.add_argument("--per-class", type=int, default=60, help="images per class")
    parser.add_argument("--size", type=int, default=256, help="image side length in pixels")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--groups-per-class",
        type=int,
        default=0,
        help=(
            "if > 0, name files so that several images share a synthetic animal id, "
            "which exercises the group-aware splitting path"
        ),
    )
    return parser.parse_args()


def skin_background(size: int, tint: tuple[int, int, int], rng: np.random.Generator) -> Image.Image:
    """A mottled, slightly noisy base so the model cannot win on flat colour alone."""
    base = np.zeros((size, size, 3), dtype=np.float32)
    for channel in range(3):
        base[..., channel] = tint[channel]

    # Low-frequency blotches, built by upsampling coarse noise.
    coarse = rng.normal(0.0, 1.0, size=(8, 8, 3)).astype(np.float32)
    blotches = np.asarray(
        Image.fromarray(((coarse - coarse.min()) / (np.ptp(coarse) + 1e-9) * 255).astype(np.uint8))
        .resize((size, size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )
    base += (blotches - 128.0) * 0.18
    base += rng.normal(0.0, 5.0, size=base.shape).astype(np.float32)

    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    return image.filter(ImageFilter.GaussianBlur(radius=0.6))


def draw_lesion(
    image: Image.Image,
    colour: tuple[int, int, int] | None,
    shape: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Draw the class-defining lesion and return its centre as image fractions."""
    size = image.width
    if colour is None or shape == "none":
        # The healthy class gets no lesion, so there is no position to report.
        return (-1.0, -1.0)

    radius = int(size * rng.uniform(0.09, 0.16))
    margin = radius + 6
    cx = int(rng.integers(margin, size - margin))
    cy = int(rng.integers(margin, size - margin))

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box = (cx - radius, cy - radius, cx + radius, cy + radius)

    if shape == "circle":
        draw.ellipse(box, fill=(*colour, 215))
    elif shape == "ring":
        draw.ellipse(box, outline=(*colour, 230), width=max(3, radius // 3))
    else:  # irregular patch
        points = []
        for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
            jitter = radius * rng.uniform(0.6, 1.3)
            points.append((cx + jitter * np.cos(angle), cy + jitter * np.sin(angle)))
        draw.polygon(points, fill=(*colour, 200))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.4))

    # Composite the semi-transparent lesion onto the background in place. The caller
    # holds an RGB image, so it is promoted to RGBA for the blend and the result is
    # pasted back over the original pixels.
    blended = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    image.paste(blended, (0, 0))
    return (cx / size, cy / size)


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    for class_index, (folder, tint, colour, shape) in enumerate(SYNTHETIC_CLASSES):
        destination = output / folder
        destination.mkdir(parents=True, exist_ok=True)

        for index in range(args.per_class):
            image = skin_background(args.size, tint, rng)
            x, y = draw_lesion(image, colour, shape, rng)

            if args.groups_per_class > 0:
                group = index % args.groups_per_class
                name = f"dog{class_index:02d}{group:03d}_01_02_{index:03d}.jpg"
            else:
                name = f"{folder.lower()}_{index:04d}.jpg"

            path = destination / name
            image.convert("RGB").save(path, quality=92)
            rows.append(
                {
                    "path": str(path.relative_to(output)),
                    "class": folder,
                    "lesion_x": round(x, 4),
                    "lesion_y": round(y, 4),
                }
            )

    positions = output / "lesion_positions.csv"
    with positions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "class", "lesion_x", "lesion_y"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic images to {output}")
    print(f"Lesion positions: {positions}")
    print("\nThese images are for pipeline verification only - never report metrics from them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
