"""Turn a saliency map into something a person can read.

A heatmap tells a user where to look but not what it says. This module reduces a map to
plain facts - how concentrated the evidence is, which part of the frame holds it, how
many separate areas there are - so the UI can state in words what the picture shows.
That matters for a diagnostic aid: a clinician needs to know whether the model fixated
on one lesion or spread its attention over the whole animal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

REGION_NAMES: tuple[tuple[str, ...], ...] = (
    ("top left", "top centre", "top right"),
    ("middle left", "centre", "middle right"),
    ("bottom left", "bottom centre", "bottom right"),
)

HOT_THRESHOLD = 0.5


def grid_contributions(saliency: np.ndarray, rows: int = 3, cols: int = 3) -> list[dict[str, Any]]:
    """Split the map into a grid and report each cell's share of the total evidence."""
    total = float(saliency.sum())
    height, width = saliency.shape
    row_edges = np.linspace(0, height, rows + 1, dtype=int)
    col_edges = np.linspace(0, width, cols + 1, dtype=int)

    cells: list[dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            block = saliency[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]]
            share = float(block.sum()) / total if total > 0 else 0.0
            name = REGION_NAMES[r][c] if rows == 3 and cols == 3 else f"r{r}c{c}"
            cells.append(
                {
                    "name": name,
                    "row": r,
                    "col": c,
                    "share": round(share, 4),
                    "mean_intensity": round(float(block.mean()), 4),
                }
            )
    return sorted(cells, key=lambda cell: cell["share"], reverse=True)


def hotspots(
    saliency: np.ndarray,
    threshold: float = HOT_THRESHOLD,
    max_regions: int = 3,
) -> list[dict[str, Any]]:
    """Find the distinct connected areas above ``threshold``.

    Bounding boxes come back as fractions of width and height, so the frontend can draw
    them over an image of any display size without knowing the model's input resolution.
    """
    from scipy import ndimage

    mask = saliency >= threshold
    if not mask.any():
        return []

    labelled, count = ndimage.label(mask)
    height, width = saliency.shape
    total = float(saliency.sum()) or 1.0

    found: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        component = labelled == index
        area_px = int(component.sum())
        # Single-pixel specks are interpolation noise, not findings.
        if area_px < max(4, (height * width) // 2000):
            continue

        rows, columns = np.where(component)
        values = saliency[component]
        found.append(
            {
                "area_fraction": round(area_px / (height * width), 4),
                "evidence_share": round(float(values.sum()) / total, 4),
                "peak_intensity": round(float(values.max()), 4),
                "mean_intensity": round(float(values.mean()), 4),
                "bbox": {
                    "x": round(float(columns.min()) / width, 4),
                    "y": round(float(rows.min()) / height, 4),
                    "width": round(float(columns.max() - columns.min() + 1) / width, 4),
                    "height": round(float(rows.max() - rows.min() + 1) / height, 4),
                },
                "centroid": {
                    "x": round(float(columns.mean()) / width, 4),
                    "y": round(float(rows.mean()) / height, 4),
                },
                "location": _name_for_point(
                    float(columns.mean()) / width, float(rows.mean()) / height
                ),
            }
        )

    found.sort(key=lambda region: region["evidence_share"], reverse=True)
    return found[:max_regions]


def _name_for_point(x: float, y: float) -> str:
    col = min(2, max(0, int(x * 3)))
    row = min(2, max(0, int(y * 3)))
    return REGION_NAMES[row][col]


def concentration(saliency: np.ndarray) -> dict[str, float]:
    """Measure how focused the map is.

    ``top_10pct_share`` is the useful number: the fraction of all evidence carried by the
    most salient tenth of the pixels. Near 1.0 means the model pinned a specific area;
    near 0.1 means the attribution is spread evenly and the explanation says little.

    ``gini`` is included as a scale-free companion measure of inequality.
    """
    flat = np.sort(saliency.ravel())[::-1]
    total = float(flat.sum())
    if total <= 0:
        return {"top_10pct_share": 0.0, "coverage_above_half": 0.0, "gini": 0.0}

    cutoff = max(1, int(0.1 * flat.size))
    top_share = float(flat[:cutoff].sum()) / total

    ascending = flat[::-1].astype(np.float64)
    n = ascending.size
    ranks = np.arange(1, n + 1)
    gini = float((2.0 * np.sum(ranks * ascending)) / (n * ascending.sum()) - (n + 1) / n)

    return {
        "top_10pct_share": round(top_share, 4),
        "coverage_above_half": round(float((saliency >= HOT_THRESHOLD).mean()), 4),
        "gini": round(gini, 4),
    }


def analyse(saliency: np.ndarray, threshold: float = HOT_THRESHOLD) -> dict[str, Any]:
    """Run the full region analysis and attach a one-sentence reading of it."""
    grid = grid_contributions(saliency)
    spots = hotspots(saliency, threshold=threshold)
    stats = concentration(saliency)

    analysis = {
        "grid": grid,
        "hotspots": spots,
        "concentration": stats,
        "threshold": threshold,
    }
    analysis["summary"] = _summarise(grid, spots, stats)
    return analysis


def _percent(value: float) -> str:
    """Format a percentage without collapsing small-but-real values to '0%'.

    Pixel-level methods such as Integrated Gradients produce genuinely tiny regions, and
    rounding those to zero reads as a bug rather than as a finding.
    """
    if value <= 0:
        return "0%"
    if value < 0.1:
        return "under 0.1%"
    if value < 10:
        return f"{value:.1f}%"
    return f"{value:.0f}%"


def _summarise(
    grid: list[dict[str, Any]],
    spots: list[dict[str, Any]],
    stats: dict[str, float],
) -> str:
    if not grid or stats["top_10pct_share"] == 0.0:
        return "The explanation is essentially flat, so no particular region drove this prediction."

    leading = grid[0]
    focus = stats["top_10pct_share"]

    if spots:
        count = len(spots)
        area = sum(spot["area_fraction"] for spot in spots) * 100
        share = sum(spot["evidence_share"] for spot in spots) * 100
        where = spots[0]["location"]
        region_word = "region" if count == 1 else "regions"
        sentence = (
            f"{_percent(share)} of the evidence comes from {count} {region_word} covering "
            f"{_percent(area)} of the image, centred on the {where}."
        )
    else:
        sentence = (
            f"Attention leans towards the {leading['name']} "
            f"({_percent(leading['share'] * 100)} of the total) without forming a distinct "
            f"hotspot."
        )

    if focus >= 0.6:
        sentence += " The attribution is tightly focused."
    elif focus <= 0.25:
        sentence += (
            " The attribution is diffuse, which is a sign the prediction rests on overall "
            "appearance rather than one specific lesion - read it with caution."
        )
    return sentence
