"""Does the heatmap actually point at what the model used?

A saliency map can look convincing and still be wrong. The standard way to check is a
deletion / insertion test: rank the pixels by attributed importance and then either
remove them from the image or add them to a blank reference, measuring the predicted
probability at each step.

* **Deletion** - replace the most important pixels first with a flat neutral grey. If
  the map is honest, confidence collapses quickly, so a LOW area under the curve is good.
* **Insertion** - reveal the most important pixels first against a blurred copy of the
  image. If the map is honest, confidence recovers quickly, so a HIGH area under the
  curve is good.

The two references are deliberately different, following the original method. Deleting
into a blur would leave each region's colour behind, and a model that leans on colour -
as a dermatology model partly must - would keep predicting the same class, making every
explanation look equally useless. Inserting from flat grey, conversely, makes the first
few patches look miraculous against a featureless field.

A random ranking is scored alongside as a control. An explanation that barely beats
random is not an explanation, and reporting that honestly is more useful than a pretty
picture that nobody checked.

Reference: Petsiuk, Das and Saenko, "RISE: Randomized Input Sampling for Explanation of
Black-box Models", BMVC 2018.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from aniderm.explain.base import make_baseline
from aniderm.utils.logging import get_logger

logger = get_logger(__name__)

# NumPy renamed trapz to trapezoid in 2.0; support both so the pinned range works.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@torch.no_grad()
def deletion_insertion(
    model: nn.Module,
    image_tensor: torch.Tensor,
    saliency: np.ndarray,
    target_index: int,
    steps: int = 10,
    deletion_baseline: str = "zero",
    insertion_baseline: str = "blur",
    include_random_control: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Score one saliency map.

    ``saliency`` must already be at the same spatial resolution as ``image_tensor``.
    """
    if image_tensor.dim() != 4 or image_tensor.shape[0] != 1:
        raise ValueError(f"Expected a (1, 3, H, W) tensor, got {tuple(image_tensor.shape)}")

    height, width = image_tensor.shape[-2:]
    if saliency.shape != (height, width):
        raise ValueError(
            f"Saliency shape {saliency.shape} does not match the input's "
            f"{(height, width)}; resize it before scoring."
        )

    steps = max(2, int(steps))
    delete_reference = make_baseline(image_tensor, deletion_baseline)
    insert_reference = make_baseline(image_tensor, insertion_baseline)
    order = np.argsort(saliency.ravel())[::-1]

    fractions = np.linspace(0.0, 1.0, steps + 1)
    deletion = _curve(
        model, image_tensor, delete_reference, order, fractions, target_index, mode="delete"
    )
    insertion = _curve(
        model, image_tensor, insert_reference, order, fractions, target_index, mode="insert"
    )

    result: dict[str, Any] = {
        "fractions": [round(float(f), 4) for f in fractions],
        "deletion_curve": [round(float(v), 5) for v in deletion],
        "insertion_curve": [round(float(v), 5) for v in insertion],
        "deletion_auc": round(float(_trapezoid(deletion, fractions)), 5),
        "insertion_auc": round(float(_trapezoid(insertion, fractions)), 5),
        "deletion_baseline": deletion_baseline,
        "insertion_baseline": insertion_baseline,
        "steps": steps,
    }

    if include_random_control:
        rng = np.random.default_rng(seed)
        random_order = rng.permutation(saliency.size)
        random_del = _curve(
            model,
            image_tensor,
            delete_reference,
            random_order,
            fractions,
            target_index,
            mode="delete",
        )
        random_ins = _curve(
            model,
            image_tensor,
            insert_reference,
            random_order,
            fractions,
            target_index,
            mode="insert",
        )
        result["random_deletion_auc"] = round(float(_trapezoid(random_del, fractions)), 5)
        result["random_insertion_auc"] = round(float(_trapezoid(random_ins, fractions)), 5)

    result["net_score"] = round(result["insertion_auc"] - result["deletion_auc"], 5)
    result["verdict"] = _verdict(result)
    return result


def _curve(
    model: nn.Module,
    image: torch.Tensor,
    baseline: torch.Tensor,
    order: np.ndarray,
    fractions: np.ndarray,
    target_index: int,
    mode: str,
) -> list[float]:
    """Probability of the target class as pixels are removed from, or added to, the image."""
    total_pixels = order.size
    flat_image = image.reshape(1, 3, -1)
    flat_baseline = baseline.reshape(1, 3, -1)
    order_tensor = torch.from_numpy(np.ascontiguousarray(order)).long().to(image.device)

    probabilities: list[float] = []
    for fraction in fractions:
        count = int(round(float(fraction) * total_pixels))
        if mode == "delete":
            working = flat_image.clone()
            if count > 0:
                selected = order_tensor[:count]
                working[:, :, selected] = flat_baseline[:, :, selected]
        else:
            working = flat_baseline.clone()
            if count > 0:
                selected = order_tensor[:count]
                working[:, :, selected] = flat_image[:, :, selected]

        logits = model(working.reshape_as(image))
        probabilities.append(float(torch.softmax(logits.float(), dim=1)[0, target_index]))
    return probabilities


def _verdict(result: dict[str, Any]) -> str:
    """State plainly whether the map beat the random control."""
    net = result["net_score"]
    random_net = None
    if "random_insertion_auc" in result:
        random_net = result["random_insertion_auc"] - result["random_deletion_auc"]

    if random_net is None:
        if net > 0.25:
            return "The highlighted pixels carry most of the prediction."
        if net > 0.05:
            return "The highlighted pixels carry some of the prediction."
        return "The highlighted pixels explain little of the prediction."

    margin = net - random_net
    if margin > 0.20:
        return "Clearly better than a random ranking - the heatmap is trustworthy for this image."
    if margin > 0.05:
        return "Moderately better than a random ranking."
    return (
        "Barely better than a random ranking, so treat this particular heatmap with "
        "scepticism."
    )
