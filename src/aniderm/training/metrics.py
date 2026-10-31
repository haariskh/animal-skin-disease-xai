"""Classification metrics and report rendering.

Macro-F1 leads throughout rather than accuracy: on an imbalanced four-class set a model
that ignores the smallest class can still post a respectable accuracy, and that is
exactly the failure mode this project must not ship.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)


@dataclass
class EvalResult:
    """Metrics for one pass over a split."""

    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    cohen_kappa: float
    top2_accuracy: float | None
    macro_auc: float | None
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    class_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "cohen_kappa": self.cohen_kappa,
            "top2_accuracy": self.top2_accuracy,
            "macro_auc": self.macro_auc,
            "per_class": self.per_class,
            "confusion": self.confusion,
            "class_keys": self.class_keys,
        }

    def headline(self) -> str:
        auc = f"{self.macro_auc:.4f}" if self.macro_auc is not None else "n/a"
        return (
            f"loss={self.loss:.4f}  acc={self.accuracy:.4f}  "
            f"macroF1={self.macro_f1:.4f}  balAcc={self.balanced_accuracy:.4f}  AUC={auc}"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_keys: list[str],
    loss: float,
) -> EvalResult:
    """Build the full metric set from labels, predictions and probabilities."""
    labels = list(range(len(class_keys)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        key: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, key in enumerate(class_keys)
    }

    return EvalResult(
        loss=float(loss),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        top2_accuracy=_safe_top_k(y_true, y_prob, labels, k=2),
        macro_auc=_safe_macro_auc(y_true, y_prob, labels),
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        class_keys=list(class_keys),
    )


def _safe_top_k(
    y_true: np.ndarray, y_prob: np.ndarray, labels: list[int], k: int
) -> float | None:
    """Top-k accuracy, or None when the split cannot support it."""
    if len(labels) <= k:
        return None
    try:
        return float(top_k_accuracy_score(y_true, y_prob, k=k, labels=labels))
    except ValueError:
        return None


def _safe_macro_auc(
    y_true: np.ndarray, y_prob: np.ndarray, labels: list[int]
) -> float | None:
    """One-vs-rest macro AUC.

    Returns None when a class is absent from ``y_true``: AUC is undefined for a class
    with no positive examples, and a small validation split frequently has one.
    """
    present = np.unique(y_true)
    if len(present) < 2 or len(present) != len(labels):
        return None
    try:
        return float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
        )
    except ValueError:
        return None


def format_report(result: EvalResult, display_names: list[str] | None = None) -> str:
    """Render metrics as a readable block, confusion matrix included."""
    names = display_names or result.class_keys
    width = max(len(n) for n in names) + 2

    lines = [
        "Overall",
        f"  accuracy          {result.accuracy:.4f}",
        f"  balanced accuracy {result.balanced_accuracy:.4f}",
        f"  macro F1          {result.macro_f1:.4f}",
        f"  weighted F1       {result.weighted_f1:.4f}",
        f"  Cohen's kappa     {result.cohen_kappa:.4f}",
    ]
    if result.top2_accuracy is not None:
        lines.append(f"  top-2 accuracy    {result.top2_accuracy:.4f}")
    if result.macro_auc is not None:
        lines.append(f"  macro AUC (ovr)   {result.macro_auc:.4f}")

    lines += ["", "Per class", f"  {'class':<{width}}{'prec':>8}{'rec':>8}{'f1':>8}{'n':>6}"]
    for key, name in zip(result.class_keys, names, strict=True):
        stats = result.per_class[key]
        lines.append(
            f"  {name:<{width}}{stats['precision']:>8.3f}{stats['recall']:>8.3f}"
            f"{stats['f1']:>8.3f}{stats['support']:>6d}"
        )

    lines += ["", "Confusion matrix (rows = true, columns = predicted)"]
    header = " " * width + "".join(f"{n[:6]:>8}" for n in names)
    lines.append("  " + header)
    for name, row in zip(names, result.confusion, strict=True):
        lines.append("  " + f"{name:<{width}}" + "".join(f"{v:>8d}" for v in row))
    return "\n".join(lines)


def sklearn_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_keys: list[str]
) -> str:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_keys))),
        target_names=class_keys,
        zero_division=0,
        digits=4,
    )


def save_metrics(result: EvalResult, path: Path, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if extra:
        payload.update(extra)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def plot_confusion_matrix(
    result: EvalResult,
    path: Path,
    display_names: list[str] | None = None,
    normalise: bool = True,
) -> None:
    """Save a confusion-matrix figure. Row-normalised by default, which is what you want
    to read per-class recall off the diagonal when the classes are imbalanced."""
    import matplotlib

    matplotlib.use("Agg")  # no display in a training container
    import matplotlib.pyplot as plt

    names = display_names or result.class_keys
    matrix = np.asarray(result.confusion, dtype=np.float64)
    if normalise:
        row_sums = matrix.sum(axis=1, keepdims=True)
        display = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    else:
        display = matrix

    figure, axis = plt.subplots(figsize=(1.6 * len(names) + 2.5, 1.4 * len(names) + 2.0))
    image = axis.imshow(display, cmap="Blues", vmin=0, vmax=display.max() or 1)

    axis.set_xticks(range(len(names)), [n.replace(" ", "\n") for n in names], fontsize=9)
    axis.set_yticks(range(len(names)), names, fontsize=9)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix" + (" (row-normalised)" if normalise else ""))

    threshold = (display.max() or 1) / 2
    for i in range(len(names)):
        for j in range(len(names)):
            text = f"{display[i, j]:.2f}" if normalise else f"{int(matrix[i, j])}"
            annotation = f"{text}\n(n={int(matrix[i, j])})" if normalise else text
            axis.text(
                j,
                i,
                annotation,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if display[i, j] > threshold else "black",
            )

    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
