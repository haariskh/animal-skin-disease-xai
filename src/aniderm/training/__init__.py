"""Training loop, losses, metrics and cross-validation."""

from aniderm.training.engine import Trainer, TrainingOutcome
from aniderm.training.metrics import (
    EvalResult,
    compute_metrics,
    format_report,
    plot_confusion_matrix,
    save_metrics,
)

__all__ = [
    "EvalResult",
    "Trainer",
    "TrainingOutcome",
    "compute_metrics",
    "format_report",
    "plot_confusion_matrix",
    "save_metrics",
]
