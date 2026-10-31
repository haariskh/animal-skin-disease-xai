"""Stratified k-fold cross-validation.

On a dataset of roughly 443 images a single 70/15/15 split puts about 66 images in the
test set. A couple of lucky or unlucky examples then move accuracy by several points, so
one number from one split is not a result worth reporting. Cross-validation retrains over
every fold and reports mean +/- standard deviation, which is the honest version.

Group awareness is preserved: when the manifest has groups - several photographs of the
same animal - folds are built with ``StratifiedGroupKFold`` so no animal appears in both
the training and the held-out portion of a fold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from aniderm.config import Config
from aniderm.data.dataset import build_dataloaders
from aniderm.data.manifest import class_weights
from aniderm.models.classifier import build_model
from aniderm.training.engine import Trainer
from aniderm.training.metrics import EvalResult
from aniderm.utils.logging import get_logger
from aniderm.utils.seed import set_seed

logger = get_logger(__name__)

AGGREGATED_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "cohen_kappa",
)


@dataclass
class FoldResult:
    fold: int
    best_metric: float
    best_epoch: int
    test: EvalResult


def make_folds(
    frame: pd.DataFrame, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build fold indices over the whole manifest, ignoring its stored split column."""
    labels = frame["label"].to_numpy()
    groups = frame["group"].to_numpy()

    if len(np.unique(groups)) < len(frame):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(frame.index.to_numpy(), labels, groups))

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(frame.index.to_numpy(), labels))


def run_cross_validation(
    config: Config,
    frame: pd.DataFrame,
    device,
    output_dir: Path,
) -> dict[str, Any]:
    """Train one model per fold and aggregate the held-out metrics."""
    n_splits = config.crossval.n_splits
    folds = make_folds(frame, n_splits=n_splits, seed=config.project.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[FoldResult] = []
    for fold_index, (train_positions, holdout_positions) in enumerate(folds, start=1):
        logger.info("=" * 70)
        logger.info("Fold %d of %d", fold_index, n_splits)
        logger.info("=" * 70)

        # Reseed per fold so each run starts from the same initialisation and the
        # differences between folds come from the data, not from the random seed.
        set_seed(config.project.seed, config.project.deterministic)

        fold_frame = frame.copy()
        fold_frame["split"] = "train"
        fold_frame.loc[fold_frame.index[holdout_positions], "split"] = "test"

        # A slice of the training portion becomes this fold's validation set, used only
        # for checkpoint selection and early stopping - never for reporting.
        train_index = fold_frame.index[train_positions]
        rng = np.random.default_rng(config.project.seed + fold_index)
        val_count = max(1, int(0.15 * len(train_index)))
        val_index = rng.choice(train_index, size=val_count, replace=False)
        fold_frame.loc[val_index, "split"] = "val"

        loaders = build_dataloaders(config, splits=("train", "val", "test"), manifest=fold_frame)
        model = build_model(config)
        trainer = Trainer(
            config=config,
            model=model,
            loaders=loaders,
            device=device,
            class_weights=(
                class_weights(fold_frame, config.dataset.num_classes)
                if config.training.class_weighting == "balanced"
                else None
            ),
            checkpoint_path=output_dir / f"fold{fold_index}.pt",
            report_dir=output_dir / f"fold{fold_index}",
        )
        outcome = trainer.fit()
        test_result = trainer.evaluate(loaders["test"], f"fold{fold_index}-test")

        logger.info("Fold %d held-out: %s", fold_index, test_result.headline())
        results.append(
            FoldResult(
                fold=fold_index,
                best_metric=outcome.best_metric,
                best_epoch=outcome.best_epoch,
                test=test_result,
            )
        )

    summary = summarise(results, config)
    path = output_dir / "crossval_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Cross-validation summary written to %s", path)
    return summary


def summarise(results: list[FoldResult], config: Config) -> dict[str, Any]:
    """Aggregate fold metrics into mean, standard deviation and range."""
    aggregate: dict[str, Any] = {
        "n_folds": len(results),
        "backbone": config.model.backbone,
        "dataset": config.dataset.name,
        "folds": [
            {
                "fold": result.fold,
                "best_epoch": result.best_epoch,
                **{name: getattr(result.test, name) for name in AGGREGATED_METRICS},
            }
            for result in results
        ],
    }

    for name in AGGREGATED_METRICS:
        values = np.array([getattr(result.test, name) for result in results], dtype=float)
        aggregate[name] = {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
        }

    # Per-class recall across folds exposes a class the model fails on in every fold,
    # which an averaged headline number hides completely.
    per_class: dict[str, dict[str, float]] = {}
    for class_key in config.dataset.class_keys:
        recalls = np.array(
            [result.test.per_class[class_key]["recall"] for result in results], dtype=float
        )
        f1s = np.array(
            [result.test.per_class[class_key]["f1"] for result in results], dtype=float
        )
        per_class[class_key] = {
            "recall_mean": round(float(recalls.mean()), 4),
            "recall_std": round(float(recalls.std(ddof=1)) if len(recalls) > 1 else 0.0, 4),
            "f1_mean": round(float(f1s.mean()), 4),
            "f1_std": round(float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0, 4),
        }
    aggregate["per_class"] = per_class
    return aggregate


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Cross-validation over {summary['n_folds']} folds "
        f"({summary['backbone']} on {summary['dataset']})",
        "",
    ]
    for name in AGGREGATED_METRICS:
        stats = summary[name]
        lines.append(
            f"  {name:<20} {stats['mean']:.4f} +/- {stats['std']:.4f}   "
            f"[{stats['min']:.4f}, {stats['max']:.4f}]"
        )

    lines += ["", "  Per-class recall (mean +/- std)"]
    for class_key, stats in summary["per_class"].items():
        lines.append(
            f"    {class_key:<40} {stats['recall_mean']:.4f} +/- {stats['recall_std']:.4f}"
        )
    return "\n".join(lines)
