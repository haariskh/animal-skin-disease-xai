"""Build and read the manifest CSV that fixes the train / val / test split.

Writing the split to disk once, rather than re-sampling it inside every script, is what
makes training, evaluation and the XAI report comparable to each other. The manifest is
the single source of truth about which image belongs to which split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from aniderm.config import REPO_ROOT, DatasetConfig, SplitConfig
from aniderm.data.discovery import ImageRecord
from aniderm.utils.logging import get_logger

logger = get_logger(__name__)

MANIFEST_COLUMNS = ["path", "class_key", "label", "group", "split", "source_split"]
SPLITS = ("train", "val", "test")


def _relative_to_repo(path: Path) -> str:
    """Store paths relative to the repo when possible, so a manifest stays portable."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_manifest(
    records: list[ImageRecord],
    split: SplitConfig,
    seed: int,
) -> pd.DataFrame:
    """Assign every record to a split and return the manifest as a DataFrame."""
    if not records:
        raise ValueError("Cannot build a manifest from zero records")

    labels = np.asarray([record.label for record in records])
    groups = np.asarray([record.group for record in records])

    assignment = _assign_splits(
        labels=labels,
        groups=groups,
        val_size=split.val_size,
        test_size=split.test_size,
        stratify=split.stratify,
        seed=seed,
    )

    frame = pd.DataFrame(
        {
            "path": [_relative_to_repo(record.path) for record in records],
            "class_key": [record.class_key for record in records],
            "label": labels,
            "group": groups,
            "split": assignment,
            "source_split": [record.source_split or "" for record in records],
        }
    )
    return (
        frame[MANIFEST_COLUMNS]
        .sort_values(["split", "class_key", "path"])
        .reset_index(drop=True)
    )


def _assign_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    val_size: float,
    test_size: float,
    stratify: bool,
    seed: int,
) -> np.ndarray:
    """Return an array of ``"train"`` / ``"val"`` / ``"test"`` labels.

    Two regimes:

    * every image is its own group - a plain two-stage stratified split;
    * images share groups (several photos of one animal) - ``StratifiedGroupKFold``, so no
      group straddles two splits. Without this, near-duplicate photographs of the same
      animal appear in train and test and the reported accuracy is meaningless.
    """
    total = len(labels)
    indices = np.arange(total)
    grouped = len(np.unique(groups)) < total

    if grouped:
        logger.info(
            "Group-aware splitting: %d images across %d groups",
            total,
            len(np.unique(groups)),
        )
        return _grouped_assignment(indices, labels, groups, val_size, test_size, seed)

    logger.info("Every image forms its own group; using plain stratified splitting")
    return _ungrouped_assignment(indices, labels, val_size, test_size, stratify, seed)


def _ungrouped_assignment(
    indices: np.ndarray,
    labels: np.ndarray,
    val_size: float,
    test_size: float,
    stratify: bool,
    seed: int,
) -> np.ndarray:
    counts = np.bincount(labels)
    if stratify and counts[counts > 0].min() < 3:
        thin = np.where((counts > 0) & (counts < 3))[0].tolist()
        raise ValueError(
            f"Stratified splitting needs at least 3 images per class; label(s) {thin} "
            f"have fewer. Either add data or set split.stratify to false."
        )

    strata = labels if stratify else None
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=strata, shuffle=True
    )

    # val_size is expressed as a fraction of the WHOLE dataset, so it has to be rescaled
    # against the portion that is left after the test split is removed.
    relative_val = val_size / (1.0 - test_size)
    strata_tv = labels[train_val_idx] if stratify else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        random_state=seed,
        stratify=strata_tv,
        shuffle=True,
    )

    assignment = np.empty(len(indices), dtype=object)
    assignment[train_idx] = "train"
    assignment[val_idx] = "val"
    assignment[test_idx] = "test"
    return assignment


def _grouped_assignment(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
) -> np.ndarray:
    def folds_for(fraction: float) -> int:
        # A k-fold with k folds holds out 1/k of the data, so k = round(1 / fraction).
        return int(max(2, min(len(np.unique(groups)), round(1.0 / fraction))))

    test_splitter = StratifiedGroupKFold(
        n_splits=folds_for(test_size), shuffle=True, random_state=seed
    )
    rest_pos, test_pos = next(test_splitter.split(indices, labels, groups))

    rest_idx = indices[rest_pos]
    relative_val = val_size / (1.0 - test_size)
    val_splitter = StratifiedGroupKFold(
        n_splits=folds_for(relative_val), shuffle=True, random_state=seed
    )
    train_rel, val_rel = next(
        val_splitter.split(rest_idx, labels[rest_idx], groups[rest_idx])
    )

    assignment = np.empty(len(indices), dtype=object)
    assignment[rest_idx[train_rel]] = "train"
    assignment[rest_idx[val_rel]] = "val"
    assignment[indices[test_pos]] = "test"
    return assignment


def save_manifest(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Manifest written to %s (%d rows)", path, len(frame))


def load_manifest(dataset: DatasetConfig) -> pd.DataFrame:
    """Read the manifest and verify it still matches the dataset config."""
    path = dataset.manifest_path
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\nBuild it first:  python scripts/prepare_data.py"
        )

    frame = pd.read_csv(path, dtype={"group": str, "source_split": str}).fillna(
        {"source_split": ""}
    )
    missing_columns = [c for c in MANIFEST_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise ValueError(f"Manifest {path} is missing column(s): {missing_columns}")

    # A manifest written against a different class list would silently train the model on
    # mismatched labels, so the label space is checked rather than trusted.
    expected = set(dataset.class_keys)
    found = set(frame["class_key"].unique())
    if not found.issubset(expected):
        raise ValueError(
            f"Manifest contains classes not in the config: {sorted(found - expected)}.\n"
            f"Re-run scripts/prepare_data.py after changing the dataset config."
        )
    for class_key, group in frame.groupby("class_key"):
        expected_label = dataset.index_of(str(class_key))
        actual = set(group["label"].unique())
        if actual != {expected_label}:
            raise ValueError(
                f"Manifest label mismatch for '{class_key}': expected {expected_label}, "
                f"found {sorted(actual)}. Re-run scripts/prepare_data.py."
            )
    return frame


def resolve_manifest_paths(frame: pd.DataFrame) -> list[Path]:
    """Turn the manifest's stored paths back into absolute ones."""
    return [
        Path(raw) if Path(raw).is_absolute() else (REPO_ROOT / raw)
        for raw in frame["path"].tolist()
    ]


def split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    return frame[frame["split"] == split].reset_index(drop=True)


def describe_manifest(frame: pd.DataFrame, dataset: DatasetConfig) -> str:
    """Render the split x class table, which is the first thing worth eyeballing."""
    table = (
        frame.pivot_table(index="class_key", columns="split", aggfunc="size", fill_value=0)
        .reindex(index=dataset.class_keys, fill_value=0)
        .reindex(columns=list(SPLITS), fill_value=0)
    )
    table["total"] = table.sum(axis=1)

    lines = ["Split distribution:", table.to_string(), ""]
    for split in SPLITS:
        subset = split_frame(frame, split)
        leak = ""
        if split != "train":
            shared = set(subset["group"]) & set(split_frame(frame, "train")["group"])
            leak = (
                f"  (groups shared with train: {len(shared)})"
                if shared
                else "  (no group overlap with train)"
            )
        lines.append(f"  {split:<5} {len(subset):>5} images{leak}")
    return "\n".join(lines)


def class_weights(frame: pd.DataFrame, num_classes: int) -> np.ndarray:
    """Inverse-frequency weights for the training split.

    Uses the scikit-learn ``balanced`` definition, ``n / (k * count_c)``, which averages
    to 1.0 and therefore does not change the overall loss scale.
    """
    train = split_frame(frame, "train")
    counts = np.bincount(train["label"].to_numpy(), minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        empty = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Training split has no examples for label(s) {empty}")
    return len(train) / (num_classes * counts)
