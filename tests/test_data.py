"""Discovery, manifest construction and the leakage guarantees of the split."""

from __future__ import annotations

import numpy as np
import pytest

from aniderm.config import Config, DatasetConfig
from aniderm.data.discovery import discover_images, inspect_tree, validate_report
from aniderm.data.manifest import (
    build_manifest,
    class_weights,
    load_manifest,
    save_manifest,
    split_frame,
)


def test_discovery_finds_nested_class_folders(dataset_config: DatasetConfig) -> None:
    report = discover_images(dataset_config)
    # 12 per class across train/ and test/ subfolders.
    assert report.total == 36
    assert report.per_class_counts == {
        "bacterial_dermatosis": 12,
        "fungal_infections": 12,
        "healthy": 12,
    }


def test_discovery_reports_unmatched_folders_rather_than_guessing(
    dataset_config: DatasetConfig,
) -> None:
    report = discover_images(dataset_config)
    assert "mystery_folder" in report.unmatched_dirs
    assert report.unmatched_dirs["mystery_folder"] == 1
    assert report.skipped_files == 1  # the README


def test_discovery_records_existing_split_folders(dataset_config: DatasetConfig) -> None:
    report = discover_images(dataset_config)
    assert report.source_splits == {"train": 24, "test": 12}


def test_discovery_extracts_groups_from_filenames(dataset_config: DatasetConfig) -> None:
    report = discover_images(dataset_config)
    # Two animal ids per class folder, three classes, across two split folders.
    assert report.group_count < report.total
    assert all(record.group for record in report.records)


def test_missing_raw_directory_gives_an_actionable_error(dataset_config: DatasetConfig) -> None:
    broken = DatasetConfig(
        name=dataset_config.name,
        raw_dir=dataset_config.raw_dir / "nope",
        manifest_path=dataset_config.manifest_path,
        classes=dataset_config.classes,
        split=dataset_config.split,
    )
    with pytest.raises(FileNotFoundError, match="download_dataset"):
        discover_images(broken)


def test_validate_rejects_a_class_with_too_few_images(
    dataset_config: DatasetConfig, raw_dataset_dir
) -> None:
    # min_images_per_class is 4 and the class starts with 12, so the count has to drop
    # below 4 for the guard to fire: remove all 8 training images and 1 of the 4 test ones.
    for path in sorted((raw_dataset_dir / "train" / "healthy").glob("*.jpg")):
        path.unlink()
    sorted((raw_dataset_dir / "test" / "healthy").glob("*.jpg"))[0].unlink()
    report = discover_images(dataset_config)
    assert report.per_class_counts["healthy"] == 3
    with pytest.raises(RuntimeError, match="min_images_per_class"):
        validate_report(report, dataset_config)


def test_inspect_tree_counts_images(dataset_config: DatasetConfig) -> None:
    rendered = inspect_tree(dataset_config.raw_dir, set(dataset_config.image_extensions))
    assert "Total images with a recognised extension: 37" in rendered


# --- splitting ------------------------------------------------------------------ #
def _manifest(config: Config):
    report = discover_images(config.dataset)
    return build_manifest(report.records, config.dataset.split, config.project.seed)


def test_every_image_lands_in_exactly_one_split(config: Config) -> None:
    frame = _manifest(config)
    assert len(frame) == 36
    assert set(frame["split"]) == {"train", "val", "test"}
    assert frame["split"].isna().sum() == 0


def test_no_group_appears_in_two_splits(config: Config) -> None:
    """The guarantee that makes the reported accuracy meaningful."""
    frame = _manifest(config)
    groups = {
        split: set(split_frame(frame, split)["group"]) for split in ("train", "val", "test")
    }
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["val"]
    assert not groups["val"] & groups["test"]


def test_every_class_is_present_in_the_training_split(config: Config) -> None:
    frame = _manifest(config)
    train = split_frame(frame, "train")
    assert set(train["class_key"]) == set(config.dataset.class_keys)


def test_split_is_reproducible_for_a_fixed_seed(config: Config) -> None:
    first = _manifest(config)
    second = _manifest(config)
    assert first["split"].tolist() == second["split"].tolist()


def test_ungrouped_data_uses_plain_stratified_splitting(config: Config) -> None:
    from dataclasses import replace

    no_groups = replace(config.dataset.split, group_pattern=None)
    report = discover_images(config.dataset)
    frame = build_manifest(report.records, no_groups, config.project.seed)
    counts = frame[frame["split"] == "test"]["class_key"].value_counts()
    # Stratification keeps the test split balanced across classes.
    assert counts.max() - counts.min() <= 1


def test_manifest_round_trips_through_csv(config: Config) -> None:
    frame = _manifest(config)
    save_manifest(frame, config.dataset.manifest_path)
    reloaded = load_manifest(config.dataset)
    assert len(reloaded) == len(frame)
    assert reloaded["class_key"].tolist() == frame["class_key"].tolist()


def test_loading_a_manifest_with_foreign_classes_is_refused(config: Config) -> None:
    frame = _manifest(config)
    frame.loc[0, "class_key"] = "eczema"
    save_manifest(frame, config.dataset.manifest_path)
    with pytest.raises(ValueError, match="classes not in the config"):
        load_manifest(config.dataset)


def test_loading_a_manifest_with_shifted_labels_is_refused(config: Config) -> None:
    """Guards the failure that would silently map predictions to the wrong disease."""
    frame = _manifest(config)
    frame["label"] = (frame["label"] + 1) % 3
    save_manifest(frame, config.dataset.manifest_path)
    with pytest.raises(ValueError, match="label mismatch"):
        load_manifest(config.dataset)


def test_class_weights_average_to_one(config: Config) -> None:
    frame = _manifest(config)
    weights = class_weights(frame, config.dataset.num_classes)
    train = split_frame(frame, "train")
    counts = np.bincount(train["label"].to_numpy(), minlength=3)
    assert np.isclose((weights * counts).sum() / len(train), 1.0)
    assert weights.min() > 0
