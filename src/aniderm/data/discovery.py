"""Work out which images belong to which class, whatever shape the download has.

Public image datasets are packaged inconsistently: sometimes ``<root>/<class>/x.jpg``,
sometimes ``<root>/train/<class>/x.jpg``, sometimes with an extra wrapper directory the
zip added. Rather than assume one layout, discovery walks the whole tree and resolves
each image's class from the nearest ancestor directory whose name matches a configured
class alias. That makes the pipeline survive a dataset being re-packaged upstream.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from aniderm.config import DatasetConfig, _normalise
from aniderm.utils.logging import get_logger

logger = get_logger(__name__)

# Directory names that describe a split rather than a class.
SPLIT_DIR_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing", "eval"}


@dataclass(frozen=True)
class ImageRecord:
    """One discovered image."""

    path: Path
    class_key: str
    label: int
    group: str
    source_split: str | None = None

    @property
    def relative_path(self) -> str:
        return self.path.as_posix()


@dataclass
class DiscoveryReport:
    """What discovery found, including the things it could not place."""

    records: list[ImageRecord]
    per_class_counts: dict[str, int]
    unmatched_dirs: dict[str, int]
    skipped_files: int
    group_count: int
    source_splits: dict[str, int]

    @property
    def total(self) -> int:
        return len(self.records)

    def format_summary(self, dataset: DatasetConfig) -> str:
        lines = [
            f"Dataset        : {dataset.name}",
            f"Raw directory  : {dataset.raw_dir}",
            f"Images found   : {self.total}",
            f"Distinct groups: {self.group_count}",
            "",
            "Per class:",
        ]
        width = max((len(k) for k in dataset.class_keys), default=10)
        for key in dataset.class_keys:
            count = self.per_class_counts.get(key, 0)
            share = (count / self.total * 100) if self.total else 0.0
            flag = "" if count >= dataset.min_images_per_class else "   <-- BELOW MINIMUM"
            lines.append(f"  {key:<{width}}  {count:>5}  ({share:5.1f}%){flag}")

        if self.source_splits:
            lines += ["", "Split folders present in the download (informational only):"]
            lines += [
                f"  {name:<12} {count:>5}"
                for name, count in sorted(self.source_splits.items())
            ]

        if self.unmatched_dirs:
            lines += [
                "",
                "Directories holding images that matched NO class alias:",
            ]
            for name, count in sorted(self.unmatched_dirs.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {name:<32} {count:>5} image(s)")
            lines += [
                "",
                "  Fix: add the folder name to the matching class's `aliases` list in the",
                "  dataset config, or delete the folder if it is not part of the dataset.",
            ]

        if self.skipped_files:
            lines.append(f"\nNon-image files ignored: {self.skipped_files}")
        return "\n".join(lines)


def _build_alias_index(dataset: DatasetConfig) -> dict[str, tuple[str, int]]:
    """Map every normalised alias to its ``(class_key, label_index)``."""
    index: dict[str, tuple[str, int]] = {}
    for label, spec in enumerate(dataset.classes):
        for token in spec.match_keys:
            index[token] = (spec.key, label)
    return index


def _resolve_group(stem: str, pattern: re.Pattern[str] | None) -> str:
    """Derive the grouping key for an image.

    Several photographs of one animal must not be split across train and test. When the
    filenames carry an animal identifier, a regex pulls it out; otherwise each image is
    its own group and grouping has no effect.
    """
    if pattern is not None:
        match = pattern.match(stem)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return stem


def discover_images(dataset: DatasetConfig) -> DiscoveryReport:
    """Walk ``dataset.raw_dir`` and classify every image file it contains."""
    root = dataset.raw_dir
    if not root.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {root}\n"
            f"Download it first:  python scripts/download_dataset.py"
        )

    alias_index = _build_alias_index(dataset)
    group_pattern = (
        re.compile(dataset.split.group_pattern) if dataset.split.group_pattern else None
    )
    extensions = set(dataset.image_extensions)

    records: list[ImageRecord] = []
    unmatched: Counter[str] = Counter()
    source_splits: Counter[str] = Counter()
    skipped = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in extensions:
            skipped += 1
            continue

        resolution = _resolve_class(path, root, alias_index)
        if resolution is None:
            unmatched[path.parent.name or "<root>"] += 1
            continue
        class_key, label = resolution

        split_hint = _resolve_split_hint(path, root)
        if split_hint:
            source_splits[split_hint] += 1

        records.append(
            ImageRecord(
                path=path,
                class_key=class_key,
                label=label,
                group=_resolve_group(path.stem, group_pattern),
                source_split=split_hint,
            )
        )

    per_class = Counter(record.class_key for record in records)
    report = DiscoveryReport(
        records=records,
        per_class_counts=dict(per_class),
        unmatched_dirs=dict(unmatched),
        skipped_files=skipped,
        group_count=len({record.group for record in records}),
        source_splits=dict(source_splits),
    )
    logger.info("Discovered %d images across %d classes", report.total, len(per_class))
    return report


def _resolve_class(
    path: Path, root: Path, alias_index: dict[str, tuple[str, int]]
) -> tuple[str, int] | None:
    """Find the class for one image from its ancestor directory names.

    The search runs from the immediate parent outwards and stops at the first match, so
    ``.../Healthy/augmented/x.jpg`` resolves to ``Healthy`` and a stray outer directory
    that happens to share a class alias cannot override a nearer, more specific one.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:  # pragma: no cover - rglob results are always under root
        return None

    for part in reversed(relative.parts[:-1]):
        token = _normalise(part)
        if token in alias_index:
            return alias_index[token]
    return None


def _resolve_split_hint(path: Path, root: Path) -> str | None:
    """Report whether the download already placed this image in a train/val/test folder."""
    try:
        relative = path.relative_to(root)
    except ValueError:  # pragma: no cover
        return None
    for part in relative.parts[:-1]:
        if part.lower() in SPLIT_DIR_NAMES:
            return part.lower()
    return None


def validate_report(report: DiscoveryReport, dataset: DatasetConfig) -> None:
    """Fail loudly on a download that cannot produce a trainable dataset."""
    if report.total == 0:
        raise RuntimeError(
            f"No images matched any class under {dataset.raw_dir}.\n"
            f"Expected folders named after one of: {', '.join(dataset.class_keys)}\n"
            f"Run with --inspect-only to print the directory tree that is actually there."
        )

    missing = [key for key in dataset.class_keys if report.per_class_counts.get(key, 0) == 0]
    if missing:
        raise RuntimeError(
            f"These configured classes have no images: {', '.join(missing)}.\n"
            f"Either the download is incomplete or the folder names differ from the "
            f"configured aliases. Run with --inspect-only to see the real layout."
        )

    thin = {
        key: count
        for key, count in report.per_class_counts.items()
        if count < dataset.min_images_per_class
    }
    if thin:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(thin.items()))
        raise RuntimeError(
            f"Classes below min_images_per_class={dataset.min_images_per_class}: {detail}.\n"
            f"Lower that threshold in the dataset config if the counts are expected."
        )


def inspect_tree(root: Path, extensions: set[str], max_depth: int = 4) -> str:
    """Render the directory tree with image counts, for diagnosing a layout mismatch."""
    if not root.exists():
        return f"{root} does not exist."

    counts: dict[Path, int] = defaultdict(int)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            counts[path.parent] += 1

    lines = [f"{root}"]
    directories = sorted({p for p in root.rglob("*") if p.is_dir()} | {root})
    for directory in directories:
        relative = directory.relative_to(root)
        depth = len(relative.parts)
        if depth > max_depth or depth == 0:
            continue
        count = counts.get(directory, 0)
        label = f"{'  ' * (depth - 1)}|- {directory.name}/"
        lines.append(f"{label}{'' if count == 0 else f'  [{count} images]'}")

    total = sum(counts.values())
    lines.append(f"\nTotal images with a recognised extension: {total}")
    return "\n".join(lines)
