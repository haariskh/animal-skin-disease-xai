#!/usr/bin/env python3
"""Inspect the downloaded dataset and write the train / val / test manifest.

Run this once after downloading, and again whenever the dataset config changes.

    python scripts/prepare_data.py                 # build the manifest
    python scripts/prepare_data.py --inspect-only   # just print what is on disk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aniderm.config import Config  # noqa: E402
from aniderm.data.discovery import discover_images, inspect_tree, validate_report  # noqa: E402
from aniderm.data.manifest import (  # noqa: E402
    build_manifest,
    describe_manifest,
    save_manifest,
)
from aniderm.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("prepare")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None, help="override the dataset config path")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="print the directory tree and per-class counts, then exit without writing",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing manifest"
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    config = Config.load(args.config, dataset_override=args.dataset)
    dataset = config.dataset

    print("=" * 78)
    print(f"Dataset: {dataset.name} - {dataset.description}")
    print("=" * 78)

    if args.inspect_only:
        print(inspect_tree(dataset.raw_dir, set(dataset.image_extensions)))
        print()
        try:
            report = discover_images(dataset)
        except FileNotFoundError as exc:
            print(exc)
            return 1
        print(report.format_summary(dataset))
        return 0

    if dataset.manifest_path.exists() and not args.force:
        logger.error(
            "Manifest already exists at %s. Pass --force to rebuild it.\n"
            "Rebuilding changes the split, which invalidates any model trained on the old one.",
            dataset.manifest_path,
        )
        return 1

    report = discover_images(dataset)
    print(report.format_summary(dataset))
    print()
    validate_report(report, dataset)

    frame = build_manifest(
        records=report.records,
        split=dataset.split,
        seed=config.project.seed,
    )
    save_manifest(frame, dataset.manifest_path)
    print()
    print(describe_manifest(frame, dataset))
    print()
    print(f"Manifest: {dataset.manifest_path}")
    print("Next:  python scripts/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
