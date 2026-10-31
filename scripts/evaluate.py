#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a manifest split.

    python scripts/evaluate.py                       # test split, best.pt
    python scripts/evaluate.py --split val
    python scripts/evaluate.py --checkpoint artifacts/checkpoints/fold3.pt
    python scripts/evaluate.py --save-errors         # dump the misclassified images

``--save-errors`` is the most useful flag here: looking at what the model gets wrong,
with its predicted and true labels side by side, tells you far more than another decimal
place on the accuracy.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aniderm.config import Config  # noqa: E402
from aniderm.data.dataset import SkinDiseaseDataset  # noqa: E402
from aniderm.data.manifest import load_manifest, resolve_manifest_paths, split_frame  # noqa: E402
from aniderm.data.transforms import build_eval_transform  # noqa: E402
from aniderm.models.classifier import load_checkpoint  # noqa: E402
from aniderm.training.metrics import (  # noqa: E402
    compute_metrics,
    format_report,
    plot_confusion_matrix,
    save_metrics,
    sklearn_report,
)
from aniderm.utils.device import select_device  # noqa: E402
from aniderm.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--checkpoint", default=None, help="defaults to serving.checkpoint")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--save-errors",
        action="store_true",
        help="copy misclassified images into the report folder",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    config = Config.load(args.config, dataset_override=args.dataset)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else config.serving.checkpoint
    device = select_device(args.device or config.training.device)
    model, meta = load_checkpoint(checkpoint_path, device)

    if meta.class_keys != config.dataset.class_keys:
        logger.warning(
            "Checkpoint class order differs from the current config.\n"
            "  checkpoint: %s\n  config:     %s\n"
            "Using the checkpoint's order, which is the one the weights were trained with.",
            meta.class_keys,
            config.dataset.class_keys,
        )

    frame = load_manifest(config.dataset)
    subset = split_frame(frame, args.split)
    if subset.empty:
        logger.error("Split '%s' is empty in %s", args.split, config.dataset.manifest_path)
        return 1

    dataset = SkinDiseaseDataset(
        frame=subset,
        transform=build_eval_transform(meta.image_size),
        class_keys=meta.class_keys,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size or config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            targets.append(labels.numpy())

    all_probs = np.concatenate(probabilities)
    all_targets = np.concatenate(targets)
    predictions = all_probs.argmax(axis=1)

    result = compute_metrics(
        y_true=all_targets,
        y_pred=predictions,
        y_prob=all_probs,
        class_keys=meta.class_keys,
        loss=float("nan"),
    )

    print("=" * 78)
    print(f"{checkpoint_path.name} on the '{args.split}' split ({len(dataset)} images)")
    print("=" * 78)
    print(format_report(result, meta.class_display_names))
    print()
    print(sklearn_report(all_targets, predictions, meta.class_keys))

    report_dir = config.project.reports_dir / "evaluation"
    save_metrics(
        result,
        report_dir / f"metrics_{args.split}.json",
        extra={"split": args.split, "checkpoint": str(checkpoint_path), "n_images": len(dataset)},
    )
    plot_confusion_matrix(
        result, report_dir / f"confusion_{args.split}.png", meta.class_display_names
    )

    if args.save_errors:
        _export_errors(
            paths=resolve_manifest_paths(subset),
            targets=all_targets,
            predictions=predictions,
            probabilities=all_probs,
            class_keys=meta.class_keys,
            destination=report_dir / f"errors_{args.split}",
        )

    print(f"\nReports written to {report_dir}")
    return 0


def _export_errors(
    paths: list[Path],
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_keys: list[str],
    destination: Path,
) -> None:
    """Copy each misclassified image into ``true__predicted/`` with its confidence."""
    wrong = np.where(targets != predictions)[0]
    if wrong.size == 0:
        logger.info("No misclassifications to export")
        return

    destination.mkdir(parents=True, exist_ok=True)
    for index in wrong:
        true_key = class_keys[targets[index]]
        pred_key = class_keys[predictions[index]]
        confidence = probabilities[index, predictions[index]]
        folder = destination / f"true-{true_key}__pred-{pred_key}"
        folder.mkdir(parents=True, exist_ok=True)
        source = paths[index]
        shutil.copy2(source, folder / f"p{confidence:.2f}_{source.name}")

    logger.info("Exported %d misclassified image(s) to %s", wrong.size, destination)


if __name__ == "__main__":
    raise SystemExit(main())
