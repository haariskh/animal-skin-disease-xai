#!/usr/bin/env python3
"""Train the classifier.

    python scripts/train.py
    python scripts/train.py --epochs-finetune 60 --backbone efficientnet_b3 --image-size 300
    python scripts/train.py --crossval            # k-fold instead of a single split

The best checkpoint by validation macro-F1 is written to
``artifacts/checkpoints/best.pt`` and carries its own class list, so the API never has
to guess which output index means which disease.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aniderm.config import Config  # noqa: E402
from aniderm.data.dataset import build_dataloaders  # noqa: E402
from aniderm.data.manifest import class_weights, describe_manifest, load_manifest  # noqa: E402
from aniderm.models.classifier import build_model  # noqa: E402
from aniderm.training.crossval import format_summary, run_cross_validation  # noqa: E402
from aniderm.training.engine import Trainer  # noqa: E402
from aniderm.training.metrics import (  # noqa: E402
    format_report,
    plot_confusion_matrix,
    save_metrics,
)
from aniderm.utils.device import select_device  # noqa: E402
from aniderm.utils.logging import get_logger, setup_logging  # noqa: E402
from aniderm.utils.seed import set_seed  # noqa: E402

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--backbone", default=None, help="override model.backbone")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs-head", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto | cuda | mps | cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="train from scratch")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--crossval", action="store_true", help="run k-fold cross-validation")
    parser.add_argument("--run-name", default=None, help="subfolder name under artifacts/reports")
    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Let CLI flags win over the YAML, without mutating the file on disk."""
    model = config.model
    if args.backbone:
        model = replace(model, backbone=args.backbone)
    if args.image_size:
        model = replace(model, image_size=args.image_size)
    if args.no_pretrained:
        model = replace(model, pretrained=False)

    training = config.training
    if args.batch_size:
        training = replace(training, batch_size=args.batch_size)
    if args.epochs_head is not None:
        training = replace(training, epochs_head=args.epochs_head)
    if args.epochs_finetune is not None:
        training = replace(training, epochs_finetune=args.epochs_finetune)
    if args.num_workers is not None:
        training = replace(training, num_workers=args.num_workers)
    if args.device:
        training = replace(training, device=args.device)
    if args.no_amp:
        training = replace(training, amp=False)

    crossval = config.crossval
    if args.crossval:
        crossval = replace(crossval, enabled=True)

    return replace(config, model=model, training=training, crossval=crossval)


def main() -> int:
    args = parse_args()
    config = apply_overrides(Config.load(args.config, dataset_override=args.dataset), args)

    run_name = args.run_name or f"{config.model.backbone}_{config.dataset.name}"
    report_dir = config.project.reports_dir / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=report_dir / "train.log")

    logger.info("Run '%s'", run_name)
    set_seed(config.project.seed, config.project.deterministic)
    device = select_device(config.training.device)

    frame = load_manifest(config.dataset)
    print(describe_manifest(frame, config.dataset))
    print()

    if config.crossval.enabled:
        summary = run_cross_validation(
            config=config,
            frame=frame,
            device=device,
            output_dir=report_dir / "crossval",
        )
        print()
        print(format_summary(summary))
        return 0

    loaders = build_dataloaders(config, manifest=frame)
    if "test" not in loaders:
        logger.warning("No test split found; final metrics will come from the validation split")

    model = build_model(config)
    trainer = Trainer(
        config=config,
        model=model,
        loaders=loaders,
        device=device,
        class_weights=(
            class_weights(frame, config.dataset.num_classes)
            if config.training.class_weighting == "balanced"
            else None
        ),
        checkpoint_path=config.project.checkpoints_dir / "best.pt",
        report_dir=report_dir,
    )
    outcome = trainer.fit()

    # Reload the best checkpoint before touching the test split: the weights in memory
    # belong to the final epoch, which is usually not the best one.
    from aniderm.models.classifier import load_checkpoint

    best_model, _ = load_checkpoint(trainer.checkpoint_path, device)
    trainer.model = best_model

    evaluation_split = "test" if "test" in loaders else "val"
    result = trainer.evaluate(loaders[evaluation_split], evaluation_split)

    print()
    print("=" * 78)
    print(f"Held-out performance on the '{evaluation_split}' split")
    print("=" * 78)
    print(format_report(result, config.dataset.display_names))
    print()

    save_metrics(
        result,
        report_dir / f"metrics_{evaluation_split}.json",
        extra={
            "split": evaluation_split,
            "backbone": config.model.backbone,
            "image_size": config.model.image_size,
            "best_epoch": outcome.best_epoch,
            "monitor": outcome.monitor,
            "best_val_metric": outcome.best_metric,
            "stopped_early": outcome.stopped_early,
        },
    )
    plot_confusion_matrix(
        result,
        report_dir / f"confusion_{evaluation_split}.png",
        config.dataset.display_names,
    )

    print(f"Checkpoint : {trainer.checkpoint_path}")
    print(f"Reports    : {report_dir}")
    print("Next:  python scripts/evaluate.py   and then   uvicorn api.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
