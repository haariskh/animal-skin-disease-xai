"""The training loop.

Training runs in two stages, which is the standard recipe for a small dataset:

1. **Head only.** The backbone is frozen and just the new classifier is trained. A
   randomly initialised head produces large gradients at first; letting those flow into
   pretrained weights damages the features you are trying to reuse.
2. **Fine-tune.** Everything unfreezes, the backbone gets a learning rate an order of
   magnitude lower than the head, and a cosine schedule with warmup takes it down.

Checkpointing and early stopping both follow macro-F1 by default, not accuracy.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aniderm.config import Config
from aniderm.models.classifier import CheckpointMeta, SkinDiseaseClassifier, save_checkpoint
from aniderm.training.losses import build_criterion
from aniderm.training.metrics import EvalResult, compute_metrics
from aniderm.utils.device import amp_enabled
from aniderm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EpochRecord:
    epoch: int
    stage: str
    learning_rate: float
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    seconds: float


@dataclass
class TrainingOutcome:
    best_metric: float
    best_epoch: int
    monitor: str
    history: list[EpochRecord] = field(default_factory=list)
    checkpoint_path: Path | None = None
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "history": [asdict(record) for record in self.history],
        }


class Trainer:
    def __init__(
        self,
        config: Config,
        model: SkinDiseaseClassifier,
        loaders: dict[str, DataLoader],
        device: torch.device,
        class_weights: np.ndarray | None,
        checkpoint_path: Path,
        report_dir: Path,
    ) -> None:
        if "train" not in loaders or "val" not in loaders:
            raise ValueError("Trainer needs both a 'train' and a 'val' dataloader")

        self.config = config
        self.training = config.training
        self.model = model.to(device)
        self.loaders = loaders
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.report_dir = report_dir

        self.criterion = build_criterion(
            class_weighting=self.training.class_weighting,
            weights=class_weights,
            label_smoothing=self.training.label_smoothing,
            device=device,
        )
        self.use_amp = amp_enabled(self.training.amp, device)
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)

        self.history: list[EpochRecord] = []
        self.best_metric = -np.inf if self.training.greater_is_better else np.inf
        self.best_epoch = -1
        self._epochs_without_improvement = 0

    # -- public ------------------------------------------------------------- #
    def fit(self) -> TrainingOutcome:
        logger.info(
            "Training %s for %d + %d epochs on %s (AMP=%s)",
            self.config.model.backbone,
            self.training.epochs_head,
            self.training.epochs_finetune,
            self.device.type,
            self.use_amp,
        )
        stopped_early = False
        epoch_counter = 0

        if self.training.epochs_head > 0:
            self.model.freeze_backbone()
            optimiser = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.training.lr_head,
                weight_decay=self.training.weight_decay,
            )
            epoch_counter, stopped_early = self._run_stage(
                stage="head",
                epochs=self.training.epochs_head,
                optimiser=optimiser,
                scheduler=None,
                start_epoch=epoch_counter,
                allow_early_stop=False,
            )

        if self.training.epochs_finetune > 0 and not stopped_early:
            self.model.unfreeze_all()
            optimiser = torch.optim.AdamW(
                [
                    {"params": self.model.backbone_parameters(), "lr": self.training.lr_backbone},
                    {"params": self.model.classifier_parameters(), "lr": self.training.lr_head},
                ],
                weight_decay=self.training.weight_decay,
            )
            scheduler = self._build_scheduler(optimiser, self.training.epochs_finetune)
            epoch_counter, stopped_early = self._run_stage(
                stage="finetune",
                epochs=self.training.epochs_finetune,
                optimiser=optimiser,
                scheduler=scheduler,
                start_epoch=epoch_counter,
                allow_early_stop=True,
            )

        outcome = TrainingOutcome(
            best_metric=float(self.best_metric),
            best_epoch=self.best_epoch,
            monitor=self.training.monitor,
            history=self.history,
            checkpoint_path=self.checkpoint_path,
            stopped_early=stopped_early,
        )
        self._save_history(outcome)
        logger.info(
            "Training finished. Best %s = %.4f at epoch %d",
            self.training.monitor,
            outcome.best_metric,
            outcome.best_epoch,
        )
        return outcome

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str = "val") -> EvalResult:
        """Run one pass over a loader and return metrics."""
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        all_probs: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        for images, targets in tqdm(loader, desc=f"eval[{split_name}]", leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            # Softmax in fp32: under autocast the logits are fp16 and the exponential can
            # overflow, which would turn a probability into NaN.
            probs = torch.softmax(logits.float(), dim=1)
            total_loss += loss.item() * images.size(0)
            total_items += images.size(0)
            all_probs.append(probs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

        probabilities = np.concatenate(all_probs)
        targets_array = np.concatenate(all_targets)
        predictions = probabilities.argmax(axis=1)

        return compute_metrics(
            y_true=targets_array,
            y_pred=predictions,
            y_prob=probabilities,
            class_keys=self.config.dataset.class_keys,
            loss=total_loss / max(1, total_items),
        )

    # -- internals ---------------------------------------------------------- #
    def _run_stage(
        self,
        stage: str,
        epochs: int,
        optimiser: torch.optim.Optimizer,
        scheduler: Any | None,
        start_epoch: int,
        allow_early_stop: bool,
    ) -> tuple[int, bool]:
        epoch = start_epoch
        for local_epoch in range(epochs):
            epoch += 1
            started = time.perf_counter()

            train_loss, train_accuracy = self._train_epoch(optimiser, epoch, stage)
            val_result = self.evaluate(self.loaders["val"], "val")

            current_lr = optimiser.param_groups[0]["lr"]
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self._monitored_value(val_result))
                else:
                    scheduler.step()

            elapsed = time.perf_counter() - started
            self.history.append(
                EpochRecord(
                    epoch=epoch,
                    stage=stage,
                    learning_rate=current_lr,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    val_loss=val_result.loss,
                    val_accuracy=val_result.accuracy,
                    val_macro_f1=val_result.macro_f1,
                    seconds=elapsed,
                )
            )
            logger.info(
                "[%s %d/%d] train loss=%.4f acc=%.4f | val %s | lr=%.2e | %.1fs",
                stage,
                local_epoch + 1,
                epochs,
                train_loss,
                train_accuracy,
                val_result.headline(),
                current_lr,
                elapsed,
            )

            if self._update_best(val_result, epoch):
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1
                if (
                    allow_early_stop
                    and self._epochs_without_improvement >= self.training.early_stopping_patience
                ):
                    logger.info(
                        "Early stopping: no improvement in %s for %d epochs",
                        self.training.monitor,
                        self._epochs_without_improvement,
                    )
                    return epoch, True
        return epoch, False

    def _train_epoch(
        self, optimiser: torch.optim.Optimizer, epoch: int, stage: str
    ) -> tuple[float, float]:
        self.model.train()
        loader = self.loaders["train"]
        running_loss = 0.0
        correct = 0
        seen = 0

        progress = tqdm(loader, desc=f"{stage} epoch {epoch}", leave=False)
        for images, targets in progress:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            if self.training.grad_clip > 0:
                # Gradients must be unscaled before clipping, otherwise the threshold
                # would be applied to loss-scaled values and effectively ignored.
                self.scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.training.grad_clip)
            self.scaler.step(optimiser)
            self.scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.detach().argmax(dim=1) == targets).sum().item()
            seen += batch_size
            progress.set_postfix(loss=f"{running_loss / seen:.4f}", acc=f"{correct / seen:.3f}")

        return running_loss / max(1, seen), correct / max(1, seen)

    def _build_scheduler(self, optimiser: torch.optim.Optimizer, epochs: int):
        kind = self.training.scheduler
        if kind == "none":
            return None
        if kind == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                mode="max" if self.training.greater_is_better else "min",
                factor=0.5,
                patience=max(2, self.training.early_stopping_patience // 3),
            )

        warmup = min(self.training.warmup_epochs, max(0, epochs - 1))

        def lr_factor(epoch_index: int) -> float:
            # Linear warmup avoids a large first step into freshly unfrozen weights,
            # then cosine decay anneals smoothly to ~0 by the final epoch.
            if warmup > 0 and epoch_index < warmup:
                return (epoch_index + 1) / warmup
            progress = (epoch_index - warmup) / max(1, epochs - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_factor)

    def _monitored_value(self, result: EvalResult) -> float:
        return {
            "val_macro_f1": result.macro_f1,
            "val_accuracy": result.accuracy,
            "val_loss": result.loss,
        }[self.training.monitor]

    def _update_best(self, result: EvalResult, epoch: int) -> bool:
        value = self._monitored_value(result)
        improved = (
            value > self.best_metric
            if self.training.greater_is_better
            else value < self.best_metric
        )
        if not improved:
            return False

        self.best_metric = value
        self.best_epoch = epoch
        meta = CheckpointMeta.from_config(
            self.config,
            metrics={
                "monitor": self.training.monitor,
                "best_metric": float(value),
                "best_epoch": epoch,
                "val": result.to_dict(),
            },
        )
        # Resolve the Grad-CAM layer before saving so the name travels with the weights.
        self.model.target_layer(self.config.xai.gradcam_target_layer)
        save_checkpoint(self.checkpoint_path, self.model, meta)
        logger.info("New best %s=%.4f at epoch %d", self.training.monitor, value, epoch)
        return True

    def _save_history(self, outcome: TrainingOutcome) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_dir / "training_history.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(outcome.to_dict(), handle, indent=2)
        logger.info("History written to %s", path)
        try:
            plot_training_curves(outcome, self.report_dir / "training_curves.png")
        except Exception as exc:  # pragma: no cover - plotting must never fail a run
            logger.warning("Could not plot training curves: %s", exc)


def plot_training_curves(outcome: TrainingOutcome, path: Path) -> None:
    """Save loss, accuracy, macro-F1 and learning-rate curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not outcome.history:
        return

    epochs = [record.epoch for record in outcome.history]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].plot(epochs, [r.train_loss for r in outcome.history], label="train")
    axes[0].plot(epochs, [r.val_loss for r in outcome.history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, [r.train_accuracy for r in outcome.history], label="train acc")
    axes[1].plot(epochs, [r.val_accuracy for r in outcome.history], label="val acc")
    axes[1].plot(epochs, [r.val_macro_f1 for r in outcome.history], label="val macro F1")
    axes[1].set_title("Accuracy and macro F1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    axes[2].plot(epochs, [r.learning_rate for r in outcome.history])
    axes[2].set_yscale("log")
    axes[2].set_title("Learning rate")
    axes[2].set_xlabel("epoch")

    # Mark where the backbone was unfrozen - the usual place to look when the curves
    # take a sudden turn.
    boundary = next((r.epoch for r in outcome.history if r.stage == "finetune"), None)
    if boundary is not None and boundary > 1:
        for axis in axes:
            axis.axvline(boundary - 0.5, color="grey", linestyle="--", linewidth=1)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
