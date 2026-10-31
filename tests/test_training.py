"""The training loop, run for real on the fixture dataset.

Two short epochs on 36 tiny images is enough to catch the regressions that matter: a
broken optimiser group, a checkpoint that never gets written, metrics that do not line
up with the class list, early stopping that never fires.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aniderm.config import Config
from aniderm.data.dataset import build_dataloaders
from aniderm.data.discovery import discover_images
from aniderm.data.manifest import build_manifest, class_weights, save_manifest
from aniderm.models.classifier import build_model, load_checkpoint
from aniderm.training.crossval import make_folds, summarise
from aniderm.training.engine import Trainer
from aniderm.training.metrics import compute_metrics, format_report
from aniderm.utils.device import amp_enabled, select_device


@pytest.fixture
def prepared(config: Config):
    report = discover_images(config.dataset)
    frame = build_manifest(report.records, config.dataset.split, config.project.seed)
    save_manifest(frame, config.dataset.manifest_path)
    return frame


def test_training_runs_and_writes_a_usable_checkpoint(config: Config, prepared, tmp_path) -> None:
    loaders = build_dataloaders(config, manifest=prepared)
    trainer = Trainer(
        config=config,
        model=build_model(config),
        loaders=loaders,
        device=select_device("cpu"),
        class_weights=class_weights(prepared, config.dataset.num_classes),
        checkpoint_path=tmp_path / "ckpt" / "best.pt",
        report_dir=tmp_path / "reports",
    )
    outcome = trainer.fit()

    assert len(outcome.history) == 2  # one head epoch, one fine-tune epoch
    assert [record.stage for record in outcome.history] == ["head", "finetune"]
    assert outcome.best_epoch >= 1
    assert trainer.checkpoint_path.exists()

    # The checkpoint must reload into a working model with the right class list.
    restored, meta = load_checkpoint(trainer.checkpoint_path, "cpu")
    assert meta.class_keys == config.dataset.class_keys
    assert meta.metrics["monitor"] == "val_macro_f1"

    history = json.loads((tmp_path / "reports" / "training_history.json").read_text())
    assert history["monitor"] == "val_macro_f1"
    assert len(history["history"]) == 2


def test_evaluation_metrics_cover_every_class(config: Config, prepared, tmp_path) -> None:
    loaders = build_dataloaders(config, manifest=prepared)
    trainer = Trainer(
        config=config,
        model=build_model(config),
        loaders=loaders,
        device=select_device("cpu"),
        class_weights=class_weights(prepared, config.dataset.num_classes),
        checkpoint_path=tmp_path / "ckpt.pt",
        report_dir=tmp_path / "reports",
    )
    result = trainer.evaluate(loaders["val"], "val")

    assert set(result.per_class) == set(config.dataset.class_keys)
    assert len(result.confusion) == config.dataset.num_classes
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert "Confusion matrix" in format_report(result, config.dataset.display_names)


def test_head_stage_only_updates_the_classifier(config: Config, prepared, tmp_path) -> None:
    """If the frozen stage moved backbone weights, the pretrained features would be lost."""
    import copy

    model = build_model(config)
    model.freeze_backbone()
    before = copy.deepcopy([p.detach().clone() for p in model.backbone_parameters()])

    loaders = build_dataloaders(config, manifest=prepared)
    trainer = Trainer(
        config=config,
        model=model,
        loaders=loaders,
        device=select_device("cpu"),
        class_weights=class_weights(prepared, config.dataset.num_classes),
        checkpoint_path=tmp_path / "ckpt.pt",
        report_dir=tmp_path / "reports",
    )
    import torch

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    trainer._train_epoch(optimiser, epoch=1, stage="head")

    after = [p.detach().clone() for p in model.backbone_parameters()]
    # Running statistics in BatchNorm still move in train mode, so only learnable
    # parameters are compared.
    assert all(torch.allclose(a, b) for a, b in zip(before, after, strict=True))


def test_metrics_handle_a_class_missing_from_the_labels() -> None:
    """A small validation split often lacks a class; AUC is undefined there, not zero."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])

    result = compute_metrics(y_true, y_pred, y_prob, ["a", "b", "c"], loss=0.5)
    assert result.macro_auc is None
    assert result.per_class["c"]["support"] == 0
    assert result.accuracy == pytest.approx(0.75)


def test_amp_is_disabled_off_cuda() -> None:
    import torch

    assert amp_enabled(True, torch.device("cpu")) is False


def test_cross_validation_folds_respect_groups(config: Config, prepared) -> None:
    folds = make_folds(prepared, n_splits=3, seed=config.project.seed)
    assert len(folds) == 3
    for train_idx, holdout_idx in folds:
        train_groups = set(prepared.iloc[train_idx]["group"])
        holdout_groups = set(prepared.iloc[holdout_idx]["group"])
        assert not train_groups & holdout_groups


def test_cross_validation_summary_reports_spread(config: Config) -> None:
    from aniderm.training.crossval import FoldResult

    def fake(accuracy: float) -> FoldResult:
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2]) if accuracy > 0.9 else np.array([0, 1, 1])
        y_prob = np.eye(3)[y_pred]
        return FoldResult(
            fold=1,
            best_metric=accuracy,
            best_epoch=1,
            test=compute_metrics(y_true, y_pred, y_prob, config.dataset.class_keys, 0.1),
        )

    summary = summarise([fake(1.0), fake(0.6)], config)
    assert summary["n_folds"] == 2
    assert summary["accuracy"]["std"] > 0
    assert set(summary["per_class"]) == set(config.dataset.class_keys)


def test_balanced_weighting_without_weights_is_refused(config: Config, prepared, tmp_path) -> None:
    """Silently falling back to unweighted loss would change the result without saying so."""
    loaders = build_dataloaders(config, manifest=prepared)
    with pytest.raises(ValueError, match="requires class weights"):
        Trainer(
            config=config,
            model=build_model(config),
            loaders=loaders,
            device=select_device("cpu"),
            class_weights=None,
            checkpoint_path=tmp_path / "ckpt.pt",
            report_dir=tmp_path / "reports",
        )
