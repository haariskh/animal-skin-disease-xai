"""The classifier wrapper, plus checkpoint saving and loading.

A checkpoint carries its own metadata - backbone name, image size, class list, metrics -
so the API can rebuild the exact model without reading any YAML. That matters because a
config file can drift after training, and a served model silently mapping class index 2
to the wrong disease is the worst possible failure for this project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aniderm.config import ClassSpec, Config, ModelConfig
from aniderm.models.backbones import create_backbone, get_module_by_name, resolve_last_conv_layer
from aniderm.utils.logging import get_logger

logger = get_logger(__name__)

CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class CheckpointMeta:
    """Everything needed to reconstruct and describe a trained model."""

    backbone: str
    image_size: int
    dropout: float
    drop_path_rate: float
    class_keys: list[str]
    class_display_names: list[str]
    class_summaries: list[str]
    class_colours: list[str]
    dataset_name: str
    format_version: int = CHECKPOINT_FORMAT_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)
    gradcam_target_layer: str | None = None

    @property
    def num_classes(self) -> int:
        return len(self.class_keys)

    def class_specs(self) -> list[ClassSpec]:
        return [
            ClassSpec(
                key=key,
                display_name=display,
                summary=summary,
                colour=colour,
            )
            for key, display, summary, colour in zip(
                self.class_keys,
                self.class_display_names,
                self.class_summaries,
                self.class_colours,
                strict=True,
            )
        ]

    @classmethod
    def from_config(cls, config: Config, metrics: dict[str, Any] | None = None) -> CheckpointMeta:
        dataset = config.dataset
        return cls(
            backbone=config.model.backbone,
            image_size=config.model.image_size,
            dropout=config.model.dropout,
            drop_path_rate=config.model.drop_path_rate,
            class_keys=dataset.class_keys,
            class_display_names=dataset.display_names,
            class_summaries=[spec.summary for spec in dataset.classes],
            class_colours=[spec.colour for spec in dataset.classes],
            dataset_name=dataset.name,
            metrics=dict(metrics or {}),
        )


class SkinDiseaseClassifier(nn.Module):
    """A timm backbone with a classification head over the configured classes."""

    def __init__(self, model_config: ModelConfig, num_classes: int) -> None:
        super().__init__()
        self.model_config = model_config
        self.num_classes = num_classes
        self.backbone = create_backbone(
            name=model_config.backbone,
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            dropout=model_config.dropout,
            drop_path_rate=model_config.drop_path_rate,
        )
        self._target_layer_name: str | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # -- parameter groups ---------------------------------------------------- #
    def classifier_parameters(self) -> list[nn.Parameter]:
        """Just the final linear layer, used for the frozen first training stage."""
        head = self.backbone.get_classifier()
        return [p for p in head.parameters()]

    def backbone_parameters(self) -> list[nn.Parameter]:
        head_ids = {id(p) for p in self.classifier_parameters()}
        return [p for p in self.backbone.parameters() if id(p) not in head_ids]

    def freeze_backbone(self) -> None:
        for param in self.backbone_parameters():
            param.requires_grad_(False)
        for param in self.classifier_parameters():
            param.requires_grad_(True)
        logger.info("Backbone frozen; training the classifier head only")

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)
        logger.info("All layers unfrozen for fine-tuning")

    # -- explainability hooks ------------------------------------------------ #
    def target_layer(self, override: str | None = None) -> nn.Module:
        """Return the layer Grad-CAM should hook."""
        if override:
            self._target_layer_name = override
            return get_module_by_name(self.backbone, override)
        if self._target_layer_name:
            return get_module_by_name(self.backbone, self._target_layer_name)
        name, module = resolve_last_conv_layer(self.backbone, self.model_config.image_size)
        self._target_layer_name = name
        return module

    @property
    def target_layer_name(self) -> str | None:
        return self._target_layer_name


def build_model(config: Config) -> SkinDiseaseClassifier:
    return SkinDiseaseClassifier(config.model, config.dataset.num_classes)


def save_checkpoint(
    path: Path,
    model: SkinDiseaseClassifier,
    meta: CheckpointMeta,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.gradcam_target_layer = model.target_layer_name
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": model.state_dict(),
        "meta": asdict(meta),
    }
    torch.save(payload, path)
    logger.info("Checkpoint saved to %s", path)


def load_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
) -> tuple[SkinDiseaseClassifier, CheckpointMeta]:
    """Rebuild a trained model from a checkpoint file.

    The backbone is constructed with ``pretrained=False`` because the saved weights are
    about to overwrite it - downloading ImageNet weights first would only waste time.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nTrain a model first:  python scripts/train.py"
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "meta" not in payload or "state_dict" not in payload:
        raise ValueError(f"{path} is not a recognised aniderm checkpoint")

    version = payload.get("format_version", 0)
    if version != CHECKPOINT_FORMAT_VERSION:
        logger.warning(
            "Checkpoint format version %s does not match the expected %s; loading anyway",
            version,
            CHECKPOINT_FORMAT_VERSION,
        )

    meta = CheckpointMeta(**payload["meta"])
    model_config = ModelConfig(
        backbone=meta.backbone,
        pretrained=False,
        image_size=meta.image_size,
        dropout=meta.dropout,
        drop_path_rate=meta.drop_path_rate,
    )
    model = SkinDiseaseClassifier(model_config, meta.num_classes)
    # strict=True so a shape mismatch, a missing key or an extra key all fail here rather
    # than producing a model that loads and then predicts nonsense.
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint does not match backbone '{meta.backbone}'. The file was most "
            f"likely written by a different architecture or a different class count.\n"
            f"Original error: {exc}"
        ) from exc

    model.to(device).eval()
    if meta.gradcam_target_layer:
        model._target_layer_name = meta.gradcam_target_layer
    logger.info(
        "Loaded '%s' trained on '%s' with %d classes",
        meta.backbone,
        meta.dataset_name,
        meta.num_classes,
    )
    return model, meta
