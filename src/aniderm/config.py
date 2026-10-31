"""Typed configuration objects loaded from the YAML files in ``configs/``.

The whole project is driven from here. No module under ``src/aniderm`` hardcodes a
class name, a class count or a filesystem path, which is what makes swapping the
dataset a one-line change in ``configs/default.yaml``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# config.py lives at <repo>/src/aniderm/config.py, so the repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    """Turn a config path into an absolute one, relative to the repository root.

    Relative paths in the YAML files stay readable in version control while still
    working no matter which directory a script is launched from.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _normalise(name: str) -> str:
    """Reduce a label to lowercase alphanumerics so folder names match aliases.

    ``"Bacterial_dermatosis"``, ``"bacterial dermatosis"`` and ``"BacterialDermatosis"``
    all collapse to ``"bacterialdermatosis"``.
    """
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key '{key}' in {context}")
    return mapping[key]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassSpec:
    """One diagnostic category the model can predict."""

    key: str
    display_name: str
    aliases: tuple[str, ...] = ()
    summary: str = ""
    colour: str = "#6b7280"

    @property
    def match_keys(self) -> set[str]:
        """Every normalised string that should resolve to this class."""
        candidates = {self.key, self.display_name, *self.aliases}
        return {_normalise(c) for c in candidates if c}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ClassSpec:
        key = str(_require(raw, "key", "classes[]"))
        return cls(
            key=key,
            display_name=str(raw.get("display_name", key.replace("_", " ").title())),
            aliases=tuple(str(a) for a in raw.get("aliases", ())),
            summary=" ".join(str(raw.get("summary", "")).split()),
            colour=str(raw.get("colour", "#6b7280")),
        )


@dataclass(frozen=True)
class SplitConfig:
    val_size: float = 0.15
    test_size: float = 0.15
    stratify: bool = True
    group_pattern: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.val_size < 1.0:
            raise ValueError(f"val_size must be in (0, 1), got {self.val_size}")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")
        if self.val_size + self.test_size >= 0.9:
            raise ValueError(
                f"val_size + test_size = {self.val_size + self.test_size:.2f} leaves too "
                "little data for training"
            )
        if self.group_pattern is not None:
            try:
                re.compile(self.group_pattern)
            except re.error as exc:
                raise ValueError(f"split.group_pattern is not a valid regex: {exc}") from exc

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SplitConfig:
        pattern = raw.get("group_pattern")
        return cls(
            val_size=float(raw.get("val_size", 0.15)),
            test_size=float(raw.get("test_size", 0.15)),
            stratify=bool(raw.get("stratify", True)),
            group_pattern=str(pattern) if pattern else None,
        )


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    raw_dir: Path
    manifest_path: Path
    classes: tuple[ClassSpec, ...]
    split: SplitConfig
    description: str = ""
    source_kind: str = "manual"
    source_identifier: str = ""
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    min_images_per_class: int = 1

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError(f"Dataset '{self.name}' declares no classes")
        keys = [c.key for c in self.classes]
        duplicates = {k for k in keys if keys.count(k) > 1}
        if duplicates:
            raise ValueError(f"Duplicate class keys in '{self.name}': {sorted(duplicates)}")
        # An alias shared by two classes would make folder resolution ambiguous, so it is
        # rejected at load time rather than silently mislabelling images later.
        seen: dict[str, str] = {}
        for spec in self.classes:
            for token in spec.match_keys:
                if token in seen and seen[token] != spec.key:
                    raise ValueError(
                        f"Alias '{token}' is claimed by both '{seen[token]}' and '{spec.key}'"
                    )
                seen[token] = spec.key

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_keys(self) -> list[str]:
        """Class keys in label order. Index i is the model's output unit i."""
        return [c.key for c in self.classes]

    @property
    def display_names(self) -> list[str]:
        return [c.display_name for c in self.classes]

    def index_of(self, key: str) -> int:
        token = _normalise(key)
        for idx, spec in enumerate(self.classes):
            if token in spec.match_keys:
                return idx
        raise KeyError(f"'{key}' does not match any class of dataset '{self.name}'")

    def spec_of(self, key: str) -> ClassSpec:
        return self.classes[self.index_of(key)]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetConfig:
        source = raw.get("source", {}) or {}
        extensions = raw.get("image_extensions") or (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        return cls(
            name=str(_require(raw, "name", "dataset config")),
            description=str(raw.get("description", "")),
            raw_dir=resolve_path(_require(raw, "raw_dir", "dataset config")),
            manifest_path=resolve_path(_require(raw, "manifest_path", "dataset config")),
            classes=tuple(
                ClassSpec.from_dict(c) for c in _require(raw, "classes", "dataset config")
            ),
            split=SplitConfig.from_dict(raw.get("split", {}) or {}),
            source_kind=str(source.get("kind", "manual")),
            source_identifier=str(source.get("identifier", "")),
            image_extensions=tuple(str(e).lower() for e in extensions),
            min_images_per_class=int(raw.get("min_images_per_class", 1)),
        )

    @classmethod
    def load(cls, path: str | Path) -> DatasetConfig:
        resolved = resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset config not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})


# --------------------------------------------------------------------------- #
# Model / training
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    image_size: int = 224
    dropout: float = 0.3
    drop_path_rate: float = 0.1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        return cls(
            backbone=str(raw.get("backbone", "efficientnet_b0")),
            pretrained=bool(raw.get("pretrained", True)),
            image_size=int(raw.get("image_size", 224)),
            dropout=float(raw.get("dropout", 0.3)),
            drop_path_rate=float(raw.get("drop_path_rate", 0.1)),
        )


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.3
    rotation_degrees: float = 25.0
    scale_jitter: tuple[float, float] = (0.75, 1.0)
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.15
    hue: float = 0.02
    random_erasing: float = 0.15

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AugmentationConfig:
        jitter = raw.get("scale_jitter", (0.75, 1.0))
        return cls(
            horizontal_flip=float(raw.get("horizontal_flip", 0.5)),
            vertical_flip=float(raw.get("vertical_flip", 0.3)),
            rotation_degrees=float(raw.get("rotation_degrees", 25.0)),
            scale_jitter=(float(jitter[0]), float(jitter[1])),
            brightness=float(raw.get("brightness", 0.2)),
            contrast=float(raw.get("contrast", 0.2)),
            saturation=float(raw.get("saturation", 0.15)),
            hue=float(raw.get("hue", 0.02)),
            random_erasing=float(raw.get("random_erasing", 0.15)),
        )


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "auto"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    amp: bool = True
    epochs_head: int = 5
    epochs_finetune: int = 40
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    label_smoothing: float = 0.05
    class_weighting: str = "balanced"
    monitor: str = "val_macro_f1"
    early_stopping_patience: int = 10

    _MONITORS = ("val_macro_f1", "val_accuracy", "val_loss")

    def __post_init__(self) -> None:
        if self.monitor not in self._MONITORS:
            raise ValueError(f"training.monitor must be one of {self._MONITORS}")
        if self.scheduler not in ("cosine", "plateau", "none"):
            raise ValueError("training.scheduler must be cosine, plateau or none")
        if self.class_weighting not in ("balanced", "none"):
            raise ValueError("training.class_weighting must be balanced or none")

    @property
    def total_epochs(self) -> int:
        return self.epochs_head + self.epochs_finetune

    @property
    def greater_is_better(self) -> bool:
        return self.monitor != "val_loss"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingConfig:
        return cls(
            device=str(raw.get("device", "auto")),
            batch_size=int(raw.get("batch_size", 32)),
            num_workers=int(raw.get("num_workers", 4)),
            pin_memory=bool(raw.get("pin_memory", True)),
            amp=bool(raw.get("amp", True)),
            epochs_head=int(raw.get("epochs_head", 5)),
            epochs_finetune=int(raw.get("epochs_finetune", 40)),
            lr_head=float(raw.get("lr_head", 1e-3)),
            lr_backbone=float(raw.get("lr_backbone", 1e-4)),
            weight_decay=float(raw.get("weight_decay", 1e-4)),
            grad_clip=float(raw.get("grad_clip", 1.0)),
            scheduler=str(raw.get("scheduler", "cosine")),
            warmup_epochs=int(raw.get("warmup_epochs", 2)),
            label_smoothing=float(raw.get("label_smoothing", 0.05)),
            class_weighting=str(raw.get("class_weighting", "balanced")),
            monitor=str(raw.get("monitor", "val_macro_f1")),
            early_stopping_patience=int(raw.get("early_stopping_patience", 10)),
        )


@dataclass(frozen=True)
class CrossValConfig:
    enabled: bool = False
    n_splits: int = 5

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CrossValConfig:
        return cls(
            enabled=bool(raw.get("enabled", False)),
            n_splits=int(raw.get("n_splits", 5)),
        )


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class XAIConfig:
    default_method: str = "gradcam_pp"
    enabled_methods: tuple[str, ...] = ("gradcam_pp", "integrated_gradients", "lime")
    overlay_alpha: float = 0.45
    colormap: str = "turbo"
    gradcam_target_layer: str | None = None
    ig_steps: int = 64
    ig_baseline: str = "blur"
    ig_internal_batch_size: int = 16
    lime_num_samples: int = 800
    lime_num_features: int = 8
    lime_batch_size: int = 32
    lime_segmentation: dict[str, Any] = field(
        default_factory=lambda: {"kernel_size": 4, "max_dist": 200, "ratio": 0.2}
    )
    faithfulness_enabled: bool = True
    faithfulness_steps: int = 10
    faithfulness_deletion_baseline: str = "zero"
    faithfulness_insertion_baseline: str = "blur"

    def __post_init__(self) -> None:
        if self.default_method not in self.enabled_methods:
            raise ValueError(
                f"xai.default_method '{self.default_method}' is not listed in enabled_methods"
            )
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError("xai.overlay_alpha must be between 0 and 1")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> XAIConfig:
        cam = raw.get("gradcam_pp", {}) or {}
        ig = raw.get("integrated_gradients", {}) or {}
        lime_cfg = raw.get("lime", {}) or {}
        faith = raw.get("faithfulness", {}) or {}
        target_layer = cam.get("target_layer")
        methods = raw.get("enabled_methods") or (
            "gradcam_pp",
            "integrated_gradients",
            "lime",
        )
        return cls(
            default_method=str(raw.get("default_method", "gradcam_pp")),
            enabled_methods=tuple(str(m) for m in methods),
            overlay_alpha=float(raw.get("overlay_alpha", 0.45)),
            colormap=str(raw.get("colormap", "turbo")),
            gradcam_target_layer=str(target_layer) if target_layer else None,
            ig_steps=int(ig.get("steps", 64)),
            ig_baseline=str(ig.get("baseline", "blur")),
            ig_internal_batch_size=int(ig.get("internal_batch_size", 16)),
            lime_num_samples=int(lime_cfg.get("num_samples", 800)),
            lime_num_features=int(lime_cfg.get("num_features", 8)),
            lime_batch_size=int(lime_cfg.get("batch_size", 32)),
            lime_segmentation=dict(
                lime_cfg.get("segmentation", {"kernel_size": 4, "max_dist": 200, "ratio": 0.2})
            ),
            faithfulness_enabled=bool(faith.get("enabled", True)),
            faithfulness_steps=int(faith.get("steps", 10)),
            faithfulness_deletion_baseline=str(faith.get("deletion_baseline", "zero")),
            faithfulness_insertion_baseline=str(faith.get("insertion_baseline", "blur")),
        )


@dataclass(frozen=True)
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    checkpoint: Path = field(default_factory=lambda: resolve_path("artifacts/checkpoints/best.pt"))
    max_upload_mb: int = 10
    allowed_origins: tuple[str, ...] = ("http://localhost:5173",)
    low_confidence_threshold: float = 0.55

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ServingConfig:
        origins = raw.get("allowed_origins") or ("http://localhost:5173",)
        return cls(
            host=str(raw.get("host", "127.0.0.1")),
            port=int(raw.get("port", 8000)),
            checkpoint=resolve_path(raw.get("checkpoint", "artifacts/checkpoints/best.pt")),
            max_upload_mb=int(raw.get("max_upload_mb", 10)),
            allowed_origins=tuple(str(o) for o in origins),
            low_confidence_threshold=float(raw.get("low_confidence_threshold", 0.55)),
        )


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProjectConfig:
    name: str = "animal-skin-disease-xai"
    seed: int = 42
    artifacts_dir: Path = field(default_factory=lambda: resolve_path("artifacts"))
    deterministic: bool = False

    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProjectConfig:
        return cls(
            name=str(raw.get("name", "animal-skin-disease-xai")),
            seed=int(raw.get("seed", 42)),
            artifacts_dir=resolve_path(raw.get("artifacts_dir", "artifacts")),
            deterministic=bool(raw.get("deterministic", False)),
        )


@dataclass(frozen=True)
class Config:
    """The fully resolved configuration tree."""

    project: ProjectConfig
    dataset: DatasetConfig
    model: ModelConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    crossval: CrossValConfig
    xai: XAIConfig
    serving: ServingConfig
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path, dataset_override: str | Path | None = None) -> Config:
        """Read ``configs/default.yaml`` (or any copy of it) and everything it points at.

        ``dataset_override`` lets a CLI flag swap the dataset definition without editing
        the YAML, which is handy when comparing two datasets side by side.
        """
        resolved = resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Config not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        dataset_ref = dataset_override or raw.get("dataset_config")
        if not dataset_ref:
            raise KeyError("Config must set 'dataset_config' (or pass a dataset override)")

        return cls(
            project=ProjectConfig.from_dict(raw.get("project", {}) or {}),
            dataset=DatasetConfig.load(dataset_ref),
            model=ModelConfig.from_dict(raw.get("model", {}) or {}),
            augmentation=AugmentationConfig.from_dict(raw.get("augmentation", {}) or {}),
            training=TrainingConfig.from_dict(raw.get("training", {}) or {}),
            crossval=CrossValConfig.from_dict(raw.get("crossval", {}) or {}),
            xai=XAIConfig.from_dict(raw.get("xai", {}) or {}),
            serving=ServingConfig.from_dict(raw.get("serving", {}) or {}),
            source_path=resolved,
        )


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
