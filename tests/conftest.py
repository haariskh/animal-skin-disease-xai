"""Shared fixtures.

Everything here is built from scratch in a temporary directory, so the suite never
depends on a downloaded dataset or a trained checkpoint being present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from aniderm.config import Config, DatasetConfig, ModelConfig  # noqa: E402
from aniderm.models.classifier import CheckpointMeta, SkinDiseaseClassifier, save_checkpoint  # noqa: E402

CLASSES = [
    {
        "key": "bacterial_dermatosis",
        "display_name": "Bacterial dermatosis",
        "aliases": ["bacterial dermatosis", "bacterial"],
        "summary": "Test class.",
        "colour": "#e8743b",
    },
    {
        "key": "fungal_infections",
        "display_name": "Fungal infections",
        "aliases": ["fungal infections", "fungal", "ringworm"],
        "summary": "Test class.",
        "colour": "#9b6bd6",
    },
    {
        "key": "healthy",
        "display_name": "Healthy",
        "aliases": ["healthy", "normal"],
        "summary": "Test class.",
        "colour": "#3fa66b",
    },
]

# Folder names deliberately differ in case and separator from the class keys, because a
# real download will not match them exactly either.
FOLDER_NAMES = ["Bacterial_dermatosis", "Fungal Infections", "healthy"]

TEST_IMAGE_SIZE = 64


def _write_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, quality=90)


@pytest.fixture
def raw_dataset_dir(tmp_path: Path) -> Path:
    """A dataset laid out as <root>/<split>/<class>/<file>, with 12 images per class."""
    root = tmp_path / "raw"
    seed = 0
    for split in ("train", "test"):
        for folder in FOLDER_NAMES:
            count = 8 if split == "train" else 4
            for index in range(count):
                # Four images share each animal id, which exercises group-aware splitting.
                name = f"dog{index // 4:02d}{folder[:3].lower()}_01_02_{index:03d}.jpg"
                _write_image(root / split / folder / name, seed)
                seed += 1
    # A stray non-image file and an unrecognised folder, both of which must be survived.
    (root / "README.txt").write_text("not an image")
    _write_image(root / "mystery_folder" / "x.jpg", 999)
    return root


@pytest.fixture
def dataset_config_path(tmp_path: Path, raw_dataset_dir: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    payload = {
        "name": "testset",
        "description": "Fixture dataset.",
        "source": {"kind": "manual", "identifier": "fixture"},
        "raw_dir": str(raw_dataset_dir),
        "manifest_path": str(tmp_path / "manifest.csv"),
        "image_extensions": [".jpg", ".png"],
        "min_images_per_class": 4,
        "classes": CLASSES,
        "split": {
            "val_size": 0.2,
            "test_size": 0.2,
            "stratify": True,
            "group_pattern": r"^[Dd]og(\w+_\d+_\d+)_\d+$",
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.fixture
def dataset_config(dataset_config_path: Path) -> DatasetConfig:
    return DatasetConfig.load(dataset_config_path)


@pytest.fixture
def config(tmp_path: Path, dataset_config_path: Path) -> Config:
    path = tmp_path / "config.yaml"
    payload = {
        "project": {"name": "test", "seed": 11, "artifacts_dir": str(tmp_path / "artifacts")},
        "dataset_config": str(dataset_config_path),
        "model": {
            "backbone": "resnet18",
            "pretrained": False,  # the suite must run without network access
            "image_size": TEST_IMAGE_SIZE,
            "dropout": 0.1,
            "drop_path_rate": 0.0,
        },
        "training": {
            "device": "cpu",
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "amp": False,
            "epochs_head": 1,
            "epochs_finetune": 1,
            "early_stopping_patience": 2,
        },
        "xai": {
            "default_method": "gradcam_pp",
            "enabled_methods": ["gradcam_pp", "integrated_gradients", "lime"],
            "integrated_gradients": {"steps": 8, "internal_batch_size": 4},
            "lime": {"num_samples": 60, "num_features": 4, "batch_size": 20},
            "faithfulness": {"enabled": True, "steps": 4},
        },
        "serving": {
            "checkpoint": str(tmp_path / "artifacts" / "checkpoints" / "best.pt"),
            "max_upload_mb": 2,
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return Config.load(path)


@pytest.fixture
def untrained_model(config: Config) -> SkinDiseaseClassifier:
    torch.manual_seed(0)
    model = SkinDiseaseClassifier(config.model, config.dataset.num_classes)
    model.eval()
    return model


@pytest.fixture
def checkpoint_path(tmp_path: Path, config: Config, untrained_model) -> Path:
    """A real checkpoint of an untrained model.

    Untrained weights are fine for everything the suite checks: shapes, plumbing, error
    handling and determinism. Accuracy is not asserted anywhere, because asserting it on
    random weights would be meaningless.
    """
    path = tmp_path / "artifacts" / "checkpoints" / "best.pt"
    untrained_model.target_layer()
    meta = CheckpointMeta.from_config(
        config, metrics={"monitor": "val_macro_f1", "best_metric": 0.5}
    )
    save_checkpoint(path, untrained_model, meta)
    return path


@pytest.fixture
def sample_image() -> Image.Image:
    rng = np.random.default_rng(3)
    return Image.fromarray(rng.integers(0, 255, size=(120, 90, 3), dtype=np.uint8))


@pytest.fixture
def sample_tensor(sample_image) -> torch.Tensor:
    from aniderm.utils.imaging import image_to_tensor

    return image_to_tensor(sample_image, TEST_IMAGE_SIZE)


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(backbone="resnet18", pretrained=False, image_size=TEST_IMAGE_SIZE)
