"""The torch ``Dataset`` and ``DataLoader`` wiring, driven by the manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from aniderm.config import Config
from aniderm.data.manifest import load_manifest, resolve_manifest_paths, split_frame
from aniderm.data.transforms import build_eval_transform, build_train_transform
from aniderm.utils.imaging import load_image
from aniderm.utils.logging import get_logger
from aniderm.utils.seed import worker_init_fn

logger = get_logger(__name__)


class SkinDiseaseDataset(Dataset):
    """Reads images listed in one split of the manifest.

    Images are decoded lazily on access rather than preloaded. On a dataset of a few
    hundred photographs either approach works; lazy decoding is kept because it scales
    unchanged if you merge in extra data later.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        transform,
        class_keys: list[str],
    ) -> None:
        if frame.empty:
            raise ValueError("Cannot build a dataset from an empty manifest split")
        self.paths: list[Path] = resolve_manifest_paths(frame)
        self.labels: list[int] = frame["label"].astype(int).tolist()
        self.class_keys = list(class_keys)
        self.transform = transform

        missing = [p for p in self.paths[:50] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} manifest path(s) do not exist, e.g. {missing[0]}.\n"
                f"Re-run scripts/prepare_data.py if the data directory moved."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.paths[index]
        image = load_image(path)
        tensor = self.transform(image)
        return tensor, self.labels[index]


def build_dataloaders(
    config: Config,
    splits: tuple[str, ...] = ("train", "val", "test"),
    manifest: pd.DataFrame | None = None,
) -> dict[str, DataLoader]:
    """Create one DataLoader per requested split.

    Only the training loader shuffles and augments. ``drop_last`` is enabled for training
    when there is more than one full batch, because a trailing batch of size 1 makes
    BatchNorm statistics unstable.
    """
    frame = manifest if manifest is not None else load_manifest(config.dataset)
    train_tf = build_train_transform(config.model.image_size, config.augmentation)
    eval_tf = build_eval_transform(config.model.image_size)

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        subset = split_frame(frame, split)
        if subset.empty:
            logger.warning("Split '%s' is empty; skipping its dataloader", split)
            continue

        is_train = split == "train"
        dataset = SkinDiseaseDataset(
            frame=subset,
            transform=train_tf if is_train else eval_tf,
            class_keys=config.dataset.class_keys,
        )
        batch_size = config.training.batch_size
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory and torch.cuda.is_available(),
            drop_last=is_train and len(dataset) > batch_size,
            worker_init_fn=worker_init_fn,
            persistent_workers=config.training.num_workers > 0,
        )
        logger.info(
            "Dataloader '%s': %d images, %d batches", split, len(dataset), len(loaders[split])
        )
    return loaders
