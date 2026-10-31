"""Single-image prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aniderm.config import ClassSpec
from aniderm.models.classifier import load_checkpoint
from aniderm.utils.device import select_device
from aniderm.utils.imaging import image_to_tensor
from aniderm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassScore:
    key: str
    display_name: str
    probability: float
    colour: str
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "probability": round(self.probability, 6),
            "colour": self.colour,
            "summary": self.summary,
        }


@dataclass
class Prediction:
    """Model output for one image, before any explanation is computed."""

    scores: list[ClassScore]
    logits: np.ndarray
    probabilities: np.ndarray
    target_index: int
    image_tensor: torch.Tensor

    @property
    def top(self) -> ClassScore:
        return self.scores[0]

    @property
    def margin(self) -> float:
        """Gap between the top two probabilities.

        A narrow margin is the honest signal that the model is torn between two
        categories, which is worth surfacing in the UI rather than hiding behind a single
        confident-looking label.
        """
        if len(self.scores) < 2:
            return 1.0
        return self.scores[0].probability - self.scores[1].probability

    @property
    def entropy(self) -> float:
        """Shannon entropy of the distribution, normalised to [0, 1]."""
        probs = np.clip(self.probabilities, 1e-12, 1.0)
        raw = float(-(probs * np.log(probs)).sum())
        return raw / float(np.log(len(probs))) if len(probs) > 1 else 0.0


class Predictor:
    """Loads a checkpoint once and classifies images with it."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device_preference: str = "auto",
    ) -> None:
        self.device = select_device(device_preference)
        self.model, self.meta = load_checkpoint(Path(checkpoint_path), self.device)
        self.model.eval()
        self.class_specs: list[ClassSpec] = self.meta.class_specs()

    @property
    def image_size(self) -> int:
        return self.meta.image_size

    def prepare(self, image: Image.Image) -> torch.Tensor:
        return image_to_tensor(image, self.meta.image_size).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        target_class: str | None = None,
    ) -> Prediction:
        """Classify one image.

        ``target_class`` forces the explanation to a class other than the top prediction,
        which is how a user asks "why did you NOT say fungal infection?" - a genuinely
        useful question that a fixed top-1 explanation cannot answer.
        """
        tensor = self.prepare(image)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

        ordering = np.argsort(probabilities)[::-1]
        scores = [
            ClassScore(
                key=self.class_specs[i].key,
                display_name=self.class_specs[i].display_name,
                probability=float(probabilities[i]),
                colour=self.class_specs[i].colour,
                summary=self.class_specs[i].summary,
            )
            for i in ordering
        ]

        target_index = int(ordering[0])
        if target_class:
            target_index = self.index_of(target_class)

        return Prediction(
            scores=scores,
            logits=logits[0].detach().float().cpu().numpy(),
            probabilities=probabilities,
            target_index=target_index,
            image_tensor=tensor,
        )

    def index_of(self, class_key: str) -> int:
        for index, spec in enumerate(self.class_specs):
            if spec.key == class_key or spec.display_name.lower() == class_key.lower():
                return index
        raise KeyError(
            f"'{class_key}' is not one of this model's classes: "
            f"{[spec.key for spec in self.class_specs]}"
        )

    def describe(self) -> dict[str, object]:
        """Model card data for the API and the UI footer."""
        return {
            "backbone": self.meta.backbone,
            "image_size": self.meta.image_size,
            "dataset": self.meta.dataset_name,
            "trained_at": self.meta.created_at,
            "num_classes": self.meta.num_classes,
            "device": self.device.type,
            "classes": [
                {
                    "key": spec.key,
                    "display_name": spec.display_name,
                    "summary": spec.summary,
                    "colour": spec.colour,
                }
                for spec in self.class_specs
            ],
            "metrics": self.meta.metrics,
        }
