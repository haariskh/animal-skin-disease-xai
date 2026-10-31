"""Model construction, checkpoint integrity and Grad-CAM target-layer resolution."""

from __future__ import annotations

import pytest
import torch

from aniderm.config import Config, ModelConfig
from aniderm.models.backbones import get_module_by_name, resolve_last_conv_layer
from aniderm.models.classifier import SkinDiseaseClassifier, load_checkpoint


def test_forward_shape(untrained_model: SkinDiseaseClassifier) -> None:
    output = untrained_model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 3)


def test_freezing_leaves_only_the_head_trainable(untrained_model: SkinDiseaseClassifier) -> None:
    untrained_model.freeze_backbone()
    trainable = [p for p in untrained_model.parameters() if p.requires_grad]
    assert trainable
    assert all(
        any(p is q for q in untrained_model.classifier_parameters()) for p in trainable
    )

    untrained_model.unfreeze_all()
    assert all(p.requires_grad for p in untrained_model.parameters())


def test_head_and_backbone_parameter_groups_do_not_overlap(
    untrained_model: SkinDiseaseClassifier,
) -> None:
    head = {id(p) for p in untrained_model.classifier_parameters()}
    backbone = {id(p) for p in untrained_model.backbone_parameters()}
    assert not head & backbone
    assert len(head | backbone) == len(list(untrained_model.parameters()))


@pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0", "mobilenetv3_small_100"])
def test_target_layer_resolves_on_several_architectures(backbone: str) -> None:
    """The XAI layer must not need a hand-written layer path per architecture."""
    model = SkinDiseaseClassifier(
        ModelConfig(backbone=backbone, pretrained=False, image_size=64), num_classes=3
    )
    name, module = resolve_last_conv_layer(model.backbone, 64)
    assert name

    captured: list[torch.Size] = []
    handle = module.register_forward_hook(lambda _m, _i, out: captured.append(out.shape))
    model(torch.zeros(1, 3, 64, 64))
    handle.remove()

    # The hooked layer must still carry spatial structure, otherwise the heatmap cannot
    # be placed back onto the photograph.
    assert len(captured[0]) == 4
    assert captured[0][2] > 1 and captured[0][3] > 1


def test_get_module_by_name_walks_containers(untrained_model: SkinDiseaseClassifier) -> None:
    module = get_module_by_name(untrained_model.backbone, "layer4.0.conv1")
    assert isinstance(module, torch.nn.Conv2d)


def test_get_module_by_name_rejects_a_bad_path(untrained_model: SkinDiseaseClassifier) -> None:
    with pytest.raises(AttributeError):
        get_module_by_name(untrained_model.backbone, "layer4.0.not_a_layer")


def test_checkpoint_round_trip_preserves_weights_and_classes(
    checkpoint_path, config: Config, untrained_model
) -> None:
    restored, meta = load_checkpoint(checkpoint_path, "cpu")

    assert meta.class_keys == config.dataset.class_keys
    assert meta.image_size == config.model.image_size
    assert meta.backbone == config.model.backbone
    assert meta.gradcam_target_layer

    sample = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(untrained_model(sample), restored(sample), atol=1e-6)


def test_checkpoint_carries_its_own_class_order(checkpoint_path, config: Config) -> None:
    """Serving must never depend on a YAML that may have changed since training."""
    _, meta = load_checkpoint(checkpoint_path, "cpu")
    specs = meta.class_specs()
    assert [spec.key for spec in specs] == config.dataset.class_keys
    assert [spec.display_name for spec in specs] == config.dataset.display_names


def test_missing_checkpoint_tells_you_what_to_run(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_checkpoint(tmp_path / "absent.pt", "cpu")


def test_mismatched_architecture_is_refused(tmp_path, checkpoint_path) -> None:
    """A checkpoint silently loaded into the wrong backbone would produce nonsense."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["meta"]["backbone"] = "efficientnet_b0"
    corrupted = tmp_path / "wrong.pt"
    torch.save(payload, corrupted)

    with pytest.raises(RuntimeError, match="does not match backbone"):
        load_checkpoint(corrupted, "cpu")
