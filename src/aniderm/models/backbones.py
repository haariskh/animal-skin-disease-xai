"""Backbone construction and layer introspection."""

from __future__ import annotations

import torch
import torch.nn as nn

from aniderm.utils.logging import get_logger

logger = get_logger(__name__)


def create_backbone(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    """Build a timm classifier with a fresh head sized to ``num_classes``.

    timm is used rather than hand-rolling a CNN because ImageNet-pretrained features are
    the single biggest factor on a dataset of a few hundred images - a network trained
    from scratch on 300 photographs will not generalise.
    """
    import timm

    try:
        model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )
    except RuntimeError as exc:
        # The usual cause is no internet on the first run, when the pretrained weights
        # still have to be fetched and cached.
        raise RuntimeError(
            f"Could not create backbone '{name}'. If this is the first run, pretrained "
            f"weights need to be downloaded once; check your connection or set "
            f"model.pretrained to false in the config.\nOriginal error: {exc}"
        ) from exc

    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Backbone '%s' ready: %.1fM parameters, %d output classes, pretrained=%s",
        name,
        total / 1e6,
        num_classes,
        pretrained,
    )
    return model


def get_module_by_name(model: nn.Module, dotted_name: str) -> nn.Module:
    """Fetch a submodule from a dotted path such as ``"blocks.6.0.conv_pwl"``.

    Numeric segments index into ``nn.Sequential`` / ``nn.ModuleList`` containers.
    """
    module: nn.Module = model
    for part in dotted_name.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            if not hasattr(module, part):
                raise AttributeError(
                    f"'{dotted_name}' is not a valid module path: no attribute '{part}'"
                )
            module = getattr(module, part)
    if not isinstance(module, nn.Module):
        raise TypeError(f"'{dotted_name}' resolved to {type(module)}, not an nn.Module")
    return module


def resolve_last_conv_layer(
    model: nn.Module,
    input_size: int,
    device: torch.device | None = None,
) -> tuple[str, nn.Module]:
    """Find the deepest module that still emits a spatial feature map.

    Grad-CAM needs the last activation that retains spatial structure - that is what makes
    the heatmap localisable back onto the photograph. Rather than hardcoding a path per
    architecture (``layer4`` for ResNet, ``conv_head`` for EfficientNet, and so on), this
    runs one dummy forward pass with hooks on every leaf module and keeps the last one
    whose output is a 4-D tensor larger than 1x1 spatially. That makes the XAI layer work
    on any timm backbone the config names.

    Returns the module's dotted name and the module itself.
    """
    device = device or next(model.parameters()).device
    captured: list[tuple[str, nn.Module]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str, module: nn.Module):
        def hook(_module, _inputs, output) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) and output else output
            if isinstance(tensor, torch.Tensor) and tensor.dim() == 4:
                if tensor.shape[2] > 1 and tensor.shape[3] > 1:
                    captured.append((name, module))

        return hook

    for name, module in model.named_modules():
        # Leaf modules only: a container's output duplicates its last child's, and
        # hooking both would make the "last" ambiguous.
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, input_size, input_size, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    if not captured:
        raise RuntimeError(
            "No convolutional feature map found in this model, so Grad-CAM cannot be "
            "applied. Vision transformers need a different explainer; use "
            "integrated_gradients or lime instead."
        )

    name, module = captured[-1]
    logger.info(
        "Grad-CAM target layer resolved automatically to '%s' (%s)",
        name,
        type(module).__name__,
    )
    return name, module
