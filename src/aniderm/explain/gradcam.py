"""Grad-CAM++ - the primary explanation method for this project.

Why Grad-CAM++ rather than plain Grad-CAM, LIME or SHAP as the default:

* It explains the network that is actually deployed, using the gradients of the predicted
  class with respect to the last convolutional feature map. Nothing is approximated by a
  surrogate model, unlike LIME, and nothing is sampled, so the same image always produces
  the same map.
* It costs one forward and one backward pass. LIME needs several hundred forward passes
  and KernelSHAP more again, which is too slow for an interactive upload-and-see UI.
* Over plain Grad-CAM it fixes a specific failure: Grad-CAM averages the gradient over
  each channel, so when a class appears in several disconnected places - which is exactly
  what multiple lesions on one animal look like - it tends to highlight only the
  strongest one. Grad-CAM++ weights each spatial position individually, so scattered
  lesions all survive into the heatmap.

Reference: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual
Explanations for Deep Convolutional Networks", WACV 2018.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from aniderm.explain.base import Explainer
from aniderm.utils.logging import get_logger

logger = get_logger(__name__)

EPS = 1e-7


class GradCAMPlusPlus(Explainer):
    name = "gradcam_pp"
    label = "Grad-CAM++"
    description = (
        "Gradient-weighted class activation mapping. Highlights the regions of the final "
        "convolutional feature map that pushed the score for the predicted class up."
    )

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        target_layer: nn.Module,
    ) -> None:
        super().__init__(model, device)
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    # -- hook management ---------------------------------------------------- #
    def _attach(self) -> None:
        def forward_hook(_module, _inputs, output) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self._activations = tensor
            # A hook on the tensor itself captures the gradient flowing back through this
            # exact activation, which is more reliable across torch versions than a
            # module backward hook (those misbehave on modules that reuse their input).
            if tensor.requires_grad:
                tensor.register_hook(self._save_gradient)

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def _detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._activations = None
        self._gradients = None

    # -- the method --------------------------------------------------------- #
    def explain(self, image_tensor: torch.Tensor, target_index: int) -> np.ndarray:
        self._detach()
        self._attach()

        was_training = self.model.training
        self.model.eval()
        try:
            # Gradients are needed even though no parameter is being updated, so the
            # usual inference_mode / no_grad wrappers must not be used here.
            with torch.enable_grad():
                image = image_tensor.clone().requires_grad_(False)
                logits = self.model(image)
                if target_index < 0 or target_index >= logits.shape[1]:
                    raise IndexError(
                        f"target_index {target_index} is outside the model's "
                        f"{logits.shape[1]} outputs"
                    )
                self.model.zero_grad(set_to_none=True)
                logits[0, target_index].backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError(
                    "Grad-CAM++ captured no activations or gradients. The chosen target "
                    "layer is probably not on the forward path of this model."
                )

            cam = self._compute_cam(self._activations.detach(), self._gradients)
        finally:
            self._detach()
            self.model.train(was_training)
        return cam

    @staticmethod
    def _compute_cam(activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        """Apply the Grad-CAM++ weighting.

        Follows the paper's closed form, which holds when the class score is passed
        through an exponential:

            alpha_ij = g_ij^2 / (2 g_ij^2 + (sum_ab A_ab) g_ij^3)
            w_k      = sum_ij alpha_ij * relu(g_ij)
            cam      = relu(sum_k w_k A_k)

        The final ReLU is the point of the method: negative contributions are evidence
        *against* the class, and mixing them into the map would make a region that argues
        against the prediction look like support for it.
        """
        if activations.shape != gradients.shape:
            raise ValueError(
                f"Activation shape {tuple(activations.shape)} does not match gradient "
                f"shape {tuple(gradients.shape)}"
            )

        grads_pow_2 = gradients.pow(2)
        grads_pow_3 = grads_pow_2 * gradients
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)

        denominator = 2.0 * grads_pow_2 + sum_activations * grads_pow_3
        alpha = grads_pow_2 / torch.where(
            denominator.abs() > EPS, denominator, torch.full_like(denominator, EPS)
        )
        # Positions with a zero gradient contribute nothing; without this they can pick
        # up a spurious weight from the epsilon guard above.
        alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))

        weights = (alpha * torch.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1))
        return cam[0].detach().cpu().numpy().astype(np.float32)
