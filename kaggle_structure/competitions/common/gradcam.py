"""Grad-CAM / saliency overlays for timm classifiers.

Thin wrapper over `pytorch_grad_cam` that auto-picks the last conv/norm layer of a
timm backbone as the CAM target. Use it to sanity-check that a medical classifier
looks at the lesion, and to produce explainability figures for writeups.

Example
-------
    from gradcam import overlay_cam
    heat = overlay_cam(model.net, input_tensor, rgb_image)  # HxWx3 uint8
    plt.imshow(heat)
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install grad-cam  (imported as pytorch_grad_cam)") from e


def _default_target_layer(model: torch.nn.Module):
    """Last 2D conv/norm module — a reasonable CAM target for most timm CNNs.

    For pure ViT backbones prefer the final block's norm and reshape_transform;
    override `target_layer` explicitly in that case.
    """
    candidate = None
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            candidate = m
    if candidate is None:
        raise ValueError("No Conv2d/BatchNorm2d found; pass target_layer explicitly.")
    return candidate


def overlay_cam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,   # (1, C, H, W), already normalized
    rgb_image: np.ndarray,        # (H, W, 3) float in [0, 1]
    target_layer=None,
    target_class: int | None = None,
) -> np.ndarray:
    """Return an RGB uint8 image with the Grad-CAM heatmap blended in."""
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    layer = target_layer or _default_target_layer(model)
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    with GradCAM(model=model, target_layers=[layer]) as cam:
        grayscale = cam(input_tensor=input_tensor, targets=targets)[0]  # (H, W)
    return show_cam_on_image(rgb_image, grayscale, use_rgb=True)
