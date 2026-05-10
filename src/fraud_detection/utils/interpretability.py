"""Model interpretability via Captum Integrated Gradients.

Provides feature-level attribution scores for each prediction so the
API can explain *why* a transaction was flagged as fraud.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

_CAPTUM_AVAILABLE = True
try:
    from captum.attr import IntegratedGradients
except ImportError:
    _CAPTUM_AVAILABLE = False
    logger.warning("Captum not installed — falling back to gradient×input interpretability.")


def _gradient_x_input(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Fallback interpretability when Captum is unavailable."""
    input_tensor = input_tensor.detach().clone().requires_grad_(True)
    logits = model(input_tensor)
    logits.sum().backward()
    attributions = (input_tensor.grad * input_tensor).detach().cpu().numpy().squeeze()
    return attributions


def compute_attributions(
    model: nn.Module,
    input_tensor: torch.Tensor,
    feature_names: list[str] | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Compute feature attributions for a single prediction.

    Uses Captum Integrated Gradients when available, otherwise falls
    back to gradient×input.

    Parameters
    ----------
    model : nn.Module
        The fraud detection model (must be in eval mode).
    input_tensor : torch.Tensor
        Shape ``(1, input_dim)`` — a single sample.
    feature_names : list[str] | None
        Human-readable names for each feature dimension.
    top_n : int
        Number of top contributing features to return.

    Returns
    -------
    dict with keys:
        method, top_features (list of {feature, index, attribution, direction}),
        all_attributions (raw float list).
    """
    model.eval()

    if _CAPTUM_AVAILABLE:
        method = "integrated_gradients"
        try:
            ig = IntegratedGradients(model)
            attr = ig.attribute(input_tensor, n_steps=50)
            attributions = attr.detach().cpu().numpy().squeeze()
        except Exception as exc:
            logger.warning("Captum IG failed (%s) — falling back to gradient×input", exc)
            method = "gradient_x_input"
            attributions = _gradient_x_input(model, input_tensor)
    else:
        method = "gradient_x_input"
        attributions = _gradient_x_input(model, input_tensor)

    abs_attr = np.abs(attributions)
    top_indices = np.argsort(abs_attr)[-top_n:][::-1].tolist()

    top_features = []
    for idx in top_indices:
        name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
        top_features.append(
            {
                "feature": name,
                "index": idx,
                "attribution": round(float(attributions[idx]), 6),
                "direction": "increases_fraud" if attributions[idx] > 0 else "decreases_fraud",
            }
        )

    return {
        "method": method,
        "top_features": top_features,
        "all_attributions": [round(float(a), 6) for a in attributions],
    }
