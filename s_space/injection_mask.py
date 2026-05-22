"""
Injection Masking — Selective Dimension Control

Masks restrict injection to specific dimensions in the model's hidden space.
This prevents collateral damage to model capabilities.

Key insight from S-space:
    - Every axis has information, but not every axis should be pushed at once
    - The need_k mechanism automatically selects WHICH axes to push
    - Masks provide an additional safety layer for WHEN you want to restrict
      injection to specific dimensions

Historical note:
    The original "lottery dimensions" concept has been superseded by the
    understanding that all axes carry information. Need-driven axis selection
    (need_k) is the primary mechanism for choosing which axes to navigate.
    Masks are now a secondary, optional safety tool.

Mask sources:
    1. From metric weights: Top-K g_k dimensions are "front axes" (coarse grain)
    2. From user specification: Manually specify which dimensions to inject
    3. From experimental data: e.g., thinking_dirs or other extracted directions
    4. No mask (default): Full-dimension injection, let need_k do the selection
"""

import torch
from typing import Dict, List, Optional


def mask_from_metric_weights(
    metric_weights: torch.Tensor,
    principal_dirs: torch.Tensor,
    top_k: int = 15,
) -> torch.Tensor:
    """Create injection mask from metric weights.

    Selects the top-K dimensions by metric weight g_k — these are the
    "front axes" that carry the most navigational signal.

    This is model-agnostic: any model with PCA data can generate this mask.

    Args:
        metric_weights: g_k, shape (K,)
        principal_dirs: ê_k, shape (K, d_model)
        top_k: Number of dimensions to select

    Returns:
        Binary mask of shape (d_model,)
    """
    K = min(top_k, metric_weights.shape[0], principal_dirs.shape[0])
    _, top_indices = metric_weights[:K].abs().topk(K)

    d_model = principal_dirs.shape[1]
    mask = torch.zeros(d_model, dtype=torch.float32)

    for idx in top_indices:
        # Each principal direction contributes to all d_model dimensions,
        # but we mark the most significant projection dimensions
        dir_abs = principal_dirs[idx].abs()
        _, dim_idx = dir_abs.topk(1)
        mask[dim_idx] = 1.0

    return mask


def mask_from_directions(
    direction_vectors: Dict[int, torch.Tensor],
    layer: int,
    top_k: int = 15,
    d_model: Optional[int] = None,
) -> torch.Tensor:
    """Create injection mask from direction vectors (e.g., thinking_dirs).

    This is the legacy "lottery mask" extraction, kept for backward
    compatibility with existing experimental data.

    Args:
        direction_vectors: Per-layer direction vectors, {layer: (d_model,)}
        layer: Which layer to extract mask for
        top_k: Number of top dimensions to select
        d_model: Model hidden dimension (inferred if not provided)

    Returns:
        Binary mask of shape (d_model,) with top_k ones
    """
    if layer not in direction_vectors:
        if d_model is not None:
            return torch.ones(d_model, dtype=torch.float32)
        return torch.ones(1, dtype=torch.float32)  # fallback

    vec = direction_vectors[layer]
    if d_model is None:
        d_model = vec.shape[0]

    _, top_idx = vec.float().abs().topk(min(top_k, vec.shape[0]))
    mask = torch.zeros(d_model, dtype=torch.float32)
    mask[top_idx] = 1.0
    return mask


def mask_from_axes(
    axis_indices: List[int],
    d_model: int,
) -> torch.Tensor:
    """Create injection mask from explicit axis indices.

    This allows users to manually specify which axes to inject into,
    e.g., based on single-knob experiments (ê₂=perspective, ê₆=systematic, etc.)

    Args:
        axis_indices: List of dimension indices to include
        d_model: Model hidden dimension

    Returns:
        Binary mask of shape (d_model,)
    """
    mask = torch.zeros(d_model, dtype=torch.float32)
    for idx in axis_indices:
        if 0 <= idx < d_model:
            mask[idx] = 1.0
    return mask


def no_mask(d_model: int) -> torch.Tensor:
    """Create a full mask (all dimensions active) — effectively no masking.

    This is the default for the model-agnostic mode: let need_k handle
    axis selection, don't restrict injection dimensions.

    Args:
        d_model: Model hidden dimension

    Returns:
        Ones mask of shape (d_model,)
    """
    return torch.ones(d_model, dtype=torch.float32)


def compute_navigation_gap(
    coords: torch.Tensor,
    target_coords: torch.Tensor,
    injection_mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute gap between current and target coordinates.

    Gap determines injection strategy:
        - Small gap: model already close, reduce injection
        - Large gap: model needs navigation, increase injection

    Args:
        coords: Current coordinates
        target_coords: Target coordinates
        injection_mask: Optional dimension mask

    Returns:
        Gap magnitude (scalar)
    """
    diff = target_coords - coords
    if injection_mask is not None and injection_mask.shape[0] >= coords.shape[0]:
        diff = diff * injection_mask[:coords.shape[0]]
    return diff.norm().item()
