"""
Lottery Dimension Masking — Law 1: Semantic Preserve

Only inject into the top-L lottery dimensions; leave the remaining
d_model - L dimensions untouched. This prevents collateral damage
to model capabilities.

Key result:
    - full_1024 injection: 2/20 questions survive (semantics destroyed)
    - lottery_15 injection: 16/20 questions survive (precise control)
    - Random 15 dimensions > full injection — proof that "dimensions
      important for classification" ≠ "dimensions friendly for injection"
"""

import torch
from typing import Dict, Optional


def extract_lottery_mask(
    thinking_dirs: Dict[int, torch.Tensor],
    layer: int,
    n_dims: int = 15,
    d_model: int = 1024,
) -> torch.Tensor:
    """Extract lottery dimension mask from thinking directions.

    Args:
        thinking_dirs: Per-layer thinking direction vectors
        layer: Which layer to extract mask for
        n_dims: Number of lottery dimensions (Law 1: 15)
        d_model: Model hidden dimension

    Returns:
        Binary mask of shape (d_model,) with n_dims ones
    """
    if layer not in thinking_dirs:
        return torch.ones(d_model, dtype=torch.float32)

    _, top_idx = thinking_dirs[layer].float().abs().topk(n_dims)
    mask = torch.zeros(d_model, dtype=torch.float32)
    mask[top_idx] = 1.0
    return mask


def extract_lottery_masks(
    thinking_dirs: Dict[int, torch.Tensor],
    layers: list,
    n_dims: int = 15,
    d_model: int = 1024,
) -> Dict[int, torch.Tensor]:
    """Extract lottery masks for multiple layers.

    Args:
        thinking_dirs: Per-layer thinking direction vectors
        layers: List of layer indices
        n_dims: Number of lottery dimensions per layer
        d_model: Model hidden dimension

    Returns:
        Dict mapping layer → binary mask
    """
    return {
        L: extract_lottery_mask(thinking_dirs, L, n_dims, d_model)
        for L in layers
        if L in thinking_dirs
    }


def compute_lottery_gap(
    coords: torch.Tensor,
    target_coords: torch.Tensor,
    lottery_mask: torch.Tensor,
) -> float:
    """Compute gap between current and target in lottery dimensions.

    Gap determines injection strategy (Law 3: Inverted-U):
        - gap < 0.8: SKIP (model already close, injection would hurt)
        - gap >= 1.0: INJECT (model needs navigation)

    Args:
        coords: Current coordinates in lottery dims
        target_coords: Target coordinates in lottery dims
        lottery_mask: Binary mask for lottery dimensions

    Returns:
        Gap magnitude (scalar)
    """
    diff = (target_coords - coords) * lottery_mask[:coords.shape[0]]
    return diff.norm().item()
