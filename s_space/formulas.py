"""
S-Space Three Core Formulas — Model-Agnostic Implementations

These formulas are derived from the mathematical structure of Transformer hidden states
and are independent of any specific model architecture or size.

Key insight: Transformer residual connections (h' = h + inject) create an affine space
structure in the hidden representation space. The three formulas exploit this structure
to achieve precise read-navigate-control capability.

References:
    - S_SPACE_FORMULAS.md: Full mathematical derivation
    - S_SPACE_CONTROLLABILITY.md: Controllability proof and experiments
"""

import torch
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Formula ③: Coordinate Reading — c_k = h · ê_k
# ═══════════════════════════════════════════════════════════════

def read_coords(
    h: torch.Tensor,
    principal_dirs: torch.Tensor,
) -> torch.Tensor:
    """Read K-dimensional coordinates from hidden state h.

    Formula ③: c_k = h · ê_k

    Projects hidden state onto principal directions to obtain coordinate
    readings. These coordinates represent the position in S-space.

    This is a linear projection — it works on ANY Transformer because
    it's just h · ê_k = matrix multiplication. No model-specific assumptions.

    Args:
        h: Hidden state vector, shape (d_model,)
        principal_dirs: Principal directions ê_k, shape (K, d_model)

    Returns:
        Coordinate vector c_k, shape (K,)

    Example:
        >>> coords = read_coords(h_l19, pca_dirs)  # c_k = h · ê_k
    """
    return principal_dirs @ h.float()


# ═══════════════════════════════════════════════════════════════
# Formula ②: Navigation — Δ_k = need_k × d_k
# ═══════════════════════════════════════════════════════════════

def compute_delta(
    coords: torch.Tensor,
    d_consensus: torch.Tensor,
    d_magnitude: torch.Tensor,
    d_confidence: torch.Tensor,
    metric_weights: torch.Tensor,
    need_saturation_scale: float = 5.0,
    need_floor: float = 0.05,
    snr_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute navigation displacement in coordinate space.

    Formula ②: Δ_k = need_k × d_k

    need_k is derived from current coordinates — it measures how far
    the current position is from the target direction. This makes
    the navigation input-adaptive (no training required).

    d_k = d_consensus × d_magnitude × d_confidence encodes the
    universal reasoning direction and its reliability per dimension.

    Args:
        coords: Current coordinates c_k, shape (K,)
        d_consensus: Consensus reasoning direction, shape (K,)
        d_magnitude: Direction magnitude, shape (K,)
        d_confidence: Direction confidence, shape (K,)
        metric_weights: Metric tensor diagonal g_k, shape (K,)
        need_saturation_scale: Saturation scaling factor
        need_floor: Minimum need value
        snr_weight: SNR gating weight (0-1)

    Returns:
        Tuple of (delta_k, need_k), each shape (K,)

    Example:
        >>> delta_k, need_k = compute_delta(coords, dc, dm, dco, mw)
    """
    has_evidence = d_consensus.abs() > 1e-6

    need = torch.zeros_like(coords)

    if has_evidence.any():
        dc_e = d_consensus[has_evidence]
        dm_e = d_magnitude[has_evidence]
        dco_e = d_confidence[has_evidence]
        mw_e = metric_weights[has_evidence] if metric_weights.shape[0] >= has_evidence.shape[0] else torch.ones(has_evidence.sum())
        coords_e = coords[has_evidence]

        # Alignment: how well current coords align with consensus direction
        alignment = dc_e * coords_e * mw_e * dco_e
        total_signal = (mw_e * coords_e.abs()).sum() + 1e-8
        saturation = alignment.clamp(min=0) / total_signal

        # Need = 1 - saturation (unsatisfied need)
        need_e = (1.0 - saturation * need_saturation_scale).clamp(
            min=need_floor, max=1.0
        )

        # Anti-alignment boost (pushing against consensus = more need)
        anti = (alignment < 0).float()
        need_e = (need_e + anti * 0.3).clamp(max=1.0)

        # SNR gating
        need_e = need_e * snr_weight

        need[has_evidence] = need_e

    # d_k = consensus × magnitude × confidence
    d_k = d_consensus * d_magnitude * d_confidence

    # Δ_k = need_k × d_k
    delta_k = need * d_k

    return delta_k, need


# ═══════════════════════════════════════════════════════════════
# Formula ①: Magnitude Control — α = r × |h| / |Δ_masked|
# ═══════════════════════════════════════════════════════════════

def compute_injection(
    h: torch.Tensor,
    delta_k: torch.Tensor,
    principal_dirs: torch.Tensor,
    metric_weights: torch.Tensor,
    r_eff: float,
    lottery_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Compute injection vector with precise magnitude control.

    Formula ①: α = r × |h| / |Δ_masked|

    Key property: cos(Δh, inject) = 1.0000 — the model offers zero
    resistance to injection. Push exactly as much as you want, get
    exactly that much.

    The formula guarantees that |inject| / |h| = r_eff, i.e., the
    injection magnitude is exactly r_eff times the hidden state norm.
    This is the ONLY steering method with a quantitative magnitude formula.

    Why this works: Transformer residual connections (h' = h + inject)
    create a linear injection pathway. The formula exploits this linearity.

    This is architecture-agnostic — any Transformer with residual
    connections has this property.

    Args:
        h: Hidden state vector, shape (d_model,)
        delta_k: Navigation displacement in K-space, shape (K,)
        principal_dirs: Principal directions ê_k, shape (K, d_model)
        metric_weights: Metric tensor diagonal g_k, shape (K,)
        r_eff: Effective injection ratio (0.0 to ~0.3)
        lottery_mask: Optional mask for lottery dimensions, shape (d_model,)

    Returns:
        Injection vector of shape (d_model,), or None if displacement is zero

    Example:
        >>> injection = compute_injection(h, delta_k, dirs, mw, r_eff=0.10, lottery_mask=mask)
        >>> h_new = h + injection  # cos(Δh, inject) ≈ 1.0
    """
    K_act = min(delta_k.shape[0], principal_dirs.shape[0])

    # Map K-dimensional displacement back to d_model space
    displacement = (
        metric_weights[:K_act].unsqueeze(1)
        * delta_k[:K_act].unsqueeze(1)
        * principal_dirs[:K_act]
    ).sum(dim=0)

    # Apply lottery mask (Law 1: only inject into lottery dimensions)
    if lottery_mask is not None:
        displacement_masked = displacement * lottery_mask
    else:
        displacement_masked = displacement

    dm_norm = displacement_masked.norm().item()
    if dm_norm < 1e-8:
        return None

    # Formula ①: α = r × |h| / |Δ_masked|
    h_norm = h.float().norm().item()
    alpha = r_eff * h_norm / dm_norm

    return displacement_masked * alpha


def compute_injection_from_coords(
    h: torch.Tensor,
    coords: torch.Tensor,
    principal_dirs: torch.Tensor,
    metric_weights: torch.Tensor,
    d_consensus: torch.Tensor,
    d_magnitude: torch.Tensor,
    d_confidence: torch.Tensor,
    r_eff: float = 0.10,
    lottery_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """End-to-end: coords → delta → injection (Formula ② + ①).

    Convenience function that combines compute_delta and compute_injection.

    Args:
        h: Hidden state vector, shape (d_model,)
        coords: Current coordinates, shape (K,)
        principal_dirs: ê_k, shape (K, d_model)
        metric_weights: g_k, shape (K,)
        d_consensus: Consensus direction, shape (K,)
        d_magnitude: Direction magnitude, shape (K,)
        d_confidence: Direction confidence, shape (K,)
        r_eff: Effective injection ratio
        lottery_mask: Optional lottery dimension mask, shape (d_model,)

    Returns:
        Tuple of (injection_vector, delta_k, need_k)
    """
    delta_k, need_k = compute_delta(
        coords, d_consensus, d_magnitude, d_confidence,
        metric_weights, **kwargs
    )
    injection = compute_injection(
        h, delta_k, principal_dirs, metric_weights,
        r_eff, lottery_mask
    )
    return injection, delta_k, need_k
