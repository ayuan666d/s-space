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
    metric_weights: torch.Tensor,
    d_consensus: Optional[torch.Tensor] = None,
    d_magnitude: Optional[torch.Tensor] = None,
    d_confidence: Optional[torch.Tensor] = None,
    need_saturation_scale: float = 5.0,
    need_floor: float = 0.05,
    snr_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute navigation displacement in coordinate space.

    Formula ②: Δ_k = need_k × d_k

    Supports two navigation modes:

    1. **Inertia mode** (default, model-agnostic):
       d_k = iw_k × c_k, where iw_k = -log(g_k / g_max)
       Direction comes entirely from current coordinates.
       Works on ANY Transformer — no pre-extracted consensus data needed.

    2. **Consensus mode** (optional, requires pre-extracted data):
       d_k = d_consensus × d_magnitude × d_confidence
       Uses empirically extracted reasoning directions for enhanced
       navigation on models where such data is available.

    need_k is derived from current coordinates — it measures how far
    the current position is from the target direction. This makes
    the navigation input-adaptive (no training required).

    Args:
        coords: Current coordinates c_k, shape (K,)
        metric_weights: Metric tensor diagonal g_k, shape (K,)
        d_consensus: Optional consensus reasoning direction, shape (K,)
        d_magnitude: Optional direction magnitude, shape (K,)
        d_confidence: Optional direction confidence, shape (K,)
        need_saturation_scale: Saturation scaling factor
        need_floor: Minimum need value
        snr_weight: SNR gating weight (0-1)

    Returns:
        Tuple of (delta_k, need_k), each shape (K,)

    Example:
        >>> # Inertia mode — works on any model
        >>> delta_k, need_k = compute_delta(coords, metric_weights)
        >>> # Consensus mode — enhanced navigation with pre-extracted data
        >>> delta_k, need_k = compute_delta(coords, mw, dc, dm, dco)
    """
    K = coords.shape[0]

    # ── Compute need_k from coordinates and metric weights ──
    # need_k = 1 - (|c_k| × g_k) / Σ(|c_k| × g_k)
    # "Saturated axes don't need pushing"
    weighted_signal = coords.abs() * metric_weights[:K]
    total_signal = weighted_signal.sum() + 1e-8
    saturation = weighted_signal / total_signal
    need = (1.0 - saturation * need_saturation_scale).clamp(
        min=need_floor, max=1.0
    )

    # Anti-alignment boost: if c_k is negative (pushing against metric),
    # increase need — this axis really needs correction
    anti = (coords * metric_weights[:K] < 0).float()
    need = (need + anti * 0.3).clamp(max=1.0)

    # SNR gating
    need = need * snr_weight

    # ── Compute d_k (navigation direction) ──
    use_consensus = (
        d_consensus is not None
        and d_magnitude is not None
        and d_confidence is not None
        and (d_consensus.abs() > 1e-6).any()
    )

    if use_consensus:
        # Consensus mode: d_k = consensus × magnitude × confidence
        d_k = d_consensus * d_magnitude * d_confidence
    else:
        # Inertia mode: d_k = iw_k × c_k (V4 pure inertia)
        # iw_k = -log(g_k / g_max): inverse metric weight as inertia
        # Low g_k → high iw_k → axis resists change (heavy = inertial)
        # High g_k → low iw_k → axis easy to shift
        # But we negate: we want to push ALONG the current coordinate
        # direction (inertia = follow what's already happening)
        g_max = metric_weights[:K].max() + 1e-8
        iw_k = -torch.log(metric_weights[:K] / g_max + 1e-8)  # (K,)
        d_k = iw_k * coords  # follow current direction, weighted by inertia

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
    injection_mask: Optional[torch.Tensor] = None,
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
        injection_mask: Optional dimension mask, shape (d_model,)

    Returns:
        Injection vector of shape (d_model,), or None if displacement is zero

    Example:
        >>> injection = compute_injection(h, delta_k, dirs, mw, r_eff=0.10)
        >>> h_new = h + injection  # cos(Δh, inject) ≈ 1.0
    """
    K_act = min(delta_k.shape[0], principal_dirs.shape[0])

    # Map K-dimensional displacement back to d_model space
    displacement = (
        metric_weights[:K_act].unsqueeze(1)
        * delta_k[:K_act].unsqueeze(1)
        * principal_dirs[:K_act]
    ).sum(dim=0)

    # Apply injection mask if provided
    if injection_mask is not None:
        displacement_masked = displacement * injection_mask
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
    d_consensus: Optional[torch.Tensor] = None,
    d_magnitude: Optional[torch.Tensor] = None,
    d_confidence: Optional[torch.Tensor] = None,
    r_eff: float = 0.10,
    injection_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """End-to-end: coords → delta → injection (Formula ② + ①).

    Convenience function that combines compute_delta and compute_injection.
    Works in both inertia mode (no consensus data) and consensus mode.

    Args:
        h: Hidden state vector, shape (d_model,)
        coords: Current coordinates, shape (K,)
        principal_dirs: ê_k, shape (K, d_model)
        metric_weights: g_k, shape (K,)
        d_consensus: Optional consensus direction, shape (K,)
        d_magnitude: Optional direction magnitude, shape (K,)
        d_confidence: Optional direction confidence, shape (K,)
        r_eff: Effective injection ratio
        injection_mask: Optional dimension mask, shape (d_model,)

    Returns:
        Tuple of (injection_vector, delta_k, need_k)
    """
    delta_k, need_k = compute_delta(
        coords, metric_weights, d_consensus, d_magnitude, d_confidence,
        **kwargs
    )
    injection = compute_injection(
        h, delta_k, principal_dirs, metric_weights,
        r_eff, injection_mask
    )
    return injection, delta_k, need_k
