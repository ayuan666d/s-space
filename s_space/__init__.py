"""
S-Space Navigation: Read, Navigate, and Control Transformer Internal Representations

Three formulas drive everything:

    ③ c_k = h · ê_k           — Locate: read K-dimensional coordinates
    ② Δ_k = need_k × d_k      — Navigate: compute displacement to target
    ① α = r × |h| / |Δ_masked| — Control: precise injection magnitude

Navigation modes:
    - Inertia mode (default): d_k = iw_k × c_k — works on any Transformer
    - Consensus mode (optional): d_k = d_consensus × d_magnitude × d_confidence

Workflow:
    input → forward → h(L) → ③ read coords → ② compute nav → ① control magnitude → inject → generate
"""

from s_space.formulas import read_coords, compute_delta, compute_injection, compute_injection_from_coords
from s_space.navigator import CoordNavigator
from s_space.space import SSpace, MetricTensor, LayerExpansionLaw
from s_space.injection_mask import mask_from_metric_weights, mask_from_axes, no_mask

__version__ = "0.2.0"
__all__ = [
    "read_coords",
    "compute_delta",
    "compute_injection",
    "compute_injection_from_coords",
    "CoordNavigator",
    "SSpace",
    "MetricTensor",
    "LayerExpansionLaw",
    "mask_from_metric_weights",
    "mask_from_axes",
    "no_mask",
]
