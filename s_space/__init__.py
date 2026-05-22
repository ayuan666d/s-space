"""
S-Space Navigation: Read, Navigate, and Control Transformer Internal Representations

Three formulas drive everything:

    ③ c_k = h · ê_k           — Locate: read K-dimensional coordinates
    ② Δ_k = need_k × d_k      — Navigate: compute displacement to target
    ① α = r × |h| / |Δ_masked| — Control: precise injection magnitude

Workflow:
    input → forward → h(L) → ③ read coords → ② compute nav → ① control magnitude → inject → generate
"""

from s_space.formulas import read_coords, compute_delta, compute_injection
from s_space.navigator import CoordNavigator
from s_space.space import SSpace, MetricTensor, LayerExpansionLaw

__version__ = "0.1.0"
__all__ = [
    "read_coords",
    "compute_delta",
    "compute_injection",
    "CoordNavigator",
    "SSpace",
    "MetricTensor",
    "LayerExpansionLaw",
]
