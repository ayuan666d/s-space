"""
S-Space Pre-tuned Data Pack for Qwen 0.8B

This module provides easy access to pre-extracted S-space parameters
for the Qwen2.5-0.8B model. Users can directly use these without
running their own PCA extraction.

The data files are stored in the project's data/ directory.
This module automatically resolves the correct path whether installed
via pip or used from the source repository.

Usage:
    from s_space.pretuned import load_08b_navigator

    nav = load_08b_navigator()
    result = nav.navigate(hidden_states, base_r=0.10)

Or with consensus enhancement:
    from s_space.pretuned import load_08b_navigator

    nav = load_08b_navigator(consensus=True)
    result = nav.navigate(hidden_states, base_r=0.10)

Available data:
    - PCA parameters (ê_k, g_k) for 21 layers, K=57
    - Reasoning consensus directions (d_consensus) for 7 layers
    - Thinking direction vectors for injection masking
    - Pure route vectors for navigation
"""

import torch
from pathlib import Path
from typing import Optional

from s_space.navigator import CoordNavigator

# Pre-tuned data directory
PRETUNED_DIR = Path(__file__).parent / "pretuned"

# File mapping
FILES = {
    "pca_params": "coord_nav_params_K100.pt",
    "consensus": "reasoning_consensus.pt",
    "thinking_dirs": "thinking_dirs.pt",
    "pure_route": "pure_route_v2.pt",
    "type_classifier": "cer_type_classifier.pt",
}


def _resolve_path(key: str) -> Optional[str]:
    """Resolve file path for pre-tuned data."""
    fname = FILES.get(key)
    if fname is None:
        return None
    path = PRETUNED_DIR / fname
    if path.exists():
        return str(path)
    # Also check data/ directory (for development)
    alt = Path(__file__).parent.parent / "data" / fname
    if alt.exists():
        return str(alt)
    return None


def load_08b_navigator(
    consensus: bool = True,
    inject_layers: Optional[list] = None,
) -> CoordNavigator:
    """Load pre-tuned CoordNavigator for Qwen 0.8B.

    Args:
        consensus: Whether to load consensus directions (default: True)
        inject_layers: Override auto-detected injection layers

    Returns:
        CoordNavigator ready to use with Qwen 0.8B hidden states
    """
    pca_path = _resolve_path("pca_params")
    if pca_path is None:
        raise FileNotFoundError(
            "Pre-tuned PCA params not found. Place coord_nav_params_K100.pt "
            "in s_space/pretuned/ or data/ directory."
        )

    consensus_path = None
    if consensus:
        consensus_path = _resolve_path("consensus")

    nav = CoordNavigator(
        params_path=pca_path,
        consensus_path=consensus_path,
        inject_layers=inject_layers,
    )

    return nav


def load_08b_params(key: str = "pca_params") -> dict:
    """Load a specific pre-tuned data file.

    Args:
        key: One of 'pca_params', 'consensus', 'thinking_dirs',
             'pure_route', 'type_classifier'

    Returns:
        Loaded dict/tensor data
    """
    path = _resolve_path(key)
    if path is None:
        raise FileNotFoundError(f"Pre-tuned data '{key}' not found")
    return torch.load(path, map_location='cpu', weights_only=False)


def list_available() -> dict:
    """List available pre-tuned data files and their status.

    Returns:
        Dict mapping key -> {path, exists, size_mb}
    """
    result = {}
    for key, fname in FILES.items():
        path = _resolve_path(key)
        if path:
            size = Path(path).stat().st_size / 1024 / 1024
            result[key] = {"path": path, "exists": True, "size_mb": round(size, 2)}
        else:
            result[key] = {"path": None, "exists": False, "size_mb": 0}
    return result
