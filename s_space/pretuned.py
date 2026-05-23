"""
S-Space Pre-tuned Data Packs

Provides easy access to pre-extracted S-Space parameters for supported
models. Users can directly use these without running their own PCA extraction.

The data files are stored in the project's data/ directory.
This module automatically resolves the correct path whether installed
via pip or used from the source repository.

Supported models:
    - Qwen3.5-0.8B: K=57, 21 layers
    - Qwen3.5-2B:   K=31, 24 layers (with thinking mode control)

Usage:
    # Auto-load by model name
    from s_space.pretuned import load_navigator
    nav = load_navigator("2b")
    result = nav.navigate(hidden_states, base_r=0.10)

    # Or use specific loaders
    from s_space.pretuned import load_08b_navigator, load_2b_navigator
    nav = load_2b_navigator(thinking_mode="skip")

    # List available data
    from s_space.pretuned import list_available
    print(list_available())
"""

import torch
from pathlib import Path
from typing import Dict, Optional

from s_space.navigator import CoordNavigator

# ═══════════════════════════════════════════════════════════════
# Data directory resolution
# ═══════════════════════════════════════════════════════════════

_DATA_DIR = Path(__file__).parent.parent / "data"

# ═══════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "0.8b": {
        "aliases": ["0.8b", "08b", "0.8", "qwen3.5-0.8b"],
        "pca_params": "coord_nav_params_K100.pt",
        "consensus": "reasoning_consensus.pt",
        "thinking_dirs": "thinking_dirs.pt",
        "extras": {
            "pure_route": "pure_route_v2.pt",
            "type_classifier": "cer_type_classifier.pt",
        },
        "description": "Qwen3.5-0.8B — K=57, 21 layers",
    },
    "2b": {
        "aliases": ["2b", "2.0b", "qwen3.5-2b"],
        "pca_params": "coord_nav_params_2B_K100.pt",
        "consensus": "reasoning_consensus_2B.pt",
        "thinking_dirs": "thinking_dirs_2B.pt",
        "extras": {
            "axis_semantics": "axis_semantics_2B.json",
            "type_centroids": "type_centroids_2B.pt",
            "novel_verification": "novel_verification_2B.json",
        },
        "description": "Qwen3.5-2B — K=31, 24 layers, thinking mode control",
    },
}


def _resolve_model_key(model: str) -> str:
    """Resolve model alias to canonical key."""
    model_lower = model.lower().strip()
    for key, info in MODEL_REGISTRY.items():
        if model_lower in info["aliases"] or model_lower == key:
            return key
    raise ValueError(
        f"Unknown model '{model}'. Available: "
        f"{list(MODEL_REGISTRY.keys())}. "
        f"Aliases: {[a for info in MODEL_REGISTRY.values() for a in info['aliases']]}"
    )


def _resolve_file(filename: str) -> Optional[str]:
    """Resolve file path in data/ directory."""
    path = _DATA_DIR / filename
    if path.exists():
        return str(path)
    return None


# ═══════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════

def load_navigator(
    model: str = "0.8b",
    consensus: bool = True,
    inject_layers: Optional[list] = None,
    device: str = 'cpu',
) -> CoordNavigator:
    """Load pre-tuned CoordNavigator for a supported model.

    This is the recommended entry point. It auto-resolves the model
    name and loads all available data.

    Args:
        model: Model identifier — "0.8b", "2b", or aliases
            (e.g., "qwen3.5-2b")
        consensus: Whether to load consensus directions (default: True)
        inject_layers: Override auto-detected injection layers
        device: Device to load data onto

    Returns:
        CoordNavigator ready to use with the specified model

    Example:
        >>> nav = load_navigator("2b")
        >>> result = nav.navigate(hidden_states, base_r=0.10)
    """
    key = _resolve_model_key(model)
    info = MODEL_REGISTRY[key]

    pca_path = _resolve_file(info["pca_params"])
    if pca_path is None:
        raise FileNotFoundError(
            f"Pre-tuned PCA params for {key} not found. "
            f"Expected: {info['pca_params']} in data/ directory."
        )

    consensus_path = None
    if consensus:
        consensus_path = _resolve_file(info["consensus"])

    nav = CoordNavigator(
        params_path=pca_path,
        consensus_path=consensus_path,
        inject_layers=inject_layers,
        device=device,
    )

    return nav


def load_08b_navigator(
    consensus: bool = True,
    inject_layers: Optional[list] = None,
    device: str = 'cpu',
) -> CoordNavigator:
    """Load pre-tuned CoordNavigator for Qwen3.5-0.8B.

    Backward-compatible shortcut for load_navigator("0.8b").

    Args:
        consensus: Whether to load consensus directions
        inject_layers: Override auto-detected injection layers
        device: Device to load data onto

    Returns:
        CoordNavigator for Qwen3.5-0.8B
    """
    return load_navigator("0.8b", consensus=consensus,
                          inject_layers=inject_layers, device=device)


def load_2b_navigator(
    consensus: bool = True,
    thinking_mode: Optional[str] = None,
    inject_layers: Optional[list] = None,
    device: str = 'cpu',
) -> CoordNavigator:
    """Load pre-tuned CoordNavigator for Qwen3.5-2B.

    Enhanced loader with thinking mode control support.

    Args:
        consensus: Whether to load consensus directions
        thinking_mode: Thinking chain control —
            None: default (no thinking control)
            "skip": skip thinking chain, direct output
            "enable": enforce thinking chain
        inject_layers: Override auto-detected injection layers
        device: Device to load data onto

    Returns:
        CoordNavigator for Qwen3.5-2B, optionally with thinking control

    Example:
        >>> # Skip thinking — direct Chinese output (0% → 87%)
        >>> nav = load_2b_navigator(thinking_mode="skip")
        >>>
        >>> # Normal mode with consensus
        >>> nav = load_2b_navigator(consensus=True)
    """
    nav = load_navigator("2b", consensus=consensus,
                         inject_layers=inject_layers, device=device)

    # Apply thinking mode if requested
    if thinking_mode is not None:
        from s_space.thinking import ThinkingController
        ctrl = ThinkingController.for_2b()
        # Store controller on navigator for use during generation
        nav._thinking_controller = ctrl
        nav._thinking_mode = thinking_mode

    return nav


def load_thinking_controller(model: str = "2b") -> "ThinkingController":
    """Load ThinkingController for a supported model.

    Args:
        model: Model identifier (currently only "2b" supported)

    Returns:
        ThinkingController instance

    Example:
        >>> ctrl = load_thinking_controller("2b")
        >>> injections = ctrl.skip_thinking(nav.principal_dirs, nav.metric_weights, nav.K)
    """
    key = _resolve_model_key(model)
    if key == "2b":
        from s_space.thinking import ThinkingController
        return ThinkingController.for_2b()
    else:
        raise ValueError(
            f"Thinking mode control is not available for model '{key}'. "
            f"Currently only supported for Qwen3.5-2B."
        )


def load_data(model: str = "0.8b", key: str = "pca_params") -> dict:
    """Load a specific pre-tuned data file for a model.

    Args:
        model: Model identifier
        key: Data key — 'pca_params', 'consensus', 'thinking_dirs',
             or model-specific keys like 'axis_semantics', 'type_centroids'

    Returns:
        Loaded dict/tensor data
    """
    mkey = _resolve_model_key(model)
    info = MODEL_REGISTRY[mkey]

    # Check core files
    if key == "pca_params":
        filename = info["pca_params"]
    elif key == "consensus":
        filename = info["consensus"]
    elif key == "thinking_dirs":
        filename = info["thinking_dirs"]
    elif key in info.get("extras", {}):
        filename = info["extras"][key]
    else:
        raise ValueError(
            f"Unknown data key '{key}' for model '{mkey}'. "
            f"Available: pca_params, consensus, thinking_dirs, "
            f"{list(info.get('extras', {}).keys())}"
        )

    path = _resolve_file(filename)
    if path is None:
        raise FileNotFoundError(
            f"Data file '{filename}' for {mkey} not found in data/ directory."
        )

    if filename.endswith(".json"):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return torch.load(path, map_location='cpu', weights_only=False)


def list_available() -> Dict:
    """List all available pre-tuned data files and their status.

    Returns:
        Dict mapping model -> {key: {path, exists, size_mb}}
    """
    result = {}
    for model_key, info in MODEL_REGISTRY.items():
        model_data = {}

        # Core files
        for key, filename in [
            ("pca_params", info["pca_params"]),
            ("consensus", info["consensus"]),
            ("thinking_dirs", info["thinking_dirs"]),
        ]:
            path = _resolve_file(filename)
            if path:
                size = Path(path).stat().st_size / 1024 / 1024
                model_data[key] = {"path": path, "exists": True,
                                   "size_mb": round(size, 2)}
            else:
                model_data[key] = {"path": None, "exists": False,
                                   "size_mb": 0}

        # Extra files
        for key, filename in info.get("extras", {}).items():
            path = _resolve_file(filename)
            if path:
                size = Path(path).stat().st_size / 1024 / 1024
                model_data[key] = {"path": path, "exists": True,
                                   "size_mb": round(size, 2)}
            else:
                model_data[key] = {"path": None, "exists": False,
                                   "size_mb": 0}

        result[model_key] = {
            "description": info["description"],
            "data": model_data,
        }

    return result
