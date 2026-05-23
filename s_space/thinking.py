"""
S-Space Thinking Mode Control — Chain-of-Thought Steering

Provides high-level API for controlling thinking/chain-of-thought behavior
in models that support it (e.g., Qwen3.5-2B with thinking mode).

Key discovery (2B axis semantics):
    - ê₁ negative direction: skip thinking, direct output
    - ê₅ positive direction: skip thinking, direct output
    - ê₈ negative direction: skip thinking, direct output
    - ê₁₀ negative direction: skip thinking, direct output

These "thinking knobs" allow S-Space to control whether the model
engages in internal reasoning (thinking chain) or produces direct
output — a capability not reported by any other steering method.

Usage:
    >>> from s_space.thinking import ThinkingController
    >>> ctrl = ThinkingController.for_2b(device='cuda')
    >>>
    >>> # Skip thinking — get direct Chinese output
    >>> ctrl.skip_thinking(navigator)
    >>>
    >>> # Enable thinking — restore reasoning chain
    >>> ctrl.enable_thinking(navigator)
    >>>
    >>> # Partial control — reduce thinking intensity
    >>> ctrl.set_thinking_intensity(navigator, intensity=0.3)
"""

import json
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
# Thinking Mode Axis Registry
# ═══════════════════════════════════════════════════════════════

# Each entry: (axis_index, direction) where direction > 0 means push
# positive, < 0 means push negative. Pushing in these directions
# causes the model to skip its internal thinking chain.
THINKING_SKIP_AXES_2B = {
    # axis: direction → effect: skip thinking, direct output
    1:  -1,   # ê₁ negative → skip thinking
    5:  +1,   # ê₅ positive → skip thinking
    8:  -1,   # ê₈ negative → skip thinking
    10: -1,   # ê₁₀ negative → skip thinking
}

# Default layers for thinking control (based on 2B verification)
# Mid-to-late layers have the strongest thinking mode signal
THINKING_LAYERS_2B = [15, 19, 23]

# Data directory
_DATA_DIR = Path(__file__).parent.parent / "data"


class ThinkingController:
    """High-level controller for thinking/chain-of-thought steering.

    Provides three modes:
        - skip_thinking(): Force model to produce direct output
          (no <think/> chain), typically boosting Chinese output ratio
          from 0-2% to 83-87%.
        - enable_thinking(): Restore model's default thinking chain.
        - set_thinking_intensity(): Fine-grained control (0.0 = full
          thinking, 1.0 = skip thinking).

    Args:
        skip_axes: Dict of {axis_index: direction} for skipping thinking
        enable_axes: Dict of {axis_index: direction} for enabling thinking
        target_layers: Layer indices to apply control
    """

    def __init__(
        self,
        skip_axes: Dict[int, int],
        target_layers: list,
    ):
        self.skip_axes = skip_axes
        self.target_layers = target_layers

        # enable_axes = opposite directions
        self.enable_axes = {k: -v for k, v in skip_axes.items()}

    @classmethod
    def for_2b(cls, target_layers: Optional[list] = None) -> "ThinkingController":
        """Create controller pre-configured for Qwen3.5-2B.

        Args:
            target_layers: Override default target layers [15, 19, 23]

        Returns:
            ThinkingController configured for 2B thinking mode axes
        """
        return cls(
            skip_axes=dict(THINKING_SKIP_AXES_2B),
            target_layers=target_layers or THINKING_LAYERS_2B,
        )

    def skip_thinking(
        self,
        principal_dirs: Dict[int, torch.Tensor],
        metric_weights: Dict[int, torch.Tensor],
        K: Optional[int] = None,
        strength: float = 1.0,
    ) -> Dict[int, torch.Tensor]:
        """Generate injection vectors to skip thinking chain.

        Pushes the skip_axes directions, causing the model to bypass
        its internal reasoning and produce direct output.

        Args:
            principal_dirs: PCA principal directions {layer: (K_L, d_model)}
            metric_weights: Metric tensor weights {layer: (K_L,)}
            K: Deprecated — per-layer K is auto-detected from tensor shapes
            strength: Injection strength (0.0–1.0)

        Returns:
            Dict of injection vectors {layer: (d_model,)}
        """
        return self._compute_thinking_injection(
            principal_dirs, metric_weights, K,
            self.skip_axes, strength,
        )

    def enable_thinking(
        self,
        principal_dirs: Dict[int, torch.Tensor],
        metric_weights: Dict[int, torch.Tensor],
        K: Optional[int] = None,
        strength: float = 1.0,
    ) -> Dict[int, torch.Tensor]:
        """Generate injection vectors to enable thinking chain.

        Pushes the opposite of skip_axes directions, reinforcing
        the model's default thinking behavior.

        Args:
            principal_dirs: PCA principal directions {layer: (K_L, d_model)}
            metric_weights: Metric tensor weights {layer: (K_L,)}
            K: Deprecated — per-layer K is auto-detected from tensor shapes
            strength: Injection strength (0.0–1.0)

        Returns:
            Dict of injection vectors {layer: (d_model,)}
        """
        return self._compute_thinking_injection(
            principal_dirs, metric_weights, K,
            self.enable_axes, strength,
        )

    def set_thinking_intensity(
        self,
        principal_dirs: Dict[int, torch.Tensor],
        metric_weights: Dict[int, torch.Tensor],
        K: Optional[int] = None,
        intensity: float = 0.5,
    ) -> Dict[int, torch.Tensor]:
        """Fine-grained thinking control.

        Args:
            principal_dirs: PCA principal directions {layer: (K_L, d_model)}
            metric_weights: Metric tensor weights {layer: (K_L,)}
            K: Deprecated — per-layer K is auto-detected from tensor shapes
            intensity: 0.0 = full thinking, 1.0 = skip thinking

        Returns:
            Dict of injection vectors {layer: (d_model,)}
        """
        return self._compute_thinking_injection(
            principal_dirs, metric_weights, K,
            self.skip_axes, intensity,
        )

    def _compute_thinking_injection(
        self,
        principal_dirs: Dict[int, torch.Tensor],
        metric_weights: Dict[int, torch.Tensor],
        K: int,
        axes: Dict[int, int],
        strength: float,
    ) -> Dict[int, torch.Tensor]:
        """Core injection computation for thinking mode.

        For each target layer, constructs an injection vector by
        combining the specified axis directions weighted by their
        metric importance and the desired strength.

        The injection is: inject = α × Σ(g_k × direction_k × ê_k)
        where α controls magnitude via Formula ①.
        """
        injections = {}

        for L in self.target_layers:
            if L not in principal_dirs:
                continue

            dirs = principal_dirs[L]    # (K_L, d_model)
            weights = metric_weights[L]  # (K_L,)
            K_L = dirs.shape[0]          # per-layer actual K

            # Build displacement in K-space
            delta_k = torch.zeros(K_L)

            for axis_idx, direction in axes.items():
                if axis_idx < K_L:
                    # Weight by metric importance of this axis
                    delta_k[axis_idx] = direction * weights[axis_idx].item()

            # Map K-space displacement to d_model space
            # inject = Σ_k (g_k × delta_k × ê_k)
            displacement = (
                weights[:K_L].unsqueeze(1)
                * delta_k[:K_L].unsqueeze(1)
                * dirs[:K_L]
            ).sum(dim=0)

            # Scale by strength
            displacement = displacement * strength

            if displacement.norm() > 1e-8:
                injections[L] = displacement

        return injections

    @staticmethod
    def load_axis_semantics(
        path: Optional[str] = None,
    ) -> Dict:
        """Load axis semantic mapping from JSON file.

        Args:
            path: Path to axis_semantics JSON. Defaults to 2B data.

        Returns:
            Dict with axis semantic information
        """
        if path is None:
            path = _DATA_DIR / "axis_semantics_2B.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_thinking_axes(path: Optional[str] = None) -> Dict[int, int]:
        """Auto-detect thinking control axes from axis semantics.

        Scans axis semantic data for axes where positive or negative
        direction causes the model to skip thinking (direct output
        without <think/> tags).

        Args:
            path: Path to axis_semantics JSON

        Returns:
            Dict {axis_index: direction} for skip-thinking control
        """
        data = ThinkingController.load_axis_semantics(path)
        skip_axes = {}

        for key, entry in data.items():
            axis = entry.get("axis", int(key.split("_")[1]))
            pos_output = entry.get("positive_output", "")
            neg_output = entry.get("negative_output", "")
            baseline = entry.get("baseline", "")

            # Detect skip: output has short/empty thinking block followed
            # by direct output. Pattern: <think...>\n\n</think\n\n or
            # the output doesn't start with <think at all.
            baseline_has_think = baseline.strip().startswith("<think")

            def _has_short_think(text):
                """Check if thinking block is trivially short (skip thinking)."""
                if not text.strip().startswith("<think"):
                    return True  # no thinking at all
                # Check if </think appears very early (within first 100 chars)
                # meaning the thinking chain was skipped
                early_text = text[:200]
                return "</think" in early_text

            pos_skip = _has_short_think(pos_output) if pos_output else False
            neg_skip = _has_short_think(neg_output) if neg_output else False

            if pos_skip and baseline_has_think:
                skip_axes[axis] = +1  # positive direction skips thinking
            elif neg_skip and baseline_has_think:
                skip_axes[axis] = -1  # negative direction skips thinking

        return skip_axes
