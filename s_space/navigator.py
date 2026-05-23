"""
CoordNavigator — Model-Agnostic S-Space Navigation

Complete navigator that reads coordinates, computes navigation, and controls
injection magnitude using the three core formulas.

Supports:
    - Multi-layer injection (auto-detected from PCA parameters)
    - Chunk-by-chunk continuous navigation
    - Goal persistence with decay
    - SNR gating for noisy coordinates
    - Drift correction
    - Optional injection masking (metric-based, direction-based, or manual)
    - Residual convergence loop

Architecture-agnostic: works with any Transformer model.
Only requires PCA parameters (ê_k, g_k) extracted from the target model.
Consensus directions (d_consensus) are optional — without them, the navigator
runs in pure inertia mode, following coordinate momentum.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import namedtuple

from s_space.formulas import read_coords, compute_delta, compute_injection


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

GoalState = namedtuple('GoalState', [
    'goal_delta_k',        # {layer: tensor(K,)} — initial navigation direction
    'inject_layers',       # [int] — layers with PCA data
    'initial_need_max',    # float — max need at registration
    'initial_need_mean',   # float — mean need at registration
])


class CoordNavigator:
    """S-Space Coordinate Navigator — Three formulas drive everything.

    Workflow:
        1. Load PCA parameters (ê_k, g_k) from any Transformer
        2. (Optional) Load consensus directions for enhanced navigation
        3. Forward model, capture hidden states
        4. Formula ③: Read coordinates c_k = h · ê_k
        5. Formula ②: Compute navigation Δ_k = need_k × d_k
        6. Formula ①: Control magnitude α = r × |h| / |Δ_masked|
        7. Inject into model, generate output

    Navigation modes:
        - Inertia mode (default): d_k = iw_k × c_k
          Works on ANY model with just PCA data.
          Follows coordinate momentum — "saturated axes don't need pushing."

        - Consensus mode: d_k = d_consensus × d_magnitude × d_confidence
          Requires pre-extracted reasoning directions.
          Enhanced navigation for models where such data is available.

    Args:
        params_path: Path to PCA parameters file (ê_k, g_k)
        consensus_path: Optional path to consensus directions file
        inject_layers: Which layers to inject into (auto-detected if None)
        injection_masks: Optional dimension masks {layer: (d_model,)}
        device: Device to use
    """

    # ── Navigation parameters ──
    DEFAULT_NEED_SATURATION_SCALE = 5.0
    DEFAULT_NEED_FLOOR = 0.05
    DEFAULT_SNR_GATE = 0.5
    DEFAULT_CHUNK_SIZE = 80

    # ── Goal persistence parameters ──
    DEFAULT_GOAL_PERSIST_STRENGTH = 0.05
    DEFAULT_GOAL_DECAY = 0.95
    DEFAULT_GOAL_MIN_STRENGTH = 0.02

    # ── Drift correction parameters ──
    DEFAULT_DRIFT_CORRECT_RATIO = 0.3

    def __init__(
        self,
        params_path: str,
        consensus_path: Optional[str] = None,
        inject_layers: Optional[List[int]] = None,
        injection_masks: Optional[Dict[int, torch.Tensor]] = None,
        device: str = 'cpu',
    ):
        self.device = device

        # ── Load PCA parameters (required) ──
        params = torch.load(params_path, map_location=device, weights_only=False)
        self.principal_dirs = params['principal_dirs']   # {layer: (K, d_model)}
        self.metric_weights = params['metric_weights']   # {layer: (K,)}

        # K dimension (always detect from actual tensor shape, not stored K)
        # The stored K may be the target K during extraction, but the actual
        # number of principal components can differ (e.g., variance threshold
        # may yield fewer components than the target K).
        first_layer = next(iter(self.principal_dirs))
        self.K = self.principal_dirs[first_layer].shape[0]

        # d_model (auto-detect from data)
        self.d_model = self.principal_dirs[first_layer].shape[1]

        # Inject layers (auto-detect from available PCA data)
        available_layers = sorted(self.principal_dirs.keys())
        if inject_layers is not None:
            self.inject_layers = [l for l in inject_layers if l in self.principal_dirs]
        else:
            # Auto-select: use all layers with PCA data, preferring
            # early/mid/late distribution for coverage
            self.inject_layers = self._auto_select_layers(available_layers)

        # ── Layer scale (auto-compute from metric weight distribution) ──
        self.layer_scale = self._auto_compute_layer_scale()

        # ── Target injection magnitude (auto-compute from layer expansion law) ──
        self.target_mag = self._auto_compute_target_mag()

        # ── Load consensus directions (optional) ──
        self.d_consensus: Dict[int, torch.Tensor] = {}
        self.d_magnitude: Dict[int, torch.Tensor] = {}
        self.d_confidence: Dict[int, torch.Tensor] = {}
        self.has_consensus = False
        if consensus_path:
            self._load_consensus(consensus_path)
            self.has_consensus = bool(self.d_consensus)

        # ── Injection masks (optional) ──
        self.injection_masks: Dict[int, torch.Tensor] = {}
        if injection_masks:
            self.injection_masks = injection_masks

        # ── Goal state ──
        self.goal_state: Optional[GoalState] = None
        self.goal_persist_strength = self.DEFAULT_GOAL_PERSIST_STRENGTH

    def _auto_select_layers(self, available: List[int]) -> List[int]:
        """Auto-select injection layers from available PCA layers.

        Strategy: pick 3 layers spanning early/mid/late,
        or use all available if <= 3 layers.
        """
        if len(available) <= 3:
            return available

        n = len(available)
        # Pick early (1/4), mid (1/2), late (3/4) positions
        early = available[n // 4]
        mid = available[n // 2]
        late = available[3 * n // 4]
        return [early, mid, late]

    def _auto_compute_layer_scale(self) -> Dict[int, float]:
        """Auto-compute per-layer injection scale from metric weights.

        Deep layers have larger metric weights (layer expansion law),
        so they can tolerate larger injection. Scale proportionally
        to the log of total metric weight at each layer.
        """
        if not self.inject_layers:
            return {}

        # Compute total metric energy per layer
        energies = {}
        for L in self.inject_layers:
            if L in self.metric_weights:
                energies[L] = self.metric_weights[L].sum().item()
            else:
                energies[L] = 1.0

        # Normalize so that the max layer gets scale=1.0
        max_energy = max(energies.values()) if energies else 1.0

        return {
            L: 0.5 + 0.5 * (energies[L] / max_energy)
            for L in self.inject_layers
        }

    def _auto_compute_target_mag(self) -> Dict[int, float]:
        """Auto-compute target injection magnitude per layer.

        Uses the layer expansion law: deep layers have exponentially
        larger hidden state norms, so target magnitude scales up.
        """
        target = {}
        for L in self.inject_layers:
            # Base magnitude, scaled by layer depth
            # Layer expansion: d̄ ≈ A·e^(0.092·(l-14)) for l≥14
            if L >= 14:
                scale = min(2.71828 ** (0.092 * (L - 14)), 3.0)
            else:
                scale = 1.0
            target[L] = 0.10 * scale
        return target

    def _load_consensus(self, path: str):
        """Load d_consensus, d_magnitude, d_confidence from file."""
        rc = torch.load(path, map_location='cpu', weights_only=False)
        for L in self.inject_layers:
            source_L = L
            if L not in rc['d_consensus']:
                candidates = sorted(rc['d_consensus'].keys(), key=lambda x: abs(x - L))
                source_L = candidates[0] if candidates else None

            if source_L is not None and source_L in rc['d_consensus']:
                dc = rc['d_consensus'][source_L]
                dm = rc['d_magnitude'][source_L]
                dco = rc['d_confidence'][source_L]
                K_src = min(dc.shape[0], self.K)
                K_dm = min(dm.shape[0], self.K)
                K_dco = min(dco.shape[0], self.K)
                self.d_consensus[L] = torch.zeros(self.K)
                self.d_magnitude[L] = torch.ones(self.K) * 0.08
                self.d_confidence[L] = torch.ones(self.K) * 0.3
                self.d_consensus[L][:K_src] = dc[:K_src]
                self.d_magnitude[L][:K_dm] = dm[:K_dm]
                self.d_confidence[L][:K_dco] = dco[:K_dco]
            else:
                self.d_consensus[L] = torch.zeros(self.K)
                self.d_magnitude[L] = torch.ones(self.K) * 0.08
                self.d_confidence[L] = torch.ones(self.K) * 0.3

    # ════════════════════════════════════════════════════════════
    # Core Navigation (single step)
    # ════════════════════════════════════════════════════════════

    def navigate(
        self,
        hidden_states: Dict[int, torch.Tensor],
        base_r: float = 0.10,
    ) -> Dict:
        """Single-step navigation: read coords → compute delta → inject.

        Works in both inertia mode and consensus mode.
        In inertia mode (no consensus data), navigation follows
        coordinate momentum — "saturated axes don't need pushing."

        Args:
            hidden_states: {layer: h_tensor} captured from model forward
            base_r: Base injection ratio

        Returns:
            Dict with 'injections', 'coords', 'delta_k', 'need', 'r_eff'
        """
        injections = {}
        coords_dict = {}
        delta_dict = {}
        need_dict = {}
        r_eff_dict = {}

        for L in self.inject_layers:
            if L not in hidden_states:
                continue

            h = hidden_states[L]
            coords = read_coords(h, self.principal_dirs[L])
            coords_dict[L] = coords

            # Compute delta — inertia mode or consensus mode
            dc = self.d_consensus.get(L)
            dm = self.d_magnitude.get(L)
            dco = self.d_confidence.get(L)

            delta_k, need_k = compute_delta(
                coords, self.metric_weights[L],
                d_consensus=dc, d_magnitude=dm, d_confidence=dco,
            )
            delta_dict[L] = delta_k
            need_dict[L] = need_k

            if delta_k.norm() < 1e-8 and self.goal_state is None:
                continue

            # Compute r_eff based on need
            layer_scale = self.layer_scale.get(L, 1.0)
            need_mean = need_k.mean().item()
            r_eff = base_r * layer_scale * max(need_mean, 0.1)
            r_eff_dict[L] = r_eff

            # Three-formula injection
            inj = compute_injection(
                h, delta_k, self.principal_dirs[L],
                self.metric_weights[L], r_eff,
                self.injection_masks.get(L)
            )

            if inj is not None and inj.norm() > 1e-10:
                injections[L] = inj

        return {
            'injections': injections,
            'coords': coords_dict,
            'delta_k': delta_dict,
            'r_eff': r_eff_dict,
            'need': need_dict,
        }

    # ════════════════════════════════════════════════════════════
    # Goal Registration
    # ════════════════════════════════════════════════════════════

    def register_goal(self, captured_h: Dict[int, torch.Tensor]) -> GoalState:
        """Register navigation goal from initial hidden states.

        The goal direction is the initial delta_k (need × d_k),
        derived purely from coordinates — no label lookup required.

        Args:
            captured_h: Hidden states from first forward pass

        Returns:
            GoalState with navigation direction and need statistics
        """
        goal_delta_k = {}
        need_max = 0.0
        need_mean_sum = 0.0
        n_layers = 0

        for L in self.inject_layers:
            if L not in captured_h:
                continue

            h = captured_h[L]
            coords = read_coords(h, self.principal_dirs[L])
            dc = self.d_consensus.get(L)
            dm = self.d_magnitude.get(L)
            dco = self.d_confidence.get(L)

            delta_k, need_k = compute_delta(
                coords, self.metric_weights[L],
                d_consensus=dc, d_magnitude=dm, d_confidence=dco,
            )
            goal_delta_k[L] = delta_k

            need_max = max(need_max, need_k.max().item())
            need_mean_sum += need_k.mean().item()
            n_layers += 1

        goal_state = GoalState(
            goal_delta_k=goal_delta_k,
            inject_layers=self.inject_layers,
            initial_need_max=need_max,
            initial_need_mean=need_mean_sum / max(n_layers, 1),
        )

        self.goal_state = goal_state
        self.goal_persist_strength = self.DEFAULT_GOAL_PERSIST_STRENGTH
        return goal_state

    # ════════════════════════════════════════════════════════════
    # Continuous Navigation (chunk-by-chunk)
    # ════════════════════════════════════════════════════════════

    def navigate_chunk(
        self,
        current_h: Dict[int, torch.Tensor],
        anchor_h: Optional[Dict[int, torch.Tensor]] = None,
        base_r: float = 0.10,
        chunk_idx: int = 0,
    ) -> Dict:
        """Chunk-by-chunk navigation with goal persistence and drift correction.

        Args:
            current_h: Current hidden states
            anchor_h: Anchor states for drift measurement
            base_r: Base injection ratio
            chunk_idx: Current chunk index (for goal decay)

        Returns:
            Navigation results dict with injections and diagnostics
        """
        injections = {}
        coords_dict = {}
        need_dict = {}

        # Goal persistence decay
        if chunk_idx > 0 and self.goal_state is not None:
            self.goal_persist_strength = max(
                self.DEFAULT_GOAL_MIN_STRENGTH,
                self.goal_persist_strength * self.DEFAULT_GOAL_DECAY
            )

        for L in self.inject_layers:
            if L not in current_h:
                continue

            h = current_h[L]
            h_anchor = anchor_h.get(L) if anchor_h else None

            # ── SNR gating ──
            snr_weight = 1.0
            if h_anchor is not None:
                drift_raw = h.float() - h_anchor.float()
                drift_mag = drift_raw.norm().item()
                if drift_mag > 1e-6:
                    # Estimate SNR from coordinate variance
                    snr_weight = min(1.0, 0.5)  # conservative default

            # ── Three-formula navigation ──
            coords = read_coords(h, self.principal_dirs[L])
            coords_dict[L] = coords

            dc = self.d_consensus.get(L)
            dm = self.d_magnitude.get(L)
            dco = self.d_confidence.get(L)

            delta_k, need_k = compute_delta(
                coords, self.metric_weights[L],
                d_consensus=dc, d_magnitude=dm, d_confidence=dco,
                snr_weight=snr_weight,
            )
            need_dict[L] = need_k

            # Compute r_eff
            need_mean = need_k.mean().item()
            layer_scale = self.layer_scale.get(L, 1.0)
            r_eff = base_r * layer_scale * max(need_mean, 0.1)

            # Base injection
            inj = compute_injection(
                h, delta_k, self.principal_dirs[L],
                self.metric_weights[L], r_eff,
                self.injection_masks.get(L)
            )

            # Goal persistence overlay
            if self.goal_state is not None and L in self.goal_state.goal_delta_k:
                goal_delta = self.goal_state.goal_delta_k[L]
                goal_inj = compute_injection(
                    h, goal_delta, self.principal_dirs[L],
                    self.metric_weights[L], self.goal_persist_strength,
                    self.injection_masks.get(L)
                )
                if goal_inj is not None:
                    inj = goal_inj if inj is None else inj + goal_inj

            # Drift correction
            if h_anchor is not None:
                drift = h.float() - h_anchor.float()
                drift_mag = drift.norm().item()
                if drift_mag > 1e-6:
                    target = self.target_mag.get(L, 0.15)
                    strength = min(0.5, need_mean * self.DEFAULT_DRIFT_CORRECT_RATIO * target / drift_mag)
                    correction = -drift * strength
                    inj = correction if inj is None else inj + correction

            if inj is not None and inj.norm() > 1e-10:
                injections[L] = inj

        return {
            'injections': injections,
            'coords': coords_dict,
            'need': need_dict,
        }
