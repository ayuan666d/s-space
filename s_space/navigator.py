"""
CoordNavigator — Production-Ready S-Space Navigation

Complete navigator that reads coordinates, computes navigation, and controls
injection magnitude using the three core formulas.

Supports:
    - Multi-layer injection (L3, L7, L19 by default, configurable)
    - Chunk-by-chunk continuous navigation
    - Goal persistence with decay
    - SNR gating for noisy coordinates
    - Drift correction
    - Lottery dimension masking (Law 1)
    - Residual convergence loop

Architecture-agnostic: works with any Transformer model.
Only requires PCA parameters extracted from the target model.
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
        1. Load PCA parameters (ê_k, g_k) and consensus directions
        2. Forward model, capture hidden states
        3. Formula ③: Read coordinates c_k = h · ê_k
        4. Formula ②: Compute navigation Δ_k = need_k × d_k
        5. Formula ①: Control magnitude α = r × |h| / |Δ_masked|
        6. Inject into model, generate output

    Args:
        params_path: Path to coord_nav_params.pt file
        consensus_path: Path to reasoning_consensus.pt file
        lottery_path: Path to thinking_dirs.pt file (for lottery masks)
        inject_layers: Which layers to inject into (default: [3, 7, 19])
        lottery_dims: Number of lottery dimensions (Law 1, default: 15)
        device: Device to use
    """

    # ── Layer specialization (experimentally validated) ──
    DEFAULT_LAYER_SCALE = {3: 0.5, 7: 0.8, 19: 1.0}

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
    DEFAULT_TARGET_MAG = {3: 0.10, 7: 0.15, 19: 0.25}

    def __init__(
        self,
        params_path: str,
        consensus_path: Optional[str] = None,
        lottery_path: Optional[str] = None,
        inject_layers: Optional[List[int]] = None,
        lottery_dims: int = 15,
        device: str = 'cpu',
    ):
        self.device = device
        self.lottery_dims = lottery_dims

        # ── Load PCA parameters ──
        params = torch.load(params_path, map_location=device, weights_only=False)
        self.principal_dirs = params['principal_dirs']   # {layer: (K, d_model)}
        self.metric_weights = params['metric_weights']   # {layer: (K,)}

        # K dimension
        if 'K_L19' in params:
            self.K = params['K_L19']
        elif 'K' in params:
            self.K = params['K']
        else:
            first = next(iter(self.principal_dirs))
            self.K = self.principal_dirs[first].shape[0]

        # d_model
        first_layer = next(iter(self.principal_dirs))
        self.d_model = self.principal_dirs[first_layer].shape[1]

        # Inject layers
        self.inject_layers = inject_layers or sorted(
            [l for l in [3, 7, 19] if l in self.principal_dirs]
        )

        # ── Load consensus directions ──
        self.d_consensus: Dict[int, torch.Tensor] = {}
        self.d_magnitude: Dict[int, torch.Tensor] = {}
        self.d_confidence: Dict[int, torch.Tensor] = {}
        if consensus_path:
            self._load_consensus(consensus_path)

        # ── Load lottery masks ──
        self.lottery_masks: Dict[int, torch.Tensor] = {}
        if lottery_path:
            self._build_lottery_masks(lottery_path)

        # ── Goal state ──
        self.goal_state: Optional[GoalState] = None
        self.goal_persist_strength = self.DEFAULT_GOAL_PERSIST_STRENGTH

        # ── Layer config ──
        self.layer_scale = {L: self.DEFAULT_LAYER_SCALE.get(L, 1.0)
                           for L in self.inject_layers}
        self.target_mag = {L: self.DEFAULT_TARGET_MAG.get(L, 0.15)
                          for L in self.inject_layers}

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
                K_old = dc.shape[0]
                self.d_consensus[L] = torch.zeros(self.K)
                self.d_magnitude[L] = torch.ones(self.K) * 0.08
                self.d_confidence[L] = torch.ones(self.K) * 0.3
                self.d_consensus[L][:K_old] = dc
                self.d_magnitude[L][:K_old] = dm
                self.d_confidence[L][:K_old] = dco
            else:
                self.d_consensus[L] = torch.zeros(self.K)
                self.d_magnitude[L] = torch.ones(self.K) * 0.08
                self.d_confidence[L] = torch.ones(self.K) * 0.3

    def _build_lottery_masks(self, path: str):
        """Build lottery dimension masks from thinking directions.

        Law 1 (Semantic Preserve): Only inject into the top lottery_dims
        dimensions. This preserves 1009 dimensions from side effects.
        """
        td = torch.load(path, map_location='cpu', weights_only=False)
        for L in self.inject_layers:
            if L in td:
                _, top_idx = td[L].float().abs().topk(self.lottery_dims)
                mask = torch.zeros(self.d_model, dtype=torch.float32)
                mask[top_idx] = 1.0
                self.lottery_masks[L] = mask

    # ════════════════════════════════════════════════════════════
    # Core Navigation (single step)
    # ════════════════════════════════════════════════════════════

    def navigate(
        self,
        hidden_states: Dict[int, torch.Tensor],
        base_r: float = 0.10,
    ) -> Dict:
        """Single-step navigation: read coords → compute delta → inject.

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

            if L not in self.d_consensus:
                continue

            delta_k, need_k = compute_delta(
                coords, self.d_consensus[L], self.d_magnitude[L],
                self.d_confidence[L], self.metric_weights[L]
            )
            delta_dict[L] = delta_k
            need_dict[L] = need_k

            if delta_k.norm() < 1e-8 and self.goal_state is None:
                continue

            # Compute r_eff based on need
            layer_scale = self.layer_scale.get(L, 1.0)
            dc = self.d_consensus[L]
            has_evidence = dc.abs() > 1e-6
            need_mean = need_k[has_evidence].mean().item() if has_evidence.any() else 0.0
            r_eff = base_r * layer_scale * max(need_mean, 0.1)
            r_eff_dict[L] = r_eff

            # Three-formula injection
            inj = compute_injection(
                h, delta_k, self.principal_dirs[L],
                self.metric_weights[L], r_eff,
                self.lottery_masks.get(L)
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

        The goal direction is the initial delta_k (need × d_consensus),
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
            if L not in captured_h or L not in self.d_consensus:
                continue

            h = captured_h[L]
            coords = read_coords(h, self.principal_dirs[L])
            delta_k, need_k = compute_delta(
                coords, self.d_consensus[L], self.d_magnitude[L],
                self.d_confidence[L], self.metric_weights[L]
            )
            goal_delta_k[L] = delta_k

            dc = self.d_consensus[L]
            has_evidence = dc.abs() > 1e-6
            if has_evidence.any():
                need_max = max(need_max, need_k[has_evidence].max().item())
                need_mean_sum += need_k[has_evidence].mean().item()
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
            if h_anchor is not None and L in self.lottery_masks:
                drift_raw = h.float() - h_anchor.float()
                drift_effective = drift_raw * self.lottery_masks[L]
                noise = drift_raw - drift_effective
                snr = drift_effective.norm().item() / (noise.norm().item() + 1e-8)
                snr_weight = min(1.0, snr / self.DEFAULT_SNR_GATE)

            # ── Three-formula navigation ──
            coords = read_coords(h, self.principal_dirs[L])
            coords_dict[L] = coords

            delta_k, need_k = compute_delta(
                coords, self.d_consensus[L], self.d_magnitude[L],
                self.d_confidence[L], self.metric_weights[L],
                snr_weight=snr_weight,
            )
            need_dict[L] = need_k

            # Compute r_eff
            dc = self.d_consensus[L]
            has_evidence = dc.abs() > 1e-6
            need_mean = need_k[has_evidence].mean().item() if has_evidence.any() else 0.0
            layer_scale = self.layer_scale.get(L, 1.0)
            r_eff = base_r * layer_scale * max(need_mean, 0.1)

            # Base injection
            inj = compute_injection(
                h, delta_k, self.principal_dirs[L],
                self.metric_weights[L], r_eff,
                self.lottery_masks.get(L)
            )

            # Goal persistence overlay
            if self.goal_state is not None and L in self.goal_state.goal_delta_k:
                goal_delta = self.goal_state.goal_delta_k[L]
                goal_inj = compute_injection(
                    h, goal_delta, self.principal_dirs[L],
                    self.metric_weights[L], self.goal_persist_strength,
                    self.lottery_masks.get(L)
                )
                if goal_inj is not None:
                    inj = goal_inj if inj is None else inj + goal_inj

            # Drift correction
            if h_anchor is not None and L in self.lottery_masks:
                drift_effective = (h.float() - h_anchor.float()) * self.lottery_masks[L]
                drift_mag = drift_effective.norm().item()
                if drift_mag > 1e-6:
                    target = self.target_mag.get(L, 0.15)
                    strength = min(0.5, need_mean * self.DEFAULT_DRIFT_CORRECT_RATIO * target / drift_mag)
                    correction = -drift_effective * strength
                    inj = correction if inj is None else inj + correction

            if inj is not None and inj.norm() > 1e-10:
                injections[L] = inj

        return {
            'injections': injections,
            'coords': coords_dict,
            'need': need_dict,
        }
