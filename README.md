# S-Space: Coordinate Navigation for Transformer Representations

**Three formulas to read, navigate, and control internal representations — zero training, any Transformer.**

Verified on 14,280 displacement vectors with additivity cos = 1.0000. Transfer function linearity cos = 1.0000 across 6 layers × 7 injection strengths.

---

## Core Formulas

| # | Formula | Function |
|---|---------|----------|
| ③ | `c_k = h · ê_k` | **Locate** — read K-dimensional coordinates of hidden state `h` |
| ② | `Δ_k = need_k × d_k` | **Navigate** — compute displacement from current to target |
| ① | `α = r × |h| / |Δ_masked|` | **Control** — calibrate injection magnitude (cos = 1.0000) |

```
Input → forward → h(L) → ③ locate → ② navigate → ① control → inject → generate
```

---

## Affine Space Axiom

S-space is an exact affine space over internal representations:

```
Δ(A→C) = Δ(A→B) + Δ(B→C)    [additivity cos = 1.0000 on 14,280 vectors]
```

Consequence: **navigation = vector arithmetic**. Any cognitive state transition can be computed as a displacement in coordinate space.

### Metric Tensor

```
d²(s_A, s_B; l) = Σ_k g_k(l) · [ê_k(l) · (s_A - s_B)]²

g_k(l) = S_k²(l) / n    [derived from SVD singular values, extractable from any model]
```

### Geometric Properties

| Property | Value | Significance |
|----------|-------|-------------|
| Additivity (unnormalized) | cos = 1.0000 | Exact affine space |
| Effective dimensionality | 28–31 / 1024 | Extreme anisotropy |
| Top-3 metric weight | 32–40% | 3 directions carry ~⅓ of structure |
| Top-10 metric weight | 65–76% | 10 dims capture majority of navigation info |
| Layer expansion rate | α = 0.092/layer (L14+) | Exponential distance growth in deep layers |
| Normalization pseudo-curvature | cos ≈ 0.95 | Normalization introduces 20–35% spurious curvature |

---

## Need-Driven Axis Selection

```
need_k = 1 - (|c_k| × g_k) / Σ(|c_k| × g_k)
```

Axes already aligned with target direction have low `need_k` (saturated → skip). Axes far from target have high `need_k` (hungry → push more). This eliminates manual hyperparameter tuning for which dimensions to steer.

### Axis Granularity

| Axis range | Metric weight g_k | Saturation | Function |
|------------|-------------------|------------|----------|
| ê₀–ê₉ | Large (32–51%) | High | Coarse-grained decisions (narrative vs. reasoning) |
| ê₁₀–ê₃₀ | Medium | Moderate | Sub-category distinctions (causal vs. logical vs. mathematical) |
| ê₃₁–ê₁₀₀ | Small | Low | Fine-grained pathways (specific reasoning patterns) |

V4.3 experiment: **62.4% of counterfactual signal resides in axes 31–100**, confirming that low-g_k axes are critical for solution paths despite carrying less structural weight.

### Single-Knob Verification

Each `ê_k` is a **semantically coherent behavioral control dimension** — not a label projection:

| Knob | Axis name | +ê_k effect | −ê_k effect |
|------|-----------|-------------|-------------|
| ê₂ | Viewpoint/signal | "accelerating massive object" → "strongest cosmic signal" | Line break emphasis |
| ê₆ | Systematization | "reason" → "physical mechanism, experimentally confirmed" | Maintains "reason" |
| ê₈ | Existence assertion | Strengthens existence claims | "successfully observed" → "a new fundamental form" |
| ê₉ | Formal register | Causal description → ontological definition | — |

**Directional symmetry**: +ê_k and −ê_k produce **semantically opposing** changes, confirming genuine behavioral dimensions rather than statistical noise.

---

## Three Empirical Laws

### Law 1: Selective Injection

| Condition | Result |
|-----------|--------|
| Full-dimension injection | Semantic destruction |
| Need-selected injection | Precise behavioral control |
| Random 15-dim injection | Outperforms full-dim — proves "classification-important" ≠ "injection-amenable" |

### Law 2: Layer Specialization

L19 is the core decision layer for reasoning vs. narrative (std = 0.280, other layers 0.044–0.084):

| Layer | Function | Injection contribution |
|-------|----------|----------------------|
| L3 | Semantic preprocessing | 13% |
| L16 | Information bottleneck | 70% |
| L19 | Reasoning core | 83% |
| L22 | Output gating | 96% |

### Law 3: Intervention Inverted-U

Optimal intervention strength exists — too weak is ineffective, too strong is destructive.

- Gap-adaptive: `gap = |target_coords - c_k|`
- Code tasks: gap ≈ 0.74 → **SKIP** (model already correct, injection harms)
- Counterfactual tasks: gap ≥ 1.03 → **INJECT** (model needs steering toward reasoning region)

---

## Two Navigation Modes

### Mode 1: Inertia (default, model-agnostic)

```
d_k = iw_k × c_k
```

Direction derived entirely from current coordinates. No pre-extracted data required — **any Transformer, zero setup**. Naturally implements "saturated axes need no pushing."

### Mode 2: Consensus (optional, enhanced)

```
d_k = d_consensus × d_magnitude × d_confidence
```

Uses experimentally extracted reasoning directions for enhanced precision. Requires pre-extracted consensus data for the target model.

---

## Comparison with Existing Work

| Capability | Anthropic SAE/NLA | RepE/ActAdd | Nature 2026 | RISER (ICLR 2026) | **S-Space** |
|------------|-------------------|-------------|-------------|--------------------|----|
| Read internal state | ✅ passive observation | — | — | — | ✅ coordinate location ③ |
| Navigate to target | — | ❌ blind push | — | — | ✅ formula ② + need_k |
| Control magnitude | ❌ no formula | ❌ manual α | — | — | ✅ formula ① cos=1.0 |
| Zero training | — | — | — | ❌ RL router | ✅ |
| Adaptive axis selection | — | — | — | — | ✅ need_k |
| GCG interpretability | — | — | ❌ black-box | — | ✅ meta-language decoding |

---

## GCG Meta-Language

GCG suffix strings are **text-level internal control instructions** — not adversarial attacks. They trigger functional processing mode switches across model vendors.

1. **Cross-vendor control**: GCG strings steer different vendors' models from text continuation → format analysis
2. **Semantic recovery**: Residual tokens + multi-model averaging + meta-language reverse-decoding extract the "meaning" of garbled strings
3. **Geometric explanation**: S-space coordinate decoding shows GCG strings shift hidden state coordinates in K-dimensional space
4. **Full pipeline**: Extract navigation directions → GCG search trigger suffix → white-box verify → black-box verify (0.8B → DeepSeek V3 600B+)

| | Nature 2026 | S-Space |
|---|-------------|---------|
| Vulnerability existence | ✅ mathematical proof | ✅ experimental confirmation |
| Vulnerability location | ❌ | ✅ L19 group_dist + coordinate location |
| Active control | ❌ passive testing only | ✅ cross-vendor internal control |
| Why garbled strings work | ❌ black-box | ✅ meta-language decoding + coordinate explanation |
| Complete toolchain | ❌ | ✅ extract → search → white-box → black-box |

---

## Quick Start

### Inertia Navigation (any model, zero setup)

```python
from s_space import CoordNavigator

nav = CoordNavigator(params_path="path/to/pca_params.pt")

# Three formulas run automatically: locate → navigate → control
result = nav.navigate(hidden_states, base_r=0.10)
injections = result['injections']

for L, inj in injections.items():
    hidden_states[L] = hidden_states[L] + inj
```

### Consensus-Enhanced Navigation (with pre-extracted data)

```python
nav = CoordNavigator(
    params_path="path/to/pca_params.pt",
    consensus_path="path/to/reasoning_consensus.pt",
)
result = nav.navigate(hidden_states, base_r=0.10)
```

### Extract S-Space Parameters from Any Model

```python
from s_space.extraction import extract_pca_params

params = extract_pca_params("Qwen/Qwen2.5-7B", layers=[3, 16, 19, 22])
torch.save(params, "pca_params.pt")
```

### Install

```bash
git clone https://github.com/ayuan666d/s-space.git
cd s-space
pip install -e .
```

---

## Project Structure

```
s-space/
├── s_space/
│   ├── formulas.py         # Three core formulas
│   ├── space.py            # Affine space + metric tensor + layer expansion law
│   ├── navigator.py        # Navigator (target registration + drift correction)
│   ├── injection_mask.py   # Selective injection masks
│   ├── explorer.py         # Single-knob axis explorer
│   ├── pretuned.py         # One-line 0.8B setup
│   └── extraction/         # PCA extraction from any HuggingFace model
├── data/                   # Pre-extracted parameters (large files in Release)
├── docs/                   # Full theoretical documentation
│   ├── S_SPACE_FORMULAS.md
│   ├── S_SPACE_CONTROLLABILITY.md
│   ├── SINGLE_KNOB_FINDINGS.md
│   ├── EXPERIMENT_META_ANALYSIS.md
│   ├── DEFINITIVE_ANSWERS.md
│   ├── METHOD_COMPARISON.md
│   └── ARCHITECTURE.md
└── experiments/            # Experiment reproduction scripts
```

---

## Verified Results

| Experiment | Key Result | Documentation |
|------------|-----------|---------------|
| Affine space validation | 14,280 vectors, additivity cos = 1.0000 | `docs/S_SPACE_FORMULAS.md` |
| Transfer function linearity | 6 layers × 7 α values, cos = 1.0000 | `docs/S_SPACE_CONTROLLABILITY.md` |
| Single-knob semantic mapping | ê₂/ê₆/ê₈/ê₉ coherent + directionally symmetric | `docs/SINGLE_KNOB_FINDINGS.md` |
| Cross-architecture validation | Qwen3-0.6B PCA in 3.6s, K=17, controllability confirmed | `experiments/` |
| Need-driven axis selection | V4 inertia: saturated axes low need, hungry axes high need | `experiments/` |
| Tail-axis reasoning | V4.3 axes 33–98: structured reasoning, correct formulas | `experiments/` |
| GCG cross-vendor control | Garbled strings control models across vendors | `experiments/` |

---

## Citation

```bibtex
@article{sspace2026,
  title={S-Space: Coordinate Navigation for Transformer Internal Representations},
  author={Anonymous},
  year={2026}
}
```

---

## License

Apache License 2.0

---

> **Note**: Core formulas and geometric properties (affine additivity, metric tensor, injection linearity) are architecture-agnostic mathematical properties verified on specific model configurations. Generalization to all architectures and scales requires further validation. Important decisions should be verified by domain experts.
