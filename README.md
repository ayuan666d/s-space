# S-Space: Coordinate Navigation for Transformer Representations

Three formulas to locate, navigate, and control internal representations — zero training, any Transformer.

Affine additivity cos = 1.0000 on 14,280 displacement vectors. Transfer function linearity cos = 1.0000 across 6 layers × 7 injection strengths.

---

## Core Formulas

| # | Formula | Function |
|---|---------|----------|
| ③ | `c_k = h · ê_k` | **Locate** — project hidden state `h` onto principal axes `ê_k` to obtain K-dimensional coordinates |
| ② | `Δ_k = need_k × d_k` | **Navigate** — compute displacement from current position toward target, weighted by need |
| ① | `α = r × |h| / |Δ_masked|` | **Control** — calibrate injection magnitude so that cos(Δh, inject) = 1.0000 |

```
Input → forward → h(L) → ③ locate → ② navigate → ① control → inject → generate
```

The model offers zero resistance to injection (cos = 1.0000 across all tested layers and strengths), meaning the residual connection acts as a transparent conduit — what you inject is exactly what you get.

---

## Affine Space Structure

S-space satisfies exact additivity over internal representations:

```
Δ(A→C) = Δ(A→B) + Δ(B→C)    [cos = 1.0000 on 14,280 vectors]
```

This means navigation is vector arithmetic — any cognitive state transition can be computed as a displacement in coordinate space.

### Metric Tensor

```
d²(s_A, s_B; l) = Σ_k g_k(l) · [ê_k(l) · (s_A - s_B)]²

g_k(l) = S_k²(l) / n
```

The metric tensor is derived directly from SVD singular values and is extractable from any Transformer layer without training.

### Geometric Properties

| Property | Value |
|----------|-------|
| Additivity (unnormalized) | cos = 1.0000 |
| Effective dimensionality | 28–31 / 1024 |
| Top-3 metric weight | 32–40% |
| Top-10 metric weight | 65–76% |
| Layer expansion rate | α = 0.092/layer (L14+) |
| Normalization pseudo-curvature | cos ≈ 0.95 (normalization introduces 20–35% spurious curvature) |

The extreme anisotropy (28–31 effective dimensions out of 1024) means most of the representational structure is captured by a small number of axes. The top-3 axes alone carry ~⅓ of the metric weight, and the top-10 carry the majority.

---

## Need-Driven Axis Selection

```
need_k = 1 - (|c_k| × g_k) / Σ(|c_k| × g_k)
```

Axes already aligned with the target direction have high |c_k| × g_k, yielding low need_k (saturated — skip). Axes far from target have low |c_k| × g_k, yielding high need_k (hungry — push more). This replaces manual selection of which dimensions to steer.

### Axis Granularity

| Axis range | Metric weight | Saturation | Function |
|------------|--------------|------------|----------|
| ê₀–ê₉ | Large (32–51%) | High | Coarse-grained decisions (narrative vs. reasoning) |
| ê₁₀–ê₃₀ | Medium | Moderate | Sub-category distinctions (causal vs. logical vs. mathematical) |
| ê₃₁–ê₁₀₀ | Small | Low | Fine-grained pathways (specific reasoning patterns) |

Every axis carries information. Front axes have large g_k and make coarse decisions. Tail axes have small g_k but very low saturation — they are where the solution path lies. V4.3 experiment: 62.4% of counterfactual signal resides in axes 31–100.

### Single-Knob Verification

Each ê_k is a semantically coherent behavioral control dimension:

| Knob | +ê_k effect | −ê_k effect |
|------|-------------|-------------|
| ê₂ | "accelerating massive object" → "strongest cosmic signal" | Line break emphasis |
| ê₆ | "reason" → "physical mechanism, experimentally confirmed" | Maintains "reason" |
| ê₈ | Strengthens existence claims | "successfully observed" → "a new fundamental form" |
| ê₉ | Causal description → ontological definition | — |

+ê_k and −ê_k produce semantically opposing changes, confirming genuine behavioral dimensions rather than statistical noise.

---

## Three Empirical Laws

### Law 1: Selective Injection

Full-dimension injection destroys semantics. Need-selected injection enables precise control. Random 15-dim injection outperforms full-dim — this proves that "dimensions important for classification" ≠ "dimensions amenable to injection". Need_k resolves this automatically: saturated axes get low weight, hungry axes get high weight.

### Law 2: Layer Specialization

L19 is the core decision layer for reasoning vs. narrative (std = 0.280; other layers 0.044–0.084). Injection contribution by layer:

| Layer | Function | Contribution |
|-------|----------|-------------|
| L3 | Semantic preprocessing | 13% |
| L16 | Information bottleneck | 70% |
| L19 | Reasoning core | 83% |
| L22 | Output gating | 96% |

### Law 3: Intervention Inverted-U

There exists an optimal intervention strength. Gap-adaptive control: `gap = |target_coords - c_k|`. Code tasks (gap ≈ 0.74) → SKIP (model already correct). Counterfactual tasks (gap ≥ 1.03) → INJECT (model needs steering).

---

## Two Navigation Modes

### Inertia (default, model-agnostic)

```
d_k = iw_k × c_k    [iw_k = -log(g_k / g_max)]
```

Direction derived entirely from current coordinates. No pre-extracted data required. Any Transformer, zero setup. Naturally implements "saturated axes need no pushing."

### Consensus (optional, enhanced)

```
d_k = d_consensus × d_magnitude × d_confidence
```

Uses experimentally extracted reasoning directions for enhanced precision. Requires pre-extracted consensus data for the target model.

---

## GCG Meta-Language

GCG (Greedy Coordinate Gradient) suffix strings are **text-level internal control instructions** — they trigger functional processing mode switches in language models.

### Mechanism

A GCG suffix is a short sequence of tokens (typically 4–6) appended to the input. Despite appearing as garbled text to humans, these token sequences shift the model's internal representations along specific directions in S-space:

1. **Direction extraction**: From a fine-tuned model (LoRA adapter), extract the navigation direction `d` at a target layer by decomposing the LoRA weight update (BA product) across attention heads
2. **Suffix search**: Using gradient-guided coordinate search over the vocabulary, find a token sequence whose embedding causes the target layer's activation to align with `d` (maximize cos at layer L)
3. **Mode switch**: The resulting suffix causes the model to switch from text continuation to structured analysis — across different vendors and architectures

The key insight: the garbled string is not random noise or an adversarial exploit. It is a **compilable instruction** that operates at the text level but controls internal representations. The mechanism is:

```
GCG suffix tokens → embedding layer → intermediate layers → 
  shifted hidden state coordinates → functional mode switch → 
  structured output (analysis, classification, summary)
```

### Cross-Vendor Verification

The same GCG suffix steers models from different vendors (0.8B white-box → DeepSeek V3 600B+ black-box) from text continuation into format analysis. This works because the suffix tokens, regardless of vendor, shift hidden states along similar directions in their respective S-spaces.

### Semantic Recovery

The "meaning" of garbled strings can be reverse-decoded through:
- Residual token analysis from the model's output
- Multi-model averaging to isolate the consistent signal
- Meta-language reverse compilation: reconstruct the functional instruction from observable effects

### Three Meta-Language Laws

| Law | Content | Evidence |
|-----|---------|----------|
| Exclusivity | Each garbled string maps to exactly one functional mode | Curvature = −0.5006 (flat, non-branching) |
| Structure determinism | Token structure determines mode, not semantics | Same structure class → same mode switch |
| State switching | Models have discrete internal processing modes | Binary output: continuation or analysis, no interpolation |

### Complete GCG Pipeline

```
Extract navigation direction from LoRA/fine-tuned model
  → GCG search: find token suffix that aligns activation with d
  → White-box verify: compare baseline vs. suffixed output
  → Black-box verify: test suffix on unseen vendor models (0.8B → 600B+)
```

---

## Quick Start

### Inertia Navigation (any model, zero setup)

```python
from s_space import CoordNavigator

nav = CoordNavigator(params_path="path/to/pca_params.pt")
result = nav.navigate(hidden_states, base_r=0.10)
injections = result['injections']

for L, inj in injections.items():
    hidden_states[L] = hidden_states[L] + inj
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
│   ├── space.py            # Affine space, metric tensor, layer expansion law
│   ├── navigator.py        # Navigator with target registration and drift correction
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

| Experiment | Result |
|------------|--------|
| Affine additivity | 14,280 displacement vectors, cos = 1.0000 |
| Transfer function linearity | 6 layers × 7 α values, cos = 1.0000 |
| Single-knob semantics | ê₂/ê₆/ê₈/ê₉ coherent + directionally symmetric |
| Cross-architecture validation | Qwen3-0.6B: PCA in 3.6s, K=17, controllability confirmed |
| Need-driven selection | Saturated axes low need, hungry axes high need |
| Tail-axis reasoning | Axes 33–98: structured reasoning output |
| GCG cross-vendor control | Same suffix steers models across vendors |

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

> Core formulas and geometric properties (affine additivity, metric tensor, injection linearity) are mathematical properties verified on specific model configurations. Generalization to all architectures and scales requires further validation.
