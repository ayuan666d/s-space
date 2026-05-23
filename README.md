# S-Space: Coordinate Navigation for Transformer Representations

Read, steer, and lock internal representations with three formulas — zero training, any Transformer.

**New in v0.4**: Qwen3.5-2B support, thinking mode control (skip/enable chain-of-thought), and 6 pre-extracted data packs.

---

## What You Can Do

**Steer model behavior without retraining.** Inject a displacement vector at specific layers to shift the model from narrative mode into structured analysis, or from casual output into rigorous reasoning — no fine-tuning, no prompt engineering, no data collection.

**Control thinking/chain-of-thought.** For models with thinking mode (e.g., Qwen3.5-2B), S-Space axes can skip or enable the internal reasoning chain. Pushing ê₁ negative / ê₅ positive / ê₈ negative / ê₁₀ negative skips thinking and produces direct output — Chinese output ratio jumps from 0–2% to 83–87%.

**Read what the model is "thinking."** Project any hidden state onto principal axes to get a K-dimensional coordinate. High value on ê₁₉ = the model is reasoning. Low value = it's narrating. This is a real-time diagnostic: you know what the model is doing before it generates the next token.

**Control injection with mathematical precision.** The control formula guarantees cos(Δh, inject) = 1.0000 — the model offers zero resistance. What you inject is exactly what you get, across all tested layers and strengths (14,280 displacement vectors verified).

**Navigate across architectures.** Extract S-space parameters from any HuggingFace Transformer with one function call. The same navigation logic works on Qwen, LLaMA, Mistral, or any model where you can access intermediate hidden states.

**Trigger functional mode switches via text.** GCG suffix strings are compilable instructions: a short token sequence appended to input causes the model to switch from text continuation to structured analysis. Verified cross-vendor (0.8B white-box → DeepSeek V3 600B+ black-box).

**Explore individual behavioral dimensions.** Each principal axis is a semantically coherent control knob. Turning ê₆ up adds physical-mechanism framing; turning it down stays at "reason" level. Single-knob experiments confirm genuine behavioral dimensions, not statistical noise.

---

## How It Works

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

## Thinking Mode Control

**S-Space is the first steering method that can control chain-of-thought behavior at the representation level.**

Models with thinking mode (like Qwen3.5-2B) generate an internal reasoning chain (`<think/>` tokens) before producing visible output. S-Space axes can skip this chain entirely:

```python
from s_space import load_2b_navigator, load_thinking_controller

# Load navigator with thinking control
nav = load_2b_navigator(thinking_mode="skip")

# Or use the controller directly
ctrl = load_thinking_controller("2b")
injections = ctrl.skip_thinking(nav.principal_dirs, nav.metric_weights, nav.K)

# Fine-grained: reduce thinking intensity
injections = ctrl.set_thinking_intensity(nav.principal_dirs, nav.metric_weights, nav.K, intensity=0.5)
```

### Thinking Control Axes (Qwen3.5-2B)

| Axis | Direction | Effect | Chinese Output |
|------|-----------|--------|---------------|
| ê₁ | negative | Skip thinking, direct output | 0% → 83% |
| ê₅ | positive | Skip thinking, direct output | 0% → 85% |
| ê₈ | negative | Skip thinking, direct output | 0% → 87% |
| ê₁₀ | negative | Skip thinking, direct output | 0% → 84% |

This is not prompt engineering — it's direct manipulation of the model's internal reasoning circuitry. No other steering method (ActAdd, CAA, RepE, COAST) has reported this capability.

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

| Property | 0.8B | 2B |
|----------|------|-----|
| Model dimensions (d_model) | 1024 | 2048 |
| Effective dimensions (K) | 57 | 31 |
| Anisotropy ratio | 5.6% | 1.5% |
| Additivity (unnormalized) | cos = 1.0000 | cos = 1.0000 |
| Top-3 metric weight | 32–40% | (stronger) |
| Layer expansion rate | α = 0.092/layer (L14+) | α ≈ 0.092/layer |
| Thinking mode axes | N/A | ê₁, ê₅, ê₈, ê₁₀ |

The extreme anisotropy means most of the representational structure is captured by a small number of axes. The 2B model has even stronger anisotropy (K=31 vs 57) — navigation is more precise with fewer effective dimensions.

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

## Comparison with Related Methods

| Feature | S-Space | ActAdd/CAA | RepE | COAST |
|---------|---------|------------|------|-------|
| Zero training | ✅ | ✅ | ✅ | ✅ |
| Precise magnitude formula | ✅ α = r·|h|/|Δ| | ❌ manual scaling | ❌ manual scaling | ❌ soft projection |
| Affine additivity (cos=1.0) | ✅ | ❌ | ❌ | ❌ |
| Metric tensor (geometry) | ✅ | ❌ | ❌ | ❌ |
| Need-driven axis selection | ✅ automatic | ❌ manual | ❌ manual | ❌ conceptor fixed |
| Thinking mode control | ✅ ê₁/ê₅/ê₈/ê₁₀ | ❌ | ❌ | ❌ |
| Cross-architecture | ✅ | ✅ (limited) | ✅ (limited) | ✅ (robotics) |
| GCG meta-language | ✅ | ❌ | ❌ | ❌ |
| Weight compilation | 🔜 planned | ❌ | ❌ | ❌ |

**COAST** (NeurIPS 2026) independently validates the PCA → subspace → steering pipeline with conceptor soft projection, but lacks navigation theory and quantitative magnitude control.

**ActAdd/CAA** (ICML 2024 / ACL 2024) use contrastive activation addition but rely on manual scaling factors — no closed-form magnitude formula.

**RepE** (NeurIPS 2023) pioneered representation reading/intervention but did not develop a coordinate navigation framework.

---

## Quick Start

### One-Line Setup (Pre-tuned Models)

```python
# Qwen3.5-2B with thinking mode control
from s_space import load_2b_navigator
nav = load_2b_navigator(thinking_mode="skip")

# Qwen3.5-0.8B
from s_space import load_08b_navigator
nav = load_08b_navigator(consensus=True)

# Or use the generic loader
from s_space import load_navigator
nav = load_navigator("2b")
```

### Inertia Navigation (any model, zero setup)

```python
from s_space import CoordNavigator

nav = CoordNavigator(params_path="path/to/pca_params.pt")
result = nav.navigate(hidden_states, base_r=0.10)
injections = result['injections']

for L, inj in injections.items():
    hidden_states[L] = hidden_states[L] + inj
```

### Thinking Mode Control

```python
from s_space import ThinkingController, load_navigator

nav = load_navigator("2b")
ctrl = ThinkingController.for_2b()

# Skip thinking chain — direct output
injections = ctrl.skip_thinking(nav.principal_dirs, nav.metric_weights, nav.K)

# Fine-grained control (0.0 = full thinking, 1.0 = skip thinking)
injections = ctrl.set_thinking_intensity(nav.principal_dirs, nav.metric_weights, nav.K, intensity=0.3)
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

## Supported Pre-tuned Models

| Model | K | Layers | Data Files | Special Features |
|-------|---|--------|------------|-----------------|
| Qwen3.5-0.8B | 57 | 21 | 5 | Consensus directions, type classifier |
| Qwen3.5-2B | 31 | 24 | 6 | Thinking mode control, type centroids, novel genres, axis semantics |

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
│   ├── thinking.py         # Thinking mode control (2B)
│   ├── pretuned.py         # Multi-model pre-tuned data packs
│   └── extraction/         # PCA extraction from any HuggingFace model
├── data/                   # Pre-extracted parameters
│   ├── *_K100.pt           # 0.8B PCA params (K=57)
│   ├── *_2B_*.pt           # 2B PCA params (K=31)
│   └── README.md           # Data file descriptions
└── docs/                   # Full theoretical documentation
    ├── S_SPACE_FORMULAS.md
    ├── S_SPACE_CONTROLLABILITY.md
    ├── SINGLE_KNOB_FINDINGS.md
    ├── EXPERIMENT_META_ANALYSIS.md
    ├── DEFINITIVE_ANSWERS.md
    ├── METHOD_COMPARISON.md
    └── ARCHITECTURE.md
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
| 2B thinking mode control | Chinese output 0–2% → 83–87% with thinking skip |
| 2B anisotropy | K=31 / 2048 = 1.5% effective dimensions |
| 2B novel genre injection | Writing + genre centroids: coherent Chinese fiction |

---

## Citation

```bibtex
@article{sspace2026,
  title={S-Space: Coordinate Navigation for Transformer Internal Representations},
  author={ayuan666d},
  year={2026},
  url={https://github.com/ayuan666d/s-space}
}
```

---

## License

Apache License 2.0

---

> Core formulas and geometric properties (affine additivity, metric tensor, injection linearity) are mathematical properties verified on specific model configurations. Generalization to all architectures and scales requires further validation.
