# Data Files

This directory contains pre-extracted S-Space parameters for Qwen3.5 models.

## Qwen3.5-0.8B (Original)

| File | Size | Description |
|------|------|-------------|
| `coord_nav_params_K100.pt` | ~5.7MB | K=57 PCA parameters (ê_k, g_k) for 21 layers |
| `reasoning_consensus.pt` | ~11KB | d_consensus reasoning directions (7 layers) |
| `thinking_dirs.pt` | ~105KB | Thinking direction vectors (for selection masks) |
| `cer_type_classifier.pt` | ~1.3MB | Type classifier weights |
| `pure_route_v2.pt` | ~28KB | Sparse selection route vectors |

## Qwen3.5-2B (v0.4)

| File | Size | Description |
|------|------|-------------|
| `coord_nav_params_2B_K100.pt` | ~16.6MB | K=31 PCA parameters (ê_k, g_k) for 24 layers |
| `reasoning_consensus_2B.pt` | ~9KB | d_consensus reasoning directions (7 layers) |
| `thinking_dirs_2B.pt` | ~204KB | Thinking direction vectors for all 24 layers |
| `axis_semantics_2B.json` | ~18KB | Axis semantic mapping (ê₁–ê₁₀ per layer) |
| `type_centroids_2B.pt` | ~701KB | 20 type centroids (14 task + 6 novel genre) |
| `novel_verification_2B.json` | ~36KB | Novel generation verification results |

### Key 2B Findings

- **Stronger anisotropy**: K=31 (vs 0.8B's K=57) — only 31 effective dimensions out of 2048
- **Thinking mode control**: ê₁/ê₅ positive + ê₈/ê₁₀ negative → skip thinking chain, direct Chinese output
- **Chinese output ratio**: 0–2% baseline → 83–87% with S-Space injection
- **Novel genre centroids**: Supports martial arts, sci-fi, modern, fantasy, horror, romance injection

## How to Extract Your Own

If you want to apply S-Space to a different model, you don't need these files —
you can extract parameters from any Transformer using the extraction CLI:

```bash
# Extract from any HuggingFace model
python -m s_space.extraction --model Qwen/Qwen3-1.7B --K 100 --layers 3,7,19

# Or use the Python API
python -c "
from s_space.extraction import extract_pca_params, save_params
params = extract_pca_params('Qwen/Qwen3-1.7B', K=100)
save_params(params, 'pca_params.pt')
"
```

See `python -m s_space.extraction --help` for all options.
