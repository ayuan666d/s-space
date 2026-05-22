# Data Files

This directory contains pre-extracted S-space parameters for Qwen3.5-0.8B.

## Included Files

| File | Size | Description |
|------|------|-------------|
| `reasoning_consensus.pt` | ~10KB | d_consensus reasoning directions (7 layers) |
| `thinking_dirs.pt` | ~10KB | Thinking direction vectors (for lottery masks) |
| `cer_type_classifier.pt` | 1.3MB | Type classifier weights |
| `pure_route_v2.pt` | ~30KB | Lottery route vectors |

## Large Files (Download Separately)

The following files are too large for git and must be downloaded separately:

| File | Size | URL | Description |
|------|------|-----|-------------|
| `coord_nav_params_fullsample.pt` | 63.8MB | [GitHub Release](../../releases) | K=100 PCA parameters (ê_k, g_k) |
| `s_raw_full.pt` | 88.6MB | [GitHub Release](../../releases) | Raw hidden states (745 × 24 layers × 1024) |
| `feats_all.pt` | 7.8MB | [GitHub Release](../../releases) | Feature matrix (1992 × 1024) |
| `s_displacements_full.pt` | 85MB | [GitHub Release](../../releases) | Displacement vectors for affine validation |

## How to Extract Your Own

If you want to apply S-Space to a different model, you don't need these files —
you can extract parameters from any Transformer using the `extraction/` scripts:

```bash
python extraction/extract_pca.py --model_name Qwen/Qwen3-1.7B --layers 3,7,19 --K 100
python extraction/extract_consensus.py --model_name Qwen/Qwen3-1.7B --layers 3,7,19
python extraction/extract_thinking.py --model_name Qwen/Qwen3-1.7B --layers 3,7,19
```
