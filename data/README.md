# Data Files

This directory contains pre-extracted S-space parameters for Qwen3.5-0.8B.

## Included Files

| File | Size | Description |
|------|------|-------------|
| `coord_nav_params_K100.pt` | ~5.7MB | K=57 PCA parameters (ê_k, g_k) for 21 layers |
| `reasoning_consensus.pt` | ~10KB | d_consensus reasoning directions (7 layers) |
| `thinking_dirs.pt` | ~10KB | Thinking direction vectors (for selection masks) |
| `cer_type_classifier.pt` | 1.3MB | Type classifier weights |
| `pure_route_v2.pt` | ~30KB | Sparse selection route vectors (historically "lottery masks") |

## How to Extract Your Own

If you want to apply S-Space to a different model, you don't need these files —
you can extract parameters from any Transformer using the `extraction/` scripts:

```bash
python extraction/extract_pca.py --model_name Qwen/Qwen3-1.7B --layers 3,7,19 --K 100
python extraction/extract_consensus.py --model_name Qwen/Qwen3-1.7B --layers 3,7,19
python extraction/extract_thinking.py --model_name Qwen/Qwen3-1.7B --layers 3,7,19
```
