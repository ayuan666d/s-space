# Data Files

This directory contains pre-extracted S-space parameters for Qwen3.5-0.8B.

## Included Files

| File | Size | Description |
|------|------|-------------|
| `coord_nav_params_K100.pt` | ~5.7MB | K=57 PCA parameters (ê_k, g_k) for 21 layers |
| `reasoning_consensus.pt` | ~10KB | d_consensus reasoning directions (7 layers) |
| `thinking_dirs.pt` | ~10KB | Thinking direction vectors (for selection masks) |
| `cer_type_classifier.pt` | 1.3MB | Type classifier weights |
| `pure_route_v2.pt` | ~30KB | Sparse selection route vectors |

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
