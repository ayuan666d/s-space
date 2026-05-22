"""
S-Space PCA Parameter Extraction

Extract principal directions (ê_k) and metric weights (g_k) from any
HuggingFace Transformer model. These parameters power the three
S-Space navigation formulas.

How it works:
    1. Load a HuggingFace model
    2. Run diverse text through it, capture hidden states per layer
    3. Center the hidden states
    4. SVD on the centered matrix → principal_dirs = Vh[:K]
    5. metric_weights = S[:K]² / n

The output .pt file can be directly loaded by CoordNavigator:

    nav = CoordNavigator(params_path="pca_params.pt")

References:
    - S_SPACE_FORMULAS.md: Mathematical derivation of PCA → S-space
    - EXPERIMENT_META_ANALYSIS.md: Validation with 198 experiments
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Default diverse prompts for capturing hidden state variance
# These cover different reasoning types to ensure broad coverage
DEFAULT_PROMPTS = [
    # Reasoning
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "What is 15% of 240? Show your reasoning.",
    "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "Explain the logical fallacy in: 'If it rains, the ground gets wet. The ground is wet, so it must have rained.'",

    # Narrative
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Describe a sunset over a mountain lake in vivid detail.",
    "Create a dialogue between two strangers stuck in an elevator.",

    # Factual / Knowledge
    "What are the three laws of thermodynamics?",
    "Explain how photosynthesis converts sunlight into chemical energy.",
    "Describe the structure of a DNA molecule.",

    # Code
    "Write a Python function that finds the longest common subsequence of two strings.",
    "Explain the difference between a stack and a queue data structure.",

    # Translation / Language
    "Translate 'The quick brown fox jumps over the lazy dog' into formal academic language.",
    "Rewrite this sentence to be more concise: 'Due to the fact that it was raining outside, we decided to stay indoors.'",

    # Counterfactual
    "What would happen if Earth had no moon?",
    "If gravity were twice as strong, how would architecture change?",

    # Causal reasoning
    "Why does hot air rise? Explain the causal mechanism.",
    "What caused the fall of the Roman Empire? List three primary factors.",

    # Creative / Abstract
    "Define 'emergence' in the context of complex systems.",
    "What is the relationship between beauty and symmetry in mathematics?",

    # Common sense
    "Is it safe to drink water from a river? Why or why not?",
    "How do you change a flat tire on a car?",
]


def collect_hidden_states(
    model_name: str,
    prompts: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    n_samples: int = 100,
    max_length: int = 128,
    device: str = 'auto',
    batch_size: int = 4,
) -> Dict[int, torch.Tensor]:
    """Collect hidden states from a HuggingFace model.

    Runs diverse text through the model and captures the last-token
    hidden state at each specified layer.

    Args:
        model_name: HuggingFace model name/path (e.g., "Qwen/Qwen2.5-7B")
        prompts: Text prompts to run (default: DEFAULT_PROMPTS × repetitions)
        layers: Layer indices to capture (default: auto-detect all)
        n_samples: Minimum number of samples to collect
        max_length: Max token length per prompt
        device: Device to use ('auto', 'cuda', 'cpu')
        batch_size: Batch size for inference

    Returns:
        Dict mapping {layer_index: tensor(n_samples, d_model)}
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required for extraction. "
            "Install with: pip install transformers"
        )

    # Device selection
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare prompts
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    # Repeat prompts to reach n_samples
    n_reps = max(1, (n_samples + len(prompts) - 1) // len(prompts))
    all_prompts = (prompts * n_reps)[:n_samples]

    logger.info(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device if device == 'cuda' else None,
    )
    if device == 'cpu':
        model = model.to(device)
    model.eval()

    # Auto-detect layers
    if layers is None:
        n_layers = model.config.num_hidden_layers
        layers = list(range(n_layers))
        logger.info(f"Auto-detected {n_layers} layers")

    # Collect hidden states
    logger.info(f"Collecting hidden states from {len(all_prompts)} prompts...")

    hidden_states = {L: [] for L in layers}
    n_collected = 0

    with torch.no_grad():
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device if hasattr(model, 'device') else device)

            # Forward with hidden states
            outputs = model(**inputs, output_hidden_states=True)

            # Extract last-token hidden state per layer
            for L in layers:
                if L < len(outputs.hidden_states):
                    # Shape: (batch, seq_len, d_model)
                    hs = outputs.hidden_states[L]

                    # Get last non-padding token for each sample
                    attention_mask = inputs.get('attention_mask', None)
                    if attention_mask is not None:
                        # Find last real token position for each sample
                        last_indices = attention_mask.sum(dim=1) - 1
                        last_hidden = hs[torch.arange(hs.shape[0]), last_indices]
                    else:
                        last_hidden = hs[:, -1, :]

                    hidden_states[L].append(last_hidden.float().cpu())

            n_collected += len(batch_prompts)
            if (n_collected % 20) == 0 or n_collected == len(all_prompts):
                logger.info(f"  Collected {n_collected}/{len(all_prompts)} samples")

    # Concatenate
    for L in layers:
        if hidden_states[L]:
            hidden_states[L] = torch.cat(hidden_states[L], dim=0)

    # Free model memory
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return hidden_states


def compute_pca(
    hidden_states: torch.Tensor,
    K: int = 100,
    variance_threshold: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Compute PCA from hidden state matrix.

    Uses centering + SVD (equivalent to PCA, no sklearn needed).

    Args:
        hidden_states: Matrix of shape (n_samples, d_model)
        K: Target number of principal components
        variance_threshold: Stop at this cumulative variance ratio

    Returns:
        (principal_dirs, singular_vals, metric_weights, global_mean, stats)
        - principal_dirs: (K_actual, d_model)
        - singular_vals: (K_actual,)
        - metric_weights: (K_actual,) = S² / n
        - global_mean: (d_model,)
        - stats: dict with variance info
    """
    n, d = hidden_states.shape
    logger.info(f"Computing PCA: n={n}, d={d}, K_target={K}")

    # Center
    global_mean = hidden_states.mean(dim=0)
    X_centered = hidden_states - global_mean.unsqueeze(0)

    # SVD: X_centered = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    # Variance explained
    total_var = (S ** 2).sum()
    explained_ratio = (S ** 2) / total_var
    cum_ratio = torch.cumsum(explained_ratio, dim=0)

    # Determine actual K
    k_by_variance = int((cum_ratio < variance_threshold).sum()) + 1
    k_actual = min(K, k_by_variance, len(S))
    k_actual = max(k_actual, 10)  # at least 10 dimensions

    # Extract components
    principal_dirs = Vh[:k_actual]       # (K, d_model)
    singular_vals = S[:k_actual]         # (K,)
    metric_weights = (S[:k_actual] ** 2) / n  # (K,)

    stats = {
        'n_samples': n,
        'd_model': d,
        'K_actual': k_actual,
        'K_by_variance': k_by_variance,
        'total_variance': total_var.item(),
        'cum_variance_at_K': cum_ratio[k_actual - 1].item() if k_actual <= len(cum_ratio) else cum_ratio[-1].item(),
        'top3_variance_pct': (explained_ratio[:3].sum() * 100).item() if len(explained_ratio) >= 3 else 0,
        'top10_variance_pct': (explained_ratio[:10].sum() * 100).item() if len(explained_ratio) >= 10 else 0,
    }

    logger.info(
        f"  K_actual={k_actual}, "
        f"top3={stats['top3_variance_pct']:.1f}%, "
        f"top10={stats['top10_variance_pct']:.1f}%, "
        f"cum@K={stats['cum_variance_at_K']:.3f}"
    )

    return principal_dirs, singular_vals, metric_weights, global_mean, stats


def extract_pca_params(
    model_name: str,
    layers: Optional[List[int]] = None,
    K: int = 100,
    n_samples: int = 100,
    max_length: int = 128,
    device: str = 'auto',
    batch_size: int = 4,
    prompts: Optional[List[str]] = None,
    variance_threshold: float = 0.95,
) -> Dict:
    """Extract S-space PCA parameters from any HuggingFace model.

    This is the main entry point. One function call extracts everything
    needed for CoordNavigator.

    Args:
        model_name: HuggingFace model name or path
        layers: Layers to extract (default: all layers)
        K: Target PCA dimension
        n_samples: Number of hidden state samples to collect
        max_length: Max token length per prompt
        device: Device ('auto', 'cuda', 'cpu')
        batch_size: Inference batch size
        prompts: Custom prompts (default: built-in diverse set)
        variance_threshold: Cumulative variance threshold for K selection

    Returns:
        Dict with 'principal_dirs', 'metric_weights', 'singular_vals',
        'global_centroids', 'K', 'model_name', and per-layer stats.

    Example:
        >>> params = extract_pca_params("Qwen/Qwen2.5-0.5B")
        >>> torch.save(params, "pca_params.pt")
        >>> nav = CoordNavigator(params_path="pca_params.pt")
    """
    logger.info(f"Extracting S-space PCA from {model_name}")

    # Step 1: Collect hidden states
    hidden_states = collect_hidden_states(
        model_name=model_name,
        prompts=prompts,
        layers=layers,
        n_samples=n_samples,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
    )

    # Step 2: PCA per layer
    principal_dirs = {}
    metric_weights_dict = {}
    singular_vals_dict = {}
    global_centroids = {}
    layer_stats = {}

    for L in sorted(hidden_states.keys()):
        hs = hidden_states[L]
        if hs.shape[0] < 10:
            logger.warning(f"  Skipping L{L}: only {hs.shape[0]} samples")
            continue

        pd, sv, mw, gm, stats = compute_pca(hs, K=K, variance_threshold=variance_threshold)
        principal_dirs[L] = pd
        singular_vals_dict[L] = sv
        metric_weights_dict[L] = mw
        global_centroids[L] = gm
        layer_stats[L] = stats

    # Step 3: Assemble output
    # Determine K from the most common value
    K_values = [v.shape[0] for v in principal_dirs.values()]
    K_common = max(set(K_values), key=K_values.count) if K_values else K

    params = {
        'principal_dirs': principal_dirs,
        'metric_weights': metric_weights_dict,
        'singular_vals': singular_vals_dict,
        'global_centroids': global_centroids,
        'K': K_common,
        'model_name': model_name,
        'n_samples': n_samples,
        'extraction_method': 'fullsample_pca',
        'layer_stats': layer_stats,
    }

    logger.info(f"Extraction complete: {len(principal_dirs)} layers, K={K_common}")
    return params


def save_params(params: Dict, path: str):
    """Save extracted parameters to a .pt file.

    Args:
        params: Output from extract_pca_params
        path: Output file path
    """
    torch.save(params, path)
    size_mb = Path(path).stat().st_size / 1024 / 1024
    logger.info(f"Saved to {path} ({size_mb:.1f} MB)")


def load_params(path: str, device: str = 'cpu') -> Dict:
    """Load extracted parameters from a .pt file.

    Args:
        path: Path to .pt file
        device: Device to load tensors to

    Returns:
        Parameters dict
    """
    params = torch.load(path, map_location=device, weights_only=False)
    return params
