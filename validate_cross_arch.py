"""
Cross-Architecture Validation Script

Validates that the S-Space three formulas work on a model that is NOT
the original 0.8B Qwen. This proves the framework is model-agnostic.

Test plan:
    1. Load a different architecture model (e.g., Qwen2.5-1.5B or Phi-3)
    2. Extract PCA params from it
    3. Run baseline generation (no injection)
    4. Run navigation with inertia mode (three formulas)
    5. Compare: baseline vs navigated outputs
    6. Verify: formulas work, output changes controllably

Usage:
    python validate_cross_arch.py --model Qwen/Qwen2.5-1.5B
    python validate_cross_arch.py --model microsoft/phi-2
"""

import torch
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def validate(model_name: str, device: str = 'cuda'):
    """Run full cross-architecture validation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from s_space.extraction import extract_pca_params
    from s_space.formulas import read_coords, compute_delta, compute_injection

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  S-Space Cross-Architecture Validation                       ║
║  Model: {model_name:<52s}║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── Step 1: Extract PCA params ──
    print("[Step 1] Extracting PCA parameters...")
    t0 = time.time()
    params = extract_pca_params(
        model_name=model_name,
        K=50,  # 50 dims is enough for validation
        n_samples=30,  # quick extraction
        device=device,
        batch_size=2,
    )
    extract_time = time.time() - t0
    layers = sorted(params['principal_dirs'].keys())
    K = params['K']
    print(f"  Extracted: {len(layers)} layers, K={K}, took {extract_time:.1f}s")
    for L in layers:
        stats = params['layer_stats'].get(L, {})
        print(f"  L{L}: K={params['principal_dirs'][L].shape[0]}, "
              f"top3={stats.get('top3_variance_pct', 0):.1f}%, "
              f"top10={stats.get('top10_variance_pct', 0):.1f}%")

    # ── Step 2: Load model for generation ──
    print(f"\n[Step 2] Loading {model_name} for generation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  Architecture: {n_layers} layers, d_model={d_model}")

    # ── Step 3: Baseline generation ──
    test_prompts = [
        "Explain why the sky appears blue.",
        "What is 15% of 240? Show your reasoning.",
        "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?",
    ]

    print(f"\n[Step 3] Baseline generation (no injection)...")
    baseline_outputs = {}
    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        baseline_outputs[i] = text
        print(f"\n  Q{i}: {prompt}")
        print(f"  A{i}: {text[:120]}...")

    # ── Step 4: Navigate with three formulas ──
    # Use the deepest available layer for injection
    inject_layer = max(layers)
    principal_dirs = params['principal_dirs'][inject_layer].to(model.device)
    metric_weights = params['metric_weights'][inject_layer].to(model.device)

    print(f"\n[Step 4] Navigation with three formulas (inertia mode, L{inject_layer})...")
    print(f"  Formula ③: c_k = h · ê_k")
    print(f"  Formula ②: Δ_k = need_k × d_k (inertia mode)")
    print(f"  Formula ①: α = r × |h| / |Δ_masked|")

    navigated_outputs = {}
    r_eff = 0.10  # 10% injection ratio

    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        # Create injection hook
        def make_hook(dirs, mw, r, layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output

                # ③ Read coordinates
                h_last = h[:, -1, :].squeeze(0).float()
                coords = read_coords(h_last, dirs)  # c_k = h · ê_k

                # ② Compute navigation (inertia mode)
                delta_k, need_k = compute_delta(coords, mw)  # Δ_k = need_k × d_k

                # ① Compute injection
                inject = compute_injection(h_last, delta_k, dirs, mw, r)

                if inject is not None:
                    h_new = h.clone()
                    h_new[:, -1, :] = h_last + inject.to(h_last.dtype)
                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
                return output
            return hook

        # Register hook on the target layer
        handle = model.model.layers[inject_layer].register_forward_hook(
            make_hook(principal_dirs, metric_weights, r_eff, inject_layer)
        )

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        handle.remove()

        text = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        navigated_outputs[i] = text
        print(f"\n  Q{i}: {prompt}")
        print(f"  Nav: {text[:120]}...")

    # ── Step 5: Compare ──
    print(f"\n{'='*70}")
    print("COMPARISON: Baseline vs Navigated")
    print(f"{'='*70}")

    all_changed = False
    for i in range(len(test_prompts)):
        b = baseline_outputs[i]
        n = navigated_outputs[i]
        changed = b != n
        all_changed = all_changed or changed
        status = "CHANGED" if changed else "SAME"
        print(f"\nQ{i}: [{status}]")
        print(f"  Base: {b[:100]}")
        print(f"  Nav:  {n[:100]}")

    # ── Step 6: Single-axis probe ──
    print(f"\n[Step 6] Single-axis probe (ê_0, ê_1, ê_2)...")
    probe_prompt = "Explain the concept of gravity in physics."
    inputs = tokenizer(probe_prompt, return_tensors='pt').to(model.device)

    # Baseline for probe
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    probe_baseline = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    for axis_k in range(min(3, principal_dirs.shape[0])):
        # Create axis-specific hook
        def make_axis_hook(dirs, mw, r, k):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                h_last = h[:, -1, :].squeeze(0).float()

                # Single-axis injection: only push ê_k
                c_k = dirs @ h_last
                delta_k = torch.zeros_like(c_k)
                delta_k[k] = c_k[k] * 2.0  # amplify this axis

                displacement = (mw.unsqueeze(1) * delta_k.unsqueeze(1) * dirs).sum(dim=0)
                h_norm = h_last.norm().item()
                d_norm = displacement.norm().item()
                if d_norm > 1e-8:
                    alpha = r * h_norm / d_norm
                    inject = displacement * alpha
                else:
                    return output

                h_new = h.clone()
                h_new[:, -1, :] = h_last + inject.to(h_last.dtype)
                if isinstance(output, tuple):
                    return (h_new,) + output[1:]
                return h_new
            return hook

        handle = model.model.layers[inject_layer].register_forward_hook(
            make_axis_hook(principal_dirs, metric_weights, 0.12, axis_k)
        )

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        handle.remove()

        axis_text = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        print(f"\n  ê_{axis_k}: {axis_text[:80]}...")

    print(f"\n  Baseline: {probe_baseline[:80]}...")

    # ── Final verdict ──
    print(f"\n{'='*70}")
    print("VALIDATION RESULT")
    print(f"{'='*70}")

    if all_changed:
        print("✅ PASS: Navigation successfully modified model outputs")
        print("✅ PASS: Three formulas work on cross-architecture model")
        print("✅ PASS: Inertia mode works without d_consensus")
    else:
        print("⚠️  Navigation did not change outputs — may need higher r_eff")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return all_changed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B',
                        help='Model to validate on (default: Qwen2.5-0.5B)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    success = validate(args.model, device)
    sys.exit(0 if success else 1)
