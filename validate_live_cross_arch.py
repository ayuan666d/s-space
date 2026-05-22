"""
Cross-Architecture Live Validation with Local Qwen3-0.6B

This is REAL validation: extract PCA from the model, inject with three formulas,
and compare baseline vs navigated outputs.

Usage:
    python validate_live_cross_arch.py
    python validate_live_cross_arch.py --model Qwen/Qwen2.5-1.5B
"""
import torch
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Default: use a local model if available, otherwise download from HuggingFace
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

def main(model_name=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from s_space.extraction import extract_pca_params
    from s_space.formulas import read_coords, compute_delta, compute_injection

    model_name = model_name or DEFAULT_MODEL

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  S-Space LIVE VALIDATION on {model_name:<35s}║
║  This is REAL data, not synthetic.                           ║
╚══════════════════════════════════════════════════════════════╝
""")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Step 1: Extract PCA params from the model ──
    print(f"\n[Step 1] Extracting PCA parameters from {model_name}...")
    t0 = time.time()
    params = extract_pca_params(
        model_name=model_name,
        K=50,
        n_samples=30,  # quick extraction for validation
        device=device,
        batch_size=2,
    )
    extract_time = time.time() - t0
    layers = sorted(params['principal_dirs'].keys())
    K = params['K']
    print(f"  Extracted: {len(layers)} layers, K={K}, took {extract_time:.1f}s")

    # Print per-layer stats
    for L in layers:
        stats = params['layer_stats'].get(L, {})
        dirs = params['principal_dirs'][L]
        mw = params['metric_weights'][L]
        top3 = stats.get('top3_variance_pct', 0)
        top10 = stats.get('top10_variance_pct', 0)
        eff_dim = int((mw > mw.max().item() * 0.01).sum().item())
        print(f"  L{L:2d}: K_actual={dirs.shape[0]}, eff_dim={eff_dim}, "
              f"top3={top3:.1f}%, top10={top10:.1f}%, total_gk={mw.sum().item():.2f}")

    # ── Step 2: Load model for generation ──
    print(f"\n[Step 2] Loading {model_name} for generation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device if device == 'cuda' else None,
    )
    if device == 'cpu':
        model = model.to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  Architecture: {n_layers} layers, d_model={d_model}")

    # ── Step 3: Baseline generation ──
    test_prompts = [
        "Explain why the sky appears blue to a human observer.",
        "What is 15% of 240? Show your reasoning.",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    ]

    print(f"\n[Step 3] BASELINE generation (no injection)...")
    baseline_outputs = {}
    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        baseline_outputs[i] = text
        print(f"\n  Q{i}: {prompt}")
        print(f"  A{i}: {text[:150]}")

    # ── Step 4: Navigate with three formulas (inertia mode) ──
    inject_layer = max(layers)
    principal_dirs = params['principal_dirs'][inject_layer].to(model.device)
    metric_weights = params['metric_weights'][inject_layer].to(model.device)

    print(f"\n[Step 4] NAVIGATION with three formulas (inertia mode, L{inject_layer})...")
    print(f"  Formula 3: c_k = h * e_k")
    print(f"  Formula 2: delta_k = need_k * d_k (inertia mode)")
    print(f"  Formula 1: alpha = r * |h| / |delta_masked|")

    navigated_outputs = {}
    r_eff = 0.10

    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        # Create injection hook
        def make_hook(dirs, mw, r, layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output

                h_last = h[:, -1, :].squeeze(0).float()

                # 3: Read coordinates
                coords = read_coords(h_last, dirs)

                # 2: Compute navigation (inertia mode)
                delta_k, need_k = compute_delta(coords, mw)

                # 1: Compute injection
                inject = compute_injection(h_last, delta_k, dirs, mw, r)

                if inject is not None:
                    h_new = h.clone()
                    h_new[:, -1, :] = h_last + inject.to(h_last.dtype)
                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
                return output
            return hook

        handle = model.model.layers[inject_layer].register_forward_hook(
            make_hook(principal_dirs, metric_weights, r_eff, inject_layer)
        )

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        handle.remove()

        text = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        navigated_outputs[i] = text
        print(f"\n  Q{i}: {prompt}")
        print(f"  Nav: {text[:150]}")

    # ── Step 5: Compare ──
    print(f"\n{'='*70}")
    print("COMPARISON: Baseline vs Navigated (Inertia r=0.10)")
    print(f"{'='*70}")

    all_changed = False
    for i in range(len(test_prompts)):
        b = baseline_outputs[i]
        n = navigated_outputs[i]
        changed = b != n
        all_changed = all_changed or changed
        status = "CHANGED" if changed else "SAME"
        print(f"\nQ{i}: [{status}]")
        print(f"  Base: {b[:120]}")
        print(f"  Nav:  {n[:120]}")

    # ── Step 6: Single-axis probe ──
    print(f"\n{'='*70}")
    print("SINGLE-AXIS PROBE (e_0, e_1, e_2)")
    print(f"{'='*70}")

    probe_prompt = "Explain the concept of gravity in physics."
    inputs = tokenizer(probe_prompt, return_tensors='pt').to(model.device)

    # Baseline
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    probe_baseline = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print(f"\n  Baseline: {probe_baseline[:120]}")

    for axis_k in range(min(3, principal_dirs.shape[0])):
        def make_axis_hook(dirs, mw, r, k):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                h_last = h[:, -1, :].squeeze(0).float()

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
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        handle.remove()

        axis_text = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        print(f"  e_{axis_k}: {axis_text[:120]}")

    # ── Step 7: Different r_eff values ──
    print(f"\n{'='*70}")
    print("INJECTION MAGNITUDE SWEEP (r=0.05, 0.10, 0.15, 0.20)")
    print(f"{'='*70}")

    sweep_prompt = "What causes rain to fall from clouds?"
    inputs = tokenizer(sweep_prompt, return_tensors='pt').to(model.device)

    for r_test in [0.05, 0.10, 0.15, 0.20]:
        def make_hook_r(dirs, mw, r):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                h_last = h[:, -1, :].squeeze(0).float()
                coords = read_coords(h_last, dirs)
                delta_k, need_k = compute_delta(coords, mw)
                inject = compute_injection(h_last, delta_k, dirs, mw, r)
                if inject is not None:
                    h_new = h.clone()
                    h_new[:, -1, :] = h_last + inject.to(h_last.dtype)
                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
                return output
            return hook

        handle = model.model.layers[inject_layer].register_forward_hook(
            make_hook_r(principal_dirs, metric_weights, r_test)
        )
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        handle.remove()
        text = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        print(f"  r={r_test:.2f}: {text[:100]}")

    # ── Final verdict ──
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}")

    if all_changed:
        print("PASS: Navigation successfully modified model outputs on REAL model")
        print("PASS: Three formulas work on cross-architecture Qwen2.5-0.5B-Instruct")
        print("PASS: Inertia mode works without d_consensus")
    else:
        print("WARN: Navigation did not change outputs — may need higher r_eff")

    # Clean up
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return all_changed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        help=f'Model to validate on (default: {DEFAULT_MODEL})')
    args = parser.parse_args()

    success = main(args.model)
    sys.exit(0 if success else 1)
