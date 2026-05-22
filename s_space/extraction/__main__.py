"""
CLI for S-Space PCA Parameter Extraction

Usage:
    python -m s_space.extraction --model Qwen/Qwen2.5-0.5B
    python -m s_space.extraction --model Qwen/Qwen2.5-7B --K 100 --layers 3,16,19,22
    python -m s_space.extraction --model ./local_model --output my_params.pt --n-samples 200
"""

import argparse
import logging
import sys

from s_space.extraction import extract_pca_params, save_params


def main():
    parser = argparse.ArgumentParser(
        description='Extract S-space PCA parameters from a HuggingFace Transformer model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from a small model (fast)
  python -m s_space.extraction --model Qwen/Qwen2.5-0.5B

  # Extract from a 7B model with custom settings
  python -m s_space.extraction --model Qwen/Qwen2.5-7B --K 100 --n-samples 200

  # Extract from a local model with specific layers
  python -m s_space.extraction --model ./my_model --layers 3,16,19,22 --output my_params.pt

  # Use GPU if available
  python -m s_space.extraction --model Qwen/Qwen2.5-1.5B --device cuda --batch-size 8
        """,
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='HuggingFace model name or local path (e.g., Qwen/Qwen2.5-7B)',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output .pt file path (default: pca_params_<model_name>.pt)',
    )
    parser.add_argument(
        '--K',
        type=int,
        default=100,
        help='Target PCA dimension (default: 100)',
    )
    parser.add_argument(
        '--layers', '-l',
        type=str,
        default=None,
        help='Comma-separated layer indices (default: all layers)',
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of hidden state samples to collect (default: 100)',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        help='Max token length per prompt (default: 128)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for inference (default: 4)',
    )
    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=0.95,
        help='Cumulative variance threshold for K selection (default: 0.95)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging',
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    # Parse layers
    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(',')]

    # Default output path
    output = args.output
    if output is None:
        model_safe = args.model.replace('/', '_').replace('\\', '_')
        output = f"pca_params_{model_safe}.pt"

    print(f"""
╔══════════════════════════════════════════════╗
║    S-Space PCA Parameter Extraction          ║
╠══════════════════════════════════════════════╣
║  Model:      {args.model:<33s}║
║  K target:   {args.K:<33d}║
║  N samples:  {args.n_samples:<33d}║
║  Layers:     {str(layers or 'all'):<33s}║
║  Device:     {args.device:<33s}║
║  Output:     {output:<33s}║
╚══════════════════════════════════════════════╝
""")

    # Extract
    params = extract_pca_params(
        model_name=args.model,
        layers=layers,
        K=args.K,
        n_samples=args.n_samples,
        max_length=args.max_length,
        device=args.device,
        batch_size=args.batch_size,
        variance_threshold=args.variance_threshold,
    )

    # Save
    save_params(params, output)

    # Print summary
    print(f"""
╔══════════════════════════════════════════════╗
║    Extraction Complete                        ║
╠══════════════════════════════════════════════╣""")

    for L in sorted(params['principal_dirs'].keys()):
        K = params['principal_dirs'][L].shape[0]
        stats = params['layer_stats'].get(L, {})
        top3 = stats.get('top3_variance_pct', 0)
        top10 = stats.get('top10_variance_pct', 0)
        print(f"║  L{L:2d}: K={K:3d}, top3={top3:.1f}%, top10={top10:.1f}%")

    print(f"""╠══════════════════════════════════════════════╣
║  Now use it:                                 ║
║  nav = CoordNavigator(params_path="{output}")  ║
╚══════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    main()
