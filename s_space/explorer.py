"""
Single-Knob Axis Explorer — Discover what each axis does

Automatically probes each S-space axis by injecting positive and negative
shifts, then comparing model outputs to baseline. Produces a report
mapping each axis to its semantic effect.

This is the "scientific instrument" for S-space: use it to discover
what each ê_k controls in any model.

Usage:
    from s_space.explorer import AxisExplorer

    explorer = AxisExplorer(model_name="Qwen/Qwen2.5-0.5B")
    report = explorer.explore_axis(axis=2, prompt="Explain gravity")
    print(report)

    # Full exploration
    results = explorer.explore_all(axes=range(10))
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AxisReport:
    """Report for a single axis exploration."""
    axis: int
    prompt: str
    baseline_output: str
    positive_output: str
    negative_output: str
    r_eff: float
    positive_delta: Optional[str] = None  # brief description of change
    negative_delta: Optional[str] = None


@dataclass
class ExplorationResult:
    """Result from exploring multiple axes."""
    model_name: str
    layer: int
    axes_tested: List[int]
    reports: Dict[int, AxisReport] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Axis Exploration: {self.model_name}, Layer {self.layer}"]
        lines.append("=" * 60)
        for k in sorted(self.reports.keys()):
            r = self.reports[k]
            lines.append(f"\nê_{k}:")
            lines.append(f"  Baseline: {r.baseline_output[:80]}...")
            lines.append(f"  Positive: {r.positive_output[:80]}...")
            lines.append(f"  Negative: {r.negative_output[:80]}...")
        return "\n".join(lines)


class AxisExplorer:
    """Explore S-space axes by single-knob injection.

    For each axis ê_k, injects positive and negative shifts to see
    how model output changes. This reveals the semantic function
    of each axis.

    Args:
        model_name: HuggingFace model name or path
        params_path: Path to PCA params (auto-extracted if None)
        layer: Which layer to inject at (default: auto-detect)
        device: Device to use
        max_new_tokens: Max tokens to generate
    """

    # Probing prompts designed to be sensitive to different axes
    PROBE_PROMPTS = [
        "Explain why the sky appears blue to a human observer.",
        "What is the relationship between mass and gravity?",
        "Describe the process of photosynthesis in detail.",
        "If a tree falls in a forest and no one is around, does it make a sound?",
        "What causes rain to fall from clouds?",
    ]

    def __init__(
        self,
        model_name: str,
        params_path: Optional[str] = None,
        layer: Optional[int] = None,
        device: str = 'auto',
        max_new_tokens: int = 80,
    ):
        self.model_name = model_name
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_new_tokens = max_new_tokens

        # Load model
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("pip install transformers")

        logger.info(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map=self.device if self.device == 'cuda' else None,
        )
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        self.model.eval()

        # Get PCA params
        if params_path:
            self.params = torch.load(params_path, map_location='cpu', weights_only=False)
        else:
            # Auto-extract
            from s_space.extraction import extract_pca_params
            logger.info("No params file provided, auto-extracting...")
            self.params = extract_pca_params(
                model_name=model_name,
                layers=[layer] if layer else None,
                n_samples=50,
                device=self.device,
            )

        self.principal_dirs = self.params['principal_dirs']
        self.metric_weights = self.params['metric_weights']

        # Select injection layer
        if layer is not None:
            self.layer = layer
        else:
            # Use the deepest available layer (most semantic)
            self.layer = max(self.principal_dirs.keys())

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _generate(self, prompt: str, hidden_hook=None) -> str:
        """Generate text from prompt, optionally with injection hook."""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

        if hidden_hook is not None:
            # Register hook for injection
            handle = self.model.model.layers[self.layer].register_forward_hook(hidden_hook)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        if hidden_hook is not None:
            handle.remove()

        # Decode only new tokens
        input_len = inputs['input_ids'].shape[1]
        new_tokens = outputs[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _make_injection_hook(self, axis: int, sign: float, r_eff: float):
        """Create a forward hook that injects along a single axis."""
        dirs = self.principal_dirs[self.layer].to(self.model.device)  # (K, d_model)
        g_k = self.metric_weights[self.layer].to(self.model.device)   # (K,)

        def hook(module, input, output):
            # output is a tuple for decoder layers, hidden state is output[0]
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            # Only inject on last token position
            h_last = h[:, -1, :].squeeze(0)  # (d_model,)

            # Read coordinate
            c_k = dirs @ h_last.float()  # (K,)

            # Create single-axis delta
            delta_k = torch.zeros_like(c_k)
            delta_k[axis] = sign * r_eff * abs(c_k[axis]).clamp(min=0.1)

            # Map back to d_model
            displacement = (g_k.unsqueeze(1) * delta_k.unsqueeze(1) * dirs).sum(dim=0)

            # Scale to desired magnitude
            h_norm = h_last.float().norm().item()
            d_norm = displacement.norm().item()
            if d_norm > 1e-8:
                alpha = r_eff * h_norm / d_norm
                inject = displacement * alpha
            else:
                inject = torch.zeros_like(h_last)

            # Apply injection
            h_new = h.clone()
            h_new[:, -1, :] = h_last + inject.to(h_last.dtype)

            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        return hook

    def explore_axis(
        self,
        axis: int,
        prompt: Optional[str] = None,
        r_eff: float = 0.15,
    ) -> AxisReport:
        """Explore a single axis with positive and negative injection.

        Args:
            axis: Which axis to probe (ê_k)
            prompt: Text prompt (default: first PROBE_PROMPT)
            r_eff: Injection ratio

        Returns:
            AxisReport with baseline, positive, negative outputs
        """
        if prompt is None:
            prompt = self.PROBE_PROMPTS[0]

        logger.info(f"Exploring ê_{axis}...")

        # Baseline
        baseline = self._generate(prompt)

        # Positive shift
        hook_pos = self._make_injection_hook(axis, sign=+1.0, r_eff=r_eff)
        positive = self._generate(prompt, hidden_hook=hook_pos)

        # Negative shift
        hook_neg = self._make_injection_hook(axis, sign=-1.0, r_eff=r_eff)
        negative = self._generate(prompt, hidden_hook=hook_neg)

        report = AxisReport(
            axis=axis,
            prompt=prompt,
            baseline_output=baseline,
            positive_output=positive,
            negative_output=negative,
            r_eff=r_eff,
        )

        logger.info(f"  Baseline: {baseline[:60]}...")
        logger.info(f"  Positive: {positive[:60]}...")
        logger.info(f"  Negative: {negative[:60]}...")

        return report

    def explore_all(
        self,
        axes: Optional[List[int]] = None,
        prompt: Optional[str] = None,
        r_eff: float = 0.15,
    ) -> ExplorationResult:
        """Explore multiple axes.

        Args:
            axes: List of axes to explore (default: first 10)
            prompt: Text prompt for all axes
            r_eff: Injection ratio

        Returns:
            ExplorationResult with reports for all axes
        """
        if axes is None:
            K = self.principal_dirs[self.layer].shape[0]
            axes = list(range(min(10, K)))

        result = ExplorationResult(
            model_name=self.model_name,
            layer=self.layer,
            axes_tested=axes,
        )

        for k in axes:
            result.reports[k] = self.explore_axis(k, prompt=prompt, r_eff=r_eff)

        return result

    def close(self):
        """Free model memory."""
        del self.model
        if self.device == 'cuda':
            torch.cuda.empty_cache()
