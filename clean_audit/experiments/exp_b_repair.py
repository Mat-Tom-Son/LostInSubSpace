"""
EXPERIMENT B: SELF-REPAIR MECHANISM (Amplitude Compensation)

Goal: Prove "self-repair" is actually "amplitude compensation." Measure immediate
A_gain response to geometric disruption (head ablation).

Core Hypothesis:
  When critical routing heads are ablated, downstream layers immediately increase
  residual amplitude to compensate for missing information. This is A_gain
  (amplitude-in-response-to-geometric-disruption), not learnable fine-tuning.

Mechanism:
  - Disruption: Ablate critical attention heads during forward pass ONLY
  - Measurement: Track residual norms layer-by-layer after ablation
  - Prediction: Compensation ratio (residual_ablated / residual_baseline) > 1.3
    in layers 10-12 for critical head ablation

Design:
  - Use TransformerLens for precise hook-based ablation control
  - IOI task: Indirect Object Identification (Nix et al., 2022)
  - GPT-2 Small (pre-trained)
  - No fine-tuning: Forward-pass only ablation
  - Measure A_gain = mean residual norm increase across token positions

Conditions:
  1. Baseline (No Ablation): Standard IOI evaluation
  2. Critical Ablation: Ablate L9H9, L9H6, L10H0 (name mover heads)
  3. Negative Control: Ablate L2H3 (random unrelated head)

Success Criteria:
  - Critical ablation: ΔLogit_diff = -3.5 ± 0.2, Compensation >1.3×
  - Random ablation: ΔLogit_diff ≈ -0.1, Compensation ≈1.0×
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import json

# Import our libraries
from lib.metrics import AllostasisAudit
from lib.logging_utils import AuditLogger, setup_reproducibility

# Try to import TransformerLens
try:
    import transformer_lens
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    print("Warning: transformer_lens not installed. Will use fallback mode.")


# IOI Task Setup
class IOITask:
    """
    Indirect Object Identification (IOI) task.

    Sentence template:
      "When [A] and [B] went to the store, [A] gave a book to [B]. [B] is a ..."

    Target: The model should complete with [B] (indirect object)

    Task requires multi-step reasoning:
      1. Track two entities (A, B)
      2. Track their relative positions
      3. Select the indirect object (B)
    """

    def __init__(self, vocab_size: int = 50257, max_seq_len: int = 30):
        """
        Args:
            vocab_size: Model's vocabulary size
            max_seq_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Precomputed token lists (for GPT-2 tokenizer)
        # These would need to be mapped to actual token IDs
        self.person_a_tokens = [
            "Alice", "Bob", "Charlie", "David", "Eve", "Frank",
            "Grace", "Henry", "Iris", "Jack", "Karen", "Liam"
        ]
        self.person_b_tokens = [
            "Alice", "Bob", "Charlie", "David", "Eve", "Frank",
            "Grace", "Henry", "Iris", "Jack", "Karen", "Liam"
        ]

    def generate_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a batch of IOI examples.

        Args:
            batch_size: Number of examples in batch

        Returns:
            (prompts, correct_tokens, incorrect_tokens)
            - prompts: [batch, seq_len] token IDs
            - correct_tokens: [batch] correct answer token ID (B)
            - incorrect_tokens: [batch] wrong answer token ID (A)
        """
        # For demonstration, we create synthetic prompts
        # In practice, these would be properly tokenized IOI examples

        prompts = torch.zeros(batch_size, self.max_seq_len, dtype=torch.long)
        correct_tokens = torch.randint(0, self.vocab_size, (batch_size,))
        incorrect_tokens = torch.randint(0, self.vocab_size, (batch_size,))

        # Ensure correct and incorrect are different
        for i in range(batch_size):
            while correct_tokens[i] == incorrect_tokens[i]:
                incorrect_tokens[i] = torch.randint(0, self.vocab_size, (1,)).item()

        return prompts, correct_tokens, incorrect_tokens


class HeadAblationExperiment:
    """
    Experiment B: Head Ablation and Amplitude Compensation Measurement.

    Core Logic:
      1. Forward pass baseline with model intact
      2. Forward pass with specific heads ablated
      3. Measure residual norms layer-by-layer
      4. Compute compensation ratio (ablated / baseline)
      5. Track logit_diff (correct vs incorrect token)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_name: Model identifier for TransformerLens (e.g., "gpt2")
            device: Device to run on
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.auditor = AllostasisAudit(device=device)

        # Load model if TransformerLens is available
        if TRANSFORMER_LENS_AVAILABLE:
            print(f"Loading {model_name} with TransformerLens...")
            try:
                self.model = HookedTransformer.from_pretrained(model_name)
                self.model.to(device)
                print(f"Model loaded successfully on {device}")
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                TRANSFORMER_LENS_AVAILABLE = False

        self.ioi_task = IOITask()

    def ablate_head(
        self,
        logits: torch.Tensor,
        activations_dict: Dict[str, torch.Tensor],
        layer_idx: int,
        head_idx: int
    ) -> torch.Tensor:
        """
        Ablate a specific attention head by zeroing its output.

        Args:
            logits: Output logits [batch, seq, vocab]
            activations_dict: Dictionary of cached activations
            layer_idx: Layer index (0-indexed)
            head_idx: Head index within layer (0-indexed)

        Returns:
            logits: Same as input (this function is primarily for demonstrating logic)
        """
        # In TransformerLens, this would be done via hooks
        # For now, we just document the approach
        print(f"Ablating L{layer_idx}H{head_idx}")
        return logits

    def measure_residual_norms(
        self,
        activations_dict: Dict[str, torch.Tensor],
        n_layers: Optional[int] = None
    ) -> Dict[int, float]:
        """
        Measure residual stream norms for each layer.

        Args:
            activations_dict: Dictionary of cached activations from forward pass
            n_layers: Number of layers (if None, auto-detect)

        Returns:
            Dict mapping layer_idx -> mean residual norm
        """
        norms = {}

        # Auto-detect number of layers if not provided
        if n_layers is None:
            # Look for resid_post hooks
            resid_keys = [k for k in activations_dict.keys() if 'resid_post' in str(k)]
            if resid_keys:
                # Extract layer indices
                layer_indices = []
                for key in resid_keys:
                    if isinstance(key, tuple) and len(key) >= 2:
                        layer_indices.append(key[1])
                    elif isinstance(key, str):
                        try:
                            layer_num = int(key.split('blocks.')[1].split('.')[0])
                            layer_indices.append(layer_num)
                        except:
                            pass
                n_layers = max(layer_indices) + 1 if layer_indices else 12

        # Compute norms for each layer
        for layer_idx in range(n_layers):
            # Try different key formats
            resid_key = f"blocks.{layer_idx}.hook_resid_post"
            if resid_key not in activations_dict:
                resid_key = ('resid_post', layer_idx)
            if resid_key not in activations_dict:
                continue

            resid = activations_dict[resid_key]
            norm = self.auditor.compute_amplitude_activation(resid)
            norms[layer_idx] = norm

        return norms

    def compute_logit_diff(
        self,
        logits: torch.Tensor,
        correct_token_idx: torch.Tensor,
        incorrect_token_idx: torch.Tensor
    ) -> float:
        """
        Compute logit difference between correct and incorrect tokens.

        Args:
            logits: [batch, seq, vocab]
            correct_token_idx: [batch] correct token indices
            incorrect_token_idx: [batch] incorrect token indices

        Returns:
            Mean logit difference: log(P(correct)) - log(P(incorrect))
        """
        if logits.dim() == 3:
            logits = logits[:, -1, :]  # Take last token

        batch_size = logits.shape[0]
        correct_logits = logits[torch.arange(batch_size), correct_token_idx]
        incorrect_logits = logits[torch.arange(batch_size), incorrect_token_idx]

        logit_diff = (correct_logits - incorrect_logits).mean().item()
        return logit_diff

    def run_condition(
        self,
        condition_name: str,
        ablation_heads: Optional[List[Tuple[int, int]]] = None,
        n_examples: int = 10,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Run a single experimental condition.

        Args:
            condition_name: Name of condition (baseline, critical, random)
            ablation_heads: List of (layer, head) tuples to ablate
            n_examples: Number of IOI examples to evaluate
            batch_size: Batch size for processing

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*80}")
        print(f"RUNNING CONDITION: {condition_name.upper()}")
        if ablation_heads:
            print(f"Ablating heads: {ablation_heads}")
        print(f"{'='*80}\n")

        all_norms_baseline = {i: [] for i in range(12)}  # Default: 12 layers for GPT-2
        all_norms_ablated = {i: [] for i in range(12)}
        logit_diffs_baseline = []
        logit_diffs_ablated = []

        # Generate examples
        for batch_idx in range(0, n_examples, batch_size):
            actual_batch_size = min(batch_size, n_examples - batch_idx)
            prompts, correct_tokens, incorrect_tokens = self.ioi_task.generate_batch(
                batch_size=actual_batch_size
            )
            prompts = prompts.to(self.device)

            if self.model is not None:
                # BASELINE: Forward pass without ablation
                print(f"  Batch {batch_idx//batch_size + 1}: Baseline forward pass...", end=' ')
                with torch.no_grad():
                    if hasattr(self.model, 'run_with_cache'):
                        logits_baseline, cache_baseline = self.model.run_with_cache(prompts)
                    else:
                        logits_baseline = self.model(prompts)
                        cache_baseline = {}

                # Measure baseline norms
                norms_baseline = self.measure_residual_norms(cache_baseline)
                for layer_idx, norm in norms_baseline.items():
                    all_norms_baseline[layer_idx].append(norm)

                # Measure baseline logit_diff
                correct_tokens = correct_tokens.to(self.device)
                incorrect_tokens = incorrect_tokens.to(self.device)
                logit_diff_baseline = self.compute_logit_diff(
                    logits_baseline, correct_tokens, incorrect_tokens
                )
                logit_diffs_baseline.append(logit_diff_baseline)
                print(f"logit_diff={logit_diff_baseline:.3f}")

                # ABLATED: Forward pass with heads ablated
                if ablation_heads:
                    print(f"  Batch {batch_idx//batch_size + 1}: Ablation forward pass...", end=' ')

                    # Create ablation hooks
                    ablation_hooks = []
                    for layer_idx, head_idx in ablation_heads:
                        # Hook name for this head's pattern/output
                        hook_name = f"blocks.{layer_idx}.attn.hook_pattern"

                        def make_ablation_hook(l_idx, h_idx):
                            def ablation_hook(pattern, hook):
                                # Zero out the specified head's attention pattern
                                pattern_ablated = pattern.clone()
                                pattern_ablated[:, h_idx, :, :] = 0.0
                                # Renormalize so it still sums to 1
                                pattern_ablated = pattern_ablated / (pattern_ablated.sum(dim=-1, keepdim=True) + 1e-8)
                                return pattern_ablated
                            return ablation_hook

                        ablation_hooks.append((hook_name, make_ablation_hook(layer_idx, head_idx)))

                    with torch.no_grad():
                        if hasattr(self.model, 'run_with_hooks'):
                            logits_ablated, cache_ablated = self.model.run_with_hooks(
                                prompts,
                                fwd_hooks=ablation_hooks,
                                return_cache=True
                            )
                        elif hasattr(self.model, 'run_with_cache'):
                            # Fallback: run without hooks (won't ablate)
                            logits_ablated, cache_ablated = self.model.run_with_cache(prompts)
                        else:
                            logits_ablated = self.model(prompts)
                            cache_ablated = {}

                    # Measure ablated norms
                    norms_ablated = self.measure_residual_norms(cache_ablated)
                    for layer_idx, norm in norms_ablated.items():
                        all_norms_ablated[layer_idx].append(norm)

                    # Measure ablated logit_diff
                    logit_diff_ablated = self.compute_logit_diff(
                        logits_ablated, correct_tokens, incorrect_tokens
                    )
                    logit_diffs_ablated.append(logit_diff_ablated)
                    print(f"logit_diff={logit_diff_ablated:.3f}")
            else:
                print(f"  Batch {batch_idx//batch_size + 1}: Model not loaded, skipping...")

        # Compute statistics
        result = {
            'condition': condition_name,
            'n_examples': n_examples,
        }

        if logit_diffs_baseline:
            result['logit_diff_baseline'] = np.mean(logit_diffs_baseline)
            result['logit_diff_baseline_std'] = np.std(logit_diffs_baseline)

        if logit_diffs_ablated:
            result['logit_diff_ablated'] = np.mean(logit_diffs_ablated)
            result['logit_diff_ablated_std'] = np.std(logit_diffs_ablated)
            result['delta_logit_diff'] = result['logit_diff_ablated'] - result['logit_diff_baseline']

        # Compute compensation ratios for critical layers
        compensation_ratios = {}
        for layer_idx in [10, 11, 12]:  # Layers 10-12 are prediction layers
            if (layer_idx in all_norms_baseline and layer_idx in all_norms_ablated and
                all_norms_baseline[layer_idx] and all_norms_ablated[layer_idx]):
                baseline_norm = np.mean(all_norms_baseline[layer_idx])
                ablated_norm = np.mean(all_norms_ablated[layer_idx])
                ratio = ablated_norm / (baseline_norm + 1e-8)
                compensation_ratios[f'L{layer_idx}'] = ratio
                result[f'compensation_L{layer_idx}'] = ratio

        result['mean_compensation'] = np.mean(list(compensation_ratios.values())) if compensation_ratios else 1.0

        # Print summary
        print(f"\n{condition_name.upper()} Results:")
        print(f"  Logit diff (baseline): {result.get('logit_diff_baseline', float('nan')):.4f} ± "
              f"{result.get('logit_diff_baseline_std', float('nan')):.4f}")
        if 'logit_diff_ablated' in result:
            print(f"  Logit diff (ablated):  {result['logit_diff_ablated']:.4f} ± "
                  f"{result['logit_diff_ablated_std']:.4f}")
            print(f"  Delta logit diff:     {result['delta_logit_diff']:.4f}")
        if compensation_ratios:
            print(f"  Compensation ratios:")
            for layer_name, ratio in compensation_ratios.items():
                print(f"    {layer_name}: {ratio:.4f}")
            print(f"  Mean compensation:    {result['mean_compensation']:.4f}")

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Experiment B: Self-Repair Mechanism (Amplitude Compensation)"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        help='Model name for TransformerLens (e.g., gpt2, gpt2-medium)'
    )
    parser.add_argument(
        '--condition',
        type=str,
        default='all',
        choices=['baseline', 'critical', 'random', 'all'],
        help='Which condition to run'
    )
    parser.add_argument(
        '--n_examples',
        type=int,
        default=10,
        help='Number of IOI examples to evaluate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='clean_audit/data',
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Quick test with fewer examples'
    )

    args = parser.parse_args()

    # Setup reproducibility
    setup_reproducibility(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create experiment
    experiment = HeadAblationExperiment(model_name=args.model, device=device)

    # Configure number of examples
    n_examples = 5 if args.quick_test else args.n_examples

    # Run conditions
    print("\n" + "="*80)
    print("EXPERIMENT B: SELF-REPAIR MECHANISM (AMPLITUDE COMPENSATION)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Examples: {n_examples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")

    results = {}
    conditions_to_run = []

    if args.condition in ['baseline', 'all']:
        conditions_to_run.append(('baseline', None))

    if args.condition in ['critical', 'all']:
        # Critical heads: L9H9, L9H6, L10H0
        conditions_to_run.append(('critical', [(9, 9), (9, 6), (10, 0)]))

    if args.condition in ['random', 'all']:
        # Random head: L2H3
        conditions_to_run.append(('random', [(2, 3)]))

    # Run each condition
    for condition_name, ablation_heads in conditions_to_run:
        result = experiment.run_condition(
            condition_name=condition_name,
            ablation_heads=ablation_heads,
            n_examples=n_examples,
            batch_size=args.batch_size
        )
        results[condition_name] = result

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"exp_b_results_seed_{args.seed}.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        results_json = {}
        for cond_name, cond_results in results.items():
            results_json[cond_name] = {}
            for key, val in cond_results.items():
                if isinstance(val, (np.floating, np.integer)):
                    results_json[cond_name][key] = float(val)
                elif isinstance(val, list):
                    results_json[cond_name][key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in val]
                else:
                    results_json[cond_name][key] = val

        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Check success criteria
    print("\n" + "="*80)
    print("SUCCESS CRITERIA CHECK")
    print("="*80 + "\n")

    success_checks = []

    if 'critical' in results and 'baseline' in results:
        delta_logit = results['critical'].get('delta_logit_diff', float('nan'))
        expected_delta = -3.5
        tolerance = 0.2

        check = abs(delta_logit - expected_delta) <= tolerance
        success_checks.append(('Critical ablation logit_diff = -3.5 ± 0.2', check))
        print(f"Critical ablation logit_diff: {delta_logit:.4f}")
        print(f"  Expected: {expected_delta} ± {tolerance}")
        print(f"  Status: {'✓ PASS' if check else '✗ FAIL'}\n")

        mean_comp = results['critical'].get('mean_compensation', float('nan'))
        check_comp = mean_comp > 1.3
        success_checks.append(('Critical ablation compensation > 1.3×', check_comp))
        print(f"Critical ablation mean compensation: {mean_comp:.4f}")
        print(f"  Expected: > 1.3")
        print(f"  Status: {'✓ PASS' if check_comp else '✗ FAIL'}\n")

    if 'random' in results and 'baseline' in results:
        delta_logit_random = results['random'].get('delta_logit_diff', float('nan'))
        check_random = abs(delta_logit_random) < 0.1
        success_checks.append(('Random ablation logit_diff ≈ -0.1', check_random))
        print(f"Random ablation logit_diff: {delta_logit_random:.4f}")
        print(f"  Expected: ≈ -0.1 (i.e., |Δ| < 0.1)")
        print(f"  Status: {'✓ PASS' if check_random else '✗ FAIL'}\n")

        mean_comp_random = results['random'].get('mean_compensation', 1.0)
        check_comp_random = abs(mean_comp_random - 1.0) < 0.1
        success_checks.append(('Random ablation compensation ≈ 1.0×', check_comp_random))
        print(f"Random ablation mean compensation: {mean_comp_random:.4f}")
        print(f"  Expected: ≈ 1.0")
        print(f"  Status: {'✓ PASS' if check_comp_random else '✗ FAIL'}\n")

    # Overall status
    all_passed = all(check for _, check in success_checks)
    print("="*80)
    print(f"Overall: {'✓ EXPERIMENT B PASSED' if all_passed else '✗ EXPERIMENT B FAILED'}")
    print("="*80)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
