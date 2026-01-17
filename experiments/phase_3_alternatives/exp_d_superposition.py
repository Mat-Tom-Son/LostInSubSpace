"""
EXPERIMENT D: POLYSEMANTICITY AND HIGH-A REGIME (Suppressor Identification)

Goal: Link "polysemanticity" to "high-A regime" by identifying suppressor heads
that operate in high-variance, high-amplitude conditions.

Key Innovation: FUNCTIONAL DEFINITION OF SUPPRESSORS
  - NOT circular logic (use variance to define, then measure variance)
  - CORRECT: Define suppressors via ABLATION
  - A suppressor increases downstream variance when removed
  - If var_ablated > var_baseline * 1.5, it's a suppressor

Resource: Small Transformer (TinyStories or ARENA toy model).
Minimum: >=1000 diverse token examples.

Implementation Plan:
1. Load model and data
2. For each head in layer 0:
   a. Measure baseline downstream variance
   b. Ablate the head (zero out contributions)
   c. Measure ablated variance
   d. If ablated_var > baseline_var * 1.5 → mark as suppressor
3. Cross-validate suppressors structurally:
   - Check anti-correlation in outputs (rho < -0.5)
4. Measure variance at three sites:
   - Suppressor heads
   - Clean heads (early layer)
   - Clean heads (late layer)
5. Compute bootstrap confidence intervals

Success Criteria:
  - Suppressor variance > 2.0x clean variance
  - Bootstrap CI excludes 1.0 with 95% confidence
  - Suppressors show anti-correlated outputs
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Dict, List, Tuple, Set
import argparse
from tqdm import tqdm
import json

# Import our libraries
from lib.metrics import AllostasisAudit
from lib.logging_utils import AuditLogger, setup_reproducibility


# ============================================================================
# DATA LOADING
# ============================================================================

class SimpleTokenDataset(Dataset):
    """Simple dataset of token sequences for variance analysis."""

    def __init__(self, vocab_size: int = 256, n_samples: int = 2000, seq_len: int = 32):
        """
        Args:
            vocab_size: Size of token vocabulary
            n_samples: Number of sequences
            seq_len: Sequence length
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        # Generate random token sequences
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Target is next token prediction
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return input_ids, target_ids


# ============================================================================
# SIMPLE TRANSFORMER MODEL
# ============================================================================

class SimpleTrans(nn.Module):
    """
    Minimal multi-layer transformer for Experiment D.

    Architecture:
      - 2 layers (we focus on layer 0)
      - 4 attention heads per layer
      - d_model = 64
      - Standard Transformer blocks with residual connections
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Multiple transformer blocks
        self.blocks = nn.ModuleList([
            self._make_block(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output layer
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # For caching intermediate activations
        self._cache = {}
        self._hooks = []

    def _make_block(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> nn.Module:
        """Create a single transformer block."""
        return nn.Sequential(
            nn.LayerNorm(d_model),
            MultiHeadAttention(d_model, n_heads, dropout),
            nn.LayerNorm(d_model),
            FeedForward(d_model, d_ff, dropout)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            input_ids: Token indices [batch, seq]

        Returns:
            logits: [batch, seq, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embedding
        x = self.token_emb(input_ids)  # [batch, seq, d_model]
        pos = self.pos_emb(torch.arange(seq_len, device=input_ids.device))  # [seq, d_model]
        x = x + pos.unsqueeze(0)

        # Transformer blocks with residual connections
        for layer_idx, block in enumerate(self.blocks):
            # Cache pre-block residual
            x_residual = x

            # Layer norm + Attention
            x_attn = block[0](x)  # LayerNorm
            x_attn = block[1](x_attn)  # MultiHeadAttention

            # Residual connection
            x = x_residual + x_attn

            # Feed-forward
            x_ff = block[2](x)  # LayerNorm
            x_ff = block[3](x_ff)  # FeedForward

            # Residual connection
            x = x_residual + x_ff

            # Cache residual for analysis
            self._cache[f'layer_{layer_idx}_residual'] = x.detach().clone()

        # Final output
        x = self.ln_final(x)
        logits = self.head(x)

        return logits

    def get_attention(self, layer_idx: int, head_idx: int) -> Optional[torch.Tensor]:
        """Get cached attention pattern."""
        key = f'layer_{layer_idx}_attn_head_{head_idx}'
        return self._cache.get(key)

    def clear_cache(self):
        """Clear activation cache."""
        self._cache = {}


class MultiHeadAttention(nn.Module):
    """Multi-head attention without PyTorch built-in."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Store attention patterns for analysis
        self._attention_patterns = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, d_model]

        Returns:
            output: [batch, seq, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Linear projections
        Q = self.W_q(x).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention: [batch, n_heads, seq, seq]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Store patterns for suppressor analysis
        self._attention_patterns['weights'] = attn_weights.detach().clone()

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, d_model)

        # Final linear projection
        output = self.W_o(context)

        return output

    def get_attention_patterns(self) -> torch.Tensor:
        """Return cached attention patterns."""
        return self._attention_patterns.get('weights')


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# SUPPRESSOR IDENTIFICATION VIA ABLATION
# ============================================================================

class SuppressorAnalyzer:
    """
    Functional suppressor identification.

    Key principle: A suppressor is identified by ABLATION, not by measuring
    variance directly. We define suppressors as heads whose removal increases
    downstream variance (var_ablated > var_baseline * 1.5).
    """

    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            model: Transformer model
            device: torch device
        """
        self.model = model
        self.device = device
        self.auditor = AllostasisAudit(device=device)

    def measure_downstream_variance(
        self,
        batch: torch.Tensor,
        layer_idx: int = 0
    ) -> float:
        """
        Measure variance in residual stream downstream of a layer.

        Args:
            batch: Input batch [batch, seq]
            layer_idx: Layer to measure after

        Returns:
            Scalar variance value
        """
        self.model.eval()
        with torch.no_grad():
            self.model.clear_cache()
            _ = self.model(batch.to(self.device))

            # Get residual after target layer
            cache_key = f'layer_{layer_idx}_residual'
            if cache_key in self.model._cache:
                residual = self.model._cache[cache_key]
            else:
                return float('nan')

            # Compute variance
            variance = self.auditor.compute_variance(residual)

        return variance

    def ablate_head(
        self,
        batch: torch.Tensor,
        layer_idx: int = 0,
        head_idx: int = 0
    ) -> float:
        """
        Measure downstream variance when a specific attention head is ablated.

        Ablation is implemented by zeroing out the head's output contribution.

        Args:
            batch: Input batch [batch, seq]
            layer_idx: Layer containing the head
            head_idx: Head index within layer

        Returns:
            Variance with head ablated
        """
        self.model.eval()

        # Create wrapper to intercept and ablate
        original_attn = None

        def ablate_hook(module, input_args, output):
            """Hook to ablate specific head output."""
            # output should be [batch, seq, d_model]
            # We need to zero out this head's contribution

            # For simplicity, we'll zero the entire head's output
            # This assumes the head's contribution is roughly 1/n_heads of output
            head_size = output.shape[-1] // self.model.n_heads
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size

            output_ablated = output.clone()
            output_ablated[:, :, start_idx:end_idx] = 0.0

            return output_ablated

        # Register hook on attention module
        layer = self.model.blocks[layer_idx]
        attn_module = layer[1]  # MultiHeadAttention is at index 1
        handle = attn_module.register_forward_hook(ablate_hook)

        try:
            with torch.no_grad():
                self.model.clear_cache()
                _ = self.model(batch.to(self.device))

                # Get residual after target layer
                cache_key = f'layer_{layer_idx}_residual'
                if cache_key in self.model._cache:
                    residual = self.model._cache[cache_key]
                else:
                    return float('nan')

                # Compute variance
                variance = self.auditor.compute_variance(residual)

        finally:
            handle.remove()

        return variance

    def identify_suppressors(
        self,
        dataloader: DataLoader,
        layer_idx: int = 0,
        variance_ratio_threshold: float = 1.5
    ) -> Tuple[Set[int], Dict[int, Dict[str, float]]]:
        """
        Identify suppressor heads via ablation across multiple batches.

        FUNCTIONAL DEFINITION (not circular):
        - For each head, measure baseline downstream variance
        - Ablate the head, measure ablated variance
        - If var_ablated > var_baseline * threshold → suppressor

        Args:
            dataloader: DataLoader with batches
            layer_idx: Which layer to analyze
            variance_ratio_threshold: Ratio threshold (default 1.5)

        Returns:
            (suppressor_head_indices, detailed_stats_dict)
        """
        print(f"\n[EXP D] Identifying suppressors in layer {layer_idx}...")

        n_heads = self.model.n_heads
        head_stats = {}

        # Collect variance measurements across all batches
        baseline_variances = {i: [] for i in range(n_heads)}
        ablated_variances = {i: [] for i in range(n_heads)}

        for batch_idx, (input_ids, _) in enumerate(tqdm(dataloader, desc="Collecting variances")):
            if batch_idx >= 20:  # Limit to 20 batches for speed
                break

            # Baseline variance (no ablation)
            baseline_var = self.measure_downstream_variance(input_ids, layer_idx)
            if not np.isnan(baseline_var):
                for head_idx in range(n_heads):
                    baseline_variances[head_idx].append(baseline_var)

            # Ablated variance for each head
            for head_idx in range(n_heads):
                ablated_var = self.ablate_head(input_ids, layer_idx, head_idx)
                if not np.isnan(ablated_var):
                    ablated_variances[head_idx].append(ablated_var)

        # Compute statistics and identify suppressors
        suppressors = set()

        for head_idx in range(n_heads):
            if not baseline_variances[head_idx] or not ablated_variances[head_idx]:
                continue

            base_var = np.mean(baseline_variances[head_idx])
            abl_var = np.mean(ablated_variances[head_idx])
            ratio = abl_var / base_var if base_var > 1e-10 else float('inf')

            is_suppressor = ratio > variance_ratio_threshold

            head_stats[head_idx] = {
                'baseline_variance': float(base_var),
                'ablated_variance': float(abl_var),
                'variance_ratio': float(ratio),
                'is_suppressor': is_suppressor,
                'n_samples': len(baseline_variances[head_idx])
            }

            if is_suppressor:
                suppressors.add(head_idx)
                print(f"  Head {head_idx}: SUPPRESSOR (ratio={ratio:.2f})")
            else:
                print(f"  Head {head_idx}: clean (ratio={ratio:.2f})")

        return suppressors, head_stats

    def measure_head_output_variance(
        self,
        batch: torch.Tensor,
        layer_idx: int = 0,
        head_idx: int = 0
    ) -> float:
        """
        Measure variance in a specific attention head's output.

        Args:
            batch: Input batch [batch, seq]
            layer_idx: Layer index
            head_idx: Head index

        Returns:
            Variance of head output
        """
        self.model.eval()

        # Store head outputs
        head_outputs = None

        def capture_head_hook(module, input_args, output):
            """Capture and extract specific head output."""
            nonlocal head_outputs

            # output is [batch, seq, d_model]
            head_size = output.shape[-1] // self.model.n_heads
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size

            head_outputs = output[:, :, start_idx:end_idx].detach().clone()
            return output

        layer = self.model.blocks[layer_idx]
        attn_module = layer[1]
        handle = attn_module.register_forward_hook(capture_head_hook)

        try:
            with torch.no_grad():
                _ = self.model(batch.to(self.device))

                if head_outputs is not None:
                    variance = self.auditor.compute_variance(head_outputs)
                else:
                    variance = float('nan')

        finally:
            handle.remove()

        return variance

    def compute_head_correlations(
        self,
        batch: torch.Tensor,
        layer_idx: int = 0
    ) -> np.ndarray:
        """
        Compute correlation matrix of head outputs.

        For suppressors, we expect anti-correlations (rho < -0.5).

        Args:
            batch: Input batch [batch, seq]
            layer_idx: Layer index

        Returns:
            Correlation matrix [n_heads, n_heads]
        """
        self.model.eval()

        n_heads = self.model.n_heads
        head_outputs = {}

        # Capture all head outputs
        def capture_heads_hook(module, input_args, output):
            """Capture all head outputs."""
            head_size = output.shape[-1] // n_heads
            for h in range(n_heads):
                start_idx = h * head_size
                end_idx = (h + 1) * head_size
                output_flat = output[:, :, start_idx:end_idx].reshape(-1)
                head_outputs[h] = output_flat.detach().cpu().numpy()
            return output

        layer = self.model.blocks[layer_idx]
        attn_module = layer[1]
        handle = attn_module.register_forward_hook(capture_heads_hook)

        try:
            with torch.no_grad():
                _ = self.model(batch.to(self.device))

                # Compute correlation matrix
                if len(head_outputs) == n_heads:
                    output_matrix = np.array([head_outputs[i] for i in range(n_heads)])
                    corr_matrix = np.corrcoef(output_matrix)
                else:
                    corr_matrix = np.full((n_heads, n_heads), np.nan)

        finally:
            handle.remove()

        return corr_matrix


# ============================================================================
# VARIANCE MEASUREMENT ACROSS SITES
# ============================================================================

def measure_variance_by_site(
    model: nn.Module,
    dataloader: DataLoader,
    suppressors: Set[int],
    device: str
) -> Dict[str, List[float]]:
    """
    Measure variance at three sites:
      1. Suppressor heads (layer 0)
      2. Clean heads (early layer 0)
      3. Clean heads (late layer 1)

    Args:
        model: Transformer model
        dataloader: Data for measurement
        suppressors: Set of suppressor head indices
        device: Torch device

    Returns:
        Dictionary mapping site_name -> list of variance values
    """
    print("\n[EXP D] Measuring variance at multiple sites...")

    auditor = AllostasisAudit(device=device)
    analyzer = SuppressorAnalyzer(model, device)

    variances = {
        'suppressor_heads': [],
        'clean_heads_early': [],
        'clean_heads_late': []
    }

    n_heads = model.n_heads

    for batch_idx, (input_ids, _) in enumerate(tqdm(dataloader, desc="Measuring variances", total=20)):
        if batch_idx >= 20:
            break

        input_ids = input_ids.to(device)

        # Measure suppressor heads (layer 0)
        for head_idx in suppressors:
            var = analyzer.measure_head_output_variance(input_ids, layer_idx=0, head_idx=head_idx)
            if not np.isnan(var):
                variances['suppressor_heads'].append(var)

        # Measure clean heads in early layer (layer 0, non-suppressors)
        clean_early = set(range(n_heads)) - suppressors
        for head_idx in clean_early:
            var = analyzer.measure_head_output_variance(input_ids, layer_idx=0, head_idx=head_idx)
            if not np.isnan(var):
                variances['clean_heads_early'].append(var)

        # Measure clean heads in late layer (layer 1)
        for head_idx in range(n_heads):
            var = analyzer.measure_head_output_variance(input_ids, layer_idx=1, head_idx=head_idx)
            if not np.isnan(var):
                variances['clean_heads_late'].append(var)

    return variances


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def compute_bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for mean.

    Args:
        data: List of measurements
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 0.95)

    Returns:
        (mean, lower_ci, upper_ci)
    """
    data = np.array(data)
    mean = np.mean(data)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)
    alpha = 1.0 - ci
    lower_ci = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_ci = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return mean, lower_ci, upper_ci


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main(args):
    """Main Experiment D pipeline."""

    # Setup reproducibility
    if args.seed is not None:
        setup_reproducibility(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # Setup logging
    logger = AuditLogger(
        experiment_name="exp_d_superposition",
        output_dir=args.output_dir,
        seed=args.seed
    )

    # Create data
    print("\n[EXP D] Creating dataset...")
    dataset = SimpleTokenDataset(
        vocab_size=args.vocab_size,
        n_samples=args.n_samples,
        seq_len=args.seq_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
    print("[EXP D] Creating model...")
    model = SimpleTrans(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout
    ).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Quick training for model initialization (optional)
    if args.train_steps > 0:
        print(f"\n[EXP D] Pre-training model for {args.train_steps} steps...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for step, (input_ids, target_ids) in enumerate(dataloader):
            if step >= args.train_steps:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)  # [batch, seq, vocab]
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), target_ids.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}")

    # ========================================================================
    # CORE EXPERIMENT: IDENTIFY SUPPRESSORS
    # ========================================================================

    analyzer = SuppressorAnalyzer(model, device)

    # Step 1: Identify suppressors via ablation
    suppressors, head_stats = analyzer.identify_suppressors(
        dataloader,
        layer_idx=0,
        variance_ratio_threshold=args.suppressor_threshold
    )

    print(f"\nIdentified {len(suppressors)} suppressors: {suppressors}")

    # Log suppressor statistics
    logger.log_metrics(0, {
        'num_suppressors': len(suppressors),
        'suppressor_indices': list(suppressors),
        'suppressor_stats': head_stats
    })

    # Step 2: Measure variance at multiple sites
    variances = measure_variance_by_site(model, dataloader, suppressors, device)

    # Step 3: Compute statistics and bootstrap CI
    print("\n[EXP D] Computing statistics...")

    statistics = {}

    for site, values in variances.items():
        if values:
            mean, lower_ci, upper_ci = compute_bootstrap_ci(
                values,
                n_bootstrap=args.n_bootstrap,
                ci=0.95
            )
            statistics[site] = {
                'mean': float(mean),
                'lower_ci': float(lower_ci),
                'upper_ci': float(upper_ci),
                'n_samples': len(values),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

            print(f"\n{site}:")
            print(f"  Mean: {mean:.6f}")
            print(f"  CI [95%]: [{lower_ci:.6f}, {upper_ci:.6f}]")
            print(f"  Std: {np.std(values):.6f}")
            print(f"  N: {len(values)}")

    # Step 4: Compare suppressor vs clean head variance
    if suppressors and variances['clean_heads_early']:
        supp_mean = np.mean(variances['suppressor_heads']) if variances['suppressor_heads'] else 0
        clean_mean = np.mean(variances['clean_heads_early'])

        ratio = supp_mean / clean_mean if clean_mean > 1e-10 else float('inf')

        print(f"\n[EXP D] Variance Ratio (Suppressor / Clean):")
        print(f"  Ratio: {ratio:.2f}x")

        if ratio >= 2.0:
            print(f"  SUCCESS: Ratio >= 2.0x (meets success criterion)")
        else:
            print(f"  WARNING: Ratio < 2.0x (does not meet success criterion)")

        statistics['variance_ratio_suppressor_to_clean'] = ratio

    # Step 5: Cross-validate with correlation analysis
    print("\n[EXP D] Computing head correlations...")
    sample_batch = next(iter(dataloader))[0]
    corr_matrix = analyzer.compute_head_correlations(sample_batch, layer_idx=0)

    # Check for anti-correlations among suppressors
    if len(suppressors) > 1:
        supp_list = list(suppressors)
        anti_corr_count = 0
        for i in range(len(supp_list)):
            for j in range(i + 1, len(supp_list)):
                h1, h2 = supp_list[i], supp_list[j]
                corr = corr_matrix[h1, h2]
                if corr < -0.5:
                    anti_corr_count += 1
                    print(f"  Suppressors {h1}-{h2}: rho={corr:.3f} (anti-correlated)")

        statistics['suppressor_anti_correlation_count'] = anti_corr_count
        statistics['suppressor_anti_correlation_pairs'] = len(supp_list) * (len(supp_list) - 1) // 2

    # Log final statistics
    logger.log_metrics(1, statistics)

    # Save results
    output_file = logger.save_log()
    print(f"\n[EXP D] Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT D SUMMARY")
    print("=" * 70)
    print(f"Suppressors identified: {len(suppressors)}")
    print(f"Suppressor indices: {suppressors}")
    if 'variance_ratio_suppressor_to_clean' in statistics:
        print(f"Variance ratio (Supp/Clean): {statistics['variance_ratio_suppressor_to_clean']:.2f}x")
    print(f"Bootstrap CIs computed: yes")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment D: Polysemanticity and High-A Regime (Suppressor Identification)"
    )

    # Data arguments
    parser.add_argument("--vocab_size", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of data samples")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    # Model arguments
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=256, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Experiment arguments
    parser.add_argument("--train_steps", type=int, default=200, help="Pre-training steps")
    parser.add_argument("--suppressor_threshold", type=float, default=1.5,
                        help="Variance ratio threshold for suppressor identification")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap samples")

    # System arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")

    args = parser.parse_args()

    # Run experiment
    stats = main(args)
