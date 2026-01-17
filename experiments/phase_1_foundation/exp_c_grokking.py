"""
EXPERIMENT C: GROKKING AS A→G PHASE TRANSITION

Goal: Reinterpret grokking as a phase transition from brute force (A) to
generalizable routing (G). Show that weight decay pressure (not loss pressure
alone) drives the transition.

Resource: ARENA 3.0 / modular_addition task (p=113)

Conditions:
  1. High Weight Decay (WD=1.0): Standard grokking setup
     - Train 50k steps
     - Prediction: Phase transition at ~10k steps
     - Early: Low Ψ, high A_param (brute force)
     - Transition: A_param crashes, Ψ spikes
     - Late: Ψ sustained, A_param stays low (efficient)

  2. No Weight Decay (WD=0.0): Same architecture, task, schedule
     - Prediction: No sharp phase transition
     - A_param rises and stays high

Metrics tracked (adaptive frequency):
  - train_acc, val_acc (Ψ: Separability)
  - A_param (sum of W_V norms, Amplitude at parameter level)
  - A_activation (mean resid norm, Amplitude at activation level)
  - G_entropy (attention entropy, Geometry measure)
  - sigma_sq (residual variance, Variance byproduct)

Logging:
  - Transition zone (15-85% accuracy): every 10 steps
  - Stable zones: every 100 steps

Success criteria:
  - High WD: Phase transition visible, ρ(acc, A_param) < -0.3
  - No WD: No transition, A_param stays high
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Dict, List, Tuple
import argparse
from tqdm import tqdm
import json

# Import our libraries
from lib.metrics import AllostasisAudit, compute_attention_entropy
from lib.logging_utils import AuditLogger, setup_reproducibility, ProgressTracker


class ModularAdditionDataset(Dataset):
    """
    Modular addition task from ARENA 3.0.

    Task: Given two numbers a and b (mod p), predict (a + b) mod p.
    This tests whether the model learns the general rule or memorizes.

    Args:
        p: Prime modulus (default 113 from ARENA)
        n_samples: Number of samples to generate
        train_fraction: Fraction of (a, b) pairs for training (default 0.5)
    """

    def __init__(
        self,
        p: int = 113,
        n_samples: int = 10000,
        train_fraction: float = 0.5,
        is_train: bool = True
    ):
        self.p = p
        self.is_train = is_train

        # Generate all possible pairs (a, b) for a, b in [0, p)
        all_pairs = []
        for a in range(p):
            for b in range(p):
                all_pairs.append((a, b))

        n_total = len(all_pairs)
        n_train = int(n_total * train_fraction)

        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(all_pairs)

        if is_train:
            self.pairs = all_pairs[:n_train]
        else:
            self.pairs = all_pairs[n_train:]

        # For multiple epochs, cycle through pairs
        self.data = self.pairs * ((n_samples // len(self.pairs)) + 1)
        self.data = self.data[:n_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b = self.data[idx]
        label = (a + b) % self.p

        # Pack a, b into input sequence: [a, b]
        # Special tokens: a_token = 0-112, b_token = 113-225, result at position 2
        input_seq = torch.LongTensor([a, b + self.p])
        target = torch.LongTensor([label])

        return input_seq, target


class GrabbingTransformer(nn.Module):
    """
    Simple 1-layer Transformer for modular addition task.

    Architecture:
      - Token embedding: vocab_size = 2*p + 1
      - 1 attention layer with 4 heads
      - Feed-forward layer
      - Output projection to [0, p)

    The "grokking" phenomenon: early training memorizes (high A_param),
    then suddenly switches to computing the operation (low A_param, high G).
    """

    def __init__(
        self,
        p: int = 113,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()

        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads

        # Vocabulary: [a tokens: 0-112] + [b tokens: 113-225] + [padding]
        vocab_size = 2 * p + 10
        self.vocab_size = vocab_size

        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Single transformer block
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Output: predict result at position 2
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, p)

        # Cache for metrics computation
        self.cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 2] with x[:, 0] = a, x[:, 1] = b + p

        Returns:
            logits: [batch, p] for classification over [0, p)
        """
        batch_size = x.shape[0]

        # Embed
        x = self.embed(x)  # [batch, 2, d_model]

        # Store pre-attn residual
        self.cache['resid_pre_attn'] = x.clone()

        # Attention
        x_ln = self.ln1(x)
        attn_out, attn_weights = self.attn(x_ln, x_ln, x_ln, need_weights=True)
        x = x + attn_out

        # Store post-attn residual
        self.cache['resid_post_attn'] = x.clone()
        self.cache['attn_weights'] = attn_weights

        # Feed-forward
        x_ln = self.ln2(x)
        ff_out = self.ff(x_ln)
        x = x + ff_out

        # Store post-FF residual
        self.cache['resid_post'] = x.clone()

        # Final projection on the result position (position 2, or last token)
        # Take last token representation
        x_final = self.ln_final(x[:, -1, :])  # [batch, d_model]
        logits = self.output_proj(x_final)  # [batch, p]

        return logits


def train_step(
    model: GrabbingTransformer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str,
    auditor: AllostasisAudit
) -> Dict[str, float]:
    """
    Single training step.

    Args:
        model: Model to train
        batch: (inputs, targets) tuple
        optimizer: Optimizer
        device: Device
        auditor: Metrics auditor

    Returns:
        Dictionary with metrics for this step
    """
    model.train()

    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device).squeeze(-1)  # [batch]

    optimizer.zero_grad()

    # Forward pass
    logits = model(inputs)

    # Compute loss
    loss = F.cross_entropy(logits, targets)

    # Backward
    loss.backward()
    optimizer.step()

    # Compute metrics
    with torch.no_grad():
        acc = auditor.compute_psi_accuracy(logits, targets)
        A_learned = auditor.compute_amplitude_learned(model)
        A_param = auditor.compute_amplitude_param(model, "W_V")

        # Residual metrics
        resid_post = model.cache.get('resid_post')
        if resid_post is not None:
            A_activation = auditor.compute_amplitude_activation(resid_post)
            sigma_sq = auditor.compute_variance(resid_post)
        else:
            A_activation = float('nan')
            sigma_sq = float('nan')

        # Attention entropy (G proxy)
        attn_weights = model.cache.get('attn_weights')
        if attn_weights is not None:
            G_entropy = compute_attention_entropy(attn_weights)
        else:
            G_entropy = float('nan')

    return {
        'loss': loss.item(),
        'acc': acc,
        'A_learned': A_learned,
        'A_param': A_param,
        'A_activation': A_activation,
        'sigma_sq': sigma_sq,
        'G_entropy': G_entropy
    }


def validate(
    model: GrabbingTransformer,
    dataloader: DataLoader,
    device: str,
    auditor: AllostasisAudit
) -> Dict[str, float]:
    """
    Validation pass.

    Args:
        model: Model to validate
        dataloader: Validation data
        device: Device
        auditor: Metrics auditor

    Returns:
        Dictionary with aggregated metrics
    """
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_A_param = 0.0
    total_A_activation = 0.0
    total_sigma_sq = 0.0
    total_G_entropy = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(-1)

            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

            acc = auditor.compute_psi_accuracy(logits, targets)
            A_learned = auditor.compute_amplitude_learned(model)
            A_param = auditor.compute_amplitude_param(model, "W_V")

            # Residual metrics
            resid_post = model.cache.get('resid_post')
            if resid_post is not None:
                A_activation = auditor.compute_amplitude_activation(resid_post)
                sigma_sq = auditor.compute_variance(resid_post)
            else:
                A_activation = float('nan')
                sigma_sq = float('nan')

            # Attention entropy
            attn_weights = model.cache.get('attn_weights')
            if attn_weights is not None:
                G_entropy = compute_attention_entropy(attn_weights)
            else:
                G_entropy = float('nan')

            total_loss += loss.item()
            total_acc += acc
            total_A_param += A_param
            total_A_activation += A_activation
            total_sigma_sq += sigma_sq
            total_G_entropy += G_entropy
            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'acc': total_acc / n_batches,
        'A_param': total_A_param / n_batches,
        'A_activation': total_A_activation / n_batches,
        'sigma_sq': total_sigma_sq / n_batches,
        'G_entropy': total_G_entropy / n_batches
    }


def run_condition(
    condition_name: str,
    weight_decay: float,
    p: int = 113,
    n_steps: int = 50000,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    d_model: int = 128,
    n_heads: int = 4
) -> Dict[str, any]:
    """
    Run a single experimental condition.

    Args:
        condition_name: Name of condition ('high_wd' or 'no_wd')
        weight_decay: Weight decay coefficient
        p: Prime modulus
        n_steps: Number of training steps
        batch_size: Batch size
        lr: Learning rate
        seed: Random seed
        d_model: Model dimension
        n_heads: Number of attention heads

    Returns:
        Dictionary with results and logged metrics
    """
    print(f"\n{'='*80}")
    print(f"RUNNING CONDITION: {condition_name} (WD={weight_decay})")
    print(f"{'='*80}\n")

    # Setup
    setup_reproducibility(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = GrabbingTransformer(
        p=p,
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.0  # No dropout for grokking
    ).to(device)

    print(f"Model created:")
    print(f"  Vocab size: {model.vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    # 50% train, 50% test (standard for grokking)
    train_dataset = ModularAdditionDataset(
        p=p, n_samples=n_steps * batch_size, train_fraction=0.5, is_train=True
    )
    val_dataset = ModularAdditionDataset(
        p=p, n_samples=10000, train_fraction=0.5, is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"Datasets created:")
    print(f"  Train pairs: {len(train_dataset)}")
    print(f"  Val pairs: {len(val_dataset)}")

    # Optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    print(f"Optimizer: Adam(lr={lr}, weight_decay={weight_decay})")

    # Metrics and logging
    auditor = AllostasisAudit(device=device)
    logger = AuditLogger(
        experiment_name=f"exp_c_grokking_{condition_name}",
        seed=seed
    )

    # Training loop
    print(f"\nTraining for {n_steps} steps...")

    train_iter = iter(train_loader)
    metrics_history = []
    step_count = 0

    for step in range(n_steps):
        # Get batch (cycle through if needed)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Training step
        train_metrics = train_step(model, batch, optimizer, device, auditor)

        # Validation (every 100 steps for efficiency)
        if step % 100 == 0 or step == n_steps - 1:
            val_metrics = validate(model, val_loader, device, auditor)
            val_acc = val_metrics['acc']
        else:
            val_acc = train_metrics['acc']  # Estimate from training

        # Adaptive logging
        should_log = logger.should_log(step, val_acc, force=(step % 100 == 0))

        if should_log:
            # Combine metrics
            metrics = {
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['acc'],
                'val_loss': val_metrics['loss'] if step % 100 == 0 else float('nan'),
                'val_acc': val_acc,
                'A_learned': train_metrics['A_learned'],
                'A_param': train_metrics['A_param'],
                'A_activation': train_metrics['A_activation'],
                'sigma_sq': train_metrics['sigma_sq'],
                'G_entropy': train_metrics['G_entropy']
            }
            logger.log_metrics(step, metrics)
            metrics_history.append((step, metrics))

        # Progress
        if step % 500 == 0 or step == n_steps - 1:
            print(f"Step {step:6d} | "
                  f"Train: {train_metrics['acc']:.3f} | "
                  f"Val: {val_acc:.3f} | "
                  f"A_param: {train_metrics['A_param']:.3f} | "
                  f"σ²: {train_metrics['sigma_sq']:.3f}")

        step_count = step

    # Save log
    logger.save_log()

    # Compute phase transition analysis
    analysis = analyze_phase_transition(metrics_history, condition_name)

    return {
        'condition': condition_name,
        'weight_decay': weight_decay,
        'metrics_history': metrics_history,
        'analysis': analysis,
        'logger': logger
    }


def analyze_phase_transition(
    metrics_history: List[Tuple[int, Dict]],
    condition_name: str
) -> Dict[str, any]:
    """
    Analyze phase transition from the metrics history.

    Computes:
      - Time of phase transition (when A_param drops)
      - Correlation between accuracy and A_param
      - Rate of A_param decrease during transition

    Args:
        metrics_history: List of (step, metrics_dict) tuples
        condition_name: Name of condition for context

    Returns:
        Dictionary with analysis results
    """
    if not metrics_history:
        return {}

    steps = np.array([s for s, _ in metrics_history])
    accs = np.array([m['val_acc'] for _, m in metrics_history])
    A_params = np.array([m['A_param'] for _, m in metrics_history])

    # Find transition region (15-85% accuracy)
    transition_mask = (accs > 0.15) & (accs < 0.85)
    transition_indices = np.where(transition_mask)[0]

    result = {
        'n_logged_steps': len(metrics_history),
        'final_accuracy': float(accs[-1]) if len(accs) > 0 else float('nan'),
        'final_A_param': float(A_params[-1]) if len(A_params) > 0 else float('nan'),
        'mean_A_param': float(np.mean(A_params[~np.isnan(A_params)])) if np.any(~np.isnan(A_params)) else float('nan'),
        'std_A_param': float(np.std(A_params[~np.isnan(A_params)])) if np.any(~np.isnan(A_params)) else float('nan'),
    }

    # Correlation between accuracy and A_param
    valid_mask = ~(np.isnan(A_params) | np.isnan(accs))
    if np.sum(valid_mask) > 1:
        corr = np.corrcoef(accs[valid_mask], A_params[valid_mask])[0, 1]
        result['correlation_acc_vs_A_param'] = float(corr)
    else:
        result['correlation_acc_vs_A_param'] = float('nan')

    # Transition timing
    if len(transition_indices) > 0:
        transition_start_step = steps[transition_indices[0]]
        transition_end_step = steps[transition_indices[-1]]
        result['transition_start_step'] = int(transition_start_step)
        result['transition_end_step'] = int(transition_end_step)
        result['transition_duration_steps'] = int(transition_end_step - transition_start_step)

        # A_param decrease during transition
        transition_A_params = A_params[transition_indices]
        if len(transition_A_params) > 1:
            result['A_param_initial_transition'] = float(transition_A_params[0])
            result['A_param_final_transition'] = float(transition_A_params[-1])
            result['A_param_decrease'] = float(transition_A_params[0] - transition_A_params[-1])

    return result


def main():
    parser = argparse.ArgumentParser(description="Experiment C: Grokking as A→G Phase Transition")
    parser.add_argument('--condition', type=str, default='all',
                       choices=['high_wd', 'no_wd', 'all'],
                       help='Which condition to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_steps', type=int, default=50000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--p', type=int, default=113, help='Prime modulus')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--quick_test', action='store_true', help='Quick test run (5k steps)')

    args = parser.parse_args()

    # Adjust for quick test
    n_steps = 5000 if args.quick_test else args.n_steps

    print("\n" + "="*80)
    print("EXPERIMENT C: GROKKING AS A→G PHASE TRANSITION")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Seed: {args.seed}")
    print(f"  Steps: {n_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Prime modulus p: {args.p}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Attention heads: {args.n_heads}")

    # Run conditions
    conditions = []
    if args.condition in ['all', 'high_wd']:
        conditions.append(('high_wd', 1.0))
    if args.condition in ['all', 'no_wd']:
        conditions.append(('no_wd', 0.0))

    results = {}

    for cond_name, wd in conditions:
        result = run_condition(
            condition_name=cond_name,
            weight_decay=wd,
            p=args.p,
            n_steps=n_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            d_model=args.d_model,
            n_heads=args.n_heads
        )
        results[cond_name] = result

    # Summary and analysis
    print(f"\n{'='*80}")
    print("EXPERIMENT C SUMMARY")
    print(f"{'='*80}\n")

    for cond_name, result in results.items():
        analysis = result['analysis']
        print(f"\nCondition: {cond_name} (WD={result['weight_decay']})")
        print(f"  Final accuracy: {analysis.get('final_accuracy', float('nan')):.4f}")
        print(f"  Final A_param: {analysis.get('final_A_param', float('nan')):.4f}")
        print(f"  Mean A_param: {analysis.get('mean_A_param', float('nan')):.4f} ± {analysis.get('std_A_param', float('nan')):.4f}")
        print(f"  Correlation (acc vs A_param): {analysis.get('correlation_acc_vs_A_param', float('nan')):.4f}")

        if 'transition_start_step' in analysis:
            print(f"  Transition: steps {analysis['transition_start_step']}-{analysis['transition_end_step']} "
                  f"({analysis['transition_duration_steps']} steps)")
            if 'A_param_decrease' in analysis:
                print(f"  A_param decrease during transition: {analysis['A_param_decrease']:.4f}")

    # Success criteria
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*80}\n")

    checks = []

    if 'high_wd' in results:
        analysis = results['high_wd']['analysis']
        final_acc = analysis.get('final_accuracy', 0.0)
        corr = analysis.get('correlation_acc_vs_A_param', 0.0)

        # Phase transition should be visible (accuracy should be high)
        check1 = final_acc > 0.85
        checks.append(('High WD achieves >85% accuracy', check1))
        print(f"✓ High WD achieves >85% accuracy: {check1} ({final_acc:.2%})")

        # Negative correlation between accuracy and A_param (learning → fewer parameters)
        check2 = corr < -0.3
        checks.append(('High WD: ρ(acc, A_param) < -0.3', check2))
        print(f"✓ High WD: ρ(acc, A_param) < -0.3: {check2} (ρ={corr:.4f})")

    if 'no_wd' in results:
        analysis = results['no_wd']['analysis']
        final_A_param = analysis.get('final_A_param', 0.0)
        mean_A_param = analysis.get('mean_A_param', 0.0)

        # Without weight decay, A_param should stay high (or increase)
        check3 = final_A_param > 0.5 * mean_A_param if mean_A_param > 0 else False
        checks.append(('No WD: A_param stays high', check3))
        print(f"✓ No WD: A_param stays high: {check3} (final={final_A_param:.4f}, mean={mean_A_param:.4f})")

    all_passed = all(check for _, check in checks)
    print(f"\nOverall: {'✓ PASS' if all_passed else '✗ FAIL'}")

    # Save summary
    summary_path = Path("data/exp_c_summary.json")
    summary = {
        'conditions': {
            cond_name: {
                'weight_decay': result['weight_decay'],
                'analysis': result['analysis']
            }
            for cond_name, result in results.items()
        },
        'success_checks': [
            {'criterion': crit, 'passed': passed}
            for crit, passed in checks
        ],
        'overall_pass': all_passed
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
