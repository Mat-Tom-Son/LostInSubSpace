"""
EXPERIMENT 2 (Part B, Interleaved Variant): Temporal Ordering - Testing G→S Sequence

Research Question: Does Geometry lock before Slack is allocated?

Design: Training Dynamics Tracking on Interleaved Task
- Train model with noisy loss (induces S reallocation)
- Track G and S metrics throughout training to convergence
- Analyze temporal lag between G stabilization and S emergence

Task: Interleaved sequences (proven convergence from Part A)
- Forces attention routing (G) to separate interfering streams
- Requires slack allocation (S) for robustness

Success Criteria:
- G stabilizes before S (lag > 0)
- Model converges (>80% accuracy)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple
import argparse
import json
from scipy.stats import spearmanr

# Import libraries
from lib.logging_utils import setup_reproducibility
from lib.part_b_utils import (
    get_qk_parameters, measure_suppressor_strength
)
from lib.part_b_losses import get_loss_function, inject_noise

# Import Interleaved dataset and model from exp_a_foundation
from experiments.exp_a_foundation import InterleavedSequenceDataset, SimpleTransformer


def find_stabilization_point(values: np.ndarray, window: int = 10, 
                             threshold: float = 0.05) -> int:
    """Find when a metric stabilizes (coefficient of variation < threshold)."""
    if len(values) < window + 1:
        return -1
    
    for i in range(window, len(values)):
        recent_values = values[i-window:i]
        cv = np.std(recent_values) / (np.mean(np.abs(recent_values)) + 1e-10)
        if cv < threshold:
            return i
    
    return -1


def find_onset_point(values: np.ndarray, baseline_percentile: float = 0.2,
                     threshold_percentile: float = 0.6) -> int:
    """Find when a metric begins to increase above baseline."""
    if len(values) < 10:
        return -1
    
    baseline = np.percentile(values[:min(10, len(values))], baseline_percentile * 100)
    threshold = np.percentile(values, threshold_percentile * 100)
    
    onset_indices = np.where(values > max(baseline * 1.5, threshold))[0]
    return int(onset_indices[0]) if len(onset_indices) > 0 else -1


def run_experiment_2_interleaved(
    seq_len: int = 128,
    vocab_size: int = 4096,
    d_model: int = 128,
    n_heads: int = 4,
    n_steps: int = 20000,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = 'cuda',
    log_freq: int = 500
):
    """
    Run Experiment 2 (Interleaved): Temporal Ordering.
    
    Protocol:
    1. Train model with noisy loss on Interleaved task
    2. Log every log_freq steps (fixed frequency)
    3. Track G (QK drift) and S (suppressor strength) throughout
    4. Analyze temporal ordering via lag detection
    """
    
    print("\n" + "="*80)
    print("EXPERIMENT 2 (PART B, INTERLEAVED): TEMPORAL ORDERING")
    print("="*80 + "\n")
    
    setup_reproducibility(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Data
    train_dataset = InterleavedSequenceDataset(
        n_samples=n_steps * batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size
    )
    val_dataset = InterleavedSequenceDataset(
        n_samples=5000,
        seq_len=seq_len,
        vocab_size=vocab_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Task: Interleaved Sequences")
    print(f"Seq len: {seq_len}, Vocab: {vocab_size}")
    print(f"Training for {n_steps} steps with logging every {log_freq} steps\n")
    
    # Model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=seq_len,
        dropout=0.0
    )
    model.to(device)
    
    # Baseline (initial state for comparison)
    baseline_qk = get_qk_parameters(model).clone()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = get_loss_function('noisy')
    
    # Metrics storage
    history = {
        'steps': [],
        'train_acc': [],
        'val_acc': [],
        'qk_drift': [],
        'suppressor_strength': [],
        'suppressor_count': [],
        'variance_ratio': [],
    }
    
    print("="*80)
    print("TRAINING WITH CONTINUOUS LOGGING")
    print("="*80 + "\n")
    
    train_iter = iter(train_loader)
    
    for step in range(n_steps):
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        logits = model(inputs)
        
        # Inject noise (for noisy training)
        if 'resid_post' in model.cache:
            model.cache['resid_post'] = inject_noise(
                model.cache['resid_post'], 
                noise_scale=2.0, 
                device=device
            )
        
        # Loss: next-token prediction entropy
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        
        # Logging (fixed frequency)
        if step % log_freq == 0 or step == n_steps - 1:
            model.eval()
            
            # Validation accuracy
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs, val_targets = val_batch
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    
                    val_logits = model(val_inputs)
                    preds = val_logits.argmax(dim=-1)
                    val_correct += (preds == val_targets).sum().item()
                    val_total += val_targets.numel()
            
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            train_acc = (logits.argmax(-1) == targets).float().mean().item()
            
            # G metric: QK drift from init
            current_qk = get_qk_parameters(model)
            qk_drift = (current_qk - baseline_qk).norm().item()
            
            # S metric: Suppressor strength
            suppressor_metrics = measure_suppressor_strength(model, val_loader, device)
            
            # Store
            history['steps'].append(step)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['qk_drift'].append(qk_drift)
            history['suppressor_strength'].append(suppressor_metrics['suppressor_strength'])
            history['suppressor_count'].append(suppressor_metrics['n_suppressors'])
            history['variance_ratio'].append(suppressor_metrics['variance_ratio'])
            
            print(f"Step {step:5d} | "
                  f"Val Acc: {val_acc:.3f} | "
                  f"QK Drift: {qk_drift:.2f} | "
                  f"Supp: {suppressor_metrics['suppressor_strength']:.3f}")
    
    # =========================================================================
    # ANALYSIS: Temporal Ordering
    # =========================================================================
    
    print("\n" + "="*80)
    print("TEMPORAL ORDERING ANALYSIS")
    print("="*80 + "\n")
    
    steps = np.array(history['steps'])
    qk_drifts = np.array(history['qk_drift'])
    suppressor_strengths = np.array(history['suppressor_strength'])
    val_accs = np.array(history['val_acc'])
    
    # Find stabilization points
    qk_stable_idx = find_stabilization_point(qk_drifts, window=5, threshold=0.03)
    qk_stable_step = steps[qk_stable_idx] if qk_stable_idx >= 0 else -1
    
    # Find suppressor onset (when it rises above baseline)
    supp_onset_idx = find_onset_point(suppressor_strengths)
    supp_onset_step = steps[supp_onset_idx] if supp_onset_idx >= 0 else -1
    
    # Compute lag
    lag = supp_onset_step - qk_stable_step if (qk_stable_step >= 0 and supp_onset_step >= 0) else None
    
    # Correlation
    if len(val_accs) > 2:
        corr_acc_supp, _ = spearmanr(val_accs, suppressor_strengths)
    else:
        corr_acc_supp = np.nan
    
    results = {
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'analysis': {
            'qk_stable_step': int(qk_stable_step) if qk_stable_step >= 0 else None,
            'suppressor_onset_step': int(supp_onset_step) if supp_onset_step >= 0 else None,
            'lag_steps': int(lag) if lag is not None else None,
            'ordering': 'G→S' if (lag is not None and lag > 0) else 'unclear',
            'corr_acc_vs_suppressor': float(corr_acc_supp),
            'final_val_acc': float(val_accs[-1]),
            'converged': float(val_accs[-1]) > 0.8
        }
    }
    
    print(f"Final Validation Accuracy: {val_accs[-1]:.3f}")
    print(f"Converged (>80%): {'✓' if results['analysis']['converged'] else '✗'}")
    print(f"\nQK Stabilization: Step {qk_stable_step if qk_stable_step >= 0 else 'N/A'}")
    print(f"Suppressor Onset: Step {supp_onset_step if supp_onset_step >= 0 else 'N/A'}")
    print(f"Temporal Lag: {lag if lag is not None else 'N/A'} steps")
    print(f"Ordering: {results['analysis']['ordering']}")
    
    # Success criterion
    success = (lag is not None and lag > 0 and results['analysis']['converged'])
    print(f"\nSuccess: {'✓ PASS' if success else '✗ FAIL'}")
    
    # Save
    output_path = Path("clean_audit/data/exp_2_interleaved_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_freq', type=int, default=500)
    parser.add_argument('--quick_test', action='store_true')
    
    args = parser.parse_args()
    n_steps = 2000 if args.quick_test else args.n_steps
    
    run_experiment_2_interleaved(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_steps=n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        log_freq=args.log_freq
    )


if __name__ == '__main__':
    main()
