"""
EXPERIMENT 2 (Part B): Temporal Ordering - Testing G→S Sequence

Research Question: Does Geometry lock before Slack is allocated?

Design: Training Dynamics Tracking
- Train model with noisy loss (induces S reallocation)
- Log every 500 steps (continuous, not adaptive)
- Track:
  - G metrics: QK drift, attention pattern stability
  - S metrics: Suppressor strength, residual direction changes
- Analyze temporal lag between G stabilization and S emergence

Predictions if G→S ordering holds:
- Attention patterns stabilize early (first 20% of training)
- Suppressor variance increases AFTER routing locks
- Lag > 5k steps between G lock and S emergence

Success Criteria:
- G stabilizes before S (lag > 0)
- Freezing QK before lock prevents S emergence
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
import argparse
import json
from scipy.stats import spearmanr

# Import libraries
from lib.metrics import AllostasisAudit
from lib.logging_utils import setup_reproducibility
from lib.part_b_utils import (
    get_qk_parameters, measure_suppressor_strength,
    compute_baseline_metrics
)
from lib.part_b_losses import get_loss_function, inject_noise

# Import model and data from exp_c
from experiments.phase_1_foundation.exp_c_grokking import GrabbingTransformer, ModularAdditionDataset


def find_stabilization_point(values: np.ndarray, window: int = 5, 
                             threshold: float = 0.1) -> int:
    """
    Find when a metric stabilizes (changes less than threshold).
    
    Args:
        values: Array of metric values over time
        window: Window size for computing change rate
        threshold: Threshold for considering "stable"
        
    Returns:
        Step index where stabilization occurs, or -1 if never stabilizes
    """
    if len(values) < window + 1:
        return -1
    
    for i in range(window, len(values)):
        # Compute change over last window
        recent_values = values[i-window:i]
        change_rate = np.std(recent_values) / (np.mean(np.abs(recent_values)) + 1e-10)
        
        if change_rate < threshold:
            return i
    
    return -1


def find_onset_point(values: np.ndarray, threshold_percentile: float = 0.5) -> int:
    """
    Find when a metric begins to increase (onset of S allocation).
    
    Args:
        values: Array of metric values over time
        threshold_percentile: Values must exceed this percentile of max
        
    Returns:
        Step index of onset, or -1 if never occurs
    """
    if len(values) == 0:
        return -1
    
    threshold = threshold_percentile * np.max(values)
    onset_indices = np.where(values > threshold)[0]
    
    return int(onset_indices[0]) if len(onset_indices) > 0 else -1


def run_experiment_2(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    n_steps: int = 20000,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = 'cuda',
    log_freq: int = 500
):
    """
    Run Experiment 2: Temporal Ordering.
    
    Protocol:
    1. Train model with noisy loss
    2. Log every 500 steps (fixed frequency)
    3. Track G and S metrics throughout
    4. Analyze temporal ordering via lag detection
    """
    
    print("\n" + "="*80)
    print("EXPERIMENT 2 (PART B): TEMPORAL ORDERING - TESTING G→S SEQUENCE")
    print("="*80 + "\n")
    
    setup_reproducibility(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Data
    train_dataset = ModularAdditionDataset(
        p=p, n_samples=n_steps * batch_size, 
        train_fraction=0.5, is_train=True
    )
    val_dataset = ModularAdditionDataset(
        p=p, n_samples=5000, 
        train_fraction=0.5, is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset: Modular Addition (p={p})")
    print(f"Training for {n_steps} steps with logging every {log_freq} steps\n")
    
    # Model
    model = GrabbingTransformer(p=p, d_model=d_model, n_heads=n_heads)
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
        'mean_variance': [],
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
        targets = targets.to(device).squeeze(-1)
        
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
        
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        
        # Logging (fixed frequency - every log_freq steps)
        if step % log_freq == 0 or step == n_steps - 1:
            model.eval()
            
            # Validation accuracy
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs, val_targets = val_batch
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device).squeeze(-1)
                    
                    val_logits = model(val_inputs)
                    preds = val_logits.argmax(dim=-1)
                    val_correct += (preds == val_targets).sum().item()
                    val_total += val_targets.size(0)
            
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
            history['mean_variance'].append(suppressor_metrics['mean_variance'])
            
            print(f"Step {step:5d} | "
                  f"Val Acc: {val_acc:.3f} | "
                  f"QK Drift: {qk_drift:.2f} | "
                  f"Supp Strength: {suppressor_metrics['suppressor_strength']:.3f}")
    
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
    qk_stable_idx = find_stabilization_point(qk_drifts, window=3, threshold=0.05)
    qk_stable_step = steps[qk_stable_idx] if qk_stable_idx >= 0 else -1
    
    # Find suppressor onset
    supp_onset_idx = find_onset_point(suppressor_strengths, threshold_percentile=0.3)
    supp_onset_step = steps[supp_onset_idx] if supp_onset_idx >= 0 else -1
    
    # Compute lag
    lag = supp_onset_step - qk_stable_step if (qk_stable_step >= 0 and supp_onset_step >= 0) else None
    
    # Correlation (Spearman)
    if len(val_accs) > 2 and len(suppressor_strengths) > 2:
        corr_acc_supp, p_val = spearmanr(val_accs, suppressor_strengths)
    else:
        corr_acc_supp, p_val = np.nan, np.nan
    
    results = {
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'analysis': {
            'qk_stable_step': int(qk_stable_step) if qk_stable_step >= 0 else None,
            'suppressor_onset_step': int(supp_onset_step) if supp_onset_step >= 0 else None,
            'lag_steps': int(lag) if lag is not None else None,
            'ordering': 'G→S' if (lag is not None and lag > 0) else 'unclear',
            'corr_acc_vs_suppressor': float(corr_acc_supp),
            'final_val_acc': float(val_accs[-1]) if len(val_accs) > 0 else 0.0,
            'final_qk_drift': float(qk_drifts[-1]) if len(qk_drifts) > 0 else 0.0,
            'final_suppressor_strength': float(suppressor_strengths[-1]) if len(suppressor_strengths) > 0 else 0.0
        }
    }
    
    print(f"QK Stabilization: Step {qk_stable_step if qk_stable_step >= 0 else 'N/A'}")
    print(f"Suppressor Onset: Step {supp_onset_step if supp_onset_step >= 0 else 'N/A'}")
    print(f"Temporal Lag: {lag if lag is not None else 'N/A'} steps")
    print(f"Ordering: {results['analysis']['ordering']}")
    print(f"Correlation (Acc vs Suppressor): {corr_acc_supp:.3f}")
    
    # Success criterion
    success = lag is not None and lag > 0
    print(f"\nSuccess Criterion (G stabilizes before S): {'✓ PASS' if success else '✗ FAIL'}")
    
    # Save results
    output_path = Path("data/exp_2_temporal_ordering_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 2 (Part B): Temporal Ordering")
    parser.add_argument('--p', type=int, default=113, help='Prime modulus')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_steps', type=int, default=20000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--log_freq', type=int, default=500, help='Logging frequency (steps)')
    parser.add_argument('--quick_test', action='store_true', help='Quick test (2000 steps)')
    
    args = parser.parse_args()
    
    n_steps = 2000 if args.quick_test else args.n_steps
    
    run_experiment_2(
        p=args.p,
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
