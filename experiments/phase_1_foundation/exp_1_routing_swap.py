"""
EXPERIMENT 1 (Part B): Routing Swap - Testing G Causality

Research Question: Does Geometry causally determine behavior?

Design: Attention Pattern Transplant
- Train two models on same task (modular arithmetic p=113)
  - Model A (standard): Standard training
  - Model B (noisy): Train under noise (different S allocation)
- Swap QK parameters from B → A (partial: 1,2,4 heads)
- Measure transfer WITHOUT retraining

Predictions if G is causal:
- Swapping QK should transfer behavioral properties
- Partial swaps should show graded transfer

Success Criteria:
- Transfer ≥30% of behavioral difference
- Swapped routing causes predictable S effects
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List
import argparse
import json

# Import libraries
from lib.metrics import AllostasisAudit
from lib.logging_utils import setup_reproducibility
from lib.part_b_utils import (
    freeze_parameters, swap_qk_parameters,
    compute_baseline_metrics, measure_suppressor_strength,
    get_qk_parameters
)
from lib.part_b_losses import get_loss_function, inject_noise

# Import model from exp_c (matches CANONICAL_ARCH)
from experiments.phase_1_foundation.exp_c_grokking import GrabbingTransformer, ModularAdditionDataset


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn_name: str,
    n_steps: int,
    lr: float,
    device: str,
    inject_noise_training: bool = False,
    noise_scale: float = 2.0
) -> Dict[str, any]:
    """Train a model to convergence."""
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = get_loss_function(loss_fn_name)
    
    train_iter = iter(train_loader)
    history = []
    
    print(f"Training with loss: {loss_fn_name}, noise={inject_noise_training}")
    
    for step in range(n_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device).squeeze(-1)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)
        
        # Inject noise if requested (for noisy training)
        if inject_noise_training and 'resid_post' in model.cache:
            model.cache['resid_post'] = inject_noise(
                model.cache['resid_post'], 
                noise_scale=noise_scale,
                device=device
            )
        
        # Compute loss
        loss = loss_fn(logits, targets)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Log periodically
        if step % 500 == 0 or step == n_steps - 1:
            val_acc = validate(model, val_loader, device)
            history.append({'step': step, 'loss': loss.item(), 'val_acc': val_acc})
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.3f}")
    
    return {'history': history, 'final_acc': history[-1]['val_acc']}


def validate(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """Quick validation accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(-1)
            
            logits = model(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    model.train()
    return correct / total if total > 0 else 0.0


def run_experiment_1(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    n_steps: int = 10000,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = 'cuda'
):
    """
    Run Experiment 1: Routing Swap.
    
    Protocol:
    1. Train Model A (standard loss)
    2. Train Model B (noisy loss)
    3. For each swap size (1, 2, 4 heads):
       - Create hybrid model
       - Swap QK from B → A
       - Evaluate WITHOUT retraining
    """
    
    print("\n" + "="*80)
    print("EXPERIMENT 1 (PART B): ROUTING SWAP - TESTING G CAUSALITY")
    print("="*80 + "\n")
    
    setup_reproducibility(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Data
    train_dataset = ModularAdditionDataset(p=p, n_samples=n_steps * batch_size, train_fraction=0.5, is_train=True)
    val_dataset = ModularAdditionDataset(p=p, n_samples=5000, train_fraction=0.5, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset: Modular Addition (p={p})")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}\n")
    
    # =========================================================================
    # PHASE 1: Train Model A (Standard)
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 1: Training Model A (Standard)")
    print("-"*80 + "\n")
    
    model_a = GrabbingTransformer(p=p, d_model=d_model, n_heads=n_heads)
    result_a = train_model(
        model_a, train_loader, val_loader,
        loss_fn_name='standard',
        n_steps=n_steps,
        lr=lr,
        device=device,
        inject_noise_training=False
    )
    
    # =========================================================================
    # PHASE 2: Train Model B (Noisy)
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 2: Training Model B (Noisy)")
    print("-"*80 + "\n")
    
    model_b = GrabbingTransformer(p=p, d_model=d_model, n_heads=n_heads)
    result_b = train_model(
        model_b, train_loader, val_loader,
        loss_fn_name='noisy',
        n_steps=n_steps,
        lr=lr,
        device=device,
        inject_noise_training=True,
        noise_scale=2.0
    )
    
    # =========================================================================
    # PHASE 3: Routing Swaps (1, 2, 4 heads)
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 3: Routing Swaps")
    print("-"*80 + "\n")
    
    results = {
        'model_a': {'acc': result_a['final_acc'], 'history': result_a['history']},
        'model_b': {'acc': result_b['final_acc'], 'history': result_b['history']},
        'swaps': []
    }
    
    auditor = AllostasisAudit(device=device)
    
    for n_heads_swap in [1, 2, 4]:
        print(f"\nSwapping {n_heads_swap} head(s) from B → A...")
        
        # Create hybrid model (start from A)
        model_hybrid = GrabbingTransformer(p=p, d_model=d_model, n_heads=n_heads)
        model_hybrid.load_state_dict(model_a.state_dict())
        model_hybrid.to(device)
        
        # Swap QK parameters
        swap_info = swap_qk_parameters(model_hybrid, model_b, n_heads=n_heads_swap)
        print(f"  Swapped {swap_info['swapped_params']} parameters")
        
        # Evaluate (NO RETRAINING)
        hybrid_acc = validate(model_hybrid, val_loader, device)
        
        # Compute metrics vs baselines
        metrics_vs_a = compute_baseline_metrics(model_hybrid, model_a, val_loader, device)
        metrics_vs_b = compute_baseline_metrics(model_hybrid, model_b, val_loader, device)
        
        # Suppressor measurement
        suppressor_metrics = measure_suppressor_strength(model_hybrid, val_loader, device)
        
        swap_result = {
            'n_heads_swapped': n_heads_swap,
            'accuracy': hybrid_acc,
            'swap_info': swap_info,
            'vs_model_a': metrics_vs_a,
            'vs_model_b': metrics_vs_b,
            'suppressor': suppressor_metrics
        }
        
        results['swaps'].append(swap_result)
        
        print(f"  Accuracy: {hybrid_acc:.3f}")
        print(f"  Attention CosSim vs A: {metrics_vs_a['attn_cosim_vs_baseline']:.3f}")
        print(f"  Attention CosSim vs B: {metrics_vs_b['attn_cosim_vs_baseline']:.3f}")
        print(f"  Residual CosSim vs A: {metrics_vs_a['resid_direction_cosim']:.3f}")
        print(f"  Residual CosSim vs B: {metrics_vs_b['resid_direction_cosim']:.3f}")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    print("\n" + "="*80)
    print("EXPERIMENT 1 ANALYSIS")
    print("="*80 + "\n")
    
    print(f"Model A (Standard): {results['model_a']['acc']:.3f}")
    print(f"Model B (Noisy):    {results['model_b']['acc']:.3f}")
    print(f"Behavioral Gap:     {abs(results['model_a']['acc'] - results['model_b']['acc']):.3f}\n")
    
    for swap in results['swaps']:
        n_heads = swap['n_heads_swapped']
        acc = swap['accuracy']
        transfer_from_a = abs(acc - results['model_a']['acc'])
        transfer_from_b = abs(acc - results['model_b']['acc'])
        
        print(f"{n_heads} head(s) swapped:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Drift from A: {transfer_from_a:.3f}")
        print(f"  Drift from B: {transfer_from_b:.3f}")
    
    # Success criterion: ≥30% transfer
    if len(results['swaps']) > 0:
        final_swap = results['swaps'][-1]  # All heads swapped
        gap = abs(results['model_a']['acc'] - results['model_b']['acc'])
        transfer = abs(final_swap['accuracy'] - results['model_a']['acc'])
        transfer_pct = (transfer / gap * 100) if gap > 0 else 0
        
        print(f"\nTransfer Percentage (all heads): {transfer_pct:.1f}%")
        print(f"Success Criterion (≥30%): {'✓ PASS' if transfer_pct >= 30 else '✗ FAIL'}")
    
    # Save results
    output_path = Path("data/exp_1_routing_swap_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 (Part B): Routing Swap")
    parser.add_argument('--p', type=int, default=113, help='Prime modulus')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_steps', type=int, default=10000, help='Training steps per model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--quick_test', action='store_true', help='Quick test (1000 steps)')
    
    args = parser.parse_args()
    
    n_steps = 1000 if args.quick_test else args.n_steps
    
    run_experiment_1(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_steps=n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main()
