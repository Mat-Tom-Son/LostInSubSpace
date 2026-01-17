"""
EXPERIMENT 3 (Part B, Interleaved Variant): Alternative S Allocations

Research Question: Can different S allocations coexist under the same G?

Design: Forced S Reallocation Under Fixed Routing
- Train baseline model to convergence on Interleaved task (>80% accuracy)
- Save checkpoint at convergence
- Freeze QK parameters (locks Geometry)
- Branch 3 conditions from checkpoint with different

 losses:
  - PRIMARY: standard, noisy, label_smoothing
  - SECONDARY: margin_penalty
- Measure residual direction CosSim (should be orthogonal if S is multidimensional)

Task: Interleaved sequences (proven convergence from Part A)

Success Criteria:
- All conditions converge (acc >75%)
- Pairwise residual CosSim < 0.5 (orthogonal S allocations)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict
import argparse
import json

# Import libraries
from lib.logging_utils import setup_reproducibility
from lib.part_b_utils import (
    freeze_parameters, get_qk_parameters, verify_freeze,
    compute_baseline_metrics, measure_suppressor_strength
)
from lib.part_b_losses import get_loss_function, inject_noise

# Import Interleaved dataset and model
from experiments.phase_1_foundation.exp_a_foundation import InterleavedSequenceDataset, SimpleTransformer


def train_with_frozen_qk(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn_name: str,
    n_steps: int,
    lr: float,
    device: str,
    initial_qk: torch.Tensor,
    vocab_size: int
) -> Dict:
    """Train model with frozen QK parameters."""
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    loss_fn = get_loss_function(loss_fn_name)
    
    train_iter = iter(train_loader)
    history = []
    
    inject_noise_flag = (loss_fn_name == 'noisy')
    
    print(f"Training with loss: {loss_fn_name}, QK frozen, {n_steps} steps")
    
    for step in range(n_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        logits = model(inputs)
        
        # Inject noise if noisy training
        if inject_noise_flag and 'resid_post' in model.cache:
            model.cache['resid_post'] = inject_noise(
                model.cache['resid_post'], 
                noise_scale=2.0, 
                device=device
            )
        
        # Compute loss (next-token prediction)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        
        # Verify QK hasn't changed
        if step % 500 == 0:
            if not verify_freeze(model, initial_qk):
                raise RuntimeError(f"QK parameters changed at step {step}!")
        
        # Log
        if step % 500 == 0 or step == n_steps - 1:
            val_acc = validate(model, val_loader, device, vocab_size)
            history.append({'step': step, 'loss': loss.item(), 'val_acc': val_acc})
            print(f"  Step {step:5d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.3f}")
    
    return {'history': history, 'final_acc': history[-1]['val_acc']}


def validate(model: nn.Module, dataloader: DataLoader, device: str, vocab_size: int) -> float:
    """Validation accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    
    model.train()
    return correct / total if total > 0 else 0.0


def run_experiment_3_interleaved(
    seq_len: int = 128,
    vocab_size: int = 4096,
    d_model: int = 128,
    n_heads: int = 4,
    baseline_steps: int = 10000,
    frozen_steps: int = 5000,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = 'cuda'
):
    """
    Run Experiment 3 (Interleaved): Alternative S Allocations.
    
    Protocol:
    1. Train baseline to convergence on Interleaved task
    2. Save checkpoint
    3. For each condition:
       - Load checkpoint
       - Freeze QK
       - Train with condition-specific loss
    4. Compute pairwise residual direction CosSim
    """
    
    print("\n" + "="*80)
    print("EXPERIMENT 3 (PART B, INTERLEAVED): ALTERNATIVE S ALLOCATIONS")
    print("="*80 + "\n")
    
    setup_reproducibility(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Data
    train_dataset = InterleavedSequenceDataset(
        n_samples=baseline_steps * batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size
    )
    val_dataset = InterleavedSequenceDataset(
        n_samples=5000,
        seq_len=seq_len,
        vocab_size=vocab_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # =========================================================================
    # PHASE 1: Train Baseline to Convergence
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 1: Training Baseline to Convergence")
    print("-"*80 + "\n")
    
    baseline_model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=seq_len,
        dropout=0.0
    )
    baseline_model.to(device)
    
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=lr)
    
    train_iter = iter(train_loader)
    for step in range(baseline_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = baseline_model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0 or step == baseline_steps - 1:
            val_acc = validate(baseline_model, val_loader, device, vocab_size)
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.3f}")
    
    baseline_acc = validate(baseline_model, val_loader, device, vocab_size)
    print(f"\nBaseline converged: {baseline_acc:.3f} accuracy")
    
    if baseline_acc < 0.75:
        print(f"WARNING: Baseline did not reach 75% threshold")
    
    # Save baseline state
    baseline_state = baseline_model.state_dict()
    baseline_qk = get_qk_parameters(baseline_model)
    
    # =========================================================================
    # PHASE 2: Train Conditions with Frozen QK
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 2: Training with Frozen QK (Branching from Checkpoint)")
    print("-"*80 + "\n")
    
    # PRIMARY conditions
    primary_conditions = ['standard', 'noisy', 'label_smooth']
    # SECONDARY conditions
    secondary_conditions = ['margin_penalty']
    
    all_conditions = primary_conditions + secondary_conditions
    
    results = {
        'baseline': {
            'acc': baseline_acc,
            'converged': baseline_acc > 0.75
        },
        'conditions': {}
    }
    
    # Store models for cross-comparison
    trained_models = {'baseline': baseline_model}
    
    for condition_name in all_conditions:
        tier = 'primary' if condition_name in primary_conditions else 'secondary'
        
        print(f"\n{'='*80}")
        print(f"Condition: {condition_name} ({tier})")
        print(f"{'='*80}\n")
        
        # Create fresh model from baseline checkpoint
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=seq_len,
            dropout=0.0
        )
        model.load_state_dict(baseline_state)
        model.to(device)
        
        # Freeze QK
        frozen_counts = freeze_parameters(model, freeze_qk=True, freeze_ov=False, freeze_mlp=False)
        print(f"Frozen QK parameters: {frozen_counts['qk']}")
        
        # Get initial QK for verification
        initial_qk = get_qk_parameters(model)
        
        # Train
        train_result = train_with_frozen_qk(
            model, train_loader, val_loader,
            loss_fn_name=condition_name,
            n_steps=frozen_steps,
            lr=lr,
            device=device,
            initial_qk=initial_qk,
            vocab_size=vocab_size
        )
        
        # Final verification
        if not verify_freeze(model, initial_qk):
            raise RuntimeError(f"QK changed during training for {condition_name}!")
        
        # Metrics vs baseline
        metrics = compute_baseline_metrics(model, baseline_model, val_loader, device)
        suppressor = measure_suppressor_strength(model, val_loader, device)
        
        results['conditions'][condition_name] = {
            'tier': tier,
            'final_acc': train_result['final_acc'],
            'history': train_result['history'],
            'vs_baseline': metrics,
            'suppressor': suppressor,
            'converged': train_result['final_acc'] > 0.75
        }
        
        trained_models[condition_name] = model
        
        print(f"\nFinal Accuracy: {train_result['final_acc']:.3f}")
        print(f"Residual CosSim vs Baseline: {metrics['resid_direction_cosim']:.3f}")
        print(f"QK Drift: {metrics['qk_norm_drift']:.6f} (should be 0.0)")
    
    # =========================================================================
    # PHASE 3: Cross-Condition Analysis
    # =========================================================================
    
    print("\n" + "="*80)
    print("CROSS-CONDITION ANALYSIS")
    print("="*80 + "\n")
    
    # Pairwise residual direction similarities
    condition_names = list(results['conditions'].keys())
    pairwise_cosim = {}
    
    for i, cond_a in enumerate(condition_names):
        for j, cond_b in enumerate(condition_names):
            if i < j:
                model_a = trained_models[cond_a]
                model_b = trained_models[cond_b]
                
                metrics_ab = compute_baseline_metrics(model_a, model_b, val_loader, device)
                pair_key = f"{cond_a}_vs_{cond_b}"
                pairwise_cosim[pair_key] = metrics_ab['resid_direction_cosim']
                
                print(f"{cond_a} vs {cond_b}: CosSim = {metrics_ab['resid_direction_cosim']:.3f}")
    
    results['pairwise_analysis'] = pairwise_cosim
    
    # =========================================================================
    # SUCCESS CRITERIA
    # =========================================================================
    
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80 + "\n")
    
    # 1. All primary conditions converge
    primary_accs = [results['conditions'][c]['final_acc'] for c in primary_conditions]
    all_converged = all(results['conditions'][c]['converged'] for c in primary_conditions)
    acc_spread = max(primary_accs) - min(primary_accs)
    
    print(f"1. All primary conditions converged (>75%): {'✓ PASS' if all_converged else '✗ FAIL'}")
    accs_str = ', '.join([f"{c}: {results['conditions'][c]['final_acc']:.3f}" for c in primary_conditions])
    print(f"   Accuracies: {accs_str}")
    
    # 2. Pairwise residual CosSim < 0.5
    primary_pairs = [k for k in pairwise_cosim.keys() if all(c in primary_conditions for c in k.split('_vs_'))]
    primary_cosims = [pairwise_cosim[k] for k in primary_pairs]
    criterion_2 = all(cs < 0.5 for cs in primary_cosims) if primary_cosims else False
    
    print(f"2. Pairwise CosSim < 0.5: {'✓ PASS' if criterion_2 else '✗ FAIL'}")
    print(f"   Mean CosSim: {np.mean(primary_cosims):.3f}, Range: [{min(primary_cosims):.3f}, {max(primary_cosims):.3f}]")
    
    overall_pass = all_converged and criterion_2
    print(f"\nOverall: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    
    results['success'] = {
        'all_converged': all_converged,
        'cosim_separated': criterion_2,
        'overall_pass': overall_pass
    }
    
    # Save
    output_path = Path("data/exp_3_interleaved_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--baseline_steps', type=int, default=10000)
    parser.add_argument('--frozen_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick_test', action='store_true')
    
    args = parser.parse_args()
    
    baseline_steps = 2000 if args.quick_test else args.baseline_steps
    frozen_steps = 500 if args.quick_test else args.frozen_steps
    
    run_experiment_3_interleaved(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        baseline_steps=baseline_steps,
        frozen_steps=frozen_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main()
