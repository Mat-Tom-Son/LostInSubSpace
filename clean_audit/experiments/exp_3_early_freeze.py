"""
EXPERIMENT 3 REVISED: Early G Freeze for S Multidimensionality

Research Question: Can different S allocations exist under the same G?

Design: Freeze G EARLY (before convergence) instead of after
- Hypothesis: "Young" G is flexible, "Mature" G is rigid
- Freeze QK at step 2000 (partial convergence)
- Train 4 conditions with different losses
- Measure if pairwise residual CosSim < 0.5

Success Criteria:
- All conditions converge (>90% accuracy)
- Pairwise residual CosSim < 0.5 (orthogonal S allocations)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict
import argparse
import json

from lib.logging_utils import setup_reproducibility
from lib.part_b_utils import (
    freeze_parameters, get_qk_parameters, verify_freeze,
    compute_baseline_metrics, measure_suppressor_strength
)
from lib.part_b_losses import get_loss_function, inject_noise

from experiments.exp_a_foundation import InterleavedSequenceDataset, SimpleTransformer


def train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_steps: int,
    lr: float,
    device: str,
    vocab_size: int,
    phase_name: str = "Phase"
) -> Dict:
    """Train for a specific number of steps."""
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    
    train_iter = iter(train_loader)
    
    print(f"\n{phase_name}: Training for {n_steps} steps")
    
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
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            val_acc = validate(model, val_loader, device, vocab_size)
            print(f"  Step {step:5d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.3f}")
    
    final_acc = validate(model, val_loader, device, vocab_size)
    print(f"  {phase_name} complete. Final acc: {final_acc:.3f}")
    
    return {'final_acc': final_acc}


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
        
        if inject_noise_flag and 'resid_post' in model.cache:
            model.cache['resid_post'] = inject_noise(
                model.cache['resid_post'], 
                noise_scale=2.0, 
                device=device
            )
        
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        
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


def run_early_freeze_experiment(
    seq_len: int = 128,
    vocab_size: int = 4096,
    d_model: int = 128,
    n_heads: int = 4,
    warmup_steps: int = 2000,
    frozen_steps: int = 8000,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = 'cuda'
):
    """
    Run Experiment 3 Revised: Early G Freeze.
    
    Protocol:
    1. Train baseline for warmup_steps (partial convergence ~70-85%)
    2. Save checkpoint and freeze QK (young G)
    3. Branch 4 conditions with different losses
    4. Measure pairwise residual CosSim
    """
    
    print("\n" + "="*80)
    print("EXPERIMENT 3 REVISED: EARLY G FREEZE FOR S MULTIDIMENSIONALITY")
    print("="*80 + "\n")
    
    setup_reproducibility(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Warmup steps: {warmup_steps}, Frozen steps: {frozen_steps}\n")
    
    # Data
    train_dataset = InterleavedSequenceDataset(
        n_samples=(warmup_steps + frozen_steps) * batch_size,
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
    # PHASE 1: Warmup (Partial Convergence)
    # =========================================================================
    
    print("-"*80)
    print("PHASE 1: Warmup Training (Partial Convergence)")
    print("-"*80)
    
    baseline_model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=seq_len,
        dropout=0.0
    )
    
    warmup_result = train_phase(
        baseline_model, train_loader, val_loader,
        n_steps=warmup_steps,
        lr=lr,
        device=device,
        vocab_size=vocab_size,
        phase_name="Warmup"
    )
    
    warmup_acc = warmup_result['final_acc']
    print(f"\nWarmup accuracy: {warmup_acc:.3f}")
    print("This is our 'Young G' - partially formed, potentially flexible")
    
    # Save early checkpoint
    early_state = baseline_model.state_dict()
    early_qk = get_qk_parameters(baseline_model)
    
    # =========================================================================
    # PHASE 2: Train Conditions with Frozen Early G
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 2: Training with Frozen Early G (Young Geometry)")
    print("-"*80 + "\n")
    
    conditions = ['standard', 'noisy', 'label_smooth', 'margin_penalty']
    
    results = {
        'warmup': {
            'acc': warmup_acc,
            'steps': warmup_steps
        },
        'conditions': {}
    }
    
    trained_models = {}
    
    for condition_name in conditions:
        print(f"\n{'='*80}")
        print(f"Condition: {condition_name}")
        print(f"{'='*80}\n")
        
        # Create model from early checkpoint
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=seq_len,
            dropout=0.0
        )
        model.load_state_dict(early_state)
        model.to(device)
        
        # Freeze QK (young G)
        frozen_counts = freeze_parameters(model, freeze_qk=True, freeze_ov=False, freeze_mlp=False)
        print(f"Frozen QK parameters: {frozen_counts['qk']}")
        
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
        
        # Verify freeze
        if not verify_freeze(model, initial_qk):
            print(f"WARNING: QK changed during training for {condition_name}!")
        
        results['conditions'][condition_name] = {
            'final_acc': train_result['final_acc'],
            'history': train_result['history'],
            'converged': train_result['final_acc'] > 0.75
        }
        
        trained_models[condition_name] = model
        print(f"\nFinal Accuracy: {train_result['final_acc']:.3f}")
    
    # =========================================================================
    # PHASE 3: Cross-Condition Analysis
    # =========================================================================
    
    print("\n" + "="*80)
    print("CROSS-CONDITION ANALYSIS")
    print("="*80 + "\n")
    
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
    
    # All converged
    all_converged = all(results['conditions'][c]['converged'] for c in condition_names)
    print(f"1. All conditions converged (>75%): {'✓ PASS' if all_converged else '✗ FAIL'}")
    
    accs = [results['conditions'][c]['final_acc'] for c in condition_names]
    print(f"   Accuracies: {', '.join([f'{c}: {results['conditions'][c]['final_acc']:.3f}' for c in condition_names])}")
    
    # Pairwise CosSim < 0.5
    cosim_values = list(pairwise_cosim.values())
    criterion_2 = all(cs < 0.5 for cs in cosim_values) if cosim_values else False
    mean_cosim = np.mean(cosim_values)
    
    print(f"2. Pairwise CosSim < 0.5: {'✓ PASS' if criterion_2 else '✗ FAIL'}")
    print(f"   Mean CosSim: {mean_cosim:.3f}, Range: [{min(cosim_values):.3f}, {max(cosim_values):.3f}]")
    
    # Compare to late freeze
    print(f"\n3. Early vs Late Freeze Comparison:")
    print(f"   Warmup acc: {warmup_acc:.3f} (Young G)")
    print(f"   Late freeze (previous): CosSim ~0.95 (Mature G was rigid)")
    print(f"   Early freeze (this): CosSim ~{mean_cosim:.3f}")
    
    if mean_cosim < 0.9:
        print(f"\n   ✓ Early freeze shows MORE S diversity than late freeze!")
        flexibility = "more_flexible"
    else:
        print(f"\n   ✗ Early freeze still shows limited S diversity")
        flexibility = "still_rigid"
    
    overall_pass = all_converged and criterion_2
    print(f"\nOverall: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    
    results['success'] = {
        'all_converged': all_converged,
        'cosim_separated': criterion_2,
        'mean_cosim': mean_cosim,
        'flexibility': flexibility,
        'overall_pass': overall_pass
    }
    
    # Save
    output_path = Path("clean_audit/data/exp_3_early_freeze_results.json")
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
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--frozen_steps', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick_test', action='store_true')
    
    args = parser.parse_args()
    
    warmup_steps = 500 if args.quick_test else args.warmup_steps
    frozen_steps = 1000 if args.quick_test else args.frozen_steps
    
    run_early_freeze_experiment(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        warmup_steps=warmup_steps,
        frozen_steps=frozen_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main()
