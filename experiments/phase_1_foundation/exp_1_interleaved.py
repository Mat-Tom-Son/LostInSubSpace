"""
EXPERIMENT 1 (Part B, Interleaved): Routing Swap - Testing G Causality

Research Question: Does Geometry causally determine behavior?

Design: Attention Pattern Transplant on CONVERGED models
- Train two models on Interleaved task (proven to converge >99%)
  - Model A (standard): Standard training
  - Model B (noisy): Train under noise (different S allocation)
- Swap QK parameters from B → A (partial: 1,2,4 heads)
- Measure transfer WITHOUT retraining

This version addresses the mod-arith limitation: models now converge,
so behavioral transfer is meaningful.

Success Criteria:
- Swapping QK should transfer behavioral properties
- Transfer should scale with number of heads swapped
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
from lib.logging_utils import setup_reproducibility
from lib.part_b_utils import (
    swap_qk_parameters,
    compute_baseline_metrics, measure_suppressor_strength,
    get_qk_parameters
)
from lib.part_b_losses import get_loss_function, inject_noise

# Import Interleaved dataset and SimpleTransformer
from experiments.phase_1_foundation.exp_a_foundation import InterleavedSequenceDataset, SimpleTransformer


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn_name: str,
    n_steps: int,
    lr: float,
    device: str,
    vocab_size: int,
    inject_noise_training: bool = False,
    noise_scale: float = 2.0
) -> Dict:
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
        targets = targets.to(device)
        
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
        
        # Compute loss (next-token prediction)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Log periodically
        if step % 500 == 0 or step == n_steps - 1:
            val_acc = validate(model, val_loader, device, vocab_size)
            history.append({'step': step, 'loss': loss.item(), 'val_acc': val_acc})
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.3f}")
    
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


def run_single_seed(
    seq_len: int = 128,
    vocab_size: int = 4096,
    d_model: int = 128,
    n_heads: int = 4,
    n_steps: int = 10000,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = 'cuda'
):
    """
    Run Experiment 1 (Interleaved): Routing Swap on Converged Models.
    
    Protocol (from Part B design):
    1. Train Model A (standard loss) to convergence
    2. Train Model B (noisy loss) to convergence
    3. For each swap size (1, 2, 4 heads):
       - Create hybrid model (start from A)
       - Swap QK from B → A
       - Evaluate WITHOUT retraining
    """
    
    print("\n" + "="*80)
    print("EXPERIMENT 1 (PART B, INTERLEAVED): ROUTING SWAP ON CONVERGED MODELS")
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Task: Interleaved Sequences (L={seq_len}, vocab={vocab_size})")
    print(f"Expected convergence: >95% accuracy\n")
    
    # =========================================================================
    # PHASE 1: Train Model A (Standard)
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 1: Training Model A (Standard)")
    print("-"*80 + "\n")
    
    model_a = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=seq_len,
        dropout=0.0
    )
    result_a = train_model(
        model_a, train_loader, val_loader,
        loss_fn_name='standard',
        n_steps=n_steps,
        lr=lr,
        device=device,
        vocab_size=vocab_size,
        inject_noise_training=False
    )
    
    if result_a['final_acc'] < 0.90:
        print(f"WARNING: Model A did not converge ({result_a['final_acc']:.3f} < 0.90)")
    
    # =========================================================================
    # PHASE 2: Train Model B (Noisy)
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 2: Training Model B (Noisy)")
    print("-"*80 + "\n")
    
    model_b = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=seq_len,
        dropout=0.0
    )
    result_b = train_model(
        model_b, train_loader, val_loader,
        loss_fn_name='noisy',
        n_steps=n_steps,
        lr=lr,
        device=device,
        vocab_size=vocab_size,
        inject_noise_training=True,
        noise_scale=2.0
    )
    
    if result_b['final_acc'] < 0.90:
        print(f"WARNING: Model B did not converge ({result_b['final_acc']:.3f} < 0.90)")
    
    # =========================================================================
    # PHASE 3: Routing Swaps (1, 2, 4 heads)
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 3: Routing Swaps (Evaluate WITHOUT Retraining)")
    print("-"*80 + "\n")
    
    # Save model states
    model_a_state = model_a.state_dict()
    model_b_state = model_b.state_dict()
    
    results = {
        'model_a': {
            'acc': result_a['final_acc'],
            'history': result_a['history'],
            'converged': result_a['final_acc'] > 0.90
        },
        'model_b': {
            'acc': result_b['final_acc'], 
            'history': result_b['history'],
            'converged': result_b['final_acc'] > 0.90
        },
        'swaps': []
    }
    
    # Compute baseline metrics between A and B
    baseline_metrics_ab = compute_baseline_metrics(model_a, model_b, val_loader, device)
    results['baseline_comparison'] = {
        'attn_cosim': baseline_metrics_ab.get('attn_cosim_vs_baseline', None),
        'resid_cosim': baseline_metrics_ab.get('resid_direction_cosim', None),
        'qk_norm_diff': baseline_metrics_ab.get('qk_norm_drift', None)
    }
    print(f"Baseline A vs B: Attn CosSim = {baseline_metrics_ab.get('attn_cosim_vs_baseline', 'N/A'):.3f}, "
          f"Resid CosSim = {baseline_metrics_ab.get('resid_direction_cosim', 'N/A'):.3f}")
    
    for n_heads_swap in [1, 2, 4]:
        print(f"\n--- Swapping {n_heads_swap} head(s) from B → A ---")
        
        # Create hybrid model (start from A)
        model_hybrid = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=seq_len,
            dropout=0.0
        )
        model_hybrid.load_state_dict(model_a_state)
        model_hybrid.to(device)
        
        # Load model B for swapping
        model_b_temp = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=seq_len,
            dropout=0.0
        )
        model_b_temp.load_state_dict(model_b_state)
        model_b_temp.to(device)
        
        # Swap QK parameters
        swap_info = swap_qk_parameters(model_hybrid, model_b_temp, n_heads=n_heads_swap)
        print(f"  Swapped {swap_info['swapped_params']} parameters")
        
        # Evaluate (NO RETRAINING - this is the key causal test)
        hybrid_acc = validate(model_hybrid, val_loader, device, vocab_size)
        
        # Compute metrics vs both parents
        metrics_vs_a = compute_baseline_metrics(model_hybrid, model_a, val_loader, device)
        metrics_vs_b = compute_baseline_metrics(model_hybrid, model_b_temp, val_loader, device)
        
        # Suppressor measurement
        suppressor_metrics = measure_suppressor_strength(model_hybrid, val_loader, device)
        
        swap_result = {
            'n_heads_swapped': n_heads_swap,
            'accuracy': hybrid_acc,
            'swap_info': swap_info,
            'vs_model_a': {
                'attn_cosim': metrics_vs_a.get('attn_cosim_vs_baseline', None),
                'resid_cosim': metrics_vs_a.get('resid_direction_cosim', None),
                'qk_drift': metrics_vs_a.get('qk_norm_drift', None)
            },
            'vs_model_b': {
                'attn_cosim': metrics_vs_b.get('attn_cosim_vs_baseline', None),
                'resid_cosim': metrics_vs_b.get('resid_direction_cosim', None),
                'qk_drift': metrics_vs_b.get('qk_norm_drift', None)
            },
            'suppressor': suppressor_metrics
        }
        
        results['swaps'].append(swap_result)
        
        print(f"  Accuracy: {hybrid_acc:.3f}")
        print(f"  vs A: Attn CosSim = {metrics_vs_a.get('attn_cosim_vs_baseline', 'N/A'):.3f}, "
              f"Resid CosSim = {metrics_vs_a.get('resid_direction_cosim', 'N/A'):.3f}")
        print(f"  vs B: Attn CosSim = {metrics_vs_b.get('attn_cosim_vs_baseline', 'N/A'):.3f}, "
              f"Resid CosSim = {metrics_vs_b.get('resid_direction_cosim', 'N/A'):.3f}")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    print("\n" + "="*80)
    print("EXPERIMENT 1 ANALYSIS")
    print("="*80 + "\n")
    
    print(f"Model A (Standard): {results['model_a']['acc']:.3f}")
    print(f"Model B (Noisy):    {results['model_b']['acc']:.3f}")
    
    behavioral_gap = abs(results['model_a']['acc'] - results['model_b']['acc'])
    print(f"Behavioral Gap: {behavioral_gap:.3f}\n")
    
    print("Swap Results:")
    for swap in results['swaps']:
        n_heads = swap['n_heads_swapped']
        acc = swap['accuracy']
        drift_from_a = abs(acc - results['model_a']['acc'])
        drift_toward_b = abs(acc - results['model_b']['acc'])
        
        print(f"  {n_heads} head(s): Acc={acc:.3f}, "
              f"Δ from A={drift_from_a:.3f}, "
              f"Δ from B={drift_toward_b:.3f}")
    
    # Key test: does routing swap cause interpretable behavioral change?
    print("\n--- Causal Interpretation ---")
    final_swap = results['swaps'][-1]  # All 4 heads
    
    # If hybrid is closer to B than to A, routing determined behavior
    hybrid_acc = final_swap['accuracy']
    distance_to_a = abs(hybrid_acc - results['model_a']['acc'])
    distance_to_b = abs(hybrid_acc - results['model_b']['acc'])
    
    if distance_to_b < distance_to_a:
        print("Hybrid is closer to B than A → Routing (G) causally influences behavior ✓")
        results['causal_direction'] = 'G_causal_toward_B'
    elif hybrid_acc < min(results['model_a']['acc'], results['model_b']['acc']):
        print("Hybrid performs worse than both parents → G/S mismatch (expected) ✓")
        results['causal_direction'] = 'G_S_mismatch'
    else:
        print("Hybrid behavior not clearly attributable to routing")
        results['causal_direction'] = 'unclear'
    
    # Success criterion
    # Transfer is meaningful if the hybrid behaves differently from A
    transfer_occurred = distance_to_a > 0.01 or distance_to_b < behavioral_gap
    print(f"\nTransfer Occurred: {'✓ YES' if transfer_occurred else '✗ NO'}")
    results['transfer_occurred'] = transfer_occurred
    
    # Save results
    output_path = Path("data/exp_1_interleaved_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def run_experiment_1_interleaved(
    seq_len: int = 128,
    vocab_size: int = 4096,
    d_model: int = 128,
    n_heads: int = 4,
    n_steps: int = 10000,
    batch_size: int = 64,
    lr: float = 1e-3,
    start_seed: int = 42,
    n_seeds: int = 5,
    device: str = 'cuda'
):
    seeds = range(start_seed, start_seed + n_seeds)
    all_results = []
    
    print("\n" + "="*80)
    print(f"RUNNING INTERLEAVED SWAP EXPERIMENT (n={n_seeds} seeds)")
    print("="*80 + "\n")
    
    for i, seed in enumerate(seeds):
        print(f"\n>>> SEED {seed} ({i+1}/{len(seeds)})")
        res = run_single_seed(
            seq_len=seq_len, vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_steps=n_steps, batch_size=batch_size, lr=lr, seed=seed, device=device
        )
        all_results.append(res)
    
    # helper for CI
    def get_stats(vals):
        arr = np.array(vals)
        mean = np.mean(arr)
        if len(arr) > 1:
            se = np.std(arr, ddof=1) / np.sqrt(len(arr))
        else:
            se = 0.0
        return mean, 1.96 * se
        
    print("\n" + "="*80)
    print("TABLE 5: ROUTING SWAP RESULTS (Multi-seed Confidence)")
    print("="*80 + "\n")
    
    # Baseline
    acc_a = [r['model_a']['acc'] for r in all_results]
    acc_b = [r['model_b']['acc'] for r in all_results]
    
    m_a, ci_a = get_stats(acc_a)
    m_b, ci_b = get_stats(acc_b)
    
    print(f"Model A (Standard): {m_a:.3%} ± {ci_a:.3%}")
    print(f"Model B (Noisy):    {m_b:.3%} ± {ci_b:.3%}")
    
    # Swaps
    # Assume all runs have same swap sizes in same order
    n_swaps = len(all_results[0]['swaps'])
    
    print("\nSwap Performance (Transfer):")
    for i in range(n_swaps):
        swap_head_count = all_results[0]['swaps'][i]['n_heads_swapped']
        
        swap_accs = [r['swaps'][i]['accuracy'] for r in all_results]
        m, ci = get_stats(swap_accs)
        
        print(f"  {swap_head_count} Heads Swapped: {m:.3%} ± {ci:.3%}")
        
    # Aggegrate crash test
    # If 4 heads (100% Geometry) is swapped, what is the result?
    final_swap_accs = [r['swaps'][-1]['accuracy'] for r in all_results]
    m_final, ci_final = get_stats(final_swap_accs)
    print(f"\nFinal Result (Full G swap): {m_final:.3%} ± {ci_final:.3%}")
    
    # Save aggregate
    save_path = Path("data/exp_1_multiseed_results.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved aggregate results to {save_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 (Part B, Interleaved): Routing Swap")
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--vocab_size', type=int, default=4096, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_steps', type=int, default=10000, help='Training steps per model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (start)')
    parser.add_argument('--n_seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--quick_test', action='store_true', help='Quick test (2000 steps)')
    
    args = parser.parse_args()
    
    n_steps = 2000 if args.quick_test else args.n_steps
    
    run_experiment_1_interleaved(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_steps=n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        start_seed=args.seed,
        n_seeds=args.n_seeds,
        device=args.device
    )


if __name__ == '__main__':
    main()
