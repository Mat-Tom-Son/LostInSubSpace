"""
Experiment: Optimizer State Necessity

Tests whether optimizer state is part of "learned structure" that cannot be
transferred, extending the G×S framework to G×S×O.

Hypothesis: Optimizer state (Adam momentum/variance) encodes trajectory information
that is jointly specialized with model weights. Resetting it causes degradation
similar to transplanting G between models.

Protocol:
1. Train model to convergence on modular addition (save weights + optimizer)
2. Test conditions:
   - CONTROL: Continue training with full state
   - RESET: Continue with weights only (fresh optimizer)
   - WARMUP: Continue with weights + LR warmup
   - SWAP: Use optimizer state from different converged model (if applicable)

Metrics:
- Accuracy immediately after intervention
- Accuracy trajectory over continued training
- Recovery time (steps to return to baseline)
- Final accuracy after N steps

Expected result: RESET causes degradation, WARMUP partially recovers,
SWAP may cause degradation similar to G×S swap.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

from lib.logging_utils import setup_reproducibility
from lib.deep_transformer import DeepModularTransformer


def create_modular_data(p: int, train_frac: float = 0.7, seed: int = 42):
    """Create modular addition dataset: (a + b) mod p."""
    np.random.seed(seed)

    # All pairs
    pairs = [(a, b) for a in range(p) for b in range(p)]
    np.random.shuffle(pairs)

    split = int(len(pairs) * train_frac)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    def to_tensors(pairs):
        x = torch.tensor([[a, b] for a, b in pairs], dtype=torch.long)
        y = torch.tensor([(a + b) % p for a, b in pairs], dtype=torch.long)
        return x, y

    return to_tensors(train_pairs), to_tensors(val_pairs)


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    """Evaluate accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        preds = logits.argmax(dim=-1)
        acc = (preds == y.to(device)).float().mean().item()
    return acc


def train_to_convergence(
    model: nn.Module,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    lr: float = 1e-3,
    max_steps: int = 10000,
    convergence_threshold: float = 0.99,
    convergence_patience: int = 100,
    log_interval: int = 100
) -> Tuple[Dict, List[Dict]]:
    """
    Train model until convergence.

    Returns:
        (final_state, trajectory) where final_state includes optimizer
    """
    x_train, y_train = train_data
    x_val, y_val = val_data

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    trajectory = []
    converged_steps = 0

    for step in range(max_steps):
        model.train()

        # Forward
        optimizer.zero_grad()
        logits = model(x_train.to(device))
        loss = F.cross_entropy(logits, y_train.to(device))

        # Backward
        loss.backward()
        optimizer.step()

        # Evaluate
        if step % log_interval == 0 or step == max_steps - 1:
            train_acc = evaluate(model, x_train, y_train, device)
            val_acc = evaluate(model, x_val, y_val, device)

            trajectory.append({
                'step': step,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'loss': loss.item()
            })

            # Check convergence
            if val_acc >= convergence_threshold:
                converged_steps += 1
                if converged_steps >= convergence_patience // log_interval:
                    print(f"  Converged at step {step}: val_acc={val_acc*100:.2f}%")
                    break
            else:
                converged_steps = 0

    # Save full state
    final_state = {
        'model_state_dict': deepcopy(model.state_dict()),
        'optimizer_state_dict': deepcopy(optimizer.state_dict()),
        'step': step,
        'val_acc': val_acc
    }

    return final_state, trajectory


def continue_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    steps: int = 500,
    log_interval: int = 10,
    warmup_steps: int = 0
) -> List[Dict]:
    """Continue training and record trajectory."""
    x_train, y_train = train_data
    x_val, y_val = val_data

    # Get base LR
    base_lr = optimizer.param_groups[0]['lr']

    trajectory = []

    # Record initial state
    initial_acc = evaluate(model, x_val, y_val, device)
    trajectory.append({
        'step': 0,
        'val_acc': initial_acc,
        'train_acc': evaluate(model, x_train, y_train, device),
        'lr': 0 if warmup_steps > 0 else base_lr
    })

    for step in range(1, steps + 1):
        model.train()

        # Warmup LR
        if warmup_steps > 0 and step <= warmup_steps:
            lr_scale = step / warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * lr_scale

        # Forward
        optimizer.zero_grad()
        logits = model(x_train.to(device))
        loss = F.cross_entropy(logits, y_train.to(device))

        # Backward
        loss.backward()
        optimizer.step()

        # Log
        if step % log_interval == 0:
            val_acc = evaluate(model, x_val, y_val, device)
            train_acc = evaluate(model, x_train, y_train, device)
            current_lr = optimizer.param_groups[0]['lr']

            trajectory.append({
                'step': step,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'loss': loss.item(),
                'lr': current_lr
            })

    return trajectory


def run_experiment(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 1,
    lr: float = 1e-3,
    continue_steps: int = 500,
    warmup_steps: int = 100,
    n_seeds: int = 3,
    device: str = None
):
    """
    Run the optimizer state necessity experiment.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("EXPERIMENT: OPTIMIZER STATE NECESSITY")
    print("="*70)
    print(f"Task: Modular Addition (p={p})")
    print(f"Architecture: {n_layers}L, d={d_model}, h={n_heads}")
    print(f"Device: {device}")
    print(f"Seeds: {n_seeds}")
    print()

    results = {
        'config': {
            'p': p, 'd_model': d_model, 'n_heads': n_heads,
            'n_layers': n_layers, 'lr': lr, 'continue_steps': continue_steps,
            'warmup_steps': warmup_steps, 'n_seeds': n_seeds
        },
        'seeds': []
    }

    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print('='*70)

        setup_reproducibility(seed)

        # Create data
        train_data, val_data = create_modular_data(p, seed=seed)
        print(f"Data: {len(train_data[0])} train, {len(val_data[0])} val")

        # Create and train model to convergence
        print("\n[Phase 1] Training to convergence...")
        model = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        model.to(device)

        converged_state, train_trajectory = train_to_convergence(
            model, train_data, val_data, device, lr=lr
        )

        baseline_acc = converged_state['val_acc']
        print(f"  Baseline accuracy: {baseline_acc*100:.2f}%")

        seed_results = {
            'seed': seed,
            'baseline_acc': baseline_acc,
            'convergence_step': converged_state['step'],
            'conditions': {}
        }

        # ===== CONDITION A: CONTROL (full state) =====
        print("\n[Condition A] CONTROL: Continue with full state...")
        model_a = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        model_a.to(device)
        model_a.load_state_dict(deepcopy(converged_state['model_state_dict']))

        opt_a = torch.optim.AdamW(model_a.parameters(), lr=lr, weight_decay=0.1)
        opt_a.load_state_dict(deepcopy(converged_state['optimizer_state_dict']))

        traj_a = continue_training(model_a, opt_a, train_data, val_data, device,
                                   steps=continue_steps, warmup_steps=0)

        print(f"  Initial: {traj_a[0]['val_acc']*100:.2f}% -> Final: {traj_a[-1]['val_acc']*100:.2f}%")
        seed_results['conditions']['control'] = {
            'initial_acc': traj_a[0]['val_acc'],
            'final_acc': traj_a[-1]['val_acc'],
            'trajectory': traj_a
        }

        # ===== CONDITION B: RESET (fresh optimizer) =====
        print("\n[Condition B] RESET: Continue with fresh optimizer...")
        model_b = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        model_b.to(device)
        model_b.load_state_dict(deepcopy(converged_state['model_state_dict']))

        opt_b = torch.optim.AdamW(model_b.parameters(), lr=lr, weight_decay=0.1)
        # NO optimizer state loading - fresh!

        traj_b = continue_training(model_b, opt_b, train_data, val_data, device,
                                   steps=continue_steps, warmup_steps=0)

        print(f"  Initial: {traj_b[0]['val_acc']*100:.2f}% -> Final: {traj_b[-1]['val_acc']*100:.2f}%")
        seed_results['conditions']['reset'] = {
            'initial_acc': traj_b[0]['val_acc'],
            'final_acc': traj_b[-1]['val_acc'],
            'trajectory': traj_b
        }

        # ===== CONDITION C: WARMUP (fresh optimizer + warmup) =====
        print(f"\n[Condition C] WARMUP: Fresh optimizer + {warmup_steps} warmup steps...")
        model_c = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        model_c.to(device)
        model_c.load_state_dict(deepcopy(converged_state['model_state_dict']))

        opt_c = torch.optim.AdamW(model_c.parameters(), lr=lr, weight_decay=0.1)
        # NO optimizer state loading - but with warmup

        traj_c = continue_training(model_c, opt_c, train_data, val_data, device,
                                   steps=continue_steps, warmup_steps=warmup_steps)

        print(f"  Initial: {traj_c[0]['val_acc']*100:.2f}% -> Final: {traj_c[-1]['val_acc']*100:.2f}%")
        seed_results['conditions']['warmup'] = {
            'initial_acc': traj_c[0]['val_acc'],
            'final_acc': traj_c[-1]['val_acc'],
            'trajectory': traj_c
        }

        # ===== CONDITION D: LOW_LR (fresh optimizer + 10x lower LR) =====
        print("\n[Condition D] LOW_LR: Fresh optimizer + 10x lower LR...")
        model_d = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        model_d.to(device)
        model_d.load_state_dict(deepcopy(converged_state['model_state_dict']))

        opt_d = torch.optim.AdamW(model_d.parameters(), lr=lr/10, weight_decay=0.1)

        traj_d = continue_training(model_d, opt_d, train_data, val_data, device,
                                   steps=continue_steps, warmup_steps=0)

        print(f"  Initial: {traj_d[0]['val_acc']*100:.2f}% -> Final: {traj_d[-1]['val_acc']*100:.2f}%")
        seed_results['conditions']['low_lr'] = {
            'initial_acc': traj_d[0]['val_acc'],
            'final_acc': traj_d[-1]['val_acc'],
            'trajectory': traj_d
        }

        results['seeds'].append(seed_results)

    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    conditions = ['control', 'reset', 'warmup', 'low_lr']

    print(f"\n{'Condition':<15} {'Initial Acc':<15} {'Final Acc':<15} {'Delta':<15}")
    print("-"*60)

    summary = {}
    for cond in conditions:
        initial_accs = [s['conditions'][cond]['initial_acc'] for s in results['seeds']]
        final_accs = [s['conditions'][cond]['final_acc'] for s in results['seeds']]

        mean_init = np.mean(initial_accs)
        mean_final = np.mean(final_accs)
        std_final = np.std(final_accs)
        delta = mean_final - mean_init

        summary[cond] = {
            'mean_initial': mean_init,
            'mean_final': mean_final,
            'std_final': std_final,
            'delta': delta
        }

        print(f"{cond:<15} {mean_init*100:>6.2f}%        {mean_final*100:>6.2f}% +/- {std_final*100:.2f}%  {delta*100:>+6.2f}%")

    results['summary'] = summary

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    control_final = summary['control']['mean_final']
    reset_final = summary['reset']['mean_final']
    warmup_final = summary['warmup']['mean_final']
    low_lr_final = summary['low_lr']['mean_final']

    reset_drop = control_final - reset_final
    warmup_recovery = warmup_final - reset_final
    low_lr_recovery = low_lr_final - reset_final

    print(f"\nOptimizer Reset Effect: {reset_drop*100:+.2f}% (CONTROL vs RESET)")
    print(f"Warmup Recovery: {warmup_recovery*100:+.2f}% (relative to RESET)")
    print(f"Low LR Recovery: {low_lr_recovery*100:+.2f}% (relative to RESET)")

    if reset_drop > 0.05:
        print("\n[PASS] SIGNIFICANT: Optimizer state reset causes degradation!")
        print("  This supports extending G×S to G×S×O framework.")
    else:
        print("\n[FAIL] Optimizer reset effect is minimal in this setup.")

    # Save results
    output_path = Path("data/exp_optimizer_state_necessity_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=113, help='Modulus for addition task')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--n_seeds', type=int, default=3, help='Number of random seeds')
    parser.add_argument('--continue_steps', type=int, default=500, help='Steps to continue after loading')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps for condition C')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer steps')

    args = parser.parse_args()

    if args.quick:
        args.n_seeds = 1
        args.continue_steps = 100
        args.warmup_steps = 20

    run_experiment(
        p=args.p,
        n_layers=args.n_layers,
        n_seeds=args.n_seeds,
        continue_steps=args.continue_steps,
        warmup_steps=args.warmup_steps
    )


if __name__ == '__main__':
    main()
