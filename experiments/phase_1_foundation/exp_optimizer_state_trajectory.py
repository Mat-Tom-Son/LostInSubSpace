"""
Experiment: Optimizer State - Trajectory Analysis

Focuses on the EARLY trajectory after optimizer reset to quantify
the transient degradation effect. Also tests at 4 layers where
metastability might make recovery harder.

Key questions:
1. How severe is the immediate degradation (first 10-50 steps)?
2. Does depth make recovery harder?
3. Does resetting during grokking (vs after) cause different effects?
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


def create_modular_data(p: int, train_frac: float = 0.3, seed: int = 42):
    """Create modular addition dataset with small train set (to force grokking)."""
    np.random.seed(seed)
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


def evaluate(model, x, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        preds = logits.argmax(dim=-1)
        acc = (preds == y.to(device)).float().mean().item()
    return acc


def train_steps(
    model, optimizer, train_data, val_data, device,
    steps: int, log_every: int = 1
) -> List[Dict]:
    """Train for N steps, logging every log_every steps."""
    x_train, y_train = train_data
    x_val, y_val = val_data

    trajectory = []

    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train.to(device))
        loss = F.cross_entropy(logits, y_train.to(device))
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            val_acc = evaluate(model, x_val, y_val, device)
            train_acc = evaluate(model, x_train, y_train, device)
            trajectory.append({
                'step': step,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'loss': loss.item()
            })

    return trajectory


def run_trajectory_experiment(
    p: int = 113,
    n_layers: int = 1,
    d_model: int = 128,
    n_heads: int = 4,
    lr: float = 1e-3,
    convergence_steps: int = 10000,
    post_reset_steps: int = 100,
    device: str = None,
    seed: int = 42
):
    """Run single trajectory comparison experiment."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    setup_reproducibility(seed)
    train_data, val_data = create_modular_data(p, train_frac=0.3, seed=seed)

    print(f"\n{'='*60}")
    print(f"TRAJECTORY EXPERIMENT: {n_layers}L model, seed={seed}")
    print(f"{'='*60}")

    # Train to convergence
    print("\n[Phase 1] Training to convergence...")
    model = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0)

    # Train with periodic logging
    train_traj = []
    for step in tqdm(range(convergence_steps), desc="Training"):
        model.train()
        optimizer.zero_grad()
        x_train, y_train = train_data
        logits = model(x_train.to(device))
        loss = F.cross_entropy(logits, y_train.to(device))
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            val_acc = evaluate(model, val_data[0], val_data[1], device)
            train_traj.append({'step': step, 'val_acc': val_acc})

            if val_acc > 0.99:
                print(f"  Converged at step {step}: {val_acc*100:.2f}%")
                break

    # Save converged state
    converged_state = {
        'model': deepcopy(model.state_dict()),
        'optimizer': deepcopy(optimizer.state_dict())
    }
    baseline_acc = evaluate(model, val_data[0], val_data[1], device)
    print(f"  Baseline accuracy: {baseline_acc*100:.2f}%")

    results = {
        'baseline_acc': baseline_acc,
        'convergence_step': step,
        'train_trajectory': train_traj,
        'conditions': {}
    }

    # ===== Compare trajectories =====
    print(f"\n[Phase 2] Comparing post-reset trajectories ({post_reset_steps} steps)...")

    conditions = {
        'control': {'load_opt': True, 'lr_scale': 1.0},
        'reset': {'load_opt': False, 'lr_scale': 1.0},
        'reset_low_lr': {'load_opt': False, 'lr_scale': 0.1},
        'reset_warmup': {'load_opt': False, 'lr_scale': 0.01},  # Start very low
    }

    for cond_name, cond_config in conditions.items():
        # Fresh model with loaded weights
        model_c = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        model_c.to(device)
        model_c.load_state_dict(deepcopy(converged_state['model']))

        # Optimizer
        opt_c = torch.optim.AdamW(model_c.parameters(), lr=lr * cond_config['lr_scale'], weight_decay=1.0)
        if cond_config['load_opt']:
            opt_c.load_state_dict(deepcopy(converged_state['optimizer']))

        # Train and log EVERY step for first 100 steps
        traj = train_steps(model_c, opt_c, train_data, val_data, device,
                          steps=post_reset_steps, log_every=1)

        results['conditions'][cond_name] = {
            'config': cond_config,
            'trajectory': traj,
            'initial_acc': traj[0]['val_acc'],
            'min_acc': min(t['val_acc'] for t in traj),
            'final_acc': traj[-1]['val_acc'],
        }

        min_acc = results['conditions'][cond_name]['min_acc']
        min_step = next(i for i, t in enumerate(traj) if t['val_acc'] == min_acc)
        print(f"  {cond_name:15s}: {traj[0]['val_acc']*100:.2f}% -> min {min_acc*100:.2f}% (step {min_step}) -> {traj[-1]['val_acc']*100:.2f}%")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--convergence_steps', type=int, default=15000)
    parser.add_argument('--post_reset_steps', type=int, default=100)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("OPTIMIZER STATE TRAJECTORY ANALYSIS")
    print("="*70)
    print(f"Layers: {args.n_layers}")
    print(f"Seeds: {args.n_seeds}")

    all_results = {'config': vars(args), 'seeds': []}

    for seed in range(args.n_seeds):
        results = run_trajectory_experiment(
            n_layers=args.n_layers,
            convergence_steps=args.convergence_steps,
            post_reset_steps=args.post_reset_steps,
            device=device,
            seed=seed
        )
        all_results['seeds'].append({'seed': seed, **results})

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Maximum Degradation After Reset")
    print("="*70)

    for cond in ['control', 'reset', 'reset_low_lr']:
        drops = []
        for s in all_results['seeds']:
            initial = s['conditions'][cond]['initial_acc']
            min_acc = s['conditions'][cond]['min_acc']
            drops.append(initial - min_acc)

        mean_drop = np.mean(drops) * 100
        std_drop = np.std(drops) * 100
        print(f"  {cond:15s}: {mean_drop:+.2f}% +/- {std_drop:.2f}% max drop")

    # Save
    output_path = Path(f"data/exp_optimizer_trajectory_{args.n_layers}L_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
