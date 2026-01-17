"""
EXPERIMENT 4.2b: Stability Characterization (4-Layer)

Measures the stochastic escape process from the metastable generalization regime.

Key Discovery (2026-01-15):
At 4 layers, we observe THREE distinct regimes:
1. Transient generalization: Both models hit ~100% accuracy early (~step 2000)
2. Metastable oscillatory: Anchor collapses/recovers repeatedly (underdamped)
3. Regularized escape: Ortho penalty acts as symmetry-breaking perturbation

The orthogonality penalty is NOT helping by "adding capacity" or "forcing diversity".
It acts as a DYNAMICAL REGULARIZER that:
- Breaks symmetry between competing solutions
- Suppresses oscillatory modes
- Pushes the system into a deeper basin
- Enables annealing into stable attractor

Metrics Tracked:
- time_to_generalization: First step where val_acc > 95%
- collapse_count: Number of times val_acc drops from >95% to <50%
- time_to_stability: First step after which val_acc never drops below 90% (or -1)
- stability_achieved: Boolean - did the model ever stabilize?
- val_acc_trajectory: Full trajectory for survival curve plotting

This characterizes a STOCHASTIC ESCAPE PROCESS, not standard ML convergence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import json
import copy

from lib.logging_utils import setup_reproducibility
from lib.part_b_utils import freeze_parameters, get_qk_parameters, verify_freeze
from lib.deep_transformer import DeepModularTransformer


class ModularAdditionDataset(Dataset):
    """Modular addition: (a + b) mod p"""

    def __init__(self, p: int = 113, split: str = 'train', train_frac: float = 0.3):
        self.p = p
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        np.random.seed(42)
        np.random.shuffle(all_pairs)
        n_train = int(len(all_pairs) * train_frac)
        if split == 'train':
            self.pairs = all_pairs[:n_train]
        else:
            self.pairs = all_pairs[n_train:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        x = torch.tensor([a, b], dtype=torch.long)
        y = torch.tensor((a + b) % self.p, dtype=torch.long)
        return x, y


def validate(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def compute_stability_metrics(val_acc_history: List[Tuple[int, float]], 
                               gen_threshold: float = 0.95,
                               collapse_threshold: float = 0.50,
                               stability_threshold: float = 0.90) -> Dict:
    """
    Compute stability metrics from validation accuracy trajectory.
    
    Returns:
        time_to_generalization: First step with val_acc > gen_threshold (-1 if never)
        collapse_count: Number of drops from >gen_threshold to <collapse_threshold
        time_to_stability: First step after which val_acc never drops below stability_threshold (-1 if never)
        stability_achieved: Boolean
    """
    
    # Time to generalization
    time_to_gen = -1
    for step, acc in val_acc_history:
        if acc >= gen_threshold:
            time_to_gen = step
            break
    
    # Collapse count
    collapse_count = 0
    was_above = False
    for step, acc in val_acc_history:
        if acc >= gen_threshold:
            was_above = True
        elif was_above and acc < collapse_threshold:
            collapse_count += 1
            was_above = False  # Reset, need to re-generalize
    
    # Time to stability (last point where we permanently stay above stability_threshold)
    time_to_stability = -1
    n = len(val_acc_history)
    for i in range(n):
        step, acc = val_acc_history[i]
        if acc >= stability_threshold:
            # Check if we ever drop below from here onwards
            stable_from_here = True
            for j in range(i + 1, n):
                _, future_acc = val_acc_history[j]
                if future_acc < stability_threshold:
                    stable_from_here = False
                    break
            if stable_from_here:
                time_to_stability = step
                break
    
    stability_achieved = time_to_stability != -1
    
    return {
        'time_to_generalization': time_to_gen,
        'collapse_count': collapse_count,
        'time_to_stability': time_to_stability,
        'stability_achieved': stability_achieved
    }


def run_single_seed(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    max_warmup_steps: int = 5000,
    freeze_threshold: float = 0.90,
    hysteresis_checks: int = 2,
    training_steps: int = 20000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    ortho_lambda: float = 0.5,
    train_frac: float = 0.5,
    seed: int = 42,
    device: str = 'cuda',
    log_interval: int = 200  # More frequent logging for survival curves
) -> Dict:
    """
    Run stability characterization experiment.
    Tracks both anchor (no ortho) and probe (with ortho) from same Young G.
    """

    print("\n" + "="*80)
    print(f"STABILITY CHARACTERIZATION (4-LAYER, seed={seed})")
    print("="*80 + "\n")

    setup_reproducibility(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Architecture: n_layers={n_layers}, d_model={d_model}")
    print(f"Training: {training_steps} steps, log every {log_interval}")
    print(f"Ortho λ: {ortho_lambda}\n")

    # Data
    train_dataset = ModularAdditionDataset(p=p, split='train', train_frac=train_frac)
    val_dataset = ModularAdditionDataset(p=p, split='val', train_frac=train_frac)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # =========================================================================
    # PHASE 1: Warmup with Auto-Detection
    # =========================================================================

    print("-"*80)
    print(f"PHASE 1: WARMUP (Auto-freeze when val_acc > {freeze_threshold:.0%})")
    print("-"*80 + "\n")

    base_model = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    base_model.to(device)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)

    warmup_history = []
    consecutive_above = 0
    actual_freeze_step = None
    
    for step in range(max_warmup_steps):
        base_model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = base_model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            val_acc = validate(base_model, val_loader, device)
            warmup_history.append((step, val_acc))
            
            if val_acc >= freeze_threshold:
                consecutive_above += 1
            else:
                consecutive_above = 0
            
            if consecutive_above >= hysteresis_checks:
                actual_freeze_step = step
                print(f">>> CRITICAL PERIOD at step {step} (val_acc={val_acc:.3f})")
                break
    
    if actual_freeze_step is None:
        actual_freeze_step = max_warmup_steps
        print(f"WARNING: Max warmup reached, freezing at {max_warmup_steps}")

    young_g_state = copy.deepcopy(base_model.state_dict())
    young_qk = get_qk_parameters(base_model)

    # =========================================================================
    # PHASE 2: Train ANCHOR (no ortho) - baseline metastable dynamics
    # =========================================================================

    print("\n" + "-"*80)
    print("PHASE 2: ANCHOR MODEL (No Ortho - Baseline Dynamics)")
    print("-"*80 + "\n")

    anchor_model = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    anchor_model.load_state_dict(young_g_state)
    anchor_model.to(device)
    freeze_parameters(anchor_model, freeze_qk=True)
    
    anchor_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, anchor_model.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    anchor_trajectory = []
    train_iter = iter(train_loader)

    for step in range(training_steps):
        anchor_model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, y = batch
        x, y = x.to(device), y.to(device)
        anchor_optimizer.zero_grad()
        logits = anchor_model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        anchor_optimizer.step()

        if step % log_interval == 0 or step == training_steps - 1:
            val_acc = validate(anchor_model, val_loader, device)
            anchor_trajectory.append((step, val_acc))
            print(f"Anchor {step:5d} | Val: {val_acc:.3f}")

    anchor_final_acc = validate(anchor_model, val_loader, device)
    anchor_metrics = compute_stability_metrics(anchor_trajectory)
    
    print(f"\nAnchor Final: {anchor_final_acc:.3f}")
    print(f"  Time to Gen: {anchor_metrics['time_to_generalization']}")
    print(f"  Collapses: {anchor_metrics['collapse_count']}")
    print(f"  Time to Stability: {anchor_metrics['time_to_stability']}")
    print(f"  Stable: {anchor_metrics['stability_achieved']}")

    # =========================================================================
    # PHASE 3: Train PROBE (with ortho) - regularized dynamics
    # =========================================================================

    print("\n" + "-"*80)
    print(f"PHASE 3: PROBE MODEL (Ortho λ={ortho_lambda} - Regularized Dynamics)")
    print("-"*80 + "\n")

    probe_model = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    probe_model.load_state_dict(young_g_state)
    probe_model.to(device)
    freeze_parameters(probe_model, freeze_qk=True)

    probe_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, probe_model.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    probe_trajectory = []
    train_iter = iter(train_loader)
    
    # Lambda warmup
    lambda_warmup_steps = min(2000, training_steps // 5)

    for step in range(training_steps):
        probe_model.train()
        anchor_model.eval()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, y = batch
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            _ = anchor_model(x)
            anchor_resid = anchor_model.get_residual().detach()

        probe_logits = probe_model(x)
        probe_resid = probe_model.get_residual()

        ce_loss = F.cross_entropy(probe_logits, y)
        
        anchor_norm = F.normalize(anchor_resid, dim=-1)
        probe_norm = F.normalize(probe_resid, dim=-1)
        cosim = (anchor_norm * probe_norm).sum(dim=-1).mean()
        ortho_loss = torch.abs(cosim)

        # Lambda warmup
        if step < lambda_warmup_steps:
            effective_lambda = ortho_lambda * (step / lambda_warmup_steps)
        else:
            effective_lambda = ortho_lambda

        total_loss = ce_loss + effective_lambda * ortho_loss

        probe_optimizer.zero_grad()
        total_loss.backward()
        probe_optimizer.step()

        if step % log_interval == 0 or step == training_steps - 1:
            val_acc = validate(probe_model, val_loader, device)
            probe_trajectory.append((step, val_acc))
            print(f"Probe  {step:5d} | Val: {val_acc:.3f} | CosSim: {cosim.item():.3f}")

    probe_final_acc = validate(probe_model, val_loader, device)
    probe_metrics = compute_stability_metrics(probe_trajectory)
    
    # Final cosine similarity
    final_cosims = []
    anchor_model.eval()
    probe_model.eval()
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            _ = anchor_model(x)
            _ = probe_model(x)
            a_resid = F.normalize(anchor_model.get_residual(), dim=-1)
            p_resid = F.normalize(probe_model.get_residual(), dim=-1)
            final_cosims.append((a_resid * p_resid).sum(dim=-1).mean().item())
    final_cosim = np.mean(final_cosims)

    print(f"\nProbe Final: {probe_final_acc:.3f}")
    print(f"  Time to Gen: {probe_metrics['time_to_generalization']}")
    print(f"  Collapses: {probe_metrics['collapse_count']}")
    print(f"  Time to Stability: {probe_metrics['time_to_stability']}")
    print(f"  Stable: {probe_metrics['stability_achieved']}")
    print(f"  Final CosSim: {final_cosim:.4f}")

    # =========================================================================
    # RESULTS
    # =========================================================================

    print("\n" + "="*80)
    print("STABILITY COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<25} {'Anchor':<15} {'Probe':<15}")
    print("-"*55)
    print(f"{'Final Accuracy':<25} {anchor_final_acc:.3f}{'':>10} {probe_final_acc:.3f}")
    print(f"{'Time to Generalization':<25} {anchor_metrics['time_to_generalization']:<15} {probe_metrics['time_to_generalization']}")
    print(f"{'Collapse Count':<25} {anchor_metrics['collapse_count']:<15} {probe_metrics['collapse_count']}")
    print(f"{'Time to Stability':<25} {anchor_metrics['time_to_stability']:<15} {probe_metrics['time_to_stability']}")
    print(f"{'Stability Achieved':<25} {anchor_metrics['stability_achieved']!s:<15} {probe_metrics['stability_achieved']!s}")

    results = {
        'config': {
            'n_layers': n_layers,
            'd_model': d_model,
            'n_heads': n_heads,
            'training_steps': training_steps,
            'ortho_lambda': ortho_lambda,
            'seed': seed
        },
        'warmup': {
            'actual_freeze_step': actual_freeze_step,
            'history': warmup_history
        },
        'anchor': {
            'trajectory': anchor_trajectory,
            'final_acc': float(anchor_final_acc),
            'metrics': anchor_metrics
        },
        'probe': {
            'trajectory': probe_trajectory,
            'final_acc': float(probe_final_acc),
            'final_cosim': float(final_cosim),
            'metrics': probe_metrics
        }
    }

    return results


def run_experiment(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    max_warmup_steps: int = 5000,
    freeze_threshold: float = 0.90,
    hysteresis_checks: int = 2,
    training_steps: int = 20000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    ortho_lambda: float = 0.5,
    train_frac: float = 0.5,
    start_seed: int = 42,
    n_seeds: int = 5,
    device: str = 'cuda'
):
    """Run stability characterization across multiple seeds."""
    
    seeds = range(start_seed, start_seed + n_seeds)
    all_results = []

    print("\n" + "="*80)
    print(f"STABILITY CHARACTERIZATION EXPERIMENT (n={n_seeds} seeds)")
    print("Measuring stochastic escape from metastable regime")
    print("="*80 + "\n")

    for i, seed in enumerate(seeds):
        print(f"\n>>> SEED {seed} ({i+1}/{len(seeds)})")
        res = run_single_seed(
            p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            max_warmup_steps=max_warmup_steps, freeze_threshold=freeze_threshold,
            hysteresis_checks=hysteresis_checks, training_steps=training_steps,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            ortho_lambda=ortho_lambda, train_frac=train_frac, seed=seed, device=device
        )
        all_results.append(res)

    # =========================================================================
    # AGGREGATE ANALYSIS
    # =========================================================================

    print("\n" + "="*80)
    print("AGGREGATE STABILITY ANALYSIS")
    print("="*80 + "\n")

    # Anchor stats
    anchor_stable_count = sum(1 for r in all_results if r['anchor']['metrics']['stability_achieved'])
    anchor_collapse_counts = [r['anchor']['metrics']['collapse_count'] for r in all_results]
    anchor_final_accs = [r['anchor']['final_acc'] for r in all_results]
    
    # Probe stats
    probe_stable_count = sum(1 for r in all_results if r['probe']['metrics']['stability_achieved'])
    probe_collapse_counts = [r['probe']['metrics']['collapse_count'] for r in all_results]
    probe_final_accs = [r['probe']['final_acc'] for r in all_results]

    print(f"{'Metric':<30} {'Anchor':<20} {'Probe':<20}")
    print("-"*70)
    print(f"{'Stability Rate':<30} {anchor_stable_count}/{n_seeds} ({100*anchor_stable_count/n_seeds:.0f}%){'':<5} {probe_stable_count}/{n_seeds} ({100*probe_stable_count/n_seeds:.0f}%)")
    print(f"{'Mean Collapse Count':<30} {np.mean(anchor_collapse_counts):.2f} ± {np.std(anchor_collapse_counts):.2f}{'':<3} {np.mean(probe_collapse_counts):.2f} ± {np.std(probe_collapse_counts):.2f}")
    print(f"{'Mean Final Accuracy':<30} {np.mean(anchor_final_accs):.3f} ± {np.std(anchor_final_accs):.3f}{'':<3} {np.mean(probe_final_accs):.3f} ± {np.std(probe_final_accs):.3f}")

    # Time-to-stability for those that stabilized
    anchor_tts = [r['anchor']['metrics']['time_to_stability'] for r in all_results 
                  if r['anchor']['metrics']['stability_achieved']]
    probe_tts = [r['probe']['metrics']['time_to_stability'] for r in all_results 
                 if r['probe']['metrics']['stability_achieved']]
    
    if anchor_tts:
        print(f"{'Mean Time-to-Stability':<30} {np.mean(anchor_tts):.0f} ± {np.std(anchor_tts):.0f}{'':<3}", end="")
    else:
        print(f"{'Mean Time-to-Stability':<30} N/A (none stabilized){'':<3}", end="")
    
    if probe_tts:
        print(f" {np.mean(probe_tts):.0f} ± {np.std(probe_tts):.0f}")
    else:
        print(" N/A (none stabilized)")

    # Save
    save_path = Path("data/exp_4_2b_stability_results.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {save_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Stability Characterization Experiment (4-Layer)")
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--max_warmup_steps', type=int, default=5000)
    parser.add_argument('--freeze_threshold', type=float, default=0.90)
    parser.add_argument('--hysteresis_checks', type=int, default=2)
    parser.add_argument('--training_steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0)
    parser.add_argument('--ortho_lambda', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_frac', type=float, default=0.5)

    args = parser.parse_args()

    run_experiment(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_warmup_steps=args.max_warmup_steps,
        freeze_threshold=args.freeze_threshold,
        hysteresis_checks=args.hysteresis_checks,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ortho_lambda=args.ortho_lambda,
        train_frac=args.train_frac,
        start_seed=args.seed,
        n_seeds=args.n_seeds,
        device=args.device
    )


if __name__ == '__main__':
    main()
