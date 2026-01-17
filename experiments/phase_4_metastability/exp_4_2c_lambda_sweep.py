"""
EXPERIMENT 4.2c: Lambda Phase Sensitivity Sweep

Tests whether orthogonality λ is a local dynamical bias vs. a landscape-altering force.

Key Questions:
1. Monotonic reduction in collapse count with λ?
2. Slight leftward shift in survival curve?
3. Evidence of optimal intermediate λ?
4. Do all λ share same asymptote and timescale?

If all λ share the same asymptote and timescale, that STRENGTHENS the claim that
ortho is a local dynamical bias, not a landscape restructuring force.

Protocol:
- λ ∈ {0, 0.05, 0.3}
- 8 seeds per λ
- Track same stability metrics as exp_4_2b
- Generate comparative survival curves
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple
import argparse
import json
import copy

from lib.logging_utils import setup_reproducibility
from lib.part_b_utils import freeze_parameters, get_qk_parameters
from lib.deep_transformer import DeepModularTransformer


class ModularAdditionDataset(Dataset):
    def __init__(self, p: int = 113, split: str = 'train', train_frac: float = 0.3):
        self.p = p
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        np.random.seed(42)
        np.random.shuffle(all_pairs)
        n_train = int(len(all_pairs) * train_frac)
        self.pairs = all_pairs[:n_train] if split == 'train' else all_pairs[n_train:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        return torch.tensor([a, b], dtype=torch.long), torch.tensor((a + b) % self.p, dtype=torch.long)


def validate(model: nn.Module, dataloader: DataLoader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def compute_stability_metrics(trajectory: List[Tuple[int, float]], 
                               gen_threshold: float = 0.95,
                               collapse_threshold: float = 0.50,
                               stability_threshold: float = 0.90) -> Dict:
    # Time to generalization
    time_to_gen = -1
    for step, acc in trajectory:
        if acc >= gen_threshold:
            time_to_gen = step
            break
    
    # Collapse count
    collapse_count = 0
    was_above = False
    for step, acc in trajectory:
        if acc >= gen_threshold:
            was_above = True
        elif was_above and acc < collapse_threshold:
            collapse_count += 1
            was_above = False
    
    # Time to stability
    time_to_stability = -1
    n = len(trajectory)
    for i in range(n):
        step, acc = trajectory[i]
        if acc >= stability_threshold:
            stable_from_here = all(trajectory[j][1] >= stability_threshold for j in range(i + 1, n))
            if stable_from_here:
                time_to_stability = step
                break
    
    return {
        'time_to_generalization': time_to_gen,
        'collapse_count': collapse_count,
        'time_to_stability': time_to_stability,
        'stability_achieved': time_to_stability != -1
    }


def run_single_condition(
    ortho_lambda: float,
    young_g_state: dict,
    young_qk: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_steps: int,
    lr: float,
    weight_decay: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    p: int,
    device,
    log_interval: int = 200
) -> Tuple[List[Tuple[int, float]], Dict]:
    """Train a single model with given λ and return trajectory + metrics."""
    
    model = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    model.load_state_dict(young_g_state)
    model.to(device)
    freeze_parameters(model, freeze_qk=True)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    
    trajectory = []
    train_iter = iter(train_loader)
    lambda_warmup = min(2000, training_steps // 5)
    
    for step in range(training_steps):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        # For λ > 0, we need an anchor - use a detached copy of current residual as self-reference
        # This creates repulsion from the model's own recent state (simpler than separate anchor)
        if ortho_lambda > 0 and step >= lambda_warmup:
            resid = model.get_residual()
            resid_norm = F.normalize(resid.detach(), dim=-1)  # Detached self
            current_norm = F.normalize(resid, dim=-1)
            # Penalize alignment with recent self (encourages exploration)
            cosim = (resid_norm * current_norm).sum(dim=-1).mean()
            effective_lambda = ortho_lambda * min(1.0, step / lambda_warmup)
            loss = loss + effective_lambda * torch.abs(cosim)
        
        loss.backward()
        optimizer.step()
        
        if step % log_interval == 0 or step == training_steps - 1:
            val_acc = validate(model, val_loader, device)
            trajectory.append((step, val_acc))
    
    final_acc = validate(model, val_loader, device)
    metrics = compute_stability_metrics(trajectory)
    metrics['final_acc'] = final_acc
    
    return trajectory, metrics


def run_lambda_sweep(
    lambdas: List[float] = [0.0, 0.05, 0.3],
    n_seeds: int = 8,
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    max_warmup_steps: int = 5000,
    freeze_threshold: float = 0.90,
    training_steps: int = 20000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    train_frac: float = 0.5,
    start_seed: int = 42,
    device: str = 'cuda'
):
    """Run stability characterization across multiple λ values."""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("LAMBDA PHASE SENSITIVITY SWEEP")
    print(f"λ ∈ {lambdas}, {n_seeds} seeds each")
    print("="*80 + "\n")
    
    all_results = {lam: [] for lam in lambdas}
    
    for seed_idx in range(n_seeds):
        seed = start_seed + seed_idx
        print(f"\n>>> SEED {seed} ({seed_idx+1}/{n_seeds})")
        
        setup_reproducibility(seed)
        
        # Data
        train_dataset = ModularAdditionDataset(p=p, split='train', train_frac=train_frac)
        val_dataset = ModularAdditionDataset(p=p, split='val', train_frac=train_frac)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Warmup to capture Young G (shared across all λ for this seed)
        print("  Capturing Young G...")
        base_model = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        base_model.to(device)
        optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
        train_iter = iter(train_loader)
        
        consecutive_above = 0
        actual_freeze_step = max_warmup_steps
        
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
            loss = F.cross_entropy(base_model(x), y)
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                val_acc = validate(base_model, val_loader, device)
                if val_acc >= freeze_threshold:
                    consecutive_above += 1
                    if consecutive_above >= 2:
                        actual_freeze_step = step
                        print(f"  Critical period at step {step}")
                        break
                else:
                    consecutive_above = 0
        
        young_g_state = copy.deepcopy(base_model.state_dict())
        young_qk = get_qk_parameters(base_model)
        
        # Run each λ condition from same Young G
        for lam in lambdas:
            print(f"  Training λ={lam}...")
            trajectory, metrics = run_single_condition(
                ortho_lambda=lam,
                young_g_state=young_g_state,
                young_qk=young_qk,
                train_loader=train_loader,
                val_loader=val_loader,
                training_steps=training_steps,
                lr=lr,
                weight_decay=weight_decay,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                p=p,
                device=device
            )
            
            all_results[lam].append({
                'seed': seed,
                'freeze_step': actual_freeze_step,
                'trajectory': trajectory,
                'metrics': metrics
            })
            
            status = "✓ Stable" if metrics['stability_achieved'] else "✗ Unstable"
            print(f"    {status} | Collapses: {metrics['collapse_count']} | Final: {metrics['final_acc']:.3f}")
    
    # Aggregate analysis
    print("\n" + "="*80)
    print("LAMBDA SWEEP RESULTS")
    print("="*80 + "\n")
    
    print(f"{'λ':<10} {'Stability':<15} {'Collapses':<20} {'Final Acc':<20}")
    print("-"*65)
    
    summary = {}
    for lam in lambdas:
        results = all_results[lam]
        stability_rate = sum(1 for r in results if r['metrics']['stability_achieved']) / len(results)
        collapse_counts = [r['metrics']['collapse_count'] for r in results]
        final_accs = [r['metrics']['final_acc'] for r in results]
        
        summary[lam] = {
            'stability_rate': stability_rate,
            'mean_collapses': np.mean(collapse_counts),
            'std_collapses': np.std(collapse_counts),
            'mean_acc': np.mean(final_accs),
            'std_acc': np.std(final_accs)
        }
        
        print(f"{lam:<10} {stability_rate*100:.0f}%{'':<10} "
              f"{np.mean(collapse_counts):.2f} ± {np.std(collapse_counts):.2f}{'':<5} "
              f"{np.mean(final_accs):.3f} ± {np.std(final_accs):.3f}")
    
    # Save
    save_data = {
        'lambdas': lambdas,
        'n_seeds': n_seeds,
        'results': {str(k): v for k, v in all_results.items()},
        'summary': {str(k): v for k, v in summary.items()}
    }
    
    save_path = Path("data/exp_4_2c_lambda_sweep_results.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {save_path}")
    
    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description="Lambda Phase Sensitivity Sweep")
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 0.05, 0.3])
    parser.add_argument('--n_seeds', type=int, default=8)
    parser.add_argument('--training_steps', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    run_lambda_sweep(
        lambdas=args.lambdas,
        n_seeds=args.n_seeds,
        training_steps=args.training_steps,
        start_seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main()
