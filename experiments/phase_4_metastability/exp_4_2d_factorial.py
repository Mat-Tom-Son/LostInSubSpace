"""
EXPERIMENT 4.2d: 2×2 Factorial — Reference Coupling Confound

Tests the separated hypotheses:
- H1: "Orthogonality regularization improves stability" — falsified by λ sweep
- H2: "Having a separate reference trajectory stabilizes training" — testing now

2×2 Design:
| | B0: No Penalty | B1: Ortho Penalty |
|---|----------------|-------------------|
| A0: No Reference | CE only | CE + ortho-to-EMA-self |
| A1: Reference | CE + anchor (no penalty) | CE + ortho-to-anchor |

Key design choices:
- Same Young G freeze protocol per seed
- Anchor is a FROZEN checkpoint taken at Young G, not co-trained
- EMA-self uses an exponential moving average of the model's own representations
- "Anchor present no penalty" trains while logging similarity to anchor but not penalizing

This isolates whether stability comes from:
1. The ortho penalty itself
2. The presence of a reference target
3. The coupling mechanics
4. Or nothing (intrinsic metastability)
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
    time_to_gen = -1
    for step, acc in trajectory:
        if acc >= gen_threshold:
            time_to_gen = step
            break
    
    collapse_count = 0
    was_above = False
    for step, acc in trajectory:
        if acc >= gen_threshold:
            was_above = True
        elif was_above and acc < collapse_threshold:
            collapse_count += 1
            was_above = False
    
    time_to_stability = -1
    n = len(trajectory)
    for i in range(n):
        step, acc = trajectory[i]
        if acc >= stability_threshold:
            stable_from_here = all(trajectory[j][1] >= stability_threshold for j in range(i + 1, n))
            if stable_from_here:
                time_to_stability = step
                break
    
    # Dwell time above threshold
    dwell_above = sum(1 for _, acc in trajectory if acc >= stability_threshold)
    
    return {
        'time_to_generalization': time_to_gen,
        'collapse_count': collapse_count,
        'time_to_stability': time_to_stability,
        'stability_achieved': time_to_stability != -1,
        'dwell_above_threshold': dwell_above
    }


def run_condition(
    condition: str,  # 'ce_only', 'ce_ema', 'ce_anchor_nopen', 'ce_anchor_ortho'
    young_g_state: dict,
    anchor_state: dict,  # Frozen anchor state (for A1 conditions)
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_steps: int,
    lr: float,
    weight_decay: float,
    ortho_lambda: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    p: int,
    device,
    log_interval: int = 200,
    ema_decay: float = 0.99
) -> Tuple[List[Tuple[int, float]], Dict]:
    """
    Run one condition of the 2×2 factorial.
    
    Conditions:
    - ce_only: Pure cross-entropy (A0B0)
    - ce_ema: CE + ortho to EMA of self (A0B1)
    - ce_anchor_nopen: CE with anchor reference, no penalty (A1B0)
    - ce_anchor_ortho: CE + ortho to anchor (A1B1)
    """
    
    # Main model
    model = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    model.load_state_dict(young_g_state)
    model.to(device)
    freeze_parameters(model, freeze_qk=True)
    
    # Anchor model (frozen, for A1 conditions)
    anchor_model = None
    if 'anchor' in condition:
        anchor_model = DeepModularTransformer(p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        anchor_model.load_state_dict(anchor_state)
        anchor_model.to(device)
        anchor_model.eval()
        for param in anchor_model.parameters():
            param.requires_grad = False
    
    # EMA model (for ce_ema condition)
    ema_resid = None
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    
    trajectory = []
    cosim_history = []
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
        ce_loss = F.cross_entropy(logits, y)
        
        resid = model.get_residual()
        resid_norm = F.normalize(resid, dim=-1)
        
        # Compute ortho loss based on condition
        ortho_loss = torch.tensor(0.0, device=device)
        cosim = 0.0
        
        if condition == 'ce_only':
            # No extra term
            loss = ce_loss
            
        elif condition == 'ce_ema':
            # Ortho to EMA of self (use mean-pooled representation to avoid batch size issues)
            resid_mean = resid.mean(dim=0)  # Pool over batch dimension
            
            if ema_resid is None:
                ema_resid = resid_mean.detach().clone()
            else:
                ema_resid = ema_decay * ema_resid + (1 - ema_decay) * resid_mean.detach()
            
            ema_norm = F.normalize(ema_resid.unsqueeze(0), dim=-1)
            resid_mean_norm = F.normalize(resid_mean.unsqueeze(0), dim=-1)
            cosim = (ema_norm * resid_mean_norm).sum(dim=-1).mean().item()
            
            if step >= lambda_warmup:
                effective_lambda = ortho_lambda * min(1.0, (step - lambda_warmup) / lambda_warmup)
                ortho_loss = effective_lambda * torch.abs((ema_norm * resid_mean_norm).sum(dim=-1).mean())
            
            loss = ce_loss + ortho_loss
            
        elif condition == 'ce_anchor_nopen':
            # Anchor present, but no penalty (log cosim only)
            with torch.no_grad():
                _ = anchor_model(x)
                anchor_resid = anchor_model.get_residual()
                anchor_norm = F.normalize(anchor_resid, dim=-1)
            cosim = (anchor_norm * resid_norm).sum(dim=-1).mean().item()
            loss = ce_loss
            
        elif condition == 'ce_anchor_ortho':
            # Ortho to anchor (original probe-vs-anchor style)
            with torch.no_grad():
                _ = anchor_model(x)
                anchor_resid = anchor_model.get_residual()
                anchor_norm = F.normalize(anchor_resid, dim=-1)
            
            cosim = (anchor_norm * resid_norm).sum(dim=-1).mean().item()
            
            if step >= lambda_warmup:
                effective_lambda = ortho_lambda * min(1.0, (step - lambda_warmup) / lambda_warmup)
                ortho_loss = effective_lambda * torch.abs((anchor_norm * resid_norm).sum(dim=-1).mean())
            
            loss = ce_loss + ortho_loss
        
        loss.backward()
        optimizer.step()
        
        if step % log_interval == 0 or step == training_steps - 1:
            val_acc = validate(model, val_loader, device)
            trajectory.append((step, val_acc))
            cosim_history.append((step, cosim))
    
    final_acc = validate(model, val_loader, device)
    metrics = compute_stability_metrics(trajectory)
    metrics['final_acc'] = final_acc
    metrics['cosim_history'] = cosim_history
    
    return trajectory, metrics


def run_factorial(
    n_seeds: int = 6,
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
    ortho_lambda: float = 0.3,
    train_frac: float = 0.5,
    start_seed: int = 42,
    device: str = 'cuda'
):
    """Run 2×2 factorial experiment."""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    conditions = ['ce_only', 'ce_ema', 'ce_anchor_nopen', 'ce_anchor_ortho']
    condition_labels = {
        'ce_only': 'A0B0: CE Only',
        'ce_ema': 'A0B1: CE + EMA-Self',
        'ce_anchor_nopen': 'A1B0: Anchor (no penalty)',
        'ce_anchor_ortho': 'A1B1: Anchor + Ortho'
    }
    
    print("\n" + "="*80)
    print("2×2 FACTORIAL: REFERENCE COUPLING CONFOUND")
    print(f"Conditions: {conditions}")
    print(f"Seeds: {n_seeds}, Training: {training_steps} steps")
    print("="*80 + "\n")
    
    all_results = {c: [] for c in conditions}
    
    for seed_idx in range(n_seeds):
        seed = start_seed + seed_idx
        print(f"\n>>> SEED {seed} ({seed_idx+1}/{n_seeds})")
        
        setup_reproducibility(seed)
        
        # Data
        train_dataset = ModularAdditionDataset(p=p, split='train', train_frac=train_frac)
        val_dataset = ModularAdditionDataset(p=p, split='val', train_frac=train_frac)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Warmup to capture Young G
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
        
        # For anchor conditions, create a frozen anchor at Young G
        # (In practice, this means the anchor IS the Young G snapshot)
        anchor_state = copy.deepcopy(young_g_state)
        
        # Run each condition
        for cond in conditions:
            print(f"  Running {condition_labels[cond]}...")
            trajectory, metrics = run_condition(
                condition=cond,
                young_g_state=young_g_state,
                anchor_state=anchor_state,
                train_loader=train_loader,
                val_loader=val_loader,
                training_steps=training_steps,
                lr=lr,
                weight_decay=weight_decay,
                ortho_lambda=ortho_lambda,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                p=p,
                device=device
            )
            
            all_results[cond].append({
                'seed': seed,
                'freeze_step': actual_freeze_step,
                'trajectory': trajectory,
                'metrics': metrics
            })
            
            status = "✓ Stable" if metrics['stability_achieved'] else "✗ Unstable"
            print(f"    {status} | Collapses: {metrics['collapse_count']} | "
                  f"Final: {metrics['final_acc']:.3f} | Dwell: {metrics['dwell_above_threshold']}")
    
    # Aggregate analysis
    print("\n" + "="*80)
    print("2×2 FACTORIAL RESULTS")
    print("="*80 + "\n")
    
    print(f"{'Condition':<25} {'Stability':<12} {'Collapses':<15} {'Final Acc':<15} {'Dwell':<10}")
    print("-"*77)
    
    summary = {}
    for cond in conditions:
        results = all_results[cond]
        stability_rate = sum(1 for r in results if r['metrics']['stability_achieved']) / len(results)
        collapse_counts = [r['metrics']['collapse_count'] for r in results]
        final_accs = [r['metrics']['final_acc'] for r in results]
        dwells = [r['metrics']['dwell_above_threshold'] for r in results]
        
        summary[cond] = {
            'stability_rate': stability_rate,
            'mean_collapses': np.mean(collapse_counts),
            'std_collapses': np.std(collapse_counts),
            'mean_acc': np.mean(final_accs),
            'std_acc': np.std(final_accs),
            'mean_dwell': np.mean(dwells),
            'std_dwell': np.std(dwells)
        }
        
        print(f"{condition_labels[cond]:<25} {stability_rate*100:>4.0f}%{'':<6} "
              f"{np.mean(collapse_counts):>5.1f} ± {np.std(collapse_counts):<5.1f} "
              f"{np.mean(final_accs):>5.3f} ± {np.std(final_accs):<5.3f} "
              f"{np.mean(dwells):>5.1f}")
    
    # 2×2 analysis
    print("\n" + "-"*40)
    print("MAIN EFFECTS (2×2 ANOVA-style)")
    print("-"*40)
    
    # Effect of Reference (A1 vs A0)
    a0_stable = (summary['ce_only']['stability_rate'] + summary['ce_ema']['stability_rate']) / 2
    a1_stable = (summary['ce_anchor_nopen']['stability_rate'] + summary['ce_anchor_ortho']['stability_rate']) / 2
    print(f"Reference effect: A0={a0_stable*100:.0f}% vs A1={a1_stable*100:.0f}% (Δ={100*(a1_stable-a0_stable):+.0f}%)")
    
    # Effect of Penalty (B1 vs B0)
    b0_stable = (summary['ce_only']['stability_rate'] + summary['ce_anchor_nopen']['stability_rate']) / 2
    b1_stable = (summary['ce_ema']['stability_rate'] + summary['ce_anchor_ortho']['stability_rate']) / 2
    print(f"Penalty effect:   B0={b0_stable*100:.0f}% vs B1={b1_stable*100:.0f}% (Δ={100*(b1_stable-b0_stable):+.0f}%)")
    
    # Save
    save_data = {
        'conditions': conditions,
        'n_seeds': n_seeds,
        'results': all_results,
        'summary': summary
    }
    
    save_path = Path("data/exp_4_2d_factorial_results.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {save_path}")
    
    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description="2×2 Factorial: Reference Coupling Confound")
    parser.add_argument('--n_seeds', type=int, default=6)
    parser.add_argument('--training_steps', type=int, default=20000)
    parser.add_argument('--ortho_lambda', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    run_factorial(
        n_seeds=args.n_seeds,
        training_steps=args.training_steps,
        ortho_lambda=args.ortho_lambda,
        start_seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main()
