"""
EXPERIMENT 2.2: Young Geometry Subspace Probe (2-Layer)

Phase 2: Prove "Young G" permits multiple S allocations at 2 layers.

Protocol:
1. Train 2-layer model on Modular Arithmetic (p=113) to step N (pre-grokking)
2. Freeze ALL QK parameters ("Young G")
3. Train Anchor model with standard CE → natural S allocation
4. Train Probe model with CE + λ×|CosSim(anchor, probe)| → forced orthogonal

Success Criteria:
- Both models converge to ≥95% validation accuracy
- Cosine similarity between final representations < 0.2
- Confirms subspace structure isn't a 1-layer artifact
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict
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
    """Validation accuracy."""
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


def run_single_seed(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    warmup_steps: int = 1500,
    anchor_steps: int = 15000,
    probe_steps: int = 15000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    ortho_lambda: float = 1.0,
    train_frac: float = 0.5,
    seed: int = 42,
    device: str = 'cuda'
) -> Dict:
    """
    Run Experiment 2.2: Young G Subspace Probe on 2-Layer Model.
    
    Protocol:
    1. Train to warmup_steps (memorization phase, young G)
    2. Freeze ALL QK parameters across all layers
    3. Train ANCHOR with standard CE → natural S allocation
    4. Train PROBE with CE + λ*|CosSim(anchor_resid, probe_resid)|
    """
    
    print("\n" + "="*80)
    print(f"EXPERIMENT 2.2: YOUNG G SUBSPACE PROBE (2-LAYER, seed={seed})")
    print("="*80 + "\n")
    
    setup_reproducibility(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Architecture: n_layers={n_layers}, d_model={d_model}")
    print(f"Modulus p={p}, Warmup={warmup_steps}, Lambda={ortho_lambda}\n")
    
    # Data
    train_dataset = ModularAdditionDataset(p=p, split='train', train_frac=train_frac)
    val_dataset = ModularAdditionDataset(p=p, split='val', train_frac=train_frac)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # =========================================================================
    # PHASE 1: Warmup (Memorization Phase → Young G)
    # =========================================================================
    
    print("\n" + "-"*80)
    print(f"PHASE 1: WARMUP (Capturing Young G at Step {warmup_steps})")
    print("-"*80 + "\n")
    
    base_model = DeepModularTransformer(
        p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers
    )
    base_model.to(device)
    
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)
    
    for step in range(warmup_steps):
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
        
        if step % 300 == 0:
            train_acc = (logits.argmax(dim=-1) == y).float().mean().item()
            val_acc = validate(base_model, val_loader, device)
            print(f"Warmup Step {step:5d} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")
    
    warmup_train_acc = (logits.argmax(dim=-1) == y).float().mean().item()
    warmup_val_acc = validate(base_model, val_loader, device)
    
    print(f"\nWarmup Complete: Train={warmup_train_acc:.3f}, Val={warmup_val_acc:.3f}")
    print("This is our 'Young G' - memorizing but not yet generalizing")
    
    # Save checkpoint
    young_g_state = copy.deepcopy(base_model.state_dict())
    young_qk = get_qk_parameters(base_model)
    
    # =========================================================================
    # PHASE 2: Train ANCHOR (Natural S Allocation)
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 2: ANCHOR MODEL (Natural S Allocation)")
    print("-"*80 + "\n")
    
    anchor_model = DeepModularTransformer(
        p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers
    )
    anchor_model.load_state_dict(young_g_state)
    anchor_model.to(device)
    
    # Freeze QK across ALL layers
    frozen_counts = freeze_parameters(anchor_model, freeze_qk=True)
    print(f"Frozen QK parameters: {frozen_counts['qk']}")
    
    anchor_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, anchor_model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    
    anchor_history = []
    train_iter = iter(train_loader)
    
    for step in range(anchor_steps):
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
        
        if step % 1000 == 0 or step == anchor_steps - 1:
            val_acc = validate(anchor_model, val_loader, device)
            anchor_history.append({'step': step, 'val_acc': val_acc, 'loss': loss.item()})
            print(f"Anchor Step {step:5d} | Loss: {loss.item():.4f} | Val: {val_acc:.3f}")
    
    anchor_final_acc = validate(anchor_model, val_loader, device)
    print(f"\nAnchor Final Accuracy: {anchor_final_acc:.3f}")
    
    # Verify freeze
    if not verify_freeze(anchor_model, young_qk):
        print("WARNING: QK changed during anchor training!")
    
    # =========================================================================
    # PHASE 3: Train PROBE with Orthogonal Regularization
    # =========================================================================
    
    print("\n" + "-"*80)
    print("PHASE 3: PROBE MODEL (Forced Orthogonal to Anchor)")
    print("-"*80 + "\n")
    
    probe_model = DeepModularTransformer(
        p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers
    )
    probe_model.load_state_dict(young_g_state)
    probe_model.to(device)
    
    # Freeze QK
    freeze_parameters(probe_model, freeze_qk=True)
    
    probe_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, probe_model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    
    probe_history = []
    train_iter = iter(train_loader)
    
    for step in range(probe_steps):
        probe_model.train()
        anchor_model.eval()  # Anchor is frozen reference
        
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        # Forward both models
        with torch.no_grad():
            _ = anchor_model(x)
            anchor_resid = anchor_model.get_residual().detach()
        
        probe_logits = probe_model(x)
        probe_resid = probe_model.get_residual()
        
        # CE Loss
        ce_loss = F.cross_entropy(probe_logits, y)
        
        # Orthogonal Loss: penalize similarity to anchor
        anchor_norm = F.normalize(anchor_resid, dim=-1)
        probe_norm = F.normalize(probe_resid, dim=-1)
        cosim = (anchor_norm * probe_norm).sum(dim=-1).mean()
        ortho_loss = torch.abs(cosim)
        
        # Total loss
        total_loss = ce_loss + ortho_lambda * ortho_loss
        
        probe_optimizer.zero_grad()
        total_loss.backward()
        probe_optimizer.step()
        
        if step % 1000 == 0 or step == probe_steps - 1:
            val_acc = validate(probe_model, val_loader, device)
            probe_history.append({
                'step': step,
                'val_acc': val_acc,
                'ce_loss': ce_loss.item(),
                'ortho_loss': ortho_loss.item(),
                'cosim': cosim.item()
            })
            print(f"Probe Step {step:5d} | CE: {ce_loss.item():.4f} | "
                  f"Ortho: {ortho_loss.item():.3f} | CosSim: {cosim.item():.3f} | Val: {val_acc:.3f}")
    
    probe_final_acc = validate(probe_model, val_loader, device)
    
    # Verify freeze
    if not verify_freeze(probe_model, young_qk):
        print("WARNING: QK changed during probe training!")
    
    # =========================================================================
    # FINAL ANALYSIS
    # =========================================================================
    
    print("\n" + "="*80)
    print("FINAL ANALYSIS")
    print("="*80 + "\n")
    
    # Compute final cosine similarity
    anchor_model.eval()
    probe_model.eval()
    
    final_cosims = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            _ = anchor_model(x)
            _ = probe_model(x)
            
            anchor_resid = anchor_model.get_residual()
            probe_resid = probe_model.get_residual()
            
            anchor_norm = F.normalize(anchor_resid, dim=-1)
            probe_norm = F.normalize(probe_resid, dim=-1)
            cosim = (anchor_norm * probe_norm).sum(dim=-1).mean().item()
            final_cosims.append(cosim)
    
    final_cosim = np.mean(final_cosims)
    
    print(f"Anchor Accuracy: {anchor_final_acc:.3f}")
    print(f"Probe Accuracy:  {probe_final_acc:.3f}")
    print(f"Final CosSim:    {final_cosim:.3f}")
    
    # Success criteria (stricter than 1-layer: CosSim < 0.2)
    print("\n" + "-"*40)
    print("SUCCESS CRITERIA (2-Layer)")
    print("-"*40 + "\n")
    
    crit_1 = anchor_final_acc > 0.95
    crit_2 = probe_final_acc > 0.95
    crit_3 = abs(final_cosim) < 0.2  # Stricter than 1-layer
    
    print(f"1. Anchor converged (>95%): {'✓ PASS' if crit_1 else '✗ FAIL'} ({anchor_final_acc:.3f})")
    print(f"2. Probe converged (>95%):  {'✓ PASS' if crit_2 else '✗ FAIL'} ({probe_final_acc:.3f})")
    print(f"3. CosSim < 0.2:            {'✓ PASS' if crit_3 else '✗ FAIL'} ({final_cosim:.3f})")
    
    overall = crit_1 and crit_2 and crit_3
    print(f"\nOverall: {'✓ G PERMITS DIVERSE S ALLOCATIONS (2-LAYER)' if overall else '✗ INCONCLUSIVE'}")
    
    results = {
        'config': {
            'n_layers': n_layers,
            'd_model': d_model,
            'n_heads': n_heads,
            'warmup_steps': warmup_steps,
            'ortho_lambda': ortho_lambda,
            'seed': seed
        },
        'warmup': {
            'steps': warmup_steps,
            'train_acc': float(warmup_train_acc),
            'val_acc': float(warmup_val_acc)
        },
        'anchor': {
            'history': anchor_history,
            'final_acc': float(anchor_final_acc)
        },
        'probe': {
            'history': probe_history,
            'final_acc': float(probe_final_acc),
            'ortho_lambda': float(ortho_lambda)
        },
        'analysis': {
            'final_cosim': float(final_cosim),
            'anchor_converged': bool(crit_1),
            'probe_converged': bool(crit_2),
            'orthogonal_achieved': bool(crit_3),
            'overall_pass': bool(overall)
        }
    }
    
    return results


def run_experiment(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    warmup_steps: int = 1500,
    anchor_steps: int = 15000,
    probe_steps: int = 15000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    ortho_lambda: float = 1.0,
    train_frac: float = 0.5,
    start_seed: int = 42,
    n_seeds: int = 5,
    device: str = 'cuda'
):
    seeds = range(start_seed, start_seed + n_seeds)
    all_results = []
    
    print("\n" + "="*80)
    print(f"RUNNING 2-LAYER YOUNG G EXPERIMENT (n={n_seeds} seeds)")
    print("="*80 + "\n")
    
    for i, seed in enumerate(seeds):
        print(f"\n>>> SEED {seed} ({i+1}/{len(seeds)})")
        res = run_single_seed(
            p=p, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            warmup_steps=warmup_steps, anchor_steps=anchor_steps, probe_steps=probe_steps,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            ortho_lambda=ortho_lambda, train_frac=train_frac, seed=seed, device=device
        )
        all_results.append(res)
    
    # Helper for CI
    def get_stats(vals):
        arr = np.array(vals)
        mean = np.mean(arr)
        if len(arr) > 1:
            se = np.std(arr, ddof=1) / np.sqrt(len(arr))
        else:
            se = 0.0
        return mean, 1.96 * se
    
    print("\n" + "="*80)
    print("EXPERIMENT 2.2 RESULTS (Multi-seed)")
    print("="*80 + "\n")
    
    anchor_accs = [r['anchor']['final_acc'] for r in all_results]
    probe_accs = [r['probe']['final_acc'] for r in all_results]
    cosims = [r['analysis']['final_cosim'] for r in all_results]
    passes = [r['analysis']['overall_pass'] for r in all_results]
    
    m_anc, ci_anc = get_stats(anchor_accs)
    m_prb, ci_prb = get_stats(probe_accs)
    m_cos, ci_cos = get_stats(cosims)
    
    print(f"Anchor Accuracy: {m_anc:.3%} ± {ci_anc:.3%}")
    print(f"Probe Accuracy:  {m_prb:.3%} ± {ci_prb:.3%}")
    print(f"Final CosSim:    {m_cos:.3f} ± {ci_cos:.3f}")
    print(f"Pass Rate:       {sum(passes)}/{len(passes)} ({(sum(passes)/len(passes)):.0%})")
    
    # Save
    save_path = Path("clean_audit/data/exp_2_2_young_g_deep_results.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {save_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--warmup_steps', type=int, default=1500)
    parser.add_argument('--anchor_steps', type=int, default=15000)
    parser.add_argument('--probe_steps', type=int, default=15000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0)
    parser.add_argument('--ortho_lambda', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_frac', type=float, default=0.5, help='Fraction of data for training (default 0.5)')
    parser.add_argument('--quick_test', action='store_true')
    
    args = parser.parse_args()
    
    warmup = 300 if args.quick_test else args.warmup_steps
    anchor = 2000 if args.quick_test else args.anchor_steps
    probe = 2000 if args.quick_test else args.probe_steps
    n_seeds = 1 if args.quick_test else args.n_seeds
    
    if args.quick_test:
        print("\n[QUICK TEST MODE] Reduced steps and seeds\n")
    
    run_experiment(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        warmup_steps=warmup,
        anchor_steps=anchor,
        probe_steps=probe,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ortho_lambda=args.ortho_lambda,
        train_frac=args.train_frac,
        start_seed=args.seed,
        n_seeds=n_seeds,
        device=args.device
    )


if __name__ == '__main__':
    main()
