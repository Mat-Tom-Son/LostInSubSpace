"""
Experiment: Geometry Coherence Sweep

Maps stability under training as a function of partial geometry destruction.

Conditions: 0, 1, 2, 3, 4 heads reinitialized in layer 1

Hypothesis: Mixed geometry is worse than erased geometry.
- Partial breaks create incompatible internal assumptions
- Full breaks restore coherence, enabling recovery
- Expect NON-MONOTONIC curve (worst in the middle = phase transition)
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
from datetime import datetime
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

from lib.logging_utils import setup_reproducibility


# =============================================================================
# MODEL (same as recovery_dynamics)
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ln = self.ln1(x)
        attn_out, _ = self.attn(h_ln, h_ln, h_ln)
        x = x + attn_out
        h_ln = self.ln2(x)
        x = x + self.ffn(h_ln)
        return x


class ModularArithmeticTransformer(nn.Module):
    def __init__(self, p: int = 113, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 512, dropout: float = 0.0):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = 2 * p + 1
        self.plus_token = p
        self.embed = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed = nn.Embedding(3, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        tok_emb = self.embed(x)
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)
        h = tok_emb + pos_emb
        for block in self.blocks:
            h = block(h)
        h_final = self.ln_final(h[:, -1, :])
        return self.head(h_final)


# =============================================================================
# DATASET
# =============================================================================

class ModularAdditionDataset(Dataset):
    def __init__(self, p: int = 113, train_fraction: float = 0.5,
                 is_train: bool = True, seed: int = 42):
        self.p = p
        self.plus_token = p
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        rng = np.random.RandomState(seed)
        rng.shuffle(all_pairs)
        n_train = int(len(all_pairs) * train_fraction)
        self.pairs = all_pairs[:n_train] if is_train else all_pairs[n_train:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        target = (a + b) % self.p
        input_seq = torch.LongTensor([a, self.plus_token, b + self.p + 1])
        return input_seq, torch.tensor(target, dtype=torch.long)


# =============================================================================
# QK REINITIALIZATION
# =============================================================================

def reinit_qk_n_heads(model: nn.Module, layer_idx: int, n_heads_to_reinit: int):
    """
    Reinitialize Q and K matrices for the first N attention heads.

    Args:
        model: The transformer model
        layer_idx: Which layer (0-indexed)
        n_heads_to_reinit: How many heads to reinitialize (0 to n_heads)
    """
    if n_heads_to_reinit == 0:
        return  # Control condition - no change

    block = model.blocks[layer_idx]
    attn = block.attn
    d_model = attn.embed_dim
    n_heads = attn.num_heads
    head_dim = d_model // n_heads

    with torch.no_grad():
        for head_idx in range(n_heads_to_reinit):
            start_row = head_idx * head_dim
            end_row = (head_idx + 1) * head_dim

            # Reinit Q for this head
            attn.in_proj_weight[start_row:end_row, :] = torch.nn.init.xavier_uniform_(
                torch.empty(head_dim, d_model, device=attn.in_proj_weight.device)
            )

            # Reinit K for this head
            k_start = d_model + start_row
            k_end = d_model + end_row
            attn.in_proj_weight[k_start:k_end, :] = torch.nn.init.xavier_uniform_(
                torch.empty(head_dim, d_model, device=attn.in_proj_weight.device)
            )

            # Reinit biases if they exist
            if attn.in_proj_bias is not None:
                nn.init.zeros_(attn.in_proj_bias[start_row:end_row])
                nn.init.zeros_(attn.in_proj_bias[k_start:k_end])


# =============================================================================
# TRAINING
# =============================================================================

def validate(model: nn.Module, dataloader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_loss += loss.item() * inputs.shape[0]
            total_samples += inputs.shape[0]
    return {'accuracy': total_correct / total_samples, 'loss': total_loss / total_samples}


def train_baseline(model, train_loader, val_loader, device, lr=1e-3, weight_decay=1.0,
                   max_steps=50000, target_accuracy=0.99, log_interval=500):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)
    trajectory = []

    pbar = tqdm(range(max_steps), desc="Baseline")
    for step in pbar:
        model.train()
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        if (step + 1) % log_interval == 0 or step == 0:
            metrics = validate(model, val_loader, device)
            metrics['step'] = step + 1
            trajectory.append(metrics)
            pbar.set_postfix({'val_acc': f"{metrics['accuracy']*100:.1f}%"})
            if metrics['accuracy'] >= target_accuracy:
                print(f"\n[OK] Converged at step {step+1}")
                return metrics, trajectory

    return validate(model, val_loader, device), trajectory


def train_recovery(model, train_loader, val_loader, device, lr=1e-3, weight_decay=1.0,
                   n_steps=20000, log_interval=200, desc="Recovery"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)
    trajectory = []

    # Log initial state
    metrics = validate(model, val_loader, device)
    metrics['step'] = 0
    trajectory.append(metrics)

    pbar = tqdm(range(n_steps), desc=desc)
    for step in pbar:
        model.train()
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        if (step + 1) % log_interval == 0:
            metrics = validate(model, val_loader, device)
            metrics['step'] = step + 1
            trajectory.append(metrics)
            pbar.set_postfix({'acc': f"{metrics['accuracy']*100:.1f}%"})

    return trajectory


# =============================================================================
# MAIN SWEEP
# =============================================================================

def run_sweep(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    d_ff: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    batch_size: int = 128,
    baseline_max_steps: int = 50000,
    recovery_steps: int = 20000,
    target_accuracy: float = 0.99,
    seed: int = 42,
    intervention_layer: int = 1,
    quick_test: bool = False
):
    if quick_test:
        baseline_max_steps = 5000
        recovery_steps = 3000
        target_accuracy = 0.95
        print("\n*** QUICK TEST MODE ***\n")

    setup_reproducibility(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print("EXPERIMENT: Geometry Coherence Sweep")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    print(f"Sweeping: 0, 1, 2, 3, 4 heads reinitialized in layer {intervention_layer}")
    print()

    # Data
    train_dataset = ModularAdditionDataset(p=p, is_train=True, seed=seed)
    val_dataset = ModularAdditionDataset(p=p, is_train=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Baseline training
    print("=" * 70)
    print("PHASE 1: Baseline Training")
    print("=" * 70)

    model = ModularArithmeticTransformer(p=p, d_model=d_model, n_heads=n_heads,
                                          n_layers=n_layers, d_ff=d_ff)
    baseline_metrics, baseline_traj = train_baseline(
        model, train_loader, val_loader, device,
        lr=lr, weight_decay=weight_decay,
        max_steps=baseline_max_steps, target_accuracy=target_accuracy
    )

    checkpoint = copy.deepcopy(model.state_dict())
    baseline_acc = baseline_metrics['accuracy']
    print(f"\nBaseline accuracy: {baseline_acc*100:.2f}%")

    # Sweep
    print("\n" + "=" * 70)
    print("PHASE 2: Head Sweep")
    print("=" * 70)

    sweep_results = {}

    for n_heads_reinit in range(n_heads + 1):  # 0, 1, 2, 3, 4
        print(f"\n--- Condition: {n_heads_reinit}/{n_heads} heads reinitialized ---")

        # Restore from checkpoint
        model.load_state_dict(copy.deepcopy(checkpoint))
        model.to(device)

        # Apply intervention
        reinit_qk_n_heads(model, layer_idx=intervention_layer, n_heads_to_reinit=n_heads_reinit)

        # Measure post-intervention
        post_metrics = validate(model, val_loader, device)
        print(f"  Post-intervention: {post_metrics['accuracy']*100:.2f}%")

        # Recovery training
        trajectory = train_recovery(
            model, train_loader, val_loader, device,
            lr=lr, weight_decay=weight_decay,
            n_steps=recovery_steps,
            log_interval=200 if not quick_test else 100,
            desc=f"{n_heads_reinit}H"
        )

        final_acc = trajectory[-1]['accuracy']
        print(f"  Final: {final_acc*100:.2f}%")

        sweep_results[n_heads_reinit] = {
            'post_intervention_acc': post_metrics['accuracy'],
            'final_acc': final_acc,
            'trajectory': trajectory,
            'recovered': final_acc > 0.9 * baseline_acc,
            'collapsed': final_acc < 0.5 * baseline_acc
        }

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print(f"\nBaseline: {baseline_acc*100:.2f}%")
    print(f"\n{'Heads':<8} {'Post-Int':<12} {'Final':<12} {'Status':<12}")
    print("-" * 44)

    for n_h in range(n_heads + 1):
        r = sweep_results[n_h]
        status = "RECOVERED" if r['recovered'] else ("COLLAPSED" if r['collapsed'] else "PARTIAL")
        print(f"{n_h:<8} {r['post_intervention_acc']*100:<12.1f} {r['final_acc']*100:<12.1f} {status:<12}")

    # Check for non-monotonic pattern
    final_accs = [sweep_results[i]['final_acc'] for i in range(n_heads + 1)]

    # Find if middle is worse than endpoints
    endpoints_min = min(final_accs[0], final_accs[-1])
    middle_min = min(final_accs[1:-1]) if len(final_accs) > 2 else final_accs[0]

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if middle_min < endpoints_min * 0.8:
        print("\n[!!] NON-MONOTONIC PATTERN DETECTED")
        print("  Middle conditions (partial reinit) show WORSE final accuracy")
        print("  than both endpoints (no reinit, full reinit).")
        print("\n  This supports the 'coherent vs incoherent' hypothesis:")
        print("  - Mixed geometry creates incompatible assumptions")
        print("  - Full erasure restores coherence, enabling recovery")
        worst_idx = final_accs.index(min(final_accs))
        print(f"\n  Worst condition: {worst_idx} heads ({final_accs[worst_idx]*100:.1f}%)")
    else:
        print("\n  Pattern is roughly monotonic.")
        print("  May need more seeds or longer training to see phase transition.")

    # Save results
    results = {
        'config': {
            'p': p, 'd_model': d_model, 'n_heads': n_heads, 'n_layers': n_layers,
            'd_ff': d_ff, 'lr': lr, 'weight_decay': weight_decay,
            'recovery_steps': recovery_steps, 'seed': seed,
            'intervention_layer': intervention_layer
        },
        'baseline_acc': baseline_acc,
        'sweep': {str(k): v for k, v in sweep_results.items()},
        'final_accuracies': final_accs,
        'run_id': run_id
    }

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / f"exp_head_sweep_{run_id}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # Plot
    plot_path = output_dir / f"exp_head_sweep_{run_id}.png"
    plot_sweep(sweep_results, baseline_acc, n_heads, str(plot_path))

    return results


def plot_sweep(sweep_results, baseline_acc, n_heads, save_path):
    """Create visualization of the sweep results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Final accuracy vs heads reinitialized
    ax1 = axes[0]
    heads = list(range(n_heads + 1))
    final_accs = [sweep_results[h]['final_acc'] * 100 for h in heads]
    post_accs = [sweep_results[h]['post_intervention_acc'] * 100 for h in heads]

    ax1.plot(heads, post_accs, 'b--o', label='Post-intervention', markersize=8)
    ax1.plot(heads, final_accs, 'r-s', label='After recovery', markersize=10, linewidth=2)
    ax1.axhline(y=baseline_acc * 100, color='green', linestyle=':', label='Baseline')
    ax1.axhline(y=100/113, color='gray', linestyle='--', alpha=0.5, label='Chance')

    ax1.set_xlabel('Heads Reinitialized', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Stability vs Partial Geometry Destruction', fontsize=14)
    ax1.set_xticks(heads)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # Plot 2: Recovery trajectories
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_heads + 1))

    for h in heads:
        traj = sweep_results[h]['trajectory']
        steps = [m['step'] for m in traj]
        accs = [m['accuracy'] * 100 for m in traj]
        ax2.plot(steps, accs, color=colors[h], linewidth=2, label=f'{h} heads')

    ax2.axhline(y=baseline_acc * 100, color='green', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Recovery Trajectories', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Geometry Coherence Sweep")
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--baseline_max_steps', type=int, default=50000)
    parser.add_argument('--recovery_steps', type=int, default=20000)
    parser.add_argument('--target_accuracy', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--intervention_layer', type=int, default=1)
    parser.add_argument('--quick_test', action='store_true')

    args = parser.parse_args()
    run_sweep(**vars(args))


if __name__ == '__main__':
    main()
